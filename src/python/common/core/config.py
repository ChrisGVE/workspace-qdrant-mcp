"""Dictionary-based configuration management for workspace-qdrant-mcp.

This module provides a robust, tolerant configuration system that uses dictionary-based
merging instead of struct-based deserialization. It gracefully handles missing fields,
provides defaults for all configuration paths, and supports type-appropriate value access.

Architecture:
    1. Parse YAML into temporary dictionary with unit conversions
    2. Create internal dictionary with ALL possible configuration labels and defaults
    3. Merge dictionaries (YAML values take precedence over defaults)
    4. Drop both starting dictionaries, keep only merged result
    5. Provide global read-only structure available to full codebase
    6. Support accessor pattern: level1.level2.level3 with type-appropriate returns

Configuration Sources:
    1. YAML configuration files (highest priority)
    2. Environment variables (medium priority)
    3. .env files in current directory
    4. Default values (lowest priority)

Supported Formats:
    - YAML configuration files: workspace_qdrant_config.yaml (shared with daemon)
    - Prefixed environment variables: WORKSPACE_QDRANT_*
    - Nested configuration: WORKSPACE_QDRANT_QDRANT__URL
    - Legacy variables: QDRANT_URL, FASTEMBED_MODEL (backward compatibility)
    - Configuration files: .env with UTF-8 encoding
    - Unit conversions: 32MB → 33554432, 45s → 45000ms

Configuration Hierarchy:
    - Server settings (host, port, debug mode)
    - Qdrant database connection (URL, API key, timeouts)
    - Embedding service (model, chunking, batch processing)
    - Workspace management (collections, GitHub integration)
    - gRPC settings (host, port, timeouts)
    - Auto-ingestion settings (enabled, batch sizes, file filters)

Example:
    ```python
    from workspace_qdrant_mcp.core.config import get_config

    # Get global configuration instance
    config = get_config()

    # Access nested configuration with dot notation
    qdrant_url = config.get("qdrant.url")
    embedding_model = config.get("embedding.model")
    chunk_size = config.get("embedding.chunk_size")

    # Get dictionaries and lists as appropriate
    qdrant_config = config.get("qdrant")  # Returns dict
    collection_types = config.get("workspace.collection_types")  # Returns list

    # Thread-safe global access
    server_host = config.get("server.host")
    ```
"""

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# Task 215: Use unified logging system for MCP stdio compliance
from loguru import logger


# Early environment setup for MCP stdio mode
def setup_stdio_environment():
    """Set up early environment configuration for MCP stdio mode compatibility.

    This function should be called as early as possible to configure environment
    variables that affect third-party libraries before they are imported.
    """
    # Detect if we're in MCP stdio mode
    if (os.getenv("WQM_STDIO_MODE", "").lower() == "true" or
        os.getenv("MCP_TRANSPORT") == "stdio" or
        ("--transport" in os.sys.argv if hasattr(os, 'sys') else False)):

        # Set comprehensive environment variables for third-party library suppression
        stdio_env_vars = {
            "WQM_STDIO_MODE": "true",
            "MCP_QUIET_MODE": "true",
            "TOKENIZERS_PARALLELISM": "false",
            "GRPC_VERBOSITY": "NONE",
            "GRPC_TRACE": "",
            "PYTHONWARNINGS": "ignore",
            "TF_CPP_MIN_LOG_LEVEL": "3",  # TensorFlow
            "TRANSFORMERS_VERBOSITY": "error",  # Transformers library
            "HF_DATASETS_VERBOSITY": "error",  # HuggingFace datasets
            "BITSANDBYTES_NOWELCOME": "1",  # BitsAndBytes welcome message
        }

        for key, value in stdio_env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

# Call early setup on module import
setup_stdio_environment()


# Unit conversion utilities
def parse_size_to_bytes(size_str: str) -> int:
    """Convert size strings like '32MB', '1GB' to bytes.

    Args:
        size_str: Size string with unit (e.g., '32MB', '1GB', '512KB')

    Returns:
        int: Size in bytes

    Raises:
        ValueError: If the size string format is invalid
    """
    if isinstance(size_str, int):
        return size_str

    if not isinstance(size_str, str):
        raise ValueError(f"Size must be string or int, got {type(size_str)}")

    # Remove whitespace and convert to uppercase
    size_str = size_str.strip().upper()

    # Pattern to match number + unit
    pattern = r'^(\d+(?:\.\d+)?)\s*([KMGT]?B?)$'
    match = re.match(pattern, size_str)

    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    value, unit = match.groups()
    value = float(value)

    # Unit multipliers
    multipliers = {
        'B': 1,
        'KB': 1024,
        'MB': 1024 ** 2,
        'GB': 1024 ** 3,
        'TB': 1024 ** 4,
        '': 1,  # No unit defaults to bytes
    }

    if unit not in multipliers:
        raise ValueError(f"Unknown size unit: {unit}")

    return int(value * multipliers[unit])


def parse_time_to_milliseconds(time_str: str) -> int:
    """Convert time strings like '45s', '2m', '500ms' to milliseconds.

    Args:
        time_str: Time string with unit (e.g., '45s', '2m', '500ms')

    Returns:
        int: Time in milliseconds

    Raises:
        ValueError: If the time string format is invalid
    """
    if isinstance(time_str, int):
        return time_str

    if not isinstance(time_str, str):
        raise ValueError(f"Time must be string or int, got {type(time_str)}")

    # Remove whitespace and convert to lowercase for units
    time_str = time_str.strip()

    # Pattern to match number + unit
    pattern = r'^(\d+(?:\.\d+)?)\s*([a-zA-Z]*)$'
    match = re.match(pattern, time_str)

    if not match:
        raise ValueError(f"Invalid time format: {time_str}")

    value, unit = match.groups()
    value = float(value)
    unit = unit.lower()

    # Unit multipliers to milliseconds
    multipliers = {
        'ms': 1,
        's': 1000,
        'm': 60 * 1000,
        'h': 60 * 60 * 1000,
        '': 1000,  # No unit defaults to seconds
    }

    if unit not in multipliers:
        raise ValueError(f"Unknown time unit: {unit}")

    return int(value * multipliers[unit])


class ConfigManager:
    """Dictionary-based configuration manager with thread-safe global access.

    This class implements the user-specified dictionary-based configuration architecture:
    1. Parse YAML into temporary dictionary with unit conversions
    2. Create internal dictionary with ALL possible configuration labels and defaults
    3. Merge dictionaries (YAML values take precedence over defaults)
    4. Drop both starting dictionaries, keep only merged result
    5. Provide global read-only structure with dot notation access

    Thread-safe singleton pattern ensures consistent configuration across the codebase.
    """

    _instance: Optional['ConfigManager'] = None
    _lock = threading.Lock()

    def __init__(self, config_file: Optional[str] = None, **kwargs) -> None:
        """Initialize configuration manager with dictionary-based architecture.

        Args:
            config_file: Path to YAML configuration file
            **kwargs: Override values for configuration parameters
        """
        self._config: Dict[str, Any] = {}
        self._load_configuration(config_file, **kwargs)

    @classmethod
    def get_instance(cls, config_file: Optional[str] = None, **kwargs) -> 'ConfigManager':
        """Get singleton instance of ConfigManager (thread-safe).

        Args:
            config_file: Path to YAML configuration file (only used on first call)
            **kwargs: Override values (only used on first call)

        Returns:
            ConfigManager: Singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_file, **kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None

    def _load_configuration(self, config_file: Optional[str] = None, **kwargs) -> None:
        """Load configuration using dictionary-based architecture.

        Implements the user-specified process:
        a) Parse YAML into temporary dictionary with unit conversions
        b) Create internal dictionary with ALL possible configuration labels and defaults
        c) Merge dictionaries (YAML values take precedence)
        d) Drop both starting dictionaries, keep only merged result
        """
        # Step a) Parse YAML into temporary dictionary with unit conversions
        yaml_config = {}
        if config_file:
            yaml_config = self._load_yaml_config(config_file)
        else:
            # Auto-discover configuration file only if not in test mode
            if not os.environ.get('WQM_TEST_MODE'):
                auto_config_file = self._find_default_config_file()
                if auto_config_file:
                    yaml_config = self._load_yaml_config(auto_config_file)

        # Apply unit conversions to YAML config
        yaml_config = self._apply_unit_conversions(yaml_config)

        # Step b) Create internal dictionary with ALL possible configuration labels and defaults
        default_config = self._create_comprehensive_defaults()

        # Load environment variables into temporary dict
        env_config = self._load_environment_variables()

        # Merge kwargs (highest precedence)
        merged_kwargs = self._apply_unit_conversions(kwargs)

        # Step c) Merge dictionaries (higher precedence overwrites lower)
        # Order: defaults < env_vars < yaml < kwargs
        self._config = self._deep_merge_configs([
            default_config,
            env_config,
            yaml_config,
            merged_kwargs
        ])

        # Step d) Both starting dictionaries are now dropped (garbage collected)
        # Only self._config (merged result) is kept

    def _create_comprehensive_defaults(self) -> Dict[str, Any]:
        """Create comprehensive default configuration with ALL possible labels.

        Returns:
            Dict containing all possible configuration paths with default values
        """
        return {
            # Server configuration
            "server": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug": False,
            },

            # Qdrant database configuration
            "qdrant": {
                "url": "http://localhost:6333",
                "api_key": None,
                "timeout": 30000,  # milliseconds
                "prefer_grpc": True,
                "transport": "grpc",
            },

            # Embedding service configuration
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "enable_sparse_vectors": True,
                "chunk_size": 800,
                "chunk_overlap": 120,
                "batch_size": 50,
            },

            # Workspace management configuration
            "workspace": {
                "collection_types": [],
                "global_collections": [],
                "github_user": None,
                "auto_create_collections": False,
                "memory_collection_name": "__memory",
                "code_collection_name": "__code",
                "custom_include_patterns": [],
                "custom_exclude_patterns": [],
                "custom_project_indicators": {},
            },

            # gRPC communication configuration
            "grpc": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 50051,
                "fallback_to_direct": True,
                "connection_timeout": 10000,  # milliseconds
                "max_retries": 3,
                "retry_backoff_multiplier": 1.5,
                "health_check_interval": 30000,  # milliseconds
                "max_message_length": 104857600,  # 100MB in bytes
                "keepalive_time": 30000,  # milliseconds
            },

            # Auto-ingestion configuration
            "auto_ingestion": {
                "enabled": True,
                "auto_create_watches": True,
                "include_common_files": True,
                "include_source_files": False,
                "target_collection_suffix": "scratchbook",
                "max_files_per_batch": 5,
                "batch_delay_seconds": 2.0,
                "max_file_size_mb": 52428800,  # 50MB in bytes
                "debounce_seconds": 10000,  # milliseconds
            },

            # Logging configuration
            "logging": {
                "level": "info",
                "use_file_logging": False,
                "log_file": None,
                "enable_metrics": False,
                "metrics_interval_secs": 60000,  # milliseconds
            },

            # Performance configuration
            "performance": {
                "max_concurrent_tasks": 4,
                "default_timeout_ms": 30000,  # milliseconds
                "enable_preemption": True,
                "chunk_size": 1000,
            },
        }

    def _apply_unit_conversions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply unit conversions to configuration values.

        Converts size strings (32MB → 33554432) and time strings (45s → 45000ms).

        Args:
            config: Configuration dictionary

        Returns:
            Dict with unit conversions applied
        """
        if not isinstance(config, dict):
            return config

        converted = {}
        for key, value in config.items():
            if isinstance(value, dict):
                converted[key] = self._apply_unit_conversions(value)
            elif isinstance(value, str):
                # Apply size conversions for size-related fields
                if any(size_key in key.lower() for size_key in ['size', 'length', 'limit']):
                    try:
                        converted[key] = parse_size_to_bytes(value)
                        continue
                    except ValueError:
                        pass  # Not a size string, keep original

                # Apply time conversions for time-related fields
                if any(time_key in key.lower() for time_key in ['timeout', 'interval', 'delay', '_time', '_seconds']):
                    try:
                        converted[key] = parse_time_to_milliseconds(value)
                        continue
                    except ValueError:
                        pass  # Not a time string, keep original

                converted[key] = value
            else:
                converted[key] = value

        return converted

    def _deep_merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Deep merge multiple configuration dictionaries.

        Later configs in the list take precedence over earlier ones.

        Args:
            configs: List of configuration dictionaries to merge

        Returns:
            Dict: Merged configuration
        """
        result = {}

        for config in configs:
            if not isinstance(config, dict):
                continue

            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge nested dictionaries
                    result[key] = self._deep_merge_configs([result[key], value])
                else:
                    # Override with new value
                    result[key] = value

        return result

    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path.

        Supports the user-specified accessor pattern: level1.level2.level3
        Returns type-appropriate values (dict, list, str, int, float, bool)

        Args:
            path: Dot-separated configuration path (e.g., "qdrant.url", "embedding.model")
            default: Default value if path is not found

        Returns:
            Any: Configuration value with appropriate type
        """
        if not path:
            return self._config

        keys = path.split('.')
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def has(self, path: str) -> bool:
        """Check if configuration path exists.

        Args:
            path: Dot-separated configuration path

        Returns:
            bool: True if path exists, False otherwise
        """
        if not path:
            return True

        keys = path.split('.')
        current = self._config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False

        return True

    def get_all(self) -> Dict[str, Any]:
        """Get complete configuration dictionary (read-only copy).

        Returns:
            Dict: Complete configuration
        """
        import copy
        return copy.deepcopy(self._config)

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues.

        Returns:
            List[str]: List of validation error messages
        """
        issues = []

        # Validate Qdrant configuration
        qdrant_url = self.get("qdrant.url")
        if not qdrant_url:
            issues.append("Qdrant URL is required")
        elif not isinstance(qdrant_url, str):
            issues.append("Qdrant URL must be a string")
        elif not (qdrant_url.startswith("http://") or qdrant_url.startswith("https://")):
            issues.append("Qdrant URL must start with http:// or https://")

        # Validate embedding configuration
        chunk_size = self.get("embedding.chunk_size")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            issues.append("Embedding chunk_size must be a positive integer")
        elif chunk_size > 10000:
            issues.append("Embedding chunk_size should not exceed 10000 for optimal performance")

        chunk_overlap = self.get("embedding.chunk_overlap")
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            issues.append("Embedding chunk_overlap must be a non-negative integer")
        elif chunk_overlap >= chunk_size:
            issues.append("Embedding chunk_overlap must be less than chunk_size")

        batch_size = self.get("embedding.batch_size")
        if not isinstance(batch_size, int) or batch_size <= 0:
            issues.append("Embedding batch_size must be a positive integer")
        elif batch_size > 1000:
            issues.append("Embedding batch_size should not exceed 1000 for memory efficiency")

        # Validate server configuration
        server_port = self.get("server.port")
        if not isinstance(server_port, int) or not (1 <= server_port <= 65535):
            issues.append("Server port must be an integer between 1 and 65535")

        # Validate workspace configuration
        collection_types = self.get("workspace.collection_types", [])
        if not isinstance(collection_types, list):
            issues.append("Workspace collection_types must be a list")
        elif len(collection_types) > 20:
            issues.append("Too many collection types configured (max 20 recommended)")

        global_collections = self.get("workspace.global_collections", [])
        if not isinstance(global_collections, list):
            issues.append("Workspace global_collections must be a list")
        elif len(global_collections) > 50:
            issues.append("Too many global collections configured (max 50 recommended)")

        return issues

    def _load_yaml_config(self, config_file: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            Dict containing the parsed YAML configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
            ValueError: If the YAML structure is invalid
        """
        config_path = Path(config_file)

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}

        if not config_path.is_file():
            logger.warning(f"Configuration path is not a file: {config_file}")
            return {}

        try:
            with config_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            if yaml_data is None:
                return {}

            if not isinstance(yaml_data, dict):
                logger.warning(f"YAML configuration must be a dictionary, got {type(yaml_data).__name__}")
                return {}

            return yaml_data

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {config_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            return {}

    def _find_default_config_file(self) -> Optional[str]:
        """Find default configuration file.

        Search order:
        1. XDG-compliant directories + /config.yaml
        2. XDG-compliant directories + /workspace_qdrant_config.yaml
        3. Project-specific configs in current directory

        Returns:
            Path to the first found config file, or None if no config file found
        """
        # Get XDG-compliant config directories
        xdg_config_dirs = self._get_xdg_config_dirs()

        # Check XDG-compliant directories
        for config_dir in xdg_config_dirs:
            for config_name in ["config.yaml", "workspace_qdrant_config.yaml"]:
                config_path = config_dir / config_name
                if config_path.exists() and config_path.is_file():
                    logger.info(f"Auto-discovered configuration file: {config_path}")
                    return str(config_path)

        # Check current directory for project-specific configs
        current_dir = Path.cwd()
        project_config_names = [
            "workspace_qdrant_config.yaml",
            "workspace_qdrant_config.yml",
            ".workspace-qdrant.yaml",
            ".workspace-qdrant.yml"
        ]

        for config_name in project_config_names:
            config_path = current_dir / config_name
            if config_path.exists() and config_path.is_file():
                logger.info(f"Auto-discovered project configuration file: {config_path}")
                return str(config_path)

        return None

    def _get_xdg_config_dirs(self) -> List[Path]:
        """Get XDG-compliant configuration directories.

        Returns:
            List of Path objects for config directories to search
        """
        import platform

        config_dirs = []

        # Check XDG_CONFIG_HOME first
        xdg_config_home = os.environ.get('XDG_CONFIG_HOME')
        if xdg_config_home:
            config_dirs.append(Path(xdg_config_home) / 'workspace-qdrant')
        else:
            # Fall back to OS-specific defaults
            home_dir = Path.home()
            system = platform.system().lower()

            if system == 'darwin':  # macOS
                config_dirs.append(home_dir / 'Library' / 'Application Support' / 'workspace-qdrant')
            elif system == 'windows':
                appdata = os.environ.get('APPDATA')
                if appdata:
                    config_dirs.append(Path(appdata) / 'workspace-qdrant')
                else:
                    config_dirs.append(home_dir / 'AppData' / 'Roaming' / 'workspace-qdrant')
            else:  # Linux/Unix
                config_dirs.append(home_dir / '.config' / 'workspace-qdrant')

        return config_dirs

    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables.

        Supports both new prefixed variables (WORKSPACE_QDRANT_*) and legacy variables.

        Returns:
            Dict containing environment variable configuration
        """
        env_config = {}

        # Load prefixed environment variables
        env_config.update(self._load_prefixed_env_vars())

        # Load legacy environment variables for backward compatibility
        env_config.update(self._load_legacy_env_vars())

        return env_config

    def _load_prefixed_env_vars(self) -> Dict[str, Any]:
        """Load prefixed environment variables (WORKSPACE_QDRANT_*).

        Returns:
            Dict containing prefixed environment configuration
        """
        env_config = {}
        prefix = "WORKSPACE_QDRANT_"

        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue

            # Remove prefix and convert to lowercase
            config_key = key[len(prefix):].lower()

            # Handle nested configuration with double underscore
            if "__" in config_key:
                # Split into section and key
                section, nested_key = config_key.split("__", 1)
                nested_key = nested_key.replace("__", ".")

                if section not in env_config:
                    env_config[section] = {}

                # Set nested value
                self._set_nested_value(env_config[section], nested_key, self._parse_env_value(value))
            else:
                # Handle server-level configuration
                if config_key in ["host", "port", "debug"]:
                    if "server" not in env_config:
                        env_config["server"] = {}
                    env_config["server"][config_key] = self._parse_env_value(value)
                else:
                    env_config[config_key] = self._parse_env_value(value)

        return env_config

    def _load_legacy_env_vars(self) -> Dict[str, Any]:
        """Load legacy environment variables for backward compatibility.

        Returns:
            Dict containing legacy environment configuration
        """
        env_config = {}

        # Legacy Qdrant configuration
        if url := os.getenv("QDRANT_URL"):
            if "qdrant" not in env_config:
                env_config["qdrant"] = {}
            env_config["qdrant"]["url"] = url

        if api_key := os.getenv("QDRANT_API_KEY"):
            if "qdrant" not in env_config:
                env_config["qdrant"] = {}
            env_config["qdrant"]["api_key"] = api_key

        # Legacy embedding configuration
        if model := os.getenv("FASTEMBED_MODEL"):
            if "embedding" not in env_config:
                env_config["embedding"] = {}
            env_config["embedding"]["model"] = model

        if sparse := os.getenv("ENABLE_SPARSE_VECTORS"):
            if "embedding" not in env_config:
                env_config["embedding"] = {}
            env_config["embedding"]["enable_sparse_vectors"] = sparse.lower() == "true"

        # Legacy workspace configuration
        if collection_types := os.getenv("COLLECTION_TYPES"):
            if "workspace" not in env_config:
                env_config["workspace"] = {}
            env_config["workspace"]["collection_types"] = [
                c.strip() for c in collection_types.split(",") if c.strip()
            ]

        if github_user := os.getenv("GITHUB_USER"):
            if "workspace" not in env_config:
                env_config["workspace"] = {}
            env_config["workspace"]["github_user"] = github_user

        return env_config

    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any) -> None:
        """Set nested value in configuration dictionary.

        Args:
            config: Configuration dictionary to modify
            path: Dot-separated path to the value
            value: Value to set
        """
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Union[str, int, float, bool, List[str]]:
        """Parse environment variable value to appropriate Python type.

        Args:
            value: Environment variable value as string

        Returns:
            Parsed value with appropriate type
        """
        # Handle boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Handle comma-separated lists
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        # Handle numeric values
        try:
            # Try integer first
            if "." not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass

        # Return as string
        return value


# Global configuration instance
_global_config: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config(config_file: Optional[str] = None, **kwargs) -> ConfigManager:
    """Get global configuration instance (thread-safe).

    This function provides the global read-only configuration structure
    as specified by the user requirements.

    Args:
        config_file: Path to YAML configuration file (only used on first call)
        **kwargs: Override values (only used on first call)

    Returns:
        ConfigManager: Global configuration instance
    """
    global _global_config

    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = ConfigManager.get_instance(config_file, **kwargs)

    return _global_config


def reset_config() -> None:
    """Reset global configuration (primarily for testing)."""
    global _global_config

    with _config_lock:
        _global_config = None
        ConfigManager.reset_instance()


# Legacy compatibility classes for backward compatibility
class LegacyConfigBase:
    """Base class for legacy configuration compatibility."""

    def __init__(self, config_manager: ConfigManager, path: str):
        self._config_manager = config_manager
        self._path = path

    def __getattr__(self, name: str) -> Any:
        """Get attribute from configuration using dot notation."""
        if self._path:
            full_path = f"{self._path}.{name}"
        else:
            full_path = name

        value = self._config_manager.get(full_path)

        # If value is a dict, return a new LegacyConfigBase for chaining
        if isinstance(value, dict):
            return LegacyConfigBase(self._config_manager, full_path)

        return value


class QdrantConfig(LegacyConfigBase):
    """Legacy compatibility for Qdrant configuration access.

    Provides backward compatibility for code that uses config.qdrant.url patterns.
    """

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, "qdrant")

    @property
    def url(self) -> str:
        return self._config_manager.get("qdrant.url")

    @property
    def api_key(self) -> Optional[str]:
        return self._config_manager.get("qdrant.api_key")

    @property
    def timeout(self) -> int:
        # Convert milliseconds to seconds for backward compatibility
        return self._config_manager.get("qdrant.timeout") // 1000

    @property
    def prefer_grpc(self) -> bool:
        return self._config_manager.get("qdrant.prefer_grpc")


class EmbeddingConfig(LegacyConfigBase):
    """Legacy compatibility for embedding configuration access."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, "embedding")

    @property
    def model(self) -> str:
        return self._config_manager.get("embedding.model")

    @property
    def enable_sparse_vectors(self) -> bool:
        return self._config_manager.get("embedding.enable_sparse_vectors")

    @property
    def chunk_size(self) -> int:
        return self._config_manager.get("embedding.chunk_size")

    @property
    def chunk_overlap(self) -> int:
        return self._config_manager.get("embedding.chunk_overlap")

    @property
    def batch_size(self) -> int:
        return self._config_manager.get("embedding.batch_size")


class WorkspaceConfig(LegacyConfigBase):
    """Legacy compatibility for workspace configuration access."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, "workspace")

    @property
    def collection_types(self) -> List[str]:
        return self._config_manager.get("workspace.collection_types", [])

    @property
    def global_collections(self) -> List[str]:
        return self._config_manager.get("workspace.global_collections", [])

    @property
    def github_user(self) -> Optional[str]:
        return self._config_manager.get("workspace.github_user")

    @property
    def auto_create_collections(self) -> bool:
        return self._config_manager.get("workspace.auto_create_collections", False)

    @property
    def memory_collection_name(self) -> str:
        return self._config_manager.get("workspace.memory_collection_name", "__memory")

    @property
    def code_collection_name(self) -> str:
        return self._config_manager.get("workspace.code_collection_name", "__code")

    @property
    def custom_include_patterns(self) -> List[str]:
        return self._config_manager.get("workspace.custom_include_patterns", [])

    @property
    def custom_exclude_patterns(self) -> List[str]:
        return self._config_manager.get("workspace.custom_exclude_patterns", [])

    @property
    def custom_project_indicators(self) -> Dict[str, Any]:
        return self._config_manager.get("workspace.custom_project_indicators", {})

    @property
    def effective_collection_types(self) -> List[str]:
        """Get effective collection types."""
        return self.collection_types

    def create_pattern_manager(self):
        """Create a PatternManager instance with custom patterns from this config.

        Returns:
            PatternManager instance configured with custom patterns
        """
        # Lazy import to avoid circular dependency
        from .pattern_manager import PatternManager

        return PatternManager(
            custom_include_patterns=self.custom_include_patterns,
            custom_exclude_patterns=self.custom_exclude_patterns,
            custom_project_indicators=self.custom_project_indicators
        )


class GrpcConfig(LegacyConfigBase):
    """Legacy compatibility for gRPC configuration access."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, "grpc")

    @property
    def enabled(self) -> bool:
        return self._config_manager.get("grpc.enabled", True)

    @property
    def host(self) -> str:
        return self._config_manager.get("grpc.host", "127.0.0.1")

    @property
    def port(self) -> int:
        return self._config_manager.get("grpc.port", 50051)

    @property
    def fallback_to_direct(self) -> bool:
        return self._config_manager.get("grpc.fallback_to_direct", True)

    @property
    def connection_timeout(self) -> float:
        # Convert milliseconds to seconds for backward compatibility
        return self._config_manager.get("grpc.connection_timeout", 10000) / 1000.0

    @property
    def max_retries(self) -> int:
        return self._config_manager.get("grpc.max_retries", 3)

    @property
    def retry_backoff_multiplier(self) -> float:
        return self._config_manager.get("grpc.retry_backoff_multiplier", 1.5)

    @property
    def health_check_interval(self) -> float:
        # Convert milliseconds to seconds for backward compatibility
        return self._config_manager.get("grpc.health_check_interval", 30000) / 1000.0

    @property
    def max_message_length(self) -> int:
        return self._config_manager.get("grpc.max_message_length", 104857600)

    @property
    def keepalive_time(self) -> int:
        # Convert milliseconds to seconds for backward compatibility
        return self._config_manager.get("grpc.keepalive_time", 30000) // 1000


class AutoIngestionConfig(LegacyConfigBase):
    """Legacy compatibility for auto-ingestion configuration access."""

    def __init__(self, config_manager: ConfigManager):
        super().__init__(config_manager, "auto_ingestion")

    @property
    def enabled(self) -> bool:
        return self._config_manager.get("auto_ingestion.enabled", True)

    @property
    def auto_create_watches(self) -> bool:
        return self._config_manager.get("auto_ingestion.auto_create_watches", True)

    @property
    def include_common_files(self) -> bool:
        return self._config_manager.get("auto_ingestion.include_common_files", True)

    @property
    def include_source_files(self) -> bool:
        return self._config_manager.get("auto_ingestion.include_source_files", False)

    @property
    def target_collection_suffix(self) -> str:
        return self._config_manager.get("auto_ingestion.target_collection_suffix", "scratchbook")

    @property
    def max_files_per_batch(self) -> int:
        return self._config_manager.get("auto_ingestion.max_files_per_batch", 5)

    @property
    def batch_delay_seconds(self) -> float:
        return self._config_manager.get("auto_ingestion.batch_delay_seconds", 2.0)

    @property
    def max_file_size_mb(self) -> int:
        # Convert bytes to MB for backward compatibility
        return self._config_manager.get("auto_ingestion.max_file_size_mb", 52428800) // (1024 * 1024)

    @property
    def debounce_seconds(self) -> int:
        # Convert milliseconds to seconds for backward compatibility
        return self._config_manager.get("auto_ingestion.debounce_seconds", 10000) // 1000


class Config:
    """Legacy compatibility wrapper for the new dictionary-based configuration.

    Provides backward compatibility for existing code that uses the old Pydantic-based
    Config class while delegating to the new ConfigManager under the hood.

    This class maintains the same interface as the original Config class but uses
    the new dictionary-based configuration system internally.
    """

    def __init__(self, config_file: Optional[str] = None, **kwargs) -> None:
        """Initialize configuration with backward compatibility.

        Args:
            config_file: Path to YAML configuration file
            **kwargs: Override values for configuration parameters
        """
        self._config_manager = get_config(config_file, **kwargs)

        # Create legacy config objects
        self._qdrant = QdrantConfig(self._config_manager)
        self._embedding = EmbeddingConfig(self._config_manager)
        self._workspace = WorkspaceConfig(self._config_manager)
        self._grpc = GrpcConfig(self._config_manager)
        self._auto_ingestion = AutoIngestionConfig(self._config_manager)

    @property
    def host(self) -> str:
        return self._config_manager.get("server.host", "127.0.0.1")

    @property
    def port(self) -> int:
        return self._config_manager.get("server.port", 8000)

    @property
    def debug(self) -> bool:
        return self._config_manager.get("server.debug", False)

    @property
    def qdrant(self) -> QdrantConfig:
        return self._qdrant

    @property
    def embedding(self) -> EmbeddingConfig:
        return self._embedding

    @property
    def workspace(self) -> WorkspaceConfig:
        return self._workspace

    @property
    def grpc(self) -> GrpcConfig:
        return self._grpc

    @property
    def auto_ingestion(self) -> AutoIngestionConfig:
        return self._auto_ingestion

    @classmethod
    def from_yaml(cls, config_file: str, **kwargs) -> "Config":
        """Create Config instance from YAML file.

        Args:
            config_file: Path to YAML configuration file
            **kwargs: Additional configuration overrides

        Returns:
            Config instance with YAML configuration loaded
        """
        return cls(config_file=config_file, **kwargs)

    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """Export current configuration to YAML format.

        Args:
            file_path: Optional path to save YAML file

        Returns:
            YAML string representation of the configuration
        """
        config_dict = self._config_manager.get_all()
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        if file_path:
            Path(file_path).write_text(yaml_str, encoding="utf-8")

        return yaml_str

    @property
    def qdrant_client_config(self) -> Dict[str, Any]:
        """Get Qdrant client configuration dictionary for QdrantClient initialization.

        Returns:
            dict: Configuration dictionary for QdrantClient
        """
        config = {
            "url": self.qdrant.url,
            "timeout": self.qdrant.timeout,
            "prefer_grpc": self.qdrant.prefer_grpc,
        }

        if self.qdrant.api_key:
            config["api_key"] = self.qdrant.api_key

        return config

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues.

        Returns:
            List[str]: List of validation error messages
        """
        return self._config_manager.validate()

    def get_auto_ingestion_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about auto-ingestion configuration.

        Returns:
            Dict containing diagnostic information
        """
        target_suffix = self.auto_ingestion.target_collection_suffix
        available_types = self.workspace.effective_collection_types
        auto_create = self.workspace.auto_create_collections

        status = "valid"
        recommendations = []

        if self.auto_ingestion.enabled:
            if target_suffix:
                if available_types and target_suffix not in available_types:
                    status = "invalid_target_suffix"
                    recommendations.append(
                        f"Add '{target_suffix}' to workspace.collection_types: {available_types}"
                    )
                elif not available_types and not auto_create:
                    status = "missing_collection_config"
                    recommendations.extend([
                        f"Add '{target_suffix}' to workspace.collection_types",
                        "OR enable workspace.auto_create_collections"
                    ])
            elif not target_suffix and available_types:
                status = "missing_target_suffix"
                recommendations.append(
                    f"Set auto_ingestion.target_collection_suffix to one of: {available_types}"
                )
            elif not target_suffix and not available_types and not auto_create:
                status = "no_collection_config"
                recommendations.extend([
                    "Set auto_ingestion.target_collection_suffix (e.g., 'scratchbook')",
                    "Add the suffix to workspace.collection_types",
                    "OR enable workspace.auto_create_collections"
                ])
        else:
            status = "disabled"

        return {
            "enabled": self.auto_ingestion.enabled,
            "target_suffix": target_suffix,
            "available_types": available_types,
            "auto_create": auto_create,
            "configuration_status": status,
            "recommendations": recommendations,
            "summary": self._get_auto_ingestion_summary(status, target_suffix, available_types)
        }

    def _get_auto_ingestion_summary(self, status: str, target_suffix: str, available_types: List[str]) -> str:
        """Get a human-readable summary of auto-ingestion configuration status."""
        if status == "disabled":
            return "Auto-ingestion is disabled"
        elif status == "valid":
            if target_suffix:
                return f"Auto-ingestion configured to use collection suffix '{target_suffix}'"
            else:
                return "Auto-ingestion enabled with fallback collection selection"
        elif status == "invalid_target_suffix":
            return f"Target suffix '{target_suffix}' not found in configured types {available_types}"
        elif status == "missing_collection_config":
            return f"Target suffix '{target_suffix}' specified but no collections configured to create it"
        elif status == "missing_target_suffix":
            return f"No target suffix specified but types available: {available_types}"
        elif status == "no_collection_config":
            return "Auto-ingestion enabled but no collection configuration found"
        else:
            return f"Unknown configuration status: {status}"

    def get_effective_auto_ingestion_behavior(self) -> str:
        """Get a user-friendly description of how auto-ingestion will behave.

        Returns:
            str: Human-readable description of auto-ingestion behavior
        """
        if not self.auto_ingestion.enabled:
            return "Auto-ingestion is disabled. No automatic file processing will occur."

        target_suffix = self.auto_ingestion.target_collection_suffix
        available_types = self.workspace.effective_collection_types
        auto_create = self.workspace.auto_create_collections

        if target_suffix and available_types and target_suffix in available_types:
            return f"Will use collection '{{project-name}}-{target_suffix}' for auto-ingestion."
        elif target_suffix and auto_create:
            return f"Will create and use collection '{{project-name}}-{target_suffix}' for auto-ingestion."
        elif not target_suffix:
            return "Will use intelligent fallback selection for auto-ingestion collections."
        else:
            return f"Configuration may need adjustment. Target suffix '{target_suffix}' specified but not in available types."