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
        """Load comprehensive default configuration from asset file.

        Returns:
            Dict containing all possible configuration paths with default values
        """
        # Get path to the default configuration asset
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent.parent
        asset_file = project_root / "assets" / "default_config.yaml"

        try:
            if asset_file.exists():
                with asset_file.open("r", encoding="utf-8") as f:
                    asset_config = yaml.safe_load(f)

                if isinstance(asset_config, dict):
                    # Remove Rust-specific section as it's not needed in Python
                    asset_config.pop("rust", None)
                    return asset_config
                else:
                    logger.warning(f"Asset config is not a dictionary: {type(asset_config)}")
            else:
                logger.warning(f"Default config asset not found: {asset_file}")
        except Exception as e:
            logger.error(f"Error loading default config asset {asset_file}: {e}")

        # Fallback to minimal hardcoded defaults if asset loading fails
        return {
            "server": {"host": "127.0.0.1", "port": 8000, "debug": False},
            "qdrant": {"url": "http://localhost:6333", "api_key": None, "timeout": 30000},
            "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2", "chunk_size": 800},
            "workspace": {"collection_types": [], "global_collections": []},
            "grpc": {"enabled": True, "host": "127.0.0.1", "port": 50051},
            "auto_ingestion": {"enabled": True, "target_collection_suffix": "scratchbook"},
            "logging": {"level": "info", "use_file_logging": False},
            "performance": {"max_concurrent_tasks": 4, "default_timeout_ms": 30000},
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

        Supports prefixed variables (WORKSPACE_QDRANT_*).

        Returns:
            Dict containing environment variable configuration
        """
        return self._load_prefixed_env_vars()

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


# End of ConfigManager implementation - all backward compatibility removed