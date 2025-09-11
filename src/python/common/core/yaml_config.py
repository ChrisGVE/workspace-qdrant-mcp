"""
YAML-first configuration system with environment variable substitution.

This module implements a comprehensive configuration system that supports:
- YAML-first configuration with environment variable substitution
- Configuration hierarchy (CLI → project → user → system → defaults)
- JSON schema validation
- Type-safe configuration access
- Hot-reload capabilities

Configuration Hierarchy (highest to lowest priority):
1. CLI --config parameter
2. Project .workspace-qdrant.yaml
3. User ~/.config/workspace-qdrant/config.yaml
4. System /etc/workspace-qdrant/config.yaml
5. Built-in defaults

Environment Variable Substitution:
- Use ${VAR_NAME} syntax in YAML files
- Falls back to empty string if variable is not set
- Supports nested substitution: ${PREFIX_${SUFFIX}}
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from jsonschema import Draft7Validator, ValidationError
from pydantic import BaseModel, Field, field_validator

from ..observability import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""

    pass


class QdrantConfig(BaseModel):
    """Qdrant database connection settings."""

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    retry_count: int = 3
    use_https: bool = False
    verify_ssl: bool = True


class DaemonConfig(BaseModel):
    """Daemon configuration and behavior."""

    database_path: str = "~/.config/workspace-qdrant/state.db"
    max_concurrent_jobs: int = 4
    job_timeout_seconds: int = 300
    max_memory_mb: int = 1024
    max_cpu_percent: int = 80

    class PriorityLevels(BaseModel):
        mcp_server: str = "high"
        cli_commands: str = "medium"
        background_watching: str = "low"

        @field_validator("mcp_server", "cli_commands", "background_watching")
        @classmethod
        def validate_priority(cls, v):
            if v not in ["low", "medium", "high"]:
                raise ValueError("Priority must be one of: low, medium, high")
            return v

    class GrpcConfig(BaseModel):
        host: str = "127.0.0.1"
        port: int = 50051
        max_message_size_mb: int = 100

    priority_levels: PriorityLevels = PriorityLevels()
    grpc: GrpcConfig = GrpcConfig()


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = "fastembed"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "prithivida/Splade_PP_en_v1"

    class OpenAIConfig(BaseModel):
        api_key: Optional[str] = None
        model: str = "text-embedding-3-small"

    class HuggingFaceConfig(BaseModel):
        api_key: Optional[str] = None
        model: str = "sentence-transformers/all-MiniLM-L6-v2"

    openai: OpenAIConfig = OpenAIConfig()
    huggingface: HuggingFaceConfig = HuggingFaceConfig()

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v):
        if v not in ["fastembed", "openai", "huggingface"]:
            raise ValueError("Provider must be one of: fastembed, openai, huggingface")
        return v


class CollectionsConfig(BaseModel):
    """Collection management settings."""

    auto_create: bool = False
    default_global: List[str] = Field(default_factory=lambda: ["scratchbook"])
    reserved_prefixes: List[str] = Field(
        default_factory=lambda: ["_", "system_", "temp_"]
    )

    class CollectionSettings(BaseModel):
        description: Optional[str] = None
        auto_ingest: bool = False

    settings: Dict[str, CollectionSettings] = Field(default_factory=dict)


class WatchingConfig(BaseModel):
    """Folder watching configuration."""

    auto_watch_project: bool = True
    debounce_seconds: int = 5
    max_file_size_mb: int = 50
    recursive: bool = True
    max_depth: int = 5
    follow_symlinks: bool = False

    include_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.txt",
            "*.md",
            "*.pdf",
            "*.epub",
            "*.docx",
            "*.py",
            "*.js",
            "*.ts",
            "*.html",
            "*.css",
        ]
    )

    ignore_patterns: List[str] = Field(
        default_factory=lambda: [
            "*.tmp",
            "*.log",
            "*.cache",
            ".git/*",
            "node_modules/*",
            "__pycache__/*",
            "*.pyc",
            ".DS_Store",
        ]
    )


class ProcessingConfig(BaseModel):
    """Processing and chunking settings."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    class PDFConfig(BaseModel):
        extract_images: bool = False
        extract_tables: bool = True

    class DocxConfig(BaseModel):
        extract_images: bool = False
        extract_styles: bool = False

    class CodeConfig(BaseModel):
        include_comments: bool = True
        language_detection: bool = True

    pdf: PDFConfig = PDFConfig()
    docx: DocxConfig = DocxConfig()
    code: CodeConfig = CodeConfig()


class WebUIConfig(BaseModel):
    """Web interface settings."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 3000
    auto_launch: bool = False
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    auth_required: bool = False


class LoggingConfig(BaseModel):
    """Logging and observability settings."""

    level: str = "INFO"
    format: str = "json"
    file_path: str = "~/.config/workspace-qdrant/logs/daemon.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    components: Dict[str, str] = Field(
        default_factory=lambda: {
            "qdrant_client": "WARN",
            "embedding": "INFO",
            "file_watcher": "INFO",
            "grpc_server": "WARN",
        }
    )

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        if v not in ["DEBUG", "INFO", "WARN", "ERROR"]:
            raise ValueError("Level must be one of: DEBUG, INFO, WARN, ERROR")
        return v

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        if v not in ["json", "text"]:
            raise ValueError("Format must be one of: json, text")
        return v


class MonitoringConfig(BaseModel):
    """Performance and monitoring settings."""

    enabled: bool = True
    metrics_port: int = 9090
    health_check_interval_seconds: int = 60

    class Thresholds(BaseModel):
        max_response_time_ms: int = 1000
        max_memory_usage_mb: int = 512
        max_cpu_usage_percent: int = 70

    thresholds: Thresholds = Thresholds()


class DevelopmentConfig(BaseModel):
    """Development and debugging settings."""

    debug_mode: bool = False
    profile_performance: bool = False
    mock_embedding: bool = False
    test_data_path: str = "tests/data"


class WorkspaceConfig(BaseModel):
    """Complete workspace-qdrant-mcp configuration."""

    qdrant: QdrantConfig = QdrantConfig()
    daemon: DaemonConfig = DaemonConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    collections: CollectionsConfig = CollectionsConfig()
    watching: WatchingConfig = WatchingConfig()
    processing: ProcessingConfig = ProcessingConfig()
    web_ui: WebUIConfig = WebUIConfig()
    logging: LoggingConfig = LoggingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    development: DevelopmentConfig = DevelopmentConfig()


class ConfigLoader:
    """
    Configuration loader with hierarchy support and environment variable substitution.
    """

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize the configuration loader."""
        self.schema_path = schema_path or (Path(__file__).parent / "config_schema.json")
        self._validator = self._load_schema_validator()

    def _load_schema_validator(self) -> Draft7Validator:
        """Load and create JSON schema validator."""
        try:
            with open(self.schema_path) as f:
                schema = json.load(f)
            return Draft7Validator(schema)
        except Exception as e:
            logger.warning(f"Failed to load schema from {self.schema_path}: {e}")
            return None

    def _substitute_env_vars(self, data: Union[Dict, List, str, Any]) -> Any:
        """
        Recursively substitute environment variables in configuration data.

        Supports ${VAR_NAME} syntax with fallback to empty string for missing variables.
        """
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Pattern to match ${VAR_NAME} and ${VAR_NAME:default_value}
            pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.getenv(var_name, default_value)

            return pattern.sub(replacer, data)
        else:
            return data

    def _get_config_paths(self, cli_config_path: Optional[Path] = None) -> List[Path]:
        """
        Get configuration file paths in order of priority (highest to lowest).
        """
        paths = []

        # 1. CLI --config parameter (highest priority)
        if cli_config_path:
            paths.append(Path(cli_config_path).expanduser())

        # 2. Project .workspace-qdrant.yaml
        project_config = Path.cwd() / ".workspace-qdrant.yaml"
        if project_config.exists():
            paths.append(project_config)

        # 3. User ~/.config/workspace-qdrant/config.yaml
        user_config_dir = Path.home() / ".config" / "workspace-qdrant"
        user_config = user_config_dir / "config.yaml"
        if user_config.exists():
            paths.append(user_config)

        # 4. System /etc/workspace-qdrant/config.yaml (Unix-like systems)
        if os.name != "nt":  # Not Windows
            system_config = Path("/etc/workspace-qdrant/config.yaml")
            if system_config.exists():
                paths.append(system_config)

        return paths

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        """Load and parse YAML configuration file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                return data
        except Exception as e:
            logger.error(f"Failed to load configuration from {path}: {e}")
            raise ConfigurationError(f"Failed to load configuration from {path}: {e}")

    def _merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple configuration dictionaries.
        Later configurations override earlier ones.
        """
        result = {}

        for config in configs:
            result = self._deep_merge(result, config)

        return result

    def _deep_merge(
        self, base: Dict[str, Any], overlay: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        """
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self, config_data: Dict[str, Any]) -> None:
        """Validate configuration against JSON schema."""
        if not self._validator:
            logger.warning("No schema validator available, skipping validation")
            return

        try:
            self._validator.validate(config_data)
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            raise ConfigurationError(f"Configuration validation failed: {e.message}")

    def load_config(self, cli_config_path: Optional[Path] = None) -> WorkspaceConfig:
        """
        Load configuration with full hierarchy support.

        Args:
            cli_config_path: Optional path to configuration file from CLI argument

        Returns:
            WorkspaceConfig: Validated configuration object

        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        logger.info("Loading workspace configuration")

        # Get all possible configuration file paths
        config_paths = self._get_config_paths(cli_config_path)

        # Load configurations from files (order matters for precedence)
        configs = []
        for path in config_paths:
            logger.info(f"Loading configuration from: {path}")
            config_data = self._load_yaml_file(path)
            configs.append(config_data)

        # Merge all configurations (later ones override earlier ones)
        merged_config = self._merge_configs(configs) if configs else {}

        # Substitute environment variables
        substituted_config = self._substitute_env_vars(merged_config)

        # Validate against schema
        self._validate_config(substituted_config)

        # Create and return typed configuration object
        try:
            config = WorkspaceConfig(**substituted_config)
            logger.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to create configuration object: {e}")
            raise ConfigurationError(f"Failed to create configuration object: {e}")

    def save_config(self, config: WorkspaceConfig, path: Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration object to save
            path: Path to save configuration file
        """
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save as YAML
            config_dict = config.dict()
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {path}: {e}")
            raise ConfigurationError(f"Failed to save configuration to {path}: {e}")


class YAMLConfigLoader:
    """
    Simplified YAML configuration loader for multi-component testing.
    """

    def __init__(self):
        """Initialize the YAML configuration loader."""
        pass

    def _substitute_env_vars(self, data: Union[Dict, List, str, Any]) -> Any:
        """
        Recursively substitute environment variables in configuration data.

        Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
        """
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Pattern to match ${VAR_NAME} and ${VAR_NAME:default_value}
            pattern = re.compile(r'\$\{([^}:]+)(?::([^}]*))?\}')

            def replacer(match):
                var_name = match.group(1)
                default_value = match.group(2) if match.group(2) is not None else ""
                return os.getenv(var_name, default_value)

            result = pattern.sub(replacer, data)
            
            # Try to convert numeric strings to appropriate types
            if result.isdigit():
                return int(result)
            try:
                return float(result)
            except ValueError:
                return result
        else:
            return data

    def _deep_merge(
        self, base: Dict[str, Any], overlay: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        """
        result = base.copy()

        for key, value in overlay.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    async def load_with_hierarchy(self, config_files: List[str]) -> Dict[str, Any]:
        """
        Load configuration files with hierarchy support.
        
        Args:
            config_files: List of configuration file paths in order of priority
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        configs = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data is None:
                        data = {}
                    configs.append(data)
            except FileNotFoundError:
                continue  # Skip missing files
            except Exception as e:
                logger.error(f"Failed to load configuration from {config_file}: {e}")
                continue

        # Merge all configurations (later ones override earlier ones)
        merged_config = {}
        for config in configs:
            merged_config = self._deep_merge(merged_config, config)

        return merged_config

    async def load_with_env_substitution(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration file with environment variable substitution.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration with environment variables substituted
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                if data is None:
                    data = {}
                
            # Substitute environment variables
            return self._substitute_env_vars(data)
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_file}: {e}")
            raise ConfigurationError(f"Failed to load configuration from {config_file}: {e}")

    async def load_and_validate(self, config_file: str) -> Dict[str, Any]:
        """
        Load and validate configuration file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Dict[str, Any]: Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = await self.load_with_env_substitution(config_file)
            
            # Basic validation - check for required fields and data types
            if 'qdrant' in config:
                qdrant_config = config['qdrant']
                if 'url' in qdrant_config:
                    url = qdrant_config['url']
                    if not isinstance(url, str) or not url.startswith(('http://', 'https://')):
                        raise ValueError("Invalid Qdrant URL format")
                        
                if 'port' in qdrant_config and not isinstance(qdrant_config['port'], (int, str)):
                    raise ValueError("Invalid port type")
            
            if 'embedding' in config:
                embedding_config = config['embedding']
                if 'batch_size' in embedding_config:
                    batch_size = embedding_config['batch_size']
                    if isinstance(batch_size, str):
                        try:
                            batch_size = int(batch_size)
                            embedding_config['batch_size'] = batch_size
                        except ValueError:
                            raise ValueError("Invalid batch_size value")
                    if batch_size < 1:
                        raise ValueError("batch_size must be positive")
            
            return config
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise ValueError(f"Invalid configuration: {e}")
            logger.error(f"Failed to validate configuration: {e}")
            raise ValueError(f"Invalid configuration: {e}")


# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(cli_config_path: Optional[str] = None) -> WorkspaceConfig:
    """
    Load workspace configuration with hierarchy support.

    Args:
        cli_config_path: Optional path to configuration file from CLI

    Returns:
        WorkspaceConfig: Validated configuration object
    """
    path = Path(cli_config_path) if cli_config_path else None
    return _config_loader.load_config(path)


def save_config(config: WorkspaceConfig, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        path: Path to save configuration file
    """
    _config_loader.save_config(config, Path(path))


def create_default_config() -> WorkspaceConfig:
    """Create a default configuration with all default values."""
    return WorkspaceConfig()


def expand_path(path: str) -> Path:
    """Expand user home directory and environment variables in path."""
    return Path(os.path.expandvars(os.path.expanduser(path)))