"""
Enhanced configuration management for workspace-qdrant-mcp with environment support.

This is a simplified version that avoids threading issues while providing
environment-based configuration loading, validation, and YAML support.
"""

from loguru import logger
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# logger imported from loguru


class SecurityConfig(BaseModel):
    """Configuration for security features and sensitive data handling."""

    mask_sensitive_logs: bool = True
    validate_ssl: bool = True
    allow_http: bool = False
    cors_enabled: bool = False
    cors_origins: List[str] = Field(default_factory=list)
    rate_limiting: bool = False
    max_requests_per_minute: int = 1000
    request_size_limit: int = 1048576  # 1MB default
    authentication_required: bool = False

    # SSL/TLS Configuration
    ssl_verify_certificates: bool = True
    ssl_ca_cert_path: Optional[str] = None
    ssl_client_cert_path: Optional[str] = None
    ssl_client_key_path: Optional[str] = None

    # Authentication Configuration
    qdrant_auth_token: Optional[str] = None
    qdrant_api_key: Optional[str] = None

    # Environment-specific SSL behavior
    development_allow_insecure_localhost: bool = True
    production_enforce_ssl: bool = True


class MonitoringConfig(BaseModel):
    """Configuration for monitoring, metrics, and observability."""

    metrics_enabled: bool = False
    tracing_enabled: bool = False
    health_endpoint: str = "/health"
    metrics_endpoint: str = "/metrics"
    log_structured: bool = False
    retention_days: int = 7
    alerts_enabled: bool = False


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization and resource limits."""

    worker_processes: int = 1
    max_connections: int = 100
    keepalive_timeout: int = 65
    request_timeout: int = 30
    memory_limit: str = "512MB"
    cpu_limit: float = 1.0


class DevelopmentConfig(BaseModel):
    """Configuration for development-specific features."""

    hot_reload: bool = False
    config_watch: bool = False
    detailed_logging: bool = False
    performance_metrics: bool = False
    mock_external_services: bool = False


class EmbeddingConfig(BaseModel):
    """Enhanced configuration for embedding generation and text processing."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_sparse_vectors: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    batch_size: int = 50
    cache_embeddings: bool = False
    embedding_timeout: int = 30
    max_concurrent_requests: int = 10

    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk_overlap is less than chunk_size."""
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v

    @validator("batch_size")
    def validate_batch_size(cls, v):
        """Ensure batch_size is within reasonable limits."""
        if v <= 0:
            raise ValueError("batch_size must be positive")
        if v > 1000:
            raise ValueError("batch_size should not exceed 1000")
        return v


class QdrantConfig(BaseModel):
    """Enhanced configuration for Qdrant vector database connection."""

    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False
    health_check_interval: int = 60
    retry_attempts: int = 3
    connection_pool_size: int = 5

    @validator("url")
    def validate_url(cls, v):
        """Ensure URL format is valid."""
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @validator("timeout")
    def validate_timeout(cls, v):
        """Ensure timeout is reasonable."""
        if v <= 0:
            raise ValueError("timeout must be positive")
        if v > 300:
            raise ValueError("timeout should not exceed 300 seconds")
        return v


class WorkspaceConfig(BaseModel):
    """Enhanced configuration for workspace and project management."""

    collection_types: List[str] = []
    global_collections: List[str] = []
    github_user: Optional[str] = None
    auto_create_collections: bool = True
    cleanup_on_exit: bool = False
    memory_collection_name: str = "__memory"
    code_collection_name: str = "__code"

    @property
    def effective_collection_types(self) -> List[str]:
        """Get effective collection types."""
        return self.collection_types

    @validator("collection_types", "global_collections")
    def validate_collections(cls, v):
        """Ensure collection lists have reasonable size."""
        if v and len(v) > 50:
            raise ValueError("Too many collection types configured (max 50)")
        return v


class EnhancedConfig(BaseSettings):
    """Enhanced configuration class with environment-based loading and validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WORKSPACE_QDRANT_",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment and server configuration
    environment: str = Field(default="development", env="APP_ENV")
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    reload: bool = False

    # Component configurations
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    def __init__(
        self,
        environment: Optional[str] = None,
        config_dir: Optional[Path] = None,
        **kwargs,
    ):
        """Initialize enhanced configuration with environment-specific loading.

        Args:
            environment: Target environment (development, staging, production)
            config_dir: Directory containing configuration files
            **kwargs: Override values for configuration parameters
        """
        # Set environment before initialization
        if environment:
            os.environ["APP_ENV"] = environment

        # Initialize with base settings
        super().__init__(**kwargs)

        # Set up configuration directory
        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"
        self._config_dir = Path(config_dir)

        # Load configuration files and environment variables
        self._config_files_loaded = []
        self._load_configuration_files()
        self._load_legacy_env_vars()
        self._load_nested_env_vars()

        # Validate configuration
        self._validation_errors = self.validate_config()

    def _load_configuration_files(self) -> None:
        """Load configuration from YAML files based on environment."""
        if not YAML_AVAILABLE:
            logger.info("PyYAML not available, skipping YAML configuration files")
            return

        config_files = []

        # Load environment-specific configuration
        env_config_file = self._config_dir / f"{self.environment}.yaml"
        if env_config_file.exists():
            config_files.append(env_config_file)

        # Load local override configuration
        local_config_file = self._config_dir / "local.yaml"
        if local_config_file.exists():
            config_files.append(local_config_file)

        # Load and merge configuration files
        for config_file in config_files:
            try:
                config_data = self._load_yaml_file(config_file)
                if config_data:
                    self._apply_config_data(config_data)
                    self._config_files_loaded.append(str(config_file))
                    logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load configuration from {config_file}: {e}")

    def _load_yaml_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load and parse YAML configuration file with variable substitution."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Perform environment variable substitution
            content = self._substitute_env_vars(content)

            # Parse YAML
            config_data = yaml.safe_load(content)
            return config_data
        except Exception as e:
            logger.error(f"Error loading YAML file {file_path}: {e}")
            return None

    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_expr = match.group(1)
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
            else:
                var_name = var_expr
                default_value = ""

            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replace_var, content)

    def _apply_config_data(self, config_data: Dict[str, Any]) -> None:
        """Apply configuration data to the config object."""
        if not config_data:
            return

        # Apply server configuration
        if "server" in config_data:
            server_config = config_data["server"]
            for key, value in server_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Apply component configurations
        component_mapping = {
            "qdrant": self.qdrant,
            "embedding": self.embedding,
            "workspace": self.workspace,
            "security": self.security,
            "monitoring": self.monitoring,
            "performance": self.performance,
            "development": self.development,
        }

        for component_name, component_obj in component_mapping.items():
            if component_name in config_data:
                component_data = config_data[component_name]
                for key, value in component_data.items():
                    if hasattr(component_obj, key):
                        setattr(component_obj, key, value)

    def _load_nested_env_vars(self) -> None:
        """Load nested configuration from environment variables."""
        # Enhanced nested environment variable loading
        env_mappings = [
            # Qdrant configuration
            ("WORKSPACE_QDRANT_QDRANT__URL", lambda v: setattr(self.qdrant, "url", v)),
            (
                "WORKSPACE_QDRANT_QDRANT__API_KEY",
                lambda v: setattr(self.qdrant, "api_key", v),
            ),
            (
                "WORKSPACE_QDRANT_QDRANT__TIMEOUT",
                lambda v: setattr(self.qdrant, "timeout", int(v)),
            ),
            (
                "WORKSPACE_QDRANT_QDRANT__PREFER_GRPC",
                lambda v: setattr(self.qdrant, "prefer_grpc", v.lower() == "true"),
            ),
            # Embedding configuration
            (
                "WORKSPACE_QDRANT_EMBEDDING__MODEL",
                lambda v: setattr(self.embedding, "model", v),
            ),
            (
                "WORKSPACE_QDRANT_EMBEDDING__ENABLE_SPARSE_VECTORS",
                lambda v: setattr(
                    self.embedding, "enable_sparse_vectors", v.lower() == "true"
                ),
            ),
            (
                "WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE",
                lambda v: setattr(self.embedding, "chunk_size", int(v)),
            ),
            (
                "WORKSPACE_QDRANT_EMBEDDING__CHUNK_OVERLAP",
                lambda v: setattr(self.embedding, "chunk_overlap", int(v)),
            ),
            (
                "WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE",
                lambda v: setattr(self.embedding, "batch_size", int(v)),
            ),
            # Workspace configuration
            (
                "WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER",
                lambda v: setattr(self.workspace, "github_user", v),
            ),
            (
                "WORKSPACE_QDRANT_WORKSPACE__AUTO_CREATE_COLLECTIONS",
                lambda v: setattr(self.workspace, "auto_create_collections", v.lower() == "true"),
            ),
            (
                "WORKSPACE_QDRANT_WORKSPACE__MEMORY_COLLECTION_NAME",
                lambda v: setattr(self.workspace, "memory_collection_name", v),
            ),
            (
                "WORKSPACE_QDRANT_WORKSPACE__CODE_COLLECTION_NAME",
                lambda v: setattr(self.workspace, "code_collection_name", v),
            ),
        ]

        for env_var, setter in env_mappings:
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setter(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} - {e}")

        # Handle list environment variables
        if collection_types := os.getenv("WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES"):
            self.workspace.collection_types = [
                c.strip() for c in collection_types.split(",") if c.strip()
            ]

        if global_collections := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS"
        ):
            self.workspace.global_collections = [
                c.strip() for c in global_collections.split(",") if c.strip()
            ]

    def _load_legacy_env_vars(self) -> None:
        """Load legacy environment variables for backward compatibility."""
        legacy_mappings = [
            # Legacy Qdrant config
            ("QDRANT_URL", lambda v: setattr(self.qdrant, "url", v)),
            ("QDRANT_API_KEY", lambda v: setattr(self.qdrant, "api_key", v)),
            # Legacy embedding config
            ("FASTEMBED_MODEL", lambda v: setattr(self.embedding, "model", v)),
            (
                "ENABLE_SPARSE_VECTORS",
                lambda v: setattr(
                    self.embedding, "enable_sparse_vectors", v.lower() == "true"
                ),
            ),
            ("CHUNK_SIZE", lambda v: setattr(self.embedding, "chunk_size", int(v))),
            (
                "CHUNK_OVERLAP",
                lambda v: setattr(self.embedding, "chunk_overlap", int(v)),
            ),
            ("BATCH_SIZE", lambda v: setattr(self.embedding, "batch_size", int(v))),
            # Legacy workspace config
            ("GITHUB_USER", lambda v: setattr(self.workspace, "github_user", v)),
        ]

        for env_var, setter in legacy_mappings:
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setter(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid legacy value for {env_var}: {value} - {e}")

        # Handle legacy list variables
        if collection_types := os.getenv("COLLECTION_TYPES"):
            self.workspace.collection_types = [
                c.strip() for c in collection_types.split(",") if c.strip()
            ]

        if global_collections := os.getenv("GLOBAL_COLLECTIONS"):
            self.workspace.global_collections = [
                c.strip() for c in global_collections.split(",") if c.strip()
            ]

    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self._validation_errors) == 0

    @property
    def validation_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self._validation_errors.copy()

    @property
    def qdrant_client_config(self) -> Dict[str, Any]:
        """Get Qdrant client configuration dictionary."""
        config = {
            "url": self.qdrant.url,
            "timeout": self.qdrant.timeout,
            "prefer_grpc": self.qdrant.prefer_grpc,
        }

        if self.qdrant.api_key:
            config["api_key"] = self.qdrant.api_key

        return config

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Environment validation
        valid_environments = {"development", "staging", "production"}
        if self.environment not in valid_environments:
            issues.append(
                f"Invalid environment '{self.environment}'. Must be one of: {', '.join(valid_environments)}"
            )

        # Qdrant validation
        if not self.qdrant.url:
            issues.append("Qdrant URL is required")
        elif not (
            self.qdrant.url.startswith("http://")
            or self.qdrant.url.startswith("https://")
        ):
            issues.append("Qdrant URL must start with http:// or https://")

        # Production-specific validation
        if self.environment == "production":
            if self.debug:
                issues.append("Debug mode should be disabled in production")
            if self.security.allow_http and not self.qdrant.url.startswith("https://"):
                issues.append("HTTPS should be used in production")
            if not self.security.mask_sensitive_logs:
                issues.append("Sensitive log masking should be enabled in production")

        # Security validation
        if self.security.cors_enabled and not self.security.cors_origins:
            issues.append("CORS origins must be specified when CORS is enabled")

        # Embedding validation - using Pydantic validators
        try:
            EmbeddingConfig(**self.embedding.dict())
        except ValueError as e:
            issues.append(f"Embedding configuration error: {e}")

        # Workspace validation - using Pydantic validators
        try:
            WorkspaceConfig(**self.workspace.dict())
        except ValueError as e:
            issues.append(f"Workspace configuration error: {e}")

        return issues

    def reload_config(self) -> None:
        """Manually reload configuration."""
        self._load_configuration_files()
        self._load_legacy_env_vars()
        self._load_nested_env_vars()
        self._validation_errors = self.validate_config()

    def mask_sensitive_value(self, value: str, mask_char: str = "*") -> str:
        """Mask sensitive values for logging."""
        if not value or not self.security.mask_sensitive_logs:
            return value

        if len(value) <= 6:
            return mask_char * len(value)

        # Show first 2 and last 2 characters
        return value[:2] + mask_char * (len(value) - 4) + value[-2:]

    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for debugging."""
        summary = {
            "environment": self.environment,
            "config_files_loaded": self._config_files_loaded,
            "validation_status": "valid" if self.is_valid else "invalid",
            "validation_errors": self.validation_errors,
            "qdrant_url": self.mask_sensitive_value(self.qdrant.url)
            if self.security.mask_sensitive_logs
            else self.qdrant.url,
            "embedding_model": self.embedding.model,
            "workspace_collection_types": self.workspace.collection_types,
            "hot_reload_enabled": self.development.hot_reload,
        }

        if self.qdrant.api_key:
            summary["qdrant_api_key_set"] = True
            if not self.security.mask_sensitive_logs:
                summary["qdrant_api_key"] = self.mask_sensitive_value(
                    self.qdrant.api_key
                )

        return summary


# Backward compatibility: alias to enhanced config
Config = EnhancedConfig
