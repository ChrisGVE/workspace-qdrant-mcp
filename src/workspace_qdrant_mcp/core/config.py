
"""
Comprehensive configuration management for workspace-qdrant-mcp.

This module provides a robust configuration system that handles environment variables,
configuration files, nested settings, and backward compatibility. It uses Pydantic
for type-safe configuration management with validation and automatic conversion.

Configuration Sources:
    1. Environment variables (highest priority)
    2. .env files in current directory
    3. Default values (lowest priority)

Supported Formats:
    - Prefixed environment variables: WORKSPACE_QDRANT_*
    - Nested configuration: WORKSPACE_QDRANT_QDRANT__URL
    - Legacy variables: QDRANT_URL, FASTEMBED_MODEL (backward compatibility)
    - Configuration files: .env with UTF-8 encoding

Configuration Hierarchy:
    - Server settings (host, port, debug mode)
    - Qdrant database connection (URL, API key, timeouts)
    - Embedding service (model, chunking, batch processing)
    - Workspace management (collections, GitHub integration)

Validation Features:
    - Type checking with Pydantic models
    - Range validation for numeric parameters
    - Required field validation
    - Logical consistency checks (e.g., chunk_overlap < chunk_size)
    - Connection parameter validation

Example:
    ```python
    from workspace_qdrant_mcp.core.config import Config

    # Load configuration from environment and .env file
    config = Config()

    # Access nested configuration
    logger.info("Qdrant URL: {config.qdrant.url}")
    logger.info("Embedding model: {config.embedding.model}")

    # Validate configuration
    issues = config.validate_config()
    if issues:
        logger.info("Configuration issues: {issues}")

    # Get Qdrant client configuration
    client_config = config.qdrant_client_config
    ```
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation and text processing.

    This class defines all parameters related to document embedding generation,
    including model selection, text chunking strategies, and batch processing
    configuration. It supports both dense semantic embeddings and sparse
    keyword vectors for optimal hybrid search performance.

    Attributes:
        model: FastEmbed model name for dense embeddings (default: all-MiniLM-L6-v2)
        enable_sparse_vectors: Whether to generate sparse BM25 vectors for hybrid search
        chunk_size: Maximum characters per text chunk (affects memory and quality)
        chunk_overlap: Characters to overlap between chunks (maintains context)
        batch_size: Number of documents to process simultaneously (affects memory)

    Performance Notes:
        - Optimal chunk_size (800) balances context and all-MiniLM-L6-v2 model efficiency
        - 15% overlap (120 chars) provides optimal boundary preservation
        - Higher batch_size improves throughput but requires more memory
        - Sparse vectors add ~30% processing time but significantly improve search quality
    """

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_sparse_vectors: bool = True
    chunk_size: int = 800
    chunk_overlap: int = 120
    batch_size: int = 50


class QdrantConfig(BaseModel):
    """Configuration for Qdrant vector database connection.

    Defines connection parameters, authentication, and performance settings
    for connecting to Qdrant vector database instances. Supports both local
    and cloud deployments with optional API key authentication.

    Attributes:
        url: Qdrant server endpoint URL (HTTP or HTTPS)
        api_key: Optional API key for authentication (required for Qdrant Cloud)
        timeout: Connection timeout in seconds for operations
        prefer_grpc: Whether to use gRPC protocol for better performance

    Connection Notes:
        - Local development typically uses http://localhost:6333
        - Cloud deployments require HTTPS URLs and API keys
        - gRPC provides better performance but HTTP is more compatible
        - Timeout should account for large batch operations
    """

    url: str = "http://localhost:6333"
    api_key: str | None = None
    timeout: int = 30
    prefer_grpc: bool = False


class WorkspaceConfig(BaseModel):
    """Configuration for workspace and project management.

    Defines workspace-level settings including global collections that span
    multiple projects, configurable project collections, GitHub integration
    for project detection, and collection organization preferences.

    Attributes:
        collections: Project collection suffixes (creates {project-name}-{suffix})
        global_collections: Collections available across all projects (user-defined)
        github_user: GitHub username for project ownership detection
        collection_prefix: Optional prefix for all collection names
        max_collections: Maximum number of collections per workspace (safety limit)
        auto_create_collections: Whether to automatically create project collections on startup

    Usage Patterns:
        - collections define project-specific collection types
        - global_collections enable cross-project knowledge sharing (user choice)
        - github_user enables intelligent project name detection
        - collection_prefix helps organize collections in shared Qdrant instances
        - max_collections prevents runaway collection creation
        - auto_create_collections controls whether collections are created automatically
        - when auto_create_collections=false, only scratchbook collection is created

    Examples:
        - collections=["project"] → creates {project-name}-project (if auto_create_collections=true)
        - collections=["docs", "tests"] → creates {project-name}-docs, {project-name}-tests (if auto_create_collections=true)
        - scratchbook collection is always created regardless of auto_create_collections setting
    """

    collections: list[str] = ["project"]
    global_collections: list[str] = ["scratchbook"]
    github_user: str | None = None
    collection_prefix: str = ""
    max_collections: int = 100
    auto_create_collections: bool = False


class GrpcConfig(BaseModel):
    """Configuration for gRPC communication with the Rust ingestion engine.

    Controls whether to use the high-performance Rust-based ingestion engine
    via gRPC for document processing, file watching, and search operations.
    Provides fallback options and connection management settings.

    Attributes:
        enabled: Whether to attempt gRPC connections to the Rust engine
        host: gRPC server host address (typically localhost for local engine)
        port: gRPC server port (default 50051 for gRPC convention)
        fallback_to_direct: Fall back to direct Qdrant access if gRPC fails
        connection_timeout: Timeout for establishing gRPC connections (seconds)
        max_retries: Maximum number of retry attempts for failed operations
        retry_backoff_multiplier: Exponential backoff multiplier for retries
        health_check_interval: Interval between background health checks (seconds)
        max_message_length: Maximum message size for gRPC operations (bytes)
        keepalive_time: Keep-alive ping interval (seconds)

    Usage Patterns:
        - enabled=true + fallback_to_direct=true: Hybrid mode (recommended)
        - enabled=true + fallback_to_direct=false: gRPC-only mode (high performance)
        - enabled=false: Direct mode only (Python-only operations)

    Performance Benefits:
        - File processing: ~2-5x faster than Python implementation
        - Large document ingestion: Significant memory efficiency improvements
        - Concurrent operations: Better resource utilization
        - File watching: Native filesystem event handling
    """

    enabled: bool = False
    host: str = "127.0.0.1" 
    port: int = 50051
    fallback_to_direct: bool = True
    connection_timeout: float = 10.0
    max_retries: int = 3
    retry_backoff_multiplier: float = 1.5
    health_check_interval: float = 30.0
    max_message_length: int = 100 * 1024 * 1024  # 100MB
    keepalive_time: int = 30


class AutoIngestionConfig(BaseModel):
    """Configuration for automatic file ingestion on server startup.

    This configuration controls the automatic detection and ingestion of project files
    when the MCP server starts up. It provides fine-grained control over which files
    are included, processing behavior, and performance characteristics.

    Attributes:
        enabled: Enable automatic ingestion on server startup (default: True)
        auto_create_watches: Automatically create project file watches (default: True)
        include_common_files: Include common document types like *.md, *.txt (default: True)
        include_source_files: Include source code files like *.py, *.js (default: False)
        max_files_per_batch: Maximum files to process simultaneously (default: 5)
        batch_delay_seconds: Delay between processing batches to prevent overload (default: 2.0)
        max_file_size_mb: Maximum file size to process in MB (default: 50)
        recursive_depth: Maximum directory recursion depth (default: 5)
        debounce_seconds: File change debounce time for watches (default: 10)

    Performance Notes:
        - Lower batch sizes reduce memory usage but increase processing time
        - Batch delays prevent overwhelming the embedding service and database
        - File size limits prevent processing of large binaries and media files
        - Recursive depth limits prevent scanning deep directory structures
    """

    enabled: bool = True
    auto_create_watches: bool = True
    include_common_files: bool = True
    include_source_files: bool = False
    max_files_per_batch: int = 5
    batch_delay_seconds: float = 2.0
    max_file_size_mb: int = 50
    recursive_depth: int = 5
    debounce_seconds: int = 10


class Config(BaseSettings):
    """Main configuration class with hierarchical settings management.

    This is the primary configuration class that combines all configuration
    domains (server, database, embedding, workspace) into a single, type-safe
    interface. It handles environment variable loading, nested configuration,
    backward compatibility, and validation.

    Features:
        - Automatic environment variable loading with WORKSPACE_QDRANT_ prefix
        - Nested configuration support (e.g., WORKSPACE_QDRANT_QDRANT__URL)
        - Legacy environment variable support for backward compatibility
        - Configuration file loading from .env files
        - Comprehensive validation with detailed error messages
        - Type safety with Pydantic models

    Environment Variable Patterns:
        - Primary: WORKSPACE_QDRANT_HOST, WORKSPACE_QDRANT_PORT
        - Nested: WORKSPACE_QDRANT_QDRANT__URL, WORKSPACE_QDRANT_EMBEDDING__MODEL
        - Legacy: QDRANT_URL, FASTEMBED_MODEL (backward compatibility)

    Example:
        ```bash
        # Set via environment
        export WORKSPACE_QDRANT_QDRANT__URL=https://my-qdrant.example.com
        export WORKSPACE_QDRANT_EMBEDDING__MODEL=sentence-transformers/all-MiniLM-L6-v2

        # Or via .env file
        WORKSPACE_QDRANT_QDRANT__URL=http://localhost:6333
        WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER=myusername
        ```
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WORKSPACE_QDRANT_",
        case_sensitive=False,
        extra="ignore",
    )

    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Component configurations
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    grpc: GrpcConfig = Field(default_factory=GrpcConfig)
    auto_ingestion: AutoIngestionConfig = Field(default_factory=AutoIngestionConfig)

    def __init__(self, config_file: Optional[str] = None, **kwargs) -> None:
        """Initialize configuration with YAML file, environment and legacy variable loading.

        Args:
            config_file: Path to YAML configuration file (takes precedence over env vars)
            **kwargs: Override values for configuration parameters
        """
        # Load YAML configuration first if provided
        yaml_config = {}
        if config_file:
            yaml_config = self._load_yaml_config(config_file)
        
        # Merge YAML config with kwargs, giving kwargs precedence
        merged_kwargs = {**yaml_config, **kwargs}
        
        super().__init__(**merged_kwargs)
        
        # Load environment variables (these have lower precedence than YAML)
        self._load_legacy_env_vars()
        self._load_nested_env_vars()
        
        # Override with YAML config again to ensure YAML takes precedence over env vars
        if yaml_config:
            self._apply_yaml_overrides(yaml_config)

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
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if not config_path.is_file():
            raise ValueError(f"Configuration path is not a file: {config_file}")
        
        try:
            with config_path.open('r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f)
            
            if yaml_data is None:
                return {}
            
            if not isinstance(yaml_data, dict):
                raise ValueError(f"YAML configuration must be a dictionary, got {type(yaml_data).__name__}")
            
            # Validate and flatten YAML structure for pydantic
            return self._process_yaml_structure(yaml_data)
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration file {config_file}: {e}") from e
        except Exception as e:
            raise ValueError(f"Error loading configuration file {config_file}: {e}") from e
    
    def _process_yaml_structure(self, yaml_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process YAML data structure to match Pydantic model structure.
        
        Args:
            yaml_data: Raw YAML data dictionary
            
        Returns:
            Processed configuration dictionary matching the model structure
        """
        processed = {}
        
        # Handle nested configuration sections
        for key, value in yaml_data.items():
            if key == 'qdrant' and isinstance(value, dict):
                processed['qdrant'] = QdrantConfig(**value)
            elif key == 'embedding' and isinstance(value, dict):
                processed['embedding'] = EmbeddingConfig(**value)
            elif key == 'workspace' and isinstance(value, dict):
                processed['workspace'] = WorkspaceConfig(**value)
            elif key in ['host', 'port', 'debug']:  # Server-level config
                processed[key] = value
            else:
                # Allow other keys to pass through
                processed[key] = value
        
        return processed
    
    def _apply_yaml_overrides(self, yaml_config: Dict[str, Any]) -> None:
        """Apply YAML configuration overrides after environment variables are loaded.
        
        Args:
            yaml_config: Processed YAML configuration dictionary
        """
        for key, value in yaml_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_yaml(cls, config_file: str, **kwargs) -> 'Config':
        """Create Config instance from YAML file.
        
        Args:
            config_file: Path to YAML configuration file
            **kwargs: Additional configuration overrides
            
        Returns:
            Config instance with YAML configuration loaded
            
        Example:
            ```python
            config = Config.from_yaml('config.yaml')
            ```
        """
        return cls(config_file=config_file, **kwargs)
    
    def to_yaml(self, file_path: Optional[str] = None) -> str:
        """Export current configuration to YAML format.
        
        Args:
            file_path: Optional path to save YAML file
            
        Returns:
            YAML string representation of the configuration
        """
        config_dict = {
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            'qdrant': {
                'url': self.qdrant.url,
                'api_key': self.qdrant.api_key,
                'timeout': self.qdrant.timeout,
                'prefer_grpc': self.qdrant.prefer_grpc,
            },
            'embedding': {
                'model': self.embedding.model,
                'enable_sparse_vectors': self.embedding.enable_sparse_vectors,
                'chunk_size': self.embedding.chunk_size,
                'chunk_overlap': self.embedding.chunk_overlap,
                'batch_size': self.embedding.batch_size,
            },
            'workspace': {
                'collections': self.workspace.collections,
                'global_collections': self.workspace.global_collections,
                'github_user': self.workspace.github_user,
                'collection_prefix': self.workspace.collection_prefix,
                'max_collections': self.workspace.max_collections,
                'auto_create_collections': self.workspace.auto_create_collections,
            }
        }
        
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if file_path:
            Path(file_path).write_text(yaml_str, encoding='utf-8')
        
        return yaml_str

    def _load_nested_env_vars(self) -> None:
        """Load nested configuration from environment variables with double underscore syntax."""

        # Qdrant nested config
        if url := os.getenv("WORKSPACE_QDRANT_QDRANT__URL"):
            self.qdrant.url = url
        if api_key := os.getenv("WORKSPACE_QDRANT_QDRANT__API_KEY"):
            self.qdrant.api_key = api_key
        if timeout := os.getenv("WORKSPACE_QDRANT_QDRANT__TIMEOUT"):
            self.qdrant.timeout = int(timeout)
        if prefer_grpc := os.getenv("WORKSPACE_QDRANT_QDRANT__PREFER_GRPC"):
            self.qdrant.prefer_grpc = prefer_grpc.lower() == "true"

        # Embedding nested config
        if model := os.getenv("WORKSPACE_QDRANT_EMBEDDING__MODEL"):
            self.embedding.model = model
        if sparse := os.getenv("WORKSPACE_QDRANT_EMBEDDING__ENABLE_SPARSE_VECTORS"):
            self.embedding.enable_sparse_vectors = sparse.lower() == "true"
        if chunk_size := os.getenv("WORKSPACE_QDRANT_EMBEDDING__CHUNK_SIZE"):
            self.embedding.chunk_size = int(chunk_size)
        if chunk_overlap := os.getenv("WORKSPACE_QDRANT_EMBEDDING__CHUNK_OVERLAP"):
            self.embedding.chunk_overlap = int(chunk_overlap)
        if batch_size := os.getenv("WORKSPACE_QDRANT_EMBEDDING__BATCH_SIZE"):
            self.embedding.batch_size = int(batch_size)

        # Workspace nested config
        if collections := os.getenv("WORKSPACE_QDRANT_WORKSPACE__COLLECTIONS"):
            # Parse comma-separated list
            self.workspace.collections = [
                c.strip() for c in collections.split(",") if c.strip()
            ]
        if global_collections := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS"
        ):
            # Parse comma-separated list
            self.workspace.global_collections = [
                c.strip() for c in global_collections.split(",") if c.strip()
            ]
        if github_user := os.getenv("WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER"):
            self.workspace.github_user = github_user
        if collection_prefix := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX"
        ):
            self.workspace.collection_prefix = collection_prefix
        if max_collections := os.getenv("WORKSPACE_QDRANT_WORKSPACE__MAX_COLLECTIONS"):
            self.workspace.max_collections = int(max_collections)
        if auto_create := os.getenv("WORKSPACE_QDRANT_WORKSPACE__AUTO_CREATE_COLLECTIONS"):
            self.workspace.auto_create_collections = auto_create.lower() == "true"

    def _load_legacy_env_vars(self) -> None:
        """Load legacy environment variables for backward compatibility."""

        # Legacy Qdrant config
        if url := os.getenv("QDRANT_URL"):
            self.qdrant.url = url
        if api_key := os.getenv("QDRANT_API_KEY"):
            self.qdrant.api_key = api_key

        # Legacy embedding config
        if model := os.getenv("FASTEMBED_MODEL"):
            self.embedding.model = model
        if sparse := os.getenv("ENABLE_SPARSE_VECTORS"):
            self.embedding.enable_sparse_vectors = sparse.lower() == "true"
        if chunk_size := os.getenv("CHUNK_SIZE"):
            self.embedding.chunk_size = int(chunk_size)
        if chunk_overlap := os.getenv("CHUNK_OVERLAP"):
            self.embedding.chunk_overlap = int(chunk_overlap)
        if batch_size := os.getenv("BATCH_SIZE"):
            self.embedding.batch_size = int(batch_size)

        # Legacy workspace config
        if collections := os.getenv("COLLECTIONS"):
            # Support both legacy COLLECTIONS and new COLLECTIONS env var
            self.workspace.collections = [
                c.strip() for c in collections.split(",") if c.strip()
            ]
        if global_collections := os.getenv("GLOBAL_COLLECTIONS"):
            self.workspace.global_collections = [
                c.strip() for c in global_collections.split(",") if c.strip()
            ]
        if github_user := os.getenv("GITHUB_USER"):
            self.workspace.github_user = github_user

    @property
    def qdrant_client_config(self) -> dict:
        """Get Qdrant client configuration dictionary for QdrantClient initialization.

        Converts the internal Qdrant configuration to the format expected by
        the QdrantClient constructor, including optional parameters only when
        they are set.

        Returns:
            dict: Configuration dictionary with keys:
                - url (str): Qdrant server endpoint
                - timeout (int): Request timeout in seconds
                - prefer_grpc (bool): Protocol preference
                - api_key (str, optional): Authentication key if configured

        Example:
            ```python
            config = Config()
            client = QdrantClient(**config.qdrant_client_config)
            ```
        """
        config = {
            "url": self.qdrant.url,
            "timeout": self.qdrant.timeout,
            "prefer_grpc": self.qdrant.prefer_grpc,
        }

        if self.qdrant.api_key:
            config["api_key"] = self.qdrant.api_key

        return config

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of issues.

        Performs comprehensive validation of all configuration parameters,
        checking for required values, valid ranges, and logical consistency.
        Returns a list of human-readable error messages for any issues found.

        Validation Checks:
            - Required fields are present and non-empty
            - Numeric values are within valid ranges
            - Logical consistency between related parameters
            - URL format validation for endpoints
            - Model name format validation

        Returns:
            List[str]: List of validation error messages. Empty list indicates
                      valid configuration.

        Example:
            ```python
            config = Config()
            issues = config.validate_config()
            if issues:
                logger.info("Configuration errors:")
                for issue in issues:
                    logger.info("  - {issue}")
                raise ConfigurationError("Configuration validation failed")
            ```
        """
        issues = []

        # Check required settings
        if not self.qdrant.url:
            issues.append("Qdrant URL is required")
        elif not (
            self.qdrant.url.startswith("http://")
            or self.qdrant.url.startswith("https://")
        ):
            issues.append("Qdrant URL must start with http:// or https://")

        # Validate embedding settings
        if self.embedding.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        elif self.embedding.chunk_size > 10000:
            issues.append(
                "Chunk size should not exceed 10000 characters for optimal performance"
            )

        if self.embedding.batch_size <= 0:
            issues.append("Batch size must be positive")
        elif self.embedding.batch_size > 1000:
            issues.append("Batch size should not exceed 1000 for memory efficiency")
        if self.embedding.chunk_overlap < 0:
            issues.append("Chunk overlap must be non-negative")
        if self.embedding.chunk_overlap >= self.embedding.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")

        # Validate workspace settings
        if not self.workspace.collections:
            issues.append("At least one project collection must be configured")
        elif len(self.workspace.collections) > 20:
            issues.append(
                "Too many project collections configured (max 20 recommended)"
            )

        if not self.workspace.global_collections:
            issues.append("At least one global collection must be configured")
        elif len(self.workspace.global_collections) > 50:
            issues.append("Too many global collections configured (max 50 recommended)")

        if self.workspace.max_collections <= 0:
            issues.append("Max collections must be positive")
        elif self.workspace.max_collections > 10000:
            issues.append("Max collections limit is too high (max 10000 recommended)")

        return issues
