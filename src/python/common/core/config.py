"""
Comprehensive configuration management for workspace-qdrant-mcp.

This module provides a robust configuration system that handles environment variables,
configuration files, nested settings, and backward compatibility. It uses Pydantic
for type-safe configuration management with validation and automatic conversion.

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
from typing import Any, Optional

import yaml

# Task 215: Use unified logging system for MCP stdio compliance
from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    for project detection, collection organization preferences, and optional
    custom pattern extensions.

    Attributes:
        collection_types: Collection types for multi-tenant architecture (e.g., 'docs', 'notes', 'scratchbook')
        global_collections: Collections available across all projects (user-defined)
        github_user: GitHub username for project ownership detection
        auto_create_collections: Whether to automatically create project collections on startup
        memory_collection_name: Name for the system memory collection with '__' prefix support (default: '__memory')
        code_collection_name: Name for the system code collection with '__' prefix support (default: '__code')
        custom_include_patterns: Optional custom file patterns to include beyond hardcoded patterns
        custom_exclude_patterns: Optional custom file patterns to exclude beyond hardcoded patterns
        custom_project_indicators: Optional custom project indicators to extend hardcoded detection

    Usage Patterns:
        - collection_types define workspace collection types with multi-tenant metadata filtering
        - global_collections enable cross-project knowledge sharing (user choice)
        - github_user enables intelligent project name detection
        - auto_create_collections controls whether collections are created automatically
        - when auto_create_collections=false, no collections are created automatically
        - custom patterns extend the built-in PatternManager patterns without replacing them

    Examples:
        - collection_types=["docs", "notes"] → creates multi-tenant collections 'docs', 'notes' (if auto_create_collections=true)
        - Project isolation is achieved via metadata filtering rather than separate collections
        - collections are only created when explicitly configured by user
        - custom_include_patterns=["**/*.myext"] → includes custom file extensions beyond standard patterns
        - custom_project_indicators={"my_indicator": {"pattern": "my_file", "confidence": 0.8}}
    """

    collection_types: list[str] = []
    global_collections: list[str] = []
    github_user: str | None = None
    auto_create_collections: bool = False
    memory_collection_name: str = "__memory"
    code_collection_name: str = "__code"

    # Custom pattern extensions for PatternManager
    custom_include_patterns: list[str] = []
    custom_exclude_patterns: list[str] = []
    custom_project_indicators: dict[str, Any] = {}

    @property
    def effective_collection_types(self) -> list[str]:
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
        target_collection_suffix: Which collection suffix to use for auto-ingestion (must be one of collection_types)
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
    target_collection_suffix: str = ""  # Which collection suffix to use for auto-ingestion when multiple are available
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
                        If None, will automatically search for default config files
            **kwargs: Override values for configuration parameters
        """
        # Load YAML configuration - either explicitly provided or automatically discovered
        yaml_config = {}
        if config_file:
            yaml_config = self._load_yaml_config(config_file)
        else:
            # Auto-discover configuration file
            auto_config_file = self._find_default_config_file()
            if auto_config_file:
                yaml_config = self._load_yaml_config(auto_config_file)

        # Merge YAML config with kwargs, giving kwargs precedence
        merged_kwargs = {**yaml_config, **kwargs}

        super().__init__(**merged_kwargs)

        # Load environment variables (these have lower precedence than YAML)
        self._load_legacy_env_vars()
        self._load_nested_env_vars()

        # Override with YAML config again to ensure YAML takes precedence over env vars
        if yaml_config:
            self._apply_yaml_overrides(yaml_config)

    def _load_yaml_config(self, config_file: str) -> dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_file: Path to YAML configuration file

        Returns:
            dict containing the parsed YAML configuration

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
            with config_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            if yaml_data is None:
                return {}

            if not isinstance(yaml_data, dict):
                raise ValueError(
                    f"YAML configuration must be a dictionary, got {type(yaml_data).__name__}"
                )

            # Validate and flatten YAML structure for pydantic
            return self._process_yaml_structure(yaml_data)

        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML configuration file {config_file}: {e}"
            ) from e
        except Exception as e:
            raise ValueError(
                f"Error loading configuration file {config_file}: {e}"
            ) from e

    def _process_yaml_structure(self, yaml_data: dict[str, Any]) -> dict[str, Any]:
        """Process YAML data structure to match Pydantic model structure.

        Args:
            yaml_data: Raw YAML data dictionary

        Returns:
            dict: Processed configuration dictionary matching the model structure
        """
        processed = {}

        # Handle nested configuration sections
        for key, value in yaml_data.items():
            if key == "qdrant" and isinstance(value, dict):
                # Filter qdrant config to only include server-compatible fields
                filtered_qdrant = self._filter_qdrant_config(value)
                processed["qdrant"] = QdrantConfig(**filtered_qdrant)
            elif key == "embedding" and isinstance(value, dict):
                processed["embedding"] = EmbeddingConfig(**value)
            elif key == "workspace" and isinstance(value, dict):
                processed["workspace"] = WorkspaceConfig(**value)
            elif key == "auto_ingestion" and isinstance(value, dict):
                # Filter auto_ingestion to only include server-compatible fields
                filtered_auto_ingestion = self._filter_auto_ingestion_config(value)
                processed["auto_ingestion"] = AutoIngestionConfig(**filtered_auto_ingestion)
            elif key == "grpc" and isinstance(value, dict):
                processed["grpc"] = GrpcConfig(**value)
            elif key in ["host", "port", "debug"]:  # Server-level config
                processed[key] = value
            else:
                # Allow other keys to pass through silently (daemon-specific config)
                # These include: log_file, use_file_logging, max_concurrent_tasks,
                # default_timeout_ms, enable_preemption, chunk_size, enable_lsp,
                # log_level, enable_metrics, metrics_interval_secs, logging, etc.
                pass

        return processed

    def _find_default_config_file(self) -> Optional[str]:
        """Find default configuration file using XDG Base Directory Specification.

        Search order:
        1. XDG-compliant directories + /config.yaml
        2. XDG-compliant directories + /workspace_qdrant_config.yaml (backward compatibility)
        3. Project-specific configs in current directory
        4. Legacy fallback locations

        XDG Base Directory Specification:
        - Use $XDG_CONFIG_HOME/workspace-qdrant/ if XDG_CONFIG_HOME is set
        - Otherwise use OS-specific config directories:
          * macOS: ~/Library/Application Support/workspace-qdrant/
          * Linux/Unix: ~/.config/workspace-qdrant/
          * Windows: %APPDATA%/workspace-qdrant/

        Returns:
            Path to the first found config file, or None if no config file found
        """
        # logger imported from loguru

        # Get XDG-compliant config directories
        xdg_config_dirs = self._get_xdg_config_dirs()

        # 1. Check XDG-compliant directories with standard config.yaml
        for config_dir in xdg_config_dirs:
            config_path = config_dir / "config.yaml"
            if config_path.exists() and config_path.is_file():
                logger.info(f"Auto-discovered XDG configuration file: {config_path}")
                return str(config_path)

        # 2. Check XDG-compliant directories with legacy name (backward compatibility)
        for config_dir in xdg_config_dirs:
            config_path = config_dir / "workspace_qdrant_config.yaml"
            if config_path.exists() and config_path.is_file():
                logger.info(f"Auto-discovered XDG legacy configuration file: {config_path}")
                return str(config_path)

        # 3. Check current directory for project-specific configs
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

    def _get_xdg_config_dirs(self) -> list[Path]:
        """Get XDG-compliant configuration directories for workspace-qdrant.

        Follows XDG Base Directory Specification:
        - Checks XDG_CONFIG_HOME environment variable first
        - Falls back to OS-specific default directories if not set

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
                # Use APPDATA on Windows
                appdata = os.environ.get('APPDATA')
                if appdata:
                    config_dirs.append(Path(appdata) / 'workspace-qdrant')
                else:
                    # Fallback if APPDATA is not set
                    config_dirs.append(home_dir / 'AppData' / 'Roaming' / 'workspace-qdrant')
            else:  # Linux/Unix and other Unix-like systems
                config_dirs.append(home_dir / '.config' / 'workspace-qdrant')

        return config_dirs

    def _apply_yaml_overrides(self, yaml_config: dict[str, Any]) -> None:
        """Apply YAML configuration overrides after environment variables are loaded.

        Args:
            yaml_config: Processed YAML configuration dictionary
        """
        for key, value in yaml_config.items():
            if hasattr(self, key):
                # Check if we're trying to overwrite a typed config object with a raw dict
                current_value = getattr(self, key)

                # If current value is already a properly typed config object (like AutoIngestionConfig)
                # and the incoming value is a raw dict, skip the override to preserve type safety
                if (isinstance(value, dict) and
                    hasattr(current_value, '__class__') and
                    current_value.__class__.__name__.endswith('Config')):
                    continue

                setattr(self, key, value)

    @classmethod
    def from_yaml(cls, config_file: str, **kwargs) -> "Config":
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
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "qdrant": {
                "url": self.qdrant.url,
                "api_key": self.qdrant.api_key,
                "timeout": self.qdrant.timeout,
                "prefer_grpc": self.qdrant.prefer_grpc,
            },
            "embedding": {
                "model": self.embedding.model,
                "enable_sparse_vectors": self.embedding.enable_sparse_vectors,
                "chunk_size": self.embedding.chunk_size,
                "chunk_overlap": self.embedding.chunk_overlap,
                "batch_size": self.embedding.batch_size,
            },
            "workspace": {
                "collection_types": self.workspace.collection_types,
                "global_collections": self.workspace.global_collections,
                "github_user": self.workspace.github_user,
                "auto_create_collections": self.workspace.auto_create_collections,
                "memory_collection_name": self.workspace.memory_collection_name,
                "code_collection_name": self.workspace.code_collection_name,
                "custom_include_patterns": self.workspace.custom_include_patterns,
                "custom_exclude_patterns": self.workspace.custom_exclude_patterns,
                "custom_project_indicators": self.workspace.custom_project_indicators,
            },
        }

        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        if file_path:
            Path(file_path).write_text(yaml_str, encoding="utf-8")

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
        if collection_types := os.getenv("WORKSPACE_QDRANT_WORKSPACE__COLLECTION_TYPES"):
            # Parse comma-separated list
            self.workspace.collection_types = [
                c.strip() for c in collection_types.split(",") if c.strip()
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
        if auto_create := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__AUTO_CREATE_COLLECTIONS"
        ):
            self.workspace.auto_create_collections = auto_create.lower() == "true"
        if memory_collection_name := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__MEMORY_COLLECTION_NAME"
        ):
            self.workspace.memory_collection_name = memory_collection_name
        if code_collection_name := os.getenv(
            "WORKSPACE_QDRANT_WORKSPACE__CODE_COLLECTION_NAME"
        ):
            self.workspace.code_collection_name = code_collection_name

        # Custom pattern extension fields
        if custom_include := os.getenv("WORKSPACE_QDRANT_WORKSPACE__CUSTOM_INCLUDE_PATTERNS"):
            # Parse comma-separated list
            self.workspace.custom_include_patterns = [
                p.strip() for p in custom_include.split(",") if p.strip()
            ]
        if custom_exclude := os.getenv("WORKSPACE_QDRANT_WORKSPACE__CUSTOM_EXCLUDE_PATTERNS"):
            # Parse comma-separated list
            self.workspace.custom_exclude_patterns = [
                p.strip() for p in custom_exclude.split(",") if p.strip()
            ]
        # Note: custom_project_indicators is complex (dict) - better set via YAML config

        # Auto-ingestion nested config
        if enabled := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__ENABLED"):
            self.auto_ingestion.enabled = enabled.lower() == "true"
        if auto_create_watches := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__AUTO_CREATE_WATCHES"):
            self.auto_ingestion.auto_create_watches = auto_create_watches.lower() == "true"
        if include_common := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__INCLUDE_COMMON_FILES"):
            self.auto_ingestion.include_common_files = include_common.lower() == "true"
        if include_source := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__INCLUDE_SOURCE_FILES"):
            self.auto_ingestion.include_source_files = include_source.lower() == "true"
        if target_suffix := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__TARGET_COLLECTION_SUFFIX"):
            self.auto_ingestion.target_collection_suffix = target_suffix
        if max_files := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__MAX_FILES_PER_BATCH"):
            self.auto_ingestion.max_files_per_batch = int(max_files)
        if batch_delay := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__BATCH_DELAY_SECONDS"):
            self.auto_ingestion.batch_delay_seconds = float(batch_delay)
        if max_size := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__MAX_FILE_SIZE_MB"):
            self.auto_ingestion.max_file_size_mb = int(max_size)
        if recursive_depth := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__RECURSIVE_DEPTH"):
            self.auto_ingestion.recursive_depth = int(recursive_depth)
        if debounce := os.getenv("WORKSPACE_QDRANT_AUTO_INGESTION__DEBOUNCE_SECONDS"):
            self.auto_ingestion.debounce_seconds = int(debounce)

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
        if collection_types := os.getenv("COLLECTION_TYPES"):
            self.workspace.collection_types = [
                c.strip() for c in collection_types.split(",") if c.strip()
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
        effective_types = self.workspace.effective_collection_types
        if len(effective_types) > 20:
            issues.append(
                "Too many collection types configured (max 20 recommended)"
            )

        if len(self.workspace.global_collections) > 50:
            issues.append("Too many global collections configured (max 50 recommended)")

        # Validate custom pattern extensions
        if len(self.workspace.custom_include_patterns) > 100:
            issues.append("Too many custom include patterns configured (max 100 recommended)")

        if len(self.workspace.custom_exclude_patterns) > 100:
            issues.append("Too many custom exclude patterns configured (max 100 recommended)")

        if len(self.workspace.custom_project_indicators) > 20:
            issues.append("Too many custom project indicators configured (max 20 recommended)")

        # Validate custom project indicator structure
        for indicator_name, indicator_config in self.workspace.custom_project_indicators.items():
            if not isinstance(indicator_config, dict):
                issues.append(f"Custom project indicator '{indicator_name}' must be a dictionary")
                continue

            # Check for required fields in project indicator
            if "pattern" not in indicator_config:
                issues.append(f"Custom project indicator '{indicator_name}' missing required 'pattern' field")

            # Validate confidence if present
            if "confidence" in indicator_config:
                confidence = indicator_config["confidence"]
                if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                    issues.append(f"Custom project indicator '{indicator_name}' confidence must be a number between 0.0 and 1.0")

        # Note: max_collections validation removed as part of multi-tenant architecture

        # Validate auto-ingestion configuration with graceful fallback behavior
        if self.auto_ingestion.enabled:
            target_suffix = self.auto_ingestion.target_collection_suffix
            available_types = self.workspace.effective_collection_types
            auto_create = self.workspace.auto_create_collections

            # Check if target_collection_suffix is specified and valid
            if target_suffix:
                if available_types and target_suffix not in available_types:
                    # Only warn if auto_create is disabled, otherwise it will create the collection
                    if not auto_create:
                        issues.append(
                            f"auto_ingestion.target_collection_suffix '{target_suffix}' "
                            f"is not in workspace.collection_types {available_types}. "
                            f"Consider adding '{target_suffix}' to collection_types or enabling auto_create_collections."
                        )
            elif not target_suffix and available_types:
                # This is a clear misconfiguration - user has types but didn't specify which to use
                issues.append(
                    "auto_ingestion.target_collection_suffix is empty but workspace.collection_types "
                    f"contains {available_types}. Please specify which type to use for auto-ingestion."
                )
            elif not target_suffix and not available_types and not auto_create:
                # This case now gets graceful fallback - we'll use a default collection name
                # No longer an error, but log a warning-level message for the user to be aware
                # logger imported from loguru
                logger.warning(
                    "Auto-ingestion enabled without explicit collection configuration. "
                    "Will use default project-based collection naming. "
                    "For better control, consider setting target_collection_suffix or enabling auto_create_collections."
                )

        return issues

    def get_auto_ingestion_diagnostics(self) -> dict[str, Any]:
        """Get diagnostic information about auto-ingestion configuration.

        Returns detailed information about auto-ingestion configuration status,
        collection availability, and potential configuration issues.

        Returns:
            Dict containing diagnostic information:
                - enabled: Whether auto-ingestion is enabled
                - target_suffix: Configured target collection suffix
                - available_types: Available collection types
                - auto_create: Whether auto collection creation is enabled
                - configuration_status: Overall configuration status
                - recommendations: List of recommendations to fix issues
        """
        target_suffix = self.auto_ingestion.target_collection_suffix
        available_types = self.workspace.effective_collection_types
        auto_create = self.workspace.auto_create_collections

        # Determine configuration status
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

    def _get_auto_ingestion_summary(self, status: str, target_suffix: str, available_types: list[str]) -> str:
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

        Returns a human-readable description of what will happen when auto-ingestion
        runs, including fallback behavior and collection selection logic.
        """
        if not self.auto_ingestion.enabled:
            return "Auto-ingestion is disabled. No automatic file processing will occur."

        target_suffix = self.auto_ingestion.target_collection_suffix
        available_types = self.workspace.effective_collection_types
        auto_create = self.workspace.auto_create_collections

        if target_suffix and available_types and target_suffix in available_types:
            return f"Will use collection '{{{self._current_project_name()}}}-{target_suffix}' for auto-ingestion."
        elif target_suffix and auto_create:
            return f"Will create and use collection '{{{self._current_project_name()}}}-{target_suffix}' for auto-ingestion."
        elif not target_suffix:
            behavior_parts = [
                "Will use intelligent fallback selection:",
                "1. Existing project collections (if any)",
                "2. Common collections like 'scratchbook' (if they exist)",
                "3. Create a default collection if no suitable collections exist"
            ]
            return " ".join(behavior_parts) if len(" ".join(behavior_parts)) < 100 else "\n  ".join(behavior_parts)
        else:
            return f"Configuration may need adjustment. Target suffix '{target_suffix}' specified but not in available types."

    def _current_project_name(self) -> str:
        """Get current project name for display purposes."""
        try:
            import os
            from ..utils.project_detection import ProjectDetector
            project_detector = ProjectDetector()
            project_info = project_detector.get_project_info(os.getcwd())
            return project_info.get("main_project", "current-project")
        except Exception:
            return "current-project"

    def _filter_qdrant_config(self, qdrant_config: dict[str, Any]) -> dict[str, Any]:
        """Filter Qdrant config to only include server-compatible fields.

        Args:
            qdrant_config: Raw Qdrant configuration from YAML

        Returns:
            dict: Filtered Qdrant configuration with only server-compatible fields
        """
        # Map daemon config fields to server config fields
        field_mapping = {
            "url": "url",
            "api_key": "api_key",
            "timeout_ms": "timeout",  # Convert milliseconds to seconds
            "prefer_grpc": "prefer_grpc"
        }

        filtered = {}
        for daemon_field, server_field in field_mapping.items():
            if daemon_field in qdrant_config:
                value = qdrant_config[daemon_field]
                if daemon_field == "timeout_ms":
                    # Convert milliseconds to seconds for server config
                    value = int(value / 1000)
                filtered[server_field] = value

        # Set prefer_grpc based on transport type if not explicitly set
        if "prefer_grpc" not in filtered and "transport" in qdrant_config:
            transport = qdrant_config["transport"].lower()
            filtered["prefer_grpc"] = transport == "grpc"

        return filtered

    def _filter_auto_ingestion_config(self, auto_ingestion_config: dict[str, Any]) -> dict[str, Any]:
        """Filter auto-ingestion config to only include server-compatible fields.

        Args:
            auto_ingestion_config: Raw auto-ingestion configuration from YAML

        Returns:
            dict: Filtered auto-ingestion configuration with only server-compatible fields
        """
        # These fields are compatible between daemon and server
        compatible_fields = [
            "enabled",
            "auto_create_watches",
            "include_common_files",
            "include_source_files",
            "target_collection_suffix",
            "max_files_per_batch",
            "batch_delay_seconds",
            "max_file_size_mb",
            "recursive_depth",
            "debounce_seconds"
        ]

        filtered = {}
        for field in compatible_fields:
            if field in auto_ingestion_config:
                filtered[field] = auto_ingestion_config[field]

        return filtered