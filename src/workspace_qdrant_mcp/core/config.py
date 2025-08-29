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
    print(f"Qdrant URL: {config.qdrant.url}")
    print(f"Embedding model: {config.embedding.model}")

    # Validate configuration
    issues = config.validate_config()
    if issues:
        print(f"Configuration issues: {issues}")

    # Get Qdrant client configuration
    client_config = config.qdrant_client_config
    ```
"""

import os
from typing import Optional

from pydantic import BaseModel, Field
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
        - Larger chunk_size improves context but increases memory usage
        - Higher batch_size improves throughput but requires more memory
        - Sparse vectors add ~30% processing time but significantly improve search quality
    """

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_sparse_vectors: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
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
        global_collections: Collections available across all projects (e.g., 'scratchbook')
        github_user: GitHub username for project ownership detection
        collection_prefix: Optional prefix for all collection names
        max_collections: Maximum number of collections per workspace (safety limit)

    Usage Patterns:
        - collections define project-specific collection types
        - global_collections enable cross-project knowledge sharing
        - github_user enables intelligent project name detection
        - collection_prefix helps organize collections in shared Qdrant instances
        - max_collections prevents runaway collection creation

    Examples:
        - collections=["project"] → creates {project-name}-project
        - collections=["scratchbook", "docs"] → creates {project-name}-scratchbook, {project-name}-docs
    """

    collections: list[str] = ["project"]
    global_collections: list[str] = ["docs", "references", "standards"]
    github_user: str | None = None
    collection_prefix: str = ""
    max_collections: int = 100


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

    def __init__(self, **kwargs) -> None:
        """Initialize configuration with environment and legacy variable loading.

        Args:
            **kwargs: Override values for configuration parameters
        """
        super().__init__(**kwargs)
        self._load_legacy_env_vars()
        self._load_nested_env_vars()

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
                print("Configuration errors:")
                for issue in issues:
                    print(f"  - {issue}")
                sys.exit(1)
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
