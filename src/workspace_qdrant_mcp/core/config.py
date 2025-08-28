"""
Configuration management for workspace-qdrant-mcp.

Handles environment variables and configuration file loading.
"""

import os
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_sparse_vectors: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50


class QdrantConfig(BaseModel):
    """Configuration for Qdrant connection."""
    
    url: str = "http://localhost:6333"
    api_key: Optional[str] = None
    timeout: int = 30
    prefer_grpc: bool = False


class WorkspaceConfig(BaseModel):
    """Configuration for workspace management."""
    
    global_collections: List[str] = ["docs", "references", "standards"]
    github_user: Optional[str] = None
    collection_prefix: str = ""
    max_collections: int = 100


class Config(BaseSettings):
    """Main configuration class."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="WORKSPACE_QDRANT_",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Server configuration
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    
    # Component configurations
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    
    def __init__(self, **kwargs):
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
        if global_collections := os.getenv("WORKSPACE_QDRANT_WORKSPACE__GLOBAL_COLLECTIONS"):
            # Parse comma-separated list
            self.workspace.global_collections = [c.strip() for c in global_collections.split(",") if c.strip()]
        if github_user := os.getenv("WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER"):
            self.workspace.github_user = github_user
        if collection_prefix := os.getenv("WORKSPACE_QDRANT_WORKSPACE__COLLECTION_PREFIX"):
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
        if collections := os.getenv("GLOBAL_COLLECTIONS"):
            self.workspace.global_collections = [
                c.strip() for c in collections.split(",")
            ]
        if github_user := os.getenv("GITHUB_USER"):
            self.workspace.github_user = github_user
    
    @property
    def qdrant_client_config(self) -> dict:
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
        
        # Check required settings
        if not self.qdrant.url:
            issues.append("Qdrant URL is required")
            
        # Validate embedding settings
        if self.embedding.chunk_size <= 0:
            issues.append("Chunk size must be positive")
        if self.embedding.batch_size <= 0:
            issues.append("Batch size must be positive")
        if self.embedding.chunk_overlap < 0:
            issues.append("Chunk overlap must be non-negative")
        if self.embedding.chunk_overlap >= self.embedding.chunk_size:
            issues.append("Chunk overlap must be less than chunk size")
            
        # Validate workspace settings
        if not self.workspace.global_collections:
            issues.append("At least one global collection must be configured")
            
        return issues