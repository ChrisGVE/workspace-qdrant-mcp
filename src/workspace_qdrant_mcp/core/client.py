"""
Qdrant workspace client for project-scoped collections.

This module provides the main client class for managing project-aware Qdrant
vector database operations. It handles automatic project detection, workspace-scoped
collection management, embedding service integration, and provides a unified
interface for all vector database operations.

Key Components:
    - Automatic project structure detection from Git repositories
    - Workspace-scoped collection management with isolation
    - Integrated embedding service with dense and sparse vector support
    - Connection pooling and async operation support
    - Comprehensive error handling and logging

The client automatically creates collections based on detected project structure:
    - Main project collection (e.g., 'my-project')
    - Subproject collections (e.g., 'my-project.frontend', 'my-project.backend')
    - Global collections ('scratchbook' for cross-project notes)

Example:
    ```python
    from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
    from workspace_qdrant_mcp.core.config import Config

    config = Config()
    client = QdrantWorkspaceClient(config)
    await client.initialize()

    # Client automatically detects project and creates collections
    collections = client.list_collections()
    status = await client.get_status()
    ```
"""

import asyncio
import logging
from typing import Optional

from qdrant_client import QdrantClient

from ..observability import get_logger
from ..utils.project_detection import ProjectDetector
from .collections import WorkspaceCollectionManager, MemoryCollectionManager
from .config import Config
from .embeddings import EmbeddingService
from .ssl_config import create_secure_qdrant_config, get_ssl_manager

# Import LLM access control system
try:
    from .llm_access_control import validate_llm_collection_access, LLMAccessControlError
except ImportError:
    # Fallback for direct imports when not used as a package
    from llm_access_control import validate_llm_collection_access, LLMAccessControlError

logger = get_logger(__name__)


class QdrantWorkspaceClient:
    """
    Main client for workspace-scoped Qdrant vector database operations.

    This class provides a high-level interface for project-aware vector database
    operations, including automatic project detection, collection management,
    embedding generation, and search capabilities. It maintains workspace isolation
    while providing seamless access to global collections like scratchbook.

    The client handles:
        - Project structure detection from Git repositories and directory structure
        - Automatic collection creation based on detected projects and subprojects
        - Embedding service management (dense + sparse vectors)
        - Connection lifecycle and error recovery
        - Workspace-scoped operations with proper isolation

    Attributes:
        config (Config): Configuration settings for the client
        client (Optional[QdrantClient]): Underlying Qdrant client instance
        collection_manager (Optional[WorkspaceCollectionManager]): Collection management
        embedding_service (EmbeddingService): Embedding generation service
        project_detector (ProjectDetector): Project structure detection
        project_info (Optional[Dict]): Detected project information
        initialized (bool): Whether the client has been initialized

    Example:
        ```python
        config = Config()
        workspace_client = QdrantWorkspaceClient(config)

        # Initialize connections and detect project structure
        await workspace_client.initialize()

        # Use the client for operations
        status = await workspace_client.get_status()
        collections = workspace_client.list_collections()

        # Clean up when done
        await workspace_client.close()
        ```
    """

    def __init__(self, config: Config) -> None:
        """Initialize the workspace client with configuration.

        Args:
            config: Configuration object containing Qdrant connection settings,
                   embedding model configuration, and workspace preferences
        """
        self.config = config
        self.client: QdrantClient | None = None
        self.collection_manager: WorkspaceCollectionManager | None = None
        self.memory_collection_manager: MemoryCollectionManager | None = None
        self.embedding_service = EmbeddingService(config)
        self.project_detector = ProjectDetector(
            github_user=self.config.workspace.github_user
        )
        self.project_info: dict | None = None
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize the Qdrant client and workspace collections.

        Performs complete workspace initialization including:
        1. Qdrant database connection establishment and testing
        2. Project structure detection from current directory
        3. Collection manager initialization with workspace isolation
        4. Embedding service setup (dense + sparse models)
        5. Automatic collection creation for detected projects

        This method is idempotent and can be safely called multiple times.

        Raises:
            ConnectionError: If Qdrant database is unreachable
            RuntimeError: If project detection or collection setup fails
            ModelError: If embedding models cannot be loaded

        Example:
            ```python
            client = QdrantWorkspaceClient(config)
            await client.initialize()  # Safe to call multiple times
            ```
        """
        if self.initialized:
            return

        try:
            # Create secure Qdrant client configuration with context-aware SSL handling
            ssl_manager = get_ssl_manager()

            # Determine environment from config or fall back to development
            environment = getattr(self.config, "environment", "development")

            # Get authentication credentials from config if available
            auth_token = (
                getattr(self.config.security, "qdrant_auth_token", None)
                if hasattr(self.config, "security")
                else None
            )
            api_key = (
                getattr(self.config.security, "qdrant_api_key", None)
                if hasattr(self.config, "security")
                else None
            )

            # Create secure client configuration
            secure_config = create_secure_qdrant_config(
                base_config=self.config.qdrant_client_config,
                url=self.config.qdrant.url,
                environment=environment,
                auth_token=auth_token,
                api_key=api_key,
            )

            # Use SSL context manager for localhost connections in development
            if (
                ssl_manager.is_localhost_url(self.config.qdrant.url)
                and environment == "development"
            ):
                with ssl_manager.for_localhost():
                    self.client = QdrantClient(**secure_config)
            else:
                self.client = QdrantClient(**secure_config)

            # Test connection with appropriate SSL context
            if (
                ssl_manager.is_localhost_url(self.config.qdrant.url)
                and environment == "development"
            ):
                with ssl_manager.for_localhost():
                    await asyncio.get_event_loop().run_in_executor(
                        None, self.client.get_collections
                    )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.client.get_collections
                )

            logger.info("Connected to Qdrant at %s", self.config.qdrant.url)

            # Detect current project and subprojects
            self.project_info = self.project_detector.get_project_info()
            logger.info(
                "Detected project: %s with subprojects: %s",
                self.project_info["main_project"],
                self.project_info["subprojects"],
            )

            # Initialize collection manager
            self.collection_manager = WorkspaceCollectionManager(
                self.client, self.config
            )
            
            # Initialize memory collection manager
            self.memory_collection_manager = MemoryCollectionManager(
                self.client, self.config
            )

            # Initialize embedding service
            await self.embedding_service.initialize()
            logger.info("Embedding service initialized")

            # Initialize workspace collections with detected project info
            await self.collection_manager.initialize_workspace_collections(
                project_name=self.project_info["main_project"],
                subprojects=self.project_info["subprojects"],
            )
            
            # Ensure memory collections exist for the main project
            if self.project_info["main_project"]:
                memory_results = await self.memory_collection_manager.ensure_memory_collections_exist(
                    project=self.project_info["main_project"]
                )
                logger.info(f"Memory collection setup: {memory_results}")
            else:
                logger.warning("No project detected for memory collection initialization")

            self.initialized = True

        except Exception as e:
            logger.error("Failed to initialize Qdrant client: %s", e)
            raise

    async def get_status(self) -> dict:
        """Get comprehensive workspace and collection status information.

        Provides detailed diagnostics about the current workspace state including
        database connectivity, project detection results, collection statistics,
        embedding model status, and configuration parameters.

        Returns:
            Dict: Status information containing:
                - connected (bool): Qdrant database connection status
                - qdrant_url (str): Configured Qdrant endpoint URL
                - collections_count (int): Total number of collections in database
                - workspace_collections (List[str]): Project-specific collection names
                - current_project (str): Main project name from detection
                - project_info (dict): Complete project detection results
                - collection_info (dict): Per-collection statistics and metadata
                - embedding_info (dict): Model information and capabilities
                - config (dict): Active configuration parameters
                - error (str): Error message if client not initialized

        Example:
            ```python
            status = await client.get_status()
            if status.get('connected'):
                logger.info("Client status",
                          project=status['current_project'],
                          collections=status['workspace_collections'])
            ```
        """
        if not self.initialized:
            return {"error": "Client not initialized"}

        try:
            # Get basic Qdrant info
            info = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )

            workspace_collections = (
                self.collection_manager.list_workspace_collections()
            )
            collection_info = self.collection_manager.get_collection_info()

            return {
                "connected": True,
                "qdrant_url": self.config.qdrant.url,
                "collections_count": len(info.collections),
                "workspace_collections": workspace_collections,
                "current_project": self.project_info["main_project"]
                if self.project_info
                else None,
                "project_info": self.project_info,
                "collection_info": collection_info,
                "embedding_info": self.embedding_service.get_model_info(),
                "config": {
                    "embedding_model": self.config.embedding.model,
                    "sparse_vectors_enabled": self.config.embedding.enable_sparse_vectors,
                    "global_collections": self.config.workspace.global_collections,
                },
            }

        except Exception as e:
            logger.error("Failed to get status: %s", e)
            return {"error": f"Failed to get status: {e}"}

    def list_collections(self) -> list[str]:
        """List all available workspace collections for the current project.

        Returns collections that are accessible within the current workspace scope,
        including project-specific collections and global collections like 'scratchbook'.

        Returns:
            List[str]: Collection names available in the current workspace.
                Returns empty list if client not initialized or on error.

        Example:
            ```python
            collections = client.list_collections()
            for collection in collections:
                logger.info("Available collection", collection=collection)
            ```
        """
        if not self.initialized:
            return []

        try:
            return self.collection_manager.list_workspace_collections()

        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []

    def get_project_info(self) -> dict | None:
        """Get current project information from detection.

        Returns:
            Optional[Dict]: Project information containing:
                - main_project (str): Primary project name
                - subprojects (List[str]): Detected subproject names
                - git_info (dict): Git repository information if available
                - directory_structure (dict): Relevant directory analysis
                Returns None if project detection hasn't been performed.
        """
        return self.project_info

    def refresh_project_detection(self) -> dict:
        """Refresh project detection from current working directory.

        Re-analyzes the current directory structure and Git repository
        to update project information. Useful when the working directory
        changes or project structure is modified.

        Returns:
            Dict: Updated project information with same structure as get_project_info()
        """
        self.project_info = self.project_detector.get_project_info()
        return self.project_info

    def get_embedding_service(self) -> EmbeddingService:
        """Get the embedding service instance for direct access.

        Provides access to the underlying embedding service for operations
        like generating embeddings, chunking text, or model information.

        Returns:
            EmbeddingService: The configured embedding service instance
        """
        return self.embedding_service

    async def ensure_collection_exists(
        self, collection_name: str, collection_type: str = "scratchbook"
    ) -> None:
        """Ensure a collection exists, creating it if necessary.

        This method creates a collection with appropriate configuration if it doesn't
        already exist. It's designed to be called by tools that need to ensure
        their target collection is available before performing operations.

        Args:
            collection_name: Name of the collection to create
            collection_type: Type of collection (used for description)

        Raises:
            RuntimeError: If client not initialized, collection creation fails, or LLM access control blocks creation
            ValueError: If collection_name is invalid

        Example:
            ```python
            # Ensure scratchbook collection exists
            await client.ensure_collection_exists("my-project-scratchbook")

            # Ensure custom collection exists
            await client.ensure_collection_exists("my-project-docs", "docs")
            ```
        """
        if not self.initialized:
            raise RuntimeError("Client not initialized")

        if not collection_name or not collection_name.strip():
            raise ValueError("Collection name cannot be empty")

        # Apply LLM access control validation for collection creation
        # TODO: Re-enable after Task 175-177 integration is complete
        # try:
        #     validate_llm_collection_access('create', collection_name, self.config)
        # except LLMAccessControlError as e:
        #     logger.warning("LLM access control blocked collection creation: %s", str(e))
        #     raise RuntimeError(f"Collection creation blocked: {str(e)}") from e

        try:
            # Import here to avoid circular imports
            from .collections import CollectionConfig

            # Create a collection configuration
            collection_config = CollectionConfig(
                name=collection_name,
                description=f"{collection_type.title()} collection",
                collection_type=collection_type,
                vector_size=self.collection_manager._get_vector_size(),
                enable_sparse_vectors=self.config.embedding.enable_sparse_vectors,
            )

            # Use the collection manager's private method to ensure the collection exists
            self.collection_manager._ensure_collection_exists(collection_config)

            logger.info("Ensured collection exists: %s", collection_name)

        except Exception as e:
            logger.error(
                "Failed to ensure collection %s exists: %s", collection_name, e
            )
            raise RuntimeError(
                f"Failed to ensure collection '{collection_name}' exists: {e}"
            ) from e

    async def close(self) -> None:
        """Clean up client connections and release resources.

        Properly closes all connections, cleans up embedding models,
        and resets the client to uninitialized state. Should be called
        when the client is no longer needed to prevent resource leaks.

        Example:
            ```python
            try:
                # Use the client
                await client.initialize()
                # ... perform operations ...
            finally:
                await client.close()  # Always clean up
            ```
        """
        if self.embedding_service:
            await self.embedding_service.close()
        if self.client:
            self.client.close()
            self.client = None
        self.initialized = False


def create_qdrant_client(config_data) -> QdrantWorkspaceClient:
    """Create a QdrantWorkspaceClient instance from configuration data.

    This is a factory function that creates and initializes a workspace client
    from configuration data. It's used throughout the CLI and tools for
    consistent client creation.

    Args:
        config_data: Configuration data that should be compatible with Config.qdrant_client_config

    Returns:
        QdrantWorkspaceClient: Initialized client instance
    """
    from .config import Config

    # Create config instance
    config = Config()

    # Create and return the client
    client = QdrantWorkspaceClient(config)
    return client
