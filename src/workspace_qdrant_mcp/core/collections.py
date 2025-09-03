"""
Collection management for workspace-scoped Qdrant collections.

This module provides comprehensive collection management for project-aware Qdrant
vector databases. It handles automatic creation, configuration, and lifecycle
management of workspace-scoped collections based on detected project structure.

Key Features:
    - Automatic collection creation based on project detection
    - Support for project-specific and global collections
    - Dense and sparse vector configuration
    - Optimized collection settings for search performance
    - Workspace isolation and collection filtering
    - Parallel collection creation for better performance

Collection Types:
    - Project collections: [project-name]-{suffix} for each configured suffix
    - Subproject collections: [subproject-name]-{suffix} for each configured suffix
    - Global collections: scratchbook, shared-notes (cross-project)

Example:
    ```python
    from workspace_qdrant_mcp.core.collections import WorkspaceCollectionManager
    from qdrant_client import QdrantClient

    client = QdrantClient("http://localhost:6333")
    manager = WorkspaceCollectionManager(client, config)

    # Initialize collections for detected project
    await manager.initialize_workspace_collections(
        project_name="my-project",
        subprojects=["frontend", "backend"]
    )

    # List available workspace collections
    collections = await manager.list_workspace_collections()
    ```
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .collection_naming import (
    CollectionNamingManager,
    CollectionPermissionError,
    CollectionType,
)
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration specification for a workspace collection.

    Defines the complete configuration for creating and managing workspace
    collections including vector parameters, metadata, and optimization settings.

    Attributes:
        name: Unique collection identifier within the Qdrant database
        description: Human-readable description of collection purpose
        collection_type: Collection category - 'scratchbook', 'docs', or 'global'
        project_name: Associated project name (None for global collections)
        vector_size: Dimension of dense embedding vectors (model-dependent)
        distance_metric: Vector similarity metric - 'Cosine', 'Euclidean', 'Dot'
        enable_sparse_vectors: Whether to enable sparse keyword-based vectors

    Example:
        ```python
        config = CollectionConfig(
            name="my-project-docs",
            description="Documentation for my-project",
            collection_type="docs",
            project_name="my-project",
            vector_size=384,
            enable_sparse_vectors=True
        )
        ```
    """

    name: str
    description: str
    collection_type: str  # 'scratchbook', 'docs', 'global'
    project_name: str | None = None
    vector_size: int = 384  # all-MiniLM-L6-v2 dimension
    distance_metric: str = "Cosine"
    enable_sparse_vectors: bool = True


class WorkspaceCollectionManager:
    """
    Manages project-scoped collections for workspace-aware Qdrant operations.

    This class handles the complete lifecycle of workspace collections including
    creation, configuration, optimization, and management. It provides workspace
    isolation by managing project-specific collections while maintaining access
    to global shared collections.

    The manager automatically:
        - Creates project and subproject collections based on detection
        - Configures dense and sparse vector support
        - Applies performance optimization settings
        - Filters workspace collections from external collections
        - Manages collection metadata and statistics

    Attributes:
        client: Underlying Qdrant client for database operations
        config: Configuration object with workspace and embedding settings
        _collections_cache: Optional cache for collection configurations

    Example:
        ```python
        from qdrant_client import QdrantClient
        from workspace_qdrant_mcp.core.config import Config

        client = QdrantClient("http://localhost:6333")
        config = Config()
        manager = WorkspaceCollectionManager(client, config)

        # Initialize workspace collections
        await manager.initialize_workspace_collections(
            project_name="my-app",
            subprojects=["frontend", "backend", "api"]
        )

        # Get workspace status
        collections = await manager.list_workspace_collections()
        info = await manager.get_collection_info()
        ```
    """

    def __init__(self, client: QdrantClient, config: Config) -> None:
        """Initialize the collection manager.

        Args:
            client: Configured Qdrant client instance for database operations
            config: Configuration object containing workspace and embedding settings
        """
        self.client = client
        self.config = config
        self._collections_cache: dict[str, CollectionConfig] | None = None
        # Initialize the naming manager with legacy global collections for compatibility
        self.naming_manager = CollectionNamingManager(
            global_collections=self.config.workspace.global_collections
        )

    async def initialize_workspace_collections(
        self, project_name: str, subprojects: list[str] | None = None
    ) -> None:
        """
        Initialize collections for the current workspace based on configuration and project structure.

        Behavior depends on the auto_create_collections setting:
        - When auto_create_collections=True: Creates project collections, subproject collections, and global collections
        - When auto_create_collections=False: Only creates scratchbook collection for basic functionality

        Collection Creation Patterns:
            With auto_create_collections=True:
                Main project: [project-name]-{suffix} for each workspace.collections suffix
                Subprojects: [subproject-name]-{suffix} for each workspace.collections suffix
                Global: All collections from workspace.global_collections

            With auto_create_collections=False:
                Only: scratchbook collection (essential for cross-project note-taking)

        Args:
            project_name: Main project identifier (used as collection name prefix)
            subprojects: Optional list of subproject names for additional collections

        Raises:
            ConnectionError: If Qdrant database is unreachable
            ResponseHandlingException: If collection creation fails due to Qdrant errors
            RuntimeError: If configuration or optimization settings are invalid

        Example:
            ```python
            # Initialize for simple project (respects auto_create_collections setting)
            await manager.initialize_workspace_collections("my-app")

            # Initialize with subprojects (respects auto_create_collections setting)
            await manager.initialize_workspace_collections(
                project_name="enterprise-system",
                subprojects=["web-frontend", "mobile-app", "api-gateway"]
            )
            ```
        """
        collections_to_create = []

        if self.config.workspace.auto_create_collections:
            # Full collection creation when auto_create_collections=True

            # Main project collections
            for suffix in self.config.workspace.collections:
                collections_to_create.append(
                    CollectionConfig(
                        name=f"{project_name}-{suffix}",
                        description=f"{suffix.title()} collection for {project_name}",
                        collection_type=suffix,
                        project_name=project_name,
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.embedding.enable_sparse_vectors,
                    )
                )

            # Subproject collections
            if subprojects:
                for subproject in subprojects:
                    for suffix in self.config.workspace.collections:
                        collections_to_create.append(
                            CollectionConfig(
                                name=f"{subproject}-{suffix}",
                                description=f"{suffix.title()} collection for {subproject}",
                                collection_type=suffix,
                                project_name=subproject,
                                vector_size=self._get_vector_size(),
                                enable_sparse_vectors=self.config.embedding.enable_sparse_vectors,
                            )
                        )

            # Global collections
            for global_collection in self.config.workspace.global_collections:
                collections_to_create.append(
                    CollectionConfig(
                        name=global_collection,
                        description=f"Global {global_collection} collection",
                        collection_type="global",
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.embedding.enable_sparse_vectors,
                    )
                )
        else:
            # Minimal collection creation when auto_create_collections=False
            # Only create scratchbook collection for essential functionality
            collections_to_create.append(
                CollectionConfig(
                    name="scratchbook",
                    description="Global scratchbook collection for cross-project notes",
                    collection_type="global",
                    vector_size=self._get_vector_size(),
                    enable_sparse_vectors=self.config.embedding.enable_sparse_vectors,
                )
            )

        # Create collections in parallel for better performance
        if collections_to_create:
            await asyncio.gather(
                *[
                    self._ensure_collection_exists(config)
                    for config in collections_to_create
                ]
            )

    async def _ensure_collection_exists(
        self, collection_config: CollectionConfig
    ) -> None:
        """
        Ensure a collection exists with proper configuration, creating if necessary.

        Performs idempotent collection creation with optimized vector settings,
        HNSW indexing parameters, and memory management configuration. If the
        collection already exists, no action is taken.

        The method configures:
            - Dense vector parameters (size, distance metric)
            - Sparse vector support (if enabled)
            - HNSW index optimization (m=16, ef_construct=100)
            - Memory mapping thresholds and segment management

        Args:
            collection_config: Complete configuration specification for the collection

        Raises:
            ResponseHandlingException: If Qdrant API calls fail
            ValueError: If configuration parameters are invalid
            ConnectionError: If database is unreachable

        Example:
            ```python
            config = CollectionConfig(
                name="project-docs",
                description="Project documentation",
                collection_type="docs",
                vector_size=384,
                enable_sparse_vectors=True
            )
            await manager._ensure_collection_exists(config)
            ```
        """
        try:
            # Check if collection already exists
            existing_collections = self.client.get_collections()
            collection_names = {col.name for col in existing_collections.collections}

            if collection_config.name in collection_names:
                logger.info("Collection %s already exists", collection_config.name)
                return

            # Create collection with correct API format for Qdrant 1.15+
            if collection_config.enable_sparse_vectors:
                # Collections with both dense and sparse vectors
                self.client.create_collection(
                    collection_name=collection_config.name,
                    vectors_config={
                        "dense": models.VectorParams(
                            size=collection_config.vector_size,
                            distance=getattr(
                                models.Distance,
                                collection_config.distance_metric.upper(),
                            ),
                        )
                    },
                    sparse_vectors_config={"sparse": models.SparseVectorParams()},
                )
            else:
                # Dense vectors only
                self.client.create_collection(
                    collection_name=collection_config.name,
                    vectors_config=models.VectorParams(
                        size=collection_config.vector_size,
                        distance=getattr(
                            models.Distance, collection_config.distance_metric.upper()
                        ),
                    ),
                )

            # Set collection metadata
            self.client.update_collection(
                collection_name=collection_config.name,
                optimizer_config=models.OptimizersConfigDiff(
                    default_segment_number=2,
                    memmap_threshold=20000,
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10000,
                ),
            )

            logger.info("Created collection: %s", collection_config.name)

        except ResponseHandlingException as e:
            logger.error(
                "Failed to create collection %s: %s", collection_config.name, e
            )
            raise
        except Exception as e:
            logger.error(
                "Unexpected error creating collection %s: %s", collection_config.name, e
            )
            raise

    def list_workspace_collections(self) -> list[str]:
        """
        List all collections that belong to the current workspace.

        Filters the complete list of Qdrant collections to return only those
        that are part of the current workspace. Excludes external collections
        like memexd daemon collections (ending in '-code') while including
        project collections and global workspace collections.

        Filtering Logic:
            - Include: [project]-docs, [project]-scratchbook collections
            - Include: Global collections defined in workspace configuration
            - Exclude: External daemon collections (e.g., memexd-*-code)
            - Exclude: Collections from other workspace instances

        Returns:
            List[str]: Sorted list of workspace collection names.
                Returns empty list if no collections found or on error.

        Example:
            ```python
            collections = manager.list_workspace_collections()
            # Example return: ['my-project-docs', 'my-project-scratchbook', 'scratchbook']

            for collection in collections:
                logger.info("Workspace collection: {collection}")
            ```
        """
        try:
            all_collections = self.client.get_collections()
            workspace_collections = []

            for collection in all_collections.collections:
                # Filter for workspace collections (exclude memexd -code collections)
                if self._is_workspace_collection(collection.name):
                    workspace_collections.append(collection.name)

            return sorted(workspace_collections)

        except Exception as e:
            logger.error("Failed to list collections: %s", e)
            return []

    def get_collection_info(self) -> dict:
        """
        Get comprehensive information about all workspace collections.

        Retrieves detailed statistics and configuration information for each
        workspace collection including vector counts, indexing status, and
        optimization parameters. Provides a complete health check of the workspace.

        Returns:
            Dict: Collection information containing:
                - collections (dict): Per-collection statistics with keys:
                    - vectors_count (int): Number of vectors stored
                    - points_count (int): Total number of points/documents
                    - status (str): Collection status (green/yellow/red)
                    - optimizer_status (dict): Indexing and optimization status
                    - config (dict): Vector configuration (distance, size)
                    - error (str): Error message if collection inaccessible
                - total_collections (int): Total number of workspace collections

        Example:
            ```python
            info = await manager.get_collection_info()
            logger.info("Total collections: {info['total_collections']}")

            for name, details in info['collections'].items():
                logger.info("{name}: {details['points_count']} documents")
                if 'error' in details:
                    logger.info("  Error: {details['error']}")
            ```
        """
        try:
            workspace_collections = self.list_workspace_collections()
            collection_info = {}

            for collection_name in workspace_collections:
                try:
                    info = self.client.get_collection(collection_name)
                    collection_info[collection_name] = {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "status": info.status,
                        "optimizer_status": info.optimizer_status,
                        "config": {
                            "distance": info.config.params.vectors.distance,
                            "vector_size": info.config.params.vectors.size,
                        },
                    }
                except Exception as e:
                    logger.warning(
                        "Failed to get info for collection %s: %s", collection_name, e
                    )
                    collection_info[collection_name] = {"error": str(e)}

            return {
                "collections": collection_info,
                "total_collections": len(workspace_collections),
            }

        except Exception as e:
            logger.error("Failed to get collection info: %s", e)
            return {"error": str(e)}

    def _is_workspace_collection(self, collection_name: str) -> bool:
        """
        Determine if a collection belongs to the current workspace.

        Applies filtering logic to distinguish workspace collections from
        external collections that may exist in the same Qdrant database.
        This enables workspace isolation while sharing the database instance.

        Inclusion Criteria:
            - Collections ending in configured project collection suffixes
            - Collections in the global_collections configuration list
            - Collections that match workspace naming patterns

        Exclusion Criteria:
            - Collections ending in '-code' (memexd daemon collections)
            - Collections from other workspace instances
            - System or temporary collections

        Args:
            collection_name: Name of the collection to evaluate

        Returns:
            bool: True if the collection belongs to this workspace,
                  False if it's external or system collection

        Example:
            ```python
            # These would return True:
            manager._is_workspace_collection("my-project-docs")         # True (if 'docs' in collections)
            manager._is_workspace_collection("scratchbook")            # True (global)

            # These would return False:
            manager._is_workspace_collection("memexd-project-code")    # False (daemon)
            manager._is_workspace_collection("other-system-temp")      # False (external)
            ```
        """
        # Exclude memexd daemon collections (those ending with -code)
        if collection_name.endswith("-code"):
            return False

        # Include global collections
        if collection_name in self.config.workspace.global_collections:
            return True

        # Include project collections (ending with configured suffixes)
        for suffix in self.config.workspace.collections:
            if collection_name.endswith(f"-{suffix}"):
                return True

        return False

    def resolve_collection_name(self, display_name: str) -> tuple[str, bool]:
        """
        Resolve a display name to the actual collection name and permission info.

        This handles the mapping from user-facing display names to actual Qdrant
        collection names, particularly for library collections that use underscore prefixes.

        Args:
            display_name: The collection name as shown to users

        Returns:
            Tuple of (actual_collection_name, is_readonly_from_mcp)

        Example:
            ```python
            # Library collection
            actual, readonly = manager.resolve_collection_name("mylib")
            # Returns: ("_mylib", True)

            # Project collection
            actual, readonly = manager.resolve_collection_name("my-project-docs")
            # Returns: ("my-project-docs", False)
            ```
        """
        # First, check if this display name corresponds to a library collection
        potential_library_name = f"_{display_name}"

        try:
            all_collections = self.client.get_collections()
            all_collection_names = [col.name for col in all_collections.collections]

            if potential_library_name in all_collection_names:
                # This is a library collection
                return potential_library_name, True
            elif display_name in all_collection_names:
                # This is a regular collection, check if it's readonly
                info = self.naming_manager.get_collection_info(display_name)
                return display_name, info.is_readonly_from_mcp
            else:
                # Collection doesn't exist - return the display name as-is for error handling
                return display_name, False

        except Exception as e:
            logger.error(f"Failed to resolve collection name '{display_name}': {e}")
            return display_name, False

    def validate_mcp_write_access(self, display_name: str) -> None:
        """
        Validate that the MCP server can write to a collection.

        Args:
            display_name: The collection display name

        Raises:
            CollectionPermissionError: If the collection is readonly from MCP
        """
        actual_name, is_readonly = self.resolve_collection_name(display_name)

        if is_readonly:
            info = self.naming_manager.get_collection_info(actual_name)
            if info.collection_type == CollectionType.LIBRARY:
                raise CollectionPermissionError(
                    f"Library collection '{display_name}' is readonly from MCP server. "
                    f"Use the CLI/Rust engine to modify library collections."
                )
            else:
                raise CollectionPermissionError(
                    f"Collection '{display_name}' is readonly from MCP server."
                )

    def get_naming_manager(self) -> CollectionNamingManager:
        """
        Get the collection naming manager for direct access.

        Returns:
            The CollectionNamingManager instance used by this manager
        """
        return self.naming_manager

    def _get_vector_size(self) -> int:
        """
        Get the vector dimension size for the currently configured embedding model.

        Maps embedding model names to their corresponding vector dimensions.
        This ensures that collections are created with the correct vector size
        for the embedding model being used.

        Supported Models:
            - sentence-transformers/all-MiniLM-L6-v2: 384 dimensions (default, lightweight)
            - BAAI/bge-base-en-v1.5: 768 dimensions (better quality)
            - BAAI/bge-large-en-v1.5: 1024 dimensions (best quality, high resource)

        Returns:
            int: Vector dimension size for the configured model.
                 Defaults to 384 if model is not recognized.

        Example:
            ```python
            # With all-MiniLM-L6-v2 configured (default)
            size = manager._get_vector_size()  # Returns 384

            # With BAAI/bge-m3 configured
            size = manager._get_vector_size()  # Returns 1024
            ```
        """
        # This will be updated when FastEmbed is integrated
        model_sizes = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-m3": 1024,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "jinaai/jina-embeddings-v2-base-en": 768,
            "thenlper/gte-base": 768,
            "nomic-ai/nomic-embed-text-v1.5": 768,
        }

        return model_sizes.get(self.config.embedding.model, 384)
