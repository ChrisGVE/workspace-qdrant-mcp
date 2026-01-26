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
    - Global collections: User-defined collections that span across projects

Example:
    ```python
    from workspace_qdrant_mcp.core.collections import WorkspaceCollectionManager
    from qdrant_client import QdrantClient
    from .ssl_config import suppress_qdrant_ssl_warnings

    with suppress_qdrant_ssl_warnings():
        client = QdrantClient("http://localhost:6333")
    manager = WorkspaceCollectionManager(client, config)

    # Initialize collections for detected project
    await manager.initialize_workspace_collections(
        project_name="my-project",
        subprojects=["frontend", "backend"]
    )

    # List available workspace collections
    collections = manager.list_workspace_collections()
    ```
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import git
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from .collection_naming import (
    CollectionNamingManager,
    CollectionPermissionError,
    build_user_collection_name,
    build_system_memory_collection_name,
)
from .collection_types import (
    COLLECTION_TYPES_AVAILABLE,
    CollectionType,
    CollectionTypeClassifier,
)
from .config import ConfigManager

# Import LLM access control system
try:
    from .llm_access_control import (
        LLMAccessControlError,
        validate_llm_collection_access,
    )
except ImportError:
    # Fallback for direct imports when not used as a package
    from llm_access_control import LLMAccessControlError, validate_llm_collection_access

# logger imported from loguru


@dataclass
class CollectionConfig:
    """Configuration specification for a workspace collection.

    Defines the complete configuration for creating and managing workspace
    collections including vector parameters, metadata, and optimization settings.

    Attributes:
        name: Unique collection identifier within the Qdrant database
        description: Human-readable description of collection purpose
        collection_type: Collection category - user-defined type or 'global'
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
    collection_type: str  # user-defined type or 'global'
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
        from .ssl_config import suppress_qdrant_ssl_warnings

        with suppress_qdrant_ssl_warnings():
            client = QdrantClient("http://localhost:6333")
        config = Config()
        manager = WorkspaceCollectionManager(client, config)

        # Initialize workspace collections
        await manager.initialize_workspace_collections(
            project_name="my-app",
            subprojects=["frontend", "backend", "api"]
        )

        # Get workspace status
        collections = manager.list_workspace_collections()
        info = await manager.get_collection_info()
        ```
    """

    def __init__(self, client: QdrantClient, config: ConfigManager) -> None:
        """Initialize the collection manager.

        Args:
            client: Configured Qdrant client instance for database operations
            config: Configuration object containing workspace and embedding settings
        """
        self.client = client
        self.config = config
        self._collections_cache: dict[str, CollectionConfig] | None = None
        self._project_info: dict | None = None  # Will be set during initialization

        # Initialize collection type classifier and naming manager
        self.type_classifier = CollectionTypeClassifier()
        self.naming_manager = CollectionNamingManager(
            config.get("workspace.global_collections", []),
            config.get("workspace.collection_types", []),
        )

    def _get_current_project_name(self) -> str | None:
        """
        Determine the current project name from working directory or git repository.

        Attempts to extract the project name from:
        1. Git repository name (if in a git repository)
        2. Current working directory name
        3. Parent directory name (if current is a subdirectory)

        Returns:
            str: Project name if determinable, None otherwise

        Example:
            For /path/to/workspace-qdrant-mcp -> "workspace-qdrant-mcp"
            For git repo myproject -> "myproject"
        """
        try:
            # Try to get project name from git repository
            try:
                repo = git.Repo(search_parent_directories=True)
                if repo.remotes:
                    # Extract from remote URL (e.g., git@github.com:user/project.git -> project)
                    remote_url = repo.remotes[0].url
                    if remote_url.endswith('.git'):
                        remote_url = remote_url[:-4]
                    project_name = remote_url.split('/')[-1]
                    if project_name and project_name != '.' and not project_name.startswith('.'):
                        return project_name
            except (git.InvalidGitRepositoryError, git.GitCommandError):
                pass

            # Fallback to directory name
            current_dir = Path.cwd()
            project_name = current_dir.name

            # Skip common subdirectory names and go to parent
            skip_dirs = {'src', 'lib', 'app', 'core', 'workspace_qdrant_mcp'}
            if project_name in skip_dirs and current_dir.parent != current_dir:
                project_name = current_dir.parent.name

            if project_name and project_name != '.' and not project_name.startswith('.'):
                return project_name

        except Exception as e:
            logger.debug("Could not determine project name: %s", e)

        return None

    def _get_all_project_names(self) -> list[str]:
        """
        Get all project names including main project and subprojects.

        Returns:
            List[str]: List of all project names that should be considered
                      for workspace collection filtering
        """
        project_names = []

        # Try to get project info from stored data first
        if self._project_info:
            main_project = self._project_info.get('main_project')
            if main_project:
                project_names.append(main_project)

            subprojects = self._project_info.get('subprojects', [])
            project_names.extend(subprojects)
        else:
            # Fallback to current project name detection
            current_project = self._get_current_project_name()
            if current_project:
                project_names.append(current_project)

        # Remove duplicates and empty values
        project_names = list({name for name in project_names if name})

        logger.debug("All project names for collection filtering: %s", project_names)
        return project_names

    def validate_collection_filtering(self) -> dict:
        """
        Validate and diagnose collection filtering configuration.

        This method provides diagnostic information about the current
        collection filtering setup, useful for debugging issues.

        Returns:
            Dict: Diagnostic information containing:
                - project_info: Stored project information
                - project_names: All project names used for filtering
                - config_info: Configuration settings
                - all_collections: All collections in Qdrant
                - workspace_collections: Filtered workspace collections
                - filtering_results: Per-collection filtering decisions
        """
        try:
            # Get current state
            all_collections = self.client.get_collections()
            all_names = [c.name for c in all_collections.collections]
            project_names = self._get_all_project_names()
            workspace_collections = self.list_workspace_collections()

            # Test filtering for each collection
            filtering_results = {}
            for name in all_names:
                filtering_results[name] = {
                    'is_workspace': self._is_workspace_collection(name),
                    'reason': self._get_filtering_reason(name)
                }

            return {
                'project_info': self._project_info,
                'project_names': project_names,
                'config_info': {
                    'effective_collection_types': self.config.get("workspace.collection_types", []),
                    'global_collections': self.config.get("workspace.global_collections", []),
                    'auto_create_collections': self.config.get("workspace.auto_create_collections", False)
                },
                'all_collections': all_names,
                'workspace_collections': workspace_collections,
                'filtering_results': filtering_results,
                'summary': {
                    'total_collections': len(all_names),
                    'workspace_collections': len(workspace_collections),
                    'excluded_collections': len(all_names) - len(workspace_collections)
                }
            }
        except Exception as e:
            logger.error("Failed to validate collection filtering: %s", e)
            return {'error': str(e)}

    def _get_filtering_reason(self, collection_name: str) -> str:
        """
        Get the reason why a collection is included or excluded from workspace.

        Args:
            collection_name: Name of the collection to check

        Returns:
            str: Human-readable reason for the filtering decision
        """
        # Check exclusion criteria first
        if collection_name.endswith("-code"):
            return "Excluded: memexd daemon collection (ends with -code)"

        # Check inclusion criteria
        if collection_name in self.config.get("workspace.global_collections", []):
            return "Included: global collection"

        for suffix in self.config.get("workspace.collection_types", []):
            if collection_name.endswith(f"-{suffix}"):
                return f"Included: ends with configured suffix '{suffix}'"

        # Check project-based inclusion
        effective_collection_types = self.config.get("workspace.collection_types", [])
        global_collections = self.config.get("workspace.global_collections", [])
        if not effective_collection_types and not global_collections:
            project_names = self._get_all_project_names()

            for project_name in project_names:
                if collection_name.startswith(f"{project_name}-"):
                    return f"Included: matches project '{project_name}' pattern"
                if collection_name == project_name:
                    return f"Included: exact match with project '{project_name}'"

            common_standalone = ["reference", "docs", "standards", "notes", "scratchbook", "memory", "knowledge"]
            if collection_name in common_standalone:
                return "Included: common standalone collection"

        return "Excluded: does not match any inclusion criteria"

    async def initialize_workspace_collections(
        self, project_name: str, subprojects: list[str] | None = None
    ) -> None:
        """
        Initialize collections for the current workspace based on configuration and project structure.

        Behavior depends on the auto_create_collections setting:
        - When auto_create_collections=True: Creates shared workspace collections with metadata-based isolation
        - When auto_create_collections=False: No collections are created automatically

        Collection Creation Patterns:
            With auto_create_collections=True (Multi-tenant mode):
                Shared collections: {suffix} for each workspace.collections suffix
                Global: All collections from workspace.global_collections
                Project isolation: Through metadata filtering (project_name, tenant_namespace)

            With auto_create_collections=False:
                Only: No collections are created (user must explicitly configure collections)

        Multi-tenant Architecture:
            - Single shared collection per type (notes, docs, scratchbook, etc.)
            - Project isolation through metadata filtering
            - Efficient resource utilization and centralized management
            - Maintains workspace boundaries through tenant_namespace metadata

        Args:
            project_name: Main project identifier (used for metadata context)
            subprojects: Optional list of subproject names for additional contexts

        Raises:
            ConnectionError: If Qdrant database is unreachable
            ResponseHandlingException: If collection creation fails due to Qdrant errors
            RuntimeError: If configuration or optimization settings are invalid

        Example:
            ```python
            # Initialize for simple project (creates shared collections with metadata isolation)
            await manager.initialize_workspace_collections("my-app")

            # Initialize with subprojects (same shared collections, different metadata contexts)
            await manager.initialize_workspace_collections(
                project_name="enterprise-system",
                subprojects=["web-frontend", "mobile-app", "api-gateway"]
            )
            ```
        """
        # Store project information for collection filtering
        self._project_info = {
            'main_project': project_name,
            'subprojects': subprojects or []
        }

        logger.debug(
            "Setting project info for collection filtering: main_project=%s, subprojects=%s",
            project_name, subprojects
        )

        collections_to_create = []

        if self.config.get("workspace.auto_create_collections", False):
            # Multi-tenant collection creation when auto_create_collections=True
            # Creates shared collections with metadata-based project isolation

            # Shared workspace collections (one per type, multi-tenant)
            for suffix in self.config.get("workspace.collection_types", []):
                collection_name = suffix  # Direct collection name, no project prefix
                collections_to_create.append(
                    CollectionConfig(
                        name=collection_name,
                        description=f"Multi-tenant {suffix.title()} collection with metadata-based project isolation",
                        collection_type=suffix,
                        project_name=None,  # Multi-tenant, no single project owner
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
                    )
                )
                logger.info(
                    "Configured multi-tenant collection",
                    collection=collection_name,
                    type=suffix,
                    projects=[project_name] + (subprojects or [])
                )

            # Global collections (unchanged)
            for global_collection in self.config.get("workspace.global_collections", []):
                collections_to_create.append(
                    CollectionConfig(
                        name=global_collection,
                        description=f"Global {global_collection} collection",
                        collection_type="global",
                        vector_size=self._get_vector_size(),
                        enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
                    )
                )

            # Create read-only _codebase collection for code content
            # This collection is read-only from MCP and designed for code search
            collections_to_create.append(
                CollectionConfig(
                    name="_codebase",
                    description="Read-only code collection with optimized indexing for code search",
                    collection_type="library",
                    vector_size=self._get_vector_size(),
                    enable_sparse_vectors=True,  # Force sparse vectors for better code search
                )
            )

            logger.info(
                "Multi-tenant workspace initialization",
                main_project=project_name,
                subprojects=subprojects or [],
                shared_collections=[c.name for c in collections_to_create if c.collection_type != "global"]
            )

        # If auto_create_collections=False, no collections are created
        # All collections must be explicitly configured by the user

        # Create collections sequentially since they use synchronous Qdrant client
        if collections_to_create:
            for config in collections_to_create:
                self._ensure_collection_exists(config)

            # Add metadata indexing for multi-tenant performance
            await self._optimize_metadata_indexing(collections_to_create)

        logger.info(
            "Workspace collections initialized",
            project_name=project_name,
            collections_created=len(collections_to_create),
            multi_tenant_mode=True
        )

    def _ensure_collection_exists(
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
            ValueError: If configuration parameters are invalid or LLM access control blocks creation
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
            # Apply LLM access control validation for collection creation
            try:
                validate_llm_collection_access('create', collection_config.name, self.config)
            except LLMAccessControlError as e:
                logger.warning("LLM access control blocked collection creation: %s", str(e))
                raise ValueError(f"Collection creation blocked: {str(e)}") from e

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
        List all collections that belong to the current workspace with display names.

        Filters the complete list of Qdrant collections to return only those
        that are part of the current workspace, returning display names that
        remove prefixes for system and library collections.

        Filtering Logic:
            - Include: System collections (__prefix) - CLI-writable, LLM-readable
            - Include: Library collections (_prefix) - CLI-managed, MCP-readonly
            - Include: Project collections ({project}-{suffix}) - user-created
            - Include: Global collections (predefined system-wide)
            - Exclude: External daemon collections (e.g., memexd-*-code)
            - Exclude: Collections from other workspace instances

        Returns:
            List[str]: Sorted list of workspace collection display names.
                System/library collections have prefixes removed for cleaner UX.
                Returns empty list if no collections found or on error.

        Example:
            ```python
            collections = manager.list_workspace_collections()
            # Example return: ['user_preferences', 'library_docs', 'project-documents']
            # (display names: '__user_preferences' -> 'user_preferences')

            for collection in collections:
                logger.info("Workspace collection: {collection}")
            ```
        """
        try:
            all_collections = self.client.get_collections()
            workspace_collections = []
            all_collection_names = [c.name for c in all_collections.collections]

            logger.debug("Filtering collections. Total collections: %d", len(all_collection_names))
            logger.debug("All collections: %s", all_collection_names)

            for collection in all_collections.collections:
                collection_name = collection.name

                # Use new collection type system for classification and display names
                collection_info = self.type_classifier.get_collection_info(collection_name)

                # Include all workspace collections (system, library, project, global)
                # but exclude unknown/external collections
                if collection_info.collection_type != CollectionType.UNKNOWN:
                    display_name = self.type_classifier.get_display_name(collection_name)
                    workspace_collections.append(display_name)

            logger.info(
                "Found %d workspace collections out of %d total collections",
                len(workspace_collections), len(all_collection_names)
            )
            logger.debug("Workspace collections: %s", workspace_collections)

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

                    # Handle the new Qdrant API structure where vectors is a dict
                    vectors_config = info.config.params.vectors
                    if isinstance(vectors_config, dict):
                        # New API: vectors is a dict with keys like 'dense', 'sparse'
                        # Use the first available vector config (usually 'dense')
                        if 'dense' in vectors_config:
                            vector_params = vectors_config['dense']
                        else:
                            # Fallback to the first available vector config
                            vector_params = next(iter(vectors_config.values()))

                        distance = vector_params.distance
                        vector_size = vector_params.size
                    else:
                        # Legacy API: vectors is directly a VectorParams object
                        distance = vectors_config.distance
                        vector_size = vectors_config.size

                    collection_info[collection_name] = {
                        "vectors_count": info.vectors_count,
                        "points_count": info.points_count,
                        "status": info.status,
                        "optimizer_status": info.optimizer_status,
                        "config": {
                            "distance": distance,
                            "vector_size": vector_size,
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

        Uses CollectionNamingManager to classify collections and determine workspace membership.
        This enables workspace isolation while sharing the database instance and properly
        handles the new readonly collection prefix system.

        Inclusion Criteria:
            - Memory collections ('memory')
            - Library collections ('_name' pattern - readonly from MCP)
            - Project collections ('{project}-{suffix}' pattern)
            - Legacy collections matching configuration or naming patterns

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
            manager._is_workspace_collection("memory")                  # True (memory collection)
            manager._is_workspace_collection("_library")               # True (library collection)
            manager._is_workspace_collection("my-project-docs")        # True (project collection)
            manager._is_workspace_collection("user-collection")        # True (if legacy configured)

            # These would return False:
            manager._is_workspace_collection("memexd-project-code")    # False (daemon)
            manager._is_workspace_collection("other-system-temp")      # False (external)
            ```
        """
        # Use CollectionNamingManager for classification
        collection_info = self.naming_manager.get_collection_info(collection_name)

        # Include all workspace collection types (memory, library, project)
        if collection_info.collection_type in [
            CollectionType.MEMORY,
            CollectionType.LIBRARY,
            CollectionType.PROJECT
        ]:
            return True

        # For legacy collections, apply the existing filtering logic
        if collection_info.collection_type == CollectionType.LEGACY:
            # Exclude memexd daemon collections (those ending with -code)
            if collection_name.endswith("-code"):
                return False

            # Include global collections
            if collection_name in self.config.get("workspace.global_collections", []):
                return True

            # Include project collections (ending with configured suffixes)
            for suffix in self.config.get("workspace.collection_types", []):
                if collection_name.endswith(f"-{suffix}"):
                    return True

            # When no specific configuration is provided, use the actual project name to identify collections
            # This provides accurate workspace isolation based on the current project context
            effective_collection_types = self.config.get("workspace.collection_types", [])
            global_collections = self.config.get("workspace.global_collections", [])
            if not effective_collection_types and not global_collections:
                # Get project information from stored project info or fallback to detection
                project_names = self._get_all_project_names()

                # Check if collection matches any project naming pattern: {project_name}-{suffix}
                for project_name in project_names:
                    if collection_name.startswith(f"{project_name}-"):
                        logger.debug("Collection %s matches project %s pattern", collection_name, project_name)
                        return True

                    # Also include standalone collections that match any project name exactly
                    if collection_name == project_name:
                        logger.debug("Collection %s matches project %s exactly", collection_name, project_name)
                        return True

                # Fallback to common standalone collections for workspace context
                common_standalone_collections = ["reference", "docs", "standards", "notes", "scratchbook", "memory", "knowledge"]
                if collection_name in common_standalone_collections:
                    logger.debug("Collection %s matches common standalone pattern", collection_name)
                    return True

        return False

    def resolve_collection_name(self, display_name: str) -> tuple[str, bool]:
        """
        Resolve a display name to the actual collection name and permission info.

        This handles the mapping from user-facing display names to actual Qdrant
        collection names, particularly for system and library collections that use prefixes.
        Uses the new collection type system for better accuracy.

        Args:
            display_name: The collection name as shown to users

        Returns:
            Tuple of (actual_collection_name, is_readonly_from_mcp)

        Example:
            ```python
            # System collection display name -> actual name
            actual, readonly = manager.resolve_collection_name("user_preferences")
            # Returns: ("__user_preferences", False)

            # Library collection display name -> actual name
            actual, readonly = manager.resolve_collection_name("library_docs")
            # Returns: ("_library_docs", True)

            # Project collection (no change)
            actual, readonly = manager.resolve_collection_name("my-project-docs")
            # Returns: ("my-project-docs", False)
            ```
        """
        try:
            all_collections = self.client.get_collections()
            all_collection_names = [col.name for col in all_collections.collections]

            if self.type_classifier and COLLECTION_TYPES_AVAILABLE:
                # Use new collection type system for reverse mapping

                # Try system collection (__ prefix)
                system_name = f"__{display_name}"
                if system_name in all_collection_names:
                    collection_info = self.type_classifier.get_collection_info(system_name)
                    return system_name, collection_info.is_readonly

                # Try library collection (_ prefix)
                library_name = f"_{display_name}"
                if library_name in all_collection_names:
                    collection_info = self.type_classifier.get_collection_info(library_name)
                    return library_name, collection_info.is_readonly

                # Check if display name matches actual collection name directly
                if display_name in all_collection_names:
                    collection_info = self.type_classifier.get_collection_info(display_name)
                    return display_name, collection_info.is_readonly

                # Collection doesn't exist - return the display name as-is for error handling
                return display_name, False

            else:
                # Fallback to legacy behavior
                # First, check if this display name corresponds to a library collection
                potential_library_name = f"_{display_name}"

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

    def list_searchable_collections(self) -> list[str]:
        """
        List collections that should be included in global searches.

        This method returns display names for collections that are globally searchable,
        excluding system collections (__ prefix) which are only accessible by explicit name.

        Returns:
            List[str]: Display names of collections that are globally searchable

        Example:
            ```python
            searchable = manager.list_searchable_collections()
            # Returns: ['library_docs', 'project-documents', 'algorithms']
            # Excludes: '__user_preferences' (system collection)
            ```
        """
        try:
            all_collections = self.client.get_collections()
            searchable_collections = []

            for collection in all_collections.collections:
                collection_name = collection.name

                # Use new collection type system to determine searchability
                collection_info = self.type_classifier.get_collection_info(collection_name)

                # Only include globally searchable collections (excludes system collections)
                if collection_info.is_searchable:
                    display_name = self.type_classifier.get_display_name(collection_name)
                    searchable_collections.append(display_name)

            return sorted(searchable_collections)

        except Exception as e:
            logger.error(f"Failed to list searchable collections: {e}")
            return []

    def list_collections_for_project(self, project_name: str) -> list[str]:
        """
        List collections filtered by project context with metadata support.

        This method uses metadata filtering to return only collections that are
        accessible within the specified project context, supporting both legacy
        and multi-tenant collection architectures.

        Args:
            project_name: Project name to filter collections for

        Returns:
            List[str]: Collection names available for the specified project

        Example:
            ```python
            collections = manager.list_collections_for_project("my-project")
            # Returns: ['my-project-docs', 'my-project-notes', 'scratchbook']
            ```
        """
        if not project_name:
            return self.list_workspace_collections()

        try:
            # Import metadata filtering if available
            try:
                from .metadata_filtering import FilterCriteria, MetadataFilterManager

                # Create metadata filter manager
                MetadataFilterManager(self.client)

                # Create filter criteria for project
                FilterCriteria(
                    project_name=project_name,
                    include_global=True,
                    include_shared=True
                )

                # Get all collections and filter by metadata
                all_collections = self.client.get_collections()
                project_collections = []

                for collection in all_collections.collections:
                    collection_name = collection.name

                    # Check if collection belongs to the project
                    if self._collection_belongs_to_project(collection_name, project_name):
                        # Use display name if collection type system is available
                        if hasattr(self, 'type_classifier'):
                            display_name = self.type_classifier.get_display_name(collection_name)
                            project_collections.append(display_name)
                        else:
                            project_collections.append(collection_name)

                logger.debug(
                    f"Found {len(project_collections)} collections for project {project_name}"
                )

                return sorted(project_collections)

            except ImportError:
                raise RuntimeError("Metadata filtering not available - system misconfigured")

        except Exception as e:
            logger.error(f"Failed to list collections for project {project_name}: {e}")
            return self.list_workspace_collections()

    def _collection_belongs_to_project(self, collection_name: str, project_name: str) -> bool:
        """
        Check if a collection belongs to the specified project.

        This method supports both legacy naming patterns and metadata-based
        project association.

        Args:
            collection_name: Name of the collection to check
            project_name: Project name to check against

        Returns:
            bool: True if collection belongs to project, False otherwise
        """
        try:
            # Check project naming patterns
            if collection_name.startswith(f"{project_name}-"):
                return True

            # Check if it's a global/shared collection
            if collection_name in self.config.get("workspace.global_collections", []):
                return True

            # Check system and library collections (available to all projects)
            if collection_name.startswith("__") or collection_name.startswith("_"):
                return True

            # Check if collection matches current project exactly
            if collection_name == project_name:
                return True

            # For multi-tenant collections, we would check metadata here
            # This would require sampling the collection to check project_id metadata
            # For now, we'll use naming patterns

            return False

        except Exception as e:
            logger.debug(f"Error checking collection project membership: {e}")
            return False


    def validate_collection_operation(self, display_name: str, operation: str) -> tuple[bool, str]:
        """
        Validate if an operation is allowed on a collection based on its type.

        Args:
            display_name: The display name of the collection
            operation: The operation to validate ('read', 'write', 'delete', 'create')

        Returns:
            Tuple[bool, str]: (is_valid, reason) where reason explains why if invalid
        """
        try:
            actual_name, is_readonly = self.resolve_collection_name(display_name)

            if self.type_classifier and COLLECTION_TYPES_AVAILABLE:
                # Use new collection type validation
                from ..core.collection_types import validate_collection_operation
                return validate_collection_operation(actual_name, operation)
            else:
                # Standard validation
                valid_operations = {'read', 'write', 'delete', 'create'}
                if operation not in valid_operations:
                    return False, f"Invalid operation '{operation}'. Must be one of: {valid_operations}"

                if operation == 'read':
                    return True, "Read operations are generally allowed"

                if is_readonly and operation in ('write', 'delete'):
                    return False, f"Collection '{display_name}' is read-only via MCP"

                return True, f"Operation '{operation}' is allowed on collection '{display_name}'"

        except Exception as e:
            return False, f"Validation error: {e}"

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

        return model_sizes.get(self.config.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2"), 384)

    async def _optimize_metadata_indexing(
        self, collections: list[CollectionConfig]
    ) -> None:
        """
        Optimize metadata indexing for multi-tenant collections.

        Creates payload indexes for metadata fields used in multi-tenant filtering
        to ensure efficient queries across project boundaries.

        Args:
            collections: List of collection configurations to optimize
        """
        metadata_fields_to_index = [
            "project_name",
            "project_id",
            "tenant_namespace",
            "collection_type",
            "workspace_scope",
            "access_level",
            "created_by"
        ]

        for collection_config in collections:
            # Skip library collections as they have different indexing needs
            if collection_config.collection_type == "library":
                continue

            try:
                # Create payload indexes for efficient metadata filtering
                for field_name in metadata_fields_to_index:
                    try:
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda field=field_name: self.client.create_payload_index(
                                collection_name=collection_config.name,
                                field_name=field,
                                field_schema=models.PayloadSchemaType.KEYWORD
                            )
                        )
                        logger.debug(
                            "Created payload index",
                            collection=collection_config.name,
                            field=field_name
                        )
                    except Exception as index_error:
                        # Index might already exist, log but continue
                        logger.debug(
                            "Payload index creation skipped (may already exist)",
                            collection=collection_config.name,
                            field=field_name,
                            error=str(index_error)
                        )

                logger.info(
                    "Metadata indexing optimized",
                    collection=collection_config.name,
                    indexed_fields=metadata_fields_to_index
                )

            except Exception as e:
                logger.warning(
                    "Failed to optimize metadata indexing",
                    collection=collection_config.name,
                    error=str(e)
                )

    async def _create_collection_with_metadata_support(
        self, collection_config: CollectionConfig, project_context: dict | None = None
    ) -> None:
        """
        Create a collection with metadata support for multi-tenant architecture.

        This method extends the basic collection creation with metadata indexing
        and project context injection.

        Args:
            collection_config: Collection configuration
            project_context: Optional project context for metadata enrichment

        Raises:
            ResponseHandlingException: If Qdrant API calls fail
            ValueError: If configuration is invalid
        """
        try:
            # Create the basic collection first
            self._ensure_collection_exists(collection_config)

            # Add metadata indexing for multi-tenant support
            if project_context:
                await self._optimize_metadata_indexing([collection_config])

                logger.info(
                    "Collection created with metadata support",
                    collection=collection_config.name,
                    project_context=project_context
                )

        except Exception as e:
            logger.error(
                f"Failed to create collection with metadata support: {e}"
            )
            raise


class MemoryCollectionManager:
    """
    Manages memory collections with proper access controls and auto-creation.

    This class handles both system memory collections (__memory_*) and project memory
    collections ({project}-memory) with appropriate access control enforcement:

    - System memory collections: CLI-writable only, LLM read-only
    - Project memory collections: MCP read-write access
    - Memory collections cannot be deleted by LLM
    - Auto-creation functionality for missing memory collections

    The manager integrates with the existing access control system and ensures
    that memory collections appear in appropriate search scopes.
    """

    def __init__(self, workspace_client: QdrantClient, config: ConfigManager) -> None:
        """
        Initialize the memory collection manager.

        Args:
            workspace_client: Configured Qdrant client instance
            config: Configuration object containing workspace and memory settings
        """
        self.workspace_client = workspace_client
        self.config = config
        self.naming_manager = CollectionNamingManager(
            config.get("workspace.global_collections", []),
            config.get("workspace.collection_types", []),
        )
        self.type_classifier = CollectionTypeClassifier()

        # Memory collection configuration
        memory_collection = config.get("memory_collection", "memory")
        if not isinstance(memory_collection, str) or not memory_collection:
            memory_collection = "memory"
        self.memory_collection = memory_collection

    async def ensure_memory_collections_exist(self, project: str) -> dict:
        """
        Ensure both system and project memory collections exist.

        Creates missing memory collections with proper access controls:
        - System memory: __{memory_collection_name} (CLI-only writable)
        - Project memory: {project}-{memory_collection_name} (MCP read-write)

        Args:
            project: Project name for project-scoped memory collection

        Returns:
            dict: Results of collection creation with keys:
                - system_memory: Creation result for system memory collection (if created)
                - project_memory: Creation result for project memory collection (if created)
                - existing: List of collections that already existed
        """
        results = {
            'existing': [],
            'created': []
        }

        # Build collection names
        system_memory = build_system_memory_collection_name(self.memory_collection)
        project_memory = build_user_collection_name(project, self.memory_collection)

        logger.info(f"Ensuring memory collections exist: system='{system_memory}', project='{project_memory}'")

        # Check and create system memory collection if missing
        if not self.collection_exists(system_memory):
            logger.info(f"Creating system memory collection: {system_memory}")
            system_result = self.create_system_memory_collection(system_memory)
            results['system_memory'] = system_result
            results['created'].append(system_memory)
        else:
            results['existing'].append(system_memory)
            logger.debug(f"System memory collection already exists: {system_memory}")

        # Check and create project memory collection if missing
        if not self.collection_exists(project_memory):
            logger.info(f"Creating project memory collection: {project_memory}")
            project_result = self.create_project_memory_collection(project_memory)
            results['project_memory'] = project_result
            results['created'].append(project_memory)
        else:
            results['existing'].append(project_memory)
            logger.debug(f"Project memory collection already exists: {project_memory}")

        return results

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Qdrant.

        Args:
            collection_name: Name of the collection to check

        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            collections = self.workspace_client.get_collections()
            existing_names = {col.name for col in collections.collections}
            return collection_name in existing_names
        except Exception as e:
            logger.error(f"Error checking collection existence for '{collection_name}': {e}")
            return False

    def create_system_memory_collection(self, collection_name: str) -> dict:
        """
        Create a system memory collection with CLI-only write access.

        System memory collections use the __ prefix and are configured as:
        - CLI-writable only (LLM cannot write)
        - LLM-readable for context retrieval
        - Not globally searchable (explicit access only)

        Args:
            collection_name: Full collection name including __ prefix

        Returns:
            dict: Creation result with collection info and access control settings

        Raises:
            ValueError: If collection name doesn't follow system memory pattern
            ResponseHandlingException: If Qdrant creation fails
        """
        # Validate system memory collection name pattern
        if not collection_name.startswith("__"):
            raise ValueError(f"System memory collection must start with '__': {collection_name}")

        # Validate against access control
        try:
            validate_llm_collection_access('create', collection_name, self.config)
        except LLMAccessControlError as e:
            # This is expected for system collections - CLI can create them
            logger.debug(f"LLM access control validation failed as expected for system collection: {e}")

        try:
            # Create collection with standard memory configuration
            collection_config = CollectionConfig(
                name=collection_name,
                description=f"System memory collection: {collection_name[2:]}",  # Remove __ prefix for description
                collection_type="system_memory",
                project_name=None,
                vector_size=self._get_vector_size(),
                enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
            )

            self._create_memory_collection(collection_config)

            result = {
                'collection_name': collection_name,
                'type': 'system_memory',
                'access_control': {
                    'cli_writable': True,
                    'llm_writable': False,
                    'llm_readable': True,
                    'mcp_readable': True,
                    'globally_searchable': False
                },
                'description': collection_config.description,
                'status': 'created'
            }

            logger.info(f"Successfully created system memory collection: {collection_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to create system memory collection '{collection_name}': {e}")
            raise

    def create_project_memory_collection(self, collection_name: str) -> dict:
        """
        Create a project memory collection with MCP read-write access.

        Project memory collections are configured as:
        - MCP read-write access (LLM can read and write)
        - Globally searchable for project context
        - Standard project collection permissions

        Args:
            collection_name: Full collection name in project-memory format

        Returns:
            dict: Creation result with collection info and access control settings

        Raises:
            ValueError: If collection name doesn't follow project memory pattern
            ResponseHandlingException: If Qdrant creation fails
        """
        # Validate project memory collection name pattern
        if not collection_name.endswith("-memory"):
            raise ValueError(f"Project memory collection must end with '-memory': {collection_name}")

        # Extract project name
        project_name = collection_name[:-7]  # Remove "-memory" suffix

        try:
            # Create collection with standard memory configuration
            collection_config = CollectionConfig(
                name=collection_name,
                description=f"Project memory collection for {project_name}",
                collection_type="project_memory",
                project_name=project_name,
                vector_size=self._get_vector_size(),
                enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
            )

            self._create_memory_collection(collection_config)

            result = {
                'collection_name': collection_name,
                'type': 'project_memory',
                'project_name': project_name,
                'access_control': {
                    'cli_writable': True,
                    'llm_writable': True,
                    'llm_readable': True,
                    'mcp_readable': True,
                    'mcp_writable': True,
                    'globally_searchable': True
                },
                'description': collection_config.description,
                'status': 'created'
            }

            logger.info(f"Successfully created project memory collection: {collection_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to create project memory collection '{collection_name}': {e}")
            raise

    def _create_memory_collection(self, collection_config: CollectionConfig) -> None:
        """
        Create a memory collection with optimized settings.

        Memory collections are optimized for:
        - Fast retrieval of contextual information
        - Efficient storage of user preferences and settings
        - Quick search across stored memory items

        Args:
            collection_config: Complete configuration for the memory collection

        Raises:
            ResponseHandlingException: If Qdrant API calls fail
        """
        try:
            # Create collection with memory-optimized settings
            if collection_config.enable_sparse_vectors:
                # Memory collections with both dense and sparse vectors for hybrid search
                self.workspace_client.create_collection(
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
                self.workspace_client.create_collection(
                    collection_name=collection_config.name,
                    vectors_config=models.VectorParams(
                        size=collection_config.vector_size,
                        distance=getattr(
                            models.Distance, collection_config.distance_metric.upper()
                        ),
                    ),
                )

            # Apply memory-optimized collection settings
            self.workspace_client.update_collection(
                collection_name=collection_config.name,
                optimizer_config=models.OptimizersConfigDiff(
                    default_segment_number=1,  # Smaller segments for memory collections
                    memmap_threshold=10000,    # Lower threshold for faster access
                ),
                hnsw_config=models.HnswConfigDiff(
                    m=24,  # Higher connectivity for better recall
                    ef_construct=200,  # Better index quality for memory retrieval
                    full_scan_threshold=5000,  # Lower threshold for small collections
                ),
            )

        except Exception as e:
            logger.error(f"Failed to create memory collection '{collection_config.name}': {e}")
            raise

    def get_memory_collections(self, project: str) -> dict:
        """
        Get information about memory collections for a project.

        Args:
            project: Project name to get memory collections for

        Returns:
            dict: Information about system and project memory collections
        """
        system_memory = build_system_memory_collection_name(self.memory_collection)
        project_memory = build_user_collection_name(project, self.memory_collection)

        try:
            collections = self.workspace_client.get_collections()
            existing_names = {col.name for col in collections.collections}

            return {
                'system_memory': {
                    'name': system_memory,
                    'display_name': system_memory[2:],  # Remove __ prefix for display
                    'exists': system_memory in existing_names,
                    'access_control': {
                        'cli_writable': True,
                        'llm_writable': False,
                        'llm_readable': True,
                        'mcp_readable': True,
                        'globally_searchable': False
                    }
                },
                'project_memory': {
                    'name': project_memory,
                    'display_name': project_memory,
                    'exists': project_memory in existing_names,
                    'project_name': project,
                    'access_control': {
                        'cli_writable': True,
                        'llm_writable': True,
                        'llm_readable': True,
                        'mcp_readable': True,
                        'mcp_writable': True,
                        'globally_searchable': True
                    }
                }
            }

        except Exception as e:
            logger.error(f"Failed to get memory collection info for project '{project}': {e}")
            return {
                'system_memory': {'name': system_memory, 'exists': False, 'error': str(e)},
                'project_memory': {'name': project_memory, 'exists': False, 'error': str(e)}
            }

    def validate_memory_collection_access(self, collection_name: str, operation: str) -> tuple[bool, str]:
        """
        Validate access to memory collections based on their type.

        Args:
            collection_name: Name of the memory collection
            operation: Operation to validate ('read', 'write', 'delete')

        Returns:
            tuple[bool, str]: (is_allowed, reason)
        """
        collection_info = self.type_classifier.get_collection_info(collection_name)

        # Memory collections cannot be deleted by LLM regardless of type
        if operation == 'delete':
            return False, f"Memory collection '{collection_name}' cannot be deleted by LLM"

        # System memory collections are read-only from LLM/MCP
        if collection_info.type == CollectionType.SYSTEM:
            if operation == 'write':
                return False, f"System memory collection '{collection_name}' is CLI-writable only"
            elif operation == 'read':
                return True, "Read access allowed for system memory collection"

        # Project memory collections allow read/write from MCP
        elif collection_info.type == CollectionType.PROJECT:
            if operation in ['read', 'write']:
                return True, f"{operation.title()} access allowed for project memory collection"

        return False, f"Unknown memory collection type for '{collection_name}'"

    def _get_vector_size(self) -> int:
        """
        Get the vector dimension size for the currently configured embedding model.

        Returns:
            int: Vector dimension size for the configured model
        """
        # This mirrors the implementation in WorkspaceCollectionManager
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

        return model_sizes.get(self.config.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2"), 384)


class CollectionSelector:
    """
    Enhanced collection selector for multi-tenant workspace collections.

    This class provides intelligent collection selection that distinguishes between
    memory_collection and code_collection types, integrates with the multi-tenant
    architecture, and provides robust fallback mechanisms.

    Key Features:
        - Distinguishes memory vs code collection types
        - Integrates with WorkspaceCollectionRegistry
        - Supports project-aware collection discovery
        - Provides fallback mechanisms for collection selection
        - Works with metadata-based tenant isolation
    """

    def __init__(self, client: QdrantClient, config: ConfigManager, project_detector=None):
        """Initialize the collection selector with required dependencies."""
        self.client = client
        self.config = config
        self.project_detector = project_detector

        # Initialize core managers
        self.workspace_manager = WorkspaceCollectionManager(client, config)
        self.memory_manager = MemoryCollectionManager(client, config)
        self.type_classifier = CollectionTypeClassifier(config)
        self.naming_manager = CollectionNamingManager()

        # Import multi-tenant components
        from .multitenant_collections import (
            ProjectIsolationManager,
            WorkspaceCollectionRegistry,
        )
        self.registry = WorkspaceCollectionRegistry()
        self.isolation_manager = ProjectIsolationManager()

    def select_collections_by_type(
        self,
        collection_type: str,
        project_name: str | None = None,
        include_shared: bool = True,
        workspace_types: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Select collections based on type with multi-tenant support.

        Args:
            collection_type: 'memory_collection' or 'code_collection'
            project_name: Project context for filtering
            include_shared: Include shared workspace collections
            workspace_types: Specific workspace types to include

        Returns:
            Dict with selected collections categorized by scope
        """
        try:
            result = {
                'memory_collections': [],
                'code_collections': [],
                'shared_collections': [],
                'project_collections': [],
                'fallback_collections': []
            }

            # Auto-detect project if not provided
            if not project_name and self.project_detector:
                project_info = self.project_detector.get_project_info()
                project_name = project_info.get("main_project")

            all_collections = self._get_all_collections_with_metadata()

            if collection_type == 'memory_collection':
                result['memory_collections'] = self._select_memory_collections(
                    all_collections, project_name, include_shared
                )

            elif collection_type == 'code_collection':
                result['code_collections'] = self._select_code_collections(
                    all_collections, project_name, workspace_types, include_shared
                )

            else:
                # Mixed selection - include both types
                result['memory_collections'] = self._select_memory_collections(
                    all_collections, project_name, include_shared
                )
                result['code_collections'] = self._select_code_collections(
                    all_collections, project_name, workspace_types, include_shared
                )

            # Add shared collections if requested
            if include_shared:
                result['shared_collections'] = self._select_shared_collections(
                    all_collections, workspace_types
                )

            # Apply fallback mechanism if no collections found
            if self._is_result_empty(result):
                result['fallback_collections'] = self._apply_fallback_selection(
                    collection_type, project_name
                )

            return result

        except Exception as e:
            logger.error(f"Collection selection failed: {e}")
            return self._get_empty_result_with_fallback(collection_type)

    def _select_memory_collections(
        self,
        all_collections: list[dict],
        project_name: str | None,
        include_shared: bool
    ) -> list[str]:
        """Select memory collections with project context."""
        memory_collections = []

        for collection in all_collections:
            collection_name = collection['name']
            collection_info = collection.get('info', {})

            # Check if it's a memory collection
            if self._is_memory_collection(collection_name, collection_info):
                # System memory collections (CLI-managed)
                if collection_name.startswith('__'):
                    if include_shared:
                        memory_collections.append(collection_name)

                # Project-specific memory collections
                elif project_name and collection_info.get('project_name') == project_name:
                    memory_collections.append(collection_name)

                # Legacy memory collection
                elif collection_name == 'memory':
                    if include_shared:
                        memory_collections.append(collection_name)

        return memory_collections

    def _select_code_collections(
        self,
        all_collections: list[dict],
        project_name: str | None,
        workspace_types: list[str] | None,
        include_shared: bool
    ) -> list[str]:
        """Select code collections with workspace type filtering."""
        code_collections = []
        target_types = workspace_types or list(self.registry.get_workspace_types())

        for collection in all_collections:
            collection_name = collection['name']
            collection_info = collection.get('info', {})

            # Skip memory collections
            if self._is_memory_collection(collection_name, collection_info):
                continue

            # Project-specific collections
            if project_name:
                for workspace_type in target_types:
                    expected_name = f"{project_name}-{workspace_type}"

                    if collection_name == expected_name:
                        code_collections.append(collection_name)
                        break

            # Library collections (user-defined with _ prefix)
            if collection_name.startswith('_') and not collection_name.startswith('__'):
                if include_shared:
                    code_collections.append(collection_name)

            # Legacy collections that don't match patterns
            elif self._is_legacy_code_collection(collection_name, collection_info):
                if include_shared:
                    code_collections.append(collection_name)

        return code_collections

    def _select_shared_collections(
        self,
        all_collections: list[dict],
        workspace_types: list[str] | None
    ) -> list[str]:
        """Select shared collections across projects."""
        shared_collections = []
        target_types = workspace_types or list(self.registry.get_workspace_types())

        for collection in all_collections:
            collection_name = collection['name']
            collection_info = collection.get('info', {})

            # Shared workspace collections like 'scratchbook'
            if collection_name in target_types:
                shared_collections.append(collection_name)

            # Collections with shared scope in metadata
            elif collection_info.get('workspace_scope') == 'shared':
                shared_collections.append(collection_name)

        return shared_collections

    def _is_memory_collection(self, collection_name: str, collection_info: dict) -> bool:
        """Check if collection is a memory collection."""
        # System memory collections
        if collection_name.startswith('__'):
            return True

        # Legacy memory collection
        if collection_name == 'memory':
            return True

        # Collections with memory collection type in metadata
        if collection_info.get('collection_type') == 'memory':
            return True

        return False

    def _is_legacy_code_collection(self, collection_name: str, collection_info: dict) -> bool:
        """Check if collection is a legacy code collection."""
        # Skip reserved patterns
        if (collection_name.startswith('_') or
            collection_name in self.naming_manager.RESERVED_NAMES or
            collection_name == 'memory'):
            return False

        # Check if it's a known workspace type
        if collection_name in self.registry.get_workspace_types():
            return True

        # Check metadata
        collection_type = collection_info.get('collection_type', 'legacy')
        return collection_type not in ['memory', 'system']

    def _apply_fallback_selection(
        self,
        collection_type: str,
        project_name: str | None
    ) -> list[str]:
        """Apply fallback mechanism when no collections are found."""
        fallback_collections = []

        try:
            # Try to get basic collection list
            available_collections = self._get_basic_collection_list()

            if collection_type == 'memory_collection':
                # Fallback to any memory-like collections
                for collection_name in available_collections:
                    if ('memory' in collection_name.lower() or
                        collection_name.startswith('__')):
                        fallback_collections.append(collection_name)

            elif collection_type == 'code_collection':
                # Fallback to workspace collections or legacy collections
                for collection_name in available_collections:
                    if not self._is_memory_collection(collection_name, {}):
                        fallback_collections.append(collection_name)

            else:
                # Mixed fallback - include all non-reserved collections
                for collection_name in available_collections:
                    if collection_name not in self.naming_manager.RESERVED_NAMES:
                        fallback_collections.append(collection_name)

            logger.warning(
                "Applied fallback collection selection",
                collection_type=collection_type,
                project_name=project_name,
                fallback_count=len(fallback_collections)
            )

        except Exception as e:
            logger.error(f"Fallback selection failed: {e}")

        return fallback_collections

    def _get_all_collections_with_metadata(self) -> list[dict]:
        """Get all collections with their metadata information."""
        collections_with_metadata = []

        try:
            # Get basic collection list
            collection_names = self._get_basic_collection_list()

            for collection_name in collection_names:
                try:
                    # Get collection info including metadata
                    collection_info = self.client.get_collection(collection_name)

                    # Extract metadata from collection if available
                    metadata = {}
                    if hasattr(collection_info, 'config') and collection_info.config:
                        metadata = getattr(collection_info.config, 'params', {})

                    collections_with_metadata.append({
                        'name': collection_name,
                        'info': metadata
                    })

                except Exception as e:
                    # Include collection with minimal info if detailed fetch fails
                    logger.debug(f"Could not get detailed info for collection {collection_name}: {e}")
                    collections_with_metadata.append({
                        'name': collection_name,
                        'info': {}
                    })

        except Exception as e:
            logger.error(f"Failed to get collections with metadata: {e}")

        return collections_with_metadata

    def _get_basic_collection_list(self) -> list[str]:
        """Get basic list of collection names with error handling."""
        try:
            collections = self.client.get_collections()
            return [collection.name for collection in collections.collections]
        except Exception as e:
            logger.error(f"Failed to get basic collection list: {e}")
            return []

    def _is_result_empty(self, result: dict[str, list[str]]) -> bool:
        """Check if selection result is empty."""
        return all(not collections for collections in result.values())

    def _get_empty_result_with_fallback(self, collection_type: str) -> dict[str, list[str]]:
        """Get empty result structure with basic fallback."""
        return {
            'memory_collections': [],
            'code_collections': [],
            'shared_collections': [],
            'project_collections': [],
            'fallback_collections': self._apply_fallback_selection(collection_type, None)
        }

    def get_searchable_collections(
        self,
        project_name: str | None = None,
        workspace_types: list[str] | None = None,
        include_memory: bool = False,
        include_shared: bool = True
    ) -> list[str]:
        """
        Get collections suitable for search operations.

        Args:
            project_name: Project context
            workspace_types: Specific workspace types to include
            include_memory: Whether to include memory collections
            include_shared: Whether to include shared collections

        Returns:
            List of collection names suitable for search
        """
        searchable = []

        # Get code collections (primary search targets)
        code_selection = self.select_collections_by_type(
            'code_collection',
            project_name=project_name,
            workspace_types=workspace_types,
            include_shared=include_shared
        )
        searchable.extend(code_selection.get('code_collections', []))
        searchable.extend(code_selection.get('shared_collections', []))

        # Add memory collections if requested
        if include_memory:
            memory_selection = self.select_collections_by_type(
                'memory_collection',
                project_name=project_name,
                include_shared=include_shared
            )
            # Only include memory collections marked as searchable in registry
            memory_collections = memory_selection.get('memory_collections', [])
            for collection_name in memory_collections:
                if self._is_memory_collection_searchable(collection_name):
                    searchable.append(collection_name)

        # Apply fallback if no searchable collections found
        if not searchable:
            fallback_selection = self.select_collections_by_type(
                'mixed',
                project_name=project_name,
                workspace_types=workspace_types,
                include_shared=include_shared
            )
            searchable.extend(fallback_selection.get('fallback_collections', []))

        return list(set(searchable))  # Remove duplicates

    def _is_memory_collection_searchable(self, collection_name: str) -> bool:
        """Check if a memory collection should be included in search."""
        # System memory collections are generally not searchable
        if collection_name.startswith('__'):
            return False

        # Legacy memory collection is searchable
        if collection_name == 'memory':
            return True

        # Check registry for memory collection searchability
        return self.registry.is_searchable('memory')

    def validate_collection_access(
        self,
        collection_name: str,
        operation: str,
        project_context: str | None = None
    ) -> tuple[bool, str]:
        """
        Validate access to a collection for a given operation.

        Args:
            collection_name: Name of the collection
            operation: Operation to validate ('read', 'write', 'delete')
            project_context: Project context for validation

        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Use existing validation logic from managers
            self.naming_manager.get_collection_info(collection_name)

            # Memory collections have special validation rules
            if self._is_memory_collection(collection_name, {}):
                return self.memory_manager.validate_memory_collection_access(
                    collection_name, operation
                )

            # Use workspace manager validation for code collections
            return self.workspace_manager.validate_collection_operation(
                collection_name, operation
            )

        except Exception as e:
            logger.error(f"Collection access validation failed: {e}")
            return False, f"Validation error: {e}"
