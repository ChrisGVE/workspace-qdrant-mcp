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

    # No config parameter needed - uses get_config() internally
    client = QdrantWorkspaceClient()
    await client.initialize()

    # Client automatically detects project and creates collections
    collections = client.list_collections()
    status = await client.get_status()
    ```
"""

import asyncio

from loguru import logger
from qdrant_client import QdrantClient

from .collections import MemoryCollectionManager, WorkspaceCollectionManager
from .config import get_config_dict, get_config_string
from .embeddings import EmbeddingService
from .ssl_config import create_secure_qdrant_config, get_ssl_manager

# Import LLM access control system
try:
    from .llm_access_control import (
        LLMAccessControlError,
        validate_llm_collection_access,
    )
except ImportError:
    # Fallback for direct imports when not used as a package
    pass

# logger imported from loguru


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
        project_detector (Optional[ProjectDetector]): Project structure detection
        project_info (Optional[Dict]): Detected project information
        initialized (bool): Whether the client has been initialized

    Example:
        ```python
        # No config parameter needed - uses get_config() internally
        workspace_client = QdrantWorkspaceClient()

        # Initialize connections and detect project structure
        await workspace_client.initialize()

        # Use the client for operations
        status = await workspace_client.get_status()
        collections = workspace_client.list_collections()

        # Clean up when done
        await workspace_client.close()
        ```
    """

    def __init__(self) -> None:
        """Initialize the workspace client with lua-style configuration access.

        Configuration is accessed directly through get_config() functions
        without requiring a ConfigManager instance to be passed.
        """
        self.client: QdrantClient | None = None
        self.collection_manager: WorkspaceCollectionManager | None = None
        self.memory_collection_manager: MemoryCollectionManager | None = None
        self.embedding_service = EmbeddingService()
        # Lazy import to avoid circular dependency
        self.project_detector = None
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
            client = QdrantWorkspaceClient()  # No config needed
            await client.initialize()  # Safe to call multiple times
            ```
        """
        if self.initialized:
            return

        try:
            # Create secure Qdrant client configuration with context-aware SSL handling
            ssl_manager = get_ssl_manager()

            # Determine environment from config or fall back to development
            environment = get_config_string("deployment.environment", "development")

            # Get authentication credentials from config if available
            auth_token = get_config_string("security.qdrant_auth_token")
            api_key = get_config_string("security.qdrant_api_key")

            # Create secure client configuration
            secure_config = create_secure_qdrant_config(
                base_config=get_config_dict("qdrant_client_config", {}),
                url=get_config_string("qdrant.url", "http://localhost:6333"),
                environment=environment,
                auth_token=auth_token,
                api_key=api_key,
            )

            # Create client with comprehensive SSL warning suppression
            import warnings

            import urllib3

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*insecure connection.*", category=urllib3.exceptions.InsecureRequestWarning)
                warnings.filterwarnings("ignore", message=".*unverified HTTPS request.*", category=urllib3.exceptions.InsecureRequestWarning)
                warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)

                from .ssl_config import suppress_qdrant_ssl_warnings
                with suppress_qdrant_ssl_warnings():
                    self.client = QdrantClient(**secure_config)

            # Test connection with SSL warning suppression
            def get_collections_with_suppression():
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*Api key is used with an insecure connection.*", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*insecure connection.*", category=urllib3.exceptions.InsecureRequestWarning)
                    warnings.filterwarnings("ignore", message=".*unverified HTTPS request.*", category=urllib3.exceptions.InsecureRequestWarning)
                    warnings.filterwarnings("ignore", message=".*SSL.*", category=UserWarning)
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    return self.client.get_collections()

            if (
                ssl_manager.is_localhost_url(get_config_string("qdrant.url", "http://localhost:6333"))
                and environment == "development"
            ):
                with ssl_manager.for_localhost():
                    await asyncio.get_event_loop().run_in_executor(
                        None, get_collections_with_suppression
                    )
            else:
                await asyncio.get_event_loop().run_in_executor(
                    None, get_collections_with_suppression
                )

            logger.info("Connected to Qdrant at %s", self.config.get("qdrant.url", "http://localhost:6333"))

            # Initialize project detector (lazy import to avoid circular dependency)
            if self.project_detector is None:
                from ..utils.project_detection import ProjectDetector
                self.project_detector = ProjectDetector(
                    github_user=self.config.get("workspace.github_user")
                )

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
                "qdrant_url": self.config.get("qdrant.url", "http://localhost:6333"),
                "collections_count": len(info.collections),
                "workspace_collections": workspace_collections,
                "current_project": self.project_info["main_project"]
                if self.project_info
                else None,
                "project_info": self.project_info,
                "collection_info": collection_info,
                "embedding_info": self.embedding_service.get_model_info(),
                "config": {
                    "embedding_model": self.config.get("embedding.model", "sentence-transformers/all-MiniLM-L6-v2"),
                    "sparse_vectors_enabled": self.config.get("embedding.enable_sparse_vectors", True),
                    "global_collections": self.config.get("workspace.global_collections", []),
                },
            }

        except Exception as e:
            logger.error("Failed to get status: %s", e)
            return {"error": f"Failed to get status: {e}"}

    def list_collections(self) -> list[str]:
        """List all available workspace collections for the current project.

        Returns collections that are accessible within the current workspace scope,
        including project-specific collections and global collections. Uses metadata
        filtering to provide project-aware collection discovery.

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
            # Use the enhanced collection manager with project filtering
            project_context = self.get_project_context()
            if project_context and hasattr(self.collection_manager, 'list_collections_for_project'):
                return self.collection_manager.list_collections_for_project(
                    project_context.get("project_name")
                )
            else:
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

    def get_project_context(self, collection_type: str = "general") -> dict | None:
        """Get project context for metadata filtering.

        Args:
            collection_type: Type of collection for context (notes, docs, etc.)

        Returns:
            Dict with project context metadata for filtering, or None if no project detected
        """
        if not self.project_info or not self.project_info.get("main_project"):
            return None

        project_name = self.project_info["main_project"]

        return {
            "project_name": project_name,
            "project_id": self._generate_project_id(project_name),
            "tenant_namespace": f"{project_name}.{collection_type}",
            "collection_type": collection_type,
            "workspace_scope": "project"
        }

    def _generate_project_id(self, project_name: str) -> str:
        """Generate stable project ID from project name.

        Args:
            project_name: Project name to generate ID for

        Returns:
            Stable 12-character project ID
        """
        import hashlib
        return hashlib.sha256(project_name.encode()).hexdigest()[:12]

    def refresh_project_detection(self) -> dict:
        """Refresh project detection from current working directory.

        Re-analyzes the current directory structure and Git repository
        to update project information. Useful when the working directory
        changes or project structure is modified.

        Returns:
            Dict: Updated project information with same structure as get_project_info()
        """
        # Ensure project detector is initialized
        if self.project_detector is None:
            from ..utils.project_detection import ProjectDetector
            self.project_detector = ProjectDetector(
                github_user=self.config.get("workspace.github_user")
            )

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
        already exist. It integrates with the multi-tenant metadata schema and
        supports both legacy and multi-tenant collection patterns.

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
            try:
                from .collection_naming_validation import CollectionNamingValidator
                from .metadata_schema import MultiTenantMetadataSchema

                # Validate collection name using new validation system
                validator = CollectionNamingValidator()
                validation_result = validator.validate_name(
                    collection_name,
                    existing_collections=self.list_collections()
                )

                if not validation_result.is_valid:
                    raise ValueError(f"Invalid collection name '{collection_name}': {validation_result.error_message}")
            except ImportError:
                # Fallback to basic validation if new components not available
                logger.warning("Multi-tenant validation components not available, using legacy validation")

            # Get project context for metadata enrichment
            project_context = self.get_project_context(collection_type)

            # Create collection configuration with multi-tenant support
            collection_config = CollectionConfig(
                name=collection_name,
                description=f"{collection_type.title()} collection",
                collection_type=collection_type,
                project_name=project_context.get("project_name") if project_context else None,
                vector_size=self.collection_manager._get_vector_size(),
                enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
            )

            # Use the collection manager to ensure the collection exists
            self.collection_manager._ensure_collection_exists(collection_config)

            logger.info("Ensured collection exists: %s", collection_name)

        except Exception as e:
            logger.error(
                "Failed to ensure collection %s exists: %s", collection_name, e
            )
            raise RuntimeError(
                f"Failed to ensure collection '{collection_name}' exists: {e}"
            ) from e

    async def search_with_project_context(
        self,
        collection_name: str,
        query_embeddings: dict,
        collection_type: str = "general",
        limit: int = 10,
        fusion_method: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        additional_filters: dict | None = None,
        include_shared: bool = True,
        **search_kwargs
    ) -> dict:
        """Perform hybrid search with automatic project context filtering.

        This method automatically injects project metadata filters while maintaining
        the full functionality of the hybrid search engine.

        Args:
            collection_name: Name of collection to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            collection_type: Type of collection for context filtering
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            additional_filters: Additional metadata filters to apply
            include_shared: Whether to include shared workspace resources
            **search_kwargs: Additional search parameters passed to hybrid_search

        Returns:
            Dict: Search results with project context applied

        Example:
            ```python
            # Search project docs with automatic context filtering
            results = await client.search_with_project_context(
                collection_name="docs",
                query_embeddings=embeddings,
                collection_type="docs",
                limit=5
            )
            ```
        """
        if not self.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Import HybridSearchEngine here to avoid circular imports
            from .hybrid_search import HybridSearchEngine

            # Get project context for metadata filtering
            project_context = self.get_project_context(collection_type)

            # Adjust workspace scope based on include_shared parameter
            if project_context and include_shared:
                project_context["include_shared"] = True

            # Create hybrid search engine
            search_engine = HybridSearchEngine(self.client)

            # Build additional filters from dict to Qdrant Filter if provided
            base_filter = None
            if additional_filters:
                from qdrant_client.http import models
                conditions = []
                for key, value in additional_filters.items():
                    if isinstance(value, str):
                        conditions.append(
                            models.FieldCondition(key=key, match=models.MatchValue(value=value))
                        )
                    elif isinstance(value, (int, float)):
                        conditions.append(
                            models.FieldCondition(key=key, match=models.MatchValue(value=value))
                        )
                    elif isinstance(value, list):
                        conditions.append(
                            models.FieldCondition(key=key, match=models.MatchAny(any=value))
                        )

                if conditions:
                    base_filter = models.Filter(must=conditions)

            # Perform search with project context
            search_results = await search_engine.hybrid_search(
                collection_name=collection_name,
                query_embeddings=query_embeddings,
                limit=limit,
                fusion_method=fusion_method,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                filter_conditions=base_filter,
                project_context=project_context,
                auto_inject_metadata=True,
                **search_kwargs
            )

            # Enrich results with project context information
            if "fused_results" in search_results:
                search_results["project_context"] = project_context
                search_results["collection_type"] = collection_type
                search_results["include_shared"] = include_shared

            logger.info(
                "Project context search completed",
                collection=collection_name,
                project_name=project_context.get("project_name") if project_context else None,
                results_count=len(search_results.get("fused_results", []))
            )

            return search_results

        except Exception as e:
            logger.error("Project context search failed: %s", e)
            return {"error": f"Project context search failed: {e}"}

    def get_enhanced_collection_selector(self):
        """Get an instance of the enhanced collection selector.

        Returns:
            CollectionSelector: Configured selector for multi-tenant collections
        """
        if not self.initialized:
            raise RuntimeError("Client must be initialized before using collection selector")

        # Ensure project detector is initialized
        if self.project_detector is None:
            from ..utils.project_detection import ProjectDetector
            self.project_detector = ProjectDetector(
                github_user=self.config.get("workspace.github_user")
            )

        from .collections import CollectionSelector
        return CollectionSelector(self.client, self.config, self.project_detector)

    def select_collections_by_type(
        self,
        collection_type: str,
        project_name: str | None = None,
        include_shared: bool = True,
        workspace_types: list[str] | None = None
    ) -> dict[str, list[str]]:
        """
        Select collections by type using enhanced multi-tenant selector.

        Args:
            collection_type: 'memory_collection', 'code_collection', or 'mixed'
            project_name: Project context (auto-detected if None)
            include_shared: Include shared workspace collections
            workspace_types: Specific workspace types to include

        Returns:
            Dict with selected collections categorized by scope
        """
        if not self.initialized:
            return {
                'memory_collections': [],
                'code_collections': [],
                'shared_collections': [],
                'project_collections': [],
                'fallback_collections': []
            }

        try:
            selector = self.get_enhanced_collection_selector()
            return selector.select_collections_by_type(
                collection_type=collection_type,
                project_name=project_name,
                include_shared=include_shared,
                workspace_types=workspace_types
            )
        except Exception as e:
            logger.error(f"Enhanced collection selection failed: {e}")
            return {
                'memory_collections': [],
                'code_collections': [],
                'shared_collections': [],
                'project_collections': [],
                'fallback_collections': []
            }

    def get_searchable_collections(
        self,
        project_name: str | None = None,
        workspace_types: list[str] | None = None,
        include_memory: bool = False,
        include_shared: bool = True
    ) -> list[str]:
        """
        Get collections suitable for search operations with enhanced selection.

        Args:
            project_name: Project context (auto-detected if None)
            workspace_types: Specific workspace types to include
            include_memory: Whether to include memory collections
            include_shared: Whether to include shared collections

        Returns:
            List of collection names suitable for search
        """
        if not self.initialized:
            return []

        try:
            selector = self.get_enhanced_collection_selector()
            return selector.get_searchable_collections(
                project_name=project_name,
                workspace_types=workspace_types,
                include_memory=include_memory,
                include_shared=include_shared
            )
        except Exception as e:
            logger.error(f"Enhanced searchable collection selection failed: {e}")
            # Fallback to original logic
            return self.list_collections()

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
        if not self.initialized:
            return False, "Client not initialized"

        try:
            selector = self.get_enhanced_collection_selector()
            return selector.validate_collection_access(
                collection_name=collection_name,
                operation=operation,
                project_context=project_context
            )
        except Exception as e:
            logger.error(f"Collection access validation failed: {e}")
            return False, f"Validation error: {e}"

    async def create_collection(
        self,
        collection_name: str,
        collection_type: str = "general",
        project_metadata: dict | None = None
    ) -> dict:
        """Create a new collection with multi-tenant metadata support.

        This method creates a collection using the multi-tenant architecture
        with proper metadata indexing and project isolation.

        Args:
            collection_name: Name of the collection to create
            collection_type: Type/category of the collection
            project_metadata: Optional project context metadata

        Returns:
            Dict: Creation result with success status and metadata

        Example:
            ```python
            result = await client.create_collection(
                collection_name="my-project-docs",
                collection_type="docs",
                project_metadata={"project_name": "my-project"}
            )
            ```
        """
        if not self.initialized:
            return {"error": "Client not initialized"}

        try:
            # Import multi-tenant components if available
            try:
                from .collection_naming_validation import CollectionNamingValidator
                from .multitenant_collections import (
                    MultiTenantWorkspaceCollectionManager,
                )

                # Validate collection name
                validator = CollectionNamingValidator()
                validation_result = validator.validate_name(
                    collection_name,
                    existing_collections=self.list_collections()
                )

                if not validation_result.is_valid:
                    return {
                        "success": False,
                        "error": f"Invalid collection name: {validation_result.error_message}",
                        "suggestions": validation_result.suggested_names
                    }

                multitenant_available = True
            except ImportError:
                logger.warning("Multi-tenant components not available, using legacy collection creation")
                multitenant_available = False

            # Get or generate project context
            if not project_metadata:
                project_metadata = self.get_project_context(collection_type)

            if multitenant_available:
                # Create multi-tenant collection manager
                mt_manager = MultiTenantWorkspaceCollectionManager(
                    self.client, self.config
                )

                # Extract project information
                project_name = None
                if project_metadata:
                    project_name = project_metadata.get("project_name")

                if not project_name and self.project_info:
                    project_name = self.project_info.get("main_project")

                if not project_name:
                    return {
                        "success": False,
                        "error": "No project context available for multi-tenant collection creation"
                    }

                # Create the collection using multi-tenant manager
                result = await mt_manager.create_workspace_collection(
                    project_name=project_name,
                    collection_type=collection_type,
                    enable_metadata_indexing=True
                )

                # Enhance result with validation metadata
                if result.get("success") and 'validation_result' in locals() and hasattr(validation_result, 'proposed_metadata'):
                    if validation_result.proposed_metadata:
                        result["metadata_schema"] = validation_result.proposed_metadata.to_dict()

                logger.info(
                    "Multi-tenant collection creation completed",
                    collection_name=collection_name,
                    success=result.get("success"),
                    project_name=project_name
                )

                return result
            else:
                # Fallback to legacy collection creation
                from .collections import CollectionConfig

                # Create collection configuration
                collection_config = CollectionConfig(
                    name=collection_name,
                    description=f"{collection_type.title()} collection",
                    collection_type=collection_type,
                    project_name=project_metadata.get("project_name") if project_metadata else None,
                    vector_size=self.collection_manager._get_vector_size(),
                    enable_sparse_vectors=self.config.get("embedding.enable_sparse_vectors", True),
                )

                # Create the collection
                self.collection_manager._ensure_collection_exists(collection_config)

                logger.info("Legacy collection creation completed", collection_name=collection_name)

                return {
                    "success": True,
                    "collection_name": collection_name,
                    "collection_type": collection_type,
                    "method": "legacy"
                }

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return {
                "success": False,
                "error": f"Collection creation failed: {e}"
            }

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


def create_qdrant_client(config_data=None) -> QdrantWorkspaceClient:
    """Create a QdrantWorkspaceClient instance with lua-style configuration.

    This is a factory function that creates a workspace client using
    the lua-style configuration pattern. Configuration is accessed
    directly via get_config() functions.

    Args:
        config_data: Deprecated parameter for backward compatibility

    Returns:
        QdrantWorkspaceClient: Initialized client instance
    """
    # Create and return the client (no config parameter needed)
    client = QdrantWorkspaceClient()
    return client
