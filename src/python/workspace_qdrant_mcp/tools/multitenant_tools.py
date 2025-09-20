"""
Multi-tenant MCP tools for workspace collections.

This module provides MCP tool extensions for multi-tenant workspace collection
management and search. It integrates with the existing FastMCP server to provide
project-aware collection operations.

New MCP Tools:
    - create_workspace_collection: Create project-specific workspace collections
    - search_workspace_by_project: Search with automatic project context
    - list_workspace_collections_by_project: List collections for specific project
    - get_workspace_collection_metadata: Get collection metadata with project context
    - add_document_with_project_context: Add documents with project metadata
"""

from typing import Dict, List, Optional
from fastmcp import FastMCP
from loguru import logger

from common.core.client import QdrantWorkspaceClient
from common.core.multitenant_collections import (
    MultiTenantWorkspaceCollectionManager,
    WorkspaceCollectionRegistry
)
from .multitenant_search import (
    search_workspace_with_project_context,
    search_workspace_by_metadata_with_project_context
)
from ..validation import (
    require_project_access,
    require_collection_access,
    log_security_events,
    validate_workspace_client
)


def register_multitenant_tools(app: FastMCP, workspace_client: QdrantWorkspaceClient) -> None:
    """
    Register multi-tenant MCP tools with the FastMCP app.

    Args:
        app: FastMCP application instance
        workspace_client: Initialized workspace client
    """

    @app.tool()
    @require_project_access(project_param="project_name", operation="create_collection", allow_none_project=False)
    @log_security_events(event_type="create_workspace_collection", include_args=True)
    async def create_workspace_collection(
        project_name: str,
        collection_type: str,
        enable_metadata_indexing: bool = True
    ) -> Dict:
        """
        Create a workspace collection with multi-tenant support.

        Args:
            project_name: Project name for tenant isolation
            collection_type: Workspace collection type (notes, docs, scratchbook, knowledge, context, memory)
            enable_metadata_indexing: Whether to create metadata indexes for efficient filtering

        Returns:
            Dict: Creation result with success status and collection details

        Example:
            ```python
            # Create a notes collection for a project
            result = await create_workspace_collection(
                project_name="my-project",
                collection_type="notes"
            )

            # Create a shared scratchbook collection
            result = await create_workspace_collection(
                project_name="team-workspace",
                collection_type="scratchbook"
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Validate collection type
            registry = WorkspaceCollectionRegistry()
            if not registry.is_multi_tenant_type(collection_type):
                valid_types = list(registry.get_workspace_types())
                return {
                    "error": f"Invalid collection type '{collection_type}'. "
                             f"Valid types: {valid_types}"
                }

            # Use the new client API for collection creation
            collection_name = f"{project_name}-{collection_type}"
            project_metadata = {"project_name": project_name}

            # Create the workspace collection using enhanced client API
            result = await workspace_client.create_collection(
                collection_name=collection_name,
                collection_type=collection_type,
                project_metadata=project_metadata
            )

            # If the new API is not available, fallback to multi-tenant manager
            if result.get("error") and "not available" in str(result.get("error", "")):
                # Fallback to direct multi-tenant manager
                mt_manager = MultiTenantWorkspaceCollectionManager(
                    workspace_client.client, workspace_client.config
                )

                result = await mt_manager.create_workspace_collection(
                    project_name=project_name,
                    collection_type=collection_type,
                    enable_metadata_indexing=enable_metadata_indexing
                )

            logger.info(
                f"Workspace collection creation {'succeeded' if result['success'] else 'failed'}",
                project_name=project_name,
                collection_type=collection_type,
                collection_name=result.get('collection_name')
            )

            return result

        except Exception as e:
            logger.error(f"Failed to create workspace collection: {e}")
            return {"error": f"Collection creation failed: {e}"}

    @app.tool()
    @require_project_access(project_param="project_name", operation="list_collections", allow_none_project=True)
    @log_security_events(event_type="list_workspace_collections_by_project", include_args=True)
    async def list_workspace_collections_by_project(
        project_name: Optional[str] = None,
        include_shared: bool = True,
        include_metadata: bool = False
    ) -> Dict:
        """
        List workspace collections for a specific project with enhanced filtering.

        Args:
            project_name: Project name to filter collections for (auto-detected if None)
            include_shared: Whether to include shared/global collections
            include_metadata: Whether to include collection metadata information

        Returns:
            Dict: List of collections with optional metadata

        Example:
            ```python
            # List collections for current project
            collections = await list_workspace_collections_by_project()

            # List collections for specific project
            collections = await list_workspace_collections_by_project(
                project_name="backend-service",
                include_metadata=True
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project name if not provided
            if not project_name:
                project_info = workspace_client.get_project_info()
                if project_info:
                    project_name = project_info.get("main_project")

            # Use the enhanced client API for project-aware collection listing
            if project_name and hasattr(workspace_client.collection_manager, 'list_collections_for_project'):
                collections = workspace_client.collection_manager.list_collections_for_project(
                    project_name
                )
            else:
                # Fallback to regular collection listing
                collections = workspace_client.list_collections()

            result = {
                "success": True,
                "project_name": project_name,
                "collections": collections,
                "collection_count": len(collections),
                "include_shared": include_shared
            }

            # Add metadata information if requested
            if include_metadata and collections:
                metadata_info = {}
                for collection_name in collections:
                    try:
                        if hasattr(workspace_client.collection_manager, 'get_collection_metadata_info'):
                            metadata = workspace_client.collection_manager.get_collection_metadata_info(collection_name)
                            metadata_info[collection_name] = metadata
                        else:
                            metadata_info[collection_name] = {"metadata_available": False}
                    except Exception as e:
                        metadata_info[collection_name] = {"error": str(e)}

                result["metadata"] = metadata_info

            logger.info(
                "Project collections listed",
                project_name=project_name,
                collection_count=len(collections),
                include_metadata=include_metadata
            )

            return result

        except Exception as e:
            logger.error(f"Failed to list project collections: {e}")
            return {"error": f"Failed to list collections: {e}"}

    @app.tool()
    @require_project_access(project_param="project_name", operation="search", allow_none_project=True)
    @log_security_events(event_type="search_workspace_by_project", include_args=True)
    async def search_workspace_by_project(
        query: str,
        project_name: Optional[str] = None,
        workspace_types: Optional[List[str]] = None,
        mode: str = "hybrid",
        limit: int = 10,
        score_threshold: float = 0.7,
        include_shared: bool = True
    ) -> Dict:
        """
        Search workspace collections with automatic project context filtering.

        Args:
            query: Natural language search query
            project_name: Project context for filtering (auto-detected if None)
            workspace_types: Specific workspace types to search (notes, docs, etc.)
            mode: Search strategy - 'hybrid' (best), 'dense' (semantic), 'sparse' (keyword)
            limit: Maximum number of results to return
            score_threshold: Minimum relevance score (0.0-1.0)
            include_shared: Include shared workspace collections

        Returns:
            Dict: Enhanced search results with project context

        Example:
            ```python
            # Search notes in current project
            results = await search_workspace_by_project(
                query="authentication implementation",
                workspace_types=["notes", "docs"]
            )

            # Search specific project
            results = await search_workspace_by_project(
                query="API endpoints",
                project_name="backend-service",
                mode="hybrid",
                limit=20
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Validate workspace types if provided
            if workspace_types:
                registry = WorkspaceCollectionRegistry()
                valid_types = registry.get_workspace_types()
                invalid_types = [t for t in workspace_types if t not in valid_types]

                if invalid_types:
                    return {
                        "error": f"Invalid workspace types: {invalid_types}. "
                                f"Valid types: {list(valid_types)}"
                    }

            # Perform project-aware search
            result = await search_workspace_with_project_context(
                client=workspace_client,
                query=query,
                project_name=project_name,
                workspace_types=workspace_types,
                mode=mode,
                limit=limit,
                score_threshold=score_threshold,
                include_shared=include_shared
            )

            logger.info(
                "Project-aware search completed",
                query_length=len(query),
                project_name=result.get('project_name'),
                workspace_types=workspace_types,
                results_count=result.get('total_results', 0)
            )

            return result

        except Exception as e:
            logger.error(f"Project-aware search failed: {e}")
            return {"error": f"Search failed: {e}"}

    @app.tool()
    async def search_workspace_metadata_by_project(
        metadata_filter: Dict,
        project_name: Optional[str] = None,
        workspace_types: Optional[List[str]] = None,
        limit: int = 10,
        include_shared: bool = True
    ) -> Dict:
        """
        Search workspace collections by metadata with project filtering.

        Args:
            metadata_filter: Metadata filter conditions (e.g., {"category": "documentation", "priority": 4})
            project_name: Project context for filtering (auto-detected if None)
            workspace_types: Specific workspace types to search
            limit: Maximum number of results to return
            include_shared: Include shared workspace collections

        Returns:
            Dict: Search results with project metadata filtering

        Example:
            ```python
            # Find high-priority documentation
            results = await search_workspace_metadata_by_project(
                metadata_filter={"category": "documentation", "priority": 4},
                workspace_types=["docs"]
            )

            # Find notes by a specific author
            results = await search_workspace_metadata_by_project(
                metadata_filter={"created_by": "developer", "tags": ["urgent"]},
                project_name="my-project"
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        if not metadata_filter:
            return {"error": "Metadata filter cannot be empty"}

        try:
            # Perform metadata search with project context
            result = await search_workspace_by_metadata_with_project_context(
                client=workspace_client,
                metadata_filter=metadata_filter,
                project_name=project_name,
                workspace_types=workspace_types,
                limit=limit,
                include_shared=include_shared
            )

            logger.info(
                "Project metadata search completed",
                filter_keys=list(metadata_filter.keys()),
                project_name=result.get('project_name'),
                results_count=result.get('total_results', 0)
            )

            return result

        except Exception as e:
            logger.error(f"Project metadata search failed: {e}")
            return {"error": f"Metadata search failed: {e}"}

    @app.tool()
    async def list_workspace_collections_by_project(
        project_name: Optional[str] = None,
        workspace_types: Optional[List[str]] = None,
        include_shared: bool = True
    ) -> Dict:
        """
        List workspace collections for a specific project context.

        Args:
            project_name: Project name (auto-detected if None)
            workspace_types: Filter by specific workspace types
            include_shared: Include shared workspace collections

        Returns:
            Dict: List of collections with project context information

        Example:
            ```python
            # List all workspace collections for current project
            collections = await list_workspace_collections_by_project()

            # List notes and docs for specific project
            collections = await list_workspace_collections_by_project(
                project_name="backend-api",
                workspace_types=["notes", "docs"]
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            if not project_name:
                return {"error": "No project context available"}

            # Get all collections
            all_collections = workspace_client.list_collections()
            project_collections = []

            # Registry for workspace types
            registry = WorkspaceCollectionRegistry()
            valid_workspace_types = registry.get_workspace_types()

            # Filter workspace types if specified
            target_types = workspace_types or list(valid_workspace_types)

            for collection_name in all_collections:
                # Check if it's a workspace collection for this project
                for workspace_type in target_types:
                    expected_name = f"{project_name}-{workspace_type}"

                    if collection_name == expected_name:
                        project_collections.append({
                            "collection_name": collection_name,
                            "project_name": project_name,
                            "workspace_type": workspace_type,
                            "scope": "project"
                        })
                        break

                    # Check for shared collections
                    if include_shared and workspace_type == "scratchbook" and collection_name == "scratchbook":
                        project_collections.append({
                            "collection_name": collection_name,
                            "project_name": "shared",
                            "workspace_type": workspace_type,
                            "scope": "shared"
                        })

            return {
                "project_name": project_name,
                "workspace_types": target_types,
                "include_shared": include_shared,
                "total_collections": len(project_collections),
                "collections": project_collections
            }

        except Exception as e:
            logger.error(f"Failed to list workspace collections: {e}")
            return {"error": f"Collection listing failed: {e}"}

    @app.tool()
    @require_collection_access(collection_param="collection", project_param="project_name", operation="write", allow_shared=True)
    @log_security_events(event_type="add_document_with_project_context", include_args=False)
    async def add_document_with_project_context(
        content: str,
        collection: str,
        project_name: Optional[str] = None,
        workspace_type: Optional[str] = None,
        metadata: Optional[Dict] = None,
        document_id: Optional[str] = None,
        chunk_text: bool = True,
        creator: str = "system"
    ) -> Dict:
        """
        Add a document to workspace collection with automatic project metadata injection.

        Args:
            content: Document text content to be indexed
            collection: Target collection name
            project_name: Project context (auto-detected if None)
            workspace_type: Workspace collection type for metadata
            metadata: Additional metadata (will be enriched with project context)
            document_id: Custom document identifier
            chunk_text: Whether to split large documents into chunks
            creator: Creator identifier for access tracking

        Returns:
            Dict: Addition result with project context metadata

        Example:
            ```python
            # Add a note with project context
            result = await add_document_with_project_context(
                content="Implementation notes for OAuth flow",
                collection="my-project-notes",
                workspace_type="notes",
                metadata={"category": "security", "priority": 4}
            )

            # Add documentation with explicit project
            result = await add_document_with_project_context(
                content=api_documentation,
                collection="backend-api-docs",
                project_name="backend-api",
                workspace_type="docs",
                creator="api-team"
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            # Auto-detect workspace type from collection name
            if not workspace_type and project_name:
                registry = WorkspaceCollectionRegistry()
                for wtype in registry.get_workspace_types():
                    if collection == f"{project_name}-{wtype}":
                        workspace_type = wtype
                        break

            # Prepare base metadata
            base_metadata = metadata or {}

            # Enrich with project context if available
            if project_name and workspace_type:
                mt_manager = MultiTenantWorkspaceCollectionManager(
                    workspace_client.client, workspace_client.config
                )

                enhanced_metadata = mt_manager.enrich_document_metadata(
                    base_metadata=base_metadata,
                    project_name=project_name,
                    collection_type=workspace_type
                )

                # Override creator if provided
                if creator != "system":
                    enhanced_metadata["created_by"] = creator

                logger.debug(
                    "Document metadata enriched with project context",
                    project_name=project_name,
                    workspace_type=workspace_type,
                    metadata_fields=list(enhanced_metadata.keys())
                )

                final_metadata = enhanced_metadata
            else:
                final_metadata = base_metadata
                logger.warning("Document added without project context enrichment")

            # Add document using existing functionality
            from .documents import add_document

            result = await add_document(
                workspace_client=workspace_client,
                content=content,
                collection=collection,
                metadata=final_metadata,
                document_id=document_id,
                chunk_text=chunk_text
            )

            # Enhance result with project context
            if result.get("success"):
                result.update({
                    "project_name": project_name,
                    "workspace_type": workspace_type,
                    "metadata_enriched": bool(project_name and workspace_type)
                })

            logger.info(
                f"Document added with project context",
                success=result.get("success"),
                collection=collection,
                project_name=project_name,
                workspace_type=workspace_type,
                document_id=result.get("document_id")
            )

            return result

        except Exception as e:
            logger.error(f"Failed to add document with project context: {e}")
            return {"error": f"Document addition failed: {e}"}

    @app.tool()
    async def initialize_project_workspace_collections(
        project_name: str,
        workspace_types: Optional[List[str]] = None,
        subprojects: Optional[List[str]] = None,
        enable_metadata_indexing: bool = True
    ) -> Dict:
        """
        Initialize workspace collections for a project with multi-tenant support.

        Args:
            project_name: Main project name
            workspace_types: Workspace types to create (default: notes, docs, scratchbook)
            subprojects: Optional list of subproject names
            enable_metadata_indexing: Whether to create metadata indexes

        Returns:
            Dict: Initialization results with creation summary

        Example:
            ```python
            # Initialize default workspace for a project
            result = await initialize_project_workspace_collections(
                project_name="my-new-project"
            )

            # Initialize with specific types and subprojects
            result = await initialize_project_workspace_collections(
                project_name="enterprise-app",
                workspace_types=["notes", "docs", "knowledge"],
                subprojects=["frontend", "backend", "mobile"]
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Validate workspace types if provided
            if workspace_types:
                registry = WorkspaceCollectionRegistry()
                valid_types = registry.get_workspace_types()
                invalid_types = [t for t in workspace_types if t not in valid_types]

                if invalid_types:
                    return {
                        "error": f"Invalid workspace types: {invalid_types}. "
                                f"Valid types: {list(valid_types)}"
                    }

            # Create multi-tenant collection manager
            mt_manager = MultiTenantWorkspaceCollectionManager(
                workspace_client.client, workspace_client.config
            )

            # Initialize workspace collections
            result = await mt_manager.initialize_workspace_collections(
                project_name=project_name,
                subprojects=subprojects,
                workspace_types=workspace_types
            )

            logger.info(
                f"Project workspace initialization {'completed' if result['success'] else 'failed'}",
                project_name=project_name,
                workspace_types=workspace_types,
                subprojects=subprojects,
                collections_created=len(result.get('collections_created', [])),
                collections_existing=len(result.get('collections_existing', [])),
                errors=len(result.get('errors', []))
            )

            return result

        except Exception as e:
            logger.error(f"Failed to initialize project workspace: {e}")
            return {"error": f"Workspace initialization failed: {e}"}

    @app.tool()
    async def get_workspace_collection_info(
        project_name: Optional[str] = None,
        include_metadata_stats: bool = True
    ) -> Dict:
        """
        Get comprehensive information about workspace collections for a project.

        Args:
            project_name: Project name (auto-detected if None)
            include_metadata_stats: Include metadata statistics and schema info

        Returns:
            Dict: Comprehensive workspace collection information

        Example:
            ```python
            # Get workspace info for current project
            info = await get_workspace_collection_info()

            # Get detailed info for specific project
            info = await get_workspace_collection_info(
                project_name="my-project",
                include_metadata_stats=True
            )
            ```
        """
        if not workspace_client or not workspace_client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Auto-detect project if not provided
            if not project_name:
                project_info = getattr(workspace_client, 'project_info', None)
                if project_info:
                    project_name = project_info.get("main_project")

            if not project_name:
                return {"error": "No project context available"}

            # Get workspace collections
            collections_result = await list_workspace_collections_by_project(
                project_name=project_name
            )

            if "error" in collections_result:
                return collections_result

            # Get collection information
            collection_details = []
            registry = WorkspaceCollectionRegistry()

            for collection_info in collections_result.get("collections", []):
                collection_name = collection_info["collection_name"]
                workspace_type = collection_info["workspace_type"]

                details = {
                    "collection_name": collection_name,
                    "workspace_type": workspace_type,
                    "project_name": collection_info["project_name"],
                    "scope": collection_info["scope"],
                    "description": registry.collection_schemas.get(workspace_type, {}).get("description", ""),
                    "searchable": registry.is_searchable(workspace_type)
                }

                # Add metadata statistics if requested
                if include_metadata_stats:
                    try:
                        collection_stats = workspace_client.client.get_collection(collection_name)
                        details["stats"] = {
                            "points_count": collection_stats.points_count,
                            "vectors_count": collection_stats.vectors_count,
                            "indexed_vectors_count": collection_stats.indexed_vectors_count
                        }
                    except Exception as e:
                        details["stats"] = {"error": f"Could not retrieve stats: {e}"}

                collection_details.append(details)

            return {
                "project_name": project_name,
                "total_workspace_collections": len(collection_details),
                "workspace_types_available": list(registry.get_workspace_types()),
                "collections": collection_details,
                "include_metadata_stats": include_metadata_stats
            }

        except Exception as e:
            logger.error(f"Failed to get workspace collection info: {e}")
            return {"error": f"Collection info retrieval failed: {e}"}