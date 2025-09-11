"""
Simplified MCP Tools Interface for workspace-qdrant-mcp.

This module implements the simplified tool set (2-5 tools) that consolidates
the functionality of 30+ existing tools into a streamlined interface compatible
with reference implementations while maintaining full backward compatibility.

The simplified interface provides:
- qdrant_store: Universal document storage (compatible with reference)
- qdrant_find: Universal search and retrieval (compatible with reference)  
- qdrant_manage: Workspace and collection management
- qdrant_watch: File monitoring and auto-ingestion (optional)

This module serves as the compatibility layer between the simplified interface
and the existing comprehensive tool implementations.
"""

import os
from typing import Dict, List, Optional, Any, Union
import logging

from ..core.error_handling import (
    ErrorRecoveryStrategy,
    with_error_handling,
    error_context,
)
from ..observability import (
    monitor_async,
    get_logger,
    record_operation,
)

# Import existing tool functions to route through simplified interface
from . import (
    documents,
    search,
    scratchbook,
    watch_management,
    grpc_tools,
)

logger = get_logger(__name__)


class SimplifiedToolsMode:
    """Configuration for simplified tools mode selection."""
    
    BASIC = "basic"          # 2 core tools (store, find) - reference compatible
    STANDARD = "standard"    # 4 tools (store, find, manage, watch) - recommended  
    FULL = "full"           # All 30+ tools - existing behavior
    COMPATIBLE = "compatible" # Reference implementation compatible mode
    
    @classmethod
    def get_mode(cls) -> str:
        """Get current mode from environment variable."""
        return os.getenv("QDRANT_MCP_MODE", cls.STANDARD).lower()
    
    @classmethod
    def is_simplified_mode(cls) -> bool:
        """Check if running in any simplified mode."""
        return cls.get_mode() in [cls.BASIC, cls.STANDARD, cls.COMPATIBLE]
    
    @classmethod
    def get_enabled_tools(cls) -> List[str]:
        """Get list of enabled tools based on current mode."""
        mode = cls.get_mode()
        
        if mode == cls.BASIC or mode == cls.COMPATIBLE:
            return ["qdrant_store", "qdrant_find"]
        elif mode == cls.STANDARD:
            return ["qdrant_store", "qdrant_find", "qdrant_manage", "qdrant_watch"]
        else:  # FULL mode
            return []  # Return empty to indicate all tools enabled


class SimplifiedToolsRouter:
    """Routes simplified tool calls to existing comprehensive implementations."""
    
    def __init__(self, workspace_client, watch_tools_manager=None):
        self.workspace_client = workspace_client
        self.watch_tools_manager = watch_tools_manager
        self.mode = SimplifiedToolsMode.get_mode()
        
        logger.info(
            "Simplified tools router initialized",
            mode=self.mode,
            enabled_tools=SimplifiedToolsMode.get_enabled_tools()
        )

    @monitor_async("qdrant_store", critical=True, timeout_warning=10.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_store")
    async def qdrant_store(
        self,
        information: str,
        collection: str = None,
        metadata: dict = None,
        document_id: str = None,
        note_type: str = "document",
        tags: List[str] = None,
        chunk_text: bool = None,
        title: str = None,
    ) -> Dict[str, Any]:
        """
        Store information in Qdrant database (Reference implementation compatible).
        
        Universal document storage that consolidates functionality from:
        - add_document_tool: Standard document ingestion
        - update_scratchbook_tool: Note and scratchbook management
        - process_document_via_grpc_tool: High-performance gRPC processing
        
        Args:
            information: Document text content or note content to store
            collection: Target collection name (auto-detected if None)
            metadata: Optional metadata dictionary for document organization
            document_id: Custom document identifier (UUID generated if None)
            note_type: Type of content - "document", "note", "scratchbook", "code"
            tags: List of tags for organization and filtering
            chunk_text: Whether to split large documents (auto-detect if None)
            title: Title for scratchbook notes (auto-generated if None)
            
        Returns:
            dict: Storage result with success status, document_id, and metadata
            
        Example:
            ```python
            # Reference implementation compatible usage
            result = await qdrant_store(
                information="This is important information to remember",
                collection="my-project"
            )
            
            # Advanced usage with full metadata
            result = await qdrant_store(
                information=file_content,
                collection="documentation",
                metadata={"file_path": "/docs/api.md", "file_type": "markdown"},
                note_type="document",
                tags=["api", "documentation"],
                chunk_text=True
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        logger.debug(
            "Simplified store request",
            content_length=len(information),
            collection=collection,
            note_type=note_type,
            tags=tags
        )
        
        try:
            # Auto-detect collection if not provided
            if not collection:
                if note_type == "scratchbook":
                    collection = "scratchbook"
                else:
                    # Use default project collection
                    status = await self.workspace_client.get_status()
                    collection = status.get("current_project", "default")
            
            # Handle different content types through appropriate tools
            if note_type in ["note", "scratchbook"]:
                # Route to scratchbook functionality
                from .scratchbook import update_scratchbook
                result = await update_scratchbook(
                    self.workspace_client,
                    content=information,
                    note_id=document_id,
                    title=title,
                    tags=tags,
                    note_type=note_type
                )
            else:
                # Route to standard document ingestion
                from .documents import add_document
                
                # Auto-detect chunking based on content size if not specified
                if chunk_text is None:
                    chunk_text = len(information) > 4000  # Chunk if > 4KB
                
                # Enhance metadata with tags if provided
                if metadata is None:
                    metadata = {}
                if tags:
                    metadata["tags"] = tags
                if note_type:
                    metadata["note_type"] = note_type
                    
                result = await add_document(
                    self.workspace_client,
                    content=information,
                    collection=collection,
                    metadata=metadata,
                    document_id=document_id,
                    chunk_text=chunk_text
                )
            
            logger.info(
                "Document stored successfully",
                collection=collection,
                document_id=result.get("document_id"),
                note_type=note_type
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to store information", error=str(e), exc_info=True)
            return {"error": f"Storage failed: {str(e)}", "success": False}

    @monitor_async("qdrant_find", timeout_warning=2.0, slow_threshold=1.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_find")
    async def qdrant_find(
        self,
        query: str,
        collection: str = None,
        limit: int = 10,
        score_threshold: float = 0.7,
        search_mode: str = "hybrid",
        filters: Dict[str, Any] = None,
        note_types: List[str] = None,
        tags: List[str] = None,
        include_relationships: bool = False,
    ) -> Dict[str, Any]:
        """
        Find relevant information from database (Reference implementation compatible).
        
        Universal search and retrieval that consolidates functionality from:
        - search_workspace_tool: Multi-collection semantic search
        - search_scratchbook_tool: Specialized note search with filtering
        - research_workspace: Advanced semantic research with context control
        - hybrid_search_advanced_tool: Configurable fusion methods
        - search_by_metadata_tool: Metadata-based filtering
        - search_via_grpc_tool: High-performance gRPC search
        
        Args:
            query: Natural language search query or exact text to find
            collection: Specific collection to search (searches all if None)
            limit: Maximum number of results to return (1-100)
            score_threshold: Minimum relevance score (0.0-1.0)
            search_mode: Search strategy - "hybrid", "semantic", "keyword", "exact"
            filters: Metadata filters for result refinement
            note_types: Filter by content types (["note", "document", "scratchbook"])
            tags: Filter by specific tags
            include_relationships: Include related documents and version chains
            
        Returns:
            dict: Search results with ranked matches and metadata
            
        Example:
            ```python
            # Reference implementation compatible usage
            results = await qdrant_find(
                query="authentication implementation"
            )
            
            # Advanced search with filtering
            results = await qdrant_find(
                query="API documentation",
                collection="my-project",
                search_mode="hybrid",
                tags=["api", "docs"],
                note_types=["document"],
                limit=5
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        # Validate parameters
        try:
            limit = int(limit) if isinstance(limit, str) else limit
            score_threshold = float(score_threshold) if isinstance(score_threshold, str) else score_threshold
            
            if limit <= 0:
                return {"error": "limit must be greater than 0"}
            if not (0.0 <= score_threshold <= 1.0):
                return {"error": "score_threshold must be between 0.0 and 1.0"}
        except (ValueError, TypeError) as e:
            return {"error": f"Invalid parameter types: {e}"}
        
        logger.debug(
            "Simplified find request",
            query_length=len(query),
            collection=collection,
            search_mode=search_mode,
            filters=filters,
            note_types=note_types,
            tags=tags
        )
        
        try:
            # Determine search context and route appropriately
            if note_types and any(nt in ["note", "scratchbook"] for nt in note_types):
                # Route to scratchbook search for notes
                from .scratchbook import ScratchbookManager
                manager = ScratchbookManager(self.workspace_client)
                
                result = await manager.search_notes(
                    query=query,
                    note_types=note_types,
                    tags=tags,
                    project_name=collection,  # Use collection as project filter
                    limit=limit,
                    mode=search_mode
                )
            elif collection and search_mode == "exact":
                # Route to metadata search for exact matches
                from .search import search_collection_by_metadata
                
                # Create metadata filter for exact text search
                metadata_filter = {"content": query}
                if filters:
                    metadata_filter.update(filters)
                
                result = await search_collection_by_metadata(
                    self.workspace_client,
                    collection=collection,
                    metadata_filter=metadata_filter,
                    limit=limit
                )
            elif include_relationships:
                # Route to advanced research functionality
                from .research import research_workspace as research_workspace_impl
                
                # Determine research mode
                if collection:
                    mode = "collection"
                    target_collection = collection
                else:
                    mode = "project"
                    target_collection = None
                
                result = await research_workspace_impl(
                    client=self.workspace_client,
                    query=query,
                    mode=mode,
                    target_collection=target_collection,
                    include_relationships=include_relationships,
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                # Route to standard workspace search
                from .search import search_workspace
                
                # Convert search_mode to internal mode parameter
                mode_mapping = {
                    "hybrid": "hybrid",
                    "semantic": "dense",
                    "keyword": "sparse",
                    "exact": "sparse"
                }
                internal_mode = mode_mapping.get(search_mode, "hybrid")
                
                # Determine collections to search
                if collection:
                    collections = [collection]
                else:
                    collections = None  # Search all accessible collections
                
                result = await search_workspace(
                    self.workspace_client,
                    query=query,
                    collections=collections,
                    mode=internal_mode,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                # Apply additional filtering if specified
                if filters or tags or note_types:
                    result = self._apply_additional_filters(
                        result, filters, tags, note_types
                    )
            
            logger.info(
                "Search completed successfully",
                results_count=result.get("total_results", 0),
                search_mode=search_mode,
                collection=collection
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to find information", error=str(e), exc_info=True)
            return {"error": f"Search failed: {str(e)}", "success": False}

    def _apply_additional_filters(
        self,
        result: Dict[str, Any],
        filters: Dict[str, Any] = None,
        tags: List[str] = None,
        note_types: List[str] = None
    ) -> Dict[str, Any]:
        """Apply additional client-side filtering to search results."""
        if not result.get("results"):
            return result
        
        filtered_results = []
        
        for item in result["results"]:
            payload = item.get("payload", {})
            
            # Apply metadata filters
            if filters:
                matches_filter = True
                for key, value in filters.items():
                    if key not in payload or payload[key] != value:
                        matches_filter = False
                        break
                if not matches_filter:
                    continue
            
            # Apply tag filters
            if tags:
                item_tags = payload.get("tags", [])
                if not any(tag in item_tags for tag in tags):
                    continue
            
            # Apply note type filters
            if note_types:
                item_note_type = payload.get("note_type", "document")
                if item_note_type not in note_types:
                    continue
            
            filtered_results.append(item)
        
        result["results"] = filtered_results
        result["total_results"] = len(filtered_results)
        return result

    @monitor_async("qdrant_manage", timeout_warning=5.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_manage")
    async def qdrant_manage(
        self,
        action: str,
        collection: str = None,
        document_id: str = None,
        note_id: str = None,
        include_vectors: bool = False,
        note_type: str = None,
        tags: List[str] = None,
        limit: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage workspace collections and documents.
        
        Consolidates functionality from:
        - workspace_status: Get workspace and collection status
        - list_workspace_collections: List available collections
        - get_document_tool: Retrieve specific documents
        - list_scratchbook_notes_tool: List notes with filtering
        - delete_scratchbook_note_tool: Delete notes and documents
        
        Args:
            action: Management action - "status", "collections", "get", "list_notes", "delete"
            collection: Target collection name (required for some actions)
            document_id: Document identifier (required for "get", "delete")
            note_id: Note identifier (required for note operations)
            include_vectors: Include embedding vectors in response
            note_type: Filter by note type for listing operations
            tags: Filter by tags for listing operations
            limit: Maximum results for listing operations
            **kwargs: Additional parameters for specific actions
            
        Returns:
            dict: Management operation results
            
        Example:
            ```python
            # Get workspace status
            status = await qdrant_manage(action="status")
            
            # List available collections
            collections = await qdrant_manage(action="collections")
            
            # Get specific document
            doc = await qdrant_manage(
                action="get",
                collection="my-project",
                document_id="doc-123"
            )
            
            # List notes with filtering
            notes = await qdrant_manage(
                action="list_notes",
                note_type="scratchbook",
                tags=["important"]
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        logger.debug("Management action requested", action=action, collection=collection)
        
        try:
            if action == "status":
                # Route to workspace status
                status = await self.workspace_client.get_status()
                logger.info("Workspace status retrieved", connected=status.get("connected"))
                return status
                
            elif action == "collections":
                # Route to list collections
                collections = self.workspace_client.list_collections()
                logger.info("Collections listed", count=len(collections))
                return {"collections": collections, "count": len(collections)}
                
            elif action == "get":
                if not document_id or not collection:
                    return {"error": "document_id and collection required for get action"}
                
                # Route to get document
                from .documents import get_document
                result = await get_document(
                    self.workspace_client,
                    document_id=document_id,
                    collection=collection,
                    include_vectors=include_vectors
                )
                logger.info("Document retrieved", document_id=document_id, collection=collection)
                return result
                
            elif action == "list_notes":
                # Route to list scratchbook notes
                from .scratchbook import ScratchbookManager
                manager = ScratchbookManager(self.workspace_client)
                
                result = await manager.list_notes(
                    project_name=collection,
                    note_type=note_type,
                    tags=tags,
                    limit=limit
                )
                logger.info("Notes listed", count=result.get("total_notes", 0))
                return result
                
            elif action == "delete":
                if note_id:
                    # Route to delete scratchbook note
                    from .scratchbook import ScratchbookManager
                    manager = ScratchbookManager(self.workspace_client)
                    
                    result = await manager.delete_note(note_id, collection)
                    logger.info("Note deleted", note_id=note_id)
                    return result
                else:
                    return {"error": "note_id required for delete action"}
                
            else:
                return {"error": f"Unknown management action: {action}"}
                
        except Exception as e:
            logger.error("Management action failed", action=action, error=str(e), exc_info=True)
            return {"error": f"Management action failed: {str(e)}", "success": False}

    @monitor_async("qdrant_watch", timeout_warning=10.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_watch")
    async def qdrant_watch(
        self,
        action: str,
        path: str = None,
        collection: str = None,
        watch_id: str = None,
        patterns: List[str] = None,
        auto_ingest: bool = True,
        recursive: bool = True,
        debounce_seconds: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manage folder watching for automatic document ingestion.
        
        Simplified interface for all 13 watch-related tools:
        - add_watch_folder, remove_watch_folder, list_watched_folders
        - configure_watch_settings, get_watch_status, validate_watch_path
        - And 7 additional advanced watch management tools
        
        Args:
            action: Watch action - "add", "remove", "list", "status", "configure", "validate"
            path: Directory path to watch (required for "add", "validate")
            collection: Target collection for ingested files
            watch_id: Watch identifier (required for "remove", "configure", "status")
            patterns: File patterns to include (e.g., ["*.pdf", "*.txt"])
            auto_ingest: Enable automatic file ingestion
            recursive: Watch subdirectories recursively
            debounce_seconds: Delay before processing file changes
            **kwargs: Additional parameters for specific actions
            
        Returns:
            dict: Watch management operation results
            
        Example:
            ```python
            # Add new folder watch
            result = await qdrant_watch(
                action="add",
                path="/home/user/documents",
                collection="my-project",
                patterns=["*.pdf", "*.docx"]
            )
            
            # List all watches
            watches = await qdrant_watch(action="list")
            
            # Get watch status
            status = await qdrant_watch(
                action="status",
                watch_id="my-watch"
            )
            ```
        """
        if not self.workspace_client or not self.watch_tools_manager:
            return {"error": "Watch management not initialized"}
        
        logger.debug("Watch action requested", action=action, path=path, watch_id=watch_id)
        
        try:
            if action == "add":
                if not path or not collection:
                    return {"error": "path and collection required for add action"}
                
                result = await self.watch_tools_manager.add_watch_folder(
                    path=path,
                    collection=collection,
                    patterns=patterns,
                    auto_ingest=auto_ingest,
                    recursive=recursive,
                    debounce_seconds=debounce_seconds,
                    **kwargs
                )
                logger.info("Watch added", path=path, collection=collection)
                return result
                
            elif action == "remove":
                if not watch_id:
                    return {"error": "watch_id required for remove action"}
                
                result = await self.watch_tools_manager.remove_watch_folder(watch_id)
                logger.info("Watch removed", watch_id=watch_id)
                return result
                
            elif action == "list":
                result = await self.watch_tools_manager.list_watched_folders(
                    **kwargs
                )
                logger.info("Watches listed", count=result.get("summary", {}).get("total_watches", 0))
                return result
                
            elif action == "status":
                result = await self.watch_tools_manager.get_watch_status(watch_id)
                logger.info("Watch status retrieved", watch_id=watch_id)
                return result
                
            elif action == "configure":
                if not watch_id:
                    return {"error": "watch_id required for configure action"}
                
                result = await self.watch_tools_manager.configure_watch_settings(
                    watch_id=watch_id,
                    patterns=patterns,
                    auto_ingest=auto_ingest,
                    recursive=recursive,
                    debounce_seconds=debounce_seconds,
                    **kwargs
                )
                logger.info("Watch configured", watch_id=watch_id)
                return result
                
            elif action == "validate":
                if not path:
                    return {"error": "path required for validate action"}
                
                from ..core.watch_validation import WatchPathValidator
                from pathlib import Path
                
                validation_result = WatchPathValidator.validate_watch_path(Path(path))
                logger.info("Path validated", path=path, valid=validation_result.valid)
                
                return {
                    "valid": validation_result.valid,
                    "path": path,
                    "error_code": validation_result.error_code,
                    "error_message": validation_result.error_message,
                    "warnings": validation_result.warnings,
                    "metadata": validation_result.metadata,
                }
                
            else:
                return {"error": f"Unknown watch action: {action}"}
                
        except Exception as e:
            logger.error("Watch action failed", action=action, error=str(e), exc_info=True)
            return {"error": f"Watch action failed: {str(e)}", "success": False}


# Global router instance (will be initialized by server)
_simplified_router: Optional[SimplifiedToolsRouter] = None

def initialize_simplified_router(workspace_client, watch_tools_manager=None):
    """Initialize the global simplified tools router."""
    global _simplified_router
    _simplified_router = SimplifiedToolsRouter(workspace_client, watch_tools_manager)
    logger.info("Simplified tools router initialized globally")

def get_simplified_router() -> Optional[SimplifiedToolsRouter]:
    """Get the global simplified tools router instance."""
    return _simplified_router


# Tool registration functions for FastMCP
async def register_simplified_tools(app, workspace_client, watch_tools_manager=None):
    """Register simplified tools with the FastMCP app if running in simplified mode."""
    mode = SimplifiedToolsMode.get_mode()
    
    if not SimplifiedToolsMode.is_simplified_mode():
        logger.info("Running in full mode - simplified tools not registered", mode=mode)
        return
    
    # Initialize router
    initialize_simplified_router(workspace_client, watch_tools_manager)
    router = get_simplified_router()
    
    enabled_tools = SimplifiedToolsMode.get_enabled_tools()
    logger.info("Registering simplified tools", mode=mode, tools=enabled_tools)
    
    # Register qdrant_store (always enabled in simplified modes)
    if "qdrant_store" in enabled_tools:
        @app.tool()
        async def qdrant_store(
            information: str,
            collection: str = None,
            metadata: dict = None,
            document_id: str = None,
            note_type: str = "document",
            tags: List[str] = None,
            chunk_text: bool = None,
            title: str = None,
        ) -> dict:
            """Store information in Qdrant database (Reference implementation compatible)."""
            return await router.qdrant_store(
                information=information,
                collection=collection,
                metadata=metadata,
                document_id=document_id,
                note_type=note_type,
                tags=tags,
                chunk_text=chunk_text,
                title=title,
            )
    
    # Register qdrant_find (always enabled in simplified modes)
    if "qdrant_find" in enabled_tools:
        @app.tool()
        async def qdrant_find(
            query: str,
            collection: str = None,
            limit: int = 10,
            score_threshold: float = 0.7,
            search_mode: str = "hybrid",
            filters: dict = None,
            note_types: List[str] = None,
            tags: List[str] = None,
            include_relationships: bool = False,
        ) -> dict:
            """Find relevant information from database (Reference implementation compatible)."""
            return await router.qdrant_find(
                query=query,
                collection=collection,
                limit=limit,
                score_threshold=score_threshold,
                search_mode=search_mode,
                filters=filters,
                note_types=note_types,
                tags=tags,
                include_relationships=include_relationships,
            )
    
    # Register qdrant_manage (enabled in standard mode)
    if "qdrant_manage" in enabled_tools:
        @app.tool()
        async def qdrant_manage(
            action: str,
            collection: str = None,
            document_id: str = None,
            note_id: str = None,
            include_vectors: bool = False,
            note_type: str = None,
            tags: List[str] = None,
            limit: int = 50,
            **kwargs
        ) -> dict:
            """Manage workspace collections and documents."""
            return await router.qdrant_manage(
                action=action,
                collection=collection,
                document_id=document_id,
                note_id=note_id,
                include_vectors=include_vectors,
                note_type=note_type,
                tags=tags,
                limit=limit,
                **kwargs
            )
    
    # Register qdrant_watch (enabled in standard mode)
    if "qdrant_watch" in enabled_tools:
        @app.tool()
        async def qdrant_watch(
            action: str,
            path: str = None,
            collection: str = None,
            watch_id: str = None,
            patterns: List[str] = None,
            auto_ingest: bool = True,
            recursive: bool = True,
            debounce_seconds: int = 5,
            **kwargs
        ) -> dict:
            """Manage folder watching for automatic document ingestion."""
            return await router.qdrant_watch(
                action=action,
                path=path,
                collection=collection,
                watch_id=watch_id,
                patterns=patterns,
                auto_ingest=auto_ingest,
                recursive=recursive,
                debounce_seconds=debounce_seconds,
                **kwargs
            )
    
    logger.info(f"Simplified tools registered successfully: {enabled_tools}")