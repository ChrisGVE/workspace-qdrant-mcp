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
import re
from enum import Enum
from typing import Dict, List, Optional, Any, Union

from common.core.error_handling import (
    ErrorRecoveryStrategy,
    with_error_handling,
    error_context,
)
from common.observability import (
    monitor_async,
    get_logger,
    record_operation,
)
from common.core.collection_naming import (
    CollectionRulesEnforcer,
    ValidationSource,
    OperationType,
    ValidationResult,
    CollectionRulesEnforcementError
)

# Import existing tool functions to route through simplified interface
from . import (
    documents,
    search,
    scratchbook,
    watch_management,
    grpc_tools,
)

# logger imported from loguru


# Search Scope System Integration
class SearchScope(Enum):
    """Enumeration of supported search scopes."""
    COLLECTION = "collection"  # Single specified collection
    PROJECT = "project"        # Collections belonging to current project
    WORKSPACE = "workspace"    # Project + global collections (excludes system)
    ALL = "all"               # All accessible collections (includes readonly, excludes system)
    MEMORY = "memory"         # Both system and project memory collections


class SearchScopeError(Exception):
    """Exception raised for search scope related errors."""
    pass


class ScopeValidationError(SearchScopeError):
    """Exception raised when scope/collection combination is invalid."""
    pass


class CollectionNotFoundError(SearchScopeError):
    """Exception raised when specified collection is not found."""
    pass


# Search scope constants
VALID_SEARCH_SCOPES = {scope.value for scope in SearchScope}
SYSTEM_MEMORY_PATTERN = r"^__[a-zA-Z0-9_]+$"
PROJECT_MEMORY_PATTERN = r"^[a-zA-Z0-9_]+-memory$"
GLOBAL_COLLECTIONS = [
    "algorithms", "codebase", "context", "documents", 
    "knowledge", "memory", "projects", "workspace"
]


def validate_search_scope(scope: str, collection: str) -> None:
    """Validate that a scope/collection combination is valid."""
    if not scope:
        raise ScopeValidationError("Search scope cannot be empty")
    
    if scope not in VALID_SEARCH_SCOPES:
        raise ScopeValidationError(
            f"Invalid search scope '{scope}'. Must be one of: {', '.join(VALID_SEARCH_SCOPES)}"
        )
    
    # Validate collection parameter based on scope
    if scope == SearchScope.COLLECTION.value:
        if not collection:
            raise ScopeValidationError(
                "Collection name is required when using 'collection' scope"
            )


def resolve_search_scope(scope: str, collection: str, client, config) -> List[str]:
    """Resolve search scope strings to collection lists."""
    validate_search_scope(scope, collection)
    
    if not client or not config:
        raise SearchScopeError("Client and config are required for scope resolution")
    
    if not hasattr(client, 'initialized') or not client.initialized:
        raise SearchScopeError("Client must be initialized before resolving search scope")
    
    try:
        if scope == SearchScope.COLLECTION.value:
            return _resolve_single_collection(collection, client)
        elif scope == SearchScope.PROJECT.value:
            return get_project_collections(client)
        elif scope == SearchScope.WORKSPACE.value:
            return get_workspace_collections(client)
        elif scope == SearchScope.ALL.value:
            return get_all_collections(client)
        elif scope == SearchScope.MEMORY.value:
            return get_memory_collections(client)
        else:
            raise ScopeValidationError(f"Unsupported scope: {scope}")
            
    except Exception as e:
        if isinstance(e, (ScopeValidationError, CollectionNotFoundError)):
            raise
        raise SearchScopeError(f"Failed to resolve search scope '{scope}': {e}") from e


def _resolve_single_collection(collection: str, client) -> List[str]:
    """Resolve a single collection name to a collection list."""
    available_collections = client.list_collections()
    
    if collection not in available_collections:
        raise CollectionNotFoundError(
            f"Collection '{collection}' not found. "
            f"Available collections: {', '.join(available_collections)}"
        )
    
    return [collection]


def get_project_collections(client) -> List[str]:
    """Get collections belonging to the current project."""
    project_collections = []
    all_collections = client.list_collections()
    
    # Get current project info
    try:
        project_info = client.get_project_info()
        if not project_info or not project_info.get("main_project"):
            return project_collections
        
        current_project = project_info["main_project"]
        
        # Filter for project collections (simple pattern matching)
        for collection_name in all_collections:
            if collection_name.startswith(f"{current_project}-"):
                project_collections.append(collection_name)
                
    except Exception:
        # If project info is not available, return empty list
        pass
    
    return sorted(project_collections)


def get_workspace_collections(client) -> List[str]:
    """Get workspace collections (project + global, excludes system)."""
    workspace_collections = []
    all_collections = client.list_collections()
    
    # Include project collections
    project_collections = get_project_collections(client)
    workspace_collections.extend(project_collections)
    
    # Include global collections and library collections
    for collection_name in all_collections:
        if (collection_name in GLOBAL_COLLECTIONS and 
            collection_name not in workspace_collections):
            workspace_collections.append(collection_name)
        
        # Include library collections (_prefix but not __)
        elif (collection_name.startswith("_") and 
              not collection_name.startswith("__") and
              collection_name not in workspace_collections):
            workspace_collections.append(collection_name)
    
    return sorted(workspace_collections)


def get_all_collections(client) -> List[str]:
    """Get all accessible collections (includes readonly, excludes system)."""
    accessible_collections = []
    all_collections = client.list_collections()
    
    for collection_name in all_collections:
        # Exclude only system collections (__ prefix)
        if not collection_name.startswith("__"):
            accessible_collections.append(collection_name)
    
    return sorted(accessible_collections)


def get_memory_collections(client) -> List[str]:
    """Get both system and project memory collections using the new collection type system."""
    from common.core.collections import MemoryCollectionManager
    from common.core.config import Config
    from common.core.collection_types import CollectionTypeClassifier
    
    try:
        # Use the collection type classifier to identify memory collections
        classifier = CollectionTypeClassifier()
        memory_collections = []
        all_collections = client.list_collections()
        
        for collection_name in all_collections:
            collection_info = classifier.get_collection_info(collection_name)
            if collection_info.is_memory_collection:
                memory_collections.append(collection_name)
        
        return sorted(memory_collections)
        
    except Exception as e:
        # Fallback to pattern matching if collection type system fails
        memory_collections = []
        all_collections = client.list_collections()
        
        # Compile patterns for memory collections
        system_memory_pattern = re.compile(SYSTEM_MEMORY_PATTERN)
        project_memory_pattern = re.compile(PROJECT_MEMORY_PATTERN)
        
        for collection_name in all_collections:
            # Check if collection matches memory patterns
            if (system_memory_pattern.match(collection_name) or 
                project_memory_pattern.match(collection_name)):
                memory_collections.append(collection_name)
        
        return sorted(memory_collections)


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
            return ["qdrant_store", "qdrant_find", "qdrant_manage", "qdrant_read"]
        else:  # FULL mode
            return []  # Return empty to indicate all tools enabled


class SimplifiedToolsRouter:
    """Routes simplified tool calls to existing comprehensive implementations."""
    
    def __init__(self, workspace_client, watch_tools_manager=None):
        self.workspace_client = workspace_client
        self.watch_tools_manager = watch_tools_manager
        self.mode = SimplifiedToolsMode.get_mode()
        
        # Initialize Task 181 Collection Rules Enforcer
        self.rules_enforcer = CollectionRulesEnforcer(getattr(workspace_client, 'config', None))
        
        # Update existing collections for validation
        if workspace_client and hasattr(workspace_client, 'list_collections'):
            try:
                existing_collections = workspace_client.list_collections()
                self.rules_enforcer.set_existing_collections(existing_collections)
            except Exception as e:
                logger.warning(f"Could not initialize existing collections for enforcer: {e}")
        
        logger.info(
            "Simplified tools router initialized with rules enforcement",
            mode=self.mode,
            enabled_tools=SimplifiedToolsMode.get_enabled_tools()
        )

    @monitor_async("qdrant_store", critical=True, timeout_warning=10.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_store")
    async def qdrant_store(
        self,
        content: str,
        collection: str,
        document_type: str = "text",
        source: str = "user_input",
        title: str = None,
        metadata: dict = None,
    ) -> Dict[str, Any]:
        """
        Universal content ingestion with clear source type classification.
        
        Routes to existing add_document functionality while maintaining all validation rules.
        Consolidates functionality from:
        - add_document_tool: Standard document ingestion
        - update_scratchbook_tool: Note and scratchbook management
        - process_document_via_grpc_tool: High-performance gRPC processing
        
        Args:
            content: Document text content to store
            collection: Target collection name (required)
            document_type: Type of content - "text", "code", "markdown", "note"
            source: Source of content - "user_input", "file", "api", "scratchbook"
            title: Optional title for the document
            metadata: Optional metadata dictionary for document organization
            
        Returns:
            dict: Storage result with success status, document_id, and metadata
            
        Example:
            ```python
            result = await qdrant_store(
                content="This is important information to remember",
                collection="my-project",
                document_type="text",
                source="user_input"
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        # Validate required parameters
        if not content or not content.strip():
            return {"error": "Content cannot be empty", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}
        
        # Task 181: Enforce collection write rules
        try:
            # Update existing collections list for validation
            existing = self.workspace_client.list_collections()
            self.rules_enforcer.set_existing_collections(existing)
            
            # Validate write access through rules enforcer (LLM source for MCP operations)
            validation_result = self.rules_enforcer.validate_collection_write(
                collection, ValidationSource.LLM
            )
            
            if not validation_result.is_valid:
                error_msg = validation_result.error_message
                return {"error": error_msg, "success": False, "validation_failed": True}
        
        except CollectionRulesEnforcementError as enforcement_error:
            logger.warning(f"Collection write blocked by rules enforcement: {collection} - {enforcement_error}")
            return {"error": str(enforcement_error), "success": False, "rules_violation": True}
        except Exception as validation_error:
            logger.error(f"Collection write validation failed: {collection} - {validation_error}")
            return {"error": f"Validation failed: {str(validation_error)}", "success": False}
        
        logger.debug(
            "Simplified store request",
            content_length=len(content),
            collection=collection,
            document_type=document_type,
            source=source
        )
        
        try:
            # Route to standard document ingestion with validation
            from .documents import add_document
            
            # Auto-detect chunking based on content size
            chunk_text = len(content) > 4000  # Chunk if > 4KB
            
            # Enhance metadata with source classification
            if metadata is None:
                metadata = {}
            metadata["document_type"] = document_type
            metadata["source"] = source
            if title:
                metadata["title"] = title
                
            # Route through existing add_document function which includes all validation
            result = await add_document(
                self.workspace_client,
                content=content,
                collection=collection,
                metadata=metadata,
                chunk_text=chunk_text
            )
            
            logger.info(
                "Document stored successfully",
                collection=collection,
                document_id=result.get("document_id"),
                document_type=document_type,
                source=source
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to store content", error=str(e), exc_info=True)
            return {"error": f"Storage failed: {str(e)}", "success": False}

    @monitor_async("qdrant_find", timeout_warning=2.0, slow_threshold=1.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_find")
    async def qdrant_find(
        self,
        query: str,
        search_scope: str = "project",
        collection: str = None,
        limit: int = 10,
        score_threshold: float = 0.7,
        document_type: str = None,
        include_metadata: bool = False,
        date_range: dict = None,
    ) -> list:
        """
        Search with precise scope control and filtering.
        
        Uses search scope architecture from task 175. Universal search and retrieval 
        that consolidates functionality from:
        - search_workspace_tool: Multi-collection semantic search
        - search_scratchbook_tool: Specialized note search with filtering
        - research_workspace: Advanced semantic research with context control
        - hybrid_search_advanced_tool: Configurable fusion methods
        - search_by_metadata_tool: Metadata-based filtering
        - search_via_grpc_tool: High-performance gRPC search
        
        Args:
            query: Natural language search query or exact text to find
            search_scope: Search scope - "collection", "project", "workspace", "all", "memory"
                - "collection": Search specific collection (collection param required)
                - "project": Search current project collections only
                - "workspace": Search project + global collections (excludes system)
                - "all": Search all collections (includes readonly, excludes system) 
                - "memory": Search both system and project memory collections
            collection: Specific collection (required for "collection" scope)
            limit: Maximum number of results to return (1-100)
            score_threshold: Minimum relevance score (0.0-1.0)
            document_type: Filter by document type ("text", "code", "markdown", "note")
            include_metadata: Include full document metadata in results
            date_range: Date range filter {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}
            
        Returns:
            list: Search results with ranked matches and metadata
            
        Example:
            ```python
            # Search within current project only
            results = await qdrant_find(
                query="authentication implementation",
                search_scope="project"
            )
            
            # Search specific collection
            results = await qdrant_find(
                query="API documentation",
                search_scope="collection",
                collection="my-project-docs"
            )
            ```
        """
        if not self.workspace_client:
            return [{"error": "Workspace client not initialized"}]
        
        # Validate required parameters
        if not query or not query.strip():
            return [{"error": "Query cannot be empty"}]
        
        # Validate parameters
        try:
            limit = int(limit) if isinstance(limit, str) else limit
            score_threshold = float(score_threshold) if isinstance(score_threshold, str) else score_threshold
            
            if limit <= 0:
                return [{"error": "limit must be greater than 0"}]
            if not (0.0 <= score_threshold <= 1.0):
                return [{"error": "score_threshold must be between 0.0 and 1.0"}]
        except (ValueError, TypeError) as e:
            return [{"error": f"Invalid parameter types: {e}"}]
        
        # Task 181: Validate read access to collections before search
        try:
            # Update existing collections list for validation
            existing = self.workspace_client.list_collections()
            self.rules_enforcer.set_existing_collections(existing)
            
            # Validate search scope and resolve target collections
            try:
                validate_search_scope(search_scope, collection)
                target_collections = resolve_search_scope(search_scope, collection, self.workspace_client, getattr(self.workspace_client, 'config', None))
            except (ScopeValidationError, CollectionNotFoundError, SearchScopeError) as scope_error:
                return [{"error": str(scope_error)}]
            
            # Validate read access to each target collection
            for target_collection in target_collections:
                validation_result = self.rules_enforcer.validate_operation(
                    OperationType.READ, target_collection, ValidationSource.LLM
                )
                if not validation_result.is_valid:
                    logger.warning(f"Read access denied to collection: {target_collection} - {validation_result.error_message}")
                    # Continue with other collections but log the violation
        
        except Exception as validation_error:
            logger.error(f"Search validation failed: {validation_error}")
            return [{"error": f"Search validation failed: {str(validation_error)}"}]
        
        logger.debug(
            "Simplified find request",
            query_length=len(query),
            search_scope=search_scope,
            collection=collection,
            document_type=document_type,
            include_metadata=include_metadata
        )
        
        try:
            # First resolve search scope to determine target collections
            try:
                target_collections = resolve_search_scope(
                    search_scope, collection, self.workspace_client, {}
                )
                logger.debug(
                    "Search scope resolved",
                    scope=search_scope,
                    resolved_collections=target_collections,
                    collection_count=len(target_collections)
                )
            except (SearchScopeError, ScopeValidationError, CollectionNotFoundError) as e:
                logger.error("Search scope resolution failed", error=str(e))
                return [{"error": f"Search scope error: {str(e)}"}]
            
            # Route to standard workspace search with scope-resolved collections
            from .search import search_workspace
            
            # Use hybrid search mode for best results
            result = await search_workspace(
                self.workspace_client,
                query=query,
                collections=target_collections,
                mode="hybrid",
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Apply document type filtering if specified
            if document_type and isinstance(result, dict) and result.get("results"):
                filtered_results = []
                for item in result["results"]:
                    payload = item.get("payload", {})
                    if payload.get("document_type") == document_type:
                        filtered_results.append(item)
                result["results"] = filtered_results
                result["total_results"] = len(filtered_results)
            
            # Apply date range filtering if specified
            if date_range and isinstance(result, dict) and result.get("results"):
                from datetime import datetime
                start_date = datetime.fromisoformat(date_range.get("start", "1970-01-01"))
                end_date = datetime.fromisoformat(date_range.get("end", "9999-12-31"))
                
                filtered_results = []
                for item in result["results"]:
                    payload = item.get("payload", {})
                    created_at = payload.get("created_at")
                    if created_at:
                        item_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                        if start_date <= item_date <= end_date:
                            filtered_results.append(item)
                    else:
                        # Include items without date if no strict filtering
                        filtered_results.append(item)
                result["results"] = filtered_results
                result["total_results"] = len(filtered_results)
            
            # Convert result to list format as specified
            if isinstance(result, dict) and result.get("results"):
                search_results = result["results"]
                if not include_metadata:
                    # Strip metadata if not requested
                    for item in search_results:
                        if "payload" in item and isinstance(item["payload"], dict):
                            # Keep only essential fields
                            essential_fields = {"content", "title", "document_type", "source"}
                            filtered_payload = {k: v for k, v in item["payload"].items() 
                                              if k in essential_fields}
                            item["payload"] = filtered_payload
                return search_results
            else:
                return []
            
        except Exception as e:
            logger.error("Failed to find information", error=str(e), exc_info=True)
            return [{"error": f"Search failed: {str(e)}"}]


    @monitor_async("qdrant_manage", timeout_warning=5.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_manage")
    async def qdrant_manage(
        self,
        action: str,
        collection: str = None,
        new_name: str = None,
    ) -> dict:
        """
        System status and collection management.
        
        Routes to existing workspace status and collection tools. Consolidates functionality from:
        - workspace_status: Get workspace and collection status
        - list_workspace_collections: List available collections
        - create_collection: Create new collections
        - delete_collection: Remove collections
        - rename_collection: Rename existing collections
        
        Args:
            action: Management action - "status", "list", "create", "delete", "rename"
            collection: Target collection name (required for create/delete/rename)
            new_name: New name for collection (required for rename)
            
        Returns:
            dict: Management operation results
            
        Example:
            ```python
            # Get workspace status
            status = await qdrant_manage(action="status")
            
            # List available collections
            collections = await qdrant_manage(action="list")
            
            # Create new collection
            result = await qdrant_manage(action="create", collection="new-collection")
            
            # Rename collection
            result = await qdrant_manage(
                action="rename",
                collection="old-name",
                new_name="new-name"
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        # Validate required parameters
        if not action or not action.strip():
            return {"error": "Action is required", "success": False}
        
        logger.debug("Management action requested", action=action, collection=collection)
        
        try:
            if action == "status":
                # Route to workspace status through existing functionality
                status = await self.workspace_client.get_status()
                logger.info("Workspace status retrieved", connected=status.get("connected"))
                return status
                
            elif action == "list":
                # Route to list collections through existing functionality
                collections = self.workspace_client.list_collections()
                logger.info("Collections listed", count=len(collections))
                return {"collections": collections, "count": len(collections), "success": True}
                
            elif action == "create":
                if not collection or not collection.strip():
                    return {"error": "Collection name required for create action", "success": False}
                
                # Task 181: Enforce collection rules before creation
                try:
                    # Update existing collections list for validation
                    existing = self.workspace_client.list_collections()
                    self.rules_enforcer.set_existing_collections(existing)
                    
                    # Validate creation through rules enforcer (LLM source for MCP operations)
                    validation_result = self.rules_enforcer.validate_collection_creation(
                        collection, ValidationSource.LLM
                    )
                    
                    if not validation_result.is_valid:
                        error_msg = validation_result.error_message
                        if validation_result.suggested_alternatives:
                            error_msg += f" Suggested alternatives: {', '.join(validation_result.suggested_alternatives)}"
                        return {"error": error_msg, "success": False, "validation_failed": True}
                    
                    # Create collection through client (additional client-side validation)
                    await self.workspace_client.create_collection(collection)
                    
                    # Update enforcer with new collection
                    updated_collections = self.workspace_client.list_collections()
                    self.rules_enforcer.set_existing_collections(updated_collections)
                    
                    logger.info(f"Collection created with rules enforcement: {collection}")
                    return {"collection": collection, "success": True, "message": "Collection created successfully"}
                    
                except CollectionRulesEnforcementError as enforcement_error:
                    logger.warning(f"Collection creation blocked by rules enforcement: {collection} - {enforcement_error}")
                    return {"error": str(enforcement_error), "success": False, "rules_violation": True}
                except Exception as create_error:
                    logger.error(f"Collection creation failed: {collection} - {create_error}")
                    return {"error": f"Failed to create collection: {str(create_error)}", "success": False}
                
            elif action == "delete":
                if not collection or not collection.strip():
                    return {"error": "Collection name required for delete action", "success": False}
                
                # Task 181: Enforce collection rules before deletion
                try:
                    # Update existing collections list for validation
                    existing = self.workspace_client.list_collections()
                    self.rules_enforcer.set_existing_collections(existing)
                    
                    # Validate deletion through rules enforcer (LLM source for MCP operations)
                    validation_result = self.rules_enforcer.validate_collection_deletion(
                        collection, ValidationSource.LLM
                    )
                    
                    if not validation_result.is_valid:
                        error_msg = validation_result.error_message
                        if validation_result.warning_message:
                            error_msg += f" Warning: {validation_result.warning_message}"
                        return {"error": error_msg, "success": False, "validation_failed": True}
                    
                    # Delete collection through client (additional client-side validation)
                    await self.workspace_client.delete_collection(collection)
                    
                    # Update enforcer with removed collection
                    updated_collections = self.workspace_client.list_collections()
                    self.rules_enforcer.set_existing_collections(updated_collections)
                    
                    logger.info(f"Collection deleted with rules enforcement: {collection}")
                    return {"collection": collection, "success": True, "message": "Collection deleted successfully"}
                    
                except CollectionRulesEnforcementError as enforcement_error:
                    logger.warning(f"Collection deletion blocked by rules enforcement: {collection} - {enforcement_error}")
                    return {"error": str(enforcement_error), "success": False, "rules_violation": True}
                except Exception as delete_error:
                    logger.error(f"Collection deletion failed: {collection} - {delete_error}")
                    return {"error": f"Failed to delete collection: {str(delete_error)}", "success": False}
                
            elif action == "rename":
                if not collection or not collection.strip():
                    return {"error": "Collection name required for rename action", "success": False}
                if not new_name or not new_name.strip():
                    return {"error": "New name required for rename action", "success": False}
                
                # Route through client collection management with validation
                try:
                    # Check if source collection exists
                    existing = self.workspace_client.list_collections()
                    if collection not in existing:
                        return {"error": f"Collection '{collection}' does not exist", "success": False}
                    if new_name in existing:
                        return {"error": f"Collection '{new_name}' already exists", "success": False}
                    
                    # Rename through client (which includes all validation)
                    await self.workspace_client.rename_collection(collection, new_name)
                    logger.info("Collection renamed", old_name=collection, new_name=new_name)
                    return {
                        "old_name": collection, 
                        "new_name": new_name, 
                        "success": True, 
                        "message": "Collection renamed successfully"
                    }
                    
                except Exception as rename_error:
                    logger.error("Collection rename failed", collection=collection, new_name=new_name, error=str(rename_error))
                    return {"error": f"Failed to rename collection: {str(rename_error)}", "success": False}
                
            else:
                valid_actions = ["status", "list", "create", "delete", "rename"]
                return {
                    "error": f"Unknown management action: {action}. Valid actions: {', '.join(valid_actions)}", 
                    "success": False
                }
                
        except Exception as e:
            logger.error("Management action failed", action=action, error=str(e), exc_info=True)
            return {"error": f"Management action failed: {str(e)}", "success": False}

    @monitor_async("qdrant_read", timeout_warning=5.0)
    @with_error_handling(ErrorRecoveryStrategy.database_strategy(), "qdrant_read")
    async def qdrant_read(
        self,
        action: str,
        collection: str,
        document_id: str = None,
        limit: int = 100,
        include_metadata: bool = True,
        sort_by: str = "ingestion_date",
    ) -> dict:
        """
        Direct document retrieval without search.
        
        Routes to existing get_document functionality. Consolidates functionality from:
        - get_document_tool: Retrieve specific documents
        - list_collection_documents: List documents in collection
        - get_document_by_metadata: Find documents by metadata
        
        Args:
            action: Read action - "get", "list", "find_by_metadata"
            collection: Target collection name (required)
            document_id: Document identifier (required for "get")
            limit: Maximum results for listing operations
            include_metadata: Include full document metadata in response
            sort_by: Sort order for listing - "ingestion_date", "title", "document_id"
            
        Returns:
            dict: Read operation results
            
        Example:
            ```python
            # Get specific document
            doc = await qdrant_read(
                action="get",
                collection="my-project",
                document_id="doc-123"
            )
            
            # List documents in collection
            docs = await qdrant_read(
                action="list",
                collection="my-project",
                limit=50
            )
            ```
        """
        if not self.workspace_client:
            return {"error": "Workspace client not initialized"}
        
        # Validate required parameters
        if not action or not action.strip():
            return {"error": "Action is required", "success": False}
        if not collection or not collection.strip():
            return {"error": "Collection name is required", "success": False}
        
        logger.debug("Read action requested", action=action, collection=collection, document_id=document_id)
        
        try:
            if action == "get":
                if not document_id or not document_id.strip():
                    return {"error": "document_id required for get action", "success": False}
                
                # Route to get_document with validation
                from .documents import get_document
                result = await get_document(
                    self.workspace_client,
                    document_id=document_id,
                    collection=collection,
                    include_vectors=False  # Don't include vectors unless specifically needed
                )
                
                # Enhance result with metadata control
                if not include_metadata and isinstance(result, dict) and "payload" in result:
                    # Strip metadata if not requested, keep only essential fields
                    essential_fields = {"content", "title", "document_type", "source"}
                    if isinstance(result["payload"], dict):
                        filtered_payload = {k: v for k, v in result["payload"].items() 
                                          if k in essential_fields}
                        result["payload"] = filtered_payload
                
                logger.info("Document retrieved", document_id=document_id, collection=collection)
                return result
                
            elif action == "list":
                # Get collection info and list documents
                try:
                    # Check if collection exists
                    existing = self.workspace_client.list_collections()
                    if collection not in existing:
                        return {"error": f"Collection '{collection}' does not exist", "success": False}
                    
                    # Get collection info through client
                    collection_info = await self.workspace_client.get_collection_info(collection)
                    
                    if not collection_info or "documents" not in collection_info:
                        return {"documents": [], "count": 0, "collection": collection, "success": True}
                    
                    # Get documents with optional sorting
                    documents = collection_info["documents"]
                    
                    # Apply sorting
                    if sort_by == "ingestion_date":
                        documents = sorted(documents, key=lambda x: x.get("created_at", ""), reverse=True)
                    elif sort_by == "title":
                        documents = sorted(documents, key=lambda x: x.get("title", x.get("id", "")))
                    elif sort_by == "document_id":
                        documents = sorted(documents, key=lambda x: x.get("id", ""))
                    
                    # Apply limit
                    if limit > 0:
                        documents = documents[:limit]
                    
                    # Apply metadata filtering
                    if not include_metadata:
                        essential_fields = {"id", "content", "title", "document_type", "source"}
                        filtered_docs = []
                        for doc in documents:
                            if isinstance(doc, dict):
                                filtered_doc = {k: v for k, v in doc.items() if k in essential_fields}
                                filtered_docs.append(filtered_doc)
                        documents = filtered_docs
                    
                    logger.info("Documents listed", collection=collection, count=len(documents))
                    return {
                        "documents": documents,
                        "count": len(documents),
                        "collection": collection,
                        "sort_by": sort_by,
                        "success": True
                    }
                    
                except Exception as list_error:
                    logger.error("Document listing failed", collection=collection, error=str(list_error))
                    return {"error": f"Failed to list documents: {str(list_error)}", "success": False}
                
            elif action == "find_by_metadata":
                # Route to metadata search functionality
                from .search import search_collection_by_metadata
                
                # This would require metadata filters to be passed, but they're not in the spec
                # Return error for now or implement basic metadata search
                return {"error": "find_by_metadata action not fully implemented - use qdrant_find instead", "success": False}
                
            else:
                valid_actions = ["get", "list", "find_by_metadata"]
                return {
                    "error": f"Unknown read action: {action}. Valid actions: {', '.join(valid_actions)}", 
                    "success": False
                }
                
        except Exception as e:
            logger.error("Read action failed", action=action, error=str(e), exc_info=True)
            return {"error": f"Read action failed: {str(e)}", "success": False}


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
            content: str,
            collection: str,
            document_type: str = "text",
            source: str = "user_input",
            title: str = None,
            metadata: dict = None,
        ) -> dict:
            """Universal content ingestion with clear source type classification."""
            return await router.qdrant_store(
                content=content,
                collection=collection,
                document_type=document_type,
                source=source,
                title=title,
                metadata=metadata,
            )
    
    # Register qdrant_find (always enabled in simplified modes)
    if "qdrant_find" in enabled_tools:
        @app.tool()
        async def qdrant_find(
            query: str,
            search_scope: str = "project",
            collection: str = None,
            limit: int = 10,
            score_threshold: float = 0.7,
            document_type: str = None,
            include_metadata: bool = False,
            date_range: dict = None,
        ) -> list:
            """Search with precise scope control and filtering."""
            return await router.qdrant_find(
                query=query,
                search_scope=search_scope,
                collection=collection,
                limit=limit,
                score_threshold=score_threshold,
                document_type=document_type,
                include_metadata=include_metadata,
                date_range=date_range,
            )
    
    # Register qdrant_manage (enabled in standard mode)
    if "qdrant_manage" in enabled_tools:
        @app.tool()
        async def qdrant_manage(
            action: str,
            collection: str = None,
            new_name: str = None,
        ) -> dict:
            """System status and collection management."""
            return await router.qdrant_manage(
                action=action,
                collection=collection,
                new_name=new_name,
            )
    
    # Register qdrant_read (enabled in standard mode) - replaces qdrant_watch
    if "qdrant_manage" in enabled_tools:  # Use same condition as manage for 4-tool architecture
        @app.tool()
        async def qdrant_read(
            action: str,
            collection: str,
            document_id: str = None,
            limit: int = 100,
            include_metadata: bool = True,
            sort_by: str = "ingestion_date",
        ) -> dict:
            """Direct document retrieval without search."""
            return await router.qdrant_read(
                action=action,
                collection=collection,
                document_id=document_id,
                limit=limit,
                include_metadata=include_metadata,
                sort_by=sort_by,
            )
    
    logger.info(f"Simplified tools registered successfully: {enabled_tools}")