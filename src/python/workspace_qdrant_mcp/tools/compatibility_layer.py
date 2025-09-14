"""
Compatibility layer for tool set simplification.

This module provides backward compatibility support for existing tool calls
when running in simplified mode. It includes:

1. Tool mapping: Maps old tool calls to new simplified interface
2. Deprecation warnings: Logs warnings for deprecated tool usage
3. Parameter translation: Converts old parameters to new interface
4. Feature toggles: Allows fine-grained control over tool availability

The compatibility layer ensures existing workflows continue to work while
encouraging migration to the simplified interface.
"""

import os
from typing import Dict, List, Optional, Any, Callable
from functools import wraps

from loguru import logger
from .simplified_interface import SimplifiedToolsMode, get_simplified_router

# logger imported from loguru


class CompatibilityMapping:
    """Maps legacy tool calls to simplified interface."""
    
    # Mapping from old tool names to new simplified tools and their parameters
    TOOL_MAPPINGS = {
        # Document operations -> qdrant_store
        "add_document_tool": {
            "new_tool": "qdrant_store",
            "param_mapping": {
                "content": "information",
                "document_id": "document_id", 
                "metadata": "metadata",
                "collection": "collection",
                "chunk_text": "chunk_text"
            },
            "default_params": {"note_type": "document"}
        },
        "update_scratchbook_tool": {
            "new_tool": "qdrant_store",
            "param_mapping": {
                "content": "information",
                "note_id": "document_id",
                "title": "title",
                "tags": "tags",
                "note_type": "note_type"
            },
            "default_params": {"note_type": "scratchbook", "collection": "scratchbook"}
        },
        "process_document_via_grpc_tool": {
            "new_tool": "qdrant_store", 
            "param_mapping": {
                "file_path": "information",  # Will need file reading
                "collection": "collection",
                "metadata": "metadata",
                "document_id": "document_id",
                "chunk_text": "chunk_text"
            },
            "default_params": {"note_type": "document"},
            "preprocessing": "read_file_content"
        },
        
        # Search operations -> qdrant_find
        "search_workspace_tool": {
            "new_tool": "qdrant_find",
            "param_mapping": {
                "query": "query",
                "collections": "collection",  # Take first collection if list
                "mode": "search_mode",
                "limit": "limit", 
                "score_threshold": "score_threshold"
            }
        },
        "search_scratchbook_tool": {
            "new_tool": "qdrant_find", 
            "param_mapping": {
                "query": "query",
                "note_types": "note_types",
                "tags": "tags", 
                "project_name": "collection",
                "limit": "limit",
                "mode": "search_mode"
            }
        },
        "research_workspace": {
            "new_tool": "qdrant_find",
            "param_mapping": {
                "query": "query",
                "target_collection": "collection",
                "include_relationships": "include_relationships",
                "limit": "limit",
                "score_threshold": "score_threshold"
            },
            "default_params": {"search_mode": "hybrid"}
        },
        "hybrid_search_advanced_tool": {
            "new_tool": "qdrant_find",
            "param_mapping": {
                "query": "query",
                "collection": "collection", 
                "limit": "limit",
                "score_threshold": "score_threshold"
            },
            "default_params": {"search_mode": "hybrid"}
        },
        "search_by_metadata_tool": {
            "new_tool": "qdrant_find",
            "param_mapping": {
                "collection": "collection",
                "metadata_filter": "filters",
                "limit": "limit"
            },
            "default_params": {"query": "*", "search_mode": "exact"}
        },
        "search_via_grpc_tool": {
            "new_tool": "qdrant_find",
            "param_mapping": {
                "query": "query",
                "collections": "collection", # Take first collection if list
                "mode": "search_mode",
                "limit": "limit",
                "score_threshold": "score_threshold"  
            }
        },
        
        # Management operations -> qdrant_manage
        "workspace_status": {
            "new_tool": "qdrant_manage",
            "param_mapping": {},
            "default_params": {"action": "status"}
        },
        "list_workspace_collections": {
            "new_tool": "qdrant_manage", 
            "param_mapping": {},
            "default_params": {"action": "collections"}
        },
        "get_document_tool": {
            "new_tool": "qdrant_manage",
            "param_mapping": {
                "document_id": "document_id",
                "collection": "collection",
                "include_vectors": "include_vectors"
            },
            "default_params": {"action": "get"}
        },
        "list_scratchbook_notes_tool": {
            "new_tool": "qdrant_manage",
            "param_mapping": {
                "project_name": "collection",
                "note_type": "note_type",
                "tags": "tags", 
                "limit": "limit"
            },
            "default_params": {"action": "list_notes"}
        },
        "delete_scratchbook_note_tool": {
            "new_tool": "qdrant_manage",
            "param_mapping": {
                "note_id": "note_id",
                "project_name": "collection"
            },
            "default_params": {"action": "delete"}
        },
        
        # Watch operations -> qdrant_watch
        "add_watch_folder": {
            "new_tool": "qdrant_watch",
            "param_mapping": {
                "path": "path", 
                "collection": "collection",
                "patterns": "patterns",
                "auto_ingest": "auto_ingest",
                "recursive": "recursive",
                "debounce_seconds": "debounce_seconds",
                "watch_id": "watch_id"
            },
            "default_params": {"action": "add"}
        },
        "remove_watch_folder": {
            "new_tool": "qdrant_watch",
            "param_mapping": {
                "watch_id": "watch_id"
            },
            "default_params": {"action": "remove"}
        },
        "list_watched_folders": {
            "new_tool": "qdrant_watch",
            "param_mapping": {
                "active_only": "active_only",
                "collection": "collection",
                "include_stats": "include_stats"
            },
            "default_params": {"action": "list"}
        },
        "configure_watch_settings": {
            "new_tool": "qdrant_watch",
            "param_mapping": {
                "watch_id": "watch_id",
                "patterns": "patterns", 
                "auto_ingest": "auto_ingest",
                "recursive": "recursive",
                "debounce_seconds": "debounce_seconds"
            },
            "default_params": {"action": "configure"}
        },
        "get_watch_status": {
            "new_tool": "qdrant_watch", 
            "param_mapping": {
                "watch_id": "watch_id"
            },
            "default_params": {"action": "status"}
        },
        "validate_watch_path": {
            "new_tool": "qdrant_watch",
            "param_mapping": {
                "path": "path"
            },
            "default_params": {"action": "validate"}
        },
    }


def create_compatibility_wrapper(old_tool_name: str, original_function: Callable):
    """Create a compatibility wrapper that routes old tool calls to simplified interface."""
    
    @wraps(original_function)
    async def wrapper(*args, **kwargs):
        # Check if we're in simplified mode
        if not SimplifiedToolsMode.is_simplified_mode():
            # Full mode - use original function
            return await original_function(*args, **kwargs)
        
        # Simplified mode - route through compatibility layer
        mapping = CompatibilityMapping.TOOL_MAPPINGS.get(old_tool_name)
        if not mapping:
            # Tool not mapped - log warning and use original function
            logger.warning(
                f"Tool '{old_tool_name}' not mapped to simplified interface, using original implementation"
            )
            return await original_function(*args, **kwargs)
        
        # Log deprecation warning
        logger.warning(
            f"Deprecated tool '{old_tool_name}' called - consider migrating to '{mapping['new_tool']}'",
            tool=old_tool_name,
            new_tool=mapping["new_tool"],
            simplified_mode=SimplifiedToolsMode.get_mode()
        )
        
        # Get simplified router
        router = get_simplified_router()
        if not router:
            logger.error("Simplified router not initialized - falling back to original function")
            return await original_function(*args, **kwargs)
        
        try:
            # Map parameters
            new_kwargs = {}
            
            # Apply default parameters first
            if "default_params" in mapping:
                new_kwargs.update(mapping["default_params"])
            
            # Map old parameters to new interface
            param_mapping = mapping.get("param_mapping", {})
            for old_param, new_param in param_mapping.items():
                if old_param in kwargs:
                    value = kwargs[old_param]
                    
                    # Handle special parameter conversions
                    if old_param == "collections" and isinstance(value, list) and value:
                        # Take first collection from list
                        value = value[0]
                    elif old_param == "mode" and value in ["dense", "sparse"]:
                        # Convert internal modes to public modes  
                        mode_mapping = {"dense": "semantic", "sparse": "keyword"}
                        value = mode_mapping.get(value, value)
                    
                    new_kwargs[new_param] = value
            
            # Handle preprocessing if needed
            preprocessing = mapping.get("preprocessing")
            if preprocessing == "read_file_content":
                # For file-based tools, read file content
                file_path = kwargs.get("file_path")
                if file_path:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            new_kwargs["information"] = f.read()
                    except Exception as e:
                        return {"error": f"Failed to read file {file_path}: {e}"}
            
            # Route to appropriate simplified tool
            new_tool = mapping["new_tool"]
            if new_tool == "qdrant_store":
                return await router.qdrant_store(**new_kwargs)
            elif new_tool == "qdrant_find": 
                return await router.qdrant_find(**new_kwargs)
            elif new_tool == "qdrant_manage":
                return await router.qdrant_manage(**new_kwargs)
            elif new_tool == "qdrant_watch":
                return await router.qdrant_watch(**new_kwargs)
            else:
                logger.error(f"Unknown simplified tool: {new_tool}")
                return await original_function(*args, **kwargs)
                
        except Exception as e:
            logger.error(
                f"Failed to route {old_tool_name} through simplified interface: {e}",
                error=str(e),
                exc_info=True
            )
            # Fall back to original function
            return await original_function(*args, **kwargs)
    
    return wrapper


def should_disable_tool(tool_name: str) -> bool:
    """Check if a tool should be disabled in current mode."""
    mode = SimplifiedToolsMode.get_mode()
    
    # In full mode, all tools are enabled
    if mode == SimplifiedToolsMode.FULL:
        return False
    
    # In simplified modes, disable tools that are mapped to simplified interface
    if tool_name in CompatibilityMapping.TOOL_MAPPINGS:
        return True
    
    # Disable advanced tools not mapped to simplified interface
    advanced_tools = [
        "configure_advanced_watch",
        "validate_watch_configuration", 
        "get_watch_health_status",
        "trigger_watch_recovery",
        "get_watch_sync_status",
        "force_watch_sync",
        "get_watch_change_history",
        "get_error_stats_tool"
    ]
    
    if tool_name in advanced_tools:
        # Only enable in standard mode if explicitly requested
        enable_advanced = os.getenv("QDRANT_MCP_ENABLE_ADVANCED", "false").lower() == "true"
        return not enable_advanced
    
    return False


def get_compatibility_message(tool_name: str) -> str:
    """Get migration message for deprecated tool."""
    mapping = CompatibilityMapping.TOOL_MAPPINGS.get(tool_name)
    if not mapping:
        return f"Tool '{tool_name}' is not available in simplified mode"
    
    new_tool = mapping["new_tool"]
    return (
        f"Tool '{tool_name}' is deprecated. "
        f"Use '{new_tool}' instead for better performance and simplified interface."
    )


class ToolRegistrationManager:
    """Manages conditional tool registration based on mode."""
    
    def __init__(self, app):
        self.app = app
        self.mode = SimplifiedToolsMode.get_mode()
        self.registered_tools = set()
    
    def should_register_tool(self, tool_name: str) -> bool:
        """Check if a tool should be registered in current mode."""
        if self.mode == SimplifiedToolsMode.FULL:
            return True
        
        return not should_disable_tool(tool_name)
    
    def register_tool_conditionally(self, tool_name: str, tool_function: Callable):
        """Register tool only if appropriate for current mode."""
        if self.should_register_tool(tool_name):
            # Wrap with compatibility layer if needed
            if SimplifiedToolsMode.is_simplified_mode() and tool_name in CompatibilityMapping.TOOL_MAPPINGS:
                wrapped_function = create_compatibility_wrapper(tool_name, tool_function)
                self.app.tool(name=tool_name)(wrapped_function)
            else:
                self.app.tool(name=tool_name)(tool_function)
            
            self.registered_tools.add(tool_name)
            logger.debug(f"Registered tool: {tool_name}", mode=self.mode)
        else:
            logger.debug(f"Skipped tool registration: {tool_name}", mode=self.mode)
    
    def log_registration_summary(self):
        """Log summary of tool registration."""
        logger.info(
            "Tool registration completed", 
            mode=self.mode,
            registered_count=len(self.registered_tools),
            tools=sorted(list(self.registered_tools))
        )