"""
Validation decorators for MCP tools with project isolation.

This module provides convenient decorators that can be applied to MCP tool functions
to automatically validate project isolation and access control before execution.

Key Features:
    - @require_project_access: Validates project access for operations
    - @require_collection_access: Validates collection ownership and access
    - @require_document_access: Validates document-level access permissions
    - @log_security_events: Logs security events for audit trails
    - Automatic error handling with sanitized messages

Example:
    ```python
    from workspace_qdrant_mcp.validation.decorators import require_project_access

    @app.tool()
    @require_project_access(project_param="project_name", operation="search")
    async def search_project_documents(project_name: str, query: str) -> Dict:
        # This function will only execute if user has access to the project
        return await perform_search(project_name, query)
    ```
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from .project_isolation import (
    ProjectIsolationValidator,
    UserContext,
    SecurityError,
    get_validator
)


def require_project_access(
    project_param: str = "project_name",
    operation: str = "access",
    user_context_param: str = "user_context",
    workspace_client_param: str = "workspace_client",
    allow_none_project: bool = False
) -> Callable:
    """
    Decorator to require project access validation for MCP tool functions.

    Args:
        project_param: Name of the parameter containing the project name
        operation: Type of operation being performed
        user_context_param: Name of parameter containing user context (optional)
        workspace_client_param: Name of parameter containing workspace client (optional)
        allow_none_project: Whether to allow None project names

    Returns:
        Callable: Decorated function with project access validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Get function signature for parameter inspection
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract parameters
                project_name = bound_args.arguments.get(project_param)
                user_context = bound_args.arguments.get(user_context_param)
                workspace_client = bound_args.arguments.get(workspace_client_param)

                # Handle None project name
                if project_name is None and not allow_none_project:
                    logger.warning(f"Project access validation skipped - no project name provided for {func.__name__}")
                    return {"error": "Project name is required"}

                if project_name is not None:
                    # Get validator and perform validation
                    validator = get_validator(workspace_client)

                    if not validator.validate_project_access(project_name, user_context, operation):
                        error_msg = validator.sanitize_error_message(
                            SecurityError(f"Access denied for project '{project_name}'"),
                            user_context
                        )
                        logger.warning(
                            f"Project access denied for {func.__name__}",
                            project_name=project_name,
                            operation=operation,
                            user_id=user_context.user_id if user_context else None
                        )
                        return {"error": error_msg}

                # Execute original function
                return await func(*args, **kwargs)

            except SecurityError as e:
                validator = get_validator()
                error_msg = validator.sanitize_error_message(e, user_context)
                logger.error(f"Security error in {func.__name__}: {error_msg}")
                return {"error": error_msg}

            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return {"error": "Operation failed due to an internal error"}

        return wrapper
    return decorator


def require_collection_access(
    collection_param: str = "collection_name",
    project_param: str = "project_name",
    operation: str = "access",
    user_context_param: str = "user_context",
    workspace_client_param: str = "workspace_client",
    allow_shared: bool = True,
    auto_detect_project: bool = True
) -> Callable:
    """
    Decorator to require collection access validation for MCP tool functions.

    Args:
        collection_param: Name of parameter containing collection name
        project_param: Name of parameter containing project name
        operation: Type of operation being performed
        user_context_param: Name of parameter containing user context (optional)
        workspace_client_param: Name of parameter containing workspace client (optional)
        allow_shared: Whether to allow shared collection access
        auto_detect_project: Whether to auto-detect project from collection name

    Returns:
        Callable: Decorated function with collection access validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Get function signature for parameter inspection
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract parameters
                collection_name = bound_args.arguments.get(collection_param)
                project_name = bound_args.arguments.get(project_param)
                user_context = bound_args.arguments.get(user_context_param)
                workspace_client = bound_args.arguments.get(workspace_client_param)

                if not collection_name:
                    return {"error": "Collection name is required"}

                # Auto-detect project from collection name if needed
                if not project_name and auto_detect_project:
                    project_name = _extract_project_from_collection(collection_name)

                if not project_name:
                    logger.warning(f"Collection validation skipped - no project context for {func.__name__}")
                    return {"error": "Project context is required"}

                # Get validator and perform validation
                validator = get_validator(workspace_client)

                if not validator.validate_collection_ownership(
                    collection_name, project_name, user_context, allow_shared
                ):
                    error_msg = validator.sanitize_error_message(
                        SecurityError(f"Collection '{collection_name}' access denied"),
                        user_context
                    )
                    logger.warning(
                        f"Collection access denied for {func.__name__}",
                        collection_name=collection_name,
                        project_name=project_name,
                        operation=operation,
                        user_id=user_context.user_id if user_context else None
                    )
                    return {"error": error_msg}

                # Execute original function
                return await func(*args, **kwargs)

            except SecurityError as e:
                validator = get_validator()
                error_msg = validator.sanitize_error_message(e, user_context)
                logger.error(f"Security error in {func.__name__}: {error_msg}")
                return {"error": error_msg}

            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return {"error": "Operation failed due to an internal error"}

        return wrapper
    return decorator


def require_document_access(
    document_param: str = "document_id",
    collection_param: str = "collection_name",
    project_param: str = "project_name",
    operation: str = "read",
    user_context_param: str = "user_context",
    workspace_client_param: str = "workspace_client"
) -> Callable:
    """
    Decorator to require document access validation for MCP tool functions.

    Args:
        document_param: Name of parameter containing document ID
        collection_param: Name of parameter containing collection name
        project_param: Name of parameter containing project name
        operation: Type of operation being performed (read, write, delete)
        user_context_param: Name of parameter containing user context (optional)
        workspace_client_param: Name of parameter containing workspace client (optional)

    Returns:
        Callable: Decorated function with document access validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Get function signature for parameter inspection
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract parameters
                document_id = bound_args.arguments.get(document_param)
                collection_name = bound_args.arguments.get(collection_param)
                project_name = bound_args.arguments.get(project_param)
                user_context = bound_args.arguments.get(user_context_param)
                workspace_client = bound_args.arguments.get(workspace_client_param)

                if not document_id:
                    return {"error": "Document ID is required"}

                if not collection_name:
                    return {"error": "Collection name is required"}

                # Auto-detect project from collection name if needed
                if not project_name:
                    project_name = _extract_project_from_collection(collection_name)

                if not project_name:
                    return {"error": "Project context is required"}

                # Get validator and perform validation
                validator = get_validator(workspace_client)

                if not validator.validate_document_access(
                    document_id, collection_name, project_name, user_context, operation
                ):
                    error_msg = validator.sanitize_error_message(
                        SecurityError(f"Document '{document_id}' access denied"),
                        user_context
                    )
                    logger.warning(
                        f"Document access denied for {func.__name__}",
                        document_id=document_id,
                        collection_name=collection_name,
                        project_name=project_name,
                        operation=operation,
                        user_id=user_context.user_id if user_context else None
                    )
                    return {"error": error_msg}

                # Execute original function
                return await func(*args, **kwargs)

            except SecurityError as e:
                validator = get_validator()
                error_msg = validator.sanitize_error_message(e, user_context)
                logger.error(f"Security error in {func.__name__}: {error_msg}")
                return {"error": error_msg}

            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return {"error": "Operation failed due to an internal error"}

        return wrapper
    return decorator


def log_security_events(
    event_type: str = "mcp_tool_access",
    include_args: bool = False,
    include_result: bool = False
) -> Callable:
    """
    Decorator to log security events for MCP tool access.

    Args:
        event_type: Type of security event to log
        include_args: Whether to include function arguments in log
        include_result: Whether to include function result in log

    Returns:
        Callable: Decorated function with security event logging
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = None
            result = None
            error = None

            try:
                start_time = logger.bind()._logger.opt().record["time"]  # Current time

                # Execute original function
                result = await func(*args, **kwargs)

                # Log successful execution
                validator = get_validator()
                log_details = {
                    "function_name": func.__name__,
                    "success": True,
                    "has_error": "error" in result if isinstance(result, dict) else False
                }

                if include_args:
                    # Sanitize arguments for logging (remove sensitive data)
                    safe_kwargs = _sanitize_log_args(kwargs)
                    log_details["arguments"] = safe_kwargs

                if include_result and isinstance(result, dict):
                    # Include basic result info without sensitive data
                    log_details["result_keys"] = list(result.keys())
                    if "error" in result:
                        log_details["error_message"] = result["error"]

                validator._log_security_event(
                    event_type=event_type,
                    details=log_details,
                    severity="INFO"
                )

                return result

            except Exception as e:
                error = e
                # Log error execution
                validator = get_validator()
                log_details = {
                    "function_name": func.__name__,
                    "success": False,
                    "error": str(e)
                }

                validator._log_security_event(
                    event_type=f"{event_type}_error",
                    details=log_details,
                    severity="ERROR"
                )

                raise e

        return wrapper
    return decorator


def validate_workspace_client(
    workspace_client_param: str = "workspace_client",
    require_initialized: bool = True
) -> Callable:
    """
    Decorator to validate workspace client before MCP tool execution.

    Args:
        workspace_client_param: Name of parameter containing workspace client
        require_initialized: Whether workspace client must be initialized

    Returns:
        Callable: Decorated function with workspace client validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Get function signature for parameter inspection
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                # Extract workspace client
                workspace_client = bound_args.arguments.get(workspace_client_param)

                if not workspace_client:
                    logger.error(f"Workspace client not provided for {func.__name__}")
                    return {"error": "Workspace client not available"}

                if require_initialized and not getattr(workspace_client, 'initialized', False):
                    logger.error(f"Workspace client not initialized for {func.__name__}")
                    return {"error": "Workspace client not initialized"}

                # Execute original function
                return await func(*args, **kwargs)

            except Exception as e:
                logger.error(f"Workspace client validation error in {func.__name__}: {e}")
                return {"error": "Operation failed due to workspace client error"}

        return wrapper
    return decorator


# Helper functions

def _extract_project_from_collection(collection_name: str) -> Optional[str]:
    """Extract project name from collection name using naming conventions."""
    if not collection_name:
        return None

    # Handle workspace collections: project-type format
    parts = collection_name.split('-')
    if len(parts) >= 2:
        # Last part should be a workspace type, everything before is project name
        from python.common.core.multitenant_collections import WorkspaceCollectionRegistry
        registry = WorkspaceCollectionRegistry()

        potential_type = parts[-1]
        if registry.is_multi_tenant_type(potential_type):
            return '-'.join(parts[:-1])

    # Handle shared collections
    if collection_name in ["scratchbook", "global", "shared"]:
        return "shared"

    # Handle legacy collections (return None to indicate no project context)
    return None


def _sanitize_log_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize function arguments for security logging."""
    sanitized = {}

    # List of parameter names that should be excluded or sanitized
    sensitive_params = {
        'password', 'token', 'key', 'secret', 'credential', 'api_key',
        'auth', 'authorization', 'session_id', 'user_context'
    }

    for key, value in args_dict.items():
        # Skip sensitive parameters
        if any(sensitive in key.lower() for sensitive in sensitive_params):
            sanitized[key] = "[REDACTED]"
        # Limit string length to prevent log pollution
        elif isinstance(value, str) and len(value) > 200:
            sanitized[key] = value[:200] + "... [TRUNCATED]"
        # Include safe values
        elif isinstance(value, (str, int, float, bool, list, dict)):
            if isinstance(value, dict) and len(str(value)) > 500:
                sanitized[key] = f"[DICT with {len(value)} keys]"
            elif isinstance(value, list) and len(str(value)) > 500:
                sanitized[key] = f"[LIST with {len(value)} items]"
            else:
                sanitized[key] = value
        else:
            sanitized[key] = f"[{type(value).__name__}]"

    return sanitized