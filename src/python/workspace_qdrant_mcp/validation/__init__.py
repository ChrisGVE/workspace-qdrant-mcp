"""
Project isolation validation package for workspace-qdrant-mcp.

This package provides comprehensive security validation for multi-tenant
workspace operations, ensuring proper project isolation and access control
across all MCP tools.

Key Components:
    - ProjectIsolationValidator: Core validation class
    - UserContext: User context for authorization
    - SecurityEvent: Security event logging
    - SecurityError: Security validation exceptions

Example:
    ```python
    from workspace_qdrant_mcp.validation import (
        validate_project_operation,
        validate_collection_operation,
        UserContext
    )

    # Create user context
    user_context = UserContext(
        user_id="user123",
        session_id="session456",
        allowed_projects={"my-project", "shared-project"}
    )

    # Validate operations
    validate_project_operation("my-project", "read", user_context)
    validate_collection_operation("my-project-notes", "my-project", "write", user_context)
    ```
"""

from .project_isolation import (
    ProjectIsolationValidator,
    UserContext,
    SecurityEvent,
    SecurityError,
    get_validator,
    validate_project_operation,
    validate_collection_operation,
)

from .decorators import (
    require_project_access,
    require_collection_access,
    require_document_access,
    log_security_events,
    validate_workspace_client,
)

__all__ = [
    "ProjectIsolationValidator",
    "UserContext",
    "SecurityEvent",
    "SecurityError",
    "get_validator",
    "validate_project_operation",
    "validate_collection_operation",
    "require_project_access",
    "require_collection_access",
    "require_document_access",
    "log_security_events",
    "validate_workspace_client",
]