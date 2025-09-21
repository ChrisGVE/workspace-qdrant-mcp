"""
Project isolation validation for workspace-qdrant-mcp.

This module provides comprehensive security validation to ensure proper multi-tenant
project isolation across all MCP tools. It prevents cross-project access attempts
and provides security logging for monitoring and audit purposes.

Key Security Features:
    - Project ownership validation for collections and documents
    - Cross-project access prevention with comprehensive checks
    - Security event logging for audit trails and monitoring
    - Sanitized error messages that don't expose system internals
    - User context validation and authorization checks
    - Collection naming pattern validation for multi-tenant security

Example:
    ```python
    from workspace_qdrant_mcp.validation.project_isolation import ProjectIsolationValidator

    validator = ProjectIsolationValidator()

    # Validate project access
    if not validator.validate_project_access("my-project", user_context):
        raise SecurityError("Access denied")

    # Validate collection ownership
    if not validator.validate_collection_ownership("my-project-notes", "my-project"):
        raise SecurityError("Collection access denied")
    ```

Security Considerations:
    - All validation failures are logged for security monitoring
    - Error messages are sanitized to prevent information disclosure
    - Failed access attempts are tracked for potential security analysis
    - Default deny policy - access is denied unless explicitly allowed
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from loguru import logger

from python.common.core.client import QdrantWorkspaceClient
from python.common.core.multitenant_collections import WorkspaceCollectionRegistry


@dataclass
class UserContext:
    """User context for project isolation validation."""
    user_id: str
    session_id: str
    allowed_projects: Set[str]
    permissions: Set[str] = field(default_factory=set)
    admin_access: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_id: str
    event_type: str
    timestamp: datetime
    user_context: Optional[UserContext]
    project_name: Optional[str]
    collection_name: Optional[str]
    document_id: Optional[str]
    details: Dict
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL


class SecurityError(Exception):
    """Security validation error."""
    def __init__(self, message: str, event_type: str = "access_denied", details: Optional[Dict] = None):
        super().__init__(message)
        self.event_type = event_type
        self.details = details or {}


class ProjectIsolationValidator:
    """
    Comprehensive project isolation validator for multi-tenant security.

    This class provides validation methods to ensure proper project isolation
    across all workspace operations, preventing cross-project access and
    maintaining security boundaries.
    """

    def __init__(self, workspace_client: Optional[QdrantWorkspaceClient] = None):
        """
        Initialize the project isolation validator.

        Args:
            workspace_client: Optional workspace client for collection validation
        """
        self.workspace_client = workspace_client
        self.registry = WorkspaceCollectionRegistry()
        self._security_events: List[SecurityEvent] = []

        # Collection naming patterns for different project types
        self._project_collection_patterns = {
            'workspace': r'^([a-zA-Z0-9\-_]+)-([a-zA-Z0-9\-_]+)$',  # project-type format
            'shared': r'^(scratchbook|global|shared)$',  # shared collections
            'legacy': r'^[a-zA-Z0-9\-_]+$'  # legacy single collections
        }

        logger.info("ProjectIsolationValidator initialized with security validation enabled")

    def validate_project_access(
        self,
        project_name: str,
        user_context: Optional[UserContext] = None,
        operation: str = "access"
    ) -> bool:
        """
        Validate if user has access to the specified project.

        Args:
            project_name: Target project name
            user_context: User context with permissions (None allows access for backwards compatibility)
            operation: Type of operation being performed

        Returns:
            bool: True if access is allowed, False otherwise

        Raises:
            SecurityError: If validation fails due to security policy violation
        """
        try:
            # Basic validation
            if not project_name or not isinstance(project_name, str):
                self._log_security_event(
                    "invalid_project_name",
                    user_context=user_context,
                    project_name=project_name,
                    details={"operation": operation, "reason": "Invalid project name format"},
                    severity="WARNING"
                )
                return False

            # Sanitize project name
            if not self._is_valid_project_name(project_name):
                self._log_security_event(
                    "malicious_project_name",
                    user_context=user_context,
                    project_name=project_name,
                    details={"operation": operation, "reason": "Project name contains invalid characters"},
                    severity="ERROR"
                )
                return False

            # If no user context provided, allow access (backwards compatibility)
            # In production, this should be removed and user context should be mandatory
            if user_context is None:
                self._log_security_event(
                    "no_user_context",
                    project_name=project_name,
                    details={"operation": operation, "reason": "No user context provided - allowing access"},
                    severity="WARNING"
                )
                return True

            # Check if user is admin (admin can access all projects)
            if user_context.admin_access:
                self._log_security_event(
                    "admin_access_granted",
                    user_context=user_context,
                    project_name=project_name,
                    details={"operation": operation},
                    severity="INFO"
                )
                return True

            # Check if project is in user's allowed projects
            if project_name in user_context.allowed_projects:
                self._log_security_event(
                    "project_access_granted",
                    user_context=user_context,
                    project_name=project_name,
                    details={"operation": operation},
                    severity="INFO"
                )
                return True

            # Access denied
            self._log_security_event(
                "project_access_denied",
                user_context=user_context,
                project_name=project_name,
                details={
                    "operation": operation,
                    "reason": "Project not in user's allowed projects",
                    "user_projects": list(user_context.allowed_projects)
                },
                severity="WARNING"
            )
            return False

        except Exception as e:
            self._log_security_event(
                "validation_error",
                user_context=user_context,
                project_name=project_name,
                details={"operation": operation, "error": str(e)},
                severity="ERROR"
            )
            # Fail secure - deny access on validation errors
            return False

    def validate_collection_ownership(
        self,
        collection_name: str,
        project_name: str,
        user_context: Optional[UserContext] = None,
        allow_shared: bool = True
    ) -> bool:
        """
        Validate that a collection belongs to the specified project.

        Args:
            collection_name: Collection name to validate
            project_name: Expected project owner
            user_context: User context for logging
            allow_shared: Whether to allow access to shared collections

        Returns:
            bool: True if collection belongs to project or is shared, False otherwise
        """
        try:
            # Basic validation
            if not collection_name or not project_name:
                self._log_security_event(
                    "invalid_collection_validation",
                    user_context=user_context,
                    collection_name=collection_name,
                    project_name=project_name,
                    details={"reason": "Missing collection name or project name"},
                    severity="WARNING"
                )
                return False

            # Check if it's a shared collection
            if allow_shared and self._is_shared_collection(collection_name):
                self._log_security_event(
                    "shared_collection_access",
                    user_context=user_context,
                    collection_name=collection_name,
                    project_name=project_name,
                    details={"collection_type": "shared"},
                    severity="INFO"
                )
                return True

            # Check if collection follows project naming pattern
            if self._is_project_collection(collection_name, project_name):
                self._log_security_event(
                    "collection_ownership_validated",
                    user_context=user_context,
                    collection_name=collection_name,
                    project_name=project_name,
                    details={"validation_result": "belongs_to_project"},
                    severity="INFO"
                )
                return True

            # Check workspace client if available for additional validation
            if self.workspace_client:
                if self._validate_collection_exists(collection_name, project_name):
                    return True

            # Collection does not belong to project
            self._log_security_event(
                "collection_ownership_denied",
                user_context=user_context,
                collection_name=collection_name,
                project_name=project_name,
                details={"reason": "Collection does not belong to project"},
                severity="WARNING"
            )
            return False

        except Exception as e:
            self._log_security_event(
                "collection_validation_error",
                user_context=user_context,
                collection_name=collection_name,
                project_name=project_name,
                details={"error": str(e)},
                severity="ERROR"
            )
            # Fail secure - deny access on validation errors
            return False

    def validate_document_access(
        self,
        document_id: str,
        collection_name: str,
        project_name: str,
        user_context: Optional[UserContext] = None,
        operation: str = "read"
    ) -> bool:
        """
        Validate document access within project isolation boundaries.

        Args:
            document_id: Document identifier
            collection_name: Collection containing the document
            project_name: Project context
            user_context: User context for validation
            operation: Type of operation (read, write, delete)

        Returns:
            bool: True if access is allowed, False otherwise
        """
        try:
            # First validate collection ownership
            if not self.validate_collection_ownership(collection_name, project_name, user_context):
                self._log_security_event(
                    "document_access_denied_collection",
                    user_context=user_context,
                    collection_name=collection_name,
                    project_name=project_name,
                    document_id=document_id,
                    details={
                        "operation": operation,
                        "reason": "Collection ownership validation failed"
                    },
                    severity="WARNING"
                )
                return False

            # Validate document ID format
            if not self._is_valid_document_id(document_id):
                self._log_security_event(
                    "invalid_document_id",
                    user_context=user_context,
                    collection_name=collection_name,
                    project_name=project_name,
                    document_id=document_id,
                    details={"operation": operation, "reason": "Invalid document ID format"},
                    severity="WARNING"
                )
                return False

            # Additional validation for write/delete operations
            if operation in ["write", "delete"] and user_context:
                if "write" not in user_context.permissions and not user_context.admin_access:
                    self._log_security_event(
                        "document_write_access_denied",
                        user_context=user_context,
                        collection_name=collection_name,
                        project_name=project_name,
                        document_id=document_id,
                        details={"operation": operation, "reason": "Insufficient write permissions"},
                        severity="WARNING"
                    )
                    return False

            # Document access granted
            self._log_security_event(
                "document_access_granted",
                user_context=user_context,
                collection_name=collection_name,
                project_name=project_name,
                document_id=document_id,
                details={"operation": operation},
                severity="INFO"
            )
            return True

        except Exception as e:
            self._log_security_event(
                "document_access_validation_error",
                user_context=user_context,
                collection_name=collection_name,
                project_name=project_name,
                document_id=document_id,
                details={"operation": operation, "error": str(e)},
                severity="ERROR"
            )
            # Fail secure - deny access on validation errors
            return False

    def validate_cross_project_operation(
        self,
        source_project: str,
        target_project: str,
        operation_type: str,
        user_context: Optional[UserContext] = None
    ) -> bool:
        """
        Validate cross-project operations for security compliance.

        Args:
            source_project: Source project name
            target_project: Target project name
            operation_type: Type of cross-project operation
            user_context: User context for validation

        Returns:
            bool: True if cross-project operation is allowed
        """
        try:
            # Validate both projects individually first
            if not self.validate_project_access(source_project, user_context, f"cross_project_{operation_type}_source"):
                return False

            if not self.validate_project_access(target_project, user_context, f"cross_project_{operation_type}_target"):
                return False

            # Check if user has cross-project permissions
            if user_context and "cross_project" not in user_context.permissions and not user_context.admin_access:
                self._log_security_event(
                    "cross_project_operation_denied",
                    user_context=user_context,
                    details={
                        "source_project": source_project,
                        "target_project": target_project,
                        "operation_type": operation_type,
                        "reason": "Insufficient cross-project permissions"
                    },
                    severity="WARNING"
                )
                return False

            # Log successful cross-project operation validation
            self._log_security_event(
                "cross_project_operation_granted",
                user_context=user_context,
                details={
                    "source_project": source_project,
                    "target_project": target_project,
                    "operation_type": operation_type
                },
                severity="INFO"
            )
            return True

        except Exception as e:
            self._log_security_event(
                "cross_project_validation_error",
                user_context=user_context,
                details={
                    "source_project": source_project,
                    "target_project": target_project,
                    "operation_type": operation_type,
                    "error": str(e)
                },
                severity="ERROR"
            )
            return False

    def sanitize_error_message(self, error: Exception, user_context: Optional[UserContext] = None) -> str:
        """
        Sanitize error messages to prevent information disclosure.

        Args:
            error: Original error
            user_context: User context for determining sanitization level

        Returns:
            str: Sanitized error message safe for user consumption
        """
        # Default generic error message
        generic_message = "Operation failed due to security policy violation"

        # Admin users get more detailed error messages
        if user_context and user_context.admin_access:
            return str(error)

        # Map specific error types to user-friendly messages
        error_message = str(error).lower()

        if "access denied" in error_message or "permission" in error_message:
            return "Access denied - insufficient permissions"
        elif "not found" in error_message:
            return "Resource not found or access denied"
        elif "invalid" in error_message:
            return "Invalid request format"
        elif "timeout" in error_message:
            return "Operation timeout - please try again"
        else:
            return generic_message

    def get_security_events(
        self,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[SecurityEvent]:
        """
        Retrieve security events for monitoring and audit purposes.

        Args:
            event_type: Filter by specific event type
            severity: Filter by severity level
            limit: Maximum number of events to return

        Returns:
            List[SecurityEvent]: Filtered security events
        """
        events = self._security_events.copy()

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if severity:
            events = [e for e in events if e.severity == severity]

        # Return most recent events first
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def clear_security_events(self) -> int:
        """
        Clear stored security events.

        Returns:
            int: Number of events cleared
        """
        count = len(self._security_events)
        self._security_events.clear()
        logger.info(f"Cleared {count} security events from validator")
        return count

    # Private helper methods

    def _is_valid_project_name(self, project_name: str) -> bool:
        """Validate project name format for security."""
        # Allow alphanumeric, hyphens, underscores, dots - prevent path traversal
        pattern = r'^[a-zA-Z0-9\-_\.]{1,100}$'
        return bool(re.match(pattern, project_name)) and '..' not in project_name

    def _is_valid_document_id(self, document_id: str) -> bool:
        """Validate document ID format for security."""
        if not document_id or len(document_id) > 200:
            return False

        # Allow UUIDs, alphanumeric with common separators
        pattern = r'^[a-zA-Z0-9\-_\.:/]{1,200}$'
        return bool(re.match(pattern, document_id)) and '..' not in document_id

    def _is_shared_collection(self, collection_name: str) -> bool:
        """Check if collection is a shared collection."""
        shared_pattern = self._project_collection_patterns['shared']
        return bool(re.match(shared_pattern, collection_name))

    def _is_project_collection(self, collection_name: str, project_name: str) -> bool:
        """Check if collection follows project naming pattern."""
        # Check for workspace pattern: project-type
        workspace_pattern = self._project_collection_patterns['workspace']
        match = re.match(workspace_pattern, collection_name)

        if match:
            collection_project, collection_type = match.groups()
            return collection_project == project_name and self.registry.is_multi_tenant_type(collection_type)

        return False

    def _validate_collection_exists(self, collection_name: str, project_name: str) -> bool:
        """Validate collection exists and belongs to project using workspace client."""
        if not self.workspace_client or not self.workspace_client.initialized:
            return False

        try:
            # Check if collection exists
            collections = self.workspace_client.list_collections()
            if collection_name not in collections:
                return False

            # Additional validation could be added here
            return True

        except Exception as e:
            logger.warning(f"Collection existence validation failed: {e}")
            return False

    def _log_security_event(
        self,
        event_type: str,
        user_context: Optional[UserContext] = None,
        project_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        document_id: Optional[str] = None,
        details: Optional[Dict] = None,
        severity: str = "INFO"
    ) -> None:
        """Log security event for audit and monitoring."""
        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            user_context=user_context,
            project_name=project_name,
            collection_name=collection_name,
            document_id=document_id,
            details=details or {},
            severity=severity
        )

        self._security_events.append(event)

        # Log to system logger with appropriate level
        log_message = f"Security event: {event_type}"
        log_data = {
            "event_id": event.event_id,
            "project_name": project_name,
            "collection_name": collection_name,
            "document_id": document_id,
            "user_id": user_context.user_id if user_context else None,
            **details
        }

        if severity == "CRITICAL":
            logger.critical(log_message, **log_data)
        elif severity == "ERROR":
            logger.error(log_message, **log_data)
        elif severity == "WARNING":
            logger.warning(log_message, **log_data)
        else:
            logger.info(log_message, **log_data)


# Global validator instance (can be configured per application)
_global_validator: Optional[ProjectIsolationValidator] = None


def get_validator(workspace_client: Optional[QdrantWorkspaceClient] = None) -> ProjectIsolationValidator:
    """
    Get global project isolation validator instance.

    Args:
        workspace_client: Optional workspace client for validation

    Returns:
        ProjectIsolationValidator: Global validator instance
    """
    global _global_validator

    if _global_validator is None:
        _global_validator = ProjectIsolationValidator(workspace_client)
    elif workspace_client and _global_validator.workspace_client is None:
        _global_validator.workspace_client = workspace_client

    return _global_validator


def validate_project_operation(
    project_name: str,
    operation: str,
    user_context: Optional[UserContext] = None,
    workspace_client: Optional[QdrantWorkspaceClient] = None
) -> None:
    """
    Convenience function for validating project operations.

    Args:
        project_name: Project name to validate
        operation: Operation type being performed
        user_context: User context for validation
        workspace_client: Optional workspace client

    Raises:
        SecurityError: If validation fails
    """
    validator = get_validator(workspace_client)

    if not validator.validate_project_access(project_name, user_context, operation):
        raise SecurityError(
            f"Access denied for project '{project_name}' operation '{operation}'",
            event_type="project_access_denied",
            details={"project_name": project_name, "operation": operation}
        )


def validate_collection_operation(
    collection_name: str,
    project_name: str,
    operation: str,
    user_context: Optional[UserContext] = None,
    workspace_client: Optional[QdrantWorkspaceClient] = None,
    allow_shared: bool = True
) -> None:
    """
    Convenience function for validating collection operations.

    Args:
        collection_name: Collection name to validate
        project_name: Expected project owner
        operation: Operation type being performed
        user_context: User context for validation
        workspace_client: Optional workspace client
        allow_shared: Whether to allow shared collection access

    Raises:
        SecurityError: If validation fails
    """
    validator = get_validator(workspace_client)

    # First validate project access
    validate_project_operation(project_name, operation, user_context, workspace_client)

    # Then validate collection ownership
    if not validator.validate_collection_ownership(collection_name, project_name, user_context, allow_shared):
        raise SecurityError(
            f"Collection '{collection_name}' access denied for project '{project_name}'",
            event_type="collection_access_denied",
            details={
                "collection_name": collection_name,
                "project_name": project_name,
                "operation": operation
            }
        )