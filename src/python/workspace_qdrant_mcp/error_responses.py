"""
Standardized MCP Error Responses

This module provides a standardized error response format for all MCP tool operations.
It ensures consistent structure, error codes, and helpful suggestions across all tools.

Features:
    - ErrorType enum categorizing errors (user_error, system_error, transient_error)
    - ErrorCode enum with predefined codes, messages, and suggestions
    - MCPErrorResponse dataclass for structured errors
    - Helper functions for common error scenarios
    - Error message sanitization to prevent sensitive data leakage

Example:
    ```python
    from workspace_qdrant_mcp.error_responses import (
        create_error_response,
        validation_error,
        not_found_error,
        daemon_unavailable_error,
        handle_tool_error,
    )

    # Create a validation error
    response = validation_error(
        field="collection_name",
        message="Collection name cannot be empty"
    )

    # Handle exceptions consistently
    try:
        await some_operation()
    except Exception as e:
        return handle_tool_error(e, operation="store")
    ```
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import re
import traceback


class ErrorType(Enum):
    """Type classification for errors."""

    USER_ERROR = "user_error"
    """Errors caused by invalid user input or parameters."""

    SYSTEM_ERROR = "system_error"
    """Errors from internal system failures."""

    TRANSIENT_ERROR = "transient_error"
    """Temporary errors that may resolve on retry."""


class ErrorCode(Enum):
    """
    Predefined error codes with associated metadata.

    Each error code includes:
    - code: Unique string identifier (e.g., "E1001")
    - message: Default human-readable message
    - suggestion: Helpful guidance for resolution
    - error_type: Classification (user/system/transient)
    """

    # Validation errors (E1xxx)
    VALIDATION_ERROR = (
        "E1001",
        "Validation failed",
        "Check the parameter values and formats",
        ErrorType.USER_ERROR,
    )
    MISSING_REQUIRED_FIELD = (
        "E1002",
        "Required field is missing",
        "Provide all required parameters",
        ErrorType.USER_ERROR,
    )
    INVALID_FORMAT = (
        "E1003",
        "Invalid format",
        "Ensure the value matches the expected format",
        ErrorType.USER_ERROR,
    )
    INVALID_PARAMETER = (
        "E1004",
        "Invalid parameter value",
        "Check the allowed values for this parameter",
        ErrorType.USER_ERROR,
    )

    # Resource errors (E2xxx)
    NOT_FOUND = (
        "E2001",
        "Resource not found",
        "Verify the resource exists and the identifier is correct",
        ErrorType.USER_ERROR,
    )
    COLLECTION_NOT_FOUND = (
        "E2002",
        "Collection not found",
        "Create the collection first or check the collection name",
        ErrorType.USER_ERROR,
    )
    DOCUMENT_NOT_FOUND = (
        "E2003",
        "Document not found",
        "Verify the document ID is correct",
        ErrorType.USER_ERROR,
    )
    PROJECT_NOT_FOUND = (
        "E2004",
        "Project not found",
        "Initialize the project first using manage(action='init_project')",
        ErrorType.USER_ERROR,
    )

    # Connection errors (E3xxx)
    DAEMON_UNAVAILABLE = (
        "E3001",
        "Daemon service is unavailable",
        "Start the daemon with 'wqm service start' or check service status",
        ErrorType.TRANSIENT_ERROR,
    )
    QDRANT_CONNECTION_ERROR = (
        "E3002",
        "Failed to connect to Qdrant",
        "Verify Qdrant is running and QDRANT_URL is correct",
        ErrorType.TRANSIENT_ERROR,
    )
    CONNECTION_TIMEOUT = (
        "E3003",
        "Connection timed out",
        "Check network connectivity and try again",
        ErrorType.TRANSIENT_ERROR,
    )
    GRPC_ERROR = (
        "E3004",
        "gRPC communication error",
        "Check daemon status and gRPC port configuration",
        ErrorType.TRANSIENT_ERROR,
    )

    # Operation errors (E4xxx)
    OPERATION_FAILED = (
        "E4001",
        "Operation failed",
        "Review the error details and retry",
        ErrorType.SYSTEM_ERROR,
    )
    STORAGE_ERROR = (
        "E4002",
        "Failed to store content",
        "Check collection permissions and storage capacity",
        ErrorType.SYSTEM_ERROR,
    )
    SEARCH_ERROR = (
        "E4003",
        "Search operation failed",
        "Simplify the query and retry",
        ErrorType.SYSTEM_ERROR,
    )
    DELETE_ERROR = (
        "E4004",
        "Failed to delete resource",
        "Verify the resource exists and you have permissions",
        ErrorType.SYSTEM_ERROR,
    )
    EMBEDDING_ERROR = (
        "E4005",
        "Failed to generate embeddings",
        "Check the embedding model configuration",
        ErrorType.SYSTEM_ERROR,
    )

    # Authorization errors (E5xxx)
    UNAUTHORIZED = (
        "E5001",
        "Unauthorized access",
        "Provide valid credentials",
        ErrorType.USER_ERROR,
    )
    PERMISSION_DENIED = (
        "E5002",
        "Permission denied",
        "Check access permissions for this resource",
        ErrorType.USER_ERROR,
    )

    # Rate limiting (E6xxx)
    RATE_LIMITED = (
        "E6001",
        "Rate limit exceeded",
        "Wait before retrying or reduce request frequency",
        ErrorType.TRANSIENT_ERROR,
    )
    QUEUE_FULL = (
        "E6002",
        "Processing queue is full",
        "Wait for queue to drain or increase queue capacity",
        ErrorType.TRANSIENT_ERROR,
    )

    # Internal errors (E9xxx)
    INTERNAL_ERROR = (
        "E9001",
        "Internal server error",
        "An unexpected error occurred. Please report this issue.",
        ErrorType.SYSTEM_ERROR,
    )
    CONFIGURATION_ERROR = (
        "E9002",
        "Configuration error",
        "Check the server configuration",
        ErrorType.SYSTEM_ERROR,
    )

    def __init__(
        self,
        code: str,
        message: str,
        suggestion: str,
        error_type: ErrorType,
    ):
        self._code = code
        self._message = message
        self._suggestion = suggestion
        self._error_type = error_type

    @property
    def code(self) -> str:
        """Get the error code string."""
        return self._code

    @property
    def default_message(self) -> str:
        """Get the default error message."""
        return self._message

    @property
    def suggestion(self) -> str:
        """Get the suggested resolution."""
        return self._suggestion

    @property
    def error_type(self) -> ErrorType:
        """Get the error type classification."""
        return self._error_type


@dataclass
class MCPErrorResponse:
    """
    Structured error response for MCP tool operations.

    This dataclass provides a consistent format for all error responses
    across the MCP server tools.
    """

    success: bool = False
    error_code: str = ""
    error_type: str = ""
    message: str = ""
    suggestion: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    operation: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "success": self.success,
            "error": {
                "code": self.error_code,
                "type": self.error_type,
                "message": self.message,
                "suggestion": self.suggestion,
            },
        }

        if self.details:
            result["error"]["details"] = self.details

        if self.operation:
            result["operation"] = self.operation

        result["timestamp"] = self.timestamp
        result["retryable"] = self.retryable

        return result


# Patterns for sensitive data that should be sanitized
SENSITIVE_PATTERNS = [
    (re.compile(r"api[_-]?key['\"]?\s*[:=]\s*['\"]?[\w-]+", re.I), "api_key=<REDACTED>"),
    (re.compile(r"password['\"]?\s*[:=]\s*['\"]?[^\s'\"]+", re.I), "password=<REDACTED>"),
    (re.compile(r"token['\"]?\s*[:=]\s*['\"]?[\w.-]+", re.I), "token=<REDACTED>"),
    (re.compile(r"secret['\"]?\s*[:=]\s*['\"]?[\w-]+", re.I), "secret=<REDACTED>"),
    (re.compile(r"bearer\s+[\w.-]+", re.I), "Bearer <REDACTED>"),
    (re.compile(r"/Users/[^/\s]+"), "/Users/<USER>"),
    (re.compile(r"\\\\Users\\\\[^\\\\\\s]+"), r"\\Users\\<USER>"),
]


def sanitize_error_message(message: str) -> str:
    """
    Sanitize error message to prevent sensitive data leakage.

    Args:
        message: The error message to sanitize

    Returns:
        Sanitized message with sensitive data redacted
    """
    sanitized = message
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def create_error_response(
    error_code: ErrorCode,
    message: str | None = None,
    suggestion: str | None = None,
    details: dict[str, Any] | None = None,
    operation: str = "",
    sanitize: bool = True,
) -> dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error_code: The ErrorCode enum value
        message: Override the default message (optional)
        suggestion: Override the default suggestion (optional)
        details: Additional error details (optional)
        operation: The operation that failed (optional)
        sanitize: Whether to sanitize the message for sensitive data

    Returns:
        Dictionary with standardized error structure
    """
    actual_message = message if message else error_code.default_message
    actual_suggestion = suggestion if suggestion else error_code.suggestion

    if sanitize:
        actual_message = sanitize_error_message(actual_message)

    response = MCPErrorResponse(
        success=False,
        error_code=error_code.code,
        error_type=error_code.error_type.value,
        message=actual_message,
        suggestion=actual_suggestion,
        details=details or {},
        operation=operation,
        retryable=error_code.error_type == ErrorType.TRANSIENT_ERROR,
    )

    return response.to_dict()


def handle_tool_error(
    exception: Exception,
    operation: str = "",
    include_traceback: bool = False,
) -> dict[str, Any]:
    """
    Handle an exception and convert to standardized error response.

    Args:
        exception: The exception that occurred
        operation: The operation that was being performed
        include_traceback: Whether to include stack trace in details

    Returns:
        Dictionary with standardized error structure
    """
    # Map known exception types to error codes
    from common.grpc.daemon_client import DaemonConnectionError

    error_code = ErrorCode.INTERNAL_ERROR
    message = str(exception)

    # Determine appropriate error code based on exception type
    exception_type = type(exception).__name__

    if isinstance(exception, DaemonConnectionError):
        error_code = ErrorCode.DAEMON_UNAVAILABLE
    elif isinstance(exception, ValueError):
        error_code = ErrorCode.VALIDATION_ERROR
    elif isinstance(exception, KeyError):
        error_code = ErrorCode.MISSING_REQUIRED_FIELD
        message = f"Missing key: {exception}"
    elif isinstance(exception, TimeoutError):
        error_code = ErrorCode.CONNECTION_TIMEOUT
    elif isinstance(exception, PermissionError):
        error_code = ErrorCode.PERMISSION_DENIED
    elif isinstance(exception, FileNotFoundError):
        error_code = ErrorCode.NOT_FOUND
        message = f"File not found: {exception}"
    elif isinstance(exception, ConnectionError):
        error_code = ErrorCode.QDRANT_CONNECTION_ERROR
    elif "grpc" in exception_type.lower():
        error_code = ErrorCode.GRPC_ERROR

    details: dict[str, Any] = {"exception_type": exception_type}
    if include_traceback:
        details["traceback"] = traceback.format_exc()

    return create_error_response(
        error_code=error_code,
        message=message,
        details=details,
        operation=operation,
    )


# Convenience functions for common error types


def validation_error(
    field: str,
    message: str,
    operation: str = "",
) -> dict[str, Any]:
    """
    Create a validation error response.

    Args:
        field: The field that failed validation
        message: Description of the validation failure
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    return create_error_response(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=message,
        details={"field": field},
        operation=operation,
    )


def missing_field_error(
    field: str,
    operation: str = "",
) -> dict[str, Any]:
    """
    Create a missing required field error response.

    Args:
        field: The name of the missing field
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    return create_error_response(
        error_code=ErrorCode.MISSING_REQUIRED_FIELD,
        message=f"Required field '{field}' is missing",
        details={"field": field},
        operation=operation,
    )


def not_found_error(
    resource_type: str,
    identifier: str,
    operation: str = "",
) -> dict[str, Any]:
    """
    Create a resource not found error response.

    Args:
        resource_type: Type of resource (e.g., "collection", "document")
        identifier: The identifier that was not found
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    # Select appropriate error code based on resource type
    error_code = ErrorCode.NOT_FOUND
    if resource_type.lower() == "collection":
        error_code = ErrorCode.COLLECTION_NOT_FOUND
    elif resource_type.lower() == "document":
        error_code = ErrorCode.DOCUMENT_NOT_FOUND
    elif resource_type.lower() == "project":
        error_code = ErrorCode.PROJECT_NOT_FOUND

    return create_error_response(
        error_code=error_code,
        message=f"{resource_type.capitalize()} '{identifier}' not found",
        details={"resource_type": resource_type, "identifier": identifier},
        operation=operation,
    )


def daemon_unavailable_error(
    operation: str = "",
    fallback_available: bool = False,
) -> dict[str, Any]:
    """
    Create a daemon unavailable error response.

    Args:
        operation: The operation being performed
        fallback_available: Whether a fallback mechanism is available

    Returns:
        Standardized error response
    """
    details: dict[str, Any] = {"daemon_available": False}
    suggestion = ErrorCode.DAEMON_UNAVAILABLE.suggestion

    if fallback_available:
        details["fallback_available"] = True
        suggestion += " Content may be queued for later processing."

    return create_error_response(
        error_code=ErrorCode.DAEMON_UNAVAILABLE,
        suggestion=suggestion,
        details=details,
        operation=operation,
    )


def collection_error(
    collection_name: str,
    action: str,
    reason: str,
    operation: str = "",
) -> dict[str, Any]:
    """
    Create a collection-related error response.

    Args:
        collection_name: Name of the collection
        action: The action that failed (e.g., "create", "delete", "access")
        reason: Reason for the failure
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    if "not found" in reason.lower() or "does not exist" in reason.lower():
        error_code = ErrorCode.COLLECTION_NOT_FOUND
    elif "permission" in reason.lower():
        error_code = ErrorCode.PERMISSION_DENIED
    else:
        error_code = ErrorCode.OPERATION_FAILED

    return create_error_response(
        error_code=error_code,
        message=f"Failed to {action} collection '{collection_name}': {reason}",
        details={"collection": collection_name, "action": action},
        operation=operation,
    )


def search_error(
    query: str,
    reason: str,
    operation: str = "search",
) -> dict[str, Any]:
    """
    Create a search-related error response.

    Args:
        query: The search query that failed
        reason: Reason for the failure
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    # Truncate long queries in error messages
    truncated_query = query[:100] + "..." if len(query) > 100 else query

    return create_error_response(
        error_code=ErrorCode.SEARCH_ERROR,
        message=f"Search failed: {reason}",
        details={"query_preview": truncated_query},
        operation=operation,
    )


def storage_error(
    reason: str,
    collection: str = "",
    operation: str = "store",
) -> dict[str, Any]:
    """
    Create a storage-related error response.

    Args:
        reason: Reason for the storage failure
        collection: The target collection (optional)
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    details: dict[str, Any] = {}
    if collection:
        details["collection"] = collection

    return create_error_response(
        error_code=ErrorCode.STORAGE_ERROR,
        message=f"Failed to store content: {reason}",
        details=details,
        operation=operation,
    )


def invalid_action_error(
    action: str,
    valid_actions: list[str],
    operation: str = "manage",
) -> dict[str, Any]:
    """
    Create an invalid action error response.

    Args:
        action: The invalid action that was requested
        valid_actions: List of valid actions
        operation: The operation being performed

    Returns:
        Standardized error response
    """
    return create_error_response(
        error_code=ErrorCode.INVALID_PARAMETER,
        message=f"Invalid action: '{action}'",
        suggestion=f"Valid actions are: {', '.join(valid_actions)}",
        details={"action": action, "valid_actions": valid_actions},
        operation=operation,
    )
