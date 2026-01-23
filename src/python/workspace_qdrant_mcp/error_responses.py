"""
MCP Error Response Utilities (Task 449)

Provides consistent, user-friendly error responses for the MCP server with:
- Standard error response format
- Error categorization (user_error, system_error, transient_error)
- User-friendly error messages
- Recovery suggestions
- Proper error logging without sensitive data leakage

Usage:
    from workspace_qdrant_mcp.error_responses import (
        create_error_response,
        ErrorCode,
        handle_tool_error
    )

    # In a tool function
    try:
        result = await some_operation()
    except DaemonConnectionError as e:
        return create_error_response(
            ErrorCode.DAEMON_UNAVAILABLE,
            context={"operation": "store"}
        )
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class ErrorType(Enum):
    """Error type classification for MCP responses."""

    USER_ERROR = "user_error"  # Client can fix by changing input
    SYSTEM_ERROR = "system_error"  # Server-side issue, client cannot fix
    TRANSIENT_ERROR = "transient_error"  # Temporary issue, retry may help


class ErrorCode(Enum):
    """
    Standard error codes with associated messages and recovery suggestions.

    Format: (code, type, message, suggestion)
    """

    # Validation errors (USER_ERROR)
    INVALID_COLLECTION_NAME = (
        "INVALID_COLLECTION_NAME",
        ErrorType.USER_ERROR,
        "The collection name is invalid",
        "Use lowercase letters, numbers, and hyphens. Start with a letter."
    )
    INVALID_SCOPE = (
        "INVALID_SCOPE",
        ErrorType.USER_ERROR,
        "The search scope is invalid",
        "Valid scopes are: 'project', 'all', or 'global'."
    )
    INVALID_ACTION = (
        "INVALID_ACTION",
        ErrorType.USER_ERROR,
        "The management action is not recognized",
        "Use manage(action='help') to see available actions."
    )
    MISSING_REQUIRED_PARAMETER = (
        "MISSING_PARAMETER",
        ErrorType.USER_ERROR,
        "A required parameter is missing",
        "Check the tool documentation for required parameters."
    )
    INVALID_PARAMETER_VALUE = (
        "INVALID_VALUE",
        ErrorType.USER_ERROR,
        "A parameter has an invalid value",
        "Check the parameter value format and try again."
    )
    DOCUMENT_NOT_FOUND = (
        "NOT_FOUND",
        ErrorType.USER_ERROR,
        "The requested document was not found",
        "Verify the document ID and collection name."
    )
    COLLECTION_NOT_FOUND = (
        "COLLECTION_NOT_FOUND",
        ErrorType.USER_ERROR,
        "The specified collection does not exist",
        "Use manage(action='list_collections') to see available collections."
    )
    WATCH_NOT_FOUND = (
        "WATCH_NOT_FOUND",
        ErrorType.USER_ERROR,
        "The specified watch folder was not found",
        "Use manage(action='list_watches') to see configured watches."
    )

    # Project activation errors (USER_ERROR)
    PROJECT_NOT_ACTIVATED = (
        "PROJECT_NOT_ACTIVE",
        ErrorType.USER_ERROR,
        "No project is currently activated",
        "Use manage(action='activate_project', config={'path': '/path/to/project'}) first."
    )
    INVALID_PROJECT_PATH = (
        "INVALID_PROJECT",
        ErrorType.USER_ERROR,
        "The specified path is not a valid project",
        "Ensure the path exists and contains a git repository or .taskmaster directory."
    )
    PATH_NOT_FOUND = (
        "PATH_NOT_FOUND",
        ErrorType.USER_ERROR,
        "The specified path does not exist",
        "Verify the path is correct and accessible."
    )
    PATH_NOT_DIRECTORY = (
        "NOT_DIRECTORY",
        ErrorType.USER_ERROR,
        "The specified path is not a directory",
        "Provide a directory path, not a file path."
    )

    # Daemon errors (TRANSIENT_ERROR)
    DAEMON_UNAVAILABLE = (
        "DAEMON_UNAVAILABLE",
        ErrorType.TRANSIENT_ERROR,
        "The ingestion daemon is not available",
        "The daemon may be starting up. Content has been queued for processing."
    )
    DAEMON_CONNECTION_ERROR = (
        "DAEMON_CONNECTION",
        ErrorType.TRANSIENT_ERROR,
        "Could not connect to the ingestion daemon",
        "Check if the daemon is running with 'wqm service status'."
    )
    DAEMON_TIMEOUT = (
        "DAEMON_TIMEOUT",
        ErrorType.TRANSIENT_ERROR,
        "The daemon operation timed out",
        "The operation may complete later. Check 'wqm admin status' for details."
    )

    # Database errors (TRANSIENT_ERROR)
    DATABASE_ERROR = (
        "DATABASE_ERROR",
        ErrorType.TRANSIENT_ERROR,
        "A database operation failed",
        "This is usually temporary. Retry the operation in a moment."
    )
    QDRANT_UNAVAILABLE = (
        "QDRANT_UNAVAILABLE",
        ErrorType.TRANSIENT_ERROR,
        "The vector database is not available",
        "Check Qdrant server status at the configured URL."
    )

    # Queue errors (TRANSIENT_ERROR)
    QUEUE_ERROR = (
        "QUEUE_ERROR",
        ErrorType.TRANSIENT_ERROR,
        "Failed to queue the operation",
        "The SQLite state database may be temporarily unavailable."
    )

    # System errors (SYSTEM_ERROR)
    INTERNAL_ERROR = (
        "INTERNAL_ERROR",
        ErrorType.SYSTEM_ERROR,
        "An internal error occurred",
        "Please report this issue if it persists."
    )
    SEARCH_FAILED = (
        "SEARCH_FAILED",
        ErrorType.SYSTEM_ERROR,
        "The search operation failed",
        "Try narrowing your search query or reducing the limit."
    )
    STORE_FAILED = (
        "STORE_FAILED",
        ErrorType.SYSTEM_ERROR,
        "Failed to store the document",
        "Check the content format and metadata, then retry."
    )
    RETRIEVAL_FAILED = (
        "RETRIEVAL_FAILED",
        ErrorType.SYSTEM_ERROR,
        "Failed to retrieve the document",
        "The document may have been deleted or the collection may be unavailable."
    )


@dataclass
class MCPErrorResponse:
    """Structured MCP error response."""

    success: bool
    error_code: str
    error_type: str
    message: str
    suggestion: str | None
    details: dict[str, Any] | None
    retryable: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP response."""
        result = {
            "success": self.success,
            "error": {
                "code": self.error_code,
                "type": self.error_type,
                "message": self.message,
            }
        }

        if self.suggestion:
            result["error"]["suggestion"] = self.suggestion

        if self.details:
            result["error"]["details"] = self.details

        if self.retryable:
            result["error"]["retryable"] = True

        return result


def create_error_response(
    error_code: ErrorCode,
    message_override: str | None = None,
    suggestion_override: str | None = None,
    context: dict[str, Any] | None = None,
    include_details: bool = False,
) -> dict[str, Any]:
    """
    Create a standardized MCP error response.

    Args:
        error_code: The error code enum value
        message_override: Override the default error message
        suggestion_override: Override the default recovery suggestion
        context: Additional context (logged but may be filtered for response)
        include_details: Whether to include context details in response

    Returns:
        Dictionary with standardized error response format
    """
    code, error_type, default_message, default_suggestion = error_code.value

    message = message_override or default_message
    suggestion = suggestion_override or default_suggestion

    # Log the error with full context
    logger.warning(
        f"MCP error: {code}",
        extra={
            "error_code": code,
            "error_type": error_type.value,
            "message": message,
            "context": context or {},
        }
    )

    # Build response
    response = MCPErrorResponse(
        success=False,
        error_code=code,
        error_type=error_type.value,
        message=message,
        suggestion=suggestion,
        details=context if include_details else None,
        retryable=error_type == ErrorType.TRANSIENT_ERROR,
    )

    return response.to_dict()


def create_simple_error_response(
    message: str,
    suggestion: str | None = None,
    error_type: ErrorType = ErrorType.SYSTEM_ERROR,
    error_code: str = "ERROR",
) -> dict[str, Any]:
    """
    Create a simple error response without using predefined error codes.

    Use this for ad-hoc errors that don't fit predefined codes.

    Args:
        message: The error message
        suggestion: Recovery suggestion
        error_type: Type of error (user, system, transient)
        error_code: Custom error code string

    Returns:
        Dictionary with error response format
    """
    response = MCPErrorResponse(
        success=False,
        error_code=error_code,
        error_type=error_type.value,
        message=message,
        suggestion=suggestion,
        details=None,
        retryable=error_type == ErrorType.TRANSIENT_ERROR,
    )

    return response.to_dict()


def handle_tool_error(
    exception: Exception,
    operation: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Handle exceptions in MCP tools and convert to standardized error responses.

    This function maps common exceptions to appropriate error codes and
    provides user-friendly error messages.

    Args:
        exception: The caught exception
        operation: Name of the operation that failed (for logging)
        context: Additional context for logging

    Returns:
        Dictionary with standardized error response
    """
    # Import here to avoid circular imports
    from common.grpc.daemon_client import DaemonConnectionError

    exception_type = type(exception).__name__
    error_message = str(exception)

    # Log the full error details (will be filtered for user)
    logger.error(
        f"Tool error in {operation}: {exception_type}",
        extra={
            "operation": operation,
            "exception_type": exception_type,
            "error_message": error_message,
            "context": context or {},
        },
        exc_info=True,
    )

    # Map exception types to error codes
    if isinstance(exception, DaemonConnectionError):
        return create_error_response(
            ErrorCode.DAEMON_CONNECTION_ERROR,
            context={"operation": operation},
        )

    if isinstance(exception, asyncio.TimeoutError):
        return create_error_response(
            ErrorCode.DAEMON_TIMEOUT,
            context={"operation": operation},
        )

    if isinstance(exception, ValueError):
        return create_error_response(
            ErrorCode.INVALID_PARAMETER_VALUE,
            message_override=_sanitize_error_message(error_message),
            context={"operation": operation},
        )

    if isinstance(exception, FileNotFoundError):
        return create_error_response(
            ErrorCode.PATH_NOT_FOUND,
            message_override=_sanitize_error_message(error_message),
            context={"operation": operation},
        )

    if isinstance(exception, PermissionError):
        return create_simple_error_response(
            message="Permission denied for the requested operation",
            suggestion="Check file/directory permissions and try again.",
            error_type=ErrorType.USER_ERROR,
            error_code="PERMISSION_DENIED",
        )

    if isinstance(exception, (ConnectionError, OSError)) and "connect" in error_message.lower():
        return create_error_response(
            ErrorCode.QDRANT_UNAVAILABLE,
            context={"operation": operation},
        )

    # Default: internal error
    return create_error_response(
        ErrorCode.INTERNAL_ERROR,
        message_override=f"{operation} failed unexpectedly",
        context={"operation": operation, "exception_type": exception_type},
    )


def _sanitize_error_message(message: str) -> str:
    """
    Sanitize error message to remove potentially sensitive information.

    Args:
        message: Raw error message

    Returns:
        Sanitized error message safe for user display
    """
    # Remove file paths (could expose system structure)
    import re

    # Remove absolute paths
    message = re.sub(r'/[\w/\-\.]+', '[path]', message)
    message = re.sub(r'[A-Z]:\\[\w\\]+', '[path]', message)

    # Remove potential API keys or tokens (long hex/base64 strings)
    message = re.sub(r'[A-Za-z0-9+/]{32,}', '[redacted]', message)

    # Remove URLs with potential credentials
    message = re.sub(r'https?://[^\s]+', '[url]', message)

    # Truncate if too long
    if len(message) > 200:
        message = message[:197] + "..."

    return message


# Import asyncio for TimeoutError check in handle_tool_error
import asyncio


# Convenience functions for common error patterns
def validation_error(
    message: str,
    field: str | None = None,
    suggestion: str | None = None,
) -> dict[str, Any]:
    """Create a validation error response."""
    full_message = f"{field}: {message}" if field else message
    return create_simple_error_response(
        message=full_message,
        suggestion=suggestion or "Check the input parameters and try again.",
        error_type=ErrorType.USER_ERROR,
        error_code="VALIDATION_ERROR",
    )


def not_found_error(
    resource: str,
    identifier: str | None = None,
    suggestion: str | None = None,
) -> dict[str, Any]:
    """Create a not found error response."""
    message = f"{resource} not found"
    if identifier:
        message = f"{resource} '{identifier}' not found"
    return create_simple_error_response(
        message=message,
        suggestion=suggestion or f"Verify the {resource.lower()} exists and try again.",
        error_type=ErrorType.USER_ERROR,
        error_code="NOT_FOUND",
    )


def daemon_unavailable_error(
    operation: str,
    queued: bool = False,
) -> dict[str, Any]:
    """Create a daemon unavailable error response."""
    if queued:
        return create_error_response(
            ErrorCode.DAEMON_UNAVAILABLE,
            suggestion_override="Content has been queued and will be processed when the daemon is available.",
            context={"operation": operation},
        )
    return create_error_response(
        ErrorCode.DAEMON_UNAVAILABLE,
        context={"operation": operation},
    )
