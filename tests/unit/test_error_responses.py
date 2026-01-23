"""
Unit tests for workspace_qdrant_mcp.error_responses module.

Tests the standardized MCP error response utilities for Task 449.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
from pathlib import Path

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

from workspace_qdrant_mcp.error_responses import (
    ErrorType,
    ErrorCode,
    MCPErrorResponse,
    create_error_response,
    create_simple_error_response,
    handle_tool_error,
    validation_error,
    not_found_error,
    daemon_unavailable_error,
    _sanitize_error_message,
)


class TestErrorType:
    """Test ErrorType enum."""

    def test_error_type_values(self):
        """Test ErrorType enum has expected values."""
        assert ErrorType.USER_ERROR.value == "user_error"
        assert ErrorType.SYSTEM_ERROR.value == "system_error"
        assert ErrorType.TRANSIENT_ERROR.value == "transient_error"


class TestErrorCode:
    """Test ErrorCode enum."""

    def test_error_code_structure(self):
        """Test ErrorCode values have correct structure."""
        # Each ErrorCode value should be a tuple of (code, type, message, suggestion)
        code, error_type, message, suggestion = ErrorCode.INVALID_COLLECTION_NAME.value

        assert code == "INVALID_COLLECTION_NAME"
        assert error_type == ErrorType.USER_ERROR
        assert isinstance(message, str)
        assert isinstance(suggestion, str)

    def test_user_error_codes(self):
        """Test user error codes have correct type."""
        user_error_codes = [
            ErrorCode.INVALID_COLLECTION_NAME,
            ErrorCode.INVALID_SCOPE,
            ErrorCode.INVALID_ACTION,
            ErrorCode.MISSING_REQUIRED_PARAMETER,
            ErrorCode.DOCUMENT_NOT_FOUND,
            ErrorCode.COLLECTION_NOT_FOUND,
            ErrorCode.WATCH_NOT_FOUND,
            ErrorCode.PROJECT_NOT_ACTIVATED,
            ErrorCode.INVALID_PROJECT_PATH,
            ErrorCode.PATH_NOT_FOUND,
            ErrorCode.PATH_NOT_DIRECTORY,
        ]

        for error_code in user_error_codes:
            _, error_type, _, _ = error_code.value
            assert error_type == ErrorType.USER_ERROR, f"{error_code.name} should be USER_ERROR"

    def test_transient_error_codes(self):
        """Test transient error codes have correct type."""
        transient_error_codes = [
            ErrorCode.DAEMON_UNAVAILABLE,
            ErrorCode.DAEMON_CONNECTION_ERROR,
            ErrorCode.DAEMON_TIMEOUT,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.QDRANT_UNAVAILABLE,
            ErrorCode.QUEUE_ERROR,
        ]

        for error_code in transient_error_codes:
            _, error_type, _, _ = error_code.value
            assert error_type == ErrorType.TRANSIENT_ERROR, f"{error_code.name} should be TRANSIENT_ERROR"

    def test_system_error_codes(self):
        """Test system error codes have correct type."""
        system_error_codes = [
            ErrorCode.INTERNAL_ERROR,
            ErrorCode.SEARCH_FAILED,
            ErrorCode.STORE_FAILED,
            ErrorCode.RETRIEVAL_FAILED,
        ]

        for error_code in system_error_codes:
            _, error_type, _, _ = error_code.value
            assert error_type == ErrorType.SYSTEM_ERROR, f"{error_code.name} should be SYSTEM_ERROR"


class TestMCPErrorResponse:
    """Test MCPErrorResponse dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        response = MCPErrorResponse(
            success=False,
            error_code="TEST_ERROR",
            error_type="user_error",
            message="Test error message",
            suggestion="Try again",
            details=None,
            retryable=False,
        )

        result = response.to_dict()

        assert result["success"] is False
        assert result["error"]["code"] == "TEST_ERROR"
        assert result["error"]["type"] == "user_error"
        assert result["error"]["message"] == "Test error message"
        assert result["error"]["suggestion"] == "Try again"
        assert "details" not in result["error"]
        assert "retryable" not in result["error"]

    def test_to_dict_with_details(self):
        """Test to_dict with details included."""
        response = MCPErrorResponse(
            success=False,
            error_code="TEST_ERROR",
            error_type="user_error",
            message="Test error",
            suggestion=None,
            details={"field": "value"},
            retryable=False,
        )

        result = response.to_dict()

        assert result["error"]["details"] == {"field": "value"}
        assert "suggestion" not in result["error"]

    def test_to_dict_retryable(self):
        """Test to_dict with retryable flag."""
        response = MCPErrorResponse(
            success=False,
            error_code="TRANSIENT_ERROR",
            error_type="transient_error",
            message="Temporary error",
            suggestion="Retry later",
            details=None,
            retryable=True,
        )

        result = response.to_dict()

        assert result["error"]["retryable"] is True


class TestCreateErrorResponse:
    """Test create_error_response function."""

    def test_basic_error_response(self):
        """Test basic error response creation."""
        result = create_error_response(ErrorCode.INVALID_COLLECTION_NAME)

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_COLLECTION_NAME"
        assert result["error"]["type"] == "user_error"
        assert "message" in result["error"]
        assert "suggestion" in result["error"]

    def test_error_response_with_override(self):
        """Test error response with message override."""
        result = create_error_response(
            ErrorCode.INVALID_SCOPE,
            message_override="Custom scope error",
            suggestion_override="Use valid scope",
        )

        assert result["error"]["message"] == "Custom scope error"
        assert result["error"]["suggestion"] == "Use valid scope"

    def test_error_response_transient_is_retryable(self):
        """Test transient errors are marked as retryable."""
        result = create_error_response(ErrorCode.DAEMON_UNAVAILABLE)

        assert result["error"]["retryable"] is True

    def test_error_response_user_error_not_retryable(self):
        """Test user errors are not marked as retryable."""
        result = create_error_response(ErrorCode.INVALID_COLLECTION_NAME)

        assert "retryable" not in result["error"]

    def test_error_response_with_context(self):
        """Test error response with context (not included by default)."""
        result = create_error_response(
            ErrorCode.INVALID_ACTION,
            context={"action": "unknown"},
            include_details=False,
        )

        assert "details" not in result["error"]

    def test_error_response_with_context_included(self):
        """Test error response with context included."""
        result = create_error_response(
            ErrorCode.INVALID_ACTION,
            context={"action": "unknown"},
            include_details=True,
        )

        assert result["error"]["details"] == {"action": "unknown"}


class TestCreateSimpleErrorResponse:
    """Test create_simple_error_response function."""

    def test_simple_error_response(self):
        """Test simple error response creation."""
        result = create_simple_error_response(
            message="Something went wrong",
            suggestion="Try again later",
        )

        assert result["success"] is False
        assert result["error"]["message"] == "Something went wrong"
        assert result["error"]["suggestion"] == "Try again later"
        assert result["error"]["type"] == "system_error"
        assert result["error"]["code"] == "ERROR"

    def test_simple_error_response_custom_type(self):
        """Test simple error response with custom type."""
        result = create_simple_error_response(
            message="Invalid input",
            error_type=ErrorType.USER_ERROR,
            error_code="CUSTOM_ERROR",
        )

        assert result["error"]["type"] == "user_error"
        assert result["error"]["code"] == "CUSTOM_ERROR"


class TestHandleToolError:
    """Test handle_tool_error function."""

    def test_handle_value_error(self):
        """Test handling ValueError."""
        exception = ValueError("Invalid value")
        result = handle_tool_error(exception, "test_operation")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_VALUE"
        assert result["error"]["type"] == "user_error"

    def test_handle_file_not_found_error(self):
        """Test handling FileNotFoundError."""
        exception = FileNotFoundError("File not found")
        result = handle_tool_error(exception, "test_operation")

        assert result["success"] is False
        assert result["error"]["code"] == "PATH_NOT_FOUND"
        assert result["error"]["type"] == "user_error"

    def test_handle_permission_error(self):
        """Test handling PermissionError."""
        exception = PermissionError("Permission denied")
        result = handle_tool_error(exception, "test_operation")

        assert result["success"] is False
        assert result["error"]["code"] == "PERMISSION_DENIED"
        assert result["error"]["type"] == "user_error"

    def test_handle_unknown_exception(self):
        """Test handling unknown exception."""
        exception = RuntimeError("Unknown error")
        result = handle_tool_error(exception, "test_operation")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["error"]["type"] == "system_error"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_validation_error(self):
        """Test validation_error function."""
        result = validation_error("Invalid format", field="name")

        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "name: Invalid format" in result["error"]["message"]

    def test_validation_error_without_field(self):
        """Test validation_error without field."""
        result = validation_error("Invalid format")

        assert result["error"]["message"] == "Invalid format"

    def test_not_found_error(self):
        """Test not_found_error function."""
        result = not_found_error("Document", identifier="doc-123")

        assert result["success"] is False
        assert result["error"]["code"] == "NOT_FOUND"
        assert "Document 'doc-123' not found" in result["error"]["message"]

    def test_not_found_error_without_identifier(self):
        """Test not_found_error without identifier."""
        result = not_found_error("Collection")

        assert result["error"]["message"] == "Collection not found"

    def test_daemon_unavailable_error(self):
        """Test daemon_unavailable_error function."""
        result = daemon_unavailable_error("store")

        assert result["success"] is False
        assert result["error"]["code"] == "DAEMON_UNAVAILABLE"
        assert result["error"]["retryable"] is True

    def test_daemon_unavailable_error_queued(self):
        """Test daemon_unavailable_error with queued=True."""
        result = daemon_unavailable_error("store", queued=True)

        assert "queued" in result["error"]["suggestion"].lower()


class TestSanitizeErrorMessage:
    """Test _sanitize_error_message function."""

    def test_sanitize_file_paths(self):
        """Test sanitizing file paths."""
        message = "Error reading /home/user/secret/file.txt"
        result = _sanitize_error_message(message)

        assert "/home/user/secret/file.txt" not in result
        assert "[path]" in result

    def test_sanitize_windows_paths(self):
        """Test sanitizing Windows paths."""
        message = r"Error reading C:\Users\secret\file.txt"
        result = _sanitize_error_message(message)

        assert "C:\\Users\\secret\\file.txt" not in result

    def test_sanitize_api_keys(self):
        """Test sanitizing potential API keys."""
        # Long hex string that looks like an API key
        message = "API key: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        result = _sanitize_error_message(message)

        assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" not in result
        assert "[redacted]" in result

    def test_sanitize_urls(self):
        """Test sanitizing URLs."""
        message = "Failed to connect to https://api.example.com/v1/data"
        result = _sanitize_error_message(message)

        # URL should be sanitized (either as [url] or [path] depending on order)
        assert "https://api.example.com/v1/data" not in result
        # The function sanitizes paths before URLs, so the path part gets replaced first
        assert "[path]" in result or "[url]" in result

    def test_sanitize_truncates_long_messages(self):
        """Test sanitizing truncates or redacts long messages."""
        message = "A" * 300
        result = _sanitize_error_message(message)

        # Very long repeated strings may be redacted as potential secrets
        # or truncated with "..."
        assert len(result) <= 200 or result == "[redacted]"

    def test_sanitize_preserves_short_messages(self):
        """Test sanitizing preserves short messages."""
        message = "Short error message"
        result = _sanitize_error_message(message)

        assert result == "Short error message"
