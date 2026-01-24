"""Unit tests for standardized MCP error responses (Task 468).

Tests the error_responses module covering:
- ErrorType and ErrorCode enums
- MCPErrorResponse dataclass
- create_error_response function
- handle_tool_error exception handling
- Convenience functions for common errors
- Error message sanitization
"""

import pytest
from unittest.mock import MagicMock

from workspace_qdrant_mcp.error_responses import (
    ErrorType,
    ErrorCode,
    MCPErrorResponse,
    sanitize_error_message,
    create_error_response,
    handle_tool_error,
    validation_error,
    missing_field_error,
    not_found_error,
    daemon_unavailable_error,
    collection_error,
    search_error,
    storage_error,
    invalid_action_error,
)


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_user_error_value(self):
        """Test USER_ERROR has correct value."""
        assert ErrorType.USER_ERROR.value == "user_error"

    def test_system_error_value(self):
        """Test SYSTEM_ERROR has correct value."""
        assert ErrorType.SYSTEM_ERROR.value == "system_error"

    def test_transient_error_value(self):
        """Test TRANSIENT_ERROR has correct value."""
        assert ErrorType.TRANSIENT_ERROR.value == "transient_error"


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_validation_error_properties(self):
        """Test VALIDATION_ERROR has correct properties."""
        code = ErrorCode.VALIDATION_ERROR
        assert code.code == "E1001"
        assert code.error_type == ErrorType.USER_ERROR
        assert code.default_message == "Validation failed"
        assert len(code.suggestion) > 0

    def test_daemon_unavailable_properties(self):
        """Test DAEMON_UNAVAILABLE is transient error."""
        code = ErrorCode.DAEMON_UNAVAILABLE
        assert code.code == "E3001"
        assert code.error_type == ErrorType.TRANSIENT_ERROR

    def test_internal_error_properties(self):
        """Test INTERNAL_ERROR is system error."""
        code = ErrorCode.INTERNAL_ERROR
        assert code.code == "E9001"
        assert code.error_type == ErrorType.SYSTEM_ERROR

    def test_not_found_properties(self):
        """Test NOT_FOUND has correct code."""
        code = ErrorCode.NOT_FOUND
        assert code.code == "E2001"
        assert code.error_type == ErrorType.USER_ERROR

    def test_all_error_codes_have_unique_codes(self):
        """Test all error codes have unique code strings."""
        codes = [code.code for code in ErrorCode]
        assert len(codes) == len(set(codes))

    def test_all_error_codes_have_messages(self):
        """Test all error codes have non-empty messages."""
        for code in ErrorCode:
            assert len(code.default_message) > 0

    def test_all_error_codes_have_suggestions(self):
        """Test all error codes have non-empty suggestions."""
        for code in ErrorCode:
            assert len(code.suggestion) > 0


class TestMCPErrorResponse:
    """Tests for MCPErrorResponse dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        response = MCPErrorResponse()
        assert response.success is False
        assert response.error_code == ""
        assert response.retryable is False

    def test_custom_values(self):
        """Test custom values are set correctly."""
        response = MCPErrorResponse(
            error_code="E1001",
            error_type="user_error",
            message="Test error",
            suggestion="Fix it",
            operation="test",
            retryable=True,
        )
        assert response.error_code == "E1001"
        assert response.message == "Test error"
        assert response.retryable is True

    def test_to_dict_basic(self):
        """Test to_dict returns correct structure."""
        response = MCPErrorResponse(
            error_code="E1001",
            error_type="user_error",
            message="Test error",
            suggestion="Fix it",
        )
        result = response.to_dict()

        assert result["success"] is False
        assert result["error"]["code"] == "E1001"
        assert result["error"]["type"] == "user_error"
        assert result["error"]["message"] == "Test error"
        assert result["error"]["suggestion"] == "Fix it"
        assert "timestamp" in result

    def test_to_dict_with_details(self):
        """Test to_dict includes details when provided."""
        response = MCPErrorResponse(
            error_code="E1001",
            error_type="user_error",
            message="Test",
            suggestion="Fix",
            details={"field": "name"},
        )
        result = response.to_dict()

        assert result["error"]["details"] == {"field": "name"}

    def test_to_dict_with_operation(self):
        """Test to_dict includes operation when provided."""
        response = MCPErrorResponse(
            error_code="E1001",
            error_type="user_error",
            message="Test",
            suggestion="Fix",
            operation="store",
        )
        result = response.to_dict()

        assert result["operation"] == "store"

    def test_timestamp_is_present(self):
        """Test timestamp is automatically set."""
        response = MCPErrorResponse()
        assert response.timestamp is not None
        assert len(response.timestamp) > 0


class TestSanitizeErrorMessage:
    """Tests for sanitize_error_message function."""

    def test_sanitizes_api_key(self):
        """Test API key is redacted."""
        message = "Error with api_key='secret123'"
        result = sanitize_error_message(message)
        assert "secret123" not in result
        assert "REDACTED" in result

    def test_sanitizes_password(self):
        """Test password is redacted."""
        message = "Error with password=mypassword"
        result = sanitize_error_message(message)
        assert "mypassword" not in result
        assert "REDACTED" in result

    def test_sanitizes_token(self):
        """Test token is redacted."""
        message = "Error with token=abc123xyz"
        result = sanitize_error_message(message)
        assert "abc123xyz" not in result
        assert "REDACTED" in result

    def test_sanitizes_bearer_token(self):
        """Test Bearer token is redacted."""
        message = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = sanitize_error_message(message)
        assert "eyJhbG" not in result
        assert "REDACTED" in result

    def test_sanitizes_user_path(self):
        """Test user paths are anonymized."""
        message = "File not found: /Users/johnsmith/secret/file.txt"
        result = sanitize_error_message(message)
        assert "johnsmith" not in result
        assert "<USER>" in result

    def test_preserves_non_sensitive_content(self):
        """Test non-sensitive content is preserved."""
        message = "Collection 'test-collection' not found"
        result = sanitize_error_message(message)
        assert result == message

    def test_sanitizes_secret(self):
        """Test secret values are redacted."""
        message = "Error with secret='mysecretvalue'"
        result = sanitize_error_message(message)
        assert "mysecretvalue" not in result


class TestCreateErrorResponse:
    """Tests for create_error_response function."""

    def test_basic_error_response(self):
        """Test basic error response creation."""
        response = create_error_response(ErrorCode.VALIDATION_ERROR)

        assert response["success"] is False
        assert response["error"]["code"] == "E1001"
        assert response["error"]["type"] == "user_error"
        assert response["retryable"] is False

    def test_custom_message(self):
        """Test custom message overrides default."""
        response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            message="Custom error message",
        )

        assert response["error"]["message"] == "Custom error message"

    def test_custom_suggestion(self):
        """Test custom suggestion overrides default."""
        response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            suggestion="Custom suggestion",
        )

        assert response["error"]["suggestion"] == "Custom suggestion"

    def test_details_included(self):
        """Test details are included in response."""
        response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            details={"field": "name", "value": "invalid"},
        )

        assert response["error"]["details"]["field"] == "name"
        assert response["error"]["details"]["value"] == "invalid"

    def test_operation_included(self):
        """Test operation is included in response."""
        response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            operation="store",
        )

        assert response["operation"] == "store"

    def test_retryable_for_transient_errors(self):
        """Test retryable is True for transient errors."""
        response = create_error_response(ErrorCode.DAEMON_UNAVAILABLE)
        assert response["retryable"] is True

    def test_not_retryable_for_user_errors(self):
        """Test retryable is False for user errors."""
        response = create_error_response(ErrorCode.VALIDATION_ERROR)
        assert response["retryable"] is False

    def test_sanitization_applied(self):
        """Test message sanitization is applied."""
        response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            message="Error with api_key='secret123'",
            sanitize=True,
        )

        assert "secret123" not in response["error"]["message"]

    def test_sanitization_can_be_disabled(self):
        """Test sanitization can be disabled."""
        response = create_error_response(
            ErrorCode.INTERNAL_ERROR,
            message="Error with special_value='test'",
            sanitize=False,
        )

        assert "special_value='test'" in response["error"]["message"]


class TestHandleToolError:
    """Tests for handle_tool_error function."""

    def test_handles_value_error(self):
        """Test ValueError is mapped correctly."""
        exc = ValueError("Invalid value")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E1001"  # VALIDATION_ERROR

    def test_handles_key_error(self):
        """Test KeyError is mapped correctly."""
        exc = KeyError("missing_key")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E1002"  # MISSING_REQUIRED_FIELD

    def test_handles_timeout_error(self):
        """Test TimeoutError is mapped correctly."""
        exc = TimeoutError("Connection timed out")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E3003"  # CONNECTION_TIMEOUT
        assert response["retryable"] is True

    def test_handles_permission_error(self):
        """Test PermissionError is mapped correctly."""
        exc = PermissionError("Access denied")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E5002"  # PERMISSION_DENIED

    def test_handles_file_not_found(self):
        """Test FileNotFoundError is mapped correctly."""
        exc = FileNotFoundError("File not found")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E2001"  # NOT_FOUND

    def test_handles_connection_error(self):
        """Test ConnectionError is mapped correctly."""
        exc = ConnectionError("Connection refused")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E3002"  # QDRANT_CONNECTION_ERROR

    def test_handles_unknown_exception(self):
        """Test unknown exceptions map to INTERNAL_ERROR."""
        exc = RuntimeError("Something went wrong")
        response = handle_tool_error(exc)

        assert response["error"]["code"] == "E9001"  # INTERNAL_ERROR

    def test_includes_exception_type(self):
        """Test exception type is included in details."""
        exc = ValueError("Test")
        response = handle_tool_error(exc)

        assert response["error"]["details"]["exception_type"] == "ValueError"

    def test_includes_operation(self):
        """Test operation is included when provided."""
        exc = ValueError("Test")
        response = handle_tool_error(exc, operation="store")

        assert response["operation"] == "store"

    def test_includes_traceback_when_requested(self):
        """Test traceback is included when requested."""
        exc = ValueError("Test")
        response = handle_tool_error(exc, include_traceback=True)

        assert "traceback" in response["error"]["details"]


class TestValidationError:
    """Tests for validation_error convenience function."""

    def test_creates_validation_error(self):
        """Test validation error is created correctly."""
        response = validation_error(
            field="collection_name",
            message="Collection name cannot be empty",
        )

        assert response["error"]["code"] == "E1001"
        assert response["error"]["details"]["field"] == "collection_name"

    def test_includes_operation(self):
        """Test operation is included."""
        response = validation_error(
            field="name",
            message="Invalid",
            operation="create",
        )

        assert response["operation"] == "create"


class TestMissingFieldError:
    """Tests for missing_field_error convenience function."""

    def test_creates_missing_field_error(self):
        """Test missing field error is created correctly."""
        response = missing_field_error(field="query")

        assert response["error"]["code"] == "E1002"
        assert "query" in response["error"]["message"]
        assert response["error"]["details"]["field"] == "query"


class TestNotFoundError:
    """Tests for not_found_error convenience function."""

    def test_collection_not_found(self):
        """Test collection not found error."""
        response = not_found_error(
            resource_type="collection",
            identifier="test-collection",
        )

        assert response["error"]["code"] == "E2002"  # COLLECTION_NOT_FOUND
        assert "test-collection" in response["error"]["message"]

    def test_document_not_found(self):
        """Test document not found error."""
        response = not_found_error(
            resource_type="document",
            identifier="doc-123",
        )

        assert response["error"]["code"] == "E2003"  # DOCUMENT_NOT_FOUND

    def test_project_not_found(self):
        """Test project not found error."""
        response = not_found_error(
            resource_type="project",
            identifier="my-project",
        )

        assert response["error"]["code"] == "E2004"  # PROJECT_NOT_FOUND

    def test_generic_not_found(self):
        """Test generic not found error."""
        response = not_found_error(
            resource_type="resource",
            identifier="unknown",
        )

        assert response["error"]["code"] == "E2001"  # NOT_FOUND


class TestDaemonUnavailableError:
    """Tests for daemon_unavailable_error convenience function."""

    def test_creates_daemon_error(self):
        """Test daemon unavailable error is created."""
        response = daemon_unavailable_error()

        assert response["error"]["code"] == "E3001"
        assert response["retryable"] is True
        assert response["error"]["details"]["daemon_available"] is False

    def test_includes_fallback_info(self):
        """Test fallback info is included when available."""
        response = daemon_unavailable_error(fallback_available=True)

        assert response["error"]["details"]["fallback_available"] is True
        assert "queued" in response["error"]["suggestion"].lower()


class TestCollectionError:
    """Tests for collection_error convenience function."""

    def test_not_found_collection_error(self):
        """Test collection not found error."""
        response = collection_error(
            collection_name="test",
            action="access",
            reason="Collection does not exist",
        )

        assert response["error"]["code"] == "E2002"

    def test_permission_collection_error(self):
        """Test collection permission error."""
        response = collection_error(
            collection_name="test",
            action="create",
            reason="Permission denied for this operation",
        )

        assert response["error"]["code"] == "E5002"

    def test_generic_collection_error(self):
        """Test generic collection error."""
        response = collection_error(
            collection_name="test",
            action="delete",
            reason="Unknown error",
        )

        assert response["error"]["code"] == "E4001"  # OPERATION_FAILED


class TestSearchError:
    """Tests for search_error convenience function."""

    def test_creates_search_error(self):
        """Test search error is created."""
        response = search_error(
            query="test query",
            reason="Index not ready",
        )

        assert response["error"]["code"] == "E4003"
        assert "Index not ready" in response["error"]["message"]

    def test_truncates_long_query(self):
        """Test long queries are truncated in details."""
        long_query = "x" * 200
        response = search_error(
            query=long_query,
            reason="Failed",
        )

        assert len(response["error"]["details"]["query_preview"]) < 200
        assert "..." in response["error"]["details"]["query_preview"]


class TestStorageError:
    """Tests for storage_error convenience function."""

    def test_creates_storage_error(self):
        """Test storage error is created."""
        response = storage_error(reason="Disk full")

        assert response["error"]["code"] == "E4002"
        assert "Disk full" in response["error"]["message"]

    def test_includes_collection(self):
        """Test collection is included when provided."""
        response = storage_error(
            reason="Error",
            collection="test-collection",
        )

        assert response["error"]["details"]["collection"] == "test-collection"


class TestInvalidActionError:
    """Tests for invalid_action_error convenience function."""

    def test_creates_invalid_action_error(self):
        """Test invalid action error is created."""
        response = invalid_action_error(
            action="invalid",
            valid_actions=["create", "delete", "list"],
        )

        assert response["error"]["code"] == "E1004"
        assert "invalid" in response["error"]["message"]
        assert "create" in response["error"]["suggestion"]
        assert response["error"]["details"]["action"] == "invalid"
        assert response["error"]["details"]["valid_actions"] == ["create", "delete", "list"]
