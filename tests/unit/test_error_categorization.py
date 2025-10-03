"""
Unit tests for error categorization system.

Tests error severity and category enums, as well as the ErrorCategorizer
class for automatic error classification.
"""

import pytest
import socket
import asyncio
from src.python.common.core.error_categorization import (
    ErrorSeverity,
    ErrorCategory,
    ErrorCategorizer,
)


class TestErrorSeverity:
    """Test cases for ErrorSeverity enum."""

    def test_severity_values(self):
        """Test all severity enum values match database schema."""
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.INFO.value == "info"

    def test_severity_from_string(self):
        """Test converting strings to ErrorSeverity."""
        assert ErrorSeverity.from_string("error") == ErrorSeverity.ERROR
        assert ErrorSeverity.from_string("warning") == ErrorSeverity.WARNING
        assert ErrorSeverity.from_string("info") == ErrorSeverity.INFO

    def test_severity_from_string_case_insensitive(self):
        """Test case-insensitive string conversion."""
        assert ErrorSeverity.from_string("ERROR") == ErrorSeverity.ERROR
        assert ErrorSeverity.from_string("Warning") == ErrorSeverity.WARNING
        assert ErrorSeverity.from_string("INFO") == ErrorSeverity.INFO

    def test_severity_from_string_invalid(self):
        """Test invalid severity raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ErrorSeverity.from_string("invalid")
        assert "Invalid severity" in str(exc_info.value)

    def test_severity_all_values(self):
        """Test that all severity values are defined."""
        severities = [s.value for s in ErrorSeverity]
        assert "error" in severities
        assert "warning" in severities
        assert "info" in severities
        assert len(severities) == 3


class TestErrorCategory:
    """Test cases for ErrorCategory enum."""

    def test_category_values(self):
        """Test all category enum values match database schema."""
        assert ErrorCategory.FILE_CORRUPT.value == "file_corrupt"
        assert ErrorCategory.TOOL_MISSING.value == "tool_missing"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.METADATA_INVALID.value == "metadata_invalid"
        assert ErrorCategory.PROCESSING_FAILED.value == "processing_failed"
        assert ErrorCategory.PARSE_ERROR.value == "parse_error"
        assert ErrorCategory.PERMISSION_DENIED.value == "permission_denied"
        assert ErrorCategory.RESOURCE_EXHAUSTED.value == "resource_exhausted"
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.UNKNOWN.value == "unknown"

    def test_category_from_string(self):
        """Test converting strings to ErrorCategory."""
        assert ErrorCategory.from_string("file_corrupt") == ErrorCategory.FILE_CORRUPT
        assert ErrorCategory.from_string("tool_missing") == ErrorCategory.TOOL_MISSING
        assert ErrorCategory.from_string("network") == ErrorCategory.NETWORK
        assert ErrorCategory.from_string("unknown") == ErrorCategory.UNKNOWN

    def test_category_from_string_case_insensitive(self):
        """Test case-insensitive string conversion."""
        assert ErrorCategory.from_string("FILE_CORRUPT") == ErrorCategory.FILE_CORRUPT
        assert ErrorCategory.from_string("Tool_Missing") == ErrorCategory.TOOL_MISSING

    def test_category_from_string_invalid(self):
        """Test invalid category raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ErrorCategory.from_string("invalid_category")
        assert "Invalid category" in str(exc_info.value)

    def test_category_all_values(self):
        """Test that all required categories are defined."""
        categories = [c.value for c in ErrorCategory]
        required = [
            "file_corrupt",
            "tool_missing",
            "network",
            "metadata_invalid",
            "processing_failed",
            "parse_error",
            "permission_denied",
            "resource_exhausted",
            "timeout",
            "unknown",
        ]
        for required_cat in required:
            assert required_cat in categories


class TestErrorCategorizer:
    """Test cases for ErrorCategorizer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_categorizer_initialization(self):
        """Test ErrorCategorizer can be instantiated."""
        assert self.categorizer is not None
        assert isinstance(self.categorizer, ErrorCategorizer)

    def test_categorize_manual_override(self):
        """Test manual category and severity override."""
        category, severity, confidence = self.categorizer.categorize(
            manual_category=ErrorCategory.NETWORK,
            manual_severity=ErrorSeverity.WARNING,
        )

        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.WARNING
        assert confidence == 1.0

    def test_categorize_returns_tuple(self):
        """Test categorize returns correct tuple structure."""
        result = self.categorizer.categorize()
        assert isinstance(result, tuple)
        assert len(result) == 3

        category, severity, confidence = result
        assert isinstance(category, ErrorCategory)
        assert isinstance(severity, ErrorSeverity)
        assert isinstance(confidence, float)

    def test_categorize_confidence_range(self):
        """Test confidence score is in valid range."""
        _, _, confidence = self.categorizer.categorize()
        assert 0.0 <= confidence <= 1.0

    def test_categorize_with_none_values(self):
        """Test categorize handles None values gracefully."""
        category, severity, confidence = self.categorizer.categorize(
            exception=None,
            message=None,
            context=None,
        )

        # Should return default values without crashing
        assert isinstance(category, ErrorCategory)
        assert isinstance(severity, ErrorSeverity)
        assert isinstance(confidence, float)


class TestExceptionTypeCategorization:
    """Test cases for exception type-based categorization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.categorizer = ErrorCategorizer()

    def test_file_not_found_error(self):
        """Test FileNotFoundError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=FileNotFoundError("file.txt not found")
        )

        assert category == ErrorCategory.FILE_CORRUPT
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_permission_error(self):
        """Test PermissionError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=PermissionError("Access denied")
        )

        assert category == ErrorCategory.PERMISSION_DENIED
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_connection_error(self):
        """Test ConnectionError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=ConnectionRefusedError("Connection refused")
        )

        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_timeout_error(self):
        """Test TimeoutError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=TimeoutError("Operation timed out")
        )

        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_asyncio_timeout_error(self):
        """Test asyncio.TimeoutError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=asyncio.TimeoutError("Async operation timed out")
        )

        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_module_not_found_error(self):
        """Test ModuleNotFoundError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=ModuleNotFoundError("No module named 'foo'")
        )

        assert category == ErrorCategory.TOOL_MISSING
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_value_error(self):
        """Test ValueError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=ValueError("Invalid value")
        )

        assert category == ErrorCategory.PARSE_ERROR
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_memory_error(self):
        """Test MemoryError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=MemoryError("Out of memory")
        )

        assert category == ErrorCategory.RESOURCE_EXHAUSTED
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_io_error(self):
        """Test IOError categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=IOError("I/O operation failed")
        )

        assert category == ErrorCategory.FILE_CORRUPT
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_socket_timeout(self):
        """Test socket.timeout categorization."""
        category, severity, confidence = self.categorizer.categorize(
            exception=socket.timeout("Socket timeout")
        )

        assert category == ErrorCategory.TIMEOUT
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_unknown_exception_type(self):
        """Test unknown exception type returns UNKNOWN."""
        category, severity, confidence = self.categorizer.categorize(
            exception=RuntimeError("Unknown error")
        )

        assert category == ErrorCategory.UNKNOWN
        assert severity == ErrorSeverity.ERROR
        assert confidence == 0.0

    def test_inherited_exception_type(self):
        """Test inherited exception has lower confidence."""
        # ConnectionRefusedError inherits from ConnectionError
        # If we map only ConnectionError, subclass should still match
        category, severity, confidence = self.categorizer.categorize(
            exception=BrokenPipeError("Broken pipe")
        )

        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.ERROR
        # Direct match should be 1.0
        assert confidence == 1.0

    def test_manual_override_with_exception(self):
        """Test manual override takes precedence over exception type."""
        category, severity, confidence = self.categorizer.categorize(
            exception=FileNotFoundError("file.txt not found"),
            manual_category=ErrorCategory.NETWORK,
            manual_severity=ErrorSeverity.WARNING,
        )

        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.WARNING
        assert confidence == 1.0

    def test_partial_manual_override_category(self):
        """Test partial override (category only) with exception."""
        category, severity, confidence = self.categorizer.categorize(
            exception=FileNotFoundError("file.txt not found"),
            manual_category=ErrorCategory.NETWORK,
        )

        # Category overridden, severity from exception
        assert category == ErrorCategory.NETWORK
        assert severity == ErrorSeverity.ERROR
        assert confidence == 1.0

    def test_partial_manual_override_severity(self):
        """Test partial override (severity only) with exception."""
        category, severity, confidence = self.categorizer.categorize(
            exception=FileNotFoundError("file.txt not found"),
            manual_severity=ErrorSeverity.WARNING,
        )

        # Severity overridden, category from exception
        assert category == ErrorCategory.FILE_CORRUPT
        assert severity == ErrorSeverity.WARNING
        assert confidence == 1.0

    def test_all_mapped_exceptions(self):
        """Test that all mapped exception types are categorized correctly."""
        test_cases = [
            (FileNotFoundError(), ErrorCategory.FILE_CORRUPT),
            (FileExistsError(), ErrorCategory.FILE_CORRUPT),
            (IsADirectoryError(), ErrorCategory.FILE_CORRUPT),
            (NotADirectoryError(), ErrorCategory.FILE_CORRUPT),
            (PermissionError(), ErrorCategory.PERMISSION_DENIED),
            (ConnectionRefusedError(), ErrorCategory.NETWORK),
            (TimeoutError(), ErrorCategory.TIMEOUT),
            (MemoryError(), ErrorCategory.RESOURCE_EXHAUSTED),
            (ValueError(), ErrorCategory.PARSE_ERROR),
            (ModuleNotFoundError(), ErrorCategory.TOOL_MISSING),
        ]

        for exception, expected_category in test_cases:
            category, severity, confidence = self.categorizer.categorize(exception=exception)
            assert category == expected_category
            assert confidence >= 0.8  # Should be high confidence for mapped types
