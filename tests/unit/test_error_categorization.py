"""
Unit tests for error categorization system.

Tests error severity and category enums, as well as the ErrorCategorizer
class for automatic error classification.
"""

import pytest
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
