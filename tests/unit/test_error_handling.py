"""
Unit tests for error_handling module.

Tests core error classes and their behavior, including the new
IncompatibleVersionError for backup/restore version validation.
"""

import pytest
from common.core.error_handling import (
    ErrorCategory,
    ErrorSeverity,
    IncompatibleVersionError,
    WorkspaceError,
)


class TestIncompatibleVersionError:
    """Test suite for IncompatibleVersionError exception class."""

    def test_incompatible_version_error_basic(self):
        """Test basic IncompatibleVersionError creation."""
        error = IncompatibleVersionError(
            "Backup version incompatible with current system version"
        )

        assert isinstance(error, WorkspaceError)
        assert error.message == "Backup version incompatible with current system version"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.retryable is False

    def test_incompatible_version_error_with_versions(self):
        """Test IncompatibleVersionError with version information."""
        error = IncompatibleVersionError(
            "Cannot restore backup from v0.1.0 to v0.2.0",
            backup_version="0.1.0",
            current_version="0.2.0"
        )

        assert error.context["backup_version"] == "0.1.0"
        assert error.context["current_version"] == "0.2.0"
        assert error.retryable is False

    def test_incompatible_version_error_str_representation(self):
        """Test string representation includes version context."""
        error = IncompatibleVersionError(
            "Version mismatch detected",
            backup_version="1.0.0",
            current_version="2.0.0"
        )

        error_str = str(error)
        assert "Version mismatch detected" in error_str
        assert "validation" in error_str.lower()

    def test_incompatible_version_error_to_dict(self):
        """Test conversion to dictionary for logging."""
        error = IncompatibleVersionError(
            "Incompatible major version",
            backup_version="1.2.3",
            current_version="2.0.0"
        )

        error_dict = error.to_dict()

        assert error_dict["message"] == "Incompatible major version"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == "high"
        assert error_dict["retryable"] is False
        assert error_dict["context"]["backup_version"] == "1.2.3"
        assert error_dict["context"]["current_version"] == "2.0.0"

    def test_incompatible_version_error_with_additional_context(self):
        """Test IncompatibleVersionError with additional context."""
        error = IncompatibleVersionError(
            "Patch version difference",
            backup_version="0.2.0",
            current_version="0.2.1",
            context={"backup_path": "/path/to/backup"}
        )

        assert error.context["backup_version"] == "0.2.0"
        assert error.context["current_version"] == "0.2.1"
        assert error.context["backup_path"] == "/path/to/backup"

    def test_incompatible_version_error_not_retryable(self):
        """Test that IncompatibleVersionError is never retryable."""
        error = IncompatibleVersionError(
            "Version incompatibility cannot be retried",
            backup_version="0.1.0",
            current_version="0.2.0"
        )

        # Version incompatibility is a permanent condition
        assert error.retryable is False

    def test_incompatible_version_error_high_severity(self):
        """Test that IncompatibleVersionError has HIGH severity."""
        error = IncompatibleVersionError(
            "Critical version mismatch",
            backup_version="1.0.0",
            current_version="2.0.0"
        )

        # Preventing data corruption requires HIGH severity
        assert error.severity == ErrorSeverity.HIGH

    def test_incompatible_version_error_validation_category(self):
        """Test that IncompatibleVersionError uses VALIDATION category."""
        error = IncompatibleVersionError(
            "Version validation failed"
        )

        assert error.category == ErrorCategory.VALIDATION

    def test_incompatible_version_error_with_cause(self):
        """Test IncompatibleVersionError with underlying cause."""
        cause = ValueError("Invalid version format")
        error = IncompatibleVersionError(
            "Version parsing failed",
            backup_version="invalid",
            current_version="0.2.0",
            cause=cause
        )

        assert error.cause is cause
        assert "caused by" in str(error)

    def test_incompatible_version_error_missing_versions(self):
        """Test IncompatibleVersionError when versions are not provided."""
        error = IncompatibleVersionError(
            "Version information missing from backup"
        )

        # Should have None values for missing version info
        assert error.context["backup_version"] is None
        assert error.context["current_version"] is None

    def test_incompatible_version_error_timestamp(self):
        """Test that error includes timestamp."""
        error = IncompatibleVersionError(
            "Test error",
            backup_version="0.1.0",
            current_version="0.2.0"
        )

        assert hasattr(error, "timestamp")
        assert error.timestamp > 0

    def test_incompatible_version_error_raises_correctly(self):
        """Test that IncompatibleVersionError can be raised and caught."""
        with pytest.raises(IncompatibleVersionError) as exc_info:
            raise IncompatibleVersionError(
                "Test raise",
                backup_version="1.0.0",
                current_version="2.0.0"
            )

        error = exc_info.value
        assert error.context["backup_version"] == "1.0.0"
        assert error.context["current_version"] == "2.0.0"
