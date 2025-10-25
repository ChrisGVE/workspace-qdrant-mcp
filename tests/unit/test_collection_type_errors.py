"""Unit tests for collection type error handling."""

import pytest
from common.core.collection_type_errors import (
    CollectionMigrationError,
    CollectionTypeConfigError,
    CollectionTypeError,
    InvalidCollectionTypeError,
    MetadataValidationError,
)
from common.core.collection_types import CollectionType


class TestCollectionTypeError:
    """Tests for base CollectionTypeError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = CollectionTypeError(
            message="Test error",
            error_code="TEST_ERROR",
        )

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert "Contact support" in error.recovery_suggestion
        assert "troubleshooting.md" in error.doc_link

    def test_error_with_details(self):
        """Test error with custom details."""
        error = CollectionTypeError(
            message="Test error",
            error_code="TEST_ERROR",
            recovery_suggestion="Try fixing the thing",
            doc_link="docs/custom.md",
            details={"key": "value"},
        )

        assert error.recovery_suggestion == "Try fixing the thing"
        assert error.doc_link == "docs/custom.md"
        assert error.details["key"] == "value"

    def test_error_to_dict(self):
        """Test error serialization to dict."""
        error = CollectionTypeError(
            message="Test error",
            error_code="TEST_ERROR",
            details={"field": "value"},
        )

        error_dict = error.to_dict()

        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test error"
        assert error_dict["exception_type"] == "CollectionTypeError"
        assert "field" in error_dict["details"]

    def test_error_str_format(self):
        """Test error string formatting."""
        error = CollectionTypeError(
            message="Test error",
            error_code="TEST_ERROR",
            recovery_suggestion="Fix it",
            doc_link="docs/test.md",
        )

        error_str = str(error)

        assert "TEST_ERROR" in error_str
        assert "Test error" in error_str
        assert "Fix it" in error_str
        assert "docs/test.md" in error_str


class TestInvalidCollectionTypeError:
    """Tests for InvalidCollectionTypeError."""

    def test_unknown_type_error(self):
        """Test error for UNKNOWN type detection."""
        error = InvalidCollectionTypeError(
            collection_name="badname",
            detected_type=CollectionType.UNKNOWN,
        )

        assert "badname" in error.message
        assert "doesn't match any known type pattern" in error.message
        assert error.error_code == "TYPE_INVALID_PATTERN"
        assert "naming-conventions" in error.doc_link

    def test_type_mismatch_error(self):
        """Test error for type mismatch."""
        error = InvalidCollectionTypeError(
            collection_name="myproject",
            detected_type=CollectionType.PROJECT,
            expected_type=CollectionType.SYSTEM,
        )

        assert "type mismatch" in error.message
        assert "PROJECT" in error.message
        assert "SYSTEM" in error.message

    def test_system_type_recovery_suggestion(self):
        """Test recovery suggestion for SYSTEM type."""
        error = InvalidCollectionTypeError(
            collection_name="docs",
            expected_type=CollectionType.SYSTEM,
        )

        assert "__docs" in error.recovery_suggestion

    def test_library_type_recovery_suggestion(self):
        """Test recovery suggestion for LIBRARY type."""
        error = InvalidCollectionTypeError(
            collection_name="python",
            expected_type=CollectionType.LIBRARY,
        )

        assert "_python" in error.recovery_suggestion

    def test_project_type_recovery_suggestion(self):
        """Test recovery suggestion for PROJECT type."""
        error = InvalidCollectionTypeError(
            collection_name="myproject",
            expected_type=CollectionType.PROJECT,
        )

        assert "myproject-docs" in error.recovery_suggestion

    def test_global_type_recovery_suggestion(self):
        """Test recovery suggestion for GLOBAL type."""
        error = InvalidCollectionTypeError(
            collection_name="invalid",
            expected_type=CollectionType.GLOBAL,
        )

        assert "workspace" in error.recovery_suggestion
        assert "knowledge" in error.recovery_suggestion

    def test_error_with_confidence(self):
        """Test error with confidence score."""
        error = InvalidCollectionTypeError(
            collection_name="test",
            detected_type=CollectionType.PROJECT,
            confidence=0.45,
        )

        assert error.confidence == 0.45
        assert error.details["confidence"] == 0.45


class TestCollectionMigrationError:
    """Tests for CollectionMigrationError."""

    def test_validation_error(self):
        """Test migration error with validation errors."""
        error = CollectionMigrationError(
            collection_name="test-collection",
            target_type=CollectionType.PROJECT,
            validation_errors=["Missing field: project_id", "Invalid field: name"],
        )

        assert "validation failed" in error.message.lower()
        assert "2 errors" in error.message
        assert len(error.validation_errors) == 2
        assert "validate-types" in error.recovery_suggestion

    def test_conflict_error(self):
        """Test migration error with conflicts."""
        error = CollectionMigrationError(
            collection_name="test",
            conflicts=["Name collision with existing collection"],
        )

        assert "conflicts detected" in error.message.lower()
        assert len(error.conflicts) == 1
        assert "Resolve naming conflicts" in error.recovery_suggestion

    def test_rollback_available(self):
        """Test migration error with rollback option."""
        error = CollectionMigrationError(
            collection_name="test",
            can_rollback=True,
        )

        assert error.can_rollback
        assert "rollback" in error.recovery_suggestion.lower()

    def test_dry_run_suggestion(self):
        """Test dry-run suggestion for general failures."""
        error = CollectionMigrationError(
            collection_name="test",
        )

        assert "--dry-run" in error.recovery_suggestion


class TestMetadataValidationError:
    """Tests for MetadataValidationError."""

    def test_missing_fields_error(self):
        """Test error with missing fields."""
        error = MetadataValidationError(
            collection_type=CollectionType.LIBRARY,
            missing_fields=["language", "created_at"],
        )

        assert "validation failed" in error.message.lower()
        assert "2 missing" in error.message
        assert "language" in error.recovery_suggestion
        assert "created_at" in error.recovery_suggestion

    def test_invalid_fields_error(self):
        """Test error with invalid field values."""
        error = MetadataValidationError(
            collection_type=CollectionType.SYSTEM,
            invalid_fields={
                "priority": "Must be between 1 and 5",
                "name": "Invalid pattern",
            },
        )

        assert "validation failed" in error.message.lower()
        assert "2 invalid" in error.message
        assert "priority" in error.recovery_suggestion

    def test_example_link_in_suggestion(self):
        """Test that example YAML is suggested."""
        error = MetadataValidationError(
            collection_type=CollectionType.PROJECT,
            missing_fields=["project_id"],
        )

        assert "project-collection-example.yaml" in error.recovery_suggestion

    def test_type_specific_doc_link(self):
        """Test that doc link is type-specific."""
        error = MetadataValidationError(
            collection_type=CollectionType.LIBRARY,
            missing_fields=["language"],
        )

        assert "library-collections" in error.doc_link


class TestCollectionTypeConfigError:
    """Tests for CollectionTypeConfigError."""

    def test_unknown_type_config_error(self):
        """Test error for UNKNOWN type config request."""
        error = CollectionTypeConfigError(
            collection_type=CollectionType.UNKNOWN,
        )

        assert "not available for UNKNOWN" in error.message
        assert "Classify collection" in error.recovery_suggestion

    def test_config_issue_error(self):
        """Test error with specific config issue."""
        error = CollectionTypeConfigError(
            collection_type=CollectionType.SYSTEM,
            config_issue="Invalid batch size: -1",
        )

        assert "SYSTEM" in error.message
        assert "Invalid batch size" in error.message

    def test_api_reference_link(self):
        """Test that API reference is linked."""
        error = CollectionTypeConfigError(
            collection_type=CollectionType.PROJECT,
        )

        assert "api-reference.md" in error.doc_link


class TestErrorInheritance:
    """Tests for error inheritance and exception handling."""

    def test_all_errors_inherit_base(self):
        """Test that all errors inherit from CollectionTypeError."""
        errors = [
            InvalidCollectionTypeError("test"),
            CollectionMigrationError("test"),
            MetadataValidationError(CollectionType.SYSTEM),
            CollectionTypeConfigError(),
        ]

        for error in errors:
            assert isinstance(error, CollectionTypeError)
            assert isinstance(error, Exception)

    def test_catch_specific_error(self):
        """Test catching specific error types."""
        with pytest.raises(InvalidCollectionTypeError) as exc_info:
            raise InvalidCollectionTypeError(
                collection_name="test",
                detected_type=CollectionType.UNKNOWN,
            )

        assert exc_info.value.collection_name == "test"
        assert exc_info.value.detected_type == CollectionType.UNKNOWN

    def test_catch_base_error(self):
        """Test catching errors via base class."""
        with pytest.raises(CollectionTypeError):
            raise MetadataValidationError(
                collection_type=CollectionType.PROJECT,
                missing_fields=["field"],
            )
