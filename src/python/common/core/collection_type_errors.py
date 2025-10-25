"""
Custom Exception Classes for Collection Type Management.

This module provides specialized exception classes for collection type operations
with error codes, recovery suggestions, and documentation links.

Exception Hierarchy:
    CollectionTypeError (base)
    ├── InvalidCollectionTypeError - Type classification failures
    ├── CollectionMigrationError - Migration operation failures
    ├── MetadataValidationError - Metadata validation failures
    └── CollectionTypeConfigError - Configuration issues

Each exception includes:
    - error_code: Unique error identifier for logging/tracking
    - message: Human-readable error description
    - recovery_suggestion: Actionable fix guidance
    - doc_link: Link to relevant documentation section

Usage:
    ```python
    from collection_type_errors import InvalidCollectionTypeError

    raise InvalidCollectionTypeError(
        collection_name="myproject",
        detected_type=CollectionType.UNKNOWN,
        error_code="TYPE_INVALID_PATTERN",
        recovery_suggestion="Rename to 'myproject-docs' for PROJECT type"
    )
    ```
"""

from typing import Any

from .collection_types import CollectionType


class CollectionTypeError(Exception):
    """
    Base exception for all collection type-related errors.

    All collection type exceptions inherit from this base class, allowing
    for unified error handling across the collection type management system.

    Attributes:
        error_code: Unique error code for tracking
        message: Human-readable error message
        recovery_suggestion: Suggested action to resolve the error
        doc_link: Link to relevant documentation
        details: Additional context about the error
    """

    def __init__(
        self,
        message: str,
        error_code: str = "COLLECTION_TYPE_ERROR",
        recovery_suggestion: str | None = None,
        doc_link: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """
        Initialize collection type error.

        Args:
            message: Human-readable error description
            error_code: Unique error identifier
            recovery_suggestion: Actionable fix guidance
            doc_link: Link to documentation section
            details: Additional error context
        """
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.recovery_suggestion = recovery_suggestion or "Contact support for assistance"
        self.doc_link = doc_link or "docs/collection_types/troubleshooting.md"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception to dictionary for logging/serialization.

        Returns:
            Dictionary representation of the error
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "recovery_suggestion": self.recovery_suggestion,
            "doc_link": self.doc_link,
            "details": self.details,
            "exception_type": self.__class__.__name__,
        }

    def __str__(self) -> str:
        """Format error message with code and suggestion."""
        msg = f"[{self.error_code}] {self.message}"
        if self.recovery_suggestion:
            msg += f"\n→ Suggestion: {self.recovery_suggestion}"
        if self.doc_link:
            msg += f"\n→ Documentation: {self.doc_link}"
        return msg


class InvalidCollectionTypeError(CollectionTypeError):
    """
    Exception raised when collection type classification fails or is invalid.

    This error occurs when:
    - Collection name doesn't match any known type pattern
    - Type assignment conflicts with name pattern
    - Type detection has low confidence
    - Manual type override is invalid

    Examples:
        - Collection "myproject" (no dash) classified as UNKNOWN
        - Trying to set SYSTEM type on collection without __ prefix
        - Collection name matches multiple patterns ambiguously
    """

    def __init__(
        self,
        collection_name: str,
        detected_type: CollectionType | None = None,
        expected_type: CollectionType | None = None,
        error_code: str = "TYPE_INVALID_PATTERN",
        message: str | None = None,
        recovery_suggestion: str | None = None,
        confidence: float | None = None,
    ):
        """
        Initialize invalid collection type error.

        Args:
            collection_name: Name of the collection
            detected_type: Type that was detected (if any)
            expected_type: Type that was expected
            error_code: Specific error code
            message: Custom error message
            recovery_suggestion: Custom recovery suggestion
            confidence: Type detection confidence (0.0-1.0)
        """
        # Build default message if not provided
        if message is None:
            if detected_type == CollectionType.UNKNOWN:
                message = (
                    f"Collection '{collection_name}' doesn't match any known type pattern. "
                    f"Cannot classify collection type."
                )
            elif expected_type and detected_type != expected_type:
                message = (
                    f"Collection '{collection_name}' type mismatch: "
                    f"detected as {detected_type.value.upper() if detected_type else 'UNKNOWN'}, "
                    f"but expected {expected_type.value.upper()}"
                )
            else:
                message = f"Invalid collection type for '{collection_name}'"

        # Build default recovery suggestion if not provided
        if recovery_suggestion is None:
            recovery_suggestion = self._build_recovery_suggestion(
                collection_name, detected_type, expected_type
            )

        details = {
            "collection_name": collection_name,
            "detected_type": detected_type.value if detected_type else None,
            "expected_type": expected_type.value if expected_type else None,
            "confidence": confidence,
        }

        super().__init__(
            message=message,
            error_code=error_code,
            recovery_suggestion=recovery_suggestion,
            doc_link="docs/collection_types/collection-type-reference.md#naming-conventions",
            details=details,
        )

        self.collection_name = collection_name
        self.detected_type = detected_type
        self.expected_type = expected_type
        self.confidence = confidence

    @staticmethod
    def _build_recovery_suggestion(
        collection_name: str,
        detected_type: CollectionType | None,
        expected_type: CollectionType | None,
    ) -> str:
        """Build type-specific recovery suggestion."""
        if expected_type == CollectionType.SYSTEM:
            return f"Rename collection to '__{collection_name}' to match SYSTEM type pattern"
        elif expected_type == CollectionType.LIBRARY:
            base_name = collection_name.lstrip("_")
            return f"Rename collection to '_{base_name}' to match LIBRARY type pattern"
        elif expected_type == CollectionType.PROJECT:
            if "-" not in collection_name:
                return f"Rename collection to '{collection_name}-docs' to match PROJECT type pattern"
            return "Ensure collection name follows PROJECT pattern: {{project_id}}-{{suffix}}"
        elif expected_type == CollectionType.GLOBAL:
            valid_names = ["algorithms", "codebase", "context", "documents",
                          "knowledge", "memory", "projects", "workspace"]
            return f"Use one of the valid GLOBAL collection names: {', '.join(valid_names)}"
        else:
            return (
                "Choose a collection type and rename accordingly:\n"
                "  - SYSTEM: __name (e.g., __system_docs)\n"
                "  - LIBRARY: _name (e.g., _python_stdlib)\n"
                "  - PROJECT: {project}-{suffix} (e.g., myproj-docs)\n"
                "  - GLOBAL: Use predefined name (e.g., workspace)"
            )


class CollectionMigrationError(CollectionTypeError):
    """
    Exception raised when collection migration fails.

    This error occurs when:
    - Migration validation fails
    - Name collision detected
    - Backup creation fails
    - Migration execution fails
    - Rollback fails

    Examples:
        - Target collection name already exists
        - Required metadata cannot be generated
        - Qdrant operation fails during migration
        - Backup restoration fails
    """

    def __init__(
        self,
        collection_name: str,
        target_type: CollectionType | None = None,
        error_code: str = "MIG_FAILED",
        message: str | None = None,
        recovery_suggestion: str | None = None,
        validation_errors: list[str] | None = None,
        conflicts: list[str] | None = None,
        can_rollback: bool = False,
    ):
        """
        Initialize collection migration error.

        Args:
            collection_name: Name of the collection being migrated
            target_type: Target collection type
            error_code: Specific error code
            message: Custom error message
            recovery_suggestion: Custom recovery suggestion
            validation_errors: List of validation errors
            conflicts: List of conflicts detected
            can_rollback: Whether rollback is possible
        """
        # Build default message if not provided
        if message is None:
            if validation_errors:
                message = (
                    f"Migration validation failed for '{collection_name}' "
                    f"to {target_type.value.upper() if target_type else 'unknown'} type: "
                    f"{len(validation_errors)} errors found"
                )
            elif conflicts:
                message = (
                    f"Migration conflicts detected for '{collection_name}': "
                    f"{len(conflicts)} conflicts"
                )
            else:
                message = f"Migration failed for collection '{collection_name}'"

        # Build default recovery suggestion if not provided
        if recovery_suggestion is None:
            recovery_suggestion = self._build_recovery_suggestion(
                validation_errors, conflicts, can_rollback
            )

        details = {
            "collection_name": collection_name,
            "target_type": target_type.value if target_type else None,
            "validation_errors": validation_errors or [],
            "conflicts": conflicts or [],
            "can_rollback": can_rollback,
        }

        super().__init__(
            message=message,
            error_code=error_code,
            recovery_suggestion=recovery_suggestion,
            doc_link="docs/collection_types/migration-guide.md#troubleshooting-migrations",
            details=details,
        )

        self.collection_name = collection_name
        self.target_type = target_type
        self.validation_errors = validation_errors or []
        self.conflicts = conflicts or []
        self.can_rollback = can_rollback

    @staticmethod
    def _build_recovery_suggestion(
        validation_errors: list[str] | None,
        conflicts: list[str] | None,
        can_rollback: bool,
    ) -> str:
        """Build migration-specific recovery suggestion."""
        suggestions = []

        if validation_errors:
            suggestions.append(
                "Run 'wqm collections validate-types' to see detailed validation errors"
            )
            suggestions.append(
                "Fix validation errors before attempting migration"
            )

        if conflicts:
            suggestions.append(
                "Resolve naming conflicts (rename or delete conflicting collections)"
            )
            suggestions.append(
                "Use --force flag to override conflict warnings (use with caution)"
            )

        if can_rollback:
            suggestions.append(
                "Rollback available: Previous state can be restored from backup"
            )

        if not suggestions:
            suggestions.append(
                "Try migration in dry-run mode first: --dry-run"
            )
            suggestions.append(
                "Check daemon logs for detailed error information"
            )

        return "\n  - ".join([""] + suggestions)


class MetadataValidationError(CollectionTypeError):
    """
    Exception raised when metadata validation fails.

    This error occurs when:
    - Required metadata fields are missing
    - Metadata field values are invalid
    - Metadata field types are incorrect
    - Metadata constraints are violated

    Examples:
        - Missing 'language' field in LIBRARY collection
        - Invalid 'collection_name' pattern for SYSTEM type
        - 'priority' value outside allowed range (1-5)
        - Wrong type for 'symbols' field (not a list)
    """

    def __init__(
        self,
        collection_type: CollectionType,
        error_code: str = "META_VALIDATION_FAILED",
        message: str | None = None,
        recovery_suggestion: str | None = None,
        missing_fields: list[str] | None = None,
        invalid_fields: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Initialize metadata validation error.

        Args:
            collection_type: Collection type being validated
            error_code: Specific error code
            message: Custom error message
            recovery_suggestion: Custom recovery suggestion
            missing_fields: List of missing required fields
            invalid_fields: Dict of invalid fields and their errors
            metadata: The metadata that failed validation
        """
        # Build default message if not provided
        if message is None:
            error_parts = []
            if missing_fields:
                error_parts.append(f"{len(missing_fields)} missing required fields")
            if invalid_fields:
                error_parts.append(f"{len(invalid_fields)} invalid field values")

            message = (
                f"Metadata validation failed for {collection_type.value.upper()} collection: "
                f"{', '.join(error_parts) if error_parts else 'validation errors detected'}"
            )

        # Build default recovery suggestion if not provided
        if recovery_suggestion is None:
            recovery_suggestion = self._build_recovery_suggestion(
                collection_type, missing_fields, invalid_fields
            )

        details = {
            "collection_type": collection_type.value,
            "missing_fields": missing_fields or [],
            "invalid_fields": invalid_fields or {},
            "metadata_provided": metadata or {},
        }

        super().__init__(
            message=message,
            error_code=error_code,
            recovery_suggestion=recovery_suggestion,
            doc_link=f"docs/collection_types/collection-type-reference.md#{collection_type.value}-collections",
            details=details,
        )

        self.collection_type = collection_type
        self.missing_fields = missing_fields or []
        self.invalid_fields = invalid_fields or {}
        self.metadata = metadata

    @staticmethod
    def _build_recovery_suggestion(
        collection_type: CollectionType,
        missing_fields: list[str] | None,
        invalid_fields: dict[str, str] | None,
    ) -> str:
        """Build metadata-specific recovery suggestion."""
        suggestions = []

        if missing_fields:
            suggestions.append(
                f"Add missing required fields: {', '.join(missing_fields)}"
            )
            suggestions.append(
                f"See example: docs/collection_types/examples/{collection_type.value}-collection-example.yaml"
            )

        if invalid_fields:
            field_list = list(invalid_fields.keys())[:3]  # Show first 3
            suggestions.append(
                f"Fix invalid field values: {', '.join(field_list)}"
            )
            suggestions.append(
                "Check field type constraints and allowed values"
            )

        suggestions.append(
            f"Reference: docs/collection_types/collection-type-reference.md#{collection_type.value}-collections"
        )

        return "\n  - ".join([""] + suggestions)


class CollectionTypeConfigError(CollectionTypeError):
    """
    Exception raised when collection type configuration is invalid.

    This error occurs when:
    - Configuration for a collection type is missing
    - Configuration contains invalid settings
    - Performance settings are out of range
    - Migration settings are incompatible

    Examples:
        - Requesting config for UNKNOWN type
        - Invalid batch size (non-positive)
        - Invalid priority weight (outside 1-5)
        - Conflicting migration settings
    """

    def __init__(
        self,
        collection_type: CollectionType | None = None,
        config_issue: str | None = None,
        error_code: str = "CONFIG_INVALID",
        message: str | None = None,
        recovery_suggestion: str | None = None,
    ):
        """
        Initialize collection type config error.

        Args:
            collection_type: Collection type with config issue
            config_issue: Description of the configuration problem
            error_code: Specific error code
            message: Custom error message
            recovery_suggestion: Custom recovery suggestion
        """
        # Build default message if not provided
        if message is None:
            if collection_type == CollectionType.UNKNOWN:
                message = "Configuration not available for UNKNOWN collection type"
            elif config_issue:
                type_str = collection_type.value.upper() if collection_type else "collection"
                message = f"Configuration error for {type_str} type: {config_issue}"
            else:
                message = "Collection type configuration error"

        # Build default recovery suggestion if not provided
        if recovery_suggestion is None:
            if collection_type == CollectionType.UNKNOWN:
                recovery_suggestion = (
                    "Classify collection to a valid type (SYSTEM, LIBRARY, PROJECT, GLOBAL) "
                    "before accessing type configuration"
                )
            else:
                recovery_suggestion = (
                    "Check configuration values against type requirements\n"
                    "  - See: docs/collection_types/collection-type-reference.md"
                )

        details = {
            "collection_type": collection_type.value if collection_type else None,
            "config_issue": config_issue,
        }

        super().__init__(
            message=message,
            error_code=error_code,
            recovery_suggestion=recovery_suggestion,
            doc_link="docs/collection_types/api-reference.md#collectiontypeconfig",
            details=details,
        )

        self.collection_type = collection_type
        self.config_issue = config_issue


# Export all exception classes
__all__ = [
    "CollectionTypeError",
    "InvalidCollectionTypeError",
    "CollectionMigrationError",
    "MetadataValidationError",
    "CollectionTypeConfigError",
]
