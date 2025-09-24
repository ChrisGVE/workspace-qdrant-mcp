"""
Exception classes for metadata workflow system.

This module defines custom exception classes for the metadata workflow system,
providing specific error handling for different failure modes during metadata
extraction, YAML generation, batch processing, and incremental updates.
"""

from typing import Optional, Any, List


class MetadataError(Exception):
    """Base exception for metadata workflow system errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        """
        Initialize metadata error.

        Args:
            message: Error message
            details: Optional error details dictionary
        """
        super().__init__(message)
        self.details = details or {}


class AggregationError(MetadataError):
    """Exception raised during metadata aggregation failures."""

    def __init__(
        self,
        message: str,
        parser_type: Optional[str] = None,
        file_path: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize aggregation error.

        Args:
            message: Error message
            parser_type: Type of parser that failed
            file_path: Path to file that caused error
            details: Additional error details
        """
        super().__init__(message, details)
        self.parser_type = parser_type
        self.file_path = file_path


class YAMLGenerationError(MetadataError):
    """Exception raised during YAML generation failures."""

    def __init__(
        self,
        message: str,
        serialization_errors: Optional[List[str]] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize YAML generation error.

        Args:
            message: Error message
            serialization_errors: List of serialization error messages
            details: Additional error details
        """
        super().__init__(message, details)
        self.serialization_errors = serialization_errors or []


class BatchProcessingError(MetadataError):
    """Exception raised during batch processing failures."""

    def __init__(
        self,
        message: str,
        failed_documents: Optional[List[str]] = None,
        partial_results: Optional[dict] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize batch processing error.

        Args:
            message: Error message
            failed_documents: List of document paths that failed
            partial_results: Results from successful documents
            details: Additional error details
        """
        super().__init__(message, details)
        self.failed_documents = failed_documents or []
        self.partial_results = partial_results or {}


class IncrementalTrackingError(MetadataError):
    """Exception raised during incremental tracking failures."""

    def __init__(
        self,
        message: str,
        storage_error: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize incremental tracking error.

        Args:
            message: Error message
            storage_error: Storage-related error message
            details: Additional error details
        """
        super().__init__(message, details)
        self.storage_error = storage_error


class WorkflowConfigurationError(MetadataError):
    """Exception raised for workflow configuration errors."""

    def __init__(
        self,
        message: str,
        invalid_config: Optional[dict] = None,
        details: Optional[dict] = None,
    ) -> None:
        """
        Initialize workflow configuration error.

        Args:
            message: Error message
            invalid_config: Invalid configuration dictionary
            details: Additional error details
        """
        super().__init__(message, details)
        self.invalid_config = invalid_config or {}


# Export all exception classes
__all__ = [
    "MetadataError",
    "AggregationError",
    "YAMLGenerationError",
    "BatchProcessingError",
    "IncrementalTrackingError",
    "WorkflowConfigurationError",
]