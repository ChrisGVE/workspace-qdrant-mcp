from ...observability import get_logger

logger = get_logger(__name__)
"""
Unified error handling system for document parsers.

This module provides a comprehensive set of exception classes for different
failure modes in document parsing, along with standardized error reporting
and recovery mechanisms.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for parsing errors."""

    LOW = "low"  # Recoverable errors, warnings
    MEDIUM = "medium"  # Significant issues but parsing can continue
    HIGH = "high"  # Critical errors that prevent parsing
    CRITICAL = "critical"  # System-level failures


class ErrorCategory(Enum):
    """Categories of parsing errors."""

    FILE_ACCESS = "file_access"  # File not found, permission issues
    FILE_FORMAT = "file_format"  # Unsupported or invalid format
    FILE_CORRUPTION = "file_corruption"  # Corrupted or malformed files
    ENCODING = "encoding"  # Character encoding issues
    MEMORY = "memory"  # Memory-related errors
    PARSING = "parsing"  # General parsing failures
    VALIDATION = "validation"  # Content validation failures
    NETWORK = "network"  # Network-related issues (future use)
    SYSTEM = "system"  # System-level failures


class ParsingError(Exception):
    """
    Base exception class for all parsing errors.

    Provides structured error information including severity, category,
    recovery suggestions, and detailed context for debugging.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        category: ErrorCategory = ErrorCategory.PARSING,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[list[str]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        Initialize parsing error.

        Args:
            message: Human-readable error message
            file_path: Path to the file being parsed
            severity: Severity level of the error
            category: Category classification
            error_code: Unique error code for programmatic handling
            context: Additional context information
            recovery_suggestions: List of potential recovery actions
            original_exception: Original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.file_path = str(file_path) if file_path else None
        self.severity = severity
        self.category = category
        self.error_code = error_code or self._generate_error_code()
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception

    def _generate_error_code(self) -> str:
        """Generate a unique error code based on class and category."""
        class_name = self.__class__.__name__.upper()
        category_code = self.category.value.upper()
        return f"{class_name}_{category_code}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for structured logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "file_path": self.file_path,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_code": self.error_code,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "original_exception": str(self.original_exception)
            if self.original_exception
            else None,
        }

    def log_error(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """Log error with appropriate severity level."""
        log = logger_instance or logger
        error_dict = self.to_dict()

        if self.severity == ErrorSeverity.CRITICAL:
            log.critical(f"Critical parsing error: {self.message}", extra=error_dict)
        elif self.severity == ErrorSeverity.HIGH:
            log.error(f"Parsing error: {self.message}", extra=error_dict)
        elif self.severity == ErrorSeverity.MEDIUM:
            log.warning(f"Parsing warning: {self.message}", extra=error_dict)
        else:
            log.info(f"Parsing info: {self.message}", extra=error_dict)


class FileAccessError(ParsingError):
    """Errors related to file access and permissions."""

    def __init__(
        self, message: str, file_path: Optional[Union[str, Path]] = None, **kwargs
    ):
        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.FILE_ACCESS,
            recovery_suggestions=[
                "Check if file exists and is accessible",
                "Verify file permissions",
                "Ensure file is not locked by another process",
            ],
            **kwargs,
        )


class FileFormatError(ParsingError):
    """Errors related to unsupported or invalid file formats."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        detected_format: Optional[str] = None,
        expected_formats: Optional[list[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "detected_format": detected_format,
                "expected_formats": expected_formats,
            }
        )

        recovery_suggestions = [
            "Verify the file format is supported",
            "Check file extension matches content",
        ]
        if expected_formats:
            recovery_suggestions.append(
                f"Supported formats: {', '.join(expected_formats)}"
            )

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.FILE_FORMAT,
            context=context,
            recovery_suggestions=recovery_suggestions,
            **kwargs,
        )


class FileCorruptionError(ParsingError):
    """Errors related to corrupted or malformed files."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        corruption_type: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"corruption_type": corruption_type})

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.FILE_CORRUPTION,
            context=context,
            recovery_suggestions=[
                "Check if file is corrupted",
                "Try re-downloading or re-creating the file",
                "Verify file integrity",
                "Use file repair tools if available",
            ],
            **kwargs,
        )


class EncodingError(ParsingError):
    """Errors related to character encoding issues."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        detected_encoding: Optional[str] = None,
        attempted_encodings: Optional[list[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "detected_encoding": detected_encoding,
                "attempted_encodings": attempted_encodings,
            }
        )

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.ENCODING,
            context=context,
            recovery_suggestions=[
                "Try specifying encoding explicitly",
                "Convert file to UTF-8 encoding",
                "Use encoding detection tools",
                "Check for byte order marks (BOM)",
            ],
            **kwargs,
        )


class MemoryError(ParsingError):
    """Errors related to memory limitations during parsing."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        file_size: Optional[int] = None,
        memory_usage: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update(
            {
                "file_size": file_size,
                "memory_usage": memory_usage,
            }
        )

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=[
                "Try processing smaller files",
                "Use streaming/chunked processing",
                "Increase available memory",
                "Close other applications to free memory",
            ],
            **kwargs,
        )


class ValidationError(ParsingError):
    """Errors related to content validation failures."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        validation_rule: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"validation_rule": validation_rule})

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            recovery_suggestions=[
                "Review content validation rules",
                "Check if content meets expected format",
                "Consider using less strict validation",
            ],
            **kwargs,
        )


class ParsingTimeout(ParsingError):
    """Errors related to parsing timeouts."""

    def __init__(
        self,
        message: str,
        file_path: Optional[Union[str, Path]] = None,
        timeout_seconds: Optional[int] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context.update({"timeout_seconds": timeout_seconds})

        super().__init__(
            message,
            file_path=file_path,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.HIGH,
            context=context,
            recovery_suggestions=[
                "Increase timeout duration",
                "Try processing smaller sections",
                "Check system performance",
                "Consider using asynchronous processing",
            ],
            **kwargs,
        )


class SystemError(ParsingError):
    """System-level errors that affect parsing operations."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL,
            recovery_suggestions=[
                "Check system resources",
                "Restart the application",
                "Check for system updates",
                "Contact system administrator",
            ],
            **kwargs,
        )


class ErrorHandler:
    """
    Centralized error handling and recovery system.

    Provides standardized error handling across all parsers with
    logging, recovery suggestions, and error statistics tracking.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize error handler.

        Args:
            logger_instance: Custom logger instance to use
        """
        self.logger = logger_instance or logger
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}

    def handle_error(
        self,
        error: Exception,
        file_path: Optional[Union[str, Path]] = None,
        context: Optional[Dict[str, Any]] = None,
        auto_recover: bool = True,
    ) -> ParsingError:
        """
        Handle and classify parsing errors.

        Args:
            error: Original exception
            file_path: Path to file being parsed
            context: Additional context information
            auto_recover: Whether to attempt automatic recovery

        Returns:
            Classified ParsingError instance
        """
        # Classify error based on type and message
        parsing_error = self._classify_error(error, file_path, context)

        # Log the error
        parsing_error.log_error(self.logger)

        # Track error statistics
        self._track_error(parsing_error)

        # Attempt recovery if enabled
        if auto_recover and parsing_error.recovery_suggestions:
            self._attempt_recovery(parsing_error)

        return parsing_error

    def _classify_error(
        self,
        error: Exception,
        file_path: Optional[Union[str, Path]],
        context: Optional[Dict[str, Any]],
    ) -> ParsingError:
        """Classify generic exception into specific ParsingError type."""
        error_message = str(error)
        error_lower = error_message.lower()

        # File access errors
        if isinstance(error, FileNotFoundError):
            return FileAccessError(
                f"File not found: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        if isinstance(error, PermissionError):
            return FileAccessError(
                f"Permission denied: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        # Memory errors
        if isinstance(error, MemoryError) or "memory" in error_lower:
            return MemoryError(
                f"Memory error during parsing: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        # Encoding errors
        if (
            isinstance(error, (UnicodeError, UnicodeDecodeError))
            or "encoding" in error_lower
        ):
            return EncodingError(
                f"Encoding error: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        # Timeout errors
        if "timeout" in error_lower or isinstance(error, TimeoutError):
            return ParsingTimeout(
                f"Parsing timeout: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        # File format/corruption errors
        if any(
            keyword in error_lower
            for keyword in ["corrupt", "invalid", "malformed", "damaged"]
        ):
            return FileCorruptionError(
                f"File corruption detected: {error_message}",
                file_path=file_path,
                context=context,
                original_exception=error,
            )

        # Generic parsing error
        return ParsingError(
            f"Parsing failed: {error_message}",
            file_path=file_path,
            context=context,
            original_exception=error,
        )

    def _track_error(self, error: ParsingError) -> None:
        """Track error statistics."""
        error_key = f"{error.category.value}_{error.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

    def _attempt_recovery(self, error: ParsingError) -> bool:
        """
        Attempt automatic error recovery.

        Args:
            error: ParsingError to attempt recovery for

        Returns:
            True if recovery was attempted
        """
        recovery_key = error.error_code
        self.recovery_attempts[recovery_key] = (
            self.recovery_attempts.get(recovery_key, 0) + 1
        )

        # Log recovery attempt
        self.logger.info(
            f"Attempting recovery for {error.error_code}: {error.recovery_suggestions[0] if error.recovery_suggestions else 'No suggestions'}"
        )

        return True

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            "error_counts": dict(self.error_counts),
            "recovery_attempts": dict(self.recovery_attempts),
            "total_errors": sum(self.error_counts.values()),
            "total_recovery_attempts": sum(self.recovery_attempts.values()),
        }

    def reset_statistics(self) -> None:
        """Reset error statistics."""
        self.error_counts.clear()
        self.recovery_attempts.clear()


# Global error handler instance
_global_error_handler = ErrorHandler()


def handle_parsing_error(
    error: Exception,
    file_path: Optional[Union[str, Path]] = None,
    context: Optional[Dict[str, Any]] = None,
    auto_recover: bool = True,
) -> ParsingError:
    """
    Handle parsing error using global error handler.

    Args:
        error: Original exception
        file_path: Path to file being parsed
        context: Additional context information
        auto_recover: Whether to attempt automatic recovery

    Returns:
        Classified ParsingError instance
    """
    return _global_error_handler.handle_error(error, file_path, context, auto_recover)


def get_error_statistics() -> Dict[str, Any]:
    """Get global error handling statistics."""
    return _global_error_handler.get_error_statistics()


def reset_error_statistics() -> None:
    """Reset global error statistics."""
    _global_error_handler.reset_statistics()
