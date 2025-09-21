"""
Tests for parser exception handling system.
"""

import logging
from pathlib import Path
from unittest.mock import Mock

import pytest

from wqm_cli.cli.parsers.exceptions import (
    ErrorCategory,
    ErrorHandler,
    ErrorSeverity,
    EncodingError,
    FileAccessError,
    FileCorruptionError,
    FileFormatError,
    MemoryError,
    ParsingError,
    ParsingTimeout,
    SystemError,
    ValidationError,
    get_error_statistics,
    handle_parsing_error,
    reset_error_statistics,
)


class TestParsingError:
    """Test ParsingError base class functionality."""

    def test_basic_error_creation(self):
        """Test creating a basic parsing error."""
        error = ParsingError(
            message="Test error message",
            file_path="test.txt",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.PARSING,
        )
        
        assert error.message == "Test error message"
        assert error.file_path == "test.txt"
        assert error.severity == ErrorSeverity.HIGH
        assert error.category == ErrorCategory.PARSING
        assert error.error_code == "PARSINGERROR_PARSING"
        assert isinstance(error.context, dict)
        assert isinstance(error.recovery_suggestions, list)

    def test_error_with_context(self):
        """Test creating error with additional context."""
        context = {"line_number": 42, "column": 15}
        recovery_suggestions = ["Check syntax", "Validate input"]
        
        error = ParsingError(
            message="Syntax error",
            context=context,
            recovery_suggestions=recovery_suggestions,
        )
        
        assert error.context == context
        assert error.recovery_suggestions == recovery_suggestions

    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        original_exception = ValueError("Original error")
        
        error = ParsingError(
            message="Test error",
            file_path="test.txt",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.VALIDATION,
            context={"test": "value"},
            original_exception=original_exception,
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "ParsingError"
        assert error_dict["message"] == "Test error"
        assert error_dict["file_path"] == "test.txt"
        assert error_dict["severity"] == "medium"
        assert error_dict["category"] == "validation"
        assert error_dict["context"] == {"test": "value"}
        assert error_dict["original_exception"] == "Original error"

    def test_error_logging(self, caplog):
        """Test error logging functionality."""
        with caplog.at_level(logging.ERROR):
            error = ParsingError(
                message="Test error",
                severity=ErrorSeverity.HIGH,
            )
            error.log_error()
            
            assert "Parsing error: Test error" in caplog.text

    def test_critical_error_logging(self, caplog):
        """Test critical error logging."""
        with caplog.at_level(logging.CRITICAL):
            error = ParsingError(
                message="Critical error",
                severity=ErrorSeverity.CRITICAL,
            )
            error.log_error()
            
            assert "Critical parsing error: Critical error" in caplog.text

    def test_warning_logging(self, caplog):
        """Test warning level logging."""
        with caplog.at_level(logging.WARNING):
            error = ParsingError(
                message="Warning message",
                severity=ErrorSeverity.MEDIUM,
            )
            error.log_error()
            
            assert "Parsing warning: Warning message" in caplog.text


class TestSpecificErrors:
    """Test specific error type implementations."""

    def test_file_access_error(self):
        """Test FileAccessError functionality."""
        error = FileAccessError(
            message="Permission denied",
            file_path="protected.txt"
        )
        
        assert error.category == ErrorCategory.FILE_ACCESS
        assert "Check if file exists" in error.recovery_suggestions[0]
        assert error.file_path == "protected.txt"

    def test_file_format_error(self):
        """Test FileFormatError functionality."""
        error = FileFormatError(
            message="Unsupported format",
            file_path="test.xyz",
            detected_format="unknown",
            expected_formats=["pdf", "txt", "md"]
        )
        
        assert error.category == ErrorCategory.FILE_FORMAT
        assert error.context["detected_format"] == "unknown"
        assert error.context["expected_formats"] == ["pdf", "txt", "md"]
        assert "Supported formats:" in error.recovery_suggestions[-1]

    def test_file_corruption_error(self):
        """Test FileCorruptionError functionality."""
        error = FileCorruptionError(
            message="File is corrupted",
            file_path="corrupt.pdf",
            corruption_type="header_damaged"
        )
        
        assert error.category == ErrorCategory.FILE_CORRUPTION
        assert error.context["corruption_type"] == "header_damaged"
        assert "Check if file is corrupted" in error.recovery_suggestions[0]

    def test_encoding_error(self):
        """Test EncodingError functionality."""
        error = EncodingError(
            message="Cannot decode file",
            file_path="text.txt",
            detected_encoding="latin1",
            attempted_encodings=["utf-8", "ascii"]
        )
        
        assert error.category == ErrorCategory.ENCODING
        assert error.context["detected_encoding"] == "latin1"
        assert error.context["attempted_encodings"] == ["utf-8", "ascii"]
        assert "Try specifying encoding" in error.recovery_suggestions[0]

    def test_memory_error(self):
        """Test MemoryError functionality."""
        error = MemoryError(
            message="Out of memory",
            file_path="large.pdf",
            file_size=1000000000,  # 1GB
            memory_usage=2000000000  # 2GB
        )
        
        assert error.category == ErrorCategory.MEMORY
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["file_size"] == 1000000000
        assert error.context["memory_usage"] == 2000000000
        assert "Try processing smaller files" in error.recovery_suggestions[0]

    def test_validation_error(self):
        """Test ValidationError functionality."""
        error = ValidationError(
            message="Content validation failed",
            file_path="invalid.xml",
            validation_rule="schema_compliance"
        )
        
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["validation_rule"] == "schema_compliance"
        assert "Review content validation" in error.recovery_suggestions[0]

    def test_parsing_timeout(self):
        """Test ParsingTimeout functionality."""
        error = ParsingTimeout(
            message="Parsing timed out",
            file_path="complex.pdf",
            timeout_seconds=300
        )
        
        assert error.category == ErrorCategory.PARSING
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["timeout_seconds"] == 300
        assert "Increase timeout duration" in error.recovery_suggestions[0]

    def test_system_error(self):
        """Test SystemError functionality."""
        error = SystemError(
            message="System failure"
        )
        
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.CRITICAL
        assert "Check system resources" in error.recovery_suggestions[0]


class TestErrorHandler:
    """Test ErrorHandler functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.handler = ErrorHandler()

    def test_handle_file_not_found(self, tmp_path):
        """Test handling FileNotFoundError."""
        original_error = FileNotFoundError("No such file")
        test_file = tmp_path / "nonexistent.txt"
        
        parsing_error = self.handler.handle_error(
            original_error,
            file_path=test_file,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, FileAccessError)
        assert parsing_error.category == ErrorCategory.FILE_ACCESS
        assert parsing_error.original_exception == original_error

    def test_handle_permission_error(self, tmp_path):
        """Test handling PermissionError."""
        original_error = PermissionError("Access denied")
        test_file = tmp_path / "protected.txt"
        
        parsing_error = self.handler.handle_error(
            original_error,
            file_path=test_file,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, FileAccessError)
        assert "Permission denied" in parsing_error.message

    def test_handle_unicode_error(self, tmp_path):
        """Test handling UnicodeDecodeError."""
        original_error = UnicodeDecodeError(
            "utf-8", b"\xff\xfe", 0, 1, "invalid start byte"
        )
        test_file = tmp_path / "binary.txt"
        
        parsing_error = self.handler.handle_error(
            original_error,
            file_path=test_file,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, EncodingError)
        assert parsing_error.category == ErrorCategory.ENCODING

    def test_handle_memory_error_by_type(self):
        """Test handling built-in MemoryError."""
        original_error = MemoryError("Out of memory")
        
        parsing_error = self.handler.handle_error(
            original_error,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, MemoryError)
        assert parsing_error.category == ErrorCategory.MEMORY

    def test_handle_timeout_error(self):
        """Test handling TimeoutError."""
        original_error = TimeoutError("Operation timed out")
        
        parsing_error = self.handler.handle_error(
            original_error,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, ParsingTimeout)
        assert parsing_error.category == ErrorCategory.PARSING

    def test_handle_corruption_keywords(self):
        """Test handling errors with corruption keywords."""
        original_error = RuntimeError("File is corrupted and cannot be read")
        
        parsing_error = self.handler.handle_error(
            original_error,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, FileCorruptionError)
        assert parsing_error.category == ErrorCategory.FILE_CORRUPTION

    def test_handle_generic_error(self):
        """Test handling generic exceptions."""
        original_error = ValueError("Generic error")
        
        parsing_error = self.handler.handle_error(
            original_error,
            auto_recover=False
        )
        
        assert isinstance(parsing_error, ParsingError)
        assert parsing_error.category == ErrorCategory.PARSING
        assert parsing_error.original_exception == original_error

    def test_error_statistics_tracking(self):
        """Test error statistics tracking."""
        # Handle several different errors
        self.handler.handle_error(FileNotFoundError("File 1"), auto_recover=False)
        self.handler.handle_error(FileNotFoundError("File 2"), auto_recover=False)
        self.handler.handle_error(UnicodeDecodeError("utf-8", b"", 0, 1, "test"), auto_recover=False)
        
        stats = self.handler.get_error_statistics()
        
        assert stats["total_errors"] == 3
        assert "file_access_high" in stats["error_counts"]
        assert "encoding_high" in stats["error_counts"]
        assert stats["error_counts"]["file_access_high"] == 2
        assert stats["error_counts"]["encoding_high"] == 1

    def test_reset_statistics(self):
        """Test resetting error statistics."""
        # Generate some errors first
        self.handler.handle_error(ValueError("Error 1"), auto_recover=False)
        self.handler.handle_error(ValueError("Error 2"), auto_recover=False)
        
        # Verify stats exist
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] == 2
        
        # Reset and verify
        self.handler.reset_statistics()
        stats = self.handler.get_error_statistics()
        assert stats["total_errors"] == 0
        assert len(stats["error_counts"]) == 0

    def test_recovery_attempt_tracking(self):
        """Test recovery attempt tracking."""
        error = ValueError("Recoverable error")
        
        # Handle with recovery enabled
        parsing_error = self.handler.handle_error(error, auto_recover=True)
        
        stats = self.handler.get_error_statistics()
        assert stats["total_recovery_attempts"] >= 1
        assert parsing_error.error_code in stats["recovery_attempts"]

    def test_custom_logger(self, caplog):
        """Test using custom logger."""
        custom_logger = logging.getLogger("test_logger")
        handler = ErrorHandler(custom_logger)
        
        with caplog.at_level(logging.ERROR, logger="test_logger"):
            handler.handle_error(ValueError("Test error"), auto_recover=False)
            
            assert "Parsing error: Parsing failed: Test error" in caplog.text


class TestGlobalErrorHandling:
    """Test global error handling functions."""

    def setup_method(self):
        """Reset global statistics before each test."""
        reset_error_statistics()

    def test_handle_parsing_error_function(self):
        """Test global handle_parsing_error function."""
        error = FileNotFoundError("Global test")
        
        parsing_error = handle_parsing_error(
            error,
            file_path="test.txt",
            auto_recover=False
        )
        
        assert isinstance(parsing_error, FileAccessError)
        assert parsing_error.file_path == "test.txt"

    def test_global_statistics(self):
        """Test global error statistics tracking."""
        # Handle some errors
        handle_parsing_error(ValueError("Error 1"), auto_recover=False)
        handle_parsing_error(ValueError("Error 2"), auto_recover=False)
        
        stats = get_error_statistics()
        assert stats["total_errors"] == 2

    def test_reset_global_statistics(self):
        """Test resetting global statistics."""
        handle_parsing_error(ValueError("Error"), auto_recover=False)
        
        # Verify error exists
        stats = get_error_statistics()
        assert stats["total_errors"] == 1
        
        # Reset and verify
        reset_error_statistics()
        stats = get_error_statistics()
        assert stats["total_errors"] == 0

    def test_error_with_context(self):
        """Test error handling with context information."""
        context = {
            "parser_type": "text",
            "file_size": 1024,
            "operation": "encoding_detection"
        }
        
        parsing_error = handle_parsing_error(
            UnicodeDecodeError("utf-8", b"", 0, 1, "test"),
            file_path="test.txt",
            context=context,
            auto_recover=False
        )
        
        assert parsing_error.context["parser_type"] == "text"
        assert parsing_error.context["file_size"] == 1024
        assert parsing_error.context["operation"] == "encoding_detection"


class TestErrorIntegration:
    """Integration tests for error handling system."""

    def test_error_chain_preservation(self):
        """Test that original exception chain is preserved."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            try:
                raise RuntimeError("Wrapper error") from e
            except RuntimeError as wrapper:
                parsing_error = handle_parsing_error(wrapper, auto_recover=False)
                
                assert parsing_error.original_exception == wrapper
                assert str(parsing_error.original_exception) == "Wrapper error"

    def test_multiple_error_handling(self):
        """Test handling multiple errors in sequence."""
        errors = [
            FileNotFoundError("File 1"),
            PermissionError("File 2"),
            UnicodeDecodeError("utf-8", b"", 0, 1, "File 3"),
            MemoryError("File 4"),
            TimeoutError("File 5")
        ]
        
        parsing_errors = []
        for error in errors:
            parsing_error = handle_parsing_error(error, auto_recover=False)
            parsing_errors.append(parsing_error)
        
        # Verify different error types were created
        error_types = {type(pe).__name__ for pe in parsing_errors}
        expected_types = {"FileAccessError", "EncodingError", "MemoryError", "ParsingTimeout"}
        assert expected_types.issubset(error_types)
        
        # Verify statistics
        stats = get_error_statistics()
        assert stats["total_errors"] == len(errors)

    def test_pathlib_path_handling(self, tmp_path):
        """Test error handling with pathlib.Path objects."""
        path_obj = tmp_path / "nonexistent.txt"
        
        parsing_error = handle_parsing_error(
            FileNotFoundError("Not found"),
            file_path=path_obj,
            auto_recover=False
        )
        
        assert parsing_error.file_path == str(path_obj)