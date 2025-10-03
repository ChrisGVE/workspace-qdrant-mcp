"""
Error Categorization System

Provides automatic error categorization based on exception types, error messages,
and contextual information. Supports severity levels, confidence scoring, and
manual overrides.

Example usage:
    >>> categorizer = ErrorCategorizer()
    >>> category, severity, confidence = categorizer.categorize(
    ...     exception=FileNotFoundError("file.txt not found"),
    ...     message="Failed to read file.txt"
    ... )
    >>> print(f"{category.value}: {severity.value} (confidence: {confidence})")
    file_corrupt: error (confidence: 1.0)

    >>> # Message-based categorization
    >>> cat, sev, conf = categorizer.categorize(
    ...     message="Connection timeout after 30 seconds"
    ... )
    >>> print(f"{cat.value}: {sev.value}")
    timeout: error

    >>> # Context-based categorization
    >>> cat, sev, conf = categorizer.categorize(
    ...     message="LSP server not available",
    ...     context={'tool_name': 'rust-analyzer'}
    ... )
    >>> print(f"{cat.value}: {sev.value}")
    tool_missing: error
"""

from enum import Enum
from typing import Optional, Tuple, Dict, Any, Type, List
import socket
import asyncio
import re


class ErrorSeverity(Enum):
    """
    Error severity levels.

    Maps to database schema values ('error', 'warning', 'info').
    """
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

    @classmethod
    def from_string(cls, value: str) -> "ErrorSeverity":
        """
        Convert string to ErrorSeverity enum.

        Args:
            value: String value ('error', 'warning', 'info')

        Returns:
            ErrorSeverity enum value

        Raises:
            ValueError: If value is not a valid severity level
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid severity: {value}. "
                f"Valid values: {', '.join([s.value for s in cls])}"
            )


class ErrorCategory(Enum):
    """
    Error categories for classification.

    Maps to database schema category values. Each category represents
    a specific type of failure that requires different handling.
    """
    # File and IO errors
    FILE_CORRUPT = "file_corrupt"

    # Tool and dependency errors
    TOOL_MISSING = "tool_missing"

    # Network and connectivity errors
    NETWORK = "network"

    # Data validation errors
    METADATA_INVALID = "metadata_invalid"

    # Processing failures
    PROCESSING_FAILED = "processing_failed"

    # Parsing errors
    PARSE_ERROR = "parse_error"

    # Permission and access errors
    PERMISSION_DENIED = "permission_denied"

    # Resource exhaustion
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # Timeout errors
    TIMEOUT = "timeout"

    # Catch-all for unknown errors
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "ErrorCategory":
        """
        Convert string to ErrorCategory enum.

        Args:
            value: String value (e.g., 'file_corrupt', 'tool_missing')

        Returns:
            ErrorCategory enum value

        Raises:
            ValueError: If value is not a valid category
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid category: {value}. "
                f"Valid values: {', '.join([c.value for c in cls])}"
            )


class ErrorCategorizer:
    """
    Automatic error categorization engine.

    Categorizes errors based on:
    1. Exception type
    2. Error message content
    3. Contextual information

    Returns category, severity, and confidence score.
    """

    # Exception type to (category, severity) mapping
    # Confidence is 1.0 for direct exception type matches
    EXCEPTION_TYPE_MAP: Dict[Type[Exception], Tuple[ErrorCategory, ErrorSeverity]] = {
        # File and IO errors
        FileNotFoundError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),
        FileExistsError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),
        IsADirectoryError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),
        NotADirectoryError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),
        IOError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),
        OSError: (ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR),

        # Permission errors
        PermissionError: (ErrorCategory.PERMISSION_DENIED, ErrorSeverity.ERROR),

        # Network errors
        ConnectionError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        ConnectionRefusedError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        ConnectionAbortedError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        ConnectionResetError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        BrokenPipeError: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        socket.gaierror: (ErrorCategory.NETWORK, ErrorSeverity.ERROR),
        socket.timeout: (ErrorCategory.TIMEOUT, ErrorSeverity.ERROR),

        # Timeout errors
        TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.ERROR),
        asyncio.TimeoutError: (ErrorCategory.TIMEOUT, ErrorSeverity.ERROR),

        # Resource exhaustion
        MemoryError: (ErrorCategory.RESOURCE_EXHAUSTED, ErrorSeverity.ERROR),
        BlockingIOError: (ErrorCategory.RESOURCE_EXHAUSTED, ErrorSeverity.WARNING),

        # Parsing errors
        ValueError: (ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR),
        SyntaxError: (ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR),
        TypeError: (ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR),

        # Module/tool errors
        ModuleNotFoundError: (ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR),
        ImportError: (ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR),
    }

    # Keyword patterns for message-based categorization
    # Order matters - more specific patterns should come first
    # Each pattern maps to (category, severity, confidence_multiplier)
    MESSAGE_PATTERNS: List[Tuple[re.Pattern, ErrorCategory, ErrorSeverity, float]] = [
        # Timeout patterns (check before network patterns)
        (re.compile(r'\btimeout\b', re.IGNORECASE), ErrorCategory.TIMEOUT, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\btimed out\b', re.IGNORECASE), ErrorCategory.TIMEOUT, ErrorSeverity.ERROR, 0.9),

        # Network patterns
        (re.compile(r'\b(network|connection|socket)\b', re.IGNORECASE), ErrorCategory.NETWORK, ErrorSeverity.ERROR, 0.8),
        (re.compile(r'\bconnect(ed|ing)?\b', re.IGNORECASE), ErrorCategory.NETWORK, ErrorSeverity.ERROR, 0.7),
        (re.compile(r'\bdisconnect(ed)?\b', re.IGNORECASE), ErrorCategory.NETWORK, ErrorSeverity.ERROR, 0.7),

        # File and IO patterns
        (re.compile(r'\bfile (not found|missing|does not exist)\b', re.IGNORECASE), ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\b(file|directory) corrupt(ed)?\b', re.IGNORECASE), ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\b(read|write|open) (file|failed)\b', re.IGNORECASE), ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR, 0.7),

        # Permission patterns
        (re.compile(r'\b(permission|access) (denied|forbidden)\b', re.IGNORECASE), ErrorCategory.PERMISSION_DENIED, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\bunauthorized\b', re.IGNORECASE), ErrorCategory.PERMISSION_DENIED, ErrorSeverity.ERROR, 0.8),

        # Parsing patterns
        (re.compile(r'\b(parse|syntax) error\b', re.IGNORECASE), ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\binvalid (json|yaml|xml|format)\b', re.IGNORECASE), ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR, 0.8),
        (re.compile(r'\bmalformed\b', re.IGNORECASE), ErrorCategory.PARSE_ERROR, ErrorSeverity.ERROR, 0.7),

        # Tool/dependency patterns
        (re.compile(r'\b(module|package|library) not found\b', re.IGNORECASE), ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\b(missing|unavailable) (tool|command|lsp|binary)\b', re.IGNORECASE), ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\blsp (server )?(not available|failed|missing)\b', re.IGNORECASE), ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR, 0.9),

        # Resource exhaustion patterns
        (re.compile(r'\bout of (memory|disk space)\b', re.IGNORECASE), ErrorCategory.RESOURCE_EXHAUSTED, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\b(memory|resource) (exhausted|exceeded)\b', re.IGNORECASE), ErrorCategory.RESOURCE_EXHAUSTED, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\btoo many (files|connections|requests)\b', re.IGNORECASE), ErrorCategory.RESOURCE_EXHAUSTED, ErrorSeverity.ERROR, 0.8),

        # Metadata patterns
        (re.compile(r'\b(invalid|missing) metadata\b', re.IGNORECASE), ErrorCategory.METADATA_INVALID, ErrorSeverity.ERROR, 0.9),
        (re.compile(r'\bmetadata (validation|error)\b', re.IGNORECASE), ErrorCategory.METADATA_INVALID, ErrorSeverity.ERROR, 0.8),

        # Processing patterns
        (re.compile(r'\bprocessing (failed|error)\b', re.IGNORECASE), ErrorCategory.PROCESSING_FAILED, ErrorSeverity.ERROR, 0.8),
        (re.compile(r'\bfailed to process\b', re.IGNORECASE), ErrorCategory.PROCESSING_FAILED, ErrorSeverity.ERROR, 0.7),
    ]

    def __init__(self):
        """Initialize the error categorizer."""
        pass

    def categorize(
        self,
        exception: Optional[Exception] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        manual_category: Optional[ErrorCategory] = None,
        manual_severity: Optional[ErrorSeverity] = None,
    ) -> Tuple[ErrorCategory, ErrorSeverity, float]:
        """
        Categorize an error and determine its severity.

        Args:
            exception: The exception object (if available)
            message: Error message string
            context: Additional context (e.g., {'file_path': '/path/to/file', 'tool_name': 'rust-analyzer'})
            manual_category: Manual category override
            manual_severity: Manual severity override

        Returns:
            Tuple of (category, severity, confidence_score)
            where confidence_score is 0.0-1.0

        Example:
            >>> categorizer = ErrorCategorizer()
            >>> cat, sev, conf = categorizer.categorize(
            ...     exception=PermissionError("Access denied"),
            ...     message="Cannot write to /etc/file"
            ... )
            >>> print(f"{cat.value}: {sev.value}")
            permission_denied: error
        """
        # Manual overrides take precedence
        if manual_category and manual_severity:
            return manual_category, manual_severity, 1.0

        # Collect all signals and their confidences
        signals: List[Tuple[ErrorCategory, ErrorSeverity, float]] = []

        # 1. Try exception type categorization
        if exception is not None:
            result = self._categorize_by_exception_type(exception)
            if result:
                signals.append(result)

        # 2. Try message-based categorization
        if message:
            result = self._categorize_by_message(message)
            if result:
                signals.append(result)

        # 3. Try context-based categorization
        if context:
            result = self._categorize_by_context(context, message)
            if result:
                signals.append(result)

        # 4. Combine signals or use default
        if signals:
            # Use highest confidence signal
            category, severity, confidence = max(signals, key=lambda x: x[2])

            # Apply manual overrides if provided
            if manual_category:
                category = manual_category
            if manual_severity:
                severity = manual_severity

            return category, severity, confidence

        # Fallback to defaults
        category = manual_category or ErrorCategory.UNKNOWN
        severity = manual_severity or ErrorSeverity.ERROR
        confidence = 1.0 if manual_category else 0.0

        return category, severity, confidence

    def _categorize_by_exception_type(
        self,
        exception: Exception
    ) -> Optional[Tuple[ErrorCategory, ErrorSeverity, float]]:
        """
        Categorize error based on exception type.

        Args:
            exception: The exception to categorize

        Returns:
            Tuple of (category, severity, confidence) or None if not found
        """
        exc_type = type(exception)

        # Direct match
        if exc_type in self.EXCEPTION_TYPE_MAP:
            category, severity = self.EXCEPTION_TYPE_MAP[exc_type]
            return category, severity, 1.0

        # Check inheritance chain (less confident for base classes)
        for mapped_type, (category, severity) in self.EXCEPTION_TYPE_MAP.items():
            if isinstance(exception, mapped_type):
                # Lower confidence for inherited matches
                return category, severity, 0.8

        return None

    def _categorize_by_message(
        self,
        message: str
    ) -> Optional[Tuple[ErrorCategory, ErrorSeverity, float]]:
        """
        Categorize error based on message content.

        Args:
            message: Error message string

        Returns:
            Tuple of (category, severity, confidence) or None if no match
        """
        if not message:
            return None

        # Try all patterns and return the first match
        for pattern, category, severity, confidence in self.MESSAGE_PATTERNS:
            if pattern.search(message):
                return category, severity, confidence

        return None

    def _categorize_by_context(
        self,
        context: Dict[str, Any],
        message: Optional[str] = None
    ) -> Optional[Tuple[ErrorCategory, ErrorSeverity, float]]:
        """
        Categorize error based on contextual information.

        Args:
            context: Context dictionary with keys like 'tool_name', 'file_path', etc.
            message: Optional message for additional hints

        Returns:
            Tuple of (category, severity, confidence) or None if cannot categorize
        """
        if not context:
            return None

        # Check for tool-related context
        if 'tool_name' in context or 'lsp_server' in context:
            return ErrorCategory.TOOL_MISSING, ErrorSeverity.ERROR, 0.7

        # Check for file-related context with certain keywords
        if 'file_path' in context and message:
            message_lower = message.lower()
            if 'not found' in message_lower or 'missing' in message_lower:
                return ErrorCategory.FILE_CORRUPT, ErrorSeverity.ERROR, 0.7
            elif 'permission' in message_lower or 'access' in message_lower:
                return ErrorCategory.PERMISSION_DENIED, ErrorSeverity.ERROR, 0.7

        # Check for metadata-related context
        if 'metadata' in context or 'tenant_id' in context:
            return ErrorCategory.METADATA_INVALID, ErrorSeverity.ERROR, 0.6

        return None
