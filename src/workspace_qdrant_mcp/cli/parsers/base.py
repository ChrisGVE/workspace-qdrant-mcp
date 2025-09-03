from ...observability import get_logger

logger = get_logger(__name__)
"""
Base document parser interface and data structures.

This module defines the common interface and data structures used by all
document parsers in the workspace-qdrant-mcp ingestion system. It provides
a consistent API for extracting text content and metadata from various
file formats.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from .exceptions import ParsingError, handle_parsing_error
from .file_detector import FileDetector, detect_file_type
from .progress import (
    ProgressPhase,
    ProgressTracker,
    ProgressUnit,
    create_progress_tracker,
)


@dataclass
class ParsedDocument:
    """
    Container for parsed document content and metadata.

    Represents the result of parsing a document file, containing the extracted
    text content along with comprehensive metadata for indexing and retrieval.

    Attributes:
        content: Extracted text content from the document
        file_path: Original file path (for reference and deduplication)
        file_type: Document format identifier (e.g., 'pdf', 'markdown', 'text')
        metadata: Additional metadata dictionary with file-specific information
        content_hash: SHA256 hash of the content for deduplication
        parsed_at: Timestamp when parsing was completed
        file_size: Size of the original file in bytes
        parsing_info: Information about the parsing process (optional)
    """

    content: str
    file_path: str
    file_type: str
    metadata: dict[str, str | int | float | bool]
    content_hash: str
    parsed_at: str
    file_size: int
    parsing_info: dict[str, str | int | float] | None = None

    @classmethod
    def create(
        cls,
        content: str,
        file_path: str | Path,
        file_type: str,
        additional_metadata: dict[str, str | int | float | bool] | None = None,
        parsing_info: dict[str, str | int | float] | None = None,
    ) -> "ParsedDocument":
        """
        Create a ParsedDocument with auto-generated metadata.

        Args:
            content: Extracted text content
            file_path: Path to the original file
            file_type: Format identifier (e.g., 'pdf', 'markdown')
            additional_metadata: Extra metadata to include
            parsing_info: Information about the parsing process

        Returns:
            ParsedDocument instance with complete metadata
        """
        file_path = Path(file_path)

        # Generate content hash for deduplication
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Base metadata with explicit typing
        metadata: dict[str, str | int | float | bool] = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "content_length": len(content),
            "line_count": content.count("\n") + 1 if content else 0,
        }

        # Add file stats if file exists
        try:
            stat = file_path.stat()
            metadata.update(
                {
                    "file_size": stat.st_size,
                    "file_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "file_created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )
            file_size = stat.st_size
        except (OSError, FileNotFoundError):
            file_size = len(content.encode("utf-8"))  # Approximate

        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)

        return cls(
            content=content,
            file_path=str(file_path),
            file_type=file_type,
            metadata=metadata,
            content_hash=content_hash,
            parsed_at=datetime.utcnow().isoformat(),
            file_size=file_size,
            parsing_info=parsing_info,
        )


class DocumentParser(ABC):
    """
    Abstract base class for document format parsers.

    Defines the interface that all document parsers must implement to provide
    consistent text extraction capabilities across different file formats.

    Each parser should:
        1. Validate that it can handle the given file format
        2. Extract text content efficiently and accurately
        3. Generate relevant metadata for the document
        4. Handle errors gracefully with informative messages
        5. Provide format-specific parsing options when needed
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        List of file extensions this parser can handle.

        Returns:
            List of file extensions (including the dot, e.g., ['.txt', '.md'])
        """
        pass

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Human-readable name of the format this parser handles.

        Returns:
            Format name string (e.g., 'Plain Text', 'PDF Document')
        """
        pass

    def can_parse(self, file_path: str | Path) -> bool:
        """
        Check if this parser can handle the given file.

        Uses both extension checking and file type detection for accuracy.

        Args:
            file_path: Path to the file to check

        Returns:
            True if this parser can handle the file format
        """
        file_path = Path(file_path)

        # First check extension (fast)
        if file_path.suffix.lower() in self.supported_extensions:
            return True

        # If extension check fails, try file type detection
        try:
            _, parser_type, _ = detect_file_type(file_path)
            # Check if detected parser type matches this parser's format
            return self._matches_parser_type(parser_type)
        except Exception:
            return False

    def _matches_parser_type(self, parser_type: str) -> bool:
        """
        Check if detected parser type matches this parser.

        Args:
            parser_type: Parser type from file detection

        Returns:
            True if this parser can handle the detected type
        """
        # Default implementation based on format name
        format_name_lower = self.format_name.lower()
        if "text" in format_name_lower:
            return parser_type in ["text", "code"]
        elif "pdf" in format_name_lower:
            return parser_type == "pdf"
        elif "markdown" in format_name_lower:
            return parser_type == "markdown"
        elif "html" in format_name_lower:
            return parser_type == "html"
        elif "docx" in format_name_lower:
            return parser_type == "docx"
        elif "pptx" in format_name_lower:
            return parser_type == "pptx"
        elif "epub" in format_name_lower:
            return parser_type == "epub"
        else:
            return False

    @abstractmethod
    async def parse(
        self,
        file_path: str | Path,
        progress_tracker: Optional[ProgressTracker] = None,
        **options: Any,
    ) -> ParsedDocument:
        """
        Parse a document file and extract its text content.

        Args:
            file_path: Path to the file to parse
            progress_tracker: Optional progress tracker for monitoring
            **options: Parser-specific options

        Returns:
            ParsedDocument with extracted content and metadata

        Raises:
            ParsingError: If parsing fails (wraps specific error types)
        """
        pass

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """
        Get available parsing options for this format.

        Returns:
            Dictionary of option names mapped to their configuration:
            {
                "option_name": {
                    "type": type,
                    "default": default_value,
                    "description": "Description of what this option does"
                }
            }
        """
        return {}

    def validate_file(self, file_path: str | Path) -> None:
        """
        Validate that a file can be parsed.

        Args:
            file_path: Path to the file to validate

        Raises:
            ParsingError: If file cannot be parsed (wraps specific error types)
        """
        file_path = Path(file_path)

        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            if not file_path.is_file():
                raise ValueError(f"Path is not a file: {file_path}")

            if not self.can_parse(file_path):
                raise ValueError(
                    f"File format not supported by {self.format_name} parser: {file_path.suffix}"
                )
        except Exception as e:
            # Convert to ParsingError for consistent error handling
            raise handle_parsing_error(e, file_path)
