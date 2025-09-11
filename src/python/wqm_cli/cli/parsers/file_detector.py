from common.observability import get_logger

logger = get_logger(__name__)
"""
File type detection and validation system.

This module provides comprehensive file type detection using multiple methods:
- MIME type detection using python-magic
- Magic number checking for binary file format validation  
- Extension-based fallback detection
- Content analysis for text files

Supports all implemented parsers in the workspace-qdrant-mcp system.
"""

import logging
import mimetypes
from pathlib import Path
from typing import Optional, Union

try:
    import magic

    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

logger = logging.getLogger(__name__)

# Magic number signatures for binary formats
MAGIC_NUMBERS = {
    # PDF format
    b"%PDF-": "application/pdf",
    # Microsoft Office formats (ZIP-based)
    b"PK\x03\x04": "application/zip",  # Could be DOCX/PPTX/EPUB
    # EPUB specific
    b"PK\x03\x04\x14\x00\x06\x00": "application/epub+zip",
    # HTML variants
    b"<!DOCTYPE": "text/html",
    b"<html": "text/html",
    b"<HTML": "text/html",
    # XML
    b"<?xml": "text/xml",
    # Plain text indicators (BOM markers)
    b"\xef\xbb\xbf": "text/plain",  # UTF-8 BOM
    b"\xff\xfe": "text/plain",  # UTF-16 LE BOM
    b"\xfe\xff": "text/plain",  # UTF-16 BE BOM
}

# File extension to MIME type mapping
EXTENSION_MIME_MAP = {
    # Text formats
    ".txt": "text/plain",
    ".text": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".rst": "text/x-rst",
    # Document formats
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".epub": "application/epub+zip",
    # Web formats
    ".html": "text/html",
    ".htm": "text/html",
    ".xhtml": "application/xhtml+xml",
    # Structured data
    ".xml": "text/xml",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".csv": "text/csv",
    # Code files
    ".py": "text/x-python",
    ".js": "application/javascript",
    ".css": "text/css",
    ".sql": "text/plain",
    ".sh": "application/x-sh",
    ".bash": "application/x-sh",
    # Log files
    ".log": "text/plain",
}

# Supported parser types mapping
MIME_TO_PARSER = {
    "text/plain": "text",
    "text/markdown": "markdown",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/epub+zip": "epub",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/x-python": "code",
    "application/javascript": "code",
    "text/css": "code",
    "application/x-sh": "code",
    "application/json": "text",
    "text/yaml": "text",
    "text/csv": "text",
    "text/xml": "text",
}


class FileTypeDetectionError(Exception):
    """Exception raised when file type detection fails."""

    pass


class UnsupportedFileTypeError(Exception):
    """Exception raised when file type is not supported by any parser."""

    pass


class FileDetector:
    """
    Comprehensive file type detection system.

    Uses multiple detection methods in order of reliability:
    1. MIME type detection using python-magic (if available)
    2. Magic number checking for binary formats
    3. Extension-based fallback detection
    4. Content analysis for ambiguous cases
    """

    def __init__(self, enable_magic: bool = True):
        """
        Initialize file detector.

        Args:
            enable_magic: Whether to use python-magic for MIME detection
        """
        self.enable_magic = enable_magic and HAS_MAGIC

        if enable_magic and not HAS_MAGIC:
            logger.warning(
                "python-magic not available, falling back to extension-based detection"
            )

    def detect_file_type(self, file_path: Union[str, Path]) -> tuple[str, str, float]:
        """
        Detect file type and determine appropriate parser.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Tuple of (mime_type, parser_type, confidence_score)

        Raises:
            FileNotFoundError: If file doesn't exist
            FileTypeDetectionError: If detection fails
            UnsupportedFileTypeError: If no parser supports the file type
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise FileTypeDetectionError(f"Path is not a file: {file_path}")

        # Try multiple detection methods
        mime_type, confidence = self._detect_mime_type(file_path)

        # Determine parser type
        parser_type = self._get_parser_type(mime_type, file_path)

        if not parser_type:
            raise UnsupportedFileTypeError(
                f"No parser available for file type: {mime_type} ({file_path.suffix})"
            )

        logger.debug(
            f"Detected file type: {mime_type} -> {parser_type} "
            f"(confidence: {confidence:.2f}) for {file_path}"
        )

        return mime_type, parser_type, confidence

    def _detect_mime_type(self, file_path: Path) -> tuple[str, float]:
        """
        Detect MIME type using multiple methods.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (mime_type, confidence_score)
        """
        # Method 1: python-magic (most reliable)
        if self.enable_magic:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                if mime_type and mime_type != "application/octet-stream":
                    return mime_type, 0.9
            except Exception as e:
                logger.debug(f"Magic detection failed for {file_path}: {e}")

        # Method 2: Magic number checking
        magic_mime, magic_confidence = self._check_magic_numbers(file_path)
        if magic_mime:
            return magic_mime, magic_confidence

        # Method 3: Built-in mimetypes module
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type, 0.7

        # Method 4: Extension-based fallback
        extension = file_path.suffix.lower()
        if extension in EXTENSION_MIME_MAP:
            return EXTENSION_MIME_MAP[extension], 0.6

        # Method 5: Content analysis for text files
        if self._is_text_file(file_path):
            return "text/plain", 0.5

        # Final fallback
        return "application/octet-stream", 0.1

    def _check_magic_numbers(self, file_path: Path) -> tuple[Optional[str], float]:
        """
        Check file magic numbers for format identification.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (mime_type, confidence_score) or (None, 0.0)
        """
        try:
            with open(file_path, "rb") as f:
                header = f.read(64)  # Read first 64 bytes

            # Check for specific magic numbers
            for magic_bytes, mime_type in MAGIC_NUMBERS.items():
                if header.startswith(magic_bytes):
                    # Special handling for ZIP-based formats
                    if mime_type == "application/zip":
                        return self._detect_zip_based_format(file_path), 0.8
                    return mime_type, 0.8

            return None, 0.0

        except Exception as e:
            logger.debug(f"Magic number check failed for {file_path}: {e}")
            return None, 0.0

    def _detect_zip_based_format(self, file_path: Path) -> str:
        """
        Detect specific format for ZIP-based files (DOCX, PPTX, EPUB).

        Args:
            file_path: Path to the file

        Returns:
            Specific MIME type for the format
        """
        import zipfile

        try:
            with zipfile.ZipFile(file_path, "r") as zip_file:
                file_list = zip_file.namelist()

                # Check for EPUB
                if "META-INF/container.xml" in file_list:
                    return "application/epub+zip"

                # Check for DOCX
                if "word/document.xml" in file_list:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

                # Check for PPTX
                if "ppt/presentation.xml" in file_list:
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"

        except Exception as e:
            logger.debug(f"ZIP format detection failed for {file_path}: {e}")

        # Fallback based on extension
        extension = file_path.suffix.lower()
        if extension == ".docx":
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif extension == ".pptx":
            return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif extension == ".epub":
            return "application/epub+zip"

        return "application/zip"

    def _is_text_file(self, file_path: Path) -> bool:
        """
        Determine if file is likely a text file through content analysis.

        Args:
            file_path: Path to the file

        Returns:
            True if file appears to be text
        """
        try:
            # Read a sample of the file
            with open(file_path, "rb") as f:
                sample = f.read(1024)  # Read first 1KB

            # Check for binary content indicators
            if b"\x00" in sample:  # Null bytes indicate binary
                return False

            # Check for high percentage of printable characters
            printable_chars = sum(
                1 for byte in sample if 32 <= byte <= 126 or byte in (9, 10, 13)
            )
            text_ratio = printable_chars / len(sample) if sample else 0

            return text_ratio > 0.75

        except Exception:
            return False

    def _get_parser_type(self, mime_type: str, file_path: Path) -> Optional[str]:
        """
        Determine appropriate parser type for MIME type.

        Args:
            mime_type: Detected MIME type
            file_path: Path to the file (for extension fallback)

        Returns:
            Parser type string or None if unsupported
        """
        # Direct MIME type mapping
        if mime_type in MIME_TO_PARSER:
            return MIME_TO_PARSER[mime_type]

        # Handle generic types with extension hints
        if mime_type == "text/plain":
            extension = file_path.suffix.lower()
            if extension in [".py", ".js", ".css", ".sh", ".bash", ".sql"]:
                return "code"
            elif extension in [".md", ".markdown"]:
                return "markdown"
            elif extension in [".html", ".htm"]:
                return "html"
            return "text"

        # Handle application/octet-stream with extension fallback
        if mime_type == "application/octet-stream":
            extension = file_path.suffix.lower()
            if extension in EXTENSION_MIME_MAP:
                fallback_mime = EXTENSION_MIME_MAP[extension]
                return MIME_TO_PARSER.get(fallback_mime)

        return None

    def get_supported_extensions(self) -> list[str]:
        """
        Get list of all supported file extensions.

        Returns:
            List of supported extensions including the dot
        """
        return sorted(EXTENSION_MIME_MAP.keys())

    def get_supported_mime_types(self) -> list[str]:
        """
        Get list of all supported MIME types.

        Returns:
            List of supported MIME types
        """
        return sorted(MIME_TO_PARSER.keys())

    def is_supported_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if file is supported by any parser.

        Args:
            file_path: Path to the file

        Returns:
            True if file is supported
        """
        try:
            _, parser_type, _ = self.detect_file_type(file_path)
            return parser_type is not None
        except (FileTypeDetectionError, UnsupportedFileTypeError):
            return False


# Module-level convenience functions
_default_detector = FileDetector()


def detect_file_type(file_path: Union[str, Path]) -> tuple[str, str, float]:
    """
    Detect file type using default detector instance.

    Args:
        file_path: Path to the file

    Returns:
        Tuple of (mime_type, parser_type, confidence_score)
    """
    return _default_detector.detect_file_type(file_path)


def is_supported_file(file_path: Union[str, Path]) -> bool:
    """
    Check if file is supported using default detector instance.

    Args:
        file_path: Path to the file

    Returns:
        True if file is supported
    """
    return _default_detector.is_supported_file(file_path)


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    return _default_detector.get_supported_extensions()


def get_supported_mime_types() -> list[str]:
    """Get list of all supported MIME types."""
    return _default_detector.get_supported_mime_types()
