"""
Document parsers for batch ingestion.

This module provides a set of document parsers that can extract text content
from various file formats for ingestion into the workspace-qdrant-mcp system.

Supported formats:
    - Plain text (.txt)
    - Markdown (.md, .markdown)
    - PDF (.pdf)
    - Microsoft Word (.docx) - optional
    - Microsoft PowerPoint (.pptx) - optional
    - HTML (.html, .htm, .xhtml) - optional
    - Web content (http, https) - with security hardening

Each parser implements the DocumentParser interface and provides:
    - Format detection and validation
    - Text extraction with metadata
    - Error handling for corrupted files
    - Content preprocessing and cleaning
"""

from .base import DocumentParser, ParsedDocument
from .exceptions import (
    ParsingError, FileAccessError, FileFormatError, FileCorruptionError,
    EncodingError, MemoryError, ValidationError, ParsingTimeout, SystemError,
    ErrorHandler, handle_parsing_error
)
from .file_detector import (
    FileDetector, FileTypeDetectionError, UnsupportedFileTypeError,
    detect_file_type, is_supported_file, get_supported_extensions, get_supported_mime_types
)
from .progress import (
    ProgressTracker, BatchProgressTracker, ProgressPhase, ProgressUnit,
    ProgressCallback, ConsoleProgressCallback, LoggingProgressCallback,
    create_progress_tracker, create_batch_progress_tracker
)
from .html_parser import HtmlParser
from .markdown_parser import MarkdownParser
from .pdf_parser import PDFParser
from .pptx_parser import PptxParser
from .text_parser import TextParser
from .docx_parser import DocxParser
from .epub_parser import EpubParser
from .code_parser import CodeParser
from .mobi_parser import MobiParser
from .web_parser import WebParser, WebIngestionInterface, create_secure_web_parser
from .web_crawler import SecureWebCrawler, SecurityConfig, CrawlResult

__all__ = [
    # Base classes
    "DocumentParser",
    "ParsedDocument",
    # Parser implementations
    "TextParser",
    "MarkdownParser",
    "PDFParser",
    "PptxParser",
    "HtmlParser",
    "DocxParser",
    "EpubParser",
    "CodeParser",
    "MobiParser",
    "WebParser",
    "WebIngestionInterface",
    "SecureWebCrawler",
    "SecurityConfig",
    "CrawlResult",
    "create_secure_web_parser",
    # Error handling
    "ParsingError",
    "FileAccessError",
    "FileFormatError",
    "FileCorruptionError",
    "EncodingError",
    "MemoryError",
    "ValidationError",
    "ParsingTimeout",
    "SystemError",
    "ErrorHandler",
    "handle_parsing_error",
    # File detection
    "FileDetector",
    "FileTypeDetectionError",
    "UnsupportedFileTypeError",
    "detect_file_type",
    "is_supported_file",
    "get_supported_extensions",
    "get_supported_mime_types",
    # Progress tracking
    "ProgressTracker",
    "BatchProgressTracker",
    "ProgressPhase",
    "ProgressUnit",
    "ProgressCallback",
    "ConsoleProgressCallback",
    "LoggingProgressCallback",
    "create_progress_tracker",
    "create_batch_progress_tracker",
]
