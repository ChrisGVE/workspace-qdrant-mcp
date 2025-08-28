"""
Document parsers for batch ingestion.

This module provides a set of document parsers that can extract text content
from various file formats for ingestion into the workspace-qdrant-mcp system.

Supported formats:
    - Plain text (.txt)
    - Markdown (.md, .markdown)
    - PDF (.pdf)
    - Microsoft Word (.docx) - optional

Each parser implements the DocumentParser interface and provides:
    - Format detection and validation
    - Text extraction with metadata
    - Error handling for corrupted files
    - Content preprocessing and cleaning
"""

from .base import DocumentParser, ParsedDocument
from .markdown_parser import MarkdownParser
from .pdf_parser import PDFParser
from .text_parser import TextParser

__all__ = [
    "DocumentParser",
    "ParsedDocument",
    "TextParser",
    "MarkdownParser",
    "PDFParser",
]
