
from ...observability import get_logger
logger = get_logger(__name__)
"""
PDF document parser.

This parser handles PDF files using PyPDF2 for text extraction with support
for metadata extraction, multi-page processing, and content analysis.
Provides fallback handling for encrypted or corrupted PDFs.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

try:
    import pypdf

    # Alias for backward compatibility with tests that expect PyPDF2
    PyPDF2 = pypdf
    HAS_PYPDF = True
    HAS_PYPDF2 = True  # For backward compatibility with tests
except ImportError:
    # Create a dummy PyPDF2 for test compatibility
    class PyPDF2:
        pass

    HAS_PYPDF = False
    HAS_PYPDF2 = False  # For backward compatibility with tests

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class PDFParser(DocumentParser):
    """
    Parser for PDF documents.

    Handles PDF files with support for:
        - Multi-page text extraction
        - PDF metadata extraction (title, author, creation date, etc.)
        - Handling of encrypted PDFs (if possible)
        - Page-by-page processing for large documents
        - Content analysis and statistics

    Uses PyPDF2 or pypdf for PDF processing. Falls back gracefully when
    these libraries are not available.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Supported PDF file extensions."""
        return [".pdf"]

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "PDF Document"

    def can_parse(self, file_path: str | Path) -> bool:
        """Check if this parser can handle the given file."""
        if not HAS_PYPDF2:
            logger.warning("pypdf not available, PDF parsing disabled")
            return False
        return super().can_parse(file_path)

    async def parse(
        self,
        file_path: str | Path,
        extract_metadata: bool = True,
        include_page_numbers: bool = False,
        max_pages: int | None = None,
        password: str | None = None,
        **options,
    ) -> ParsedDocument:
        """
        Parse a PDF file.

        Args:
            file_path: Path to the PDF file
            extract_metadata: Whether to extract PDF metadata
            include_page_numbers: Whether to include page numbers in content
            max_pages: Maximum number of pages to process (None for all)
            password: Password for encrypted PDFs
            **options: Additional parsing options

        Returns:
            ParsedDocument with extracted text content and metadata

        Raises:
            ImportError: If PyPDF2/pypdf is not installed
            RuntimeError: If PDF parsing fails
        """
        if not HAS_PYPDF2:
            raise ImportError(
                "PDF parsing requires pypdf. Install with: pip install pypdf"
            )

        file_path = Path(file_path)
        self.validate_file(file_path)

        parsing_info: dict[str, str | int | float] = {}

        try:
            with open(file_path, "rb") as pdf_file:
                # Create PDF reader
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Handle encrypted PDFs
                if pdf_reader.is_encrypted:
                    if password:
                        if not pdf_reader.decrypt(password):
                            raise RuntimeError("Invalid password for encrypted PDF")
                        parsing_info["encrypted"] = True
                        parsing_info["decrypted"] = True
                    else:
                        raise RuntimeError("PDF is encrypted but no password provided")

                # Get basic PDF info
                num_pages: int = len(pdf_reader.pages)
                parsing_info["total_pages"] = num_pages
                parsing_info["pages_processed"] = (
                    min(num_pages, max_pages) if max_pages else num_pages
                )

                # Extract PDF metadata
                pdf_metadata: dict[str, str | int | float | bool] = {}
                if extract_metadata:
                    pdf_metadata = await self._extract_pdf_metadata(pdf_reader)
                    parsing_info["has_metadata"] = bool(pdf_metadata)

                # Extract text content
                content_parts = []
                pages_processed = 0

                for page_num, page in enumerate(pdf_reader.pages):
                    if max_pages and page_num >= max_pages:
                        break

                    try:
                        page_text = page.extract_text()

                        if page_text.strip():  # Only include non-empty pages
                            if include_page_numbers:
                                content_parts.append(f"\n--- Page {page_num + 1} ---\n")
                                content_parts.append(page_text)
                            else:
                                content_parts.append(page_text)

                        pages_processed += 1

                    except Exception as e:
                        logger.warning(
                            f"Failed to extract text from page {page_num + 1}: {e}"
                        )
                        parsing_info[f"page_{page_num + 1}_error"] = str(e)
                        continue

                # Combine all content
                content = "\n\n".join(content_parts).strip()

                # Clean and normalize content
                content = self._clean_pdf_text(content)

                # Generate text analysis
                text_stats = self._analyze_pdf_content(content, pages_processed)
                parsing_info.update(text_stats)

                # Create comprehensive metadata
                additional_metadata: dict[str, str | int | float | bool] = {
                    "parser": self.format_name,
                    "page_count": pages_processed,
                    "total_pages": num_pages,
                    "word_count": len(content.split()) if content else 0,
                    "character_count": len(content),
                    "avg_words_per_page": len(content.split()) / pages_processed
                    if pages_processed > 0 and content
                    else 0.0,
                }

                # Add PDF metadata
                additional_metadata.update(pdf_metadata)

                # Add parsing statistics
                if parsing_info.get("empty_pages", 0) > 0:
                    additional_metadata["empty_pages"] = parsing_info["empty_pages"]

                return ParsedDocument.create(
                    content=content,
                    file_path=file_path,
                    file_type="pdf",
                    additional_metadata=additional_metadata,
                    parsing_info=parsing_info,
                )

        except Exception as e:
            logger.error(f"Failed to parse PDF file {file_path}: {e}")
            raise RuntimeError(f"PDF parsing failed: {e}") from e

    async def _extract_pdf_metadata(
        self, pdf_reader
    ) -> dict[str, str | int | float | bool]:
        """
        Extract metadata from PDF document.

        Args:
            pdf_reader: PyPDF2 PdfReader instance

        Returns:
            Dictionary with PDF metadata
        """
        metadata: dict[str, str | int | float | bool] = {}

        try:
            if hasattr(pdf_reader, "metadata") and pdf_reader.metadata:
                pdf_meta = pdf_reader.metadata

                # Common PDF metadata fields
                metadata_mapping = {
                    "/Title": "title",
                    "/Author": "author",
                    "/Subject": "subject",
                    "/Creator": "creator",
                    "/Producer": "producer",
                    "/CreationDate": "creation_date",
                    "/ModDate": "modification_date",
                    "/Keywords": "keywords",
                }

                for pdf_key, meta_key in metadata_mapping.items():
                    if pdf_key in pdf_meta:
                        value = pdf_meta[pdf_key]

                        # Convert PDF date format if needed
                        if "date" in meta_key.lower() and isinstance(value, str):
                            parsed_date = self._parse_pdf_date(value)
                            metadata[meta_key] = parsed_date
                        elif isinstance(value, str | int | float | bool):
                            metadata[meta_key] = value

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")

        return metadata

    def _parse_pdf_date(self, pdf_date: str) -> str:
        """
        Parse PDF date format to ISO format.

        PDF dates are typically in format: D:YYYYMMDDHHmmSSOHH'mm'
        """
        try:
            if pdf_date.startswith("D:"):
                date_part = pdf_date[2:16]  # YYYYMMDDHHmmSS
                if len(date_part) >= 8:
                    # Convert to ISO format: YYYY-MM-DDTHH:mm:SS
                    year = date_part[:4]
                    month = date_part[4:6]
                    day = date_part[6:8]
                    hour = date_part[8:10] if len(date_part) > 8 else "00"
                    minute = date_part[10:12] if len(date_part) > 10 else "00"
                    second = date_part[12:14] if len(date_part) > 12 else "00"

                    return f"{year}-{month}-{day}T{hour}:{minute}:{second}"

            return pdf_date  # Return as-is if can't parse

        except Exception:
            return pdf_date

    def _clean_pdf_text(self, content: str) -> str:
        """
        Clean and normalize text extracted from PDF.

        Args:
            content: Raw text content from PDF

        Returns:
            Cleaned text content
        """
        if not content:
            return content

        # Remove excessive whitespace and normalize line endings
        lines = []
        for line in content.split("\n"):
            line = line.strip()
            # Skip very short lines that are likely artifacts
            if len(line) > 2 or (len(line) > 0 and line.isalnum()):
                lines.append(line)

        # Join lines and handle paragraph breaks
        content = "\n".join(lines)

        # Fix common PDF extraction issues
        content = content.replace("\x0c", "\n")  # Form feed to newline
        content = content.replace("\xa0", " ")  # Non-breaking space to regular space

        # Normalize excessive whitespace
        import re

        content = re.sub(r"\n\s*\n\s*\n", "\n\n", content)  # Max 2 consecutive newlines
        content = re.sub(r" {2,}", " ", content)  # Multiple spaces to single space

        return content.strip()

    def _analyze_pdf_content(
        self, content: str, pages_processed: int
    ) -> dict[str, Any]:
        """
        Analyze PDF content and generate statistics.

        Args:
            content: Extracted text content
            pages_processed: Number of pages processed

        Returns:
            Dictionary with content analysis results
        """
        if not content:
            return {
                "word_count": 0,
                "character_count": 0,
                "pages_with_content": 0,
                "empty_pages": pages_processed,
                "avg_words_per_page": 0,
            }

        words = content.split()
        lines = content.split("\n")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        return {
            "word_count": len(words),
            "character_count": len(content),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "pages_with_content": pages_processed,
            "empty_pages": 0,  # Assuming processed pages have content
            "avg_words_per_page": len(words) / pages_processed
            if pages_processed > 0
            else 0,
            "avg_chars_per_page": len(content) / pages_processed
            if pages_processed > 0
            else 0,
        }

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for PDF files."""
        return {
            "extract_metadata": {
                "type": bool,
                "default": True,
                "description": "Whether to extract PDF metadata (title, author, etc.)",
            },
            "include_page_numbers": {
                "type": bool,
                "default": False,
                "description": "Whether to include page number markers in content",
            },
            "max_pages": {
                "type": int,
                "default": None,
                "description": "Maximum number of pages to process (None for all)",
            },
            "password": {
                "type": str,
                "default": None,
                "description": "Password for encrypted PDFs",
            },
        }
