from ...observability import get_logger

logger = get_logger(__name__)
"""
EPUB document parser for extracting text content and metadata.

This module provides functionality to parse EPUB ebook files, extracting
readable text content from chapters while preserving metadata like title,
author, and publication information.
"""

import logging
from pathlib import Path
from typing import Any

try:
    import ebooklib
    from bs4 import BeautifulSoup
    from ebooklib import epub

    EPUB_AVAILABLE = True
except ImportError:
    EPUB_AVAILABLE = False

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class EpubParser(DocumentParser):
    """
    Parser for EPUB ebook files.

    Extracts text content from EPUB chapters and metadata including
    title, author, publisher, language, and publication date.
    Handles both EPUB2 and EPUB3 formats.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """EPUB file extensions."""
        return [".epub"]

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "EPUB Ebook"

    def _check_availability(self) -> None:
        """Check if required libraries are available."""
        if not EPUB_AVAILABLE:
            raise RuntimeError(
                "EPUB parsing requires 'ebooklib' and 'beautifulsoup4'. "
                "Install with: pip install ebooklib beautifulsoup4"
            )

    async def parse(self, file_path: str | Path, **options: Any) -> ParsedDocument:
        """
        Parse EPUB file and extract text content.

        Args:
            file_path: Path to EPUB file
            **options: Parsing options
                - include_images: bool = False - Include image descriptions
                - max_chapter_size: int = 50000 - Max chars per chapter
                - chapter_separator: str = "\n\n---\n\n" - Chapter separator

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            RuntimeError: If required libraries are not installed
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        self._check_availability()
        self.validate_file(file_path)

        file_path = Path(file_path)

        try:
            # Parse options
            include_images = options.get("include_images", False)
            max_chapter_size = options.get("max_chapter_size", 50000)
            chapter_separator = options.get("chapter_separator", "\n\n---\n\n")

            # Read EPUB file
            book = epub.read_epub(str(file_path))

            # Extract metadata
            metadata = await self._extract_metadata(book)

            # Extract text content from chapters
            text_content = await self._extract_text_content(
                book, include_images, max_chapter_size, chapter_separator
            )

            # Parsing information
            parsing_info = {
                "total_chapters": len(
                    list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
                ),
                "content_length": len(text_content),
                "include_images": include_images,
                "max_chapter_size": max_chapter_size,
            }

            logger.info(
                f"Successfully parsed EPUB: {file_path.name} "
                f"({parsing_info['total_chapters']} chapters, "
                f"{parsing_info['content_length']:,} characters)"
            )

            return ParsedDocument.create(
                content=text_content,
                file_path=file_path,
                file_type="epub",
                additional_metadata=metadata,
                parsing_info=parsing_info,
            )

        except Exception as e:
            logger.error(f"Failed to parse EPUB {file_path}: {e}")
            raise RuntimeError(f"EPUB parsing failed: {e}") from e

    async def _extract_metadata(
        self, book: epub.EpubBook
    ) -> dict[str, str | int | float | bool]:
        """Extract metadata from EPUB book."""
        metadata = {}

        # Basic metadata
        if book.get_metadata("DC", "title"):
            metadata["title"] = book.get_metadata("DC", "title")[0][0]

        if book.get_metadata("DC", "creator"):
            authors = [item[0] for item in book.get_metadata("DC", "creator")]
            metadata["author"] = ", ".join(authors)
            metadata["author_count"] = len(authors)

        if book.get_metadata("DC", "publisher"):
            metadata["publisher"] = book.get_metadata("DC", "publisher")[0][0]

        if book.get_metadata("DC", "date"):
            metadata["publication_date"] = book.get_metadata("DC", "date")[0][0]

        if book.get_metadata("DC", "language"):
            metadata["language"] = book.get_metadata("DC", "language")[0][0]

        if book.get_metadata("DC", "identifier"):
            identifiers = book.get_metadata("DC", "identifier")
            for identifier, attrs in identifiers:
                if attrs and "scheme" in attrs:
                    if attrs["scheme"].upper() == "ISBN":
                        metadata["isbn"] = identifier
                    elif attrs["scheme"].upper() == "UUID":
                        metadata["uuid"] = identifier

        # Additional metadata
        if book.get_metadata("DC", "description"):
            description = book.get_metadata("DC", "description")[0][0]
            metadata["description"] = (
                description[:500] + "..." if len(description) > 500 else description
            )

        if book.get_metadata("DC", "subject"):
            subjects = [item[0] for item in book.get_metadata("DC", "subject")]
            metadata["subjects"] = ", ".join(subjects)
            metadata["subject_count"] = len(subjects)

        # Count chapters and images
        chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))

        metadata["chapter_count"] = len(chapters)
        metadata["image_count"] = len(images)
        metadata["has_images"] = len(images) > 0

        return metadata

    async def _extract_text_content(
        self,
        book: epub.EpubBook,
        include_images: bool,
        max_chapter_size: int,
        chapter_separator: str,
    ) -> str:
        """Extract text content from all chapters."""
        chapters = []

        # Get all document items (chapters)
        items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)

        for item in items:
            try:
                # Get chapter content as HTML
                content = item.get_content().decode("utf-8")

                # Parse HTML with BeautifulSoup
                soup = BeautifulSoup(content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()

                # Handle images if requested
                if include_images:
                    for img in soup.find_all("img"):
                        alt_text = img.get("alt", "")
                        title_text = img.get("title", "")
                        src = img.get("src", "")

                        img_description = []
                        if alt_text:
                            img_description.append(f"Image: {alt_text}")
                        if title_text and title_text != alt_text:
                            img_description.append(f"({title_text})")
                        if not img_description and src:
                            img_description.append(f"[Image: {Path(src).name}]")

                        if img_description:
                            img.replace_with(f"[{' '.join(img_description)}]")
                        else:
                            img.extract()
                else:
                    # Remove all images
                    for img in soup.find_all("img"):
                        img.extract()

                # Extract text content
                text = soup.get_text()

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines()]
                text = "\n".join(line for line in lines if line)

                # Truncate if chapter is too long
                if len(text) > max_chapter_size:
                    text = (
                        text[:max_chapter_size] + "\n\n[Chapter content truncated...]"
                    )

                if text.strip():
                    chapters.append(text)

            except Exception as e:
                logger.warning(
                    f"Failed to extract content from chapter {item.get_name()}: {e}"
                )
                continue

        # Join chapters with separator
        return chapter_separator.join(chapters)

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for EPUB files."""
        return {
            "include_images": {
                "type": bool,
                "default": False,
                "description": "Include image descriptions in extracted text",
            },
            "max_chapter_size": {
                "type": int,
                "default": 50000,
                "description": "Maximum characters per chapter (prevents memory issues)",
            },
            "chapter_separator": {
                "type": str,
                "default": "\n\n---\n\n",
                "description": "Text separator between chapters",
            },
        }
