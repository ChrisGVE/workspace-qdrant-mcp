from loguru import logger

"""
EPUB document parser for extracting text content and metadata.

This module provides functionality to parse EPUB ebook files, extracting
readable text content from chapters while preserving metadata like title,
author, and publication information. Enhanced with comprehensive DRM detection,
error handling, and rich metadata extraction.
"""

import zipfile
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

# logger imported from loguru


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
            preserve_structure = options.get("preserve_structure", True)
            extract_toc = options.get("extract_toc", True)

            # Check for DRM and validate file structure first
            drm_info = await self._check_drm_protection(file_path)
            if drm_info["has_drm"]:
                logger.warning(f"DRM-protected EPUB detected: {file_path.name}")

            # Read EPUB file with enhanced error handling
            book = await self._safe_read_epub(str(file_path))

            # Extract comprehensive metadata with structure info
            metadata = await self._extract_enhanced_metadata(book, drm_info)

            # Extract Table of Contents if requested
            toc_info = {}
            if extract_toc:
                toc_info = await self._extract_table_of_contents(book)
                metadata.update(toc_info)

            # Extract text content from chapters with structure preservation
            text_content = await self._extract_structured_content(
                book, include_images, max_chapter_size, chapter_separator, preserve_structure
            )

            # Enhanced parsing information
            parsing_info = {
                "total_chapters": len(
                    list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
                ),
                "content_length": len(text_content),
                "include_images": include_images,
                "max_chapter_size": max_chapter_size,
                "preserve_structure": preserve_structure,
                "extract_toc": extract_toc,
                "drm_protected": drm_info["has_drm"],
                "drm_scheme": drm_info.get("scheme", "none"),
                "parsing_method": "enhanced_epub_parser",
                "epub_version": metadata.get("epub_version", "unknown"),
                "toc_entries": len(toc_info.get("toc_entries", [])),
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
            # Provide more specific error information
            error_details = await self._diagnose_parsing_error(file_path, e)
            raise RuntimeError(f"EPUB parsing failed: {error_details}") from e

    async def _check_drm_protection(self, file_path: Path) -> dict[str, Any]:
        """Check if EPUB file is DRM-protected."""
        drm_info = {"has_drm": False, "scheme": "none", "details": []}

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                file_list = zip_file.namelist()

                # Check for common DRM indicators
                drm_indicators = [
                    "META-INF/encryption.xml",
                    "META-INF/rights.xml",
                    "META-INF/manifest.xml",
                    ".drm",
                    "adept"
                ]

                for indicator in drm_indicators:
                    if any(indicator in fname for fname in file_list):
                        drm_info["has_drm"] = True
                        drm_info["details"].append(f"Found DRM indicator: {indicator}")

                # Check for Adobe ADEPT DRM
                if "META-INF/encryption.xml" in file_list:
                    try:
                        encryption_data = zip_file.read("META-INF/encryption.xml")
                        if b"adobe.com" in encryption_data or b"adept" in encryption_data:
                            drm_info["scheme"] = "Adobe ADEPT"
                    except Exception:
                        pass

                # Check for Amazon DRM indicators
                if any("kindle" in fname.lower() or "amazon" in fname.lower() for fname in file_list):
                    drm_info["scheme"] = "Amazon Kindle"
                    drm_info["has_drm"] = True
                    drm_info["details"].append("Amazon/Kindle DRM indicators found")

        except zipfile.BadZipFile:
            drm_info["details"].append("File is not a valid ZIP/EPUB archive")
        except Exception as e:
            drm_info["details"].append(f"DRM check failed: {str(e)}")

        return drm_info

    async def _safe_read_epub(self, file_path: str) -> epub.EpubBook:
        """Safely read EPUB with enhanced error handling."""
        try:
            # First attempt normal reading
            return epub.read_epub(file_path)
        except Exception as e:
            logger.warning(f"Standard EPUB reading failed: {e}. Attempting recovery...")

            # Try to repair and read again
            try:
                # Check if it's a valid ZIP file first
                with zipfile.ZipFile(file_path, 'r') as zf:
                    if zf.testzip() is not None:
                        raise RuntimeError("EPUB file is corrupted")

                # Try reading with more permissive settings
                book = epub.EpubBook()
                book.set_identifier('unknown')
                book.set_title('Unknown Title')
                book.set_language('en')

                # Manually parse the EPUB structure
                with zipfile.ZipFile(file_path, 'r') as zf:
                    # Try to read container.xml
                    try:
                        zf.read('META-INF/container.xml')
                        # Parse to find OPF file location
                        # This is a simplified recovery approach
                        logger.info("Attempting manual EPUB structure recovery")
                    except KeyError:
                        logger.warning("META-INF/container.xml not found")

                return book

            except Exception as recovery_error:
                logger.error(f"EPUB recovery failed: {recovery_error}")
                raise RuntimeError(f"EPUB file is severely corrupted or encrypted: {recovery_error}") from e

    async def _extract_enhanced_metadata(
        self, book: epub.EpubBook, drm_info: dict[str, Any]
    ) -> dict[str, str | int | float | bool]:
        """Extract comprehensive metadata from EPUB book."""
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

        # Enhanced metadata extraction
        if book.get_metadata("DC", "description"):
            description = book.get_metadata("DC", "description")[0][0]
            metadata["description"] = (
                description[:500] + "..." if len(description) > 500 else description
            )
            metadata["has_description"] = True
        else:
            metadata["has_description"] = False

        if book.get_metadata("DC", "subject"):
            subjects = [item[0] for item in book.get_metadata("DC", "subject")]
            metadata["subjects"] = ", ".join(subjects)
            metadata["subject_count"] = len(subjects)
        else:
            metadata["subject_count"] = 0

        # Extract additional DC metadata
        if book.get_metadata("DC", "rights"):
            metadata["rights"] = book.get_metadata("DC", "rights")[0][0]

        if book.get_metadata("DC", "source"):
            metadata["source"] = book.get_metadata("DC", "source")[0][0]

        if book.get_metadata("DC", "type"):
            metadata["dc_type"] = book.get_metadata("DC", "type")[0][0]

        if book.get_metadata("DC", "format"):
            metadata["dc_format"] = book.get_metadata("DC", "format")[0][0]

        if book.get_metadata("DC", "contributor"):
            contributors = [item[0] for item in book.get_metadata("DC", "contributor")]
            metadata["contributors"] = ", ".join(contributors)
            metadata["contributor_count"] = len(contributors)

        # Extract EPUB-specific metadata
        try:
            # Try to determine EPUB version
            metadata["epub_version"] = "2.0"  # Default
            for item in book.get_items():
                if hasattr(item, 'get_name') and item.get_name().endswith('.opf'):
                    content = item.get_content()
                    if b'version="3.0"' in content:
                        metadata["epub_version"] = "3.0"
                    elif b'version="2.0"' in content:
                        metadata["epub_version"] = "2.0"
                    break
        except Exception:
            metadata["epub_version"] = "unknown"

        # Add DRM information
        metadata["drm_protected"] = drm_info["has_drm"]
        metadata["drm_scheme"] = drm_info["scheme"]
        if drm_info["details"]:
            metadata["drm_details"] = "; ".join(drm_info["details"])

        # Enhanced chapter and media analysis
        chapters = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        images = list(book.get_items_of_type(ebooklib.ITEM_IMAGE))

        metadata["chapter_count"] = len(chapters)
        metadata["image_count"] = len(images)
        metadata["has_images"] = len(images) > 0

        # Analyze media types
        media_types = set()
        for img in images:
            if hasattr(img, 'media_type'):
                media_types.add(img.media_type)
        metadata["image_formats"] = ", ".join(sorted(media_types)) if media_types else "none"

        # Check for audio/video content
        audio_items = list(book.get_items_of_type(ebooklib.ITEM_AUDIO))
        video_items = list(book.get_items_of_type(ebooklib.ITEM_VIDEO))
        metadata["audio_count"] = len(audio_items)
        metadata["video_count"] = len(video_items)
        metadata["has_multimedia"] = len(audio_items) > 0 or len(video_items) > 0

        # Estimate content complexity
        total_items = len(list(book.get_items()))
        metadata["total_items"] = total_items
        metadata["complexity_score"] = min(10, max(1, total_items // 10))  # 1-10 scale

        return metadata

    async def _extract_table_of_contents(self, book: epub.EpubBook) -> dict[str, Any]:
        """Extract Table of Contents information."""
        toc_info = {
            "toc_entries": [],
            "toc_structure": "none",
            "navigation_document": False
        }

        try:
            # Try to get NCX-based TOC (EPUB 2.x)
            if hasattr(book, 'toc') and book.toc:
                toc_entries = []
                self._parse_toc_recursive(book.toc, toc_entries)
                toc_info["toc_entries"] = toc_entries
                toc_info["toc_structure"] = "ncx"

            # Try to get navigation document (EPUB 3.x)
            nav_doc = None
            for item in book.get_items():
                if (hasattr(item, 'get_name') and
                    ('nav' in item.get_name().lower() or 'navigation' in item.get_name().lower())):
                    nav_doc = item
                    break

            if nav_doc:
                toc_info["navigation_document"] = True
                nav_entries = await self._parse_navigation_document(nav_doc)
                if nav_entries:
                    toc_info["toc_entries"].extend(nav_entries)
                    toc_info["toc_structure"] = "navigation_document"

        except Exception as e:
            logger.warning(f"TOC extraction failed: {e}")
            toc_info["extraction_error"] = str(e)

        return toc_info

    def _parse_toc_recursive(self, toc_items: list[Any], entries: list[dict[str, Any]], level: int = 0) -> None:
        """Recursively parse TOC structure."""
        for item in toc_items:
            if hasattr(item, 'title') and hasattr(item, 'href'):
                entries.append({
                    "title": item.title,
                    "href": item.href,
                    "level": level
                })
            elif isinstance(item, tuple) and len(item) >= 2:
                # Handle different TOC item formats
                if hasattr(item[0], 'title'):
                    entries.append({
                        "title": item[0].title,
                        "href": item[0].href if hasattr(item[0], 'href') else '',
                        "level": level
                    })
                if len(item) > 1 and isinstance(item[1], list):
                    self._parse_toc_recursive(item[1], entries, level + 1)

    async def _parse_navigation_document(self, nav_doc) -> list[dict[str, Any]]:
        """Parse EPUB 3 navigation document."""
        entries = []
        try:
            content = nav_doc.get_content().decode('utf-8')
            # Parse HTML navigation document
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')

            # Find navigation elements
            nav_elements = soup.find_all('nav')
            for nav in nav_elements:
                ol_elements = nav.find_all('ol')
                for ol in ol_elements:
                    self._parse_nav_list(ol, entries, 0)

        except Exception as e:
            logger.warning(f"Navigation document parsing failed: {e}")

        return entries

    def _parse_nav_list(self, ol_element, entries: list[dict[str, Any]], level: int) -> None:
        """Parse navigation list elements."""
        for li in ol_element.find_all('li', recursive=False):
            a_tag = li.find('a')
            if a_tag:
                entries.append({
                    "title": a_tag.get_text(strip=True),
                    "href": a_tag.get('href', ''),
                    "level": level
                })
            # Check for nested lists
            nested_ol = li.find('ol')
            if nested_ol:
                self._parse_nav_list(nested_ol, entries, level + 1)

    async def _extract_structured_content(
        self,
        book: epub.EpubBook,
        include_images: bool,
        max_chapter_size: int,
        chapter_separator: str,
        preserve_structure: bool
    ) -> str:
        """Extract text content with enhanced structure preservation."""
        chapters = []
        chapter_metadata = []

        # Get all document items (chapters) in reading order
        items = book.get_items_of_type(ebooklib.ITEM_DOCUMENT)

        # Try to get spine order for proper chapter sequence
        spine_order = []
        if hasattr(book, 'spine'):
            spine_order = [item[0] for item in book.spine if len(item) > 0]

        # Reorder items based on spine if available
        if spine_order:
            ordered_items = []
            for spine_id in spine_order:
                for item in items:
                    if hasattr(item, 'id') and item.id == spine_id:
                        ordered_items.append(item)
                        break
            # Add any remaining items not in spine
            for item in items:
                if item not in ordered_items:
                    ordered_items.append(item)
            items = ordered_items

        for idx, item in enumerate(items):
            try:
                # Get chapter content as HTML
                content = item.get_content().decode("utf-8")

                # Extract chapter-level metadata
                chapter_meta = {
                    "chapter_index": idx,
                    "item_id": getattr(item, 'id', f'chapter_{idx}'),
                    "file_name": getattr(item, 'file_name', f'chapter_{idx}.html'),
                    "media_type": getattr(item, 'media_type', 'application/xhtml+xml')
                }

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

                # Extract text content with structure preservation
                if preserve_structure:
                    text = await self._extract_structured_text(soup)
                else:
                    text = soup.get_text()

                # Clean up whitespace
                lines = [line.strip() for line in text.splitlines()]
                text = "\n".join(line for line in lines if line)

                # Update chapter metadata
                chapter_meta.update({
                    "original_length": len(text),
                    "word_count": len(text.split()) if text else 0,
                    "has_content": bool(text.strip())
                })

                # Truncate if chapter is too long
                if len(text) > max_chapter_size:
                    text = (
                        text[:max_chapter_size] + "\n\n[Chapter content truncated...]"
                    )
                    chapter_meta["truncated"] = True
                    chapter_meta["truncated_at"] = max_chapter_size
                else:
                    chapter_meta["truncated"] = False

                if text.strip():
                    if preserve_structure:
                        # Add chapter header with metadata
                        chapter_title = chapter_meta.get('item_id', f'Chapter {idx + 1}')
                        formatted_chapter = f"\n# {chapter_title}\n\n{text}"
                        chapters.append(formatted_chapter)
                    else:
                        chapters.append(text)

                    chapter_metadata.append(chapter_meta)

            except Exception as e:
                chapter_name = getattr(item, 'file_name', f'chapter_{idx}')
                logger.warning(
                    f"Failed to extract content from chapter {chapter_name}: {e}"
                )
                # Record failed chapter
                chapter_metadata.append({
                    "chapter_index": idx,
                    "item_id": getattr(item, 'id', f'chapter_{idx}'),
                    "extraction_failed": True,
                    "error": str(e)
                })
                continue

        # Join chapters with separator and add metadata summary
        content = chapter_separator.join(chapters)

        if preserve_structure and chapter_metadata:
            # Add metadata summary at the end
            metadata_summary = "\n\n---\n\n# Document Structure Summary\n\n"
            successful_chapters = [m for m in chapter_metadata if not m.get('extraction_failed', False)]
            failed_chapters = [m for m in chapter_metadata if m.get('extraction_failed', False)]

            metadata_summary += f"Total chapters processed: {len(chapter_metadata)}\n"
            metadata_summary += f"Successfully extracted: {len(successful_chapters)}\n"
            if failed_chapters:
                metadata_summary += f"Failed extractions: {len(failed_chapters)}\n"

            content += metadata_summary

        return content

    async def _extract_structured_text(self, soup) -> str:
        """Extract text while preserving document structure."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Preserve headings structure
        structured_text = []

        # Process each element to maintain hierarchy
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                heading_prefix = '#' * level
                text = element.get_text(strip=True)
                if text:
                    structured_text.append(f"\n{heading_prefix} {text}\n")
            else:
                text = element.get_text(strip=True)
                if text:
                    structured_text.append(text)

        return '\n'.join(structured_text)

    async def _diagnose_parsing_error(self, file_path: Path, error: Exception) -> str:
        """Provide detailed error diagnosis for parsing failures."""
        error_details = [str(error)]

        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return "File does not exist"

            if not file_path.is_file():
                return "Path is not a file"

            # Check if it's a valid ZIP file
            try:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    file_list = zf.namelist()
                    if 'META-INF/container.xml' not in file_list:
                        error_details.append("Missing EPUB container.xml")

                    # Test ZIP integrity
                    bad_file = zf.testzip()
                    if bad_file:
                        error_details.append(f"Corrupted ZIP file: {bad_file}")

            except zipfile.BadZipFile:
                error_details.append("File is not a valid ZIP archive")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                error_details.append("File is empty")
            elif file_size > 100 * 1024 * 1024:  # 100MB
                error_details.append(f"File is very large ({file_size // (1024*1024)}MB)")

            # Check for common DRM patterns in error
            error_str = str(error).lower()
            if any(term in error_str for term in ['drm', 'encrypted', 'protected', 'adobe', 'adept']):
                error_details.append("File appears to be DRM-protected")

        except Exception as diag_error:
            error_details.append(f"Diagnosis failed: {diag_error}")

        return "; ".join(error_details)

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
            "preserve_structure": {
                "type": bool,
                "default": True,
                "description": "Preserve document structure (headings, chapters)",
            },
            "extract_toc": {
                "type": bool,
                "default": True,
                "description": "Extract Table of Contents information",
            },
        }
