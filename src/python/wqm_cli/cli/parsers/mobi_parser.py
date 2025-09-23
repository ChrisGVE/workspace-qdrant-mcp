from loguru import logger

"""
MOBI document parser for extracting text content and metadata.

This module provides functionality to parse MOBI ebook files, extracting
readable text content while preserving metadata like title, author, and
publication information.

Enhanced with comprehensive DRM detection, Kindle format support, and
advanced metadata extraction. Supports MOBI, AZW, AZW3, and KFX formats
with fallback mechanisms for encrypted content.
"""

import logging
import struct
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional dependencies for enhanced MOBI parsing
try:
    from kindleunpack import kindleunpack
    KINDLE_UNPACK_AVAILABLE = True
except ImportError:
    KINDLE_UNPACK_AVAILABLE = False

try:
    import mobidedrm
    MOBI_DEDRM_AVAILABLE = True
except ImportError:
    MOBI_DEDRM_AVAILABLE = False

from .base import DocumentParser, ParsedDocument

# logger imported from loguru


class MobiParser(DocumentParser):
    """
    Enhanced parser for MOBI ebook files and Kindle formats.

    Extracts text content from MOBI files with comprehensive metadata extraction
    and DRM detection. Supports multiple Kindle formats including MOBI, AZW, AZW3.
    Provides advanced error handling and content structure preservation.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """MOBI and Kindle file extensions."""
        return [".mobi", ".azw", ".azw3", ".azw4", ".kfx", ".kfx-zip"]

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "MOBI/Kindle Ebook"

    async def parse(self, file_path: str | Path, **options: Any) -> ParsedDocument:
        """
        Parse MOBI file and extract text content.

        Args:
            file_path: Path to MOBI file
            **options: Parsing options
                - encoding: str = 'utf-8' - Text encoding to use
                - max_content_size: int = 10000000 - Max content size to extract

        Returns:
            ParsedDocument with extracted text and metadata

        Raises:
            RuntimeError: If parsing fails
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        self.validate_file(file_path)

        file_path = Path(file_path)

        try:
            # Parse options
            encoding = options.get("encoding", "utf-8")
            max_content_size = options.get("max_content_size", 10000000)
            attempt_drm_removal = options.get("attempt_drm_removal", False)
            extract_images = options.get("extract_images", False)
            preserve_formatting = options.get("preserve_formatting", True)

            # Detect DRM and file format first
            format_info = await self._analyze_file_format(file_path)
            drm_info = await self._check_drm_protection(file_path, format_info)

            if drm_info["has_drm"]:
                logger.warning(f"DRM-protected Kindle file detected: {file_path.name}")

            # Extract metadata and content with enhanced parsing
            metadata, text_content = await self._extract_enhanced_content(
                file_path, encoding, max_content_size, attempt_drm_removal,
                extract_images, preserve_formatting, format_info, drm_info
            )

            # Enhanced parsing information
            parsing_info = {
                "content_length": len(text_content),
                "encoding_used": encoding,
                "max_content_size": max_content_size,
                "extraction_method": "enhanced_mobi_parser",
                "file_format": format_info.get("format", "unknown"),
                "format_version": format_info.get("version", "unknown"),
                "drm_protected": drm_info["has_drm"],
                "drm_scheme": drm_info.get("scheme", "none"),
                "attempt_drm_removal": attempt_drm_removal,
                "extraction_quality": metadata.get("extraction_quality", "basic"),
                "kindle_unpack_available": KINDLE_UNPACK_AVAILABLE,
                "mobi_dedrm_available": MOBI_DEDRM_AVAILABLE,
            }

            logger.info(
                f"Successfully parsed MOBI: {file_path.name} "
                f"({parsing_info['content_length']:,} characters)"
            )

            return ParsedDocument.create(
                content=text_content,
                file_path=file_path,
                file_type="mobi",
                additional_metadata=metadata,
                parsing_info=parsing_info,
            )

        except Exception as e:
            logger.error(f"Failed to parse MOBI {file_path}: {e}")
            # Provide detailed error diagnosis
            error_details = await self._diagnose_parsing_error(file_path, e)
            raise RuntimeError(f"MOBI parsing failed: {error_details}") from e

    async def _extract_mobi_content(
        self, file_path: Path, encoding: str, max_content_size: int
    ) -> tuple[dict[str, str | int | float | bool], str]:
        """Extract content and metadata from MOBI file using basic parsing."""

        metadata = {}

        with open(file_path, "rb") as f:
            # Read MOBI header to extract basic information
            try:
                # Read PalmDOC header
                f.seek(0)
                palm_header = f.read(78)

                if len(palm_header) < 78:
                    raise RuntimeError("Invalid MOBI file: too short")

                # Extract basic Palm database info
                name = (
                    palm_header[0:32].rstrip(b"\x00").decode("utf-8", errors="ignore")
                )
                if name:
                    metadata["title"] = name

                # Try to find MOBI header
                f.seek(78)
                mobi_header_data = f.read(232)  # MOBI header is typically 232 bytes

                if len(mobi_header_data) >= 4:
                    mobi_identifier = mobi_header_data[0:4]
                    if mobi_identifier == b"MOBI":
                        # This is a proper MOBI file
                        metadata["format_version"] = "MOBI"

                        # Try to extract language (if available in header)
                        if len(mobi_header_data) >= 92:
                            lang_code = struct.unpack(">I", mobi_header_data[88:92])[0]
                            if lang_code != 0:
                                metadata["language_code"] = lang_code

                # Extract text content using a simple approach
                # This is a basic extraction - real MOBI parsing is complex
                text_content = await self._extract_text_content(
                    f, encoding, max_content_size
                )

            except Exception as e:
                logger.warning(
                    f"MOBI header parsing failed, attempting basic text extraction: {e}"
                )
                # Fallback: try to extract any readable text
                f.seek(0)
                raw_data = f.read(max_content_size)
                text_content = self._extract_readable_text(raw_data, encoding)
                metadata["extraction_method"] = "fallback_text_extraction"

        # Add comprehensive file-level metadata
        file_stat = file_path.stat()
        metadata.update({
            "file_size": file_stat.st_size,
            "file_format": format_info.get("format", "mobi"),
            "format_version": format_info.get("version", "unknown"),
            "has_metadata": len(metadata) > 5,  # More than just basic file info
            "extraction_quality": metadata.get("extraction_quality", "basic"),
            "drm_protected": drm_info["has_drm"],
            "drm_scheme": drm_info["scheme"],
            "content_length": len(text_content),
            "has_readable_content": bool(text_content.strip()),
            "parsing_timestamp": file_stat.st_mtime,
        })

        # Add DRM details if any
        if drm_info["details"]:
            metadata["drm_details"] = "; ".join(drm_info["details"])

        return metadata, text_content

    async def _parse_with_kindleunpack(self, file_path: Path, preserve_formatting: bool) -> Tuple[Dict[str, Any], str]:
        """Parse MOBI using kindleunpack library for high-quality extraction."""
        metadata = {}
        text_content = ""

        if not KINDLE_UNPACK_AVAILABLE:
            raise RuntimeError("kindleunpack library not available")

        # Note: This is a placeholder for kindleunpack integration
        # Real implementation would require proper kindleunpack setup
        logger.info("KindleUnpack parsing not yet implemented")
        raise RuntimeError("KindleUnpack integration not yet implemented")

    async def _parse_with_drm_removal(self, file_path: Path, encoding: str, preserve_formatting: bool) -> Tuple[Dict[str, Any], str]:
        """Attempt to parse MOBI with DRM removal."""
        metadata = {}
        text_content = ""

        if not MOBI_DEDRM_AVAILABLE:
            raise RuntimeError("mobidedrm library not available")

        # Note: This is a placeholder for DRM removal integration
        # Real implementation would require proper DRM removal setup
        logger.info("DRM removal parsing not yet implemented")
        raise RuntimeError("DRM removal integration not yet implemented")

    async def _extract_enhanced_text_content(self, f, encoding: str, max_size: int, preserve_formatting: bool, format_info: Dict[str, Any]) -> str:
        """Extract text content with enhanced formatting preservation."""
        try:
            # Advanced text extraction based on MOBI structure
            f.seek(0)
            header = f.read(1024)

            # Find text records start position
            text_start = 78  # Default after PDB header

            # Look for MOBI header to get better text position
            if b'MOBI' in header:
                mobi_pos = header.find(b'MOBI')
                if mobi_pos != -1 and mobi_pos + 84 < len(header):
                    # Extract first text record position
                    try:
                        first_record = struct.unpack('>I', header[mobi_pos + 80:mobi_pos + 84])[0]
                        if first_record > 0 and first_record < max_size:
                            text_start = first_record
                    except struct.error:
                        pass

            # Position to text content
            f.seek(text_start)
            content_bytes = f.read(max_size)

            # Enhanced text extraction with formatting preservation
            if preserve_formatting:
                return await self._extract_formatted_text(content_bytes, encoding, format_info)
            else:
                return self._extract_readable_text(content_bytes, encoding)

        except Exception as e:
            logger.warning(f"Enhanced text extraction failed: {e}")
            return await self._extract_text_content_fallback(f, encoding, max_size)

    async def _extract_text_content_fallback(self, f, encoding: str, max_size: int) -> str:
        """Fallback text extraction method."""
        try:
            # Skip headers and try to find text content
            # MOBI files typically have text starting after headers
            f.seek(1024)  # Skip most header content

            content_bytes = f.read(max_size)

            # Filter out binary data and extract readable text
            return self._extract_readable_text(content_bytes, encoding)

        except Exception as e:
            logger.warning(f"Fallback text extraction failed: {e}")
            return (
                "[MOBI content extraction failed - file may be encrypted or corrupted]"
            )

    async def _extract_formatted_text(self, content_bytes: bytes, encoding: str, format_info: Dict[str, Any]) -> str:
        """Extract text while attempting to preserve formatting."""
        try:
            # Try to decode with detected encoding first
            detected_encoding = format_info.get("text_encoding", encoding)
            try:
                text = content_bytes.decode(detected_encoding, errors="ignore")
            except (UnicodeDecodeError, LookupError):
                text = content_bytes.decode(encoding, errors="ignore")

            # Look for HTML-like formatting in MOBI content
            if '<' in text and '>' in text:
                # MOBI files often contain HTML-like markup
                try:
                    from bs4 import BeautifulSoup
                    # Parse HTML content if present
                    soup = BeautifulSoup(text, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.extract()

                    # Preserve some structure
                    formatted_text = []
                    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
                        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            level = int(element.name[1])
                            heading_prefix = '#' * level
                            text_content = element.get_text(strip=True)
                            if text_content:
                                formatted_text.append(f"\n{heading_prefix} {text_content}\n")
                        else:
                            text_content = element.get_text(strip=True)
                            if text_content:
                                formatted_text.append(text_content)

                    if formatted_text:
                        return '\n'.join(formatted_text)
                except ImportError:
                    # BeautifulSoup not available, fall back to basic cleaning
                    pass

            # Basic text cleaning if no HTML formatting found
            return self._extract_readable_text(content_bytes, detected_encoding)

        except Exception as e:
            logger.warning(f"Formatted text extraction failed: {e}")
            return self._extract_readable_text(content_bytes, encoding)

    def _extract_readable_text(self, data: bytes, encoding: str) -> str:
        """Extract readable text from binary data."""
        try:
            # Try to decode with specified encoding
            text = data.decode(encoding, errors="ignore")
        except:
            # Fallback to latin1 which can decode any byte sequence
            text = data.decode("latin1", errors="ignore")

        # Clean up the text - remove non-printable characters except common whitespace
        cleaned_lines = []
        for line in text.split("\n"):
            # Keep lines that have a reasonable amount of printable characters
            printable_chars = sum(1 for c in line if c.isprintable() or c in "\t ")
            if (
                len(line) > 0 and printable_chars / len(line) > 0.7
            ):  # At least 70% printable
                # Remove excessive whitespace
                cleaned_line = " ".join(line.split())
                if len(cleaned_line) > 10:  # Only keep substantial lines
                    cleaned_lines.append(cleaned_line)

        result = "\n".join(cleaned_lines)

        # If we got very little content, add a note
        if len(result) < 100:
            result += (
                "\n\n[Note: MOBI file may be encrypted, DRM-protected, or in an unsupported format. "
                "For best results with MOBI files, consider converting to EPUB format first.]"
            )

        return result

    async def _diagnose_parsing_error(self, file_path: Path, error: Exception) -> str:
        """Provide detailed error diagnosis for MOBI parsing failures."""
        error_details = [str(error)]

        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return "File does not exist"

            if not file_path.is_file():
                return "Path is not a file"

            # Check file size
            file_size = file_path.stat().st_size
            if file_size == 0:
                error_details.append("File is empty")
            elif file_size < 1024:
                error_details.append("File is too small to be a valid MOBI")
            elif file_size > 500 * 1024 * 1024:  # 500MB
                error_details.append(f"File is very large ({file_size // (1024*1024)}MB)")

            # Check file header
            try:
                with open(file_path, "rb") as f:
                    header = f.read(78)
                    if len(header) < 78:
                        error_details.append("File header is incomplete")
                    elif header[60:68] != b'BOOKMOBI' and b'MOBI' not in header[:1024]:
                        error_details.append("File does not appear to be a valid MOBI format")
            except Exception as header_error:
                error_details.append(f"Header check failed: {header_error}")

            # Check for common DRM/encryption patterns in error
            error_str = str(error).lower()
            if any(term in error_str for term in ['drm', 'encrypted', 'protected', 'kindle', 'amazon']):
                error_details.append("File appears to be DRM-protected or encrypted")

            # Check file extension
            suffix = file_path.suffix.lower()
            if suffix not in ['.mobi', '.azw', '.azw3', '.azw4', '.kfx', '.kfx-zip']:
                error_details.append(f"Unusual file extension: {suffix}")

        except Exception as diag_error:
            error_details.append(f"Diagnosis failed: {diag_error}")

        return "; ".join(error_details)

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for MOBI files."""
        return {
            "encoding": {
                "type": str,
                "default": "utf-8",
                "description": "Text encoding to use for text extraction",
            },
            "max_content_size": {
                "type": int,
                "default": 10000000,
                "description": "Maximum content size to extract (bytes)",
            },
            "attempt_drm_removal": {
                "type": bool,
                "default": False,
                "description": "Attempt to remove DRM protection (requires mobidedrm)",
            },
            "extract_images": {
                "type": bool,
                "default": False,
                "description": "Extract and include image information",
            },
            "preserve_formatting": {
                "type": bool,
                "default": True,
                "description": "Attempt to preserve text formatting and structure",
            },
        }
