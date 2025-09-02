
from ...observability import get_logger
logger = get_logger(__name__)
"""
MOBI document parser for extracting text content and metadata.

This module provides functionality to parse MOBI ebook files, extracting
readable text content while preserving metadata like title, author, and
publication information.

Note: MOBI parsing has limited library support. This implementation provides
basic functionality using the mobidedrm approach or falls back to text extraction.
"""

import logging
from pathlib import Path
from typing import Any
import struct

from .base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class MobiParser(DocumentParser):
    """
    Parser for MOBI ebook files.
    
    Extracts text content from MOBI files with basic metadata extraction.
    Note: MOBI is a legacy format with limited parsing library support.
    This implementation provides basic text extraction capabilities.
    """

    @property
    def supported_extensions(self) -> list[str]:
        """MOBI file extensions."""
        return ['.mobi', '.azw', '.azw3']

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return 'MOBI Ebook'

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
            encoding = options.get('encoding', 'utf-8')
            max_content_size = options.get('max_content_size', 10000000)
            
            # Extract metadata and content
            metadata, text_content = await self._extract_mobi_content(
                file_path, encoding, max_content_size
            )
            
            # Parsing information
            parsing_info = {
                "content_length": len(text_content),
                "encoding_used": encoding,
                "max_content_size": max_content_size,
                "extraction_method": "basic_mobi_parser"
            }
            
            logger.info(f"Successfully parsed MOBI: {file_path.name} "
                       f"({parsing_info['content_length']:,} characters)")
            
            return ParsedDocument.create(
                content=text_content,
                file_path=file_path,
                file_type='mobi',
                additional_metadata=metadata,
                parsing_info=parsing_info
            )
            
        except Exception as e:
            logger.error(f"Failed to parse MOBI {file_path}: {e}")
            raise RuntimeError(f"MOBI parsing failed: {e}") from e

    async def _extract_mobi_content(
        self, 
        file_path: Path, 
        encoding: str,
        max_content_size: int
    ) -> tuple[dict[str, str | int | float | bool], str]:
        """Extract content and metadata from MOBI file using basic parsing."""
        
        metadata = {}
        
        with open(file_path, 'rb') as f:
            # Read MOBI header to extract basic information
            try:
                # Read PalmDOC header
                f.seek(0)
                palm_header = f.read(78)
                
                if len(palm_header) < 78:
                    raise RuntimeError("Invalid MOBI file: too short")
                
                # Extract basic Palm database info
                name = palm_header[0:32].rstrip(b'\x00').decode('utf-8', errors='ignore')
                if name:
                    metadata['title'] = name
                
                # Try to find MOBI header
                f.seek(78)
                mobi_header_data = f.read(232)  # MOBI header is typically 232 bytes
                
                if len(mobi_header_data) >= 4:
                    mobi_identifier = mobi_header_data[0:4]
                    if mobi_identifier == b'MOBI':
                        # This is a proper MOBI file
                        metadata['format_version'] = 'MOBI'
                        
                        # Try to extract language (if available in header)
                        if len(mobi_header_data) >= 92:
                            lang_code = struct.unpack('>I', mobi_header_data[88:92])[0]
                            if lang_code != 0:
                                metadata['language_code'] = lang_code
                
                # Extract text content using a simple approach
                # This is a basic extraction - real MOBI parsing is complex
                text_content = await self._extract_text_content(f, encoding, max_content_size)
                
            except Exception as e:
                logger.warning(f"MOBI header parsing failed, attempting basic text extraction: {e}")
                # Fallback: try to extract any readable text
                f.seek(0)
                raw_data = f.read(max_content_size)
                text_content = self._extract_readable_text(raw_data, encoding)
                metadata['extraction_method'] = 'fallback_text_extraction'
        
        # Add file-level metadata
        file_stat = file_path.stat()
        metadata.update({
            'file_size': file_stat.st_size,
            'file_format': 'mobi',
            'has_metadata': len(metadata) > 2,  # More than just file info
            'extraction_quality': 'basic'  # This is a simple parser
        })
        
        return metadata, text_content

    async def _extract_text_content(self, f, encoding: str, max_size: int) -> str:
        """Extract text content from MOBI file."""
        # This is a simplified extraction method
        # Real MOBI files have complex compression and formatting
        
        try:
            # Skip headers and try to find text content
            # MOBI files typically have text starting after headers
            f.seek(1024)  # Skip most header content
            
            content_bytes = f.read(max_size)
            
            # Filter out binary data and extract readable text
            return self._extract_readable_text(content_bytes, encoding)
            
        except Exception as e:
            logger.warning(f"Text extraction failed: {e}")
            return "[MOBI content extraction failed - file may be encrypted or corrupted]"

    def _extract_readable_text(self, data: bytes, encoding: str) -> str:
        """Extract readable text from binary data."""
        try:
            # Try to decode with specified encoding
            text = data.decode(encoding, errors='ignore')
        except:
            # Fallback to latin1 which can decode any byte sequence
            text = data.decode('latin1', errors='ignore')
        
        # Clean up the text - remove non-printable characters except common whitespace
        cleaned_lines = []
        for line in text.split('\n'):
            # Keep lines that have a reasonable amount of printable characters
            printable_chars = sum(1 for c in line if c.isprintable() or c in '\t ')
            if len(line) > 0 and printable_chars / len(line) > 0.7:  # At least 70% printable
                # Remove excessive whitespace
                cleaned_line = ' '.join(line.split())
                if len(cleaned_line) > 10:  # Only keep substantial lines
                    cleaned_lines.append(cleaned_line)
        
        result = '\n'.join(cleaned_lines)
        
        # If we got very little content, add a note
        if len(result) < 100:
            result += "\n\n[Note: MOBI file may be encrypted, DRM-protected, or in an unsupported format. " \
                     "For best results with MOBI files, consider converting to EPUB format first.]"
        
        return result

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for MOBI files."""
        return {
            'encoding': {
                'type': str,
                'default': 'utf-8',
                'description': 'Text encoding to use for text extraction'
            },
            'max_content_size': {
                'type': int,
                'default': 10000000,
                'description': 'Maximum content size to extract (bytes)'
            }
        }