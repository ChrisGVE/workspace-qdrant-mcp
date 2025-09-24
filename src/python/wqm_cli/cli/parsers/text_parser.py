from loguru import logger

"""
Plain text document parser.

This parser handles plain text files (.txt) and other text-based formats,
providing encoding detection, content cleaning, and basic text analysis
for the workspace-qdrant-mcp ingestion system.
"""

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import chardet

try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None

from .base import DocumentParser, ParsedDocument
from .exceptions import EncodingError, handle_parsing_error
from .progress import ProgressPhase, ProgressTracker, ProgressUnit

# logger imported from loguru


class TextParser(DocumentParser):
    """
    Parser for plain text documents.

    Handles various text file formats with automatic encoding detection,
    content validation, and basic text analysis. Supports common text
    encodings and provides options for content preprocessing.

    Features:
        - Automatic encoding detection using chardet
        - Support for various text file extensions
        - Content cleaning and normalization options
        - Basic text statistics and analysis
        - Graceful handling of encoding issues
    """

    @property
    def supported_extensions(self) -> list[str]:
        """Supported text file extensions."""
        return [
            ".txt",
            ".text",
            ".log",
            ".csv",
            ".tsv",
            ".rtf",
            ".json",
            ".jsonl",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
            ".properties",
            ".env",
            ".py",
            ".js",
            ".html",
            ".css",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            ".bat",
            ".cmd",
        ]

    @property
    def format_name(self) -> str:
        """Human-readable format name."""
        return "Text Document"

    async def parse(
        self,
        file_path: str | Path,
        progress_tracker: Optional[ProgressTracker] = None,
        encoding: str | None = None,
        detect_encoding: bool = True,
        clean_content: bool = True,
        preserve_whitespace: bool = False,
        enable_structured_parsing: bool = True,
        csv_delimiter: Optional[str] = None,
        csv_has_header: bool = True,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB default limit
        **options,
    ) -> ParsedDocument:
        """
        Parse a text document with comprehensive format support.

        Args:
            file_path: Path to the text file
            progress_tracker: Optional progress tracker for monitoring
            encoding: Specific encoding to use (if None, auto-detect)
            detect_encoding: Whether to auto-detect encoding if not specified
            clean_content: Whether to normalize and clean the text content
            preserve_whitespace: Whether to preserve original whitespace formatting
            enable_structured_parsing: Whether to parse structured formats (CSV, LOG, etc.)
            csv_delimiter: Delimiter for CSV files (auto-detected if None)
            csv_has_header: Whether CSV files have headers
            max_file_size: Maximum file size to process (bytes)
            **options: Additional parsing options

        Returns:
            ParsedDocument with extracted text content and metadata

        Raises:
            ParsingError: If parsing fails (wraps specific error types)
        """
        file_path = Path(file_path)
        self.validate_file(file_path)

        # Setup progress tracking
        if progress_tracker is None:
            file_size = file_path.stat().st_size if file_path.exists() else 0
            progress_tracker = ProgressTracker(total=file_size, unit=ProgressUnit.BYTES)

        progress_tracker.set_file_info(file_path, file_path.stat().st_size)
        progress_tracker.set_phase(ProgressPhase.DETECTING_TYPE)

        parsing_info: dict[str, str | int | float] = {}

        try:
            progress_tracker.set_phase(ProgressPhase.LOADING)

            # Read file content with encoding detection
            if encoding:
                content = await self._read_with_encoding(
                    file_path, encoding, progress_tracker
                )
                parsing_info["encoding"] = encoding
                parsing_info["encoding_detection"] = "specified"
            elif detect_encoding:
                progress_tracker.update(0, "Detecting encoding")
                encoding, confidence = await self._detect_encoding(file_path)
                content = await self._read_with_encoding(
                    file_path, encoding, progress_tracker
                )
                parsing_info["encoding"] = encoding
                parsing_info["encoding_confidence"] = confidence
                parsing_info["encoding_detection"] = "auto-detected"
            else:
                # Fallback to UTF-8
                content = await self._read_with_encoding(
                    file_path, "utf-8", progress_tracker
                )
                parsing_info["encoding"] = "utf-8"
                parsing_info["encoding_detection"] = "default"

            # Content processing
            progress_tracker.set_phase(ProgressPhase.PROCESSING)
            original_length = len(content)
            progress_tracker.update(
                progress_tracker.metrics.total // 2, "Processing content"
            )

            if clean_content:
                progress_tracker.update(
                    progress_tracker.metrics.total * 3 // 4, "Cleaning content"
                )
                content = self._clean_content(content, preserve_whitespace)
                parsing_info["content_cleaned"] = True
                parsing_info["size_reduction"] = original_length - len(content)

            # Generate text statistics
            progress_tracker.set_phase(ProgressPhase.ANALYZING)
            text_stats = self._analyze_text(content)
            parsing_info.update(text_stats)

            progress_tracker.set_phase(ProgressPhase.FINALIZING)
            progress_tracker.update(progress_tracker.metrics.total, "Creating document")

            # Create metadata
            additional_metadata: dict[str, str | int | float | bool] = {
                "parser": self.format_name,
                "encoding": parsing_info.get("encoding", "utf-8"),
                "word_count": text_stats.get("word_count", 0),
                "character_count": len(content),
                "paragraph_count": text_stats.get("paragraph_count", 0),
            }

            # Add file-type specific metadata
            file_extension = file_path.suffix.lower()
            if file_extension in [".py", ".js", ".html", ".css", ".sql"]:
                additional_metadata["content_type"] = "code"
                additional_metadata["language"] = self._detect_language(file_extension)
            elif file_extension in [".json", ".xml", ".yaml", ".yml"]:
                additional_metadata["content_type"] = "structured_data"
            elif file_extension in [".log"]:
                additional_metadata["content_type"] = "log_file"
            else:
                additional_metadata["content_type"] = "plain_text"

            parsed_doc = ParsedDocument.create(
                content=content,
                file_path=file_path,
                file_type="text",
                additional_metadata=additional_metadata,
                parsing_info=parsing_info,
            )

            # Mark progress as completed
            progress_tracker.set_phase(ProgressPhase.COMPLETED)

            return parsed_doc

        except Exception as e:
            logger.error(f"Failed to parse text file {file_path}: {e}")
            if progress_tracker:
                progress_tracker.set_phase(ProgressPhase.FAILED)
            # Use error handler for consistent error processing
            raise handle_parsing_error(e, file_path)

    async def _detect_encoding_comprehensive(self, file_path: Path) -> tuple[str, float]:
        """
        Comprehensive encoding detection with multiple algorithms and fallbacks.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (encoding_name, confidence_score)
        """
        try:
            # Read sample data for encoding detection (larger sample for better accuracy)
            with open(file_path, "rb") as f:
                raw_data = f.read(32768)  # Read first 32KB for detection

            # Primary detection with chardet
            chardet_result = chardet.detect(raw_data)
            chardet_encoding = chardet_result.get("encoding", "utf-8")
            chardet_confidence = chardet_result.get("confidence", 0.0)

            # Secondary detection with charset-normalizer if available
            charset_normalizer_encoding = None
            charset_normalizer_confidence = 0.0

            if charset_normalizer:
                try:
                    results = charset_normalizer.from_bytes(raw_data)
                    if results:
                        charset_normalizer_encoding = results.best().encoding
                        charset_normalizer_confidence = results.best().chaos
                except Exception:
                    pass

            # Choose best detection result
            final_encoding, final_confidence = self._select_best_encoding(
                (chardet_encoding, chardet_confidence),
                (charset_normalizer_encoding, charset_normalizer_confidence),
                raw_data
            )

            logger.debug(
                f"Encoding detection for {file_path}: {final_encoding} "
                f"(confidence: {final_confidence:.2f})"
            )

            return final_encoding, final_confidence

        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8", 0.0  # Safe fallback

    async def _read_with_encoding_robust(
        self,
        file_path: Path,
        encoding: str,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> tuple[str, str]:
        """
        Robustly read file content with encoding fallback chain.

        Args:
            file_path: Path to the file
            encoding: Character encoding to use initially
            progress_tracker: Optional progress tracker

        Returns:
            Tuple of (file_content, actual_encoding_used)

        Raises:
            EncodingError: If file cannot be read with any encoding
        """
        # Define encoding fallback chain
        encoding_chain = [encoding, "utf-8", "latin-1", "ascii", "windows-1252"]
        # Remove duplicates while preserving order
        seen = set()
        encoding_chain = [x for x in encoding_chain if not (x in seen or seen.add(x))]

        attempted_encodings = []
        last_error = None

        for attempt_encoding in encoding_chain:
            try:
                if progress_tracker:
                    progress_tracker.update(0, f"Trying {attempt_encoding} encoding")

                # Try strict decoding first
                try:
                    with open(file_path, encoding=attempt_encoding, errors="strict") as f:
                        content = f.read()
                        if progress_tracker:
                            progress_tracker.update(len(content.encode("utf-8")), "File loaded")
                        return content, attempt_encoding
                except UnicodeDecodeError:
                    # Fall back to replace mode for this encoding
                    with open(file_path, encoding=attempt_encoding, errors="replace") as f:
                        content = f.read()
                        logger.debug(
                            f"File {file_path} loaded with {attempt_encoding} encoding (replace mode)"
                        )
                        if progress_tracker:
                            progress_tracker.add_warning()
                            progress_tracker.update(
                                len(content.encode("utf-8")), f"File loaded with {attempt_encoding}"
                            )
                        return content, attempt_encoding

            except Exception as e:
                attempted_encodings.append(attempt_encoding)
                last_error = e
                logger.debug(f"Failed to read {file_path} with {attempt_encoding}: {e}")
                continue

        # All encodings failed
        raise EncodingError(
            f"Unable to read file {file_path} with any encoding. Tried: {', '.join(attempted_encodings)}",
            file_path=file_path,
            attempted_encodings=attempted_encodings,
            original_exception=last_error,
        )

    def _clean_content(self, content: str, preserve_whitespace: bool = False, file_extension: str = "") -> str:
        """
        Clean and normalize text content with format-aware processing.

        Args:
            content: Raw text content
            preserve_whitespace: Whether to preserve original whitespace
            file_extension: File extension for format-specific cleaning

        Returns:
            Cleaned text content
        """
        if not content:
            return content

        # Remove null bytes and other problematic control characters
        content = content.replace("\x00", "")

        # Remove other problematic control characters but preserve useful ones
        # Keep: \t (tab), \n (newline), \r (carriage return)
        import string
        allowed_chars = string.printable
        if file_extension in [".csv", ".tsv"]:
            # For structured formats, be more permissive with special characters
            content = ''.join(char for char in content if char in allowed_chars or ord(char) > 127)
        else:
            # For plain text, be more aggressive in cleaning
            content = ''.join(char for char in content if char.isprintable() or char in '\t\n\r')

        # Normalize line endings
        content = content.replace("\r\n", "\n").replace("\r", "\n")

        if not preserve_whitespace:
            # Remove excessive whitespace
            lines = []
            for line in content.split("\n"):
                # Strip trailing whitespace but preserve some indentation
                stripped = line.rstrip()
                if stripped or len(lines) == 0 or lines[-1].strip():
                    lines.append(stripped)

            # Remove excessive empty lines (max 2 consecutive)
            cleaned_lines = []
            empty_count = 0
            for line in lines:
                if line.strip():
                    cleaned_lines.append(line)
                    empty_count = 0
                elif empty_count < 2:
                    cleaned_lines.append(line)
                    empty_count += 1

            content = "\n".join(cleaned_lines)

        # Format-specific final cleaning
        if file_extension in [".csv", ".tsv"]:
            # Don't strip CSV/TSV files as leading/trailing whitespace might be meaningful
            return content
        elif file_extension == ".log":
            # For log files, preserve line structure but remove excessive blank lines
            lines = content.split("\n")
            cleaned_lines = []
            blank_count = 0
            for line in lines:
                if line.strip():
                    cleaned_lines.append(line)
                    blank_count = 0
                elif blank_count < 1:  # Allow max 1 consecutive blank line
                    cleaned_lines.append(line)
                    blank_count += 1
            return "\n".join(cleaned_lines)
        else:
            return content.strip()

    def _analyze_text(self, content: str) -> dict[str, int]:
        """
        Analyze text content and generate statistics.

        Args:
            content: Text content to analyze

        Returns:
            Dictionary with text analysis results
        """
        if not content:
            return {
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0,
                "character_count": 0,
                "non_whitespace_chars": 0,
            }

        lines = content.split("\n")
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        words = content.split()

        return {
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "character_count": len(content),
            "non_whitespace_chars": len(
                content.replace(" ", "").replace("\n", "").replace("\t", "")
            ),
        }

    def _detect_language(self, extension: str) -> str:
        """Detect programming language from file extension."""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".html": "html",
            ".css": "css",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".zsh": "zsh",
            ".fish": "fish",
            ".ps1": "powershell",
            ".bat": "batch",
            ".cmd": "batch",
        }
        return language_map.get(extension.lower(), "text")

    def _select_best_encoding(
        self,
        chardet_result: tuple[str, float],
        charset_normalizer_result: tuple[Optional[str], float],
        raw_data: bytes,
    ) -> tuple[str, float]:
        """
        Select the best encoding result from multiple detection methods.

        Args:
            chardet_result: (encoding, confidence) from chardet
            charset_normalizer_result: (encoding, confidence) from charset-normalizer
            raw_data: Raw file data for additional validation

        Returns:
            Tuple of (best_encoding, confidence)
        """
        chardet_encoding, chardet_confidence = chardet_result
        charset_normalizer_encoding, charset_normalizer_confidence = charset_normalizer_result

        # If charset-normalizer is available and has high confidence, prefer it
        if (
            charset_normalizer_encoding
            and charset_normalizer_confidence > 0.8
            and charset_normalizer_confidence > chardet_confidence
        ):
            return charset_normalizer_encoding, charset_normalizer_confidence

        # If chardet has high confidence, use it
        if chardet_confidence > 0.7:
            return chardet_encoding, chardet_confidence

        # For low confidence, apply heuristics
        return self._apply_encoding_heuristics(
            chardet_encoding, chardet_confidence, raw_data
        )

    def _apply_encoding_heuristics(
        self, encoding: str, confidence: float, raw_data: bytes
    ) -> tuple[str, float]:
        """
        Apply heuristic rules for encoding detection when confidence is low.

        Args:
            encoding: Detected encoding
            confidence: Detection confidence
            raw_data: Raw file data

        Returns:
            Tuple of (heuristic_encoding, adjusted_confidence)
        """
        # Check for BOM markers
        if raw_data.startswith(b'\xff\xfe'):
            return "utf-16-le", 0.9
        elif raw_data.startswith(b'\xfe\xff'):
            return "utf-16-be", 0.9
        elif raw_data.startswith(b'\xef\xbb\xbf'):
            return "utf-8-sig", 0.9

        # Check for common patterns
        try:
            # Test if it's valid UTF-8
            raw_data.decode('utf-8')
            return "utf-8", max(confidence, 0.8)
        except UnicodeDecodeError:
            pass

        # Check for Windows-1252 specific characters
        if any(b in raw_data for b in [b'\x80', b'\x82', b'\x83', b'\x84']):
            return "windows-1252", max(confidence, 0.6)

        # Default fallback with slightly better confidence
        return encoding or "utf-8", max(confidence, 0.5)

    def get_parsing_options(self) -> dict[str, dict[str, Any]]:
        """Get available parsing options for text files."""
        return {
            "encoding": {
                "type": str,
                "default": None,
                "description": "Character encoding to use (auto-detected if None)",
            },
            "detect_encoding": {
                "type": bool,
                "default": True,
                "description": "Whether to auto-detect file encoding",
            },
            "clean_content": {
                "type": bool,
                "default": True,
                "description": "Whether to clean and normalize text content",
            },
            "preserve_whitespace": {
                "type": bool,
                "default": False,
                "description": "Whether to preserve original whitespace formatting",
            },
            "enable_structured_parsing": {
                "type": bool,
                "default": True,
                "description": "Whether to parse structured formats (CSV, LOG, etc.)",
            },
            "csv_delimiter": {
                "type": str,
                "default": None,
                "description": "Delimiter for CSV files (auto-detected if None)",
            },
            "csv_has_header": {
                "type": bool,
                "default": True,
                "description": "Whether CSV files have headers",
            },
            "max_file_size": {
                "type": int,
                "default": 100 * 1024 * 1024,
                "description": "Maximum file size to process (bytes)",
            },
        }
