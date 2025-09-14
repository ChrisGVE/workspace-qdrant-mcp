from common.logging.loguru_config import get_logger

"""
Plain text document parser.

This parser handles plain text files (.txt) and other text-based formats,
providing encoding detection, content cleaning, and basic text analysis
for the workspace-qdrant-mcp ingestion system.
"""

import logging
from pathlib import Path
from typing import Any, Optional, Union

import chardet

from .base import DocumentParser, ParsedDocument
from .exceptions import EncodingError, handle_parsing_error
from .progress import ProgressPhase, ProgressTracker, ProgressUnit

logger = get_logger(__name__)


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
            ".json",
            ".xml",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
            ".conf",
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
        return "Plain Text"

    async def parse(
        self,
        file_path: str | Path,
        progress_tracker: Optional[ProgressTracker] = None,
        encoding: str | None = None,
        detect_encoding: bool = True,
        clean_content: bool = True,
        preserve_whitespace: bool = False,
        **options,
    ) -> ParsedDocument:
        """
        Parse a plain text file.

        Args:
            file_path: Path to the text file
            progress_tracker: Optional progress tracker for monitoring
            encoding: Specific encoding to use (if None, auto-detect)
            detect_encoding: Whether to auto-detect encoding if not specified
            clean_content: Whether to normalize and clean the text content
            preserve_whitespace: Whether to preserve original whitespace formatting
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

    async def _detect_encoding(self, file_path: Path) -> tuple[str, float]:
        """
        Detect the character encoding of a text file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (encoding_name, confidence_score)
        """
        try:
            # Read a sample of the file for encoding detection
            with open(file_path, "rb") as f:
                raw_data = f.read(8192)  # Read first 8KB for detection

            detection = chardet.detect(raw_data)
            encoding = detection.get("encoding", "utf-8")
            confidence = detection.get("confidence", 0.0)

            # Fallback to utf-8 if confidence is too low
            if confidence < 0.7:
                logger.warning(
                    f"Low confidence ({confidence:.2f}) encoding detection for {file_path}. "
                    f"Detected: {encoding}, using utf-8 as fallback."
                )
                encoding = "utf-8"
                confidence = 0.5  # Mark as fallback

            return encoding, confidence

        except Exception as e:
            logger.warning(f"Encoding detection failed for {file_path}: {e}")
            return "utf-8", 0.0  # Safe fallback

    async def _read_with_encoding(
        self,
        file_path: Path,
        encoding: str,
        progress_tracker: Optional[ProgressTracker] = None,
    ) -> str:
        """
        Read file content with specified encoding.

        Args:
            file_path: Path to the file
            encoding: Character encoding to use

        Returns:
            File content as string

        Raises:
            RuntimeError: If file cannot be read with the specified encoding
        """
        try:
            if progress_tracker:
                progress_tracker.update(0, f"Reading with {encoding} encoding")

            with open(file_path, encoding=encoding, errors="replace") as f:
                content = f.read()
                if progress_tracker:
                    progress_tracker.update(len(content.encode("utf-8")), "File loaded")
                return content
        except Exception as e:
            # Try with error handling for corrupted files
            try:
                with open(file_path, encoding=encoding, errors="ignore") as f:
                    content = f.read()
                    logger.warning(
                        f"File {file_path} had encoding errors, some characters were ignored"
                    )
                    if progress_tracker:
                        progress_tracker.add_warning()
                        progress_tracker.update(
                            len(content.encode("utf-8")), "File loaded with warnings"
                        )
                    return content
            except Exception as inner_e:
                # Raise as EncodingError for proper classification
                raise EncodingError(
                    f"Unable to read file {file_path} with encoding {encoding}: {e}",
                    file_path=file_path,
                    attempted_encodings=[encoding],
                    original_exception=inner_e,
                ) from inner_e

    def _clean_content(self, content: str, preserve_whitespace: bool = False) -> str:
        """
        Clean and normalize text content.

        Args:
            content: Raw text content
            preserve_whitespace: Whether to preserve original whitespace

        Returns:
            Cleaned text content
        """
        if not content:
            return content

        # Remove null bytes and other control characters
        content = content.replace("\x00", "")

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
        }
