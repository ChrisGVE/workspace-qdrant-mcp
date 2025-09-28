"""
PDF document parser.

This parser handles PDF files using PyPDF2 for text extraction with support
for metadata extraction, multi-page processing, and content analysis.
Provides fallback handling for encrypted or corrupted PDFs.
"""

from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger

# Optional OCR detection dependencies
try:
    from PIL import Image
    import fitz  # PyMuPDF for image extraction and OCR detection
    HAS_OCR_DEPS = True
except ImportError:
    HAS_OCR_DEPS = False

# State management import
try:
    from ...common.core.sqlite_state_manager import SQLiteStateManager, FileProcessingStatus
    HAS_STATE_MANAGER = True
except ImportError:
    HAS_STATE_MANAGER = False
    SQLiteStateManager = None
    FileProcessingStatus = None

try:
    import pypdf

    # Alias for tests that expect PyPDF2
    PyPDF2 = pypdf
    HAS_PYPDF = True
    HAS_PYPDF2 = True  # For test compatibility
except ImportError:
    # Create a dummy PyPDF2 for test compatibility
    class PyPDF2:
        pass

    HAS_PYPDF = False
    HAS_PYPDF2 = False  # For test compatibility

from .base import DocumentParser, ParsedDocument
from .progress import ProgressTracker

# logger imported from loguru


class PDFParser(DocumentParser):
    """
    Enhanced parser for PDF documents with OCR detection.

    Handles PDF files with support for:
        - Multi-page text extraction
        - PDF metadata extraction (title, author, creation date, etc.)
        - Handling of encrypted PDFs (if possible)
        - Page-by-page processing for large documents
        - Content analysis and statistics
        - OCR detection for image-based PDFs
        - State management for OCR-required documents
        - Automatic fallback to OCR recommendations

    Uses PyPDF2/pypdf for text extraction and PyMuPDF for OCR detection.
    Integrates with state management for tracking OCR-required documents.
    """

    def __init__(self, state_manager: Optional[SQLiteStateManager] = None):
        """
        Initialize PDF parser with optional state management.

        Args:
            state_manager: Optional SQLite state manager for OCR tracking
        """
        self.state_manager = state_manager
        self._check_ocr_availability()

    def _check_ocr_availability(self) -> None:
        """Check if OCR detection dependencies are available."""
        if not HAS_OCR_DEPS:
            logger.debug("OCR detection dependencies not available (PIL, PyMuPDF)")
        if not HAS_STATE_MANAGER:
            logger.debug("State manager not available - OCR tracking disabled")

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
        progress_tracker: Optional[ProgressTracker] = None,
        extract_metadata: bool = True,
        include_page_numbers: bool = False,
        max_pages: int | None = None,
        password: str | None = None,
        detect_ocr_needed: bool = True,
        ocr_confidence_threshold: float = 0.1,
        **options,
    ) -> ParsedDocument:
        """
        Parse a PDF file with OCR detection and state management.

        Args:
            file_path: Path to the PDF file
            progress_tracker: Optional progress tracker for monitoring
            extract_metadata: Whether to extract PDF metadata
            include_page_numbers: Whether to include page numbers in content
            max_pages: Maximum number of pages to process (None for all)
            password: Password for encrypted PDFs
            detect_ocr_needed: Whether to detect if OCR is needed for image-based PDFs
            ocr_confidence_threshold: Minimum text confidence to avoid OCR recommendation
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

        parsing_info: dict[str, str | int | float | bool] = {}

        if progress_tracker:
            progress_tracker.update_phase("initializing", "Setting up PDF parsing")

        try:
            with open(file_path, "rb") as pdf_file:
                # Create PDF reader
                if progress_tracker:
                    progress_tracker.update_phase("reading", "Opening PDF file")
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
                if progress_tracker:
                    progress_tracker.update_phase("extracting", "Extracting text from pages")
                content_parts = []
                pages_processed = 0
                text_extraction_stats = {
                    "pages_with_text": 0,
                    "pages_with_minimal_text": 0,
                    "total_extracted_chars": 0,
                }

                for page_num, page in enumerate(pdf_reader.pages):
                    if max_pages and page_num >= max_pages:
                        break

                    try:
                        page_text = page.extract_text()
                        text_length = len(page_text.strip())
                        text_extraction_stats["total_extracted_chars"] += text_length

                        if page_text.strip():  # Only include non-empty pages
                            if text_length > 50:  # Substantial text
                                text_extraction_stats["pages_with_text"] += 1
                            else:  # Minimal text (might be image-based)
                                text_extraction_stats["pages_with_minimal_text"] += 1

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
                if progress_tracker:
                    progress_tracker.update_phase("cleaning", "Cleaning extracted text")
                content = self._clean_pdf_text(content)

                # OCR detection and state management
                ocr_needed = False
                ocr_confidence = 1.0
                if detect_ocr_needed:
                    if content:
                        if progress_tracker:
                            progress_tracker.update_phase("ocr_detection", "Analyzing OCR requirements")
                        ocr_needed, ocr_confidence = await self._detect_ocr_needed(
                            file_path, content, text_extraction_stats, ocr_confidence_threshold
                        )
                        parsing_info["ocr_needed"] = ocr_needed
                        parsing_info["text_confidence"] = ocr_confidence

                        # Store OCR requirement in state database if needed
                        if ocr_needed and self.state_manager and HAS_STATE_MANAGER:
                            await self._record_ocr_requirement(file_path, ocr_confidence)
                    elif not content:
                        # No text extracted - likely needs OCR
                        ocr_needed = True
                        ocr_confidence = 0.0
                        parsing_info["ocr_needed"] = True
                        parsing_info["text_confidence"] = 0.0
                        if self.state_manager and HAS_STATE_MANAGER:
                            await self._record_ocr_requirement(file_path, 0.0)

                # Generate text analysis
                text_stats = self._analyze_pdf_content(content, pages_processed)
                parsing_info.update(text_stats)

                # Create comprehensive metadata
                if progress_tracker:
                    progress_tracker.update_phase("finalizing", "Creating metadata")
                additional_metadata: dict[str, str | int | float | bool] = {
                    "parser": self.format_name,
                    "page_count": pages_processed,
                    "total_pages": num_pages,
                    "word_count": len(content.split()) if content else 0,
                    "character_count": len(content),
                    "avg_words_per_page": len(content.split()) / pages_processed
                    if pages_processed > 0 and content
                    else 0.0,
                    "pages_with_text": text_extraction_stats["pages_with_text"],
                    "pages_with_minimal_text": text_extraction_stats["pages_with_minimal_text"],
                }

                # Add OCR metadata only if detection was enabled
                if detect_ocr_needed:
                    additional_metadata["ocr_needed"] = ocr_needed
                    additional_metadata["text_confidence"] = ocr_confidence

                # Add PDF metadata
                additional_metadata.update(pdf_metadata)

                # Add parsing statistics
                if parsing_info.get("empty_pages", 0) > 0:
                    additional_metadata["empty_pages"] = parsing_info["empty_pages"]

                # Add OCR recommendation to content if needed (only when detection is enabled)
                if detect_ocr_needed and ocr_needed:
                    if content:
                        content = f"[OCR RECOMMENDED: This PDF appears to contain image-based text with low confidence ({ocr_confidence:.2f}). Consider using OCR for better text extraction.]\n\n{content}"
                    else:
                        content = "[OCR REQUIRED: This PDF contains no extractable text and likely consists of scanned images. OCR processing is required to extract text content.]"

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
            "detect_ocr_needed": {
                "type": bool,
                "default": True,
                "description": "Whether to detect if OCR is needed for image-based PDFs",
            },
            "ocr_confidence_threshold": {
                "type": float,
                "default": 0.1,
                "description": "Minimum text confidence to avoid OCR recommendation",
            },
        }

    async def _detect_ocr_needed(
        self,
        file_path: Path,
        extracted_text: str,
        text_stats: dict[str, int],
        confidence_threshold: float = 0.1,
    ) -> tuple[bool, float]:
        """
        Detect if a PDF requires OCR by analyzing extracted text and images.

        This method uses multiple heuristics to determine if a PDF contains
        primarily image-based content that would benefit from OCR processing:
        1. Text extraction ratio vs. total pages
        2. Average text per page analysis
        3. Image content analysis using PyMuPDF
        4. Text confidence scoring

        Args:
            file_path: Path to the PDF file
            extracted_text: Text extracted by pypdf
            text_stats: Statistics about text extraction
            confidence_threshold: Minimum confidence to avoid OCR recommendation

        Returns:
            Tuple of (ocr_needed: bool, confidence: float)
        """
        try:
            # Quick checks first
            if not extracted_text or len(extracted_text.strip()) < 10:
                logger.debug(f"PDF {file_path.name}: No meaningful text extracted")
                return True, 0.0

            # Text density analysis
            total_chars = text_stats.get("total_extracted_chars", len(extracted_text))
            pages_with_text = text_stats.get("pages_with_text", 0)
            pages_with_minimal = text_stats.get("pages_with_minimal_text", 0)
            total_pages = text_stats.get("total_pages", 1)

            # Calculate text confidence metrics
            text_density = total_chars / total_pages if total_pages > 0 else 0
            substantial_text_ratio = pages_with_text / total_pages if total_pages > 0 else 0
            minimal_text_ratio = pages_with_minimal / total_pages if total_pages > 0 else 0

            # Base confidence from text statistics
            text_confidence = min(1.0, text_density / 500)  # 500 chars = reasonable page
            text_confidence *= substantial_text_ratio  # Weight by substantial text ratio

            logger.debug(
                f"PDF {file_path.name} text analysis: "
                f"density={text_density:.1f}, substantial_ratio={substantial_text_ratio:.2f}, "
                f"minimal_ratio={minimal_text_ratio:.2f}, confidence={text_confidence:.2f}"
            )

            # If we have OCR dependencies, do deeper analysis
            if HAS_OCR_DEPS and text_confidence < 0.8:
                image_confidence = await self._analyze_pdf_images(file_path)
                # Combine text and image analysis
                final_confidence = (text_confidence * 0.7) + (image_confidence * 0.3)
                logger.debug(
                    f"PDF {file_path.name} image analysis: "
                    f"image_confidence={image_confidence:.2f}, final={final_confidence:.2f}"
                )
            else:
                final_confidence = text_confidence

            # OCR recommendation logic
            needs_ocr = final_confidence < confidence_threshold

            # Additional heuristics for edge cases
            if not needs_ocr and minimal_text_ratio > 0.3:
                # Many pages with minimal text might indicate image-based content
                logger.debug(f"PDF {file_path.name}: High minimal text ratio, flagging for OCR")
                needs_ocr = True
                final_confidence = min(final_confidence, 0.05)

            return needs_ocr, final_confidence

        except Exception as e:
            logger.warning(f"OCR detection failed for {file_path}: {e}")
            # Conservative approach: if we can't analyze, assume text is good
            return False, 0.5

    async def _analyze_pdf_images(self, file_path: Path) -> float:
        """
        Analyze PDF for image content using PyMuPDF.

        This method examines the PDF for images and attempts to estimate
        how much of the content is image-based vs. text-based.

        Args:
            file_path: Path to the PDF file

        Returns:
            Confidence score (0.0 = all images, 1.0 = mostly text)
        """
        if not HAS_OCR_DEPS:
            return 0.5  # Neutral confidence without analysis tools

        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            total_area = 0
            image_area = 0
            text_area = 0
            pages_analyzed = 0

            # Analyze up to first 10 pages to avoid performance issues
            max_analyze_pages = min(10, len(doc))

            for page_num in range(max_analyze_pages):
                page = doc[page_num]
                page_rect = page.rect
                page_area = page_rect.width * page_rect.height
                total_area += page_area
                pages_analyzed += 1

                # Get images on this page
                image_list = page.get_images(full=True)
                page_image_area = 0

                for img_index, img in enumerate(image_list):
                    try:
                        # Get image object
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Get image dimensions and position
                        image_rects = page.get_image_rects(img)
                        for rect in image_rects:
                            img_area = rect.width * rect.height
                            page_image_area += img_area

                    except Exception as e:
                        logger.debug(f"Could not analyze image {img_index} on page {page_num}: {e}")
                        continue

                image_area += page_image_area

                # Estimate text area (page area minus image area)
                page_text_area = max(0, page_area - page_image_area)
                text_area += page_text_area

                logger.debug(
                    f"Page {page_num + 1}: area={page_area:.0f}, "
                    f"images={page_image_area:.0f}, text={page_text_area:.0f}"
                )

            doc.close()

            if total_area == 0:
                return 0.5

            # Calculate confidence based on text vs image area ratio
            text_ratio = text_area / total_area if total_area > 0 else 0
            image_ratio = image_area / total_area if total_area > 0 else 0

            # Text confidence: higher when more text area, lower when more images
            confidence = text_ratio / (text_ratio + image_ratio) if (text_ratio + image_ratio) > 0 else 0.5

            logger.debug(
                f"PDF image analysis: {pages_analyzed} pages, "
                f"text_ratio={text_ratio:.2f}, image_ratio={image_ratio:.2f}, "
                f"confidence={confidence:.2f}"
            )

            return confidence

        except Exception as e:
            logger.warning(f"PDF image analysis failed for {file_path}: {e}")
            return 0.5  # Neutral confidence on analysis failure

    async def _record_ocr_requirement(
        self, file_path: Path, confidence: float
    ) -> None:
        """
        Record OCR requirement in state database.

        This method stores information about PDFs that require OCR processing
        so that users can be notified and batch processing can be implemented.

        Args:
            file_path: Path to the PDF file requiring OCR
            confidence: Text confidence score (0.0 = definitely needs OCR)
        """
        if not self.state_manager or not HAS_STATE_MANAGER:
            logger.debug("State manager not available - cannot record OCR requirement")
            return

        try:
            # Record the OCR requirement with metadata
            ocr_metadata = {
                "text_confidence": confidence,
                "requires_ocr": True,
                "analysis_date": self._get_current_timestamp(),
                "file_size": file_path.stat().st_size,
                "file_type": "pdf",
                "recommendation": (
                    "OCR required - no extractable text" if confidence == 0.0
                    else f"OCR recommended - low text confidence ({confidence:.2f})"
                ),
            }

            # Store in state database
            await self.state_manager.record_file_status(
                file_path=str(file_path),
                status=FileProcessingStatus.OCR_REQUIRED,
                metadata=ocr_metadata,
            )

            logger.info(
                f"Recorded OCR requirement for {file_path.name} "
                f"(confidence: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to record OCR requirement for {file_path}: {e}")

    def _get_current_timestamp(self) -> str:
        """
        Get current timestamp in ISO format.

        Returns:
            ISO formatted timestamp string
        """
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    async def get_ocr_queue(self) -> list[dict[str, Any]]:
        """
        Retrieve list of files requiring OCR processing.

        Returns:
            List of dictionaries containing file information and OCR metadata
        """
        if not self.state_manager or not HAS_STATE_MANAGER:
            logger.warning("State manager not available - cannot retrieve OCR queue")
            return []

        try:
            ocr_files = await self.state_manager.get_files_by_status(
                FileProcessingStatus.OCR_REQUIRED
            )

            return [
                {
                    "file_path": file_record["file_path"],
                    "confidence": file_record.get("metadata", {}).get("text_confidence", 0.0),
                    "file_size": file_record.get("metadata", {}).get("file_size", 0),
                    "analysis_date": file_record.get("metadata", {}).get("analysis_date", ""),
                    "recommendation": file_record.get("metadata", {}).get("recommendation", ""),
                }
                for file_record in ocr_files
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve OCR queue: {e}")
            return []
