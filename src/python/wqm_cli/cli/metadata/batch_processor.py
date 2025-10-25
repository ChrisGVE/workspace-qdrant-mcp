"""
Batch processing system for document metadata workflow.

This module provides efficient batch processing capabilities for handling
multiple documents, supporting progress tracking, error recovery, and
parallel processing for large document collections.
"""

import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from loguru import logger

from ..parsers.base import DocumentParser
from ..parsers.file_detector import detect_file_type
from ..parsers.progress import ProgressTracker, create_progress_tracker
from .aggregator import DocumentMetadata, MetadataAggregator
from .exceptions import BatchProcessingError


class BatchConfig:
    """Configuration for batch processing operations."""

    def __init__(
        self,
        batch_size: int = 50,
        max_workers: int = 4,
        timeout_seconds: float = 300.0,
        continue_on_error: bool = True,
        progress_reporting: bool = True,
        parallel_processing: bool = True,
        max_memory_usage_mb: int | None = None,
        retry_failed: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize batch configuration.

        Args:
            batch_size: Number of documents to process in each batch
            max_workers: Maximum number of worker threads
            timeout_seconds: Timeout for individual document processing
            continue_on_error: Continue processing if individual documents fail
            progress_reporting: Enable progress reporting
            parallel_processing: Enable parallel processing
            max_memory_usage_mb: Maximum memory usage in MB (None for no limit)
            retry_failed: Retry failed documents
            max_retries: Maximum retry attempts for failed documents
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.continue_on_error = continue_on_error
        self.progress_reporting = progress_reporting
        self.parallel_processing = parallel_processing
        self.max_memory_usage_mb = max_memory_usage_mb
        self.retry_failed = retry_failed
        self.max_retries = max_retries


class BatchResult:
    """Results from batch processing operation."""

    def __init__(
        self,
        successful_documents: list[DocumentMetadata],
        failed_documents: list[tuple[str, str]],  # (file_path, error_message)
        processing_stats: dict[str, Any],
    ):
        """
        Initialize batch result.

        Args:
            successful_documents: List of successfully processed document metadata
            failed_documents: List of failed documents with error messages
            processing_stats: Statistics about the processing operation
        """
        self.successful_documents = successful_documents
        self.failed_documents = failed_documents
        self.processing_stats = processing_stats

    @property
    def success_count(self) -> int:
        """Number of successfully processed documents."""
        return len(self.successful_documents)

    @property
    def failure_count(self) -> int:
        """Number of failed documents."""
        return len(self.failed_documents)

    @property
    def total_count(self) -> int:
        """Total number of documents processed."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.total_count == 0:
            return 0.0
        return (self.success_count / self.total_count) * 100.0


class BatchProcessor:
    """
    Efficient batch processing system for document metadata workflow.

    This class provides comprehensive batch processing capabilities for handling
    large collections of documents, with support for progress tracking,
    error recovery, and parallel processing.
    """

    def __init__(
        self,
        metadata_aggregator: MetadataAggregator | None = None,
        config: BatchConfig | None = None,
    ):
        """
        Initialize batch processor.

        Args:
            metadata_aggregator: Optional metadata aggregator instance
            config: Optional batch processing configuration
        """
        self.metadata_aggregator = metadata_aggregator or MetadataAggregator()
        self.config = config or BatchConfig()
        self._parsers: dict[str, DocumentParser] = {}
        self._load_parsers()

    def _load_parsers(self) -> None:
        """Load available document parsers."""
        try:
            # Import parsers dynamically to avoid circular imports
            from ..parsers.docx_parser import DOCXParser
            from ..parsers.epub_parser import EPUBParser
            from ..parsers.html_parser import HTMLParser
            from ..parsers.markdown_parser import MarkdownParser
            from ..parsers.mobi_parser import MOBIParser
            from ..parsers.pdf_parser import PDFParser
            from ..parsers.pptx_parser import PPTXParser
            from ..parsers.text_parser import TextParser
            from ..parsers.web_parser import WebParser

            # Register parsers
            parsers = [
                PDFParser(),
                EPUBParser(),
                MOBIParser(),
                WebParser(),
                TextParser(),
                MarkdownParser(),
                HTMLParser(),
                DOCXParser(),
                PPTXParser(),
            ]

            for parser in parsers:
                for ext in parser.supported_extensions:
                    self._parsers[ext] = parser

            logger.debug(f"Loaded {len(self._parsers)} parser mappings")

        except ImportError as e:
            logger.warning(f"Some parsers could not be loaded: {e}")

    async def process_documents(
        self,
        file_paths: list[str | Path],
        project_name: str | None = None,
        collection_type: str = "documents",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchResult:
        """
        Process multiple documents in batches.

        Args:
            file_paths: List of file paths to process
            project_name: Optional project name for collection metadata
            collection_type: Type of collection for metadata
            progress_callback: Optional callback for progress updates

        Returns:
            BatchResult with processing results

        Raises:
            BatchProcessingError: If batch processing fails completely
        """
        start_time = time.time()
        successful_documents = []
        failed_documents = []

        try:
            # Filter valid file paths
            valid_paths = self._filter_valid_paths(file_paths)

            if not valid_paths:
                raise BatchProcessingError(
                    "No valid file paths found for processing",
                    failed_documents=[str(p) for p in file_paths],
                )

            # Create progress tracker
            progress_tracker = None
            if self.config.progress_reporting:
                progress_tracker = create_progress_tracker(
                    total_items=len(valid_paths),
                    description="Processing documents"
                )

            # Process in batches
            total_processed = 0
            for batch_start in range(0, len(valid_paths), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(valid_paths))
                batch_paths = valid_paths[batch_start:batch_end]

                logger.debug(f"Processing batch {batch_start//self.config.batch_size + 1}: "
                           f"{len(batch_paths)} documents")

                # Process current batch
                batch_successful, batch_failed = await self._process_batch(
                    batch_paths,
                    project_name,
                    collection_type,
                    progress_tracker,
                )

                successful_documents.extend(batch_successful)
                failed_documents.extend(batch_failed)
                total_processed += len(batch_paths)

                # Update progress callback
                if progress_callback:
                    progress_callback(total_processed, len(valid_paths))

                # Memory management check
                if self.config.max_memory_usage_mb:
                    self._check_memory_usage()

            # Retry failed documents if configured
            if self.config.retry_failed and failed_documents:
                logger.info(f"Retrying {len(failed_documents)} failed documents")
                retry_paths = [path for path, _ in failed_documents]
                retry_successful, retry_failed = await self._retry_failed_documents(
                    retry_paths, project_name, collection_type
                )
                successful_documents.extend(retry_successful)
                failed_documents = retry_failed

            # Calculate processing statistics
            end_time = time.time()
            processing_stats = {
                "total_files": len(file_paths),
                "valid_files": len(valid_paths),
                "successful_count": len(successful_documents),
                "failed_count": len(failed_documents),
                "processing_time_seconds": end_time - start_time,
                "documents_per_second": len(valid_paths) / (end_time - start_time) if end_time > start_time else 0,
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
            }

            logger.info(
                f"Batch processing completed: {len(successful_documents)} successful, "
                f"{len(failed_documents)} failed, {processing_stats['processing_time_seconds']:.2f}s"
            )

            return BatchResult(successful_documents, failed_documents, processing_stats)

        except Exception as e:
            raise BatchProcessingError(
                f"Batch processing failed: {str(e)}",
                failed_documents=[str(p) for p in file_paths],
                details={"original_error": str(e)},
            ) from e

    async def _process_batch(
        self,
        file_paths: list[Path],
        project_name: str | None,
        collection_type: str,
        progress_tracker: ProgressTracker | None,
    ) -> tuple[list[DocumentMetadata], list[tuple[str, str]]]:
        """
        Process a single batch of documents.

        Args:
            file_paths: Batch of file paths to process
            project_name: Optional project name
            collection_type: Collection type
            progress_tracker: Optional progress tracker

        Returns:
            Tuple of (successful_documents, failed_documents)
        """
        successful_documents = []
        failed_documents = []

        if self.config.parallel_processing and len(file_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self._process_single_document,
                        file_path,
                        project_name,
                        collection_type,
                    ): file_path
                    for file_path in file_paths
                }

                # Collect results as they complete
                for future in as_completed(future_to_path, timeout=self.config.timeout_seconds):
                    file_path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            successful_documents.append(result)
                    except Exception as e:
                        error_msg = f"Processing failed: {str(e)}"
                        failed_documents.append((str(file_path), error_msg))
                        logger.error(f"Failed to process {file_path}: {e}")

                    # Update progress
                    if progress_tracker:
                        progress_tracker.update(1)

        else:
            # Sequential processing
            for file_path in file_paths:
                try:
                    result = self._process_single_document(
                        file_path, project_name, collection_type
                    )
                    if result:
                        successful_documents.append(result)
                except Exception as e:
                    error_msg = f"Processing failed: {str(e)}"
                    failed_documents.append((str(file_path), error_msg))
                    logger.error(f"Failed to process {file_path}: {e}")

                    if not self.config.continue_on_error:
                        break

                # Update progress
                if progress_tracker:
                    progress_tracker.update(1)

        return successful_documents, failed_documents

    def _process_single_document(
        self,
        file_path: Path,
        project_name: str | None,
        collection_type: str,
    ) -> DocumentMetadata | None:
        """
        Process a single document.

        Args:
            file_path: Path to document to process
            project_name: Optional project name
            collection_type: Collection type

        Returns:
            DocumentMetadata if successful, None if failed
        """
        try:
            # Detect file type and get parser
            file_type, parser_type, confidence = detect_file_type(file_path)
            parser = self._get_parser_for_file(file_path)

            if not parser:
                logger.warning(f"No parser available for file: {file_path}")
                return None

            # Parse document (run async parser in sync context)
            if hasattr(parser, 'parse') and asyncio.iscoroutinefunction(parser.parse):
                # Handle async parsers
                loop = None
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                parsed_doc = loop.run_until_complete(parser.parse(file_path))
            else:
                # Handle sync parsers
                parsed_doc = parser.parse(file_path)

            # Aggregate metadata
            document_metadata = self.metadata_aggregator.aggregate_metadata(
                parsed_doc, project_name, collection_type
            )

            return document_metadata

        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            raise

    def _get_parser_for_file(self, file_path: Path) -> DocumentParser | None:
        """
        Get appropriate parser for file.

        Args:
            file_path: Path to file

        Returns:
            DocumentParser instance or None
        """
        # Try extension-based lookup first
        extension = file_path.suffix.lower()
        if extension in self._parsers:
            return self._parsers[extension]

        # Try file type detection
        try:
            file_type, parser_type, confidence = detect_file_type(file_path)

            # Map parser type to extension
            type_extension_map = {
                "pdf": ".pdf",
                "epub": ".epub",
                "html": ".html",
                "text": ".txt",
                "markdown": ".md",
                "docx": ".docx",
                "pptx": ".pptx",
            }

            mapped_extension = type_extension_map.get(parser_type)
            if mapped_extension and mapped_extension in self._parsers:
                return self._parsers[mapped_extension]

        except Exception as e:
            logger.warning(f"Could not detect file type for {file_path}: {e}")

        return None

    def _filter_valid_paths(self, file_paths: list[str | Path]) -> list[Path]:
        """
        Filter valid file paths for processing.

        Args:
            file_paths: List of file paths to filter

        Returns:
            List of valid Path objects
        """
        valid_paths = []

        for path in file_paths:
            try:
                path_obj = Path(path)

                if not path_obj.exists():
                    logger.warning(f"File does not exist: {path}")
                    continue

                if not path_obj.is_file():
                    logger.warning(f"Path is not a file: {path}")
                    continue

                # Check if we have a parser for this file
                if self._get_parser_for_file(path_obj):
                    valid_paths.append(path_obj)
                else:
                    logger.warning(f"No parser available for: {path}")

            except Exception as e:
                logger.warning(f"Invalid file path {path}: {e}")

        return valid_paths

    async def _retry_failed_documents(
        self,
        failed_paths: list[str],
        project_name: str | None,
        collection_type: str,
    ) -> tuple[list[DocumentMetadata], list[tuple[str, str]]]:
        """
        Retry processing failed documents.

        Args:
            failed_paths: List of failed document paths
            project_name: Optional project name
            collection_type: Collection type

        Returns:
            Tuple of (successful_documents, still_failed_documents)
        """
        successful_documents = []
        still_failed = []

        for attempt in range(self.config.max_retries):
            if not failed_paths:
                break

            logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} "
                       f"for {len(failed_paths)} documents")

            retry_successful, retry_failed = await self._process_batch(
                [Path(p) for p in failed_paths],
                project_name,
                collection_type,
                None,  # No progress tracking for retries
            )

            successful_documents.extend(retry_successful)
            failed_paths = [path for path, _ in retry_failed]
            still_failed = retry_failed

            # Add delay between retry attempts
            if failed_paths and attempt < self.config.max_retries - 1:
                await asyncio.sleep(2.0)

        return successful_documents, still_failed

    def _check_memory_usage(self) -> None:
        """Check and manage memory usage if limits are configured."""
        if not self.config.max_memory_usage_mb:
            return

        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > self.config.max_memory_usage_mb:
                logger.warning(
                    f"Memory usage ({memory_mb:.1f}MB) exceeds limit "
                    f"({self.config.max_memory_usage_mb}MB)"
                )
                # Force garbage collection
                import gc
                gc.collect()

        except ImportError:
            logger.debug("psutil not available for memory monitoring")
        except Exception as e:
            logger.warning(f"Memory check failed: {e}")

    def get_processing_statistics(self) -> dict[str, Any]:
        """
        Get current processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "parsers_loaded": len(self._parsers),
            "supported_extensions": list(self._parsers.keys()),
            "batch_size": self.config.batch_size,
            "max_workers": self.config.max_workers,
            "parallel_processing": self.config.parallel_processing,
            "continue_on_error": self.config.continue_on_error,
        }


# Export main classes
__all__ = ["BatchProcessor", "BatchConfig", "BatchResult"]
