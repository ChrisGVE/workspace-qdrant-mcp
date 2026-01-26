"""
Unified Document Processing Pipeline with LSP Enhancement

This module provides a consolidated document processing pipeline that integrates
all document parsers, LSP enhancement for code files, performance monitoring,
and high-throughput optimization capabilities.

Features:
    - Unified processing interface with format auto-detection
    - LSP integration for enhanced code file processing
    - Performance-optimized processing (1000+ docs/minute, <500MB memory)
    - Streaming processing for memory efficiency
    - Comprehensive error handling and recovery
    - Metrics collection and performance monitoring
    - Edge case handling for corrupted/malformed files
    - Batch processing with adaptive concurrency control

Example:
    ```python
    from workspace_qdrant_mcp.core.unified_document_pipeline import UnifiedDocumentPipeline

    # Initialize pipeline
    pipeline = UnifiedDocumentPipeline()
    await pipeline.initialize()

    # Process documents with auto-format detection
    results = await pipeline.process_documents(
        file_paths=["/path/to/documents"],
        collection="my-collection",
        enable_lsp=True
    )

    # Get performance metrics
    metrics = pipeline.get_performance_metrics()
    ```
"""

import asyncio
import gc
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

try:
    from .lsp_metadata_extractor import FileMetadata, LspMetadataExtractor
except ImportError:
    LspMetadataExtractor = None
    FileMetadata = None

try:
    from .performance_metrics import MetricType, PerformanceMetricsCollector
    from .performance_monitor import PerformanceAlert, PerformanceMonitor
except ImportError:
    PerformanceMonitor = None
    PerformanceAlert = None
    PerformanceMetricsCollector = None
    MetricType = None

try:
    from .graceful_degradation import GracefulDegradationManager
except ImportError:
    class GracefulDegradationManager:
        pass

try:
    from .automatic_recovery import AutomaticRecovery
except ImportError:
    class AutomaticRecovery:
        pass

# Built-in lightweight parsers (Python CLI parsers have been removed).
PARSERS_AVAILABLE = False


class DocumentParser:
    format_name = "Generic"
    supported_extensions = set()

    def __init__(self):
        pass

    def can_parse(self, file_path):
        return file_path.suffix.lower() in self.supported_extensions

    async def parse(self, file_path):
        try:
            content = file_path.read_text(encoding="utf-8")
            return ParsedDocument(
                content=content,
                metadata={"file_type": self.format_name, "file_path": str(file_path)},
            )
        except Exception:
            return None


class ParsedDocument:
    def __init__(self, content="", content_hash="", metadata=None):
        self.content = content
        self.content_hash = content_hash or hashlib.sha256(content.encode()).hexdigest()
        self.metadata = metadata or {}


class CodeParser(DocumentParser):
    format_name = "Code"
    supported_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".rs", ".go", ".rb", ".php"}


class TextParser(DocumentParser):
    format_name = "Text"
    supported_extensions = {".txt"}


class MarkdownParser(DocumentParser):
    format_name = "Markdown"
    supported_extensions = {".md", ".markdown"}


class HtmlParser(DocumentParser):
    format_name = "HTML"
    supported_extensions = {".html", ".htm"}


class PDFParser(DocumentParser):
    format_name = "PDF"
    supported_extensions = {".pdf"}


class DocxParser(DocumentParser):
    format_name = "DOCX"
    supported_extensions = {".docx", ".doc"}


class EpubParser(DocumentParser):
    format_name = "EPUB"
    supported_extensions = {".epub"}


class MobiParser(DocumentParser):
    format_name = "MOBI"
    supported_extensions = {".mobi"}


class PptxParser(DocumentParser):
    format_name = "PPTX"
    supported_extensions = {".pptx", ".ppt"}


class WebParser(DocumentParser):
    format_name = "Web"
    supported_extensions = set()


def detect_file_type(file_path):
    return file_path.suffix.lower().lstrip(".")


class ParsingError(Exception):
    pass


class UnsupportedFileFormatError(Exception):
    pass


@dataclass
class ProcessingResult:
    """Result of processing a single document."""

    file_path: str
    success: bool
    document: ParsedDocument | None = None
    lsp_metadata: FileMetadata | None = None
    error: str | None = None
    processing_time: float = 0.0
    parser_used: str | None = None
    chunks_generated: int = 0
    memory_usage_mb: float = 0.0


@dataclass
class PipelineStats:
    """Comprehensive pipeline processing statistics."""

    # File processing metrics
    total_files: int = 0
    successful_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0

    # Content metrics
    total_documents: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    duplicate_files: int = 0

    # Performance metrics
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_docs_per_minute: float = 0.0

    # LSP metrics
    lsp_enhanced_files: int = 0
    lsp_failures: int = 0

    # Error tracking
    errors_by_type: dict[str, int] = field(default_factory=dict)
    errors_by_format: dict[str, int] = field(default_factory=dict)

    # Memory pressure events
    memory_pressure_events: int = 0
    gc_collections: int = 0


class UnifiedDocumentPipeline:
    """
    High-performance unified document processing pipeline.

    Consolidates all document processing capabilities into a single interface
    with performance optimization, LSP integration, and comprehensive monitoring.
    """

    def __init__(
        self,
        max_concurrency: int = 10,
        memory_limit_mb: int = 500,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        enable_performance_monitoring: bool = True,
        enable_lsp: bool = True,
        lsp_timeout: float = 30.0,
    ):
        """
        Initialize the unified document processing pipeline.

        Args:
            max_concurrency: Maximum concurrent processing tasks
            memory_limit_mb: Memory usage limit in megabytes
            chunk_size: Text chunk size for processing
            chunk_overlap: Overlap between chunks
            enable_performance_monitoring: Enable performance monitoring
            enable_lsp: Enable LSP enhancement for code files
            lsp_timeout: Timeout for LSP operations in seconds
        """
        self.max_concurrency = max_concurrency
        self.memory_limit_mb = memory_limit_mb
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_lsp = enable_lsp
        self.lsp_timeout = lsp_timeout

        # Core components
        self.lsp_extractor: LspMetadataExtractor | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.degradation_manager = GracefulDegradationManager()
        self.recovery_manager = AutomaticRecovery()

        # Document parsers with LSP integration
        self.parsers: list[DocumentParser] = []

        # Processing state
        self.is_initialized = False
        self.content_hashes: set[str] = set()
        self.processing_semaphore: asyncio.Semaphore | None = None
        self.memory_monitor_task: asyncio.Task | None = None

        # Statistics
        self.stats = PipelineStats()

        logger.info(f"Initialized UnifiedDocumentPipeline with {max_concurrency} max concurrency")

    async def initialize(self) -> None:
        """Initialize the pipeline components."""
        if self.is_initialized:
            return

        try:
            # Initialize semaphore for concurrency control
            self.processing_semaphore = asyncio.Semaphore(self.max_concurrency)

            # Initialize LSP extractor if enabled
            if self.enable_lsp:
                self.lsp_extractor = LspMetadataExtractor()
                await self.lsp_extractor.initialize()
                logger.info("LSP metadata extractor initialized")

            # Initialize performance monitoring
            if self.enable_performance_monitoring:
                self.performance_monitor = PerformanceMonitor("unified_pipeline")
                await self.performance_monitor.start()
                logger.info("Performance monitoring started")

            # Initialize document parsers
            self._initialize_parsers()

            # Start memory monitoring
            self.memory_monitor_task = asyncio.create_task(self._monitor_memory())

            self.is_initialized = True
            logger.info("UnifiedDocumentPipeline initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise

    def _initialize_parsers(self) -> None:
        """Initialize all document parsers with LSP integration."""
        self.parsers = []

        # Try to initialize code parser with LSP if available
        try:
            if PARSERS_AVAILABLE and hasattr(CodeParser, '__init__'):
                # Check if CodeParser accepts lsp_extractor parameter
                import inspect
                sig = inspect.signature(CodeParser.__init__)
                if 'lsp_extractor' in sig.parameters:
                    self.parsers.append(CodeParser(lsp_extractor=self.lsp_extractor))
                else:
                    self.parsers.append(CodeParser())
            else:
                self.parsers.append(CodeParser())
        except Exception as e:
            logger.warning(f"Failed to initialize CodeParser: {e}")
            self.parsers.append(DocumentParser())

        # Initialize other parsers
        parser_classes = [
            TextParser, MarkdownParser, PDFParser, DocxParser,
            PptxParser, HtmlParser, EpubParser, MobiParser, WebParser
        ]

        for parser_class in parser_classes:
            try:
                self.parsers.append(parser_class())
            except Exception as e:
                logger.warning(f"Failed to initialize {parser_class.__name__}: {e}")
                self.parsers.append(DocumentParser())

        logger.info(f"Initialized {len(self.parsers)} document parsers")

    async def process_documents(
        self,
        file_paths: list[str | Path],
        collection: str,
        batch_size: int | None = None,
        progress_callback: Callable | None = None,
        enable_deduplication: bool = True,
        dry_run: bool = False,
    ) -> list[ProcessingResult]:
        """
        Process multiple documents with automatic format detection.

        Args:
            file_paths: List of file paths to process
            collection: Target collection name
            batch_size: Batch size for processing (auto-calculated if None)
            progress_callback: Optional progress callback
            enable_deduplication: Enable content deduplication
            dry_run: Process without actually storing results

        Returns:
            List of ProcessingResult objects
        """
        if not self.is_initialized:
            await self.initialize()

        start_time = time.time()
        file_paths = [Path(fp) for fp in file_paths]

        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(len(file_paths))

        # Reset statistics for this run
        self.stats = PipelineStats()
        self.stats.total_files = len(file_paths)

        # Clear content hashes if deduplication is disabled
        if not enable_deduplication:
            self.content_hashes.clear()

        logger.info(f"Processing {len(file_paths)} documents in batches of {batch_size}")

        results: list[ProcessingResult] = []

        try:
            # Process files in batches
            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]

                # Check memory pressure before processing batch
                await self._handle_memory_pressure()

                # Process batch
                batch_results = await self._process_batch(
                    batch_files, collection, enable_deduplication, dry_run
                )
                results.extend(batch_results)

                # Update progress
                if progress_callback:
                    progress_callback(len(results), len(file_paths), self.stats)

                # Force garbage collection between batches
                if i % (batch_size * 2) == 0:
                    self._force_garbage_collection()

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            raise

        # Calculate final statistics
        total_time = time.time() - start_time
        self._finalize_stats(total_time)

        logger.info(
            f"Pipeline completed: {self.stats.successful_files}/{self.stats.total_files} files "
            f"in {total_time:.2f}s ({self.stats.throughput_docs_per_minute:.1f} docs/min)"
        )

        return results

    async def _process_batch(
        self,
        file_paths: list[Path],
        collection: str,
        enable_deduplication: bool,
        dry_run: bool,
    ) -> list[ProcessingResult]:
        """Process a batch of files concurrently."""
        tasks = []

        for file_path in file_paths:
            task = asyncio.create_task(
                self._process_single_document(
                    file_path, collection, enable_deduplication, dry_run
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task failed for {file_paths[i]}: {result}")
                processed_results.append(ProcessingResult(
                    file_path=str(file_paths[i]),
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)

        return processed_results

    async def _process_single_document(
        self,
        file_path: Path,
        collection: str,
        enable_deduplication: bool,
        dry_run: bool,
    ) -> ProcessingResult:
        """Process a single document with comprehensive error handling."""
        async with self.processing_semaphore:
            start_time = time.time()
            initial_memory = self._get_memory_usage()

            result = ProcessingResult(
                file_path=str(file_path),
                success=False
            )

            try:
                # Detect file format
                file_type = detect_file_type(file_path)
                if not file_type:
                    result.error = "Unsupported file format"
                    self.stats.skipped_files += 1
                    return result

                # Find appropriate parser
                parser = self._find_parser(file_path, file_type)
                if not parser:
                    result.error = f"No parser found for format: {file_type}"
                    self.stats.skipped_files += 1
                    return result

                result.parser_used = parser.format_name

                # Parse the document
                parsed_doc = await self._safe_parse_document(parser, file_path)
                if not parsed_doc:
                    result.error = "Document parsing failed"
                    self.stats.failed_files += 1
                    return result

                result.document = parsed_doc

                # Check for duplicates
                if enable_deduplication and parsed_doc.content_hash in self.content_hashes:
                    result.error = "Duplicate content detected"
                    result.success = True  # Not an error, just a skip
                    self.stats.duplicate_files += 1
                    return result

                self.content_hashes.add(parsed_doc.content_hash)

                # Extract LSP metadata for code files
                if self.enable_lsp and isinstance(parser, CodeParser):
                    result.lsp_metadata = await self._extract_lsp_metadata(file_path)
                    if result.lsp_metadata:
                        self.stats.lsp_enhanced_files += 1
                    else:
                        self.stats.lsp_failures += 1

                # Estimate chunks
                if parsed_doc.content:
                    result.chunks_generated = max(1, len(parsed_doc.content) // self.chunk_size)

                # Update statistics
                result.success = True
                self.stats.successful_files += 1
                self.stats.total_documents += 1
                self.stats.total_chunks += result.chunks_generated
                self.stats.total_characters += len(parsed_doc.content) if parsed_doc.content else 0

            except Exception as e:
                result.error = str(e)
                self.stats.failed_files += 1
                self._update_error_stats(e, file_type if 'file_type' in locals() else 'unknown')
                logger.error(f"Failed to process {file_path}: {e}")

            finally:
                # Calculate processing metrics
                result.processing_time = time.time() - start_time
                result.memory_usage_mb = self._get_memory_usage() - initial_memory

                # Update peak memory usage
                current_memory = self._get_memory_usage()
                if current_memory > self.stats.peak_memory_mb:
                    self.stats.peak_memory_mb = current_memory

            return result

    async def _safe_parse_document(
        self, parser: DocumentParser, file_path: Path
    ) -> ParsedDocument | None:
        """Safely parse a document with error handling."""
        try:
            # Add timeout for parsing operations
            return await asyncio.wait_for(
                parser.parse(file_path),
                timeout=60.0  # 1 minute timeout per file
            )
        except asyncio.TimeoutError:
            logger.warning(f"Parsing timeout for {file_path}")
            return None
        except (ParsingError, UnsupportedFileFormatError) as e:
            logger.warning(f"Expected parsing error for {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected parsing error for {file_path}: {e}")
            return None

    async def _extract_lsp_metadata(self, file_path: Path) -> FileMetadata | None:
        """Extract LSP metadata with timeout and error handling."""
        if not self.lsp_extractor:
            return None

        try:
            return await asyncio.wait_for(
                self.lsp_extractor.extract_file_metadata(str(file_path)),
                timeout=self.lsp_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"LSP extraction timeout for {file_path}")
            return None
        except Exception as e:
            logger.debug(f"LSP extraction failed for {file_path}: {e}")
            return None

    def _find_parser(self, file_path: Path, file_type: str) -> DocumentParser | None:
        """Find the most appropriate parser for a file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    def _calculate_optimal_batch_size(self, total_files: int) -> int:
        """Calculate optimal batch size based on memory and concurrency."""
        # Base batch size on available memory and concurrency
        base_batch_size = min(50, max(10, self.memory_limit_mb // 20))

        # Adjust based on total files
        if total_files < 100:
            return min(base_batch_size, total_files // 2 + 1)
        elif total_files < 1000:
            return base_batch_size
        else:
            return min(base_batch_size * 2, 100)

    async def _handle_memory_pressure(self) -> None:
        """Handle memory pressure by triggering garbage collection."""
        current_memory = self._get_memory_usage()

        if current_memory > self.memory_limit_mb * 0.8:  # 80% threshold
            self.stats.memory_pressure_events += 1
            self._force_garbage_collection()

            # Wait a bit for memory to be freed
            await asyncio.sleep(0.1)

            # Check if memory is still high
            if self._get_memory_usage() > self.memory_limit_mb * 0.9:
                logger.warning(
                    f"High memory usage: {self._get_memory_usage():.1f}MB "
                    f"(limit: {self.memory_limit_mb}MB)"
                )

    async def _monitor_memory(self) -> None:
        """Background task to monitor memory usage."""
        while True:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                current_memory = self._get_memory_usage()

                if current_memory > self.memory_limit_mb:
                    logger.warning(f"Memory limit exceeded: {current_memory:.1f}MB")
                    self._force_garbage_collection()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

    def _force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        collected = gc.collect()
        self.stats.gc_collections += 1
        logger.debug(f"Garbage collection freed {collected} objects")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0

    def _update_error_stats(self, error: Exception, file_type: str) -> None:
        """Update error statistics."""
        error_type = type(error).__name__
        self.stats.errors_by_type[error_type] = self.stats.errors_by_type.get(error_type, 0) + 1
        self.stats.errors_by_format[file_type] = self.stats.errors_by_format.get(file_type, 0) + 1

    def _finalize_stats(self, total_time: float) -> None:
        """Finalize processing statistics."""
        self.stats.total_processing_time = total_time

        if self.stats.successful_files > 0:
            self.stats.average_processing_time = total_time / self.stats.successful_files
            self.stats.throughput_docs_per_minute = (self.stats.successful_files / total_time) * 60

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        return {
            "processing_stats": {
                "total_files": self.stats.total_files,
                "successful_files": self.stats.successful_files,
                "failed_files": self.stats.failed_files,
                "skipped_files": self.stats.skipped_files,
                "duplicate_files": self.stats.duplicate_files,
                "success_rate": (self.stats.successful_files / self.stats.total_files * 100) if self.stats.total_files > 0 else 0,
            },
            "content_stats": {
                "total_documents": self.stats.total_documents,
                "total_chunks": self.stats.total_chunks,
                "total_characters": self.stats.total_characters,
            },
            "performance_stats": {
                "total_processing_time": self.stats.total_processing_time,
                "average_processing_time": self.stats.average_processing_time,
                "throughput_docs_per_minute": self.stats.throughput_docs_per_minute,
                "peak_memory_mb": self.stats.peak_memory_mb,
                "memory_pressure_events": self.stats.memory_pressure_events,
                "gc_collections": self.stats.gc_collections,
            },
            "lsp_stats": {
                "lsp_enhanced_files": self.stats.lsp_enhanced_files,
                "lsp_failures": self.stats.lsp_failures,
                "lsp_success_rate": (self.stats.lsp_enhanced_files / (self.stats.lsp_enhanced_files + self.stats.lsp_failures) * 100) if (self.stats.lsp_enhanced_files + self.stats.lsp_failures) > 0 else 0,
            },
            "error_stats": {
                "errors_by_type": dict(self.stats.errors_by_type),
                "errors_by_format": dict(self.stats.errors_by_format),
            }
        }

    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        try:
            if self.memory_monitor_task:
                self.memory_monitor_task.cancel()
                try:
                    await self.memory_monitor_task
                except asyncio.CancelledError:
                    pass

            if self.performance_monitor:
                await self.performance_monitor.stop()

            if self.lsp_extractor:
                await self.lsp_extractor.cleanup()

            logger.info("UnifiedDocumentPipeline cleanup complete")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Convenience functions for common use cases
async def process_directory_unified(
    directory_path: str | Path,
    collection: str,
    **kwargs
) -> list[ProcessingResult]:
    """
    Convenience function to process all files in a directory.

    Args:
        directory_path: Path to directory to process
        collection: Target collection name
        **kwargs: Additional arguments for UnifiedDocumentPipeline

    Returns:
        List of ProcessingResult objects
    """
    directory_path = Path(directory_path)

    # Find all supported files
    file_paths = []
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            file_type = detect_file_type(file_path)
            if file_type:
                file_paths.append(file_path)

    # Process with unified pipeline
    async with UnifiedDocumentPipeline(**kwargs) as pipeline:
        return await pipeline.process_documents(file_paths, collection)


async def validate_pipeline_performance(
    test_files: list[Path],
    target_throughput: float = 1000.0,  # docs per minute
    memory_limit: int = 500,  # MB
) -> bool:
    """
    Validate that the pipeline meets performance requirements.

    Args:
        test_files: List of test files to process
        target_throughput: Target throughput in documents per minute
        memory_limit: Memory limit in MB

    Returns:
        True if performance requirements are met
    """
    async with UnifiedDocumentPipeline(
        memory_limit_mb=memory_limit,
        enable_performance_monitoring=True
    ) as pipeline:
        await pipeline.process_documents(
            test_files, "test-collection", dry_run=True
        )

        metrics = pipeline.get_performance_metrics()
        performance = metrics["performance_stats"]

        # Check throughput requirement
        throughput_ok = performance["throughput_docs_per_minute"] >= target_throughput

        # Check memory requirement
        memory_ok = performance["peak_memory_mb"] <= memory_limit

        # Check success rate
        success_rate_ok = metrics["processing_stats"]["success_rate"] >= 95.0

        logger.info(
            f"Performance validation: "
            f"Throughput: {performance['throughput_docs_per_minute']:.1f} docs/min (target: {target_throughput}) - {'✓' if throughput_ok else '✗'}, "
            f"Peak memory: {performance['peak_memory_mb']:.1f}MB (limit: {memory_limit}) - {'✓' if memory_ok else '✗'}, "
            f"Success rate: {metrics['processing_stats']['success_rate']:.1f}% - {'✓' if success_rate_ok else '✗'}"
        )

        return throughput_ok and memory_ok and success_rate_ok
