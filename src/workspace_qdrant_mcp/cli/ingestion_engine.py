"""
Document ingestion engine for batch processing.

This module provides the core engine for batch document ingestion into
workspace-qdrant collections. It handles directory traversal, format detection,
concurrent processing, deduplication, and comprehensive error handling.
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from ..core.client import QdrantWorkspaceClient
from ..tools.documents import add_document
from .parsers import (
    DocumentParser,
    MarkdownParser,
    PDFParser,
    TextParser,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics and metrics for batch ingestion operation."""

    # File processing stats
    files_found: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0

    # Content stats
    total_documents: int = 0
    total_chunks: int = 0
    total_characters: int = 0
    total_words: int = 0

    # Timing stats
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: float = 0.0

    # Error tracking
    errors: list[dict[str, str]] = field(default_factory=list)
    skipped_files: list[dict[str, str]] = field(default_factory=list)
    duplicates_found: int = 0

    @property
    def files_per_second(self) -> float:
        """Calculate processing rate in files per second."""
        if self.processing_time > 0:
            return self.files_processed / self.processing_time
        return 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.files_found > 0:
            return (self.files_processed / self.files_found) * 100
        return 0.0


@dataclass
class IngestionResult:
    """Result of a batch ingestion operation."""

    success: bool
    stats: IngestionStats
    collection: str
    message: str
    dry_run: bool = False


class DocumentIngestionEngine:
    """
    High-performance batch document ingestion engine.

    Provides concurrent document processing with support for multiple formats,
    deduplication, error recovery, and comprehensive progress tracking.

    Features:
        - Concurrent processing with configurable concurrency limits
        - Multi-format document parsing (text, markdown, PDF)
        - SHA256-based content deduplication
        - Comprehensive error handling and recovery
        - Detailed statistics and progress reporting
        - Dry-run mode for safe operation preview
        - Extensible parser architecture
    """

    def __init__(
        self,
        client: QdrantWorkspaceClient,
        concurrency: int = 5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the ingestion engine.

        Args:
            client: Initialized workspace client
            concurrency: Maximum concurrent file processing operations
            chunk_size: Size limit for text chunks
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.client = client
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize parsers
        self.parsers: list[DocumentParser] = [
            TextParser(),
            MarkdownParser(),
            PDFParser(),
        ]

        # Content hash tracking for deduplication
        self.content_hashes: set[str] = set()

        # Semaphore for controlling concurrency
        self.semaphore = asyncio.Semaphore(concurrency)

        logger.info(
            f"Initialized DocumentIngestionEngine with {len(self.parsers)} parsers"
        )

    async def process_directory(
        self,
        directory_path: Union[str, Path],
        collection: str,
        formats: Optional[list[str]] = None,
        dry_run: bool = False,
        progress_callback: Optional[callable] = None,
        recursive: bool = True,
        exclude_patterns: Optional[list[str]] = None,
    ) -> IngestionResult:
        """
        Process all documents in a directory.

        Args:
            directory_path: Path to directory containing documents
            collection: Target collection name
            formats: List of formats to process (None for all supported)
            dry_run: If True, analyze files but don't actually ingest
            progress_callback: Optional callback for progress updates
            recursive: Whether to process subdirectories
            exclude_patterns: List of glob patterns to exclude

        Returns:
            IngestionResult with comprehensive operation results
        """
        directory_path = Path(directory_path)

        if not directory_path.exists():
            return IngestionResult(
                success=False,
                stats=IngestionStats(),
                collection=collection,
                message=f"Directory not found: {directory_path}",
                dry_run=dry_run,
            )

        stats = IngestionStats()
        stats.start_time = datetime.now(timezone.utc)

        try:
            # Find all processable files
            files_to_process = await self._find_files(
                directory_path, formats, recursive, exclude_patterns
            )

            stats.files_found = len(files_to_process)

            if not files_to_process:
                return IngestionResult(
                    success=True,
                    stats=stats,
                    collection=collection,
                    message=f"No processable files found in {directory_path}",
                    dry_run=dry_run,
                )

            logger.info(f"Found {len(files_to_process)} files to process")

            # Ensure collection exists (create if not dry run)
            if not dry_run:
                available_collections = await self.client.list_collections()
                if collection not in available_collections:
                    logger.info(f"Collection '{collection}' not found. Creating it...")

                    # Create collection using the client's collection manager
                    from ..core.collections import CollectionConfig

                    collection_config = CollectionConfig(
                        name=collection,
                        description=f"Ingested documents collection: {collection}",
                        collection_type="docs",  # Type for ingested documents
                        project_name=None,  # Generic collection
                        vector_size=384,  # all-MiniLM-L6-v2 dimension
                        distance_metric="Cosine",
                        enable_sparse_vectors=True,
                    )

                    # Use the client's collection manager to create the collection
                    await self.client.collection_manager._ensure_collection_exists(
                        collection_config
                    )

                    # Small delay to allow for collection consistency across Qdrant
                    await asyncio.sleep(0.5)

                    logger.info(f"Successfully created collection '{collection}'")

            # Process files with concurrency control
            await self._process_files_batch(
                files_to_process, collection, stats, dry_run, progress_callback
            )

            stats.end_time = datetime.now(timezone.utc)
            stats.processing_time = (stats.end_time - stats.start_time).total_seconds()

            success_message = (
                f"{'Analyzed' if dry_run else 'Processed'} {stats.files_processed} files "
                f"({stats.total_chunks} chunks) in {stats.processing_time:.2f}s. "
                f"Rate: {stats.files_per_second:.1f} files/sec"
            )

            if stats.files_failed > 0:
                success_message += f". {stats.files_failed} files failed"

            if stats.duplicates_found > 0:
                success_message += f". {stats.duplicates_found} duplicates skipped"

            return IngestionResult(
                success=True,
                stats=stats,
                collection=collection,
                message=success_message,
                dry_run=dry_run,
            )

        except Exception as e:
            stats.end_time = datetime.now(timezone.utc)
            logger.error(f"Batch ingestion failed: {e}")
            return IngestionResult(
                success=False,
                stats=stats,
                collection=collection,
                message=f"Batch ingestion failed: {e}",
                dry_run=dry_run,
            )

    async def _find_files(
        self,
        directory: Path,
        formats: Optional[list[str]],
        recursive: bool,
        exclude_patterns: Optional[list[str]],
    ) -> list[Path]:
        """Find all processable files in directory."""

        files = []

        # Get supported extensions
        supported_extensions = set()
        format_filter = set(formats) if formats else None

        for parser in self.parsers:
            if not format_filter or any(
                fmt in parser.format_name.lower() for fmt in format_filter
            ):
                supported_extensions.update(parser.supported_extensions)

        # Search for files
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in supported_extensions:
                continue

            # Check exclude patterns
            if exclude_patterns:
                excluded = False
                for pattern in exclude_patterns:
                    if file_path.match(pattern):
                        excluded = True
                        break
                if excluded:
                    continue

            files.append(file_path)

        return sorted(files)

    async def _process_files_batch(
        self,
        files: list[Path],
        collection: str,
        stats: IngestionStats,
        dry_run: bool,
        progress_callback: Optional[callable],
    ) -> None:
        """Process files with concurrency control."""

        tasks = []

        for file_path in files:
            task = asyncio.create_task(
                self._process_single_file(file_path, collection, stats, dry_run)
            )
            tasks.append(task)

        # Process with progress updates
        completed = 0
        for task in asyncio.as_completed(tasks):
            try:
                await task
                completed += 1

                if progress_callback:
                    progress_callback(completed, len(files), stats)

            except Exception as e:
                logger.error(f"Task failed: {e}")
                stats.files_failed += 1

    async def _process_single_file(
        self, file_path: Path, collection: str, stats: IngestionStats, dry_run: bool
    ) -> None:
        """Process a single file with concurrency control."""

        async with self.semaphore:
            try:
                # Find appropriate parser
                parser = self._find_parser(file_path)
                if not parser:
                    stats.files_skipped += 1
                    stats.skipped_files.append(
                        {"file": str(file_path), "reason": "No suitable parser found"}
                    )
                    return

                # Parse the document
                parsed_doc = await parser.parse(file_path)

                # Check for duplicate content
                if parsed_doc.content_hash in self.content_hashes:
                    stats.duplicates_found += 1
                    stats.files_skipped += 1
                    stats.skipped_files.append(
                        {
                            "file": str(file_path),
                            "reason": "Duplicate content (SHA256 match)",
                        }
                    )
                    return

                self.content_hashes.add(parsed_doc.content_hash)

                # Update statistics
                stats.total_characters += len(parsed_doc.content)
                stats.total_words += (
                    len(parsed_doc.content.split()) if parsed_doc.content else 0
                )

                if not dry_run:
                    # Add to collection
                    result = await add_document(
                        client=self.client,
                        content=parsed_doc.content,
                        collection=collection,
                        metadata=parsed_doc.metadata,
                        document_id=f"{file_path.stem}_{parsed_doc.content_hash[:8]}",
                        chunk_text=len(parsed_doc.content) > self.chunk_size,
                    )

                    if result.get("error"):
                        raise RuntimeError(result["error"])

                    stats.total_chunks += result.get("points_added", 1)
                else:
                    # Dry run - estimate chunks
                    estimated_chunks = max(
                        1, len(parsed_doc.content) // self.chunk_size
                    )
                    stats.total_chunks += estimated_chunks

                stats.files_processed += 1
                stats.total_documents += 1

                logger.debug(f"{'Analyzed' if dry_run else 'Processed'} {file_path}")

            except Exception as e:
                stats.files_failed += 1
                stats.errors.append(
                    {
                        "file": str(file_path),
                        "error": str(e),
                        "parser": parser.format_name if parser else "unknown",
                    }
                )
                logger.error(f"Failed to process {file_path}: {e}")

    def _find_parser(self, file_path: Path) -> Optional[DocumentParser]:
        """Find the appropriate parser for a file."""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None

    async def get_supported_formats(self) -> dict[str, dict[str, Any]]:
        """Get information about supported file formats."""
        formats = {}

        for parser in self.parsers:
            formats[parser.format_name] = {
                "extensions": parser.supported_extensions,
                "options": parser.get_parsing_options(),
                "available": True,  # Could check dependencies here
            }

        return formats

    async def estimate_processing_time(
        self, directory_path: Union[str, Path], formats: Optional[list[str]] = None
    ) -> dict[str, Any]:
        """Estimate processing time and resource usage."""
        directory_path = Path(directory_path)

        files = await self._find_files(directory_path, formats, True, None)

        # Simple estimation based on file sizes and types
        total_size = 0
        file_types = defaultdict(int)

        for file_path in files:
            try:
                size = file_path.stat().st_size
                total_size += size

                parser = self._find_parser(file_path)
                if parser:
                    file_types[parser.format_name] += 1
            except OSError:
                continue

        # Rough estimation (adjust based on empirical data)
        estimated_seconds = (total_size / (1024 * 1024)) * 2  # ~2 seconds per MB
        estimated_seconds /= self.concurrency  # Account for concurrency

        return {
            "files_found": len(files),
            "total_size_mb": total_size / (1024 * 1024),
            "file_types": dict(file_types),
            "estimated_time_seconds": estimated_seconds,
            "estimated_time_human": f"{estimated_seconds // 60:.0f}m {estimated_seconds % 60:.0f}s",
        }
