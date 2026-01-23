"""
Unit tests for optimized auto-ingestion module.

Tests the OptimizedAutoIngestion class and its integration with
BatchProcessingManager for high-performance file processing.

Task 447: Optimize auto-ingestion performance for large file sets.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.core.optimized_auto_ingestion import (
    IngestionPriority,
    IngestionProgress,
    IngestionResult,
    IngestionStatistics,
    OptimizedAutoIngestion,
    create_optimized_ingestion,
)


class TestIngestionProgress:
    """Tests for IngestionProgress dataclass."""

    def test_percentage_calculation_empty(self):
        """Test percentage with no files."""
        progress = IngestionProgress()
        assert progress.percentage == 0.0

    def test_percentage_calculation_partial(self):
        """Test percentage calculation with partial progress."""
        progress = IngestionProgress(
            total_files=100,
            files_processed=50,
            files_failed=10,
            files_skipped=5,
        )
        assert progress.percentage == 65.0

    def test_percentage_calculation_complete(self):
        """Test percentage calculation when complete."""
        progress = IngestionProgress(
            total_files=100,
            files_processed=95,
            files_failed=3,
            files_skipped=2,
        )
        assert progress.percentage == 100.0

    def test_files_per_second_calculation(self):
        """Test processing rate calculation."""
        progress = IngestionProgress(
            files_processed=100,
            elapsed_seconds=10.0,
        )
        assert progress.files_per_second == 10.0

    def test_files_per_second_zero_time(self):
        """Test processing rate with zero elapsed time."""
        progress = IngestionProgress(
            files_processed=100,
            elapsed_seconds=0.0,
        )
        assert progress.files_per_second == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        progress = IngestionProgress(
            total_files=100,
            files_processed=50,
            current_file="/path/to/file.py",
        )
        result = progress.to_dict()

        assert result["total_files"] == 100
        assert result["files_processed"] == 50
        assert result["current_file"] == "/path/to/file.py"
        assert "percentage" in result
        assert "files_per_second" in result


class TestIngestionResult:
    """Tests for IngestionResult dataclass."""

    def test_successful_result(self):
        """Test successful ingestion result."""
        result = IngestionResult(
            file_path="/path/to/file.py",
            success=True,
            processing_time_ms=150.5,
            file_size_bytes=1024,
        )
        assert result.success is True
        assert result.error_message is None

    def test_failed_result(self):
        """Test failed ingestion result."""
        result = IngestionResult(
            file_path="/path/to/file.py",
            success=False,
            processing_time_ms=50.0,
            error_message="File not found",
        )
        assert result.success is False
        assert result.error_message == "File not found"

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = IngestionResult(
            file_path="/path/to/file.py",
            success=True,
            processing_time_ms=150.567,
            file_size_bytes=1024,
            metadata={"extension": ".py"},
        )
        data = result.to_dict()

        assert data["file_path"] == "/path/to/file.py"
        assert data["success"] is True
        assert data["processing_time_ms"] == 150.57  # Rounded
        assert data["file_size_bytes"] == 1024
        assert data["metadata"]["extension"] == ".py"


class TestIngestionStatistics:
    """Tests for IngestionStatistics dataclass."""

    def test_initial_statistics(self):
        """Test initial statistics values."""
        stats = IngestionStatistics()
        assert stats.total_files_processed == 0
        assert stats.total_files_failed == 0
        assert stats.batches_processed == 0

    def test_throughput_calculation(self):
        """Test throughput calculation in to_dict."""
        stats = IngestionStatistics(
            total_files_processed=100,
            total_processing_time_ms=10000.0,  # 10 seconds
        )
        data = stats.to_dict()
        assert data["throughput_files_per_second"] == 10.0

    def test_to_dict_all_fields(self):
        """Test all fields in dictionary conversion."""
        stats = IngestionStatistics(
            total_files_processed=100,
            total_files_failed=5,
            total_files_skipped=10,
            total_bytes_processed=1024000,
            total_processing_time_ms=5000.0,
            batches_processed=10,
            deduplication_hits=15,
            priority_distribution={"normal": 80, "high": 20},
        )
        data = stats.to_dict()

        assert data["total_files_processed"] == 100
        assert data["total_files_failed"] == 5
        assert data["total_files_skipped"] == 10
        assert data["deduplication_hits"] == 15
        assert data["priority_distribution"]["normal"] == 80


@pytest.fixture
def temp_directory():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create test files
        (tmppath / "file1.py").write_text("print('hello')")
        (tmppath / "file2.py").write_text("print('world')")
        (tmppath / "file3.md").write_text("# README")
        (tmppath / "subdir").mkdir()
        (tmppath / "subdir" / "nested.py").write_text("import os")

        yield tmppath


@pytest.fixture
def mock_file_processor():
    """Create a mock file processor."""
    async def processor(file_path: Path) -> dict[str, Any]:
        return {"processed": str(file_path)}
    return processor


class TestOptimizedAutoIngestion:
    """Tests for OptimizedAutoIngestion class."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test basic initialization."""
        ingestion = OptimizedAutoIngestion()

        assert ingestion.max_concurrent_files > 0
        assert ingestion.batch_size > 0
        assert ingestion.is_processing is False

    @pytest.mark.asyncio
    async def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        ingestion = OptimizedAutoIngestion(
            max_concurrent_files=5,
            batch_size=25,
        )

        assert ingestion.max_concurrent_files == 5
        assert ingestion.batch_size == 25

    @pytest.mark.asyncio
    async def test_ingest_empty_list(self):
        """Test ingesting empty file list."""
        ingestion = OptimizedAutoIngestion()
        results = await ingestion.ingest_files([])

        assert results == []

    @pytest.mark.asyncio
    async def test_ingest_single_file(self, temp_directory, mock_file_processor):
        """Test ingesting a single file."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        test_file = temp_directory / "file1.py"
        results = await ingestion.ingest_files([test_file])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].file_path == str(test_file)

    @pytest.mark.asyncio
    async def test_ingest_multiple_files(self, temp_directory, mock_file_processor):
        """Test ingesting multiple files."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        test_files = [
            temp_directory / "file1.py",
            temp_directory / "file2.py",
            temp_directory / "file3.md",
        ]
        results = await ingestion.ingest_files(test_files)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_progress_callback(self, temp_directory):
        """Test progress callback is called."""
        progress_updates = []

        def progress_callback(progress: IngestionProgress):
            progress_updates.append(progress.to_dict())

        ingestion = OptimizedAutoIngestion(
            progress_callback=progress_callback,
            batch_size=2,  # Small batch to trigger multiple updates
        )

        test_files = [
            temp_directory / "file1.py",
            temp_directory / "file2.py",
            temp_directory / "file3.md",
        ]
        await ingestion.ingest_files(test_files)

        # Should have at least one progress update
        assert len(progress_updates) > 0

    @pytest.mark.asyncio
    async def test_ingest_directory(self, temp_directory, mock_file_processor):
        """Test ingesting a directory."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        results = await ingestion.ingest_directory(
            temp_directory,
            patterns=["*.py"],
            recursive=True,
        )

        # Should find file1.py, file2.py, and subdir/nested.py
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_ingest_directory_non_recursive(self, temp_directory, mock_file_processor):
        """Test non-recursive directory ingestion."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        results = await ingestion.ingest_directory(
            temp_directory,
            patterns=["*.py"],
            recursive=False,
        )

        # Should only find file1.py and file2.py (not subdir/nested.py)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_ingest_directory_with_ignore_patterns(self, temp_directory, mock_file_processor):
        """Test directory ingestion with ignore patterns."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        results = await ingestion.ingest_directory(
            temp_directory,
            patterns=["*.py", "*.md"],
            ignore_patterns=["*.md"],
            recursive=True,
        )

        # Should find only .py files
        assert all(".py" in r.file_path for r in results)

    @pytest.mark.asyncio
    async def test_deduplication(self, temp_directory, mock_file_processor):
        """Test file deduplication within session."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        test_file = temp_directory / "file1.py"

        # Ingest same file twice
        await ingestion.ingest_files([test_file])
        await ingestion.ingest_files([test_file])

        stats = ingestion.get_statistics()
        assert stats.deduplication_hits == 1

    @pytest.mark.asyncio
    async def test_statistics_tracking(self, temp_directory, mock_file_processor):
        """Test statistics are tracked correctly."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        test_files = [
            temp_directory / "file1.py",
            temp_directory / "file2.py",
        ]
        await ingestion.ingest_files(test_files)

        stats = ingestion.get_statistics()
        assert stats.total_files_processed == 2
        assert stats.total_bytes_processed > 0

    @pytest.mark.asyncio
    async def test_reset_statistics(self, temp_directory, mock_file_processor):
        """Test statistics reset."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        await ingestion.ingest_files([temp_directory / "file1.py"])

        stats = ingestion.get_statistics()
        assert stats.total_files_processed == 1

        ingestion.reset_statistics()

        stats = ingestion.get_statistics()
        assert stats.total_files_processed == 0

    @pytest.mark.asyncio
    async def test_get_progress(self, temp_directory, mock_file_processor):
        """Test getting current progress."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )
        await ingestion.ingest_files([temp_directory / "file1.py"])

        progress = ingestion.get_progress()
        assert progress.total_files == 1
        assert progress.files_processed == 1

    @pytest.mark.asyncio
    async def test_cancel_processing(self, temp_directory):
        """Test cancellation of processing."""
        # Create many files to ensure processing takes time
        for i in range(50):
            (temp_directory / f"file_{i}.txt").write_text(f"content {i}")

        async def slow_processor(file_path: Path) -> dict:
            await asyncio.sleep(0.1)  # Simulate slow processing
            return {"processed": str(file_path)}

        ingestion = OptimizedAutoIngestion(
            file_processor=slow_processor,
            batch_size=5,
        )

        # Start processing in background
        async def process_and_cancel():
            task = asyncio.create_task(ingestion.ingest_directory(temp_directory))
            await asyncio.sleep(0.2)
            ingestion.cancel()
            return await task

        results = await process_and_cancel()

        # Should have processed some but not all files
        assert len(results) < 50

    @pytest.mark.asyncio
    async def test_priority_levels(self, temp_directory, mock_file_processor):
        """Test different priority levels."""
        ingestion = OptimizedAutoIngestion(
            file_processor=mock_file_processor,
        )

        test_file = temp_directory / "file1.py"

        # Test all priority levels
        for priority in IngestionPriority:
            await ingestion.ingest_files([test_file], priority=priority)

        stats = ingestion.get_statistics()
        assert len(stats.priority_distribution) == len(IngestionPriority)

    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        ingestion = OptimizedAutoIngestion()
        results = await ingestion.ingest_files(["/nonexistent/path/file.py"])

        # File doesn't exist so it's filtered out
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_invalid_directory_handling(self):
        """Test handling of invalid directory."""
        ingestion = OptimizedAutoIngestion()
        results = await ingestion.ingest_directory("/nonexistent/directory")

        assert results == []

    @pytest.mark.asyncio
    async def test_file_processor_exception_handling(self, temp_directory):
        """Test handling of file processor exceptions."""
        async def failing_processor(file_path: Path) -> dict:
            raise ValueError("Processing failed")

        ingestion = OptimizedAutoIngestion(
            file_processor=failing_processor,
        )

        results = await ingestion.ingest_files([temp_directory / "file1.py"])

        assert len(results) == 1
        assert results[0].success is False
        assert "Processing failed" in results[0].error_message


class TestFactoryFunction:
    """Tests for the create_optimized_ingestion factory function."""

    @pytest.mark.asyncio
    async def test_create_balanced_mode(self):
        """Test creating balanced mode ingestion."""
        ingestion = await create_optimized_ingestion(mode="balanced")
        assert isinstance(ingestion, OptimizedAutoIngestion)

    @pytest.mark.asyncio
    async def test_create_high_throughput_mode(self):
        """Test creating high throughput mode ingestion."""
        ingestion = await create_optimized_ingestion(mode="high_throughput")
        assert isinstance(ingestion, OptimizedAutoIngestion)

    @pytest.mark.asyncio
    async def test_create_memory_efficient_mode(self):
        """Test creating memory efficient mode ingestion."""
        ingestion = await create_optimized_ingestion(mode="memory_efficient")
        assert isinstance(ingestion, OptimizedAutoIngestion)

    @pytest.mark.asyncio
    async def test_create_with_callback(self):
        """Test creating with progress callback."""
        progress_updates = []

        def callback(progress: IngestionProgress):
            progress_updates.append(progress)

        ingestion = await create_optimized_ingestion(
            progress_callback=callback,
        )
        assert ingestion.progress_callback is not None


class TestBatchProcessing:
    """Tests for batch processing behavior."""

    @pytest.mark.asyncio
    async def test_batch_size_respected(self, temp_directory):
        """Test that batch size is respected."""
        batch_count = 0
        files_per_batch = []

        original_process_batch = OptimizedAutoIngestion._process_batch

        async def mock_process_batch(self, batch, priority, collection):
            nonlocal batch_count
            batch_count += 1
            files_per_batch.append(len(batch))
            return await original_process_batch(self, batch, priority, collection)

        # Create multiple test files
        for i in range(10):
            (temp_directory / f"file_{i}.py").write_text(f"content {i}")

        with patch.object(OptimizedAutoIngestion, '_process_batch', mock_process_batch):
            ingestion = OptimizedAutoIngestion(batch_size=3)
            await ingestion.ingest_directory(temp_directory, patterns=["*.py"])

        # Should have multiple batches
        assert batch_count > 1
        # Each batch should be <= batch_size
        assert all(size <= 3 for size in files_per_batch)

    @pytest.mark.asyncio
    async def test_concurrent_limit_respected(self, temp_directory):
        """Test that concurrent processing limit is respected."""
        concurrent_count = 0
        max_concurrent = 0

        async def counting_processor(file_path: Path) -> dict:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)  # Small delay to allow overlap
            concurrent_count -= 1
            return {"processed": str(file_path)}

        # Create test files
        for i in range(10):
            (temp_directory / f"file_{i}.py").write_text(f"content {i}")

        ingestion = OptimizedAutoIngestion(
            file_processor=counting_processor,
            max_concurrent_files=3,
            batch_size=10,
        )
        await ingestion.ingest_directory(temp_directory, patterns=["*.py"])

        # Max concurrent should not exceed limit
        assert max_concurrent <= 3
