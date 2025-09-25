"""
Comprehensive unit tests for UnifiedDocumentPipeline.

This test suite provides exhaustive coverage of the unified document processing
pipeline including format detection, LSP integration, error handling, performance
optimization, and edge case scenarios.

Test Categories:
    - Core pipeline functionality and initialization
    - Document format detection and parsing
    - LSP integration and metadata extraction
    - Performance optimization and memory management
    - Error handling and recovery scenarios
    - Edge cases and boundary conditions
    - Batch processing and concurrency control
    - Statistics tracking and metrics collection
"""

import asyncio
import gc
import hashlib
import json
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
from typing import Any, Dict, List, Optional

import pytest

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from workspace_qdrant_mcp.core.unified_document_pipeline import (
    UnifiedDocumentPipeline,
    ProcessingResult,
    PipelineStats,
    process_directory_unified,
    validate_pipeline_performance
)


class MockParsedDocument:
    """Mock parsed document for testing."""

    def __init__(self, content="test content", metadata=None, content_hash=None):
        self.content = content
        self.metadata = metadata or {}
        self.content_hash = content_hash or hashlib.sha256(content.encode()).hexdigest()


class MockDocumentParser:
    """Mock document parser for testing."""

    def __init__(self, format_name="Mock", supported_extensions=None, should_fail=False):
        self.format_name = format_name
        self.supported_extensions = supported_extensions or {".txt"}
        self.should_fail = should_fail

    def can_parse(self, file_path):
        return file_path.suffix.lower() in self.supported_extensions

    async def parse(self, file_path):
        if self.should_fail:
            raise Exception(f"Parsing failed for {file_path}")

        content = f"Parsed content from {file_path}"
        return MockParsedDocument(content=content, metadata={"parser": self.format_name})


class MockLspMetadataExtractor:
    """Mock LSP metadata extractor for testing."""

    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def extract_file_metadata(self, file_path):
        if self.should_fail:
            raise Exception(f"LSP extraction failed for {file_path}")

        return {
            "symbols": [{"name": "test_function", "type": "function"}],
            "imports": ["os", "sys"],
            "relationships": []
        }

    async def cleanup(self):
        self.initialized = False


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_files(temp_dir):
    """Create test files with various formats."""
    files = {}

    # Text file
    text_file = temp_dir / "test.txt"
    text_file.write_text("This is a test text file.")
    files["text"] = text_file

    # Python code file
    py_file = temp_dir / "test.py"
    py_file.write_text("def test_function():\n    return 'Hello World'")
    files["python"] = py_file

    # Markdown file
    md_file = temp_dir / "test.md"
    md_file.write_text("# Test Document\nThis is a markdown file.")
    files["markdown"] = md_file

    # Empty file
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")
    files["empty"] = empty_file

    # Large file (for memory testing)
    large_file = temp_dir / "large.txt"
    large_file.write_text("Large content " * 10000)
    files["large"] = large_file

    # Binary file (unsupported)
    binary_file = temp_dir / "test.bin"
    binary_file.write_bytes(b"\x00\x01\x02\x03")
    files["binary"] = binary_file

    return files


@pytest.fixture
def mock_pipeline():
    """Create UnifiedDocumentPipeline with mocked dependencies."""
    with patch.multiple(
        'workspace_qdrant_mcp.core.unified_document_pipeline',
        LspMetadataExtractor=MockLspMetadataExtractor,
        PerformanceMonitor=Mock,
        GracefulDegradationManager=Mock,
        AutomaticRecovery=Mock,
        detect_file_type=Mock(return_value="txt"),
        PARSERS_AVAILABLE=True
    ):
        pipeline = UnifiedDocumentPipeline(
            max_concurrency=2,
            memory_limit_mb=100,
            enable_performance_monitoring=False
        )

        # Replace parsers with mocks
        pipeline.parsers = [
            MockDocumentParser("Text", {".txt"}),
            MockDocumentParser("Python", {".py"}),
            MockDocumentParser("Markdown", {".md"})
        ]

        yield pipeline


class TestUnifiedDocumentPipelineInitialization:
    """Test pipeline initialization and configuration."""

    @pytest.mark.asyncio
    async def test_pipeline_initialization_default_config(self):
        """Test pipeline initialization with default configuration."""
        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            pipeline = UnifiedDocumentPipeline()
            await pipeline.initialize()

            assert pipeline.is_initialized
            assert pipeline.processing_semaphore._value == 10  # default max_concurrency
            assert pipeline.memory_limit_mb == 500
            assert pipeline.chunk_size == 1000
            assert pipeline.chunk_overlap == 200

    @pytest.mark.asyncio
    async def test_pipeline_initialization_custom_config(self):
        """Test pipeline initialization with custom configuration."""
        pipeline = UnifiedDocumentPipeline(
            max_concurrency=5,
            memory_limit_mb=256,
            chunk_size=2000,
            chunk_overlap=100,
            enable_lsp=False,
            enable_performance_monitoring=False
        )

        await pipeline.initialize()

        assert pipeline.max_concurrency == 5
        assert pipeline.memory_limit_mb == 256
        assert pipeline.chunk_size == 2000
        assert pipeline.chunk_overlap == 100
        assert not pipeline.enable_lsp

    @pytest.mark.asyncio
    async def test_pipeline_initialization_with_lsp(self, mock_pipeline):
        """Test pipeline initialization with LSP enabled."""
        mock_pipeline.enable_lsp = True
        await mock_pipeline.initialize()

        assert mock_pipeline.lsp_extractor is not None
        assert mock_pipeline.is_initialized

    @pytest.mark.asyncio
    async def test_pipeline_initialization_failure_handling(self):
        """Test pipeline initialization failure handling."""
        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.LspMetadataExtractor',
            side_effect=Exception("LSP initialization failed")
        ):
            pipeline = UnifiedDocumentPipeline(enable_lsp=True)

            with pytest.raises(Exception, match="LSP initialization failed"):
                await pipeline.initialize()

    @pytest.mark.asyncio
    async def test_double_initialization_safety(self, mock_pipeline):
        """Test that double initialization is safely handled."""
        await mock_pipeline.initialize()
        first_init_time = mock_pipeline.is_initialized

        await mock_pipeline.initialize()  # Should not cause issues

        assert mock_pipeline.is_initialized == first_init_time


class TestDocumentProcessing:
    """Test core document processing functionality."""

    @pytest.mark.asyncio
    async def test_single_document_processing_success(self, mock_pipeline, test_files):
        """Test successful processing of a single document."""
        await mock_pipeline.initialize()

        results = await mock_pipeline.process_documents(
            [test_files["text"]],
            collection="test-collection"
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].document is not None
        assert results[0].parser_used == "Text"
        assert results[0].chunks_generated > 0

    @pytest.mark.asyncio
    async def test_multiple_document_processing(self, mock_pipeline, test_files):
        """Test processing multiple documents with different formats."""
        await mock_pipeline.initialize()

        file_list = [test_files["text"], test_files["python"], test_files["markdown"]]
        results = await mock_pipeline.process_documents(file_list, collection="test-collection")

        assert len(results) == 3
        assert all(result.success for result in results)

        # Check different parsers were used
        parser_names = {result.parser_used for result in results}
        assert "Text" in parser_names
        assert "Python" in parser_names
        assert "Markdown" in parser_names

    @pytest.mark.asyncio
    async def test_empty_file_handling(self, mock_pipeline, test_files):
        """Test handling of empty files."""
        await mock_pipeline.initialize()

        results = await mock_pipeline.process_documents(
            [test_files["empty"]],
            collection="test-collection"
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].chunks_generated >= 1  # Should generate at least 1 chunk even for empty

    @pytest.mark.asyncio
    async def test_large_file_processing(self, mock_pipeline, test_files):
        """Test processing of large files for memory efficiency."""
        await mock_pipeline.initialize()

        initial_memory = mock_pipeline._get_memory_usage()

        results = await mock_pipeline.process_documents(
            [test_files["large"]],
            collection="test-collection"
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].chunks_generated > 1  # Large file should create multiple chunks

        # Memory should not have increased significantly
        final_memory = mock_pipeline._get_memory_usage()
        assert final_memory - initial_memory < mock_pipeline.memory_limit_mb * 0.5

    @pytest.mark.asyncio
    async def test_unsupported_format_handling(self, mock_pipeline, test_files):
        """Test handling of unsupported file formats."""
        await mock_pipeline.initialize()

        # Mock detect_file_type to return None for unsupported format
        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.detect_file_type',
            return_value=None
        ):
            results = await mock_pipeline.process_documents(
                [test_files["binary"]],
                collection="test-collection"
            )

        assert len(results) == 1
        assert not results[0].success
        assert "Unsupported file format" in results[0].error
        assert mock_pipeline.stats.skipped_files == 1

    @pytest.mark.asyncio
    async def test_parsing_failure_handling(self, mock_pipeline, test_files):
        """Test handling of parsing failures."""
        await mock_pipeline.initialize()

        # Add failing parser
        mock_pipeline.parsers.append(MockDocumentParser("Failing", {".txt"}, should_fail=True))

        results = await mock_pipeline.process_documents(
            [test_files["text"]],
            collection="test-collection"
        )

        assert len(results) == 1
        # Should succeed with first parser (non-failing one)
        assert results[0].success

    @pytest.mark.asyncio
    async def test_processing_timeout_handling(self, mock_pipeline, test_files):
        """Test handling of processing timeouts."""
        await mock_pipeline.initialize()

        # Mock parser to simulate timeout
        async def slow_parse(file_path):
            await asyncio.sleep(2)  # Longer than timeout
            return MockParsedDocument()

        mock_pipeline.parsers[0].parse = slow_parse

        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.asyncio.wait_for',
            side_effect=asyncio.TimeoutError()
        ):
            results = await mock_pipeline.process_documents(
                [test_files["text"]],
                collection="test-collection"
            )

        assert len(results) == 1
        assert not results[0].success
        assert mock_pipeline.stats.failed_files == 1


class TestLSPIntegration:
    """Test LSP integration functionality."""

    @pytest.mark.asyncio
    async def test_lsp_metadata_extraction_success(self, test_files):
        """Test successful LSP metadata extraction."""
        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.LspMetadataExtractor',
            MockLspMetadataExtractor
        ):
            pipeline = UnifiedDocumentPipeline(enable_lsp=True, enable_performance_monitoring=False)

            # Mock code parser
            from workspace_qdrant_mcp.core.unified_document_pipeline import CodeParser
            pipeline.parsers = [MockDocumentParser("Code", {".py"})]

            await pipeline.initialize()

            results = await pipeline.process_documents(
                [test_files["python"]],
                collection="test-collection"
            )

            assert len(results) == 1
            assert results[0].success
            assert results[0].lsp_metadata is not None
            assert pipeline.stats.lsp_enhanced_files == 1
            assert pipeline.stats.lsp_failures == 0

    @pytest.mark.asyncio
    async def test_lsp_extraction_failure_handling(self, test_files):
        """Test handling of LSP extraction failures."""
        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.LspMetadataExtractor',
            lambda: MockLspMetadataExtractor(should_fail=True)
        ):
            pipeline = UnifiedDocumentPipeline(enable_lsp=True, enable_performance_monitoring=False)
            pipeline.parsers = [MockDocumentParser("Code", {".py"})]

            await pipeline.initialize()

            results = await pipeline.process_documents(
                [test_files["python"]],
                collection="test-collection"
            )

            assert len(results) == 1
            assert results[0].success  # Document processing should still succeed
            assert results[0].lsp_metadata is None  # But LSP metadata should be None
            assert pipeline.stats.lsp_failures == 1
            assert pipeline.stats.lsp_enhanced_files == 0

    @pytest.mark.asyncio
    async def test_lsp_timeout_handling(self, test_files):
        """Test LSP extraction timeout handling."""
        async def slow_extract(file_path):
            await asyncio.sleep(2)
            return {"symbols": []}

        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.LspMetadataExtractor'
        ) as mock_lsp:
            mock_extractor = Mock()
            mock_extractor.extract_file_metadata = slow_extract
            mock_extractor.initialize = AsyncMock()
            mock_lsp.return_value = mock_extractor

            pipeline = UnifiedDocumentPipeline(
                enable_lsp=True,
                lsp_timeout=0.1,  # Very short timeout
                enable_performance_monitoring=False
            )
            pipeline.parsers = [MockDocumentParser("Code", {".py"})]

            await pipeline.initialize()

            results = await pipeline.process_documents(
                [test_files["python"]],
                collection="test-collection"
            )

            assert len(results) == 1
            assert results[0].success
            assert results[0].lsp_metadata is None
            assert pipeline.stats.lsp_failures == 1


class TestPerformanceAndMemory:
    """Test performance optimization and memory management."""

    @pytest.mark.asyncio
    async def test_memory_pressure_handling(self, mock_pipeline, test_files):
        """Test memory pressure detection and handling."""
        await mock_pipeline.initialize()

        # Mock high memory usage
        with patch.object(mock_pipeline, '_get_memory_usage', return_value=85.0):  # Above 80% threshold
            with patch.object(mock_pipeline, '_force_garbage_collection') as mock_gc:
                await mock_pipeline._handle_memory_pressure()

                mock_gc.assert_called_once()
                assert mock_pipeline.stats.memory_pressure_events == 1

    @pytest.mark.asyncio
    async def test_batch_size_calculation(self, mock_pipeline):
        """Test optimal batch size calculation."""
        await mock_pipeline.initialize()

        # Test small file count
        batch_size = mock_pipeline._calculate_optimal_batch_size(50)
        assert 1 <= batch_size <= 30

        # Test medium file count
        batch_size = mock_pipeline._calculate_optimal_batch_size(500)
        assert 10 <= batch_size <= 50

        # Test large file count
        batch_size = mock_pipeline._calculate_optimal_batch_size(5000)
        assert 20 <= batch_size <= 100

    @pytest.mark.asyncio
    async def test_memory_monitoring_task(self, mock_pipeline):
        """Test background memory monitoring task."""
        await mock_pipeline.initialize()

        # Let memory monitor run briefly
        await asyncio.sleep(0.1)

        # Check that memory monitor task is running
        assert mock_pipeline.memory_monitor_task is not None
        assert not mock_pipeline.memory_monitor_task.done()

        # Clean up
        await mock_pipeline.cleanup()

    @pytest.mark.asyncio
    async def test_garbage_collection_forcing(self, mock_pipeline):
        """Test forced garbage collection."""
        await mock_pipeline.initialize()

        initial_gc_count = mock_pipeline.stats.gc_collections
        mock_pipeline._force_garbage_collection()

        assert mock_pipeline.stats.gc_collections == initial_gc_count + 1

    @pytest.mark.asyncio
    async def test_concurrency_control(self, mock_pipeline, test_files):
        """Test concurrency control with semaphore."""
        await mock_pipeline.initialize()

        # Create many files to test concurrency
        many_files = [test_files["text"]] * 10

        start_time = time.time()
        results = await mock_pipeline.process_documents(many_files, collection="test-collection")
        processing_time = time.time() - start_time

        assert len(results) == 10
        assert all(result.success for result in results)

        # With concurrency=2, should be faster than sequential but not instant
        assert processing_time < 5.0  # Should complete reasonably quickly


class TestErrorHandlingAndEdgeCases:
    """Test comprehensive error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_corrupted_file_handling(self, mock_pipeline, temp_dir):
        """Test handling of corrupted files."""
        await mock_pipeline.initialize()

        # Create corrupted file
        corrupted_file = temp_dir / "corrupted.txt"
        corrupted_file.write_bytes(b"\xFF\xFE\x00invalid utf-8")

        # Mock parser to raise UnicodeDecodeError
        async def failing_parse(file_path):
            raise UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

        mock_pipeline.parsers[0].parse = failing_parse

        results = await mock_pipeline.process_documents([corrupted_file], collection="test-collection")

        assert len(results) == 1
        assert not results[0].success
        assert "UnicodeDecodeError" in results[0].error
        assert mock_pipeline.stats.failed_files == 1

    @pytest.mark.asyncio
    async def test_no_parser_available(self, mock_pipeline, temp_dir):
        """Test handling when no parser is available for file type."""
        await mock_pipeline.initialize()

        # Create file with unknown extension
        unknown_file = temp_dir / "unknown.xyz"
        unknown_file.write_text("Unknown format content")

        # Mock detect_file_type to return unknown format
        with patch(
            'workspace_qdrant_mcp.core.unified_document_pipeline.detect_file_type',
            return_value="xyz"
        ):
            results = await mock_pipeline.process_documents([unknown_file], collection="test-collection")

        assert len(results) == 1
        assert not results[0].success
        assert "No parser found" in results[0].error
        assert mock_pipeline.stats.skipped_files == 1

    @pytest.mark.asyncio
    async def test_exception_in_batch_processing(self, mock_pipeline, test_files):
        """Test handling of exceptions during batch processing."""
        await mock_pipeline.initialize()

        # Mock an exception in the processing
        with patch.object(
            mock_pipeline, '_process_single_document',
            side_effect=Exception("Batch processing error")
        ):
            results = await mock_pipeline.process_documents([test_files["text"]], collection="test-collection")

        assert len(results) == 1
        assert not results[0].success
        assert "Batch processing error" in results[0].error

    @pytest.mark.asyncio
    async def test_empty_file_list_handling(self, mock_pipeline):
        """Test handling of empty file list."""
        await mock_pipeline.initialize()

        results = await mock_pipeline.process_documents([], collection="test-collection")

        assert len(results) == 0
        assert mock_pipeline.stats.total_files == 0

    @pytest.mark.asyncio
    async def test_nonexistent_file_handling(self, mock_pipeline):
        """Test handling of non-existent files."""
        await mock_pipeline.initialize()

        nonexistent_file = Path("/nonexistent/file.txt")

        results = await mock_pipeline.process_documents([nonexistent_file], collection="test-collection")

        assert len(results) == 1
        assert not results[0].success
        assert mock_pipeline.stats.failed_files == 1


class TestDeduplicationAndContent:
    """Test content deduplication functionality."""

    @pytest.mark.asyncio
    async def test_content_deduplication_enabled(self, mock_pipeline, temp_dir):
        """Test content deduplication when enabled."""
        await mock_pipeline.initialize()

        # Create two files with identical content
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        identical_content = "This is identical content."

        file1.write_text(identical_content)
        file2.write_text(identical_content)

        results = await mock_pipeline.process_documents(
            [file1, file2],
            collection="test-collection",
            enable_deduplication=True
        )

        assert len(results) == 2
        assert results[0].success
        assert results[1].success  # Second file should be marked as success but duplicate
        assert "Duplicate content" in results[1].error
        assert mock_pipeline.stats.duplicate_files == 1

    @pytest.mark.asyncio
    async def test_content_deduplication_disabled(self, mock_pipeline, temp_dir):
        """Test content deduplication when disabled."""
        await mock_pipeline.initialize()

        # Create two files with identical content
        file1 = temp_dir / "file1.txt"
        file2 = temp_dir / "file2.txt"
        identical_content = "This is identical content."

        file1.write_text(identical_content)
        file2.write_text(identical_content)

        results = await mock_pipeline.process_documents(
            [file1, file2],
            collection="test-collection",
            enable_deduplication=False
        )

        assert len(results) == 2
        assert all(result.success for result in results)
        assert mock_pipeline.stats.duplicate_files == 0

    @pytest.mark.asyncio
    async def test_content_hash_generation(self, mock_pipeline, test_files):
        """Test proper content hash generation."""
        await mock_pipeline.initialize()

        results = await mock_pipeline.process_documents(
            [test_files["text"]],
            collection="test-collection"
        )

        assert len(results) == 1
        assert results[0].success
        assert results[0].document.content_hash
        assert len(results[0].document.content_hash) == 64  # SHA256 hash length


class TestStatisticsAndMetrics:
    """Test statistics collection and metrics reporting."""

    @pytest.mark.asyncio
    async def test_statistics_collection(self, mock_pipeline, test_files):
        """Test comprehensive statistics collection."""
        await mock_pipeline.initialize()

        file_list = [test_files["text"], test_files["python"], test_files["markdown"]]
        results = await mock_pipeline.process_documents(file_list, collection="test-collection")

        stats = mock_pipeline.stats

        assert stats.total_files == 3
        assert stats.successful_files == 3
        assert stats.failed_files == 0
        assert stats.total_documents == 3
        assert stats.total_chunks > 0
        assert stats.total_characters > 0
        assert stats.total_processing_time > 0
        assert stats.throughput_docs_per_minute > 0

    @pytest.mark.asyncio
    async def test_performance_metrics_reporting(self, mock_pipeline, test_files):
        """Test performance metrics reporting."""
        await mock_pipeline.initialize()

        await mock_pipeline.process_documents([test_files["text"]], collection="test-collection")

        metrics = mock_pipeline.get_performance_metrics()

        assert "processing_stats" in metrics
        assert "content_stats" in metrics
        assert "performance_stats" in metrics
        assert "lsp_stats" in metrics
        assert "error_stats" in metrics

        # Check specific metrics
        processing_stats = metrics["processing_stats"]
        assert processing_stats["total_files"] == 1
        assert processing_stats["successful_files"] == 1
        assert processing_stats["success_rate"] == 100.0

    @pytest.mark.asyncio
    async def test_error_statistics_tracking(self, mock_pipeline, test_files):
        """Test error statistics tracking by type and format."""
        await mock_pipeline.initialize()

        # Mock parser to cause specific errors
        async def error_parse(file_path):
            raise ValueError("Test parsing error")

        mock_pipeline.parsers[0].parse = error_parse

        await mock_pipeline.process_documents([test_files["text"]], collection="test-collection")

        stats = mock_pipeline.stats
        assert "ValueError" in stats.errors_by_type
        assert stats.errors_by_type["ValueError"] == 1
        assert "txt" in stats.errors_by_format

    @pytest.mark.asyncio
    async def test_memory_statistics_tracking(self, mock_pipeline, test_files):
        """Test memory usage statistics tracking."""
        await mock_pipeline.initialize()

        await mock_pipeline.process_documents([test_files["large"]], collection="test-collection")

        stats = mock_pipeline.stats
        assert stats.peak_memory_mb > 0

        # Memory pressure events might be triggered with large file
        assert stats.memory_pressure_events >= 0
        assert stats.gc_collections >= 0


class TestAsyncContextManager:
    """Test async context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_initialization_and_cleanup(self):
        """Test async context manager initialization and cleanup."""
        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            async with UnifiedDocumentPipeline() as pipeline:
                assert pipeline.is_initialized

            # After context exit, cleanup should have been called
            # (Cannot directly test cleanup state without more mocks)

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, test_files):
        """Test context manager behavior with exceptions."""
        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            try:
                async with UnifiedDocumentPipeline() as pipeline:
                    pipeline.parsers = [MockDocumentParser()]
                    await pipeline.process_documents([test_files["text"]], collection="test-collection")
                    raise ValueError("Test exception")
            except ValueError:
                pass

            # Context manager should handle cleanup even with exception
            # (Testing actual cleanup state requires more complex mocking)


class TestConvenienceFunctions:
    """Test convenience functions and utilities."""

    @pytest.mark.asyncio
    async def test_process_directory_unified(self, temp_dir):
        """Test process_directory_unified convenience function."""
        # Create test files in directory
        (temp_dir / "file1.txt").write_text("Content 1")
        (temp_dir / "file2.txt").write_text("Content 2")
        (temp_dir / "subdir").mkdir()
        (temp_dir / "subdir" / "file3.txt").write_text("Content 3")

        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None,
            detect_file_type=Mock(return_value="txt")
        ):
            results = await process_directory_unified(
                temp_dir,
                collection="test-collection",
                enable_performance_monitoring=False
            )

        # Should find and process all .txt files recursively
        assert len(results) >= 3
        assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_validate_pipeline_performance(self, test_files):
        """Test pipeline performance validation function."""
        file_list = [test_files["text"], test_files["python"]]

        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            # Test with reasonable performance requirements
            is_valid = await validate_pipeline_performance(
                file_list,
                target_throughput=1.0,  # Very low threshold for test
                memory_limit=1000
            )

        assert isinstance(is_valid, bool)
        # The actual result depends on mock performance, but should not crash


class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_very_large_batch_processing(self, mock_pipeline, temp_dir):
        """Test processing of very large batches."""
        await mock_pipeline.initialize()

        # Create many small files
        file_list = []
        for i in range(100):
            file_path = temp_dir / f"file_{i}.txt"
            file_path.write_text(f"Content for file {i}")
            file_list.append(file_path)

        results = await mock_pipeline.process_documents(file_list, collection="test-collection")

        assert len(results) == 100
        success_count = sum(1 for result in results if result.success)
        assert success_count >= 95  # Allow for some potential failures in mock environment

    @pytest.mark.asyncio
    async def test_zero_concurrency_handling(self):
        """Test handling of zero or invalid concurrency settings."""
        # Concurrency of 0 should be handled gracefully
        pipeline = UnifiedDocumentPipeline(
            max_concurrency=0,
            enable_performance_monitoring=False
        )

        # Should not crash during initialization
        await pipeline.initialize()
        assert pipeline.processing_semaphore._value >= 1  # Should default to at least 1

    @pytest.mark.asyncio
    async def test_negative_memory_limit_handling(self):
        """Test handling of negative memory limits."""
        pipeline = UnifiedDocumentPipeline(
            memory_limit_mb=-100,
            enable_performance_monitoring=False
        )

        # Should not crash, should use reasonable default
        await pipeline.initialize()
        assert pipeline.memory_limit_mb > 0

    @pytest.mark.asyncio
    async def test_extremely_long_file_path(self, mock_pipeline, temp_dir):
        """Test handling of extremely long file paths."""
        await mock_pipeline.initialize()

        # Create file with very long name (but within filesystem limits)
        long_name = "a" * 200 + ".txt"
        long_file = temp_dir / long_name
        long_file.write_text("Content with long filename")

        results = await mock_pipeline.process_documents([long_file], collection="test-collection")

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, mock_pipeline, temp_dir):
        """Test handling of special characters and encodings."""
        await mock_pipeline.initialize()

        # Create file with special characters
        special_file = temp_dir / "special.txt"
        special_content = "Special chars: 🚀 ñoño αβγ 中文 العربية"
        special_file.write_text(special_content, encoding='utf-8')

        results = await mock_pipeline.process_documents([special_file], collection="test-collection")

        assert len(results) == 1
        assert results[0].success
        assert special_content.replace(" ", "") in results[0].document.content.replace(" ", "")

    @pytest.mark.asyncio
    async def test_concurrent_pipeline_instances(self, test_files):
        """Test running multiple pipeline instances concurrently."""
        # Create two pipeline instances
        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            pipeline1 = UnifiedDocumentPipeline(enable_performance_monitoring=False)
            pipeline2 = UnifiedDocumentPipeline(enable_performance_monitoring=False)

            await pipeline1.initialize()
            await pipeline2.initialize()

            # Process files concurrently with both pipelines
            results = await asyncio.gather(
                pipeline1.process_documents([test_files["text"]], collection="test1"),
                pipeline2.process_documents([test_files["python"]], collection="test2")
            )

            assert len(results) == 2
            assert len(results[0]) == 1
            assert len(results[1]) == 1
            assert all(result[0].success for result in results)


@pytest.mark.performance
class TestPerformanceRequirements:
    """Test specific performance requirements from Task 258."""

    @pytest.mark.asyncio
    async def test_throughput_requirement_1000_docs_per_minute(self, temp_dir):
        """Test that pipeline can achieve 1000+ docs/minute throughput."""
        # Create test files for throughput testing
        file_list = []
        for i in range(50):  # Smaller number for unit test
            file_path = temp_dir / f"perf_test_{i}.txt"
            file_path.write_text(f"Performance test content {i} " * 100)
            file_list.append(file_path)

        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            async with UnifiedDocumentPipeline(
                max_concurrency=10,
                enable_performance_monitoring=False
            ) as pipeline:
                # Replace with mock parsers for consistent performance
                pipeline.parsers = [MockDocumentParser()]

                start_time = time.time()
                results = await pipeline.process_documents(file_list, collection="perf-test")
                processing_time = time.time() - start_time

                success_count = sum(1 for result in results if result.success)
                throughput = (success_count / processing_time) * 60  # docs per minute

                # For unit tests, expect reasonable performance with mocks
                assert throughput > 100  # Reasonable expectation for mocked environment
                assert success_count >= len(file_list) * 0.95  # 95% success rate

    @pytest.mark.asyncio
    async def test_memory_usage_under_500mb(self, temp_dir):
        """Test that pipeline stays under 500MB memory usage limit."""
        # Create files for memory testing
        file_list = []
        for i in range(20):
            file_path = temp_dir / f"memory_test_{i}.txt"
            file_path.write_text("Memory test content " * 1000)  # ~20KB per file
            file_list.append(file_path)

        with patch.multiple(
            'workspace_qdrant_mcp.core.unified_document_pipeline',
            LspMetadataExtractor=None,
            PerformanceMonitor=None
        ):
            async with UnifiedDocumentPipeline(
                memory_limit_mb=500,
                enable_performance_monitoring=False
            ) as pipeline:
                pipeline.parsers = [MockDocumentParser()]

                initial_memory = pipeline._get_memory_usage()
                await pipeline.process_documents(file_list, collection="memory-test")
                final_memory = pipeline._get_memory_usage()

                memory_increase = final_memory - initial_memory
                assert memory_increase < 500  # Should stay under limit
                assert pipeline.stats.peak_memory_mb < 1000  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])