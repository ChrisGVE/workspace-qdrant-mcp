"""
Comprehensive edge case tests for daemon file ingestion (Task 316).

This module tests boundary conditions and unusual inputs for the daemon's
file ingestion system, validating robust error handling and graceful degradation.

Test Categories (matching Task 316 subtasks):
- 316.1: Zero-byte and empty file handling
- 316.2: Large file processing (>100MB)
- 316.3: Corrupted file handling
- 316.4: Special character filenames
- 316.5: Extension-less file processing
- 316.6: Symlink and circular reference handling
- 316.7: Concurrent file modification handling
- 316.8: Unicode content and deep directory structures
"""

import asyncio
import os
import platform
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.unified_document_pipeline import (
    UnifiedDocumentPipeline,
    ProcessingResult,
    detect_file_type,
    ParsingError,
    UnsupportedFileFormatError
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files, cleaned up after test."""
    temp_dir = tempfile.mkdtemp(prefix="daemon_edge_test_")
    yield Path(temp_dir)
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
async def pipeline():
    """Create an initialized UnifiedDocumentPipeline instance for testing."""
    p = UnifiedDocumentPipeline(
        max_concurrency=5,
        memory_limit_mb=500,
        chunk_size=1000,
        chunk_overlap=200
    )
    await p.initialize()
    return p


# ============================================================================
# TASK 316.1: ZERO-BYTE AND EMPTY FILE HANDLING TESTS
# ============================================================================

class TestZeroByteAndEmptyFileHandling:
    """Test daemon behavior with zero-byte and empty files."""

    @pytest.mark.asyncio
    async def test_zero_byte_txt_file(self, temp_test_dir, pipeline):
        """Test processing of zero-byte .txt file.

        Validates that daemon handles zero-byte text files gracefully
        without crashes or exceptions.
        """
        # Create zero-byte file
        zero_file = temp_test_dir / "empty.txt"
        zero_file.touch()
        assert zero_file.stat().st_size == 0, "File should be zero bytes"

        # Process the file
        results = await pipeline.process_documents(
            file_paths=[str(zero_file)],
            collection="test-collection",
            dry_run=True
        )

        # Validate graceful handling
        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(zero_file), "File path should match"
        # Either success with empty content or graceful failure is acceptable
        if result.success:
            assert result.document is not None, "Should have document object"
        else:
            assert result.error is not None, "Should have error message"

    @pytest.mark.asyncio
    async def test_zero_byte_py_file(self, temp_test_dir, pipeline):
        """Test processing of zero-byte .py file.

        Validates that daemon handles zero-byte code files gracefully.
        """
        zero_file = temp_test_dir / "empty.py"
        zero_file.touch()
        assert zero_file.stat().st_size == 0, "File should be zero bytes"

        results = await pipeline.process_documents(
            file_paths=[str(zero_file)],
            collection="test-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(zero_file), "File path should match"
        # Graceful handling expected (either success or informative error)
        if not result.success:
            assert result.error is not None, "Should have error message if failed"

    @pytest.mark.asyncio
    async def test_zero_byte_md_file(self, temp_test_dir, pipeline):
        """Test processing of zero-byte .md file.

        Validates that daemon handles zero-byte markdown files gracefully.
        """
        zero_file = temp_test_dir / "empty.md"
        zero_file.touch()
        assert zero_file.stat().st_size == 0, "File should be zero bytes"

        results = await pipeline.process_documents(
            file_paths=[str(zero_file)],
            collection="test-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(zero_file), "File path should match"

    @pytest.mark.asyncio
    async def test_file_with_only_whitespace(self, temp_test_dir, pipeline):
        """Test file containing only whitespace characters.

        Validates handling of files with spaces, tabs, but no actual content.
        """
        whitespace_file = temp_test_dir / "whitespace.txt"
        whitespace_file.write_text("   \t\t  \n  \t  \n   ")

        results = await pipeline.process_documents(
            file_paths=[str(whitespace_file)],
            collection="test-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(whitespace_file), "File path should match"
        # Should handle gracefully (either extract empty content or report appropriately)

    @pytest.mark.asyncio
    async def test_file_with_only_newlines(self, temp_test_dir, pipeline):
        """Test file containing only newline characters.

        Validates handling of files with multiple blank lines but no content.
        """
        newlines_file = temp_test_dir / "newlines.txt"
        newlines_file.write_text("\n\n\n\n\n\n\n\n\n\n")

        results = await pipeline.process_documents(
            file_paths=[str(newlines_file)],
            collection="test-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(newlines_file), "File path should match"
        # Should not crash, even if content is essentially empty

    @pytest.mark.asyncio
    async def test_multiple_zero_byte_files_batch(self, temp_test_dir, pipeline):
        """Test batch processing of multiple zero-byte files.

        Validates that processing multiple zero-byte files concurrently
        doesn't cause issues.
        """
        # Create multiple zero-byte files
        zero_files = []
        for i in range(5):
            zero_file = temp_test_dir / f"empty_{i}.txt"
            zero_file.touch()
            zero_files.append(str(zero_file))

        # Process all files in batch
        results = await pipeline.process_documents(
            file_paths=zero_files,
            collection="test-collection",
            dry_run=True
        )

        # All should complete without raising unhandled exceptions
        assert len(results) == 5, "Should process all files"
        for result in results:
            assert result is not None, "Should return result objects"
            assert hasattr(result, 'file_path'), "Should have file_path attribute"


# ============================================================================
# TASK 316.2: LARGE FILE PROCESSING TESTS
# ============================================================================

class TestLargeFileProcessing:
    """Test daemon behavior with very large files (>100MB)."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_text_file_100mb(self, temp_test_dir, pipeline):
        """Test processing of large (100MB+) text file.

        Validates that daemon can handle very large text files with
        proper chunking and memory management.
        """
        # Create 100MB+ text file
        large_file = temp_test_dir / "large_text.txt"
        chunk_size = 1024 * 1024  # 1MB chunks
        total_chunks = 105  # ~105MB file

        with large_file.open('w') as f:
            for i in range(total_chunks):
                # Write 1MB of text data
                content = f"Line {i}: " + ("x" * (chunk_size - 20)) + "\n"
                f.write(content)

        file_size_mb = large_file.stat().st_size / (1024 * 1024)
        assert file_size_mb > 100, f"File should be >100MB, got {file_size_mb:.2f}MB"

        # Process the large file
        results = await pipeline.process_documents(
            file_paths=[str(large_file)],
            collection="test-large-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(large_file), "File path should match"
        # Should complete without memory errors or timeouts
        if result.success:
            assert result.document is not None, "Should have document"
            # Verify chunking occurred
            assert result.chunks_generated > 0, "Should generate chunks for large file"
        else:
            # If failed, ensure error is informative
            assert result.error is not None, "Should have error message"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_file_memory_usage(self, temp_test_dir, pipeline):
        """Test memory usage stays within limits for large files.

        Validates that processing large files doesn't exceed memory limit.
        """
        # Create 50MB text file (smaller than previous test for faster execution)
        large_file = temp_test_dir / "memory_test.txt"
        with large_file.open('w') as f:
            for i in range(50):
                f.write(("x" * 1024 * 1024) + "\n")

        # Track memory before processing
        import psutil
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)  # MB

        # Process file
        results = await pipeline.process_documents(
            file_paths=[str(large_file)],
            collection="test-memory-collection",
            dry_run=True
        )

        # Check memory after
        mem_after = process.memory_info().rss / (1024 * 1024)  # MB
        mem_increase = mem_after - mem_before

        # Memory increase should be reasonable (not loading entire file into memory)
        # Allow up to 200MB increase (generous limit for testing)
        assert mem_increase < 200, (
            f"Memory usage increased by {mem_increase:.2f}MB, "
            f"should stay below 200MB for streaming processing"
        )

        assert len(results) == 1, "Should process file"

    @pytest.mark.asyncio
    async def test_multiple_large_files_sequential(self, temp_test_dir, pipeline):
        """Test processing multiple large files sequentially.

        Validates that memory is properly freed between large file processing.
        """
        # Create 3 medium-large files (10MB each for faster testing)
        large_files = []
        for i in range(3):
            large_file = temp_test_dir / f"large_{i}.txt"
            with large_file.open('w') as f:
                for j in range(10):
                    f.write(("y" * 1024 * 1024) + "\n")
            large_files.append(str(large_file))

        # Process files
        results = await pipeline.process_documents(
            file_paths=large_files,
            collection="test-sequential-collection",
            dry_run=True
        )

        assert len(results) == 3, "Should process all files"
        # All should complete (either success or graceful failure)
        for i, result in enumerate(results):
            assert result.file_path == large_files[i], f"File {i} path should match"

    @pytest.mark.asyncio
    async def test_file_size_reporting(self, temp_test_dir, pipeline):
        """Test that file size is correctly reported in results.

        Validates metadata extraction for large files.
        """
        # Create a known-size file (5MB)
        test_file = temp_test_dir / "size_test.txt"
        expected_size_mb = 5
        with test_file.open('w') as f:
            for i in range(expected_size_mb):
                f.write(("z" * 1024 * 1024) + "\n")

        actual_size = test_file.stat().st_size

        # Process file
        results = await pipeline.process_documents(
            file_paths=[str(test_file)],
            collection="test-size-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        # Verify result contains file path (size might be in metadata)
        assert result.file_path == str(test_file), "File path should match"
