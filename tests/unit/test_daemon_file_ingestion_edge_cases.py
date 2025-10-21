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
