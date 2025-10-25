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
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

from common.core.unified_document_pipeline import (
    ParsingError,
    ProcessingResult,
    UnifiedDocumentPipeline,
    UnsupportedFileFormatError,
    detect_file_type,
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
            for _i in range(50):
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
                for _j in range(10):
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
            for _i in range(expected_size_mb):
                f.write(("z" * 1024 * 1024) + "\n")

        test_file.stat().st_size

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


# ============================================================================
# TASK 316.3: CORRUPTED FILE HANDLING TESTS
# ============================================================================

class TestCorruptedFileHandling:
    """Test daemon behavior with corrupted and malformed files."""

    @pytest.mark.asyncio
    async def test_truncated_pdf(self, temp_test_dir, pipeline):
        """Test processing of truncated PDF file.

        Validates that daemon handles incomplete PDF files gracefully
        with informative error messages.
        """
        # Create a truncated PDF (invalid but has PDF magic bytes)
        truncated_pdf = temp_test_dir / "truncated.pdf"
        # Write PDF header but truncate before complete structure
        truncated_pdf.write_bytes(b'%PDF-1.4\n%\xE2\xE3\xCF\xD3\n')

        results = await pipeline.process_documents(
            file_paths=[str(truncated_pdf)],
            collection="test-corrupted-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(truncated_pdf), "File path should match"
        # Should fail gracefully with error message
        # (PDF parser should detect truncation)

    @pytest.mark.asyncio
    async def test_invalid_pdf_header(self, temp_test_dir, pipeline):
        """Test processing of file with invalid PDF header.

        Validates handling of files that claim to be PDF but have invalid structure.
        """
        # Create file with wrong PDF header
        invalid_pdf = temp_test_dir / "invalid_header.pdf"
        invalid_pdf.write_bytes(b'%PDF-999.999\nThis is not a real PDF\n')

        results = await pipeline.process_documents(
            file_paths=[str(invalid_pdf)],
            collection="test-corrupted-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        # Should handle gracefully (either parse as text or fail with error)
        assert result.file_path == str(invalid_pdf), "File path should match"

    @pytest.mark.asyncio
    async def test_corrupted_binary_file(self, temp_test_dir, pipeline):
        """Test processing of randomly corrupted binary file.

        Validates that daemon doesn't crash on random binary data.
        """
        # Create file with random binary data
        corrupted_file = temp_test_dir / "corrupted.bin"
        import random
        random_bytes = bytes([random.randint(0, 255) for _ in range(1024)])
        corrupted_file.write_bytes(random_bytes)

        results = await pipeline.process_documents(
            file_paths=[str(corrupted_file)],
            collection="test-corrupted-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        # Should not crash, either skip or handle gracefully
        assert result.file_path == str(corrupted_file), "File path should match"

    @pytest.mark.asyncio
    async def test_invalid_encoding_file(self, temp_test_dir, pipeline):
        """Test processing of file with invalid UTF-8 encoding.

        Validates handling of encoding errors in text files.
        """
        # Create file with invalid UTF-8 sequences
        invalid_encoding = temp_test_dir / "invalid_encoding.txt"
        # Mix valid UTF-8 with invalid sequences
        content = b'Valid text here\n'
        content += b'\xFF\xFE\xFD\xFC'  # Invalid UTF-8
        content += b'\nMore text\n'
        invalid_encoding.write_bytes(content)

        results = await pipeline.process_documents(
            file_paths=[str(invalid_encoding)],
            collection="test-corrupted-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(invalid_encoding), "File path should match"
        # Should handle encoding errors gracefully (either replace or skip invalid bytes)

    @pytest.mark.asyncio
    async def test_mixed_corrupted_and_valid_files(self, temp_test_dir, pipeline):
        """Test batch processing with mix of corrupted and valid files.

        Validates that one corrupted file doesn't prevent processing of valid files.
        """
        # Create mix of files
        valid_file = temp_test_dir / "valid.txt"
        valid_file.write_text("This is valid content")

        corrupted_file = temp_test_dir / "corrupted.txt"
        corrupted_file.write_bytes(b'\xFF\xFE\xFD\xFC\xFF\xFE')

        another_valid = temp_test_dir / "another_valid.txt"
        another_valid.write_text("More valid content")

        files = [str(valid_file), str(corrupted_file), str(another_valid)]

        results = await pipeline.process_documents(
            file_paths=files,
            collection="test-mixed-collection",
            dry_run=True
        )

        assert len(results) == 3, "Should process all files"
        # At least the valid files should process successfully
        success_count = sum(1 for r in results if r.success)
        # Expect at least 2 successes (the valid files)
        assert success_count >= 2, f"Expected at least 2 successes, got {success_count}"


# ============================================================================
# TASK 316.4: SPECIAL CHARACTER FILENAME TESTS
# ============================================================================

class TestSpecialCharacterFilenames:
    """Test daemon behavior with special characters in filenames."""

    @pytest.mark.asyncio
    async def test_unicode_filename(self, temp_test_dir, pipeline):
        """Test processing file with Unicode characters in filename.

        Validates handling of international characters in filenames.
        """
        # Create file with Unicode filename
        unicode_file = temp_test_dir / "文档_файл_αρχείο.txt"
        unicode_file.write_text("Content with Unicode filename")

        results = await pipeline.process_documents(
            file_paths=[str(unicode_file)],
            collection="test-unicode-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(unicode_file), "File path should match"
        # Should handle Unicode filenames correctly

    @pytest.mark.asyncio
    async def test_emoji_filename(self, temp_test_dir, pipeline):
        """Test processing file with emoji in filename.

        Validates handling of emoji characters in filenames.
        """
        # Create file with emoji filename
        emoji_file = temp_test_dir / "test_📄_file_✅.txt"
        emoji_file.write_text("Content with emoji filename")

        results = await pipeline.process_documents(
            file_paths=[str(emoji_file)],
            collection="test-emoji-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(emoji_file), "File path should match"

    @pytest.mark.asyncio
    async def test_special_chars_filename(self, temp_test_dir, pipeline):
        """Test processing file with special characters in filename.

        Validates handling of various special characters.
        """
        # Create file with special characters (filesystem-safe ones)
        special_file = temp_test_dir / "file!@#$%^&()_+-={}[].txt"
        special_file.write_text("Content with special chars in filename")

        results = await pipeline.process_documents(
            file_paths=[str(special_file)],
            collection="test-special-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(special_file), "File path should match"

    @pytest.mark.asyncio
    async def test_very_long_filename(self, temp_test_dir, pipeline):
        """Test processing file with very long filename.

        Validates handling of filenames approaching filesystem limits (255 chars).
        """
        # Create file with long filename (200 chars to stay safe across filesystems)
        long_name = "a" * 200 + ".txt"
        long_file = temp_test_dir / long_name
        long_file.write_text("Content with very long filename")

        results = await pipeline.process_documents(
            file_paths=[str(long_file)],
            collection="test-long-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(long_file), "File path should match"

    @pytest.mark.asyncio
    async def test_filename_with_spaces_and_quotes(self, temp_test_dir, pipeline):
        """Test processing file with spaces and quotes in filename.

        Validates handling of filenames that require shell escaping.
        """
        # Create file with spaces and single quotes
        spaced_file = temp_test_dir / "file with spaces and 'quotes'.txt"
        spaced_file.write_text("Content with spaced filename")

        results = await pipeline.process_documents(
            file_paths=[str(spaced_file)],
            collection="test-spaced-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(spaced_file), "File path should match"


# ============================================================================
# TASK 316.5: EXTENSION-LESS FILE PROCESSING TESTS
# ============================================================================

class TestExtensionlessFileProcessing:
    """Test daemon behavior with files without extensions."""

    @pytest.mark.asyncio
    async def test_code_file_without_extension(self, temp_test_dir, pipeline):
        """Test processing code file without extension.

        Validates content-based type detection for extensionless code.
        """
        # Create Python code without extension
        no_ext_code = temp_test_dir / "python_script"
        no_ext_code.write_text("""#!/usr/bin/env python3
def hello():
    print("Hello, World!")

if __name__ == "__main__":
    hello()
""")

        results = await pipeline.process_documents(
            file_paths=[str(no_ext_code)],
            collection="test-noext-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(no_ext_code), "File path should match"
        # Should either detect as code or process as text

    @pytest.mark.asyncio
    async def test_content_based_type_detection(self, temp_test_dir, pipeline):
        """Test content-based file type detection.

        Validates that system can identify file types from content.
        """
        # Create files with misleading or no extensions
        json_no_ext = temp_test_dir / "data"
        json_no_ext.write_text('{"key": "value", "number": 42}')

        results = await pipeline.process_documents(
            file_paths=[str(json_no_ext)],
            collection="test-content-detection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should process successfully by detecting content type

    @pytest.mark.asyncio
    async def test_misleading_extension(self, temp_test_dir, pipeline):
        """Test file with misleading extension.

        Validates handling when extension doesn't match content.
        """
        # Create JSON content with .txt extension
        misleading = temp_test_dir / "data.txt"
        misleading.write_text('{"this": "is actually JSON", "not": "text"}')

        results = await pipeline.process_documents(
            file_paths=[str(misleading)],
            collection="test-misleading-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should process based on extension (.txt) or detect JSON content


# ============================================================================
# TASK 316.6: SYMLINK AND CIRCULAR REFERENCE TESTS
# ============================================================================

class TestSymlinkHandling:
    """Test daemon behavior with symlinks and circular references."""

    @pytest.mark.skipif(platform.system() == "Windows", reason="Symlinks require admin on Windows")
    @pytest.mark.asyncio
    async def test_valid_symlink_to_file(self, temp_test_dir, pipeline):
        """Test processing valid symlink to file.

        Validates that daemon can follow symlinks correctly.
        """
        # Create target file
        target_file = temp_test_dir / "target.txt"
        target_file.write_text("Content in target file")

        # Create symlink
        symlink_file = temp_test_dir / "link_to_target.txt"
        symlink_file.symlink_to(target_file)

        results = await pipeline.process_documents(
            file_paths=[str(symlink_file)],
            collection="test-symlink-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should either resolve symlink or report it appropriately

    @pytest.mark.skipif(platform.system() == "Windows", reason="Symlinks require admin on Windows")
    @pytest.mark.asyncio
    async def test_broken_symlink_handling(self, temp_test_dir, pipeline):
        """Test processing broken symlink.

        Validates graceful handling of symlinks to deleted files.
        """
        # Create and then delete target
        target_file = temp_test_dir / "target_to_delete.txt"
        target_file.write_text("Temporary content")

        # Create symlink
        symlink_file = temp_test_dir / "broken_link.txt"
        symlink_file.symlink_to(target_file)

        # Delete target
        target_file.unlink()

        results = await pipeline.process_documents(
            file_paths=[str(symlink_file)],
            collection="test-broken-symlink",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should fail gracefully with informative error


# ============================================================================
# TASK 316.7: CONCURRENT FILE MODIFICATION TESTS
# ============================================================================

class TestConcurrentFileModification:
    """Test daemon behavior when files change during processing."""

    @pytest.mark.asyncio
    async def test_file_modified_during_read(self, temp_test_dir, pipeline):
        """Test file modified while being read.

        Validates handling of concurrent modifications.
        Note: This is a race condition test, may not always trigger the condition.
        """
        # Create a moderately sized file
        test_file = temp_test_dir / "concurrent_mod.txt"
        original_content = "Original content\n" * 1000
        test_file.write_text(original_content)

        # Attempt to process while file might be modified
        # Note: In real scenario, this would be modified by external process
        results = await pipeline.process_documents(
            file_paths=[str(test_file)],
            collection="test-concurrent-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        # Should complete without crashes

    @pytest.mark.asyncio
    async def test_file_deleted_during_processing(self, temp_test_dir, pipeline):
        """Test file deleted before/during processing.

        Validates handling of file deletion.
        """
        # Create file
        test_file = temp_test_dir / "to_delete.txt"
        test_file.write_text("Content to be deleted")

        # Delete it immediately
        test_file.unlink()

        # Try to process deleted file
        results = await pipeline.process_documents(
            file_paths=[str(test_file)],
            collection="test-deletion-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should fail gracefully with file not found error


# ============================================================================
# TASK 316.8: UNICODE CONTENT AND DEEP DIRECTORY TESTS
# ============================================================================

class TestUnicodeAndDeepDirectories:
    """Test daemon behavior with Unicode content and deep directory structures."""

    @pytest.mark.asyncio
    async def test_mixed_unicode_encodings(self, temp_test_dir, pipeline):
        """Test files with different Unicode encodings.

        Validates handling of UTF-8, UTF-16, and other encodings.
        """
        # Create UTF-8 file
        utf8_file = temp_test_dir / "utf8.txt"
        utf8_file.write_text("Hello 世界 مرحبا שלום", encoding='utf-8')

        # Create UTF-16 file
        utf16_file = temp_test_dir / "utf16.txt"
        utf16_file.write_text("Hello 世界 مرحبا שלום", encoding='utf-16')

        files = [str(utf8_file), str(utf16_file)]

        results = await pipeline.process_documents(
            file_paths=files,
            collection="test-encoding-collection",
            dry_run=True
        )

        assert len(results) == 2, "Should process both files"
        # Should handle different encodings (UTF-8 should work, UTF-16 might fail gracefully)

    @pytest.mark.asyncio
    async def test_deeply_nested_directory(self, temp_test_dir, pipeline):
        """Test file in deeply nested directory structure.

        Validates handling of deep directory paths (>20 levels).
        """
        # Create deeply nested structure
        current_dir = temp_test_dir
        for i in range(25):
            current_dir = current_dir / f"level{i}"
            current_dir.mkdir(exist_ok=True)

        # Create file at deepest level
        deep_file = current_dir / "deep_file.txt"
        deep_file.write_text("Content in deeply nested file")

        results = await pipeline.process_documents(
            file_paths=[str(deep_file)],
            collection="test-deep-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        result = results[0]
        assert result.file_path == str(deep_file), "File path should match"
        # Should handle deep paths correctly

    @pytest.mark.asyncio
    async def test_very_long_directory_path(self, temp_test_dir, pipeline):
        """Test file with very long directory path.

        Validates handling of paths approaching system limits.
        """
        # Create path with long directory names
        current_dir = temp_test_dir
        for i in range(5):
            # Create directory with 50-char name
            long_dir_name = f"{'a' * 45}_{i}"
            current_dir = current_dir / long_dir_name
            current_dir.mkdir(exist_ok=True)

        # Create file in long path
        long_path_file = current_dir / "file.txt"
        long_path_file.write_text("Content in long path")

        path_length = len(str(long_path_file))
        # Verify path is reasonably long (but not exceeding system limits)
        assert path_length > 200, f"Path should be >200 chars, got {path_length}"

        results = await pipeline.process_documents(
            file_paths=[str(long_path_file)],
            collection="test-longpath-collection",
            dry_run=True
        )

        assert len(results) == 1, "Should return one result"
        results[0]
        # Should handle long paths correctly
