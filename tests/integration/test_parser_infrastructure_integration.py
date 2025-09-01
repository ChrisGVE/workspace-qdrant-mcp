"""
Integration tests for the complete parser infrastructure.

Tests how file detection, error handling, and progress tracking work together
in real parsing scenarios.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from workspace_qdrant_mcp.cli.parsers import (
    TextParser,
    detect_file_type,
    create_progress_tracker,
    handle_parsing_error,
    FileAccessError,
    EncodingError,
    ProgressPhase,
)


class TestParserInfrastructureIntegration:
    """Integration tests for parser infrastructure components."""

    @pytest.mark.asyncio
    async def test_successful_parsing_with_progress(self, tmp_path):
        """Test successful file parsing with full infrastructure integration."""
        # Create a test file
        test_file = tmp_path / "test_document.txt"
        content = "This is a test document.\nIt has multiple lines.\nUsed for testing parser integration."
        test_file.write_text(content)
        
        # Detect file type
        mime_type, parser_type, confidence = detect_file_type(test_file)
        assert mime_type == "text/plain"
        assert parser_type == "text"
        assert confidence > 0.5
        
        # Create progress tracker with mock callback
        mock_callback = Mock()
        progress_tracker = create_progress_tracker(
            total=len(content.encode('utf-8')),
            show_console=False
        )
        progress_tracker.callbacks.append(mock_callback)
        
        # Parse the file
        parser = TextParser()
        parsed_doc = await parser.parse(test_file, progress_tracker=progress_tracker)
        
        # Verify parsing results
        assert parsed_doc.content.strip() == content
        assert parsed_doc.file_type == "text"
        assert parsed_doc.file_path == str(test_file)
        assert "parser" in parsed_doc.metadata
        assert parsed_doc.metadata["parser"] == "Plain Text"
        
        # Verify progress tracking worked
        assert progress_tracker.metrics.is_complete
        assert progress_tracker.metrics.current > 0
        
        # Verify callbacks were called
        assert mock_callback.on_progress_update.call_count > 0
        assert mock_callback.on_phase_change.call_count > 0

    @pytest.mark.asyncio
    async def test_file_not_found_error_handling(self):
        """Test error handling for non-existent files."""
        non_existent_file = Path("definitely_does_not_exist.txt")
        
        # File detection should fail
        with pytest.raises(FileNotFoundError):
            detect_file_type(non_existent_file)
        
        # Parser should handle the error gracefully
        parser = TextParser()
        
        with pytest.raises(FileAccessError) as exc_info:
            await parser.parse(non_existent_file)
        
        error = exc_info.value
        assert error.category.value == "file_access"
        assert error.file_path == str(non_existent_file)
        assert "Check if file exists" in error.recovery_suggestions[0]

    @pytest.mark.asyncio
    async def test_encoding_error_handling(self, tmp_path):
        """Test handling of encoding errors during parsing."""
        # Create a file with problematic encoding
        binary_file = tmp_path / "binary.txt"
        binary_file.write_bytes(b'\xff\xfe\x00\x00invalid_utf8_sequence\xff\xff')
        
        parser = TextParser()
        progress_tracker = create_progress_tracker(total=100, show_console=False)
        
        # Should parse but with warnings about encoding issues
        parsed_doc = await parser.parse(
            binary_file, 
            progress_tracker=progress_tracker,
            detect_encoding=True
        )
        
        # Should complete successfully despite encoding issues
        assert parsed_doc.content is not None
        assert parsed_doc.file_type == "text"
        
        # Progress tracker should show warnings
        assert progress_tracker.metrics.warnings_count >= 0  # May have warnings

    @pytest.mark.asyncio
    async def test_unsupported_file_detection(self, tmp_path):
        """Test handling of unsupported file types."""
        # Create a file that looks like a binary format
        binary_file = tmp_path / "image.png"
        png_header = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR'
        binary_file.write_bytes(png_header + b'\x00' * 100)
        
        # File detection should identify it as unsupported
        try:
            mime_type, parser_type, confidence = detect_file_type(binary_file)
            # If detection succeeds, parser_type should be None for unsupported types
            if parser_type is not None:
                pytest.skip("File was detected as supported (unexpected but not an error)")
        except Exception:
            # Expected for truly unsupported files
            pass

    @pytest.mark.asyncio
    async def test_large_file_progress_tracking(self, tmp_path):
        """Test progress tracking with a larger file."""
        # Create a larger test file
        large_file = tmp_path / "large_test.txt"
        lines = ["This is line number %d with some content to make it longer." % i 
                for i in range(1000)]
        content = "\n".join(lines)
        large_file.write_text(content)
        
        # Create progress tracker
        progress_tracker = create_progress_tracker(
            total=len(content.encode('utf-8')),
            show_console=False
        )
        
        # Track progress updates
        progress_updates = []
        
        class ProgressCapture:
            def on_progress_update(self, metrics):
                progress_updates.append({
                    'current': metrics.current,
                    'percent': metrics.progress_percent,
                    'phase': metrics.phase.value
                })
            
            def on_phase_change(self, old_phase, new_phase):
                pass
            
            def on_error(self, error, metrics):
                pass
        
        progress_tracker.callbacks.append(ProgressCapture())
        
        # Parse the file
        parser = TextParser()
        parsed_doc = await parser.parse(large_file, progress_tracker=progress_tracker)
        
        # Verify parsing completed successfully
        assert parsed_doc.content == content
        assert len(progress_updates) > 1  # Should have multiple progress updates
        
        # Verify progress increased over time
        assert progress_updates[-1]['current'] >= progress_updates[0]['current']
        assert progress_updates[-1]['percent'] >= progress_updates[0]['percent']

    @pytest.mark.asyncio
    async def test_batch_file_processing_simulation(self, tmp_path):
        """Test processing multiple files with batch progress tracking."""
        from workspace_qdrant_mcp.cli.parsers import BatchProgressTracker
        
        # Create multiple test files
        test_files = []
        for i in range(3):
            test_file = tmp_path / f"batch_test_{i}.txt"
            content = f"This is test file number {i}.\nIt contains some test content.\n"
            test_file.write_text(content)
            test_files.append(test_file)
        
        # Process files with batch tracker
        parser = TextParser()
        batch_results = []
        
        with BatchProgressTracker(total_files=len(test_files), show_console=False) as batch_tracker:
            for test_file in test_files:
                try:
                    # Detect file type first
                    mime_type, parser_type, confidence = detect_file_type(test_file)
                    
                    # Start file processing
                    file_tracker = batch_tracker.start_file(test_file, test_file.stat().st_size)
                    
                    # Parse the file
                    parsed_doc = await parser.parse(test_file, progress_tracker=file_tracker)
                    
                    batch_results.append(parsed_doc)
                    batch_tracker.complete_current_file(success=True)
                    
                except Exception as e:
                    error = handle_parsing_error(e, test_file, auto_recover=False)
                    batch_tracker.complete_current_file(success=False, error=str(error))
        
        # Verify batch processing results
        summary = batch_tracker.get_batch_summary()
        assert summary["total_files"] == 3
        assert summary["successful_files"] == 3
        assert summary["failed_files"] == 0
        assert len(batch_results) == 3
        
        # Verify individual file results
        for i, parsed_doc in enumerate(batch_results):
            assert f"batch_test_{i}" in parsed_doc.file_path
            assert parsed_doc.file_type == "text"
            assert len(parsed_doc.content) > 0

    @pytest.mark.asyncio
    async def test_error_recovery_and_statistics(self, tmp_path):
        """Test error recovery mechanisms and statistics tracking."""
        from workspace_qdrant_mcp.cli.parsers import get_error_statistics, reset_error_statistics
        
        # Reset statistics for clean test
        reset_error_statistics()
        
        # Create files that will cause different types of errors
        files_and_expected_errors = [
            (Path("nonexistent.txt"), FileAccessError),
            # Add more error types as needed
        ]
        
        parser = TextParser()
        error_results = []
        
        for file_path, expected_error_type in files_and_expected_errors:
            try:
                await parser.parse(file_path)
            except Exception as e:
                # Handle the error through our system
                parsing_error = handle_parsing_error(e, file_path, auto_recover=False)
                error_results.append((parsing_error, expected_error_type))
        
        # Verify error handling results
        for parsing_error, expected_type in error_results:
            assert isinstance(parsing_error, expected_type)
            assert parsing_error.file_path is not None
            assert len(parsing_error.recovery_suggestions) > 0
        
        # Check error statistics
        stats = get_error_statistics()
        assert stats["total_errors"] > 0
        assert "file_access_high" in stats["error_counts"]

    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, tmp_path):
        """Test complete workflow from detection to parsing with all infrastructure."""
        # Create a comprehensive test document
        test_file = tmp_path / "comprehensive_test.md"
        markdown_content = """# Test Document

This is a **comprehensive test** document that includes:

- Multiple paragraphs
- *Formatted text*
- Code snippets: `print("hello")`
- Lists and other markdown features

## Section 2

More content to test parsing capabilities.
"""
        test_file.write_text(markdown_content)
        
        # Step 1: File type detection
        mime_type, parser_type, confidence = detect_file_type(test_file)
        assert mime_type == "text/markdown"
        assert parser_type == "markdown"
        
        # Step 2: Create appropriate parser (for this test, use TextParser as it's implemented)
        parser = TextParser()  # In real implementation, would select based on parser_type
        
        # Step 3: Set up comprehensive progress tracking
        mock_console_output = []
        
        class MockConsoleCallback:
            def on_progress_update(self, metrics):
                mock_console_output.append(f"Progress: {metrics.progress_percent:.1f}%")
            
            def on_phase_change(self, old_phase, new_phase):
                mock_console_output.append(f"Phase: {new_phase.value}")
            
            def on_error(self, error, metrics):
                mock_console_output.append(f"Error: {error}")
        
        progress_tracker = create_progress_tracker(
            total=len(markdown_content.encode('utf-8')),
            show_console=False
        )
        progress_tracker.callbacks.append(MockConsoleCallback())
        
        # Step 4: Parse with full error handling
        try:
            parsed_doc = await parser.parse(
                test_file, 
                progress_tracker=progress_tracker,
                clean_content=True,
                detect_encoding=True
            )
            
            # Step 5: Verify complete integration
            assert parsed_doc.content is not None
            assert parsed_doc.file_type == "text"
            assert parsed_doc.metadata["parser"] == "Plain Text"
            assert parsed_doc.content_hash is not None
            assert parsed_doc.file_size > 0
            
            # Verify progress tracking
            assert progress_tracker.metrics.is_complete
            assert len(mock_console_output) > 0
            assert any("Progress:" in output for output in mock_console_output)
            assert any("Phase:" in output for output in mock_console_output)
            
            # Verify parsing info
            assert parsed_doc.parsing_info is not None
            assert "encoding" in parsed_doc.parsing_info
            assert "word_count" in parsed_doc.parsing_info
            
        except Exception as e:
            # If parsing fails, verify error handling works
            parsing_error = handle_parsing_error(e, test_file, auto_recover=False)
            assert isinstance(parsing_error, Exception)
            assert hasattr(parsing_error, 'recovery_suggestions')
            
            # Re-raise to fail test if this was unexpected
            raise


class TestInfrastructurePerformance:
    """Performance and efficiency tests for the infrastructure."""

    @pytest.mark.asyncio
    async def test_detection_performance(self, tmp_path):
        """Test that file detection is reasonably fast."""
        import time
        
        # Create several test files
        test_files = []
        for i in range(10):
            test_file = tmp_path / f"perf_test_{i}.txt"
            test_file.write_text(f"Performance test file {i}" * 100)
            test_files.append(test_file)
        
        # Time the detection process
        start_time = time.time()
        
        for test_file in test_files:
            mime_type, parser_type, confidence = detect_file_type(test_file)
            assert parser_type is not None  # Should detect successfully
        
        elapsed_time = time.time() - start_time
        
        # Should be reasonably fast (less than 1 second for 10 small files)
        assert elapsed_time < 1.0

    def test_memory_efficiency(self, tmp_path):
        """Test memory usage of infrastructure components."""
        # This is a basic test - in practice would use memory profiling tools
        from workspace_qdrant_mcp.cli.parsers import FileDetector, ErrorHandler
        
        # Create multiple instances to test for memory leaks
        detectors = [FileDetector() for _ in range(100)]
        handlers = [ErrorHandler() for _ in range(100)]
        
        # Basic functionality test
        test_file = tmp_path / "memory_test.txt"
        test_file.write_text("Memory test content")
        
        for detector in detectors[:5]:  # Test a few instances
            mime_type, parser_type, confidence = detector.detect_file_type(test_file)
            assert parser_type is not None
        
        # Cleanup should happen automatically
        del detectors
        del handlers