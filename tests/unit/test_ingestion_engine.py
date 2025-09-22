"""
Unit tests for the DocumentIngestionEngine.

Tests the batch processing engine including file discovery, concurrent processing,
deduplication, error handling, and statistics tracking.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from wqm_cli.cli.ingestion_engine import (
    DocumentIngestionEngine,
    IngestionResult,
    IngestionStats,
)
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient


class TestIngestionStats:
    """Test the IngestionStats data class."""

    def test_stats_initialization(self):
        """Test stats initialization with defaults."""
        stats = IngestionStats()

        assert stats.files_found == 0
        assert stats.files_processed == 0
        assert stats.files_failed == 0
        assert stats.errors == []
        assert stats.skipped_files == []
        assert stats.duplicates_found == 0

    def test_files_per_second_calculation(self):
        """Test processing rate calculation."""
        stats = IngestionStats()
        stats.files_processed = 10
        stats.processing_time = 5.0

        assert stats.files_per_second == 2.0

        # Test zero division protection
        stats.processing_time = 0
        assert stats.files_per_second == 0.0

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        stats = IngestionStats()
        stats.files_found = 10
        stats.files_processed = 8

        assert stats.success_rate == 80.0

        # Test zero division protection
        stats.files_found = 0
        assert stats.success_rate == 0.0


class TestIngestionResult:
    """Test the IngestionResult data class."""

    def test_result_creation(self):
        """Test result creation with all fields."""
        stats = IngestionStats()
        result = IngestionResult(
            success=True,
            stats=stats,
            collection="test-collection",
            message="Operation completed",
            dry_run=False,
        )

        assert result.success is True
        assert result.stats is stats
        assert result.collection == "test-collection"
        assert result.message == "Operation completed"
        assert result.dry_run is False


class TestDocumentIngestionEngine:
    """Test the main ingestion engine."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock workspace client."""
        client = AsyncMock(spec=QdrantWorkspaceClient)
        client.list_collections.return_value = ["test-collection"]
        return client

    @pytest.fixture
    def engine(self, mock_client):
        """Create an ingestion engine with mocked client."""
        return DocumentIngestionEngine(
            client=mock_client, concurrency=2, chunk_size=1000, chunk_overlap=100
        )

    def test_engine_initialization(self, mock_client):
        """Test engine initialization."""
        engine = DocumentIngestionEngine(
            client=mock_client, concurrency=5, chunk_size=2000, chunk_overlap=200
        )

        assert engine.client is mock_client
        assert engine.concurrency == 5
        assert engine.chunk_size == 2000
        assert engine.chunk_overlap == 200
        assert len(engine.parsers) == 3  # text, markdown, pdf
        assert engine.semaphore._value == 5

    @pytest.mark.asyncio
    async def test_find_files_recursive(self, engine):
        """Test file discovery with recursive search."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "file1.txt").write_text("content1")
            (tmppath / "file2.md").write_text("content2")
            (tmppath / "file3.pdf").touch()
            (tmppath / "ignored.xyz").touch()  # Unsupported format

            # Create subdirectory
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "file4.txt").write_text("content4")

            files = await engine._find_files(tmppath, None, True, None)

            assert len(files) == 4  # txt, md, pdf, and subdir txt
            file_names = [f.name for f in files]
            assert "file1.txt" in file_names
            assert "file2.md" in file_names
            assert "file3.pdf" in file_names
            assert "file4.txt" in file_names
            assert "ignored.xyz" not in file_names

    @pytest.mark.asyncio
    async def test_find_files_non_recursive(self, engine):
        """Test file discovery without recursion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "file1.txt").write_text("content1")

            # Create subdirectory
            subdir = tmppath / "subdir"
            subdir.mkdir()
            (subdir / "file2.txt").write_text("content2")

            files = await engine._find_files(tmppath, None, False, None)

            assert len(files) == 1
            assert files[0].name == "file1.txt"

    @pytest.mark.asyncio
    async def test_find_files_with_format_filter(self, engine):
        """Test file discovery with format filtering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create various file types
            (tmppath / "file1.txt").write_text("content1")
            (tmppath / "file2.md").write_text("content2")
            (tmppath / "file3.pdf").touch()

            # Filter for only markdown
            files = await engine._find_files(tmppath, ["markdown"], True, None)

            assert len(files) == 1
            assert files[0].suffix == ".md"

    @pytest.mark.asyncio
    async def test_find_files_with_exclusions(self, engine):
        """Test file discovery with exclusion patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test files
            (tmppath / "keep.txt").write_text("content1")
            (tmppath / "exclude.txt").write_text("content2")
            (tmppath / "temp.tmp").write_text("temp content")

            files = await engine._find_files(
                tmppath, None, True, ["exclude.txt", "*.tmp"]
            )

            assert len(files) == 1
            assert files[0].name == "keep.txt"

    @pytest.mark.asyncio
    async def test_process_directory_nonexistent(self, engine):
        """Test processing non-existent directory."""
        result = await engine.process_directory(
            "/nonexistent/path", collection="test-collection"
        )

        assert result.success is False
        assert "Directory not found" in result.message
        assert result.stats.files_found == 0

    @pytest.mark.asyncio
    async def test_process_directory_no_files(self, engine):
        """Test processing directory with no supported files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create unsupported file
            (Path(tmpdir) / "unsupported.xyz").touch()

            result = await engine.process_directory(
                tmpdir, collection="test-collection"
            )

            assert result.success is True
            assert "No processable files found" in result.message
            assert result.stats.files_found == 0

    @pytest.mark.asyncio
    async def test_process_directory_collection_not_found(self, engine, mock_client):
        """Test processing when target collection doesn't exist."""
        mock_client.list_collections.return_value = ["other-collection"]

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("content")

            result = await engine.process_directory(
                tmpdir, collection="missing-collection"
            )

            assert result.success is False
            assert "Collection 'missing-collection' not found" in result.message

    @pytest.mark.asyncio
    async def test_process_directory_successful(self, engine, mock_client):
        """Test successful directory processing."""
        # Mock add_document function
        with patch(
            "wqm_cli.cli.ingestion_engine.add_document"
        ) as mock_add:
            mock_add.return_value = {"points_added": 1, "document_id": "test-doc"}

            with tempfile.TemporaryDirectory() as tmpdir:
                (Path(tmpdir) / "test.txt").write_text("Hello world!")

                result = await engine.process_directory(
                    tmpdir, collection="test-collection"
                )

                assert result.success is True
                assert result.stats.files_found == 1
                assert result.stats.files_processed == 1
                assert result.stats.total_documents == 1
                assert result.stats.total_chunks == 1
                assert "Rate:" in result.message

    @pytest.mark.asyncio
    async def test_process_directory_dry_run(self, engine, mock_client):
        """Test dry run mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "test.txt").write_text("Hello world!")

            result = await engine.process_directory(
                tmpdir, collection="test-collection", dry_run=True
            )

            assert result.success is True
            assert result.dry_run is True
            assert result.stats.files_processed == 1
            assert "Analyzed" in result.message
            # Should not call add_document in dry run
            mock_client.list_collections.assert_not_called()

    @pytest.mark.asyncio
    async def test_deduplication(self, engine):
        """Test content deduplication."""
        with patch(
            "wqm_cli.cli.ingestion_engine.add_document"
        ) as mock_add:
            mock_add.return_value = {"points_added": 1}

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create two files with identical content
                (Path(tmpdir) / "file1.txt").write_text("Duplicate content")
                (Path(tmpdir) / "file2.txt").write_text("Duplicate content")
                (Path(tmpdir) / "file3.txt").write_text("Unique content")

                result = await engine.process_directory(
                    tmpdir, collection="test-collection"
                )

                assert result.success is True
                assert result.stats.files_found == 3
                assert result.stats.files_processed == 2  # One duplicate skipped
                assert result.stats.duplicates_found == 1

    @pytest.mark.asyncio
    async def test_processing_with_errors(self, engine):
        """Test handling of file processing errors."""
        # Mock parser to raise error for specific file
        original_find_parser = engine._find_parser

        def mock_find_parser(file_path):
            if "error" in file_path.name:
                return None  # No parser found
            return original_find_parser(file_path)

        engine._find_parser = mock_find_parser

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "good.txt").write_text("Good content")
            (Path(tmpdir) / "error.txt").write_text("Error content")

            result = await engine.process_directory(
                tmpdir,
                collection="test-collection",
                dry_run=True,  # Use dry run to avoid add_document complexity
            )

            assert result.success is True
            assert result.stats.files_found == 2
            assert result.stats.files_processed == 1
            assert result.stats.files_skipped == 1
            assert len(result.stats.skipped_files) == 1

    @pytest.mark.asyncio
    async def test_progress_callback(self, engine):
        """Test progress callback functionality."""
        progress_calls = []

        def progress_callback(completed, total, stats):
            progress_calls.append((completed, total, stats))

        with patch(
            "wqm_cli.cli.ingestion_engine.add_document"
        ) as mock_add:
            mock_add.return_value = {"points_added": 1}

            with tempfile.TemporaryDirectory() as tmpdir:
                (Path(tmpdir) / "file1.txt").write_text("content1")
                (Path(tmpdir) / "file2.txt").write_text("content2")

                await engine.process_directory(
                    tmpdir,
                    collection="test-collection",
                    progress_callback=progress_callback,
                )

                # Should have received progress updates
                assert len(progress_calls) > 0
                final_call = progress_calls[-1]
                assert final_call[1] == 2  # total files

    def test_find_parser(self, engine):
        """Test parser selection for different file types."""
        # Test text parser selection
        parser = engine._find_parser(Path("test.txt"))
        assert parser is not None
        assert parser.format_name == "Plain Text"

        # Test markdown parser selection
        parser = engine._find_parser(Path("test.md"))
        assert parser is not None
        assert parser.format_name == "Markdown"

        # Test no parser for unsupported format
        parser = engine._find_parser(Path("test.xyz"))
        assert parser is None

    @pytest.mark.asyncio
    async def test_get_supported_formats(self, engine):
        """Test getting supported format information."""
        formats = await engine.get_supported_formats()

        assert "Plain Text" in formats
        assert "Markdown" in formats
        assert "PDF Document" in formats

        # Check format details
        text_format = formats["Plain Text"]
        assert ".txt" in text_format["extensions"]
        assert "options" in text_format
        assert text_format["available"] is True

    @pytest.mark.asyncio
    async def test_estimate_processing_time(self, engine):
        """Test processing time estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files of different sizes
            (Path(tmpdir) / "small.txt").write_text("Small content")
            (Path(tmpdir) / "large.txt").write_text("Large content " * 1000)

            estimation = await engine.estimate_processing_time(tmpdir)

            assert estimation["files_found"] == 2
            assert estimation["total_size_mb"] > 0
            assert "Plain Text" in estimation["file_types"]
            assert estimation["file_types"]["Plain Text"] == 2
            assert estimation["estimated_time_seconds"] > 0
            assert "estimated_time_human" in estimation

    @pytest.mark.asyncio
    async def test_concurrent_processing(self, engine):
        """Test that files are processed concurrently."""
        process_times = []

        # Mock parser that tracks processing times
        original_process_single = engine._process_single_file

        async def timed_process_single(file_path, collection, stats, dry_run):
            import time

            start = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time
            result = await original_process_single(
                file_path, collection, stats, dry_run
            )
            end = time.time()
            process_times.append(end - start)
            return result

        engine._process_single_file = timed_process_single

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple files
            for i in range(4):
                (Path(tmpdir) / f"file{i}.txt").write_text(f"content{i}")

            start_time = asyncio.get_event_loop().time()

            await engine.process_directory(
                tmpdir, collection="test-collection", dry_run=True
            )

            end_time = asyncio.get_event_loop().time()
            total_time = end_time - start_time

            # With concurrency=2, should take roughly half the time
            # of sequential processing (plus some overhead)
            expected_sequential_time = 0.1 * 4  # 4 files * 0.1s each
            assert (
                total_time < expected_sequential_time * 0.8
            )  # Allow for some overhead


@pytest.mark.unit
class TestIngestionEngineIntegration:
    """Integration tests for the ingestion engine with real file operations."""

    @pytest.mark.asyncio
    async def test_full_processing_workflow(self):
        """Test complete processing workflow with real files."""
        mock_client = AsyncMock(spec=QdrantWorkspaceClient)
        mock_client.list_collections.return_value = ["test-collection"]

        engine = DocumentIngestionEngine(mock_client, concurrency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create various test files
            (tmppath / "text.txt").write_text("This is a text file with some content.")
            (tmppath / "markdown.md").write_text(
                "# Markdown File\n\nWith **bold** text."
            )

            # Create a subdirectory
            subdir = tmppath / "docs"
            subdir.mkdir()
            (subdir / "nested.txt").write_text("Nested file content.")

            # Mock the add_document function
            with patch(
                "wqm_cli.cli.ingestion_engine.add_document"
            ) as mock_add:
                mock_add.return_value = {"points_added": 1, "document_id": "test"}

                result = await engine.process_directory(
                    str(tmppath), collection="test-collection", recursive=True
                )

                assert result.success is True
                assert result.stats.files_found == 3
                assert result.stats.files_processed == 3
                assert result.stats.total_documents == 3
                assert result.stats.files_per_second > 0
                assert result.stats.success_rate == 100.0

                # Verify add_document was called for each file
                assert mock_add.call_count == 3
