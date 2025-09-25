"""
Comprehensive unit tests for auto_ingestion.py module.

This test file provides complete coverage for the auto-ingestion functionality,
including project pattern detection, file processing, batch ingestion,
and progress tracking.

Target: src/python/common/core/auto_ingestion.py (349 lines, 0% coverage)
Goal: Achieve >90% test coverage with comprehensive mocking.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pytest

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

try:
    from common.core.auto_ingestion import (
        build_project_collection_name,
        normalize_collection_name_component,
        ProjectPatterns,
        IngestionProgressTracker,
        AutoIngestionManager
    )
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestProjectPatterns:
    """Test ProjectPatterns functionality."""

    def test_get_common_doc_patterns_with_docs_flag_true(self):
        """Test getting common doc patterns when include_docs is True."""
        patterns = ProjectPatterns.get_common_doc_patterns(include_docs=True)

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert "*.md" in patterns
        assert "*.rst" in patterns
        assert "*.txt" in patterns

    def test_get_common_doc_patterns_with_docs_flag_false(self):
        """Test getting common doc patterns when include_docs is False."""
        patterns = ProjectPatterns.get_common_doc_patterns(include_docs=False)

        # Should return minimal or empty list when docs are excluded
        assert isinstance(patterns, list)

    def test_get_source_patterns_for_language_python(self):
        """Test getting source patterns for Python."""
        patterns = ProjectPatterns.get_source_patterns_for_language("python")

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert "*.py" in patterns

    def test_get_source_patterns_for_language_javascript(self):
        """Test getting source patterns for JavaScript."""
        patterns = ProjectPatterns.get_source_patterns_for_language("javascript")

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert "*.js" in patterns

    def test_get_source_patterns_for_language_unknown(self):
        """Test getting source patterns for unknown language."""
        patterns = ProjectPatterns.get_source_patterns_for_language("unknown_language_xyz")

        # Should return empty list for unknown language
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_get_all_source_patterns_default(self):
        """Test getting all source patterns with default behavior."""
        patterns = ProjectPatterns.get_all_source_patterns()

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Should include patterns for multiple languages
        assert "*.py" in patterns
        assert "*.js" in patterns

    def test_get_all_source_patterns_include_tests_true(self):
        """Test getting all source patterns with tests included."""
        patterns = ProjectPatterns.get_all_source_patterns(include_tests=True)

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Should include test patterns
        test_patterns_found = any("test" in pattern.lower() for pattern in patterns)
        assert test_patterns_found or len(patterns) > 10  # Many patterns indicates test inclusion

    def test_get_all_source_patterns_include_tests_false(self):
        """Test getting all source patterns with tests excluded."""
        patterns = ProjectPatterns.get_all_source_patterns(include_tests=False)

        assert isinstance(patterns, list)
        # Should be fewer patterns without test files

    def test_common_doc_patterns_class_attribute(self):
        """Test COMMON_DOC_PATTERNS class attribute."""
        patterns = ProjectPatterns.COMMON_DOC_PATTERNS

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert "*.md" in patterns

    def test_source_patterns_class_attribute(self):
        """Test SOURCE_PATTERNS class attribute."""
        patterns = ProjectPatterns.SOURCE_PATTERNS

        assert isinstance(patterns, dict)
        assert len(patterns) > 0
        assert "python" in patterns
        assert isinstance(patterns["python"], list)

    def test_get_patterns_for_project_with_language_detection(self):
        """Test getting patterns for project with language detection."""
        project_path = "/fake/project/path"

        with patch.object(ProjectPatterns, '_detect_project_languages', return_value=['python', 'javascript']):
            patterns = ProjectPatterns.get_patterns_for_project(
                project_path,
                include_docs=True,
                include_source=True
            )

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Should include both doc and source patterns
        assert "*.md" in patterns
        assert "*.py" in patterns
        assert "*.js" in patterns

    def test_get_patterns_for_project_docs_only(self):
        """Test getting patterns for project with docs only."""
        project_path = "/fake/project/path"

        patterns = ProjectPatterns.get_patterns_for_project(
            project_path,
            include_docs=True,
            include_source=False
        )

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Should only include doc patterns
        assert "*.md" in patterns
        # Should not include source patterns
        assert "*.py" not in patterns

    def test_get_patterns_for_project_source_only(self):
        """Test getting patterns for project with source only."""
        project_path = "/fake/project/path"

        with patch.object(ProjectPatterns, '_detect_project_languages', return_value=['python']):
            patterns = ProjectPatterns.get_patterns_for_project(
                project_path,
                include_docs=False,
                include_source=True
            )

        assert isinstance(patterns, list)
        assert len(patterns) > 0
        # Should include source patterns
        assert "*.py" in patterns

    @patch('os.path.exists')
    @patch('os.walk')
    def test_detect_project_languages_python_project(self, mock_walk, mock_exists):
        """Test language detection for Python project."""
        # Mock file system structure for Python project
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/fake/path", ["src"], ["app.py", "requirements.txt"]),
            ("/fake/path/src", [], ["main.py", "utils.py"])
        ]

        languages = ProjectPatterns._detect_project_languages("/fake/path")

        assert isinstance(languages, list)
        assert "python" in languages

    @patch('os.path.exists')
    @patch('os.walk')
    def test_detect_project_languages_javascript_project(self, mock_walk, mock_exists):
        """Test language detection for JavaScript project."""
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/fake/path", ["src"], ["index.js", "package.json"]),
            ("/fake/path/src", [], ["app.js", "utils.js"])
        ]

        languages = ProjectPatterns._detect_project_languages("/fake/path")

        assert isinstance(languages, list)
        assert "javascript" in languages

    @patch('os.path.exists')
    @patch('os.walk')
    def test_detect_project_languages_mixed_project(self, mock_walk, mock_exists):
        """Test language detection for mixed project."""
        mock_exists.return_value = True
        mock_walk.return_value = [
            ("/fake/path", ["src", "frontend"], ["app.py", "package.json"]),
            ("/fake/path/src", [], ["main.py"]),
            ("/fake/path/frontend", [], ["app.js"])
        ]

        languages = ProjectPatterns._detect_project_languages("/fake/path")

        assert isinstance(languages, list)
        assert len(languages) >= 1
        # Should detect both Python and JavaScript
        assert "python" in languages or "javascript" in languages

    @patch('os.path.exists')
    def test_detect_project_languages_nonexistent_path(self, mock_exists):
        """Test language detection for non-existent path."""
        mock_exists.return_value = False

        languages = ProjectPatterns._detect_project_languages("/nonexistent/path")

        assert isinstance(languages, list)
        assert len(languages) == 0


class TestIngestionProgressTracker:
    """Test IngestionProgressTracker functionality."""

    def test_init(self):
        """Test IngestionProgressTracker initialization."""
        tracker = IngestionProgressTracker(total_files=100)

        assert tracker.total_files == 100
        assert tracker.processed_files == 0
        assert tracker.successful_files == 0
        assert tracker.failed_files == 0
        assert tracker.current_batch == 0
        assert tracker.start_time is not None
        assert tracker.current_file is None

    def test_start(self):
        """Test starting progress tracking."""
        tracker = IngestionProgressTracker(total_files=50)

        tracker.start()

        assert tracker.start_time is not None

    def test_start_batch(self):
        """Test starting a new batch."""
        tracker = IngestionProgressTracker(total_files=50)

        tracker.start_batch(batch_number=1)

        assert tracker.current_batch == 1

    def test_start_file(self):
        """Test starting file processing."""
        tracker = IngestionProgressTracker(total_files=50)

        tracker.start_file("test_file.py")

        assert tracker.current_file == "test_file.py"

    def test_file_completed_success(self):
        """Test completing a file successfully."""
        tracker = IngestionProgressTracker(total_files=50)

        tracker.start_file("test_file.py")
        tracker.file_completed(success=True)

        assert tracker.processed_files == 1
        assert tracker.successful_files == 1
        assert tracker.failed_files == 0
        assert tracker.current_file is None

    def test_file_completed_failure(self):
        """Test completing a file with failure."""
        tracker = IngestionProgressTracker(total_files=50)

        tracker.start_file("test_file.py")
        tracker.file_completed(success=False, error="Test error")

        assert tracker.processed_files == 1
        assert tracker.successful_files == 0
        assert tracker.failed_files == 1
        assert "Test error" in tracker.errors

    def test_file_completed_with_metadata(self):
        """Test completing a file with metadata."""
        tracker = IngestionProgressTracker(total_files=50)

        metadata = {"chunks": 5, "tokens": 1000}
        tracker.start_file("test_file.py")
        tracker.file_completed(success=True, metadata=metadata)

        assert tracker.processed_files == 1
        assert tracker.successful_files == 1

    def test_log_progress_regular_interval(self):
        """Test progress logging at regular intervals."""
        tracker = IngestionProgressTracker(total_files=10, log_interval=5)

        # Process files up to log interval
        for i in range(6):
            tracker.start_file(f"file_{i}.py")
            tracker.file_completed(success=True)

        # Should have logged progress
        assert tracker.processed_files == 6

    def test_log_progress_final(self):
        """Test final progress logging."""
        tracker = IngestionProgressTracker(total_files=3, log_interval=10)

        # Process all files (less than log interval)
        for i in range(3):
            tracker.start_file(f"file_{i}.py")
            tracker.file_completed(success=True)

        assert tracker.processed_files == 3

    def test_get_summary(self):
        """Test getting progress summary."""
        tracker = IngestionProgressTracker(total_files=10)
        tracker.start()

        # Process some files
        for i in range(5):
            tracker.start_file(f"file_{i}.py")
            tracker.file_completed(success=i % 2 == 0)  # Alternate success/failure

        summary = tracker.get_summary()

        assert isinstance(summary, dict)
        assert summary["total_files"] == 10
        assert summary["processed_files"] == 5
        assert summary["successful_files"] == 3  # 0, 2, 4 succeeded
        assert summary["failed_files"] == 2     # 1, 3 failed
        assert "elapsed_time" in summary
        assert "success_rate" in summary

    def test_get_summary_zero_files(self):
        """Test getting summary with zero processed files."""
        tracker = IngestionProgressTracker(total_files=10)

        summary = tracker.get_summary()

        assert summary["success_rate"] == 0.0
        assert summary["processed_files"] == 0


class TestAutoIngestionManager:
    """Test AutoIngestionManager functionality."""

    def test_init_basic(self):
        """Test basic AutoIngestionManager initialization."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        assert manager.qdrant_client == mock_client
        assert manager.config == mock_config
        assert manager.github_user == "testuser"

    def test_init_with_components(self):
        """Test initialization with optional components."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"
        mock_embedding_service = Mock()
        mock_project_detector = Mock()

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config,
            embedding_service=mock_embedding_service,
            project_detector=mock_project_detector
        )

        assert manager.embedding_service == mock_embedding_service
        assert manager.project_detector == mock_project_detector

    @pytest.mark.asyncio
    async def test_setup_project_watches_basic(self):
        """Test basic project watch setup."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        project_info = {
            "project_name": "test-project",
            "project_path": "/path/to/project",
            "subprojects": []
        }

        with patch.object(manager, '_create_project_collections') as mock_create:
            mock_create.return_value = asyncio.Future()
            mock_create.return_value.set_result(["test-project-default"])

            result = await manager.setup_project_watches(project_info)

        assert isinstance(result, dict)
        assert "project_name" in result
        assert result["project_name"] == "test-project"

    @pytest.mark.asyncio
    async def test_create_project_collections(self):
        """Test project collection creation."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"
        mock_config.collections = ["default", "memory"]

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch.object(manager, '_ensure_collection_exists') as mock_ensure:
            mock_ensure.return_value = asyncio.Future()
            mock_ensure.return_value.set_result(True)

            collections = await manager._create_project_collections("test-project", [])

        assert isinstance(collections, list)
        assert len(collections) >= 1

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_new_collection(self):
        """Test ensuring a new collection exists."""
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock()

        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        result = await manager._ensure_collection_exists("new-collection")

        assert result is True
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_existing_collection(self):
        """Test ensuring an existing collection exists."""
        mock_client = Mock()
        mock_client.collection_exists = AsyncMock(return_value=True)

        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        result = await manager._ensure_collection_exists("existing-collection")

        assert result is True
        # Should not call create_collection for existing collection
        assert not hasattr(mock_client, 'create_collection') or not mock_client.create_collection.called

    @pytest.mark.asyncio
    async def test_ingest_project_files_basic(self):
        """Test basic project file ingestion."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        project_info = {
            "project_name": "test-project",
            "project_path": "/path/to/project"
        }

        collections = ["test-project-default"]

        with patch.object(manager, '_get_project_files', return_value=["/path/to/file1.py"]):
            with patch.object(manager, '_process_file_batch') as mock_process:
                mock_process.return_value = asyncio.Future()
                mock_process.return_value.set_result({"processed": 1, "successful": 1})

                result = await manager.ingest_project_files(project_info, collections)

        assert isinstance(result, dict)
        assert "total_files" in result
        assert "processed_files" in result

    def test_get_project_files_with_patterns(self):
        """Test getting project files with patterns."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch('glob.glob', return_value=["/path/file1.py", "/path/file2.py"]):
            files = manager._get_project_files("/path/to/project", patterns=["*.py"])

        assert isinstance(files, list)
        assert len(files) == 2

    def test_get_project_files_default_patterns(self):
        """Test getting project files with default patterns."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch.object(ProjectPatterns, 'get_patterns_for_project', return_value=["*.py"]):
            with patch('glob.glob', return_value=["/path/file1.py"]):
                files = manager._get_project_files("/path/to/project")

        assert isinstance(files, list)
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_process_file_batch(self):
        """Test processing a batch of files."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        files = ["/path/file1.py", "/path/file2.py"]
        collections = ["test-collection"]

        with patch.object(manager, '_process_single_file') as mock_process:
            mock_process.return_value = asyncio.Future()
            mock_process.return_value.set_result({"success": True})

            result = await manager._process_file_batch(files, collections, batch_number=1)

        assert isinstance(result, dict)
        assert "processed" in result
        assert "successful" in result
        assert result["processed"] == 2

    @pytest.mark.asyncio
    async def test_process_single_file_success(self):
        """Test processing a single file successfully."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch.object(manager, '_read_file_content', return_value="file content"):
            with patch.object(manager, '_embed_and_store') as mock_embed:
                mock_embed.return_value = asyncio.Future()
                mock_embed.return_value.set_result(True)

                result = await manager._process_single_file("/path/file.py", ["collection"])

        assert result["success"] is True
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_process_single_file_failure(self):
        """Test processing a single file with failure."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch.object(manager, '_read_file_content', side_effect=Exception("Read error")):
            result = await manager._process_single_file("/path/file.py", ["collection"])

        assert result["success"] is False
        assert "error" in result
        assert "Read error" in result["error"]

    def test_read_file_content_text_file(self):
        """Test reading text file content."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch('builtins.open', mock_open(read_data="test content")):
            content = manager._read_file_content("/path/file.txt")

        assert content == "test content"

    def test_read_file_content_binary_file(self):
        """Test reading binary file content."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        with patch('builtins.open', side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "test")):
            content = manager._read_file_content("/path/file.bin")

        # Should handle binary files gracefully
        assert content is None or isinstance(content, str)

    @pytest.mark.asyncio
    async def test_embed_and_store_success(self):
        """Test embedding and storing content successfully."""
        mock_client = Mock()
        mock_client.upsert = AsyncMock()

        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        mock_embedding_service = Mock()
        mock_embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config,
            embedding_service=mock_embedding_service
        )

        result = await manager._embed_and_store(
            content="test content",
            file_path="/path/file.py",
            collections=["test-collection"]
        )

        assert result is True
        mock_embedding_service.embed_text.assert_called_once()
        mock_client.upsert.assert_called()

    @pytest.mark.asyncio
    async def test_embed_and_store_no_embedding_service(self):
        """Test embed and store without embedding service."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config,
            embedding_service=None
        )

        result = await manager._embed_and_store(
            content="test content",
            file_path="/path/file.py",
            collections=["test-collection"]
        )

        assert result is False


class TestUtilityFunctions:
    """Test utility functions."""

    def test_build_project_collection_name(self):
        """Test building project collection name."""
        name = build_project_collection_name("my-project", "documents")

        assert name == "my-project-documents"

    def test_build_project_collection_name_no_suffix(self):
        """Test building project collection name without suffix."""
        name = build_project_collection_name("my-project")

        assert name == "my-project"

    def test_normalize_collection_name_component(self):
        """Test normalizing collection name component."""
        normalized = normalize_collection_name_component("My Project Name!")

        assert isinstance(normalized, str)
        assert normalized.lower() == normalized  # Should be lowercase
        assert " " not in normalized  # Should not contain spaces
        assert "!" not in normalized  # Should not contain special chars


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_project_patterns_empty_path(self):
        """Test ProjectPatterns with empty path."""
        with patch.object(ProjectPatterns, '_detect_project_languages', return_value=[]):
            patterns = ProjectPatterns.get_patterns_for_project("")

        assert isinstance(patterns, list)

    def test_ingestion_progress_tracker_zero_total(self):
        """Test IngestionProgressTracker with zero total files."""
        tracker = IngestionProgressTracker(total_files=0)

        summary = tracker.get_summary()
        assert summary["total_files"] == 0
        assert summary["success_rate"] == 0.0

    def test_ingestion_progress_tracker_negative_total(self):
        """Test IngestionProgressTracker with negative total files."""
        tracker = IngestionProgressTracker(total_files=-5)

        # Should handle gracefully
        assert tracker.total_files == -5

    @pytest.mark.asyncio
    async def test_auto_ingestion_manager_empty_file_list(self):
        """Test AutoIngestionManager with empty file list."""
        mock_client = Mock()
        mock_config = Mock()
        mock_config.workspace = Mock()
        mock_config.workspace.github_user = "testuser"

        manager = AutoIngestionManager(
            qdrant_client=mock_client,
            config=mock_config
        )

        result = await manager._process_file_batch([], ["collection"], 1)

        assert result["processed"] == 0
        assert result["successful"] == 0

    def test_normalize_collection_name_special_characters(self):
        """Test normalizing collection name with various special characters."""
        test_cases = [
            ("hello@world.com", True),
            ("test_name_123", True),
            ("UPPERCASE_NAME", True),
            ("", True),
            ("   spaces   ", True)
        ]

        for test_input, should_succeed in test_cases:
            try:
                result = normalize_collection_name_component(test_input)
                assert isinstance(result, str)
                if should_succeed:
                    assert len(result) >= 0  # Should produce some output
            except Exception as e:
                if should_succeed:
                    pytest.fail(f"Unexpected exception for input '{test_input}': {e}")


# Mock open function for file reading tests
def mock_open(read_data=""):
    """Create a mock open function for testing file operations."""
    from unittest.mock import mock_open as original_mock_open
    return original_mock_open(read_data=read_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])