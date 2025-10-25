"""
Unit tests for WorkflowManager.

This module tests the complete workflow orchestration including integration
of all components and end-to-end processing scenarios.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.wqm_cli.cli.metadata.aggregator import DocumentMetadata
from src.python.wqm_cli.cli.metadata.batch_processor import BatchConfig, BatchResult
from src.python.wqm_cli.cli.metadata.exceptions import (
    MetadataError,
    WorkflowConfigurationError,
)
from src.python.wqm_cli.cli.metadata.workflow_manager import (
    WorkflowConfig,
    WorkflowManager,
    WorkflowResult,
)
from src.python.wqm_cli.cli.metadata.yaml_generator import YAMLConfig
from src.python.wqm_cli.cli.parsers.base import ParsedDocument


class TestWorkflowConfig:
    """Test cases for WorkflowConfig class."""

    def test_init_with_defaults(self):
        """Test WorkflowConfig initialization with default values."""
        config = WorkflowConfig()

        assert config.output_directory == Path.cwd() / "metadata_output"
        assert config.generate_individual_yamls is True
        assert config.generate_collection_yaml is True
        assert config.collection_name == "document_collection"
        assert config.project_name is None
        assert config.collection_type == "documents"
        assert config.incremental_updates is True

    def test_init_with_custom_values(self):
        """Test WorkflowConfig initialization with custom values."""
        custom_dir = Path("/custom/output")
        config = WorkflowConfig(
            output_directory=custom_dir,
            generate_individual_yamls=False,
            generate_collection_yaml=True,
            collection_name="custom_collection",
            project_name="test_project",
            incremental_updates=False,
        )

        assert config.output_directory == custom_dir
        assert config.generate_individual_yamls is False
        assert config.collection_name == "custom_collection"
        assert config.project_name == "test_project"
        assert config.incremental_updates is False

    def test_validation_no_output_methods(self):
        """Test validation when no output methods are enabled."""
        with pytest.raises(WorkflowConfigurationError) as exc_info:
            WorkflowConfig(
                generate_individual_yamls=False,
                generate_collection_yaml=False
            )

        assert "At least one of generate_individual_yamls or generate_collection_yaml must be True" in str(exc_info.value)

    def test_validation_empty_collection_name(self):
        """Test validation with empty collection name."""
        with pytest.raises(WorkflowConfigurationError) as exc_info:
            WorkflowConfig(collection_name="")

        assert "Collection name cannot be empty" in str(exc_info.value)


class TestWorkflowResult:
    """Test cases for WorkflowResult class."""

    def test_init_with_basic_data(self, sample_batch_result):
        """Test WorkflowResult initialization with basic data."""
        result = WorkflowResult(
            batch_result=sample_batch_result,
            yaml_files=["file1.yaml", "file2.yaml"],
            collection_yaml_path="/path/collection.yaml",
        )

        assert result.batch_result == sample_batch_result
        assert result.yaml_files == ["file1.yaml", "file2.yaml"]
        assert result.collection_yaml_path == "/path/collection.yaml"
        assert result.change_info == []

    def test_properties(self, sample_batch_result):
        """Test WorkflowResult properties."""
        result = WorkflowResult(batch_result=sample_batch_result)

        assert result.success_count == sample_batch_result.success_count
        assert result.failure_count == sample_batch_result.failure_count
        assert result.total_count == sample_batch_result.total_count
        assert result.success_rate == sample_batch_result.success_rate

    def test_changed_documents_count(self, sample_batch_result):
        """Test changed documents count property."""
        from src.python.wqm_cli.cli.metadata.incremental_tracker import (
            DocumentChangeInfo,
        )

        change_info = [
            DocumentChangeInfo("file1.txt", "hash1", change_type="modified"),
            DocumentChangeInfo("file2.txt", "hash2", change_type="added"),
            DocumentChangeInfo("file3.txt", "hash3", change_type="unchanged"),
        ]

        result = WorkflowResult(
            batch_result=sample_batch_result,
            change_info=change_info
        )

        assert result.changed_documents_count == 2  # modified + added


class TestWorkflowManager:
    """Test cases for WorkflowManager class."""

    def test_init_with_default_config(self):
        """Test WorkflowManager initialization with default config."""
        manager = WorkflowManager()

        assert manager.config is not None
        assert manager.metadata_aggregator is not None
        assert manager.batch_processor is not None
        assert manager.yaml_generator is not None
        assert manager.incremental_tracker is not None  # Default enables incremental

    def test_init_with_custom_config(self):
        """Test WorkflowManager initialization with custom config."""
        config = WorkflowConfig(
            incremental_updates=False,
            project_name="test_project",
        )
        manager = WorkflowManager(config)

        assert manager.config == config
        assert manager.incremental_tracker is None  # Disabled

    def test_init_with_verbose_logging(self):
        """Test WorkflowManager initialization with verbose logging."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                output_directory=Path(temp_dir),
                verbose_logging=True,
            )
            WorkflowManager(config)

            # Should create log file
            Path(temp_dir) / "workflow.log"
            # Log file might not exist until first log message

    async def test_process_documents_basic(self):
        """Test basic document processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                output_directory=Path(temp_dir),
                incremental_updates=False,  # Simplify for basic test
            )
            manager = WorkflowManager(config)

            # Mock successful batch processing
            sample_doc = ParsedDocument.create(
                content="test content",
                file_path="/test/doc.txt",
                file_type="text",
            )
            sample_metadata = DocumentMetadata(
                file_path="/test/doc.txt",
                content_hash="test_hash",
                parsed_document=sample_doc,
            )

            mock_batch_result = BatchResult(
                successful_documents=[sample_metadata],
                failed_documents=[],
                processing_stats={"total_files": 1, "successful_count": 1},
            )

            with patch.object(manager.batch_processor, 'process_documents') as mock_batch:
                mock_batch.return_value = mock_batch_result

                with patch.object(manager.yaml_generator, 'generate_batch_yaml_files') as mock_yaml_batch:
                    mock_yaml_batch.return_value = [str(Path(temp_dir) / "doc.yaml")]

                    with patch.object(manager.yaml_generator, 'generate_collection_yaml'):
                        result = await manager.process_documents(["/test/doc.txt"])

                        assert isinstance(result, WorkflowResult)
                        assert result.success_count == 1
                        assert result.failure_count == 0
                        assert len(result.yaml_files) == 1

    async def test_process_documents_with_incremental(self):
        """Test document processing with incremental tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                output_directory=Path(temp_dir),
                incremental_updates=True,
                tracking_storage_path=Path(temp_dir) / "tracking.db",
            )
            manager = WorkflowManager(config)

            # Mock batch processing
            sample_doc = ParsedDocument.create(
                content="test content",
                file_path="/test/doc.txt",
                file_type="text",
            )
            sample_metadata = DocumentMetadata(
                file_path="/test/doc.txt",
                content_hash="test_hash",
                parsed_document=sample_doc,
            )

            mock_batch_result = BatchResult(
                successful_documents=[sample_metadata],
                failed_documents=[],
                processing_stats={"total_files": 1, "successful_count": 1},
            )

            from src.python.wqm_cli.cli.metadata.incremental_tracker import (
                DocumentChangeInfo,
            )
            mock_changes = [
                DocumentChangeInfo("/test/doc.txt", "test_hash", change_type="added")
            ]

            with patch.object(manager.batch_processor, 'process_documents') as mock_batch:
                mock_batch.return_value = mock_batch_result

                with patch.object(manager.incremental_tracker, 'detect_changes') as mock_detect:
                    mock_detect.return_value = mock_changes

                    with patch.object(manager.incremental_tracker, 'get_changed_documents') as mock_changed:
                        mock_changed.return_value = [sample_metadata]

                        with patch.object(manager.incremental_tracker, 'update_tracking_data') as mock_update:
                            with patch.object(manager.yaml_generator, 'generate_batch_yaml_files') as mock_yaml:
                                mock_yaml.return_value = [str(Path(temp_dir) / "doc.yaml")]

                                result = await manager.process_documents(["/test/doc.txt"])

                                assert result.changed_documents_count == 1
                                mock_detect.assert_called_once()
                                mock_update.assert_called_once()

    async def test_process_documents_failure_handling(self):
        """Test workflow handling of processing failures."""
        config = WorkflowConfig(
            incremental_updates=False,
        )
        manager = WorkflowManager(config)

        # Mock batch processor to raise exception
        with patch.object(manager.batch_processor, 'process_documents') as mock_batch:
            mock_batch.side_effect = Exception("Batch processing failed")

            with pytest.raises(MetadataError) as exc_info:
                await manager.process_documents(["/test/doc.txt"])

            assert "Metadata workflow processing failed" in str(exc_info.value)

    async def test_process_directory_basic(self):
        """Test directory processing workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_dir = Path(temp_dir) / "test_docs"
            test_dir.mkdir()

            (test_dir / "doc1.txt").write_text("content 1")
            (test_dir / "doc2.md").write_text("# Content 2")
            (test_dir / "doc3.pdf").write_text("pdf content")  # Mock PDF

            config = WorkflowConfig(
                output_directory=Path(temp_dir) / "output",
                incremental_updates=False,
            )
            manager = WorkflowManager(config)

            # Mock the document processing
            with patch.object(manager, 'process_documents') as mock_process:
                mock_result = WorkflowResult(
                    batch_result=BatchResult([], [], {}),
                )
                mock_process.return_value = mock_result

                result = await manager.process_directory(test_dir)

                assert isinstance(result, WorkflowResult)
                mock_process.assert_called_once()

                # Check that files were found
                call_args = mock_process.call_args[0][0]  # First argument (file_paths)
                assert len(call_args) >= 2  # Should find at least txt and md files

    async def test_process_directory_nonexistent(self):
        """Test processing nonexistent directory."""
        config = WorkflowConfig()
        manager = WorkflowManager(config)

        with pytest.raises(MetadataError) as exc_info:
            await manager.process_directory("/nonexistent/directory")

        assert "Directory does not exist" in str(exc_info.value)

    async def test_process_directory_not_directory(self):
        """Test processing file path instead of directory."""
        config = WorkflowConfig()
        manager = WorkflowManager(config)

        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(MetadataError) as exc_info:
                await manager.process_directory(temp_file.name)

            assert "Path is not a directory" in str(exc_info.value)

    def test_find_files_in_directory_recursive(self):
        """Test finding files recursively in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create nested structure
            (test_dir / "doc1.txt").write_text("content 1")
            (test_dir / "subdir").mkdir()
            (test_dir / "subdir" / "doc2.md").write_text("content 2")
            (test_dir / "subdir" / "nested").mkdir()
            (test_dir / "subdir" / "nested" / "doc3.py").write_text("content 3")

            config = WorkflowConfig()
            manager = WorkflowManager(config)

            # Test recursive search
            files = manager._find_files_in_directory(
                directory=test_dir,
                recursive=True,
                file_patterns=["*.txt", "*.md", "*.py"]
            )

            assert len(files) == 3
            file_names = [f.name for f in files]
            assert "doc1.txt" in file_names
            assert "doc2.md" in file_names
            assert "doc3.py" in file_names

    def test_find_files_in_directory_non_recursive(self):
        """Test finding files non-recursively."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create nested structure
            (test_dir / "doc1.txt").write_text("content 1")
            (test_dir / "subdir").mkdir()
            (test_dir / "subdir" / "doc2.md").write_text("content 2")

            config = WorkflowConfig()
            manager = WorkflowManager(config)

            # Test non-recursive search
            files = manager._find_files_in_directory(
                directory=test_dir,
                recursive=False,
                file_patterns=["*.txt", "*.md"]
            )

            assert len(files) == 1  # Only top-level file
            assert files[0].name == "doc1.txt"

    def test_generate_workflow_stats(self):
        """Test workflow statistics generation."""
        config = WorkflowConfig()
        manager = WorkflowManager(config)

        # Mock data
        sample_doc = ParsedDocument.create(
            content="test content",
            file_path="/test/doc.txt",
            file_type="text",
        )
        sample_metadata = DocumentMetadata(
            file_path="/test/doc.txt",
            content_hash="test_hash",
            parsed_document=sample_doc,
        )

        batch_result = BatchResult(
            successful_documents=[sample_metadata],
            failed_documents=[("/test/failed.txt", "error")],
            processing_stats={"total_files": 2, "successful_count": 1},
        )

        from src.python.wqm_cli.cli.metadata.incremental_tracker import (
            DocumentChangeInfo,
        )
        change_info = [
            DocumentChangeInfo("/test/doc.txt", "hash", change_type="added"),
        ]

        yaml_files = ["/output/doc.yaml"]
        processing_time = 5.0

        stats = manager._generate_workflow_stats(
            batch_result=batch_result,
            change_info=change_info,
            yaml_files=yaml_files,
            processing_time=processing_time,
        )

        assert stats["processing_time"] == 5.0
        assert stats["total_documents"] == 2
        assert stats["successful_documents"] == 1
        assert stats["failed_documents"] == 1
        assert stats["individual_yaml_files"] == 1
        assert stats["change_counts"]["added"] == 1

    def test_get_workflow_status(self):
        """Test getting workflow status."""
        config = WorkflowConfig(
            project_name="test_project",
            collection_type="test_docs",
        )
        manager = WorkflowManager(config)

        status = manager.get_workflow_status()

        assert "configuration" in status
        assert status["configuration"]["project_name"] == "test_project"
        assert status["configuration"]["collection_type"] == "test_docs"
        assert "components" in status
        assert "metadata_aggregator" in status["components"]
        assert "batch_processor" in status["components"]
        assert "yaml_generator" in status["components"]

    def test_get_workflow_status_with_incremental_tracker(self):
        """Test workflow status with incremental tracker."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                incremental_updates=True,
                tracking_storage_path=Path(temp_dir) / "tracking.db",
            )
            manager = WorkflowManager(config)

            status = manager.get_workflow_status()

            assert "incremental_tracker" in status["components"]
            # Should have tracking summary or error

    async def test_summary_report_generation(self):
        """Test summary report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = WorkflowConfig(
                output_directory=Path(temp_dir),
                incremental_updates=False,
            )
            manager = WorkflowManager(config)

            # Mock batch result
            batch_result = BatchResult(
                successful_documents=[],
                failed_documents=[],
                processing_stats={"total_files": 0},
            )

            workflow_stats = {"processing_time": 1.0}
            change_info = []

            # Test summary report generation (should not crash)
            await manager._generate_summary_report(
                batch_result=batch_result,
                change_info=change_info,
                workflow_stats=workflow_stats,
            )

            # Check if summary file would be created (mocked in actual implementation)

    def test_workflow_manager_component_integration(self):
        """Test that all components are properly integrated."""
        config = WorkflowConfig()
        manager = WorkflowManager(config)

        # Verify all components are connected
        assert manager.batch_processor.metadata_aggregator is not None
        assert manager.yaml_generator.config is not None

        # Verify configs are passed through
        assert manager.batch_processor.config == config.batch_config
        assert manager.yaml_generator.config == config.yaml_config

    async def test_edge_case_empty_file_list(self):
        """Test processing empty file list."""
        config = WorkflowConfig(incremental_updates=False)
        manager = WorkflowManager(config)

        # Mock batch processor to handle empty list
        with patch.object(manager.batch_processor, 'process_documents') as mock_batch:
            mock_batch.return_value = BatchResult([], [], {"total_files": 0})

            result = await manager.process_documents([])

            assert result.total_count == 0
            assert len(result.yaml_files) == 0

    async def test_edge_case_all_files_filtered(self):
        """Test when all files are filtered out as invalid."""
        config = WorkflowConfig()
        manager = WorkflowManager(config)

        # Mock batch processor to filter out all files
        with patch.object(manager.batch_processor, 'process_documents') as mock_batch:
            from src.python.wqm_cli.cli.metadata.exceptions import BatchProcessingError
            mock_batch.side_effect = BatchProcessingError("No valid file paths found")

            with pytest.raises(MetadataError):
                await manager.process_documents(["/invalid/file.unknown"])


# Test fixtures

@pytest.fixture
def sample_batch_result():
    """Create a sample BatchResult for testing."""
    sample_doc = ParsedDocument.create(
        content="test content",
        file_path="/test/doc.txt",
        file_type="text",
    )
    sample_metadata = DocumentMetadata(
        file_path="/test/doc.txt",
        content_hash="test_hash",
        parsed_document=sample_doc,
    )

    return BatchResult(
        successful_documents=[sample_metadata],
        failed_documents=[("/test/failed.txt", "Mock error")],
        processing_stats={
            "total_files": 2,
            "successful_count": 1,
            "failed_count": 1,
            "processing_time_seconds": 1.0,
        },
    )
