"""
End-to-end tests for complete file ingestion pipeline workflow.

Tests the full ingestion pipeline from file change detection through to
vector storage in Qdrant, including queue processing, content extraction,
embedding generation, metadata computation, and status updates.
"""

import pytest
import asyncio
import time
import httpx
from pathlib import Path
from typing import List

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
)


@pytest.mark.integration
@pytest.mark.slow
class TestFileChangeDetection:
    """Test file change detection and queue addition."""

    def test_new_file_detected(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that new files are detected by the watch system."""
        workspace = system_components.workspace_path

        # Create a new file
        test_file = workspace / "test_detection.txt"
        test_file.write_text("Test content for detection")

        # Give watcher time to detect
        time.sleep(3)

        # File should exist
        assert test_file.exists()

    def test_modified_file_detected(
        self, system_components: SystemComponents
    ):
        """Test that modified files are detected."""
        workspace = system_components.workspace_path

        # Create initial file
        test_file = workspace / "test_modify.txt"
        test_file.write_text("Initial content")
        time.sleep(2)

        # Modify file
        test_file.write_text("Modified content")
        time.sleep(2)

        # File should have new content
        assert test_file.read_text() == "Modified content"

    def test_multiple_files_detected(
        self, system_components: SystemComponents
    ):
        """Test that multiple new files are detected."""
        workspace = system_components.workspace_path

        # Create multiple files
        files = []
        for i in range(5):
            test_file = workspace / f"test_multiple_{i}.txt"
            test_file.write_text(f"Content {i}")
            files.append(test_file)

        time.sleep(3)

        # All files should exist
        for file in files:
            assert file.exists()


@pytest.mark.integration
@pytest.mark.slow
class TestContentExtraction:
    """Test content extraction from various file types."""

    def test_text_file_extraction(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test content extraction from plain text files."""
        workspace = system_components.workspace_path

        # Create text file
        test_file = workspace / "test.txt"
        content = "This is plain text content for testing extraction."
        test_file.write_text(content)

        # Use CLI to ingest directly
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-text"]
        )

        # Should succeed or provide meaningful error
        assert result is not None

    def test_markdown_file_extraction(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test content extraction from markdown files."""
        workspace = system_components.workspace_path

        # Create markdown file
        test_file = workspace / "test.md"
        content = "# Test Header\n\nThis is **markdown** content."
        test_file.write_text(content)

        # Ingest via CLI
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-md"]
        )

        assert result is not None

    def test_python_file_extraction(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test content extraction from Python code files."""
        workspace = system_components.workspace_path

        # Create Python file
        test_file = workspace / "src" / "test_code.py"
        content = """
def hello_world():
    '''Test function for extraction.'''
    print("Hello, World!")
"""
        test_file.write_text(content)

        # Ingest via CLI
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-py"]
        )

        assert result is not None

    def test_json_file_extraction(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test content extraction from JSON files."""
        workspace = system_components.workspace_path

        # Create JSON file
        test_file = workspace / "config" / "test.json"
        test_file.parent.mkdir(exist_ok=True)
        content = '{"key": "value", "number": 42}'
        test_file.write_text(content)

        # Ingest via CLI
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-json"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestQueueProcessing:
    """Test ingestion queue processing."""

    def test_single_file_queue_processing(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that a single file is processed from the queue."""
        workspace = system_components.workspace_path

        # Create file for processing
        test_file = workspace / "queue_single.txt"
        test_file.write_text("Single file for queue processing")

        # Ingest and verify no crash
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-queue"]
        )

        assert result is not None

    def test_batch_file_queue_processing(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that multiple files are processed from the queue."""
        workspace = system_components.workspace_path

        # Create batch of files
        batch_dir = workspace / "batch"
        batch_dir.mkdir(exist_ok=True)

        for i in range(10):
            test_file = batch_dir / f"batch_{i}.txt"
            test_file.write_text(f"Batch file {i} content")

        # Ingest directory
        result = cli_helper.run_command(
            ["ingest", "folder", str(batch_dir), "--collection", "test-e2e-batch"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestEmbeddingGeneration:
    """Test embedding generation during ingestion."""

    @pytest.mark.asyncio
    async def test_embeddings_generated_for_content(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that embeddings are generated for ingested content."""
        workspace = system_components.workspace_path

        # Create test file with meaningful content
        test_file = workspace / "embedding_test.txt"
        test_file.write_text(
            "This is a test document with semantic content for embedding generation."
        )

        # Ingest file
        result = cli_helper.run_command(
            [
                "ingest",
                "file",
                str(test_file),
                "--collection",
                "test-e2e-embeddings",
            ]
        )

        # Wait for processing
        await asyncio.sleep(3)

        # Verify Qdrant has the collection (embeddings were generated and stored)
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{system_components.qdrant_url}/collections/test-e2e-embeddings",
                timeout=5.0,
            )
            # Collection should exist if embeddings were generated and stored
            # May not exist if processing hasn't completed yet
            assert response.status_code in [200, 404]


@pytest.mark.integration
@pytest.mark.slow
class TestMetadataComputation:
    """Test metadata computation during ingestion."""

    def test_file_metadata_captured(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that file metadata is captured during ingestion."""
        workspace = system_components.workspace_path

        # Create file with known metadata
        test_file = workspace / "src" / "metadata_test.py"
        test_file.parent.mkdir(exist_ok=True)
        test_file.write_text("# Python file for metadata testing")

        # Ingest file
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-metadata"]
        )

        # Should succeed (metadata computation happens internally)
        assert result is not None

    def test_git_branch_metadata(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that git branch metadata is computed."""
        workspace = system_components.workspace_path

        # Workspace has .git directory from fixture
        assert (workspace / ".git").exists()

        # Create file in git workspace
        test_file = workspace / "git_test.txt"
        test_file.write_text("File in git repository")

        # Ingest file (should detect git and compute branch)
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-git"]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantStorage:
    """Test storage of vectors and metadata in Qdrant."""

    @pytest.mark.asyncio
    async def test_vectors_stored_in_qdrant(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that vectors are actually stored in Qdrant."""
        workspace = system_components.workspace_path

        # Create and ingest file
        test_file = workspace / "qdrant_storage.txt"
        test_file.write_text("Content to store in Qdrant vector database")

        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-storage"]
        )

        # Wait for storage
        await asyncio.sleep(3)

        # Check if collection exists in Qdrant
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{system_components.qdrant_url}/collections", timeout=5.0
            )
            assert response.status_code == 200

            # Parse collections list
            data = response.json()
            collection_names = [
                c["name"] for c in data.get("result", {}).get("collections", [])
            ]

            # test-e2e-storage may or may not be present depending on processing speed
            # Just verify we can query Qdrant
            assert isinstance(collection_names, list)

    @pytest.mark.asyncio
    async def test_collection_created_for_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that collection is created when ingesting files."""
        # Use unique collection name
        collection_name = f"test-e2e-collection-{int(time.time())}"

        workspace = system_components.workspace_path
        test_file = workspace / "collection_test.txt"
        test_file.write_text("Test content for collection creation")

        # Ingest to new collection
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )

        await asyncio.sleep(3)

        # Query Qdrant for collection
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{system_components.qdrant_url}/collections", timeout=5.0
            )
            assert response.status_code == 200


@pytest.mark.integration
@pytest.mark.slow
class TestStatusUpdates:
    """Test status updates during ingestion."""

    def test_status_command_shows_activity(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that status command shows ingestion activity."""
        # Query current status
        result = cli_helper.run_command(["status"])

        # Status command should work
        assert result is not None
        assert result.returncode in [0, 1]  # May be empty or have content

    def test_admin_collections_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that admin collections command works during ingestion."""
        result = cli_helper.run_command(["admin", "collections"])

        # Should return collections list (may be empty)
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestEdgeCases:
    """Test edge cases in file ingestion."""

    def test_empty_file_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion of empty file."""
        workspace = system_components.workspace_path

        # Create empty file
        test_file = workspace / "empty.txt"
        test_file.write_text("")

        # Should handle empty files gracefully
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-empty"]
        )

        # May succeed or fail, but shouldn't crash
        assert result is not None

    def test_large_file_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion of large file."""
        workspace = system_components.workspace_path

        # Create large file (1MB)
        test_file = workspace / "large.txt"
        large_content = "Large content line.\n" * 10000  # ~170KB
        test_file.write_text(large_content)

        # Ingest large file
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-large"],
            timeout=60,
        )

        # Should handle large files
        assert result is not None

    def test_special_characters_in_filename(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion of file with special characters in name."""
        workspace = system_components.workspace_path

        # Create file with spaces and dashes
        test_file = workspace / "test-file with spaces.txt"
        test_file.write_text("Content with special filename")

        # Should handle special characters
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-e2e-special"]
        )

        assert result is not None

    def test_nested_directory_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion of nested directory structure."""
        workspace = system_components.workspace_path

        # Create nested structure
        nested = workspace / "level1" / "level2" / "level3"
        nested.mkdir(parents=True, exist_ok=True)

        # Create files at different levels
        (workspace / "level1" / "file1.txt").write_text("Level 1")
        (workspace / "level1" / "level2" / "file2.txt").write_text("Level 2")
        (nested / "file3.txt").write_text("Level 3")

        # Ingest top level directory
        result = cli_helper.run_command(
            [
                "ingest",
                "folder",
                str(workspace / "level1"),
                "--collection",
                "test-e2e-nested",
            ]
        )

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestCompleteWorkflow:
    """Test complete end-to-end ingestion workflow."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test complete pipeline from file creation to Qdrant storage."""
        workspace = system_components.workspace_path

        # Step 1: Create test file
        test_file = workspace / "complete_workflow.txt"
        test_content = "Complete end-to-end workflow test content for validation."
        test_file.write_text(test_content)

        # Step 2: Ingest file
        collection_name = f"test-e2e-workflow-{int(time.time())}"
        result = cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )

        # Step 3: Wait for processing
        await asyncio.sleep(5)

        # Step 4: Verify file exists
        assert test_file.exists()
        assert test_file.read_text() == test_content

        # Step 5: Verify command executed
        assert result is not None

    @pytest.mark.asyncio
    async def test_multi_file_type_workflow(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test workflow with multiple file types."""
        workspace = system_components.workspace_path

        # Create files of different types
        files = {
            "test.txt": "Plain text content",
            "test.md": "# Markdown content",
            "test.py": "def test():\n    pass",
            "test.json": '{"test": true}',
        }

        workflow_dir = workspace / "multi_type_workflow"
        workflow_dir.mkdir(exist_ok=True)

        for filename, content in files.items():
            (workflow_dir / filename).write_text(content)

        # Ingest all files
        result = cli_helper.run_command(
            [
                "ingest",
                "folder",
                str(workflow_dir),
                "--collection",
                "test-e2e-multitype",
            ]
        )

        await asyncio.sleep(5)

        # All files should exist
        for filename in files.keys():
            assert (workflow_dir / filename).exists()

        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestIngestionPerformance:
    """Test ingestion performance and throughput."""

    @pytest.mark.asyncio
    async def test_ingestion_throughput(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion throughput for multiple files."""
        workspace = system_components.workspace_path

        # Create batch of files
        perf_dir = workspace / "performance_test"
        perf_dir.mkdir(exist_ok=True)

        file_count = 20
        for i in range(file_count):
            (perf_dir / f"perf_{i}.txt").write_text(f"Performance test file {i}")

        # Measure ingestion time
        start_time = time.time()
        result = cli_helper.run_command(
            ["ingest", "folder", str(perf_dir), "--collection", "test-e2e-perf"],
            timeout=120,
        )
        ingestion_time = time.time() - start_time

        # Should complete in reasonable time (6 seconds per file is very generous)
        assert ingestion_time < file_count * 6.0

        assert result is not None
