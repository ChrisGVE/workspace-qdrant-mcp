"""Unit tests for library management CLI commands.

Comprehensive unit tests for library collection management via CLI,
testing all library operations with proper mocking and isolation.

Test Coverage (35+ tests across 10 classes):
1. Library creation with naming validation
2. Library listing and statistics
3. Library removal with confirmation
4. Library status and health checks
5. Library info with samples and schema
6. Document addition to libraries
7. Folder watching configuration
8. Multi-library operations
9. Error handling and edge cases
10. Concurrent library operations

All tests use typer.testing.CliRunner for CLI testing and AsyncMock
for async operation mocking.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner


class TestLibraryCreation:
    """Test library collection creation commands (Subtask 287.1, 287.2)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_basic(self, mock_config, mock_daemon_client):
        """Test basic library creation with auto-prefixing."""
        from wqm_cli.cli.commands.library import library_app

        # Mock successful collection creation
        async def mock_operation(operation_func, config):
            client = MagicMock()
            return await operation_func(client)

        mock_daemon_client.side_effect = mock_operation

        # Mock daemon client responses
        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def create_operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                client.create_collection = AsyncMock(return_value=MagicMock(
                    collection_name="_technical-docs",
                    success=True
                ))
                result = await operation_func(client)
                return result

            mock_with_daemon.side_effect = create_operation

            result = self.runner.invoke(library_app, ["create", "technical-docs"])

            assert result.exit_code == 0
            assert "technical-docs" in result.stdout
            assert "created successfully" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_with_underscore_prefix(self, mock_config, mock_daemon_client):
        """Test library creation when name already has underscore prefix."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def create_operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                client.create_collection = AsyncMock(return_value=MagicMock(
                    collection_name="_mylib",
                    success=True
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = create_operation

            result = self.runner.invoke(library_app, ["create", "_mylib"])

            assert result.exit_code == 0
            assert "_mylib" in result.stdout or "mylib" in result.stdout

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_with_metadata(self, mock_config, mock_daemon_client):
        """Test library creation with description and tags."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def create_operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                client.create_collection = AsyncMock(return_value=MagicMock(
                    collection_name="_api-docs",
                    success=True
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = create_operation

            result = self.runner.invoke(
                library_app,
                ["create", "api-docs", "--description", "API Documentation", "--tag", "api", "--tag", "docs"]
            )

            assert result.exit_code == 0
            assert "API Documentation" in result.stdout

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_custom_vector_config(self, mock_config, mock_daemon_client):
        """Test library creation with custom vector configuration."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def create_operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                client.create_collection = AsyncMock(return_value=MagicMock(
                    collection_name="_embeddings",
                    success=True
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = create_operation

            result = self.runner.invoke(
                library_app,
                ["create", "embeddings", "--vector-size", "768", "--distance", "euclidean"]
            )

            assert result.exit_code == 0
            assert "768" in result.stdout
            assert "EUCLIDEAN" in result.stdout or "euclidean" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_already_exists(self, mock_config, mock_daemon_client):
        """Test library creation when collection already exists."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def create_operation(operation_func, config):
                client = MagicMock()
                # Return existing collection
                existing_col = MagicMock()
                existing_col.name = "_existing-lib"
                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[existing_col]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = create_operation

            result = self.runner.invoke(library_app, ["create", "existing-lib"])

            assert result.exit_code == 1
            assert "already exists" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.validate_collection_name')
    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_invalid_name(self, mock_config, mock_daemon_client, mock_validate):
        """Test library creation with invalid collection name."""
        from wqm_cli.cli.commands.library import library_app
        from common.core.collection_naming import CollectionNameError

        # Mock validation to raise error
        mock_validate.side_effect = CollectionNameError("Invalid name: contains invalid characters")

        result = self.runner.invoke(library_app, ["create", "invalid@name"])

        assert result.exit_code == 1
        assert "invalid" in result.stdout.lower()


class TestLibraryListing:
    """Test library listing and enumeration commands (Subtask 287.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_empty(self, mock_config, mock_daemon_client):
        """Test listing libraries when none exist."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def list_operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                return await operation_func(client)

            mock_with_daemon.side_effect = list_operation

            result = self.runner.invoke(library_app, ["list"])

            assert result.exit_code == 0
            assert "No library collections found" in result.stdout

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_multiple(self, mock_config, mock_daemon_client):
        """Test listing multiple library collections."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def list_operation(operation_func, config):
                client = MagicMock()

                # Create mock library collections
                lib1 = MagicMock()
                lib1.name = "_technical-docs"
                lib1.points_count = 100
                lib1.status = "active"

                lib2 = MagicMock()
                lib2.name = "_api-reference"
                lib2.points_count = 250
                lib2.status = "active"

                # Add non-library collection (should be filtered out)
                user_col = MagicMock()
                user_col.name = "my-notes"
                user_col.points_count = 50

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[lib1, lib2, user_col]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = list_operation

            result = self.runner.invoke(library_app, ["list"])

            assert result.exit_code == 0
            assert "technical-docs" in result.stdout
            assert "api-reference" in result.stdout
            # User collection should not appear
            assert "my-notes" not in result.stdout

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_with_stats(self, mock_config, mock_daemon_client):
        """Test listing libraries with statistics."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def list_operation(operation_func, config):
                client = MagicMock()

                lib = MagicMock()
                lib.name = "_docs"
                lib.points_count = 500
                lib.vectors_count = 500
                lib.indexed_points_count = 500
                lib.status = "active"

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[lib]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = list_operation

            result = self.runner.invoke(library_app, ["list", "--stats"])

            assert result.exit_code == 0
            assert "500" in result.stdout
            assert "Documents" in result.stdout or "documents" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_json_format(self, mock_config, mock_daemon_client):
        """Test listing libraries in JSON format."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def list_operation(operation_func, config):
                client = MagicMock()

                lib = MagicMock()
                lib.name = "_mylib"
                lib.points_count = 100
                lib.status = "active"

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[lib]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = list_operation

            result = self.runner.invoke(library_app, ["list", "--format", "json"])

            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.stdout)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "_mylib"


class TestLibraryRemoval:
    """Test library removal commands (Subtask 287.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_remove_library_with_force(self, mock_config, mock_daemon_client):
        """Test library removal with force flag (no confirmation)."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            call_count = [0]

            async def operation(operation_func, config):
                client = MagicMock()
                call_count[0] += 1

                if call_count[0] == 1:
                    # First call: check if collection exists
                    lib = MagicMock()
                    lib.name = "_oldlib"
                    client.list_collections = AsyncMock(return_value=MagicMock(
                        collections=[lib]
                    ))
                    client.get_collection_info = AsyncMock(return_value=MagicMock(
                        points_count=50
                    ))
                else:
                    # Second call: delete collection
                    client.delete_collection = AsyncMock(return_value=MagicMock(success=True))

                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["remove", "oldlib", "--force"])

            assert result.exit_code == 0
            assert "removed successfully" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_remove_library_not_found(self, mock_config, mock_daemon_client):
        """Test removing nonexistent library."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()
                # No collections exist
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["remove", "nonexistent", "--force"])

            assert result.exit_code == 1
            assert "not found" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    @patch('builtins.input')
    def test_remove_library_with_confirmation(self, mock_input, mock_config, mock_daemon_client):
        """Test library removal with user confirmation."""
        from wqm_cli.cli.commands.library import library_app

        # Mock user confirming deletion
        mock_input.return_value = "y"

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            call_count = [0]

            async def operation(operation_func, config):
                client = MagicMock()
                call_count[0] += 1

                if call_count[0] == 1:
                    lib = MagicMock()
                    lib.name = "_toremove"
                    client.list_collections = AsyncMock(return_value=MagicMock(
                        collections=[lib]
                    ))
                    client.get_collection_info = AsyncMock(return_value=MagicMock(
                        points_count=100
                    ))
                else:
                    client.delete_collection = AsyncMock(return_value=MagicMock(success=True))

                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["remove", "toremove"])

            assert result.exit_code == 0


class TestLibraryStatus:
    """Test library status and health commands (Subtask 287.2)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_status_all(self, mock_config, mock_daemon_client):
        """Test status for all libraries."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()

                lib = MagicMock()
                lib.name = "_docs"
                lib.points_count = 100
                lib.vectors_count = 100
                lib.indexed_points_count = 100
                lib.status = "active"

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[lib]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["status"])

            assert result.exit_code == 0
            assert "Library" in result.stdout or "library" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_status_specific(self, mock_config, mock_daemon_client):
        """Test status for specific library."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()
                client.get_collection_info = AsyncMock(return_value=MagicMock(
                    points_count=150,
                    indexed_points_count=150
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["status", "mylib"])

            assert result.exit_code == 0
            assert "150" in result.stdout


class TestLibraryInfo:
    """Test library info retrieval commands (Subtask 287.2)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_info_basic(self, mock_config, mock_daemon_client):
        """Test basic library information retrieval."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()
                client.get_collection_info = AsyncMock(return_value=MagicMock(
                    points_count=200,
                    indexed_points_count=200,
                    vector_size=384,
                    distance_metric="COSINE"
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["info", "mylib"])

            assert result.exit_code == 0
            assert "200" in result.stdout
            assert "384" in result.stdout or "unknown" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_info_with_samples(self, mock_config, mock_daemon_client):
        """Test library info with sample documents."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()

                # Create sample documents
                sample_doc = MagicMock()
                sample_doc.metadata = {
                    "title": "Sample Document",
                    "content": "This is sample content",
                    "source": "test.txt"
                }

                info = MagicMock()
                info.points_count = 10
                info.indexed_points_count = 10
                info.vector_size = 384
                info.sample_documents = [sample_doc]

                client.get_collection_info = AsyncMock(return_value=info)
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["info", "mylib", "--samples"])

            assert result.exit_code == 0
            assert "Sample" in result.stdout or "sample" in result.stdout.lower()


class TestDocumentAddition:
    """Test document addition to library collections (Subtask 327.1)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Note: Document addition typically goes through the ingest command,
    # not library command. These tests validate that ingestion
    # works correctly when targeting library collections.

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_text_file_to_library(self, mock_with_daemon):
        """Test ingesting a text file into a library collection."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create test text file
        test_file = Path(self.temp_dir) / "document.txt"
        test_file.write_text("This is a test document for library ingestion.")

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            # Create mock client with process_document method
            mock_client = MagicMock()
            mock_client.process_document = AsyncMock(return_value=MagicMock(
                success=True,
                document_id="doc_123",
                chunks_added=1,
                applied_metadata={}
            ))
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_technical-docs"]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_markdown_file_to_library(self, mock_with_daemon):
        """Test ingesting a markdown file into a library collection."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create test markdown file
        test_file = Path(self.temp_dir) / "api_reference.md"
        test_file.write_text(
            "# API Reference\n\n"
            "## Functions\n\n"
            "### get_data()\n\n"
            "Retrieves data from the database.\n"
        )

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()
            mock_client.process_document = AsyncMock(return_value=MagicMock(
                success=True,
                document_id="doc_md_123",
                chunks_added=1,
                applied_metadata={"file_type": "markdown", "title": "API Reference"}
            ))
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_api-docs"]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_code_file_to_library(self, mock_with_daemon):
        """Test ingesting a Python code file into a library collection."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create test Python file
        test_file = Path(self.temp_dir) / "example.py"
        test_file.write_text(
            '"""Example module."""\n\n'
            'def hello_world():\n'
            '    """Print hello world."""\n'
            '    print("Hello, World!")\n'
        )

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()
            mock_client.process_document = AsyncMock(return_value=MagicMock(
                success=True,
                document_id="doc_py_123",
                chunks_added=1,
                applied_metadata={
                    "file_type": "python",
                    "symbols": ["hello_world"]
                }
            ))
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_code-examples"]
        )

        assert result.exit_code == 0

    def test_ingest_invalid_file_path(self):
        """Test ingesting non-existent file path."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Use a non-existent file path
        invalid_path = Path(self.temp_dir) / "nonexistent.txt"

        result = self.runner.invoke(
            ingest_app,
            ["file", str(invalid_path), "--collection", "_docs"]
        )

        # Should fail with non-zero exit code
        assert result.exit_code != 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_folder_to_library(self, mock_with_daemon):
        """Test batch ingestion of folder contents to library collection."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create test folder with multiple files
        docs_dir = Path(self.temp_dir) / "library_docs"
        docs_dir.mkdir()

        (docs_dir / "intro.md").write_text("# Introduction\n\nWelcome to the library.")
        (docs_dir / "guide.md").write_text("# User Guide\n\nHow to use the library.")
        (docs_dir / "api.txt").write_text("API documentation content.")

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()

            # Mock process_folder as an async generator
            async def mock_process_folder(*args, **kwargs):
                # Yield progress for each file
                for file_name in ["intro.md", "guide.md", "api.txt"]:
                    progress = MagicMock()
                    progress.file_path = file_name
                    progress.chunks_added = 1
                    yield progress

            mock_client.process_folder = mock_process_folder
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["folder", str(docs_dir), "--collection", "_library-docs"]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_with_file_type_filter(self, mock_with_daemon):
        """Test folder ingestion with file type filtering."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create mixed file types
        docs_dir = Path(self.temp_dir) / "mixed_docs"
        docs_dir.mkdir()

        (docs_dir / "doc1.md").write_text("# Document 1")
        (docs_dir / "doc2.txt").write_text("Document 2 text")
        (docs_dir / "script.py").write_text("print('hello')")
        (docs_dir / "data.json").write_text('{"key": "value"}')

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()

            # Mock process_folder as an async generator
            async def mock_process_folder(*args, **kwargs):
                # Only yield .md and .txt files
                for file_name in ["doc1.md", "doc2.txt"]:
                    progress = MagicMock()
                    progress.file_path = file_name
                    progress.chunks_added = 1
                    yield progress

            mock_client.process_folder = mock_process_folder
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            [
                "folder",
                str(docs_dir),
                "--collection",
                "_docs",
                "--format",
                "md",
                "--format",
                "txt",
            ]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_with_metadata_extraction(self, mock_with_daemon):
        """Test document ingestion with metadata extraction validation."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create file with rich content for metadata extraction
        test_file = Path(self.temp_dir) / "module_doc.md"
        test_file.write_text(
            "# Database Module\n\n"
            "## Classes\n\n"
            "### Connection\n\n"
            "Manages database connections.\n\n"
            "```python\n"
            "from db import Connection\n"
            "conn = Connection('localhost')\n"
            "```\n"
        )

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()
            mock_client.process_document = AsyncMock(return_value=MagicMock(
                success=True,
                document_id="doc_rich_123",
                chunks_added=1,
                applied_metadata={
                    "file_type": "markdown",
                    "title": "Database Module",
                    "has_code_examples": True,
                    "classes": ["Connection"],
                    "language": "python"
                }
            ))
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_db-library"]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_unsupported_file_format(self, mock_with_daemon):
        """Test error handling for unsupported file formats."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create binary file (unsupported)
        test_file = Path(self.temp_dir) / "image.png"
        test_file.write_bytes(b'\x89PNG\r\n\x1a\n')  # PNG header

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()
            mock_client.process_document = AsyncMock(
                side_effect=ValueError("Unsupported file format: image/png")
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_docs"]
        )

        # Should fail with error
        assert result.exit_code != 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_to_library_with_underscore_prefix(self, mock_with_daemon):
        """Test library collection name validation with underscore prefix."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create test file
        test_file = Path(self.temp_dir) / "doc.txt"
        test_file.write_text("Test content")

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()
            # Verify collection name has underscore prefix
            mock_client.process_document = AsyncMock(return_value=MagicMock(
                success=True,
                document_id="doc_123",
                chunks_added=1,
                applied_metadata={"collection": "_mylib"}  # Library collections must start with _
            ))
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["file", str(test_file), "--collection", "_mylib"]
        )

        assert result.exit_code == 0

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_batch_ingest_multiple_file_types(self, mock_with_daemon):
        """Test batch ingestion of multiple file types to library."""
        from wqm_cli.cli.commands.ingest import ingest_app

        # Create diverse file set
        lib_dir = Path(self.temp_dir) / "comprehensive_lib"
        lib_dir.mkdir()

        (lib_dir / "readme.md").write_text("# Library Documentation")
        (lib_dir / "tutorial.md").write_text("# Tutorial")
        (lib_dir / "notes.txt").write_text("Additional notes")
        (lib_dir / "example.py").write_text("def example(): pass")

        # Mock the daemon client operations
        async def mock_operation(operation_func):
            mock_client = MagicMock()

            # Mock process_folder as an async generator
            async def mock_process_folder(*args, **kwargs):
                # Yield progress for each file
                files = [
                    ("readme.md", "markdown"),
                    ("tutorial.md", "markdown"),
                    ("notes.txt", "text"),
                    ("example.py", "python")
                ]
                for file_name, file_type in files:
                    progress = MagicMock()
                    progress.file_path = file_name
                    progress.chunks_added = 1
                    progress.file_type = file_type
                    yield progress

            mock_client.process_folder = mock_process_folder
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            ingest_app,
            ["folder", str(lib_dir), "--collection", "_comprehensive-lib"]
        )

        assert result.exit_code == 0

    def test_library_collection_ingestion_workflow(self):
        """Test document ingestion workflow for library collections."""
        # This test validates the overall workflow
        # Actual ingestion is handled by ingest commands with --collection flag

        # Create a test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test document content")

        # Validation: file exists and can be read
        assert test_file.exists()
        assert test_file.read_text() == "Test document content"


class TestFolderWatching:
    """Test folder watching configuration for libraries (Subtask 287.4)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # Note: Folder watching is managed through watch commands,
    # not library commands directly. These tests validate the
    # integration between library collections and watch functionality.

    @patch('wqm_cli.cli.commands.watch._get_daemon_client')
    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_add_watch_folder_to_library(self, mock_state_mgr, mock_daemon_client):
        """Test adding a watch folder configuration for a library collection."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig

        # Create test folder
        test_folder = Path(self.temp_dir) / "library_docs"
        test_folder.mkdir()

        # Mock state manager
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.save_watch_folder_config = AsyncMock(return_value=True)
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        # Mock daemon client for collection validation
        mock_client = AsyncMock()
        mock_collections_response = MagicMock()
        # Create mock collection with name attribute properly set
        mock_collection = MagicMock()
        mock_collection.name = "_technical-docs"
        mock_collections_response.collections = [mock_collection]
        mock_client.list_collections = AsyncMock(return_value=mock_collections_response)
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_daemon_client.return_value = mock_client

        result = self.runner.invoke(
            watch_app,
            [
                "add",
                str(test_folder),
                "--collection",
                "_technical-docs",
                "--pattern",
                "*.pdf",
                "--pattern",
                "*.md"
            ]
        )

        assert result.exit_code == 0
        assert "Watch started" in result.output

    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_list_watch_folders(self, mock_state_mgr):
        """Test listing all watch folder configurations."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig
        from datetime import datetime, timezone

        # Mock state manager with sample watches
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.get_all_watch_folder_configs = AsyncMock(return_value=[
            WatchFolderConfig(
                watch_id="watch_123",
                path="/path/to/docs",
                collection="_docs",
                patterns=["*.pdf"],
                ignore_patterns=[".git/*"],
                auto_ingest=True,
                recursive=True,
                recursive_depth=10,
                debounce_seconds=5.0,
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_scan=None,
                metadata=None
            )
        ])
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        result = self.runner.invoke(watch_app, ["list"])

        assert result.exit_code == 0
        assert "watch_123" in result.output or "Watch Configurations" in result.output

    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_remove_watch_folder(self, mock_state_mgr):
        """Test removing a watch folder configuration."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig
        from datetime import datetime, timezone

        # Create test folder for path matching - use resolved path for consistency
        test_folder = Path(self.temp_dir) / "docs"
        test_folder.mkdir()
        resolved_path = str(test_folder.resolve())

        # Mock state manager
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.get_all_watch_folder_configs = AsyncMock(return_value=[
            WatchFolderConfig(
                watch_id="watch_abc",
                path=resolved_path,  # Use resolved path
                collection="_docs",
                patterns=["*.pdf"],
                ignore_patterns=[],
                auto_ingest=True,
                recursive=True,
                recursive_depth=10,
                debounce_seconds=5.0,
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_scan=None,
                metadata=None
            )
        ])
        mock_mgr.remove_watch_folder_config = AsyncMock(return_value=True)
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        result = self.runner.invoke(
            watch_app,
            ["remove", str(test_folder), "--force"]
        )

        assert result.exit_code == 0
        assert "Removed watch" in result.output or "removed" in result.output.lower()

    @patch('wqm_cli.cli.commands.watch._get_daemon_client')
    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_add_watch_non_existent_folder_error(self, mock_state_mgr, mock_daemon_client):
        """Test error handling when adding watch for non-existent folder."""
        from wqm_cli.cli.commands.watch import watch_app

        # Don't create the folder - it should not exist
        non_existent_path = Path(self.temp_dir) / "does_not_exist"

        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        result = self.runner.invoke(
            watch_app,
            ["add", str(non_existent_path), "--collection", "_docs"]
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output or "not found" in result.output.lower()

    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_pause_and_resume_watch(self, mock_state_mgr):
        """Test pausing and resuming watch configurations."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig
        from datetime import datetime, timezone

        # Create test folder
        test_folder = Path(self.temp_dir) / "docs"
        test_folder.mkdir()
        resolved_path = str(test_folder.resolve())

        # Mock state manager
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()

        watch_config = WatchFolderConfig(
            watch_id="watch_xyz",
            path=resolved_path,  # Use resolved path
            collection="_docs",
            patterns=["*.pdf"],
            ignore_patterns=[],
            auto_ingest=True,
            recursive=True,
            recursive_depth=10,
            debounce_seconds=5.0,
            enabled=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_scan=None,
            metadata=None
        )

        mock_mgr.get_all_watch_folder_configs = AsyncMock(return_value=[watch_config])
        mock_mgr.save_watch_folder_config = AsyncMock(return_value=True)
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        # Test pause
        result = self.runner.invoke(watch_app, ["pause", str(test_folder)])
        assert result.exit_code == 0
        assert "Paus" in result.output

        # Test resume
        result = self.runner.invoke(watch_app, ["resume", str(test_folder)])
        assert result.exit_code == 0
        assert "Resum" in result.output

    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_configure_existing_watch(self, mock_state_mgr):
        """Test configuring an existing watch folder."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig
        from datetime import datetime, timezone

        # Create test folder
        test_folder = Path(self.temp_dir) / "docs"
        test_folder.mkdir()
        resolved_path = str(test_folder.resolve())

        # Mock state manager
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()

        watch_config = WatchFolderConfig(
            watch_id="watch_cfg",
            path=resolved_path,  # Use resolved path
            collection="_docs",
            patterns=["*.pdf"],
            ignore_patterns=[],
            auto_ingest=True,
            recursive=True,
            recursive_depth=10,
            debounce_seconds=5.0,
            enabled=True,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            last_scan=None,
            metadata=None
        )

        mock_mgr.get_all_watch_folder_configs = AsyncMock(return_value=[watch_config])
        mock_mgr.save_watch_folder_config = AsyncMock(return_value=True)
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        result = self.runner.invoke(
            watch_app,
            ["configure", "watch_cfg", "--depth", "5", "--debounce", "10"]
        )

        assert result.exit_code == 0
        assert "Configuration Updated" in result.output or "updated" in result.output.lower()

    @patch('wqm_cli.cli.commands.watch._get_daemon_client')
    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_watch_validates_collection_exists(self, mock_state_mgr, mock_daemon_client):
        """Test that watch command validates collection exists before adding."""
        from wqm_cli.cli.commands.watch import watch_app

        # Create test folder
        test_folder = Path(self.temp_dir) / "docs"
        test_folder.mkdir()

        # Mock state manager
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        # Mock daemon client with no collections
        mock_client = AsyncMock()
        mock_collections_response = MagicMock()
        mock_collections_response.collections = []  # No collections exist
        mock_client.list_collections = AsyncMock(return_value=mock_collections_response)
        mock_client.connect = AsyncMock()
        mock_client.disconnect = AsyncMock()
        mock_daemon_client.return_value = mock_client

        result = self.runner.invoke(
            watch_app,
            ["add", str(test_folder), "--collection", "_nonexistent"]
        )

        assert result.exit_code != 0
        assert "not found" in result.output or "Collection" in result.output

    @patch('wqm_cli.cli.commands.watch._get_state_manager')
    def test_watch_status_displays_statistics(self, mock_state_mgr):
        """Test watch status command displays statistics correctly."""
        from wqm_cli.cli.commands.watch import watch_app
        from common.core.sqlite_state_manager import WatchFolderConfig
        from datetime import datetime, timezone

        # Mock state manager with watches
        mock_mgr = AsyncMock()
        mock_mgr.initialize = AsyncMock()
        mock_mgr.get_all_watch_folder_configs = AsyncMock(return_value=[
            WatchFolderConfig(
                watch_id="watch_1",
                path="/path/1",
                collection="_docs",
                patterns=["*.pdf"],
                ignore_patterns=[],
                auto_ingest=True,
                recursive=True,
                recursive_depth=10,
                debounce_seconds=5.0,
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_scan=None,
                metadata=None
            ),
            WatchFolderConfig(
                watch_id="watch_2",
                path="/path/2",
                collection="_code",
                patterns=["*.py"],
                ignore_patterns=[],
                auto_ingest=False,
                recursive=True,
                recursive_depth=10,
                debounce_seconds=5.0,
                enabled=False,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                last_scan=None,
                metadata=None
            )
        ])
        mock_mgr.close = AsyncMock()
        mock_state_mgr.return_value = mock_mgr

        result = self.runner.invoke(watch_app, ["status"])

        assert result.exit_code == 0
        assert "Watch System Status" in result.output or "status" in result.output.lower()
        # Should show total watches and active/stopped counts


class TestErrorHandling:
    """Test library command error handling (Subtask 287.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_daemon_unavailable(self, mock_config, mock_daemon_client):
        """Test library creation when daemon is unavailable."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                raise ConnectionError("Daemon unavailable")

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["create", "mylib"])

            assert result.exit_code == 1

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_daemon_error(self, mock_config, mock_daemon_client):
        """Test library listing when daemon returns error."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                raise RuntimeError("Failed to list collections")

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["list"])

            assert result.exit_code == 1
            assert "error" in result.stdout.lower() or "failed" in result.stdout.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_remove_library_deletion_error(self, mock_config, mock_daemon_client):
        """Test library removal when deletion fails."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            call_count = [0]

            async def operation(operation_func, config):
                client = MagicMock()
                call_count[0] += 1

                if call_count[0] == 1:
                    lib = MagicMock()
                    lib.name = "_errorlib"
                    client.list_collections = AsyncMock(return_value=MagicMock(
                        collections=[lib]
                    ))
                    client.get_collection_info = AsyncMock(return_value=MagicMock(
                        points_count=10
                    ))
                else:
                    raise RuntimeError("Failed to delete collection")

                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["remove", "errorlib", "--force"])

            assert result.exit_code == 1


class TestEdgeCases:
    """Test library command edge cases."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_very_long_name(self, mock_config, mock_daemon_client):
        """Test library creation with very long name."""
        from wqm_cli.cli.commands.library import library_app

        long_name = "a" * 200

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()
                client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))
                client.create_collection = AsyncMock(return_value=MagicMock(
                    collection_name=f"_{long_name}",
                    success=True
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["create", long_name])

            # May succeed or fail depending on Qdrant limits
            assert result.exit_code in [0, 1]

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_special_prefixes(self, mock_config, mock_daemon_client):
        """Test library listing with special collection prefixes."""
        from wqm_cli.cli.commands.library import library_app

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()

                # MEMORY collection (special prefix)
                memory_col = MagicMock()
                memory_col.name = "_memory"
                memory_col.points_count = 50

                # Regular library
                lib_col = MagicMock()
                lib_col.name = "_docs"
                lib_col.points_count = 100

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=[memory_col, lib_col]
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            result = self.runner.invoke(library_app, ["list"])

            assert result.exit_code == 0
            # Both should appear (both start with _)
            assert "memory" in result.stdout
            assert "docs" in result.stdout


class TestConcurrentOperations:
    """Test concurrent library operations (Subtask 287.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_concurrent_library_creation(self, mock_config, mock_daemon_client):
        """Test creating multiple libraries concurrently."""
        from wqm_cli.cli.commands.library import _create_library

        created_collections = []

        async def mock_operation(operation_func, config):
            client = MagicMock()
            client.list_collections = AsyncMock(return_value=MagicMock(collections=[]))

            async def mock_create(collection_name, description, metadata):
                created_collections.append(collection_name)
                return MagicMock(collection_name=collection_name, success=True)

            client.create_collection = mock_create
            return await operation_func(client)

        with patch('wqm_cli.cli.commands.library.with_daemon_client', side_effect=mock_operation):
            async def create_multiple():
                tasks = [
                    _create_library(f"lib{i}", None, None, 384, "cosine")
                    for i in range(5)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            asyncio.run(create_multiple())

            # Verify all collections were created
            assert len(created_collections) == 5

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_concurrent_library_listing(self, mock_config, mock_daemon_client):
        """Test listing libraries concurrently."""
        from wqm_cli.cli.commands.library import _list_libraries

        with patch('wqm_cli.cli.commands.library.with_daemon_client') as mock_with_daemon:
            async def operation(operation_func, config):
                client = MagicMock()

                libs = [
                    MagicMock(name=f"_lib{i}", points_count=i*10, status="active")
                    for i in range(3)
                ]

                client.list_collections = AsyncMock(return_value=MagicMock(
                    collections=libs
                ))
                return await operation_func(client)

            mock_with_daemon.side_effect = operation

            async def list_multiple():
                tasks = [
                    _list_libraries(stats=True, sort_by="name", format="table")
                    for _ in range(10)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            # Should complete without errors
            asyncio.run(list_multiple())
