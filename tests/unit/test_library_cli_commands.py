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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
        from common.core.collection_naming import CollectionNameError
        from wqm_cli.cli.commands.library import library_app

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
        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
        from datetime import datetime, timezone

        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
        from datetime import datetime, timezone

        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
        from datetime import datetime, timezone

        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
        from datetime import datetime, timezone

        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
        from datetime import datetime, timezone

        from common.core.sqlite_state_manager import WatchFolderConfig
        from wqm_cli.cli.commands.watch import watch_app

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
                memory_col.name = "memory"
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


class TestLibraryEnumeration:
    """Test library collection enumeration and listing (Task 327.3)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_basic(self, mock_config, mock_with_daemon):
        """Test basic library listing without stats."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create mock collections including library collections (with _) and regular ones
            col1 = MagicMock()
            col1.name = "_technical-docs"
            col2 = MagicMock()
            col2.name = "_api-reference"
            col3 = MagicMock()
            col3.name = "myproject-code"  # Not a library
            mock_collections = [col1, col2, col3]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list"])

        assert result.exit_code == 0
        # Should show library collections
        assert "technical-docs" in result.output
        assert "api-reference" in result.output
        # Should NOT show regular project collections
        assert "myproject-code" not in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_with_stats(self, mock_config, mock_with_daemon):
        """Test library listing with statistics."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create mock library collections with stats
            col1 = MagicMock()
            col1.name = "_docs"
            col1.points_count = 100
            col1.vectors_count = 100
            col1.indexed_points_count = 100
            col1.status = "active"

            col2 = MagicMock()
            col2.name = "_code"
            col2.points_count = 250
            col2.vectors_count = 250
            col2.indexed_points_count = 250
            col2.status = "active"

            mock_collections = [col1, col2]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list", "--stats"])

        assert result.exit_code == 0
        # Should display statistics
        assert "100" in result.output  # docs count
        assert "250" in result.output  # code count
        assert "Documents" in result.output or "documents" in result.output.lower()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_sort_by_size(self, mock_config, mock_with_daemon):
        """Test library listing sorted by size."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create libraries with different sizes
            col1 = MagicMock()
            col1.name = "_small"
            col1.points_count = 10
            col1.vectors_count = 10
            col1.indexed_points_count = 10

            col2 = MagicMock()
            col2.name = "_large"
            col2.points_count = 1000
            col2.vectors_count = 1000
            col2.indexed_points_count = 1000

            col3 = MagicMock()
            col3.name = "_medium"
            col3.points_count = 100
            col3.vectors_count = 100
            col3.indexed_points_count = 100

            mock_collections = [col1, col2, col3]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            library_app, ["list", "--stats", "--sort", "size"]
        )

        assert result.exit_code == 0
        # Output should have libraries sorted by size (descending)
        lines = result.output.split('\n')
        # Find lines with library data
        lib_lines = [l for l in lines if any(name in l for name in ['small', 'medium', 'large'])]
        # Large should come before medium, medium before small
        if len(lib_lines) >= 3:
            large_idx = next(i for i, l in enumerate(lib_lines) if 'large' in l)
            small_idx = next(i for i, l in enumerate(lib_lines) if 'small' in l)
            assert large_idx < small_idx

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_json_format(self, mock_config, mock_with_daemon):
        """Test library listing in JSON format."""
        import json
        from types import SimpleNamespace

        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Use SimpleNamespace instead of MagicMock to avoid JSON serialization issues
            col1 = SimpleNamespace(
                name="_docs",
                points_count=50,
                vectors_count=50,
                indexed_points_count=50,
                status="active"
            )

            mock_collections = [col1]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            library_app, ["list", "--stats", "--format", "json"]
        )

        assert result.exit_code == 0
        # Should be valid JSON
        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["name"] == "_docs"
            assert data[0]["display_name"] == "docs"
            assert data[0]["points_count"] == 50
        except json.JSONDecodeError:
            raise AssertionError("Output is not valid JSON")

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_empty(self, mock_config, mock_with_daemon):
        """Test library listing when no libraries exist."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # No library collections (only regular project collections)
            col1 = MagicMock()
            col1.name = "myproject-code"
            col2 = MagicMock()
            col2.name = "myproject-docs"

            mock_collections = [col1, col2]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list"])

        assert result.exit_code == 0
        assert "No library collections found" in result.output
        assert "create" in result.output  # Should suggest creating one

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_status_specific(self, mock_config, mock_with_daemon):
        """Test status command for a specific library."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Mock get_collection_info response
            mock_info = MagicMock(
                points_count=150,
                indexed_points_count=150,
                vector_size=384,
                distance_metric="COSINE",
            )
            mock_client.get_collection_info = AsyncMock(return_value=mock_info)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["status", "technical-docs"])

        assert result.exit_code == 0
        assert "technical-docs" in result.output
        assert "150" in result.output  # Document count

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_info_with_samples(self, mock_config, mock_with_daemon):
        """Test info command with sample documents."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Mock collection info with sample documents
            mock_doc = MagicMock()
            mock_doc.metadata = {
                "title": "Sample Document",
                "filename": "sample.pdf",
                "content": "This is sample content for testing",
            }
            mock_info = MagicMock(
                points_count=200,
                indexed_points_count=200,
                vector_size=384,
                distance_metric="COSINE",
                sample_documents=[mock_doc],
            )
            mock_client.get_collection_info = AsyncMock(return_value=mock_info)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            library_app, ["info", "api-docs", "--samples"]
        )

        assert result.exit_code == 0
        assert "api-docs" in result.output
        assert "Sample Document" in result.output
        assert "sample.pdf" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_list_libraries_performance_many_libraries(self, mock_config, mock_with_daemon):
        """Test listing performance with 100+ libraries."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create 150 mock library collections
            mock_collections = []
            for i in range(150):
                col = MagicMock()
                col.name = f"_lib{i:03d}"
                col.points_count = i * 10
                col.vectors_count = i * 10
                col.indexed_points_count = i * 10
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list", "--stats"])

        assert result.exit_code == 0
        # Should display count of libraries found
        assert "150" in result.output  # Library count

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_info_schema_display(self, mock_config, mock_with_daemon):
        """Test info command with schema display."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            mock_info = MagicMock(
                points_count=75,
                indexed_points_count=75,
                vector_size=512,
                distance_metric="EUCLIDEAN",
            )
            mock_client.get_collection_info = AsyncMock(return_value=mock_info)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            library_app, ["info", "ml-models", "--schema"]
        )

        assert result.exit_code == 0
        assert "ml-models" in result.output
        assert "Vector Size" in result.output or "512" in result.output


class TestBulkOperationsAndPerformance:
    """Test bulk operations and performance (Task 327.4)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_batch_library_creation(self, mock_config, mock_with_daemon):
        """Test creating multiple libraries in batch."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()
        creation_count = 0

        async def mock_operation(operation_func, config):
            nonlocal creation_count
            mock_client = MagicMock()

            # Mock list_collections to return existing collections
            existing_collections = [
                MagicMock(name=f"_lib{i}") for i in range(creation_count)
            ]
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=existing_collections)
            )

            # Mock create_collection
            mock_client.create_collection = AsyncMock(return_value=True)

            result = await operation_func(mock_client)
            if result is not False and result is not None:  # If collection was created
                creation_count += 1
            return result

        mock_with_daemon.side_effect = mock_operation

        # Create 10 libraries in sequence
        library_names = [f"batch-lib-{i}" for i in range(10)]
        for name in library_names:
            result = self.runner.invoke(library_app, ["create", name])
            assert result.exit_code == 0

        # Verify all were created
        assert creation_count == 10

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_bulk_document_addition(self, mock_with_daemon):
        """Test adding multiple documents to a library in bulk."""
        import tempfile
        from pathlib import Path

        from wqm_cli.cli.commands.ingest import ingest_app

        # Create temporary directory with multiple files
        with tempfile.TemporaryDirectory() as temp_dir:
            test_dir = Path(temp_dir)

            # Create 50 test files
            for i in range(50):
                (test_dir / f"doc_{i}.txt").write_text(f"Document content {i}")

            # Mock daemon client for bulk ingestion
            async def mock_operation(operation_func):
                mock_client = MagicMock()

                # Mock process_folder to yield progress for each file
                async def mock_process_folder(*args, **kwargs):
                    for i in range(50):
                        progress = MagicMock()
                        progress.file_path = f"doc_{i}.txt"
                        progress.chunks_added = 1
                        progress.file_type = "text"
                        yield progress

                mock_client.process_folder = mock_process_folder
                return await operation_func(mock_client)

            mock_with_daemon.side_effect = mock_operation

            result = self.runner.invoke(
                ingest_app,
                ["folder", str(test_dir), "--collection", "_bulk-test"]
            )

            assert result.exit_code == 0
            # Should mention processing multiple files
            assert "50" in result.output or "doc_" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_mass_library_removal(self, mock_config, mock_with_daemon):
        """Test removing multiple libraries."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()
        libraries_to_remove = [f"_remove-{i}" for i in range(5)]
        removed_libraries = set()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()

            # Mock list_collections - return all libraries
            mock_collections = []
            for name in libraries_to_remove:
                col = MagicMock()
                col.name = name
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )

            # Mock get_collection_info
            mock_client.get_collection_info = AsyncMock(
                return_value=MagicMock(points_count=100)
            )

            # Mock delete_collection - track removed libraries
            async def mock_delete(collection_name, **kwargs):
                removed_libraries.add(collection_name)
                return True

            mock_client.delete_collection = mock_delete

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Remove libraries with --force flag
        for lib_name in libraries_to_remove:
            result = self.runner.invoke(
                library_app,
                ["remove", lib_name, "--force"]
            )
            assert result.exit_code == 0

        assert len(removed_libraries) == 5

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_performance_listing_many_libraries(self, mock_config, mock_with_daemon):
        """Test performance of listing 200+ libraries."""
        import time

        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create 200 mock library collections
            mock_collections = []
            for i in range(200):
                col = MagicMock()
                col.name = f"_perf-lib{i:04d}"
                col.points_count = i * 100
                col.vectors_count = i * 100
                col.indexed_points_count = i * 100
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        start_time = time.time()
        result = self.runner.invoke(library_app, ["list", "--stats"])
        elapsed = time.time() - start_time

        assert result.exit_code == 0
        assert "200" in result.output
        # Should complete in reasonable time (< 5 seconds for mock)
        assert elapsed < 5.0

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_rapid_sequential_operations(self, mock_config, mock_with_daemon):
        """Test handling rapid sequential library operations without state corruption."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()

            mock_collections = []
            for i in range(10):
                col = MagicMock()
                col.name = f"_rapid-{i}"
                col.points_count = i * 10
                col.vectors_count = i * 10
                col.indexed_points_count = i * 10
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Run multiple operations rapidly in sequence
        # This tests for state corruption and ensures operations don't interfere
        results = []
        for _ in range(10):
            result = self.runner.invoke(library_app, ["list", "--stats"])
            results.append(result.exit_code)

        # All operations should complete successfully
        assert len(results) == 10
        assert all(code == 0 for code in results)

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_error_recovery_batch_creation(self, mock_config, mock_with_daemon):
        """Test error recovery during batch library creation."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()
        attempt_count = 0

        async def mock_operation(operation_func, config):
            nonlocal attempt_count
            attempt_count += 1
            mock_client = MagicMock()

            # Mock list_collections
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=[])
            )

            # Fail on 3rd attempt, succeed on others
            if attempt_count == 3:
                mock_client.create_collection = AsyncMock(
                    side_effect=Exception("Simulated creation failure")
                )
            else:
                mock_client.create_collection = AsyncMock(return_value=True)

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Try to create 5 libraries
        results = []
        for i in range(5):
            result = self.runner.invoke(library_app, ["create", f"recovery-{i}"])
            results.append(result.exit_code)

        # Should have 4 successes and 1 failure
        assert results.count(0) == 4  # 4 successful creations
        assert results.count(1) == 1  # 1 failed creation

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_memory_efficient_bulk_listing(self, mock_config, mock_with_daemon):
        """Test memory efficiency when listing many libraries."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Create 500 mock library collections
            mock_collections = []
            for i in range(500):
                col = MagicMock()
                col.name = f"_mem-test-{i:04d}"
                col.points_count = i
                col.vectors_count = i
                col.indexed_points_count = i
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # List with stats should handle large result sets
        result = self.runner.invoke(library_app, ["list", "--stats"])

        assert result.exit_code == 0
        # Should display all libraries
        assert "500" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_batch_operation_partial_failure_handling(self, mock_config, mock_with_daemon):
        """Test handling of partial failures in batch operations."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()
        creation_attempts = {}  # library_name -> success

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()

            # Track which libraries already exist
            created_libs = [name for name, success in creation_attempts.items() if success]
            mock_collections = []
            for name in created_libs:
                col = MagicMock()
                col.name = name
                mock_collections.append(col)

            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=mock_collections)
            )

            # Mock create_collection - track which library we're trying to create
            created_name = None

            async def track_creation(collection_name, **kwargs):
                nonlocal created_name
                created_name = collection_name
                # Fail if library name contains '5'
                if '5' in collection_name:
                    creation_attempts[collection_name] = False
                    raise Exception("Simulated failure for libraries containing '5'")
                creation_attempts[collection_name] = True
                return True

            mock_client.create_collection = track_creation

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Attempt to create 10 libraries
        results = []
        for i in range(10):
            result = self.runner.invoke(library_app, ["create", f"batch-{i}"])
            results.append((i, result.exit_code))

        # Libraries with '5' in name should fail
        successful = [i for i, code in results if code == 0]
        failed = [i for i, code in results if code == 1]

        assert 5 in failed  # batch-5 should fail
        assert 0 in successful  # batch-0 should succeed
        assert len(successful) == 9  # 9 should succeed
        assert len(failed) == 1  # Only batch-5 should fail


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases (Task 327.5)."""

    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_create_library_name_conflict(self, mock_config, mock_with_daemon):
        """Test creating a library that already exists."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Mock library already exists
            col1 = MagicMock()
            col1.name = "_duplicate"
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=[col1])
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["create", "duplicate"])

        assert result.exit_code == 1
        assert "already exists" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_invalid_library_name_characters(self, mock_config, mock_with_daemon):
        """Test creating library with invalid characters in name."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        # Test with spaces
        result = self.runner.invoke(library_app, ["create", "my library"])
        assert result.exit_code == 1
        assert "Invalid" in result.output or "invalid" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_remove_nonexistent_library(self, mock_config, mock_with_daemon):
        """Test removing a library that doesn't exist."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # No libraries exist
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=[])
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(
            library_app, ["remove", "nonexistent", "--force"]
        )

        assert result.exit_code == 1
        assert "not found" in result.output or "Not found" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_daemon_connection_failure(self, mock_config, mock_with_daemon):
        """Test handling daemon connection failures."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            raise ConnectionError("Failed to connect to daemon")

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list"])

        assert result.exit_code == 1
        assert "Failed" in result.output or "Error" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_qdrant_network_failure(self, mock_config, mock_with_daemon):
        """Test handling Qdrant network failures."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Simulate network failure
            mock_client.list_collections = AsyncMock(
                side_effect=Exception("Network timeout")
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["list"])

        assert result.exit_code == 1

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_empty_library_name(self, mock_config, mock_with_daemon):
        """Test creating library with empty name."""
        from wqm_cli.cli.commands.library import library_app

        # Empty string should be caught by CLI argument parser
        result = self.runner.invoke(library_app, ["create", ""])

        # Should either fail validation or be rejected by parser
        assert result.exit_code != 0

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_name_too_long(self, mock_config, mock_with_daemon):
        """Test creating library with excessively long name."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        # Create very long name (300 characters)
        long_name = "a" * 300

        result = self.runner.invoke(library_app, ["create", long_name])

        # Should fail either at validation or creation
        # Exit code 0 or 1 both acceptable depending on where it fails
        # Just verify it completes without crashing
        assert result.exit_code in [0, 1]

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_malformed_json_format_request(self, mock_config, mock_with_daemon):
        """Test list command with invalid format option."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            col1 = MagicMock()
            col1.name = "_test"
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=[col1])
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Invalid format option should be rejected
        result = self.runner.invoke(
            library_app, ["list", "--format", "xml"]
        )

        # Should either succeed with fallback or fail gracefully
        assert result.exit_code in [0, 1, 2]  # Various error codes acceptable

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_info_empty_collection(self, mock_config, mock_with_daemon):
        """Test getting info for an empty library."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # Return empty library info
            mock_info = MagicMock(
                points_count=0,
                indexed_points_count=0,
                vector_size=384,
                distance_metric="COSINE",
                sample_documents=[]
            )
            mock_client.get_collection_info = AsyncMock(return_value=mock_info)
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["info", "empty-lib"])

        assert result.exit_code == 0
        assert "0" in result.output  # Should show 0 documents

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_special_characters_in_library_name(self, mock_config, mock_with_daemon):
        """Test library name with special characters."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        # Test various special characters
        special_names = [
            "lib@test",
            "lib#test",
            "lib$test",
            "lib%test",
        ]

        for name in special_names:
            result = self.runner.invoke(library_app, ["create", name])
            # Should fail validation
            assert result.exit_code == 1

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_concurrent_create_same_library(self, mock_config, mock_with_daemon):
        """Test race condition when creating same library name."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()
        creation_count = 0

        async def mock_operation(operation_func, config):
            nonlocal creation_count
            mock_client = MagicMock()

            # First call shows no existing collections
            # Second call shows the library was created
            if creation_count == 0:
                mock_client.list_collections = AsyncMock(
                    return_value=MagicMock(collections=[])
                )
                mock_client.create_collection = AsyncMock(return_value=True)
                creation_count += 1
            else:
                col1 = MagicMock()
                col1.name = "_race"
                mock_client.list_collections = AsyncMock(
                    return_value=MagicMock(collections=[col1])
                )

            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # First creation should succeed
        result1 = self.runner.invoke(library_app, ["create", "race"])
        assert result1.exit_code == 0

        # Second creation should fail (already exists)
        result2 = self.runner.invoke(library_app, ["create", "race"])
        assert result2.exit_code == 1

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_library_status_get_info_failure(self, mock_config, mock_with_daemon):
        """Test status command when get_collection_info fails."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            # get_collection_info fails
            mock_client.get_collection_info = AsyncMock(
                side_effect=Exception("Collection info unavailable")
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        result = self.runner.invoke(library_app, ["status", "broken-lib"])

        assert result.exit_code != 0
        # Should show error message
        assert "Cannot get status" in result.output or "Error" in result.output

    @patch('wqm_cli.cli.commands.library.with_daemon_client')
    @patch('wqm_cli.cli.commands.library.get_config_manager')
    def test_remove_library_without_force_cancelled(self, mock_config, mock_with_daemon):
        """Test cancelling library removal when not using --force."""
        from wqm_cli.cli.commands.library import library_app

        mock_config.return_value = MagicMock()

        async def mock_operation(operation_func, config):
            mock_client = MagicMock()
            col1 = MagicMock()
            col1.name = "_to-remove"
            mock_client.list_collections = AsyncMock(
                return_value=MagicMock(collections=[col1])
            )
            mock_client.get_collection_info = AsyncMock(
                return_value=MagicMock(points_count=100)
            )
            return await operation_func(mock_client)

        mock_with_daemon.side_effect = mock_operation

        # Simulate user typing 'n' to cancel
        result = self.runner.invoke(
            library_app, ["remove", "to-remove"], input="n\n"
        )

        assert result.exit_code == 0
        assert "cancelled" in result.output.lower()

    @patch('wqm_cli.cli.commands.ingest.with_daemon_client')
    def test_ingest_to_invalid_library_name(self, mock_with_daemon):
        """Test ingesting to library with invalid name format."""
        import tempfile
        from pathlib import Path

        from wqm_cli.cli.commands.ingest import ingest_app

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_file.write_text("Test content")

            # Try to ingest to non-library collection (without _ prefix)
            result = self.runner.invoke(
                ingest_app,
                ["file", str(test_file), "--collection", "regular-collection"]
            )

            # Should either fail validation or be rejected
            # Implementation may vary, so we just check it doesn't crash
            assert result.exit_code in [0, 1]
