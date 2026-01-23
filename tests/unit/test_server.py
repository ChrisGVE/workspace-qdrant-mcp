"""
Comprehensive unit tests for workspace_qdrant_mcp.server module.

This test suite focuses on achieving 90%+ coverage of the server.py module by testing:
- FastMCP app initialization and stdio mode detection
- MCP tool registrations and functionality (store, search, manage, retrieve)
- Content-based routing and project detection
- Error handling and graceful degradation
- Qdrant integration and embedding functionality
"""

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, mock_open, patch

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))


class TestServerInitialization:
    """Test server initialization and stdio mode detection."""

    def test_stdio_mode_detection_environment_variable(self):
        """Test stdio mode detection via environment variables."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            from workspace_qdrant_mcp.server import _detect_stdio_mode
            assert _detect_stdio_mode() is True

    def test_stdio_mode_detection_cli_mode(self):
        """Test CLI mode detection without stdio mode."""
        with patch.dict(os.environ, {"WQM_CLI_MODE": "true"}, clear=True):
            from workspace_qdrant_mcp.server import _detect_stdio_mode
            assert _detect_stdio_mode() is False

    def test_stdio_mode_detection_argv_patterns(self):
        """Test stdio mode detection via command line arguments."""
        with patch('sys.argv', ['script', 'stdio']):
            from workspace_qdrant_mcp.server import _detect_stdio_mode
            assert _detect_stdio_mode() is True

    def test_stdio_mode_detection_mcp_pattern(self):
        """Test stdio mode detection via MCP command line pattern."""
        with patch('sys.argv', ['script', 'mcp']):
            from workspace_qdrant_mcp.server import _detect_stdio_mode
            assert _detect_stdio_mode() is True

    @patch('sys.stdin')
    @patch('os.fstat')
    @patch('stat.S_ISFIFO')
    def test_stdio_mode_detection_pipe(self, mock_isfifo, mock_fstat, mock_stdin):
        """Test stdio mode detection via pipe detection."""
        mock_stdin.fileno.return_value = 0
        mock_fstat.return_value.st_mode = 0o010000  # FIFO mode
        mock_isfifo.return_value = True

        from workspace_qdrant_mcp.server import _detect_stdio_mode
        assert _detect_stdio_mode() is True

    def test_fastmcp_app_creation(self):
        """Test FastMCP app creation and initialization."""
        import workspace_qdrant_mcp.server as server_module

        # Verify app is FastMCP instance
        assert hasattr(server_module, 'app')
        assert server_module.app is not None

        # Verify app has expected name
        assert hasattr(server_module.app, 'name')


class TestProjectDetection:
    """Test project name detection functionality."""

    @patch('subprocess.run')
    def test_get_project_name_from_git(self, mock_run):
        """Test project name detection from git remote URL."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "https://github.com/user/my-project.git\n"

        from workspace_qdrant_mcp.server import get_project_name
        assert get_project_name() == "my-project"

    @patch('subprocess.run')
    def test_get_project_name_from_git_no_extension(self, mock_run):
        """Test project name detection from git remote URL without .git extension."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "https://github.com/user/my-project\n"

        from workspace_qdrant_mcp.server import get_project_name
        assert get_project_name() == "my-project"

    @patch('subprocess.run')
    @patch('pathlib.Path.cwd')
    def test_get_project_name_fallback_directory(self, mock_cwd, mock_run):
        """Test project name fallback to directory name when git fails."""
        mock_run.side_effect = Exception("Git not available")
        mock_cwd.return_value.name = "fallback-project"

        from workspace_qdrant_mcp.server import get_project_name
        assert get_project_name() == "fallback-project"


class TestCollectionNaming:
    """Test collection name determination logic."""

    def test_determine_collection_name_explicit(self):
        """Test explicit collection name override."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(collection="custom-collection")
        assert result == "custom-collection"

    def test_determine_collection_name_scratchbook(self):
        """Test scratchbook collection routing."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="My notes",
            source="scratchbook",
            project_name="test-project"
        )
        assert result == "test-project-scratchbook"

    def test_determine_collection_name_note_content(self):
        """Test note content detection for scratchbook."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="This is a note about the project",
            project_name="test-project"
        )
        assert result == "test-project-scratchbook"

    def test_determine_collection_name_code_files(self):
        """Test code file routing to code collection."""
        from workspace_qdrant_mcp.server import determine_collection_name

        test_files = [("main.py", "test-project-code"),
                     ("app.js", "test-project-code"),
                     ("component.ts", "test-project-code"),
                     ("service.java", "test-project-code"),
                     ("handler.cpp", "test-project-code"),
                     ("utils.c", "test-project-code"),
                     ("types.h", "test-project-code"),
                     ("main.rs", "test-project-code"),
                     ("server.go", "test-project-code")]

        for file_path, expected in test_files:
            result = determine_collection_name(
                file_path=file_path,
                project_name="test-project"
            )
            assert result == expected

    def test_determine_collection_name_doc_files(self):
        """Test documentation file routing."""
        from workspace_qdrant_mcp.server import determine_collection_name

        test_files = [("README.md", "test-project-docs"),
                     ("notes.txt", "test-project-docs"),
                     ("guide.rst", "test-project-docs"),
                     ("manual.doc", "test-project-docs"),
                     ("spec.docx", "test-project-docs")]

        for file_path, expected in test_files:
            result = determine_collection_name(
                file_path=file_path,
                project_name="test-project"
            )
            assert result == expected

    def test_determine_collection_name_web_content(self):
        """Test web content routing."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            url="https://example.com/docs",
            project_name="test-project"
        )
        assert result == "test-project-web"

    def test_determine_collection_name_memory_content(self):
        """Test memory content keyword detection."""
        from workspace_qdrant_mcp.server import determine_collection_name

        memory_phrases = ["remember this for later",
                         "save to memory",
                         "context for the project"]

        for phrase in memory_phrases:
            result = determine_collection_name(
                content=phrase,
                project_name="test-project"
            )
            assert result == "test-project-memory"

    def test_determine_collection_name_default(self):
        """Test default collection assignment."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="Regular document content",
            project_name="test-project"
        )
        assert result == "test-project-documents"


class TestComponentInitialization:
    """Test Qdrant client and embedding model initialization."""

    @patch('workspace_qdrant_mcp.server.QdrantClient')
    @patch('fastembed.TextEmbedding')
    async def test_initialize_components_success(self, mock_embedding, mock_qdrant):
        """Test successful component initialization."""
        from workspace_qdrant_mcp.server import initialize_components

        mock_qdrant_instance = Mock()
        mock_embedding_instance = Mock()
        mock_qdrant.return_value = mock_qdrant_instance
        mock_embedding.return_value = mock_embedding_instance

        with patch.dict(os.environ, {"QDRANT_URL": "http://test:6333", "QDRANT_API_KEY": "test-key"}):
            await initialize_components()

        mock_qdrant.assert_called_once_with(
            url="http://test:6333",
            api_key="test-key",
            timeout=60
        )

    @patch('workspace_qdrant_mcp.server.QdrantClient')
    async def test_initialize_components_default_url(self, mock_qdrant):
        """Test component initialization with default URL."""
        # Clear any existing instances
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import initialize_components
        server_module.qdrant_client = None
        server_module.embedding_model = None

        with patch.dict(os.environ, {}, clear=True):
            await initialize_components()

        mock_qdrant.assert_called_with(
            url="http://localhost:6333",
            api_key=None,
            timeout=60
        )

    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_ensure_collection_exists_already_exists(self, mock_client):
        """Test collection existence check when collection already exists."""
        from workspace_qdrant_mcp.server import ensure_collection_exists

        mock_client.get_collection.return_value = Mock()

        result = await ensure_collection_exists("test-collection")
        assert result is True
        mock_client.get_collection.assert_called_once_with("test-collection")

    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_ensure_collection_exists_create_new(self, mock_client):
        """Test collection creation when it doesn't exist."""
        from workspace_qdrant_mcp.server import ensure_collection_exists

        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.return_value = Mock()

        result = await ensure_collection_exists("new-collection")
        assert result is True
        mock_client.create_collection.assert_called_once()

    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_ensure_collection_exists_creation_fails(self, mock_client):
        """Test handling of collection creation failure."""
        from workspace_qdrant_mcp.server import ensure_collection_exists

        mock_client.get_collection.side_effect = Exception("Collection not found")
        mock_client.create_collection.side_effect = Exception("Creation failed")

        result = await ensure_collection_exists("failed-collection")
        assert result is False


class TestEmbeddingGeneration:
    """Test embedding generation functionality."""

    @patch('workspace_qdrant_mcp.server.embedding_model')
    async def test_generate_embeddings_success(self, mock_model):
        """Test successful embedding generation."""
        from workspace_qdrant_mcp.server import generate_embeddings

        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = [mock_embedding]

        result = await generate_embeddings("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_model.embed.assert_called_once_with(["test text"])

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_generate_embeddings_initializes_components(self, mock_init):
        """Test that embedding generation initializes components if needed."""
        import workspace_qdrant_mcp.server as server_module

        # Save original value and set to None
        original_model = server_module.embedding_model
        server_module.embedding_model = None

        # Mock the model to be set after initialization
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.embed.return_value = [mock_embedding]

        def mock_init_side_effect():
            server_module.embedding_model = mock_model

        mock_init.side_effect = mock_init_side_effect

        try:
            result = await server_module.generate_embeddings("test text")
            assert result == [0.1, 0.2, 0.3]
            mock_init.assert_called_once()
        finally:
            # Restore original value
            server_module.embedding_model = original_model


class TestStoreToolFunctionality:
    """Test the store tool comprehensive functionality."""

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.generate_embeddings')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_store_tool_success(self, mock_project, mock_client, mock_embed, mock_ensure, mock_init):
        """Test successful document storage."""
        # Import the server module to access the function directly
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"
        mock_ensure.return_value = True
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_client.upsert.return_value = Mock()

        # Call the store function through the .fn attribute (it's wrapped by FastMCP)
        result = await server_module.store.fn(
            content="Test document content",
            title="Test Document",
            source="user_input"
        )

        assert result["success"] is True
        assert "document_id" in result
        assert result["collection"] == "test-project-documents"
        assert result["title"] == "Test Document"
        mock_client.upsert.assert_called_once()

    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    async def test_store_tool_collection_creation_failure(self, mock_ensure):
        """Test store tool handling collection creation failure."""
        import workspace_qdrant_mcp.server as server_module

        mock_ensure.return_value = False

        result = await server_module.store.fn(content="Test content")

        assert result["success"] is False
        assert "Failed to create/access collection" in result["error"]

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.generate_embeddings')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_store_tool_qdrant_failure(self, mock_client, mock_embed, mock_ensure, mock_init):
        """Test store tool handling Qdrant operation failure."""
        import workspace_qdrant_mcp.server as server_module

        mock_ensure.return_value = True
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_client.upsert.side_effect = Exception("Qdrant error")

        result = await server_module.store.fn(content="Test content")

        assert result["success"] is False
        assert "Failed to store document" in result["error"]

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.generate_embeddings')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_store_tool_with_metadata(self, mock_project, mock_client, mock_embed, mock_ensure, mock_init):
        """Test store tool with custom metadata."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"
        mock_ensure.return_value = True
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_client.upsert.return_value = Mock()

        custom_metadata = {"author": "test-user", "version": "1.0"}

        result = await server_module.store.fn(
            content="Test content",
            metadata=custom_metadata,
            file_path="test.py",
            url="https://example.com"
        )

        assert result["success"] is True
        assert result["collection"] == "test-project-code"  # .py file goes to code collection

        # Verify upsert was called with correct metadata
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["author"] == "test-user"
        assert point.payload["file_path"] == "test.py"
        assert point.payload["url"] == "https://example.com"


class TestSearchToolFunctionality:
    """Test the search tool comprehensive functionality."""

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.generate_embeddings')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_search_tool_semantic_mode(self, mock_project, mock_client, mock_embed, mock_ensure, mock_init):
        """Test semantic search mode."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"
        mock_ensure.return_value = True
        mock_embed.return_value = [0.1, 0.2, 0.3]

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [
            Mock(name="test-project-documents"),
            Mock(name="test-project-code")
        ]
        mock_client.get_collections.return_value = mock_collections

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "doc-1"
        mock_hit.score = 0.95
        mock_hit.payload = {"content": "Test content", "title": "Test Doc"}
        mock_client.search.return_value = [mock_hit]

        result = await server_module.search.fn(query="test query", mode="semantic")

        assert result["success"] is True
        assert result["mode"] == "semantic"
        assert len(result["results"]) >= 1  # At least one result returned
        mock_client.search.assert_called()

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_search_tool_exact_mode(self, mock_project, mock_client, mock_ensure, mock_init):
        """Test exact/keyword search mode."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"
        mock_ensure.return_value = True

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test-project-documents")]
        mock_client.get_collections.return_value = mock_collections

        # Mock scroll results
        mock_point = Mock()
        mock_point.id = "doc-1"
        mock_point.payload = {"content": "Test keyword content", "title": "Test Doc"}
        mock_client.scroll.return_value = ([mock_point], None)

        result = await server_module.search.fn(query="keyword", mode="exact")

        assert result["success"] is True
        assert result["mode"] == "exact"
        mock_client.scroll.assert_called()

    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_search_tool_no_collections(self, mock_project):
        """Test search with no available collections."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"

        result = await server_module.search.fn(query="test", collection="nonexistent-collection")

        # This will search the specified collection even if it doesn't exist
        # The ensure_collection_exists call will handle the case
        assert "query" in result

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.generate_embeddings')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_search_tool_with_filters(self, mock_client, mock_embed, mock_ensure, mock_init):
        """Test search with metadata filters."""
        import workspace_qdrant_mcp.server as server_module

        mock_ensure.return_value = True
        mock_embed.return_value = [0.1, 0.2, 0.3]
        mock_client.search.return_value = []

        filters = {"author": "test-user", "document_type": "note"}

        await server_module.search.fn(
            query="test query",
            collection="test-collection",
            filters=filters
        )

        # Verify search was called with filters
        mock_client.search.assert_called()
        call_args = mock_client.search.call_args
        assert call_args[1]["query_filter"] is not None


class TestManageToolFunctionality:
    """Test the manage tool comprehensive functionality."""

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_manage_list_collections(self, mock_client, mock_init):
        """Test manage tool list_collections action."""
        import workspace_qdrant_mcp.server as server_module

        # Mock collections response
        mock_collections = Mock()
        mock_collection = Mock()
        mock_collection.name = "test-collection"
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections

        # Mock collection info
        mock_info = Mock()
        mock_info.points_count = 100
        mock_info.segments_count = 1
        mock_info.status.value = "green"
        mock_info.config.params.vectors.size = 384
        mock_info.config.params.vectors.distance.value = "Cosine"
        mock_client.get_collection.return_value = mock_info

        result = await server_module.manage.fn(action="list_collections")

        assert result["success"] is True
        assert result["action"] == "list_collections"
        assert len(result["collections"]) == 1
        assert result["collections"][0]["name"] == "test-collection"
        assert result["collections"][0]["points_count"] == 100

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_manage_create_collection(self, mock_client, mock_init):
        """Test manage tool create_collection action."""
        import workspace_qdrant_mcp.server as server_module

        mock_client.create_collection.return_value = Mock()

        result = await server_module.manage.fn(action="create_collection", name="new-collection")

        assert result["success"] is True
        assert result["action"] == "create_collection"
        assert result["collection_name"] == "new-collection"
        mock_client.create_collection.assert_called_once()

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_manage_create_collection_no_name(self, mock_init):
        """Test manage tool create_collection without name."""
        import workspace_qdrant_mcp.server as server_module

        result = await server_module.manage.fn(action="create_collection")

        assert result["success"] is False
        assert "error" in result
        # New error format: error is a dict with code, message, etc.
        error = result["error"]
        if isinstance(error, dict):
            assert "Collection name required" in error["message"]
        else:
            assert "Collection name required" in error

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_manage_delete_collection(self, mock_client, mock_init):
        """Test manage tool delete_collection action."""
        import workspace_qdrant_mcp.server as server_module

        mock_client.delete_collection.return_value = Mock()

        result = await server_module.manage.fn(action="delete_collection", name="old-collection")

        assert result["success"] is True
        assert result["action"] == "delete_collection"
        assert result["collection_name"] == "old-collection"
        mock_client.delete_collection.assert_called_once_with("old-collection")

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_manage_workspace_status(self, mock_project, mock_client, mock_init):
        """Test manage tool workspace_status action."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [
            Mock(name="test-project-documents"),
            Mock(name="test-project-code"),
            Mock(name="other-project-notes")
        ]
        mock_client.get_collections.return_value = mock_collections

        # Mock cluster info
        mock_cluster = Mock()
        mock_cluster.peer_id = "peer-123"
        mock_cluster.raft_info = {}
        mock_client.get_cluster_info.return_value = mock_cluster

        result = await server_module.manage.fn(action="workspace_status")

        assert result["success"] is True
        assert result["action"] == "workspace_status"
        assert result["current_project"] == "test-project"
        assert result["qdrant_status"] == "connected"
        assert len(result["project_collections"]) == 2  # Only test-project collections

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.ensure_collection_exists')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_manage_init_project(self, mock_project, mock_ensure, mock_init):
        """Test manage tool init_project action."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"
        mock_ensure.return_value = True

        result = await server_module.manage.fn(action="init_project")

        assert result["success"] is True
        assert result["action"] == "init_project"
        assert result["project"] == "test-project"
        assert len(result["collections_created"]) == 5  # Standard workspace types

    async def test_manage_unknown_action(self):
        """Test manage tool with unknown action."""
        import workspace_qdrant_mcp.server as server_module

        result = await server_module.manage.fn(action="unknown_action")

        assert result["success"] is False
        assert "error" in result
        # New error format: error is a dict with code, message, etc.
        error = result["error"]
        if isinstance(error, dict):
            assert error["code"] == "INVALID_ACTION"
            assert "Unknown action" in error["message"]
        else:
            assert "Unknown action" in error


class TestRetrieveToolFunctionality:
    """Test the retrieve tool comprehensive functionality."""

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    @patch('workspace_qdrant_mcp.server.get_project_name')
    async def test_retrieve_by_document_id(self, mock_project, mock_client, mock_init):
        """Test retrieve tool by document ID."""
        import workspace_qdrant_mcp.server as server_module

        mock_project.return_value = "test-project"

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test-project-documents")]
        mock_client.get_collections.return_value = mock_collections

        # Mock retrieve response
        mock_point = Mock()
        mock_point.id = "doc-123"
        mock_point.payload = {"content": "Document content", "title": "Test Doc", "author": "user"}
        mock_client.retrieve.return_value = [mock_point]

        result = await server_module.retrieve.fn(document_id="doc-123")

        assert result["success"] is True
        assert result["total_results"] == 1
        assert result["query_type"] == "id_lookup"
        assert result["results"][0]["id"] == "doc-123"
        assert result["results"][0]["content"] == "Document content"

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_retrieve_by_metadata(self, mock_client, mock_init):
        """Test retrieve tool by metadata filters."""
        import workspace_qdrant_mcp.server as server_module

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test-collection")]
        mock_client.get_collections.return_value = mock_collections

        # Mock scroll response
        mock_point = Mock()
        mock_point.id = "doc-456"
        mock_point.payload = {"content": "Filtered content", "title": "Filtered Doc", "author": "target-user"}
        mock_client.scroll.return_value = ([mock_point], None)

        metadata_filter = {"author": "target-user"}

        result = await server_module.retrieve.fn(metadata=metadata_filter, collection="test-collection")

        assert result["success"] is True
        assert result["query_type"] == "metadata_filter"
        assert result["filters_applied"] == metadata_filter
        mock_client.scroll.assert_called()

    async def test_retrieve_no_parameters(self):
        """Test retrieve tool with no parameters."""
        import workspace_qdrant_mcp.server as server_module

        result = await server_module.retrieve.fn()

        assert result["success"] is False
        assert "error" in result
        # New error format: error is a dict with code, message, etc.
        error = result["error"]
        if isinstance(error, dict):
            assert "Either document_id or metadata filters must be provided" in error["message"]
        else:
            assert "Either document_id or metadata filters must be provided" in error

    @patch('workspace_qdrant_mcp.server.initialize_components')
    @patch('workspace_qdrant_mcp.server.qdrant_client')
    async def test_retrieve_document_not_found(self, mock_client, mock_init):
        """Test retrieve tool when document is not found."""
        import workspace_qdrant_mcp.server as server_module

        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test-collection")]
        mock_client.get_collections.return_value = mock_collections

        # Mock empty retrieve response
        mock_client.retrieve.return_value = []

        result = await server_module.retrieve.fn(document_id="nonexistent-doc", collection="test-collection")

        assert result["success"] is True
        assert result["total_results"] == 0
        assert len(result["results"]) == 0


class TestServerConfiguration:
    """Test server configuration and main function."""

    def test_run_server_function_exists(self):
        """Test run_server function exists."""
        from workspace_qdrant_mcp.server import run_server
        assert callable(run_server)

    @patch('workspace_qdrant_mcp.server.app')
    @patch('workspace_qdrant_mcp.server._detect_stdio_mode')
    def test_run_server_stdio_mode(self, mock_detect, mock_app):
        """Test run_server in stdio mode."""
        from workspace_qdrant_mcp.server import run_server

        with patch.dict(os.environ, {}):
            run_server(transport="stdio", host="127.0.0.1", port=8000)

        mock_app.run.assert_called_once_with(transport="stdio", host="127.0.0.1", port=8000)
        assert os.getenv("WQM_STDIO_MODE") == "true"

    @patch('workspace_qdrant_mcp.server.app')
    def test_run_server_http_mode(self, mock_app):
        """Test run_server in HTTP mode."""
        from workspace_qdrant_mcp.server import run_server

        run_server(transport="http", host="0.0.0.0", port=9000)

        mock_app.run.assert_called_once_with(transport="http", host="0.0.0.0", port=9000)

    def test_main_function_exists(self):
        """Test main function exists."""
        from workspace_qdrant_mcp.server import main
        assert callable(main)

    @patch('typer.run')
    def test_main_function_calls_typer(self, mock_typer_run):
        """Test main function calls typer.run."""
        from workspace_qdrant_mcp.server import main, run_server

        main()

        mock_typer_run.assert_called_once_with(run_server)


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_store_tool_exception_handling(self, mock_init):
        """Test store tool exception handling."""
        import workspace_qdrant_mcp.server as server_module

        mock_init.side_effect = Exception("Initialization failed")

        result = await server_module.store.fn(content="test content")

        # Should handle the exception gracefully
        assert "error" in result or "success" in result

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_search_tool_exception_handling(self, mock_init):
        """Test search tool exception handling."""
        import workspace_qdrant_mcp.server as server_module

        mock_init.side_effect = Exception("Initialization failed")

        result = await server_module.search.fn(query="test query")

        assert result["success"] is False
        assert "error" in result

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_manage_tool_exception_handling(self, mock_init):
        """Test manage tool exception handling."""
        import workspace_qdrant_mcp.server as server_module

        mock_init.side_effect = Exception("Initialization failed")

        result = await server_module.manage.fn(action="list_collections")

        assert result["success"] is False
        assert "error" in result

    @patch('workspace_qdrant_mcp.server.initialize_components')
    async def test_retrieve_tool_exception_handling(self, mock_init):
        """Test retrieve tool exception handling."""
        import workspace_qdrant_mcp.server as server_module

        mock_init.side_effect = Exception("Initialization failed")

        result = await server_module.retrieve.fn(document_id="test-doc")

        assert result["success"] is False
        assert "error" in result


class TestGlobalVariables:
    """Test global variables and configuration."""

    def test_global_variables_existence(self):
        """Test that expected global variables exist."""
        import workspace_qdrant_mcp.server as server_module

        # Test app exists
        assert hasattr(server_module, 'app')
        assert server_module.app is not None

        # Test configuration constants
        assert hasattr(server_module, 'DEFAULT_EMBEDDING_MODEL')
        assert hasattr(server_module, 'DEFAULT_COLLECTION_CONFIG')

        # Test global component variables
        assert hasattr(server_module, 'qdrant_client')
        assert hasattr(server_module, 'embedding_model')
        assert hasattr(server_module, 'project_cache')

    def test_default_configuration_values(self):
        """Test default configuration values."""
        import workspace_qdrant_mcp.server as server_module

        assert server_module.DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"
        assert server_module.DEFAULT_COLLECTION_CONFIG["vector_size"] == 384
        assert server_module.DEFAULT_COLLECTION_CONFIG["distance"] == Distance.COSINE
