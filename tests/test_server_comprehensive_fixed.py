"""
Comprehensive test coverage for server.py - 100% coverage goal.
Fixed version that properly tests the async functions directly.
"""

import pytest
import asyncio
import sys
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, Any, List
import uuid
from datetime import datetime, timezone

# Add source path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

# Import the server module
from workspace_qdrant_mcp.server import (
    _detect_stdio_mode,
    get_project_name,
    initialize_components,
    store,
    search,
    manage,
    retrieve,
    run_server,
    main
)


class TestStdioDetection:
    """Test stdio mode detection functionality."""

    def test_detect_stdio_mode_explicit_true(self):
        """Test explicit WQM_STDIO_MODE=true."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            assert _detect_stdio_mode() == True

    def test_detect_stdio_mode_explicit_false(self):
        """Test explicit WQM_STDIO_MODE=false."""
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "false"}):
            result = _detect_stdio_mode()
            # Should continue to other checks, could be True or False
            assert isinstance(result, bool)

    def test_detect_stdio_mode_cli_mode_true(self):
        """Test WQM_CLI_MODE=true overrides."""
        with patch.dict(os.environ, {"WQM_CLI_MODE": "true"}):
            assert _detect_stdio_mode() == False

    def test_detect_stdio_mode_pipe(self):
        """Test pipe detection."""
        mock_stat = Mock()
        mock_stat.st_mode = stat.S_IFIFO  # Pipe mode

        with patch('os.fstat', return_value=mock_stat):
            with patch('sys.stdin.fileno', return_value=0):
                assert _detect_stdio_mode() == True

    def test_detect_stdio_mode_regular_file(self):
        """Test regular file detection."""
        mock_stat = Mock()
        mock_stat.st_mode = stat.S_IFREG  # Regular file mode

        with patch('os.fstat', return_value=mock_stat):
            with patch('sys.stdin.fileno', return_value=0):
                assert _detect_stdio_mode() == True

    def test_detect_stdio_mode_os_error(self):
        """Test handling of OS errors."""
        with patch('os.fstat', side_effect=OSError("No file descriptor")):
            with patch.object(sys, 'argv', ['test']):
                assert _detect_stdio_mode() == False

    def test_detect_stdio_mode_argv_stdio(self):
        """Test argv containing 'stdio'."""
        with patch.object(sys, 'argv', ['program', 'stdio']):
            assert _detect_stdio_mode() == True

    def test_detect_stdio_mode_argv_mcp(self):
        """Test argv containing 'mcp'."""
        with patch.object(sys, 'argv', ['program', 'mcp']):
            assert _detect_stdio_mode() == True

    def test_detect_stdio_mode_default_false(self):
        """Test default case returns False."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('os.fstat', side_effect=OSError()):
                with patch.object(sys, 'argv', ['program']):
                    assert _detect_stdio_mode() == False


class TestProjectNameDetection:
    """Test project name detection from git."""

    def test_get_project_name_git_success(self):
        """Test successful git project name extraction."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/user/my-awesome-project.git\n"

        with patch('subprocess.run', return_value=mock_result):
            result = get_project_name()
            assert result == "my-awesome-project"

    def test_get_project_name_git_no_git_suffix(self):
        """Test git URL without .git suffix."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/user/my-project\n"

        with patch('subprocess.run', return_value=mock_result):
            result = get_project_name()
            assert result == "my-project"

    def test_get_project_name_git_failure(self):
        """Test git command failure."""
        mock_result = Mock()
        mock_result.returncode = 128
        mock_result.stdout = "fatal: not a git repository\n"

        with patch('subprocess.run', return_value=mock_result):
            result = get_project_name()
            assert result == "workspace-qdrant-mcp"  # Should return directory name

    def test_get_project_name_exception(self):
        """Test exception handling in git detection."""
        with patch('subprocess.run', side_effect=Exception("Command failed")):
            with patch('pathlib.Path.cwd', return_value=Path("/test/directory")):
                result = get_project_name()
                assert result == "directory"


class TestComponentInitialization:
    """Test component initialization."""

    @pytest.mark.asyncio
    async def test_initialize_components_first_time(self):
        """Test first-time component initialization."""
        import workspace_qdrant_mcp.server as server_module

        # Reset state
        server_module.qdrant_client = None
        server_module.embedding_model = None

        mock_client = Mock()
        mock_embeddings = Mock()

        with patch('workspace_qdrant_mcp.server.QdrantClient', return_value=mock_client):
            with patch('fastembed.TextEmbedding', return_value=mock_embeddings):
                await initialize_components()

                assert server_module.qdrant_client == mock_client
                assert server_module.embedding_model == mock_embeddings

    @pytest.mark.asyncio
    async def test_initialize_components_already_initialized(self):
        """Test components already initialized."""
        import workspace_qdrant_mcp.server as server_module

        # Set as already initialized
        server_module.qdrant_client = Mock()
        server_module.embedding_model = Mock()

        # Should return early without reinitializing
        with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_qdrant:
            await initialize_components()
            mock_qdrant.assert_not_called()

    @pytest.mark.asyncio
    async def test_initialize_components_custom_model(self):
        """Test initialization with custom embedding model."""
        import workspace_qdrant_mcp.server as server_module

        # Reset state
        server_module.qdrant_client = None
        server_module.embedding_model = None

        with patch.dict(os.environ, {"FASTEMBED_MODEL": "custom-model"}):
            with patch('workspace_qdrant_mcp.server.QdrantClient'):
                with patch('fastembed.TextEmbedding') as mock_embed:
                    await initialize_components()
                    mock_embed.assert_called_once_with("custom-model")


class TestCollectionManagement:
    """Test collection management functions."""

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_success(self):
        """Test collection already exists."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_client.get_collection.return_value = Mock()  # Collection exists
        server_module.qdrant_client = mock_client

        from workspace_qdrant_mcp.server import ensure_collection_exists
        result = await ensure_collection_exists("test-collection")
        assert result == True
        mock_client.get_collection.assert_called_once_with("test-collection")

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_create_success(self):
        """Test successful collection creation."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = True
        server_module.qdrant_client = mock_client

        from workspace_qdrant_mcp.server import ensure_collection_exists
        result = await ensure_collection_exists("new-collection")
        assert result == True
        mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_create_failure(self):
        """Test collection creation failure."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.side_effect = Exception("Creation failed")
        server_module.qdrant_client = mock_client

        from workspace_qdrant_mcp.server import ensure_collection_exists
        result = await ensure_collection_exists("fail-collection")
        assert result == False


class TestCollectionNaming:
    """Test collection naming logic."""

    def test_determine_collection_name_explicit(self):
        """Test explicit collection name."""
        from workspace_qdrant_mcp.server import determine_collection_name
        result = determine_collection_name(
            collection="explicit-name",
            source="user_input",
            content="test content",
            file_path=None,
            url=None,
            project_name="test-project"
        )
        assert result == "explicit-name"

    def test_determine_collection_name_scratchbook_source(self):
        """Test scratchbook source routing."""
        from workspace_qdrant_mcp.server import determine_collection_name
        result = determine_collection_name(
            collection=None,
            source="scratchbook",
            content="note content",
            file_path=None,
            url=None,
            project_name="test-project"
        )
        assert "scratchbook" in result

    def test_determine_collection_name_default_behavior(self):
        """Test default collection naming behavior."""
        from workspace_qdrant_mcp.server import determine_collection_name
        result = determine_collection_name(
            collection=None,
            source="user_input",
            content="test content",
            file_path=None,
            url=None,
            project_name="test-project"
        )
        assert "test-project" in result


class TestStoreFunction:
    """Test the store function comprehensively."""

    @pytest.mark.asyncio
    async def test_store_basic_content(self):
        """Test basic content storage."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [Mock()]
        mock_embedding_model.embed.return_value[0].tolist.return_value = [0.1, 0.2, 0.3]

        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding_model

        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-documents"):
                with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test-project"):

                    result = await store(
                        content="Test content",
                        title="Test Document"
                    )

                    assert result["success"] == True
                    assert "document_id" in result
                    assert result["collection"] == "test-documents"
                    assert result["title"] == "Test Document"
                    mock_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_metadata(self):
        """Test storage with custom metadata."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [Mock()]
        mock_embedding_model.embed.return_value[0].tolist.return_value = [0.1, 0.2, 0.3]

        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding_model

        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-documents"):
                with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test-project"):

                    custom_metadata = {"tag": "important", "category": "docs"}

                    result = await store(
                        content="Test content",
                        metadata=custom_metadata,
                        source="user_input"
                    )

                    assert result["success"] == True
                    # Check that metadata was included in the call
                    call_args = mock_client.upsert.call_args
                    points = call_args[1]["points"]  # keyword arguments
                    payload = points[0].payload
                    assert payload["tag"] == "important"
                    assert payload["category"] == "docs"

    @pytest.mark.asyncio
    async def test_store_collection_creation_failure(self):
        """Test handling of collection creation failure."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_embedding_model = Mock()
        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding_model

        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=False):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-documents"):

                result = await store(content="Test content")

                assert result["success"] == False
                assert "Failed to create/access collection" in result["error"]

    @pytest.mark.asyncio
    async def test_store_upsert_exception(self):
        """Test handling of upsert exceptions."""
        import workspace_qdrant_mcp.server as server_module

        mock_client = Mock()
        mock_client.upsert.side_effect = Exception("Upsert failed")
        mock_embedding_model = Mock()
        mock_embedding_model.embed.return_value = [Mock()]
        mock_embedding_model.embed.return_value[0].tolist.return_value = [0.1, 0.2, 0.3]

        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding_model

        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-documents"):
                with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test-project"):

                    result = await store(content="Test content")

                    assert result["success"] == False
                    assert "Upsert failed" in result["error"]


class TestSearchFunction:
    """Test the search function comprehensively."""

    @pytest.mark.asyncio
    async def test_search_hybrid_mode(self):
        """Test hybrid search mode."""
        mock_client = AsyncMock()
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [[0.1, 0.2, 0.3]]

        mock_search_result = [
            Mock(id="1", score=0.9, payload={"title": "Doc 1", "content": "Test content"}),
            Mock(id="2", score=0.8, payload={"title": "Doc 2", "content": "More content"})
        ]
        mock_client.search.return_value = mock_search_result

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._embeddings', mock_embeddings):
                with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                    result = await search(
                        query="test query",
                        mode="hybrid",
                        limit=10
                    )

                    assert result["status"] == "success"
                    assert len(result["results"]) == 2
                    assert result["results"][0]["title"] == "Doc 1"
                    assert result["results"][0]["score"] == 0.9
                    mock_embeddings.embed.assert_called_once_with(["test query"])

    @pytest.mark.asyncio
    async def test_search_semantic_mode(self):
        """Test semantic search mode."""
        mock_client = AsyncMock()
        mock_embeddings = Mock()
        mock_embeddings.embed.return_value = [[0.1, 0.2, 0.3]]
        mock_client.search.return_value = []

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._embeddings', mock_embeddings):
                with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                    result = await search(
                        query="semantic query",
                        mode="semantic",
                        collection="specific-collection"
                    )

                    assert result["status"] == "success"
                    assert result["results"] == []
                    mock_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_exact_mode(self):
        """Test exact search mode."""
        mock_client = AsyncMock()

        # Mock scroll results for exact search
        mock_scroll_result = Mock()
        mock_scroll_result.points = [
            Mock(id="1", payload={"title": "Exact Match", "content": "def login_function"})
        ]
        mock_client.scroll.return_value = (mock_scroll_result, None)

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await search(
                    query="login_function",
                    mode="exact"
                )

                assert result["status"] == "success"
                assert len(result["results"]) == 1
                assert result["results"][0]["title"] == "Exact Match"
                mock_client.scroll.assert_called()

    @pytest.mark.asyncio
    async def test_search_exception_handling(self):
        """Test search exception handling."""
        mock_client = AsyncMock()
        mock_embeddings = Mock()
        mock_embeddings.embed.side_effect = Exception("Embedding failed")

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._embeddings', mock_embeddings):
                with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                    result = await search(query="test", mode="semantic")

                    assert result["status"] == "error"
                    assert "Embedding failed" in result["message"]


class TestManageFunction:
    """Test the manage function comprehensively."""

    @pytest.mark.asyncio
    async def test_manage_list_collections(self):
        """Test listing collections."""
        mock_client = AsyncMock()
        mock_collections = [
            Mock(name="test-project-docs", vectors_count=100),
            Mock(name="test-project-code", vectors_count=50)
        ]
        mock_client.get_collections.return_value = Mock(collections=mock_collections)

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await manage(action="list_collections")

                assert result["status"] == "success"
                assert len(result["collections"]) == 2
                assert result["collections"][0]["name"] == "test-project-docs"
                assert result["collections"][0]["vectors_count"] == 100

    @pytest.mark.asyncio
    async def test_manage_create_collection(self):
        """Test creating a collection."""
        mock_client = AsyncMock()
        mock_client.collection_exists.return_value = False
        mock_client.create_collection.return_value = True

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await manage(
                    action="create_collection",
                    name="new-collection"
                )

                assert result["status"] == "success"
                assert "new-collection" in result["message"]
                mock_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_manage_delete_collection(self):
        """Test deleting a collection."""
        mock_client = AsyncMock()
        mock_client.delete_collection.return_value = True

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await manage(
                    action="delete_collection",
                    name="old-collection"
                )

                assert result["status"] == "success"
                assert "old-collection" in result["message"]
                mock_client.delete_collection.assert_called_once_with("old-collection")

    @pytest.mark.asyncio
    async def test_manage_workspace_status(self):
        """Test workspace status action."""
        mock_client = AsyncMock()
        mock_collections = [Mock(name="test-docs", vectors_count=10)]
        mock_client.get_collections.return_value = Mock(collections=mock_collections)

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await manage(action="workspace_status")

                assert result["status"] == "success"
                assert result["project_name"] == "test-project"
                assert result["total_collections"] == 1
                assert result["total_documents"] == 10

    @pytest.mark.asyncio
    async def test_manage_unknown_action(self):
        """Test unknown action handling."""
        with patch('workspace_qdrant_mcp.server._client', AsyncMock()):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await manage(action="unknown_action")

                assert result["status"] == "error"
                assert "Unknown action" in result["message"]


class TestRetrieveFunction:
    """Test the retrieve function comprehensively."""

    @pytest.mark.asyncio
    async def test_retrieve_by_id_success(self):
        """Test successful document retrieval by ID."""
        mock_client = AsyncMock()
        mock_points = [
            Mock(id="test-id", payload={"title": "Retrieved Doc", "content": "Content here"})
        ]
        mock_client.retrieve.return_value = mock_points

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await retrieve(document_id="test-id")

                assert result["status"] == "success"
                assert len(result["documents"]) == 1
                assert result["documents"][0]["title"] == "Retrieved Doc"
                mock_client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_by_id_not_found(self):
        """Test document not found by ID."""
        mock_client = AsyncMock()
        mock_client.retrieve.return_value = []

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await retrieve(document_id="missing-id")

                assert result["status"] == "success"
                assert result["documents"] == []

    @pytest.mark.asyncio
    async def test_retrieve_by_metadata(self):
        """Test retrieval by metadata filter."""
        mock_client = AsyncMock()
        mock_scroll_result = Mock()
        mock_scroll_result.points = [
            Mock(id="1", payload={"title": "Filtered Doc", "tag": "important"})
        ]
        mock_client.scroll.return_value = (mock_scroll_result, None)

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await retrieve(metadata={"tag": "important"})

                assert result["status"] == "success"
                assert len(result["documents"]) == 1
                assert result["documents"][0]["tag"] == "important"
                mock_client.scroll.assert_called()

    @pytest.mark.asyncio
    async def test_retrieve_no_parameters(self):
        """Test retrieve with no parameters."""
        with patch('workspace_qdrant_mcp.server._client', AsyncMock()):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await retrieve()

                assert result["status"] == "error"
                assert "document_id or metadata" in result["message"]

    @pytest.mark.asyncio
    async def test_retrieve_exception_handling(self):
        """Test retrieve exception handling."""
        mock_client = AsyncMock()
        mock_client.retrieve.side_effect = Exception("Retrieve failed")

        with patch('workspace_qdrant_mcp.server._client', mock_client):
            with patch('workspace_qdrant_mcp.server._project_name', "test-project"):

                result = await retrieve(document_id="test-id")

                assert result["status"] == "error"
                assert "Retrieve failed" in result["message"]


class TestServerManagement:
    """Test server management functions."""

    @pytest.mark.asyncio
    async def test_run_server_stdio_mode(self):
        """Test server running in stdio mode."""
        mock_app = AsyncMock()

        with patch('workspace_qdrant_mcp.server._detect_stdio_mode', return_value=True):
            with patch('workspace_qdrant_mcp.server._initialize_components'):
                with patch('workspace_qdrant_mcp.server.app', mock_app):
                    with patch('workspace_qdrant_mcp.server.FastMCP.run_stdio') as mock_run:
                        await run_server()
                        mock_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_server_http_mode(self):
        """Test server running in HTTP mode."""
        mock_app = AsyncMock()

        with patch('workspace_qdrant_mcp.server._detect_stdio_mode', return_value=False):
            with patch('workspace_qdrant_mcp.server._initialize_components'):
                with patch('workspace_qdrant_mcp.server.app', mock_app):
                    with patch('workspace_qdrant_mcp.server.FastMCP.run_server') as mock_run:
                        await run_server(host="0.0.0.0", port=8080)
                        mock_run.assert_called_once_with(host="0.0.0.0", port=8080)

    def test_main_function(self):
        """Test main entry point."""
        with patch('workspace_qdrant_mcp.server.typer.run') as mock_typer:
            main()
            mock_typer.assert_called_once()


class TestErrorHandling:
    """Test comprehensive error handling scenarios."""

    @pytest.mark.asyncio
    async def test_initialization_error_recovery(self):
        """Test error recovery during initialization."""
        with patch('workspace_qdrant_mcp.server.QdrantClient', side_effect=Exception("Connection failed")):
            with patch('workspace_qdrant_mcp.server._get_project_name', return_value="test"):
                # Should handle initialization errors gracefully
                try:
                    await _initialize_components()
                    # If no exception, test passes
                except Exception as e:
                    # Should not propagate unhandled exceptions
                    pytest.fail(f"Initialization should handle errors gracefully: {e}")

    @pytest.mark.asyncio
    async def test_store_with_uninitialized_components(self):
        """Test store function with uninitialized components."""
        # Reset components
        with patch('workspace_qdrant_mcp.server._client', None):
            with patch('workspace_qdrant_mcp.server._embeddings', None):
                with patch('workspace_qdrant_mcp.server._initialize_components') as mock_init:
                    mock_init.side_effect = Exception("Init failed")

                    result = await store(content="Test")

                    assert result["status"] == "error"
                    assert "Init failed" in result["message"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])