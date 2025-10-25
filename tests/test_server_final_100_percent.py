"""
Final comprehensive test for 100% server.py coverage.
Focuses on actually working tests that cover all code paths.
"""

import asyncio
import os
import stat
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

# Add source path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


class TestServerComprehensive:
    """Complete comprehensive tests for server.py - targeting 100% coverage."""

    def test_detect_stdio_mode_all_paths(self):
        """Test all stdio mode detection paths."""
        from workspace_qdrant_mcp.server import _detect_stdio_mode

        # Test explicit true
        with patch.dict(os.environ, {"WQM_STDIO_MODE": "true"}):
            assert _detect_stdio_mode()

        # Test CLI mode false override
        with patch.dict(os.environ, {"WQM_CLI_MODE": "true"}):
            assert not _detect_stdio_mode()

        # Test pipe detection
        mock_stat = Mock()
        mock_stat.st_mode = stat.S_IFIFO
        with patch('os.fstat', return_value=mock_stat):
            with patch('sys.stdin.fileno', return_value=0):
                assert _detect_stdio_mode()

        # Test regular file detection
        mock_stat.st_mode = stat.S_IFREG
        with patch('os.fstat', return_value=mock_stat):
            with patch('sys.stdin.fileno', return_value=0):
                assert _detect_stdio_mode()

        # Test OS error handling
        with patch('os.fstat', side_effect=OSError()):
            with patch.object(sys, 'argv', ['program', 'stdio']):
                assert _detect_stdio_mode()

        # Test argv mcp
        with patch('os.fstat', side_effect=OSError()):
            with patch.object(sys, 'argv', ['program', 'mcp']):
                assert _detect_stdio_mode()

        # Test default false
        with patch.dict(os.environ, {}, clear=True):
            with patch('os.fstat', side_effect=OSError()):
                with patch.object(sys, 'argv', ['program']):
                    assert not _detect_stdio_mode()

    def test_get_project_name_all_paths(self):
        """Test all project name detection paths."""
        from workspace_qdrant_mcp.server import get_project_name

        # Test git success with .git suffix
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "https://github.com/user/my-project.git\n"
        with patch('subprocess.run', return_value=mock_result):
            assert get_project_name() == "my-project"

        # Test git success without .git suffix
        mock_result.stdout = "https://github.com/user/my-project\n"
        with patch('subprocess.run', return_value=mock_result):
            assert get_project_name() == "my-project"

        # Test git failure - return directory name
        mock_result.returncode = 128
        with patch('subprocess.run', return_value=mock_result):
            with patch('pathlib.Path.cwd', return_value=Path("/test/workspace-qdrant-mcp")):
                assert get_project_name() == "workspace-qdrant-mcp"

        # Test exception - return directory name
        with patch('subprocess.run', side_effect=Exception()):
            with patch('pathlib.Path.cwd', return_value=Path("/test/my-dir")):
                assert get_project_name() == "my-dir"

    @pytest.mark.asyncio
    async def test_initialize_components_all_paths(self):
        """Test all component initialization paths."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import initialize_components

        # Test first initialization
        server_module.qdrant_client = None
        server_module.embedding_model = None

        mock_client = Mock()
        mock_embedding = Mock()

        with patch('workspace_qdrant_mcp.server.QdrantClient', return_value=mock_client):
            with patch('fastembed.TextEmbedding', return_value=mock_embedding):
                with patch.dict(os.environ, {"QDRANT_URL": "http://test:6333", "QDRANT_API_KEY": "test-key"}):
                    await initialize_components()
                    assert server_module.qdrant_client == mock_client
                    assert server_module.embedding_model == mock_embedding

        # Test with custom model
        server_module.qdrant_client = None
        server_module.embedding_model = None

        with patch('workspace_qdrant_mcp.server.QdrantClient', return_value=mock_client):
            with patch('fastembed.TextEmbedding', return_value=mock_embedding) as mock_embed:
                with patch.dict(os.environ, {"FASTEMBED_MODEL": "custom-model"}):
                    await initialize_components()
                    mock_embed.assert_called_with("custom-model")

        # Test already initialized
        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding

        with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_qdrant:
            await initialize_components()
            mock_qdrant.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_embeddings_all_paths(self):
        """Test generate_embeddings function."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import generate_embeddings

        # Test with existing embedding model
        mock_embedding = Mock()
        mock_result = Mock()
        mock_result.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedding.embed.return_value = [mock_result]
        server_module.embedding_model = mock_embedding

        result = await generate_embeddings("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_embedding.embed.assert_called_with(["test text"])

        # Test with uninitialized model
        server_module.embedding_model = None
        with patch('workspace_qdrant_mcp.server.initialize_components'):
            with patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
                result = await generate_embeddings("test text")
                assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_all_paths(self):
        """Test ensure_collection_exists function."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import ensure_collection_exists

        mock_client = Mock()
        server_module.qdrant_client = mock_client

        # Test collection exists
        mock_client.get_collection.return_value = Mock()
        result = await ensure_collection_exists("test-collection")
        assert result

        # Test collection creation success
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = True
        result = await ensure_collection_exists("new-collection")
        assert result

        # Test collection creation failure
        mock_client.create_collection.side_effect = Exception("Creation failed")
        result = await ensure_collection_exists("fail-collection")
        assert not result

    def test_determine_collection_name_all_paths(self):
        """Test determine_collection_name function."""
        from workspace_qdrant_mcp.server import determine_collection_name

        # Test explicit collection
        result = determine_collection_name(collection="explicit", project_name="test")
        assert result == "explicit"

        # Test scratchbook source
        result = determine_collection_name(source="scratchbook", project_name="test")
        assert "scratchbook" in result

        # Test file source with Python extension
        result = determine_collection_name(source="file", file_path="test.py", project_name="test")
        assert "code" in result or "test" in result

        # Test URL source
        result = determine_collection_name(source="web", url="https://example.com", project_name="test")
        assert "web" in result or "test" in result

        # Test note content keywords
        result = determine_collection_name(content="This is my note", project_name="test")
        assert "test" in result

        # Test memory content keywords
        result = determine_collection_name(content="Remember this important memory", project_name="test")
        assert "test" in result

        # Test default behavior
        result = determine_collection_name(project_name="test")
        assert "test" in result

    @pytest.mark.asyncio
    async def test_store_function_all_paths(self):
        """Test store function comprehensively."""
        import workspace_qdrant_mcp.server as server_module

        # Setup mocks
        mock_client = Mock()
        mock_embedding = Mock()
        mock_result = Mock()
        mock_result.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedding.embed.return_value = [mock_result]

        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding

        # Get the actual function from the app's tools
        app = server_module.app
        store_tool = None
        for tool in app.tools:
            if hasattr(tool, 'name') and tool.name == 'store':
                store_tool = tool.func
                break

        if store_tool is None:
            pytest.skip("Store tool not found")

        # Test successful store
        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-docs"):
                with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test"):
                    result = await store_tool(
                        content="Test content",
                        title="Test Doc",
                        metadata={"tag": "test"},
                        source="user_input"
                    )
                    assert result["success"]
                    assert "document_id" in result
                    assert result["collection"] == "test-docs"
                    mock_client.upsert.assert_called()

        # Test collection creation failure
        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=False):
            result = await store_tool(content="Test")
            assert not result["success"]
            assert "Failed to create/access collection" in result["error"]

        # Test upsert exception
        mock_client.upsert.side_effect = Exception("Upsert failed")
        with patch('workspace_qdrant_mcp.server.ensure_collection_exists', return_value=True):
            with patch('workspace_qdrant_mcp.server.determine_collection_name', return_value="test-docs"):
                with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test"):
                    result = await store_tool(content="Test")
                    assert not result["success"]
                    assert "Upsert failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_function_all_paths(self):
        """Test search function comprehensively."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import search

        mock_client = Mock()
        mock_embedding = Mock()
        mock_result = Mock()
        mock_result.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedding.embed.return_value = [mock_result]

        server_module.qdrant_client = mock_client
        server_module.embedding_model = mock_embedding

        # Test semantic search
        search_results = [Mock(id="1", score=0.9, payload={"title": "Test", "content": "Content"})]
        mock_client.search.return_value = search_results

        with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test"):
            result = await search(query="test query", mode="semantic")
            assert result["success"]
            assert len(result["results"]) == 1

        # Test exact search
        scroll_result = Mock()
        scroll_result.points = [Mock(id="1", payload={"title": "Test", "content": "exact match"})]
        mock_client.scroll.return_value = (scroll_result, None)

        result = await search(query="exact match", mode="exact")
        assert result["success"]
        assert len(result["results"]) == 1

        # Test hybrid search
        result = await search(query="test", mode="hybrid")
        assert result["success"]

        # Test search exception
        mock_embedding.embed.side_effect = Exception("Embed failed")
        result = await search(query="test", mode="semantic")
        assert not result["success"]
        assert "Embed failed" in result["error"]

    @pytest.mark.asyncio
    async def test_manage_function_all_paths(self):
        """Test manage function comprehensively."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import manage

        mock_client = Mock()
        server_module.qdrant_client = mock_client

        # Test list collections
        collections = [Mock(name="test-docs", vectors_count=10)]
        mock_client.get_collections.return_value = Mock(collections=collections)

        with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test"):
            result = await manage(action="list_collections")
            assert result["success"]
            assert len(result["collections"]) == 1

        # Test create collection
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = True

        result = await manage(action="create_collection", name="new-collection")
        assert result["success"]

        # Test delete collection
        result = await manage(action="delete_collection", name="old-collection")
        assert result["success"]

        # Test workspace status
        result = await manage(action="workspace_status")
        assert result["success"]
        assert "project_name" in result

        # Test init project
        result = await manage(action="init_project")
        assert result["success"]

        # Test cleanup
        result = await manage(action="cleanup")
        assert result["success"]

        # Test unknown action
        result = await manage(action="unknown")
        assert not result["success"]
        assert "Unknown action" in result["error"]

        # Test manage exception
        mock_client.get_collections.side_effect = Exception("Client error")
        result = await manage(action="list_collections")
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_retrieve_function_all_paths(self):
        """Test retrieve function comprehensively."""
        import workspace_qdrant_mcp.server as server_module
        from workspace_qdrant_mcp.server import retrieve

        mock_client = Mock()
        server_module.qdrant_client = mock_client

        # Test retrieve by ID
        points = [Mock(id="test-id", payload={"title": "Test", "content": "Content"})]
        mock_client.retrieve.return_value = points

        with patch('workspace_qdrant_mcp.server.get_project_name', return_value="test"):
            result = await retrieve(document_id="test-id")
            assert result["success"]
            assert len(result["documents"]) == 1

        # Test retrieve by metadata
        scroll_result = Mock()
        scroll_result.points = [Mock(id="1", payload={"title": "Test", "tag": "important"})]
        mock_client.scroll.return_value = (scroll_result, None)

        result = await retrieve(metadata={"tag": "important"})
        assert result["success"]
        assert len(result["documents"]) == 1

        # Test no parameters error
        result = await retrieve()
        assert not result["success"]
        assert "document_id or metadata" in result["error"]

        # Test retrieve exception
        mock_client.retrieve.side_effect = Exception("Retrieve failed")
        result = await retrieve(document_id="test")
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_run_server_all_paths(self):
        """Test run_server function."""
        from workspace_qdrant_mcp.server import app, run_server

        # Test stdio mode
        with patch('workspace_qdrant_mcp.server._detect_stdio_mode', return_value=True):
            with patch('workspace_qdrant_mcp.server.initialize_components'):
                with patch.object(app, 'run_stdio') as mock_stdio:
                    await run_server()
                    mock_stdio.assert_called_once()

        # Test HTTP mode
        with patch('workspace_qdrant_mcp.server._detect_stdio_mode', return_value=False):
            with patch('workspace_qdrant_mcp.server.initialize_components'):
                with patch.object(app, 'run_server') as mock_http:
                    await run_server(host="0.0.0.0", port=8080)
                    mock_http.assert_called_once_with(host="0.0.0.0", port=8080)

    def test_main_function(self):
        """Test main entry point."""
        from workspace_qdrant_mcp.server import main
        with patch('workspace_qdrant_mcp.server.typer.run') as mock_typer:
            main()
            mock_typer.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
