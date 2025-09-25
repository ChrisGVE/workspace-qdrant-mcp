"""
Comprehensive tests for the streamlined 4-tool MCP server.

This test suite validates all functionality of the new server that consolidates
36 tools into 4 comprehensive tools with content-based routing.
"""

import pytest
import sys
import uuid
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))


def get_function_from_tool(tool):
    """Extract actual function from FastMCP FunctionTool."""
    if hasattr(tool, 'fn') and tool.fn:
        return tool.fn
    elif hasattr(tool, 'func'):
        return tool.func
    elif hasattr(tool, '_func'):
        return tool._func
    elif hasattr(tool, '__call__'):
        return tool
    else:
        return tool


class TestServerImports:
    """Test that the server can be imported and has correct structure."""

    def test_fastmcp_available(self):
        """Test FastMCP is available."""
        from fastmcp import FastMCP
        assert FastMCP is not None

    def test_qdrant_client_available(self):
        """Test Qdrant client is available."""
        from qdrant_client import QdrantClient
        assert QdrantClient is not None

    def test_server_imports(self):
        """Test the server module imports."""
        from workspace_qdrant_mcp.server import app, run_server, main
        assert app is not None
        assert run_server is not None
        assert main is not None

    def test_server_has_four_tools(self):
        """Test that server has exactly 4 tools."""
        from workspace_qdrant_mcp.server import store, search, manage, retrieve

        # Check that all 4 tools exist
        assert store is not None
        assert search is not None
        assert manage is not None
        assert retrieve is not None


class TestUtilityFunctions:
    """Test utility functions that don't require external dependencies."""

    def test_get_project_name_fallback(self):
        """Test project name detection fallback."""
        from workspace_qdrant_mcp.server import get_project_name

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Git failed")
            with patch('pathlib.Path.cwd') as mock_cwd:
                mock_cwd.return_value.name = 'test-project'
                name = get_project_name()
                assert name == 'test-project'

    def test_get_project_name_git_success(self):
        """Test project name detection from git."""
        from workspace_qdrant_mcp.server import get_project_name

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = 'https://github.com/user/my-repo.git\n'

        with patch('subprocess.run', return_value=mock_result):
            name = get_project_name()
            assert name == 'my-repo'

    def test_determine_collection_name_scratchbook(self):
        """Test collection name determination for scratchbook."""
        from workspace_qdrant_mcp.server import determine_collection_name

        collection = determine_collection_name(
            content="test note",
            source="scratchbook",
            project_name="test-proj"
        )
        assert collection == "test-proj-scratchbook"

    def test_determine_collection_name_code_file(self):
        """Test collection name determination for code files."""
        from workspace_qdrant_mcp.server import determine_collection_name

        collection = determine_collection_name(
            content="def main(): pass",
            file_path="main.py",
            project_name="test-proj"
        )
        assert collection == "test-proj-code"

    def test_determine_collection_name_web_url(self):
        """Test collection name determination for web URLs."""
        from workspace_qdrant_mcp.server import determine_collection_name

        collection = determine_collection_name(
            content="webpage content",
            url="https://example.com",
            project_name="test-proj"
        )
        assert collection == "test-proj-web"

    def test_determine_collection_name_memory_content(self):
        """Test collection name determination for memory content."""
        from workspace_qdrant_mcp.server import determine_collection_name

        collection = determine_collection_name(
            content="remember this important context",
            project_name="test-proj"
        )
        assert collection == "test-proj-memory"

    def test_determine_collection_name_default(self):
        """Test collection name determination default case."""
        from workspace_qdrant_mcp.server import determine_collection_name

        collection = determine_collection_name(
            content="generic content",
            project_name="test-proj"
        )
        assert collection == "test-proj-documents"


class TestMockedServerOperations:
    """Test server operations with mocked dependencies."""

    @pytest.fixture
    def mock_components(self):
        """Setup mocked Qdrant client and embedding model."""
        mock_client = Mock()
        mock_embedding = Mock()

        # Mock embedding returns numpy-like array
        import numpy as np
        mock_embedding.embed.return_value = [np.array([0.1, 0.2, 0.3] * 128)]  # 384 dims

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_client), \
             patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
            yield mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_store_tool_basic(self, mock_components):
        """Test basic store functionality."""
        from workspace_qdrant_mcp.server import store

        mock_client, mock_embedding = mock_components
        mock_client.get_collection.side_effect = Exception("Collection doesn't exist")
        mock_client.create_collection.return_value = None
        mock_client.upsert.return_value = None

        store_func = get_function_from_tool(store)

        result = await store_func(
            content="Test content",
            title="Test Document",
            project_name="test-proj"
        )

        assert result["success"] is True
        assert "document_id" in result
        assert result["collection"] == "test-proj-documents"
        assert result["title"] == "Test Document"
        assert result["content_length"] == 12

    @pytest.mark.asyncio
    async def test_store_tool_content_based_routing(self, mock_components):
        """Test store tool content-based routing."""
        from workspace_qdrant_mcp.server import store

        mock_client, mock_embedding = mock_components
        mock_client.get_collection.side_effect = Exception("Collection doesn't exist")
        mock_client.create_collection.return_value = None
        mock_client.upsert.return_value = None

        # Test scratchbook routing
        store_func = get_function_from_tool(store)
        result = await store_func(
            content="Quick note",
            source="scratchbook",
            project_name="test-proj"
        )
        assert result["collection"] == "test-proj-scratchbook"

        # Test code file routing
        store_func = get_function_from_tool(store)
        result = await store_func(
            content="def hello(): pass",
            file_path="hello.py",
            project_name="test-proj"
        )
        assert result["collection"] == "test-proj-code"

        # Test web content routing
        store_func = get_function_from_tool(store)
        result = await store_func(
            content="Web page content",
            url="https://example.com",
            project_name="test-proj"
        )
        assert result["collection"] == "test-proj-web"

    @pytest.mark.asyncio
    async def test_search_tool_semantic_mode(self, mock_components):
        """Test search tool semantic mode."""
        from workspace_qdrant_mcp.server import search

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-proj-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock search results
        mock_hit = Mock()
        mock_hit.id = "test-id"
        mock_hit.score = 0.9
        mock_hit.payload = {
            "content": "test content",
            "title": "test title",
            "metadata": "test"
        }
        mock_client.search.return_value = [mock_hit]
        mock_client.get_collection.return_value = None  # Collection exists

        search_func = get_function_from_tool(search)
        result = await search_func(
            query="test query",
            mode="semantic",
            project_name="test-proj"
        )

        assert result["success"] is True
        assert result["mode"] == "semantic"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "test-id"
        assert result["results"][0]["score"] == 0.9

    @pytest.mark.asyncio
    async def test_search_tool_keyword_mode(self, mock_components):
        """Test search tool keyword mode."""
        from workspace_qdrant_mcp.server import search

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-proj-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock scroll results for keyword search
        mock_point = Mock()
        mock_point.id = "test-id"
        mock_point.payload = {
            "content": "this is a test document with test keyword",
            "title": "test title"
        }
        mock_client.scroll.return_value = ([mock_point], None)
        mock_client.get_collection.return_value = None  # Collection exists

        search_func = get_function_from_tool(search)
        result = await search_func(
            query="test",
            mode="keyword",
            project_name="test-proj"
        )

        assert result["success"] is True
        assert result["mode"] == "keyword"
        assert result["total_results"] == 1
        assert len(result["results"]) == 1

    @pytest.mark.asyncio
    async def test_manage_tool_list_collections(self, mock_components):
        """Test manage tool list collections action."""
        from workspace_qdrant_mcp.server import manage

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-collection"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.segments_count = 1
        mock_collection_info.status.value = "green"
        mock_collection_info.config.params.vectors.size = 384
        mock_collection_info.config.params.vectors.distance.value = "cosine"
        mock_client.get_collection.return_value = mock_collection_info

        manage_func = get_function_from_tool(manage)
        result = await manage_func(action="list_collections")

        assert result["success"] is True
        assert result["action"] == "list_collections"
        assert result["total_collections"] == 1
        assert len(result["collections"]) == 1
        assert result["collections"][0]["name"] == "test-collection"
        assert result["collections"][0]["points_count"] == 100

    @pytest.mark.asyncio
    async def test_manage_tool_create_collection(self, mock_components):
        """Test manage tool create collection action."""
        from workspace_qdrant_mcp.server import manage

        mock_client, mock_embedding = mock_components
        mock_client.create_collection.return_value = None

        manage_func = get_function_from_tool(manage)
        result = await manage_func(
            action="create_collection",
            name="new-collection"
        )

        assert result["success"] is True
        assert result["action"] == "create_collection"
        assert result["collection_name"] == "new-collection"
        assert "created successfully" in result["message"]

    @pytest.mark.asyncio
    async def test_manage_tool_workspace_status(self, mock_components):
        """Test manage tool workspace status action."""
        from workspace_qdrant_mcp.server import manage

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "workspace-qdrant-mcp-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock cluster info
        mock_cluster_info = Mock()
        mock_cluster_info.peer_id = "peer-123"
        mock_cluster_info.raft_info = {}
        mock_client.get_cluster_info.return_value = mock_cluster_info

        manage_func = get_function_from_tool(manage)
        result = await manage_func(action="workspace_status")

        assert result["success"] is True
        assert result["action"] == "workspace_status"
        assert result["qdrant_status"] == "connected"
        assert "current_project" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_retrieve_tool_by_id(self, mock_components):
        """Test retrieve tool by document ID."""
        from workspace_qdrant_mcp.server import retrieve

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-proj-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock retrieve result
        mock_point = Mock()
        mock_point.id = "test-document-id"
        mock_point.payload = {
            "content": "retrieved content",
            "title": "retrieved title",
            "metadata": "test"
        }
        mock_client.retrieve.return_value = [mock_point]

        retrieve_func = get_function_from_tool(retrieve)
        result = await retrieve_func(
            document_id="test-document-id",
            project_name="test-proj"
        )

        assert result["success"] is True
        assert result["total_results"] == 1
        assert result["query_type"] == "id_lookup"
        assert result["results"][0]["id"] == "test-document-id"
        assert result["results"][0]["content"] == "retrieved content"

    @pytest.mark.asyncio
    async def test_retrieve_tool_by_metadata(self, mock_components):
        """Test retrieve tool by metadata filters."""
        from workspace_qdrant_mcp.server import retrieve

        mock_client, mock_embedding = mock_components

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-proj-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Mock scroll result for metadata search
        mock_point = Mock()
        mock_point.id = "filtered-document"
        mock_point.payload = {
            "content": "filtered content",
            "title": "filtered title",
            "source": "test_source"
        }
        mock_client.scroll.return_value = ([mock_point], None)

        retrieve_func = get_function_from_tool(retrieve)
        result = await retrieve_func(
            metadata={"source": "test_source"},
            project_name="test-proj"
        )

        assert result["success"] is True
        assert result["total_results"] == 1
        assert result["query_type"] == "metadata_filter"
        assert result["results"][0]["id"] == "filtered-document"


class TestErrorHandling:
    """Test error handling in various scenarios."""

    @pytest.fixture
    def mock_components_with_errors(self):
        """Setup mocked components that raise errors."""
        mock_client = Mock()
        mock_embedding = Mock()

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_client), \
             patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
            yield mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_store_collection_creation_failure(self, mock_components_with_errors):
        """Test store tool when collection creation fails."""
        from workspace_qdrant_mcp.server import store

        mock_client, mock_embedding = mock_components_with_errors
        mock_client.get_collection.side_effect = Exception("Collection doesn't exist")
        mock_client.create_collection.side_effect = Exception("Creation failed")

        store_func = get_function_from_tool(store)
        result = await store_func(content="test", project_name="test-proj")

        assert result["success"] is False
        assert "Failed to create/access collection" in result["error"]

    @pytest.mark.asyncio
    async def test_store_upsert_failure(self, mock_components_with_errors):
        """Test store tool when upsert fails."""
        from workspace_qdrant_mcp.server import store

        mock_client, mock_embedding = mock_components_with_errors
        mock_client.get_collection.return_value = None  # Collection exists
        mock_client.upsert.side_effect = Exception("Upsert failed")

        # Setup embedding mock for this test
        import numpy as np
        mock_embedding.embed.return_value = [np.array([0.1, 0.2, 0.3] * 128)]

        store_func = get_function_from_tool(store)
        result = await store_func(content="test", project_name="test-proj")

        assert result["success"] is False
        assert "Failed to store document" in result["error"]

    @pytest.mark.asyncio
    async def test_search_no_collections(self, mock_components_with_errors):
        """Test search tool when no collections are found."""
        from workspace_qdrant_mcp.server import search

        mock_client, mock_embedding = mock_components_with_errors
        mock_collections_response = Mock()
        mock_collections_response.collections = []
        mock_client.get_collections.return_value = mock_collections_response

        search_func = get_function_from_tool(search)
        result = await search_func(query="test", project_name="test-proj")

        assert result["success"] is False
        assert "No collections found to search" in result["error"]

    @pytest.mark.asyncio
    async def test_search_collection_failure(self, mock_components_with_errors):
        """Test search tool when collection operations fail."""
        from workspace_qdrant_mcp.server import search

        mock_client, mock_embedding = mock_components_with_errors

        # Mock collections list
        mock_collection = Mock()
        mock_collection.name = "test-proj-documents"
        mock_collections_response = Mock()
        mock_collections_response.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections_response

        # Make search fail
        mock_client.get_collection.side_effect = Exception("Collection access failed")
        mock_client.create_collection.side_effect = Exception("Cannot create")

        search_func = get_function_from_tool(search)
        result = await search_func(query="test", project_name="test-proj")

        assert result["success"] is True  # Should continue with other collections
        assert result["total_results"] == 0

    @pytest.mark.asyncio
    async def test_manage_unknown_action(self, mock_components_with_errors):
        """Test manage tool with unknown action."""
        from workspace_qdrant_mcp.server import manage

        mock_client, mock_embedding = mock_components_with_errors

        manage_func = get_function_from_tool(manage)
        result = await manage_func(action="unknown_action")

        assert result["success"] is False
        assert "Unknown action: unknown_action" in result["error"]
        assert "available_actions" in result

    @pytest.mark.asyncio
    async def test_retrieve_no_parameters(self, mock_components_with_errors):
        """Test retrieve tool with missing parameters."""
        from workspace_qdrant_mcp.server import retrieve

        mock_client, mock_embedding = mock_components_with_errors

        retrieve_func = get_function_from_tool(retrieve)
        result = await retrieve_func()

        assert result["success"] is False
        assert "Either document_id or metadata filters must be provided" in result["error"]


class TestContentBasedRouting:
    """Test the content-based routing logic extensively."""

    def test_routing_scratchbook_source(self):
        """Test routing for scratchbook source."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="any content",
            source="scratchbook",
            project_name="myapp"
        )
        assert result == "myapp-scratchbook"

    def test_routing_note_content(self):
        """Test routing for note-like content."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="this is a note about the project",
            project_name="myapp"
        )
        assert result == "myapp-scratchbook"

    def test_routing_code_files(self):
        """Test routing for various code file extensions."""
        from workspace_qdrant_mcp.server import determine_collection_name

        code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go']

        for ext in code_extensions:
            result = determine_collection_name(
                content="code content",
                file_path=f"test{ext}",
                project_name="myapp"
            )
            assert result == "myapp-code", f"Failed for extension {ext}"

    def test_routing_document_files(self):
        """Test routing for document file extensions."""
        from workspace_qdrant_mcp.server import determine_collection_name

        doc_extensions = ['.md', '.txt', '.rst', '.doc', '.docx']

        for ext in doc_extensions:
            result = determine_collection_name(
                content="document content",
                file_path=f"test{ext}",
                project_name="myapp"
            )
            assert result == "myapp-docs", f"Failed for extension {ext}"

    def test_routing_other_files(self):
        """Test routing for other file types."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="file content",
            file_path="test.pdf",
            project_name="myapp"
        )
        assert result == "myapp-files"

    def test_routing_web_content(self):
        """Test routing for web URLs."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="web content",
            url="https://example.com/page",
            project_name="myapp"
        )
        assert result == "myapp-web"

    def test_routing_memory_content(self):
        """Test routing for memory-related content."""
        from workspace_qdrant_mcp.server import determine_collection_name

        memory_keywords = ['memory', 'remember', 'context']

        for keyword in memory_keywords:
            result = determine_collection_name(
                content=f"please {keyword} this information",
                project_name="myapp"
            )
            assert result == "myapp-memory", f"Failed for keyword {keyword}"

    def test_routing_explicit_collection(self):
        """Test that explicit collection overrides routing."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="any content",
            source="scratchbook",
            file_path="test.py",
            url="https://example.com",
            collection="explicit-collection",
            project_name="myapp"
        )
        assert result == "explicit-collection"

    def test_routing_default_case(self):
        """Test default routing fallback."""
        from workspace_qdrant_mcp.server import determine_collection_name

        result = determine_collection_name(
            content="generic content without special indicators",
            project_name="myapp"
        )
        assert result == "myapp-documents"


class TestAsyncComponentInitialization:
    """Test async component initialization."""

    @pytest.mark.asyncio
    async def test_initialize_components_first_time(self):
        """Test component initialization when components are None."""
        from workspace_qdrant_mcp.server import initialize_components

        with patch('workspace_qdrant_mcp.server.qdrant_client', None), \
             patch('workspace_qdrant_mcp.server.embedding_model', None):

            with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_qdrant, \
                 patch('fastembed.TextEmbedding') as mock_embed:

                mock_client_instance = AsyncMock()
                mock_qdrant.return_value = mock_client_instance

                mock_embed_instance = Mock()
                mock_embed.return_value = mock_embed_instance

                await initialize_components()

                # Verify QdrantClient was called with correct parameters
                mock_qdrant.assert_called_once()
                call_kwargs = mock_qdrant.call_args[1]
                assert 'url' in call_kwargs
                assert 'timeout' in call_kwargs
                assert call_kwargs['timeout'] == 60

    @pytest.mark.asyncio
    async def test_initialize_components_already_initialized(self):
        """Test component initialization when components already exist."""
        from workspace_qdrant_mcp.server import initialize_components

        mock_existing_client = AsyncMock()
        mock_existing_embedding = Mock()

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_existing_client), \
             patch('workspace_qdrant_mcp.server.embedding_model', mock_existing_embedding):

            with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_qdrant, \
                 patch('fastembed.TextEmbedding') as mock_embed:

                await initialize_components()

                # Verify no new instances were created
                mock_qdrant.assert_not_called()
                mock_embed.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])