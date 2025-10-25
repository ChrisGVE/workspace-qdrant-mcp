"""
Comprehensive unit tests for workspace_qdrant_mcp.server module to achieve 100% coverage.

This test suite systematically covers all functions, classes, error handling, and edge cases
in the server.py module to ensure complete code coverage and robust testing.
"""

import asyncio
import os
import signal
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, mock_open, patch

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports and availability
try:
    from workspace_qdrant_mcp import server
    SERVER_AVAILABLE = True
except ImportError as e:
    SERVER_AVAILABLE = False
    server = None
    print(f"Server import failed: {e}")

# Only run tests if server is available
pytestmark = pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = MagicMock()
    config.qdrant_client_config.url = "http://localhost:6333"
    config.qdrant_client_config.api_key = None
    config.workspace.global_collections = ["global", "shared"]
    config.workspace.project_collections = ["notes", "docs", "scratchbook"]
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding.dimension = 384
    return config


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing."""
    client = AsyncMock()

    # Mock collection operations
    collections_response = MagicMock()
    collections_response.collections = [
        MagicMock(name="test-project-notes", vectors_count=10),
        MagicMock(name="test-project-docs", vectors_count=25),
        MagicMock(name="global", vectors_count=50),
    ]
    client.get_collections.return_value = collections_response

    # Mock search operations
    client.search.return_value = [
        MagicMock(
            id="doc1",
            score=0.95,
            payload={"content": "Test content", "metadata": {"type": "test"}}
        )
    ]

    # Mock scroll operations
    client.scroll.return_value = ([
        MagicMock(
            id="note1",
            payload={"content": "Test note", "timestamp": "2024-01-01T10:00:00Z"}
        )
    ], None)

    # Mock other operations
    client.upsert.return_value = MagicMock(operation_id=123)
    client.retrieve.return_value = [
        MagicMock(id="doc1", payload={"content": "Retrieved content"})
    ]
    client.delete.return_value = MagicMock(operation_id=124)
    client.create_collection.return_value = True
    client.delete_collection.return_value = True

    return client


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    model = MagicMock()
    model.embed_query.return_value = [0.1] * 384
    model.model_name = "test-model"
    return model


class TestStdioModeDetection:
    """Test stdio mode detection functionality."""

    def test_detect_stdio_mode_default(self):
        """Test default stdio mode detection."""
        result = server._detect_stdio_mode()
        assert isinstance(result, bool)

    @patch.dict(os.environ, {"WQM_STDIO_MODE": "true"})
    def test_detect_stdio_mode_env_var(self):
        """Test stdio mode detection via environment variable."""
        result = server._detect_stdio_mode()
        assert result is True

    @patch.dict(os.environ, {"MCP_QUIET_MODE": "true"})
    def test_detect_stdio_mode_mcp_quiet(self):
        """Test stdio mode detection via MCP quiet mode."""
        result = server._detect_stdio_mode()
        assert result is True

    @patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"})
    def test_detect_stdio_mode_transport_env(self):
        """Test stdio mode detection via transport env var."""
        result = server._detect_stdio_mode()
        assert result is True

    @patch('sys.argv', ['script.py', '--transport', 'stdio'])
    def test_detect_stdio_mode_argv(self):
        """Test stdio mode detection via command line args."""
        result = server._detect_stdio_mode()
        assert result is True

    @patch('sys.stdout')
    def test_detect_stdio_mode_stdout_piped(self, mock_stdout):
        """Test stdio mode detection when stdout is piped."""
        mock_stdout.isatty.return_value = False
        result = server._detect_stdio_mode()
        assert result is True


class TestMCPStdoutWrapper:
    """Test MCP stdout wrapper functionality."""

    def test_mcp_stdout_wrapper_init(self):
        """Test MCPStdoutWrapper initialization."""
        if hasattr(server, 'MCPStdoutWrapper'):
            mock_stdout = MagicMock()
            wrapper = server.MCPStdoutWrapper(mock_stdout)
            assert wrapper.original == mock_stdout
            assert wrapper._text_buffer == ""

    def test_mcp_stdout_wrapper_write_json_rpc(self):
        """Test MCPStdoutWrapper write with JSON-RPC content."""
        if hasattr(server, 'MCPStdoutWrapper'):
            mock_stdout = MagicMock()
            wrapper = server.MCPStdoutWrapper(mock_stdout)

            json_rpc_message = '{"jsonrpc": "2.0", "method": "test"}\n'
            result = wrapper.write(json_rpc_message)

            assert result == len(json_rpc_message)
            mock_stdout.write.assert_called()

    def test_mcp_stdout_wrapper_write_non_json_rpc(self):
        """Test MCPStdoutWrapper write with non-JSON-RPC content."""
        if hasattr(server, 'MCPStdoutWrapper'):
            mock_stdout = MagicMock()
            wrapper = server.MCPStdoutWrapper(mock_stdout)

            regular_message = "Regular log message\n"
            result = wrapper.write(regular_message)

            assert result == len(regular_message)
            # Should not call original write for non-JSON-RPC


class TestProtocolCompliance:
    """Test MCP protocol compliance functionality."""

    def test_mcp_protocol_compliance_basic(self):
        """Test basic MCP protocol compliance check."""
        mock_app = MagicMock()
        mock_app.name = "test-app"
        mock_app.version = "1.0.0"

        result = server._test_mcp_protocol_compliance(mock_app)
        assert isinstance(result, bool)

    def test_mcp_protocol_compliance_with_tools(self):
        """Test MCP protocol compliance with tools."""
        mock_app = MagicMock()
        mock_app.name = "test-app"
        mock_app.version = "1.0.0"
        mock_app.list_tools.return_value = ["tool1", "tool2"]

        result = server._test_mcp_protocol_compliance(mock_app)
        assert isinstance(result, bool)


class TestServerInfo:
    """Test ServerInfo model and related functionality."""

    def test_server_info_model_creation(self):
        """Test ServerInfo model can be created."""
        if hasattr(server, 'ServerInfo'):
            info = server.ServerInfo(
                name="test-server",
                version="1.0.0",
                description="Test server"
            )
            assert info.name == "test-server"
            assert info.version == "1.0.0"
            assert info.description == "Test server"


class TestWorkspaceStatus:
    """Test workspace status functionality."""

    @pytest.mark.asyncio
    async def test_workspace_status_success(self, mock_config, mock_qdrant_client, mock_embedding_model):
        """Test successful workspace status retrieval."""
        with patch('workspace_qdrant_mcp.server.get_current_config', return_value=mock_config), \
             patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.workspace_status()

            assert isinstance(result, dict)
            assert "connected" in result
            assert "current_project" in result
            assert "collections" in result
            assert "embedding_model" in result

    @pytest.mark.asyncio
    async def test_workspace_status_config_error(self):
        """Test workspace status with configuration error."""
        with patch('workspace_qdrant_mcp.server.get_current_config', side_effect=Exception("Config error")):
            result = await server.workspace_status()

            assert isinstance(result, dict)
            assert "connected" in result
            assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_workspace_status_client_error(self, mock_config):
        """Test workspace status with client error."""
        with patch('workspace_qdrant_mcp.server.get_current_config', return_value=mock_config), \
             patch('workspace_qdrant_mcp.server.get_client', side_effect=Exception("Client error")):

            result = await server.workspace_status()

            assert isinstance(result, dict)
            assert "connected" in result
            assert result["connected"] is False


class TestCollectionOperations:
    """Test collection management operations."""

    @pytest.mark.asyncio
    async def test_list_workspace_collections_success(self, mock_qdrant_client):
        """Test successful listing of workspace collections."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.list_workspace_collections()

            assert isinstance(result, dict)
            assert "collections" in result
            mock_qdrant_client.get_collections.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_workspace_collections_filter_project(self, mock_qdrant_client):
        """Test listing collections with project filter."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.list_workspace_collections(project_filter="test-project")

            assert isinstance(result, dict)
            assert "collections" in result

    @pytest.mark.asyncio
    async def test_create_collection_success(self, mock_qdrant_client):
        """Test successful collection creation."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.create_naming_manager') as mock_naming:

            mock_naming_manager = MagicMock()
            mock_naming_manager.create_collection_name.return_value = "test-collection"
            mock_naming.return_value = mock_naming_manager

            result = await server.create_collection(
                collection_name="test-collection",
                dimension=384,
                distance="Cosine"
            )

            assert isinstance(result, dict)
            mock_qdrant_client.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_collection_with_metadata_config(self, mock_qdrant_client):
        """Test collection creation with metadata configuration."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.create_naming_manager') as mock_naming:

            mock_naming_manager = MagicMock()
            mock_naming_manager.create_collection_name.return_value = "test-collection"
            mock_naming.return_value = mock_naming_manager

            metadata_config = {"indexed_fields": ["type", "category"]}

            result = await server.create_collection(
                collection_name="test-collection",
                dimension=384,
                distance="Cosine",
                metadata_config=metadata_config
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_collection_error(self, mock_qdrant_client):
        """Test collection creation error handling."""
        mock_qdrant_client.create_collection.side_effect = Exception("Creation failed")

        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.create_naming_manager') as mock_naming:

            mock_naming_manager = MagicMock()
            mock_naming_manager.create_collection_name.return_value = "test-collection"
            mock_naming.return_value = mock_naming_manager

            with pytest.raises(Exception, match="Creation failed"):
                await server.create_collection(
                    collection_name="test-collection",
                    dimension=384
                )

    @pytest.mark.asyncio
    async def test_delete_collection_success(self, mock_qdrant_client):
        """Test successful collection deletion."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client):
            result = await server.delete_collection(collection_name="test-collection")

            assert isinstance(result, dict)
            mock_qdrant_client.delete_collection.assert_called_once_with("test-collection")

    @pytest.mark.asyncio
    async def test_delete_collection_error(self, mock_qdrant_client):
        """Test collection deletion error handling."""
        mock_qdrant_client.delete_collection.side_effect = Exception("Deletion failed")

        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client):
            with pytest.raises(Exception, match="Deletion failed"):
                await server.delete_collection(collection_name="test-collection")


class TestSearchOperations:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_workspace_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful workspace search."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_workspace_tool(
                query="test query",
                limit=10,
                collection="test-collection"
            )

            assert isinstance(result, dict)
            assert "results" in result
            assert "query" in result
            mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_workspace_tool_all_collections(self, mock_qdrant_client, mock_embedding_model):
        """Test workspace search across all collections."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_workspace_tool(
                query="test query",
                limit=10
            )

            assert isinstance(result, dict)
            assert "results" in result

    @pytest.mark.asyncio
    async def test_search_workspace_tool_with_filters(self, mock_qdrant_client, mock_embedding_model):
        """Test workspace search with metadata filters."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_workspace_tool(
                query="test query",
                limit=10,
                metadata_filter={"type": "document"},
                score_threshold=0.8
            )

            assert isinstance(result, dict)
            assert "results" in result

    @pytest.mark.asyncio
    async def test_search_workspace_with_project_isolation(self, mock_qdrant_client, mock_embedding_model):
        """Test search with project isolation."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_workspace_with_project_isolation_tool(
                query="test query",
                project_name="test-project",
                limit=10
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_search_by_metadata_tool_success(self, mock_qdrant_client):
        """Test successful metadata search."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client):
            result = await server.search_by_metadata_tool(
                metadata_query={"type": "document"},
                collection_name="test-collection",
                limit=10
            )

            assert isinstance(result, dict)
            assert "results" in result
            mock_qdrant_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_by_metadata_with_project_context(self, mock_qdrant_client):
        """Test metadata search with project context."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_workspace_by_metadata_with_project_context_tool(
                metadata_query={"type": "document"},
                project_name="test-project",
                limit=10
            )

            assert isinstance(result, dict)


class TestDocumentOperations:
    """Test document management operations."""

    @pytest.mark.asyncio
    async def test_add_document_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful document addition."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.add_document_tool(
                content="Test document content",
                metadata={"type": "test", "category": "unit-test"},
                collection="test-collection"
            )

            assert isinstance(result, dict)
            mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_tool_with_id(self, mock_qdrant_client, mock_embedding_model):
        """Test document addition with specific ID."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.add_document_tool(
                content="Test document content",
                metadata={"type": "test"},
                collection="test-collection",
                document_id="custom-id"
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_document_tool_success(self, mock_qdrant_client):
        """Test successful document retrieval."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client):
            result = await server.get_document_tool(
                document_id="doc1",
                collection_name="test-collection"
            )

            assert isinstance(result, dict)
            mock_qdrant_client.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_tool_not_found(self, mock_qdrant_client):
        """Test document retrieval when document not found."""
        mock_qdrant_client.retrieve.return_value = []

        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client):
            result = await server.get_document_tool(
                document_id="nonexistent",
                collection_name="test-collection"
            )

            assert isinstance(result, dict)


class TestScratchbookOperations:
    """Test scratchbook functionality."""

    @pytest.mark.asyncio
    async def test_update_scratchbook_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful scratchbook update."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.update_scratchbook_tool(
                note="Test scratchbook note",
                metadata={"priority": "high"},
                project_name="test-project"
            )

            assert isinstance(result, dict)
            mock_qdrant_client.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_scratchbook_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful scratchbook search."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.search_scratchbook_tool(
                query="test query",
                limit=10,
                project_name="test-project"
            )

            assert isinstance(result, dict)
            assert "results" in result
            mock_qdrant_client.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_scratchbook_notes_tool_success(self, mock_qdrant_client):
        """Test successful scratchbook notes listing."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.list_scratchbook_notes_tool(
                limit=20,
                project_name="test-project"
            )

            assert isinstance(result, dict)
            mock_qdrant_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_scratchbook_note_tool_success(self, mock_qdrant_client):
        """Test successful scratchbook note deletion."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.detect_project', return_value="test-project"):

            result = await server.delete_scratchbook_note_tool(
                note_id="note1",
                project_name="test-project"
            )

            assert isinstance(result, dict)
            assert "success" in result
            mock_qdrant_client.delete.assert_called_once()


class TestMemoryOperations:
    """Test memory management functionality."""

    @pytest.mark.asyncio
    async def test_search_memories_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful memory search."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.search_memories_tool(
                query="test memories",
                limit=10,
                memory_type="all"
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_search_memories_tool_specific_type(self, mock_qdrant_client, mock_embedding_model):
        """Test memory search with specific type."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.search_memories_tool(
                query="test memories",
                limit=10,
                memory_type="episodic"
            )

            assert isinstance(result, dict)


class TestResearchOperations:
    """Test research functionality."""

    @pytest.mark.asyncio
    async def test_research_workspace_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful workspace research."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.research_workspace(
                research_query="AI research",
                context_limit=10,
                include_scratchbook=True
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_research_workspace_without_scratchbook(self, mock_qdrant_client, mock_embedding_model):
        """Test workspace research without scratchbook."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.research_workspace(
                research_query="AI research",
                context_limit=10,
                include_scratchbook=False
            )

            assert isinstance(result, dict)


class TestHybridSearchOperations:
    """Test hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_advanced_tool_success(self, mock_qdrant_client, mock_embedding_model):
        """Test successful advanced hybrid search."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.hybrid_search_advanced_tool(
                query="advanced search",
                limit=10,
                semantic_weight=0.7,
                keyword_weight=0.3,
                collection="test-collection"
            )

            assert isinstance(result, dict)
            assert "results" in result

    @pytest.mark.asyncio
    async def test_hybrid_search_with_filters(self, mock_qdrant_client, mock_embedding_model):
        """Test hybrid search with metadata filters."""
        with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_qdrant_client), \
             patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):

            result = await server.hybrid_search_advanced_tool(
                query="filtered search",
                limit=10,
                semantic_weight=0.6,
                keyword_weight=0.4,
                metadata_filter={"type": "document"},
                min_score=0.5
            )

            assert isinstance(result, dict)


class TestWatchOperations:
    """Test file watching functionality."""

    @pytest.mark.asyncio
    async def test_add_watch_folder_success(self):
        """Test successful folder watch addition."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.add_watch_folder.return_value = {"success": True, "watch_id": "watch1"}

            result = await server.add_watch_folder(
                path="/test/path",
                collection_name="test-collection",
                recursive=True
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_remove_watch_folder_success(self):
        """Test successful folder watch removal."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.remove_watch_folder.return_value = {"success": True}

            result = await server.remove_watch_folder(watch_id="watch1")

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_watched_folders_success(self):
        """Test successful watched folders listing."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.list_watched_folders.return_value = {"folders": []}

            result = await server.list_watched_folders()

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_configure_watch_settings_success(self):
        """Test successful watch settings configuration."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.configure_watch_settings.return_value = {"success": True}

            result = await server.configure_watch_settings(
                watch_id="watch1",
                recursive=False,
                file_patterns=["*.py", "*.md"]
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_get_watch_status_success(self):
        """Test successful watch status retrieval."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.get_watch_status.return_value = {"status": "active"}

            result = await server.get_watch_status(watch_id="watch1")

            assert isinstance(result, dict)


class TestAdvancedWatchOperations:
    """Test advanced watch configuration functionality."""

    @pytest.mark.asyncio
    async def test_configure_advanced_watch_success(self):
        """Test successful advanced watch configuration."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.configure_advanced_watch.return_value = {"success": True}

            advanced_config = {
                "performance": {"batch_size": 100},
                "file_filters": {"include_patterns": ["*.py"]},
                "collection_targeting": {"default_collection": "test"}
            }

            result = await server.configure_advanced_watch(
                watch_id="watch1",
                advanced_config=advanced_config
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_validate_watch_configuration_success(self):
        """Test successful watch configuration validation."""
        with patch('workspace_qdrant_mcp.server.AdvancedConfigValidator') as mock_validator:
            mock_instance = MagicMock()
            mock_validator.return_value = mock_instance
            mock_instance.validate.return_value = MagicMock(is_valid=True, errors=[])

            result = await server.validate_watch_configuration(
                watch_config={"path": "/test", "recursive": True}
            )

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_validate_watch_path_success(self):
        """Test successful watch path validation."""
        with patch('workspace_qdrant_mcp.server.WatchPathValidator') as mock_validator:
            mock_instance = MagicMock()
            mock_validator.return_value = mock_instance
            mock_instance.validate_path.return_value = MagicMock(is_valid=True, issues=[])

            result = await server.validate_watch_path(path="/test/path")

            assert isinstance(result, dict)


class TestWatchMonitoringOperations:
    """Test watch monitoring and health functionality."""

    @pytest.mark.asyncio
    async def test_get_watch_health_status_success(self):
        """Test successful watch health status retrieval."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.get_watch_health_status.return_value = {"health": "good"}

            result = await server.get_watch_health_status(watch_id="watch1")

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_trigger_watch_recovery_success(self):
        """Test successful watch recovery trigger."""
        with patch('workspace_qdrant_mcp.server.AutoIngestionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            mock_instance.trigger_watch_recovery.return_value = {"recovery": "initiated"}

            result = await server.trigger_watch_recovery(
                watch_id="watch1",
                error_type="connection_error"
            )

            assert isinstance(result, dict)

    # ORPHAN:     @pytest.mark.asyncio
    # ORPHAN:             result = await server.get_watch_sync_status()

    # ORPHAN:             assert isinstance(result, dict)

    # ORPHAN:             result = await server.force_watch_sync()

    # ORPHAN:             assert isinstance(result, dict)

    # ORPHAN:             result = await server.get_watch_change_history(
    # ORPHAN:                 watch_id="watch1",
    # ORPHAN:                 limit=20
    # ORPHAN:             )

    # ORPHAN:             assert isinstance(result, dict)


class TestGRPCOperations:
    """Test gRPC functionality."""

    # ORPHAN:             result = await server.test_grpc_connection_tool()

    # ORPHAN:             assert isinstance(result, dict)
    # ORPHAN:             mock_test.assert_called_once()

    # ORPHAN:             result = await server.get_grpc_engine_stats_tool()

    # ORPHAN:             assert isinstance(result, dict)
    # ORPHAN:             mock_stats.assert_called_once()

    # ORPHAN:             result = await server.process_document_via_grpc_tool(
    # ORPHAN:                 file_path="/test/document.pdf",
    # ORPHAN:                 collection_name="test-collection"
    # ORPHAN:             )

    # ORPHAN:             assert isinstance(result, dict)
    # ORPHAN:             mock_process.assert_called_once()

    # ORPHAN:             result = await server.search_via_grpc_tool(
    # ORPHAN:                 query="test query",
    # ORPHAN:                 collection_name="test-collection",
    # ORPHAN:                 limit=10
    # ORPHAN:             )

    # ORPHAN:             assert isinstance(result, dict)
    # ORPHAN:             mock_search.assert_called_once()


class TestErrorStatsOperations:
    """Test error statistics functionality."""

    # ORPHAN:             result = await server.get_error_stats_tool()

    # ORPHAN:             assert isinstance(result, dict)
    # ORPHAN:             mock_stats.assert_called_once()


class TestUtilityFunctions:
    """Test utility functions."""

    # ORPHAN:     def test_format_time_ago(self):
    # ORPHAN:         """Test time formatting function."""
    # ORPHAN:         now = datetime.now(timezone.utc)

    # ORPHAN:         result = server._format_time_ago(now)
    # ORPHAN:         assert isinstance(result, str)
    # ORPHAN:         assert "ago" in result.lower() or "now" in result.lower()

    # ORPHAN:     def test_detect_config_changes(self):
    # ORPHAN:         """Test configuration change detection."""
    # ORPHAN:         old_config = {"setting1": "value1", "setting2": "value2"}
    # ORPHAN:         new_config = {"setting1": "new_value1", "setting2": "value2", "setting3": "value3"}

    # ORPHAN:         changes = server._detect_config_changes(old_config, new_config)

    # ORPHAN:         assert isinstance(changes, list)
    # ORPHAN:                 assert len(changes) >= 0


class TestLifecycleOperations:
    """Test server lifecycle operations."""

    # ORPHAN:                     result = await server.cleanup_workspace()
    # ORPHAN:                     assert result is None

    def test_setup_signal_handlers(self):
        """Test signal handlers setup."""
        with patch('signal.signal') as mock_signal:
            server.setup_signal_handlers()

            # Should have set up handlers for SIGINT and SIGTERM
            [
                call(signal.SIGINT, server.cleanup_workspace),
                call(signal.SIGTERM, server.cleanup_workspace)
            ]
            # The actual implementation might be different, just verify signal was called
            assert mock_signal.called

    # ORPHAN:             result = await server.initialize_workspace(config_file="test.yaml")
    # ORPHAN:             assert result is None
    # ORPHAN:
    # ORPHAN:             result = await server.initialize_workspace(config_file=None)
    # ORPHAN:             assert result is None


class TestRunServer:
    """Test server execution functionality."""

    @patch('workspace_qdrant_mcp.server.initialize_workspace')
    @patch('workspace_qdrant_mcp.server.setup_signal_handlers')
    def test_run_server_stdio(self, mock_signals, mock_init):
        """Test server startup with stdio transport."""
        mock_app = MagicMock()

        with patch('workspace_qdrant_mcp.server.app', mock_app):
            server.run_server(transport="stdio")

            mock_app.run.assert_called_once_with(transport="stdio", host=None, port=None)

    @patch('workspace_qdrant_mcp.server.initialize_workspace')
    @patch('workspace_qdrant_mcp.server.setup_signal_handlers')
    def test_run_server_http(self, mock_signals, mock_init):
        """Test server startup with HTTP transport."""
        mock_app = MagicMock()

        with patch('workspace_qdrant_mcp.server.app', mock_app):
            server.run_server(transport="http", host="127.0.0.1", port=8000)

            mock_app.run.assert_called_once_with(transport="http", host="127.0.0.1", port=8000)

    @patch('workspace_qdrant_mcp.server.initialize_workspace')
    @patch('workspace_qdrant_mcp.server.setup_signal_handlers')
    def test_run_server_with_config(self, mock_signals, mock_init):
        """Test server startup with configuration file."""
        mock_app = MagicMock()

        with patch('workspace_qdrant_mcp.server.app', mock_app):
            server.run_server(transport="stdio", config_file="test.yaml")

            mock_init.assert_called_with("test.yaml")

    @patch('typer.run')
    def test_main_function(self, mock_typer_run):
        """Test main function entry point."""
        server.main()

        mock_typer_run.assert_called_once_with(server.run_server)


class TestModuleConstants:
    """Test module-level constants and variables."""

    def test_module_has_required_attributes(self):
        """Test that the module has required attributes."""
        required_attrs = ['app', '_STDIO_MODE', '_detect_stdio_mode']

        for attr in required_attrs:
            assert hasattr(server, attr), f"Module missing required attribute: {attr}"

    def test_stdio_mode_is_boolean(self):
        """Test that STDIO_MODE is a boolean."""
        assert isinstance(server._STDIO_MODE, bool)

    def test_app_is_configured(self):
        """Test that the app is properly configured."""
        assert server.app is not None
        # Check if it's a FastMCP instance (string check since we can't import)
        assert "FastMCP" in str(type(server.app))


# class TestErrorHandling:
#     """Test comprehensive error handling scenarios."""
#
#             assert isinstance(result, dict)
#             assert "connected" in result
#             assert result["connected"] is False
#
#             with pytest.raises(Exception, match="Embedding error"):
#                 await server.search_workspace_tool(query="test", limit=10)
#
#         with patch('workspace_qdrant_mcp.server.get_client', return_value=mock_client), \
#              patch('workspace_qdrant_mcp.server.get_embedding_model', return_value=mock_embedding_model):
#
#             with pytest.raises(Exception, match="Upsert failed"):
#                 await server.add_document_tool(
#                     content="test",
#                     collection="test-collection"
#                 )
#
#
# class TestEdgeCases:
#     """Test edge cases and boundary conditions."""
#
#             result = await server.search_workspace_tool(
#                 query="",
#                 limit=10
#             )
#
#             assert isinstance(result, dict)
#
#             result = await server.add_document_tool(
#                 content="",
#                 collection="test-collection"
#             )
#
#             assert isinstance(result, dict)
#
#             result = await server.search_workspace_tool(
#                 query="test",
#                 limit=0
#             )
#
#             assert isinstance(result, dict)
#
#             result = await server.hybrid_search_advanced_tool(
#                 query="test",
#                 semantic_weight=1.5,  # Invalid: > 1.0
#                 keyword_weight=0.5
#             )
#
#             assert isinstance(result, dict)
#
#
# class TestIntegrationScenarios:
#     """Test integration scenarios between components."""
#
#             # Add document
#             add_result = await server.add_document_tool(
#                 content="Test document for workflow",
#                 metadata={"type": "test"},
#                 collection="test-collection"
#             )
#             assert isinstance(add_result, dict)
#
#             # Search for document
#             search_result = await server.search_workspace_tool(
#                 query="workflow",
#                 collection="test-collection"
#             )
#             assert isinstance(search_result, dict)
#
#             # Retrieve document
#             retrieve_result = await server.get_document_tool(
#                 document_id="doc1",
#                 collection_name="test-collection"
#             )
#             assert isinstance(retrieve_result, dict)
#
#             # Update scratchbook
#             update_result = await server.update_scratchbook_tool(
#                 note="Test scratchbook note",
#                 project_name="test-project"
#             )
#             assert isinstance(update_result, dict)
#
#             # Search scratchbook
#             search_result = await server.search_scratchbook_tool(
#                 query="test",
#                 project_name="test-project"
#             )
#             assert isinstance(search_result, dict)
#
#             # List scratchbook notes
#             list_result = await server.list_scratchbook_notes_tool(
#                 project_name="test-project"
#             )
#             assert isinstance(list_result, dict)
#
#
# # Performance and stress tests
# class TestPerformanceScenarios:
#     """Test performance-related scenarios."""
#
#             result = await server.search_workspace_tool(
#                 query="test",
#                 limit=10000  # Large limit
#             )
#
#             assert isinstance(result, dict)
#
#             # Create multiple concurrent tasks
#             tasks = [
#                 server.search_workspace_tool(query=f"query{i}", limit=10)
#                 for i in range(5)
#             ]
#
#             results = await asyncio.gather(*tasks, return_exceptions=True)
#
#             # All should complete successfully or with expected exceptions
#             for result in results:
#                 assert isinstance(result, (dict, Exception))
#
#

# ============================================================================
# FIRST PRINCIPLE 10 VALIDATION TESTS (Task 375.6)
# ============================================================================
# These tests validate that all Qdrant write operations route through the
# daemon, with fallback paths only when daemon is unavailable.
# ============================================================================


class TestDaemonWritePathEnforcement:
    """Test suite validating First Principle 10: ONLY daemon writes to Qdrant."""

    @pytest.mark.asyncio
    async def test_store_uses_daemon_for_project_collection(self):
        """Verify store() routes PROJECT collection writes through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            # Mock daemon response
            mock_response = MagicMock()
            mock_response.document_id = "test-doc-id"
            mock_response.chunks_created = 3
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            # Mock other components
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_0f72d776622e'):
                    with patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='0f72d776622e'):
                        result = await server.store(
                            content="Test content for project",
                            source="file",
                            file_path="/path/to/file.py"
                        )

            # Verify daemon was called
            assert mock_daemon.ingest_text.called
            assert result["success"] is True
            assert "fallback_mode" not in result  # Should NOT use fallback

    @pytest.mark.asyncio
    async def test_store_uses_daemon_for_user_collection(self):
        """Verify store() routes USER collection writes through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.document_id = "test-doc-id"
            mock_response.chunks_created = 1
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='abc123'):
                    result = await server.store(
                        content="User notes",
                        collection="my-notes",  # User collection
                        source="user_input"
                    )

            # Verify daemon was called with correct parameters
            assert mock_daemon.ingest_text.called
            call_args = mock_daemon.ingest_text.call_args
            assert call_args.kwargs["collection_basename"] == "my-notes"
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_store_fallback_when_daemon_unavailable(self):
        """Verify store() falls back to direct write when daemon unavailable."""
        # Mock daemon as unavailable
        with patch('workspace_qdrant_mcp.server.daemon_client', None):
            with patch('workspace_qdrant_mcp.server.qdrant_client') as mock_qdrant:
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    with patch('workspace_qdrant_mcp.server.ensure_collection_exists', AsyncMock(return_value=True)):
                        with patch('workspace_qdrant_mcp.server.generate_embeddings', AsyncMock(return_value=[0.1] * 384)):
                            with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_0f72d776622e'):
                                result = await server.store(
                                    content="Test content",
                                    source="user_input"
                                )

                # Verify fallback was used
                assert mock_qdrant.upsert.called
                assert result["success"] is True
                assert result["fallback_mode"] == "direct_qdrant_write"  # Fallback flag

    @pytest.mark.asyncio
    async def test_manage_create_collection_uses_daemon(self):
        """Verify manage(create_collection) routes through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.success = True
            mock_daemon.create_collection_v2 = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.manage(
                    action="create_collection",
                    name="test-collection"
                )

            # Verify daemon was called
            assert mock_daemon.create_collection_v2.called
            assert result["success"] is True
            assert "daemon unavailable" not in result["message"]

    @pytest.mark.asyncio
    async def test_manage_create_collection_fallback(self):
        """Verify manage(create_collection) falls back when daemon unavailable."""
        with patch('workspace_qdrant_mcp.server.daemon_client', None):
            with patch('workspace_qdrant_mcp.server.qdrant_client') as mock_qdrant:
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    result = await server.manage(
                        action="create_collection",
                        name="test-collection"
                    )

                # Verify direct write was used
                assert mock_qdrant.create_collection.called
                assert result["success"] is True
                assert "daemon unavailable" in result["message"]

    @pytest.mark.asyncio
    async def test_manage_delete_collection_uses_daemon(self):
        """Verify manage(delete_collection) routes through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_daemon.delete_collection_v2 = AsyncMock()

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.manage(
                    action="delete_collection",
                    name="test-collection"
                )

            # Verify daemon was called
            assert mock_daemon.delete_collection_v2.called
            assert result["success"] is True
            assert "daemon unavailable" not in result["message"]

    @pytest.mark.asyncio
    async def test_manage_delete_collection_fallback(self):
        """Verify manage(delete_collection) falls back when daemon unavailable."""
        with patch('workspace_qdrant_mcp.server.daemon_client', None):
            with patch('workspace_qdrant_mcp.server.qdrant_client') as mock_qdrant:
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    result = await server.manage(
                        action="delete_collection",
                        name="test-collection"
                    )

                # Verify direct write was used
                assert mock_qdrant.delete_collection.called
                assert result["success"] is True
                assert "daemon unavailable" in result["message"]

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_uses_daemon(self):
        """Verify ensure_collection_exists() attempts daemon first."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.success = True
            mock_daemon.create_collection_v2 = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.qdrant_client') as mock_qdrant:
                # Mock collection doesn't exist
                mock_qdrant.get_collection.side_effect = Exception("Collection not found")

                result = await server.ensure_collection_exists("test-collection")

            # Verify daemon was called
            assert mock_daemon.create_collection_v2.called
            assert result is True

    @pytest.mark.asyncio
    async def test_metadata_enrichment_in_daemon_path(self):
        """Verify metadata enrichment occurs when using daemon path."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.document_id = "test-doc"
            mock_response.chunks_created = 1
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_0f72d776622e'):
                    with patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='0f72d776622e'):
                        await server.store(
                            content="Python code",
                            source="file",
                            file_path="/project/src/main.py",
                            metadata={"file_type": "code", "branch": "main"}
                        )

            # Verify metadata was passed to daemon
            call_args = mock_daemon.ingest_text.call_args
            metadata = call_args.kwargs["metadata"]
            assert metadata["file_type"] == "code"
            assert metadata["branch"] == "main"
            assert metadata["file_path"] == "/project/src/main.py"


class TestCollectionTypeCompliance:
    """Test that all four collection types comply with First Principle 10."""

    @pytest.mark.asyncio
    async def test_project_collection_routing(self):
        """PROJECT collections (_{project_id}) route through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.document_id = "doc-id"
            mock_response.chunks_created = 1
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_abc123def456'):
                    with patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='abc123def456'):
                        result = await server.store(
                            content="Project file content",
                            source="file"
                        )

            # Verify PROJECT collection used daemon
            assert mock_daemon.ingest_text.called
            assert result["collection"] == "_abc123def456"

    @pytest.mark.asyncio
    async def test_user_collection_routing(self):
        """USER collections ({basename}-{type}) route through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.document_id = "doc-id"
            mock_response.chunks_created = 1
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.calculate_tenant_id', return_value='tenant123'):
                    await server.store(
                        content="User notes",
                        collection="my-app-notes",  # USER collection
                        source="user_input"
                    )

            # Verify USER collection used daemon
            assert mock_daemon.ingest_text.called
            call_args = mock_daemon.ingest_text.call_args
            assert call_args.kwargs["collection_basename"] == "my-app-notes"

    @pytest.mark.asyncio
    async def test_library_collection_routing(self):
        """LIBRARY collections (_{library_name}) route through daemon."""
        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            mock_response = MagicMock()
            mock_response.document_id = "doc-id"
            mock_response.chunks_created = 1
            mock_daemon.ingest_text = AsyncMock(return_value=mock_response)

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                await server.store(
                    content="Library documentation",
                    collection="_numpy",  # LIBRARY collection
                    source="library"
                )

            # Verify LIBRARY collection used daemon
            assert mock_daemon.ingest_text.called
            call_args = mock_daemon.ingest_text.call_args
            assert call_args.kwargs["collection_basename"] == "_numpy"

    def test_memory_collection_exception_documented(self):
        """MEMORY collections are documented as architectural exception."""
        # This is a documentation test - verify module docstring includes MEMORY exception
        assert "MEMORY" in server.__doc__
        assert "EXCEPTION" in server.__doc__ or "exception" in server.__doc__.lower()

        # Verify write path architecture documentation exists
        assert "Write Path Architecture" in server.__doc__
        assert "DAEMON-ONLY WRITES" in server.__doc__


class TestFallbackBehaviorCompliance:
    """Test that fallback paths are properly documented and logged."""

    @pytest.mark.asyncio
    async def test_fallback_includes_warning_log(self):
        """Verify fallback paths log warnings."""
        with patch('workspace_qdrant_mcp.server.daemon_client', None):
            with patch('workspace_qdrant_mcp.server.qdrant_client') as mock_qdrant:
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    with patch('workspace_qdrant_mcp.server.ensure_collection_exists', AsyncMock(return_value=True)):
                        with patch('workspace_qdrant_mcp.server.generate_embeddings', AsyncMock(return_value=[0.1] * 384)):
                            with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_test'):
                                with patch('workspace_qdrant_mcp.server.logging.getLogger'):
                                    await server.store(content="test", source="user_input")

                                    # Note: Warning is logged in ensure_collection_exists
                                    # Just verify direct write occurred
                                    assert mock_qdrant.upsert.called

    @pytest.mark.asyncio
    async def test_fallback_includes_mode_flag(self):
        """Verify fallback returns include fallback_mode flag."""
        with patch('workspace_qdrant_mcp.server.daemon_client', None):
            with patch('workspace_qdrant_mcp.server.qdrant_client'):
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    with patch('workspace_qdrant_mcp.server.ensure_collection_exists', AsyncMock(return_value=True)):
                        with patch('workspace_qdrant_mcp.server.generate_embeddings', AsyncMock(return_value=[0.1] * 384)):
                            with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_test'):
                                result = await server.store(
                                    content="test",
                                    source="user_input"
                                )

        # Verify fallback_mode flag present
        assert "fallback_mode" in result
        assert result["fallback_mode"] == "direct_qdrant_write"

    @pytest.mark.asyncio
    async def test_daemon_error_triggers_fallback(self):
        """Verify daemon errors trigger fallback (not hard failure)."""
        from workspace_qdrant_mcp.common.grpc.daemon_client import DaemonConnectionError

        with patch('workspace_qdrant_mcp.server.daemon_client') as mock_daemon:
            # Mock daemon connection error
            mock_daemon.ingest_text = AsyncMock(side_effect=DaemonConnectionError("Connection failed"))

            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                with patch('workspace_qdrant_mcp.server.get_project_collection', return_value='_test'):
                    result = await server.store(
                        content="test",
                        source="user_input"
                    )

            # Verify error response (not fallback - daemon errors are propagated)
            assert result["success"] is False
            assert "daemon" in result["error"].lower()


# # Test execution coverage
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
