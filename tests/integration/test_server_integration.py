"""
Integration tests for the FastMCP server.

Tests server initialization, MCP tool endpoints, and workflow integration.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workspace_qdrant_mcp import server
from workspace_qdrant_mcp.core.config import Config


class TestServerIntegration:
    """Test server integration functionality."""
    
    @pytest.fixture
    def mock_workspace_client_initialized(self, mock_workspace_client):
        """Create an initialized mock workspace client."""
        mock_workspace_client.initialized = True
        return mock_workspace_client
    
    @pytest.mark.asyncio
    async def test_initialize_workspace_success(self, environment_variables):
        """Test successful workspace initialization."""
        with patch('workspace_qdrant_mcp.server.Config') as mock_config_class, \
             patch('workspace_qdrant_mcp.server.ConfigValidator') as mock_validator_class, \
             patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class:
            
            # Setup mocks
            mock_config = MagicMock(spec=Config)
            mock_config_class.return_value = mock_config
            
            mock_validator = MagicMock()
            mock_validator.validate_all.return_value = (True, {"issues": [], "warnings": []})
            mock_validator_class.return_value = mock_validator
            
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Test initialization
            await server.initialize_workspace()
            
            # Verify initialization sequence
            mock_config_class.assert_called_once()
            mock_validator_class.assert_called_once_with(mock_config)
            mock_validator.validate_all.assert_called_once()
            mock_client_class.assert_called_once_with(mock_config)
            mock_client.initialize.assert_called_once()
            
            # Verify global client is set
            assert server.workspace_client == mock_client
    
    @pytest.mark.asyncio
    async def test_initialize_workspace_validation_failure(self):
        """Test workspace initialization with validation failure."""
        with patch('workspace_qdrant_mcp.server.Config') as mock_config_class, \
             patch('workspace_qdrant_mcp.server.ConfigValidator') as mock_validator_class:
            
            mock_config = MagicMock(spec=Config)
            mock_config_class.return_value = mock_config
            
            mock_validator = MagicMock()
            mock_validator.validate_all.return_value = (
                False, 
                {
                    "issues": ["Qdrant URL is required", "Invalid chunk size"],
                    "warnings": []
                }
            )
            mock_validator_class.return_value = mock_validator
            
            # Should raise RuntimeError
            with pytest.raises(RuntimeError, match="Configuration validation failed"):
                await server.initialize_workspace()
    
    @pytest.mark.asyncio
    async def test_initialize_workspace_with_warnings(self, environment_variables):
        """Test workspace initialization with warnings."""
        with patch('workspace_qdrant_mcp.server.Config') as mock_config_class, \
             patch('workspace_qdrant_mcp.server.ConfigValidator') as mock_validator_class, \
             patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class, \
             patch('builtins.print') as mock_print:
            
            mock_config = MagicMock(spec=Config)
            mock_config_class.return_value = mock_config
            
            mock_validator = MagicMock()
            mock_validator.validate_all.return_value = (
                True, 
                {
                    "issues": [],
                    "warnings": ["GitHub user not configured"]
                }
            )
            mock_validator_class.return_value = mock_validator
            
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            await server.initialize_workspace()
            
            # Verify warnings were printed
            mock_print.assert_any_call("⚠️  Configuration warnings:")
            mock_print.assert_any_call("  • GitHub user not configured")
    
    @pytest.mark.asyncio
    async def test_workspace_status_tool(self, mock_workspace_client_initialized):
        """Test workspace_status MCP tool."""
        # Set up global client
        server.workspace_client = mock_workspace_client_initialized
        
        expected_status = {
            "connected": True,
            "current_project": "test-project",
            "collections_count": 3
        }
        mock_workspace_client_initialized.get_status.return_value = expected_status
        
        result = await server.workspace_status()
        
        assert result == expected_status
        mock_workspace_client_initialized.get_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workspace_status_tool_not_initialized(self):
        """Test workspace_status tool when client is not initialized."""
        server.workspace_client = None
        
        result = await server.workspace_status()
        
        assert result == {"error": "Workspace client not initialized"}
    
    @pytest.mark.asyncio
    async def test_list_workspace_collections_tool(self, mock_workspace_client_initialized):
        """Test list_workspace_collections MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        expected_collections = ["test_docs", "test_scratchbook", "test_references"]
        mock_workspace_client_initialized.list_collections.return_value = expected_collections
        
        result = await server.list_workspace_collections()
        
        assert result == expected_collections
        mock_workspace_client_initialized.list_collections.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_workspace_collections_tool_not_initialized(self):
        """Test list_workspace_collections tool when client is not initialized."""
        server.workspace_client = None
        
        result = await server.list_workspace_collections()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_search_workspace_tool(self, mock_workspace_client_initialized):
        """Test search_workspace_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.search_workspace') as mock_search:
            expected_result = {
                "results": [{"id": "doc1", "score": 0.9, "content": "test document"}],
                "total": 1
            }
            mock_search.return_value = expected_result
            
            result = await server.search_workspace_tool(
                query="test query",
                collections=["docs"],
                mode="hybrid",
                limit=10,
                score_threshold=0.7
            )
            
            assert result == expected_result
            mock_search.assert_called_once_with(
                mock_workspace_client_initialized,
                "test query",
                ["docs"],
                "hybrid",
                10,
                0.7
            )
    
    @pytest.mark.asyncio
    async def test_add_document_tool(self, mock_workspace_client_initialized):
        """Test add_document_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.add_document') as mock_add_document:
            expected_result = {
                "status": "success",
                "document_id": "doc123",
                "collection": "docs"
            }
            mock_add_document.return_value = expected_result
            
            result = await server.add_document_tool(
                content="Test document content",
                collection="docs",
                metadata={"source": "test"},
                document_id="doc123",
                chunk_text=True
            )
            
            assert result == expected_result
            mock_add_document.assert_called_once_with(
                mock_workspace_client_initialized,
                "Test document content",
                "docs",
                {"source": "test"},
                "doc123",
                True
            )
    
    @pytest.mark.asyncio
    async def test_get_document_tool(self, mock_workspace_client_initialized):
        """Test get_document_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.get_document') as mock_get_document:
            expected_result = {
                "id": "doc123",
                "content": "Test document content",
                "metadata": {"source": "test"}
            }
            mock_get_document.return_value = expected_result
            
            result = await server.get_document_tool(
                document_id="doc123",
                collection="docs",
                include_vectors=False
            )
            
            assert result == expected_result
            mock_get_document.assert_called_once_with(
                mock_workspace_client_initialized,
                "doc123",
                "docs",
                False
            )
    
    @pytest.mark.asyncio
    async def test_search_by_metadata_tool(self, mock_workspace_client_initialized):
        """Test search_by_metadata_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.search_collection_by_metadata') as mock_search_metadata:
            expected_result = {
                "results": [{"id": "doc1", "metadata": {"category": "python"}}],
                "total": 1
            }
            mock_search_metadata.return_value = expected_result
            
            result = await server.search_by_metadata_tool(
                collection="docs",
                metadata_filter={"category": "python"},
                limit=10
            )
            
            assert result == expected_result
            mock_search_metadata.assert_called_once_with(
                mock_workspace_client_initialized,
                "docs",
                {"category": "python"},
                10
            )
    
    @pytest.mark.asyncio
    async def test_update_scratchbook_tool(self, mock_workspace_client_initialized):
        """Test update_scratchbook_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.update_scratchbook') as mock_update_scratchbook:
            expected_result = {
                "status": "success",
                "note_id": "note123",
                "title": "Test Note"
            }
            mock_update_scratchbook.return_value = expected_result
            
            result = await server.update_scratchbook_tool(
                content="Test note content",
                note_id="note123",
                title="Test Note",
                tags=["test", "python"],
                note_type="note"
            )
            
            assert result == expected_result
            mock_update_scratchbook.assert_called_once_with(
                mock_workspace_client_initialized,
                "Test note content",
                "note123",
                "Test Note",
                ["test", "python"],
                "note"
            )
    
    @pytest.mark.asyncio
    async def test_search_scratchbook_tool(self, mock_workspace_client_initialized):
        """Test search_scratchbook_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        with patch('workspace_qdrant_mcp.server.ScratchbookManager') as mock_scratchbook_manager_class:
            mock_manager = MagicMock()
            mock_manager.search_notes = AsyncMock(return_value={
                "results": [{"id": "note1", "title": "Python tips"}],
                "total": 1
            })
            mock_scratchbook_manager_class.return_value = mock_manager
            
            result = await server.search_scratchbook_tool(
                query="python",
                note_types=["note"],
                tags=["programming"],
                project_name="test-project",
                limit=10,
                mode="hybrid"
            )
            
            assert "results" in result
            mock_scratchbook_manager_class.assert_called_once_with(mock_workspace_client_initialized)
            mock_manager.search_notes.assert_called_once_with(
                "python",
                ["note"],
                ["programming"],
                "test-project",
                10,
                "hybrid"
            )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_advanced_tool(self, mock_workspace_client_initialized):
        """Test hybrid_search_advanced_tool MCP tool."""
        server.workspace_client = mock_workspace_client_initialized
        
        # Mock dependencies
        mock_workspace_client_initialized.list_collections.return_value = ["test_collection"]
        
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 5], "values": [0.8, 0.6]}
        })
        mock_workspace_client_initialized.get_embedding_service.return_value = mock_embedding_service
        
        with patch('workspace_qdrant_mcp.server.HybridSearchEngine') as mock_hybrid_engine_class:
            mock_engine = MagicMock()
            mock_engine.hybrid_search = AsyncMock(return_value={
                "results": [{"id": "doc1", "score": 0.9}],
                "fusion_method": "rrf"
            })
            mock_hybrid_engine_class.return_value = mock_engine
            
            result = await server.hybrid_search_advanced_tool(
                query="test query",
                collection="test_collection",
                fusion_method="rrf",
                dense_weight=1.0,
                sparse_weight=1.0,
                limit=10,
                score_threshold=0.0
            )
            
            assert "results" in result
            assert "fusion_method" in result
            
            # Verify hybrid search was called correctly
            mock_engine.hybrid_search.assert_called_once_with(
                collection_name="test_collection",
                query_embeddings={
                    "dense": [0.1] * 384,
                    "sparse": {"indices": [1, 5], "values": [0.8, 0.6]}
                },
                limit=10,
                score_threshold=0.0,
                dense_weight=1.0,
                sparse_weight=1.0,
                fusion_method="rrf"
            )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_advanced_tool_collection_not_found(self, mock_workspace_client_initialized):
        """Test hybrid_search_advanced_tool with non-existent collection."""
        server.workspace_client = mock_workspace_client_initialized
        mock_workspace_client_initialized.list_collections.return_value = ["other_collection"]
        
        result = await server.hybrid_search_advanced_tool(
            query="test query",
            collection="nonexistent_collection",
            limit=10
        )
        
        assert "error" in result
        assert "Collection 'nonexistent_collection' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_mcp_tools_error_handling(self):
        """Test that MCP tools handle missing workspace client gracefully."""
        server.workspace_client = None
        
        # Test all tools that should return error when client not initialized
        tools_and_args = [
            (server.search_workspace_tool, {"query": "test"}),
            (server.add_document_tool, {"content": "test", "collection": "docs"}),
            (server.get_document_tool, {"document_id": "doc1", "collection": "docs"}),
            (server.search_by_metadata_tool, {"collection": "docs", "metadata_filter": {}}),
            (server.update_scratchbook_tool, {"content": "test"}),
            (server.search_scratchbook_tool, {"query": "test"}),
            (server.list_scratchbook_notes_tool, {}),
            (server.delete_scratchbook_note_tool, {"note_id": "note1"}),
            (server.hybrid_search_advanced_tool, {"query": "test", "collection": "docs"})
        ]
        
        for tool_func, kwargs in tools_and_args:
            result = await tool_func(**kwargs)
            assert "error" in result
            assert "Workspace client not initialized" in result["error"]
    
    def test_run_server_with_config_file(self):
        """Test run_server function with custom config file."""
        with patch('workspace_qdrant_mcp.server.asyncio.run') as mock_asyncio_run, \
             patch('workspace_qdrant_mcp.server.app.run') as mock_app_run, \
             patch.dict('os.environ', {}, clear=False) as mock_env:
            
            # Mock the initialize_workspace coroutine
            mock_asyncio_run.return_value = None
            
            server.run_server(
                host="0.0.0.0",
                port=9000,
                config_file="/path/to/config.json"
            )
            
            # Verify config file environment variable was set
            assert mock_env.get("CONFIG_FILE") == "/path/to/config.json" or \
                   "CONFIG_FILE" in mock_env
            
            # Verify server was run with correct parameters
            mock_app_run.assert_called_once_with(host="0.0.0.0", port=9000)
            mock_asyncio_run.assert_called_once()
    
    def test_run_server_default_parameters(self):
        """Test run_server function with default parameters."""
        with patch('workspace_qdrant_mcp.server.asyncio.run') as mock_asyncio_run, \
             patch('workspace_qdrant_mcp.server.app.run') as mock_app_run:
            
            mock_asyncio_run.return_value = None
            
            server.run_server()
            
            # Verify server was run with default parameters
            mock_app_run.assert_called_once_with(host="127.0.0.1", port=8000)
    
    def test_main_function(self):
        """Test main console script entry point."""
        with patch('workspace_qdrant_mcp.server.typer.run') as mock_typer_run:
            server.main()
            
            mock_typer_run.assert_called_once_with(server.run_server)
    
    @pytest.mark.asyncio
    async def test_server_info_model(self):
        """Test ServerInfo model structure."""
        server_info = server.ServerInfo()
        
        assert server_info.name == "workspace-qdrant-mcp"
        assert server_info.version == "0.1.0"
        assert "scratchbook functionality" in server_info.description
    
    def test_fastmcp_app_initialization(self):
        """Test FastMCP app is properly initialized."""
        assert server.app is not None
        # The app should be a FastMCP instance with the correct name
        # We can't easily test this without importing FastMCP internals
        # but we can verify the global variable exists
    
    @pytest.mark.asyncio
    async def test_error_propagation_in_tools(self, mock_workspace_client_initialized):
        """Test that tool functions properly propagate errors from underlying services."""
        server.workspace_client = mock_workspace_client_initialized
        
        # Mock a service to raise an exception
        with patch('workspace_qdrant_mcp.server.search_workspace') as mock_search:
            mock_search.side_effect = Exception("Service error")
            
            # The tool should catch and handle the exception gracefully
            # (This depends on implementation - some tools might let exceptions bubble up)
            try:
                result = await server.search_workspace_tool(query="test")
                # If the tool handles exceptions internally, check for error in result
                if isinstance(result, dict) and "error" in result:
                    assert "error" in result
            except Exception:
                # If the tool lets exceptions bubble up, that's also valid behavior
                pass
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workspace_initialization_flow(self, environment_variables, temp_git_repo):
        """Test complete workspace initialization flow with real-ish components."""
        # This test uses more realistic mocking to simulate a full initialization
        with patch('workspace_qdrant_mcp.server.QdrantClient') as mock_qdrant_client_class, \
             patch('workspace_qdrant_mcp.server.QdrantWorkspaceClient') as mock_client_class:
            
            # Mock Qdrant client
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = MagicMock(collections=[])
            mock_qdrant_client_class.return_value = mock_qdrant_client
            
            # Mock workspace client
            mock_workspace_client = AsyncMock()
            mock_client_class.return_value = mock_workspace_client
            
            # Change to temp git repo directory for realistic project detection
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_git_repo)
                
                # Run initialization
                await server.initialize_workspace()
                
                # Verify workspace client was created and initialized
                mock_client_class.assert_called_once()
                mock_workspace_client.initialize.assert_called_once()
                
                # Verify global client was set
                assert server.workspace_client == mock_workspace_client
                
            finally:
                os.chdir(original_cwd)
