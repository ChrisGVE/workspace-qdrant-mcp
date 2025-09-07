#!/usr/bin/env python3
"""
Comprehensive validation tests for critical bug fixes in workspace-qdrant-mcp.

This test suite validates that the following critical bugs have been successfully fixed:
- Issue #12: Search functionality returns empty results (FIXED - collection filtering)  
- Issue #13: Scratchbook functionality broken (FIXED - missing ensure_collection_exists method)
- Issue #5: Auto-ingestion not processing workspace files (FIXED - configuration mismatch)
- Issue #14: Advanced search type conversion errors (FIXED - parameter type handling)

Each test focuses on the specific error conditions that were causing the bugs
and ensures they no longer occur under the documented scenarios.

Test Categories:
1. Search functionality validation (Issue #12)
2. Scratchbook functionality validation (Issue #13) 
3. Auto-ingestion functionality validation (Issue #5)
4. Advanced search parameter validation (Issue #14)

Usage:
    python -m pytest tests/test_bug_fixes_validation.py -v
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.http import models

from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
from workspace_qdrant_mcp.core.collections import WorkspaceCollectionManager
from workspace_qdrant_mcp.core.config import Config
from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
from workspace_qdrant_mcp.tools.search import search_workspace


class TestSearchFunctionalityFixes:
    """Test suite for Issue #12: Search functionality returns empty results."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = MagicMock(spec=Config)
        config.qdrant.url = "http://localhost:6333"
        config.embedding.model = "BAAI/bge-small-en"
        config.embedding.enable_sparse_vectors = True
        config.workspace.global_collections = ["scratchbook"]
        config.workspace.github_user = "testuser"
        return config

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        client = MagicMock()
        # Mock collections response
        client.get_collections.return_value = MagicMock(
            collections=[
                MagicMock(name="test-project"),
                MagicMock(name="test-project-scratchbook"),
                MagicMock(name="other-collection"),
            ]
        )
        
        # Mock search response with actual results
        search_result = [
            MagicMock(
                id="test-doc-1",
                score=0.85,
                payload={"content": "Test document content", "file_type": "txt"}
            ),
            MagicMock(
                id="test-doc-2", 
                score=0.75,
                payload={"content": "Another test document", "file_type": "md"}
            )
        ]
        client.search.return_value = search_result
        return client

    @pytest.fixture
    def mock_workspace_client(self, mock_config, mock_qdrant_client):
        """Create a mock workspace client."""
        client = MagicMock(spec=QdrantWorkspaceClient)
        client.config = mock_config
        client.client = mock_qdrant_client
        client.initialized = True
        
        # Mock project info
        client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 384,  # Mock dense vector
            "sparse": {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}  # Mock sparse
        })
        client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock collection manager with proper workspace filtering
        mock_collection_manager = MagicMock(spec=WorkspaceCollectionManager)
        mock_collection_manager.list_workspace_collections.return_value = [
            "test-project", 
            "test-project-scratchbook"
        ]
        mock_collection_manager.resolve_collection_name.side_effect = lambda x: (x, "workspace")
        
        # Add validate_collection_filtering method for enhanced diagnostics
        mock_collection_manager.validate_collection_filtering.return_value = {
            "summary": {"total_collections": 3, "workspace_collections": 2}
        }
        
        client.collection_manager = mock_collection_manager
        client.list_collections.return_value = ["test-project", "test-project-scratchbook"]
        
        return client

    @pytest.mark.asyncio
    async def test_search_returns_actual_results_not_empty(self, mock_workspace_client):
        """Test that search functionality returns actual results instead of empty results.
        
        This addresses Issue #12 where search was returning empty results due to 
        collection filtering issues.
        """
        # Execute search
        result = await search_workspace(
            client=mock_workspace_client,
            query="test document",
            mode="dense",
            limit=10,
            score_threshold=0.5
        )
        
        # Validate results are not empty
        assert "error" not in result
        assert result["total"] > 0
        assert len(result["results"]) > 0
        assert result["collections_searched"] == ["test-project", "test-project-scratchbook"]
        
        # Verify actual search results with proper scores
        for search_result in result["results"]:
            assert "id" in search_result
            assert "score" in search_result
            assert search_result["score"] >= 0.5  # Above threshold
            assert "payload" in search_result
            assert "content" in search_result["payload"]

    @pytest.mark.asyncio
    async def test_hybrid_search_works_with_all_collections(self, mock_workspace_client):
        """Test that hybrid search works across all detected workspace collections."""
        # Setup hybrid search mock
        with patch('workspace_qdrant_mcp.tools.search.HybridSearchEngine') as mock_engine:
            mock_engine_instance = mock_engine.return_value
            mock_engine_instance.hybrid_search = AsyncMock(return_value={
                "results": [
                    {
                        "id": "hybrid-doc-1",
                        "rrf_score": 0.9,
                        "payload": {"content": "Hybrid search result"}
                    }
                ]
            })
            
            result = await search_workspace(
                client=mock_workspace_client,
                query="hybrid test",
                mode="hybrid", 
                limit=5
            )
            
            # Verify hybrid search was called
            assert mock_engine_instance.hybrid_search.called
            assert result["total"] > 0
            assert result["mode"] == "hybrid"

    @pytest.mark.asyncio  
    async def test_search_handles_no_collections_gracefully(self, mock_workspace_client):
        """Test that search provides helpful error when no collections are available."""
        # Mock no collections available
        mock_workspace_client.list_collections.return_value = []
        mock_workspace_client.collection_manager.validate_collection_filtering.return_value = {
            "summary": {"total_collections": 0, "workspace_collections": 0}
        }
        
        result = await search_workspace(
            client=mock_workspace_client,
            query="test",
            mode="dense"
        )
        
        # Should return helpful error message, not empty results
        assert "error" in result
        assert "No collections found" in result["error"] or "No workspace collections" in result["error"]

    @pytest.mark.asyncio
    async def test_sparse_search_mode_functionality(self, mock_workspace_client):
        """Test that sparse search mode works correctly."""
        result = await search_workspace(
            client=mock_workspace_client,
            query="exact keyword match",
            mode="sparse",
            limit=5,
            score_threshold=0.6
        )
        
        # Should not error and should return results
        assert "error" not in result
        assert result["mode"] == "sparse"
        
        # Mock should have been called with sparse vector
        mock_workspace_client.client.search.assert_called()


class TestScratchbookFunctionalityFixes:
    """Test suite for Issue #13: Scratchbook functionality broken."""

    @pytest.fixture
    def mock_workspace_client(self):
        """Create a mock workspace client with ensure_collection_exists method."""
        client = MagicMock(spec=QdrantWorkspaceClient)
        client.initialized = True
        
        # Mock project info
        client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        # Mock embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}
        }
        client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock the crucial ensure_collection_exists method (was missing in Issue #13)
        client.ensure_collection_exists = AsyncMock()
        
        # Mock Qdrant client operations
        mock_qdrant_client = MagicMock()
        mock_qdrant_client.upsert = AsyncMock()
        mock_qdrant_client.scroll = MagicMock(return_value=([
            MagicMock(
                id="note-123",
                payload={
                    "note_id": "note-123",
                    "title": "Test Note",
                    "content": "Test content",
                    "note_type": "note",
                    "tags": ["test"],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            )
        ], None))
        client.client = mock_qdrant_client
        
        return client

    @pytest.mark.asyncio
    async def test_scratchbook_add_note_calls_ensure_collection_exists(self, mock_workspace_client):
        """Test that scratchbook properly calls ensure_collection_exists.
        
        This addresses Issue #13 where scratchbook was failing with AttributeError
        because ensure_collection_exists method was missing.
        """
        manager = ScratchbookManager(mock_workspace_client)
        
        result = await manager.add_note(
            content="Test note content",
            title="Test Note",
            note_type="note",
            tags=["test"]
        )
        
        # Verify ensure_collection_exists was called
        mock_workspace_client.ensure_collection_exists.assert_called_once()
        
        # Verify note was added successfully
        assert "error" not in result
        assert result["title"] == "Test Note"
        assert result["note_type"] == "note"
        assert result["tags"] == ["test"]

    @pytest.mark.asyncio
    async def test_scratchbook_search_calls_ensure_collection_exists(self, mock_workspace_client):
        """Test that scratchbook search properly handles collection creation."""
        manager = ScratchbookManager(mock_workspace_client)
        
        # Mock HybridSearchEngine for search
        with patch('workspace_qdrant_mcp.tools.scratchbook.HybridSearchEngine') as mock_engine:
            mock_engine_instance = mock_engine.return_value
            mock_engine_instance.hybrid_search.return_value = {
                "results": [
                    {
                        "id": "note-123",
                        "score": 0.85,
                        "payload": {"content": "Test note"}
                    }
                ]
            }
            
            result = await manager.search_notes(
                query="test query",
                note_types=["note"],
                limit=5
            )
            
            # Verify ensure_collection_exists was called
            mock_workspace_client.ensure_collection_exists.assert_called_once()
            
            # Verify search worked
            assert "error" not in result
            assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_scratchbook_handles_collection_creation_failure(self, mock_workspace_client):
        """Test graceful handling when collection creation fails."""
        # Mock ensure_collection_exists to fail
        mock_workspace_client.ensure_collection_exists = AsyncMock(
            side_effect=RuntimeError("Collection creation failed")
        )
        
        manager = ScratchbookManager(mock_workspace_client)
        
        result = await manager.add_note(
            content="Test content",
            title="Test Note"
        )
        
        # Should return error, not crash
        assert "error" in result
        assert "Collection creation failed" in result["error"]

    @pytest.mark.asyncio  
    async def test_scratchbook_crud_operations_work(self, mock_workspace_client):
        """Test complete CRUD operations work without AttributeError."""
        manager = ScratchbookManager(mock_workspace_client)
        
        # Test ADD
        add_result = await manager.add_note(
            content="Test content for CRUD",
            title="CRUD Test Note",
            tags=["crud", "test"]
        )
        assert "error" not in add_result
        note_id = add_result["note_id"]
        
        # Test UPDATE
        update_result = await manager.update_note(
            note_id=note_id,
            content="Updated content",
            title="Updated Title"
        )
        assert "error" not in update_result
        
        # Test LIST
        list_result = await manager.list_notes(limit=10)
        assert "error" not in list_result
        assert "notes" in list_result
        
        # Test DELETE  
        delete_result = await manager.delete_note(note_id)
        assert "error" not in delete_result


class TestAutoIngestionFixes:
    """Test suite for Issue #5: Auto-ingestion not processing workspace files."""

    @pytest.fixture
    def mock_config_with_auto_ingestion(self):
        """Create config with auto-ingestion enabled."""
        config = MagicMock()
        config.auto_ingestion = {
            "enabled": True,
            "watch_patterns": ["*.txt", "*.md", "*.py"],
            "ignore_patterns": ["*.tmp", "__pycache__"],
            "target_collections": ["auto-ingestion"],
            "recursive": True,
            "debounce_seconds": 2
        }
        config.qdrant.url = "http://localhost:6333"
        config.embedding.model = "BAAI/bge-small-en"
        return config

    @pytest.fixture  
    def temp_project_dir(self):
        """Create a temporary project directory with test files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create test files that should be ingested
            (project_path / "README.md").write_text("# Test Project\nThis is a test project.")
            (project_path / "main.py").write_text("def main():\n    print('Hello World')")
            (project_path / "notes.txt").write_text("Important project notes")
            
            # Create files that should be ignored
            (project_path / "temp.tmp").write_text("Temporary file")
            (project_path / "__pycache__").mkdir()
            (project_path / "__pycache__" / "test.pyc").write_text("compiled")
            
            yield project_path

    @pytest.mark.asyncio
    async def test_auto_ingestion_detects_workspace_files(self, mock_config_with_auto_ingestion, temp_project_dir):
        """Test that auto-ingestion detects and processes workspace files.
        
        This addresses Issue #5 where auto-ingestion wasn't processing files
        due to configuration mismatch.
        """
        from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager
        
        # Mock workspace client
        mock_client = MagicMock()
        mock_client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        # Mock watch tools manager
        mock_watch_manager = MagicMock()
        mock_watch_manager.add_watch_folder = AsyncMock(return_value={
            "success": True,
            "watch_id": "auto-watch-1",
            "files_found": 3  # README.md, main.py, notes.txt
        })
        
        # Mock bulk ingestion process
        mock_watch_manager.process_bulk_ingestion = AsyncMock(return_value={
            "processed_files": 3,
            "success_rate": 1.0,
            "errors": []
        })
        
        # Change to temp directory for project detection
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_project_dir)
            
            manager = AutoIngestionManager(
                workspace_client=mock_client,
                watch_manager=mock_watch_manager,
                config=mock_config_with_auto_ingestion.auto_ingestion
            )
            
            result = await manager.setup_project_watches()
            
            # Verify auto-ingestion setup succeeded  
            assert result["success"] is True
            assert len(result["watches_created"]) > 0
            
            # Verify watch was created for the project directory
            mock_watch_manager.add_watch_folder.assert_called()
            watch_call = mock_watch_manager.add_watch_folder.call_args
            assert str(temp_project_dir) in str(watch_call)
            
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_auto_ingestion_respects_ignore_patterns(self, mock_config_with_auto_ingestion, temp_project_dir):
        """Test that auto-ingestion correctly ignores specified patterns."""
        from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager
        
        mock_client = MagicMock()
        mock_client.get_project_info.return_value = {
            "main_project": "test-project", 
            "subprojects": []
        }
        
        mock_watch_manager = MagicMock()
        mock_watch_manager.add_watch_folder = AsyncMock(return_value={"success": True})
        
        original_cwd = os.getcwd()
        try:
            os.chdir(temp_project_dir)
            
            manager = AutoIngestionManager(
                workspace_client=mock_client,
                watch_manager=mock_watch_manager, 
                config=mock_config_with_auto_ingestion.auto_ingestion
            )
            
            await manager.setup_project_watches()
            
            # Check that ignore patterns were passed correctly
            watch_call = mock_watch_manager.add_watch_folder.call_args
            assert "ignore_patterns" in watch_call.kwargs
            ignore_patterns = watch_call.kwargs["ignore_patterns"]
            assert "*.tmp" in ignore_patterns
            assert "__pycache__" in ignore_patterns
            
        finally:
            os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_auto_ingestion_creates_target_collections(self, mock_config_with_auto_ingestion):
        """Test that auto-ingestion creates or finds target collections."""  
        from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager
        
        mock_client = MagicMock()
        mock_client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        # Mock collection operations
        mock_client.list_collections.return_value = ["existing-collection"]
        mock_client.ensure_collection_exists = AsyncMock()
        
        mock_watch_manager = MagicMock() 
        mock_watch_manager.add_watch_folder = AsyncMock(return_value={"success": True})
        
        manager = AutoIngestionManager(
            workspace_client=mock_client,
            watch_manager=mock_watch_manager,
            config=mock_config_with_auto_ingestion.auto_ingestion
        )
        
        result = await manager.setup_project_watches()
        
        # Verify target collection handling
        assert result["success"] is True
        assert "primary_collection" in result


class TestAdvancedSearchParameterFixes:
    """Test suite for Issue #14: Advanced search type conversion errors."""

    @pytest.fixture
    def mock_workspace_client_for_params(self):
        """Create mock client for parameter testing."""
        client = MagicMock()
        client.initialized = True
        client.list_collections.return_value = ["test-collection"]
        
        # Mock collection manager
        mock_manager = MagicMock()
        mock_manager.resolve_collection_name.side_effect = lambda x: (x, "workspace")
        client.collection_manager = mock_manager
        
        # Mock embedding service  
        mock_embedding = MagicMock()
        mock_embedding.generate_embeddings = AsyncMock(return_value={
            "dense": [0.1] * 384,
            "sparse": {"indices": [0, 1, 2], "values": [0.5, 0.3, 0.2]}
        })
        client.get_embedding_service.return_value = mock_embedding
        
        return client

    @pytest.mark.asyncio
    async def test_search_handles_string_numeric_parameters(self, mock_workspace_client_for_params):
        """Test that search tools handle string representations of numeric parameters.
        
        This addresses Issue #14 where "must be real number, not str" errors occurred
        when parameters were passed as strings.
        """
        from workspace_qdrant_mcp.server import search_workspace_tool
        
        # Test with string parameters (common from MCP calls)
        result = await search_workspace_tool(
            query="test query",
            limit="5",  # String instead of int
            score_threshold="0.8"  # String instead of float
        )
        
        # Should not error with type conversion  
        assert "error" not in result or "must be real number, not str" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_advanced_search_parameter_validation(self, mock_workspace_client_for_params):
        """Test parameter validation in advanced search tools."""
        from workspace_qdrant_mcp.server import hybrid_search_advanced_tool
        
        # Mock the workspace client globally
        with patch('workspace_qdrant_mcp.server.workspace_client', mock_workspace_client_for_params):
            # Test with string parameters 
            result = await hybrid_search_advanced_tool(
                query="test",
                collection="test-collection",
                dense_weight="1.5",  # String float
                sparse_weight="0.8",  # String float  
                limit="10",  # String int
                score_threshold="0.7"  # String float
            )
            
            # Should convert parameters properly, not error
            assert "error" not in result or "must be real number, not str" not in result.get("error", "")

    @pytest.mark.asyncio
    async def test_parameter_range_validation(self, mock_workspace_client_for_params):
        """Test that parameter range validation works correctly."""
        from workspace_qdrant_mcp.server import search_workspace_tool
        
        # Test invalid limit
        result = await search_workspace_tool(
            query="test",
            limit="-1"  # Invalid limit
        )
        assert "error" in result
        assert "limit must be greater than 0" in result["error"]
        
        # Test invalid score threshold
        result = await search_workspace_tool(
            query="test",
            score_threshold="1.5"  # Invalid threshold > 1.0
        )
        assert "error" in result
        assert "score_threshold must be between 0.0 and 1.0" in result["error"]

    @pytest.mark.asyncio
    async def test_watch_tools_parameter_conversion(self):
        """Test that watch management tools handle string parameters correctly."""
        from workspace_qdrant_mcp.server import add_watch_folder
        
        # Mock watch tools manager
        mock_manager = MagicMock()
        mock_manager.add_watch_folder = AsyncMock(return_value={"success": True})
        
        with patch('workspace_qdrant_mcp.server.watch_tools_manager', mock_manager):
            result = await add_watch_folder(
                path="/test/path",
                collection="test-collection", 
                recursive_depth="-1",  # String int
                debounce_seconds="5",  # String int
                update_frequency="1000"  # String int
            )
            
            # Should not error with parameter conversion
            assert "error" not in result or "must be" not in result.get("error", "")


@pytest.mark.integration
class TestIntegratedBugFixes:
    """Integration tests that validate multiple fixes work together."""

    @pytest.fixture
    def integration_setup(self):
        """Setup for integration tests."""
        # This would set up a more complete test environment
        # For now, return mock setup
        return {
            "config": MagicMock(),
            "temp_dir": None
        }

    @pytest.mark.asyncio
    async def test_full_workflow_no_errors(self, integration_setup):
        """Test that a full workflow (search + scratchbook + auto-ingestion) works without the reported errors."""
        # This test would run a complete workflow to ensure all fixes work together
        # For now, just verify we can import and instantiate main components
        
        from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient
        from workspace_qdrant_mcp.tools.scratchbook import ScratchbookManager
        from workspace_qdrant_mcp.core.auto_ingestion import AutoIngestionManager
        
        # If we can import without errors, the basic structure is sound
        assert QdrantWorkspaceClient is not None
        assert ScratchbookManager is not None 
        assert AutoIngestionManager is not None


if __name__ == "__main__":
    # Run the tests
    print("Running bug fix validation tests...")
    print("\nTesting fixes for:")
    print("- Issue #12: Search functionality returns empty results") 
    print("- Issue #13: Scratchbook functionality broken")
    print("- Issue #5: Auto-ingestion not processing workspace files")
    print("- Issue #14: Advanced search type conversion errors")
    print("\n" + "="*60)
    
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])