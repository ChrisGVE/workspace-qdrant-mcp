"""
Unit tests for scratchbook management tools.

Tests scratchbook note operations, search, and management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid
from datetime import datetime

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from workspace_qdrant_mcp.tools.scratchbook import (
    ScratchbookManager,
    update_scratchbook
)
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient


class TestScratchbookManager:
    """Test ScratchbookManager class."""
    
    @pytest.fixture
    def manager(self, mock_workspace_client):
        """Create ScratchbookManager instance for testing."""
        mock_workspace_client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": [],
            "github_user": "testuser"
        }
        return ScratchbookManager(mock_workspace_client)
    
    def test_init(self, mock_workspace_client):
        """Test ScratchbookManager initialization."""
        mock_workspace_client.get_project_info.return_value = {
            "main_project": "test-project",
            "subprojects": []
        }
        
        manager = ScratchbookManager(mock_workspace_client)
        
        assert manager.client == mock_workspace_client
        assert manager.project_info["main_project"] == "test-project"
        mock_workspace_client.get_project_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_note_client_not_initialized(self, manager):
        """Test add_note with uninitialized client."""
        manager.client.initialized = False
        
        result = await manager.add_note("Test note content")
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_add_note_empty_content(self, manager):
        """Test add_note with empty content."""
        manager.client.initialized = True
        
        # Test empty string
        result = await manager.add_note("")
        assert result["error"] == "Note content cannot be empty"
        
        # Test whitespace-only string
        result = await manager.add_note("   ")
        assert result["error"] == "Note content cannot be empty"
    
    @pytest.mark.asyncio
    async def test_add_note_collection_not_found(self, manager):
        """Test add_note with non-existent collection."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["other-collection"]
        
        result = await manager.add_note("Test content")
        
        assert "error" in result
        assert "Scratchbook collection 'test-project-scratchbook' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_add_note_success(self, manager, mock_embedding_service):
        """Test successful note addition."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        
        result = await manager.add_note(
            "Test note content",
            title="Test Note",
            tags=["test", "python"],
            note_type="idea"
        )
        
        assert "note_id" in result
        assert result["title"] == "Test Note"
        assert result["collection"] == "test-project-scratchbook"
        assert result["note_type"] == "idea"
        assert result["tags"] == ["test", "python"]
        assert result["content_length"] == 17
        assert "created_at" in result
        
        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "test-project-scratchbook"
        
        point = call_args[1]["points"][0]
        assert point.payload["content"] == "Test note content"
        assert point.payload["title"] == "Test Note"
        assert point.payload["note_type"] == "idea"
        assert point.payload["tags"] == ["test", "python"]
        assert point.payload["is_scratchbook_note"] is True
        assert "dense" in point.vector
        assert "sparse" in point.vector
    
    @pytest.mark.asyncio
    async def test_add_note_auto_title_generation(self, manager, mock_embedding_service):
        """Test automatic title generation."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        
        with patch.object(manager, '_generate_title_from_content', return_value="Auto Title") as mock_gen_title:
            result = await manager.add_note("This is a test note without title")
        
        assert result["title"] == "Auto Title"
        mock_gen_title.assert_called_once_with("This is a test note without title")
    
    @pytest.mark.asyncio
    async def test_add_note_default_project(self, mock_workspace_client, mock_embedding_service):
        """Test add_note with no project info (default project)."""
        # Create manager with no project info
        mock_workspace_client.get_project_info.return_value = None
        manager = ScratchbookManager(mock_workspace_client)
        
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["default-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        
        result = await manager.add_note("Test content")
        
        assert result["collection"] == "default-scratchbook"
        
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "default-scratchbook"
        point = call_args[1]["points"][0]
        assert point.payload["project_name"] == "default"
    
    @pytest.mark.asyncio
    async def test_add_note_custom_project(self, manager, mock_embedding_service):
        """Test add_note with custom project name."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["custom-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        
        result = await manager.add_note(
            "Test content",
            project_name="custom-project"
        )
        
        assert result["collection"] == "custom-project-scratchbook"
    
    @pytest.mark.asyncio
    async def test_add_note_no_sparse_vectors(self, manager, mock_embedding_service):
        """Test note addition without sparse vectors."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        # No sparse vectors in embeddings
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        
        result = await manager.add_note("Test content")
        
        # Verify point structure
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert "dense" in point.vector
        assert "sparse" not in point.vector
    
    @pytest.mark.asyncio
    async def test_add_note_exception_handling(self, manager):
        """Test add_note exception handling."""
        manager.client.initialized = True
        manager.client.list_collections.side_effect = Exception("Connection error")
        
        result = await manager.add_note("Test content")
        
        assert "error" in result
        assert "Failed to add note" in result["error"]
        assert "Connection error" in result["error"]
    
    def test_generate_title_from_content_short(self, manager):
        """Test title generation from short content."""
        content = "This is a short note"
        title = manager._generate_title_from_content(content)
        
        assert title == "This is a short note"
    
    def test_generate_title_from_content_long(self, manager):
        """Test title generation from long content."""
        content = "This is a very long note that exceeds fifty characters and should be truncated properly"
        title = manager._generate_title_from_content(content)
        
        assert len(title) <= 50
        assert title.endswith("...")
        assert content.startswith(title[:-3])  # Remove "..." for comparison
    
    def test_generate_title_from_content_multiline(self, manager):
        """Test title generation from multiline content."""
        content = "This is the first line\nThis is the second line\nThis is the third line"
        title = manager._generate_title_from_content(content)
        
        assert title == "This is the first line"
    
    @pytest.mark.asyncio
    async def test_update_note_client_not_initialized(self, manager):
        """Test update_note with uninitialized client."""
        manager.client.initialized = False
        
        result = await manager.update_note("note123")
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_update_note_not_found(self, manager):
        """Test update_note with non-existent note."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = ([], None)  # Empty results
        
        result = await manager.update_note("nonexistent_note")
        
        assert "error" in result
        assert "Note 'nonexistent_note' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_update_note_success_metadata_only(self, manager):
        """Test successful note update (metadata only)."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        
        # Mock existing note
        mock_point = MagicMock()
        mock_point.id = "note123"
        mock_point.payload = {
            "content": "Original content",
            "title": "Original Title",
            "tags": ["old"],
            "version": 1,
            "created_at": "2023-01-01T00:00:00"
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await manager.update_note(
            "note123",
            title="Updated Title",
            tags=["new", "updated"]
        )
        
        assert result["note_id"] == "note123"
        assert result["title"] == "Updated Title"
        assert result["content_updated"] is False
        assert result["metadata_updated"] is True
        
        # Verify upsert was called
        mock_client.upsert.assert_called_once()
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["title"] == "Updated Title"
        assert point.payload["tags"] == ["new", "updated"]
        assert point.payload["version"] == 2  # Version incremented
        assert "updated_at" in point.payload
    
    @pytest.mark.asyncio
    async def test_update_note_with_content(self, manager, mock_embedding_service):
        """Test note update with new content."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock new embeddings
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.2] * 384,
            "sparse": {"indices": [3, 4], "values": [0.9, 0.7]}
        }
        
        # Mock existing note
        mock_point = MagicMock()
        mock_point.id = "note123"
        mock_point.payload = {
            "content": "Original content",
            "title": "Original Title",
            "version": 1
        }
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        result = await manager.update_note(
            "note123",
            content="Updated content",
            title="Updated Title"
        )
        
        assert result["content_updated"] is True
        assert result["metadata_updated"] is True
        
        # Verify new vectors were generated
        call_args = mock_client.upsert.call_args
        point = call_args[1]["points"][0]
        assert point.payload["content"] == "Updated content"
        assert "dense" in point.vector
        assert "sparse" in point.vector
    
    @pytest.mark.asyncio
    async def test_search_notes_client_not_initialized(self, manager):
        """Test search_notes with uninitialized client."""
        manager.client.initialized = False
        
        result = await manager.search_notes("test query")
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_search_notes_collection_not_found(self, manager):
        """Test search_notes with non-existent collection."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["other-collection"]
        
        result = await manager.search_notes("test query")
        
        assert "error" in result
        assert "Scratchbook collection 'test-project-scratchbook' not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_search_notes_success(self, manager, mock_embedding_service):
        """Test successful note search."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        # Mock embeddings
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384,
            "sparse": {"indices": [1, 2], "values": [0.8, 0.6]}
        }
        
        # Mock search results
        mock_results = [
            models.ScoredPoint(
                id="note1",
                score=0.9,
                version=0,
                payload={"title": "Test Note 1", "content": "Content 1", "tags": ["test"]}
            ),
            models.ScoredPoint(
                id="note2",
                score=0.8,
                version=0,
                payload={"title": "Test Note 2", "content": "Content 2", "tags": ["python"]}
            )
        ]
        
        # Mock hybrid search engine
        with patch('workspace_qdrant_mcp.tools.scratchbook.HybridSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.hybrid_search.return_value = {
                "results": [
                    {"id": "note1", "score": 0.9, "payload": mock_results[0].payload},
                    {"id": "note2", "score": 0.8, "payload": mock_results[1].payload}
                ],
                "total": 2
            }
            mock_engine_class.return_value = mock_engine
            
            result = await manager.search_notes(
                "test query",
                note_types=["note"],
                tags=["test"],
                limit=10
            )
        
        assert "results" in result
        assert "total" in result
        assert len(result["results"]) == 2
        assert result["total"] == 2
        
        # Verify search was called with correct parameters
        mock_engine.hybrid_search.assert_called_once()
        call_args = mock_engine.hybrid_search.call_args
        assert call_args[1]["collection_name"] == "test-project-scratchbook"
        assert call_args[1]["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_search_notes_with_filters(self, manager, mock_embedding_service):
        """Test note search with various filters."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        manager.client.get_embedding_service.return_value = mock_embedding_service
        
        mock_embedding_service.generate_embeddings.return_value = {
            "dense": [0.1] * 384
        }
        
        with patch('workspace_qdrant_mcp.tools.scratchbook.HybridSearchEngine') as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine.hybrid_search.return_value = {"results": [], "total": 0}
            mock_engine_class.return_value = mock_engine
            
            await manager.search_notes(
                "test query",
                note_types=["idea", "todo"],
                tags=["python", "ai"],
                project_name="custom-project",
                limit=20,
                mode="dense"
            )
        
        # Verify filters were applied
        call_args = mock_engine.hybrid_search.call_args
        filters = call_args[1]["search_filter"]
        assert filters is not None
        
        # Should search in custom project collection
        assert call_args[1]["collection_name"] == "custom-project-scratchbook"
    
    @pytest.mark.asyncio
    async def test_list_notes_success(self, manager):
        """Test successful note listing."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        
        # Mock scroll results
        mock_points = [
            MagicMock(id="note1", payload={"title": "Note 1", "note_type": "note"}),
            MagicMock(id="note2", payload={"title": "Note 2", "note_type": "idea"})
        ]
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = (mock_points, None)
        
        result = await manager.list_notes(limit=10)
        
        assert "notes" in result
        assert "total" in result
        assert len(result["notes"]) == 2
        assert result["total"] == 2
        
        # Verify scroll was called with correct parameters
        mock_client.scroll.assert_called_once()
        call_args = mock_client.scroll.call_args
        assert call_args[1]["collection_name"] == "test-project-scratchbook"
        assert call_args[1]["limit"] == 10
    
    @pytest.mark.asyncio
    async def test_delete_note_success(self, manager):
        """Test successful note deletion."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        
        # Mock existing note
        mock_point = MagicMock()
        mock_point.id = "note123"
        mock_point.payload = {"title": "Test Note", "note_type": "note"}
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = ([mock_point], None)
        
        # Mock successful delete
        mock_delete_result = MagicMock()
        mock_delete_result.operation_id = 12345
        mock_client.delete.return_value = mock_delete_result
        
        result = await manager.delete_note("note123")
        
        assert result["note_id"] == "note123"
        assert result["status"] == "success"
        assert "deleted_at" in result
        
        # Verify delete was called
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args
        assert call_args[1]["collection_name"] == "test-project-scratchbook"
    
    @pytest.mark.asyncio
    async def test_delete_note_not_found(self, manager):
        """Test delete of non-existent note."""
        manager.client.initialized = True
        manager.client.list_collections.return_value = ["test-project-scratchbook"]
        
        mock_client = MagicMock()
        manager.client.client = mock_client
        mock_client.scroll.return_value = ([], None)  # Empty results
        
        result = await manager.delete_note("nonexistent_note")
        
        assert "error" in result
        assert "Note 'nonexistent_note' not found" in result["error"]


class TestUpdateScratchbook:
    """Test update_scratchbook function."""
    
    @pytest.mark.asyncio
    async def test_update_scratchbook_client_not_initialized(self, mock_workspace_client):
        """Test update_scratchbook with uninitialized client."""
        mock_workspace_client.initialized = False
        
        result = await update_scratchbook(
            mock_workspace_client,
            "Test content"
        )
        
        assert result["error"] == "Workspace client not initialized"
    
    @pytest.mark.asyncio
    async def test_update_scratchbook_add_new_note(self, mock_workspace_client):
        """Test update_scratchbook adding new note."""
        mock_workspace_client.initialized = True
        
        # Mock ScratchbookManager
        with patch('workspace_qdrant_mcp.tools.scratchbook.ScratchbookManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.add_note.return_value = {
                "note_id": "note123",
                "title": "Test Note",
                "status": "success"
            }
            mock_manager_class.return_value = mock_manager
            
            result = await update_scratchbook(
                mock_workspace_client,
                "Test content",
                title="Test Note",
                tags=["test"]
            )
        
        assert result["note_id"] == "note123"
        assert result["title"] == "Test Note"
        
        # Verify add_note was called
        mock_manager.add_note.assert_called_once_with(
            "Test content",
            "Test Note",
            ["test"],
            "note",
            None
        )
    
    @pytest.mark.asyncio
    async def test_update_scratchbook_update_existing_note(self, mock_workspace_client):
        """Test update_scratchbook updating existing note."""
        mock_workspace_client.initialized = True
        
        with patch('workspace_qdrant_mcp.tools.scratchbook.ScratchbookManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager.update_note.return_value = {
                "note_id": "note123",
                "title": "Updated Note",
                "status": "success"
            }
            mock_manager_class.return_value = mock_manager
            
            result = await update_scratchbook(
                mock_workspace_client,
                "Updated content",
                note_id="note123",
                title="Updated Note"
            )
        
        assert result["note_id"] == "note123"
        assert result["title"] == "Updated Note"
        
        # Verify update_note was called
        mock_manager.update_note.assert_called_once_with(
            "note123",
            "Updated content",
            "Updated Note",
            None,
            None
        )
    
    @pytest.mark.asyncio
    async def test_update_scratchbook_exception_handling(self, mock_workspace_client):
        """Test update_scratchbook exception handling."""
        mock_workspace_client.initialized = True
        
        with patch('workspace_qdrant_mcp.tools.scratchbook.ScratchbookManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("Manager error")
            
            result = await update_scratchbook(
                mock_workspace_client,
                "Test content"
            )
        
        assert "error" in result
        assert "Failed to update scratchbook" in result["error"]
        assert "Manager error" in result["error"]
