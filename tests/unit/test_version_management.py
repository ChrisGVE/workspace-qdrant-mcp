"""
Unit tests for version-aware document management system.

Tests the implementation of Task 12 from PRD v2.0 specifications.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from workspace_qdrant_mcp.tools.documents import (
    find_document_versions,
    ingest_new_version, 
    get_document_type_config
)


class TestVersionManagement:
    """Test version management functionality."""

    def test_get_document_type_config(self):
        """Test document type configuration retrieval."""
        # Test book type
        book_config = get_document_type_config("book")
        assert book_config["primary_version"] == "edition"
        assert "title" in book_config["required_metadata"]
        assert book_config["retention_policy"] == "latest_only"
        
        # Test code file type  
        code_config = get_document_type_config("code_file")
        assert code_config["primary_version"] == "git_tag"
        assert code_config["auto_metadata"] is True
        
        # Test generic fallback
        generic_config = get_document_type_config("unknown_type")
        assert generic_config["primary_version"] == "timestamp"
        
    @pytest.mark.asyncio
    async def test_find_document_versions_empty(self):
        """Test finding versions when no documents exist."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.client.scroll = AsyncMock(return_value=([], None))
        
        versions = await find_document_versions(mock_client, "test-doc", "test-collection")
        
        assert versions == []
        mock_client.client.scroll.assert_called_once()
        
    @pytest.mark.asyncio  
    async def test_find_document_versions_with_data(self):
        """Test finding versions when documents exist."""
        mock_client = MagicMock()
        mock_client.initialized = True
        
        # Mock existing document
        mock_point = MagicMock()
        mock_point.id = "point-123"
        mock_point.payload = {
            "document_id": "test-doc",
            "version": "1.0.0", 
            "is_latest": True,
            "added_at": "2025-01-01T00:00:00Z"
        }
        
        mock_client.client.scroll = AsyncMock(return_value=([mock_point], None))
        
        versions = await find_document_versions(mock_client, "test-doc", "test-collection")
        
        assert len(versions) == 1
        assert versions[0]["document_id"] == "test-doc"
        assert versions[0]["version"] == "1.0.0"
        assert versions[0]["is_latest"] is True
        
    @pytest.mark.asyncio
    async def test_ingest_new_version_first_version(self):
        """Test ingesting first version of a document."""
        mock_client = MagicMock()
        mock_client.initialized = True
        
        # Mock no existing versions
        with patch(
            "workspace_qdrant_mcp.tools.documents.find_document_versions", 
            return_value=[]
        ), patch(
            "workspace_qdrant_mcp.tools.documents.add_document",
            return_value={"document_id": "test-doc", "points_added": 1}
        ) as mock_add:
            
            result = await ingest_new_version(
                client=mock_client,
                content="Test content",
                collection="test-collection",
                document_id="test-doc", 
                version="1.0.0",
                document_type="book"
            )
            
        assert "error" not in result
        assert result["versions_superseded"] == 0
        assert result["is_new_version"] is False
        
        # Check that add_document was called with version metadata
        mock_add.assert_called_once()
        call_args = mock_add.call_args
        metadata = call_args.kwargs["metadata"]
        assert metadata["version"] == "1.0.0"
        assert metadata["is_latest"] is True
        assert metadata["document_type"] == "book"
        
    @pytest.mark.asyncio
    async def test_ingest_new_version_supersedes_existing(self):
        """Test ingesting new version that supersedes existing versions."""
        mock_client = MagicMock()
        mock_client.initialized = True
        mock_client.client.set_payload = AsyncMock()
        
        # Mock existing version
        existing_versions = [
            {
                "point_id": "old-point-123",
                "document_id": "test-doc",
                "version": "1.0.0",
                "is_latest": True
            }
        ]
        
        with patch(
            "workspace_qdrant_mcp.tools.documents.find_document_versions",
            return_value=existing_versions
        ), patch(
            "workspace_qdrant_mcp.tools.documents.add_document", 
            return_value={"document_id": "test-doc", "points_added": 1}
        ) as mock_add:
            
            result = await ingest_new_version(
                client=mock_client,
                content="Updated content", 
                collection="test-collection",
                document_id="test-doc",
                version="2.0.0"
            )
            
        assert "error" not in result
        assert result["versions_superseded"] == 1
        assert result["is_new_version"] is True
        
        # Check that old version was de-prioritized
        mock_client.client.set_payload.assert_called_once_with(
            collection_name="test-collection",
            points=["old-point-123"],
            payload={"is_latest": False, "search_priority": 0.1}
        )
        
        # Check that new version metadata includes supersedes
        call_args = mock_add.call_args
        metadata = call_args.kwargs["metadata"] 
        assert "old-point-123" in metadata["supersedes"]
        assert metadata["version"] == "2.0.0"
        assert metadata["is_latest"] is True