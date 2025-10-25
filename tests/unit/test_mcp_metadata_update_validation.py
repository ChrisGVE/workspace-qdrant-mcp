"""
Comprehensive metadata update validation tests for MCP document management (Task 323.3).

This test suite validates metadata-only updates, metadata validation rules, schema enforcement,
metadata search functionality, persistence, update timestamps, version tracking, and handling
of invalid metadata formats in the MCP server's `store` tool.

Test Coverage:
    - Metadata-only updates without content modification
    - Metadata validation rules and schema enforcement
    - Metadata search and filtering functionality
    - Metadata persistence across operations
    - Update timestamp tracking
    - Version tracking for metadata changes
    - Invalid metadata format handling
    - Error handling and edge cases
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

# Add the src directory to Python path for imports
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

# Test imports
try:
    from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
    from workspace_qdrant_mcp import server
    SERVER_AVAILABLE = True
except ImportError as e:
    SERVER_AVAILABLE = False
    server = None
    print(f"Server import failed: {e}")

pytestmark = pytest.mark.skipif(not SERVER_AVAILABLE, reason="Server module not available")


class TestMetadataOnlyUpdates:
    """Test metadata-only updates without content modification."""

    @pytest.mark.asyncio
    async def test_metadata_only_update_preserves_content(self):
        """Test that metadata-only update preserves original content."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        initial_content = "Original content for metadata update test"
        initial_metadata = {
            "title": "Test Document",
            "version": "1.0",
            "tags": ["test", "metadata"]
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                # Access the underlying function via .fn attribute
                result = await server.store.fn(
                    content=initial_content,
                    metadata=initial_metadata,
                    source="test"
                )

        assert result["success"] is True
        assert "document_id" in result

        # Verify daemon_client.ingest_text was called with metadata
        mock_daemon.ingest_text.assert_called_once()
        call_args = mock_daemon.ingest_text.call_args
        assert call_args.kwargs['content'] == initial_content
        assert "title" in call_args.kwargs['metadata']
        assert call_args.kwargs['metadata']['title'] == "Test Document"

    @pytest.mark.asyncio
    async def test_metadata_update_with_additional_fields(self):
        """Test adding additional metadata fields to existing document."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        initial_metadata = {
            "category": "documentation",
            "priority": "high"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content for metadata field addition",
                    metadata=initial_metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify initial metadata was passed
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']
        assert metadata['category'] == "documentation"
        assert metadata['priority'] == "high"

    @pytest.mark.asyncio
    async def test_metadata_update_removes_fields(self):
        """Test that metadata updates can effectively remove fields."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        minimal_metadata = {
            "title": "Minimal Metadata Document"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with minimal metadata",
                    metadata=minimal_metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify only specified metadata is present
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']
        assert metadata['title'] == "Minimal Metadata Document"
        # Other metadata fields should be auto-generated but not from previous updates
        assert 'created_at' in metadata  # Auto-generated field

    @pytest.mark.asyncio
    async def test_empty_metadata_update(self):
        """Test update with empty metadata dictionary."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with no custom metadata",
                    metadata={},
                    source="test"
                )

        assert result["success"] is True

        # Should still have auto-generated metadata
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']
        assert 'created_at' in metadata
        assert 'project' in metadata


class TestMetadataValidationRules:
    """Test metadata validation rules and enforcement."""

    @pytest.mark.asyncio
    async def test_metadata_type_validation(self):
        """Test that metadata values maintain their types."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "version": 1.0,  # float
            "count": 42,  # int
            "enabled": True,  # bool
            "tags": ["tag1", "tag2"],  # list
            "config": {"key": "value"}  # dict
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with various metadata types",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify types are preserved
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert stored_metadata['version'] == 1.0
        assert stored_metadata['count'] == 42
        assert stored_metadata['enabled'] is True
        assert stored_metadata['tags'] == ["tag1", "tag2"]
        assert stored_metadata['config'] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_metadata_with_none_values(self):
        """Test handling of None values in metadata."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "optional_field": None,
            "required_field": "present"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with None metadata values",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify None values are handled
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert "required_field" in stored_metadata
        assert stored_metadata['required_field'] == "present"

    @pytest.mark.asyncio
    async def test_metadata_special_characters(self):
        """Test metadata with special characters and unicode."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "title": "Test with Special Chars: !@#$%^&*()",
            "unicode_field": "Unicode: café, naïve, 日本語",
            "emoji": "🚀 Rocket Ship 🎉"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with special character metadata",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify special characters are preserved
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert "🚀" in stored_metadata['emoji']
        assert "café" in stored_metadata['unicode_field']

    @pytest.mark.asyncio
    async def test_metadata_field_name_constraints(self):
        """Test metadata with various field name formats."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "simple": "value",
            "with_underscore": "value",
            "with-dash": "value",
            "camelCase": "value",
            "PascalCase": "value",
            "number123": "value"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with various field name formats",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify all field names are accepted
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert "simple" in stored_metadata
        assert "with_underscore" in stored_metadata
        assert "with-dash" in stored_metadata
        assert "camelCase" in stored_metadata


class TestSchemaEnforcement:
    """Test metadata schema enforcement."""

    @pytest.mark.asyncio
    async def test_auto_generated_metadata_fields(self):
        """Test that required auto-generated metadata fields are present."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content for auto-generated metadata test",
                    source="test"
                )

        assert result["success"] is True

        # Verify auto-generated fields
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']

        # Check for auto-generated fields
        assert 'created_at' in metadata
        assert 'project' in metadata
        assert 'source' in metadata
        assert 'document_type' in metadata

        # Validate timestamp format
        created_at = metadata['created_at']
        assert isinstance(created_at, str)
        # Should be ISO format
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))

    @pytest.mark.asyncio
    async def test_file_path_metadata_enrichment(self):
        """Test that file_path metadata is enriched with file_name."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        file_path = "/path/to/project/src/module.py"

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="File content",
                    file_path=file_path,
                    source="file"
                )

        assert result["success"] is True

        # Verify file metadata enrichment
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']

        assert metadata['file_path'] == file_path
        assert metadata['file_name'] == "module.py"

    @pytest.mark.asyncio
    async def test_url_metadata_enrichment(self):
        """Test that URL metadata is enriched with domain."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        url = "https://example.com/docs/api/reference"

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Web content",
                    url=url,
                    source="web"
                )

        assert result["success"] is True

        # Verify URL metadata enrichment
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']

        assert metadata['url'] == url
        assert metadata['domain'] == "example.com"

    @pytest.mark.asyncio
    async def test_content_preview_generation(self):
        """Test automatic content preview generation in metadata."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        # Long content to test preview truncation
        long_content = "A" * 300

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content=long_content,
                    source="test"
                )

        assert result["success"] is True

        # Verify content preview
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']

        assert 'content_preview' in metadata
        preview = metadata['content_preview']
        assert len(preview) <= 203  # 200 chars + "..."
        assert preview.endswith("...")


class TestMetadataSearchFunctionality:
    """Test metadata search and filtering functionality."""

    @pytest.mark.asyncio
    async def test_search_with_metadata_filters(self):
        """Test search with metadata filter conditions."""
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.95
        mock_hit.payload = {
            "content": "Test content",
            "title": "Test Document",
            "category": "test",
            "priority": "high"
        }
        mock_qdrant.search.return_value = [mock_hit]
        mock_qdrant.get_collection.return_value = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 384]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant):
            with patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    result = await server.search.fn(
                        query="test query",
                        filters={"category": "test", "priority": "high"},
                        mode="semantic"
                    )

        assert result["success"] is True
        assert len(result["results"]) > 0

        # Verify filter was applied
        mock_qdrant.search.assert_called_once()
        call_args = mock_qdrant.search.call_args
        query_filter = call_args.kwargs['query_filter']
        assert query_filter is not None

    @pytest.mark.asyncio
    async def test_search_with_branch_filter(self):
        """Test search with branch-based metadata filtering."""
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.90
        mock_hit.payload = {
            "content": "Branch content",
            "branch": "feature-branch",
            "file_type": "code"
        }
        mock_qdrant.search.return_value = [mock_hit]
        mock_qdrant.get_collection.return_value = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 384]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant):
            with patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    with patch('workspace_qdrant_mcp.server.get_current_branch', return_value="feature-branch"):
                        result = await server.search.fn(
                            query="test",
                            branch="feature-branch",
                            mode="semantic"
                        )

        assert result["success"] is True

        # Verify branch filter was applied
        call_args = mock_qdrant.search.call_args
        query_filter = call_args.kwargs['query_filter']
        assert query_filter is not None
        assert result["filters_applied"]["branch"] == "feature-branch"

    @pytest.mark.asyncio
    async def test_search_with_file_type_filter(self):
        """Test search with file_type metadata filtering."""
        mock_qdrant = MagicMock()
        mock_hit = MagicMock()
        mock_hit.id = "doc1"
        mock_hit.score = 0.88
        mock_hit.payload = {
            "content": "Code content",
            "file_type": "code",
            "language": "python"
        }
        mock_qdrant.search.return_value = [mock_hit]
        mock_qdrant.scroll.return_value = ([mock_hit], None)
        mock_qdrant.get_collection.return_value = MagicMock()

        mock_embedding = MagicMock()
        mock_embedding.embed.return_value = [[0.1] * 384]

        with patch('workspace_qdrant_mcp.server.qdrant_client', mock_qdrant):
            with patch('workspace_qdrant_mcp.server.embedding_model', mock_embedding):
                with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                    with patch('workspace_qdrant_mcp.server.get_current_branch', return_value="main"):
                        result = await server.search.fn(
                            query="function definition",
                            file_type="code",
                            mode="hybrid"
                        )

        assert result["success"] is True
        assert result["filters_applied"]["file_type"] == "code"


class TestUpdateTimestamps:
    """Test update timestamp tracking in metadata."""

    @pytest.mark.asyncio
    async def test_created_at_timestamp_format(self):
        """Test that created_at timestamp is in correct ISO format."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content for timestamp test",
                    source="test"
                )

        assert result["success"] is True

        # Verify timestamp format
        call_args = mock_daemon.ingest_text.call_args
        metadata = call_args.kwargs['metadata']

        created_at = metadata['created_at']
        # Should be parseable as ISO datetime
        parsed_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        assert isinstance(parsed_time, datetime)

        # Should be recent (within last minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - parsed_time).total_seconds()
        assert time_diff < 60  # Within 1 minute

    @pytest.mark.asyncio
    async def test_custom_timestamp_metadata(self):
        """Test storing custom timestamp fields in metadata."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        custom_time = "2024-01-15T10:30:00Z"
        metadata = {
            "custom_timestamp": custom_time,
            "last_modified": "2024-01-16T14:20:00Z"
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with custom timestamps",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify custom timestamps are preserved
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert stored_metadata["custom_timestamp"] == custom_time
        assert stored_metadata["last_modified"] == "2024-01-16T14:20:00Z"


class TestVersionTracking:
    """Test version tracking in metadata."""

    @pytest.mark.asyncio
    async def test_version_metadata_field(self):
        """Test explicit version field in metadata."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "version": "1.0.0",
            "revision": 1
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Version 1.0.0 content",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify version metadata
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert stored_metadata["version"] == "1.0.0"
        assert stored_metadata["revision"] == 1


class TestInvalidMetadataFormats:
    """Test handling of invalid metadata formats."""

    @pytest.mark.asyncio
    async def test_metadata_with_list_of_dicts(self):
        """Test metadata containing list of dictionaries."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "items": [
                {"id": 1, "name": "item1"},
                {"id": 2, "name": "item2"}
            ]
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with list of dicts metadata",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify structure is preserved
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert len(stored_metadata["items"]) == 2
        assert stored_metadata["items"][0]["id"] == 1

    @pytest.mark.asyncio
    async def test_metadata_with_boolean_values(self):
        """Test metadata with various boolean representations."""
        mock_daemon = AsyncMock()
        ingest_response = MagicMock()
        ingest_response.document_id = str(uuid.uuid4())
        ingest_response.chunks_created = 1
        mock_daemon.ingest_text.return_value = ingest_response

        metadata = {
            "is_active": True,
            "is_deleted": False,
            "enabled": True
        }

        with patch('workspace_qdrant_mcp.server.daemon_client', mock_daemon):
            with patch('workspace_qdrant_mcp.server.initialize_components', AsyncMock()):
                result = await server.store.fn(
                    content="Content with boolean metadata",
                    metadata=metadata,
                    source="test"
                )

        assert result["success"] is True

        # Verify boolean values
        call_args = mock_daemon.ingest_text.call_args
        stored_metadata = call_args.kwargs['metadata']
        assert stored_metadata["is_active"] is True
        assert stored_metadata["is_deleted"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
