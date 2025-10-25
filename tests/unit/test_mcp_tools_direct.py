"""
Direct Unit Tests for MCP Server Tools (Task 242.2).

This test suite provides direct function-level testing of MCP server tools
without complex infrastructure dependencies. Tests individual tool functions
with proper mocking of external dependencies.

Focus on the core MCP tools:
- workspace_status, search_workspace_tool
- collection management: list_workspace_collections, create_collection, delete_collection
- document tools: add_document_tool, get_document_tool, search_by_metadata_tool
- scratchbook tools: update_scratchbook_tool, search_scratchbook_tool, list_scratchbook_notes_tool
- hybrid search: hybrid_search_advanced_tool
"""

import asyncio
import os

# Direct imports for testing
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src/python'))

# Import server module for testing individual functions
import workspace_qdrant_mcp.server as server_module


class TestDirectMCPTools:
    """Direct testing of MCP tool functions without complex infrastructure."""

    @pytest.mark.asyncio
    async def test_workspace_status_function_direct(self):
        """Test workspace_status function directly with mocked dependencies."""

        with patch.object(server_module, 'get_current_config') as mock_config, \
             patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'detect_project') as mock_project, \
             patch.object(server_module, 'QdrantClient') as mock_qdrant_class:

            # Setup mocks
            mock_config.return_value = MagicMock(
                qdrant_client_config=MagicMock(url="http://localhost:6333"),
                workspace=MagicMock(global_collections=["global-test"])
            )

            mock_qdrant = AsyncMock()
            mock_qdrant.get_collections.return_value = MagicMock(
                collections=[
                    MagicMock(name="test-project-scratchbook"),
                    MagicMock(name="test-project-docs"),
                    MagicMock(name="global-test")
                ]
            )
            mock_qdrant_class.return_value = mock_qdrant
            mock_get_client.return_value = mock_qdrant
            mock_project.return_value = "test-project"

            # Call the function directly
            result = await server_module.workspace_status()

            # Verify the result structure
            assert isinstance(result, dict)
            assert "connected" in result
            assert "current_project" in result
            assert "collections" in result

            # Verify expected values
            assert result["connected"]
            assert result["current_project"] == "test-project"
            assert isinstance(result["collections"], (list, dict))

    @pytest.mark.asyncio
    async def test_workspace_status_connection_failure(self):
        """Test workspace_status handles connection failures gracefully."""

        with patch.object(server_module, 'get_current_config') as mock_config:
            mock_config.side_effect = Exception("Qdrant connection failed")

            # Call should handle exception gracefully
            result = await server_module.workspace_status()

            assert isinstance(result, dict)
            assert "connected" in result
            assert not result["connected"]
            assert "error" in result

    @pytest.mark.asyncio
    async def test_list_workspace_collections_direct(self):
        """Test list_workspace_collections function directly."""

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'detect_project') as mock_project, \
             patch.object(server_module, 'create_naming_manager'):

            # Setup mocks
            mock_project.return_value = "test-project"

            mock_qdrant = AsyncMock()
            mock_qdrant.get_collections.return_value = MagicMock(
                collections=[
                    MagicMock(name="test-project-scratchbook", vectors_count=15),
                    MagicMock(name="test-project-docs", vectors_count=42),
                    MagicMock(name="global-shared", vectors_count=100)
                ]
            )
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.list_workspace_collections()

            # Verify result structure
            assert isinstance(result, dict)
            assert "collections" in result
            assert isinstance(result["collections"], list)
            assert len(result["collections"]) == 3

    @pytest.mark.asyncio
    async def test_create_collection_direct(self):
        """Test create_collection function directly."""

        collection_name = "test-new-collection"
        dimension = 384
        distance = "Cosine"

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'create_naming_manager') as mock_naming:

            # Setup mocks
            mock_qdrant = AsyncMock()
            mock_qdrant.create_collection = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_qdrant

            mock_naming_manager = MagicMock()
            mock_naming_manager.create_collection_name.return_value = collection_name
            mock_naming.return_value = mock_naming_manager

            # Call the function
            result = await server_module.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                distance=distance
            )

            # Verify result
            assert isinstance(result, dict)
            assert result.get("success")
            assert collection_name in str(result)

    @pytest.mark.asyncio
    async def test_search_workspace_tool_direct(self):
        """Test search_workspace_tool function directly."""

        query = "test search query"
        limit = 5

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'get_embedding_model') as mock_embedding, \
             patch.object(server_module, 'detect_project') as mock_project:

            # Setup mocks
            mock_project.return_value = "test-project"

            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            mock_qdrant = AsyncMock()
            mock_qdrant.search.return_value = [
                MagicMock(
                    id="result1",
                    score=0.95,
                    payload={"content": "Test result 1", "metadata": {"type": "doc"}}
                ),
                MagicMock(
                    id="result2",
                    score=0.87,
                    payload={"content": "Test result 2", "metadata": {"type": "doc"}}
                )
            ]
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.search_workspace_tool(
                query=query,
                limit=limit
            )

            # Verify result structure
            assert isinstance(result, dict)
            assert "results" in result
            assert isinstance(result["results"], list)
            assert len(result["results"]) <= limit

            # Verify search was called with embedding
            mock_qdrant.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_document_tool_direct(self):
        """Test add_document_tool function directly."""

        content = "This is test document content for unit testing."
        metadata = {"type": "test", "category": "unit-test"}
        collection_name = "test-collection"

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'get_embedding_model') as mock_embedding:

            # Setup mocks
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            mock_qdrant = AsyncMock()
            mock_qdrant.upsert = AsyncMock(return_value=MagicMock(operation_id=1))
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.add_document_tool(
                content=content,
                metadata=metadata,
                collection_name=collection_name
            )

            # Verify result
            assert isinstance(result, dict)
            assert result.get("success") or "document_id" in result

            # Verify document was added to Qdrant
            mock_qdrant.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_scratchbook_tool_direct(self):
        """Test update_scratchbook_tool function directly."""

        note = "This is a test scratchbook note."
        metadata = {"priority": "high", "category": "testing"}
        project_name = "test-project"

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'get_embedding_model') as mock_embedding, \
             patch.object(server_module, 'detect_project') as mock_project:

            # Setup mocks
            mock_project.return_value = project_name

            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            mock_qdrant = AsyncMock()
            mock_qdrant.upsert = AsyncMock(return_value=MagicMock(operation_id=1))
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.update_scratchbook_tool(
                note=note,
                metadata=metadata,
                project_name=project_name
            )

            # Verify result
            assert isinstance(result, dict)
            assert "note_id" in result or result.get("success")

    @pytest.mark.asyncio
    async def test_search_scratchbook_tool_direct(self):
        """Test search_scratchbook_tool function directly."""

        query = "testing notes"
        limit = 5
        project_name = "test-project"

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'get_embedding_model') as mock_embedding, \
             patch.object(server_module, 'detect_project') as mock_project:

            # Setup mocks
            mock_project.return_value = project_name

            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            mock_qdrant = AsyncMock()
            mock_qdrant.search.return_value = [
                MagicMock(
                    id="note1",
                    score=0.95,
                    payload={"note": "Test note 1", "metadata": {"priority": "high"}}
                )
            ]
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.search_scratchbook_tool(
                query=query,
                limit=limit,
                project_name=project_name
            )

            # Verify result
            assert isinstance(result, dict)
            assert "results" in result
            assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    async def test_hybrid_search_advanced_tool_direct(self):
        """Test hybrid_search_advanced_tool function directly."""

        query = "advanced hybrid search test"
        limit = 10
        semantic_weight = 0.7
        keyword_weight = 0.3

        with patch.object(server_module, 'get_client') as mock_get_client, \
             patch.object(server_module, 'get_embedding_model') as mock_embedding:

            # Setup mocks
            mock_embedding_model = MagicMock()
            mock_embedding_model.embed_query.return_value = [0.1] * 384
            mock_embedding.return_value = mock_embedding_model

            mock_qdrant = AsyncMock()
            mock_qdrant.search.return_value = [
                MagicMock(
                    id="hybrid1",
                    score=0.92,
                    payload={"content": "Hybrid search result", "metadata": {"relevance": "high"}}
                )
            ]
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.hybrid_search_advanced_tool(
                query=query,
                limit=limit,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )

            # Verify result
            assert isinstance(result, dict)
            assert "results" in result

    @pytest.mark.asyncio
    async def test_delete_collection_direct(self):
        """Test delete_collection function directly."""

        collection_name = "test-collection-to-delete"

        with patch.object(server_module, 'get_client') as mock_get_client:

            # Setup mocks
            mock_qdrant = AsyncMock()
            mock_qdrant.delete_collection = AsyncMock(return_value=True)
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.delete_collection(collection_name=collection_name)

            # Verify result
            assert isinstance(result, dict)
            assert result.get("success")

            # Verify collection was deleted
            mock_qdrant.delete_collection.assert_called_once_with(collection_name)

    @pytest.mark.asyncio
    async def test_get_document_tool_direct(self):
        """Test get_document_tool function directly."""

        document_id = "test-doc-123"
        collection_name = "test-collection"

        with patch.object(server_module, 'get_client') as mock_get_client:

            # Setup mocks
            mock_qdrant = AsyncMock()
            mock_qdrant.retrieve.return_value = [
                MagicMock(
                    id=document_id,
                    payload={"content": "Test document content", "metadata": {"type": "test"}}
                )
            ]
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.get_document_tool(
                document_id=document_id,
                collection_name=collection_name
            )

            # Verify result
            assert isinstance(result, dict)
            assert "content" in result or "document" in result

    @pytest.mark.asyncio
    async def test_search_by_metadata_tool_direct(self):
        """Test search_by_metadata_tool function directly."""

        metadata_query = {"type": "test", "category": "document"}
        collection_name = "test-collection"
        limit = 10

        with patch.object(server_module, 'get_client') as mock_get_client:

            # Setup mocks
            mock_qdrant = AsyncMock()
            mock_qdrant.scroll.return_value = (
                [
                    MagicMock(
                        id="meta1",
                        score=0.95,
                        payload={"content": "Metadata result 1", "metadata": {"type": "test"}}
                    )
                ],
                None
            )
            mock_get_client.return_value = mock_qdrant

            # Call the function
            result = await server_module.search_by_metadata_tool(
                metadata_query=metadata_query,
                collection_name=collection_name,
                limit=limit
            )

            # Verify result
            assert isinstance(result, dict)
            assert "results" in result
            assert isinstance(result["results"], list)


class TestParameterValidation:
    """Test parameter validation for MCP tools."""

    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty search queries."""

        with patch.object(server_module, 'get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_get_client.return_value = mock_qdrant

            # Call with empty query
            result = await server_module.search_workspace_tool(query="", limit=5)

            # Should handle gracefully
            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_negative_limit_handling(self):
        """Test handling of negative limit values."""

        with patch.object(server_module, 'get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_get_client.return_value = mock_qdrant

            # Should handle negative limit gracefully
            result = await server_module.search_workspace_tool(query="test", limit=-5)

            assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_none_parameter_handling(self):
        """Test handling of None parameter values."""

        with patch.object(server_module, 'get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_get_client.return_value = mock_qdrant

            # Call with None collection_name should use default
            result = await server_module.search_workspace_tool(
                query="test",
                limit=5,
                collection_name=None
            )

            assert isinstance(result, dict)


class TestErrorHandling:
    """Test error handling in MCP tools."""

    @pytest.mark.asyncio
    async def test_qdrant_unavailable_error(self):
        """Test behavior when Qdrant is unavailable."""

        with patch.object(server_module, 'get_client') as mock_get_client:
            mock_get_client.side_effect = Exception("Qdrant unavailable")

            # Should handle gracefully
            result = await server_module.workspace_status()

            assert isinstance(result, dict)
            assert not result.get("connected")

    @pytest.mark.asyncio
    async def test_embedding_model_unavailable(self):
        """Test behavior when embedding model fails."""

        with patch.object(server_module, 'get_embedding_model') as mock_embedding:
            mock_embedding.side_effect = Exception("Embedding model failed")

            # Should handle embedding failure gracefully
            result = await server_module.search_workspace_tool(query="test")

            assert isinstance(result, dict)
            # Should either fail gracefully or provide error message

    @pytest.mark.asyncio
    async def test_collection_not_found_error(self):
        """Test behavior when collection doesn't exist."""

        with patch.object(server_module, 'get_client') as mock_get_client:
            mock_qdrant = AsyncMock()
            mock_qdrant.search.side_effect = Exception("Collection not found")
            mock_get_client.return_value = mock_qdrant

            result = await server_module.search_workspace_tool(
                query="test",
                collection="non-existent-collection"
            )

            # Should handle missing collection gracefully
            assert isinstance(result, dict)


class TestIntegrationMocking:
    """Test integration points with proper mocking."""

    @pytest.mark.asyncio
    async def test_project_detection_integration(self):
        """Test project detection integration."""

        with patch.object(server_module, 'detect_project') as mock_project:
            mock_project.return_value = "detected-project-name"

            with patch.object(server_module, 'get_client') as mock_get_client:
                mock_qdrant = AsyncMock()
                mock_qdrant.get_collections.return_value = MagicMock(collections=[])
                mock_get_client.return_value = mock_qdrant

                result = await server_module.workspace_status()

                assert result["current_project"] == "detected-project-name"

    @pytest.mark.asyncio
    async def test_collection_naming_integration(self):
        """Test collection naming manager integration."""

        with patch.object(server_module, 'create_naming_manager') as mock_naming:
            mock_manager = MagicMock()
            mock_manager.create_collection_name.return_value = "formatted-collection-name"
            mock_naming.return_value = mock_manager

            with patch.object(server_module, 'get_client') as mock_get_client:
                mock_qdrant = AsyncMock()
                mock_qdrant.create_collection = AsyncMock(return_value=True)
                mock_get_client.return_value = mock_qdrant

                await server_module.create_collection(
                    collection_name="test-collection",
                    dimension=384
                )

                # Verify naming manager was used
                mock_manager.create_collection_name.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_model_integration(self):
        """Test embedding model integration."""

        test_embedding = [0.1, 0.2, 0.3] * 128  # 384 dimensions

        with patch.object(server_module, 'get_embedding_model') as mock_embedding:
            mock_model = MagicMock()
            mock_model.embed_query.return_value = test_embedding
            mock_embedding.return_value = mock_model

            with patch.object(server_module, 'get_client') as mock_get_client:
                mock_qdrant = AsyncMock()
                mock_qdrant.search.return_value = []
                mock_get_client.return_value = mock_qdrant

                await server_module.search_workspace_tool(query="test query")

                # Verify embedding was generated
                mock_model.embed_query.assert_called_once_with("test query")
