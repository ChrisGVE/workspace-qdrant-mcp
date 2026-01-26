"""
Unit tests for Memory Schema Daemon Routing (Task 30).

Tests ADR-002 compliance: all writes route through daemon or unified_queue.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from qdrant_client.models import Distance

from common.memory.schema import MemoryCollectionSchema
from common.memory.types import MemoryRule, MemoryCategory, AuthorityLevel


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client."""
    client = AsyncMock()
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.create_collection = AsyncMock()
    client.upsert = MagicMock()
    client.delete = AsyncMock(return_value=MagicMock(status="completed"))
    return client


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = AsyncMock()
    service.generate_embeddings = AsyncMock(return_value=[[0.1] * 384])
    return service


@pytest.fixture
def mock_daemon_client():
    """Create a mock daemon client."""
    client = AsyncMock()
    client.check_health = AsyncMock(return_value=MagicMock(healthy=True))
    client.create_collection_v2 = AsyncMock()
    client.ingest_text = AsyncMock(return_value=MagicMock(document_id="doc-123"))
    client.delete_document = AsyncMock()
    return client


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    manager = AsyncMock()
    manager.enqueue_unified = AsyncMock(return_value=("queue-id-123", True))
    return manager


@pytest.fixture
def sample_rule():
    """Create a sample memory rule for testing."""
    return MemoryRule(
        id="rule-test-123",
        rule="Always use type hints in Python code.",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "typing"],
        tags=["python", "best-practices"],
        source="test",
        created_at=datetime.now(),
    )


class TestMemorySchemaConstruction:
    """Test MemoryCollectionSchema construction with optional dependencies."""

    def test_construction_minimal(self, mock_qdrant_client, mock_embedding_service):
        """Test construction with only required dependencies."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        assert schema.client == mock_qdrant_client
        assert schema.embedding_service == mock_embedding_service
        assert schema.daemon_client is None
        assert schema.state_manager is None

    def test_construction_with_daemon(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client
    ):
        """Test construction with daemon client."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        assert schema.daemon_client == mock_daemon_client

    def test_construction_with_state_manager(
        self, mock_qdrant_client, mock_embedding_service, mock_state_manager
    ):
        """Test construction with state manager for queue fallback."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            state_manager=mock_state_manager,
        )

        assert schema.state_manager == mock_state_manager


class TestDaemonAvailability:
    """Test daemon availability checking."""

    @pytest.mark.asyncio
    async def test_daemon_available(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client
    ):
        """Test daemon availability when healthy."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema._is_daemon_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_daemon_unavailable_no_client(
        self, mock_qdrant_client, mock_embedding_service
    ):
        """Test daemon unavailable when no client configured."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        result = await schema._is_daemon_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_daemon_unavailable_unhealthy(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client
    ):
        """Test daemon unavailable when unhealthy."""
        mock_daemon_client.check_health = AsyncMock(
            return_value=MagicMock(healthy=False)
        )

        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema._is_daemon_available()
        assert result is False


class TestEnsureCollectionExists:
    """Test ensure_collection_exists with daemon routing."""

    @pytest.mark.asyncio
    async def test_collection_already_exists(
        self, mock_qdrant_client, mock_embedding_service
    ):
        """Test when collection already exists."""
        # Mock collection exists - use spec_set to avoid MagicMock name reservation issue
        mock_collection = MagicMock()
        mock_collection.name = "memory"  # Set name as attribute, not constructor param

        mock_qdrant_client.get_collections = AsyncMock(
            return_value=MagicMock(collections=[mock_collection])
        )

        # Mock collection info with proper schema
        mock_vectors = MagicMock()
        mock_vectors.size = 384
        mock_vectors.distance = Distance.COSINE

        mock_params = MagicMock()
        mock_params.vectors = mock_vectors

        mock_config = MagicMock()
        mock_config.params = mock_params

        mock_collection_info = MagicMock()
        mock_collection_info.config = mock_config

        mock_qdrant_client.get_collection = AsyncMock(return_value=mock_collection_info)

        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        result = await schema.ensure_collection_exists()

        assert result["success"] is True
        assert result["fallback_mode"] == "existing"
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_via_daemon(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client
    ):
        """Test collection creation via daemon (ADR-002 primary path)."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema.ensure_collection_exists()

        assert result["success"] is True
        assert result["fallback_mode"] == "daemon"
        mock_daemon_client.create_collection_v2.assert_called_once()
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_via_queue_fallback(
        self, mock_qdrant_client, mock_embedding_service, mock_state_manager
    ):
        """Test collection creation via queue when daemon unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            state_manager=mock_state_manager,
        )

        result = await schema.ensure_collection_exists()

        assert result["success"] is True
        assert result["queued"] is True
        assert result["fallback_mode"] == "unified_queue"
        assert result["queue_id"] == "queue-id-123"
        mock_state_manager.enqueue_unified.assert_called_once()
        mock_qdrant_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_direct_fallback(
        self, mock_qdrant_client, mock_embedding_service
    ):
        """Test direct Qdrant fallback when both daemon and queue unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        result = await schema.ensure_collection_exists()

        assert result["success"] is True
        assert result["fallback_mode"] == "direct_qdrant"
        assert "warning" in result
        mock_qdrant_client.create_collection.assert_called_once()


class TestStoreRule:
    """Test store_rule with daemon routing."""

    @pytest.mark.asyncio
    async def test_store_via_daemon(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client, sample_rule
    ):
        """Test rule storage via daemon (ADR-002 primary path)."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema.store_rule(sample_rule)

        assert result["success"] is True
        assert result["fallback_mode"] == "daemon"
        assert result["rule_id"] == sample_rule.id
        mock_daemon_client.ingest_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_via_queue_fallback(
        self, mock_qdrant_client, mock_embedding_service, mock_state_manager, sample_rule
    ):
        """Test rule storage via queue when daemon unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            state_manager=mock_state_manager,
        )

        result = await schema.store_rule(sample_rule)

        assert result["success"] is True
        assert result["queued"] is True
        assert result["fallback_mode"] == "unified_queue"
        mock_state_manager.enqueue_unified.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_direct_fallback(
        self, mock_qdrant_client, mock_embedding_service, sample_rule
    ):
        """Test direct Qdrant fallback when both daemon and queue unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        result = await schema.store_rule(sample_rule)

        assert result["success"] is True
        assert result["fallback_mode"] == "direct_qdrant"
        assert "warning" in result


class TestDeleteRule:
    """Test delete_rule with daemon routing."""

    @pytest.mark.asyncio
    async def test_delete_via_daemon(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client
    ):
        """Test rule deletion via daemon (ADR-002 primary path)."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema.delete_rule("rule-123")

        assert result["success"] is True
        assert result["fallback_mode"] == "daemon"
        mock_daemon_client.delete_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_via_queue_fallback(
        self, mock_qdrant_client, mock_embedding_service, mock_state_manager
    ):
        """Test rule deletion via queue when daemon unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            state_manager=mock_state_manager,
        )

        result = await schema.delete_rule("rule-123")

        assert result["success"] is True
        assert result["queued"] is True
        assert result["fallback_mode"] == "unified_queue"
        mock_state_manager.enqueue_unified.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_direct_fallback(
        self, mock_qdrant_client, mock_embedding_service
    ):
        """Test direct Qdrant fallback when both daemon and queue unavailable."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
        )

        result = await schema.delete_rule("rule-123")

        assert result["success"] is True
        assert result["fallback_mode"] == "direct_qdrant"


class TestUpdateRule:
    """Test update_rule delegates to store_rule."""

    @pytest.mark.asyncio
    async def test_update_delegates_to_store(
        self, mock_qdrant_client, mock_embedding_service, mock_daemon_client, sample_rule
    ):
        """Test that update_rule delegates to store_rule."""
        schema = MemoryCollectionSchema(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            daemon_client=mock_daemon_client,
        )

        result = await schema.update_rule(sample_rule)

        # Should use same path as store_rule
        assert result["success"] is True
        assert result["fallback_mode"] == "daemon"
        mock_daemon_client.ingest_text.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
