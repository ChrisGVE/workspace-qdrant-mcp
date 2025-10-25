"""
Tests for memory schema system.

Comprehensive tests for memory rule storage schema including Qdrant collection
management, rule serialization/deserialization, search operations, and versioning.
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from workspace_qdrant_mcp.memory.schema import MemoryCollectionSchema
from workspace_qdrant_mcp.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


class TestMemoryCollectionSchemaInit:
    """Test memory collection schema initialization."""

    def test_init_basic(self):
        """Test basic schema initialization."""
        mock_client = Mock()
        mock_embedding = Mock()

        schema = MemoryCollectionSchema(mock_client, mock_embedding)

        assert schema.client == mock_client
        assert schema.embedding_service == mock_embedding
        assert schema.collection_name == "memory_rules"
        assert schema.vector_size == 384

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_client = Mock()
        mock_embedding = Mock()

        schema = MemoryCollectionSchema(
            mock_client,
            mock_embedding,
            collection_name="custom_memory",
            vector_size=768
        )

        assert schema.collection_name == "custom_memory"
        assert schema.vector_size == 768


class TestCollectionManagement:
    """Test Qdrant collection management."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_new(self, schema_setup):
        """Test creating new collection when it doesn't exist."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock collection doesn't exist
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection = AsyncMock()

        result = await schema.ensure_collection_exists()

        assert result is True
        mock_client.create_collection.assert_called_once()

        # Verify collection creation parameters
        call_args = mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "memory_rules"
        assert call_args[1]["vectors_config"].size == 384

    @pytest.mark.asyncio
    async def test_ensure_collection_exists_already_exists(self, schema_setup):
        """Test when collection already exists."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock collection exists
        existing_collection = Mock()
        existing_collection.name = "memory_rules"
        mock_client.get_collections.return_value = Mock(collections=[existing_collection])

        result = await schema.ensure_collection_exists()

        assert result is True
        mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_collection_creation_failure(self, schema_setup):
        """Test handling of collection creation failure."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock collection doesn't exist and creation fails
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection = AsyncMock(side_effect=Exception("Creation failed"))

        result = await schema.ensure_collection_exists()

        assert result is False

    @pytest.mark.asyncio
    async def test_collection_info_retrieval(self, schema_setup):
        """Test retrieving collection information."""
        schema, mock_client, mock_embedding = schema_setup

        mock_info = Mock()
        mock_info.points_count = 150
        mock_info.segments_count = 3
        mock_client.get_collection.return_value = mock_info

        info = await schema.get_collection_info()

        assert info == mock_info
        mock_client.get_collection.assert_called_once_with("memory_rules")

    @pytest.mark.asyncio
    async def test_collection_stats(self, schema_setup):
        """Test getting collection statistics."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.count.return_value = Mock(count=75)

        count = await schema.get_rule_count()

        assert count == 75
        mock_client.count.assert_called_once_with(
            collection_name="memory_rules",
            exact=True
        )


class TestRuleStorage:
    """Test memory rule storage operations."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.generate_embeddings.return_value = [[0.1] * 384]  # Mock embedding
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_store_rule_success(self, schema_setup):
        """Test successful rule storage."""
        schema, mock_client, mock_embedding = schema_setup

        rule = MemoryRule(
            rule="Test rule for storage",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing"],
            tags=["test", "storage"]
        )

        mock_client.upsert = AsyncMock()

        result = await schema.store_rule(rule)

        assert result is True
        mock_client.upsert.assert_called_once()

        # Verify upsert call structure
        call_args = mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "memory_rules"
        points = call_args[1]["points"]
        assert len(points) == 1

        point = points[0]
        assert point.id == rule.id
        assert len(point.vector) == 384
        assert point.payload["rule"] == rule.rule
        assert point.payload["category"] == "behavior"

    @pytest.mark.asyncio
    async def test_store_rule_embedding_failure(self, schema_setup):
        """Test rule storage when embedding generation fails."""
        schema, mock_client, mock_embedding = schema_setup

        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Mock embedding failure
        mock_embedding.generate_embeddings.side_effect = Exception("Embedding failed")

        result = await schema.store_rule(rule)

        assert result is False
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_rule_qdrant_failure(self, schema_setup):
        """Test rule storage when Qdrant operation fails."""
        schema, mock_client, mock_embedding = schema_setup

        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Mock Qdrant failure
        mock_client.upsert = AsyncMock(side_effect=Exception("Qdrant failed"))

        result = await schema.store_rule(rule)

        assert result is False

    @pytest.mark.asyncio
    async def test_store_multiple_rules(self, schema_setup):
        """Test storing multiple rules at once."""
        schema, mock_client, mock_embedding = schema_setup

        rules = []
        for i in range(5):
            rules.append(MemoryRule(
                rule=f"Test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ))

        # Mock embeddings for all rules
        mock_embedding.generate_embeddings.return_value = [[0.1] * 384] * 5
        mock_client.upsert = AsyncMock()

        result = await schema.store_rules(rules)

        assert result is True
        mock_client.upsert.assert_called_once()

        # Should batch all rules in single call
        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 5


class TestRuleRetrieval:
    """Test memory rule retrieval operations."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_get_rule_exists(self, schema_setup):
        """Test retrieving existing rule by ID."""
        schema, mock_client, mock_embedding = schema_setup

        rule_id = str(uuid4())
        mock_payload = {
            "rule": "Test rule",
            "category": "behavior",
            "authority": "default",
            "scope": [],
            "tags": [],
            "source": "user_cli",
            "use_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "last_used": None
        }

        mock_record = Mock()
        mock_record.id = rule_id
        mock_record.payload = mock_payload

        mock_client.retrieve.return_value = [mock_record]

        rule = await schema.get_rule(rule_id)

        assert rule is not None
        assert isinstance(rule, MemoryRule)
        assert rule.id == rule_id
        assert rule.rule == "Test rule"
        assert rule.category == MemoryCategory.BEHAVIOR

    @pytest.mark.asyncio
    async def test_get_rule_not_found(self, schema_setup):
        """Test retrieving non-existent rule."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.retrieve.return_value = []

        rule = await schema.get_rule("nonexistent-id")

        assert rule is None

    @pytest.mark.asyncio
    async def test_get_rule_malformed_data(self, schema_setup):
        """Test handling malformed rule data."""
        schema, mock_client, mock_embedding = schema_setup

        rule_id = str(uuid4())
        mock_payload = {
            "rule": "Test rule",
            # Missing required fields
        }

        mock_record = Mock()
        mock_record.id = rule_id
        mock_record.payload = mock_payload

        mock_client.retrieve.return_value = [mock_record]

        # Should handle gracefully
        rule = await schema.get_rule(rule_id)
        assert rule is None

    @pytest.mark.asyncio
    async def test_list_all_rules_no_filters(self, schema_setup):
        """Test listing all rules without filters."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock rule records
        mock_records = []
        for i in range(3):
            mock_payload = {
                "rule": f"Test rule {i}",
                "category": "behavior" if i % 2 else "preference",
                "authority": "default",
                "scope": [],
                "tags": [],
                "source": "user_cli",
                "use_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": None,
                "last_used": None
            }

            mock_record = Mock()
            mock_record.id = str(uuid4())
            mock_record.payload = mock_payload
            mock_records.append(mock_record)

        mock_client.scroll.return_value = (mock_records, None)

        rules = await schema.list_all_rules()

        assert len(rules) == 3
        assert all(isinstance(rule, MemoryRule) for rule in rules)

        # Verify scroll call
        mock_client.scroll.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_rules_with_category_filter(self, schema_setup):
        """Test listing rules with category filter."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.scroll.return_value = ([], None)

        await schema.list_all_rules(
            category_filter=MemoryCategory.BEHAVIOR
        )

        # Verify filter was applied
        call_args = mock_client.scroll.call_args
        scroll_filter = call_args[1]["scroll_filter"]
        assert scroll_filter is not None

    @pytest.mark.asyncio
    async def test_list_rules_with_authority_filter(self, schema_setup):
        """Test listing rules with authority filter."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.scroll.return_value = ([], None)

        await schema.list_all_rules(
            authority_filter=AuthorityLevel.ABSOLUTE
        )

        call_args = mock_client.scroll.call_args
        scroll_filter = call_args[1]["scroll_filter"]
        assert scroll_filter is not None

    @pytest.mark.asyncio
    async def test_list_rules_pagination(self, schema_setup):
        """Test rule listing with pagination."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock paginated results
        first_batch = [Mock() for _ in range(10)]
        second_batch = [Mock() for _ in range(5)]

        mock_client.scroll.side_effect = [
            (first_batch, "next_page_token"),
            (second_batch, None)
        ]

        # Mock payload data
        for i, record in enumerate(first_batch + second_batch):
            record.id = str(uuid4())
            record.payload = {
                "rule": f"Rule {i}",
                "category": "behavior",
                "authority": "default",
                "scope": [],
                "tags": [],
                "source": "user_cli",
                "use_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": None,
                "last_used": None
            }

        rules = await schema.list_all_rules()

        assert len(rules) == 15
        assert mock_client.scroll.call_count == 2


class TestRuleSearch:
    """Test semantic search operations."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.generate_embeddings.return_value = [[0.5] * 384]
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_search_similar_rules(self, schema_setup):
        """Test semantic search for similar rules."""
        schema, mock_client, mock_embedding = schema_setup

        query_text = "Python testing best practices"

        # Mock search results
        mock_results = []
        for i in range(3):
            mock_payload = {
                "rule": f"Python test rule {i}",
                "category": "behavior",
                "authority": "default",
                "scope": ["python", "testing"],
                "tags": [],
                "source": "user_cli",
                "use_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": None,
                "last_used": None
            }

            mock_result = Mock()
            mock_result.id = str(uuid4())
            mock_result.payload = mock_payload
            mock_result.score = 0.9 - (i * 0.1)  # Decreasing similarity
            mock_results.append(mock_result)

        mock_client.search.return_value = mock_results

        rules = await schema.search_similar_rules(query_text, limit=5, threshold=0.7)

        assert len(rules) == 3
        assert all(isinstance(rule, MemoryRule) for rule in rules)

        # Verify search parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert call_args[1]["collection_name"] == "memory_rules"
        assert call_args[1]["limit"] == 5
        assert len(call_args[1]["query_vector"]) == 384

    @pytest.mark.asyncio
    async def test_search_with_threshold(self, schema_setup):
        """Test search with similarity threshold filtering."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock results with varying scores
        mock_results = []
        scores = [0.95, 0.85, 0.65, 0.45]  # Some above/below threshold

        for i, score in enumerate(scores):
            mock_payload = {
                "rule": f"Test rule {i}",
                "category": "behavior",
                "authority": "default",
                "scope": [],
                "tags": [],
                "source": "user_cli",
                "use_count": 0,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": None,
                "last_used": None
            }

            mock_result = Mock()
            mock_result.id = str(uuid4())
            mock_result.payload = mock_payload
            mock_result.score = score
            mock_results.append(mock_result)

        mock_client.search.return_value = mock_results

        rules = await schema.search_similar_rules(
            "test query",
            limit=10,
            threshold=0.7
        )

        # Should only return rules above threshold (0.95, 0.85)
        assert len(rules) == 2

    @pytest.mark.asyncio
    async def test_search_with_filters(self, schema_setup):
        """Test search with category/scope filters."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.search.return_value = []

        await schema.search_similar_rules(
            "test query",
            category_filter=MemoryCategory.BEHAVIOR,
            scope_filter=["python", "testing"]
        )

        # Verify filter was applied in search
        call_args = mock_client.search.call_args
        search_filter = call_args[1]["query_filter"]
        assert search_filter is not None

    @pytest.mark.asyncio
    async def test_search_embedding_failure(self, schema_setup):
        """Test search when embedding generation fails."""
        schema, mock_client, mock_embedding = schema_setup

        mock_embedding.generate_embeddings.side_effect = Exception("Embedding failed")

        rules = await schema.search_similar_rules("test query")

        assert rules == []
        mock_client.search.assert_not_called()


class TestRuleUpdates:
    """Test rule update operations."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.generate_embeddings.return_value = [[0.1] * 384]
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_update_rule_success(self, schema_setup):
        """Test successful rule update."""
        schema, mock_client, mock_embedding = schema_setup

        rule_id = str(uuid4())
        updates = {
            "rule": "Updated rule text",
            "authority": "absolute",
            "use_count": 5
        }

        # Mock existing rule retrieval
        existing_payload = {
            "rule": "Original rule text",
            "category": "behavior",
            "authority": "default",
            "scope": [],
            "tags": [],
            "source": "user_cli",
            "use_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "last_used": None
        }

        mock_record = Mock()
        mock_record.id = rule_id
        mock_record.payload = existing_payload
        mock_client.retrieve.return_value = [mock_record]

        mock_client.upsert = AsyncMock()

        result = await schema.update_rule(rule_id, updates)

        assert result is True
        mock_client.upsert.assert_called_once()

        # Verify updated payload
        call_args = mock_client.upsert.call_args
        updated_point = call_args[1]["points"][0]
        assert updated_point.payload["rule"] == "Updated rule text"
        assert updated_point.payload["authority"] == "absolute"
        assert updated_point.payload["use_count"] == 5
        assert updated_point.payload["updated_at"] is not None

    @pytest.mark.asyncio
    async def test_update_rule_not_found(self, schema_setup):
        """Test updating non-existent rule."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.retrieve.return_value = []

        result = await schema.update_rule("nonexistent-id", {"rule": "New text"})

        assert result is False
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_rule_text_reembedding(self, schema_setup):
        """Test that rule text updates trigger re-embedding."""
        schema, mock_client, mock_embedding = schema_setup

        rule_id = str(uuid4())
        updates = {"rule": "Completely new rule text"}

        # Mock existing rule
        existing_payload = {
            "rule": "Old rule text",
            "category": "behavior",
            "authority": "default",
            "scope": [],
            "tags": [],
            "source": "user_cli",
            "use_count": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": None,
            "last_used": None
        }

        mock_record = Mock()
        mock_record.id = rule_id
        mock_record.payload = existing_payload
        mock_client.retrieve.return_value = [mock_record]

        result = await schema.update_rule(rule_id, updates)

        assert result is True
        # Should generate new embedding for updated text
        mock_embedding.generate_embeddings.assert_called_once()


class TestRuleDeletion:
    """Test rule deletion operations."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema with mocked dependencies."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_delete_rule_success(self, schema_setup):
        """Test successful rule deletion."""
        schema, mock_client, mock_embedding = schema_setup

        rule_id = str(uuid4())
        mock_client.delete.return_value = Mock(status="completed")

        result = await schema.delete_rule(rule_id)

        assert result is True
        mock_client.delete.assert_called_once_with(
            collection_name="memory_rules",
            points_selector=[rule_id]
        )

    @pytest.mark.asyncio
    async def test_delete_rule_failure(self, schema_setup):
        """Test rule deletion failure."""
        schema, mock_client, mock_embedding = schema_setup

        mock_client.delete = AsyncMock(side_effect=Exception("Delete failed"))

        result = await schema.delete_rule("some-rule-id")

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_multiple_rules(self, schema_setup):
        """Test deleting multiple rules."""
        schema, mock_client, mock_embedding = schema_setup

        rule_ids = [str(uuid4()) for _ in range(5)]
        mock_client.delete.return_value = Mock(status="completed")

        result = await schema.delete_rules(rule_ids)

        assert result is True
        mock_client.delete.assert_called_once_with(
            collection_name="memory_rules",
            points_selector=rule_ids
        )


class TestSerialization:
    """Test rule serialization and deserialization."""

    def test_serialize_rule_to_payload(self):
        """Test converting MemoryRule to Qdrant payload."""
        rule = MemoryRule(
            rule="Test serialization rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "testing"],
            tags=["serialization", "test"],
            source="conversation",
            use_count=3
        )

        schema = MemoryCollectionSchema(Mock(), Mock())
        payload = schema._rule_to_payload(rule)

        expected_payload = {
            "rule": "Test serialization rule",
            "category": "behavior",
            "authority": "absolute",
            "scope": ["python", "testing"],
            "tags": ["serialization", "test"],
            "source": "conversation",
            "use_count": 3,
            "created_at": rule.created_at.isoformat(),
            "updated_at": None,
            "last_used": None
        }

        assert payload == expected_payload

    def test_deserialize_payload_to_rule(self):
        """Test converting Qdrant payload to MemoryRule."""
        rule_id = str(uuid4())
        created_at = datetime.utcnow()

        payload = {
            "rule": "Test deserialization rule",
            "category": "preference",
            "authority": "default",
            "scope": ["javascript"],
            "tags": ["deserialization"],
            "source": "user_cli",
            "use_count": 1,
            "created_at": created_at.isoformat(),
            "updated_at": None,
            "last_used": None
        }

        schema = MemoryCollectionSchema(Mock(), Mock())
        rule = schema._payload_to_rule(rule_id, payload)

        assert isinstance(rule, MemoryRule)
        assert rule.id == rule_id
        assert rule.rule == "Test deserialization rule"
        assert rule.category == MemoryCategory.PREFERENCE
        assert rule.authority == AuthorityLevel.DEFAULT
        assert rule.scope == ["javascript"]
        assert rule.tags == ["deserialization"]
        assert rule.source == "user_cli"
        assert rule.use_count == 1
        assert rule.created_at == created_at

    def test_deserialize_malformed_payload(self):
        """Test handling malformed payload data."""
        rule_id = str(uuid4())

        # Missing required fields
        incomplete_payload = {
            "rule": "Incomplete rule",
            # Missing category, authority, etc.
        }

        schema = MemoryCollectionSchema(Mock(), Mock())
        rule = schema._payload_to_rule(rule_id, incomplete_payload)

        # Should handle gracefully and return None or raise appropriate error
        assert rule is None

    def test_serialize_rule_with_datetime_fields(self):
        """Test serialization with datetime fields."""
        now = datetime.utcnow()

        rule = MemoryRule(
            rule="Rule with dates",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Manually set datetime fields
        rule.created_at = now
        rule.updated_at = now
        rule.last_used = now

        schema = MemoryCollectionSchema(Mock(), Mock())
        payload = schema._rule_to_payload(rule)

        assert payload["created_at"] == now.isoformat()
        assert payload["updated_at"] == now.isoformat()
        assert payload["last_used"] == now.isoformat()


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def schema_setup(self):
        """Create schema for error testing."""
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        schema = MemoryCollectionSchema(mock_client, mock_embedding)
        return schema, mock_client, mock_embedding

    @pytest.mark.asyncio
    async def test_network_errors(self, schema_setup):
        """Test handling of network/connection errors."""
        schema, mock_client, mock_embedding = schema_setup

        # Mock network failure
        mock_client.scroll.side_effect = Exception("Network error")

        rules = await schema.list_all_rules()

        # Should handle gracefully
        assert rules == []

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, schema_setup):
        """Test concurrent schema operations."""
        schema, mock_client, mock_embedding = schema_setup

        # Create multiple rules for concurrent storage
        rules = []
        for i in range(10):
            rules.append(MemoryRule(
                rule=f"Concurrent rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ))

        mock_client.upsert = AsyncMock()

        # Store rules concurrently
        tasks = [schema.store_rule(rule) for rule in rules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert result is True

    @pytest.mark.asyncio
    async def test_large_rule_handling(self, schema_setup):
        """Test handling of very large rules."""
        schema, mock_client, mock_embedding = schema_setup

        # Create rule with very long text
        large_rule = MemoryRule(
            rule="x" * 10000,  # Very large rule text
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        mock_client.upsert = AsyncMock()

        result = await schema.store_rule(large_rule)

        # Should handle large rules (implementation dependent)
        assert isinstance(result, bool)
