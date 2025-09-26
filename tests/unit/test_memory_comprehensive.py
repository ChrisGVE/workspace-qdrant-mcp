"""
Comprehensive unit tests for memory.py module.

This test file provides complete coverage for the memory system functionality,
including user preferences, LLM behavioral rules, agent library management,
and conversational memory updates.

Target: src/python/common/core/memory.py (2,626 lines, 0% coverage)
Goal: Achieve >90% test coverage with comprehensive mocking.
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum
import pytest

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

try:
    from common.core.memory import (
        MemoryManager,
        MemoryEntry,
        MemoryType,
        AuthorityLevel,
        AgentDefinition,
        ConversationalMemoryProcessor,
        MemoryConflictDetector,
        MemoryOptimizer,
        MemorySessionInitializer,
        MemorySearchResult,
        MemoryStats,
        BehavioralRule,
        UserPreference,
        MemoryCollectionManager,
        MemoryVectorStore,
        MemoryAnalytics,
        ConflictResolutionStrategy
    )
    from common.core.collection_naming import CollectionNamingManager, CollectionType
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class MockQdrantClient:
    """Mock Qdrant client for testing memory operations."""

    def __init__(self):
        self.collections = {}
        self.points = {}
        self.search_responses = []
        self.scroll_responses = []
        self.create_collection_calls = []
        self.upsert_calls = []
        self.delete_calls = []
        self.search_calls = []

    async def create_collection(self, collection_name, vectors_config, **kwargs):
        """Mock collection creation."""
        self.create_collection_calls.append({
            "collection_name": collection_name,
            "vectors_config": vectors_config,
            "kwargs": kwargs
        })
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "points_count": 0
        }

    async def collection_exists(self, collection_name):
        """Mock collection existence check."""
        return collection_name in self.collections

    async def get_collection(self, collection_name):
        """Mock get collection info."""
        if collection_name not in self.collections:
            raise Exception(f"Collection {collection_name} not found")
        return Mock(
            points_count=self.collections[collection_name]["points_count"],
            vectors_count=self.collections[collection_name]["points_count"]
        )

    async def upsert(self, collection_name, points):
        """Mock point upsert."""
        self.upsert_calls.append({
            "collection_name": collection_name,
            "points": points
        })

        if collection_name not in self.points:
            self.points[collection_name] = {}

        for point in points:
            self.points[collection_name][point.id] = point

        self.collections[collection_name]["points_count"] = len(self.points[collection_name])

    async def search(self, collection_name, query_vector=None, query_filter=None,
                    limit=10, with_payload=True, **kwargs):
        """Mock search operation."""
        self.search_calls.append({
            "collection_name": collection_name,
            "query_vector": query_vector,
            "query_filter": query_filter,
            "limit": limit,
            "kwargs": kwargs
        })

        if self.search_responses:
            return self.search_responses.pop(0)

        # Default mock search results
        mock_points = []
        collection_points = self.points.get(collection_name, {})

        for i, (point_id, point) in enumerate(list(collection_points.items())[:limit]):
            mock_points.append(Mock(
                id=point_id,
                score=0.9 - (i * 0.1),
                payload=point.payload if hasattr(point, 'payload') else {}
            ))

        return mock_points

    async def scroll(self, collection_name, scroll_filter=None, limit=100, **kwargs):
        """Mock scroll operation."""
        if self.scroll_responses:
            return self.scroll_responses.pop(0)

        collection_points = self.points.get(collection_name, {})
        points = list(collection_points.values())[:limit]

        return (points, None)  # (points, next_page_offset)

    async def delete(self, collection_name, points_selector):
        """Mock delete operation."""
        self.delete_calls.append({
            "collection_name": collection_name,
            "points_selector": points_selector
        })

        if hasattr(points_selector, 'points'):
            # Delete specific points
            collection_points = self.points.get(collection_name, {})
            for point_id in points_selector.points:
                if point_id in collection_points:
                    del collection_points[point_id]

    def set_search_response(self, response):
        """Set predetermined search response."""
        self.search_responses.append(response)

    def set_scroll_response(self, response):
        """Set predetermined scroll response."""
        self.scroll_responses.append(response)


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, dimension=384):
        self.dimension = dimension

    async def embed_text(self, text: str):
        """Mock text embedding."""
        # Return deterministic embedding based on text hash
        text_hash = hash(text) % 1000000
        return [float((text_hash + i) % 100) / 100.0 for i in range(self.dimension)]

    async def embed_batch(self, texts: List[str]):
        """Mock batch embedding."""
        return [await self.embed_text(text) for text in texts]

    def get_dimension(self):
        """Mock get embedding dimension."""
        return self.dimension


class MockCollectionNamingManager:
    """Mock collection naming manager."""

    def __init__(self):
        self.validation_results = {}

    def build_collection_name(self, collection_type: CollectionType, project_name: str = None,
                            suffix: str = None):
        """Mock collection name building."""
        if project_name and suffix:
            return f"{project_name}-{suffix}"
        elif project_name:
            return f"{project_name}-{collection_type.value}"
        else:
            return collection_type.value

    def validate_collection_name(self, name: str):
        """Mock collection name validation."""
        return self.validation_results.get(name, True)

    def set_validation_result(self, name: str, result: bool):
        """Set validation result for testing."""
        self.validation_results[name] = result


@pytest.fixture
def mock_qdrant_client():
    """Fixture providing mock Qdrant client."""
    return MockQdrantClient()


@pytest.fixture
def mock_embedding_service():
    """Fixture providing mock embedding service."""
    return MockEmbeddingService()


@pytest.fixture
def mock_collection_naming_manager():
    """Fixture providing mock collection naming manager."""
    return MockCollectionNamingManager()


@pytest.fixture
def sample_memory_entry():
    """Fixture providing sample memory entry."""
    return MemoryEntry(
        id="test-entry-1",
        content="Always use uv for Python package management",
        memory_type=MemoryType.USER_PREFERENCE,
        authority_level=AuthorityLevel.DEFAULT,
        tags=["python", "package-management"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_agent_definition():
    """Fixture providing sample agent definition."""
    return AgentDefinition(
        name="test-agent",
        description="A test agent for unit testing",
        capabilities=["testing", "mocking", "validation"],
        cost_per_task=0.01,
        specialization="testing",
        availability=True,
        version="1.0.0"
    )


class TestMemoryEntry:
    """Test MemoryEntry class functionality."""

    def test_init_basic(self):
        """Test basic MemoryEntry initialization."""
        entry = MemoryEntry(
            id="test1",
            content="Test content",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT
        )

        assert entry.id == "test1"
        assert entry.content == "Test content"
        assert entry.memory_type == MemoryType.USER_PREFERENCE
        assert entry.authority_level == AuthorityLevel.DEFAULT
        assert entry.tags == []
        assert isinstance(entry.created_at, datetime)
        assert isinstance(entry.updated_at, datetime)

    def test_init_with_all_fields(self):
        """Test MemoryEntry initialization with all fields."""
        created_time = datetime.now(timezone.utc)
        updated_time = created_time + timedelta(hours=1)
        metadata = {"key": "value"}

        entry = MemoryEntry(
            id="test2",
            content="Test content with metadata",
            memory_type=MemoryType.BEHAVIORAL_RULE,
            authority_level=AuthorityLevel.ABSOLUTE,
            tags=["tag1", "tag2"],
            metadata=metadata,
            created_at=created_time,
            updated_at=updated_time,
            version=2
        )

        assert entry.id == "test2"
        assert entry.content == "Test content with metadata"
        assert entry.memory_type == MemoryType.BEHAVIORAL_RULE
        assert entry.authority_level == AuthorityLevel.ABSOLUTE
        assert entry.tags == ["tag1", "tag2"]
        assert entry.metadata == metadata
        assert entry.created_at == created_time
        assert entry.updated_at == updated_time
        assert entry.version == 2

    def test_to_dict(self):
        """Test MemoryEntry to_dict conversion."""
        entry = MemoryEntry(
            id="test3",
            content="Test content",
            memory_type=MemoryType.AGENT_LIBRARY,
            authority_level=AuthorityLevel.DEFAULT,
            tags=["test"]
        )

        entry_dict = entry.to_dict()

        assert entry_dict["id"] == "test3"
        assert entry_dict["content"] == "Test content"
        assert entry_dict["memory_type"] == "AGENT_LIBRARY"
        assert entry_dict["authority_level"] == "DEFAULT"
        assert entry_dict["tags"] == ["test"]
        assert "created_at" in entry_dict
        assert "updated_at" in entry_dict

    def test_from_dict(self):
        """Test MemoryEntry from_dict creation."""
        entry_dict = {
            "id": "test4",
            "content": "Test from dict",
            "memory_type": "USER_PREFERENCE",
            "authority_level": "ABSOLUTE",
            "tags": ["from", "dict"],
            "version": 1,
            "metadata": {"source": "dict"}
        }

        entry = MemoryEntry.from_dict(entry_dict)

        assert entry.id == "test4"
        assert entry.content == "Test from dict"
        assert entry.memory_type == MemoryType.USER_PREFERENCE
        assert entry.authority_level == AuthorityLevel.ABSOLUTE
        assert entry.tags == ["from", "dict"]
        assert entry.version == 1
        assert entry.metadata == {"source": "dict"}

    def test_equality(self):
        """Test MemoryEntry equality comparison."""
        entry1 = MemoryEntry("test", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        entry2 = MemoryEntry("test", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        entry3 = MemoryEntry("different", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)

        assert entry1 == entry2
        assert entry1 != entry3

    def test_hash(self):
        """Test MemoryEntry hash functionality."""
        entry1 = MemoryEntry("test", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        entry2 = MemoryEntry("test", "different", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)

        # Should hash based on ID
        assert hash(entry1) == hash(entry2)

    def test_update_content(self):
        """Test updating memory entry content."""
        entry = MemoryEntry("test", "original", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        original_updated = entry.updated_at

        # Wait a tiny bit to ensure timestamp difference
        import time
        time.sleep(0.001)

        entry.update_content("updated content")

        assert entry.content == "updated content"
        assert entry.updated_at > original_updated
        assert entry.version == 2


class TestAgentDefinition:
    """Test AgentDefinition class functionality."""

    def test_init_basic(self):
        """Test basic AgentDefinition initialization."""
        agent = AgentDefinition(
            name="test-agent",
            description="Test agent",
            capabilities=["test"],
            cost_per_task=0.05
        )

        assert agent.name == "test-agent"
        assert agent.description == "Test agent"
        assert agent.capabilities == ["test"]
        assert agent.cost_per_task == 0.05
        assert agent.specialization == ""
        assert agent.availability is True
        assert agent.version == "1.0.0"

    def test_init_with_all_fields(self):
        """Test AgentDefinition with all fields."""
        metadata = {"model": "gpt-4", "provider": "openai"}

        agent = AgentDefinition(
            name="advanced-agent",
            description="Advanced test agent",
            capabilities=["analysis", "generation"],
            cost_per_task=0.10,
            specialization="data-analysis",
            availability=False,
            version="2.1.0",
            metadata=metadata
        )

        assert agent.name == "advanced-agent"
        assert agent.description == "Advanced test agent"
        assert agent.capabilities == ["analysis", "generation"]
        assert agent.cost_per_task == 0.10
        assert agent.specialization == "data-analysis"
        assert agent.availability is False
        assert agent.version == "2.1.0"
        assert agent.metadata == metadata

    def test_to_dict(self):
        """Test AgentDefinition to_dict conversion."""
        agent = AgentDefinition(
            name="dict-agent",
            description="Agent for dict test",
            capabilities=["dict"],
            cost_per_task=0.01
        )

        agent_dict = agent.to_dict()

        assert agent_dict["name"] == "dict-agent"
        assert agent_dict["description"] == "Agent for dict test"
        assert agent_dict["capabilities"] == ["dict"]
        assert agent_dict["cost_per_task"] == 0.01
        assert agent_dict["availability"] is True
        assert agent_dict["version"] == "1.0.0"

    def test_from_dict(self):
        """Test AgentDefinition from_dict creation."""
        agent_dict = {
            "name": "from-dict-agent",
            "description": "Created from dict",
            "capabilities": ["parsing"],
            "cost_per_task": 0.02,
            "specialization": "parsing",
            "availability": True,
            "version": "1.5.0"
        }

        agent = AgentDefinition.from_dict(agent_dict)

        assert agent.name == "from-dict-agent"
        assert agent.description == "Created from dict"
        assert agent.capabilities == ["parsing"]
        assert agent.cost_per_task == 0.02
        assert agent.specialization == "parsing"
        assert agent.availability is True
        assert agent.version == "1.5.0"

    def test_is_available(self):
        """Test agent availability check."""
        available_agent = AgentDefinition(
            name="available", description="", capabilities=[], cost_per_task=0, availability=True
        )
        unavailable_agent = AgentDefinition(
            name="unavailable", description="", capabilities=[], cost_per_task=0, availability=False
        )

        assert available_agent.is_available() is True
        assert unavailable_agent.is_available() is False

    def test_has_capability(self):
        """Test agent capability checking."""
        agent = AgentDefinition(
            name="capable", description="",
            capabilities=["testing", "mocking", "validation"],
            cost_per_task=0
        )

        assert agent.has_capability("testing") is True
        assert agent.has_capability("mocking") is True
        assert agent.has_capability("nonexistent") is False

    def test_get_cost_estimate(self):
        """Test cost estimation for tasks."""
        agent = AgentDefinition(
            name="costly", description="", capabilities=[], cost_per_task=0.10
        )

        assert agent.get_cost_estimate(1) == 0.10
        assert agent.get_cost_estimate(5) == 0.50
        assert agent.get_cost_estimate(0) == 0.0


class TestMemoryManager:
    """Test MemoryManager main functionality."""

    def test_init_basic(self, mock_qdrant_client, mock_embedding_service, mock_collection_naming_manager):
        """Test basic MemoryManager initialization."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        assert manager.qdrant_client == mock_qdrant_client
        assert manager.embedding_service == mock_embedding_service
        assert manager.collection_naming_manager == mock_collection_naming_manager
        assert manager.collection_name is not None
        assert isinstance(manager.analytics, MemoryAnalytics)

    @pytest.mark.asyncio
    async def test_initialize_memory_collection(self, mock_qdrant_client, mock_embedding_service,
                                               mock_collection_naming_manager):
        """Test memory collection initialization."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Should have created memory collection
        assert len(mock_qdrant_client.create_collection_calls) == 1
        create_call = mock_qdrant_client.create_collection_calls[0]
        assert "memory" in create_call["collection_name"]

    @pytest.mark.asyncio
    async def test_store_memory_entry(self, mock_qdrant_client, mock_embedding_service,
                                     mock_collection_naming_manager, sample_memory_entry):
        """Test storing memory entry."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()
        stored_id = await manager.store_memory(sample_memory_entry)

        assert stored_id == sample_memory_entry.id
        assert len(mock_qdrant_client.upsert_calls) == 1

        upsert_call = mock_qdrant_client.upsert_calls[0]
        assert upsert_call["collection_name"] == manager.collection_name
        assert len(upsert_call["points"]) == 1

    @pytest.mark.asyncio
    async def test_search_memories(self, mock_qdrant_client, mock_embedding_service,
                                  mock_collection_naming_manager):
        """Test searching memories."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Set up mock search response
        mock_response = [
            Mock(
                id="mem1",
                score=0.95,
                payload={
                    "content": "Test memory",
                    "memory_type": "USER_PREFERENCE",
                    "authority_level": "DEFAULT",
                    "tags": ["test"],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            )
        ]
        mock_qdrant_client.set_search_response(mock_response)

        results = await manager.search_memories("test query", limit=10)

        assert len(results) == 1
        assert results[0].id == "mem1"
        assert results[0].score == 0.95
        assert results[0].entry.content == "Test memory"
        assert len(mock_qdrant_client.search_calls) == 1

    @pytest.mark.asyncio
    async def test_search_by_type(self, mock_qdrant_client, mock_embedding_service,
                                 mock_collection_naming_manager):
        """Test searching memories by type."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        results = await manager.search_by_type(MemoryType.USER_PREFERENCE, limit=5)

        # Should have performed search with type filter
        assert len(mock_qdrant_client.search_calls) == 1
        search_call = mock_qdrant_client.search_calls[0]
        assert search_call["limit"] == 5

    @pytest.mark.asyncio
    async def test_search_by_tags(self, mock_qdrant_client, mock_embedding_service,
                                 mock_collection_naming_manager):
        """Test searching memories by tags."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        results = await manager.search_by_tags(["python", "testing"], limit=10)

        # Should have performed search with tag filters
        assert len(mock_qdrant_client.search_calls) == 1

    @pytest.mark.asyncio
    async def test_get_memory_by_id(self, mock_qdrant_client, mock_embedding_service,
                                   mock_collection_naming_manager):
        """Test retrieving memory by ID."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Mock search response for ID lookup
        mock_response = [
            Mock(
                id="specific-id",
                score=1.0,
                payload={
                    "content": "Specific memory",
                    "memory_type": "BEHAVIORAL_RULE",
                    "authority_level": "ABSOLUTE",
                    "tags": [],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            )
        ]
        mock_qdrant_client.set_search_response(mock_response)

        entry = await manager.get_memory_by_id("specific-id")

        assert entry is not None
        assert entry.id == "specific-id"
        assert entry.content == "Specific memory"
        assert entry.memory_type == MemoryType.BEHAVIORAL_RULE

    @pytest.mark.asyncio
    async def test_get_memory_by_id_not_found(self, mock_qdrant_client, mock_embedding_service,
                                             mock_collection_naming_manager):
        """Test retrieving non-existent memory by ID."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Empty search response
        mock_qdrant_client.set_search_response([])

        entry = await manager.get_memory_by_id("nonexistent")

        assert entry is None

    @pytest.mark.asyncio
    async def test_update_memory(self, mock_qdrant_client, mock_embedding_service,
                                mock_collection_naming_manager, sample_memory_entry):
        """Test updating existing memory."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # First store the memory
        await manager.store_memory(sample_memory_entry)

        # Update the memory
        sample_memory_entry.content = "Updated content"
        updated_id = await manager.update_memory(sample_memory_entry)

        assert updated_id == sample_memory_entry.id
        assert sample_memory_entry.version == 2
        assert len(mock_qdrant_client.upsert_calls) == 2  # Store + update

    @pytest.mark.asyncio
    async def test_delete_memory(self, mock_qdrant_client, mock_embedding_service,
                                mock_collection_naming_manager):
        """Test deleting memory by ID."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        success = await manager.delete_memory("test-id")

        assert success is True
        assert len(mock_qdrant_client.delete_calls) == 1
        delete_call = mock_qdrant_client.delete_calls[0]
        assert delete_call["collection_name"] == manager.collection_name

    @pytest.mark.asyncio
    async def test_get_all_memories(self, mock_qdrant_client, mock_embedding_service,
                                   mock_collection_naming_manager):
        """Test retrieving all memories."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Set up mock scroll response
        mock_points = [
            Mock(
                id=f"mem{i}",
                payload={
                    "content": f"Memory {i}",
                    "memory_type": "USER_PREFERENCE",
                    "authority_level": "DEFAULT",
                    "tags": [],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ) for i in range(3)
        ]
        mock_qdrant_client.set_scroll_response((mock_points, None))

        memories = await manager.get_all_memories()

        assert len(memories) == 3
        for i, memory in enumerate(memories):
            assert memory.id == f"mem{i}"
            assert memory.content == f"Memory {i}"

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, mock_qdrant_client, mock_embedding_service,
                                   mock_collection_naming_manager):
        """Test getting memory statistics."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Mock collection info
        mock_collection_info = Mock(points_count=10, vectors_count=10)
        mock_qdrant_client.collections[manager.collection_name] = {
            "points_count": 10,
            "vectors_count": 10
        }

        stats = await manager.get_memory_stats()

        assert isinstance(stats, MemoryStats)
        assert stats.total_memories == 10
        assert stats.collection_size == 10

    def test_build_filters_basic(self, mock_qdrant_client, mock_embedding_service,
                                mock_collection_naming_manager):
        """Test basic filter building."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        # Test memory type filter
        type_filter = manager._build_type_filter(MemoryType.USER_PREFERENCE)
        assert type_filter is not None

        # Test authority filter
        authority_filter = manager._build_authority_filter(AuthorityLevel.ABSOLUTE)
        assert authority_filter is not None

        # Test tag filter
        tag_filter = manager._build_tag_filter(["python", "testing"])
        assert tag_filter is not None

    def test_build_filters_empty(self, mock_qdrant_client, mock_embedding_service,
                                mock_collection_naming_manager):
        """Test filter building with empty values."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        # Empty tag filter should return None
        empty_tag_filter = manager._build_tag_filter([])
        assert empty_tag_filter is None


class TestConversationalMemoryProcessor:
    """Test ConversationalMemoryProcessor functionality."""

    def test_init(self):
        """Test ConversationalMemoryProcessor initialization."""
        processor = ConversationalMemoryProcessor()

        assert processor.memory_triggers is not None
        assert len(processor.memory_triggers) > 0

    def test_detect_memory_intent_basic(self):
        """Test basic memory intent detection."""
        processor = ConversationalMemoryProcessor()

        # Positive cases
        assert processor.detect_memory_intent("Remember that I prefer TypeScript") is True
        assert processor.detect_memory_intent("Note: call me Chris") is True
        assert processor.detect_memory_intent("Please remember to use uv") is True

        # Negative cases
        assert processor.detect_memory_intent("What's the weather like?") is False
        assert processor.detect_memory_intent("Run the tests") is False

    def test_extract_memory_from_conversation(self):
        """Test memory extraction from conversation."""
        processor = ConversationalMemoryProcessor()

        text = "Remember that I prefer to use uv for Python package management"
        memory_entry = processor.extract_memory_from_conversation(text)

        assert memory_entry is not None
        assert "uv" in memory_entry.content
        assert "Python" in memory_entry.content
        assert memory_entry.memory_type == MemoryType.USER_PREFERENCE

    def test_extract_memory_behavioral_rule(self):
        """Test extracting behavioral rules."""
        processor = ConversationalMemoryProcessor()

        text = "Always make atomic commits when working with Git"
        memory_entry = processor.extract_memory_from_conversation(text)

        assert memory_entry is not None
        assert memory_entry.memory_type == MemoryType.BEHAVIORAL_RULE
        assert "atomic commits" in memory_entry.content

    def test_extract_memory_agent_related(self):
        """Test extracting agent-related memories."""
        processor = ConversationalMemoryProcessor()

        text = "Remember that the testing agent costs $0.05 per task"
        memory_entry = processor.extract_memory_from_conversation(text)

        assert memory_entry is not None
        assert memory_entry.memory_type == MemoryType.AGENT_LIBRARY
        assert "testing agent" in memory_entry.content

    def test_extract_memory_with_authority_level(self):
        """Test authority level detection."""
        processor = ConversationalMemoryProcessor()

        # Absolute authority
        absolute_text = "Never use global variables in Python code"
        absolute_memory = processor.extract_memory_from_conversation(absolute_text)
        assert absolute_memory.authority_level == AuthorityLevel.ABSOLUTE

        # Default authority
        default_text = "I prefer using pytest for testing"
        default_memory = processor.extract_memory_from_conversation(default_text)
        assert default_memory.authority_level == AuthorityLevel.DEFAULT

    def test_generate_tags(self):
        """Test tag generation from content."""
        processor = ConversationalMemoryProcessor()

        content = "Use uv for Python package management and virtual environments"
        tags = processor._generate_tags(content)

        assert "python" in [tag.lower() for tag in tags]
        assert len(tags) > 0

    def test_determine_memory_type(self):
        """Test memory type determination."""
        processor = ConversationalMemoryProcessor()

        # User preference
        pref_type = processor._determine_memory_type("I prefer using VS Code")
        assert pref_type == MemoryType.USER_PREFERENCE

        # Behavioral rule
        rule_type = processor._determine_memory_type("Always validate inputs")
        assert rule_type == MemoryType.BEHAVIORAL_RULE

        # Agent library
        agent_type = processor._determine_memory_type("The test agent handles automation")
        assert agent_type == MemoryType.AGENT_LIBRARY

    def test_determine_authority_level(self):
        """Test authority level determination."""
        processor = ConversationalMemoryProcessor()

        # Absolute
        absolute_level = processor._determine_authority_level("Never use eval() in Python")
        assert absolute_level == AuthorityLevel.ABSOLUTE

        # Default
        default_level = processor._determine_authority_level("I like using pytest")
        assert default_level == AuthorityLevel.DEFAULT


class TestMemoryConflictDetector:
    """Test MemoryConflictDetector functionality."""

    def test_init(self, mock_embedding_service):
        """Test MemoryConflictDetector initialization."""
        detector = MemoryConflictDetector(embedding_service=mock_embedding_service)

        assert detector.embedding_service == mock_embedding_service
        assert detector.similarity_threshold == 0.85

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflict(self, mock_embedding_service):
        """Test conflict detection with no conflicts."""
        detector = MemoryConflictDetector(embedding_service=mock_embedding_service)

        new_entry = MemoryEntry(
            id="new", content="Use pytest for testing",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT
        )

        existing_entries = [
            MemoryEntry(
                id="existing", content="Use uv for package management",
                memory_type=MemoryType.USER_PREFERENCE,
                authority_level=AuthorityLevel.DEFAULT
            )
        ]

        conflicts = await detector.detect_conflicts(new_entry, existing_entries)

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_detect_conflicts_with_conflict(self, mock_embedding_service):
        """Test conflict detection with semantic conflicts."""
        # Create a special mock that returns high similarity for similar content
        class ConflictMockEmbeddingService(MockEmbeddingService):
            async def embed_text(self, text: str):
                if "pytest" in text:
                    return [1.0] * self.dimension  # Same embedding for similar content
                elif "unittest" in text:
                    return [1.0] * self.dimension  # Same embedding (conflict)
                else:
                    return [0.0] * self.dimension

        mock_embedding = ConflictMockEmbeddingService()
        detector = MemoryConflictDetector(
            embedding_service=mock_embedding,
            similarity_threshold=0.99  # High threshold for exact match
        )

        new_entry = MemoryEntry(
            id="new", content="Use pytest for all testing",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT
        )

        existing_entries = [
            MemoryEntry(
                id="existing", content="Use unittest for testing",
                memory_type=MemoryType.USER_PREFERENCE,
                authority_level=AuthorityLevel.DEFAULT
            )
        ]

        conflicts = await detector.detect_conflicts(new_entry, existing_entries)

        assert len(conflicts) == 1
        assert conflicts[0].existing_entry.id == "existing"
        assert conflicts[0].new_entry.id == "new"

    def test_suggest_resolution_default_wins(self, mock_embedding_service):
        """Test conflict resolution suggestion - default authority wins."""
        detector = MemoryConflictDetector(embedding_service=mock_embedding_service)

        new_entry = MemoryEntry(
            id="new", content="Use pytest",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT
        )

        existing_entry = MemoryEntry(
            id="existing", content="Use unittest",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.ABSOLUTE
        )

        resolution = detector.suggest_resolution(new_entry, existing_entry)

        assert resolution.strategy == ConflictResolutionStrategy.KEEP_EXISTING
        assert resolution.reason is not None

    def test_suggest_resolution_newer_wins(self, mock_embedding_service):
        """Test conflict resolution - newer entry wins."""
        detector = MemoryConflictDetector(embedding_service=mock_embedding_service)

        older_time = datetime.now(timezone.utc) - timedelta(hours=1)
        newer_time = datetime.now(timezone.utc)

        new_entry = MemoryEntry(
            id="new", content="Use pytest",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT,
            created_at=newer_time
        )

        existing_entry = MemoryEntry(
            id="existing", content="Use unittest",
            memory_type=MemoryType.USER_PREFERENCE,
            authority_level=AuthorityLevel.DEFAULT,
            created_at=older_time
        )

        resolution = detector.suggest_resolution(new_entry, existing_entry)

        assert resolution.strategy == ConflictResolutionStrategy.REPLACE_EXISTING


class TestMemoryOptimizer:
    """Test MemoryOptimizer functionality."""

    def test_init(self):
        """Test MemoryOptimizer initialization."""
        optimizer = MemoryOptimizer()

        assert optimizer.max_memories_per_type is not None
        assert optimizer.cleanup_threshold_days > 0

    def test_optimize_memory_storage_basic(self):
        """Test basic memory storage optimization."""
        optimizer = MemoryOptimizer()

        memories = [
            MemoryEntry(f"mem{i}", f"Content {i}", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
            for i in range(5)
        ]

        optimized = optimizer.optimize_memory_storage(memories)

        assert len(optimized) <= len(memories)
        assert all(isinstance(mem, MemoryEntry) for mem in optimized)

    def test_deduplicate_memories(self):
        """Test memory deduplication."""
        optimizer = MemoryOptimizer()

        # Create duplicate memories
        mem1 = MemoryEntry("1", "Use pytest", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        mem2 = MemoryEntry("2", "Use pytest for testing", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
        mem3 = MemoryEntry("3", "Use uv", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)

        memories = [mem1, mem2, mem3]
        deduplicated = optimizer.deduplicate_memories(memories)

        # Should keep unique memories
        assert len(deduplicated) <= len(memories)

    def test_remove_stale_memories(self):
        """Test stale memory removal."""
        optimizer = MemoryOptimizer(cleanup_threshold_days=30)

        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=40)  # Older than threshold
        recent_time = now - timedelta(days=10)  # Within threshold

        old_memory = MemoryEntry(
            "old", "Old content", MemoryType.USER_PREFERENCE,
            AuthorityLevel.DEFAULT, created_at=old_time
        )

        recent_memory = MemoryEntry(
            "recent", "Recent content", MemoryType.USER_PREFERENCE,
            AuthorityLevel.DEFAULT, created_at=recent_time
        )

        memories = [old_memory, recent_memory]
        cleaned = optimizer.remove_stale_memories(memories)

        # Should only keep recent memory
        assert len(cleaned) == 1
        assert cleaned[0].id == "recent"

    def test_limit_memories_per_type(self):
        """Test limiting memories per type."""
        optimizer = MemoryOptimizer()
        optimizer.max_memories_per_type[MemoryType.USER_PREFERENCE] = 2

        memories = [
            MemoryEntry(f"pref{i}", f"Pref {i}", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)
            for i in range(5)
        ]

        limited = optimizer.limit_memories_per_type(memories)

        # Should only keep 2 user preferences
        pref_memories = [m for m in limited if m.memory_type == MemoryType.USER_PREFERENCE]
        assert len(pref_memories) <= 2

    def test_calculate_memory_score(self):
        """Test memory scoring for optimization."""
        optimizer = MemoryOptimizer()

        recent_absolute = MemoryEntry(
            "recent_abs", "Recent absolute", MemoryType.BEHAVIORAL_RULE,
            AuthorityLevel.ABSOLUTE, created_at=datetime.now(timezone.utc)
        )

        old_default = MemoryEntry(
            "old_def", "Old default", MemoryType.USER_PREFERENCE,
            AuthorityLevel.DEFAULT,
            created_at=datetime.now(timezone.utc) - timedelta(days=100)
        )

        recent_score = optimizer._calculate_memory_score(recent_absolute)
        old_score = optimizer._calculate_memory_score(old_default)

        # Recent absolute should score higher than old default
        assert recent_score > old_score


class TestMemoryAnalytics:
    """Test MemoryAnalytics functionality."""

    def test_init(self):
        """Test MemoryAnalytics initialization."""
        analytics = MemoryAnalytics()

        assert analytics.usage_stats is not None

    def test_track_memory_access(self):
        """Test memory access tracking."""
        analytics = MemoryAnalytics()

        analytics.track_memory_access("mem1", "search")
        analytics.track_memory_access("mem1", "retrieval")
        analytics.track_memory_access("mem2", "search")

        assert analytics.usage_stats["mem1"]["search"] == 1
        assert analytics.usage_stats["mem1"]["retrieval"] == 1
        assert analytics.usage_stats["mem2"]["search"] == 1

    def test_get_most_accessed_memories(self):
        """Test getting most accessed memories."""
        analytics = MemoryAnalytics()

        # Track different access patterns
        for _ in range(5):
            analytics.track_memory_access("popular", "search")

        for _ in range(2):
            analytics.track_memory_access("less_popular", "search")

        most_accessed = analytics.get_most_accessed_memories(limit=2)

        assert len(most_accessed) <= 2
        assert most_accessed[0][0] == "popular"  # (memory_id, access_count)
        assert most_accessed[0][1] == 5

    def test_get_memory_type_distribution(self):
        """Test memory type distribution analysis."""
        analytics = MemoryAnalytics()

        memories = [
            MemoryEntry("1", "Content 1", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT),
            MemoryEntry("2", "Content 2", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT),
            MemoryEntry("3", "Content 3", MemoryType.BEHAVIORAL_RULE, AuthorityLevel.DEFAULT),
        ]

        distribution = analytics.get_memory_type_distribution(memories)

        assert distribution[MemoryType.USER_PREFERENCE] == 2
        assert distribution[MemoryType.BEHAVIORAL_RULE] == 1
        assert distribution.get(MemoryType.AGENT_LIBRARY, 0) == 0

    def test_get_authority_distribution(self):
        """Test authority level distribution analysis."""
        analytics = MemoryAnalytics()

        memories = [
            MemoryEntry("1", "Content 1", MemoryType.USER_PREFERENCE, AuthorityLevel.ABSOLUTE),
            MemoryEntry("2", "Content 2", MemoryType.USER_PREFERENCE, AuthorityLevel.ABSOLUTE),
            MemoryEntry("3", "Content 3", MemoryType.BEHAVIORAL_RULE, AuthorityLevel.DEFAULT),
        ]

        distribution = analytics.get_authority_distribution(memories)

        assert distribution[AuthorityLevel.ABSOLUTE] == 2
        assert distribution[AuthorityLevel.DEFAULT] == 1

    def test_analyze_memory_trends(self):
        """Test memory trend analysis."""
        analytics = MemoryAnalytics()

        # Create memories with different timestamps
        now = datetime.now(timezone.utc)
        memories = []

        for i in range(5):
            memories.append(MemoryEntry(
                f"mem{i}", f"Content {i}", MemoryType.USER_PREFERENCE,
                AuthorityLevel.DEFAULT, created_at=now - timedelta(days=i*7)
            ))

        trends = analytics.analyze_memory_trends(memories, days=30)

        assert "creation_rate" in trends
        assert "peak_creation_day" in trends
        assert isinstance(trends["creation_rate"], (int, float))


class TestMemorySessionInitializer:
    """Test MemorySessionInitializer functionality."""

    @pytest.mark.asyncio
    async def test_init(self, mock_qdrant_client, mock_embedding_service, mock_collection_naming_manager):
        """Test MemorySessionInitializer initialization."""
        memory_manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        initializer = MemorySessionInitializer(memory_manager=memory_manager)

        assert initializer.memory_manager == memory_manager
        assert initializer.claude_code_integration is not None

    @pytest.mark.asyncio
    async def test_initialize_session_basic(self, mock_qdrant_client, mock_embedding_service,
                                           mock_collection_naming_manager):
        """Test basic session initialization."""
        memory_manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await memory_manager.initialize()

        initializer = MemorySessionInitializer(memory_manager=memory_manager)

        session_config = await initializer.initialize_session()

        assert session_config is not None
        assert "loaded_rules" in session_config
        assert "memory_stats" in session_config

    @pytest.mark.asyncio
    async def test_load_session_rules(self, mock_qdrant_client, mock_embedding_service,
                                     mock_collection_naming_manager):
        """Test loading session rules from memory."""
        memory_manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await memory_manager.initialize()

        # Mock memories for different rule types
        mock_scroll_response = [
            Mock(
                id="rule1",
                payload={
                    "content": "Always make atomic commits",
                    "memory_type": "BEHAVIORAL_RULE",
                    "authority_level": "ABSOLUTE",
                    "tags": ["git"],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            ),
            Mock(
                id="pref1",
                payload={
                    "content": "Use uv for Python",
                    "memory_type": "USER_PREFERENCE",
                    "authority_level": "DEFAULT",
                    "tags": ["python"],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            )
        ]
        mock_qdrant_client.set_scroll_response((mock_scroll_response, None))

        initializer = MemorySessionInitializer(memory_manager=memory_manager)
        rules = await initializer.load_session_rules()

        assert "behavioral_rules" in rules
        assert "user_preferences" in rules
        assert len(rules["behavioral_rules"]) >= 1
        assert len(rules["user_preferences"]) >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_memory_manager_connection_error(self, mock_embedding_service, mock_collection_naming_manager):
        """Test MemoryManager with connection errors."""
        # Create a mock client that raises exceptions
        error_client = Mock()
        error_client.create_collection = AsyncMock(side_effect=Exception("Connection failed"))
        error_client.collection_exists = AsyncMock(return_value=False)

        manager = MemoryManager(
            qdrant_client=error_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        # Should handle initialization error gracefully
        with pytest.raises(Exception):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_search_with_empty_collection(self, mock_qdrant_client, mock_embedding_service,
                                               mock_collection_naming_manager):
        """Test searching in empty collection."""
        manager = MemoryManager(
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service,
            collection_naming_manager=mock_collection_naming_manager
        )

        await manager.initialize()

        # Empty search response
        mock_qdrant_client.set_search_response([])

        results = await manager.search_memories("test query")

        assert results == []

    def test_memory_entry_invalid_data(self):
        """Test MemoryEntry with invalid data."""
        # Test with None values
        entry = MemoryEntry(None, None, None, None)

        assert entry.id is None
        assert entry.content is None
        # These should still have valid defaults or handle None gracefully

    def test_conversational_processor_edge_cases(self):
        """Test ConversationalMemoryProcessor with edge cases."""
        processor = ConversationalMemoryProcessor()

        # Empty text
        assert processor.detect_memory_intent("") is False
        assert processor.extract_memory_from_conversation("") is None

        # Very long text
        long_text = "Remember " + "x" * 10000
        result = processor.extract_memory_from_conversation(long_text)
        assert result is not None  # Should handle gracefully

        # Special characters
        special_text = "Remember: !@#$%^&*()_+-="
        result = processor.extract_memory_from_conversation(special_text)
        assert result is not None

    def test_conflict_detector_edge_cases(self, mock_embedding_service):
        """Test ConflictDetector with edge cases."""
        detector = MemoryConflictDetector(embedding_service=mock_embedding_service)

        # Empty existing entries
        new_entry = MemoryEntry("new", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)

        async def test_empty_existing():
            conflicts = await detector.detect_conflicts(new_entry, [])
            assert conflicts == []

        # Run async test
        asyncio.run(test_empty_existing())

    def test_optimizer_edge_cases(self):
        """Test MemoryOptimizer with edge cases."""
        optimizer = MemoryOptimizer()

        # Empty memory list
        result = optimizer.optimize_memory_storage([])
        assert result == []

        # Single memory
        single_memory = [MemoryEntry("1", "content", MemoryType.USER_PREFERENCE, AuthorityLevel.DEFAULT)]
        result = optimizer.optimize_memory_storage(single_memory)
        assert len(result) == 1

    def test_analytics_edge_cases(self):
        """Test MemoryAnalytics with edge cases."""
        analytics = MemoryAnalytics()

        # No tracked accesses
        most_accessed = analytics.get_most_accessed_memories()
        assert most_accessed == []

        # Empty memory list for distribution
        distribution = analytics.get_memory_type_distribution([])
        assert distribution == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])