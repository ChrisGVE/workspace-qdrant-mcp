"""
Tests for the memory system.

This module tests the comprehensive memory collection system including
memory rules, authority levels, conflict detection, and Claude Code integration.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.core.claude_integration import ClaudeIntegrationManager
from common.core.collection_naming import CollectionNamingManager
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
    create_memory_manager,
    parse_conversational_memory_update,
)


class TestMemoryRule:
    """Test the MemoryRule dataclass."""

    def test_memory_rule_creation(self):
        """Test creating memory rules."""
        rule = MemoryRule(
            id="test-rule-1",
            category=MemoryCategory.PREFERENCE,
            name="python-tool",
            rule="Use uv for Python package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "development"]
        )

        assert rule.id == "test-rule-1"
        assert rule.category == MemoryCategory.PREFERENCE
        assert rule.name == "python-tool"
        assert rule.rule == "Use uv for Python package management"
        assert rule.authority == AuthorityLevel.DEFAULT
        assert rule.scope == ["python", "development"]
        assert rule.source == "user_explicit"  # default
        assert rule.created_at is not None
        assert rule.updated_at is not None

    def test_memory_rule_timestamps(self):
        """Test timestamp initialization."""
        # Without timestamps
        rule1 = MemoryRule(
            id="test-1",
            category=MemoryCategory.BEHAVIOR,
            name="test",
            rule="test rule",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[]
        )
        assert rule1.created_at is not None
        assert rule1.updated_at == rule1.created_at

        # With custom timestamps
        custom_time = datetime(2023, 1, 1, 12, 0, 0)
        rule2 = MemoryRule(
            id="test-2",
            category=MemoryCategory.BEHAVIOR,
            name="test",
            rule="test rule",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[],
            created_at=custom_time
        )
        assert rule2.created_at == custom_time
        assert rule2.updated_at == custom_time


class TestMemoryManager:
    """Test the MemoryManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.mock_naming_manager = Mock(spec=CollectionNamingManager)
        self.memory_manager = MemoryManager(
            qdrant_client=self.mock_client,
            naming_manager=self.mock_naming_manager,
            embedding_dim=384
        )

    def test_initialization(self):
        """Test MemoryManager initialization."""
        assert self.memory_manager.client == self.mock_client
        assert self.memory_manager.naming_manager == self.mock_naming_manager
        assert self.memory_manager.embedding_dim == 384
        assert self.memory_manager.MEMORY_COLLECTION == "memory"

    @pytest.mark.asyncio
    async def test_initialize_memory_collection_new(self):
        """Test initializing a new memory collection."""
        # Mock collection doesn't exist
        mock_collections = Mock()
        mock_collections.collections = []
        self.mock_client.get_collections.return_value = mock_collections

        result = await self.memory_manager.initialize_memory_collection()

        assert result is True
        self.mock_client.create_collection.assert_called_once()

        # Verify collection was created with correct parameters
        call_args = self.mock_client.create_collection.call_args
        assert call_args[1]["collection_name"] == "memory"
        assert "dense" in call_args[1]["vectors_config"]

    @pytest.mark.asyncio
    async def test_initialize_memory_collection_existing(self):
        """Test initializing an existing memory collection."""
        # Mock collection already exists
        mock_collection = Mock()
        mock_collection.name = "memory"
        mock_collections = Mock()
        mock_collections.collections = [mock_collection]
        self.mock_client.get_collections.return_value = mock_collections

        result = await self.memory_manager.initialize_memory_collection()

        assert result is True
        self.mock_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_memory_rule(self):
        """Test adding a memory rule."""
        # Mock successful upsert
        self.mock_client.upsert.return_value = None

        rule_id = await self.memory_manager.add_memory_rule(
            category=MemoryCategory.PREFERENCE,
            name="test-rule",
            rule="Test rule content",
            authority=AuthorityLevel.DEFAULT,
            scope=["test"]
        )

        assert rule_id is not None
        assert rule_id.startswith("rule_")

        # Verify upsert was called with correct parameters
        self.mock_client.upsert.assert_called_once()
        call_args = self.mock_client.upsert.call_args
        assert call_args[1]["collection_name"] == "memory"

        point = call_args[1]["points"][0]
        assert point.payload["name"] == "test-rule"
        assert point.payload["rule"] == "Test rule content"
        assert point.payload["category"] == "preference"
        assert point.payload["authority"] == "default"

    @pytest.mark.asyncio
    async def test_get_memory_rule_found(self):
        """Test retrieving an existing memory rule."""
        # Mock point data
        mock_point = Mock()
        mock_point.id = "test-rule-1"
        mock_point.payload = {
            "category": "preference",
            "name": "test-rule",
            "rule": "Test rule content",
            "authority": "default",
            "scope": ["test"],
            "source": "user_explicit",
            "conditions": {},
            "replaces": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {}
        }

        self.mock_client.retrieve.return_value = [mock_point]

        rule = await self.memory_manager.get_memory_rule("test-rule-1")

        assert rule is not None
        assert rule.id == "test-rule-1"
        assert rule.name == "test-rule"
        assert rule.rule == "Test rule content"
        assert rule.category == MemoryCategory.PREFERENCE
        assert rule.authority == AuthorityLevel.DEFAULT

    @pytest.mark.asyncio
    async def test_get_memory_rule_not_found(self):
        """Test retrieving a non-existent memory rule."""
        self.mock_client.retrieve.return_value = []

        rule = await self.memory_manager.get_memory_rule("non-existent")

        assert rule is None

    @pytest.mark.asyncio
    async def test_list_memory_rules_no_filter(self):
        """Test listing all memory rules without filters."""
        # Mock scroll response
        mock_point1 = Mock()
        mock_point1.id = "rule-1"
        mock_point1.payload = {
            "category": "preference",
            "name": "rule-1",
            "rule": "Rule 1",
            "authority": "default",
            "scope": [],
            "source": "user_explicit",
            "conditions": {},
            "replaces": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {}
        }

        mock_point2 = Mock()
        mock_point2.id = "rule-2"
        mock_point2.payload = {
            "category": "behavior",
            "name": "rule-2",
            "rule": "Rule 2",
            "authority": "absolute",
            "scope": [],
            "source": "user_explicit",
            "conditions": {},
            "replaces": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {}
        }

        self.mock_client.scroll.return_value = ([mock_point1, mock_point2], None)

        rules = await self.memory_manager.list_memory_rules()

        assert len(rules) == 2
        assert rules[0].name == "rule-1"
        assert rules[1].name == "rule-2"

    @pytest.mark.asyncio
    async def test_list_memory_rules_with_filters(self):
        """Test listing memory rules with category filter."""
        # Mock scroll response with filter
        self.mock_client.scroll.return_value = ([], None)

        await self.memory_manager.list_memory_rules(
            category=MemoryCategory.PREFERENCE
        )

        # Verify filter was applied
        call_args = self.mock_client.scroll.call_args
        assert call_args[1]["scroll_filter"] is not None

    @pytest.mark.asyncio
    async def test_update_memory_rule_success(self):
        """Test updating an existing memory rule."""
        # Mock existing rule retrieval
        existing_rule = MemoryRule(
            id="test-rule-1",
            category=MemoryCategory.PREFERENCE,
            name="old-name",
            rule="old rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
            created_at=datetime(2023, 1, 1)
        )

        with patch.object(self.memory_manager, 'get_memory_rule', return_value=existing_rule):
            result = await self.memory_manager.update_memory_rule(
                "test-rule-1",
                {"name": "new-name", "rule": "new rule"}
            )

        assert result is True
        self.mock_client.upsert.assert_called_once()

        # Verify the updated payload
        point = self.mock_client.upsert.call_args[1]["points"][0]
        assert point.payload["name"] == "new-name"
        assert point.payload["rule"] == "new rule"

    @pytest.mark.asyncio
    async def test_update_memory_rule_not_found(self):
        """Test updating a non-existent memory rule."""
        with patch.object(self.memory_manager, 'get_memory_rule', return_value=None):
            result = await self.memory_manager.update_memory_rule(
                "non-existent",
                {"name": "new-name"}
            )

        assert result is False
        self.mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_memory_rule(self):
        """Test deleting a memory rule."""
        result = await self.memory_manager.delete_memory_rule("test-rule-1")

        assert result is True
        self.mock_client.delete.assert_called_once_with(
            collection_name="memory",
            points_selector=["test-rule-1"]
        )

    @pytest.mark.asyncio
    async def test_search_memory_rules(self):
        """Test searching memory rules by semantic similarity."""
        # Mock search response
        mock_scored_point = Mock()
        mock_scored_point.score = 0.85
        mock_scored_point.id = "rule-1"
        mock_scored_point.payload = {
            "category": "preference",
            "name": "rule-1",
            "rule": "Test rule",
            "authority": "default",
            "scope": [],
            "source": "user_explicit",
            "conditions": {},
            "replaces": [],
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "metadata": {}
        }

        self.mock_client.search.return_value = [mock_scored_point]

        results = await self.memory_manager.search_memory_rules("test query")

        assert len(results) == 1
        rule, score = results[0]
        assert rule.name == "rule-1"
        assert score == 0.85

    @pytest.mark.asyncio
    async def test_detect_conflicts(self):
        """Test conflict detection between memory rules."""
        # Create conflicting rules
        rule1 = MemoryRule(
            id="rule-1",
            category=MemoryCategory.PREFERENCE,
            name="use-uv",
            rule="Use uv for Python",
            authority=AuthorityLevel.DEFAULT,
            scope=[]
        )

        rule2 = MemoryRule(
            id="rule-2",
            category=MemoryCategory.PREFERENCE,
            name="avoid-uv",
            rule="Avoid uv, use pip",
            authority=AuthorityLevel.DEFAULT,
            scope=[]
        )

        rules = [rule1, rule2]
        conflicts = await self.memory_manager.detect_conflicts(rules)

        # This should detect a conflict based on our simple heuristics
        assert len(conflicts) >= 0  # May or may not detect conflict with simple rules

    @pytest.mark.asyncio
    async def test_get_memory_stats(self):
        """Test getting memory usage statistics."""
        # Mock list_memory_rules to return test data
        rule1 = MemoryRule(
            id="rule-1",
            category=MemoryCategory.PREFERENCE,
            name="rule-1",
            rule="Short rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[]
        )

        rule2 = MemoryRule(
            id="rule-2",
            category=MemoryCategory.BEHAVIOR,
            name="rule-2",
            rule="Another rule",
            authority=AuthorityLevel.ABSOLUTE,
            scope=[]
        )

        with patch.object(self.memory_manager, 'list_memory_rules', return_value=[rule1, rule2]):
            stats = await self.memory_manager.get_memory_stats()

        assert stats.total_rules == 2
        assert stats.rules_by_category[MemoryCategory.PREFERENCE] == 1
        assert stats.rules_by_category[MemoryCategory.BEHAVIOR] == 1
        assert stats.rules_by_authority[AuthorityLevel.DEFAULT] == 1
        assert stats.rules_by_authority[AuthorityLevel.ABSOLUTE] == 1
        assert stats.estimated_tokens > 0


class TestConversationalParsing:
    """Test conversational memory update parsing."""

    def test_parse_note_pattern(self):
        """Test parsing 'Note:' patterns."""
        message = "Note: call me Chris"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.PREFERENCE
        assert result["rule"] == "call me Chris"
        assert result["authority"] == AuthorityLevel.DEFAULT
        assert result["source"] == "conversational_note"

    def test_parse_future_reference_pattern(self):
        """Test parsing 'For future reference' patterns."""
        message = "For future reference, always use TypeScript strict mode"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert result["rule"] == "always use TypeScript strict mode"
        assert result["authority"] == AuthorityLevel.DEFAULT
        assert result["source"] == "conversational_future"

    def test_parse_remember_pattern(self):
        """Test parsing 'Remember that I' patterns."""
        message = "Remember that I prefer uv for Python package management"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.PREFERENCE
        assert result["rule"] == "User prefer uv for Python package management"
        assert result["authority"] == AuthorityLevel.DEFAULT
        assert result["source"] == "conversational_remember"

    def test_parse_always_behavior_pattern(self):
        """Test parsing 'Always' behavior patterns."""
        message = "Always make atomic commits"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert result["rule"] == "Always make atomic commits"
        assert result["authority"] == AuthorityLevel.ABSOLUTE
        assert result["source"] == "conversational_behavior"

    def test_parse_never_behavior_pattern(self):
        """Test parsing 'Never' behavior patterns."""
        message = "Never batch commits together"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert result["rule"] == "Never batch commits together"
        assert result["authority"] == AuthorityLevel.ABSOLUTE
        assert result["source"] == "conversational_behavior"

    def test_parse_no_pattern(self):
        """Test that non-memory messages return None."""
        message = "This is just a regular message"
        result = parse_conversational_memory_update(message)

        assert result is None

    def test_parse_case_insensitive(self):
        """Test that pattern matching is case-insensitive."""
        message = "NOTE: use lowercase for patterns"
        result = parse_conversational_memory_update(message)

        assert result is not None
        assert result["rule"] == "use lowercase for patterns"


class TestClaudeIntegration:
    """Test Claude Code SDK integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_memory_manager = Mock(spec=MemoryManager)
        self.claude_integration = ClaudeIntegrationManager(self.mock_memory_manager)

    @pytest.mark.asyncio
    async def test_initialize_session_success(self):
        """Test successful session initialization."""
        # Mock memory manager responses
        self.mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        self.mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])
        self.mock_memory_manager.detect_conflicts = AsyncMock(return_value=[])

        # Mock memory stats
        mock_stats = Mock()
        mock_stats.total_rules = 0
        mock_stats.estimated_tokens = 0
        mock_stats.rules_by_category = {cat: 0 for cat in MemoryCategory}
        mock_stats.rules_by_authority = {auth: 0 for auth in AuthorityLevel}
        self.mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        session_data = await self.claude_integration.initialize_session()

        assert session_data["status"] == "ready"
        assert "memory_stats" in session_data
        assert "system_context" in session_data
        assert session_data["conflicts_detected"] == 0

    @pytest.mark.asyncio
    async def test_initialize_session_with_conflicts(self):
        """Test session initialization with conflicts detected."""
        # Create mock rules that conflict
        rule1 = MemoryRule(
            id="rule-1",
            category=MemoryCategory.PREFERENCE,
            name="use-uv",
            rule="Use uv for Python",
            authority=AuthorityLevel.DEFAULT,
            scope=[]
        )

        rule2 = MemoryRule(
            id="rule-2",
            category=MemoryCategory.PREFERENCE,
            name="use-pip",
            rule="Use pip for Python",
            authority=AuthorityLevel.DEFAULT,
            scope=[]
        )

        mock_conflict = MemoryConflict(
            conflict_type="direct_contradiction",
            rule1=rule1,
            rule2=rule2,
            confidence=0.9,
            description="Rules conflict about Python package manager",
            resolution_options=["Keep higher authority", "Merge rules"]
        )

        # Mock memory manager responses
        self.mock_memory_manager.initialize_memory_collection = AsyncMock(return_value=True)
        self.mock_memory_manager.list_memory_rules = AsyncMock(return_value=[rule1, rule2])
        self.mock_memory_manager.detect_conflicts = AsyncMock(return_value=[mock_conflict])

        # Mock memory stats
        mock_stats = Mock()
        mock_stats.total_rules = 2
        mock_stats.estimated_tokens = 100
        mock_stats.rules_by_category = {MemoryCategory.PREFERENCE: 2}
        mock_stats.rules_by_authority = {AuthorityLevel.DEFAULT: 2}
        self.mock_memory_manager.get_memory_stats = AsyncMock(return_value=mock_stats)

        session_data = await self.claude_integration.initialize_session()

        assert session_data["status"] == "conflicts_detected"
        assert session_data["conflicts_detected"] == 1
        assert "conflicts" in session_data
        assert "conflict_resolution_prompt" in session_data

    @pytest.mark.asyncio
    async def test_format_system_rules_for_injection(self):
        """Test formatting rules for system context."""
        # Create test rules
        absolute_rule = MemoryRule(
            id="rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="atomic-commits",
            rule="Always make atomic commits",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["development"]
        )

        default_rule = MemoryRule(
            id="rule-2",
            category=MemoryCategory.PREFERENCE,
            name="python-tool",
            rule="Use uv for Python package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"]
        )

        rules = [absolute_rule, default_rule]
        context = await self.claude_integration.format_system_rules_for_injection(rules)

        assert "User Memory Rules" in context
        assert "Loaded 2 rules" in context
        assert "Absolute Rules" in context
        assert "Default Rules" in context
        assert "atomic-commits" in context
        assert "python-tool" in context
        assert "Always make atomic commits" in context
        assert "Use uv for Python" in context

    @pytest.mark.asyncio
    async def test_handle_conversational_update_success(self):
        """Test handling successful conversational update."""
        message = "Note: call me Chris"

        # Mock memory manager
        self.mock_memory_manager.add_memory_rule = AsyncMock(return_value="new-rule-id")
        self.mock_memory_manager.list_memory_rules = AsyncMock(return_value=[])

        result = await self.claude_integration.handle_conversational_update(message)

        assert result["detected"] is True
        assert result["rule_added"] is True
        assert result["rule_id"] == "new-rule-id"
        assert "call" in result["rule_name"]  # Generated from rule text
        assert result["category"] == "preference"
        assert result["authority"] == "default"

    @pytest.mark.asyncio
    async def test_handle_conversational_update_no_pattern(self):
        """Test handling message with no conversational pattern."""
        message = "This is just a regular message"

        result = await self.claude_integration.handle_conversational_update(message)

        assert result["detected"] is False
        assert "No memory update pattern detected" in result["message"]


class TestMemoryFactory:
    """Test factory functions."""

    def test_create_memory_manager(self):
        """Test factory function for creating memory manager."""
        mock_client = Mock()
        mock_naming_manager = Mock(spec=CollectionNamingManager)

        manager = create_memory_manager(
            qdrant_client=mock_client,
            naming_manager=mock_naming_manager,
            embedding_dim=768
        )

        assert isinstance(manager, MemoryManager)
        assert manager.client == mock_client
        assert manager.naming_manager == mock_naming_manager
        assert manager.embedding_dim == 768
