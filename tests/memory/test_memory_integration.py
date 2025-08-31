"""
Integration tests for memory system.

These tests verify that the memory system components work together correctly.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.workspace_qdrant_mcp.core.config import Config
from src.workspace_qdrant_mcp.memory.conflict_detector import ConflictDetector
from src.workspace_qdrant_mcp.memory.manager import MemoryManager
from src.workspace_qdrant_mcp.memory.token_counter import TokenCounter, TokenUsage
from src.workspace_qdrant_mcp.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    MemoryCategory,
    MemoryContext,
    MemoryRule,
)


class TestMemoryManagerIntegration:
    """Test memory manager with mocked dependencies."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock(spec=Config)
        config.qdrant_url = "http://localhost:6333"
        config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        config.enable_memory_ai_analysis = False  # Disable AI analysis for tests
        config.max_memory_tokens = 5000
        return config

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        # Mock collection creation
        client.get_collections.return_value = Mock(collections=[])
        client.create_collection = AsyncMock()
        client.upsert = AsyncMock()
        client.retrieve = AsyncMock(return_value=[])
        client.delete = AsyncMock(return_value=Mock(status="completed"))
        client.search = AsyncMock(return_value=[])
        client.scroll = AsyncMock(return_value=([], None))
        client.count = AsyncMock(return_value=Mock(count=0))
        client.get_collection = AsyncMock(return_value=Mock(
            points_count=0,
            segments_count=0,
            payload_schema={}
        ))
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        service = AsyncMock()
        service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3] * 128])  # 384 dimensions
        return service

    @pytest.fixture
    async def memory_manager(self, mock_config, mock_qdrant_client, mock_embedding_service):
        """Create memory manager with mocked dependencies."""
        manager = MemoryManager(
            config=mock_config,
            qdrant_client=mock_qdrant_client,
            embedding_service=mock_embedding_service
        )
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initializes correctly."""
        assert memory_manager._initialized is True
        assert memory_manager.schema is not None
        assert memory_manager.conflict_detector is not None
        assert memory_manager.token_counter is not None
        assert memory_manager.claude_integration is not None

    @pytest.mark.asyncio
    async def test_add_and_retrieve_rule(self, memory_manager):
        """Test adding and retrieving a memory rule."""
        # Create test rule
        rule = MemoryRule(
            rule="Always use uv for Python package management",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            tags=["python", "tooling"]
        )

        # Mock storage success
        memory_manager.schema.store_rule = AsyncMock(return_value=True)
        memory_manager.schema.get_rule = AsyncMock(return_value=rule)

        # Add rule
        rule_id, conflicts = await memory_manager.add_rule(rule, check_conflicts=False)

        assert rule_id == rule.id
        assert conflicts == []
        memory_manager.schema.store_rule.assert_called_once()

        # Retrieve rule
        retrieved_rule = await memory_manager.get_rule(rule.id)

        assert retrieved_rule is not None
        assert retrieved_rule.rule == rule.rule
        assert retrieved_rule.category == rule.category
        assert retrieved_rule.authority == rule.authority

    @pytest.mark.asyncio
    async def test_list_rules_with_filters(self, memory_manager):
        """Test listing rules with various filters."""
        # Create test rules
        rules = [
            MemoryRule(
                rule="Python preference",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Behavior rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        # Mock list operation
        memory_manager.schema.list_all_rules = AsyncMock(return_value=rules)

        # Test listing all rules
        all_rules = await memory_manager.list_rules()
        assert len(all_rules) == 2

        # Test filtering by category
        await memory_manager.list_rules(category_filter=MemoryCategory.PREFERENCE)
        memory_manager.schema.list_all_rules.assert_called_with(
            category_filter=MemoryCategory.PREFERENCE,
            authority_filter=None,
            source_filter=None
        )

        # Test filtering by authority
        await memory_manager.list_rules(authority_filter=AuthorityLevel.ABSOLUTE)
        memory_manager.schema.list_all_rules.assert_called_with(
            category_filter=None,
            authority_filter=AuthorityLevel.ABSOLUTE,
            source_filter=None
        )

    @pytest.mark.asyncio
    async def test_token_usage_calculation(self, memory_manager):
        """Test token usage calculation."""
        # Create test rules
        rules = [
            MemoryRule(
                rule="Short rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="This is a much longer rule that should use more tokens",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        # Mock rule listing
        memory_manager.schema.list_all_rules = AsyncMock(return_value=rules)

        # Get token usage
        token_usage = await memory_manager.get_token_usage()

        assert isinstance(token_usage, TokenUsage)
        assert token_usage.total_tokens > 0
        assert token_usage.rules_count == 2
        assert token_usage.preference_tokens > 0
        assert token_usage.behavior_tokens > 0
        assert token_usage.absolute_tokens > 0
        assert token_usage.default_tokens > 0

    @pytest.mark.asyncio
    async def test_context_aware_rule_optimization(self, memory_manager):
        """Test context-aware rule optimization."""
        # Create test rules with different scopes
        rules = [
            MemoryRule(
                rule="Global rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Python specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["python"]
            ),
            MemoryRule(
                rule="JavaScript specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript"]
            ),
        ]

        # Mock rule listing
        memory_manager.schema.list_all_rules = AsyncMock(return_value=rules)

        # Create context for Python development
        context = MemoryContext(
            session_id="test-session",
            project_name="python-project",
            active_scopes=["python"]
        )

        # Optimize rules for context
        selected_rules, token_usage = await memory_manager.optimize_rules_for_context(
            context, max_tokens=1000
        )

        # Should include global rule and Python-specific rule, exclude JavaScript rule
        assert len(selected_rules) == 2

        rule_texts = [rule.rule for rule in selected_rules]
        assert "Global rule" in rule_texts
        assert "Python specific rule" in rule_texts
        assert "JavaScript specific rule" not in rule_texts

    @pytest.mark.asyncio
    async def test_conversational_text_processing(self, memory_manager):
        """Test processing conversational text for memory updates."""
        # Mock rule addition
        memory_manager.add_rule = AsyncMock(return_value=("rule-id", []))

        # Test conversational text
        text = "Note: call me Chris. Also, I prefer using uv for Python."

        # Process text
        new_rules = await memory_manager.process_conversational_text(text)

        # Should detect at least one rule
        assert len(new_rules) >= 1

        # Check that add_rule was called
        assert memory_manager.add_rule.call_count >= 1

    @pytest.mark.asyncio
    async def test_claude_session_initialization(self, memory_manager):
        """Test Claude Code session initialization."""
        # Create test session
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/path/to/project",
            user_name="chris",
            project_name="test-project"
        )

        # Mock rule listing
        memory_manager.schema.list_all_rules = AsyncMock(return_value=[
            MemoryRule(
                rule="Test rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            )
        ])

        # Mock Claude integration
        with patch.object(memory_manager.claude_integration, 'initialize_session') as mock_init:
            mock_init.return_value = Mock(
                success=True,
                rules_injected=1,
                total_tokens_used=100,
                remaining_context_tokens=199900,
                skipped_rules=[],
                errors=[]
            )

            result = await memory_manager.initialize_claude_session(session)

            assert result.success is True
            assert result.rules_injected == 1
            mock_init.assert_called_once()


class TestMemorySystemComponents:
    """Test individual memory system components."""

    def test_token_counter_initialization(self):
        """Test token counter initialization."""
        counter = TokenCounter(context_window_size=200000)

        assert counter.context_window_size == 200000
        assert counter.method.value == "simple"  # Default method

    def test_token_counter_rule_counting(self):
        """Test token counting for individual rules."""
        counter = TokenCounter()

        rule = MemoryRule(
            rule="This is a test rule for token counting",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )

        token_count = counter.count_rule_tokens(rule)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_token_counter_optimization(self):
        """Test rule optimization for token budget."""
        counter = TokenCounter()

        rules = [
            MemoryRule(
                rule="Short rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="This is a much longer rule that should use significantly more tokens",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        # Optimize with small token budget
        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=20, preserve_absolute=True
        )

        # Should preserve absolute rule even if it's longer
        assert len(selected_rules) >= 1
        assert any(rule.authority == AuthorityLevel.ABSOLUTE for rule in selected_rules)

    def test_conflict_detector_initialization(self):
        """Test conflict detector initialization."""
        detector = ConflictDetector(enable_ai_analysis=False)

        assert detector.enable_ai_analysis is False
        assert detector.anthropic_client is None

    @pytest.mark.asyncio
    async def test_conflict_detector_rule_based(self):
        """Test rule-based conflict detection."""
        detector = ConflictDetector(enable_ai_analysis=False)

        # Create conflicting rules
        rule1 = MemoryRule(
            rule="Always use X for development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        rule2 = MemoryRule(
            rule="Never use X for development",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        # Detect conflicts
        conflicts = await detector.detect_conflicts(rule1, [rule2])

        # Should detect authority conflict (both absolute in same domain)
        assert len(conflicts) > 0
        conflict = conflicts[0]
        assert conflict.conflict_type in ["authority", "direct"]
        assert conflict.severity in ["high", "critical"]
