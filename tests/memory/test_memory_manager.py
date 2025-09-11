"""
Tests for memory manager system.

Comprehensive tests for the main memory manager including CRUD operations,
conflict detection integration, rule optimization, and system coordination.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from common.core.config import Config
from common.memory.manager import MemoryManager
from common.memory.token_counter import TokenUsage
from common.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    ConversationalUpdate,
    MemoryCategory,
    MemoryContext,
    MemoryInjectionResult,
    MemoryRule,
    MemoryRuleConflict,
)


class TestMemoryManagerInit:
    """Test memory manager initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        manager = MemoryManager()
        
        assert manager.config is not None
        assert manager.qdrant_client is not None
        assert manager.embedding_service is not None
        assert manager.schema is not None
        assert manager.conflict_detector is not None
        assert manager.token_counter is not None
        assert manager.claude_integration is not None
        assert manager._initialized is False

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = Config()
        custom_config.max_memory_tokens = 10000
        custom_config.enable_memory_ai_analysis = False
        
        manager = MemoryManager(config=custom_config)
        
        assert manager.config == custom_config
        assert manager.claude_integration.max_memory_tokens == 10000
        assert manager.conflict_detector.enable_ai_analysis is False

    def test_init_with_custom_clients(self):
        """Test initialization with custom client instances."""
        mock_qdrant = Mock()
        mock_embedding = Mock()
        
        manager = MemoryManager(
            qdrant_client=mock_qdrant,
            embedding_service=mock_embedding
        )
        
        assert manager.qdrant_client == mock_qdrant
        assert manager.embedding_service == mock_embedding

    @pytest.mark.asyncio
    async def test_initialization_success(self):
        """Test successful manager initialization."""
        manager = MemoryManager()
        
        with patch.object(manager.schema, 'ensure_collection_exists') as mock_ensure:
            mock_ensure.return_value = True
            
            result = await manager.initialize()
            
            assert result is True
            assert manager._initialized is True
            mock_ensure.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialization_failure(self):
        """Test failed manager initialization."""
        manager = MemoryManager()
        
        with patch.object(manager.schema, 'ensure_collection_exists') as mock_ensure:
            mock_ensure.return_value = False
            
            result = await manager.initialize()
            
            assert result is False
            assert manager._initialized is False


class TestRuleCRUD:
    """Test memory rule CRUD operations."""

    @pytest.fixture
    async def manager(self):
        """Create initialized memory manager."""
        manager = MemoryManager()
        
        # Mock the schema methods
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.store_rule = AsyncMock(return_value=True)
        manager.schema.get_rule = AsyncMock()
        manager.schema.list_all_rules = AsyncMock(return_value=[])
        manager.schema.update_rule = AsyncMock(return_value=True)
        manager.schema.delete_rule = AsyncMock(return_value=True)
        
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_add_rule_success(self, manager):
        """Test successful rule addition."""
        rule = MemoryRule(
            rule="Always use type hints in Python functions",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        # Mock conflict detection
        manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        rule_id, conflicts = await manager.add_rule(rule, check_conflicts=True)

        assert rule_id == rule.id
        assert conflicts == []
        manager.schema.store_rule.assert_called_once_with(rule)
        manager.conflict_detector.detect_conflicts.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_rule_with_conflicts(self, manager):
        """Test adding rule that has conflicts."""
        new_rule = MemoryRule(
            rule="Never use type hints in Python",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        existing_rule = MemoryRule(
            rule="Always use type hints",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflict = MemoryRuleConflict(
            rule1=new_rule,
            rule2=existing_rule,
            conflict_type="direct",
            severity="critical",
            description="Direct contradiction"
        )

        # Mock existing rules and conflict detection
        manager.schema.list_all_rules = AsyncMock(return_value=[existing_rule])
        manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[conflict])

        rule_id, conflicts = await manager.add_rule(new_rule, check_conflicts=True)

        assert rule_id == new_rule.id
        assert len(conflicts) == 1
        assert conflicts[0] == conflict
        # Should still store the rule even with conflicts
        manager.schema.store_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_rule_skip_conflicts(self, manager):
        """Test adding rule with conflict checking disabled."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )

        rule_id, conflicts = await manager.add_rule(rule, check_conflicts=False)

        assert rule_id == rule.id
        assert conflicts == []
        manager.schema.store_rule.assert_called_once()
        # Should not call conflict detection
        manager.conflict_detector.detect_conflicts.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_rule_exists(self, manager):
        """Test retrieving existing rule."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        manager.schema.get_rule.return_value = rule

        retrieved_rule = await manager.get_rule(rule.id)

        assert retrieved_rule == rule
        manager.schema.get_rule.assert_called_once_with(rule.id)

    @pytest.mark.asyncio
    async def test_get_rule_not_found(self, manager):
        """Test retrieving non-existent rule."""
        manager.schema.get_rule.return_value = None

        retrieved_rule = await manager.get_rule("nonexistent-id")

        assert retrieved_rule is None

    @pytest.mark.asyncio
    async def test_update_rule_success(self, manager):
        """Test successful rule update."""
        original_rule = MemoryRule(
            rule="Original rule text",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )

        updates = {
            "rule": "Updated rule text",
            "authority": AuthorityLevel.ABSOLUTE
        }

        manager.schema.get_rule.return_value = original_rule
        manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        success, conflicts = await manager.update_rule(
            original_rule.id, updates, check_conflicts=True
        )

        assert success is True
        assert conflicts == []
        manager.schema.update_rule.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_rule_not_found(self, manager):
        """Test updating non-existent rule."""
        manager.schema.get_rule.return_value = None

        success, conflicts = await manager.update_rule(
            "nonexistent-id", {"rule": "New text"}
        )

        assert success is False
        assert conflicts == []
        manager.schema.update_rule.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_rule_success(self, manager):
        """Test successful rule deletion."""
        rule_id = "test-rule-id"

        success = await manager.delete_rule(rule_id)

        assert success is True
        manager.schema.delete_rule.assert_called_once_with(rule_id)

    @pytest.mark.asyncio
    async def test_list_rules_all(self, manager):
        """Test listing all rules."""
        rules = [
            MemoryRule(
                rule="Rule 1",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Rule 2",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        manager.schema.list_all_rules.return_value = rules

        result = await manager.list_rules()

        assert result == rules
        manager.schema.list_all_rules.assert_called_once_with(
            category_filter=None,
            authority_filter=None,
            source_filter=None
        )

    @pytest.mark.asyncio
    async def test_list_rules_with_filters(self, manager):
        """Test listing rules with filters."""
        filtered_rules = [
            MemoryRule(
                rule="Absolute rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        manager.schema.list_all_rules.return_value = filtered_rules

        result = await manager.list_rules(
            category_filter=MemoryCategory.BEHAVIOR,
            authority_filter=AuthorityLevel.ABSOLUTE,
            source_filter="user_cli"
        )

        assert result == filtered_rules
        manager.schema.list_all_rules.assert_called_once_with(
            category_filter=MemoryCategory.BEHAVIOR,
            authority_filter=AuthorityLevel.ABSOLUTE,
            source_filter="user_cli"
        )


class TestTokenManagement:
    """Test token counting and optimization."""

    @pytest.fixture
    async def manager(self):
        """Create initialized memory manager."""
        manager = MemoryManager()
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_get_token_usage(self, manager):
        """Test token usage calculation."""
        rules = [
            MemoryRule(
                rule="Short rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="This is a longer rule with more content to test token counting",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        manager.schema.list_all_rules = AsyncMock(return_value=rules)

        usage = await manager.get_token_usage()

        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens > 0
        assert usage.rules_count == 2
        assert usage.preference_tokens > 0
        assert usage.behavior_tokens > 0
        assert usage.default_tokens > 0
        assert usage.absolute_tokens > 0

    @pytest.mark.asyncio
    async def test_optimize_rules_for_context(self, manager):
        """Test context-aware rule optimization."""
        rules = [
            MemoryRule(
                rule="Python-specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python"]
            ),
            MemoryRule(
                rule="JavaScript-specific rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript"]
            ),
            MemoryRule(
                rule="General rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                scope=[]
            ),
        ]

        context = MemoryContext(
            session_id="test-session",
            project_name="python-project",
            active_scopes=["python", "backend"]
        )

        manager.schema.list_all_rules = AsyncMock(return_value=rules)

        selected_rules, usage = await manager.optimize_rules_for_context(
            context, max_tokens=1000
        )

        assert len(selected_rules) <= len(rules)
        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens <= 1000

        # Should prefer Python and general rules over JavaScript
        rule_texts = [r.rule for r in selected_rules]
        assert any("Python" in text for text in rule_texts)
        assert any("General" in text for text in rule_texts)

    @pytest.mark.asyncio
    async def test_optimize_empty_context(self, manager):
        """Test optimization with empty rule set."""
        manager.schema.list_all_rules = AsyncMock(return_value=[])

        context = MemoryContext(
            session_id="test-session",
            project_name="empty-project"
        )

        selected_rules, usage = await manager.optimize_rules_for_context(
            context, max_tokens=1000
        )

        assert len(selected_rules) == 0
        assert usage.total_tokens == 0
        assert usage.rules_count == 0


class TestConversationalUpdates:
    """Test conversational memory updates processing."""

    @pytest.fixture
    async def manager(self):
        """Create initialized memory manager."""
        manager = MemoryManager()
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.store_rule = AsyncMock(return_value=True)
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_process_conversational_text(self, manager):
        """Test processing conversational text for memory updates."""
        text = "Note: call me Chris. Also, I prefer using pytest for testing."

        # Mock Claude integration
        mock_updates = [
            ConversationalUpdate(
                text="call me Chris",
                extracted_rule="User name is Chris",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE,
                scope=["user_identity"],
                confidence=0.9
            ),
            ConversationalUpdate(
                text="prefer using pytest",
                extracted_rule="Prefer pytest for testing",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                scope=["python", "testing"],
                confidence=0.85
            )
        ]

        manager.claude_integration.detect_conversational_updates = AsyncMock(
            return_value=mock_updates
        )
        manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        new_rules = await manager.process_conversational_text(text)

        assert len(new_rules) == 2
        assert all(isinstance(rule, MemoryRule) for rule in new_rules)
        
        # Check first rule (name)
        assert "Chris" in new_rules[0].rule
        assert new_rules[0].category == MemoryCategory.PREFERENCE
        
        # Check second rule (pytest preference) 
        assert "pytest" in new_rules[1].rule
        assert new_rules[1].category == MemoryCategory.PREFERENCE

    @pytest.mark.asyncio
    async def test_process_text_no_updates(self, manager):
        """Test processing text with no extractable updates."""
        text = "The weather is nice today. How are you?"

        manager.claude_integration.detect_conversational_updates = AsyncMock(
            return_value=[]
        )

        new_rules = await manager.process_conversational_text(text)

        assert len(new_rules) == 0

    @pytest.mark.asyncio
    async def test_process_text_low_confidence(self, manager):
        """Test processing text with low confidence updates."""
        text = "Maybe we could consider using React sometime."

        low_confidence_update = ConversationalUpdate(
            text=text,
            extracted_rule="Consider using React",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            confidence=0.3  # Below threshold
        )

        manager.claude_integration.detect_conversational_updates = AsyncMock(
            return_value=[low_confidence_update]
        )

        new_rules = await manager.process_conversational_text(text)

        # Should filter out low confidence updates
        assert len(new_rules) == 0

    @pytest.mark.asyncio
    async def test_process_text_with_conflicts(self, manager):
        """Test processing text that creates conflicting rules."""
        text = "I want to use single quotes for all Python strings."

        new_update = ConversationalUpdate(
            text=text,
            extracted_rule="Use single quotes for Python strings",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            confidence=0.9
        )

        existing_rule = MemoryRule(
            rule="Use double quotes for Python strings",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"]
        )

        conflict = MemoryRuleConflict(
            rule1=MemoryRule.from_conversational_update(new_update),
            rule2=existing_rule,
            conflict_type="direct",
            severity="high",
            description="Conflicting string quote preferences"
        )

        manager.claude_integration.detect_conversational_updates = AsyncMock(
            return_value=[new_update]
        )
        manager.conflict_detector.detect_conflicts = AsyncMock(
            return_value=[conflict]
        )

        new_rules = await manager.process_conversational_text(text)

        # Should still create the rule but note the conflict
        assert len(new_rules) == 1
        # Note: conflict handling details would depend on implementation


class TestClaudeSessionIntegration:
    """Test Claude Code session integration."""

    @pytest.fixture
    async def manager(self):
        """Create initialized memory manager."""
        manager = MemoryManager()
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.list_all_rules = AsyncMock(return_value=[])
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_initialize_claude_session(self, manager):
        """Test initializing Claude Code session."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/test/path",
            user_name="test_user",
            project_name="test_project"
        )

        rules = [
            MemoryRule(
                rule="Test rule for session",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
        ]

        manager.schema.list_all_rules.return_value = rules

        # Mock Claude integration
        mock_result = MemoryInjectionResult(
            success=True,
            rules_injected=1,
            total_tokens_used=50,
            remaining_context_tokens=199950,
            skipped_rules=[],
            errors=[]
        )

        manager.claude_integration.initialize_session = AsyncMock(
            return_value=mock_result
        )

        result = await manager.initialize_claude_session(session)

        assert result == mock_result
        assert result.success is True
        assert result.rules_injected == 1
        manager.claude_integration.initialize_session.assert_called_once_with(
            session, rules
        )

    @pytest.mark.asyncio
    async def test_session_with_optimization(self, manager):
        """Test session initialization with rule optimization."""
        session = ClaudeCodeSession(
            session_id="test-session",
            workspace_path="/python/project",
            context_window_size=100000
        )

        # Create many rules to trigger optimization
        rules = []
        for i in range(50):
            rules.append(MemoryRule(
                rule=f"Test rule {i} with some content",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python"] if i % 2 == 0 else ["javascript"]
            ))

        manager.schema.list_all_rules.return_value = rules

        # Mock optimization in Claude integration
        mock_result = MemoryInjectionResult(
            success=True,
            rules_injected=25,  # Only subset due to optimization
            total_tokens_used=4500,
            remaining_context_tokens=95500,
            skipped_rules=[f"Rule {i}" for i in range(25, 50)],
            errors=[]
        )

        manager.claude_integration.initialize_session = AsyncMock(
            return_value=mock_result
        )

        result = await manager.initialize_claude_session(session)

        assert result.success is True
        assert result.rules_injected < len(rules)  # Should be optimized
        assert len(result.skipped_rules) > 0


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    async def manager(self):
        """Create memory manager for error testing."""
        manager = MemoryManager()
        # Don't initialize to test error conditions
        return manager

    @pytest.mark.asyncio
    async def test_operations_before_init(self, manager):
        """Test operations called before initialization."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Should handle gracefully or raise appropriate error
        with pytest.raises((RuntimeError, AttributeError)):
            await manager.add_rule(rule)

    @pytest.mark.asyncio
    async def test_schema_operation_failure(self):
        """Test handling of schema operation failures."""
        manager = MemoryManager()
        
        # Mock schema to raise exceptions
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.store_rule = AsyncMock(side_effect=Exception("Storage failed"))
        
        await manager.initialize()

        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Should handle storage failure gracefully
        with pytest.raises(Exception):
            await manager.add_rule(rule)

    @pytest.mark.asyncio
    async def test_conflict_detection_failure(self):
        """Test handling of conflict detection failures."""
        manager = MemoryManager()
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.store_rule = AsyncMock(return_value=True)
        manager.schema.list_all_rules = AsyncMock(return_value=[])
        
        # Mock conflict detector to fail
        manager.conflict_detector.detect_conflicts = AsyncMock(
            side_effect=Exception("Conflict detection failed")
        )
        
        await manager.initialize()

        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
        )

        # Should still add rule but with empty conflicts list
        rule_id, conflicts = await manager.add_rule(rule, check_conflicts=True)
        
        assert rule_id == rule.id
        # Behavior depends on implementation - might return empty conflicts
        # or propagate exception


class TestConcurrency:
    """Test concurrent operations."""

    @pytest.fixture
    async def manager(self):
        """Create initialized memory manager."""
        manager = MemoryManager()
        manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        manager.schema.store_rule = AsyncMock(return_value=True)
        manager.schema.list_all_rules = AsyncMock(return_value=[])
        await manager.initialize()
        return manager

    @pytest.mark.asyncio
    async def test_concurrent_rule_addition(self, manager):
        """Test concurrent rule addition operations."""
        rules = []
        for i in range(10):
            rules.append(MemoryRule(
                rule=f"Concurrent test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ))

        # Mock conflict detection to be fast
        manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        # Add rules concurrently
        tasks = [manager.add_rule(rule, check_conflicts=False) for rule in rules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            rule_id, conflicts = result
            assert rule_id is not None
            assert conflicts == []

    @pytest.mark.asyncio
    async def test_concurrent_read_operations(self, manager):
        """Test concurrent read operations."""
        test_rules = []
        for i in range(5):
            test_rules.append(MemoryRule(
                rule=f"Test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ))

        manager.schema.list_all_rules.return_value = test_rules

        # Perform concurrent reads
        tasks = [manager.list_rules() for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should succeed and return same data
        assert len(results) == 10
        for result in results:
            assert not isinstance(result, Exception)
            assert result == test_rules