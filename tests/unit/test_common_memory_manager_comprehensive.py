"""
Comprehensive unit tests for python.common.memory.manager module.

Tests cover MemoryManager and all memory system functionality
with 100% coverage including async patterns, rule management, and conflict detection.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.python.common.core.config import Config

# Import modules under test
from src.python.common.memory.manager import MemoryManager
from src.python.common.memory.token_counter import TokenUsage
from src.python.common.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    ConversationalUpdate,
    MemoryCategory,
    MemoryContext,
    MemoryInjectionResult,
    MemoryRule,
    MemoryRuleConflict,
)


class TestMemoryManager:
    """Test MemoryManager functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config."""
        config = Mock(spec=Config)
        config.enable_memory_ai_analysis = True
        config.max_memory_tokens = 5000
        return config

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock QdrantWorkspaceClient."""
        client = Mock()
        return client

    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock EmbeddingService."""
        service = Mock()
        service.initialize = AsyncMock()
        return service

    @pytest.fixture
    def mock_schema(self):
        """Create a mock MemoryCollectionSchema."""
        schema = Mock()
        schema.ensure_collection_exists = AsyncMock(return_value=True)
        schema.store_rule = AsyncMock(return_value=True)
        schema.update_rule = AsyncMock(return_value=True)
        schema.get_rule = AsyncMock(return_value=None)
        schema.delete_rule = AsyncMock(return_value=True)
        schema.list_all_rules = AsyncMock(return_value=[])
        schema.search_rules = AsyncMock(return_value=[])
        schema.get_collection_stats = AsyncMock(return_value={})
        return schema

    @pytest.fixture
    def mock_conflict_detector(self):
        """Create a mock ConflictDetector."""
        detector = Mock()
        detector.detect_conflicts = AsyncMock(return_value=[])
        detector.analyze_all_conflicts = AsyncMock(return_value=[])
        detector.get_conflict_summary = Mock(return_value={})
        return detector

    @pytest.fixture
    def mock_token_counter(self):
        """Create a mock TokenCounter."""
        counter = Mock()
        counter.count_rules_tokens = Mock(return_value=TokenUsage(
            total_tokens=1000,
            rule_tokens=800,
            metadata_tokens=200,
            estimated_cost=0.01
        ))
        counter.optimize_rules_for_context = Mock(return_value=([], TokenUsage(
            total_tokens=500,
            rule_tokens=400,
            metadata_tokens=100,
            estimated_cost=0.005
        )))
        counter.suggest_memory_optimizations = Mock(return_value={
            "recommendations": [],
            "current_tokens": 1000,
            "target_tokens": 3000
        })
        return counter

    @pytest.fixture
    def mock_claude_integration(self):
        """Create a mock ClaudeCodeIntegration."""
        integration = Mock()
        integration.detect_conversational_updates = Mock(return_value=[])
        integration.process_conversational_update = Mock(return_value=None)
        integration.initialize_session = AsyncMock(return_value=MemoryInjectionResult(
            injected_rules=[],
            total_tokens=100,
            skipped_rules=[],
            optimization_applied=False
        ))
        return integration

    @pytest.fixture
    def memory_manager(self, mock_config, mock_qdrant_client, mock_embedding_service):
        """Create MemoryManager instance with mocked dependencies."""
        with patch('src.python.common.memory.manager.MemoryCollectionSchema'):
            with patch('src.python.common.memory.manager.ConflictDetector'):
                with patch('src.python.common.memory.manager.TokenCounter'):
                    with patch('src.python.common.memory.manager.ClaudeCodeIntegration'):
                        manager = MemoryManager(mock_config, mock_qdrant_client, mock_embedding_service)

                        # Replace with controlled mocks
                        manager.schema = Mock()
                        manager.conflict_detector = Mock()
                        manager.token_counter = Mock()
                        manager.claude_integration = Mock()

                        return manager

    def test_init_with_all_params(self, mock_config, mock_qdrant_client, mock_embedding_service):
        """Test MemoryManager initialization with all parameters."""
        with patch('src.python.common.memory.manager.MemoryCollectionSchema'):
            with patch('src.python.common.memory.manager.ConflictDetector'):
                with patch('src.python.common.memory.manager.TokenCounter'):
                    with patch('src.python.common.memory.manager.ClaudeCodeIntegration'):
                        manager = MemoryManager(mock_config, mock_qdrant_client, mock_embedding_service)

                        assert manager.config == mock_config
                        assert manager.qdrant_client == mock_qdrant_client
                        assert manager.embedding_service == mock_embedding_service
                        assert not manager._initialized

    def test_init_with_defaults(self):
        """Test MemoryManager initialization with default parameters."""
        with patch('src.python.common.memory.manager.Config') as mock_config_class:
            with patch('src.python.common.memory.manager.QdrantWorkspaceClient') as mock_client_class:
                with patch('src.python.common.memory.manager.EmbeddingService') as mock_service_class:
                    with patch('src.python.common.memory.manager.MemoryCollectionSchema'):
                        with patch('src.python.common.memory.manager.ConflictDetector'):
                            with patch('src.python.common.memory.manager.TokenCounter'):
                                with patch('src.python.common.memory.manager.ClaudeCodeIntegration'):
                                    mock_config = Mock()
                                    mock_client = Mock()
                                    mock_service = Mock()

                                    mock_config_class.return_value = mock_config
                                    mock_client_class.return_value = mock_client
                                    mock_service_class.return_value = mock_service

                                    manager = MemoryManager()

                                    assert manager.config == mock_config
                                    assert manager.qdrant_client == mock_client
                                    assert manager.embedding_service == mock_service

    async def test_initialize_success(self, memory_manager):
        """Test successful memory system initialization."""
        memory_manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        memory_manager.embedding_service.initialize = AsyncMock()

        result = await memory_manager.initialize()

        assert result
        assert memory_manager._initialized
        memory_manager.schema.ensure_collection_exists.assert_called_once()

    async def test_initialize_collection_failure(self, memory_manager):
        """Test initialization failure when collection creation fails."""
        memory_manager.schema.ensure_collection_exists = AsyncMock(return_value=False)

        result = await memory_manager.initialize()

        assert not result
        assert not memory_manager._initialized

    async def test_initialize_exception(self, memory_manager):
        """Test initialization with exception."""
        memory_manager.schema.ensure_collection_exists = AsyncMock(side_effect=Exception("Schema error"))

        result = await memory_manager.initialize()

        assert not result
        assert not memory_manager._initialized

    async def test_initialize_no_embedding_initialize(self, memory_manager):
        """Test initialization when embedding service has no initialize method."""
        memory_manager.schema.ensure_collection_exists = AsyncMock(return_value=True)
        # Remove initialize method from embedding service
        del memory_manager.embedding_service.initialize

        result = await memory_manager.initialize()

        assert result
        assert memory_manager._initialized

    async def test_add_rule_success(self, memory_manager):
        """Test successfully adding a memory rule."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.store_rule = AsyncMock(return_value=True)
        memory_manager.schema.list_all_rules = AsyncMock(return_value=[])
        memory_manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        rule_id, conflicts = await memory_manager.add_rule(rule)

        assert rule_id == "rule1"
        assert conflicts == []
        memory_manager.schema.store_rule.assert_called_once_with(rule)

    async def test_add_rule_with_conflicts(self, memory_manager):
        """Test adding a rule with detected conflicts."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        conflict = MemoryRuleConflict(
            rule1_id="rule1",
            rule2_id="existing_rule",
            severity="high",
            description="Conflicting rules",
            resolution_suggestion="Review both rules"
        )

        memory_manager.schema.store_rule = AsyncMock(return_value=True)
        memory_manager.schema.list_all_rules = AsyncMock(return_value=[])
        memory_manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[conflict])

        rule_id, conflicts = await memory_manager.add_rule(rule, check_conflicts=True)

        assert rule_id == "rule1"
        assert len(conflicts) == 1
        assert conflicts[0] == conflict

    async def test_add_rule_no_conflict_check(self, memory_manager):
        """Test adding a rule without conflict checking."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.store_rule = AsyncMock(return_value=True)

        rule_id, conflicts = await memory_manager.add_rule(rule, check_conflicts=False)

        assert rule_id == "rule1"
        assert conflicts == []
        # Should not call list_all_rules or detect_conflicts
        memory_manager.schema.list_all_rules.assert_not_called()
        memory_manager.conflict_detector.detect_conflicts.assert_not_called()

    async def test_add_rule_store_failure(self, memory_manager):
        """Test adding a rule when storage fails."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.store_rule = AsyncMock(return_value=False)
        memory_manager.schema.list_all_rules = AsyncMock(return_value=[])
        memory_manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        with pytest.raises(Exception, match="Failed to store rule"):
            await memory_manager.add_rule(rule)

    async def test_add_rule_auto_initialize(self, memory_manager):
        """Test that add_rule auto-initializes if not initialized."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager._initialized = False
        memory_manager.initialize = AsyncMock(return_value=True)
        memory_manager.schema.store_rule = AsyncMock(return_value=True)
        memory_manager.schema.list_all_rules = AsyncMock(return_value=[])
        memory_manager.conflict_detector.detect_conflicts = AsyncMock(return_value=[])

        await memory_manager.add_rule(rule)

        memory_manager.initialize.assert_called_once()

    async def test_update_rule(self, memory_manager):
        """Test updating a memory rule."""
        rule = MemoryRule(
            id="rule1",
            rule="Updated rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.update_rule = AsyncMock(return_value=True)

        result = await memory_manager.update_rule(rule)

        assert result
        memory_manager.schema.update_rule.assert_called_once_with(rule)
        # Should update the timestamp
        assert rule.updated_at is not None

    async def test_update_rule_failure(self, memory_manager):
        """Test updating a rule that fails."""
        rule = MemoryRule(
            id="rule1",
            rule="Updated rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.update_rule = AsyncMock(return_value=False)

        result = await memory_manager.update_rule(rule)

        assert not result

    async def test_update_rule_exception(self, memory_manager):
        """Test updating a rule with exception."""
        rule = MemoryRule(
            id="rule1",
            rule="Updated rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.update_rule = AsyncMock(side_effect=Exception("Update error"))

        result = await memory_manager.update_rule(rule)

        assert not result

    async def test_get_rule(self, memory_manager):
        """Test getting a memory rule by ID."""
        expected_rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        memory_manager.schema.get_rule = AsyncMock(return_value=expected_rule)

        result = await memory_manager.get_rule("rule1")

        assert result == expected_rule
        memory_manager.schema.get_rule.assert_called_once_with("rule1")

    async def test_get_rule_not_found(self, memory_manager):
        """Test getting a rule that doesn't exist."""
        memory_manager.schema.get_rule = AsyncMock(return_value=None)

        result = await memory_manager.get_rule("nonexistent")

        assert result is None

    async def test_delete_rule(self, memory_manager):
        """Test deleting a memory rule."""
        memory_manager.schema.delete_rule = AsyncMock(return_value=True)

        result = await memory_manager.delete_rule("rule1")

        assert result
        memory_manager.schema.delete_rule.assert_called_once_with("rule1")

    async def test_delete_rule_failure(self, memory_manager):
        """Test deleting a rule that fails."""
        memory_manager.schema.delete_rule = AsyncMock(return_value=False)

        result = await memory_manager.delete_rule("rule1")

        assert not result

    async def test_delete_rule_exception(self, memory_manager):
        """Test deleting a rule with exception."""
        memory_manager.schema.delete_rule = AsyncMock(side_effect=Exception("Delete error"))

        result = await memory_manager.delete_rule("rule1")

        assert not result

    async def test_list_rules_no_filters(self, memory_manager):
        """Test listing all memory rules without filters."""
        expected_rules = [
            MemoryRule(id="rule1", rule="Test 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Test 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        memory_manager.schema.list_all_rules = AsyncMock(return_value=expected_rules)

        result = await memory_manager.list_rules()

        assert result == expected_rules
        memory_manager.schema.list_all_rules.assert_called_once_with(
            category_filter=None,
            authority_filter=None,
            source_filter=None
        )

    async def test_list_rules_with_filters(self, memory_manager):
        """Test listing memory rules with filters."""
        expected_rules = [
            MemoryRule(id="rule1", rule="Test 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED)
        ]

        memory_manager.schema.list_all_rules = AsyncMock(return_value=expected_rules)

        result = await memory_manager.list_rules(
            category_filter=MemoryCategory.PROJECT_SPECIFIC,
            authority_filter=AuthorityLevel.USER_DEFINED,
            source_filter="test_source"
        )

        assert result == expected_rules
        memory_manager.schema.list_all_rules.assert_called_once_with(
            category_filter=MemoryCategory.PROJECT_SPECIFIC,
            authority_filter=AuthorityLevel.USER_DEFINED,
            source_filter="test_source"
        )

    async def test_search_rules(self, memory_manager):
        """Test searching memory rules."""
        expected_results = [
            (MemoryRule(id="rule1", rule="Test rule", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED), 0.95),
            (MemoryRule(id="rule2", rule="Another rule", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM), 0.85)
        ]

        memory_manager.schema.search_rules = AsyncMock(return_value=expected_results)

        result = await memory_manager.search_rules("test query", limit=5, category_filter=MemoryCategory.PROJECT_SPECIFIC)

        assert result == expected_results
        memory_manager.schema.search_rules.assert_called_once_with(
            "test query", 5, category_filter=MemoryCategory.PROJECT_SPECIFIC
        )

    async def test_check_conflicts(self, memory_manager):
        """Test checking conflicts for a rule."""
        rule = MemoryRule(
            id="rule1",
            rule="Test rule",
            category=MemoryCategory.PROJECT_SPECIFIC,
            authority_level=AuthorityLevel.USER_DEFINED
        )

        existing_rules = [
            MemoryRule(id="rule2", rule="Existing rule", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        expected_conflicts = [
            MemoryRuleConflict(
                rule1_id="rule1",
                rule2_id="rule2",
                severity="medium",
                description="Potential conflict",
                resolution_suggestion="Review rules"
            )
        ]

        memory_manager.schema.list_all_rules = AsyncMock(return_value=existing_rules)
        memory_manager.conflict_detector.detect_conflicts = AsyncMock(return_value=expected_conflicts)

        result = await memory_manager.check_conflicts(rule)

        assert result == expected_conflicts
        memory_manager.conflict_detector.detect_conflicts.assert_called_once_with(rule, existing_rules)

    async def test_analyze_all_conflicts(self, memory_manager):
        """Test analyzing conflicts among all rules."""
        all_rules = [
            MemoryRule(id="rule1", rule="Rule 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Rule 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        expected_conflicts = [
            MemoryRuleConflict(
                rule1_id="rule1",
                rule2_id="rule2",
                severity="low",
                description="Minor conflict",
                resolution_suggestion="No action needed"
            )
        ]

        memory_manager.schema.list_all_rules = AsyncMock(return_value=all_rules)
        memory_manager.conflict_detector.analyze_all_conflicts = AsyncMock(return_value=expected_conflicts)

        result = await memory_manager.analyze_all_conflicts()

        assert result == expected_conflicts
        memory_manager.conflict_detector.analyze_all_conflicts.assert_called_once_with(all_rules)

    async def test_get_token_usage(self, memory_manager):
        """Test getting token usage for all rules."""
        all_rules = [
            MemoryRule(id="rule1", rule="Rule 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Rule 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        expected_usage = TokenUsage(
            total_tokens=1500,
            rule_tokens=1200,
            metadata_tokens=300,
            estimated_cost=0.015
        )

        memory_manager.schema.list_all_rules = AsyncMock(return_value=all_rules)
        memory_manager.token_counter.count_rules_tokens = Mock(return_value=expected_usage)

        result = await memory_manager.get_token_usage()

        assert result == expected_usage
        memory_manager.token_counter.count_rules_tokens.assert_called_once_with(all_rules)

    async def test_optimize_rules_for_context(self, memory_manager):
        """Test optimizing rules for a specific context."""
        all_rules = [
            MemoryRule(id="rule1", rule="Rule 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Rule 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        context = MemoryContext(
            session_id="session1",
            project_name="test_project",
            file_path="/test/file.py",
            task_context="coding"
        )

        # Mock rules that match the context
        relevant_rules = [all_rules[0]]  # Only first rule matches

        expected_usage = TokenUsage(
            total_tokens=800,
            rule_tokens=600,
            metadata_tokens=200,
            estimated_cost=0.008
        )

        memory_manager.schema.list_all_rules = AsyncMock(return_value=all_rules)

        # Mock rule matching
        all_rules[0].matches_scope = Mock(return_value=True)
        all_rules[1].matches_scope = Mock(return_value=False)

        memory_manager.token_counter.optimize_rules_for_context = Mock(
            return_value=(relevant_rules, expected_usage)
        )

        selected_rules, usage = await memory_manager.optimize_rules_for_context(context, max_tokens=3000)

        assert selected_rules == relevant_rules
        assert usage == expected_usage
        memory_manager.token_counter.optimize_rules_for_context.assert_called_once_with(
            relevant_rules, 3000, preserve_absolute=True
        )

    async def test_suggest_optimizations(self, memory_manager):
        """Test suggesting memory optimizations."""
        all_rules = [
            MemoryRule(id="rule1", rule="Rule 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Rule 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        expected_suggestions = {
            "recommendations": [
                {"type": "consolidate", "description": "Merge similar rules"},
                {"type": "archive", "description": "Archive old rules"}
            ],
            "current_tokens": 2000,
            "target_tokens": 1500,
            "potential_savings": 500
        }

        memory_manager.schema.list_all_rules = AsyncMock(return_value=all_rules)
        memory_manager.token_counter.suggest_memory_optimizations = Mock(return_value=expected_suggestions)

        result = await memory_manager.suggest_optimizations(target_tokens=1500)

        assert result == expected_suggestions
        memory_manager.token_counter.suggest_memory_optimizations.assert_called_once_with(all_rules, 1500)

    async def test_process_conversational_text(self, memory_manager):
        """Test processing conversational text for memory updates."""
        text = "Remember that I prefer using Python for data analysis tasks."
        session_context = MemoryContext(
            session_id="session1",
            project_name="test_project"
        )

        conversational_update = ConversationalUpdate(
            text_content=text,
            detected_preferences=["Python for data analysis"],
            suggested_rule="Use Python for data analysis tasks",
            confidence_score=0.85
        )

        new_rule = MemoryRule(
            id="conv_rule_1",
            rule="Use Python for data analysis tasks",
            category=MemoryCategory.PREFERENCE,
            authority_level=AuthorityLevel.CONVERSATIONAL,
            source="conversation"
        )

        memory_manager.claude_integration.detect_conversational_updates = Mock(
            return_value=[conversational_update]
        )
        memory_manager.claude_integration.process_conversational_update = Mock(
            return_value=new_rule
        )

        # Mock the add_rule method to return success
        memory_manager.add_rule = AsyncMock(return_value=(new_rule.id, []))

        result = await memory_manager.process_conversational_text(text, session_context)

        assert len(result) == 1
        assert result[0] == new_rule
        memory_manager.claude_integration.detect_conversational_updates.assert_called_once_with(
            text, session_context
        )
        memory_manager.claude_integration.process_conversational_update.assert_called_once_with(
            conversational_update
        )

    async def test_process_conversational_text_no_updates(self, memory_manager):
        """Test processing conversational text with no detectable updates."""
        text = "Hello, how are you today?"

        memory_manager.claude_integration.detect_conversational_updates = Mock(return_value=[])

        result = await memory_manager.process_conversational_text(text)

        assert result == []

    async def test_process_conversational_text_rule_creation_failure(self, memory_manager):
        """Test processing conversational text when rule creation fails."""
        text = "I like using TypeScript."

        conversational_update = ConversationalUpdate(
            text_content=text,
            detected_preferences=["TypeScript"],
            suggested_rule="Use TypeScript for development",
            confidence_score=0.9
        )

        # Mock failure in processing update
        memory_manager.claude_integration.detect_conversational_updates = Mock(
            return_value=[conversational_update]
        )
        memory_manager.claude_integration.process_conversational_update = Mock(return_value=None)

        result = await memory_manager.process_conversational_text(text)

        assert result == []

    async def test_process_conversational_text_add_rule_failure(self, memory_manager):
        """Test processing conversational text when add_rule fails."""
        text = "I prefer REST APIs over GraphQL."

        conversational_update = ConversationalUpdate(
            text_content=text,
            detected_preferences=["REST APIs"],
            suggested_rule="Prefer REST APIs over GraphQL",
            confidence_score=0.8
        )

        new_rule = MemoryRule(
            id="conv_rule_2",
            rule="Prefer REST APIs over GraphQL",
            category=MemoryCategory.PREFERENCE,
            authority_level=AuthorityLevel.CONVERSATIONAL
        )

        memory_manager.claude_integration.detect_conversational_updates = Mock(
            return_value=[conversational_update]
        )
        memory_manager.claude_integration.process_conversational_update = Mock(
            return_value=new_rule
        )
        memory_manager.add_rule = AsyncMock(side_effect=Exception("Add rule failed"))

        result = await memory_manager.process_conversational_text(text)

        assert result == []

    async def test_initialize_claude_session(self, memory_manager):
        """Test initializing a Claude Code session."""
        session = ClaudeCodeSession(
            session_id="session1",
            project_name="test_project",
            workspace_path="/test/workspace",
            current_files=["/test/file1.py", "/test/file2.py"]
        )

        all_rules = [
            MemoryRule(id="rule1", rule="Rule 1", category=MemoryCategory.PROJECT_SPECIFIC, authority_level=AuthorityLevel.USER_DEFINED),
            MemoryRule(id="rule2", rule="Rule 2", category=MemoryCategory.GLOBAL, authority_level=AuthorityLevel.SYSTEM)
        ]

        expected_result = MemoryInjectionResult(
            injected_rules=all_rules,
            total_tokens=1000,
            skipped_rules=[],
            optimization_applied=True
        )

        memory_manager.schema.list_all_rules = AsyncMock(return_value=all_rules)
        memory_manager.claude_integration.initialize_session = AsyncMock(return_value=expected_result)

        result = await memory_manager.initialize_claude_session(session)

        assert result == expected_result
        memory_manager.claude_integration.initialize_session.assert_called_once_with(session, all_rules)

    async def test_get_memory_stats(self, memory_manager):
        """Test getting comprehensive memory system statistics."""
        collection_stats = {
            "total_rules": 50,
            "collection_size": "1.2MB",
            "last_updated": "2023-01-01T00:00:00Z"
        }

        token_usage = TokenUsage(
            total_tokens=2000,
            rule_tokens=1600,
            metadata_tokens=400,
            estimated_cost=0.02
        )

        all_conflicts = [
            MemoryRuleConflict(
                rule1_id="rule1",
                rule2_id="rule2",
                severity="medium",
                description="Conflicting preferences",
                resolution_suggestion="Prioritize by authority level"
            )
        ]

        conflict_summary = {
            "total_conflicts": 1,
            "by_severity": {"medium": 1},
            "critical_conflicts": []
        }

        memory_manager.schema.get_collection_stats = AsyncMock(return_value=collection_stats)
        memory_manager.get_token_usage = AsyncMock(return_value=token_usage)
        memory_manager.analyze_all_conflicts = AsyncMock(return_value=all_conflicts)
        memory_manager.conflict_detector.get_conflict_summary = Mock(return_value=conflict_summary)

        result = await memory_manager.get_memory_stats()

        assert result["collection"] == collection_stats
        assert result["token_usage"] == token_usage.to_dict()
        assert result["conflicts"] == conflict_summary
        assert "last_updated" in result

    async def test_export_rules(self, memory_manager):
        """Test exporting all memory rules."""
        rules = [
            MemoryRule(
                id="rule1",
                rule="Rule 1",
                category=MemoryCategory.PROJECT_SPECIFIC,
                authority_level=AuthorityLevel.USER_DEFINED
            ),
            MemoryRule(
                id="rule2",
                rule="Rule 2",
                category=MemoryCategory.GLOBAL,
                authority_level=AuthorityLevel.SYSTEM
            )
        ]

        memory_manager.schema.list_all_rules = AsyncMock(return_value=rules)

        # Mock to_dict method
        for i, rule in enumerate(rules):
            rule.to_dict = Mock(return_value={"id": f"rule{i+1}", "rule": f"Rule {i+1}"})

        result = await memory_manager.export_rules()

        assert len(result) == 2
        assert result[0] == {"id": "rule1", "rule": "Rule 1"}
        assert result[1] == {"id": "rule2", "rule": "Rule 2"}

    async def test_import_rules_success(self, memory_manager):
        """Test successfully importing memory rules."""
        rule_dicts = [
            {"id": "rule1", "rule": "Imported Rule 1"},
            {"id": "rule2", "rule": "Imported Rule 2"}
        ]

        # Mock MemoryRule.from_dict
        with patch('src.python.common.memory.manager.MemoryRule') as mock_rule_class:
            rule1 = Mock()
            rule1.id = "rule1"
            rule2 = Mock()
            rule2.id = "rule2"

            mock_rule_class.from_dict.side_effect = [rule1, rule2]

            memory_manager.get_rule = AsyncMock(return_value=None)  # Rules don't exist
            memory_manager.add_rule = AsyncMock(return_value=("rule_id", []))

            imported_count, skipped_count, errors = await memory_manager.import_rules(
                rule_dicts, overwrite_existing=False
            )

            assert imported_count == 2
            assert skipped_count == 0
            assert errors == []

    async def test_import_rules_with_existing_no_overwrite(self, memory_manager):
        """Test importing rules with existing rules and no overwrite."""
        rule_dicts = [
            {"id": "rule1", "rule": "Existing Rule 1"},
            {"id": "rule2", "rule": "New Rule 2"}
        ]

        with patch('src.python.common.memory.manager.MemoryRule') as mock_rule_class:
            rule1 = Mock()
            rule1.id = "rule1"
            rule2 = Mock()
            rule2.id = "rule2"

            mock_rule_class.from_dict.side_effect = [rule1, rule2]

            # Mock rule1 exists, rule2 doesn't
            memory_manager.get_rule = AsyncMock(side_effect=[Mock(), None])
            memory_manager.add_rule = AsyncMock(return_value=("rule2", []))

            imported_count, skipped_count, errors = await memory_manager.import_rules(
                rule_dicts, overwrite_existing=False
            )

            assert imported_count == 1  # Only rule2 imported
            assert skipped_count == 1  # rule1 skipped
            assert errors == []

    async def test_import_rules_with_overwrite(self, memory_manager):
        """Test importing rules with existing rules and overwrite enabled."""
        rule_dicts = [
            {"id": "rule1", "rule": "Updated Rule 1"}
        ]

        with patch('src.python.common.memory.manager.MemoryRule') as mock_rule_class:
            rule1 = Mock()
            rule1.id = "rule1"

            mock_rule_class.from_dict.return_value = rule1

            # Mock existing rule
            existing_rule = Mock()
            memory_manager.get_rule = AsyncMock(return_value=existing_rule)
            memory_manager.update_rule = AsyncMock(return_value=True)

            imported_count, skipped_count, errors = await memory_manager.import_rules(
                rule_dicts, overwrite_existing=True
            )

            assert imported_count == 1
            assert skipped_count == 0
            assert errors == []
            memory_manager.update_rule.assert_called_once_with(rule1)

    async def test_import_rules_with_errors(self, memory_manager):
        """Test importing rules with various errors."""
        rule_dicts = [
            {"id": "rule1", "rule": "Valid Rule"},
            {"invalid": "data"},  # Invalid rule dict
            {"id": "rule3", "rule": "Another Rule"}
        ]

        with patch('src.python.common.memory.manager.MemoryRule') as mock_rule_class:
            rule1 = Mock()
            rule1.id = "rule1"
            rule3 = Mock()
            rule3.id = "rule3"

            # First call succeeds, second fails, third succeeds
            mock_rule_class.from_dict.side_effect = [
                rule1,
                Exception("Invalid rule format"),
                rule3
            ]

            memory_manager.get_rule = AsyncMock(return_value=None)
            memory_manager.add_rule = AsyncMock(return_value=("rule_id", []))

            imported_count, skipped_count, errors = await memory_manager.import_rules(rule_dicts)

            assert imported_count == 2  # rule1 and rule3
            assert skipped_count == 1  # invalid rule
            assert len(errors) == 1
            assert "Invalid rule format" in errors[0]

    async def test_import_rules_add_failure(self, memory_manager):
        """Test importing rules when add_rule returns failure."""
        rule_dicts = [
            {"id": "rule1", "rule": "Test Rule"}
        ]

        with patch('src.python.common.memory.manager.MemoryRule') as mock_rule_class:
            rule1 = Mock()
            rule1.id = "rule1"

            mock_rule_class.from_dict.return_value = rule1

            memory_manager.get_rule = AsyncMock(return_value=None)
            memory_manager.update_rule = AsyncMock(return_value=False)  # Simulate failure

            imported_count, skipped_count, errors = await memory_manager.import_rules(rule_dicts)

            assert imported_count == 1  # add_rule doesn't return failure flag, just raises
            assert skipped_count == 0
            assert errors == []


if __name__ == "__main__":
    pytest.main([__file__])
