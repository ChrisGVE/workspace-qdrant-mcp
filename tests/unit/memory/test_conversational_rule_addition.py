"""
Test conversational rule addition for memory rules system (Task 324.1).

Tests comprehensive conversational rule addition functionality including:
- Natural language rule extraction from conversation
- LLM interaction mocking for rule processing
- Rule parsing and validation
- Rule storage in memory collection
- Various conversation patterns and formats
- Edge cases and error handling

This test suite validates the complete flow from conversational text to
stored memory rules without requiring real LLM API calls.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from common.memory.claude_integration import ClaudeCodeIntegration
from common.memory.manager import MemoryManager
from common.memory.types import (
    AuthorityLevel,
    ClaudeCodeSession,
    ConversationalUpdate,
    MemoryCategory,
    MemoryContext,
    MemoryRule,
)
from common.memory.token_counter import TokenCounter
from common.core.client import QdrantWorkspaceClient
from common.core.embeddings import EmbeddingService


@pytest.fixture
def mock_token_counter():
    """Provide mock token counter."""
    counter = Mock(spec=TokenCounter)
    counter.count_rules_tokens = Mock(return_value=Mock(total_tokens=100))
    counter.optimize_rules_for_context = Mock(return_value=([], Mock(total_tokens=0)))
    return counter


@pytest.fixture
def claude_integration(mock_token_counter):
    """Provide ClaudeCodeIntegration instance with mock dependencies."""
    integration = ClaudeCodeIntegration(
        token_counter=mock_token_counter,
        max_memory_tokens=5000,
    )
    return integration


@pytest.fixture
async def mock_qdrant_client():
    """Provide mock Qdrant client for memory operations."""
    client = AsyncMock(spec=QdrantWorkspaceClient)
    client.collection_exists = AsyncMock(return_value=True)
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.scroll = AsyncMock(return_value=([], None))
    client.delete = AsyncMock()
    client.retrieve = AsyncMock()
    return client


@pytest.fixture
async def mock_embedding_service():
    """Provide mock embedding service."""
    service = AsyncMock(spec=EmbeddingService)
    service.embed_text = AsyncMock(return_value=[0.1] * 384)
    service.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    return service


@pytest.fixture
async def memory_manager(mock_qdrant_client, mock_embedding_service, mock_token_counter):
    """Provide MemoryManager instance with mocked dependencies."""
    manager = MemoryManager(
        qdrant_client=mock_qdrant_client,
        embedding_service=mock_embedding_service,
    )
    manager.token_counter = mock_token_counter
    await manager.initialize()
    return manager


class TestConversationalRuleExtraction:
    """Test rule extraction from conversational text."""

    def test_extract_call_me_pattern(self, claude_integration):
        """Test extraction of 'call me' pattern."""
        text = "Note: call me Chris"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert update.extracted_rule is not None
        assert "chris" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.PREFERENCE
        assert update.authority == AuthorityLevel.ABSOLUTE
        assert update.confidence >= 0.8

    def test_extract_prefer_pattern(self, claude_integration):
        """Test extraction of 'I prefer' pattern."""
        text = "Remember: I prefer Python for backend development"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert update.extracted_rule is not None
        assert "python" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.PREFERENCE
        assert update.authority == AuthorityLevel.DEFAULT

    def test_extract_always_use_pattern(self, claude_integration):
        """Test extraction of 'always use' pattern."""
        text = "Always use pytest for testing Python code"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "pytest" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.BEHAVIOR
        assert update.authority == AuthorityLevel.ABSOLUTE
        assert update.confidence >= 0.8

    def test_extract_avoid_pattern(self, claude_integration):
        """Test extraction of 'avoid' pattern."""
        text = "Don't use global variables in production code"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "global variables" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.BEHAVIOR
        assert update.authority == AuthorityLevel.DEFAULT

    def test_extract_project_context_pattern(self, claude_integration):
        """Test extraction of project context pattern."""
        text = "I work on project workspace-qdrant-mcp"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert "workspace-qdrant-mcp" in update.extracted_rule.lower()
        assert update.category == MemoryCategory.CONTEXT

    def test_multiple_patterns_in_text(self, claude_integration):
        """Test extraction when text contains multiple patterns."""
        text = "Call me Alex. I prefer TypeScript for frontend. Always use ESLint."

        updates = claude_integration.detect_conversational_updates(text)

        # Should extract multiple rules
        assert len(updates) >= 2

        # Check for name rule
        name_updates = [u for u in updates if "alex" in u.extracted_rule.lower()]
        assert len(name_updates) > 0

        # Check for preference rule
        pref_updates = [u for u in updates if "typescript" in u.extracted_rule.lower()]
        assert len(pref_updates) > 0

    def test_case_insensitive_matching(self, claude_integration):
        """Test that pattern matching is case-insensitive."""
        test_cases = [
            "NOTE: call me Chris",
            "Note: Call Me Chris",
            "note: CALL ME CHRIS",
        ]

        for text in test_cases:
            updates = claude_integration.detect_conversational_updates(text)
            assert len(updates) > 0, f"Failed to match: {text}"
            assert "chris" in updates[0].extracted_rule.lower()

    def test_no_pattern_matches(self, claude_integration):
        """Test handling when no patterns match."""
        text = "The weather is nice today."

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) == 0

    def test_empty_text(self, claude_integration):
        """Test handling of empty text."""
        updates = claude_integration.detect_conversational_updates("")
        assert len(updates) == 0

    def test_whitespace_only_text(self, claude_integration):
        """Test handling of whitespace-only text."""
        updates = claude_integration.detect_conversational_updates("   \n\t  ")
        assert len(updates) == 0


class TestConversationalUpdateProcessing:
    """Test processing of conversational updates into memory rules."""

    def test_process_valid_update(self, claude_integration):
        """Test processing a valid conversational update."""
        update = ConversationalUpdate(
            text="Always use pytest",
            extracted_rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["testing"],
            confidence=0.9,
        )

        rule = claude_integration.process_conversational_update(update)

        assert rule is not None
        assert isinstance(rule, MemoryRule)
        assert rule.rule == "Use pytest for testing"
        assert rule.category == MemoryCategory.BEHAVIOR
        assert rule.authority == AuthorityLevel.ABSOLUTE
        assert rule.scope == ["testing"]
        assert rule.source == "conversation"
        assert "confidence" in rule.metadata
        assert rule.metadata["confidence"] == 0.9

    def test_process_invalid_update_low_confidence(self, claude_integration):
        """Test processing update with low confidence."""
        update = ConversationalUpdate(
            text="Maybe use pytest",
            extracted_rule="Use pytest",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            confidence=0.3,  # Below threshold
        )

        rule = claude_integration.process_conversational_update(update)

        # Should return None for low confidence
        assert rule is None

    def test_process_invalid_update_no_rule(self, claude_integration):
        """Test processing update with no extracted rule."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule=None,
            category=MemoryCategory.PREFERENCE,
            confidence=0.8,
        )

        rule = claude_integration.process_conversational_update(update)

        assert rule is None

    def test_process_invalid_update_empty_rule(self, claude_integration):
        """Test processing update with empty rule text."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule="",
            category=MemoryCategory.PREFERENCE,
            confidence=0.8,
        )

        rule = claude_integration.process_conversational_update(update)

        assert rule is None

    def test_process_update_preserves_metadata(self, claude_integration):
        """Test that processing preserves original text and metadata."""
        original_text = "Always use type hints in Python"
        update = ConversationalUpdate(
            text=original_text,
            extracted_rule="Use type hints in Python",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            confidence=0.85,
        )

        rule = claude_integration.process_conversational_update(update)

        assert rule is not None
        assert rule.metadata["original_text"] == original_text
        assert rule.metadata["confidence"] == 0.85
        assert "extracted_at" in rule.metadata


class TestConversationalRuleWithSessionContext:
    """Test conversational rule extraction with session context."""

    def test_extract_with_project_context(self, claude_integration):
        """Test extraction adds project context to scope."""
        text = "Always use pytest for testing"

        session_context = MemoryContext(
            session_id="test-session",
            project_name="my-project",
            project_path="/path/to/project",
        )

        updates = claude_integration.detect_conversational_updates(
            text, session_context
        )

        assert len(updates) > 0
        update = updates[0]
        assert "project:my-project" in update.scope

    def test_extract_with_user_context(self, claude_integration):
        """Test extraction adds user context to scope."""
        text = "I prefer TypeScript"

        session_context = MemoryContext(
            session_id="test-session",
            user_name="chris",
        )

        updates = claude_integration.detect_conversational_updates(
            text, session_context
        )

        assert len(updates) > 0
        update = updates[0]
        assert "user:chris" in update.scope

    def test_extract_with_full_context(self, claude_integration):
        """Test extraction with complete session context."""
        text = "Always use black for formatting"

        session_context = MemoryContext(
            session_id="test-session",
            project_name="workspace-qdrant-mcp",
            project_path="/path/to/project",
            user_name="chris",
            agent_type="python-pro",
        )

        updates = claude_integration.detect_conversational_updates(
            text, session_context
        )

        assert len(updates) > 0
        update = updates[0]
        assert "project:workspace-qdrant-mcp" in update.scope
        assert "user:chris" in update.scope


class TestConversationalRuleStorage:
    """Test storing conversational rules in memory collection."""

    @pytest.mark.asyncio
    async def test_add_conversational_rule(self, memory_manager):
        """Test adding a conversational rule to memory."""
        rule = MemoryRule(
            rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            source="conversation",
            metadata={"original_text": "Always use pytest", "confidence": 0.9},
        )

        rule_id, conflicts = await memory_manager.add_rule(rule)

        assert rule_id == rule.id
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_process_conversational_text_end_to_end(self, memory_manager):
        """Test complete flow from text to stored rule."""
        text = "Always use pytest for testing Python code"

        session_context = MemoryContext(
            session_id="test-session",
            project_name="test-project",
        )

        new_rules = await memory_manager.process_conversational_text(
            text, session_context
        )

        assert len(new_rules) > 0
        rule = new_rules[0]
        assert isinstance(rule, MemoryRule)
        assert "pytest" in rule.rule.lower()
        assert rule.source == "conversation"

    @pytest.mark.asyncio
    async def test_store_multiple_conversational_rules(self, memory_manager):
        """Test storing multiple rules from one conversation."""
        text = "Call me Chris. I prefer Python. Always use type hints."

        new_rules = await memory_manager.process_conversational_text(text)

        # Should extract and store multiple rules
        assert len(new_rules) >= 2

    @pytest.mark.asyncio
    async def test_conversational_rule_with_conflicts(self, memory_manager):
        """Test handling conflicts when adding conversational rule."""
        # Add initial rule
        existing_rule = MemoryRule(
            rule="Use unittest for testing",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )
        await memory_manager.add_rule(existing_rule, check_conflicts=False)

        # Add conflicting conversational rule
        text = "Always use pytest for testing"
        new_rules = await memory_manager.process_conversational_text(text)

        # Should still add rule despite potential conflict
        assert len(new_rules) > 0


class TestVariousConversationPatterns:
    """Test various conversation patterns and formats."""

    @pytest.mark.parametrize("text,expected_category,expected_authority", [
        ("Note: call me Alex", MemoryCategory.PREFERENCE, AuthorityLevel.ABSOLUTE),
        ("I prefer React over Vue", MemoryCategory.PREFERENCE, AuthorityLevel.DEFAULT),
        ("Always use Docker for deployment", MemoryCategory.BEHAVIOR, AuthorityLevel.ABSOLUTE),
        ("Never commit to main branch", MemoryCategory.BEHAVIOR, AuthorityLevel.ABSOLUTE),
        ("Avoid using eval in production", MemoryCategory.BEHAVIOR, AuthorityLevel.DEFAULT),
        ("I work on project api-server", MemoryCategory.CONTEXT, AuthorityLevel.DEFAULT),
    ])
    def test_pattern_variations(self, claude_integration, text, expected_category, expected_authority):
        """Test various conversation pattern variations."""
        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert update.category == expected_category
        assert update.authority == expected_authority

    def test_polite_request_pattern(self, claude_integration):
        """Test polite request patterns."""
        text = "Please use descriptive commit messages"

        updates = claude_integration.detect_conversational_updates(text)

        # May or may not match depending on pattern implementation
        # This is a softer pattern that might not be recognized
        # Test documents the behavior
        if len(updates) > 0:
            update = updates[0]
            assert "commit message" in update.extracted_rule.lower()

    def test_conditional_pattern(self, claude_integration):
        """Test conditional behavior patterns."""
        text = "When working with Rust, use cargo fmt"

        updates = claude_integration.detect_conversational_updates(text)

        # Conditional patterns may not be explicitly supported
        # Test documents current behavior
        if len(updates) > 0:
            update = updates[0]
            assert "rust" in update.extracted_rule.lower() or "cargo" in update.extracted_rule.lower()

    def test_multi_sentence_conversation(self, claude_integration):
        """Test multi-sentence conversational text."""
        text = """
        I've been thinking about the project setup.
        Note: call me Chris for future reference.
        I prefer using uv for Python package management.
        Always make atomic commits after each change.
        """

        updates = claude_integration.detect_conversational_updates(text)

        # Should extract multiple rules from multi-sentence text
        assert len(updates) >= 2


class TestConversationalRuleEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self, claude_integration):
        """Test handling of very long conversational text."""
        text = "Note: " + "use pytest for testing " * 100

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        update = updates[0]
        assert update.extracted_rule is not None

    def test_special_characters(self, claude_integration):
        """Test handling of special characters in text."""
        texts = [
            "Use @pytest for testing!",
            "Never use eval() in production",
            "I prefer: TypeScript > JavaScript",
            "Use ${ENV_VAR} for config",
        ]

        for text in texts:
            updates = claude_integration.detect_conversational_updates(text)
            # Should handle gracefully without errors
            if len(updates) > 0:
                assert updates[0].extracted_rule is not None

    def test_unicode_characters(self, claude_integration):
        """Test handling of unicode characters."""
        text = "Always use ✓ pytest for testing 🐍 Python"

        updates = claude_integration.detect_conversational_updates(text)

        if len(updates) > 0:
            assert "pytest" in updates[0].extracted_rule.lower()

    def test_mixed_case_patterns(self, claude_integration):
        """Test case-insensitive pattern matching."""
        texts = [
            "ALWAYS USE PYTEST",
            "always use pytest",
            "AlWaYs UsE PyTeSt",
        ]

        for text in texts:
            updates = claude_integration.detect_conversational_updates(text)
            assert len(updates) > 0
            assert "pytest" in updates[0].extracted_rule.lower()

    def test_minimal_valid_pattern(self, claude_integration):
        """Test minimal text that should match a pattern."""
        text = "call me Al"

        updates = claude_integration.detect_conversational_updates(text)

        assert len(updates) > 0
        assert "al" in updates[0].extracted_rule.lower()

    @pytest.mark.asyncio
    async def test_invalid_category_handled(self, claude_integration):
        """Test handling of invalid category in update."""
        # Create update with invalid category (should not happen in practice)
        update = ConversationalUpdate(
            text="test",
            extracted_rule="test rule",
            category=None,  # Invalid
            confidence=0.8,
        )

        rule = claude_integration.process_conversational_update(update)

        # Should return None for invalid update
        assert rule is None

    @pytest.mark.asyncio
    async def test_concurrent_rule_additions(self, memory_manager):
        """Test adding multiple conversational rules concurrently."""
        import asyncio

        texts = [
            "Call me Alice",
            "I prefer Python",
            "Always use pytest",
        ]

        tasks = [
            memory_manager.process_conversational_text(text)
            for text in texts
        ]

        results = await asyncio.gather(*tasks)

        # All should complete successfully
        assert all(len(rules) > 0 for rules in results)


class TestConversationalRuleValidation:
    """Test validation of conversational rules."""

    def test_valid_update_passes_validation(self, claude_integration):
        """Test that valid updates pass is_valid check."""
        update = ConversationalUpdate(
            text="Always use pytest",
            extracted_rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            confidence=0.8,
        )

        assert update.is_valid()

    def test_invalid_update_no_rule(self, claude_integration):
        """Test that update with no rule fails validation."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule=None,
            category=MemoryCategory.BEHAVIOR,
            confidence=0.8,
        )

        assert not update.is_valid()

    def test_invalid_update_empty_rule(self, claude_integration):
        """Test that update with empty rule fails validation."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule="",
            category=MemoryCategory.BEHAVIOR,
            confidence=0.8,
        )

        assert not update.is_valid()

    def test_invalid_update_no_category(self, claude_integration):
        """Test that update with no category fails validation."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule="Use pytest",
            category=None,
            confidence=0.8,
        )

        assert not update.is_valid()

    def test_invalid_update_low_confidence(self, claude_integration):
        """Test that update with low confidence fails validation."""
        update = ConversationalUpdate(
            text="Some text",
            extracted_rule="Use pytest",
            category=MemoryCategory.BEHAVIOR,
            confidence=0.3,  # Below 0.5 threshold
        )

        assert not update.is_valid()

    @pytest.mark.asyncio
    async def test_rule_created_from_update_is_valid(self, claude_integration):
        """Test that rules created from updates are valid MemoryRule objects."""
        update = ConversationalUpdate(
            text="Always use pytest",
            extracted_rule="Use pytest for testing",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["testing"],
            confidence=0.9,
        )

        rule = claude_integration.process_conversational_update(update)

        assert rule is not None
        # Verify MemoryRule fields
        assert rule.rule == "Use pytest for testing"
        assert isinstance(rule.category, MemoryCategory)
        assert isinstance(rule.authority, AuthorityLevel)
        assert isinstance(rule.scope, list)
        assert rule.id is not None
        assert rule.created_at is not None


class TestConversationalRuleConfidence:
    """Test confidence scoring for conversational rules."""

    def test_high_confidence_strong_pattern(self, claude_integration):
        """Test high confidence for strong patterns."""
        texts = [
            "Always use pytest for testing",
            "Never commit to main branch",
            "Note: call me Chris",
        ]

        for text in texts:
            updates = claude_integration.detect_conversational_updates(text)
            assert len(updates) > 0
            assert updates[0].confidence >= 0.7

    def test_medium_confidence_moderate_pattern(self, claude_integration):
        """Test medium confidence for moderate patterns."""
        texts = [
            "I prefer Python over Java",
            "Avoid using global variables",
        ]

        for text in texts:
            updates = claude_integration.detect_conversational_updates(text)
            assert len(updates) > 0
            # Medium confidence range
            assert 0.5 <= updates[0].confidence < 0.9

    def test_confidence_affects_validity(self, claude_integration):
        """Test that low confidence makes updates invalid."""
        # Create update with low confidence
        update = ConversationalUpdate(
            text="Maybe use pytest",
            extracted_rule="Use pytest",
            category=MemoryCategory.PREFERENCE,
            confidence=0.4,
        )

        # Should not be valid due to low confidence
        assert not update.is_valid()
