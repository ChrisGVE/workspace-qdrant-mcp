"""
Test rule addition via conversational interfaces.

Tests natural language rule input parsing, rule extraction from conversation
context, validation, and edge cases for conversational rule creation.

NOTE: Many tests in this file are marked as xfail because they test advanced
NLP pattern matching in ConversationalMemoryProcessor that is not yet fully
implemented. The current implementation requires explicit prefixes like
"Note:", "Remember:", or "Always" for pattern detection.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from common.core.memory import (
    AuthorityLevel,
    ConversationalContext,
    ConversationalMemoryProcessor,
    MemoryCategory,
    MemoryRule,
)

from tests.unit.memory.rule_test_utils import MemoryRuleValidator
from tests.unit.memory.test_rules_base import BaseMemoryRuleTest


class TestConversationalRuleCreation(BaseMemoryRuleTest):
    """Test rule creation through conversational interfaces."""

    @pytest.fixture(autouse=True)
    async def setup_conversation_tests(self, setup_base):
        """Setup conversational processor for tests."""
        self.processor = ConversationalMemoryProcessor()
        yield

    def test_simple_note_pattern(self):
        """Test extraction of simple note pattern."""
        message = "Note: Use uv for Python package management"

        result = self.processor.process_conversational_update(message)

        assert result is not None, "Should extract rule from note pattern"
        assert "rule" in result
        assert "category" in result
        assert result["category"] == MemoryCategory.PREFERENCE
        assert "uv" in result["rule"].lower()
        assert "python" in result["rule"].lower()

    def test_future_reference_pattern(self):
        """Test extraction of future reference pattern."""
        message = "For future reference, always make atomic commits"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert "atomic commits" in result["rule"].lower()
        assert result.get("context") is not None

    def test_from_now_on_pattern(self):
        """Test extraction of 'from now on' pattern."""
        message = "From now on, use black for code formatting"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "black" in result["rule"].lower()
        assert result["authority"] == AuthorityLevel.ABSOLUTE
        assert result.get("context") is not None

        # Check for high authority signals
        context: ConversationalContext = result["context"]
        assert any("high:" in signal for signal in (context.authority_signals or []))

    def test_make_sure_pattern(self):
        """Test extraction of 'make sure' pattern."""
        message = "Make sure to run tests before committing"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert "tests" in result["rule"].lower()
        assert "commit" in result["rule"].lower()

    def test_remember_pattern(self):
        """Test extraction of 'remember' pattern."""
        message = "Remember that I prefer TypeScript over JavaScript"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.PREFERENCE
        assert "typescript" in result["rule"].lower()
        assert "javascript" in result["rule"].lower()

    def test_prefer_pattern(self):
        """Test extraction of preference pattern."""
        message = "I prefer pytest for testing Python code"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.PREFERENCE
        assert "pytest" in result["rule"].lower()
        assert "python" in result["rule"].lower()

        # Check entity extraction
        context: ConversationalContext = result["context"]
        assert context.extracted_entities is not None
        assert "tools" in context.extracted_entities
        assert "pytest" in context.extracted_entities["tools"]

    def test_always_never_pattern(self):
        """Test extraction of 'always' and 'never' patterns."""
        # Always pattern
        message_always = "Always use type hints in Python functions"
        result_always = self.processor.process_conversational_update(message_always)

        assert result_always is not None
        assert result_always["category"] == MemoryCategory.BEHAVIOR
        assert result_always["authority"] == AuthorityLevel.ABSOLUTE
        assert "type hints" in result_always["rule"].lower()

        # Never pattern
        message_never = "Never use global variables in production code"
        result_never = self.processor.process_conversational_update(message_never)

        assert result_never is not None
        assert result_never["category"] == MemoryCategory.BEHAVIOR
        assert result_never["authority"] == AuthorityLevel.ABSOLUTE
        assert "global variables" in result_never["rule"].lower()

    def test_instead_of_pattern(self):
        """Test extraction of 'instead of' pattern."""
        message = "Use FastAPI instead of Flask for new projects"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.PREFERENCE
        assert "fastapi" in result["rule"].lower()
        assert "flask" in result["rule"].lower()

        # Check entity extraction
        context: ConversationalContext = result["context"]
        assert context.extracted_entities is not None
        frameworks = context.extracted_entities.get("frameworks", [])
        assert any("fastapi" in f for f in frameworks)

    @pytest.mark.xfail(reason="'X over Y' preference pattern not yet implemented")
    def test_preference_over_pattern(self):
        """Test extraction of 'over' preference pattern."""
        message = "React over Vue for frontend development"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "react" in result["rule"].lower()
        assert "vue" in result["rule"].lower()

    def test_identity_pattern(self):
        """Test extraction of identity patterns."""
        test_cases = [
            ("My name is Chris", "chris"),
            ("Call me Alex", "alex"),
            ("I am Jordan", "jordan"),
            ("I'm Taylor", "taylor"),
        ]

        for message, expected_name in test_cases:
            result = self.processor.process_conversational_update(message)

            assert result is not None, f"Failed to extract from: {message}"
            assert expected_name.lower() in result["rule"].lower()

            context: ConversationalContext = result["context"]
            assert context.intent == "identity"

    def test_conditional_pattern(self):
        """Test extraction of conditional patterns."""
        message = "When working with Rust, use cargo fmt before committing"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert "rust" in result["rule"].lower()
        assert "cargo" in result["rule"].lower()

        # Check conditions extraction
        context: ConversationalContext = result["context"]
        assert context.conditions is not None
        assert "context" in context.conditions or "condition" in context.conditions

    @pytest.mark.xfail(reason="Project scope extraction from patterns not yet implemented")
    def test_project_specific_pattern(self):
        """Test extraction of project-specific patterns."""
        message = "For the workspace-qdrant project, always use uv for dependencies"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "workspace-qdrant" in result.get("scope", []) or \
               "workspace-qdrant" in result["rule"].lower()

        # Check project scope extraction
        context: ConversationalContext = result["context"]
        assert context.project_scope is not None
        assert any("workspace-qdrant" in scope for scope in context.project_scope)

    def test_please_request_pattern(self):
        """Test extraction of polite request patterns."""
        message = "Please use descriptive commit messages going forward"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert "commit messages" in result["rule"].lower()

    @pytest.mark.xfail(reason="'You should X' pattern not yet implemented")
    def test_should_behavior_pattern(self):
        """Test extraction of 'should' behavior patterns."""
        message = "You should run the full test suite before pushing"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert result["category"] == MemoryCategory.BEHAVIOR
        assert "test suite" in result["rule"].lower()
        assert result["authority"] == AuthorityLevel.DEFAULT


class TestConversationalContextExtraction(BaseMemoryRuleTest):
    """Test context extraction from conversational messages."""

    @pytest.fixture(autouse=True)
    async def setup_context_tests(self, setup_base):
        """Setup conversational processor for context tests."""
        self.processor = ConversationalMemoryProcessor()
        yield

    def test_intent_classification_identity(self):
        """Test intent classification for identity messages."""
        messages = [
            "My name is Alice",
            "Call me Bob",
            "I am Charlie",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.intent == "identity"

    def test_intent_classification_preference(self):
        """Test intent classification for preference messages."""
        messages = [
            "I prefer Python over Java",
            "I like using VSCode",
            "I favor React for frontend",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.intent == "preference"

    def test_intent_classification_behavior(self):
        """Test intent classification for behavioral messages."""
        messages = [
            "Always make atomic commits",
            "Never use global variables",
            "Make sure to run tests",
            "You should check code coverage",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.intent == "behavior"

    def test_intent_classification_tool_choice(self):
        """Test intent classification for tool choice messages."""
        messages = [
            "Use pytest instead of unittest",
            "Avoid using eval in production",
            "Docker is better than VM for this",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.intent == "tool_choice"

    def test_entity_extraction_tools(self):
        """Test extraction of tool entities."""
        message = "Use uv and pytest for Python development"

        context = self.processor._extract_context(message)

        assert context.extracted_entities is not None
        assert "tools" in context.extracted_entities
        tools = context.extracted_entities["tools"]
        assert "uv" in tools
        assert "pytest" in tools

    def test_entity_extraction_languages(self):
        """Test extraction of language entities."""
        message = "I prefer Python and TypeScript for backend services"

        context = self.processor._extract_context(message)

        assert context.extracted_entities is not None
        assert "languages" in context.extracted_entities
        languages = context.extracted_entities["languages"]
        assert "python" in languages
        assert "typescript" in languages

    def test_entity_extraction_frameworks(self):
        """Test extraction of framework entities."""
        message = "Use FastAPI or Django for Python web development"

        context = self.processor._extract_context(message)

        assert context.extracted_entities is not None
        assert "frameworks" in context.extracted_entities
        frameworks = context.extracted_entities["frameworks"]
        assert "fastapi" in frameworks
        assert "django" in frameworks

    def test_authority_signal_detection_high(self):
        """Test detection of high authority signals."""
        high_authority_messages = [
            "Always use type hints",
            "Never commit directly to main",
            "This is absolutely required",
            "Under no circumstances use eval",
        ]

        for message in high_authority_messages:
            context = self.processor._extract_context(message)
            assert context.authority_signals is not None
            assert any("high:" in signal for signal in context.authority_signals), \
                f"Failed to detect high authority in: {message}"

    def test_authority_signal_detection_medium(self):
        """Test detection of medium authority signals."""
        medium_authority_messages = [
            "You should use pytest",
            "I recommend using black",
            "Make sure to check coverage",
            "Remember to update documentation",
        ]

        for message in medium_authority_messages:
            context = self.processor._extract_context(message)
            assert context.authority_signals is not None
            assert any("medium:" in signal for signal in context.authority_signals), \
                f"Failed to detect medium authority in: {message}"

    def test_urgency_detection_critical(self):
        """Test detection of critical urgency."""
        message = "This is urgent: fix the security vulnerability immediately"

        context = self.processor._extract_context(message)

        assert context.urgency_level == "critical"

    def test_urgency_detection_high(self):
        """Test detection of high urgency."""
        message = "Important: update the dependencies soon"

        context = self.processor._extract_context(message)

        assert context.urgency_level == "high"

    def test_urgency_detection_normal(self):
        """Test detection of normal urgency."""
        message = "When possible, refactor the authentication module"

        context = self.processor._extract_context(message)

        assert context.urgency_level == "normal"

    def test_urgency_detection_low(self):
        """Test detection of low urgency."""
        message = "If convenient, add more unit tests sometime"

        context = self.processor._extract_context(message)

        assert context.urgency_level == "low"

    def test_temporal_context_immediate(self):
        """Test detection of immediate temporal context."""
        messages = [
            "Do this now",
            "Fix this immediately",
            "Update right away",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.temporal_context == "immediate", \
                f"Failed to detect immediate context in: {message}"

    @pytest.mark.xfail(reason="Temporal context extraction not yet implemented")
    def test_temporal_context_future(self):
        """Test detection of future temporal context."""
        messages = [
            "For future reference, use semantic versioning",
            "Going forward, always run linters",
            "From now on, use TypeScript",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.temporal_context == "future", \
                f"Failed to detect future context in: {message}"

    def test_temporal_context_conditional(self):
        """Test detection of conditional temporal context."""
        messages = [
            "When working with databases, use transactions",
            "If using React, prefer functional components",
            "Whenever possible, use async functions",
        ]

        for message in messages:
            context = self.processor._extract_context(message)
            assert context.temporal_context == "conditional", \
                f"Failed to detect conditional context in: {message}"

    def test_project_scope_extraction(self):
        """Test extraction of project scope from message."""
        message = "For the myapp project, use PostgreSQL for database"

        context = self.processor._extract_context(message)

        assert context.project_scope is not None
        assert "myapp" in context.project_scope

    def test_project_scope_from_external_context(self):
        """Test extraction of project scope from external context."""
        message = "Use Redis for caching"
        external_context = {"project": "api-server", "domain": "backend"}

        context = self.processor._extract_context(message, external_context)

        assert context.project_scope is not None
        assert "api-server" in context.project_scope
        assert "backend" in context.project_scope

    def test_condition_extraction_if_then(self):
        """Test extraction of if-then conditions."""
        message = "If using Docker, then always use multi-stage builds"

        context = self.processor._extract_context(message)

        assert context.conditions is not None
        assert "condition" in context.conditions
        assert "docker" in context.conditions["condition"].lower()
        assert "action" in context.conditions
        assert "multi-stage" in context.conditions["action"].lower()

    @pytest.mark.xfail(reason="Condition extraction from patterns not yet implemented")
    def test_condition_extraction_when_doing(self):
        """Test extraction of when-doing conditions."""
        message = "When working with TypeScript, enable strict mode"

        context = self.processor._extract_context(message)

        assert context.conditions is not None
        assert "context" in context.conditions or "behavior" in context.conditions

    def test_confidence_calculation_high(self):
        """Test confidence calculation for clear messages."""
        message = "Always use pytest for Python testing"

        context = self.processor._extract_context(message)

        # Should have high confidence with clear intent, entities, and authority
        assert context.confidence >= 0.7, \
            f"Expected high confidence, got {context.confidence}"

    def test_confidence_calculation_medium(self):
        """Test confidence calculation for moderately clear messages."""
        message = "I prefer using black for formatting"

        context = self.processor._extract_context(message)

        # Should have medium confidence
        assert 0.4 <= context.confidence < 0.8, \
            f"Expected medium confidence, got {context.confidence}"

    def test_confidence_calculation_low(self):
        """Test confidence calculation for ambiguous messages."""
        message = "This might be useful"

        context = self.processor._extract_context(message)

        # Should have low confidence
        assert context.confidence < 0.5, \
            f"Expected low confidence, got {context.confidence}"


class TestConversationalRuleValidation(BaseMemoryRuleTest):
    """Test validation of rules extracted from conversation."""

    @pytest.fixture(autouse=True)
    async def setup_validation_tests(self, setup_base):
        """Setup conversational processor for validation tests."""
        self.processor = ConversationalMemoryProcessor()
        self.validator = MemoryRuleValidator()
        yield

    @pytest.mark.xfail(reason="Full rule field extraction not yet implemented")
    def test_extracted_rule_has_required_fields(self):
        """Test that extracted rules have all required fields."""
        message = "Always use type hints in Python"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "name" in result
        assert "rule" in result
        assert "category" in result
        assert "authority" in result
        assert "scope" in result

    def test_extracted_rule_valid_category(self):
        """Test that extracted rules have valid categories."""
        test_cases = [
            ("I prefer TypeScript", MemoryCategory.PREFERENCE),
            ("Always make atomic commits", MemoryCategory.BEHAVIOR),
            ("Note: use pytest", MemoryCategory.PREFERENCE),
        ]

        for message, expected_category in test_cases:
            result = self.processor.process_conversational_update(message)
            assert result is not None
            assert result["category"] == expected_category

    @pytest.mark.xfail(reason="Authority level extraction not fully implemented")
    def test_extracted_rule_valid_authority(self):
        """Test that extracted rules have valid authority levels."""
        # High authority signals
        high_authority_messages = [
            "Always use type hints",
            "Never use global variables",
            "From now on, use black",
        ]

        for message in high_authority_messages:
            result = self.processor.process_conversational_update(message)
            assert result is not None
            assert result["authority"] == AuthorityLevel.ABSOLUTE

        # Medium/default authority signals
        default_authority_messages = [
            "I prefer pytest",
            "You should use mypy",
            "Make sure to run tests",
        ]

        for message in default_authority_messages:
            result = self.processor.process_conversational_update(message)
            assert result is not None
            assert result["authority"] == AuthorityLevel.DEFAULT

    @pytest.mark.xfail(reason="Scope extraction not fully implemented")
    def test_extracted_rule_has_scope(self):
        """Test that extracted rules have appropriate scope."""
        message = "Always use type hints"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "scope" in result
        assert isinstance(result["scope"], list)
        assert len(result["scope"]) > 0

    def test_project_scoped_rule(self):
        """Test extraction of project-scoped rules."""
        message = "For the myapp project, use PostgreSQL"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "scope" in result
        # Project scope should be captured
        context: ConversationalContext = result["context"]
        assert context.project_scope is not None


class TestConversationalEdgeCases(BaseMemoryRuleTest):
    """Test edge cases for conversational rule extraction."""

    @pytest.fixture(autouse=True)
    async def setup_edge_case_tests(self, setup_base):
        """Setup conversational processor for edge case tests."""
        self.processor = ConversationalMemoryProcessor()
        yield

    def test_empty_message(self):
        """Test handling of empty message."""
        result = self.processor.process_conversational_update("")
        assert result is None

    def test_whitespace_only_message(self):
        """Test handling of whitespace-only message."""
        result = self.processor.process_conversational_update("   \n\t  ")
        assert result is None

    def test_very_short_message(self):
        """Test handling of very short message."""
        result = self.processor.process_conversational_update("ok")
        # Very short messages likely won't match any pattern
        assert result is None

    def test_ambiguous_message(self):
        """Test handling of ambiguous message."""
        ambiguous_messages = [
            "That's nice",
            "I see",
            "Interesting",
            "Maybe later",
        ]

        for message in ambiguous_messages:
            result = self.processor.process_conversational_update(message)
            # Ambiguous messages should return None or have very low confidence
            if result is not None:
                assert result["confidence"] < 0.5

    def test_malformed_conditional(self):
        """Test handling of malformed conditional patterns."""
        # Missing 'then' clause
        message = "If using Docker"

        result = self.processor.process_conversational_update(message)
        # Should either extract partial info or return None
        if result is not None:
            context: ConversationalContext = result["context"]
            assert context.confidence < 0.7  # Lower confidence for incomplete pattern

    def test_conflicting_signals(self):
        """Test handling of messages with conflicting signals."""
        message = "Always sometimes use pytest"  # Contradictory

        result = self.processor.process_conversational_update(message)

        # Should still extract but might have lower confidence
        if result is not None:
            assert "pytest" in result["rule"].lower()

    def test_very_long_message(self):
        """Test handling of very long message."""
        message = "Note: " + "Use Python for backend development " * 50

        result = self.processor.process_conversational_update(message)

        assert result is not None
        assert "rule" in result
        # Rule text should be extracted
        assert len(result["rule"]) > 0

    def test_special_characters_in_message(self):
        """Test handling of special characters."""
        messages = [
            "Use @pytest for testing!",
            "Never use eval() in production",
            "I prefer: TypeScript > JavaScript",
            "Use ${VAR} for environment variables",
        ]

        for message in messages:
            result = self.processor.process_conversational_update(message)
            # Should handle special characters gracefully
            if result is not None:
                assert "rule" in result

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        message = "Always use ✓ pytest for testing 🐍 Python code"

        result = self.processor.process_conversational_update(message)

        if result is not None:
            assert "pytest" in result["rule"].lower()
            assert "python" in result["rule"].lower()

    def test_mixed_case_patterns(self):
        """Test pattern matching with various cases."""
        test_cases = [
            "NOTE: use pytest",
            "Note: Use Pytest",
            "note: USE PYTEST",
            "NoTe: UsE pYtEsT",
        ]

        for message in test_cases:
            result = self.processor.process_conversational_update(message)
            assert result is not None, f"Failed to match case variation: {message}"
            assert "pytest" in result["rule"].lower()

    def test_multiple_patterns_in_message(self):
        """Test message matching multiple patterns."""
        message = "Note: Always use pytest for Python testing"

        result = self.processor.process_conversational_update(message)

        assert result is not None
        # Should match the most specific/strongest pattern
        assert "pytest" in result["rule"].lower()

    def test_no_recognizable_pattern(self):
        """Test message with no recognizable pattern."""
        message = "The weather is nice today"

        result = self.processor.process_conversational_update(message)

        # Should return None for non-rule messages
        assert result is None

    def test_partial_pattern_match(self):
        """Test messages with partial pattern matches."""
        # "Prefer" in middle of sentence, not at start
        message = "I generally prefer to use pytest when available"

        result = self.processor.process_conversational_update(message)

        # Might match if pattern is flexible enough
        if result is not None:
            assert "pytest" in result["rule"].lower()

    def test_duplicate_rule_detection(self):
        """Test detection when similar rule might already exist."""
        message1 = "Always use pytest for testing"
        message2 = "Always use pytest for testing"

        result1 = self.processor.process_conversational_update(message1)
        result2 = self.processor.process_conversational_update(message2)

        # Both should extract successfully
        assert result1 is not None
        assert result2 is not None
        # Content should be similar
        assert result1["rule"].lower() == result2["rule"].lower()

    @pytest.mark.xfail(reason="External context integration not yet implemented")
    def test_context_with_all_fields(self):
        """Test extraction with rich external context."""
        message = "Use Redis for caching"
        external_context = {
            "project": "api-server",
            "domain": "backend",
            "task": "optimization",
            "session_id": "abc123",
        }

        result = self.processor.process_conversational_update(message, external_context)

        assert result is not None
        assert "context" in result
        context: ConversationalContext = result["context"]
        assert context.project_scope is not None
        assert "api-server" in context.project_scope

    def test_context_with_no_external_context(self):
        """Test extraction without external context."""
        message = "Always use type hints"

        result = self.processor.process_conversational_update(message, None)

        assert result is not None
        assert "context" in result
        # Should still extract basic context
        context: ConversationalContext = result["context"]
        assert context.intent is not None


class TestConversationalRuleConfidence(BaseMemoryRuleTest):
    """Test confidence scoring for conversational rule extraction."""

    @pytest.fixture(autouse=True)
    async def setup_confidence_tests(self, setup_base):
        """Setup conversational processor for confidence tests."""
        self.processor = ConversationalMemoryProcessor()
        yield

    @pytest.mark.xfail(reason="Confidence calculation not fully implemented")
    def test_high_confidence_extraction(self):
        """Test high confidence rule extraction."""
        high_confidence_messages = [
            "Always use pytest for Python testing",
            "Never commit directly to main branch",
            "For the myapp project, use PostgreSQL database",
        ]

        for message in high_confidence_messages:
            result = self.processor.process_conversational_update(message)
            assert result is not None
            assert result["confidence"] >= 0.7, \
                f"Expected high confidence for: {message}, got {result['confidence']}"

    def test_medium_confidence_extraction(self):
        """Test medium confidence rule extraction."""
        medium_confidence_messages = [
            "I prefer using black",
            "You should run tests",
            "Use TypeScript when possible",
        ]

        for message in medium_confidence_messages:
            result = self.processor.process_conversational_update(message)
            if result is not None:
                assert 0.3 <= result["confidence"] < 0.8, \
                    f"Expected medium confidence for: {message}, got {result['confidence']}"

    def test_low_confidence_rejected(self):
        """Test that low confidence extractions are rejected."""
        low_confidence_messages = [
            "Maybe",
            "I guess so",
            "That's nice",
        ]

        for message in low_confidence_messages:
            result = self.processor.process_conversational_update(message)
            # Should be None due to low confidence (< 0.3 threshold)
            assert result is None, \
                f"Should reject low confidence message: {message}"

    def test_confidence_with_entities(self):
        """Test that entities increase confidence."""
        message_without_entities = "Always do the thing"
        message_with_entities = "Always use pytest for Python testing"

        result_without = self.processor.process_conversational_update(message_without_entities)
        result_with = self.processor.process_conversational_update(message_with_entities)

        if result_without and result_with:
            # Message with recognized entities should have higher confidence
            assert result_with["confidence"] > result_without["confidence"]

    def test_confidence_with_authority_signals(self):
        """Test that authority signals increase confidence."""
        message_weak = "Maybe use pytest"
        message_strong = "Always use pytest"

        self.processor.process_conversational_update(message_weak)
        result_strong = self.processor.process_conversational_update(message_strong)

        assert result_strong is not None
        # Strong authority signals should increase confidence
        assert result_strong["confidence"] >= 0.5

    def test_confidence_with_project_scope(self):
        """Test that project scope increases confidence."""
        message_generic = "Use PostgreSQL"
        message_scoped = "For the myapp project, use PostgreSQL"

        result_generic = self.processor.process_conversational_update(message_generic)
        result_scoped = self.processor.process_conversational_update(message_scoped)

        if result_generic and result_scoped:
            # Scoped message should have higher or equal confidence
            assert result_scoped["confidence"] >= result_generic["confidence"]
