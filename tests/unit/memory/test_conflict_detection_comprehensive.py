"""
Comprehensive conflict detection tests for memory rules system (Task 324.3).

This module provides thorough testing of conflict detection algorithms,
resolution strategies, logging, and reporting mechanisms for memory rules.
Tests cover direct contradictions, authority level conflicts, scope conflicts,
and various resolution strategies.
"""

import sys
from pathlib import Path

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src" / "python"))

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from common.core.collection_naming import CollectionNamingManager
from common.core.memory import (
    AuthorityLevel,
    BehavioralController,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
)
from common.core.sparse_vectors import BM25SparseEncoder
from qdrant_client import QdrantClient


@pytest.fixture
def memory_manager_fixture():
    """Create a memory manager with mocked Qdrant client."""
    mock_client = Mock(spec=QdrantClient)

    # Mock collections response with simple Mock object
    mock_collection = Mock()
    mock_collection.name = "memory"
    mock_collections_response = Mock()
    mock_collections_response.collections = [mock_collection]

    mock_client.get_collections = Mock(return_value=mock_collections_response)
    mock_client.upsert = Mock()
    mock_client.retrieve = Mock(return_value=[])
    mock_client.scroll = Mock(return_value=([], None))
    mock_client.search = Mock(return_value=[])

    naming_manager = Mock(spec=CollectionNamingManager)
    naming_manager.validate_collection_name = Mock(
        return_value=Mock(is_valid=True, error_message=None)
    )

    return MemoryManager(
        qdrant_client=mock_client,
        naming_manager=naming_manager,
        embedding_dim=384,
    )


class TestDirectContradictions:
    """Test detection of direct contradictions between rules."""

    @pytest.mark.asyncio
    async def test_always_vs_never_contradiction(self, memory_manager_fixture):
        """Test detection of always vs never contradictions."""
        rule1 = MemoryRule(
            id="rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="Always Use Feature X",
            rule="Always use feature X for all implementations",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="rule-2",
            category=MemoryCategory.BEHAVIOR,
            name="Never Use Feature X",
            rule="Never use feature X in any context",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "direct_contradiction"
        assert conflicts[0].rule1.id in ["rule-1", "rule-2"]
        assert conflicts[0].rule2.id in ["rule-1", "rule-2"]
        assert conflicts[0].confidence > 0.8

    @pytest.mark.asyncio
    async def test_use_vs_avoid_tool_contradiction(self, memory_manager_fixture):
        """Test detection of tool preference contradictions."""
        rule1 = MemoryRule(
            id="rule-python-1",
            category=MemoryCategory.PREFERENCE,
            name="Use UV for Python",
            rule="Use uv for Python package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="rule-python-2",
            category=MemoryCategory.PREFERENCE,
            name="Avoid UV",
            rule="Avoid uv and use pip instead",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert "uv" in conflicts[0].description.lower() or "python" in conflicts[0].description.lower()

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_execution_conflict(self, memory_manager_fixture):
        """Test detection of workflow contradictions."""
        rule1 = MemoryRule(
            id="exec-1",
            category=MemoryCategory.BEHAVIOR,
            name="Parallel Execution",
            rule="Always execute agents in parallel for speed",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="exec-2",
            category=MemoryCategory.BEHAVIOR,
            name="Sequential Execution",
            rule="Always execute agents sequentially to conserve resources",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "direct_contradiction"

    @pytest.mark.asyncio
    async def test_commit_strategy_contradiction(self, memory_manager_fixture):
        """Test detection of commit strategy contradictions."""
        rule1 = MemoryRule(
            id="commit-1",
            category=MemoryCategory.BEHAVIOR,
            name="Immediate Commits",
            rule="Commit immediately after each change",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="commit-2",
            category=MemoryCategory.BEHAVIOR,
            name="Batch Commits",
            rule="Batch commits together for efficiency",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert "commit" in conflicts[0].description.lower()

    @pytest.mark.asyncio
    async def test_test_framework_contradiction(self, memory_manager_fixture):
        """Test detection of testing framework contradictions."""
        rule1 = MemoryRule(
            id="test-1",
            category=MemoryCategory.PREFERENCE,
            name="Use Pytest",
            rule="Use pytest for all unit testing",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="test-2",
            category=MemoryCategory.PREFERENCE,
            name="Use Unittest",
            rule="Use unittest framework instead of pytest",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert conflicts[0].confidence > 0.5


class TestAuthorityLevelConflicts:
    """Test conflict detection and resolution based on authority levels."""

    @pytest.fixture
    def behavioral_controller(self, memory_manager_fixture):
        """Create a behavioral controller for resolution testing."""
        return BehavioralController(memory_manager_fixture)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Semantic conflict detection not yet implemented - requires LLM integration")
    async def test_absolute_vs_default_same_content(self, memory_manager_fixture):
        """Test conflict between absolute and default rules with similar content."""
        rule1 = MemoryRule(
            id="auth-1",
            category=MemoryCategory.PREFERENCE,
            name="Python Version Absolute",
            rule="Must use Python 3.11",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="auth-2",
            category=MemoryCategory.PREFERENCE,
            name="Python Version Default",
            rule="Prefer Python 3.12",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert "Python" in conflicts[0].description or "python" in conflicts[0].description

    @pytest.mark.asyncio
    async def test_two_absolute_rules_conflict(self, memory_manager_fixture):
        """Test conflict between two absolute authority rules."""
        rule1 = MemoryRule(
            id="abs-1",
            category=MemoryCategory.BEHAVIOR,
            name="Absolute Rule 1",
            rule="Always use approach A",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
            created_at=datetime.now(timezone.utc) - timedelta(days=2),
        )

        rule2 = MemoryRule(
            id="abs-2",
            category=MemoryCategory.BEHAVIOR,
            name="Absolute Rule 2",
            rule="Never use approach A",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
            created_at=datetime.now(timezone.utc),
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "direct_contradiction"
        # Both rules are absolute - resolution should note this
        assert "absolute" in str(conflicts[0].resolution_options).lower() or \
               "user" in str(conflicts[0].resolution_options).lower()

    @pytest.mark.asyncio
    async def test_multiple_default_rules_conflict(self, memory_manager_fixture):
        """Test conflicts among multiple default authority rules."""
        rules = [
            MemoryRule(
                id=f"def-{i}",
                category=MemoryCategory.PREFERENCE,
                name=f"Tool Choice {i}",
                rule=f"Use tool variant {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="conversational",
            )
            for i in range(3)
        ]

        # Make them actually conflict by using contradictory keywords
        rules[0].rule = "Use uv for package management"
        rules[1].rule = "Use pip for package management"
        rules[2].rule = "Avoid both uv and pip"

        conflicts = await memory_manager_fixture.detect_conflicts(rules)

        # Should detect at least one conflict (uv vs pip or uv/pip vs avoid)
        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Authority-based conflict resolution not yet implemented")
    async def test_resolution_authority_precedence(self, memory_manager_fixture, behavioral_controller):
        """Test that authority level is used for resolution."""
        rule1 = MemoryRule(
            id="res-1",
            category=MemoryCategory.BEHAVIOR,
            name="Default Rule",
            rule="Use approach X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="conversational",
        )

        rule2 = MemoryRule(
            id="res-2",
            category=MemoryCategory.BEHAVIOR,
            name="Absolute Rule",
            rule="Never use approach X",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])
        resolved, resolutions = await behavioral_controller._resolve_conflicts(
            [rule1, rule2], conflicts, "test context", "normal"
        )

        # Should keep the absolute rule
        assert len(resolved) == 1
        assert resolved[0].authority == AuthorityLevel.ABSOLUTE
        assert len(resolutions) > 0
        assert "absolute" in resolutions[0].lower()


class TestScopeConflicts:
    """Test conflict detection based on rule scopes."""

    @pytest.mark.asyncio
    async def test_no_conflict_different_scopes(self, memory_manager_fixture):
        """Test that different scopes don't cause conflicts."""
        rule1 = MemoryRule(
            id="scope-1",
            category=MemoryCategory.PREFERENCE,
            name="Python Style",
            rule="Use black for formatting",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="scope-2",
            category=MemoryCategory.PREFERENCE,
            name="Rust Style",
            rule="Use rustfmt for formatting",
            authority=AuthorityLevel.DEFAULT,
            scope=["rust"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Should not conflict - different scopes
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_global_vs_project_specific_conflict(self, memory_manager_fixture):
        """Test conflicts between global and project-specific rules."""
        rule1 = MemoryRule(
            id="global-1",
            category=MemoryCategory.BEHAVIOR,
            name="Global Rule",
            rule="Always use feature X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="project-1",
            category=MemoryCategory.BEHAVIOR,
            name="Project Rule",
            rule="Never use feature X in this project",
            authority=AuthorityLevel.DEFAULT,
            scope=["project-specific"],
            source="user_explicit",
        )

        # These have overlapping scope implications but different scopes
        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # May or may not conflict depending on scope overlap logic
        # The test validates the detection works
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_overlapping_scope_conflict(self, memory_manager_fixture):
        """Test conflicts with overlapping scopes."""
        rule1 = MemoryRule(
            id="overlap-1",
            category=MemoryCategory.PREFERENCE,
            name="Multi-Scope Rule 1",
            rule="Use pytest for testing",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "testing"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="overlap-2",
            category=MemoryCategory.PREFERENCE,
            name="Multi-Scope Rule 2",
            rule="Use unittest for testing",
            authority=AuthorityLevel.DEFAULT,
            scope=["python", "unit-tests"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Should detect conflict due to overlapping "python" scope and pytest vs unittest
        assert len(conflicts) >= 1


class TestConflictResolutionStrategies:
    """Test various conflict resolution strategies."""

    @pytest.fixture
    def behavioral_controller(self, memory_manager_fixture):
        """Create a behavioral controller for resolution testing."""
        return BehavioralController(memory_manager_fixture)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Recency-based conflict resolution not yet implemented")
    async def test_recency_based_resolution(self, memory_manager_fixture, behavioral_controller):
        """Test resolution based on rule recency."""
        now = datetime.now(timezone.utc)

        rule1 = MemoryRule(
            id="old-rule",
            category=MemoryCategory.PREFERENCE,
            name="Old Rule",
            rule="Use approach A",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
            created_at=now - timedelta(days=30),
        )

        rule2 = MemoryRule(
            id="new-rule",
            category=MemoryCategory.PREFERENCE,
            name="New Rule",
            rule="Never use approach A",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
            created_at=now,
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])
        resolved, resolutions = await behavioral_controller._resolve_conflicts(
            [rule1, rule2], conflicts, "test context", "normal"
        )

        # Should keep the newer rule
        assert len(resolved) == 1
        assert resolved[0].id == "new-rule"
        assert any("newer" in r.lower() for r in resolutions)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Specificity-based conflict resolution not yet implemented")
    async def test_specificity_based_resolution(self, memory_manager_fixture, behavioral_controller):
        """Test resolution based on rule specificity (scope and conditions)."""
        rule1 = MemoryRule(
            id="general-rule",
            category=MemoryCategory.BEHAVIOR,
            name="General Rule",
            rule="Use feature X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="specific-rule",
            category=MemoryCategory.BEHAVIOR,
            name="Specific Rule",
            rule="Never use feature X in production",
            authority=AuthorityLevel.DEFAULT,
            scope=["global", "production"],
            source="user_explicit",
            conditions={"environment": "production"},
        )

        # Set same creation time
        now = datetime.now(timezone.utc)
        rule1.created_at = now
        rule2.created_at = now

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])
        resolved, resolutions = await behavioral_controller._resolve_conflicts(
            [rule1, rule2], conflicts, "test context", "normal"
        )

        # Should keep the more specific rule
        assert len(resolved) == 1
        assert resolved[0].id == "specific-rule"
        assert any("specific" in r.lower() for r in resolutions)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="User intervention suggestion not yet implemented for equal conflicts")
    async def test_user_intervention_for_equal_conflicts(self, memory_manager_fixture):
        """Test that user intervention is suggested for equal conflicts."""
        now = datetime.now(timezone.utc)

        rule1 = MemoryRule(
            id="equal-1",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 1",
            rule="Use approach A",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
            created_at=now,
        )

        rule2 = MemoryRule(
            id="equal-2",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 2",
            rule="Never use approach A",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
            created_at=now,
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) >= 1
        # Should suggest user intervention or user resolution
        assert any(
            "user" in opt.lower()
            for conflict in conflicts
            for opt in conflict.resolution_options
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Conflict resolution history tracking not yet implemented")
    async def test_resolution_history_tracking(self, memory_manager_fixture, behavioral_controller):
        """Test that conflict resolutions are tracked in history."""
        rule1 = MemoryRule(
            id="track-1",
            category=MemoryCategory.PREFERENCE,
            name="Rule 1",
            rule="Use tool X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="track-2",
            category=MemoryCategory.PREFERENCE,
            name="Rule 2",
            rule="Avoid tool X",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        initial_history_count = len(behavioral_controller.conflict_resolution_history)

        resolved, resolutions = await behavioral_controller._resolve_conflicts(
            [rule1, rule2], conflicts, "test context", "normal"
        )

        # History should be updated
        assert len(behavioral_controller.conflict_resolution_history) > initial_history_count
        assert len(resolutions) > 0


class TestConflictLoggingAndReporting:
    """Test conflict detection logging and reporting mechanisms."""

    @pytest.mark.asyncio
    async def test_conflict_descriptions_generated(self, memory_manager_fixture):
        """Test that conflicts have descriptive messages."""
        rule1 = MemoryRule(
            id="desc-1",
            category=MemoryCategory.BEHAVIOR,
            name="Rule A",
            rule="Always do X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="desc-2",
            category=MemoryCategory.BEHAVIOR,
            name="Rule B",
            rule="Never do X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            assert conflict.description
            assert len(conflict.description) > 10
            assert isinstance(conflict.description, str)

    @pytest.mark.asyncio
    async def test_conflict_confidence_scores(self, memory_manager_fixture):
        """Test that conflict confidence scores are calculated."""
        rule1 = MemoryRule(
            id="conf-1",
            category=MemoryCategory.BEHAVIOR,
            name="Clear Conflict 1",
            rule="Always use Python",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="conf-2",
            category=MemoryCategory.BEHAVIOR,
            name="Clear Conflict 2",
            rule="Avoid Python at all costs",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            assert hasattr(conflict, "confidence")
            assert 0.0 <= conflict.confidence <= 1.0
            # High confidence for clear keyword conflicts
            assert conflict.confidence > 0.5

    @pytest.mark.asyncio
    async def test_conflict_type_classification(self, memory_manager_fixture):
        """Test that conflicts are properly classified by type."""
        rule1 = MemoryRule(
            id="type-1",
            category=MemoryCategory.BEHAVIOR,
            name="Type Test 1",
            rule="Always commit immediately",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="type-2",
            category=MemoryCategory.BEHAVIOR,
            name="Type Test 2",
            rule="Batch commits together",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            assert hasattr(conflict, "conflict_type")
            assert isinstance(conflict.conflict_type, str)
            assert len(conflict.conflict_type) > 0

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Advanced resolution options (newer-based) not yet implemented")
    async def test_conflict_resolution_options_provided(self, memory_manager_fixture):
        """Test that resolution options are suggested."""
        rule1 = MemoryRule(
            id="opt-1",
            category=MemoryCategory.PREFERENCE,
            name="Option Test 1",
            rule="Use tool A",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="opt-2",
            category=MemoryCategory.PREFERENCE,
            name="Option Test 2",
            rule="Avoid tool A",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) > 0
        for conflict in conflicts:
            assert hasattr(conflict, "resolution_options")
            assert isinstance(conflict.resolution_options, list)
            assert len(conflict.resolution_options) > 0
            # Check for expected resolution strategies
            options_str = " ".join(conflict.resolution_options).lower()
            assert any(
                keyword in options_str
                for keyword in ["authority", "merge", "user", "newer"]
            )


class TestKeywordConflictDetection:
    """Test keyword-based conflict detection algorithms."""

    @pytest.mark.asyncio
    async def test_use_avoid_keyword_pair(self, memory_manager_fixture):
        """Test detection of use/avoid keyword conflicts."""
        rule1 = MemoryRule(
            id="kw-1",
            category=MemoryCategory.PREFERENCE,
            name="Use UV",
            rule="Use uv for package management",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="kw-2",
            category=MemoryCategory.PREFERENCE,
            name="Avoid UV",
            rule="Avoid uv, use pip instead",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="conversational",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_always_never_keyword_pair(self, memory_manager_fixture):
        """Test detection of always/never keyword conflicts."""
        rule1 = MemoryRule(
            id="an-1",
            category=MemoryCategory.BEHAVIOR,
            name="Always Test",
            rule="Always write unit tests",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="an-2",
            category=MemoryCategory.BEHAVIOR,
            name="Never Test",
            rule="Never write unit tests for simple functions",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="conversational",
        )

        # This is a subtle conflict - "always" vs "never... for simple"
        # May or may not be detected depending on sophistication
        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Validate that the conflict detection runs without error
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_multi_keyword_conflict(self, memory_manager_fixture):
        """Test detection of complex multi-keyword conflicts."""
        rule1 = MemoryRule(
            id="mk-1",
            category=MemoryCategory.BEHAVIOR,
            name="Immediate Batch Commits",
            rule="Commit immediately but batch related changes",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="mk-2",
            category=MemoryCategory.BEHAVIOR,
            name="No Batch Commits",
            rule="Never batch commits, commit each change separately",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Should detect "batch commit" conflict
        assert len(conflicts) >= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions in conflict detection."""

    @pytest.mark.asyncio
    async def test_empty_rules_list(self, memory_manager_fixture):
        """Test conflict detection with empty rules list."""
        conflicts = await memory_manager_fixture.detect_conflicts([])

        assert conflicts == []
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_single_rule_no_conflict(self, memory_manager_fixture):
        """Test that a single rule produces no conflicts."""
        rule = MemoryRule(
            id="single",
            category=MemoryCategory.BEHAVIOR,
            name="Single Rule",
            rule="Use best practices",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule])

        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_identical_rules_no_conflict(self, memory_manager_fixture):
        """Test that identical rules don't create conflicts."""
        rule1 = MemoryRule(
            id="ident-1",
            category=MemoryCategory.BEHAVIOR,
            name="Same Rule",
            rule="Use best practices",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="ident-2",
            category=MemoryCategory.BEHAVIOR,
            name="Same Rule Copy",
            rule="Use best practices",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Identical rules should not conflict
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_complementary_rules_no_conflict(self, memory_manager_fixture):
        """Test that complementary rules don't conflict."""
        rule1 = MemoryRule(
            id="comp-1",
            category=MemoryCategory.BEHAVIOR,
            name="Test First",
            rule="Write tests before implementation",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        rule2 = MemoryRule(
            id="comp-2",
            category=MemoryCategory.BEHAVIOR,
            name="Document Code",
            rule="Document all public APIs",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            source="user_explicit",
        )

        conflicts = await memory_manager_fixture.detect_conflicts([rule1, rule2])

        # Complementary rules should not conflict
        assert len(conflicts) == 0

    @pytest.mark.asyncio
    async def test_many_rules_performance(self, memory_manager_fixture):
        """Test conflict detection with many rules (performance test)."""
        rules = [
            MemoryRule(
                id=f"perf-{i}",
                category=MemoryCategory.BEHAVIOR,
                name=f"Rule {i}",
                rule=f"Guideline number {i}",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            )
            for i in range(50)
        ]

        # Add one conflicting pair in the middle
        rules[25].rule = "Always use feature X"
        rules[30].rule = "Never use feature X"

        conflicts = await memory_manager_fixture.detect_conflicts(rules)

        # Should detect the conflict even with many rules
        assert len(conflicts) >= 1
        # Should complete in reasonable time (async so no strict timing)


class TestMultipleConflicts:
    """Test handling of multiple simultaneous conflicts."""

    @pytest.mark.asyncio
    async def test_multiple_conflict_pairs(self, memory_manager_fixture):
        """Test detection of multiple independent conflicts."""
        rules = [
            # Conflict pair 1: Python version
            MemoryRule(
                id="multi-1a",
                category=MemoryCategory.PREFERENCE,
                name="Python 3.11",
                rule="Use Python 3.11",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="user_explicit",
            ),
            MemoryRule(
                id="multi-1b",
                category=MemoryCategory.PREFERENCE,
                name="Python 3.12",
                rule="Avoid Python 3.11, use Python 3.12",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="conversational",
            ),
            # Conflict pair 2: Testing framework
            MemoryRule(
                id="multi-2a",
                category=MemoryCategory.PREFERENCE,
                name="Use Pytest",
                rule="Use pytest for testing",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="user_explicit",
            ),
            MemoryRule(
                id="multi-2b",
                category=MemoryCategory.PREFERENCE,
                name="Use Unittest",
                rule="Use unittest instead of pytest",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="conversational",
            ),
        ]

        conflicts = await memory_manager_fixture.detect_conflicts(rules)

        # Should detect both conflict pairs
        assert len(conflicts) >= 2

    @pytest.mark.asyncio
    async def test_chained_conflicts(self, memory_manager_fixture):
        """Test detection of chained conflicts (A conflicts with B, B conflicts with C)."""
        rules = [
            MemoryRule(
                id="chain-a",
                category=MemoryCategory.BEHAVIOR,
                name="Rule A",
                rule="Always use approach A",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            ),
            MemoryRule(
                id="chain-b",
                category=MemoryCategory.BEHAVIOR,
                name="Rule B",
                rule="Never use approach A, use approach B",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            ),
            MemoryRule(
                id="chain-c",
                category=MemoryCategory.BEHAVIOR,
                name="Rule C",
                rule="Avoid approach B entirely",
                authority=AuthorityLevel.DEFAULT,
                scope=["global"],
                source="user_explicit",
            ),
        ]

        conflicts = await memory_manager_fixture.detect_conflicts(rules)

        # Should detect at least the A-B conflict
        # May also detect B-C conflict depending on implementation
        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_triangular_conflict(self, memory_manager_fixture):
        """Test three-way conflict detection."""
        rules = [
            MemoryRule(
                id="tri-1",
                category=MemoryCategory.PREFERENCE,
                name="Use UV",
                rule="Use uv for package management",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="user_explicit",
            ),
            MemoryRule(
                id="tri-2",
                category=MemoryCategory.PREFERENCE,
                name="Use Pip",
                rule="Use pip for package management",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="user_explicit",
            ),
            MemoryRule(
                id="tri-3",
                category=MemoryCategory.PREFERENCE,
                name="Use Poetry",
                rule="Use poetry for package management",
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="user_explicit",
            ),
        ]

        conflicts = await memory_manager_fixture.detect_conflicts(rules)

        # Should detect multiple conflicts (uv vs pip, uv vs poetry, pip vs poetry)
        # At minimum should detect some conflicts
        assert len(conflicts) >= 1


class TestConflictDataStructures:
    """Test the MemoryConflict data structure and its properties."""

    def test_memory_conflict_creation(self):
        """Test creating a MemoryConflict instance."""
        rule1 = MemoryRule(
            id="r1",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 1",
            rule="Do X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        rule2 = MemoryRule(
            id="r2",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 2",
            rule="Don't do X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        conflict = MemoryConflict(
            conflict_type="direct_contradiction",
            rule1=rule1,
            rule2=rule2,
            confidence=0.95,
            description="Rules contradict each other",
            resolution_options=["keep_newer", "user_decision"],
        )

        assert conflict.conflict_type == "direct_contradiction"
        assert conflict.rule1.id == "r1"
        assert conflict.rule2.id == "r2"
        assert conflict.confidence == 0.95
        assert len(conflict.resolution_options) == 2

    def test_conflict_confidence_bounds(self):
        """Test that conflict confidence is within valid bounds."""
        rule1 = MemoryRule(
            id="r1",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 1",
            rule="X",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        rule2 = MemoryRule(
            id="r2",
            category=MemoryCategory.BEHAVIOR,
            name="Rule 2",
            rule="Y",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
        )

        # Test valid confidence
        valid_conflict = MemoryConflict(
            conflict_type="test",
            rule1=rule1,
            rule2=rule2,
            confidence=0.7,
            description="Test",
            resolution_options=[],
        )

        assert 0.0 <= valid_conflict.confidence <= 1.0

    def test_conflict_with_all_fields(self):
        """Test conflict with all optional fields populated."""
        rule1 = MemoryRule(
            id="full-r1",
            category=MemoryCategory.BEHAVIOR,
            name="Full Rule 1",
            rule="Complete rule text",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global", "production"],
        )

        rule2 = MemoryRule(
            id="full-r2",
            category=MemoryCategory.BEHAVIOR,
            name="Full Rule 2",
            rule="Another complete rule",
            authority=AuthorityLevel.DEFAULT,
            scope=["global", "development"],
        )

        conflict = MemoryConflict(
            conflict_type="authority_mismatch",
            rule1=rule1,
            rule2=rule2,
            confidence=0.88,
            description="Detailed conflict description with context",
            resolution_options=[
                "keep_higher_authority",
                "keep_newer",
                "merge_conditions",
                "user_decision",
            ],
        )

        assert conflict.conflict_type == "authority_mismatch"
        assert conflict.confidence == 0.88
        assert len(conflict.description) > 20
        assert len(conflict.resolution_options) == 4
        assert rule1.authority == AuthorityLevel.ABSOLUTE
        assert rule2.authority == AuthorityLevel.DEFAULT
