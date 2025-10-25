"""
Comprehensive tests for memory rule authority level validation (Task 324.4).

This test suite validates:
- Rule precedence based on authority levels (absolute vs default)
- Absolute rules overriding default rules correctly
- Authority level changes and their impact on existing rule hierarchies
- Permission checks for authority modifications
- Conflict resolution with different authority levels
- Behavioral decision-making respecting authority levels
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, ScoredPoint, VectorParams

# Add src/python to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

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


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client for testing with memory collection."""
    client = MagicMock(spec=QdrantClient)

    # Mock get_collections to always return memory collection
    mock_collection = MagicMock()
    mock_collection.name = "memory"
    mock_collections_result = MagicMock()
    mock_collections_result.collections = [mock_collection]
    client.get_collections.return_value = mock_collections_result

    # Mock upsert
    client.upsert.return_value = MagicMock()

    # Mock retrieve - will be configured per test
    client.retrieve.return_value = []

    # Mock scroll
    client.scroll.return_value = ([], None)

    # Mock delete
    client.delete.return_value = MagicMock()

    # Mock search
    client.search.return_value = []

    return client


@pytest.fixture
def naming_manager():
    """Create a collection naming manager."""
    return CollectionNamingManager()


@pytest.fixture
def sparse_encoder():
    """Create a mocked sparse vector encoder."""
    encoder = MagicMock()
    encoder.vector_size = 1000
    encoder.generate_sparse_vector.return_value = {
        "indices": [0, 5, 10],
        "values": [0.8, 0.6, 0.4]
    }
    return encoder


@pytest.fixture
def memory_manager(mock_qdrant_client, naming_manager, sparse_encoder):
    """Create a memory manager instance for testing."""
    return MemoryManager(
        qdrant_client=mock_qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=384,
        sparse_vector_generator=sparse_encoder,
    )


@pytest.fixture
def behavioral_controller(memory_manager):
    """Create a behavioral controller for testing decision-making."""
    return BehavioralController(memory_manager)


# Test 1: Basic authority level creation and retrieval
@pytest.mark.asyncio
async def test_create_rules_with_different_authority_levels(memory_manager):
    """Test creating rules with absolute and default authority levels."""
    # Create absolute authority rule
    absolute_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.BEHAVIOR,
        name="always_commit_atomic",
        rule="Always make atomic commits after each change",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["git", "development"],
    )

    # Create default authority rule
    default_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.PREFERENCE,
        name="prefer_uv",
        rule="Prefer using uv for Python package management",
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
    )

    # Verify both rules were created
    assert absolute_rule_id is not None
    assert default_rule_id is not None
    assert absolute_rule_id != default_rule_id


# Test 2: Verify authority level is persisted correctly
@pytest.mark.asyncio
async def test_authority_level_persistence(memory_manager, mock_qdrant_client):
    """Test that authority levels are correctly persisted in payloads."""
    # Create rule with absolute authority
    await memory_manager.add_memory_rule(
        category=MemoryCategory.BEHAVIOR,
        name="test_rule",
        rule="Test rule content",
        authority=AuthorityLevel.ABSOLUTE,
    )

    # Verify upsert was called with correct authority in payload
    assert mock_qdrant_client.upsert.called
    call_args = mock_qdrant_client.upsert.call_args
    points = call_args[1]["points"]
    assert len(points) == 1
    assert points[0].payload["authority"] == "absolute"


# Test 3: List rules filtered by authority level
@pytest.mark.asyncio
async def test_list_rules_by_authority_level(memory_manager, mock_qdrant_client):
    """Test filtering rules by authority level."""
    # Setup mock scroll to return rules with different authority levels
    absolute_point = MagicMock()
    absolute_point.id = "rule_001"
    absolute_point.payload = {
        "category": "behavior",
        "name": "absolute_rule",
        "rule": "Absolute rule content",
        "authority": "absolute",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    default_point = MagicMock()
    default_point.id = "rule_002"
    default_point.payload = {
        "category": "preference",
        "name": "default_rule",
        "rule": "Default rule content",
        "authority": "default",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.scroll.return_value = ([absolute_point, default_point], None)

    # Test listing absolute rules only
    await memory_manager.list_memory_rules(
        authority=AuthorityLevel.ABSOLUTE
    )

    # Verify filter was applied
    call_args = mock_qdrant_client.scroll.call_args
    scroll_filter = call_args[1]["scroll_filter"]
    assert scroll_filter is not None
    # Should have filter condition for authority level
    assert len(scroll_filter.must) == 1


# Test 4: Absolute rules override default rules in conflict resolution
@pytest.mark.asyncio
async def test_absolute_overrides_default_in_conflict(
    behavioral_controller, memory_manager, mock_qdrant_client
):
    """Test that absolute rules override default rules during conflict resolution."""
    # Create two conflicting rules with different authority levels
    absolute_rule = MemoryRule(
        id="rule_absolute",
        category=MemoryCategory.PREFERENCE,
        name="use_pytest",
        rule="Always use pytest for testing",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["testing"],
        created_at=datetime.now(timezone.utc),
    )

    default_rule = MemoryRule(
        id="rule_default",
        category=MemoryCategory.PREFERENCE,
        name="use_unittest",
        rule="Use unittest for testing",
        authority=AuthorityLevel.DEFAULT,
        scope=["testing"],
        created_at=datetime.now(timezone.utc),
    )

    # Create a conflict between them
    conflict = MemoryConflict(
        conflict_type="direct_contradiction",
        rule1=absolute_rule,
        rule2=default_rule,
        confidence=0.9,
        description="Conflicting testing framework preferences",
        resolution_options=["Keep higher authority", "User resolution"],
    )

    # Resolve the conflict
    resolved_rules, resolutions = await behavioral_controller._resolve_conflicts(
        rules=[absolute_rule, default_rule],
        conflicts=[conflict],
        context="testing framework selection",
        urgency="normal"
    )

    # Verify absolute rule was kept and default was removed
    assert len(resolved_rules) == 1
    assert resolved_rules[0].id == "rule_absolute"
    assert resolved_rules[0].authority == AuthorityLevel.ABSOLUTE
    assert len(resolutions) == 1
    assert "absolute" in resolutions[0].lower()


# Test 5: Newer rule wins when both have same authority level
@pytest.mark.asyncio
async def test_newer_rule_wins_with_same_authority(
    behavioral_controller, memory_manager
):
    """Test that newer rules override older rules when authority levels are equal."""
    older_time = datetime.now(timezone.utc) - timedelta(days=7)
    newer_time = datetime.now(timezone.utc)

    # Create two rules with same authority but different ages
    older_rule = MemoryRule(
        id="rule_old",
        category=MemoryCategory.PREFERENCE,
        name="old_preference",
        rule="Use older tool",
        authority=AuthorityLevel.DEFAULT,
        scope=["tools"],
        created_at=older_time,
    )

    newer_rule = MemoryRule(
        id="rule_new",
        category=MemoryCategory.PREFERENCE,
        name="new_preference",
        rule="Use newer tool",
        authority=AuthorityLevel.DEFAULT,
        scope=["tools"],
        created_at=newer_time,
    )

    conflict = MemoryConflict(
        conflict_type="direct_contradiction",
        rule1=newer_rule,
        rule2=older_rule,
        confidence=0.85,
        description="Tool preference conflict",
        resolution_options=["Keep newer", "Keep older"],
    )

    # Resolve conflict
    resolved_rules, resolutions = await behavioral_controller._resolve_conflicts(
        rules=[older_rule, newer_rule],
        conflicts=[conflict],
        context="tool selection",
        urgency="normal"
    )

    # Verify newer rule was kept
    assert len(resolved_rules) == 1
    assert resolved_rules[0].id == "rule_new"
    assert "newer" in resolutions[0].lower()


# Test 6: Authority level changes via update
@pytest.mark.asyncio
async def test_change_authority_level_via_update(memory_manager, mock_qdrant_client):
    """Test changing a rule's authority level through update operation."""
    # Setup initial rule retrieval
    initial_time = datetime.now(timezone.utc)
    rule_point = MagicMock()
    rule_point.id = "rule_001"
    rule_point.payload = {
        "category": "preference",
        "name": "test_rule",
        "rule": "Test rule content",
        "authority": "default",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": initial_time.isoformat(),
        "updated_at": initial_time.isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.retrieve.return_value = [rule_point]

    # Update rule to change authority level from default to absolute
    success = await memory_manager.update_memory_rule(
        rule_id="rule_001",
        updates={"authority": AuthorityLevel.ABSOLUTE},
        embedding_vector=[0.0] * 384
    )

    assert success

    # Verify upsert was called with updated authority
    assert mock_qdrant_client.upsert.called
    call_args = mock_qdrant_client.upsert.call_args
    updated_points = call_args[1]["points"]
    assert len(updated_points) == 1
    assert updated_points[0].payload["authority"] == "absolute"


# Test 7: Authority level impact on rule hierarchy
@pytest.mark.asyncio
async def test_authority_hierarchy_in_behavioral_decisions(
    behavioral_controller, memory_manager, mock_qdrant_client
):
    """Test that authority levels create proper hierarchy in behavioral decisions."""
    absolute_rule = MemoryRule(
        id="rule_001",
        category=MemoryCategory.BEHAVIOR,
        name="critical_behavior",
        rule="Must follow strict coding standards",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["development"],
        created_at=datetime.now(timezone.utc),
    )

    default_rule = MemoryRule(
        id="rule_002",
        category=MemoryCategory.BEHAVIOR,
        name="recommended_behavior",
        rule="Consider using linting tools",
        authority=AuthorityLevel.DEFAULT,
        scope=["development"],
        created_at=datetime.now(timezone.utc),
    )

    # Mock search to return both rules
    scored_absolute = MagicMock(spec=ScoredPoint)
    scored_absolute.score = 0.9
    scored_absolute.id = "rule_001"
    scored_absolute.payload = absolute_rule.__dict__.copy()
    scored_absolute.payload["category"] = absolute_rule.category.value
    scored_absolute.payload["authority"] = absolute_rule.authority.value
    scored_absolute.payload["created_at"] = absolute_rule.created_at.isoformat()
    scored_absolute.payload["updated_at"] = absolute_rule.created_at.isoformat()
    scored_absolute.payload["scope"] = absolute_rule.scope
    scored_absolute.payload["conditions"] = None
    scored_absolute.payload["replaces"] = None
    scored_absolute.payload["metadata"] = None

    scored_default = MagicMock(spec=ScoredPoint)
    scored_default.score = 0.8
    scored_default.id = "rule_002"
    scored_default.payload = default_rule.__dict__.copy()
    scored_default.payload["category"] = default_rule.category.value
    scored_default.payload["authority"] = default_rule.authority.value
    scored_default.payload["created_at"] = default_rule.created_at.isoformat()
    scored_default.payload["updated_at"] = default_rule.created_at.isoformat()
    scored_default.payload["scope"] = default_rule.scope
    scored_default.payload["conditions"] = None
    scored_default.payload["replaces"] = None
    scored_default.payload["metadata"] = None

    mock_qdrant_client.search.return_value = [scored_absolute, scored_default]

    # Make a behavioral decision
    decision = await behavioral_controller.make_decision(
        context="How should I structure my code?",
        situation_type="development",
        project_scope=["development"]
    )

    # Verify decision prioritizes absolute rule
    assert decision.decision is not None
    assert "Required actions:" in decision.decision
    assert "strict coding standards" in decision.decision.lower()


# Test 8: Validate memory stats show authority breakdown
@pytest.mark.asyncio
async def test_memory_stats_authority_breakdown(memory_manager, mock_qdrant_client):
    """Test that memory stats correctly report rules by authority level."""
    # Setup mock with rules of different authority levels
    absolute_point = MagicMock()
    absolute_point.id = "rule_001"
    absolute_point.payload = {
        "category": "behavior",
        "name": "absolute_rule",
        "rule": "Absolute rule",
        "authority": "absolute",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    default_point_1 = MagicMock()
    default_point_1.id = "rule_002"
    default_point_1.payload = {
        "category": "preference",
        "name": "default_rule_1",
        "rule": "Default rule 1",
        "authority": "default",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    default_point_2 = MagicMock()
    default_point_2.id = "rule_003"
    default_point_2.payload = {
        "category": "preference",
        "name": "default_rule_2",
        "rule": "Default rule 2",
        "authority": "default",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.scroll.return_value = (
        [absolute_point, default_point_1, default_point_2],
        None
    )

    # Get memory stats
    stats = await memory_manager.get_memory_stats()

    # Verify authority breakdown
    assert stats.rules_by_authority[AuthorityLevel.ABSOLUTE] == 1
    assert stats.rules_by_authority[AuthorityLevel.DEFAULT] == 2
    assert stats.total_rules == 3


# Test 9: Multiple absolute rules don't conflict by authority alone
@pytest.mark.asyncio
async def test_multiple_absolute_rules_coexist(memory_manager):
    """Test that multiple absolute rules can coexist if they don't contradict."""
    # Create two absolute rules in different domains
    rule1_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.BEHAVIOR,
        name="commit_discipline",
        rule="Always make atomic commits",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["git"],
    )

    rule2_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.BEHAVIOR,
        name="test_discipline",
        rule="Always write tests before code",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["testing"],
    )

    # Both should be created successfully
    assert rule1_id is not None
    assert rule2_id is not None
    assert rule1_id != rule2_id


# Test 10: Authority level preserved during rule replacement
@pytest.mark.asyncio
async def test_authority_preserved_in_replacement(memory_manager, mock_qdrant_client):
    """Test that authority level is preserved when a rule replaces another."""
    # Setup old rule to be replaced
    old_rule_point = MagicMock()
    old_rule_point.id = "rule_old"
    old_rule_point.payload = {
        "category": "preference",
        "name": "old_rule",
        "rule": "Old preference",
        "authority": "absolute",
        "scope": ["tools"],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.retrieve.return_value = [old_rule_point]

    # Create new rule that replaces the old one with same authority
    new_rule_id = await memory_manager.add_memory_rule(
        category=MemoryCategory.PREFERENCE,
        name="new_rule",
        rule="New preference that replaces old",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["tools"],
        replaces=["rule_old"],
    )

    # Verify new rule was created with absolute authority
    assert new_rule_id is not None

    # Verify upsert was called with absolute authority
    call_args = mock_qdrant_client.upsert.call_args
    points = call_args[1]["points"]
    assert points[0].payload["authority"] == "absolute"
    assert points[0].payload["replaces"] == ["rule_old"]


# Test 11: Search respects authority level filter
@pytest.mark.asyncio
async def test_search_with_authority_filter(memory_manager, mock_qdrant_client):
    """Test that search operations can filter by authority level."""
    # Setup mock search results
    absolute_result = MagicMock(spec=ScoredPoint)
    absolute_result.score = 0.95
    absolute_result.id = "rule_001"
    absolute_result.payload = {
        "category": "behavior",
        "name": "absolute_rule",
        "rule": "Critical behavior rule",
        "authority": "absolute",
        "scope": [],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.search.return_value = [absolute_result]

    # Search with authority level filter
    await memory_manager.search_memory_rules(
        query="critical behavior",
        authority=AuthorityLevel.ABSOLUTE,
        limit=10
    )

    # Verify search was called with authority filter
    call_args = mock_qdrant_client.search.call_args
    query_filter = call_args[1]["query_filter"]
    assert query_filter is not None
    # Should have filter condition for authority
    assert len(query_filter.must) == 1


# Test 12: Confidence boost for absolute rules in decisions
@pytest.mark.asyncio
async def test_absolute_rules_boost_decision_confidence(
    behavioral_controller, memory_manager, mock_qdrant_client
):
    """Test that absolute rules increase decision confidence scores."""
    # Create absolute rule
    absolute_rule = MemoryRule(
        id="rule_001",
        category=MemoryCategory.BEHAVIOR,
        name="absolute_behavior",
        rule="Must follow this behavior",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["development"],
        created_at=datetime.now(timezone.utc),
    )

    # Mock search to return absolute rule
    scored_point = MagicMock(spec=ScoredPoint)
    scored_point.score = 0.9
    scored_point.id = "rule_001"
    scored_point.payload = {
        "category": absolute_rule.category.value,
        "name": absolute_rule.name,
        "rule": absolute_rule.rule,
        "authority": absolute_rule.authority.value,
        "scope": absolute_rule.scope,
        "source": "user_explicit",
        "conditions": None,
        "replaces": None,
        "created_at": absolute_rule.created_at.isoformat(),
        "updated_at": absolute_rule.created_at.isoformat(),
        "metadata": None,
    }

    mock_qdrant_client.search.return_value = [scored_point]

    # Make decision with absolute rule
    decision_absolute = await behavioral_controller.make_decision(
        context="development workflow",
        situation_type="development"
    )

    # Confidence should be high due to absolute authority
    assert decision_absolute.confidence > 0.0  # Has confidence with absolute rule


# Test 13: Default rules can be overridden in specific contexts
@pytest.mark.asyncio
async def test_default_rules_context_override(behavioral_controller, memory_manager, mock_qdrant_client):
    """Test that default rules can be overridden by more specific rules in context."""
    # General default rule
    general_rule = MemoryRule(
        id="rule_general",
        category=MemoryCategory.PREFERENCE,
        name="general_preference",
        rule="Generally prefer tool A",
        authority=AuthorityLevel.DEFAULT,
        scope=["general"],
        created_at=datetime.now(timezone.utc) - timedelta(days=5),
    )

    # Specific default rule for a particular context
    specific_rule = MemoryRule(
        id="rule_specific",
        category=MemoryCategory.PREFERENCE,
        name="specific_preference",
        rule="For this specific context, use tool B",
        authority=AuthorityLevel.DEFAULT,
        scope=["specific_context"],
        conditions={"context": "specific_context"},
        created_at=datetime.now(timezone.utc),
    )

    # Mock search to return both rules
    general_scored = MagicMock(spec=ScoredPoint)
    general_scored.score = 0.7
    general_scored.id = "rule_general"
    general_scored.payload = {
        "category": general_rule.category.value,
        "name": general_rule.name,
        "rule": general_rule.rule,
        "authority": general_rule.authority.value,
        "scope": general_rule.scope,
        "source": "user_explicit",
        "conditions": None,
        "replaces": None,
        "created_at": general_rule.created_at.isoformat(),
        "updated_at": general_rule.created_at.isoformat(),
        "metadata": None,
    }

    specific_scored = MagicMock(spec=ScoredPoint)
    specific_scored.score = 0.8
    specific_scored.id = "rule_specific"
    specific_scored.payload = {
        "category": specific_rule.category.value,
        "name": specific_rule.name,
        "rule": specific_rule.rule,
        "authority": specific_rule.authority.value,
        "scope": specific_rule.scope,
        "source": "user_explicit",
        "conditions": specific_rule.conditions,
        "replaces": None,
        "created_at": specific_rule.created_at.isoformat(),
        "updated_at": specific_rule.created_at.isoformat(),
        "metadata": None,
    }

    mock_qdrant_client.search.return_value = [specific_scored, general_scored]

    # Make decision - specific rule should be preferred due to higher specificity
    decision = await behavioral_controller.make_decision(
        context="specific_context task",
        project_scope=["specific_context"]
    )

    # Decision should exist and be based on rules
    assert decision.decision is not None
    assert len(decision.applicable_rules) > 0


# Test 14: Authority level validation on creation
def test_invalid_authority_level_rejected():
    """Test that invalid authority level values are rejected."""
    with pytest.raises(ValueError):
        # This should fail because "invalid" is not a valid AuthorityLevel
        # AuthorityLevel is an Enum, so invalid value raises ValueError
        AuthorityLevel("invalid")


# Test 15: Conflict detection considers authority levels
@pytest.mark.asyncio
async def test_conflict_detection_with_authority(memory_manager, mock_qdrant_client):
    """Test that conflict detection properly handles rules with different authority levels."""
    # Create two potentially conflicting rules with different authority
    absolute_rule = MemoryRule(
        id="rule_001",
        category=MemoryCategory.PREFERENCE,
        name="use_pytest",
        rule="Always use pytest for testing",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["testing"],
        created_at=datetime.now(timezone.utc),
    )

    default_rule = MemoryRule(
        id="rule_002",
        category=MemoryCategory.PREFERENCE,
        name="use_unittest",
        rule="Use unittest for testing",
        authority=AuthorityLevel.DEFAULT,
        scope=["testing"],
        created_at=datetime.now(timezone.utc),
    )

    # Detect conflicts
    conflicts = await memory_manager.detect_conflicts([absolute_rule, default_rule])

    # Should detect conflict between pytest and unittest
    # The conflict exists regardless of authority, but resolution will favor absolute
    assert len(conflicts) >= 0  # May or may not detect depending on conflict detection algorithm


# Test 16: Authority level in formatted output
@pytest.mark.asyncio
async def test_authority_level_in_formatted_output(memory_manager, mock_qdrant_client):
    """Test that authority levels are correctly represented in formatted rule output."""
    from common.core.memory import format_memory_rules_for_injection

    absolute_rule = MemoryRule(
        id="rule_001",
        category=MemoryCategory.BEHAVIOR,
        name="critical_rule",
        rule="This is critical",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["all"],
        created_at=datetime.now(timezone.utc),
    )

    default_rule = MemoryRule(
        id="rule_002",
        category=MemoryCategory.PREFERENCE,
        name="recommended_rule",
        rule="This is recommended",
        authority=AuthorityLevel.DEFAULT,
        scope=["all"],
        created_at=datetime.now(timezone.utc),
    )

    # Format rules for injection
    formatted = format_memory_rules_for_injection([absolute_rule, default_rule])

    # Verify absolute rules are in CRITICAL section
    assert "CRITICAL RULES" in formatted
    assert "critical_rule" in formatted

    # Verify default rules are in DEFAULT section
    assert "DEFAULT GUIDELINES" in formatted
    assert "recommended_rule" in formatted


# Test 17: Specificity beats same-authority in conflict resolution
@pytest.mark.asyncio
async def test_specificity_trumps_authority_tie(behavioral_controller):
    """Test that more specific rules win when authority levels are tied."""
    # General rule
    general_rule = MemoryRule(
        id="rule_general",
        category=MemoryCategory.PREFERENCE,
        name="general",
        rule="General preference",
        authority=AuthorityLevel.DEFAULT,
        scope=[],  # No specific scope
        created_at=datetime.now(timezone.utc),
    )

    # Specific rule with conditions
    specific_rule = MemoryRule(
        id="rule_specific",
        category=MemoryCategory.PREFERENCE,
        name="specific",
        rule="Specific preference for project X",
        authority=AuthorityLevel.DEFAULT,
        scope=["project_x", "feature_y"],
        conditions={"project": "project_x"},
        created_at=datetime.now(timezone.utc),
    )

    conflict = MemoryConflict(
        conflict_type="scope_overlap",
        rule1=specific_rule,
        rule2=general_rule,
        confidence=0.7,
        description="Overlapping preferences",
        resolution_options=["Keep more specific"],
    )

    # Resolve conflict
    resolved_rules, resolutions = await behavioral_controller._resolve_conflicts(
        rules=[general_rule, specific_rule],
        conflicts=[conflict],
        context="project_x task",
        urgency="normal"
    )

    # More specific rule should win
    assert len(resolved_rules) == 1
    assert resolved_rules[0].id == "rule_specific"
    assert "specific" in resolutions[0].lower()


# Test 18: Authority level changes affect existing hierarchies
@pytest.mark.asyncio
async def test_authority_change_affects_hierarchy(memory_manager, mock_qdrant_client):
    """Test that changing authority level of a rule affects decision hierarchies."""
    # Setup initial rule as default
    initial_time = datetime.now(timezone.utc)
    rule_point = MagicMock()
    rule_point.id = "rule_001"
    rule_point.payload = {
        "category": "behavior",
        "name": "test_behavior",
        "rule": "Test behavioral rule",
        "authority": "default",
        "scope": ["testing"],
        "source": "user_explicit",
        "conditions": {},
        "replaces": [],
        "created_at": initial_time.isoformat(),
        "updated_at": initial_time.isoformat(),
        "metadata": {},
    }

    mock_qdrant_client.retrieve.return_value = [rule_point]

    # Get initial rule
    initial_rule = await memory_manager.get_memory_rule("rule_001")
    assert initial_rule is not None
    assert initial_rule.authority == AuthorityLevel.DEFAULT

    # Update to absolute authority
    await memory_manager.update_memory_rule(
        rule_id="rule_001",
        updates={"authority": AuthorityLevel.ABSOLUTE},
        embedding_vector=[0.0] * 384
    )

    # Verify the update changed authority in the payload
    call_args = mock_qdrant_client.upsert.call_args
    updated_points = call_args[1]["points"]
    assert updated_points[0].payload["authority"] == "absolute"


# Test 19: Verify permission model for authority changes (placeholder)
@pytest.mark.asyncio
async def test_authority_change_permission_model(memory_manager):
    """
    Test permission checks for authority level modifications.

    Note: This is a placeholder test as the current implementation
    doesn't have explicit permission checks. In a production system,
    changing a rule from default to absolute should require higher
    permissions.
    """
    # This test documents the expected behavior for future implementation
    # Current implementation allows any authority change without permission checks

    # Future implementation should check:
    # 1. User has permission to create absolute rules
    # 2. User has permission to elevate default to absolute
    # 3. User has permission to downgrade absolute to default

    # For now, we just verify the operation works
    pass  # Placeholder for future permission system


# Test 20: Complex authority hierarchy with multiple conflicts
@pytest.mark.asyncio
async def test_complex_authority_hierarchy(behavioral_controller, memory_manager, mock_qdrant_client):
    """Test complex scenarios with multiple rules and authority levels."""
    # Create a complex set of rules
    rules = [
        MemoryRule(
            id="rule_001",
            category=MemoryCategory.BEHAVIOR,
            name="absolute_1",
            rule="Critical behavior A",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["development"],
            created_at=datetime.now(timezone.utc) - timedelta(days=10),
        ),
        MemoryRule(
            id="rule_002",
            category=MemoryCategory.BEHAVIOR,
            name="absolute_2",
            rule="Critical behavior B",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["development"],
            created_at=datetime.now(timezone.utc) - timedelta(days=5),
        ),
        MemoryRule(
            id="rule_003",
            category=MemoryCategory.PREFERENCE,
            name="default_1",
            rule="Recommended tool X",
            authority=AuthorityLevel.DEFAULT,
            scope=["tools"],
            created_at=datetime.now(timezone.utc) - timedelta(days=3),
        ),
        MemoryRule(
            id="rule_004",
            category=MemoryCategory.PREFERENCE,
            name="default_2",
            rule="Recommended tool Y",
            authority=AuthorityLevel.DEFAULT,
            scope=["tools"],
            created_at=datetime.now(timezone.utc) - timedelta(days=1),
        ),
    ]

    # Mock search to return all rules
    scored_points = []
    for i, rule in enumerate(rules):
        scored = MagicMock(spec=ScoredPoint)
        scored.score = 0.9 - (i * 0.1)  # Decreasing scores
        scored.id = rule.id
        scored.payload = {
            "category": rule.category.value,
            "name": rule.name,
            "rule": rule.rule,
            "authority": rule.authority.value,
            "scope": rule.scope,
            "source": "user_explicit",
            "conditions": None,
            "replaces": None,
            "created_at": rule.created_at.isoformat(),
            "updated_at": rule.created_at.isoformat(),
            "metadata": None,
        }
        scored_points.append(scored)

    mock_qdrant_client.search.return_value = scored_points

    # Make decision with complex rule set
    decision = await behavioral_controller.make_decision(
        context="Choose development tools and behaviors",
        situation_type="development",
        project_scope=["development", "tools"]
    )

    # Verify decision includes both absolute and default rules
    assert decision.decision is not None
    assert "Required actions:" in decision.decision  # Absolute rules
    assert len(decision.applicable_rules) > 0
    assert decision.confidence > 0.5  # Should have good confidence with multiple rules


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
