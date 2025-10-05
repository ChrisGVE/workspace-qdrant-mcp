"""
Tests for interactive rule trimmer.

Tests the InteractiveTrimmer class for managing rule selection when
budgets are exceeded. Focuses on core logic separate from CLI presentation.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from src.python.common.core.context_injection.interactive_trimmer import (
    BudgetVisualization,
    InteractiveTrimmer,
    RuleDisplay,
    TrimDecision,
    TrimDecisionType,
)
from src.python.common.core.context_injection.rule_prioritizer import (
    PrioritizationStrategy,
    RulePrioritizer,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


@pytest.fixture
def sample_rules() -> List[MemoryRule]:
    """Create sample rules for testing."""
    return [
        MemoryRule(
            id="rule-1",
            category=MemoryCategory.BEHAVIOR,
            name="Code Style",
            rule="Always use black formatting for Python code. Never deviate from PEP 8 standards.",
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            metadata={"priority": 70},
        ),
        MemoryRule(
            id="rule-2",
            category=MemoryCategory.BEHAVIOR,
            name="Security",
            rule="Always validate user input. Use parameterized queries for database access. Never trust external data.",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            metadata={"priority": 95},
        ),
        MemoryRule(
            id="rule-3",
            category=MemoryCategory.PREFERENCE,
            name="Testing",
            rule="Write unit tests for all new functions. Aim for 90% test coverage minimum.",
            authority=AuthorityLevel.DEFAULT,
            scope=["testing"],
            metadata={"priority": 85},
        ),
        MemoryRule(
            id="rule-4",
            category=MemoryCategory.BEHAVIOR,
            name="Documentation",
            rule="Document all public APIs with docstrings. Include examples for complex functions.",
            authority=AuthorityLevel.DEFAULT,
            scope=["documentation"],
            metadata={"priority": 60},
        ),
        MemoryRule(
            id="rule-5",
            category=MemoryCategory.PREFERENCE,
            name="Error Handling",
            rule="Always use specific exception types. Log errors with context. Never use bare except clauses.",
            authority=AuthorityLevel.DEFAULT,
            scope=["error_handling"],
            metadata={"priority": 80},
        ),
    ]


@pytest.fixture
def prioritizer() -> RulePrioritizer:
    """Create RulePrioritizer for testing."""
    return RulePrioritizer(
        strategy=PrioritizationStrategy.HYBRID,
        importance_weight=0.5,
        frequency_weight=0.3,
        recency_weight=0.2,
        use_accurate_counting=False,  # Use estimation for speed
    )


@pytest.fixture
def trimmer(sample_rules, prioritizer) -> InteractiveTrimmer:
    """Create InteractiveTrimmer for testing."""
    return InteractiveTrimmer(
        rules=sample_rules,
        budget=500,  # Tight budget to trigger trimming
        tool_name="claude",
        prioritizer=prioritizer,
        auto_apply_suggestions=False,
    )


def test_initialization_basic(sample_rules, prioritizer):
    """Test basic initialization."""
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=1000,
        tool_name="claude",
        prioritizer=prioritizer,
    )

    assert len(trimmer.rules) == 5
    assert trimmer.budget == 1000
    assert trimmer.tool_name == "claude"
    assert trimmer.strategy == prioritizer.strategy
    assert len(trimmer.selection) == 5  # All rules initialized
    assert len(trimmer.scores_by_id) == 5


def test_initialization_with_auto_apply(sample_rules, prioritizer):
    """Test initialization with auto-apply suggestions."""
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=500,
        tool_name="claude",
        prioritizer=prioritizer,
        auto_apply_suggestions=True,
    )

    viz = trimmer.get_budget_visualization()

    # Should be within budget after auto-apply
    assert viz.used_tokens <= 500
    assert viz.over_budget is False

    # Absolute rule should always be included
    assert trimmer.selection.get("rule-2", False) is True


def test_protected_rules(sample_rules, prioritizer):
    """Test that absolute authority rules are protected."""
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=100,  # Very tight budget
        tool_name="claude",
        prioritizer=prioritizer,
        protect_absolute=True,
        auto_apply_suggestions=True,
    )

    # rule-2 is absolute authority, should be included
    assert trimmer.selection.get("rule-2") is True
    assert "rule-2" in trimmer.protected_rules

    # Try to exclude protected rule should raise error
    with pytest.raises(ValueError, match="Cannot exclude protected rule"):
        trimmer.exclude_rule("rule-2")


def test_apply_auto_suggestions(trimmer):
    """Test applying automatic suggestions."""
    result = trimmer.apply_auto_suggestions()

    assert "rules_included" in result
    assert "rules_excluded" in result
    assert "tokens_used" in result
    assert "within_budget" in result

    # Should fit within budget
    assert result["within_budget"] is True
    assert result["tokens_used"] <= trimmer.budget

    # Absolute rule should be included
    assert trimmer.selection.get("rule-2") is True


def test_toggle_rule(trimmer):
    """Test toggling rule inclusion."""
    # Initially all rules included
    assert trimmer.selection.get("rule-1", True) is True

    # Toggle to exclude
    new_state = trimmer.toggle_rule("rule-1")
    assert new_state is False
    assert trimmer.selection["rule-1"] is False

    # Decision should be manual
    decision = trimmer.decisions.get("rule-1")
    assert decision is not None
    assert decision.decision_type == "manual"
    assert decision.decision == TrimDecisionType.EXCLUDE

    # Toggle back to include
    new_state = trimmer.toggle_rule("rule-1")
    assert new_state is True
    assert trimmer.selection["rule-1"] is True


def test_include_exclude_rules(trimmer):
    """Test explicit include/exclude methods."""
    # Exclude a rule
    trimmer.exclude_rule("rule-1")
    assert trimmer.selection["rule-1"] is False

    # Include it back
    trimmer.include_rule("rule-1")
    assert trimmer.selection["rule-1"] is True

    # Include already included rule (no change)
    trimmer.include_rule("rule-1")
    assert trimmer.selection["rule-1"] is True


def test_include_all(trimmer):
    """Test including all rules."""
    # First exclude some rules
    trimmer.exclude_rule("rule-1")
    trimmer.exclude_rule("rule-3")

    # Include all
    count = trimmer.include_all()

    assert count == len(trimmer.rules)
    for rule in trimmer.rules:
        assert trimmer.selection[rule.id] is True


def test_exclude_all_non_protected(trimmer):
    """Test excluding all non-protected rules."""
    count = trimmer.exclude_all_non_protected()

    # Should exclude 4 rules (all except absolute authority rule-2)
    assert count == 4

    # Absolute rule should still be included
    assert trimmer.selection["rule-2"] is True

    # Others should be excluded
    assert trimmer.selection["rule-1"] is False
    assert trimmer.selection["rule-3"] is False
    assert trimmer.selection["rule-4"] is False
    assert trimmer.selection["rule-5"] is False


def test_reset_to_auto_suggestions(trimmer):
    """Test resetting to auto suggestions."""
    # Make some manual changes
    trimmer.toggle_rule("rule-1")
    trimmer.toggle_rule("rule-3")

    # Reset to auto
    result = trimmer.reset_to_auto_suggestions()

    assert "rules_included" in result
    assert result["within_budget"] is True

    # Decisions should be automatic again
    for rule in trimmer.rules:
        decision = trimmer.decisions.get(rule.id)
        if decision:
            assert decision.decision_type == "automatic"


def test_get_budget_visualization(trimmer):
    """Test budget visualization generation."""
    viz = trimmer.get_budget_visualization()

    assert isinstance(viz, BudgetVisualization)
    assert viz.total_budget == trimmer.budget
    assert viz.used_tokens > 0
    assert viz.remaining_tokens >= 0
    assert viz.utilization_pct >= 0
    assert viz.rules_count == len(trimmer.rules)
    assert viz.protected_count == 1  # Only rule-2 is absolute

    # Initially all rules included
    assert viz.included_count == 5
    assert viz.excluded_count == 0


def test_budget_visualization_over_budget(sample_rules, prioritizer):
    """Test visualization when over budget."""
    # Create trimmer with very small budget
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=50,  # Very small budget
        tool_name="claude",
        prioritizer=prioritizer,
        auto_apply_suggestions=False,  # Don't auto-trim
    )

    viz = trimmer.get_budget_visualization()

    assert viz.over_budget is True
    assert viz.over_budget_amount > 0
    assert viz.used_tokens > viz.total_budget


def test_get_rule_displays(trimmer):
    """Test getting rule display information."""
    displays = trimmer.get_rule_displays()

    assert len(displays) == len(trimmer.rules)

    for display in displays:
        assert isinstance(display, RuleDisplay)
        assert display.rule is not None
        assert display.score is not None
        assert display.display_index > 0
        assert isinstance(display.included, bool)
        assert isinstance(display.protected, bool)

    # Absolute authority rule should be protected
    rule2_display = next(d for d in displays if d.rule.id == "rule-2")
    assert rule2_display.protected is True


def test_get_rule_displays_sorting(trimmer):
    """Test different sorting options for rule displays."""
    # Sort by priority (default)
    priority_sorted = trimmer.get_rule_displays(sort_by="priority")

    # Sort by name
    name_sorted = trimmer.get_rule_displays(sort_by="name")

    # Sort by tokens
    token_sorted = trimmer.get_rule_displays(sort_by="tokens")

    # Sort by category
    category_sorted = trimmer.get_rule_displays(sort_by="category")

    # All should have same length
    assert len(priority_sorted) == len(name_sorted)
    assert len(priority_sorted) == len(token_sorted)
    assert len(priority_sorted) == len(category_sorted)

    # Verify sorting is different
    priority_ids = [d.rule.id for d in priority_sorted]
    name_ids = [d.rule.id for d in name_sorted]
    assert priority_ids != name_ids  # Different order


def test_get_comparison(trimmer):
    """Test getting before/after comparison."""
    # Make some manual changes
    trimmer.apply_auto_suggestions()
    original_included = len(trimmer.get_selected_rules())

    # Make manual changes
    trimmer.toggle_rule("rule-1")
    trimmer.toggle_rule("rule-3")

    comparison = trimmer.get_comparison()

    assert "auto_suggestions" in comparison
    assert "current_selection" in comparison
    assert "manual_changes" in comparison
    assert "budget" in comparison

    # Manual changes should be recorded
    assert comparison["manual_changes"] >= 2


def test_save_and_load_session(trimmer, tmp_path):
    """Test saving and loading session."""
    # Apply auto suggestions and make some changes
    trimmer.apply_auto_suggestions()
    trimmer.toggle_rule("rule-1")

    # Save session
    session_file = tmp_path / "test_session.json"
    trimmer.save_session(session_file)

    assert session_file.exists()

    # Verify JSON content
    with open(session_file) as f:
        session_data = json.load(f)

    assert session_data["tool_name"] == "claude"
    assert session_data["budget"] == 500
    assert "decisions" in session_data
    assert len(session_data["decisions"]) > 0

    # Create new trimmer and load session
    new_trimmer = InteractiveTrimmer(
        rules=trimmer.rules,
        budget=trimmer.budget,
        tool_name=trimmer.tool_name,
        prioritizer=trimmer.prioritizer,
    )

    new_trimmer.load_session(session_file)

    # Decisions should match
    assert len(new_trimmer.decisions) == len(trimmer.decisions)
    assert new_trimmer.selection["rule-1"] == trimmer.selection["rule-1"]


def test_load_session_nonexistent_file(trimmer, tmp_path):
    """Test loading from nonexistent file."""
    session_file = tmp_path / "nonexistent.json"

    with pytest.raises(FileNotFoundError):
        trimmer.load_session(session_file)


def test_get_selected_and_excluded_rules(trimmer):
    """Test getting selected and excluded rule lists."""
    # Apply auto suggestions
    trimmer.apply_auto_suggestions()

    selected = trimmer.get_selected_rules()
    excluded = trimmer.get_excluded_rules()

    # All rules should be either selected or excluded
    assert len(selected) + len(excluded) == len(trimmer.rules)

    # No overlap
    selected_ids = {r.id for r in selected}
    excluded_ids = {r.id for r in excluded}
    assert len(selected_ids.intersection(excluded_ids)) == 0


def test_decision_tracking(trimmer):
    """Test that decisions are properly tracked."""
    # Apply auto suggestions
    trimmer.apply_auto_suggestions()

    # Check automatic decisions
    for rule in trimmer.rules:
        decision = trimmer.decisions.get(rule.id)
        assert decision is not None
        assert decision.decision_type == "automatic"
        assert isinstance(decision.timestamp, datetime)

    # Make manual change
    trimmer.toggle_rule("rule-1")

    # Check manual decision
    decision = trimmer.decisions["rule-1"]
    assert decision.decision_type == "manual"
    assert "Manually toggled" in decision.reason


def test_multiple_strategies(sample_rules, prioritizer):
    """Test using different prioritization strategies."""
    # Test with IMPORTANCE strategy
    trimmer_importance = InteractiveTrimmer(
        rules=sample_rules,
        budget=500,
        tool_name="claude",
        prioritizer=prioritizer,
        strategy=PrioritizationStrategy.IMPORTANCE,
        auto_apply_suggestions=True,
    )

    # Test with COST_BENEFIT strategy
    trimmer_cost_benefit = InteractiveTrimmer(
        rules=sample_rules,
        budget=500,
        tool_name="claude",
        prioritizer=prioritizer,
        strategy=PrioritizationStrategy.COST_BENEFIT,
        auto_apply_suggestions=True,
    )

    # Both should fit within budget
    viz1 = trimmer_importance.get_budget_visualization()
    viz2 = trimmer_cost_benefit.get_budget_visualization()

    assert viz1.used_tokens <= 500
    assert viz2.used_tokens <= 500

    # Selection might be different due to different strategies
    selected1 = trimmer_importance.get_selected_rules()
    selected2 = trimmer_cost_benefit.get_selected_rules()

    # Both should include absolute rule
    assert any(r.id == "rule-2" for r in selected1)
    assert any(r.id == "rule-2" for r in selected2)


def test_empty_rules_list(prioritizer):
    """Test with empty rules list."""
    trimmer = InteractiveTrimmer(
        rules=[],
        budget=1000,
        tool_name="claude",
        prioritizer=prioritizer,
    )

    viz = trimmer.get_budget_visualization()
    assert viz.rules_count == 0
    assert viz.included_count == 0
    assert viz.used_tokens == 0

    displays = trimmer.get_rule_displays()
    assert len(displays) == 0


def test_very_large_budget(sample_rules, prioritizer):
    """Test with very large budget (all rules fit)."""
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=100000,  # Very large budget
        tool_name="claude",
        prioritizer=prioritizer,
        auto_apply_suggestions=True,
    )

    viz = trimmer.get_budget_visualization()

    # All rules should be included
    assert viz.included_count == len(sample_rules)
    assert viz.excluded_count == 0
    assert viz.over_budget is False


def test_protect_absolute_false(sample_rules, prioritizer):
    """Test with protect_absolute=False."""
    trimmer = InteractiveTrimmer(
        rules=sample_rules,
        budget=100,  # Very tight budget
        tool_name="claude",
        prioritizer=prioritizer,
        protect_absolute=False,
        auto_apply_suggestions=True,
    )

    # Absolute rule might be excluded if it doesn't fit
    viz = trimmer.get_budget_visualization()
    assert viz.used_tokens <= 100

    # Protected rules set should be empty
    assert len(trimmer.protected_rules) == 0

    # Should be able to exclude absolute rule
    if trimmer.selection.get("rule-2", True):
        trimmer.exclude_rule("rule-2")
        assert trimmer.selection["rule-2"] is False


def test_session_persistence_format(trimmer, tmp_path):
    """Test session file format and structure."""
    trimmer.apply_auto_suggestions()
    session_file = tmp_path / "session.json"
    trimmer.save_session(session_file)

    with open(session_file) as f:
        data = json.load(f)

    # Verify required fields
    assert "tool_name" in data
    assert "budget" in data
    assert "strategy" in data
    assert "decisions" in data
    assert "total_tokens_used" in data
    assert "rules_included" in data
    assert "rules_excluded" in data
    assert "created_at" in data
    assert "metadata" in data

    # Verify strategy is serialized as string
    assert isinstance(data["strategy"], str)

    # Verify decisions format
    for rule_id, decision in data["decisions"].items():
        assert "rule_id" in decision
        assert "decision" in decision
        assert "decision_type" in decision
        assert "reason" in decision
        assert "timestamp" in decision
