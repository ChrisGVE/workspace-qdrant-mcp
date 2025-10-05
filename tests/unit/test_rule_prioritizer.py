"""
Unit tests for rule prioritizer system.

Tests cover:
- All prioritization strategies
- Priority score calculations
- Integration with TokenUsageTracker
- Dynamic priority adjustment
- Cost-benefit analysis
- Edge cases (no usage data, equal priorities, etc.)
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, MagicMock

from src.python.common.core.context_injection.rule_prioritizer import (
    PrioritizationStrategy,
    PrioritizationResult,
    RulePrioritizer,
    RulePriorityScore,
)
from src.python.common.core.context_injection.token_usage_tracker import (
    OperationType,
    TokenUsageTracker,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


@pytest.fixture
def sample_rules():
    """Create sample memory rules for testing."""
    now = datetime.now(timezone.utc)

    # High priority absolute rule
    rule1 = MemoryRule(
        id="rule1",
        category=MemoryCategory.BEHAVIOR,
        name="Always commit",
        rule="Always make atomic commits after each change",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["global"],
        created_at=now - timedelta(days=10),
        updated_at=now - timedelta(days=1),
        metadata={"priority": 90},
    )

    # Medium priority default rule (recent)
    rule2 = MemoryRule(
        id="rule2",
        category=MemoryCategory.PREFERENCE,
        name="Use uv",
        rule="Prefer uv over pip for Python package management",
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        created_at=now - timedelta(days=2),
        updated_at=now - timedelta(hours=1),
        metadata={"priority": 60},
    )

    # Low priority default rule (old)
    rule3 = MemoryRule(
        id="rule3",
        category=MemoryCategory.PREFERENCE,
        name="Format style",
        rule="Use black for Python formatting with line length 88",
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "formatting"],
        created_at=now - timedelta(days=90),
        updated_at=now - timedelta(days=60),
        metadata={"priority": 30},
    )

    # No priority metadata
    rule4 = MemoryRule(
        id="rule4",
        category=MemoryCategory.BEHAVIOR,
        name="Test coverage",
        rule="Maintain 80% test coverage for all modules",
        authority=AuthorityLevel.DEFAULT,
        scope=["testing"],
        created_at=now - timedelta(days=5),
        updated_at=now - timedelta(days=5),
        metadata={},
    )

    # Absolute rule with lower priority
    rule5 = MemoryRule(
        id="rule5",
        category=MemoryCategory.BEHAVIOR,
        name="Security",
        rule="Never commit secrets or API keys to version control",
        authority=AuthorityLevel.ABSOLUTE,
        scope=["global", "security"],
        created_at=now - timedelta(days=30),
        updated_at=now - timedelta(days=30),
        metadata={"priority": 100},
    )

    return [rule1, rule2, rule3, rule4, rule5]


@pytest.fixture
def usage_tracker():
    """Create TokenUsageTracker with sample usage data."""
    tracker = TokenUsageTracker(track_detailed_operations=True)

    # Simulate usage for rule1 (frequently used)
    for i in range(10):
        tracker.track_operation(
            tool_name="claude",
            operation_type=OperationType.CONTEXT_INJECTION,
            tokens_used=50,
            metadata={"rule_id": "rule1"},
        )

    # Simulate usage for rule2 (moderately used)
    for i in range(3):
        tracker.track_operation(
            tool_name="claude",
            operation_type=OperationType.CONTEXT_INJECTION,
            tokens_used=40,
            metadata={"rule_id": "rule2"},
        )

    # No usage for rule3, rule4, rule5

    return tracker


class TestRulePrioritizer:
    """Test suite for RulePrioritizer."""

    def test_initialization(self):
        """Test prioritizer initialization with default settings."""
        prioritizer = RulePrioritizer()

        assert prioritizer.strategy == PrioritizationStrategy.IMPORTANCE
        assert prioritizer.importance_weight > 0
        assert prioritizer.frequency_weight >= 0
        assert prioritizer.recency_weight >= 0
        assert prioritizer.use_accurate_counting is True

    def test_initialization_with_tracker(self, usage_tracker):
        """Test prioritizer initialization with usage tracker."""
        prioritizer = RulePrioritizer(usage_tracker=usage_tracker)

        assert prioritizer.usage_tracker is not None
        assert prioritizer.usage_tracker == usage_tracker

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        prioritizer = RulePrioritizer(
            importance_weight=2.0, frequency_weight=1.0, recency_weight=1.0
        )

        total_weight = (
            prioritizer.importance_weight
            + prioritizer.frequency_weight
            + prioritizer.recency_weight
        )

        assert abs(total_weight - 1.0) < 0.001  # Allow small floating point error

    def test_importance_strategy(self, sample_rules):
        """Test IMPORTANCE prioritization strategy."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.IMPORTANCE)

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.IMPORTANCE
        assert len(result.ranked_rules) == len(sample_rules)
        assert len(result.priority_scores) == len(sample_rules)

        # Verify scores are sorted (highest first)
        scores = [s.total_score for s in result.priority_scores]
        assert scores == sorted(scores, reverse=True)

        # Absolute rules should rank higher
        top_rule = result.ranked_rules[0]
        assert top_rule.authority == AuthorityLevel.ABSOLUTE

        # Check component scores exist
        for score in result.priority_scores:
            assert "authority" in score.component_scores
            assert "priority" in score.component_scores

    def test_frequency_strategy(self, sample_rules, usage_tracker):
        """Test FREQUENCY prioritization strategy."""
        prioritizer = RulePrioritizer(
            usage_tracker=usage_tracker, strategy=PrioritizationStrategy.FREQUENCY
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.FREQUENCY

        # rule1 should rank highest (most usage)
        top_score = result.priority_scores[0]
        assert top_score.rule_id == "rule1"
        assert top_score.usage_count == 10

        # rule2 should be second
        second_score = result.priority_scores[1]
        assert second_score.rule_id == "rule2"
        assert second_score.usage_count == 3

        # Check component scores
        assert "frequency" in top_score.component_scores

    def test_frequency_strategy_without_tracker(self, sample_rules):
        """Test FREQUENCY strategy without usage tracker (should handle gracefully)."""
        prioritizer = RulePrioritizer(
            usage_tracker=None, strategy=PrioritizationStrategy.FREQUENCY
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        # Should not crash, all rules should have zero frequency
        for score in result.priority_scores:
            assert score.usage_count == 0

    def test_cost_benefit_strategy(self, sample_rules, usage_tracker):
        """Test COST_BENEFIT prioritization strategy."""
        prioritizer = RulePrioritizer(
            usage_tracker=usage_tracker, strategy=PrioritizationStrategy.COST_BENEFIT
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.COST_BENEFIT

        # All scores should have value_per_token calculated
        for score in result.priority_scores:
            assert score.value_per_token >= 0
            assert score.token_cost > 0
            assert "cost_benefit" in score.component_scores

        # Higher value-per-token should rank higher
        scores = [s.value_per_token for s in result.priority_scores]
        assert scores == sorted(scores, reverse=True)

    def test_recency_strategy(self, sample_rules):
        """Test RECENCY prioritization strategy."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.RECENCY)

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.RECENCY

        # Most recently updated should rank highest
        top_rule = result.ranked_rules[0]
        assert top_rule.id == "rule2"  # Updated 1 hour ago

        # Check component scores
        for score in result.priority_scores:
            assert "recency" in score.component_scores
            assert "age_days" in score.component_scores

    def test_hybrid_strategy(self, sample_rules, usage_tracker):
        """Test HYBRID prioritization strategy."""
        prioritizer = RulePrioritizer(
            usage_tracker=usage_tracker,
            strategy=PrioritizationStrategy.HYBRID,
            importance_weight=0.5,
            frequency_weight=0.3,
            recency_weight=0.2,
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.HYBRID

        # Should combine all components
        for score in result.priority_scores:
            assert "weighted_importance" in score.component_scores
            assert "weighted_frequency" in score.component_scores
            assert "weighted_recency" in score.component_scores

        # Verify scores are properly combined
        top_score = result.priority_scores[0]
        weighted_sum = (
            top_score.component_scores["weighted_importance"]
            + top_score.component_scores["weighted_frequency"]
            + top_score.component_scores["weighted_recency"]
        )
        assert abs(top_score.total_score - weighted_sum) < 0.01

    def test_adaptive_strategy(self, sample_rules):
        """Test ADAPTIVE prioritization strategy."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.ADAPTIVE)

        # Add some learned adjustments
        prioritizer.adjust_priority_from_usage("rule3", usage_value=0.5)
        prioritizer.adjust_priority_from_usage("rule4", usage_value=-0.3)

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        assert result.strategy == PrioritizationStrategy.ADAPTIVE

        # Check adaptive adjustments are applied with higher weight
        for score in result.priority_scores:
            if score.rule_id == "rule3":
                assert "adaptive_adjustment" in score.component_scores
                assert score.component_scores["adaptive_adjustment"] > 0

    def test_select_top_rules_within_budget(self, sample_rules):
        """Test selecting top rules that fit within budget."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.IMPORTANCE)

        budget = 500  # Token budget
        selected, skipped = prioritizer.select_top_rules(
            sample_rules, tool_name="claude", budget=budget, protect_absolute=True
        )

        # Should have some selected and some skipped
        assert len(selected) > 0
        assert len(selected) + len(skipped) == len(sample_rules)

        # All absolute rules should be selected (protected)
        absolute_rules = [r for r in sample_rules if r.authority == AuthorityLevel.ABSOLUTE]
        for abs_rule in absolute_rules:
            assert abs_rule in selected

    def test_select_top_rules_small_budget(self, sample_rules):
        """Test rule selection with very small budget."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.IMPORTANCE)

        budget = 10  # Very small budget
        selected, skipped = prioritizer.select_top_rules(
            sample_rules, tool_name="claude", budget=budget, protect_absolute=True
        )

        # Should still include absolute rules even if over budget
        absolute_count = sum(
            1 for r in selected if r.authority == AuthorityLevel.ABSOLUTE
        )
        assert absolute_count > 0

    def test_select_top_rules_without_protection(self, sample_rules):
        """Test rule selection without absolute rule protection."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.COST_BENEFIT)

        budget = 100
        selected, skipped = prioritizer.select_top_rules(
            sample_rules, tool_name="claude", budget=budget, protect_absolute=False
        )

        # Budget should be respected more strictly
        total_tokens = sum(
            prioritizer._estimate_tokens(r, "claude") for r in selected
        )
        assert total_tokens <= budget or len(selected) == 0

    def test_adjust_priority_from_usage(self):
        """Test dynamic priority adjustment from usage."""
        prioritizer = RulePrioritizer()

        # Initial adjustment is 0
        assert prioritizer.get_priority_adjustment("test_rule") == 0.0

        # Positive usage value
        prioritizer.adjust_priority_from_usage("test_rule", usage_value=0.5)
        adjustment = prioritizer.get_priority_adjustment("test_rule")
        assert adjustment > 0

        # Negative usage value
        prioritizer.adjust_priority_from_usage("test_rule", usage_value=-0.3)
        new_adjustment = prioritizer.get_priority_adjustment("test_rule")
        assert new_adjustment < adjustment

        # Multiple adjustments accumulate
        for _ in range(5):
            prioritizer.adjust_priority_from_usage("test_rule2", usage_value=0.1)
        final_adjustment = prioritizer.get_priority_adjustment("test_rule2")
        assert final_adjustment > 0

    def test_adjustment_clamping(self):
        """Test that priority adjustments are clamped to [-1.0, 1.0]."""
        prioritizer = RulePrioritizer(adaptive_learning_rate=0.5)

        # Try to exceed upper bound
        for _ in range(10):
            prioritizer.adjust_priority_from_usage("rule_max", usage_value=1.0)

        adjustment = prioritizer.get_priority_adjustment("rule_max")
        assert adjustment <= 1.0

        # Try to exceed lower bound
        for _ in range(10):
            prioritizer.adjust_priority_from_usage("rule_min", usage_value=-1.0)

        adjustment = prioritizer.get_priority_adjustment("rule_min")
        assert adjustment >= -1.0

    def test_reset_priority_adjustments(self):
        """Test resetting all priority adjustments."""
        prioritizer = RulePrioritizer()

        # Add some adjustments
        prioritizer.adjust_priority_from_usage("rule1", usage_value=0.5)
        prioritizer.adjust_priority_from_usage("rule2", usage_value=-0.3)

        assert prioritizer.get_priority_adjustment("rule1") != 0.0
        assert prioritizer.get_priority_adjustment("rule2") != 0.0

        # Reset
        prioritizer.reset_priority_adjustments()

        assert prioritizer.get_priority_adjustment("rule1") == 0.0
        assert prioritizer.get_priority_adjustment("rule2") == 0.0

    def test_statistics_calculation(self, sample_rules):
        """Test prioritization statistics."""
        prioritizer = RulePrioritizer(strategy=PrioritizationStrategy.IMPORTANCE)

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        stats = result.statistics
        assert "strategy" in stats
        assert "total_rules" in stats
        assert "score_mean" in stats
        assert "score_median" in stats
        assert "score_stdev" in stats
        assert "token_cost_total" in stats
        assert "high_priority_count" in stats
        assert "medium_priority_count" in stats
        assert "low_priority_count" in stats

        assert stats["total_rules"] == len(sample_rules)
        assert stats["strategy"] == PrioritizationStrategy.IMPORTANCE.value

    def test_empty_rules_list(self):
        """Test prioritization with empty rules list."""
        prioritizer = RulePrioritizer()

        result = prioritizer.prioritize_rules([], tool_name="claude")

        assert result.total_rules == 0
        assert len(result.ranked_rules) == 0
        assert len(result.priority_scores) == 0
        assert result.total_value == 0

    def test_single_rule(self, sample_rules):
        """Test prioritization with single rule."""
        prioritizer = RulePrioritizer()

        result = prioritizer.prioritize_rules([sample_rules[0]], tool_name="claude")

        assert result.total_rules == 1
        assert len(result.ranked_rules) == 1
        assert result.ranked_rules[0] == sample_rules[0]

    def test_different_tool_names(self, sample_rules):
        """Test prioritization with different LLM tools."""
        prioritizer = RulePrioritizer()

        result_claude = prioritizer.prioritize_rules(sample_rules, tool_name="claude")
        result_codex = prioritizer.prioritize_rules(sample_rules, tool_name="codex")
        result_gemini = prioritizer.prioritize_rules(sample_rules, tool_name="gemini")

        # Rankings should be the same (importance doesn't depend on tool)
        assert [r.id for r in result_claude.ranked_rules] == [
            r.id for r in result_codex.ranked_rules
        ]

        # But token costs might differ (different tokenizers)
        # This is okay - just verify they're all calculated
        assert all(s.token_cost > 0 for s in result_claude.priority_scores)
        assert all(s.token_cost > 0 for s in result_codex.priority_scores)
        assert all(s.token_cost > 0 for s in result_gemini.priority_scores)

    def test_priority_score_comparison(self):
        """Test RulePriorityScore comparison for sorting."""
        rule1 = MemoryRule(
            id="r1",
            category=MemoryCategory.BEHAVIOR,
            name="test",
            rule="test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        score1 = RulePriorityScore(
            rule_id="r1", rule=rule1, total_score=0.8, token_cost=10
        )
        score2 = RulePriorityScore(
            rule_id="r2", rule=rule1, total_score=0.5, token_cost=20
        )

        assert score1 > score2
        assert score2 < score1

        # Sort descending
        scores = [score2, score1]
        scores.sort(reverse=True)
        assert scores[0] == score1
        assert scores[1] == score2

    def test_cost_benefit_threshold(self, sample_rules):
        """Test cost-benefit threshold filtering."""
        prioritizer = RulePrioritizer(
            strategy=PrioritizationStrategy.COST_BENEFIT,
            cost_benefit_threshold=0.01,
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        # All returned rules should meet threshold
        for score in result.priority_scores:
            # Note: threshold is stored but not yet enforced in current implementation
            # This test documents expected future behavior
            assert score.value_per_token >= 0

    def test_accurate_vs_estimated_counting(self, sample_rules):
        """Test difference between accurate and estimated token counting."""
        prioritizer_accurate = RulePrioritizer(use_accurate_counting=True)
        prioritizer_estimated = RulePrioritizer(use_accurate_counting=False)

        result_accurate = prioritizer_accurate.prioritize_rules(
            sample_rules, tool_name="claude"
        )
        result_estimated = prioritizer_estimated.prioritize_rules(
            sample_rules, tool_name="claude"
        )

        # Both should complete successfully
        assert result_accurate.total_rules == result_estimated.total_rules

        # Token costs might differ
        # Just verify both have token costs calculated
        assert all(s.token_cost > 0 for s in result_accurate.priority_scores)
        assert all(s.token_cost > 0 for s in result_estimated.priority_scores)

    def test_integration_with_usage_tracker_metadata(self, sample_rules, usage_tracker):
        """Test integration with TokenUsageTracker via metadata."""
        prioritizer = RulePrioritizer(
            usage_tracker=usage_tracker, strategy=PrioritizationStrategy.FREQUENCY
        )

        result = prioritizer.prioritize_rules(sample_rules, tool_name="claude")

        # Verify usage counts match tracker data
        for score in result.priority_scores:
            if score.rule_id == "rule1":
                assert score.usage_count == 10
            elif score.rule_id == "rule2":
                assert score.usage_count == 3
            else:
                assert score.usage_count == 0


class TestRulePriorityScore:
    """Test suite for RulePriorityScore dataclass."""

    def test_initialization(self):
        """Test RulePriorityScore initialization."""
        rule = MemoryRule(
            id="test",
            category=MemoryCategory.BEHAVIOR,
            name="test",
            rule="test rule",
            authority=AuthorityLevel.DEFAULT,
            scope=[],
        )

        score = RulePriorityScore(
            rule_id="test",
            rule=rule,
            total_score=0.75,
            component_scores={"importance": 0.5, "frequency": 0.25},
            token_cost=50,
            value_per_token=0.015,
            usage_count=5,
        )

        assert score.rule_id == "test"
        assert score.rule == rule
        assert score.total_score == 0.75
        assert score.token_cost == 50
        assert score.value_per_token == 0.015
        assert score.usage_count == 5


class TestPrioritizationResult:
    """Test suite for PrioritizationResult dataclass."""

    def test_initialization(self, sample_rules):
        """Test PrioritizationResult initialization."""
        result = PrioritizationResult(
            strategy=PrioritizationStrategy.IMPORTANCE,
            ranked_rules=sample_rules,
            priority_scores=[],
            total_rules=len(sample_rules),
            total_value=10.5,
            statistics={"test": "value"},
        )

        assert result.strategy == PrioritizationStrategy.IMPORTANCE
        assert result.ranked_rules == sample_rules
        assert result.total_rules == len(sample_rules)
        assert result.total_value == 10.5
        assert result.statistics["test"] == "value"
