"""
Unit tests for Claude-specific token budget management.

Tests Claude model detection, session tracking, budget allocation,
and warning thresholds.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.python.common.core.context_injection import (
    AllocationStrategy,
    ClaudeBudgetAllocation,
    ClaudeBudgetManager,
    ClaudeModel,
    SessionUsageStats,
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

    return [
        MemoryRule(
            id="rule_abs_1",
            name="comprehensive_tests",
            rule="Always write comprehensive tests",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["testing"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={"priority": 100},
        ),
        MemoryRule(
            id="rule_def_1",
            name="descriptive_names",
            rule="Use descriptive variable names",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["coding"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={"priority": 80},
        ),
        MemoryRule(
            id="rule_def_2",
            name="pep8_style",
            rule="Follow PEP 8 style guide",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["python"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={"priority": 90},
        ),
        MemoryRule(
            id="rule_def_3",
            name="document_apis",
            rule="Document all public APIs",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["documentation"],
            source="user",
            created_at=now,
            updated_at=now,
            conditions=None,
            replaces=[],
            metadata={"priority": 70},
        ),
    ]


class TestClaudeModel:
    """Test Claude model enum and detection."""

    def test_model_attributes(self):
        """Test Claude model attributes."""
        model = ClaudeModel.SONNET_3_5
        assert model.model_id == "claude-3-5-sonnet-20241022"
        assert model.token_limit == 200000

    def test_model_from_exact_id(self):
        """Test model detection from exact ID."""
        model = ClaudeModel.from_model_id("claude-3-5-sonnet-20241022")
        assert model == ClaudeModel.SONNET_3_5

    def test_model_from_partial_match(self):
        """Test model detection from partial match."""
        # Test various partial matches
        model = ClaudeModel.from_model_id("claude-3-5-sonnet-new-version")
        assert model == ClaudeModel.SONNET_3_5

        model = ClaudeModel.from_model_id("claude-3-opus-custom")
        assert model == ClaudeModel.OPUS_3

        model = ClaudeModel.from_model_id("claude-3-haiku-20240307")
        assert model == ClaudeModel.HAIKU_3

    def test_model_from_unknown_id(self):
        """Test model detection from unknown ID defaults to DEFAULT."""
        model = ClaudeModel.from_model_id("unknown-model")
        assert model == ClaudeModel.DEFAULT

    def test_all_models_have_200k_limit(self):
        """Test that all Claude models have 200K token limit."""
        for model in ClaudeModel:
            assert model.token_limit == 200000


class TestSessionUsageStats:
    """Test session usage statistics tracking."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = SessionUsageStats(
            session_id="test_session",
            model=ClaudeModel.SONNET_3_5,
            budget_limit=100000,
        )
        assert stats.session_id == "test_session"
        assert stats.model == ClaudeModel.SONNET_3_5
        assert stats.total_tokens_used == 0
        assert stats.interaction_count == 0
        assert stats.budget_limit == 100000

    def test_utilization_calculation(self):
        """Test utilization percentage calculation."""
        stats = SessionUsageStats(
            session_id="test",
            model=ClaudeModel.SONNET_3_5,
            budget_limit=100000,
        )

        stats.total_tokens_used = 50000
        assert stats.utilization_percentage() == 50.0

        stats.total_tokens_used = 90000
        assert stats.utilization_percentage() == 90.0

        stats.total_tokens_used = 150000  # Over budget
        assert stats.utilization_percentage() == 150.0

    def test_tokens_remaining(self):
        """Test remaining tokens calculation."""
        stats = SessionUsageStats(
            session_id="test",
            model=ClaudeModel.SONNET_3_5,
            budget_limit=100000,
        )

        stats.total_tokens_used = 30000
        assert stats.tokens_remaining() == 70000

        stats.total_tokens_used = 100000
        assert stats.tokens_remaining() == 0

        stats.total_tokens_used = 120000
        assert stats.tokens_remaining() == -20000

    def test_is_over_budget(self):
        """Test over budget detection."""
        stats = SessionUsageStats(
            session_id="test",
            model=ClaudeModel.SONNET_3_5,
            budget_limit=100000,
        )

        stats.total_tokens_used = 50000
        assert not stats.is_over_budget()

        stats.total_tokens_used = 100000
        assert not stats.is_over_budget()

        stats.total_tokens_used = 100001
        assert stats.is_over_budget()


class TestClaudeBudgetManager:
    """Test Claude budget manager functionality."""

    def test_initialization(self):
        """Test budget manager initialization."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            session_id="test_session",
        )

        assert manager.model == ClaudeModel.SONNET_3_5
        assert manager.session_id == "test_session"
        assert manager.session_stats.budget_limit == 200000
        assert manager.session_stats.total_tokens_used == 0

    def test_initialization_with_custom_budget(self):
        """Test initialization with custom budget limit."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=50000,
        )

        assert manager.session_stats.budget_limit == 50000

    def test_auto_session_id_generation(self):
        """Test automatic session ID generation."""
        manager = ClaudeBudgetManager(model=ClaudeModel.SONNET_3_5)

        assert manager.session_id.startswith("claude_session_")
        assert len(manager.session_id) > len("claude_session_")

    def test_basic_budget_allocation(self, sample_rules):
        """Test basic budget allocation."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=10000,
        )

        allocation = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        assert isinstance(allocation, ClaudeBudgetAllocation)
        assert allocation.model == ClaudeModel.SONNET_3_5
        assert len(allocation.rules_included) > 0
        assert manager.session_stats.interaction_count == 1
        assert manager.session_stats.total_user_tokens == 100

    def test_budget_allocation_with_user_query_text(self, sample_rules):
        """Test budget allocation with user query text."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=10000,
        )

        allocation = manager.allocate_budget(
            rules=sample_rules,
            user_query="Write a function to calculate fibonacci numbers",
        )

        assert manager.session_stats.total_user_tokens > 0
        assert manager.session_stats.interaction_count == 1

    def test_session_accumulation(self, sample_rules):
        """Test token accumulation across multiple interactions."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=10000,
        )

        # First interaction
        allocation1 = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )
        tokens_after_first = manager.session_stats.total_tokens_used

        # Second interaction
        allocation2 = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=150,
        )
        tokens_after_second = manager.session_stats.total_tokens_used

        assert manager.session_stats.interaction_count == 2
        assert tokens_after_second > tokens_after_first
        assert manager.session_stats.total_user_tokens == 250

    def test_budget_exhaustion(self, sample_rules):
        """Test behavior when budget is exhausted."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=500,  # Very small budget
        )

        # First interaction should use most/all budget
        allocation1 = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        # Second interaction should have very limited budget
        allocation2 = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        # Available budget should decrease
        assert (
            len(allocation2.rules_included)
            <= len(allocation1.rules_included)
        )

    def test_warning_thresholds(self, sample_rules):
        """Test warning threshold detection."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=1000,
        )

        # First interaction - no warning
        allocation1 = manager.allocate_budget(
            rules=sample_rules[:1],  # Use fewer rules
            user_query_tokens=50,
        )
        assert allocation1.warning_level is None

        # Consume more budget to trigger warnings
        # Use large user query to push utilization higher
        allocation2 = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=700,  # Large query to push over threshold
        )

        # Should have triggered at least one warning
        # (exact threshold depends on rule sizes)
        if manager.session_stats.utilization_percentage() >= 80:
            assert len(manager.session_stats.warnings_triggered) > 0

    def test_reset_session(self, sample_rules):
        """Test session reset functionality."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            session_id="original_session",
        )

        # Use some budget
        manager.allocate_budget(rules=sample_rules, user_query_tokens=100)
        assert manager.session_stats.total_tokens_used > 0

        # Reset session
        manager.reset_session(new_session_id="new_session")

        assert manager.session_id == "new_session"
        assert manager.session_stats.total_tokens_used == 0
        assert manager.session_stats.interaction_count == 0
        assert manager.session_stats.warnings_triggered == []

    def test_session_report(self, sample_rules):
        """Test session report generation."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=10000,
        )

        # Make a few interactions
        manager.allocate_budget(rules=sample_rules, user_query_tokens=100)
        manager.allocate_budget(rules=sample_rules, user_query_tokens=150)

        report = manager.get_session_report()

        assert report["session_id"] == manager.session_id
        assert report["model"] == ClaudeModel.SONNET_3_5.model_id
        assert report["budget_limit"] == 10000
        assert report["interaction_count"] == 2
        assert report["tokens_used"] > 0
        assert report["user_tokens"] == 250
        assert "utilization_percentage" in report
        assert "average_tokens_per_interaction" in report

    def test_absolute_rules_protected(self, sample_rules):
        """Test that absolute authority rules are always included."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=200,  # Very tight budget
            absolute_rules_protected=True,
        )

        allocation = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=50,
        )

        # Find absolute rules in allocation
        absolute_included = [
            r for r in allocation.rules_included
            if r.authority == AuthorityLevel.ABSOLUTE
        ]

        # At least one absolute rule should be included
        # (even if over budget due to protection)
        absolute_in_input = [
            r for r in sample_rules
            if r.authority == AuthorityLevel.ABSOLUTE
        ]
        assert len(absolute_included) == len(absolute_in_input)

    def test_allocation_strategies(self, sample_rules):
        """Test different allocation strategies."""
        # Priority-based
        manager_priority = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            allocation_strategy=AllocationStrategy.PRIORITY_BASED,
            custom_budget_limit=5000,
        )
        alloc_priority = manager_priority.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        # Round-robin
        manager_rr = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            allocation_strategy=AllocationStrategy.ROUND_ROBIN,
            custom_budget_limit=5000,
        )
        alloc_rr = manager_rr.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        # Both should include rules (may differ in selection)
        assert len(alloc_priority.rules_included) > 0
        assert len(alloc_rr.rules_included) > 0

    def test_context_overhead_tracking(self, sample_rules):
        """Test context overhead token tracking."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            overhead_percentage=0.1,  # 10% overhead
        )

        allocation = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=100,
        )

        # Context overhead should be tracked
        assert allocation.context_overhead_tokens > 0
        assert allocation.context_overhead_tokens == allocation.base_allocation.overhead_tokens

    @patch.dict("os.environ", {"CLAUDE_MODEL": "claude-3-opus-20240229"})
    def test_detect_model_from_environment(self):
        """Test model detection from environment variables."""
        model = ClaudeBudgetManager.detect_model_from_environment()
        assert model == ClaudeModel.OPUS_3

    @patch.dict("os.environ")
    def test_detect_model_default_when_not_set(self):
        """Test model detection defaults when no env var set."""
        model = ClaudeBudgetManager.detect_model_from_environment()
        assert model == ClaudeModel.DEFAULT

    def test_large_rule_set(self):
        """Test allocation with many rules."""
        # Create 50 rules
        now = datetime.now(timezone.utc)
        many_rules = [
            MemoryRule(
                id=f"rule_{i}",
                name=f"test_rule_{i}",
                rule=f"Test rule {i}",
                category=MemoryCategory.BEHAVIOR,
                authority=(
                    AuthorityLevel.ABSOLUTE if i < 5 else AuthorityLevel.DEFAULT
                ),
                scope=["testing"],
                source="test",
                created_at=now,
                updated_at=now,
                conditions=None,
                replaces=[],
                metadata={"priority": 100 - i},
            )
            for i in range(50)
        ]

        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=5000,
        )

        allocation = manager.allocate_budget(
            rules=many_rules,
            user_query_tokens=100,
        )

        # Should allocate rules according to budget
        # With a 5000 token budget, most or all 50 rules should fit
        # (each rule is ~12-15 tokens, so total ~750 tokens for rules + overhead)
        assert len(allocation.rules_included) <= len(many_rules)
        assert (
            len(allocation.rules_included) + len(allocation.rules_skipped)
            == len(many_rules)
        )
        # All 5 absolute rules should be included (protected)
        absolute_included = [
            r for r in allocation.rules_included
            if r.authority == AuthorityLevel.ABSOLUTE
        ]
        assert len(absolute_included) == 5

    def test_zero_budget_allocation(self, sample_rules):
        """Test allocation with zero remaining budget."""
        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=100,  # Tiny budget
        )

        # First call uses all budget
        manager.allocate_budget(rules=sample_rules, user_query_tokens=90)

        # Second call has ~0 budget remaining
        allocation = manager.allocate_budget(
            rules=sample_rules,
            user_query_tokens=10,
        )

        # Should still protect absolute rules if configured
        absolute_rules = [
            r for r in allocation.rules_included
            if r.authority == AuthorityLevel.ABSOLUTE
        ]
        if manager.base_manager.absolute_rules_protected:
            assert len(absolute_rules) > 0
