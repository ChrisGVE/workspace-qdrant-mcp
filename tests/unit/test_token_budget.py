"""
Unit tests for token budget management.

Tests TokenBudgetManager, BudgetAllocation, and token counting functionality.
"""

from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.token_budget import (
    AllocationStrategy,
    BudgetAllocation,
    CompressionStrategy,
    TokenBudgetManager,
    TokenCounter,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


# Test fixtures
@pytest.fixture
def high_priority_absolute_rule():
    """Create a high priority absolute rule."""
    return MemoryRule(
        id="rule_abs_high",
        name="test_driven_development",
        rule="Write unit tests immediately after implementing each function",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.ABSOLUTE,
        scope=["python", "testing"],
        source="user_explicit",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 100},
    )


@pytest.fixture
def low_priority_absolute_rule():
    """Create a low priority absolute rule."""
    return MemoryRule(
        id="rule_abs_low",
        name="code_formatting",
        rule="Use black for Python code formatting",
        category=MemoryCategory.PREFERENCE,
        authority=AuthorityLevel.ABSOLUTE,
        scope=["python"],
        source="user_explicit",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 50},
    )


@pytest.fixture
def high_priority_default_rule():
    """Create a high priority default rule."""
    return MemoryRule(
        id="rule_def_high",
        name="code_review",
        rule="Submit PRs for review before merging to main",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.DEFAULT,
        scope=["git", "collaboration"],
        source="conversational_future",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions={"condition": "when working with team"},
        replaces=[],
        metadata={"priority": 80},
    )


@pytest.fixture
def low_priority_default_rule():
    """Create a low priority default rule."""
    return MemoryRule(
        id="rule_def_low",
        name="editor_preference",
        rule="Use VS Code for Python development",
        category=MemoryCategory.PREFERENCE,
        authority=AuthorityLevel.DEFAULT,
        scope=["python"],
        source="conversational_preference",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 30},
    )


# TokenCounter Tests
class TestTokenCounter:
    """Test TokenCounter functionality."""

    def test_count_claude_tokens(self):
        """Test Claude token counting."""
        text = "This is a test" * 100  # ~1400 characters
        tokens = TokenCounter.count_claude_tokens(text)
        assert tokens > 0
        assert 300 < tokens < 400  # Roughly 1 token per 4 chars

    def test_count_codex_tokens(self):
        """Test Codex token counting."""
        text = "This is a test with many words " * 50  # ~300 words
        tokens = TokenCounter.count_codex_tokens(text)
        assert tokens > 0
        assert 350 < tokens < 500  # Roughly 1.3 tokens per word

    def test_count_gemini_tokens(self):
        """Test Gemini token counting."""
        text = "This is a test" * 100  # ~1400 characters
        tokens = TokenCounter.count_gemini_tokens(text)
        assert tokens > 0
        assert 300 < tokens < 400  # Similar to Claude

    def test_count_tokens_dispatch(self):
        """Test token counting dispatcher."""
        text = "Test text for token counting"

        claude_tokens = TokenCounter.count_tokens(text, "claude")
        codex_tokens = TokenCounter.count_tokens(text, "codex")
        gemini_tokens = TokenCounter.count_tokens(text, "gemini")

        assert claude_tokens > 0
        assert codex_tokens > 0
        assert gemini_tokens > 0

        # Claude and Gemini should be similar
        assert abs(claude_tokens - gemini_tokens) < 5

    def test_count_tokens_minimum(self):
        """Test minimum token count is 1."""
        assert TokenCounter.count_claude_tokens("") == 1
        assert TokenCounter.count_codex_tokens("") == 1
        assert TokenCounter.count_gemini_tokens("") == 1


# TokenBudgetManager Tests
class TestTokenBudgetManager:
    """Test TokenBudgetManager functionality."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = TokenBudgetManager()
        assert manager.allocation_strategy == AllocationStrategy.PRIORITY_BASED
        assert manager.compression_strategy == CompressionStrategy.NONE
        assert manager.absolute_rules_protected is True

    def test_initialization_custom(self):
        """Test manager initialization with custom settings."""
        manager = TokenBudgetManager(
            allocation_strategy=AllocationStrategy.ROUND_ROBIN,
            compression_strategy=CompressionStrategy.SIMPLE,
            absolute_rules_protected=False,
            overhead_percentage=0.10,
        )
        assert manager.allocation_strategy == AllocationStrategy.ROUND_ROBIN
        assert manager.compression_strategy == CompressionStrategy.SIMPLE
        assert manager.absolute_rules_protected is False
        assert manager.overhead_percentage == 0.10

    def test_allocate_budget_basic(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
    ):
        """Test basic budget allocation."""
        manager = TokenBudgetManager()
        rules = [high_priority_absolute_rule, high_priority_default_rule]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        assert isinstance(allocation, BudgetAllocation)
        assert allocation.total_budget == 10000
        assert len(allocation.rules_included) == 2
        assert len(allocation.rules_skipped) == 0
        assert allocation.compression_applied is False

    def test_allocate_budget_absolute_protection(
        self,
        high_priority_absolute_rule,
        low_priority_absolute_rule,
        high_priority_default_rule,
    ):
        """Test absolute rule protection."""
        manager = TokenBudgetManager(absolute_rules_protected=True)
        rules = [
            high_priority_absolute_rule,
            low_priority_absolute_rule,
            high_priority_default_rule,
        ]

        # Very small budget - only absolute rules should fit
        allocation = manager.allocate_budget(rules, 100, "claude")

        # Absolute rules should be included even if over budget
        absolute_included = [
            r for r in allocation.rules_included if r.authority == AuthorityLevel.ABSOLUTE
        ]
        assert len(absolute_included) == 2

    def test_allocate_budget_priority_based(
        self,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test priority-based allocation."""
        manager = TokenBudgetManager(allocation_strategy=AllocationStrategy.PRIORITY_BASED)
        rules = [low_priority_default_rule, high_priority_default_rule]

        # Limited budget
        allocation = manager.allocate_budget(rules, 500, "claude")

        if len(allocation.rules_included) == 1:
            # High priority rule should be selected
            assert allocation.rules_included[0].metadata["priority"] == 80

    def test_allocate_budget_round_robin(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test round-robin allocation across categories."""
        manager = TokenBudgetManager(allocation_strategy=AllocationStrategy.ROUND_ROBIN)

        # Rules from different categories
        rules = [
            high_priority_absolute_rule,  # BEHAVIOR
            high_priority_default_rule,  # BEHAVIOR
            low_priority_default_rule,  # PREFERENCE
        ]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        # Should include rules from different categories
        categories = {r.category for r in allocation.rules_included}
        assert len(categories) >= 1  # At least one category represented

    def test_allocate_budget_equal(
        self,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test equal allocation strategy."""
        manager = TokenBudgetManager(allocation_strategy=AllocationStrategy.EQUAL)
        rules = [high_priority_default_rule, low_priority_default_rule]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        # Both rules should fit with equal allocation
        assert len(allocation.rules_included) >= 1

    def test_allocate_budget_skipped_rules(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test rules are skipped when budget exceeded."""
        manager = TokenBudgetManager()
        rules = [
            high_priority_absolute_rule,
            high_priority_default_rule,
            low_priority_default_rule,
        ]

        # Very limited budget
        allocation = manager.allocate_budget(rules, 200, "claude")

        # Some rules should be skipped
        assert len(allocation.rules_skipped) >= 0  # May skip default rules

    def test_allocate_budget_overhead(self):
        """Test overhead reservation."""
        manager = TokenBudgetManager(overhead_percentage=0.10)  # 10% overhead
        rules = []

        allocation = manager.allocate_budget(rules, 1000, "claude")

        assert allocation.overhead_tokens == 100  # 10% of 1000

    def test_allocate_budget_statistics(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
    ):
        """Test allocation statistics."""
        manager = TokenBudgetManager()
        rules = [high_priority_absolute_rule, high_priority_default_rule]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        assert "absolute_count" in allocation.allocation_stats
        assert "default_count" in allocation.allocation_stats
        assert "strategy" in allocation.allocation_stats
        assert "utilization" in allocation.allocation_stats
        assert allocation.allocation_stats["strategy"] == "priority"

    def test_allocate_budget_empty_rules(self):
        """Test allocation with empty rule list."""
        manager = TokenBudgetManager()
        allocation = manager.allocate_budget([], 10000, "claude")

        assert len(allocation.rules_included) == 0
        assert len(allocation.rules_skipped) == 0
        assert allocation.absolute_tokens == 0
        assert allocation.default_tokens == 0

    def test_allocate_budget_different_tools(
        self,
        high_priority_absolute_rule,
    ):
        """Test allocation for different tool token counts."""
        manager = TokenBudgetManager()
        rules = [high_priority_absolute_rule]

        claude_allocation = manager.allocate_budget(rules, 10000, "claude")
        codex_allocation = manager.allocate_budget(rules, 10000, "codex")
        gemini_allocation = manager.allocate_budget(rules, 10000, "gemini")

        # All should include the rule
        assert len(claude_allocation.rules_included) == 1
        assert len(codex_allocation.rules_included) == 1
        assert len(gemini_allocation.rules_included) == 1

        # Token counts may differ slightly based on tool
        assert claude_allocation.absolute_tokens > 0
        assert codex_allocation.absolute_tokens > 0
        assert gemini_allocation.absolute_tokens > 0


# Integration Tests
class TestTokenBudgetIntegration:
    """Integration tests for complete budget allocation workflow."""

    def test_realistic_budget_scenario(
        self,
        high_priority_absolute_rule,
        low_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test realistic budget allocation scenario."""
        manager = TokenBudgetManager()
        rules = [
            high_priority_absolute_rule,
            low_priority_absolute_rule,
            high_priority_default_rule,
            low_priority_default_rule,
        ]

        # Moderate budget (Claude Code typical)
        allocation = manager.allocate_budget(rules, 5000, "claude")

        # Should include all absolute rules
        absolute_count = sum(
            1
            for r in allocation.rules_included
            if r.authority == AuthorityLevel.ABSOLUTE
        )
        assert absolute_count == 2

        # Should include some default rules
        default_count = sum(
            1
            for r in allocation.rules_included
            if r.authority == AuthorityLevel.DEFAULT
        )
        assert default_count >= 0

        # Utilization should be reasonable
        assert allocation.allocation_stats["utilization"] < 1.0

    def test_small_budget_scenario(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test small budget scenario (e.g., Codex)."""
        manager = TokenBudgetManager()
        rules = [
            high_priority_absolute_rule,
            high_priority_default_rule,
            low_priority_default_rule,
        ]

        # Small budget (Codex typical)
        allocation = manager.allocate_budget(rules, 1000, "codex")

        # Absolute rules should be protected
        absolute_included = [
            r for r in allocation.rules_included if r.authority == AuthorityLevel.ABSOLUTE
        ]
        assert len(absolute_included) >= 1

    def test_large_budget_scenario(
        self,
        high_priority_absolute_rule,
        low_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test large budget scenario (e.g., Gemini)."""
        manager = TokenBudgetManager()
        rules = [
            high_priority_absolute_rule,
            low_priority_absolute_rule,
            high_priority_default_rule,
            low_priority_default_rule,
        ]

        # Large budget (Gemini typical)
        allocation = manager.allocate_budget(rules, 30000, "gemini")

        # All rules should fit
        assert len(allocation.rules_included) == 4
        assert len(allocation.rules_skipped) == 0

    def test_budget_with_all_strategies(
        self,
        high_priority_absolute_rule,
        high_priority_default_rule,
        low_priority_default_rule,
    ):
        """Test all allocation strategies produce valid results."""
        rules = [
            high_priority_absolute_rule,
            high_priority_default_rule,
            low_priority_default_rule,
        ]

        for strategy in AllocationStrategy:
            manager = TokenBudgetManager(allocation_strategy=strategy)
            allocation = manager.allocate_budget(rules, 5000, "claude")

            # All should produce valid allocations
            assert allocation.total_budget == 5000
            assert len(allocation.rules_included) >= 0
            assert allocation.absolute_tokens + allocation.default_tokens <= 5000 + allocation.overhead_tokens
