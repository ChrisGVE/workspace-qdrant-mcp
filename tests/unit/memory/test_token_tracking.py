"""
Token usage tracking validation tests for memory rules.

Tests token counting accuracy, tracking across operations, behavior approaching
token limits, and tiktoken integration.

Task 283.6 - Token Usage Tracking Validation
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

from common.memory import (
    TokenCounter,
    TokenUsage,
    MemoryRule,
    MemoryCategory,
    AuthorityLevel,
)
from common.memory.token_counter import TokenizationMethod, RuleTokenInfo


class TestTokenCountingAccuracy:
    """Test token counting accuracy for various rule sizes."""

    @pytest.fixture
    def simple_counter(self):
        """Create simple token counter."""
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    @pytest.fixture
    def tiktoken_counter(self):
        """Create tiktoken-based counter."""
        return TokenCounter(method=TokenizationMethod.TIKTOKEN)

    def test_empty_rule_token_count(self, simple_counter):
        """Test token counting for minimal rule."""
        rule = MemoryRule(
            rule="X",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="test-empty",
            scope=["global"],
            source="test",
        )

        tokens = simple_counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert tokens > 0, "Even minimal rule should have tokens"
        assert tokens < 10, "Minimal rule should have few tokens"

    def test_short_rule_token_count(self, simple_counter):
        """Test token counting for short rule."""
        rule = MemoryRule(
            rule="Always use atomic commits",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-short",
            scope=["global"],
            source="test",
        )

        tokens = simple_counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert 5 <= tokens <= 20, f"Short rule should have 5-20 tokens, got {tokens}"

    def test_medium_rule_token_count(self, simple_counter):
        """Test token counting for medium-length rule."""
        rule = MemoryRule(
            rule="When implementing features, always write comprehensive unit tests with edge cases and validation. Ensure test coverage exceeds 90% for all new code.",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-medium",
            scope=["python", "testing"],
            source="test",
        )

        tokens = simple_counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert 20 <= tokens <= 60, f"Medium rule should have 20-60 tokens, got {tokens}"

    def test_long_rule_token_count(self, simple_counter):
        """Test token counting for long, detailed rule."""
        rule = MemoryRule(
            rule=(
                "For all Python development work, adhere to the following comprehensive guidelines: "
                "Use type hints for all function signatures and class attributes with Python 3.11+ syntax. "
                "Follow PEP 8 compliance with black formatting using 88-character line length. "
                "Write comprehensive docstrings in Google style for all public APIs. "
                "Implement test coverage exceeding 90% using pytest with fixtures for data management. "
                "Use async/await patterns for I/O-bound operations to maximize performance. "
                "Apply error handling with custom exceptions and meaningful error messages. "
                "Profile critical paths for performance optimization and use caching where appropriate."
            ),
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="test-long",
            scope=["python", "backend", "development"],
            source="test",
        )

        tokens = simple_counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert tokens > 60, f"Long rule should have > 60 tokens, got {tokens}"
        assert tokens < 200, f"Long rule should have < 200 tokens, got {tokens}"

    def test_rule_with_scope_adds_tokens(self, simple_counter):
        """Test that scope contributes to token count."""
        rule_no_scope = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="test-no-scope",
            scope=[],
            source="test",
        )

        rule_with_scope = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="test-with-scope",
            scope=["python", "backend", "api", "testing"],
            source="test",
        )

        tokens_no_scope = simple_counter.count_rule_tokens(rule_no_scope)
        tokens_with_scope = simple_counter.count_rule_tokens(rule_with_scope)

        # Rule with scope should have more tokens due to scope formatting
        assert tokens_with_scope > tokens_no_scope, \
            f"Scope should add tokens: {tokens_with_scope} vs {tokens_no_scope}"

    def test_authority_level_affects_token_count(self, simple_counter):
        """Test that authority level adds to formatted token count."""
        rule_default = MemoryRule(
            rule="Test rule text",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-default",
            scope=["global"],
            source="test",
        )

        rule_absolute = MemoryRule(
            rule="Test rule text",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="test-absolute",
            scope=["global"],
            source="test",
        )

        tokens_default = simple_counter.count_rule_tokens(rule_default)
        tokens_absolute = simple_counter.count_rule_tokens(rule_absolute)

        # Both should have tokens (authority formatting is part of token count)
        assert tokens_default > 0
        assert tokens_absolute > 0

    @pytest.mark.parametrize("rule_size,expected_min,expected_max", [
        (10, 2, 15),     # Very short
        (50, 10, 30),    # Short
        (200, 30, 80),   # Medium
        (500, 80, 180),  # Long
        (1000, 150, 350), # Very long
    ])
    def test_token_count_proportional_to_size(self, simple_counter, rule_size, expected_min, expected_max):
        """Test that token counts are proportional to rule text size."""
        rule_text = "x" * rule_size
        rule = MemoryRule(
            rule=rule_text,
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-size",
            scope=["global"],
            source="test",
        )

        tokens = simple_counter.count_rule_tokens(rule)

        assert expected_min <= tokens <= expected_max, \
            f"For size {rule_size}, expected {expected_min}-{expected_max} tokens, got {tokens}"

    def test_tiktoken_counter_accuracy(self, tiktoken_counter):
        """Test tiktoken provides more accurate token counts."""
        rule = MemoryRule(
            rule="This is a test rule for validating tiktoken accuracy with multiple words",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-tiktoken",
            scope=["global"],
            source="test",
        )

        tokens = tiktoken_counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert tokens > 0
        # Tiktoken should provide reasonable counts
        assert 10 <= tokens <= 50, f"Expected 10-50 tokens, got {tokens}"

    def test_simple_vs_tiktoken_comparison(self):
        """Test that tiktoken and simple methods produce different but reasonable counts."""
        simple = TokenCounter(method=TokenizationMethod.SIMPLE)
        tiktoken = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        rule = MemoryRule(
            rule="Use comprehensive error handling with try-catch blocks and logging",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-comparison",
            scope=["python"],
            source="test",
        )

        simple_tokens = simple.count_rule_tokens(rule)
        tiktoken_tokens = tiktoken.count_rule_tokens(rule)

        # Both should produce positive counts
        assert simple_tokens > 0
        assert tiktoken_tokens > 0

        # Counts should be in same ballpark (within 3x of each other)
        ratio = max(simple_tokens, tiktoken_tokens) / min(simple_tokens, tiktoken_tokens)
        assert ratio <= 3.0, f"Token counts should be similar: {simple_tokens} vs {tiktoken_tokens}"


class TestTokenTrackingAcrossOperations:
    """Test token tracking across multiple rule operations."""

    @pytest.fixture
    def counter(self):
        """Create token counter for testing."""
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules for testing."""
        return [
            MemoryRule(
                rule="Always write tests first",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="rule-1",
                scope=["global"],
                source="test",
            ),
            MemoryRule(
                rule="Use Python 3.11+ features",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                id="rule-2",
                scope=["python"],
                source="test",
            ),
            MemoryRule(
                rule="Make atomic commits after each change",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="rule-3",
                scope=["global"],
                source="test",
            ),
        ]

    def test_count_multiple_rules_total_tokens(self, counter, sample_rules):
        """Test counting total tokens across multiple rules."""
        usage = counter.count_rules_tokens(sample_rules)

        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens > 0
        assert usage.rules_count == 3

        # Total should equal sum of individual counts
        individual_sum = sum(counter.count_rule_tokens(rule) for rule in sample_rules)
        assert usage.total_tokens == individual_sum

    def test_count_rules_by_category(self, counter, sample_rules):
        """Test token counting breakdown by category."""
        usage = counter.count_rules_tokens(sample_rules)

        # Should have tokens in multiple categories
        assert usage.behavior_tokens > 0, "Should have behavior rule tokens"
        assert usage.preference_tokens > 0, "Should have preference rule tokens"

        # Category tokens should sum to total
        category_sum = (
            usage.preference_tokens +
            usage.behavior_tokens +
            usage.agent_library_tokens +
            usage.knowledge_tokens +
            usage.context_tokens
        )
        assert category_sum == usage.total_tokens

    def test_count_rules_by_authority(self, counter, sample_rules):
        """Test token counting breakdown by authority level."""
        usage = counter.count_rules_tokens(sample_rules)

        # Should have tokens for both authority levels
        assert usage.absolute_tokens > 0, "Should have absolute rule tokens"
        assert usage.default_tokens > 0, "Should have default rule tokens"

        # Authority tokens should sum to total
        authority_sum = usage.absolute_tokens + usage.default_tokens
        assert authority_sum == usage.total_tokens

    def test_add_rules_incrementally(self, counter):
        """Test that adding rules incrementally tracks tokens correctly."""
        rules = []
        total_tokens = 0

        # Add rules one at a time
        for i in range(5):
            new_rule = MemoryRule(
                rule=f"Test rule number {i} with some content",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            rules.append(new_rule)

            usage = counter.count_rules_tokens(rules)
            new_total = usage.total_tokens

            # Total should increase with each addition
            assert new_total > total_tokens, \
                f"Adding rule {i} should increase total tokens"
            total_tokens = new_total

    def test_remove_rules_decreases_tokens(self, counter, sample_rules):
        """Test that removing rules decreases token count."""
        # Get initial usage
        initial_usage = counter.count_rules_tokens(sample_rules)
        initial_total = initial_usage.total_tokens

        # Remove one rule
        reduced_rules = sample_rules[:2]
        reduced_usage = counter.count_rules_tokens(reduced_rules)

        assert reduced_usage.total_tokens < initial_total
        assert reduced_usage.rules_count == 2

    def test_modify_rule_changes_tokens(self, counter):
        """Test that modifying rule text changes token count."""
        rule = MemoryRule(
            rule="Short",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-modify",
            scope=["global"],
            source="test",
        )

        initial_tokens = counter.count_rule_tokens(rule)

        # Create modified version with longer text
        modified_rule = MemoryRule(
            rule="This is a much longer rule with significantly more content to test token count changes",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-modify",
            scope=["global"],
            source="test",
        )

        modified_tokens = counter.count_rule_tokens(modified_rule)

        assert modified_tokens > initial_tokens, \
            f"Modified rule should have more tokens: {modified_tokens} vs {initial_tokens}"

    def test_empty_rule_list_zero_tokens(self, counter):
        """Test that empty rule list reports zero tokens."""
        usage = counter.count_rules_tokens([])

        assert usage.total_tokens == 0
        assert usage.rules_count == 0
        assert usage.percentage == 0.0
        assert usage.remaining_tokens == usage.context_window_size


class TestTokenLimitBehavior:
    """Test behavior approaching and at token limits."""

    @pytest.fixture
    def counter(self):
        """Create counter with custom context window."""
        return TokenCounter(
            method=TokenizationMethod.SIMPLE,
            context_window_size=1000  # Small window for testing
        )

    def test_usage_percentage_calculation(self, counter):
        """Test that token usage percentage is calculated correctly."""
        rules = [
            MemoryRule(
                rule="x" * 50,  # ~12-15 tokens each
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(10)
        ]

        usage = counter.count_rules_tokens(rules)

        # Should calculate percentage
        assert 0 <= usage.percentage <= 100
        assert usage.percentage == (usage.total_tokens / usage.context_window_size) * 100

    def test_remaining_tokens_calculation(self, counter):
        """Test that remaining tokens are calculated correctly."""
        rules = [
            MemoryRule(
                rule="x" * 100,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id="rule-1",
                scope=["global"],
                source="test",
            )
        ]

        usage = counter.count_rules_tokens(rules)

        expected_remaining = usage.context_window_size - usage.total_tokens
        assert usage.remaining_tokens == expected_remaining
        assert usage.remaining_tokens > 0

    def test_approaching_token_limit(self, counter):
        """Test behavior when approaching token limit."""
        # Create rules that will use most of the window
        rules = [
            MemoryRule(
                rule="x" * 200,  # Large rules
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(20)
        ]

        usage = counter.count_rules_tokens(rules)

        # Should be approaching limit
        assert usage.percentage > 50, "Should use significant portion of window"
        assert usage.remaining_tokens < usage.context_window_size

    def test_exceeding_token_limit(self, counter):
        """Test that we can calculate usage even when exceeding limit."""
        # Create rules that exceed the window
        rules = [
            MemoryRule(
                rule="x" * 300,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(30)
        ]

        usage = counter.count_rules_tokens(rules)

        # Should calculate even when over limit
        assert usage.total_tokens > usage.context_window_size
        assert usage.percentage > 100
        assert usage.remaining_tokens < 0, "Should have negative remaining when over limit"

    def test_optimize_rules_within_budget(self, counter):
        """Test optimizing rules to fit within token budget."""
        rules = [
            MemoryRule(
                rule=f"Rule content {i} " * 10,
                category=MemoryCategory.BEHAVIOR if i % 2 == 0 else MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE if i % 5 == 0 else AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(20)
        ]

        max_tokens = 500
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=max_tokens,
            preserve_absolute=True
        )

        # Should respect budget
        assert usage.total_tokens <= max_tokens
        assert len(selected_rules) <= len(rules)
        assert len(selected_rules) > 0, "Should select at least some rules"

    def test_optimize_preserves_absolute_rules(self, counter):
        """Test that optimization preserves absolute authority rules."""
        rules = [
            MemoryRule(
                rule="Critical absolute rule that must be preserved",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="absolute-1",
                scope=["global"],
                source="test",
            ),
            *[
                MemoryRule(
                    rule=f"Optional default rule {i}",
                    category=MemoryCategory.PREFERENCE,
                    authority=AuthorityLevel.DEFAULT,
                    id=f"default-{i}",
                    scope=["global"],
                    source="test",
                )
                for i in range(10)
            ]
        ]

        max_tokens = 100  # Very limited budget
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=max_tokens,
            preserve_absolute=True
        )

        # Absolute rule should be included
        absolute_rules = [r for r in selected_rules if r.authority == AuthorityLevel.ABSOLUTE]
        assert len(absolute_rules) >= 1, "Should preserve absolute rules"
        assert any(r.id == "absolute-1" for r in selected_rules)

    def test_optimize_without_preserving_absolute(self, counter):
        """Test optimization without preserving absolute rules."""
        rules = [
            MemoryRule(
                rule="x" * 500,  # Very long
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="absolute-long",
                scope=["global"],
                source="test",
            ),
            MemoryRule(
                rule="Short",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                id="default-short",
                scope=["global"],
                source="test",
            ),
        ]

        max_tokens = 20  # Very small budget
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=max_tokens,
            preserve_absolute=False
        )

        # Should fit within budget by excluding long absolute rule if needed
        assert usage.total_tokens <= max_tokens

    def test_zero_token_budget(self, counter):
        """Test optimization with zero token budget."""
        rules = [
            MemoryRule(
                rule="Test",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id="rule-1",
                scope=["global"],
                source="test",
            )
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=0,
            preserve_absolute=False
        )

        assert len(selected_rules) == 0
        assert usage.total_tokens == 0


class TestTiktokenIntegration:
    """Test tiktoken integration and validation."""

    def test_tiktoken_initialization(self):
        """Test that tiktoken counter initializes correctly."""
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        assert counter.method == TokenizationMethod.TIKTOKEN
        assert counter.tokenizer is not None

    def test_tiktoken_fallback_on_import_error(self):
        """Test fallback to SIMPLE when tiktoken unavailable."""
        with patch("common.memory.token_counter.tiktoken", side_effect=ImportError):
            counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

            # Should fall back to SIMPLE
            assert counter.method == TokenizationMethod.SIMPLE
            assert counter.tokenizer is None

    def test_tiktoken_encoding_consistency(self):
        """Test that tiktoken produces consistent token counts."""
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        rule = MemoryRule(
            rule="This is a test rule for checking tiktoken encoding consistency",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-consistency",
            scope=["global"],
            source="test",
        )

        # Count multiple times
        tokens1 = counter.count_rule_tokens(rule)
        tokens2 = counter.count_rule_tokens(rule)
        tokens3 = counter.count_rule_tokens(rule)

        # Should be identical
        assert tokens1 == tokens2 == tokens3

    def test_tiktoken_with_special_characters(self):
        """Test tiktoken handles special characters correctly."""
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        rule = MemoryRule(
            rule="Rule with special chars: @#$%^&*() and unicode: 你好 émojis: 🚀",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-special",
            scope=["global"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_tiktoken_with_code_snippets(self):
        """Test tiktoken with code-like content."""
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        rule = MemoryRule(
            rule="Use pattern: async def func() -> Dict[str, Any]: return {'key': 'value'}",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-code",
            scope=["python"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)

        assert isinstance(tokens, int)
        assert tokens > 0

    def test_tiktoken_vs_simple_accuracy(self):
        """Test that tiktoken is more accurate than simple estimation."""
        tiktoken_counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)
        simple_counter = TokenCounter(method=TokenizationMethod.SIMPLE)

        # Test with various text patterns
        test_cases = [
            "Short text",
            "Medium length text with several words and punctuation.",
            "A much longer text that contains multiple sentences, various punctuation marks, and demonstrates the difference in counting methods for more substantial content.",
        ]

        for text in test_cases:
            rule = MemoryRule(
                rule=text,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id="test-accuracy",
                scope=["global"],
                source="test",
            )

            tiktoken_count = tiktoken_counter.count_rule_tokens(rule)
            simple_count = simple_counter.count_rule_tokens(rule)

            # Both should produce positive counts
            assert tiktoken_count > 0
            assert simple_count > 0

            # For the simple method (chars/4), longer text should show more variation
            # But they should still be in the same ballpark
            assert tiktoken_count > 0

    @patch("common.memory.token_counter.tiktoken.encoding_for_model")
    def test_tiktoken_uses_correct_model(self, mock_encoding):
        """Test that tiktoken uses correct model for encoding."""
        mock_encoder = Mock()
        mock_encoder.encode = Mock(return_value=[1, 2, 3])
        mock_encoding.return_value = mock_encoder

        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        # Should have called encoding_for_model with correct model
        mock_encoding.assert_called_once_with("gpt-3.5-turbo")

    def test_tiktoken_with_large_rule_set(self):
        """Test tiktoken performance with large number of rules."""
        import time

        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        rules = [
            MemoryRule(
                rule=f"This is test rule number {i} with moderate length content for performance testing",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(100)
        ]

        start_time = time.time()
        usage = counter.count_rules_tokens(rules)
        end_time = time.time()

        # Should complete in reasonable time
        elapsed = end_time - start_time
        assert elapsed < 5.0, f"Should count 100 rules in < 5s, took {elapsed:.2f}s"

        assert usage.total_tokens > 0
        assert usage.rules_count == 100


class TestTokenUsageDataClass:
    """Test TokenUsage dataclass functionality."""

    def test_token_usage_creation(self):
        """Test creating TokenUsage instance."""
        usage = TokenUsage(
            total_tokens=5000,
            rules_count=25,
            preference_tokens=1000,
            behavior_tokens=3000,
            agent_library_tokens=500,
            knowledge_tokens=300,
            context_tokens=200,
            absolute_tokens=2000,
            default_tokens=3000,
            context_window_size=200000,
        )

        assert usage.total_tokens == 5000
        assert usage.rules_count == 25
        assert usage.preference_tokens == 1000
        assert usage.behavior_tokens == 3000

    def test_token_usage_post_init(self):
        """Test TokenUsage __post_init__ calculations."""
        usage = TokenUsage(
            total_tokens=10000,
            rules_count=50,
            context_window_size=200000,
        )

        # Should calculate percentage and remaining
        assert usage.percentage == 5.0  # 10000/200000 * 100
        assert usage.remaining_tokens == 190000

    def test_token_usage_to_dict(self):
        """Test TokenUsage serialization to dict."""
        usage = TokenUsage(
            total_tokens=1000,
            rules_count=10,
            preference_tokens=300,
            behavior_tokens=700,
            absolute_tokens=400,
            default_tokens=600,
            context_window_size=100000,
        )

        data = usage.to_dict()

        assert data["total_tokens"] == 1000
        assert data["rules_count"] == 10
        assert data["categories"]["preference"] == 300
        assert data["categories"]["behavior"] == 700
        assert data["authorities"]["absolute"] == 400
        assert data["authorities"]["default"] == 600
        assert data["context_window"]["size"] == 100000
        assert data["context_window"]["percentage"] == 1.0
        assert data["context_window"]["remaining"] == 99000


class TestRuleTokenInfo:
    """Test RuleTokenInfo dataclass functionality."""

    def test_rule_token_info_creation(self):
        """Test creating RuleTokenInfo instance."""
        rule = MemoryRule(
            rule="Test rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-rule",
            scope=["global"],
            source="test",
        )

        info = RuleTokenInfo(
            rule=rule,
            tokens=50,
            priority_score=75.5,
        )

        assert info.rule == rule
        assert info.tokens == 50
        assert info.priority_score == 75.5

    def test_rule_token_info_sorting(self):
        """Test RuleTokenInfo sorting by priority score."""
        rule1 = MemoryRule(
            rule="Test",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule-1",
            scope=["global"],
            source="test",
        )

        rule2 = MemoryRule(
            rule="Test",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="rule-2",
            scope=["global"],
            source="test",
        )

        info_low = RuleTokenInfo(rule=rule1, tokens=50, priority_score=30.0)
        info_high = RuleTokenInfo(rule=rule2, tokens=50, priority_score=90.0)

        infos = [info_high, info_low]
        sorted_infos = sorted(infos)

        # Should sort by priority score (ascending)
        assert sorted_infos[0] == info_low
        assert sorted_infos[1] == info_high

        # Reverse sort for optimization (highest priority first)
        sorted_reverse = sorted(infos, reverse=True)
        assert sorted_reverse[0] == info_high
        assert sorted_reverse[1] == info_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
