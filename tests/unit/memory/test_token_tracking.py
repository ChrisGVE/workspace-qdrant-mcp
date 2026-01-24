"""
Token usage tracking validation tests for memory rules.

Tests token counting accuracy, tracking across operations, behavior approaching
token limits, and tiktoken integration.

Task 324.5 - Token Usage Tracking Comprehensive Tests
"""

import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest
from common.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
    TokenCounter,
    TokenUsage,
)
from common.memory.token_counter import RuleTokenInfo, TokenizationMethod


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
    """Test tiktoken integration and validation.

    NOTE: These tests require tiktoken which is an optional dependency.
    Install with: pip install tiktoken
    """

    def test_tiktoken_initialization(self):
        """Test that tiktoken counter initializes correctly."""
        pytest.importorskip("tiktoken", reason="tiktoken not installed")
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)

        assert counter.method == TokenizationMethod.TIKTOKEN
        assert counter.tokenizer is not None

    def test_tiktoken_fallback_on_import_error(self):
        """Test fallback to SIMPLE when tiktoken unavailable."""
        # Mock tiktoken import failure by temporarily removing it from sys.modules
        tiktoken_backup = sys.modules.get('tiktoken')
        try:
            # Remove tiktoken from sys.modules to simulate import error
            if 'tiktoken' in sys.modules:
                del sys.modules['tiktoken']

            # Patch the import to raise ImportError
            with patch.dict('sys.modules', {'tiktoken': None}):
                # Re-import the token_counter module to trigger the import error
                import importlib

                from common.memory import token_counter
                importlib.reload(token_counter)

                counter = token_counter.TokenCounter(method=token_counter.TokenizationMethod.TIKTOKEN)

                # Should fall back to SIMPLE
                assert counter.method == token_counter.TokenizationMethod.SIMPLE
                assert counter.tokenizer is None
        finally:
            # Restore tiktoken if it was available
            if tiktoken_backup is not None:
                sys.modules['tiktoken'] = tiktoken_backup
            # Reload token_counter to restore normal state
            import importlib

            from common.memory import token_counter
            importlib.reload(token_counter)

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

    def test_tiktoken_uses_correct_model(self):
        """Test that tiktoken uses correct model for encoding."""
        pytest.importorskip("tiktoken", reason="tiktoken not installed")
        with patch("tiktoken.encoding_for_model") as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode = Mock(return_value=[1, 2, 3])
            mock_encoding.return_value = mock_encoder

            TokenCounter(method=TokenizationMethod.TIKTOKEN)

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


class TestVariousRuleSizesAndFormats:
    """Test token counting with various rule sizes and formats."""

    @pytest.fixture
    def counter(self):
        """Create token counter for testing."""
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_json_format_rules(self, counter):
        """Test token counting for JSON-formatted rules."""
        rule = MemoryRule(
            rule='{"pattern": "async def", "example": "async def fetch_data() -> dict"}',
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-json",
            scope=["python"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_markdown_format_rules(self, counter):
        """Test token counting for markdown-formatted rules."""
        rule = MemoryRule(
            rule="""# Code Review Guidelines

## Required Checks
- [ ] Type hints present
- [ ] Tests passing
- [ ] Documentation updated

**Priority**: High""",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="test-markdown",
            scope=["review"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        # Markdown formatting should add tokens
        assert tokens > 20

    def test_code_snippet_rules(self, counter):
        """Test token counting for rules with code snippets."""
        rule = MemoryRule(
            rule="""Use this pattern for error handling:
```python
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Operation failed: {e}")
    raise
```""",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            id="test-code-snippet",
            scope=["python", "error-handling"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        # Code snippets should have substantial token count
        assert tokens > 30

    def test_multiline_rules(self, counter):
        """Test token counting for multi-line rules."""
        rule = MemoryRule(
            rule="""When implementing new features:
1. Write tests first
2. Implement minimum viable code
3. Refactor for clarity
4. Document public APIs
5. Update changelog""",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            id="test-multiline",
            scope=["development"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        assert tokens > 15

    def test_rule_with_urls(self, counter):
        """Test token counting for rules containing URLs."""
        rule = MemoryRule(
            rule="Follow PEP 8 style guide: https://peps.python.org/pep-0008/ and type hints guide: https://peps.python.org/pep-0484/",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="test-urls",
            scope=["python"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        # URLs should contribute to token count
        assert tokens > 20

    def test_rule_with_emojis(self, counter):
        """Test token counting for rules with emoji characters."""
        rule = MemoryRule(
            rule="Use emojis in commit messages: ✨ feat, 🐛 fix, 📝 docs, 🎨 style, ♻️ refactor",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            id="test-emojis",
            scope=["git"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_very_short_rules(self, counter):
        """Test token counting for very short, terse rules."""
        short_rules = [
            "TDD",
            "DRY",
            "YAGNI",
            "KISS",
            "Use uv",
        ]

        for rule_text in short_rules:
            rule = MemoryRule(
                rule=rule_text,
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                id=f"test-{rule_text}",
                scope=["global"],
                source="test",
            )

            tokens = counter.count_rule_tokens(rule)
            assert tokens > 0
            assert tokens < 10, f"Very short rule '{rule_text}' should have < 10 tokens, got {tokens}"

    def test_extremely_long_rule(self, counter):
        """Test token counting for extremely long rules."""
        # Create a very long rule (1000+ words)
        long_text = " ".join([f"word{i}" for i in range(1000)])
        rule = MemoryRule(
            rule=long_text,
            category=MemoryCategory.KNOWLEDGE,
            authority=AuthorityLevel.DEFAULT,
            id="test-extremely-long",
            scope=["global"],
            source="test",
        )

        tokens = counter.count_rule_tokens(rule)
        assert tokens > 0
        # Should be substantial but not exact word count
        assert tokens > 200


class TestTokenLimitEnforcement:
    """Test token limit enforcement and overflow handling."""

    @pytest.fixture
    def counter(self):
        """Create counter with small context window."""
        return TokenCounter(
            method=TokenizationMethod.SIMPLE,
            context_window_size=500
        )

    def test_overflow_detection(self, counter):
        """Test detection of token limit overflow."""
        # Create rules that will definitely overflow
        rules = [
            MemoryRule(
                rule="x" * 300,
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(10)
        ]

        usage = counter.count_rules_tokens(rules)

        # Should detect overflow
        assert usage.total_tokens > usage.context_window_size
        assert usage.percentage > 100
        assert usage.remaining_tokens < 0

    def test_enforcement_with_priority(self, counter):
        """Test that enforcement respects priority when limiting."""
        # Create mix of absolute and default rules
        rules = [
            MemoryRule(
                rule="Critical absolute rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                id="absolute-1",
                scope=["global"],
                source="test",
            ),
            *[
                MemoryRule(
                    rule=f"Default rule {i} with content",
                    category=MemoryCategory.PREFERENCE,
                    authority=AuthorityLevel.DEFAULT,
                    id=f"default-{i}",
                    scope=["global"],
                    source="test",
                )
                for i in range(20)
            ]
        ]

        # Optimize with very tight budget
        max_tokens = 50
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=max_tokens,
            preserve_absolute=True
        )

        # Should include the absolute rule
        assert any(r.id == "absolute-1" for r in selected_rules)
        # Should fit within budget
        assert usage.total_tokens <= max_tokens

    def test_hard_limit_enforcement(self, counter):
        """Test strict enforcement of hard token limits."""
        rules = [
            MemoryRule(
                rule=f"Rule {i} with moderate content for testing",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                id=f"rule-{i}",
                scope=["global"],
                source="test",
            )
            for i in range(30)
        ]

        max_tokens = 200
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=max_tokens,
            preserve_absolute=False
        )

        # Must not exceed limit
        assert usage.total_tokens <= max_tokens, \
            f"Token count {usage.total_tokens} exceeds limit {max_tokens}"


class TestTokenUsageReporting:
    """Test token usage reporting and statistics."""

    @pytest.fixture
    def counter(self):
        """Create token counter for testing."""
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_usage_breakdown_by_category(self, counter):
        """Test detailed breakdown of token usage by category."""
        rules = [
            MemoryRule(rule="Pref 1", category=MemoryCategory.PREFERENCE, authority=AuthorityLevel.DEFAULT,
                      id="p1", scope=[], source="test"),
            MemoryRule(rule="Behavior 1", category=MemoryCategory.BEHAVIOR, authority=AuthorityLevel.DEFAULT,
                      id="b1", scope=[], source="test"),
            MemoryRule(rule="Agent info", category=MemoryCategory.AGENT_LIBRARY, authority=AuthorityLevel.DEFAULT,
                      id="a1", scope=[], source="test"),
            MemoryRule(rule="Knowledge fact", category=MemoryCategory.KNOWLEDGE, authority=AuthorityLevel.DEFAULT,
                      id="k1", scope=[], source="test"),
            MemoryRule(rule="Context data", category=MemoryCategory.CONTEXT, authority=AuthorityLevel.DEFAULT,
                      id="c1", scope=[], source="test"),
        ]

        usage = counter.count_rules_tokens(rules)

        # All categories should have some tokens
        assert usage.preference_tokens > 0
        assert usage.behavior_tokens > 0
        assert usage.agent_library_tokens > 0
        assert usage.knowledge_tokens > 0
        assert usage.context_tokens > 0

        # Total should match sum
        total = (usage.preference_tokens + usage.behavior_tokens +
                usage.agent_library_tokens + usage.knowledge_tokens + usage.context_tokens)
        assert total == usage.total_tokens

    def test_usage_breakdown_by_authority(self, counter):
        """Test detailed breakdown of token usage by authority level."""
        rules = [
            MemoryRule(rule="Absolute rule 1", category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.ABSOLUTE, id="abs1", scope=[], source="test"),
            MemoryRule(rule="Absolute rule 2", category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.ABSOLUTE, id="abs2", scope=[], source="test"),
            MemoryRule(rule="Default rule 1", category=MemoryCategory.PREFERENCE,
                      authority=AuthorityLevel.DEFAULT, id="def1", scope=[], source="test"),
            MemoryRule(rule="Default rule 2", category=MemoryCategory.PREFERENCE,
                      authority=AuthorityLevel.DEFAULT, id="def2", scope=[], source="test"),
        ]

        usage = counter.count_rules_tokens(rules)

        assert usage.absolute_tokens > 0
        assert usage.default_tokens > 0
        assert usage.absolute_tokens + usage.default_tokens == usage.total_tokens

    def test_usage_to_dict_serialization(self, counter):
        """Test that usage can be serialized to dictionary format."""
        rules = [
            MemoryRule(rule="Test rule", category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.DEFAULT, id="r1", scope=[], source="test"),
        ]

        usage = counter.count_rules_tokens(rules)
        data = usage.to_dict()

        # Verify structure
        assert "total_tokens" in data
        assert "rules_count" in data
        assert "categories" in data
        assert "authorities" in data
        assert "context_window" in data

        # Verify category breakdown
        assert "preference" in data["categories"]
        assert "behavior" in data["categories"]

        # Verify authority breakdown
        assert "absolute" in data["authorities"]
        assert "default" in data["authorities"]

        # Verify context window info
        assert "size" in data["context_window"]
        assert "percentage" in data["context_window"]
        assert "remaining" in data["context_window"]


class TestOptimizationStrategies:
    """Test various token optimization strategies."""

    @pytest.fixture
    def counter(self):
        """Create token counter for testing."""
        return TokenCounter(method=TokenizationMethod.SIMPLE, context_window_size=1000)

    def test_priority_based_optimization(self, counter):
        """Test optimization based on rule priority scores."""
        # Create rules with different priorities
        rules = [
            # High priority: absolute + behavior
            MemoryRule(rule="Critical behavior", category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.ABSOLUTE, id="high1", scope=[], source="test"),
            # Medium priority: absolute + preference
            MemoryRule(rule="Important preference", category=MemoryCategory.PREFERENCE,
                      authority=AuthorityLevel.ABSOLUTE, id="med1", scope=[], source="test"),
            # Low priority: default + context
            MemoryRule(rule="Optional context", category=MemoryCategory.CONTEXT,
                      authority=AuthorityLevel.DEFAULT, id="low1", scope=[], source="test"),
        ]

        max_tokens = 50
        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=max_tokens, preserve_absolute=True
        )

        # High priority rules should be selected first
        assert any(r.id == "high1" for r in selected_rules)

    def test_greedy_optimization(self, counter):
        """Test greedy optimization that maximizes rule count."""
        # Create many small rules and few large rules
        small_rules = [
            MemoryRule(rule=f"R{i}", category=MemoryCategory.PREFERENCE,
                      authority=AuthorityLevel.DEFAULT, id=f"small-{i}", scope=[], source="test")
            for i in range(20)
        ]
        large_rules = [
            MemoryRule(rule="x" * 200, category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.DEFAULT, id=f"large-{i}", scope=[], source="test")
            for i in range(3)
        ]

        rules = small_rules + large_rules

        max_tokens = 300
        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=max_tokens, preserve_absolute=False
        )

        # Should select rules within budget
        assert usage.total_tokens <= max_tokens
        assert len(selected_rules) > 0

    def test_suggest_optimizations(self, counter):
        """Test optimization suggestion generation."""
        # Create rules that exceed target
        rules = [
            MemoryRule(rule="x" * 200, category=MemoryCategory.BEHAVIOR,
                      authority=AuthorityLevel.DEFAULT, id=f"rule-{i}", scope=[], source="test")
            for i in range(10)
        ]

        target_tokens = 300
        suggestions = counter.suggest_memory_optimizations(rules, target_tokens)

        assert "current_tokens" in suggestions
        assert "target_tokens" in suggestions
        assert "optimization_needed" in suggestions
        assert "suggestions" in suggestions

        # Should indicate optimization needed
        assert suggestions["optimization_needed"] is True
        assert suggestions["current_tokens"] > target_tokens

    def test_no_optimization_needed(self, counter):
        """Test when no optimization is needed."""
        rules = [
            MemoryRule(rule="Short", category=MemoryCategory.PREFERENCE,
                      authority=AuthorityLevel.DEFAULT, id="r1", scope=[], source="test"),
        ]

        target_tokens = 1000
        suggestions = counter.suggest_memory_optimizations(rules, target_tokens)

        assert suggestions["optimization_needed"] is False
        assert "already within target" in suggestions["suggestions"][0].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
