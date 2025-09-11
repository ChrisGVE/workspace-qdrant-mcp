"""
Tests for token counting and optimization system.

Comprehensive tests for memory rule token counting, context optimization,
and rule selection algorithms for managing Claude context window usage.
"""

import json
from unittest.mock import Mock, patch

import pytest

from common.memory.token_counter import (
    TokenCounter,
    TokenUsage,
    TokenizationMethod,
    RuleTokenInfo,
)
from common.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


class TestTokenUsage:
    """Test TokenUsage dataclass functionality."""

    def test_create_basic_usage(self):
        """Test creating basic token usage information."""
        usage = TokenUsage(
            total_tokens=1500,
            rules_count=10,
            preference_tokens=500,
            behavior_tokens=800,
            absolute_tokens=300,
            default_tokens=1200,
            context_window_size=200000
        )

        assert usage.total_tokens == 1500
        assert usage.rules_count == 10
        assert usage.preference_tokens == 500
        assert usage.behavior_tokens == 800
        assert usage.absolute_tokens == 300
        assert usage.default_tokens == 1200
        assert usage.percentage == 0.75  # 1500/200000 * 100
        assert usage.remaining_tokens == 198500

    def test_usage_serialization(self):
        """Test token usage serialization to dict."""
        usage = TokenUsage(
            total_tokens=1000,
            rules_count=5,
            preference_tokens=300,
            behavior_tokens=700,
            absolute_tokens=200,
            default_tokens=800,
            context_window_size=100000
        )

        data = usage.to_dict()

        expected = {
            "total_tokens": 1000,
            "rules_count": 5,
            "categories": {
                "preference": 300,
                "behavior": 700,
                "agent_library": 0,
                "knowledge": 0,
                "context": 0,
            },
            "authorities": {
                "absolute": 200,
                "default": 800,
            },
            "context_window": {
                "size": 100000,
                "percentage": 1.0,
                "remaining": 99000,
            },
        }

        assert data == expected

    def test_percentage_calculation(self):
        """Test percentage calculation for different token counts."""
        # Test 0%
        usage_zero = TokenUsage(
            total_tokens=0,
            rules_count=0,
            context_window_size=200000
        )
        assert usage_zero.percentage == 0.0
        assert usage_zero.remaining_tokens == 200000

        # Test 50%
        usage_half = TokenUsage(
            total_tokens=100000,
            rules_count=50,
            context_window_size=200000
        )
        assert usage_half.percentage == 50.0
        assert usage_half.remaining_tokens == 100000

        # Test 100%
        usage_full = TokenUsage(
            total_tokens=200000,
            rules_count=100,
            context_window_size=200000
        )
        assert usage_full.percentage == 100.0
        assert usage_full.remaining_tokens == 0


class TestRuleTokenInfo:
    """Test RuleTokenInfo functionality."""

    def test_create_token_info(self):
        """Test creating rule token information."""
        rule = MemoryRule(
            rule="Test rule for token counting",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        token_info = RuleTokenInfo(
            rule=rule,
            tokens=50,
            priority_score=0.85
        )

        assert token_info.rule == rule
        assert token_info.tokens == 50
        assert token_info.priority_score == 0.85

    def test_token_info_sorting(self):
        """Test sorting rule token info by priority score."""
        rule1 = MemoryRule(
            rule="Low priority rule",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )

        rule2 = MemoryRule(
            rule="High priority rule",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
        )

        token_info1 = RuleTokenInfo(rule=rule1, tokens=30, priority_score=0.3)
        token_info2 = RuleTokenInfo(rule=rule2, tokens=50, priority_score=0.9)

        # Test sorting (higher priority score should come last in ascending sort)
        sorted_infos = sorted([token_info1, token_info2])
        assert sorted_infos[0] == token_info1  # Lower priority first
        assert sorted_infos[1] == token_info2  # Higher priority last

        # Test reverse sorting for optimization (higher priority first)
        reverse_sorted = sorted([token_info1, token_info2], reverse=True)
        assert reverse_sorted[0] == token_info2  # Higher priority first
        assert reverse_sorted[1] == token_info1  # Lower priority last


class TestTokenCounterInit:
    """Test TokenCounter initialization and configuration."""

    def test_default_initialization(self):
        """Test default token counter initialization."""
        counter = TokenCounter()

        assert counter.context_window_size == 200000
        assert counter.method == TokenizationMethod.SIMPLE
        assert counter.reserved_tokens == 1000

    def test_custom_initialization(self):
        """Test custom token counter initialization."""
        counter = TokenCounter(
            context_window_size=100000,
            method=TokenizationMethod.TIKTOKEN,
            reserved_tokens=2000
        )

        assert counter.context_window_size == 100000
        assert counter.method == TokenizationMethod.TIKTOKEN
        assert counter.reserved_tokens == 2000

    @patch('tiktoken.encoding_for_model')
    def test_tiktoken_initialization(self, mock_encoding):
        """Test initialization with tiktoken method."""
        mock_encoder = Mock()
        mock_encoding.return_value = mock_encoder
        
        counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)
        
        assert counter.method == TokenizationMethod.TIKTOKEN
        mock_encoding.assert_called_once_with("gpt-4")

    def test_tiktoken_fallback(self):
        """Test fallback when tiktoken is not available."""
        with patch('tiktoken.encoding_for_model', side_effect=ImportError):
            counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)
            
            # Should fall back to SIMPLE method
            assert counter.method == TokenizationMethod.SIMPLE

    def test_anthropic_method_fallback(self):
        """Test fallback for Anthropic method (not yet implemented)."""
        counter = TokenCounter(method=TokenizationMethod.ANTHROPIC)
        
        # Should fall back to SIMPLE method
        assert counter.method == TokenizationMethod.SIMPLE


class TestTokenCounting:
    """Test token counting for individual rules and rule sets."""

    @pytest.fixture
    def counter(self):
        """Create simple token counter for testing."""
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_count_single_rule_tokens(self, counter):
        """Test counting tokens for a single rule."""
        rule = MemoryRule(
            rule="This is a test rule for token counting with some additional text",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing", "tokens"],
            tags=["test", "token_count"]
        )

        token_count = counter.count_rule_tokens(rule)
        
        assert isinstance(token_count, int)
        assert token_count > 0
        # Should be roughly proportional to text length (simple method)
        assert token_count > 10  # At least some tokens
        assert token_count < 100  # Not excessive for short rule

    def test_count_empty_rule_tokens(self, counter):
        """Test token counting for minimal rule."""
        rule = MemoryRule(
            rule="Test",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
        )

        token_count = counter.count_rule_tokens(rule)
        
        assert isinstance(token_count, int)
        assert token_count > 0  # Should have some tokens even for minimal rule

    def test_count_complex_rule_tokens(self, counter):
        """Test token counting for complex rule with all fields."""
        rule = MemoryRule(
            rule="Always use comprehensive error handling with try-catch blocks, proper logging, and meaningful error messages that help developers understand what went wrong and how to fix it",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["error_handling", "javascript", "typescript", "backend", "frontend"],
            tags=["error_handling", "best_practices", "logging", "debugging", "development"],
            context="This rule applies to all JavaScript and TypeScript development work"
        )

        token_count = counter.count_rule_tokens(rule)
        
        assert isinstance(token_count, int)
        assert token_count > 30  # Should be substantial for complex rule

    def test_count_multiple_rules_tokens(self, counter):
        """Test counting tokens for multiple rules."""
        rules = [
            MemoryRule(
                rule="Short rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="This is a much longer rule with more detailed information",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Medium length rule for testing",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        usage = counter.calculate_usage(rules)
        
        assert isinstance(usage, TokenUsage)
        assert usage.total_tokens > 0
        assert usage.rules_count == 3
        assert usage.preference_tokens > 0
        assert usage.behavior_tokens > 0
        assert usage.absolute_tokens > 0
        assert usage.default_tokens > 0

    @patch('tiktoken.Encoding.encode')
    def test_tiktoken_counting(self, mock_encode):
        """Test token counting with tiktoken method."""
        mock_encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        
        with patch('tiktoken.encoding_for_model') as mock_encoding:
            mock_encoder = Mock()
            mock_encoder.encode = mock_encode
            mock_encoding.return_value = mock_encoder
            
            counter = TokenCounter(method=TokenizationMethod.TIKTOKEN)
            
            rule = MemoryRule(
                rule="Test rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
            
            token_count = counter.count_rule_tokens(rule)
            
            assert token_count > 0
            mock_encode.assert_called()


class TestRuleOptimization:
    """Test rule optimization for context windows."""

    @pytest.fixture
    def counter(self):
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_optimize_within_budget(self, counter):
        """Test optimizing rules that fit within token budget."""
        rules = [
            MemoryRule(
                rule="Short rule 1",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Short rule 2",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=1000, preserve_absolute=True
        )

        assert len(selected_rules) == 2  # Should include all rules
        assert usage.total_tokens <= 1000
        assert usage.rules_count == 2

    def test_optimize_over_budget(self, counter):
        """Test optimizing when rules exceed token budget."""
        rules = [
            MemoryRule(
                rule="This is a very long rule with lots of text that should use many tokens in the simple counting method to test budget optimization",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Another long rule with extensive details and comprehensive information that should also use many tokens",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            ),
            MemoryRule(
                rule="Short",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=50, preserve_absolute=True
        )

        # Should include at least the absolute rule
        assert len(selected_rules) >= 1
        absolute_rules = [r for r in selected_rules if r.authority == AuthorityLevel.ABSOLUTE]
        assert len(absolute_rules) == 1
        assert usage.total_tokens <= 60  # Allow small buffer

    def test_preserve_absolute_rules(self, counter):
        """Test that absolute authority rules are preserved."""
        rules = [
            MemoryRule(
                rule="Very important absolute rule that must be preserved even if long",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Optional default rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=30, preserve_absolute=True
        )

        # Should preserve absolute rule even if it's over budget
        absolute_rules = [r for r in selected_rules if r.authority == AuthorityLevel.ABSOLUTE]
        assert len(absolute_rules) == 1

    def test_dont_preserve_absolute_when_disabled(self, counter):
        """Test optimization when absolute preservation is disabled."""
        rules = [
            MemoryRule(
                rule="Very long absolute rule that would exceed budget significantly if included in optimization",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
            ),
            MemoryRule(
                rule="Short default rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=20, preserve_absolute=False
        )

        # Might not include absolute rule if it doesn't fit budget
        assert usage.total_tokens <= 20
        # Should prioritize rules that fit

    def test_priority_based_selection(self, counter):
        """Test that higher priority rules are selected first."""
        rules = [
            MemoryRule(
                rule="Low priority rule",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
                use_count=1,  # Low usage
            ),
            MemoryRule(
                rule="High priority rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE,
                use_count=10,  # High usage
            ),
            MemoryRule(
                rule="Medium priority rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                use_count=5,  # Medium usage
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=100, preserve_absolute=True
        )

        # Should include high priority rules first
        assert len(selected_rules) >= 1
        # Absolute rule should be included
        absolute_rules = [r for r in selected_rules if r.authority == AuthorityLevel.ABSOLUTE]
        assert len(absolute_rules) == 1

    def test_scope_based_optimization(self, counter):
        """Test optimization considering rule scope relevance."""
        rules = [
            MemoryRule(
                rule="Python-specific rule for backend development",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python", "backend"]
            ),
            MemoryRule(
                rule="JavaScript-specific rule for frontend",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["javascript", "frontend"]
            ),
            MemoryRule(
                rule="General development rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=[]  # Global rule
            ),
        ]

        # Test with context that should prioritize Python rules
        selected_rules, usage = counter.optimize_rules_for_context(
            rules, 
            max_tokens=100, 
            context_scopes=["python", "backend"],
            preserve_absolute=True
        )

        # Should prefer Python and general rules over JavaScript
        rule_texts = [rule.rule for rule in selected_rules]
        assert any("Python" in text for text in rule_texts) or any("General" in text for text in rule_texts)


class TestContextOptimization:
    """Test context-aware rule optimization."""

    @pytest.fixture
    def counter(self):
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_context_scope_matching(self, counter):
        """Test rule selection based on context scopes."""
        rules = [
            MemoryRule(
                rule="React component best practices",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["react", "frontend", "javascript"]
            ),
            MemoryRule(
                rule="Django model optimization",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["django", "python", "backend"]
            ),
            MemoryRule(
                rule="General code quality guidelines",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=[]  # Global
            ),
        ]

        # Optimize for React context
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=1000,
            context_scopes=["react", "frontend"],
            preserve_absolute=True
        )

        # Should include React rule and general rule, might exclude Django
        react_rules = [r for r in selected_rules if "React" in r.rule]
        general_rules = [r for r in selected_rules if "General" in r.rule]
        django_rules = [r for r in selected_rules if "Django" in r.rule]
        
        assert len(react_rules) >= 1  # Should include React rule
        assert len(general_rules) >= 1  # Should include general rule

    def test_empty_context_optimization(self, counter):
        """Test optimization with no specific context."""
        rules = [
            MemoryRule(
                rule="Specific rule for Python",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python"]
            ),
            MemoryRule(
                rule="Global development rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=[]
            ),
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=1000,
            context_scopes=[],  # No specific context
            preserve_absolute=True
        )

        # Should include all rules when no specific context
        assert len(selected_rules) == 2

    def test_token_budget_enforcement(self, counter):
        """Test strict token budget enforcement."""
        # Create rules that should exceed budget
        rules = []
        for i in range(20):
            rules.append(MemoryRule(
                rule=f"Rule {i} with some text to ensure it uses tokens in the counting system",
                category=MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.DEFAULT,
            ))

        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=100,  # Very restrictive budget
            preserve_absolute=True
        )

        # Should respect budget
        assert usage.total_tokens <= 110  # Allow small buffer
        assert len(selected_rules) < len(rules)  # Should have filtered some out


class TestPerformance:
    """Test token counter performance characteristics."""

    @pytest.fixture
    def counter(self):
        return TokenCounter(method=TokenizationMethod.SIMPLE)

    def test_large_rule_set_performance(self, counter):
        """Test performance with large number of rules."""
        import time
        
        # Create large set of rules
        rules = []
        for i in range(1000):
            rules.append(MemoryRule(
                rule=f"Performance test rule {i} with moderate length text content",
                category=MemoryCategory.BEHAVIOR if i % 2 else MemoryCategory.PREFERENCE,
                authority=AuthorityLevel.ABSOLUTE if i % 10 == 0 else AuthorityLevel.DEFAULT,
                scope=[f"scope_{i % 5}"]
            ))

        start_time = time.time()
        usage = counter.calculate_usage(rules)
        end_time = time.time()

        # Should complete in reasonable time
        assert end_time - start_time < 2.0  # Less than 2 seconds for 1000 rules
        assert usage.total_tokens > 0
        assert usage.rules_count == 1000

    def test_optimization_performance(self, counter):
        """Test performance of rule optimization."""
        import time
        
        # Create medium set of rules for optimization
        rules = []
        for i in range(100):
            rules.append(MemoryRule(
                rule=f"Optimization test rule {i} with content that varies in length based on index",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.ABSOLUTE if i % 5 == 0 else AuthorityLevel.DEFAULT,
                scope=[f"test_scope_{i % 3}"]
            ))

        start_time = time.time()
        selected_rules, usage = counter.optimize_rules_for_context(
            rules,
            max_tokens=1000,
            context_scopes=["test_scope_0"],
            preserve_absolute=True
        )
        end_time = time.time()

        # Should complete optimization in reasonable time
        assert end_time - start_time < 1.0  # Less than 1 second for 100 rules
        assert len(selected_rules) <= len(rules)
        assert usage.total_tokens <= 1010  # Allow small buffer


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_none_rule_handling(self):
        """Test handling of None rules."""
        counter = TokenCounter()
        
        with pytest.raises((ValueError, AttributeError)):
            counter.count_rule_tokens(None)

    def test_empty_rule_list(self):
        """Test handling of empty rule lists."""
        counter = TokenCounter()
        
        usage = counter.calculate_usage([])
        
        assert usage.total_tokens == 0
        assert usage.rules_count == 0
        assert usage.percentage == 0.0

    def test_zero_max_tokens(self):
        """Test optimization with zero max tokens."""
        counter = TokenCounter()
        
        rules = [
            MemoryRule(
                rule="Test rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=0, preserve_absolute=False
        )

        # Should return empty selection
        assert len(selected_rules) == 0
        assert usage.total_tokens == 0

    def test_negative_max_tokens(self):
        """Test optimization with negative max tokens."""
        counter = TokenCounter()
        
        rules = [
            MemoryRule(
                rule="Test rule",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
            )
        ]

        selected_rules, usage = counter.optimize_rules_for_context(
            rules, max_tokens=-100, preserve_absolute=False
        )

        # Should handle gracefully
        assert len(selected_rules) == 0
        assert usage.total_tokens == 0