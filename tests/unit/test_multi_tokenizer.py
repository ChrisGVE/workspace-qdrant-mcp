"""
Unit tests for multi-tokenizer token counting system.

Tests TokenizerFactory, enhanced TokenCounter with tiktoken/transformers,
and backward compatibility with existing tests.
"""

from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.token_budget import (
    AllocationStrategy,
    TokenBudgetManager,
    TokenCounter,
    TokenizerFactory,
    TokenizerType,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


# Test fixtures
@pytest.fixture
def sample_text():
    """Sample text for token counting."""
    return "This is a test sentence for token counting with multiple words."


@pytest.fixture
def long_text():
    """Long text for stress testing."""
    return " ".join(["word"] * 1000)


@pytest.fixture
def sample_rule():
    """Create a sample memory rule."""
    return MemoryRule(
        id="rule_test",
        name="test_rule",
        rule="Always write comprehensive unit tests for new features",
        category=MemoryCategory.BEHAVIOR,
        authority=AuthorityLevel.DEFAULT,
        scope=["python", "testing"],
        source="user_explicit",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        conditions=None,
        replaces=[],
        metadata={"priority": 80},
    )


# TokenizerFactory Tests
class TestTokenizerFactory:
    """Test TokenizerFactory functionality."""

    def test_check_tiktoken_availability(self):
        """Test tiktoken availability check."""
        available = TokenizerFactory.is_tiktoken_available()
        # Should be bool (True if installed, False otherwise)
        assert isinstance(available, bool)

    def test_check_transformers_availability(self):
        """Test transformers availability check."""
        available = TokenizerFactory.is_transformers_available()
        # Should be bool (True if installed, False otherwise)
        assert isinstance(available, bool)

    def test_get_tiktoken_encoding_when_available(self):
        """Test getting tiktoken encoding when available."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        encoding = TokenizerFactory.get_tiktoken_encoding("cl100k_base")
        assert encoding is not None
        # Should be able to encode text
        tokens = encoding.encode("test")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_get_tiktoken_encoding_when_unavailable(self):
        """Test error when tiktoken not available."""
        if TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken is installed")

        with pytest.raises(ImportError, match="tiktoken library not available"):
            TokenizerFactory.get_tiktoken_encoding("cl100k_base")

    def test_get_transformers_tokenizer_when_available(self):
        """Test getting transformers tokenizer when available."""
        if not TokenizerFactory.is_transformers_available():
            pytest.skip("transformers not installed")

        # Use a small, fast model for testing
        tokenizer = TokenizerFactory.get_transformers_tokenizer("google/gemma-2b")
        assert tokenizer is not None
        # Should be able to encode text
        tokens = tokenizer.encode("test")
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_get_transformers_tokenizer_when_unavailable(self):
        """Test error when transformers not available."""
        if TokenizerFactory.is_transformers_available():
            pytest.skip("transformers is installed")

        with pytest.raises(ImportError, match="transformers library not available"):
            TokenizerFactory.get_transformers_tokenizer("google/gemma-2b")

    def test_tokenizer_caching(self):
        """Test tokenizer instance caching."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        # Clear cache first
        TokenizerFactory.clear_cache()

        # Get encoding twice
        enc1 = TokenizerFactory.get_tiktoken_encoding("cl100k_base")
        enc2 = TokenizerFactory.get_tiktoken_encoding("cl100k_base")

        # Should be same instance (cached)
        assert enc1 is enc2

    def test_clear_cache(self):
        """Test cache clearing."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        # Get encoding to populate cache
        TokenizerFactory.get_tiktoken_encoding("cl100k_base")
        cache_size_before = len(TokenizerFactory._cache)
        assert cache_size_before > 0

        # Clear cache
        TokenizerFactory.clear_cache()
        cache_size_after = len(TokenizerFactory._cache)

        # Cache should be empty
        assert cache_size_after == 0

        # Availability flags should be reset
        assert TokenizerFactory._tiktoken_available is None

    def test_preferred_tokenizer_type_codex(self):
        """Test preferred tokenizer for codex."""
        tokenizer_type = TokenizerFactory.get_preferred_tokenizer_type("codex")

        if TokenizerFactory.is_tiktoken_available():
            assert tokenizer_type == TokenizerType.TIKTOKEN
        else:
            assert tokenizer_type == TokenizerType.ESTIMATION

    def test_preferred_tokenizer_type_gemini(self):
        """Test preferred tokenizer for gemini."""
        tokenizer_type = TokenizerFactory.get_preferred_tokenizer_type("gemini")

        if TokenizerFactory.is_transformers_available():
            assert tokenizer_type == TokenizerType.TRANSFORMERS
        else:
            assert tokenizer_type == TokenizerType.ESTIMATION

    def test_preferred_tokenizer_type_claude(self):
        """Test preferred tokenizer for claude."""
        tokenizer_type = TokenizerFactory.get_preferred_tokenizer_type("claude")

        if TokenizerFactory.is_tiktoken_available():
            assert tokenizer_type == TokenizerType.TIKTOKEN
        else:
            assert tokenizer_type == TokenizerType.ESTIMATION


# Enhanced TokenCounter Tests
class TestEnhancedTokenCounter:
    """Test enhanced TokenCounter with multi-tokenizer support."""

    def test_count_claude_tokens_with_tokenizer(self, sample_text):
        """Test Claude token counting with actual tokenizer."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        tokens_accurate = TokenCounter.count_claude_tokens(
            sample_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_claude_tokens(
            sample_text, use_tokenizer=False
        )

        assert tokens_accurate > 0
        assert tokens_estimate > 0
        # Accurate count should be different from estimation
        # (but might occasionally match by chance)
        assert isinstance(tokens_accurate, int)

    def test_count_claude_tokens_estimation_fallback(self, sample_text):
        """Test Claude token estimation fallback."""
        tokens = TokenCounter.count_claude_tokens(sample_text, use_tokenizer=False)
        expected = len(sample_text) // 4
        assert tokens == max(1, expected)

    def test_count_codex_tokens_with_tokenizer(self, sample_text):
        """Test Codex token counting with actual tokenizer."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        tokens_accurate = TokenCounter.count_codex_tokens(
            sample_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_codex_tokens(
            sample_text, use_tokenizer=False
        )

        assert tokens_accurate > 0
        assert tokens_estimate > 0
        assert isinstance(tokens_accurate, int)

    def test_count_codex_tokens_estimation_fallback(self, sample_text):
        """Test Codex token estimation fallback."""
        tokens = TokenCounter.count_codex_tokens(sample_text, use_tokenizer=False)
        words = len(sample_text.split())
        expected = int(words * 1.3)
        assert tokens == max(1, expected)

    def test_count_gemini_tokens_with_tokenizer(self, sample_text):
        """Test Gemini token counting with actual tokenizer."""
        if not TokenizerFactory.is_transformers_available():
            pytest.skip("transformers not installed")

        tokens_accurate = TokenCounter.count_gemini_tokens(
            sample_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_gemini_tokens(
            sample_text, use_tokenizer=False
        )

        assert tokens_accurate > 0
        assert tokens_estimate > 0
        assert isinstance(tokens_accurate, int)

    def test_count_gemini_tokens_estimation_fallback(self, sample_text):
        """Test Gemini token estimation fallback."""
        tokens = TokenCounter.count_gemini_tokens(sample_text, use_tokenizer=False)
        expected = len(sample_text) // 4
        assert tokens == max(1, expected)

    def test_count_tokens_dispatch(self, sample_text):
        """Test token counting dispatcher with all tools."""
        claude_tokens = TokenCounter.count_tokens(sample_text, "claude")
        codex_tokens = TokenCounter.count_tokens(sample_text, "codex")
        gemini_tokens = TokenCounter.count_tokens(sample_text, "gemini")

        assert claude_tokens > 0
        assert codex_tokens > 0
        assert gemini_tokens > 0

    def test_count_tokens_with_model_gpt(self, sample_text):
        """Test explicit model-based counting for GPT."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        tokens = TokenCounter.count_tokens_with_model(sample_text, "gpt-4")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_with_model_gemini(self, sample_text):
        """Test explicit model-based counting for Gemini."""
        if not TokenizerFactory.is_transformers_available():
            pytest.skip("transformers not installed")

        tokens = TokenCounter.count_tokens_with_model(sample_text, "gemini-pro")
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_with_model_fallback(self, sample_text):
        """Test model-based counting fallback to estimation."""
        tokens = TokenCounter.count_tokens_with_model(
            sample_text, "unknown-model", tokenizer_type=TokenizerType.ESTIMATION
        )
        expected = len(sample_text) // 4
        assert tokens == max(1, expected)

    def test_count_tokens_empty_string(self):
        """Test minimum token count for empty string."""
        # With tokenizer or without, should always return at least 1
        assert TokenCounter.count_claude_tokens("", use_tokenizer=False) == 1
        assert TokenCounter.count_codex_tokens("", use_tokenizer=False) == 1
        assert TokenCounter.count_gemini_tokens("", use_tokenizer=False) == 1

        # Even with tokenizer, should return 1
        if TokenizerFactory.is_tiktoken_available():
            assert TokenCounter.count_claude_tokens("", use_tokenizer=True) == 1
            assert TokenCounter.count_codex_tokens("", use_tokenizer=True) == 1

    def test_count_tokens_long_text(self, long_text):
        """Test token counting with long text."""
        tokens = TokenCounter.count_claude_tokens(long_text)
        assert tokens > 100  # Should count many tokens


# Backward Compatibility Tests
class TestBackwardCompatibility:
    """Test backward compatibility with existing code."""

    def test_existing_api_unchanged(self, sample_text):
        """Test that existing API still works."""
        # Original API (without use_tokenizer parameter)
        claude_tokens = TokenCounter.count_claude_tokens(sample_text)
        codex_tokens = TokenCounter.count_codex_tokens(sample_text)
        gemini_tokens = TokenCounter.count_gemini_tokens(sample_text)

        assert claude_tokens > 0
        assert codex_tokens > 0
        assert gemini_tokens > 0

    def test_count_tokens_original_signature(self, sample_text):
        """Test original count_tokens signature."""
        tokens = TokenCounter.count_tokens(sample_text, "claude")
        assert tokens > 0

    def test_token_budget_manager_unchanged(self, sample_rule):
        """Test TokenBudgetManager works with enhanced counter."""
        manager = TokenBudgetManager()
        rules = [sample_rule]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        assert len(allocation.rules_included) == 1
        assert allocation.total_budget == 10000
        # Should have tokenizer_used in stats
        assert "tokenizer_used" in allocation.allocation_stats

    def test_token_budget_manager_with_accurate_counting_disabled(
        self, sample_rule
    ):
        """Test TokenBudgetManager with accurate counting disabled."""
        manager = TokenBudgetManager(use_accurate_counting=False)
        rules = [sample_rule]

        allocation = manager.allocate_budget(rules, 10000, "claude")

        assert len(allocation.rules_included) == 1
        assert allocation.allocation_stats["tokenizer_used"] == "estimation"


# Accuracy Comparison Tests
class TestAccuracyComparison:
    """Compare estimation vs actual tokenizer accuracy."""

    def test_accuracy_improvement_claude(self, long_text):
        """Test accuracy improvement for Claude."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed - cannot compare accuracy")

        tokens_accurate = TokenCounter.count_claude_tokens(
            long_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_claude_tokens(
            long_text, use_tokenizer=False
        )

        # Both should be positive
        assert tokens_accurate > 0
        assert tokens_estimate > 0

        # Calculate accuracy difference
        diff_percentage = abs(tokens_accurate - tokens_estimate) / tokens_accurate

        # Log for analysis (not a strict assertion)
        print("\nClaude accuracy test:")
        print(f"  Accurate: {tokens_accurate}")
        print(f"  Estimate: {tokens_estimate}")
        print(f"  Diff: {diff_percentage:.2%}")

    def test_accuracy_improvement_codex(self, long_text):
        """Test accuracy improvement for Codex."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed - cannot compare accuracy")

        tokens_accurate = TokenCounter.count_codex_tokens(
            long_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_codex_tokens(
            long_text, use_tokenizer=False
        )

        # Both should be positive
        assert tokens_accurate > 0
        assert tokens_estimate > 0

        # Calculate accuracy difference
        diff_percentage = abs(tokens_accurate - tokens_estimate) / tokens_accurate

        # Log for analysis
        print("\nCodex accuracy test:")
        print(f"  Accurate: {tokens_accurate}")
        print(f"  Estimate: {tokens_estimate}")
        print(f"  Diff: {diff_percentage:.2%}")

    def test_accuracy_improvement_gemini(self, long_text):
        """Test accuracy improvement for Gemini."""
        if not TokenizerFactory.is_transformers_available():
            pytest.skip("transformers not installed - cannot compare accuracy")

        tokens_accurate = TokenCounter.count_gemini_tokens(
            long_text, use_tokenizer=True
        )
        tokens_estimate = TokenCounter.count_gemini_tokens(
            long_text, use_tokenizer=False
        )

        # Both should be positive
        assert tokens_accurate > 0
        assert tokens_estimate > 0

        # Calculate accuracy difference
        diff_percentage = abs(tokens_accurate - tokens_estimate) / tokens_accurate

        # Log for analysis
        print("\nGemini accuracy test:")
        print(f"  Accurate: {tokens_accurate}")
        print(f"  Estimate: {tokens_estimate}")
        print(f"  Diff: {diff_percentage:.2%}")


# Integration Tests
class TestMultiTokenizerIntegration:
    """Integration tests for multi-tokenizer system."""

    def test_budget_allocation_with_tiktoken(self, sample_rule):
        """Test budget allocation using tiktoken."""
        if not TokenizerFactory.is_tiktoken_available():
            pytest.skip("tiktoken not installed")

        manager = TokenBudgetManager(use_accurate_counting=True)
        rules = [sample_rule] * 5

        allocation = manager.allocate_budget(rules, 1000, "codex")

        assert len(allocation.rules_included) <= 5
        assert allocation.allocation_stats["tokenizer_used"] == "tiktoken"

    def test_budget_allocation_with_transformers(self, sample_rule):
        """Test budget allocation using transformers."""
        if not TokenizerFactory.is_transformers_available():
            pytest.skip("transformers not installed")

        manager = TokenBudgetManager(use_accurate_counting=True)
        rules = [sample_rule] * 5

        allocation = manager.allocate_budget(rules, 1000, "gemini")

        assert len(allocation.rules_included) <= 5
        assert allocation.allocation_stats["tokenizer_used"] == "transformers"

    def test_budget_allocation_estimation_fallback(self, sample_rule):
        """Test budget allocation falls back to estimation."""
        manager = TokenBudgetManager(use_accurate_counting=True)
        rules = [sample_rule] * 5

        # Use unknown tool (should fall back to estimation)
        allocation = manager.allocate_budget(rules, 1000, "unknown_tool")

        assert len(allocation.rules_included) <= 5
        assert allocation.allocation_stats["tokenizer_used"] == "estimation"

    def test_multiple_tools_same_budget(self, sample_rule):
        """Test allocation for multiple tools with same budget."""
        manager = TokenBudgetManager(use_accurate_counting=True)
        rules = [sample_rule] * 3

        claude_alloc = manager.allocate_budget(rules, 5000, "claude")
        codex_alloc = manager.allocate_budget(rules, 5000, "codex")
        gemini_alloc = manager.allocate_budget(rules, 5000, "gemini")

        # All should include rules
        assert len(claude_alloc.rules_included) > 0
        assert len(codex_alloc.rules_included) > 0
        assert len(gemini_alloc.rules_included) > 0

    def test_strategy_preservation(self, sample_rule):
        """Test that allocation strategy is preserved."""
        manager = TokenBudgetManager(
            allocation_strategy=AllocationStrategy.ROUND_ROBIN,
            use_accurate_counting=True,
        )
        rules = [sample_rule] * 3

        allocation = manager.allocate_budget(rules, 5000, "claude")

        assert allocation.allocation_stats["strategy"] == "round_robin"
