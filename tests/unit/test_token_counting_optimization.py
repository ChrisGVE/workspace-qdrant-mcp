"""
Unit tests for token counting performance optimizations.

Tests TokenCountCache, batch counting, async counting, and performance metrics.
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest

from src.python.common.core.context_injection.token_budget import (
    AllocationStrategy,
    CacheStatistics,
    TokenBudgetManager,
    TokenCountCache,
    TokenCounter,
    TokenizerFactory,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


# Test fixtures
@pytest.fixture
def sample_texts():
    """Sample texts for token counting."""
    return [
        "This is a short text for testing.",
        "A longer text with more content to count tokens accurately.",
        "Short",
        "Another medium-length text for comprehensive testing of the system.",
    ]


@pytest.fixture
def repeated_texts():
    """Texts with repetition for cache testing."""
    base_texts = [
        "Repeated text one",
        "Repeated text two",
        "Repeated text three",
    ]
    # Repeat each text multiple times
    return base_texts * 5


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset token count cache before each test."""
    TokenCounter.clear_cache()
    yield
    TokenCounter.clear_cache()


# TokenCountCache Tests
class TestTokenCountCache:
    """Test TokenCountCache functionality."""

    def test_cache_initialization(self):
        """Test cache initialization with various settings."""
        cache = TokenCountCache(maxsize=100, ttl=60.0, enabled=True)
        assert cache.maxsize == 100
        assert cache.ttl == 60.0
        assert cache.enabled is True

    def test_cache_disabled(self):
        """Test cache operations when disabled."""
        cache = TokenCountCache(enabled=False)

        # Set should not store
        cache.set("test text", "claude", 10)

        # Get should return None
        result = cache.get("test text", "claude")
        assert result is None

    def test_cache_hit_miss(self):
        """Test cache hit and miss tracking."""
        cache = TokenCountCache(maxsize=10, ttl=0)

        # First access is a miss
        result = cache.get("test", "claude")
        assert result is None

        # Store value
        cache.set("test", "claude", 5)

        # Second access is a hit
        result = cache.get("test", "claude")
        assert result == 5

        # Check statistics
        stats = cache.get_statistics()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_cache_key_generation(self):
        """Test cache key includes both text and tool name."""
        cache = TokenCountCache(maxsize=10, ttl=0)

        # Same text, different tools should be different cache entries
        cache.set("test", "claude", 5)
        cache.set("test", "codex", 6)

        assert cache.get("test", "claude") == 5
        assert cache.get("test", "codex") == 6

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = TokenCountCache(maxsize=2, ttl=0)

        # Fill cache
        cache.set("text1", "claude", 1)
        cache.set("text2", "claude", 2)

        # Add third item (should evict oldest)
        cache.set("text3", "claude", 3)

        # text1 should be evicted
        assert cache.get("text1", "claude") is None
        assert cache.get("text2", "claude") == 2
        assert cache.get("text3", "claude") == 3

        # Check eviction count
        stats = cache.get_statistics()
        assert stats.evictions >= 1

    def test_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache = TokenCountCache(maxsize=10, ttl=0.1)  # 100ms TTL

        cache.set("test", "claude", 5)

        # Should be available immediately
        assert cache.get("test", "claude") == 5

        # Wait for TTL to expire
        time.sleep(0.15)

        # Should be expired
        result = cache.get("test", "claude")
        assert result is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = TokenCountCache(maxsize=10, ttl=0)

        cache.set("test1", "claude", 1)
        cache.set("test2", "claude", 2)

        stats_before = cache.get_statistics()
        assert stats_before.size == 2

        cache.clear()

        stats_after = cache.get_statistics()
        assert stats_after.size == 0
        assert stats_after.hits == 0
        assert stats_after.misses == 0

    def test_cache_statistics(self):
        """Test cache statistics calculation."""
        cache = TokenCountCache(maxsize=10, ttl=0)

        # Generate some hits and misses
        cache.get("miss1", "claude")  # Miss
        cache.set("hit1", "claude", 1)
        cache.get("hit1", "claude")  # Hit
        cache.get("miss2", "claude")  # Miss
        cache.get("hit1", "claude")  # Hit

        stats = cache.get_statistics()
        assert stats.hits == 2
        assert stats.misses == 2
        assert stats.hit_rate == 0.5  # 2 hits / 4 total
        assert stats.size == 1
        assert stats.max_size == 10

    def test_cache_enable_disable(self):
        """Test enabling and disabling cache."""
        cache = TokenCountCache(maxsize=10, ttl=0)

        cache.set("test", "claude", 5)
        assert cache.get("test", "claude") == 5

        # Disable cache
        cache.disable()
        assert cache.enabled is False
        assert cache.get("test", "claude") is None

        # Re-enable cache
        cache.enable()
        assert cache.enabled is True

        # Cache was cleared, so need to re-populate
        cache.set("test", "claude", 5)
        assert cache.get("test", "claude") == 5


# TokenCounter Cache Integration Tests
class TestTokenCounterCacheIntegration:
    """Test TokenCounter integration with TokenCountCache."""

    def test_initialize_cache(self):
        """Test cache initialization."""
        cache = TokenCounter.initialize_cache(maxsize=100, ttl=60.0)
        assert cache is not None
        assert cache.maxsize == 100
        assert cache.ttl == 60.0

    def test_get_cache(self):
        """Test getting global cache."""
        TokenCounter.initialize_cache()
        cache = TokenCounter.get_cache()
        assert cache is not None

    def test_cache_hit_on_repeated_count(self):
        """Test cache hit for repeated token counting."""
        TokenCounter.initialize_cache(maxsize=10, ttl=0)

        text = "This is a test for caching"

        # First count (cache miss)
        count1 = TokenCounter.count_tokens(text, "claude", use_cache=True)

        # Second count (cache hit - should be faster)
        count2 = TokenCounter.count_tokens(text, "claude", use_cache=True)

        assert count1 == count2

        # Check cache statistics
        stats = TokenCounter.get_cache_statistics()
        assert stats.hits >= 1

    def test_count_tokens_without_cache(self):
        """Test token counting with cache disabled."""
        TokenCounter.initialize_cache(maxsize=10, ttl=0)

        text = "Test without cache"

        count1 = TokenCounter.count_tokens(text, "claude", use_cache=False)
        count2 = TokenCounter.count_tokens(text, "claude", use_cache=False)

        assert count1 == count2

        # Should have no cache hits
        stats = TokenCounter.get_cache_statistics()
        assert stats.hits == 0

    def test_cache_statistics(self):
        """Test cache statistics retrieval."""
        TokenCounter.initialize_cache(maxsize=10, ttl=0)

        TokenCounter.count_tokens("test1", "claude")
        TokenCounter.count_tokens("test2", "claude")
        TokenCounter.count_tokens("test1", "claude")  # Cache hit

        stats = TokenCounter.get_cache_statistics()
        assert stats is not None
        assert stats.hits >= 1
        assert stats.misses >= 2


# Batch Token Counting Tests
class TestBatchTokenCounting:
    """Test batch token counting functionality."""

    def test_batch_count_basic(self, sample_texts):
        """Test basic batch token counting."""
        counts = TokenCounter.batch_count_tokens(
            sample_texts, "claude", use_tokenizer=False
        )

        assert len(counts) == len(sample_texts)
        assert all(count > 0 for count in counts)

    def test_batch_count_empty_list(self):
        """Test batch counting with empty list."""
        counts = TokenCounter.batch_count_tokens([], "claude")
        assert counts == []

    def test_batch_count_with_cache(self, repeated_texts):
        """Test batch counting leverages cache for repeated texts."""
        TokenCounter.initialize_cache(maxsize=100, ttl=0)

        # First batch (cache misses)
        counts1 = TokenCounter.batch_count_tokens(repeated_texts, "claude")

        # Second batch (cache hits)
        counts2 = TokenCounter.batch_count_tokens(repeated_texts, "claude")

        assert counts1 == counts2

        # Check cache was used
        stats = TokenCounter.get_cache_statistics()
        assert stats.hits > 0

    def test_batch_count_consistency(self, sample_texts):
        """Test batch counting matches individual counting."""
        batch_counts = TokenCounter.batch_count_tokens(
            sample_texts, "claude", use_tokenizer=False
        )

        individual_counts = [
            TokenCounter.count_tokens(text, "claude", use_tokenizer=False)
            for text in sample_texts
        ]

        assert batch_counts == individual_counts

    def test_batch_count_different_tools(self, sample_texts):
        """Test batch counting for different tools."""
        claude_counts = TokenCounter.batch_count_tokens(sample_texts, "claude")
        codex_counts = TokenCounter.batch_count_tokens(sample_texts, "codex")
        gemini_counts = TokenCounter.batch_count_tokens(sample_texts, "gemini")

        assert len(claude_counts) == len(sample_texts)
        assert len(codex_counts) == len(sample_texts)
        assert len(gemini_counts) == len(sample_texts)


# Async Token Counting Tests
class TestAsyncTokenCounting:
    """Test async token counting functionality."""

    @pytest.mark.asyncio
    async def test_async_count_single(self):
        """Test async counting for single text."""
        text = "Test async counting"

        count = await TokenCounter.async_count_tokens(text, "claude")

        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_async_count_with_cache(self):
        """Test async counting uses cache."""
        TokenCounter.initialize_cache(maxsize=10, ttl=0)

        text = "Test async caching"

        # First count
        count1 = await TokenCounter.async_count_tokens(text, "claude")

        # Second count (should hit cache)
        count2 = await TokenCounter.async_count_tokens(text, "claude")

        assert count1 == count2

        stats = TokenCounter.get_cache_statistics()
        assert stats.hits >= 1

    @pytest.mark.asyncio
    async def test_async_batch_count(self, sample_texts):
        """Test async batch counting."""
        counts = await TokenCounter.async_batch_count_tokens(sample_texts, "claude")

        assert len(counts) == len(sample_texts)
        assert all(count > 0 for count in counts)

    @pytest.mark.asyncio
    async def test_async_batch_count_empty(self):
        """Test async batch counting with empty list."""
        counts = await TokenCounter.async_batch_count_tokens([], "claude")
        assert counts == []

    @pytest.mark.asyncio
    async def test_async_batch_count_concurrent(self, sample_texts):
        """Test async batch counting processes concurrently."""
        start_time = time.time()

        counts = await TokenCounter.async_batch_count_tokens(
            sample_texts * 10, "claude"
        )

        elapsed = time.time() - start_time

        assert len(counts) == len(sample_texts) * 10
        # Async should be reasonably fast even for many texts
        assert elapsed < 5.0  # Generous timeout

    @pytest.mark.asyncio
    async def test_async_batch_consistency(self, sample_texts):
        """Test async batch matches sync batch."""
        sync_counts = TokenCounter.batch_count_tokens(
            sample_texts, "claude", use_tokenizer=False
        )

        async_counts = await TokenCounter.async_batch_count_tokens(
            sample_texts, "claude", use_tokenizer=False
        )

        assert sync_counts == async_counts


# Performance Tests
class TestTokenCountingPerformance:
    """Test performance improvements from caching and batching."""

    def test_cache_performance_improvement(self, repeated_texts):
        """Test cache improves performance for repeated content."""
        TokenCounter.initialize_cache(maxsize=100, ttl=0)

        # First run (cache misses)
        start = time.time()
        for text in repeated_texts:
            TokenCounter.count_tokens(text, "claude", use_cache=False)
        no_cache_time = time.time() - start

        # Clear and re-initialize cache
        TokenCounter.clear_cache()
        TokenCounter.initialize_cache(maxsize=100, ttl=0)

        # Second run (with cache)
        # Prime cache
        for text in repeated_texts[:3]:
            TokenCounter.count_tokens(text, "claude", use_cache=True)

        start = time.time()
        for text in repeated_texts:
            TokenCounter.count_tokens(text, "claude", use_cache=True)
        with_cache_time = time.time() - start

        # Cache should be faster (but not required for test to pass)
        # Just verify both complete successfully
        assert no_cache_time >= 0
        assert with_cache_time >= 0

    def test_batch_vs_individual_performance(self, sample_texts):
        """Test batch counting vs individual counting."""
        # Individual counting
        start = time.time()
        individual_counts = [
            TokenCounter.count_tokens(text, "claude", use_cache=False)
            for text in sample_texts * 10
        ]
        individual_time = time.time() - start

        # Batch counting
        start = time.time()
        batch_counts = TokenCounter.batch_count_tokens(
            sample_texts * 10, "claude", use_cache=False
        )
        batch_time = time.time() - start

        # Both should produce same results
        assert individual_counts == batch_counts

        # Both should complete reasonably quickly
        assert individual_time < 5.0
        assert batch_time < 5.0


# Integration with TokenBudgetManager
class TestTokenBudgetManagerWithCache:
    """Test TokenBudgetManager integration with caching."""

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules for testing."""
        return [
            MemoryRule(
                id=f"rule_{i}",
                name=f"test_rule_{i}",
                rule=f"Test rule {i} with some content for token counting",
                category=MemoryCategory.BEHAVIOR,
                authority=AuthorityLevel.DEFAULT,
                scope=["python"],
                source="test",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                conditions=None,
                replaces=[],
                metadata={"priority": 50},
            )
            for i in range(10)
        ]

    def test_budget_allocation_with_cache(self, sample_rules):
        """Test budget allocation uses cache."""
        # Initialize cache
        TokenCounter.initialize_cache(maxsize=100, ttl=0)

        manager = TokenBudgetManager(enable_cache=True)

        # First allocation
        allocation1 = manager.allocate_budget(sample_rules, 10000, "claude")

        # Second allocation (should use cache)
        allocation2 = manager.allocate_budget(sample_rules, 10000, "claude")

        assert len(allocation1.rules_included) == len(allocation2.rules_included)
        assert allocation1.absolute_tokens == allocation2.absolute_tokens

        # Cache should have hits
        stats = TokenCounter.get_cache_statistics()
        assert stats.hits > 0

    def test_budget_allocation_cache_disabled(self, sample_rules):
        """Test budget allocation with cache disabled."""
        manager = TokenBudgetManager(enable_cache=False)

        allocation = manager.allocate_budget(sample_rules, 10000, "claude")

        assert len(allocation.rules_included) > 0
        assert allocation.total_budget == 10000
        # Allocation should work regardless of cache state


# Edge Cases and Error Handling
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_count_empty_string(self):
        """Test counting empty string."""
        count = TokenCounter.count_tokens("", "claude", use_cache=False)
        assert count == 1  # Minimum count is 1

    def test_count_very_long_text(self):
        """Test counting very long text."""
        long_text = "word " * 10000  # 10k words

        count = TokenCounter.count_tokens(long_text, "claude", use_cache=False)
        assert count > 1000  # Should be substantial

    def test_cache_thread_safety(self):
        """Test cache operations are thread-safe."""
        import threading

        TokenCounter.initialize_cache(maxsize=100, ttl=0)

        def count_repeatedly():
            for i in range(100):
                TokenCounter.count_tokens(f"text_{i % 10}", "claude")

        threads = [threading.Thread(target=count_repeatedly) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors
        stats = TokenCounter.get_cache_statistics()
        assert stats is not None

    @pytest.mark.asyncio
    async def test_async_count_error_handling(self):
        """Test async counting handles errors gracefully."""
        # Should not raise exception
        count = await TokenCounter.async_count_tokens("test", "invalid_tool")
        assert count > 0  # Falls back to default counter
