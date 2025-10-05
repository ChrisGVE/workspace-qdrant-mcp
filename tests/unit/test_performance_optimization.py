"""
Unit tests for performance optimization utilities.
"""

import time
from datetime import datetime

import pytest

from src.python.common.core.context_injection.performance_optimization import (
    BatchProcessor,
    PerformanceMetrics,
    RuleCache,
    RuleIndex,
)
from src.python.common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)


@pytest.fixture
def sample_rules():
    """Create sample memory rules for testing."""
    return [
        MemoryRule(
            id="rule1",
            name="Python Standards",
            rule="Use type hints and follow PEP 8",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["python", "coding"],
            metadata={"project_id": "project1", "priority": 100},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        MemoryRule(
            id="rule2",
            name="Testing Guidelines",
            rule="Write pytest tests for all functions",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.DEFAULT,
            scope=["testing"],
            metadata={"project_id": "project1", "priority": 50},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        MemoryRule(
            id="rule3",
            name="Documentation Standards",
            rule="Add docstrings to all public APIs",
            category=MemoryCategory.BEHAVIOR,
            authority=AuthorityLevel.DEFAULT,
            scope=["docs"],
            metadata={"project_id": "project2", "priority": 75},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
        MemoryRule(
            id="rule4",
            name="Security Requirements",
            rule="Validate all user inputs",
            category=MemoryCategory.PREFERENCE,
            authority=AuthorityLevel.ABSOLUTE,
            scope=["security", "python"],
            metadata={"project_id": "project2", "priority": 100},
            created_at=datetime.now(),
            updated_at=datetime.now(),
        ),
    ]


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = PerformanceMetrics()
        assert metrics.cache_hit is False
        assert metrics.fetch_time_ms == 0.0
        assert metrics.filter_time_ms == 0.0
        assert metrics.total_rules_fetched == 0
        assert metrics.rules_after_filtering == 0
        assert metrics.index_lookup_time_ms is None
        assert metrics.batch_processing_time_ms is None

    def test_custom_values(self):
        """Test setting custom metric values."""
        metrics = PerformanceMetrics(
            cache_hit=True,
            fetch_time_ms=12.5,
            filter_time_ms=3.2,
            total_rules_fetched=100,
            rules_after_filtering=25,
            index_lookup_time_ms=0.5,
            batch_processing_time_ms=8.0,
        )
        assert metrics.cache_hit is True
        assert metrics.fetch_time_ms == 12.5
        assert metrics.filter_time_ms == 3.2
        assert metrics.total_rules_fetched == 100
        assert metrics.rules_after_filtering == 25
        assert metrics.index_lookup_time_ms == 0.5
        assert metrics.batch_processing_time_ms == 8.0


class TestRuleCache:
    """Test RuleCache functionality."""

    def test_cache_initialization(self):
        """Test cache is properly initialized."""
        cache = RuleCache(maxsize=10, ttl=60)
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["current_size"] == 0
        assert stats["max_size"] == 10

    def test_cache_miss(self, sample_rules):
        """Test cache miss."""
        cache = RuleCache()
        key = ("test", "key")
        result = cache.get(key)
        assert result is None
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_cache_hit(self, sample_rules):
        """Test cache hit."""
        cache = RuleCache()
        key = ("test", "key")
        cache.set(key, sample_rules[:2])

        result = cache.get(key)
        assert result is not None
        assert len(result) == 2
        assert result[0].id == "rule1"

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_cache_hit_rate(self, sample_rules):
        """Test cache hit rate calculation."""
        cache = RuleCache()
        key1 = ("test", "key1")
        key2 = ("test", "key2")

        # Miss
        cache.get(key1)
        # Set
        cache.set(key1, sample_rules[:2])
        # Hit
        cache.get(key1)
        # Miss
        cache.get(key2)
        # Set
        cache.set(key2, sample_rules[2:])
        # Hit
        cache.get(key2)
        # Hit
        cache.get(key1)

        stats = cache.get_stats()
        assert stats["hits"] == 3
        assert stats["misses"] == 2
        assert stats["hit_rate"] == 0.6  # 3/5

    def test_cache_clear(self, sample_rules):
        """Test cache clearing."""
        cache = RuleCache()
        cache.set(("key1",), sample_rules[:2])
        cache.set(("key2",), sample_rules[2:])

        stats = cache.get_stats()
        assert stats["current_size"] == 2

        cache.clear()
        stats = cache.get_stats()
        assert stats["current_size"] == 0

    def test_cache_ttl_expiration(self, sample_rules):
        """Test TTL expiration (fast test with short TTL)."""
        cache = RuleCache(maxsize=10, ttl=1)  # 1 second TTL
        key = ("test", "key")
        cache.set(key, sample_rules[:2])

        # Immediate retrieval should work
        result = cache.get(key)
        assert result is not None

        # Wait for TTL expiration
        time.sleep(1.1)

        # Should be expired now
        result = cache.get(key)
        assert result is None

    def test_cache_maxsize(self, sample_rules):
        """Test LRU eviction when maxsize is reached."""
        cache = RuleCache(maxsize=2, ttl=60)

        # Add 3 items (should evict first one)
        cache.set(("key1",), sample_rules[:1])
        cache.set(("key2",), sample_rules[1:2])
        cache.set(("key3",), sample_rules[2:3])

        stats = cache.get_stats()
        assert stats["current_size"] == 2  # Only 2 items fit
        assert stats["max_size"] == 2


class TestRuleIndex:
    """Test RuleIndex functionality."""

    def test_index_initialization(self):
        """Test index is properly initialized."""
        index = RuleIndex()
        assert not index.is_built()

    def test_index_build(self, sample_rules):
        """Test building index from rules."""
        index = RuleIndex()
        index.build_index(sample_rules)

        assert index.is_built()
        all_rules = index.get_all_rules()
        assert len(all_rules) == 4

    def test_lookup_by_category(self, sample_rules):
        """Test category lookup."""
        index = RuleIndex()
        index.build_index(sample_rules)

        prefs = index.lookup_by_category(MemoryCategory.PREFERENCE)
        assert len(prefs) == 3
        assert all(r.category == MemoryCategory.PREFERENCE for r in prefs)

        behaviors = index.lookup_by_category(MemoryCategory.BEHAVIOR)
        assert len(behaviors) == 1
        assert behaviors[0].id == "rule3"

    def test_lookup_by_authority(self, sample_rules):
        """Test authority lookup."""
        index = RuleIndex()
        index.build_index(sample_rules)

        absolute_rules = index.lookup_by_authority(AuthorityLevel.ABSOLUTE)
        assert len(absolute_rules) == 2
        assert all(r.authority == AuthorityLevel.ABSOLUTE for r in absolute_rules)

        default_rules = index.lookup_by_authority(AuthorityLevel.DEFAULT)
        assert len(default_rules) == 2
        assert all(r.authority == AuthorityLevel.DEFAULT for r in default_rules)

    def test_lookup_by_project(self, sample_rules):
        """Test project ID lookup."""
        index = RuleIndex()
        index.build_index(sample_rules)

        project1_rules = index.lookup_by_project("project1")
        assert len(project1_rules) == 2
        assert all(
            r.metadata.get("project_id") == "project1" for r in project1_rules
        )

        project2_rules = index.lookup_by_project("project2")
        assert len(project2_rules) == 2

    def test_lookup_by_scope(self, sample_rules):
        """Test scope lookup."""
        index = RuleIndex()
        index.build_index(sample_rules)

        python_rules = index.lookup_by_scope("python")
        assert len(python_rules) == 2
        assert all("python" in r.scope for r in python_rules)

        testing_rules = index.lookup_by_scope("testing")
        assert len(testing_rules) == 1
        assert testing_rules[0].id == "rule2"

    def test_lookup_before_build_raises_error(self):
        """Test that lookups before building raise error."""
        index = RuleIndex()

        with pytest.raises(RuntimeError, match="Index not built"):
            index.lookup_by_category(MemoryCategory.PREFERENCE)

        with pytest.raises(RuntimeError, match="Index not built"):
            index.lookup_by_authority(AuthorityLevel.ABSOLUTE)

        with pytest.raises(RuntimeError, match="Index not built"):
            index.lookup_by_project("project1")

        with pytest.raises(RuntimeError, match="Index not built"):
            index.get_all_rules()

    def test_index_rebuild(self, sample_rules):
        """Test rebuilding index with new data."""
        index = RuleIndex()
        index.build_index(sample_rules[:2])

        assert len(index.get_all_rules()) == 2

        # Rebuild with more rules
        index.build_index(sample_rules)
        assert len(index.get_all_rules()) == 4

    def test_index_clear(self, sample_rules):
        """Test clearing index."""
        index = RuleIndex()
        index.build_index(sample_rules)

        assert index.is_built()

        index.clear()
        assert not index.is_built()


class TestBatchProcessor:
    """Test BatchProcessor functionality."""

    def test_process_in_batches(self, sample_rules):
        """Test batch processing."""

        def process_batch(batch):
            return [r.id for r in batch]

        results = BatchProcessor.process_in_batches(
            sample_rules, process_batch, batch_size=2
        )

        assert len(results) == 2  # 4 rules / 2 per batch
        assert results[0] == ["rule1", "rule2"]
        assert results[1] == ["rule3", "rule4"]

    def test_process_single_batch(self, sample_rules):
        """Test processing with batch size larger than data."""

        def process_batch(batch):
            return len(batch)

        results = BatchProcessor.process_in_batches(
            sample_rules, process_batch, batch_size=10
        )

        assert len(results) == 1
        assert results[0] == 4

    def test_flatten_batch_results(self):
        """Test flattening batch results."""
        batch_results = [["a", "b"], ["c", "d"], ["e"]]
        flattened = BatchProcessor.flatten_batch_results(batch_results)

        assert flattened == ["a", "b", "c", "d", "e"]

    def test_empty_batch_processing(self):
        """Test processing empty list."""

        def process_batch(batch):
            return batch

        results = BatchProcessor.process_in_batches([], process_batch, batch_size=10)

        assert len(results) == 0

    def test_batch_size_one(self, sample_rules):
        """Test batch size of 1."""

        def process_batch(batch):
            return batch[0].id

        results = BatchProcessor.process_in_batches(
            sample_rules, process_batch, batch_size=1
        )

        assert len(results) == 4
        assert results == ["rule1", "rule2", "rule3", "rule4"]
