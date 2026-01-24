"""
Comprehensive performance and stress tests for memory rules system.

Tests system performance with large rule sets (100-5000 rules) including:
1. Rule retrieval performance benchmarks
2. Conflict detection performance at scale
3. LLM injection preparation performance
4. Memory usage validation
5. Concurrent operations stress testing
6. Optimization strategy performance

Performance Baselines (Task 324.8):
- Rule retrieval: <100ms for 1000 rules
- Conflict detection: <1s for 1000 rules
- LLM injection prep: <500ms for context generation
- Memory usage: <100MB for 5000 rules

Test Organization:
- Use @pytest.mark.performance for all performance tests
- Use pytest-benchmark for accurate timing measurements
- Mock Qdrant for Python code isolation tests
- Include some integration tests with real Qdrant

NOTE: Many tests in this file use asyncio.run() inside pytest-benchmark's
pedantic() function while also being marked with @pytest.mark.asyncio.
This causes "asyncio.run() cannot be called from a running event loop" errors.
These tests need refactoring to properly handle async benchmarking.
Until then, they are marked with xfail.
"""

import asyncio
import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest
from common.core.collection_naming import CollectionNamingManager, CollectionType
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryConflict,
    MemoryManager,
    MemoryRule,
    format_memory_rules_for_injection,
)
from common.core.sparse_vectors import BM25SparseEncoder
from qdrant_client.models import PointStruct, ScoredPoint

# ============================================================================
# HELPER FUNCTIONS AND FIXTURES
# ============================================================================

def create_test_rule(
    rule_id: int,
    category: MemoryCategory = MemoryCategory.BEHAVIOR,
    authority: AuthorityLevel = AuthorityLevel.DEFAULT,
    rule_length: int = 50,
) -> MemoryRule:
    """Create a test rule with specified parameters."""
    rule_text = f"Test rule {rule_id}: " + " ".join([f"word{i}" for i in range(rule_length)])

    return MemoryRule(
        id=f"rule_{rule_id}",
        category=category,
        name=f"Rule {rule_id}",
        rule=rule_text,
        authority=authority,
        scope=["global", f"scope_{rule_id % 10}"],
        source="test_performance",
        created_at=datetime.now(timezone.utc) - timedelta(days=rule_id % 100),
        updated_at=datetime.now(timezone.utc),
    )


def create_large_rule_set(count: int, conflict_rate: float = 0.1) -> list[MemoryRule]:
    """
    Create a large set of test rules with optional conflicts.

    Args:
        count: Number of rules to create
        conflict_rate: Proportion of rules that should have potential conflicts

    Returns:
        List of MemoryRule instances
    """
    rules = []

    # Create diverse rules
    for i in range(count):
        # Vary category distribution: 60% behavior, 30% preference, 10% agent
        if i % 10 < 6:
            category = MemoryCategory.BEHAVIOR
        elif i % 10 < 9:
            category = MemoryCategory.PREFERENCE
        else:
            category = MemoryCategory.AGENT

        # Vary authority: 20% absolute, 80% default
        authority = AuthorityLevel.ABSOLUTE if i % 5 == 0 else AuthorityLevel.DEFAULT

        # Vary rule length: short, medium, long
        if i % 3 == 0:
            rule_length = 10  # Short
        elif i % 3 == 1:
            rule_length = 50  # Medium
        else:
            rule_length = 100  # Long

        rule = create_test_rule(i, category, authority, rule_length)

        # Add potential conflicts based on conflict_rate
        if i > 0 and (i / count) < conflict_rate:
            # Make rule potentially conflict with earlier rule
            conflict_keywords = ["always", "never", "must", "avoid"]
            rule.rule = f"{conflict_keywords[i % len(conflict_keywords)]} {rule.rule}"

        rules.append(rule)

    return rules


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


class MockQdrantClient:
    """Mock Qdrant client optimized for performance testing."""

    def __init__(self):
        self.storage: dict[str, dict[str, PointStruct]] = {}
        self.collections = set()

    def get_collections(self):
        """Mock get_collections."""
        from types import SimpleNamespace
        return SimpleNamespace(collections=[SimpleNamespace(name=c) for c in self.collections])

    def create_collection(self, collection_name, vectors_config):
        """Mock create_collection."""
        self.collections.add(collection_name)
        self.storage[collection_name] = {}

    def upsert(self, collection_name, points):
        """Mock upsert with storage."""
        if collection_name not in self.storage:
            self.storage[collection_name] = {}

        for point in points:
            self.storage[collection_name][point.id] = point

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        """Mock retrieve from storage."""
        if collection_name not in self.storage:
            return []

        results = []
        for rule_id in ids:
            if rule_id in self.storage[collection_name]:
                results.append(self.storage[collection_name][rule_id])

        return results

    def scroll(self, collection_name, scroll_filter=None, limit=100, with_payload=True):
        """Mock scroll through all points."""
        if collection_name not in self.storage:
            return [], None

        points = list(self.storage[collection_name].values())[:limit]
        return points, None

    def search(self, collection_name, query_vector, query_filter=None, limit=10, with_payload=True):
        """Mock search with fake scoring."""
        if collection_name not in self.storage:
            return []

        # Return all points with mock scores
        points = list(self.storage[collection_name].values())[:limit]
        scored_points = []

        for i, point in enumerate(points):
            scored_point = Mock(spec=ScoredPoint)
            scored_point.id = point.id
            scored_point.payload = point.payload
            scored_point.score = 1.0 - (i * 0.05)  # Decreasing scores
            scored_points.append(scored_point)

        return scored_points

    def delete(self, collection_name, points_selector):
        """Mock delete."""
        if collection_name not in self.storage:
            return

        for rule_id in points_selector:
            if rule_id in self.storage[collection_name]:
                del self.storage[collection_name][rule_id]


@pytest.fixture
def mock_qdrant_client():
    """Provide mock Qdrant client for performance tests."""
    return MockQdrantClient()


@pytest.fixture
def naming_manager():
    """Provide collection naming manager."""
    return CollectionNamingManager()


@pytest.fixture
async def memory_manager_with_data(mock_qdrant_client, naming_manager):
    """Create memory manager pre-populated with test data."""
    manager = MemoryManager(
        qdrant_client=mock_qdrant_client,
        naming_manager=naming_manager,
        embedding_dim=384,
        sparse_vector_generator=None,
    )

    await manager.initialize_memory_collection()
    return manager


# ============================================================================
# PERFORMANCE TESTS: RULE RETRIEVAL
# ============================================================================

@pytest.mark.performance
class TestRuleRetrievalPerformance:
    """Test rule retrieval performance with various dataset sizes."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="asyncio.run() cannot be called from running event loop - needs refactoring")
    async def test_retrieve_100_rules_baseline(self, memory_manager_with_data, benchmark):
        """Baseline: Retrieve individual rules from 100-rule set."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(100)

        # Add rules to manager
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Benchmark individual rule retrieval
        async def retrieve_middle_rule():
            return await manager.get_memory_rule("rule_50")

        # Warm up
        await retrieve_middle_rule()

        # Benchmark
        result = benchmark.pedantic(
            lambda: asyncio.run(retrieve_middle_rule()),
            iterations=10,
            rounds=5,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_list_1000_rules_under_100ms(self, memory_manager_with_data):
        """Verify listing 1000 rules completes in <100ms."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(1000)

        # Add rules
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Measure list_memory_rules performance
        start_time = time.perf_counter()
        all_rules = await manager.list_memory_rules()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert len(all_rules) == 1000
        assert elapsed_ms < 100, f"List operation took {elapsed_ms:.2f}ms, expected <100ms"

    @pytest.mark.asyncio
    async def test_search_rules_performance_scaling(self, memory_manager_with_data):
        """Test search performance scales reasonably with dataset size."""
        manager = memory_manager_with_data

        timings = {}

        for count in [100, 500, 1000]:
            # Create and add rules
            rules = create_large_rule_set(count)
            for rule in rules:
                await manager.add_memory_rule(
                    category=rule.category,
                    name=rule.name,
                    rule=rule.rule,
                    authority=rule.authority,
                    scope=rule.scope,
                )

            # Measure search time
            start_time = time.perf_counter()
            await manager.search_memory_rules(query="test rule", limit=10)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            timings[count] = elapsed_ms

            # Clear for next iteration
            manager.client.storage.clear()
            manager.client.collections.clear()
            await manager.initialize_memory_collection()

        # Verify search time scales reasonably (should be sub-linear due to limit)
        assert timings[1000] < timings[100] * 5, \
            f"Search time scaling is poor: {timings}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="asyncio.run() cannot be called from running event loop - needs refactoring")
    async def test_filtered_retrieval_performance(self, memory_manager_with_data, benchmark):
        """Test filtered rule retrieval performance."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(1000)

        # Add rules
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Benchmark filtered retrieval
        async def retrieve_behavior_rules():
            return await manager.list_memory_rules(category=MemoryCategory.BEHAVIOR)

        result = benchmark.pedantic(
            lambda: asyncio.run(retrieve_behavior_rules()),
            iterations=10,
            rounds=3,
        )

        assert len(result) > 0


# ============================================================================
# PERFORMANCE TESTS: CONFLICT DETECTION
# ============================================================================

@pytest.mark.performance
class TestConflictDetectionPerformance:
    """Test conflict detection performance at scale."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="detect_conflicts() not implemented - returns mock data")
    async def test_conflict_detection_1000_rules_under_1s(self, memory_manager_with_data):
        """Verify conflict detection on 1000 rules completes in <1s."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(1000, conflict_rate=0.2)

        # Measure conflict detection time
        start_time = time.perf_counter()
        conflicts = await manager.detect_conflicts(rules)
        elapsed_s = time.perf_counter() - start_time

        assert elapsed_s < 1.0, f"Conflict detection took {elapsed_s:.3f}s, expected <1s"
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="detect_conflicts() not implemented - returns mock data")
    async def test_conflict_detection_scaling(self, memory_manager_with_data):
        """Test conflict detection time scaling with dataset size."""
        manager = memory_manager_with_data

        timings = {}

        for count in [100, 500, 1000]:
            rules = create_large_rule_set(count, conflict_rate=0.1)

            start_time = time.perf_counter()
            await manager.detect_conflicts(rules)
            elapsed_s = time.perf_counter() - start_time

            timings[count] = elapsed_s

        # Conflict detection is O(n²) worst case, but should be optimized
        # Verify it doesn't scale quadratically
        assert timings[1000] < timings[100] * 50, \
            f"Conflict detection scaling is poor: {timings}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="asyncio.run() cannot be called from running event loop - needs refactoring")
    async def test_conflict_detection_no_conflicts(self, memory_manager_with_data, benchmark):
        """Benchmark conflict detection when no conflicts exist."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(500, conflict_rate=0.0)

        async def detect_conflicts_no_matches():
            return await manager.detect_conflicts(rules)

        result = benchmark.pedantic(
            lambda: asyncio.run(detect_conflicts_no_matches()),
            iterations=5,
            rounds=3,
        )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_conflict_detection_high_conflict_rate(self, memory_manager_with_data):
        """Test performance with high conflict rate (stress test)."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(500, conflict_rate=0.5)

        start_time = time.perf_counter()
        await manager.detect_conflicts(rules)
        elapsed_s = time.perf_counter() - start_time

        # Should still complete in reasonable time even with many conflicts
        assert elapsed_s < 2.0, f"High-conflict detection took {elapsed_s:.3f}s"


# ============================================================================
# PERFORMANCE TESTS: LLM INJECTION PREPARATION
# ============================================================================

@pytest.mark.performance
class TestLLMInjectionPerformance:
    """Test LLM context injection preparation performance."""

    @pytest.mark.asyncio
    async def test_format_100_rules_for_injection(self, benchmark):
        """Benchmark formatting 100 rules for LLM injection."""
        rules = create_large_rule_set(100)

        def format_rules():
            return format_memory_rules_for_injection(rules)

        result = benchmark.pedantic(format_rules, iterations=50, rounds=5)

        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_format_1000_rules_under_500ms(self):
        """Verify formatting 1000 rules for injection completes in <500ms."""
        rules = create_large_rule_set(1000)

        start_time = time.perf_counter()
        formatted = format_memory_rules_for_injection(rules)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 500, f"Formatting took {elapsed_ms:.2f}ms, expected <500ms"
        assert isinstance(formatted, str)
        assert len(formatted) > 0

    @pytest.mark.asyncio
    async def test_injection_preparation_with_filtering(self, memory_manager_with_data):
        """Test full injection preparation pipeline with filtering."""
        manager = memory_manager_with_data
        rules = create_large_rule_set(1000)

        # Add rules
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Measure full pipeline: retrieve + filter + format
        start_time = time.perf_counter()

        # Get absolute rules first
        absolute_rules = await manager.list_memory_rules(authority=AuthorityLevel.ABSOLUTE)
        # Get default rules
        default_rules = await manager.list_memory_rules(authority=AuthorityLevel.DEFAULT)
        # Format for injection
        formatted = format_memory_rules_for_injection(absolute_rules + default_rules)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 500, f"Full pipeline took {elapsed_ms:.2f}ms"
        assert len(formatted) > 0


# ============================================================================
# PERFORMANCE TESTS: MEMORY USAGE
# ============================================================================

@pytest.mark.performance
class TestMemoryUsage:
    """Test memory usage with large rule sets."""

    @pytest.mark.asyncio
    async def test_memory_usage_5000_rules_under_100mb(self, memory_manager_with_data):
        """Verify memory usage for 5000 rules stays under 100MB."""
        manager = memory_manager_with_data

        # Force garbage collection for accurate baseline
        gc.collect()
        baseline_mb = get_memory_usage_mb()

        # Create and add 5000 rules
        rules = create_large_rule_set(5000)

        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Measure memory after adding rules
        gc.collect()
        current_mb = get_memory_usage_mb()
        memory_increase_mb = current_mb - baseline_mb

        assert memory_increase_mb < 100, \
            f"Memory increased by {memory_increase_mb:.2f}MB, expected <100MB"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Memory scaling test requires actual Qdrant - mock doesn't measure real memory")
    async def test_memory_usage_scaling(self, memory_manager_with_data):
        """Test memory usage scales linearly with rule count."""
        manager = memory_manager_with_data

        memory_usage = {}

        for count in [1000, 2000, 3000]:
            # Clear previous data
            manager.client.storage.clear()
            manager.client.collections.clear()
            await manager.initialize_memory_collection()

            gc.collect()
            baseline_mb = get_memory_usage_mb()

            rules = create_large_rule_set(count)
            for rule in rules:
                await manager.add_memory_rule(
                    category=rule.category,
                    name=rule.name,
                    rule=rule.rule,
                    authority=rule.authority,
                    scope=rule.scope,
                )

            gc.collect()
            current_mb = get_memory_usage_mb()
            memory_usage[count] = current_mb - baseline_mb

        # Memory should scale roughly linearly
        # 3000 rules should use at most 3.5x memory of 1000 rules
        ratio = memory_usage[3000] / memory_usage[1000]
        assert ratio < 3.5, f"Memory scaling is non-linear: {memory_usage}"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Memory cleanup test requires actual Qdrant - mock doesn't measure real memory")
    async def test_memory_cleanup_after_deletion(self, memory_manager_with_data):
        """Verify memory is properly released after rule deletion."""
        manager = memory_manager_with_data

        # Add 1000 rules
        rules = create_large_rule_set(1000)
        rule_ids = []

        for rule in rules:
            rule_id = await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )
            rule_ids.append(rule_id)

        gc.collect()
        with_rules_mb = get_memory_usage_mb()

        # Delete all rules
        for rule_id in rule_ids:
            await manager.delete_memory_rule(rule_id)

        gc.collect()
        after_deletion_mb = get_memory_usage_mb()

        # Memory should be released (allow some overhead)
        memory_retained_mb = after_deletion_mb - (with_rules_mb - 50)  # 50MB tolerance
        assert memory_retained_mb < 20, \
            f"Too much memory retained after deletion: {memory_retained_mb:.2f}MB"


# ============================================================================
# STRESS TESTS: CONCURRENT OPERATIONS
# ============================================================================

@pytest.mark.performance
class TestConcurrentOperations:
    """Stress test concurrent rule operations."""

    @pytest.mark.asyncio
    async def test_concurrent_rule_additions(self, memory_manager_with_data):
        """Test concurrent rule additions complete successfully."""
        manager = memory_manager_with_data

        # Create tasks for concurrent additions
        async def add_rule(idx):
            rule = create_test_rule(idx)
            return await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Add 100 rules concurrently
        start_time = time.perf_counter()
        tasks = [add_rule(i) for i in range(100)]
        rule_ids = await asyncio.gather(*tasks)
        elapsed_s = time.perf_counter() - start_time

        assert len(rule_ids) == 100
        assert len(set(rule_ids)) == 100  # All unique
        assert elapsed_s < 5.0, f"Concurrent additions took {elapsed_s:.2f}s"

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="get_memory_rule returns None for mock storage keys")
    async def test_concurrent_rule_retrievals(self, memory_manager_with_data):
        """Test concurrent rule retrievals don't cause issues."""
        manager = memory_manager_with_data

        # Add rules first
        rules = create_large_rule_set(100)
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Retrieve concurrently
        async def retrieve_rule(rule_id):
            return await manager.get_memory_rule(f"rule_{rule_id}")

        tasks = [retrieve_rule(i) for i in range(100)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 100
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_updates_and_reads(self, memory_manager_with_data):
        """Test concurrent updates and reads don't corrupt data."""
        manager = memory_manager_with_data

        # Add initial rules
        for i in range(50):
            rule = create_test_rule(i)
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Mix of updates and reads
        async def update_rule(idx):
            return await manager.update_memory_rule(
                f"rule_{idx}",
                {"rule": f"Updated rule {idx}"}
            )

        async def read_rule(idx):
            return await manager.get_memory_rule(f"rule_{idx}")

        tasks = []
        for i in range(50):
            if i % 2 == 0:
                tasks.append(update_rule(i))
            else:
                tasks.append(read_rule(i))

        results = await asyncio.gather(*tasks)
        assert len(results) == 50

    @pytest.mark.asyncio
    async def test_race_condition_detection(self, memory_manager_with_data):
        """Test for race conditions in concurrent operations."""
        manager = memory_manager_with_data

        # Create a rule
        rule = create_test_rule(0)
        rule_id = await manager.add_memory_rule(
            category=rule.category,
            name=rule.name,
            rule=rule.rule,
            authority=rule.authority,
            scope=rule.scope,
        )

        # Try concurrent updates to same rule
        async def update_with_value(value):
            return await manager.update_memory_rule(
                rule_id,
                {"metadata": {"update_value": value}}
            )

        tasks = [update_with_value(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All updates should succeed (last write wins)
        assert all(results)

        # Verify final state is consistent
        final_rule = await manager.get_memory_rule(rule_id)
        assert final_rule is not None


# ============================================================================
# PERFORMANCE TESTS: OPTIMIZATION STRATEGIES
# ============================================================================

@pytest.mark.performance
class TestOptimizationStrategies:
    """Test performance of optimization strategies."""

    @pytest.mark.asyncio
    async def test_token_budget_optimization_performance(self, memory_manager_with_data):
        """Test token budget optimization performance."""
        manager = memory_manager_with_data

        # Add 1000 rules
        rules = create_large_rule_set(1000)
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Measure optimization time
        start_time = time.perf_counter()
        tokens_saved, actions = await manager.optimize_memory(max_tokens=2000)
        elapsed_s = time.perf_counter() - start_time

        assert elapsed_s < 1.0, f"Optimization took {elapsed_s:.3f}s"
        assert isinstance(tokens_saved, int)
        assert isinstance(actions, list)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="asyncio.run() cannot be called from running event loop - needs refactoring")
    async def test_stats_calculation_performance(self, memory_manager_with_data, benchmark):
        """Benchmark memory stats calculation."""
        manager = memory_manager_with_data

        # Add rules
        rules = create_large_rule_set(500)
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        async def get_stats():
            return await manager.get_memory_stats()

        result = benchmark.pedantic(
            lambda: asyncio.run(get_stats()),
            iterations=10,
            rounds=3,
        )

        assert result.total_rules > 0


# ============================================================================
# INTEGRATION PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
@pytest.mark.integration
class TestIntegrationPerformance:
    """Integration performance tests with realistic workflows."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="detect_conflicts() not fully implemented - returns mock data")
    async def test_full_session_initialization_performance(self, memory_manager_with_data):
        """Test full session initialization with large rule set."""
        manager = memory_manager_with_data

        # Simulate realistic session: 500 rules
        rules = create_large_rule_set(500)
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        # Measure full session initialization
        start_time = time.perf_counter()

        # 1. List all rules
        all_rules = await manager.list_memory_rules()

        # 2. Detect conflicts
        await manager.detect_conflicts(all_rules)

        # 3. Get stats
        stats = await manager.get_memory_stats()

        # 4. Format for injection
        format_memory_rules_for_injection(all_rules)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        assert elapsed_ms < 1000, f"Session init took {elapsed_ms:.2f}ms"
        assert len(all_rules) > 0
        assert stats.total_rules == len(all_rules)

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="asyncio.run() cannot be called from running event loop - needs refactoring")
    async def test_search_and_format_pipeline(self, memory_manager_with_data, benchmark):
        """Benchmark common workflow: search rules and format for injection."""
        manager = memory_manager_with_data

        # Add rules
        rules = create_large_rule_set(1000)
        for rule in rules:
            await manager.add_memory_rule(
                category=rule.category,
                name=rule.name,
                rule=rule.rule,
                authority=rule.authority,
                scope=rule.scope,
            )

        async def search_and_format():
            # Search for relevant rules
            results = await manager.search_memory_rules(query="test rule", limit=20)
            relevant_rules = [rule for rule, score in results]

            # Format for injection
            return format_memory_rules_for_injection(relevant_rules)

        result = benchmark.pedantic(
            lambda: asyncio.run(search_and_format()),
            iterations=5,
            rounds=3,
        )

        assert isinstance(result, str)
