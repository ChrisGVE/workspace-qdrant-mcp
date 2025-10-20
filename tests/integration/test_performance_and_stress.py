"""
Performance Impact and Stress Testing for Memory Rules (Task 337.7).

Tests system performance and stability under various load conditions:
- Rule injection overhead measurement
- Large rule set handling
- Response time impact
- Concurrent operation stability
- Memory usage monitoring
- Token budget adherence

Performance Metrics:
1. Injection latency (time to add/retrieve rules)
2. LLM response time with varying rule counts
3. Memory footprint with large rule sets
4. Concurrent operation throughput
5. Token consumption efficiency
"""

import asyncio
import pytest
import time
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from unittest.mock import AsyncMock, Mock, patch
import statistics

from src.python.common.memory.types import (
    AuthorityLevel,
    MemoryCategory,
    MemoryRule,
)

# Import test harness from Task 337.1
from tests.integration.test_llm_behavioral_harness import (
    LLMBehavioralHarness,
    MockLLMProvider,
    ExecutionMode,
    BehavioralMetrics,
    LLMResponse,
)

# Try to import real components
try:
    from src.python.common.core.memory import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
except ImportError:
    MEMORY_MANAGER_AVAILABLE = False


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize performance monitor."""
        self.metrics: Dict[str, List[float]] = {}
        self.start_times: Dict[str, float] = {}

    def start_timer(self, operation: str):
        """Start timing an operation.

        Args:
            operation: Operation name
        """
        self.start_times[operation] = time.perf_counter()

    def stop_timer(self, operation: str) -> float:
        """Stop timing and record duration.

        Args:
            operation: Operation name

        Returns:
            Duration in seconds
        """
        if operation not in self.start_times:
            return 0.0

        duration = time.perf_counter() - self.start_times[operation]

        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append(duration)
        return duration

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation.

        Args:
            operation: Operation name

        Returns:
            Dictionary with mean, median, min, max, stddev
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "stddev": 0.0
            }

        values = self.metrics[operation]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0
        }

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.start_times.clear()


class MemoryProfiler:
    """Profile memory usage."""

    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots: List[Dict[str, int]] = []

    def take_snapshot(self, label: str = ""):
        """Take a memory snapshot.

        Args:
            label: Optional label for snapshot
        """
        # Get current memory usage estimate
        # In real implementation, would use tracemalloc or similar
        snapshot = {
            "label": label,
            "timestamp": time.time(),
            "rules_count": 0,  # Would track actual count
            "estimated_bytes": 0  # Would track actual memory
        }
        self.snapshots.append(snapshot)

    def get_peak_usage(self) -> int:
        """Get peak memory usage.

        Returns:
            Peak memory in bytes
        """
        if not self.snapshots:
            return 0

        return max(s["estimated_bytes"] for s in self.snapshots)

    def get_current_usage(self) -> int:
        """Get current memory usage estimate.

        Returns:
            Current memory in bytes
        """
        if not self.snapshots:
            return 0

        return self.snapshots[-1]["estimated_bytes"]


@pytest.fixture
def performance_monitor():
    """Provide performance monitor."""
    return PerformanceMonitor()


@pytest.fixture
def memory_profiler():
    """Provide memory profiler."""
    return MemoryProfiler()


@pytest.fixture
async def fast_memory_manager():
    """Provide fast memory manager for performance tests."""
    manager = AsyncMock(spec=MemoryManager)
    manager._rules = []

    async def add_rule(rule: MemoryRule):
        manager._rules.append(rule)

    async def get_rules():
        return manager._rules.copy()

    async def delete_rule(rule_id: str):
        manager._rules = [r for r in manager._rules if r.id != rule_id]

    manager.add_rule = AsyncMock(side_effect=add_rule)
    manager.get_rules = AsyncMock(side_effect=get_rules)
    manager.delete_rule = AsyncMock(side_effect=delete_rule)
    manager.initialize = AsyncMock()

    await manager.initialize()
    return manager


def create_test_rule(
    index: int,
    category: MemoryCategory = MemoryCategory.BEHAVIOR,
    authority: AuthorityLevel = AuthorityLevel.DEFAULT
) -> MemoryRule:
    """Create a test rule.

    Args:
        index: Rule index
        category: Rule category
        authority: Authority level

    Returns:
        Test rule
    """
    return MemoryRule(
        rule=f"Test rule {index}",
        category=category,
        authority=authority,
        id=f"perf_test_rule_{index}",
        source="performance_test",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
@pytest.mark.performance
class TestRuleInjectionPerformance:
    """Test performance of rule injection operations."""

    async def test_single_rule_injection_latency(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test latency of injecting a single rule."""
        rule = create_test_rule(1)

        # Measure injection time
        performance_monitor.start_timer("single_injection")
        await fast_memory_manager.add_rule(rule)
        duration = performance_monitor.stop_timer("single_injection")

        # Verify rule was added
        rules = await fast_memory_manager.get_rules()
        assert len(rules) == 1

        # Latency should be very low for single rule
        assert duration < 0.1  # 100ms threshold

    async def test_bulk_rule_injection_performance(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test performance of bulk rule injection."""
        num_rules = 100

        # Measure bulk injection
        performance_monitor.start_timer("bulk_injection")
        for i in range(num_rules):
            rule = create_test_rule(i)
            await fast_memory_manager.add_rule(rule)
        duration = performance_monitor.stop_timer("bulk_injection")

        # Verify all rules added
        rules = await fast_memory_manager.get_rules()
        assert len(rules) == num_rules

        # Calculate per-rule average
        avg_per_rule = duration / num_rules

        # Should maintain low per-rule latency even in bulk
        assert avg_per_rule < 0.01  # 10ms per rule average

    async def test_rule_retrieval_performance(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test performance of retrieving rules."""
        # Add test rules
        num_rules = 50
        for i in range(num_rules):
            await fast_memory_manager.add_rule(create_test_rule(i))

        # Measure retrieval time
        performance_monitor.start_timer("retrieval")
        rules = await fast_memory_manager.get_rules()
        duration = performance_monitor.stop_timer("retrieval")

        assert len(rules) == num_rules

        # Retrieval should be fast even with many rules
        assert duration < 0.1  # 100ms threshold


@pytest.mark.asyncio
@pytest.mark.performance
class TestLargeRuleSetHandling:
    """Test system behavior with large rule sets."""

    async def test_large_rule_set_performance(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test performance with large number of rules."""
        large_count = 500

        # Add many rules
        for i in range(large_count):
            await fast_memory_manager.add_rule(create_test_rule(i))

        # Measure retrieval with large set
        performance_monitor.start_timer("large_retrieval")
        rules = await fast_memory_manager.get_rules()
        duration = performance_monitor.stop_timer("large_retrieval")

        assert len(rules) == large_count

        # Should handle large sets efficiently
        assert duration < 0.5  # 500ms threshold for 500 rules

    async def test_rule_filtering_performance(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test filtering performance with large rule set."""
        # Add mixed rules
        for i in range(200):
            category = MemoryCategory.BEHAVIOR if i % 2 == 0 else MemoryCategory.PREFERENCE
            await fast_memory_manager.add_rule(create_test_rule(i, category=category))

        # Measure filtering
        performance_monitor.start_timer("filtering")
        rules = await fast_memory_manager.get_rules()
        behavior_rules = [r for r in rules if r.category == MemoryCategory.BEHAVIOR]
        duration = performance_monitor.stop_timer("filtering")

        assert len(behavior_rules) == 100

        # Filtering should be fast
        assert duration < 0.2  # 200ms threshold


@pytest.mark.asyncio
@pytest.mark.performance
class TestResponseTimeImpact:
    """Test impact of rules on LLM response times."""

    async def test_response_time_with_no_rules(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Baseline: response time with no rules."""
        provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=provider,
            memory_manager=fast_memory_manager,
            mode=ExecutionMode.MOCK
        )

        # Measure response time
        performance_monitor.start_timer("no_rules_response")
        metrics, with_rules, without_rules = await harness.run_behavioral_test(
            prompt="Test prompt",
            rules=[]
        )
        duration = performance_monitor.stop_timer("no_rules_response")

        assert metrics is not None
        assert with_rules is not None

        # Baseline should be fast
        assert duration < 0.5

    async def test_response_time_with_few_rules(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test response time with small rule set."""
        # Add a few rules
        rules = [create_test_rule(i) for i in range(5)]
        for rule in rules:
            await fast_memory_manager.add_rule(rule)

        provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=provider,
            memory_manager=fast_memory_manager,
            mode=ExecutionMode.MOCK
        )

        # Measure response time
        performance_monitor.start_timer("few_rules_response")
        metrics, with_rules, without_rules = await harness.run_behavioral_test(
            prompt="Test prompt",
            rules=rules
        )
        duration = performance_monitor.stop_timer("few_rules_response")

        assert metrics is not None
        assert with_rules is not None

        # Should have minimal overhead
        assert duration < 0.5

    async def test_response_time_scaling(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test how response time scales with rule count."""
        provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=provider,
            memory_manager=fast_memory_manager,
            mode=ExecutionMode.MOCK
        )

        rule_counts = [0, 5, 10, 25, 50]
        response_times = []

        for count in rule_counts:
            # Clear previous rules
            fast_memory_manager._rules.clear()

            # Add rules for this count
            rules = [create_test_rule(i) for i in range(count)]
            for rule in rules:
                await fast_memory_manager.add_rule(rule)

            # Measure response time
            performance_monitor.start_timer(f"scaling_{count}")
            await harness.run_behavioral_test(
                prompt="Test prompt",
                rules=rules
            )
            duration = performance_monitor.stop_timer(f"scaling_{count}")
            response_times.append(duration)

        # Response time should scale sub-linearly
        # (not grow proportionally with rule count)
        if len(response_times) > 2:
            # Ratio of time increase should be less than ratio of count increase
            time_ratio = response_times[-1] / max(response_times[0], 0.001)
            count_ratio = rule_counts[-1] / max(rule_counts[0], 1)

            # Time should not grow as fast as count (allow some overhead)
            assert time_ratio < count_ratio * 2  # Allow 2x factor for overhead


@pytest.mark.asyncio
@pytest.mark.stress
class TestConcurrentOperations:
    """Test system stability under concurrent rule operations."""

    async def test_concurrent_rule_additions(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test concurrent rule additions."""
        num_concurrent = 20

        async def add_rule_batch(start_idx: int):
            """Add a batch of rules."""
            for i in range(10):
                rule = create_test_rule(start_idx * 10 + i)
                await fast_memory_manager.add_rule(rule)

        # Execute concurrent additions
        performance_monitor.start_timer("concurrent_additions")
        await asyncio.gather(*[
            add_rule_batch(i) for i in range(num_concurrent)
        ])
        duration = performance_monitor.stop_timer("concurrent_additions")

        # Verify all rules added
        rules = await fast_memory_manager.get_rules()
        assert len(rules) == num_concurrent * 10

        # Should handle concurrency efficiently
        assert duration < 2.0  # 2 second threshold

    async def test_concurrent_read_write(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Test concurrent reads and writes."""
        # Add initial rules
        for i in range(50):
            await fast_memory_manager.add_rule(create_test_rule(i))

        async def read_rules():
            """Read rules repeatedly."""
            for _ in range(10):
                await fast_memory_manager.get_rules()

        async def write_rules(start_idx: int):
            """Write rules repeatedly."""
            for i in range(5):
                rule = create_test_rule(start_idx * 100 + i)
                await fast_memory_manager.add_rule(rule)

        # Execute mixed concurrent operations
        performance_monitor.start_timer("concurrent_read_write")
        await asyncio.gather(
            read_rules(),
            read_rules(),
            write_rules(1),
            read_rules(),
            write_rules(2),
        )
        duration = performance_monitor.stop_timer("concurrent_read_write")

        # Should handle mixed operations
        assert duration < 1.0


@pytest.mark.asyncio
@pytest.mark.performance
class TestMemoryUsage:
    """Test memory footprint of rule system."""

    async def test_memory_footprint_small_set(
        self,
        fast_memory_manager,
        memory_profiler
    ):
        """Test memory usage with small rule set."""
        memory_profiler.take_snapshot("before")

        # Add small set of rules
        for i in range(10):
            await fast_memory_manager.add_rule(create_test_rule(i))

        memory_profiler.take_snapshot("after")

        # Memory growth should be reasonable
        # (actual values would be measured with real profiler)
        rules = await fast_memory_manager.get_rules()
        assert len(rules) == 10

    async def test_memory_footprint_large_set(
        self,
        fast_memory_manager,
        memory_profiler
    ):
        """Test memory usage with large rule set."""
        memory_profiler.take_snapshot("before_large")

        # Add large set of rules
        for i in range(500):
            await fast_memory_manager.add_rule(create_test_rule(i))

        memory_profiler.take_snapshot("after_large")

        rules = await fast_memory_manager.get_rules()
        assert len(rules) == 500


@pytest.mark.asyncio
@pytest.mark.performance
class TestTokenBudgetAdherence:
    """Test adherence to token budgets."""

    async def test_rule_token_estimation(self):
        """Test token estimation for rules."""
        rule = create_test_rule(1)

        # Estimate tokens (rough approximation: 1 token per 4 characters)
        estimated_tokens = len(rule.rule) // 4

        # Should be reasonable
        assert estimated_tokens > 0
        assert estimated_tokens < 100  # Single rule shouldn't use many tokens

    async def test_total_token_budget(
        self,
        fast_memory_manager
    ):
        """Test total token usage stays within budget."""
        # Add rules up to a budget
        max_tokens = 1000
        current_tokens = 0
        rules_added = 0

        while current_tokens < max_tokens:
            rule = create_test_rule(rules_added)
            estimated_tokens = len(rule.rule) // 4

            if current_tokens + estimated_tokens > max_tokens:
                break

            await fast_memory_manager.add_rule(rule)
            current_tokens += estimated_tokens
            rules_added += 1

        # Should stay within budget
        assert current_tokens <= max_tokens

        # Should have added reasonable number of rules
        rules = await fast_memory_manager.get_rules()
        assert len(rules) == rules_added
        assert len(rules) > 0


@pytest.mark.asyncio
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarks."""

    async def test_end_to_end_benchmark(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """End-to-end performance benchmark."""
        # Setup: Add initial rules
        for i in range(20):
            await fast_memory_manager.add_rule(create_test_rule(i))

        provider = MockLLMProvider()
        harness = LLMBehavioralHarness(
            provider=provider,
            memory_manager=fast_memory_manager,
            mode=ExecutionMode.MOCK
        )

        # Benchmark full workflow
        performance_monitor.start_timer("e2e_benchmark")

        # 1. Add new rule
        new_rule = create_test_rule(100)
        await fast_memory_manager.add_rule(new_rule)

        # 2. Retrieve all rules
        rules = await fast_memory_manager.get_rules()

        # 3. Run behavioral test with rules
        await harness.run_behavioral_test(
            prompt="Test prompt",
            rules=rules
        )

        duration = performance_monitor.stop_timer("e2e_benchmark")

        # Full workflow should complete quickly
        assert duration < 1.0

        # Verify correctness
        assert len(rules) == 21

    async def test_performance_regression_check(
        self,
        fast_memory_manager,
        performance_monitor
    ):
        """Check for performance regressions."""
        # This test would compare against baseline metrics
        # For now, just verify operations complete in reasonable time

        operations = [
            ("add_10_rules", lambda: asyncio.gather(*[
                fast_memory_manager.add_rule(create_test_rule(i))
                for i in range(10)
            ])),
            ("get_rules", lambda: fast_memory_manager.get_rules()),
        ]

        for op_name, op_func in operations:
            performance_monitor.start_timer(op_name)
            await op_func()
            duration = performance_monitor.stop_timer(op_name)

            # Should complete in reasonable time
            assert duration < 1.0

            # Get statistics
            stats = performance_monitor.get_statistics(op_name)
            assert stats["count"] == 1
            assert stats["mean"] < 1.0
