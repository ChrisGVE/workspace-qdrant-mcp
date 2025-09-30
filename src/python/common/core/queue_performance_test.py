"""
Queue Performance Testing and Benchmarking

Comprehensive performance test suite for SQLite queue system targeting 1000+ docs/min
throughput with concurrent access from daemon, MCP server, and CLI.

Features:
    - Throughput benchmarks (enqueue, dequeue, batch operations)
    - Concurrent access simulation (multiple processes/threads)
    - Latency measurements with percentiles
    - WAL checkpoint performance testing
    - Memory usage profiling
    - Query optimization with EXPLAIN QUERY PLAN
    - Performance regression detection

Usage:
    ```python
    # Run full test suite
    python -m workspace_qdrant_mcp.core.queue_performance_test

    # Run specific benchmark
    python -m workspace_qdrant_mcp.core.queue_performance_test --benchmark throughput
    ```
"""

import asyncio
import multiprocessing
import os
import statistics
import time
import tracemalloc
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from .queue_client import QueueOperation, SQLiteQueueClient
from .queue_connection import QueueConnectionPool


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""

    operation: str
    total_operations: int
    duration_seconds: float
    throughput_per_second: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_max_ms: float
    memory_peak_mb: float
    errors: int


@dataclass
class PerformanceTarget:
    """Performance target thresholds."""

    # Throughput targets
    min_throughput_per_second: float = 17.0  # 1000 docs/min = 16.67/sec

    # Latency targets (milliseconds)
    max_mean_latency_ms: float = 50.0
    max_p95_latency_ms: float = 100.0
    max_p99_latency_ms: float = 200.0

    # Memory targets
    max_memory_mb: float = 256.0

    # Error tolerance
    max_error_rate: float = 0.01  # 1%


class QueuePerformanceTester:
    """
    Comprehensive performance testing suite for queue operations.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        target: Optional[PerformanceTarget] = None
    ):
        """
        Initialize performance tester.

        Args:
            db_path: Database path (temp if None)
            target: Performance target thresholds
        """
        if db_path is None:
            import tempfile
            self.temp_dir = tempfile.mkdtemp(prefix="queue_perf_test_")
            self.db_path = os.path.join(self.temp_dir, "test_queue.db")
        else:
            self.db_path = db_path
            self.temp_dir = None

        self.target = target or PerformanceTarget()
        self.results: List[PerformanceMetrics] = []

    async def setup(self):
        """Initialize test environment."""
        # Create connection pool
        self.pool = QueueConnectionPool(self.db_path)
        await self.pool.initialize()

        # Create queue client
        self.client = SQLiteQueueClient(self.pool)

        # Initialize schema
        schema_path = Path(__file__).parent / "queue_schema.sql"
        schema_sql = schema_path.read_text()

        async with self.pool.get_connection_async() as conn:
            conn.executescript(schema_sql)
            conn.commit()

        logger.info(f"Performance test environment initialized: {self.db_path}")

    async def teardown(self):
        """Clean up test environment."""
        await self.pool.close()

        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temp directory: {self.temp_dir}")

    async def benchmark_enqueue(
        self,
        num_items: int = 1000,
        batch_size: int = 1
    ) -> PerformanceMetrics:
        """
        Benchmark enqueue operations.

        Args:
            num_items: Number of items to enqueue
            batch_size: Batch size (1 = individual enqueue)

        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking enqueue: {num_items} items, batch_size={batch_size}")

        latencies = []
        errors = 0

        tracemalloc.start()
        start_time = time.time()

        for i in range(0, num_items, batch_size):
            batch_start = time.time()

            try:
                if batch_size == 1:
                    # Individual enqueue
                    await self.client.enqueue_file(
                        file_path=f"/test/file_{i}.txt",
                        collection="test-collection",
                        priority=5,
                        operation=QueueOperation.INGEST
                    )
                else:
                    # Batch enqueue
                    batch_items = [
                        {
                            "file_path": f"/test/file_{j}.txt",
                            "collection": "test-collection",
                            "priority": 5,
                            "operation": QueueOperation.INGEST
                        }
                        for j in range(i, min(i + batch_size, num_items))
                    ]
                    await self.client.enqueue_batch(batch_items)

                latencies.append((time.time() - batch_start) * 1000)

            except Exception as e:
                logger.error(f"Enqueue error: {e}")
                errors += 1

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        throughput = num_items / duration if duration > 0 else 0

        metrics = PerformanceMetrics(
            operation="enqueue",
            total_operations=num_items,
            duration_seconds=duration,
            throughput_per_second=throughput,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=self._percentile(latencies, 0.95) if latencies else 0,
            latency_p99_ms=self._percentile(latencies, 0.99) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            memory_peak_mb=peak / (1024 * 1024),
            errors=errors
        )

        self.results.append(metrics)
        self._log_metrics(metrics)

        return metrics

    async def benchmark_dequeue(
        self,
        num_items: int = 1000,
        batch_size: int = 10
    ) -> PerformanceMetrics:
        """
        Benchmark dequeue operations.

        Args:
            num_items: Number of items to enqueue first
            batch_size: Dequeue batch size

        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking dequeue: {num_items} items, batch_size={batch_size}")

        # Enqueue test data first
        await self.benchmark_enqueue(num_items, batch_size=100)

        latencies = []
        errors = 0
        total_dequeued = 0

        tracemalloc.start()
        start_time = time.time()

        while total_dequeued < num_items:
            batch_start = time.time()

            try:
                items = await self.client.dequeue_batch(batch_size)
                latencies.append((time.time() - batch_start) * 1000)
                total_dequeued += len(items)

                if not items:
                    break

            except Exception as e:
                logger.error(f"Dequeue error: {e}")
                errors += 1
                break

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        duration = end_time - start_time
        throughput = total_dequeued / duration if duration > 0 else 0

        metrics = PerformanceMetrics(
            operation="dequeue",
            total_operations=total_dequeued,
            duration_seconds=duration,
            throughput_per_second=throughput,
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=self._percentile(latencies, 0.95) if latencies else 0,
            latency_p99_ms=self._percentile(latencies, 0.99) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            memory_peak_mb=peak / (1024 * 1024),
            errors=errors
        )

        self.results.append(metrics)
        self._log_metrics(metrics)

        return metrics

    async def benchmark_concurrent_access(
        self,
        num_processes: int = 3,
        operations_per_process: int = 100
    ) -> PerformanceMetrics:
        """
        Benchmark concurrent access from multiple processes.

        Args:
            num_processes: Number of concurrent processes
            operations_per_process: Operations per process

        Returns:
            Aggregate performance metrics
        """
        logger.info(
            f"Benchmarking concurrent access: {num_processes} processes, "
            f"{operations_per_process} ops each"
        )

        # Use asyncio tasks to simulate concurrent access
        # (In real system, would use multiprocessing)

        async def worker(worker_id: int):
            latencies = []
            errors = 0

            for i in range(operations_per_process):
                start = time.time()

                try:
                    # Alternate between enqueue and dequeue
                    if i % 2 == 0:
                        await self.client.enqueue_file(
                            file_path=f"/test/worker_{worker_id}_file_{i}.txt",
                            collection="test-collection",
                            priority=5
                        )
                    else:
                        await self.client.dequeue_batch(1)

                    latencies.append((time.time() - start) * 1000)

                except Exception as e:
                    errors += 1

            return latencies, errors

        tracemalloc.start()
        start_time = time.time()

        # Run workers concurrently
        tasks = [worker(i) for i in range(num_processes)]
        results = await asyncio.gather(*tasks)

        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Aggregate results
        all_latencies = []
        total_errors = 0

        for latencies, errors in results:
            all_latencies.extend(latencies)
            total_errors += errors

        total_ops = num_processes * operations_per_process
        duration = end_time - start_time
        throughput = total_ops / duration if duration > 0 else 0

        metrics = PerformanceMetrics(
            operation="concurrent_access",
            total_operations=total_ops,
            duration_seconds=duration,
            throughput_per_second=throughput,
            latency_mean_ms=statistics.mean(all_latencies) if all_latencies else 0,
            latency_p50_ms=statistics.median(all_latencies) if all_latencies else 0,
            latency_p95_ms=self._percentile(all_latencies, 0.95) if all_latencies else 0,
            latency_p99_ms=self._percentile(all_latencies, 0.99) if all_latencies else 0,
            latency_max_ms=max(all_latencies) if all_latencies else 0,
            memory_peak_mb=peak / (1024 * 1024),
            errors=total_errors
        )

        self.results.append(metrics)
        self._log_metrics(metrics)

        return metrics

    async def benchmark_wal_checkpoint(
        self,
        num_operations: int = 1000
    ) -> PerformanceMetrics:
        """
        Benchmark WAL checkpoint performance.

        Args:
            num_operations: Number of operations before checkpoint

        Returns:
            Performance metrics
        """
        logger.info(f"Benchmarking WAL checkpoint: {num_operations} operations")

        # Enqueue items to generate WAL writes
        await self.benchmark_enqueue(num_operations, batch_size=100)

        latencies = []

        tracemalloc.start()

        # Perform checkpoints with different modes
        for mode in ["PASSIVE", "FULL"]:
            start = time.time()

            async with self.pool.get_connection_async() as conn:
                cursor = conn.execute(f"PRAGMA wal_checkpoint({mode})")
                result = cursor.fetchone()

            latency = (time.time() - start) * 1000
            latencies.append(latency)

            logger.debug(f"Checkpoint {mode}: {latency:.2f}ms, result={result}")

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = PerformanceMetrics(
            operation="wal_checkpoint",
            total_operations=len(latencies),
            duration_seconds=sum(latencies) / 1000,
            throughput_per_second=0,  # Not applicable
            latency_mean_ms=statistics.mean(latencies) if latencies else 0,
            latency_p50_ms=statistics.median(latencies) if latencies else 0,
            latency_p95_ms=self._percentile(latencies, 0.95) if latencies else 0,
            latency_p99_ms=self._percentile(latencies, 0.99) if latencies else 0,
            latency_max_ms=max(latencies) if latencies else 0,
            memory_peak_mb=peak / (1024 * 1024),
            errors=0
        )

        self.results.append(metrics)
        self._log_metrics(metrics)

        return metrics

    async def analyze_query_performance(self) -> Dict[str, Any]:
        """
        Analyze query performance with EXPLAIN QUERY PLAN.

        Returns:
            Dictionary of query analysis results
        """
        logger.info("Analyzing query performance")

        # Sample queries to analyze
        queries = {
            "dequeue_batch": """
                SELECT file_absolute_path, collection_name, tenant_id, branch,
                       operation, priority, queued_timestamp, retry_count
                FROM ingestion_queue
                ORDER BY priority DESC, queued_timestamp ASC
                LIMIT 10
            """,
            "get_stats": """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN priority >= 8 THEN 1 ELSE 0 END) as urgent
                FROM ingestion_queue
            """,
            "mark_error": """
                UPDATE ingestion_queue
                SET retry_count = retry_count + 1, error_message_id = 1
                WHERE file_absolute_path = '/test/file.txt'
            """,
        }

        analysis = {}

        async with self.pool.get_connection_async() as conn:
            for query_name, query in queries.items():
                cursor = conn.execute(f"EXPLAIN QUERY PLAN {query}")
                plan = cursor.fetchall()

                analysis[query_name] = {
                    "query": query.strip(),
                    "plan": [dict(row) for row in plan]
                }

                logger.debug(f"Query plan for {query_name}:")
                for row in plan:
                    logger.debug(f"  {dict(row)}")

        return analysis

    def validate_performance(self, metrics: PerformanceMetrics) -> Tuple[bool, List[str]]:
        """
        Validate metrics against performance targets.

        Args:
            metrics: Performance metrics to validate

        Returns:
            Tuple of (passed, failures)
        """
        failures = []

        # Check throughput (only for enqueue/dequeue)
        if metrics.operation in ["enqueue", "dequeue"]:
            if metrics.throughput_per_second < self.target.min_throughput_per_second:
                failures.append(
                    f"Throughput {metrics.throughput_per_second:.2f}/s < "
                    f"target {self.target.min_throughput_per_second:.2f}/s"
                )

        # Check latencies
        if metrics.latency_mean_ms > self.target.max_mean_latency_ms:
            failures.append(
                f"Mean latency {metrics.latency_mean_ms:.2f}ms > "
                f"target {self.target.max_mean_latency_ms:.2f}ms"
            )

        if metrics.latency_p95_ms > self.target.max_p95_latency_ms:
            failures.append(
                f"P95 latency {metrics.latency_p95_ms:.2f}ms > "
                f"target {self.target.max_p95_latency_ms:.2f}ms"
            )

        if metrics.latency_p99_ms > self.target.max_p99_latency_ms:
            failures.append(
                f"P99 latency {metrics.latency_p99_ms:.2f}ms > "
                f"target {self.target.max_p99_latency_ms:.2f}ms"
            )

        # Check memory
        if metrics.memory_peak_mb > self.target.max_memory_mb:
            failures.append(
                f"Peak memory {metrics.memory_peak_mb:.2f}MB > "
                f"target {self.target.max_memory_mb:.2f}MB"
            )

        # Check error rate
        error_rate = metrics.errors / metrics.total_operations if metrics.total_operations > 0 else 0
        if error_rate > self.target.max_error_rate:
            failures.append(
                f"Error rate {error_rate:.2%} > target {self.target.max_error_rate:.2%}"
            )

        passed = len(failures) == 0

        return passed, failures

    async def run_full_suite(self) -> Dict[str, Any]:
        """
        Run complete performance test suite.

        Returns:
            Test results summary
        """
        logger.info("Running full performance test suite")

        await self.setup()

        try:
            # Run all benchmarks
            enqueue_metrics = await self.benchmark_enqueue(1000, batch_size=1)
            batch_enqueue_metrics = await self.benchmark_enqueue(1000, batch_size=100)
            dequeue_metrics = await self.benchmark_dequeue(1000, batch_size=10)
            concurrent_metrics = await self.benchmark_concurrent_access(3, 100)
            wal_metrics = await self.benchmark_wal_checkpoint(1000)

            # Analyze queries
            query_analysis = await self.analyze_query_performance()

            # Validate all results
            validation_results = {}
            all_passed = True

            for metrics in self.results:
                passed, failures = self.validate_performance(metrics)
                validation_results[metrics.operation] = {
                    "passed": passed,
                    "failures": failures
                }

                if not passed:
                    all_passed = False
                    logger.warning(
                        f"Performance validation failed for {metrics.operation}: {failures}"
                    )

            # Summary
            summary = {
                "all_tests_passed": all_passed,
                "metrics": [self._metrics_to_dict(m) for m in self.results],
                "validation": validation_results,
                "query_analysis": query_analysis,
                "target": {
                    "min_throughput_per_second": self.target.min_throughput_per_second,
                    "max_mean_latency_ms": self.target.max_mean_latency_ms,
                    "max_p95_latency_ms": self.target.max_p95_latency_ms,
                    "max_p99_latency_ms": self.target.max_p99_latency_ms,
                    "max_memory_mb": self.target.max_memory_mb,
                    "max_error_rate": self.target.max_error_rate,
                }
            }

            logger.info(f"Performance test suite complete: {'PASSED' if all_passed else 'FAILED'}")

            return summary

        finally:
            await self.teardown()

    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]

    def _log_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        logger.info(
            f"Performance: {metrics.operation} - "
            f"{metrics.throughput_per_second:.2f} ops/s, "
            f"mean={metrics.latency_mean_ms:.2f}ms, "
            f"p95={metrics.latency_p95_ms:.2f}ms, "
            f"p99={metrics.latency_p99_ms:.2f}ms, "
            f"mem={metrics.memory_peak_mb:.2f}MB"
        )

    def _metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "operation": metrics.operation,
            "total_operations": metrics.total_operations,
            "duration_seconds": metrics.duration_seconds,
            "throughput_per_second": metrics.throughput_per_second,
            "latency_mean_ms": metrics.latency_mean_ms,
            "latency_p50_ms": metrics.latency_p50_ms,
            "latency_p95_ms": metrics.latency_p95_ms,
            "latency_p99_ms": metrics.latency_p99_ms,
            "latency_max_ms": metrics.latency_max_ms,
            "memory_peak_mb": metrics.memory_peak_mb,
            "errors": metrics.errors,
        }


async def main():
    """Run performance tests."""
    import json

    tester = QueuePerformanceTester()
    results = await tester.run_full_suite()

    # Print results
    print("\n" + "=" * 80)
    print("QUEUE PERFORMANCE TEST RESULTS")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print("=" * 80)

    # Exit with appropriate code
    exit(0 if results["all_tests_passed"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
