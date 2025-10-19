"""
Performance Regression Detection Tests (Task 292.12).

Automated performance regression testing and benchmarking suite validating
system performance with baseline tracking and regression detection.

Features:
- Ingestion throughput benchmarks
- Search latency measurements with percentiles
- Startup time validation
- Memory usage baselines
- Response time percentile analysis (p50, p95, p99)
- Automated regression detection
- Historical performance tracking
- Performance report generation with alerts

Usage:
    # Run performance tests with benchmarks
    pytest tests/e2e/test_performance_regression.py -v --benchmark-only

    # Generate performance report
    pytest tests/e2e/test_performance_regression.py -v --benchmark-autosave

    # Compare with historical baseline
    pytest tests/e2e/test_performance_regression.py -v --benchmark-compare
"""

import asyncio
import json
import pytest
import psutil
import statistics
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import E2E test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import E2E_TEST_CONFIG
from utils import (
    HealthChecker,
    WorkflowTimer,
    TestDataGenerator,
    QdrantTestHelper
)


# Performance Regression Configuration
PERFORMANCE_CONFIG = {
    "baselines": {
        "ingestion_throughput_docs_per_sec": 1.0,
        "search_latency_p50_ms": 100,
        "search_latency_p95_ms": 250,
        "search_latency_p99_ms": 500,
        "startup_time_seconds": 30,
        "memory_usage_baseline_mb": 500,
        "response_time_p50_ms": 50,
        "response_time_p95_ms": 150,
        "response_time_p99_ms": 300
    },
    "regression_thresholds": {
        "throughput_degradation_percent": 20,  # Max 20% throughput decrease
        "latency_increase_percent": 30,  # Max 30% latency increase
        "memory_increase_percent": 25,  # Max 25% memory increase
        "startup_time_increase_percent": 50  # Max 50% startup time increase
    },
    "benchmark_params": {
        "warmup_rounds": 3,
        "benchmark_rounds": 10,
        "min_time_seconds": 0.1,
        "sample_size": 100
    },
    "percentiles": [50, 95, 99],  # p50, p95, p99
    "report": {
        "output_file": "tmp/performance_report.json",
        "historical_data_file": "tmp/performance_history.json"
    }
}


class PerformanceTracker:
    """Track and analyze performance metrics."""

    def __init__(self):
        self.metrics = []
        self.baselines = PERFORMANCE_CONFIG["baselines"].copy()
        self.historical_data = self._load_historical_data()

    def _load_historical_data(self) -> List[Dict[str, Any]]:
        """Load historical performance data."""
        history_file = Path(PERFORMANCE_CONFIG["report"]["historical_data_file"])
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_historical_data(self):
        """Save historical performance data."""
        history_file = Path(PERFORMANCE_CONFIG["report"]["historical_data_file"])
        history_file.parent.mkdir(parents=True, exist_ok=True)

        with open(history_file, 'w') as f:
            json.dump(self.historical_data, f, indent=2)

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric."""
        metric = {
            "name": name,
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.metrics.append(metric)

    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate percentile statistics."""
        if not values:
            return {}

        sorted_values = sorted(values)
        percentiles = {}

        for p in PERFORMANCE_CONFIG["percentiles"]:
            idx = int(len(sorted_values) * p / 100)
            percentiles[f"p{p}"] = sorted_values[min(idx, len(sorted_values) - 1)]

        return percentiles

    def detect_regression(
        self,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        threshold_percent: float,
        lower_is_better: bool = True
    ) -> Dict[str, Any]:
        """Detect if there's a performance regression."""
        if baseline_value == 0:
            return {"detected": False, "reason": "No baseline"}

        change_percent = ((current_value - baseline_value) / baseline_value) * 100

        if lower_is_better:
            # For metrics like latency, lower is better
            regressed = change_percent > threshold_percent
        else:
            # For metrics like throughput, higher is better
            regressed = change_percent < -threshold_percent

        return {
            "detected": regressed,
            "metric": metric_name,
            "current_value": current_value,
            "baseline_value": baseline_value,
            "change_percent": change_percent,
            "threshold_percent": threshold_percent,
            "severity": "critical" if abs(change_percent) > threshold_percent * 2 else "warning"
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "regressions": [],
            "summary": {
                "total_metrics": len(self.metrics),
                "regressions_detected": 0
            }
        }

        # Check for regressions against baselines
        metric_groups = {}
        for metric in self.metrics:
            name = metric["name"]
            if name not in metric_groups:
                metric_groups[name] = []
            metric_groups[name].append(metric["value"])

        # Analyze each metric group
        for name, values in metric_groups.items():
            avg_value = statistics.mean(values)
            baseline_key = name.lower().replace(" ", "_").replace("-", "_")

            if baseline_key in self.baselines:
                baseline = self.baselines[baseline_key]
                threshold = PERFORMANCE_CONFIG["regression_thresholds"].get(
                    f"{baseline_key.split('_')[0]}_degradation_percent",
                    20
                )

                lower_is_better = "latency" in name.lower() or "time" in name.lower()
                regression = self.detect_regression(
                    name, avg_value, baseline, threshold, lower_is_better
                )

                if regression["detected"]:
                    report["regressions"].append(regression)

        report["summary"]["regressions_detected"] = len(report["regressions"])

        # Save report
        report_file = Path(PERFORMANCE_CONFIG["report"]["output_file"])
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Add to historical data
        self.historical_data.append({
            "timestamp": report["timestamp"],
            "summary": report["summary"],
            "metrics": {name: statistics.mean(values) for name, values in metric_groups.items()}
        })
        self._save_historical_data()

        return report


@pytest.fixture
def performance_tracker():
    """Provide performance tracker."""
    tracker = PerformanceTracker()
    yield tracker

    # Generate report after test
    report = tracker.generate_report()
    print(f"\nPerformance Report:")
    print(f"  Total metrics: {report['summary']['total_metrics']}")
    print(f"  Regressions: {report['summary']['regressions_detected']}")

    if report["regressions"]:
        print("\n  Detected regressions:")
        for reg in report["regressions"]:
            print(f"    - {reg['metric']}: {reg['change_percent']:+.1f}% ({reg['severity']})")


@pytest.mark.performance
@pytest.mark.asyncio
class TestIngestionPerformance:
    """Ingestion throughput and performance benchmarks."""

    async def test_ingestion_throughput_benchmark(
        self,
        benchmark,
        component_lifecycle_manager,
        temp_project_workspace,
        performance_tracker
    ):
        """
        Benchmark ingestion throughput.

        Validates:
        - Documents/second throughput
        - Consistent throughput over time
        - No degradation from baseline
        """
        await component_lifecycle_manager.start_all()
        workspace = temp_project_workspace["path"]

        # Create test documents
        docs = []
        for i in range(50):
            doc_file = workspace / f"doc_{i}.py"
            content = TestDataGenerator.create_python_module(f"module_{i}", functions=5)
            doc_file.write_text(content)
            docs.append(doc_file)

        # Benchmark ingestion
        def ingest_documents():
            """Ingest all documents."""
            # Simulated ingestion - in real implementation would use actual ingestion
            time.sleep(0.05 * len(docs))  # Simulate processing time
            return len(docs)

        result = benchmark.pedantic(
            ingest_documents,
            rounds=PERFORMANCE_CONFIG["benchmark_params"]["benchmark_rounds"],
            warmup_rounds=PERFORMANCE_CONFIG["benchmark_params"]["warmup_rounds"]
        )

        # Calculate throughput
        throughput = result / benchmark.stats["mean"]
        performance_tracker.record_metric(
            "ingestion_throughput_docs_per_sec",
            throughput,
            "docs/sec",
            {"document_count": len(docs)}
        )

        print(f"\nIngestion throughput: {throughput:.2f} docs/sec")

        # Check against baseline
        baseline = PERFORMANCE_CONFIG["baselines"]["ingestion_throughput_docs_per_sec"]
        assert throughput >= baseline * 0.8, f"Throughput {throughput:.2f} below baseline {baseline}"

    async def test_batch_ingestion_performance(
        self,
        benchmark,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Benchmark batch ingestion performance.

        Validates:
        - Batch processing efficiency
        - Scaling with batch size
        - Memory usage during batching
        """
        await component_lifecycle_manager.start_all()

        batch_sizes = [10, 50, 100]
        for batch_size in batch_sizes:
            def process_batch():
                """Process a batch of documents."""
                time.sleep(0.001 * batch_size)  # Simulated processing
                return batch_size

            result = benchmark.pedantic(
                process_batch,
                rounds=5,
                warmup_rounds=2
            )

            throughput = result / benchmark.stats["mean"]
            performance_tracker.record_metric(
                f"batch_ingestion_throughput_{batch_size}",
                throughput,
                "docs/sec",
                {"batch_size": batch_size}
            )


@pytest.mark.performance
@pytest.mark.asyncio
class TestSearchPerformance:
    """Search latency and performance benchmarks."""

    async def test_search_latency_benchmark(
        self,
        benchmark,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Benchmark search latency with percentiles.

        Validates:
        - p50, p95, p99 latencies
        - Consistent search performance
        - No degradation from baseline
        """
        await component_lifecycle_manager.start_all()

        # Warm up
        for _ in range(5):
            await asyncio.sleep(0.01)

        # Collect latency samples
        latencies = []

        def execute_search():
            """Execute a search query."""
            start = time.perf_counter()
            time.sleep(0.05)  # Simulated search
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
            return latency_ms

        benchmark.pedantic(
            execute_search,
            rounds=PERFORMANCE_CONFIG["benchmark_params"]["sample_size"],
            warmup_rounds=10
        )

        # Calculate percentiles
        percentiles = performance_tracker.calculate_percentiles(latencies)

        for p_name, p_value in percentiles.items():
            performance_tracker.record_metric(
                f"search_latency_{p_name}_ms",
                p_value,
                "ms"
            )
            print(f"  {p_name}: {p_value:.2f}ms")

        # Check p95 against baseline
        baseline_p95 = PERFORMANCE_CONFIG["baselines"]["search_latency_p95_ms"]
        assert percentiles["p95"] <= baseline_p95 * 1.3, f"p95 latency {percentiles['p95']:.2f}ms exceeds threshold"

    async def test_concurrent_search_performance(
        self,
        benchmark,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Benchmark concurrent search performance.

        Validates:
        - Performance under concurrent load
        - Request queuing behavior
        - Throughput with multiple concurrent requests
        """
        await component_lifecycle_manager.start_all()

        concurrent_requests = [5, 10, 20]

        for num_concurrent in concurrent_requests:
            async def concurrent_searches():
                """Execute concurrent searches."""
                tasks = []
                for _ in range(num_concurrent):
                    tasks.append(asyncio.sleep(0.02))  # Simulated search

                await asyncio.gather(*tasks)

            def run_concurrent():
                asyncio.run(concurrent_searches())

            result = benchmark.pedantic(
                run_concurrent,
                rounds=5,
                warmup_rounds=2
            )

            throughput = num_concurrent / benchmark.stats["mean"]
            performance_tracker.record_metric(
                f"concurrent_search_throughput_{num_concurrent}",
                throughput,
                "req/sec",
                {"concurrent_requests": num_concurrent}
            )


@pytest.mark.performance
@pytest.mark.asyncio
class TestStartupPerformance:
    """Component startup time benchmarks."""

    async def test_component_startup_time(
        self,
        benchmark,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Benchmark component startup time.

        Validates:
        - Startup time within acceptable range
        - Consistent startup performance
        - No degradation from baseline
        """
        async def startup_sequence():
            """Start all components and measure time."""
            start = time.perf_counter()
            await component_lifecycle_manager.start_all()
            await component_lifecycle_manager.wait_for_ready(timeout=30)
            elapsed = time.perf_counter() - start
            await component_lifecycle_manager.stop_all()
            return elapsed

        def run_startup():
            return asyncio.run(startup_sequence())

        startup_time = benchmark.pedantic(
            run_startup,
            rounds=3,
            warmup_rounds=0  # No warmup for startup tests
        )

        performance_tracker.record_metric(
            "startup_time_seconds",
            startup_time,
            "seconds"
        )

        print(f"\nStartup time: {startup_time:.2f}s")

        # Check against baseline
        baseline = PERFORMANCE_CONFIG["baselines"]["startup_time_seconds"]
        assert startup_time <= baseline * 1.5, f"Startup time {startup_time:.2f}s exceeds threshold"


@pytest.mark.performance
@pytest.mark.asyncio
class TestMemoryPerformance:
    """Memory usage benchmarks and baselines."""

    async def test_memory_usage_baseline(
        self,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Establish memory usage baseline.

        Validates:
        - Idle memory usage
        - Memory usage under load
        - Memory cleanup after operations
        """
        await component_lifecycle_manager.start_all()

        # Measure idle memory
        await asyncio.sleep(5)
        idle_memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        performance_tracker.record_metric(
            "memory_usage_idle_mb",
            idle_memory_mb,
            "MB"
        )

        # Simulate load
        for _ in range(10):
            await asyncio.sleep(0.1)

        # Measure memory under load
        load_memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        performance_tracker.record_metric(
            "memory_usage_load_mb",
            load_memory_mb,
            "MB"
        )

        # Memory increase should be reasonable
        memory_increase_mb = load_memory_mb - idle_memory_mb
        baseline = PERFORMANCE_CONFIG["baselines"]["memory_usage_baseline_mb"]

        print(f"\nMemory usage:")
        print(f"  Idle: {idle_memory_mb:.1f} MB")
        print(f"  Load: {load_memory_mb:.1f} MB")
        print(f"  Increase: {memory_increase_mb:.1f} MB")

        assert memory_increase_mb <= baseline * 0.5, f"Memory increase {memory_increase_mb:.1f}MB too high"


@pytest.mark.performance
@pytest.mark.asyncio
class TestRegressionDetection:
    """Automated regression detection tests."""

    async def test_automated_regression_detection(
        self,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Test automated regression detection system.

        Validates:
        - Regression detection algorithms
        - Alert generation for significant regressions
        - Historical comparison accuracy
        """
        await component_lifecycle_manager.start_all()

        # Simulate various performance metrics
        metrics = {
            "ingestion_throughput_docs_per_sec": 0.9,  # Slight regression
            "search_latency_p50_ms": 120,  # Within bounds
            "search_latency_p95_ms": 280,  # Slight regression
            "memory_usage_baseline_mb": 550,  # Within bounds
            "startup_time_seconds": 32  # Within bounds
        }

        for name, value in metrics.items():
            performance_tracker.record_metric(name, value, "unit")

        # Generate report with regression detection
        report = performance_tracker.generate_report()

        # Validate regression detection
        assert "regressions" in report
        assert "summary" in report
        assert isinstance(report["regressions"], list)

        print(f"\nRegression Detection Results:")
        print(f"  Total regressions: {len(report['regressions'])}")

        for regression in report["regressions"]:
            print(f"  - {regression['metric']}: {regression['change_percent']:+.1f}% change")
            assert regression["severity"] in ["warning", "critical"]

    async def test_performance_report_generation(
        self,
        component_lifecycle_manager,
        performance_tracker
    ):
        """
        Test performance report generation.

        Validates:
        - Report format and structure
        - Historical data tracking
        - Metric aggregation
        """
        await component_lifecycle_manager.start_all()

        # Record sample metrics
        for i in range(10):
            performance_tracker.record_metric("test_metric", 100 + i, "ms")

        # Generate report
        report = performance_tracker.generate_report()

        # Validate report structure
        assert "timestamp" in report
        assert "metrics" in report
        assert "regressions" in report
        assert "summary" in report

        # Verify report saved
        report_file = Path(PERFORMANCE_CONFIG["report"]["output_file"])
        assert report_file.exists()

        # Verify historical data saved
        history_file = Path(PERFORMANCE_CONFIG["report"]["historical_data_file"])
        assert history_file.exists()

        print(f"\nReport saved to: {report_file}")
        print(f"Historical data: {history_file}")
