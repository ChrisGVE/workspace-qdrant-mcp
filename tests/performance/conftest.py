"""
Performance Testing Configuration and Fixtures.

Provides shared fixtures, configuration, and utilities for performance testing
across the workspace-qdrant-mcp project.
"""

import asyncio
import gc
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import pytest

# Import performance test configuration
from . import PERFORMANCE_TEST_CONFIG, REGRESSION_THRESHOLDS


def pytest_configure(config):
    """Configure pytest for performance testing."""

    # Register custom markers
    config.addinivalue_line(
        "markers",
        "performance: Performance and benchmark tests"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: Pytest-benchmark microbenchmarks"
    )
    config.addinivalue_line(
        "markers",
        "load_testing: K6 load testing scenarios"
    )
    config.addinivalue_line(
        "markers",
        "memory_profiling: Memory usage and leak detection tests"
    )
    config.addinivalue_line(
        "markers",
        "concurrency: Concurrent operation performance tests"
    )
    config.addinivalue_line(
        "markers",
        "scaling: Dataset scaling behavior tests"
    )
    config.addinivalue_line(
        "markers",
        "regression: Performance regression detection tests"
    )
    config.addinivalue_line(
        "markers",
        "monitoring: Real-time performance monitoring tests"
    )
    config.addinivalue_line(
        "markers",
        "requires_k6: Tests requiring k6 load testing tool"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance testing."""

    # Apply timeout to all performance tests
    default_timeout = PERFORMANCE_TEST_CONFIG["default_timeout"]

    for item in items:
        # Add timeout marker to performance tests
        if any(marker.name == "performance" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(default_timeout))

        # Skip k6 tests if k6 is not available (unless explicitly requested)
        if any(marker.name == "requires_k6" for marker in item.iter_markers()):
            if not _check_k6_available() and not config.getoption("--run-k6", default=False):
                item.add_marker(pytest.mark.skip(reason="k6 not available, use --run-k6 to force"))


def pytest_addoption(parser):
    """Add performance testing command line options."""

    parser.addoption(
        "--run-k6",
        action="store_true",
        default=False,
        help="Force running k6 load tests even if k6 is not available"
    )

    parser.addoption(
        "--perf-baseline",
        action="store",
        default=None,
        help="Path to performance baseline file for regression testing"
    )

    parser.addoption(
        "--perf-report-dir",
        action="store",
        default="performance_reports",
        help="Directory to store performance test reports"
    )

    parser.addoption(
        "--perf-iterations",
        action="store",
        type=int,
        default=PERFORMANCE_TEST_CONFIG["default_iterations"],
        help="Number of iterations for benchmark tests"
    )


def _check_k6_available() -> bool:
    """Check if k6 is available in the system."""
    try:
        import subprocess
        result = subprocess.run(["k6", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def performance_config():
    """Provide performance test configuration."""
    return PERFORMANCE_TEST_CONFIG.copy()


@pytest.fixture(scope="session")
def regression_thresholds():
    """Provide performance regression thresholds."""
    return REGRESSION_THRESHOLDS.copy()


@pytest.fixture(scope="session")
def temp_performance_dir():
    """Provide temporary directory for performance test artifacts."""
    temp_dir = Path(tempfile.mkdtemp(prefix="perf_test_"))
    yield temp_dir

    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def system_monitor():
    """Provide system resource monitoring utilities."""

    class SystemMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.monitoring_data = []
            self.monitoring_active = False

        def start_monitoring(self, interval: float = 1.0):
            """Start monitoring system resources."""
            self.monitoring_active = True
            self.monitoring_data.clear()

        def stop_monitoring(self):
            """Stop monitoring system resources."""
            self.monitoring_active = False

        def get_current_metrics(self) -> dict[str, float]:
            """Get current system metrics."""
            memory_info = self.process.memory_info()

            return {
                'cpu_percent': self.process.cpu_percent(),
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
                'num_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            }

        def record_snapshot(self) -> dict[str, float]:
            """Record a snapshot of current metrics."""
            metrics = self.get_current_metrics()
            metrics['timestamp'] = time.time()
            self.monitoring_data.append(metrics)
            return metrics

        def get_monitoring_summary(self) -> dict[str, Any]:
            """Get summary of monitoring data."""
            if not self.monitoring_data:
                return {'error': 'No monitoring data collected'}

            metrics = {}
            for key in ['cpu_percent', 'memory_rss_mb', 'memory_vms_mb', 'memory_percent']:
                values = [d[key] for d in self.monitoring_data]
                metrics[key] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }

            return {
                'summary': metrics,
                'duration_seconds': self.monitoring_data[-1]['timestamp'] - self.monitoring_data[0]['timestamp'],
                'sample_count': len(self.monitoring_data)
            }

    return SystemMonitor()


@pytest.fixture
async def mock_qdrant_client():
    """Provide mock Qdrant client optimized for performance testing."""

    mock_client = AsyncMock()

    # Realistic response times for different operations
    async def mock_search(*args, **kwargs):
        await asyncio.sleep(0.01)  # 10ms search time
        return MagicMock(
            points=[
                MagicMock(
                    id=f"doc_{i}",
                    score=0.95 - (i * 0.1),
                    payload={"content": f"test document {i}", "type": "test"}
                )
                for i in range(min(kwargs.get('limit', 10), 10))
            ]
        )

    async def mock_upsert(*args, **kwargs):
        # Simulate varying upsert times based on batch size
        points = kwargs.get('points', [])
        batch_size = len(points) if isinstance(points, list) else 1
        base_time = 0.05  # 50ms base time
        batch_overhead = batch_size * 0.01  # 10ms per additional point

        await asyncio.sleep(base_time + batch_overhead)
        return MagicMock(operation_id=12345, status="completed")

    async def mock_get_collection(*args, **kwargs):
        await asyncio.sleep(0.005)  # 5ms collection info time
        return MagicMock(
            config=MagicMock(
                params=MagicMock(
                    vectors=MagicMock(size=384),
                    distance="Cosine"
                )
            ),
            points_count=1000,
            status="green"
        )

    async def mock_list_collections(*args, **kwargs):
        await asyncio.sleep(0.002)  # 2ms list time
        return MagicMock(
            collections=[
                MagicMock(name=f"collection_{i}")
                for i in range(5)
            ]
        )

    async def mock_create_collection(*args, **kwargs):
        await asyncio.sleep(0.02)  # 20ms creation time
        return True

    async def mock_delete_collection(*args, **kwargs):
        await asyncio.sleep(0.01)  # 10ms deletion time
        return True

    # Assign mock methods
    mock_client.search = mock_search
    mock_client.upsert = mock_upsert
    mock_client.get_collection = mock_get_collection
    mock_client.list_collections = mock_list_collections
    mock_client.create_collection = mock_create_collection
    mock_client.delete_collection = mock_delete_collection

    return mock_client


@pytest.fixture
async def mock_embedding_service():
    """Provide mock embedding service for performance testing."""

    mock_service = AsyncMock()

    async def mock_embed(texts: list[str], *args, **kwargs):
        # Simulate embedding time based on text length and count
        total_chars = sum(len(text) for text in texts)
        base_time = 0.02  # 20ms base time
        char_time = total_chars * 0.0001  # 0.1ms per character

        await asyncio.sleep(base_time + char_time)

        # Return mock embeddings
        return [[0.1 + i * 0.01] * 384 for i in range(len(texts))]

    async def mock_embed_query(query: str, *args, **kwargs):
        await asyncio.sleep(0.015)  # 15ms for single query
        return [0.2 + len(query) * 0.001] * 384

    mock_service.embed = mock_embed
    mock_service.embed_query = mock_embed_query

    return mock_service


@pytest.fixture
def performance_test_data():
    """Provide test data for performance testing."""

    return {
        'small_documents': [
            f"Small test document {i} with basic content."
            for i in range(100)
        ],
        'medium_documents': [
            f"Medium test document {i} with more detailed content. " * 10
            for i in range(500)
        ],
        'large_documents': [
            f"Large test document {i} with extensive content. " * 50
            for i in range(1000)
        ],
        'search_queries': [
            "performance testing",
            "document search functionality",
            "vector similarity matching",
            "hybrid search capabilities",
            "workspace management",
            "collection operations",
            "memory optimization",
            "concurrent processing",
            "scalability analysis",
            "regression testing",
        ],
        'metadata_samples': [
            {"type": "test", "category": "performance", "size": "small"},
            {"type": "document", "category": "functional", "size": "medium"},
            {"type": "code", "category": "integration", "size": "large"},
            {"type": "config", "category": "system", "size": "small"},
            {"type": "log", "category": "monitoring", "size": "medium"},
        ]
    }


@pytest.fixture
def memory_profiler():
    """Provide memory profiling utilities."""

    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
            self.tracemalloc_active = False

        def start_profiling(self):
            """Start memory profiling."""
            import tracemalloc
            tracemalloc.start()
            self.tracemalloc_active = True
            self.snapshots.clear()

        def stop_profiling(self):
            """Stop memory profiling."""
            if self.tracemalloc_active:
                import tracemalloc
                tracemalloc.stop()
                self.tracemalloc_active = False

        def take_snapshot(self, label: str = None):
            """Take a memory snapshot."""
            if self.tracemalloc_active:
                import tracemalloc
                snapshot = tracemalloc.take_snapshot()
                self.snapshots.append({
                    'label': label or f"snapshot_{len(self.snapshots)}",
                    'timestamp': time.time(),
                    'snapshot': snapshot
                })
                return snapshot
            return None

        def compare_snapshots(self, start_idx: int = 0, end_idx: int = -1):
            """Compare memory snapshots."""
            if len(self.snapshots) < 2:
                return None

            start_snapshot = self.snapshots[start_idx]['snapshot']
            end_snapshot = self.snapshots[end_idx]['snapshot']

            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')

            return {
                'total_diff_mb': sum(stat.size_diff for stat in top_stats) / 1024 / 1024,
                'top_differences': [
                    {
                        'filename': stat.traceback.format()[0] if stat.traceback else 'unknown',
                        'size_diff_mb': stat.size_diff / 1024 / 1024,
                        'count_diff': stat.count_diff
                    }
                    for stat in top_stats[:10]
                ]
            }

        def get_current_memory_usage(self):
            """Get current memory usage."""
            process = psutil.Process()
            memory_info = process.memory_info()

            tracemalloc_memory = 0
            if self.tracemalloc_active:
                import tracemalloc
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc_memory = current / 1024 / 1024

            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'tracemalloc_mb': tracemalloc_memory,
                'available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }

    return MemoryProfiler()


@pytest.fixture(autouse=True)
def gc_management():
    """Manage garbage collection for consistent performance testing."""

    # Disable automatic garbage collection during tests for consistent timing
    gc.disable()

    yield

    # Re-enable garbage collection and clean up
    gc.enable()
    gc.collect()


@pytest.fixture
def performance_baseline_manager(tmp_path):
    """Manage performance baselines for regression testing."""

    class BaselineManager:
        def __init__(self, baseline_dir: Path):
            self.baseline_dir = baseline_dir
            self.baseline_dir.mkdir(exist_ok=True)

        def save_baseline(self, test_name: str, metrics: dict[str, Any]):
            """Save performance baseline."""
            baseline_file = self.baseline_dir / f"{test_name}_baseline.json"

            import json
            with open(baseline_file, 'w') as f:
                json.dump({
                    'test_name': test_name,
                    'timestamp': time.time(),
                    'metrics': metrics
                }, f, indent=2)

        def load_baseline(self, test_name: str) -> dict[str, Any] | None:
            """Load performance baseline."""
            baseline_file = self.baseline_dir / f"{test_name}_baseline.json"

            if not baseline_file.exists():
                return None

            import json
            try:
                with open(baseline_file) as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return None

        def compare_to_baseline(self, test_name: str, current_metrics: dict[str, Any]) -> dict[str, Any]:
            """Compare current metrics to baseline."""
            baseline = self.load_baseline(test_name)

            if not baseline:
                return {'error': 'No baseline found', 'test_name': test_name}

            baseline_metrics = baseline.get('metrics', {})

            comparisons = {}
            regressions = []

            for metric_name, current_value in current_metrics.items():
                if metric_name in baseline_metrics:
                    baseline_value = baseline_metrics[metric_name]

                    if baseline_value > 0:  # Avoid division by zero
                        change_percent = ((current_value - baseline_value) / baseline_value) * 100

                        comparisons[metric_name] = {
                            'baseline': baseline_value,
                            'current': current_value,
                            'change_percent': change_percent,
                            'absolute_change': current_value - baseline_value
                        }

                        # Check for regressions based on metric type
                        if self._is_regression(metric_name, change_percent):
                            regressions.append({
                                'metric': metric_name,
                                'change_percent': change_percent,
                                'threshold_exceeded': True
                            })

            return {
                'test_name': test_name,
                'comparisons': comparisons,
                'regressions': regressions,
                'baseline_age_hours': (time.time() - baseline.get('timestamp', 0)) / 3600
            }

        def _is_regression(self, metric_name: str, change_percent: float) -> bool:
            """Determine if a change represents a performance regression."""

            # Performance degradation thresholds
            if any(keyword in metric_name.lower() for keyword in ['time', 'latency', 'duration']):
                return change_percent > REGRESSION_THRESHOLDS['response_time_increase_percent']
            elif 'memory' in metric_name.lower():
                return change_percent > REGRESSION_THRESHOLDS['memory_increase_percent']
            elif any(keyword in metric_name.lower() for keyword in ['throughput', 'rps', 'ops']):
                return change_percent < -REGRESSION_THRESHOLDS['throughput_decrease_percent']
            elif 'error' in metric_name.lower():
                return change_percent > REGRESSION_THRESHOLDS['error_rate_increase_percent']

            return False

    return BaselineManager(tmp_path / "baselines")


@pytest.fixture
async def mcp_server_mock():
    """Provide mock MCP server for performance testing."""

    class MockMCPServer:
        def __init__(self):
            self.call_count = 0
            self.response_times = []

        async def call_tool(self, tool_name: str, **kwargs):
            """Mock MCP tool call with realistic response times."""
            self.call_count += 1
            start_time = time.perf_counter()

            # Simulate different response times for different tools
            tool_times = {
                'workspace_status': 0.01,      # 10ms
                'search_workspace': 0.05,      # 50ms
                'add_document': 0.1,           # 100ms
                'get_document': 0.02,          # 20ms
                'list_collections': 0.005,     # 5ms
                'create_collection': 0.02,     # 20ms
                'update_scratchbook': 0.03,    # 30ms
                'search_scratchbook': 0.04,    # 40ms
                'research_workspace': 0.15,    # 150ms
                'hybrid_search_advanced': 0.08, # 80ms
            }

            response_time = tool_times.get(tool_name, 0.05)  # Default 50ms
            await asyncio.sleep(response_time)

            end_time = time.perf_counter()
            actual_time = end_time - start_time
            self.response_times.append(actual_time)

            return {
                'tool': tool_name,
                'result': {'status': 'success', 'data': f'mock result for {tool_name}'},
                'response_time_ms': actual_time * 1000
            }

        def get_stats(self):
            """Get mock server statistics."""
            if not self.response_times:
                return {'error': 'No calls made'}

            return {
                'total_calls': self.call_count,
                'avg_response_time_ms': (sum(self.response_times) / len(self.response_times)) * 1000,
                'max_response_time_ms': max(self.response_times) * 1000,
                'min_response_time_ms': min(self.response_times) * 1000,
            }

    return MockMCPServer()


# Pytest hooks for performance test reporting
def pytest_runtest_makereport(item, call):
    """Create performance test reports."""

    if call.when == "call" and hasattr(item, "obj"):
        # Add performance markers to test reports
        if any(marker.name == "performance" for marker in item.iter_markers()):
            # This could be extended to generate detailed performance reports
            pass
