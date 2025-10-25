"""
Stress Testing Configuration and Fixtures.

Provides shared fixtures, configuration, and utilities for stress testing
the daemon file ingestion system.
"""

import asyncio
import gc
import json
import os
import random
import shutil
import string
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import psutil
import pytest

from . import MONITORING_CONFIG, PERFORMANCE_THRESHOLDS, STRESS_CONFIG


def pytest_configure(config):
    """Configure pytest for stress testing."""

    # Register stress test marker
    config.addinivalue_line(
        "markers",
        "stress: Stress tests for high load and resource constraints (may take 30+ minutes)"
    )
    config.addinivalue_line(
        "markers",
        "high_volume: High-volume file processing stress tests"
    )
    config.addinivalue_line(
        "markers",
        "rapid_ingestion: Rapid file ingestion rate tests"
    )
    config.addinivalue_line(
        "markers",
        "multi_folder: Multiple folder watching tests"
    )
    config.addinivalue_line(
        "markers",
        "memory_constraint: Memory-constrained environment tests"
    )
    config.addinivalue_line(
        "markers",
        "disk_saturation: Disk I/O saturation tests"
    )
    config.addinivalue_line(
        "markers",
        "network_interruption: Network failure and recovery tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for stress testing."""

    # Apply timeout to all stress tests
    default_timeout = STRESS_CONFIG["timeout_seconds"]

    for item in items:
        # Add timeout marker to stress tests
        if any(marker.name == "stress" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.timeout(default_timeout))


@pytest.fixture(scope="session")
def stress_config():
    """Provide stress test configuration."""
    return STRESS_CONFIG.copy()


@pytest.fixture(scope="session")
def monitoring_config():
    """Provide monitoring configuration."""
    return MONITORING_CONFIG.copy()


@pytest.fixture(scope="session")
def performance_thresholds():
    """Provide performance threshold configuration."""
    return PERFORMANCE_THRESHOLDS.copy()


@pytest.fixture
def stress_temp_dir():
    """Create temporary directory for stress test artifacts."""
    temp_dir = Path(tempfile.mkdtemp(prefix="stress_test_"))
    yield temp_dir

    # Cleanup
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def resource_monitor():
    """Provide system resource monitoring utilities."""

    class ResourceMonitor:
        def __init__(self):
            self.process = psutil.Process()
            self.monitoring_data: list[dict[str, Any]] = []
            self.monitoring_active = False
            self.start_time = None
            self.baseline_memory_mb = None

        def start_monitoring(self):
            """Start monitoring system resources."""
            self.monitoring_active = True
            self.monitoring_data.clear()
            self.start_time = time.time()

            # Record baseline memory
            memory_info = self.process.memory_info()
            self.baseline_memory_mb = memory_info.rss / 1024 / 1024

        def stop_monitoring(self):
            """Stop monitoring system resources."""
            self.monitoring_active = False

        def record_snapshot(self) -> dict[str, Any]:
            """Record a snapshot of current metrics."""
            memory_info = self.process.memory_info()
            cpu_percent = self.process.cpu_percent()

            snapshot = {
                'timestamp': time.time() - (self.start_time or time.time()),
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'num_threads': self.process.num_threads(),
            }

            # Add file descriptors on Unix-like systems
            if hasattr(self.process, 'num_fds'):
                snapshot['num_fds'] = self.process.num_fds()

            # Add disk I/O if available
            try:
                io_counters = self.process.io_counters()
                snapshot['disk_read_mb'] = io_counters.read_bytes / 1024 / 1024
                snapshot['disk_write_mb'] = io_counters.write_bytes / 1024 / 1024
            except (AttributeError, psutil.AccessDenied):
                pass

            self.monitoring_data.append(snapshot)
            return snapshot

        def get_summary(self) -> dict[str, Any]:
            """Get summary of monitoring data."""
            if not self.monitoring_data:
                return {'error': 'No monitoring data collected'}

            # Calculate statistics for each metric
            metrics = {}
            numeric_keys = ['cpu_percent', 'memory_rss_mb', 'memory_vms_mb',
                          'memory_percent', 'num_threads']

            for key in numeric_keys:
                values = [d[key] for d in self.monitoring_data if key in d]
                if values:
                    metrics[key] = {
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'final': values[-1],
                    }

            # Calculate memory growth
            if self.baseline_memory_mb:
                final_memory = metrics.get('memory_rss_mb', {}).get('final', 0)
                metrics['memory_growth_mb'] = final_memory - self.baseline_memory_mb
                metrics['memory_growth_percent'] = (
                    (final_memory - self.baseline_memory_mb) / self.baseline_memory_mb * 100
                    if self.baseline_memory_mb > 0 else 0
                )

            return {
                'summary': metrics,
                'duration_seconds': self.monitoring_data[-1]['timestamp'],
                'sample_count': len(self.monitoring_data),
                'baseline_memory_mb': self.baseline_memory_mb,
            }

        def check_thresholds(self) -> dict[str, list[str]]:
            """Check if any thresholds were exceeded."""
            warnings = []
            criticals = []

            summary = self.get_summary()
            metrics = summary.get('summary', {})

            # Check memory thresholds
            max_memory = metrics.get('memory_rss_mb', {}).get('max', 0)
            if max_memory > MONITORING_CONFIG['memory_critical_threshold_mb']:
                criticals.append(f"Memory exceeded critical threshold: {max_memory:.1f}MB")
            elif max_memory > MONITORING_CONFIG['memory_warning_threshold_mb']:
                warnings.append(f"Memory exceeded warning threshold: {max_memory:.1f}MB")

            # Check CPU thresholds
            max_cpu = metrics.get('cpu_percent', {}).get('max', 0)
            if max_cpu > MONITORING_CONFIG['cpu_critical_threshold_percent']:
                criticals.append(f"CPU exceeded critical threshold: {max_cpu:.1f}%")
            elif max_cpu > MONITORING_CONFIG['cpu_warning_threshold_percent']:
                warnings.append(f"CPU exceeded warning threshold: {max_cpu:.1f}%")

            return {'warnings': warnings, 'criticals': criticals}

    return ResourceMonitor()


@pytest.fixture
def file_generator():
    """Provide utilities for generating test files."""

    class FileGenerator:
        def __init__(self):
            self.random = random.Random(42)  # Fixed seed for reproducibility

        def generate_random_content(self, size_bytes: int) -> str:
            """Generate random text content of specified size."""
            # Generate in chunks to avoid memory issues with large files
            chunk_size = 1024
            chunks = []
            remaining = size_bytes

            while remaining > 0:
                chunk_len = min(chunk_size, remaining)
                chunk = ''.join(
                    self.random.choices(
                        string.ascii_letters + string.digits + ' \n',
                        k=chunk_len
                    )
                )
                chunks.append(chunk)
                remaining -= chunk_len

            return ''.join(chunks)

        def create_test_file(
            self,
            directory: Path,
            filename: str,
            content: str | None = None,
            size_bytes: int | None = None
        ) -> Path:
            """Create a single test file."""
            file_path = directory / filename

            if content is None and size_bytes is not None:
                content = self.generate_random_content(size_bytes)
            elif content is None:
                content = f"Test content for {filename}"

            file_path.write_text(content, encoding='utf-8')
            return file_path

        def create_test_files_batch(
            self,
            directory: Path,
            count: int,
            prefix: str = "test_",
            extension: str = ".txt",
            size_bytes: int = 100
        ) -> list[Path]:
            """Create a batch of test files efficiently."""
            files = []
            content = self.generate_random_content(size_bytes)

            for i in range(count):
                filename = f"{prefix}{i:06d}{extension}"
                file_path = directory / filename
                file_path.write_text(content, encoding='utf-8')
                files.append(file_path)

            return files

        async def create_files_at_rate(
            self,
            directory: Path,
            count: int,
            rate_per_second: int,
            prefix: str = "rapid_",
            extension: str = ".txt"
        ) -> list[Path]:
            """Create files at a specified rate (files/second)."""
            files = []
            interval = 1.0 / rate_per_second
            content = self.generate_random_content(100)

            for i in range(count):
                start = time.time()

                filename = f"{prefix}{i:06d}{extension}"
                file_path = directory / filename
                file_path.write_text(content, encoding='utf-8')
                files.append(file_path)

                # Sleep to maintain rate
                elapsed = time.time() - start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            return files

    return FileGenerator()


@pytest.fixture
def performance_tracker():
    """Provide performance tracking utilities."""

    class PerformanceTracker:
        def __init__(self):
            self.metrics: dict[str, list[float]] = {}
            self.start_times: dict[str, float] = {}

        def start_operation(self, operation_name: str):
            """Start timing an operation."""
            self.start_times[operation_name] = time.time()

        def end_operation(self, operation_name: str):
            """End timing an operation and record duration."""
            if operation_name not in self.start_times:
                return

            duration = time.time() - self.start_times[operation_name]

            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(duration)

            del self.start_times[operation_name]

        def record_metric(self, metric_name: str, value: float):
            """Record a custom metric value."""
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)

        def get_summary(self) -> dict[str, dict[str, float]]:
            """Get summary statistics for all metrics."""
            summary = {}

            for name, values in self.metrics.items():
                if not values:
                    continue

                sorted_values = sorted(values)
                count = len(sorted_values)

                summary[name] = {
                    'count': count,
                    'min': sorted_values[0],
                    'max': sorted_values[-1],
                    'mean': sum(sorted_values) / count,
                    'median': sorted_values[count // 2],
                    'p95': sorted_values[int(count * 0.95)] if count > 0 else 0,
                    'p99': sorted_values[int(count * 0.99)] if count > 0 else 0,
                }

            return summary

        def save_to_file(self, file_path: Path):
            """Save performance metrics to JSON file."""
            summary = self.get_summary()
            with open(file_path, 'w') as f:
                json.dump(summary, f, indent=2)

    return PerformanceTracker()


@pytest.fixture
async def mock_pipeline():
    """Provide mock UnifiedDocumentPipeline for stress testing."""
    from unittest.mock import AsyncMock, MagicMock

    pipeline = MagicMock()
    pipeline.max_concurrency = STRESS_CONFIG.get("concurrent_operations", 10)

    # Mock process_documents to simulate processing with realistic delays
    async def mock_process_documents(file_paths, collection, **kwargs):
        results = []

        # Simulate processing time based on file count
        processing_time = len(file_paths) * 0.001  # 1ms per file
        await asyncio.sleep(min(processing_time, 0.1))  # Cap at 100ms

        for file_path in file_paths:
            results.append(MagicMock(
                file_path=file_path,
                success=True,
                chunks_created=1,
                error=None
            ))

        return results

    pipeline.process_documents = mock_process_documents
    pipeline.initialize = AsyncMock()

    return pipeline


@pytest.fixture
def memory_tracker():
    """Provide memory tracking with tracemalloc."""

    class MemoryTracker:
        def __init__(self):
            self.snapshots = []
            self.baseline = None

        def start(self):
            """Start memory tracking."""
            tracemalloc.start()
            gc.collect()
            self.baseline = tracemalloc.take_snapshot()

        def snapshot(self) -> dict[str, Any]:
            """Take a memory snapshot and return stats."""
            gc.collect()
            current = tracemalloc.take_snapshot()

            stats = {
                'timestamp': time.time(),
                'current_mb': sum(stat.size for stat in current.statistics('lineno')) / 1024 / 1024,
            }

            if self.baseline:
                top_stats = current.compare_to(self.baseline, 'lineno')
                stats['diff_mb'] = sum(stat.size_diff for stat in top_stats) / 1024 / 1024

            self.snapshots.append(stats)
            return stats

        def stop(self) -> dict[str, Any]:
            """Stop tracking and return summary."""
            if not self.snapshots:
                return {}

            summary = {
                'baseline_mb': self.snapshots[0].get('current_mb', 0),
                'final_mb': self.snapshots[-1].get('current_mb', 0),
                'max_mb': max(s.get('current_mb', 0) for s in self.snapshots),
                'total_growth_mb': self.snapshots[-1].get('diff_mb', 0),
            }

            tracemalloc.stop()
            return summary

    return MemoryTracker()
