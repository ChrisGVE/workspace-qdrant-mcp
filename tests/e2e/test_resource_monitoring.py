"""
End-to-end tests for resource monitoring and performance tracking.

Tests comprehensive monitoring of system resources (memory, CPU, disk I/O,
network, file descriptors, threads, queue depths) and performance metrics
(ingestion throughput, search latency, response times). Validates resource
usage stays within acceptable bounds and detects performance regressions.
"""

import pytest
import asyncio
import time
import psutil
from pathlib import Path
from typing import List, Dict, Optional

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
    ResourceMonitor,
    ResourceMetrics,
)


@pytest.mark.integration
@pytest.mark.slow
class TestMemoryMonitoring:
    """Test memory usage monitoring and tracking."""

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test tracking memory usage over time."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Allow monitoring to collect data
        await asyncio.sleep(3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have collected metrics
        assert len(metrics) > 0

        # Verify memory metrics present
        for metric in metrics:
            assert metric.memory_mb > 0

    @pytest.mark.asyncio
    async def test_memory_growth_detection(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test detection of memory growth patterns."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations that may increase memory
        for i in range(5):
            test_file = workspace / f"memory_test_{i}.txt"
            test_file.write_text(f"Memory test content {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-memory-{i}",
                ]
            )
            await asyncio.sleep(1)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have memory measurements
        assert len(metrics) > 0

    @pytest.mark.asyncio
    async def test_memory_leak_detection(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test detection of memory leaks."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform repeated operations
        for _ in range(10):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Memory should be relatively stable (no unbounded growth)
        if len(metrics) > 2:
            first_memory = metrics[0].memory_mb
            last_memory = metrics[-1].memory_mb

            # Allow some growth but not excessive
            growth_ratio = last_memory / first_memory if first_memory > 0 else 1
            assert growth_ratio < 2.0  # Less than 2x growth


@pytest.mark.integration
@pytest.mark.slow
class TestCPUMonitoring:
    """Test CPU utilization monitoring."""

    @pytest.mark.asyncio
    async def test_cpu_usage_tracking(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test tracking CPU usage."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        await asyncio.sleep(3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have CPU metrics
        assert len(metrics) > 0
        for metric in metrics:
            assert 0 <= metric.cpu_percent <= 100

    @pytest.mark.asyncio
    async def test_cpu_spike_detection(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test detection of CPU usage spikes."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.3)

        # Create CPU-intensive workload
        for i in range(3):
            test_file = workspace / f"cpu_test_{i}.txt"
            test_file.write_text(f"CPU test content {i}" * 100)
            cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", f"test-cpu-{i}"]
            )

        await asyncio.sleep(2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have recorded some CPU activity
        assert len(metrics) > 0
        assert any(metric.cpu_percent > 0 for metric in metrics)


@pytest.mark.integration
@pytest.mark.slow
class TestFileDescriptorMonitoring:
    """Test file descriptor usage monitoring."""

    @pytest.mark.asyncio
    async def test_file_descriptor_tracking(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test tracking file descriptor usage."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        await asyncio.sleep(2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have FD metrics
        assert len(metrics) > 0
        for metric in metrics:
            assert metric.open_files >= 0

    @pytest.mark.asyncio
    async def test_file_descriptor_leak_detection(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test detection of file descriptor leaks."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations that open files
        for _ in range(10):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # File descriptors should be stable
        if len(metrics) > 2:
            first_fds = metrics[0].open_files
            last_fds = metrics[-1].open_files

            # Allow some variation but not unbounded growth
            fd_growth = last_fds - first_fds
            assert fd_growth < 50  # Not more than 50 FDs growth


@pytest.mark.integration
@pytest.mark.slow
class TestThreadMonitoring:
    """Test thread count monitoring."""

    @pytest.mark.asyncio
    async def test_thread_count_tracking(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test tracking thread counts."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        await asyncio.sleep(2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have thread metrics
        assert len(metrics) > 0
        for metric in metrics:
            assert metric.thread_count > 0

    @pytest.mark.asyncio
    async def test_thread_leak_detection(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test detection of thread leaks."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations
        for _ in range(10):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Thread count should be stable
        if len(metrics) > 2:
            first_threads = metrics[0].thread_count
            last_threads = metrics[-1].thread_count

            # Allow some variation but not excessive growth
            thread_growth = last_threads - first_threads
            assert thread_growth < 20  # Not more than 20 threads growth


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceMetrics:
    """Test performance metric tracking."""

    def test_ingestion_throughput_measurement(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test measuring ingestion throughput."""
        workspace = system_components.workspace_path

        # Create test files
        file_count = 10
        for i in range(file_count):
            test_file = workspace / f"throughput_{i}.txt"
            test_file.write_text(f"Throughput test content {i}")

        # Measure throughput
        start_time = time.time()
        cli_helper.run_command(
            [
                "ingest",
                "folder",
                str(workspace),
                "--collection",
                "test-throughput",
            ],
            timeout=60,
        )
        duration = time.time() - start_time

        # Calculate throughput
        throughput = file_count / duration if duration > 0 else 0

        # Should process files reasonably quickly
        assert throughput > 0.1  # At least 0.1 files/second

    def test_search_latency_measurement(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test measuring search latency."""
        workspace = system_components.workspace_path
        collection_name = f"test-latency-{int(time.time())}"

        # Ingest content
        test_file = workspace / "latency_test.txt"
        test_file.write_text("Search latency measurement content")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # Measure search latency
        start_time = time.time()
        cli_helper.run_command(
            ["search", "latency measurement", "--collection", collection_name],
            timeout=15,
        )
        latency = time.time() - start_time

        # Search should be fast
        assert latency < 10.0  # Less than 10 seconds

    def test_response_time_tracking(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test tracking command response times."""
        response_times = []

        # Measure multiple operations
        for _ in range(5):
            start = time.time()
            cli_helper.run_command(["status", "--quiet"])
            response_times.append(time.time() - start)
            time.sleep(0.5)

        # All responses should be fast
        assert all(rt < 5.0 for rt in response_times)

        # Calculate average response time
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 3.0  # Average under 3 seconds


@pytest.mark.integration
@pytest.mark.slow
class TestResourceUsageBounds:
    """Test resource usage stays within acceptable bounds."""

    @pytest.mark.asyncio
    async def test_memory_stays_bounded(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test memory usage stays within bounds."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform various operations
        for i in range(5):
            test_file = workspace / f"bound_test_{i}.txt"
            test_file.write_text(f"Bound test {i}")
            cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", f"test-bound-{i}"]
            )
            await asyncio.sleep(0.5)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Memory should stay reasonable (under 1GB for test workload)
        if metrics:
            max_memory = max(m.memory_mb for m in metrics)
            assert max_memory < 1024  # Less than 1GB

    @pytest.mark.asyncio
    async def test_cpu_stays_reasonable(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test CPU usage stays reasonable."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations
        for _ in range(10):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Average CPU should be reasonable
        if metrics:
            avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)
            assert avg_cpu < 80  # Average under 80%

    @pytest.mark.asyncio
    async def test_file_descriptors_stay_bounded(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test file descriptor count stays bounded."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations
        for _ in range(20):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # FDs should stay reasonable
        if metrics:
            max_fds = max(m.open_files for m in metrics)
            assert max_fds < 1000  # Less than 1000 open files


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBaseline:
    """Test establishing performance baselines."""

    def test_status_command_baseline(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test baseline for status command."""
        times = []

        # Measure multiple runs
        for _ in range(10):
            start = time.time()
            cli_helper.run_command(["status", "--quiet"])
            times.append(time.time() - start)
            time.sleep(0.1)

        # Calculate baseline metrics
        avg_time = sum(times) / len(times)
        max_time = max(times)

        # Baselines
        assert avg_time < 3.0  # Average under 3s
        assert max_time < 5.0  # Max under 5s

    def test_ingestion_baseline(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test baseline for file ingestion."""
        workspace = system_components.workspace_path

        times = []

        # Measure multiple ingestions
        for i in range(3):
            test_file = workspace / f"baseline_{i}.txt"
            test_file.write_text(f"Baseline test {i}")

            start = time.time()
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-baseline-{i}",
                ]
            )
            times.append(time.time() - start)
            time.sleep(1)

        # Baseline for single file ingestion
        avg_time = sum(times) / len(times)
        assert avg_time < 30.0  # Average under 30s per file

    def test_search_baseline(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test baseline for search operations."""
        workspace = system_components.workspace_path
        collection_name = f"test-search-baseline-{int(time.time())}"

        # Prepare content
        test_file = workspace / "search_baseline.txt"
        test_file.write_text("Search baseline content for testing")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # Measure search times
        times = []
        for _ in range(5):
            start = time.time()
            cli_helper.run_command(
                ["search", "baseline content", "--collection", collection_name],
                timeout=15,
            )
            times.append(time.time() - start)
            time.sleep(0.5)

        # Search baseline
        avg_time = sum(times) / len(times)
        assert avg_time < 10.0  # Average under 10s


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceRegression:
    """Test detection of performance regressions."""

    def test_detect_slow_status_command(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test detection of slow status command."""
        # Measure status command time
        start = time.time()
        cli_helper.run_command(["status", "--quiet"], timeout=10)
        duration = time.time() - start

        # Should be fast (regression if >5s)
        assert duration < 5.0

    def test_detect_slow_ingestion(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test detection of slow ingestion."""
        workspace = system_components.workspace_path

        test_file = workspace / "regression_test.txt"
        test_file.write_text("Regression test content")

        # Measure ingestion time
        start = time.time()
        cli_helper.run_command(
            [
                "ingest",
                "file",
                str(test_file),
                "--collection",
                "test-regression",
            ],
            timeout=60,
        )
        duration = time.time() - start

        # Should be reasonably fast (regression if >30s for single file)
        assert duration < 30.0

    def test_detect_slow_search(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test detection of slow search."""
        workspace = system_components.workspace_path
        collection_name = f"test-slow-search-{int(time.time())}"

        # Prepare content
        test_file = workspace / "slow_search.txt"
        test_file.write_text("Slow search regression test")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # Measure search time
        start = time.time()
        cli_helper.run_command(
            ["search", "regression test", "--collection", collection_name],
            timeout=15,
        )
        duration = time.time() - start

        # Should be fast (regression if >10s)
        assert duration < 10.0


@pytest.mark.integration
@pytest.mark.slow
class TestLoadPerformance:
    """Test performance under load."""

    @pytest.mark.asyncio
    async def test_performance_under_light_load(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system performance under light load."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Light load (5 operations)
        for i in range(5):
            test_file = workspace / f"light_load_{i}.txt"
            test_file.write_text(f"Light load test {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-light-{i}",
                ]
            )
            await asyncio.sleep(0.5)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Resources should be reasonable under light load
        if metrics:
            max_memory = max(m.memory_mb for m in metrics)
            avg_cpu = sum(m.cpu_percent for m in metrics) / len(metrics)

            assert max_memory < 512  # Less than 512MB
            assert avg_cpu < 50  # Average CPU under 50%

    @pytest.mark.asyncio
    async def test_performance_under_moderate_load(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system performance under moderate load."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Moderate load (10 operations)
        for i in range(10):
            test_file = workspace / f"moderate_load_{i}.txt"
            test_file.write_text(f"Moderate load test {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-moderate-{i}",
                ]
            )
            await asyncio.sleep(0.3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Resources should stay reasonable
        if metrics:
            max_memory = max(m.memory_mb for m in metrics)
            assert max_memory < 1024  # Less than 1GB

    def test_throughput_under_load(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test ingestion throughput under load."""
        workspace = system_components.workspace_path

        # Create batch of files
        file_count = 20
        for i in range(file_count):
            test_file = workspace / f"throughput_load_{i}.txt"
            test_file.write_text(f"Throughput load test {i}")

        # Measure batch throughput
        start = time.time()
        cli_helper.run_command(
            [
                "ingest",
                "folder",
                str(workspace),
                "--collection",
                "test-throughput-load",
            ],
            timeout=120,
        )
        duration = time.time() - start

        # Calculate throughput
        throughput = file_count / duration if duration > 0 else 0

        # Should maintain reasonable throughput
        assert throughput > 0.1  # At least 0.1 files/second


@pytest.mark.integration
@pytest.mark.slow
class TestResourceRecovery:
    """Test resource recovery after load."""

    @pytest.mark.asyncio
    async def test_memory_recovery_after_load(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test memory returns to baseline after load."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Record initial memory
        await asyncio.sleep(1)
        initial_metrics = list(resource_monitor.metrics)
        initial_memory = initial_metrics[-1].memory_mb if initial_metrics else 0

        # Create load
        for i in range(10):
            test_file = workspace / f"recovery_{i}.txt"
            test_file.write_text(f"Recovery test {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-recovery-{i}",
                ]
            )
            await asyncio.sleep(0.2)

        # Allow recovery time
        await asyncio.sleep(3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Memory should recover somewhat
        if len(metrics) > 5:
            final_memory = metrics[-1].memory_mb
            # Allow some growth but should not be excessive
            growth = final_memory - initial_memory
            assert growth < 500  # Less than 500MB growth

    @pytest.mark.asyncio
    async def test_cpu_recovery_after_load(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test CPU returns to idle after load."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Create CPU load
        for i in range(5):
            test_file = workspace / f"cpu_recovery_{i}.txt"
            test_file.write_text(f"CPU recovery test {i}" * 100)
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-cpu-recovery-{i}",
                ]
            )
            await asyncio.sleep(0.3)

        # Allow CPU to return to idle
        await asyncio.sleep(3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Final CPU usage should be low
        if len(metrics) > 3:
            final_cpu = metrics[-1].cpu_percent
            assert final_cpu < 30  # Under 30% after recovery


@pytest.mark.integration
@pytest.mark.slow
class TestContinuousMonitoring:
    """Test continuous monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_long_duration_monitoring(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test monitoring over extended period."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=1.0)

        # Monitor for extended period
        await asyncio.sleep(10)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should have collected many metrics
        assert len(metrics) >= 8  # At least 8 samples over 10 seconds

    @pytest.mark.asyncio
    async def test_monitoring_data_collection(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test monitoring collects all required data."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        await asyncio.sleep(3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Verify all metrics collected
        assert len(metrics) > 0
        for metric in metrics:
            assert metric.timestamp > 0
            assert metric.memory_mb >= 0
            assert 0 <= metric.cpu_percent <= 100
            assert metric.open_files >= 0
            assert metric.thread_count > 0


@pytest.mark.integration
@pytest.mark.slow
class TestMetricsReporting:
    """Test metrics reporting and analysis."""

    @pytest.mark.asyncio
    async def test_metrics_summary_generation(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test generating metrics summary."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        # Perform operations
        for _ in range(5):
            cli_helper.run_command(["status", "--quiet"])
            await asyncio.sleep(0.3)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Generate summary
        if metrics:
            summary = {
                "count": len(metrics),
                "avg_memory": sum(m.memory_mb for m in metrics) / len(metrics),
                "max_memory": max(m.memory_mb for m in metrics),
                "avg_cpu": sum(m.cpu_percent for m in metrics) / len(metrics),
                "max_cpu": max(m.cpu_percent for m in metrics),
            }

            # Summary should be reasonable
            assert summary["count"] > 0
            assert summary["avg_memory"] > 0
            assert summary["max_memory"] >= summary["avg_memory"]

    @pytest.mark.asyncio
    async def test_metrics_export(
        self, system_components: SystemComponents, resource_monitor: ResourceMonitor
    ):
        """Test exporting metrics data."""
        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=0.5)

        await asyncio.sleep(2)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Metrics should be exportable
        assert len(metrics) > 0

        # Verify metrics can be serialized
        for metric in metrics:
            metric_dict = {
                "timestamp": metric.timestamp,
                "memory_mb": metric.memory_mb,
                "cpu_percent": metric.cpu_percent,
                "open_files": metric.open_files,
                "thread_count": metric.thread_count,
            }
            assert all(isinstance(v, (int, float)) for v in metric_dict.values())
