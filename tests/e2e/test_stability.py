"""
End-to-end 24-hour stability test with real-world simulation.

Tests long-running system stability by simulating real-world usage patterns:
continuous file changes, periodic searches, project switching, concurrent
operations, varying load patterns. Monitors for memory leaks, performance
degradation, resource exhaustion, and overall system stability over time.

NOTE: This test is designed to run for extended periods (hours or days) and
should be executed separately from the normal test suite. Use the stability
test runner script (scripts/run_stability_test.py) to execute these tests.
"""

import pytest
import asyncio
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tests.e2e.fixtures import (
    SystemComponents,
    CLIHelper,
    ResourceMonitor,
    ResourceMetrics,
)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestShortStabilityBaseline:
    """Short-duration stability tests for baseline validation (can run in CI)."""

    @pytest.mark.asyncio
    async def test_one_hour_stability_baseline(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system stability over 1 hour (short baseline)."""
        workspace = system_components.workspace_path
        duration_seconds = 3600  # 1 hour

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=60)

        start_time = time.time()
        operation_count = 0

        # Run for 1 hour
        while time.time() - start_time < duration_seconds:
            # Simulate user operations
            test_file = workspace / f"stability_{operation_count}.txt"
            test_file.write_text(f"Stability test {operation_count}")

            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-stability-{operation_count % 10}",
                ]
            )

            operation_count += 1

            # Periodic status check
            if operation_count % 10 == 0:
                cli_helper.run_command(["status", "--quiet"])

            # Wait between operations
            await asyncio.sleep(30)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Verify stability
        assert operation_count >= 100  # Should complete many operations

        # Check for memory leaks
        if len(metrics) > 2:
            initial_memory = metrics[0].memory_mb
            final_memory = metrics[-1].memory_mb
            memory_growth_ratio = final_memory / initial_memory if initial_memory > 0 else 1
            assert memory_growth_ratio < 2.0  # Less than 2x growth

    @pytest.mark.asyncio
    async def test_six_hour_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system stability over 6 hours."""
        workspace = system_components.workspace_path
        duration_seconds = 21600  # 6 hours

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=300)

        start_time = time.time()
        operations = {
            "ingestion": 0,
            "search": 0,
            "status": 0,
        }

        # Run for 6 hours
        while time.time() - start_time < duration_seconds:
            # Vary operations
            op_type = random.choice(["ingest", "search", "status"])

            if op_type == "ingest":
                test_file = workspace / f"six_hour_{operations['ingestion']}.txt"
                test_file.write_text(f"Six hour test {operations['ingestion']}")
                cli_helper.run_command(
                    [
                        "ingest",
                        "file",
                        str(test_file),
                        "--collection",
                        f"test-six-{operations['ingestion'] % 5}",
                    ]
                )
                operations["ingestion"] += 1

            elif op_type == "search":
                cli_helper.run_command(
                    [
                        "search",
                        "test content",
                        "--collection",
                        f"test-six-{operations['search'] % 5}",
                    ],
                    timeout=15,
                )
                operations["search"] += 1

            else:
                cli_helper.run_command(["status", "--quiet"])
                operations["status"] += 1

            # Wait between operations
            await asyncio.sleep(60)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Verify diverse operations completed
        assert operations["ingestion"] > 0
        assert operations["search"] > 0
        assert operations["status"] > 0

        # Check stability
        if len(metrics) > 2:
            memory_growth = metrics[-1].memory_mb - metrics[0].memory_mb
            assert memory_growth < 1000  # Less than 1GB growth


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
@pytest.mark.extended
class TestExtendedStability:
    """Extended stability tests (24+ hours) - run separately from main suite."""

    @pytest.mark.asyncio
    async def test_24_hour_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system stability over 24 hours with real-world simulation."""
        workspace = system_components.workspace_path
        duration_seconds = 86400  # 24 hours

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=600)

        start_time = time.time()
        elapsed_hours = 0

        operations = {
            "ingestion": 0,
            "search": 0,
            "status": 0,
            "collection_ops": 0,
            "errors": 0,
        }

        # Run for 24 hours
        while time.time() - start_time < duration_seconds:
            current_elapsed = (time.time() - start_time) / 3600
            if int(current_elapsed) > elapsed_hours:
                elapsed_hours = int(current_elapsed)
                print(f"Stability test: {elapsed_hours} hours elapsed")

            try:
                # Simulate varying load patterns
                if elapsed_hours % 4 == 0:  # Heavy load every 4 hours
                    # Concurrent operations
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = []
                        for i in range(10):
                            test_file = workspace / f"heavy_{operations['ingestion']}_{i}.txt"
                            test_file.write_text(f"Heavy load test {i}")
                            future = executor.submit(
                                cli_helper.run_command,
                                [
                                    "ingest",
                                    "file",
                                    str(test_file),
                                    "--collection",
                                    f"test-24h-{i % 5}",
                                ],
                            )
                            futures.append(future)

                        for future in as_completed(futures):
                            future.result()
                            operations["ingestion"] += 1

                else:  # Normal load
                    # Regular ingestion
                    test_file = workspace / f"normal_{operations['ingestion']}.txt"
                    test_file.write_text(f"Normal operation {operations['ingestion']}")
                    cli_helper.run_command(
                        [
                            "ingest",
                            "file",
                            str(test_file),
                            "--collection",
                            f"test-24h-{operations['ingestion'] % 10}",
                        ]
                    )
                    operations["ingestion"] += 1

                    # Periodic searches
                    if operations["ingestion"] % 5 == 0:
                        cli_helper.run_command(
                            [
                                "search",
                                "test content",
                                "--collection",
                                f"test-24h-{operations['search'] % 10}",
                            ],
                            timeout=15,
                        )
                        operations["search"] += 1

                    # Regular status checks
                    if operations["ingestion"] % 10 == 0:
                        cli_helper.run_command(["status", "--quiet"])
                        operations["status"] += 1

                    # Collection management
                    if operations["ingestion"] % 20 == 0:
                        cli_helper.run_command(["admin", "collections"])
                        operations["collection_ops"] += 1

            except Exception as e:
                print(f"Error during operation: {e}")
                operations["errors"] += 1

            # Variable wait time (simulate real-world usage)
            wait_time = random.uniform(10, 60)
            await asyncio.sleep(wait_time)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Report final statistics
        print(f"\n24-hour Stability Test Results:")
        print(f"  Ingestions: {operations['ingestion']}")
        print(f"  Searches: {operations['search']}")
        print(f"  Status checks: {operations['status']}")
        print(f"  Collection ops: {operations['collection_ops']}")
        print(f"  Errors: {operations['errors']}")

        # Verify system remained operational
        assert operations["ingestion"] >= 1000  # At least 1000 operations
        assert operations["errors"] < operations["ingestion"] * 0.05  # <5% error rate

        # Check for memory leaks
        if len(metrics) > 2:
            initial_memory = metrics[0].memory_mb
            final_memory = metrics[-1].memory_mb
            memory_growth_ratio = final_memory / initial_memory if initial_memory > 0 else 1
            print(f"  Memory growth ratio: {memory_growth_ratio:.2f}x")
            assert memory_growth_ratio < 3.0  # Less than 3x growth over 24h


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestConcurrentStabilityPatterns:
    """Test stability under various concurrent usage patterns."""

    @pytest.mark.asyncio
    async def test_multi_user_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test stability with multiple concurrent users."""
        workspace = system_components.workspace_path
        duration_seconds = 1800  # 30 minutes

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=30)

        start_time = time.time()

        def user_workflow(user_id: int) -> int:
            """Simulate a user's workflow."""
            operations = 0
            user_start = time.time()

            while time.time() - user_start < duration_seconds:
                try:
                    # User operations
                    test_file = workspace / f"user_{user_id}_{operations}.txt"
                    test_file.write_text(f"User {user_id} operation {operations}")

                    cli_helper.run_command(
                        [
                            "ingest",
                            "file",
                            str(test_file),
                            "--collection",
                            f"test-user-{user_id}",
                        ]
                    )

                    operations += 1

                    # Occasional search
                    if operations % 3 == 0:
                        cli_helper.run_command(
                            ["search", "test content", "--collection", f"test-user-{user_id}"],
                            timeout=15,
                        )

                    time.sleep(random.uniform(5, 15))

                except Exception:
                    pass

            return operations

        # Simulate 5 concurrent users
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(user_workflow, i) for i in range(5)]
            results = [f.result() for f in as_completed(futures)]

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Verify all users completed operations
        assert len(results) == 5
        assert all(r > 0 for r in results)
        total_operations = sum(results)
        assert total_operations >= 50  # At least 50 operations total

        # System should remain stable
        if len(metrics) > 2:
            memory_growth = metrics[-1].memory_mb - metrics[0].memory_mb
            assert memory_growth < 500  # Less than 500MB growth


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestLoadPatternStability:
    """Test stability under varying load patterns."""

    @pytest.mark.asyncio
    async def test_burst_load_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test stability with burst load patterns."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=10)

        # Burst pattern: heavy load followed by idle
        for burst in range(5):
            # Heavy burst
            for i in range(20):
                test_file = workspace / f"burst_{burst}_{i}.txt"
                test_file.write_text(f"Burst {burst} file {i}")
                cli_helper.run_command(
                    [
                        "ingest",
                        "file",
                        str(test_file),
                        "--collection",
                        f"test-burst-{burst}",
                    ]
                )
                await asyncio.sleep(0.5)

            # Idle period
            await asyncio.sleep(30)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # System should handle bursts
        assert len(metrics) > 0

        # Check recovery after bursts
        if len(metrics) > 10:
            # CPU should drop during idle periods
            peak_cpu = max(m.cpu_percent for m in metrics[:len(metrics)//2])
            idle_cpu = max(m.cpu_percent for m in metrics[len(metrics)//2:])
            # Idle should be lower than peak (with tolerance)
            assert idle_cpu < peak_cpu + 20

    @pytest.mark.asyncio
    async def test_sustained_load_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test stability under sustained load."""
        workspace = system_components.workspace_path
        duration_seconds = 1800  # 30 minutes

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=60)

        start_time = time.time()
        operation_count = 0

        # Sustained load
        while time.time() - start_time < duration_seconds:
            test_file = workspace / f"sustained_{operation_count}.txt"
            test_file.write_text(f"Sustained test {operation_count}")

            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-sustained-{operation_count % 5}",
                ]
            )

            operation_count += 1
            await asyncio.sleep(5)  # Consistent load

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Should complete many operations
        assert operation_count >= 300  # At least 300 operations

        # Performance should remain stable
        if len(metrics) > 4:
            # Compare early vs late performance
            early_metrics = metrics[:len(metrics)//4]
            late_metrics = metrics[3*len(metrics)//4:]

            early_avg_memory = sum(m.memory_mb for m in early_metrics) / len(early_metrics)
            late_avg_memory = sum(m.memory_mb for m in late_metrics) / len(late_metrics)

            # Memory shouldn't grow excessively
            memory_growth_ratio = late_avg_memory / early_avg_memory if early_avg_memory > 0 else 1
            assert memory_growth_ratio < 2.0


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestProjectSwitchingStability:
    """Test stability with frequent project switching."""

    @pytest.mark.asyncio
    async def test_project_switching_stability(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
        tmp_path,
    ):
        """Test stability when switching between projects."""
        # Create multiple projects
        projects = []
        for i in range(5):
            proj = tmp_path / f"stability_proj_{i}"
            proj.mkdir()
            projects.append(proj)

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=30)

        # Switch between projects frequently
        for cycle in range(20):
            proj = projects[cycle % len(projects)]

            # Work in project
            test_file = proj / f"file_{cycle}.txt"
            test_file.write_text(f"Project switching test {cycle}")

            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-switch-{cycle % len(projects)}",
                ]
            )

            await asyncio.sleep(5)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # System should handle project switching
        assert len(metrics) > 0

        # Check for resource leaks
        if len(metrics) > 2:
            memory_growth = metrics[-1].memory_mb - metrics[0].memory_mb
            assert memory_growth < 300  # Less than 300MB growth


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestPerformanceDegradationDetection:
    """Test detection of performance degradation over time."""

    @pytest.mark.asyncio
    async def test_performance_consistency(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
    ):
        """Test that performance remains consistent over time."""
        workspace = system_components.workspace_path

        # Measure performance at different points
        early_times = []
        late_times = []

        # Early performance
        for i in range(10):
            test_file = workspace / f"perf_early_{i}.txt"
            test_file.write_text(f"Early performance test {i}")

            start = time.time()
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-perf-{i}",
                ]
            )
            early_times.append(time.time() - start)
            await asyncio.sleep(2)

        # Intermediate load
        for i in range(100):
            test_file = workspace / f"perf_load_{i}.txt"
            test_file.write_text(f"Load test {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-perf-load-{i % 10}",
                ]
            )
            await asyncio.sleep(1)

        # Late performance
        for i in range(10):
            test_file = workspace / f"perf_late_{i}.txt"
            test_file.write_text(f"Late performance test {i}")

            start = time.time()
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-perf-late-{i}",
                ]
            )
            late_times.append(time.time() - start)
            await asyncio.sleep(2)

        # Compare performance
        avg_early = sum(early_times) / len(early_times)
        avg_late = sum(late_times) / len(late_times)

        # Performance shouldn't degrade significantly
        degradation_ratio = avg_late / avg_early if avg_early > 0 else 1
        assert degradation_ratio < 2.0  # Less than 2x slowdown


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.stability
class TestSystemRecoveryStability:
    """Test system recovery and stability after stress."""

    @pytest.mark.asyncio
    async def test_recovery_after_stress(
        self,
        system_components: SystemComponents,
        cli_helper: CLIHelper,
        resource_monitor: ResourceMonitor,
    ):
        """Test system recovers after stress period."""
        workspace = system_components.workspace_path

        # Start monitoring
        await resource_monitor.start_monitoring(system_components, interval=10)

        # Record baseline
        await asyncio.sleep(5)
        baseline_metrics = list(resource_monitor.metrics)

        # Stress period
        for i in range(50):
            test_file = workspace / f"stress_{i}.txt"
            test_file.write_text(f"Stress test {i}" * 100)
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-stress-{i % 5}",
                ]
            )
            await asyncio.sleep(0.5)

        # Recovery period
        await asyncio.sleep(30)

        # Stop monitoring
        metrics = await resource_monitor.stop_monitoring()

        # Verify recovery
        if len(metrics) > 10 and baseline_metrics:
            baseline_memory = baseline_metrics[-1].memory_mb
            final_memory = metrics[-1].memory_mb

            # Memory should recover somewhat
            recovery_ratio = final_memory / baseline_memory if baseline_memory > 0 else 1
            assert recovery_ratio < 2.5  # Some growth acceptable but not excessive
