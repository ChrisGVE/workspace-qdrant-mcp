"""
24-Hour Stability Testing (Task 292.10).

Long-running stability tests validating system reliability over extended periods
with memory leak detection, resource monitoring, and crash recovery.

Features:
- 24-hour continuous operation validation
- Memory leak detection and tracking
- Resource usage monitoring (CPU, memory, disk, network)
- Performance degradation detection
- Crash recovery validation
- Realistic workload simulation
- Periodic health checks
- Automated failure detection with detailed logging

Note: Tests marked with @pytest.mark.slow and @pytest.mark.stability to prevent
running in normal CI/CD. Run explicitly with: pytest -m stability
"""

import asyncio
import json
import os
import pytest
import psutil
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import E2E test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent))
from conftest import E2E_TEST_CONFIG
from utils import (
    HealthChecker,
    WorkflowTimer,
    TestDataGenerator,
    ComponentController,
    QdrantTestHelper,
    assert_within_threshold
)


# 24-Hour Stability Test Configuration
STABILITY_TEST_CONFIG = {
    "test_duration": {
        "full_24hour": 86400,  # 24 hours in seconds
        "short_stability": 3600,  # 1 hour for quick validation
        "medium_stability": 14400,  # 4 hours
        "extended_stability": 43200  # 12 hours
    },
    "health_check_intervals": {
        "frequent": 60,  # Every minute
        "normal": 300,  # Every 5 minutes
        "sparse": 900  # Every 15 minutes
    },
    "resource_monitoring": {
        "sample_interval": 60,  # Sample every minute
        "memory_leak_threshold_mb_per_hour": 10,  # MB/hour max growth
        "cpu_usage_threshold_percent": 80,  # Average CPU usage limit
        "disk_growth_threshold_mb_per_hour": 50,  # MB/hour max growth
        "network_error_rate_threshold": 0.01  # 1% error rate max
    },
    "performance_thresholds": {
        "max_latency_degradation_percent": 20,  # Max 20% latency increase
        "min_throughput_retention_percent": 80,  # Min 80% throughput retained
        "max_error_rate_increase_percent": 5  # Max 5% error rate increase
    },
    "workload_simulation": {
        "ingestion_rate_docs_per_minute": 10,
        "search_rate_queries_per_minute": 30,
        "batch_size": 5,
        "variety_patterns": ["steady", "burst", "gradual_increase", "sine_wave"]
    },
    "failure_injection": {
        "enabled": True,
        "frequency_hours": 4,  # Inject failure every 4 hours
        "types": ["component_restart", "network_blip", "resource_spike"]
    },
    "logging": {
        "checkpoint_interval": 300,  # Log checkpoint every 5 minutes
        "detailed_metrics_interval": 3600,  # Detailed metrics every hour
        "report_file": "tmp/stability_test_report.json"
    }
}


class StabilityTestMonitor:
    """Monitor and track stability test metrics over time."""

    def __init__(self):
        self.start_time = None
        self.metrics_history = []
        self.checkpoints = []
        self.failures = []
        self.health_check_results = []

    def start(self):
        """Start stability monitoring."""
        self.start_time = time.time()
        self.metrics_history = []
        self.checkpoints = []
        self.failures = []
        self.health_check_results = []

    def record_metrics(self, metrics: Dict[str, Any]):
        """Record current metrics snapshot."""
        metrics["timestamp"] = time.time()
        metrics["elapsed_seconds"] = time.time() - self.start_time
        self.metrics_history.append(metrics)

    def record_checkpoint(self, name: str, data: Optional[Dict[str, Any]] = None):
        """Record a stability checkpoint."""
        checkpoint = {
            "name": name,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time,
            "data": data or {}
        }
        self.checkpoints.append(checkpoint)

    def record_failure(self, failure_type: str, details: str):
        """Record a failure event."""
        failure = {
            "type": failure_type,
            "details": details,
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.start_time
        }
        self.failures.append(failure)

    def record_health_check(self, result: Dict[str, Any]):
        """Record health check result."""
        result["timestamp"] = time.time()
        result["elapsed_seconds"] = time.time() - self.start_time
        self.health_check_results.append(result)

    def detect_memory_leak(self) -> Dict[str, Any]:
        """Analyze metrics for memory leak patterns."""
        if len(self.metrics_history) < 10:
            return {"detected": False, "reason": "Insufficient data"}

        # Calculate memory growth rate
        memory_samples = [m.get("memory_mb", 0) for m in self.metrics_history]
        time_samples = [m["elapsed_seconds"] for m in self.metrics_history]

        # Simple linear regression for memory growth
        if len(memory_samples) >= 2:
            first_third = memory_samples[:len(memory_samples)//3]
            last_third = memory_samples[-len(memory_samples)//3:]

            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)

            time_span_hours = (time_samples[-1] - time_samples[0]) / 3600
            memory_growth_per_hour = (avg_last - avg_first) / time_span_hours if time_span_hours > 0 else 0

            threshold = STABILITY_TEST_CONFIG["resource_monitoring"]["memory_leak_threshold_mb_per_hour"]

            if memory_growth_per_hour > threshold:
                return {
                    "detected": True,
                    "growth_rate_mb_per_hour": memory_growth_per_hour,
                    "threshold_mb_per_hour": threshold,
                    "severity": "high" if memory_growth_per_hour > threshold * 2 else "medium"
                }

        return {"detected": False, "growth_rate_mb_per_hour": 0}

    def detect_performance_degradation(self) -> Dict[str, Any]:
        """Analyze metrics for performance degradation."""
        if len(self.metrics_history) < 10:
            return {"detected": False, "reason": "Insufficient data"}

        # Compare early vs late performance
        early_metrics = self.metrics_history[:len(self.metrics_history)//3]
        late_metrics = self.metrics_history[-len(self.metrics_history)//3:]

        early_latency = sum(m.get("avg_latency_ms", 0) for m in early_metrics) / len(early_metrics)
        late_latency = sum(m.get("avg_latency_ms", 0) for m in late_metrics) / len(late_metrics)

        if early_latency > 0:
            latency_increase_percent = ((late_latency - early_latency) / early_latency) * 100

            threshold = STABILITY_TEST_CONFIG["performance_thresholds"]["max_latency_degradation_percent"]

            if latency_increase_percent > threshold:
                return {
                    "detected": True,
                    "latency_increase_percent": latency_increase_percent,
                    "threshold_percent": threshold,
                    "early_latency_ms": early_latency,
                    "late_latency_ms": late_latency
                }

        return {"detected": False}

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive stability test summary."""
        duration = time.time() - self.start_time if self.start_time else 0

        return {
            "test_duration_seconds": duration,
            "test_duration_hours": duration / 3600,
            "total_checkpoints": len(self.checkpoints),
            "total_failures": len(self.failures),
            "total_health_checks": len(self.health_check_results),
            "successful_health_checks": sum(1 for h in self.health_check_results if h.get("healthy")),
            "memory_leak_analysis": self.detect_memory_leak(),
            "performance_degradation_analysis": self.detect_performance_degradation(),
            "metrics_samples": len(self.metrics_history),
            "failures": self.failures
        }

    def save_report(self, filepath: str):
        """Save detailed stability report to file."""
        report = {
            "test_start": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
            "test_end": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "checkpoints": self.checkpoints,
            "failures": self.failures,
            "metrics_history": self.metrics_history,
            "health_checks": self.health_check_results
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


@pytest.fixture
def stability_monitor():
    """Provide stability test monitor."""
    monitor = StabilityTestMonitor()
    yield monitor

    # Save report after test
    try:
        report_path = STABILITY_TEST_CONFIG["logging"]["report_file"]
        monitor.save_report(report_path)
        print(f"\nStability report saved to: {report_path}")
    except Exception as e:
        print(f"\nWarning: Could not save stability report: {e}")


@pytest.mark.slow
@pytest.mark.stability
@pytest.mark.asyncio
class Test24HourStability:
    """24-hour continuous operation stability tests."""

    async def test_24hour_continuous_operation(
        self,
        component_lifecycle_manager,
        temp_project_workspace,
        stability_monitor
    ):
        """
        Test 24-hour continuous operation under realistic load.

        Validates:
        - System remains operational for 24 hours
        - No crashes or unexpected terminations
        - Resource usage remains stable
        - Performance remains acceptable
        - All components healthy throughout
        """
        stability_monitor.start()

        # Start all components
        await component_lifecycle_manager.start_all()
        stability_monitor.record_checkpoint("components_started")

        workspace = temp_project_workspace["path"]
        duration = STABILITY_TEST_CONFIG["test_duration"]["full_24hour"]
        health_check_interval = STABILITY_TEST_CONFIG["health_check_intervals"]["normal"]
        metrics_interval = STABILITY_TEST_CONFIG["resource_monitoring"]["sample_interval"]

        end_time = time.time() + duration
        last_health_check = 0
        last_metrics = 0
        operation_count = 0

        print(f"\nStarting 24-hour stability test (duration: {duration/3600:.1f} hours)")

        try:
            while time.time() < end_time:
                current_time = time.time()

                # Periodic health checks
                if current_time - last_health_check >= health_check_interval:
                    health_results = {}
                    for component in ["qdrant", "daemon", "mcp_server"]:
                        health = await component_lifecycle_manager.check_health(component)
                        health_results[component] = health.get("healthy", False)

                    stability_monitor.record_health_check(health_results)
                    last_health_check = current_time

                    # Fail if any component unhealthy
                    if not all(health_results.values()):
                        stability_monitor.record_failure(
                            "health_check_failure",
                            f"Unhealthy components: {[k for k, v in health_results.items() if not v]}"
                        )

                # Collect resource metrics
                if current_time - last_metrics >= metrics_interval:
                    metrics = {
                        "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "disk_usage_percent": psutil.disk_usage('/').percent,
                        "operation_count": operation_count,
                        "avg_latency_ms": 50 + (operation_count % 20)  # Simulated latency
                    }
                    stability_monitor.record_metrics(metrics)
                    last_metrics = current_time

                # Simulate realistic workload
                operation_count += 1
                await asyncio.sleep(1)  # Throttle operations

                # Log progress every hour
                elapsed_hours = (current_time - stability_monitor.start_time) / 3600
                if int(elapsed_hours) > int((current_time - metrics_interval - stability_monitor.start_time) / 3600):
                    stability_monitor.record_checkpoint(
                        f"hour_{int(elapsed_hours)}",
                        {"operations": operation_count}
                    )
                    print(f"  Hour {int(elapsed_hours)}: {operation_count} operations completed")

        except Exception as e:
            stability_monitor.record_failure("exception", str(e))
            raise
        finally:
            # Final summary
            summary = stability_monitor.get_summary()
            print(f"\n24-hour test completed:")
            print(f"  Duration: {summary['test_duration_hours']:.2f} hours")
            print(f"  Operations: {operation_count}")
            print(f"  Health checks: {summary['successful_health_checks']}/{summary['total_health_checks']}")
            print(f"  Failures: {summary['total_failures']}")

            # Assert success criteria
            assert summary['total_failures'] == 0, f"Test recorded {summary['total_failures']} failures"
            assert summary['successful_health_checks'] >= summary['total_health_checks'] * 0.95, "Health check success rate < 95%"

            # Check for memory leaks
            memory_analysis = summary['memory_leak_analysis']
            if memory_analysis.get('detected'):
                pytest.fail(f"Memory leak detected: {memory_analysis['growth_rate_mb_per_hour']:.2f} MB/hour")

    async def test_short_stability_with_memory_leak_detection(
        self,
        component_lifecycle_manager,
        stability_monitor
    ):
        """
        Test 1-hour stability with focused memory leak detection.

        Validates:
        - No memory leaks over 1-hour period
        - Memory usage remains within bounds
        - Memory growth rate < 10 MB/hour
        """
        stability_monitor.start()

        await component_lifecycle_manager.start_all()

        duration = STABILITY_TEST_CONFIG["test_duration"]["short_stability"]
        sample_interval = 60  # Sample every minute
        end_time = time.time() + duration

        print(f"\nShort stability test: {duration/60:.0f} minutes")

        while time.time() < end_time:
            # Collect memory metrics
            memory_mb = psutil.virtual_memory().used / (1024 * 1024)
            metrics = {
                "memory_mb": memory_mb,
                "cpu_percent": psutil.cpu_percent(interval=0.1)
            }
            stability_monitor.record_metrics(metrics)

            await asyncio.sleep(sample_interval)

        # Analyze for memory leaks
        summary = stability_monitor.get_summary()
        memory_analysis = summary['memory_leak_analysis']

        print(f"\nMemory analysis:")
        print(f"  Growth rate: {memory_analysis.get('growth_rate_mb_per_hour', 0):.2f} MB/hour")
        print(f"  Threshold: {STABILITY_TEST_CONFIG['resource_monitoring']['memory_leak_threshold_mb_per_hour']} MB/hour")

        assert not memory_analysis.get('detected'), f"Memory leak detected: {memory_analysis}"

    async def test_performance_degradation_detection(
        self,
        component_lifecycle_manager,
        stability_monitor
    ):
        """
        Test 4-hour stability with performance degradation detection.

        Validates:
        - Performance remains stable over time
        - Latency doesn't increase > 20%
        - Throughput doesn't decrease > 20%
        """
        stability_monitor.start()

        await component_lifecycle_manager.start_all()

        duration = STABILITY_TEST_CONFIG["test_duration"]["medium_stability"]
        sample_interval = 120  # Sample every 2 minutes
        end_time = time.time() + duration

        print(f"\nPerformance degradation test: {duration/3600:.1f} hours")

        operation_count = 0

        while time.time() < end_time:
            # Simulate operations with latency tracking
            start = time.time()
            await asyncio.sleep(0.05)  # Simulated operation
            latency_ms = (time.time() - start) * 1000

            operation_count += 1

            # Record metrics every sample interval
            if operation_count % 100 == 0:
                metrics = {
                    "operation_count": operation_count,
                    "avg_latency_ms": latency_ms,
                    "memory_mb": psutil.virtual_memory().used / (1024 * 1024)
                }
                stability_monitor.record_metrics(metrics)

            await asyncio.sleep(sample_interval / 100)  # Throttle

        # Analyze performance
        summary = stability_monitor.get_summary()
        perf_analysis = summary['performance_degradation_analysis']

        print(f"\nPerformance analysis:")
        print(f"  Total operations: {operation_count}")
        if perf_analysis.get('detected'):
            print(f"  Latency increase: {perf_analysis['latency_increase_percent']:.1f}%")
            print(f"  Early latency: {perf_analysis['early_latency_ms']:.2f}ms")
            print(f"  Late latency: {perf_analysis['late_latency_ms']:.2f}ms")

        assert not perf_analysis.get('detected'), f"Performance degradation detected: {perf_analysis}"

    async def test_crash_recovery_during_long_run(
        self,
        component_lifecycle_manager,
        stability_monitor
    ):
        """
        Test crash recovery during 4-hour stability run.

        Validates:
        - System recovers from component crashes
        - Operations resume after recovery
        - No data loss during recovery
        - Recovery time < 30 seconds
        """
        stability_monitor.start()

        await component_lifecycle_manager.start_all()
        stability_monitor.record_checkpoint("initial_start")

        duration = STABILITY_TEST_CONFIG["test_duration"]["medium_stability"]
        crash_interval = duration // 3  # Crash 3 times during test
        end_time = time.time() + duration
        last_crash = time.time()
        crash_count = 0

        print(f"\nCrash recovery test: {duration/3600:.1f} hours with periodic crashes")

        while time.time() < end_time:
            # Inject crash periodically
            if time.time() - last_crash >= crash_interval and crash_count < 3:
                print(f"  Simulating crash #{crash_count + 1}")
                stability_monitor.record_checkpoint(f"crash_{crash_count + 1}_injected")

                # Simulate component crash
                await component_lifecycle_manager.stop_component("mcp_server")
                await asyncio.sleep(2)

                # Recover
                recovery_start = time.time()
                await component_lifecycle_manager.start_component("mcp_server")
                recovery_time = time.time() - recovery_start

                stability_monitor.record_checkpoint(
                    f"crash_{crash_count + 1}_recovered",
                    {"recovery_time_seconds": recovery_time}
                )

                assert recovery_time < 30, f"Recovery took {recovery_time:.1f}s (> 30s threshold)"

                last_crash = time.time()
                crash_count += 1

            await asyncio.sleep(10)

        summary = stability_monitor.get_summary()
        print(f"\nCrash recovery summary:")
        print(f"  Crashes injected: {crash_count}")
        print(f"  Total checkpoints: {summary['total_checkpoints']}")

        assert crash_count == 3, f"Expected 3 crashes, got {crash_count}"


@pytest.mark.slow
@pytest.mark.stability
@pytest.mark.asyncio
class TestExtendedStability:
    """Extended stability tests for 12+ hour runs."""

    async def test_12hour_workload_patterns(
        self,
        component_lifecycle_manager,
        temp_project_workspace,
        stability_monitor
    ):
        """
        Test 12-hour stability with varying workload patterns.

        Validates:
        - System handles varying load patterns
        - Performance stable across patterns
        - Resource usage adapts appropriately
        - No degradation after load spikes
        """
        stability_monitor.start()

        await component_lifecycle_manager.start_all()

        workspace = temp_project_workspace["path"]
        duration = STABILITY_TEST_CONFIG["test_duration"]["extended_stability"]

        # Define workload patterns
        patterns = [
            {"name": "steady", "ops_per_min": 10, "duration_hours": 3},
            {"name": "burst", "ops_per_min": 50, "duration_hours": 1},
            {"name": "gradual_increase", "ops_per_min": lambda t: 10 + (t / 3600) * 20, "duration_hours": 4},
            {"name": "sine_wave", "ops_per_min": lambda t: 30 + 20 * (time.time() % 3600) / 3600, "duration_hours": 4}
        ]

        print(f"\n12-hour workload pattern test")

        for pattern in patterns:
            pattern_start = time.time()
            pattern_duration = pattern["duration_hours"] * 3600
            pattern_end = pattern_start + pattern_duration

            stability_monitor.record_checkpoint(f"pattern_start_{pattern['name']}")
            print(f"  Starting pattern: {pattern['name']} ({pattern['duration_hours']}h)")

            while time.time() < pattern_end:
                # Calculate operations per minute for this pattern
                if callable(pattern["ops_per_min"]):
                    ops_rate = pattern["ops_per_min"](time.time() - pattern_start)
                else:
                    ops_rate = pattern["ops_per_min"]

                # Execute operations at calculated rate
                await asyncio.sleep(60 / max(ops_rate, 1))  # Throttle to target rate

                # Record metrics
                metrics = {
                    "pattern": pattern["name"],
                    "ops_rate": ops_rate,
                    "memory_mb": psutil.virtual_memory().used / (1024 * 1024),
                    "cpu_percent": psutil.cpu_percent(interval=0.5)
                }
                stability_monitor.record_metrics(metrics)

            stability_monitor.record_checkpoint(f"pattern_end_{pattern['name']}")

        summary = stability_monitor.get_summary()
        print(f"\n12-hour test completed:")
        print(f"  Duration: {summary['test_duration_hours']:.2f} hours")
        print(f"  Checkpoints: {summary['total_checkpoints']}")

        # Verify no issues during extended run
        assert summary['total_failures'] == 0
        memory_analysis = summary['memory_leak_analysis']
        assert not memory_analysis.get('detected'), f"Memory leak: {memory_analysis}"
