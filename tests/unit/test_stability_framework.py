"""
Unit Tests for 24-Hour Stability Testing Framework

Tests cover:
- ResourceMonitor: Metrics collection and monitoring
- MemoryLeakDetector: Trend analysis and leak detection
- StabilityMetricsCollector: Checkpoint recording and scoring
- StabilityTestManager: Test lifecycle, checkpoint/resume

Task: #305.3
"""

import asyncio
import tempfile
import time
from pathlib import Path
from typing import List
import pytest
import sqlite3

from tests.framework.stress_orchestration import (
    ResourceMonitor,
    ResourceMetrics,
    MemoryLeakDetector,
    MemoryLeakReport,
    StabilityMetricsCollector,
    StabilityTestManager,
    StabilityTestConfig,
    StabilityTestReport,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def stability_manager(temp_db):
    """Create StabilityTestManager with temp database."""
    return StabilityTestManager(temp_db)


@pytest.fixture
def metrics_collector(temp_db):
    """Create StabilityMetricsCollector with temp database."""
    # Initialize tables via StabilityTestManager
    manager = StabilityTestManager(temp_db)
    return StabilityMetricsCollector(temp_db)


class TestResourceMonitor:
    """Test ResourceMonitor functionality."""

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping resource monitoring."""
        monitor = ResourceMonitor(component_name="test_component")

        # Start monitoring with short interval
        await monitor.start_monitoring(interval_seconds=1)

        # Let it collect a few samples
        await asyncio.sleep(2.5)

        # Stop monitoring
        await monitor.stop_monitoring()

        # Verify metrics collected
        metrics = monitor.get_metrics()
        assert len(metrics) >= 2, "Should collect at least 2 samples"

        # Verify metric structure
        sample = metrics[0]
        assert sample.component_name == "test_component"
        assert sample.cpu_percent >= 0
        assert sample.memory_rss_mb > 0
        assert sample.memory_available_mb > 0
        assert sample.timestamp > 0

    @pytest.mark.asyncio
    async def test_metrics_history_accumulation(self):
        """Test that metrics accumulate over time."""
        monitor = ResourceMonitor()

        await monitor.start_monitoring(interval_seconds=1)
        await asyncio.sleep(1.5)

        first_count = len(monitor.get_metrics())
        assert first_count >= 1

        await asyncio.sleep(1.5)
        second_count = len(monitor.get_metrics())
        assert second_count > first_count

        await monitor.stop_monitoring()

    @pytest.mark.asyncio
    async def test_detect_memory_leak_insufficient_data(self):
        """Test memory leak detection with insufficient data."""
        monitor = ResourceMonitor()

        # No metrics yet
        leak_report = monitor.detect_memory_leak(threshold_mb_per_hour=5.0)
        assert leak_report is not None
        assert not leak_report.detected
        assert leak_report.samples_analyzed == 0

        # Add single metric
        monitor.metrics_history.append(
            ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=10.0,
                memory_rss_mb=100.0,
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                net_sent_mb=0.0,
                net_recv_mb=0.0,
                open_fds=10
            )
        )

        # With only 1 sample, detector returns report with detected=False
        leak_report = monitor.detect_memory_leak(threshold_mb_per_hour=5.0)
        assert leak_report is not None
        assert not leak_report.detected
        assert leak_report.samples_analyzed == 1


class TestMemoryLeakDetector:
    """Test MemoryLeakDetector functionality."""

    def test_calculate_growth_rate_no_leak(self):
        """Test growth rate calculation with stable memory."""
        detector = MemoryLeakDetector()

        # Create metrics with stable memory (100 MB)
        base_time = time.time()
        metrics = [
            ResourceMetrics(
                timestamp=base_time + i * 60,
                cpu_percent=10.0,
                memory_rss_mb=100.0,
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                net_sent_mb=0.0,
                net_recv_mb=0.0,
                open_fds=10
            )
            for i in range(10)
        ]

        values = [m.memory_rss_mb for m in metrics]
        timestamps = [m.timestamp for m in metrics]

        growth_rate = detector.calculate_growth_rate(values, timestamps)

        # Should be near zero for stable memory
        assert abs(growth_rate) < 0.1

    def test_calculate_growth_rate_with_leak(self):
        """Test growth rate calculation with memory leak."""
        detector = MemoryLeakDetector()

        # Create metrics with growing memory (10 MB/hour)
        base_time = time.time()
        metrics = [
            ResourceMetrics(
                timestamp=base_time + i * 360,  # 6 minutes apart
                cpu_percent=10.0,
                memory_rss_mb=100.0 + i * 1.0,  # Grow by 1 MB per sample
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                net_sent_mb=0.0,
                net_recv_mb=0.0,
                open_fds=10
            )
            for i in range(10)
        ]

        values = [m.memory_rss_mb for m in metrics]
        timestamps = [m.timestamp for m in metrics]

        growth_rate = detector.calculate_growth_rate(values, timestamps)

        # Should detect ~10 MB/hour growth
        assert growth_rate > 9.0
        assert growth_rate < 11.0

    def test_is_leak_detected(self):
        """Test leak detection threshold."""
        detector = MemoryLeakDetector()

        # Below threshold
        assert not detector.is_leak_detected(growth_rate=4.0, threshold=5.0)

        # At threshold
        assert not detector.is_leak_detected(growth_rate=5.0, threshold=5.0)

        # Above threshold
        assert detector.is_leak_detected(growth_rate=6.0, threshold=5.0)

    def test_analyze_memory_trend_no_leak(self):
        """Test full memory trend analysis with no leak."""
        detector = MemoryLeakDetector()

        base_time = time.time()
        metrics = [
            ResourceMetrics(
                timestamp=base_time + i * 60,
                cpu_percent=10.0,
                memory_rss_mb=100.0,
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                net_sent_mb=0.0,
                net_recv_mb=0.0,
                open_fds=10
            )
            for i in range(20)
        ]

        report = detector.analyze_memory_trend(
            metrics,
            threshold_mb_per_hour=5.0,
            component_name="test_component"
        )

        assert not report.detected
        assert report.affected_component == "test_component"
        assert report.samples_analyzed == 20
        assert abs(report.growth_rate_mb_per_hour) < 0.5
        assert report.confidence > 0

    def test_analyze_memory_trend_with_leak(self):
        """Test full memory trend analysis with leak."""
        detector = MemoryLeakDetector()

        base_time = time.time()
        metrics = [
            ResourceMetrics(
                timestamp=base_time + i * 360,  # 6 minutes
                cpu_percent=10.0,
                memory_rss_mb=100.0 + i * 1.0,  # 10 MB/hour growth
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                net_sent_mb=0.0,
                net_recv_mb=0.0,
                open_fds=10
            )
            for i in range(20)
        ]

        report = detector.analyze_memory_trend(
            metrics,
            threshold_mb_per_hour=5.0,
            component_name="test_component"
        )

        assert report.detected
        assert report.growth_rate_mb_per_hour > 5.0
        assert report.start_memory_mb == 100.0
        assert report.end_memory_mb == 119.0
        assert report.confidence > 0


class TestStabilityMetricsCollector:
    """Test StabilityMetricsCollector functionality."""

    def test_record_checkpoint(self, metrics_collector):
        """Test checkpoint recording."""
        run_id = "test_run_123"

        # Record checkpoint
        metrics = {
            "elapsed_hours": 1.5,
            "errors": 5,
            "downtime_seconds": 30.0,
            "resource_samples": 100
        }

        metrics_collector.record_checkpoint(run_id, 1.5, metrics)

        # Verify checkpoint saved
        with sqlite3.connect(metrics_collector.database_path) as conn:
            row = conn.execute("""
                SELECT run_id, elapsed_hours, metrics_json
                FROM stability_checkpoints
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row is not None
        assert row[0] == run_id
        assert row[1] == 1.5

    def test_get_stability_score_perfect(self, metrics_collector, temp_db):
        """Test stability score calculation with perfect run."""
        run_id = "perfect_run"

        # Create perfect stability run
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO stability_runs
                (run_id, config_json, start_time, status, duration_hours,
                 total_errors, uptime_percentage, performance_degradation_percent,
                 memory_leak_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, '{}', time.time(), 'completed', 24.0,
                0, 100.0, 0.0, 0
            ))

        score = metrics_collector.get_stability_score(run_id)

        # Perfect score: 100% uptime (40) + 0 errors (30) + 0 degradation (20) + no leak (10)
        assert score == 100.0

    def test_get_stability_score_with_issues(self, metrics_collector, temp_db):
        """Test stability score calculation with issues."""
        run_id = "problematic_run"

        # Create problematic run
        with sqlite3.connect(temp_db) as conn:
            conn.execute("""
                INSERT INTO stability_runs
                (run_id, config_json, start_time, status, duration_hours,
                 total_errors, uptime_percentage, performance_degradation_percent,
                 memory_leak_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, '{}', time.time(), 'completed', 24.0,
                120, 95.0, 25.0, 1  # 5 errors/hour, 95% uptime, 25% degradation, leak
            ))

        score = metrics_collector.get_stability_score(run_id)

        # Should be less than perfect but > 0
        assert 0 < score < 100.0
        assert score < 90.0  # With these issues, should be notably lower

    def test_summarize_resources(self, metrics_collector):
        """Test resource summarization."""
        base_time = time.time()
        metrics = [
            ResourceMetrics(
                timestamp=base_time + i,
                cpu_percent=10.0 + i,
                memory_rss_mb=100.0 + i * 10,
                memory_vms_mb=200.0,
                memory_available_mb=1000.0,
                disk_read_mb=float(i),
                disk_write_mb=float(i * 2),
                net_sent_mb=float(i),
                net_recv_mb=float(i * 3),
                open_fds=10 + i
            )
            for i in range(10)
        ]

        summary = metrics_collector._summarize_resources(metrics)

        # Verify summary structure
        assert "cpu_percent" in summary
        assert "memory_rss_mb" in summary
        assert "disk_read_mb" in summary

        # Verify CPU summary
        cpu_summary = summary["cpu_percent"]
        assert "avg" in cpu_summary
        assert "min" in cpu_summary
        assert "max" in cpu_summary
        assert "p95" in cpu_summary

        # Verify calculations
        assert cpu_summary["min"] == 10.0
        assert cpu_summary["max"] == 19.0
        assert 14.0 < cpu_summary["avg"] < 15.0


class TestStabilityTestManager:
    """Test StabilityTestManager functionality."""

    @pytest.mark.asyncio
    async def test_start_stability_run(self, stability_manager):
        """Test starting a stability run."""
        config = StabilityTestConfig(
            duration_hours=1.0,
            checkpoint_interval_minutes=10,
            resource_monitoring_interval_seconds=5,
            memory_leak_detection_enabled=True
        )

        run_id = await stability_manager.start_stability_run(config)

        assert run_id.startswith("stability_")
        assert run_id in stability_manager.active_runs
        assert run_id in stability_manager.resource_monitors

        # Verify database record
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT run_id, status
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row is not None
        assert row[0] == run_id
        assert row[1] == "running"

        # Cleanup
        await stability_manager.stop_run(run_id, "test_cleanup")

    @pytest.mark.asyncio
    async def test_checkpoint_run(self, stability_manager):
        """Test checkpointing a stability run."""
        config = StabilityTestConfig(duration_hours=1.0)
        run_id = await stability_manager.start_stability_run(config)

        # Let monitoring collect some data
        await asyncio.sleep(0.5)

        # Checkpoint the run
        await stability_manager.checkpoint_run(run_id)

        # Verify checkpoint saved
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT run_id, elapsed_hours
                FROM stability_checkpoints
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row is not None
        assert row[0] == run_id
        assert row[1] >= 0

        # Verify checkpoint counter incremented
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT checkpoints_completed
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row[0] == 1

        # Cleanup
        await stability_manager.stop_run(run_id, "test_cleanup")

    @pytest.mark.asyncio
    async def test_stop_run(self, stability_manager):
        """Test stopping a stability run."""
        config = StabilityTestConfig(
            duration_hours=1.0,
            memory_leak_detection_enabled=True,
            memory_leak_threshold_mb_per_hour=5.0
        )
        run_id = await stability_manager.start_stability_run(config)

        # Let it run briefly
        await asyncio.sleep(1.0)

        # Stop the run
        report = await stability_manager.stop_run(run_id, "manual_stop")

        assert isinstance(report, StabilityTestReport)
        assert report.run_id == run_id
        assert report.duration_hours > 0
        assert 0 <= report.stability_score <= 100
        assert report.test_stopped_reason == "manual_stop"

        # Verify run not in active runs
        assert run_id not in stability_manager.active_runs

        # Verify database updated
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT status, stopped_reason
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row[0] == "completed"
        assert row[1] == "manual_stop"

    @pytest.mark.asyncio
    async def test_resume_run(self, stability_manager):
        """Test resuming a stability run from checkpoint."""
        config = StabilityTestConfig(duration_hours=1.0)
        run_id = await stability_manager.start_stability_run(config)

        # Checkpoint the run
        await asyncio.sleep(0.5)
        await stability_manager.checkpoint_run(run_id)

        # Simulate crash by removing from active runs
        original_errors = stability_manager.active_runs[run_id]['errors']
        del stability_manager.active_runs[run_id]
        await stability_manager.resource_monitors[run_id].stop_monitoring()
        del stability_manager.resource_monitors[run_id]

        # Resume the run
        await stability_manager.resume_run(run_id)

        # Verify run resumed
        assert run_id in stability_manager.active_runs
        assert run_id in stability_manager.resource_monitors
        assert stability_manager.active_runs[run_id]['errors'] == original_errors

        # Verify status updated
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT status
                FROM stability_runs
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row[0] == "running"

        # Cleanup
        await stability_manager.stop_run(run_id, "test_cleanup")

    @pytest.mark.asyncio
    async def test_resume_completed_run_fails(self, stability_manager):
        """Test that resuming completed run raises error."""
        config = StabilityTestConfig(duration_hours=1.0)
        run_id = await stability_manager.start_stability_run(config)

        # Complete the run
        await stability_manager.stop_run(run_id, "completed")

        # Attempt to resume should fail
        with pytest.raises(ValueError, match="already completed"):
            await stability_manager.resume_run(run_id)

    @pytest.mark.asyncio
    async def test_memory_leak_detection_integration(self, stability_manager):
        """Test memory leak detection during stability run."""
        config = StabilityTestConfig(
            duration_hours=1.0,
            resource_monitoring_interval_seconds=1,
            memory_leak_detection_enabled=True,
            memory_leak_threshold_mb_per_hour=1.0  # Very low threshold for testing
        )

        run_id = await stability_manager.start_stability_run(config)

        # Manually inject growing memory metrics to simulate leak
        monitor = stability_manager.resource_monitors[run_id]
        base_time = time.time()

        for i in range(10):
            monitor.metrics_history.append(
                ResourceMetrics(
                    timestamp=base_time + i * 360,  # 6 minutes apart
                    cpu_percent=10.0,
                    memory_rss_mb=100.0 + i * 2.0,  # Grow 20 MB/hour
                    memory_vms_mb=200.0,
                    memory_available_mb=1000.0,
                    disk_read_mb=0.0,
                    disk_write_mb=0.0,
                    net_sent_mb=0.0,
                    net_recv_mb=0.0,
                    open_fds=10
                )
            )

        # Stop and check report
        report = await stability_manager.stop_run(run_id, "test_complete")

        assert report.memory_leak_detected
        assert report.memory_growth_rate_mb_per_hour > 1.0

        # Verify leak report saved
        with sqlite3.connect(stability_manager.database_path) as conn:
            row = conn.execute("""
                SELECT detected, growth_rate_mb_per_hour
                FROM memory_leak_reports
                WHERE run_id = ?
            """, (run_id,)).fetchone()

        assert row is not None
        assert row[0] == 1  # detected
        assert row[1] > 1.0

    @pytest.mark.asyncio
    async def test_resource_samples_saved(self, stability_manager):
        """Test that resource samples are persisted to database."""
        config = StabilityTestConfig(
            duration_hours=1.0,
            resource_monitoring_interval_seconds=1
        )

        run_id = await stability_manager.start_stability_run(config)

        # Let monitoring collect samples
        await asyncio.sleep(2.5)

        # Checkpoint to save samples
        await stability_manager.checkpoint_run(run_id)

        # Verify samples saved
        with sqlite3.connect(stability_manager.database_path) as conn:
            count = conn.execute("""
                SELECT COUNT(*)
                FROM resource_samples
                WHERE run_id = ?
            """, (run_id,)).fetchone()[0]

        assert count >= 2

        # Cleanup
        await stability_manager.stop_run(run_id, "test_cleanup")


class TestStabilityTestConfig:
    """Test StabilityTestConfig defaults and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StabilityTestConfig()

        assert config.duration_hours == 24.0
        assert config.checkpoint_interval_minutes == 30
        assert config.resource_monitoring_interval_seconds == 60
        assert config.memory_leak_detection_enabled is True
        assert config.auto_stop_on_degradation is False
        assert config.memory_leak_threshold_mb_per_hour == 5.0

    def test_custom_config(self):
        """Test custom configuration values."""
        config = StabilityTestConfig(
            duration_hours=48.0,
            checkpoint_interval_minutes=15,
            memory_leak_threshold_mb_per_hour=10.0
        )

        assert config.duration_hours == 48.0
        assert config.checkpoint_interval_minutes == 15
        assert config.memory_leak_threshold_mb_per_hour == 10.0


class TestStabilityTestReport:
    """Test StabilityTestReport structure."""

    def test_report_creation(self):
        """Test creating stability test report."""
        report = StabilityTestReport(
            run_id="test_run",
            duration_hours=24.0,
            stability_score=95.5,
            uptime_percentage=99.9,
            total_errors=10,
            error_rate_per_hour=0.42,
            memory_leak_detected=False,
            memory_growth_rate_mb_per_hour=1.2,
            performance_degradation_percent=5.0,
            recovery_incidents=[],
            resource_usage_summary={
                "cpu_percent": {"avg": 15.0, "min": 5.0, "max": 50.0, "p95": 45.0},
                "memory_rss_mb": {"avg": 500.0, "min": 450.0, "max": 550.0, "p95": 540.0}
            },
            checkpoints_completed=48,
            test_stopped_reason="completed"
        )

        assert report.run_id == "test_run"
        assert report.stability_score == 95.5
        assert not report.memory_leak_detected
        assert report.checkpoints_completed == 48
        assert "cpu_percent" in report.resource_usage_summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
