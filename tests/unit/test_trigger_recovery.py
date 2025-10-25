"""
Unit tests for trigger recovery and event logging system (Task 301.6).

Tests the trigger recovery, retry logic, event logging, and health monitoring
functionality including:
- Trigger retry with exponential backoff
- Event logging with timestamps and metadata
- Health monitoring and alerting
- Performance analytics
- Missed trigger detection
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.python.common.core.context_injection.session_trigger import (
    SessionTrigger,
    TriggerContext,
    TriggerEvent,
    TriggerEventLogger,
    TriggerHealthMetrics,
    TriggerHealthMonitor,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerResult,
    TriggerRetryPolicy,
)
from src.python.common.memory.types import AuthorityLevel, MemoryCategory, MemoryRule


class FailingTrigger(SessionTrigger):
    """Test trigger that fails configurable number of times."""

    def __init__(self, name: str, failures_before_success: int = 0, error_message: str = "Test failure"):
        super().__init__(name=name, phase=TriggerPhase.PRE_SESSION)
        self.failures_before_success = failures_before_success
        self.error_message = error_message
        self.attempt_count = 0

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute trigger, failing N times before succeeding."""
        import time
        start = time.time()

        self.attempt_count += 1

        if self.attempt_count <= self.failures_before_success:
            # Fail
            execution_time = (time.time() - start) * 1000
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=f"{self.error_message} (attempt {self.attempt_count})",
            )
        else:
            # Succeed
            execution_time = (time.time() - start) * 1000
            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
            )


@pytest.fixture
def mock_memory_manager():
    """Provide mock MemoryManager."""
    manager = AsyncMock()
    manager.get_rules = AsyncMock(return_value=[])
    return manager


@pytest.fixture
def mock_claude_code_session():
    """Provide mock ClaudeCodeSession."""
    from src.python.common.core.context_injection.claude_code_detector import (
        ClaudeCodeSession,
    )

    return ClaudeCodeSession(
        is_active=True,
        entrypoint="cli",
        detection_method="test",
        session_id="test_session",
    )


@pytest.mark.asyncio
class TestRetryPolicy:
    """Test retry policy configuration."""

    def test_initialization(self):
        """Test retry policy initialization."""
        policy = TriggerRetryPolicy(
            max_retries=5,
            initial_delay_seconds=2.0,
            max_delay_seconds=120.0,
            exponential_base=3.0,
            jitter=0.2,
        )

        assert policy.max_retries == 5
        assert policy.initial_delay_seconds == 2.0
        assert policy.max_delay_seconds == 120.0
        assert policy.exponential_base == 3.0
        assert policy.jitter == 0.2

    def test_default_retryable_errors(self):
        """Test default retryable error patterns."""
        policy = TriggerRetryPolicy()

        assert policy.is_retryable("Connection timeout error")
        assert policy.is_retryable("Temporary failure occurred")
        assert policy.is_retryable("Transient network issue")
        assert not policy.is_retryable("Fatal error")
        assert not policy.is_retryable("Invalid configuration")

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        policy = TriggerRetryPolicy(
            initial_delay_seconds=1.0,
            exponential_base=2.0,
            jitter=0.0,  # No jitter for testing
        )

        delay0 = policy.calculate_delay(0)
        delay1 = policy.calculate_delay(1)
        delay2 = policy.calculate_delay(2)

        assert delay0 == 1.0  # 1 * 2^0
        assert delay1 == 2.0  # 1 * 2^1
        assert delay2 == 4.0  # 1 * 2^2

    def test_max_delay_capping(self):
        """Test that delay is capped at max_delay."""
        policy = TriggerRetryPolicy(
            initial_delay_seconds=1.0,
            max_delay_seconds=10.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        delay10 = policy.calculate_delay(10)  # Would be 1024 without cap
        assert delay10 == 10.0

    def test_jitter_variation(self):
        """Test that jitter adds variation to delays."""
        policy = TriggerRetryPolicy(
            initial_delay_seconds=10.0,
            exponential_base=2.0,
            jitter=0.5,  # 50% jitter
        )

        # Calculate multiple delays and verify they differ
        delays = [policy.calculate_delay(1) for _ in range(10)]
        unique_delays = set(delays)

        # With jitter, we should get different values
        assert len(unique_delays) > 1


@pytest.mark.asyncio
class TestEventLogger:
    """Test event logger functionality."""

    def test_initialization(self):
        """Test event logger initialization."""
        logger = TriggerEventLogger(max_events=500)
        assert logger._max_events == 500
        assert len(logger._events) == 0

    def test_log_event(self):
        """Test logging events."""
        logger = TriggerEventLogger()

        event = TriggerEvent(
            timestamp=time.time(),
            trigger_name="test_trigger",
            phase=TriggerPhase.PRE_SESSION,
            event_type="completed",
            success=True,
            execution_time_ms=100.0,
        )

        logger.log_event(event)

        events = logger.get_events()
        assert len(events) == 1
        assert events[0].trigger_name == "test_trigger"

    def test_max_events_trimming(self):
        """Test that events are trimmed to max size."""
        logger = TriggerEventLogger(max_events=10)

        # Log 20 events
        for i in range(20):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name=f"trigger_{i}",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=True,
            )
            logger.log_event(event)

        events = logger.get_events()
        assert len(events) == 10  # Only most recent 10

        # Verify we kept the most recent ones
        trigger_names = [e.trigger_name for e in events]
        assert "trigger_19" in trigger_names
        assert "trigger_0" not in trigger_names

    def test_filter_by_trigger_name(self):
        """Test filtering events by trigger name."""
        logger = TriggerEventLogger()

        for name in ["trigger_a", "trigger_b", "trigger_a"]:
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name=name,
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=True,
            )
            logger.log_event(event)

        events_a = logger.get_events(trigger_name="trigger_a")
        assert len(events_a) == 2

    def test_filter_by_success(self):
        """Test filtering events by success status."""
        logger = TriggerEventLogger()

        for success in [True, False, True, False]:
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name="test",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=success,
            )
            logger.log_event(event)

        failures = logger.get_events(success=False)
        assert len(failures) == 2

    def test_get_recent_failures(self):
        """Test getting recent failures."""
        logger = TriggerEventLogger()

        # Log failures at different times
        now = time.time()
        old_event = TriggerEvent(
            timestamp=now - 7200,  # 2 hours ago
            trigger_name="old_fail",
            phase=TriggerPhase.PRE_SESSION,
            event_type="failed",
            success=False,
        )
        logger.log_event(old_event)

        recent_event = TriggerEvent(
            timestamp=now,
            trigger_name="recent_fail",
            phase=TriggerPhase.PRE_SESSION,
            event_type="failed",
            success=False,
        )
        logger.log_event(recent_event)

        # Get failures from last hour
        recent_failures = logger.get_recent_failures(minutes=60)
        assert len(recent_failures) == 1
        assert recent_failures[0].trigger_name == "recent_fail"

    def test_clear_events(self):
        """Test clearing events."""
        logger = TriggerEventLogger()

        for i in range(5):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name=f"trigger_{i}",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=True,
            )
            logger.log_event(event)

        cleared = logger.clear_events()
        assert cleared == 5
        assert len(logger.get_events()) == 0


@pytest.mark.asyncio
class TestHealthMonitor:
    """Test health monitoring functionality."""

    def test_initialization(self):
        """Test health monitor initialization."""
        monitor = TriggerHealthMonitor(
            failure_threshold=5,
            failure_rate_threshold=0.3,
        )

        assert monitor._failure_threshold == 5
        assert monitor._failure_rate_threshold == 0.3

    def test_record_successful_event(self):
        """Test recording successful event."""
        monitor = TriggerHealthMonitor()

        event = TriggerEvent(
            timestamp=time.time(),
            trigger_name="test_trigger",
            phase=TriggerPhase.PRE_SESSION,
            event_type="completed",
            success=True,
            execution_time_ms=50.0,
        )

        monitor.record_event(event)

        metrics = monitor.get_metrics("test_trigger")
        assert "test_trigger" in metrics
        assert metrics["test_trigger"].total_executions == 1
        assert metrics["test_trigger"].successful_executions == 1
        assert metrics["test_trigger"].consecutive_failures == 0

    def test_record_failed_event(self):
        """Test recording failed event."""
        monitor = TriggerHealthMonitor()

        event = TriggerEvent(
            timestamp=time.time(),
            trigger_name="test_trigger",
            phase=TriggerPhase.PRE_SESSION,
            event_type="failed",
            success=False,
            error="Test error",
        )

        monitor.record_event(event)

        metrics = monitor.get_metrics("test_trigger")
        assert metrics["test_trigger"].failed_executions == 1
        assert metrics["test_trigger"].consecutive_failures == 1

    def test_consecutive_failure_alert(self):
        """Test alert on consecutive failures."""
        monitor = TriggerHealthMonitor(failure_threshold=3)

        # Record 3 failures
        for i in range(3):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name="failing_trigger",
                phase=TriggerPhase.PRE_SESSION,
                event_type="failed",
                success=False,
                error=f"Failure {i+1}",
            )
            monitor.record_event(event)

        # Check for alerts
        alerts = monitor.get_alerts()
        assert len(alerts) > 0
        assert "failing_trigger" in alerts[0]
        assert "3 consecutive failures" in alerts[0]

    def test_failure_rate_alert(self):
        """Test alert on high failure rate."""
        monitor = TriggerHealthMonitor(failure_rate_threshold=0.5)

        # Record 10 executions: 6 failures, 4 successes
        for i in range(10):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name="unreliable_trigger",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=i < 4,  # First 4 succeed, rest fail
                error="Error" if i >= 4 else None,
            )
            monitor.record_event(event)

        # Check for alerts (60% failure rate > 50% threshold)
        alerts = monitor.get_alerts()
        assert len(alerts) > 0
        assert any("60.0%" in alert for alert in alerts)

    def test_get_unhealthy_triggers(self):
        """Test identifying unhealthy triggers."""
        monitor = TriggerHealthMonitor(
            failure_threshold=2,
            failure_rate_threshold=0.5,
        )

        # Healthy trigger
        for _i in range(5):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name="healthy_trigger",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=True,
            )
            monitor.record_event(event)

        # Unhealthy trigger (consecutive failures)
        for _i in range(3):
            event = TriggerEvent(
                timestamp=time.time(),
                trigger_name="unhealthy_trigger",
                phase=TriggerPhase.PRE_SESSION,
                event_type="failed",
                success=False,
            )
            monitor.record_event(event)

        unhealthy = monitor.get_unhealthy_triggers()
        assert "unhealthy_trigger" in unhealthy
        assert "healthy_trigger" not in unhealthy

    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = TriggerHealthMonitor()

        # Record events for multiple triggers
        for trigger in ["trigger_a", "trigger_b"]:
            for i in range(5):
                event = TriggerEvent(
                    timestamp=time.time(),
                    trigger_name=trigger,
                    phase=TriggerPhase.PRE_SESSION,
                    event_type="completed",
                    success=i < 3,  # 3 successes, 2 failures each
                    execution_time_ms=100.0,
                )
                monitor.record_event(event)

        summary = monitor.get_performance_summary()

        assert summary["total_triggers"] == 2
        assert summary["total_executions"] == 10
        assert summary["overall_success_rate"] == 0.6  # 6/10
        assert summary["average_execution_time_ms"] == 100.0


@pytest.mark.asyncio
class TestTriggerManagerRetry:
    """Test TriggerManager retry integration."""

    async def test_retry_on_failure(self, mock_memory_manager, tmp_path):
        """Test that failed triggers are retried."""
        # Create trigger that fails twice then succeeds
        trigger = FailingTrigger("retry_test", failures_before_success=2, error_message="Temporary connection error")

        # Create manager with retry enabled
        retry_policy = TriggerRetryPolicy(
            max_retries=3,
            initial_delay_seconds=0.1,  # Fast for testing
            retryable_errors=["connection"],
        )

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            retry_policy=retry_policy,
        )

        manager.register_trigger(trigger)

        # Execute phase
        results = await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
            enable_retry=True,
        )

        # Should have succeeded after retries
        assert len(results) == 1
        assert results[0].success
        assert trigger.attempt_count == 3  # Initial + 2 retries

    async def test_retry_exhaustion(self, mock_memory_manager, tmp_path):
        """Test that retry stops after max attempts."""
        # Create trigger that always fails
        trigger = FailingTrigger("always_fails", failures_before_success=999, error_message="Temporary error")

        retry_policy = TriggerRetryPolicy(
            max_retries=2,
            initial_delay_seconds=0.1,
            retryable_errors=["temporary"],
        )

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            retry_policy=retry_policy,
        )

        manager.register_trigger(trigger)

        results = await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
            enable_retry=True,
        )

        # Should have failed after max retries
        assert len(results) == 1
        assert not results[0].success
        assert trigger.attempt_count == 3  # Initial + 2 retries

    async def test_no_retry_for_non_retryable_error(self, mock_memory_manager, tmp_path):
        """Test that non-retryable errors are not retried."""
        trigger = FailingTrigger("non_retryable", failures_before_success=999, error_message="Fatal configuration error")

        retry_policy = TriggerRetryPolicy(
            max_retries=3,
            initial_delay_seconds=0.1,
            retryable_errors=["temporary", "connection"],  # "Fatal" not in list
        )

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            retry_policy=retry_policy,
        )

        manager.register_trigger(trigger)

        results = await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
            enable_retry=True,
        )

        # Should have failed without retries
        assert len(results) == 1
        assert not results[0].success
        assert trigger.attempt_count == 1  # No retries


@pytest.mark.asyncio
class TestTriggerManagerEventLogging:
    """Test TriggerManager event logging integration."""

    async def test_events_logged(self, mock_memory_manager, tmp_path):
        """Test that trigger executions are logged as events."""
        trigger = FailingTrigger("logging_test", failures_before_success=1, error_message="Temporary error")

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            enable_event_logging=True,
        )

        manager.register_trigger(trigger)

        await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
        )

        # Get logged events
        events = manager.get_recent_events(trigger_name="logging_test")

        # Should have start, failed, retrying, and completed events
        assert len(events) > 0

        event_types = {e.event_type for e in events}
        assert "started" in event_types
        assert "completed" in event_types or "failed" in event_types

    async def test_health_metrics_updated(self, mock_memory_manager, tmp_path):
        """Test that health metrics are updated from events."""
        trigger = FailingTrigger("health_test", failures_before_success=0)

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            enable_health_monitoring=True,
        )

        manager.register_trigger(trigger)

        # Execute multiple times
        for _ in range(5):
            await manager.execute_phase(
                TriggerPhase.PRE_SESSION,
                project_root=tmp_path,
                enable_retry=False,
            )

        # Check metrics
        metrics = manager.get_health_metrics("health_test")

        assert "health_test" in metrics
        assert metrics["health_test"].total_executions == 5
        assert metrics["health_test"].successful_executions == 5

    async def test_get_performance_summary(self, mock_memory_manager, tmp_path):
        """Test getting performance summary."""
        trigger1 = FailingTrigger("trigger_1", failures_before_success=0)
        trigger2 = FailingTrigger("trigger_2", failures_before_success=1, error_message="Temporary error")

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            enable_health_monitoring=True,
        )

        manager.register_trigger(trigger1)
        manager.register_trigger(trigger2)

        await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
        )

        summary = manager.get_performance_summary()

        assert summary["total_triggers"] == 2
        assert summary["total_executions"] >= 2


@pytest.mark.asyncio
class TestTriggerManagerAnalytics:
    """Test TriggerManager analytics methods."""

    async def test_clear_old_events(self, mock_memory_manager):
        """Test clearing old events."""
        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            enable_event_logging=True,
        )

        event_logger = manager.get_event_logger()
        assert event_logger is not None

        # Log some events
        for i in range(10):
            event = TriggerEvent(
                timestamp=time.time() - (i * 86400),  # i days ago
                trigger_name=f"trigger_{i}",
                phase=TriggerPhase.PRE_SESSION,
                event_type="completed",
                success=True,
            )
            event_logger.log_event(event)

        # Clear events older than 5 days
        cleared = manager.clear_old_events(days=5)

        # Should have cleared events 5-9 (5 events)
        assert cleared == 5

    async def test_get_health_alerts(self, mock_memory_manager, tmp_path):
        """Test getting health alerts."""
        trigger = FailingTrigger("alert_test", failures_before_success=999, error_message="Persistent error")

        manager = TriggerManager(
            memory_manager=mock_memory_manager,
            enable_health_monitoring=True,
            retry_policy=TriggerRetryPolicy(max_retries=0),  # Don't retry
        )

        manager.register_trigger(trigger)

        # Execute multiple times to trigger alert
        for _ in range(5):
            await manager.execute_phase(
                TriggerPhase.PRE_SESSION,
                project_root=tmp_path,
                enable_retry=False,
            )

        # Get alerts
        alerts = manager.get_health_alerts()

        assert len(alerts) > 0
        assert any("alert_test" in alert for alert in alerts)
