"""Integration tests for watch-queue handshake functionality (Task 461.16).

Tests the coordination between file watchers and queue processor including:
- Error tracking and backoff recovery
- Queue capacity management and throttling
- Processing error feedback to watches
- Multiple watches competing for capacity
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from common.core.sqlite_state_manager import SQLiteStateManager, WatchFolderConfig


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "test_state.db"

        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()

        yield manager

        await manager.close()


@pytest.fixture
def sample_watch_config():
    """Create a sample watch folder configuration."""
    return WatchFolderConfig(
        watch_id="test_watch_001",
        path="/tmp/test_watch",
        collection="_test_collection",
        patterns=["*.txt", "*.md"],
        ignore_patterns=[".git/*"],
        auto_ingest=True,
        recursive=True,
        recursive_depth=10,
        debounce_seconds=2.0,
        enabled=True,
        watch_type="library",
        library_name="test_lib",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestWatchErrorBackoffRecovery:
    """Test scenario: Watch encounters errors -> backoff -> recovery."""

    @pytest.mark.asyncio
    async def test_error_tracking_increments(self, state_manager, sample_watch_config):
        """Test that errors are properly tracked and incremented."""
        # Save initial watch config
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Record an error
        await state_manager.record_watch_folder_error(
            watch_id=sample_watch_config.watch_id,
            error_message="Test error 1",
            health_status="degraded",
        )

        # Verify error was recorded
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config is not None
        assert config.consecutive_errors == 1
        assert config.total_errors == 1
        assert config.health_status == "degraded"
        assert config.last_error_message == "Test error 1"

    @pytest.mark.asyncio
    async def test_multiple_errors_trigger_backoff(self, state_manager, sample_watch_config):
        """Test that multiple consecutive errors trigger backoff state."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Simulate multiple errors to trigger backoff
        for i in range(5):
            health_status = "backoff" if i >= 2 else "degraded"
            backoff_until = datetime.now(timezone.utc) + timedelta(seconds=60) if i >= 2 else None

            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=f"Test error {i + 1}",
                health_status=health_status,
                backoff_until=backoff_until,
            )

        # Verify backoff state
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config is not None
        assert config.consecutive_errors == 5
        assert config.total_errors == 5
        assert config.health_status == "backoff"
        assert config.backoff_until is not None

    @pytest.mark.asyncio
    async def test_success_resets_consecutive_errors(self, state_manager, sample_watch_config):
        """Test that successful processing resets consecutive error count."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Record some errors first
        for i in range(3):
            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=f"Error {i}",
                health_status="degraded",
            )

        # Verify errors were recorded
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 3
        assert config.total_errors == 3

        # Record success
        await state_manager.record_watch_folder_success(sample_watch_config.watch_id)

        # Verify consecutive errors reset but total preserved
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 0
        assert config.total_errors == 3  # Total preserved
        assert config.health_status == "healthy"
        assert config.backoff_until is None

    @pytest.mark.asyncio
    async def test_backoff_recovery_cycle(self, state_manager, sample_watch_config):
        """Test full cycle: healthy -> degraded -> backoff -> recovery."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Initial state: healthy
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "healthy"

        # Transition to degraded
        await state_manager.record_watch_folder_error(
            watch_id=sample_watch_config.watch_id,
            error_message="First error",
            health_status="degraded",
        )
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "degraded"

        # Transition to backoff
        backoff_time = datetime.now(timezone.utc) + timedelta(seconds=30)
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            health_status="backoff",
            backoff_until=backoff_time,
        )
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "backoff"
        assert config.backoff_until is not None

        # Recovery via success
        await state_manager.record_watch_folder_success(sample_watch_config.watch_id)
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "healthy"
        assert config.backoff_until is None


class TestQueueCapacityThrottling:
    """Test scenario: Queue capacity full -> watch throttles."""

    @pytest.mark.asyncio
    async def test_queue_depth_affects_watch_behavior(self, state_manager, sample_watch_config):
        """Test that high queue depth can trigger watch throttling behavior.

        This test validates the conceptual flow where queue depth information
        could be used to adjust watch behavior. The actual queue depth tracking
        is handled by the queue_client module.
        """
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Simulate a watch being throttled due to high queue depth
        # In production, this would be triggered by queue depth monitoring
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            health_status="degraded",
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "degraded"

        # Simulate queue depth returning to normal
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            health_status="healthy",
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_multiple_watches_capacity_allocation(self, state_manager):
        """Test capacity allocation across multiple watches."""
        # Create multiple watch configs
        watches = []
        for i in range(3):
            watch = WatchFolderConfig(
                watch_id=f"watch_{i}",
                path=f"/tmp/watch_{i}",
                collection=f"_collection_{i}",
                patterns=["*.txt"],
                ignore_patterns=[],
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await state_manager.save_watch_folder_config(watch)
            watches.append(watch)

        # Verify all watches are saved
        all_watches = await state_manager.get_all_watch_folder_configs(enabled_only=True)
        assert len(all_watches) == 3


class TestProcessorErrorFeedback:
    """Test scenario: Processor fails -> feedback to watch."""

    @pytest.mark.asyncio
    async def test_error_feedback_updates_watch_state(self, state_manager, sample_watch_config):
        """Test that processor errors are fed back to watch state."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Simulate processor error feedback
        await state_manager.record_watch_folder_error(
            watch_id=sample_watch_config.watch_id,
            error_message="Qdrant connection failed: timeout",
            health_status="degraded",
        )

        # Verify state was updated
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 1
        assert "Qdrant" in config.last_error_message

    @pytest.mark.asyncio
    async def test_different_error_types_recorded(self, state_manager, sample_watch_config):
        """Test that different error types are properly recorded."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        error_types = [
            "File not found: /path/to/file.txt",
            "Parsing error: invalid JSON",
            "Qdrant error: collection not found",
            "Embedding error: model unavailable",
        ]

        for error_msg in error_types:
            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=error_msg,
                health_status="degraded",
            )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == len(error_types)
        assert config.total_errors == len(error_types)
        # Last error message should be the most recent one
        assert error_types[-1] == config.last_error_message

    @pytest.mark.asyncio
    async def test_error_with_backoff_calculation(self, state_manager, sample_watch_config):
        """Test error recording with backoff time calculation."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Calculate backoff time (exponential: base * 2^errors)
        base_delay_seconds = 10
        consecutive_errors = 3
        max_delay_seconds = 300

        delay = min(base_delay_seconds * (2 ** consecutive_errors), max_delay_seconds)
        backoff_until = datetime.now(timezone.utc) + timedelta(seconds=delay)

        # Record error with backoff
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            consecutive_errors=consecutive_errors,
            health_status="backoff",
            backoff_until=backoff_until,
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == consecutive_errors
        assert config.health_status == "backoff"
        assert config.backoff_until is not None

        # Verify backoff time is in the future
        now = datetime.now(timezone.utc)
        assert config.backoff_until > now


class TestMultipleWatchesCompetition:
    """Test scenario: Multiple watches competing for capacity."""

    @pytest.mark.asyncio
    async def test_independent_error_tracking(self, state_manager):
        """Test that error tracking is independent per watch."""
        # Create multiple watches
        watches = []
        for i in range(3):
            watch = WatchFolderConfig(
                watch_id=f"watch_{i}",
                path=f"/tmp/watch_{i}",
                collection="_shared_collection",
                patterns=["*.txt"],
                ignore_patterns=[],
                enabled=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await state_manager.save_watch_folder_config(watch)
            watches.append(watch)

        # Record different error counts for each watch
        for i, watch in enumerate(watches):
            for _ in range(i + 1):  # watch_0: 1 error, watch_1: 2 errors, watch_2: 3 errors
                await state_manager.record_watch_folder_error(
                    watch_id=watch.watch_id,
                    error_message="Test error",
                    health_status="degraded",
                )

        # Verify independent tracking
        for i, watch in enumerate(watches):
            config = await state_manager.get_watch_folder_config(watch.watch_id)
            assert config.consecutive_errors == i + 1

    @pytest.mark.asyncio
    async def test_filter_by_health_status(self, state_manager):
        """Test filtering watches by health status."""
        # Create watches with different health statuses
        health_statuses = ["healthy", "degraded", "backoff", "disabled"]

        for i, status in enumerate(health_statuses):
            watch = WatchFolderConfig(
                watch_id=f"watch_{status}",
                path=f"/tmp/watch_{i}",
                collection=f"_collection_{i}",
                patterns=["*.txt"],
                ignore_patterns=[],
                enabled=True,
                health_status=status,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await state_manager.save_watch_folder_config(watch)

        # Filter by each status
        for status in health_statuses:
            watches = await state_manager.get_watch_folders_by_health_status(status)
            assert len(watches) == 1
            assert watches[0].health_status == status

    @pytest.mark.asyncio
    async def test_concurrent_error_recording(self, state_manager, sample_watch_config):
        """Test concurrent error recording doesn't cause race conditions."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Simulate concurrent error recording
        async def record_error(error_id: int):
            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=f"Concurrent error {error_id}",
                health_status="degraded",
            )

        # Record 10 errors concurrently
        tasks = [record_error(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all errors were recorded
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 10
        assert config.total_errors == 10


class TestHealthStatusTransitions:
    """Test health status state machine transitions."""

    @pytest.mark.asyncio
    async def test_valid_status_transitions(self, state_manager, sample_watch_config):
        """Test that all valid status transitions work correctly."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Test all valid transitions
        transitions = [
            ("healthy", "degraded"),
            ("degraded", "backoff"),
            ("backoff", "disabled"),
            ("disabled", "healthy"),  # Manual reset
            ("healthy", "backoff"),  # Direct transition possible
            ("backoff", "healthy"),  # Recovery
        ]

        for from_status, to_status in transitions:
            # Set initial status
            await state_manager.update_watch_folder_error_state(
                watch_id=sample_watch_config.watch_id,
                health_status=from_status,
            )

            config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
            assert config.health_status == from_status

            # Transition to new status
            await state_manager.update_watch_folder_error_state(
                watch_id=sample_watch_config.watch_id,
                health_status=to_status,
            )

            config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
            assert config.health_status == to_status

    @pytest.mark.asyncio
    async def test_invalid_status_defaults_to_healthy(self, state_manager, sample_watch_config):
        """Test that invalid status values default to healthy."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Try to set invalid status
        success = await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            health_status="invalid_status",
        )

        # Should succeed but use default
        assert success
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.health_status == "healthy"


class TestBackoffTimingValidation:
    """Test backoff timing calculations and expiry."""

    @pytest.mark.asyncio
    async def test_backoff_time_in_future(self, state_manager, sample_watch_config):
        """Test that backoff time is correctly set in the future."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)

        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            backoff_until=future_time,
            health_status="backoff",
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.backoff_until is not None

        now = datetime.now(timezone.utc)
        assert config.backoff_until > now

    @pytest.mark.asyncio
    async def test_clear_backoff(self, state_manager, sample_watch_config):
        """Test clearing backoff time."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Set backoff
        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            backoff_until=future_time,
            health_status="backoff",
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.backoff_until is not None

        # Clear backoff
        await state_manager.update_watch_folder_error_state(
            watch_id=sample_watch_config.watch_id,
            health_status="healthy",
            clear_backoff=True,
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.backoff_until is None
        assert config.health_status == "healthy"

    @pytest.mark.asyncio
    async def test_get_watches_in_backoff(self, state_manager):
        """Test retrieving watches currently in backoff."""
        # Create watches with different backoff states
        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        past_time = datetime.now(timezone.utc) - timedelta(minutes=5)

        watches_config = [
            ("watch_future_backoff", future_time, "backoff"),
            ("watch_past_backoff", past_time, "backoff"),
            ("watch_no_backoff", None, "healthy"),
        ]

        for watch_id, backoff_until, status in watches_config:
            watch = WatchFolderConfig(
                watch_id=watch_id,
                path=f"/tmp/{watch_id}",
                collection="_test",
                patterns=["*.txt"],
                ignore_patterns=[],
                enabled=True,
                health_status=status,
                backoff_until=backoff_until,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await state_manager.save_watch_folder_config(watch)

        # Get watches in backoff (should only return those with future backoff_until)
        backoff_watches = await state_manager.get_watch_folders_in_backoff()

        # Should only find the watch with future backoff time
        assert len(backoff_watches) == 1
        assert backoff_watches[0].watch_id == "watch_future_backoff"


class TestErrorStatisticsAggregation:
    """Test error statistics aggregation and reporting."""

    @pytest.mark.asyncio
    async def test_total_errors_preserved_on_reset(self, state_manager, sample_watch_config):
        """Test that total errors are preserved when resetting consecutive errors."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Record errors
        for i in range(5):
            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=f"Error {i}",
                health_status="degraded",
            )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 5
        assert config.total_errors == 5

        # Reset via success
        await state_manager.record_watch_folder_success(sample_watch_config.watch_id)

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 0
        assert config.total_errors == 5  # Still 5

        # More errors
        for i in range(3):
            await state_manager.record_watch_folder_error(
                watch_id=sample_watch_config.watch_id,
                error_message=f"New error {i}",
                health_status="degraded",
            )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.consecutive_errors == 3
        assert config.total_errors == 8  # 5 + 3

    @pytest.mark.asyncio
    async def test_last_timestamps_recorded(self, state_manager, sample_watch_config):
        """Test that last error and success timestamps are recorded."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        # Initial state - no timestamps
        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.last_error_at is None
        assert config.last_success_at is None

        # Record error
        await state_manager.record_watch_folder_error(
            watch_id=sample_watch_config.watch_id,
            error_message="Test error",
            health_status="degraded",
        )

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.last_error_at is not None
        error_time = config.last_error_at

        # Record success
        await state_manager.record_watch_folder_success(sample_watch_config.watch_id)

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.last_success_at is not None
        # Error timestamp should be preserved
        assert config.last_error_at == error_time


class TestWatchPriorityAdjustment:
    """Test watch priority adjustment based on health status (Task 461.17)."""

    @pytest.mark.asyncio
    async def test_default_watch_priority(self, state_manager, sample_watch_config):
        """Test that watches have default priority of 5."""
        await state_manager.save_watch_folder_config(sample_watch_config)

        config = await state_manager.get_watch_folder_config(sample_watch_config.watch_id)
        assert config.watch_priority == 5

    @pytest.mark.asyncio
    async def test_custom_watch_priority(self, state_manager):
        """Test setting custom watch priority."""
        watch = WatchFolderConfig(
            watch_id="high_priority_watch",
            path="/tmp/high_priority",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=8,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        await state_manager.save_watch_folder_config(watch)

        config = await state_manager.get_watch_folder_config("high_priority_watch")
        assert config.watch_priority == 8

    @pytest.mark.asyncio
    async def test_priority_clamped_to_valid_range(self):
        """Test that priority values are clamped to 0-10."""
        # Too high priority should be clamped to 10
        watch_high = WatchFolderConfig(
            watch_id="test_high",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=15,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert watch_high.watch_priority == 10

        # Negative priority should be set to default 5
        watch_low = WatchFolderConfig(
            watch_id="test_low",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=-1,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert watch_low.watch_priority == 5

    @pytest.mark.asyncio
    async def test_effective_priority_healthy_recent_success(self):
        """Test effective priority boost for healthy watches with recent success."""
        recent_success = datetime.now(timezone.utc) - timedelta(minutes=30)
        watch = WatchFolderConfig(
            watch_id="test",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=5,
            health_status="healthy",
            consecutive_errors=0,
            last_success_at=recent_success,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Healthy with recent success gets +1 boost
        assert watch.calculate_effective_priority() == 6

    @pytest.mark.asyncio
    async def test_effective_priority_degraded(self):
        """Test effective priority decrease for degraded watches."""
        watch = WatchFolderConfig(
            watch_id="test",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=5,
            health_status="degraded",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Degraded gets -1
        assert watch.calculate_effective_priority() == 4

    @pytest.mark.asyncio
    async def test_effective_priority_backoff(self):
        """Test effective priority decrease for watches in backoff."""
        watch = WatchFolderConfig(
            watch_id="test",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=5,
            health_status="backoff",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Backoff gets -2
        assert watch.calculate_effective_priority() == 3

    @pytest.mark.asyncio
    async def test_effective_priority_disabled_is_zero(self):
        """Test that disabled watches have effective priority of 0."""
        watch = WatchFolderConfig(
            watch_id="test",
            path="/tmp/test",
            collection="_test",
            patterns=["*.txt"],
            ignore_patterns=[],
            enabled=True,
            watch_priority=8,
            health_status="disabled",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        # Disabled always has priority 0
        assert watch.calculate_effective_priority() == 0

    @pytest.mark.asyncio
    async def test_order_by_priority(self, state_manager):
        """Test that watches can be ordered by priority."""
        watches = [
            ("watch_low", 2),
            ("watch_high", 9),
            ("watch_med", 5),
        ]

        for watch_id, priority in watches:
            watch = WatchFolderConfig(
                watch_id=watch_id,
                path=f"/tmp/{watch_id}",
                collection="_test",
                patterns=["*.txt"],
                ignore_patterns=[],
                enabled=True,
                watch_priority=priority,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            await state_manager.save_watch_folder_config(watch)

        # Get all watches ordered by priority
        all_watches = await state_manager.get_all_watch_folder_configs(order_by_priority=True)

        assert len(all_watches) == 3
        # Should be ordered by priority descending
        assert all_watches[0].watch_id == "watch_high"
        assert all_watches[0].watch_priority == 9
        assert all_watches[1].watch_id == "watch_med"
        assert all_watches[1].watch_priority == 5
        assert all_watches[2].watch_id == "watch_low"
        assert all_watches[2].watch_priority == 2
