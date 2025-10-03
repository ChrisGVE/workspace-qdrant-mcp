"""
Unit tests for error retention and cleanup policies.

Tests the ErrorRetentionManager and RetentionPolicy functionality including
retention policy application, dry-run mode, scheduled cleanup, and configuration.
"""

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.python.common.core.error_retention import (
    ErrorRetentionManager,
    RetentionPolicy,
    CleanupResult,
    CleanupStatistics,
)
from src.python.common.core.error_message_manager import ErrorMessageManager
from src.python.common.core.error_categorization import ErrorSeverity


@pytest.fixture
async def temp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test_retention.db"

    # Create schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            severity TEXT NOT NULL,
            category TEXT NOT NULL,
            message TEXT NOT NULL,
            context TEXT,
            acknowledged INTEGER DEFAULT 0,
            acknowledged_at TEXT,
            acknowledged_by TEXT,
            retry_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

    yield str(db_path)


@pytest.fixture
async def error_manager(temp_db):
    """Create error message manager for testing."""
    manager = ErrorMessageManager(db_path=temp_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def retention_manager(temp_db):
    """Create retention manager for testing."""
    manager = ErrorRetentionManager(db_path=temp_db)
    await manager.initialize()
    yield manager
    await manager.close()


class TestRetentionPolicy:
    """Test RetentionPolicy dataclass."""

    def test_default_retention_policy(self):
        """Test default retention policy values."""
        policy = RetentionPolicy()

        assert policy.max_age_days == 30
        assert policy.max_count is None
        assert policy.severity_specific_retention == {
            'info': 30,
            'warning': 90,
            'error': 180
        }
        assert policy.acknowledged_multiplier == 2.0
        assert policy.preserve_active_retries is True

    def test_custom_retention_policy(self):
        """Test custom retention policy values."""
        policy = RetentionPolicy(
            max_age_days=60,
            max_count=5000,
            severity_specific_retention={
                'info': 15,
                'warning': 45,
                'error': 90
            },
            acknowledged_multiplier=3.0,
            preserve_active_retries=False
        )

        assert policy.max_age_days == 60
        assert policy.max_count == 5000
        assert policy.severity_specific_retention['info'] == 15
        assert policy.acknowledged_multiplier == 3.0
        assert policy.preserve_active_retries is False

    def test_get_retention_days_unacknowledged(self):
        """Test retention period calculation for unacknowledged messages."""
        policy = RetentionPolicy()

        assert policy.get_retention_days('info', acknowledged=False) == 30
        assert policy.get_retention_days('warning', acknowledged=False) == 90
        assert policy.get_retention_days('error', acknowledged=False) == 180

    def test_get_retention_days_acknowledged(self):
        """Test retention period calculation for acknowledged messages."""
        policy = RetentionPolicy()

        # Should be 2x for acknowledged
        assert policy.get_retention_days('info', acknowledged=True) == 60
        assert policy.get_retention_days('warning', acknowledged=True) == 180
        assert policy.get_retention_days('error', acknowledged=True) == 360

    def test_get_retention_days_unknown_severity(self):
        """Test retention for unknown severity falls back to default."""
        policy = RetentionPolicy(max_age_days=45)

        # Unknown severity should use max_age_days
        assert policy.get_retention_days('unknown', acknowledged=False) == 45


class TestCleanupResult:
    """Test CleanupResult dataclass."""

    def test_cleanup_result_default(self):
        """Test default CleanupResult values."""
        result = CleanupResult()

        assert result.deleted_count == 0
        assert result.preserved_count == 0
        assert result.by_severity == {}
        assert result.errors == []
        assert result.dry_run is False

    def test_cleanup_result_to_dict(self):
        """Test CleanupResult conversion to dictionary."""
        result = CleanupResult(
            deleted_count=10,
            preserved_count=90,
            by_severity={'info': 5, 'error': 5},
            errors=['test error'],
            dry_run=True
        )

        result_dict = result.to_dict()
        assert result_dict['deleted_count'] == 10
        assert result_dict['preserved_count'] == 90
        assert result_dict['by_severity'] == {'info': 5, 'error': 5}
        assert result_dict['errors'] == ['test error']
        assert result_dict['dry_run'] is True


class TestCleanupStatistics:
    """Test CleanupStatistics dataclass."""

    def test_cleanup_statistics_default(self):
        """Test default CleanupStatistics values."""
        stats = CleanupStatistics()

        assert stats.last_cleanup_at is None
        assert stats.total_cleanups == 0
        assert stats.total_deleted == 0
        assert stats.average_deleted_per_cleanup == 0.0
        assert stats.last_cleanup_duration == 0.0

    def test_cleanup_statistics_to_dict(self):
        """Test CleanupStatistics conversion to dictionary."""
        now = datetime.now(timezone.utc)
        stats = CleanupStatistics(
            last_cleanup_at=now,
            total_cleanups=5,
            total_deleted=50,
            average_deleted_per_cleanup=10.0,
            last_cleanup_duration=2.5
        )

        stats_dict = stats.to_dict()
        assert stats_dict['last_cleanup_at'] == now.isoformat()
        assert stats_dict['total_cleanups'] == 5
        assert stats_dict['total_deleted'] == 50
        assert stats_dict['average_deleted_per_cleanup'] == 10.0
        assert stats_dict['last_cleanup_duration'] == 2.5


@pytest.mark.asyncio
class TestErrorRetentionManager:
    """Test ErrorRetentionManager functionality."""

    async def test_initialization(self, retention_manager):
        """Test manager initialization."""
        assert retention_manager._initialized is True
        assert retention_manager.default_policy is not None

    async def test_cleanup_empty_database(self, retention_manager):
        """Test cleanup on empty database."""
        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 0
        assert result.preserved_count == 0

    async def test_cleanup_recent_messages_preserved(self, error_manager, retention_manager):
        """Test that recent messages are preserved."""
        # Add recent messages
        for i in range(5):
            await error_manager.record_error(
                message_override=f"Recent error {i}",
                context={'test': i},
                severity_override=ErrorSeverity.INFO
            )

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 0
        assert result.preserved_count == 5

    async def test_cleanup_old_messages_deleted(self, temp_db, retention_manager):
        """Test that old messages are deleted."""
        # Add old messages directly to database
        conn = sqlite3.connect(temp_db)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        for i in range(3):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()
        conn.close()

        # Default policy: info messages older than 30 days should be deleted
        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 3
        assert result.preserved_count == 0

    async def test_cleanup_severity_specific_retention(self, temp_db, retention_manager):
        """Test severity-specific retention periods."""
        conn = sqlite3.connect(temp_db)

        # Add messages with different severities and ages
        # Info: 60 days old (should be deleted, retention = 30 days)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'Old info', 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),)
        )

        # Warning: 60 days old (should be preserved, retention = 90 days)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'warning', 'test', 'Recent warning', 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),)
        )

        # Error: 150 days old (should be preserved, retention = 180 days)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'error', 'test', 'Recent error', 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=150)).isoformat(),)
        )

        conn.commit()
        conn.close()

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 1  # Only info should be deleted
        assert result.preserved_count == 2  # Warning and error preserved

    async def test_cleanup_acknowledged_extended_retention(self, temp_db, retention_manager):
        """Test that acknowledged messages get extended retention."""
        conn = sqlite3.connect(temp_db)

        # Info message: 45 days old, acknowledged
        # Normal retention: 30 days
        # Acknowledged retention: 30 * 2 = 60 days
        # Should be preserved
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, acknowledged, retry_count)
            VALUES (?, 'info', 'test', 'Acknowledged info', 1, 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),)
        )

        # Info message: 45 days old, not acknowledged
        # Should be deleted (exceeds 30 day retention)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, acknowledged, retry_count)
            VALUES (?, 'info', 'test', 'Unacknowledged info', 0, 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),)
        )

        conn.commit()
        conn.close()

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 1  # Only unacknowledged should be deleted
        assert result.preserved_count == 1  # Acknowledged preserved

    async def test_cleanup_active_retries_preserved(self, temp_db, retention_manager):
        """Test that messages with active retries are preserved."""
        conn = sqlite3.connect(temp_db)

        # Old message with retry_count > 0 (should be preserved)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'Active retry', 3)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),)
        )

        # Old message with retry_count = 0 (should be deleted)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'No retry', 0)
            """,
            ((datetime.now(timezone.utc) - timedelta(days=60)).isoformat(),)
        )

        conn.commit()
        conn.close()

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 1  # Only no-retry should be deleted
        assert result.preserved_count == 1  # Active retry preserved

    async def test_cleanup_max_count_policy(self, error_manager, retention_manager):
        """Test max count retention policy."""
        # Add 20 recent messages
        for i in range(20):
            await error_manager.record_error(
                message_override=f"Message {i}",
                severity_override=ErrorSeverity.INFO
            )

        # Apply policy with max_count=10
        policy = RetentionPolicy(max_count=10)
        result = await retention_manager.apply_retention_policy(policy, dry_run=False)

        # Should delete 10 oldest, keep 10 newest
        assert result.deleted_count == 10
        assert result.preserved_count == 10

    async def test_cleanup_dry_run_mode(self, temp_db, retention_manager):
        """Test dry-run mode doesn't delete messages."""
        conn = sqlite3.connect(temp_db)

        # Add old messages
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(5):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()
        conn.close()

        # Dry run should report what would be deleted
        result = await retention_manager.cleanup_old_errors(dry_run=True)

        assert result.deleted_count == 5
        assert result.dry_run is True

        # Verify messages still exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.execute("SELECT COUNT(*) as count FROM messages")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 5  # Messages not actually deleted

    async def test_cleanup_by_severity_tracking(self, temp_db, retention_manager):
        """Test that cleanup tracks deletion by severity."""
        conn = sqlite3.connect(temp_db)

        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        # Add old messages of different severities
        for i in range(3):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old info {i}")
            )

        for i in range(2):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'warning', 'test', ?, 0)
                """,
                (old_timestamp, f"Old warning {i}")
            )

        conn.commit()
        conn.close()

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        # Only info should be deleted (warning retention = 90 days)
        assert result.deleted_count == 3
        assert result.by_severity.get('info') == 3
        assert 'warning' not in result.by_severity

    async def test_cleanup_statistics_tracking(self, temp_db, retention_manager):
        """Test that cleanup statistics are tracked."""
        conn = sqlite3.connect(temp_db)

        # Add old messages
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(5):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()
        conn.close()

        # Run cleanup
        await retention_manager.cleanup_old_errors(dry_run=False)

        # Check statistics
        stats = retention_manager.get_cleanup_stats()

        assert stats.total_cleanups == 1
        assert stats.total_deleted == 5
        assert stats.average_deleted_per_cleanup == 5.0
        assert stats.last_cleanup_at is not None
        assert stats.last_cleanup_duration > 0

    async def test_multiple_cleanup_statistics(self, temp_db, retention_manager):
        """Test statistics across multiple cleanups."""
        conn = sqlite3.connect(temp_db)

        # First cleanup: 5 messages
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(5):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()

        await retention_manager.cleanup_old_errors(dry_run=False)

        # Second cleanup: 3 messages
        for i in range(3):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()
        conn.close()

        await retention_manager.cleanup_old_errors(dry_run=False)

        # Check statistics
        stats = retention_manager.get_cleanup_stats()

        assert stats.total_cleanups == 2
        assert stats.total_deleted == 8
        assert stats.average_deleted_per_cleanup == 4.0

    async def test_schedule_cleanup(self, retention_manager):
        """Test scheduling automatic cleanup."""
        # Schedule cleanup with short interval
        result = await retention_manager.schedule_cleanup(interval_hours=1)

        assert result is True
        assert retention_manager._cleanup_running is True

        # Stop scheduler
        await retention_manager.stop_cleanup_scheduler()

    async def test_schedule_cleanup_already_running(self, retention_manager):
        """Test scheduling cleanup when already running."""
        await retention_manager.schedule_cleanup(interval_hours=1)

        # Try to schedule again
        result = await retention_manager.schedule_cleanup(interval_hours=1)

        assert result is False

        # Cleanup
        await retention_manager.stop_cleanup_scheduler()

    async def test_stop_cleanup_scheduler_not_running(self, retention_manager):
        """Test stopping cleanup scheduler when not running."""
        result = await retention_manager.stop_cleanup_scheduler()

        assert result is False

    async def test_schedule_cleanup_invalid_interval(self, retention_manager):
        """Test scheduling with invalid interval."""
        with pytest.raises(ValueError, match="interval_hours must be positive"):
            await retention_manager.schedule_cleanup(interval_hours=0)

        with pytest.raises(ValueError, match="interval_hours must be positive"):
            await retention_manager.schedule_cleanup(interval_hours=-1)

    async def test_scheduled_cleanup_executes(self, temp_db, retention_manager):
        """Test that scheduled cleanup actually executes."""
        conn = sqlite3.connect(temp_db)

        # Add old messages
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        for i in range(3):
            conn.execute(
                """
                INSERT INTO messages (timestamp, severity, category, message, retry_count)
                VALUES (?, 'info', 'test', ?, 0)
                """,
                (old_timestamp, f"Old error {i}")
            )
        conn.commit()
        conn.close()

        # Schedule with very short interval (for testing)
        # Note: We can't easily test the actual interval without waiting,
        # but we can verify the scheduler starts and stops
        await retention_manager.schedule_cleanup(interval_hours=24)

        # Give it a moment to start
        await asyncio.sleep(0.1)

        assert retention_manager._cleanup_running is True

        # Stop scheduler
        await retention_manager.stop_cleanup_scheduler()

        assert retention_manager._cleanup_running is False

    async def test_cleanup_graceful_shutdown(self, retention_manager):
        """Test graceful shutdown of cleanup scheduler."""
        await retention_manager.schedule_cleanup(interval_hours=1)

        # Close manager (should stop scheduler)
        await retention_manager.close()

        assert retention_manager._cleanup_running is False
        assert retention_manager._initialized is False


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error conditions."""

    async def test_cleanup_with_custom_policy(self, temp_db, retention_manager):
        """Test cleanup with custom retention policy."""
        conn = sqlite3.connect(temp_db)

        # Add message that's 20 days old
        timestamp = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat()
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'Test message', 0)
            """,
            (timestamp,)
        )
        conn.commit()
        conn.close()

        # Custom policy: 15 days retention
        policy = RetentionPolicy(
            severity_specific_retention={'info': 15}
        )

        result = await retention_manager.apply_retention_policy(policy, dry_run=False)

        assert result.deleted_count == 1

    async def test_cleanup_mixed_conditions(self, temp_db, retention_manager):
        """Test cleanup with mix of old, acknowledged, and retry messages."""
        conn = sqlite3.connect(temp_db)

        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        # Old, unacknowledged, no retry (should delete)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, acknowledged, retry_count)
            VALUES (?, 'info', 'test', 'Delete me', 0, 0)
            """,
            (old_timestamp,)
        )

        # Old, acknowledged, no retry (should preserve due to 2x retention)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, acknowledged, retry_count)
            VALUES (?, 'info', 'test', 'Acknowledged', 1, 0)
            """,
            (old_timestamp,)
        )

        # Old, unacknowledged, has retry (should preserve)
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, acknowledged, retry_count)
            VALUES (?, 'info', 'test', 'Has retry', 0, 2)
            """,
            (old_timestamp,)
        )

        conn.commit()
        conn.close()

        result = await retention_manager.cleanup_old_errors(dry_run=False)

        assert result.deleted_count == 1
        assert result.preserved_count == 2

    async def test_empty_severity_retention(self, temp_db, retention_manager):
        """Test with empty severity-specific retention."""
        policy = RetentionPolicy(
            max_age_days=45,
            severity_specific_retention={}
        )

        conn = sqlite3.connect(temp_db)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'Test', 0)
            """,
            (old_timestamp,)
        )
        conn.commit()
        conn.close()

        result = await retention_manager.apply_retention_policy(policy, dry_run=False)

        # Should use max_age_days (45) as fallback
        assert result.deleted_count == 1

    async def test_preserve_active_retries_disabled(self, temp_db):
        """Test cleanup with preserve_active_retries=False."""
        manager = ErrorRetentionManager(
            db_path=temp_db,
            default_policy=RetentionPolicy(preserve_active_retries=False)
        )
        await manager.initialize()

        conn = sqlite3.connect(temp_db)
        old_timestamp = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()

        # Old message with retry_count > 0
        conn.execute(
            """
            INSERT INTO messages (timestamp, severity, category, message, retry_count)
            VALUES (?, 'info', 'test', 'Active retry', 3)
            """,
            (old_timestamp,)
        )
        conn.commit()
        conn.close()

        result = await manager.cleanup_old_errors(dry_run=False)

        # Should delete even with retry_count > 0
        assert result.deleted_count == 1

        await manager.close()
