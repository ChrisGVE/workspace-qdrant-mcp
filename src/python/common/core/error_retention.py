"""
Error Message Retention and Cleanup Policies

Provides configurable retention policies and automated cleanup for error messages
with support for severity-specific retention, acknowledged message preservation,
and background scheduled cleanup operations.

Features:
    - Configurable retention periods by severity
    - Acknowledged message preservation (2x retention)
    - Active retry protection (never delete retry_count > 0)
    - Max count retention policies
    - Dry-run mode for preview
    - Background cleanup scheduler
    - Comprehensive statistics

Example:
    ```python
    from workspace_qdrant_mcp.core.error_retention import (
        ErrorRetentionManager, RetentionPolicy
    )

    # Initialize manager
    manager = ErrorRetentionManager()
    await manager.initialize()

    # Apply retention policy
    policy = RetentionPolicy(
        max_age_days=30,
        max_count=10000,
        severity_specific_retention={
            'info': 30,
            'warning': 90,
            'error': 180
        }
    )
    result = await manager.apply_retention_policy(policy, dry_run=True)
    print(f"Would delete {result.deleted_count} messages")

    # Schedule automatic cleanup
    await manager.schedule_cleanup(interval_hours=24)
    ```
"""

import asyncio
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from loguru import logger

from .queue_connection import QueueConnectionPool, ConnectionConfig


@dataclass
class RetentionPolicy:
    """
    Configurable retention policy for error messages.

    Attributes:
        max_age_days: Default maximum age in days for all messages
        max_count: Maximum number of messages to keep (keeps most recent)
        severity_specific_retention: Per-severity retention periods in days
        acknowledged_multiplier: Multiplier for acknowledged message retention (default: 2x)
        preserve_active_retries: Never delete messages with retry_count > 0 (default: True)
    """
    max_age_days: int = 30
    max_count: Optional[int] = None
    severity_specific_retention: Dict[str, int] = field(default_factory=lambda: {
        'info': 30,
        'warning': 90,
        'error': 180
    })
    acknowledged_multiplier: float = 2.0
    preserve_active_retries: bool = True

    def get_retention_days(self, severity: str, acknowledged: bool = False) -> int:
        """
        Get retention period for a specific severity and acknowledgment status.

        Args:
            severity: Error severity ('info', 'warning', 'error')
            acknowledged: Whether message is acknowledged

        Returns:
            Retention period in days
        """
        base_retention = self.severity_specific_retention.get(
            severity.lower(),
            self.max_age_days
        )

        if acknowledged:
            return int(base_retention * self.acknowledged_multiplier)

        return base_retention


@dataclass
class CleanupResult:
    """
    Results from a cleanup operation.

    Attributes:
        deleted_count: Number of messages deleted
        preserved_count: Number of messages preserved (active retries, recent, etc.)
        by_severity: Count of deleted messages by severity
        errors: List of errors encountered during cleanup
        dry_run: Whether this was a dry-run operation
    """
    deleted_count: int = 0
    preserved_count: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    errors: list = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'deleted_count': self.deleted_count,
            'preserved_count': self.preserved_count,
            'by_severity': self.by_severity,
            'errors': self.errors,
            'dry_run': self.dry_run
        }


@dataclass
class CleanupStatistics:
    """
    Statistics from cleanup operations.

    Attributes:
        last_cleanup_at: Timestamp of last cleanup
        total_cleanups: Total number of cleanup operations
        total_deleted: Total messages deleted across all cleanups
        average_deleted_per_cleanup: Average messages deleted per cleanup
        last_cleanup_duration: Duration of last cleanup in seconds
    """
    last_cleanup_at: Optional[datetime] = None
    total_cleanups: int = 0
    total_deleted: int = 0
    average_deleted_per_cleanup: float = 0.0
    last_cleanup_duration: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'last_cleanup_at': self.last_cleanup_at.isoformat() if self.last_cleanup_at else None,
            'total_cleanups': self.total_cleanups,
            'total_deleted': self.total_deleted,
            'average_deleted_per_cleanup': self.average_deleted_per_cleanup,
            'last_cleanup_duration': self.last_cleanup_duration
        }


class ErrorRetentionManager:
    """
    Manages retention policies and cleanup operations for error messages.

    Provides methods for applying retention policies, scheduled cleanup,
    and statistics generation.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None,
        default_policy: Optional[RetentionPolicy] = None
    ):
        """
        Initialize error retention manager.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
            default_policy: Optional default retention policy
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self._initialized = False
        self.default_policy = default_policy or RetentionPolicy()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_running = False
        self._cleanup_stats = CleanupStatistics()

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the error retention manager."""
        if self._initialized:
            return

        await self.connection_pool.initialize()
        self._initialized = True
        logger.info("Error retention manager initialized")

    async def close(self):
        """Close the error retention manager and cleanup scheduler."""
        if not self._initialized:
            return

        # Stop cleanup scheduler if running
        if self._cleanup_running:
            await self.stop_cleanup_scheduler()

        await self.connection_pool.close()
        self._initialized = False
        logger.info("Error retention manager closed")

    async def cleanup_old_errors(self, dry_run: bool = False) -> CleanupResult:
        """
        Clean up old error messages using default retention policy.

        Args:
            dry_run: If True, don't delete messages, just report what would be deleted

        Returns:
            CleanupResult with deletion statistics
        """
        return await self.apply_retention_policy(self.default_policy, dry_run=dry_run)

    async def apply_retention_policy(
        self,
        policy: RetentionPolicy,
        dry_run: bool = False
    ) -> CleanupResult:
        """
        Apply retention policy to delete old error messages.

        Args:
            policy: Retention policy to apply
            dry_run: If True, don't delete messages, just report what would be deleted

        Returns:
            CleanupResult with deletion statistics

        Raises:
            sqlite3.Error: If database operation fails
        """
        start_time = datetime.now(timezone.utc)
        result = CleanupResult(dry_run=dry_run)

        try:
            async with self.connection_pool.get_connection_async() as conn:
                # Process each severity level
                for severity, retention_days in policy.severity_specific_retention.items():
                    # Calculate cutoff dates for acknowledged and unacknowledged
                    unack_cutoff = (
                        datetime.now(timezone.utc) - timedelta(days=retention_days)
                    ).isoformat()

                    ack_retention_days = int(retention_days * policy.acknowledged_multiplier)
                    ack_cutoff = (
                        datetime.now(timezone.utc) - timedelta(days=ack_retention_days)
                    ).isoformat()

                    # Build query to identify deletable messages
                    query = """
                        SELECT id, severity, acknowledged, retry_count
                        FROM messages
                        WHERE severity = ?
                        AND (
                            (acknowledged = 0 AND timestamp < ?)
                            OR (acknowledged = 1 AND timestamp < ?)
                        )
                    """

                    # Add retry_count filter if configured
                    if policy.preserve_active_retries:
                        query += " AND retry_count = 0"

                    cursor = conn.execute(query, (severity, unack_cutoff, ack_cutoff))
                    deletable_ids = [row['id'] for row in cursor.fetchall()]

                    if deletable_ids:
                        severity_count = len(deletable_ids)
                        result.by_severity[severity] = severity_count
                        result.deleted_count += severity_count

                        if not dry_run:
                            # Delete messages in batches
                            placeholders = ','.join('?' * len(deletable_ids))
                            delete_query = f"DELETE FROM messages WHERE id IN ({placeholders})"
                            conn.execute(delete_query, deletable_ids)

                # Apply max_count policy if specified
                if policy.max_count:
                    count_query = "SELECT COUNT(*) as count FROM messages"
                    cursor = conn.execute(count_query)
                    total_count = cursor.fetchone()['count']

                    if total_count > policy.max_count:
                        excess_count = total_count - policy.max_count

                        # Get oldest messages to delete (excluding active retries if configured)
                        oldest_query = """
                            SELECT id FROM messages
                        """
                        if policy.preserve_active_retries:
                            oldest_query += " WHERE retry_count = 0"

                        oldest_query += " ORDER BY timestamp ASC LIMIT ?"

                        cursor = conn.execute(oldest_query, (excess_count,))
                        excess_ids = [row['id'] for row in cursor.fetchall()]

                        if excess_ids:
                            result.deleted_count += len(excess_ids)

                            if not dry_run:
                                placeholders = ','.join('?' * len(excess_ids))
                                delete_query = f"DELETE FROM messages WHERE id IN ({placeholders})"
                                conn.execute(delete_query, excess_ids)

                # Count preserved messages
                preserved_query = "SELECT COUNT(*) as count FROM messages"
                cursor = conn.execute(preserved_query)
                result.preserved_count = cursor.fetchone()['count']

                if not dry_run:
                    conn.commit()
                    logger.info(
                        f"Cleanup completed: deleted {result.deleted_count} messages, "
                        f"preserved {result.preserved_count} messages"
                    )
                else:
                    logger.info(
                        f"Dry-run cleanup: would delete {result.deleted_count} messages, "
                        f"preserve {result.preserved_count} messages"
                    )

                # Update statistics
                if not dry_run:
                    duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                    self._cleanup_stats.last_cleanup_at = start_time
                    self._cleanup_stats.total_cleanups += 1
                    self._cleanup_stats.total_deleted += result.deleted_count
                    self._cleanup_stats.average_deleted_per_cleanup = (
                        self._cleanup_stats.total_deleted / self._cleanup_stats.total_cleanups
                    )
                    self._cleanup_stats.last_cleanup_duration = duration

        except sqlite3.Error as e:
            error_msg = f"Database error during cleanup: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            raise

        return result

    async def schedule_cleanup(self, interval_hours: int = 24) -> bool:
        """
        Schedule automatic cleanup operations at regular intervals.

        Args:
            interval_hours: Cleanup interval in hours (default: 24)

        Returns:
            True if scheduler started successfully, False if already running

        Raises:
            ValueError: If interval_hours is invalid
        """
        if interval_hours <= 0:
            raise ValueError("interval_hours must be positive")

        if self._cleanup_running:
            logger.warning("Cleanup scheduler already running")
            return False

        self._cleanup_running = True
        interval_seconds = interval_hours * 3600

        async def cleanup_loop():
            """Background cleanup loop."""
            while self._cleanup_running:
                try:
                    logger.info("Starting scheduled cleanup")
                    result = await self.cleanup_old_errors(dry_run=False)
                    logger.info(
                        f"Scheduled cleanup completed: deleted {result.deleted_count} messages"
                    )
                except Exception as e:
                    logger.error(f"Error during scheduled cleanup: {e}")

                # Wait for next interval
                await asyncio.sleep(interval_seconds)

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Cleanup scheduler started with {interval_hours}h interval")
        return True

    async def stop_cleanup_scheduler(self) -> bool:
        """
        Stop the background cleanup scheduler.

        Returns:
            True if scheduler was stopped, False if not running
        """
        if not self._cleanup_running:
            logger.warning("Cleanup scheduler not running")
            return False

        self._cleanup_running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

        logger.info("Cleanup scheduler stopped")
        return True

    def get_cleanup_stats(self) -> CleanupStatistics:
        """
        Get cleanup operation statistics.

        Returns:
            CleanupStatistics with aggregated cleanup data
        """
        return self._cleanup_stats
