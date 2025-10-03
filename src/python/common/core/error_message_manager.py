"""
Error Message Management API

Provides a high-level API for recording, retrieving, and managing error messages
with automatic categorization, severity tracking, and acknowledgment workflow.

Features:
    - Automatic error categorization using ErrorCategorizer
    - Severity tracking (error, warning, info)
    - Context preservation as JSON
    - Acknowledgment workflow
    - Advanced filtering capabilities
    - Statistics generation

Example:
    ```python
    from workspace_qdrant_mcp.core.error_message_manager import ErrorMessageManager

    # Initialize manager
    manager = ErrorMessageManager()
    await manager.initialize()

    # Record an error
    try:
        # Some operation
        pass
    except Exception as e:
        error_id = await manager.record_error(
            exception=e,
            context={
                'file_path': '/path/to/file.py',
                'collection': 'my-project',
                'tenant_id': 'default'
            }
        )

    # Retrieve errors
    errors = await manager.get_errors(severity='error', limit=10)

    # Acknowledge error
    await manager.acknowledge_error(error_id, acknowledged_by='admin')

    # Get statistics
    stats = await manager.get_error_stats()
    ```
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from .error_categorization import ErrorCategorizer, ErrorCategory, ErrorSeverity
from .queue_connection import QueueConnectionPool, ConnectionConfig


@dataclass
class ErrorMessage:
    """
    Represents an error message record.

    Attributes:
        id: Unique identifier
        timestamp: When the error occurred
        severity: Error severity level
        category: Error category
        message: Human-readable error message
        context: Additional context as dictionary
        acknowledged: Whether error has been acknowledged
        acknowledged_at: When error was acknowledged
        acknowledged_by: Who acknowledged the error
        retry_count: Number of retries attempted
    """
    id: int
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    context: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    retry_count: int = 0

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "ErrorMessage":
        """
        Create ErrorMessage from database row.

        Args:
            row: SQLite row from messages table

        Returns:
            ErrorMessage instance
        """
        # Parse context JSON
        context = None
        if row["context"]:
            try:
                context = json.loads(row["context"])
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in context for error {row['id']}")
                context = {"raw": row["context"]}

        # Parse timestamps
        timestamp = datetime.fromisoformat(row["timestamp"])
        acknowledged_at = None
        if row["acknowledged_at"]:
            acknowledged_at = datetime.fromisoformat(row["acknowledged_at"])

        return cls(
            id=row["id"],
            timestamp=timestamp,
            severity=ErrorSeverity.from_string(row["severity"]),
            category=ErrorCategory.from_string(row["category"]),
            message=row["message"],
            context=context,
            acknowledged=bool(row["acknowledged"]),
            acknowledged_at=acknowledged_at,
            acknowledged_by=row["acknowledged_by"],
            retry_count=row["retry_count"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all error message fields
        """
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "message": self.message,
            "context": self.context,
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "retry_count": self.retry_count
        }


@dataclass
class ErrorStatistics:
    """
    Aggregated error statistics.

    Attributes:
        total_count: Total number of errors
        by_severity: Count by severity level
        by_category: Count by category
        unacknowledged_count: Number of unacknowledged errors
        last_error_at: Timestamp of most recent error
    """
    total_count: int
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    unacknowledged_count: int = 0
    last_error_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.

        Returns:
            Dictionary with all statistics
        """
        return {
            "total_count": self.total_count,
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "unacknowledged_count": self.unacknowledged_count,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None
        }


class ErrorMessageManager:
    """
    High-level manager for error message operations.

    Provides methods for recording errors with automatic categorization,
    retrieving errors with filtering, acknowledgment workflow, and statistics.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        connection_config: Optional[ConnectionConfig] = None
    ):
        """
        Initialize error message manager.

        Args:
            db_path: Optional custom database path
            connection_config: Optional connection configuration
        """
        self.connection_pool = QueueConnectionPool(
            db_path=db_path or self._get_default_db_path(),
            config=connection_config or ConnectionConfig()
        )
        self._initialized = False
        self.categorizer = ErrorCategorizer()

    def _get_default_db_path(self) -> str:
        """Get default database path from OS directories."""
        from ..utils.os_directories import OSDirectories
        os_dirs = OSDirectories()
        os_dirs.ensure_directories()
        return str(os_dirs.get_state_file("workspace_state.db"))

    async def initialize(self):
        """Initialize the error message manager."""
        if self._initialized:
            return

        await self.connection_pool.initialize()

        # Check if schema exists
        if not await self._check_schema_exists():
            logger.warning(
                "Error messages schema not found. "
                "Run migration: python -m src.python.common.core.migrate_error_messages"
            )

        self._initialized = True
        logger.info("Error message manager initialized")

    async def close(self):
        """Close the error message manager."""
        if not self._initialized:
            return

        await self.connection_pool.close()
        self._initialized = False
        logger.info("Error message manager closed")

    async def _check_schema_exists(self) -> bool:
        """
        Check if error messages schema exists.

        Returns:
            True if schema exists, False otherwise
        """
        try:
            async with self.connection_pool.get_connection_async() as conn:
                cursor = conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='messages'"
                )
                result = cursor.fetchone()

                if not result:
                    return False

                table_sql = result[0]
                # Check for enhanced schema fields
                return 'severity' in table_sql and 'category' in table_sql
        except sqlite3.Error:
            return False

    async def record_error(
        self,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
        severity_override: Optional[ErrorSeverity] = None,
        category_override: Optional[ErrorCategory] = None,
        message_override: Optional[str] = None
    ) -> int:
        """
        Record an error message with automatic categorization.

        Args:
            exception: The exception object (if available)
            context: Context dictionary with keys like:
                - file_path: File path where error occurred
                - collection: Collection name
                - tenant_id: Tenant identifier
                - (any other arbitrary fields)
            severity_override: Manual severity override
            category_override: Manual category override
            message_override: Manual message override (defaults to exception message)

        Returns:
            Error message ID

        Raises:
            ValueError: If neither exception nor message_override provided
            sqlite3.Error: If database operation fails
        """
        if not exception and not message_override:
            raise ValueError("Either exception or message_override must be provided")

        # Extract message
        message = message_override or str(exception)

        # Categorize error
        category, severity, confidence = self.categorizer.categorize(
            exception=exception,
            message=message,
            context=context,
            manual_category=category_override,
            manual_severity=severity_override
        )

        # Prepare context JSON - preserve empty dict as valid JSON
        context_json = None
        if context is not None:
            context_json = json.dumps(context)

        # Insert into database
        query = """
            INSERT INTO messages (
                timestamp, severity, category, message, context, retry_count
            ) VALUES (?, ?, ?, ?, ?, ?)
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (
                datetime.now(timezone.utc).isoformat(),
                severity.value,
                category.value,
                message,
                context_json,
                0  # retry_count starts at 0
            ))
            conn.commit()

            error_id = cursor.lastrowid

            logger.debug(
                f"Recorded error {error_id}: {severity.value}/{category.value} - {message[:50]}..."
            )

            return error_id

    async def get_error_by_id(self, error_id: int) -> Optional[ErrorMessage]:
        """
        Retrieve a single error message by ID.

        Args:
            error_id: Error message ID

        Returns:
            ErrorMessage instance or None if not found
        """
        query = """
            SELECT
                id, timestamp, severity, category, message, context,
                acknowledged, acknowledged_at, acknowledged_by, retry_count
            FROM messages
            WHERE id = ?
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (error_id,))
            row = cursor.fetchone()

            if not row:
                return None

            return ErrorMessage.from_db_row(row)

    async def get_errors(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        acknowledged: Optional[bool] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ErrorMessage]:
        """
        Retrieve error messages with filtering.

        Args:
            severity: Filter by severity ('error', 'warning', 'info')
            category: Filter by category (e.g., 'file_corrupt', 'network')
            acknowledged: Filter by acknowledgment status
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            List of ErrorMessage instances

        Raises:
            ValueError: If severity or category is invalid
        """
        # Validate inputs
        if severity:
            ErrorSeverity.from_string(severity)  # Raises ValueError if invalid
        if category:
            ErrorCategory.from_string(category)  # Raises ValueError if invalid

        # Build query
        query = """
            SELECT
                id, timestamp, severity, category, message, context,
                acknowledged, acknowledged_at, acknowledged_by, retry_count
            FROM messages
        """

        conditions = []
        params = []

        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        if category:
            conditions.append("category = ?")
            params.append(category)

        if acknowledged is not None:
            conditions.append("acknowledged = ?")
            params.append(1 if acknowledged else 0)

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date.isoformat())

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Order by timestamp descending (most recent first)
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            rows = cursor.fetchall()

            errors = [ErrorMessage.from_db_row(row) for row in rows]

            logger.debug(f"Retrieved {len(errors)} error messages")
            return errors

    async def acknowledge_error(
        self,
        error_id: int,
        acknowledged_by: str
    ) -> bool:
        """
        Mark an error as acknowledged.

        Args:
            error_id: Error message ID
            acknowledged_by: Username or identifier of who acknowledged

        Returns:
            True if acknowledged, False if not found

        Raises:
            ValueError: If acknowledged_by is empty
        """
        if not acknowledged_by:
            raise ValueError("acknowledged_by must be provided")

        query = """
            UPDATE messages
            SET acknowledged = 1, acknowledged_by = ?
            WHERE id = ?
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, (acknowledged_by, error_id))
            conn.commit()

            updated = cursor.rowcount > 0

            if updated:
                logger.debug(f"Acknowledged error {error_id} by {acknowledged_by}")
            else:
                logger.warning(f"Error {error_id} not found for acknowledgment")

            return updated

    async def get_error_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> ErrorStatistics:
        """
        Get aggregated error statistics.

        Args:
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering

        Returns:
            ErrorStatistics with aggregated data
        """
        # Build base query
        where_clause = ""
        params = []

        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append("timestamp >= ?")
                params.append(start_date.isoformat())
            if end_date:
                conditions.append("timestamp <= ?")
                params.append(end_date.isoformat())
            where_clause = " WHERE " + " AND ".join(conditions)

        # Get overall stats
        query = f"""
            SELECT
                COUNT(*) as total_count,
                SUM(CASE WHEN acknowledged = 0 THEN 1 ELSE 0 END) as unacknowledged_count,
                MAX(timestamp) as last_error_at
            FROM messages
            {where_clause}
        """

        async with self.connection_pool.get_connection_async() as conn:
            cursor = conn.execute(query, tuple(params))
            row = cursor.fetchone()

            total_count = row["total_count"] or 0
            unacknowledged_count = row["unacknowledged_count"] or 0
            last_error_at = None
            if row["last_error_at"]:
                last_error_at = datetime.fromisoformat(row["last_error_at"])

            # Get severity breakdown
            query = f"""
                SELECT severity, COUNT(*) as count
                FROM messages
                {where_clause}
                GROUP BY severity
            """
            cursor = conn.execute(query, tuple(params))
            by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}

            # Get category breakdown
            query = f"""
                SELECT category, COUNT(*) as count
                FROM messages
                {where_clause}
                GROUP BY category
            """
            cursor = conn.execute(query, tuple(params))
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}

            return ErrorStatistics(
                total_count=total_count,
                by_severity=by_severity,
                by_category=by_category,
                unacknowledged_count=unacknowledged_count,
                last_error_at=last_error_at
            )
