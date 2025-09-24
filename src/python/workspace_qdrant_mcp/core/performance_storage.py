"""
Performance Storage System for Workspace Qdrant MCP.

This module provides persistent storage capabilities for performance metrics,
operation traces, and analysis reports.

Task 265: Performance storage for the workspace_qdrant_mcp system.
"""

import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from loguru import logger

from .performance_metrics import PerformanceMetric, OperationTrace, MetricType
from .performance_analytics import PerformanceReport


class PerformanceStorage:
    """Storage system for performance metrics and analysis results."""

    def __init__(self, project_id: str, storage_path: Optional[Path] = None):
        self.project_id = project_id
        self.storage_path = storage_path or Path.cwd() / ".performance" / project_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.db_path = self.storage_path / "performance.db"
        self._init_database()
        self.retention_days = 7  # Keep data for 7 days

    def _init_database(self):
        """Initialize SQLite database for performance data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_type TEXT NOT NULL,
                    value REAL NOT NULL,
                    unit TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    context TEXT,
                    tags TEXT,
                    operation_id TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS operation_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation_id TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    project_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    context TEXT,
                    error_message TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    report_data TEXT NOT NULL
                )
            """)

            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_operation ON operation_traces(operation_type)")

    async def store_metric(self, metric: PerformanceMetric):
        """Store a single performance metric."""
        await self.store_metrics_batch([metric])

    async def store_metrics_batch(self, metrics: List[PerformanceMetric]):
        """Store a batch of metrics for performance."""
        if not metrics:
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                values = []
                for metric in metrics:
                    values.append((
                        metric.timestamp.isoformat(),
                        metric.metric_type.value,
                        metric.value,
                        metric.unit,
                        metric.project_id,
                        json.dumps(metric.context),
                        json.dumps(metric.tags),
                        metric.operation_id
                    ))

                conn.executemany("""
                    INSERT INTO metrics
                    (timestamp, metric_type, value, unit, project_id, context, tags, operation_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, values)

        except Exception as e:
            logger.error(f"Failed to store metrics batch: {e}")
            raise

    async def store_operation_trace(self, trace: OperationTrace):
        """Store an operation trace."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO operation_traces
                    (operation_id, operation_type, start_time, end_time, project_id, status, context, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trace.operation_id,
                    trace.operation_type,
                    trace.start_time.isoformat(),
                    trace.end_time.isoformat() if trace.end_time else None,
                    trace.project_id,
                    trace.status,
                    json.dumps(trace.context),
                    trace.error_message
                ))

        except Exception as e:
            logger.error(f"Failed to store operation trace: {e}")
            raise

    async def store_performance_report(self, report: PerformanceReport):
        """Store a performance analysis report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_reports
                    (project_id, generated_at, report_data)
                    VALUES (?, ?, ?)
                """, (
                    report.project_id,
                    report.generated_at.isoformat(),
                    json.dumps(report.to_dict())
                ))

        except Exception as e:
            logger.error(f"Failed to store performance report: {e}")
            raise

    async def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Retrieve metrics with optional filtering."""
        try:
            query = "SELECT * FROM metrics WHERE project_id = ?"
            params = [self.project_id]

            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type.value)

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY timestamp DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                metrics = []
                for row in rows:
                    metric = PerformanceMetric(
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        metric_type=MetricType(row['metric_type']),
                        value=row['value'],
                        unit=row['unit'],
                        project_id=row['project_id'],
                        context=json.loads(row['context']) if row['context'] else {},
                        tags=json.loads(row['tags']) if row['tags'] else [],
                        operation_id=row['operation_id']
                    )
                    metrics.append(metric)

                return metrics

        except Exception as e:
            logger.error(f"Failed to retrieve metrics: {e}")
            return []

    async def get_operation_traces(
        self,
        operation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[OperationTrace]:
        """Retrieve operation traces with optional filtering."""
        try:
            query = "SELECT * FROM operation_traces WHERE project_id = ?"
            params = [self.project_id]

            if operation_type:
                query += " AND operation_type = ?"
                params.append(operation_type)

            if start_time:
                query += " AND start_time >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND start_time <= ?"
                params.append(end_time.isoformat())

            query += " ORDER BY start_time DESC"

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()

                traces = []
                for row in rows:
                    trace = OperationTrace(
                        operation_id=row['operation_id'],
                        operation_type=row['operation_type'],
                        start_time=datetime.fromisoformat(row['start_time']),
                        end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                        project_id=row['project_id'],
                        status=row['status'],
                        context=json.loads(row['context']) if row['context'] else {},
                        error_message=row['error_message']
                    )
                    traces.append(trace)

                return traces

        except Exception as e:
            logger.error(f"Failed to retrieve operation traces: {e}")
            return []

    async def cleanup_old_data(self, retention_days: Optional[int] = None):
        """Remove old data beyond retention period."""
        retention = retention_days or self.retention_days
        cutoff_time = datetime.now() - timedelta(days=retention)

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Clean old metrics
                conn.execute("""
                    DELETE FROM metrics
                    WHERE timestamp < ? AND project_id = ?
                """, (cutoff_time.isoformat(), self.project_id))

                # Clean old traces
                conn.execute("""
                    DELETE FROM operation_traces
                    WHERE start_time < ? AND project_id = ?
                """, (cutoff_time.isoformat(), self.project_id))

                # Clean old reports
                conn.execute("""
                    DELETE FROM performance_reports
                    WHERE generated_at < ? AND project_id = ?
                """, (cutoff_time.isoformat(), self.project_id))

            logger.info(f"Cleaned up performance data older than {retention} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {}

            # Database file size
            if self.db_path.exists():
                stats["database_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

            # Total storage directory size
            total_size = sum(f.stat().st_size for f in self.storage_path.rglob('*') if f.is_file())
            stats["total_size_mb"] = total_size / (1024 * 1024)

            # Record counts
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM metrics WHERE project_id = ?", (self.project_id,))
                stats["metrics_count"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM operation_traces WHERE project_id = ?", (self.project_id,))
                stats["traces_count"] = cursor.fetchone()[0]

                cursor = conn.execute("SELECT COUNT(*) FROM performance_reports WHERE project_id = ?", (self.project_id,))
                stats["reports_count"] = cursor.fetchone()[0]

            return stats

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    def close(self):
        """Close storage connections."""
        # SQLite connections are closed automatically in context managers
        pass


# Global storage instances
_storage_instances: Dict[str, PerformanceStorage] = {}


async def get_performance_storage(project_id: str) -> PerformanceStorage:
    """Get or create a performance storage instance for a project."""
    if project_id not in _storage_instances:
        _storage_instances[project_id] = PerformanceStorage(project_id)
    return _storage_instances[project_id]


# Re-export all classes
__all__ = [
    "PerformanceStorage",
    "get_performance_storage"
]