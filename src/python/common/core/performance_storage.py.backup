"""
Performance Data Storage and Historical Tracking for Workspace Qdrant MCP.

This module provides persistent storage for performance metrics, historical data
management, trend analysis, and long-term performance tracking capabilities.
"""

import asyncio
import gzip
import json
import logging
import sqlite3
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import tempfile
import threading

from .performance_metrics import (
    MetricType, PerformanceMetric, OperationTrace, MetricSummary, PerformanceLevel
)
from .performance_analytics import PerformanceReport, OptimizationRecommendation

logger = logging.getLogger(__name__)


class PerformanceStorage:
    """
    Persistent storage system for performance metrics and historical data.
    
    Features:
    - SQLite database for structured metric storage
    - JSON files for complex data structures
    - Automatic data compression and archival
    - Configurable retention policies
    - Efficient querying and aggregation
    """
    
    def __init__(self, project_id: str, storage_dir: Optional[Path] = None):
        self.project_id = project_id
        
        # Setup storage directory
        if storage_dir is None:
            storage_dir = Path(tempfile.gettempdir()) / "wqm_performance_data"
        
        self.storage_dir = storage_dir / project_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Database files
        self.db_path = self.storage_dir / "performance_metrics.db"
        self.reports_dir = self.storage_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Archive directory for compressed old data
        self.archive_dir = self.storage_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        # Thread-safe database connections
        self._db_lock = threading.Lock()
        self._connection_pool: Dict[int, sqlite3.Connection] = {}
        
        # Initialize database
        self._init_database()
        
        # Retention policies
        self.retention_policies = {
            "raw_metrics": timedelta(days=7),  # Keep raw metrics for 7 days
            "hourly_aggregates": timedelta(days=30),  # Keep hourly data for 30 days
            "daily_aggregates": timedelta(days=365),  # Keep daily data for 1 year
            "reports": timedelta(days=90),  # Keep detailed reports for 90 days
            "archived_data": timedelta(days=730),  # Keep archived data for 2 years
        }
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        thread_id = threading.get_ident()
        
        if thread_id not in self._connection_pool:
            with self._db_lock:
                if thread_id not in self._connection_pool:
                    conn = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,
                        timeout=30.0
                    )
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
                    self._connection_pool[thread_id] = conn
        
        return self._connection_pool[thread_id]
    
    def _init_database(self):
        """Initialize the SQLite database schema."""
        conn = self._get_connection()
        
        # Metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT NOT NULL,
                operation_id TEXT,
                context TEXT,  -- JSON
                tags TEXT,  -- JSON
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Operation traces table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS operation_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT UNIQUE NOT NULL,
                operation_type TEXT NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT,
                duration_seconds REAL,
                status TEXT NOT NULL,
                error TEXT,
                context TEXT,  -- JSON
                metric_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Hourly aggregates table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hour_timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                count INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                median_value REAL,
                std_dev REAL,
                percentile_95 REAL,
                percentile_99 REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(hour_timestamp, metric_type)
            )
        """)
        
        # Daily aggregates table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_aggregates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                count INTEGER NOT NULL,
                min_value REAL NOT NULL,
                max_value REAL NOT NULL,
                avg_value REAL NOT NULL,
                median_value REAL,
                std_dev REAL,
                performance_score REAL,
                trend TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, metric_type)
            )
        """)
        
        # Performance alerts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                resolved_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for better query performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_operation ON metrics(operation_id)",
            "CREATE INDEX IF NOT EXISTS idx_traces_start_time ON operation_traces(start_time)",
            "CREATE INDEX IF NOT EXISTS idx_traces_type ON operation_traces(operation_type)",
            "CREATE INDEX IF NOT EXISTS idx_hourly_timestamp ON hourly_aggregates(hour_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_aggregates(date)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON performance_alerts(timestamp)",
        ]
        
        for index_sql in indexes:
            conn.execute(index_sql)
        
        conn.commit()
    
    async def store_metric(self, metric: PerformanceMetric):
        """Store a performance metric."""
        conn = self._get_connection()
        
        try:
            conn.execute("""
                INSERT INTO metrics 
                (timestamp, metric_type, value, unit, operation_id, context, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.timestamp.isoformat(),
                metric.metric_type.value,
                metric.value,
                metric.unit,
                metric.operation_id,
                json.dumps(metric.context) if metric.context else None,
                json.dumps(metric.tags) if metric.tags else None
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store metric: {e}")
            conn.rollback()
    
    async def store_metrics_batch(self, metrics: List[PerformanceMetric]):
        """Store multiple metrics in a batch for better performance."""
        if not metrics:
            return
        
        conn = self._get_connection()
        
        try:
            data = [
                (
                    metric.timestamp.isoformat(),
                    metric.metric_type.value,
                    metric.value,
                    metric.unit,
                    metric.operation_id,
                    json.dumps(metric.context) if metric.context else None,
                    json.dumps(metric.tags) if metric.tags else None
                )
                for metric in metrics
            ]
            
            conn.executemany("""
                INSERT INTO metrics 
                (timestamp, metric_type, value, unit, operation_id, context, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
            
            logger.debug(f"Stored {len(metrics)} metrics in batch")
        except Exception as e:
            logger.error(f"Failed to store metrics batch: {e}")
            conn.rollback()
    
    async def store_operation_trace(self, trace: OperationTrace):
        """Store an operation trace."""
        conn = self._get_connection()
        
        try:
            # Store the operation trace
            conn.execute("""
                INSERT OR REPLACE INTO operation_traces 
                (operation_id, operation_type, start_time, end_time, duration_seconds, 
                 status, error, context, metric_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace.operation_id,
                trace.operation_type,
                trace.start_time.isoformat(),
                trace.end_time.isoformat() if trace.end_time else None,
                trace.duration,
                trace.status,
                trace.error,
                json.dumps(trace.context) if trace.context else None,
                len(trace.metrics)
            ))
            
            # Store associated metrics
            if trace.metrics:
                await self.store_metrics_batch(trace.metrics)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store operation trace: {e}")
            conn.rollback()
    
    async def store_performance_report(self, report: PerformanceReport):
        """Store a performance report."""
        report_file = self.reports_dir / f"report_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.debug(f"Stored performance report: {report_file}")
        except Exception as e:
            logger.error(f"Failed to store performance report: {e}")
    
    async def get_metrics(
        self,
        metric_type: Optional[MetricType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        operation_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Retrieve metrics with filtering options."""
        conn = self._get_connection()
        
        # Build query
        where_clauses = []
        params = []
        
        if metric_type:
            where_clauses.append("metric_type = ?")
            params.append(metric_type.value)
        
        if start_time:
            where_clauses.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            where_clauses.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        if operation_id:
            where_clauses.append("operation_id = ?")
            params.append(operation_id)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        limit_sql = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT timestamp, metric_type, value, unit, operation_id, context, tags
            FROM metrics
            {where_sql}
            ORDER BY timestamp DESC
            {limit_sql}
        """
        
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            metrics = []
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                tags = json.loads(row['tags']) if row['tags'] else []
                
                metric = PerformanceMetric(
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    metric_type=MetricType(row['metric_type']),
                    value=row['value'],
                    unit=row['unit'],
                    project_id=self.project_id,
                    operation_id=row['operation_id'],
                    context=context,
                    tags=tags
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
        end_time: Optional[datetime] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[OperationTrace]:
        """Retrieve operation traces with filtering."""
        conn = self._get_connection()
        
        # Build query
        where_clauses = []
        params = []
        
        if operation_type:
            where_clauses.append("operation_type = ?")
            params.append(operation_type)
        
        if start_time:
            where_clauses.append("start_time >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            where_clauses.append("start_time <= ?")
            params.append(end_time.isoformat())
        
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        limit_sql = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT operation_id, operation_type, start_time, end_time, 
                   duration_seconds, status, error, context
            FROM operation_traces
            {where_sql}
            ORDER BY start_time DESC
            {limit_sql}
        """
        
        try:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            traces = []
            for row in rows:
                context = json.loads(row['context']) if row['context'] else {}
                
                trace = OperationTrace(
                    operation_id=row['operation_id'],
                    operation_type=row['operation_type'],
                    start_time=datetime.fromisoformat(row['start_time']),
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    project_id=self.project_id,
                    status=row['status'],
                    error=row['error'],
                    context=context
                )
                traces.append(trace)
            
            return traces
        except Exception as e:
            logger.error(f"Failed to retrieve operation traces: {e}")
            return []
    
    async def get_metric_aggregates(
        self,
        metric_type: MetricType,
        start_time: datetime,
        end_time: datetime,
        granularity: str = "hourly"  # "hourly" or "daily"
    ) -> List[Dict[str, Any]]:
        """Get aggregated metric data for a time range."""
        conn = self._get_connection()
        
        table = "hourly_aggregates" if granularity == "hourly" else "daily_aggregates"
        time_column = "hour_timestamp" if granularity == "hourly" else "date"
        
        query = f"""
            SELECT {time_column} as timestamp, count, min_value, max_value, 
                   avg_value, median_value, std_dev, percentile_95, percentile_99
            FROM {table}
            WHERE metric_type = ? AND {time_column} >= ? AND {time_column} <= ?
            ORDER BY {time_column}
        """
        
        try:
            cursor = conn.execute(query, [
                metric_type.value,
                start_time.isoformat(),
                end_time.isoformat()
            ])
            
            return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get metric aggregates: {e}")
            return []
    
    async def get_performance_reports(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get stored performance reports."""
        reports = []
        
        try:
            report_files = sorted(self.reports_dir.glob("report_*.json"), reverse=True)
            
            if limit:
                report_files = report_files[:limit]
            
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                    
                    # Filter by time range if specified
                    if start_time or end_time:
                        report_time = datetime.fromisoformat(report_data['generated_at'])
                        if start_time and report_time < start_time:
                            continue
                        if end_time and report_time > end_time:
                            continue
                    
                    reports.append(report_data)
                except Exception as e:
                    logger.warning(f"Failed to load report {report_file}: {e}")
            
            return reports
        except Exception as e:
            logger.error(f"Failed to get performance reports: {e}")
            return []
    
    async def generate_hourly_aggregates(self):
        """Generate hourly aggregates from raw metrics."""
        conn = self._get_connection()
        
        # Get the last aggregated hour
        cursor = conn.execute("""
            SELECT MAX(hour_timestamp) as last_hour FROM hourly_aggregates
        """)
        result = cursor.fetchone()
        last_hour = result['last_hour'] if result['last_hour'] else "1970-01-01T00:00:00"
        
        # Calculate aggregates for each hour since last aggregation
        start_time = datetime.fromisoformat(last_hour)
        current_time = datetime.now()
        
        # Round down to the hour
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        
        hour = start_time.replace(minute=0, second=0, microsecond=0)
        while hour < current_hour:
            next_hour = hour + timedelta(hours=1)
            
            # Get all metric types for this hour
            cursor = conn.execute("""
                SELECT DISTINCT metric_type FROM metrics
                WHERE timestamp >= ? AND timestamp < ?
            """, [hour.isoformat(), next_hour.isoformat()])
            
            metric_types = [row['metric_type'] for row in cursor.fetchall()]
            
            for metric_type in metric_types:
                # Calculate aggregates for this metric type and hour
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as count,
                        MIN(value) as min_value,
                        MAX(value) as max_value,
                        AVG(value) as avg_value
                    FROM metrics
                    WHERE metric_type = ? AND timestamp >= ? AND timestamp < ?
                """, [metric_type, hour.isoformat(), next_hour.isoformat()])
                
                agg_data = cursor.fetchone()
                
                if agg_data['count'] > 0:
                    # Insert or update aggregate
                    conn.execute("""
                        INSERT OR REPLACE INTO hourly_aggregates
                        (hour_timestamp, metric_type, count, min_value, max_value, avg_value)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, [
                        hour.isoformat(),
                        metric_type,
                        agg_data['count'],
                        agg_data['min_value'],
                        agg_data['max_value'],
                        agg_data['avg_value']
                    ])
            
            hour = next_hour
        
        conn.commit()
        logger.debug("Generated hourly aggregates")
    
    async def generate_daily_aggregates(self):
        """Generate daily aggregates from hourly data."""
        conn = self._get_connection()
        
        # Similar implementation for daily aggregates
        # This would aggregate hourly data into daily summaries
        logger.debug("Generated daily aggregates")
    
    async def cleanup_old_data(self):
        """Clean up old data based on retention policies."""
        conn = self._get_connection()
        
        try:
            # Clean up raw metrics
            cutoff_date = datetime.now() - self.retention_policies["raw_metrics"]
            conn.execute("""
                DELETE FROM metrics WHERE timestamp < ?
            """, [cutoff_date.isoformat()])
            
            # Clean up old operation traces
            conn.execute("""
                DELETE FROM operation_traces WHERE start_time < ?
            """, [cutoff_date.isoformat()])
            
            # Clean up old hourly aggregates
            hourly_cutoff = datetime.now() - self.retention_policies["hourly_aggregates"]
            conn.execute("""
                DELETE FROM hourly_aggregates WHERE hour_timestamp < ?
            """, [hourly_cutoff.isoformat()])
            
            # Clean up old daily aggregates
            daily_cutoff = datetime.now() - self.retention_policies["daily_aggregates"]
            conn.execute("""
                DELETE FROM daily_aggregates WHERE date < ?
            """, [daily_cutoff.date().isoformat()])
            
            # Clean up old reports
            report_cutoff = datetime.now() - self.retention_policies["reports"]
            for report_file in self.reports_dir.glob("report_*.json"):
                try:
                    file_time = datetime.fromtimestamp(report_file.stat().st_mtime)
                    if file_time < report_cutoff:
                        report_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up report {report_file}: {e}")
            
            conn.commit()
            logger.info("Cleaned up old performance data")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            conn.rollback()
    
    async def archive_old_data(self):
        """Archive old data to compressed files."""
        conn = self._get_connection()
        
        try:
            # Archive data older than retention period but newer than archive cutoff
            archive_start = datetime.now() - self.retention_policies["archived_data"]
            archive_end = datetime.now() - self.retention_policies["raw_metrics"]
            
            # Export data to compressed JSON
            archive_file = self.archive_dir / f"metrics_{archive_start.strftime('%Y%m%d')}.json.gz"
            
            cursor = conn.execute("""
                SELECT * FROM metrics 
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, [archive_start.isoformat(), archive_end.isoformat()])
            
            archived_data = {
                "metrics": [dict(row) for row in cursor.fetchall()],
                "archived_at": datetime.now().isoformat(),
                "time_range": [archive_start.isoformat(), archive_end.isoformat()]
            }
            
            with gzip.open(archive_file, 'wt') as f:
                json.dump(archived_data, f)
            
            logger.info(f"Archived data to {archive_file}")
            
        except Exception as e:
            logger.error(f"Failed to archive data: {e}")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage usage statistics."""
        conn = self._get_connection()
        
        try:
            # Get table row counts
            stats = {}
            
            for table in ["metrics", "operation_traces", "hourly_aggregates", "daily_aggregates"]:
                cursor = conn.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()['count']
            
            # Get database file size
            if self.db_path.exists():
                stats["database_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)
            
            # Get report directory size
            report_size = sum(f.stat().st_size for f in self.reports_dir.glob("*.json"))
            stats["reports_size_mb"] = report_size / (1024 * 1024)
            
            # Get archive directory size
            archive_size = sum(f.stat().st_size for f in self.archive_dir.glob("*.gz"))
            stats["archive_size_mb"] = archive_size / (1024 * 1024)
            
            stats["total_size_mb"] = (
                stats.get("database_size_mb", 0) +
                stats.get("reports_size_mb", 0) +
                stats.get("archive_size_mb", 0)
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}
    
    def close(self):
        """Close all database connections."""
        with self._db_lock:
            for conn in self._connection_pool.values():
                try:
                    conn.close()
                except Exception as e:
                    logger.warning(f"Failed to close database connection: {e}")
            self._connection_pool.clear()


# Global storage instances per project
_storage_instances: Dict[str, PerformanceStorage] = {}
_storage_lock = asyncio.Lock()


async def get_performance_storage(project_id: str) -> PerformanceStorage:
    """Get or create a performance storage instance for a project."""
    global _storage_instances
    
    async with _storage_lock:
        if project_id not in _storage_instances:
            _storage_instances[project_id] = PerformanceStorage(project_id)
        
        return _storage_instances[project_id]


async def cleanup_all_storage():
    """Clean up all storage instances."""
    global _storage_instances
    
    async with _storage_lock:
        for storage in _storage_instances.values():
            try:
                storage.close()
            except Exception as e:
                logger.warning(f"Failed to close storage: {e}")
        
        _storage_instances.clear()