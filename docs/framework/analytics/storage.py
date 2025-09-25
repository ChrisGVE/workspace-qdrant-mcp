"""Analytics storage backend using SQLite for documentation usage tracking."""

import sqlite3
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import logging


logger = logging.getLogger(__name__)


@dataclass
class AnalyticsEvent:
    """Represents a single analytics event."""
    event_type: str
    timestamp: datetime
    session_id: str
    page_path: str
    user_agent_hash: Optional[str] = None
    duration_ms: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metadata'] = json.dumps(self.metadata) if self.metadata else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsEvent':
        """Create from dictionary loaded from storage."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('metadata'):
            data['metadata'] = json.loads(data['metadata'])
        return cls(**data)


@dataclass
class AnalyticsStats:
    """Analytics statistics summary."""
    total_events: int
    unique_sessions: int
    total_page_views: int
    avg_session_duration_ms: float
    top_pages: List[Dict[str, Any]]
    top_search_queries: List[Dict[str, Any]]
    error_rate: float


class AnalyticsStorage:
    """SQLite-based storage for analytics data with privacy controls."""

    def __init__(self, db_path: Union[str, Path], retention_days: int = 90):
        """Initialize analytics storage.

        Args:
            db_path: Path to SQLite database file
            retention_days: Number of days to retain analytics data
        """
        self.db_path = Path(db_path)
        self.retention_days = retention_days
        self._lock = threading.RLock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    page_path TEXT NOT NULL,
                    user_agent_hash TEXT,
                    duration_ms INTEGER,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_timestamp
                ON analytics_events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session
                ON analytics_events(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_type
                ON analytics_events(event_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_page
                ON analytics_events(page_path)
            """)

            # Sessions table for aggregated data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analytics_sessions (
                    session_id TEXT PRIMARY KEY,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    page_count INTEGER DEFAULT 0,
                    total_duration_ms INTEGER DEFAULT 0,
                    user_agent_hash TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a thread-safe database connection."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def store_event(self, event: AnalyticsEvent) -> bool:
        """Store an analytics event.

        Args:
            event: Analytics event to store

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Insert event
                event_data = event.to_dict()
                cursor.execute("""
                    INSERT INTO analytics_events
                    (event_type, timestamp, session_id, page_path,
                     user_agent_hash, duration_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    event_data['event_type'],
                    event_data['timestamp'],
                    event_data['session_id'],
                    event_data['page_path'],
                    event_data['user_agent_hash'],
                    event_data['duration_ms'],
                    event_data['metadata']
                ))

                # Update or insert session data
                cursor.execute("""
                    INSERT OR REPLACE INTO analytics_sessions
                    (session_id, first_seen, last_seen, page_count,
                     total_duration_ms, user_agent_hash, updated_at)
                    VALUES (
                        ?,
                        COALESCE((SELECT first_seen FROM analytics_sessions WHERE session_id = ?), ?),
                        ?,
                        COALESCE((SELECT page_count FROM analytics_sessions WHERE session_id = ?), 0) + 1,
                        COALESCE((SELECT total_duration_ms FROM analytics_sessions WHERE session_id = ?), 0) + ?,
                        ?,
                        CURRENT_TIMESTAMP
                    )
                """, (
                    event.session_id,
                    event.session_id, event.timestamp.isoformat(),
                    event.timestamp.isoformat(),
                    event.session_id,
                    event.session_id, event.duration_ms or 0,
                    event.user_agent_hash
                ))

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to store analytics event: {e}")
            return False

    def get_events(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_type: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyticsEvent]:
        """Retrieve analytics events with optional filtering.

        Args:
            start_date: Start date filter
            end_date: End date filter
            event_type: Event type filter
            session_id: Session ID filter
            limit: Maximum number of events to return

        Returns:
            List of matching analytics events
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                query = "SELECT * FROM analytics_events WHERE 1=1"
                params = []

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date.isoformat())

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date.isoformat())

                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)

                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                events = []
                for row in rows:
                    event_data = dict(row)
                    # Remove database-specific fields that aren't part of AnalyticsEvent
                    event_data.pop('id', None)
                    event_data.pop('created_at', None)
                    events.append(AnalyticsEvent.from_dict(event_data))

                return events

        except Exception as e:
            logger.error(f"Failed to retrieve analytics events: {e}")
            return []

    def get_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Optional[AnalyticsStats]:
        """Get analytics statistics summary.

        Args:
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Analytics statistics or None if query failed
        """
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Basic stats
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_events,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        COUNT(CASE WHEN event_type = 'page_view' THEN 1 END) as total_page_views
                    FROM analytics_events
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))

                basic_stats = cursor.fetchone()

                # Average session duration
                cursor.execute("""
                    SELECT AVG(total_duration_ms) as avg_duration
                    FROM analytics_sessions
                    WHERE first_seen BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))

                avg_duration_result = cursor.fetchone()
                avg_duration = avg_duration_result['avg_duration'] if avg_duration_result['avg_duration'] else 0.0

                # Top pages
                cursor.execute("""
                    SELECT page_path, COUNT(*) as views
                    FROM analytics_events
                    WHERE event_type = 'page_view'
                    AND timestamp BETWEEN ? AND ?
                    GROUP BY page_path
                    ORDER BY views DESC
                    LIMIT 10
                """, (start_date.isoformat(), end_date.isoformat()))

                top_pages = [{'page': row['page_path'], 'views': row['views']}
                           for row in cursor.fetchall()]

                # Top search queries
                cursor.execute("""
                    SELECT
                        JSON_EXTRACT(metadata, '$.query') as query,
                        COUNT(*) as count
                    FROM analytics_events
                    WHERE event_type = 'search'
                    AND timestamp BETWEEN ? AND ?
                    AND JSON_EXTRACT(metadata, '$.query') IS NOT NULL
                    GROUP BY JSON_EXTRACT(metadata, '$.query')
                    ORDER BY count DESC
                    LIMIT 10
                """, (start_date.isoformat(), end_date.isoformat()))

                top_searches = [{'query': row['query'], 'count': row['count']}
                              for row in cursor.fetchall()]

                # Error rate
                cursor.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(CASE WHEN event_type = 'error' THEN 1 END) as errors
                    FROM analytics_events
                    WHERE timestamp BETWEEN ? AND ?
                """, (start_date.isoformat(), end_date.isoformat()))

                error_stats = cursor.fetchone()
                error_rate = (error_stats['errors'] / error_stats['total']) * 100 if error_stats['total'] > 0 else 0.0

                return AnalyticsStats(
                    total_events=basic_stats['total_events'],
                    unique_sessions=basic_stats['unique_sessions'],
                    total_page_views=basic_stats['total_page_views'],
                    avg_session_duration_ms=avg_duration,
                    top_pages=top_pages,
                    top_search_queries=top_searches,
                    error_rate=error_rate
                )

        except Exception as e:
            logger.error(f"Failed to get analytics stats: {e}")
            return None

    def cleanup_old_data(self) -> int:
        """Remove analytics data older than retention period.

        Returns:
            Number of records deleted
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Delete old events
                cursor.execute("""
                    DELETE FROM analytics_events
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))

                deleted_events = cursor.rowcount

                # Delete orphaned sessions
                cursor.execute("""
                    DELETE FROM analytics_sessions
                    WHERE session_id NOT IN (
                        SELECT DISTINCT session_id FROM analytics_events
                    )
                """)

                deleted_sessions = cursor.rowcount

                conn.commit()

                logger.info(f"Cleaned up {deleted_events} old events and {deleted_sessions} orphaned sessions")
                return deleted_events + deleted_sessions

        except Exception as e:
            logger.error(f"Failed to cleanup old analytics data: {e}")
            return 0

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the analytics database.

        Returns:
            Database information including size and record counts
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Count records
                cursor.execute("SELECT COUNT(*) as count FROM analytics_events")
                event_count = cursor.fetchone()['count']

                cursor.execute("SELECT COUNT(*) as count FROM analytics_sessions")
                session_count = cursor.fetchone()['count']

                # Database file size
                db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

                return {
                    'database_path': str(self.db_path),
                    'database_size_bytes': db_size,
                    'event_count': event_count,
                    'session_count': session_count,
                    'retention_days': self.retention_days
                }

        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return {
                'database_path': str(self.db_path),
                'error': str(e)
            }

    def hash_user_agent(self, user_agent: str) -> str:
        """Create a privacy-safe hash of user agent string.

        Args:
            user_agent: Raw user agent string

        Returns:
            SHA-256 hash of user agent (first 16 characters)
        """
        if not user_agent:
            return ""

        hash_obj = hashlib.sha256(user_agent.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for storage efficiency

    def export_data(self, output_path: Path, start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> bool:
        """Export analytics data to JSON file.

        Args:
            output_path: Path for exported data
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            True if export successful, False otherwise
        """
        try:
            events = self.get_events(start_date=start_date, end_date=end_date, limit=10000)
            stats = self.get_stats(start_date=start_date, end_date=end_date)

            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'date_range': {
                    'start': start_date.isoformat() if start_date else None,
                    'end': end_date.isoformat() if end_date else None
                },
                'stats': asdict(stats) if stats else None,
                'event_count': len(events),
                'events': [event.to_dict() for event in events]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported {len(events)} analytics events to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export analytics data: {e}")
            return False