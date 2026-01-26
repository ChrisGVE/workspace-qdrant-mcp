"""Unit tests for analytics storage system."""

import json
import os
import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add docs framework to path for testing (if present)
docs_framework_path = os.path.join(os.path.dirname(__file__), '../../docs/framework')
sys.path.insert(0, docs_framework_path)

try:
    from analytics.storage import AnalyticsEvent, AnalyticsStats, AnalyticsStorage
except ModuleNotFoundError:
    pytest.skip("analytics framework not available", allow_module_level=True)


class TestAnalyticsEvent:
    """Test AnalyticsEvent data class."""

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        event = AnalyticsEvent(
            event_type="page_view",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            session_id="test-session",
            page_path="/test-page",
            user_agent_hash="abc123",
            duration_ms=5000,
            metadata={"test": "data"}
        )

        result = event.to_dict()

        assert result["event_type"] == "page_view"
        assert result["timestamp"] == "2023-01-01T12:00:00"
        assert result["session_id"] == "test-session"
        assert result["page_path"] == "/test-page"
        assert result["user_agent_hash"] == "abc123"
        assert result["duration_ms"] == 5000
        assert result["metadata"] == '{"test": "data"}'

    def test_to_dict_with_none_values(self):
        """Test conversion with None values."""
        event = AnalyticsEvent(
            event_type="page_view",
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            session_id="test-session",
            page_path="/test-page"
        )

        result = event.to_dict()

        assert result["user_agent_hash"] is None
        assert result["duration_ms"] is None
        assert result["metadata"] is None

    def test_from_dict_conversion(self):
        """Test creation from dictionary."""
        data = {
            "event_type": "page_view",
            "timestamp": "2023-01-01T12:00:00",
            "session_id": "test-session",
            "page_path": "/test-page",
            "user_agent_hash": "abc123",
            "duration_ms": 5000,
            "metadata": '{"test": "data"}'
        }

        event = AnalyticsEvent.from_dict(data)

        assert event.event_type == "page_view"
        assert event.timestamp == datetime(2023, 1, 1, 12, 0, 0)
        assert event.session_id == "test-session"
        assert event.page_path == "/test-page"
        assert event.user_agent_hash == "abc123"
        assert event.duration_ms == 5000
        assert event.metadata == {"test": "data"}

    def test_from_dict_with_none_metadata(self):
        """Test creation from dict with None metadata."""
        data = {
            "event_type": "page_view",
            "timestamp": "2023-01-01T12:00:00",
            "session_id": "test-session",
            "page_path": "/test-page",
            "user_agent_hash": None,
            "duration_ms": None,
            "metadata": None
        }

        event = AnalyticsEvent.from_dict(data)

        assert event.metadata is None
        assert event.user_agent_hash is None
        assert event.duration_ms is None


class TestAnalyticsStorage:
    """Test analytics storage functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_analytics.db"
        self.storage = AnalyticsStorage(self.db_path, retention_days=30)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test storage initialization."""
        assert self.storage.db_path == self.db_path
        assert self.storage.retention_days == 30
        assert self.db_path.exists()

    def test_database_schema_creation(self):
        """Test that database schema is created correctly."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name IN ('analytics_events', 'analytics_sessions')
            """)
            tables = [row[0] for row in cursor.fetchall()]

            assert "analytics_events" in tables
            assert "analytics_sessions" in tables

            # Check indexes exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='index' AND name LIKE 'idx_events_%'
            """)
            indexes = [row[0] for row in cursor.fetchall()]

            assert len(indexes) >= 4  # Should have at least 4 indexes

    def test_store_event_success(self):
        """Test successful event storage."""
        event = AnalyticsEvent(
            event_type="page_view",
            timestamp=datetime.now(),
            session_id="test-session",
            page_path="/test-page",
            user_agent_hash="abc123",
            duration_ms=5000,
            metadata={"test": "data"}
        )

        result = self.storage.store_event(event)
        assert result is True

        # Verify event was stored
        events = self.storage.get_events(limit=1)
        assert len(events) == 1
        assert events[0].event_type == "page_view"

    def test_store_event_with_none_values(self):
        """Test storing event with None values."""
        event = AnalyticsEvent(
            event_type="page_view",
            timestamp=datetime.now(),
            session_id="test-session",
            page_path="/test-page"
        )

        result = self.storage.store_event(event)
        assert result is True

        events = self.storage.get_events(limit=1)
        assert len(events) == 1
        assert events[0].user_agent_hash is None
        assert events[0].duration_ms is None
        assert events[0].metadata is None

    def test_store_multiple_events_updates_session(self):
        """Test that storing multiple events updates session data."""
        session_id = "test-session"
        events = [
            AnalyticsEvent(
                event_type="page_view",
                timestamp=datetime.now(),
                session_id=session_id,
                page_path="/page1",
                duration_ms=1000
            ),
            AnalyticsEvent(
                event_type="page_view",
                timestamp=datetime.now() + timedelta(seconds=30),
                session_id=session_id,
                page_path="/page2",
                duration_ms=2000
            )
        ]

        for event in events:
            self.storage.store_event(event)

        # Check session was updated
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT page_count, total_duration_ms FROM analytics_sessions WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()

            assert row is not None
            assert row[0] == 2  # page_count
            assert row[1] == 3000  # total_duration_ms

    def test_get_events_no_filters(self):
        """Test retrieving events without filters."""
        events = [
            AnalyticsEvent("page_view", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("search", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("page_view", datetime.now(), "session2", "/page2")
        ]

        for event in events:
            self.storage.store_event(event)

        retrieved = self.storage.get_events()
        assert len(retrieved) == 3

    def test_get_events_with_date_filter(self):
        """Test retrieving events with date filters."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time - timedelta(days=2), "session1", "/page1"),
            AnalyticsEvent("page_view", base_time, "session2", "/page2"),
            AnalyticsEvent("page_view", base_time + timedelta(days=2), "session3", "/page3")
        ]

        for event in events:
            self.storage.store_event(event)

        # Filter for events from yesterday to tomorrow
        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        retrieved = self.storage.get_events(start_date=start_date, end_date=end_date)
        assert len(retrieved) == 1
        assert retrieved[0].session_id == "session2"

    def test_get_events_with_type_filter(self):
        """Test retrieving events with event type filter."""
        events = [
            AnalyticsEvent("page_view", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("search", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("error", datetime.now(), "session2", "/page2")
        ]

        for event in events:
            self.storage.store_event(event)

        retrieved = self.storage.get_events(event_type="page_view")
        assert len(retrieved) == 1
        assert retrieved[0].event_type == "page_view"

    def test_get_events_with_session_filter(self):
        """Test retrieving events with session ID filter."""
        events = [
            AnalyticsEvent("page_view", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("search", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("page_view", datetime.now(), "session2", "/page2")
        ]

        for event in events:
            self.storage.store_event(event)

        retrieved = self.storage.get_events(session_id="session1")
        assert len(retrieved) == 2
        assert all(event.session_id == "session1" for event in retrieved)

    def test_get_events_with_limit(self):
        """Test retrieving events with limit."""
        events = [
            AnalyticsEvent("page_view", datetime.now(), f"session{i}", f"/page{i}")
            for i in range(10)
        ]

        for event in events:
            self.storage.store_event(event)

        retrieved = self.storage.get_events(limit=5)
        assert len(retrieved) == 5

    def test_get_events_ordered_by_timestamp_desc(self):
        """Test that events are returned in descending timestamp order."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1"),
            AnalyticsEvent("page_view", base_time + timedelta(minutes=1), "session2", "/page2"),
            AnalyticsEvent("page_view", base_time + timedelta(minutes=2), "session3", "/page3")
        ]

        for event in events:
            self.storage.store_event(event)

        retrieved = self.storage.get_events()

        # Should be ordered by timestamp DESC (newest first)
        assert retrieved[0].session_id == "session3"
        assert retrieved[1].session_id == "session2"
        assert retrieved[2].session_id == "session1"

    def test_get_events_database_error(self):
        """Test handling of database errors during event retrieval."""
        with patch.object(self.storage, '_get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database error")

            result = self.storage.get_events()
            assert result == []

    def test_get_stats_success(self):
        """Test successful statistics retrieval."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time, "session1", "/page1", duration_ms=1000),
            AnalyticsEvent("page_view", base_time, "session1", "/page2", duration_ms=2000),
            AnalyticsEvent("search", base_time, "session2", "/page1", metadata={"query": "test"}),
            AnalyticsEvent("error", base_time, "session2", "/page1")
        ]

        for event in events:
            self.storage.store_event(event)

        stats = self.storage.get_stats()
        assert stats is not None
        assert stats.total_events == 4
        assert stats.unique_sessions == 2
        assert stats.total_page_views == 2
        assert stats.error_rate == 25.0  # 1 error out of 4 events

    def test_get_stats_with_date_range(self):
        """Test statistics with date range filter."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time - timedelta(days=10), "session1", "/page1"),
            AnalyticsEvent("page_view", base_time, "session2", "/page2"),
            AnalyticsEvent("page_view", base_time + timedelta(days=10), "session3", "/page3")
        ]

        for event in events:
            self.storage.store_event(event)

        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        stats = self.storage.get_stats(start_date, end_date)
        assert stats is not None
        assert stats.total_events == 1
        assert stats.total_page_views == 1

    def test_get_stats_empty_database(self):
        """Test statistics retrieval with empty database."""
        stats = self.storage.get_stats()
        assert stats is not None
        assert stats.total_events == 0
        assert stats.unique_sessions == 0
        assert stats.total_page_views == 0

    def test_get_stats_database_error(self):
        """Test handling of database errors during stats retrieval."""
        with patch.object(self.storage, '_get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database error")

            result = self.storage.get_stats()
            assert result is None

    def test_cleanup_old_data(self):
        """Test cleanup of old data based on retention policy."""
        base_time = datetime.now()
        cutoff_days = self.storage.retention_days

        events = [
            AnalyticsEvent("page_view", base_time - timedelta(days=cutoff_days + 1), "session1", "/page1"),
            AnalyticsEvent("page_view", base_time - timedelta(days=cutoff_days - 1), "session2", "/page2"),
            AnalyticsEvent("page_view", base_time, "session3", "/page3")
        ]

        for event in events:
            self.storage.store_event(event)

        # Verify all events are stored
        assert len(self.storage.get_events()) == 3

        # Run cleanup
        deleted_count = self.storage.cleanup_old_data()
        assert deleted_count > 0

        # Verify old event was removed
        remaining_events = self.storage.get_events()
        assert len(remaining_events) == 2

        # Old event should be gone
        session_ids = [event.session_id for event in remaining_events]
        assert "session1" not in session_ids
        assert "session2" in session_ids
        assert "session3" in session_ids

    def test_cleanup_old_data_no_old_data(self):
        """Test cleanup when no old data exists."""
        event = AnalyticsEvent("page_view", datetime.now(), "session1", "/page1")
        self.storage.store_event(event)

        deleted_count = self.storage.cleanup_old_data()
        assert deleted_count == 0

        # Event should still be there
        events = self.storage.get_events()
        assert len(events) == 1

    def test_cleanup_old_data_database_error(self):
        """Test handling of database errors during cleanup."""
        with patch.object(self.storage, '_get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database error")

            result = self.storage.cleanup_old_data()
            assert result == 0

    def test_get_database_info(self):
        """Test database info retrieval."""
        # Store some events
        event = AnalyticsEvent("page_view", datetime.now(), "session1", "/page1")
        self.storage.store_event(event)

        info = self.storage.get_database_info()

        assert "database_path" in info
        assert "database_size_bytes" in info
        assert "event_count" in info
        assert "session_count" in info
        assert "retention_days" in info
        assert info["event_count"] == 1
        assert info["session_count"] == 1
        assert info["retention_days"] == 30

    def test_get_database_info_error(self):
        """Test database info retrieval with error."""
        with patch.object(self.storage, '_get_connection') as mock_conn:
            mock_conn.side_effect = Exception("Database error")

            info = self.storage.get_database_info()
            assert "error" in info
            assert "Database error" in info["error"]

    def test_hash_user_agent(self):
        """Test user agent hashing."""
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        hash_result = self.storage.hash_user_agent(user_agent)

        assert len(hash_result) == 16
        assert hash_result.isalnum()

        # Same input should produce same hash
        hash_result2 = self.storage.hash_user_agent(user_agent)
        assert hash_result == hash_result2

        # Different input should produce different hash
        hash_result3 = self.storage.hash_user_agent("Different user agent")
        assert hash_result != hash_result3

    def test_hash_user_agent_empty(self):
        """Test user agent hashing with empty input."""
        assert self.storage.hash_user_agent("") == ""
        assert self.storage.hash_user_agent(None) == ""

    def test_export_data_success(self):
        """Test successful data export."""
        events = [
            AnalyticsEvent("page_view", datetime.now(), "session1", "/page1"),
            AnalyticsEvent("search", datetime.now(), "session1", "/page1", metadata={"query": "test"})
        ]

        for event in events:
            self.storage.store_event(event)

        export_path = Path(self.temp_dir) / "export.json"
        result = self.storage.export_data(export_path)

        assert result is True
        assert export_path.exists()

        # Verify exported data
        with open(export_path, encoding='utf-8') as f:
            data = json.load(f)

        assert "export_timestamp" in data
        assert "stats" in data
        assert "event_count" in data
        assert "events" in data
        assert data["event_count"] == 2
        assert len(data["events"]) == 2

    def test_export_data_with_date_filter(self):
        """Test data export with date filters."""
        base_time = datetime.now()
        events = [
            AnalyticsEvent("page_view", base_time - timedelta(days=2), "session1", "/page1"),
            AnalyticsEvent("page_view", base_time, "session2", "/page2"),
        ]

        for event in events:
            self.storage.store_event(event)

        export_path = Path(self.temp_dir) / "export_filtered.json"
        start_date = base_time - timedelta(days=1)
        end_date = base_time + timedelta(days=1)

        result = self.storage.export_data(export_path, start_date, end_date)
        assert result is True

        with open(export_path, encoding='utf-8') as f:
            data = json.load(f)

        assert data["event_count"] == 1  # Only one event in date range

    def test_export_data_file_error(self):
        """Test export data with file write error."""
        event = AnalyticsEvent("page_view", datetime.now(), "session1", "/page1")
        self.storage.store_event(event)

        # Try to export to non-existent directory
        export_path = Path("/non/existent/directory/export.json")
        result = self.storage.export_data(export_path)

        assert result is False

    def test_connection_timeout_handling(self):
        """Test database connection timeout handling."""
        # Create storage with very short timeout for testing
        storage = AnalyticsStorage(self.db_path, retention_days=30)

        # Store event should still work despite potential timeout
        event = AnalyticsEvent("page_view", datetime.now(), "session1", "/page1")
        result = storage.store_event(event)
        assert result is True

    def test_concurrent_access_safety(self):
        """Test thread safety of storage operations."""
        import threading
        import time

        results = []

        def store_events():
            for i in range(10):
                event = AnalyticsEvent(
                    "page_view",
                    datetime.now(),
                    f"session{threading.current_thread().ident}",
                    f"/page{i}"
                )
                result = self.storage.store_event(event)
                results.append(result)
                time.sleep(0.001)  # Small delay to test concurrency

        threads = [threading.Thread(target=store_events) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results)

        # Should have 30 events total (3 threads × 10 events each)
        events = self.storage.get_events(limit=50)
        assert len(events) == 30

    def test_malformed_json_metadata_handling(self):
        """Test handling of malformed JSON in metadata."""
        # This tests the robustness of the from_dict method with bad JSON
        data = {
            "event_type": "page_view",
            "timestamp": "2023-01-01T12:00:00",
            "session_id": "test-session",
            "page_path": "/test-page",
            "user_agent_hash": "abc123",
            "duration_ms": 5000,
            "metadata": "invalid json {"  # Malformed JSON
        }

        with pytest.raises(json.JSONDecodeError):
            AnalyticsEvent.from_dict(data)

    def test_very_large_metadata_handling(self):
        """Test handling of very large metadata."""
        large_metadata = {"data": "x" * 100000}  # 100KB of data

        event = AnalyticsEvent(
            event_type="page_view",
            timestamp=datetime.now(),
            session_id="test-session",
            page_path="/test-page",
            metadata=large_metadata
        )

        result = self.storage.store_event(event)
        assert result is True

        retrieved = self.storage.get_events(limit=1)
        assert len(retrieved) == 1
        assert retrieved[0].metadata == large_metadata
