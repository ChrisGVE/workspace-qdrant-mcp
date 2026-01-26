"""Unit tests for analytics collector system."""

import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add docs framework to path for testing (if present)
docs_framework_path = os.path.join(os.path.dirname(__file__), '../../docs/framework')
sys.path.insert(0, docs_framework_path)

try:
    from analytics.collector import AnalyticsCollector, EventType, TrackingConfig
    from analytics.storage import AnalyticsEvent, AnalyticsStorage
except ModuleNotFoundError:
    pytest.skip("analytics framework not available", allow_module_level=True)


class TestTrackingConfig:
    """Test tracking configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TrackingConfig()

        assert config.enabled is True
        assert config.respect_do_not_track is True
        assert config.anonymize_ips is True
        assert config.batch_size == 10
        assert config.flush_interval_seconds == 30
        assert config.max_queue_size == 1000
        assert config.track_performance is True
        assert config.track_errors is True
        assert config.excluded_paths == ['/health', '/metrics', '/favicon.ico']

    def test_custom_config(self):
        """Test custom configuration values."""
        custom_paths = ['/admin', '/private']
        config = TrackingConfig(
            enabled=False,
            batch_size=5,
            excluded_paths=custom_paths
        )

        assert config.enabled is False
        assert config.batch_size == 5
        assert config.excluded_paths == custom_paths

    def test_excluded_paths_initialization(self):
        """Test that excluded_paths is properly initialized."""
        config = TrackingConfig(excluded_paths=None)
        assert config.excluded_paths == ['/health', '/metrics', '/favicon.ico']

        config = TrackingConfig(excluded_paths=[])
        assert config.excluded_paths == []


class TestAnalyticsCollector:
    """Test analytics collector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_analytics.db"
        self.storage = AnalyticsStorage(self.db_path, retention_days=30)
        self.config = TrackingConfig(
            batch_size=2,  # Small batch for testing
            flush_interval_seconds=1  # Fast flush for testing
        )
        self.collector = AnalyticsCollector(self.storage, self.config)

    def teardown_method(self):
        """Clean up test fixtures."""
        self.collector.shutdown()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector.storage == self.storage
        assert self.collector.config == self.config
        assert self.collector._event_queue == []
        assert self.collector._shutdown is False
        assert self.collector._flush_task is not None
        assert self.collector._flush_task.is_alive()

    def test_initialization_disabled(self):
        """Test collector initialization when disabled."""
        config = TrackingConfig(enabled=False)
        collector = AnalyticsCollector(self.storage, config)

        assert collector._flush_task is None

    def test_create_session_id(self):
        """Test session ID generation."""
        session_id = self.collector.create_session_id()

        assert isinstance(session_id, str)
        assert len(session_id) > 0

        # Should be unique
        session_id2 = self.collector.create_session_id()
        assert session_id != session_id2

    def test_is_enabled(self):
        """Test enabled status check."""
        assert self.collector.is_enabled() is True

        config = TrackingConfig(enabled=False)
        collector = AnalyticsCollector(self.storage, config)
        assert collector.is_enabled() is False

    def test_should_track_enabled(self):
        """Test should_track method when enabled."""
        assert self.collector._should_track("/test-page") is True
        assert self.collector._should_track("/another-page") is True

    def test_should_track_disabled(self):
        """Test should_track method when disabled."""
        config = TrackingConfig(enabled=False)
        collector = AnalyticsCollector(self.storage, config)

        assert collector._should_track("/test-page") is False

    def test_should_track_excluded_paths(self):
        """Test should_track with excluded paths."""
        assert self.collector._should_track("/health") is False
        assert self.collector._should_track("/metrics") is False
        assert self.collector._should_track("/favicon.ico") is False
        assert self.collector._should_track("/health/status") is False  # Starts with /health

    def test_track_page_view_success(self):
        """Test successful page view tracking."""
        result = self.collector.track_page_view(
            page_path="/test-page",
            session_id="test-session",
            user_agent="Mozilla/5.0...",
            referrer="https://google.com",
            duration_ms=5000,
            viewport_size="1920x1080"
        )

        assert result is True
        # Events are auto-flushed when batch size (2) is reached, but we only have 1
        assert self.collector.get_queue_size() == 1

    def test_track_page_view_excluded_path(self):
        """Test page view tracking for excluded path."""
        result = self.collector.track_page_view(
            page_path="/health",
            session_id="test-session"
        )

        assert result is False
        assert self.collector.get_queue_size() == 0

    def test_track_search_success(self):
        """Test successful search tracking."""
        result = self.collector.track_search(
            query="test search",
            session_id="test-session",
            page_path="/search",
            results_count=10,
            duration_ms=250,
            selected_result_index=2
        )

        assert result is True
        assert self.collector.get_queue_size() == 1

    def test_track_search_query_sanitization(self):
        """Test search query sanitization."""
        # Mock the sanitization method to verify it's called
        with patch.object(self.collector, '_sanitize_search_query') as mock_sanitize:
            mock_sanitize.return_value = "sanitized query"

            self.collector.track_search(
                query="test@example.com search",
                session_id="test-session",
                page_path="/search",
                results_count=5
            )

            mock_sanitize.assert_called_once_with("test@example.com search")

    def test_track_code_execution_success(self):
        """Test successful code execution tracking."""
        result = self.collector.track_code_execution(
            session_id="test-session",
            page_path="/docs/example",
            language="python",
            success=True,
            execution_time_ms=150,
            code_length=50
        )

        assert result is True
        assert self.collector.get_queue_size() == 1

    def test_track_code_execution_failure(self):
        """Test code execution tracking with failure."""
        result = self.collector.track_code_execution(
            session_id="test-session",
            page_path="/docs/example",
            language="python",
            success=False,
            execution_time_ms=100,
            code_length=30,
            error_type="SyntaxError"
        )

        assert result is True

        # Wait for flush and verify metadata
        time.sleep(0.1)
        self.collector.flush()
        events = self.storage.get_events(limit=1)
        assert len(events) == 1
        assert events[0].metadata["success"] is False
        assert events[0].metadata["error_type"] == "SyntaxError"

    def test_track_error_success(self):
        """Test successful error tracking."""
        result = self.collector.track_error(
            session_id="test-session",
            page_path="/test-page",
            error_type="JavaScript",
            error_message="ReferenceError: x is not defined",
            stack_trace="at line 10 in script.js",
            user_agent="Mozilla/5.0..."
        )

        assert result is True
        assert self.collector.get_queue_size() == 1

    def test_track_error_disabled(self):
        """Test error tracking when disabled."""
        config = TrackingConfig(track_errors=False)
        collector = AnalyticsCollector(self.storage, config)

        result = collector.track_error(
            session_id="test-session",
            page_path="/test-page",
            error_type="JavaScript",
            error_message="Error occurred"
        )

        assert result is False

    def test_track_interaction_success(self):
        """Test successful interaction tracking."""
        result = self.collector.track_interaction(
            session_id="test-session",
            page_path="/docs/guide",
            interaction_type="click",
            element_id="search-button",
            duration_ms=200,
            metadata={"button_type": "primary"}
        )

        assert result is True
        assert self.collector.get_queue_size() == 1

    def test_performance_measurement_lifecycle(self):
        """Test complete performance measurement lifecycle."""
        session_id = "test-session"
        measurement_name = "page-load"

        # Start measurement
        result1 = self.collector.start_performance_measurement(session_id, measurement_name)
        assert result1 is True

        # Wait a bit
        time.sleep(0.01)

        # End measurement
        result2 = self.collector.end_performance_measurement(
            session_id,
            measurement_name,
            "/test-page",
            metadata={"page_type": "documentation"}
        )
        assert result2 is True
        assert self.collector.get_queue_size() == 1

        # Measurement should be cleaned up
        assert session_id not in self.collector._performance_marks

    def test_performance_measurement_disabled(self):
        """Test performance measurement when disabled."""
        config = TrackingConfig(track_performance=False)
        collector = AnalyticsCollector(self.storage, config)

        result = collector.start_performance_measurement("session", "test")
        assert result is False

        result = collector.end_performance_measurement("session", "test", "/page")
        assert result is False

    def test_performance_measurement_without_start(self):
        """Test ending measurement without starting it."""
        result = self.collector.end_performance_measurement(
            "session",
            "nonexistent-measurement",
            "/test-page"
        )
        assert result is False

    def test_queue_event_batch_flush(self):
        """Test automatic batch flushing when queue reaches batch size."""
        # Mock storage to verify flush is called
        mock_storage = Mock()
        mock_storage.store_event = Mock(return_value=True)
        collector = AnalyticsCollector(mock_storage, self.config)

        # Add events up to batch size
        for i in range(self.config.batch_size):
            collector.track_page_view(f"/page{i}", "session", duration_ms=1000)

        # Should have triggered flush
        assert mock_storage.store_event.call_count == self.config.batch_size

    def test_queue_event_max_size_overflow(self):
        """Test queue overflow handling."""
        config = TrackingConfig(max_queue_size=5, batch_size=10)  # Won't auto-flush
        mock_storage = Mock()
        mock_storage.store_event = Mock(return_value=True)
        collector = AnalyticsCollector(mock_storage, config)

        # Fill queue beyond max size
        for i in range(10):
            collector.track_page_view(f"/page{i}", "session")

        # Queue should be trimmed
        assert collector.get_queue_size() <= config.max_queue_size

    def test_queue_event_failure(self):
        """Test handling of queue event failure."""
        with patch.object(self.collector, '_queue_lock') as mock_lock:
            mock_lock.__enter__.side_effect = Exception("Lock error")

            result = self.collector.track_page_view("/page", "session")
            assert result is False

    def test_flush_events_success(self):
        """Test successful event flushing."""
        # Add events to queue
        self.collector.track_page_view("/page1", "session")
        self.collector.track_page_view("/page2", "session")

        initial_queue_size = self.collector.get_queue_size()
        assert initial_queue_size > 0

        # Flush events
        self.collector._flush_events()

        # Queue should be empty
        assert self.collector.get_queue_size() == 0

        # Events should be in storage
        events = self.storage.get_events()
        assert len(events) == initial_queue_size

    def test_flush_events_partial_failure(self):
        """Test event flushing with partial storage failures."""
        mock_storage = Mock()
        # First call succeeds, second fails
        mock_storage.store_event = Mock(side_effect=[True, False])
        collector = AnalyticsCollector(mock_storage, self.config)

        collector.track_page_view("/page1", "session")
        collector.track_page_view("/page2", "session")

        # Mock warning log to verify partial failure is logged
        with patch('analytics.collector.logger') as mock_logger:
            collector._flush_events()
            mock_logger.warning.assert_called()

    def test_flush_events_empty_queue(self):
        """Test flushing when queue is empty."""
        # Should not raise error
        self.collector._flush_events()
        assert self.collector.get_queue_size() == 0

    def test_sanitize_search_query_email_removal(self):
        """Test search query email sanitization."""
        query = "Contact user@example.com for help"
        result = self.collector._sanitize_search_query(query)

        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_sanitize_search_query_phone_removal(self):
        """Test search query phone number sanitization."""
        query = "Call 555-123-4567 for support"
        result = self.collector._sanitize_search_query(query)

        assert "[PHONE]" in result
        assert "555-123-4567" not in result

    def test_sanitize_search_query_number_removal(self):
        """Test search query long number sanitization."""
        query = "User ID 123456789 not found"
        result = self.collector._sanitize_search_query(query)

        assert "[NUMBER]" in result
        assert "123456789" not in result

    def test_sanitize_search_query_length_limit(self):
        """Test search query length limiting."""
        long_query = "x" * 300
        result = self.collector._sanitize_search_query(long_query)

        assert len(result) == 200

    def test_sanitize_error_message_path_removal(self):
        """Test error message path sanitization."""
        message = "File not found: /Users/johndoe/documents/file.txt"
        result = self.collector._sanitize_error_message(message)

        assert "/Users/[USER]" in result
        assert "johndoe" not in result

    def test_sanitize_error_message_windows_path(self):
        """Test error message Windows path sanitization."""
        message = "Access denied: C:\\Users\\janedoe\\Desktop\\file.txt"
        result = self.collector._sanitize_error_message(message)

        assert "C:\\Users\\[USER]" in result
        assert "janedoe" not in result

    def test_sanitize_error_message_length_limit(self):
        """Test error message length limiting."""
        long_message = "x" * 600
        result = self.collector._sanitize_error_message(long_message)

        assert len(result) == 500

    def test_anonymize_referrer_success(self):
        """Test referrer anonymization."""
        referrer = "https://www.google.com/search?q=test"
        result = self.collector._anonymize_referrer(referrer)

        assert result == "www.google.com"

    def test_anonymize_referrer_invalid_url(self):
        """Test referrer anonymization with invalid URL."""
        referrer = "not-a-url"
        result = self.collector._anonymize_referrer(referrer)

        assert result == "unknown"

    def test_anonymize_referrer_exception(self):
        """Test referrer anonymization with exception."""
        with patch('analytics.collector.urlparse') as mock_parse:
            mock_parse.side_effect = Exception("Parse error")

            result = self.collector._anonymize_referrer("http://example.com")
            assert result == "unknown"

    def test_flush_method(self):
        """Test manual flush method."""
        self.collector.track_page_view("/page", "session")
        initial_queue_size = self.collector.get_queue_size()
        assert initial_queue_size > 0

        self.collector.flush()

        assert self.collector.get_queue_size() == 0

    def test_shutdown_method(self):
        """Test collector shutdown."""
        # Add some events
        self.collector.track_page_view("/page", "session")
        assert self.collector.get_queue_size() > 0

        # Shutdown should flush events and stop flush task
        self.collector.shutdown()

        assert self.collector._shutdown is True
        assert self.collector.get_queue_size() == 0

        # Flush task should be terminated
        if self.collector._flush_task:
            assert not self.collector._flush_task.is_alive()

    def test_get_queue_size(self):
        """Test queue size getter."""
        assert self.collector.get_queue_size() == 0

        self.collector.track_page_view("/page", "session")
        assert self.collector.get_queue_size() == 1

        self.collector.track_search("query", "session", "/page", 5)
        # Should have 2 events, which triggers auto-flush due to batch_size=2
        assert self.collector.get_queue_size() == 0  # Queue should be empty after flush

    def test_background_flush_task(self):
        """Test that background flush task works."""
        # Create collector with very fast flush interval
        config = TrackingConfig(flush_interval_seconds=0.1, batch_size=10)
        collector = AnalyticsCollector(self.storage, config)

        try:
            # Add event
            collector.track_page_view("/page", "session")
            assert collector.get_queue_size() == 1

            # Wait for background flush
            time.sleep(0.2)

            # Event should be flushed
            assert collector.get_queue_size() == 0

        finally:
            collector.shutdown()

    def test_thread_safety(self):
        """Test thread safety of collector operations."""
        results = []

        def track_events():
            for i in range(10):
                result = self.collector.track_page_view(
                    f"/page{i}",
                    f"session{threading.current_thread().ident}",
                    duration_ms=100
                )
                results.append(result)

        threads = [threading.Thread(target=track_events) for _ in range(3)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results)
        assert len(results) == 30  # 3 threads × 10 events each

    def test_performance_marks_cleanup(self):
        """Test that performance marks are properly cleaned up."""
        session_id = "test-session"

        # Start multiple measurements
        self.collector.start_performance_measurement(session_id, "measure1")
        self.collector.start_performance_measurement(session_id, "measure2")

        assert session_id in self.collector._performance_marks
        assert len(self.collector._performance_marks[session_id]) == 2

        # End one measurement
        self.collector.end_performance_measurement(session_id, "measure1", "/page")

        # Should still have session but with one less measurement
        assert session_id in self.collector._performance_marks
        assert len(self.collector._performance_marks[session_id]) == 1

        # End last measurement
        self.collector.end_performance_measurement(session_id, "measure2", "/page")

        # Session should be completely cleaned up
        assert session_id not in self.collector._performance_marks

    def test_metadata_edge_cases(self):
        """Test handling of edge cases in metadata."""
        # None metadata
        result = self.collector.track_page_view("/page", "session", duration_ms=1000)
        assert result is True

        # Empty metadata
        result = self.collector.track_interaction("session", "/page", "click", metadata={})
        assert result is True

        # Complex nested metadata
        complex_metadata = {
            "nested": {"level1": {"level2": "value"}},
            "array": [1, 2, 3],
            "unicode": "🚀📊",
            "null_value": None
        }
        result = self.collector.track_interaction("session", "/page", "click", metadata=complex_metadata)
        assert result is True

    def test_very_long_page_paths(self):
        """Test handling of very long page paths."""
        long_path = "/very/long/path/" + ("segment/" * 100) + "page.html"

        result = self.collector.track_page_view(long_path, "session")
        assert result is True

        self.collector.flush()
        events = self.storage.get_events(limit=1)
        assert len(events) == 1
        assert events[0].page_path == long_path

    def test_unicode_handling(self):
        """Test handling of unicode characters in various fields."""
        # Unicode in search query
        result = self.collector.track_search(
            "🔍 поиск тест search",
            "session",
            "/search",
            5
        )
        assert result is True

        # Unicode in error message
        result = self.collector.track_error(
            "session",
            "/page",
            "UnicodeError",
            "Unicode error: 测试错误 тест"
        )
        assert result is True

        # Should auto-flush since we have 2 events and batch_size=2
        events = self.storage.get_events(limit=2)
        assert len(events) == 2

    def test_extreme_duration_values(self):
        """Test handling of extreme duration values."""
        # Very large duration
        result = self.collector.track_page_view("/page1", "session", duration_ms=999999999)
        assert result is True

        # Zero duration
        result = self.collector.track_page_view("/page2", "session", duration_ms=0)
        assert result is True

        # This should auto-flush since we have batch_size=2
        events = self.storage.get_events(limit=3)
        assert len(events) == 2

        # Negative duration (should still be stored as-is)
        result = self.collector.track_page_view("/page3", "session", duration_ms=-100)
        assert result is True

        self.collector.flush()  # Flush the remaining event
        events = self.storage.get_events(limit=3)
        assert len(events) == 3
