"""Analytics event collector for tracking documentation usage patterns."""

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass
import hashlib
import threading
import time

from .storage import AnalyticsStorage, AnalyticsEvent


logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of analytics events that can be tracked."""
    PAGE_VIEW = "page_view"
    SEARCH = "search"
    CODE_EXECUTION = "code_execution"
    DOWNLOAD = "download"
    ERROR = "error"
    INTERACTION = "interaction"
    PERFORMANCE = "performance"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


@dataclass
class TrackingConfig:
    """Configuration for analytics tracking."""
    enabled: bool = True
    respect_do_not_track: bool = True
    anonymize_ips: bool = True
    batch_size: int = 10
    flush_interval_seconds: int = 30
    max_queue_size: int = 1000
    track_performance: bool = True
    track_errors: bool = True
    excluded_paths: List[str] = None

    def __post_init__(self):
        if self.excluded_paths is None:
            self.excluded_paths = ['/health', '/metrics', '/favicon.ico']


class AnalyticsCollector:
    """Privacy-focused analytics collector for documentation usage tracking."""

    def __init__(self, storage: AnalyticsStorage, config: TrackingConfig = None):
        """Initialize analytics collector.

        Args:
            storage: Analytics storage backend
            config: Tracking configuration
        """
        self.storage = storage
        self.config = config or TrackingConfig()
        self._event_queue: List[AnalyticsEvent] = []
        self._queue_lock = threading.RLock()
        self._flush_task = None
        self._shutdown = False
        self._session_cache: Dict[str, Dict[str, Any]] = {}
        self._performance_marks: Dict[str, Dict[str, float]] = {}

        if self.config.enabled:
            self._start_flush_task()

    def _start_flush_task(self):
        """Start background task to flush events to storage."""
        def flush_loop():
            while not self._shutdown:
                try:
                    time.sleep(self.config.flush_interval_seconds)
                    self._flush_events()
                except Exception as e:
                    logger.error(f"Error in flush loop: {e}")

        self._flush_task = threading.Thread(target=flush_loop, daemon=True)
        self._flush_task.start()

    def track_page_view(
        self,
        page_path: str,
        session_id: str,
        user_agent: Optional[str] = None,
        referrer: Optional[str] = None,
        duration_ms: Optional[int] = None,
        viewport_size: Optional[str] = None
    ) -> bool:
        """Track a page view event.

        Args:
            page_path: Path of the viewed page
            session_id: User session identifier
            user_agent: User agent string (will be hashed)
            referrer: Referrer URL (will be anonymized)
            duration_ms: Time spent on page in milliseconds
            viewport_size: Browser viewport size (e.g., "1920x1080")

        Returns:
            True if event was queued successfully
        """
        if not self._should_track(page_path):
            return False

        metadata = {}
        if referrer:
            metadata['referrer_domain'] = self._anonymize_referrer(referrer)
        if viewport_size:
            metadata['viewport_size'] = viewport_size

        event = AnalyticsEvent(
            event_type=EventType.PAGE_VIEW.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            user_agent_hash=self.storage.hash_user_agent(user_agent) if user_agent else None,
            duration_ms=duration_ms,
            metadata=metadata
        )

        return self._queue_event(event)

    def track_search(
        self,
        query: str,
        session_id: str,
        page_path: str,
        results_count: int,
        duration_ms: Optional[int] = None,
        selected_result_index: Optional[int] = None
    ) -> bool:
        """Track a search event.

        Args:
            query: Search query (will be stored as-is for improving docs)
            session_id: User session identifier
            page_path: Page where search was performed
            results_count: Number of search results returned
            duration_ms: Search duration in milliseconds
            selected_result_index: Index of selected result (if any)

        Returns:
            True if event was queued successfully
        """
        if not self._should_track(page_path):
            return False

        # Sanitize query to remove potential personal information
        sanitized_query = self._sanitize_search_query(query)

        metadata = {
            'query': sanitized_query,
            'results_count': results_count
        }

        if selected_result_index is not None:
            metadata['selected_result_index'] = selected_result_index

        event = AnalyticsEvent(
            event_type=EventType.SEARCH.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            duration_ms=duration_ms,
            metadata=metadata
        )

        return self._queue_event(event)

    def track_code_execution(
        self,
        session_id: str,
        page_path: str,
        language: str,
        success: bool,
        execution_time_ms: int,
        code_length: int,
        error_type: Optional[str] = None
    ) -> bool:
        """Track code execution in interactive examples.

        Args:
            session_id: User session identifier
            page_path: Page where code was executed
            language: Programming language
            success: Whether execution was successful
            execution_time_ms: Execution time in milliseconds
            code_length: Length of executed code in characters
            error_type: Type of error if execution failed

        Returns:
            True if event was queued successfully
        """
        if not self._should_track(page_path):
            return False

        metadata = {
            'language': language,
            'success': success,
            'code_length': code_length
        }

        if error_type:
            metadata['error_type'] = error_type

        event = AnalyticsEvent(
            event_type=EventType.CODE_EXECUTION.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            duration_ms=execution_time_ms,
            metadata=metadata
        )

        return self._queue_event(event)

    def track_error(
        self,
        session_id: str,
        page_path: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """Track an error event.

        Args:
            session_id: User session identifier
            page_path: Page where error occurred
            error_type: Type/category of error
            error_message: Error message (will be sanitized)
            stack_trace: Stack trace (will be sanitized)
            user_agent: User agent string

        Returns:
            True if event was queued successfully
        """
        if not self.config.track_errors or not self._should_track(page_path):
            return False

        # Sanitize error information to remove potential personal data
        metadata = {
            'error_type': error_type,
            'error_message': self._sanitize_error_message(error_message)
        }

        if stack_trace:
            metadata['stack_trace_hash'] = hashlib.sha256(
                stack_trace.encode('utf-8')
            ).hexdigest()[:16]

        event = AnalyticsEvent(
            event_type=EventType.ERROR.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            user_agent_hash=self.storage.hash_user_agent(user_agent) if user_agent else None,
            metadata=metadata
        )

        return self._queue_event(event)

    def track_interaction(
        self,
        session_id: str,
        page_path: str,
        interaction_type: str,
        element_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Track user interaction events.

        Args:
            session_id: User session identifier
            page_path: Page where interaction occurred
            interaction_type: Type of interaction (click, scroll, etc.)
            element_id: ID of interacted element
            duration_ms: Duration of interaction
            metadata: Additional interaction metadata

        Returns:
            True if event was queued successfully
        """
        if not self._should_track(page_path):
            return False

        event_metadata = {
            'interaction_type': interaction_type
        }

        if element_id:
            event_metadata['element_id'] = element_id

        if metadata:
            event_metadata.update(metadata)

        event = AnalyticsEvent(
            event_type=EventType.INTERACTION.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            duration_ms=duration_ms,
            metadata=event_metadata
        )

        return self._queue_event(event)

    def start_performance_measurement(
        self,
        session_id: str,
        measurement_name: str
    ) -> bool:
        """Start a performance measurement.

        Args:
            session_id: User session identifier
            measurement_name: Name of the measurement

        Returns:
            True if measurement started successfully
        """
        if not self.config.track_performance:
            return False

        if session_id not in self._performance_marks:
            self._performance_marks[session_id] = {}

        self._performance_marks[session_id][measurement_name] = time.time() * 1000

        return True

    def end_performance_measurement(
        self,
        session_id: str,
        measurement_name: str,
        page_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """End a performance measurement and track the result.

        Args:
            session_id: User session identifier
            measurement_name: Name of the measurement
            page_path: Page where measurement occurred
            metadata: Additional measurement metadata

        Returns:
            True if measurement was tracked successfully
        """
        if not self.config.track_performance or not self._should_track(page_path):
            return False

        if (session_id not in self._performance_marks or
            measurement_name not in self._performance_marks[session_id]):
            return False

        start_time = self._performance_marks[session_id][measurement_name]
        end_time = time.time() * 1000
        duration_ms = int(end_time - start_time)

        # Clean up the mark
        del self._performance_marks[session_id][measurement_name]
        if not self._performance_marks[session_id]:
            del self._performance_marks[session_id]

        event_metadata = {
            'measurement_name': measurement_name,
            'duration_ms': duration_ms
        }

        if metadata:
            event_metadata.update(metadata)

        event = AnalyticsEvent(
            event_type=EventType.PERFORMANCE.value,
            timestamp=datetime.now(),
            session_id=session_id,
            page_path=page_path,
            duration_ms=duration_ms,
            metadata=event_metadata
        )

        return self._queue_event(event)

    def create_session_id(self) -> str:
        """Create a new session ID.

        Returns:
            Unique session identifier
        """
        return str(uuid.uuid4())

    def _should_track(self, page_path: str) -> bool:
        """Check if a page path should be tracked.

        Args:
            page_path: Page path to check

        Returns:
            True if should track, False otherwise
        """
        if not self.config.enabled:
            return False

        # Check excluded paths
        for excluded_path in self.config.excluded_paths:
            if page_path.startswith(excluded_path):
                return False

        return True

    def _queue_event(self, event: AnalyticsEvent) -> bool:
        """Add event to the queue for batch processing.

        Args:
            event: Event to queue

        Returns:
            True if queued successfully
        """
        try:
            with self._queue_lock:
                if len(self._event_queue) >= self.config.max_queue_size:
                    # Queue is full, drop oldest events
                    self._event_queue = self._event_queue[-(self.config.max_queue_size // 2):]
                    logger.warning("Analytics queue full, dropped old events")

                self._event_queue.append(event)

                # Flush immediately if batch size reached
                if len(self._event_queue) >= self.config.batch_size:
                    self._flush_events()

                return True

        except Exception as e:
            logger.error(f"Failed to queue analytics event: {e}")
            return False

    def _flush_events(self):
        """Flush queued events to storage."""
        events_to_flush = []

        with self._queue_lock:
            if not self._event_queue:
                return

            events_to_flush = self._event_queue.copy()
            self._event_queue.clear()

        # Store events in storage
        success_count = 0
        for event in events_to_flush:
            if self.storage.store_event(event):
                success_count += 1

        if success_count < len(events_to_flush):
            logger.warning(f"Only {success_count}/{len(events_to_flush)} events stored successfully")

    def _sanitize_search_query(self, query: str) -> str:
        """Sanitize search query to remove personal information.

        Args:
            query: Raw search query

        Returns:
            Sanitized query
        """
        # Limit length
        query = query[:200]

        # Remove common patterns that might contain personal info
        # (email addresses, phone numbers, etc.)
        import re

        # Remove email addresses
        query = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', query)

        # Remove phone numbers (simple patterns)
        query = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', query)

        # Remove long sequences of numbers (might be IDs, SSNs, etc.)
        query = re.sub(r'\b\d{6,}\b', '[NUMBER]', query)

        return query.strip()

    def _sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information.

        Args:
            message: Raw error message

        Returns:
            Sanitized error message
        """
        # Limit length
        message = message[:500]

        # Remove file paths that might contain usernames
        import re
        message = re.sub(r'/Users/[^/\s]+', '/Users/[USER]', message)
        message = re.sub(r'C:\\Users\\[^\\s]+', 'C:\\Users\\[USER]', message)

        return message.strip()

    def _anonymize_referrer(self, referrer: str) -> str:
        """Extract domain from referrer for privacy.

        Args:
            referrer: Full referrer URL

        Returns:
            Domain only
        """
        try:
            from urllib.parse import urlparse
            parsed = urlparse(referrer)
            return parsed.netloc or 'unknown'
        except Exception:
            return 'unknown'

    def flush(self):
        """Force flush all queued events to storage."""
        self._flush_events()

    def shutdown(self):
        """Shutdown the collector and flush remaining events."""
        self._shutdown = True

        # Flush any remaining events
        self._flush_events()

        # Wait for flush task to complete
        if self._flush_task and self._flush_task.is_alive():
            self._flush_task.join(timeout=5)

    def get_queue_size(self) -> int:
        """Get current size of event queue.

        Returns:
            Number of events in queue
        """
        with self._queue_lock:
            return len(self._event_queue)

    def is_enabled(self) -> bool:
        """Check if analytics collection is enabled.

        Returns:
            True if enabled, False otherwise
        """
        return self.config.enabled