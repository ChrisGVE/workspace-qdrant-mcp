"""
Unit tests for ErrorMessageManager.

Tests cover:
- Error recording with automatic categorization
- Context preservation
- Error retrieval with filtering
- Acknowledgment workflow
- Statistics generation
- Edge cases and error handling
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.python.common.core.error_message_manager import (
    ErrorMessage,
    ErrorMessageManager,
    ErrorStatistics,
)
from src.python.common.core.error_categorization import (
    ErrorCategory,
    ErrorSeverity,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def db_with_schema(temp_db):
    """Create a database with enhanced error messages schema."""
    conn = sqlite3.connect(temp_db)

    # Read schema from file
    schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
    with open(schema_file, 'r') as f:
        schema_sql = f.read()

    # Create schema (but rename messages_enhanced to messages)
    conn.executescript(schema_sql)
    conn.execute("DROP TABLE IF EXISTS messages")
    conn.execute("ALTER TABLE messages_enhanced RENAME TO messages")
    conn.commit()
    conn.close()

    return temp_db


@pytest.fixture
async def manager(db_with_schema):
    """Create and initialize an ErrorMessageManager."""
    mgr = ErrorMessageManager(db_path=db_with_schema)
    await mgr.initialize()
    yield mgr
    await mgr.close()


class TestErrorMessage:
    """Test ErrorMessage dataclass."""

    def test_from_db_row(self, db_with_schema):
        """Test creating ErrorMessage from database row."""
        conn = sqlite3.connect(db_with_schema)
        conn.row_factory = sqlite3.Row

        # Insert test message
        conn.execute("""
            INSERT INTO messages (severity, category, message, context, acknowledged, retry_count)
            VALUES ('error', 'parse_error', 'Test error', '{"file_path": "/test.py"}', 0, 2)
        """)
        conn.commit()

        cursor = conn.execute("SELECT * FROM messages WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        error_msg = ErrorMessage.from_db_row(row)

        assert error_msg.id == 1
        assert error_msg.severity == ErrorSeverity.ERROR
        assert error_msg.category == ErrorCategory.PARSE_ERROR
        assert error_msg.message == "Test error"
        assert error_msg.context == {"file_path": "/test.py"}
        assert error_msg.acknowledged is False
        assert error_msg.retry_count == 2

    def test_to_dict(self):
        """Test converting ErrorMessage to dictionary."""
        now = datetime.now(timezone.utc)
        error_msg = ErrorMessage(
            id=1,
            timestamp=now,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.NETWORK,
            message="Connection failed",
            context={"url": "http://example.com"},
            acknowledged=True,
            acknowledged_at=now,
            acknowledged_by="admin",
            retry_count=3
        )

        result = error_msg.to_dict()

        assert result["id"] == 1
        assert result["severity"] == "error"
        assert result["category"] == "network"
        assert result["message"] == "Connection failed"
        assert result["context"] == {"url": "http://example.com"}
        assert result["acknowledged"] is True
        assert result["acknowledged_by"] == "admin"
        assert result["retry_count"] == 3

    def test_from_db_row_with_invalid_json(self, db_with_schema):
        """Test handling of invalid JSON in context."""
        conn = sqlite3.connect(db_with_schema)
        conn.row_factory = sqlite3.Row

        # Insert message with invalid JSON
        conn.execute("""
            INSERT INTO messages (severity, category, message, context)
            VALUES ('error', 'unknown', 'Test', 'invalid json')
        """)
        conn.commit()

        cursor = conn.execute("SELECT * FROM messages WHERE id = 1")
        row = cursor.fetchone()
        conn.close()

        error_msg = ErrorMessage.from_db_row(row)

        # Should wrap invalid JSON in raw field
        assert error_msg.context == {"raw": "invalid json"}


class TestErrorStatistics:
    """Test ErrorStatistics dataclass."""

    def test_to_dict(self):
        """Test converting ErrorStatistics to dictionary."""
        now = datetime.now(timezone.utc)
        stats = ErrorStatistics(
            total_count=100,
            by_severity={"error": 60, "warning": 30, "info": 10},
            by_category={"network": 40, "parse_error": 35, "unknown": 25},
            unacknowledged_count=25,
            last_error_at=now
        )

        result = stats.to_dict()

        assert result["total_count"] == 100
        assert result["by_severity"] == {"error": 60, "warning": 30, "info": 10}
        assert result["by_category"] == {"network": 40, "parse_error": 35, "unknown": 25}
        assert result["unacknowledged_count"] == 25
        assert result["last_error_at"] == now.isoformat()


class TestErrorMessageManager:
    """Test ErrorMessageManager functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, db_with_schema):
        """Test manager initialization."""
        manager = ErrorMessageManager(db_path=db_with_schema)
        await manager.initialize()

        assert manager._initialized is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_record_error_with_exception(self, manager):
        """Test recording error from exception."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            error_id = await manager.record_error(
                exception=e,
                context={
                    "file_path": "/path/to/file.py",
                    "collection": "test-project",
                    "tenant_id": "default"
                }
            )

        assert error_id > 0

        # Verify error was recorded
        error = await manager.get_error_by_id(error_id)
        assert error is not None
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.ERROR
        assert error.category == ErrorCategory.PARSE_ERROR  # ValueError maps to parse_error
        assert error.context["file_path"] == "/path/to/file.py"
        assert error.context["collection"] == "test-project"
        assert error.context["tenant_id"] == "default"

    @pytest.mark.asyncio
    async def test_record_error_with_message_override(self, manager):
        """Test recording error with message override."""
        error_id = await manager.record_error(
            message_override="Custom error message",
            context={"custom_field": "value"}
        )

        assert error_id > 0

        error = await manager.get_error_by_id(error_id)
        assert error.message == "Custom error message"
        assert error.context == {"custom_field": "value"}

    @pytest.mark.asyncio
    async def test_record_error_with_severity_override(self, manager):
        """Test recording error with manual severity override."""
        try:
            raise ValueError("Warning-level error")
        except ValueError as e:
            error_id = await manager.record_error(
                exception=e,
                severity_override=ErrorSeverity.WARNING
            )

        error = await manager.get_error_by_id(error_id)
        assert error.severity == ErrorSeverity.WARNING

    @pytest.mark.asyncio
    async def test_record_error_with_category_override(self, manager):
        """Test recording error with manual category override."""
        try:
            raise ValueError("Network-related error")
        except ValueError as e:
            error_id = await manager.record_error(
                exception=e,
                category_override=ErrorCategory.NETWORK
            )

        error = await manager.get_error_by_id(error_id)
        assert error.category == ErrorCategory.NETWORK

    @pytest.mark.asyncio
    async def test_record_error_automatic_categorization(self, manager):
        """Test automatic categorization integration."""
        # Test FileNotFoundError -> file_corrupt
        try:
            raise FileNotFoundError("/missing/file.txt")
        except FileNotFoundError as e:
            error_id = await manager.record_error(exception=e)

        error = await manager.get_error_by_id(error_id)
        assert error.category == ErrorCategory.FILE_CORRUPT
        assert error.severity == ErrorSeverity.ERROR

        # Test PermissionError -> permission_denied
        try:
            raise PermissionError("Access denied")
        except PermissionError as e:
            error_id2 = await manager.record_error(exception=e)

        error2 = await manager.get_error_by_id(error_id2)
        assert error2.category == ErrorCategory.PERMISSION_DENIED

    @pytest.mark.asyncio
    async def test_record_error_without_exception_or_message(self, manager):
        """Test that recording fails without exception or message."""
        with pytest.raises(ValueError, match="Either exception or message_override must be provided"):
            await manager.record_error()

    @pytest.mark.asyncio
    async def test_record_error_preserves_arbitrary_context(self, manager):
        """Test that arbitrary context fields are preserved."""
        error_id = await manager.record_error(
            message_override="Test",
            context={
                "file_path": "/test.py",
                "line_number": 42,
                "column": 10,
                "custom_metric": 123.45,
                "nested": {"key": "value"}
            }
        )

        error = await manager.get_error_by_id(error_id)
        assert error.context["line_number"] == 42
        assert error.context["column"] == 10
        assert error.context["custom_metric"] == 123.45
        assert error.context["nested"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_error_by_id_not_found(self, manager):
        """Test retrieving non-existent error."""
        error = await manager.get_error_by_id(999999)
        assert error is None

    @pytest.mark.asyncio
    async def test_get_errors_no_filter(self, manager):
        """Test retrieving all errors without filter."""
        # Create multiple errors
        for i in range(5):
            await manager.record_error(message_override=f"Error {i}")

        errors = await manager.get_errors()
        assert len(errors) == 5

    @pytest.mark.asyncio
    async def test_get_errors_filter_by_severity(self, manager):
        """Test filtering errors by severity."""
        # Create errors with different severities
        await manager.record_error(
            message_override="Error 1",
            severity_override=ErrorSeverity.ERROR
        )
        await manager.record_error(
            message_override="Warning 1",
            severity_override=ErrorSeverity.WARNING
        )
        await manager.record_error(
            message_override="Info 1",
            severity_override=ErrorSeverity.INFO
        )

        errors = await manager.get_errors(severity="error")
        assert len(errors) == 1
        assert errors[0].severity == ErrorSeverity.ERROR

        warnings = await manager.get_errors(severity="warning")
        assert len(warnings) == 1
        assert warnings[0].severity == ErrorSeverity.WARNING

    @pytest.mark.asyncio
    async def test_get_errors_filter_by_category(self, manager):
        """Test filtering errors by category."""
        await manager.record_error(
            message_override="Parse error",
            category_override=ErrorCategory.PARSE_ERROR
        )
        await manager.record_error(
            message_override="Network error",
            category_override=ErrorCategory.NETWORK
        )

        errors = await manager.get_errors(category="parse_error")
        assert len(errors) == 1
        assert errors[0].category == ErrorCategory.PARSE_ERROR

    @pytest.mark.asyncio
    async def test_get_errors_filter_by_acknowledged(self, manager):
        """Test filtering errors by acknowledgment status."""
        error_id1 = await manager.record_error(message_override="Error 1")
        error_id2 = await manager.record_error(message_override="Error 2")

        # Acknowledge one error
        await manager.acknowledge_error(error_id1, "admin")

        # Get unacknowledged
        unack = await manager.get_errors(acknowledged=False)
        assert len(unack) == 1
        assert unack[0].id == error_id2

        # Get acknowledged
        ack = await manager.get_errors(acknowledged=True)
        assert len(ack) == 1
        assert ack[0].id == error_id1

    @pytest.mark.asyncio
    async def test_get_errors_filter_by_date_range(self, manager):
        """Test filtering errors by date range."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Record error
        error_id = await manager.record_error(message_override="Test error")

        # Filter by date range (should include error)
        errors = await manager.get_errors(
            start_date=yesterday,
            end_date=tomorrow
        )
        assert len(errors) == 1
        assert errors[0].id == error_id

        # Filter with range that excludes error
        past_start = now - timedelta(days=10)
        past_end = now - timedelta(days=5)
        errors = await manager.get_errors(
            start_date=past_start,
            end_date=past_end
        )
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_get_errors_pagination(self, manager):
        """Test pagination with limit and offset."""
        # Create 10 errors
        for i in range(10):
            await manager.record_error(message_override=f"Error {i}")

        # Get first 5
        page1 = await manager.get_errors(limit=5, offset=0)
        assert len(page1) == 5

        # Get next 5
        page2 = await manager.get_errors(limit=5, offset=5)
        assert len(page2) == 5

        # Verify no overlap
        page1_ids = {e.id for e in page1}
        page2_ids = {e.id for e in page2}
        assert len(page1_ids & page2_ids) == 0

    @pytest.mark.asyncio
    async def test_get_errors_ordered_by_timestamp(self, manager):
        """Test that errors are returned in descending timestamp order."""
        # Create errors with slight delays
        error_id1 = await manager.record_error(message_override="First")
        error_id2 = await manager.record_error(message_override="Second")
        error_id3 = await manager.record_error(message_override="Third")

        errors = await manager.get_errors()

        # Most recent should be first
        assert errors[0].id == error_id3
        assert errors[1].id == error_id2
        assert errors[2].id == error_id1

    @pytest.mark.asyncio
    async def test_get_errors_invalid_severity(self, manager):
        """Test that invalid severity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid severity"):
            await manager.get_errors(severity="invalid")

    @pytest.mark.asyncio
    async def test_get_errors_invalid_category(self, manager):
        """Test that invalid category raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category"):
            await manager.get_errors(category="invalid_category")

    @pytest.mark.asyncio
    async def test_acknowledge_error(self, manager):
        """Test acknowledging an error."""
        error_id = await manager.record_error(message_override="Test error")

        # Acknowledge
        result = await manager.acknowledge_error(error_id, "admin")
        assert result is True

        # Verify acknowledgment
        error = await manager.get_error_by_id(error_id)
        assert error.acknowledged is True
        assert error.acknowledged_by == "admin"
        assert error.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_error_not_found(self, manager):
        """Test acknowledging non-existent error."""
        result = await manager.acknowledge_error(999999, "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_acknowledge_error_without_acknowledged_by(self, manager):
        """Test that acknowledging without acknowledged_by raises ValueError."""
        error_id = await manager.record_error(message_override="Test")

        with pytest.raises(ValueError, match="acknowledged_by must be provided"):
            await manager.acknowledge_error(error_id, "")

    @pytest.mark.asyncio
    async def test_get_error_stats_basic(self, manager):
        """Test getting basic error statistics."""
        # Create errors
        await manager.record_error(
            message_override="Error 1",
            severity_override=ErrorSeverity.ERROR
        )
        await manager.record_error(
            message_override="Warning 1",
            severity_override=ErrorSeverity.WARNING
        )
        await manager.record_error(
            message_override="Error 2",
            severity_override=ErrorSeverity.ERROR,
            category_override=ErrorCategory.NETWORK
        )

        stats = await manager.get_error_stats()

        assert stats.total_count == 3
        assert stats.by_severity["error"] == 2
        assert stats.by_severity["warning"] == 1
        assert stats.unacknowledged_count == 3
        assert stats.last_error_at is not None

    @pytest.mark.asyncio
    async def test_get_error_stats_with_acknowledgments(self, manager):
        """Test statistics with acknowledged errors."""
        error_id1 = await manager.record_error(message_override="Error 1")
        error_id2 = await manager.record_error(message_override="Error 2")

        # Acknowledge one
        await manager.acknowledge_error(error_id1, "admin")

        stats = await manager.get_error_stats()

        assert stats.total_count == 2
        assert stats.unacknowledged_count == 1

    @pytest.mark.asyncio
    async def test_get_error_stats_by_category(self, manager):
        """Test statistics by category."""
        await manager.record_error(
            message_override="Parse 1",
            category_override=ErrorCategory.PARSE_ERROR
        )
        await manager.record_error(
            message_override="Parse 2",
            category_override=ErrorCategory.PARSE_ERROR
        )
        await manager.record_error(
            message_override="Network 1",
            category_override=ErrorCategory.NETWORK
        )

        stats = await manager.get_error_stats()

        assert stats.by_category["parse_error"] == 2
        assert stats.by_category["network"] == 1

    @pytest.mark.asyncio
    async def test_get_error_stats_date_range(self, manager):
        """Test statistics with date range filtering."""
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)

        # Create error
        await manager.record_error(message_override="Test")

        # Stats within range
        stats = await manager.get_error_stats(
            start_date=yesterday,
            end_date=tomorrow
        )
        assert stats.total_count == 1

        # Stats outside range
        past_start = now - timedelta(days=10)
        past_end = now - timedelta(days=5)
        stats = await manager.get_error_stats(
            start_date=past_start,
            end_date=past_end
        )
        assert stats.total_count == 0

    @pytest.mark.asyncio
    async def test_get_error_stats_empty_database(self, manager):
        """Test statistics on empty database."""
        stats = await manager.get_error_stats()

        assert stats.total_count == 0
        assert stats.by_severity == {}
        assert stats.by_category == {}
        assert stats.unacknowledged_count == 0
        assert stats.last_error_at is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_manager_without_schema(self, temp_db):
        """Test manager behavior when schema doesn't exist."""
        # Create database without schema
        manager = ErrorMessageManager(db_path=temp_db)
        await manager.initialize()

        # Schema check should return False, but manager should still initialize
        assert manager._initialized is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_record_error_with_none_context(self, manager):
        """Test recording error with None context."""
        error_id = await manager.record_error(
            message_override="Test",
            context=None
        )

        error = await manager.get_error_by_id(error_id)
        assert error.context is None

    @pytest.mark.asyncio
    async def test_record_error_with_empty_context(self, manager):
        """Test recording error with empty context."""
        error_id = await manager.record_error(
            message_override="Test",
            context={}
        )

        error = await manager.get_error_by_id(error_id)
        assert error.context == {}

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, manager):
        """Test thread safety of concurrent operations."""
        import asyncio

        # Create multiple errors concurrently
        tasks = [
            manager.record_error(message_override=f"Error {i}")
            for i in range(10)
        ]

        error_ids = await asyncio.gather(*tasks)

        # All should succeed
        assert len(error_ids) == 10
        assert all(eid > 0 for eid in error_ids)

        # All should be retrievable
        errors = await manager.get_errors()
        assert len(errors) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
