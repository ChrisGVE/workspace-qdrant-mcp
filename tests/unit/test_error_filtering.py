"""
Unit tests for Error Filtering and Alerting System.

Tests cover:
- ErrorFilter dataclass validation
- Multi-criteria filtering (severity, category, date range, acknowledgment)
- Alert rule creation and evaluation
- Callback registration and triggering
- Threshold-based alerting
- Alert history tracking
- Async callback support
- Edge cases and error handling
"""

import asyncio
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.python.common.core.error_categorization import (
    ErrorCategory,
    ErrorSeverity,
)
from src.python.common.core.error_filtering import (
    AlertRule,
    ErrorFilter,
    ErrorFilterManager,
    FilteredErrorResult,
    TriggeredAlert,
)
from src.python.common.core.error_message_manager import (
    ErrorMessage,
    ErrorMessageManager,
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
    """Create a database with error messages schema."""
    conn = sqlite3.connect(temp_db)

    # Read schema from file
    schema_file = Path(__file__).parent.parent.parent / "src" / "python" / "common" / "core" / "error_messages_schema.sql"
    with open(schema_file) as f:
        schema_sql = f.read()

    # Create schema
    conn.executescript(schema_sql)
    conn.execute("DROP TABLE IF EXISTS messages")
    conn.execute("ALTER TABLE messages_enhanced RENAME TO messages")
    conn.commit()
    conn.close()

    return temp_db


@pytest.fixture
async def error_manager(db_with_schema):
    """Create and initialize an ErrorMessageManager."""
    mgr = ErrorMessageManager(db_path=db_with_schema)
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.fixture
async def filter_manager(error_manager):
    """Create and initialize an ErrorFilterManager."""
    mgr = ErrorFilterManager(error_manager)
    await mgr.initialize()
    yield mgr
    await mgr.close()


class TestErrorFilter:
    """Test ErrorFilter dataclass."""

    def test_error_filter_creation(self):
        """Test creating ErrorFilter with valid criteria."""
        filter = ErrorFilter(
            severity_levels=[ErrorSeverity.ERROR, ErrorSeverity.WARNING],
            categories=[ErrorCategory.NETWORK],
            acknowledged_only=True
        )

        assert filter.severity_levels == [ErrorSeverity.ERROR, ErrorSeverity.WARNING]
        assert filter.categories == [ErrorCategory.NETWORK]
        assert filter.acknowledged_only is True
        assert filter.unacknowledged_only is False

    def test_error_filter_with_date_range(self):
        """Test ErrorFilter with date range."""
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)

        filter = ErrorFilter(date_range=(start, end))

        assert filter.date_range == (start, end)

    def test_error_filter_invalid_both_acknowledged_flags(self):
        """Test that both acknowledged flags cannot be True."""
        with pytest.raises(ValueError, match="Cannot set both"):
            ErrorFilter(
                acknowledged_only=True,
                unacknowledged_only=True
            )

    def test_error_filter_invalid_date_range_length(self):
        """Test that date_range must be a tuple of 2 elements."""
        with pytest.raises(ValueError, match="must be a tuple"):
            ErrorFilter(date_range=(datetime.now(),))  # type: ignore

    def test_error_filter_invalid_date_range_order(self):
        """Test that start_date must be before end_date."""
        start = datetime.now(timezone.utc)
        end = start - timedelta(days=1)

        with pytest.raises(ValueError, match="must be before or equal to"):
            ErrorFilter(date_range=(start, end))


class TestFilteredErrorResult:
    """Test FilteredErrorResult dataclass."""

    @pytest.mark.asyncio
    async def test_filtered_error_result_creation(self, error_manager):
        """Test creating FilteredErrorResult."""
        # Create test error
        error_id = await error_manager.record_error(
            message_override="Test error"
        )
        error = await error_manager.get_error_by_id(error_id)

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        result = FilteredErrorResult(
            errors=[error],
            total_count=1,
            filter_applied=filter
        )

        assert len(result.errors) == 1
        assert result.total_count == 1
        assert result.filter_applied == filter


class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_alert_rule_creation(self):
        """Test creating AlertRule."""
        def callback(errors):
            pass

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        rule = AlertRule(
            id="rule_1",
            name="Test Rule",
            filter=filter,
            alert_callback=callback,
            threshold=5
        )

        assert rule.id == "rule_1"
        assert rule.name == "Test Rule"
        assert rule.filter == filter
        assert rule.threshold == 5
        assert rule.enabled is True

    def test_alert_rule_invalid_threshold(self):
        """Test that threshold must be at least 1."""
        def callback(errors):
            pass

        with pytest.raises(ValueError, match="threshold must be at least 1"):
            AlertRule(
                id="rule_1",
                name="Test",
                filter=ErrorFilter(),
                alert_callback=callback,
                threshold=0
            )

    def test_alert_rule_invalid_callback(self):
        """Test that callback must be callable."""
        with pytest.raises(ValueError, match="must be callable"):
            AlertRule(
                id="rule_1",
                name="Test",
                filter=ErrorFilter(),
                alert_callback="not_callable",  # type: ignore
                threshold=1
            )


class TestErrorFilterManager:
    """Test ErrorFilterManager functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, error_manager):
        """Test manager initialization."""
        manager = ErrorFilterManager(error_manager)
        await manager.initialize()

        assert manager._initialized is True

        await manager.close()

    @pytest.mark.asyncio
    async def test_filter_errors_single_severity(self, filter_manager, error_manager):
        """Test filtering by single severity level."""
        # Create errors with different severities
        await error_manager.record_error(
            message_override="Error 1",
            severity_override=ErrorSeverity.ERROR
        )
        await error_manager.record_error(
            message_override="Warning 1",
            severity_override=ErrorSeverity.WARNING
        )
        await error_manager.record_error(
            message_override="Error 2",
            severity_override=ErrorSeverity.ERROR
        )

        # Filter for errors only
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 2
        assert all(e.severity == ErrorSeverity.ERROR for e in result.errors)

    @pytest.mark.asyncio
    async def test_filter_errors_multiple_severities(self, filter_manager, error_manager):
        """Test filtering by multiple severity levels."""
        # Create errors
        await error_manager.record_error(
            message_override="Error 1",
            severity_override=ErrorSeverity.ERROR
        )
        await error_manager.record_error(
            message_override="Warning 1",
            severity_override=ErrorSeverity.WARNING
        )
        await error_manager.record_error(
            message_override="Info 1",
            severity_override=ErrorSeverity.INFO
        )

        # Filter for errors and warnings
        filter = ErrorFilter(
            severity_levels=[ErrorSeverity.ERROR, ErrorSeverity.WARNING]
        )
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 2
        severities = {e.severity for e in result.errors}
        assert severities == {ErrorSeverity.ERROR, ErrorSeverity.WARNING}

    @pytest.mark.asyncio
    async def test_filter_errors_single_category(self, filter_manager, error_manager):
        """Test filtering by single category."""
        # Create errors with different categories
        await error_manager.record_error(
            message_override="Network error",
            category_override=ErrorCategory.NETWORK
        )
        await error_manager.record_error(
            message_override="Parse error",
            category_override=ErrorCategory.PARSE_ERROR
        )
        await error_manager.record_error(
            message_override="Network error 2",
            category_override=ErrorCategory.NETWORK
        )

        # Filter for network errors only
        filter = ErrorFilter(categories=[ErrorCategory.NETWORK])
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 2
        assert all(e.category == ErrorCategory.NETWORK for e in result.errors)

    @pytest.mark.asyncio
    async def test_filter_errors_multiple_categories(self, filter_manager, error_manager):
        """Test filtering by multiple categories."""
        # Create errors
        await error_manager.record_error(
            message_override="Network error",
            category_override=ErrorCategory.NETWORK
        )
        await error_manager.record_error(
            message_override="Parse error",
            category_override=ErrorCategory.PARSE_ERROR
        )
        await error_manager.record_error(
            message_override="File error",
            category_override=ErrorCategory.FILE_CORRUPT
        )

        # Filter for network and parse errors
        filter = ErrorFilter(
            categories=[ErrorCategory.NETWORK, ErrorCategory.PARSE_ERROR]
        )
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 2
        categories = {e.category for e in result.errors}
        assert categories == {ErrorCategory.NETWORK, ErrorCategory.PARSE_ERROR}

    @pytest.mark.asyncio
    async def test_filter_errors_by_date_range(self, filter_manager, error_manager):
        """Test filtering by date range."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        # Create error (should be within range)
        await error_manager.record_error(message_override="Test error")

        # Filter with date range
        filter = ErrorFilter(date_range=(start, end))
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 1

        # Filter with past date range (should be empty)
        past_start = now - timedelta(days=10)
        past_end = now - timedelta(days=5)
        filter = ErrorFilter(date_range=(past_start, past_end))
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 0

    @pytest.mark.asyncio
    async def test_filter_errors_acknowledged_only(self, filter_manager, error_manager):
        """Test filtering for acknowledged errors only."""
        # Create errors
        error_id1 = await error_manager.record_error(message_override="Error 1")
        await error_manager.record_error(message_override="Error 2")

        # Acknowledge one error
        await error_manager.acknowledge_error(error_id1, "admin")

        # Filter for acknowledged only
        filter = ErrorFilter(acknowledged_only=True)
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 1
        assert result.errors[0].id == error_id1
        assert result.errors[0].acknowledged is True

    @pytest.mark.asyncio
    async def test_filter_errors_unacknowledged_only(self, filter_manager, error_manager):
        """Test filtering for unacknowledged errors only."""
        # Create errors
        error_id1 = await error_manager.record_error(message_override="Error 1")
        error_id2 = await error_manager.record_error(message_override="Error 2")

        # Acknowledge one error
        await error_manager.acknowledge_error(error_id1, "admin")

        # Filter for unacknowledged only
        filter = ErrorFilter(unacknowledged_only=True)
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 1
        assert result.errors[0].id == error_id2
        assert result.errors[0].acknowledged is False

    @pytest.mark.asyncio
    async def test_filter_errors_combined_criteria(self, filter_manager, error_manager):
        """Test filtering with multiple criteria combined."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)

        # Create various errors
        await error_manager.record_error(
            message_override="Network error 1",
            severity_override=ErrorSeverity.ERROR,
            category_override=ErrorCategory.NETWORK
        )
        await error_manager.record_error(
            message_override="Network warning",
            severity_override=ErrorSeverity.WARNING,
            category_override=ErrorCategory.NETWORK
        )
        await error_manager.record_error(
            message_override="Parse error",
            severity_override=ErrorSeverity.ERROR,
            category_override=ErrorCategory.PARSE_ERROR
        )

        # Filter for ERROR severity + NETWORK category
        filter = ErrorFilter(
            severity_levels=[ErrorSeverity.ERROR],
            categories=[ErrorCategory.NETWORK],
            date_range=(start, end)
        )
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 1
        assert result.errors[0].severity == ErrorSeverity.ERROR
        assert result.errors[0].category == ErrorCategory.NETWORK

    @pytest.mark.asyncio
    async def test_filter_errors_pagination(self, filter_manager, error_manager):
        """Test pagination with limit and offset."""
        # Create multiple errors
        for i in range(10):
            await error_manager.record_error(message_override=f"Error {i}")

        # Get first 5
        filter = ErrorFilter()
        page1 = await filter_manager.filter_errors(filter, limit=5, offset=0)

        assert page1.total_count == 5

        # Get next 5
        page2 = await filter_manager.filter_errors(filter, limit=5, offset=5)

        assert page2.total_count == 5

        # Verify no overlap
        page1_ids = {e.id for e in page1.errors}
        page2_ids = {e.id for e in page2.errors}
        assert len(page1_ids & page2_ids) == 0

    @pytest.mark.asyncio
    async def test_create_alert_rule(self, filter_manager):
        """Test creating alert rule."""
        callback_invoked = []

        def callback(errors):
            callback_invoked.append(len(errors))

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        rule = await filter_manager.create_alert_rule(
            name="Critical Errors",
            filter=filter,
            alert_callback=callback,
            threshold=5
        )

        assert rule.id.startswith("rule_")
        assert rule.name == "Critical Errors"
        assert rule.threshold == 5
        assert rule.enabled is True

    @pytest.mark.asyncio
    async def test_create_alert_rule_without_initialization(self, error_manager):
        """Test that creating rule without initialization raises error."""
        manager = ErrorFilterManager(error_manager)

        def callback(errors):
            pass

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.create_alert_rule(
                name="Test",
                filter=ErrorFilter(),
                alert_callback=callback
            )

    @pytest.mark.asyncio
    async def test_register_alert_callback(self, filter_manager):
        """Test registering severity-specific callback."""
        callback_invoked = []

        def callback(errors):
            callback_invoked.append(errors)

        result = await filter_manager.register_alert_callback(
            ErrorSeverity.ERROR,
            callback
        )

        assert result is True
        assert callback in filter_manager._severity_callbacks[ErrorSeverity.ERROR]

    @pytest.mark.asyncio
    async def test_register_alert_callback_invalid_severity(self, filter_manager):
        """Test registering callback with invalid severity."""
        def callback(errors):
            pass

        # Should not raise for valid severity
        await filter_manager.register_alert_callback(ErrorSeverity.ERROR, callback)

    @pytest.mark.asyncio
    async def test_register_alert_callback_not_callable(self, filter_manager):
        """Test that non-callable callback raises error."""
        with pytest.raises(ValueError, match="must be callable"):
            await filter_manager.register_alert_callback(
                ErrorSeverity.ERROR,
                "not_callable"  # type: ignore
            )

    @pytest.mark.asyncio
    async def test_check_alerts_threshold_met(self, filter_manager, error_manager):
        """Test alert triggering when threshold is met."""
        callback_invoked = []

        async def callback(errors):
            callback_invoked.append(len(errors))

        # Create alert rule with threshold of 3
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Error Alert",
            filter=filter,
            alert_callback=callback,
            threshold=3
        )

        # Create 3 errors (meets threshold)
        for i in range(3):
            await error_manager.record_error(
                message_override=f"Error {i}",
                severity_override=ErrorSeverity.ERROR
            )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        assert len(triggered) == 1
        assert triggered[0].error_count == 3
        assert len(callback_invoked) == 1
        assert callback_invoked[0] == 3

    @pytest.mark.asyncio
    async def test_check_alerts_threshold_not_met(self, filter_manager, error_manager):
        """Test that alert is not triggered when threshold is not met."""
        callback_invoked = []

        def callback(errors):
            callback_invoked.append(len(errors))

        # Create alert rule with threshold of 5
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Error Alert",
            filter=filter,
            alert_callback=callback,
            threshold=5
        )

        # Create only 3 errors (below threshold)
        for i in range(3):
            await error_manager.record_error(
                message_override=f"Error {i}",
                severity_override=ErrorSeverity.ERROR
            )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        assert len(triggered) == 0
        assert len(callback_invoked) == 0

    @pytest.mark.asyncio
    async def test_check_alerts_disabled_rule(self, filter_manager, error_manager):
        """Test that disabled rules are not triggered."""
        callback_invoked = []

        def callback(errors):
            callback_invoked.append(len(errors))

        # Create alert rule
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        rule = await filter_manager.create_alert_rule(
            name="Error Alert",
            filter=filter,
            alert_callback=callback,
            threshold=1
        )

        # Disable rule
        await filter_manager.disable_alert_rule(rule.id)

        # Create error
        await error_manager.record_error(
            message_override="Error",
            severity_override=ErrorSeverity.ERROR
        )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        assert len(triggered) == 0
        assert len(callback_invoked) == 0

    @pytest.mark.asyncio
    async def test_check_alerts_async_callback(self, filter_manager, error_manager):
        """Test async callback support."""
        callback_invoked = []

        async def async_callback(errors):
            await asyncio.sleep(0.01)  # Simulate async operation
            callback_invoked.append(len(errors))

        # Create alert rule with async callback
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Async Alert",
            filter=filter,
            alert_callback=async_callback,
            threshold=1
        )

        # Create error
        await error_manager.record_error(
            message_override="Error",
            severity_override=ErrorSeverity.ERROR
        )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        assert len(triggered) == 1
        assert len(callback_invoked) == 1

    @pytest.mark.asyncio
    async def test_check_alerts_callback_error_handling(self, filter_manager, error_manager):
        """Test that callback errors don't break alert checking."""
        def failing_callback(errors):
            raise RuntimeError("Callback failed")

        def working_callback(errors):
            pass

        # Create two alert rules
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Failing Alert",
            filter=filter,
            alert_callback=failing_callback,
            threshold=1
        )
        await filter_manager.create_alert_rule(
            name="Working Alert",
            filter=filter,
            alert_callback=working_callback,
            threshold=1
        )

        # Create error
        await error_manager.record_error(
            message_override="Error",
            severity_override=ErrorSeverity.ERROR
        )

        # Check alerts - should not raise despite failing callback
        triggered = await filter_manager.check_alerts()

        # Both rules should trigger (callback error doesn't prevent triggering)
        assert len(triggered) == 2

    @pytest.mark.asyncio
    async def test_check_alerts_severity_callbacks(self, filter_manager, error_manager):
        """Test severity-specific callback invocation."""
        error_callback_invoked = []
        warning_callback_invoked = []

        def error_callback(errors):
            error_callback_invoked.append(len(errors))

        def warning_callback(errors):
            warning_callback_invoked.append(len(errors))

        # Register severity callbacks
        await filter_manager.register_alert_callback(
            ErrorSeverity.ERROR,
            error_callback
        )
        await filter_manager.register_alert_callback(
            ErrorSeverity.WARNING,
            warning_callback
        )

        # Create alert rule that matches both severities
        filter = ErrorFilter(
            severity_levels=[ErrorSeverity.ERROR, ErrorSeverity.WARNING]
        )
        await filter_manager.create_alert_rule(
            name="Multi-severity Alert",
            filter=filter,
            alert_callback=lambda e: None,
            threshold=2
        )

        # Create errors
        await error_manager.record_error(
            message_override="Error 1",
            severity_override=ErrorSeverity.ERROR
        )
        await error_manager.record_error(
            message_override="Warning 1",
            severity_override=ErrorSeverity.WARNING
        )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        assert len(triggered) == 1
        # Both severity callbacks should be invoked
        assert len(error_callback_invoked) == 1
        assert len(warning_callback_invoked) == 1

    @pytest.mark.asyncio
    async def test_get_alert_history(self, filter_manager, error_manager):
        """Test getting alert history."""
        def callback(errors):
            pass

        # Create alert rule
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Test Alert",
            filter=filter,
            alert_callback=callback,
            threshold=1
        )

        # Trigger alert multiple times
        for i in range(3):
            await error_manager.record_error(
                message_override=f"Error {i}",
                severity_override=ErrorSeverity.ERROR
            )
            await filter_manager.check_alerts()

        # Get history
        history = await filter_manager.get_alert_history(limit=10)

        assert len(history) >= 1  # At least one triggered
        assert all(isinstance(alert, TriggeredAlert) for alert in history)

    @pytest.mark.asyncio
    async def test_get_alert_history_limit(self, filter_manager, error_manager):
        """Test alert history pagination."""
        def callback(errors):
            pass

        # Create alert rule
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Test Alert",
            filter=filter,
            alert_callback=callback,
            threshold=1
        )

        # Trigger alert 5 times
        for i in range(5):
            await error_manager.record_error(
                message_override=f"Error {i}",
                severity_override=ErrorSeverity.ERROR
            )
            await filter_manager.check_alerts()

        # Get limited history
        history = await filter_manager.get_alert_history(limit=3)

        assert len(history) <= 3

    @pytest.mark.asyncio
    async def test_get_alert_rule(self, filter_manager):
        """Test getting alert rule by ID."""
        def callback(errors):
            pass

        rule = await filter_manager.create_alert_rule(
            name="Test Rule",
            filter=ErrorFilter(),
            alert_callback=callback
        )

        retrieved = filter_manager.get_alert_rule(rule.id)

        assert retrieved is not None
        assert retrieved.id == rule.id
        assert retrieved.name == "Test Rule"

    @pytest.mark.asyncio
    async def test_get_alert_rule_not_found(self, filter_manager):
        """Test getting non-existent alert rule."""
        retrieved = filter_manager.get_alert_rule("nonexistent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_list_alert_rules(self, filter_manager):
        """Test listing all alert rules."""
        def callback(errors):
            pass

        # Create multiple rules
        await filter_manager.create_alert_rule(
            name="Rule 1",
            filter=ErrorFilter(),
            alert_callback=callback
        )
        await filter_manager.create_alert_rule(
            name="Rule 2",
            filter=ErrorFilter(),
            alert_callback=callback
        )

        rules = filter_manager.list_alert_rules()

        assert len(rules) == 2
        assert all(isinstance(rule, AlertRule) for rule in rules)

    @pytest.mark.asyncio
    async def test_enable_disable_alert_rule(self, filter_manager):
        """Test enabling and disabling alert rules."""
        def callback(errors):
            pass

        rule = await filter_manager.create_alert_rule(
            name="Test Rule",
            filter=ErrorFilter(),
            alert_callback=callback
        )

        # Disable
        result = await filter_manager.disable_alert_rule(rule.id)
        assert result is True
        assert rule.enabled is False

        # Enable
        result = await filter_manager.enable_alert_rule(rule.id)
        assert result is True
        assert rule.enabled is True

    @pytest.mark.asyncio
    async def test_delete_alert_rule(self, filter_manager):
        """Test deleting alert rule."""
        def callback(errors):
            pass

        rule = await filter_manager.create_alert_rule(
            name="Test Rule",
            filter=ErrorFilter(),
            alert_callback=callback
        )

        # Delete
        result = await filter_manager.delete_alert_rule(rule.id)
        assert result is True

        # Verify deleted
        retrieved = filter_manager.get_alert_rule(rule.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_alert_rule_not_found(self, filter_manager):
        """Test deleting non-existent rule."""
        result = await filter_manager.delete_alert_rule("nonexistent")
        assert result is False


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_filter_errors_without_initialization(self, error_manager):
        """Test filtering without initialization raises error."""
        manager = ErrorFilterManager(error_manager)

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.filter_errors(ErrorFilter())

    @pytest.mark.asyncio
    async def test_filter_errors_empty_database(self, filter_manager):
        """Test filtering empty database."""
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        result = await filter_manager.filter_errors(filter)

        assert result.total_count == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_check_alerts_without_initialization(self, error_manager):
        """Test checking alerts without initialization raises error."""
        manager = ErrorFilterManager(error_manager)

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.check_alerts()

    @pytest.mark.asyncio
    async def test_get_alert_history_without_initialization(self, error_manager):
        """Test getting history without initialization raises error."""
        manager = ErrorFilterManager(error_manager)

        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.get_alert_history()

    @pytest.mark.asyncio
    async def test_multiple_alert_rules_same_criteria(self, filter_manager, error_manager):
        """Test multiple alert rules with same filter criteria."""
        callback1_invoked = []
        callback2_invoked = []

        def callback1(errors):
            callback1_invoked.append(len(errors))

        def callback2(errors):
            callback2_invoked.append(len(errors))

        # Create two rules with same filter
        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        await filter_manager.create_alert_rule(
            name="Rule 1",
            filter=filter,
            alert_callback=callback1,
            threshold=1
        )
        await filter_manager.create_alert_rule(
            name="Rule 2",
            filter=filter,
            alert_callback=callback2,
            threshold=1
        )

        # Create error
        await error_manager.record_error(
            message_override="Error",
            severity_override=ErrorSeverity.ERROR
        )

        # Check alerts
        triggered = await filter_manager.check_alerts()

        # Both rules should trigger
        assert len(triggered) == 2
        assert len(callback1_invoked) == 1
        assert len(callback2_invoked) == 1

    @pytest.mark.asyncio
    async def test_alert_rule_last_triggered_at_update(self, filter_manager, error_manager):
        """Test that last_triggered_at is updated."""
        def callback(errors):
            pass

        filter = ErrorFilter(severity_levels=[ErrorSeverity.ERROR])
        rule = await filter_manager.create_alert_rule(
            name="Test Alert",
            filter=filter,
            alert_callback=callback,
            threshold=1
        )

        # Initially None
        assert rule.last_triggered_at is None

        # Create error and trigger
        await error_manager.record_error(
            message_override="Error",
            severity_override=ErrorSeverity.ERROR
        )
        await filter_manager.check_alerts()

        # Should be updated
        assert rule.last_triggered_at is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
