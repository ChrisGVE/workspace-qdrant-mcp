"""
Unit tests for budget warning system.

Tests comprehensive warning functionality including threshold detection,
notification channels, throttling, and integration with budget managers.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from common.core.context_injection.budget_warning_system import (
    BudgetThreshold,
    BudgetWarningSystem,
    ThrottleConfig,
    WarningEvent,
    WarningLevel,
)
from common.core.context_injection.claude_budget_manager import (
    ClaudeBudgetManager,
    ClaudeModel,
)
from common.core.context_injection.token_usage_tracker import (
    OperationType,
    TokenUsageTracker,
)


class TestWarningLevel:
    """Test WarningLevel enum."""

    def test_warning_levels(self):
        """Test warning level values and thresholds."""
        assert WarningLevel.INFO.value == "info"
        assert WarningLevel.WARNING.value == "warning"
        assert WarningLevel.CRITICAL.value == "critical"

    def test_default_thresholds(self):
        """Test default threshold percentages."""
        assert WarningLevel.INFO.default_threshold == 75.0
        assert WarningLevel.WARNING.default_threshold == 85.0
        assert WarningLevel.CRITICAL.default_threshold == 95.0


class TestBudgetThreshold:
    """Test BudgetThreshold dataclass."""

    def test_valid_threshold(self):
        """Test creating valid threshold."""
        threshold = BudgetThreshold(WarningLevel.WARNING, 80.0)
        assert threshold.level == WarningLevel.WARNING
        assert threshold.percentage == 80.0
        assert threshold.enabled is True

    def test_threshold_with_enabled(self):
        """Test threshold with enabled flag."""
        threshold = BudgetThreshold(WarningLevel.INFO, 75.0, enabled=False)
        assert threshold.enabled is False

    def test_invalid_percentage_too_low(self):
        """Test invalid percentage below 0."""
        with pytest.raises(ValueError, match="Threshold percentage must be 0-100"):
            BudgetThreshold(WarningLevel.INFO, -10.0)

    def test_invalid_percentage_too_high(self):
        """Test invalid percentage above 100."""
        with pytest.raises(ValueError, match="Threshold percentage must be 0-100"):
            BudgetThreshold(WarningLevel.CRITICAL, 150.0)

    def test_boundary_percentages(self):
        """Test boundary values 0 and 100."""
        threshold1 = BudgetThreshold(WarningLevel.INFO, 0.0)
        assert threshold1.percentage == 0.0

        threshold2 = BudgetThreshold(WarningLevel.CRITICAL, 100.0)
        assert threshold2.percentage == 100.0


class TestWarningEvent:
    """Test WarningEvent dataclass."""

    def test_create_warning_event(self):
        """Test creating warning event."""
        timestamp = datetime.now(timezone.utc)
        event = WarningEvent(
            timestamp=timestamp,
            level=WarningLevel.WARNING,
            tool_name="claude",
            current_usage=8500,
            budget_limit=10000,
            utilization_percentage=85.0,
            message="Test warning",
        )

        assert event.timestamp == timestamp
        assert event.level == WarningLevel.WARNING
        assert event.tool_name == "claude"
        assert event.current_usage == 8500
        assert event.budget_limit == 10000
        assert event.utilization_percentage == 85.0
        assert event.message == "Test warning"
        assert event.metadata == {}

    def test_warning_event_with_metadata(self):
        """Test warning event with metadata."""
        event = WarningEvent(
            timestamp=datetime.now(timezone.utc),
            level=WarningLevel.CRITICAL,
            tool_name=None,
            current_usage=9500,
            budget_limit=10000,
            utilization_percentage=95.0,
            message="Critical warning",
            metadata={"scope": "global", "tokens_remaining": 500},
        )

        assert event.tool_name is None
        assert event.metadata["scope"] == "global"
        assert event.metadata["tokens_remaining"] == 500

    def test_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        event = WarningEvent(
            timestamp=timestamp,
            level=WarningLevel.INFO,
            tool_name="codex",
            current_usage=7500,
            budget_limit=10000,
            utilization_percentage=75.0,
            message="Info message",
            metadata={"test": "data"},
        )

        event_dict = event.to_dict()
        assert event_dict["timestamp"] == "2024-01-15T12:00:00+00:00"
        assert event_dict["level"] == "info"
        assert event_dict["tool_name"] == "codex"
        assert event_dict["current_usage"] == 7500
        assert event_dict["budget_limit"] == 10000
        assert event_dict["utilization_percentage"] == 75.0
        assert event_dict["message"] == "Info message"
        assert event_dict["metadata"]["test"] == "data"


class TestThrottleConfig:
    """Test ThrottleConfig dataclass."""

    def test_default_config(self):
        """Test default throttle configuration."""
        config = ThrottleConfig()
        assert config.enabled is True
        assert config.min_interval_seconds == 60.0
        assert config.max_warnings_per_minute == 5

    def test_custom_config(self):
        """Test custom throttle configuration."""
        config = ThrottleConfig(
            enabled=False, min_interval_seconds=30.0, max_warnings_per_minute=10
        )
        assert config.enabled is False
        assert config.min_interval_seconds == 30.0
        assert config.max_warnings_per_minute == 10


class TestBudgetWarningSystem:
    """Test BudgetWarningSystem class."""

    def test_initialization(self):
        """Test system initialization with defaults."""
        system = BudgetWarningSystem()

        assert system.global_budget_limit is None
        assert system.per_tool_limits == {}
        assert system.enable_logging is True
        assert system.enable_cli_output is False
        assert len(system.thresholds) == 3

    def test_initialization_with_limits(self):
        """Test initialization with budget limits."""
        system = BudgetWarningSystem(
            global_budget_limit=200000,
            per_tool_limits={"claude": 100000, "codex": 50000},
        )

        assert system.global_budget_limit == 200000
        assert system.per_tool_limits["claude"] == 100000
        assert system.per_tool_limits["codex"] == 50000

    def test_initialization_with_custom_thresholds(self):
        """Test initialization with custom thresholds."""
        thresholds = [
            BudgetThreshold(WarningLevel.INFO, 70.0),
            BudgetThreshold(WarningLevel.WARNING, 80.0),
            BudgetThreshold(WarningLevel.CRITICAL, 90.0),
        ]

        system = BudgetWarningSystem(thresholds=thresholds)

        assert len(system.thresholds) == 3
        assert system.thresholds[0].percentage == 70.0
        assert system.thresholds[1].percentage == 80.0
        assert system.thresholds[2].percentage == 90.0

    def test_register_callback(self):
        """Test registering warning callback."""
        system = BudgetWarningSystem()

        def test_callback(event: WarningEvent):
            pass

        system.register_callback(test_callback)
        assert test_callback in system._warning_callbacks

    def test_unregister_callback(self):
        """Test unregistering warning callback."""
        system = BudgetWarningSystem()

        def test_callback(event: WarningEvent):
            pass

        system.register_callback(test_callback)
        system.unregister_callback(test_callback)
        assert test_callback not in system._warning_callbacks

    def test_check_usage_tracker_no_warnings(self):
        """Test checking tracker with usage below thresholds."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        # Use only 50% of budget
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 5000)

        warnings = system.check_usage_tracker(tracker)
        assert len(warnings) == 0

    def test_check_usage_tracker_info_warning(self):
        """Test tracker triggering INFO warning."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        # Use 76% of budget (triggers INFO at 75%)
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        warnings = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings) == 1
        assert warnings[0].level == WarningLevel.INFO
        assert warnings[0].tool_name == "claude"
        assert warnings[0].current_usage == 7600
        assert warnings[0].utilization_percentage == 76.0

    def test_check_usage_tracker_multiple_thresholds(self):
        """Test tracker triggering multiple warning levels."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()

        # First check - 76% triggers INFO
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        warnings1 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings1) == 1
        assert warnings1[0].level == WarningLevel.INFO

        # Second check - 86% triggers WARNING
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 1000)
        warnings2 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings2) == 1
        assert warnings2[0].level == WarningLevel.WARNING

        # Third check - 96% triggers CRITICAL
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 1000)
        warnings3 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings3) == 1
        assert warnings3[0].level == WarningLevel.CRITICAL

    def test_check_usage_tracker_global_budget(self):
        """Test checking global budget limit."""
        system = BudgetWarningSystem(
            global_budget_limit=20000,
        )

        tracker = TokenUsageTracker()
        # Use 76% of global budget
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 10000)
        tracker.track_operation("codex", OperationType.SEARCH, 5200)

        warnings = system.check_usage_tracker(tracker)
        assert len(warnings) == 1
        assert warnings[0].level == WarningLevel.INFO
        assert warnings[0].tool_name is None
        assert warnings[0].current_usage == 15200

    def test_check_claude_budget_manager(self):
        """Test checking ClaudeBudgetManager."""
        system = BudgetWarningSystem()

        manager = ClaudeBudgetManager(
            model=ClaudeModel.SONNET_3_5,
            custom_budget_limit=10000,
        )

        # Allocate tokens to trigger warning
        # Start with high usage
        manager.session_stats.total_tokens_used = 7600

        warnings = system.check_claude_budget_manager(manager)

        # Should trigger INFO warning
        assert len(warnings) >= 1
        assert any(w.level == WarningLevel.INFO for w in warnings)

    def test_throttling_same_level(self):
        """Test throttling prevents duplicate warnings."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
            throttle_config=ThrottleConfig(min_interval_seconds=60.0),
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        # First check triggers warning
        warnings1 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings1) == 1

        # Second immediate check is throttled
        warnings2 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings2) == 0

    def test_throttling_disabled(self):
        """Test throttling can be disabled."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
            throttle_config=ThrottleConfig(enabled=False),
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        # Both checks should trigger warnings (no throttling)
        warnings1 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings1) == 1

        warnings2 = system.check_usage_tracker(tracker, tool_name="claude")
        # Still triggers because same threshold (already triggered)
        assert len(warnings2) == 0  # But threshold tracking prevents duplicate

    def test_throttling_per_minute_limit(self):
        """Test per-minute warning limit throttling."""
        system = BudgetWarningSystem(
            per_tool_limits={f"tool{i}": 10000 for i in range(10)},
            throttle_config=ThrottleConfig(max_warnings_per_minute=3),
        )

        tracker = TokenUsageTracker()

        # Trigger warnings for multiple tools
        warnings = []
        for i in range(5):
            tool_name = f"tool{i}"
            tracker.track_operation(tool_name, OperationType.CONTEXT_INJECTION, 7600)
            w = system.check_usage_tracker(tracker, tool_name=tool_name)
            warnings.extend(w)

        # Should be throttled after 3 warnings
        assert len(warnings) == 3

    def test_get_warning_history(self):
        """Test retrieving warning history."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        system.check_usage_tracker(tracker, tool_name="claude")

        history = system.get_warning_history()
        assert len(history) == 1
        assert history[0].tool_name == "claude"

    def test_get_warning_history_filtered_by_tool(self):
        """Test filtering history by tool name."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000, "codex": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        tracker.track_operation("codex", OperationType.SEARCH, 7600)

        system.check_usage_tracker(tracker, tool_name="claude")
        system.check_usage_tracker(tracker, tool_name="codex")

        claude_history = system.get_warning_history(tool_name="claude")
        assert len(claude_history) == 1
        assert claude_history[0].tool_name == "claude"

    def test_get_warning_history_filtered_by_level(self):
        """Test filtering history by warning level."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        system.check_usage_tracker(tracker, tool_name="claude")

        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 1000)
        system.check_usage_tracker(tracker, tool_name="claude")

        info_history = system.get_warning_history(level=WarningLevel.INFO)
        assert len(info_history) == 1
        assert info_history[0].level == WarningLevel.INFO

    def test_get_warning_history_filtered_by_time(self):
        """Test filtering history by timestamp."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        system.check_usage_tracker(tracker, tool_name="claude")

        # Get warnings from 1 minute ago
        since = datetime.now(timezone.utc) - timedelta(minutes=1)
        recent_history = system.get_warning_history(since=since)
        assert len(recent_history) == 1

        # Get warnings from the future (should be empty)
        future = datetime.now(timezone.utc) + timedelta(minutes=1)
        future_history = system.get_warning_history(since=future)
        assert len(future_history) == 0

    def test_get_warning_summary(self):
        """Test warning summary generation."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        system.check_usage_tracker(tracker, tool_name="claude")

        summary = system.get_warning_summary()
        assert summary["total_warnings"] == 1
        assert summary["warnings_by_level"]["info"] == 1
        assert summary["warnings_by_tool"]["claude"] == 1
        assert len(summary["thresholds_configured"]) == 3

    def test_clear_history_all(self):
        """Test clearing all warning history."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        system.check_usage_tracker(tracker, tool_name="claude")

        cleared = system.clear_history()
        assert cleared == 1
        assert len(system.get_warning_history()) == 0

    def test_clear_history_by_tool(self):
        """Test clearing history for specific tool."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000, "codex": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        tracker.track_operation("codex", OperationType.SEARCH, 7600)

        system.check_usage_tracker(tracker, tool_name="claude")
        system.check_usage_tracker(tracker, tool_name="codex")

        cleared = system.clear_history(tool_name="claude")
        assert cleared == 1

        history = system.get_warning_history()
        assert len(history) == 1
        assert history[0].tool_name == "codex"

    @patch("common.core.context_injection.budget_warning_system.logger")
    def test_logging_notification(self, mock_logger):
        """Test logging notifications are sent."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
            enable_logging=True,
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        system.check_usage_tracker(tracker, tool_name="claude")

        # Should have logged INFO warning
        assert mock_logger.info.called

    def test_callback_notification(self):
        """Test callback notifications are invoked."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        callback_events = []

        def test_callback(event: WarningEvent):
            callback_events.append(event)

        system.register_callback(test_callback)

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        system.check_usage_tracker(tracker, tool_name="claude")

        assert len(callback_events) == 1
        assert callback_events[0].tool_name == "claude"

    def test_callback_error_handling(self):
        """Test callback errors are handled gracefully."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        def bad_callback(event: WarningEvent):
            raise ValueError("Callback error")

        system.register_callback(bad_callback)

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        # Should not raise exception
        warnings = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings) == 1

    def test_disabled_threshold(self):
        """Test disabled thresholds are not triggered."""
        thresholds = [
            BudgetThreshold(WarningLevel.INFO, 75.0, enabled=False),
            BudgetThreshold(WarningLevel.WARNING, 85.0),
        ]

        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
            thresholds=thresholds,
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        warnings = system.check_usage_tracker(tracker, tool_name="claude")
        # INFO threshold disabled, should not trigger
        assert len(warnings) == 0

    def test_no_duplicate_warnings_same_threshold(self):
        """Test that same threshold doesn't trigger twice."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)

        # First check
        warnings1 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings1) == 1

        # Add more tokens (still in INFO range)
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 100)

        # Second check - no new warning (INFO already triggered)
        warnings2 = system.check_usage_tracker(tracker, tool_name="claude")
        assert len(warnings2) == 0

    def test_check_all_tools(self):
        """Test checking all tools in tracker."""
        system = BudgetWarningSystem(
            per_tool_limits={"claude": 10000, "codex": 10000},
        )

        tracker = TokenUsageTracker()
        tracker.track_operation("claude", OperationType.CONTEXT_INJECTION, 7600)
        tracker.track_operation("codex", OperationType.SEARCH, 8600)

        # Check all tools at once
        warnings = system.check_usage_tracker(tracker)

        # Both tools should trigger warnings (codex triggers 2: INFO and WARNING)
        assert len(warnings) == 3
        tool_names = {w.tool_name for w in warnings}
        assert "claude" in tool_names
        assert "codex" in tool_names
        
        # Verify levels
        claude_warnings = [w for w in warnings if w.tool_name == "claude"]
        assert len(claude_warnings) == 1
        assert claude_warnings[0].level == WarningLevel.INFO
        
        codex_warnings = [w for w in warnings if w.tool_name == "codex"]
        assert len(codex_warnings) == 2
        assert any(w.level == WarningLevel.INFO for w in codex_warnings)
        assert any(w.level == WarningLevel.WARNING for w in codex_warnings)
