"""
Budget warning and alerting system for token usage monitoring.

This module provides a comprehensive warning system for token budget management
with configurable thresholds, multiple notification channels, and integration
with TokenUsageTracker and ClaudeBudgetManager.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from .claude_budget_manager import ClaudeBudgetManager, SessionUsageStats
from .token_usage_tracker import TokenUsageTracker, ToolUsageStats


class WarningLevel(Enum):
    """Warning severity levels for budget threshold violations."""

    INFO = "info"  # 75% - informational notice
    WARNING = "warning"  # 85% - caution advised
    CRITICAL = "critical"  # 95% - urgent attention required

    @property
    def default_threshold(self) -> float:
        """
        Get default threshold percentage for this warning level.

        Returns:
            Threshold percentage (0-100)
        """
        return {
            WarningLevel.INFO: 75.0,
            WarningLevel.WARNING: 85.0,
            WarningLevel.CRITICAL: 95.0,
        }[self]


@dataclass
class BudgetThreshold:
    """
    Budget threshold configuration for warnings.

    Attributes:
        level: Warning severity level
        percentage: Utilization percentage that triggers warning (0-100)
        enabled: Whether this threshold is active
    """

    level: WarningLevel
    percentage: float
    enabled: bool = True

    def __post_init__(self):
        """Validate threshold configuration."""
        if not 0 <= self.percentage <= 100:
            raise ValueError(
                f"Threshold percentage must be 0-100, got {self.percentage}"
            )


@dataclass
class WarningEvent:
    """
    Record of a budget warning event.

    Attributes:
        timestamp: When the warning occurred
        level: Warning severity level
        tool_name: Tool that triggered warning (None for global)
        current_usage: Current token usage
        budget_limit: Budget limit
        utilization_percentage: Usage as percentage of limit
        message: Human-readable warning message
        metadata: Additional event-specific data
    """

    timestamp: datetime
    level: WarningLevel
    tool_name: Optional[str]
    current_usage: int
    budget_limit: int
    utilization_percentage: float
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "tool_name": self.tool_name,
            "current_usage": self.current_usage,
            "budget_limit": self.budget_limit,
            "utilization_percentage": self.utilization_percentage,
            "message": self.message,
            "metadata": self.metadata,
        }


@dataclass
class ThrottleConfig:
    """
    Throttling configuration to prevent warning spam.

    Attributes:
        enabled: Whether throttling is active
        min_interval_seconds: Minimum time between warnings for same threshold
        max_warnings_per_minute: Maximum warnings allowed per minute
    """

    enabled: bool = True
    min_interval_seconds: float = 60.0  # 1 minute between same-level warnings
    max_warnings_per_minute: int = 5


# Type alias for warning callbacks
WarningCallback = Callable[[WarningEvent], None]


class BudgetWarningSystem:
    """
    Comprehensive budget warning and alerting system.

    Provides:
    - Configurable warning thresholds (INFO, WARNING, CRITICAL)
    - Multiple notification channels (logging, CLI, callbacks)
    - Per-tool and global budget monitoring
    - Warning event history and analytics
    - Throttling to prevent warning spam
    - Integration with TokenUsageTracker and ClaudeBudgetManager
    """

    def __init__(
        self,
        global_budget_limit: Optional[int] = None,
        per_tool_limits: Optional[Dict[str, int]] = None,
        thresholds: Optional[List[BudgetThreshold]] = None,
        throttle_config: Optional[ThrottleConfig] = None,
        enable_logging: bool = True,
        enable_cli_output: bool = False,
    ):
        """
        Initialize budget warning system.

        Args:
            global_budget_limit: Optional global token budget limit
            per_tool_limits: Optional per-tool budget limits
            thresholds: Custom warning thresholds (uses defaults if not provided)
            throttle_config: Throttling configuration (uses defaults if not provided)
            enable_logging: Enable logging notifications
            enable_cli_output: Enable CLI output notifications
        """
        self.global_budget_limit = global_budget_limit
        self.per_tool_limits = per_tool_limits or {}
        self.enable_logging = enable_logging
        self.enable_cli_output = enable_cli_output

        # Configure thresholds
        if thresholds is None:
            self.thresholds = [
                BudgetThreshold(WarningLevel.INFO, 75.0),
                BudgetThreshold(WarningLevel.WARNING, 85.0),
                BudgetThreshold(WarningLevel.CRITICAL, 95.0),
            ]
        else:
            self.thresholds = thresholds

        # Sort thresholds by percentage
        self.thresholds.sort(key=lambda t: t.percentage)

        # Throttling configuration
        self.throttle_config = throttle_config or ThrottleConfig()

        # Thread-safe state
        self._lock = Lock()
        self._warning_history: List[WarningEvent] = []
        self._triggered_thresholds: Dict[str, Set[WarningLevel]] = defaultdict(set)
        self._last_warning_times: Dict[str, Dict[WarningLevel, datetime]] = (
            defaultdict(dict)
        )
        self._warning_callbacks: List[WarningCallback] = []
        self._warnings_this_minute: List[datetime] = []

        logger.debug(
            f"Initialized BudgetWarningSystem "
            f"(global_limit: {global_budget_limit}, "
            f"thresholds: {len(self.thresholds)})"
        )

    def register_callback(self, callback: WarningCallback) -> None:
        """
        Register custom warning callback handler.

        Callbacks are invoked for every warning event after throttling.

        Args:
            callback: Function to call on warning events
        """
        with self._lock:
            self._warning_callbacks.append(callback)
            logger.debug(f"Registered warning callback: {callback.__name__}")

    def unregister_callback(self, callback: WarningCallback) -> None:
        """
        Unregister warning callback handler.

        Args:
            callback: Callback to remove
        """
        with self._lock:
            if callback in self._warning_callbacks:
                self._warning_callbacks.remove(callback)
                logger.debug(f"Unregistered warning callback: {callback.__name__}")

    def check_usage_tracker(
        self, tracker: TokenUsageTracker, tool_name: Optional[str] = None
    ) -> List[WarningEvent]:
        """
        Check TokenUsageTracker for budget threshold violations.

        Args:
            tracker: Token usage tracker to monitor
            tool_name: Specific tool to check (checks all if None)

        Returns:
            List of warning events triggered (may be empty)
        """
        warnings = []

        if tool_name:
            # Check specific tool
            tool_stats = tracker.get_tool_stats(tool_name)
            if tool_stats:
                tool_limit = self.per_tool_limits.get(tool_name)
                if tool_limit:
                    warnings.extend(
                        self._check_tool_usage(tool_name, tool_stats, tool_limit)
                    )
        else:
            # Check all tools
            for tool, stats in tracker.get_all_tool_stats().items():
                tool_limit = self.per_tool_limits.get(tool)
                if tool_limit:
                    warnings.extend(self._check_tool_usage(tool, stats, tool_limit))

        # Check global usage
        if self.global_budget_limit:
            total_tokens = tracker.get_total_tokens()
            warnings.extend(
                self._check_global_usage(total_tokens, self.global_budget_limit)
            )

        return warnings

    def check_claude_budget_manager(
        self, manager: ClaudeBudgetManager
    ) -> List[WarningEvent]:
        """
        Check ClaudeBudgetManager for session budget violations.

        Extends existing ClaudeBudgetManager warnings with additional
        notification channels and event tracking.

        Args:
            manager: Claude budget manager to monitor

        Returns:
            List of warning events triggered (may be empty)
        """
        warnings = []
        stats = manager.session_stats
        tool_name = f"claude_{manager.model.model_id}"

        # Check session budget
        warnings.extend(
            self._check_session_usage(tool_name, stats, stats.budget_limit)
        )

        return warnings

    def get_warning_history(
        self,
        tool_name: Optional[str] = None,
        level: Optional[WarningLevel] = None,
        since: Optional[datetime] = None,
    ) -> List[WarningEvent]:
        """
        Get warning event history with optional filtering.

        Args:
            tool_name: Filter by tool name (None for all)
            level: Filter by warning level (None for all)
            since: Filter events after this timestamp (None for all)

        Returns:
            List of matching warning events
        """
        with self._lock:
            events = self._warning_history.copy()

        # Apply filters
        if tool_name:
            events = [e for e in events if e.tool_name == tool_name]
        if level:
            events = [e for e in events if e.level == level]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    def get_warning_summary(self) -> Dict[str, Any]:
        """
        Generate summary of warning activity.

        Returns:
            Dictionary with warning statistics
        """
        with self._lock:
            total_warnings = len(self._warning_history)

            # Count by level
            by_level = defaultdict(int)
            for event in self._warning_history:
                by_level[event.level.value] += 1

            # Count by tool
            by_tool = defaultdict(int)
            for event in self._warning_history:
                tool = event.tool_name or "global"
                by_tool[tool] += 1

            # Recent warnings (last hour)
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_warnings = [
                e for e in self._warning_history if e.timestamp >= one_hour_ago
            ]

            return {
                "total_warnings": total_warnings,
                "warnings_by_level": dict(by_level),
                "warnings_by_tool": dict(by_tool),
                "recent_warnings_1h": len(recent_warnings),
                "thresholds_configured": [
                    {
                        "level": t.level.value,
                        "percentage": t.percentage,
                        "enabled": t.enabled,
                    }
                    for t in self.thresholds
                ],
                "callbacks_registered": len(self._warning_callbacks),
            }

    def clear_history(self, tool_name: Optional[str] = None) -> int:
        """
        Clear warning history.

        Args:
            tool_name: Clear history for specific tool (None for all)

        Returns:
            Number of events cleared
        """
        with self._lock:
            if tool_name:
                # Clear specific tool
                original_count = len(self._warning_history)
                self._warning_history = [
                    e for e in self._warning_history if e.tool_name != tool_name
                ]
                cleared = original_count - len(self._warning_history)

                # Clear triggered thresholds for tool
                if tool_name in self._triggered_thresholds:
                    del self._triggered_thresholds[tool_name]
                if tool_name in self._last_warning_times:
                    del self._last_warning_times[tool_name]
            else:
                # Clear everything
                cleared = len(self._warning_history)
                self._warning_history.clear()
                self._triggered_thresholds.clear()
                self._last_warning_times.clear()

        logger.debug(f"Cleared {cleared} warning events")
        return cleared

    def _check_tool_usage(
        self, tool_name: str, stats: ToolUsageStats, limit: int
    ) -> List[WarningEvent]:
        """
        Check tool usage against budget limit.

        Args:
            tool_name: Tool identifier
            stats: Tool usage statistics
            limit: Budget limit for this tool

        Returns:
            List of triggered warning events
        """
        if limit <= 0:
            return []

        utilization = (stats.total_tokens / limit) * 100.0
        return self._check_thresholds(tool_name, stats.total_tokens, limit, utilization)

    def _check_session_usage(
        self, tool_name: str, stats: SessionUsageStats, limit: int
    ) -> List[WarningEvent]:
        """
        Check session usage against budget limit.

        Args:
            tool_name: Tool identifier
            stats: Session usage statistics
            limit: Budget limit

        Returns:
            List of triggered warning events
        """
        if limit <= 0:
            return []

        utilization = stats.utilization_percentage()
        return self._check_thresholds(
            tool_name, stats.total_tokens_used, limit, utilization
        )

    def _check_global_usage(self, total_tokens: int, limit: int) -> List[WarningEvent]:
        """
        Check global usage against budget limit.

        Args:
            total_tokens: Total tokens used
            limit: Global budget limit

        Returns:
            List of triggered warning events
        """
        if limit <= 0:
            return []

        utilization = (total_tokens / limit) * 100.0
        return self._check_thresholds(None, total_tokens, limit, utilization)

    def _check_thresholds(
        self,
        tool_name: Optional[str],
        current_usage: int,
        limit: int,
        utilization: float,
    ) -> List[WarningEvent]:
        """
        Check usage against configured thresholds.

        Args:
            tool_name: Tool name (None for global)
            current_usage: Current token usage
            limit: Budget limit
            utilization: Utilization percentage

        Returns:
            List of triggered warning events
        """
        warnings = []
        scope = tool_name or "global"

        with self._lock:
            for threshold in self.thresholds:
                if not threshold.enabled:
                    continue

                # Check if threshold crossed
                if utilization >= threshold.percentage:
                    # Check if already triggered
                    if threshold.level in self._triggered_thresholds[scope]:
                        continue

                    # Check throttling
                    if self._is_throttled(scope, threshold.level):
                        continue

                    # Create warning event
                    event = self._create_warning_event(
                        threshold.level,
                        tool_name,
                        current_usage,
                        limit,
                        utilization,
                    )

                    # Record event
                    self._warning_history.append(event)
                    self._triggered_thresholds[scope].add(threshold.level)
                    self._last_warning_times[scope][threshold.level] = event.timestamp
                    warnings.append(event)

                    # Notify
                    self._notify_warning(event)

        return warnings

    def _create_warning_event(
        self,
        level: WarningLevel,
        tool_name: Optional[str],
        current_usage: int,
        limit: int,
        utilization: float,
    ) -> WarningEvent:
        """
        Create warning event record.

        Args:
            level: Warning severity level
            tool_name: Tool name (None for global)
            current_usage: Current token usage
            limit: Budget limit
            utilization: Utilization percentage

        Returns:
            WarningEvent instance
        """
        scope = tool_name or "global"
        remaining = limit - current_usage

        message = (
            f"{level.value.upper()} - {scope} budget at {utilization:.1f}% "
            f"({current_usage:,} / {limit:,} tokens, "
            f"{remaining:,} remaining)"
        )

        return WarningEvent(
            timestamp=datetime.now(timezone.utc),
            level=level,
            tool_name=tool_name,
            current_usage=current_usage,
            budget_limit=limit,
            utilization_percentage=utilization,
            message=message,
            metadata={
                "tokens_remaining": remaining,
                "scope": scope,
            },
        )

    def _is_throttled(self, scope: str, level: WarningLevel) -> bool:
        """
        Check if warning is throttled.

        Args:
            scope: Tool name or "global"
            level: Warning level

        Returns:
            True if throttled (should skip warning), False otherwise
        """
        if not self.throttle_config.enabled:
            return False

        now = datetime.now(timezone.utc)

        # Check per-minute limit
        one_minute_ago = now - timedelta(minutes=1)
        self._warnings_this_minute = [
            t for t in self._warnings_this_minute if t >= one_minute_ago
        ]

        if len(self._warnings_this_minute) >= self.throttle_config.max_warnings_per_minute:
            return True

        # Check minimum interval for same level
        last_time = self._last_warning_times[scope].get(level)
        if last_time:
            elapsed = (now - last_time).total_seconds()
            if elapsed < self.throttle_config.min_interval_seconds:
                return True

        # Not throttled - record this potential warning time
        self._warnings_this_minute.append(now)
        return False

    def _notify_warning(self, event: WarningEvent) -> None:
        """
        Send warning notifications through configured channels.

        Args:
            event: Warning event to notify
        """
        # Logging notification
        if self.enable_logging:
            log_func = {
                WarningLevel.INFO: logger.info,
                WarningLevel.WARNING: logger.warning,
                WarningLevel.CRITICAL: logger.error,
            }[event.level]
            log_func(event.message)

        # CLI output notification
        if self.enable_cli_output:
            prefix = {
                WarningLevel.INFO: "[INFO]",
                WarningLevel.WARNING: "[WARNING]",
                WarningLevel.CRITICAL: "[CRITICAL]",
            }[event.level]
            print(f"{prefix} {event.message}")

        # Custom callbacks
        for callback in self._warning_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Warning callback error: {e}", exc_info=True)
