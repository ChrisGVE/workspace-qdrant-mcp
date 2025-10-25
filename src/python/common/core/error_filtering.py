"""
Error Filtering and Alerting System

Provides advanced filtering capabilities and alerting system for error messages.
Supports multi-criteria filtering, threshold-based alerting, and flexible callback
integration for monitoring tools.

Features:
    - Multi-criteria filtering (severity, category, date range, acknowledgment)
    - Alert rules with threshold-based triggering
    - Callback registration for specific severity levels
    - Alert history tracking
    - Async callback support with error handling

Example:
    ```python
    from workspace_qdrant_mcp.core.error_filtering import ErrorFilterManager, ErrorFilter
    from workspace_qdrant_mcp.core.error_categorization import ErrorSeverity

    # Initialize manager
    manager = ErrorFilterManager(error_manager)
    await manager.initialize()

    # Filter errors
    filter = ErrorFilter(
        severity_levels=[ErrorSeverity.ERROR],
        date_range=(start_date, end_date),
        acknowledged_only=False
    )
    result = await manager.filter_errors(filter)

    # Create alert rule
    async def alert_callback(errors):
        print(f"Alert triggered: {len(errors)} errors found")

    rule = await manager.create_alert_rule(
        name="Critical errors",
        filter=ErrorFilter(severity_levels=[ErrorSeverity.ERROR]),
        alert_callback=alert_callback,
        threshold=10
    )

    # Check alerts
    triggered = await manager.check_alerts()
    ```
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from .error_categorization import ErrorCategory, ErrorSeverity
from .error_message_manager import ErrorMessage, ErrorMessageManager


@dataclass
class ErrorFilter:
    """
    Filter criteria for error messages.

    Attributes:
        severity_levels: List of severity levels to filter by (AND logic)
        categories: List of categories to filter by (AND logic)
        date_range: Tuple of (start_date, end_date) for filtering
        acknowledged_only: If True, only acknowledged errors
        unacknowledged_only: If True, only unacknowledged errors
    """
    severity_levels: list[ErrorSeverity] | None = None
    categories: list[ErrorCategory] | None = None
    date_range: tuple[datetime, datetime] | None = None
    acknowledged_only: bool = False
    unacknowledged_only: bool = False

    def __post_init__(self):
        """Validate filter criteria."""
        if self.acknowledged_only and self.unacknowledged_only:
            raise ValueError(
                "Cannot set both acknowledged_only and unacknowledged_only to True"
            )

        if self.date_range:
            if len(self.date_range) != 2:
                raise ValueError("date_range must be a tuple of (start_date, end_date)")
            start, end = self.date_range
            if start > end:
                raise ValueError("start_date must be before or equal to end_date")


@dataclass
class FilteredErrorResult:
    """
    Result of filtered error query.

    Attributes:
        errors: List of filtered error messages
        total_count: Total count of errors matching filter
        filter_applied: The filter that was applied
    """
    errors: list[ErrorMessage]
    total_count: int
    filter_applied: ErrorFilter


@dataclass
class AlertRule:
    """
    Alert rule configuration.

    Attributes:
        id: Unique identifier for the rule
        name: Human-readable name
        filter: Error filter to apply
        alert_callback: Async callback function to invoke when triggered
        threshold: Minimum number of errors to trigger alert (default: 1)
        enabled: Whether the rule is active
        created_at: When the rule was created
        last_triggered_at: When the rule was last triggered
    """
    id: str
    name: str
    filter: ErrorFilter
    alert_callback: Callable[[list[ErrorMessage]], Any]
    threshold: int = 1
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now())
    last_triggered_at: datetime | None = None

    def __post_init__(self):
        """Validate alert rule."""
        if self.threshold < 1:
            raise ValueError("threshold must be at least 1")
        if not callable(self.alert_callback):
            raise ValueError("alert_callback must be callable")


@dataclass
class TriggeredAlert:
    """
    Record of a triggered alert.

    Attributes:
        rule_id: ID of the rule that triggered
        rule_name: Name of the rule
        triggered_at: When the alert was triggered
        error_count: Number of errors that triggered the alert
        errors: List of errors that triggered the alert
    """
    rule_id: str
    rule_name: str
    triggered_at: datetime
    error_count: int
    errors: list[ErrorMessage]


class ErrorFilterManager:
    """
    Advanced error filtering and alerting manager.

    Provides multi-criteria filtering, alert rule management, callback registration,
    and alert history tracking.
    """

    def __init__(self, error_manager: ErrorMessageManager):
        """
        Initialize error filter manager.

        Args:
            error_manager: ErrorMessageManager instance for data access
        """
        self.error_manager = error_manager
        self._initialized = False
        self._alert_rules: dict[str, AlertRule] = {}
        self._alert_history: list[TriggeredAlert] = []
        self._severity_callbacks: dict[ErrorSeverity, list[Callable]] = {
            ErrorSeverity.ERROR: [],
            ErrorSeverity.WARNING: [],
            ErrorSeverity.INFO: []
        }
        self._next_rule_id = 1

    async def initialize(self):
        """Initialize the filter manager."""
        if self._initialized:
            return

        # Ensure error manager is initialized
        if not self.error_manager._initialized:
            await self.error_manager.initialize()

        self._initialized = True
        logger.info("Error filter manager initialized")

    async def close(self):
        """Close the filter manager."""
        if not self._initialized:
            return

        self._alert_rules.clear()
        self._alert_history.clear()
        self._severity_callbacks.clear()
        self._initialized = False
        logger.info("Error filter manager closed")

    async def filter_errors(
        self,
        filter: ErrorFilter,
        limit: int = 100,
        offset: int = 0
    ) -> FilteredErrorResult:
        """
        Filter errors based on multiple criteria.

        Args:
            filter: ErrorFilter with criteria to apply
            limit: Maximum number of results
            offset: Offset for pagination

        Returns:
            FilteredErrorResult with filtered errors

        Raises:
            ValueError: If filter criteria are invalid
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Build query parameters from filter
        errors: list[ErrorMessage] = []

        # Determine if we need to do multi-query or in-memory filtering
        use_category_filter = filter.categories and len(filter.categories) == 1

        # If multiple severity levels, need to query each and combine
        if filter.severity_levels and len(filter.severity_levels) > 1:
            # Query for each severity level
            for severity in filter.severity_levels:
                severity_errors = await self._query_with_criteria(
                    severity=severity.value,
                    category=filter.categories[0].value if use_category_filter else None,
                    date_range=filter.date_range,
                    acknowledged_only=filter.acknowledged_only,
                    unacknowledged_only=filter.unacknowledged_only,
                    limit=limit,
                    offset=offset
                )
                errors.extend(severity_errors)

            # Remove duplicates and sort by timestamp
            seen_ids = set()
            unique_errors = []
            for error in errors:
                if error.id not in seen_ids:
                    seen_ids.add(error.id)
                    unique_errors.append(error)

            # Sort by timestamp descending
            unique_errors.sort(key=lambda e: e.timestamp, reverse=True)

            # Apply limit
            errors = unique_errors[:limit]

        else:
            # Single or no severity level - direct query
            severity = filter.severity_levels[0].value if filter.severity_levels else None
            errors = await self._query_with_criteria(
                severity=severity,
                category=filter.categories[0].value if use_category_filter else None,
                date_range=filter.date_range,
                acknowledged_only=filter.acknowledged_only,
                unacknowledged_only=filter.unacknowledged_only,
                limit=limit,
                offset=offset
            )

        # For categories, need to filter in-memory if multiple
        if filter.categories and len(filter.categories) > 1:
            category_values = {cat.value for cat in filter.categories}
            errors = [e for e in errors if e.category.value in category_values]

        return FilteredErrorResult(
            errors=errors,
            total_count=len(errors),
            filter_applied=filter
        )

    async def _query_with_criteria(
        self,
        severity: str | None = None,
        category: str | None = None,
        date_range: tuple[datetime, datetime] | None = None,
        acknowledged_only: bool = False,
        unacknowledged_only: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> list[ErrorMessage]:
        """
        Query errors with specific criteria.

        Args:
            severity: Single severity level
            category: Single category value
            date_range: Date range tuple
            acknowledged_only: Filter for acknowledged only
            unacknowledged_only: Filter for unacknowledged only
            limit: Result limit
            offset: Result offset

        Returns:
            List of ErrorMessage instances
        """
        # Determine acknowledged filter
        acknowledged = None
        if acknowledged_only:
            acknowledged = True
        elif unacknowledged_only:
            acknowledged = False

        # Extract date range
        start_date = None
        end_date = None
        if date_range:
            start_date, end_date = date_range

        # Query via error manager
        return await self.error_manager.get_errors(
            severity=severity,
            category=category,
            acknowledged=acknowledged,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )

    async def create_alert_rule(
        self,
        name: str,
        filter: ErrorFilter,
        alert_callback: Callable[[list[ErrorMessage]], Any],
        threshold: int = 1
    ) -> AlertRule:
        """
        Create an alert rule.

        Args:
            name: Human-readable name for the rule
            filter: ErrorFilter to apply
            alert_callback: Callback function (can be async)
            threshold: Minimum number of errors to trigger alert

        Returns:
            Created AlertRule

        Raises:
            ValueError: If parameters are invalid
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        rule_id = f"rule_{self._next_rule_id}"
        self._next_rule_id += 1

        rule = AlertRule(
            id=rule_id,
            name=name,
            filter=filter,
            alert_callback=alert_callback,
            threshold=threshold
        )

        self._alert_rules[rule_id] = rule
        logger.info(f"Created alert rule: {rule_id} - {name} (threshold: {threshold})")

        return rule

    async def register_alert_callback(
        self,
        severity: ErrorSeverity,
        callback_func: Callable[[list[ErrorMessage]], Any]
    ) -> bool:
        """
        Register a callback for a specific severity level.

        Args:
            severity: Severity level to register for
            callback_func: Callback function (can be async)

        Returns:
            True if registered successfully

        Raises:
            ValueError: If severity is invalid or callback is not callable
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        if not callable(callback_func):
            raise ValueError("callback_func must be callable")

        if severity not in self._severity_callbacks:
            raise ValueError(f"Invalid severity: {severity}")

        self._severity_callbacks[severity].append(callback_func)
        logger.info(f"Registered callback for severity: {severity.value}")

        return True

    async def check_alerts(self) -> list[TriggeredAlert]:
        """
        Check all active alert rules and trigger callbacks.

        Returns:
            List of TriggeredAlert for rules that were triggered

        Raises:
            RuntimeError: If manager not initialized
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        triggered_alerts: list[TriggeredAlert] = []

        for _rule_id, rule in self._alert_rules.items():
            if not rule.enabled:
                continue

            # Filter errors based on rule criteria
            result = await self.filter_errors(rule.filter, limit=1000)

            # Check threshold
            if result.total_count >= rule.threshold:
                # Trigger alert
                triggered_alert = TriggeredAlert(
                    rule_id=rule.id,
                    rule_name=rule.name,
                    triggered_at=datetime.now(),
                    error_count=result.total_count,
                    errors=result.errors
                )

                # Update rule last triggered time
                rule.last_triggered_at = triggered_alert.triggered_at

                # Add to history
                self._alert_history.append(triggered_alert)
                triggered_alerts.append(triggered_alert)

                # Invoke callback
                await self._invoke_callback(
                    rule.alert_callback,
                    result.errors,
                    f"Alert rule '{rule.name}'"
                )

                # Also invoke severity-specific callbacks
                await self._invoke_severity_callbacks(result.errors)

                logger.info(
                    f"Alert triggered: {rule.name} "
                    f"({result.total_count} errors, threshold: {rule.threshold})"
                )

        return triggered_alerts

    async def _invoke_callback(
        self,
        callback: Callable,
        errors: list[ErrorMessage],
        callback_name: str
    ):
        """
        Safely invoke a callback with error handling.

        Args:
            callback: Callback function to invoke
            errors: List of errors to pass to callback
            callback_name: Name for logging purposes
        """
        try:
            # Check if callback is async
            if asyncio.iscoroutinefunction(callback):
                await callback(errors)
            else:
                callback(errors)
        except Exception as e:
            logger.error(
                f"Error in {callback_name} callback: {e}",
                exc_info=True
            )

    async def _invoke_severity_callbacks(self, errors: list[ErrorMessage]):
        """
        Invoke severity-specific callbacks for errors.

        Args:
            errors: List of errors to process
        """
        # Group errors by severity
        severity_groups: dict[ErrorSeverity, list[ErrorMessage]] = {}
        for error in errors:
            if error.severity not in severity_groups:
                severity_groups[error.severity] = []
            severity_groups[error.severity].append(error)

        # Invoke callbacks for each severity
        for severity, severity_errors in severity_groups.items():
            callbacks = self._severity_callbacks.get(severity, [])
            for callback in callbacks:
                await self._invoke_callback(
                    callback,
                    severity_errors,
                    f"Severity '{severity.value}'"
                )

    async def get_alert_history(self, limit: int = 100) -> list[TriggeredAlert]:
        """
        Get alert history.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of TriggeredAlert, most recent first
        """
        if not self._initialized:
            raise RuntimeError("Manager not initialized. Call initialize() first.")

        # Return most recent first
        return list(reversed(self._alert_history[-limit:]))

    def get_alert_rule(self, rule_id: str) -> AlertRule | None:
        """
        Get an alert rule by ID.

        Args:
            rule_id: Rule ID to retrieve

        Returns:
            AlertRule or None if not found
        """
        return self._alert_rules.get(rule_id)

    def list_alert_rules(self) -> list[AlertRule]:
        """
        List all alert rules.

        Returns:
            List of all alert rules
        """
        return list(self._alert_rules.values())

    async def enable_alert_rule(self, rule_id: str) -> bool:
        """
        Enable an alert rule.

        Args:
            rule_id: Rule ID to enable

        Returns:
            True if enabled, False if not found
        """
        rule = self._alert_rules.get(rule_id)
        if rule:
            rule.enabled = True
            logger.info(f"Enabled alert rule: {rule_id}")
            return True
        return False

    async def disable_alert_rule(self, rule_id: str) -> bool:
        """
        Disable an alert rule.

        Args:
            rule_id: Rule ID to disable

        Returns:
            True if disabled, False if not found
        """
        rule = self._alert_rules.get(rule_id)
        if rule:
            rule.enabled = False
            logger.info(f"Disabled alert rule: {rule_id}")
            return True
        return False

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """
        Delete an alert rule.

        Args:
            rule_id: Rule ID to delete

        Returns:
            True if deleted, False if not found
        """
        if rule_id in self._alert_rules:
            rule = self._alert_rules[rule_id]
            del self._alert_rules[rule_id]
            logger.info(f"Deleted alert rule: {rule_id} - {rule.name}")
            return True
        return False
