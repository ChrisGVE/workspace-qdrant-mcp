"""
Session trigger handling for Claude Code context injection.

This module provides pre-session and post-session trigger mechanisms that activate
before and after Claude Code sessions to prepare context injection, perform cleanup,
and manage session lifecycle.
"""

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from ..memory import MemoryManager
from .claude_code_detector import ClaudeCodeDetector, ClaudeCodeSession
from .claude_md_injector import ClaudeMdInjector
from .formatters import FormatManager
from .llm_tool_detector import LLMToolDetector, LLMToolType, UnifiedLLMSession
from .rule_retrieval import RuleFilter
from .system_prompt_injector import SystemPromptConfig, SystemPromptInjector


class TriggerPhase(Enum):
    """Session trigger execution phases."""

    PRE_SESSION = "pre_session"  # Before session starts
    POST_SESSION = "post_session"  # After session ends
    ON_DEMAND = "on_demand"  # Manual refresh requested by user
    ON_RULE_UPDATE = "on_rule_update"  # When rules are updated
    ON_FILE_CHANGE = "on_file_change"  # When CLAUDE.md changes


class TriggerPriority(Enum):
    """Trigger execution priority."""

    CRITICAL = 100  # Must execute first (e.g., environment setup)
    HIGH = 75  # Important triggers (e.g., context injection)
    NORMAL = 50  # Standard triggers
    LOW = 25  # Optional triggers (e.g., cleanup)
    DEFERRED = 0  # Execute last


@dataclass
class TriggerResult:
    """
    Result of trigger execution.

    Attributes:
        success: Whether trigger executed successfully
        phase: Trigger phase that was executed
        trigger_name: Name of the trigger
        execution_time_ms: Execution time in milliseconds
        error: Error message if failed
        metadata: Additional result metadata
    """

    success: bool
    phase: TriggerPhase
    trigger_name: str
    execution_time_ms: float
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerContext:
    """
    Context information available to triggers.

    Attributes:
        session: Claude Code session information
        project_root: Project root directory
        memory_manager: MemoryManager instance
        trigger_metadata: Shared metadata between triggers
    """

    session: ClaudeCodeSession
    project_root: Path
    memory_manager: MemoryManager
    trigger_metadata: dict[str, Any] = field(default_factory=dict)


class SessionTrigger(ABC):
    """
    Abstract base class for session triggers.

    Triggers are hooks that execute at specific points in the session lifecycle.
    Subclasses implement the execute() method to perform specific actions.
    """

    def __init__(
        self,
        name: str,
        phase: TriggerPhase,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        enabled: bool = True,
    ):
        """
        Initialize the session trigger.

        Args:
            name: Trigger name for identification
            phase: When this trigger should execute
            priority: Execution priority
            enabled: Whether trigger is enabled
        """
        self.name = name
        self.phase = phase
        self.priority = priority
        self.enabled = enabled

    @abstractmethod
    async def execute(self, context: TriggerContext) -> TriggerResult:
        """
        Execute the trigger.

        Args:
            context: Trigger execution context

        Returns:
            TriggerResult with execution status

        Raises:
            Exception: If trigger execution fails critically
        """
        pass

    async def cleanup(self, context: TriggerContext) -> None:
        """
        Optional cleanup after trigger execution.

        Args:
            context: Trigger execution context
        """
        pass


class ClaudeMdFileTrigger(SessionTrigger):
    """
    Trigger that injects CLAUDE.md content before session starts.

    This trigger discovers and injects CLAUDE.md content into the session,
    writing formatted content to a target file for Claude Code to consume.
    """

    def __init__(
        self,
        output_path: Path,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
        priority: TriggerPriority = TriggerPriority.HIGH,
    ):
        """
        Initialize the CLAUDE.md file trigger.

        Args:
            output_path: Where to write injected content
            token_budget: Token budget for content
            filter: Optional filter for memory rules
            priority: Execution priority
        """
        super().__init__(
            name="claude_md_file_injection",
            phase=TriggerPhase.PRE_SESSION,
            priority=priority,
        )
        self.output_path = output_path
        self.token_budget = token_budget
        self.filter = filter
        self._injector: ClaudeMdInjector | None = None

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute CLAUDE.md file injection."""
        import time

        start_time = time.time()

        try:
            # Create injector
            self._injector = ClaudeMdInjector(
                memory_manager=context.memory_manager,
                enable_watching=False,  # Watching handled separately
            )

            # Inject content to file
            success = await self._injector.inject_to_file(
                output_path=self.output_path,
                project_root=context.project_root,
                token_budget=self.token_budget,
                filter=self.filter,
            )

            execution_time = (time.time() - start_time) * 1000

            if success:
                logger.info(f"Injected CLAUDE.md content to {self.output_path}")
                return TriggerResult(
                    success=True,
                    phase=self.phase,
                    trigger_name=self.name,
                    execution_time_ms=execution_time,
                    metadata={"output_path": str(self.output_path)},
                )
            else:
                return TriggerResult(
                    success=False,
                    phase=self.phase,
                    trigger_name=self.name,
                    execution_time_ms=execution_time,
                    error="Failed to inject content",
                )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"CLAUDE.md injection failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )

    async def cleanup(self, context: TriggerContext) -> None:
        """Cleanup injector resources."""
        if self._injector:
            self._injector.stop_watching()


class SystemPromptTrigger(SessionTrigger):
    """
    Trigger that generates system prompt for API/MCP injection.

    This trigger creates a formatted system prompt suitable for direct
    API injection or MCP context, storing it for later retrieval.
    """

    def __init__(
        self,
        config: SystemPromptConfig | None = None,
        filter: RuleFilter | None = None,
        output_path: Path | None = None,
        priority: TriggerPriority = TriggerPriority.HIGH,
    ):
        """
        Initialize the system prompt trigger.

        Args:
            config: System prompt configuration
            filter: Optional filter for memory rules
            output_path: Optional file path to write prompt
            priority: Execution priority
        """
        super().__init__(
            name="system_prompt_generation",
            phase=TriggerPhase.PRE_SESSION,
            priority=priority,
        )
        self.config = config or SystemPromptConfig()
        self.filter = filter
        self.output_path = output_path
        self._generated_prompt: str | None = None

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute system prompt generation."""
        import time

        start_time = time.time()

        try:
            # Create injector
            injector = SystemPromptInjector(memory_manager=context.memory_manager)

            # Generate system prompt
            self._generated_prompt = await injector.generate_system_prompt(
                config=self.config, filter=self.filter
            )

            # Optionally write to file
            if self.output_path and self._generated_prompt:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                self.output_path.write_text(self._generated_prompt, encoding="utf-8")
                logger.info(f"Wrote system prompt to {self.output_path}")

            # Store in context for other triggers
            context.trigger_metadata["system_prompt"] = self._generated_prompt

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"Generated system prompt ({len(self._generated_prompt)} chars)")
            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                metadata={
                    "prompt_length": len(self._generated_prompt),
                    "output_path": str(self.output_path) if self.output_path else None,
                },
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"System prompt generation failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def get_prompt(self) -> str | None:
        """
        Get the generated system prompt.

        Returns:
            Generated prompt or None if not executed
        """
        return self._generated_prompt


class CleanupTrigger(SessionTrigger):
    """
    Trigger that cleans up temporary files after session ends.

    This trigger removes temporary injection files, cleans up resources,
    and ensures no leaked state remains after session completion.
    """

    def __init__(
        self,
        cleanup_paths: list[Path] | None = None,
        priority: TriggerPriority = TriggerPriority.LOW,
    ):
        """
        Initialize the cleanup trigger.

        Args:
            cleanup_paths: List of paths to remove
            priority: Execution priority
        """
        super().__init__(
            name="session_cleanup",
            phase=TriggerPhase.POST_SESSION,
            priority=priority,
        )
        self.cleanup_paths = cleanup_paths or []

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute cleanup operations."""
        import time

        start_time = time.time()
        removed_files = []
        errors = []

        try:
            # Remove cleanup paths
            for path in self.cleanup_paths:
                try:
                    if path.exists():
                        if path.is_file():
                            path.unlink()
                            removed_files.append(str(path))
                            logger.debug(f"Removed file: {path}")
                        elif path.is_dir():
                            # Only remove if empty
                            if not any(path.iterdir()):
                                path.rmdir()
                                removed_files.append(str(path))
                                logger.debug(f"Removed empty directory: {path}")
                except Exception as e:
                    errors.append(f"{path}: {e}")
                    logger.warning(f"Failed to remove {path}: {e}")

            execution_time = (time.time() - start_time) * 1000

            success = len(errors) == 0
            logger.info(f"Cleanup completed: {len(removed_files)} items removed")

            return TriggerResult(
                success=success,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error="; ".join(errors) if errors else None,
                metadata={
                    "removed_files": removed_files,
                    "error_count": len(errors),
                },
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Cleanup failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def add_cleanup_path(self, path: Path) -> None:
        """
        Add a path to the cleanup list.

        Args:
            path: Path to clean up
        """
        if path not in self.cleanup_paths:
            self.cleanup_paths.append(path)


class CustomCallbackTrigger(SessionTrigger):
    """
    Trigger that executes a custom callback function.

    This trigger allows users to inject custom logic into the session
    lifecycle without subclassing SessionTrigger.
    """

    def __init__(
        self,
        name: str,
        callback: Callable[[TriggerContext], Any],
        phase: TriggerPhase = TriggerPhase.PRE_SESSION,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        is_async: bool = False,
    ):
        """
        Initialize the custom callback trigger.

        Args:
            name: Trigger name
            callback: Callback function to execute
            phase: Trigger phase
            priority: Execution priority
            is_async: Whether callback is async
        """
        super().__init__(name=name, phase=phase, priority=priority)
        self.callback = callback
        self.is_async = is_async

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute custom callback."""
        import time

        start_time = time.time()

        try:
            # Execute callback
            if self.is_async:
                result = await self.callback(context)
            else:
                result = self.callback(context)

            execution_time = (time.time() - start_time) * 1000

            logger.info(f"Custom trigger '{self.name}' executed successfully")
            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                metadata={"callback_result": result},
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Custom trigger '{self.name}' failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )


class PostUpdateTrigger(SessionTrigger):
    """
    Trigger that executes after rule or configuration updates.

    This trigger monitors for changes to CLAUDE.md files, memory rules,
    and project configuration, then triggers context refresh. Includes
    debouncing to prevent excessive triggers from frequent changes.
    """

    def __init__(
        self,
        debounce_seconds: float = 2.0,
        batch_window_seconds: float = 5.0,
        output_path: Path | None = None,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
        priority: TriggerPriority = TriggerPriority.HIGH,
    ):
        """
        Initialize the post-update trigger.

        Args:
            debounce_seconds: Minimum time between trigger executions
            batch_window_seconds: Time window to batch multiple changes
            output_path: Where to write updated content
            token_budget: Token budget for content
            filter: Optional filter for memory rules
            priority: Execution priority
        """
        super().__init__(
            name="post_update_refresh",
            phase=TriggerPhase.ON_RULE_UPDATE,
            priority=priority,
        )
        self.debounce_seconds = debounce_seconds
        self.batch_window_seconds = batch_window_seconds
        self.output_path = output_path
        self.token_budget = token_budget
        self.filter = filter

        # Tracking for debouncing and batching
        self._last_trigger_time: float | None = None
        self._pending_changes: list[str] = []
        self._batch_task: asyncio.Task | None = None

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute post-update refresh."""
        import time

        start_time = time.time()

        try:
            # Check debounce threshold
            if (
                self._last_trigger_time
                and (start_time - self._last_trigger_time) < self.debounce_seconds
            ):
                logger.debug(
                    f"Debouncing post-update trigger (last trigger was {start_time - self._last_trigger_time:.2f}s ago)"
                )
                execution_time = (time.time() - start_time) * 1000
                return TriggerResult(
                    success=True,
                    phase=self.phase,
                    trigger_name=self.name,
                    execution_time_ms=execution_time,
                    metadata={
                        "debounced": True,
                        "pending_changes": len(self._pending_changes),
                    },
                )

            # Perform refresh
            injector = ClaudeMdInjector(
                memory_manager=context.memory_manager,
                enable_watching=False,
            )

            output_path = self.output_path or (
                context.project_root / ".claude" / "context.md"
            )

            success = await injector.inject_to_file(
                output_path=output_path,
                project_root=context.project_root,
                token_budget=self.token_budget,
                filter=self.filter,
            )

            if not success:
                raise RuntimeError("Failed to refresh after update")

            # Update tracking
            self._last_trigger_time = time.time()
            changes_count = len(self._pending_changes)
            self._pending_changes = []

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Post-update refresh completed, processed {changes_count} changes"
            )

            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                metadata={
                    "changes_processed": changes_count,
                    "output_path": str(output_path),
                },
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Post-update refresh failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def record_change(self, change_description: str) -> None:
        """
        Record a change for batch processing.

        Args:
            change_description: Description of the change
        """
        self._pending_changes.append(change_description)
        logger.debug(
            f"Recorded change: {change_description}, "
            f"total pending: {len(self._pending_changes)}"
        )

    def get_pending_changes(self) -> list[str]:
        """
        Get list of pending changes.

        Returns:
            List of change descriptions
        """
        return self._pending_changes.copy()


class OnDemandRefreshTrigger(SessionTrigger):
    """
    Trigger for manual context refresh requests.

    This trigger handles user-initiated refresh requests via CLI commands,
    API endpoints, or keyboard shortcuts. It refreshes context injection
    immediately without waiting for pre-session hooks.
    """

    def __init__(
        self,
        refresh_type: str = "full",  # full, rules_only, context_only
        output_path: Path | None = None,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
        priority: TriggerPriority = TriggerPriority.HIGH,
    ):
        """
        Initialize the on-demand refresh trigger.

        Args:
            refresh_type: Type of refresh (full, rules_only, context_only)
            output_path: Where to write refreshed content
            token_budget: Token budget for content
            filter: Optional filter for memory rules
            priority: Execution priority
        """
        super().__init__(
            name=f"on_demand_refresh_{refresh_type}",
            phase=TriggerPhase.ON_DEMAND,
            priority=priority,
        )
        self.refresh_type = refresh_type
        self.output_path = output_path
        self.token_budget = token_budget
        self.filter = filter
        self._last_refresh_time: float | None = None
        self._refresh_count: int = 0

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute on-demand refresh."""
        import time

        start_time = time.time()

        try:
            # Check if this is a duplicate request (within 1 second)
            if (
                self._last_refresh_time
                and (start_time - self._last_refresh_time) < 1.0
            ):
                logger.warning(
                    f"Ignoring duplicate refresh request (last refresh was {start_time - self._last_refresh_time:.2f}s ago)"
                )
                execution_time = (time.time() - start_time) * 1000
                return TriggerResult(
                    success=True,
                    phase=self.phase,
                    trigger_name=self.name,
                    execution_time_ms=execution_time,
                    metadata={"skipped": True, "reason": "duplicate_request"},
                )

            # Perform refresh based on type
            if self.refresh_type in ("full", "rules_only"):
                # Refresh rules and context
                injector = ClaudeMdInjector(
                    memory_manager=context.memory_manager,
                    enable_watching=False,
                )

                output_path = self.output_path or (
                    context.project_root / ".claude" / "context.md"
                )

                success = await injector.inject_to_file(
                    output_path=output_path,
                    project_root=context.project_root,
                    token_budget=self.token_budget,
                    filter=self.filter,
                )

                if not success:
                    raise RuntimeError("Failed to refresh context")

            # Update refresh tracking
            self._last_refresh_time = time.time()
            self._refresh_count += 1

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"On-demand refresh completed ({self.refresh_type}), "
                f"refresh #{self._refresh_count}"
            )

            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                metadata={
                    "refresh_type": self.refresh_type,
                    "refresh_count": self._refresh_count,
                    "output_path": str(self.output_path) if self.output_path else None,
                },
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"On-demand refresh failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
            )

    def get_refresh_stats(self) -> dict[str, Any]:
        """
        Get refresh statistics.

        Returns:
            Dictionary with refresh statistics
        """
        import time

        return {
            "refresh_count": self._refresh_count,
            "last_refresh_time": self._last_refresh_time,
            "seconds_since_last_refresh": (
                time.time() - self._last_refresh_time
                if self._last_refresh_time
                else None
            ),
            "refresh_type": self.refresh_type,
        }


@dataclass
class TriggerEvent:
    """
    Detailed event record for trigger execution.

    Captures comprehensive information about each trigger execution
    for logging, debugging, and analytics.

    Attributes:
        timestamp: When event occurred
        trigger_name: Name of trigger
        phase: Execution phase
        event_type: Type of event (started, completed, failed, retrying)
        success: Whether execution succeeded
        execution_time_ms: Execution duration
        error: Error message if failed
        retry_count: Number of retries attempted
        metadata: Additional event data
    """
    timestamp: float
    trigger_name: str
    phase: TriggerPhase
    event_type: str  # started, completed, failed, retrying
    success: bool
    execution_time_ms: float = 0.0
    error: str | None = None
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TriggerHealthMetrics:
    """
    Health metrics for trigger monitoring.

    Tracks trigger reliability, performance, and failure patterns
    to identify issues and optimize trigger configuration.

    Attributes:
        trigger_name: Name of trigger
        total_executions: Total execution count
        successful_executions: Successful execution count
        failed_executions: Failed execution count
        total_retries: Total retry attempts
        average_execution_time_ms: Average execution time
        last_success_timestamp: Last successful execution
        last_failure_timestamp: Last failed execution
        consecutive_failures: Current failure streak
        failure_rate: Percentage of failed executions
    """
    trigger_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_retries: int = 0
    average_execution_time_ms: float = 0.0
    last_success_timestamp: float | None = None
    last_failure_timestamp: float | None = None
    consecutive_failures: int = 0
    failure_rate: float = 0.0

    def update_from_event(self, event: "TriggerEvent") -> None:
        """Update metrics from trigger event."""
        import time

        self.total_executions += 1

        if event.success:
            self.successful_executions += 1
            self.last_success_timestamp = time.time()
            self.consecutive_failures = 0
        else:
            self.failed_executions += 1
            self.last_failure_timestamp = time.time()
            self.consecutive_failures += 1

        self.total_retries += event.retry_count

        # Update average execution time (moving average)
        if self.average_execution_time_ms == 0:
            self.average_execution_time_ms = event.execution_time_ms
        else:
            self.average_execution_time_ms = (
                (self.average_execution_time_ms * (self.total_executions - 1) +
                 event.execution_time_ms) / self.total_executions
            )

        # Calculate failure rate
        if self.total_executions > 0:
            self.failure_rate = self.failed_executions / self.total_executions


@dataclass
class TriggerRetryPolicy:
    """
    Retry policy for failed triggers.

    Configures automatic retry behavior with exponential backoff
    to handle transient failures gracefully.

    Attributes:
        max_retries: Maximum retry attempts
        initial_delay_seconds: Initial retry delay
        max_delay_seconds: Maximum retry delay
        exponential_base: Exponential backoff multiplier
        jitter: Add random jitter to delays (0.0-1.0)
        retryable_errors: List of error patterns to retry
    """
    max_retries: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_errors: list[str] = field(default_factory=lambda: [
        "timeout",
        "connection",
        "temporary",
        "transient",
    ])

    def calculate_delay(self, retry_count: int) -> float:
        """Calculate delay for retry attempt."""
        import random

        # Exponential backoff
        delay = min(
            self.initial_delay_seconds * (self.exponential_base ** retry_count),
            self.max_delay_seconds
        )

        # Add jitter
        if self.jitter > 0:
            jitter_amount = delay * self.jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)

    def is_retryable(self, error: str) -> bool:
        """Check if error should be retried."""
        error_lower = error.lower()
        return any(pattern in error_lower for pattern in self.retryable_errors)


class TriggerEventLogger:
    """
    Event logger for trigger executions.

    Maintains detailed event log with timestamps, outcomes, and errors.
    Supports querying and filtering of event history.
    """

    def __init__(self, max_events: int = 1000):
        """
        Initialize event logger.

        Args:
            max_events: Maximum events to retain in memory
        """
        self._events: list[TriggerEvent] = []
        self._max_events = max_events

    def log_event(self, event: TriggerEvent) -> None:
        """
        Log a trigger event.

        Args:
            event: Event to log
        """
        self._events.append(event)

        # Trim to max size (keep most recent)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        # Log to logger
        level = "error" if not event.success else "info"
        logger.log(
            level.upper(),
            f"Trigger '{event.trigger_name}' {event.event_type}: "
            f"success={event.success}, time={event.execution_time_ms:.2f}ms"
            + (f", error={event.error}" if event.error else "")
        )

    def get_events(
        self,
        trigger_name: str | None = None,
        phase: TriggerPhase | None = None,
        success: bool | None = None,
        since_timestamp: float | None = None,
        limit: int | None = None,
    ) -> list[TriggerEvent]:
        """
        Query event history.

        Args:
            trigger_name: Filter by trigger name
            phase: Filter by phase
            success: Filter by success status
            since_timestamp: Filter events after timestamp
            limit: Maximum events to return

        Returns:
            Filtered list of events
        """
        events = self._events

        if trigger_name:
            events = [e for e in events if e.trigger_name == trigger_name]

        if phase:
            events = [e for e in events if e.phase == phase]

        if success is not None:
            events = [e for e in events if e.success == success]

        if since_timestamp:
            events = [e for e in events if e.timestamp >= since_timestamp]

        if limit:
            events = events[-limit:]

        return events

    def get_recent_failures(self, minutes: int = 60) -> list[TriggerEvent]:
        """
        Get recent failed events.

        Args:
            minutes: Look back this many minutes

        Returns:
            List of failed events
        """
        import time
        cutoff = time.time() - (minutes * 60)
        return self.get_events(success=False, since_timestamp=cutoff)

    def clear_events(self, before_timestamp: float | None = None) -> int:
        """
        Clear old events.

        Args:
            before_timestamp: Clear events before this timestamp

        Returns:
            Number of events cleared
        """
        if before_timestamp:
            original_count = len(self._events)
            self._events = [e for e in self._events if e.timestamp >= before_timestamp]
            return original_count - len(self._events)
        else:
            count = len(self._events)
            self._events = []
            return count


class TriggerHealthMonitor:
    """
    Health monitoring for triggers.

    Tracks trigger reliability, performance, and failure patterns.
    Provides health metrics and alerts for degraded triggers.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        failure_rate_threshold: float = 0.5,
    ):
        """
        Initialize health monitor.

        Args:
            failure_threshold: Alert after this many consecutive failures
            failure_rate_threshold: Alert when failure rate exceeds this
        """
        self._metrics: dict[str, TriggerHealthMetrics] = {}
        self._failure_threshold = failure_threshold
        self._failure_rate_threshold = failure_rate_threshold
        self._alerts: list[str] = []

    def record_event(self, event: TriggerEvent) -> None:
        """
        Record trigger event for health monitoring.

        Args:
            event: Event to record
        """
        if event.trigger_name not in self._metrics:
            self._metrics[event.trigger_name] = TriggerHealthMetrics(
                trigger_name=event.trigger_name
            )

        metrics = self._metrics[event.trigger_name]
        metrics.update_from_event(event)

        # Check for health issues
        self._check_health(metrics)

    def _check_health(self, metrics: TriggerHealthMetrics) -> None:
        """Check trigger health and generate alerts."""
        # Check consecutive failures
        if metrics.consecutive_failures >= self._failure_threshold:
            alert = (
                f"Trigger '{metrics.trigger_name}' has {metrics.consecutive_failures} "
                f"consecutive failures (threshold: {self._failure_threshold})"
            )
            if alert not in self._alerts:
                self._alerts.append(alert)
                logger.warning(alert)

        # Check failure rate
        if (metrics.total_executions >= 10 and
            metrics.failure_rate >= self._failure_rate_threshold):
            alert = (
                f"Trigger '{metrics.trigger_name}' has {metrics.failure_rate:.1%} "
                f"failure rate (threshold: {self._failure_rate_threshold:.1%})"
            )
            if alert not in self._alerts:
                self._alerts.append(alert)
                logger.warning(alert)

    def get_metrics(self, trigger_name: str | None = None) -> dict[str, TriggerHealthMetrics]:
        """
        Get health metrics.

        Args:
            trigger_name: Get metrics for specific trigger, or all if None

        Returns:
            Dictionary of trigger metrics
        """
        if trigger_name:
            if trigger_name in self._metrics:
                return {trigger_name: self._metrics[trigger_name]}
            return {}
        return self._metrics.copy()

    def get_unhealthy_triggers(self) -> list[str]:
        """
        Get list of unhealthy trigger names.

        Returns:
            List of trigger names with health issues
        """
        unhealthy = []

        for name, metrics in self._metrics.items():
            if (metrics.consecutive_failures >= self._failure_threshold or
                metrics.failure_rate >= self._failure_rate_threshold):
                unhealthy.append(name)

        return unhealthy

    def get_alerts(self, clear: bool = False) -> list[str]:
        """
        Get current health alerts.

        Args:
            clear: Clear alerts after retrieving

        Returns:
            List of alert messages
        """
        alerts = self._alerts.copy()
        if clear:
            self._alerts = []
        return alerts

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary across all triggers.

        Returns:
            Dictionary with summary statistics
        """
        if not self._metrics:
            return {
                "total_triggers": 0,
                "total_executions": 0,
                "overall_success_rate": 0.0,
                "average_execution_time_ms": 0.0,
            }

        total_executions = sum(m.total_executions for m in self._metrics.values())
        total_successes = sum(m.successful_executions for m in self._metrics.values())
        avg_time = sum(m.average_execution_time_ms for m in self._metrics.values()) / len(self._metrics)

        return {
            "total_triggers": len(self._metrics),
            "total_executions": total_executions,
            "overall_success_rate": total_successes / total_executions if total_executions > 0 else 0.0,
            "average_execution_time_ms": avg_time,
            "unhealthy_triggers": self.get_unhealthy_triggers(),
        }


class TriggerManager:
    """
    Manages session triggers and orchestrates execution.

    The TriggerManager handles:
    - Trigger registration and organization
    - Execution ordering by priority
    - Error handling and recovery
    - Session lifecycle coordination
    - Cleanup management
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        detector: ClaudeCodeDetector | None = None,
        retry_policy: TriggerRetryPolicy | None = None,
        enable_event_logging: bool = True,
        enable_health_monitoring: bool = True,
    ):
        """
        Initialize the trigger manager.

        Args:
            memory_manager: MemoryManager instance
            detector: ClaudeCodeDetector instance (created if not provided)
            retry_policy: Retry policy for failed triggers
            enable_event_logging: Enable detailed event logging
            enable_health_monitoring: Enable health monitoring
        """
        self.memory_manager = memory_manager
        self.detector = detector or ClaudeCodeDetector()
        self._triggers: dict[TriggerPhase, list[SessionTrigger]] = {
            phase: [] for phase in TriggerPhase
        }
        self._execution_history: list[TriggerResult] = []

        # Event logging and monitoring
        self._event_logger = TriggerEventLogger() if enable_event_logging else None
        self._health_monitor = TriggerHealthMonitor() if enable_health_monitoring else None
        self._retry_policy = retry_policy or TriggerRetryPolicy()

    def register_trigger(self, trigger: SessionTrigger) -> None:
        """
        Register a trigger for execution.

        Args:
            trigger: Trigger to register
        """
        phase_triggers = self._triggers[trigger.phase]
        if trigger not in phase_triggers:
            phase_triggers.append(trigger)
            logger.debug(f"Registered trigger '{trigger.name}' for phase {trigger.phase.value}")

    def unregister_trigger(self, trigger: SessionTrigger) -> bool:
        """
        Unregister a trigger.

        Args:
            trigger: Trigger to unregister

        Returns:
            True if trigger was found and removed, False otherwise
        """
        phase_triggers = self._triggers[trigger.phase]
        if trigger in phase_triggers:
            phase_triggers.remove(trigger)
            logger.debug(f"Unregistered trigger '{trigger.name}'")
            return True
        return False

    def get_triggers(self, phase: TriggerPhase) -> list[SessionTrigger]:
        """
        Get all triggers for a specific phase.

        Args:
            phase: Trigger phase

        Returns:
            List of triggers sorted by priority (highest first)
        """
        triggers = [t for t in self._triggers[phase] if t.enabled]
        return sorted(triggers, key=lambda t: t.priority.value, reverse=True)

    async def execute_phase(
        self,
        phase: TriggerPhase,
        project_root: Path | None = None,
        fail_fast: bool = False,
        enable_retry: bool = True,
    ) -> list[TriggerResult]:
        """
        Execute all triggers for a specific phase.

        Args:
            phase: Phase to execute
            project_root: Project root directory (default: current directory)
            fail_fast: Stop execution on first failure
            enable_retry: Enable retry logic for failed triggers

        Returns:
            List of TriggerResult objects
        """

        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root).resolve()

        # Detect session
        session = self.detector.detect()

        # Create context
        context = TriggerContext(
            session=session,
            project_root=project_root,
            memory_manager=self.memory_manager,
        )

        # Get triggers for this phase
        triggers = self.get_triggers(phase)

        if not triggers:
            logger.debug(f"No triggers registered for phase {phase.value}")
            return []

        logger.info(f"Executing {len(triggers)} triggers for phase {phase.value}")

        results = []
        for trigger in triggers:
            # Execute trigger with retry logic
            result = await self._execute_trigger_with_retry(
                trigger=trigger,
                context=context,
                phase=phase,
                enable_retry=enable_retry,
            )

            results.append(result)
            self._execution_history.append(result)

            if not result.success:
                logger.warning(
                    f"Trigger '{trigger.name}' failed: {result.error}"
                )
                if fail_fast:
                    logger.error("Stopping execution due to fail_fast=True")
                    break

        # Log summary
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Phase {phase.value} completed: {success_count}/{len(results)} succeeded"
        )

        return results

    async def _execute_trigger_with_retry(
        self,
        trigger: SessionTrigger,
        context: TriggerContext,
        phase: TriggerPhase,
        enable_retry: bool = True,
    ) -> TriggerResult:
        """
        Execute a trigger with retry logic.

        Args:
            trigger: Trigger to execute
            context: Trigger context
            phase: Execution phase
            enable_retry: Enable retry logic

        Returns:
            TriggerResult with retry information
        """
        import time

        retry_count = 0
        last_error = None

        # Log start event
        if self._event_logger:
            start_event = TriggerEvent(
                timestamp=time.time(),
                trigger_name=trigger.name,
                phase=phase,
                event_type="started",
                success=True,
            )
            self._event_logger.log_event(start_event)

        while retry_count <= (self._retry_policy.max_retries if enable_retry else 0):
            try:
                # Execute trigger
                logger.debug(
                    f"Executing trigger '{trigger.name}'"
                    + (f" (retry {retry_count}/{self._retry_policy.max_retries})" if retry_count > 0 else "")
                )

                result = await trigger.execute(context)

                # Update retry count in metadata
                if retry_count > 0:
                    result.metadata["retry_count"] = retry_count

                # Log completion event
                if self._event_logger:
                    completion_event = TriggerEvent(
                        timestamp=time.time(),
                        trigger_name=trigger.name,
                        phase=phase,
                        event_type="completed" if result.success else "failed",
                        success=result.success,
                        execution_time_ms=result.execution_time_ms,
                        error=result.error,
                        retry_count=retry_count,
                        metadata=result.metadata,
                    )
                    self._event_logger.log_event(completion_event)

                    # Update health metrics
                    if self._health_monitor:
                        self._health_monitor.record_event(completion_event)

                # Return if successful or if error is not retryable
                if result.success:
                    return result

                if not enable_retry or not self._retry_policy.is_retryable(result.error or ""):
                    return result

                last_error = result.error
                retry_count += 1

                # If we've exhausted retries, return the last result
                if retry_count > self._retry_policy.max_retries:
                    return result

                # Log retry event
                if self._event_logger:
                    retry_event = TriggerEvent(
                        timestamp=time.time(),
                        trigger_name=trigger.name,
                        phase=phase,
                        event_type="retrying",
                        success=False,
                        error=last_error,
                        retry_count=retry_count,
                    )
                    self._event_logger.log_event(retry_event)

                # Wait before retry with exponential backoff
                delay = self._retry_policy.calculate_delay(retry_count - 1)
                logger.info(
                    f"Retrying trigger '{trigger.name}' in {delay:.2f}s "
                    f"(attempt {retry_count}/{self._retry_policy.max_retries})"
                )
                await asyncio.sleep(delay)

            except Exception as e:
                logger.error(f"Trigger '{trigger.name}' raised exception: {e}")
                last_error = str(e)

                # Log failure event
                if self._event_logger:
                    failure_event = TriggerEvent(
                        timestamp=time.time(),
                        trigger_name=trigger.name,
                        phase=phase,
                        event_type="failed",
                        success=False,
                        error=last_error,
                        retry_count=retry_count,
                    )
                    self._event_logger.log_event(failure_event)

                    # Update health metrics
                    if self._health_monitor:
                        self._health_monitor.record_event(failure_event)

                # Check if we should retry
                if not enable_retry or not self._retry_policy.is_retryable(last_error):
                    return TriggerResult(
                        success=False,
                        phase=phase,
                        trigger_name=trigger.name,
                        execution_time_ms=0,
                        error=last_error,
                        metadata={"retry_count": retry_count},
                    )

                retry_count += 1

                # If we've exhausted retries, return error result
                if retry_count > self._retry_policy.max_retries:
                    return TriggerResult(
                        success=False,
                        phase=phase,
                        trigger_name=trigger.name,
                        execution_time_ms=0,
                        error=last_error,
                        metadata={"retry_count": retry_count - 1},
                    )

                # Log retry event
                if self._event_logger:
                    retry_event = TriggerEvent(
                        timestamp=time.time(),
                        trigger_name=trigger.name,
                        phase=phase,
                        event_type="retrying",
                        success=False,
                        error=last_error,
                        retry_count=retry_count,
                    )
                    self._event_logger.log_event(retry_event)

                # Wait before retry
                delay = self._retry_policy.calculate_delay(retry_count - 1)
                logger.info(
                    f"Retrying trigger '{trigger.name}' in {delay:.2f}s "
                    f"(attempt {retry_count}/{self._retry_policy.max_retries})"
                )
                await asyncio.sleep(delay)

        # Should not reach here
        return TriggerResult(
            success=False,
            phase=phase,
            trigger_name=trigger.name,
            execution_time_ms=0,
            error=last_error or "Unknown error",
            metadata={"retry_count": self._retry_policy.max_retries},
        )

    async def trigger_manual_refresh(
        self,
        project_root: Path | None = None,
        refresh_type: str = "full",
        fail_fast: bool = False,
    ) -> list[TriggerResult]:
        """
        Trigger manual refresh of context injection.

        This method provides on-demand refresh functionality for users
        who want to update context without restarting their session.

        Args:
            project_root: Project root directory (default: current directory)
            refresh_type: Type of refresh (full, rules_only, context_only)
            fail_fast: Stop execution on first failure

        Returns:
            List of TriggerResult objects

        Example:
            >>> manager = TriggerManager(memory_manager)
            >>> results = await manager.trigger_manual_refresh()
        """
        # Execute ON_DEMAND triggers
        return await self.execute_phase(
            TriggerPhase.ON_DEMAND,
            project_root=project_root,
            fail_fast=fail_fast,
        )

    async def execute_on_demand(
        self,
        project_root: Path | None = None,
        fail_fast: bool = False,
    ) -> list[TriggerResult]:
        """
        Execute on-demand triggers (alias for trigger_manual_refresh).

        Args:
            project_root: Project root directory
            fail_fast: Stop execution on first failure

        Returns:
            List of TriggerResult objects
        """
        return await self.trigger_manual_refresh(
            project_root=project_root,
            refresh_type="full",
            fail_fast=fail_fast,
        )

    async def execute_session_lifecycle(
        self,
        project_root: Path | None = None,
        fail_fast: bool = False,
    ) -> dict[TriggerPhase, list[TriggerResult]]:
        """
        Execute complete session lifecycle (pre-session → post-session).

        Args:
            project_root: Project root directory
            fail_fast: Stop execution on first failure

        Returns:
            Dictionary mapping phase to list of results
        """
        results = {}

        # Execute pre-session triggers
        logger.info("Starting pre-session triggers")
        results[TriggerPhase.PRE_SESSION] = await self.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=project_root,
            fail_fast=fail_fast,
        )

        # Execute post-session triggers (regardless of pre-session failures)
        logger.info("Starting post-session triggers")
        results[TriggerPhase.POST_SESSION] = await self.execute_phase(
            TriggerPhase.POST_SESSION,
            project_root=project_root,
            fail_fast=False,  # Always try to cleanup
        )

        return results

    async def cleanup_all(self, project_root: Path | None = None) -> None:
        """
        Execute cleanup on all registered triggers.

        Args:
            project_root: Project root directory
        """
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root).resolve()

        session = self.detector.detect()
        context = TriggerContext(
            session=session,
            project_root=project_root,
            memory_manager=self.memory_manager,
        )

        logger.info("Executing cleanup on all triggers")

        for phase_triggers in self._triggers.values():
            for trigger in phase_triggers:
                try:
                    await trigger.cleanup(context)
                except Exception as e:
                    logger.warning(
                        f"Cleanup failed for trigger '{trigger.name}': {e}"
                    )

    def get_execution_history(self) -> list[TriggerResult]:
        """
        Get history of all trigger executions.

        Returns:
            List of TriggerResult objects
        """
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()

    # Event logging and monitoring methods

    def get_event_logger(self) -> TriggerEventLogger | None:
        """
        Get event logger instance.

        Returns:
            TriggerEventLogger if enabled, None otherwise
        """
        return self._event_logger

    def get_health_monitor(self) -> TriggerHealthMonitor | None:
        """
        Get health monitor instance.

        Returns:
            TriggerHealthMonitor if enabled, None otherwise
        """
        return self._health_monitor

    def get_recent_events(
        self,
        trigger_name: str | None = None,
        phase: TriggerPhase | None = None,
        minutes: int = 60,
        limit: int | None = 100,
    ) -> list[TriggerEvent]:
        """
        Get recent trigger events.

        Args:
            trigger_name: Filter by trigger name
            phase: Filter by phase
            minutes: Look back this many minutes
            limit: Maximum events to return

        Returns:
            List of recent events
        """
        if not self._event_logger:
            return []

        import time
        since_timestamp = time.time() - (minutes * 60)

        return self._event_logger.get_events(
            trigger_name=trigger_name,
            phase=phase,
            since_timestamp=since_timestamp,
            limit=limit,
        )

    def get_recent_failures(self, minutes: int = 60) -> list[TriggerEvent]:
        """
        Get recent failed trigger events.

        Args:
            minutes: Look back this many minutes

        Returns:
            List of failed events
        """
        if not self._event_logger:
            return []

        return self._event_logger.get_recent_failures(minutes)

    def get_health_metrics(
        self,
        trigger_name: str | None = None
    ) -> dict[str, TriggerHealthMetrics]:
        """
        Get health metrics for triggers.

        Args:
            trigger_name: Get metrics for specific trigger, or all if None

        Returns:
            Dictionary of trigger health metrics
        """
        if not self._health_monitor:
            return {}

        return self._health_monitor.get_metrics(trigger_name)

    def get_unhealthy_triggers(self) -> list[str]:
        """
        Get list of unhealthy trigger names.

        Returns:
            List of trigger names with health issues
        """
        if not self._health_monitor:
            return []

        return self._health_monitor.get_unhealthy_triggers()

    def get_health_alerts(self, clear: bool = False) -> list[str]:
        """
        Get current health alerts.

        Args:
            clear: Clear alerts after retrieving

        Returns:
            List of alert messages
        """
        if not self._health_monitor:
            return []

        return self._health_monitor.get_alerts(clear)

    def get_performance_summary(self) -> dict[str, Any]:
        """
        Get performance summary across all triggers.

        Returns:
            Dictionary with summary statistics including:
            - total_triggers: Number of unique triggers
            - total_executions: Total execution count
            - overall_success_rate: Success rate across all triggers
            - average_execution_time_ms: Average execution time
            - unhealthy_triggers: List of unhealthy trigger names
        """
        if not self._health_monitor:
            return {
                "total_triggers": 0,
                "total_executions": 0,
                "overall_success_rate": 0.0,
                "average_execution_time_ms": 0.0,
                "unhealthy_triggers": [],
            }

        return self._health_monitor.get_performance_summary()

    def clear_old_events(self, days: int = 7) -> int:
        """
        Clear events older than specified days.

        Args:
            days: Clear events older than this many days

        Returns:
            Number of events cleared
        """
        if not self._event_logger:
            return 0

        import time
        cutoff_timestamp = time.time() - (days * 24 * 60 * 60)
        return self._event_logger.clear_events(before_timestamp=cutoff_timestamp)


# Convenience functions for common patterns


async def prepare_claude_code_session(
    memory_manager: MemoryManager,
    project_root: Path | None = None,
    output_path: Path | None = None,
    token_budget: int = 50000,
) -> list[TriggerResult]:
    """
    Convenience function to prepare Claude Code session with file injection.

    Args:
        memory_manager: MemoryManager instance
        project_root: Project root directory
        output_path: Where to write injected content (default: .claude/context.md)
        token_budget: Token budget for content

    Returns:
        List of TriggerResult objects

    Example:
        >>> from context_injection import prepare_claude_code_session
        >>> results = await prepare_claude_code_session(memory_manager)
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if output_path is None:
        output_path = project_root / ".claude" / "context.md"

    # Create trigger manager
    manager = TriggerManager(memory_manager)

    # Register file injection trigger
    file_trigger = ClaudeMdFileTrigger(
        output_path=output_path,
        token_budget=token_budget,
    )
    manager.register_trigger(file_trigger)

    # Execute pre-session triggers
    results = await manager.execute_phase(
        TriggerPhase.PRE_SESSION,
        project_root=project_root,
    )

    return results


async def cleanup_claude_code_session(
    memory_manager: MemoryManager,
    cleanup_paths: list[Path] | None = None,
    project_root: Path | None = None,
) -> list[TriggerResult]:
    """
    Convenience function to cleanup after Claude Code session.

    Args:
        memory_manager: MemoryManager instance
        cleanup_paths: Paths to remove during cleanup
        project_root: Project root directory

    Returns:
        List of TriggerResult objects

    Example:
        >>> from context_injection import cleanup_claude_code_session
        >>> results = await cleanup_claude_code_session(
        ...     memory_manager,
        ...     cleanup_paths=[Path(".claude/context.md")]
        ... )
    """
    if project_root is None:
        project_root = Path.cwd()

    # Create trigger manager
    manager = TriggerManager(memory_manager)

    # Register cleanup trigger
    cleanup_trigger = CleanupTrigger(cleanup_paths=cleanup_paths or [])
    manager.register_trigger(cleanup_trigger)

    # Execute post-session triggers
    results = await manager.execute_phase(
        TriggerPhase.POST_SESSION,
        project_root=project_root,
    )

    return results


class ToolAwareTrigger(SessionTrigger):
    """
    Tool-aware trigger wrapper that automatically detects the LLM tool
    and applies appropriate formatting.

    This trigger uses LLMToolDetector to identify which LLM tool is active
    (Claude Code, Copilot, Cursor, etc.) and routes formatting through
    FormatManager to apply tool-specific formatting rules.

    Features:
    - Automatic tool detection
    - Tool-specific formatting via FormatManager
    - Dynamic tool switching
    - Formatter validation
    - Tool capabilities awareness
    """

    def __init__(
        self,
        name: str,
        phase: TriggerPhase,
        output_path: Path | None = None,
        token_budget: int = 50000,
        filter: RuleFilter | None = None,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        force_tool_type: LLMToolType | None = None,
    ):
        """
        Initialize tool-aware trigger.

        Args:
            name: Trigger name
            phase: Execution phase
            output_path: Where to write formatted output
            token_budget: Token budget for formatting
            filter: Optional rule filter
            priority: Execution priority
            force_tool_type: Force specific tool type (bypass detection)
        """
        super().__init__(name=name, phase=phase, priority=priority)
        self.output_path = output_path
        self.token_budget = token_budget
        self.filter = filter
        self.force_tool_type = force_tool_type

        # Initialize managers
        self._format_manager = FormatManager()
        self._llm_detector = LLMToolDetector()
        self._current_tool_session: UnifiedLLMSession | None = None

    def _detect_tool(self) -> UnifiedLLMSession:
        """
        Detect active LLM tool.

        Returns:
            UnifiedLLMSession with detected tool information
        """
        if self.force_tool_type:
            # Forced tool type
            return UnifiedLLMSession(
                tool_type=self.force_tool_type,
                is_active=True,
                detection_method="forced_override",
            )

        # Auto-detect tool
        return self._llm_detector.detect()

    def _get_tool_name_for_formatter(self, tool_type: LLMToolType) -> str:
        """
        Map LLMToolType to FormatManager tool name.

        Args:
            tool_type: Detected tool type

        Returns:
            Tool name for FormatManager ("claude", "codex", "gemini")
        """
        tool_mapping = {
            LLMToolType.CLAUDE_CODE: "claude",
            LLMToolType.GITHUB_COPILOT: "codex",
            LLMToolType.CODEX_API: "codex",
            LLMToolType.GOOGLE_GEMINI: "gemini",
            LLMToolType.CURSOR: "codex",  # Cursor uses Codex-style formatting
            LLMToolType.JETBRAINS_AI: "codex",  # JetBrains AI uses Codex-style
            LLMToolType.TABNINE: "codex",  # Tabnine uses Codex-style
            LLMToolType.UNKNOWN: "claude",  # Default to Claude formatting
        }

        return tool_mapping.get(tool_type, "claude")

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """
        Execute tool-aware trigger with automatic tool detection.

        Args:
            context: Trigger execution context

        Returns:
            TriggerResult with tool-specific metadata
        """
        import time

        start_time = time.time()

        try:
            # Detect active tool
            tool_session = self._detect_tool()
            self._current_tool_session = tool_session

            if not tool_session.is_active:
                logger.info(f"No active LLM tool detected for trigger '{self.name}'")
                execution_time = (time.time() - start_time) * 1000
                return TriggerResult(
                    success=True,
                    phase=self.phase,
                    trigger_name=self.name,
                    execution_time_ms=execution_time,
                    metadata={
                        "tool_detected": False,
                        "skipped": True,
                        "reason": "no_active_tool",
                    },
                )

            logger.info(
                f"Tool-aware trigger '{self.name}' detected {tool_session.tool_type.value} "
                f"via {tool_session.detection_method}"
            )

            # Get rules from memory manager
            rules = await context.memory_manager.get_rules()

            if self.filter:
                rules = self.filter.apply(rules)

            # Get formatter for detected tool
            tool_name = self._get_tool_name_for_formatter(tool_session.tool_type)

            # Format rules using tool-specific formatter
            formatted_context = self._format_manager.format_for_tool(
                tool_name=tool_name,
                rules=rules,
                token_budget=self.token_budget,
                options={},
            )

            # Determine output path
            output_path = self.output_path or (
                context.project_root / ".claude" / "context.md"
            )

            # Write formatted content
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted_context.formatted_text, encoding="utf-8")

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"Tool-aware trigger '{self.name}' completed for {tool_session.tool_type.value} "
                f"({formatted_context.token_count:,} tokens)"
            )

            return TriggerResult(
                success=True,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                metadata={
                    "tool_type": tool_session.tool_type.value,
                    "tool_name": tool_name,
                    "detection_method": tool_session.detection_method,
                    "formatter_used": tool_name,
                    "rules_formatted": formatted_context.rules_formatted,
                    "rules_skipped": formatted_context.rules_skipped,
                    "token_count": formatted_context.token_count,
                    "output_path": str(output_path),
                },
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool-aware trigger '{self.name}' failed: {e}")
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=execution_time,
                error=str(e),
                metadata={
                    "tool_type": self._current_tool_session.tool_type.value if self._current_tool_session else "unknown",
                },
            )

    def get_current_tool_session(self) -> UnifiedLLMSession | None:
        """
        Get the currently detected tool session.

        Returns:
            UnifiedLLMSession if tool detected, None otherwise
        """
        return self._current_tool_session


async def refresh_claude_code_context(
    memory_manager: MemoryManager,
    project_root: Path | None = None,
    output_path: Path | None = None,
    token_budget: int = 50000,
    refresh_type: str = "full",
) -> list[TriggerResult]:
    """
    Convenience function to manually refresh Claude Code context.

    This function provides on-demand context refresh without restarting
    the session. Useful when rules or project files have changed and
    you want to update the injected context immediately.

    Args:
        memory_manager: MemoryManager instance
        project_root: Project root directory
        output_path: Where to write refreshed content (default: .claude/context.md)
        token_budget: Token budget for content
        refresh_type: Type of refresh (full, rules_only, context_only)

    Returns:
        List of TriggerResult objects

    Example:
        >>> from context_injection import refresh_claude_code_context
        >>> results = await refresh_claude_code_context(memory_manager)
        >>> if all(r.success for r in results):
        ...     print("Context refreshed successfully")
    """
    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    if output_path is None:
        output_path = project_root / ".claude" / "context.md"

    # Create trigger manager
    manager = TriggerManager(memory_manager)

    # Register on-demand refresh trigger
    refresh_trigger = OnDemandRefreshTrigger(
        refresh_type=refresh_type,
        output_path=output_path,
        token_budget=token_budget,
    )
    manager.register_trigger(refresh_trigger)

    # Execute on-demand triggers
    results = await manager.execute_on_demand(
        project_root=project_root,
    )

    return results
