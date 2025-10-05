"""
Live refresh mechanism for rule updates during active Claude Code sessions.

This module provides intelligent live refresh capabilities that detect changes to
CLAUDE.md files and memory rules, then incrementally update the injected context
without requiring session restart.

Key Features:
- File watching for CLAUDE.md changes (leverages ClaudeMdInjector)
- Memory rule change detection via Qdrant collection monitoring
- Throttling and debouncing to prevent refresh storms
- Budget-aware refreshes with session state preservation
- Multiple refresh modes: Automatic, Periodic, Manual
- Integration with ClaudeBudgetManager for token tracking
- Uses SessionTrigger phases for consistent lifecycle management

Refresh Modes:
1. AUTOMATIC: Triggered by file/rule changes
2. PERIODIC: Time-based refresh at intervals
3. MANUAL: Explicit on-demand refresh via API call
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from ..memory import MemoryManager, MemoryRule
from .claude_budget_manager import ClaudeBudgetManager, SessionUsageStats
from .claude_md_injector import ClaudeMdInjector
from .rule_retrieval import RuleFilter, RuleRetrieval
from .session_trigger import (
    SessionTrigger,
    TriggerContext,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerResult,
)


class RefreshMode(Enum):
    """Refresh trigger modes."""

    AUTOMATIC = "automatic"  # Triggered by file/rule changes
    PERIODIC = "periodic"  # Time-based refresh at intervals
    MANUAL = "manual"  # Explicit on-demand refresh


@dataclass
class RefreshThrottleConfig:
    """
    Configuration for refresh throttling and debouncing.

    Attributes:
        debounce_seconds: Minimum time between change detection and refresh
        min_refresh_interval_seconds: Minimum time between consecutive refreshes
        max_refresh_rate_per_minute: Maximum number of refreshes per minute
        change_aggregation_window_seconds: Time window to aggregate multiple changes
    """

    debounce_seconds: float = 2.0
    min_refresh_interval_seconds: float = 5.0
    max_refresh_rate_per_minute: int = 10
    change_aggregation_window_seconds: float = 3.0


@dataclass
class RefreshState:
    """
    State tracking for refresh operations.

    Attributes:
        last_refresh_at: Timestamp of last successful refresh
        last_file_hash: Hash of CLAUDE.md content from last refresh
        last_rules_hash: Hash of memory rules from last refresh
        pending_changes: Set of pending change sources
        refresh_count: Total number of refreshes in current session
        refresh_count_last_minute: Number of refreshes in last minute
        refresh_history: List of refresh timestamps
        is_refreshing: Whether a refresh is currently in progress
    """

    last_refresh_at: Optional[datetime] = None
    last_file_hash: Optional[str] = None
    last_rules_hash: Optional[str] = None
    pending_changes: Set[str] = field(default_factory=set)
    refresh_count: int = 0
    refresh_count_last_minute: int = 0
    refresh_history: List[datetime] = field(default_factory=list)
    is_refreshing: bool = False


@dataclass
class RefreshResult:
    """
    Result of a refresh operation.

    Attributes:
        success: Whether refresh succeeded
        mode: Refresh mode that triggered this refresh
        execution_time_ms: Execution time in milliseconds
        changes_detected: List of detected changes (e.g., "claude_md", "memory_rules")
        rules_updated: Number of rules updated
        tokens_used: Tokens used in refresh
        error: Error message if failed
        metadata: Additional result metadata
    """

    success: bool
    mode: RefreshMode
    execution_time_ms: float
    changes_detected: List[str] = field(default_factory=list)
    rules_updated: int = 0
    tokens_used: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LiveRefreshManager:
    """
    Manages live refresh of context injection during active Claude Code sessions.

    This class orchestrates the detection and application of rule changes without
    requiring session restart, while preserving budget tracking and session state.

    Features:
    - File change detection for CLAUDE.md via ClaudeMdInjector
    - Memory rule change detection via Qdrant monitoring
    - Intelligent throttling to prevent refresh storms
    - Budget-aware refreshes respecting token limits
    - State preservation across refreshes
    - Multiple refresh modes (automatic, periodic, manual)
    - Integration with ClaudeBudgetManager and SessionTrigger

    Workflow:
    1. Monitor CLAUDE.md files and memory rules
    2. Detect changes (file modifications, new/updated rules)
    3. Throttle/debounce to aggregate changes
    4. Execute refresh with budget awareness
    5. Update injected content incrementally
    6. Preserve session state and budget tracking
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        budget_manager: Optional[ClaudeBudgetManager] = None,
        claude_md_injector: Optional[ClaudeMdInjector] = None,
        rule_retrieval: Optional[RuleRetrieval] = None,
        trigger_manager: Optional[TriggerManager] = None,
        throttle_config: Optional[RefreshThrottleConfig] = None,
        enable_file_watching: bool = True,
        enable_rule_monitoring: bool = True,
        enable_periodic_refresh: bool = False,
        periodic_interval_seconds: float = 300.0,
    ):
        """
        Initialize the live refresh manager.

        Args:
            memory_manager: MemoryManager instance for rule access
            budget_manager: Optional ClaudeBudgetManager for budget tracking
            claude_md_injector: Optional ClaudeMdInjector for file watching
            rule_retrieval: Optional RuleRetrieval for rule queries
            trigger_manager: Optional TriggerManager for lifecycle integration
            throttle_config: Optional throttle configuration
            enable_file_watching: Enable CLAUDE.md file watching
            enable_rule_monitoring: Enable memory rule change monitoring
            enable_periodic_refresh: Enable periodic time-based refresh
            periodic_interval_seconds: Interval for periodic refresh (if enabled)
        """
        self.memory_manager = memory_manager
        self.budget_manager = budget_manager
        self.rule_retrieval = rule_retrieval or RuleRetrieval(memory_manager)
        self.trigger_manager = trigger_manager

        # Create ClaudeMdInjector if not provided
        self.claude_md_injector = claude_md_injector or ClaudeMdInjector(
            memory_manager=memory_manager,
            rule_retrieval=self.rule_retrieval,
            enable_watching=enable_file_watching,
        )

        # Configuration
        self.throttle_config = throttle_config or RefreshThrottleConfig()
        self.enable_file_watching = enable_file_watching
        self.enable_rule_monitoring = enable_rule_monitoring
        self.enable_periodic_refresh = enable_periodic_refresh
        self.periodic_interval_seconds = periodic_interval_seconds

        # State tracking
        self.refresh_state = RefreshState()

        # Refresh callbacks
        self._refresh_callbacks: List[Callable[[RefreshResult], Any]] = []

        # Background tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        self._periodic_refresh_task: Optional[asyncio.Task] = None

        # Project context
        self._project_root: Optional[Path] = None
        self._output_path: Optional[Path] = None

        logger.info(
            f"Initialized LiveRefreshManager "
            f"(file_watching={enable_file_watching}, "
            f"rule_monitoring={enable_rule_monitoring}, "
            f"periodic={enable_periodic_refresh})"
        )

    async def start(
        self,
        project_root: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> bool:
        """
        Start live refresh monitoring.

        Args:
            project_root: Project root directory (default: current directory)
            output_path: Output path for injected content

        Returns:
            True if started successfully, False otherwise
        """
        if project_root is None:
            project_root = Path.cwd()
        else:
            project_root = Path(project_root).resolve()

        self._project_root = project_root
        self._output_path = output_path

        logger.info(f"Starting live refresh monitoring for {project_root}")

        # Start file watching
        if self.enable_file_watching:
            success = self.claude_md_injector.start_watching(
                project_root=project_root,
                callback=self._handle_file_change,
            )
            if not success:
                logger.warning("Failed to start file watching")
                return False

        # Start rule monitoring
        if self.enable_rule_monitoring:
            task = asyncio.create_task(self._monitor_memory_rules())
            self._monitoring_tasks.append(task)
            logger.debug("Started memory rule monitoring")

        # Start periodic refresh
        if self.enable_periodic_refresh:
            self._periodic_refresh_task = asyncio.create_task(
                self._periodic_refresh_loop()
            )
            logger.debug(
                f"Started periodic refresh (interval: {self.periodic_interval_seconds}s)"
            )

        # Initialize baseline hashes
        await self._initialize_baseline_hashes()

        logger.info("Live refresh monitoring started successfully")
        return True

    async def stop(self) -> None:
        """Stop live refresh monitoring."""
        logger.info("Stopping live refresh monitoring")

        # Stop file watching
        if self.enable_file_watching:
            self.claude_md_injector.stop_watching()

        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._monitoring_tasks.clear()

        # Cancel periodic refresh
        if self._periodic_refresh_task:
            self._periodic_refresh_task.cancel()
            try:
                await self._periodic_refresh_task
            except asyncio.CancelledError:
                pass
            self._periodic_refresh_task = None

        logger.info("Live refresh monitoring stopped")

    async def refresh_now(
        self,
        force: bool = False,
        mode: RefreshMode = RefreshMode.MANUAL,
    ) -> RefreshResult:
        """
        Execute immediate refresh, bypassing throttling.

        Args:
            force: If True, bypass all throttling and refresh immediately
            mode: Refresh mode for this operation

        Returns:
            RefreshResult with execution details
        """
        logger.info(f"Manual refresh requested (force={force}, mode={mode.value})")

        # Check if refresh is allowed
        if not force and not await self._should_refresh():
            logger.debug("Refresh throttled")
            return RefreshResult(
                success=False,
                mode=mode,
                execution_time_ms=0,
                error="Refresh throttled",
                metadata={"throttled": True},
            )

        # Execute refresh
        return await self._execute_refresh(mode=mode)

    def add_refresh_callback(
        self, callback: Callable[[RefreshResult], Any]
    ) -> None:
        """
        Add a callback to be notified of refresh events.

        Args:
            callback: Function that takes RefreshResult as argument
        """
        if callback not in self._refresh_callbacks:
            self._refresh_callbacks.append(callback)
            # Use getattr with default for __name__ to avoid AttributeError with MagicMock
            callback_name = getattr(callback, "__name__", repr(callback))
            logger.debug(f"Added refresh callback: {callback_name}")

    def get_refresh_stats(self) -> Dict[str, Any]:
        """
        Get refresh statistics.

        Returns:
            Dictionary with refresh statistics
        """
        now = datetime.now(timezone.utc)
        recent_refreshes = [
            ts for ts in self.refresh_state.refresh_history
            if now - ts < timedelta(minutes=1)
        ]

        return {
            "total_refreshes": self.refresh_state.refresh_count,
            "last_refresh_at": (
                self.refresh_state.last_refresh_at.isoformat()
                if self.refresh_state.last_refresh_at
                else None
            ),
            "refreshes_last_minute": len(recent_refreshes),
            "is_refreshing": self.refresh_state.is_refreshing,
            "pending_changes": list(self.refresh_state.pending_changes),
            "file_watching_active": (
                self.claude_md_injector._observer is not None
                and self.claude_md_injector._observer.is_alive()
                if self.enable_file_watching
                else False
            ),
            "rule_monitoring_active": len(self._monitoring_tasks) > 0,
            "periodic_refresh_active": (
                self._periodic_refresh_task is not None
                and not self._periodic_refresh_task.done()
            ),
        }

    # Private methods

    def _handle_file_change(self, file_path: Path) -> None:
        """
        Handle CLAUDE.md file change event.

        Args:
            file_path: Path to changed file
        """
        logger.info(f"File change detected: {file_path}")
        self.refresh_state.pending_changes.add("claude_md")

        # Schedule refresh via asyncio
        asyncio.create_task(
            self._debounced_refresh(mode=RefreshMode.AUTOMATIC, source="claude_md")
        )

    async def _monitor_memory_rules(self) -> None:
        """
        Background task to monitor memory rule changes.

        Polls the memory collection at intervals to detect new/updated rules.
        """
        logger.debug("Memory rule monitoring started")

        try:
            while True:
                # Sleep first to avoid immediate check
                await asyncio.sleep(self.periodic_interval_seconds)

                # Check for rule changes
                current_hash = await self._compute_rules_hash()
                if (
                    self.refresh_state.last_rules_hash
                    and current_hash != self.refresh_state.last_rules_hash
                ):
                    logger.info("Memory rule changes detected")
                    self.refresh_state.pending_changes.add("memory_rules")

                    # Schedule refresh
                    await self._debounced_refresh(
                        mode=RefreshMode.AUTOMATIC, source="memory_rules"
                    )

        except asyncio.CancelledError:
            logger.debug("Memory rule monitoring cancelled")
        except Exception as e:
            logger.error(f"Error in memory rule monitoring: {e}")

    async def _periodic_refresh_loop(self) -> None:
        """
        Background task for periodic time-based refresh.
        """
        logger.debug("Periodic refresh loop started")

        try:
            while True:
                await asyncio.sleep(self.periodic_interval_seconds)
                logger.debug("Periodic refresh triggered")
                await self._debounced_refresh(
                    mode=RefreshMode.PERIODIC, source="periodic"
                )

        except asyncio.CancelledError:
            logger.debug("Periodic refresh loop cancelled")
        except Exception as e:
            logger.error(f"Error in periodic refresh loop: {e}")

    async def _debounced_refresh(
        self, mode: RefreshMode, source: str
    ) -> None:
        """
        Execute refresh with debouncing.

        Waits for aggregation window to collect multiple changes before
        executing refresh.

        Args:
            mode: Refresh mode
            source: Source of the change
        """
        # Wait for aggregation window
        await asyncio.sleep(self.throttle_config.change_aggregation_window_seconds)

        # Check if we should refresh
        if await self._should_refresh():
            await self._execute_refresh(mode=mode)
        else:
            logger.debug(f"Refresh throttled for source: {source}")

    async def _should_refresh(self) -> bool:
        """
        Determine if refresh should proceed based on throttling rules.

        Returns:
            True if refresh should proceed, False if throttled
        """
        # Check if already refreshing
        if self.refresh_state.is_refreshing:
            logger.debug("Refresh already in progress")
            return False

        # Check minimum refresh interval
        if self.refresh_state.last_refresh_at:
            elapsed = (
                datetime.now(timezone.utc) - self.refresh_state.last_refresh_at
            ).total_seconds()
            if elapsed < self.throttle_config.min_refresh_interval_seconds:
                logger.debug(
                    f"Minimum refresh interval not met ({elapsed:.1f}s < "
                    f"{self.throttle_config.min_refresh_interval_seconds}s)"
                )
                return False

        # Check refresh rate limit (per minute)
        now = datetime.now(timezone.utc)
        recent_refreshes = [
            ts for ts in self.refresh_state.refresh_history
            if now - ts < timedelta(minutes=1)
        ]
        if len(recent_refreshes) >= self.throttle_config.max_refresh_rate_per_minute:
            logger.warning(
                f"Refresh rate limit reached "
                f"({len(recent_refreshes)}/{self.throttle_config.max_refresh_rate_per_minute} per minute)"
            )
            return False

        return True

    async def _execute_refresh(self, mode: RefreshMode) -> RefreshResult:
        """
        Execute the actual refresh operation.

        Args:
            mode: Refresh mode

        Returns:
            RefreshResult with execution details
        """
        start_time = time.time()
        self.refresh_state.is_refreshing = True

        try:
            logger.info(f"Executing refresh (mode={mode.value})")

            # Detect changes
            changes_detected = await self._detect_changes()

            if not changes_detected:
                logger.debug("No changes detected, skipping refresh")
                return RefreshResult(
                    success=True,
                    mode=mode,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    changes_detected=[],
                    metadata={"skipped": True},
                )

            # Re-inject content
            if self._output_path:
                success = await self.claude_md_injector.inject_to_file(
                    output_path=self._output_path,
                    project_root=self._project_root,
                    token_budget=(
                        self.budget_manager.session_stats.budget_limit
                        if self.budget_manager
                        else 50000
                    ),
                )
            else:
                # Just retrieve content without writing
                _ = await self.claude_md_injector.inject_from_files(
                    project_root=self._project_root,
                    token_budget=(
                        self.budget_manager.session_stats.budget_limit
                        if self.budget_manager
                        else 50000
                    ),
                )
                success = True

            if not success:
                raise Exception("Failed to inject content")

            # Update state
            await self._update_refresh_state()

            # Clear pending changes
            self.refresh_state.pending_changes.clear()

            execution_time_ms = (time.time() - start_time) * 1000

            result = RefreshResult(
                success=True,
                mode=mode,
                execution_time_ms=execution_time_ms,
                changes_detected=changes_detected,
                rules_updated=len(changes_detected),
                tokens_used=0,  # TODO: Track actual token usage
            )

            logger.info(
                f"Refresh completed successfully in {execution_time_ms:.1f}ms "
                f"(changes: {changes_detected})"
            )

            # Notify callbacks
            await self._notify_refresh_callbacks(result)

            # Trigger ON_RULE_UPDATE phase if using TriggerManager
            if self.trigger_manager and changes_detected:
                await self.trigger_manager.execute_phase(
                    phase=TriggerPhase.ON_RULE_UPDATE,
                    project_root=self._project_root,
                )

            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Refresh failed: {e}")

            result = RefreshResult(
                success=False,
                mode=mode,
                execution_time_ms=execution_time_ms,
                error=str(e),
            )

            # Notify callbacks even on failure
            await self._notify_refresh_callbacks(result)

            return result

        finally:
            self.refresh_state.is_refreshing = False

    async def _detect_changes(self) -> List[str]:
        """
        Detect what has changed since last refresh.

        Returns:
            List of change sources (e.g., ["claude_md", "memory_rules"])
        """
        changes = []

        # Check CLAUDE.md changes
        current_file_hash = await self._compute_file_hash()
        if (
            self.refresh_state.last_file_hash
            and current_file_hash != self.refresh_state.last_file_hash
        ):
            changes.append("claude_md")
            logger.debug("CLAUDE.md content changed")

        # Check memory rule changes
        current_rules_hash = await self._compute_rules_hash()
        if (
            self.refresh_state.last_rules_hash
            and current_rules_hash != self.refresh_state.last_rules_hash
        ):
            changes.append("memory_rules")
            logger.debug("Memory rules changed")

        return changes

    async def _initialize_baseline_hashes(self) -> None:
        """Initialize baseline hashes for change detection."""
        self.refresh_state.last_file_hash = await self._compute_file_hash()
        self.refresh_state.last_rules_hash = await self._compute_rules_hash()
        logger.debug("Initialized baseline hashes")

    async def _compute_file_hash(self) -> str:
        """
        Compute hash of CLAUDE.md files.

        Returns:
            Hash string
        """
        try:
            locations = self.claude_md_injector.discover_claude_md_files(
                self._project_root
            )
            if not locations:
                return ""

            # Concatenate all file contents
            content_parts = []
            for location in locations:
                content = location.path.read_text(encoding="utf-8")
                content_parts.append(content)

            combined_content = "\n".join(content_parts)
            return hashlib.sha256(combined_content.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Failed to compute file hash: {e}")
            return ""

    async def _compute_rules_hash(self) -> str:
        """
        Compute hash of memory rules.

        Returns:
            Hash string
        """
        try:
            # Get all memory rules
            filter = RuleFilter(limit=1000)
            result = await self.rule_retrieval.get_rules(filter)

            if not result.rules:
                return ""

            # Sort by ID for deterministic hash
            sorted_rules = sorted(result.rules, key=lambda r: r.id)

            # Concatenate rule content
            rule_parts = [
                f"{rule.id}:{rule.rule}:{rule.updated_at}"
                for rule in sorted_rules
            ]
            combined_rules = "\n".join(rule_parts)

            return hashlib.sha256(combined_rules.encode()).hexdigest()

        except Exception as e:
            logger.error(f"Failed to compute rules hash: {e}")
            return ""

    async def _update_refresh_state(self) -> None:
        """Update refresh state after successful refresh."""
        now = datetime.now(timezone.utc)

        # Update hashes
        self.refresh_state.last_file_hash = await self._compute_file_hash()
        self.refresh_state.last_rules_hash = await self._compute_rules_hash()

        # Update timestamps
        self.refresh_state.last_refresh_at = now
        self.refresh_state.refresh_count += 1
        self.refresh_state.refresh_history.append(now)

        # Trim history to last hour
        cutoff = now - timedelta(hours=1)
        self.refresh_state.refresh_history = [
            ts for ts in self.refresh_state.refresh_history if ts > cutoff
        ]

        logger.debug(
            f"Updated refresh state (total: {self.refresh_state.refresh_count})"
        )

    async def _notify_refresh_callbacks(self, result: RefreshResult) -> None:
        """
        Notify registered callbacks of refresh event.

        Args:
            result: RefreshResult to pass to callbacks
        """
        for callback in self._refresh_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                callback_name = getattr(callback, "__name__", repr(callback))
                logger.error(f"Error in refresh callback {callback_name}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        # Create event loop if needed for cleanup
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Schedule cleanup
                asyncio.create_task(self.stop())
            else:
                # Run cleanup synchronously
                loop.run_until_complete(self.stop())
        except Exception as e:
            logger.error(f"Error during context manager cleanup: {e}")


# Convenience function for simple refresh setup


async def start_live_refresh(
    memory_manager: MemoryManager,
    project_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    budget_manager: Optional[ClaudeBudgetManager] = None,
    enable_periodic: bool = False,
) -> LiveRefreshManager:
    """
    Convenience function to start live refresh monitoring.

    Args:
        memory_manager: MemoryManager instance
        project_root: Project root directory (default: current directory)
        output_path: Output path for injected content
        budget_manager: Optional ClaudeBudgetManager
        enable_periodic: Enable periodic time-based refresh

    Returns:
        LiveRefreshManager instance (already started)

    Example:
        >>> from context_injection import start_live_refresh
        >>> refresh_manager = await start_live_refresh(
        ...     memory_manager,
        ...     output_path=Path(".claude/context.md")
        ... )
    """
    if project_root is None:
        project_root = Path.cwd()

    manager = LiveRefreshManager(
        memory_manager=memory_manager,
        budget_manager=budget_manager,
        enable_file_watching=True,
        enable_rule_monitoring=True,
        enable_periodic_refresh=enable_periodic,
    )

    await manager.start(project_root=project_root, output_path=output_path)

    logger.info("Live refresh started successfully")
    return manager
