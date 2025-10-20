"""
Session trigger handling for Claude Code context injection.

This module provides pre-session and post-session trigger mechanisms that activate
before and after Claude Code sessions to prepare context injection, perform cleanup,
and manage session lifecycle.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from loguru import logger

from ..memory import MemoryManager
from .claude_code_detector import ClaudeCodeDetector, ClaudeCodeSession
from .claude_md_injector import ClaudeMdInjector
from .system_prompt_injector import SystemPromptInjector, SystemPromptConfig
from .rule_retrieval import RuleFilter


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
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    trigger_metadata: Dict[str, Any] = field(default_factory=dict)


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
        filter: Optional[RuleFilter] = None,
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
        self._injector: Optional[ClaudeMdInjector] = None

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
        config: Optional[SystemPromptConfig] = None,
        filter: Optional[RuleFilter] = None,
        output_path: Optional[Path] = None,
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
        self._generated_prompt: Optional[str] = None

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

    def get_prompt(self) -> Optional[str]:
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
        cleanup_paths: Optional[List[Path]] = None,
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
        output_path: Optional[Path] = None,
        token_budget: int = 50000,
        filter: Optional[RuleFilter] = None,
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
        self._last_trigger_time: Optional[float] = None
        self._pending_changes: List[str] = []
        self._batch_task: Optional[asyncio.Task] = None

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

    def get_pending_changes(self) -> List[str]:
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
        output_path: Optional[Path] = None,
        token_budget: int = 50000,
        filter: Optional[RuleFilter] = None,
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
        self._last_refresh_time: Optional[float] = None
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

    def get_refresh_stats(self) -> Dict[str, Any]:
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
        detector: Optional[ClaudeCodeDetector] = None,
    ):
        """
        Initialize the trigger manager.

        Args:
            memory_manager: MemoryManager instance
            detector: ClaudeCodeDetector instance (created if not provided)
        """
        self.memory_manager = memory_manager
        self.detector = detector or ClaudeCodeDetector()
        self._triggers: Dict[TriggerPhase, List[SessionTrigger]] = {
            phase: [] for phase in TriggerPhase
        }
        self._execution_history: List[TriggerResult] = []

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

    def get_triggers(self, phase: TriggerPhase) -> List[SessionTrigger]:
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
        project_root: Optional[Path] = None,
        fail_fast: bool = False,
    ) -> List[TriggerResult]:
        """
        Execute all triggers for a specific phase.

        Args:
            phase: Phase to execute
            project_root: Project root directory (default: current directory)
            fail_fast: Stop execution on first failure

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
            try:
                logger.debug(f"Executing trigger '{trigger.name}'")
                result = await trigger.execute(context)
                results.append(result)
                self._execution_history.append(result)

                if not result.success:
                    logger.warning(
                        f"Trigger '{trigger.name}' failed: {result.error}"
                    )
                    if fail_fast:
                        logger.error("Stopping execution due to fail_fast=True")
                        break

            except Exception as e:
                logger.error(f"Trigger '{trigger.name}' raised exception: {e}")
                result = TriggerResult(
                    success=False,
                    phase=phase,
                    trigger_name=trigger.name,
                    execution_time_ms=0,
                    error=str(e),
                )
                results.append(result)
                self._execution_history.append(result)

                if fail_fast:
                    logger.error("Stopping execution due to fail_fast=True")
                    break

        # Log summary
        success_count = sum(1 for r in results if r.success)
        logger.info(
            f"Phase {phase.value} completed: {success_count}/{len(results)} succeeded"
        )

        return results

    async def trigger_manual_refresh(
        self,
        project_root: Optional[Path] = None,
        refresh_type: str = "full",
        fail_fast: bool = False,
    ) -> List[TriggerResult]:
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
        project_root: Optional[Path] = None,
        fail_fast: bool = False,
    ) -> List[TriggerResult]:
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
        project_root: Optional[Path] = None,
        fail_fast: bool = False,
    ) -> Dict[TriggerPhase, List[TriggerResult]]:
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

    async def cleanup_all(self, project_root: Optional[Path] = None) -> None:
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

    def get_execution_history(self) -> List[TriggerResult]:
        """
        Get history of all trigger executions.

        Returns:
            List of TriggerResult objects
        """
        return self._execution_history.copy()

    def clear_history(self) -> None:
        """Clear execution history."""
        self._execution_history.clear()


# Convenience functions for common patterns


async def prepare_claude_code_session(
    memory_manager: MemoryManager,
    project_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    token_budget: int = 50000,
) -> List[TriggerResult]:
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
    cleanup_paths: Optional[List[Path]] = None,
    project_root: Optional[Path] = None,
) -> List[TriggerResult]:
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


async def refresh_claude_code_context(
    memory_manager: MemoryManager,
    project_root: Optional[Path] = None,
    output_path: Optional[Path] = None,
    token_budget: int = 50000,
    refresh_type: str = "full",
) -> List[TriggerResult]:
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
