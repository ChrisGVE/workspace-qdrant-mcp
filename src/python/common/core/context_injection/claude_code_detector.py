"""
Claude Code session detection for context injection.

This module provides utilities to detect when code is running within a Claude Code
CLI session, enabling automatic context injection and tool-specific formatting.
Includes session lifecycle monitoring and metadata enrichment.
"""

import asyncio
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger


class SessionEvent(Enum):
    """Session lifecycle events."""

    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    SESSION_ACTIVE = "session_active"  # Periodic heartbeat
    SESSION_INACTIVE = "session_inactive"


@dataclass
class ProjectContextMetadata:
    """
    Project context metadata for a Claude Code session.

    Attributes:
        project_id: Unique project identifier
        project_root: Absolute path to project root
        current_path: Current working directory
        scope: Current scope contexts (e.g., ["python", "testing"])
        is_submodule: Whether this is a git submodule
        parent_project_id: Parent project ID if this is a submodule
    """

    project_id: str | None = None
    project_root: Path | None = None
    current_path: Path | None = None
    scope: list[str] = field(default_factory=list)
    is_submodule: bool = False
    parent_project_id: str | None = None


@dataclass
class ClaudeCodeSession:
    """
    Information about a detected Claude Code session.

    Attributes:
        is_active: Whether a Claude Code session is currently active
        entrypoint: The entry point type (e.g., "cli", "api") if detected
        detection_method: How the session was detected
        session_id: Unique identifier for this session instance
        start_time: When the session was detected (timestamp)
        project_context: Project context metadata
        configuration: Additional configuration metadata
        metadata: Arbitrary metadata dictionary
    """

    is_active: bool
    entrypoint: str | None = None
    detection_method: str | None = None
    session_id: str | None = None
    start_time: float | None = None
    project_context: ProjectContextMetadata | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ClaudeCodeDetector:
    """
    Detects Claude Code CLI sessions using multiple detection methods.

    This detector checks for Claude Code sessions by examining:
    1. Environment variables (CLAUDECODE, CLAUDE_CODE_ENTRYPOINT)
    2. Process information (parent process name)
    3. Standard input/output characteristics

    The detector uses multiple methods for robustness and provides
    information about how the session was detected. Can enrich sessions
    with project context and configuration metadata.
    """

    # Environment variable names that indicate Claude Code session
    ENV_CLAUDECODE = "CLAUDECODE"
    ENV_ENTRYPOINT = "CLAUDE_CODE_ENTRYPOINT"

    @classmethod
    def detect(cls, enrich_metadata: bool = False) -> ClaudeCodeSession:
        """
        Detect if currently running in a Claude Code session.

        Uses multiple detection methods in order of reliability:
        1. CLAUDECODE environment variable (most reliable)
        2. CLAUDE_CODE_ENTRYPOINT environment variable
        3. Parent process name inspection

        Args:
            enrich_metadata: If True, enrich session with project context
                and configuration metadata

        Returns:
            ClaudeCodeSession with detection results and optional metadata
        """
        # Method 1: Check CLAUDECODE environment variable
        claudecode_env = os.environ.get(cls.ENV_CLAUDECODE)
        if claudecode_env == "1":
            entrypoint = os.environ.get(cls.ENV_ENTRYPOINT)
            logger.debug(
                f"Claude Code session detected via {cls.ENV_CLAUDECODE} environment variable"
            )
            session = ClaudeCodeSession(
                is_active=True,
                entrypoint=entrypoint,
                detection_method="environment_variable_claudecode",
                session_id=cls._generate_session_id(),
                start_time=time.time(),
            )
            if enrich_metadata:
                cls._enrich_session_metadata(session)
            return session

        # Method 2: Check CLAUDE_CODE_ENTRYPOINT environment variable
        entrypoint_env = os.environ.get(cls.ENV_ENTRYPOINT)
        if entrypoint_env:
            logger.debug(
                f"Claude Code session detected via {cls.ENV_ENTRYPOINT} environment variable"
            )
            session = ClaudeCodeSession(
                is_active=True,
                entrypoint=entrypoint_env,
                detection_method="environment_variable_entrypoint",
                session_id=cls._generate_session_id(),
                start_time=time.time(),
            )
            if enrich_metadata:
                cls._enrich_session_metadata(session)
            return session

        # Method 3: Check parent process name (fallback)
        if cls._is_claude_parent_process():
            logger.debug("Claude Code session detected via parent process inspection")
            session = ClaudeCodeSession(
                is_active=True,
                entrypoint=None,
                detection_method="parent_process",
                session_id=cls._generate_session_id(),
                start_time=time.time(),
            )
            if enrich_metadata:
                cls._enrich_session_metadata(session)
            return session

        # No Claude Code session detected
        logger.debug("No Claude Code session detected")
        return ClaudeCodeSession(
            is_active=False, entrypoint=None, detection_method=None
        )

    @classmethod
    def is_active(cls) -> bool:
        """
        Check if currently running in a Claude Code session.

        Convenience method that returns just the boolean active status.

        Returns:
            True if Claude Code session is detected, False otherwise
        """
        return cls.detect().is_active

    @staticmethod
    def _generate_session_id() -> str:
        """
        Generate a unique session identifier.

        Returns:
            Unique session ID string
        """
        import uuid

        return f"claude-session-{uuid.uuid4().hex[:8]}"

    @classmethod
    def _enrich_session_metadata(cls, session: ClaudeCodeSession) -> None:
        """
        Enrich session with project context and configuration metadata.

        Args:
            session: Session to enrich (modified in-place)
        """
        try:
            # Import here to avoid circular dependencies
            from .project_context import ProjectContextDetector

            # Detect project context
            detector = ProjectContextDetector()
            project_context = detector.detect_project_context()

            if project_context:
                # Create ProjectContextMetadata from detected context
                session.project_context = ProjectContextMetadata(
                    project_id=project_context.project_id,
                    project_root=project_context.project_root,
                    current_path=project_context.current_path,
                    scope=project_context.scope,
                    is_submodule=project_context.is_submodule,
                    parent_project_id=project_context.parent_project_id,
                )

                logger.debug(
                    f"Enriched session with project context: {project_context.project_id}"
                )

            # Add configuration metadata from environment if available
            cls._add_configuration_metadata(session)

        except ImportError as e:
            logger.warning(f"Failed to import project context detector: {e}")
        except Exception as e:
            logger.warning(f"Failed to enrich session metadata: {e}")

    @staticmethod
    def _add_configuration_metadata(session: ClaudeCodeSession) -> None:
        """
        Add configuration metadata from environment variables.

        Args:
            session: Session to add metadata to (modified in-place)
        """
        # Check for common Claude Code configuration env vars
        config_vars = {
            "CLAUDE_CODE_VERSION": os.environ.get("CLAUDE_CODE_VERSION"),
            "CLAUDE_CODE_DEBUG": os.environ.get("CLAUDE_CODE_DEBUG"),
            "CLAUDE_CODE_LOG_LEVEL": os.environ.get("CLAUDE_CODE_LOG_LEVEL"),
            "CLAUDE_CODE_CONFIG_PATH": os.environ.get("CLAUDE_CODE_CONFIG_PATH"),
        }

        # Filter out None values and add to configuration
        session.configuration.update(
            {k: v for k, v in config_vars.items() if v is not None}
        )

        # Add current working directory
        try:
            session.configuration["cwd"] = os.getcwd()
        except Exception:
            pass

        # Add Python version info
        session.configuration["python_version"] = sys.version

    @staticmethod
    def _is_claude_parent_process() -> bool:
        """
        Check if the parent process is Claude Code.

        This method inspects the parent process to determine if it's
        named "claude" or contains "claude" in its command line.

        Returns:
            True if parent process appears to be Claude Code, False otherwise
        """
        try:
            import psutil

            current_process = psutil.Process(os.getpid())
            parent = current_process.parent()

            if parent is None:
                return False

            # Check if parent process name is "claude"
            parent_name = parent.name().lower()
            if "claude" in parent_name:
                logger.debug(f"Parent process name contains 'claude': {parent_name}")
                return True

            # Check parent's command line for "claude"
            try:
                cmdline = " ".join(parent.cmdline()).lower()
                if "claude" in cmdline:
                    logger.debug(f"Parent process cmdline contains 'claude': {cmdline}")
                    return True
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                # Can't access cmdline, skip this check
                pass

        except ImportError:
            # psutil not available, skip this detection method
            logger.debug("psutil not available for parent process detection")
            pass
        except Exception as e:
            # Log but don't fail on process inspection errors
            logger.debug(f"Error inspecting parent process: {e}")
            pass

        return False


# Convenience function for simple detection
def is_claude_code_session() -> bool:
    """
    Check if currently running in a Claude Code session.

    Convenience function that wraps ClaudeCodeDetector.is_active().

    Returns:
        True if Claude Code session is detected, False otherwise

    Example:
        >>> from context_injection import is_claude_code_session
        >>> if is_claude_code_session():
        ...     # Apply Claude Code specific formatting
        ...     pass
    """
    return ClaudeCodeDetector.is_active()


def get_claude_code_session(enrich_metadata: bool = False) -> ClaudeCodeSession:
    """
    Get detailed information about the current Claude Code session.

    Convenience function that wraps ClaudeCodeDetector.detect().

    Args:
        enrich_metadata: If True, enrich session with project context
            and configuration metadata

    Returns:
        ClaudeCodeSession with detection results and optional metadata

    Example:
        >>> from context_injection import get_claude_code_session
        >>> session = get_claude_code_session(enrich_metadata=True)
        >>> if session.is_active:
        ...     print(f"Entrypoint: {session.entrypoint}")
        ...     print(f"Detected via: {session.detection_method}")
        ...     if session.project_context:
        ...         print(f"Project: {session.project_context.project_id}")
    """
    return ClaudeCodeDetector.detect(enrich_metadata=enrich_metadata)


class ClaudeSessionMonitor:
    """
    Monitors Claude Code session lifecycle and emits events.

    This monitor can detect session start/end events and call registered
    callbacks when session state changes. Useful for triggering actions
    when Claude Code sessions begin or end.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        enrich_metadata: bool = True,
        detector: ClaudeCodeDetector | None = None,
    ):
        """
        Initialize the session monitor.

        Args:
            poll_interval: How often to check for session state (seconds)
            enrich_metadata: Whether to enrich sessions with metadata
            detector: Optional ClaudeCodeDetector instance
        """
        self.poll_interval = poll_interval
        self.enrich_metadata = enrich_metadata
        self.detector = detector or ClaudeCodeDetector
        self._current_session: ClaudeCodeSession | None = None
        self._callbacks: dict[SessionEvent, list[Callable]] = {
            event: [] for event in SessionEvent
        }
        self._monitoring = False
        self._monitor_task: asyncio.Task | None = None

    def register_callback(
        self, event: SessionEvent, callback: Callable[[ClaudeCodeSession], None]
    ) -> None:
        """
        Register a callback for a specific session event.

        Args:
            event: Event type to listen for
            callback: Function to call when event occurs
                (receives ClaudeCodeSession as argument)
        """
        if callback not in self._callbacks[event]:
            self._callbacks[event].append(callback)
            logger.debug(f"Registered callback for event {event.value}")

    def unregister_callback(
        self, event: SessionEvent, callback: Callable[[ClaudeCodeSession], None]
    ) -> bool:
        """
        Unregister a callback.

        Args:
            event: Event type
            callback: Callback to remove

        Returns:
            True if callback was found and removed, False otherwise
        """
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            logger.debug(f"Unregistered callback for event {event.value}")
            return True
        return False

    async def start_monitoring(self) -> None:
        """
        Start monitoring for session lifecycle events.

        Begins polling for session state changes and emitting events.
        """
        if self._monitoring:
            logger.warning("Session monitor already running")
            return

        self._monitoring = True
        logger.info(f"Starting session monitor (poll_interval={self.poll_interval}s)")

        # Detect initial session state
        self._current_session = self.detector.detect(
            enrich_metadata=self.enrich_metadata
        )

        if self._current_session.is_active:
            await self._emit_event(SessionEvent.SESSION_STARTED, self._current_session)

        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self) -> None:
        """
        Stop monitoring for session lifecycle events.
        """
        if not self._monitoring:
            return

        self._monitoring = False
        logger.info("Stopping session monitor")

        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Emit session ended if there was an active session
        if self._current_session and self._current_session.is_active:
            await self._emit_event(SessionEvent.SESSION_ENDED, self._current_session)

    async def _monitor_loop(self) -> None:
        """
        Main monitoring loop that polls for session state changes.
        """
        while self._monitoring:
            try:
                # Detect current session state
                new_session = self.detector.detect(enrich_metadata=self.enrich_metadata)

                # Check for state transitions
                if self._current_session is None:
                    # First detection
                    if new_session.is_active:
                        await self._emit_event(SessionEvent.SESSION_STARTED, new_session)
                elif self._current_session.is_active and not new_session.is_active:
                    # Session ended
                    await self._emit_event(SessionEvent.SESSION_ENDED, self._current_session)
                elif not self._current_session.is_active and new_session.is_active:
                    # Session started
                    await self._emit_event(SessionEvent.SESSION_STARTED, new_session)
                elif self._current_session.is_active and new_session.is_active:
                    # Session still active - periodic heartbeat
                    await self._emit_event(SessionEvent.SESSION_ACTIVE, new_session)

                self._current_session = new_session

                # Wait before next poll
                await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session monitor loop: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _emit_event(
        self, event: SessionEvent, session: ClaudeCodeSession
    ) -> None:
        """
        Emit an event by calling all registered callbacks.

        Args:
            event: Event type
            session: Session information
        """
        logger.debug(f"Emitting event: {event.value}")

        callbacks = self._callbacks[event]
        for callback in callbacks:
            try:
                # Call callback (handle both sync and async)
                if asyncio.iscoroutinefunction(callback):
                    await callback(session)
                else:
                    callback(session)
            except Exception as e:
                logger.error(f"Error in callback for {event.value}: {e}")

    def get_current_session(self) -> ClaudeCodeSession | None:
        """
        Get the current session state.

        Returns:
            Current ClaudeCodeSession or None if monitoring not started
        """
        return self._current_session

    @property
    def is_monitoring(self) -> bool:
        """
        Check if monitoring is active.

        Returns:
            True if monitoring is running
        """
        return self._monitoring
