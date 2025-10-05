"""
Claude Code session detection for context injection.

This module provides utilities to detect when code is running within a Claude Code
CLI session, enabling automatic context injection and tool-specific formatting.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

from loguru import logger


@dataclass
class ClaudeCodeSession:
    """
    Information about a detected Claude Code session.

    Attributes:
        is_active: Whether a Claude Code session is currently active
        entrypoint: The entry point type (e.g., "cli", "api") if detected
        detection_method: How the session was detected
    """

    is_active: bool
    entrypoint: Optional[str] = None
    detection_method: Optional[str] = None


class ClaudeCodeDetector:
    """
    Detects Claude Code CLI sessions using multiple detection methods.

    This detector checks for Claude Code sessions by examining:
    1. Environment variables (CLAUDECODE, CLAUDE_CODE_ENTRYPOINT)
    2. Process information (parent process name)
    3. Standard input/output characteristics

    The detector uses multiple methods for robustness and provides
    information about how the session was detected.
    """

    # Environment variable names that indicate Claude Code session
    ENV_CLAUDECODE = "CLAUDECODE"
    ENV_ENTRYPOINT = "CLAUDE_CODE_ENTRYPOINT"

    @classmethod
    def detect(cls) -> ClaudeCodeSession:
        """
        Detect if currently running in a Claude Code session.

        Uses multiple detection methods in order of reliability:
        1. CLAUDECODE environment variable (most reliable)
        2. CLAUDE_CODE_ENTRYPOINT environment variable
        3. Parent process name inspection

        Returns:
            ClaudeCodeSession with detection results and metadata
        """
        # Method 1: Check CLAUDECODE environment variable
        claudecode_env = os.environ.get(cls.ENV_CLAUDECODE)
        if claudecode_env == "1":
            entrypoint = os.environ.get(cls.ENV_ENTRYPOINT)
            logger.debug(
                f"Claude Code session detected via {cls.ENV_CLAUDECODE} environment variable"
            )
            return ClaudeCodeSession(
                is_active=True,
                entrypoint=entrypoint,
                detection_method="environment_variable_claudecode",
            )

        # Method 2: Check CLAUDE_CODE_ENTRYPOINT environment variable
        entrypoint_env = os.environ.get(cls.ENV_ENTRYPOINT)
        if entrypoint_env:
            logger.debug(
                f"Claude Code session detected via {cls.ENV_ENTRYPOINT} environment variable"
            )
            return ClaudeCodeSession(
                is_active=True,
                entrypoint=entrypoint_env,
                detection_method="environment_variable_entrypoint",
            )

        # Method 3: Check parent process name (fallback)
        if cls._is_claude_parent_process():
            logger.debug("Claude Code session detected via parent process inspection")
            return ClaudeCodeSession(
                is_active=True,
                entrypoint=None,
                detection_method="parent_process",
            )

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


def get_claude_code_session() -> ClaudeCodeSession:
    """
    Get detailed information about the current Claude Code session.

    Convenience function that wraps ClaudeCodeDetector.detect().

    Returns:
        ClaudeCodeSession with detection results and metadata

    Example:
        >>> from context_injection import get_claude_code_session
        >>> session = get_claude_code_session()
        >>> if session.is_active:
        ...     print(f"Entrypoint: {session.entrypoint}")
        ...     print(f"Detected via: {session.detection_method}")
    """
    return ClaudeCodeDetector.detect()
