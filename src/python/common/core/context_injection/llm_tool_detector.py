"""
Unified LLM tool detection system for context injection.

This module provides a unified detection system that can detect which LLM tool
is currently active (Claude Code, GitHub Copilot, Cursor, etc.) and return
consistent session information across all tools. It integrates with the
FormatManager to select the appropriate formatter for each tool.

Detection Priority:
1. Claude Code (highest priority - MCP-based tool)
2. Cursor (has its own AI)
3. JetBrains AI
4. GitHub Copilot (VSCode/other IDEs)
5. Other tools
6. Unknown (if nothing detected)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .claude_code_detector import ClaudeCodeDetector, ClaudeCodeSession
from .copilot_detector import CopilotDetector, CopilotSession, CopilotSessionType
from .formatters import FormatManager, LLMToolAdapter, ToolCapabilities


class LLMToolType(Enum):
    """Standardized LLM tool types for unified detection."""

    CLAUDE_CODE = "claude_code"
    GITHUB_COPILOT = "github_copilot"
    CODEX_API = "codex_api"
    CURSOR = "cursor"
    JETBRAINS_AI = "jetbrains_ai"
    GOOGLE_GEMINI = "google_gemini"
    TABNINE = "tabnine"
    UNKNOWN = "unknown"


@dataclass
class UnifiedLLMSession:
    """
    Unified session information across all LLM tools.

    This dataclass provides a consistent interface for session information
    regardless of which LLM tool is detected.

    Attributes:
        tool_type: Type of LLM tool detected
        is_active: Whether an LLM tool session is currently active
        detection_method: How the session was detected
        session_id: Optional unique identifier for this session
        ide_name: Optional name of the IDE (e.g., "vscode", "cursor")
        workspace_path: Optional workspace/project path
        capabilities: Tool capabilities for formatting
        metadata: Additional tool-specific metadata
    """

    tool_type: LLMToolType
    is_active: bool
    detection_method: str
    session_id: Optional[str] = None
    ide_name: Optional[str] = None
    workspace_path: Optional[Path] = None
    capabilities: Optional[ToolCapabilities] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMToolDetector:
    """
    Unified detection system for LLM tools.

    This detector uses multiple detection methods to identify which LLM tool
    is currently active and provides a unified interface for session information.

    Detection Priority:
    1. Claude Code (highest priority - MCP-based tool)
    2. Cursor (has its own AI)
    3. JetBrains AI
    4. GitHub Copilot
    5. Other tools
    6. Unknown (if nothing detected)

    The detector integrates with FormatManager to provide the appropriate
    formatter for each detected tool.
    """

    _format_manager: Optional[FormatManager] = None

    @classmethod
    def _get_format_manager(cls) -> FormatManager:
        """
        Get or create the format manager singleton.

        Returns:
            FormatManager instance
        """
        if cls._format_manager is None:
            cls._format_manager = FormatManager()
        return cls._format_manager

    @classmethod
    def detect(cls) -> UnifiedLLMSession:
        """
        Detect which LLM tool is currently active.

        Uses multiple detection methods in priority order:
        1. Claude Code (highest priority)
        2. Cursor
        3. JetBrains AI
        4. GitHub Copilot
        5. Other tools

        Returns:
            UnifiedLLMSession with detection results and metadata
        """
        # Priority 1: Check for Claude Code session (highest priority)
        claude_session = ClaudeCodeDetector.detect(enrich_metadata=True)
        if claude_session.is_active:
            logger.debug("Detected active Claude Code session")
            return cls._create_unified_from_claude(claude_session)

        # Priority 2-5: Check for Copilot/IDE sessions
        copilot_session = CopilotDetector.detect()
        if copilot_session.is_active:
            logger.debug(f"Detected active {copilot_session.session_type.value} session")
            return cls._create_unified_from_copilot(copilot_session)

        # No LLM tool detected
        logger.debug("No active LLM tool session detected")
        return UnifiedLLMSession(
            tool_type=LLMToolType.UNKNOWN,
            is_active=False,
            detection_method="none",
        )

    @classmethod
    def is_active(cls) -> bool:
        """
        Check if any LLM tool session is currently active.

        Convenience method that returns just the boolean active status.

        Returns:
            True if any LLM tool session is detected, False otherwise
        """
        session = cls.detect()
        return session.is_active

    @classmethod
    def get_active_tool_type(cls) -> Optional[LLMToolType]:
        """
        Get the type of currently active LLM tool.

        Returns:
            LLMToolType if a tool is active, None otherwise
        """
        session = cls.detect()
        if session.is_active:
            return session.tool_type
        return None

    @classmethod
    def get_formatter(cls) -> Optional[LLMToolAdapter]:
        """
        Get the appropriate formatter for the currently active LLM tool.

        Uses FormatManager to retrieve the tool-specific adapter.

        Returns:
            LLMToolAdapter if a tool is active and has a registered adapter,
            None otherwise
        """
        session = cls.detect()
        if not session.is_active:
            logger.debug("No active LLM tool session - no formatter available")
            return None

        # Map tool type to FormatManager adapter name
        adapter_mapping = {
            LLMToolType.CLAUDE_CODE: "claude",
            LLMToolType.GITHUB_COPILOT: "codex",
            LLMToolType.CURSOR: "codex",  # Cursor uses Codex-style formatting
            LLMToolType.JETBRAINS_AI: "codex",  # JetBrains uses Codex-style
            LLMToolType.CODEX_API: "codex",
            LLMToolType.GOOGLE_GEMINI: "gemini",
            LLMToolType.TABNINE: "codex",  # Tabnine uses Codex-style
        }

        adapter_name = adapter_mapping.get(session.tool_type)
        if adapter_name is None:
            logger.debug(f"No adapter mapping for tool type: {session.tool_type}")
            return None

        format_manager = cls._get_format_manager()
        adapter = format_manager.get_adapter(adapter_name)

        if adapter is None:
            logger.warning(
                f"No formatter registered for tool type: {session.tool_type} "
                f"(adapter: {adapter_name})"
            )
        else:
            logger.debug(
                f"Retrieved formatter for {session.tool_type.value}: {adapter_name}"
            )

        return adapter

    @classmethod
    def _create_unified_from_claude(
        cls, claude_session: ClaudeCodeSession
    ) -> UnifiedLLMSession:
        """
        Create UnifiedLLMSession from ClaudeCodeSession.

        Args:
            claude_session: ClaudeCodeSession instance

        Returns:
            UnifiedLLMSession with Claude Code information
        """
        # Extract workspace path from project context
        workspace_path = None
        if claude_session.project_context:
            workspace_path = claude_session.project_context.project_root

        # Get capabilities from formatter
        capabilities = None
        format_manager = cls._get_format_manager()
        adapter = format_manager.get_adapter("claude")
        if adapter:
            capabilities = adapter.get_capabilities()

        # Build metadata from Claude session
        metadata = {
            "entrypoint": claude_session.entrypoint,
            "start_time": claude_session.start_time,
        }

        # Add project context if available
        if claude_session.project_context:
            metadata["project_id"] = claude_session.project_context.project_id
            metadata["current_path"] = str(claude_session.project_context.current_path)
            metadata["scope"] = claude_session.project_context.scope
            metadata["is_submodule"] = claude_session.project_context.is_submodule
            if claude_session.project_context.parent_project_id:
                metadata[
                    "parent_project_id"
                ] = claude_session.project_context.parent_project_id

        # Add configuration
        metadata["configuration"] = claude_session.configuration

        return UnifiedLLMSession(
            tool_type=LLMToolType.CLAUDE_CODE,
            is_active=True,
            detection_method=claude_session.detection_method or "unknown",
            session_id=claude_session.session_id,
            ide_name="claude_code_cli",
            workspace_path=workspace_path,
            capabilities=capabilities,
            metadata=metadata,
        )

    @classmethod
    def _create_unified_from_copilot(
        cls, copilot_session: CopilotSession
    ) -> UnifiedLLMSession:
        """
        Create UnifiedLLMSession from CopilotSession.

        Args:
            copilot_session: CopilotSession instance

        Returns:
            UnifiedLLMSession with Copilot/IDE information
        """
        # Map CopilotSessionType to LLMToolType
        tool_type_mapping = {
            CopilotSessionType.GITHUB_COPILOT: LLMToolType.GITHUB_COPILOT,
            CopilotSessionType.CODEX_API: LLMToolType.CODEX_API,
            CopilotSessionType.CURSOR: LLMToolType.CURSOR,
            CopilotSessionType.JETBRAINS_AI: LLMToolType.JETBRAINS_AI,
            CopilotSessionType.TABNINE: LLMToolType.TABNINE,
            CopilotSessionType.UNKNOWN: LLMToolType.UNKNOWN,
        }

        tool_type = tool_type_mapping.get(
            copilot_session.session_type, LLMToolType.UNKNOWN
        )

        # Get capabilities from formatter
        capabilities = None
        format_manager = cls._get_format_manager()

        # Determine which adapter to use based on tool type
        adapter_name = None
        if tool_type in {
            LLMToolType.GITHUB_COPILOT,
            LLMToolType.CURSOR,
            LLMToolType.JETBRAINS_AI,
            LLMToolType.CODEX_API,
            LLMToolType.TABNINE,
        }:
            adapter_name = "codex"

        if adapter_name:
            adapter = format_manager.get_adapter(adapter_name)
            if adapter:
                capabilities = adapter.get_capabilities()

        # Build metadata
        metadata = {
            "ide_version": copilot_session.ide_version,
        }

        return UnifiedLLMSession(
            tool_type=tool_type,
            is_active=True,
            detection_method=copilot_session.detection_method or "unknown",
            session_id=None,  # Copilot sessions don't have session IDs
            ide_name=copilot_session.ide_name,
            workspace_path=copilot_session.workspace_path,
            capabilities=capabilities,
            metadata=metadata,
        )


# Convenience functions


def get_active_llm_tool() -> UnifiedLLMSession:
    """
    Get information about the currently active LLM tool.

    Convenience function that wraps LLMToolDetector.detect().

    Returns:
        UnifiedLLMSession with detection results

    Example:
        >>> from context_injection import get_active_llm_tool
        >>> session = get_active_llm_tool()
        >>> if session.is_active:
        ...     print(f"Tool: {session.tool_type.value}")
        ...     print(f"IDE: {session.ide_name}")
        ...     print(f"Detected via: {session.detection_method}")
    """
    return LLMToolDetector.detect()


def is_llm_tool_active() -> bool:
    """
    Check if any LLM tool is currently active.

    Convenience function that wraps LLMToolDetector.is_active().

    Returns:
        True if any LLM tool is detected, False otherwise

    Example:
        >>> from context_injection import is_llm_tool_active
        >>> if is_llm_tool_active():
        ...     # Apply LLM-specific context injection
        ...     pass
    """
    return LLMToolDetector.is_active()


def get_llm_formatter() -> Optional[LLMToolAdapter]:
    """
    Get the formatter for the currently active LLM tool.

    Convenience function that wraps LLMToolDetector.get_formatter().

    Returns:
        LLMToolAdapter if a tool is active, None otherwise

    Example:
        >>> from context_injection import get_llm_formatter
        >>> formatter = get_llm_formatter()
        >>> if formatter:
        ...     formatted = formatter.format_rules(rules, token_budget)
        ...     print(f"Formatted for: {formatted.tool_name}")
    """
    return LLMToolDetector.get_formatter()
