"""
Format manager for routing formatting requests to tool-specific adapters.

This module provides the FormatManager class which manages registration
and routing of formatting requests to appropriate LLM tool adapters.
"""

from typing import Dict, List, Optional

from loguru import logger

from ...memory import MemoryRule
from .base import FormattedContext, LLMToolAdapter


class FormatManager:
    """
    Manages LLM tool adapters and routes formatting requests.

    The FormatManager maintains a registry of tool adapters and provides
    a unified interface for formatting memory rules for different LLM tools.
    """

    def __init__(self):
        """Initialize the format manager with built-in adapters."""
        self._adapters: Dict[str, LLMToolAdapter] = {}
        self._register_builtin_adapters()

    def _register_builtin_adapters(self):
        """Register built-in tool adapters."""
        # Import here to avoid circular dependencies
        try:
            from .claude_code import ClaudeCodeAdapter

            self.register_adapter("claude", ClaudeCodeAdapter())
        except ImportError:
            logger.warning("ClaudeCodeAdapter not available")

        try:
            from .github_codex import GitHubCodexAdapter

            self.register_adapter("codex", GitHubCodexAdapter())
        except ImportError:
            logger.warning("GitHubCodexAdapter not available")

        try:
            from .google_gemini import GoogleGeminiAdapter

            self.register_adapter("gemini", GoogleGeminiAdapter())
        except ImportError:
            logger.warning("GoogleGeminiAdapter not available")

    def register_adapter(self, tool_name: str, adapter: LLMToolAdapter):
        """
        Register a new tool adapter.

        Args:
            tool_name: Tool identifier (e.g., "claude", "codex")
            adapter: Adapter implementation for this tool
        """
        self._adapters[tool_name] = adapter
        logger.info(f"Registered adapter for tool: {tool_name}")

    def get_adapter(self, tool_name: str) -> Optional[LLMToolAdapter]:
        """
        Get adapter for a specific tool.

        Args:
            tool_name: Tool identifier

        Returns:
            LLMToolAdapter if found, None otherwise
        """
        return self._adapters.get(tool_name)

    def format_for_tool(
        self,
        tool_name: str,
        rules: List[MemoryRule],
        token_budget: int,
        options: Optional[Dict[str, any]] = None,
    ) -> FormattedContext:
        """
        Format rules for a specific LLM tool.

        Args:
            tool_name: Target tool name
            rules: Rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with tool-specific formatting

        Raises:
            ValueError: If tool adapter not found
        """
        adapter = self.get_adapter(tool_name)
        if not adapter:
            raise ValueError(f"No adapter registered for tool: {tool_name}")

        logger.debug(
            f"Formatting {len(rules)} rules for {tool_name} "
            f"(budget: {token_budget} tokens)"
        )

        formatted = adapter.format_rules(rules, token_budget, options)

        # Validate formatted output
        if not adapter.validate_format(formatted):
            logger.warning(f"Formatted content failed validation for {tool_name}")

        return formatted

    def list_supported_tools(self) -> List[str]:
        """
        Get list of supported tools.

        Returns:
            List of tool names
        """
        return list(self._adapters.keys())
