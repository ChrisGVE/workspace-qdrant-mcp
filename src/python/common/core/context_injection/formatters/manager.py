"""
Format manager for routing formatting requests to tool-specific adapters.

This module provides the FormatManager class which manages registration
and routing of formatting requests to appropriate LLM tool adapters with
integrated token limit validation.
"""


from loguru import logger

from ...memory import MemoryRule
from .base import FormattedContext, LLMToolAdapter


class FormatManager:
    """
    Manages LLM tool adapters and routes formatting requests.

    The FormatManager maintains a registry of tool adapters and provides
    a unified interface for formatting memory rules for different LLM tools.
    Includes integrated token limit validation using ToolTokenManager.
    """

    def __init__(self):
        """Initialize the format manager with built-in adapters."""
        self._adapters: dict[str, LLMToolAdapter] = {}
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

    def get_adapter(self, tool_name: str) -> LLMToolAdapter | None:
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
        rules: list[MemoryRule],
        token_budget: int,
        options: dict[str, any] | None = None,
    ) -> FormattedContext:
        """
        Format rules for a specific LLM tool with token limit validation.

        Validates token budget against tool's maximum context window before
        formatting. Logs warnings if approaching limits and raises ValueError
        if budget exceeds tool's maximum.

        Args:
            tool_name: Target tool name (e.g., "claude", "codex", "gemini")
            rules: Rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with tool-specific formatting

        Raises:
            ValueError: If tool adapter not found or token budget exceeds limit

        Example:
            >>> from context_injection import FormatManager
            >>> manager = FormatManager()
            >>> formatted = manager.format_for_tool(
            ...     "claude",
            ...     rules,
            ...     token_budget=150000
            ... )
        """
        adapter = self.get_adapter(tool_name)
        if not adapter:
            raise ValueError(f"No adapter registered for tool: {tool_name}")

        # Validate token budget against tool limits
        # Import here to avoid circular dependency
        from ..llm_tool_detector import LLMToolType
        from ..tool_token_manager import ToolTokenManager

        # Map tool name to LLMToolType
        tool_type_mapping = {
            "claude": LLMToolType.CLAUDE_CODE,
            "codex": LLMToolType.GITHUB_COPILOT,
            "gemini": LLMToolType.GOOGLE_GEMINI,
        }

        tool_type = tool_type_mapping.get(tool_name, LLMToolType.UNKNOWN)

        # Validate budget
        is_valid, message = ToolTokenManager.validate_token_count(
            tool_type, token_budget
        )

        if not is_valid:
            # Budget exceeds limit - this is an error
            limits = ToolTokenManager.get_limits(tool_type)
            logger.error(
                f"Token budget exceeds {tool_name} limit: {token_budget:,} > {limits.max_context_tokens:,}"
            )
            raise ValueError(message)

        if message:
            # Valid but has warning/critical message
            if "CRITICAL" in message:
                logger.warning(message)
            elif "WARNING" in message:
                logger.info(message)

        logger.debug(
            f"Formatting {len(rules)} rules for {tool_name} "
            f"(budget: {token_budget:,} tokens)"
        )

        formatted = adapter.format_rules(rules, token_budget, options)

        # Validate formatted output
        if not adapter.validate_format(formatted):
            logger.warning(f"Formatted content failed validation for {tool_name}")

        # Log actual token usage vs budget
        if formatted.token_count > token_budget:
            logger.warning(
                f"Formatted content exceeded budget: "
                f"{formatted.token_count:,} > {token_budget:,} tokens "
                f"({formatted.rules_skipped} rules skipped)"
            )

        return formatted

    def list_supported_tools(self) -> list[str]:
        """
        Get list of supported tools.

        Returns:
            List of tool names
        """
        return list(self._adapters.keys())
