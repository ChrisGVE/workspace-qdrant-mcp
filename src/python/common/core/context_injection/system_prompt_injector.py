"""
System prompt injection mechanism for Claude Code sessions.

This module provides an alternative to CLAUDE.md file injection by formatting
memory rules as system prompts suitable for direct API injection or MCP context.
System prompts have stricter token limits than file-based injection.
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from ..memory import MemoryManager, MemoryRule
from .formatters.claude_code import ClaudeCodeAdapter
from .formatters.base import FormattedContext
from .rule_retrieval import RuleFilter, RuleRetrieval
from .token_budget import TokenBudgetManager, AllocationStrategy


class InjectionMode(Enum):
    """System prompt injection modes."""

    MCP = "mcp"  # MCP server context injection
    API = "api"  # Direct API call system prompt
    CUSTOM = "custom"  # Custom injection mechanism


@dataclass
class SystemPromptConfig:
    """
    Configuration for system prompt generation.

    Attributes:
        token_budget: Maximum tokens for system prompt (default: 15000)
        injection_mode: How the prompt will be injected
        include_metadata: Include generation metadata in prompt
        allocation_strategy: Strategy for token budget allocation
        overhead_percentage: Percentage to reserve for formatting (default: 5%)
        compact_format: Use more compact formatting to save tokens
    """

    token_budget: int = 15000  # Conservative default for system prompts
    injection_mode: InjectionMode = InjectionMode.MCP
    include_metadata: bool = True
    allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED
    overhead_percentage: float = 0.05
    compact_format: bool = False


class SystemPromptInjector:
    """
    System prompt injection mechanism for Claude Code.

    This class provides an alternative to CLAUDE.md file injection for scenarios
    where file-based injection isn't available or suitable:
    - MCP server context injection
    - Direct API calls with system prompts
    - Custom injection mechanisms

    System prompts typically have smaller token budgets than CLAUDE.md files,
    so this class enforces stricter token management.
    """

    # Default token budgets by injection mode
    DEFAULT_BUDGETS = {
        InjectionMode.MCP: 15000,  # Conservative for MCP context
        InjectionMode.API: 10000,  # More conservative for API system prompts
        InjectionMode.CUSTOM: 20000,  # Flexible for custom usage
    }

    def __init__(
        self,
        memory_manager: MemoryManager,
        rule_retrieval: Optional[RuleRetrieval] = None,
        adapter: Optional[ClaudeCodeAdapter] = None,
        token_budget_manager: Optional[TokenBudgetManager] = None,
    ):
        """
        Initialize the system prompt injector.

        Args:
            memory_manager: MemoryManager instance for rule storage
            rule_retrieval: RuleRetrieval instance (created if not provided)
            adapter: ClaudeCodeAdapter instance (created if not provided)
            token_budget_manager: TokenBudgetManager (created if not provided)
        """
        self.memory_manager = memory_manager
        self.rule_retrieval = rule_retrieval or RuleRetrieval(memory_manager)
        self.adapter = adapter or ClaudeCodeAdapter()
        self.token_budget_manager = token_budget_manager or TokenBudgetManager()

    async def generate_system_prompt(
        self,
        config: Optional[SystemPromptConfig] = None,
        filter: Optional[RuleFilter] = None,
    ) -> str:
        """
        Generate system prompt from memory rules.

        This method:
        1. Retrieves applicable memory rules
        2. Allocates token budget across rules
        3. Formats rules using ClaudeCodeAdapter
        4. Returns compact system prompt string

        Args:
            config: SystemPromptConfig (uses defaults if not provided)
            filter: Optional filter for memory rules

        Returns:
            Formatted system prompt string ready for injection

        Example:
            >>> injector = SystemPromptInjector(memory_manager)
            >>> config = SystemPromptConfig(
            ...     token_budget=10000,
            ...     injection_mode=InjectionMode.API
            ... )
            >>> prompt = await injector.generate_system_prompt(config)
        """
        config = config or SystemPromptConfig()

        # Get rules from retrieval system
        if filter is None:
            filter = RuleFilter(limit=100)

        result = await self.rule_retrieval.get_rules(filter)

        if not result.rules:
            logger.debug("No memory rules found for system prompt")
            return ""

        logger.info(f"Retrieved {len(result.rules)} rules for system prompt generation")

        # Format using ClaudeCodeAdapter
        formatted = self._format_for_system_prompt(
            rules=result.rules, config=config
        )

        if not formatted.content:
            logger.warning("Failed to format system prompt")
            return ""

        # Add metadata header if requested
        if config.include_metadata:
            prompt = self._add_metadata_header(formatted, config)
        else:
            prompt = formatted.content

        logger.info(
            f"Generated system prompt: {formatted.rules_included} rules, "
            f"{formatted.token_count} tokens, "
            f"{formatted.rules_skipped} skipped"
        )

        return prompt

    async def inject_to_file(
        self,
        output_path: Path,
        config: Optional[SystemPromptConfig] = None,
        filter: Optional[RuleFilter] = None,
    ) -> bool:
        """
        Generate system prompt and write to file.

        Useful for testing or integration with tools that read prompts from files.

        Args:
            output_path: Path to write system prompt
            config: SystemPromptConfig
            filter: Optional filter for memory rules

        Returns:
            True if successful, False otherwise
        """
        try:
            prompt = await self.generate_system_prompt(config=config, filter=filter)

            if not prompt:
                logger.warning("No system prompt content to write")
                return False

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            output_path.write_text(prompt, encoding="utf-8")
            logger.info(f"Wrote {len(prompt)} characters to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to inject system prompt to {output_path}: {e}")
            return False

    def get_recommended_budget(self, mode: InjectionMode) -> int:
        """
        Get recommended token budget for injection mode.

        Args:
            mode: Injection mode

        Returns:
            Recommended token budget
        """
        return self.DEFAULT_BUDGETS.get(mode, 15000)

    def _format_for_system_prompt(
        self, rules: List[MemoryRule], config: SystemPromptConfig
    ) -> FormattedContext:
        """
        Format rules for system prompt injection.

        Uses ClaudeCodeAdapter but with stricter token management and
        optional compact formatting.

        Args:
            rules: List of memory rules to format
            config: System prompt configuration

        Returns:
            FormattedContext with formatted content
        """
        # Use adapter's format_rules method
        options = {
            "compact": config.compact_format,
            "include_sources": not config.compact_format,
        }

        formatted = self.adapter.format_rules(
            rules=rules,
            token_budget=config.token_budget,
            options=options,
        )

        # Apply compact formatting if requested
        if config.compact_format:
            formatted = self._apply_compact_formatting(formatted)

        return formatted

    def _apply_compact_formatting(
        self, formatted: FormattedContext
    ) -> FormattedContext:
        """
        Apply compact formatting to reduce token usage.

        Compact formatting:
        - Remove extra whitespace
        - Shorten section headers
        - Remove decorative elements
        - Consolidate similar rules

        Args:
            formatted: Original formatted context

        Returns:
            Compacted FormattedContext
        """
        content = formatted.content

        # Shorten headers
        content = content.replace("## CRITICAL RULES (Always Follow)", "## CRITICAL RULES")
        content = content.replace(
            "## DEFAULT GUIDELINES (Unless Overridden)", "## GUIDELINES"
        )

        # Remove descriptive text
        content = content.replace(
            "These rules are non-negotiable and must always be followed.\n", ""
        )
        content = content.replace(
            "These are recommended practices unless explicitly overridden by the user or PRD.\n",
            "",
        )

        # Remove auto-generation metadata (will be in header instead)
        lines = content.split("\n")
        filtered_lines = []
        skip_next = False
        for line in lines:
            if "**Auto-generated" in line or "**Generated:" in line:
                skip_next = True
                continue
            if skip_next and line.strip() == "":
                skip_next = False
                continue
            filtered_lines.append(line)

        content = "\n".join(filtered_lines)

        # Remove excessive blank lines
        while "\n\n\n" in content:
            content = content.replace("\n\n\n", "\n\n")

        # Recalculate token count
        token_count = self.adapter.estimate_token_count(content)

        return FormattedContext(
            tool_name=formatted.tool_name,
            format_type=formatted.format_type,
            content=content,
            token_count=token_count,
            rules_included=formatted.rules_included,
            rules_skipped=formatted.rules_skipped,
            metadata={**formatted.metadata, "compact": True},
        )

    def _add_metadata_header(
        self, formatted: FormattedContext, config: SystemPromptConfig
    ) -> str:
        """
        Add metadata header to system prompt.

        Args:
            formatted: Formatted context
            config: System prompt configuration

        Returns:
            Content with metadata header
        """
        header_lines = [
            "# Memory System Context",
            f"<!-- Generated for {config.injection_mode.value} injection -->",
            f"<!-- Token budget: {config.token_budget}, Used: {formatted.token_count} -->",
            f"<!-- Rules: {formatted.rules_included} included, {formatted.rules_skipped} skipped -->",
            "",
        ]

        return "\n".join(header_lines) + formatted.content


# Convenience functions for common patterns


async def generate_mcp_context(
    memory_manager: MemoryManager,
    token_budget: int = 15000,
    filter: Optional[RuleFilter] = None,
) -> str:
    """
    Convenience function to generate MCP context injection.

    Args:
        memory_manager: MemoryManager instance
        token_budget: Token budget for context (default: 15000)
        filter: Optional filter for memory rules

    Returns:
        Formatted context string for MCP injection

    Example:
        >>> from context_injection import generate_mcp_context
        >>> context = await generate_mcp_context(memory_manager)
    """
    config = SystemPromptConfig(
        token_budget=token_budget,
        injection_mode=InjectionMode.MCP,
        compact_format=False,
    )

    injector = SystemPromptInjector(memory_manager)
    return await injector.generate_system_prompt(config=config, filter=filter)


async def generate_api_system_prompt(
    memory_manager: MemoryManager,
    token_budget: int = 10000,
    compact: bool = True,
    filter: Optional[RuleFilter] = None,
) -> str:
    """
    Convenience function to generate system prompt for direct API usage.

    Args:
        memory_manager: MemoryManager instance
        token_budget: Token budget (default: 10000 for API system prompts)
        compact: Use compact formatting (default: True)
        filter: Optional filter for memory rules

    Returns:
        Formatted system prompt string

    Example:
        >>> from context_injection import generate_api_system_prompt
        >>> prompt = await generate_api_system_prompt(memory_manager, compact=True)
    """
    config = SystemPromptConfig(
        token_budget=token_budget,
        injection_mode=InjectionMode.API,
        compact_format=compact,
        include_metadata=False,  # Cleaner for API usage
    )

    injector = SystemPromptInjector(memory_manager)
    return await injector.generate_system_prompt(config=config, filter=filter)
