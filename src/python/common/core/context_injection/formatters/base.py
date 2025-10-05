"""
Base adapter interface for LLM-specific context formatting.

This module defines the abstract base class and supporting dataclasses
for formatting memory rules into LLM-specific formats.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from ...memory import AuthorityLevel, MemoryRule


class FormatType(Enum):
    """Supported format types for LLM tools."""

    MARKDOWN = "markdown"
    PLAIN_TEXT = "plain"
    XML = "xml"
    JSON = "json"
    SYSTEM_INSTRUCTION = "system_instruction"


@dataclass
class ToolCapabilities:
    """
    Capabilities of an LLM tool for context injection.

    Attributes:
        tool_name: Name of the tool (e.g., "claude", "codex", "gemini")
        format_type: Preferred format type
        max_context_tokens: Maximum context size in tokens
        supports_sections: Whether tool supports sectioned context
        supports_markdown: Whether tool supports markdown formatting
        supports_priorities: Whether tool respects priority levels
        injection_method: How context is injected ("file", "api", "cli")
    """

    tool_name: str
    format_type: FormatType
    max_context_tokens: int
    supports_sections: bool
    supports_markdown: bool
    supports_priorities: bool
    injection_method: str


@dataclass
class FormattedContext:
    """
    Formatted context ready for injection.

    Attributes:
        tool_name: Target tool name
        format_type: Format type used
        content: Formatted content string
        token_count: Estimated token count
        rules_included: Number of rules included
        rules_skipped: Number of rules skipped (over budget)
        metadata: Additional formatting metadata
    """

    tool_name: str
    format_type: FormatType
    content: str
    token_count: int
    rules_included: int
    rules_skipped: int
    metadata: Dict[str, any]


class LLMToolAdapter(ABC):
    """
    Base adapter for formatting memory rules for LLM tools.

    Each LLM tool (Claude Code, GitHub Codex, Google Gemini) implements
    this interface with tool-specific formatting logic.
    """

    def __init__(self, capabilities: ToolCapabilities):
        """
        Initialize the adapter.

        Args:
            capabilities: Tool capabilities for this adapter
        """
        self.capabilities = capabilities

    @abstractmethod
    def format_rules(
        self,
        rules: List[MemoryRule],
        token_budget: int,
        options: Optional[Dict[str, any]] = None,
    ) -> FormattedContext:
        """
        Format memory rules for this LLM tool.

        Args:
            rules: List of memory rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with formatted content
        """
        pass

    @abstractmethod
    def get_capabilities(self) -> ToolCapabilities:
        """
        Get the capabilities of this LLM tool.

        Returns:
            ToolCapabilities describing what this tool supports
        """
        pass

    @abstractmethod
    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """
        Validate that formatted content meets tool requirements.

        Args:
            formatted_context: The formatted context to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for text using tool-specific logic.

        Default implementation uses simple heuristics. Override for
        tool-specific tokenizers.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple approximation: 1.3x words + 0.5x punctuation
        words = len(text.split())
        punctuation = len([c for c in text if c in '.,!?;:()[]{}"\'-'])
        return int(words * 1.3 + punctuation * 0.5)

    def separate_by_authority(
        self, rules: List[MemoryRule]
    ) -> tuple[List[MemoryRule], List[MemoryRule]]:
        """
        Separate rules by authority level.

        Args:
            rules: List of memory rules

        Returns:
            Tuple of (absolute_rules, default_rules)
        """
        absolute = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]
        return absolute, default

    def sort_by_priority(self, rules: List[MemoryRule]) -> List[MemoryRule]:
        """
        Sort rules by priority (within same authority level).

        Args:
            rules: List of memory rules

        Returns:
            Sorted list of rules
        """
        return sorted(
            rules,
            key=lambda r: (
                r.authority == AuthorityLevel.ABSOLUTE,  # Absolute first
                getattr(r.metadata, "priority", 50)
                if r.metadata
                else 50,  # Priority
                -r.created_at.timestamp(),  # Most recent first
            ),
            reverse=True,
        )
