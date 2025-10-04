# LLM-Specific Formatting Strategies

**Version:** 0.2.1dev1
**Date:** 2025-10-04
**Task:** 295.3 - Design LLM-specific formatting strategies

## Table of Contents

1. [Overview](#overview)
2. [Format Adapter Architecture](#format-adapter-architecture)
3. [Claude Code Format](#claude-code-format)
4. [GitHub Codex Format](#github-codex-format)
5. [Google Gemini Format](#google-gemini-format)
6. [Format Validation](#format-validation)
7. [Template System](#template-system)
8. [Implementation Plan](#implementation-plan)

## Overview

LLM-specific formatting strategies convert memory rules from the generic MemoryRule format into tool-specific prompt formats optimized for each LLM's context injection mechanism.

### Design Principles

1. **Tool-Specific Optimization**: Format rules according to each tool's best practices
2. **Maintainability**: Common abstraction with tool-specific overrides
3. **Extensibility**: Easy addition of new LLM tools
4. **Validation**: Ensure formatted output meets tool requirements
5. **Efficiency**: Minimize token usage while maintaining clarity

### Formatting Pipeline

```
MemoryRules → Format Manager → Tool Adapter → Formatted Context → LLM Tool
                    ↓
              Template System
```

## Format Adapter Architecture

### Base Adapter Interface

**File:** `src/python/common/core/context_injection/formatters/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ...memory import MemoryRule, AuthorityLevel


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
        self,
        rules: List[MemoryRule]
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
                getattr(r.metadata, 'priority', 50) if r.metadata else 50,  # Priority
                -r.created_at.timestamp()  # Most recent first
            ),
            reverse=True
        )
```

### Format Manager

**File:** `src/python/common/core/context_injection/formatters/manager.py`

```python
from typing import Dict, List, Optional
from loguru import logger

from .base import LLMToolAdapter, FormattedContext
from .claude_code import ClaudeCodeAdapter
from .github_codex import GitHubCodexAdapter
from .google_gemini import GoogleGeminiAdapter
from ...memory import MemoryRule


class FormatManager:
    """
    Manages LLM tool adapters and routes formatting requests.
    """

    def __init__(self):
        """Initialize the format manager with built-in adapters."""
        self._adapters: Dict[str, LLMToolAdapter] = {}
        self._register_builtin_adapters()

    def _register_builtin_adapters(self):
        """Register built-in tool adapters."""
        self.register_adapter("claude", ClaudeCodeAdapter())
        self.register_adapter("codex", GitHubCodexAdapter())
        self.register_adapter("gemini", GoogleGeminiAdapter())

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
```

## Claude Code Format

### Format Specification

**Target:** `.claude/context.md` file or system prompt injection

**Format:** Markdown with hierarchical sections

**Characteristics:**
- Supports markdown formatting
- Section-based organization
- Clear authority level separation
- Scope annotations

### Claude Code Adapter

**File:** `src/python/common/core/context_injection/formatters/claude_code.py`

```python
from typing import Dict, List, Optional
from collections import defaultdict

from .base import (
    LLMToolAdapter,
    ToolCapabilities,
    FormatType,
    FormattedContext,
)
from ...memory import MemoryRule, MemoryCategory


class ClaudeCodeAdapter(LLMToolAdapter):
    """
    Format adapter for Claude Code.

    Generates markdown-formatted context optimized for Claude Code's
    context window and instruction following.
    """

    def __init__(self):
        capabilities = ToolCapabilities(
            tool_name="claude",
            format_type=FormatType.MARKDOWN,
            max_context_tokens=100000,  # Claude 3.5 Sonnet context window
            supports_sections=True,
            supports_markdown=True,
            supports_priorities=True,
            injection_method="file",  # .claude/context.md
        )
        super().__init__(capabilities)

    def format_rules(
        self,
        rules: List[MemoryRule],
        token_budget: int,
        options: Optional[Dict[str, any]] = None,
    ) -> FormattedContext:
        """
        Format rules as markdown for Claude Code.

        Format structure:
        ```markdown
        # Project Memory Rules

        ## CRITICAL RULES (Always Follow)
        These rules are non-negotiable and must always be followed.

        ### [Category Name]
        - **[Rule Name]**: [Rule Text]
          - Scope: [scope1, scope2]
          - Source: [source]

        ## DEFAULT GUIDELINES (Unless Overridden)
        These are recommended practices unless explicitly overridden.

        ### [Category Name]
        - **[Rule Name]**: [Rule Text]
          - Scope: [scope1, scope2]
        ```
        """
        options = options or {}

        # Separate by authority
        absolute_rules, default_rules = self.separate_by_authority(rules)

        # Sort by priority
        absolute_rules = self.sort_by_priority(absolute_rules)
        default_rules = self.sort_by_priority(default_rules)

        # Build sections
        sections = []
        token_count = 0
        rules_included = 0
        rules_skipped = 0

        # Header
        header = "# Project Memory Rules\n\n"
        header += "**Auto-generated context from memory system**\n\n"
        sections.append(header)
        token_count += self.estimate_token_count(header)

        # Absolute rules section
        if absolute_rules:
            absolute_section, stats = self._format_rule_section(
                title="## CRITICAL RULES (Always Follow)",
                description="These rules are non-negotiable and must always be followed.\n",
                rules=absolute_rules,
                remaining_budget=token_budget - token_count,
                include_all=True,  # Always include absolute rules
            )
            sections.append(absolute_section)
            token_count += stats['tokens']
            rules_included += stats['included']
            rules_skipped += stats['skipped']

        # Default rules section
        if default_rules and token_count < token_budget:
            default_section, stats = self._format_rule_section(
                title="## DEFAULT GUIDELINES (Unless Overridden)",
                description="These are recommended practices unless explicitly overridden by the user or PRD.\n",
                rules=default_rules,
                remaining_budget=token_budget - token_count,
                include_all=False,  # Can truncate default rules
            )
            sections.append(default_section)
            token_count += stats['tokens']
            rules_included += stats['included']
            rules_skipped += stats['skipped']

        # Footer
        if rules_skipped > 0:
            footer = f"\n---\n*{rules_skipped} additional rules skipped due to token budget constraints.*\n"
            sections.append(footer)
            token_count += self.estimate_token_count(footer)

        content = "\n".join(sections)

        return FormattedContext(
            tool_name="claude",
            format_type=FormatType.MARKDOWN,
            content=content,
            token_count=token_count,
            rules_included=rules_included,
            rules_skipped=rules_skipped,
            metadata={
                "absolute_rules": len(absolute_rules),
                "default_rules": len(default_rules),
            }
        )

    def _format_rule_section(
        self,
        title: str,
        description: str,
        rules: List[MemoryRule],
        remaining_budget: int,
        include_all: bool = False,
    ) -> tuple[str, Dict[str, int]]:
        """
        Format a section of rules grouped by category.

        Args:
            title: Section title
            description: Section description
            rules: Rules to include
            remaining_budget: Remaining token budget
            include_all: If True, include all rules regardless of budget

        Returns:
            Tuple of (formatted_section, stats_dict)
        """
        section_lines = [title, "", description]
        section_tokens = self.estimate_token_count("\n".join(section_lines))
        rules_included = 0
        rules_skipped = 0

        # Group by category
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        # Format each category
        for category in sorted(by_category.keys(), key=lambda c: c.value):
            category_rules = by_category[category]
            category_title = f"\n### {category.value.title()}\n"
            category_tokens = self.estimate_token_count(category_title)

            # Check budget for category header
            if not include_all and section_tokens + category_tokens > remaining_budget:
                rules_skipped += len(category_rules)
                continue

            section_lines.append(category_title)
            section_tokens += category_tokens

            # Format individual rules
            for rule in category_rules:
                rule_text = self._format_single_rule(rule)
                rule_tokens = self.estimate_token_count(rule_text)

                # Check budget
                if not include_all and section_tokens + rule_tokens > remaining_budget:
                    rules_skipped += 1
                    continue

                section_lines.append(rule_text)
                section_tokens += rule_tokens
                rules_included += 1

        return (
            "\n".join(section_lines),
            {
                'tokens': section_tokens,
                'included': rules_included,
                'skipped': rules_skipped,
            }
        )

    def _format_single_rule(self, rule: MemoryRule) -> str:
        """Format a single rule as markdown list item."""
        lines = [f"- **{rule.name}**: {rule.rule}"]

        # Add scope if specified
        if rule.scope:
            lines.append(f"  - Scope: {', '.join(rule.scope)}")

        # Add source if not default
        if rule.source and rule.source != "user_explicit":
            lines.append(f"  - Source: {rule.source}")

        # Add conditions if present
        if rule.conditions:
            condition_str = str(rule.conditions.get('condition', ''))
            if condition_str:
                lines.append(f"  - Condition: {condition_str}")

        return "\n".join(lines) + "\n"

    def get_capabilities(self) -> ToolCapabilities:
        """Return Claude Code capabilities."""
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """
        Validate formatted content for Claude Code.

        Checks:
        - Content is not empty
        - Markdown structure is valid
        - Token count within limits
        """
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        # Basic markdown validation
        if not formatted_context.content.startswith("#"):
            return False

        return True
```

### Example Claude Code Output

```markdown
# Project Memory Rules

**Auto-generated context from memory system**

## CRITICAL RULES (Always Follow)
These rules are non-negotiable and must always be followed.

### Behavior
- **atomic_commits**: Always make atomic commits with clear, descriptive messages
  - Scope: git
  - Source: user_explicit

- **test_driven_development**: Write unit tests immediately after implementing each function
  - Scope: python, testing
  - Source: conversational_directive

### Preference
- **python_package_manager**: Use uv for all Python package management operations
  - Scope: python
  - Source: user_explicit

## DEFAULT GUIDELINES (Unless Overridden)
These are recommended practices unless explicitly overridden by the user or PRD.

### Behavior
- **code_review_process**: Submit PRs for review before merging to main
  - Scope: git, collaboration
  - Source: conversational_future

- **documentation_updates**: Update relevant documentation when changing public APIs
  - Scope: documentation
  - Condition: when modifying public interfaces

### Preference
- **testing_framework**: Prefer pytest over unittest for Python testing
  - Scope: python, testing
  - Source: conversational_preference
```

## GitHub Codex Format

### Format Specification

**Target:** `.github/codex/context.txt` or inline comments

**Format:** Plain text with structured sections

**Characteristics:**
- Simple plain text format
- Line-based structure
- Minimal formatting
- Explicit rule numbering

### GitHub Codex Adapter

**File:** `src/python/common/core/context_injection/formatters/github_codex.py`

```python
from typing import Dict, List, Optional
from collections import defaultdict

from .base import (
    LLMToolAdapter,
    ToolCapabilities,
    FormatType,
    FormattedContext,
)
from ...memory import MemoryRule


class GitHubCodexAdapter(LLMToolAdapter):
    """
    Format adapter for GitHub Codex.

    Generates plain text format optimized for Codex's context processing.
    """

    def __init__(self):
        capabilities = ToolCapabilities(
            tool_name="codex",
            format_type=FormatType.PLAIN_TEXT,
            max_context_tokens=4096,  # Codex context limit
            supports_sections=True,
            supports_markdown=False,
            supports_priorities=False,
            injection_method="file",  # .github/codex/context.txt
        )
        super().__init__(capabilities)

    def format_rules(
        self,
        rules: List[MemoryRule],
        token_budget: int,
        options: Optional[Dict[str, any]] = None,
    ) -> FormattedContext:
        """
        Format rules as plain text for GitHub Codex.

        Format structure:
        ```
        PROJECT CONTEXT RULES
        Generated: 2025-10-04T20:00:00Z

        === CRITICAL RULES (ALWAYS FOLLOW) ===

        [1] atomic_commits (Behavior)
        Always make atomic commits with clear, descriptive messages
        Scope: git

        [2] test_driven_development (Behavior)
        Write unit tests immediately after implementing each function
        Scope: python, testing

        === DEFAULT GUIDELINES ===

        [3] code_review_process (Behavior)
        Submit PRs for review before merging to main
        Scope: git, collaboration
        ```
        """
        options = options or {}

        # Separate by authority
        absolute_rules, default_rules = self.separate_by_authority(rules)

        # Sort by priority
        absolute_rules = self.sort_by_priority(absolute_rules)
        default_rules = self.sort_by_priority(default_rules)

        # Build sections
        sections = []
        token_count = 0
        rules_included = 0
        rules_skipped = 0
        rule_number = 1

        # Header
        from datetime import datetime, timezone
        header = "PROJECT CONTEXT RULES\n"
        header += f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"
        sections.append(header)
        token_count += self.estimate_token_count(header)

        # Absolute rules
        if absolute_rules:
            section_header = "=== CRITICAL RULES (ALWAYS FOLLOW) ===\n\n"
            sections.append(section_header)
            token_count += self.estimate_token_count(section_header)

            for rule in absolute_rules:
                rule_text = self._format_single_rule(rule, rule_number)
                rule_tokens = self.estimate_token_count(rule_text)

                if token_count + rule_tokens <= token_budget:
                    sections.append(rule_text)
                    token_count += rule_tokens
                    rules_included += 1
                    rule_number += 1
                else:
                    rules_skipped += 1

        # Default rules
        if default_rules and token_count < token_budget:
            section_header = "\n=== DEFAULT GUIDELINES ===\n\n"
            sections.append(section_header)
            token_count += self.estimate_token_count(section_header)

            for rule in default_rules:
                rule_text = self._format_single_rule(rule, rule_number)
                rule_tokens = self.estimate_token_count(rule_text)

                if token_count + rule_tokens <= token_budget:
                    sections.append(rule_text)
                    token_count += rule_tokens
                    rules_included += 1
                    rule_number += 1
                else:
                    rules_skipped += 1

        # Footer
        if rules_skipped > 0:
            footer = f"\n--- {rules_skipped} rules omitted due to context limits ---\n"
            sections.append(footer)
            token_count += self.estimate_token_count(footer)

        content = "".join(sections)

        return FormattedContext(
            tool_name="codex",
            format_type=FormatType.PLAIN_TEXT,
            content=content,
            token_count=token_count,
            rules_included=rules_included,
            rules_skipped=rules_skipped,
            metadata={
                "absolute_rules": len(absolute_rules),
                "default_rules": len(default_rules),
            }
        )

    def _format_single_rule(self, rule: MemoryRule, number: int) -> str:
        """Format a single rule in plain text."""
        lines = [
            f"[{number}] {rule.name} ({rule.category.value.title()})",
            rule.rule,
        ]

        if rule.scope:
            lines.append(f"Scope: {', '.join(rule.scope)}")

        if rule.conditions:
            condition = rule.conditions.get('condition', '')
            if condition:
                lines.append(f"Condition: {condition}")

        return "\n".join(lines) + "\n\n"

    def get_capabilities(self) -> ToolCapabilities:
        """Return Codex capabilities."""
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """Validate formatted content for Codex."""
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        return True
```

## Google Gemini Format

### Format Specification

**Target:** System instructions via API

**Format:** Structured text with clear sections

**Characteristics:**
- Clear section markers
- Concise formatting
- Numbered rules for reference
- Explicit instruction phrasing

### Google Gemini Adapter

**File:** `src/python/common/core/context_injection/formatters/google_gemini.py`

```python
from typing import Dict, List, Optional

from .base import (
    LLMToolAdapter,
    ToolCapabilities,
    FormatType,
    FormattedContext,
)
from ...memory import MemoryRule


class GoogleGeminiAdapter(LLMToolAdapter):
    """
    Format adapter for Google Gemini.

    Generates system instruction format optimized for Gemini's
    instruction following capabilities.
    """

    def __init__(self):
        capabilities = ToolCapabilities(
            tool_name="gemini",
            format_type=FormatType.SYSTEM_INSTRUCTION,
            max_context_tokens=32000,  # Gemini 1.5 Pro context
            supports_sections=True,
            supports_markdown=False,
            supports_priorities=True,
            injection_method="api",  # System instructions parameter
        )
        super().__init__(capabilities)

    def format_rules(
        self,
        rules: List[MemoryRule],
        token_budget: int,
        options: Optional[Dict[str, any]] = None,
    ) -> FormattedContext:
        """
        Format rules as system instructions for Google Gemini.

        Format structure:
        ```
        You are an AI assistant with the following context-specific rules and guidelines.

        MANDATORY RULES:
        You MUST follow these rules without exception:

        1. atomic_commits: Always make atomic commits with clear, descriptive messages
           [Scope: git]

        2. test_driven_development: Write unit tests immediately after implementing each function
           [Scope: python, testing]

        RECOMMENDED GUIDELINES:
        Follow these guidelines unless explicitly overridden:

        3. code_review_process: Submit PRs for review before merging to main
           [Scope: git, collaboration]
        ```
        """
        options = options or {}

        # Separate by authority
        absolute_rules, default_rules = self.separate_by_authority(rules)

        # Sort by priority
        absolute_rules = self.sort_by_priority(absolute_rules)
        default_rules = self.sort_by_priority(default_rules)

        # Build sections
        sections = []
        token_count = 0
        rules_included = 0
        rules_skipped = 0
        rule_number = 1

        # System instruction header
        header = "You are an AI assistant with the following context-specific rules and guidelines.\n\n"
        sections.append(header)
        token_count += self.estimate_token_count(header)

        # Mandatory rules
        if absolute_rules:
            mandatory_header = "MANDATORY RULES:\nYou MUST follow these rules without exception:\n\n"
            sections.append(mandatory_header)
            token_count += self.estimate_token_count(mandatory_header)

            for rule in absolute_rules:
                rule_text = self._format_single_rule(rule, rule_number)
                rule_tokens = self.estimate_token_count(rule_text)

                if token_count + rule_tokens <= token_budget:
                    sections.append(rule_text)
                    token_count += rule_tokens
                    rules_included += 1
                    rule_number += 1
                else:
                    rules_skipped += 1

        # Recommended guidelines
        if default_rules and token_count < token_budget:
            guidelines_header = "\nRECOMMENDED GUIDELINES:\nFollow these guidelines unless explicitly overridden:\n\n"
            sections.append(guidelines_header)
            token_count += self.estimate_token_count(guidelines_header)

            for rule in default_rules:
                rule_text = self._format_single_rule(rule, rule_number)
                rule_tokens = self.estimate_token_count(rule_text)

                if token_count + rule_tokens <= token_budget:
                    sections.append(rule_text)
                    token_count += rule_tokens
                    rules_included += 1
                    rule_number += 1
                else:
                    rules_skipped += 1

        content = "".join(sections)

        return FormattedContext(
            tool_name="gemini",
            format_type=FormatType.SYSTEM_INSTRUCTION,
            content=content,
            token_count=token_count,
            rules_included=rules_included,
            rules_skipped=rules_skipped,
            metadata={
                "absolute_rules": len(absolute_rules),
                "default_rules": len(default_rules),
            }
        )

    def _format_single_rule(self, rule: MemoryRule, number: int) -> str:
        """Format a single rule for Gemini."""
        text = f"{number}. {rule.name}: {rule.rule}\n"

        if rule.scope:
            text += f"   [Scope: {', '.join(rule.scope)}]\n"

        if rule.conditions:
            condition = rule.conditions.get('condition', '')
            if condition:
                text += f"   [Condition: {condition}]\n"

        return text + "\n"

    def get_capabilities(self) -> ToolCapabilities:
        """Return Gemini capabilities."""
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """Validate formatted content for Gemini."""
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        # Should start with system instruction
        if not formatted_context.content.startswith("You are"):
            return False

        return True
```

## Format Validation

### Validation Framework

**Common Validation Checks:**

1. **Content Existence**: Formatted content is not empty
2. **Token Budget**: Content fits within tool's token limits
3. **Format Structure**: Correct structure for tool format type
4. **Character Encoding**: Valid UTF-8 encoding
5. **Special Characters**: Proper escaping of tool-specific characters

**Tool-Specific Validation:**

```python
# Claude Code: Markdown structure
def validate_markdown_structure(content: str) -> bool:
    """Validate markdown structure."""
    if not content.startswith("#"):
        return False

    # Check for required sections
    required_sections = ["# Project Memory Rules", "## CRITICAL RULES", "## DEFAULT"]
    for section in required_sections:
        if section not in content:
            return False

    return True


# GitHub Codex: Plain text format
def validate_plain_text_format(content: str) -> bool:
    """Validate plain text format."""
    if "PROJECT CONTEXT RULES" not in content:
        return False

    if "===" not in content:  # Section markers
        return False

    return True


# Google Gemini: System instruction format
def validate_system_instruction(content: str) -> bool:
    """Validate system instruction format."""
    if not content.startswith("You are"):
        return False

    if "MANDATORY RULES" not in content:
        return False

    return True
```

## Template System

### Template Architecture

**Purpose:** Provide customizable templates for different use cases

**Template Location:** `src/python/common/core/context_injection/templates/`

**Template Types:**
1. Default templates (built-in)
2. Custom templates (user-provided)
3. Project-specific templates (per-project overrides)

### Template Definition

```python
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ContextTemplate:
    """
    Template for formatting memory rules.

    Attributes:
        name: Template name
        tool_name: Target tool name
        header_template: Template for section headers
        rule_template: Template for individual rules
        footer_template: Template for section footers
        variables: Available template variables
    """
    name: str
    tool_name: str
    header_template: str
    rule_template: str
    footer_template: str
    variables: Dict[str, str]


# Example: Claude Code template
CLAUDE_CODE_TEMPLATE = ContextTemplate(
    name="claude_default",
    tool_name="claude",
    header_template="""# {project_name} Memory Rules

**Generated:** {timestamp}
**Project:** {project_id}

""",
    rule_template="""- **{rule_name}**: {rule_text}
  - Scope: {scope}
  - Authority: {authority}
""",
    footer_template="""
---
*{rules_count} rules loaded from memory system.*
""",
    variables={
        "project_name": "Current project name",
        "timestamp": "ISO 8601 timestamp",
        "project_id": "Project identifier",
        "rule_name": "Rule name",
        "rule_text": "Rule content",
        "scope": "Comma-separated scopes",
        "authority": "Authority level",
        "rules_count": "Total rules included",
    }
)
```

## Implementation Plan

### Phase 1: Base Infrastructure

**Files to Create:**
1. `src/python/common/core/context_injection/formatters/__init__.py`
2. `src/python/common/core/context_injection/formatters/base.py`
3. `src/python/common/core/context_injection/formatters/manager.py`

**Tasks:**
- [ ] Implement LLMToolAdapter abstract base class
- [ ] Implement ToolCapabilities dataclass
- [ ] Implement FormattedContext dataclass
- [ ] Implement FormatManager with adapter registry

### Phase 2: Tool-Specific Adapters

**Files to Create:**
1. `src/python/common/core/context_injection/formatters/claude_code.py`
2. `src/python/common/core/context_injection/formatters/github_codex.py`
3. `src/python/common/core/context_injection/formatters/google_gemini.py`

**Tasks:**
- [ ] Implement ClaudeCodeAdapter with markdown formatting
- [ ] Implement GitHubCodexAdapter with plain text formatting
- [ ] Implement GoogleGeminiAdapter with system instruction formatting
- [ ] Add token counting logic for each adapter
- [ ] Implement authority level separation

### Phase 3: Validation Framework

**Files to Create:**
1. `src/python/common/core/context_injection/formatters/validation.py`

**Tasks:**
- [ ] Implement validation utilities
- [ ] Add format-specific validation logic
- [ ] Implement validation reporting

### Phase 4: Template System

**Files to Create:**
1. `src/python/common/core/context_injection/templates/__init__.py`
2. `src/python/common/core/context_injection/templates/template.py`
3. `src/python/common/core/context_injection/templates/defaults.py`

**Tasks:**
- [ ] Implement ContextTemplate dataclass
- [ ] Create default templates for each tool
- [ ] Implement template variable substitution
- [ ] Add custom template loading

### Phase 5: Testing

**Files to Create:**
1. `tests/unit/test_formatters.py`
2. `tests/integration/test_formatting_pipeline.py`

**Tasks:**
- [ ] Unit tests for each adapter
- [ ] Token count accuracy tests
- [ ] Format validation tests
- [ ] Template substitution tests
- [ ] Integration tests with real rules

## References

- **llm-context-injection.md**: Overall architecture
- **rule-fetching-mechanism.md**: Rule retrieval design (dependency)
- **src/python/common/core/memory.py**: MemoryRule schema
- **PRDv3.txt**: System specification
