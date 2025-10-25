"""
Claude Code format adapter for memory rule formatting.

This module provides markdown-formatted context optimized for Claude Code's
context window and instruction following capabilities.
"""

from collections import defaultdict
from datetime import datetime, timezone

from ...memory import MemoryRule
from .base import FormattedContext, FormatType, LLMToolAdapter, ToolCapabilities


class ClaudeCodeAdapter(LLMToolAdapter):
    """
    Format adapter for Claude Code.

    Generates markdown-formatted context optimized for Claude Code's
    context window and instruction following.
    """

    def __init__(self):
        """Initialize Claude Code adapter with capabilities."""
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
        rules: list[MemoryRule],
        token_budget: int,
        options: dict[str, any] | None = None,
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

        Args:
            rules: List of memory rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with markdown content
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
        header += "**Auto-generated context from memory system**\n"
        header += f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n"
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
            token_count += stats["tokens"]
            rules_included += stats["included"]
            rules_skipped += stats["skipped"]

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
            token_count += stats["tokens"]
            rules_included += stats["included"]
            rules_skipped += stats["skipped"]

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
            },
        )

    def _format_rule_section(
        self,
        title: str,
        description: str,
        rules: list[MemoryRule],
        remaining_budget: int,
        include_all: bool = False,
    ) -> tuple[str, dict[str, int]]:
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
                "tokens": section_tokens,
                "included": rules_included,
                "skipped": rules_skipped,
            },
        )

    def _format_single_rule(self, rule: MemoryRule) -> str:
        """
        Format a single rule as markdown list item.

        Args:
            rule: Memory rule to format

        Returns:
            Formatted markdown string
        """
        lines = [f"- **{rule.name}**: {rule.rule}"]

        # Add scope if specified
        if rule.scope:
            lines.append(f"  - Scope: {', '.join(rule.scope)}")

        # Add source if not default
        if rule.source and rule.source != "user_explicit":
            lines.append(f"  - Source: {rule.source}")

        # Add conditions if present
        if rule.conditions:
            condition_str = str(rule.conditions.get("condition", ""))
            if condition_str:
                lines.append(f"  - Condition: {condition_str}")

        return "\n".join(lines) + "\n"

    def get_capabilities(self) -> ToolCapabilities:
        """
        Return Claude Code capabilities.

        Returns:
            ToolCapabilities for Claude Code
        """
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """
        Validate formatted content for Claude Code.

        Checks:
        - Content is not empty
        - Markdown structure is valid
        - Token count within limits

        Args:
            formatted_context: The formatted context to validate

        Returns:
            True if valid, False otherwise
        """
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        # Basic markdown validation
        if not formatted_context.content.startswith("#"):
            return False

        return True
