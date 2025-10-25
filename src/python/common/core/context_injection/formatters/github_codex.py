"""
GitHub Codex format adapter for memory rule formatting.

This module provides plain text format optimized for GitHub Codex's
smaller context window and code-focused interactions.
"""

from datetime import datetime, timezone

from ...memory import MemoryRule
from .base import FormattedContext, FormatType, LLMToolAdapter, ToolCapabilities


class GitHubCodexAdapter(LLMToolAdapter):
    """
    Format adapter for GitHub Codex.

    Generates plain text format optimized for Codex's context processing
    and smaller context window.
    """

    def __init__(self):
        """Initialize GitHub Codex adapter with capabilities."""
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
        rules: list[MemoryRule],
        token_budget: int,
        options: dict[str, any] | None = None,
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

        Args:
            rules: List of memory rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with plain text content
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
            },
        )

    def _format_single_rule(self, rule: MemoryRule, number: int) -> str:
        """
        Format a single rule in plain text.

        Args:
            rule: Memory rule to format
            number: Rule number for reference

        Returns:
            Formatted plain text string
        """
        lines = [
            f"[{number}] {rule.name} ({rule.category.value.title()})",
            rule.rule,
        ]

        if rule.scope:
            lines.append(f"Scope: {', '.join(rule.scope)}")

        if rule.conditions:
            condition = rule.conditions.get("condition", "")
            if condition:
                lines.append(f"Condition: {condition}")

        return "\n".join(lines) + "\n\n"

    def get_capabilities(self) -> ToolCapabilities:
        """
        Return Codex capabilities.

        Returns:
            ToolCapabilities for GitHub Codex
        """
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """
        Validate formatted content for Codex.

        Checks:
        - Content is not empty
        - Token count within limits
        - Expected structure present

        Args:
            formatted_context: The formatted context to validate

        Returns:
            True if valid, False otherwise
        """
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        # Should have expected header
        if "PROJECT CONTEXT RULES" not in formatted_context.content:
            return False

        return True
