"""
Google Gemini format adapter for memory rule formatting.

This module provides system instruction format optimized for Google Gemini's
instruction following and larger context window.
"""

from typing import Dict, List, Optional

from ...memory import MemoryRule
from .base import FormattedContext, FormatType, LLMToolAdapter, ToolCapabilities


class GoogleGeminiAdapter(LLMToolAdapter):
    """
    Format adapter for Google Gemini.

    Generates system instruction format optimized for Gemini's
    instruction following capabilities.
    """

    def __init__(self):
        """Initialize Google Gemini adapter with capabilities."""
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

        Args:
            rules: List of memory rules to format
            token_budget: Available token budget
            options: Optional formatting options

        Returns:
            FormattedContext with system instruction content
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
            mandatory_header = (
                "MANDATORY RULES:\nYou MUST follow these rules without exception:\n\n"
            )
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
            },
        )

    def _format_single_rule(self, rule: MemoryRule, number: int) -> str:
        """
        Format a single rule for Gemini.

        Args:
            rule: Memory rule to format
            number: Rule number for reference

        Returns:
            Formatted system instruction string
        """
        text = f"{number}. {rule.name}: {rule.rule}\n"

        if rule.scope:
            text += f"   [Scope: {', '.join(rule.scope)}]\n"

        if rule.conditions:
            condition = rule.conditions.get("condition", "")
            if condition:
                text += f"   [Condition: {condition}]\n"

        return text + "\n"

    def get_capabilities(self) -> ToolCapabilities:
        """
        Return Gemini capabilities.

        Returns:
            ToolCapabilities for Google Gemini
        """
        return self.capabilities

    def validate_format(self, formatted_context: FormattedContext) -> bool:
        """
        Validate formatted content for Gemini.

        Checks:
        - Content is not empty
        - Token count within limits
        - System instruction format

        Args:
            formatted_context: The formatted context to validate

        Returns:
            True if valid, False otherwise
        """
        if not formatted_context.content:
            return False

        if formatted_context.token_count > self.capabilities.max_context_tokens:
            return False

        # Should start with system instruction
        if not formatted_context.content.startswith("You are"):
            return False

        return True
