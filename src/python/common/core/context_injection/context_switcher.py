"""
Context switching validation system for context injection.

This module provides validation and execution of context switches between different
LLM tools (e.g., Claude Code to GitHub Copilot). Ensures token limits are respected,
formats are compatible, and users are warned about potential data loss.

Key Features:
- Validates token limits before switching
- Reformats rules for target tool
- Detects format compatibility issues
- Generates clear warnings and errors
- Supports automatic truncation with smart prioritization
- Handles upgrade/downgrade scenarios

Example Usage:
    # Validate a potential switch
    result = ContextSwitcher.validate_switch(
        source_tool=LLMToolType.CLAUDE_CODE,
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=rules,
        current_token_count=50000
    )

    if result.is_valid:
        print("Switch is safe!")
        for warning in result.warnings:
            print(f"Warning: {warning}")
    else:
        print("Cannot switch:")
        for error in result.errors:
            print(f"Error: {error}")

    # Perform switch with automatic truncation
    formatted_rules, result = ContextSwitcher.perform_switch(
        target_tool=LLMToolType.GITHUB_COPILOT,
        rules=rules,
        auto_truncate=True
    )
"""

from dataclasses import dataclass, field

from loguru import logger

from ..memory import MemoryRule
from .formatters import FormatManager
from .llm_tool_detector import LLMToolDetector, LLMToolType
from .rule_prioritizer import RulePrioritizer
from .tool_token_manager import ToolTokenManager


@dataclass
class SwitchValidationResult:
    """
    Results of context switching validation.

    Provides comprehensive information about the viability and consequences
    of switching from one LLM tool to another.

    Attributes:
        is_valid: Overall validation status (True if no errors)
        source_tool: Tool being switched from
        target_tool: Tool being switched to
        source_token_count: Token count in source format
        target_token_count: Token count in target format (after reformatting)
        token_limit_ok: Whether target token count fits within target limit
        format_compatible: Whether formats are compatible (always True in current impl)
        rules_truncated: Number of rules that won't fit in target limit
        warnings: List of warning messages
        errors: List of error messages
    """

    is_valid: bool
    source_tool: LLMToolType
    target_tool: LLMToolType
    source_token_count: int
    target_token_count: int
    token_limit_ok: bool
    format_compatible: bool
    rules_truncated: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        lines = [
            f"Switch Validation: {status}",
            f"  From: {self.source_tool.value} ({self.source_token_count:,} tokens)",
            f"  To: {self.target_tool.value} ({self.target_token_count:,} tokens)",
            f"  Token Limit OK: {self.token_limit_ok}",
            f"  Rules Truncated: {self.rules_truncated}",
        ]

        if self.warnings:
            lines.append("  Warnings:")
            for warning in self.warnings:
                lines.append(f"    - {warning}")

        if self.errors:
            lines.append("  Errors:")
            for error in self.errors:
                lines.append(f"    - {error}")

        return "\n".join(lines)


class ContextSwitcher:
    """
    Validates and performs context switching between LLM tools.

    All methods are static as there is no instance state. Uses FormatManager
    for reformatting, ToolTokenManager for validation, and RulePrioritizer
    for smart truncation.
    """

    @staticmethod
    def validate_switch(
        source_tool: LLMToolType,
        target_tool: LLMToolType,
        rules: list[MemoryRule],
        current_token_count: int,
    ) -> SwitchValidationResult:
        """
        Validate switching from source to target tool.

        Checks token limits, reformats rules, and generates warnings/errors.

        Args:
            source_tool: Tool being switched from
            target_tool: Tool being switched to
            rules: Rules to validate
            current_token_count: Current token count in source format

        Returns:
            SwitchValidationResult with detailed validation information

        Example:
            >>> result = ContextSwitcher.validate_switch(
            ...     LLMToolType.CLAUDE_CODE,
            ...     LLMToolType.GITHUB_COPILOT,
            ...     rules,
            ...     50000
            ... )
            >>> if not result.is_valid:
            ...     print("Cannot switch:", result.errors)
        """
        logger.debug(
            f"Validating switch from {source_tool.value} to {target_tool.value} "
            f"({len(rules)} rules, {current_token_count:,} tokens)"
        )

        # Quick check: same tool = no switch needed
        if source_tool == target_tool:
            return ContextSwitcher._validate_same_tool(
                source_tool, current_token_count, len(rules)
            )

        # Get token limits for both tools
        source_limits = ToolTokenManager.get_limits(source_tool)
        target_limits = ToolTokenManager.get_limits(target_tool)

        # Reformat rules for target tool
        format_manager = FormatManager()
        tool_name_mapping = {
            LLMToolType.CLAUDE_CODE: "claude",
            LLMToolType.GITHUB_COPILOT: "codex",
            LLMToolType.CODEX_API: "codex",
            LLMToolType.CURSOR: "codex",
            LLMToolType.JETBRAINS_AI: "codex",
            LLMToolType.GOOGLE_GEMINI: "gemini",
            LLMToolType.TABNINE: "codex",
        }

        target_tool_name = tool_name_mapping.get(target_tool, "codex")
        source_tool_name = tool_name_mapping.get(source_tool, "codex")

        if source_tool_name == target_tool_name:
            target_token_count = current_token_count
            format_compatible = True
            rules_truncated = 0
        else:
            # Format rules with very large budget to get full token count
            # We'll validate against actual limit after
            try:
                formatted = format_manager.format_for_tool(
                    target_tool_name, rules, target_limits.max_context_tokens
                )
                target_token_count = formatted.token_count
                adapter = format_manager.get_adapter(target_tool_name)
                format_compatible = adapter.validate_format(formatted) if adapter else True
                rules_truncated = formatted.rules_skipped
            except Exception as e:
                logger.error(f"Failed to reformat rules for {target_tool.value}: {e}")
                return SwitchValidationResult(
                    is_valid=False,
                    source_tool=source_tool,
                    target_tool=target_tool,
                    source_token_count=current_token_count,
                    target_token_count=0,
                    token_limit_ok=False,
                    format_compatible=False,
                    rules_truncated=len(rules),
                    warnings=[],
                    errors=[f"Failed to reformat rules for {target_tool.value}: {str(e)}"],
                )

        # Check if target tokens fit within target limit
        token_limit_ok = target_token_count <= target_limits.max_context_tokens

        # Calculate how many rules would be truncated if exceeding limit
        if not token_limit_ok:
            # Estimate how many rules fit
            prioritizer = RulePrioritizer()
            selected, skipped = prioritizer.select_top_rules(
                rules, target_tool_name, target_limits.max_context_tokens
            )
            rules_truncated = max(rules_truncated, len(skipped))

        # Generate warnings and errors
        warnings = ContextSwitcher._generate_warnings(
            source_tool,
            target_tool,
            source_limits,
            target_limits,
            current_token_count,
            target_token_count,
            rules_truncated,
            len(rules),
        )

        errors = ContextSwitcher._generate_errors(
            target_tool, target_token_count, target_limits, rules_truncated
        )

        is_valid = len(errors) == 0

        result = SwitchValidationResult(
            is_valid=is_valid,
            source_tool=source_tool,
            target_tool=target_tool,
            source_token_count=current_token_count,
            target_token_count=target_token_count,
            token_limit_ok=token_limit_ok,
            format_compatible=format_compatible,
            rules_truncated=rules_truncated,
            warnings=warnings,
            errors=errors,
        )

        logger.debug(f"Validation result: {result.is_valid}, {len(warnings)} warnings")

        return result

    @staticmethod
    def perform_switch(
        target_tool: LLMToolType,
        rules: list[MemoryRule],
        auto_truncate: bool = False,
    ) -> tuple[list[MemoryRule], SwitchValidationResult]:
        """
        Perform context switch to target tool.

        Reformats rules for target tool and optionally truncates if exceeding limits.

        Args:
            target_tool: Tool to switch to
            rules: Rules to switch
            auto_truncate: If True, automatically truncate rules to fit limit

        Returns:
            Tuple of (formatted_rules, validation_result)
            - formatted_rules: Rules formatted for target tool (possibly truncated)
            - validation_result: Detailed validation information

        Example:
            >>> rules, result = ContextSwitcher.perform_switch(
            ...     LLMToolType.GITHUB_COPILOT,
            ...     rules,
            ...     auto_truncate=True
            ... )
            >>> print(f"Switched with {result.rules_truncated} rules truncated")
        """
        # Auto-detect source tool
        session = LLMToolDetector.detect()
        source_tool = session.tool_type if session.is_active else LLMToolType.UNKNOWN

        logger.info(
            f"Performing switch from {source_tool.value} to {target_tool.value} "
            f"({len(rules)} rules, auto_truncate={auto_truncate})"
        )

        # Get target tool limits and formatter
        target_limits = ToolTokenManager.get_limits(target_tool)
        format_manager = FormatManager()

        tool_name_mapping = {
            LLMToolType.CLAUDE_CODE: "claude",
            LLMToolType.GITHUB_COPILOT: "codex",
            LLMToolType.CODEX_API: "codex",
            LLMToolType.CURSOR: "codex",
            LLMToolType.JETBRAINS_AI: "codex",
            LLMToolType.GOOGLE_GEMINI: "gemini",
            LLMToolType.TABNINE: "codex",
        }

        target_tool_name = tool_name_mapping.get(target_tool, "codex")

        # Format all rules to check token count
        try:
            formatted_all = format_manager.format_for_tool(
                target_tool_name, rules, target_limits.max_context_tokens
            )
            total_tokens = formatted_all.token_count
            rules_skipped = formatted_all.rules_skipped
        except Exception as e:
            logger.error(f"Failed to format rules: {e}")
            # Return original rules with error
            result = SwitchValidationResult(
                is_valid=False,
                source_tool=source_tool,
                target_tool=target_tool,
                source_token_count=0,
                target_token_count=0,
                token_limit_ok=False,
                format_compatible=False,
                rules_truncated=len(rules),
                warnings=[],
                errors=[f"Failed to format rules for {target_tool.value}: {str(e)}"],
            )
            return rules, result

        # Check if we need to truncate
        needs_truncate = (
            total_tokens > target_limits.max_context_tokens or rules_skipped > 0
        )

        if needs_truncate and auto_truncate:
            logger.warning(
                f"Rules exceed {target_tool.value} limit "
                f"({total_tokens:,} > {target_limits.max_context_tokens:,} tokens). "
                "Auto-truncating..."
            )

            # Use RulePrioritizer to select top rules that fit
            prioritizer = RulePrioritizer()
            selected_rules, skipped_rules = prioritizer.select_top_rules(
                rules, target_tool_name, target_limits.max_context_tokens
            )

            logger.info(
                f"Selected {len(selected_rules)}/{len(rules)} rules "
                f"({len(skipped_rules)} truncated)"
            )

            # Reformat selected rules
            formatted_selected = format_manager.format_for_tool(
                target_tool_name, selected_rules, target_limits.max_context_tokens
            )

            # Validate final result
            result = ContextSwitcher.validate_switch(
                source_tool, target_tool, selected_rules, formatted_selected.token_count
            )
            result.rules_truncated = len(skipped_rules)

            # Add truncation warning
            if len(skipped_rules) > 0:
                result.warnings.insert(
                    0,
                    f"Auto-truncated {len(skipped_rules)} rules to fit within "
                    f"{target_tool.value} token limit",
                )

            return selected_rules, result

        else:
            # No truncation needed/wanted - validate and return all rules
            result = ContextSwitcher.validate_switch(
                source_tool, target_tool, rules, total_tokens
            )

            return rules, result

    @staticmethod
    def can_switch_safely(
        source_tool: LLMToolType, target_tool: LLMToolType, rules: list[MemoryRule]
    ) -> bool:
        """
        Quick check if switching is safe (no errors).

        Args:
            source_tool: Tool being switched from
            target_tool: Tool being switched to
            rules: Rules to validate

        Returns:
            True if switch has no errors, False otherwise

        Example:
            >>> if ContextSwitcher.can_switch_safely(source, target, rules):
            ...     # Proceed with switch
            ...     pass
        """
        # Estimate current token count (use source tool formatter)
        format_manager = FormatManager()
        source_limits = ToolTokenManager.get_limits(source_tool)

        tool_name_mapping = {
            LLMToolType.CLAUDE_CODE: "claude",
            LLMToolType.GITHUB_COPILOT: "codex",
            LLMToolType.CODEX_API: "codex",
            LLMToolType.CURSOR: "codex",
            LLMToolType.JETBRAINS_AI: "codex",
            LLMToolType.GOOGLE_GEMINI: "gemini",
            LLMToolType.TABNINE: "codex",
        }

        source_tool_name = tool_name_mapping.get(source_tool, "codex")

        try:
            formatted = format_manager.format_for_tool(
                source_tool_name, rules, source_limits.max_context_tokens
            )
            current_token_count = formatted.token_count
        except Exception:
            # If we can't format, it's not safe
            return False

        result = ContextSwitcher.validate_switch(
            source_tool, target_tool, rules, current_token_count
        )

        return (
            result.is_valid
            and result.token_limit_ok
            and result.format_compatible
            and result.rules_truncated == 0
        )

    @staticmethod
    def _validate_same_tool(
        tool: LLMToolType, token_count: int, rule_count: int
    ) -> SwitchValidationResult:
        """
        Handle same-tool switch (no changes needed).

        Args:
            tool: Tool type (both source and target)
            token_count: Current token count
            rule_count: Number of rules

        Returns:
            SwitchValidationResult indicating no change needed
        """
        return SwitchValidationResult(
            is_valid=True,
            source_tool=tool,
            target_tool=tool,
            source_token_count=token_count,
            target_token_count=token_count,
            token_limit_ok=True,
            format_compatible=True,
            rules_truncated=0,
            warnings=[f"Already using {tool.value}, no switch needed"],
            errors=[],
        )

    @staticmethod
    def _generate_warnings(
        source_tool: LLMToolType,
        target_tool: LLMToolType,
        source_limits,
        target_limits,
        source_token_count: int,
        target_token_count: int,
        rules_truncated: int,
        total_rules: int,
    ) -> list[str]:
        """
        Generate warning messages for switch validation.

        Args:
            source_tool: Source tool type
            target_tool: Target tool type
            source_limits: Source tool token limits
            target_limits: Target tool token limits
            source_token_count: Token count in source format
            target_token_count: Token count in target format
            rules_truncated: Number of rules that won't fit
            total_rules: Total number of rules

        Returns:
            List of warning messages
        """
        warnings = []

        # Downgrade warning (high limit → low limit)
        if source_limits.max_context_tokens > target_limits.max_context_tokens:
            warnings.append(
                f"Switching from {source_tool.value} to {target_tool.value} will reduce "
                f"available context from {source_limits.max_context_tokens:,} to "
                f"{target_limits.max_context_tokens:,} tokens"
            )

        # Upgrade info (low limit → high limit)
        elif source_limits.max_context_tokens < target_limits.max_context_tokens:
            warnings.append(
                f"Switching to {target_tool.value} will increase available context from "
                f"{source_limits.max_context_tokens:,} to {target_limits.max_context_tokens:,} tokens"
            )

        # Format change warning (if token count changes significantly)
        token_change_pct = abs(target_token_count - source_token_count) / max(
            1, source_token_count
        )
        if token_change_pct > 0.1:  # >10% change
            if target_token_count > source_token_count:
                warnings.append(
                    f"Token usage will increase from {source_token_count:,} to "
                    f"{target_token_count:,} tokens after reformatting "
                    f"({token_change_pct * 100:.1f}% increase)"
                )
            else:
                warnings.append(
                    f"Token usage will decrease from {source_token_count:,} to "
                    f"{target_token_count:,} tokens after reformatting "
                    f"({token_change_pct * 100:.1f}% decrease)"
                )

        # Truncation warning
        if rules_truncated > 0:
            warnings.append(
                f"Switching to {target_tool.value} will require truncating {rules_truncated} "
                f"of {total_rules} rules (exceeds {target_limits.max_context_tokens:,} token limit)"
            )
            warnings.append(
                "Consider prioritizing rules or using a tool with a larger context window"
            )

        # Near-limit warning (>80% of target limit)
        usage_pct = (target_token_count / target_limits.max_context_tokens) * 100
        if target_token_count <= target_limits.max_context_tokens and usage_pct > 80:
            warnings.append(
                f"Target tool usage will be {usage_pct:.1f}% of {target_tool.value} limit "
                f"({target_token_count:,} / {target_limits.max_context_tokens:,} tokens)"
            )

        return warnings

    @staticmethod
    def _generate_errors(
        target_tool: LLMToolType,
        target_token_count: int,
        target_limits,
        rules_truncated: int,
    ) -> list[str]:
        """
        Generate error messages for switch validation.

        Args:
            target_tool: Target tool type
            target_token_count: Token count in target format
            target_limits: Target tool token limits
            rules_truncated: Number of rules that won't fit

        Returns:
            List of error messages (empty if no errors)
        """
        errors = []

        # Only generate error if exceeding limit AND we're reporting it as invalid
        # (perform_switch with auto_truncate=True will handle this gracefully)
        # This is just for validation without auto-truncate
        if target_token_count > target_limits.max_context_tokens and rules_truncated > 0:
            overflow = target_token_count - target_limits.max_context_tokens
            errors.append(
                f"Rules exceed {target_tool.value} token limit by {overflow:,} tokens "
                f"({target_token_count:,} > {target_limits.max_context_tokens:,})"
            )
            errors.append(
                f"Enable auto_truncate to automatically select top {rules_truncated} "
                "rules that fit within limit"
            )

        return errors
