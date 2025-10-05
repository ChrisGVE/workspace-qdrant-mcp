"""
Token budget management for memory rule context injection.

This module provides token budget allocation and optimization strategies
to ensure memory rules fit within LLM context windows.
"""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from ..memory import AuthorityLevel, MemoryRule


class AllocationStrategy(Enum):
    """Token allocation strategies."""

    PRIORITY_BASED = "priority"  # Prioritize by authority + priority
    ROUND_ROBIN = "round_robin"  # Distribute equally across categories
    EQUAL = "equal"  # Fixed allocation per rule


class CompressionStrategy(Enum):
    """Rule compression strategies."""

    NONE = "none"  # No compression
    SIMPLE = "simple"  # Remove optional fields
    INTELLIGENT = "intelligent"  # Smart summarization


@dataclass
class BudgetAllocation:
    """
    Token budget allocation result.

    Attributes:
        total_budget: Total available tokens
        absolute_tokens: Tokens used by absolute authority rules
        default_tokens: Tokens used by default authority rules
        overhead_tokens: Tokens reserved for formatting overhead
        rules_included: List of rules included in allocation
        rules_skipped: List of rules skipped due to budget constraints
        compression_applied: Whether compression was applied
        allocation_stats: Additional allocation statistics
    """

    total_budget: int
    absolute_tokens: int
    default_tokens: int
    overhead_tokens: int
    rules_included: List[MemoryRule]
    rules_skipped: List[MemoryRule]
    compression_applied: bool
    allocation_stats: Dict[str, any]


class TokenCounter:
    """Tool-specific token counting utilities."""

    @staticmethod
    def count_claude_tokens(text: str) -> int:
        """
        Estimate tokens for Claude (approximation).

        Claude uses roughly 1 token per 4 characters.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)

    @staticmethod
    def count_codex_tokens(text: str) -> int:
        """
        Estimate tokens for Codex (GPT-based approximation).

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        words = len(text.split())
        return max(1, int(words * 1.3))

    @staticmethod
    def count_gemini_tokens(text: str) -> int:
        """
        Estimate tokens for Gemini (similar to Claude).

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)

    @staticmethod
    def count_tokens(text: str, tool_name: str) -> int:
        """
        Count tokens using tool-specific counter.

        Args:
            text: Text to count tokens for
            tool_name: Target tool name ("claude", "codex", "gemini")

        Returns:
            Estimated token count
        """
        counters = {
            "claude": TokenCounter.count_claude_tokens,
            "codex": TokenCounter.count_codex_tokens,
            "gemini": TokenCounter.count_gemini_tokens,
        }
        counter = counters.get(tool_name, TokenCounter.count_claude_tokens)
        return counter(text)


class TokenBudgetManager:
    """
    Manages token budget allocation and optimization for memory rules.

    Provides intelligent budget allocation strategies to fit memory rules
    within LLM context windows while protecting high-priority rules.
    """

    def __init__(
        self,
        allocation_strategy: AllocationStrategy = AllocationStrategy.PRIORITY_BASED,
        compression_strategy: CompressionStrategy = CompressionStrategy.NONE,
        absolute_rules_protected: bool = True,
        overhead_percentage: float = 0.05,
    ):
        """
        Initialize token budget manager.

        Args:
            allocation_strategy: Strategy for allocating budget to rules
            compression_strategy: Strategy for compressing rules if needed
            absolute_rules_protected: If True, absolute rules always included
            overhead_percentage: Percentage of budget reserved for formatting overhead
        """
        self.allocation_strategy = allocation_strategy
        self.compression_strategy = compression_strategy
        self.absolute_rules_protected = absolute_rules_protected
        self.overhead_percentage = overhead_percentage

    def allocate_budget(
        self,
        rules: List[MemoryRule],
        total_budget: int,
        tool_name: str,
    ) -> BudgetAllocation:
        """
        Allocate token budget across rules.

        Algorithm:
        1. Separate absolute vs default rules
        2. Allocate tokens to absolute rules (protected)
        3. Calculate remaining budget after overhead
        4. Allocate remaining budget to default rules per strategy
        5. Apply compression if needed
        6. Return allocation result with statistics

        Args:
            rules: List of memory rules to allocate
            total_budget: Total available token budget
            tool_name: Target LLM tool name for token counting

        Returns:
            BudgetAllocation with selected rules and statistics
        """
        # Separate by authority
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Count absolute rule tokens
        absolute_tokens = sum(
            self._estimate_tokens(r, tool_name) for r in absolute_rules
        )

        # Reserve budget for overhead (headers, footers, formatting)
        overhead_tokens = int(total_budget * self.overhead_percentage)
        available_budget = total_budget - overhead_tokens

        # Check if absolute rules fit
        if absolute_tokens > available_budget:
            if self.absolute_rules_protected:
                # Include all absolute rules even if over budget
                # This ensures critical rules are never skipped
                return BudgetAllocation(
                    total_budget=total_budget,
                    absolute_tokens=absolute_tokens,
                    default_tokens=0,
                    overhead_tokens=overhead_tokens,
                    rules_included=absolute_rules,
                    rules_skipped=default_rules,
                    compression_applied=False,
                    allocation_stats={
                        "over_budget": True,
                        "absolute_count": len(absolute_rules),
                        "default_count": 0,
                        "strategy": self.allocation_strategy.value,
                    },
                )
            else:
                # Truncate absolute rules to fit budget
                absolute_rules = self._truncate_rules(
                    absolute_rules, available_budget, tool_name
                )
                absolute_tokens = sum(
                    self._estimate_tokens(r, tool_name) for r in absolute_rules
                )

        # Allocate remaining budget to default rules
        remaining_budget = available_budget - absolute_tokens
        selected_default_rules = self._allocate_to_default_rules(
            default_rules, remaining_budget, tool_name
        )

        default_tokens = sum(
            self._estimate_tokens(r, tool_name) for r in selected_default_rules
        )

        all_included = absolute_rules + selected_default_rules
        all_skipped = [r for r in default_rules if r not in selected_default_rules]

        return BudgetAllocation(
            total_budget=total_budget,
            absolute_tokens=absolute_tokens,
            default_tokens=default_tokens,
            overhead_tokens=overhead_tokens,
            rules_included=all_included,
            rules_skipped=all_skipped,
            compression_applied=False,
            allocation_stats={
                "over_budget": False,
                "absolute_count": len(absolute_rules),
                "default_count": len(selected_default_rules),
                "skipped_count": len(all_skipped),
                "strategy": self.allocation_strategy.value,
                "utilization": (absolute_tokens + default_tokens) / total_budget
                if total_budget > 0
                else 0,
            },
        )

    def _allocate_to_default_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Allocate budget to default rules based on strategy.

        Args:
            rules: Default authority rules
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules that fit within budget
        """
        if self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            return self._priority_based_allocation(rules, budget, tool_name)
        elif self.allocation_strategy == AllocationStrategy.ROUND_ROBIN:
            return self._round_robin_allocation(rules, budget, tool_name)
        else:  # EQUAL
            return self._equal_allocation(rules, budget, tool_name)

    def _priority_based_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Allocate by priority (highest priority first).

        Sorts rules by metadata priority and creation time, then
        includes rules sequentially until budget is exhausted.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        sorted_rules = sorted(
            rules,
            key=lambda r: (
                getattr(r.metadata, "priority", 50) if r.metadata else 50,
                -r.created_at.timestamp(),
            ),
            reverse=True,
        )

        selected = []
        used_tokens = 0

        for rule in sorted_rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if used_tokens + rule_tokens <= budget:
                selected.append(rule)
                used_tokens += rule_tokens
            else:
                break

        return selected

    def _round_robin_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Distribute tokens equally across categories.

        Splits budget evenly across rule categories, then applies
        priority-based allocation within each category.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        by_category = defaultdict(list)
        for rule in rules:
            by_category[rule.category].append(rule)

        budget_per_category = budget // len(by_category) if by_category else 0
        selected = []

        for category_rules in by_category.values():
            category_selected = self._priority_based_allocation(
                category_rules, budget_per_category, tool_name
            )
            selected.extend(category_selected)

        return selected

    def _equal_allocation(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Fixed token allocation per rule.

        Divides budget equally among rules, including only rules
        that fit within per-rule allocation.

        Args:
            rules: Rules to allocate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        if not rules:
            return []

        tokens_per_rule = budget // len(rules)
        selected = []

        for rule in rules:
            rule_tokens = self._estimate_tokens(rule, tool_name)
            if rule_tokens <= tokens_per_rule:
                selected.append(rule)

        return selected

    def _truncate_rules(
        self,
        rules: List[MemoryRule],
        budget: int,
        tool_name: str,
    ) -> List[MemoryRule]:
        """
        Truncate rules to fit within budget.

        Uses priority-based allocation to select highest priority
        rules that fit within budget.

        Args:
            rules: Rules to truncate
            budget: Available token budget
            tool_name: Target tool name

        Returns:
            List of selected rules
        """
        return self._priority_based_allocation(rules, budget, tool_name)

    def _estimate_tokens(self, rule: MemoryRule, tool_name: str) -> int:
        """
        Estimate token count for a rule (tool-specific).

        Includes rule text, scope, and basic metadata in estimation.

        Args:
            rule: Memory rule to estimate
            tool_name: Target tool name

        Returns:
            Estimated token count
        """
        # Build complete rule text for estimation
        text = rule.rule

        if rule.scope:
            text += " " + " ".join(rule.scope)

        if rule.source:
            text += f" source:{rule.source}"

        # Use tool-specific counter
        return TokenCounter.count_tokens(text, tool_name)
