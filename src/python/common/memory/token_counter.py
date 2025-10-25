"""
Token counting and optimization for memory rules.

This module provides token counting utilities for memory rules to help manage
Claude Code context window usage and optimize memory rule injection.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import AuthorityLevel, MemoryCategory, MemoryRule

logger = logging.getLogger(__name__)


class TokenizationMethod(Enum):
    """Methods for estimating token counts."""

    SIMPLE = "simple"  # Simple character-based estimation
    TIKTOKEN = "tiktoken"  # OpenAI's tiktoken library (more accurate)
    ANTHROPIC = "anthropic"  # Anthropic's tokenizer (most accurate for Claude)


@dataclass
class TokenUsage:
    """Token usage information for memory rules."""

    total_tokens: int
    rules_count: int

    # Breakdown by category
    preference_tokens: int = 0
    behavior_tokens: int = 0
    agent_library_tokens: int = 0
    knowledge_tokens: int = 0
    context_tokens: int = 0

    # Breakdown by authority
    absolute_tokens: int = 0
    default_tokens: int = 0

    # Context window information
    context_window_size: int = 200000  # Default Claude context window
    percentage: float = 0.0  # Percentage of context window used
    remaining_tokens: int = 0  # Tokens remaining in context window

    def __post_init__(self):
        """Calculate derived fields."""
        self.percentage = (self.total_tokens / self.context_window_size) * 100
        self.remaining_tokens = self.context_window_size - self.total_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_tokens": self.total_tokens,
            "rules_count": self.rules_count,
            "categories": {
                "preference": self.preference_tokens,
                "behavior": self.behavior_tokens,
                "agent_library": self.agent_library_tokens,
                "knowledge": self.knowledge_tokens,
                "context": self.context_tokens,
            },
            "authorities": {
                "absolute": self.absolute_tokens,
                "default": self.default_tokens,
            },
            "context_window": {
                "size": self.context_window_size,
                "percentage": self.percentage,
                "remaining": self.remaining_tokens,
            },
        }


@dataclass
class RuleTokenInfo:
    """Token information for a single rule."""

    rule: MemoryRule
    tokens: int
    priority_score: float  # For optimization ranking

    def __lt__(self, other):
        """Enable sorting by priority score."""
        return self.priority_score < other.priority_score


class TokenCounter:
    """
    Counts and manages tokens for memory rules.

    Provides accurate token counting for memory rules to help optimize
    Claude Code context window usage and determine which rules to inject.
    """

    def __init__(
        self,
        method: TokenizationMethod = TokenizationMethod.SIMPLE,
        context_window_size: int = 200000,
    ):
        """
        Initialize token counter.

        Args:
            method: Tokenization method to use
            context_window_size: Size of Claude's context window
        """
        self.method = method
        self.context_window_size = context_window_size
        self.tokenizer = None

        # Initialize tokenizer based on method
        if method == TokenizationMethod.TIKTOKEN:
            try:
                import tiktoken

                self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
                logger.info("Initialized tiktoken tokenizer")
            except ImportError:
                logger.warning(
                    "tiktoken not available, falling back to simple estimation"
                )
                self.method = TokenizationMethod.SIMPLE

        elif method == TokenizationMethod.ANTHROPIC:
            try:
                import anthropic

                # Note: Anthropic doesn't provide a public tokenizer yet
                # Fall back to tiktoken or simple estimation
                logger.warning(
                    "Anthropic tokenizer not available, falling back to simple estimation"
                )
                self.method = TokenizationMethod.SIMPLE
            except ImportError:
                logger.warning(
                    "Anthropic library not available, falling back to simple estimation"
                )
                self.method = TokenizationMethod.SIMPLE

    def count_rule_tokens(self, rule: MemoryRule) -> int:
        """
        Count tokens for a single memory rule.

        Args:
            rule: MemoryRule to count tokens for

        Returns:
            Estimated token count
        """
        # Create formatted rule text as it would appear in context
        formatted_rule = self._format_rule_for_context(rule)
        return self._count_text_tokens(formatted_rule)

    def count_rules_tokens(self, rules: list[MemoryRule]) -> TokenUsage:
        """
        Count tokens for a list of memory rules.

        Args:
            rules: List of MemoryRule objects

        Returns:
            TokenUsage object with detailed breakdown
        """
        total_tokens = 0
        category_tokens = dict.fromkeys(MemoryCategory, 0)
        authority_tokens = dict.fromkeys(AuthorityLevel, 0)

        for rule in rules:
            rule_tokens = self.count_rule_tokens(rule)
            total_tokens += rule_tokens

            category_tokens[rule.category] += rule_tokens
            authority_tokens[rule.authority] += rule_tokens

        return TokenUsage(
            total_tokens=total_tokens,
            rules_count=len(rules),
            preference_tokens=category_tokens[MemoryCategory.PREFERENCE],
            behavior_tokens=category_tokens[MemoryCategory.BEHAVIOR],
            agent_library_tokens=category_tokens[MemoryCategory.AGENT_LIBRARY],
            knowledge_tokens=category_tokens[MemoryCategory.KNOWLEDGE],
            context_tokens=category_tokens[MemoryCategory.CONTEXT],
            absolute_tokens=authority_tokens[AuthorityLevel.ABSOLUTE],
            default_tokens=authority_tokens[AuthorityLevel.DEFAULT],
            context_window_size=self.context_window_size,
        )

    def optimize_rules_for_context(
        self, rules: list[MemoryRule], max_tokens: int, preserve_absolute: bool = True
    ) -> tuple[list[MemoryRule], TokenUsage]:
        """
        Optimize rule selection to fit within token budget.

        Args:
            rules: List of all available rules
            max_tokens: Maximum tokens to use
            preserve_absolute: Whether to always include absolute authority rules

        Returns:
            Tuple of (selected_rules, token_usage)
        """
        # Calculate token info for each rule
        rule_infos = []
        for rule in rules:
            tokens = self.count_rule_tokens(rule)
            priority_score = self._calculate_priority_score(rule)
            rule_infos.append(RuleTokenInfo(rule, tokens, priority_score))

        # Sort by priority (highest first)
        rule_infos.sort(key=lambda x: x.priority_score, reverse=True)

        selected_rules = []
        used_tokens = 0

        # First pass: Include all absolute authority rules if preserve_absolute is True
        if preserve_absolute:
            for rule_info in rule_infos:
                if (
                    rule_info.rule.authority == AuthorityLevel.ABSOLUTE
                    and used_tokens + rule_info.tokens <= max_tokens
                ):
                    selected_rules.append(rule_info.rule)
                    used_tokens += rule_info.tokens

        # Second pass: Fill remaining space with highest priority rules
        for rule_info in rule_infos:
            if rule_info.rule in selected_rules:
                continue  # Already included

            if used_tokens + rule_info.tokens <= max_tokens:
                selected_rules.append(rule_info.rule)
                used_tokens += rule_info.tokens
            else:
                break  # Can't fit any more rules

        # Calculate final usage
        usage = self.count_rules_tokens(selected_rules)

        return selected_rules, usage

    def suggest_memory_optimizations(
        self, rules: list[MemoryRule], target_tokens: int
    ) -> dict[str, Any]:
        """
        Suggest optimizations to reduce memory token usage.

        Args:
            rules: List of current rules
            target_tokens: Target token count

        Returns:
            Dictionary with optimization suggestions
        """
        current_usage = self.count_rules_tokens(rules)

        if current_usage.total_tokens <= target_tokens:
            return {
                "current_tokens": current_usage.total_tokens,
                "target_tokens": target_tokens,
                "optimization_needed": False,
                "suggestions": ["Memory usage is already within target"],
            }

        tokens_to_reduce = current_usage.total_tokens - target_tokens
        suggestions = []
        potential_savings = []

        # Analyze rules by various criteria
        rule_infos = []
        for rule in rules:
            tokens = self.count_rule_tokens(rule)
            priority_score = self._calculate_priority_score(rule)
            rule_infos.append(RuleTokenInfo(rule, tokens, priority_score))

        # Find rules that could be removed (lowest priority first)
        rule_infos.sort(key=lambda x: x.priority_score)

        savings_from_removal = 0
        removable_rules = []

        for rule_info in rule_infos:
            if (
                rule_info.rule.authority == AuthorityLevel.DEFAULT
            ):  # Don't suggest removing absolute rules
                removable_rules.append(rule_info)
                savings_from_removal += rule_info.tokens
                if savings_from_removal >= tokens_to_reduce:
                    break

        if removable_rules:
            suggestions.append(
                f"Remove {len(removable_rules)} lowest-priority default rules "
                f"(saves ~{savings_from_removal} tokens)"
            )
            potential_savings.append(
                ("remove_low_priority", savings_from_removal, removable_rules)
            )

        # Look for rules that could be shortened
        long_rules = [r for r in rule_infos if r.tokens > 50]  # Arbitrarily long rules
        if long_rules:
            potential_shortening_savings = sum(
                max(0, r.tokens - 30) for r in long_rules
            )
            suggestions.append(
                f"Shorten {len(long_rules)} verbose rules "
                f"(potential savings: ~{potential_shortening_savings} tokens)"
            )
            potential_savings.append(
                ("shorten_rules", potential_shortening_savings, long_rules)
            )

        # Look for duplicate or similar rules
        duplicate_groups = self._find_similar_rules(rules)
        if duplicate_groups:
            duplicate_savings = sum(
                sum(
                    self.count_rule_tokens(rule) for rule in group[1:]
                )  # Keep first rule
                for group in duplicate_groups
            )
            suggestions.append(
                f"Merge {len(duplicate_groups)} groups of similar rules "
                f"(saves ~{duplicate_savings} tokens)"
            )
            potential_savings.append(
                ("merge_similar", duplicate_savings, duplicate_groups)
            )

        return {
            "current_tokens": current_usage.total_tokens,
            "target_tokens": target_tokens,
            "tokens_to_reduce": tokens_to_reduce,
            "optimization_needed": True,
            "suggestions": suggestions,
            "potential_savings": potential_savings,
            "detailed_usage": current_usage.to_dict(),
        }

    def _count_text_tokens(self, text: str) -> int:
        """
        Count tokens in text using the configured method.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        if self.method == TokenizationMethod.TIKTOKEN and self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Simple estimation: ~4 characters per token on average
            # This is a rough approximation but works for optimization
            return len(text) // 4

    def _format_rule_for_context(self, rule: MemoryRule) -> str:
        """
        Format a rule as it would appear in Claude Code context.

        Args:
            rule: MemoryRule to format

        Returns:
            Formatted rule text
        """
        # Simulate how rule might be injected into context
        authority_prefix = (
            "[ABSOLUTE]" if rule.authority == AuthorityLevel.ABSOLUTE else "[DEFAULT]"
        )
        scope_suffix = f" (Scope: {', '.join(rule.scope)})" if rule.scope else ""

        return f"{authority_prefix} {rule.rule}{scope_suffix}"

    def _calculate_priority_score(self, rule: MemoryRule) -> float:
        """
        Calculate priority score for a rule (higher = more important).

        Args:
            rule: MemoryRule to score

        Returns:
            Priority score (0.0 to 100.0)
        """
        score = 0.0

        # Authority level (absolute rules have higher priority)
        if rule.authority == AuthorityLevel.ABSOLUTE:
            score += 50.0
        else:
            score += 20.0

        # Category importance
        category_scores = {
            MemoryCategory.BEHAVIOR: 20.0,  # Core behavioral rules
            MemoryCategory.PREFERENCE: 15.0,  # User preferences
            MemoryCategory.AGENT_LIBRARY: 10.0,  # Agent definitions
            MemoryCategory.KNOWLEDGE: 8.0,  # Knowledge facts
            MemoryCategory.CONTEXT: 5.0,  # Session context
        }
        score += category_scores.get(rule.category, 0.0)

        # Usage frequency (rules used more often are more important)
        if rule.use_count > 0:
            score += min(10.0, rule.use_count * 0.5)  # Cap at 10 points

        # Recency (recently used rules are more important)
        if rule.last_used:
            from datetime import datetime, timezone

            days_since_use = (datetime.now(timezone.utc) - rule.last_used).days
            recency_score = max(0, 5.0 - (days_since_use * 0.1))  # Decay over time
            score += recency_score

        # Scope specificity (more specific rules might be more important in context)
        if rule.scope:
            score += len(rule.scope) * 2.0  # More specific = higher priority
        else:
            score += 1.0  # Global rules get small bonus

        return score

    def _find_similar_rules(self, rules: list[MemoryRule]) -> list[list[MemoryRule]]:
        """
        Find groups of similar rules that might be duplicates.

        Args:
            rules: List of rules to analyze

        Returns:
            List of groups of similar rules
        """
        # Simple similarity detection based on keyword overlap
        # In a more sophisticated implementation, this could use semantic similarity

        similar_groups = []
        processed = set()

        for i, rule1 in enumerate(rules):
            if rule1.id in processed:
                continue

            similar_rules = [rule1]
            processed.add(rule1.id)

            # Find rules with high keyword overlap
            rule1_words = set(self._extract_keywords(rule1.rule))

            for _j, rule2 in enumerate(rules[i + 1 :], i + 1):
                if rule2.id in processed:
                    continue

                rule2_words = set(self._extract_keywords(rule2.rule))

                # Calculate Jaccard similarity
                intersection = len(rule1_words & rule2_words)
                union = len(rule1_words | rule2_words)

                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.6:  # High similarity threshold
                        similar_rules.append(rule2)
                        processed.add(rule2.id)

            if len(similar_rules) > 1:
                similar_groups.append(similar_rules)

        return similar_groups

    def _extract_keywords(self, text: str) -> list[str]:
        """
        Extract keywords from rule text.

        Args:
            text: Rule text

        Returns:
            List of keywords
        """
        # Simple keyword extraction - remove common words and extract meaningful terms

        # Convert to lowercase and extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out common stop words
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "always",
            "never",
            "should",
            "must",
            "use",
            "when",
            "where",
            "what",
            "how",
            "why",
            "this",
        }

        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
