"""
Intelligent rule prioritization system for context injection.

This module provides sophisticated rule ranking and selection algorithms that
go beyond simple priority sorting. Supports usage-based learning, cost-benefit
optimization, and dynamic priority adjustment.

Key Features:
- Multiple prioritization strategies (importance, frequency, cost-benefit, hybrid)
- Usage tracking integration for learning from patterns
- Cost-benefit optimization to maximize value per token
- Dynamic priority adjustment based on actual usage
- Configurable weighting for strategy tuning
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger

from ..memory import AuthorityLevel, MemoryRule
from .token_budget import TokenCounter
from .token_usage_tracker import OperationType, TokenUsageTracker, ToolUsageStats


class PrioritizationStrategy(Enum):
    """Rule prioritization strategies."""

    IMPORTANCE = "importance"  # Based on authority + metadata priority
    FREQUENCY = "frequency"  # Based on usage frequency
    COST_BENEFIT = "cost_benefit"  # Value per token ratio
    HYBRID = "hybrid"  # Combined weighted strategy
    RECENCY = "recency"  # Recently created/updated first
    ADAPTIVE = "adaptive"  # Learn from usage patterns over time


@dataclass
class RulePriorityScore:
    """
    Priority score for a memory rule.

    Attributes:
        rule_id: Unique identifier of the rule
        rule: The memory rule being scored
        total_score: Final priority score (higher = more important)
        component_scores: Breakdown of score components
        token_cost: Estimated token cost for this rule
        value_per_token: Value-to-token ratio
        usage_count: Number of times rule was used
        last_used: Last time rule was used
        metadata: Additional scoring metadata
    """

    rule_id: str
    rule: MemoryRule
    total_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    token_cost: int = 0
    value_per_token: float = 0.0
    usage_count: int = 0
    last_used: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other: "RulePriorityScore") -> bool:
        """Compare scores for sorting (higher score = higher priority)."""
        return self.total_score < other.total_score


@dataclass
class PrioritizationResult:
    """
    Result of rule prioritization.

    Attributes:
        strategy: Strategy used for prioritization
        ranked_rules: Rules ordered by priority (highest first)
        priority_scores: Detailed scores for each rule
        total_rules: Total number of rules ranked
        total_value: Sum of all priority scores
        statistics: Prioritization statistics
    """

    strategy: PrioritizationStrategy
    ranked_rules: List[MemoryRule]
    priority_scores: List[RulePriorityScore]
    total_rules: int
    total_value: float
    statistics: Dict[str, Any] = field(default_factory=dict)


class RulePrioritizer:
    """
    Intelligent rule prioritization with multiple ranking strategies.

    Provides sophisticated rule ranking beyond simple priority sorting,
    including usage-based learning, cost-benefit optimization, and
    dynamic priority adjustment.

    Usage:
        prioritizer = RulePrioritizer(
            usage_tracker=tracker,
            strategy=PrioritizationStrategy.HYBRID,
            importance_weight=0.4,
            frequency_weight=0.3,
            recency_weight=0.3
        )

        result = prioritizer.prioritize_rules(rules, tool_name="claude")
        top_rules = result.ranked_rules[:10]  # Top 10 rules
    """

    def __init__(
        self,
        usage_tracker: Optional[TokenUsageTracker] = None,
        strategy: PrioritizationStrategy = PrioritizationStrategy.IMPORTANCE,
        importance_weight: float = 0.5,
        frequency_weight: float = 0.3,
        recency_weight: float = 0.2,
        cost_benefit_threshold: float = 0.0,
        adaptive_learning_rate: float = 0.1,
        use_accurate_counting: bool = True,
    ):
        """
        Initialize rule prioritizer.

        Args:
            usage_tracker: Optional TokenUsageTracker for frequency-based prioritization
            strategy: Default prioritization strategy
            importance_weight: Weight for importance component (0.0-1.0)
            frequency_weight: Weight for frequency component (0.0-1.0)
            recency_weight: Weight for recency component (0.0-1.0)
            cost_benefit_threshold: Minimum value-per-token ratio to consider
            adaptive_learning_rate: Rate of priority adjustment from usage (0.0-1.0)
            use_accurate_counting: If True, use actual tokenizer for counting
        """
        self.usage_tracker = usage_tracker
        self.strategy = strategy
        self.importance_weight = importance_weight
        self.frequency_weight = frequency_weight
        self.recency_weight = recency_weight
        self.cost_benefit_threshold = cost_benefit_threshold
        self.adaptive_learning_rate = adaptive_learning_rate
        self.use_accurate_counting = use_accurate_counting

        # Dynamic priority adjustments (learned from usage)
        self._priority_adjustments: Dict[str, float] = defaultdict(float)

        # Normalize weights to sum to 1.0
        total_weight = importance_weight + frequency_weight + recency_weight
        if total_weight > 0:
            self.importance_weight /= total_weight
            self.frequency_weight /= total_weight
            self.recency_weight /= total_weight

        logger.debug(
            f"Initialized RulePrioritizer (strategy: {strategy.value}, "
            f"weights: I={self.importance_weight:.2f}, "
            f"F={self.frequency_weight:.2f}, "
            f"R={self.recency_weight:.2f})"
        )

    def prioritize_rules(
        self,
        rules: List[MemoryRule],
        tool_name: str,
        strategy: Optional[PrioritizationStrategy] = None,
    ) -> PrioritizationResult:
        """
        Prioritize rules using specified strategy.

        Args:
            rules: Rules to prioritize
            tool_name: Target tool name for token counting
            strategy: Override default strategy (optional)

        Returns:
            PrioritizationResult with ranked rules and scores
        """
        strategy = strategy or self.strategy

        # Compute priority scores
        scores = self._compute_priority_scores(rules, tool_name, strategy)

        # Sort by total score (highest first)
        scores.sort(reverse=True)
        ranked_rules = [score.rule for score in scores]

        # Calculate statistics
        total_value = sum(score.total_score for score in scores)
        statistics = self._compute_statistics(scores, strategy)

        result = PrioritizationResult(
            strategy=strategy,
            ranked_rules=ranked_rules,
            priority_scores=scores,
            total_rules=len(rules),
            total_value=total_value,
            statistics=statistics,
        )

        logger.debug(
            f"Prioritized {len(rules)} rules using {strategy.value} "
            f"(total_value: {total_value:.2f})"
        )

        return result

    def select_top_rules(
        self,
        rules: List[MemoryRule],
        tool_name: str,
        budget: int,
        strategy: Optional[PrioritizationStrategy] = None,
        protect_absolute: bool = True,
    ) -> Tuple[List[MemoryRule], List[MemoryRule]]:
        """
        Select top rules that fit within token budget.

        Prioritizes rules and selects highest-value rules up to budget limit.

        Args:
            rules: Rules to select from
            tool_name: Target tool name
            budget: Available token budget
            strategy: Prioritization strategy (optional)
            protect_absolute: If True, always include absolute authority rules

        Returns:
            Tuple of (selected_rules, skipped_rules)
        """
        # Separate absolute vs default rules
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        selected = []
        skipped = []
        used_tokens = 0

        # Include absolute rules if protected
        if protect_absolute:
            for rule in absolute_rules:
                rule_tokens = self._estimate_tokens(rule, tool_name)
                selected.append(rule)
                used_tokens += rule_tokens

        # Prioritize and select default rules
        if default_rules:
            remaining_budget = budget - used_tokens
            prioritization = self.prioritize_rules(default_rules, tool_name, strategy)

            for score in prioritization.priority_scores:
                if used_tokens + score.token_cost <= budget:
                    selected.append(score.rule)
                    used_tokens += score.token_cost
                else:
                    skipped.append(score.rule)

        # Add any unselected absolute rules to selected (even if over budget)
        if not protect_absolute:
            skipped.extend(absolute_rules)

        logger.debug(
            f"Selected {len(selected)}/{len(rules)} rules "
            f"({used_tokens}/{budget} tokens)"
        )

        return selected, skipped

    def adjust_priority_from_usage(
        self,
        rule_id: str,
        usage_value: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Adjust rule priority based on usage feedback.

        Implements adaptive learning by adjusting priorities based on
        actual usage patterns.

        Args:
            rule_id: Rule to adjust
            usage_value: Usage value signal (-1.0 to 1.0, higher = more valuable)
            learning_rate: Override default learning rate
        """
        learning_rate = learning_rate or self.adaptive_learning_rate
        current_adjustment = self._priority_adjustments[rule_id]
        new_adjustment = current_adjustment + (learning_rate * usage_value)

        # Clamp adjustment to reasonable bounds (-1.0 to 1.0)
        new_adjustment = max(-1.0, min(1.0, new_adjustment))

        self._priority_adjustments[rule_id] = new_adjustment

        logger.debug(
            f"Adjusted priority for rule {rule_id}: "
            f"{current_adjustment:.3f} -> {new_adjustment:.3f} "
            f"(usage_value: {usage_value:.3f})"
        )

    def get_priority_adjustment(self, rule_id: str) -> float:
        """
        Get current priority adjustment for a rule.

        Args:
            rule_id: Rule ID

        Returns:
            Priority adjustment value (-1.0 to 1.0)
        """
        return self._priority_adjustments.get(rule_id, 0.0)

    def reset_priority_adjustments(self) -> None:
        """Reset all learned priority adjustments."""
        self._priority_adjustments.clear()
        logger.info("Reset all priority adjustments")

    def _compute_priority_scores(
        self,
        rules: List[MemoryRule],
        tool_name: str,
        strategy: PrioritizationStrategy,
    ) -> List[RulePriorityScore]:
        """
        Compute priority scores for all rules.

        Args:
            rules: Rules to score
            tool_name: Target tool name
            strategy: Prioritization strategy

        Returns:
            List of priority scores
        """
        scores = []

        for rule in rules:
            if strategy == PrioritizationStrategy.IMPORTANCE:
                score = self._score_by_importance(rule, tool_name)
            elif strategy == PrioritizationStrategy.FREQUENCY:
                score = self._score_by_frequency(rule, tool_name)
            elif strategy == PrioritizationStrategy.COST_BENEFIT:
                score = self._score_by_cost_benefit(rule, tool_name)
            elif strategy == PrioritizationStrategy.RECENCY:
                score = self._score_by_recency(rule, tool_name)
            elif strategy == PrioritizationStrategy.ADAPTIVE:
                score = self._score_by_adaptive(rule, tool_name)
            else:  # HYBRID
                score = self._score_by_hybrid(rule, tool_name)

            scores.append(score)

        return scores

    def _score_by_importance(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule by importance (authority + priority).

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Base score from authority (absolute=1.0, default=0.5)
        authority_score = 1.0 if rule.authority == AuthorityLevel.ABSOLUTE else 0.5

        # Priority from metadata (normalized to 0-1 range, assuming 0-100 scale)
        metadata_priority = getattr(rule.metadata, "priority", 50) if rule.metadata else 50
        priority_score = metadata_priority / 100.0

        # Combined importance
        total_score = (authority_score * 0.7) + (priority_score * 0.3)

        # Apply learned adjustment
        adjustment = self._priority_adjustments.get(rule.id, 0.0)
        total_score += adjustment

        token_cost = self._estimate_tokens(rule, tool_name)

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores={
                "authority": authority_score,
                "priority": priority_score,
                "adjustment": adjustment,
            },
            token_cost=token_cost,
        )

    def _score_by_frequency(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule by usage frequency.

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Get usage statistics from tracker
        usage_count = 0
        last_used = None

        if self.usage_tracker:
            # Check if rule ID appears in operation metadata
            tool_stats = self.usage_tracker.get_tool_stats(tool_name)
            if tool_stats:
                for op in tool_stats.operations:
                    if op.metadata.get("rule_id") == rule.id:
                        usage_count += 1
                        if last_used is None or op.timestamp > last_used:
                            last_used = op.timestamp

        # Normalize frequency (logarithmic scale to prevent extreme values)
        import math

        frequency_score = math.log1p(usage_count) / 10.0  # Soft cap at ~2.3

        # Apply learned adjustment
        adjustment = self._priority_adjustments.get(rule.id, 0.0)
        total_score = frequency_score + adjustment

        token_cost = self._estimate_tokens(rule, tool_name)

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores={
                "frequency": frequency_score,
                "adjustment": adjustment,
            },
            token_cost=token_cost,
            usage_count=usage_count,
            last_used=last_used,
        )

    def _score_by_cost_benefit(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule by cost-benefit ratio (value per token).

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Calculate base value (importance + usage)
        importance_score = self._score_by_importance(rule, tool_name)
        frequency_score = self._score_by_frequency(rule, tool_name)

        base_value = (
            importance_score.component_scores.get("authority", 0.5) * 0.6
            + importance_score.component_scores.get("priority", 0.5) * 0.2
            + frequency_score.component_scores.get("frequency", 0.0) * 0.2
        )

        token_cost = self._estimate_tokens(rule, tool_name)

        # Value per token (prevent division by zero)
        value_per_token = base_value / max(1, token_cost)

        # Total score is value per token (higher = better efficiency)
        total_score = value_per_token

        # Apply learned adjustment
        adjustment = self._priority_adjustments.get(rule.id, 0.0)
        total_score += adjustment

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores={
                "base_value": base_value,
                "cost_benefit": value_per_token,
                "adjustment": adjustment,
            },
            token_cost=token_cost,
            value_per_token=value_per_token,
            usage_count=frequency_score.usage_count,
            last_used=frequency_score.last_used,
        )

    def _score_by_recency(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule by recency (recently updated first).

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Calculate age in days
        now = datetime.now(timezone.utc)
        updated_at = rule.updated_at or rule.created_at
        age_days = (now - updated_at).total_seconds() / 86400.0

        # Recency score (exponential decay, half-life ~30 days)
        import math

        recency_score = math.exp(-age_days / 30.0)

        # Apply learned adjustment
        adjustment = self._priority_adjustments.get(rule.id, 0.0)
        total_score = recency_score + adjustment

        token_cost = self._estimate_tokens(rule, tool_name)

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores={
                "recency": recency_score,
                "age_days": age_days,
                "adjustment": adjustment,
            },
            token_cost=token_cost,
        )

    def _score_by_adaptive(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule using adaptive learning from usage patterns.

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Start with hybrid base score
        hybrid_score = self._score_by_hybrid(rule, tool_name)

        # Heavy weight on learned adjustments
        adjustment = self._priority_adjustments.get(rule.id, 0.0)

        # Adaptive total: base hybrid + 2x adjustment weight
        total_score = hybrid_score.total_score + (adjustment * 2.0)

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores={
                **hybrid_score.component_scores,
                "adaptive_adjustment": adjustment * 2.0,
            },
            token_cost=hybrid_score.token_cost,
            value_per_token=hybrid_score.value_per_token,
            usage_count=hybrid_score.usage_count,
            last_used=hybrid_score.last_used,
        )

    def _score_by_hybrid(
        self, rule: MemoryRule, tool_name: str
    ) -> RulePriorityScore:
        """
        Score rule using weighted combination of strategies.

        Args:
            rule: Rule to score
            tool_name: Target tool name

        Returns:
            Priority score
        """
        # Compute component scores
        importance_score = self._score_by_importance(rule, tool_name)
        frequency_score = self._score_by_frequency(rule, tool_name)
        recency_score = self._score_by_recency(rule, tool_name)

        # Weighted combination
        total_score = (
            importance_score.total_score * self.importance_weight
            + frequency_score.component_scores.get("frequency", 0.0)
            * self.frequency_weight
            + recency_score.component_scores.get("recency", 0.0) * self.recency_weight
        )

        # Combine component scores
        combined_components = {
            **importance_score.component_scores,
            **frequency_score.component_scores,
            **recency_score.component_scores,
            "weighted_importance": importance_score.total_score * self.importance_weight,
            "weighted_frequency": frequency_score.component_scores.get("frequency", 0.0)
            * self.frequency_weight,
            "weighted_recency": recency_score.component_scores.get("recency", 0.0)
            * self.recency_weight,
        }

        token_cost = self._estimate_tokens(rule, tool_name)
        value_per_token = total_score / max(1, token_cost)

        return RulePriorityScore(
            rule_id=rule.id,
            rule=rule,
            total_score=total_score,
            component_scores=combined_components,
            token_cost=token_cost,
            value_per_token=value_per_token,
            usage_count=frequency_score.usage_count,
            last_used=frequency_score.last_used,
        )

    def _estimate_tokens(self, rule: MemoryRule, tool_name: str) -> int:
        """
        Estimate token count for a rule.

        Args:
            rule: Memory rule
            tool_name: Target tool name

        Returns:
            Token count
        """
        # Build complete rule text
        text = rule.rule

        if rule.scope:
            text += " " + " ".join(rule.scope)

        if rule.source:
            text += f" source:{rule.source}"

        # Use tool-specific counter
        return TokenCounter.count_tokens(
            text, tool_name, use_tokenizer=self.use_accurate_counting
        )

    def _compute_statistics(
        self, scores: List[RulePriorityScore], strategy: PrioritizationStrategy
    ) -> Dict[str, Any]:
        """
        Compute prioritization statistics.

        Args:
            scores: Priority scores
            strategy: Strategy used

        Returns:
            Statistics dictionary
        """
        if not scores:
            return {
                "strategy": strategy.value,
                "total_rules": 0,
            }

        total_scores = [s.total_score for s in scores]
        token_costs = [s.token_cost for s in scores]
        value_per_tokens = [s.value_per_token for s in scores]

        import statistics as stats

        return {
            "strategy": strategy.value,
            "total_rules": len(scores),
            "score_mean": stats.mean(total_scores),
            "score_median": stats.median(total_scores),
            "score_stdev": stats.stdev(total_scores) if len(total_scores) > 1 else 0.0,
            "score_min": min(total_scores),
            "score_max": max(total_scores),
            "token_cost_total": sum(token_costs),
            "token_cost_mean": stats.mean(token_costs),
            "value_per_token_mean": stats.mean(value_per_tokens),
            "high_priority_count": sum(1 for s in scores if s.total_score > 0.7),
            "medium_priority_count": sum(
                1 for s in scores if 0.3 <= s.total_score <= 0.7
            ),
            "low_priority_count": sum(1 for s in scores if s.total_score < 0.3),
        }
