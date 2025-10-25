"""
Interactive rule trimming UI for token budget management.

This module provides an interactive interface for managing rule selection when
budgets are exceeded. Features include:
- Visual display of rules with token costs and priorities
- Automatic trimming suggestions from RulePrioritizer
- Manual rule selection/deselection
- Real-time budget impact visualization
- Save/load trimming decisions

Key Features:
- Core logic separated from CLI presentation (testable)
- Multiple prioritization strategies
- Automatic suggestions with manual override
- Budget visualization and tracking
- Persistent trimming decisions
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger

from ..memory import AuthorityLevel, MemoryRule
from .rule_prioritizer import (
    PrioritizationResult,
    PrioritizationStrategy,
    RulePrioritizer,
    RulePriorityScore,
)


class TrimDecisionType(Enum):
    """Types of trim decisions."""

    INCLUDE = "include"  # Rule selected for inclusion
    EXCLUDE = "exclude"  # Rule excluded due to budget
    AUTOMATIC = "automatic"  # Auto-suggested by prioritizer
    MANUAL = "manual"  # Manually selected/deselected


@dataclass
class TrimDecision:
    """
    Decision record for a single rule.

    Attributes:
        rule_id: Rule identifier
        decision: Include or exclude
        decision_type: How decision was made (auto vs manual)
        reason: Human-readable reason for decision
        timestamp: When decision was made
    """

    rule_id: str
    decision: TrimDecisionType
    decision_type: str  # 'automatic' or 'manual'
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrimSession:
    """
    Complete trimming session state.

    Attributes:
        tool_name: Target tool name
        budget: Total token budget
        strategy: Prioritization strategy used
        decisions: Decisions for each rule
        total_tokens_used: Total tokens after trimming
        rules_included: Count of included rules
        rules_excluded: Count of excluded rules
        created_at: Session creation time
        metadata: Additional session metadata
    """

    tool_name: str
    budget: int
    strategy: PrioritizationStrategy
    decisions: dict[str, TrimDecision] = field(default_factory=dict)
    total_tokens_used: int = 0
    rules_included: int = 0
    rules_excluded: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetVisualization:
    """
    Budget visualization data for display.

    Attributes:
        total_budget: Target budget
        used_tokens: Current tokens used
        remaining_tokens: Remaining budget
        utilization_pct: Budget utilization percentage
        over_budget: Whether current selection is over budget
        over_budget_amount: Tokens over budget (if over)
        rules_count: Total rules being considered
        included_count: Rules currently included
        excluded_count: Rules currently excluded
        protected_count: Absolute authority rules (always included)
    """

    total_budget: int
    used_tokens: int
    remaining_tokens: int
    utilization_pct: float
    over_budget: bool
    over_budget_amount: int
    rules_count: int
    included_count: int
    excluded_count: int
    protected_count: int


@dataclass
class RuleDisplay:
    """
    Display information for a single rule.

    Attributes:
        rule: The memory rule
        score: Priority score from prioritizer
        included: Whether rule is included in current selection
        protected: Whether rule is protected (absolute authority)
        auto_suggested: Whether auto-suggested for inclusion/exclusion
        decision_type: How decision was made
        display_index: Display index in list
    """

    rule: MemoryRule
    score: RulePriorityScore
    included: bool
    protected: bool
    auto_suggested: bool
    decision_type: TrimDecisionType
    display_index: int


class InteractiveTrimmer:
    """
    Core logic for interactive rule trimming.

    Provides functionality for rule selection, budget tracking, and
    trimming decision management. Separated from CLI presentation
    for testability.

    Usage:
        trimmer = InteractiveTrimmer(
            rules=rules,
            budget=2000,
            tool_name="claude",
            prioritizer=prioritizer
        )

        # Apply automatic suggestions
        auto_result = trimmer.apply_auto_suggestions()

        # Manual adjustments
        trimmer.toggle_rule("rule-123")

        # Get current state
        viz = trimmer.get_budget_visualization()
        displays = trimmer.get_rule_displays()

        # Save session
        trimmer.save_session("my_session.json")
    """

    def __init__(
        self,
        rules: list[MemoryRule],
        budget: int,
        tool_name: str,
        prioritizer: RulePrioritizer,
        strategy: PrioritizationStrategy | None = None,
        protect_absolute: bool = True,
        auto_apply_suggestions: bool = False,
    ):
        """
        Initialize interactive trimmer.

        Args:
            rules: Rules to manage
            budget: Target token budget
            tool_name: Target tool name
            prioritizer: RulePrioritizer for ranking and suggestions
            strategy: Prioritization strategy (uses prioritizer default if None)
            protect_absolute: If True, absolute rules cannot be excluded
            auto_apply_suggestions: If True, auto-apply suggestions on init
        """
        self.rules = rules
        self.budget = budget
        self.tool_name = tool_name
        self.prioritizer = prioritizer
        self.strategy = strategy or prioritizer.strategy
        self.protect_absolute = protect_absolute

        # Prioritize all rules
        self.prioritization: PrioritizationResult = prioritizer.prioritize_rules(
            rules, tool_name, self.strategy
        )

        # Build score lookup
        self.scores_by_id: dict[str, RulePriorityScore] = {
            score.rule_id: score for score in self.prioritization.priority_scores
        }

        # Track selection state (rule_id -> included boolean)
        self.selection: dict[str, bool] = {}

        # Track decision types (rule_id -> TrimDecision)
        self.decisions: dict[str, TrimDecision] = {}

        # Identify protected rules
        self.protected_rules: set[str] = (
            {r.id for r in rules if r.authority == AuthorityLevel.ABSOLUTE}
            if protect_absolute
            else set()
        )

        # Initialize with no selection or auto-apply
        if auto_apply_suggestions:
            self.apply_auto_suggestions()
        else:
            # Start with all rules included
            for rule in rules:
                self.selection[rule.id] = True
                self.decisions[rule.id] = TrimDecision(
                    rule_id=rule.id,
                    decision=TrimDecisionType.INCLUDE,
                    decision_type="automatic",
                    reason="Initial state",
                )

        logger.debug(
            f"Initialized InteractiveTrimmer: {len(rules)} rules, "
            f"budget={budget}, strategy={self.strategy.value}"
        )

    def apply_auto_suggestions(self) -> dict[str, Any]:
        """
        Apply automatic trimming suggestions from prioritizer.

        Uses RulePrioritizer.select_top_rules() to determine which rules
        should be included to fit within budget.

        Returns:
            Dictionary with suggestion results and statistics
        """
        # Get auto-suggestions from prioritizer
        selected_rules, skipped_rules = self.prioritizer.select_top_rules(
            rules=self.rules,
            tool_name=self.tool_name,
            budget=self.budget,
            strategy=self.strategy,
            protect_absolute=self.protect_absolute,
        )

        # Update selection and decisions
        self.selection.clear()
        self.decisions.clear()

        for rule in selected_rules:
            self.selection[rule.id] = True
            self.decisions[rule.id] = TrimDecision(
                rule_id=rule.id,
                decision=TrimDecisionType.INCLUDE,
                decision_type="automatic",
                reason="Auto-suggested by prioritizer",
            )

        for rule in skipped_rules:
            self.selection[rule.id] = False
            self.decisions[rule.id] = TrimDecision(
                rule_id=rule.id,
                decision=TrimDecisionType.EXCLUDE,
                decision_type="automatic",
                reason="Excluded to fit budget",
            )

        # Calculate statistics
        used_tokens = sum(
            self.scores_by_id[rule.id].token_cost
            for rule in selected_rules
            if rule.id in self.scores_by_id
        )

        result = {
            "rules_included": len(selected_rules),
            "rules_excluded": len(skipped_rules),
            "tokens_used": used_tokens,
            "budget": self.budget,
            "within_budget": used_tokens <= self.budget,
            "utilization_pct": (used_tokens / self.budget * 100)
            if self.budget > 0
            else 0,
        }

        logger.info(
            f"Applied auto-suggestions: {result['rules_included']} included, "
            f"{result['rules_excluded']} excluded, {used_tokens}/{self.budget} tokens"
        )

        return result

    def toggle_rule(self, rule_id: str) -> bool:
        """
        Toggle a rule's inclusion state.

        Args:
            rule_id: Rule to toggle

        Returns:
            New inclusion state (True = included, False = excluded)

        Raises:
            ValueError: If rule is protected and cannot be excluded
        """
        if rule_id in self.protected_rules and self.selection.get(rule_id, True):
            raise ValueError(
                f"Cannot exclude protected rule: {rule_id} "
                f"(absolute authority rules are always included)"
            )

        # Toggle state
        current = self.selection.get(rule_id, True)
        new_state = not current

        self.selection[rule_id] = new_state

        # Update decision
        self.decisions[rule_id] = TrimDecision(
            rule_id=rule_id,
            decision=TrimDecisionType.INCLUDE if new_state else TrimDecisionType.EXCLUDE,
            decision_type="manual",
            reason="Manually toggled by user",
        )

        logger.debug(
            f"Toggled rule {rule_id}: {'included' if new_state else 'excluded'}"
        )

        return new_state

    def include_rule(self, rule_id: str) -> None:
        """
        Explicitly include a rule.

        Args:
            rule_id: Rule to include
        """
        if not self.selection.get(rule_id, False):
            self.toggle_rule(rule_id)

    def exclude_rule(self, rule_id: str) -> None:
        """
        Explicitly exclude a rule.

        Args:
            rule_id: Rule to exclude

        Raises:
            ValueError: If rule is protected
        """
        if self.selection.get(rule_id, True):
            self.toggle_rule(rule_id)

    def reset_to_auto_suggestions(self) -> dict[str, Any]:
        """
        Reset all selections to automatic suggestions.

        Returns:
            Dictionary with reset results
        """
        logger.info("Resetting to auto-suggestions")
        return self.apply_auto_suggestions()

    def include_all(self) -> int:
        """
        Include all rules.

        Returns:
            Number of rules included
        """
        for rule in self.rules:
            if not self.selection.get(rule.id, False):
                self.selection[rule.id] = True
                self.decisions[rule.id] = TrimDecision(
                    rule_id=rule.id,
                    decision=TrimDecisionType.INCLUDE,
                    decision_type="manual",
                    reason="Included via include_all",
                )

        logger.info("Included all rules")
        return len(self.rules)

    def exclude_all_non_protected(self) -> int:
        """
        Exclude all non-protected rules.

        Returns:
            Number of rules excluded
        """
        excluded_count = 0
        for rule in self.rules:
            if rule.id not in self.protected_rules:
                if self.selection.get(rule.id, True):
                    self.selection[rule.id] = False
                    self.decisions[rule.id] = TrimDecision(
                        rule_id=rule.id,
                        decision=TrimDecisionType.EXCLUDE,
                        decision_type="manual",
                        reason="Excluded via exclude_all_non_protected",
                    )
                    excluded_count += 1

        logger.info(f"Excluded {excluded_count} non-protected rules")
        return excluded_count

    def get_budget_visualization(self) -> BudgetVisualization:
        """
        Get current budget visualization data.

        Returns:
            BudgetVisualization with current state
        """
        # Calculate tokens used by currently selected rules
        used_tokens = 0
        included_count = 0
        excluded_count = 0

        for rule in self.rules:
            if self.selection.get(rule.id, True):
                included_count += 1
                if rule.id in self.scores_by_id:
                    used_tokens += self.scores_by_id[rule.id].token_cost
            else:
                excluded_count += 1

        remaining = self.budget - used_tokens
        over_budget = used_tokens > self.budget
        over_budget_amount = max(0, used_tokens - self.budget)
        utilization = (used_tokens / self.budget * 100) if self.budget > 0 else 0

        return BudgetVisualization(
            total_budget=self.budget,
            used_tokens=used_tokens,
            remaining_tokens=remaining,
            utilization_pct=utilization,
            over_budget=over_budget,
            over_budget_amount=over_budget_amount,
            rules_count=len(self.rules),
            included_count=included_count,
            excluded_count=excluded_count,
            protected_count=len(self.protected_rules),
        )

    def get_rule_displays(
        self, sort_by: str = "priority"
    ) -> list[RuleDisplay]:
        """
        Get rule display information for UI rendering.

        Args:
            sort_by: Sort order - 'priority' (default), 'name', 'tokens', 'category'

        Returns:
            List of RuleDisplay objects with rendering information
        """
        displays = []

        for idx, score in enumerate(self.prioritization.priority_scores):
            rule = score.rule
            included = self.selection.get(rule.id, True)
            protected = rule.id in self.protected_rules
            decision = self.decisions.get(rule.id)
            auto_suggested = decision and decision.decision_type == "automatic"

            # Determine decision type for display
            if protected:
                decision_type = TrimDecisionType.INCLUDE  # Always included
            elif decision:
                decision_type = decision.decision
            else:
                decision_type = (
                    TrimDecisionType.INCLUDE if included else TrimDecisionType.EXCLUDE
                )

            displays.append(
                RuleDisplay(
                    rule=rule,
                    score=score,
                    included=included,
                    protected=protected,
                    auto_suggested=auto_suggested,
                    decision_type=decision_type,
                    display_index=idx + 1,
                )
            )

        # Sort displays
        if sort_by == "name":
            displays.sort(key=lambda d: d.rule.name.lower())
        elif sort_by == "tokens":
            displays.sort(key=lambda d: d.score.token_cost, reverse=True)
        elif sort_by == "category":
            displays.sort(key=lambda d: d.rule.category.value)
        # else: keep priority order (already sorted by prioritizer)

        # Update display indices after sorting
        for idx, display in enumerate(displays):
            display.display_index = idx + 1

        return displays

    def get_comparison(self) -> dict[str, Any]:
        """
        Get before/after comparison of rule sets.

        Returns:
            Dictionary with comparison data
        """
        # Auto-suggestions (before manual changes)
        selected_auto, skipped_auto = self.prioritizer.select_top_rules(
            rules=self.rules,
            tool_name=self.tool_name,
            budget=self.budget,
            strategy=self.strategy,
            protect_absolute=self.protect_absolute,
        )

        auto_tokens = sum(
            self.scores_by_id[r.id].token_cost
            for r in selected_auto
            if r.id in self.scores_by_id
        )

        # Current selection (after manual changes)
        current_included = [r for r in self.rules if self.selection.get(r.id, True)]
        current_excluded = [r for r in self.rules if not self.selection.get(r.id, True)]

        current_tokens = sum(
            self.scores_by_id[r.id].token_cost
            for r in current_included
            if r.id in self.scores_by_id
        )

        # Count manual changes
        manual_changes = sum(
            1
            for r in self.rules
            if self.decisions.get(r.id)
            and self.decisions[r.id].decision_type == "manual"
        )

        return {
            "auto_suggestions": {
                "included_count": len(selected_auto),
                "excluded_count": len(skipped_auto),
                "tokens_used": auto_tokens,
                "within_budget": auto_tokens <= self.budget,
            },
            "current_selection": {
                "included_count": len(current_included),
                "excluded_count": len(current_excluded),
                "tokens_used": current_tokens,
                "within_budget": current_tokens <= self.budget,
            },
            "manual_changes": manual_changes,
            "budget": self.budget,
        }

    def save_session(self, filepath: Path) -> None:
        """
        Save current trimming session to file.

        Args:
            filepath: Path to save session file
        """
        viz = self.get_budget_visualization()

        session = TrimSession(
            tool_name=self.tool_name,
            budget=self.budget,
            strategy=self.strategy,
            decisions=self.decisions,
            total_tokens_used=viz.used_tokens,
            rules_included=viz.included_count,
            rules_excluded=viz.excluded_count,
            metadata={
                "over_budget": viz.over_budget,
                "utilization_pct": viz.utilization_pct,
                "protected_count": viz.protected_count,
            },
        )

        # Convert to JSON-serializable format
        session_dict = {
            "tool_name": session.tool_name,
            "budget": session.budget,
            "strategy": session.strategy.value,
            "decisions": {
                rule_id: {
                    "rule_id": decision.rule_id,
                    "decision": decision.decision.value,
                    "decision_type": decision.decision_type,
                    "reason": decision.reason,
                    "timestamp": decision.timestamp.isoformat(),
                }
                for rule_id, decision in session.decisions.items()
            },
            "total_tokens_used": session.total_tokens_used,
            "rules_included": session.rules_included,
            "rules_excluded": session.rules_excluded,
            "created_at": session.created_at.isoformat(),
            "metadata": session.metadata,
        }

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(session_dict, f, indent=2)

        logger.info(f"Saved trimming session to {filepath}")

    def load_session(self, filepath: Path) -> None:
        """
        Load trimming session from file.

        Args:
            filepath: Path to session file

        Raises:
            FileNotFoundError: If session file doesn't exist
            ValueError: If session is incompatible with current rules
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Session file not found: {filepath}")

        with open(filepath) as f:
            session_dict = json.load(f)

        # Validate compatibility
        if session_dict["budget"] != self.budget:
            logger.warning(
                f"Session budget ({session_dict['budget']}) differs from "
                f"current budget ({self.budget})"
            )

        # Load decisions
        self.decisions.clear()
        self.selection.clear()

        for rule_id, decision_dict in session_dict["decisions"].items():
            # Only load decisions for rules we currently have
            if any(r.id == rule_id for r in self.rules):
                decision = TrimDecision(
                    rule_id=decision_dict["rule_id"],
                    decision=TrimDecisionType(decision_dict["decision"]),
                    decision_type=decision_dict["decision_type"],
                    reason=decision_dict["reason"],
                    timestamp=datetime.fromisoformat(decision_dict["timestamp"]),
                )
                self.decisions[rule_id] = decision
                self.selection[rule_id] = (
                    decision.decision == TrimDecisionType.INCLUDE
                )

        logger.info(f"Loaded trimming session from {filepath}")

    def get_selected_rules(self) -> list[MemoryRule]:
        """
        Get currently selected rules.

        Returns:
            List of included rules
        """
        return [r for r in self.rules if self.selection.get(r.id, True)]

    def get_excluded_rules(self) -> list[MemoryRule]:
        """
        Get currently excluded rules.

        Returns:
            List of excluded rules
        """
        return [r for r in self.rules if not self.selection.get(r.id, True)]
