"""
Code comment injection strategy for GitHub Copilot context injection.

This module provides utilities to inject memory rules as structured code comments
that Copilot can parse and use for context-aware suggestions. Supports multiple
programming languages, placement strategies, and conflict resolution.

Key features:
- Language-specific comment formatting (Python, JS/TS, Rust, Go, Java, etc.)
- Multiple placement strategies (file header, docstring, inline)
- Priority-based conflict resolution
- Integration with existing memory rule system
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger

from ...memory import AuthorityLevel, MemoryRule, MemoryCategory
from .copilot_detector import CopilotDetector


class PlacementStrategy(Enum):
    """Where to place injected comment rules in code files."""

    FILE_HEADER = "file_header"  # Top of file after shebang/encoding
    DOCSTRING = "docstring"  # In function/class documentation
    INLINE = "inline"  # Before relevant code blocks


class ConflictResolution(Enum):
    """Strategy for resolving conflicting rules."""

    HIGHEST_PRIORITY = "highest_priority"  # Use rule with highest priority
    MOST_SPECIFIC = "most_specific"  # Use most specific scope
    LAST_WINS = "last_wins"  # Most recently created rule wins
    MERGE_ALL = "merge_all"  # Include all rules with conflict warnings


class ConflictType(Enum):
    """Types of conflicts between rules."""

    CONTRADICTORY = "contradictory"  # Rules give opposite instructions
    OVERLAPPING = "overlapping"  # Rules overlap in scope but differ in details
    REDUNDANT = "redundant"  # Rules are essentially duplicates


@dataclass
class RuleConflict:
    """
    Detected conflict between two rules.

    Attributes:
        rule1: First conflicting rule
        rule2: Second conflicting rule
        conflict_type: Type of conflict detected
        description: Human-readable conflict description
        suggested_resolution: Optional suggestion for resolving conflict
    """

    rule1: MemoryRule
    rule2: MemoryRule
    conflict_type: ConflictType
    description: str
    suggested_resolution: Optional[str] = None


@dataclass
class InjectedComment:
    """
    Result of comment injection formatting.

    Attributes:
        content: Formatted comment content ready for injection
        language: Programming language used for formatting
        placement: Where the comment should be placed
        rules_included: Number of rules included in comment
        conflicts_detected: List of conflicts found
        metadata: Additional formatting metadata
    """

    content: str
    language: str
    placement: PlacementStrategy
    rules_included: int
    conflicts_detected: List[RuleConflict]
    metadata: Dict[str, any]


class CommentInjector:
    """
    Injects memory rules as structured code comments for Copilot.

    This class formats memory rules into language-specific code comments that
    GitHub Copilot can parse and use for context. Handles priority resolution,
    conflict detection, and multiple placement strategies.

    Example:
        >>> injector = CommentInjector()
        >>> rules = [rule1, rule2, rule3]
        >>> result = injector.format_as_comment(
        ...     rules=rules,
        ...     language="python",
        ...     placement=PlacementStrategy.FILE_HEADER
        ... )
        >>> print(result.content)
    """

    def __init__(self, detector: Optional[CopilotDetector] = None):
        """
        Initialize the comment injector.

        Args:
            detector: CopilotDetector instance (creates new if not provided)
        """
        self.detector = detector or CopilotDetector()

    def format_as_comment(
        self,
        rules: List[MemoryRule],
        language: str,
        placement: PlacementStrategy,
        priority_filter: Optional[str] = None,
        detect_conflicts: bool = True,
    ) -> InjectedComment:
        """
        Format rules as language-specific code comments.

        Args:
            rules: List of memory rules to format
            language: Programming language (e.g., "python", "javascript")
            placement: Where to place the comments
            priority_filter: Optional filter (e.g., "P1", "P2", "P3")
            detect_conflicts: Whether to detect and report conflicts

        Returns:
            InjectedComment with formatted content and metadata
        """
        # Filter by priority if specified
        if priority_filter:
            rules = self._filter_by_priority(rules, priority_filter)

        # Detect conflicts
        conflicts = []
        if detect_conflicts:
            conflicts = self.detect_conflicts(rules)
            if conflicts:
                logger.warning(f"Detected {len(conflicts)} rule conflicts")

        # Format based on placement strategy
        if placement == PlacementStrategy.FILE_HEADER:
            content = self._format_file_header(rules, language)
        elif placement == PlacementStrategy.DOCSTRING:
            content = self._format_docstring_section(rules, language)
        elif placement == PlacementStrategy.INLINE:
            content = self._format_inline_directives(rules, language)
        else:
            raise ValueError(f"Unknown placement strategy: {placement}")

        return InjectedComment(
            content=content,
            language=language,
            placement=placement,
            rules_included=len(rules),
            conflicts_detected=conflicts,
            metadata={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "priority_filter": priority_filter,
                "conflict_detection": detect_conflicts,
            },
        )

    def detect_conflicts(self, rules: List[MemoryRule]) -> List[RuleConflict]:
        """
        Detect conflicts between rules.

        Checks for:
        - Contradictory rules (opposite instructions)
        - Overlapping scope with different priorities
        - Redundant rules (duplicate content)

        Args:
            rules: List of memory rules to check

        Returns:
            List of detected conflicts
        """
        conflicts = []

        # Compare all rule pairs
        for i, rule1 in enumerate(rules):
            for rule2 in rules[i + 1 :]:
                conflict = self._check_rule_pair(rule1, rule2)
                if conflict:
                    conflicts.append(conflict)

        return conflicts

    def resolve_conflicts(
        self,
        rules: List[MemoryRule],
        strategy: ConflictResolution = ConflictResolution.HIGHEST_PRIORITY,
    ) -> List[MemoryRule]:
        """
        Resolve conflicting rules using specified strategy.

        Args:
            rules: List of rules with potential conflicts
            strategy: Resolution strategy to use

        Returns:
            List of rules with conflicts resolved
        """
        conflicts = self.detect_conflicts(rules)

        if not conflicts:
            return rules

        # Track rules to remove
        rules_to_remove: Set[str] = set()

        for conflict in conflicts:
            if strategy == ConflictResolution.HIGHEST_PRIORITY:
                # Keep rule with higher priority
                priority1 = self._get_rule_priority(conflict.rule1)
                priority2 = self._get_rule_priority(conflict.rule2)

                if priority1 < priority2:
                    rules_to_remove.add(conflict.rule2.name)
                elif priority2 < priority1:
                    rules_to_remove.add(conflict.rule1.name)
                # If equal priority, keep both

            elif strategy == ConflictResolution.MOST_SPECIFIC:
                # Keep rule with more specific scope
                if self._is_more_specific(conflict.rule1, conflict.rule2):
                    rules_to_remove.add(conflict.rule2.name)
                else:
                    rules_to_remove.add(conflict.rule1.name)

            elif strategy == ConflictResolution.LAST_WINS:
                # Keep most recently created rule
                if conflict.rule1.created_at > conflict.rule2.created_at:
                    rules_to_remove.add(conflict.rule2.name)
                else:
                    rules_to_remove.add(conflict.rule1.name)

            elif strategy == ConflictResolution.MERGE_ALL:
                # Keep all rules, no removal
                pass

        # Filter out removed rules
        return [r for r in rules if r.name not in rules_to_remove]

    def _format_file_header(self, rules: List[MemoryRule], language: str) -> str:
        """
        Format rules as file header comments.

        Args:
            rules: Memory rules to format
            language: Programming language

        Returns:
            Formatted header comment block
        """
        line_prefix, _ = self.detector.get_comment_style(language)

        # Separate by authority level
        absolute_rules = [r for r in rules if r.authority == AuthorityLevel.ABSOLUTE]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Sort by priority
        absolute_rules = self._sort_by_priority(absolute_rules)
        default_rules = self._sort_by_priority(default_rules)

        lines = []

        # Header
        separator = "=" * 60
        lines.append(f"{line_prefix} {separator}")
        lines.append(f"{line_prefix} COPILOT CODING RULES - Automatically Generated")
        lines.append(
            f"{line_prefix} Last Updated: {datetime.now(timezone.utc).isoformat()}"
        )
        lines.append(f"{line_prefix} {separator}")
        lines.append(f"{line_prefix}")

        # Absolute rules (critical)
        if absolute_rules:
            for rule in absolute_rules:
                priority = self._get_rule_priority(rule)
                lines.append(
                    f"{line_prefix} [P{priority}] {rule.name} - CRITICAL"
                )
                lines.append(f"{line_prefix} {rule.rule}")
                if rule.scope:
                    lines.append(f"{line_prefix} Scope: {', '.join(rule.scope)}")
                lines.append(f"{line_prefix}")

        # Default rules
        if default_rules:
            lines.append(f"{line_prefix} --- Default Guidelines ---")
            lines.append(f"{line_prefix}")
            for rule in default_rules:
                priority = self._get_rule_priority(rule)
                lines.append(f"{line_prefix} [P{priority}] {rule.name}")
                lines.append(f"{line_prefix} {rule.rule}")
                if rule.scope:
                    lines.append(f"{line_prefix} Scope: {', '.join(rule.scope)}")
                lines.append(f"{line_prefix}")

        # Footer
        lines.append(f"{line_prefix} {separator}")

        return "\n".join(lines)

    def _format_docstring_section(
        self, rules: List[MemoryRule], language: str
    ) -> str:
        """
        Format rules as docstring section.

        This creates a section that can be added to function/class docstrings.

        Args:
            rules: Memory rules to format
            language: Programming language

        Returns:
            Formatted docstring section
        """
        if language.lower() == "python":
            return self._format_python_docstring_section(rules)
        elif language.lower() in ["javascript", "typescript", "jsx", "tsx"]:
            return self._format_jsdoc_section(rules)
        elif language.lower() == "rust":
            return self._format_rust_doc_section(rules)
        else:
            # Fallback to inline format
            return self._format_inline_directives(rules, language)

    def _format_python_docstring_section(self, rules: List[MemoryRule]) -> str:
        """Format rules for Python docstrings."""
        lines = ["", "Coding Rules:"]

        for rule in self._sort_by_priority(rules):
            priority = self._get_rule_priority(rule)
            lines.append(f"    [P{priority}] {rule.name}: {rule.rule}")

        return "\n".join(lines)

    def _format_jsdoc_section(self, rules: List[MemoryRule]) -> str:
        """Format rules for JSDoc comments."""
        lines = []

        for rule in self._sort_by_priority(rules):
            priority = self._get_rule_priority(rule)
            lines.append(
                f" * @copilot-rule [P{priority}] {rule.name} - {rule.rule}"
            )
            if rule.scope:
                lines.append(f" * Scope: {', '.join(rule.scope)}")

        return "\n".join(lines)

    def _format_rust_doc_section(self, rules: List[MemoryRule]) -> str:
        """Format rules for Rust doc comments."""
        lines = []

        for rule in self._sort_by_priority(rules):
            priority = self._get_rule_priority(rule)
            lines.append(f"/// CODING RULE [P{priority}]: {rule.name}")
            lines.append(f"/// {rule.rule}")
            if rule.scope:
                lines.append(f"/// Scope: {', '.join(rule.scope)}")

        return "\n".join(lines)

    def _format_inline_directives(
        self, rules: List[MemoryRule], language: str
    ) -> str:
        """
        Format rules as inline comment directives.

        These are meant to be placed immediately before relevant code blocks.

        Args:
            rules: Memory rules to format
            language: Programming language

        Returns:
            Formatted inline directives
        """
        line_prefix, _ = self.detector.get_comment_style(language)
        lines = []

        for rule in self._sort_by_priority(rules):
            priority = self._get_rule_priority(rule)
            lines.append(f"{line_prefix} RULE [P{priority}]: {rule.rule}")

        return "\n".join(lines)

    def _check_rule_pair(
        self, rule1: MemoryRule, rule2: MemoryRule
    ) -> Optional[RuleConflict]:
        """
        Check if two rules conflict.

        Args:
            rule1: First rule
            rule2: Second rule

        Returns:
            RuleConflict if conflict detected, None otherwise
        """
        # Check for redundant rules (same content)
        if self._are_redundant(rule1, rule2):
            return RuleConflict(
                rule1=rule1,
                rule2=rule2,
                conflict_type=ConflictType.REDUNDANT,
                description=f"Rules '{rule1.name}' and '{rule2.name}' are redundant",
                suggested_resolution="Remove one of the duplicate rules",
            )

        # Check for contradictory rules
        if self._are_contradictory(rule1, rule2):
            return RuleConflict(
                rule1=rule1,
                rule2=rule2,
                conflict_type=ConflictType.CONTRADICTORY,
                description=f"Rules '{rule1.name}' and '{rule2.name}' give opposite instructions",
                suggested_resolution="Keep rule with higher priority or resolve manually",
            )

        # Check for overlapping scope
        if self._have_overlapping_scope(rule1, rule2):
            priority1 = self._get_rule_priority(rule1)
            priority2 = self._get_rule_priority(rule2)

            if priority1 != priority2:
                return RuleConflict(
                    rule1=rule1,
                    rule2=rule2,
                    conflict_type=ConflictType.OVERLAPPING,
                    description=f"Rules '{rule1.name}' and '{rule2.name}' overlap in scope with different priorities",
                    suggested_resolution=f"Consider aligning priorities or narrowing scope",
                )

        return None

    def _are_redundant(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """Check if two rules are essentially duplicates."""
        # Simple similarity check - could be enhanced with NLP
        return (
            rule1.rule.lower().strip() == rule2.rule.lower().strip()
            or rule1.name == rule2.name
        )

    def _are_contradictory(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """Check if two rules give opposite instructions."""
        # Look for negation patterns
        contradiction_patterns = [
            ("always", "never"),
            ("must", "must not"),
            ("should", "should not"),
            ("do", "don't"),
            ("use", "avoid"),
            ("prefer", "avoid"),
        ]

        text1 = rule1.rule.lower()
        text2 = rule2.rule.lower()

        for positive, negative in contradiction_patterns:
            if (positive in text1 and negative in text2) or (
                negative in text1 and positive in text2
            ):
                # Check if they're talking about the same thing
                if self._have_overlapping_scope(rule1, rule2):
                    return True

        return False

    def _have_overlapping_scope(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """Check if two rules have overlapping scope."""
        if not rule1.scope or not rule2.scope:
            return False

        scope1 = set(rule1.scope)
        scope2 = set(rule2.scope)

        return bool(scope1 & scope2)

    def _is_more_specific(self, rule1: MemoryRule, rule2: MemoryRule) -> bool:
        """Check if rule1 has more specific scope than rule2."""
        if not rule1.scope or not rule2.scope:
            return False

        return len(rule1.scope) > len(rule2.scope)

    def _get_rule_priority(self, rule: MemoryRule) -> int:
        """
        Get priority level for a rule.

        Returns priority as integer (1=highest, 3=lowest).
        Uses metadata if available, otherwise infers from authority level.

        Args:
            rule: Memory rule

        Returns:
            Priority level (1-3)
        """
        # Check metadata first
        if rule.metadata and "priority" in rule.metadata:
            priority = rule.metadata["priority"]
            if isinstance(priority, int):
                return min(max(priority, 1), 3)

        # Infer from authority level
        if rule.authority == AuthorityLevel.ABSOLUTE:
            return 1  # P1 - Critical
        else:
            return 2  # P2 - Important

    def _sort_by_priority(self, rules: List[MemoryRule]) -> List[MemoryRule]:
        """
        Sort rules by priority (highest first).

        Args:
            rules: List of memory rules

        Returns:
            Sorted list
        """
        return sorted(
            rules,
            key=lambda r: (
                self._get_rule_priority(r),  # Priority
                r.authority == AuthorityLevel.ABSOLUTE,  # Absolute first
                -r.created_at.timestamp(),  # Most recent first
            ),
        )

    def _filter_by_priority(
        self, rules: List[MemoryRule], priority_filter: str
    ) -> List[MemoryRule]:
        """
        Filter rules by priority level.

        Args:
            rules: List of rules to filter
            priority_filter: Priority filter (e.g., "P1", "P2", "P3")

        Returns:
            Filtered list of rules
        """
        # Parse priority filter (e.g., "P1" -> 1)
        try:
            target_priority = int(priority_filter.strip().upper().replace("P", ""))
        except (ValueError, AttributeError):
            logger.warning(f"Invalid priority filter: {priority_filter}")
            return rules

        return [r for r in rules if self._get_rule_priority(r) == target_priority]
