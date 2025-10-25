"""
Authority level filtering system for memory rules.

This module handles authority-based filtering, precedence resolution,
and hierarchical rule application (global > project > local).
"""

from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..memory import AuthorityLevel, MemoryRule


class RuleHierarchy(Enum):
    """Rule scope hierarchy levels."""

    GLOBAL = "global"  # Empty scope - applies everywhere
    PROJECT = "project"  # Project-specific rules
    LOCAL = "local"  # Context-specific rules


@dataclass
class AuthorityPrecedence:
    """
    Authority precedence rules for conflict resolution.

    Precedence order (highest to lowest):
    1. ABSOLUTE authority at any level
    2. DEFAULT authority at more specific level (local > project > global)
    3. DEFAULT authority at less specific level
    """

    authority: AuthorityLevel
    hierarchy: RuleHierarchy
    priority: int = 50  # Default priority within same authority/hierarchy


@dataclass
class FilteredRules:
    """
    Result of authority-based filtering.

    Attributes:
        absolute_rules: Rules with absolute authority
        default_rules: Rules with default authority
        conflicts_resolved: Number of conflicts resolved
        rules_by_hierarchy: Rules grouped by hierarchy level
    """

    absolute_rules: list[MemoryRule]
    default_rules: list[MemoryRule]
    conflicts_resolved: int
    rules_by_hierarchy: dict[RuleHierarchy, list[MemoryRule]]


class AuthorityFilter:
    """
    Authority-based filtering system for memory rules.

    This class implements:
    1. Authority level validation
    2. Precedence resolution for conflicting authorities
    3. Hierarchical rule application (global > project > local)
    4. Conflict detection and resolution
    """

    def __init__(self, enable_precedence_resolution: bool = True):
        """
        Initialize the authority filter.

        Args:
            enable_precedence_resolution: Whether to auto-resolve conflicts
        """
        self.enable_precedence_resolution = enable_precedence_resolution

    def filter_by_authority(
        self,
        rules: list[MemoryRule],
        project_id: str | None = None,
        scope: list[str] | None = None,
    ) -> FilteredRules:
        """
        Filter rules by authority level with hierarchy support.

        Args:
            rules: List of memory rules to filter
            project_id: Current project ID for project-specific filtering
            scope: Current scope contexts for local filtering

        Returns:
            FilteredRules with separated and organized rules
        """
        # Separate by authority level
        absolute_rules = [
            r for r in rules if r.authority == AuthorityLevel.ABSOLUTE
        ]
        default_rules = [r for r in rules if r.authority == AuthorityLevel.DEFAULT]

        # Group by hierarchy level
        rules_by_hierarchy = self._group_by_hierarchy(rules, project_id, scope)

        # Resolve conflicts if enabled
        conflicts_resolved = 0
        if self.enable_precedence_resolution:
            # Check for overlapping rules and resolve
            conflicts_resolved = self._resolve_authority_conflicts(
                absolute_rules, default_rules, project_id, scope
            )

        logger.debug(
            f"Authority filtering: {len(absolute_rules)} absolute, "
            f"{len(default_rules)} default, {conflicts_resolved} conflicts resolved"
        )

        return FilteredRules(
            absolute_rules=absolute_rules,
            default_rules=default_rules,
            conflicts_resolved=conflicts_resolved,
            rules_by_hierarchy=rules_by_hierarchy,
        )

    def apply_authority_precedence(
        self, rules: list[MemoryRule], project_id: str | None = None
    ) -> list[MemoryRule]:
        """
        Apply authority precedence to resolve conflicts.

        Precedence order:
        1. Absolute authority (always takes precedence)
        2. More specific scope (local > project > global)
        3. Higher priority within same authority/scope
        4. More recent creation time

        Args:
            rules: List of rules (may have conflicts)
            project_id: Current project ID

        Returns:
            List of rules with conflicts resolved
        """
        if not rules:
            return []

        # Build precedence map for each rule
        rule_precedence = {}
        for rule in rules:
            precedence = self._calculate_precedence(rule, project_id)
            rule_precedence[rule.id] = precedence

        # Sort by precedence (higher precedence first)
        sorted_rules = sorted(
            rules,
            key=lambda r: self._precedence_sort_key(rule_precedence[r.id], r),
            reverse=True,
        )

        # Detect and remove conflicting rules (keep highest precedence)
        resolved_rules = self._remove_conflicts(sorted_rules, project_id)

        logger.debug(
            f"Applied precedence: {len(rules)} -> {len(resolved_rules)} "
            f"({len(rules) - len(resolved_rules)} conflicts removed)"
        )

        return resolved_rules

    def validate_authority_level(
        self, rule: MemoryRule, allow_absolute: bool = True
    ) -> bool:
        """
        Validate that rule's authority level is valid.

        Args:
            rule: Rule to validate
            allow_absolute: Whether absolute authority is allowed

        Returns:
            True if authority level is valid
        """
        # Check if authority is a valid enum value
        if not isinstance(rule.authority, AuthorityLevel):
            logger.warning(f"Invalid authority type for rule {rule.id}")
            return False

        # Check if absolute authority is allowed
        if rule.authority == AuthorityLevel.ABSOLUTE and not allow_absolute:
            logger.warning(
                f"Absolute authority not allowed for rule {rule.id} in this context"
            )
            return False

        return True

    def get_effective_rules(
        self,
        rules: list[MemoryRule],
        project_id: str | None = None,
        scope: list[str] | None = None,
    ) -> list[MemoryRule]:
        """
        Get effective rules after applying authority and hierarchy filtering.

        This combines:
        1. Authority filtering (absolute + default)
        2. Hierarchy filtering (global + project + local)
        3. Precedence resolution
        4. Conflict removal

        Args:
            rules: All available rules
            project_id: Current project ID
            scope: Current scope contexts

        Returns:
            List of effective rules to apply
        """
        # Filter by authority and hierarchy
        filtered = self.filter_by_authority(rules, project_id, scope)

        # Combine absolute and default rules
        all_rules = filtered.absolute_rules + filtered.default_rules

        # Apply precedence to resolve any remaining conflicts
        effective_rules = self.apply_authority_precedence(all_rules, project_id)

        return effective_rules

    def _group_by_hierarchy(
        self,
        rules: list[MemoryRule],
        project_id: str | None,
        scope: list[str] | None,
    ) -> dict[RuleHierarchy, list[MemoryRule]]:
        """
        Group rules by hierarchy level.

        Args:
            rules: Rules to group
            project_id: Current project ID
            scope: Current scope contexts

        Returns:
            Dictionary mapping hierarchy level to rules
        """
        hierarchy_map = {
            RuleHierarchy.GLOBAL: [],
            RuleHierarchy.PROJECT: [],
            RuleHierarchy.LOCAL: [],
        }

        for rule in rules:
            hierarchy_level = self._determine_hierarchy(rule, project_id, scope)
            hierarchy_map[hierarchy_level].append(rule)

        return hierarchy_map

    def _determine_hierarchy(
        self,
        rule: MemoryRule,
        project_id: str | None,
        scope: list[str] | None,
    ) -> RuleHierarchy:
        """
        Determine hierarchy level for a rule.

        Args:
            rule: Rule to classify
            project_id: Current project ID
            scope: Current scope contexts

        Returns:
            Hierarchy level for this rule
        """
        # Check if rule has specific scope first (most specific)
        if rule.scope:
            # Local: has specific scope (regardless of project_id)
            # Local is more specific than project
            return RuleHierarchy.LOCAL

        # Project: has project_id matching current project (but no specific scope)
        rule_project_id = rule.metadata.get("project_id") if rule.metadata else None
        if rule_project_id and rule_project_id == project_id:
            return RuleHierarchy.PROJECT

        # Global: no scope and no project_id
        return RuleHierarchy.GLOBAL

    def _calculate_precedence(
        self, rule: MemoryRule, project_id: str | None
    ) -> AuthorityPrecedence:
        """
        Calculate precedence for a rule.

        Args:
            rule: Rule to calculate precedence for
            project_id: Current project ID

        Returns:
            AuthorityPrecedence object
        """
        hierarchy = self._determine_hierarchy(rule, project_id, None)
        priority = rule.metadata.get("priority", 50) if rule.metadata else 50

        return AuthorityPrecedence(
            authority=rule.authority, hierarchy=hierarchy, priority=priority
        )

    def _precedence_sort_key(
        self, precedence: AuthorityPrecedence, rule: MemoryRule
    ) -> tuple:
        """
        Generate sort key for precedence ordering.

        Args:
            precedence: Rule's precedence
            rule: The rule itself (for tie-breaking)

        Returns:
            Tuple for sorting (higher values = higher precedence)
        """
        # Authority level (absolute = 2, default = 1)
        authority_level = 2 if precedence.authority == AuthorityLevel.ABSOLUTE else 1

        # Hierarchy level (local = 3, project = 2, global = 1)
        hierarchy_level = {
            RuleHierarchy.LOCAL: 3,
            RuleHierarchy.PROJECT: 2,
            RuleHierarchy.GLOBAL: 1,
        }[precedence.hierarchy]

        # Priority within same authority/hierarchy
        priority = precedence.priority

        # Recency (newer rules preferred)
        recency = rule.created_at.timestamp()

        return (authority_level, hierarchy_level, priority, recency)

    def _remove_conflicts(
        self, sorted_rules: list[MemoryRule], project_id: str | None
    ) -> list[MemoryRule]:
        """
        Remove conflicting rules, keeping highest precedence.

        Two rules conflict if they apply to the same scope/context
        but have different instructions.

        Args:
            sorted_rules: Rules sorted by precedence (highest first)
            project_id: Current project ID

        Returns:
            List of rules with conflicts removed
        """
        if not sorted_rules:
            return []

        # Track which scopes/contexts are covered
        covered_contexts: set[str] = set()
        resolved_rules = []

        for rule in sorted_rules:
            # Determine rule's context key
            context_key = self._get_context_key(rule, project_id)

            # If this context is already covered by a higher-precedence rule, skip
            if context_key in covered_contexts:
                logger.debug(
                    f"Skipping rule {rule.id} due to higher-precedence rule "
                    f"covering same context: {context_key}"
                )
                continue

            # Add this rule and mark its context as covered
            resolved_rules.append(rule)
            covered_contexts.add(context_key)

        return resolved_rules

    def _get_context_key(self, rule: MemoryRule, project_id: str | None) -> str:
        """
        Get context key for conflict detection.

        Args:
            rule: Rule to get context for
            project_id: Current project ID

        Returns:
            String key representing rule's context
        """
        # Combine category, scope, and project into a unique key
        category = rule.category.value
        scope = ":".join(sorted(rule.scope)) if rule.scope else "global"
        proj = rule.metadata.get("project_id", "global") if rule.metadata else "global"

        return f"{category}:{scope}:{proj}"

    def _resolve_authority_conflicts(
        self,
        absolute_rules: list[MemoryRule],
        default_rules: list[MemoryRule],
        project_id: str | None,
        scope: list[str] | None,
    ) -> int:
        """
        Resolve conflicts between absolute and default rules.

        Absolute rules always take precedence over default rules
        in the same context.

        Args:
            absolute_rules: Rules with absolute authority
            default_rules: Rules with default authority
            project_id: Current project ID
            scope: Current scope contexts

        Returns:
            Number of conflicts resolved
        """
        if not absolute_rules or not default_rules:
            return 0

        conflicts_resolved = 0

        # Build set of contexts covered by absolute rules
        absolute_contexts = set()
        for rule in absolute_rules:
            context_key = self._get_context_key(rule, project_id)
            absolute_contexts.add(context_key)

        # Check each default rule for conflicts
        for rule in default_rules:
            context_key = self._get_context_key(rule, project_id)
            if context_key in absolute_contexts:
                logger.debug(
                    f"Conflict detected: default rule {rule.id} conflicts with "
                    f"absolute rule in context {context_key}"
                )
                conflicts_resolved += 1

        return conflicts_resolved
