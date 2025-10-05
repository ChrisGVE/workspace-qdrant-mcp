"""
Rule retrieval module for fetching memory rules from Qdrant via daemon.

This module provides the interface for retrieving memory rules from the
memory collection with filtering by scope, project, category, and authority level.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from ..memory import MemoryRule, MemoryCategory, AuthorityLevel, MemoryManager


@dataclass
class RuleFilter:
    """
    Filter criteria for rule retrieval.

    Attributes:
        scope: Filter by scope (empty = global rules)
        project_id: Filter by project
        category: Filter by category
        authority: Filter by authority level
        tags: Filter by tags
        limit: Maximum number of rules to return
    """

    scope: Optional[List[str]] = None
    project_id: Optional[str] = None
    category: Optional[MemoryCategory] = None
    authority: Optional[AuthorityLevel] = None
    tags: Optional[List[str]] = None
    limit: int = 100


@dataclass
class RuleRetrievalResult:
    """
    Result of rule retrieval operation.

    Attributes:
        rules: Retrieved memory rules
        total_count: Total number of matching rules
        filtered_count: Number of rules after filtering
        cache_hit: Whether result came from cache
    """

    rules: List[MemoryRule]
    total_count: int
    filtered_count: int
    cache_hit: bool = False


class RuleRetrieval:
    """
    Rule retrieval service for fetching memory rules.

    This class provides high-level methods for retrieving memory rules
    from the memory collection with filtering and caching.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initialize the rule retrieval service.

        Args:
            memory_manager: MemoryManager instance for accessing rules
        """
        self.memory_manager = memory_manager

    async def get_rules(self, filter: RuleFilter) -> RuleRetrievalResult:
        """
        Retrieve memory rules with filtering.

        Args:
            filter: Filter criteria for rule retrieval

        Returns:
            RuleRetrievalResult with retrieved rules
        """
        try:
            # Get all rules from memory manager with basic filters
            all_rules = await self.memory_manager.list_memory_rules(
                category=filter.category,
                authority=filter.authority,
                scope=filter.scope[0] if filter.scope else None,
            )

            # Apply additional filters
            filtered_rules = self._apply_filters(all_rules, filter)

            # Sort by authority and priority
            sorted_rules = self._sort_by_priority(filtered_rules)

            # Apply limit
            limited_rules = sorted_rules[: filter.limit]

            logger.debug(
                f"Retrieved {len(limited_rules)} rules "
                f"(from {len(all_rules)} total, {len(filtered_rules)} after filtering)"
            )

            return RuleRetrievalResult(
                rules=limited_rules,
                total_count=len(all_rules),
                filtered_count=len(filtered_rules),
                cache_hit=False,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve rules: {e}")
            return RuleRetrievalResult(
                rules=[], total_count=0, filtered_count=0, cache_hit=False
            )

    async def get_rules_by_scope(
        self, scope: List[str], project_id: Optional[str] = None
    ) -> List[MemoryRule]:
        """
        Retrieve rules by scope and optional project.

        Args:
            scope: List of scope contexts
            project_id: Optional project ID filter

        Returns:
            List of matching memory rules
        """
        filter = RuleFilter(scope=scope, project_id=project_id)
        result = await self.get_rules(filter)
        return result.rules

    async def get_absolute_rules(self, scope: List[str]) -> List[MemoryRule]:
        """
        Retrieve absolute authority rules for given scope.

        Args:
            scope: List of scope contexts

        Returns:
            List of absolute authority rules
        """
        filter = RuleFilter(scope=scope, authority=AuthorityLevel.ABSOLUTE)
        result = await self.get_rules(filter)
        return result.rules

    async def get_rules_by_category(
        self, category: MemoryCategory
    ) -> List[MemoryRule]:
        """
        Retrieve rules by category.

        Args:
            category: Memory category to filter by

        Returns:
            List of rules in the specified category
        """
        filter = RuleFilter(category=category)
        result = await self.get_rules(filter)
        return result.rules

    async def search_rules(
        self, query: str, limit: int = 10, filter: Optional[RuleFilter] = None
    ) -> List[Tuple[MemoryRule, float]]:
        """
        Search rules by semantic similarity.

        Args:
            query: Search query
            limit: Maximum number of results
            filter: Optional filter criteria

        Returns:
            List of (MemoryRule, relevance_score) tuples
        """
        try:
            # Use memory manager's search functionality
            # Note: only pass category and authority to search since those are built into the search
            results = await self.memory_manager.search_memory_rules(
                query=query,
                limit=limit * 2 if filter else limit,  # Get more results if filtering
                category=filter.category if filter else None,
                authority=filter.authority if filter else None,
            )

            # Apply additional filters if provided (for project_id, scope, tags, etc.)
            if filter and (filter.project_id or filter.scope or filter.tags):
                filtered_results = []
                for rule, score in results:
                    if self._matches_filter(rule, filter):
                        filtered_results.append((rule, score))
                results = filtered_results[:limit]  # Apply limit after filtering
            elif filter:
                results = results[:limit]

            logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to search rules: {e}")
            return []

    def _apply_filters(
        self, rules: List[MemoryRule], filter: RuleFilter
    ) -> List[MemoryRule]:
        """
        Apply filter criteria to rules.

        Args:
            rules: List of rules to filter
            filter: Filter criteria

        Returns:
            Filtered list of rules
        """
        filtered = []

        for rule in rules:
            if self._matches_filter(rule, filter):
                filtered.append(rule)

        return filtered

    def _matches_filter(self, rule: MemoryRule, filter: RuleFilter) -> bool:
        """
        Check if a rule matches filter criteria.

        Args:
            rule: Memory rule to check
            filter: Filter criteria

        Returns:
            True if rule matches all filter criteria
        """
        # Check project_id
        if filter.project_id:
            rule_project_id = (
                rule.metadata.get("project_id") if rule.metadata else None
            )
            if rule_project_id != filter.project_id:
                return False

        # Check scope (any overlap)
        if filter.scope:
            if rule.scope:
                if not any(s in rule.scope for s in filter.scope):
                    return False
            else:
                # Rule with empty scope is global, always matches
                pass

        # Check tags (any overlap)
        if filter.tags:
            rule_tags = rule.metadata.get("tags", []) if rule.metadata else []
            if not any(t in rule_tags for t in filter.tags):
                return False

        return True

    def _sort_by_priority(self, rules: List[MemoryRule]) -> List[MemoryRule]:
        """
        Sort rules by authority and priority.

        Args:
            rules: List of rules to sort

        Returns:
            Sorted list of rules
        """
        return sorted(
            rules,
            key=lambda r: (
                -(1 if r.authority == AuthorityLevel.ABSOLUTE else 0),  # Absolute first (negative for sort)
                -(r.metadata.get("priority", 50) if r.metadata else 50),  # Higher priority first
                -r.created_at.timestamp(),  # More recent first
            ),
        )
