"""
Rule retrieval module for fetching memory rules from Qdrant via daemon.

This module provides the interface for retrieving memory rules from the
memory collection with filtering by scope, project, category, and authority level.

Performance features:
- Optional caching with TTL for repeated queries
- Optional indexing for fast lookups
- Pagination support via offset parameter
- Performance metrics collection
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from loguru import logger

from ...memory.manager import MemoryManager
from ...memory.types import AuthorityLevel, MemoryCategory, MemoryRule
from .performance_optimization import (
    PerformanceMetrics,
    RuleCache,
    RuleIndex,
)


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
        offset: Offset for pagination support
    """

    scope: Optional[List[str]] = None
    project_id: Optional[str] = None
    category: Optional[MemoryCategory] = None
    authority: Optional[AuthorityLevel] = None
    tags: Optional[List[str]] = None
    limit: int = 100
    offset: int = 0


@dataclass
class RuleRetrievalResult:
    """
    Result of rule retrieval operation.

    Attributes:
        rules: Retrieved memory rules
        total_count: Total number of matching rules
        filtered_count: Number of rules after filtering
        cache_hit: Whether result came from cache
        metrics: Performance metrics for this retrieval
    """

    rules: List[MemoryRule]
    total_count: int
    filtered_count: int
    cache_hit: bool = False
    metrics: Optional[PerformanceMetrics] = None


class RuleRetrieval:
    """
    Rule retrieval service for fetching memory rules.

    This class provides high-level methods for retrieving memory rules
    from the memory collection with filtering and caching.

    Performance features:
    - Optional LRU cache with TTL for repeated queries
    - Optional indexing for O(1) lookups by category/authority/project
    - Pagination support via offset/limit
    - Performance metrics collection
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        enable_cache: bool = False,
        cache_maxsize: int = 128,
        cache_ttl: int = 300,
        enable_indexing: bool = False,
    ):
        """
        Initialize the rule retrieval service.

        Args:
            memory_manager: MemoryManager instance for accessing rules
            enable_cache: Enable caching of retrieval results
            cache_maxsize: Maximum number of cached items
            cache_ttl: Cache time-to-live in seconds
            enable_indexing: Enable indexing for fast lookups
        """
        self.memory_manager = memory_manager
        self.enable_cache = enable_cache
        self.enable_indexing = enable_indexing

        # Initialize cache if enabled
        self._cache: Optional[RuleCache] = None
        if enable_cache:
            self._cache = RuleCache(maxsize=cache_maxsize, ttl=cache_ttl)
            logger.debug(
                f"Rule cache enabled (maxsize={cache_maxsize}, ttl={cache_ttl}s)"
            )

        # Initialize index if enabled
        self._index: Optional[RuleIndex] = None
        if enable_indexing:
            self._index = RuleIndex()
            logger.debug("Rule indexing enabled")

    async def get_rules(self, filter: RuleFilter) -> RuleRetrievalResult:
        """
        Retrieve memory rules with filtering.

        Supports caching and indexing for improved performance.

        Args:
            filter: Filter criteria for rule retrieval

        Returns:
            RuleRetrievalResult with retrieved rules and metrics
        """
        metrics = PerformanceMetrics()
        start_time = time.perf_counter()

        try:
            # Generate cache key from filter parameters
            cache_key = self._make_cache_key(filter)

            # Check cache if enabled
            if self._cache:
                cached_rules = self._cache.get(cache_key)
                if cached_rules is not None:
                    # Apply pagination to cached results
                    paginated_rules = cached_rules[
                        filter.offset : filter.offset + filter.limit
                    ]

                    metrics.cache_hit = True
                    metrics.fetch_time_ms = (time.perf_counter() - start_time) * 1000

                    logger.debug(
                        f"Cache hit: Retrieved {len(paginated_rules)} rules from cache"
                    )

                    return RuleRetrievalResult(
                        rules=paginated_rules,
                        total_count=len(cached_rules),
                        filtered_count=len(cached_rules),
                        cache_hit=True,
                        metrics=metrics,
                    )

            # Cache miss - fetch from storage
            fetch_start = time.perf_counter()

            # Use index if enabled and filter allows
            if self._index and self._index.is_built():
                all_rules = await self._get_rules_with_index(filter)
                metrics.index_lookup_time_ms = (
                    time.perf_counter() - fetch_start
                ) * 1000
            else:
                # Get all rules from memory manager with basic filters
                all_rules = await self.memory_manager.list_rules(
                    category_filter=filter.category,
                    authority_filter=filter.authority,
                    # Note: list_rules doesn't have a scope filter, we'll filter in-memory
                )

            metrics.fetch_time_ms = (time.perf_counter() - fetch_start) * 1000

            # Apply additional filters
            filter_start = time.perf_counter()
            filtered_rules = self._apply_filters(all_rules, filter)

            # Sort by authority and priority
            sorted_rules = self._sort_by_priority(filtered_rules)
            metrics.filter_time_ms = (time.perf_counter() - filter_start) * 1000

            metrics.total_rules_fetched = len(all_rules)
            metrics.rules_after_filtering = len(sorted_rules)

            # Cache the sorted results if caching enabled
            if self._cache:
                self._cache.set(cache_key, sorted_rules)

            # Apply pagination
            paginated_rules = sorted_rules[
                filter.offset : filter.offset + filter.limit
            ]

            logger.debug(
                f"Retrieved {len(paginated_rules)} rules "
                f"(from {len(all_rules)} total, {len(filtered_rules)} after filtering) "
                f"in {metrics.fetch_time_ms + metrics.filter_time_ms:.2f}ms"
            )

            return RuleRetrievalResult(
                rules=paginated_rules,
                total_count=len(all_rules),
                filtered_count=len(sorted_rules),
                cache_hit=False,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve rules: {e}")
            return RuleRetrievalResult(
                rules=[],
                total_count=0,
                filtered_count=0,
                cache_hit=False,
                metrics=metrics,
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
            # Note: search_rules returns List[Tuple[MemoryRule, float]]
            results = await self.memory_manager.search_rules(
                query=query,
                limit=limit * 2 if filter else limit,  # Get more results if filtering
                category_filter=filter.category if filter else None,
                authority_filter=filter.authority if filter else None,
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

    def _make_cache_key(self, filter: RuleFilter) -> Tuple:
        """
        Generate cache key from filter parameters.

        Args:
            filter: Filter criteria

        Returns:
            Hashable tuple for use as cache key
        """
        return (
            tuple(sorted(filter.scope)) if filter.scope else (),
            filter.project_id,
            filter.category.value if filter.category else None,
            filter.authority.value if filter.authority else None,
            tuple(sorted(filter.tags)) if filter.tags else (),
            filter.limit,
            filter.offset,
        )

    async def _get_rules_with_index(self, filter: RuleFilter) -> List[MemoryRule]:
        """
        Get rules using index for fast lookup.

        Args:
            filter: Filter criteria

        Returns:
            List of rules from index
        """
        if not self._index or not self._index.is_built():
            # Fallback to regular fetch
            return await self.memory_manager.list_rules(
                category_filter=filter.category,
                authority_filter=filter.authority,
                # Note: list_rules doesn't have scope filter
            )

        # Use index for fast lookup if filter criteria match
        if filter.category:
            return self._index.lookup_by_category(filter.category)
        elif filter.authority:
            return self._index.lookup_by_authority(filter.authority)
        elif filter.project_id:
            return self._index.lookup_by_project(filter.project_id)
        elif filter.scope and len(filter.scope) == 1:
            return self._index.lookup_by_scope(filter.scope[0])
        else:
            # Complex filter - use all rules
            return self._index.get_all_rules()

    async def build_index(self) -> None:
        """
        Build index from all rules in memory.

        Should be called after significant rule additions/updates.
        """
        if not self._index:
            logger.warning("Index not enabled - skipping build")
            return

        # Fetch all rules
        all_rules = await self.memory_manager.list_rules()

        # Build index
        self._index.build_index(all_rules)

        logger.info(f"Built index for {len(all_rules)} rules")

    def clear_cache(self) -> None:
        """Clear the rule cache if enabled."""
        if self._cache:
            self._cache.clear()
            logger.debug("Cache cleared")

    def get_cache_stats(self) -> Optional[dict]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats or None if cache disabled
        """
        if self._cache:
            return self._cache.get_stats()
        return None
