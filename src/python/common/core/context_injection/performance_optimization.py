"""
Performance optimization utilities for rule retrieval and processing.

This module provides caching, indexing, batch processing, and metrics
collection for efficient handling of large rule sets.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from cachetools import TTLCache
from loguru import logger

from ..memory import AuthorityLevel, MemoryCategory, MemoryRule


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for rule operations.

    Attributes:
        cache_hit: Whether the result came from cache
        fetch_time_ms: Time spent fetching rules from storage
        filter_time_ms: Time spent filtering rules
        total_rules_fetched: Total number of rules fetched
        rules_after_filtering: Number of rules after filtering
        index_lookup_time_ms: Time spent on index lookups (if used)
        batch_processing_time_ms: Time spent on batch processing (if used)
    """

    cache_hit: bool = False
    fetch_time_ms: float = 0.0
    filter_time_ms: float = 0.0
    total_rules_fetched: int = 0
    rules_after_filtering: int = 0
    index_lookup_time_ms: Optional[float] = None
    batch_processing_time_ms: Optional[float] = None


class RuleCache:
    """
    LRU cache with TTL for rule retrieval results.

    This cache stores rule retrieval results with automatic expiration
    to prevent stale data while improving performance for repeated queries.
    """

    def __init__(self, maxsize: int = 128, ttl: int = 300):
        """
        Initialize the rule cache.

        Args:
            maxsize: Maximum number of cached items
            ttl: Time-to-live in seconds for cached items
        """
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0

    def get(self, key: Tuple[Any, ...]) -> Optional[List[MemoryRule]]:
        """
        Get cached rules by key.

        Args:
            key: Cache key tuple

        Returns:
            Cached rules if present, None otherwise
        """
        try:
            result = self._cache[key]
            self._hits += 1
            logger.debug(f"Cache hit for key: {key[:2]}...")
            return result
        except KeyError:
            self._misses += 1
            logger.debug(f"Cache miss for key: {key[:2]}...")
            return None

    def set(self, key: Tuple[Any, ...], value: List[MemoryRule]) -> None:
        """
        Store rules in cache.

        Args:
            key: Cache key tuple
            value: Rules to cache
        """
        self._cache[key] = value
        logger.debug(f"Cached {len(value)} rules for key: {key[:2]}...")

    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        logger.debug("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "current_size": len(self._cache),
            "max_size": self._cache.maxsize,
        }


class RuleIndex:
    """
    Multi-field index for fast rule lookups.

    Maintains in-memory indices for common query patterns to avoid
    O(n) list filtering operations.
    """

    def __init__(self):
        """Initialize empty indices."""
        self._by_category: Dict[MemoryCategory, List[MemoryRule]] = defaultdict(list)
        self._by_authority: Dict[AuthorityLevel, List[MemoryRule]] = defaultdict(
            list
        )
        self._by_project: Dict[str, List[MemoryRule]] = defaultdict(list)
        self._by_scope: Dict[str, List[MemoryRule]] = defaultdict(list)
        self._all_rules: List[MemoryRule] = []
        self._is_built = False

    def build_index(self, rules: List[MemoryRule]) -> None:
        """
        Build indices from rule list.

        Args:
            rules: List of rules to index
        """
        start_time = time.perf_counter()

        # Clear existing indices
        self._by_category.clear()
        self._by_authority.clear()
        self._by_project.clear()
        self._by_scope.clear()
        self._all_rules = list(rules)

        # Build indices
        for rule in rules:
            # Index by category
            self._by_category[rule.category].append(rule)

            # Index by authority
            self._by_authority[rule.authority].append(rule)

            # Index by project_id
            project_id = rule.metadata.get("project_id") if rule.metadata else None
            if project_id:
                self._by_project[project_id].append(rule)

            # Index by scope
            if rule.scope:
                for scope_item in rule.scope:
                    self._by_scope[scope_item].append(rule)

        self._is_built = True
        build_time_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Built index for {len(rules)} rules in {build_time_ms:.2f}ms "
            f"(categories: {len(self._by_category)}, "
            f"authorities: {len(self._by_authority)}, "
            f"projects: {len(self._by_project)}, "
            f"scopes: {len(self._by_scope)})"
        )

    def lookup_by_category(self, category: MemoryCategory) -> List[MemoryRule]:
        """
        Fast lookup by category.

        Args:
            category: Category to lookup

        Returns:
            List of rules in category
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._by_category.get(category, [])

    def lookup_by_authority(self, authority: AuthorityLevel) -> List[MemoryRule]:
        """
        Fast lookup by authority.

        Args:
            authority: Authority level to lookup

        Returns:
            List of rules with authority level
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._by_authority.get(authority, [])

    def lookup_by_project(self, project_id: str) -> List[MemoryRule]:
        """
        Fast lookup by project ID.

        Args:
            project_id: Project ID to lookup

        Returns:
            List of rules for project
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._by_project.get(project_id, [])

    def lookup_by_scope(self, scope: str) -> List[MemoryRule]:
        """
        Fast lookup by scope.

        Args:
            scope: Scope to lookup

        Returns:
            List of rules with scope
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._by_scope.get(scope, [])

    def get_all_rules(self) -> List[MemoryRule]:
        """
        Get all indexed rules.

        Returns:
            List of all rules
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        return self._all_rules

    def is_built(self) -> bool:
        """
        Check if index is built.

        Returns:
            True if index is built
        """
        return self._is_built

    def clear(self) -> None:
        """Clear all indices."""
        self._by_category.clear()
        self._by_authority.clear()
        self._by_project.clear()
        self._by_scope.clear()
        self._all_rules.clear()
        self._is_built = False
        logger.debug("Index cleared")


class BatchProcessor:
    """
    Utility for processing rules in batches.

    Useful for memory-intensive operations on large rule sets.
    """

    @staticmethod
    def process_in_batches(
        rules: List[MemoryRule],
        process_fn: Callable[[List[MemoryRule]], Any],
        batch_size: int = 100,
    ) -> List[Any]:
        """
        Process rules in batches.

        Args:
            rules: List of rules to process
            process_fn: Function to apply to each batch
            batch_size: Number of rules per batch

        Returns:
            List of results from processing each batch
        """
        start_time = time.perf_counter()
        results = []

        for i in range(0, len(rules), batch_size):
            batch = rules[i : i + batch_size]
            result = process_fn(batch)
            results.append(result)

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        num_batches = (len(rules) + batch_size - 1) // batch_size

        logger.debug(
            f"Processed {len(rules)} rules in {num_batches} batches "
            f"({batch_size} rules/batch) in {processing_time_ms:.2f}ms"
        )

        return results

    @staticmethod
    def flatten_batch_results(results: List[List[Any]]) -> List[Any]:
        """
        Flatten batch processing results.

        Args:
            results: List of batch results

        Returns:
            Flattened list of all results
        """
        return [item for batch in results for item in batch]
