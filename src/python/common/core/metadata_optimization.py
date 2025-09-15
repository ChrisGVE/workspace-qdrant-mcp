"""
Efficient metadata filtering strategies for Qdrant workspace collections.

This module implements advanced filtering optimization techniques to maintain sub-3ms
response times while providing comprehensive metadata-based filtering capabilities.
Key features include filter caching, query optimization, and metadata indexing strategies.

Performance Targets:
    - Sub-3ms response times for filtered searches
    - Efficient metadata field indexing
    - Filter condition caching and reuse
    - Query optimization techniques

Task 233.3: Implementing efficient metadata filtering strategies with:
    - FilterOptimizer for advanced filter caching and query optimization
    - MetadataIndexManager for optimized indexing strategies
    - QueryOptimizer for sub-3ms query optimization techniques
    - PerformanceTracker for monitoring performance targets
"""

import asyncio
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
from datetime import datetime, timedelta

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

from .multitenant_collections import ProjectMetadata, ProjectIsolationManager


@dataclass
class FilterCacheEntry:
    """Cached filter entry with performance metadata."""

    filter_condition: models.Filter
    cache_key: str
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    avg_execution_time: float = 0.0

    def update_performance(self, execution_time: float) -> None:
        """Update performance metrics for this cached filter."""
        self.access_count += 1
        self.last_accessed = datetime.now()

        # Update rolling average execution time
        if self.avg_execution_time == 0.0:
            self.avg_execution_time = execution_time
        else:
            # Weighted average favoring recent performance
            weight = 0.7
            self.avg_execution_time = (weight * execution_time +
                                     (1 - weight) * self.avg_execution_time)


@dataclass
class PerformanceMetrics:
    """Performance tracking for metadata filtering operations."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    sub_3ms_percentage: float = 0.0
    response_times: List[float] = field(default_factory=list)

    def add_response_time(self, response_time: float) -> None:
        """Add a response time measurement and update metrics."""
        self.total_queries += 1
        self.response_times.append(response_time)

        # Keep only recent measurements (last 1000)
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]

        # Update metrics
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update calculated metrics from response times."""
        if not self.response_times:
            return

        sorted_times = sorted(self.response_times)
        n = len(sorted_times)

        self.avg_response_time = sum(sorted_times) / n
        self.p95_response_time = sorted_times[int(0.95 * n)] if n > 0 else 0.0
        self.p99_response_time = sorted_times[int(0.99 * n)] if n > 0 else 0.0

        # Calculate percentage of sub-3ms responses
        sub_3ms_count = sum(1 for t in sorted_times if t < 3.0)
        self.sub_3ms_percentage = (sub_3ms_count / n) * 100 if n > 0 else 0.0


class FilterOptimizer:
    """
    Advanced filter caching and query optimization for metadata filtering.

    Implements intelligent caching strategies to minimize filter construction overhead
    and optimize query performance for commonly used filter patterns.
    """

    def __init__(self, cache_size: int = 500, cache_ttl_minutes: int = 60):
        """
        Initialize filter optimizer with caching configuration.

        Args:
            cache_size: Maximum number of cached filter entries
            cache_ttl_minutes: Time-to-live for cached entries in minutes
        """
        self.cache_size = cache_size
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._filter_cache: Dict[str, FilterCacheEntry] = {}
        self._performance = PerformanceMetrics()

        logger.info("Filter optimizer initialized",
                   cache_size=cache_size, cache_ttl_minutes=cache_ttl_minutes)

    def get_optimized_filter(
        self,
        project_context: Optional[Union[Dict, ProjectMetadata]] = None,
        collection_type: Optional[str] = None,
        additional_filters: Optional[Dict] = None,
        base_filter: Optional[models.Filter] = None
    ) -> Tuple[models.Filter, bool]:
        """
        Get optimized filter with caching and performance tracking.

        Args:
            project_context: Project context for metadata filtering
            collection_type: Optional collection type filter
            additional_filters: Additional metadata filters
            base_filter: Base filter to combine with

        Returns:
            Tuple of (optimized_filter, cache_hit)
        """
        start_time = time.time()

        # Generate cache key for this filter combination
        cache_key = self._generate_cache_key(
            project_context, collection_type, additional_filters, base_filter
        )

        # Check cache first
        cached_entry = self._get_from_cache(cache_key)
        if cached_entry:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            cached_entry.update_performance(execution_time)
            self._performance.cache_hits += 1
            self._performance.add_response_time(execution_time)

            logger.debug("Filter cache hit", cache_key=cache_key[:16],
                        execution_time=execution_time)
            return cached_entry.filter_condition, True

        # Cache miss - create optimized filter
        self._performance.cache_misses += 1
        optimized_filter = self._create_optimized_filter(
            project_context, collection_type, additional_filters, base_filter
        )

        # Store in cache
        execution_time = (time.time() - start_time) * 1000
        self._store_in_cache(cache_key, optimized_filter, execution_time)
        self._performance.add_response_time(execution_time)

        logger.debug("Filter created and cached", cache_key=cache_key[:16],
                    execution_time=execution_time)
        return optimized_filter, False

    def _generate_cache_key(
        self,
        project_context: Optional[Union[Dict, ProjectMetadata]],
        collection_type: Optional[str],
        additional_filters: Optional[Dict],
        base_filter: Optional[models.Filter]
    ) -> str:
        """Generate stable cache key for filter combination."""
        key_components = []

        # Project context component
        if isinstance(project_context, ProjectMetadata):
            key_components.append(f"pm:{project_context.tenant_namespace}")
        elif isinstance(project_context, dict):
            tenant_ns = project_context.get("tenant_namespace", "")
            project_name = project_context.get("project_name", "")
            key_components.append(f"pd:{tenant_ns}:{project_name}")
        else:
            key_components.append("pc:none")

        # Collection type component
        key_components.append(f"ct:{collection_type or 'none'}")

        # Additional filters component
        if additional_filters:
            sorted_filters = sorted(additional_filters.items())
            filters_str = str(sorted_filters)
            key_components.append(f"af:{hashlib.md5(filters_str.encode()).hexdigest()[:8]}")
        else:
            key_components.append("af:none")

        # Base filter component (simplified)
        if base_filter:
            # Create a simple hash of filter structure
            filter_str = str(base_filter.dict() if hasattr(base_filter, 'dict') else str(base_filter))
            key_components.append(f"bf:{hashlib.md5(filter_str.encode()).hexdigest()[:8]}")
        else:
            key_components.append("bf:none")

        cache_key = "|".join(key_components)
        return hashlib.sha256(cache_key.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[FilterCacheEntry]:
        """Get filter from cache if valid and not expired."""
        if cache_key not in self._filter_cache:
            return None

        entry = self._filter_cache[cache_key]

        # Check if expired
        if datetime.now() - entry.created_at > self.cache_ttl:
            del self._filter_cache[cache_key]
            logger.debug("Cache entry expired", cache_key=cache_key[:16])
            return None

        return entry

    def _store_in_cache(
        self,
        cache_key: str,
        filter_condition: models.Filter,
        execution_time: float
    ) -> None:
        """Store filter in cache with LRU eviction."""
        # Implement LRU eviction if cache is full
        if len(self._filter_cache) >= self.cache_size:
            self._evict_lru_entries()

        entry = FilterCacheEntry(
            filter_condition=filter_condition,
            cache_key=cache_key,
            created_at=datetime.now(),
            avg_execution_time=execution_time
        )

        self._filter_cache[cache_key] = entry

    def _evict_lru_entries(self) -> None:
        """Evict least recently used cache entries."""
        if not self._filter_cache:
            return

        # Sort by last accessed time (None means never accessed, so oldest)
        sorted_entries = sorted(
            self._filter_cache.items(),
            key=lambda x: x[1].last_accessed or x[1].created_at
        )

        # Remove oldest 20% of entries
        entries_to_remove = max(1, len(sorted_entries) // 5)
        for i in range(entries_to_remove):
            cache_key, _ = sorted_entries[i]
            del self._filter_cache[cache_key]

        logger.debug("Evicted LRU cache entries", count=entries_to_remove)

    def _create_optimized_filter(
        self,
        project_context: Optional[Union[Dict, ProjectMetadata]],
        collection_type: Optional[str],
        additional_filters: Optional[Dict],
        base_filter: Optional[models.Filter]
    ) -> models.Filter:
        """Create optimized filter using best practices."""
        conditions = []

        # Add project context conditions (most selective first)
        if project_context:
            if isinstance(project_context, ProjectMetadata):
                # Use tenant namespace for most precise filtering
                conditions.append(
                    models.FieldCondition(
                        key="tenant_namespace",
                        match=models.MatchValue(value=project_context.tenant_namespace)
                    )
                )

                # Add collection type if different from context
                if collection_type and collection_type != project_context.collection_type:
                    conditions.append(
                        models.FieldCondition(
                            key="collection_type",
                            match=models.MatchValue(value=collection_type)
                        )
                    )
            elif isinstance(project_context, dict):
                # Legacy dict format
                if "tenant_namespace" in project_context:
                    conditions.append(
                        models.FieldCondition(
                            key="tenant_namespace",
                            match=models.MatchValue(value=project_context["tenant_namespace"])
                        )
                    )
                elif "project_name" in project_context:
                    conditions.append(
                        models.FieldCondition(
                            key="project_name",
                            match=models.MatchValue(value=project_context["project_name"])
                        )
                    )

                # Add collection type if specified
                if collection_type:
                    conditions.append(
                        models.FieldCondition(
                            key="collection_type",
                            match=models.MatchValue(value=collection_type)
                        )
                    )

        # Add additional filters (optimize for exact matches)
        if additional_filters:
            for key, value in additional_filters.items():
                if isinstance(value, str):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                elif isinstance(value, (int, float)):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )
                elif isinstance(value, list):
                    conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=value)
                        )
                    )

        # Combine with base filter if provided
        if base_filter:
            # Merge conditions efficiently
            base_conditions = base_filter.must or []
            all_conditions = base_conditions + conditions

            return models.Filter(
                must=all_conditions,
                should=base_filter.should,
                must_not=base_filter.must_not
            )
        else:
            return models.Filter(must=conditions)

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics."""
        cache_hit_rate = (
            (self._performance.cache_hits /
             (self._performance.cache_hits + self._performance.cache_misses)) * 100
            if (self._performance.cache_hits + self._performance.cache_misses) > 0
            else 0.0
        )

        return {
            "filter_cache_size": len(self._filter_cache),
            "cache_hit_rate": cache_hit_rate,
            "total_queries": self._performance.total_queries,
            "avg_response_time": self._performance.avg_response_time,
            "p95_response_time": self._performance.p95_response_time,
            "p99_response_time": self._performance.p99_response_time,
            "sub_3ms_percentage": self._performance.sub_3ms_percentage,
            "performance_target_met": self._performance.sub_3ms_percentage >= 95.0
        }

    def clear_cache(self) -> None:
        """Clear the filter cache."""
        self._filter_cache.clear()
        logger.info("Filter cache cleared")


class MetadataIndexManager:
    """
    Optimized indexing strategies for metadata fields in Qdrant collections.

    Manages creation and optimization of payload indexes to ensure efficient
    metadata filtering with minimal performance impact.
    """

    def __init__(self, client: QdrantClient):
        """Initialize metadata index manager."""
        self.client = client
        self._indexed_collections: Set[str] = set()
        self._index_configurations: Dict[str, Dict] = {}

        # Optimal index configurations for different field types
        self.optimal_index_configs = {
            # High-cardinality exact match fields
            "tenant_namespace": models.KeywordIndexParams(),
            "project_name": models.KeywordIndexParams(),
            "collection_type": models.KeywordIndexParams(),
            "workspace_scope": models.KeywordIndexParams(),

            # Medium-cardinality fields
            "created_by": models.KeywordIndexParams(),
            "access_level": models.KeywordIndexParams(),
            "category": models.KeywordIndexParams(),

            # Numeric fields
            "version": models.IntegerIndexParams(),
            "priority": models.IntegerIndexParams(),

            # Full-text search fields (if needed)
            "tags": models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True
            )
        }

        logger.info("Metadata index manager initialized")

    async def ensure_optimal_indexes(
        self,
        collection_name: str,
        force_recreate: bool = False
    ) -> Dict[str, bool]:
        """
        Ensure optimal indexes exist for metadata filtering.

        Args:
            collection_name: Collection to optimize
            force_recreate: Whether to recreate existing indexes

        Returns:
            Dict mapping field names to success status
        """
        if collection_name in self._indexed_collections and not force_recreate:
            logger.debug("Collection already optimized", collection=collection_name)
            return {}

        logger.info("Creating optimal metadata indexes", collection=collection_name)

        results = {}

        for field_name, index_config in self.optimal_index_configs.items():
            try:
                success = await self._create_field_index(
                    collection_name, field_name, index_config, force_recreate
                )
                results[field_name] = success

                if success:
                    logger.debug("Index created",
                               collection=collection_name, field=field_name)

            except Exception as e:
                logger.warning("Index creation failed",
                             collection=collection_name, field=field_name, error=str(e))
                results[field_name] = False

        # Mark collection as indexed
        if any(results.values()):
            self._indexed_collections.add(collection_name)
            self._index_configurations[collection_name] = results

        success_count = sum(results.values())
        logger.info("Index optimization completed",
                   collection=collection_name,
                   successful=success_count,
                   total=len(results))

        return results

    async def _create_field_index(
        self,
        collection_name: str,
        field_name: str,
        index_config: Union[models.KeywordIndexParams, models.IntegerIndexParams, models.TextIndexParams],
        force_recreate: bool = False
    ) -> bool:
        """Create index for a specific field."""
        try:
            # Check if index already exists (if not forcing recreate)
            if not force_recreate:
                try:
                    collection_info = self.client.get_collection(collection_name)
                    # Qdrant doesn't easily expose index info, so we try to create and handle conflicts
                except Exception:
                    pass

            # Create the index
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=index_config
                )
            )

            return True

        except Exception as e:
            # Index might already exist or there might be other issues
            error_msg = str(e).lower()
            if "already exists" in error_msg or "duplicate" in error_msg:
                logger.debug("Index already exists",
                           collection=collection_name, field=field_name)
                return True
            else:
                logger.warning("Failed to create index",
                             collection=collection_name,
                             field=field_name,
                             error=str(e))
                return False

    async def optimize_collection_settings(
        self,
        collection_name: str,
        expected_points: Optional[int] = None
    ) -> bool:
        """
        Optimize collection settings for metadata filtering performance.

        Args:
            collection_name: Collection to optimize
            expected_points: Expected number of points for optimization

        Returns:
            Success status
        """
        try:
            # Calculate optimal indexing threshold
            if expected_points:
                # Use lower threshold for smaller collections to enable indexing sooner
                indexing_threshold = min(10000, max(1000, expected_points // 10))
            else:
                # Default threshold optimized for metadata filtering
                indexing_threshold = 5000

            # Update optimizer configuration
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.update_collection(
                    collection_name=collection_name,
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=indexing_threshold
                    )
                )
            )

            logger.info("Collection settings optimized",
                       collection=collection_name,
                       indexing_threshold=indexing_threshold)
            return True

        except Exception as e:
            logger.error("Failed to optimize collection settings",
                        collection=collection_name, error=str(e))
            return False

    def get_indexed_collections(self) -> Set[str]:
        """Get set of collections with optimized indexes."""
        return self._indexed_collections.copy()

    def get_index_status(self, collection_name: str) -> Optional[Dict[str, bool]]:
        """Get index status for a collection."""
        return self._index_configurations.get(collection_name)


class QueryOptimizer:
    """
    Query optimization techniques to maintain sub-3ms response times.

    Implements various optimization strategies including query planning,
    filter ordering, and performance monitoring.
    """

    def __init__(self, target_response_time: float = 3.0):
        """Initialize query optimizer."""
        self.target_response_time = target_response_time
        self._query_stats: Dict[str, List[float]] = defaultdict(list)
        self._optimization_cache: Dict[str, Dict] = {}

        logger.info("Query optimizer initialized",
                   target_ms=target_response_time)

    def optimize_search_params(
        self,
        collection_name: str,
        query_type: str = "hybrid",
        limit: int = 10,
        has_filters: bool = False
    ) -> models.SearchParams:
        """
        Generate optimized search parameters for performance.

        Args:
            collection_name: Target collection
            query_type: Type of query (hybrid, dense, sparse)
            limit: Number of results requested
            has_filters: Whether the query includes filters

        Returns:
            Optimized SearchParams
        """
        cache_key = f"{collection_name}:{query_type}:{limit}:{has_filters}"

        if cache_key in self._optimization_cache:
            cached_params = self._optimization_cache[cache_key]
            logger.debug("Using cached search params", cache_key=cache_key[:20])
            return models.SearchParams(**cached_params)

        # Calculate optimal parameters based on query characteristics
        if has_filters:
            # With filters, use more conservative settings for accuracy
            ef = min(128, max(64, limit * 4))
            exact = limit <= 5  # Use exact search for small result sets
        else:
            # Without filters, optimize for speed
            ef = min(64, max(32, limit * 2))
            exact = False

        search_params = models.SearchParams(
            hnsw_ef=ef,
            exact=exact
        )

        # Cache the parameters
        self._optimization_cache[cache_key] = {
            "hnsw_ef": ef,
            "exact": exact
        }

        logger.debug("Generated optimized search params",
                    collection=collection_name,
                    ef=ef, exact=exact, has_filters=has_filters)

        return search_params

    def track_query_performance(
        self,
        query_type: str,
        response_time: float,
        result_count: int,
        has_filters: bool = False
    ) -> Dict:
        """
        Track query performance and provide optimization recommendations.

        Args:
            query_type: Type of query executed
            response_time: Response time in milliseconds
            result_count: Number of results returned
            has_filters: Whether filters were applied

        Returns:
            Performance analysis and recommendations
        """
        stats_key = f"{query_type}:{'filtered' if has_filters else 'unfiltered'}"
        self._query_stats[stats_key].append(response_time)

        # Keep only recent measurements
        if len(self._query_stats[stats_key]) > 100:
            self._query_stats[stats_key] = self._query_stats[stats_key][-100:]

        # Calculate statistics
        recent_times = self._query_stats[stats_key]
        avg_time = sum(recent_times) / len(recent_times)
        target_met = response_time <= self.target_response_time
        recent_target_rate = sum(1 for t in recent_times[-10:] if t <= self.target_response_time) / min(10, len(recent_times)) * 100

        analysis = {
            "current_response_time": response_time,
            "target_met": target_met,
            "avg_response_time": avg_time,
            "recent_target_rate": recent_target_rate,
            "sample_count": len(recent_times),
            "recommendations": []
        }

        # Generate recommendations
        if avg_time > self.target_response_time:
            if has_filters:
                analysis["recommendations"].append("Consider optimizing metadata indexes")
                analysis["recommendations"].append("Review filter selectivity")
            else:
                analysis["recommendations"].append("Consider adjusting HNSW parameters")
                analysis["recommendations"].append("Evaluate collection indexing threshold")

        if recent_target_rate < 90:
            analysis["recommendations"].append("Performance degradation detected - investigate recent changes")

        logger.debug("Query performance tracked",
                    query_type=query_type,
                    response_time=response_time,
                    target_met=target_met)

        return analysis

    def get_performance_summary(self) -> Dict:
        """Get overall query performance summary."""
        summary = {
            "target_response_time": self.target_response_time,
            "query_types": {}
        }

        for stats_key, times in self._query_stats.items():
            if not times:
                continue

            avg_time = sum(times) / len(times)
            target_met_rate = sum(1 for t in times if t <= self.target_response_time) / len(times) * 100

            summary["query_types"][stats_key] = {
                "sample_count": len(times),
                "avg_response_time": avg_time,
                "target_met_rate": target_met_rate,
                "latest_response_time": times[-1] if times else 0
            }

        return summary


class PerformanceTracker:
    """
    Monitor and ensure performance targets are met for metadata filtering.

    Tracks response times, cache performance, and provides alerts when
    performance degrades below target thresholds.
    """

    def __init__(self, target_response_time: float = 3.0):
        """Initialize performance tracker."""
        self.target_response_time = target_response_time
        self._measurements: List[Tuple[str, float, datetime]] = []
        self._alerts: List[Dict] = []

        logger.info("Performance tracker initialized", target_ms=target_response_time)

    def record_measurement(
        self,
        operation: str,
        response_time: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Record a performance measurement."""
        timestamp = datetime.now()
        self._measurements.append((operation, response_time, timestamp))

        # Keep only recent measurements (last 1000)
        if len(self._measurements) > 1000:
            self._measurements = self._measurements[-1000:]

        # Check for performance issues
        if response_time > self.target_response_time:
            self._create_performance_alert(operation, response_time, metadata)

        logger.debug("Performance measurement recorded",
                    operation=operation, response_time=response_time)

    def _create_performance_alert(
        self,
        operation: str,
        response_time: float,
        metadata: Optional[Dict] = None
    ) -> None:
        """Create performance alert when target is exceeded."""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "response_time": response_time,
            "target_response_time": self.target_response_time,
            "severity": "warning" if response_time < self.target_response_time * 2 else "critical",
            "metadata": metadata or {}
        }

        self._alerts.append(alert)

        # Keep only recent alerts (last 100)
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

        logger.warning("Performance target exceeded",
                      operation=operation,
                      response_time=response_time,
                      target=self.target_response_time)

    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self._measurements:
            return {"error": "No measurements available"}

        # Calculate statistics
        recent_measurements = [
            (op, time) for op, time, ts in self._measurements
            if datetime.now() - ts < timedelta(hours=1)
        ]

        if not recent_measurements:
            recent_measurements = self._measurements[-50:]  # Fallback to last 50

        times = [time for _, time in recent_measurements]
        operations = defaultdict(list)

        for op, time in recent_measurements:
            operations[op].append(time)

        # Overall statistics
        avg_time = sum(times) / len(times)
        target_met_rate = sum(1 for t in times if t <= self.target_response_time) / len(times) * 100
        sorted_times = sorted(times)
        p95_time = sorted_times[int(0.95 * len(sorted_times))] if sorted_times else 0
        p99_time = sorted_times[int(0.99 * len(sorted_times))] if sorted_times else 0

        # Per-operation statistics
        operation_stats = {}
        for op, op_times in operations.items():
            operation_stats[op] = {
                "count": len(op_times),
                "avg_time": sum(op_times) / len(op_times),
                "target_met_rate": sum(1 for t in op_times if t <= self.target_response_time) / len(op_times) * 100,
                "latest_time": op_times[-1] if op_times else 0
            }

        report = {
            "target_response_time": self.target_response_time,
            "measurement_period_hours": 1,
            "total_measurements": len(times),
            "overall_stats": {
                "avg_response_time": avg_time,
                "p95_response_time": p95_time,
                "p99_response_time": p99_time,
                "target_met_rate": target_met_rate,
                "performance_target_met": target_met_rate >= 95.0
            },
            "operation_stats": operation_stats,
            "recent_alerts": len([a for a in self._alerts if datetime.now() - datetime.fromisoformat(a["timestamp"]) < timedelta(hours=1)]),
            "critical_alerts": len([a for a in self._alerts if a["severity"] == "critical"])
        }

        return report

    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent performance alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_alerts = [
            alert for alert in self._alerts
            if datetime.fromisoformat(alert["timestamp"]) > cutoff
        ]

        return recent_alerts


# Export main classes for use in hybrid search engine
__all__ = [
    "FilterOptimizer",
    "MetadataIndexManager",
    "QueryOptimizer",
    "PerformanceTracker",
    "PerformanceMetrics"
]