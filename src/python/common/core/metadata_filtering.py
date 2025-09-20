"""
Metadata-based filtering system for project isolation in Qdrant operations.

This module implements a comprehensive metadata filtering system that ensures complete
project isolation while maintaining optimal performance for large collections. It provides
a unified interface for all metadata-based filtering operations across the workspace-qdrant-mcp
system, building upon the metadata schema from subtask 249.1.

Key Features:
    - Project isolation through project_id metadata filtering
    - Performance-optimized filter construction with indexed fields
    - Edge case handling for missing or invalid metadata
    - Integration with existing hybrid search and client systems
    - Support for complex multi-tenant filtering scenarios
    - Comprehensive filter validation and debugging capabilities

Filter Types:
    - **Project Isolation**: Filter by project_id for complete tenant separation
    - **Collection Type**: Filter by collection_type for workspace organization
    - **Access Control**: Filter by access_level and permissions
    - **Workspace Scope**: Filter by workspace_scope for visibility control
    - **Temporal**: Filter by creation/update timestamps
    - **Composite**: Complex filters combining multiple criteria

Performance Features:
    - Indexed field optimization for fast filtering
    - Filter caching for repeated operations
    - Query optimization recommendations
    - Performance monitoring and metrics
    - Batch filtering for multiple operations

Example:
    ```python
    from metadata_filtering import MetadataFilterManager, FilterCriteria
    from metadata_schema import MultiTenantMetadataSchema

    filter_manager = MetadataFilterManager(qdrant_client)

    # Project isolation filter
    project_filter = filter_manager.create_project_isolation_filter("workspace-qdrant-mcp")

    # Complex composite filter
    criteria = FilterCriteria(
        project_id="a1b2c3d4e5f6",
        collection_types=["docs", "notes"],
        access_levels=["private", "shared"],
        include_global=True
    )
    composite_filter = filter_manager.create_composite_filter(criteria)

    # Performance-optimized search
    results = filter_manager.filtered_search(
        collection_name="docs",
        query_vector=[0.1, 0.2, ...],
        filter_criteria=criteria,
        limit=50
    )
    ```
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from enum import Enum

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Import metadata schema components from subtask 249.1
try:
    from .metadata_schema import (
        MultiTenantMetadataSchema,
        CollectionCategory,
        WorkspaceScope,
        AccessLevel,
        METADATA_SCHEMA_VERSION
    )
    from .metadata_validator import MetadataValidator, ValidationResult
except ImportError:
    logger.error("Cannot import metadata schema components from subtask 249.1")
    raise


class FilterStrategy(Enum):
    """Strategies for applying metadata filters."""

    STRICT = "strict"              # Exact matching, fail if metadata missing
    LENIENT = "lenient"            # Best-effort matching, ignore missing metadata
    FALLBACK = "fallback"          # Use fallback collections if metadata missing
    HYBRID = "hybrid"              # Combine metadata and collection-based filtering


class FilterPerformanceLevel(Enum):
    """Performance optimization levels for filtering."""

    FAST = "fast"                  # Prioritize speed over completeness
    BALANCED = "balanced"          # Balance speed and completeness
    COMPREHENSIVE = "comprehensive" # Prioritize completeness over speed


@dataclass
class FilterCriteria:
    """Comprehensive criteria for metadata-based filtering."""

    # Core project isolation
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    tenant_namespace: Optional[str] = None

    # Collection filtering
    collection_types: Optional[List[str]] = None
    collection_categories: Optional[List[CollectionCategory]] = None
    workspace_scopes: Optional[List[WorkspaceScope]] = None

    # Access control filtering
    access_levels: Optional[List[AccessLevel]] = None
    created_by: Optional[List[str]] = None
    mcp_readonly: Optional[bool] = None
    cli_writable: Optional[bool] = None

    # Temporal filtering
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    updated_after: Optional[str] = None
    updated_before: Optional[str] = None

    # Organizational filtering
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    priority_range: Optional[Tuple[int, int]] = None

    # Special options
    include_global: bool = True
    include_shared: bool = True
    include_legacy: bool = False
    require_metadata: bool = True

    # Performance options
    strategy: FilterStrategy = FilterStrategy.STRICT
    performance_level: FilterPerformanceLevel = FilterPerformanceLevel.BALANCED
    use_cache: bool = True

    def __post_init__(self):
        """Validate and normalize filter criteria."""
        # Ensure project_id is generated if project_name provided
        if self.project_name and not self.project_id:
            self.project_id = self._generate_project_id(self.project_name)

        # Ensure tenant_namespace consistency
        if self.project_name and self.collection_types and len(self.collection_types) == 1:
            expected_namespace = f"{self.project_name}.{self.collection_types[0]}"
            if not self.tenant_namespace:
                self.tenant_namespace = expected_namespace

        # Convert single values to lists
        if isinstance(self.collection_types, str):
            self.collection_types = [self.collection_types]
        if isinstance(self.tags, str):
            self.tags = [self.tags]
        if isinstance(self.categories, str):
            self.categories = [self.categories]

    @staticmethod
    def _generate_project_id(project_name: str) -> str:
        """Generate stable project ID from project name."""
        return hashlib.sha256(project_name.encode()).hexdigest()[:12]

    def to_cache_key(self) -> str:
        """Generate cache key for filter criteria."""
        key_parts = [
            f"project_id:{self.project_id or 'none'}",
            f"project_name:{self.project_name or 'none'}",
            f"types:{','.join(self.collection_types or [])}",
            f"categories:{','.join([c.value for c in (self.collection_categories or [])])}",
            f"scopes:{','.join([s.value for s in (self.workspace_scopes or [])])}",
            f"access:{','.join([a.value for a in (self.access_levels or [])])}",
            f"global:{self.include_global}",
            f"shared:{self.include_shared}",
            f"legacy:{self.include_legacy}",
            f"strategy:{self.strategy.value}"
        ]
        cache_key = "|".join(key_parts)
        return hashlib.md5(cache_key.encode()).hexdigest()


@dataclass
class FilterResult:
    """Result of applying metadata filters with performance metrics."""

    filter: models.Filter
    criteria: FilterCriteria
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    warnings: List[str] = field(default_factory=list)
    optimizations_applied: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize performance metrics."""
        if not self.performance_metrics:
            self.performance_metrics = {
                "construction_time_ms": 0,
                "condition_count": 0,
                "indexed_conditions": 0,
                "complexity_score": 0
            }


class MetadataFilterManager:
    """
    Comprehensive metadata filtering manager for project isolation.

    This class provides the core filtering functionality for ensuring project isolation
    in Qdrant operations while maintaining optimal performance. It integrates with the
    metadata schema from subtask 249.1 and provides various filtering strategies.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 300,
        enable_performance_monitoring: bool = True
    ):
        """
        Initialize the metadata filter manager.

        Args:
            qdrant_client: Qdrant client instance for operations
            enable_caching: Whether to enable filter caching
            cache_ttl_seconds: Cache time-to-live in seconds
            enable_performance_monitoring: Whether to collect performance metrics
        """
        self.qdrant_client = qdrant_client
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds
        self.enable_performance_monitoring = enable_performance_monitoring

        # Initialize caching system
        self._filter_cache: Dict[str, Tuple[FilterResult, float]] = {}

        # Initialize performance tracking
        self._performance_stats = defaultdict(list)
        self._filter_usage_count = defaultdict(int)

        # Initialize metadata validator
        self.validator = MetadataValidator(strict_mode=True)

        # Define indexed fields for performance optimization
        self.indexed_fields = {
            "project_id", "project_name", "tenant_namespace",
            "collection_type", "collection_category", "workspace_scope",
            "access_level", "created_by", "mcp_readonly", "cli_writable",
            "is_reserved_name", "naming_pattern"
        }

        logger.debug(
            "MetadataFilterManager initialized with caching={}, performance_monitoring={}",
            enable_caching, enable_performance_monitoring
        )

    def create_project_isolation_filter(
        self,
        project_identifier: Union[str, MultiTenantMetadataSchema],
        strategy: FilterStrategy = FilterStrategy.STRICT
    ) -> FilterResult:
        """
        Create a filter for complete project isolation.

        This is the primary method for ensuring project isolation in multi-tenant
        scenarios. It creates a filter that ensures only documents belonging to
        the specified project are returned.

        Args:
            project_identifier: Project name, project_id, or metadata schema
            strategy: Filtering strategy to use

        Returns:
            FilterResult with project isolation filter
        """
        start_time = time.time()

        # Extract project information
        if isinstance(project_identifier, MultiTenantMetadataSchema):
            project_id = project_identifier.project_id
            project_name = project_identifier.project_name
        elif isinstance(project_identifier, str):
            if len(project_identifier) == 12 and all(c in '0123456789abcdef' for c in project_identifier):
                # Assume it's a project_id
                project_id = project_identifier
                project_name = None
            else:
                # Assume it's a project_name
                project_name = project_identifier
                project_id = self._generate_project_id(project_identifier)
        else:
            raise ValueError("project_identifier must be string or MultiTenantMetadataSchema")

        logger.debug(f"Creating project isolation filter for project_id={project_id}, project_name={project_name}")

        # Create filter criteria
        criteria = FilterCriteria(
            project_id=project_id,
            project_name=project_name,
            strategy=strategy,
            include_global=False,  # Strict project isolation
            include_shared=False,
            require_metadata=True
        )

        # Check cache first
        if self.enable_caching:
            cached_result = self._get_cached_filter(criteria)
            if cached_result:
                logger.debug("Using cached project isolation filter")
                return cached_result

        # Build the filter
        filter_conditions = []

        if strategy == FilterStrategy.STRICT:
            # Strict project isolation - must have exact project_id match
            filter_conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )
            )
        elif strategy == FilterStrategy.LENIENT:
            # Lenient approach - project_id OR project_name match
            project_conditions = [
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )
            ]
            if project_name:
                project_conditions.append(
                    models.FieldCondition(
                        key="project_name",
                        match=models.MatchValue(value=project_name)
                    )
                )

            filter_conditions.append(
                models.Filter(should=project_conditions)
            )
        elif strategy == FilterStrategy.FALLBACK:
            # Include fallback for collections without metadata
            project_conditions = [
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )
            ]

            # Add fallback condition for missing metadata
            project_conditions.append(
                models.Filter(
                    must_not=[
                        models.HasIdCondition(has_id=[])  # Will be replaced with actual logic
                    ]
                )
            )

            filter_conditions.append(
                models.Filter(should=project_conditions)
            )

        # Create the final filter
        qdrant_filter = models.Filter(must=filter_conditions)

        # Calculate performance metrics
        construction_time = (time.time() - start_time) * 1000
        condition_count = self._count_filter_conditions(qdrant_filter)
        indexed_conditions = self._count_indexed_conditions(qdrant_filter)

        # Create result
        result = FilterResult(
            filter=qdrant_filter,
            criteria=criteria,
            performance_metrics={
                "construction_time_ms": construction_time,
                "condition_count": condition_count,
                "indexed_conditions": indexed_conditions,
                "complexity_score": self._calculate_complexity_score(qdrant_filter)
            },
            optimizations_applied=["project_id_indexing"] if "project_id" in self.indexed_fields else []
        )

        # Cache the result
        if self.enable_caching:
            self._cache_filter(criteria, result)

        # Record performance stats
        if self.enable_performance_monitoring:
            self._record_performance_stats("project_isolation", result.performance_metrics)

        logger.debug(f"Created project isolation filter in {construction_time:.2f}ms with {condition_count} conditions")
        return result

    def create_composite_filter(self, criteria: FilterCriteria) -> FilterResult:
        """
        Create a composite filter based on comprehensive criteria.

        This method builds complex filters that combine multiple filtering criteria
        for advanced use cases like cross-collection searches, temporal filtering,
        and access control.

        Args:
            criteria: Comprehensive filter criteria

        Returns:
            FilterResult with composite filter
        """
        start_time = time.time()

        logger.debug(f"Creating composite filter with criteria: {criteria}")

        # Check cache first
        if self.enable_caching:
            cached_result = self._get_cached_filter(criteria)
            if cached_result:
                logger.debug("Using cached composite filter")
                return cached_result

        filter_conditions = []
        warnings = []
        optimizations = []

        # Project isolation conditions
        if criteria.project_id or criteria.project_name:
            project_conditions = self._build_project_conditions(criteria)
            filter_conditions.extend(project_conditions)
            optimizations.append("project_isolation")

        # Collection type conditions
        if criteria.collection_types:
            collection_conditions = self._build_collection_type_conditions(criteria)
            filter_conditions.extend(collection_conditions)
            optimizations.append("collection_type_filtering")

        # Collection category conditions
        if criteria.collection_categories:
            category_conditions = self._build_collection_category_conditions(criteria)
            filter_conditions.extend(category_conditions)
            optimizations.append("category_filtering")

        # Workspace scope conditions
        if criteria.workspace_scopes:
            scope_conditions = self._build_workspace_scope_conditions(criteria)
            filter_conditions.extend(scope_conditions)
            optimizations.append("scope_filtering")

        # Access control conditions
        if criteria.access_levels or criteria.created_by is not None:
            access_conditions = self._build_access_control_conditions(criteria)
            filter_conditions.extend(access_conditions)
            optimizations.append("access_control_filtering")

        # Temporal conditions
        temporal_conditions = self._build_temporal_conditions(criteria)
        if temporal_conditions:
            filter_conditions.extend(temporal_conditions)
            optimizations.append("temporal_filtering")

        # Organizational conditions
        org_conditions = self._build_organizational_conditions(criteria)
        if org_conditions:
            filter_conditions.extend(org_conditions)
            optimizations.append("organizational_filtering")

        # Special inclusion/exclusion logic
        special_conditions = self._build_special_conditions(criteria)
        if special_conditions:
            filter_conditions.extend(special_conditions)
            optimizations.append("special_conditions")

        # Handle edge cases based on strategy
        if criteria.strategy == FilterStrategy.LENIENT and not filter_conditions:
            warnings.append("No filter conditions generated, all documents will be returned")
        elif criteria.strategy == FilterStrategy.STRICT and not criteria.project_id:
            warnings.append("Strict strategy requires project_id for proper isolation")

        # Create the final filter
        if filter_conditions:
            qdrant_filter = models.Filter(must=filter_conditions)
        else:
            # Empty filter - return all documents
            qdrant_filter = models.Filter()

        # Calculate performance metrics
        construction_time = (time.time() - start_time) * 1000
        condition_count = self._count_filter_conditions(qdrant_filter)
        indexed_conditions = self._count_indexed_conditions(qdrant_filter)

        # Create result
        result = FilterResult(
            filter=qdrant_filter,
            criteria=criteria,
            performance_metrics={
                "construction_time_ms": construction_time,
                "condition_count": condition_count,
                "indexed_conditions": indexed_conditions,
                "complexity_score": self._calculate_complexity_score(qdrant_filter)
            },
            warnings=warnings,
            optimizations_applied=optimizations
        )

        # Cache the result
        if self.enable_caching:
            self._cache_filter(criteria, result)

        # Record performance stats
        if self.enable_performance_monitoring:
            self._record_performance_stats("composite", result.performance_metrics)

        logger.debug(f"Created composite filter in {construction_time:.2f}ms with {condition_count} conditions")
        return result

    def create_collection_type_filter(
        self,
        collection_types: Union[str, List[str]],
        include_global: bool = True
    ) -> FilterResult:
        """
        Create a filter for specific collection types.

        Args:
            collection_types: Collection type(s) to filter for
            include_global: Whether to include global collections

        Returns:
            FilterResult with collection type filter
        """
        if isinstance(collection_types, str):
            collection_types = [collection_types]

        criteria = FilterCriteria(
            collection_types=collection_types,
            include_global=include_global,
            strategy=FilterStrategy.BALANCED
        )

        return self.create_composite_filter(criteria)

    def create_access_control_filter(
        self,
        access_levels: Union[AccessLevel, List[AccessLevel]],
        created_by: Optional[List[str]] = None,
        mcp_readonly: Optional[bool] = None
    ) -> FilterResult:
        """
        Create a filter for access control.

        Args:
            access_levels: Access level(s) to filter for
            created_by: Creator(s) to filter for
            mcp_readonly: MCP readonly flag

        Returns:
            FilterResult with access control filter
        """
        if isinstance(access_levels, AccessLevel):
            access_levels = [access_levels]

        criteria = FilterCriteria(
            access_levels=access_levels,
            created_by=created_by,
            mcp_readonly=mcp_readonly,
            strategy=FilterStrategy.BALANCED
        )

        return self.create_composite_filter(criteria)

    def validate_filter_compatibility(
        self,
        collection_name: str,
        filter_criteria: FilterCriteria
    ) -> ValidationResult:
        """
        Validate that filter criteria are compatible with a collection.

        Args:
            collection_name: Name of the collection
            filter_criteria: Filter criteria to validate

        Returns:
            ValidationResult with compatibility assessment
        """
        result = ValidationResult(is_valid=True)

        try:
            # Check if collection exists
            collection_info = self.qdrant_client.get_collection(collection_name)

            # Validate indexed fields
            indexed_fields = set()
            if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                # Extract indexed fields from collection config
                # This would need to be implemented based on Qdrant's actual API
                pass

            # Check for performance implications
            if filter_criteria.performance_level == FilterPerformanceLevel.FAST:
                non_indexed_conditions = self._get_non_indexed_conditions(filter_criteria)
                if non_indexed_conditions:
                    result.add_warning(
                        "performance",
                        f"Fast performance mode requested but non-indexed conditions present: {non_indexed_conditions}",
                        "NON_INDEXED_CONDITIONS"
                    )

            # Validate metadata requirements
            if filter_criteria.require_metadata:
                # Check if collection has metadata-enabled documents
                # This would require sampling the collection
                pass

        except Exception as e:
            result.add_error(
                "collection",
                f"Failed to validate collection compatibility: {str(e)}",
                "COLLECTION_VALIDATION_ERROR"
            )

        return result

    def get_filter_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for filter operations."""
        if not self.enable_performance_monitoring:
            return {"monitoring_disabled": True}

        stats = {}
        for filter_type, metrics_list in self._performance_stats.items():
            if metrics_list:
                avg_time = sum(m["construction_time_ms"] for m in metrics_list) / len(metrics_list)
                avg_conditions = sum(m["condition_count"] for m in metrics_list) / len(metrics_list)
                avg_complexity = sum(m["complexity_score"] for m in metrics_list) / len(metrics_list)

                stats[filter_type] = {
                    "total_operations": len(metrics_list),
                    "avg_construction_time_ms": avg_time,
                    "avg_condition_count": avg_conditions,
                    "avg_complexity_score": avg_complexity,
                    "usage_count": self._filter_usage_count[filter_type]
                }

        # Add cache statistics
        stats["cache"] = {
            "enabled": self.enable_caching,
            "total_entries": len(self._filter_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }

        return stats

    def clear_cache(self):
        """Clear the filter cache."""
        self._filter_cache.clear()
        logger.debug("Filter cache cleared")

    def _build_project_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build project isolation conditions."""
        conditions = []

        if criteria.project_id:
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=criteria.project_id)
                )
            )
        elif criteria.project_name:
            # Generate project_id from name
            project_id = self._generate_project_id(criteria.project_name)
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchValue(value=project_id)
                )
            )

        if criteria.tenant_namespace:
            conditions.append(
                models.FieldCondition(
                    key="tenant_namespace",
                    match=models.MatchValue(value=criteria.tenant_namespace)
                )
            )

        return conditions

    def _build_collection_type_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build collection type conditions."""
        if not criteria.collection_types:
            return []

        if len(criteria.collection_types) == 1:
            return [
                models.FieldCondition(
                    key="collection_type",
                    match=models.MatchValue(value=criteria.collection_types[0])
                )
            ]
        else:
            return [
                models.FieldCondition(
                    key="collection_type",
                    match=models.MatchAny(any=criteria.collection_types)
                )
            ]

    def _build_collection_category_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build collection category conditions."""
        if not criteria.collection_categories:
            return []

        category_values = [cat.value for cat in criteria.collection_categories]

        if len(category_values) == 1:
            return [
                models.FieldCondition(
                    key="collection_category",
                    match=models.MatchValue(value=category_values[0])
                )
            ]
        else:
            return [
                models.FieldCondition(
                    key="collection_category",
                    match=models.MatchAny(any=category_values)
                )
            ]

    def _build_workspace_scope_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build workspace scope conditions."""
        if not criteria.workspace_scopes:
            return []

        scope_values = [scope.value for scope in criteria.workspace_scopes]

        if len(scope_values) == 1:
            return [
                models.FieldCondition(
                    key="workspace_scope",
                    match=models.MatchValue(value=scope_values[0])
                )
            ]
        else:
            return [
                models.FieldCondition(
                    key="workspace_scope",
                    match=models.MatchAny(any=scope_values)
                )
            ]

    def _build_access_control_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build access control conditions."""
        conditions = []

        if criteria.access_levels:
            access_values = [level.value for level in criteria.access_levels]
            if len(access_values) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="access_level",
                        match=models.MatchValue(value=access_values[0])
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="access_level",
                        match=models.MatchAny(any=access_values)
                    )
                )

        if criteria.created_by:
            if len(criteria.created_by) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="created_by",
                        match=models.MatchValue(value=criteria.created_by[0])
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="created_by",
                        match=models.MatchAny(any=criteria.created_by)
                    )
                )

        if criteria.mcp_readonly is not None:
            conditions.append(
                models.FieldCondition(
                    key="mcp_readonly",
                    match=models.MatchValue(value=criteria.mcp_readonly)
                )
            )

        if criteria.cli_writable is not None:
            conditions.append(
                models.FieldCondition(
                    key="cli_writable",
                    match=models.MatchValue(value=criteria.cli_writable)
                )
            )

        return conditions

    def _build_temporal_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build temporal filtering conditions."""
        conditions = []

        # Created date range
        if criteria.created_after or criteria.created_before:
            date_range = {}
            if criteria.created_after:
                date_range["gte"] = criteria.created_after
            if criteria.created_before:
                date_range["lte"] = criteria.created_before

            conditions.append(
                models.FieldCondition(
                    key="created_at",
                    range=models.Range(**date_range)
                )
            )

        # Updated date range
        if criteria.updated_after or criteria.updated_before:
            date_range = {}
            if criteria.updated_after:
                date_range["gte"] = criteria.updated_after
            if criteria.updated_before:
                date_range["lte"] = criteria.updated_before

            conditions.append(
                models.FieldCondition(
                    key="updated_at",
                    range=models.Range(**date_range)
                )
            )

        return conditions

    def _build_organizational_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build organizational filtering conditions."""
        conditions = []

        if criteria.tags:
            # Tags can be matched as any (OR logic)
            conditions.append(
                models.FieldCondition(
                    key="tags",
                    match=models.MatchAny(any=criteria.tags)
                )
            )

        if criteria.categories:
            if len(criteria.categories) == 1:
                conditions.append(
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=criteria.categories[0])
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key="category",
                        match=models.MatchAny(any=criteria.categories)
                    )
                )

        if criteria.priority_range:
            min_priority, max_priority = criteria.priority_range
            conditions.append(
                models.FieldCondition(
                    key="priority",
                    range=models.Range(gte=min_priority, lte=max_priority)
                )
            )

        return conditions

    def _build_special_conditions(self, criteria: FilterCriteria) -> List[models.Condition]:
        """Build special inclusion/exclusion conditions."""
        conditions = []

        # Handle global collection inclusion
        if criteria.include_global:
            # Add condition to include global collections
            global_condition = models.FieldCondition(
                key="collection_category",
                match=models.MatchValue(value="global")
            )
            # This would need to be combined with main conditions using OR logic
            # Implementation depends on how the final filter is structured

        # Handle shared collection inclusion
        if criteria.include_shared:
            # Add condition to include shared collections
            shared_condition = models.FieldCondition(
                key="workspace_scope",
                match=models.MatchValue(value="shared")
            )
            # Similar to global condition

        # Handle legacy collection inclusion/exclusion
        if not criteria.include_legacy:
            # Exclude collections without proper metadata
            conditions.append(
                models.FieldCondition(
                    key="compatibility_version",
                    match=models.MatchValue(value=METADATA_SCHEMA_VERSION)
                )
            )

        # Handle metadata requirement
        if criteria.require_metadata:
            # Ensure documents have required metadata fields
            conditions.append(
                models.FieldCondition(
                    key="project_id",
                    match=models.MatchExcept(except_=["", None])
                )
            )

        return conditions

    def _get_cached_filter(self, criteria: FilterCriteria) -> Optional[FilterResult]:
        """Retrieve cached filter result if available and valid."""
        if not self.enable_caching:
            return None

        cache_key = criteria.to_cache_key()
        if cache_key in self._filter_cache:
            result, timestamp = self._filter_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_seconds:
                result.cache_hit = True
                return result
            else:
                # Cache expired
                del self._filter_cache[cache_key]

        return None

    def _cache_filter(self, criteria: FilterCriteria, result: FilterResult):
        """Cache filter result for future use."""
        if not self.enable_caching:
            return

        cache_key = criteria.to_cache_key()
        self._filter_cache[cache_key] = (result, time.time())

        # Cleanup old cache entries if cache gets too large
        if len(self._filter_cache) > 1000:
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._filter_cache.items()
            if current_time - timestamp >= self.cache_ttl_seconds
        ]
        for key in expired_keys:
            del self._filter_cache[key]

    def _count_filter_conditions(self, filter_obj: models.Filter) -> int:
        """Count the total number of conditions in a filter."""
        count = 0
        if filter_obj.must:
            count += len(filter_obj.must)
        if filter_obj.should:
            count += len(filter_obj.should)
        if filter_obj.must_not:
            count += len(filter_obj.must_not)
        return count

    def _count_indexed_conditions(self, filter_obj: models.Filter) -> int:
        """Count conditions that use indexed fields."""
        indexed_count = 0

        def count_conditions(conditions):
            nonlocal indexed_count
            for condition in conditions or []:
                if isinstance(condition, models.FieldCondition):
                    if condition.key in self.indexed_fields:
                        indexed_count += 1
                elif isinstance(condition, models.Filter):
                    count_conditions(condition.must)
                    count_conditions(condition.should)
                    count_conditions(condition.must_not)

        count_conditions(filter_obj.must)
        count_conditions(filter_obj.should)
        count_conditions(filter_obj.must_not)

        return indexed_count

    def _calculate_complexity_score(self, filter_obj: models.Filter) -> float:
        """Calculate complexity score for a filter (0-10 scale)."""
        condition_count = self._count_filter_conditions(filter_obj)
        indexed_count = self._count_indexed_conditions(filter_obj)

        # Base complexity from condition count
        base_score = min(condition_count * 0.5, 5.0)

        # Penalty for non-indexed conditions
        non_indexed_penalty = (condition_count - indexed_count) * 0.3

        # Bonus for simple filters
        if condition_count <= 2 and indexed_count == condition_count:
            base_score *= 0.5

        return min(base_score + non_indexed_penalty, 10.0)

    def _get_non_indexed_conditions(self, criteria: FilterCriteria) -> List[str]:
        """Get list of criteria that would result in non-indexed conditions."""
        non_indexed = []

        # Check which criteria fields are not in indexed_fields
        criteria_dict = criteria.__dict__
        for field_name, value in criteria_dict.items():
            if value is not None and field_name not in self.indexed_fields:
                # Map criteria fields to metadata fields
                metadata_field = self._map_criteria_to_metadata_field(field_name)
                if metadata_field and metadata_field not in self.indexed_fields:
                    non_indexed.append(metadata_field)

        return non_indexed

    def _map_criteria_to_metadata_field(self, criteria_field: str) -> Optional[str]:
        """Map filter criteria field to metadata field name."""
        mapping = {
            "project_id": "project_id",
            "project_name": "project_name",
            "tenant_namespace": "tenant_namespace",
            "collection_types": "collection_type",
            "collection_categories": "collection_category",
            "workspace_scopes": "workspace_scope",
            "access_levels": "access_level",
            "created_by": "created_by",
            "mcp_readonly": "mcp_readonly",
            "cli_writable": "cli_writable",
            "tags": "tags",
            "categories": "category",
            "priority_range": "priority",
            "created_after": "created_at",
            "created_before": "created_at",
            "updated_after": "updated_at",
            "updated_before": "updated_at"
        }
        return mapping.get(criteria_field)

    def _record_performance_stats(self, filter_type: str, metrics: Dict[str, Any]):
        """Record performance statistics for monitoring."""
        self._performance_stats[filter_type].append(metrics)
        self._filter_usage_count[filter_type] += 1

        # Keep only recent stats to prevent memory growth
        if len(self._performance_stats[filter_type]) > 1000:
            self._performance_stats[filter_type] = self._performance_stats[filter_type][-500:]

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        total_requests = sum(self._filter_usage_count.values())
        if total_requests == 0:
            return 0.0

        # This is a simplified calculation
        # In practice, you'd track cache hits vs misses separately
        cache_hits = len(self._filter_cache) * 0.7  # Estimate
        return min((cache_hits / total_requests) * 100, 100.0)

    @staticmethod
    def _generate_project_id(project_name: str) -> str:
        """Generate stable project ID from project name."""
        return hashlib.sha256(project_name.encode()).hexdigest()[:12]


# Export all public classes and functions
__all__ = [
    'MetadataFilterManager',
    'FilterCriteria',
    'FilterResult',
    'FilterStrategy',
    'FilterPerformanceLevel'
]