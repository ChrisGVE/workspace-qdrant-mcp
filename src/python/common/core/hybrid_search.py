"""
Advanced hybrid search implementation with multiple fusion strategies.

This module provides a sophisticated hybrid search system that combines dense semantic
vector search with sparse keyword-based search to achieve optimal retrieval performance.
It implements multiple fusion algorithms including Reciprocal Rank Fusion (RRF),
weighted sum, and maximum score fusion for different use cases.

Key Features:
    - Multiple fusion strategies (RRF, weighted sum, max score)
    - Configurable weights for dense and sparse components
    - Detailed fusion analysis and explanation capabilities
    - Benchmark tools for comparing fusion methods
    - Production-ready error handling and logging
    - Optimal result ranking across multiple search modalities

Fusion Algorithms:
    - **RRF (Reciprocal Rank Fusion)**: Industry-standard fusion using reciprocal ranks
    - **Weighted Sum**: Score normalization with configurable weights
    - **Max Score**: Takes maximum score across search modalities

Performance Characteristics:
    - RRF: Best for balanced precision/recall, handles score distribution differences
    - Weighted Sum: Good when score ranges are similar, allows fine-tuned control
    - Max Score: Emphasizes best matches, good for high-precision scenarios

Example:
    ```python
    from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine
    from qdrant_client import QdrantClient
    from .ssl_config import suppress_qdrant_ssl_warnings

    with suppress_qdrant_ssl_warnings():
        client = QdrantClient("http://localhost:6333")
    engine = HybridSearchEngine(client)

    # Hybrid search with RRF fusion
    results = await engine.hybrid_search(
        collection_name="documents",
        query_embeddings={
            "dense": [0.1, 0.2, ...],  # 384-dim semantic vector
            "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
        },
        limit=10,
        fusion_method="rrf",
        dense_weight=1.0,
        sparse_weight=1.0
    )

    # Analyze fusion process
    ranker = RRFFusionRanker()
    explanation = ranker.explain_fusion(dense_results, sparse_results)
    ```

Task 215: Migrated to unified logging system for MCP stdio compliance.
Task 233.1: Enhanced for multi-tenant metadata-based filtering with project isolation.
"""

# Task 215: Replace direct logging import with unified logging system
# import logging  # MIGRATED to unified system
from collections import defaultdict
from typing import Optional, Union, Dict, List
import time

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Task 222: Import loguru-based logging system
from loguru import logger

from .sparse_vectors import create_named_sparse_vector
from .multitenant_collections import (
    ProjectIsolationManager,
    WorkspaceCollectionRegistry,
    ProjectMetadata
)
from .metadata_optimization import (
    FilterOptimizer,
    MetadataIndexManager,
    QueryOptimizer,
    PerformanceTracker
)
from .performance_monitoring import (
    MetadataFilteringPerformanceMonitor,
    PerformanceBaseline
)
# Task 249.3: Import new comprehensive metadata filtering system
from .metadata_filtering import (
    MetadataFilterManager,
    FilterCriteria,
    FilterStrategy,
    FilterPerformanceLevel
)
from collections import defaultdict
from dataclasses import dataclass

# Task 215: Use unified logging system instead of logging.getLogger(__name__)
# logger imported from loguru


@dataclass
class TenantAwareResult:
    """Result container with tenant-aware metadata for multi-tenant result aggregation."""

    id: str
    score: float
    payload: dict
    collection: str
    search_type: str
    tenant_metadata: dict = None
    project_context: dict = None
    deduplication_key: str = None

    def __post_init__(self):
        """Post-init processing for tenant-aware results."""
        if self.tenant_metadata is None:
            self.tenant_metadata = {}

        if self.project_context is None:
            self.project_context = {}

        # Generate deduplication key based on content hash or document identifier
        if self.deduplication_key is None:
            content_hash = self.payload.get("content_hash")
            file_path = self.payload.get("file_path")
            doc_id = self.payload.get("document_id")

            # Use content hash if available, fallback to file_path or doc_id
            self.deduplication_key = content_hash or file_path or doc_id or self.id


class TenantAwareResultDeduplicator:
    """
    Handles deduplication of search results across tenant boundaries.

    This class implements sophisticated deduplication logic that considers:
    - Content-based deduplication using content hashes
    - Tenant isolation requirements
    - Score aggregation across duplicate instances
    - Metadata preservation from the highest-scoring instance

    Task 233.5: Added for multi-tenant result deduplication.
    """

    def __init__(self, preserve_tenant_isolation: bool = True):
        """
        Initialize deduplicator with tenant isolation settings.

        Args:
            preserve_tenant_isolation: If True, maintains separate results for different tenants
        """
        self.preserve_tenant_isolation = preserve_tenant_isolation

    def deduplicate_results(
        self,
        results: List[TenantAwareResult],
        aggregation_method: str = "max_score"
    ) -> List[TenantAwareResult]:
        """
        Deduplicate results while preserving tenant isolation.

        Args:
            results: List of tenant-aware search results
            aggregation_method: How to aggregate scores ("max_score", "avg_score", "sum_score")

        Returns:
            Deduplicated list of results with proper tenant isolation
        """
        if not results:
            return []

        logger.debug(
            "Starting result deduplication",
            total_results=len(results),
            preserve_isolation=self.preserve_tenant_isolation,
            aggregation_method=aggregation_method
        )

        # Group results by deduplication key
        result_groups = defaultdict(list)
        for result in results:
            group_key = self._get_group_key(result)
            result_groups[group_key].append(result)

        deduplicated_results = []

        for group_key, group_results in result_groups.items():
            if len(group_results) == 1:
                # No duplicates, keep as-is
                deduplicated_results.append(group_results[0])
            else:
                # Handle duplicates
                deduplicated_result = self._aggregate_duplicate_results(
                    group_results, aggregation_method
                )
                deduplicated_results.append(deduplicated_result)

        # Sort by score (descending)
        deduplicated_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(
            "Result deduplication completed",
            original_count=len(results),
            deduplicated_count=len(deduplicated_results),
            duplicates_found=len(results) - len(deduplicated_results)
        )

        return deduplicated_results

    def _get_group_key(self, result: TenantAwareResult) -> str:
        """
        Generate grouping key for deduplication.

        Args:
            result: Result to generate key for

        Returns:
            String key for grouping
        """
        if self.preserve_tenant_isolation:
            # Include tenant information in the key to maintain isolation
            project_name = result.project_context.get("project_name", "")
            tenant_namespace = result.tenant_metadata.get("tenant_namespace", "")
            return f"{result.deduplication_key}:{project_name}:{tenant_namespace}"
        else:
            # Global deduplication across all tenants
            return result.deduplication_key

    def _aggregate_duplicate_results(
        self,
        group_results: List[TenantAwareResult],
        aggregation_method: str
    ) -> TenantAwareResult:
        """
        Aggregate multiple duplicate results into a single result.

        Args:
            group_results: List of duplicate results to aggregate
            aggregation_method: How to aggregate scores

        Returns:
            Single aggregated result
        """
        if not group_results:
            return None

        # Sort by score to get the best result first
        group_results.sort(key=lambda x: x.score, reverse=True)
        best_result = group_results[0]

        # Calculate aggregated score
        scores = [r.score for r in group_results]
        if aggregation_method == "max_score":
            aggregated_score = max(scores)
        elif aggregation_method == "avg_score":
            aggregated_score = sum(scores) / len(scores)
        elif aggregation_method == "sum_score":
            aggregated_score = sum(scores)
        else:
            logger.warning(f"Unknown aggregation method {aggregation_method}, using max_score")
            aggregated_score = max(scores)

        # Aggregate metadata from all instances
        collection_sources = list(set(r.collection for r in group_results))
        search_types = list(set(r.search_type for r in group_results))

        # Create aggregated result based on best result
        aggregated_result = TenantAwareResult(
            id=best_result.id,
            score=aggregated_score,
            payload=best_result.payload.copy(),
            collection=best_result.collection,
            search_type=best_result.search_type,
            tenant_metadata=best_result.tenant_metadata.copy(),
            project_context=best_result.project_context.copy(),
            deduplication_key=best_result.deduplication_key
        )

        # Add aggregation metadata
        aggregated_result.payload["deduplication_info"] = {
            "duplicate_count": len(group_results),
            "score_aggregation": aggregation_method,
            "original_scores": scores,
            "collection_sources": collection_sources,
            "search_types": search_types
        }

        logger.debug(
            "Aggregated duplicate results",
            deduplication_key=best_result.deduplication_key,
            duplicate_count=len(group_results),
            original_scores=scores,
            aggregated_score=aggregated_score
        )

        return aggregated_result


class MultiTenantResultAggregator:
    """
    Advanced result aggregator for multi-tenant workspace collections.

    This class handles sophisticated result aggregation across multiple tenant contexts,
    ensuring proper isolation, deduplication, and ranking while maintaining API consistency.

    Key Features:
    - Tenant-aware result deduplication
    - Cross-collection score normalization
    - Metadata consistency enforcement
    - Performance-optimized aggregation algorithms

    Task 233.5: Added for multi-tenant search result aggregation.
    """

    def __init__(
        self,
        preserve_tenant_isolation: bool = True,
        enable_score_normalization: bool = True,
        default_aggregation_method: str = "max_score"
    ):
        """
        Initialize multi-tenant result aggregator.

        Args:
            preserve_tenant_isolation: Whether to maintain tenant isolation in results
            enable_score_normalization: Whether to normalize scores across collections
            default_aggregation_method: Default method for score aggregation
        """
        self.preserve_tenant_isolation = preserve_tenant_isolation
        self.enable_score_normalization = enable_score_normalization
        self.default_aggregation_method = default_aggregation_method
        self.deduplicator = TenantAwareResultDeduplicator(preserve_tenant_isolation)

        logger.debug(
            "Initialized MultiTenantResultAggregator",
            preserve_isolation=preserve_tenant_isolation,
            score_normalization=enable_score_normalization,
            aggregation_method=default_aggregation_method
        )

    def aggregate_multi_collection_results(
        self,
        collection_results: Dict[str, List],
        project_contexts: Dict[str, dict] = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        aggregation_method: str = None
    ) -> Dict:
        """
        Aggregate search results from multiple collections with tenant isolation.

        Args:
            collection_results: Dict mapping collection names to their search results
            project_contexts: Optional dict mapping collection names to project contexts
            limit: Maximum number of results to return
            score_threshold: Minimum score threshold for inclusion
            aggregation_method: Score aggregation method (overrides default)

        Returns:
            Aggregated results dict with consistent API format
        """
        aggregation_method = aggregation_method or self.default_aggregation_method
        project_contexts = project_contexts or {}

        logger.info(
            "Starting multi-collection result aggregation",
            collection_count=len(collection_results),
            total_raw_results=sum(len(results) for results in collection_results.values()),
            limit=limit,
            score_threshold=score_threshold,
            aggregation_method=aggregation_method
        )

        # Convert raw results to tenant-aware results
        tenant_aware_results = []

        for collection_name, results in collection_results.items():
            project_context = project_contexts.get(collection_name, {})

            for result in results:
                # Extract tenant metadata from result payload
                payload = getattr(result, 'payload', {}) or {}

                tenant_metadata = {
                    "project_name": payload.get("project_name"),
                    "collection_type": payload.get("collection_type"),
                    "workspace_scope": payload.get("workspace_scope"),
                    "tenant_namespace": payload.get("tenant_namespace"),
                    "access_level": payload.get("access_level")
                }

                # Create tenant-aware result
                tenant_result = TenantAwareResult(
                    id=getattr(result, 'id', ''),
                    score=getattr(result, 'score', 0.0),
                    payload=payload,
                    collection=collection_name,
                    search_type=getattr(result, 'search_type', 'unknown'),
                    tenant_metadata=tenant_metadata,
                    project_context=project_context
                )

                # Apply score threshold
                if tenant_result.score >= score_threshold:
                    tenant_aware_results.append(tenant_result)

        # Normalize scores across collections if enabled
        if self.enable_score_normalization:
            tenant_aware_results = self._normalize_cross_collection_scores(
                tenant_aware_results, collection_results.keys()
            )

        # Deduplicate results
        deduplicated_results = self.deduplicator.deduplicate_results(
            tenant_aware_results, aggregation_method
        )

        # Apply final limit
        final_results = deduplicated_results[:limit]

        # Convert back to API format
        api_results = self._convert_to_api_format(final_results)

        # Build response
        aggregated_response = {
            "total_results": len(api_results),
            "results": api_results,
            "aggregation_metadata": {
                "collection_count": len(collection_results),
                "raw_result_count": sum(len(results) for results in collection_results.values()),
                "post_threshold_count": len(tenant_aware_results),
                "post_deduplication_count": len(deduplicated_results),
                "final_count": len(final_results),
                "score_normalization_enabled": self.enable_score_normalization,
                "tenant_isolation_preserved": self.preserve_tenant_isolation,
                "aggregation_method": aggregation_method,
                "score_threshold": score_threshold
            }
        }

        logger.info(
            "Multi-collection result aggregation completed",
            raw_results=sum(len(results) for results in collection_results.values()),
            post_threshold=len(tenant_aware_results),
            post_deduplication=len(deduplicated_results),
            final_results=len(final_results)
        )

        return aggregated_response

    def _normalize_cross_collection_scores(
        self,
        results: List[TenantAwareResult],
        collection_names: List[str]
    ) -> List[TenantAwareResult]:
        """
        Normalize scores across different collections for fair comparison.

        Args:
            results: List of results to normalize
            collection_names: Names of collections being aggregated

        Returns:
            List of results with normalized scores
        """
        if not results:
            return results

        # Group results by collection
        collection_groups = defaultdict(list)
        for result in results:
            collection_groups[result.collection].append(result)

        # Calculate normalization factors for each collection
        normalization_factors = {}

        for collection_name, collection_results in collection_groups.items():
            if not collection_results:
                continue

            scores = [r.score for r in collection_results]
            max_score = max(scores)
            min_score = min(scores)

            # Use min-max normalization to [0, 1] range
            if max_score > min_score:
                normalization_factors[collection_name] = {
                    'min': min_score,
                    'range': max_score - min_score
                }
            else:
                # All scores are the same
                normalization_factors[collection_name] = {
                    'min': min_score,
                    'range': 1.0  # Avoid division by zero
                }

        # Apply normalization
        normalized_results = []
        for result in results:
            factor = normalization_factors.get(result.collection)
            if factor:
                normalized_score = (result.score - factor['min']) / factor['range']

                # Create new result with normalized score
                normalized_result = TenantAwareResult(
                    id=result.id,
                    score=normalized_score,
                    payload=result.payload.copy(),
                    collection=result.collection,
                    search_type=result.search_type,
                    tenant_metadata=result.tenant_metadata.copy(),
                    project_context=result.project_context.copy(),
                    deduplication_key=result.deduplication_key
                )

                # Add normalization info to payload
                normalized_result.payload["score_normalization"] = {
                    "original_score": result.score,
                    "normalized_score": normalized_score,
                    "collection_min": factor['min'],
                    "collection_range": factor['range']
                }

                normalized_results.append(normalized_result)
            else:
                normalized_results.append(result)

        logger.debug(
            "Cross-collection score normalization completed",
            collection_count=len(collection_groups),
            result_count=len(normalized_results)
        )

        return normalized_results

    def _convert_to_api_format(self, results: List[TenantAwareResult]) -> List[Dict]:
        """
        Convert tenant-aware results to standard API format.

        Args:
            results: List of tenant-aware results

        Returns:
            List of results in standard API dict format
        """
        api_results = []

        for result in results:
            api_result = {
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "collection": result.collection,
                "search_type": result.search_type
            }

            # Add tenant metadata if present
            if result.tenant_metadata:
                api_result["tenant_metadata"] = result.tenant_metadata

            # Add project context if present
            if result.project_context:
                api_result["project_context"] = result.project_context

            api_results.append(api_result)

        return api_results


class RRFFusionRanker:
    """
    Advanced Reciprocal Rank Fusion (RRF) implementation for multi-modal search fusion.

    Implements the industry-standard RRF algorithm for combining rankings from multiple
    retrieval systems. RRF provides a robust method for fusion that doesn't depend on
    score magnitudes or distributions, making it ideal for combining heterogeneous
    search results like dense semantic and sparse keyword vectors.

    The RRF formula: RRF(d) = Σ(1 / (k + r(d)))
    Where:
        - d is a document
        - k is a small constant (typically 60)
        - r(d) is the rank of document d in a particular ranking

    This implementation provides detailed fusion analysis, configurable parameters,
    and comprehensive logging for production monitoring and debugging.

    Task 215: Enhanced with unified logging system for better observability.
    """

    def __init__(self, k: int = 60, boost_weights: Optional[dict] = None) -> None:
        """Initialize RRF ranker with fusion parameters.

        Args:
            k: RRF constant parameter (default: 60, standard in literature)
            boost_weights: Optional weights for boosting specific result types
        """
        self.k = k
        self.boost_weights = boost_weights or {}
        logger.debug("Initialized RRF ranker", k=k, boost_weights=boost_weights)

    def fuse(
        self, dense_results: list, sparse_results: list, weights: Optional[dict] = None
    ) -> list:
        """Fuse dense and sparse search results using RRF algorithm.

        Args:
            dense_results: Results from dense (semantic) vector search
            sparse_results: Results from sparse (keyword) vector search
            weights: Optional weights for dense/sparse results

        Returns:
            List of fused results sorted by RRF score
        """
        weights = weights or {"dense": 1.0, "sparse": 1.0}
        logger.debug(
            "Starting RRF fusion",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            weights=weights,
        )

        # Create RRF scores for all documents
        rrf_scores = defaultdict(float)

        # Process dense results
        for rank, result in enumerate(dense_results):
            doc_id = result.id
            rrf_score = weights.get("dense", 1.0) / (self.k + rank + 1)
            rrf_scores[doc_id] += rrf_score

            logger.debug(
                "Dense result RRF score",
                doc_id=doc_id,
                rank=rank + 1,
                rrf_score=rrf_score,
                total_score=rrf_scores[doc_id],
            )

        # Process sparse results
        for rank, result in enumerate(sparse_results):
            doc_id = result.id
            rrf_score = weights.get("sparse", 1.0) / (self.k + rank + 1)
            rrf_scores[doc_id] += rrf_score

            logger.debug(
                "Sparse result RRF score",
                doc_id=doc_id,
                rank=rank + 1,
                rrf_score=rrf_score,
                total_score=rrf_scores[doc_id],
            )

        # Apply boost weights if configured
        if self.boost_weights:
            for doc_id in rrf_scores.keys():
                # Apply boosts based on document characteristics (can be extended)
                for boost_type, boost_value in self.boost_weights.items():
                    logger.debug(
                        "Applying boost weight",
                        doc_id=doc_id,
                        boost_type=boost_type,
                        boost_value=boost_value,
                    )
                    rrf_scores[doc_id] *= boost_value

        # Create document ID to result mapping for final sorting
        all_results = {result.id: result for result in dense_results + sparse_results}

        # Sort by RRF score (descending)
        sorted_results = sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Build final result list with RRF scores
        fused_results = []
        for doc_id, rrf_score in sorted_results:
            if doc_id in all_results:
                result = all_results[doc_id]
                # Add RRF score to result metadata
                if hasattr(result, "payload"):
                    result.payload = result.payload or {}
                    result.payload["rrf_score"] = rrf_score
                fused_results.append(result)

        logger.info(
            "RRF fusion completed",
            input_dense=len(dense_results),
            input_sparse=len(sparse_results),
            output_count=len(fused_results),
            top_score=sorted_results[0][1] if sorted_results else 0,
        )

        return fused_results

    def explain_fusion(
        self, dense_results: list, sparse_results: list, top_k: int = 5
    ) -> dict:
        """Provide detailed explanation of fusion process.

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            top_k: Number of top results to explain

        Returns:
            Detailed fusion analysis dictionary
        """
        logger.debug("Generating fusion explanation", top_k=top_k)

        explanation = {
            "algorithm": "Reciprocal Rank Fusion (RRF)",
            "parameters": {"k": self.k, "boost_weights": self.boost_weights},
            "input_stats": {
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
                "unique_documents": len(
                    set([r.id for r in dense_results + sparse_results])
                ),
            },
            "top_results_analysis": [],
        }

        # Analyze top results
        fused_results = self.fuse(dense_results, sparse_results)

        for i, result in enumerate(fused_results[:top_k]):
            doc_id = result.id

            # Find positions in original rankings
            dense_rank = next(
                (
                    idx + 1
                    for idx, r in enumerate(dense_results)
                    if r.id == doc_id
                ),
                None,
            )
            sparse_rank = next(
                (
                    idx + 1
                    for idx, r in enumerate(sparse_results)
                    if r.id == doc_id
                ),
                None,
            )

            # Calculate individual RRF contributions
            dense_contribution = (
                1.0 / (self.k + dense_rank) if dense_rank else 0.0
            )
            sparse_contribution = (
                1.0 / (self.k + sparse_rank) if sparse_rank else 0.0
            )

            result_analysis = {
                "final_rank": i + 1,
                "document_id": doc_id,
                "rrf_score": getattr(result, "payload", {}).get("rrf_score", 0),
                "dense_rank": dense_rank,
                "sparse_rank": sparse_rank,
                "dense_contribution": dense_contribution,
                "sparse_contribution": sparse_contribution,
                "fusion_explanation": f"RRF = {dense_contribution:.4f} (dense) + {sparse_contribution:.4f} (sparse) = {dense_contribution + sparse_contribution:.4f}",
            }

            explanation["top_results_analysis"].append(result_analysis)

        logger.info("Generated fusion explanation", analyzed_results=len(explanation["top_results_analysis"]))
        return explanation


class WeightedSumFusionRanker:
    """
    Weighted sum fusion for hybrid search results.

    Combines search results by normalizing scores and applying weighted summation.
    Best used when score ranges are similar between dense and sparse results.

    Task 215: Enhanced with unified logging system.
    """

    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> None:
        """Initialize weighted sum ranker.

        Args:
            dense_weight: Weight for dense (semantic) results
            sparse_weight: Weight for sparse (keyword) results
        """
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        logger.debug(
            "Initialized weighted sum ranker",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    def fuse(self, dense_results: list, sparse_results: list) -> list:
        """Fuse results using weighted sum of normalized scores.

        Args:
            dense_results: Dense search results with scores
            sparse_results: Sparse search results with scores

        Returns:
            List of fused results sorted by weighted sum
        """
        logger.debug(
            "Starting weighted sum fusion",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # Normalize scores within each result set
        dense_scores = self._normalize_scores(dense_results)
        sparse_scores = self._normalize_scores(sparse_results)

        # Combine scores with weights
        combined_scores = defaultdict(float)

        for result, norm_score in zip(dense_results, dense_scores):
            combined_scores[result.id] += self.dense_weight * norm_score

        for result, norm_score in zip(sparse_results, sparse_scores):
            combined_scores[result.id] += self.sparse_weight * norm_score

        # Create final results
        all_results = {result.id: result for result in dense_results + sparse_results}
        sorted_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        fused_results = []
        for doc_id, weighted_score in sorted_results:
            if doc_id in all_results:
                result = all_results[doc_id]
                if hasattr(result, "payload"):
                    result.payload = result.payload or {}
                    result.payload["weighted_score"] = weighted_score
                fused_results.append(result)

        logger.info(
            "Weighted sum fusion completed",
            input_dense=len(dense_results),
            input_sparse=len(sparse_results),
            output_count=len(fused_results),
        )

        return fused_results

    def _normalize_scores(self, results: list) -> list:
        """Normalize scores to [0, 1] range."""
        if not results:
            return []

        scores = [result.score for result in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        normalized = [(score - min_score) / (max_score - min_score) for score in scores]
        logger.debug("Normalized scores", min=min_score, max=max_score, count=len(scores))
        return normalized


class HybridSearchEngine:
    """
    Advanced hybrid search engine with multiple fusion strategies and optimized metadata filtering.

    Provides comprehensive hybrid search capabilities by combining dense semantic
    and sparse keyword vector searches with configurable fusion methods. Enhanced
    with advanced metadata filtering optimization for sub-3ms response times.

    Task 215: Enhanced with unified logging system for comprehensive observability.
    Task 233.3: Enhanced with advanced metadata filtering optimization strategies.
    Task 233.5: Enhanced with multi-tenant result aggregation capabilities.
    Task 249.3: Integrated comprehensive metadata filtering system for enhanced project isolation.
    """

    def __init__(
        self,
        client: QdrantClient,
        enable_optimizations: bool = True,
        enable_multi_tenant_aggregation: bool = True,
        enable_performance_monitoring: bool = True,
        performance_baseline_config: Optional[Dict] = None
    ) -> None:
        """Initialize hybrid search engine with optimization, aggregation, and performance monitoring features.

        Args:
            client: Qdrant client for vector database operations
            enable_optimizations: Whether to enable advanced filtering optimizations
            enable_multi_tenant_aggregation: Whether to enable multi-tenant result aggregation
            enable_performance_monitoring: Whether to enable comprehensive performance monitoring
            performance_baseline_config: Optional performance baseline configuration
        """
        self.client = client
        self.rrf_ranker = RRFFusionRanker()
        self.weighted_ranker = WeightedSumFusionRanker()

        # Task 233.1: Initialize multi-tenant components for enhanced metadata filtering
        self.isolation_manager = ProjectIsolationManager()
        self.workspace_registry = WorkspaceCollectionRegistry()

        # Task 249.3: Initialize comprehensive metadata filtering system
        self.metadata_filter_manager = MetadataFilterManager(
            qdrant_client=client,
            enable_caching=True,
            enable_performance_monitoring=enable_performance_monitoring
        )

        # Task 233.5: Initialize multi-tenant result aggregation components
        self.multi_tenant_aggregation_enabled = enable_multi_tenant_aggregation
        if enable_multi_tenant_aggregation:
            self.result_aggregator = MultiTenantResultAggregator(
                preserve_tenant_isolation=True,
                enable_score_normalization=True,
                default_aggregation_method="max_score"
            )
        else:
            self.result_aggregator = None

        # Task 233.3: Initialize optimization components
        self.optimizations_enabled = enable_optimizations
        if enable_optimizations:
            self.filter_optimizer = FilterOptimizer(cache_size=500, cache_ttl_minutes=60)
            self.index_manager = MetadataIndexManager(client)
            self.query_optimizer = QueryOptimizer(target_response_time=3.0)
            self.performance_tracker = PerformanceTracker(target_response_time=3.0)
        else:
            self.filter_optimizer = None
            self.index_manager = None
            self.query_optimizer = None
            self.performance_tracker = None

        # Task 233.6: Initialize comprehensive performance monitoring system
        self.performance_monitoring_enabled = enable_performance_monitoring
        if enable_performance_monitoring:
            self.performance_monitor = MetadataFilteringPerformanceMonitor(
                search_engine=self,
                baseline_config=performance_baseline_config
            )
        else:
            self.performance_monitor = None

        logger.info(
            "Initialized hybrid search engine with comprehensive monitoring support",
            optimizations_enabled=enable_optimizations,
            multi_tenant_aggregation_enabled=enable_multi_tenant_aggregation,
            performance_monitoring_enabled=enable_performance_monitoring
        )

    async def hybrid_search(
        self,
        collection_name: str,
        query_embeddings: dict,
        limit: int = 10,
        fusion_method: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        filter_conditions: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        project_context: Optional[Union[dict, ProjectMetadata]] = None,
        auto_inject_metadata: bool = True,
        additional_filters: Optional[dict] = None,
    ) -> dict:
        """Execute optimized hybrid search with metadata filtering and sub-3ms performance targeting.

        Args:
            collection_name: Name of collection to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            filter_conditions: Optional Qdrant filters
            search_params: Optional search parameters (auto-optimized if None)
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors
            project_context: Optional project context (dict or ProjectMetadata) for metadata filtering
            auto_inject_metadata: Whether to automatically inject project metadata filters
            additional_filters: Additional metadata filters as dict

        Returns:
            Dictionary with fused results, search metadata, and performance metrics
        """
        # Task 233.3: Start performance tracking
        search_start_time = time.time() if (self.optimizations_enabled or self.performance_monitoring_enabled) else None

        logger.info(
            "Starting optimized hybrid search",
            collection=collection_name,
            limit=limit,
            fusion_method=fusion_method,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            project_context=project_context,
            auto_inject_metadata=auto_inject_metadata,
            optimizations_enabled=self.optimizations_enabled
        )

        # Task 233.3: Build enhanced filter with optimization
        if self.optimizations_enabled and self.filter_optimizer:
            enhanced_filter, cache_hit = self.filter_optimizer.get_optimized_filter(
                project_context=project_context,
                additional_filters=additional_filters,
                base_filter=filter_conditions
            )
        else:
            # Fallback to original method
            enhanced_filter = self._build_enhanced_filter(
                base_filter=filter_conditions,
                project_context=project_context,
                auto_inject=auto_inject_metadata
            )
            cache_hit = False

        # Task 233.3: Optimize search parameters if not provided
        if search_params is None and self.optimizations_enabled and self.query_optimizer:
            has_filters = enhanced_filter is not None
            search_params = self.query_optimizer.optimize_search_params(
                collection_name=collection_name,
                query_type="hybrid",
                limit=limit,
                has_filters=has_filters
            )

        search_results = {"dense_results": [], "sparse_results": [], "fused_results": []}

        # Dense vector search
        if "dense" in query_embeddings and query_embeddings["dense"]:
            try:
                logger.debug("Executing dense vector search")
                dense_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_embeddings["dense"],
                    limit=limit * 2,  # Get more results for better fusion
                    query_filter=enhanced_filter,
                    search_params=search_params,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                )
                search_results["dense_results"] = dense_results
                logger.debug("Dense search completed", results_count=len(dense_results))

            except Exception as e:
                logger.error("Dense search failed", error=str(e), collection=collection_name)
                raise

        # Sparse vector search
        if "sparse" in query_embeddings and query_embeddings["sparse"]:
            try:
                logger.debug("Executing sparse vector search")
                sparse_vector = create_named_sparse_vector(query_embeddings["sparse"])
                sparse_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=sparse_vector,
                    limit=limit * 2,  # Get more results for better fusion
                    query_filter=enhanced_filter,
                    search_params=search_params,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                )
                search_results["sparse_results"] = sparse_results
                logger.debug("Sparse search completed", results_count=len(sparse_results))

            except Exception as e:
                logger.error("Sparse search failed", error=str(e), collection=collection_name)
                raise

        # Fuse results based on selected method
        if search_results["dense_results"] or search_results["sparse_results"]:
            logger.debug("Starting result fusion", method=fusion_method)

            if fusion_method == "rrf":
                fusion_ranker = RRFFusionRanker()
                fused_results = fusion_ranker.fuse(
                    search_results["dense_results"],
                    search_results["sparse_results"],
                    weights={"dense": dense_weight, "sparse": sparse_weight},
                )
            elif fusion_method == "weighted_sum":
                fusion_ranker = WeightedSumFusionRanker(dense_weight, sparse_weight)
                fused_results = fusion_ranker.fuse(
                    search_results["dense_results"], search_results["sparse_results"]
                )
            elif fusion_method == "max_score":
                fused_results = self._max_score_fusion(
                    search_results["dense_results"], search_results["sparse_results"]
                )
            else:
                logger.warning("Unknown fusion method, using RRF", method=fusion_method)
                fusion_ranker = RRFFusionRanker()
                fused_results = fusion_ranker.fuse(
                    search_results["dense_results"],
                    search_results["sparse_results"],
                )

            # Limit final results
            search_results["fused_results"] = fused_results[:limit]

            # Task 233.3 & 233.6: Track performance and add monitoring metadata
            if (self.optimizations_enabled or self.performance_monitoring_enabled) and search_start_time:
                total_search_time = (time.time() - search_start_time) * 1000  # Convert to ms

                # Track performance with existing optimization tracker
                if self.optimizations_enabled and self.performance_tracker:
                    self.performance_tracker.record_measurement(
                        operation="hybrid_search",
                        response_time=total_search_time,
                        metadata={
                            "collection": collection_name,
                            "fusion_method": fusion_method,
                            "has_filters": enhanced_filter is not None,
                            "cache_hit": cache_hit
                        }
                    )

                if self.optimizations_enabled and self.query_optimizer:
                    query_analysis = self.query_optimizer.track_query_performance(
                        query_type="hybrid",
                        response_time=total_search_time,
                        result_count=len(search_results["fused_results"]),
                        has_filters=enhanced_filter is not None
                    )

                # Task 233.6: Record performance monitoring metrics
                if self.performance_monitoring_enabled and self.performance_monitor:
                    # Record real-time metric for dashboard
                    self.performance_monitor.dashboard.record_real_time_metric(
                        operation_type="hybrid_search",
                        response_time=total_search_time,
                        metadata={
                            "collection": collection_name,
                            "fusion_method": fusion_method,
                            "has_filters": enhanced_filter is not None,
                            "cache_hit": cache_hit,
                            "result_count": len(search_results["fused_results"]),
                            "project_context": str(project_context) if project_context else None
                        }
                    )

                # Add comprehensive performance metadata to results
                search_results["performance"] = {
                    "response_time_ms": total_search_time,
                    "cache_hit": cache_hit,
                    "target_met": total_search_time <= self.performance_monitor.baseline.target_response_time if self.performance_monitor else total_search_time <= 3.0,
                    "baseline_response_time": self.performance_monitor.baseline.target_response_time if self.performance_monitor else 2.18,
                    "optimizations_used": self.optimizations_enabled,
                    "performance_monitoring_enabled": self.performance_monitoring_enabled
                }

                logger.info(
                    "Hybrid search completed with monitoring",
                    collection=collection_name,
                    dense_count=len(search_results["dense_results"]),
                    sparse_count=len(search_results["sparse_results"]),
                    final_count=len(search_results["fused_results"]),
                    fusion_method=fusion_method,
                    response_time_ms=total_search_time,
                    cache_hit=cache_hit,
                    target_met=search_results["performance"]["target_met"],
                    monitoring_enabled=self.performance_monitoring_enabled
                )
            else:
                logger.info(
                    "Hybrid search completed",
                    collection=collection_name,
                    dense_count=len(search_results["dense_results"]),
                    sparse_count=len(search_results["sparse_results"]),
                    final_count=len(search_results["fused_results"]),
                    fusion_method=fusion_method,
                )

        return search_results

    def _max_score_fusion(self, dense_results: list, sparse_results: list) -> list:
        """Simple max score fusion strategy.

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results

        Returns:
            Results sorted by maximum score across modalities
        """
        logger.debug("Executing max score fusion")

        all_results = {}

        # Collect all results with their max scores
        for result in dense_results + sparse_results:
            doc_id = result.id
            if doc_id not in all_results or result.score > all_results[doc_id].score:
                all_results[doc_id] = result

        # Sort by score (descending)
        sorted_results = sorted(
            all_results.values(), key=lambda x: x.score, reverse=True
        )

        logger.debug("Max score fusion completed", final_count=len(sorted_results))
        return sorted_results

    def _build_enhanced_filter(
        self,
        base_filter: Optional[models.Filter],
        project_context: Optional[Union[dict, ProjectMetadata]],
        auto_inject: bool = True
    ) -> Optional[models.Filter]:
        """Build enhanced filter with project metadata constraints using comprehensive filtering system.

        Task 233.1: Enhanced to leverage ProjectIsolationManager for consistent filtering
        and support ProjectMetadata objects directly.
        Task 249.3: Updated to use new comprehensive MetadataFilterManager for improved
        performance and advanced filtering capabilities.

        Args:
            base_filter: Optional base filter conditions
            project_context: Project context (dict or ProjectMetadata) for metadata injection
            auto_inject: Whether to automatically inject project metadata

        Returns:
            Enhanced filter with project metadata constraints
        """
        if not auto_inject or not project_context:
            return base_filter

        logger.debug("Building enhanced filter with project context", context=str(project_context))

        # Task 249.3: Use new comprehensive metadata filtering system
        try:
            # Convert context to FilterCriteria for the new system
            if isinstance(project_context, ProjectMetadata):
                # Direct conversion from ProjectMetadata
                from .metadata_schema import CollectionCategory, WorkspaceScope, AccessLevel

                criteria = FilterCriteria(
                    project_name=project_context.project_name,
                    collection_types=[project_context.collection_type] if project_context.collection_type else None,
                    tenant_namespace=project_context.tenant_namespace,
                    workspace_scopes=[WorkspaceScope(project_context.workspace_scope)] if project_context.workspace_scope else None,
                    strategy=FilterStrategy.STRICT,
                    performance_level=FilterPerformanceLevel.FAST
                )
            else:
                # Dict format conversion
                project_name = project_context.get("project_name")
                collection_type = project_context.get("collection_type")
                tenant_namespace = project_context.get("tenant_namespace")
                workspace_scope = project_context.get("workspace_scope", "project")

                criteria = FilterCriteria(
                    project_name=project_name,
                    collection_types=[collection_type] if collection_type else None,
                    tenant_namespace=tenant_namespace,
                    strategy=FilterStrategy.STRICT,
                    performance_level=FilterPerformanceLevel.FAST,
                    include_shared=(workspace_scope != "global")
                )

            # Use the new metadata filter manager for enhanced filtering
            filter_result = self.metadata_filter_manager.create_composite_filter(criteria)

            # Log performance metrics
            logger.debug(
                "Created filter using new metadata filtering system",
                construction_time_ms=filter_result.performance_metrics.get("construction_time_ms", 0),
                condition_count=filter_result.performance_metrics.get("condition_count", 0),
                complexity_score=filter_result.performance_metrics.get("complexity_score", 0),
                cache_hit=filter_result.cache_hit,
                optimizations=filter_result.optimizations_applied
            )

            project_filter = filter_result.filter

            # Combine with base filter if both exist
            if base_filter and project_filter:
                # Merge filters by combining must conditions
                existing_conditions = base_filter.must or []
                project_conditions = project_filter.must or []

                enhanced_filter = models.Filter(
                    must=existing_conditions + project_conditions,
                    should=base_filter.should,
                    must_not=base_filter.must_not
                )

                logger.debug(
                    "Enhanced filter built with new metadata filtering system",
                    base_conditions=len(existing_conditions),
                    project_conditions=len(project_conditions),
                    total_conditions=len(existing_conditions + project_conditions)
                )

                return enhanced_filter

            elif project_filter:
                # Use project filter only
                logger.debug("Using project filter from new metadata filtering system")
                return project_filter

            else:
                # No enhancement needed or possible
                return base_filter

        except Exception as e:
            logger.error("Failed to create enhanced filter", error=str(e))
            return base_filter


    def create_project_isolation_filter(
        self,
        project_identifier: Union[str, ProjectMetadata],
        strategy: FilterStrategy = FilterStrategy.STRICT
    ) -> Optional[models.Filter]:
        """Create a filter for complete project isolation using the new metadata filtering system.

        Task 249.3: Added direct access to project isolation filtering with the new system.

        Args:
            project_identifier: Project name, project_id, or metadata schema
            strategy: Filtering strategy to use

        Returns:
            Qdrant filter for project isolation, or None if creation fails
        """
        try:
            filter_result = self.metadata_filter_manager.create_project_isolation_filter(
                project_identifier, strategy
            )

            logger.debug(
                "Created project isolation filter",
                construction_time_ms=filter_result.performance_metrics.get("construction_time_ms", 0),
                cache_hit=filter_result.cache_hit,
                optimizations=filter_result.optimizations_applied
            )

            return filter_result.filter

        except Exception as e:
            logger.error("Failed to create project isolation filter", error=str(e))
            return None

    def get_filter_performance_stats(self) -> Dict[str, dict]:
        """Get comprehensive filter performance statistics.

        Task 249.3: Added access to metadata filtering performance metrics.

        Returns:
            Dictionary containing filtering performance statistics
        """
        return self.metadata_filter_manager.get_filter_performance_stats()

    async def search_project_workspace(
        self,
        collection_name: str,
        query_embeddings: dict,
        project_name: str,
        workspace_type: str = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        include_shared: bool = True,
        **kwargs
    ) -> dict:
        """Convenience method for searching within a specific project workspace.

        Task 233.1: Added for simplified multi-tenant project searching.

        Args:
            collection_name: Name of collection to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            project_name: Project name for tenant isolation
            workspace_type: Optional workspace type filter (notes, docs, etc.)
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm to use
            include_shared: Whether to include shared workspace resources
            **kwargs: Additional search parameters

        Returns:
            Dictionary with search results and metadata
        """
        # Validate workspace type if provided
        if workspace_type and not self.workspace_registry.is_multi_tenant_type(workspace_type):
            logger.warning("Invalid workspace type", type=workspace_type)
            return {"fused_results": [], "error": f"Invalid workspace type: {workspace_type}"}

        # Create project metadata for filtering
        project_metadata = ProjectMetadata.create_project_metadata(
            project_name=project_name,
            collection_type=workspace_type or "project",
            workspace_scope="shared" if include_shared else "project"
        )

        return await self.hybrid_search(
            collection_name=collection_name,
            query_embeddings=query_embeddings,
            project_context=project_metadata,
            limit=limit,
            fusion_method=fusion_method,
            auto_inject_metadata=True,
            **kwargs
        )

    async def search_tenant_namespace(
        self,
        collection_name: str,
        query_embeddings: dict,
        tenant_namespace: str,
        limit: int = 10,
        fusion_method: str = "rrf",
        **kwargs
    ) -> dict:
        """Search within a specific tenant namespace for precise multi-tenant isolation.

        Task 233.1: Added for tenant namespace-based searching.

        Args:
            collection_name: Name of collection to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            tenant_namespace: Tenant namespace for precise isolation
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm to use
            **kwargs: Additional search parameters

        Returns:
            Dictionary with search results and metadata
        """
        project_context = {"tenant_namespace": tenant_namespace}

        return await self.hybrid_search(
            collection_name=collection_name,
            query_embeddings=query_embeddings,
            project_context=project_context,
            limit=limit,
            fusion_method=fusion_method,
            auto_inject_metadata=True,
            **kwargs
        )

    def get_supported_workspace_types(self) -> set:
        """Get all supported workspace collection types.

        Task 233.1: Added for workspace type validation.

        Returns:
            Set of supported workspace types
        """
        return self.workspace_registry.get_workspace_types()

    def validate_workspace_type(self, workspace_type: str) -> bool:
        """Validate if a workspace type is supported for multi-tenant operations.

        Task 233.1: Added for workspace type validation.

        Args:
            workspace_type: Workspace type to validate

        Returns:
            True if supported, False otherwise
        """
        return self.workspace_registry.is_multi_tenant_type(workspace_type)

    # Task 233.5: Add multi-tenant result aggregation methods

    async def multi_collection_hybrid_search(
        self,
        collection_names: List[str],
        query_embeddings: dict,
        project_contexts: Dict[str, dict] = None,
        limit: int = 10,
        fusion_method: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        filter_conditions: Optional[models.Filter] = None,
        search_params: Optional[models.SearchParams] = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        score_threshold: float = 0.0,
        enable_deduplication: bool = True,
        aggregation_method: str = "max_score"
    ) -> dict:
        """
        Execute hybrid search across multiple collections with multi-tenant result aggregation.

        This method performs hybrid search across multiple collections and applies sophisticated
        result aggregation including deduplication, tenant-aware ranking, and score normalization.

        Args:
            collection_names: List of collections to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            project_contexts: Optional dict mapping collection names to project contexts
            limit: Maximum number of results to return across all collections
            fusion_method: Fusion algorithm ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            filter_conditions: Optional Qdrant filters
            search_params: Optional search parameters
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors
            score_threshold: Minimum score threshold for inclusion
            enable_deduplication: Whether to deduplicate results across collections
            aggregation_method: How to aggregate duplicate scores ("max_score", "avg_score", "sum_score")

        Returns:
            Dictionary with aggregated results and detailed metadata

        Task 233.5: Added for multi-tenant search result aggregation.
        """
        search_start_time = time.time() if self.optimizations_enabled else None

        logger.info(
            "Starting multi-collection hybrid search with result aggregation",
            collection_count=len(collection_names),
            limit=limit,
            fusion_method=fusion_method,
            enable_deduplication=enable_deduplication,
            aggregation_method=aggregation_method,
            multi_tenant_aggregation_enabled=self.multi_tenant_aggregation_enabled
        )

        if not self.multi_tenant_aggregation_enabled:
            # Fallback to individual collection searches without advanced aggregation
            logger.warning("Multi-tenant aggregation disabled, using basic aggregation")
            return await self._basic_multi_collection_search(
                collection_names, query_embeddings, limit, fusion_method,
                dense_weight, sparse_weight, filter_conditions, search_params,
                with_payload, with_vectors, score_threshold
            )

        # Perform individual collection searches
        collection_results = {}
        total_raw_results = 0

        for collection_name in collection_names:
            try:
                # Get project context for this collection
                project_context = project_contexts.get(collection_name) if project_contexts else None

                # Perform hybrid search on individual collection
                collection_search_result = await self.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=limit * 2,  # Get more results per collection for better aggregation
                    fusion_method=fusion_method,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    filter_conditions=filter_conditions,
                    search_params=search_params,
                    with_payload=with_payload,
                    with_vectors=with_vectors,
                    project_context=project_context,
                    auto_inject_metadata=True
                )

                # Extract fused results
                fused_results = collection_search_result.get("fused_results", [])
                collection_results[collection_name] = fused_results
                total_raw_results += len(fused_results)

                logger.debug(
                    "Collection search completed",
                    collection=collection_name,
                    results_count=len(fused_results)
                )

            except Exception as e:
                logger.error(
                    "Failed to search collection",
                    collection=collection_name,
                    error=str(e)
                )
                collection_results[collection_name] = []
                continue

        # Use multi-tenant result aggregator for sophisticated aggregation
        if enable_deduplication and self.result_aggregator:
            aggregated_response = self.result_aggregator.aggregate_multi_collection_results(
                collection_results=collection_results,
                project_contexts=project_contexts,
                limit=limit,
                score_threshold=score_threshold,
                aggregation_method=aggregation_method
            )
        else:
            # Simple aggregation without deduplication
            all_results = []
            for collection_name, results in collection_results.items():
                for result in results:
                    if hasattr(result, 'score') and result.score >= score_threshold:
                        result_dict = {
                            "id": getattr(result, 'id', ''),
                            "score": getattr(result, 'score', 0.0),
                            "payload": getattr(result, 'payload', {}),
                            "collection": collection_name,
                            "search_type": getattr(result, 'search_type', 'hybrid')
                        }
                        all_results.append(result_dict)

            # Sort and limit
            all_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = all_results[:limit]

            aggregated_response = {
                "total_results": len(final_results),
                "results": final_results,
                "aggregation_metadata": {
                    "collection_count": len(collection_names),
                    "raw_result_count": total_raw_results,
                    "final_count": len(final_results),
                    "deduplication_enabled": False,
                    "score_threshold": score_threshold
                }
            }

        # Add performance metrics if available
        if search_start_time and self.optimizations_enabled:
            total_search_time = (time.time() - search_start_time) * 1000
            aggregated_response["performance"] = {
                "response_time_ms": total_search_time,
                "target_met": total_search_time <= 3.0,
                "multi_tenant_aggregation_enabled": self.multi_tenant_aggregation_enabled
            }

            if self.performance_tracker:
                self.performance_tracker.record_measurement(
                    operation="multi_collection_hybrid_search",
                    response_time=total_search_time,
                    metadata={
                        "collection_count": len(collection_names),
                        "fusion_method": fusion_method,
                        "deduplication_enabled": enable_deduplication,
                        "aggregation_method": aggregation_method
                    }
                )

        logger.info(
            "Multi-collection hybrid search completed",
            collection_count=len(collection_names),
            raw_results=total_raw_results,
            final_results=aggregated_response["total_results"],
            response_time_ms=aggregated_response.get("performance", {}).get("response_time_ms")
        )

        return aggregated_response

    async def _basic_multi_collection_search(
        self,
        collection_names: List[str],
        query_embeddings: dict,
        limit: int,
        fusion_method: str,
        dense_weight: float,
        sparse_weight: float,
        filter_conditions: Optional[models.Filter],
        search_params: Optional[models.SearchParams],
        with_payload: bool,
        with_vectors: bool,
        score_threshold: float
    ) -> dict:
        """
        Basic multi-collection search without advanced aggregation features.

        This is used as a fallback when multi-tenant aggregation is disabled.

        Args:
            collection_names: Collections to search
            query_embeddings: Search embeddings
            limit: Result limit
            fusion_method: Fusion method
            dense_weight: Dense search weight
            sparse_weight: Sparse search weight
            filter_conditions: Optional filters
            search_params: Optional search params
            with_payload: Include payloads
            with_vectors: Include vectors
            score_threshold: Score threshold

        Returns:
            Basic aggregated search results
        """
        all_results = []

        for collection_name in collection_names:
            try:
                collection_search = await self.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=limit,
                    fusion_method=fusion_method,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight,
                    filter_conditions=filter_conditions,
                    search_params=search_params,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )

                for result in collection_search.get("fused_results", []):
                    if hasattr(result, 'score') and result.score >= score_threshold:
                        result_dict = {
                            "id": getattr(result, 'id', ''),
                            "score": getattr(result, 'score', 0.0),
                            "payload": getattr(result, 'payload', {}),
                            "collection": collection_name,
                            "search_type": "hybrid"
                        }
                        all_results.append(result_dict)

            except Exception as e:
                logger.error(f"Basic search failed for collection {collection_name}: {e}")
                continue

        # Sort and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = all_results[:limit]

        return {
            "total_results": len(final_results),
            "results": final_results,
            "aggregation_metadata": {
                "collection_count": len(collection_names),
                "final_count": len(final_results),
                "basic_aggregation": True,
                "score_threshold": score_threshold
            }
        }

    def configure_result_aggregation(
        self,
        preserve_tenant_isolation: bool = None,
        enable_score_normalization: bool = None,
        default_aggregation_method: str = None
    ) -> Dict:
        """
        Configure multi-tenant result aggregation settings.

        Args:
            preserve_tenant_isolation: Whether to preserve tenant isolation
            enable_score_normalization: Whether to enable score normalization
            default_aggregation_method: Default aggregation method

        Returns:
            Dict with current configuration

        Task 233.5: Added for result aggregation configuration.
        """
        if not self.multi_tenant_aggregation_enabled or not self.result_aggregator:
            return {"error": "Multi-tenant aggregation not enabled"}

        if preserve_tenant_isolation is not None:
            self.result_aggregator.preserve_tenant_isolation = preserve_tenant_isolation
            self.result_aggregator.deduplicator.preserve_tenant_isolation = preserve_tenant_isolation

        if enable_score_normalization is not None:
            self.result_aggregator.enable_score_normalization = enable_score_normalization

        if default_aggregation_method is not None:
            self.result_aggregator.default_aggregation_method = default_aggregation_method

        current_config = {
            "preserve_tenant_isolation": self.result_aggregator.preserve_tenant_isolation,
            "enable_score_normalization": self.result_aggregator.enable_score_normalization,
            "default_aggregation_method": self.result_aggregator.default_aggregation_method,
            "multi_tenant_aggregation_enabled": True
        }

        logger.info("Result aggregation configuration updated", config=current_config)
        return current_config

    def get_result_aggregation_stats(self) -> Dict:
        """
        Get statistics about result aggregation performance.

        Returns:
            Dict with aggregation statistics

        Task 233.5: Added for aggregation performance monitoring.
        """
        if not self.multi_tenant_aggregation_enabled:
            return {"error": "Multi-tenant aggregation not enabled"}

        stats = {
            "multi_tenant_aggregation_enabled": True,
            "preserve_tenant_isolation": self.result_aggregator.preserve_tenant_isolation if self.result_aggregator else None,
            "enable_score_normalization": self.result_aggregator.enable_score_normalization if self.result_aggregator else None,
            "default_aggregation_method": self.result_aggregator.default_aggregation_method if self.result_aggregator else None
        }

        return stats

    # Task 233.3: Add optimization management methods

    async def ensure_collection_optimized(self, collection_name: str, force_recreate: bool = False) -> Dict:
        """Ensure collection has optimal metadata indexes for filtering performance.

        Args:
            collection_name: Collection to optimize
            force_recreate: Whether to recreate existing indexes

        Returns:
            Dict with optimization results
        """
        if not self.optimizations_enabled or not self.index_manager:
            return {"error": "Optimizations not enabled"}

        try:
            index_results = await self.index_manager.ensure_optimal_indexes(
                collection_name, force_recreate
            )

            settings_optimized = await self.index_manager.optimize_collection_settings(
                collection_name
            )

            logger.info("Collection optimization completed",
                       collection=collection_name,
                       indexes_created=sum(index_results.values()),
                       settings_optimized=settings_optimized)

            return {
                "collection": collection_name,
                "index_results": index_results,
                "settings_optimized": settings_optimized,
                "optimizations_enabled": True
            }

        except Exception as e:
            logger.error("Collection optimization failed",
                        collection=collection_name, error=str(e))
            return {"error": f"Optimization failed: {e}"}

    def get_optimization_performance(self) -> Dict:
        """Get comprehensive performance metrics for all optimization components.

        Returns:
            Dict with performance metrics from all optimizers
        """
        if not self.optimizations_enabled:
            return {"error": "Optimizations not enabled"}

        performance_data = {
            "optimizations_enabled": True,
            "target_response_time_ms": 3.0
        }

        # Filter optimizer metrics
        if self.filter_optimizer:
            performance_data["filter_cache"] = self.filter_optimizer.get_performance_metrics()

        # Query optimizer metrics
        if self.query_optimizer:
            performance_data["query_optimization"] = self.query_optimizer.get_performance_summary()

        # Performance tracker metrics
        if self.performance_tracker:
            performance_data["overall_performance"] = self.performance_tracker.get_performance_report()

        # Index manager status
        if self.index_manager:
            performance_data["indexed_collections"] = list(self.index_manager.get_indexed_collections())

        return performance_data

    def clear_optimization_caches(self) -> Dict:
        """Clear all optimization caches to free memory or reset performance tracking.

        Returns:
            Dict with clearing results
        """
        if not self.optimizations_enabled:
            return {"error": "Optimizations not enabled"}

        results = {}

        if self.filter_optimizer:
            self.filter_optimizer.clear_cache()
            results["filter_cache_cleared"] = True

        logger.info("Optimization caches cleared", results=results)
        return results

    def get_performance_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent performance alerts from the performance tracker.

        Args:
            hours: Number of hours back to look for alerts

        Returns:
            List of recent performance alerts
        """
        alerts = []

        # Get alerts from optimization tracker
        if self.optimizations_enabled and self.performance_tracker:
            alerts.extend(self.performance_tracker.get_recent_alerts(hours))

        # Get alerts from performance monitoring system
        if self.performance_monitoring_enabled and self.performance_monitor:
            accuracy_alerts = self.performance_monitor.accuracy_tracker.get_recent_accuracy_alerts(hours)
            alerts.extend(accuracy_alerts)

        return alerts

    # Task 233.6: Add performance monitoring methods

    def get_performance_monitoring_status(self) -> Dict:
        """Get comprehensive performance monitoring status.

        Returns:
            Dict with monitoring status and metrics
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}

        return self.performance_monitor.get_performance_status()

    def get_performance_dashboard_data(self) -> Dict:
        """Get real-time dashboard data.

        Returns:
            Dict with dashboard data for visualization
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}

        return self.performance_monitor.dashboard.get_real_time_dashboard()

    async def run_performance_benchmark(
        self,
        collection_name: str,
        query_count: int = 50,
        iterations: int = 10
    ) -> Optional[Dict]:
        """Run performance benchmark against baselines.

        Args:
            collection_name: Collection to benchmark
            query_count: Number of test queries
            iterations: Iterations per query

        Returns:
            Benchmark results or None if monitoring disabled
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            logger.warning("Cannot run benchmark - performance monitoring not enabled")
            return None

        # Generate test queries (simplified for this integration)
        import random
        test_queries = []
        for i in range(query_count):
            query = {
                "embeddings": {
                    "dense": [random.gauss(0, 1) for _ in range(384)],
                    "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
                },
                "project_context": {
                    "project_name": f"test_project_{i % 3}",
                    "collection_type": "project"
                },
                "expected_results": [f"result_{j}" for j in range(5)]
            }
            test_queries.append(query)

        # Run benchmark
        result = await self.performance_monitor.benchmark_suite.run_metadata_filtering_benchmark(
            collection_name=collection_name,
            test_queries=test_queries,
            iterations=iterations
        )

        # Convert to dict for JSON serialization
        return {
            "benchmark_id": result.benchmark_id,
            "timestamp": result.timestamp.isoformat(),
            "test_name": result.test_name,
            "avg_response_time": result.avg_response_time,
            "p95_response_time": result.p95_response_time,
            "avg_precision": result.avg_precision,
            "avg_recall": result.avg_recall,
            "passes_baseline": result.passes_baseline(self.performance_monitor.baseline),
            "performance_regression": result.performance_regression,
            "accuracy_regression": result.accuracy_regression,
            "baseline_comparison": result.baseline_comparison
        }

    def record_search_accuracy(
        self,
        query_id: str,
        query_text: str,
        collection_name: str,
        search_results: List,
        expected_results: List,
        tenant_context: Optional[str] = None
    ) -> Optional[Dict]:
        """Record search accuracy measurement for monitoring.

        Args:
            query_id: Unique query identifier
            query_text: Query text
            collection_name: Collection searched
            search_results: Actual search results
            expected_results: Expected results
            tenant_context: Optional tenant context

        Returns:
            Accuracy measurement or None if monitoring disabled
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return None

        measurement = self.performance_monitor.accuracy_tracker.record_search_accuracy(
            query_id=query_id,
            query_text=query_text,
            collection_name=collection_name,
            search_results=search_results,
            expected_results=expected_results,
            tenant_context=tenant_context
        )

        return {
            "query_id": measurement.query_id,
            "precision": measurement.precision,
            "recall": measurement.recall,
            "f1_score": measurement.f1_score,
            "timestamp": measurement.timestamp.isoformat()
        }

    async def export_performance_report(self, filepath: Optional[str] = None) -> Optional[Dict]:
        """Export comprehensive performance report.

        Args:
            filepath: Optional output file path

        Returns:
            Export result or None if monitoring disabled
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return None

        return self.performance_monitor.dashboard.export_performance_report(filepath)

    def get_baseline_configuration(self) -> Optional[Dict]:
        """Get current performance baseline configuration.

        Returns:
            Baseline configuration or None if monitoring disabled
        """
        if not self.performance_monitoring_enabled or not self.performance_monitor:
            return None

        return self.performance_monitor.baseline.to_dict()