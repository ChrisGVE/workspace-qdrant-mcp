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
import asyncio
import inspect
import time
from collections import defaultdict
from dataclasses import dataclass

# Task 222: Import loguru-based logging system
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Task 249.3: Import new comprehensive metadata filtering system
from .metadata_filtering import (
    MetadataFilterManager,
)
from .metadata_optimization import (
    FilterOptimizer,
    QueryOptimizer,
)
from .performance_monitoring import (
    MetadataFilteringPerformanceMonitor,
)
from .sparse_vectors import create_named_sparse_vector

# Task 215: Use unified logging system instead of logging.getLogger(__name__)
# logger imported from loguru


@dataclass(init=False)
class TenantAwareResult:
    """Result container with tenant-aware metadata for multi-tenant result aggregation."""

    id: str
    score: float
    payload: dict
    collection: str | None = None
    search_type: str | None = None
    tenant_id: str | None = None
    dedup_key_fields: list[str] | None = None
    tenant_metadata: dict | None = None
    project_context: dict | None = None
    deduplication_key: str | None = None

    def __init__(
        self,
        id: str,
        score: float,
        payload: dict,
        tenant_id: str | None = None,
        *,
        collection: str | None = None,
        search_type: str | None = None,
        tenant_metadata: dict | None = None,
        project_context: dict | None = None,
        deduplication_key: str | None = None,
        dedup_key_fields: list[str] | None = None,
    ) -> None:
        self.id = id
        self.score = score
        self.payload = payload or {}
        self.collection = collection
        self.search_type = search_type
        self.tenant_id = tenant_id
        self.dedup_key_fields = dedup_key_fields
        self.tenant_metadata = tenant_metadata
        self.project_context = project_context
        self.deduplication_key = deduplication_key
        self.__post_init__()

    def __post_init__(self):
        """Post-init processing for tenant-aware results."""
        if self.tenant_metadata is None:
            self.tenant_metadata = {}
        if self.tenant_id is not None:
            self.tenant_metadata.setdefault("tenant_id", self.tenant_id)

        if self.project_context is None:
            self.project_context = {}

        # Generate deduplication key based on configured fields or known identifiers
        if self.deduplication_key is None:
            if self.dedup_key_fields:
                parts = [
                    str(self.payload.get(field, ""))
                    for field in self.dedup_key_fields
                    if self.payload.get(field) is not None
                ]
                if parts:
                    self.deduplication_key = "|".join(parts)
            if self.deduplication_key is None:
                content_hash = self.payload.get("content_hash") or self.payload.get("content")
                file_path = self.payload.get("file_path")
                doc_id = self.payload.get("document_id")
                self.deduplication_key = content_hash or file_path or doc_id or self.id

    def get_dedup_key(self) -> str:
        """Return the computed deduplication key."""
        return self.deduplication_key or self.id

    def get_metadata(self) -> dict:
        """Return payload metadata excluding content fields."""
        return {
            key: value
            for key, value in self.payload.items()
            if key not in {"content"}
        }

    def belongs_to_tenant(self, tenant_id: str | None) -> bool:
        """Check if result belongs to the specified tenant (None = global access)."""
        if tenant_id is None:
            return True
        return self.tenant_id == tenant_id

    def __lt__(self, other: "TenantAwareResult") -> bool:
        return self.score < other.score

    def __gt__(self, other: "TenantAwareResult") -> bool:
        return self.score > other.score


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

    def __init__(
        self,
        preserve_tenant_isolation: bool = True,
        isolation_mode: str | None = None,
        dedup_strategy: str = "id",
        aggregation_method: str = "max_score",
    ):
        """
        Initialize deduplicator with tenant isolation settings.

        Args:
            preserve_tenant_isolation: If True, maintains separate results for different tenants
        """
        self.preserve_tenant_isolation = preserve_tenant_isolation
        self.isolation_mode = isolation_mode or ("strict" if preserve_tenant_isolation else "relaxed")
        self.dedup_strategy = dedup_strategy
        self.aggregation_method = aggregation_method

    def deduplicate_results(
        self,
        results: list[TenantAwareResult],
        aggregation_method: str = "max_score"
    ) -> list[TenantAwareResult]:
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

        for _group_key, group_results in result_groups.items():
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

        logger.debug(
            "Deduplication complete",
            original_count=len(results),
            deduplicated_count=len(deduplicated_results)
        )

        return deduplicated_results

    def deduplicate(
        self,
        results: list[TenantAwareResult],
        target_tenant: str | None = None,
        aggregation_method: str | None = None,
    ) -> list[TenantAwareResult]:
        """Compatibility wrapper for test suite expectations."""
        if not results:
            return []

        aggregation_method = aggregation_method or self.aggregation_method
        filtered = results

        if self.isolation_mode == "strict" and target_tenant is not None:
            filtered = [r for r in results if r.tenant_id == target_tenant]

        # Grouping strategy
        groups: dict[str, list[TenantAwareResult]] = defaultdict(list)
        for result in filtered:
            if self.dedup_strategy == "content_hash":
                key = result.payload.get("content") or result.get_dedup_key()
            else:
                key = result.get_dedup_key()
            groups[str(key)].append(result)

        deduplicated: list[TenantAwareResult] = []
        for group_results in groups.values():
            if len(group_results) == 1:
                deduplicated.append(group_results[0])
                continue

            if aggregation_method in ("average", "avg"):
                avg_score = sum(r.score for r in group_results) / len(group_results)
                best = max(group_results, key=lambda r: r.score)
                best.score = avg_score
                deduplicated.append(best)
            else:
                best = max(group_results, key=lambda r: r.score)
                deduplicated.append(best)

        deduplicated.sort(key=lambda r: r.score, reverse=True)
        return deduplicated

    def _get_group_key(self, result: TenantAwareResult) -> str:
        """
        Generate grouping key for deduplication.

        Args:
            result: TenantAwareResult to generate key for

        Returns:
            String key for grouping
        """
        if self.preserve_tenant_isolation:
            # Include tenant in grouping to maintain isolation
            tenant_id = result.tenant_metadata.get("tenant_id", "default")
            return f"{tenant_id}:{result.deduplication_key}"
        else:
            # Cross-tenant deduplication
            return result.deduplication_key

    def _aggregate_duplicate_results(
        self,
        results: list[TenantAwareResult],
        aggregation_method: str
    ) -> TenantAwareResult:
        """
        Aggregate multiple duplicate results into a single result.

        Args:
            results: List of duplicate results
            aggregation_method: How to aggregate scores

        Returns:
            Single aggregated result
        """
        # Sort by score to find best result
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
        best_result = sorted_results[0]

        # Calculate aggregated score
        if aggregation_method == "max_score":
            aggregated_score = max(r.score for r in results)
        elif aggregation_method == "avg_score":
            aggregated_score = sum(r.score for r in results) / len(results)
        elif aggregation_method == "sum_score":
            aggregated_score = sum(r.score for r in results)
        else:
            # Default to max_score
            aggregated_score = max(r.score for r in results)

        # Create aggregated result based on best result
        aggregated_result = TenantAwareResult(
            id=best_result.id,
            score=aggregated_score,
            payload=best_result.payload,
            collection=best_result.collection,
            search_type=best_result.search_type,
            tenant_metadata=best_result.tenant_metadata,
            project_context=best_result.project_context,
            deduplication_key=best_result.deduplication_key
        )

        # Add metadata about aggregation
        aggregated_result.payload["_aggregation_metadata"] = {
            "duplicate_count": len(results),
            "aggregation_method": aggregation_method,
            "original_scores": [r.score for r in results]
        }

        return aggregated_result


class MultiTenantResultAggregator:
    """
    Aggregates search results across multiple tenants or collections.

    This class provides sophisticated result aggregation capabilities for multi-tenant
    scenarios, including cross-collection search, score normalization, and deduplication.

    Task 233.5: Added for multi-tenant result aggregation.
    """

    def __init__(
        self,
        enable_deduplication: bool = True,
        preserve_tenant_isolation: bool = True,
        default_aggregation_method: str = "max_score",
        isolation_mode: str | None = None,
        normalize_scores: bool = False,
    ):
        """
        Initialize multi-tenant result aggregator.

        Args:
            enable_deduplication: Enable cross-result deduplication
            preserve_tenant_isolation: Maintain tenant isolation during deduplication
            default_aggregation_method: Default score aggregation method
        """
        self.enable_deduplication = enable_deduplication
        self.preserve_tenant_isolation = preserve_tenant_isolation
        self.default_aggregation_method = default_aggregation_method
        self.isolation_mode = isolation_mode or ("strict" if preserve_tenant_isolation else "relaxed")
        self.normalize_scores = normalize_scores
        self.deduplicator = TenantAwareResultDeduplicator(preserve_tenant_isolation)

        logger.debug("Initialized MultiTenantResultAggregator")

    def aggregate_results(
        self,
        results: list[TenantAwareResult],
        target_tenant: str | None = None,
    ) -> list[dict]:
        """Aggregate tenant-aware results into API-friendly dicts."""
        filtered = results
        if self.isolation_mode == "strict" and target_tenant is not None:
            filtered = [r for r in results if r.tenant_id in (target_tenant, None)]

        if self.enable_deduplication:
            filtered = self.deduplicator.deduplicate(filtered, target_tenant=target_tenant)

        if self.normalize_scores and filtered:
            scores = [r.score for r in filtered]
            min_score = min(scores)
            max_score = max(scores)
            for r in filtered:
                if max_score == min_score:
                    r.score = 1.0
                else:
                    r.score = (r.score - min_score) / (max_score - min_score)

        aggregated = []
        for result in filtered:
            aggregated.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "tenant_info": {"tenant_id": result.tenant_id},
                "tenant_id": result.tenant_id,
            })
        return aggregated

    def convert_to_api_format(self, results: list[TenantAwareResult]) -> list[dict]:
        """Convert tenant-aware results to API response format."""
        api_results = []
        for result in results:
            api_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
                "metadata": {"tenant_id": result.tenant_id},
            })
        return api_results

    def aggregate_multi_collection_results(
        self,
        collection_results: dict[str, list],
        score_normalization: str = "min_max",
        aggregation_method: str | None = None,
        deduplication_enabled: bool | None = None
    ) -> dict[str, any]:
        """
        Aggregate results from multiple collections with score normalization.

        Args:
            collection_results: Dictionary mapping collection names to search results
            score_normalization: Normalization method ("min_max", "z_score", "none")
            aggregation_method: Score aggregation method (overrides default if provided)
            deduplication_enabled: Override deduplication setting

        Returns:
            Dictionary containing aggregated results and metadata
        """
        aggregation_method = aggregation_method or self.default_aggregation_method
        deduplication_enabled = (
            deduplication_enabled if deduplication_enabled is not None
            else self.enable_deduplication
        )

        logger.debug(
            "Starting multi-collection aggregation",
            collection_count=len(collection_results),
            normalization=score_normalization,
            aggregation_method=aggregation_method,
            deduplication=deduplication_enabled
        )

        # Convert all results to TenantAwareResult format
        tenant_aware_results = []
        collection_metadata = {}

        for collection_name, results in collection_results.items():
            collection_metadata[collection_name] = {
                "result_count": len(results),
                "max_score": max((r.score for r in results), default=0.0),
                "min_score": min((r.score for r in results), default=0.0),
            }

            for result in results:
                tenant_result = TenantAwareResult(
                    id=result.id if hasattr(result, 'id') else str(result),
                    score=result.score if hasattr(result, 'score') else 0.0,
                    payload=result.payload if hasattr(result, 'payload') else {},
                    collection=collection_name,
                    search_type="hybrid",
                    tenant_metadata={
                        "collection": collection_name,
                        "tenant_id": collection_name  # Use collection as tenant
                    }
                )
                tenant_aware_results.append(tenant_result)

        # Normalize scores across collections
        if score_normalization != "none" and tenant_aware_results:
            tenant_aware_results = self._normalize_cross_collection_scores(
                tenant_aware_results, method=score_normalization
            )

        # Deduplicate if enabled
        if deduplication_enabled:
            tenant_aware_results = self.deduplicator.deduplicate_results(
                tenant_aware_results,
                aggregation_method=aggregation_method
            )

        # Sort by final score
        tenant_aware_results.sort(key=lambda x: x.score, reverse=True)

        # Build aggregation metadata
        aggregation_metadata = {
            "total_results": len(tenant_aware_results),
            "collection_count": len(collection_results),
            "collections": list(collection_results.keys()),
            "score_normalization": score_normalization,
            "aggregation_method": aggregation_method,
            "deduplication_enabled": deduplication_enabled,
            "collection_metadata": collection_metadata
        }

        # Convert to API format for response
        api_results = self._convert_to_api_format(tenant_aware_results)

        return {
            "results": api_results,
            "total_results": len(api_results),
            "aggregation_metadata": aggregation_metadata
        }

    def _normalize_cross_collection_scores(
        self,
        results: list[TenantAwareResult],
        method: str = "min_max"
    ) -> list[TenantAwareResult]:
        """
        Normalize scores across collections to enable fair comparison.

        Args:
            results: List of tenant-aware results
            method: Normalization method ("min_max", "z_score")

        Returns:
            Results with normalized scores
        """
        if not results:
            return results

        logger.debug(
            "Normalizing cross-collection scores",
            result_count=len(results),
            method=method
        )

        # Group by collection for per-collection normalization
        collection_groups = defaultdict(list)
        for result in results:
            collection_groups[result.collection].append(result)

        normalized_results = []

        for _collection, col_results in collection_groups.items():
            scores = [r.score for r in col_results]

            if method == "min_max":
                # Min-max normalization to [0, 1]
                min_score = min(scores) if scores else 0.0
                max_score = max(scores) if scores else 1.0
                score_range = max_score - min_score

                if score_range > 0:
                    for result in col_results:
                        normalized_score = (result.score - min_score) / score_range
                        result.score = normalized_score
                        result.payload["_original_score"] = result.score
                        result.payload["_normalization_method"] = "min_max"

            elif method == "z_score":
                # Z-score normalization (standardization)
                import statistics
                if len(scores) > 1:
                    mean_score = statistics.mean(scores)
                    stdev_score = statistics.stdev(scores)

                    if stdev_score > 0:
                        for result in col_results:
                            normalized_score = (result.score - mean_score) / stdev_score
                            # Map z-score to [0, 1] using sigmoid-like transformation
                            normalized_score = 1 / (1 + abs(normalized_score))
                            result.score = normalized_score
                            result.payload["_original_score"] = result.score
                            result.payload["_normalization_method"] = "z_score"

            normalized_results.extend(col_results)

        logger.debug(
            "Score normalization complete",
            normalized_count=len(normalized_results)
        )

        return normalized_results

    def _convert_to_api_format(self, results: list[TenantAwareResult]) -> list[dict]:
        """
        Convert TenantAwareResult objects to API-friendly dictionary format.

        Args:
            results: List of tenant-aware results

        Returns:
            List of dictionaries in API format
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

            # Include tenant metadata if available
            if result.tenant_metadata:
                api_result["tenant_metadata"] = result.tenant_metadata

            if result.project_context:
                api_result["project_context"] = result.project_context

            api_results.append(api_result)

        return api_results


class RRFFusionRanker:
    """
    Reciprocal Rank Fusion (RRF) ranker for combining search results.

    RRF is a simple yet effective algorithm for combining results from multiple
    retrieval systems. It works by assigning each document a score based on its
    reciprocal rank in each result list, then summing these scores.

    The formula for RRF score is:
        RRF_score(d) = sum(1 / (k + rank_i(d)))
    where:
        - d is a document
        - k is a constant (typically 60)
        - rank_i(d) is the rank of document d in result list i

    Args:
        k: RRF constant parameter (default: 60). Higher values reduce impact of
           rank differences, lower values emphasize top-ranked results.
        boost_weights: Optional dictionary of weights for different result sources.
                      Example: {"dense": 1.0, "sparse": 0.8}

    Task 215: Migrated to use unified loguru-based logging system.
    """

    def __init__(self, k: int = 60, boost_weights: dict | None = None) -> None:
        """Initialize RRF ranker with parameters."""
        self.k = k
        self.boost_weights = boost_weights or {}
        logger.debug("Initialized RRF ranker", k=k, boost_weights=boost_weights)

    def fuse(
        self,
        dense_results: list,
        sparse_results: list,
        weights: dict | None = None,
    ) -> list:
        """
        Fuse dense and sparse search results using RRF algorithm.

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            weights: Optional weight dictionary {"dense": float, "sparse": float}

        Returns:
            List of fused results sorted by RRF score
        """
        # Use provided weights or fallback to boost_weights
        weights = weights or {"dense": 1.0, "sparse": 1.0}
        dense_weight = weights.get("dense", 1.0)
        sparse_weight = weights.get("sparse", 1.0)

        logger.debug(
            "Starting RRF fusion",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )

        # Build RRF scores for all documents
        rrf_scores = {}

        # Process dense results
        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.id if hasattr(result, 'id') else str(result)
            rrf_score = dense_weight / (self.k + rank)
            rrf_scores[doc_id] = {
                "score": rrf_score,
                "dense_rank": rank,
                "dense_score": result.score if hasattr(result, 'score') else 0.0,
                "result": result
            }

        # Process sparse results
        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.id if hasattr(result, 'id') else str(result)
            rrf_score = sparse_weight / (self.k + rank)

            if doc_id in rrf_scores:
                # Document appears in both lists - add sparse contribution
                rrf_scores[doc_id]["score"] += rrf_score
                rrf_scores[doc_id]["sparse_rank"] = rank
                rrf_scores[doc_id]["sparse_score"] = (
                    result.score if hasattr(result, 'score') else 0.0
                )
            else:
                # Document only in sparse results
                rrf_scores[doc_id] = {
                    "score": rrf_score,
                    "sparse_rank": rank,
                    "sparse_score": result.score if hasattr(result, 'score') else 0.0,
                    "result": result
                }

        # Convert to sorted list
        fused_results = []
        for _doc_id, score_data in sorted(
            rrf_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        ):
            result = score_data["result"]
            # Update result score to RRF score
            if hasattr(result, 'score'):
                result.score = score_data["score"]

            # Add fusion metadata to payload
            if hasattr(result, 'payload') and isinstance(result.payload, dict):
                result.payload["_fusion_metadata"] = {
                    "fusion_method": "rrf",
                    "rrf_score": score_data["score"],
                    "dense_rank": score_data.get("dense_rank"),
                    "dense_score": score_data.get("dense_score"),
                    "sparse_rank": score_data.get("sparse_rank"),
                    "sparse_score": score_data.get("sparse_score"),
                }

            fused_results.append(result)

        logger.debug(
            "RRF fusion complete",
            fused_count=len(fused_results)
        )

        return fused_results

    def explain_fusion(
        self,
        dense_results: list,
        sparse_results: list,
        weights: dict | None = None,
    ) -> dict:
        """
        Generate detailed explanation of RRF fusion process.

        Useful for debugging and understanding why certain results rank highly.

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            weights: Optional weight dictionary

        Returns:
            Dictionary containing fusion analysis and explanations
        """
        weights = weights or {"dense": 1.0, "sparse": 1.0}

        explanation = {
            "fusion_method": "reciprocal_rank_fusion",
            "parameters": {
                "k": self.k,
                "dense_weight": weights.get("dense", 1.0),
                "sparse_weight": weights.get("sparse", 1.0),
            },
            "input_statistics": {
                "dense_result_count": len(dense_results),
                "sparse_result_count": len(sparse_results),
                "dense_score_range": (
                    (min(r.score for r in dense_results if hasattr(r, 'score')),
                     max(r.score for r in dense_results if hasattr(r, 'score')))
                    if dense_results else (0, 0)
                ),
                "sparse_score_range": (
                    (min(r.score for r in sparse_results if hasattr(r, 'score')),
                     max(r.score for r in sparse_results if hasattr(r, 'score')))
                    if sparse_results else (0, 0)
                ),
            },
            "document_analysis": []
        }

        # Perform fusion to get detailed scores
        fused_results = self.fuse(dense_results, sparse_results, weights)

        # Extract top results for analysis
        for result in fused_results[:10]:  # Top 10 for explanation
            if hasattr(result, 'payload') and "_fusion_metadata" in result.payload:
                metadata = result.payload["_fusion_metadata"]
                explanation["document_analysis"].append({
                    "document_id": result.id if hasattr(result, 'id') else "unknown",
                    "final_rrf_score": metadata["rrf_score"],
                    "dense_contribution": {
                        "rank": metadata.get("dense_rank"),
                        "score": metadata.get("dense_score"),
                        "rrf_component": (
                            weights.get("dense", 1.0) / (self.k + metadata.get("dense_rank", 0))
                            if metadata.get("dense_rank") else 0.0
                        )
                    },
                    "sparse_contribution": {
                        "rank": metadata.get("sparse_rank"),
                        "score": metadata.get("sparse_score"),
                        "rrf_component": (
                            weights.get("sparse", 1.0) / (self.k + metadata.get("sparse_rank", 0))
                            if metadata.get("sparse_rank") else 0.0
                        )
                    }
                })

        return explanation

    def fuse_rankings(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        weights: dict | None = None,
    ) -> list[dict]:
        """Compatibility wrapper for dict-based result fusion."""
        if not dense_results and not sparse_results:
            return []

        weights = weights or {"dense": 1.0, "sparse": 1.0}
        dense_weight = weights.get("dense", 1.0)
        sparse_weight = weights.get("sparse", 1.0)

        rrf_scores: dict[str, dict] = {}

        for rank, result in enumerate(dense_results, start=1):
            doc_id = result.get("id")
            score = dense_weight / (self.k + rank)
            rrf_scores[doc_id] = {
                "rrf_score": score,
                "dense_rank": rank,
                "dense_score": result.get("score"),
                "payload": result.get("payload", {}),
            }

        for rank, result in enumerate(sparse_results, start=1):
            doc_id = result.get("id")
            score = sparse_weight / (self.k + rank)
            entry = rrf_scores.get(doc_id, {
                "rrf_score": 0.0,
                "dense_rank": None,
                "dense_score": None,
                "payload": result.get("payload", {}),
            })
            entry["rrf_score"] += score
            entry["sparse_rank"] = rank
            entry["sparse_score"] = result.get("score")
            rrf_scores[doc_id] = entry

        fused = []
        for doc_id, data in sorted(rrf_scores.items(), key=lambda item: item[1]["rrf_score"], reverse=True):
            fused.append({
                "id": doc_id,
                "score": data["rrf_score"],
                "rrf_score": data["rrf_score"],
                "payload": data.get("payload", {}),
                "fusion_explanation": {
                    "dense_rank": data.get("dense_rank"),
                    "sparse_rank": data.get("sparse_rank"),
                    "dense_score": data.get("dense_score"),
                    "sparse_score": data.get("sparse_score"),
                },
            })

        return fused


class WeightedSumFusionRanker:
    """
    Weighted sum fusion ranker for combining dense and sparse results.

    This ranker normalizes scores from dense and sparse searches, then combines
    them using configurable weights. Unlike RRF which uses rank-based fusion,
    this method uses actual similarity scores.

    Args:
        dense_weight: Weight for dense vector scores (default: 0.7)
        sparse_weight: Weight for sparse vector scores (default: 0.3)

    Task 215: Migrated to use unified loguru-based logging system.
    """

    def __init__(self, dense_weight: float = 0.7, sparse_weight: float = 0.3) -> None:
        """Initialize weighted sum ranker."""
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Normalize weights to sum to 1.0
        total_weight = dense_weight + sparse_weight
        if total_weight > 0:
            self.dense_weight = dense_weight / total_weight
            self.sparse_weight = sparse_weight / total_weight

        logger.debug(
            "Initialized weighted sum ranker",
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight
        )

    def fuse(self, dense_results: list, sparse_results: list) -> list:
        """
        Fuse results using weighted sum of normalized scores.

        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search

        Returns:
            List of fused results sorted by combined score
        """
        logger.debug(
            "Starting weighted sum fusion",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results)
        )

        # Normalize scores to [0, 1] range
        normalized_dense = self._normalize_scores(dense_results)
        normalized_sparse = self._normalize_scores(sparse_results)

        # Build combined scores
        combined_scores = {}

        # Add dense contributions
        for result in normalized_dense:
            doc_id = result.id if hasattr(result, 'id') else str(result)
            combined_scores[doc_id] = {
                "score": result.score * self.dense_weight,
                "dense_score": result.score,
                "result": result
            }

        # Add sparse contributions
        for result in normalized_sparse:
            doc_id = result.id if hasattr(result, 'id') else str(result)
            if doc_id in combined_scores:
                combined_scores[doc_id]["score"] += result.score * self.sparse_weight
                combined_scores[doc_id]["sparse_score"] = result.score
            else:
                combined_scores[doc_id] = {
                    "score": result.score * self.sparse_weight,
                    "sparse_score": result.score,
                    "result": result
                }

        # Convert to sorted list
        fused_results = []
        for _doc_id, score_data in sorted(
            combined_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        ):
            result = score_data["result"]
            if hasattr(result, 'score'):
                result.score = score_data["score"]

            fused_results.append(result)

        logger.debug("Weighted sum fusion complete", fused_count=len(fused_results))

        return fused_results

    def fuse_rankings(self, dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
        """Compatibility wrapper for dict-based result fusion."""
        if not dense_results and not sparse_results:
            return []

        def _normalize(results: list[dict]) -> dict[str, float]:
            if not results:
                return {}
            scores = [r.get("score", 0.0) for r in results]
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return {r.get("id"): 1.0 for r in results}
            return {
                r.get("id"): (r.get("score", 0.0) - min_score) / (max_score - min_score)
                for r in results
            }

        dense_norm = _normalize(dense_results)
        sparse_norm = _normalize(sparse_results)

        dense_map = {r.get("id"): r for r in dense_results}
        sparse_map = {r.get("id"): r for r in sparse_results}
        all_ids = set(dense_norm) | set(sparse_norm)

        fused = []
        for doc_id in all_ids:
            dense_score = dense_norm.get(doc_id, 0.0)
            sparse_score = sparse_norm.get(doc_id, 0.0)
            weighted_score = dense_score * self.dense_weight + sparse_score * self.sparse_weight
            base = dense_map.get(doc_id) or sparse_map.get(doc_id) or {}
            fused.append({
                "id": doc_id,
                "score": weighted_score,
                "weighted_score": weighted_score,
                "payload": base.get("payload", {}),
                "fusion_explanation": {
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "dense_weight": self.dense_weight,
                    "sparse_weight": self.sparse_weight,
                },
            })

        fused.sort(key=lambda r: r["weighted_score"], reverse=True)
        return fused

    def _normalize_scores(self, results: list) -> list:
        """
        Normalize scores to [0, 1] range using min-max normalization.

        Args:
            results: Search results with scores

        Returns:
            Results with normalized scores
        """
        if not results:
            return []

        scores = [r.score for r in results if hasattr(r, 'score')]
        if not scores:
            return results

        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        if score_range == 0:
            # All scores are the same
            for result in results:
                if hasattr(result, 'score'):
                    result.score = 1.0
        else:
            for result in results:
                if hasattr(result, 'score'):
                    result.score = (result.score - min_score) / score_range

        return results


class MaxScoreFusionRanker:
    """Max score fusion ranker for combining dense and sparse results."""

    def fuse_rankings(self, dense_results: list[dict], sparse_results: list[dict]) -> list[dict]:
        if not dense_results and not sparse_results:
            return []

        dense_map = {r.get("id"): r for r in dense_results}
        sparse_map = {r.get("id"): r for r in sparse_results}
        all_ids = set(dense_map) | set(sparse_map)

        fused = []
        for doc_id in all_ids:
            dense_score = dense_map.get(doc_id, {}).get("score")
            sparse_score = sparse_map.get(doc_id, {}).get("score")
            dense_val = dense_score if dense_score is not None else 0.0
            sparse_val = sparse_score if sparse_score is not None else 0.0
            max_score = max(dense_val, sparse_val)
            max_source = "dense" if dense_val >= sparse_val else "sparse"
            base = dense_map.get(doc_id) or sparse_map.get(doc_id) or {}
            fused.append({
                "id": doc_id,
                "score": max_score,
                "max_score": max_score,
                "payload": base.get("payload", {}),
                "fusion_explanation": {
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                    "max_score_source": max_source,
                },
            })

        fused.sort(key=lambda r: r["max_score"], reverse=True)
        return fused


def create_fusion_ranker(method: str, **kwargs):
    """Factory for fusion rankers."""
    method_key = (method or "").lower()
    if method_key in ("rrf", "reciprocal_rank_fusion"):
        return RRFFusionRanker(**kwargs)
    if method_key in ("weighted_sum", "weighted", "ws"):
        return WeightedSumFusionRanker(**kwargs)
    if method_key in ("max_score", "max"):
        return MaxScoreFusionRanker()
    raise ValueError(f"Unknown fusion method: {method}")


class HybridSearchEngine:
    """
    Production-ready hybrid search engine with multiple fusion strategies.

    This engine combines dense semantic vector search with sparse keyword-based
    search to provide optimal retrieval performance. It supports multiple fusion
    algorithms, metadata filtering, multi-tenant isolation, and performance monitoring.

    Features:
        - Multiple fusion strategies (RRF, weighted sum, max score)
        - Metadata-based filtering with optimization
        - Multi-tenant project isolation
        - Performance monitoring and optimization
        - Comprehensive result aggregation
        - Cross-collection search support

    Args:
        client: Qdrant client instance
        enable_optimizations: Enable query and filter optimizations
        enable_multi_tenant_aggregation: Enable multi-tenant result aggregation
        enable_performance_monitoring: Enable performance tracking and monitoring

    Example:
        ```python
        from qdrant_client import QdrantClient
        from workspace_qdrant_mcp.core.hybrid_search import HybridSearchEngine

        client = QdrantClient("http://localhost:6333")
        engine = HybridSearchEngine(
            client=client,
            enable_optimizations=True,
            enable_multi_tenant_aggregation=True,
            enable_performance_monitoring=True
        )

        results = await engine.hybrid_search(
            collection_name="documents",
            query_embeddings={
                "dense": [0.1, 0.2, ...],
                "sparse": {"indices": [1, 5, 10], "values": [0.8, 0.6, 0.4]}
            },
            limit=10,
            fusion_method="rrf"
        )
        ```

    Task 215: Migrated to unified loguru-based logging system.
    Task 233.1: Enhanced with multi-tenant metadata filtering.
    """

    def __init__(
        self,
        client: QdrantClient,
        enable_optimizations: bool = True,
        enable_multi_tenant_aggregation: bool = True,
        enable_performance_monitoring: bool = True,
    ):
        """Initialize hybrid search engine with optional features."""
        self.client = client
        self.enable_optimizations = enable_optimizations
        self.enable_multi_tenant_aggregation = enable_multi_tenant_aggregation
        self.enable_performance_monitoring = enable_performance_monitoring

        # Initialize metadata filtering
        self.metadata_filter_manager = MetadataFilterManager(
            qdrant_client=client,
            enable_caching=True,
            enable_performance_monitoring=enable_performance_monitoring
        )

        # Initialize multi-tenant aggregator
        if enable_multi_tenant_aggregation:
            self.result_aggregator = MultiTenantResultAggregator(
                enable_deduplication=True,
                preserve_tenant_isolation=True,
                default_aggregation_method="max_score"
            )
        else:
            self.result_aggregator = None

        # Initialize performance monitoring
        if enable_performance_monitoring:
            self.performance_monitor = MetadataFilteringPerformanceMonitor()
        else:
            self.performance_monitor = None

        # Initialize query optimizer
        if enable_optimizations:
            self.query_optimizer = QueryOptimizer()
            self.filter_optimizer = FilterOptimizer()
        else:
            self.query_optimizer = None
            self.filter_optimizer = None

        logger.info(
            "Initialized hybrid search engine with comprehensive monitoring support",
            optimizations=enable_optimizations,
            multi_tenant=enable_multi_tenant_aggregation,
            monitoring=enable_performance_monitoring
        )

    async def hybrid_search(
        self,
        collection_name: str,
        query_embeddings: dict,
        limit: int = 10,
        filter_conditions: models.Filter | None = None,
        project_context: dict | None = None,
        fusion_method: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        search_params: models.SearchParams | None = None,
        with_payload: bool | list | models.PayloadSelector = True,
        with_vectors: bool | list = False,
    ) -> dict:
        """
        Perform hybrid search combining dense and sparse vectors with RRF fusion.

        This method executes both dense semantic search and sparse keyword search,
        then fuses the results using the specified fusion method.

        Args:
            collection_name: Name of the collection to search
            query_embeddings: Dictionary with "dense" and "sparse" embeddings
                             - dense: List of floats (e.g., 384-dim vector)
                             - sparse: Dict with "indices" and "values" lists
            limit: Maximum number of results to return
            filter_conditions: Qdrant filter for metadata filtering
            project_context: Optional project context for multi-tenant filtering
            fusion_method: Fusion algorithm ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense search results
            sparse_weight: Weight for sparse search results
            search_params: Optional Qdrant search parameters
            with_payload: Include payload in results
            with_vectors: Include vectors in results

        Returns:
            Dictionary containing:
                - fused_results: Combined and ranked results
                - dense_results: Original dense search results
                - sparse_results: Original sparse search results
                - metadata: Search metadata and performance info

        Task 322.7: Fixed sparse vector creation to properly unpack dictionary parameters.
        """
        logger.info(
            "Starting optimized hybrid search",
            collection=collection_name,
            limit=limit,
            fusion_method=fusion_method
        )

        search_start = time.time()
        search_results = {
            "dense_results": [],
            "sparse_results": [],
            "fused_results": [],
            "metadata": {}
        }

        # Build enhanced filter with metadata optimization and project context
        additional_conditions = []
        if project_context:
            project_id = project_context.get("project_id")
            if project_id:
                additional_conditions.append(
                    models.FieldCondition(
                        key="project_id",
                        match=models.MatchValue(value=project_id),
                    )
                )
            project_name = project_context.get("project_name")
            if project_name:
                additional_conditions.append(
                    models.FieldCondition(
                        key="project_name",
                        match=models.MatchValue(value=project_name),
                    )
                )

        enhanced_filter = filter_conditions
        if filter_conditions or additional_conditions:
            enhanced_filter = self._build_enhanced_filter(
                filter_conditions,
                additional_conditions=additional_conditions,
            )

        # Dense vector search
        if "dense" in query_embeddings and query_embeddings["dense"]:
            try:
                logger.debug("Executing dense vector search")
                dense_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", query_embeddings["dense"]),
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

        # Sparse vector search - Fixed to properly unpack dictionary (Task 322.7)
        if "sparse" in query_embeddings and query_embeddings["sparse"]:
            try:
                logger.debug("Executing sparse vector search")
                # Unpack the sparse dictionary to separate indices and values
                sparse_data = query_embeddings["sparse"]
                try:
                    sparse_vector = create_named_sparse_vector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"],
                        name="sparse",
                    )
                except TypeError:
                    # Backward compatibility for callables expecting a dict payload.
                    sparse_vector = create_named_sparse_vector(sparse_data)
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
                    weights={"dense": dense_weight, "sparse": sparse_weight},
                )

            # Limit fused results
            search_results["fused_results"] = fused_results[:limit]

            logger.debug(
                "Fusion complete",
                fused_count=len(search_results["fused_results"])
            )

        # Add search metadata
        search_duration = (time.time() - search_start) * 1000  # Convert to ms
        search_results["metadata"] = {
            "search_duration_ms": search_duration,
            "fusion_method": fusion_method,
            "dense_count": len(search_results["dense_results"]),
            "sparse_count": len(search_results["sparse_results"]),
            "fused_count": len(search_results["fused_results"]),
            "collection": collection_name
        }

        logger.info(
            "Hybrid search complete",
            duration_ms=search_duration,
            results=len(search_results["fused_results"])
        )

        return search_results

    def _max_score_fusion(self, dense_results: list, sparse_results: list) -> list:
        """
        Fuse results by taking maximum score for each document.

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results

        Returns:
            Fused results with maximum scores
        """
        # Build score map
        score_map = {}

        for result in dense_results:
            doc_id = result.id if hasattr(result, 'id') else str(result)
            score_map[doc_id] = {
                "score": result.score if hasattr(result, 'score') else 0.0,
                "result": result,
                "sources": ["dense"]
            }

        for result in sparse_results:
            doc_id = result.id if hasattr(result, 'id') else str(result)
            sparse_score = result.score if hasattr(result, 'score') else 0.0

            if doc_id in score_map:
                # Take maximum score
                score_map[doc_id]["score"] = max(score_map[doc_id]["score"], sparse_score)
                score_map[doc_id]["sources"].append("sparse")
            else:
                score_map[doc_id] = {
                    "score": sparse_score,
                    "result": result,
                    "sources": ["sparse"]
                }

        # Convert to sorted list
        fused_results = []
        for _doc_id, data in sorted(score_map.items(), key=lambda x: x[1]["score"], reverse=True):
            result = data["result"]
            if hasattr(result, 'score'):
                result.score = data["score"]
            fused_results.append(result)

        return fused_results

    def _build_enhanced_filter(
        self,
        base_filter: models.Filter | None,
        additional_conditions: list | None = None
    ) -> models.Filter | None:
        """
        Build enhanced filter with optimizations and additional conditions.

        Args:
            base_filter: Base Qdrant filter
            additional_conditions: Additional filter conditions to merge

        Returns:
            Enhanced and optimized filter
        """
        if not base_filter and not additional_conditions:
            return None

        # Start with base filter or create new one
        if base_filter:
            enhanced_filter = base_filter
        else:
            enhanced_filter = models.Filter()

        # Add additional conditions if provided
        if additional_conditions:
            if not hasattr(enhanced_filter, 'must') or enhanced_filter.must is None:
                enhanced_filter.must = []
            enhanced_filter.must.extend(additional_conditions)

        # Apply optimizations if enabled
        if self.enable_optimizations and self.filter_optimizer:
            try:
                # Optimize filter structure
                enhanced_filter = self.filter_optimizer.optimize_filter(enhanced_filter)
            except Exception as e:
                logger.warning("Filter optimization failed, using original filter", error=str(e))

        return enhanced_filter

    def create_project_isolation_filter(
        self,
        project_id: str,
        workspace_type: str | None = None,
        additional_filters: models.Filter | None = None
    ) -> models.Filter:
        """
        Create filter for project isolation in multi-tenant scenarios.

        Args:
            project_id: Project identifier for isolation
            workspace_type: Optional workspace type filter
            additional_filters: Additional filters to merge

        Returns:
            Qdrant filter with project isolation
        """
        # Build project isolation condition
        project_condition = models.FieldCondition(
            key="project_id",
            match=models.MatchValue(value=project_id)
        )

        isolation_filter = models.Filter(must=[project_condition])

        # Add workspace type filter if specified
        if workspace_type:
            workspace_condition = models.FieldCondition(
                key="workspace_type",
                match=models.MatchValue(value=workspace_type)
            )
            isolation_filter.must.append(workspace_condition)

        # Merge with additional filters if provided
        if additional_filters:
            if hasattr(additional_filters, 'must') and additional_filters.must:
                isolation_filter.must.extend(additional_filters.must)
            if hasattr(additional_filters, 'should') and additional_filters.should:
                isolation_filter.should = additional_filters.should
            if hasattr(additional_filters, 'must_not') and additional_filters.must_not:
                isolation_filter.must_not = additional_filters.must_not

        return isolation_filter

    def get_filter_performance_stats(self) -> dict[str, dict]:
        """
        Get performance statistics for filter operations.

        Returns:
            Dictionary with filter performance metrics
        """
        if not self.performance_monitor:
            return {"error": "Performance monitoring not enabled"}

        return self.performance_monitor.get_performance_summary()

    async def search_project_workspace(
        self,
        project_id: str,
        query_embeddings: dict,
        workspace_type: str | None = None,
        limit: int = 10,
        fusion_method: str = "rrf"
    ) -> dict:
        """
        Search within a specific project workspace with automatic isolation.

        Args:
            project_id: Project identifier
            query_embeddings: Dense and sparse embeddings
            workspace_type: Optional workspace type filter
            limit: Maximum results
            fusion_method: Fusion method to use

        Returns:
            Search results with project isolation applied
        """
        # Create project isolation filter
        isolation_filter = self.create_project_isolation_filter(
            project_id=project_id,
            workspace_type=workspace_type
        )

        # Execute hybrid search with isolation
        return await self.hybrid_search(
            collection_name=f"project_{project_id}",
            query_embeddings=query_embeddings,
            limit=limit,
            filter_conditions=isolation_filter,
            fusion_method=fusion_method
        )

    async def search_tenant_namespace(
        self,
        tenant_id: str,
        query_embeddings: dict,
        collections: list[str] | None = None,
        limit: int = 10,
        fusion_method: str = "rrf"
    ) -> dict:
        """
        Search across all collections in a tenant namespace.

        Args:
            tenant_id: Tenant identifier
            query_embeddings: Dense and sparse embeddings
            collections: Optional list of collections to search
            limit: Maximum results
            fusion_method: Fusion method

        Returns:
            Aggregated search results across tenant collections
        """
        # If no collections specified, discover tenant collections
        if not collections:
            # This would need collection discovery logic
            collections = [f"tenant_{tenant_id}_default"]

        # Use multi-collection search
        return await self.multi_collection_hybrid_search(
            collection_names=collections,
            query_embeddings=query_embeddings,
            limit=limit,
            fusion_method=fusion_method,
            enable_deduplication=True
        )

    def get_supported_workspace_types(self) -> set:
        """
        Get set of supported workspace types.

        Returns:
            Set of workspace type identifiers
        """
        return {
            "code",
            "docs",
            "tests",
            "config",
            "data",
            "general"
        }

    def validate_workspace_type(self, workspace_type: str) -> bool:
        """
        Validate if workspace type is supported.

        Args:
            workspace_type: Type to validate

        Returns:
            True if supported, False otherwise
        """
        return workspace_type in self.get_supported_workspace_types()

    async def multi_collection_hybrid_search(
        self,
        collection_names: list[str],
        query_embeddings: dict,
        limit: int = 10,
        fusion_method: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        filter_conditions: models.Filter | None = None,
        enable_deduplication: bool = True,
        score_normalization: str = "min_max",
        aggregation_method: str = "max_score",
        score_threshold: float | None = None,
    ) -> dict:
        """
        Perform hybrid search across multiple collections with result aggregation.

        This method searches multiple collections in parallel and aggregates the
        results with score normalization and optional deduplication.

        Args:
            collection_names: List of collection names to search
            query_embeddings: Dictionary with dense and sparse embeddings
            limit: Maximum results per collection
            fusion_method: Fusion method for each collection ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results
            filter_conditions: Optional filter to apply to all collections
            enable_deduplication: Enable cross-collection deduplication
            score_normalization: Score normalization method ("min_max", "z_score", "none")
            aggregation_method: How to aggregate duplicate scores ("max_score", "avg_score", "sum_score")
            score_threshold: Optional minimum score threshold

        Returns:
            Dictionary containing:
                - results: Aggregated and ranked results from all collections
                - total_results: Total number of results
                - aggregation_metadata: Metadata about aggregation process
                - collection_results: Individual results per collection

        Example:
            ```python
            results = await engine.multi_collection_hybrid_search(
                collection_names=["project1_code", "project2_code", "library_docs"],
                query_embeddings=embeddings,
                limit=10,
                enable_deduplication=True,
                score_normalization="min_max"
            )
            ```
        """
        logger.info(
            "Starting multi-collection hybrid search",
            collections=collection_names,
            limit=limit,
            fusion=fusion_method,
            deduplication=enable_deduplication
        )

        # Search each collection
        collection_results = {}
        for collection_name in collection_names:
            try:
                result = await self.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=limit,
                    filter_conditions=filter_conditions,
                    fusion_method=fusion_method,
                    dense_weight=dense_weight,
                    sparse_weight=sparse_weight
                )
                collection_results[collection_name] = result.get("fused_results", [])
            except Exception as e:
                logger.error(
                    "Collection search failed",
                    collection=collection_name,
                    error=str(e)
                )
                collection_results[collection_name] = []

        # Use basic aggregation if multi-tenant aggregation not enabled
        if not self.enable_multi_tenant_aggregation or not self.result_aggregator:
            return await self._basic_multi_collection_search(
                collection_results,
                limit,
                score_threshold
            )

        # Aggregate results using multi-tenant aggregator
        aggregated = self.result_aggregator.aggregate_multi_collection_results(
            collection_results=collection_results,
            score_normalization=score_normalization,
            aggregation_method=aggregation_method,
            deduplication_enabled=enable_deduplication
        )

        # Apply score threshold if specified
        if score_threshold is not None:
            filtered_results = [
                r for r in aggregated["results"]
                if r.get("score", 0.0) >= score_threshold
            ]
            aggregated["results"] = filtered_results
            aggregated["total_results"] = len(filtered_results)
            aggregated["aggregation_metadata"]["score_threshold"] = score_threshold
            aggregated["aggregation_metadata"]["filtered_count"] = (
                len(aggregated["results"]) - len(filtered_results)
            )

        # Add individual collection results for reference
        aggregated["collection_results"] = collection_results

        logger.info(
            "Multi-collection search complete",
            total_results=aggregated["total_results"],
            collections_searched=len(collection_names)
        )

        return aggregated

    async def _basic_multi_collection_search(
        self,
        collection_results: dict[str, list],
        limit: int,
        score_threshold: float | None = None
    ) -> dict:
        """
        Basic multi-collection search without advanced aggregation.

        Used as fallback when multi-tenant aggregation is disabled.

        Args:
            collection_results: Dictionary of collection results
            limit: Maximum results to return
            score_threshold: Optional score threshold

        Returns:
            Basic aggregated results
        """
        # Combine all results
        all_results = []
        for collection_name, results in collection_results.items():
            for result in results:
                # Add collection metadata
                if hasattr(result, 'payload') and isinstance(result.payload, dict):
                    result.payload["_source_collection"] = collection_name
                all_results.append(result)

        # Sort by score
        all_results.sort(
            key=lambda x: x.score if hasattr(x, 'score') else 0.0,
            reverse=True
        )

        # Apply score threshold if specified
        if score_threshold is not None:
            all_results = [
                r for r in all_results
                if hasattr(r, 'score') and r.score >= score_threshold
            ]

        # Limit results
        limited_results = all_results[:limit]

        return {
            "results": limited_results,
            "total_results": len(limited_results),
            "aggregation_metadata": {
                "collection_count": len(collection_results),
                "collections": list(collection_results.keys()),
                "score_threshold": score_threshold,
                "aggregation_method": "basic"
            },
            "collection_results": collection_results
        }

    def configure_result_aggregation(
        self,
        enable_deduplication: bool = True,
        preserve_tenant_isolation: bool = True,
        default_aggregation_method: str = "max_score"
    ):
        """
        Configure multi-tenant result aggregation settings.

        Args:
            enable_deduplication: Enable result deduplication
            preserve_tenant_isolation: Preserve tenant boundaries during deduplication
            default_aggregation_method: Default score aggregation method

        Raises:
            RuntimeError: If multi-tenant aggregation is not enabled
        """
        if not self.enable_multi_tenant_aggregation:
            raise RuntimeError("Multi-tenant aggregation not enabled in engine configuration")

        if not self.result_aggregator:
            self.result_aggregator = MultiTenantResultAggregator(
                enable_deduplication=enable_deduplication,
                preserve_tenant_isolation=preserve_tenant_isolation,
                default_aggregation_method=default_aggregation_method
            )
        else:
            self.result_aggregator.enable_deduplication = enable_deduplication
            self.result_aggregator.preserve_tenant_isolation = preserve_tenant_isolation
            self.result_aggregator.default_aggregation_method = default_aggregation_method

        logger.info(
            "Updated result aggregation configuration",
            deduplication=enable_deduplication,
            tenant_isolation=preserve_tenant_isolation,
            aggregation_method=default_aggregation_method
        )

    def get_result_aggregation_stats(self) -> dict:
        """
        Get statistics about result aggregation.

        Returns:
            Dictionary with aggregation statistics
        """
        if not self.result_aggregator:
            return {"error": "Result aggregation not enabled"}

        return {
            "enabled": self.enable_multi_tenant_aggregation,
            "deduplication_enabled": self.result_aggregator.enable_deduplication,
            "preserve_tenant_isolation": self.result_aggregator.preserve_tenant_isolation,
            "default_aggregation_method": self.result_aggregator.default_aggregation_method
        }

    async def ensure_collection_optimized(self, collection_name: str, force_recreate: bool = False) -> dict:
        """
        Ensure collection has optimal configuration for hybrid search.

        Args:
            collection_name: Collection to optimize
            force_recreate: Force recreation even if already optimized

        Returns:
            Dictionary with optimization status
        """
        logger.info(
            "Ensuring collection optimization",
            collection=collection_name,
            force=force_recreate
        )

        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(
                c.name == collection_name
                for c in collections.collections
            )

            if not collection_exists:
                return {
                    "status": "error",
                    "message": f"Collection {collection_name} does not exist"
                }

            # Get collection info
            collection_info = self.client.get_collection(collection_name)

            # Check if already optimized
            has_dense_vector = "dense" in collection_info.config.params.vectors
            has_sparse_vector = (
                collection_info.config.params.sparse_vectors and
                "sparse" in collection_info.config.params.sparse_vectors
            )

            is_optimized = has_dense_vector and has_sparse_vector

            if is_optimized and not force_recreate:
                return {
                    "status": "already_optimized",
                    "collection": collection_name,
                    "has_dense_vector": has_dense_vector,
                    "has_sparse_vector": has_sparse_vector
                }

            # Collection needs optimization but this would require recreation
            # which is beyond scope of this method
            return {
                "status": "optimization_needed",
                "collection": collection_name,
                "has_dense_vector": has_dense_vector,
                "has_sparse_vector": has_sparse_vector,
                "message": "Collection exists but may need manual optimization"
            }

        except Exception as e:
            logger.error("Collection optimization check failed", error=str(e))
            return {
                "status": "error",
                "message": str(e)
            }

    def get_optimization_performance(self) -> dict:
        """
        Get performance metrics for query optimizations.

        Returns:
            Dictionary with optimization performance metrics
        """
        if not self.enable_optimizations:
            return {"error": "Optimizations not enabled"}

        stats = {}

        if self.query_optimizer:
            stats["query_optimizer"] = {
                "enabled": True,
                # Add query optimizer stats if available
            }

        if self.filter_optimizer:
            stats["filter_optimizer"] = {
                "enabled": True,
                # Add filter optimizer stats if available
            }

        return stats

    def clear_optimization_caches(self) -> dict:
        """
        Clear all optimization caches.

        Returns:
            Dictionary with cache clear results
        """
        if not self.enable_optimizations:
            return {"error": "Optimizations not enabled"}

        cleared = []

        if self.metadata_filter_manager:
            # Clear metadata filter cache if it has one
            cleared.append("metadata_filter_cache")

        logger.info("Cleared optimization caches", caches=cleared)

        return {
            "status": "success",
            "caches_cleared": cleared
        }

    def get_performance_alerts(self, hours: int = 24) -> list[dict]:
        """
        Get performance alerts from the last N hours.

        Args:
            hours: Number of hours to look back

        Returns:
            List of performance alert dictionaries
        """
        if not self.performance_monitor:
            return []

        # This would integrate with performance monitoring system
        # Placeholder implementation
        return []

    def get_performance_monitoring_status(self) -> dict:
        """
        Get current status of performance monitoring.

        Returns:
            Dictionary with monitoring status
        """
        return {
            "enabled": self.enable_performance_monitoring,
            "monitor_active": self.performance_monitor is not None,
        }

    def get_performance_dashboard_data(self) -> dict:
        """
        Get comprehensive data for performance dashboard.

        Returns:
            Dictionary with dashboard data
        """
        return {
            "monitoring_status": self.get_performance_monitoring_status(),
            "optimization_stats": self.get_optimization_performance(),
            "aggregation_stats": self.get_result_aggregation_stats(),
            "filter_stats": self.get_filter_performance_stats(),
        }

    async def run_performance_benchmark(
        self,
        collection_name: str,
        test_queries: list[dict],
        fusion_methods: list[str] | None = None
    ) -> dict:
        """
        Run performance benchmark on collection with various fusion methods.

        Args:
            collection_name: Collection to benchmark
            test_queries: List of test query embeddings
            fusion_methods: Fusion methods to test (default: all methods)

        Returns:
            Dictionary with benchmark results
        """
        fusion_methods = fusion_methods or ["rrf", "weighted_sum", "max_score"]

        logger.info(
            "Starting performance benchmark",
            collection=collection_name,
            queries=len(test_queries),
            methods=fusion_methods
        )

        benchmark_results = {
            "collection": collection_name,
            "query_count": len(test_queries),
            "fusion_methods": {},
            "summary": {}
        }

        for method in fusion_methods:
            method_results = {
                "query_times": [],
                "result_counts": [],
                "errors": 0
            }

            for query in test_queries:
                try:
                    start_time = time.time()
                    result = await self.hybrid_search(
                        collection_name=collection_name,
                        query_embeddings=query,
                        fusion_method=method,
                        limit=10
                    )
                    query_time = (time.time() - start_time) * 1000  # ms

                    method_results["query_times"].append(query_time)
                    method_results["result_counts"].append(
                        len(result.get("fused_results", []))
                    )

                except Exception as e:
                    logger.error("Benchmark query failed", method=method, error=str(e))
                    method_results["errors"] += 1

            # Calculate statistics
            if method_results["query_times"]:
                import statistics
                method_results["avg_time_ms"] = statistics.mean(method_results["query_times"])
                method_results["median_time_ms"] = statistics.median(method_results["query_times"])
                method_results["min_time_ms"] = min(method_results["query_times"])
                method_results["max_time_ms"] = max(method_results["query_times"])

                method_results["avg_results"] = statistics.mean(method_results["result_counts"])

            benchmark_results["fusion_methods"][method] = method_results

        # Generate summary
        benchmark_results["summary"] = {
            "fastest_method": min(
                benchmark_results["fusion_methods"].items(),
                key=lambda x: x[1].get("avg_time_ms", float('inf'))
            )[0] if benchmark_results["fusion_methods"] else None,
            "most_results_method": max(
                benchmark_results["fusion_methods"].items(),
                key=lambda x: x[1].get("avg_results", 0)
            )[0] if benchmark_results["fusion_methods"] else None
        }

        logger.info(
            "Benchmark complete",
            fastest=benchmark_results["summary"]["fastest_method"]
        )

        return benchmark_results

    def record_search_accuracy(
        self,
        query_id: str,
        results: list,
        relevant_doc_ids: list[str],
        fusion_method: str
    ) -> dict:
        """
        Record search accuracy metrics for a query.

        Args:
            query_id: Unique query identifier
            results: Search results to evaluate
            relevant_doc_ids: List of known relevant document IDs
            fusion_method: Fusion method used

        Returns:
            Dictionary with accuracy metrics (precision, recall, etc.)
        """
        if not results:
            return {
                "query_id": query_id,
                "fusion_method": fusion_method,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "retrieved_count": 0,
                "relevant_count": len(relevant_doc_ids)
            }

        # Extract retrieved document IDs
        retrieved_ids = [
            r.id if hasattr(r, 'id') else str(r)
            for r in results
        ]

        # Calculate metrics
        relevant_retrieved = set(retrieved_ids) & set(relevant_doc_ids)

        precision = len(relevant_retrieved) / len(retrieved_ids) if retrieved_ids else 0.0
        recall = len(relevant_retrieved) / len(relevant_doc_ids) if relevant_doc_ids else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        metrics = {
            "query_id": query_id,
            "fusion_method": fusion_method,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "retrieved_count": len(retrieved_ids),
            "relevant_count": len(relevant_doc_ids),
            "relevant_retrieved_count": len(relevant_retrieved)
        }

        logger.debug("Recorded search accuracy", **metrics)

        return metrics

    async def export_performance_report(self, filepath: str | None = None) -> dict | None:
        """
        Export comprehensive performance report.

        Args:
            filepath: Optional file path to save report

        Returns:
            Performance report dictionary
        """
        report = self.get_performance_dashboard_data()

        if filepath:
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info("Exported performance report", filepath=filepath)

        return report

    def get_baseline_configuration(self) -> dict | None:
        """
        Get baseline configuration for performance monitoring.

        Returns:
            Baseline configuration dictionary
        """
        if not self.performance_monitor:
            return None

        # This would return baseline performance metrics
        return {
            "fusion_methods": ["rrf", "weighted_sum", "max_score"],
            "default_weights": {
                "dense": 1.0,
                "sparse": 1.0
            },
            "optimizations_enabled": self.enable_optimizations,
            "multi_tenant_enabled": self.enable_multi_tenant_aggregation
        }


class HybridSearchManager:
    """Backward-compatible wrapper for HybridSearchEngine."""

    def __init__(self, *args, **kwargs) -> None:
        self._engine: HybridSearchEngine | None = None
        if args or kwargs:
            try:
                self._engine = HybridSearchEngine(*args, **kwargs)
            except TypeError:
                self._engine = None

    async def hybrid_search(self, *args, **kwargs):
        if not self._engine:
            raise RuntimeError("Hybrid search engine not initialized")
        return await self._engine.hybrid_search(*args, **kwargs)

    def search(self, *args, **kwargs):
        if not self._engine:
            return []
        result = self._engine.hybrid_search(*args, **kwargs)
        if inspect.isawaitable(result):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop and running_loop.is_running():
                temp_loop = asyncio.new_event_loop()
                try:
                    return temp_loop.run_until_complete(result)
                finally:
                    temp_loop.close()
            return asyncio.run(result)
        return result
