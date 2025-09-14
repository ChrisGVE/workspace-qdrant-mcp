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
"""

# Task 215: Replace direct logging import with unified logging system
# import logging  # MIGRATED to unified system
from collections import defaultdict
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Task 215: Import unified logging system
from ..observability.logger import get_logger

from .sparse_vectors import create_named_sparse_vector

# Task 215: Use unified logging system instead of logging.getLogger(__name__)
logger = get_logger(__name__)


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
    Advanced hybrid search engine with multiple fusion strategies.

    Provides comprehensive hybrid search capabilities by combining dense semantic
    and sparse keyword vector searches with configurable fusion methods.

    Task 215: Enhanced with unified logging system for comprehensive observability.
    """

    def __init__(self, client: QdrantClient) -> None:
        """Initialize hybrid search engine.

        Args:
            client: Qdrant client for vector database operations
        """
        self.client = client
        self.rrf_ranker = RRFFusionRanker()
        self.weighted_ranker = WeightedSumFusionRanker()
        logger.info("Initialized hybrid search engine")

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
    ) -> dict:
        """Execute hybrid search with specified fusion method.

        Args:
            collection_name: Name of collection to search
            query_embeddings: Dict with 'dense' and/or 'sparse' embeddings
            limit: Maximum number of results to return
            fusion_method: Fusion algorithm ("rrf", "weighted_sum", "max_score")
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            filter_conditions: Optional Qdrant filters
            search_params: Optional search parameters
            with_payload: Whether to return payloads
            with_vectors: Whether to return vectors

        Returns:
            Dictionary with fused results and search metadata
        """
        logger.info(
            "Starting hybrid search",
            collection=collection_name,
            limit=limit,
            fusion_method=fusion_method,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
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
                    query_filter=filter_conditions,
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
                    query_filter=filter_conditions,
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