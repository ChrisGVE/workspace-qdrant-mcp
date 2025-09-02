
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
"""

import logging
from collections import defaultdict
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .sparse_vectors import create_named_sparse_vector

logger = logging.getLogger(__name__)


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
        - k is a constant (typically 60)
        - r(d) is the rank of document d in each ranking

    Key Advantages:
        - Score-agnostic: Works regardless of score distributions
        - Rank-based: Focuses on relative ordering rather than absolute scores
        - Proven effectiveness: Widely used in information retrieval research
        - Handles missing documents gracefully (documents not in all rankings)

    Attributes:
        k (int): RRF constant parameter controlling rank contribution decay

    Example:
        ```python
        ranker = RRFFusionRanker(k=60)

        dense_results = [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.7}]
        sparse_results = [{"id": "doc2", "score": 0.8}, {"id": "doc3", "score": 0.6}]

        fused = ranker.fuse_rankings(dense_results, sparse_results)
        # Result combines both rankings with RRF scoring
        ```
    """

    def __init__(self, k: int = 60) -> None:
        """
        Initialize RRF ranker with configurable constant parameter.

        Args:
            k: RRF constant parameter that controls how quickly rank contribution
               decays. Typical values:
               - 60: Standard value from literature (recommended)
               - 10-30: More emphasis on top-ranked results
               - 100+: More uniform contribution across ranks
        """
        self.k = k

    def fuse_rankings(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> list[dict]:
        """
        Fuse dense and sparse search results using RRF.

        Args:
            dense_results: Dense vector search results
            sparse_results: Sparse vector search results
            dense_weight: Weight for dense results
            sparse_weight: Weight for sparse results

        Returns:
            List of fused results sorted by RRF score
        """
        # Create document score accumulator
        doc_scores = defaultdict(float)
        doc_data = {}  # Store document metadata

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            rrf_score = dense_weight / (self.k + rank)
            doc_scores[doc_id] += rrf_score

            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "payload": result.get("payload", {}),
                    "dense_score": result.get("score", 0.0),
                    "sparse_score": 0.0,
                    "dense_rank": rank,
                    "sparse_rank": None,
                    "search_types": ["dense"],
                }
            else:
                doc_data[doc_id]["dense_score"] = result.get("score", 0.0)
                doc_data[doc_id]["dense_rank"] = rank
                if "dense" not in doc_data[doc_id]["search_types"]:
                    doc_data[doc_id]["search_types"].append("dense")

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result["id"]
            rrf_score = sparse_weight / (self.k + rank)
            doc_scores[doc_id] += rrf_score

            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "id": doc_id,
                    "payload": result.get("payload", {}),
                    "dense_score": 0.0,
                    "sparse_score": result.get("score", 0.0),
                    "dense_rank": None,
                    "sparse_rank": rank,
                    "search_types": ["sparse"],
                }
            else:
                doc_data[doc_id]["sparse_score"] = result.get("score", 0.0)
                doc_data[doc_id]["sparse_rank"] = rank
                if "sparse" not in doc_data[doc_id]["search_types"]:
                    doc_data[doc_id]["search_types"].append("sparse")

        # Create final results with RRF scores
        fused_results = []
        for doc_id, rrf_score in doc_scores.items():
            result = doc_data[doc_id].copy()
            result["rrf_score"] = rrf_score
            result["search_type"] = "hybrid"
            fused_results.append(result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return fused_results

    def explain_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> dict:
        """
        Provide detailed explanation of RRF fusion process.

        Returns:
            Dictionary with fusion explanation and statistics
        """
        fused_results = self.fuse_rankings(
            dense_results, sparse_results, dense_weight, sparse_weight
        )

        # Calculate statistics
        dense_only = sum(1 for r in fused_results if r["search_types"] == ["dense"])
        sparse_only = sum(1 for r in fused_results if r["search_types"] == ["sparse"])
        both = sum(1 for r in fused_results if len(r["search_types"]) == 2)

        return {
            "fusion_method": "Reciprocal Rank Fusion (RRF)",
            "k_parameter": self.k,
            "weights": {"dense": dense_weight, "sparse": sparse_weight},
            "input_stats": {
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
            },
            "fusion_stats": {
                "total_fused_results": len(fused_results),
                "dense_only": dense_only,
                "sparse_only": sparse_only,
                "found_in_both": both,
                "top_rrf_score": fused_results[0]["rrf_score"]
                if fused_results
                else 0.0,
            },
            "fused_results": fused_results,
        }


class HybridSearchEngine:
    """
    Production-ready hybrid search engine with multiple fusion strategies.

    This class provides a comprehensive interface for performing hybrid search that
    combines dense semantic embeddings with sparse keyword vectors. It supports
    multiple fusion algorithms, configurable weighting, and detailed analysis
    capabilities for optimal search performance.

    The engine is designed for production use with:
    - Multiple fusion strategies for different use cases
    - Robust error handling and logging
    - Performance optimization with expanded result sets
    - Comprehensive result metadata for analysis
    - Benchmarking tools for algorithm comparison

    Supported Fusion Methods:
    1. **RRF (Reciprocal Rank Fusion)**: Score-agnostic, rank-based fusion
       - Best for: General use, handling different score distributions
       - Performance: Balanced precision/recall

    2. **Weighted Sum**: Normalized scores with configurable weights
       - Best for: Fine-tuned control, similar score ranges
       - Performance: Good when score distributions are well-understood

    3. **Max Score**: Takes maximum score across modalities
       - Best for: High-precision scenarios, emphasizing best matches
       - Performance: High precision, may reduce recall

    Attributes:
        client (QdrantClient): Qdrant database client for vector operations
        rrf_ranker (RRFFusionRanker): RRF fusion algorithm implementation

    Example:
        ```python
        from qdrant_client import QdrantClient

        client = QdrantClient("http://localhost:6333")
        engine = HybridSearchEngine(client)

        # Perform hybrid search
        results = await engine.hybrid_search(
            collection_name="documents",
            query_embeddings={
                "dense": dense_vector,
                "sparse": {"indices": indices, "values": values}
            },
            limit=10,
            fusion_method="rrf"
        )

        # Compare fusion methods
        benchmark = engine.benchmark_fusion_methods(
            collection_name="documents",
            query_embeddings=embeddings,
            limit=10
        )
        ```
    """

    def __init__(self, qdrant_client: QdrantClient) -> None:
        """Initialize hybrid search engine with Qdrant client.

        Args:
            qdrant_client: Configured Qdrant client for database operations
        """
        self.client = qdrant_client
        self.rrf_ranker = RRFFusionRanker()

    async def hybrid_search(
        self,
        collection_name: str,
        query_embeddings: dict,
        limit: int = 10,
        score_threshold: float = 0.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fusion_method: str = "rrf",
        query_filter: models.Filter | None = None,
    ) -> dict:
        """
        Perform comprehensive hybrid search combining dense semantic and sparse keyword vectors.

        This method executes the complete hybrid search pipeline: performing both
        dense and sparse searches against the Qdrant collection, then fusing the
        results using the specified algorithm. It's optimized for production use
        with enhanced result sets and comprehensive error handling.

        Search Pipeline:
        1. **Dense Search**: Semantic similarity using embedding vectors
        2. **Sparse Search**: Keyword matching using BM25-style sparse vectors
        3. **Result Fusion**: Combines rankings using selected fusion algorithm
        4. **Post-processing**: Applies limits and formats comprehensive results

        Fusion Methods Available:
        - **rrf**: Reciprocal Rank Fusion (recommended for most cases)
        - **weighted_sum**: Score normalization with configurable weights
        - **max**: Maximum score fusion for high-precision scenarios

        Args:
            collection_name: Target Qdrant collection name
            query_embeddings: Dictionary containing embedding vectors:
                - 'dense' (List[float]): Semantic embedding vector (e.g., 384-dim)
                - 'sparse' (Dict): Sparse vector with 'indices' and 'values' arrays
            limit: Maximum number of results in final ranking (1-1000)
            score_threshold: Minimum relevance score threshold (0.0-1.0)
            dense_weight: Multiplicative weight for dense search contribution
            sparse_weight: Multiplicative weight for sparse search contribution
            fusion_method: Algorithm for combining results ('rrf', 'weighted_sum', 'max')
            query_filter: Optional Qdrant filter for metadata-based filtering

        Returns:
            Dict: Comprehensive search results containing:
                - collection (str): Source collection name
                - fusion_method (str): Fusion algorithm used
                - total_results (int): Number of results returned
                - dense_results_count (int): Results from dense search
                - sparse_results_count (int): Results from sparse search
                - weights (Dict): Applied weights for fusion
                - results (List[Dict]): Final fused results with metadata:
                    - id (str): Document identifier
                    - payload (Dict): Document content and metadata
                    - rrf_score/score (float): Final fusion score
                    - dense_score (float): Original dense similarity score
                    - sparse_score (float): Original sparse matching score
                    - search_type (str): Result type ('hybrid')
                - error (str): Error message if search failed

        Raises:
            ValueError: If fusion_method is not supported
            RuntimeError: If both dense and sparse embeddings are missing
            ConnectionError: If Qdrant database is unreachable

        Example:
            ```python
            engine = HybridSearchEngine(qdrant_client)

            # Basic hybrid search
            results = await engine.hybrid_search(
                collection_name="documents",
                query_embeddings={
                    "dense": [0.1, 0.2, 0.3, ...],  # 384 dimensions
                    "sparse": {
                        "indices": [42, 128, 1337],
                        "values": [0.8, 0.6, 0.4]
                    }
                },
                limit=20,
                fusion_method="rrf"
            )

            # Advanced search with filtering and custom weights
            results = await engine.hybrid_search(
                collection_name="technical-docs",
                query_embeddings=embeddings,
                limit=10,
                score_threshold=0.7,
                dense_weight=1.2,  # Emphasize semantic similarity
                sparse_weight=0.8,  # De-emphasize keyword matching
                fusion_method="weighted_sum",
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value="tutorial")
                        )
                    ]
                )
            )

            logger.info("Found {results['total_results']} results")
            logger.info("Dense: {results['dense_results_count']}, Sparse: {results['sparse_results_count']}")

            for result in results['results']:
                logger.info("Score: {result.get('rrf_score', result.get('score')):.3f}")
                logger.info("Title: {result['payload'].get('title', 'Untitled')}")
            ```
        """
        try:
            # Perform dense search
            dense_results = []
            if "dense" in query_embeddings:
                dense_search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=("dense", query_embeddings["dense"]),
                    limit=limit * 2,  # Get more results for better fusion
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )

                dense_results = [
                    {"id": result.id, "score": result.score, "payload": result.payload}
                    for result in dense_search_results
                ]

            # Perform sparse search
            sparse_results = []
            if "sparse" in query_embeddings:
                sparse_vector = create_named_sparse_vector(
                    indices=query_embeddings["sparse"]["indices"],
                    values=query_embeddings["sparse"]["values"],
                    name="sparse",
                )

                sparse_search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=sparse_vector,
                    limit=limit * 2,  # Get more results for better fusion
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True,
                )

                sparse_results = [
                    {"id": result.id, "score": result.score, "payload": result.payload}
                    for result in sparse_search_results
                ]

            # Fuse results
            if fusion_method == "rrf":
                fused_results = self.rrf_ranker.fuse_rankings(
                    dense_results, sparse_results, dense_weight, sparse_weight
                )
            elif fusion_method == "weighted_sum":
                fused_results = self._weighted_sum_fusion(
                    dense_results, sparse_results, dense_weight, sparse_weight
                )
            elif fusion_method == "max":
                fused_results = self._max_fusion(dense_results, sparse_results)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            # Apply final limit
            final_results = fused_results[:limit]

            return {
                "collection": collection_name,
                "fusion_method": fusion_method,
                "total_results": len(final_results),
                "dense_results_count": len(dense_results),
                "sparse_results_count": len(sparse_results),
                "weights": {"dense": dense_weight, "sparse": sparse_weight},
                "results": final_results,
            }

        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            return {"error": f"Hybrid search failed: {e}"}

    def _weighted_sum_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        dense_weight: float,
        sparse_weight: float,
    ) -> list[dict]:
        """Simple weighted sum fusion of scores."""
        all_docs = {}

        # Normalize and weight dense scores
        if dense_results:
            max_dense_score = max(r["score"] for r in dense_results)
            for result in dense_results:
                doc_id = result["id"]
                normalized_score = result["score"] / max_dense_score
                all_docs[doc_id] = {
                    "id": doc_id,
                    "payload": result["payload"],
                    "score": normalized_score * dense_weight,
                    "search_type": "hybrid",
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                }

        # Normalize and weight sparse scores
        if sparse_results:
            max_sparse_score = max(r["score"] for r in sparse_results)
            for result in sparse_results:
                doc_id = result["id"]
                normalized_score = result["score"] / max_sparse_score

                if doc_id in all_docs:
                    all_docs[doc_id]["score"] += normalized_score * sparse_weight
                    all_docs[doc_id]["sparse_score"] = result["score"]
                else:
                    all_docs[doc_id] = {
                        "id": doc_id,
                        "payload": result["payload"],
                        "score": normalized_score * sparse_weight,
                        "search_type": "hybrid",
                        "dense_score": 0.0,
                        "sparse_score": result["score"],
                    }

        # Sort by combined score
        results = list(all_docs.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    def _max_fusion(
        self, dense_results: list[dict], sparse_results: list[dict]
    ) -> list[dict]:
        """Max score fusion - take maximum score for each document."""
        all_docs = {}

        # Add dense results
        for result in dense_results:
            doc_id = result["id"]
            all_docs[doc_id] = {
                "id": doc_id,
                "payload": result["payload"],
                "score": result["score"],
                "search_type": "hybrid",
                "dense_score": result["score"],
                "sparse_score": 0.0,
            }

        # Add sparse results, taking max score
        for result in sparse_results:
            doc_id = result["id"]
            if doc_id in all_docs:
                all_docs[doc_id]["score"] = max(
                    all_docs[doc_id]["score"], result["score"]
                )
                all_docs[doc_id]["sparse_score"] = result["score"]
            else:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "payload": result["payload"],
                    "score": result["score"],
                    "search_type": "hybrid",
                    "dense_score": 0.0,
                    "sparse_score": result["score"],
                }

        # Sort by score
        results = list(all_docs.values())
        results.sort(key=lambda x: x["score"], reverse=True)

        return results

    async def benchmark_fusion_methods(
        self, collection_name: str, query_embeddings: dict, limit: int = 10
    ) -> dict:
        """
        Benchmark and compare all available fusion methods for performance analysis.

        This method runs the same hybrid search query using all supported fusion
        algorithms (RRF, weighted sum, max score) and provides comparative analysis.
        It's useful for understanding which fusion method works best for specific
        query types or collections.

        Use Cases:
        - Evaluating fusion method effectiveness for specific data types
        - A/B testing different fusion algorithms
        - Research and optimization of search performance
        - Understanding fusion behavior with different query patterns

        Args:
            collection_name: Qdrant collection to search
            query_embeddings: Query vectors (dense and/or sparse)
            limit: Maximum results per fusion method

        Returns:
            Dict: Comprehensive benchmark results containing:
                - benchmark_results (Dict): Results for each fusion method:
                    - 'rrf' (Dict): RRF fusion results or error
                    - 'weighted_sum' (Dict): Weighted sum results or error
                    - 'max' (Dict): Max score results or error
                - query_info (Dict): Query characteristics:
                    - has_dense (bool): Whether dense embeddings provided
                    - has_sparse (bool): Whether sparse embeddings provided
                    - limit (int): Result limit used

        Performance Analysis:
        The benchmark helps identify:
        - Which method provides best result diversity
        - Score distribution characteristics
        - Consensus between dense and sparse search
        - Method-specific strengths and weaknesses

        Example:
            ```python
            engine = HybridSearchEngine(qdrant_client)

            benchmark = engine.benchmark_fusion_methods(
                collection_name="research-papers",
                query_embeddings={
                    "dense": semantic_vector,
                    "sparse": keyword_vector
                },
                limit=10
            )

            # Analyze results
            for method, results in benchmark['benchmark_results'].items():
                if 'error' in results:
                    logger.info("{method}: Failed - {results['error']}")
                else:
                    logger.info("{method}: {results['total_results']} results")
                    logger.info("  Top score: {results['results'][0]['score']:.3f}")
                    logger.info("  Unique results: {len(set(r['id'] for r in results['results']))}")

            # Compare result overlap
            rrf_ids = set(r['id'] for r in benchmark['benchmark_results']['rrf']['results'])
            ws_ids = set(r['id'] for r in benchmark['benchmark_results']['weighted_sum']['results'])
            overlap = len(rrf_ids & ws_ids)
            logger.info("RRF/WeightedSum overlap: {overlap}/{limit} documents")
            ```
        """
        methods = ["rrf", "weighted_sum", "max"]
        results = {}

        for method in methods:
            try:
                result = await self.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=limit,
                    fusion_method=method,
                )
                results[method] = result
            except Exception as e:
                results[method] = {"error": str(e)}

        return {
            "benchmark_results": results,
            "query_info": {
                "has_dense": "dense" in query_embeddings,
                "has_sparse": "sparse" in query_embeddings,
                "limit": limit,
            },
        }
