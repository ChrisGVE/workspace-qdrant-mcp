"""
Hybrid search implementation with Reciprocal Rank Fusion (RRF).

Combines dense and sparse vector search results for improved retrieval performance.
"""

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .sparse_vectors import create_named_sparse_vector

logger = logging.getLogger(__name__)


class RRFFusionRanker:
    """
    Reciprocal Rank Fusion (RRF) implementation for combining search results.
    
    Combines rankings from multiple retrieval systems using the RRF formula:
    RRF(d) = Σ(1 / (k + r(d)))
    """
    
    def __init__(self, k: int = 60):
        """
        Initialize RRF ranker.
        
        Args:
            k: RRF constant parameter (typically 60)
        """
        self.k = k
    
    def fuse_rankings(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> List[Dict]:
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
                    "search_types": ["dense"]
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
                    "search_types": ["sparse"]
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
        dense_results: List[Dict],
        sparse_results: List[Dict],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0
    ) -> Dict:
        """
        Provide detailed explanation of RRF fusion process.
        
        Returns:
            Dictionary with fusion explanation and statistics
        """
        fused_results = self.fuse_rankings(dense_results, sparse_results, dense_weight, sparse_weight)
        
        # Calculate statistics
        dense_only = sum(1 for r in fused_results if r["search_types"] == ["dense"])
        sparse_only = sum(1 for r in fused_results if r["search_types"] == ["sparse"])
        both = sum(1 for r in fused_results if len(r["search_types"]) == 2)
        
        return {
            "fusion_method": "Reciprocal Rank Fusion (RRF)",
            "k_parameter": self.k,
            "weights": {
                "dense": dense_weight,
                "sparse": sparse_weight
            },
            "input_stats": {
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results)
            },
            "fusion_stats": {
                "total_fused_results": len(fused_results),
                "dense_only": dense_only,
                "sparse_only": sparse_only,
                "found_in_both": both,
                "top_rrf_score": fused_results[0]["rrf_score"] if fused_results else 0.0
            },
            "fused_results": fused_results
        }


class HybridSearchEngine:
    """
    Hybrid search engine combining dense and sparse vector search.
    
    Provides unified interface for hybrid search with customizable fusion methods.
    """
    
    def __init__(self, qdrant_client: QdrantClient):
        self.client = qdrant_client
        self.rrf_ranker = RRFFusionRanker()
    
    async def hybrid_search(
        self,
        collection_name: str,
        query_embeddings: Dict,
        limit: int = 10,
        score_threshold: float = 0.0,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        fusion_method: str = "rrf",
        query_filter: Optional[models.Filter] = None
    ) -> Dict:
        """
        Perform hybrid search combining dense and sparse vectors.
        
        Args:
            collection_name: Qdrant collection name
            query_embeddings: Dictionary with 'dense' and 'sparse' embeddings
            limit: Maximum number of results
            score_threshold: Minimum score threshold
            dense_weight: Weight for dense results in fusion
            sparse_weight: Weight for sparse results in fusion
            fusion_method: Fusion method ('rrf', 'weighted_sum', 'max')
            query_filter: Optional Qdrant filter
            
        Returns:
            Dictionary with hybrid search results
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
                    with_payload=True
                )
                
                dense_results = [
                    {
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload
                    }
                    for result in dense_search_results
                ]
            
            # Perform sparse search  
            sparse_results = []
            if "sparse" in query_embeddings:
                sparse_vector = create_named_sparse_vector(
                    indices=query_embeddings["sparse"]["indices"],
                    values=query_embeddings["sparse"]["values"],
                    name="sparse"
                )
                
                sparse_search_results = self.client.search(
                    collection_name=collection_name,
                    query_vector=sparse_vector,
                    limit=limit * 2,  # Get more results for better fusion
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True
                )
                
                sparse_results = [
                    {
                        "id": result.id,
                        "score": result.score,
                        "payload": result.payload
                    }
                    for result in sparse_search_results
                ]
            
            # Fuse results
            if fusion_method == "rrf":
                fused_results = self.rrf_ranker.fuse_rankings(
                    dense_results,
                    sparse_results,
                    dense_weight,
                    sparse_weight
                )
            elif fusion_method == "weighted_sum":
                fused_results = self._weighted_sum_fusion(
                    dense_results,
                    sparse_results,
                    dense_weight,
                    sparse_weight
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
                "weights": {
                    "dense": dense_weight,
                    "sparse": sparse_weight
                },
                "results": final_results
            }
            
        except Exception as e:
            logger.error("Hybrid search failed: %s", e)
            return {"error": f"Hybrid search failed: {e}"}
    
    def _weighted_sum_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        dense_weight: float,
        sparse_weight: float
    ) -> List[Dict]:
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
                    "sparse_score": 0.0
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
                        "sparse_score": result["score"]
                    }
        
        # Sort by combined score
        results = list(all_docs.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def _max_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict]
    ) -> List[Dict]:
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
                "sparse_score": 0.0
            }
        
        # Add sparse results, taking max score
        for result in sparse_results:
            doc_id = result["id"]
            if doc_id in all_docs:
                all_docs[doc_id]["score"] = max(all_docs[doc_id]["score"], result["score"])
                all_docs[doc_id]["sparse_score"] = result["score"]
            else:
                all_docs[doc_id] = {
                    "id": doc_id,
                    "payload": result["payload"],
                    "score": result["score"],
                    "search_type": "hybrid",
                    "dense_score": 0.0,
                    "sparse_score": result["score"]
                }
        
        # Sort by score
        results = list(all_docs.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    def benchmark_fusion_methods(
        self,
        collection_name: str,
        query_embeddings: Dict,
        limit: int = 10
    ) -> Dict:
        """
        Benchmark different fusion methods for comparison.
        
        Returns:
            Dictionary comparing different fusion methods
        """
        methods = ["rrf", "weighted_sum", "max"]
        results = {}
        
        for method in methods:
            try:
                result = self.hybrid_search(
                    collection_name=collection_name,
                    query_embeddings=query_embeddings,
                    limit=limit,
                    fusion_method=method
                )
                results[method] = result
            except Exception as e:
                results[method] = {"error": str(e)}
        
        return {
            "benchmark_results": results,
            "query_info": {
                "has_dense": "dense" in query_embeddings,
                "has_sparse": "sparse" in query_embeddings,
                "limit": limit
            }
        }