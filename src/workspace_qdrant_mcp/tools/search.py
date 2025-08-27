"""
Search tools for workspace-qdrant-mcp.

Provides MCP tools for searching across workspace collections.
"""

import logging
from typing import Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.client import QdrantWorkspaceClient
from ..core.sparse_vectors import create_named_sparse_vector
from ..core.hybrid_search import HybridSearchEngine

logger = logging.getLogger(__name__)


async def search_workspace(
    client: QdrantWorkspaceClient,
    query: str,
    collections: Optional[List[str]] = None,
    mode: str = "hybrid",
    limit: int = 10,
    score_threshold: float = 0.7
) -> Dict:
    """
    Search across workspace collections with hybrid search.
    
    Args:
        client: Workspace client instance
        query: Search query text
        collections: Collections to search (defaults to all workspace collections)
        mode: Search mode ('dense', 'sparse', 'hybrid')
        limit: Maximum number of results
        score_threshold: Minimum score threshold
        
    Returns:
        Dictionary with search results
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Get embedding service
        embedding_service = client.get_embedding_service()
        
        # Generate embeddings for query
        embeddings = await embedding_service.generate_embeddings(
            query, 
            include_sparse=(mode in ["sparse", "hybrid"])
        )
        
        # Get collections to search
        if collections is None:
            collections = await client.list_collections()
        else:
            # Validate collections exist
            available_collections = await client.list_collections()
            invalid_collections = [c for c in collections if c not in available_collections]
            if invalid_collections:
                return {"error": f"Collections not found: {', '.join(invalid_collections)}"}
        
        # Search each collection
        all_results = []
        
        for collection_name in collections:
            try:
                collection_results = await _search_collection(
                    client.client,
                    collection_name,
                    embeddings,
                    mode,
                    limit,
                    score_threshold
                )
                
                # Add collection info to results
                for result in collection_results:
                    result["collection"] = collection_name
                    
                all_results.extend(collection_results)
                
            except Exception as e:
                logger.warning("Failed to search collection %s: %s", collection_name, e)
                continue
        
        # Sort by score and limit results
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        final_results = all_results[:limit]
        
        return {
            "query": query,
            "mode": mode,
            "collections_searched": collections,
            "total_results": len(final_results),
            "results": final_results
        }
        
    except Exception as e:
        logger.error("Search failed: %s", e)
        return {"error": f"Search failed: {e}"}


async def _search_collection(
    qdrant_client: QdrantClient,
    collection_name: str,
    embeddings: Dict,
    mode: str,
    limit: int,
    score_threshold: float
) -> List[Dict]:
    """Search a single collection with hybrid search support."""
    
    try:
        if mode == "hybrid" and "dense" in embeddings and "sparse" in embeddings:
            # Use hybrid search engine for RRF fusion
            hybrid_engine = HybridSearchEngine(qdrant_client)
            
            result = await hybrid_engine.hybrid_search(
                collection_name=collection_name,
                query_embeddings=embeddings,
                limit=limit,
                score_threshold=score_threshold
            )
            
            if "error" in result:
                logger.error("Hybrid search failed: %s", result["error"])
                return []
            
            # Convert hybrid results to expected format
            search_results = []
            for r in result.get("results", []):
                search_results.append({
                    "id": r["id"],
                    "score": r.get("rrf_score", r.get("score", 0.0)),
                    "payload": r["payload"],
                    "search_type": "hybrid"
                })
            
            return search_results
        
        # Single mode searches (dense or sparse)
        search_results = []
        
        if mode == "dense" and "dense" in embeddings:
            # Dense vector search
            dense_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=("dense", embeddings["dense"]),
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            for result in dense_results:
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "search_type": "dense"
                })
        
        elif mode == "sparse" and "sparse" in embeddings:
            # Sparse vector search using enhanced BM25
            sparse_vector = create_named_sparse_vector(
                indices=embeddings["sparse"]["indices"],
                values=embeddings["sparse"]["values"],
                name="sparse"
            )
            
            sparse_results = qdrant_client.search(
                collection_name=collection_name,
                query_vector=sparse_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            for result in sparse_results:
                search_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "search_type": "sparse"
                })
        
        return search_results[:limit]
        
    except ResponseHandlingException as e:
        logger.error("Qdrant search error in collection %s: %s", collection_name, e)
        return []
    except Exception as e:
        logger.error("Unexpected search error in collection %s: %s", collection_name, e)
        return []


async def search_collection_by_metadata(
    client: QdrantWorkspaceClient,
    collection: str,
    metadata_filter: Dict,
    limit: int = 10
) -> Dict:
    """
    Search collection by metadata filter.
    
    Args:
        client: Workspace client instance
        collection: Collection name to search
        metadata_filter: Metadata filter conditions
        limit: Maximum number of results
        
    Returns:
        Dictionary with search results
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Build Qdrant filter
        qdrant_filter = _build_metadata_filter(metadata_filter)
        
        # Search with metadata filter
        results = client.client.scroll(
            collection_name=collection,
            scroll_filter=qdrant_filter,
            limit=limit,
            with_payload=True
        )
        
        formatted_results = []
        for result in results[0]:  # results is (points, next_page_offset)
            formatted_results.append({
                "id": result.id,
                "payload": result.payload
            })
        
        return {
            "collection": collection,
            "filter": metadata_filter,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error("Metadata search failed: %s", e)
        return {"error": f"Metadata search failed: {e}"}


def _build_metadata_filter(metadata_filter: Dict) -> models.Filter:
    """Build Qdrant filter from metadata dictionary."""
    conditions = []
    
    for key, value in metadata_filter.items():
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
    
    return models.Filter(must=conditions) if conditions else None