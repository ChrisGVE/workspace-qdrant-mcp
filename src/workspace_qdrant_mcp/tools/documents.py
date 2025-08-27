"""
Document management tools for workspace-qdrant-mcp.

Provides MCP tools for adding and managing documents in collections.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException

from ..core.client import QdrantWorkspaceClient

logger = logging.getLogger(__name__)


async def add_document(
    client: QdrantWorkspaceClient,
    content: str,
    collection: str,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    chunk_text: bool = True
) -> Dict:
    """
    Add document to specified collection.
    
    Args:
        client: Workspace client instance
        content: Document content
        collection: Target collection name
        metadata: Optional metadata dictionary
        document_id: Optional document ID (generates UUID if not provided)
        chunk_text: Whether to chunk large text
        
    Returns:
        Dictionary with operation result
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    if not content or not content.strip():
        return {"error": "Content cannot be empty"}
    
    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Get embedding service
        embedding_service = client.get_embedding_service()
        
        # Prepare metadata
        doc_metadata = metadata or {}
        doc_metadata.update({
            "document_id": document_id,
            "added_at": datetime.utcnow().isoformat(),
            "content_length": len(content),
            "collection": collection
        })
        
        points_added = 0
        
        if chunk_text and len(content) > embedding_service.config.embedding.chunk_size:
            # Split into chunks
            chunks = embedding_service.chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{document_id}_chunk_{i}"
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "is_chunk": True
                })
                
                # Add chunk to collection
                success = await _add_single_document(
                    client,
                    chunk,
                    collection,
                    chunk_metadata,
                    chunk_id
                )
                
                if success:
                    points_added += 1
                    
        else:
            # Add as single document
            success = await _add_single_document(
                client,
                content,
                collection,
                doc_metadata,
                document_id
            )
            
            if success:
                points_added = 1
        
        return {
            "document_id": document_id,
            "collection": collection,
            "points_added": points_added,
            "content_length": len(content),
            "chunked": chunk_text and points_added > 1,
            "metadata": doc_metadata
        }
        
    except Exception as e:
        logger.error("Failed to add document: %s", e)
        return {"error": f"Failed to add document: {e}"}


async def _add_single_document(
    client: QdrantWorkspaceClient,
    content: str,
    collection: str,
    metadata: Dict[str, Any],
    point_id: str
) -> bool:
    """Add a single document/chunk to collection."""
    try:
        # Generate embeddings
        embedding_service = client.get_embedding_service()
        embeddings = await embedding_service.generate_embeddings(content)
        
        # Prepare vectors
        vectors = {"dense": embeddings["dense"]}
        if "sparse" in embeddings:
            vectors["sparse"] = models.SparseVector(
                indices=embeddings["sparse"]["indices"],
                values=embeddings["sparse"]["values"]
            )
        
        # Add content to metadata
        payload = metadata.copy()
        payload["content"] = content
        
        # Create point
        point = models.PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload
        )
        
        # Insert into Qdrant
        client.client.upsert(
            collection_name=collection,
            points=[point]
        )
        
        logger.debug("Added document point %s to collection %s", point_id, collection)
        return True
        
    except Exception as e:
        logger.error("Failed to add document point %s: %s", point_id, e)
        return False


async def update_document(
    client: QdrantWorkspaceClient,
    document_id: str,
    collection: str,
    content: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Update an existing document in collection.
    
    Args:
        client: Workspace client instance
        document_id: Document ID to update
        collection: Collection name
        content: New content (optional)
        metadata: New metadata (optional)
        
    Returns:
        Dictionary with operation result
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Find existing document
        existing_points = client.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True
        )
        
        if not existing_points[0]:
            return {"error": f"Document '{document_id}' not found in collection '{collection}'"}
        
        points_updated = 0
        
        for point in existing_points[0]:
            try:
                # Update payload
                new_payload = point.payload.copy()
                
                if metadata:
                    new_payload.update(metadata)
                    
                if content:
                    # Generate new embeddings if content changed
                    embedding_service = client.get_embedding_service()
                    embeddings = await embedding_service.generate_embeddings(content)
                    
                    # Update vectors
                    vectors = {"dense": embeddings["dense"]}
                    if "sparse" in embeddings:
                        vectors["sparse"] = models.SparseVector(
                            indices=embeddings["sparse"]["indices"],
                            values=embeddings["sparse"]["values"]
                        )
                    
                    new_payload["content"] = content
                    new_payload["content_length"] = len(content)
                    new_payload["updated_at"] = datetime.utcnow().isoformat()
                    
                    # Update point with new vectors and payload
                    updated_point = models.PointStruct(
                        id=point.id,
                        vector=vectors,
                        payload=new_payload
                    )
                else:
                    # Update only payload
                    updated_point = models.PointStruct(
                        id=point.id,
                        payload=new_payload
                    )
                
                client.client.upsert(
                    collection_name=collection,
                    points=[updated_point]
                )
                
                points_updated += 1
                
            except Exception as e:
                logger.error("Failed to update point %s: %s", point.id, e)
                continue
        
        return {
            "document_id": document_id,
            "collection": collection,
            "points_updated": points_updated,
            "content_updated": content is not None,
            "metadata_updated": metadata is not None
        }
        
    except Exception as e:
        logger.error("Failed to update document: %s", e)
        return {"error": f"Failed to update document: {e}"}


async def delete_document(
    client: QdrantWorkspaceClient,
    document_id: str,
    collection: str
) -> Dict:
    """
    Delete a document from collection.
    
    Args:
        client: Workspace client instance
        document_id: Document ID to delete
        collection: Collection name
        
    Returns:
        Dictionary with operation result
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Delete points with matching document_id
        result = client.client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
        
        return {
            "document_id": document_id,
            "collection": collection,
            "points_deleted": result.operation_id is not None,
            "status": "success"
        }
        
    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        return {"error": f"Failed to delete document: {e}"}


async def get_document(
    client: QdrantWorkspaceClient,
    document_id: str,
    collection: str,
    include_vectors: bool = False
) -> Dict:
    """
    Retrieve a document from collection.
    
    Args:
        client: Workspace client instance
        document_id: Document ID to retrieve
        collection: Collection name
        include_vectors: Whether to include vector data
        
    Returns:
        Dictionary with document data
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}
    
    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}
        
        # Find document points
        points = client.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=include_vectors
        )
        
        if not points[0]:
            return {"error": f"Document '{document_id}' not found in collection '{collection}'"}
        
        # Format results
        document_points = []
        for point in points[0]:
            point_data = {
                "id": point.id,
                "payload": point.payload
            }
            
            if include_vectors and point.vector:
                point_data["vectors"] = point.vector
                
            document_points.append(point_data)
        
        # Sort chunks by index if this is a chunked document
        if len(document_points) > 1 and all("chunk_index" in p["payload"] for p in document_points):
            document_points.sort(key=lambda x: x["payload"]["chunk_index"])
        
        return {
            "document_id": document_id,
            "collection": collection,
            "points": document_points,
            "total_points": len(document_points),
            "is_chunked": len(document_points) > 1
        }
        
    except Exception as e:
        logger.error("Failed to get document: %s", e)
        return {"error": f"Failed to get document: {e}"}