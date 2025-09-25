"""
Simple MCP Server with exactly 4 tools.

The content and parameters passed to each tool determine what action is taken.
No complex mode switching or backward compatibility - just 4 clear, powerful tools.
"""

import asyncio
import hashlib
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Setup logging
logger = logging.getLogger(__name__)

# Initialize the FastMCP app
app = FastMCP("Workspace Qdrant MCP")

# Global instances
qdrant_client = None
embedding_model = None


async def initialize_components():
    """Initialize Qdrant client and embedding model."""
    global qdrant_client, embedding_model

    if qdrant_client is None:
        # Connect to Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=60
        )

        # Initialize embedding model (import here to avoid slow module-level import)
        from fastembed import TextEmbedding
        embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")

        logger.info("Qdrant client and embedding model initialized")


def get_project_name():
    """Get current project name from git or directory."""
    try:
        # Try to get from git remote
        if Path(".git").exists():
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    url = result.stdout.strip()
                    # Extract repo name from URL
                    parsed = urlparse(url)
                    if parsed.path:
                        return Path(parsed.path).stem.replace('.git', '')
            except:
                pass

        # Fallback to directory name
        return Path.cwd().name
    except:
        return "default"


async def ensure_collection_exists(collection_name: str):
    """Ensure collection exists in Qdrant."""
    try:
        collections = await qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if collection_name not in collection_names:
            await qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to ensure collection {collection_name}: {e}")
        raise


@app.tool()
async def store(
    content: str,
    title: str = None,
    metadata: dict = None,
    collection: str = None,
    source: str = "user_input",
    document_type: str = "text",
    file_path: str = None,
    url: str = None
) -> dict:
    """
    Store any type of content in the vector database.

    The parameters determine what type of storage operation:
    - content + title: Store text/note content
    - file_path: Process and store file content
    - url: Store web content reference
    - collection: Store in specific collection (auto-detected if None)

    Args:
        content: The text content to store
        title: Optional title for the content
        metadata: Optional metadata dictionary
        collection: Target collection (auto-detected if None)
        source: Source type (user_input, file, web, etc.)
        document_type: Type of document (text, code, pdf, etc.)
        file_path: Path to file for file-based storage
        url: URL for web-based storage

    Returns:
        dict: Storage result with document_id and status
    """
    await initialize_components()

    try:
        # Determine collection
        if collection is None:
            project = get_project_name()
            if source == "scratchbook" or "note" in document_type:
                collection = f"{project}-scratchbook"
            elif file_path and file_path.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                collection = f"{project}-code"
            elif url:
                collection = f"{project}-web"
            else:
                collection = f"{project}-documents"

        # Ensure collection exists
        await ensure_collection_exists(collection)

        # Handle file content
        if file_path and not content:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                title = title or Path(file_path).name
                metadata = metadata or {}
                metadata.update({"file_path": file_path})
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to read file {file_path}: {e}",
                    "document_id": None
                }

        if not content:
            return {
                "success": False,
                "error": "No content provided",
                "document_id": None
            }

        # Generate embeddings
        embeddings = list(embedding_model.embed([content]))
        if not embeddings:
            return {
                "success": False,
                "error": "Failed to generate embeddings",
                "document_id": None
            }

        # Create document ID
        document_id = str(uuid.uuid4())

        # Prepare metadata
        doc_metadata = {
            "title": title or "Untitled",
            "source": source,
            "document_type": document_type,
            "content_length": len(content),
            "created_at": datetime.now().isoformat(),
            **(metadata or {})
        }

        if file_path:
            doc_metadata["file_path"] = file_path
        if url:
            doc_metadata["url"] = url

        # Store in Qdrant
        point = PointStruct(
            id=document_id,
            vector=embeddings[0].tolist(),
            payload={
                "content": content,
                **doc_metadata
            }
        )

        await qdrant_client.upsert(
            collection_name=collection,
            points=[point]
        )

        return {
            "success": True,
            "document_id": document_id,
            "collection": collection,
            "source": source,
            "title": title
        }

    except Exception as e:
        logger.error(f"Store operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "document_id": None
        }


@app.tool()
async def search(
    query: str,
    collection: str = None,
    limit: int = 10,
    threshold: float = 0.7,
    mode: str = "semantic",
    filters: dict = None,
    project: str = None,
    document_type: str = None
) -> list:
    """
    Search for content using semantic search.

    The query content and filters determine the search strategy:
    - Simple text: Semantic similarity search
    - filters with metadata: Metadata-filtered search
    - project specified: Project-scoped search
    - collection specified: Collection-scoped search

    Args:
        query: Search query (natural language or keywords)
        collection: Specific collection to search (searches all if None)
        limit: Maximum results to return
        threshold: Minimum similarity score (0.0-1.0)
        mode: Search mode (semantic only for now)
        filters: Metadata filters for refined search
        project: Project name for project-scoped search
        document_type: Filter by document type

    Returns:
        list: Search results with content, metadata, and scores
    """
    await initialize_components()

    try:
        # Generate query embeddings
        query_embeddings = list(embedding_model.embed([query]))
        if not query_embeddings:
            return []

        # Determine collections to search
        collections_to_search = []
        if collection:
            collections_to_search = [collection]
        elif project:
            # Get all collections for project
            all_collections = await qdrant_client.get_collections()
            collections_to_search = [
                c.name for c in all_collections.collections
                if c.name.startswith(f"{project}-")
            ]
        else:
            # Search all collections
            all_collections = await qdrant_client.get_collections()
            collections_to_search = [c.name for c in all_collections.collections]

        all_results = []

        for col in collections_to_search:
            try:
                # Build filters
                search_filter = None
                if filters or document_type:
                    conditions = []

                    if document_type:
                        conditions.append(
                            FieldCondition(
                                key="document_type",
                                match=MatchValue(value=document_type)
                            )
                        )

                    if filters:
                        for key, value in filters.items():
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=value)
                                )
                            )

                    if conditions:
                        search_filter = Filter(must=conditions)

                # Search in collection
                results = await qdrant_client.search(
                    collection_name=col,
                    query_vector=query_embeddings[0].tolist(),
                    query_filter=search_filter,
                    limit=limit,
                    score_threshold=threshold
                )

                # Format results
                for result in results:
                    all_results.append({
                        "id": str(result.id),
                        "score": float(result.score),
                        "content": result.payload.get("content", ""),
                        "title": result.payload.get("title", ""),
                        "collection": col,
                        "metadata": {k: v for k, v in result.payload.items() if k != "content"}
                    })

            except Exception as e:
                logger.warning(f"Search failed for collection {col}: {e}")
                continue

        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:limit]

    except Exception as e:
        logger.error(f"Search operation failed: {e}")
        return []


@app.tool()
async def manage(
    action: str,
    collection: str = None,
    document_id: str = None,
    new_name: str = None,
    metadata_update: dict = None,
    force: bool = False
) -> dict:
    """
    Manage collections, documents, and system operations.

    The action parameter determines what management operation to perform:
    - "list": List collections or documents
    - "create": Create new collection
    - "delete": Delete collection or document
    - "stats": Get system statistics
    - "health": Check system health

    Args:
        action: Management action to perform
        collection: Collection name for collection operations
        document_id: Document ID for document operations
        new_name: New name for rename operations
        metadata_update: New metadata for update operations
        force: Force operation (for deletions)

    Returns:
        dict: Operation result with status and data
    """
    await initialize_components()

    try:
        if action == "list":
            if collection:
                # List documents in collection
                try:
                    results = await qdrant_client.scroll(
                        collection_name=collection,
                        limit=100,
                        with_payload=True
                    )
                    documents = []
                    for point in results[0]:
                        documents.append({
                            "id": str(point.id),
                            "title": point.payload.get("title", ""),
                            "document_type": point.payload.get("document_type", ""),
                            "created_at": point.payload.get("created_at", ""),
                            "content_length": point.payload.get("content_length", 0)
                        })

                    return {
                        "success": True,
                        "type": "documents",
                        "collection": collection,
                        "documents": documents,
                        "count": len(documents)
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": f"Failed to list documents in {collection}: {e}"
                    }
            else:
                # List all collections
                collections = await qdrant_client.get_collections()
                collection_info = []
                for col in collections.collections:
                    info = await qdrant_client.get_collection(col.name)
                    collection_info.append({
                        "name": col.name,
                        "points_count": info.points_count,
                        "status": info.status.name
                    })

                return {
                    "success": True,
                    "type": "collections",
                    "collections": collection_info,
                    "count": len(collection_info)
                }

        elif action == "create":
            if not collection:
                return {
                    "success": False,
                    "error": "Collection name required for create action"
                }

            await ensure_collection_exists(collection)
            return {
                "success": True,
                "action": "create",
                "collection": collection
            }

        elif action == "delete":
            if document_id:
                # Delete specific document
                if not collection:
                    return {
                        "success": False,
                        "error": "Collection name required when deleting document"
                    }

                await qdrant_client.delete(
                    collection_name=collection,
                    points_selector=[document_id]
                )

                return {
                    "success": True,
                    "action": "delete",
                    "type": "document",
                    "document_id": document_id,
                    "collection": collection
                }
            else:
                # Delete entire collection
                if not collection:
                    return {
                        "success": False,
                        "error": "Collection name required for delete action"
                    }

                if not force:
                    return {
                        "success": False,
                        "error": "Collection deletion requires force=True for safety"
                    }

                await qdrant_client.delete_collection(collection)
                return {
                    "success": True,
                    "action": "delete",
                    "type": "collection",
                    "collection": collection
                }

        elif action == "stats":
            if collection:
                info = await qdrant_client.get_collection(collection)
                return {
                    "success": True,
                    "type": "collection_stats",
                    "collection": collection,
                    "stats": {
                        "points_count": info.points_count,
                        "status": info.status.name,
                        "vectors_count": info.vectors_count,
                        "segments_count": info.segments_count
                    }
                }
            else:
                collections = await qdrant_client.get_collections()
                total_points = 0
                for col in collections.collections:
                    info = await qdrant_client.get_collection(col.name)
                    total_points += info.points_count

                return {
                    "success": True,
                    "type": "system_stats",
                    "stats": {
                        "collections_count": len(collections.collections),
                        "total_points": total_points,
                        "collections": [col.name for col in collections.collections]
                    }
                }

        elif action == "health":
            try:
                collections = await qdrant_client.get_collections()
                return {
                    "success": True,
                    "type": "health_check",
                    "health": {
                        "status": "healthy",
                        "qdrant_connected": True,
                        "collections_count": len(collections.collections),
                        "embedding_model": "all-MiniLM-L6-v2"
                    }
                }
            except Exception as e:
                return {
                    "success": False,
                    "type": "health_check",
                    "health": {
                        "status": "unhealthy",
                        "qdrant_connected": False,
                        "error": str(e)
                    }
                }

        else:
            return {
                "success": False,
                "error": f"Unknown action: {action}",
                "available_actions": ["list", "create", "delete", "stats", "health"]
            }

    except Exception as e:
        logger.error(f"Manage operation failed: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@app.tool()
async def retrieve(
    document_id: str = None,
    collection: str = None,
    filters: dict = None,
    limit: int = 100,
    sort_by: str = "created_at",
    include_content: bool = True,
    include_metadata: bool = True
) -> dict:
    """
    Retrieve documents directly without search ranking.

    The parameters determine the retrieval strategy:
    - document_id specified: Get specific document
    - collection + filters: Get filtered documents from collection
    - collection only: Get recent documents from collection
    - no params: Get recent documents from all collections

    Args:
        document_id: Specific document to retrieve
        collection: Collection to retrieve from
        filters: Metadata filters for retrieval
        limit: Maximum documents to retrieve
        sort_by: Sort field (created_at, updated_at, title)
        include_content: Whether to include document content
        include_metadata: Whether to include metadata

    Returns:
        dict: Retrieved documents with metadata and content
    """
    await initialize_components()

    try:
        if document_id:
            # Retrieve specific document
            if not collection:
                return {
                    "success": False,
                    "error": "Collection name required when retrieving specific document"
                }

            points = await qdrant_client.retrieve(
                collection_name=collection,
                ids=[document_id],
                with_payload=True
            )

            if not points:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found in collection {collection}",
                    "documents": []
                }

            point = points[0]
            doc = {
                "id": str(point.id),
                "collection": collection
            }

            if include_content:
                doc["content"] = point.payload.get("content", "")

            if include_metadata:
                doc["metadata"] = {k: v for k, v in point.payload.items() if k != "content"}

            return {
                "success": True,
                "type": "specific_document",
                "document": doc,
                "document_id": document_id
            }

        elif collection:
            # Retrieve from specific collection
            search_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    search_filter = Filter(must=conditions)

            results = await qdrant_client.scroll(
                collection_name=collection,
                scroll_filter=search_filter,
                limit=limit,
                with_payload=True
            )

            documents = []
            for point in results[0]:
                doc = {
                    "id": str(point.id),
                    "collection": collection
                }

                if include_content:
                    doc["content"] = point.payload.get("content", "")

                if include_metadata:
                    doc["metadata"] = {k: v for k, v in point.payload.items() if k != "content"}

                documents.append(doc)

            # Simple sorting by created_at if available
            if sort_by == "created_at":
                documents.sort(
                    key=lambda x: x.get("metadata", {}).get("created_at", ""),
                    reverse=True
                )

            return {
                "success": True,
                "type": "collection_documents",
                "collection": collection,
                "documents": documents,
                "count": len(documents),
                "filters": filters
            }

        else:
            # Retrieve from all collections
            all_collections = await qdrant_client.get_collections()
            all_documents = []

            for col in all_collections.collections:
                try:
                    results = await qdrant_client.scroll(
                        collection_name=col.name,
                        limit=limit,
                        with_payload=True
                    )

                    for point in results[0]:
                        doc = {
                            "id": str(point.id),
                            "collection": col.name
                        }

                        if include_content:
                            doc["content"] = point.payload.get("content", "")

                        if include_metadata:
                            doc["metadata"] = {k: v for k, v in point.payload.items() if k != "content"}

                        all_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Failed to retrieve from collection {col.name}: {e}")
                    continue

            # Sort by created_at if available
            if sort_by == "created_at":
                all_documents.sort(
                    key=lambda x: x.get("metadata", {}).get("created_at", ""),
                    reverse=True
                )

            # Limit results
            all_documents = all_documents[:limit]

            return {
                "success": True,
                "type": "recent_documents",
                "documents": all_documents,
                "count": len(all_documents)
            }

    except Exception as e:
        logger.error(f"Retrieve operation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "documents": []
        }


def create_simple_server() -> FastMCP:
    """Create and return the simple 4-tool MCP server."""
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)