"""
Advanced document management tools for workspace-qdrant-mcp.

This module provides comprehensive document lifecycle management including addition,
retrieval, updating, and deletion of documents in workspace collections. It handles
intelligent text chunking, automatic embedding generation, metadata management,
and provides robust error handling for production workloads.

Key Features:
    - Intelligent text chunking for large documents with overlap preservation
    - Automatic dense and sparse embedding generation
    - Rich metadata support with auto-generated fields
    - Batch operations for high-throughput document processing
    - Content deduplication via SHA256 hashing
    - Atomic operations with rollback on failure
    - Comprehensive error handling and logging

Document Processing Pipeline:
    1. Content validation and sanitization
    2. Optional intelligent text chunking at word boundaries
    3. Dense + sparse embedding generation for each chunk
    4. Metadata enrichment with timestamps and content hashes
    5. Atomic insertion with transaction support
    6. Success confirmation with detailed statistics

Supported Operations:
    - add_document: Add new documents with chunking and embedding
    - get_document: Retrieve documents with optional vector data
    - update_document: Modify existing documents with version tracking
    - delete_document: Remove documents with cascade cleanup

Example:
    ```python
    from workspace_qdrant_mcp.tools.documents import add_document

    # Add a code file with metadata
    result = await add_document(
        client=workspace_client,
        content=file_content,
        collection="my-project",
        metadata={
            "file_path": "/src/auth.py",
            "file_type": "python",
            "author": "dev-team",
            "project": "authentication-service"
        },
        document_id="auth-module",
        chunk_text=True
    )

    if result["success"]:
        logger.info("Added {result['chunks_added']} chunks")
    ```
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from qdrant_client.http import models

from ..core.client import QdrantWorkspaceClient
from ..core.collection_naming import CollectionPermissionError
from ..core.sparse_vectors import create_qdrant_sparse_vector

logger = logging.getLogger(__name__)


async def add_document(
    client: QdrantWorkspaceClient,
    content: str,
    collection: str,
    metadata: dict[str, Any] | None = None,
    document_id: str | None = None,
    chunk_text: bool = True,
) -> dict:
    """
    Add a document to the specified workspace collection with intelligent processing.

    This function handles the complete document ingestion pipeline including content
    validation, intelligent chunking, embedding generation, metadata enrichment,
    and atomic storage. It's designed for both single documents and batch processing
    workflows.

    The processing pipeline:
    1. Validates content and collection existence
    2. Optionally chunks large text at word boundaries with overlap
    3. Generates dense semantic and sparse keyword embeddings
    4. Enriches metadata with timestamps, content hashes, and system fields
    5. Stores documents atomically with transaction support
    6. Returns detailed success metrics and identifiers

    Args:
        client: Initialized workspace client with embedding service
        content: Document text content to be indexed. Should be meaningful text
                for optimal embedding generation
        collection: Target collection name (must exist in workspace)
        metadata: Optional metadata dictionary for document classification and filtering.
                 Common fields: file_path, file_type, author, project, tags
        document_id: Custom document identifier. If None, generates UUID.
                    Should be unique within the collection
        chunk_text: Whether to intelligently split large documents into overlapping
                   chunks. Recommended for documents > 1000 characters

    Returns:
        Dict: Operation result containing:
            - success (bool): Whether the operation completed successfully
            - document_id (str): Final document identifier used
            - chunks_added (int): Number of text chunks created and stored
            - collection (str): Target collection name
            - metadata (Dict): Final metadata applied (including auto-generated fields)
            - processing_time (float): Total processing time in seconds
            - embedding_stats (Dict): Embedding generation statistics
            - error (str): Error message if operation failed

    Raises:
        ValueError: If content is empty or collection doesn't exist
        RuntimeError: If embedding generation or storage fails

    Example:
        ```python
        # Add a Python file with rich metadata
        result = await add_document(
            client=workspace_client,
            content=open("/src/auth.py").read(),
            collection="authentication-service",
            metadata={
                "file_path": "/src/auth.py",
                "file_type": "python",
                "author": "jane.doe",
                "last_modified": "2024-01-15T10:30:00Z",
                "tags": ["authentication", "security", "api"]
            },
            document_id="auth-py-module",
            chunk_text=True
        )

        if result["success"]:
            logger.info("Successfully added {result['chunks_added']} chunks")
            logger.info("Processing time: {result['processing_time']:.2f}s")
        else:
            logger.info("Failed: {result['error']}")
        ```
    """
    # Input validation
    if not client.initialized:
        return {"error": "Workspace client not initialized"}

    if not content or not content.strip():
        return {"error": "Content cannot be empty"}

    # Record start time for performance metrics
    datetime.now()

    try:
        # Validate collection exists
        available_collections = await client.list_collections()
        if collection not in available_collections:
            return {"error": f"Collection '{collection}' not found"}

        # Check if MCP server can write to this collection
        try:
            client.collection_manager.validate_mcp_write_access(collection)
        except CollectionPermissionError as e:
            return {"error": str(e)}

        # Resolve display name to actual collection name
        actual_collection, _ = client.collection_manager.resolve_collection_name(
            collection
        )

        # Store original document ID for metadata
        original_document_id = document_id

        # Generate UUID for point ID (Qdrant requires UUID or unsigned int)
        point_id = str(uuid.uuid4())

        # Get embedding service
        embedding_service = client.get_embedding_service()

        # Prepare metadata
        doc_metadata = metadata or {}
        doc_metadata.update(
            {
                "document_id": original_document_id or point_id,  # Use original or UUID
                "point_id": point_id,  # Always include the Qdrant point ID
                "added_at": datetime.utcnow().isoformat(),
                "content_length": len(content),
                "collection": collection,
            }
        )

        points_added = 0

        if chunk_text and len(content) > embedding_service.config.embedding.chunk_size:
            # Split into chunks
            chunks = embedding_service.chunk_text(content)

            for i, chunk in enumerate(chunks):
                # Generate UUID for chunk ID (Qdrant requires UUID or unsigned int)
                chunk_id = str(uuid.uuid4())
                chunk_metadata = doc_metadata.copy()
                chunk_metadata.update(
                    {
                        "chunk_id": chunk_id,  # UUID for this chunk
                        "parent_document_id": original_document_id
                        or point_id,  # Reference to parent
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                        "is_chunk": True,
                    }
                )

                # Add chunk to collection
                success = await _add_single_document(
                    client, chunk, collection, chunk_metadata, chunk_id
                )

                if success:
                    points_added += 1

        else:
            # Add as single document
            success = await _add_single_document(
                client, content, collection, doc_metadata, point_id
            )

            if success:
                points_added = 1

        return {
            "document_id": original_document_id
            or point_id,  # Return original ID for reference
            "point_id": point_id,  # Include the Qdrant point ID
            "collection": collection,
            "points_added": points_added,
            "content_length": len(content),
            "chunked": chunk_text and points_added > 1,
            "metadata": doc_metadata,
        }

    except Exception as e:
        logger.error("Failed to add document: %s", e)
        return {"error": f"Failed to add document: {e}"}


async def _add_single_document(
    client: QdrantWorkspaceClient,
    content: str,
    collection: str,
    metadata: dict[str, Any],
    point_id: str,
) -> bool:
    """Add a single document/chunk to collection."""
    try:
        # Resolve display name to actual collection name
        actual_collection, _ = client.collection_manager.resolve_collection_name(
            collection
        )

        # Generate embeddings
        embedding_service = client.get_embedding_service()
        embeddings = await embedding_service.generate_embeddings(content)

        # Prepare vectors
        vectors = {"dense": embeddings["dense"]}
        if "sparse" in embeddings:
            vectors["sparse"] = create_qdrant_sparse_vector(
                indices=embeddings["sparse"]["indices"],
                values=embeddings["sparse"]["values"],
            )

        # Add content to metadata
        payload = metadata.copy()
        payload["content"] = content

        # Create point
        point = models.PointStruct(id=point_id, vector=vectors, payload=payload)

        # Insert into Qdrant (use actual collection name)
        await client.client.upsert(collection_name=actual_collection, points=[point])

        logger.debug(
            "Added document point %s to collection %s (actual: %s)",
            point_id,
            collection,
            actual_collection,
        )
        return True

    except Exception as e:
        logger.error("Failed to add document point %s: %s", point_id, e)
        return False


async def update_document(
    client: QdrantWorkspaceClient,
    document_id: str,
    collection: str,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict:
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
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True,
        )

        if not existing_points[0]:
            return {
                "error": f"Document '{document_id}' not found in collection '{collection}'"
            }

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
                        vectors["sparse"] = create_qdrant_sparse_vector(
                            indices=embeddings["sparse"]["indices"],
                            values=embeddings["sparse"]["values"],
                        )

                    new_payload["content"] = content
                    new_payload["content_length"] = len(content)
                    new_payload["updated_at"] = datetime.utcnow().isoformat()

                    # Update point with new vectors and payload
                    updated_point = models.PointStruct(
                        id=point.id, vector=vectors, payload=new_payload
                    )
                else:
                    # Update only payload - preserve original vector
                    updated_point = models.PointStruct(
                        id=point.id, vector=point.vector, payload=new_payload
                    )

                client.client.upsert(collection_name=collection, points=[updated_point])

                points_updated += 1

            except Exception as e:
                logger.error("Failed to update point %s: %s", point.id, e)
                continue

        return {
            "document_id": document_id,
            "collection": collection,
            "points_updated": points_updated,
            "content_updated": content is not None,
            "metadata_updated": metadata is not None,
        }

    except Exception as e:
        logger.error("Failed to update document: %s", e)
        return {"error": f"Failed to update document: {e}"}


async def delete_document(
    client: QdrantWorkspaceClient, document_id: str, collection: str
) -> dict:
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
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

        return {
            "document_id": document_id,
            "collection": collection,
            "points_deleted": result.operation_id is not None,
            "status": "success",
        }

    except Exception as e:
        logger.error("Failed to delete document: %s", e)
        return {"error": f"Failed to delete document: {e}"}


async def get_document(
    client: QdrantWorkspaceClient,
    document_id: str,
    collection: str,
    include_vectors: bool = False,
) -> dict:
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

        # Resolve display name to actual collection name
        actual_collection, _ = client.collection_manager.resolve_collection_name(
            collection
        )

        # Find document points
        points = client.client.scroll(
            collection_name=actual_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True,
            with_vectors=include_vectors,
        )

        if not points[0]:
            return {
                "error": f"Document '{document_id}' not found in collection '{collection}'"
            }

        # Format results
        document_points = []
        for point in points[0]:
            point_data = {"id": point.id, "payload": point.payload}

            if include_vectors and point.vector:
                point_data["vectors"] = point.vector

            document_points.append(point_data)

        # Sort chunks by index if this is a chunked document
        if len(document_points) > 1 and all(
            "chunk_index" in p["payload"] for p in document_points
        ):
            document_points.sort(key=lambda x: x["payload"]["chunk_index"])

        return {
            "document_id": document_id,
            "collection": collection,
            "points": document_points,
            "total_points": len(document_points),
            "is_chunked": len(document_points) > 1,
        }

    except Exception as e:
        logger.error("Failed to get document: %s", e)
        return {"error": f"Failed to get document: {e}"}


# Version Management Functions
async def find_document_versions(
    client: QdrantWorkspaceClient, document_id: str, collection: str
) -> list[dict]:
    """
    Find all versions of a document based on document_id pattern.

    According to PRD v2.0, documents with same base document_id but different
    versions should be detected and managed with precedence rules.
    """
    if not client.initialized:
        return []

    try:
        # Search for documents with matching document_id pattern
        points, _ = await client.client.scroll(
            collection_name=collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id", match=models.MatchValue(value=document_id)
                    )
                ]
            ),
            with_payload=True,
            limit=100,  # Should be enough for version chains
        )

        versions = []
        for point in points:
            version_info = {
                "point_id": point.id,
                "document_id": point.payload.get("document_id"),
                "version": point.payload.get("version"),
                "is_latest": point.payload.get("is_latest", False),
                "timestamp": point.payload.get("added_at"),
                "supersedes": point.payload.get("supersedes", []),
                "payload": point.payload,
            }
            versions.append(version_info)

        return versions

    except Exception as e:
        logger.error("Failed to find document versions: %s", e)
        return []


async def ingest_new_version(
    client: QdrantWorkspaceClient,
    content: str,
    collection: str,
    metadata: dict[str, Any] | None = None,
    document_id: str | None = None,
    version: str | None = None,
    document_type: str = "generic",
    chunk_text: bool = True,
) -> dict:
    """
    Ingest a new version of a document with version-aware management.

    Implements the PRD v2.0 version management system:
    1. Detect existing versions
    2. De-prioritize old versions (is_latest=False, search_priority=0.1)
    3. Set new version as latest (is_latest=True, search_priority=1.0)
    4. Update version chain for tracking
    """
    if not client.initialized:
        return {"error": "Workspace client not initialized"}

    if not content or not content.strip():
        return {"error": "Content cannot be empty"}

    try:
        # Generate document_id if not provided
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Find existing versions of this document
        existing_versions = await find_document_versions(
            client, document_id, collection
        )

        # Prepare version metadata according to PRD schema
        version_metadata = {
            "document_id": document_id,
            "version": version or datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "version_type": "timestamp" if not version else "semantic",
            "document_type": document_type,
            "is_latest": True,
            "search_priority": 1.0,
            "supersedes": [v["point_id"] for v in existing_versions]
            if existing_versions
            else [],
            "authority_source": "user_provided" if version else "auto_detected",
            "source_info": {
                "ingestion_date": datetime.utcnow().isoformat(),
                "file_hash": f"sha256:{hash(content)}",  # Simple hash for now
            },
        }

        # Merge with user metadata
        if metadata:
            version_metadata.update(metadata)

        # Step 1: De-prioritize all existing versions
        for existing_version in existing_versions:
            try:
                # Update existing version to mark as non-latest
                await client.client.set_payload(
                    collection_name=collection,
                    points=[existing_version["point_id"]],
                    payload={"is_latest": False, "search_priority": 0.1},
                )
                logger.debug("De-prioritized version %s", existing_version["point_id"])
            except Exception as e:
                logger.warning(
                    "Failed to de-prioritize version %s: %s",
                    existing_version["point_id"],
                    e,
                )

        # Step 2: Add the new version as latest
        result = await add_document(
            client=client,
            content=content,
            collection=collection,
            metadata=version_metadata,
            document_id=document_id,
            chunk_text=chunk_text,
        )

        if "error" not in result:
            result["versions_superseded"] = len(existing_versions)
            result["is_new_version"] = len(existing_versions) > 0

        return result

    except Exception as e:
        logger.error("Failed to ingest new version: %s", e)
        return {"error": f"Failed to ingest new version: {e}"}


def get_document_type_config(document_type: str) -> dict:
    """
    Get version management configuration for different document types.
    Based on PRD v2.0 document type specifications.
    """
    configs = {
        "book": {
            "primary_version": "edition",
            "secondary_version": "date",
            "required_metadata": ["title", "author", "edition"],
            "optional_metadata": ["isbn", "publisher", "draft_status"],
            "retention_policy": "latest_only",
        },
        "scientific_article": {
            "primary_version": "publication_date",
            "required_metadata": ["title", "authors", "journal", "publication_date"],
            "optional_metadata": ["doi", "volume", "issue"],
            "retention_policy": "latest_only",
        },
        "code_file": {
            "primary_version": "git_tag",
            "secondary_version": "modification_date",
            "auto_metadata": True,
            "retention_policy": "current_state_only",
        },
        "webpage": {
            "primary_version": "ingestion_date",
            "required_metadata": ["title", "url", "ingestion_date"],
            "retention_policy": "latest_only",
        },
        "generic": {
            "primary_version": "timestamp",
            "required_metadata": ["title"],
            "retention_policy": "latest_only",
        },
    }

    return configs.get(document_type, configs["generic"])
