"""
Memory collection schema for Qdrant storage.

This module defines the vector database schema and operations for the memory system.
The memory collection stores rules as both vectorized content and structured metadata.

Write Path Architecture (ADR-002 Compliance)
============================================

Per ADR-002 "Daemon-Only Write Policy", ALL writes to Qdrant MUST route through
the Rust daemon (memexd). The memory collection is NO LONGER an exception.

Write Priority Order:
1. PRIMARY: DaemonClient.create_collection_v2() / ingest_text() / delete()
2. FALLBACK: enqueue_unified() to SQLite queue for later daemon processing
3. PROHIBITED: Direct Qdrant writes (unless both daemon and queue unavailable)

When daemon is unavailable:
- Writes are queued to unified_queue for daemon to process later
- Response includes fallback_mode='unified_queue' and queued=True
- Warning logged: "Memory write fallback: daemon unavailable"

Collection Types:
- Memory collection: '_memory' (reserved for agent memory rules)
- Uses item_type='content' for rule storage
- Uses item_type='project' for collection creation

See also: docs/adr/ADR-002-daemon-only-write-policy.md
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from qdrant_client.models import (
    CollectionInfo,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from ..core.embeddings import EmbeddingService
from .types import AuthorityLevel, MemoryCategory, MemoryRule

if TYPE_CHECKING:
    from ..core.sqlite_state_manager import SQLiteStateManager
    from ..grpc.daemon_client import DaemonClient

logger = logging.getLogger(__name__)


class MemoryCollectionSchema:
    """
    Manages the Qdrant schema for the memory collection.

    The memory collection stores memory rules with:
    - Dense vector embeddings of rule text for semantic similarity
    - Structured metadata for filtering and organization
    - Full rule data for retrieval and reconstruction

    Schema Design:
    - Collection Name: "memory" (reserved)
    - Vector Dimension: 384 (all-MiniLM-L6-v2)
    - Distance Metric: Cosine
    - Payload Fields: Full rule metadata + searchable fields

    Write Path (ADR-002):
    - All writes route through daemon or unified_queue
    - Direct Qdrant writes only for reads and fallback when both unavailable
    """

    COLLECTION_NAME = "memory"
    VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 embeddings
    DISTANCE_METRIC = Distance.COSINE
    TENANT_ID = "_memory"  # Reserved tenant ID for memory collection

    def __init__(
        self,
        qdrant_client,
        embedding_service: EmbeddingService,
        daemon_client: DaemonClient | None = None,
        state_manager: SQLiteStateManager | None = None,
    ):
        """
        Initialize memory collection schema manager.

        Args:
            qdrant_client: Qdrant client instance (for reads and fallback writes)
            embedding_service: Service for generating embeddings
            daemon_client: Optional DaemonClient for routed writes (preferred)
            state_manager: Optional SQLiteStateManager for queue fallback
        """
        self.client = qdrant_client
        self.embedding_service = embedding_service
        self.daemon_client = daemon_client
        self.state_manager = state_manager

    async def _is_daemon_available(self) -> bool:
        """Check if daemon is available for write operations."""
        if not self.daemon_client:
            return False
        try:
            health = await self.daemon_client.check_health()
            return health.healthy
        except Exception:
            return False

    async def ensure_collection_exists(self) -> dict[str, Any]:
        """
        Ensure the memory collection exists with proper schema.

        Routes through daemon per ADR-002. Falls back to unified_queue if
        daemon unavailable, or direct Qdrant as last resort.

        Returns:
            Dict with keys:
            - success: True if collection exists or was created/queued
            - queued: True if creation was queued (daemon unavailable)
            - queue_id: Queue ID if queued
            - fallback_mode: 'daemon', 'unified_queue', or 'direct_qdrant'
        """
        try:
            # Check if collection exists (read operation - direct Qdrant OK)
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.COLLECTION_NAME in collection_names:
                logger.debug(
                    f"Memory collection '{self.COLLECTION_NAME}' already exists"
                )

                # Verify collection schema is compatible
                collection_info = await self.client.get_collection(self.COLLECTION_NAME)
                if self._validate_collection_schema(collection_info):
                    return {"success": True, "fallback_mode": "existing"}
                else:
                    logger.warning("Memory collection schema is incompatible")
                    return {"success": False, "error": "schema_incompatible"}

            # Collection doesn't exist - need to create it
            logger.info(f"Creating memory collection '{self.COLLECTION_NAME}'")

            # Try daemon first (ADR-002: daemon-only writes)
            if await self._is_daemon_available():
                try:
                    await self.daemon_client.create_collection_v2(
                        collection_name=self.COLLECTION_NAME,
                        tenant_id=self.TENANT_ID,
                        vector_size=self.VECTOR_DIMENSION,
                        distance=self.DISTANCE_METRIC.name.lower(),
                    )
                    logger.info("Successfully created memory collection via daemon")
                    return {
                        "success": True,
                        "fallback_mode": "daemon",
                        "collection": self.COLLECTION_NAME,
                    }
                except Exception as e:
                    logger.warning(
                        f"Memory write fallback: daemon create_collection failed: {e}"
                    )

            # Fallback: Queue to unified_queue for daemon to process later
            if self.state_manager:
                from ..core.sqlite_state_manager import (
                    UnifiedQueueItemType,
                    UnifiedQueueOperation,
                )

                queue_id, is_new = await self.state_manager.enqueue_unified(
                    item_type=UnifiedQueueItemType.PROJECT,
                    op=UnifiedQueueOperation.INGEST,
                    tenant_id=self.TENANT_ID,
                    collection=self.COLLECTION_NAME,
                    payload={
                        "operation": "create_collection",
                        "vector_size": self.VECTOR_DIMENSION,
                        "distance": self.DISTANCE_METRIC.name.lower(),
                        "schema": {
                            "category": "keyword",
                            "authority": "keyword",
                            "source": "keyword",
                            "scope": "keyword",
                            "tags": "keyword",
                            "created_at": "datetime",
                            "updated_at": "datetime",
                            "use_count": "integer",
                        },
                    },
                    priority=8,
                )

                if is_new:
                    logger.warning(
                        f"Memory write fallback: collection creation queued: {queue_id}"
                    )
                else:
                    logger.debug(
                        f"Collection creation already queued (idempotency): {queue_id}"
                    )

                return {
                    "success": True,
                    "queued": True,
                    "queue_id": queue_id,
                    "fallback_mode": "unified_queue",
                    "message": "Collection creation queued for daemon processing.",
                }

            # Last resort: Direct Qdrant write (ADR-002 violation - logged)
            logger.warning(
                "Memory write fallback: daemon unavailable, no queue - "
                "using direct Qdrant write (ADR-002 exception)"
            )
            await self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIMENSION,
                    distance=self.DISTANCE_METRIC,
                ),
                optimizers_config={
                    "default_segment_number": 1,
                    "memmap_threshold": 20000,
                    "indexing_threshold": 10000,
                },
                payload_schema={
                    "category": "keyword",
                    "authority": "keyword",
                    "source": "keyword",
                    "scope": "keyword",
                    "tags": "keyword",
                    "created_at": "datetime",
                    "updated_at": "datetime",
                    "use_count": "integer",
                },
            )

            logger.info("Successfully created memory collection (direct fallback)")
            return {
                "success": True,
                "fallback_mode": "direct_qdrant",
                "warning": "Used direct Qdrant write - daemon unavailable",
            }

        except Exception as e:
            logger.error(f"Failed to ensure memory collection exists: {e}")
            return {"success": False, "error": str(e)}

    async def store_rule(self, rule: MemoryRule) -> dict[str, Any]:
        """
        Store a memory rule in the collection.

        Routes through daemon per ADR-002. Falls back to unified_queue if
        daemon unavailable, or direct Qdrant as last resort.

        Args:
            rule: MemoryRule to store

        Returns:
            Dict with keys:
            - success: True if stored or queued successfully
            - queued: True if storage was queued (daemon unavailable)
            - queue_id: Queue ID if queued
            - fallback_mode: 'daemon', 'unified_queue', or 'direct_qdrant'
        """
        try:
            # Generate embedding for the rule text
            embeddings = await self.embedding_service.generate_embeddings([rule.rule])
            if not embeddings:
                logger.error(f"Failed to generate embedding for rule: {rule.id}")
                return {"success": False, "error": "embedding_failed"}

            embedding = embeddings[0]

            # Prepare payload with full rule data and indexed fields
            payload = {
                # Core rule data
                "id": rule.id,
                "rule": rule.rule,
                "category": rule.category.value,
                "authority": rule.authority.value,
                "scope": rule.scope,
                "tags": rule.tags,
                "source": rule.source,
                # Timestamps (convert to ISO strings for Qdrant)
                "created_at": rule.created_at.isoformat(),
                "updated_at": rule.updated_at.isoformat() if rule.updated_at else None,
                "last_used": rule.last_used.isoformat() if rule.last_used else None,
                # Usage statistics
                "use_count": rule.use_count,
                # Extended metadata
                "metadata": rule.metadata,
            }

            # Try daemon first (ADR-002: daemon-only writes)
            if await self._is_daemon_available():
                try:
                    # Use ingest_text with the rule content
                    response = await self.daemon_client.ingest_text(
                        content=rule.rule,
                        collection_basename="memory",
                        tenant_id=self.TENANT_ID,
                        metadata={
                            "rule_id": rule.id,
                            "category": rule.category.value,
                            "authority": rule.authority.value,
                            "source": rule.source,
                            **payload,
                        },
                        chunk_text=False,  # Rules are single documents
                    )
                    logger.debug(f"Stored memory rule via daemon: {rule.id}")
                    return {
                        "success": True,
                        "rule_id": rule.id,
                        "document_id": response.document_id,
                        "fallback_mode": "daemon",
                    }
                except Exception as e:
                    logger.warning(
                        f"Memory write fallback: daemon ingest failed: {e}"
                    )

            # Fallback: Queue to unified_queue for daemon to process later
            if self.state_manager:
                from ..core.sqlite_state_manager import (
                    UnifiedQueueItemType,
                    UnifiedQueueOperation,
                )

                queue_id, is_new = await self.state_manager.enqueue_unified(
                    item_type=UnifiedQueueItemType.CONTENT,
                    op=UnifiedQueueOperation.INGEST,
                    tenant_id=rule.id,  # Use rule.id as tenant for idempotency
                    collection=self.COLLECTION_NAME,
                    payload={
                        "content": rule.rule,
                        "source_type": "memory_rule",
                        "rule_payload": payload,
                        "embedding": embedding,  # Pre-computed embedding
                    },
                    priority=8,
                    metadata={
                        "rule_id": rule.id,
                        "category": rule.category.value,
                        "authority": rule.authority.value,
                    },
                )

                if is_new:
                    logger.warning(
                        f"Memory write fallback: rule queued: {queue_id} (rule={rule.id})"
                    )
                else:
                    logger.debug(
                        f"Rule already queued (idempotency): {queue_id}"
                    )

                return {
                    "success": True,
                    "rule_id": rule.id,
                    "queued": True,
                    "queue_id": queue_id,
                    "fallback_mode": "unified_queue",
                    "message": "Rule queued for daemon processing.",
                }

            # Last resort: Direct Qdrant write (ADR-002 violation - logged)
            logger.warning(
                f"Memory write fallback: daemon unavailable, no queue - "
                f"using direct Qdrant write for rule {rule.id} (ADR-002 exception)"
            )

            # Create point for insertion
            point = PointStruct(id=rule.id, vector=embedding, payload=payload)

            # Insert point into collection
            self.client.upsert(
                collection_name=self.COLLECTION_NAME, points=[point]
            )

            logger.debug(f"Stored memory rule (direct fallback): {rule.id}")
            return {
                "success": True,
                "rule_id": rule.id,
                "fallback_mode": "direct_qdrant",
                "warning": "Used direct Qdrant write - daemon unavailable",
            }

        except Exception as e:
            logger.error(f"Failed to store memory rule {rule.id}: {e}")
            return {"success": False, "error": str(e)}

    async def update_rule(self, rule: MemoryRule) -> dict[str, Any]:
        """
        Update an existing memory rule.

        Routes through daemon per ADR-002 (same as store_rule).

        Args:
            rule: Updated MemoryRule

        Returns:
            Dict with success status and fallback_mode (same as store_rule)
        """
        # Update is the same as store due to upsert behavior
        return await self.store_rule(rule)

    async def get_rule(self, rule_id: str) -> MemoryRule | None:
        """
        Retrieve a memory rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            MemoryRule if found, None otherwise
        """
        try:
            points = await self.client.retrieve(
                collection_name=self.COLLECTION_NAME, ids=[rule_id], with_payload=True
            )

            if not points:
                return None

            point = points[0]
            return self._point_to_rule(point)

        except Exception as e:
            logger.error(f"Failed to retrieve memory rule {rule_id}: {e}")
            return None

    async def delete_rule(self, rule_id: str) -> dict[str, Any]:
        """
        Delete a memory rule by ID.

        Routes through daemon per ADR-002. Falls back to unified_queue if
        daemon unavailable, or direct Qdrant as last resort.

        Args:
            rule_id: Rule identifier

        Returns:
            Dict with success status and fallback_mode
        """
        try:
            # Try daemon first (ADR-002: daemon-only writes)
            if await self._is_daemon_available():
                try:
                    await self.daemon_client.delete_document(
                        collection_name=self.COLLECTION_NAME,
                        document_id=rule_id,
                    )
                    logger.debug(f"Deleted memory rule via daemon: {rule_id}")
                    return {
                        "success": True,
                        "rule_id": rule_id,
                        "fallback_mode": "daemon",
                    }
                except Exception as e:
                    logger.warning(
                        f"Memory write fallback: daemon delete failed: {e}"
                    )

            # Fallback: Queue to unified_queue for daemon to process later
            if self.state_manager:
                from ..core.sqlite_state_manager import (
                    UnifiedQueueItemType,
                    UnifiedQueueOperation,
                )

                queue_id, is_new = await self.state_manager.enqueue_unified(
                    item_type=UnifiedQueueItemType.DELETE_DOCUMENT,
                    op=UnifiedQueueOperation.DELETE,
                    tenant_id=self.TENANT_ID,
                    collection=self.COLLECTION_NAME,
                    payload={
                        "doc_id": rule_id,
                        "rule_id": rule_id,
                    },
                    priority=8,
                )

                if is_new:
                    logger.warning(
                        f"Memory write fallback: delete queued: {queue_id} (rule={rule_id})"
                    )
                else:
                    logger.debug(
                        f"Delete already queued (idempotency): {queue_id}"
                    )

                return {
                    "success": True,
                    "rule_id": rule_id,
                    "queued": True,
                    "queue_id": queue_id,
                    "fallback_mode": "unified_queue",
                    "message": "Delete queued for daemon processing.",
                }

            # Last resort: Direct Qdrant write (ADR-002 violation - logged)
            logger.warning(
                f"Memory write fallback: daemon unavailable, no queue - "
                f"using direct Qdrant delete for rule {rule_id} (ADR-002 exception)"
            )

            result = await self.client.delete(
                collection_name=self.COLLECTION_NAME, points_selector=[rule_id]
            )

            success = result.status == "completed"
            return {
                "success": success,
                "rule_id": rule_id,
                "fallback_mode": "direct_qdrant",
                "warning": "Used direct Qdrant delete - daemon unavailable",
            }

        except Exception as e:
            logger.error(f"Failed to delete memory rule {rule_id}: {e}")
            return {"success": False, "error": str(e)}

    async def search_rules(
        self,
        query: str,
        limit: int = 10,
        category_filter: MemoryCategory | None = None,
        authority_filter: AuthorityLevel | None = None,
        scope_filter: list[str] | None = None,
        tag_filter: list[str] | None = None,
    ) -> list[tuple[MemoryRule, float]]:
        """
        Search for memory rules using semantic similarity and filters.

        Args:
            query: Text query for semantic search
            limit: Maximum number of results
            category_filter: Filter by category
            authority_filter: Filter by authority level
            scope_filter: Filter by scope (any match)
            tag_filter: Filter by tags (any match)

        Returns:
            List of (MemoryRule, score) tuples sorted by relevance
        """
        try:
            # Generate query embedding
            embeddings = await self.embedding_service.generate_embeddings([query])
            if not embeddings:
                logger.error("Failed to generate query embedding")
                return []

            query_embedding = embeddings[0]

            # Build filter conditions
            filter_conditions = []

            if category_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="category", match=MatchValue(value=category_filter.value)
                    )
                )

            if authority_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="authority", match=MatchValue(value=authority_filter.value)
                    )
                )

            if scope_filter:
                filter_conditions.append(
                    FieldCondition(key="scope", match=MatchAny(any=scope_filter))
                )

            if tag_filter:
                filter_conditions.append(
                    FieldCondition(key="tags", match=MatchAny(any=tag_filter))
                )

            # Perform semantic search
            search_result = await self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_embedding,
                query_filter=Filter(must=filter_conditions)
                if filter_conditions
                else None,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Convert results to rules
            results = []
            for scored_point in search_result:
                rule = self._point_to_rule(scored_point)
                if rule:
                    results.append((rule, scored_point.score))

            return results

        except Exception as e:
            logger.error(f"Failed to search memory rules: {e}")
            return []

    async def list_all_rules(
        self,
        category_filter: MemoryCategory | None = None,
        authority_filter: AuthorityLevel | None = None,
        source_filter: str | None = None,
    ) -> list[MemoryRule]:
        """
        List all memory rules with optional filtering.

        Args:
            category_filter: Filter by category
            authority_filter: Filter by authority level
            source_filter: Filter by source

        Returns:
            List of MemoryRule objects
        """
        try:
            # Build filter conditions
            filter_conditions = []

            if category_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="category", match=MatchValue(value=category_filter.value)
                    )
                )

            if authority_filter:
                filter_conditions.append(
                    FieldCondition(
                        key="authority", match=MatchValue(value=authority_filter.value)
                    )
                )

            if source_filter:
                filter_conditions.append(
                    FieldCondition(key="source", match=MatchValue(value=source_filter))
                )

            # Scroll through all points
            results = []
            offset = None

            while True:
                scroll_result = await self.client.scroll(
                    collection_name=self.COLLECTION_NAME,
                    scroll_filter=Filter(must=filter_conditions)
                    if filter_conditions
                    else None,
                    limit=100,  # Process in batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                points = scroll_result[0]  # (points, next_offset)
                if not points:
                    break

                for point in points:
                    rule = self._point_to_rule(point)
                    if rule:
                        results.append(rule)

                offset = scroll_result[1]
                if offset is None:
                    break

            # Sort by creation date (newest first)
            results.sort(key=lambda r: r.created_at, reverse=True)
            return results

        except Exception as e:
            logger.error(f"Failed to list memory rules: {e}")
            return []

    async def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the memory collection.

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_info = await self.client.get_collection(self.COLLECTION_NAME)

            # Get count by category and authority
            categories = {}
            authorities = {}

            for category in MemoryCategory:
                count_result = await self.client.count(
                    collection_name=self.COLLECTION_NAME,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="category", match=MatchValue(value=category.value)
                            )
                        ]
                    ),
                )
                categories[category.value] = count_result.count

            for authority in AuthorityLevel:
                count_result = await self.client.count(
                    collection_name=self.COLLECTION_NAME,
                    count_filter=Filter(
                        must=[
                            FieldCondition(
                                key="authority", match=MatchValue(value=authority.value)
                            )
                        ]
                    ),
                )
                authorities[authority.value] = count_result.count

            return {
                "total_rules": collection_info.points_count,
                "categories": categories,
                "authorities": authorities,
                "collection_size_bytes": collection_info.segments_count,
                "indexed_fields": list(collection_info.payload_schema.keys())
                if collection_info.payload_schema
                else [],
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    def _validate_collection_schema(self, collection_info: CollectionInfo) -> bool:
        """
        Validate that the existing collection has compatible schema.

        Args:
            collection_info: Qdrant collection information

        Returns:
            True if schema is compatible, False otherwise
        """
        try:
            # Check vector configuration
            vector_config = collection_info.config.params.vectors
            if vector_config.size != self.VECTOR_DIMENSION:
                logger.error(
                    f"Vector dimension mismatch: expected {self.VECTOR_DIMENSION}, got {vector_config.size}"
                )
                return False

            if vector_config.distance != self.DISTANCE_METRIC:
                logger.error(
                    f"Distance metric mismatch: expected {self.DISTANCE_METRIC}, got {vector_config.distance}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to validate collection schema: {e}")
            return False

    def _point_to_rule(self, point) -> MemoryRule | None:
        """
        Convert a Qdrant point to a MemoryRule object.

        Args:
            point: Qdrant point with payload

        Returns:
            MemoryRule object or None if conversion fails
        """
        try:
            payload = point.payload

            # Parse timestamps
            created_at = datetime.fromisoformat(payload["created_at"])
            updated_at = None
            if payload.get("updated_at"):
                updated_at = datetime.fromisoformat(payload["updated_at"])
            last_used = None
            if payload.get("last_used"):
                last_used = datetime.fromisoformat(payload["last_used"])

            return MemoryRule(
                id=payload["id"],
                rule=payload["rule"],
                category=MemoryCategory(payload["category"]),
                authority=AuthorityLevel(payload["authority"]),
                scope=payload.get("scope", []),
                tags=payload.get("tags", []),
                source=payload.get("source", "unknown"),
                created_at=created_at,
                updated_at=updated_at,
                last_used=last_used,
                use_count=payload.get("use_count", 0),
                metadata=payload.get("metadata", {}),
            )

        except Exception as e:
            logger.error(f"Failed to convert point to rule: {e}")
            return None
