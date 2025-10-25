"""
Memory collection schema for Qdrant storage.

This module defines the vector database schema and operations for the memory system.
The memory collection stores rules as both vectorized content and structured metadata.
"""

import logging
from datetime import datetime
from typing import Any

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
    """

    COLLECTION_NAME = "memory"
    VECTOR_DIMENSION = 384  # all-MiniLM-L6-v2 embeddings
    DISTANCE_METRIC = Distance.COSINE

    def __init__(self, qdrant_client, embedding_service: EmbeddingService):
        """
        Initialize memory collection schema manager.

        Args:
            qdrant_client: Qdrant client instance
            embedding_service: Service for generating embeddings
        """
        self.client = qdrant_client
        self.embedding_service = embedding_service

    async def ensure_collection_exists(self) -> bool:
        """
        Ensure the memory collection exists with proper schema.

        Returns:
            True if collection was created or already exists, False on error
        """
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating memory collection '{self.COLLECTION_NAME}'")

                # Create collection with proper schema
                await self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=self.VECTOR_DIMENSION,
                        distance=self.DISTANCE_METRIC,
                    ),
                    # Optimize for memory and search performance
                    optimizers_config={
                        "default_segment_number": 1,
                        "memmap_threshold": 20000,
                        "indexing_threshold": 10000,
                    },
                    # Enable payload indexing for efficient filtering
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

                logger.info("Successfully created memory collection")
                return True
            else:
                logger.debug(
                    f"Memory collection '{self.COLLECTION_NAME}' already exists"
                )

                # Verify collection schema is compatible
                collection_info = await self.client.get_collection(self.COLLECTION_NAME)
                if self._validate_collection_schema(collection_info):
                    return True
                else:
                    logger.warning("Memory collection schema is incompatible")
                    return False

        except Exception as e:
            logger.error(f"Failed to ensure memory collection exists: {e}")
            return False

    async def store_rule(self, rule: MemoryRule) -> bool:
        """
        Store a memory rule in the collection.

        Args:
            rule: MemoryRule to store

        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Generate embedding for the rule text
            embeddings = await self.embedding_service.generate_embeddings([rule.rule])
            if not embeddings:
                logger.error(f"Failed to generate embedding for rule: {rule.id}")
                return False

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

            # Create point for insertion
            point = PointStruct(id=rule.id, vector=embedding, payload=payload)

            # Insert point into collection
            self.client.upsert(
                collection_name=self.COLLECTION_NAME, points=[point]
            )

            logger.debug(f"Stored memory rule: {rule.id}")
            return True

        except Exception as e:
            logger.error(f"Failed to store memory rule {rule.id}: {e}")
            return False

    async def update_rule(self, rule: MemoryRule) -> bool:
        """
        Update an existing memory rule.

        Args:
            rule: Updated MemoryRule

        Returns:
            True if updated successfully, False otherwise
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

    async def delete_rule(self, rule_id: str) -> bool:
        """
        Delete a memory rule by ID.

        Args:
            rule_id: Rule identifier

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            result = await self.client.delete(
                collection_name=self.COLLECTION_NAME, points_selector=[rule_id]
            )

            return result.status == "completed"

        except Exception as e:
            logger.error(f"Failed to delete memory rule {rule_id}: {e}")
            return False

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
