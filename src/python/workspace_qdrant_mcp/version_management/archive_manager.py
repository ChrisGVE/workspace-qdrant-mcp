"""
Archive Management System for workspace-qdrant-mcp.

This module implements archive collection management for superseded document versions.
Part of Task 262 requirements:

3. Archive Collections:
   - Version history maintenance
   - Automatic archiving of superseded versions
   - Archive collection management APIs
   - Archive cleanup policies
   - Historical version retrieval
"""

import asyncio
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from loguru import logger
from qdrant_client.http import models

from python.common.core.client import QdrantWorkspaceClient


class ArchivePolicy(Enum):
    """Archive retention policies."""
    KEEP_ALL = "keep_all"                     # Never delete archived versions
    TIME_BASED = "time_based"                 # Delete after specified time
    VERSION_COUNT = "version_count"           # Keep only N latest versions
    SIZE_BASED = "size_based"                 # Delete when archive exceeds size limit
    IMPORTANCE_BASED = "importance_based"     # Keep versions based on importance score


class ArchiveStatus(Enum):
    """Status of archived documents."""
    ACTIVE = "active"           # Currently active version
    ARCHIVED = "archived"       # Superseded but available
    DEPRECATED = "deprecated"   # Marked for deletion
    DELETED = "deleted"         # Removed from system


@dataclass
class ArchiveEntry:
    """Represents an archived document version."""
    point_id: str
    document_id: str
    version: str
    archive_date: datetime
    original_collection: str
    archive_collection: str
    status: ArchiveStatus
    metadata: Dict[str, Any]
    superseded_by: Optional[str] = None
    deletion_date: Optional[datetime] = None
    importance_score: float = 0.5


class ArchiveManager:
    """
    Manages document version archives with configurable retention policies.

    Provides automatic archiving of superseded versions, policy-based cleanup,
    and historical version retrieval capabilities.
    """

    def __init__(self, client: QdrantWorkspaceClient):
        """Initialize archive manager with workspace client."""
        self.client = client

        # Default archive policies per collection type
        self.default_policies = {
            "project": {
                "policy": ArchivePolicy.VERSION_COUNT,
                "max_versions": 10,
                "cleanup_interval_days": 30
            },
            "scratchbook": {
                "policy": ArchivePolicy.TIME_BASED,
                "retention_days": 90,
                "cleanup_interval_days": 7
            },
            "global": {
                "policy": ArchivePolicy.IMPORTANCE_BASED,
                "min_importance": 0.3,
                "max_versions": 100,
                "cleanup_interval_days": 60
            }
        }

        # Archive collection naming convention
        self.archive_suffix = "_archive"

    def get_archive_collection_name(self, original_collection: str) -> str:
        """Generate archive collection name from original collection."""
        return f"{original_collection}{self.archive_suffix}"

    async def ensure_archive_collection(self, original_collection: str) -> bool:
        """
        Ensure archive collection exists for the given original collection.

        Creates the archive collection with appropriate configuration if it doesn't exist.
        """
        if not self.client.initialized:
            logger.error("Workspace client not initialized")
            return False

        archive_collection = self.get_archive_collection_name(original_collection)

        try:
            # Check if archive collection exists
            existing_collections = self.client.list_collections()
            if archive_collection in existing_collections:
                return True

            # Create archive collection with same vector configuration as original
            collection_info = await self.client.client.get_collection(original_collection)

            # Archive collections use same vector config but with reduced search priority
            await self.client.client.create_collection(
                collection_name=archive_collection,
                vectors_config=collection_info.config.params.vectors,
                sparse_vectors_config=collection_info.config.params.sparse_vectors,
            )

            logger.info("Created archive collection: %s", archive_collection)
            return True

        except Exception as e:
            logger.error("Failed to ensure archive collection %s: %s", archive_collection, e)
            return False

    async def archive_document_version(
        self,
        point_id: str,
        original_collection: str,
        superseded_by_point_id: Optional[str] = None,
        metadata_updates: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Archive a specific document version.

        Moves the document from the original collection to the archive collection
        with updated metadata indicating archive status.
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        archive_collection = self.get_archive_collection_name(original_collection)

        try:
            # Ensure archive collection exists
            if not await self.ensure_archive_collection(original_collection):
                return {"error": f"Failed to create archive collection for {original_collection}"}

            # Get the document to be archived
            points = await self.client.client.retrieve(
                collection_name=original_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not points:
                return {"error": f"Document {point_id} not found in {original_collection}"}

            point = points[0]

            # Prepare archive metadata
            archive_metadata = point.payload.copy()
            archive_metadata.update({
                "archive_date": datetime.now(timezone.utc).isoformat(),
                "original_collection": original_collection,
                "archive_status": ArchiveStatus.ARCHIVED.value,
                "is_latest": False,
                "search_priority": 0.1,  # Reduce search priority for archived versions
                "superseded_by": superseded_by_point_id
            })

            # Apply any additional metadata updates
            if metadata_updates:
                archive_metadata.update(metadata_updates)

            # Insert into archive collection
            await self.client.client.upsert(
                collection_name=archive_collection,
                points=[models.PointStruct(
                    id=point_id,
                    vector=point.vector,
                    payload=archive_metadata
                )]
            )

            # Remove from original collection
            await self.client.client.delete(
                collection_name=original_collection,
                points_selector=models.PointIdsList(points=[point_id])
            )

            logger.info("Archived document %s from %s to %s", point_id, original_collection, archive_collection)

            return {
                "success": True,
                "archived_point_id": point_id,
                "archive_collection": archive_collection,
                "archive_date": archive_metadata["archive_date"]
            }

        except Exception as e:
            logger.error("Failed to archive document %s: %s", point_id, e)
            return {"error": f"Failed to archive document: {e}"}

    async def retrieve_archived_versions(
        self,
        document_id: str,
        original_collection: str,
        limit: int = 50
    ) -> List[ArchiveEntry]:
        """
        Retrieve archived versions of a document.

        Returns list of archived versions sorted by archive date (newest first).
        """
        if not self.client.initialized:
            return []

        archive_collection = self.get_archive_collection_name(original_collection)

        try:
            # Check if archive collection exists
            existing_collections = self.client.list_collections()
            if archive_collection not in existing_collections:
                return []

            # Search for archived versions
            points, _ = await self.client.client.scroll(
                collection_name=archive_collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        ),
                        models.FieldCondition(
                            key="archive_status",
                            match=models.MatchValue(value=ArchiveStatus.ARCHIVED.value)
                        )
                    ]
                ),
                with_payload=True,
                limit=limit
            )

            archive_entries = []
            for point in points:
                payload = point.payload

                entry = ArchiveEntry(
                    point_id=str(point.id),
                    document_id=payload.get("document_id", ""),
                    version=payload.get("version", ""),
                    archive_date=datetime.fromisoformat(
                        payload.get("archive_date", datetime.now(timezone.utc).isoformat())
                    ),
                    original_collection=payload.get("original_collection", original_collection),
                    archive_collection=archive_collection,
                    status=ArchiveStatus(payload.get("archive_status", ArchiveStatus.ARCHIVED.value)),
                    metadata=payload,
                    superseded_by=payload.get("superseded_by"),
                    importance_score=payload.get("importance_score", 0.5)
                )
                archive_entries.append(entry)

            # Sort by archive date (newest first)
            archive_entries.sort(key=lambda x: x.archive_date, reverse=True)

            return archive_entries

        except Exception as e:
            logger.error("Failed to retrieve archived versions for %s: %s", document_id, e)
            return []

    async def restore_archived_version(
        self,
        point_id: str,
        archive_collection: str,
        target_collection: str,
        make_latest: bool = False
    ) -> Dict[str, Any]:
        """
        Restore an archived version back to the active collection.

        Optionally marks the restored version as the latest version.
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        try:
            # Get the archived document
            points = await self.client.client.retrieve(
                collection_name=archive_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=True
            )

            if not points:
                return {"error": f"Archived document {point_id} not found in {archive_collection}"}

            point = points[0]

            # Prepare restoration metadata
            restored_metadata = point.payload.copy()

            # Remove archive-specific metadata
            archive_keys = ["archive_date", "archive_status", "superseded_by"]
            for key in archive_keys:
                restored_metadata.pop(key, None)

            # Update metadata for restoration
            restored_metadata.update({
                "restored_date": datetime.now(timezone.utc).isoformat(),
                "restored_from_archive": True,
                "search_priority": 1.0 if make_latest else 0.8
            })

            if make_latest:
                restored_metadata["is_latest"] = True

                # If making this the latest, de-prioritize other versions
                await self._deprioritize_existing_versions(
                    restored_metadata.get("document_id"),
                    target_collection
                )

            # Insert into target collection
            await self.client.client.upsert(
                collection_name=target_collection,
                points=[models.PointStruct(
                    id=point_id,
                    vector=point.vector,
                    payload=restored_metadata
                )]
            )

            # Update archive status (don't delete, mark as restored)
            await self.client.client.set_payload(
                collection_name=archive_collection,
                points=[point_id],
                payload={
                    "archive_status": ArchiveStatus.ACTIVE.value,
                    "restored_date": restored_metadata["restored_date"],
                    "restored_to": target_collection
                }
            )

            logger.info("Restored document %s from archive %s to %s", point_id, archive_collection, target_collection)

            return {
                "success": True,
                "restored_point_id": point_id,
                "target_collection": target_collection,
                "made_latest": make_latest
            }

        except Exception as e:
            logger.error("Failed to restore archived document %s: %s", point_id, e)
            return {"error": f"Failed to restore document: {e}"}

    async def _deprioritize_existing_versions(
        self,
        document_id: str,
        collection: str
    ):
        """De-prioritize existing versions when restoring an archived version as latest."""
        try:
            # Find existing versions
            points, _ = await self.client.client.scroll(
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
                limit=50
            )

            # De-prioritize all existing versions
            for point in points:
                await self.client.client.set_payload(
                    collection_name=collection,
                    points=[point.id],
                    payload={
                        "is_latest": False,
                        "search_priority": 0.5
                    }
                )

        except Exception as e:
            logger.warning("Failed to de-prioritize existing versions: %s", e)

    async def apply_cleanup_policy(
        self,
        collection: str,
        policy: Optional[ArchivePolicy] = None,
        policy_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply cleanup policy to archive collection.

        Removes or marks for deletion archive entries based on the specified policy.
        """
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        # Determine collection type and get default policy
        collection_type = self._get_collection_type(collection)
        default_config = self.default_policies.get(collection_type, self.default_policies["global"])

        # Use provided policy or fall back to default
        cleanup_policy = policy or ArchivePolicy(default_config["policy"])
        config = policy_config or default_config

        archive_collection = self.get_archive_collection_name(collection)

        try:
            # Check if archive collection exists
            existing_collections = self.client.list_collections()
            if archive_collection not in existing_collections:
                return {"success": True, "message": "No archive collection to clean up"}

            cleaned_count = 0

            if cleanup_policy == ArchivePolicy.TIME_BASED:
                cleaned_count = await self._cleanup_by_time(archive_collection, config)

            elif cleanup_policy == ArchivePolicy.VERSION_COUNT:
                cleaned_count = await self._cleanup_by_version_count(archive_collection, config)

            elif cleanup_policy == ArchivePolicy.SIZE_BASED:
                cleaned_count = await self._cleanup_by_size(archive_collection, config)

            elif cleanup_policy == ArchivePolicy.IMPORTANCE_BASED:
                cleaned_count = await self._cleanup_by_importance(archive_collection, config)

            return {
                "success": True,
                "policy": cleanup_policy.value,
                "archive_collection": archive_collection,
                "cleaned_entries": cleaned_count
            }

        except Exception as e:
            logger.error("Failed to apply cleanup policy to %s: %s", archive_collection, e)
            return {"error": f"Failed to apply cleanup policy: {e}"}

    async def _cleanup_by_time(
        self,
        archive_collection: str,
        config: Dict[str, Any]
    ) -> int:
        """Clean up archived entries older than specified retention period."""
        retention_days = config.get("retention_days", 90)
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)

        # Find old entries
        points, _ = await self.client.client.scroll(
            collection_name=archive_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="archive_date",
                        range=models.Range(
                            lt=cutoff_date.isoformat()
                        )
                    )
                ]
            ),
            with_payload=True,
            limit=1000  # Process in batches
        )

        if not points:
            return 0

        # Delete old entries
        point_ids = [point.id for point in points]
        await self.client.client.delete(
            collection_name=archive_collection,
            points_selector=models.PointIdsList(points=point_ids)
        )

        logger.info("Cleaned up %d archived entries older than %d days", len(point_ids), retention_days)
        return len(point_ids)

    async def _cleanup_by_version_count(
        self,
        archive_collection: str,
        config: Dict[str, Any]
    ) -> int:
        """Keep only the N most recent versions per document."""
        max_versions = config.get("max_versions", 10)

        # Group documents and keep only latest versions
        # This is simplified - in practice, you'd want to process document groups individually
        points, _ = await self.client.client.scroll(
            collection_name=archive_collection,
            with_payload=True,
            limit=10000
        )

        if not points:
            return 0

        # Group by document_id
        document_groups = {}
        for point in points:
            doc_id = point.payload.get("document_id")
            if doc_id not in document_groups:
                document_groups[doc_id] = []
            document_groups[doc_id].append(point)

        points_to_delete = []

        for doc_id, doc_points in document_groups.items():
            # Sort by archive date (newest first)
            doc_points.sort(
                key=lambda p: p.payload.get("archive_date", ""),
                reverse=True
            )

            # Mark excess versions for deletion
            if len(doc_points) > max_versions:
                points_to_delete.extend(doc_points[max_versions:])

        if points_to_delete:
            point_ids = [point.id for point in points_to_delete]
            await self.client.client.delete(
                collection_name=archive_collection,
                points_selector=models.PointIdsList(points=point_ids)
            )

            logger.info("Cleaned up %d excess archived versions", len(point_ids))
            return len(point_ids)

        return 0

    async def _cleanup_by_size(
        self,
        archive_collection: str,
        config: Dict[str, Any]
    ) -> int:
        """Clean up archives when collection exceeds size limit."""
        # This is a simplified implementation
        # In practice, you'd want to get actual collection size metrics
        max_size_mb = config.get("max_size_mb", 1000)

        # For now, we'll use document count as a proxy for size
        max_documents = max_size_mb * 10  # Rough approximation

        points, _ = await self.client.client.scroll(
            collection_name=archive_collection,
            with_payload=True,
            limit=max_documents + 100
        )

        if len(points) <= max_documents:
            return 0

        # Sort by importance score and archive date, keep most important/recent
        points.sort(key=lambda p: (
            p.payload.get("importance_score", 0.5),
            p.payload.get("archive_date", "")
        ), reverse=True)

        # Delete excess documents
        points_to_delete = points[max_documents:]
        if points_to_delete:
            point_ids = [point.id for point in points_to_delete]
            await self.client.client.delete(
                collection_name=archive_collection,
                points_selector=models.PointIdsList(points=point_ids)
            )

            logger.info("Cleaned up %d archived entries due to size limit", len(point_ids))
            return len(point_ids)

        return 0

    async def _cleanup_by_importance(
        self,
        archive_collection: str,
        config: Dict[str, Any]
    ) -> int:
        """Clean up archived entries with low importance scores."""
        min_importance = config.get("min_importance", 0.3)

        # Find low-importance entries
        points, _ = await self.client.client.scroll(
            collection_name=archive_collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="importance_score",
                        range=models.Range(
                            lt=min_importance
                        )
                    )
                ]
            ),
            with_payload=True,
            limit=1000
        )

        if not points:
            return 0

        # Delete low-importance entries
        point_ids = [point.id for point in points]
        await self.client.client.delete(
            collection_name=archive_collection,
            points_selector=models.PointIdsList(points=point_ids)
        )

        logger.info("Cleaned up %d low-importance archived entries", len(point_ids))
        return len(point_ids)

    def _get_collection_type(self, collection: str) -> str:
        """Determine collection type from collection name."""
        if "scratchbook" in collection.lower():
            return "scratchbook"
        elif "global" in collection.lower():
            return "global"
        else:
            return "project"

    async def get_archive_statistics(self, collection: str) -> Dict[str, Any]:
        """Get statistics about archived documents for a collection."""
        if not self.client.initialized:
            return {"error": "Workspace client not initialized"}

        archive_collection = self.get_archive_collection_name(collection)

        try:
            # Check if archive collection exists
            existing_collections = self.client.list_collections()
            if archive_collection not in existing_collections:
                return {
                    "collection": collection,
                    "archive_collection": archive_collection,
                    "total_archived": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                    "status_counts": {}
                }

            # Get all archived documents
            points, _ = await self.client.client.scroll(
                collection_name=archive_collection,
                with_payload=True,
                limit=10000
            )

            if not points:
                return {
                    "collection": collection,
                    "archive_collection": archive_collection,
                    "total_archived": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                    "status_counts": {}
                }

            # Calculate statistics
            archive_dates = []
            status_counts = {}
            document_counts = {}

            for point in points:
                payload = point.payload

                # Collect archive dates
                if "archive_date" in payload:
                    archive_dates.append(payload["archive_date"])

                # Count statuses
                status = payload.get("archive_status", ArchiveStatus.ARCHIVED.value)
                status_counts[status] = status_counts.get(status, 0) + 1

                # Count documents
                doc_id = payload.get("document_id")
                if doc_id:
                    document_counts[doc_id] = document_counts.get(doc_id, 0) + 1

            archive_dates.sort()

            return {
                "collection": collection,
                "archive_collection": archive_collection,
                "total_archived": len(points),
                "unique_documents": len(document_counts),
                "oldest_entry": archive_dates[0] if archive_dates else None,
                "newest_entry": archive_dates[-1] if archive_dates else None,
                "status_counts": status_counts,
                "avg_versions_per_document": len(points) / len(document_counts) if document_counts else 0
            }

        except Exception as e:
            logger.error("Failed to get archive statistics for %s: %s", collection, e)
            return {"error": f"Failed to get archive statistics: {e}"}