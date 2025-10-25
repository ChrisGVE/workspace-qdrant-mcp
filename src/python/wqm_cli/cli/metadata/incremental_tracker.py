"""
Incremental tracking system for document metadata workflow.

This module provides change detection and incremental update capabilities,
tracking document modifications over time and enabling efficient processing
of only changed documents in large collections.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from .aggregator import DocumentMetadata
from .exceptions import IncrementalTrackingError


class DocumentChangeInfo:
    """Information about document changes."""

    def __init__(
        self,
        file_path: str,
        current_hash: str,
        previous_hash: str | None = None,
        last_modified: str | None = None,
        change_type: str = "unknown",
    ):
        """
        Initialize document change information.

        Args:
            file_path: Path to the document
            current_hash: Current content hash
            previous_hash: Previous content hash (if available)
            last_modified: Last modification timestamp
            change_type: Type of change (added, modified, deleted, unchanged)
        """
        self.file_path = file_path
        self.current_hash = current_hash
        self.previous_hash = previous_hash
        self.last_modified = last_modified
        self.change_type = change_type
        self.detected_at = datetime.now(timezone.utc).isoformat()

    @property
    def has_changed(self) -> bool:
        """Check if document has changed."""
        return self.change_type != "unchanged"

    @property
    def is_new(self) -> bool:
        """Check if document is new."""
        return self.change_type == "added"

    @property
    def is_modified(self) -> bool:
        """Check if document is modified."""
        return self.change_type == "modified"

    @property
    def is_deleted(self) -> bool:
        """Check if document is deleted."""
        return self.change_type == "deleted"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "current_hash": self.current_hash,
            "previous_hash": self.previous_hash,
            "last_modified": self.last_modified,
            "change_type": self.change_type,
            "detected_at": self.detected_at,
        }


class IncrementalTracker:
    """
    Tracks document changes for incremental metadata updates.

    This class provides persistent tracking of document modifications,
    enabling efficient incremental processing by identifying only
    changed documents in large collections.
    """

    def __init__(
        self,
        storage_path: str | Path | None = None,
        project_name: str | None = None,
    ):
        """
        Initialize incremental tracker.

        Args:
            storage_path: Path to SQLite database file for storage
            project_name: Optional project name for storage isolation
        """
        self.project_name = project_name or "default"
        self.storage_path = Path(storage_path) if storage_path else self._get_default_storage_path()
        self._ensure_storage_directory()
        self._init_database()

    def _get_default_storage_path(self) -> Path:
        """Get default storage path for the tracker database."""
        return Path.home() / ".wqm" / "metadata_tracking" / f"{self.project_name}.db"

    def _ensure_storage_directory(self) -> None:
        """Ensure storage directory exists."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise IncrementalTrackingError(
                f"Failed to create storage directory: {self.storage_path.parent}",
                storage_error=str(e),
            ) from e

    def _init_database(self) -> None:
        """Initialize SQLite database for tracking."""
        try:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS document_tracking (
                        file_path TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        file_size INTEGER,
                        last_modified TEXT,
                        last_processed TEXT,
                        metadata_json TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS change_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        change_type TEXT NOT NULL,
                        previous_hash TEXT,
                        current_hash TEXT NOT NULL,
                        detected_at TEXT NOT NULL,
                        processed_at TEXT,
                        metadata_json TEXT
                    )
                """)

                # Create indexes for performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_document_hash
                    ON document_tracking(content_hash)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_change_history_path
                    ON change_history(file_path)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_change_history_type
                    ON change_history(change_type)
                """)

                conn.commit()

        except Exception as e:
            raise IncrementalTrackingError(
                f"Failed to initialize tracking database: {self.storage_path}",
                storage_error=str(e),
            ) from e

    def detect_changes(
        self,
        document_metadata_list: list[DocumentMetadata],
    ) -> list[DocumentChangeInfo]:
        """
        Detect changes in a list of documents.

        Args:
            document_metadata_list: List of current document metadata

        Returns:
            List of DocumentChangeInfo objects describing changes

        Raises:
            IncrementalTrackingError: If change detection fails
        """
        try:
            changes = []
            current_documents = {
                doc.file_path: doc for doc in document_metadata_list
            }

            # Get existing tracking data
            existing_tracking = self._get_all_tracking_data()

            # Check for new and modified documents
            for file_path, doc_metadata in current_documents.items():
                current_hash = doc_metadata.content_hash

                if file_path in existing_tracking:
                    # Document exists in tracking
                    previous_hash = existing_tracking[file_path]["content_hash"]
                    last_modified = existing_tracking[file_path]["last_modified"]

                    if current_hash != previous_hash:
                        # Document has been modified
                        change_info = DocumentChangeInfo(
                            file_path=file_path,
                            current_hash=current_hash,
                            previous_hash=previous_hash,
                            last_modified=last_modified,
                            change_type="modified",
                        )
                        changes.append(change_info)
                    else:
                        # Document unchanged
                        change_info = DocumentChangeInfo(
                            file_path=file_path,
                            current_hash=current_hash,
                            previous_hash=previous_hash,
                            last_modified=last_modified,
                            change_type="unchanged",
                        )
                        changes.append(change_info)
                else:
                    # New document
                    change_info = DocumentChangeInfo(
                        file_path=file_path,
                        current_hash=current_hash,
                        change_type="added",
                    )
                    changes.append(change_info)

            # Check for deleted documents
            tracked_paths = set(existing_tracking.keys())
            current_paths = set(current_documents.keys())
            deleted_paths = tracked_paths - current_paths

            for deleted_path in deleted_paths:
                previous_data = existing_tracking[deleted_path]
                change_info = DocumentChangeInfo(
                    file_path=deleted_path,
                    current_hash="",  # No current hash for deleted files
                    previous_hash=previous_data["content_hash"],
                    last_modified=previous_data["last_modified"],
                    change_type="deleted",
                )
                changes.append(change_info)

            # Record changes in history
            self._record_change_history(changes)

            logger.info(
                f"Detected changes: {len([c for c in changes if c.has_changed])} changed, "
                f"{len([c for c in changes if not c.has_changed])} unchanged"
            )

            return changes

        except Exception as e:
            raise IncrementalTrackingError(
                "Failed to detect document changes",
                storage_error=str(e),
            ) from e

    def update_tracking_data(
        self,
        document_metadata_list: list[DocumentMetadata],
    ) -> None:
        """
        Update tracking data for processed documents.

        Args:
            document_metadata_list: List of processed document metadata

        Raises:
            IncrementalTrackingError: If tracking update fails
        """
        try:
            current_time = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.storage_path) as conn:
                for doc_metadata in document_metadata_list:
                    file_path = doc_metadata.file_path
                    content_hash = doc_metadata.content_hash
                    file_size = doc_metadata.parsed_document.file_size

                    # Get file modification time if available
                    last_modified = None
                    try:
                        path_obj = Path(file_path)
                        if path_obj.exists():
                            last_modified = datetime.fromtimestamp(
                                path_obj.stat().st_mtime, tz=timezone.utc
                            ).isoformat()
                    except Exception as e:
                        logger.warning(f"Could not get modification time for {file_path}: {e}")

                    # Serialize metadata for storage
                    metadata_json = json.dumps(
                        doc_metadata.to_dict(include_content=False),
                        default=str,
                        sort_keys=True,
                    )

                    # Insert or update tracking record
                    conn.execute("""
                        INSERT OR REPLACE INTO document_tracking (
                            file_path, content_hash, file_size, last_modified,
                            last_processed, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?,
                                  COALESCE((SELECT created_at FROM document_tracking WHERE file_path = ?), ?),
                                  ?)
                    """, (
                        file_path, content_hash, file_size, last_modified,
                        current_time, metadata_json, file_path, current_time, current_time
                    ))

                conn.commit()

            logger.debug(f"Updated tracking data for {len(document_metadata_list)} documents")

        except Exception as e:
            raise IncrementalTrackingError(
                "Failed to update tracking data",
                storage_error=str(e),
            ) from e

    def get_changed_documents(
        self,
        document_metadata_list: list[DocumentMetadata],
    ) -> list[DocumentMetadata]:
        """
        Get only changed documents from a list.

        Args:
            document_metadata_list: List of document metadata to filter

        Returns:
            List of only changed document metadata

        Raises:
            IncrementalTrackingError: If filtering fails
        """
        try:
            changes = self.detect_changes(document_metadata_list)
            changed_paths = {
                change.file_path for change in changes
                if change.has_changed and not change.is_deleted
            }

            changed_documents = [
                doc for doc in document_metadata_list
                if doc.file_path in changed_paths
            ]

            logger.info(f"Filtered to {len(changed_documents)} changed documents")
            return changed_documents

        except Exception as e:
            raise IncrementalTrackingError(
                "Failed to filter changed documents",
                storage_error=str(e),
            ) from e

    def get_change_summary(self) -> dict[str, Any]:
        """
        Get summary of recent changes.

        Returns:
            Dictionary with change statistics

        Raises:
            IncrementalTrackingError: If summary generation fails
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Get total tracked documents
                total_tracked = conn.execute(
                    "SELECT COUNT(*) FROM document_tracking"
                ).fetchone()[0]

                # Get recent changes (last 24 hours)
                recent_changes = conn.execute("""
                    SELECT change_type, COUNT(*)
                    FROM change_history
                    WHERE datetime(detected_at) > datetime('now', '-1 day')
                    GROUP BY change_type
                """).fetchall()

                # Get change history counts
                history_counts = conn.execute("""
                    SELECT change_type, COUNT(*)
                    FROM change_history
                    GROUP BY change_type
                """).fetchall()

                # Get last processing time
                last_processed = conn.execute("""
                    SELECT MAX(last_processed) FROM document_tracking
                    WHERE last_processed IS NOT NULL
                """).fetchone()[0]

            # Format results
            recent_changes_dict = dict(recent_changes)
            history_counts_dict = dict(history_counts)

            return {
                "total_tracked_documents": total_tracked,
                "last_processed": last_processed,
                "recent_changes_24h": recent_changes_dict,
                "change_history_totals": history_counts_dict,
                "database_path": str(self.storage_path),
                "project_name": self.project_name,
            }

        except Exception as e:
            raise IncrementalTrackingError(
                "Failed to generate change summary",
                storage_error=str(e),
            ) from e

    def cleanup_deleted_documents(self, days_old: int = 30) -> int:
        """
        Clean up tracking data for documents that no longer exist.

        Args:
            days_old: Remove tracking data for documents deleted more than this many days ago

        Returns:
            Number of records cleaned up

        Raises:
            IncrementalTrackingError: If cleanup fails
        """
        try:
            with sqlite3.connect(self.storage_path) as conn:
                # Find deleted documents older than specified days
                deleted_records = conn.execute(f"""
                    SELECT dt.file_path
                    FROM document_tracking dt
                    WHERE NOT EXISTS (
                        SELECT 1 FROM change_history ch
                        WHERE ch.file_path = dt.file_path
                        AND ch.change_type != 'deleted'
                        AND datetime(ch.detected_at) > datetime('now', '-{days_old} days')
                    )
                """).fetchall()

                deleted_paths = [record[0] for record in deleted_records]

                # Check if files actually still exist
                actually_deleted = []
                for path in deleted_paths:
                    if not Path(path).exists():
                        actually_deleted.append(path)

                if actually_deleted:
                    # Remove tracking data for actually deleted files
                    for path in actually_deleted:
                        conn.execute(
                            "DELETE FROM document_tracking WHERE file_path = ?",
                            (path,)
                        )
                        conn.execute(
                            "DELETE FROM change_history WHERE file_path = ?",
                            (path,)
                        )

                    conn.commit()

                cleanup_count = len(actually_deleted)
                if cleanup_count > 0:
                    logger.info(f"Cleaned up tracking data for {cleanup_count} deleted documents")

                return cleanup_count

        except Exception as e:
            raise IncrementalTrackingError(
                "Failed to cleanup deleted documents",
                storage_error=str(e),
            ) from e

    def _get_all_tracking_data(self) -> dict[str, dict[str, Any]]:
        """
        Get all current tracking data.

        Returns:
            Dictionary mapping file paths to tracking data
        """
        with sqlite3.connect(self.storage_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT file_path, content_hash, file_size, last_modified,
                       last_processed, metadata_json, created_at, updated_at
                FROM document_tracking
            """).fetchall()

            return {
                row["file_path"]: {
                    "content_hash": row["content_hash"],
                    "file_size": row["file_size"],
                    "last_modified": row["last_modified"],
                    "last_processed": row["last_processed"],
                    "metadata_json": row["metadata_json"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            }

    def _record_change_history(self, changes: list[DocumentChangeInfo]) -> None:
        """
        Record changes in the change history table.

        Args:
            changes: List of document changes to record
        """
        with sqlite3.connect(self.storage_path) as conn:
            for change in changes:
                # Only record actual changes in history
                if change.has_changed:
                    conn.execute("""
                        INSERT INTO change_history (
                            file_path, change_type, previous_hash, current_hash,
                            detected_at, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        change.file_path,
                        change.change_type,
                        change.previous_hash,
                        change.current_hash,
                        change.detected_at,
                        json.dumps(change.to_dict(), default=str)
                    ))

            conn.commit()

    def export_tracking_data(self, output_path: str | Path) -> None:
        """
        Export tracking data to JSON file.

        Args:
            output_path: Path to output JSON file

        Raises:
            IncrementalTrackingError: If export fails
        """
        try:
            tracking_data = self._get_all_tracking_data()

            export_data = {
                "project_name": self.project_name,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "document_count": len(tracking_data),
                "documents": tracking_data,
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Exported tracking data to {output_path}")

        except Exception as e:
            raise IncrementalTrackingError(
                f"Failed to export tracking data to {output_path}",
                storage_error=str(e),
            ) from e

    def import_tracking_data(self, input_path: str | Path) -> None:
        """
        Import tracking data from JSON file.

        Args:
            input_path: Path to input JSON file

        Raises:
            IncrementalTrackingError: If import fails
        """
        try:
            with open(input_path, encoding="utf-8") as f:
                import_data = json.load(f)

            documents = import_data.get("documents", {})
            current_time = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.storage_path) as conn:
                for file_path, tracking_data in documents.items():
                    conn.execute("""
                        INSERT OR REPLACE INTO document_tracking (
                            file_path, content_hash, file_size, last_modified,
                            last_processed, metadata_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_path,
                        tracking_data["content_hash"],
                        tracking_data["file_size"],
                        tracking_data["last_modified"],
                        tracking_data["last_processed"],
                        tracking_data["metadata_json"],
                        tracking_data.get("created_at", current_time),
                        current_time,
                    ))

                conn.commit()

            logger.info(f"Imported tracking data for {len(documents)} documents from {input_path}")

        except Exception as e:
            raise IncrementalTrackingError(
                f"Failed to import tracking data from {input_path}",
                storage_error=str(e),
            ) from e


# Export main classes
__all__ = ["IncrementalTracker", "DocumentChangeInfo"]
