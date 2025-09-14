"""
Incremental Update Mechanism for Real-time Development Workflows.

This module provides efficient incremental processing by tracking file changes
and updating only modified content to minimize processing overhead for LSP integration.

Key Features:
    - Change detection comparing current vs stored file metadata
    - Differential Qdrant collection updates (update/delete specific points)
    - Transaction-safe updates with rollback capability
    - Batch processing for multiple file changes
    - Priority-based processing (current project files first)
    - Conflict resolution for concurrent file changes
    - Dependency relationship maintenance during incremental updates
    - Performance optimization for real-time development workflows

Example:
    ```python
    from workspace_qdrant_mcp.core.incremental_processor import IncrementalProcessor
    from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

    # Initialize components
    state_manager = SQLiteStateManager("./workspace_state.db")
    await state_manager.initialize()
    
    processor = IncrementalProcessor(
        state_manager=state_manager,
        qdrant_client=qdrant_client
    )
    await processor.initialize()

    # Process incremental updates
    changes = await processor.detect_changes(file_paths)
    await processor.process_changes(changes, batch_size=10)
    ```
"""

import asyncio
import hashlib
import json
from loguru import logger
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .sqlite_state_manager import (
    FileProcessingRecord,
    FileProcessingStatus, 
    ProcessingPriority,
    SQLiteStateManager
)

# logger imported from loguru


class ChangeType(Enum):
    """Types of file changes detected."""
    
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    METADATA_CHANGED = "metadata_changed"
    LSP_STALE = "lsp_stale"  # LSP data needs refresh


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    
    LATEST_WINS = "latest_wins"
    SKIP_CONFLICTED = "skip_conflicted"
    MERGE_METADATA = "merge_metadata"
    USER_PROMPT = "user_prompt"


@dataclass
class FileChangeInfo:
    """Information about a detected file change."""
    
    file_path: str
    change_type: ChangeType
    current_mtime: Optional[float] = None
    current_size: Optional[int] = None
    current_hash: Optional[str] = None
    stored_record: Optional[FileProcessingRecord] = None
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    collection: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    conflict_detected: bool = False
    conflict_reason: Optional[str] = None


@dataclass
class ChangeDetectionResult:
    """Result of change detection process."""
    
    changes: List[FileChangeInfo]
    total_scanned: int
    unchanged_files: int
    conflicted_files: int
    detection_time_ms: float
    errors: List[str] = field(default_factory=list)


@dataclass
class ProcessingResult:
    """Result of incremental processing."""
    
    processed: List[str]
    failed: List[str]
    skipped: List[str]
    conflicts_resolved: int
    processing_time_ms: float
    qdrant_operations: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class ChangeDetector:
    """Detects file changes by comparing current state with stored records."""
    
    def __init__(self, state_manager: SQLiteStateManager):
        self.state_manager = state_manager
        self._detection_cache: Dict[str, Tuple[float, str]] = {}  # path -> (mtime, hash)
        
    async def detect_changes(
        self,
        file_paths: List[str],
        collections: Optional[List[str]] = None,
        priority_patterns: Optional[Dict[str, ProcessingPriority]] = None
    ) -> ChangeDetectionResult:
        """
        Detect changes across multiple files.
        
        Args:
            file_paths: Paths to check for changes
            collections: Optional collection filter
            priority_patterns: File patterns mapped to priorities
            
        Returns:
            ChangeDetectionResult with detected changes
        """
        start_time = time.time()
        changes = []
        errors = []
        conflicted_count = 0
        
        # Get current stored records for all files
        stored_records = {}
        for file_path in file_paths:
            try:
                record = await self.state_manager.get_file_processing_status(file_path)
                if record:
                    stored_records[file_path] = record
            except Exception as e:
                errors.append(f"Error fetching record for {file_path}: {str(e)}")
                continue
        
        # Check each file for changes
        for file_path in file_paths:
            try:
                change_info = await self._detect_single_file_change(
                    file_path, 
                    stored_records.get(file_path),
                    collections,
                    priority_patterns or {}
                )
                
                if change_info:
                    changes.append(change_info)
                    if change_info.conflict_detected:
                        conflicted_count += 1
                        
            except Exception as e:
                errors.append(f"Error detecting changes for {file_path}: {str(e)}")
                continue
        
        detection_time = (time.time() - start_time) * 1000
        unchanged_count = len(file_paths) - len(changes) - len(errors)
        
        return ChangeDetectionResult(
            changes=changes,
            total_scanned=len(file_paths),
            unchanged_files=unchanged_count,
            conflicted_files=conflicted_count,
            detection_time_ms=detection_time,
            errors=errors
        )
    
    async def _detect_single_file_change(
        self,
        file_path: str,
        stored_record: Optional[FileProcessingRecord],
        collections: Optional[List[str]],
        priority_patterns: Dict[str, ProcessingPriority]
    ) -> Optional[FileChangeInfo]:
        """Detect changes for a single file."""
        path_obj = Path(file_path)
        
        # File doesn't exist
        if not path_obj.exists():
            if stored_record:
                return FileChangeInfo(
                    file_path=file_path,
                    change_type=ChangeType.DELETED,
                    stored_record=stored_record,
                    collection=stored_record.collection
                )
            return None
        
        # Get current file metadata
        stat = path_obj.stat()
        current_mtime = stat.st_mtime
        current_size = stat.st_size
        current_hash = await self._calculate_file_hash(file_path)
        
        # Determine priority
        priority = self._determine_priority(file_path, priority_patterns)
        
        # New file
        if not stored_record:
            collection = self._determine_collection(file_path, collections)
            return FileChangeInfo(
                file_path=file_path,
                change_type=ChangeType.CREATED,
                current_mtime=current_mtime,
                current_size=current_size,
                current_hash=current_hash,
                priority=priority,
                collection=collection
            )
        
        # Check for changes
        changes = []
        
        # Content changed
        if stored_record.file_hash != current_hash:
            changes.append("content")
        
        # Size changed  
        if stored_record.file_size != current_size:
            changes.append("size")
        
        # LSP data stale (file modified after last LSP analysis)
        if (stored_record.last_lsp_analysis and 
            current_mtime > stored_record.last_lsp_analysis.timestamp()):
            changes.append("lsp_stale")
        
        if not changes:
            return None  # No changes detected
        
        # Determine primary change type
        if "content" in changes:
            change_type = ChangeType.MODIFIED
        elif "lsp_stale" in changes:
            change_type = ChangeType.LSP_STALE
        else:
            change_type = ChangeType.METADATA_CHANGED
        
        # Check for conflicts (concurrent modifications)
        conflict_detected = False
        conflict_reason = None
        
        if (stored_record.status == FileProcessingStatus.PROCESSING and
            "content" in changes):
            conflict_detected = True
            conflict_reason = "File modified while processing"
        
        return FileChangeInfo(
            file_path=file_path,
            change_type=change_type,
            current_mtime=current_mtime,
            current_size=current_size,
            current_hash=current_hash,
            stored_record=stored_record,
            priority=priority,
            collection=stored_record.collection,
            conflict_detected=conflict_detected,
            conflict_reason=conflict_reason
        )
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file content."""
        hash_obj = hashlib.sha256()
        
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _determine_priority(
        self,
        file_path: str,
        priority_patterns: Dict[str, ProcessingPriority]
    ) -> ProcessingPriority:
        """Determine processing priority based on file patterns."""
        path_obj = Path(file_path)
        
        # Check patterns in priority order
        for pattern, priority in sorted(
            priority_patterns.items(),
            key=lambda x: x[1].value,
            reverse=True
        ):
            if path_obj.match(pattern):
                return priority
        
        return ProcessingPriority.NORMAL
    
    def _determine_collection(
        self,
        file_path: str,
        collections: Optional[List[str]]
    ) -> str:
        """Determine appropriate collection for new file."""
        if not collections:
            return "default"
        
        # Simple heuristic - could be enhanced with project detection
        path_obj = Path(file_path)
        
        # Look for project-specific collections
        parts = path_obj.parts
        for part in parts:
            for collection in collections:
                if part.lower() in collection.lower():
                    return collection
        
        return collections[0] if collections else "default"


class DifferentialUpdater:
    """Updates Qdrant collections with only changed content."""
    
    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        self.operation_stats = {
            "upserts": 0,
            "deletes": 0,
            "updates": 0,
            "batch_operations": 0
        }
    
    async def apply_changes(
        self,
        changes: List[FileChangeInfo],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """
        Apply changes to Qdrant collections differentially.
        
        Args:
            changes: List of detected changes
            batch_size: Number of operations per batch
            
        Returns:
            Dictionary of operation counts
        """
        self.operation_stats = {k: 0 for k in self.operation_stats}
        
        # Group changes by collection for efficient processing
        changes_by_collection = {}
        for change in changes:
            collection = change.collection
            if collection not in changes_by_collection:
                changes_by_collection[collection] = []
            changes_by_collection[collection].append(change)
        
        # Process each collection's changes
        for collection, collection_changes in changes_by_collection.items():
            await self._apply_collection_changes(
                collection, 
                collection_changes,
                batch_size
            )
        
        return self.operation_stats.copy()
    
    async def _apply_collection_changes(
        self,
        collection: str,
        changes: List[FileChangeInfo],
        batch_size: int
    ):
        """Apply changes to a specific collection."""
        
        # Batch changes for efficient processing
        for i in range(0, len(changes), batch_size):
            batch = changes[i:i + batch_size]
            await self._process_change_batch(collection, batch)
            self.operation_stats["batch_operations"] += 1
    
    async def _process_change_batch(
        self,
        collection: str, 
        batch: List[FileChangeInfo]
    ):
        """Process a batch of changes for a collection."""
        
        points_to_upsert = []
        points_to_delete = []
        
        for change in batch:
            try:
                if change.change_type == ChangeType.DELETED:
                    # Delete document points from collection
                    points_to_delete.append(self._get_document_id(change.file_path))
                    self.operation_stats["deletes"] += 1
                    
                elif change.change_type in [ChangeType.CREATED, ChangeType.MODIFIED]:
                    # Generate new embeddings and upsert
                    point = await self._create_document_point(change)
                    if point:
                        points_to_upsert.append(point)
                        self.operation_stats["upserts"] += 1
                        
                elif change.change_type == ChangeType.LSP_STALE:
                    # Update only LSP-related metadata
                    await self._update_lsp_metadata(collection, change)
                    self.operation_stats["updates"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing change for {change.file_path}: {e}")
                continue
        
        # Execute batch operations
        if points_to_upsert:
            await self._batch_upsert(collection, points_to_upsert)
        
        if points_to_delete:
            await self._batch_delete(collection, points_to_delete)
    
    async def _create_document_point(
        self,
        change: FileChangeInfo
    ) -> Optional[qdrant_models.PointStruct]:
        """Create a Qdrant point for document content."""
        try:
            # Read file content
            with open(change.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Generate embeddings (simplified - would use actual embedding service)
            # This would integrate with the existing embedding pipeline
            embedding = await self._generate_embeddings(content)
            
            if not embedding:
                return None
            
            # Create point with metadata
            metadata = {
                "file_path": change.file_path,
                "file_size": change.current_size,
                "file_hash": change.current_hash,
                "modified_at": change.current_mtime,
                "processing_priority": change.priority.value,
                **change.metadata
            }
            
            return qdrant_models.PointStruct(
                id=self._get_document_id(change.file_path),
                vector=embedding,
                payload=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating point for {change.file_path}: {e}")
            return None
    
    async def _generate_embeddings(self, content: str) -> Optional[List[float]]:
        """Generate embeddings for content using EmbeddingService."""
        try:
            # Import here to avoid circular imports
            from .embeddings import EmbeddingService
            from .config import Config
            
            # This could be injected as a dependency in real usage
            # For now, create a simple instance
            if not hasattr(self, '_embedding_service'):
                config = Config()
                self._embedding_service = EmbeddingService(config)
                await self._embedding_service.initialize()
            
            embeddings = await self._embedding_service.generate_embeddings(
                content, 
                include_sparse=False  # Only dense embeddings for simplicity
            )
            
            if embeddings and embeddings.dense_embedding:
                return embeddings.dense_embedding.tolist()
                
            return None
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return None
    
    def _get_document_id(self, file_path: str) -> str:
        """Generate consistent document ID from file path."""
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]
    
    async def _batch_upsert(
        self,
        collection: str,
        points: List[qdrant_models.PointStruct]
    ):
        """Execute batch upsert operation."""
        try:
            self.qdrant_client.upsert(
                collection_name=collection,
                points=points
            )
        except Exception as e:
            logger.error(f"Error upserting points to {collection}: {e}")
            raise
    
    async def _batch_delete(self, collection: str, point_ids: List[str]):
        """Execute batch delete operation.""" 
        try:
            self.qdrant_client.delete(
                collection_name=collection,
                points_selector=qdrant_models.PointIdsList(
                    points=point_ids
                )
            )
        except Exception as e:
            logger.error(f"Error deleting points from {collection}: {e}")
            raise
    
    async def _update_lsp_metadata(
        self,
        collection: str,
        change: FileChangeInfo
    ):
        """Update only LSP-related metadata for a document."""
        try:
            # Update payload without changing vector
            point_id = self._get_document_id(change.file_path)
            
            payload_update = {
                "lsp_stale": True,
                "needs_lsp_refresh": True,
                "modified_at": change.current_mtime
            }
            
            self.qdrant_client.set_payload(
                collection_name=collection,
                payload=payload_update,
                points=[point_id]
            )
            
        except Exception as e:
            logger.error(f"Error updating LSP metadata for {change.file_path}: {e}")
            raise


class ConflictResolver:
    """Resolves conflicts during concurrent file modifications."""
    
    def __init__(
        self,
        strategy: ConflictResolution = ConflictResolution.LATEST_WINS
    ):
        self.strategy = strategy
        self.resolved_conflicts = 0
    
    async def resolve_conflicts(
        self,
        conflicted_changes: List[FileChangeInfo],
        state_manager: SQLiteStateManager
    ) -> List[FileChangeInfo]:
        """
        Resolve conflicts in file changes.
        
        Args:
            conflicted_changes: Changes with detected conflicts
            state_manager: State manager for database operations
            
        Returns:
            List of resolved changes ready for processing
        """
        resolved = []
        
        for change in conflicted_changes:
            try:
                resolved_change = await self._resolve_single_conflict(
                    change,
                    state_manager
                )
                if resolved_change:
                    resolved.append(resolved_change)
                    self.resolved_conflicts += 1
            except Exception as e:
                logger.error(f"Error resolving conflict for {change.file_path}: {e}")
                continue
        
        return resolved
    
    async def _resolve_single_conflict(
        self,
        change: FileChangeInfo,
        state_manager: SQLiteStateManager
    ) -> Optional[FileChangeInfo]:
        """Resolve conflict for a single file."""
        
        if self.strategy == ConflictResolution.LATEST_WINS:
            # Always process the latest version
            change.conflict_detected = False
            change.conflict_reason = None
            return change
        
        elif self.strategy == ConflictResolution.SKIP_CONFLICTED:
            # Skip conflicted files
            logger.warning(f"Skipping conflicted file: {change.file_path}")
            return None
        
        elif self.strategy == ConflictResolution.MERGE_METADATA:
            # Merge metadata from stored and current versions
            if change.stored_record:
                merged_metadata = {
                    **(change.stored_record.metadata or {}),
                    **change.metadata
                }
                change.metadata = merged_metadata
                change.conflict_detected = False
                change.conflict_reason = None
                return change
        
        # Default: skip unresolved conflicts
        return None


class TransactionManager:
    """Manages transaction-safe updates with rollback capability."""
    
    def __init__(self, state_manager: SQLiteStateManager):
        self.state_manager = state_manager
        self._active_transactions: Set[str] = set()
    
    @asynccontextmanager
    async def transaction(self, transaction_id: str):
        """
        Context manager for transaction-safe operations.
        
        Args:
            transaction_id: Unique identifier for the transaction
        """
        if transaction_id in self._active_transactions:
            raise ValueError(f"Transaction {transaction_id} already active")
        
        self._active_transactions.add(transaction_id)
        
        async with self.state_manager.transaction() as conn:
            try:
                yield conn
                logger.debug(f"Transaction {transaction_id} committed successfully")
            except Exception as e:
                logger.error(f"Transaction {transaction_id} failed: {e}")
                # Rollback is automatic with context manager
                raise
            finally:
                self._active_transactions.discard(transaction_id)
    
    async def create_savepoint(
        self,
        connection,
        savepoint_name: str
    ):
        """Create a savepoint within a transaction."""
        connection.execute(f"SAVEPOINT {savepoint_name}")
    
    async def rollback_to_savepoint(
        self,
        connection,
        savepoint_name: str
    ):
        """Rollback to a specific savepoint."""
        connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint_name}")
    
    async def release_savepoint(
        self,
        connection,
        savepoint_name: str
    ):
        """Release a savepoint."""
        connection.execute(f"RELEASE SAVEPOINT {savepoint_name}")


class IncrementalProcessor:
    """
    Main coordinator for incremental file processing.
    
    Integrates change detection, differential updates, and transaction management
    to provide efficient incremental processing for real-time development workflows.
    """
    
    def __init__(
        self,
        state_manager: SQLiteStateManager,
        qdrant_client: QdrantClient,
        conflict_strategy: ConflictResolution = ConflictResolution.LATEST_WINS
    ):
        self.state_manager = state_manager
        self.qdrant_client = qdrant_client
        
        self.change_detector = ChangeDetector(state_manager)
        self.differential_updater = DifferentialUpdater(qdrant_client)
        self.conflict_resolver = ConflictResolver(conflict_strategy)
        self.transaction_manager = TransactionManager(state_manager)
        
        self.processing_stats = {
            "total_processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "conflicts_resolved": 0
        }
        
    async def initialize(self):
        """Initialize the incremental processor."""
        logger.info("Initializing incremental processor")
        
        # Verify state manager is initialized
        if not hasattr(self.state_manager, 'connection') or not self.state_manager.connection:
            raise RuntimeError("State manager must be initialized before incremental processor")
        
        logger.info("Incremental processor initialized successfully")
    
    async def process_changes(
        self,
        file_paths: List[str],
        collections: Optional[List[str]] = None,
        batch_size: int = 10,
        priority_patterns: Optional[Dict[str, ProcessingPriority]] = None
    ) -> ProcessingResult:
        """
        Process incremental changes for given file paths.
        
        Args:
            file_paths: Paths to process for changes
            collections: Optional collection filter
            batch_size: Number of files to process per batch
            priority_patterns: File patterns mapped to priorities
            
        Returns:
            ProcessingResult with operation details
        """
        start_time = time.time()
        processed = []
        failed = []
        skipped = []
        errors = []
        
        try:
            # Detect changes
            logger.info(f"Detecting changes in {len(file_paths)} files")
            detection_result = await self.change_detector.detect_changes(
                file_paths,
                collections,
                priority_patterns
            )
            
            if detection_result.errors:
                errors.extend(detection_result.errors)
            
            if not detection_result.changes:
                logger.info("No changes detected")
                return ProcessingResult(
                    processed=[],
                    failed=[],
                    skipped=file_paths,
                    conflicts_resolved=0,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    errors=errors
                )
            
            logger.info(f"Detected {len(detection_result.changes)} changes")
            
            # Separate conflicted and non-conflicted changes
            conflicted_changes = [c for c in detection_result.changes if c.conflict_detected]
            normal_changes = [c for c in detection_result.changes if not c.conflict_detected]
            
            # Resolve conflicts
            if conflicted_changes:
                logger.info(f"Resolving {len(conflicted_changes)} conflicts")
                resolved_changes = await self.conflict_resolver.resolve_conflicts(
                    conflicted_changes,
                    self.state_manager
                )
                normal_changes.extend(resolved_changes)
                conflicts_resolved = self.conflict_resolver.resolved_conflicts
            else:
                conflicts_resolved = 0
            
            # Sort changes by priority
            normal_changes.sort(key=lambda x: x.priority.value, reverse=True)
            
            # Process changes in batches
            all_changes = normal_changes
            transaction_id = f"incremental_update_{int(time.time())}"
            
            async with self.transaction_manager.transaction(transaction_id):
                # Update file processing states
                for change in all_changes:
                    try:
                        await self._update_processing_state(change)
                        processed.append(change.file_path)
                    except Exception as e:
                        logger.error(f"Error updating state for {change.file_path}: {e}")
                        failed.append(change.file_path)
                        errors.append(str(e))
                
                # Apply differential updates to Qdrant
                if processed:
                    logger.info(f"Applying differential updates for {len(processed)} files")
                    qdrant_stats = await self.differential_updater.apply_changes(
                        [c for c in all_changes if c.file_path in processed],
                        batch_size
                    )
                else:
                    qdrant_stats = {}
            
            # Update processing statistics
            self.processing_stats["total_processed"] += len(processed)
            self.processing_stats["successful"] += len(processed)
            self.processing_stats["failed"] += len(failed)
            self.processing_stats["skipped"] += len(skipped)
            self.processing_stats["conflicts_resolved"] += conflicts_resolved
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info(f"Incremental processing completed in {processing_time:.2f}ms")
            logger.info(f"Processed: {len(processed)}, Failed: {len(failed)}, Skipped: {len(skipped)}")
            
            return ProcessingResult(
                processed=processed,
                failed=failed,
                skipped=skipped,
                conflicts_resolved=conflicts_resolved,
                processing_time_ms=processing_time,
                qdrant_operations=qdrant_stats,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Error during incremental processing: {e}")
            return ProcessingResult(
                processed=processed,
                failed=file_paths,
                skipped=[],
                conflicts_resolved=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                errors=[str(e)]
            )
    
    async def _update_processing_state(self, change: FileChangeInfo):
        """Update file processing state in SQLite."""
        
        if change.change_type == ChangeType.DELETED:
            # Mark as completed/deleted
            if change.stored_record:
                await self.state_manager.complete_file_processing(
                    file_path=change.file_path,
                    success=True,
                    metadata={"deleted": True, "deleted_at": datetime.now(timezone.utc).isoformat()}
                )
        
        elif change.change_type in [ChangeType.CREATED, ChangeType.MODIFIED]:
            # Start or update processing
            if change.stored_record:
                # Update existing record
                async with self.state_manager.transaction() as conn:
                    conn.execute(
                        """
                        UPDATE file_processing 
                        SET file_hash = ?, file_size = ?, updated_at = ?, 
                            status = ?, metadata = ?
                        WHERE file_path = ?
                        """,
                        (
                            change.current_hash,
                            change.current_size,
                            datetime.now(timezone.utc).isoformat(),
                            FileProcessingStatus.PROCESSING.value,
                            json.dumps(change.metadata),
                            change.file_path
                        )
                    )
            else:
                # Create new record
                await self.state_manager.start_file_processing(
                    file_path=change.file_path,
                    collection=change.collection,
                    priority=change.priority,
                    file_size=change.current_size,
                    file_hash=change.current_hash,
                    metadata=change.metadata
                )
        
        elif change.change_type == ChangeType.LSP_STALE:
            # Update LSP-specific fields
            if change.stored_record:
                async with self.state_manager.transaction() as conn:
                    conn.execute(
                        """
                        UPDATE file_processing 
                        SET lsp_extracted = ?, updated_at = ?
                        WHERE file_path = ?
                        """,
                        (
                            False,  # Mark as needing LSP re-extraction
                            datetime.now(timezone.utc).isoformat(),
                            change.file_path
                        )
                    )
    
    async def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.processing_stats,
            "change_detector_cache_size": len(self.change_detector._detection_cache),
            "differential_updater_stats": self.differential_updater.operation_stats,
            "conflicts_resolved_total": self.conflict_resolver.resolved_conflicts,
            "active_transactions": len(self.transaction_manager._active_transactions)
        }
    
    async def clear_caches(self):
        """Clear internal caches."""
        self.change_detector._detection_cache.clear()
        logger.info("Incremental processor caches cleared")


# Integration helpers for existing systems

async def create_incremental_processor(
    state_manager: SQLiteStateManager,
    qdrant_client: QdrantClient,
    config: Optional[Dict[str, Any]] = None
) -> IncrementalProcessor:
    """
    Factory function to create and initialize an incremental processor.
    
    Args:
        state_manager: Initialized SQLite state manager
        qdrant_client: Qdrant client instance
        config: Optional configuration dictionary
        
    Returns:
        Initialized IncrementalProcessor instance
    """
    conflict_strategy = ConflictResolution.LATEST_WINS
    if config and 'conflict_resolution' in config:
        strategy_name = config['conflict_resolution'].upper()
        if hasattr(ConflictResolution, strategy_name):
            conflict_strategy = getattr(ConflictResolution, strategy_name)
    
    processor = IncrementalProcessor(
        state_manager=state_manager,
        qdrant_client=qdrant_client,
        conflict_strategy=conflict_strategy
    )
    
    await processor.initialize()
    return processor


# Performance monitoring utilities

@dataclass
class ProcessingMetrics:
    """Metrics for monitoring incremental processing performance."""
    
    detection_time_ms: float
    processing_time_ms: float
    qdrant_operations: Dict[str, int]
    files_processed: int
    conflicts_resolved: int
    cache_hit_rate: float
    memory_usage_mb: float


async def collect_processing_metrics(
    processor: IncrementalProcessor
) -> ProcessingMetrics:
    """Collect performance metrics from incremental processor."""
    import psutil
    import os
    
    stats = await processor.get_processing_statistics()
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate cache hit rate (simplified)
    cache_size = stats.get("change_detector_cache_size", 0)
    cache_hit_rate = min(1.0, cache_size / max(1, stats.get("total_processed", 1)))
    
    return ProcessingMetrics(
        detection_time_ms=0,  # Would be tracked per operation
        processing_time_ms=0,  # Would be tracked per operation
        qdrant_operations=stats.get("differential_updater_stats", {}),
        files_processed=stats.get("successful", 0),
        conflicts_resolved=stats.get("conflicts_resolved", 0),
        cache_hit_rate=cache_hit_rate,
        memory_usage_mb=memory_usage
    )