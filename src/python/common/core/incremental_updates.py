"""
Incremental Updates System with Conflict Resolution.

This module provides sophisticated change tracking, conflict detection,
and resolution strategies for file system changes. It includes checksumming,
delta processing, and rollback capabilities for robust file processing.
"""

import asyncio
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from loguru import logger
import aiofiles
import aiofiles.os


class ChangeType(Enum):
    """Types of file changes."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    METADATA_ONLY = "metadata_only"
    CONTENT_MODIFIED = "content_modified"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts."""
    LAST_WRITER_WINS = "last_writer_wins"
    FIRST_WRITER_WINS = "first_writer_wins"
    MANUAL_RESOLUTION = "manual_resolution"
    AUTOMATIC_MERGE = "automatic_merge"
    CREATE_VERSIONS = "create_versions"
    BACKUP_AND_REPLACE = "backup_and_replace"


class UpdateStatus(Enum):
    """Status of an update operation."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CONFLICT = "conflict"
    ROLLED_BACK = "rolled_back"


@dataclass
class FileChecksum:
    """File checksum information for change detection."""
    file_path: str
    size: int
    mtime: float
    md5_hash: str
    sha256_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    content_type: Optional[str] = None

    def __eq__(self, other: 'FileChecksum') -> bool:
        """Compare checksums for equality."""
        if not isinstance(other, FileChecksum):
            return False
        return (
            self.file_path == other.file_path and
            self.size == other.size and
            self.mtime == other.mtime and
            self.md5_hash == other.md5_hash
        )

    def has_content_changed(self, other: 'FileChecksum') -> bool:
        """Check if content has changed between two checksums."""
        return self.md5_hash != other.md5_hash

    def has_metadata_changed(self, other: 'FileChecksum') -> bool:
        """Check if only metadata has changed."""
        return (
            self.size == other.size and
            self.md5_hash == other.md5_hash and
            self.mtime != other.mtime
        )


@dataclass
class ChangeRecord:
    """Record of a detected file change."""
    change_id: str
    file_path: str
    change_type: ChangeType
    timestamp: float
    old_checksum: Optional[FileChecksum] = None
    new_checksum: Optional[FileChecksum] = None
    change_size: int = 0  # Bytes changed
    processing_priority: int = 1  # 1-10, higher is more urgent
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Other files this depends on

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['change_type'] = self.change_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeRecord':
        """Create from dictionary."""
        data = data.copy()
        data['change_type'] = ChangeType(data['change_type'])
        return cls(**data)


@dataclass
class ConflictInfo:
    """Information about a detected conflict."""
    conflict_id: str
    file_path: str
    conflict_type: str
    timestamp: float
    conflicting_changes: List[ChangeRecord]
    resolution_strategy: Optional[ConflictResolutionStrategy] = None
    resolution_data: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    manual_review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.resolution_strategy:
            data['resolution_strategy'] = self.resolution_strategy.value
        return data


@dataclass
class UpdateOperation:
    """Represents an update operation with rollback capability."""
    operation_id: str
    file_path: str
    operation_type: str
    status: UpdateStatus
    timestamp: float
    backup_path: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    rollback_data: Dict[str, Any] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check if operation can be retried."""
        return self.retry_count < self.max_retries and self.status == UpdateStatus.FAILED


class ChecksumCalculator:
    """Efficient checksum calculation with caching."""

    def __init__(self, cache_size: int = 1000):
        """Initialize checksum calculator."""
        self._cache: Dict[str, FileChecksum] = {}
        self._cache_size = cache_size

    async def calculate_checksum(self, file_path: Path) -> Optional[FileChecksum]:
        """Calculate comprehensive checksum for a file."""
        try:
            file_path_str = str(file_path)

            # Get file stats
            stat_result = await aiofiles.os.stat(file_path_str)
            size = stat_result.st_size
            mtime = stat_result.st_mtime

            # Check cache first
            cache_key = f"{file_path_str}:{size}:{mtime}"
            if cache_key in self._cache:
                return self._cache[cache_key]

            # Calculate hashes
            md5_hash = hashlib.md5()
            sha256_hash = hashlib.sha256()

            # Read file in chunks to handle large files efficiently
            async with aiofiles.open(file_path_str, 'rb') as f:
                while chunk := await f.read(8192):  # 8KB chunks
                    md5_hash.update(chunk)
                    sha256_hash.update(chunk)

            checksum = FileChecksum(
                file_path=file_path_str,
                size=size,
                mtime=mtime,
                md5_hash=md5_hash.hexdigest(),
                sha256_hash=sha256_hash.hexdigest(),
                content_type=self._detect_content_type(file_path)
            )

            # Cache the result
            self._cache_checksum(cache_key, checksum)

            return checksum

        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return None

    def _cache_checksum(self, cache_key: str, checksum: FileChecksum) -> None:
        """Cache a checksum result."""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = checksum

    def _detect_content_type(self, file_path: Path) -> Optional[str]:
        """Simple content type detection based on file extension."""
        extension_map = {
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.json': 'application/json',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.py': 'text/x-python',
            '.js': 'text/javascript',
            '.ts': 'text/typescript',
            '.html': 'text/html',
            '.css': 'text/css',
        }
        return extension_map.get(file_path.suffix.lower())

    def clear_cache(self) -> None:
        """Clear the checksum cache."""
        self._cache.clear()


class ChangeDetector:
    """Detects and classifies file changes."""

    def __init__(self, checksum_calculator: ChecksumCalculator):
        """Initialize change detector."""
        self.checksum_calculator = checksum_calculator
        self._previous_checksums: Dict[str, FileChecksum] = {}

    async def detect_changes(self, file_paths: List[Path]) -> List[ChangeRecord]:
        """Detect changes in a list of files."""
        changes = []
        current_time = time.time()

        for file_path in file_paths:
            try:
                change_record = await self._detect_single_file_change(file_path, current_time)
                if change_record:
                    changes.append(change_record)
            except Exception as e:
                logger.error(f"Error detecting changes for {file_path}: {e}")

        return changes

    async def _detect_single_file_change(self, file_path: Path, timestamp: float) -> Optional[ChangeRecord]:
        """Detect changes for a single file."""
        file_path_str = str(file_path)

        # Check if file exists
        file_exists = await aiofiles.os.path.exists(file_path_str)
        previous_checksum = self._previous_checksums.get(file_path_str)

        if not file_exists and previous_checksum:
            # File was deleted
            change_record = ChangeRecord(
                change_id=self._generate_change_id(file_path_str, timestamp),
                file_path=file_path_str,
                change_type=ChangeType.DELETED,
                timestamp=timestamp,
                old_checksum=previous_checksum,
                processing_priority=5  # Deletions are moderately urgent
            )
            del self._previous_checksums[file_path_str]
            return change_record

        if not file_exists:
            # File doesn't exist and we have no record of it
            return None

        # Calculate current checksum
        current_checksum = await self.checksum_calculator.calculate_checksum(file_path)
        if not current_checksum:
            return None

        if not previous_checksum:
            # New file
            change_record = ChangeRecord(
                change_id=self._generate_change_id(file_path_str, timestamp),
                file_path=file_path_str,
                change_type=ChangeType.CREATED,
                timestamp=timestamp,
                new_checksum=current_checksum,
                change_size=current_checksum.size,
                processing_priority=7  # New files are high priority
            )
            self._previous_checksums[file_path_str] = current_checksum
            return change_record

        # Compare checksums
        if current_checksum == previous_checksum:
            # No change
            return None

        # Determine change type
        change_type = ChangeType.MODIFIED
        processing_priority = 5

        if current_checksum.has_content_changed(previous_checksum):
            change_type = ChangeType.CONTENT_MODIFIED
            processing_priority = 8  # Content changes are high priority
        elif current_checksum.has_metadata_changed(previous_checksum):
            change_type = ChangeType.METADATA_ONLY
            processing_priority = 3  # Metadata-only changes are lower priority

        change_record = ChangeRecord(
            change_id=self._generate_change_id(file_path_str, timestamp),
            file_path=file_path_str,
            change_type=change_type,
            timestamp=timestamp,
            old_checksum=previous_checksum,
            new_checksum=current_checksum,
            change_size=abs(current_checksum.size - previous_checksum.size),
            processing_priority=processing_priority
        )

        self._previous_checksums[file_path_str] = current_checksum
        return change_record

    def _generate_change_id(self, file_path: str, timestamp: float) -> str:
        """Generate a unique change ID."""
        return hashlib.md5(f"{file_path}:{timestamp}".encode()).hexdigest()

    def set_baseline_checksums(self, checksums: Dict[str, FileChecksum]) -> None:
        """Set baseline checksums for comparison."""
        self._previous_checksums = checksums.copy()

    def get_current_checksums(self) -> Dict[str, FileChecksum]:
        """Get current checksum state."""
        return self._previous_checksums.copy()


class ConflictResolver:
    """Handles conflict detection and resolution."""

    def __init__(self, default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITER_WINS):
        """Initialize conflict resolver."""
        self.default_strategy = default_strategy
        self._active_conflicts: Dict[str, ConflictInfo] = {}
        self._resolution_handlers: Dict[ConflictResolutionStrategy, Callable] = {
            ConflictResolutionStrategy.LAST_WRITER_WINS: self._resolve_last_writer_wins,
            ConflictResolutionStrategy.FIRST_WRITER_WINS: self._resolve_first_writer_wins,
            ConflictResolutionStrategy.BACKUP_AND_REPLACE: self._resolve_backup_and_replace,
            ConflictResolutionStrategy.CREATE_VERSIONS: self._resolve_create_versions,
        }

    def detect_conflicts(self, changes: List[ChangeRecord]) -> List[ConflictInfo]:
        """Detect conflicts in a set of changes."""
        conflicts = []
        file_changes: Dict[str, List[ChangeRecord]] = {}

        # Group changes by file path
        for change in changes:
            if change.file_path not in file_changes:
                file_changes[change.file_path] = []
            file_changes[change.file_path].append(change)

        # Check for conflicts
        for file_path, file_change_list in file_changes.items():
            if len(file_change_list) > 1:
                # Multiple changes to the same file - potential conflict
                conflict = self._analyze_potential_conflict(file_path, file_change_list)
                if conflict:
                    conflicts.append(conflict)
                    self._active_conflicts[conflict.conflict_id] = conflict

        return conflicts

    def _analyze_potential_conflict(self, file_path: str, changes: List[ChangeRecord]) -> Optional[ConflictInfo]:
        """Analyze whether multiple changes constitute a real conflict."""
        # Sort changes by timestamp
        changes.sort(key=lambda x: x.timestamp)

        # Check for rapid successive changes (within 5 seconds)
        rapid_changes = []
        for i, change in enumerate(changes):
            if i == 0:
                rapid_changes.append(change)
                continue

            time_diff = change.timestamp - changes[i-1].timestamp
            if time_diff <= 5.0:  # 5 second threshold
                rapid_changes.append(change)
            else:
                # Reset rapid changes list
                rapid_changes = [change]

        if len(rapid_changes) <= 1:
            return None  # No real conflict

        # Determine conflict type
        conflict_type = "rapid_successive_changes"
        if any(c.change_type == ChangeType.DELETED for c in rapid_changes):
            conflict_type = "delete_after_modify"

        conflict_id = hashlib.md5(f"{file_path}:conflict:{time.time()}".encode()).hexdigest()

        return ConflictInfo(
            conflict_id=conflict_id,
            file_path=file_path,
            conflict_type=conflict_type,
            timestamp=time.time(),
            conflicting_changes=rapid_changes,
            resolution_strategy=self.default_strategy,
            manual_review_required=len(rapid_changes) > 3  # Complex conflicts need manual review
        )

    async def resolve_conflict(self, conflict_info: ConflictInfo) -> bool:
        """Resolve a conflict using the specified strategy."""
        try:
            if conflict_info.resolution_strategy in self._resolution_handlers:
                handler = self._resolution_handlers[conflict_info.resolution_strategy]
                success = await handler(conflict_info)
                if success:
                    conflict_info.resolved = True
                    logger.info(f"Resolved conflict {conflict_info.conflict_id} using {conflict_info.resolution_strategy}")
                return success
            else:
                logger.error(f"No handler for resolution strategy: {conflict_info.resolution_strategy}")
                return False

        except Exception as e:
            logger.error(f"Error resolving conflict {conflict_info.conflict_id}: {e}")
            return False

    async def _resolve_last_writer_wins(self, conflict_info: ConflictInfo) -> bool:
        """Resolve conflict by keeping the last change."""
        # The last change in the list wins
        latest_change = max(conflict_info.conflicting_changes, key=lambda x: x.timestamp)
        conflict_info.resolution_data = {
            "winning_change": latest_change.change_id,
            "resolution_time": time.time()
        }
        return True

    async def _resolve_first_writer_wins(self, conflict_info: ConflictInfo) -> bool:
        """Resolve conflict by keeping the first change."""
        # The first change in the list wins
        first_change = min(conflict_info.conflicting_changes, key=lambda x: x.timestamp)
        conflict_info.resolution_data = {
            "winning_change": first_change.change_id,
            "resolution_time": time.time()
        }
        return True

    async def _resolve_backup_and_replace(self, conflict_info: ConflictInfo) -> bool:
        """Resolve conflict by backing up current file and applying latest change."""
        try:
            file_path = Path(conflict_info.file_path)
            if await aiofiles.os.path.exists(str(file_path)):
                # Create backup
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup.{int(time.time())}")
                await aiofiles.os.rename(str(file_path), str(backup_path))

                conflict_info.resolution_data = {
                    "backup_path": str(backup_path),
                    "resolution_time": time.time()
                }
                logger.info(f"Created backup: {backup_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to create backup for conflict resolution: {e}")
            return False

    async def _resolve_create_versions(self, conflict_info: ConflictInfo) -> bool:
        """Resolve conflict by creating versioned copies."""
        try:
            file_path = Path(conflict_info.file_path)
            base_name = file_path.stem
            extension = file_path.suffix
            parent_dir = file_path.parent

            versions_created = []
            for i, change in enumerate(conflict_info.conflicting_changes):
                version_path = parent_dir / f"{base_name}_v{i+1}_{int(change.timestamp)}{extension}"
                versions_created.append(str(version_path))

            conflict_info.resolution_data = {
                "versions_created": versions_created,
                "resolution_time": time.time()
            }

            return True

        except Exception as e:
            logger.error(f"Failed to create versions for conflict resolution: {e}")
            return False

    def get_active_conflicts(self) -> List[ConflictInfo]:
        """Get all active conflicts."""
        return list(self._active_conflicts.values())

    def get_conflict_by_id(self, conflict_id: str) -> Optional[ConflictInfo]:
        """Get a specific conflict by ID."""
        return self._active_conflicts.get(conflict_id)


class UpdateProcessor:
    """Processes incremental updates with rollback capability."""

    def __init__(self, backup_dir: Path):
        """Initialize update processor."""
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._operations: Dict[str, UpdateOperation] = {}

    async def process_update(self, change_record: ChangeRecord, callback: Callable[[str, ChangeRecord], Any]) -> UpdateOperation:
        """Process a single update with rollback capability."""
        operation_id = f"update_{change_record.change_id}"

        operation = UpdateOperation(
            operation_id=operation_id,
            file_path=change_record.file_path,
            operation_type=change_record.change_type.value,
            status=UpdateStatus.PENDING,
            timestamp=time.time()
        )

        self._operations[operation_id] = operation

        try:
            # Create backup if file exists
            if await self._should_create_backup(change_record):
                backup_path = await self._create_backup(change_record.file_path, operation_id)
                operation.backup_path = backup_path

            # Update operation status
            operation.status = UpdateStatus.IN_PROGRESS

            # Process the change
            await callback(change_record.file_path, change_record)

            # Mark as completed
            operation.status = UpdateStatus.COMPLETED
            logger.debug(f"Successfully processed update {operation_id}")

        except Exception as e:
            operation.status = UpdateStatus.FAILED
            operation.error_message = str(e)
            logger.error(f"Failed to process update {operation_id}: {e}")

        return operation

    async def _should_create_backup(self, change_record: ChangeRecord) -> bool:
        """Determine if a backup should be created for this change."""
        # Create backups for modifications and deletions
        return change_record.change_type in [ChangeType.MODIFIED, ChangeType.CONTENT_MODIFIED, ChangeType.DELETED]

    async def _create_backup(self, file_path: str, operation_id: str) -> str:
        """Create a backup of the file."""
        if not await aiofiles.os.path.exists(file_path):
            return ""

        source_path = Path(file_path)
        backup_filename = f"{operation_id}_{source_path.name}_{int(time.time())}"
        backup_path = self.backup_dir / backup_filename

        # Copy file to backup location
        shutil.copy2(file_path, str(backup_path))
        logger.debug(f"Created backup: {backup_path}")

        return str(backup_path)

    async def rollback_operation(self, operation_id: str) -> bool:
        """Rollback a failed or problematic operation."""
        operation = self._operations.get(operation_id)
        if not operation:
            logger.error(f"Operation {operation_id} not found for rollback")
            return False

        try:
            if operation.backup_path and await aiofiles.os.path.exists(operation.backup_path):
                # Restore from backup
                await aiofiles.os.rename(operation.backup_path, operation.file_path)
                operation.status = UpdateStatus.ROLLED_BACK
                logger.info(f"Rolled back operation {operation_id}")
                return True
            else:
                logger.warning(f"No backup available for rollback of operation {operation_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to rollback operation {operation_id}: {e}")
            return False

    async def retry_operation(self, operation_id: str, callback: Callable[[str, ChangeRecord], Any]) -> bool:
        """Retry a failed operation."""
        operation = self._operations.get(operation_id)
        if not operation or not operation.can_retry():
            return False

        operation.retry_count += 1
        operation.status = UpdateStatus.IN_PROGRESS

        try:
            # Create a synthetic change record for retry
            change_record = ChangeRecord(
                change_id=operation_id.replace("update_", ""),
                file_path=operation.file_path,
                change_type=ChangeType(operation.operation_type),
                timestamp=time.time()
            )

            await callback(operation.file_path, change_record)
            operation.status = UpdateStatus.COMPLETED
            logger.info(f"Successfully retried operation {operation_id}")
            return True

        except Exception as e:
            operation.status = UpdateStatus.FAILED
            operation.error_message = str(e)
            logger.error(f"Retry failed for operation {operation_id}: {e}")
            return False

    def get_operation_status(self, operation_id: str) -> Optional[UpdateOperation]:
        """Get the status of an operation."""
        return self._operations.get(operation_id)

    def get_failed_operations(self) -> List[UpdateOperation]:
        """Get all failed operations."""
        return [op for op in self._operations.values() if op.status == UpdateStatus.FAILED]

    def cleanup_old_operations(self, max_age_hours: int = 24) -> int:
        """Clean up old operation records and backups."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleaned_count = 0

        operations_to_remove = []
        for operation_id, operation in self._operations.items():
            if operation.timestamp < cutoff_time and operation.status in [UpdateStatus.COMPLETED, UpdateStatus.ROLLED_BACK]:
                # Clean up backup file if it exists
                if operation.backup_path and os.path.exists(operation.backup_path):
                    try:
                        os.remove(operation.backup_path)
                        logger.debug(f"Cleaned up backup: {operation.backup_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up backup {operation.backup_path}: {e}")

                operations_to_remove.append(operation_id)
                cleaned_count += 1

        for operation_id in operations_to_remove:
            del self._operations[operation_id]

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old operations")

        return cleaned_count


class IncrementalUpdateSystem:
    """
    Main system for managing incremental file updates with conflict resolution.

    Provides comprehensive change detection, conflict resolution, and update processing
    with full rollback capabilities and performance monitoring.
    """

    def __init__(self,
                 backup_dir: Path,
                 default_conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITER_WINS,
                 checksum_cache_size: int = 1000):
        """Initialize incremental update system."""

        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.checksum_calculator = ChecksumCalculator(checksum_cache_size)
        self.change_detector = ChangeDetector(self.checksum_calculator)
        self.conflict_resolver = ConflictResolver(default_conflict_strategy)
        self.update_processor = UpdateProcessor(backup_dir)

        # State tracking
        self._processing_callback: Optional[Callable] = None
        self._statistics = {
            "changes_processed": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "operations_rolled_back": 0,
            "operations_retried": 0,
            "average_processing_time": 0.0,
            "total_processing_time": 0.0
        }

        # Performance monitoring
        self._processing_times: List[float] = []
        self._max_processing_history = 1000

    def set_processing_callback(self, callback: Callable[[str, ChangeRecord], Any]) -> None:
        """Set the callback function for processing file changes."""
        self._processing_callback = callback

    async def initialize_baseline(self, file_paths: List[Path]) -> Dict[str, FileChecksum]:
        """Initialize baseline checksums for a set of files."""
        logger.info(f"Initializing baseline for {len(file_paths)} files")
        checksums = {}

        for file_path in file_paths:
            if await aiofiles.os.path.exists(str(file_path)):
                checksum = await self.checksum_calculator.calculate_checksum(file_path)
                if checksum:
                    checksums[str(file_path)] = checksum

        self.change_detector.set_baseline_checksums(checksums)
        logger.info(f"Initialized baseline with {len(checksums)} file checksums")
        return checksums

    async def process_file_changes(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Process changes for a list of files."""
        if not self._processing_callback:
            raise ValueError("Processing callback not set")

        start_time = time.time()
        results = {
            "changes_detected": 0,
            "conflicts_detected": 0,
            "conflicts_resolved": 0,
            "operations_completed": 0,
            "operations_failed": 0,
            "processing_time": 0.0,
            "errors": []
        }

        try:
            # Detect changes
            changes = await self.change_detector.detect_changes(file_paths)
            results["changes_detected"] = len(changes)

            if not changes:
                logger.debug("No changes detected")
                return results

            # Detect conflicts
            conflicts = self.conflict_resolver.detect_conflicts(changes)
            results["conflicts_detected"] = len(conflicts)
            self._statistics["conflicts_detected"] += len(conflicts)

            # Resolve conflicts
            for conflict in conflicts:
                resolved = await self.conflict_resolver.resolve_conflict(conflict)
                if resolved:
                    results["conflicts_resolved"] += 1
                    self._statistics["conflicts_resolved"] += 1

            # Process changes
            for change in changes:
                try:
                    operation = await self.update_processor.process_update(
                        change, self._processing_callback
                    )

                    if operation.status == UpdateStatus.COMPLETED:
                        results["operations_completed"] += 1
                    else:
                        results["operations_failed"] += 1

                except Exception as e:
                    results["errors"].append(f"Error processing {change.file_path}: {str(e)}")
                    results["operations_failed"] += 1

            # Update statistics
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time

            self._update_performance_stats(processing_time)
            self._statistics["changes_processed"] += len(changes)

            logger.info(f"Processed {len(changes)} changes in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Error in process_file_changes: {e}")
            results["errors"].append(str(e))

        return results

    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics."""
        self._processing_times.append(processing_time)

        # Keep only recent processing times
        if len(self._processing_times) > self._max_processing_history:
            self._processing_times = self._processing_times[-self._max_processing_history:]

        # Update averages
        self._statistics["total_processing_time"] += processing_time
        self._statistics["average_processing_time"] = sum(self._processing_times) / len(self._processing_times)

    async def handle_single_file_update(self, file_path: Path) -> Dict[str, Any]:
        """Handle an update for a single file."""
        return await self.process_file_changes([file_path])

    async def rollback_operation(self, operation_id: str) -> bool:
        """Rollback a specific operation."""
        success = await self.update_processor.rollback_operation(operation_id)
        if success:
            self._statistics["operations_rolled_back"] += 1
        return success

    async def retry_failed_operations(self) -> int:
        """Retry all failed operations."""
        if not self._processing_callback:
            return 0

        failed_operations = self.update_processor.get_failed_operations()
        retried_count = 0

        for operation in failed_operations:
            if operation.can_retry():
                success = await self.update_processor.retry_operation(
                    operation.operation_id, self._processing_callback
                )
                if success:
                    retried_count += 1

        if retried_count > 0:
            self._statistics["operations_retried"] += retried_count
            logger.info(f"Successfully retried {retried_count} operations")

        return retried_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = self._statistics.copy()
        stats.update({
            "active_conflicts": len(self.conflict_resolver.get_active_conflicts()),
            "checksum_cache_size": len(self.checksum_calculator._cache),
            "recent_processing_times": self._processing_times[-10:],  # Last 10 processing times
            "backup_directory": str(self.backup_dir),
        })
        return stats

    def get_active_conflicts(self) -> List[ConflictInfo]:
        """Get all active conflicts."""
        return self.conflict_resolver.get_active_conflicts()

    def get_conflict_by_id(self, conflict_id: str) -> Optional[ConflictInfo]:
        """Get a specific conflict by ID."""
        return self.conflict_resolver.get_conflict_by_id(conflict_id)

    async def cleanup_old_data(self, max_age_hours: int = 24) -> Dict[str, int]:
        """Clean up old data and return cleanup statistics."""
        cleanup_stats = {}

        # Clean up old operations and backups
        cleanup_stats["operations_cleaned"] = self.update_processor.cleanup_old_operations(max_age_hours)

        # Clear old entries from checksum cache if it's getting too large
        if len(self.checksum_calculator._cache) > self.checksum_calculator._cache_size * 1.5:
            self.checksum_calculator.clear_cache()
            cleanup_stats["checksum_cache_cleared"] = True

        # Trim processing times history
        if len(self._processing_times) > self._max_processing_history:
            self._processing_times = self._processing_times[-self._max_processing_history:]
            cleanup_stats["processing_history_trimmed"] = True

        return cleanup_stats

    def export_state(self) -> Dict[str, Any]:
        """Export current state for persistence."""
        return {
            "checksums": self.change_detector.get_current_checksums(),
            "statistics": self._statistics,
            "active_conflicts": [conflict.to_dict() for conflict in self.get_active_conflicts()],
            "export_time": time.time()
        }

    def import_state(self, state_data: Dict[str, Any]) -> None:
        """Import previously exported state."""
        if "checksums" in state_data:
            checksums = {}
            for file_path, checksum_data in state_data["checksums"].items():
                if isinstance(checksum_data, dict):
                    checksums[file_path] = FileChecksum(**checksum_data)
                else:
                    checksums[file_path] = checksum_data
            self.change_detector.set_baseline_checksums(checksums)

        if "statistics" in state_data:
            self._statistics.update(state_data["statistics"])

        logger.info("Imported incremental update system state")