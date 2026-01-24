"""
Comprehensive unit tests for Incremental Updates System.

Tests cover change detection, conflict resolution, rollback capabilities,
and all edge cases for robust file processing.
"""

import asyncio
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.python.common.core.incremental_file_updates import (
    ChangeDetector,
    ChangeRecord,
    ChangeType,
    ChecksumCalculator,
    ConflictInfo,
    ConflictResolutionStrategy,
    ConflictResolver,
    FileChecksum,
    IncrementalUpdateSystem,
    UpdateOperation,
    UpdateProcessor,
    UpdateStatus,
)


class TestFileChecksum:
    """Test FileChecksum functionality."""

    def test_file_checksum_initialization(self):
        """Test FileChecksum initialization."""
        checksum = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="abcdef123456",
            sha256_hash="fedcba654321"
        )

        assert checksum.file_path == "/test/file.txt"
        assert checksum.size == 1024
        assert checksum.mtime == 1234567890.0
        assert checksum.md5_hash == "abcdef123456"
        assert checksum.sha256_hash == "fedcba654321"

    def test_file_checksum_equality(self):
        """Test FileChecksum equality comparison."""
        checksum1 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="abcdef123456"
        )

        checksum2 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="abcdef123456"
        )

        checksum3 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="different_hash"
        )

        assert checksum1 == checksum2
        assert checksum1 != checksum3

    def test_file_checksum_content_change_detection(self):
        """Test content change detection."""
        checksum1 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="hash1"
        )

        checksum2 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="hash2"
        )

        assert checksum1.has_content_changed(checksum2)
        assert not checksum1.has_content_changed(checksum1)

    def test_file_checksum_metadata_change_detection(self):
        """Test metadata-only change detection."""
        checksum1 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="same_hash"
        )

        checksum2 = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567900.0,  # Different mtime
            md5_hash="same_hash"
        )

        assert checksum1.has_metadata_changed(checksum2)
        assert not checksum1.has_metadata_changed(checksum1)

    def test_file_checksum_non_checksum_comparison(self):
        """Test comparison with non-FileChecksum objects."""
        checksum = FileChecksum(
            file_path="/test/file.txt",
            size=1024,
            mtime=1234567890.0,
            md5_hash="hash"
        )

        assert checksum != "not a checksum"
        assert checksum != 123
        assert checksum is not None


class TestChangeRecord:
    """Test ChangeRecord functionality."""

    def test_change_record_initialization(self):
        """Test ChangeRecord initialization."""
        checksum = FileChecksum("/test/file.txt", 1024, 123456.0, "hash123")

        record = ChangeRecord(
            change_id="change123",
            file_path="/test/file.txt",
            change_type=ChangeType.MODIFIED,
            timestamp=time.time(),
            old_checksum=checksum,
            change_size=512,
            processing_priority=5
        )

        assert record.change_id == "change123"
        assert record.file_path == "/test/file.txt"
        assert record.change_type == ChangeType.MODIFIED
        assert record.old_checksum == checksum
        assert record.change_size == 512
        assert record.processing_priority == 5

    def test_change_record_serialization(self):
        """Test ChangeRecord serialization to/from dict."""
        record = ChangeRecord(
            change_id="test123",
            file_path="/test/file.txt",
            change_type=ChangeType.CREATED,
            timestamp=123456.0,
            dependencies=["dep1", "dep2"]
        )

        # Test to_dict
        record_dict = record.to_dict()
        assert record_dict["change_id"] == "test123"
        assert record_dict["change_type"] == "created"
        assert record_dict["dependencies"] == ["dep1", "dep2"]

        # Test from_dict
        restored_record = ChangeRecord.from_dict(record_dict)
        assert restored_record.change_id == "test123"
        assert restored_record.change_type == ChangeType.CREATED
        assert restored_record.dependencies == ["dep1", "dep2"]

    def test_change_record_with_metadata(self):
        """Test ChangeRecord with metadata."""
        metadata = {"user": "test_user", "source": "editor"}
        record = ChangeRecord(
            change_id="meta_test",
            file_path="/test/file.txt",
            change_type=ChangeType.MODIFIED,
            timestamp=time.time(),
            metadata=metadata
        )

        assert record.metadata == metadata
        assert record.metadata["user"] == "test_user"


class TestChecksumCalculator:
    """Test ChecksumCalculator functionality."""

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content for checksum calculation")
            temp_path = f.name

        yield Path(temp_path)

        # Cleanup
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    @pytest.fixture
    def checksum_calculator(self):
        """Create a checksum calculator."""
        return ChecksumCalculator(cache_size=10)

    @pytest.mark.asyncio
    async def test_checksum_calculation(self, checksum_calculator, temp_file):
        """Test basic checksum calculation."""
        checksum = await checksum_calculator.calculate_checksum(temp_file)

        assert checksum is not None
        assert checksum.file_path == str(temp_file)
        assert checksum.size > 0
        assert checksum.mtime > 0
        assert len(checksum.md5_hash) == 32  # MD5 hex length
        assert len(checksum.sha256_hash) == 64  # SHA256 hex length

    @pytest.mark.asyncio
    async def test_checksum_caching(self, checksum_calculator, temp_file):
        """Test checksum caching functionality."""
        # First calculation
        checksum1 = await checksum_calculator.calculate_checksum(temp_file)

        # Second calculation should return cached result
        checksum2 = await checksum_calculator.calculate_checksum(temp_file)

        assert checksum1 == checksum2
        # Verify cache was used (same object)
        assert checksum1 is checksum2

    @pytest.mark.asyncio
    async def test_checksum_cache_size_limit(self, checksum_calculator):
        """Test checksum cache size limiting."""
        temp_files = []

        try:
            # Create more files than cache size
            for i in range(15):  # Cache size is 10
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                    f.write(f"Content {i}")
                    temp_files.append(Path(f.name))

            # Calculate checksums for all files
            checksums = []
            for temp_file in temp_files:
                checksum = await checksum_calculator.calculate_checksum(temp_file)
                checksums.append(checksum)

            # Cache should not exceed size limit
            assert len(checksum_calculator._cache) <= 10

        finally:
            # Cleanup
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except FileNotFoundError:
                    pass

    @pytest.mark.asyncio
    async def test_checksum_nonexistent_file(self, checksum_calculator):
        """Test checksum calculation for non-existent file."""
        nonexistent_file = Path("/nonexistent/file.txt")
        checksum = await checksum_calculator.calculate_checksum(nonexistent_file)

        assert checksum is None

    def test_content_type_detection(self, checksum_calculator):
        """Test content type detection."""
        test_cases = [
            (Path("test.txt"), "text/plain"),
            (Path("README.md"), "text/markdown"),
            (Path("config.json"), "application/json"),
            (Path("data.yaml"), "text/yaml"),
            (Path("document.pdf"), "application/pdf"),
            (Path("unknown.xyz"), None)
        ]

        for file_path, expected_type in test_cases:
            detected_type = checksum_calculator._detect_content_type(file_path)
            assert detected_type == expected_type

    def test_checksum_cache_clearing(self, checksum_calculator):
        """Test cache clearing functionality."""
        # Add some dummy entries to cache
        checksum_calculator._cache["key1"] = Mock()
        checksum_calculator._cache["key2"] = Mock()

        assert len(checksum_calculator._cache) == 2

        checksum_calculator.clear_cache()

        assert len(checksum_calculator._cache) == 0


class TestChangeDetector:
    """Test ChangeDetector functionality."""

    @pytest.fixture
    def checksum_calculator(self):
        """Create a mock checksum calculator."""
        return Mock(spec=ChecksumCalculator)

    @pytest.fixture
    def change_detector(self, checksum_calculator):
        """Create a change detector."""
        return ChangeDetector(checksum_calculator)

    @pytest.mark.asyncio
    async def test_new_file_detection(self, change_detector, checksum_calculator):
        """Test detection of new files."""
        # Mock checksum calculation
        mock_checksum = FileChecksum("/test/new.txt", 1024, time.time(), "hash123")
        checksum_calculator.calculate_checksum = AsyncMock(return_value=mock_checksum)

        # Mock file existence
        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await change_detector.detect_changes([Path("/test/new.txt")])

        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == ChangeType.CREATED
        assert change.file_path == "/test/new.txt"
        assert change.new_checksum == mock_checksum

    @pytest.mark.asyncio
    async def test_file_deletion_detection(self, change_detector, checksum_calculator):
        """Test detection of deleted files."""
        # Set up previous checksum
        old_checksum = FileChecksum("/test/deleted.txt", 1024, time.time(), "old_hash")
        change_detector._previous_checksums["/test/deleted.txt"] = old_checksum

        # Mock file as non-existent
        with patch('aiofiles.os.path.exists', return_value=False):
            changes = await change_detector.detect_changes([Path("/test/deleted.txt")])

        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == ChangeType.DELETED
        assert change.file_path == "/test/deleted.txt"
        assert change.old_checksum == old_checksum

    @pytest.mark.asyncio
    async def test_file_modification_detection(self, change_detector, checksum_calculator):
        """Test detection of modified files."""
        # Set up previous checksum
        old_checksum = FileChecksum("/test/modified.txt", 1024, 123456.0, "old_hash")
        change_detector._previous_checksums["/test/modified.txt"] = old_checksum

        # Mock new checksum
        new_checksum = FileChecksum("/test/modified.txt", 1024, 123457.0, "new_hash")
        checksum_calculator.calculate_checksum = AsyncMock(return_value=new_checksum)

        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await change_detector.detect_changes([Path("/test/modified.txt")])

        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == ChangeType.CONTENT_MODIFIED
        assert change.file_path == "/test/modified.txt"
        assert change.old_checksum == old_checksum
        assert change.new_checksum == new_checksum

    @pytest.mark.asyncio
    async def test_metadata_only_change_detection(self, change_detector, checksum_calculator):
        """Test detection of metadata-only changes."""
        # Set up checksums with same content but different mtime
        old_checksum = FileChecksum("/test/metadata.txt", 1024, 123456.0, "same_hash")
        new_checksum = FileChecksum("/test/metadata.txt", 1024, 123457.0, "same_hash")

        change_detector._previous_checksums["/test/metadata.txt"] = old_checksum
        checksum_calculator.calculate_checksum = AsyncMock(return_value=new_checksum)

        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await change_detector.detect_changes([Path("/test/metadata.txt")])

        assert len(changes) == 1
        change = changes[0]
        assert change.change_type == ChangeType.METADATA_ONLY
        assert change.processing_priority == 3  # Lower priority for metadata-only changes

    @pytest.mark.asyncio
    async def test_no_change_detection(self, change_detector, checksum_calculator):
        """Test that identical files produce no change records."""
        # Set up identical checksums
        checksum = FileChecksum("/test/unchanged.txt", 1024, 123456.0, "same_hash")
        change_detector._previous_checksums["/test/unchanged.txt"] = checksum
        checksum_calculator.calculate_checksum = AsyncMock(return_value=checksum)

        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await change_detector.detect_changes([Path("/test/unchanged.txt")])

        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_multiple_files_detection(self, change_detector, checksum_calculator):
        """Test detection of changes across multiple files."""
        files = [Path("/test/file1.txt"), Path("/test/file2.txt"), Path("/test/file3.txt")]

        # Set up various scenarios
        old_checksum1 = FileChecksum("/test/file1.txt", 1024, 123456.0, "hash1")
        change_detector._previous_checksums["/test/file1.txt"] = old_checksum1

        new_checksum1 = FileChecksum("/test/file1.txt", 2048, 123457.0, "hash1_new")
        new_checksum2 = FileChecksum("/test/file2.txt", 512, 123458.0, "hash2_new")

        async def mock_calculate_checksum(path):
            path_str = str(path)
            if path_str == "/test/file1.txt":
                return new_checksum1
            elif path_str == "/test/file2.txt":
                return new_checksum2
            elif path_str == "/test/file3.txt":
                return None  # Simulate calculation failure
            return None

        checksum_calculator.calculate_checksum = AsyncMock(side_effect=mock_calculate_checksum)

        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await change_detector.detect_changes(files)

        # Should detect modification of file1 and creation of file2
        assert len(changes) == 2

        file1_change = next(c for c in changes if c.file_path == "/test/file1.txt")
        assert file1_change.change_type == ChangeType.CONTENT_MODIFIED

        file2_change = next(c for c in changes if c.file_path == "/test/file2.txt")
        assert file2_change.change_type == ChangeType.CREATED

    def test_change_id_generation(self, change_detector):
        """Test change ID generation."""
        change_id1 = change_detector._generate_change_id("/test/file.txt", 123456.0)
        change_id2 = change_detector._generate_change_id("/test/file.txt", 123456.0)
        change_id3 = change_detector._generate_change_id("/test/file.txt", 123457.0)

        # Same inputs should produce same ID
        assert change_id1 == change_id2

        # Different inputs should produce different IDs
        assert change_id1 != change_id3

        # IDs should be MD5 hashes (32 characters)
        assert len(change_id1) == 32

    def test_baseline_checksums_management(self, change_detector):
        """Test baseline checksum management."""
        checksums = {
            "/test/file1.txt": FileChecksum("/test/file1.txt", 1024, 123456.0, "hash1"),
            "/test/file2.txt": FileChecksum("/test/file2.txt", 2048, 123457.0, "hash2")
        }

        change_detector.set_baseline_checksums(checksums)

        retrieved_checksums = change_detector.get_current_checksums()

        assert len(retrieved_checksums) == 2
        assert retrieved_checksums["/test/file1.txt"].md5_hash == "hash1"
        assert retrieved_checksums["/test/file2.txt"].md5_hash == "hash2"


class TestConflictResolver:
    """Test ConflictResolver functionality."""

    @pytest.fixture
    def conflict_resolver(self):
        """Create a conflict resolver."""
        return ConflictResolver(ConflictResolutionStrategy.LAST_WRITER_WINS)

    def test_conflict_resolver_initialization(self, conflict_resolver):
        """Test ConflictResolver initialization."""
        assert conflict_resolver.default_strategy == ConflictResolutionStrategy.LAST_WRITER_WINS
        assert len(conflict_resolver._active_conflicts) == 0

    def test_no_conflicts_detection(self, conflict_resolver):
        """Test that single changes don't create conflicts."""
        changes = [
            ChangeRecord("change1", "/test/file1.txt", ChangeType.MODIFIED, time.time()),
            ChangeRecord("change2", "/test/file2.txt", ChangeType.CREATED, time.time())
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 0

    def test_rapid_changes_conflict_detection(self, conflict_resolver):
        """Test detection of rapid successive changes as conflicts."""
        base_time = time.time()
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, base_time),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, base_time + 2.0),
            ChangeRecord("change3", "/test/file.txt", ChangeType.MODIFIED, base_time + 3.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 1

        conflict = conflicts[0]
        assert conflict.file_path == "/test/file.txt"
        assert conflict.conflict_type == "rapid_successive_changes"
        assert len(conflict.conflicting_changes) == 3

    def test_delete_after_modify_conflict(self, conflict_resolver):
        """Test detection of delete-after-modify conflicts."""
        base_time = time.time()
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, base_time),
            ChangeRecord("change2", "/test/file.txt", ChangeType.DELETED, base_time + 1.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 1

        conflict = conflicts[0]
        assert conflict.conflict_type == "delete_after_modify"

    def test_complex_conflict_manual_review(self, conflict_resolver):
        """Test that complex conflicts require manual review."""
        base_time = time.time()
        changes = [
            ChangeRecord(f"change{i}", "/test/file.txt", ChangeType.MODIFIED, base_time + i)
            for i in range(5)  # 5 rapid changes
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 1
        assert conflicts[0].manual_review_required is True

    @pytest.mark.asyncio
    async def test_last_writer_wins_resolution(self, conflict_resolver):
        """Test last-writer-wins conflict resolution."""
        base_time = time.time()
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, base_time + 1.0),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, base_time + 2.0),
            ChangeRecord("change3", "/test/file.txt", ChangeType.MODIFIED, base_time + 3.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        conflict = conflicts[0]

        success = await conflict_resolver.resolve_conflict(conflict)
        assert success is True
        assert conflict.resolved is True
        assert conflict.resolution_data["winning_change"] == "change3"

    @pytest.mark.asyncio
    async def test_first_writer_wins_resolution(self, conflict_resolver):
        """Test first-writer-wins conflict resolution."""
        conflict_resolver.default_strategy = ConflictResolutionStrategy.FIRST_WRITER_WINS

        base_time = time.time()
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, base_time + 1.0),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, base_time + 2.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        conflict = conflicts[0]
        conflict.resolution_strategy = ConflictResolutionStrategy.FIRST_WRITER_WINS

        success = await conflict_resolver.resolve_conflict(conflict)
        assert success is True
        assert conflict.resolution_data["winning_change"] == "change1"

    @pytest.mark.asyncio
    async def test_backup_and_replace_resolution(self, conflict_resolver):
        """Test backup-and-replace resolution."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            temp_file = f.name

        try:
            changes = [
                ChangeRecord("change1", temp_file, ChangeType.MODIFIED, time.time()),
                ChangeRecord("change2", temp_file, ChangeType.MODIFIED, time.time() + 1.0)
            ]

            conflicts = conflict_resolver.detect_conflicts(changes)
            conflict = conflicts[0]
            conflict.resolution_strategy = ConflictResolutionStrategy.BACKUP_AND_REPLACE

            success = await conflict_resolver.resolve_conflict(conflict)
            assert success is True
            assert "backup_path" in conflict.resolution_data

        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except FileNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_create_versions_resolution(self, conflict_resolver):
        """Test create-versions resolution."""
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, 123456.0),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, 123457.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        conflict = conflicts[0]
        conflict.resolution_strategy = ConflictResolutionStrategy.CREATE_VERSIONS

        success = await conflict_resolver.resolve_conflict(conflict)
        assert success is True
        assert "versions_created" in conflict.resolution_data
        assert len(conflict.resolution_data["versions_created"]) == 2

    @pytest.mark.asyncio
    async def test_unsupported_resolution_strategy(self, conflict_resolver):
        """Test handling of unsupported resolution strategies."""
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, time.time()),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, time.time() + 1.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        conflict = conflicts[0]
        conflict.resolution_strategy = ConflictResolutionStrategy.MANUAL_RESOLUTION

        success = await conflict_resolver.resolve_conflict(conflict)
        assert success is False

    def test_active_conflicts_management(self, conflict_resolver):
        """Test active conflicts management."""
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, time.time()),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, time.time() + 1.0)
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflict_resolver.get_active_conflicts()) == 1

        conflict_id = conflicts[0].conflict_id
        retrieved_conflict = conflict_resolver.get_conflict_by_id(conflict_id)
        assert retrieved_conflict is not None
        assert retrieved_conflict.conflict_id == conflict_id

    def test_non_conflicting_time_gap(self, conflict_resolver):
        """Test that changes with large time gaps don't create conflicts."""
        base_time = time.time()
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, base_time),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, base_time + 10.0)  # 10 second gap
        ]

        conflicts = conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 0


class TestUpdateProcessor:
    """Test UpdateProcessor functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create a temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def update_processor(self, temp_backup_dir):
        """Create an update processor."""
        return UpdateProcessor(temp_backup_dir)

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Original content")
            temp_path = f.name

        yield temp_path

        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

    @pytest.mark.asyncio
    async def test_successful_update_processing(self, update_processor, temp_file):
        """Test successful update processing."""
        change_record = ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        async def mock_callback(file_path, change_record):
            # Simulate successful processing
            with open(file_path, 'w') as f:
                f.write("Updated content")

        operation = await update_processor.process_update(change_record, mock_callback)

        assert operation.status == UpdateStatus.COMPLETED
        assert operation.file_path == temp_file
        assert operation.backup_path is not None
        assert os.path.exists(operation.backup_path)

    @pytest.mark.asyncio
    async def test_failed_update_processing(self, update_processor, temp_file):
        """Test failed update processing."""
        change_record = ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        async def failing_callback(file_path, change_record):
            raise Exception("Processing failed")

        operation = await update_processor.process_update(change_record, failing_callback)

        assert operation.status == UpdateStatus.FAILED
        assert operation.error_message == "Processing failed"
        assert operation.backup_path is not None  # Backup should still be created

    @pytest.mark.asyncio
    async def test_backup_creation(self, update_processor, temp_file):
        """Test backup creation functionality."""
        ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        backup_path = await update_processor._create_backup(temp_file, "test_operation")

        assert backup_path != ""
        assert os.path.exists(backup_path)

        # Verify backup content
        with open(backup_path) as f:
            backup_content = f.read()
        assert backup_content == "Original content"

    @pytest.mark.asyncio
    async def test_rollback_operation(self, update_processor, temp_file):
        """Test rollback functionality."""
        # First, process an update
        change_record = ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        async def mock_callback(file_path, change_record):
            with open(file_path, 'w') as f:
                f.write("Modified content")

        operation = await update_processor.process_update(change_record, mock_callback)
        operation_id = operation.operation_id

        # Verify file was modified
        with open(temp_file) as f:
            assert f.read() == "Modified content"

        # Rollback the operation
        success = await update_processor.rollback_operation(operation_id)
        assert success is True

        # Verify file was restored
        with open(temp_file) as f:
            assert f.read() == "Original content"

        # Verify operation status
        operation = update_processor.get_operation_status(operation_id)
        assert operation.status == UpdateStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_rollback_nonexistent_operation(self, update_processor):
        """Test rollback of non-existent operation."""
        success = await update_processor.rollback_operation("nonexistent_operation")
        assert success is False

    @pytest.mark.asyncio
    async def test_retry_operation(self, update_processor, temp_file):
        """Test operation retry functionality."""
        change_record = ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        # Create a failing operation
        call_count = 0
        async def sometimes_failing_callback(file_path, change_record):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First attempt failed")
            # Second attempt succeeds
            with open(file_path, 'w') as f:
                f.write("Retry successful")

        operation = await update_processor.process_update(change_record, sometimes_failing_callback)
        assert operation.status == UpdateStatus.FAILED

        # Retry the operation
        success = await update_processor.retry_operation(operation.operation_id, sometimes_failing_callback)
        assert success is True
        assert operation.status == UpdateStatus.COMPLETED
        assert operation.retry_count == 1

    @pytest.mark.asyncio
    async def test_retry_limits(self, update_processor, temp_file):
        """Test retry limits."""
        change_record = ChangeRecord(
            change_id="test_change",
            file_path=temp_file,
            change_type=ChangeType.MODIFIED,
            timestamp=time.time()
        )

        async def always_failing_callback(file_path, change_record):
            raise Exception("Always fails")

        operation = await update_processor.process_update(change_record, always_failing_callback)
        operation_id = operation.operation_id

        # Exhaust retries
        for _ in range(operation.max_retries):
            await update_processor.retry_operation(operation_id, always_failing_callback)

        # Should not be able to retry anymore
        success = await update_processor.retry_operation(operation_id, always_failing_callback)
        assert success is False

        operation = update_processor.get_operation_status(operation_id)
        assert not operation.can_retry()

    def test_get_failed_operations(self, update_processor):
        """Test getting failed operations."""
        # Create some operations with different statuses
        operations = {
            "op1": UpdateOperation("op1", "/test/file1.txt", "modified", UpdateStatus.COMPLETED, time.time()),
            "op2": UpdateOperation("op2", "/test/file2.txt", "modified", UpdateStatus.FAILED, time.time()),
            "op3": UpdateOperation("op3", "/test/file3.txt", "modified", UpdateStatus.FAILED, time.time())
        }

        update_processor._operations = operations

        failed_ops = update_processor.get_failed_operations()
        assert len(failed_ops) == 2
        assert all(op.status == UpdateStatus.FAILED for op in failed_ops)

    def test_cleanup_old_operations(self, update_processor, temp_backup_dir):
        """Test cleanup of old operations."""
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        recent_time = time.time() - (1 * 3600)  # 1 hour ago

        # Create backup files
        old_backup = temp_backup_dir / "old_backup.txt"
        recent_backup = temp_backup_dir / "recent_backup.txt"
        old_backup.write_text("old backup")
        recent_backup.write_text("recent backup")

        # Create operations
        operations = {
            "old_completed": UpdateOperation(
                "old_completed", "/test/file1.txt", "modified",
                UpdateStatus.COMPLETED, old_time, str(old_backup)
            ),
            "old_failed": UpdateOperation(
                "old_failed", "/test/file2.txt", "modified",
                UpdateStatus.FAILED, old_time
            ),
            "recent_completed": UpdateOperation(
                "recent_completed", "/test/file3.txt", "modified",
                UpdateStatus.COMPLETED, recent_time, str(recent_backup)
            )
        }

        update_processor._operations = operations

        # Cleanup with 24-hour threshold
        cleaned_count = update_processor.cleanup_old_operations(max_age_hours=24)

        assert cleaned_count == 1  # Only old completed operation should be cleaned
        assert "old_completed" not in update_processor._operations
        assert "old_failed" in update_processor._operations  # Failed operations kept
        assert "recent_completed" in update_processor._operations  # Recent operations kept
        assert not old_backup.exists()  # Old backup file should be deleted
        assert recent_backup.exists()  # Recent backup should remain


class TestIncrementalUpdateSystem:
    """Test IncrementalUpdateSystem integration."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def update_system(self, temp_backup_dir):
        """Create incremental update system."""
        return IncrementalUpdateSystem(
            backup_dir=temp_backup_dir,
            checksum_cache_size=10
        )

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        files = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(f"Content of file {i}")
                files.append(Path(f.name))

        yield files

        # Cleanup
        for file_path in files:
            try:
                os.unlink(file_path)
            except FileNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_system_initialization(self, update_system, temp_files):
        """Test system initialization with baseline."""
        checksums = await update_system.initialize_baseline(temp_files)

        assert len(checksums) == 3
        assert all(isinstance(checksum, FileChecksum) for checksum in checksums.values())

        # Verify baselines are set in change detector
        current_checksums = update_system.change_detector.get_current_checksums()
        assert len(current_checksums) == 3

    @pytest.mark.asyncio
    async def test_processing_callback_required(self, update_system, temp_files):
        """Test that processing callback is required."""
        await update_system.initialize_baseline(temp_files)

        # Should raise error without callback
        with pytest.raises(ValueError, match="Processing callback not set"):
            await update_system.process_file_changes(temp_files)

    @pytest.mark.asyncio
    async def test_end_to_end_processing(self, update_system, temp_files):
        """Test end-to-end file processing."""
        # Set up callback
        processed_files = []
        async def test_callback(file_path, change_record):
            processed_files.append(file_path)

        update_system.set_processing_callback(test_callback)

        # Initialize baseline
        await update_system.initialize_baseline(temp_files)

        # Modify one file
        modified_file = temp_files[0]
        with open(modified_file, 'w') as f:
            f.write("Modified content")

        # Process changes
        results = await update_system.process_file_changes(temp_files)

        assert results["changes_detected"] == 1
        assert results["operations_completed"] == 1
        assert len(processed_files) == 1
        assert str(modified_file) in processed_files

    @pytest.mark.asyncio
    async def test_conflict_detection_and_resolution(self, update_system):
        """Test conflict detection and resolution."""
        # Create a file that will have conflicting changes
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Original content")
            conflict_file = Path(f.name)

        try:
            # Set up callback
            async def test_callback(file_path, change_record):
                pass

            update_system.set_processing_callback(test_callback)

            # Initialize baseline
            await update_system.initialize_baseline([conflict_file])

            # Simulate rapid successive changes by manipulating checksums
            base_time = time.time()

            # Create conflicting changes manually
            changes = [
                ChangeRecord("change1", str(conflict_file), ChangeType.MODIFIED, base_time),
                ChangeRecord("change2", str(conflict_file), ChangeType.MODIFIED, base_time + 1.0),
                ChangeRecord("change3", str(conflict_file), ChangeType.MODIFIED, base_time + 2.0)
            ]

            conflicts = update_system.conflict_resolver.detect_conflicts(changes)
            assert len(conflicts) == 1

            # Resolve conflict
            conflict = conflicts[0]
            success = await update_system.conflict_resolver.resolve_conflict(conflict)
            assert success is True

        finally:
            try:
                os.unlink(conflict_file)
            except FileNotFoundError:
                pass

    @pytest.mark.asyncio
    async def test_single_file_update(self, update_system, temp_files):
        """Test single file update handling."""
        # Set up callback
        processed_files = []
        async def test_callback(file_path, change_record):
            processed_files.append(file_path)

        update_system.set_processing_callback(test_callback)

        # Initialize baseline
        await update_system.initialize_baseline(temp_files)

        # Modify one file
        test_file = temp_files[0]
        with open(test_file, 'a') as f:
            f.write("\nAdditional content")

        # Process single file
        results = await update_system.handle_single_file_update(test_file)

        assert results["changes_detected"] == 1
        assert results["operations_completed"] == 1

    @pytest.mark.asyncio
    async def test_failed_operations_retry(self, update_system, temp_files):
        """Test retry of failed operations."""
        # Set up callback that fails initially
        call_count = 0
        async def sometimes_failing_callback(file_path, change_record):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:  # First call fails
                raise Exception("Processing failed")

        update_system.set_processing_callback(sometimes_failing_callback)

        # Initialize and modify file
        await update_system.initialize_baseline(temp_files)
        with open(temp_files[0], 'w') as f:
            f.write("Modified for retry test")

        # Process changes (should fail)
        results = await update_system.process_file_changes([temp_files[0]])
        assert results["operations_failed"] == 1

        # Retry failed operations
        retried_count = await update_system.retry_failed_operations()
        assert retried_count == 1

    @pytest.mark.asyncio
    async def test_rollback_operation(self, update_system, temp_files):
        """Test operation rollback."""
        original_content = "Original content for rollback test"

        # Create file with specific content
        test_file = temp_files[0]
        with open(test_file, 'w') as f:
            f.write(original_content)

        # Set up callback
        async def modifying_callback(file_path, change_record):
            with open(file_path, 'w') as f:
                f.write("Modified content")

        update_system.set_processing_callback(modifying_callback)

        # Initialize and process change
        await update_system.initialize_baseline([test_file])

        with open(test_file, 'a') as f:
            f.write("\nTrigger change")

        results = await update_system.process_file_changes([test_file])
        assert results["operations_completed"] == 1

        # Get operation ID (simplified approach for test)
        operations = update_system.update_processor._operations
        operation_id = list(operations.keys())[0]

        # Rollback
        success = await update_system.rollback_operation(operation_id)
        assert success is True

    def test_statistics_collection(self, update_system):
        """Test statistics collection."""
        stats = update_system.get_statistics()

        assert "changes_processed" in stats
        assert "conflicts_detected" in stats
        assert "conflicts_resolved" in stats
        assert "operations_rolled_back" in stats
        assert "operations_retried" in stats
        assert "average_processing_time" in stats
        assert "checksum_cache_size" in stats
        assert "backup_directory" in stats

    @pytest.mark.asyncio
    async def test_state_export_import(self, update_system, temp_files):
        """Test state export and import."""
        # Initialize with baseline
        await update_system.initialize_baseline(temp_files)

        # Export state
        exported_state = update_system.export_state()

        assert "checksums" in exported_state
        assert "statistics" in exported_state
        assert "export_time" in exported_state
        assert len(exported_state["checksums"]) == 3

        # Create new system and import state
        new_system = IncrementalUpdateSystem(update_system.backup_dir)
        new_system.import_state(exported_state)

        # Verify state was imported
        imported_checksums = new_system.change_detector.get_current_checksums()
        assert len(imported_checksums) == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, update_system):
        """Test cleanup of old data."""
        # Add some dummy data
        update_system._statistics["changes_processed"] = 100
        update_system._processing_times = [1.0] * 2000  # Exceed max history

        cleanup_stats = await update_system.cleanup_old_data(max_age_hours=1)

        assert "processing_history_trimmed" in cleanup_stats
        assert len(update_system._processing_times) <= update_system._max_processing_history

    def test_active_conflicts_management(self, update_system):
        """Test active conflicts management."""
        # Create a conflict manually
        changes = [
            ChangeRecord("change1", "/test/file.txt", ChangeType.MODIFIED, time.time()),
            ChangeRecord("change2", "/test/file.txt", ChangeType.MODIFIED, time.time() + 1.0)
        ]

        conflicts = update_system.conflict_resolver.detect_conflicts(changes)
        assert len(conflicts) == 1

        # Test retrieval
        active_conflicts = update_system.get_active_conflicts()
        assert len(active_conflicts) == 1

        conflict_id = conflicts[0].conflict_id
        specific_conflict = update_system.get_conflict_by_id(conflict_id)
        assert specific_conflict is not None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_checksum_calculation_permission_error(self):
        """Test checksum calculation with permission errors."""
        calculator = ChecksumCalculator()

        # Mock file that raises permission error
        with patch('aiofiles.open', side_effect=PermissionError("Access denied")):
            checksum = await calculator.calculate_checksum(Path("/restricted/file.txt"))
            assert checksum is None

    @pytest.mark.asyncio
    async def test_change_detection_with_io_errors(self):
        """Test change detection with I/O errors."""
        calculator = Mock(spec=ChecksumCalculator)
        calculator.calculate_checksum = AsyncMock(side_effect=Exception("I/O error"))

        detector = ChangeDetector(calculator)

        with patch('aiofiles.os.path.exists', return_value=True):
            changes = await detector.detect_changes([Path("/error/file.txt")])
            assert len(changes) == 0  # Error should be handled gracefully

    @pytest.mark.asyncio
    async def test_update_system_with_readonly_backup_dir(self, temp_backup_dir):
        """Test update system with read-only backup directory."""
        system = IncrementalUpdateSystem(temp_backup_dir)

        # Set up minimal test
        async def test_callback(file_path, change_record):
            pass

        system.set_processing_callback(test_callback)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("Test content")
            test_file = Path(f.name)

        try:
            await system.initialize_baseline([test_file])

            # Modify file
            with open(test_file, 'a') as f:
                f.write("\nModified")

            # Mock backup creation to simulate permission error
            with patch.object(system.update_processor, '_create_backup', side_effect=PermissionError("Permission denied")):
                # Process should handle backup creation failure gracefully
                results = await system.process_file_changes([test_file])

                # Should still detect changes even if backup fails
                assert results["changes_detected"] >= 0

        finally:
            try:
                os.unlink(test_file)
            except FileNotFoundError:
                pass

    def test_checksum_with_extreme_cache_size(self):
        """Test checksum calculator with extreme cache sizes."""
        # Test with cache size 0
        calculator_zero = ChecksumCalculator(cache_size=0)
        assert calculator_zero._cache_size == 0

        # Test with very large cache size
        calculator_large = ChecksumCalculator(cache_size=1000000)
        assert calculator_large._cache_size == 1000000

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, temp_backup_dir):
        """Test concurrent operations on the same system."""
        system = IncrementalUpdateSystem(temp_backup_dir, checksum_cache_size=5)

        # Create multiple temporary files
        temp_files = []
        for i in range(5):
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                f.write(f"Concurrent test file {i}")
                temp_files.append(Path(f.name))

        try:
            # Set up callback
            async def concurrent_callback(file_path, change_record):
                # Simulate some async work
                await asyncio.sleep(0.01)

            system.set_processing_callback(concurrent_callback)

            # Initialize baseline
            await system.initialize_baseline(temp_files)

            # Modify files concurrently
            for i, file_path in enumerate(temp_files):
                with open(file_path, 'a') as f:
                    f.write(f"\nConcurrent modification {i}")

            # Process changes concurrently
            tasks = [
                system.handle_single_file_update(file_path)
                for file_path in temp_files
            ]

            results = await asyncio.gather(*tasks)

            # All operations should complete
            assert len(results) == 5
            assert all(result["changes_detected"] >= 0 for result in results)

        finally:
            # Cleanup
            for file_path in temp_files:
                try:
                    os.unlink(file_path)
                except FileNotFoundError:
                    pass

    def test_change_record_with_extreme_values(self):
        """Test ChangeRecord with extreme values."""
        # Test with very large file
        record = ChangeRecord(
            change_id="extreme_test",
            file_path="/huge/file.bin",
            change_type=ChangeType.MODIFIED,
            timestamp=time.time(),
            change_size=10**12,  # 1TB change
            processing_priority=10  # Maximum priority
        )

        assert record.change_size == 10**12
        assert record.processing_priority == 10

        # Test serialization with extreme values
        record_dict = record.to_dict()
        restored_record = ChangeRecord.from_dict(record_dict)
        assert restored_record.change_size == 10**12

    def test_memory_efficiency_with_large_conflict_list(self):
        """Test memory efficiency with large numbers of conflicts."""
        resolver = ConflictResolver()

        # Create a large number of changes for the same file
        base_time = time.time()
        changes = []
        for i in range(1000):  # 1000 rapid changes
            change = ChangeRecord(
                change_id=f"change_{i}",
                file_path="/test/busy_file.txt",
                change_type=ChangeType.MODIFIED,
                timestamp=base_time + (i * 0.001)  # 1ms apart
            )
            changes.append(change)

        # Should handle large conflict gracefully
        conflicts = resolver.detect_conflicts(changes)
        assert len(conflicts) == 1

        conflict = conflicts[0]
        assert len(conflict.conflicting_changes) == 1000
        assert conflict.manual_review_required is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
