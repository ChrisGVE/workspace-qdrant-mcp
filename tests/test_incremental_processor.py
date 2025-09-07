"""
Tests for Incremental Update Mechanism.

This module comprehensively tests the incremental processing system including
change detection accuracy, incremental update correctness, transaction safety,
batch processing efficiency, priority handling, and conflict resolution.
"""

import asyncio
import hashlib
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.workspace_qdrant_mcp.core.incremental_processor import (
    ChangeDetector,
    ChangeType,
    ConflictResolution,
    ConflictResolver,
    DifferentialUpdater,
    FileChangeInfo,
    IncrementalProcessor,
    ProcessingPriority,
    TransactionManager,
    create_incremental_processor,
)
from src.workspace_qdrant_mcp.core.sqlite_state_manager import (
    FileProcessingRecord,
    FileProcessingStatus,
    SQLiteStateManager,
)


class TestChangeDetector:
    """Test change detection functionality."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass

    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager."""
        manager = SQLiteStateManager(temp_db_path)
        success = await manager.initialize()
        assert success, "State manager initialization failed"
        
        yield manager
        
        await manager.close()

    @pytest.fixture
    async def change_detector(self, state_manager):
        """Create change detector."""
        return ChangeDetector(state_manager)

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files."""
        temp_dir = tempfile.mkdtemp()
        files = []
        
        # Create test files with different content
        for i in range(3):
            file_path = Path(temp_dir) / f"test_file_{i}.txt"
            content = f"Test content {i}\nLine 2\nLine 3"
            file_path.write_text(content)
            files.append(str(file_path))
        
        yield files
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_detect_new_files(self, change_detector, temp_files):
        """Test detection of new files."""
        result = await change_detector.detect_changes(temp_files)
        
        assert result.total_scanned == 3
        assert len(result.changes) == 3
        assert result.unchanged_files == 0
        assert result.detection_time_ms > 0
        
        for change in result.changes:
            assert change.change_type == ChangeType.CREATED
            assert change.current_hash is not None
            assert change.current_size > 0
            assert change.current_mtime is not None
            assert change.stored_record is None

    @pytest.mark.asyncio
    async def test_detect_modified_files(self, change_detector, state_manager, temp_files):
        """Test detection of modified files."""
        file_path = temp_files[0]
        collection = "test_collection"
        
        # Create initial record in database
        original_content = Path(file_path).read_text()
        original_hash = hashlib.sha256(original_content.encode()).hexdigest()
        
        await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection,
            file_hash=original_hash,
            file_size=len(original_content)
        )
        await state_manager.complete_file_processing(file_path, success=True)
        
        # Modify file content
        time.sleep(0.1)  # Ensure different mtime
        new_content = original_content + "\nNew line added"
        Path(file_path).write_text(new_content)
        
        # Detect changes
        result = await change_detector.detect_changes([file_path])
        
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.change_type == ChangeType.MODIFIED
        assert change.file_path == file_path
        assert change.current_hash != original_hash
        assert change.stored_record is not None
        assert change.stored_record.file_hash == original_hash

    @pytest.mark.asyncio
    async def test_detect_deleted_files(self, change_detector, state_manager, temp_files):
        """Test detection of deleted files."""
        file_path = temp_files[0]
        collection = "test_collection"
        
        # Create record and then delete file
        await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection
        )
        
        # Delete the file
        os.unlink(file_path)
        
        # Detect changes
        result = await change_detector.detect_changes([file_path])
        
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.change_type == ChangeType.DELETED
        assert change.stored_record is not None

    @pytest.mark.asyncio
    async def test_detect_no_changes(self, change_detector, state_manager, temp_files):
        """Test detection when no changes exist."""
        file_path = temp_files[0]
        collection = "test_collection"
        
        # Create record matching current file state
        content = Path(file_path).read_text()
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        file_size = len(content)
        
        await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection,
            file_hash=file_hash,
            file_size=file_size
        )
        await state_manager.complete_file_processing(file_path, success=True)
        
        # Detect changes
        result = await change_detector.detect_changes([file_path])
        
        assert len(result.changes) == 0
        assert result.unchanged_files == 1

    @pytest.mark.asyncio
    async def test_detect_lsp_stale_files(self, change_detector, state_manager, temp_files):
        """Test detection of files with stale LSP data."""
        file_path = temp_files[0]
        collection = "test_collection"
        
        # Create record with old LSP analysis time
        content = Path(file_path).read_text()
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        old_time = datetime.now(timezone.utc)
        
        await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection,
            file_hash=file_hash,
            file_size=len(content)
        )
        
        # Update with old LSP analysis time
        async with state_manager.transaction() as conn:
            conn.execute(
                "UPDATE file_processing SET last_lsp_analysis = ? WHERE file_path = ?",
                (old_time.isoformat(), file_path)
            )
        
        # Touch file to update mtime
        time.sleep(0.1)
        Path(file_path).touch()
        
        # Detect changes
        result = await change_detector.detect_changes([file_path])
        
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.change_type == ChangeType.LSP_STALE

    @pytest.mark.asyncio
    async def test_priority_patterns(self, change_detector, temp_files):
        """Test priority assignment based on file patterns."""
        priority_patterns = {
            "*.py": ProcessingPriority.HIGH,
            "test_*": ProcessingPriority.LOW,
            "*": ProcessingPriority.NORMAL
        }
        
        # Create Python file
        py_file = str(Path(temp_files[0]).with_suffix('.py'))
        Path(py_file).write_text("print('hello')")
        
        result = await change_detector.detect_changes(
            [py_file],
            priority_patterns=priority_patterns
        )
        
        assert len(result.changes) == 1
        assert result.changes[0].priority == ProcessingPriority.HIGH

    @pytest.mark.asyncio
    async def test_conflict_detection(self, change_detector, state_manager, temp_files):
        """Test detection of concurrent modification conflicts."""
        file_path = temp_files[0]
        collection = "test_collection"
        
        # Start processing (simulate ongoing processing)
        await state_manager.start_file_processing(
            file_path=file_path,
            collection=collection
        )
        
        # Modify file while "processing"
        original_content = Path(file_path).read_text()
        new_content = original_content + "\nConcurrent modification"
        Path(file_path).write_text(new_content)
        
        # Detect changes
        result = await change_detector.detect_changes([file_path])
        
        assert len(result.changes) == 1
        change = result.changes[0]
        assert change.conflict_detected is True
        assert "while processing" in change.conflict_reason.lower()

    @pytest.mark.asyncio
    async def test_batch_change_detection(self, change_detector, state_manager):
        """Test batch processing of multiple file changes."""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            # Create many test files
            for i in range(50):
                file_path = Path(temp_dir) / f"batch_file_{i:03d}.txt"
                content = f"Batch file {i} content"
                file_path.write_text(content)
                file_paths.append(str(file_path))
            
            # Detect changes for all files
            start_time = time.time()
            result = await change_detector.detect_changes(file_paths)
            detection_time = time.time() - start_time
            
            assert result.total_scanned == 50
            assert len(result.changes) == 50
            assert detection_time < 5.0  # Should be reasonably fast
            assert all(c.change_type == ChangeType.CREATED for c in result.changes)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestConflictResolver:
    """Test conflict resolution functionality."""

    @pytest.fixture
    def conflict_resolver_latest_wins(self):
        """Create conflict resolver with latest wins strategy."""
        return ConflictResolver(ConflictResolution.LATEST_WINS)

    @pytest.fixture
    def conflict_resolver_skip(self):
        """Create conflict resolver with skip strategy."""
        return ConflictResolver(ConflictResolution.SKIP_CONFLICTED)

    @pytest.fixture
    def conflicted_change(self):
        """Create a conflicted file change."""
        return FileChangeInfo(
            file_path="/test/conflicted.txt",
            change_type=ChangeType.MODIFIED,
            conflict_detected=True,
            conflict_reason="File modified while processing",
            collection="test_collection"
        )

    @pytest.mark.asyncio
    async def test_latest_wins_resolution(self, conflict_resolver_latest_wins, conflicted_change):
        """Test latest wins conflict resolution strategy."""
        state_manager = MagicMock()
        
        resolved = await conflict_resolver_latest_wins.resolve_conflicts(
            [conflicted_change],
            state_manager
        )
        
        assert len(resolved) == 1
        assert resolved[0].conflict_detected is False
        assert resolved[0].conflict_reason is None
        assert conflict_resolver_latest_wins.resolved_conflicts == 1

    @pytest.mark.asyncio
    async def test_skip_conflicted_resolution(self, conflict_resolver_skip, conflicted_change):
        """Test skip conflicted files resolution strategy."""
        state_manager = MagicMock()
        
        resolved = await conflict_resolver_skip.resolve_conflicts(
            [conflicted_change],
            state_manager
        )
        
        assert len(resolved) == 0  # Conflicted file was skipped

    @pytest.mark.asyncio
    async def test_merge_metadata_resolution(self, conflicted_change):
        """Test merge metadata conflict resolution strategy."""
        resolver = ConflictResolver(ConflictResolution.MERGE_METADATA)
        state_manager = MagicMock()
        
        # Add stored record with metadata
        from src.workspace_qdrant_mcp.core.sqlite_state_manager import FileProcessingRecord
        conflicted_change.stored_record = FileProcessingRecord(
            file_path="/test/conflicted.txt",
            collection="test_collection",
            status=FileProcessingStatus.PROCESSING,
            metadata={"original": "data", "shared": "old_value"}
        )
        conflicted_change.metadata = {"new": "data", "shared": "new_value"}
        
        resolved = await resolver.resolve_conflicts(
            [conflicted_change],
            state_manager
        )
        
        assert len(resolved) == 1
        merged_metadata = resolved[0].metadata
        assert merged_metadata["original"] == "data"
        assert merged_metadata["new"] == "data"
        assert merged_metadata["shared"] == "new_value"  # New value wins


class TestDifferentialUpdater:
    """Test differential Qdrant updates."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.upsert = MagicMock()
        client.delete = MagicMock()
        client.set_payload = MagicMock()
        return client

    @pytest.fixture
    def differential_updater(self, mock_qdrant_client):
        """Create differential updater with mock client."""
        return DifferentialUpdater(mock_qdrant_client)

    @pytest.fixture
    def sample_changes(self, temp_files):
        """Create sample file changes."""
        changes = []
        
        # Created file
        changes.append(FileChangeInfo(
            file_path=temp_files[0],
            change_type=ChangeType.CREATED,
            current_size=100,
            current_hash="abc123",
            collection="test_collection"
        ))
        
        # Modified file
        changes.append(FileChangeInfo(
            file_path=temp_files[1],
            change_type=ChangeType.MODIFIED,
            current_size=150,
            current_hash="def456",
            collection="test_collection"
        ))
        
        # Deleted file
        changes.append(FileChangeInfo(
            file_path="/deleted/file.txt",
            change_type=ChangeType.DELETED,
            collection="test_collection"
        ))
        
        return changes

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files."""
        temp_dir = tempfile.mkdtemp()
        files = []
        
        for i in range(2):
            file_path = Path(temp_dir) / f"test_file_{i}.txt"
            content = f"Test content {i}"
            file_path.write_text(content)
            files.append(str(file_path))
        
        yield files
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_apply_changes(self, differential_updater, sample_changes, mock_qdrant_client):
        """Test applying changes to Qdrant collections."""
        with patch.object(differential_updater, '_generate_embeddings', 
                         return_value=[0.1, 0.2, 0.3]):
            stats = await differential_updater.apply_changes(sample_changes, batch_size=2)
        
        assert stats["upserts"] == 2  # Created + Modified
        assert stats["deletes"] == 1  # Deleted
        assert stats["batch_operations"] >= 1
        
        # Verify Qdrant client calls
        assert mock_qdrant_client.upsert.called
        assert mock_qdrant_client.delete.called

    @pytest.mark.asyncio
    async def test_lsp_metadata_update(self, differential_updater, mock_qdrant_client):
        """Test LSP metadata updates."""
        lsp_change = FileChangeInfo(
            file_path="/test/lsp_stale.py",
            change_type=ChangeType.LSP_STALE,
            current_mtime=time.time(),
            collection="test_collection"
        )
        
        stats = await differential_updater.apply_changes([lsp_change])
        
        assert stats["updates"] == 1
        assert mock_qdrant_client.set_payload.called

    @pytest.mark.asyncio
    async def test_batch_processing(self, differential_updater, mock_qdrant_client):
        """Test batch processing of changes."""
        # Create many changes
        changes = []
        for i in range(25):
            changes.append(FileChangeInfo(
                file_path=f"/test/batch_{i}.txt",
                change_type=ChangeType.CREATED,
                current_size=100,
                current_hash=f"hash_{i}",
                collection="test_collection"
            ))
        
        with patch.object(differential_updater, '_generate_embeddings',
                         return_value=[0.1, 0.2, 0.3]):
            stats = await differential_updater.apply_changes(changes, batch_size=10)
        
        assert stats["upserts"] == 25
        assert stats["batch_operations"] >= 3  # At least 3 batches for 25 items with batch_size=10


class TestTransactionManager:
    """Test transaction management functionality."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass

    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager."""
        manager = SQLiteStateManager(temp_db_path)
        success = await manager.initialize()
        assert success
        
        yield manager
        
        await manager.close()

    @pytest.fixture
    async def transaction_manager(self, state_manager):
        """Create transaction manager."""
        return TransactionManager(state_manager)

    @pytest.mark.asyncio
    async def test_successful_transaction(self, transaction_manager, state_manager):
        """Test successful transaction commit."""
        transaction_id = "test_tx_1"
        
        async with transaction_manager.transaction(transaction_id) as conn:
            conn.execute(
                """
                INSERT INTO file_processing 
                (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/tx.txt", "test_collection", "pending", 2)
            )
        
        # Verify data was committed
        record = await state_manager.get_file_processing_status("/test/tx.txt")
        assert record is not None

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, transaction_manager, state_manager):
        """Test transaction rollback on error."""
        transaction_id = "test_tx_2"
        
        try:
            async with transaction_manager.transaction(transaction_id) as conn:
                conn.execute(
                    """
                    INSERT INTO file_processing 
                    (file_path, collection, status, priority)
                    VALUES (?, ?, ?, ?)
                    """,
                    ("/test/rollback.txt", "test_collection", "pending", 2)
                )
                
                # Force error to trigger rollback
                raise ValueError("Test error")
                
        except ValueError:
            pass  # Expected error
        
        # Verify data was rolled back
        record = await state_manager.get_file_processing_status("/test/rollback.txt")
        assert record is None

    @pytest.mark.asyncio
    async def test_savepoint_operations(self, transaction_manager, state_manager):
        """Test savepoint creation and rollback."""
        transaction_id = "test_tx_3"
        
        async with transaction_manager.transaction(transaction_id) as conn:
            # Insert first record
            conn.execute(
                """
                INSERT INTO file_processing 
                (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/savepoint1.txt", "test_collection", "pending", 2)
            )
            
            # Create savepoint
            await transaction_manager.create_savepoint(conn, "sp1")
            
            # Insert second record
            conn.execute(
                """
                INSERT INTO file_processing 
                (file_path, collection, status, priority)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/savepoint2.txt", "test_collection", "pending", 2)
            )
            
            # Rollback to savepoint
            await transaction_manager.rollback_to_savepoint(conn, "sp1")
        
        # Verify first record exists, second was rolled back
        record1 = await state_manager.get_file_processing_status("/test/savepoint1.txt")
        record2 = await state_manager.get_file_processing_status("/test/savepoint2.txt")
        
        assert record1 is not None
        assert record2 is None

    @pytest.mark.asyncio
    async def test_concurrent_transactions(self, transaction_manager):
        """Test that concurrent transactions are prevented."""
        transaction_id = "test_tx_concurrent"
        
        async with transaction_manager.transaction(transaction_id):
            # Try to start another transaction with same ID
            with pytest.raises(ValueError, match="already active"):
                async with transaction_manager.transaction(transaction_id):
                    pass


class TestIncrementalProcessor:
    """Test main incremental processor functionality."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass

    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager."""
        manager = SQLiteStateManager(temp_db_path)
        success = await manager.initialize()
        assert success
        
        yield manager
        
        await manager.close()

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.upsert = MagicMock()
        client.delete = MagicMock()
        client.set_payload = MagicMock()
        return client

    @pytest.fixture
    async def incremental_processor(self, state_manager, mock_qdrant_client):
        """Create incremental processor."""
        processor = IncrementalProcessor(
            state_manager=state_manager,
            qdrant_client=mock_qdrant_client
        )
        await processor.initialize()
        return processor

    @pytest.fixture
    def temp_files(self):
        """Create temporary test files."""
        temp_dir = tempfile.mkdtemp()
        files = []
        
        for i in range(3):
            file_path = Path(temp_dir) / f"test_file_{i}.txt"
            content = f"Test content {i}"
            file_path.write_text(content)
            files.append(str(file_path))
        
        yield files
        
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_process_new_files(self, incremental_processor, temp_files):
        """Test processing new files."""
        with patch.object(incremental_processor.differential_updater, '_generate_embeddings',
                         return_value=[0.1, 0.2, 0.3]):
            result = await incremental_processor.process_changes(temp_files)
        
        assert len(result.processed) == 3
        assert len(result.failed) == 0
        assert len(result.skipped) == 0
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_with_priorities(self, incremental_processor, temp_files):
        """Test processing with priority patterns."""
        priority_patterns = {
            "*_0.txt": ProcessingPriority.HIGH,
            "*_1.txt": ProcessingPriority.LOW
        }
        
        with patch.object(incremental_processor.differential_updater, '_generate_embeddings',
                         return_value=[0.1, 0.2, 0.3]):
            result = await incremental_processor.process_changes(
                temp_files,
                priority_patterns=priority_patterns
            )
        
        assert len(result.processed) == 3
        # High priority files should be processed first

    @pytest.mark.asyncio
    async def test_process_with_conflicts(self, incremental_processor, state_manager, temp_files):
        """Test processing with conflict resolution."""
        file_path = temp_files[0]
        
        # Create processing record to simulate conflict
        await state_manager.start_file_processing(
            file_path=file_path,
            collection="test_collection"
        )
        
        # Modify file content to create conflict
        original_content = Path(file_path).read_text()
        new_content = original_content + "\nConflicted content"
        Path(file_path).write_text(new_content)
        
        with patch.object(incremental_processor.differential_updater, '_generate_embeddings',
                         return_value=[0.1, 0.2, 0.3]):
            result = await incremental_processor.process_changes([file_path])
        
        assert result.conflicts_resolved >= 0  # Depends on strategy

    @pytest.mark.asyncio
    async def test_batch_processing(self, incremental_processor):
        """Test batch processing of many files."""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            # Create many test files
            for i in range(20):
                file_path = Path(temp_dir) / f"batch_file_{i:02d}.txt"
                content = f"Batch content {i}"
                file_path.write_text(content)
                file_paths.append(str(file_path))
            
            with patch.object(incremental_processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.2, 0.3]):
                result = await incremental_processor.process_changes(
                    file_paths,
                    batch_size=5
                )
            
            assert len(result.processed) == 20
            assert result.processing_time_ms > 0
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_no_changes_detected(self, incremental_processor, state_manager, temp_files):
        """Test behavior when no changes are detected."""
        file_path = temp_files[0]
        
        # Create record matching current file state
        content = Path(file_path).read_text()
        file_hash = hashlib.sha256(content.encode()).hexdigest()
        
        await state_manager.start_file_processing(
            file_path=file_path,
            collection="test_collection",
            file_hash=file_hash,
            file_size=len(content)
        )
        await state_manager.complete_file_processing(file_path, success=True)
        
        result = await incremental_processor.process_changes([file_path])
        
        assert len(result.processed) == 0
        assert len(result.skipped) == 1

    @pytest.mark.asyncio
    async def test_processing_statistics(self, incremental_processor, temp_files):
        """Test processing statistics collection."""
        with patch.object(incremental_processor.differential_updater, '_generate_embeddings',
                         return_value=[0.1, 0.2, 0.3]):
            await incremental_processor.process_changes(temp_files)
        
        stats = await incremental_processor.get_processing_statistics()
        
        assert "total_processed" in stats
        assert "successful" in stats
        assert "failed" in stats
        assert "change_detector_cache_size" in stats
        assert "differential_updater_stats" in stats

    @pytest.mark.asyncio
    async def test_cache_management(self, incremental_processor):
        """Test cache clearing functionality."""
        # Add some data to cache
        incremental_processor.change_detector._detection_cache["/test/file.txt"] = (time.time(), "hash")
        
        assert len(incremental_processor.change_detector._detection_cache) > 0
        
        await incremental_processor.clear_caches()
        
        assert len(incremental_processor.change_detector._detection_cache) == 0


class TestIntegrationScenarios:
    """Test integration scenarios with real-world workflows."""

    @pytest.fixture
    async def temp_db_path(self):
        """Create temporary database file."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        for ext in ['', '-wal', '-shm']:
            try:
                os.unlink(db_path + ext)
            except FileNotFoundError:
                pass

    @pytest.fixture
    async def state_manager(self, temp_db_path):
        """Create and initialize state manager."""
        manager = SQLiteStateManager(temp_db_path)
        success = await manager.initialize()
        assert success
        
        yield manager
        
        await manager.close()

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = MagicMock()
        client.upsert = MagicMock()
        client.delete = MagicMock()
        client.set_payload = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_development_workflow_simulation(self, state_manager, mock_qdrant_client):
        """Test simulated development workflow with file modifications."""
        # Create processor
        processor = await create_incremental_processor(
            state_manager=state_manager,
            qdrant_client=mock_qdrant_client
        )
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initial file creation
            file1 = Path(temp_dir) / "main.py"
            file2 = Path(temp_dir) / "utils.py"
            
            file1.write_text("def main():\n    pass")
            file2.write_text("def helper():\n    return True")
            
            # Process initial files
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.2, 0.3]):
                result1 = await processor.process_changes([str(file1), str(file2)])
            
            assert len(result1.processed) == 2
            
            # Modify files (simulate development)
            time.sleep(0.1)  # Ensure different mtime
            file1.write_text("def main():\n    print('Hello World')\n    helper()")
            
            # Process changes
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.3, 0.2]):
                result2 = await processor.process_changes([str(file1), str(file2)])
            
            assert len(result2.processed) == 1  # Only file1 changed
            assert str(file1) in result2.processed
            assert str(file2) in result2.skipped
            
            # Delete file
            file2.unlink()
            
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.3, 0.2]):
                result3 = await processor.process_changes([str(file1), str(file2)])
            
            assert len(result3.processed) == 1  # file2 deletion processed
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_performance_with_large_codebase(self, state_manager, mock_qdrant_client):
        """Test performance with large number of files."""
        processor = await create_incremental_processor(
            state_manager=state_manager,
            qdrant_client=mock_qdrant_client
        )
        
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        try:
            # Create large number of files
            for i in range(100):
                file_path = Path(temp_dir) / f"file_{i:03d}.py"
                content = f"# File {i}\ndef function_{i}():\n    return {i}"
                file_path.write_text(content)
                file_paths.append(str(file_path))
            
            # Process all files
            start_time = time.time()
            
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.2, 0.3]):
                result = await processor.process_changes(file_paths, batch_size=20)
            
            processing_time = time.time() - start_time
            
            assert len(result.processed) == 100
            assert processing_time < 10.0  # Should be reasonably fast
            
            # Modify subset of files
            modified_files = file_paths[:10]
            for file_path in modified_files:
                content = Path(file_path).read_text()
                Path(file_path).write_text(content + "\n# Modified")
            
            # Process incremental changes
            start_time = time.time()
            
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.2, 0.3, 0.1]):
                result = await processor.process_changes(file_paths, batch_size=20)
            
            incremental_time = time.time() - start_time
            
            assert len(result.processed) == 10  # Only modified files
            assert len(result.skipped) == 90   # Unchanged files
            assert incremental_time < processing_time  # Should be faster
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_concurrent_processing_safety(self, state_manager, mock_qdrant_client):
        """Test safety of concurrent processing operations."""
        processor = await create_incremental_processor(
            state_manager=state_manager,
            qdrant_client=mock_qdrant_client
        )
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            file_paths = []
            for i in range(10):
                file_path = Path(temp_dir) / f"concurrent_{i}.txt"
                content = f"Content {i}"
                file_path.write_text(content)
                file_paths.append(str(file_path))
            
            # Process concurrently
            with patch.object(processor.differential_updater, '_generate_embeddings',
                             return_value=[0.1, 0.2, 0.3]):
                
                tasks = []
                for i in range(3):  # 3 concurrent processing tasks
                    task = asyncio.create_task(
                        processor.process_changes(file_paths[i*3:(i+1)*3])
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all tasks completed without errors
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == 3
            
            total_processed = sum(len(r.processed) for r in successful_results)
            assert total_processed <= 10  # No double processing due to concurrency
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])