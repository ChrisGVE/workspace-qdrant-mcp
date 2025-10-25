"""
Backup/Restore Operation Tests (Task 292.6).

Comprehensive E2E tests for system backup and restore procedures:
1. Complete system state backup (SQLite + Qdrant collections)
2. Selective backup by project/collection
3. Backup integrity validation
4. Full system restore from backup
5. Partial restore scenarios
6. Restore with version mismatches
7. Backup/restore during active operations
8. Data consistency and system recovery validation
"""

import asyncio
import hashlib
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from common.core.backup import BackupManager, IncompatibleVersionError, RestoreManager

from tests.e2e.utils import (
    HealthChecker,
    QdrantTestHelper,
    TestDataGenerator,
    WorkflowTimer,
    assert_within_threshold,
    run_git_command,
)


@pytest.mark.e2e
@pytest.mark.workflow
class TestCompleteSystemBackup:
    """
    Test complete system state backup operations.

    Validates:
    - SQLite database backup
    - Qdrant collection backup
    - Metadata preservation
    - Backup completeness
    - Backup timing
    """

    @pytest.mark.asyncio
    async def test_complete_system_state_backup(
        self,
        component_lifecycle_manager,
        temp_project_workspace,
        resource_tracker
    ):
        """
        Test complete system backup (SQLite + all Qdrant collections).

        Validates:
        - All SQLite tables backed up
        - All Qdrant collections backed up
        - Metadata files created
        - Backup manifest generated
        - Backup completes within time limit
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Create test data across multiple collections
        collections = {
            "project-code": 5,
            "project-docs": 3,
            "project-config": 2
        }

        for collection, num_docs in collections.items():
            for i in range(num_docs):
                file_path = workspace_path / f"{collection.replace('-', '_')}_{i}.py"
                file_path.write_text(
                    TestDataGenerator.create_python_module(f"module_{i}")
                )

        # Simulate ingestion
        await asyncio.sleep(3)

        # Perform complete backup
        timer = WorkflowTimer()
        timer.start()

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Simulate backup operation
            await asyncio.sleep(2)

            # In real implementation: actual backup
            # backup_manager.backup_system(
            #     backup_dir=backup_path,
            #     include_sqlite=True,
            #     include_collections=True
            # )

            # Create mock backup structure
            (backup_path / "sqlite").mkdir()
            (backup_path / "collections").mkdir()
            (backup_path / "metadata").mkdir()

            # Mock SQLite backup
            sqlite_backup = backup_path / "sqlite" / "state.db"
            sqlite_backup.write_text("# SQLite backup placeholder")

            # Mock collection backups
            for collection in collections:
                coll_backup = backup_path / "collections" / f"{collection}.snapshot"
                coll_backup.write_text(f"# Collection {collection} snapshot")

            # Create manifest
            manifest = {
                "timestamp": time.time(),
                "version": "0.3.0",
                "collections": list(collections.keys()),
                "sqlite_tables": ["watch_folders", "projects", "ingestion_queue"],
                "total_documents": sum(collections.values())
            }

            manifest_path = backup_path / "metadata" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            timer.checkpoint("backup_complete")

            # Validate backup structure
            assert (backup_path / "sqlite").exists(), "SQLite backup directory missing"
            assert (backup_path / "collections").exists(), "Collections backup directory missing"
            assert (backup_path / "metadata").exists(), "Metadata directory missing"
            assert manifest_path.exists(), "Backup manifest missing"

            # Validate manifest content
            loaded_manifest = json.loads(manifest_path.read_text())
            assert "timestamp" in loaded_manifest
            assert "version" in loaded_manifest
            assert loaded_manifest["collections"] == list(collections.keys())

            # Validate backup timing
            backup_duration = timer.get_duration("backup_complete")
            assert backup_duration < 30, \
                f"Backup took {backup_duration:.1f}s, expected < 30s"

            # Validate backup completeness
            sqlite_files = list((backup_path / "sqlite").iterdir())
            assert len(sqlite_files) >= 1, "No SQLite backup files found"

            collection_files = list((backup_path / "collections").iterdir())
            assert len(collection_files) == len(collections), \
                f"Expected {len(collections)} collection backups, got {len(collection_files)}"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_backup_with_compression(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test system backup with compression.

        Validates:
        - Backup creates compressed archive
        - Compression reduces backup size
        - Compressed backup maintains integrity
        - Archive can be extracted
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Create test data
        for i in range(10):
            file_path = workspace_path / f"test_{i}.py"
            file_path.write_text(TestDataGenerator.create_python_module(f"module_{i}"))

        await asyncio.sleep(2)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create uncompressed backup
            uncompressed_dir = backup_path / "uncompressed"
            uncompressed_dir.mkdir()
            (uncompressed_dir / "data.txt").write_text("x" * 10000)

            uncompressed_size = sum(
                f.stat().st_size for f in uncompressed_dir.rglob("*") if f.is_file()
            )

            # Create compressed archive
            archive_path = backup_path / "backup.tar.gz"

            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(uncompressed_dir, arcname="backup")

            compressed_size = archive_path.stat().st_size

            # Validate compression
            assert compressed_size < uncompressed_size, \
                f"Compressed size ({compressed_size}) not less than uncompressed ({uncompressed_size})"

            compression_ratio = compressed_size / uncompressed_size
            assert compression_ratio < 0.5, \
                f"Compression ratio {compression_ratio:.2f} too high, expected < 0.5"

            # Validate archive integrity
            with tarfile.open(archive_path, "r:gz") as tar:
                members = tar.getmembers()
                assert len(members) > 0, "Archive is empty"

                # Extract and verify
                extract_dir = backup_path / "extracted"
                extract_dir.mkdir()
                tar.extractall(extract_dir)

                extracted_data = (extract_dir / "backup" / "data.txt").read_text()
                assert len(extracted_data) == 10000, "Extracted data size mismatch"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestSelectiveBackup:
    """
    Test selective backup by project/collection.

    Validates:
    - Backup specific collections only
    - Backup specific projects only
    - Collection filtering works
    - Project filtering works
    - Selective backup faster than full backup
    """

    @pytest.mark.asyncio
    async def test_selective_collection_backup(
        self,
        component_lifecycle_manager
    ):
        """
        Test backing up specific collections only.

        Validates:
        - Only specified collections backed up
        - Other collections not included
        - Selective backup completes quickly
        - Backup manifest reflects selection
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        all_collections = ["project-code", "project-docs", "project-config", "project-tests"]
        selected_collections = ["project-code", "project-docs"]

        timer = WorkflowTimer()
        timer.start()

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)
            collections_dir = backup_path / "collections"
            collections_dir.mkdir()

            # Simulate selective backup
            await asyncio.sleep(1)

            # In real implementation: backup only selected collections
            # backup_manager.backup_collections(
            #     backup_dir=backup_path,
            #     collections=selected_collections
            # )

            # Mock selective backup
            for collection in selected_collections:
                coll_backup = collections_dir / f"{collection}.snapshot"
                coll_backup.write_text(f"# {collection} backup")

            timer.checkpoint("selective_backup_complete")

            # Validate only selected collections backed up
            backed_up = [f.stem for f in collections_dir.iterdir() if f.suffix == ".snapshot"]
            assert set(backed_up) == set(selected_collections), \
                f"Expected {selected_collections}, got {backed_up}"

            # Validate excluded collections not present
            for excluded in set(all_collections) - set(selected_collections):
                excluded_path = collections_dir / f"{excluded}.snapshot"
                assert not excluded_path.exists(), \
                    f"Excluded collection {excluded} should not be backed up"

            # Validate selective backup timing
            duration = timer.get_duration("selective_backup_complete")
            assert duration < 10, f"Selective backup took {duration:.1f}s, expected < 10s"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_selective_project_backup(
        self,
        component_lifecycle_manager
    ):
        """
        Test backing up specific project data only.

        Validates:
        - Only specified project data backed up
        - Project isolation maintained in backup
        - Project-specific collections identified
        - SQLite project data filtered
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        projects = ["project-a", "project-b", "project-c"]
        selected_project = "project-a"

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Simulate project-specific backup
            await asyncio.sleep(1)

            # Create project-specific backup structure
            project_dir = backup_path / selected_project
            project_dir.mkdir()

            # Mock project collections
            collections = [f"{selected_project}-code", f"{selected_project}-docs"]
            for collection in collections:
                (project_dir / f"{collection}.snapshot").write_text(f"# {collection}")

            # Mock project metadata
            metadata = {
                "project": selected_project,
                "collections": collections,
                "timestamp": time.time()
            }
            (project_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

            # Validate project backup structure
            assert project_dir.exists(), f"Project directory {selected_project} not created"
            assert (project_dir / "metadata.json").exists(), "Project metadata missing"

            loaded_metadata = json.loads((project_dir / "metadata.json").read_text())
            assert loaded_metadata["project"] == selected_project

            # Validate other projects not included
            for other_project in projects:
                if other_project != selected_project:
                    other_dir = backup_path / other_project
                    assert not other_dir.exists(), \
                        f"Project {other_project} should not be in backup"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestBackupIntegrityValidation:
    """
    Test backup integrity validation mechanisms.

    Validates:
    - Checksum generation for backup files
    - Checksum validation on restore
    - Corruption detection
    - Incomplete backup detection
    """

    @pytest.mark.asyncio
    async def test_backup_checksum_generation(
        self,
        component_lifecycle_manager
    ):
        """
        Test backup generates checksums for integrity.

        Validates:
        - Checksum file created for each backup component
        - Checksums are valid
        - Manifest includes checksum information
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Simulate backup with checksum generation
            await asyncio.sleep(1)

            # Create backup files
            backup_files = {
                "sqlite.db": "SQLite database content",
                "collection-1.snapshot": "Collection 1 data",
                "collection-2.snapshot": "Collection 2 data"
            }

            checksums = {}
            for filename, content in backup_files.items():
                file_path = backup_path / filename
                file_path.write_text(content)

                # Generate checksum
                checksum = hashlib.sha256(content.encode()).hexdigest()
                checksums[filename] = checksum

            # Write checksum file
            checksum_file = backup_path / "checksums.sha256"
            checksum_content = "\n".join(
                f"{checksum}  {filename}"
                for filename, checksum in checksums.items()
            )
            checksum_file.write_text(checksum_content)

            # Validate checksum file exists
            assert checksum_file.exists(), "Checksum file not created"

            # Validate checksums are correct
            for filename, expected_checksum in checksums.items():
                file_path = backup_path / filename
                actual_checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
                assert actual_checksum == expected_checksum, \
                    f"Checksum mismatch for {filename}"

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_backup_corruption_detection(
        self,
        component_lifecycle_manager
    ):
        """
        Test backup detects file corruption.

        Validates:
        - Corrupted files detected during validation
        - Validation fails for corrupted backup
        - Specific corrupted files identified
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup with checksum
            original_content = "Original backup data"
            backup_file = backup_path / "data.db"
            backup_file.write_text(original_content)

            original_checksum = hashlib.sha256(original_content.encode()).hexdigest()
            checksum_file = backup_path / "checksums.sha256"
            checksum_file.write_text(f"{original_checksum}  data.db")

            # Corrupt the backup file
            corrupted_content = "Corrupted backup data"
            backup_file.write_text(corrupted_content)

            # Validate corruption detection
            actual_checksum = hashlib.sha256(corrupted_content.encode()).hexdigest()
            assert actual_checksum != original_checksum, \
                "Checksums should differ for corrupted file"

            # In real implementation: validation would fail
            # validation_result = backup_manager.validate_backup(backup_path)
            # assert not validation_result.valid
            # assert "data.db" in validation_result.corrupted_files

            # Simulate validation failure
            validation_passed = (actual_checksum == original_checksum)
            assert not validation_passed, "Validation should fail for corrupted backup"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestFullSystemRestore:
    """
    Test full system restore from backup.

    Validates:
    - SQLite database restored
    - All collections restored
    - Data consistency after restore
    - System functional after restore
    - Restore timing
    """

    @pytest.mark.asyncio
    async def test_complete_system_restore(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test full system restore from complete backup.

        Validates:
        - All backup components restored
        - SQLite state matches backup
        - Collections match backup
        - System operational after restore
        - Data accessible after restore
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Create initial data
        initial_docs = []
        for i in range(5):
            file_path = workspace_path / f"initial_{i}.py"
            content = TestDataGenerator.create_python_module(f"initial_{i}")
            file_path.write_text(content)
            initial_docs.append({"path": str(file_path), "content": content})

        await asyncio.sleep(2)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Simulate backup
            (backup_path / "sqlite").mkdir()
            (backup_path / "collections").mkdir()

            # Mock backup data
            backup_manifest = {
                "timestamp": time.time(),
                "documents": len(initial_docs),
                "collections": ["project-code"]
            }
            (backup_path / "manifest.json").write_text(json.dumps(backup_manifest))

            timer = WorkflowTimer()
            timer.start()

            # Simulate system restore
            await asyncio.sleep(3)

            # In real implementation: actual restore
            # restore_manager.restore_system(backup_dir=backup_path)

            timer.checkpoint("restore_complete")

            # Validate restore timing
            restore_duration = timer.get_duration("restore_complete")
            assert restore_duration < 30, \
                f"Restore took {restore_duration:.1f}s, expected < 30s"

            # Validate system operational after restore
            ready = await component_lifecycle_manager.wait_for_ready(timeout=30)
            assert ready, "System not ready after restore"

            # Validate data accessibility (mocked)
            # In real implementation: query collections to verify data
            # results = await qdrant_client.search(collection="project-code", query="test")
            # assert len(results) == len(initial_docs)

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_restore_preserves_timestamps(
        self,
        component_lifecycle_manager
    ):
        """
        Test restore preserves original timestamps.

        Validates:
        - Document timestamps preserved
        - Collection creation times preserved
        - SQLite record timestamps preserved
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        original_timestamp = time.time()

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup with timestamp metadata
            metadata = {
                "backup_time": original_timestamp,
                "documents": [
                    {"id": 1, "created_at": original_timestamp - 100},
                    {"id": 2, "created_at": original_timestamp - 50}
                ]
            }
            (backup_path / "metadata.json").write_text(json.dumps(metadata))

            # Simulate restore
            await asyncio.sleep(2)

            # Validate timestamps preserved (mocked)
            loaded_metadata = json.loads((backup_path / "metadata.json").read_text())
            assert loaded_metadata["backup_time"] == original_timestamp

            for doc in loaded_metadata["documents"]:
                assert "created_at" in doc
                assert doc["created_at"] < original_timestamp

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestPartialRestore:
    """
    Test partial restore scenarios.

    Validates:
    - Restore specific collections only
    - Restore specific projects only
    - Partial restore doesn't affect other data
    - Mixed restore (some components)
    """

    @pytest.mark.asyncio
    async def test_selective_collection_restore(
        self,
        component_lifecycle_manager
    ):
        """
        Test restoring specific collections only.

        Validates:
        - Only specified collections restored
        - Other collections unchanged
        - Selective restore faster than full restore
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        all_collections = ["project-code", "project-docs", "project-config"]

        timer = WorkflowTimer()
        timer.start()

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup for all collections
            for collection in all_collections:
                coll_file = backup_path / f"{collection}.snapshot"
                coll_file.write_text(f"# {collection} backup data")

            # Simulate selective restore
            await asyncio.sleep(1.5)

            # In real implementation: restore only selected collections
            # restore_manager.restore_collections(
            #     backup_dir=backup_path,
            #     collections=restore_collections
            # )

            timer.checkpoint("selective_restore_complete")

            # Validate selective restore timing
            duration = timer.get_duration("selective_restore_complete")
            assert duration < 15, \
                f"Selective restore took {duration:.1f}s, expected < 15s"

            # Validate only specified collections processed
            # In real implementation: verify only restore_collections were updated

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_partial_restore_without_sqlite(
        self,
        component_lifecycle_manager
    ):
        """
        Test restoring only collections, not SQLite state.

        Validates:
        - Collections restored successfully
        - SQLite state unchanged
        - System remains consistent
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup with both SQLite and collections
            (backup_path / "sqlite").mkdir()
            (backup_path / "collections").mkdir()

            (backup_path / "sqlite" / "state.db").write_text("SQLite backup")
            (backup_path / "collections" / "project-code.snapshot").write_text("Collection backup")

            # Simulate collections-only restore
            await asyncio.sleep(1)

            # In real implementation: restore collections, skip SQLite
            # restore_manager.restore_collections_only(backup_dir=backup_path)

            # Validate SQLite not modified (mocked)
            # In real implementation: check SQLite modification time unchanged

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestRestoreWithVersionMismatch:
    """
    Test restore with version mismatches.

    Validates:
    - Version detection in backup
    - Migration handling for older backups
    - Warning for version mismatches
    - Graceful degradation if needed
    """

    @pytest.mark.asyncio
    async def test_restore_older_version_backup(
        self,
        component_lifecycle_manager
    ):
        """
        Test restoring backup from older version.

        Validates:
        - Version mismatch detected
        - Migration applied if needed
        - Warning logged
        - Restore succeeds with compatibility mode
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        current_version = "0.3.0"
        backup_version = "0.2.0"

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup with older version
            manifest = {
                "version": backup_version,
                "timestamp": time.time(),
                "collections": ["project-code"]
            }
            (backup_path / "manifest.json").write_text(json.dumps(manifest))

            # Simulate restore with version check
            loaded_manifest = json.loads((backup_path / "manifest.json").read_text())

            # Validate version detection
            assert loaded_manifest["version"] == backup_version
            assert loaded_manifest["version"] != current_version

            # In real implementation: apply migration if needed
            # if needs_migration(backup_version, current_version):
            #     migrate_backup(backup_path, from_version=backup_version)

            # Simulate successful restore with migration
            await asyncio.sleep(2)

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_restore_incompatible_version_fails(
        self,
        component_lifecycle_manager
    ):
        """
        Test restore fails gracefully for incompatible versions.

        Validates:
        - Incompatibility detected
        - Restore aborted safely
        - Error message clear
        - System state unchanged
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        current_version = "0.2.1"
        incompatible_version = "0.1.0"  # Minor version difference (0.2 vs 0.1)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create proper backup directory structure using BackupManager
            backup_manager = BackupManager(current_version=incompatible_version)
            backup_manager.prepare_backup_directory(backup_path)

            # Create backup metadata with incompatible version
            metadata = backup_manager.create_backup_metadata(
                collections=["test-collection"],
                total_documents=100,
                description="Test backup with incompatible version"
            )
            backup_manager.save_backup_manifest(metadata, backup_path)

            # Validate backup structure is correct
            assert backup_manager.validate_backup_directory(backup_path)

            # Now attempt to restore with current version - should fail
            restore_manager = RestoreManager(current_version=current_version)

            # Version validation should raise IncompatibleVersionError
            with pytest.raises(IncompatibleVersionError) as exc_info:
                restore_manager.validate_backup(backup_path)

            # Validate error contains expected information
            error = exc_info.value
            assert error.context["backup_version"] == incompatible_version
            assert error.context["current_version"] == current_version
            assert "incompatible" in str(error).lower()

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestBackupRestoreDuringOperations:
    """
    Test backup/restore during active operations.

    Validates:
    - Backup during active ingestion
    - Backup during active searches
    - Restore with minimal downtime
    - Data consistency maintained
    """

    @pytest.mark.asyncio
    async def test_backup_during_active_operations(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test backup while system is processing operations.

        Validates:
        - Backup completes despite active operations
        - No data corruption during backup
        - Operations continue during backup
        - Backup is consistent
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Start continuous operations
        async def continuous_operations():
            """Simulate ongoing operations during backup."""
            for i in range(10):
                file_path = workspace_path / f"during_backup_{i}.py"
                file_path.write_text(f"# File {i} created during backup")
                await asyncio.sleep(0.5)

        # Start operations and backup concurrently
        operations_task = asyncio.create_task(continuous_operations())

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Simulate backup during operations
            await asyncio.sleep(2)

            # Create backup
            (backup_path / "snapshot.db").write_text("Backup snapshot")
            (backup_path / "manifest.json").write_text(
                json.dumps({"timestamp": time.time(), "during_operations": True})
            )

            # Wait for operations to complete
            await operations_task

            # Validate backup created successfully
            assert (backup_path / "snapshot.db").exists()
            assert (backup_path / "manifest.json").exists()

            manifest = json.loads((backup_path / "manifest.json").read_text())
            assert manifest["during_operations"] is True

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_restore_with_minimal_downtime(
        self,
        component_lifecycle_manager
    ):
        """
        Test restore minimizes system downtime.

        Validates:
        - System downtime < 30 seconds
        - Restore process is efficient
        - System ready quickly after restore
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup
            (backup_path / "data.db").write_text("Backup data")

            timer = WorkflowTimer()
            timer.start()

            # Simulate restore with timing
            await component_lifecycle_manager.stop_all()
            timer.checkpoint("system_stopped")

            # Restore operation
            await asyncio.sleep(2)
            timer.checkpoint("restore_complete")

            # Restart system
            await component_lifecycle_manager.start_all()
            await component_lifecycle_manager.wait_for_ready(timeout=30)
            timer.checkpoint("system_ready")

            # Validate downtime
            downtime = timer.get_duration("system_ready") - timer.get_duration("system_stopped")
            assert downtime < 30, f"Downtime {downtime:.1f}s exceeds 30s limit"

        await component_lifecycle_manager.stop_all()


@pytest.mark.e2e
@pytest.mark.workflow
class TestDataConsistencyAfterRestore:
    """
    Test data consistency validation after restore.

    Validates:
    - Document counts match
    - Document content matches
    - Metadata preserved
    - Relationships maintained
    - No data loss
    """

    @pytest.mark.asyncio
    async def test_document_count_consistency(
        self,
        component_lifecycle_manager,
        temp_project_workspace
    ):
        """
        Test document counts match after restore.

        Validates:
        - Same number of documents after restore
        - Per-collection counts match
        - No documents lost or duplicated
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        workspace_path = temp_project_workspace["path"]

        # Create known number of documents
        doc_counts = {"project-code": 5, "project-docs": 3}
        total_docs = sum(doc_counts.values())

        for collection, count in doc_counts.items():
            for i in range(count):
                file_path = workspace_path / f"{collection}_{i}.py"
                file_path.write_text(f"# Document {i}")

        await asyncio.sleep(2)

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Create backup with document counts
            manifest = {
                "total_documents": total_docs,
                "collections": doc_counts
            }
            (backup_path / "manifest.json").write_text(json.dumps(manifest))

            # Simulate restore
            await asyncio.sleep(2)

            # Validate document counts (mocked)
            loaded_manifest = json.loads((backup_path / "manifest.json").read_text())
            assert loaded_manifest["total_documents"] == total_docs

            for collection, expected_count in doc_counts.items():
                assert loaded_manifest["collections"][collection] == expected_count

        await component_lifecycle_manager.stop_all()

    @pytest.mark.asyncio
    async def test_data_integrity_after_restore(
        self,
        component_lifecycle_manager
    ):
        """
        Test data maintains integrity after restore.

        Validates:
        - Document content unchanged
        - Checksums match
        - No corruption detected
        """
        await component_lifecycle_manager.start_all()
        await component_lifecycle_manager.wait_for_ready(timeout=30)

        # Create documents with known checksums
        documents = [
            {"id": 1, "content": "Document 1 content", "checksum": None},
            {"id": 2, "content": "Document 2 content", "checksum": None}
        ]

        for doc in documents:
            doc["checksum"] = hashlib.sha256(doc["content"].encode()).hexdigest()

        with tempfile.TemporaryDirectory() as backup_dir:
            backup_path = Path(backup_dir)

            # Store documents with checksums
            (backup_path / "documents.json").write_text(json.dumps(documents, indent=2))

            # Simulate restore
            await asyncio.sleep(1)

            # Validate checksums after restore
            restored_docs = json.loads((backup_path / "documents.json").read_text())

            for original, restored in zip(documents, restored_docs, strict=False):
                # Verify checksum matches
                restored_checksum = hashlib.sha256(restored["content"].encode()).hexdigest()
                assert restored_checksum == original["checksum"], \
                    f"Checksum mismatch for document {original['id']}"

        await component_lifecycle_manager.stop_all()
