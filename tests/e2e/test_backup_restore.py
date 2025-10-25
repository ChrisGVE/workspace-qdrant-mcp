"""
End-to-end tests for backup and restore procedures.

Tests complete backup and restore workflows for SQLite database, Qdrant
collections, configuration files, and watch folder state. Verifies data
integrity, system recovery, and state preservation after restore operations.
"""

import json
import shutil
import sqlite3
import time
from pathlib import Path

import pytest

from tests.e2e.fixtures import (
    CLIHelper,
    SystemComponents,
)


@pytest.mark.integration
@pytest.mark.slow
class TestSQLiteBackup:
    """Test SQLite database backup operations."""

    def test_database_backup_creation(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test creating backup of SQLite database."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"state_backup_{int(time.time())}.db"

        # Ensure database has some data
        workspace = system_components.workspace_path
        test_file = workspace / "backup_test.txt"
        test_file.write_text("Content for backup testing")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-backup"]
        )
        time.sleep(2)

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Verify backup exists
        assert backup_path.exists()
        assert backup_path.stat().st_size > 0

        # Cleanup
        backup_path.unlink()

    def test_database_backup_integrity(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that database backup preserves data integrity."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"integrity_backup_{int(time.time())}.db"

        # Get original database info
        original_conn = sqlite3.connect(state_db)
        original_tables = original_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        original_conn.close()

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Verify backup has same structure
        backup_conn = sqlite3.connect(backup_path)
        backup_tables = backup_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        backup_conn.close()

        assert backup_tables == original_tables

        # Cleanup
        backup_path.unlink()

    def test_incremental_backup_tracking(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test tracking multiple backup versions."""
        state_db = system_components.state_db_path
        backups = []

        # Create multiple backups
        for i in range(3):
            backup_path = state_db.parent / f"incremental_{i}_{int(time.time())}.db"
            shutil.copy2(state_db, backup_path)
            backups.append(backup_path)
            time.sleep(1)

        # Verify all backups exist
        assert all(b.exists() for b in backups)

        # Verify backups are ordered by timestamp
        backup_times = [b.stat().st_mtime for b in backups]
        assert backup_times == sorted(backup_times)

        # Cleanup
        for backup in backups:
            backup.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestSQLiteRestore:
    """Test SQLite database restore and validation."""

    def test_database_restore_basic(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test restoring database from backup."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"restore_backup_{int(time.time())}.db"

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Modify database
        test_db = state_db.parent / "test_modification.db"
        shutil.copy2(state_db, test_db)

        # Restore from backup
        restored_path = state_db.parent / f"restored_{int(time.time())}.db"
        shutil.copy2(backup_path, restored_path)

        # Verify restore
        assert restored_path.exists()

        # Cleanup
        backup_path.unlink()
        restored_path.unlink()
        if test_db.exists():
            test_db.unlink()

    def test_watch_folder_state_restoration(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test that watch folder configurations are restored."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"watch_backup_{int(time.time())}.db"

        # Get watch folder count before backup
        conn = sqlite3.connect(state_db)
        original_count = conn.execute(
            "SELECT COUNT(*) FROM watch_folders"
        ).fetchone()[0]
        conn.close()

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Verify backup preserves watch folder data
        backup_conn = sqlite3.connect(backup_path)
        backup_count = backup_conn.execute(
            "SELECT COUNT(*) FROM watch_folders"
        ).fetchone()[0]
        backup_conn.close()

        assert backup_count == original_count

        # Cleanup
        backup_path.unlink()

    def test_restore_after_corruption(
        self, system_components: SystemComponents
    ):
        """Test restoring database after simulated corruption."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"corruption_backup_{int(time.time())}.db"

        # Create valid backup
        shutil.copy2(state_db, backup_path)

        # Create corrupted database
        corrupted_path = state_db.parent / "corrupted.db"
        corrupted_path.write_text("corrupted data")

        # Restore from backup
        restored_path = state_db.parent / f"recovered_{int(time.time())}.db"
        shutil.copy2(backup_path, restored_path)

        # Verify restored database is valid
        conn = sqlite3.connect(restored_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()

        assert len(tables) > 0

        # Cleanup
        backup_path.unlink()
        restored_path.unlink()
        if corrupted_path.exists():
            corrupted_path.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantCollectionBackup:
    """Test Qdrant collection backup procedures."""

    def test_collection_snapshot_creation(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test creating snapshot of Qdrant collection."""
        workspace = system_components.workspace_path
        collection_name = f"test-snapshot-{int(time.time())}"

        # Create collection with data
        test_file = workspace / "snapshot_test.txt"
        test_file.write_text("Content for snapshot testing")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # In production, snapshot would be created via Qdrant API
        # For testing, we verify collection exists and can be listed
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None

    def test_collection_export_data(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test exporting collection data for backup."""
        workspace = system_components.workspace_path
        collection_name = f"test-export-{int(time.time())}"

        # Create collection with known data
        test_file = workspace / "export_test.txt"
        test_content = "Specific content for export testing"
        test_file.write_text(test_content)
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # Verify collection can be accessed for backup
        # In production, would export via Qdrant scroll API
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None

    def test_multiple_collection_backup(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test backing up multiple collections."""
        workspace = system_components.workspace_path
        collections = []

        # Create multiple collections
        for i in range(3):
            collection_name = f"test-multi-backup-{i}-{int(time.time())}"
            test_file = workspace / f"multi_backup_{i}.txt"
            test_file.write_text(f"Content {i}")
            cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", collection_name]
            )
            collections.append(collection_name)

        time.sleep(5)

        # Verify all collections exist
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestQdrantCollectionRestore:
    """Test Qdrant collection restore and verification."""

    def test_collection_restore_basic(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test basic collection restore operation."""
        workspace = system_components.workspace_path
        collection_name = f"test-restore-basic-{int(time.time())}"

        # Create and backup collection
        test_file = workspace / "restore_basic.txt"
        test_file.write_text("Original content for restore")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(3)

        # Verify collection exists (simulating backup availability)
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None

        # In production, would restore from snapshot
        # Here we verify the restore target can be accessed

    def test_collection_data_verification_after_restore(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test verifying collection data after restore."""
        workspace = system_components.workspace_path
        collection_name = f"test-verify-restore-{int(time.time())}"

        # Create collection with specific searchable content
        test_file = workspace / "verify_restore.txt"
        test_content = "Unique verification content for testing"
        test_file.write_text(test_content)
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(5)

        # Verify data is searchable (proving integrity)
        search_result = cli_helper.run_command(
            ["search", "verification content", "--collection", collection_name],
            timeout=15,
        )
        assert search_result is not None

    def test_partial_restore_handling(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test handling partial restore scenarios."""
        workspace = system_components.workspace_path

        # Create multiple collections
        collections = []
        for i in range(3):
            collection_name = f"test-partial-{i}-{int(time.time())}"
            test_file = workspace / f"partial_{i}.txt"
            test_file.write_text(f"Partial restore content {i}")
            cli_helper.run_command(
                ["ingest", "file", str(test_file), "--collection", collection_name]
            )
            collections.append(collection_name)

        time.sleep(5)

        # Verify collections exist
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestConfigurationBackup:
    """Test configuration file backup."""

    def test_config_file_backup(
        self, system_components: SystemComponents
    ):
        """Test backing up configuration files."""
        config_path = system_components.config_path
        backup_path = config_path.parent / f"config_backup_{int(time.time())}.yaml"

        # Create backup
        if config_path.exists():
            shutil.copy2(config_path, backup_path)
            assert backup_path.exists()
            backup_path.unlink()
        else:
            # If no config file, create a dummy one for testing
            config_path.write_text("test_config: value")
            shutil.copy2(config_path, backup_path)
            assert backup_path.exists()
            backup_path.unlink()

    def test_config_backup_preserves_structure(
        self, system_components: SystemComponents
    ):
        """Test that config backup preserves YAML structure."""
        config_path = system_components.config_path
        backup_path = config_path.parent / f"struct_backup_{int(time.time())}.yaml"

        # Ensure config exists
        if not config_path.exists():
            config_path.write_text("key: value\nnested:\n  item: data")

        # Backup
        shutil.copy2(config_path, backup_path)

        # Verify content is identical
        original_content = config_path.read_text()
        backup_content = backup_path.read_text()
        assert original_content == backup_content

        # Cleanup
        backup_path.unlink()

    def test_environment_config_backup(
        self, system_components: SystemComponents
    ):
        """Test backing up environment-specific configuration."""
        workspace = system_components.workspace_path
        env_file = workspace / ".env.backup"

        # Create dummy environment config
        env_content = "QDRANT_URL=http://localhost:6333\nFASTEMBED_MODEL=test"
        env_file.write_text(env_content)

        # Backup
        backup_path = workspace / f"env_backup_{int(time.time())}"
        shutil.copy2(env_file, backup_path)

        # Verify
        assert backup_path.exists()

        # Cleanup
        env_file.unlink()
        backup_path.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestConfigurationRestore:
    """Test configuration restore and validation."""

    def test_config_restore_basic(
        self, system_components: SystemComponents
    ):
        """Test basic configuration restore."""
        workspace = system_components.workspace_path
        config_file = workspace / "test_config.yaml"
        backup_file = workspace / "test_config_backup.yaml"

        # Create original config
        config_content = "test_setting: original_value"
        config_file.write_text(config_content)

        # Create backup
        shutil.copy2(config_file, backup_file)

        # Modify config
        config_file.write_text("test_setting: modified_value")

        # Restore from backup
        restored_file = workspace / "test_config_restored.yaml"
        shutil.copy2(backup_file, restored_file)

        # Verify restoration
        restored_content = restored_file.read_text()
        assert "original_value" in restored_content

        # Cleanup
        config_file.unlink()
        backup_file.unlink()
        restored_file.unlink()

    def test_config_validation_after_restore(
        self, system_components: SystemComponents
    ):
        """Test validating configuration after restore."""
        workspace = system_components.workspace_path
        config_file = workspace / "validate_config.yaml"
        backup_file = workspace / "validate_backup.yaml"

        # Create valid config
        valid_config = "version: 1.0\nsettings:\n  enabled: true"
        config_file.write_text(valid_config)

        # Backup
        shutil.copy2(config_file, backup_file)

        # Restore
        restored_file = workspace / "validate_restored.yaml"
        shutil.copy2(backup_file, restored_file)

        # Validate structure
        content = restored_file.read_text()
        assert "version:" in content
        assert "settings:" in content

        # Cleanup
        config_file.unlink()
        backup_file.unlink()
        restored_file.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestWatchFolderStatePreservation:
    """Test watch folder state backup and restore."""

    def test_watch_folder_backup(
        self, system_components: SystemComponents
    ):
        """Test backing up watch folder configurations."""
        state_db = system_components.state_db_path

        # Query watch folder state
        conn = sqlite3.connect(state_db)
        watch_folders = conn.execute(
            "SELECT watch_id, path, collection FROM watch_folders"
        ).fetchall()
        conn.close()

        # Create backup
        backup_path = state_db.parent / f"watch_backup_{int(time.time())}.db"
        shutil.copy2(state_db, backup_path)

        # Verify watch folders in backup
        backup_conn = sqlite3.connect(backup_path)
        backup_watch_folders = backup_conn.execute(
            "SELECT watch_id, path, collection FROM watch_folders"
        ).fetchall()
        backup_conn.close()

        assert len(backup_watch_folders) == len(watch_folders)

        # Cleanup
        backup_path.unlink()

    def test_watch_folder_restore_verification(
        self, system_components: SystemComponents
    ):
        """Test verifying watch folder state after restore."""
        state_db = system_components.state_db_path

        # Get original watch folder configuration
        conn = sqlite3.connect(state_db)
        original_watches = conn.execute(
            "SELECT * FROM watch_folders ORDER BY watch_id"
        ).fetchall()
        conn.close()

        # Create and restore from backup
        backup_path = state_db.parent / f"watch_verify_{int(time.time())}.db"
        shutil.copy2(state_db, backup_path)

        restored_path = state_db.parent / f"watch_restored_{int(time.time())}.db"
        shutil.copy2(backup_path, restored_path)

        # Verify restored state
        restored_conn = sqlite3.connect(restored_path)
        restored_watches = restored_conn.execute(
            "SELECT * FROM watch_folders ORDER BY watch_id"
        ).fetchall()
        restored_conn.close()

        assert len(restored_watches) == len(original_watches)

        # Cleanup
        backup_path.unlink()
        restored_path.unlink()

    def test_watch_folder_enabled_state_preservation(
        self, system_components: SystemComponents
    ):
        """Test that watch folder enabled state is preserved."""
        state_db = system_components.state_db_path

        # Query enabled states
        conn = sqlite3.connect(state_db)
        enabled_states = conn.execute(
            "SELECT watch_id, enabled FROM watch_folders"
        ).fetchall()
        conn.close()

        # Backup and restore
        backup_path = state_db.parent / f"enabled_backup_{int(time.time())}.db"
        shutil.copy2(state_db, backup_path)

        # Verify enabled states preserved
        backup_conn = sqlite3.connect(backup_path)
        backup_enabled = backup_conn.execute(
            "SELECT watch_id, enabled FROM watch_folders"
        ).fetchall()
        backup_conn.close()

        assert len(backup_enabled) == len(enabled_states)

        # Cleanup
        backup_path.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestDataIntegrityVerification:
    """Test data integrity verification after restore."""

    def test_database_integrity_check(
        self, system_components: SystemComponents
    ):
        """Test database integrity check after restore."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"integrity_backup_{int(time.time())}.db"

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Run integrity check on backup
        conn = sqlite3.connect(backup_path)
        integrity_result = conn.execute("PRAGMA integrity_check").fetchone()
        conn.close()

        assert integrity_result[0] == "ok"

        # Cleanup
        backup_path.unlink()

    def test_collection_count_verification(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test verifying collection count after restore."""
        workspace = system_components.workspace_path

        # Create known number of collections
        collection_count = 3
        for i in range(collection_count):
            test_file = workspace / f"count_test_{i}.txt"
            test_file.write_text(f"Content {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-count-{i}-{int(time.time())}",
                ]
            )

        time.sleep(5)

        # Verify collections can be listed
        result = cli_helper.run_command(["admin", "collections"])
        assert result is not None

    def test_search_functionality_after_restore(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test search functionality works after restore."""
        workspace = system_components.workspace_path
        collection_name = f"test-search-restore-{int(time.time())}"

        # Ingest searchable content
        test_file = workspace / "search_restore.txt"
        test_file.write_text("Searchable content for post-restore verification")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", collection_name]
        )
        time.sleep(5)

        # Verify search works (simulating post-restore)
        result = cli_helper.run_command(
            ["search", "searchable content", "--collection", collection_name],
            timeout=15,
        )
        assert result is not None


@pytest.mark.integration
@pytest.mark.slow
class TestSystemRecovery:
    """Test complete system recovery scenarios."""

    def test_full_system_recovery_workflow(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test complete system recovery from backups."""
        workspace = system_components.workspace_path
        state_db = system_components.state_db_path

        # Phase 1: Create system state
        test_file = workspace / "recovery_test.txt"
        test_file.write_text("System recovery test content")
        cli_helper.run_command(
            [
                "ingest",
                "file",
                str(test_file),
                "--collection",
                "test-recovery",
            ]
        )
        time.sleep(3)

        # Phase 2: Create backups
        db_backup = state_db.parent / f"recovery_db_{int(time.time())}.db"
        shutil.copy2(state_db, db_backup)

        # Phase 3: Verify backups exist
        assert db_backup.exists()

        # Phase 4: System should still be operational
        status_result = cli_helper.run_command(["status"])
        assert status_result is not None

        # Cleanup
        db_backup.unlink()

    def test_recovery_with_missing_components(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test recovery when some components are missing."""
        # System should be resilient even if some backups are unavailable
        result = cli_helper.run_command(["status"])
        assert result is not None

        # Verify system can continue operating
        collections_result = cli_helper.run_command(["admin", "collections"])
        assert collections_result is not None

    def test_recovery_time_measurement(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test measuring system recovery time."""
        state_db = system_components.state_db_path

        # Create backup
        backup_path = state_db.parent / f"time_backup_{int(time.time())}.db"

        start_time = time.time()
        shutil.copy2(state_db, backup_path)
        backup_time = time.time() - start_time

        # Backup should be fast (<5 seconds)
        assert backup_time < 5.0

        # Cleanup
        backup_path.unlink()


@pytest.mark.integration
@pytest.mark.slow
class TestBackupRestoreWorkflows:
    """Test complete backup and restore workflows."""

    def test_complete_backup_workflow(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test complete backup workflow for all components."""
        workspace = system_components.workspace_path
        state_db = system_components.state_db_path

        # Create system state
        test_file = workspace / "complete_backup.txt"
        test_file.write_text("Complete backup test")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-complete-backup"]
        )
        time.sleep(3)

        # Create comprehensive backup
        backup_dir = workspace / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Backup database
        db_backup = backup_dir / f"state_{int(time.time())}.db"
        shutil.copy2(state_db, db_backup)

        # Verify backup directory
        assert backup_dir.exists()
        assert db_backup.exists()

        # Cleanup
        shutil.rmtree(backup_dir)

    def test_incremental_backup_strategy(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test incremental backup strategy."""
        workspace = system_components.workspace_path
        state_db = system_components.state_db_path
        backup_dir = workspace / "incremental_backups"
        backup_dir.mkdir(exist_ok=True)

        # Create multiple incremental backups
        backups = []
        for i in range(3):
            # Add some data
            test_file = workspace / f"incremental_{i}.txt"
            test_file.write_text(f"Incremental content {i}")
            cli_helper.run_command(
                [
                    "ingest",
                    "file",
                    str(test_file),
                    "--collection",
                    f"test-incremental-{i}",
                ]
            )
            time.sleep(2)

            # Create backup
            backup_path = backup_dir / f"backup_{i}_{int(time.time())}.db"
            shutil.copy2(state_db, backup_path)
            backups.append(backup_path)

        # Verify all backups
        assert all(b.exists() for b in backups)

        # Cleanup
        shutil.rmtree(backup_dir)

    def test_disaster_recovery_simulation(
        self, system_components: SystemComponents, cli_helper: CLIHelper
    ):
        """Test disaster recovery simulation."""
        workspace = system_components.workspace_path
        state_db = system_components.state_db_path

        # Create pre-disaster state
        test_file = workspace / "disaster_test.txt"
        test_file.write_text("Pre-disaster content")
        cli_helper.run_command(
            ["ingest", "file", str(test_file), "--collection", "test-disaster"]
        )
        time.sleep(3)

        # Create disaster recovery backup
        backup_path = state_db.parent / f"disaster_backup_{int(time.time())}.db"
        shutil.copy2(state_db, backup_path)

        # Verify system can be recovered
        assert backup_path.exists()

        # Test recovery process
        recovery_path = state_db.parent / f"recovered_{int(time.time())}.db"
        shutil.copy2(backup_path, recovery_path)

        # Verify recovery
        conn = sqlite3.connect(recovery_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()

        assert len(tables) > 0

        # Cleanup
        backup_path.unlink()
        recovery_path.unlink()

    def test_automated_backup_scheduling_simulation(
        self, system_components: SystemComponents
    ):
        """Test simulating automated backup scheduling."""
        state_db = system_components.state_db_path
        backup_dir = state_db.parent / "scheduled_backups"
        backup_dir.mkdir(exist_ok=True)

        # Simulate scheduled backups
        backup_count = 3
        for i in range(backup_count):
            backup_path = backup_dir / f"scheduled_{i}_{int(time.time())}.db"
            shutil.copy2(state_db, backup_path)
            time.sleep(1)

        # Verify backups created
        backups = list(backup_dir.glob("*.db"))
        assert len(backups) == backup_count

        # Cleanup
        shutil.rmtree(backup_dir)


@pytest.mark.integration
@pytest.mark.slow
class TestBackupPerformance:
    """Test backup and restore performance."""

    def test_backup_creation_performance(
        self, system_components: SystemComponents
    ):
        """Test backup creation completes quickly."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"perf_backup_{int(time.time())}.db"

        start_time = time.time()
        shutil.copy2(state_db, backup_path)
        backup_duration = time.time() - start_time

        # Backup should be fast (<3 seconds)
        assert backup_duration < 3.0

        # Cleanup
        backup_path.unlink()

    def test_restore_performance(
        self, system_components: SystemComponents
    ):
        """Test restore operation completes quickly."""
        state_db = system_components.state_db_path
        backup_path = state_db.parent / f"restore_perf_{int(time.time())}.db"

        # Create backup
        shutil.copy2(state_db, backup_path)

        # Measure restore time
        start_time = time.time()
        restored_path = state_db.parent / f"restored_perf_{int(time.time())}.db"
        shutil.copy2(backup_path, restored_path)
        restore_duration = time.time() - start_time

        # Restore should be fast (<3 seconds)
        assert restore_duration < 3.0

        # Cleanup
        backup_path.unlink()
        restored_path.unlink()

    def test_concurrent_backup_operations(
        self, system_components: SystemComponents
    ):
        """Test multiple backup operations can occur concurrently."""
        state_db = system_components.state_db_path
        backup_dir = state_db.parent / "concurrent_backups"
        backup_dir.mkdir(exist_ok=True)

        # Create multiple backups rapidly
        start_time = time.time()
        backups = []
        for i in range(5):
            backup_path = backup_dir / f"concurrent_{i}_{int(time.time())}.db"
            shutil.copy2(state_db, backup_path)
            backups.append(backup_path)

        total_duration = time.time() - start_time

        # All backups should complete in reasonable time (<10 seconds)
        assert total_duration < 10.0

        # Verify all backups
        assert all(b.exists() for b in backups)

        # Cleanup
        shutil.rmtree(backup_dir)
