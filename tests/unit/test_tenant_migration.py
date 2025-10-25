"""
Unit tests for tenant ID migration system.

Tests cover:
- Tenant change detection
- Migration planning
- Queue entry migration
- Metadata entry migration
- Migration validation
- Rollback functionality
- Progress tracking
- Audit logging
"""

import asyncio
import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.python.common.core.sqlite_state_manager import SQLiteStateManager
from src.python.common.core.tenant_migration import (
    MigrationPlan,
    MigrationResult,
    MigrationStatistics,
    MigrationStatus,
    TenantMigrationManager,
)


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    manager = SQLiteStateManager(db_path=db_path)
    await manager.initialize()

    yield manager

    await manager.close()
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
async def migration_manager(state_manager):
    """Create a tenant migration manager instance."""
    return TenantMigrationManager(state_manager)


@pytest.fixture
async def populated_db(state_manager):
    """Populate database with test data."""
    # Add queue entries
    with state_manager._lock:
        for i in range(10):
            state_manager.connection.execute(
                """
                INSERT INTO ingestion_queue (
                    file_absolute_path, collection_name, tenant_id, branch, operation
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    f"/test/project/file{i}.py",
                    "test-collection",
                    "old_tenant_id",
                    "main",
                    "ingest"
                )
            )

        # Add file processing entries
        for i in range(5):
            metadata = json.dumps({"tenant_id": "old_tenant_id"})
            state_manager.connection.execute(
                """
                INSERT INTO file_processing (
                    file_path, collection, status, metadata
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    f"/test/project/file{i}.py",
                    "test-collection",
                    "completed",
                    metadata
                )
            )

        state_manager.connection.commit()

    return state_manager


class TestTenantMigrationManager:
    """Test suite for TenantMigrationManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, migration_manager):
        """Test manager initialization creates migration table."""
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tenant_migrations'"
        )
        assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_detect_tenant_change_no_stored_id(self, migration_manager, tmp_path):
        """Test detection when no stored tenant ID exists."""
        project_root = tmp_path / "test-project"
        project_root.mkdir()

        changed, old_id, new_id = await migration_manager.detect_tenant_change(
            project_root, stored_tenant_id=None
        )

        assert not changed
        assert old_id is None
        assert new_id is not None  # Should calculate new tenant ID

    @pytest.mark.asyncio
    async def test_detect_tenant_change_no_change(self, migration_manager, tmp_path):
        """Test detection when tenant ID hasn't changed."""
        project_root = tmp_path / "test-project"
        project_root.mkdir()

        # Mock calculate_tenant_id to return consistent value
        with patch('src.python.common.core.tenant_migration.calculate_tenant_id') as mock_calc:
            mock_calc.return_value = "same_tenant_id"

            changed, old_id, new_id = await migration_manager.detect_tenant_change(
                project_root, stored_tenant_id="same_tenant_id"
            )

            assert not changed
            assert old_id == "same_tenant_id"
            assert new_id == "same_tenant_id"

    @pytest.mark.asyncio
    async def test_detect_tenant_change_changed(self, migration_manager, tmp_path):
        """Test detection when tenant ID has changed."""
        project_root = tmp_path / "test-project"
        project_root.mkdir()

        with patch('src.python.common.core.tenant_migration.calculate_tenant_id') as mock_calc:
            mock_calc.return_value = "new_tenant_id"

            changed, old_id, new_id = await migration_manager.detect_tenant_change(
                project_root, stored_tenant_id="old_tenant_id"
            )

            assert changed
            assert old_id == "old_tenant_id"
            assert new_id == "new_tenant_id"

    @pytest.mark.asyncio
    async def test_plan_migration(self, migration_manager, populated_db, tmp_path):
        """Test migration plan creation."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        assert plan.migration_id > 0
        assert plan.old_tenant_id == "old_tenant_id"
        assert plan.new_tenant_id == "new_tenant_id"
        assert plan.status == MigrationStatus.PLANNING
        assert plan.statistics.queue_entries_total == 10
        assert plan.statistics.metadata_entries_total == 5
        assert len(plan.audit_log) > 0

    @pytest.mark.asyncio
    async def test_plan_migration_same_tenant_ids(self, migration_manager):
        """Test that planning fails with same old/new tenant IDs."""
        with pytest.raises(ValueError, match="Old and new tenant IDs are identical"):
            await migration_manager.plan_migration(
                Path("/test/project"), "same_id", "same_id"
            )

    @pytest.mark.asyncio
    async def test_execute_migration_success(self, migration_manager, populated_db, tmp_path):
        """Test successful migration execution."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        result = await migration_manager.execute_migration(plan)

        assert result.success
        assert result.migration_id == plan.migration_id
        assert result.statistics.queue_entries_migrated == 10
        assert result.statistics.metadata_entries_migrated == 5
        assert plan.status == MigrationStatus.COMPLETED

        # Verify queue entries were migrated
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("new_tenant_id",)
        )
        assert cursor.fetchone()[0] == 10

        # Verify no entries with old tenant ID remain
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("old_tenant_id",)
        )
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_execute_migration_failure(self, migration_manager, tmp_path):
        """Test migration failure handling."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        # Force a failure by corrupting the database connection
        original_connection = migration_manager.state_manager.connection
        migration_manager.state_manager.connection = None

        result = await migration_manager.execute_migration(plan)

        # Restore connection
        migration_manager.state_manager.connection = original_connection

        assert not result.success
        assert result.error_message is not None
        assert len(plan.statistics.errors) > 0
        assert plan.status == MigrationStatus.FAILED

    @pytest.mark.asyncio
    async def test_migrate_queue_entries(self, migration_manager, populated_db):
        """Test queue entry migration."""
        plan = MigrationPlan(
            migration_id=1,
            project_root="/test/project",
            old_tenant_id="old_tenant_id",
            new_tenant_id="new_tenant_id",
        )

        migrated_count = await migration_manager._migrate_queue_entries(plan)

        assert migrated_count == 10

        # Verify migration
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT tenant_id FROM ingestion_queue"
        )
        tenant_ids = [row[0] for row in cursor.fetchall()]
        assert all(tid == "new_tenant_id" for tid in tenant_ids)

    @pytest.mark.asyncio
    async def test_migrate_metadata_entries(self, migration_manager, populated_db):
        """Test metadata entry migration."""
        plan = MigrationPlan(
            migration_id=1,
            project_root="/test/project",
            old_tenant_id="old_tenant_id",
            new_tenant_id="new_tenant_id",
        )

        migrated_count = await migration_manager._migrate_metadata_entries(plan)

        assert migrated_count == 5

        # Verify migration
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT metadata FROM file_processing"
        )
        for row in cursor.fetchall():
            metadata = json.loads(row[0])
            assert metadata["tenant_id"] == "new_tenant_id"

    @pytest.mark.asyncio
    async def test_migrate_metadata_null_handling(self, migration_manager, state_manager):
        """Test migration handles NULL metadata gracefully."""
        # Add entry with NULL metadata
        with state_manager._lock:
            state_manager.connection.execute(
                """
                INSERT INTO file_processing (file_path, collection, status, metadata)
                VALUES (?, ?, ?, ?)
                """,
                ("/test/project/file.py", "test-collection", "completed", None)
            )
            state_manager.connection.commit()

        plan = MigrationPlan(
            migration_id=1,
            project_root="/test/project",
            old_tenant_id="old_tenant_id",
            new_tenant_id="new_tenant_id",
        )

        migrated_count = await migration_manager._migrate_metadata_entries(plan)

        # Should create metadata with new tenant ID
        assert migrated_count == 1

        cursor = state_manager.connection.execute(
            "SELECT metadata FROM file_processing WHERE file_path = ?",
            ("/test/project/file.py",)
        )
        metadata = json.loads(cursor.fetchone()[0])
        assert metadata["tenant_id"] == "new_tenant_id"

    @pytest.mark.asyncio
    async def test_validate_migration_success(self, migration_manager, populated_db, tmp_path):
        """Test migration validation succeeds after successful migration."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        # Execute migration
        await migration_manager.execute_migration(plan)

        # Validation should pass
        is_valid = await migration_manager._validate_migration(plan)
        assert is_valid

    @pytest.mark.asyncio
    async def test_validate_migration_failure(self, migration_manager, populated_db):
        """Test migration validation fails when old tenant ID entries remain."""
        plan = MigrationPlan(
            migration_id=1,
            project_root="/test/project",
            old_tenant_id="old_tenant_id",
            new_tenant_id="new_tenant_id",
        )

        # Set expected counts but don't actually migrate
        plan.statistics.queue_entries_migrated = 10

        # Validation should fail
        is_valid = await migration_manager._validate_migration(plan)
        assert not is_valid

    @pytest.mark.asyncio
    async def test_rollback_migration(self, migration_manager, populated_db, tmp_path):
        """Test migration rollback."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        # Execute migration
        await migration_manager.execute_migration(plan)

        # Verify migration succeeded
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("new_tenant_id",)
        )
        assert cursor.fetchone()[0] == 10

        # Rollback
        rollback_success = await migration_manager.rollback_migration(plan)
        assert rollback_success

        # Verify rollback
        cursor = migration_manager.state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("old_tenant_id",)
        )
        assert cursor.fetchone()[0] == 10

        cursor = migration_manager.state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("new_tenant_id",)
        )
        assert cursor.fetchone()[0] == 0

        assert plan.status == MigrationStatus.ROLLED_BACK

    @pytest.mark.asyncio
    async def test_migration_audit_log(self, migration_manager, populated_db, tmp_path):
        """Test audit logging during migration."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        await migration_manager.execute_migration(plan)

        # Verify audit log entries
        assert len(plan.audit_log) >= 5
        actions = [entry["action"] for entry in plan.audit_log]
        assert "plan_created" in actions
        assert "queue_migrated" in actions
        assert "metadata_migrated" in actions
        assert "validation_passed" in actions

        # Verify all entries have timestamps
        for entry in plan.audit_log:
            assert "timestamp" in entry
            assert "action" in entry
            assert "details" in entry
            assert "success" in entry

    @pytest.mark.asyncio
    async def test_migration_statistics(self, migration_manager, populated_db, tmp_path):
        """Test migration statistics tracking."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        plan = await migration_manager.plan_migration(
            project_root, "old_tenant_id", "new_tenant_id"
        )

        result = await migration_manager.execute_migration(plan)

        stats = result.statistics
        assert stats.queue_entries_total == 10
        assert stats.queue_entries_migrated == 10
        assert stats.metadata_entries_total == 5
        assert stats.metadata_entries_migrated == 5
        assert stats.start_time is not None
        assert stats.end_time is not None
        assert stats.end_time > stats.start_time

        # Test to_dict conversion
        stats_dict = stats.to_dict()
        assert "duration_seconds" in stats_dict
        assert stats_dict["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_get_migration_history(self, migration_manager, populated_db, tmp_path):
        """Test retrieving migration history."""
        project_root = tmp_path / "test" / "project"
        project_root.mkdir(parents=True)

        # Execute multiple migrations
        for i in range(3):
            plan = await migration_manager.plan_migration(
                project_root, f"old_id_{i}", f"new_id_{i}"
            )
            await migration_manager.execute_migration(plan)

        # Get history
        history = await migration_manager.get_migration_history(limit=10)

        assert len(history) == 3
        assert history[0]["old_tenant_id"] == "old_id_2"  # Most recent first
        assert all("audit_log" in record for record in history)

    @pytest.mark.asyncio
    async def test_get_migration_history_filtered(self, migration_manager, populated_db, tmp_path):
        """Test retrieving migration history filtered by project."""
        project_root1 = tmp_path / "project1"
        project_root2 = tmp_path / "project2"
        project_root1.mkdir()
        project_root2.mkdir()

        # Create migrations for different projects
        plan1 = await migration_manager.plan_migration(
            project_root1, "old_id_1", "new_id_1"
        )
        await migration_manager.execute_migration(plan1)

        plan2 = await migration_manager.plan_migration(
            project_root2, "old_id_2", "new_id_2"
        )
        await migration_manager.execute_migration(plan2)

        # Get history for project1 only
        history = await migration_manager.get_migration_history(
            project_root=project_root1, limit=10
        )

        assert len(history) == 1
        assert history[0]["project_root"] == str(project_root1.resolve())

    @pytest.mark.asyncio
    async def test_concurrent_migrations(self, migration_manager, state_manager):
        """Test that concurrent migrations are handled safely."""
        # Add test data for two different projects
        with state_manager._lock:
            for i in range(5):
                state_manager.connection.execute(
                    """
                    INSERT INTO ingestion_queue (
                        file_absolute_path, collection_name, tenant_id, branch
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        f"/project1/file{i}.py",
                        "test-collection",
                        "project1_old",
                        "main"
                    )
                )
                state_manager.connection.execute(
                    """
                    INSERT INTO ingestion_queue (
                        file_absolute_path, collection_name, tenant_id, branch
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        f"/project2/file{i}.py",
                        "test-collection",
                        "project2_old",
                        "main"
                    )
                )
            state_manager.connection.commit()

        # Create plans for both projects
        plan1 = await migration_manager.plan_migration(
            Path("/project1"), "project1_old", "project1_new"
        )
        plan2 = await migration_manager.plan_migration(
            Path("/project2"), "project2_old", "project2_new"
        )

        # Execute migrations concurrently
        results = await asyncio.gather(
            migration_manager.execute_migration(plan1),
            migration_manager.execute_migration(plan2),
        )

        # Both should succeed
        assert all(r.success for r in results)

        # Verify each project migrated correctly
        cursor = state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("project1_new",)
        )
        assert cursor.fetchone()[0] == 5

        cursor = state_manager.connection.execute(
            "SELECT COUNT(*) FROM ingestion_queue WHERE tenant_id = ?",
            ("project2_new",)
        )
        assert cursor.fetchone()[0] == 5

    @pytest.mark.asyncio
    async def test_migration_plan_to_db_dict(self):
        """Test conversion of migration plan to database dictionary."""
        plan = MigrationPlan(
            migration_id=1,
            project_root="/test/project",
            old_tenant_id="old_id",
            new_tenant_id="new_id",
            status=MigrationStatus.COMPLETED,
        )

        plan.add_audit_entry("test_action", "test details")
        plan.started_at = datetime.now(timezone.utc)
        plan.completed_at = datetime.now(timezone.utc)
        plan.statistics.queue_entries_migrated = 10

        db_dict = plan.to_db_dict()

        assert db_dict["id"] == 1
        assert db_dict["project_root"] == "/test/project"
        assert db_dict["old_tenant_id"] == "old_id"
        assert db_dict["new_tenant_id"] == "new_id"
        assert db_dict["status"] == "completed"
        assert db_dict["entries_migrated"] == 10
        assert "rollback_info" in db_dict
        assert "audit_log" in db_dict

    @pytest.mark.asyncio
    async def test_migration_result_to_dict(self):
        """Test conversion of migration result to dictionary."""
        stats = MigrationStatistics(
            queue_entries_total=10,
            queue_entries_migrated=10,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
        )

        result = MigrationResult(
            success=True,
            migration_id=1,
            old_tenant_id="old_id",
            new_tenant_id="new_id",
            statistics=stats,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["migration_id"] == 1
        assert "statistics" in result_dict
        assert "duration_seconds" in result_dict["statistics"]
