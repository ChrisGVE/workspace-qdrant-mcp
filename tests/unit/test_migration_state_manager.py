"""Unit tests for MigrationStateManager (Task 410).

Tests SQLite-based migration state tracking for rollback capability.
"""

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest
from wqm_cli.cli.commands.migrate import (
    MigrationState,
    MigrationStateManager,
    MigrationStatus,
)


@pytest.fixture
def state_manager():
    """Create a MigrationStateManager with a temporary database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_migrations.db"
        manager = MigrationStateManager(db_path=db_path)
        yield manager


class TestMigrationStateManager:
    """Tests for MigrationStateManager."""

    def test_initialization_creates_database(self, state_manager):
        """Test that initialization creates the database file."""
        assert state_manager.db_path.exists()

    def test_start_migration_returns_uuid(self, state_manager):
        """Test that start_migration returns a valid UUID."""
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1", "_proj2"],
        )

        assert migration_id is not None
        assert len(migration_id) == 36  # UUID format: 8-4-4-4-12
        assert "-" in migration_id

    def test_start_migration_stores_state(self, state_manager):
        """Test that start_migration stores the migration state."""
        collections = ["_proj1", "_proj2"]
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=collections,
        )

        state = state_manager.get_migration(migration_id)

        assert state is not None
        assert state.migration_id == migration_id
        assert state.migration_type == "to-multitenant"
        assert state.status == MigrationStatus.RUNNING
        assert state.collections_migrated == collections
        assert state.points_migrated == 0
        assert state.error_message is None

    def test_update_progress(self, state_manager):
        """Test that update_progress updates the points count."""
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1"],
        )

        state_manager.update_progress(migration_id, 500)
        state = state_manager.get_migration(migration_id)
        assert state.points_migrated == 500

        state_manager.update_progress(migration_id, 1000)
        state = state_manager.get_migration(migration_id)
        assert state.points_migrated == 1000

    def test_complete_migration(self, state_manager):
        """Test that complete_migration marks the migration as completed."""
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1"],
        )

        report_path = "/tmp/report.json"
        state_manager.complete_migration(
            migration_id=migration_id,
            points_migrated=1500,
            report_path=report_path,
        )

        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.COMPLETED
        assert state.points_migrated == 1500
        assert state.completed_at is not None
        assert state.report_path == report_path
        assert state.error_message is None

    def test_fail_migration(self, state_manager):
        """Test that fail_migration marks the migration as failed."""
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1"],
        )

        error_msg = "Connection timeout"
        state_manager.fail_migration(migration_id, error_msg)

        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.FAILED
        assert state.error_message == error_msg
        assert state.completed_at is not None

    def test_rollback_migration(self, state_manager):
        """Test that rollback_migration marks the migration as rolled back."""
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1", "_proj2"],
        )

        state_manager.complete_migration(
            migration_id=migration_id,
            points_migrated=1000,
        )

        state_manager.rollback_migration(migration_id)

        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.ROLLED_BACK

    def test_list_migrations_returns_all(self, state_manager):
        """Test that list_migrations returns all migrations."""
        id1 = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1"],
        )
        id2 = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj2"],
        )

        migrations = state_manager.list_migrations()

        assert len(migrations) == 2
        ids = [m.migration_id for m in migrations]
        assert id1 in ids
        assert id2 in ids

    def test_list_migrations_with_status_filter(self, state_manager):
        """Test that list_migrations filters by status."""
        id1 = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj1"],
        )
        state_manager.complete_migration(id1, 100)

        id2 = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj2"],
        )
        state_manager.fail_migration(id2, "Error")

        id3 = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=["_proj3"],
        )
        # id3 stays RUNNING

        completed = state_manager.list_migrations(status=MigrationStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].migration_id == id1

        failed = state_manager.list_migrations(status=MigrationStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].migration_id == id2

        running = state_manager.list_migrations(status=MigrationStatus.RUNNING)
        assert len(running) == 1
        assert running[0].migration_id == id3

    def test_list_migrations_with_limit(self, state_manager):
        """Test that list_migrations respects the limit parameter."""
        for i in range(5):
            state_manager.start_migration(
                migration_type="to-multitenant",
                collections=[f"_proj{i}"],
            )

        migrations = state_manager.list_migrations(limit=3)
        assert len(migrations) == 3

    def test_get_migration_returns_none_for_missing(self, state_manager):
        """Test that get_migration returns None for non-existent ID."""
        result = state_manager.get_migration("non-existent-id")
        assert result is None

    def test_migration_state_dataclass(self):
        """Test MigrationState dataclass fields."""
        state = MigrationState(
            migration_id="test-id",
            migration_type="to-multitenant",
            started_at="2024-01-01T00:00:00Z",
            completed_at=None,
            status=MigrationStatus.RUNNING,
            collections_migrated=["_proj1"],
            points_migrated=0,
            error_message=None,
            report_path=None,
        )

        assert state.migration_id == "test-id"
        assert state.status == MigrationStatus.RUNNING
        assert len(state.collections_migrated) == 1

    def test_migration_status_enum_values(self):
        """Test MigrationStatus enum has correct values."""
        assert MigrationStatus.RUNNING.value == "running"
        assert MigrationStatus.COMPLETED.value == "completed"
        assert MigrationStatus.FAILED.value == "failed"
        assert MigrationStatus.ROLLED_BACK.value == "rolled_back"

    def test_multiple_operations_on_same_migration(self, state_manager):
        """Test a full lifecycle: start -> update -> complete -> rollback."""
        collections = ["_proj1", "_proj2", "_proj3"]
        migration_id = state_manager.start_migration(
            migration_type="to-multitenant",
            collections=collections,
        )

        # Check initial state
        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.RUNNING
        assert state.points_migrated == 0

        # Update progress
        state_manager.update_progress(migration_id, 500)
        state = state_manager.get_migration(migration_id)
        assert state.points_migrated == 500

        # Complete
        state_manager.complete_migration(migration_id, 1500, "/tmp/report.json")
        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.COMPLETED
        assert state.points_migrated == 1500

        # Rollback
        state_manager.rollback_migration(migration_id)
        state = state_manager.get_migration(migration_id)
        assert state.status == MigrationStatus.ROLLED_BACK
        # Points count preserved after rollback
        assert state.points_migrated == 1500
