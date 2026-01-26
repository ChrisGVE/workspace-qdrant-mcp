"""Tests for active project state tracking (Task 36 - code audit round 2).

This module tests the ActiveProjectState dataclass and associated CRUD methods
in SQLiteStateManager for the fairness scheduler.
"""

import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ActiveProjectState,
    WatchFolderConfig,
)


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_state.db")
        manager = SQLiteStateManager(db_path=db_path)
        await manager.initialize()
        yield manager
        await manager.close()


class TestActiveProjectState:
    """Test the ActiveProjectState dataclass."""

    def test_dataclass_creation(self):
        """Test creating an ActiveProjectState with required fields."""
        state = ActiveProjectState(
            project_id="test-project-123",
            tenant_id="tenant-1",
        )

        assert state.project_id == "test-project-123"
        assert state.tenant_id == "tenant-1"
        assert state.items_processed_count == 0
        assert state.items_in_queue == 0
        assert state.watch_enabled is False
        assert state.watch_folder_id is None
        assert state.created_at is not None
        assert state.updated_at is not None
        assert state.last_activity_at is not None

    def test_dataclass_with_all_fields(self):
        """Test creating an ActiveProjectState with all fields."""
        now = datetime.now(timezone.utc)
        state = ActiveProjectState(
            project_id="test-project-456",
            tenant_id="tenant-2",
            last_activity_at=now,
            items_processed_count=100,
            items_in_queue=5,
            watch_enabled=True,
            watch_folder_id="watch-123",
            created_at=now,
            updated_at=now,
            metadata={"key": "value"},
        )

        assert state.project_id == "test-project-456"
        assert state.tenant_id == "tenant-2"
        assert state.last_activity_at == now
        assert state.items_processed_count == 100
        assert state.items_in_queue == 5
        assert state.watch_enabled is True
        assert state.watch_folder_id == "watch-123"
        assert state.metadata == {"key": "value"}


class TestRegisterActiveProject:
    """Test register_active_project method."""

    @pytest.mark.asyncio
    async def test_register_new_project(self, state_manager):
        """Test registering a new active project."""
        result = await state_manager.register_active_project(
            project_id="project-new",
            tenant_id="tenant-1",
        )

        assert result is not None
        assert result.project_id == "project-new"
        assert result.tenant_id == "tenant-1"
        assert result.items_processed_count == 0
        assert result.items_in_queue == 0
        assert result.watch_enabled is False

    @pytest.mark.asyncio
    async def test_register_project_with_watch(self, state_manager):
        """Test registering a project with watch folder."""
        # First create a watch folder to satisfy foreign key constraint
        watch_config = WatchFolderConfig(
            watch_id="watch-123",
            path="/tmp/test-project",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
        )
        await state_manager.save_watch_folder_config(watch_config)

        result = await state_manager.register_active_project(
            project_id="project-watched",
            tenant_id="tenant-1",
            watch_folder_id="watch-123",
        )

        assert result is not None
        assert result.watch_enabled is True
        assert result.watch_folder_id == "watch-123"

    @pytest.mark.asyncio
    async def test_register_project_with_metadata(self, state_manager):
        """Test registering a project with metadata."""
        metadata = {"git_remote": "https://github.com/example/repo"}
        result = await state_manager.register_active_project(
            project_id="project-meta",
            tenant_id="tenant-1",
            metadata=metadata,
        )

        assert result is not None
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_update_existing_project(self, state_manager):
        """Test updating an existing project's activity."""
        # First create a watch folder to satisfy foreign key constraint
        watch_config = WatchFolderConfig(
            watch_id="watch-456",
            path="/tmp/test-project-update",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
        )
        await state_manager.save_watch_folder_config(watch_config)

        # Register project first
        await state_manager.register_active_project(
            project_id="project-update",
            tenant_id="tenant-1",
        )

        # Re-register with watch folder
        result = await state_manager.register_active_project(
            project_id="project-update",
            tenant_id="tenant-1",
            watch_folder_id="watch-456",
        )

        assert result is not None
        assert result.watch_enabled is True
        assert result.watch_folder_id == "watch-456"


class TestGetActiveProject:
    """Test get_active_project method."""

    @pytest.mark.asyncio
    async def test_get_existing_project(self, state_manager):
        """Test getting an existing project."""
        await state_manager.register_active_project(
            project_id="project-get",
            tenant_id="tenant-1",
        )

        result = await state_manager.get_active_project("project-get")

        assert result is not None
        assert result.project_id == "project-get"

    @pytest.mark.asyncio
    async def test_get_nonexistent_project(self, state_manager):
        """Test getting a project that doesn't exist."""
        result = await state_manager.get_active_project("nonexistent")

        assert result is None


class TestListActiveProjects:
    """Test list_active_projects method."""

    @pytest.mark.asyncio
    async def test_list_all_projects(self, state_manager):
        """Test listing all active projects."""
        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project("project-2", "tenant-1")
        await state_manager.register_active_project("project-3", "tenant-2")

        result = await state_manager.list_active_projects()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_by_tenant(self, state_manager):
        """Test listing projects filtered by tenant."""
        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project("project-2", "tenant-1")
        await state_manager.register_active_project("project-3", "tenant-2")

        result = await state_manager.list_active_projects(tenant_id="tenant-1")

        assert len(result) == 2
        assert all(p.tenant_id == "tenant-1" for p in result)

    @pytest.mark.asyncio
    async def test_list_watched_only(self, state_manager):
        """Test listing only watched projects."""
        # Create watch folder first
        watch_config = WatchFolderConfig(
            watch_id="watch-1",
            path="/tmp/test-project-watch",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
        )
        await state_manager.save_watch_folder_config(watch_config)

        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project(
            "project-2", "tenant-1", watch_folder_id="watch-1"
        )
        await state_manager.register_active_project("project-3", "tenant-1")

        result = await state_manager.list_active_projects(watch_enabled_only=True)

        assert len(result) == 1
        assert result[0].project_id == "project-2"

    @pytest.mark.asyncio
    async def test_list_with_limit(self, state_manager):
        """Test listing with a limit."""
        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project("project-2", "tenant-1")
        await state_manager.register_active_project("project-3", "tenant-1")

        result = await state_manager.list_active_projects(limit=2)

        assert len(result) == 2


class TestUpdateActiveProjectActivity:
    """Test update_active_project_activity method."""

    @pytest.mark.asyncio
    async def test_update_activity(self, state_manager):
        """Test updating project activity."""
        await state_manager.register_active_project("project-activity", "tenant-1")

        result = await state_manager.update_active_project_activity(
            "project-activity", items_processed=5
        )

        assert result is True

        project = await state_manager.get_active_project("project-activity")
        assert project.items_processed_count == 5

    @pytest.mark.asyncio
    async def test_update_activity_cumulative(self, state_manager):
        """Test that activity updates are cumulative."""
        await state_manager.register_active_project("project-cumulative", "tenant-1")

        await state_manager.update_active_project_activity("project-cumulative", 3)
        await state_manager.update_active_project_activity("project-cumulative", 2)

        project = await state_manager.get_active_project("project-cumulative")
        assert project.items_processed_count == 5

    @pytest.mark.asyncio
    async def test_update_nonexistent_project(self, state_manager):
        """Test updating a nonexistent project."""
        result = await state_manager.update_active_project_activity("nonexistent", 1)

        assert result is False


class TestUpdateActiveProjectQueueCount:
    """Test update_active_project_queue_count method."""

    @pytest.mark.asyncio
    async def test_increment_queue(self, state_manager):
        """Test incrementing queue count."""
        await state_manager.register_active_project("project-queue", "tenant-1")

        result = await state_manager.update_active_project_queue_count(
            "project-queue", queue_delta=3
        )

        assert result is True

        project = await state_manager.get_active_project("project-queue")
        assert project.items_in_queue == 3

    @pytest.mark.asyncio
    async def test_decrement_queue(self, state_manager):
        """Test decrementing queue count."""
        await state_manager.register_active_project("project-dec", "tenant-1")
        await state_manager.update_active_project_queue_count("project-dec", 5)

        result = await state_manager.update_active_project_queue_count(
            "project-dec", queue_delta=-2
        )

        assert result is True

        project = await state_manager.get_active_project("project-dec")
        assert project.items_in_queue == 3

    @pytest.mark.asyncio
    async def test_queue_cannot_go_negative(self, state_manager):
        """Test that queue count cannot go below zero."""
        await state_manager.register_active_project("project-neg", "tenant-1")
        await state_manager.update_active_project_queue_count("project-neg", 2)

        # Try to decrement by more than current count
        await state_manager.update_active_project_queue_count("project-neg", -10)

        project = await state_manager.get_active_project("project-neg")
        assert project.items_in_queue == 0


class TestGarbageCollectStaleProjects:
    """Test garbage_collect_stale_projects method."""

    @pytest.mark.asyncio
    async def test_gc_removes_stale_projects(self, state_manager):
        """Test that garbage collection removes stale projects."""
        # Register projects
        await state_manager.register_active_project("project-active", "tenant-1")
        await state_manager.register_active_project("project-stale", "tenant-1")

        # Manually update project-stale to have old activity timestamp
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE active_projects
                SET last_activity_at = datetime('now', '-48 hours')
                WHERE project_id = 'project-stale'
                """
            )

        # Run garbage collection
        removed = await state_manager.garbage_collect_stale_projects(
            max_inactive_hours=24
        )

        assert removed == 1

        # Verify active project still exists
        active = await state_manager.get_active_project("project-active")
        assert active is not None

        # Verify stale project was removed
        stale = await state_manager.get_active_project("project-stale")
        assert stale is None

    @pytest.mark.asyncio
    async def test_gc_keeps_recent_projects(self, state_manager):
        """Test that garbage collection keeps recent projects."""
        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project("project-2", "tenant-1")

        removed = await state_manager.garbage_collect_stale_projects(
            max_inactive_hours=24
        )

        assert removed == 0

        projects = await state_manager.list_active_projects()
        assert len(projects) == 2


class TestRemoveActiveProject:
    """Test remove_active_project method."""

    @pytest.mark.asyncio
    async def test_remove_existing_project(self, state_manager):
        """Test removing an existing project."""
        await state_manager.register_active_project("project-remove", "tenant-1")

        result = await state_manager.remove_active_project("project-remove")

        assert result is True

        project = await state_manager.get_active_project("project-remove")
        assert project is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_project(self, state_manager):
        """Test removing a nonexistent project."""
        result = await state_manager.remove_active_project("nonexistent")

        assert result is False


class TestActiveProjectsStats:
    """Test get_active_projects_stats method."""

    @pytest.mark.asyncio
    async def test_stats_with_projects(self, state_manager):
        """Test getting stats with active projects."""
        # Create watch folder first
        watch_config = WatchFolderConfig(
            watch_id="watch-stats",
            path="/tmp/test-project-stats",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=["__pycache__/*"],
        )
        await state_manager.save_watch_folder_config(watch_config)

        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project(
            "project-2", "tenant-1", watch_folder_id="watch-stats"
        )

        # Update activity
        await state_manager.update_active_project_activity("project-1", 10)
        await state_manager.update_active_project_activity("project-2", 5)
        await state_manager.update_active_project_queue_count("project-1", 3)

        stats = await state_manager.get_active_projects_stats()

        assert stats["total_projects"] == 2
        assert stats["watched_projects"] == 1
        assert stats["total_items_processed"] == 15
        assert stats["total_items_in_queue"] == 3
        assert stats["active_last_hour"] == 2

    @pytest.mark.asyncio
    async def test_stats_empty(self, state_manager):
        """Test getting stats with no projects."""
        stats = await state_manager.get_active_projects_stats()

        # Empty stats should return zeros
        assert stats.get("total_projects", 0) == 0


class TestGetStaleProjects:
    """Test get_stale_projects method."""

    @pytest.mark.asyncio
    async def test_list_stale_projects(self, state_manager):
        """Test listing stale projects."""
        await state_manager.register_active_project("project-recent", "tenant-1")
        await state_manager.register_active_project("project-old", "tenant-1")

        # Make project-old stale
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE active_projects
                SET last_activity_at = datetime('now', '-48 hours')
                WHERE project_id = 'project-old'
                """
            )

        stale = await state_manager.get_stale_projects()

        assert len(stale) == 1
        assert stale[0]["project_id"] == "project-old"
        assert stale[0]["days_inactive"] > 1

    @pytest.mark.asyncio
    async def test_no_stale_projects(self, state_manager):
        """Test when there are no stale projects."""
        await state_manager.register_active_project("project-recent", "tenant-1")

        stale = await state_manager.get_stale_projects()

        assert len(stale) == 0
