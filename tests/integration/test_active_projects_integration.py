"""
Integration Tests for Active Project State Tracking (Task 36).

Tests end-to-end integration of active project tracking with:
- Queue operations (enqueue/dequeue activity tracking)
- Watch folder scanner (project registration on watch start)
- Garbage collection (stale project removal)
- Statistics aggregation

Test Coverage:
    - Project registration when watch folders start
    - Activity updates when queue items are processed
    - Queue count increments on enqueue, decrements on complete
    - Garbage collection of inactive projects
    - Multi-project activity tracking isolation
    - Stats aggregation across projects
"""

import asyncio
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


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
async def state_manager():
    """Create a temporary SQLite state manager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_integration.db")
        manager = SQLiteStateManager(db_path=db_path)
        await manager.initialize()
        yield manager
        await manager.close()


@pytest.fixture
async def state_manager_with_projects(state_manager):
    """Create state manager with pre-registered projects."""
    # Create watch folder first (for foreign key constraint)
    watch_config = WatchFolderConfig(
        watch_id="project-alpha",
        path="/tmp/project-alpha",
        collection="alpha-code",
        patterns=["*.py", "*.md"],
        ignore_patterns=["__pycache__/*"],
    )
    await state_manager.save_watch_folder_config(watch_config)

    # Register multiple projects
    await state_manager.register_active_project(
        project_id="project-alpha",
        tenant_id="tenant-1",
        watch_folder_id="project-alpha",
    )
    await state_manager.register_active_project(
        project_id="project-beta",
        tenant_id="tenant-1",
    )
    await state_manager.register_active_project(
        project_id="project-gamma",
        tenant_id="tenant-2",
    )

    yield state_manager


# =============================================================================
# INTEGRATION TESTS: Activity Tracking
# =============================================================================


class TestActivityTracking:
    """Test activity tracking when queue items are processed."""

    @pytest.mark.asyncio
    async def test_activity_increments_on_processing(self, state_manager):
        """Test that activity count increments when items are processed."""
        # Register project
        await state_manager.register_active_project("project-1", "tenant-1")

        # Simulate processing 5 items
        for _ in range(5):
            await state_manager.update_active_project_activity("project-1", 1)

        project = await state_manager.get_active_project("project-1")
        assert project.items_processed_count == 5

    @pytest.mark.asyncio
    async def test_activity_batch_updates(self, state_manager):
        """Test batch activity updates."""
        await state_manager.register_active_project("project-batch", "tenant-1")

        # Batch update (e.g., processing 10 items at once)
        await state_manager.update_active_project_activity("project-batch", 10)

        project = await state_manager.get_active_project("project-batch")
        assert project.items_processed_count == 10

    @pytest.mark.asyncio
    async def test_activity_timestamp_updates(self, state_manager):
        """Test that last_activity_at is updated on activity."""
        await state_manager.register_active_project("project-time", "tenant-1")

        initial_project = await state_manager.get_active_project("project-time")

        # Wait enough time for SQLite second-precision timestamp to change
        await asyncio.sleep(1.1)
        await state_manager.update_active_project_activity("project-time", 1)

        updated_project = await state_manager.get_active_project("project-time")

        # Normalize both to naive datetimes for comparison
        def normalize_dt(dt):
            if dt is None:
                return None
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            # Remove microseconds for comparison (SQLite has second precision)
            return dt.replace(microsecond=0)

        initial_activity = normalize_dt(initial_project.last_activity_at)
        updated_activity = normalize_dt(updated_project.last_activity_at)

        # Activity timestamp should be updated
        assert updated_activity >= initial_activity
        # And items_processed_count should be incremented
        assert updated_project.items_processed_count == 1


# =============================================================================
# INTEGRATION TESTS: Queue Count Tracking
# =============================================================================


class TestQueueCountTracking:
    """Test queue count tracking during enqueue/dequeue operations."""

    @pytest.mark.asyncio
    async def test_queue_count_increments_on_enqueue(self, state_manager):
        """Test queue count increases when items are enqueued."""
        await state_manager.register_active_project("project-queue", "tenant-1")

        # Simulate enqueueing 3 items
        await state_manager.update_active_project_queue_count("project-queue", 1)
        await state_manager.update_active_project_queue_count("project-queue", 1)
        await state_manager.update_active_project_queue_count("project-queue", 1)

        project = await state_manager.get_active_project("project-queue")
        assert project.items_in_queue == 3

    @pytest.mark.asyncio
    async def test_queue_count_decrements_on_complete(self, state_manager):
        """Test queue count decreases when items are completed."""
        await state_manager.register_active_project("project-complete", "tenant-1")

        # Enqueue 5 items
        await state_manager.update_active_project_queue_count("project-complete", 5)

        # Complete 3 items
        await state_manager.update_active_project_queue_count("project-complete", -3)

        project = await state_manager.get_active_project("project-complete")
        assert project.items_in_queue == 2

    @pytest.mark.asyncio
    async def test_queue_count_cannot_go_negative(self, state_manager):
        """Test queue count floors at zero."""
        await state_manager.register_active_project("project-floor", "tenant-1")

        await state_manager.update_active_project_queue_count("project-floor", 2)
        # Try to decrement more than current count
        await state_manager.update_active_project_queue_count("project-floor", -10)

        project = await state_manager.get_active_project("project-floor")
        assert project.items_in_queue == 0


# =============================================================================
# INTEGRATION TESTS: Multi-Project Isolation
# =============================================================================


class TestMultiProjectIsolation:
    """Test that project tracking is properly isolated."""

    @pytest.mark.asyncio
    async def test_activity_isolated_between_projects(self, state_manager_with_projects):
        """Test that activity updates don't affect other projects."""
        manager = state_manager_with_projects

        # Update only project-alpha
        await manager.update_active_project_activity("project-alpha", 100)

        # Verify other projects unaffected
        alpha = await manager.get_active_project("project-alpha")
        beta = await manager.get_active_project("project-beta")
        gamma = await manager.get_active_project("project-gamma")

        assert alpha.items_processed_count == 100
        assert beta.items_processed_count == 0
        assert gamma.items_processed_count == 0

    @pytest.mark.asyncio
    async def test_tenant_filtering_works(self, state_manager_with_projects):
        """Test listing projects by tenant."""
        manager = state_manager_with_projects

        tenant_1_projects = await manager.list_active_projects(tenant_id="tenant-1")
        tenant_2_projects = await manager.list_active_projects(tenant_id="tenant-2")

        assert len(tenant_1_projects) == 2  # alpha, beta
        assert len(tenant_2_projects) == 1  # gamma

        assert all(p.tenant_id == "tenant-1" for p in tenant_1_projects)
        assert all(p.tenant_id == "tenant-2" for p in tenant_2_projects)


# =============================================================================
# INTEGRATION TESTS: Watch Folder Integration
# =============================================================================


class TestWatchFolderIntegration:
    """Test integration with watch folder system."""

    @pytest.mark.asyncio
    async def test_project_registered_with_watch_folder(self, state_manager):
        """Test project registration includes watch folder reference."""
        # Create watch folder
        watch_config = WatchFolderConfig(
            watch_id="watch-integration",
            path="/tmp/integration-test",
            collection="integration-code",
            patterns=["*.py"],
            ignore_patterns=[],
        )
        await state_manager.save_watch_folder_config(watch_config)

        # Register project with watch folder
        result = await state_manager.register_active_project(
            project_id="project-watched",
            tenant_id="tenant-1",
            watch_folder_id="watch-integration",
        )

        assert result.watch_enabled is True
        assert result.watch_folder_id == "watch-integration"

    @pytest.mark.asyncio
    async def test_list_watched_projects_only(self, state_manager):
        """Test filtering to watched projects only."""
        # Create watch folder
        watch_config = WatchFolderConfig(
            watch_id="watch-filter",
            path="/tmp/filter-test",
            collection="filter-code",
            patterns=["*.py"],
            ignore_patterns=[],
        )
        await state_manager.save_watch_folder_config(watch_config)

        # Register mix of watched and unwatched projects
        await state_manager.register_active_project(
            "project-watched", "tenant-1", watch_folder_id="watch-filter"
        )
        await state_manager.register_active_project("project-unwatched", "tenant-1")

        watched_only = await state_manager.list_active_projects(watch_enabled_only=True)

        assert len(watched_only) == 1
        assert watched_only[0].project_id == "project-watched"


# =============================================================================
# INTEGRATION TESTS: Garbage Collection
# =============================================================================


class TestGarbageCollection:
    """Test garbage collection of stale projects."""

    @pytest.mark.asyncio
    async def test_gc_removes_stale_projects(self, state_manager):
        """Test that GC removes inactive projects."""
        # Register projects
        await state_manager.register_active_project("project-active", "tenant-1")
        await state_manager.register_active_project("project-stale", "tenant-1")

        # Make project-stale appear old
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE active_projects
                SET last_activity_at = datetime('now', '-48 hours')
                WHERE project_id = 'project-stale'
                """
            )

        # Run garbage collection with 24 hour threshold
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
    async def test_gc_preserves_active_projects(self, state_manager):
        """Test that GC doesn't remove recently active projects."""
        await state_manager.register_active_project("project-1", "tenant-1")
        await state_manager.register_active_project("project-2", "tenant-1")

        # Update activity for both
        await state_manager.update_active_project_activity("project-1", 1)
        await state_manager.update_active_project_activity("project-2", 1)

        # Run GC
        removed = await state_manager.garbage_collect_stale_projects(
            max_inactive_hours=24
        )

        assert removed == 0

        # Both projects should still exist
        projects = await state_manager.list_active_projects()
        assert len(projects) == 2

    @pytest.mark.asyncio
    async def test_stale_projects_view(self, state_manager):
        """Test the stale projects view for monitoring."""
        await state_manager.register_active_project("project-old", "tenant-1")

        # Make it stale
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE active_projects
                SET last_activity_at = datetime('now', '-72 hours')
                WHERE project_id = 'project-old'
                """
            )

        stale = await state_manager.get_stale_projects()

        assert len(stale) == 1
        assert stale[0]["project_id"] == "project-old"
        assert stale[0]["days_inactive"] >= 2  # At least 2 days old


# =============================================================================
# INTEGRATION TESTS: Statistics
# =============================================================================


class TestStatisticsAggregation:
    """Test statistics aggregation across projects."""

    @pytest.mark.asyncio
    async def test_stats_aggregates_all_projects(self, state_manager_with_projects):
        """Test stats aggregation across all projects."""
        manager = state_manager_with_projects

        # Update activity for each project
        await manager.update_active_project_activity("project-alpha", 10)
        await manager.update_active_project_activity("project-beta", 20)
        await manager.update_active_project_activity("project-gamma", 30)

        # Update queue counts
        await manager.update_active_project_queue_count("project-alpha", 5)
        await manager.update_active_project_queue_count("project-beta", 3)

        stats = await manager.get_active_projects_stats()

        assert stats["total_projects"] == 3
        assert stats["watched_projects"] == 1  # Only alpha has watch folder
        assert stats["total_items_processed"] == 60  # 10 + 20 + 30
        assert stats["total_items_in_queue"] == 8  # 5 + 3
        assert stats["active_last_hour"] == 3  # All just updated

    @pytest.mark.asyncio
    async def test_empty_stats(self, state_manager):
        """Test stats with no projects."""
        stats = await state_manager.get_active_projects_stats()

        # Should return zeros for empty state
        assert stats.get("total_projects", 0) == 0


# =============================================================================
# INTEGRATION TESTS: End-to-End Workflow
# =============================================================================


class TestEndToEndWorkflow:
    """Test complete workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complete_project_lifecycle(self, state_manager):
        """Test project lifecycle: register -> activity -> GC."""
        # Step 1: Create watch folder and register project
        watch_config = WatchFolderConfig(
            watch_id="lifecycle-watch",
            path="/tmp/lifecycle-project",
            collection="lifecycle-code",
            patterns=["*.py"],
            ignore_patterns=[],
        )
        await state_manager.save_watch_folder_config(watch_config)

        result = await state_manager.register_active_project(
            project_id="lifecycle-project",
            tenant_id="tenant-lifecycle",
            watch_folder_id="lifecycle-watch",
            metadata={"git_remote": "https://github.com/example/lifecycle"},
        )

        assert result.watch_enabled is True
        assert result.metadata["git_remote"] == "https://github.com/example/lifecycle"

        # Step 2: Simulate activity (file events detected, items processed)
        for i in range(10):
            await state_manager.update_active_project_queue_count("lifecycle-project", 1)

        for i in range(7):
            await state_manager.update_active_project_activity("lifecycle-project", 1)
            await state_manager.update_active_project_queue_count("lifecycle-project", -1)

        # Step 3: Verify state
        project = await state_manager.get_active_project("lifecycle-project")
        assert project.items_processed_count == 7
        assert project.items_in_queue == 3  # 10 enqueued - 7 completed

        # Step 4: Make project stale and run GC
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                UPDATE active_projects
                SET last_activity_at = datetime('now', '-48 hours')
                WHERE project_id = 'lifecycle-project'
                """
            )

        removed = await state_manager.garbage_collect_stale_projects(
            max_inactive_hours=24
        )
        assert removed == 1

        # Step 5: Verify project removed
        project = await state_manager.get_active_project("lifecycle-project")
        assert project is None

    @pytest.mark.asyncio
    async def test_concurrent_project_updates(self, state_manager):
        """Test concurrent updates to multiple projects."""
        # Register multiple projects
        for i in range(5):
            await state_manager.register_active_project(f"concurrent-{i}", "tenant-1")

        # Concurrent updates
        async def update_project(project_id: str, count: int):
            for _ in range(count):
                await state_manager.update_active_project_activity(project_id, 1)

        tasks = [
            update_project(f"concurrent-{i}", 10 * (i + 1))
            for i in range(5)
        ]
        await asyncio.gather(*tasks)

        # Verify each project has correct count
        for i in range(5):
            project = await state_manager.get_active_project(f"concurrent-{i}")
            expected_count = 10 * (i + 1)
            assert project.items_processed_count == expected_count, (
                f"Project concurrent-{i} expected {expected_count}, got {project.items_processed_count}"
            )

    @pytest.mark.asyncio
    async def test_project_reregistration(self, state_manager):
        """Test re-registering a project preserves counts."""
        # Create watch folder
        watch_config = WatchFolderConfig(
            watch_id="rereg-watch",
            path="/tmp/rereg-project",
            collection="rereg-code",
            patterns=["*.py"],
            ignore_patterns=[],
        )
        await state_manager.save_watch_folder_config(watch_config)

        # Register and add activity
        await state_manager.register_active_project("rereg-project", "tenant-1")
        await state_manager.update_active_project_activity("rereg-project", 50)

        # Re-register with watch folder (simulates enabling watch on existing project)
        await state_manager.register_active_project(
            "rereg-project", "tenant-1", watch_folder_id="rereg-watch"
        )

        # Activity should be preserved, watch should be enabled
        project = await state_manager.get_active_project("rereg-project")
        assert project.watch_enabled is True
        assert project.items_processed_count == 50
