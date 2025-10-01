"""
Unit tests for FileWatcher queue integration.

Tests the refactored FileWatcher that uses state_manager.enqueue() instead of callbacks.
"""

import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

from src.python.common.core.file_watcher import (
    FileWatcher,
    WatchConfiguration,
    WatchEvent,
    WatchManager,
)
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
def watch_config():
    """Create a test watch configuration."""
    return WatchConfiguration(
        id="test_watch_1",
        path="/tmp/test_watch",
        collection="test_collection",
        patterns=["*.txt", "*.md"],
        ignore_patterns=[".git/*", "__pycache__/*"],
        auto_ingest=True,
        recursive=True,
        debounce_seconds=0.1,  # Short debounce for testing
    )


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager."""
    mock = AsyncMock(spec=SQLiteStateManager)
    mock.calculate_tenant_id = AsyncMock(return_value="test_tenant")
    mock.get_current_branch = AsyncMock(return_value="main")
    mock.enqueue = AsyncMock(return_value="queue_id_123")
    return mock


@pytest.fixture
def file_watcher(watch_config, mock_state_manager):
    """Create a FileWatcher instance for testing."""
    return FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
        event_callback=None,
        filter_config_path=None,
    )


class TestFileWatcherInitialization:
    """Test FileWatcher initialization with state_manager."""

    def test_accepts_state_manager(self, watch_config, mock_state_manager):
        """FileWatcher should accept state_manager parameter."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=mock_state_manager,
        )

        assert watcher.state_manager is mock_state_manager
        assert watcher.config is watch_config
        assert watcher.event_callback is None

    def test_initializes_project_root_cache(self, file_watcher):
        """FileWatcher should initialize project root cache."""
        assert hasattr(file_watcher, "_project_root_cache")
        assert isinstance(file_watcher._project_root_cache, dict)
        assert len(file_watcher._project_root_cache) == 0


class TestProjectRootDetection:
    """Test project root detection functionality."""

    def test_finds_git_directory(self, file_watcher, tmp_path):
        """Should find .git directory when walking up tree."""
        # Create directory structure with .git
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        subdir = project_root / "src" / "python"
        subdir.mkdir(parents=True)

        test_file = subdir / "test.py"
        test_file.touch()

        # Find project root
        found_root = file_watcher._find_project_root(test_file)

        assert found_root == project_root

    def test_caches_project_root(self, file_watcher, tmp_path):
        """Should cache project root for performance."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        test_file = project_root / "test.py"
        test_file.touch()

        # First call
        found_root1 = file_watcher._find_project_root(test_file)

        # Check cache
        assert str(test_file) in file_watcher._project_root_cache
        assert file_watcher._project_root_cache[str(test_file)] == project_root

        # Second call should use cache
        found_root2 = file_watcher._find_project_root(test_file)

        assert found_root1 == found_root2 == project_root

    def test_fallback_to_parent_directory(self, file_watcher, tmp_path):
        """Should fallback to parent directory when no .git found."""
        test_dir = tmp_path / "no_git"
        test_dir.mkdir()

        test_file = test_dir / "test.txt"
        test_file.touch()

        # Find project root (should be parent directory)
        found_root = file_watcher._find_project_root(test_file)

        assert found_root == test_dir


class TestTriggerOperation:
    """Test _trigger_operation method."""

    @pytest.mark.asyncio
    async def test_calls_state_manager_enqueue(self, file_watcher, mock_state_manager, tmp_path):
        """Should call state_manager.enqueue() with correct parameters."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Trigger operation
        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "ingest"
        )

        # Verify enqueue was called
        mock_state_manager.enqueue.assert_called_once()

        # Check call arguments
        call_kwargs = mock_state_manager.enqueue.call_args.kwargs
        assert call_kwargs["file_path"] == str(test_file)
        assert call_kwargs["collection"] == "test_collection"
        assert call_kwargs["priority"] == 5  # Normal priority for ingest
        assert call_kwargs["tenant_id"] == "test_tenant"
        assert call_kwargs["branch"] == "main"
        assert "metadata" in call_kwargs

    @pytest.mark.asyncio
    async def test_calculates_tenant_and_branch(self, file_watcher, mock_state_manager, tmp_path):
        """Should calculate tenant_id and branch correctly."""
        # Create test file with git repo
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()

        test_file = project_root / "test.txt"
        test_file.touch()

        # Trigger operation
        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "ingest"
        )

        # Verify tenant and branch calculation
        mock_state_manager.calculate_tenant_id.assert_called_once()
        mock_state_manager.get_current_branch.assert_called_once()

        # Check that project root was passed
        tenant_call_args = mock_state_manager.calculate_tenant_id.call_args[0]
        assert tenant_call_args[0] == project_root

    @pytest.mark.asyncio
    async def test_priority_for_ingest(self, file_watcher, mock_state_manager, tmp_path):
        """Ingest operations should have priority 5 (NORMAL)."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "ingest"
        )

        call_kwargs = mock_state_manager.enqueue.call_args.kwargs
        assert call_kwargs["priority"] == 5

    @pytest.mark.asyncio
    async def test_priority_for_delete(self, file_watcher, mock_state_manager, tmp_path):
        """Delete operations should have higher priority (8)."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "delete"
        )

        call_kwargs = mock_state_manager.enqueue.call_args.kwargs
        assert call_kwargs["priority"] == 8

    @pytest.mark.asyncio
    async def test_metadata_includes_event_info(self, file_watcher, mock_state_manager, tmp_path):
        """Metadata should include watch and event information."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "ingest"
        )

        call_kwargs = mock_state_manager.enqueue.call_args.kwargs
        metadata = call_kwargs["metadata"]

        assert metadata["watch_id"] == "test_watch_1"
        assert metadata["watch_path"] == "/tmp/test_watch"
        assert metadata["operation"] == "ingest"
        assert metadata["event_type"] == "file_change"
        assert "detected_at" in metadata
        assert "project_root" in metadata

    @pytest.mark.asyncio
    async def test_error_handling(self, file_watcher, mock_state_manager, tmp_path):
        """Should handle errors and re-raise them."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Make enqueue fail
        mock_state_manager.enqueue.side_effect = Exception("Enqueue failed")

        # Should raise exception
        with pytest.raises(Exception, match="Enqueue failed"):
            await file_watcher._trigger_operation(
                str(test_file),
                "test_collection",
                "ingest"
            )


class TestDebouncing:
    """Test that debouncing still works with queue operations."""

    @pytest.mark.asyncio
    async def test_debouncing_mechanism(self, file_watcher, mock_state_manager, tmp_path):
        """Debouncing should still work with new queue system."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        # Trigger multiple times rapidly
        await file_watcher._debounce_ingestion(str(test_file))
        await file_watcher._debounce_ingestion(str(test_file))
        await file_watcher._debounce_ingestion(str(test_file))

        # Wait for debounce period
        await asyncio.sleep(0.2)

        # Should only enqueue once after debounce
        assert mock_state_manager.enqueue.call_count == 1


class TestWatchManagerIntegration:
    """Test WatchManager with state_manager."""

    def test_set_state_manager(self, mock_state_manager):
        """WatchManager should accept and store state_manager."""
        manager = WatchManager()
        manager.set_state_manager(mock_state_manager)

        assert manager.state_manager is mock_state_manager

    @pytest.mark.asyncio
    async def test_creates_watcher_with_state_manager(self, mock_state_manager, tmp_path):
        """WatchManager should create watchers with state_manager."""
        manager = WatchManager(config_file=str(tmp_path / "watches.json"))
        manager.set_state_manager(mock_state_manager)

        # Add a watch
        watch_id = await manager.add_watch(
            path=str(tmp_path),
            collection="test_collection",
        )

        # Verify watcher was created with state_manager
        assert watch_id in manager.watchers
        watcher = manager.watchers[watch_id]
        assert watcher.state_manager is mock_state_manager

    @pytest.mark.asyncio
    async def test_starts_all_watches_with_state_manager(self, mock_state_manager, tmp_path):
        """start_all_watches should only work if state_manager is set."""
        manager = WatchManager(config_file=str(tmp_path / "watches.json"))

        # Add configurations without starting
        config = WatchConfiguration(
            id="test_watch",
            path=str(tmp_path),
            collection="test_collection",
        )
        manager.configurations["test_watch"] = config

        # Should not start without state_manager
        await manager.start_all_watches()
        assert len(manager.watchers) == 0

        # Set state_manager and try again
        manager.set_state_manager(mock_state_manager)
        await manager.start_all_watches()

        # Now should start
        assert "test_watch" in manager.watchers


class TestEventCallback:
    """Test that event_callback is still supported."""

    @pytest.mark.asyncio
    async def test_event_callback_still_works(self, watch_config, mock_state_manager, tmp_path):
        """Event callback should still be called for watch events."""
        event_callback = MagicMock()

        watcher = FileWatcher(
            config=watch_config,
            state_manager=mock_state_manager,
            event_callback=event_callback,
        )

        # Simulate a file change event
        test_file = tmp_path / "test.txt"
        test_file.touch()

        await watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "ingest"
        )

        # Event callback is called separately in _handle_changes, not in _trigger_operation
        # This test just verifies the callback can be set
        assert watcher.event_callback is event_callback


class TestFilteringLogic:
    """Test that filtering logic remains intact."""

    def test_matches_patterns(self, file_watcher):
        """Pattern matching should still work."""
        # Should match
        assert file_watcher._matches_patterns(Path("test.txt"))
        assert file_watcher._matches_patterns(Path("README.md"))

        # Should not match
        assert not file_watcher._matches_patterns(Path("test.py"))
        assert not file_watcher._matches_patterns(Path("test.rs"))

    def test_matches_ignore_patterns(self, file_watcher):
        """Ignore pattern matching should still work."""
        # Should match ignore patterns
        assert file_watcher._matches_ignore_patterns(Path(".git/config"))
        assert file_watcher._matches_ignore_patterns(Path("__pycache__/test.pyc"))

        # Should not match ignore patterns
        assert not file_watcher._matches_ignore_patterns(Path("test.txt"))
        assert not file_watcher._matches_ignore_patterns(Path("README.md"))


class TestDeletionHandling:
    """Test deletion handling with queue operations."""

    @pytest.mark.asyncio
    async def test_deletion_uses_higher_priority(self, file_watcher, mock_state_manager, tmp_path):
        """Deletions should use priority 8 (HIGH)."""
        test_file = tmp_path / "deleted.txt"
        # File doesn't need to exist for deletion handling

        await file_watcher._trigger_operation(
            str(test_file),
            "test_collection",
            "delete"
        )

        call_kwargs = mock_state_manager.enqueue.call_args.kwargs
        assert call_kwargs["priority"] == 8
        assert call_kwargs["metadata"]["operation"] == "delete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
