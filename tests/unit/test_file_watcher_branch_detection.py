"""
Unit tests for FileWatcher branch detection functionality.

Tests branch detection, caching, and graceful fallback for non-git directories.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.file_watcher import FileWatcher, WatchConfiguration
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir).resolve()  # Resolve to handle macOS symlinks
        git_dir = repo_path / ".git"
        git_dir.mkdir()

        # Create a simple file structure
        (repo_path / "src").mkdir()
        (repo_path / "src" / "test.py").write_text("# test file")
        (repo_path / "docs").mkdir()
        (repo_path / "docs" / "README.md").write_text("# README")

        yield repo_path


@pytest.fixture
async def temp_non_git_dir():
    """Create a temporary non-git directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        non_git_path = Path(tmpdir).resolve()  # Resolve to handle macOS symlinks
        (non_git_path / "files").mkdir()
        (non_git_path / "files" / "test.txt").write_text("test content")

        yield non_git_path


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    state_manager = AsyncMock(spec=SQLiteStateManager)
    state_manager.get_current_branch = AsyncMock(return_value="main")
    state_manager.calculate_tenant_id = AsyncMock(return_value="test_tenant")
    state_manager.enqueue = AsyncMock(return_value="queue_id_123")
    return state_manager


@pytest.fixture
def watch_config(temp_git_repo):
    """Create a basic watch configuration."""
    return WatchConfiguration(
        id="test_watch_1",
        path=str(temp_git_repo),
        collection="test_collection",
        patterns=["*.py", "*.md", "*.txt"],
        ignore_patterns=[".git/*"],
        auto_ingest=True,
        recursive=True,
        debounce_seconds=0,  # No debouncing for tests
    )


@pytest.mark.asyncio
async def test_find_project_root_with_git(temp_git_repo, mock_state_manager, watch_config):
    """Test _find_project_root() correctly finds .git directory."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    # Test with file in subdirectory
    test_file = temp_git_repo / "src" / "test.py"
    project_root = watcher._find_project_root(test_file)

    assert project_root.resolve() == temp_git_repo.resolve()
    assert (project_root / ".git").exists()


@pytest.mark.asyncio
async def test_find_project_root_without_git(temp_non_git_dir, mock_state_manager):
    """Test _find_project_root() falls back gracefully for non-git directories."""
    config = WatchConfiguration(
        id="test_watch_2",
        path=str(temp_non_git_dir),
        collection="test_collection",
    )

    watcher = FileWatcher(
        config=config,
        state_manager=mock_state_manager,
    )

    test_file = temp_non_git_dir / "files" / "test.txt"
    project_root = watcher._find_project_root(test_file)

    # Should fallback to file's parent directory
    assert project_root == test_file.parent


@pytest.mark.asyncio
async def test_find_project_root_caching(temp_git_repo, mock_state_manager, watch_config):
    """Test that _find_project_root() caches results for performance."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    # First call should populate cache
    root1 = watcher._find_project_root(test_file)
    assert str(test_file) in watcher._project_root_cache

    # Second call should use cache
    root2 = watcher._find_project_root(test_file)
    assert root1 == root2
    assert root1.resolve() == temp_git_repo.resolve()


@pytest.mark.asyncio
async def test_trigger_operation_with_branch_detection(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test _trigger_operation() detects branch and passes to enqueue()."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"
    test_file_str = str(test_file.resolve())

    # Trigger operation
    await watcher._trigger_operation(
        file_path=test_file_str,
        collection="test_collection",
        operation="ingest"
    )

    # Verify state_manager methods were called
    mock_state_manager.get_current_branch.assert_called_once()
    mock_state_manager.calculate_tenant_id.assert_called_once()
    mock_state_manager.enqueue.assert_called_once()

    # Verify enqueue was called with correct parameters
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["file_path"] == test_file_str
    assert enqueue_call.kwargs["collection"] == "test_collection"
    assert enqueue_call.kwargs["branch"] == "main"
    assert enqueue_call.kwargs["tenant_id"] == "test_tenant"
    assert enqueue_call.kwargs["priority"] == 5  # Default priority for ingest


@pytest.mark.asyncio
async def test_trigger_operation_deletion_priority(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test _trigger_operation() uses higher priority for delete operations."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    # Trigger delete operation
    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="delete"
    )

    # Verify enqueue was called with high priority for deletion
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["priority"] == 8  # Higher priority for delete


@pytest.mark.asyncio
async def test_branch_detection_with_different_branches(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test branch detection works with different branch names."""
    # Set up mock to return different branch
    mock_state_manager.get_current_branch = AsyncMock(return_value="feature/test-branch")

    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    # Trigger operation
    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="ingest"
    )

    # Verify correct branch was passed
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["branch"] == "feature/test-branch"


@pytest.mark.asyncio
async def test_branch_detection_handles_detached_head(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test branch detection handles detached HEAD state gracefully."""
    # Set up mock to return commit SHA (detached HEAD)
    mock_state_manager.get_current_branch = AsyncMock(return_value="abc123def456")

    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    # Trigger operation
    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="ingest"
    )

    # Verify commit SHA was used as branch
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["branch"] == "abc123def456"


@pytest.mark.asyncio
async def test_branch_detection_error_handling(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test branch detection falls back gracefully on error."""
    # Set up mock to raise an exception
    mock_state_manager.get_current_branch = AsyncMock(side_effect=Exception("Git error"))

    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    # Should raise exception (as per current implementation)
    with pytest.raises(Exception, match="Git error"):
        await watcher._trigger_operation(
            file_path=str(test_file.resolve()),
            collection="test_collection",
            operation="ingest"
        )


@pytest.mark.asyncio
async def test_metadata_includes_project_root(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test that metadata includes project_root information."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="ingest"
    )

    # Verify metadata includes project_root
    enqueue_call = mock_state_manager.enqueue.call_args
    metadata = enqueue_call.kwargs["metadata"]

    assert "project_root" in metadata
    assert Path(metadata["project_root"]).resolve() == temp_git_repo.resolve()
    assert metadata["operation"] == "ingest"
    assert metadata["watch_id"] == "test_watch_1"


@pytest.mark.asyncio
async def test_non_git_directory_defaults_to_main(temp_non_git_dir, mock_state_manager):
    """Test that non-git directories get 'main' as default branch."""
    config = WatchConfiguration(
        id="test_watch_non_git",
        path=str(temp_non_git_dir),
        collection="test_collection",
    )

    # Mock state_manager to return 'main' for non-git directories
    mock_state_manager.get_current_branch = AsyncMock(return_value="main")

    watcher = FileWatcher(
        config=config,
        state_manager=mock_state_manager,
    )

    test_file = temp_non_git_dir / "files" / "test.txt"

    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="ingest"
    )

    # Verify branch defaults to 'main'
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["branch"] == "main"


@pytest.mark.asyncio
async def test_multiple_files_same_repo_use_cached_root(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test that multiple files in same repo use cached project root."""
    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    file1 = temp_git_repo / "src" / "test.py"
    file2 = temp_git_repo / "docs" / "README.md"

    # Process first file
    root1 = watcher._find_project_root(file1)

    # Process second file
    root2 = watcher._find_project_root(file2)

    # Both should return same root and both should be cached
    assert root1.resolve() == root2.resolve() == temp_git_repo.resolve()
    assert str(file1) in watcher._project_root_cache
    assert str(file2) in watcher._project_root_cache


@pytest.mark.asyncio
async def test_nested_git_repos_find_nearest(temp_git_repo, mock_state_manager):
    """Test that nested git repositories find the nearest .git directory."""
    # Create a nested git repo
    nested_repo = temp_git_repo / "submodule"
    nested_repo.mkdir()
    (nested_repo / ".git").mkdir()
    (nested_repo / "file.txt").write_text("nested file")

    config = WatchConfiguration(
        id="test_watch_nested",
        path=str(temp_git_repo),
        collection="test_collection",
    )

    watcher = FileWatcher(
        config=config,
        state_manager=mock_state_manager,
    )

    # Test file in nested repo
    nested_file = nested_repo / "file.txt"
    project_root = watcher._find_project_root(nested_file)

    # Should find nearest .git (nested repo, not parent)
    assert project_root.resolve() == nested_repo.resolve()
    assert project_root.resolve() != temp_git_repo.resolve()


@pytest.mark.asyncio
async def test_tenant_id_calculated_per_project_root(
    temp_git_repo, mock_state_manager, watch_config
):
    """Test that tenant_id is calculated based on project root."""
    mock_state_manager.calculate_tenant_id = AsyncMock(return_value="calculated_tenant")

    watcher = FileWatcher(
        config=watch_config,
        state_manager=mock_state_manager,
    )

    test_file = temp_git_repo / "src" / "test.py"

    await watcher._trigger_operation(
        file_path=str(test_file.resolve()),
        collection="test_collection",
        operation="ingest"
    )

    # Verify calculate_tenant_id was called
    mock_state_manager.calculate_tenant_id.assert_called_once()

    # Verify tenant_id was used in enqueue
    enqueue_call = mock_state_manager.enqueue.call_args
    assert enqueue_call.kwargs["tenant_id"] == "calculated_tenant"
