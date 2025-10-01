#!/usr/bin/env python3
"""
Temporary test file for branch-aware metadata tracker methods.

Tests the new methods added in subtask 350.7:
- get_tracked_files_by_branch()
- get_branches_with_tracked_files()
- update_file_branch()
- Enhanced get_tracked_file_count() with by_branch stats
- Enhanced requeue_when_tools_available() with branch parameter
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from src.python.common.core.missing_metadata_tracker import MissingMetadataTracker
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_tracker_branch.db"
        yield str(db_path)


@pytest.fixture
async def state_manager(temp_db):
    """Create and initialize a state manager."""
    manager = SQLiteStateManager(db_path=temp_db)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def tracker(state_manager):
    """Create a missing metadata tracker."""
    return MissingMetadataTracker(state_manager)


class TestBranchAwareMethods:
    """Test new branch-aware functionality."""

    @pytest.mark.asyncio
    async def test_get_tracked_files_by_branch(self, tracker):
        """Test retrieving files by specific branch."""
        # Track files on different branches
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "develop", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file4.rs", "rust", "feature/new", missing_lsp=True
        )

        # Get files for main branch
        main_files = await tracker.get_tracked_files_by_branch("main")
        assert len(main_files) == 2
        assert all(f["branch"] == "main" for f in main_files)
        assert all(f["language_name"] == "python" for f in main_files)

        # Get files for develop branch
        dev_files = await tracker.get_tracked_files_by_branch("develop")
        assert len(dev_files) == 1
        assert dev_files[0]["branch"] == "develop"

        # Get files for feature branch
        feature_files = await tracker.get_tracked_files_by_branch("feature/new")
        assert len(feature_files) == 1
        assert feature_files[0]["branch"] == "feature/new"
        assert feature_files[0]["language_name"] == "rust"

        # Get files for non-existent branch
        no_files = await tracker.get_tracked_files_by_branch("nonexistent")
        assert len(no_files) == 0

    @pytest.mark.asyncio
    async def test_get_branches_with_tracked_files(self, tracker):
        """Test getting list of all branches with tracked files."""
        # Initially no branches
        branches = await tracker.get_branches_with_tracked_files()
        assert len(branches) == 0

        # Add files on different branches
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "develop", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "feature/a", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file4.py", "python", "main", missing_lsp=True
        )

        # Get all branches
        branches = await tracker.get_branches_with_tracked_files()
        assert len(branches) == 3
        assert "main" in branches
        assert "develop" in branches
        assert "feature/a" in branches
        # Should be sorted alphabetically
        assert branches == sorted(branches)

    @pytest.mark.asyncio
    async def test_get_branches_excludes_empty_branches(self, tracker):
        """Test that empty or NULL branches are excluded."""
        # Add files with various branch values
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "", missing_lsp=True
        )

        # Get branches
        branches = await tracker.get_branches_with_tracked_files()
        assert len(branches) == 1
        assert branches[0] == "main"
        assert "" not in branches

    @pytest.mark.asyncio
    async def test_update_file_branch(self, tracker):
        """Test updating branch for a tracked file."""
        # Track a file
        file_path = "/test/file.py"
        await tracker.track_missing_metadata(
            file_path, "python", "main", missing_lsp=True
        )

        # Verify initial branch
        files = await tracker.get_tracked_files_by_branch("main")
        assert len(files) == 1

        # Update to develop branch
        updated = await tracker.update_file_branch(file_path, "develop")
        assert updated is True

        # Verify branch changed
        main_files = await tracker.get_tracked_files_by_branch("main")
        assert len(main_files) == 0

        dev_files = await tracker.get_tracked_files_by_branch("develop")
        assert len(dev_files) == 1
        assert dev_files[0]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_update_file_branch_nonexistent(self, tracker):
        """Test updating branch for non-existent file."""
        updated = await tracker.update_file_branch("/nonexistent/file.py", "main")
        assert updated is False

    @pytest.mark.asyncio
    async def test_get_tracked_file_count_with_branch_stats(self, tracker):
        """Test enhanced statistics with branch grouping."""
        # Add files on multiple branches
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "develop", missing_lsp=False, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file4.rs", "rust", "feature/x", missing_lsp=True, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file5.rs", "rust", "feature/x", missing_lsp=True, missing_ts=False
        )

        # Get stats
        stats = await tracker.get_tracked_file_count()

        # Check overall stats
        assert stats["total"] == 5
        assert stats["missing_lsp"] == 4  # file1, file2, file4, file5
        assert stats["missing_ts"] == 2  # file3, file4
        assert stats["missing_both"] == 1  # file4

        # Check branch breakdown
        assert "by_branch" in stats
        assert stats["by_branch"]["main"] == 2
        assert stats["by_branch"]["develop"] == 1
        assert stats["by_branch"]["feature/x"] == 2

        # Verify sorted by count descending, then branch name
        branch_counts = list(stats["by_branch"].items())
        assert branch_counts[0][0] in ["main", "feature/x"]  # Both have 2 files
        assert branch_counts[2][0] == "develop"  # Has 1 file

    @pytest.mark.asyncio
    async def test_get_tracked_file_count_empty_branch_stats(self, tracker):
        """Test branch stats when no files tracked."""
        stats = await tracker.get_tracked_file_count()

        assert stats["total"] == 0
        assert stats["by_branch"] == {}

    @pytest.mark.asyncio
    async def test_requeue_with_branch_filter(self, tracker, state_manager):
        """Test requeuing files with branch filtering."""
        # Setup: Add language with available LSP
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, lsp_name, lsp_absolute_path, lsp_missing)
                VALUES ('python', 'pyright', '/usr/bin/pyright', 0)
                """
            )

        # Track files on different branches
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "develop", missing_lsp=True
        )

        # Requeue only main branch files
        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="python",
            branch="main",
            priority=5
        )

        assert result["tool_available"] is True
        assert result["branch"] == "main"
        assert result["files_requeued"] == 2
        assert result["files_removed"] == 2

        # Verify only develop branch file remains
        remaining = await tracker.get_files_missing_metadata(language="python")
        assert len(remaining) == 1
        assert remaining[0]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_requeue_all_branches(self, tracker, state_manager):
        """Test requeuing files without branch filter (all branches)."""
        # Setup: Add language with available LSP
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, lsp_name, lsp_absolute_path, lsp_missing)
                VALUES ('python', 'pyright', '/usr/bin/pyright', 0)
                """
            )

        # Track files on different branches
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "develop", missing_lsp=True
        )

        # Requeue all branches (no branch filter)
        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="python",
            branch=None,
            priority=5
        )

        assert result["tool_available"] is True
        assert result["branch"] is None
        assert result["files_requeued"] == 2
        assert result["files_removed"] == 2

        # Verify all files removed
        remaining = await tracker.get_files_missing_metadata(language="python")
        assert len(remaining) == 0

    @pytest.mark.asyncio
    async def test_multi_branch_workflow(self, tracker):
        """Test complete multi-branch workflow scenario."""
        # Step 1: Track files on multiple branches
        files = [
            ("/project/src/main.py", "python", "main"),
            ("/project/src/utils.py", "python", "main"),
            ("/project/src/new_feature.py", "python", "feature/api"),
            ("/project/src/test.rs", "rust", "feature/api"),
        ]

        for file_path, lang, branch in files:
            await tracker.track_missing_metadata(
                file_path, lang, branch, missing_lsp=True
            )

        # Step 2: Verify branch listing
        branches = await tracker.get_branches_with_tracked_files()
        assert len(branches) == 2
        assert "main" in branches
        assert "feature/api" in branches

        # Step 3: Check stats by branch
        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 4
        assert stats["by_branch"]["main"] == 2
        assert stats["by_branch"]["feature/api"] == 2

        # Step 4: User switches branch, update file tracking
        updated = await tracker.update_file_branch(
            "/project/src/new_feature.py", "main"
        )
        assert updated is True

        # Step 5: Verify updated branch distribution
        main_files = await tracker.get_tracked_files_by_branch("main")
        assert len(main_files) == 3

        feature_files = await tracker.get_tracked_files_by_branch("feature/api")
        assert len(feature_files) == 1

        # Step 6: Remove one file
        removed = await tracker.remove_tracked_file("/project/src/test.rs")
        assert removed is True

        # Step 7: Final verification
        branches = await tracker.get_branches_with_tracked_files()
        assert len(branches) == 1
        assert branches[0] == "main"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
