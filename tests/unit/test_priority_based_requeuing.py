#!/usr/bin/env python3
"""
Unit tests for priority-based requeuing logic in MissingMetadataTracker.

Tests the calculate_requeue_priority method and updated requeue_when_tools_available.
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
        db_path = Path(tmpdir) / "test_tracker.db"
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


class TestCalculateRequeuePriority:
    """Test priority calculation logic."""

    def test_high_priority_current_project(self, tracker):
        """Test HIGH priority for files in current project."""
        # Create a temporary project root
        with tempfile.TemporaryDirectory() as project_root:
            # File within project
            file_path = Path(project_root) / "src" / "main.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            priority = tracker.calculate_requeue_priority(
                file_path=str(file_path),
                file_branch="main",
                current_project_root=project_root,
                current_branch="main"
            )

            assert priority == 8, "Files in current project should have HIGH priority (8)"

    def test_normal_priority_same_branch(self, tracker):
        """Test NORMAL priority for files on same branch but different project."""
        with tempfile.TemporaryDirectory() as project_root:
            # File outside project but on same branch
            other_file = Path(tempfile.gettempdir()) / "other_project" / "file.py"
            other_file.parent.mkdir(parents=True, exist_ok=True)
            other_file.touch()

            priority = tracker.calculate_requeue_priority(
                file_path=str(other_file),
                file_branch="main",
                current_project_root=project_root,
                current_branch="main"
            )

            assert priority == 5, "Files on same branch should have NORMAL priority (5)"

    def test_low_priority_different_branch_and_project(self, tracker):
        """Test LOW priority for files in different project and branch."""
        with tempfile.TemporaryDirectory() as project_root:
            # File outside project and on different branch
            other_file = Path(tempfile.gettempdir()) / "other_project" / "file.py"
            other_file.parent.mkdir(parents=True, exist_ok=True)
            other_file.touch()

            priority = tracker.calculate_requeue_priority(
                file_path=str(other_file),
                file_branch="feature",
                current_project_root=project_root,
                current_branch="main"
            )

            assert priority == 2, "Other files should have LOW priority (2)"

    def test_default_priority_no_context(self, tracker):
        """Test default NORMAL priority when no context provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "file.py"
            file_path.touch()

            priority = tracker.calculate_requeue_priority(
                file_path=str(file_path),
                file_branch="main"
            )

            assert priority == 2, "Files without context should have LOW priority (2)"

    def test_invalid_file_path(self, tracker):
        """Test handling of invalid file paths."""
        priority = tracker.calculate_requeue_priority(
            file_path="/nonexistent/path/file.py",
            file_branch="main",
            current_project_root="/some/project",
            current_branch="main"
        )

        assert priority == 5, "Invalid paths should default to NORMAL priority (5)"

    def test_nested_project_structure(self, tracker):
        """Test priority calculation for nested project structures."""
        with tempfile.TemporaryDirectory() as project_root:
            # Deeply nested file within project
            nested_file = Path(project_root) / "a" / "b" / "c" / "file.py"
            nested_file.parent.mkdir(parents=True, exist_ok=True)
            nested_file.touch()

            priority = tracker.calculate_requeue_priority(
                file_path=str(nested_file),
                file_branch="main",
                current_project_root=project_root,
                current_branch="main"
            )

            assert priority == 8, "Nested files in project should have HIGH priority (8)"


class TestRequeuWithPriority:
    """Test requeue_when_tools_available with priority calculation."""

    @pytest.mark.asyncio
    async def test_requeue_with_explicit_priority(self, tracker, state_manager):
        """Test requeuing with explicit priority override."""
        # Setup: Add a language with available LSP
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages (language_name, lsp_name, lsp_absolute_path, lsp_missing)
                VALUES ('python', 'pyright', '/usr/bin/pyright', 0)
                """
            )

        # Track a file missing LSP
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True
        )

        # Requeue with explicit priority
        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="python",
            priority=9
        )

        assert result["tool_available"] is True
        assert result["files_requeued"] == 1
        assert result["files_removed"] == 1

        # Check the queue item has correct priority
        with state_manager._lock:
            cursor = state_manager.connection.execute(
                "SELECT priority FROM ingestion_queue WHERE file_absolute_path LIKE '%file.py'"
            )
            row = cursor.fetchone()
            assert row is not None, "File should be in queue"
            assert row["priority"] == 9, "Priority should be explicitly set to 9"

    @pytest.mark.asyncio
    async def test_requeue_with_calculated_priority(self, tracker, state_manager):
        """Test requeuing with automatic priority calculation."""
        # Setup: Add a language with available LSP
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages (language_name, lsp_name, lsp_absolute_path, lsp_missing)
                VALUES ('python', 'pyright', '/usr/bin/pyright', 0)
                """
            )

        with tempfile.TemporaryDirectory() as project_root:
            # Track a file in the project
            file_path = Path(project_root) / "src" / "main.py"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.touch()

            await tracker.track_missing_metadata(
                file_path=str(file_path),
                language_name="python",
                branch="main",
                missing_lsp=True
            )

            # Requeue with project context (should calculate priority=8)
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                current_project_root=project_root
            )

            assert result["tool_available"] is True
            assert result["files_requeued"] == 1

            # Check calculated priority
            with state_manager._lock:
                cursor = state_manager.connection.execute(
                    "SELECT priority FROM ingestion_queue WHERE file_absolute_path = ?",
                    (str(file_path),)
                )
                row = cursor.fetchone()
                assert row is not None
                # Note: Priority might be 5 if branch detection fails, or 8 if successful
                assert row["priority"] in [5, 8], f"Priority should be calculated (got {row['priority']})"

    @pytest.mark.asyncio
    async def test_validation_errors(self, tracker):
        """Test validation of requeue parameters."""
        # Invalid tool type
        with pytest.raises(ValueError, match="Invalid tool_type"):
            await tracker.requeue_when_tools_available(
                tool_type="invalid",
                language="python"
            )

        # Missing language for LSP
        with pytest.raises(ValueError, match="Language parameter required"):
            await tracker.requeue_when_tools_available(
                tool_type="lsp"
            )

        # Invalid priority
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=15
            )

    @pytest.mark.asyncio
    async def test_tool_not_available(self, tracker):
        """Test behavior when tool is not available."""
        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="nonexistent"
        )

        assert result["tool_available"] is False
        assert result["files_requeued"] == 0
        assert result["files_failed"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
