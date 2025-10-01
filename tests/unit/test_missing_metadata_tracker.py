#!/usr/bin/env python3
"""
Unit tests for MissingMetadataTracker.

Tests the core functionality of tracking files with missing LSP or Tree-sitter
metadata, including CRUD operations, filtering, statistics collection, and
tool availability detection.
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


@pytest.fixture
async def state_manager_with_languages(state_manager):
    """Create state manager with sample language data."""
    async with state_manager.transaction() as conn:
        # Insert sample languages with various tool configurations
        conn.execute(
            """
            INSERT INTO languages
            (language_name, lsp_name, lsp_absolute_path, lsp_missing,
             ts_grammar, ts_cli_absolute_path, ts_missing)
            VALUES
            ('python', 'pyright-langserver', '/usr/bin/pyright', 0,
             'tree-sitter-python', '/usr/bin/tree-sitter', 0),
            ('rust', 'rust-analyzer', '/usr/bin/rust-analyzer', 0,
             'tree-sitter-rust', '/usr/bin/tree-sitter', 0),
            ('javascript', 'typescript-language-server', NULL, 1,
             'tree-sitter-javascript', '/usr/bin/tree-sitter', 0),
            ('go', 'gopls', '/usr/bin/gopls', 0,
             NULL, NULL, 0),
            ('cpp', 'clangd', NULL, 1,
             'tree-sitter-cpp', NULL, 1),
            ('java', NULL, NULL, 0,
             NULL, NULL, 0)
            """
        )
    yield state_manager


class TestMissingMetadataTrackerBasics:
    """Test basic functionality of MissingMetadataTracker."""

    @pytest.mark.asyncio
    async def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker is not None
        assert tracker.state_manager is not None

    @pytest.mark.asyncio
    async def test_track_missing_lsp_metadata(self, tracker):
        """Test tracking a file with missing LSP metadata."""
        success = await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False,
        )

        assert success is True

        # Verify the file was tracked
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["file_absolute_path"].endswith("file.py")
        assert files[0]["language_name"] == "python"
        assert files[0]["branch"] == "main"
        assert files[0]["missing_lsp_metadata"] is True
        assert files[0]["missing_ts_metadata"] is False

    @pytest.mark.asyncio
    async def test_track_missing_ts_metadata(self, tracker):
        """Test tracking a file with missing Tree-sitter metadata."""
        success = await tracker.track_missing_metadata(
            file_path="/test/file.rs",
            language_name="rust",
            branch="develop",
            missing_lsp=False,
            missing_ts=True,
        )

        assert success is True

        # Verify the file was tracked
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["language_name"] == "rust"
        assert files[0]["branch"] == "develop"
        assert files[0]["missing_lsp_metadata"] is False
        assert files[0]["missing_ts_metadata"] is True

    @pytest.mark.asyncio
    async def test_track_missing_both_metadata(self, tracker):
        """Test tracking a file with both LSP and Tree-sitter metadata missing."""
        success = await tracker.track_missing_metadata(
            file_path="/test/file.go",
            language_name="go",
            branch="main",
            missing_lsp=True,
            missing_ts=True,
        )

        assert success is True

        # Verify the file was tracked
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["missing_lsp_metadata"] is True
        assert files[0]["missing_ts_metadata"] is True

    @pytest.mark.asyncio
    async def test_track_duplicate_upsert(self, tracker):
        """Test that tracking the same file twice updates the record."""
        # Track file first time
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False,
        )

        # Track same file again with different flags
        success = await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=False,
            missing_ts=True,
        )

        assert success is True

        # Verify only one record exists with updated flags
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["missing_lsp_metadata"] is False
        assert files[0]["missing_ts_metadata"] is True


class TestMissingMetadataTrackerFiltering:
    """Test filtering functionality of MissingMetadataTracker."""

    @pytest.mark.asyncio
    async def test_filter_by_language(self, tracker):
        """Test filtering tracked files by language."""
        # Track multiple files with different languages
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.rs", "rust", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "main", missing_lsp=True
        )

        # Filter by Python
        python_files = await tracker.get_files_missing_metadata(language="python")
        assert len(python_files) == 2
        assert all(f["language_name"] == "python" for f in python_files)

        # Filter by Rust
        rust_files = await tracker.get_files_missing_metadata(language="rust")
        assert len(rust_files) == 1
        assert rust_files[0]["language_name"] == "rust"

    @pytest.mark.asyncio
    async def test_filter_by_missing_lsp(self, tracker):
        """Test filtering by missing LSP metadata."""
        # Track files with different metadata flags
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=False, missing_ts=True
        )

        # Filter by missing LSP
        lsp_files = await tracker.get_files_missing_metadata(missing_lsp=True)
        assert len(lsp_files) == 1
        assert lsp_files[0]["missing_lsp_metadata"] is True

        # Filter by NOT missing LSP
        no_lsp_files = await tracker.get_files_missing_metadata(missing_lsp=False)
        assert len(no_lsp_files) == 1
        assert no_lsp_files[0]["missing_lsp_metadata"] is False

    @pytest.mark.asyncio
    async def test_filter_by_missing_ts(self, tracker):
        """Test filtering by missing Tree-sitter metadata."""
        # Track files with different metadata flags
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=False, missing_ts=True
        )

        # Filter by missing Tree-sitter
        ts_files = await tracker.get_files_missing_metadata(missing_ts=True)
        assert len(ts_files) == 1
        assert ts_files[0]["missing_ts_metadata"] is True

    @pytest.mark.asyncio
    async def test_filter_by_branch(self, tracker):
        """Test filtering tracked files by branch."""
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

        # Filter by main branch
        main_files = await tracker.get_files_missing_metadata(branch="main")
        assert len(main_files) == 2
        assert all(f["branch"] == "main" for f in main_files)

        # Filter by develop branch
        dev_files = await tracker.get_files_missing_metadata(branch="develop")
        assert len(dev_files) == 1
        assert dev_files[0]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_filter_combined(self, tracker):
        """Test combining multiple filters."""
        # Track various files
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=False, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.rs", "rust", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file4.py", "python", "develop", missing_lsp=True, missing_ts=False
        )

        # Filter: Python + main branch + missing LSP
        files = await tracker.get_files_missing_metadata(
            language="python", branch="main", missing_lsp=True
        )
        assert len(files) == 1
        assert files[0]["language_name"] == "python"
        assert files[0]["branch"] == "main"
        assert files[0]["missing_lsp_metadata"] is True


class TestMissingMetadataTrackerRemoval:
    """Test removal functionality of MissingMetadataTracker."""

    @pytest.mark.asyncio
    async def test_remove_tracked_file(self, tracker):
        """Test removing a tracked file."""
        # Track a file
        file_path = "/test/file.py"
        await tracker.track_missing_metadata(
            file_path, "python", "main", missing_lsp=True
        )

        # Verify it exists
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1

        # Remove the file
        success = await tracker.remove_tracked_file(file_path)
        assert success is True

        # Verify it's gone
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_remove_nonexistent_file(self, tracker):
        """Test removing a file that doesn't exist in tracking."""
        success = await tracker.remove_tracked_file("/nonexistent/file.py")
        assert success is False

    @pytest.mark.asyncio
    async def test_remove_preserves_other_files(self, tracker):
        """Test that removing one file doesn't affect others."""
        # Track multiple files
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "main", missing_lsp=True
        )

        # Remove one file
        await tracker.remove_tracked_file("/test/file2.py")

        # Verify others remain
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 2
        file_paths = [f["file_absolute_path"] for f in files]
        assert any("file1.py" in p for p in file_paths)
        assert any("file3.py" in p for p in file_paths)
        assert not any("file2.py" in p for p in file_paths)


class TestMissingMetadataTrackerStatistics:
    """Test statistics functionality of MissingMetadataTracker."""

    @pytest.mark.asyncio
    async def test_get_empty_count(self, tracker):
        """Test getting count when no files are tracked."""
        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 0
        assert stats["missing_lsp"] == 0
        assert stats["missing_ts"] == 0
        assert stats["missing_both"] == 0

    @pytest.mark.asyncio
    async def test_get_count_with_lsp_only(self, tracker):
        """Test counting files missing only LSP metadata."""
        # Track files missing only LSP
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=True, missing_ts=False
        )

        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 2
        assert stats["missing_lsp"] == 2
        assert stats["missing_ts"] == 0
        assert stats["missing_both"] == 0

    @pytest.mark.asyncio
    async def test_get_count_with_ts_only(self, tracker):
        """Test counting files missing only Tree-sitter metadata."""
        # Track files missing only Tree-sitter
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=False, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=False, missing_ts=True
        )

        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 2
        assert stats["missing_lsp"] == 0
        assert stats["missing_ts"] == 2
        assert stats["missing_both"] == 0

    @pytest.mark.asyncio
    async def test_get_count_with_both(self, tracker):
        """Test counting files missing both LSP and Tree-sitter metadata."""
        # Track files missing both
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=True, missing_ts=True
        )

        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 2
        assert stats["missing_lsp"] == 2
        assert stats["missing_ts"] == 2
        assert stats["missing_both"] == 2

    @pytest.mark.asyncio
    async def test_get_count_mixed(self, tracker):
        """Test counting with mixed metadata flags."""
        # Track files with various combinations
        await tracker.track_missing_metadata(
            "/test/file1.py", "python", "main", missing_lsp=True, missing_ts=False
        )
        await tracker.track_missing_metadata(
            "/test/file2.py", "python", "main", missing_lsp=False, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file3.py", "python", "main", missing_lsp=True, missing_ts=True
        )
        await tracker.track_missing_metadata(
            "/test/file4.py", "python", "main", missing_lsp=False, missing_ts=False
        )

        stats = await tracker.get_tracked_file_count()
        assert stats["total"] == 4
        assert stats["missing_lsp"] == 2  # file1 and file3
        assert stats["missing_ts"] == 2  # file2 and file3
        assert stats["missing_both"] == 1  # file3


class TestMissingMetadataTrackerEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_track_absolute_path_normalization(self, tracker):
        """Test that file paths are normalized to absolute paths."""
        # Track with relative path
        await tracker.track_missing_metadata(
            "relative/file.py", "python", "main", missing_lsp=True
        )

        # Retrieve and verify path is absolute
        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert Path(files[0]["file_absolute_path"]).is_absolute()

    @pytest.mark.asyncio
    async def test_track_with_empty_strings(self, tracker):
        """Test tracking with empty string values."""
        success = await tracker.track_missing_metadata(
            "/test/file.py", "", "", missing_lsp=True, missing_ts=False
        )
        assert success is True

        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["language_name"] == ""
        assert files[0]["branch"] == ""

    @pytest.mark.asyncio
    async def test_get_files_no_results(self, tracker):
        """Test querying when no files match filters."""
        # Track a Python file
        await tracker.track_missing_metadata(
            "/test/file.py", "python", "main", missing_lsp=True
        )

        # Query for Rust files (should be empty)
        rust_files = await tracker.get_files_missing_metadata(language="rust")
        assert len(rust_files) == 0
        assert isinstance(rust_files, list)

    @pytest.mark.asyncio
    async def test_timestamps_present(self, tracker):
        """Test that created_at and updated_at timestamps are set."""
        await tracker.track_missing_metadata(
            "/test/file.py", "python", "main", missing_lsp=True
        )

        files = await tracker.get_files_missing_metadata()
        assert len(files) == 1
        assert files[0]["created_at"] is not None
        assert files[0]["updated_at"] is not None
        assert len(files[0]["created_at"]) > 0
        assert len(files[0]["updated_at"]) > 0


class TestToolAvailabilityDetection:
    """Test tool availability detection methods."""

    @pytest.mark.asyncio
    async def test_check_lsp_available_exists(self, state_manager_with_languages):
        """Test checking LSP availability for language with available LSP."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_lsp_available("python")

        assert result["language"] == "python"
        assert result["available"] is True
        assert result["path"] == "/usr/bin/pyright"

    @pytest.mark.asyncio
    async def test_check_lsp_available_missing(self, state_manager_with_languages):
        """Test checking LSP availability for language with missing LSP."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_lsp_available("javascript")

        assert result["language"] == "javascript"
        assert result["available"] is False
        assert result["path"] is None

    @pytest.mark.asyncio
    async def test_check_lsp_available_nonexistent_language(
        self, state_manager_with_languages
    ):
        """Test checking LSP availability for non-existent language."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_lsp_available("nonexistent")

        assert result["language"] == "nonexistent"
        assert result["available"] is False
        assert result["path"] is None

    @pytest.mark.asyncio
    async def test_check_lsp_available_empty_db(self, tracker):
        """Test checking LSP availability with empty database."""
        result = await tracker.check_lsp_available("python")

        assert result["language"] == "python"
        assert result["available"] is False
        assert result["path"] is None

    @pytest.mark.asyncio
    async def test_check_tree_sitter_available(self, state_manager_with_languages):
        """Test checking tree-sitter CLI availability when available."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_tree_sitter_available()

        assert result["available"] is True
        assert result["path"] == "/usr/bin/tree-sitter"

    @pytest.mark.asyncio
    async def test_check_tree_sitter_available_missing(self, tracker):
        """Test checking tree-sitter CLI availability when not available."""
        result = await tracker.check_tree_sitter_available()

        assert result["available"] is False
        assert result["path"] is None

    @pytest.mark.asyncio
    async def test_check_tools_available_both(self, state_manager_with_languages):
        """Test checking both LSP and tree-sitter for language with both."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_tools_available("python")

        assert result["language"] == "python"
        assert result["lsp"]["available"] is True
        assert result["lsp"]["path"] == "/usr/bin/pyright"
        assert result["tree_sitter"]["available"] is True
        assert result["tree_sitter"]["path"] == "/usr/bin/tree-sitter"

    @pytest.mark.asyncio
    async def test_check_tools_available_lsp_only(self, state_manager_with_languages):
        """Test checking tools for language with LSP but no tree-sitter config."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_tools_available("go")

        assert result["language"] == "go"
        assert result["lsp"]["available"] is True
        assert result["lsp"]["path"] == "/usr/bin/gopls"
        assert result["tree_sitter"]["available"] is True
        assert result["tree_sitter"]["path"] == "/usr/bin/tree-sitter"

    @pytest.mark.asyncio
    async def test_check_tools_available_missing_lsp(
        self, state_manager_with_languages
    ):
        """Test checking tools for language with missing LSP."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_tools_available("javascript")

        assert result["language"] == "javascript"
        assert result["lsp"]["available"] is False
        assert result["lsp"]["path"] is None
        assert result["tree_sitter"]["available"] is True
        assert result["tree_sitter"]["path"] == "/usr/bin/tree-sitter"

    @pytest.mark.asyncio
    async def test_check_tools_available_both_missing(
        self, state_manager_with_languages
    ):
        """Test checking tools for language with both tools missing."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        result = await tracker.check_tools_available("cpp")

        assert result["language"] == "cpp"
        assert result["lsp"]["available"] is False
        assert result["lsp"]["path"] is None
        # Tree-sitter is global - available since python/rust have it
        assert result["tree_sitter"]["available"] is True
        assert result["tree_sitter"]["path"] == "/usr/bin/tree-sitter"

    @pytest.mark.asyncio
    async def test_get_missing_tools_summary(self, state_manager_with_languages):
        """Test getting summary of missing tools across all languages."""
        tracker = MissingMetadataTracker(state_manager_with_languages)

        summary = await tracker.get_missing_tools_summary()

        # Both available: python, rust, go (has LSP, tree-sitter is global)
        assert "python" in summary["both_available"]
        assert "rust" in summary["both_available"]
        assert "go" in summary["both_available"]

        # Missing LSP: javascript, cpp
        assert "javascript" in summary["missing_lsp"]
        assert "cpp" in summary["missing_lsp"]

        # Missing tree-sitter: cpp
        assert "cpp" in summary["missing_tree_sitter"]

        # Java has no tool configuration, should not appear in any list
        assert "java" not in summary["both_available"]
        assert "java" not in summary["missing_lsp"]
        assert "java" not in summary["missing_tree_sitter"]

    @pytest.mark.asyncio
    async def test_get_missing_tools_summary_empty_db(self, tracker):
        """Test getting summary with empty database."""
        summary = await tracker.get_missing_tools_summary()

        assert summary["missing_lsp"] == []
        assert summary["missing_tree_sitter"] == []
        assert summary["both_available"] == []

    @pytest.mark.asyncio
    async def test_get_missing_tools_summary_all_available(self, state_manager):
        """Test getting summary when all configured tools are available."""
        # Insert languages with all tools available
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, lsp_name, lsp_absolute_path, lsp_missing,
                 ts_grammar, ts_cli_absolute_path, ts_missing)
                VALUES
                ('python', 'pyright', '/usr/bin/pyright', 0,
                 'tree-sitter-python', '/usr/bin/tree-sitter', 0),
                ('rust', 'rust-analyzer', '/usr/bin/rust-analyzer', 0,
                 'tree-sitter-rust', '/usr/bin/tree-sitter', 0)
                """
            )

        tracker = MissingMetadataTracker(state_manager)
        summary = await tracker.get_missing_tools_summary()

        assert len(summary["both_available"]) == 2
        assert len(summary["missing_lsp"]) == 0
        assert len(summary["missing_tree_sitter"]) == 0

    @pytest.mark.asyncio
    async def test_get_missing_tools_summary_all_missing(self, state_manager):
        """Test getting summary when all configured tools are missing."""
        # Insert languages with all tools missing
        async with state_manager.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages
                (language_name, lsp_name, lsp_absolute_path, lsp_missing,
                 ts_grammar, ts_cli_absolute_path, ts_missing)
                VALUES
                ('python', 'pyright', NULL, 1,
                 'tree-sitter-python', NULL, 1),
                ('rust', 'rust-analyzer', NULL, 1,
                 'tree-sitter-rust', NULL, 1)
                """
            )

        tracker = MissingMetadataTracker(state_manager)
        summary = await tracker.get_missing_tools_summary()

        assert len(summary["both_available"]) == 0
        assert "python" in summary["missing_lsp"]
        assert "rust" in summary["missing_lsp"]
        assert "python" in summary["missing_tree_sitter"]
        assert "rust" in summary["missing_tree_sitter"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
