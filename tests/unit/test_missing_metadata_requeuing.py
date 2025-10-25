#!/usr/bin/env python3
"""
Unit tests for MissingMetadataTracker requeuing functionality.

Tests the tool-available requeuing system that automatically requeues files
for processing when missing tools (LSP or tree-sitter) become available.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.python.common.core.missing_metadata_tracker import MissingMetadataTracker
from src.python.common.core.sqlite_state_manager import SQLiteStateManager


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_requeuing.db"
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
async def state_manager_with_lsp_available(state_manager):
    """Create state manager with available LSP for Python."""
    async with state_manager.transaction() as conn:
        conn.execute(
            """
            INSERT INTO languages
            (language_name, lsp_name, lsp_absolute_path, lsp_missing)
            VALUES ('python', 'pyright', '/usr/bin/pyright', 0)
            """
        )
    yield state_manager


@pytest.fixture
async def state_manager_with_ts_available(state_manager):
    """Create state manager with available tree-sitter."""
    async with state_manager.transaction() as conn:
        conn.execute(
            """
            INSERT INTO languages
            (language_name, ts_grammar, ts_cli_absolute_path, ts_missing)
            VALUES ('python', 'tree-sitter-python', '/usr/bin/tree-sitter', 0)
            """
        )
    yield state_manager


@pytest.fixture
async def state_manager_with_lsp_missing(state_manager):
    """Create state manager with missing LSP for Python."""
    async with state_manager.transaction() as conn:
        conn.execute(
            """
            INSERT INTO languages
            (language_name, lsp_name, lsp_absolute_path, lsp_missing)
            VALUES ('python', 'pyright', NULL, 1)
            """
        )
    yield state_manager


class TestRequeuingValidation:
    """Test input validation for requeuing methods."""

    @pytest.mark.asyncio
    async def test_requeue_invalid_tool_type(self, tracker):
        """Test requeuing with invalid tool type."""
        with pytest.raises(ValueError, match="Invalid tool_type"):
            await tracker.requeue_when_tools_available(
                tool_type="invalid",
                language="python",
                priority=5
            )

    @pytest.mark.asyncio
    async def test_requeue_invalid_priority_low(self, tracker):
        """Test requeuing with priority below valid range."""
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=-1
            )

    @pytest.mark.asyncio
    async def test_requeue_invalid_priority_high(self, tracker):
        """Test requeuing with priority above valid range."""
        with pytest.raises(ValueError, match="Priority must be between 0 and 10"):
            await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=11
            )

    @pytest.mark.asyncio
    async def test_requeue_lsp_without_language(self, tracker):
        """Test LSP requeuing without language parameter."""
        with pytest.raises(ValueError, match="Language parameter required"):
            await tracker.requeue_when_tools_available(
                tool_type="lsp",
                priority=5
            )

    @pytest.mark.asyncio
    async def test_requeue_tree_sitter_accepts_no_language(self, state_manager_with_ts_available):
        """Test tree-sitter requeuing without language parameter is valid."""
        tracker = MissingMetadataTracker(state_manager_with_ts_available)

        # Should not raise - tree-sitter doesn't require language
        result = await tracker.requeue_when_tools_available(
            tool_type="tree_sitter",
            priority=5
        )

        assert result["tool_type"] == "tree_sitter"
        assert result["language"] is None


class TestRequeuingLSP:
    """Test LSP requeuing functionality."""

    @pytest.mark.asyncio
    async def test_requeue_lsp_tool_not_available(self, state_manager_with_lsp_missing):
        """Test requeuing when LSP is not available."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_missing)

        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="python",
            priority=5
        )

        assert result["tool_type"] == "lsp"
        assert result["language"] == "python"
        assert result["tool_available"] is False
        assert result["files_requeued"] == 0
        assert result["files_failed"] == 0
        assert result["files_removed"] == 0

    @pytest.mark.asyncio
    async def test_requeue_lsp_no_tracked_files(self, state_manager_with_lsp_available):
        """Test requeuing when LSP is available but no files are tracked."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        result = await tracker.requeue_when_tools_available(
            tool_type="lsp",
            language="python",
            priority=5
        )

        assert result["tool_type"] == "lsp"
        assert result["language"] == "python"
        assert result["tool_available"] is True
        assert result["tool_path"] == "/usr/bin/pyright"
        assert result["files_requeued"] == 0
        assert result["files_failed"] == 0
        assert result["files_removed"] == 0

    @pytest.mark.asyncio
    async def test_requeue_lsp_single_file(self, state_manager_with_lsp_available):
        """Test requeuing a single file when LSP becomes available."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track a file missing LSP
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        # Mock the enqueue method to avoid actual queue operations
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )

        assert result["tool_available"] is True
        assert result["files_requeued"] == 1
        assert result["files_failed"] == 0
        assert result["files_removed"] == 1

        # Verify enqueue was called with correct parameters
        mock_enqueue.assert_called_once()
        call_kwargs = mock_enqueue.call_args.kwargs
        assert call_kwargs["priority"] == 5
        assert call_kwargs["branch"] == "main"
        assert call_kwargs["metadata"]["requeued_for"] == "lsp"
        assert call_kwargs["metadata"]["language"] == "python"

        # Verify file was removed from tracking
        files = await tracker.get_files_missing_metadata(language="python", missing_lsp=True)
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_requeue_lsp_multiple_files(self, state_manager_with_lsp_available):
        """Test requeuing multiple files when LSP becomes available."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track multiple files missing LSP
        for i in range(5):
            await tracker.track_missing_metadata(
                file_path=f"/test/file{i}.py",
                language_name="python",
                branch="main",
                missing_lsp=True,
                missing_ts=False
            )

        # Mock the enqueue method
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=7
            )

        assert result["files_requeued"] == 5
        assert result["files_failed"] == 0
        assert result["files_removed"] == 5
        assert mock_enqueue.call_count == 5

        # Verify all files were removed from tracking
        files = await tracker.get_files_missing_metadata(language="python", missing_lsp=True)
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_requeue_lsp_batch_processing(self, state_manager_with_lsp_available):
        """Test that large number of files are processed in batches."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track 150 files (should be processed in batches of 100)
        for i in range(150):
            await tracker.track_missing_metadata(
                file_path=f"/test/file{i}.py",
                language_name="python",
                branch="main",
                missing_lsp=True,
                missing_ts=False
            )

        # Mock the enqueue method
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )

        assert result["files_requeued"] == 150
        assert result["files_failed"] == 0
        assert mock_enqueue.call_count == 150

    @pytest.mark.asyncio
    async def test_requeue_lsp_partial_failure(self, state_manager_with_lsp_available):
        """Test requeuing with some files failing."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track multiple files
        for i in range(5):
            await tracker.track_missing_metadata(
                file_path=f"/test/file{i}.py",
                language_name="python",
                branch="main",
                missing_lsp=True,
                missing_ts=False
            )

        # Mock enqueue to fail for some files
        call_count = 0
        async def mock_enqueue_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count in (2, 4):  # Fail on 2nd and 4th calls
                raise Exception("Enqueue failed")

        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            mock_enqueue.side_effect = mock_enqueue_side_effect

            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )

        assert result["files_requeued"] == 3  # 3 succeeded
        assert result["files_failed"] == 2    # 2 failed
        assert result["files_removed"] == 3   # Only successful ones removed

        # Verify failed files are still tracked
        files = await tracker.get_files_missing_metadata(language="python", missing_lsp=True)
        assert len(files) == 2  # Failed files remain tracked

    @pytest.mark.asyncio
    async def test_requeue_lsp_different_branches(self, state_manager_with_lsp_available):
        """Test requeuing files from different branches."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track files on different branches
        await tracker.track_missing_metadata(
            file_path="/test/file1.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )
        await tracker.track_missing_metadata(
            file_path="/test/file2.py",
            language_name="python",
            branch="feature/test",
            missing_lsp=True,
            missing_ts=False
        )

        branches_seen = []

        async def track_branch(*args, **kwargs):
            branches_seen.append(kwargs.get("branch"))

        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            mock_enqueue.side_effect = track_branch

            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )

        assert result["files_requeued"] == 2
        assert "main" in branches_seen
        assert "feature/test" in branches_seen


class TestRequeuingTreeSitter:
    """Test tree-sitter requeuing functionality."""

    @pytest.mark.asyncio
    async def test_requeue_tree_sitter_single_file(self, state_manager_with_ts_available):
        """Test requeuing a single file when tree-sitter becomes available."""
        tracker = MissingMetadataTracker(state_manager_with_ts_available)

        # Track a file missing tree-sitter
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=False,
            missing_ts=True
        )

        # Mock the enqueue method
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="tree_sitter",
                priority=5
            )

        assert result["tool_available"] is True
        assert result["files_requeued"] == 1
        assert result["files_failed"] == 0
        assert result["files_removed"] == 1

        # Verify enqueue was called with correct parameters
        mock_enqueue.assert_called_once()
        call_kwargs = mock_enqueue.call_args.kwargs
        assert call_kwargs["metadata"]["requeued_for"] == "tree_sitter"

        # Verify file was removed from tracking
        files = await tracker.get_files_missing_metadata(missing_ts=True)
        assert len(files) == 0

    @pytest.mark.asyncio
    async def test_requeue_tree_sitter_multiple_languages(self, state_manager_with_ts_available):
        """Test requeuing files from multiple languages when tree-sitter becomes available."""
        tracker = MissingMetadataTracker(state_manager_with_ts_available)

        # Add rust language to database
        async with state_manager_with_ts_available.transaction() as conn:
            conn.execute(
                """
                INSERT INTO languages (language_name)
                VALUES ('rust')
                """
            )

        # Track files from different languages missing tree-sitter
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=False,
            missing_ts=True
        )
        await tracker.track_missing_metadata(
            file_path="/test/file.rs",
            language_name="rust",
            branch="main",
            missing_lsp=False,
            missing_ts=True
        )

        languages_seen = []

        async def track_language(*args, **kwargs):
            languages_seen.append(kwargs.get("metadata", {}).get("language"))

        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            mock_enqueue.side_effect = track_language

            result = await tracker.requeue_when_tools_available(
                tool_type="tree_sitter",
                priority=5
            )

        assert result["files_requeued"] == 2
        assert "python" in languages_seen
        assert "rust" in languages_seen


class TestRequeuingHelpers:
    """Test helper methods for requeuing."""

    @pytest.mark.asyncio
    async def test_get_languages_with_missing_lsp(self, state_manager):
        """Test getting distinct languages with missing LSP."""
        tracker = MissingMetadataTracker(state_manager)

        # Track files for multiple languages
        await tracker.track_missing_metadata(
            file_path="/test/file1.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )
        await tracker.track_missing_metadata(
            file_path="/test/file2.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )
        await tracker.track_missing_metadata(
            file_path="/test/file.rs",
            language_name="rust",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        languages = await tracker._get_languages_with_missing_lsp()

        assert len(languages) == 2
        assert "python" in languages
        assert "rust" in languages

    @pytest.mark.asyncio
    async def test_get_languages_with_missing_lsp_empty(self, tracker):
        """Test getting languages when no files are tracked."""
        languages = await tracker._get_languages_with_missing_lsp()
        assert len(languages) == 0


class TestRequeuingEdgeCases:
    """Test edge cases in requeuing functionality."""

    @pytest.mark.asyncio
    async def test_requeue_priority_boundaries(self, state_manager_with_lsp_available):
        """Test requeuing with priority at boundaries (0 and 10)."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        # Track a file
        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        # Test priority 0 (minimum)
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=0
            )
            assert result["files_requeued"] == 1
            assert mock_enqueue.call_args.kwargs["priority"] == 0

        # Track another file
        await tracker.track_missing_metadata(
            file_path="/test/file2.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        # Test priority 10 (maximum)
        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=10
            )
            assert result["files_requeued"] == 1
            assert mock_enqueue.call_args.kwargs["priority"] == 10

    @pytest.mark.asyncio
    async def test_requeue_default_priority(self, state_manager_with_lsp_available):
        """Test requeuing uses default priority of 5."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        with patch.object(
            tracker.state_manager, 'enqueue', new_callable=AsyncMock
        ) as mock_enqueue:
            # Don't specify priority, should use default of 5
            await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python"
            )
            assert mock_enqueue.call_args.kwargs["priority"] == 5

    @pytest.mark.asyncio
    async def test_requeue_exception_handling(self, state_manager_with_lsp_available):
        """Test that exceptions during requeue are handled gracefully."""
        tracker = MissingMetadataTracker(state_manager_with_lsp_available)

        await tracker.track_missing_metadata(
            file_path="/test/file.py",
            language_name="python",
            branch="main",
            missing_lsp=True,
            missing_ts=False
        )

        # Mock get_files_missing_metadata to raise an exception
        with patch.object(
            tracker, 'get_files_missing_metadata', side_effect=Exception("Database error")
        ):
            result = await tracker.requeue_when_tools_available(
                tool_type="lsp",
                language="python",
                priority=5
            )

        # Should return result with error information
        assert result["files_requeued"] == 0
        assert result["files_failed"] == 0
        assert len(result["errors"]) > 0
        assert "Database error" in result["errors"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
