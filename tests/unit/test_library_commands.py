"""Unit tests for library CLI commands (Task 399).

Tests multi-tenant library watch folder management.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestLibraryAddCommand:
    """Tests for wqm library add command."""

    @pytest.mark.asyncio
    async def test_add_library_watch_saves_config(self, tmp_path):
        """Test that add command saves watch configuration."""
        from wqm_cli.cli.commands.library import _add_library_watch

        # Create a temporary directory
        watch_dir = tmp_path / "test_library"
        watch_dir.mkdir()

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm, \
             patch("wqm_cli.cli.commands.library.get_daemon_client") as mock_daemon:

            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value=None)
            mock_state_manager.save_library_watch = AsyncMock(return_value=True)
            mock_get_sm.return_value = mock_state_manager

            mock_daemon.return_value.connect = AsyncMock()

            await _add_library_watch(
                path=str(watch_dir),
                name="test-lib",
                patterns=None,
                ignore=None,
                recursive=True,
                depth=10,
                debounce=5.0,
                auto_ingest=True,
            )

            # Verify save was called with correct args
            mock_state_manager.save_library_watch.assert_called_once()
            call_kwargs = mock_state_manager.save_library_watch.call_args[1]
            assert call_kwargs["library_name"] == "test-lib"
            assert call_kwargs["path"] == str(watch_dir)
            assert call_kwargs["recursive"] is True
            assert call_kwargs["recursive_depth"] == 10

    @pytest.mark.asyncio
    async def test_add_library_watch_normalizes_name(self, tmp_path):
        """Test that library name is normalized (lowercase, no prefix)."""
        from wqm_cli.cli.commands.library import _add_library_watch

        watch_dir = tmp_path / "lib"
        watch_dir.mkdir()

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm, \
             patch("wqm_cli.cli.commands.library.get_daemon_client") as mock_daemon:

            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value=None)
            mock_state_manager.save_library_watch = AsyncMock(return_value=True)
            mock_get_sm.return_value = mock_state_manager
            mock_daemon.return_value.connect = AsyncMock()

            # Test with underscore prefix
            await _add_library_watch(
                path=str(watch_dir),
                name="_MyLibrary",  # Should normalize to "mylibrary"
                patterns=None,
                ignore=None,
                recursive=True,
                depth=10,
                debounce=5.0,
                auto_ingest=False,
            )

            call_kwargs = mock_state_manager.save_library_watch.call_args[1]
            assert call_kwargs["library_name"] == "mylibrary"

    @pytest.mark.asyncio
    async def test_add_library_watch_rejects_nonexistent_path(self):
        """Test that add command rejects non-existent paths."""
        from wqm_cli.cli.commands.library import _add_library_watch
        import typer

        with pytest.raises(typer.Exit):
            await _add_library_watch(
                path="/nonexistent/path/12345",
                name="test",
                patterns=None,
                ignore=None,
                recursive=True,
                depth=10,
                debounce=5.0,
                auto_ingest=False,
            )


class TestLibraryWatchesCommand:
    """Tests for wqm library watches command."""

    @pytest.mark.asyncio
    async def test_list_watches_empty(self):
        """Test listing watches when none exist."""
        from wqm_cli.cli.commands.library import _list_library_watches

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.list_library_watches = AsyncMock(return_value=[])
            mock_get_sm.return_value = mock_state_manager

            # Should not raise
            await _list_library_watches(all_watches=False, format="table")

    @pytest.mark.asyncio
    async def test_list_watches_json_format(self):
        """Test listing watches in JSON format."""
        from wqm_cli.cli.commands.library import _list_library_watches

        watches = [
            {
                "library_name": "langchain",
                "path": "/docs/langchain",
                "enabled": True,
                "document_count": 100,
                "last_scan": "2024-01-01 12:00:00",
            }
        ]

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.list_library_watches = AsyncMock(return_value=watches)
            mock_get_sm.return_value = mock_state_manager

            await _list_library_watches(all_watches=True, format="json")


class TestLibraryUnwatchCommand:
    """Tests for wqm library unwatch command."""

    @pytest.mark.asyncio
    async def test_unwatch_removes_config(self):
        """Test that unwatch removes the watch configuration."""
        from wqm_cli.cli.commands.library import _remove_library_watch

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value={
                "library_name": "test",
                "path": "/test",
                "document_count": 10,
                "enabled": True,
            })
            mock_state_manager.remove_library_watch = AsyncMock(return_value=True)
            mock_get_sm.return_value = mock_state_manager

            await _remove_library_watch(
                name="test",
                delete_collection=False,
                force=True,
            )

            mock_state_manager.remove_library_watch.assert_called_once_with("test")

    @pytest.mark.asyncio
    async def test_unwatch_nonexistent_library(self):
        """Test that unwatch fails for non-existent library."""
        from wqm_cli.cli.commands.library import _remove_library_watch
        import typer

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value=None)
            mock_get_sm.return_value = mock_state_manager

            with pytest.raises(typer.Exit):
                await _remove_library_watch(
                    name="nonexistent",
                    delete_collection=False,
                    force=True,
                )


class TestLibraryRescanCommand:
    """Tests for wqm library rescan command."""

    @pytest.mark.asyncio
    async def test_rescan_queues_files(self, tmp_path):
        """Test that rescan queues matching files."""
        from wqm_cli.cli.commands.library import _rescan_library

        # Create test files
        (tmp_path / "doc1.pdf").touch()
        (tmp_path / "doc2.md").touch()
        (tmp_path / "ignored.py").touch()

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value={
                "library_name": "test",
                "path": str(tmp_path),
                "patterns": ["*.pdf", "*.md"],
                "ignore_patterns": [],
                "recursive": True,
                "recursive_depth": 10,
                "enabled": True,
            })
            mock_state_manager.add_to_ingestion_queue = AsyncMock()
            mock_state_manager.update_library_watch_stats = AsyncMock()
            mock_get_sm.return_value = mock_state_manager

            await _rescan_library(name="test", clear_first=False, force=True)

            # Should queue 2 files (pdf and md, not py)
            assert mock_state_manager.add_to_ingestion_queue.call_count == 2

    @pytest.mark.asyncio
    async def test_rescan_nonexistent_library(self):
        """Test that rescan fails for non-existent library."""
        from wqm_cli.cli.commands.library import _rescan_library
        import typer

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value=None)
            mock_get_sm.return_value = mock_state_manager

            with pytest.raises(typer.Exit):
                await _rescan_library(name="nonexistent", clear_first=False, force=True)


class TestLibraryWatchStatusCommand:
    """Tests for wqm library watch-status command."""

    @pytest.mark.asyncio
    async def test_watch_status_displays_info(self):
        """Test that watch-status displays configuration."""
        from wqm_cli.cli.commands.library import _library_watch_status

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value={
                "library_name": "langchain",
                "path": "/docs/langchain",
                "patterns": ["*.pdf", "*.md"],
                "ignore_patterns": [".git/*"],
                "enabled": True,
                "recursive": True,
                "recursive_depth": 10,
                "debounce_seconds": 5.0,
                "document_count": 50,
                "last_scan": "2024-01-01 12:00:00",
                "added_at": "2024-01-01 10:00:00",
            })
            mock_get_sm.return_value = mock_state_manager

            # Should not raise
            await _library_watch_status(name="langchain", detailed=False)

    @pytest.mark.asyncio
    async def test_watch_status_nonexistent(self):
        """Test that watch-status fails for non-existent library."""
        from wqm_cli.cli.commands.library import _library_watch_status
        import typer

        with patch("wqm_cli.cli.commands.library._get_state_manager") as mock_get_sm:
            mock_state_manager = MagicMock()
            mock_state_manager.get_library_watch = AsyncMock(return_value=None)
            mock_get_sm.return_value = mock_state_manager

            with pytest.raises(typer.Exit):
                await _library_watch_status(name="nonexistent", detailed=False)
