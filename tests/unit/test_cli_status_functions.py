"""Unit tests for CLI status functions (Task 463).

Tests the SQLite-based status functions that replaced temporary stubs.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from common.core.sqlite_state_manager import SQLiteStateManager, WatchFolderConfig


@pytest.fixture
async def temp_state_manager():
    """Create a temporary SQLiteStateManager for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_state.db"
        manager = SQLiteStateManager(db_path=str(db_path))
        await manager.initialize()
        yield manager
        await manager.close()


class TestGetProcessingStatus:
    """Tests for get_processing_status function."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, temp_state_manager):
        """Test that get_processing_status returns success=True."""
        # Import here to avoid module-level issues
        from wqm_cli.cli.status import _get_state_manager, get_processing_status

        # Mock the state manager
        with patch.object(
            temp_state_manager.__class__,
            'get_processing_states',
            new_callable=AsyncMock,
            return_value=[]
        ):
            # Patch the global state manager
            import wqm_cli.cli.status as status_module
            status_module._state_manager = temp_state_manager

            result = await get_processing_status()

            assert result.get("success") is True
            assert "processing_info" in result
            assert "recent_files" in result

    @pytest.mark.asyncio
    async def test_processing_info_structure(self, temp_state_manager):
        """Test that processing_info has expected keys."""
        from wqm_cli.cli.status import get_processing_status

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_processing_status()

        processing_info = result.get("processing_info", {})
        assert "currently_processing" in processing_info
        assert "recent_successful" in processing_info
        assert "recent_failed" in processing_info

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self, temp_state_manager):
        """Test that exceptions are handled gracefully."""
        from wqm_cli.cli.status import get_processing_status

        import wqm_cli.cli.status as status_module

        # Create a mock that raises an exception
        mock_manager = MagicMock(spec=SQLiteStateManager)
        mock_manager.get_processing_states = AsyncMock(
            side_effect=Exception("Database error")
        )
        status_module._state_manager = mock_manager

        result = await get_processing_status()

        assert result.get("success") is False
        assert "error" in result


class TestGetQueueStats:
    """Tests for get_queue_stats function."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, temp_state_manager):
        """Test that get_queue_stats returns success=True."""
        from wqm_cli.cli.status import get_queue_stats

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_queue_stats()

        assert result.get("success") is True
        assert "queue_stats" in result

    @pytest.mark.asyncio
    async def test_queue_stats_structure(self, temp_state_manager):
        """Test that queue_stats has expected keys."""
        from wqm_cli.cli.status import get_queue_stats

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_queue_stats()

        queue_stats = result.get("queue_stats", {})
        assert "total" in queue_stats
        assert "file_queue" in queue_stats
        assert "content_queue" in queue_stats

    @pytest.mark.asyncio
    async def test_handles_exception_gracefully(self):
        """Test that exceptions are handled gracefully."""
        from wqm_cli.cli.status import get_queue_stats

        import wqm_cli.cli.status as status_module

        # Create a mock that raises an exception
        mock_manager = MagicMock(spec=SQLiteStateManager)
        mock_manager.get_queue_depth = AsyncMock(
            side_effect=Exception("Database error")
        )
        status_module._state_manager = mock_manager

        result = await get_queue_stats()

        assert result.get("success") is False
        assert "error" in result


class TestGetWatchFolderConfigs:
    """Tests for get_watch_folder_configs function."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, temp_state_manager):
        """Test that get_watch_folder_configs returns success=True."""
        from wqm_cli.cli.status import get_watch_folder_configs

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_watch_folder_configs()

        assert result.get("success") is True
        assert "watch_configs" in result

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_configs(self, temp_state_manager):
        """Test that empty list is returned when no configs exist."""
        from wqm_cli.cli.status import get_watch_folder_configs

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_watch_folder_configs()

        assert result.get("watch_configs") == []

    @pytest.mark.asyncio
    async def test_returns_config_list_when_configs_exist(self, temp_state_manager):
        """Test that configs are returned when they exist."""
        from wqm_cli.cli.status import get_watch_folder_configs

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        # Add a watch config
        config = WatchFolderConfig(
            watch_id="test-watch",
            path="/tmp/test",
            collection="test-collection",
            patterns=["*.py"],
            ignore_patterns=[],
            enabled=True,
        )
        await temp_state_manager.save_watch_folder_config(config)

        result = await get_watch_folder_configs()

        assert len(result.get("watch_configs", [])) == 1
        assert result["watch_configs"][0]["watch_id"] == "test-watch"


class TestGetDatabaseStats:
    """Tests for get_database_stats function."""

    @pytest.mark.asyncio
    async def test_returns_success_true(self, temp_state_manager):
        """Test that get_database_stats returns success=True."""
        from wqm_cli.cli.status import get_database_stats

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_database_stats()

        assert result.get("success") is True
        assert "database_stats" in result

    @pytest.mark.asyncio
    async def test_database_stats_structure(self, temp_state_manager):
        """Test that database_stats has expected keys."""
        from wqm_cli.cli.status import get_database_stats

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_database_stats()

        db_stats = result.get("database_stats", {})
        assert "total_size_mb" in db_stats
        assert "total_size_bytes" in db_stats
        assert "total_records" in db_stats
        assert "recent_processing" in db_stats

    @pytest.mark.asyncio
    async def test_database_size_is_positive(self, temp_state_manager):
        """Test that database size is a positive number."""
        from wqm_cli.cli.status import get_database_stats

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        result = await get_database_stats()

        db_stats = result.get("database_stats", {})
        assert db_stats.get("total_size_bytes", 0) >= 0
        assert db_stats.get("total_size_mb", 0) >= 0


class TestGetComprehensiveStatus:
    """Tests for get_comprehensive_status function."""

    @pytest.mark.asyncio
    async def test_returns_all_status_sections(self, temp_state_manager):
        """Test that get_comprehensive_status returns all expected sections."""
        from wqm_cli.cli.status import get_comprehensive_status

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        # Mock the gRPC stats to avoid network calls
        with patch(
            'wqm_cli.cli.status.get_grpc_engine_stats',
            new_callable=AsyncMock,
            return_value={"success": False, "error": "Not connected"}
        ):
            result = await get_comprehensive_status()

        assert "processing_status" in result
        assert "queue_stats" in result
        assert "watch_configs" in result
        assert "database_stats" in result
        assert "grpc_stats" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_all_sections_have_success_field(self, temp_state_manager):
        """Test that all status sections have a success field."""
        from wqm_cli.cli.status import get_comprehensive_status

        import wqm_cli.cli.status as status_module
        status_module._state_manager = temp_state_manager

        # Mock the gRPC stats to avoid network calls
        with patch(
            'wqm_cli.cli.status.get_grpc_engine_stats',
            new_callable=AsyncMock,
            return_value={"success": False, "error": "Not connected"}
        ):
            result = await get_comprehensive_status()

        assert result.get("processing_status", {}).get("success") is True
        assert result.get("queue_stats", {}).get("success") is True
        assert result.get("watch_configs", {}).get("success") is True
        assert result.get("database_stats", {}).get("success") is True
