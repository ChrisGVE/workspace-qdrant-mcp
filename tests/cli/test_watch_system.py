"""
Tests for the file watching system.

Tests the file watching functionality including FileWatcher, WatchManager,
WatchService, and CLI integration.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest

from workspace_qdrant_mcp.core.file_watcher import (
    WatchConfiguration,
    WatchEvent,
    FileWatcher,
    WatchManager
)
from workspace_qdrant_mcp.cli.watch_service import WatchService
from workspace_qdrant_mcp.core.client import QdrantWorkspaceClient


@pytest.fixture
def sample_watch_config():
    """Create a sample watch configuration."""
    return WatchConfiguration(
        id="test_watch_1",
        path="/tmp/test_watch",
        collection="_test_library",
        patterns=["*.txt", "*.md"],
        ignore_patterns=[".git/*", "*.tmp"],
        auto_ingest=True,
        recursive=True,
        debounce_seconds=2,
    )


@pytest.fixture
def mock_client():
    """Create a mock Qdrant client."""
    client = Mock(spec=QdrantWorkspaceClient)
    client.list_collections = AsyncMock(return_value=["_test_library", "_docs"])
    return client


@pytest.fixture
def temp_watch_dir():
    """Create a temporary directory for testing file watching."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create some test files
        (temp_path / "test.txt").write_text("Test content")
        (temp_path / "README.md").write_text("# Test\nContent")
        (temp_path / "ignore.tmp").write_text("Ignore this")
        
        # Create subdirectory
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "nested.txt").write_text("Nested content")
        
        yield temp_path


class TestWatchConfiguration:
    """Test WatchConfiguration dataclass."""
    
    def test_to_dict(self, sample_watch_config):
        """Test serialization to dictionary."""
        data = sample_watch_config.to_dict()
        
        assert data["id"] == "test_watch_1"
        assert data["path"] == "/tmp/test_watch"
        assert data["collection"] == "_test_library"
        assert data["patterns"] == ["*.txt", "*.md"]
        assert data["auto_ingest"] is True
        
    def test_from_dict(self, sample_watch_config):
        """Test deserialization from dictionary."""
        data = sample_watch_config.to_dict()
        restored = WatchConfiguration.from_dict(data)
        
        assert restored.id == sample_watch_config.id
        assert restored.path == sample_watch_config.path
        assert restored.collection == sample_watch_config.collection
        assert restored.patterns == sample_watch_config.patterns


class TestWatchEvent:
    """Test WatchEvent dataclass."""
    
    def test_creation(self):
        """Test watch event creation."""
        event = WatchEvent(
            change_type="added",
            file_path="/path/to/file.txt",
            collection="_test_library"
        )
        
        assert event.change_type == "added"
        assert event.file_path == "/path/to/file.txt"
        assert event.collection == "_test_library"
        assert event.timestamp  # Should have a timestamp


@pytest.mark.asyncio
class TestFileWatcher:
    """Test FileWatcher functionality."""
    
    async def test_file_watcher_creation(self, sample_watch_config):
        """Test creating a FileWatcher."""
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        watcher = FileWatcher(
            config=sample_watch_config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback,
        )
        
        assert watcher.config == sample_watch_config
        assert watcher.ingestion_callback == ingestion_callback
        assert watcher.event_callback == event_callback
        assert not watcher.is_running()
    
    async def test_pattern_matching(self, sample_watch_config, temp_watch_dir):
        """Test file pattern matching."""
        sample_watch_config.path = str(temp_watch_dir)
        
        ingestion_callback = AsyncMock()
        watcher = FileWatcher(
            config=sample_watch_config,
            ingestion_callback=ingestion_callback,
        )
        
        # Test pattern matching
        txt_file = temp_watch_dir / "test.txt"
        md_file = temp_watch_dir / "README.md"
        tmp_file = temp_watch_dir / "ignore.tmp"
        
        assert watcher._matches_patterns(txt_file)
        assert watcher._matches_patterns(md_file)
        assert not watcher._matches_patterns(tmp_file)  # Not in patterns
        
        # Test ignore patterns
        git_file = temp_watch_dir / ".git" / "config"
        assert watcher._matches_ignore_patterns(git_file)
        assert watcher._matches_ignore_patterns(tmp_file)  # In ignore patterns
    
    @pytest.mark.slow
    async def test_start_stop_watcher(self, sample_watch_config, temp_watch_dir):
        """Test starting and stopping a file watcher."""
        sample_watch_config.path = str(temp_watch_dir)
        
        ingestion_callback = AsyncMock()
        watcher = FileWatcher(
            config=sample_watch_config,
            ingestion_callback=ingestion_callback,
        )
        
        # Start watcher
        await watcher.start()
        assert watcher.is_running()
        assert watcher.config.status == "active"
        
        # Let it run briefly
        await asyncio.sleep(0.1)
        
        # Stop watcher
        await watcher.stop()
        assert not watcher.is_running()
        assert watcher.config.status == "paused"


@pytest.mark.asyncio
class TestWatchManager:
    """Test WatchManager functionality."""
    
    async def test_watch_manager_creation(self):
        """Test creating a WatchManager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_watches.json"
            manager = WatchManager(config_file=str(config_file))
            
            assert manager.config_file == config_file
            assert len(manager.configurations) == 0
            assert len(manager.watchers) == 0
    
    async def test_add_watch(self, temp_watch_dir):
        """Test adding a watch configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_watches.json"
            manager = WatchManager(config_file=str(config_file))
            
            # Set up callback
            manager.set_ingestion_callback(AsyncMock())
            
            watch_id = await manager.add_watch(
                path=str(temp_watch_dir),
                collection="_test_library",
                patterns=["*.txt"],
                auto_ingest=True,
            )
            
            assert watch_id in manager.configurations
            config = manager.configurations[watch_id]
            assert config.path == str(temp_watch_dir.resolve())
            assert config.collection == "_test_library"
            assert config.patterns == ["*.txt"]
    
    async def test_save_load_configurations(self, temp_watch_dir):
        """Test saving and loading watch configurations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_watches.json"
            
            # Create manager and add watch
            manager = WatchManager(config_file=str(config_file))
            manager.set_ingestion_callback(AsyncMock())
            
            watch_id = await manager.add_watch(
                path=str(temp_watch_dir),
                collection="_test_library",
                patterns=["*.txt"],
            )
            
            # Create new manager and load configurations
            manager2 = WatchManager(config_file=str(config_file))
            await manager2.load_configurations()
            
            assert watch_id in manager2.configurations
            config = manager2.configurations[watch_id]
            assert config.path == str(temp_watch_dir.resolve())
            assert config.collection == "_test_library"
    
    async def test_remove_watch(self, temp_watch_dir):
        """Test removing a watch configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = Path(temp_dir) / "test_watches.json"
            manager = WatchManager(config_file=str(config_file))
            manager.set_ingestion_callback(AsyncMock())
            
            # Add watch
            watch_id = await manager.add_watch(
                path=str(temp_watch_dir),
                collection="_test_library",
            )
            
            assert watch_id in manager.configurations
            
            # Remove watch
            result = await manager.remove_watch(watch_id)
            assert result is True
            assert watch_id not in manager.configurations
            
            # Try to remove non-existent watch
            result = await manager.remove_watch("nonexistent")
            assert result is False


@pytest.mark.asyncio
class TestWatchService:
    """Test WatchService integration."""
    
    async def test_watch_service_creation(self, mock_client):
        """Test creating a WatchService."""
        service = WatchService(mock_client)
        await service.initialize()
        
        assert service.client == mock_client
        assert service.watch_manager is not None
        assert service.ingestion_engine is not None
    
    async def test_add_watch_validation(self, mock_client, temp_watch_dir):
        """Test watch addition with validation."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Test successful addition
        watch_id = await service.add_watch(
            path=str(temp_watch_dir),
            collection="_test_library",  # Exists in mock
        )
        
        assert watch_id is not None
        watches = await service.list_watches()
        assert len(watches) == 1
        assert watches[0].collection == "_test_library"
    
    async def test_add_watch_validation_errors(self, mock_client):
        """Test watch addition validation errors."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Test non-existent path
        with pytest.raises(ValueError, match="Path does not exist"):
            await service.add_watch(
                path="/nonexistent/path",
                collection="_test_library",
            )
        
        # Test non-library collection (doesn't start with _)
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Collection must start with underscore"):
                await service.add_watch(
                    path=temp_dir,
                    collection="regular_collection",
                )
        
        # Test non-existent collection
        mock_client.list_collections.return_value = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Collection .* not found"):
                await service.add_watch(
                    path=temp_dir,
                    collection="_nonexistent",
                )
    
    async def test_watch_service_status(self, mock_client, temp_watch_dir):
        """Test getting watch service status."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Add some watches
        await service.add_watch(str(temp_watch_dir), "_test_library")
        
        # Get status
        status = await service.get_watch_status()
        
        assert "total_watches" in status
        assert "active_watches" in status
        assert "recent_activity" in status
        assert status["total_watches"] == 1
    
    async def test_sync_watched_folders(self, mock_client, temp_watch_dir):
        """Test manual sync of watched folders."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Mock the ingestion engine
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "Processed 2 files"
        mock_result.stats = Mock()
        mock_result.stats.files_processed = 2
        mock_result.stats.files_failed = 0
        
        with patch.object(service.ingestion_engine, 'process_directory', return_value=mock_result):
            # Add watch
            await service.add_watch(str(temp_watch_dir), "_test_library")
            
            # Sync
            results = await service.sync_watched_folders(dry_run=True)
            
            assert len(results) == 1
            result_key = list(results.keys())[0]
            assert results[result_key]["success"] is True
            assert "Processed 2 files" in results[result_key]["message"]


@pytest.mark.asyncio 
class TestFileIngestionIntegration:
    """Test file ingestion integration."""
    
    async def test_handle_file_ingestion(self, mock_client, temp_watch_dir):
        """Test automatic file ingestion handling."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Create a test file
        test_file = temp_watch_dir / "new_file.txt"
        test_file.write_text("New file content")
        
        # Mock the ingestion engine
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "Successfully processed file"
        
        with patch.object(service.ingestion_engine, 'process_directory', return_value=mock_result):
            # Test file ingestion
            await service._handle_file_ingestion(str(test_file), "_test_library")
            
            # Verify ingestion engine was called
            service.ingestion_engine.process_directory.assert_called_once()
            call_args = service.ingestion_engine.process_directory.call_args
            
            assert call_args[1]["collection"] == "_test_library"
            assert call_args[1]["dry_run"] is False
            assert call_args[1]["recursive"] is False
    
    async def test_handle_missing_file_ingestion(self, mock_client):
        """Test handling of missing file during ingestion."""
        service = WatchService(mock_client)
        await service.initialize()
        
        # Try to ingest non-existent file
        with patch('workspace_qdrant_mcp.cli.watch_service.logger') as mock_logger:
            await service._handle_file_ingestion("/nonexistent/file.txt", "_test_library")
            
            # Should log a warning and return without error
            mock_logger.warning.assert_called_once()


@pytest.mark.integration
class TestWatchSystemIntegration:
    """Integration tests for the complete watch system."""
    
    @pytest.mark.slow
    async def test_end_to_end_file_watching(self, mock_client):
        """Test complete file watching workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            service = WatchService(mock_client)
            await service.initialize()
            
            # Add watch
            watch_id = await service.add_watch(
                path=str(temp_path),
                collection="_test_library",
                debounce_seconds=1,  # Short debounce for testing
            )
            
            # Mock ingestion
            mock_result = Mock()
            mock_result.success = True
            mock_result.message = "Processed 1 file"
            
            with patch.object(service.ingestion_engine, 'process_directory', return_value=mock_result):
                # Start watching
                await service.start_all_watches()
                
                # Give watcher time to start
                await asyncio.sleep(0.2)
                
                # Create a file (this should trigger ingestion)
                test_file = temp_path / "test.txt"
                test_file.write_text("Test content")
                
                # Wait for debounce + processing
                await asyncio.sleep(2)
                
                # Check if ingestion was triggered
                assert len(service.activity_log) > 0
                
                # Stop watching
                await service.stop_all_watches()


class TestCLIIntegration:
    """Test CLI command integration (without actual async execution)."""
    
    def test_watch_commands_exist(self):
        """Test that watch CLI commands are properly registered."""
        from workspace_qdrant_mcp.cli.commands.watch import watch_app
        
        # Get command names
        command_names = [cmd.name for cmd in watch_app.registered_commands.values()]
        
        expected_commands = ["add", "list", "remove", "status", "pause", "resume", "sync"]
        for expected in expected_commands:
            assert expected in command_names
    
    def test_async_helper(self):
        """Test the async helper function."""
        from workspace_qdrant_mcp.cli.commands.watch import handle_async
        
        async def dummy_coro():
            return "success"
        
        result = handle_async(dummy_coro())
        assert result == "success"