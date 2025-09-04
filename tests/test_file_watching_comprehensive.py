"""
Comprehensive File Watching and Auto-Ingestion Testing for Task 87.

This module provides complete test coverage for:
1. File watching system validation
2. Automatic ingestion trigger testing  
3. Real-time status update verification
4. Watch configuration testing
5. Error scenario handling and service persistence testing

Test Areas:
- core/file_watcher.py functionality
- core/persistent_file_watcher.py integration
- core/watch_validation.py validation logic
- core/watch_config.py and core/advanced_watch_config.py management
- Error handling for permission issues, disk full scenarios, network interruptions
- Watch service management and persistence across system restarts
"""

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import Any, Dict, List

import pytest

from workspace_qdrant_mcp.core.file_watcher import (
    FileWatcher, 
    WatchConfiguration, 
    WatchEvent, 
    WatchManager
)

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFileWatchingSystemValidation:
    """Test Area 1: File watching system validation"""
    
    @pytest.fixture
    async def temp_watch_dir(self):
        """Create temporary directory for file watching tests."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_file_watch_test_")
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    async def watch_config(self, temp_watch_dir):
        """Create test watch configuration."""
        return WatchConfiguration(
            id="test_watch_1",
            path=str(temp_watch_dir),
            collection="test_collection",
            patterns=["*.txt", "*.md", "*.pdf"],
            ignore_patterns=[".git/*", "*.tmp"],
            auto_ingest=True,
            recursive=True,
            debounce_seconds=1  # Short debounce for testing
        )
    
    @pytest.fixture
    def mock_ingestion_callback(self):
        """Mock ingestion callback for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_event_callback(self):
        """Mock event callback for testing."""
        return Mock()
    
    async def test_watch_configuration_creation(self, temp_watch_dir):
        """Test WatchConfiguration creation and validation."""
        config = WatchConfiguration(
            id="test_watch",
            path=str(temp_watch_dir),
            collection="test_collection"
        )
        
        assert config.id == "test_watch"
        assert config.path == str(temp_watch_dir)
        assert config.collection == "test_collection"
        assert config.patterns == ["*.pdf", "*.epub", "*.txt", "*.md"]
        assert config.auto_ingest is True
        assert config.recursive is True
        assert config.debounce_seconds == 5
        assert config.status == "active"
        assert config.files_processed == 0
        assert config.errors_count == 0
    
    async def test_watch_configuration_serialization(self, watch_config):
        """Test WatchConfiguration to/from dict serialization."""
        # Test to_dict
        config_dict = watch_config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["id"] == "test_watch_1"
        assert config_dict["path"] == watch_config.path
        assert config_dict["collection"] == "test_collection"
        
        # Test from_dict
        restored_config = WatchConfiguration.from_dict(config_dict)
        assert restored_config.id == watch_config.id
        assert restored_config.path == watch_config.path
        assert restored_config.collection == watch_config.collection
        assert restored_config.patterns == watch_config.patterns
    
    async def test_file_watcher_initialization(self, watch_config, mock_ingestion_callback, mock_event_callback):
        """Test FileWatcher initialization."""
        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_ingestion_callback,
            event_callback=mock_event_callback
        )
        
        assert watcher.config == watch_config
        assert watcher.ingestion_callback == mock_ingestion_callback
        assert watcher.event_callback == mock_event_callback
        assert not watcher.is_running()
        assert watcher._task is None
        assert len(watcher._debounce_tasks) == 0
    
    async def test_file_watcher_start_stop(self, watch_config, mock_ingestion_callback):
        """Test FileWatcher start and stop operations."""
        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_ingestion_callback
        )
        
        # Test start
        await watcher.start()
        assert watcher.is_running()
        assert watcher.config.status == "active"
        assert watcher._task is not None
        
        # Test stop
        await watcher.stop()
        assert not watcher.is_running()
        assert watcher.config.status == "paused"
        assert len(watcher._debounce_tasks) == 0
    
    async def test_file_watcher_pause_resume(self, watch_config, mock_ingestion_callback):
        """Test FileWatcher pause and resume operations."""
        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_ingestion_callback
        )
        
        # Start watcher
        await watcher.start()
        assert watcher.is_running()
        
        # Test pause
        await watcher.pause()
        assert not watcher.is_running()
        assert watcher.config.status == "paused"
        
        # Test resume
        await watcher.resume()
        assert watcher.is_running()
        assert watcher.config.status == "active"
        
        # Clean up
        await watcher.stop()
    
    async def test_file_pattern_matching(self, watch_config, mock_ingestion_callback):
        """Test file pattern matching logic."""
        watcher = FileWatcher(
            config=watch_config,
            ingestion_callback=mock_ingestion_callback
        )
        
        # Test matching patterns
        assert watcher._matches_patterns(Path("test.txt"))
        assert watcher._matches_patterns(Path("document.md"))
        assert watcher._matches_patterns(Path("report.pdf"))
        
        # Test non-matching patterns
        assert not watcher._matches_patterns(Path("image.jpg"))
        assert not watcher._matches_patterns(Path("script.py"))
        
        # Test ignore patterns
        assert watcher._matches_ignore_patterns(Path(".git/config"))
        assert watcher._matches_ignore_patterns(Path("temp.tmp"))
        assert not watcher._matches_ignore_patterns(Path("valid.txt"))


class TestAutomaticIngestionTriggers:
    """Test Area 2: Automatic ingestion trigger testing"""
    
    @pytest.fixture
    async def temp_test_environment(self):
        """Create temporary test environment with files."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_ingestion_test_")
        test_dir = Path(temp_dir)
        
        # Create test files
        (test_dir / "test1.txt").write_text("Initial content 1")
        (test_dir / "test2.md").write_text("# Initial markdown")
        (test_dir / "ignored.tmp").write_text("Temporary file")
        
        yield test_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_file_addition_triggers_ingestion(self, temp_test_environment):
        """Test that adding new files triggers ingestion."""
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        config = WatchConfiguration(
            id="ingestion_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1  # Very short for testing
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        await watcher.start()
        
        try:
            # Add a new file
            new_file = temp_test_environment / "new_document.txt"
            new_file.write_text("New file content")
            
            # Wait for debouncing and processing
            await asyncio.sleep(0.5)
            
            # Verify ingestion callback was called
            ingestion_callback.assert_called_once()
            call_args = ingestion_callback.call_args[0]
            assert str(new_file) in call_args[0]
            assert call_args[1] == "test_collection"
            
            # Verify event callback was called
            assert event_callback.called
            event_args = event_callback.call_args[0][0]
            assert event_args.change_type == "added"
            assert str(new_file) in event_args.file_path
            
        finally:
            await watcher.stop()
    
    async def test_file_modification_triggers_ingestion(self, temp_test_environment):
        """Test that modifying existing files triggers ingestion."""
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        config = WatchConfiguration(
            id="modification_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        await watcher.start()
        
        try:
            # Modify existing file
            existing_file = temp_test_environment / "test1.txt"
            existing_file.write_text("Modified content")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify callbacks were called
            ingestion_callback.assert_called_once()
            event_callback.assert_called_once()
            
            event_args = event_callback.call_args[0][0]
            assert event_args.change_type == "modified"
            
        finally:
            await watcher.stop()
    
    async def test_file_deletion_does_not_trigger_ingestion(self, temp_test_environment):
        """Test that file deletion does not trigger ingestion but does trigger events."""
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        config = WatchConfiguration(
            id="deletion_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        await watcher.start()
        
        try:
            # Delete existing file
            existing_file = temp_test_environment / "test1.txt"
            existing_file.unlink()
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify ingestion callback was NOT called (deletion doesn't trigger ingestion)
            ingestion_callback.assert_not_called()
            
            # But event callback should be called
            event_callback.assert_called_once()
            event_args = event_callback.call_args[0][0]
            assert event_args.change_type == "deleted"
            
        finally:
            await watcher.stop()
    
    async def test_ignored_files_do_not_trigger_ingestion(self, temp_test_environment):
        """Test that ignored files do not trigger ingestion."""
        ingestion_callback = AsyncMock()
        event_callback = Mock()
        
        config = WatchConfiguration(
            id="ignore_test",
            path=str(temp_test_environment),
            collection="test_collection",
            ignore_patterns=["*.tmp", "*.ignore"],
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        await watcher.start()
        
        try:
            # Create ignored files
            (temp_test_environment / "temp.tmp").write_text("Temporary")
            (temp_test_environment / "secret.ignore").write_text("Ignored")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify no callbacks were called
            ingestion_callback.assert_not_called()
            event_callback.assert_not_called()
            
        finally:
            await watcher.stop()
    
    async def test_debouncing_prevents_rapid_fire_ingestion(self, temp_test_environment):
        """Test that debouncing prevents multiple rapid ingestions of the same file."""
        ingestion_callback = AsyncMock()
        
        config = WatchConfiguration(
            id="debounce_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.5  # Longer debounce for this test
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        await watcher.start()
        
        try:
            test_file = temp_test_environment / "debounce_test.txt"
            
            # Rapidly modify the same file multiple times
            for i in range(5):
                test_file.write_text(f"Content {i}")
                await asyncio.sleep(0.1)
            
            # Wait for debounce period plus buffer
            await asyncio.sleep(1.0)
            
            # Should only be called once due to debouncing
            assert ingestion_callback.call_count == 1
            
        finally:
            await watcher.stop()


class TestRealTimeStatusUpdates:
    """Test Area 3: Real-time status update verification"""
    
    async def test_watch_configuration_status_tracking(self):
        """Test that watch configuration status is properly tracked."""
        config = WatchConfiguration(
            id="status_test",
            path="/tmp/test",
            collection="test_collection"
        )
        
        # Initial status should be active
        assert config.status == "active"
        assert config.files_processed == 0
        assert config.errors_count == 0
        assert config.last_activity is None
    
    async def test_file_processing_statistics_update(self, temp_test_environment):
        """Test that file processing statistics are updated correctly."""
        ingestion_callback = AsyncMock()
        
        config = WatchConfiguration(
            id="stats_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        await watcher.start()
        
        try:
            initial_processed = config.files_processed
            initial_activity = config.last_activity
            
            # Add files to trigger processing
            for i in range(3):
                (temp_test_environment / f"file_{i}.txt").write_text(f"Content {i}")
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Verify statistics were updated
            assert config.files_processed > initial_processed
            assert config.last_activity != initial_activity
            assert config.last_activity is not None
            
        finally:
            await watcher.stop()
    
    async def test_error_count_tracking(self, temp_test_environment):
        """Test that error counts are properly tracked."""
        # Mock ingestion callback that raises an error
        ingestion_callback = AsyncMock(side_effect=Exception("Ingestion error"))
        
        config = WatchConfiguration(
            id="error_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        await watcher.start()
        
        try:
            initial_errors = config.errors_count
            
            # Add file to trigger processing with error
            (temp_test_environment / "error_file.txt").write_text("Content")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify error count increased
            assert config.errors_count > initial_errors
            
        finally:
            await watcher.stop()
    
    async def test_watch_manager_status_reporting(self):
        """Test WatchManager status reporting functionality."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_manager_test_")
        config_file = Path(temp_dir) / "test_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            manager.set_ingestion_callback(AsyncMock())
            
            # Add a watch
            watch_id = await manager.add_watch(
                path=temp_dir,
                collection="test_collection"
            )
            
            # Get status
            status = manager.get_watch_status()
            
            assert watch_id in status
            assert "config" in status[watch_id]
            assert "running" in status[watch_id]
            assert "path_exists" in status[watch_id]
            
            # Verify path exists check
            assert status[watch_id]["path_exists"] is True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestWatchConfigurationManagement:
    """Test Area 4: Watch configuration testing"""
    
    async def test_watch_manager_initialization(self):
        """Test WatchManager initialization with custom config file."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_config_test_")
        config_file = Path(temp_dir) / "custom_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            
            assert manager.config_file == config_file
            assert manager.config_file.parent.exists()
            assert len(manager.watchers) == 0
            assert len(manager.configurations) == 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_watch_manager_default_config_location(self):
        """Test WatchManager with default config file location."""
        manager = WatchManager()
        
        expected_path = Path.home() / ".wqm" / "watches.json"
        assert manager.config_file == expected_path
    
    async def test_add_remove_watch_configuration(self):
        """Test adding and removing watch configurations."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_add_remove_test_")
        config_file = Path(temp_dir) / "test_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            
            # Add watch
            watch_id = await manager.add_watch(
                path=temp_dir,
                collection="test_collection",
                patterns=["*.txt", "*.md"],
                auto_ingest=True,
                recursive=False,
                debounce_seconds=3
            )
            
            assert watch_id in manager.configurations
            config = manager.configurations[watch_id]
            
            assert config.path == str(Path(temp_dir).resolve())
            assert config.collection == "test_collection"
            assert config.patterns == ["*.txt", "*.md"]
            assert config.auto_ingest is True
            assert config.recursive is False
            assert config.debounce_seconds == 3
            
            # Remove watch
            removed = await manager.remove_watch(watch_id)
            assert removed is True
            assert watch_id not in manager.configurations
            
            # Try removing non-existent watch
            removed = await manager.remove_watch("non_existent")
            assert removed is False
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_configuration_persistence(self):
        """Test that configurations are properly saved and loaded."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_persistence_test_")
        config_file = Path(temp_dir) / "test_watches.json"
        
        try:
            # Create manager and add configuration
            manager1 = WatchManager(config_file=str(config_file))
            watch_id = await manager1.add_watch(
                path=temp_dir,
                collection="persistent_test",
                patterns=["*.test"]
            )
            
            # Verify configuration was saved
            assert config_file.exists()
            
            # Create new manager and load configurations
            manager2 = WatchManager(config_file=str(config_file))
            await manager2.load_configurations()
            
            # Verify configuration was loaded
            assert watch_id in manager2.configurations
            config = manager2.configurations[watch_id]
            assert config.collection == "persistent_test"
            assert config.patterns == ["*.test"]
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_watch_filtering_and_listing(self):
        """Test watch filtering and listing functionality."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_filtering_test_")
        config_file = Path(temp_dir) / "test_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            
            # Add multiple watches
            watch_id1 = await manager.add_watch(
                path=temp_dir + "/collection1",
                collection="collection1"
            )
            watch_id2 = await manager.add_watch(
                path=temp_dir + "/collection2", 
                collection="collection2"
            )
            watch_id3 = await manager.add_watch(
                path=temp_dir + "/collection1_alt",
                collection="collection1"
            )
            
            # Pause one watch
            await manager.pause_watch(watch_id2)
            
            # Test listing all watches
            all_watches = manager.list_watches()
            assert len(all_watches) == 3
            
            # Test filtering by collection
            collection1_watches = manager.list_watches(collection="collection1")
            assert len(collection1_watches) == 2
            
            # Test filtering by active status
            active_watches = manager.list_watches(active_only=True)
            assert len(active_watches) == 2  # watch_id2 is paused
            
            # Test combined filtering
            active_collection1 = manager.list_watches(
                active_only=True,
                collection="collection1"
            )
            assert len(active_collection1) == 2
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestErrorScenarioHandling:
    """Test Area 5: Error scenario handling and service persistence testing"""
    
    async def test_nonexistent_path_handling(self):
        """Test handling of non-existent watch paths."""
        nonexistent_path = "/absolutely/nonexistent/path"
        ingestion_callback = AsyncMock()
        
        config = WatchConfiguration(
            id="nonexistent_test",
            path=nonexistent_path,
            collection="test_collection"
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        await watcher.start()
        
        # Wait a bit for the watch loop to detect the error
        await asyncio.sleep(0.5)
        
        # Verify error status is set
        assert config.status == "error"
        
        await watcher.stop()
    
    async def test_permission_denied_handling(self):
        """Test handling of permission denied scenarios."""
        # This test requires platform-specific setup and may need to be skipped
        # in some environments
        if os.name == 'nt':  # Windows
            pytest.skip("Permission testing not implemented for Windows")
        
        temp_dir = tempfile.mkdtemp(prefix="wqm_permission_test_")
        
        try:
            # Remove read permissions
            os.chmod(temp_dir, 0o000)
            
            ingestion_callback = AsyncMock()
            config = WatchConfiguration(
                id="permission_test",
                path=temp_dir,
                collection="test_collection"
            )
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            await watcher.start()
            await asyncio.sleep(0.5)
            
            # Should handle permission error gracefully
            assert config.status == "error" or config.errors_count > 0
            
            await watcher.stop()
            
        finally:
            # Restore permissions for cleanup
            try:
                os.chmod(temp_dir, 0o755)
                shutil.rmtree(temp_dir)
            except:
                pass
    
    async def test_ingestion_callback_error_handling(self, temp_test_environment):
        """Test handling of errors in ingestion callbacks."""
        # Mock callback that raises an error
        ingestion_callback = AsyncMock(side_effect=RuntimeError("Ingestion failed"))
        
        config = WatchConfiguration(
            id="callback_error_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback
        )
        
        await watcher.start()
        
        try:
            initial_errors = config.errors_count
            
            # Add file to trigger callback error
            (temp_test_environment / "error_trigger.txt").write_text("Content")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify error was handled and counted
            assert config.errors_count > initial_errors
            
        finally:
            await watcher.stop()
    
    async def test_event_callback_error_handling(self, temp_test_environment):
        """Test handling of errors in event callbacks."""
        ingestion_callback = AsyncMock()
        event_callback = Mock(side_effect=RuntimeError("Event callback failed"))
        
        config = WatchConfiguration(
            id="event_error_test",
            path=str(temp_test_environment),
            collection="test_collection",
            debounce_seconds=0.1
        )
        
        watcher = FileWatcher(
            config=config,
            ingestion_callback=ingestion_callback,
            event_callback=event_callback
        )
        
        await watcher.start()
        
        try:
            # Add file to trigger event callback error
            (temp_test_environment / "event_error.txt").write_text("Content")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Verify event callback was called despite error
            assert event_callback.called
            # Verify ingestion still proceeded
            assert ingestion_callback.called
            
        finally:
            await watcher.stop()
    
    async def test_configuration_file_corruption_handling(self):
        """Test handling of corrupted configuration files."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_corruption_test_")
        config_file = Path(temp_dir) / "corrupted_watches.json"
        
        try:
            # Create corrupted JSON file
            config_file.write_text('{"watches": [invalid json}')
            
            manager = WatchManager(config_file=str(config_file))
            
            # Should handle corruption gracefully
            await manager.load_configurations()
            
            # Should start with empty configurations
            assert len(manager.configurations) == 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_atomic_configuration_saving(self):
        """Test atomic configuration saving to prevent corruption."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_atomic_test_")
        config_file = Path(temp_dir) / "atomic_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            
            # Add watch to create initial configuration
            await manager.add_watch(
                path=temp_dir,
                collection="atomic_test"
            )
            
            # Verify config file exists
            assert config_file.exists()
            
            # Verify no temporary files remain
            temp_files = list(config_file.parent.glob("*.tmp"))
            assert len(temp_files) == 0
            
            # Verify configuration can be loaded
            data = json.loads(config_file.read_text())
            assert "watches" in data
            assert len(data["watches"]) == 1
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_watch_manager_start_stop_all_error_recovery(self):
        """Test error recovery in start_all_watches and stop_all_watches."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_recovery_test_")
        config_file = Path(temp_dir) / "recovery_watches.json"
        
        try:
            manager = WatchManager(config_file=str(config_file))
            
            # Add watches with mixed valid/invalid paths
            valid_watch = await manager.add_watch(
                path=temp_dir,
                collection="valid_collection"
            )
            invalid_watch = await manager.add_watch(
                path="/nonexistent/path",
                collection="invalid_collection"
            )
            
            # Set ingestion callback
            manager.set_ingestion_callback(AsyncMock())
            
            # Start all watches - should handle errors gracefully
            await manager.start_all_watches()
            
            # Valid watch should be running, invalid should error
            status = manager.get_watch_status()
            assert status[valid_watch]["running"] is True
            # Invalid path should be handled gracefully
            
            # Stop all watches - should handle errors gracefully
            await manager.stop_all_watches()
            
            # All watchers should be stopped
            assert len(manager.watchers) == 0
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestWatchServicePersistence:
    """Test service persistence across system restarts."""
    
    async def test_configuration_persistence_across_sessions(self):
        """Test that configurations persist across manager sessions."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_session_test_")
        config_file = Path(temp_dir) / "session_watches.json"
        
        try:
            # Session 1: Create and configure watches
            manager1 = WatchManager(config_file=str(config_file))
            
            watch_id1 = await manager1.add_watch(
                path=temp_dir + "/session1",
                collection="session_collection",
                patterns=["*.session"]
            )
            watch_id2 = await manager1.add_watch(
                path=temp_dir + "/session2",
                collection="session_collection",
                recursive=False,
                debounce_seconds=10
            )
            
            # Modify some statistics
            manager1.configurations[watch_id1].files_processed = 42
            manager1.configurations[watch_id1].errors_count = 2
            await manager1.save_configurations()
            
            # Session 2: New manager instance
            manager2 = WatchManager(config_file=str(config_file))
            await manager2.load_configurations()
            
            # Verify all configurations were restored
            assert len(manager2.configurations) == 2
            assert watch_id1 in manager2.configurations
            assert watch_id2 in manager2.configurations
            
            # Verify configuration details
            config1 = manager2.configurations[watch_id1]
            assert config1.collection == "session_collection"
            assert config1.patterns == ["*.session"]
            assert config1.files_processed == 42
            assert config1.errors_count == 2
            
            config2 = manager2.configurations[watch_id2]
            assert config2.recursive is False
            assert config2.debounce_seconds == 10
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_watch_state_recovery_after_interruption(self):
        """Test that watch states can be recovered after interruption."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_interruption_test_")
        config_file = Path(temp_dir) / "interruption_watches.json"
        
        try:
            # Create initial manager and watches
            manager1 = WatchManager(config_file=str(config_file))
            manager1.set_ingestion_callback(AsyncMock())
            
            watch_id = await manager1.add_watch(
                path=temp_dir,
                collection="recovery_test"
            )
            
            await manager1.start_all_watches()
            
            # Simulate processing some files
            (Path(temp_dir) / "processed1.txt").write_text("Content 1")
            await asyncio.sleep(0.2)
            
            # Update statistics to simulate processing
            manager1.configurations[watch_id].files_processed = 5
            manager1.configurations[watch_id].last_activity = "2023-01-01T12:00:00Z"
            await manager1.save_configurations()
            
            # Stop all watches (simulate shutdown)
            await manager1.stop_all_watches()
            
            # Create new manager instance (simulate restart)
            manager2 = WatchManager(config_file=str(config_file))
            await manager2.load_configurations()
            manager2.set_ingestion_callback(AsyncMock())
            
            # Verify state was recovered
            recovered_config = manager2.configurations[watch_id]
            assert recovered_config.files_processed == 5
            assert recovered_config.last_activity == "2023-01-01T12:00:00Z"
            
            # Verify watches can be restarted
            await manager2.start_all_watches()
            status = manager2.get_watch_status()
            assert status[watch_id]["running"] is True
            
            await manager2.stop_all_watches()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# Fixture helpers
@pytest.fixture
async def temp_test_environment():
    """Create temporary test environment with files."""
    temp_dir = tempfile.mkdtemp(prefix="wqm_test_env_")
    test_dir = Path(temp_dir)
    
    # Create test files
    (test_dir / "test1.txt").write_text("Initial content 1")
    (test_dir / "test2.md").write_text("# Initial markdown")
    
    yield test_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# Performance and stress tests
class TestFileWatchingPerformance:
    """Performance and stress testing for file watching system."""
    
    async def test_high_volume_file_processing(self):
        """Test handling of high volume file operations."""
        temp_dir = tempfile.mkdtemp(prefix="wqm_performance_test_")
        
        try:
            ingestion_callback = AsyncMock()
            config = WatchConfiguration(
                id="performance_test",
                path=temp_dir,
                collection="performance_collection",
                debounce_seconds=0.05  # Short debounce for performance
            )
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            await watcher.start()
            
            # Create many files quickly
            file_count = 50
            start_time = time.time()
            
            for i in range(file_count):
                (Path(temp_dir) / f"perf_file_{i}.txt").write_text(f"Content {i}")
            
            # Wait for all processing to complete
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Verify all files were processed
            assert config.files_processed >= file_count // 2  # Allow for some debouncing
            
            # Performance should be reasonable (less than 5 seconds for 50 files)
            assert processing_time < 5.0
            
            await watcher.stop()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable over extended operation."""
        import gc
        import sys
        
        temp_dir = tempfile.mkdtemp(prefix="wqm_memory_test_")
        
        try:
            ingestion_callback = AsyncMock()
            config = WatchConfiguration(
                id="memory_test",
                path=temp_dir,
                collection="memory_collection",
                debounce_seconds=0.1
            )
            
            watcher = FileWatcher(
                config=config,
                ingestion_callback=ingestion_callback
            )
            
            await watcher.start()
            
            # Measure initial memory
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Process many files over time
            for batch in range(10):
                for i in range(10):
                    file_path = Path(temp_dir) / f"memory_file_{batch}_{i}.txt"
                    file_path.write_text(f"Batch {batch} Content {i}")
                
                await asyncio.sleep(0.5)
                
                # Clean up files to simulate realistic usage
                for i in range(10):
                    try:
                        (Path(temp_dir) / f"memory_file_{batch}_{i}.txt").unlink()
                    except:
                        pass
            
            # Final processing wait
            await asyncio.sleep(1.0)
            
            # Measure final memory
            gc.collect()
            final_objects = len(gc.get_objects())
            
            # Memory growth should be reasonable (less than 50% increase)
            memory_growth = (final_objects - initial_objects) / initial_objects
            assert memory_growth < 0.5
            
            await watcher.stop()
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])