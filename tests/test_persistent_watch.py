"""
Comprehensive tests for persistent folder watching functionality.

Tests cover persistence across server restarts, configuration validation,
error recovery scenarios, concurrent access handling, and integration
with the file watcher system.
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from workspace_qdrant_mcp.core.watch_config import (
    PersistentWatchConfigManager,
    WatchConfigurationPersistent,
)
from workspace_qdrant_mcp.core.persistent_file_watcher import (
    PersistentFileWatcher,
    PersistentWatchManager,
)
from workspace_qdrant_mcp.core.watch_validation import (
    WatchPathValidator,
    WatchErrorRecovery,
    WatchHealthMonitor,
)
from workspace_qdrant_mcp.core.watch_sync import (
    SynchronizedWatchConfigManager,
    ConfigChangeEvent,
    FileLockManager,
    WatchEventNotifier,
)
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager


class TestPersistentWatchConfiguration:
    """Test persistent watch configuration management."""
    
    def setup_method(self):
        """Set up test environment with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_watches.json"
        self.config_manager = PersistentWatchConfigManager(str(self.config_file))
        
        # Create test watch directories
        self.watch_dir1 = Path(self.temp_dir) / "watch1"
        self.watch_dir2 = Path(self.temp_dir) / "watch2"
        self.watch_dir1.mkdir()
        self.watch_dir2.mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_config_persistence_basic(self):
        """Test basic configuration persistence operations."""
        # Create watch configuration
        config = WatchConfigurationPersistent(
            id="test-watch-1",
            path=str(self.watch_dir1),
            collection="test-collection",
            patterns=["*.txt", "*.md"],
            auto_ingest=True,
            recursive=True,
        )
        
        # Add configuration
        success = await self.config_manager.add_watch_config(config)
        assert success
        assert self.config_file.exists()
        
        # Load and verify
        configs = await self.config_manager.list_watch_configs()
        assert len(configs) == 1
        assert configs[0].id == "test-watch-1"
        assert configs[0].path == str(self.watch_dir1)
        
        # Update configuration
        config.debounce_seconds = 15
        config.patterns = ["*.pdf"]
        success = await self.config_manager.update_watch_config(config)
        assert success
        
        # Verify update persisted
        updated_config = await self.config_manager.get_watch_config("test-watch-1")
        assert updated_config.debounce_seconds == 15
        assert updated_config.patterns == ["*.pdf"]
        
        # Remove configuration
        success = await self.config_manager.remove_watch_config("test-watch-1")
        assert success
        
        configs = await self.config_manager.list_watch_configs()
        assert len(configs) == 0
    
    async def test_config_file_corruption_recovery(self):
        """Test recovery from corrupted configuration files."""
        # Create valid configuration first
        config = WatchConfigurationPersistent(
            id="test-watch-1",
            path=str(self.watch_dir1),
            collection="test-collection"
        )
        await self.config_manager.add_watch_config(config)
        
        # Corrupt the config file
        with open(self.config_file, 'w') as f:
            f.write("invalid json content {")
        
        # Should recover gracefully
        configs = await self.config_manager.list_watch_configs()
        assert len(configs) == 0  # Fresh start due to corruption
        
        # Should still be able to add new config
        config2 = WatchConfigurationPersistent(
            id="test-watch-2",
            path=str(self.watch_dir2),
            collection="test-collection"
        )
        success = await self.config_manager.add_watch_config(config2)
        assert success
    
    async def test_concurrent_config_access(self):
        """Test concurrent access to configuration files."""
        config1 = WatchConfigurationPersistent(
            id="test-watch-1",
            path=str(self.watch_dir1),
            collection="collection1"
        )
        config2 = WatchConfigurationPersistent(
            id="test-watch-2",
            path=str(self.watch_dir2),
            collection="collection2"
        )
        
        # Simulate concurrent additions
        tasks = [
            self.config_manager.add_watch_config(config1),
            self.config_manager.add_watch_config(config2),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Both should succeed
        assert all(isinstance(r, bool) and r for r in results)
        
        # Verify both configs are present
        configs = await self.config_manager.list_watch_configs()
        assert len(configs) == 2
        
        config_ids = {c.id for c in configs}
        assert config_ids == {"test-watch-1", "test-watch-2"}
    
    async def test_invalid_path_validation(self):
        """Test validation of invalid watch paths."""
        # Test non-existent path
        config = WatchConfigurationPersistent(
            id="test-invalid",
            path="/non/existent/path",
            collection="test-collection"
        )
        
        issues = config.validate()
        assert len(issues) > 0
        assert any("does not exist" in issue for issue in issues)
        
        # Test file instead of directory
        test_file = Path(self.temp_dir) / "test_file.txt"
        test_file.write_text("test content")
        
        config.path = str(test_file)
        issues = config.validate()
        assert len(issues) > 0
        assert any("not a directory" in issue for issue in issues)


class TestWatchPathValidation:
    """Test watch path validation functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "test_watch"
        self.test_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_valid_directory_validation(self):
        """Test validation of valid directories."""
        result = WatchPathValidator.validate_watch_path(self.test_dir)
        
        assert result.valid
        assert result.error_code is None
        assert result.error_message is None
        assert "resolved_path" in result.metadata
    
    def test_nonexistent_path_validation(self):
        """Test validation of non-existent paths."""
        nonexistent_path = self.test_dir / "does_not_exist"
        result = WatchPathValidator.validate_watch_path(nonexistent_path)
        
        assert not result.valid
        assert result.error_code == "PATH_NOT_EXISTS"
        assert "does not exist" in result.error_message
    
    def test_file_instead_of_directory(self):
        """Test validation when path is a file, not a directory."""
        test_file = self.test_dir / "test_file.txt"
        test_file.write_text("test content")
        
        result = WatchPathValidator.validate_watch_path(test_file)
        
        assert not result.valid
        assert result.error_code == "PATH_NOT_DIRECTORY"
        assert "not a directory" in result.error_message
    
    def test_symlink_validation(self):
        """Test validation of symbolic links."""
        # Create target directory and symlink
        target_dir = self.test_dir / "target"
        target_dir.mkdir()
        
        symlink_path = self.test_dir / "symlink"
        symlink_path.symlink_to(target_dir)
        
        result = WatchPathValidator.validate_watch_path(symlink_path)
        
        assert result.valid
        assert len(result.warnings) > 0
        assert any("symbolic link" in warning for warning in result.warnings)
        assert result.metadata.get("is_symlink", False)
        assert result.metadata.get("symlink_target") == str(target_dir.resolve())
    
    def test_broken_symlink_validation(self):
        """Test validation of broken symbolic links."""
        # Create symlink to non-existent target
        nonexistent_target = self.test_dir / "nonexistent"
        symlink_path = self.test_dir / "broken_symlink"
        
        # Create symlink
        symlink_path.symlink_to(nonexistent_target)
        
        result = WatchPathValidator.validate_watch_path(symlink_path)
        
        assert not result.valid
        assert result.error_code == "SYMLINK_BROKEN"
        assert "non-existent target" in result.error_message
    
    @pytest.mark.skipif(os.name == 'nt', reason="Permission tests require Unix-like system")
    def test_permission_validation(self):
        """Test permission validation (Unix systems only)."""
        # Create directory with no read permission
        no_read_dir = self.test_dir / "no_read"
        no_read_dir.mkdir()
        no_read_dir.chmod(0o000)
        
        try:
            result = WatchPathValidator.validate_permissions(no_read_dir)
            
            assert not result.valid
            assert "not readable" in result.error_message.lower() or "permission" in result.error_message.lower()
        finally:
            # Restore permissions for cleanup
            no_read_dir.chmod(0o755)


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "test_watch"
        self.error_recovery = WatchErrorRecovery()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_path_recreation_recovery(self):
        """Test recovery when watch path is recreated."""
        watch_id = "test-recovery-watch"
        
        # Initial recovery attempt for non-existent path
        success, details = await self.error_recovery.attempt_recovery(
            watch_id, "PATH_NOT_EXISTS", self.test_dir, "Path was deleted"
        )
        
        assert not success
        assert "not accessible" in details
        
        # Create the directory
        self.test_dir.mkdir(parents=True)
        
        # Retry recovery
        success, details = await self.error_recovery.attempt_recovery(
            watch_id, "PATH_NOT_EXISTS", self.test_dir, "Path was recreated"
        )
        
        assert success
        assert "now accessible" in details
    
    async def test_recovery_backoff(self):
        """Test progressive backoff in recovery attempts."""
        watch_id = "test-backoff-watch"
        
        # Make multiple recovery attempts
        start_time = time.time()
        
        for i in range(3):
            success, details = await self.error_recovery.attempt_recovery(
                watch_id, "PATH_NOT_EXISTS", self.test_dir, f"Attempt {i+1}"
            )
            assert not success
        
        elapsed_time = time.time() - start_time
        
        # Should have taken some time due to backoff (1 + 2 + 5 = 8 seconds minimum)
        assert elapsed_time >= 3  # At least some backoff occurred
        
        # Check recovery history
        history = self.error_recovery.get_recovery_history(watch_id)
        assert len(history) == 3
        assert all(not attempt["success"] for attempt in history)
    
    async def test_max_recovery_attempts(self):
        """Test maximum recovery attempts limit."""
        watch_id = "test-max-attempts"
        
        # Make maximum number of attempts
        for i in range(self.error_recovery.max_recovery_attempts):
            success, details = await self.error_recovery.attempt_recovery(
                watch_id, "PATH_NOT_EXISTS", self.test_dir, f"Attempt {i+1}"
            )
            assert not success
        
        # Additional attempt should be rejected
        success, details = await self.error_recovery.attempt_recovery(
            watch_id, "PATH_NOT_EXISTS", self.test_dir, "Exceeded attempt"
        )
        
        assert not success
        assert "Maximum recovery attempts" in details


class TestHealthMonitoring:
    """Test health monitoring functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "test_watch"
        self.test_dir.mkdir()
        
        self.error_recovery = WatchErrorRecovery()
        self.health_monitor = WatchHealthMonitor(self.error_recovery)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_health_check_registration(self):
        """Test watch registration for health monitoring."""
        watch_id = "test-health-watch"
        
        # Register watch
        self.health_monitor.register_watch(watch_id, self.test_dir)
        
        # Verify registration
        health_status = self.health_monitor.get_health_status(watch_id)
        assert health_status is not None
        assert health_status["path"] == self.test_dir
        assert health_status["status"] == "unknown"
        assert health_status["consecutive_failures"] == 0
    
    async def test_health_monitoring_lifecycle(self):
        """Test health monitoring start/stop lifecycle."""
        watch_id = "test-lifecycle-watch"
        self.health_monitor.register_watch(watch_id, self.test_dir)
        
        # Start monitoring
        assert not self.health_monitor.is_monitoring()
        await self.health_monitor.start_monitoring()
        assert self.health_monitor.is_monitoring()
        
        # Let monitoring run briefly
        await asyncio.sleep(0.1)
        
        # Stop monitoring
        await self.health_monitor.stop_monitoring()
        assert not self.health_monitor.is_monitoring()
    
    async def test_health_status_updates(self):
        """Test health status updates during monitoring."""
        watch_id = "test-status-watch"
        self.health_monitor.register_watch(watch_id, self.test_dir)
        
        # Perform manual health check
        await self.health_monitor._perform_health_checks()
        
        # Verify status was updated
        health_status = self.health_monitor.get_health_status(watch_id)
        assert health_status["status"] == "healthy"
        assert health_status["last_check"] is not None
        assert health_status["consecutive_failures"] == 0


class TestSynchronization:
    """Test configuration synchronization and file locking."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "sync_test.json"
        self.sync_manager = SynchronizedWatchConfigManager(self.config_file)
        
        self.test_dir = Path(self.temp_dir) / "test_watch"
        self.test_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_file_locking(self):
        """Test file locking mechanism."""
        lock_manager = FileLockManager(self.config_file)
        
        # Acquire lock
        async with lock_manager.acquire_lock():
            # Lock should be acquired
            assert lock_manager.lock_file.exists()
            
            # Try to acquire another lock (should timeout quickly for testing)
            lock_manager2 = FileLockManager(self.config_file)
            lock_manager2._lock_timeout = 0.1  # Very short timeout
            
            with pytest.raises(TimeoutError):
                async with lock_manager2.acquire_lock():
                    pass
        
        # Lock should be released
        assert not lock_manager.lock_file.exists()
    
    async def test_event_notifications(self):
        """Test configuration change event notifications."""
        await self.sync_manager.initialize()
        
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        # Subscribe to events
        self.sync_manager.subscribe_to_changes(event_handler)
        
        # Create watch configuration
        config = WatchConfigurationPersistent(
            id="test-sync-watch",
            path=str(self.test_dir),
            collection="test-collection"
        )
        
        # Add configuration (should trigger event)
        await self.sync_manager.add_watch_config(config)
        
        # Give event system time to process
        await asyncio.sleep(0.1)
        
        # Verify event was received
        assert len(events_received) == 1
        event = events_received[0]
        assert event.event_type == "added"
        assert event.watch_id == "test-sync-watch"
        assert event.new_config is not None
        
        # Update configuration
        config.debounce_seconds = 10
        await self.sync_manager.update_watch_config(config)
        await asyncio.sleep(0.1)
        
        # Should have received update event
        assert len(events_received) == 2
        update_event = events_received[1]
        assert update_event.event_type == "modified"
        assert update_event.watch_id == "test-sync-watch"
    
    async def test_config_caching(self):
        """Test configuration caching behavior."""
        await self.sync_manager.initialize()
        
        # Load config multiple times
        config1 = await self.sync_manager.load_config()
        config2 = await self.sync_manager.load_config()
        
        # Should use cached version
        assert config1 is config2
        
        # Force reload should get new instance
        config3 = await self.sync_manager.load_config(force_reload=True)
        # Note: Content should be same, but might be different object
        assert len(config3.watches) == len(config1.watches)


class TestIntegration:
    """Integration tests for complete persistent watching system."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.watch_dir = Path(self.temp_dir) / "watch_target"
        self.watch_dir.mkdir()
        
        # Mock workspace client
        self.mock_workspace_client = MagicMock()
        self.mock_workspace_client.list_collections.return_value = asyncio.Future()
        self.mock_workspace_client.list_collections.return_value.set_result(["test-collection"])
        
        self.watch_tools_manager = WatchToolsManager(self.mock_workspace_client)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_end_to_end_watch_lifecycle(self):
        """Test complete watch lifecycle from creation to cleanup."""
        # Initialize the watch tools manager
        await self.watch_tools_manager.initialize()
        
        # Add watch folder
        result = await self.watch_tools_manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            patterns=["*.txt"],
            auto_ingest=False,  # Disable to avoid needing real ingestion
        )
        
        assert result["success"]
        watch_id = result["watch_id"]
        
        # List watches
        list_result = await self.watch_tools_manager.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == 1
        assert list_result["watches"][0]["id"] == watch_id
        
        # Configure watch settings
        config_result = await self.watch_tools_manager.configure_watch_settings(
            watch_id=watch_id,
            debounce_seconds=15,
            patterns=["*.md", "*.txt"],
        )
        assert config_result["success"]
        
        # Get watch status
        status_result = await self.watch_tools_manager.get_watch_status(watch_id)
        assert status_result["success"]
        assert status_result["runtime_info"] is not None
        
        # Remove watch
        remove_result = await self.watch_tools_manager.remove_watch_folder(watch_id)
        assert remove_result["success"]
        
        # Verify removal
        list_result = await self.watch_tools_manager.list_watched_folders()
        assert len(list_result["watches"]) == 0
        
        # Cleanup
        await self.watch_tools_manager.cleanup()
    
    async def test_persistence_across_restarts(self):
        """Test that watch configurations persist across manager restarts."""
        # Initialize and add watch
        await self.watch_tools_manager.initialize()
        
        result = await self.watch_tools_manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Cleanup current manager
        await self.watch_tools_manager.cleanup()
        
        # Create new manager instance (simulating restart)
        new_manager = WatchToolsManager(self.mock_workspace_client)
        await new_manager.initialize()
        
        # Verify watch was recovered
        list_result = await new_manager.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == 1
        assert list_result["watches"][0]["id"] == watch_id
        
        # Cleanup new manager
        await new_manager.cleanup()
    
    async def test_invalid_configuration_handling(self):
        """Test handling of invalid configurations."""
        await self.watch_tools_manager.initialize()
        
        # Try to add watch with non-existent path
        result = await self.watch_tools_manager.add_watch_folder(
            path="/non/existent/path",
            collection="test-collection",
        )
        
        assert not result["success"]
        assert "validation" in result.get("error_type", "").lower() or "not_found" in result.get("error_type", "")
        
        # Try to add watch with invalid collection
        self.mock_workspace_client.list_collections.return_value = asyncio.Future()
        self.mock_workspace_client.list_collections.return_value.set_result(["other-collection"])
        
        result = await self.watch_tools_manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="nonexistent-collection",
        )
        
        assert not result["success"]
        assert "collection_not_found" in result.get("error_type", "")
        
        await self.watch_tools_manager.cleanup()
    
    @patch('workspace_qdrant_mcp.tools.documents.add_document')
    async def test_file_ingestion_integration(self, mock_add_document):
        """Test integration with file ingestion system."""
        # Mock successful document addition
        mock_add_document.return_value = asyncio.Future()
        mock_add_document.return_value.set_result({"success": True, "document_id": "test-doc"})
        
        await self.watch_tools_manager.initialize()
        
        # Add watch with auto-ingestion enabled
        result = await self.watch_tools_manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            patterns=["*.txt"],
            auto_ingest=True,
        )
        assert result["success"]
        
        # Create a test file (would trigger ingestion in real scenario)
        test_file = self.watch_dir / "test.txt"
        test_file.write_text("Test content")
        
        # In integration test, we verify the system is set up correctly
        # Real file watching would be tested in e2e tests
        
        await self.watch_tools_manager.cleanup()


class TestAdvancedConfiguration:
    """Test advanced configuration options."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.watch_dir = Path(self.temp_dir) / "advanced_watch"
        self.watch_dir.mkdir()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_recursive_depth_configuration(self):
        """Test recursive depth limit configuration."""
        from workspace_qdrant_mcp.core.advanced_watch_config import RecursiveConfig
        
        # Test valid configurations
        config = RecursiveConfig(enabled=True, max_depth=5)
        assert config.enabled
        assert config.max_depth == 5
        
        # Test unlimited depth
        config = RecursiveConfig(enabled=True, max_depth=-1)
        assert config.max_depth == -1
        
        # Test validation limits
        with pytest.raises(Exception):  # Should fail validation
            RecursiveConfig(enabled=True, max_depth=25)  # Exceeds max limit
    
    def test_file_filter_configuration(self):
        """Test file filtering configuration."""
        from workspace_qdrant_mcp.core.advanced_watch_config import FileFilterConfig
        
        # Test valid filter config
        config = FileFilterConfig(
            include_patterns=["*.pdf", "*.txt"],
            exclude_patterns=["*.tmp", "*~"],
            size_limits={"min_bytes": 1024, "max_bytes": 10*1024*1024},
            regex_patterns={"include": r".*\.(pdf|txt)$", "exclude": r".*\.tmp$"}
        )
        
        assert len(config.include_patterns) == 2
        assert config.size_limits["min_bytes"] == 1024
        
        # Test invalid regex
        with pytest.raises(Exception):
            FileFilterConfig(
                regex_patterns={"include": "[invalid regex"}
            )
    
    def test_performance_configuration(self):
        """Test performance configuration options."""
        from workspace_qdrant_mcp.core.advanced_watch_config import PerformanceConfig
        
        config = PerformanceConfig(
            update_frequency_ms=2000,
            debounce_seconds=10,
            batch_processing=True,
            batch_size=5,
            max_concurrent_ingestions=3
        )
        
        assert config.update_frequency_ms == 2000
        assert config.batch_processing
        assert config.max_concurrent_ingestions == 3
        
        # Test validation limits
        with pytest.raises(Exception):
            PerformanceConfig(update_frequency_ms=50)  # Below minimum
        
        with pytest.raises(Exception):
            PerformanceConfig(debounce_seconds=500)  # Above maximum
    
    def test_collection_targeting(self):
        """Test collection targeting and routing."""
        from workspace_qdrant_mcp.core.advanced_watch_config import CollectionTargeting
        
        config = CollectionTargeting(
            default_collection="main-docs",
            routing_rules=[
                {"pattern": "*.pdf", "collection": "pdf-docs", "type": "glob"},
                {"pattern": r".*\.log$", "collection": "logs", "type": "regex"},
                {"pattern": ".md", "collection": "markdown", "type": "extension"}
            ],
            collection_prefixes={"extension": "ext-", "directory": "dir-"}
        )
        
        assert config.default_collection == "main-docs"
        assert len(config.routing_rules) == 3
        assert config.collection_prefixes["extension"] == "ext-"


if __name__ == "__main__":
    pytest.main([__file__])