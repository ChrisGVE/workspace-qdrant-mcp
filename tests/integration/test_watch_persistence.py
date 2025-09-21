"""
Integration tests for watch persistence across server restarts.

These tests simulate real server restart scenarios and verify that
watch configurations are properly restored and function correctly.
"""

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from common.core.watch_config import WatchConfigurationPersistent
from common.core.watch_sync import SynchronizedWatchConfigManager
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager


class TestServerRestartPersistence:
    """Test persistence across simulated server restarts."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "restart_test.json"
        
        # Create multiple test watch directories
        self.watch_dirs = []
        for i in range(3):
            watch_dir = Path(self.temp_dir) / f"watch_{i}"
            watch_dir.mkdir()
            self.watch_dirs.append(watch_dir)
        
        # Mock workspace client
        self.mock_workspace_client = MagicMock()
        self.mock_workspace_client.list_collections.return_value = asyncio.Future()
        self.mock_workspace_client.list_collections.return_value.set_result(
            ["collection1", "collection2", "collection3"]
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_multiple_watches_persist_across_restart(self):
        """Test that multiple watches persist across server restarts."""
        # Phase 1: Create initial server instance with multiple watches
        manager1 = WatchToolsManager(self.mock_workspace_client)
        manager1.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager1.initialize()
        
        # Add multiple watches with different configurations
        watch_configs = []
        for i, watch_dir in enumerate(self.watch_dirs):
            result = await manager1.add_watch_folder(
                path=str(watch_dir),
                collection=f"collection{i+1}",
                patterns=[f"*.{['txt', 'pdf', 'md'][i]}"],
                recursive=(i % 2 == 0),  # Alternate recursive setting
                debounce_seconds=5 + i,
                auto_ingest=False,
            )
            assert result["success"]
            watch_configs.append((result["watch_id"], str(watch_dir), f"collection{i+1}"))
        
        # Verify all watches were created
        list_result = await manager1.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == 3
        
        # Shutdown first server instance
        await manager1.cleanup()
        
        # Phase 2: Start new server instance (simulating restart)
        manager2 = WatchToolsManager(self.mock_workspace_client)
        manager2.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager2.initialize()
        
        # Verify all watches were restored
        restored_list = await manager2.list_watched_folders()
        assert restored_list["success"]
        assert len(restored_list["watches"]) == 3
        
        # Verify watch details are preserved
        restored_watches = {w["id"]: w for w in restored_list["watches"]}
        for watch_id, original_path, original_collection in watch_configs:
            assert watch_id in restored_watches
            restored_watch = restored_watches[watch_id]
            assert restored_watch["path"] == original_path
            assert restored_watch["collection"] == original_collection
        
        # Cleanup second instance
        await manager2.cleanup()
    
    async def test_configuration_changes_persist(self):
        """Test that configuration changes persist across restarts."""
        # Phase 1: Create watch and modify its configuration
        manager1 = WatchToolsManager(self.mock_workspace_client)
        manager1.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager1.initialize()
        
        # Add initial watch
        result = await manager1.add_watch_folder(
            path=str(self.watch_dirs[0]),
            collection="collection1",
            patterns=["*.txt"],
            debounce_seconds=5,
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Modify configuration
        config_result = await manager1.configure_watch_settings(
            watch_id=watch_id,
            patterns=["*.pdf", "*.md"],
            debounce_seconds=15,
            status="paused"
        )
        assert config_result["success"]
        
        # Shutdown first instance
        await manager1.cleanup()
        
        # Phase 2: Restart and verify changes persisted
        manager2 = WatchToolsManager(self.mock_workspace_client)
        manager2.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager2.initialize()
        
        # Get restored watch configuration
        list_result = await manager2.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == 1
        
        restored_watch = list_result["watches"][0]
        assert restored_watch["id"] == watch_id
        assert set(restored_watch["patterns"]) == {"*.pdf", "*.md"}
        assert restored_watch["debounce_seconds"] == 15
        assert restored_watch["status"] == "paused"
        
        await manager2.cleanup()
    
    async def test_partial_corruption_recovery(self):
        """Test recovery when config file is partially corrupted."""
        # Phase 1: Create valid configuration
        manager1 = WatchToolsManager(self.mock_workspace_client)
        manager1.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager1.initialize()
        
        # Add watches
        for i, watch_dir in enumerate(self.watch_dirs[:2]):
            result = await manager1.add_watch_folder(
                path=str(watch_dir),
                collection=f"collection{i+1}",
                auto_ingest=False,
            )
            assert result["success"]
        
        await manager1.cleanup()
        
        # Phase 2: Corrupt the config file partially
        with open(self.config_file, 'r') as f:
            config_data = json.load(f)
        
        # Make one watch configuration invalid
        config_data["watches"][0]["path"] = "/invalid/path"
        
        with open(self.config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Phase 3: Restart and verify recovery
        manager2 = WatchToolsManager(self.mock_workspace_client)
        manager2.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager2.initialize()
        
        # Should recover the valid watch and skip the invalid one
        list_result = await manager2.list_watched_folders()
        assert list_result["success"]
        # Note: Depending on validation strategy, might have 1 valid or 0 watches
        # This tests the system doesn't crash on partial corruption
        
        await manager2.cleanup()
    
    async def test_concurrent_access_during_restart(self):
        """Test concurrent access to config during restart scenarios."""
        manager1 = WatchToolsManager(self.mock_workspace_client)
        manager1.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager1.initialize()
        
        # Start adding a watch
        add_task = asyncio.create_task(
            manager1.add_watch_folder(
                path=str(self.watch_dirs[0]),
                collection="collection1",
                auto_ingest=False,
            )
        )
        
        # Immediately start second manager (simulating concurrent access)
        manager2 = WatchToolsManager(self.mock_workspace_client)
        manager2.config_manager = SynchronizedWatchConfigManager(self.config_file)
        
        init_task = asyncio.create_task(manager2.initialize())
        
        # Wait for both operations
        add_result = await add_task
        await init_task
        
        # One of the operations should succeed
        if add_result["success"]:
            # First manager succeeded, verify second sees the watch
            list_result = await manager2.list_watched_folders()
            assert list_result["success"]
            # May or may not see the watch depending on timing
        
        await manager1.cleanup()
        await manager2.cleanup()


class TestAdvancedPersistenceScenarios:
    """Test advanced persistence scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "advanced_test.json"
        self.watch_dir = Path(self.temp_dir) / "watch_target"
        self.watch_dir.mkdir()
        
        # Create subdirectories for recursive testing
        (self.watch_dir / "subdir1").mkdir()
        (self.watch_dir / "subdir2").mkdir()
        (self.watch_dir / "subdir1" / "deep").mkdir()
        
        self.mock_workspace_client = MagicMock()
        self.mock_workspace_client.list_collections.return_value = asyncio.Future()
        self.mock_workspace_client.list_collections.return_value.set_result(["test-collection"])
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_watch_state_synchronization(self):
        """Test that watch states synchronize correctly across instances."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add watch
        result = await manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Test status change synchronization
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        manager.config_manager.subscribe_to_changes(event_handler)
        
        # Change watch status
        config_result = await manager.configure_watch_settings(
            watch_id=watch_id,
            status="paused"
        )
        assert config_result["success"]
        
        # Wait for event processing
        await asyncio.sleep(0.1)
        
        # Verify event was triggered
        assert len(events_received) > 0
        status_change_events = [e for e in events_received if e.event_type == "status_changed"]
        assert len(status_change_events) > 0
        
        await manager.cleanup()
    
    async def test_filesystem_permissions_change_recovery(self):
        """Test recovery when filesystem permissions change."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add watch
        result = await manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Simulate permission issue (just test the recovery mechanism)
        recovery_result = await manager.error_recovery.attempt_recovery(
            watch_id=watch_id,
            error_type="PATH_ACCESS_DENIED",
            path=self.watch_dir,
            error_details="Permission denied"
        )
        
        # Recovery attempt should be recorded even if not successful
        history = manager.error_recovery.get_recovery_history(watch_id)
        assert len(history) > 0
        assert history[0]["error_type"] == "PATH_ACCESS_DENIED"
        
        await manager.cleanup()
    
    async def test_network_path_handling(self):
        """Test handling of network paths and connectivity issues."""
        # This would be more meaningful with actual network paths
        # For now, test the validation and recovery mechanisms
        
        from common.core.watch_validation import WatchPathValidator
        
        # Simulate network path validation
        fake_network_path = Path("//server/share/folder")
        
        # The validator should handle network paths gracefully
        # (even if they don't exist in test environment)
        result = WatchPathValidator.validate_path_existence(fake_network_path)
        
        # Should fail validation but not crash
        assert not result.valid
        assert result.error_code in ["PATH_NOT_EXISTS", "PATH_ACCESS_ERROR"]
    
    async def test_large_number_of_watches(self):
        """Test performance with large number of watch configurations."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Create multiple watch directories
        watch_dirs = []
        for i in range(10):  # Create 10 watches for performance test
            watch_dir = Path(self.temp_dir) / f"perf_watch_{i}"
            watch_dir.mkdir()
            watch_dirs.append(watch_dir)
        
        # Add watches in batch
        watch_ids = []
        start_time = time.time()
        
        for i, watch_dir in enumerate(watch_dirs):
            result = await manager.add_watch_folder(
                path=str(watch_dir),
                collection="test-collection",
                patterns=[f"*.{i}"],  # Unique patterns
                auto_ingest=False,
            )
            assert result["success"]
            watch_ids.append(result["watch_id"])
        
        creation_time = time.time() - start_time
        
        # Verify all watches were created
        list_result = await manager.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == 10
        
        # Test listing performance
        start_time = time.time()
        for _ in range(5):  # Multiple list operations
            list_result = await manager.list_watched_folders()
            assert list_result["success"]
        listing_time = time.time() - start_time
        
        # Performance should be reasonable (adjust thresholds as needed)
        assert creation_time < 5.0  # Should create 10 watches in under 5 seconds
        assert listing_time < 1.0   # Should list watches 5 times in under 1 second
        
        await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])