"""
Functional tests for concurrent access to watch management.

These tests verify that the system handles concurrent MCP tool usage,
configuration updates, and file locking correctly under various scenarios.
"""

import asyncio
import json
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from workspace_qdrant_mcp.core.watch_sync import SynchronizedWatchConfigManager, FileLockManager
from workspace_qdrant_mcp.tools.watch_management import WatchToolsManager


class TestConcurrentMCPToolUsage:
    """Test concurrent usage of MCP tools."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "concurrent_test.json"
        
        # Create test watch directories
        self.watch_dirs = []
        for i in range(5):
            watch_dir = Path(self.temp_dir) / f"concurrent_watch_{i}"
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
    
    async def test_concurrent_watch_additions(self):
        """Test concurrent addition of multiple watches."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Create tasks for concurrent watch additions
        tasks = []
        for i, watch_dir in enumerate(self.watch_dirs):
            task = manager.add_watch_folder(
                path=str(watch_dir),
                collection=f"collection{(i % 3) + 1}",
                patterns=[f"*.{i}"],
                auto_ingest=False,
            )
            tasks.append(task)
        
        # Execute all additions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        assert len(successful_results) == len(self.watch_dirs)
        
        # Verify all watches were persisted
        list_result = await manager.list_watched_folders()
        assert list_result["success"]
        assert len(list_result["watches"]) == len(self.watch_dirs)
        
        # Verify unique watch IDs
        watch_ids = {w["id"] for w in list_result["watches"]}
        assert len(watch_ids) == len(self.watch_dirs)
        
        await manager.cleanup()
    
    async def test_concurrent_configuration_updates(self):
        """Test concurrent updates to watch configurations."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add initial watches
        watch_ids = []
        for watch_dir in self.watch_dirs[:3]:  # Use first 3 watches
            result = await manager.add_watch_folder(
                path=str(watch_dir),
                collection="collection1",
                auto_ingest=False,
            )
            assert result["success"]
            watch_ids.append(result["watch_id"])
        
        # Create concurrent update tasks
        update_tasks = []
        for i, watch_id in enumerate(watch_ids):
            task = manager.configure_watch_settings(
                watch_id=watch_id,
                debounce_seconds=10 + i,
                patterns=[f"*.update{i}"],
            )
            update_tasks.append(task)
        
        # Execute updates concurrently
        update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # All updates should succeed
        successful_updates = [r for r in update_results if isinstance(r, dict) and r.get("success")]
        assert len(successful_updates) == len(watch_ids)
        
        # Verify updates were applied
        list_result = await manager.list_watched_folders()
        assert list_result["success"]
        
        for i, watch in enumerate(list_result["watches"]):
            assert watch["debounce_seconds"] == 10 + i
            assert f"*.update{i}" in watch["patterns"]
        
        await manager.cleanup()
    
    async def test_concurrent_add_remove_operations(self):
        """Test concurrent add and remove operations."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add initial watches
        initial_watch_ids = []
        for watch_dir in self.watch_dirs[:2]:
            result = await manager.add_watch_folder(
                path=str(watch_dir),
                collection="collection1",
                auto_ingest=False,
            )
            assert result["success"]
            initial_watch_ids.append(result["watch_id"])
        
        # Create mixed concurrent operations
        mixed_tasks = []
        
        # Add new watches
        for watch_dir in self.watch_dirs[2:4]:
            task = manager.add_watch_folder(
                path=str(watch_dir),
                collection="collection2",
                auto_ingest=False,
            )
            mixed_tasks.append(("add", task))
        
        # Remove existing watches
        for watch_id in initial_watch_ids:
            task = manager.remove_watch_folder(watch_id)
            mixed_tasks.append(("remove", task))
        
        # Execute all operations concurrently
        tasks = [task for _, task in mixed_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All operations should complete (success depends on timing)
        assert len(results) == 4
        
        # Final state should be consistent
        final_list = await manager.list_watched_folders()
        assert final_list["success"]
        
        # Should have some watches (exact number depends on operation timing)
        # Important thing is system remains consistent
        assert isinstance(final_list["watches"], list)
        
        await manager.cleanup()
    
    async def test_concurrent_status_queries(self):
        """Test concurrent status queries while modifications occur."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add initial watch
        result = await manager.add_watch_folder(
            path=str(self.watch_dirs[0]),
            collection="collection1",
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Create concurrent status queries and modifications
        async def status_query_worker():
            results = []
            for _ in range(10):  # Multiple status queries
                status_result = await manager.get_watch_status(watch_id)
                results.append(status_result)
                await asyncio.sleep(0.01)  # Small delay
            return results
        
        async def modification_worker():
            modifications = []
            for i in range(5):  # Multiple modifications
                mod_result = await manager.configure_watch_settings(
                    watch_id=watch_id,
                    debounce_seconds=5 + i,
                )
                modifications.append(mod_result)
                await asyncio.sleep(0.02)  # Small delay
            return modifications
        
        # Run both workers concurrently
        status_task = asyncio.create_task(status_query_worker())
        modification_task = asyncio.create_task(modification_worker())
        
        status_results, modification_results = await asyncio.gather(
            status_task, modification_task, return_exceptions=True
        )
        
        # All status queries should succeed
        assert len(status_results) == 10
        assert all(isinstance(r, dict) and r.get("success") for r in status_results)
        
        # All modifications should succeed
        assert len(modification_results) == 5
        assert all(isinstance(r, dict) and r.get("success") for r in modification_results)
        
        await manager.cleanup()


class TestFileLockingUnderStress:
    """Test file locking mechanism under stress conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "lock_stress_test.json"
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_rapid_lock_acquisition_release(self):
        """Test rapid acquisition and release of file locks."""
        lock_manager = FileLockManager(self.config_file)
        
        # Perform rapid lock operations
        for i in range(20):  # 20 rapid lock cycles
            async with lock_manager.acquire_lock():
                # Simulate brief work
                await asyncio.sleep(0.001)
            
            # Brief pause between cycles
            await asyncio.sleep(0.001)
        
        # No locks should remain
        assert not lock_manager.lock_file.exists()
    
    async def test_concurrent_lock_contention(self):
        """Test concurrent lock contention with multiple managers."""
        lock_managers = [FileLockManager(self.config_file) for _ in range(5)]
        
        # Set short timeout for faster testing
        for manager in lock_managers:
            manager._lock_timeout = 1.0
        
        results = []
        
        async def lock_worker(manager_id, manager):
            try:
                async with manager.acquire_lock():
                    # Simulate some work
                    await asyncio.sleep(0.1)
                    results.append(f"success_{manager_id}")
            except TimeoutError:
                results.append(f"timeout_{manager_id}")
            except Exception as e:
                results.append(f"error_{manager_id}_{type(e).__name__}")
        
        # Create concurrent lock tasks
        tasks = [
            lock_worker(i, manager) 
            for i, manager in enumerate(lock_managers)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should have exactly one success and others timeout/error
        successes = [r for r in results if r.startswith("success")]
        assert len(successes) == 1
        
        # Others should timeout or error
        failures = [r for r in results if not r.startswith("success")]
        assert len(failures) == 4
        
        # No locks should remain
        assert not any(manager.lock_file.exists() for manager in lock_managers)
    
    async def test_stale_lock_detection(self):
        """Test detection and cleanup of stale lock files."""
        lock_manager = FileLockManager(self.config_file)
        
        # Create a fake stale lock file
        lock_info = {
            "pid": 99999,  # Non-existent PID
            "timestamp": "2020-01-01T00:00:00Z",  # Old timestamp
            "config_file": str(self.config_file)
        }
        
        lock_manager.lock_file.write_text(json.dumps(lock_info))
        
        # Verify stale lock is detected
        is_stale = await lock_manager._is_stale_lock()
        assert is_stale
        
        # Should be able to acquire lock (cleaning up stale lock)
        async with lock_manager.acquire_lock():
            assert lock_manager.lock_file.exists()
        
        # Lock should be cleaned up
        assert not lock_manager.lock_file.exists()


class TestEventSystemUnderLoad:
    """Test event notification system under load."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "event_load_test.json"
        self.watch_dir = Path(self.temp_dir) / "event_watch"
        self.watch_dir.mkdir()
        
        self.mock_workspace_client = MagicMock()
        self.mock_workspace_client.list_collections.return_value = asyncio.Future()
        self.mock_workspace_client.list_collections.return_value.set_result(["test-collection"])
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    async def test_high_frequency_configuration_changes(self):
        """Test system behavior with high-frequency configuration changes."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add initial watch
        result = await manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            auto_ingest=False,
        )
        assert result["success"]
        watch_id = result["watch_id"]
        
        # Track events
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        manager.config_manager.subscribe_to_changes(event_handler)
        
        # Generate rapid configuration changes
        change_tasks = []
        for i in range(20):  # 20 rapid changes
            task = manager.configure_watch_settings(
                watch_id=watch_id,
                debounce_seconds=5 + (i % 10),  # Cycle through different values
            )
            change_tasks.append(task)
        
        # Execute all changes
        change_results = await asyncio.gather(*change_tasks, return_exceptions=True)
        
        # Most changes should succeed (some might be redundant)
        successful_changes = [
            r for r in change_results 
            if isinstance(r, dict) and r.get("success")
        ]
        assert len(successful_changes) > 0  # At least some should succeed
        
        # Give event system time to process
        await asyncio.sleep(0.5)
        
        # Should have received events
        assert len(events_received) > 0
        
        await manager.cleanup()
    
    async def test_event_handler_exceptions(self):
        """Test system stability when event handlers raise exceptions."""
        manager = WatchToolsManager(self.mock_workspace_client)
        manager.config_manager = SynchronizedWatchConfigManager(self.config_file)
        await manager.initialize()
        
        # Add event handlers that raise exceptions
        def failing_handler(event):
            raise ValueError("Handler intentionally failed")
        
        def working_handler(event):
            working_handler.events_received = getattr(working_handler, 'events_received', [])
            working_handler.events_received.append(event)
        
        manager.config_manager.subscribe_to_changes(failing_handler)
        manager.config_manager.subscribe_to_changes(working_handler)
        
        # Trigger configuration change
        result = await manager.add_watch_folder(
            path=str(self.watch_dir),
            collection="test-collection",
            auto_ingest=False,
        )
        assert result["success"]
        
        # Give event system time to process
        await asyncio.sleep(0.2)
        
        # Working handler should still receive events despite failing handler
        assert hasattr(working_handler, 'events_received')
        assert len(working_handler.events_received) > 0
        
        await manager.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])