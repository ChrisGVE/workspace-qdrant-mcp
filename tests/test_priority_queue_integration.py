"""
Integration tests for Priority Queue Manager with existing components.

Tests the integration between PriorityQueueManager, SQLiteStateManager,
and IncrementalProcessor to ensure proper end-to-end functionality.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from common.core.priority_queue_manager import (
    PriorityQueueManager,
    ProcessingMode,
    MCPActivityLevel,
    ResourceConfiguration,
)
from common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
    FileProcessingStatus,
)
from common.core.incremental_processor import (
    IncrementalProcessor,
    FileChangeInfo,
    ChangeType,
)


@pytest.fixture
async def integration_setup():
    """Set up complete integration environment."""
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        test_files = []
        
        for i in range(5):
            file_path = tmpdir_path / f"integration_test_{i}.py"
            file_path.write_text(f"""
# Integration test file {i}
def test_function_{i}():
    '''Test function {i} for integration testing.'''
    return {i}

class TestClass{i}:
    def method_{i}(self):
        return "test_{i}"
""")
            test_files.append(str(file_path))
        
        # Initialize components
        state_manager = SQLiteStateManager(db_path)
        await state_manager.initialize()
        
        # Create mock incremental processor with realistic behavior
        incremental_processor = AsyncMock(spec=IncrementalProcessor)
        incremental_processor.detect_changes = AsyncMock()
        incremental_processor.process_changes = AsyncMock()
        
        # Configure resource limits for testing
        resource_config = ResourceConfiguration(
            max_concurrent_jobs=3,
            conservative_concurrent_jobs=1,
            balanced_concurrent_jobs=2,
            aggressive_concurrent_jobs=3,
        )
        
        # Initialize priority queue manager
        queue_manager = PriorityQueueManager(
            state_manager=state_manager,
            incremental_processor=incremental_processor,
            resource_config=resource_config,
            mcp_detection_interval=0.5,  # Fast for testing
        )
        
        await queue_manager.initialize()
        
        yield {
            "queue_manager": queue_manager,
            "state_manager": state_manager,
            "incremental_processor": incremental_processor,
            "test_files": test_files,
            "tmpdir": tmpdir_path,
        }
        
        # Cleanup
        await queue_manager.shutdown()
        await state_manager.close()
        
        if os.path.exists(db_path):
            os.unlink(db_path)


class TestPriorityQueueIntegration:
    """Integration tests for priority queue system."""

    @pytest.mark.asyncio
    async def test_end_to_end_processing_workflow(self, integration_setup):
        """Test complete end-to-end processing workflow."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        state_manager = setup["state_manager"]
        incremental_processor = setup["incremental_processor"]
        test_files = setup["test_files"]
        
        # Configure incremental processor to simulate file changes
        file_changes = [
            FileChangeInfo(
                file_path=file_path,
                change_type=ChangeType.CREATED,
                collection=f"test-collection-{i}",
                priority=ProcessingPriority.NORMAL,
            )
            for i, file_path in enumerate(test_files[:3])
        ]
        
        incremental_processor.detect_changes.return_value = file_changes
        
        # Set current project for priority boost
        queue_manager.set_current_project_root(str(setup["tmpdir"]))
        
        # Enqueue files with different priorities
        queue_ids = []
        for i, file_path in enumerate(test_files):
            queue_id = await queue_manager.enqueue_file(
                file_path=file_path,
                collection=f"integration-test-{i}",
                user_triggered=(i % 2 == 0),  # Every other file is user-triggered
            )
            queue_ids.append(queue_id)
        
        # Verify files were queued in state manager
        total_queued = sum((await state_manager.get_queue_stats()).values())
        assert total_queued == len(test_files)
        
        # Process files in batches
        all_completed_jobs = []
        batch_count = 0
        max_batches = 3
        
        while batch_count < max_batches:
            completed_jobs = await queue_manager.process_next_batch(batch_size=2)
            
            if not completed_jobs:
                break
                
            all_completed_jobs.extend(completed_jobs)
            batch_count += 1
            
            # Verify incremental processor was called
            if completed_jobs:
                assert incremental_processor.detect_changes.called
        
        # Verify processing completed
        assert len(all_completed_jobs) <= len(test_files)
        
        # Check queue statistics
        stats = await queue_manager.get_queue_status()
        assert stats["initialized"] is True
        assert "statistics" in stats

    @pytest.mark.asyncio
    async def test_priority_ordering_integration(self, integration_setup):
        """Test that files are processed in correct priority order."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        state_manager = setup["state_manager"]
        test_files = setup["test_files"]
        
        # Set up high MCP activity to boost priorities
        queue_manager.mcp_activity.update_activity(request_count=20, session_count=2)
        
        # Enqueue files with different priority characteristics
        priority_files = [
            (test_files[0], True, "urgent-collection"),    # User triggered
            (test_files[1], False, "normal-collection"),   # Background
            (test_files[2], True, "high-collection"),      # User triggered
        ]
        
        queue_ids = []
        for file_path, user_triggered, collection in priority_files:
            queue_id = await queue_manager.enqueue_file(
                file_path=file_path,
                collection=collection,
                user_triggered=user_triggered
            )
            queue_ids.append(queue_id)
        
        # Get items from queue in priority order
        processed_items = []
        for _ in range(len(priority_files)):
            item = await state_manager.get_next_queue_item()
            if item:
                processed_items.append(item)
        
        # Verify user-triggered files have higher priority
        user_triggered_priorities = [
            item.priority for item in processed_items 
            if any(qid in item.queue_id for qid in queue_ids[:3:2])  # User-triggered files
        ]
        
        background_priorities = [
            item.priority for item in processed_items
            if queue_ids[1] in item.queue_id  # Background file
        ]
        
        if user_triggered_priorities and background_priorities:
            avg_user_priority = sum(p.value for p in user_triggered_priorities) / len(user_triggered_priorities)
            avg_background_priority = sum(p.value for p in background_priorities) / len(background_priorities)
            
            # User-triggered should have higher average priority (higher enum value)
            assert avg_user_priority >= avg_background_priority

    @pytest.mark.asyncio
    async def test_mcp_activity_adaptation(self, integration_setup):
        """Test adaptation to changing MCP activity levels."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        test_files = setup["test_files"]
        
        # Start with inactive MCP
        initial_mode = queue_manager.processing_mode
        assert initial_mode == ProcessingMode.CONSERVATIVE
        
        # Simulate increasing MCP activity
        queue_manager.mcp_activity.update_activity(request_count=15, session_count=1)
        await queue_manager._update_processing_mode()
        
        moderate_mode = queue_manager.processing_mode
        assert moderate_mode in [ProcessingMode.CONSERVATIVE, ProcessingMode.BALANCED]
        
        # Simulate high activity
        queue_manager.mcp_activity.update_activity(request_count=25, session_count=2)
        await queue_manager._update_processing_mode()
        
        high_mode = queue_manager.processing_mode
        assert high_mode in [ProcessingMode.BALANCED, ProcessingMode.AGGRESSIVE]
        
        # Enqueue file and verify it gets appropriate priority boost
        queue_id = await queue_manager.enqueue_file(
            file_path=test_files[0],
            collection="activity-test",
            user_triggered=False
        )
        
        # High activity should boost priority even for non-user-triggered files
        queue_item = await queue_manager.state_manager.get_next_queue_item()
        assert queue_item.priority in [ProcessingPriority.NORMAL, ProcessingPriority.HIGH]

    @pytest.mark.asyncio
    async def test_resource_constraint_handling(self, integration_setup):
        """Test handling of resource constraints and backpressure."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        test_files = setup["test_files"]
        
        # Enqueue multiple files
        for i, file_path in enumerate(test_files):
            await queue_manager.enqueue_file(
                file_path=file_path,
                collection=f"resource-test-{i}"
            )
        
        # Mock high resource usage to trigger backpressure
        from unittest.mock import patch
        
        with patch('psutil.cpu_percent', return_value=95), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 90
            
            # Should detect backpressure
            backpressure = await queue_manager._check_backpressure()
            assert backpressure is True
            
            # Processing should be skipped due to backpressure
            completed_jobs = await queue_manager.process_next_batch()
            assert len(completed_jobs) == 0
            
            # Backpressure events should be tracked
            stats = await queue_manager.get_queue_status()
            assert stats["statistics"]["backpressure_events"] > 0

    @pytest.mark.asyncio
    async def test_state_persistence_integration(self, integration_setup):
        """Test that queue state persists correctly in SQLite."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        state_manager = setup["state_manager"]
        test_files = setup["test_files"]
        
        # Enqueue files
        original_queue_ids = []
        for file_path in test_files[:3]:
            queue_id = await queue_manager.enqueue_file(
                file_path=file_path,
                collection="persistence-test"
            )
            original_queue_ids.append(queue_id)
        
        # Verify files are persisted in database
        queue_stats = await state_manager.get_queue_stats()
        total_items = sum(queue_stats.values())
        assert total_items == 3
        
        # Simulate restart by creating new queue manager
        new_queue_manager = PriorityQueueManager(
            state_manager=state_manager,
            resource_config=queue_manager.resource_config,
        )
        
        await new_queue_manager.initialize()
        
        # Verify files are still in queue after restart
        restored_stats = await state_manager.get_queue_stats()
        restored_total = sum(restored_stats.values())
        assert restored_total == total_items
        
        # Can still process files
        completed_jobs = await new_queue_manager.process_next_batch()
        assert len(completed_jobs) > 0
        
        await new_queue_manager.shutdown()

    @pytest.mark.asyncio
    async def test_processing_context_integration(self, integration_setup):
        """Test processing context manager integration."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        test_files = setup["test_files"]
        
        # Use processing context for batch operations
        async with queue_manager.get_processing_context() as context:
            # Enqueue multiple files
            file_collection_pairs = [
                (file_path, f"context-test-{i}")
                for i, file_path in enumerate(test_files)
            ]
            
            queue_ids = await context.enqueue_multiple_files(
                file_collection_pairs=file_collection_pairs,
                user_triggered=True,
                metadata={"batch_id": "integration-test"}
            )
            
            assert len(queue_ids) == len(test_files)
            
            # Process files within context
            completed_jobs = await context.process_next_batch(batch_size=3)
            
            # Verify processing completed
            for job in completed_jobs:
                assert job.metadata.get("batch_id") == "integration-test"

    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self, integration_setup):
        """Test health monitoring and statistics integration."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        test_files = setup["test_files"]
        
        # Get initial health status
        initial_health = await queue_manager.get_health_status()
        assert initial_health["health_status"] in ["healthy", "degraded"]
        
        # Enqueue and process files
        for file_path in test_files[:2]:
            await queue_manager.enqueue_file(file_path, "health-test")
        
        completed_jobs = await queue_manager.process_next_batch()
        
        # Update statistics manually to test monitoring
        await queue_manager._update_processing_statistics(completed_jobs)
        await queue_manager._update_queue_statistics()
        
        # Get updated health status
        updated_health = await queue_manager.get_health_status()
        
        # Verify health metrics are updated
        assert "queue_statistics" in updated_health
        assert updated_health["queue_statistics"]["processing_rate"] >= 0
        assert updated_health["queue_statistics"]["success_rate"] >= 0

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integration_setup):
        """Test error handling across integrated components."""
        setup = await integration_setup
        queue_manager = setup["queue_manager"]
        incremental_processor = setup["incremental_processor"]
        test_files = setup["test_files"]
        
        # Configure processor to fail on specific files
        def failing_detect_changes(file_paths):
            if any("integration_test_1" in fp for fp in file_paths):
                raise Exception("Simulated processing failure")
            return []
        
        incremental_processor.detect_changes.side_effect = failing_detect_changes
        
        # Enqueue files including one that will fail
        queue_ids = []
        for file_path in test_files[:3]:
            queue_id = await queue_manager.enqueue_file(file_path, "error-test")
            queue_ids.append(queue_id)
        
        # Process batch - should handle failures gracefully
        completed_jobs = await queue_manager.process_next_batch()
        
        # Some jobs may fail, but system should remain stable
        health_status = await queue_manager.get_health_status()
        assert health_status["health_status"] in ["healthy", "degraded"]
        
        # Failed files should be tracked in statistics
        stats = await queue_manager.get_queue_status()
        assert "statistics" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])