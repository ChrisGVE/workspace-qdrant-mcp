"""
Comprehensive tests for Priority Queue Manager.

Tests cover priority calculation, resource optimization, queue operations,
MCP activity detection, backpressure handling, and integration with
SQLite state management and incremental processing.
"""

import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import psutil

from src.workspace_qdrant_mcp.core.priority_queue_manager import (
    PriorityQueueManager,
    MCPActivityLevel,
    ProcessingMode,
    QueueHealthStatus,
    MCPActivityMetrics,
    PriorityCalculationContext,
    ProcessingJob,
    QueueStatistics,
    ResourceConfiguration,
    ProcessingContextManager,
)
from src.workspace_qdrant_mcp.core.sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
    FileProcessingStatus,
    FileProcessingRecord,
    ProcessingQueueItem,
)
from src.workspace_qdrant_mcp.core.incremental_processor import (
    IncrementalProcessor,
    FileChangeInfo,
    ChangeType,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
async def state_manager(temp_db_path):
    """Create and initialize a SQLite state manager."""
    manager = SQLiteStateManager(temp_db_path)
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
def incremental_processor(state_manager):
    """Create a mock incremental processor."""
    processor = AsyncMock(spec=IncrementalProcessor)
    processor.detect_changes = AsyncMock(return_value=[])
    processor.process_changes = AsyncMock()
    return processor


@pytest.fixture
def resource_config():
    """Create a test resource configuration."""
    return ResourceConfiguration(
        max_concurrent_jobs=2,
        max_memory_mb=256,
        max_cpu_percent=80,
        conservative_concurrent_jobs=1,
        balanced_concurrent_jobs=2,
        aggressive_concurrent_jobs=4,
        burst_concurrent_jobs=6,
    )


@pytest.fixture
async def queue_manager(state_manager, incremental_processor, resource_config):
    """Create a priority queue manager."""
    manager = PriorityQueueManager(
        state_manager=state_manager,
        incremental_processor=incremental_processor,
        resource_config=resource_config,
        mcp_detection_interval=1,  # Fast for testing
        statistics_retention_hours=1,
    )
    
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def temp_files():
    """Create temporary test files."""
    files = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create test files
        for i in range(3):
            file_path = tmpdir_path / f"test_file_{i}.txt"
            file_path.write_text(f"Test content {i}")
            files.append(str(file_path))
        
        yield files


class TestPriorityQueueManager:
    """Test suite for PriorityQueueManager."""

    @pytest.mark.asyncio
    async def test_initialization(self, queue_manager):
        """Test queue manager initialization."""
        assert queue_manager._initialized is True
        assert queue_manager.processing_mode == ProcessingMode.CONSERVATIVE
        assert queue_manager.mcp_activity.activity_level == MCPActivityLevel.INACTIVE
        assert queue_manager.statistics.health_status == QueueHealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_shutdown(self, state_manager, resource_config):
        """Test graceful shutdown."""
        manager = PriorityQueueManager(
            state_manager=state_manager,
            resource_config=resource_config
        )
        
        await manager.initialize()
        assert manager._initialized is True
        
        await manager.shutdown()
        assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_enqueue_file(self, queue_manager, temp_files):
        """Test file enqueueing with priority calculation."""
        file_path = temp_files[0]
        collection = "test-collection"
        
        queue_id = await queue_manager.enqueue_file(
            file_path=file_path,
            collection=collection,
            user_triggered=False
        )
        
        assert queue_id is not None
        assert queue_manager.statistics.total_items == 1
        
        # Check that file was added to SQLite queue
        queue_item = await queue_manager.state_manager.get_next_queue_item()
        assert queue_item is not None
        assert queue_item.file_path == file_path
        assert queue_item.collection == collection

    @pytest.mark.asyncio
    async def test_priority_calculation_user_triggered(self, queue_manager, temp_files):
        """Test priority calculation for user-triggered files."""
        file_path = temp_files[0]
        
        queue_id = await queue_manager.enqueue_file(
            file_path=file_path,
            collection="test",
            user_triggered=True
        )
        
        # User-triggered files should get higher priority
        queue_item = await queue_manager.state_manager.get_next_queue_item()
        assert queue_item.priority in [ProcessingPriority.HIGH, ProcessingPriority.URGENT]

    @pytest.mark.asyncio
    async def test_priority_calculation_current_project(self, queue_manager, temp_files):
        """Test priority calculation for current project files."""
        file_path = temp_files[0]
        
        # Set current project root
        queue_manager.set_current_project_root(str(Path(file_path).parent))
        
        queue_id = await queue_manager.enqueue_file(
            file_path=file_path,
            collection="test",
            user_triggered=False
        )
        
        # Current project files should get elevated priority
        queue_item = await queue_manager.state_manager.get_next_queue_item()
        assert queue_item.priority in [ProcessingPriority.NORMAL, ProcessingPriority.HIGH]

    @pytest.mark.asyncio
    async def test_mcp_activity_detection(self, queue_manager):
        """Test MCP activity level detection and processing mode adaptation."""
        # Simulate high MCP activity
        queue_manager.mcp_activity.update_activity(request_count=25, session_count=2)
        
        assert queue_manager.mcp_activity.activity_level == MCPActivityLevel.HIGH
        
        # Update processing mode based on activity
        await queue_manager._update_processing_mode()
        
        # Should switch to more aggressive mode with high activity
        assert queue_manager.processing_mode in [ProcessingMode.BALANCED, ProcessingMode.AGGRESSIVE]

    @pytest.mark.asyncio
    async def test_processing_mode_resource_adaptation(self, queue_manager):
        """Test processing mode adaptation based on system resources."""
        # Mock high CPU usage
        with patch('psutil.cpu_percent', return_value=85), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 75
            
            # Set high MCP activity
            queue_manager.mcp_activity.update_activity(request_count=30, session_count=3)
            
            await queue_manager._update_processing_mode()
            
            # Should use conservative mode due to high resource usage
            assert queue_manager.processing_mode in [ProcessingMode.CONSERVATIVE, ProcessingMode.BALANCED]

    @pytest.mark.asyncio
    async def test_backpressure_detection(self, queue_manager):
        """Test backpressure detection and handling."""
        # Mock high system resource usage
        with patch('psutil.cpu_percent', return_value=95), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value.percent = 90
            
            backpressure = await queue_manager._check_backpressure()
            assert backpressure is True

    @pytest.mark.asyncio
    async def test_process_next_batch_empty_queue(self, queue_manager):
        """Test processing when queue is empty."""
        completed_jobs = await queue_manager.process_next_batch()
        assert len(completed_jobs) == 0

    @pytest.mark.asyncio
    async def test_process_next_batch_with_jobs(self, queue_manager, temp_files):
        """Test batch processing of queued files."""
        # Enqueue multiple files
        for i, file_path in enumerate(temp_files):
            await queue_manager.enqueue_file(
                file_path=file_path,
                collection=f"collection-{i}"
            )
        
        # Process batch
        completed_jobs = await queue_manager.process_next_batch(batch_size=2)
        
        # Should process up to batch_size items
        assert len(completed_jobs) <= 2
        
        # Check that jobs were processed
        for job in completed_jobs:
            assert isinstance(job, ProcessingJob)
            assert job.metadata.get("success", False) is True

    @pytest.mark.asyncio
    async def test_processing_context_manager(self, queue_manager, temp_files):
        """Test processing context manager for batch operations."""
        # Enqueue files
        file_collection_pairs = [(f, f"collection-{i}") for i, f in enumerate(temp_files)]
        
        async with queue_manager.get_processing_context() as context:
            queue_ids = await context.enqueue_multiple_files(
                file_collection_pairs=file_collection_pairs,
                user_triggered=True
            )
            
            assert len(queue_ids) == len(temp_files)
            
            # Process batch
            completed_jobs = await context.process_next_batch()
            assert len(completed_jobs) <= len(temp_files)

    @pytest.mark.asyncio
    async def test_queue_statistics_tracking(self, queue_manager, temp_files):
        """Test queue statistics and monitoring."""
        initial_stats = queue_manager.statistics
        
        # Enqueue files with different priorities
        await queue_manager.enqueue_file(temp_files[0], "test", user_triggered=True)
        await queue_manager.enqueue_file(temp_files[1], "test", user_triggered=False)
        
        # Check statistics updated
        assert queue_manager.statistics.total_items == 2
        assert len(queue_manager.statistics.items_by_priority) > 0

    @pytest.mark.asyncio
    async def test_health_status_monitoring(self, queue_manager):
        """Test health status monitoring and reporting."""
        health_status = await queue_manager.get_health_status()
        
        assert "health_status" in health_status
        assert "system_metrics" in health_status
        assert "queue_statistics" in health_status
        assert "resource_status" in health_status
        assert "mcp_activity" in health_status

    @pytest.mark.asyncio
    async def test_queue_status_reporting(self, queue_manager):
        """Test comprehensive queue status reporting."""
        status = await queue_manager.get_queue_status()
        
        assert status["initialized"] is True
        assert "processing_mode" in status
        assert "mcp_activity" in status
        assert "statistics" in status
        assert "active_jobs" in status

    @pytest.mark.asyncio
    async def test_priority_calculation_hooks(self, queue_manager, temp_files):
        """Test custom priority calculation hooks."""
        hook_called = False
        
        async def custom_priority_hook(context, current_score):
            nonlocal hook_called
            hook_called = True
            # Boost priority for test files
            if "test_file" in context.file_path:
                return current_score + 10
            return current_score
        
        queue_manager.add_priority_calculation_hook(custom_priority_hook)
        
        await queue_manager.enqueue_file(temp_files[0], "test")
        
        assert hook_called is True

    @pytest.mark.asyncio
    async def test_processing_hooks(self, queue_manager, temp_files):
        """Test processing completion hooks."""
        hook_calls = []
        
        async def processing_hook(job, success, processing_time):
            hook_calls.append({
                "job_id": job.queue_id,
                "success": success,
                "processing_time": processing_time
            })
        
        queue_manager.add_processing_hook(processing_hook)
        
        await queue_manager.enqueue_file(temp_files[0], "test")
        completed_jobs = await queue_manager.process_next_batch()
        
        # Hook should have been called for processed jobs
        assert len(hook_calls) == len(completed_jobs)

    @pytest.mark.asyncio
    async def test_monitoring_hooks(self, queue_manager):
        """Test monitoring hooks for statistics updates."""
        hook_calls = []
        
        async def monitoring_hook(statistics, health_metrics):
            hook_calls.append({
                "total_items": statistics.total_items,
                "health_status": statistics.health_status.value,
                "timestamp": datetime.now(timezone.utc)
            })
        
        queue_manager.add_monitoring_hook(monitoring_hook)
        
        # Trigger monitoring update
        await queue_manager._update_health_metrics()
        await queue_manager._update_queue_statistics()
        
        # Wait for monitoring loop to run
        await asyncio.sleep(0.1)

    @pytest.mark.asyncio
    async def test_crash_recovery(self, state_manager, resource_config):
        """Test crash recovery functionality."""
        # Create manager and simulate processing items
        manager = PriorityQueueManager(
            state_manager=state_manager,
            resource_config=resource_config
        )
        
        await manager.initialize()
        
        # Add item to queue
        await state_manager.add_to_processing_queue(
            file_path="/test/file.txt",
            collection="test",
            priority=ProcessingPriority.HIGH
        )
        
        # Simulate crash by shutting down without completing
        await manager.shutdown()
        
        # Create new manager (simulating restart)
        new_manager = PriorityQueueManager(
            state_manager=state_manager,
            resource_config=resource_config
        )
        
        await new_manager.initialize()
        
        # Should perform crash recovery
        assert new_manager._initialized is True
        
        await new_manager.shutdown()

    @pytest.mark.asyncio
    async def test_queue_clearing(self, queue_manager, temp_files):
        """Test queue clearing functionality."""
        # Enqueue files
        for file_path in temp_files:
            await queue_manager.enqueue_file(file_path, "test")
        
        assert queue_manager.statistics.total_items == len(temp_files)
        
        # Clear queue
        cleared_count = await queue_manager.clear_queue()
        assert cleared_count == len(temp_files)

    @pytest.mark.asyncio
    async def test_resource_configuration_modes(self, state_manager):
        """Test different resource configuration modes."""
        configs = [
            (ProcessingMode.CONSERVATIVE, 1),
            (ProcessingMode.BALANCED, 2), 
            (ProcessingMode.AGGRESSIVE, 4),
            (ProcessingMode.BURST, 6),
        ]
        
        for mode, expected_workers in configs:
            config = ResourceConfiguration(
                conservative_concurrent_jobs=1,
                balanced_concurrent_jobs=2,
                aggressive_concurrent_jobs=4,
                burst_concurrent_jobs=6,
            )
            
            manager = PriorityQueueManager(
                state_manager=state_manager,
                resource_config=config
            )
            
            await manager.initialize()
            manager.processing_mode = mode
            await manager._configure_executor()
            
            # Verify executor configuration
            assert manager.executor is not None
            
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_optimal_batch_size_calculation(self, queue_manager):
        """Test optimal batch size calculation based on processing mode."""
        # Test different modes
        modes_and_sizes = [
            (ProcessingMode.CONSERVATIVE, 1),
            (ProcessingMode.BALANCED, 2),
            (ProcessingMode.AGGRESSIVE, 4),
            (ProcessingMode.BURST, 6),
        ]
        
        for mode, expected_base_size in modes_and_sizes:
            queue_manager.processing_mode = mode
            batch_size = await queue_manager._get_optimal_batch_size()
            
            # Should return reasonable batch size
            assert batch_size >= 1
            assert batch_size <= 10

    @pytest.mark.asyncio
    async def test_job_failure_handling(self, queue_manager, temp_files):
        """Test job failure handling and retry logic."""
        # Mock incremental processor to fail
        queue_manager.incremental_processor.detect_changes.side_effect = Exception("Test failure")
        
        await queue_manager.enqueue_file(temp_files[0], "test")
        
        # Process batch - should handle failure gracefully
        completed_jobs = await queue_manager.process_next_batch()
        
        # Should return empty list for failed jobs
        assert len(completed_jobs) == 0


class TestMCPActivityMetrics:
    """Test suite for MCP activity metrics."""

    def test_activity_level_calculation(self):
        """Test activity level calculation based on request rate."""
        metrics = MCPActivityMetrics()
        metrics.session_start_time = datetime.now(timezone.utc)
        
        # Test different activity levels
        test_cases = [
            (0, 0, MCPActivityLevel.INACTIVE),
            (5, 1, MCPActivityLevel.LOW),
            (15, 1, MCPActivityLevel.MODERATE),
            (25, 2, MCPActivityLevel.HIGH),
        ]
        
        for request_count, session_count, expected_level in test_cases:
            metrics.update_activity(request_count, session_count)
            assert metrics.activity_level == expected_level

    def test_burst_detection(self):
        """Test burst activity detection."""
        metrics = MCPActivityMetrics()
        metrics.session_start_time = datetime.now(timezone.utc)
        
        # Simulate burst activity
        metrics.update_activity(request_count=10, session_count=1)
        
        # Should detect burst for high request count in short time
        if metrics.burst_detected:
            assert metrics.activity_level == MCPActivityLevel.BURST


class TestPriorityCalculationContext:
    """Test suite for priority calculation context."""

    def test_context_creation(self):
        """Test priority calculation context creation."""
        context = PriorityCalculationContext(
            file_path="/test/file.py",
            collection="test-collection",
            mcp_activity=MCPActivityMetrics(),
            is_user_triggered=True,
            is_current_project=True,
            is_recently_modified=True
        )
        
        assert context.file_path == "/test/file.py"
        assert context.collection == "test-collection"
        assert context.is_user_triggered is True
        assert context.is_current_project is True
        assert context.is_recently_modified is True


class TestQueueStatistics:
    """Test suite for queue statistics."""

    def test_statistics_initialization(self):
        """Test queue statistics initialization."""
        stats = QueueStatistics()
        
        assert stats.total_items == 0
        assert stats.processing_rate == 0.0
        assert stats.success_rate == 1.0
        assert stats.health_status == QueueHealthStatus.HEALTHY

    def test_statistics_updates(self):
        """Test statistics updates."""
        stats = QueueStatistics()
        
        # Update statistics
        stats.total_items = 10
        stats.processing_rate = 5.0
        stats.success_rate = 0.95
        stats.backpressure_events = 2
        
        assert stats.total_items == 10
        assert stats.processing_rate == 5.0
        assert stats.success_rate == 0.95
        assert stats.backpressure_events == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])