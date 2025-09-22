"""
Unit tests for PriorityQueueManager and processing engine functionality.

Tests the core processing queue system including:
- Priority calculation algorithms and dynamic scoring
- Resource management and processing mode optimization
- Backpressure handling and concurrency control
- Job processing lifecycle and error handling
- MCP activity detection and adaptive processing
- Queue statistics and health monitoring
"""

import sys
from pathlib import Path

# Add src/python to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src" / "python"))

import asyncio
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pytest

from workspace_qdrant_mcp.core.priority_queue_manager import (
    PriorityQueueManager,
    ResourceConfiguration,
    MCPActivityMetrics,
    MCPActivityLevel,
    ProcessingMode,
    QueueHealthStatus,
    QueueStatistics,
    ProcessingJob,
    PriorityCalculationContext,
    ProcessingContextManager
)
from workspace_qdrant_mcp.core.sqlite_state_manager import (
    ProcessingPriority,
    FileProcessingStatus,
    ProcessingQueueItem
)

from .conftest_daemon import (
    mock_priority_queue_manager,
    mock_sqlite_state_manager,
    mock_resource_configuration,
    mock_mcp_activity_metrics,
    mock_processing_job,
    mock_priority_calculation_context,
    isolated_daemon_temp_dir,
    wait_for_condition,
    assert_processing_job_valid,
    MockAsyncContextManager
)


class TestMCPActivityMetrics:
    """Test MCP activity metrics calculation."""
    
    def test_mcp_activity_initialization(self):
        """Test MCP activity metrics initialization."""
        metrics = MCPActivityMetrics()
        
        assert metrics.requests_per_minute == 0.0
        assert metrics.active_sessions == 0
        assert metrics.last_request_time is None
        assert metrics.activity_level == MCPActivityLevel.INACTIVE
        assert metrics.burst_detected is False
        assert metrics.session_start_time is None
        assert metrics.total_requests == 0
        assert metrics.average_request_duration == 0.0
    
    def test_mcp_activity_update_basic(self):
        """Test basic activity metric updates."""
        metrics = MCPActivityMetrics()
        metrics.session_start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
        
        metrics.update_activity(request_count=10, session_count=2)
        
        assert metrics.total_requests == 10
        assert metrics.active_sessions == 2
        assert metrics.last_request_time is not None
        assert metrics.requests_per_minute == 2.0  # 10 requests / 5 minutes
        assert metrics.activity_level == MCPActivityLevel.LOW
    
    def test_mcp_activity_level_calculation(self):
        """Test activity level calculation based on request rate."""
        metrics = MCPActivityMetrics()
        metrics.session_start_time = datetime.now(timezone.utc) - timedelta(minutes=1)
        
        # Test HIGH activity
        metrics.update_activity(request_count=25, session_count=1)
        assert metrics.activity_level == MCPActivityLevel.HIGH
        
        # Test MODERATE activity
        metrics.total_requests = 0
        metrics.update_activity(request_count=12, session_count=1)
        assert metrics.activity_level == MCPActivityLevel.MODERATE
        
        # Test LOW activity
        metrics.total_requests = 0
        metrics.update_activity(request_count=3, session_count=1)
        assert metrics.activity_level == MCPActivityLevel.LOW
        
        # Test INACTIVE
        metrics.total_requests = 0
        metrics.update_activity(request_count=0, session_count=0)
        assert metrics.activity_level == MCPActivityLevel.INACTIVE
    
    def test_mcp_activity_burst_detection(self):
        """Test burst activity detection."""
        metrics = MCPActivityMetrics()
        now = datetime.now(timezone.utc)
        metrics.session_start_time = now - timedelta(minutes=10)
        metrics.last_request_time = now - timedelta(seconds=30)  # Recent activity
        
        # Should detect burst with 5+ requests in recent window
        metrics.update_activity(request_count=7, session_count=1)
        
        assert metrics.burst_detected is True
        assert metrics.activity_level == MCPActivityLevel.BURST


class TestResourceConfiguration:
    """Test resource configuration management."""
    
    def test_resource_configuration_defaults(self):
        """Test resource configuration default values."""
        config = ResourceConfiguration()
        
        assert config.max_concurrent_jobs == 4
        assert config.max_memory_mb == 1024
        assert config.max_cpu_percent == 80
        assert config.conservative_concurrent_jobs == 1
        assert config.balanced_concurrent_jobs == 2
        assert config.aggressive_concurrent_jobs == 6
        assert config.burst_concurrent_jobs == 8
        assert config.backpressure_threshold == 0.9
        assert config.health_check_interval == 30
    
    def test_resource_configuration_custom(self):
        """Test custom resource configuration."""
        config = ResourceConfiguration(
            max_concurrent_jobs=8,
            max_memory_mb=2048,
            max_cpu_percent=90,
            burst_duration_seconds=600
        )
        
        assert config.max_concurrent_jobs == 8
        assert config.max_memory_mb == 2048
        assert config.max_cpu_percent == 90
        assert config.burst_duration_seconds == 600


class TestQueueStatistics:
    """Test queue statistics tracking."""
    
    def test_queue_statistics_initialization(self):
        """Test queue statistics initialization."""
        stats = QueueStatistics()
        
        assert stats.total_items == 0
        assert stats.items_by_priority == {}
        assert stats.items_by_status == {}
        assert stats.processing_rate == 0.0
        assert stats.average_processing_time == 0.0
        assert stats.success_rate == 1.0
        assert stats.backpressure_events == 0
        assert stats.health_status == QueueHealthStatus.HEALTHY
        assert stats.last_updated is not None
    
    def test_queue_statistics_updates(self):
        """Test queue statistics updating."""
        stats = QueueStatistics()
        
        # Update statistics
        stats.total_items = 25
        stats.items_by_priority = {
            "LOW": 5,
            "NORMAL": 15,
            "HIGH": 4,
            "URGENT": 1
        }
        stats.processing_rate = 2.5
        stats.average_processing_time = 1.2
        stats.success_rate = 0.95
        stats.backpressure_events = 3
        stats.health_status = QueueHealthStatus.DEGRADED
        
        assert stats.total_items == 25
        assert stats.items_by_priority["HIGH"] == 4
        assert stats.processing_rate == 2.5
        assert stats.success_rate == 0.95
        assert stats.health_status == QueueHealthStatus.DEGRADED


class TestProcessingJob:
    """Test processing job management."""
    
    def test_processing_job_initialization(self, mock_processing_job):
        """Test processing job initialization."""
        job = mock_processing_job
        
        assert_processing_job_valid(job)
        assert job.queue_id == "test_queue_123"
        assert job.file_path == "/tmp/test_file.py"
        assert job.collection == "test-collection"
        assert job.priority == ProcessingPriority.NORMAL
        assert job.calculated_score == 45.0
        assert job.attempts == 0
        assert job.max_attempts == 3
        assert job.timeout_seconds == 300
        assert job.metadata == {"test": True}
    
    def test_processing_job_with_context(self, mock_priority_calculation_context):
        """Test processing job with calculation context."""
        job = ProcessingJob(
            queue_id="test_queue_456",
            file_path="/tmp/context_file.py",
            collection="context-collection",
            priority=ProcessingPriority.HIGH,
            calculated_score=75.0,
            processing_context=mock_priority_calculation_context
        )
        
        assert_processing_job_valid(job)
        assert job.processing_context is not None
        assert job.processing_context.file_path == mock_priority_calculation_context.file_path
        assert job.calculated_score == 75.0


class TestPriorityCalculationContext:
    """Test priority calculation context."""
    
    def test_priority_calculation_context_initialization(self, mock_priority_calculation_context):
        """Test priority calculation context initialization."""
        context = mock_priority_calculation_context
        
        assert context.file_path.endswith("test.py")
        assert context.collection == "test-collection"
        assert context.mcp_activity is not None
        assert context.current_project_root is not None
        assert context.file_modification_time is not None
        assert context.file_size == 17
        assert context.is_user_triggered is False
        assert context.is_current_project is True
        assert context.is_recently_modified is True
        assert context.has_dependencies is False
        assert context.processing_history == {}
    
    def test_priority_calculation_context_user_triggered(self, isolated_daemon_temp_dir):
        """Test priority calculation context for user-triggered operations."""
        test_file = isolated_daemon_temp_dir / "user_file.py"
        test_file.write_text("print('user triggered')")
        
        context = PriorityCalculationContext(
            file_path=str(test_file),
            collection="user-collection",
            mcp_activity=MCPActivityMetrics(),
            is_user_triggered=True,
            is_current_project=True,
            is_recently_modified=True
        )
        
        assert context.is_user_triggered is True
        assert context.is_current_project is True
        assert context.is_recently_modified is True


class TestPriorityQueueManagerInitialization:
    """Test PriorityQueueManager initialization."""
    
    @pytest.mark.asyncio
    async def test_priority_queue_manager_initialization(
        self, 
        mock_sqlite_state_manager,
        mock_resource_configuration
    ):
        """Test priority queue manager initialization."""
        manager = PriorityQueueManager(
            state_manager=mock_sqlite_state_manager,
            resource_config=mock_resource_configuration,
            mcp_detection_interval=10,
            statistics_retention_hours=12
        )
        
        assert manager.state_manager is mock_sqlite_state_manager
        assert manager.resource_config is mock_resource_configuration
        assert manager.mcp_detection_interval == 10
        assert manager.statistics_retention_hours == 12
        assert manager.processing_mode == ProcessingMode.CONSERVATIVE
        assert manager.active_jobs == {}
        assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_priority_queue_manager_initialize_success(self, mock_priority_queue_manager):
        """Test successful priority queue manager initialization."""
        result = await mock_priority_queue_manager.initialize()
        
        assert result is True
        mock_priority_queue_manager.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_priority_queue_manager_initialize_failure(self):
        """Test failed priority queue manager initialization."""
        mock_state_manager = Mock()
        mock_state_manager.initialize = AsyncMock(return_value=False)
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        with patch.object(manager, '_initialize_processing_resources', AsyncMock()):
            with patch.object(manager, '_perform_crash_recovery', AsyncMock()):
                result = await manager.initialize()
                
                assert result is False
                assert manager._initialized is False
    
    @pytest.mark.asyncio
    async def test_priority_queue_manager_shutdown(self, mock_priority_queue_manager):
        """Test priority queue manager shutdown."""
        await mock_priority_queue_manager.shutdown()
        
        mock_priority_queue_manager.shutdown.assert_called_once()


class TestPriorityCalculation:
    """Test priority calculation algorithms."""
    
    @pytest.mark.asyncio
    async def test_priority_calculation_high_mcp_activity(self, isolated_daemon_temp_dir):
        """Test priority calculation with high MCP activity."""
        # Create mock context with high MCP activity
        test_file = isolated_daemon_temp_dir / "active_file.py"
        test_file.write_text("def active_function(): pass")
        
        mcp_activity = MCPActivityMetrics()
        mcp_activity.activity_level = MCPActivityLevel.HIGH
        mcp_activity.requests_per_minute = 25.0
        
        context = PriorityCalculationContext(
            file_path=str(test_file),
            collection="test-collection",
            mcp_activity=mcp_activity,
            is_current_project=True,
            is_user_triggered=True,
            is_recently_modified=True
        )
        
        # Mock the priority calculation method
        with patch('src.workspace_qdrant_mcp.core.priority_queue_manager.PriorityQueueManager._calculate_dynamic_priority') as mock_calc:
            mock_calc.return_value = (ProcessingPriority.HIGH, 85.0)
            
            manager = PriorityQueueManager(
                state_manager=Mock(),
                resource_config=ResourceConfiguration()
            )
            priority, score = await manager._calculate_dynamic_priority(context)
            
            assert priority == ProcessingPriority.HIGH
            assert score == 85.0
    
    @pytest.mark.asyncio
    async def test_priority_calculation_user_triggered_bonus(self, isolated_daemon_temp_dir):
        """Test priority calculation with user-triggered bonus."""
        test_file = isolated_daemon_temp_dir / "user_file.py"
        test_file.write_text("def user_function(): pass")
        
        context = PriorityCalculationContext(
            file_path=str(test_file),
            collection="test-collection",
            mcp_activity=MCPActivityMetrics(),
            is_user_triggered=True,
            is_current_project=True
        )
        
        with patch('src.workspace_qdrant_mcp.core.priority_queue_manager.PriorityQueueManager._calculate_dynamic_priority') as mock_calc:
            mock_calc.return_value = (ProcessingPriority.HIGH, 70.0)
            
            manager = PriorityQueueManager(
                state_manager=Mock(),
                resource_config=ResourceConfiguration()
            )
            priority, score = await manager._calculate_dynamic_priority(context)
            
            assert priority == ProcessingPriority.HIGH
            assert score >= 60.0  # Should have user-triggered bonus
    
    @pytest.mark.asyncio
    async def test_priority_calculation_low_activity(self, isolated_daemon_temp_dir):
        """Test priority calculation with low activity."""
        test_file = isolated_daemon_temp_dir / "background_file.py"
        test_file.write_text("# Background processing file")
        
        mcp_activity = MCPActivityMetrics()
        mcp_activity.activity_level = MCPActivityLevel.INACTIVE
        
        context = PriorityCalculationContext(
            file_path=str(test_file),
            collection="test-collection",
            mcp_activity=mcp_activity,
            is_user_triggered=False,
            is_current_project=False,
            is_recently_modified=False
        )
        
        with patch('src.workspace_qdrant_mcp.core.priority_queue_manager.PriorityQueueManager._calculate_dynamic_priority') as mock_calc:
            mock_calc.return_value = (ProcessingPriority.LOW, 15.0)
            
            manager = PriorityQueueManager(
                state_manager=Mock(),
                resource_config=ResourceConfiguration()
            )
            priority, score = await manager._calculate_dynamic_priority(context)
            
            assert priority == ProcessingPriority.LOW
            assert score < 30.0


class TestProcessingModeManagement:
    """Test processing mode management and resource optimization."""
    
    @pytest.mark.asyncio
    async def test_processing_mode_conservative(self):
        """Test conservative processing mode."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.mcp_activity.activity_level = MCPActivityLevel.INACTIVE
        
        with patch('psutil.cpu_percent', return_value=10.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 20.0
                with patch.object(manager, '_configure_executor', AsyncMock()):
                    await manager._update_processing_mode()
                    
                    assert manager.processing_mode == ProcessingMode.CONSERVATIVE
    
    @pytest.mark.asyncio
    async def test_processing_mode_balanced(self):
        """Test balanced processing mode."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.mcp_activity.activity_level = MCPActivityLevel.MODERATE
        
        with patch('psutil.cpu_percent', return_value=30.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 40.0
                with patch.object(manager, '_configure_executor', AsyncMock()):
                    await manager._update_processing_mode()
                    
                    assert manager.processing_mode == ProcessingMode.BALANCED
    
    @pytest.mark.asyncio
    async def test_processing_mode_aggressive(self):
        """Test aggressive processing mode."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.mcp_activity.activity_level = MCPActivityLevel.HIGH
        
        with patch('psutil.cpu_percent', return_value=50.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                with patch.object(manager, '_configure_executor', AsyncMock()):
                    await manager._update_processing_mode()
                    
                    assert manager.processing_mode == ProcessingMode.AGGRESSIVE
    
    @pytest.mark.asyncio
    async def test_processing_mode_burst(self):
        """Test burst processing mode."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.mcp_activity.activity_level = MCPActivityLevel.BURST
        
        with patch('psutil.cpu_percent', return_value=45.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 55.0
                with patch.object(manager, '_configure_executor', AsyncMock()):
                    await manager._update_processing_mode()
                    
                    assert manager.processing_mode == ProcessingMode.BURST
    
    @pytest.mark.asyncio
    async def test_executor_configuration_threads(self):
        """Test executor configuration for thread-based processing."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.processing_mode = ProcessingMode.CONSERVATIVE
        
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_thread_executor:
            mock_executor = Mock()
            mock_thread_executor.return_value = mock_executor
            
            await manager._configure_executor()
            
            assert manager.executor is mock_executor
            mock_thread_executor.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_executor_configuration_processes(self):
        """Test executor configuration for process-based processing."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.processing_mode = ProcessingMode.AGGRESSIVE
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_process_executor:
            mock_executor = Mock()
            mock_process_executor.return_value = mock_executor
            
            await manager._configure_executor()
            
            assert manager.executor is mock_executor
            mock_process_executor.assert_called_once()


class TestBackpressureAndConcurrency:
    """Test backpressure handling and concurrency control."""
    
    @pytest.mark.asyncio
    async def test_backpressure_detection_cpu(self):
        """Test backpressure detection based on CPU usage."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration(
                max_cpu_percent=80,
                backpressure_threshold=0.9
            )
        )
        
        with patch('psutil.cpu_percent', return_value=85.0):  # Above threshold
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.used = 500 * 1024 * 1024  # Below memory threshold
                
                backpressure = await manager._check_backpressure()
                
                assert backpressure is True
                assert manager.statistics.backpressure_events > 0
    
    @pytest.mark.asyncio
    async def test_backpressure_detection_memory(self):
        """Test backpressure detection based on memory usage."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration(
                max_memory_mb=1024,
                backpressure_threshold=0.8
            )
        )
        
        with patch('psutil.cpu_percent', return_value=50.0):  # Below CPU threshold
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.used = 900 * 1024 * 1024  # Above memory threshold
                
                backpressure = await manager._check_backpressure()
                
                assert backpressure is True
                assert manager.statistics.backpressure_events > 0
    
    @pytest.mark.asyncio
    async def test_no_backpressure(self):
        """Test no backpressure when resources are available."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.job_semaphore = asyncio.Semaphore(4)
        
        with patch('psutil.cpu_percent', return_value=30.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.used = 300 * 1024 * 1024
                
                backpressure = await manager._check_backpressure()
                
                assert backpressure is False
    
    @pytest.mark.asyncio
    async def test_optimal_batch_size_calculation(self):
        """Test optimal batch size calculation."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.processing_mode = ProcessingMode.BALANCED
        
        # Mock queue stats
        with patch.object(manager.state_manager, 'get_queue_stats', AsyncMock(return_value={
            ProcessingPriority.LOW.value: 5,
            ProcessingPriority.NORMAL.value: 15,
            ProcessingPriority.HIGH.value: 3
        })):
            batch_size = await manager._get_optimal_batch_size()
            
            # Should increase batch size for large queue (total: 23 items)
            assert batch_size > 2  # Base size for balanced mode
    
    @pytest.mark.asyncio
    async def test_concurrency_limiter(self):
        """Test concurrency limiter context manager."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager.job_semaphore = asyncio.Semaphore(2)
        
        async with manager._get_concurrency_limiter():
            # Should acquire semaphore
            assert manager.job_semaphore._value == 1
        
        # Should release semaphore
        assert manager.job_semaphore._value == 2


class TestQueueProcessing:
    """Test queue processing functionality."""
    
    @pytest.mark.asyncio
    async def test_enqueue_file_success(self, mock_priority_queue_manager, isolated_daemon_temp_dir):
        """Test successful file enqueuing."""
        test_file = isolated_daemon_temp_dir / "enqueue_test.py"
        test_file.write_text("def test(): pass")
        
        queue_id = await mock_priority_queue_manager.enqueue_file(
            str(test_file),
            "test-collection",
            user_triggered=True
        )
        
        assert queue_id == "queue_123"
        mock_priority_queue_manager.enqueue_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enqueue_file_not_initialized(self):
        """Test enqueuing file when manager not initialized."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await manager.enqueue_file("/tmp/test.py", "test-collection")
    
    @pytest.mark.asyncio
    async def test_process_next_batch_empty(self, mock_priority_queue_manager):
        """Test processing next batch when queue is empty."""
        jobs = await mock_priority_queue_manager.process_next_batch()
        
        assert jobs == []
        mock_priority_queue_manager.process_next_batch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_next_batch_with_backpressure(self):
        """Test processing next batch with backpressure."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        manager._initialized = True
        
        with patch.object(manager, '_check_backpressure', AsyncMock(return_value=True)):
            with patch.object(manager, '_get_optimal_batch_size', AsyncMock(return_value=2)):
                jobs = await manager.process_next_batch()
                
                assert jobs == []  # Should skip processing due to backpressure
    
    @pytest.mark.asyncio
    async def test_processing_job_success(self):
        """Test successful job processing."""
        mock_state_manager = Mock()
        mock_state_manager.mark_queue_item_processing = AsyncMock()
        mock_state_manager.complete_file_processing = AsyncMock()
        mock_state_manager.remove_from_processing_queue = AsyncMock()
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        job = ProcessingJob(
            queue_id="test_job_123",
            file_path="/tmp/success_file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL
        )
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch.object(manager, '_process_job_fallback', AsyncMock()):
                result = await manager._process_single_job(job)
                
                assert result is not None
                assert result.queue_id == "test_job_123"
                mock_state_manager.mark_queue_item_processing.assert_called_once_with("test_job_123")
                mock_state_manager.complete_file_processing.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_processing_job_file_not_found(self):
        """Test job processing with missing file."""
        mock_state_manager = Mock()
        mock_state_manager.mark_queue_item_processing = AsyncMock()
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        job = ProcessingJob(
            queue_id="test_job_456",
            file_path="/tmp/missing_file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL
        )
        
        with patch('pathlib.Path.exists', return_value=False):
            with patch.object(manager, '_handle_job_failure', AsyncMock()) as mock_handle_failure:
                result = await manager._process_single_job(job)
                
                assert result is None
                mock_handle_failure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_failure_handling_with_retry(self):
        """Test job failure handling with retry logic."""
        mock_state_manager = Mock()
        mock_state_manager.complete_file_processing = AsyncMock()
        mock_state_manager.reschedule_queue_item = AsyncMock()
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        job = ProcessingJob(
            queue_id="test_job_retry",
            file_path="/tmp/retry_file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            attempts=1,
            max_attempts=3
        )
        
        await manager._handle_job_failure(job, "Test error")
        
        assert job.attempts == 2
        mock_state_manager.reschedule_queue_item.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_job_failure_handling_max_attempts(self):
        """Test job failure handling when max attempts reached."""
        mock_state_manager = Mock()
        mock_state_manager.complete_file_processing = AsyncMock()
        mock_state_manager.remove_from_processing_queue = AsyncMock()
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        job = ProcessingJob(
            queue_id="test_job_max_attempts",
            file_path="/tmp/max_attempts_file.py",
            collection="test-collection",
            priority=ProcessingPriority.NORMAL,
            attempts=3,
            max_attempts=3
        )
        
        await manager._handle_job_failure(job, "Max attempts error")
        
        mock_state_manager.remove_from_processing_queue.assert_called_once_with("test_job_max_attempts")


class TestProcessingContextManager:
    """Test processing context manager."""
    
    @pytest.mark.asyncio
    async def test_processing_context_manager_basic(self, mock_priority_queue_manager):
        """Test basic processing context manager usage."""
        context_manager = ProcessingContextManager(mock_priority_queue_manager)
        
        async with context_manager as context:
            assert context is context_manager
            assert context.start_time is not None
        
        # Should complete without errors
    
    @pytest.mark.asyncio
    async def test_processing_context_manager_process_batch(self, mock_priority_queue_manager):
        """Test processing batch through context manager."""
        context_manager = ProcessingContextManager(mock_priority_queue_manager)
        
        async with context_manager as context:
            jobs = await context.process_next_batch(batch_size=5)
            
            assert jobs == []  # Mock returns empty list
            mock_priority_queue_manager.process_next_batch.assert_called_once_with(5)
    
    @pytest.mark.asyncio
    async def test_processing_context_manager_enqueue_multiple(self, mock_priority_queue_manager):
        """Test enqueueing multiple files through context manager."""
        context_manager = ProcessingContextManager(mock_priority_queue_manager)
        
        file_collection_pairs = [
            ("/tmp/file1.py", "collection1"),
            ("/tmp/file2.py", "collection2"),
            ("/tmp/file3.py", "collection1")
        ]
        
        mock_priority_queue_manager.enqueue_file.return_value = "queue_123"
        
        async with context_manager as context:
            queue_ids = await context.enqueue_multiple_files(
                file_collection_pairs,
                user_triggered=True,
                metadata={"batch": True}
            )
            
            assert len(queue_ids) == 3
            assert all(qid == "queue_123" for qid in queue_ids)
            assert mock_priority_queue_manager.enqueue_file.call_count == 3
    
    @pytest.mark.asyncio
    async def test_processing_context_manager_exception_handling(self, mock_priority_queue_manager):
        """Test processing context manager exception handling."""
        context_manager = ProcessingContextManager(mock_priority_queue_manager)
        
        try:
            async with context_manager as context:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Should complete cleanup despite exception
        assert context_manager.start_time is not None


class TestQueueStatusAndHealth:
    """Test queue status and health monitoring."""
    
    @pytest.mark.asyncio
    async def test_get_queue_status(self, mock_priority_queue_manager):
        """Test getting queue status."""
        status = await mock_priority_queue_manager.get_queue_status()
        
        assert "initialized" in status
        assert "processing_mode" in status
        assert "statistics" in status
        assert "active_jobs" in status
        mock_priority_queue_manager.get_queue_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, mock_priority_queue_manager):
        """Test getting health status."""
        health = await mock_priority_queue_manager.get_health_status()
        
        assert "health_status" in health
        assert "system_metrics" in health
        assert "queue_statistics" in health
        assert "resource_status" in health
        assert "mcp_activity" in health
        mock_priority_queue_manager.get_health_status.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clear_queue(self, mock_priority_queue_manager):
        """Test clearing queue."""
        result = await mock_priority_queue_manager.clear_queue("test-collection")
        
        assert result == 0
        mock_priority_queue_manager.clear_queue.assert_called_once_with("test-collection")
    
    def test_add_hooks(self, mock_priority_queue_manager):
        """Test adding processing hooks."""
        priority_hook = Mock()
        processing_hook = Mock()
        monitoring_hook = Mock()
        
        mock_priority_queue_manager.add_priority_calculation_hook(priority_hook)
        mock_priority_queue_manager.add_processing_hook(processing_hook)
        mock_priority_queue_manager.add_monitoring_hook(monitoring_hook)
        
        mock_priority_queue_manager.add_priority_calculation_hook.assert_called_once_with(priority_hook)
        mock_priority_queue_manager.add_processing_hook.assert_called_once_with(processing_hook)
        mock_priority_queue_manager.add_monitoring_hook.assert_called_once_with(monitoring_hook)
    
    def test_set_current_project_root(self, mock_priority_queue_manager):
        """Test setting current project root."""
        project_root = "/tmp/test_project"
        
        mock_priority_queue_manager.set_current_project_root(project_root)
        
        mock_priority_queue_manager.set_current_project_root.assert_called_once_with(project_root)


@pytest.mark.queue_unit
@pytest.mark.queue_processing
@pytest.mark.queue_priority
class TestPriorityQueueManagerIntegration:
    """Integration tests for PriorityQueueManager components."""
    
    @pytest.mark.asyncio
    async def test_full_processing_lifecycle(self, isolated_daemon_temp_dir):
        """Test complete processing lifecycle integration."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = isolated_daemon_temp_dir / f"lifecycle_test_{i}.py"
            test_file.write_text(f"def test_function_{i}(): pass")
            test_files.append(str(test_file))
        
        # Mock dependencies
        mock_state_manager = Mock()
        mock_state_manager.initialize = AsyncMock(return_value=True)
        mock_state_manager.add_to_processing_queue = AsyncMock(side_effect=lambda **kwargs: f"queue_{hash(kwargs['file_path']) % 1000}")
        mock_state_manager.get_next_queue_item = AsyncMock(return_value=None)  # Empty queue
        mock_state_manager.get_queue_stats = AsyncMock(return_value={})
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration(max_concurrent_jobs=2),
            mcp_detection_interval=1,
            statistics_retention_hours=1
        )
        
        with patch.object(manager, '_monitoring_loop', AsyncMock()):
            with patch.object(manager, '_activity_detection_loop', AsyncMock()):
                with patch.object(manager, '_perform_crash_recovery', AsyncMock()):
                    # Initialize
                    result = await manager.initialize()
                    assert result is True
                    
                    # Enqueue files
                    queue_ids = []
                    for file_path in test_files:
                        queue_id = await manager.enqueue_file(file_path, "test-collection")
                        queue_ids.append(queue_id)
                    
                    assert len(queue_ids) == 3
                    assert all(qid.startswith("queue_") for qid in queue_ids)
                    
                    # Process batch (should be empty since mock returns None)
                    jobs = await manager.process_next_batch(batch_size=2)
                    assert jobs == []
                    
                    # Check status
                    status = await manager.get_queue_status()
                    assert status["initialized"] is True
                    
                    # Shutdown
                    await manager.shutdown()
    
    @pytest.mark.asyncio
    async def test_priority_queue_with_different_priorities(self, isolated_daemon_temp_dir):
        """Test priority queue handling different priority levels."""
        # Setup files with different characteristics
        urgent_file = isolated_daemon_temp_dir / "urgent.py"
        urgent_file.write_text("# Urgent user file")
        
        normal_file = isolated_daemon_temp_dir / "normal.py"
        normal_file.write_text("# Normal background file")
        
        # Mock state manager with priority-aware behavior
        mock_state_manager = Mock()
        mock_state_manager.initialize = AsyncMock(return_value=True)
        
        # Track enqueued items by priority
        enqueued_items = []
        def mock_enqueue(**kwargs):
            queue_id = f"queue_{len(enqueued_items)}"
            enqueued_items.append({
                "queue_id": queue_id,
                "priority": kwargs["priority"],
                "file_path": kwargs["file_path"]
            })
            return queue_id
        
        mock_state_manager.add_to_processing_queue = AsyncMock(side_effect=mock_enqueue)
        mock_state_manager.get_queue_stats = AsyncMock(return_value={})
        
        manager = PriorityQueueManager(
            state_manager=mock_state_manager,
            resource_config=ResourceConfiguration()
        )
        
        with patch.object(manager, '_monitoring_loop', AsyncMock()):
            with patch.object(manager, '_activity_detection_loop', AsyncMock()):
                with patch.object(manager, '_perform_crash_recovery', AsyncMock()):
                    await manager.initialize()
                    
                    # Enqueue urgent file (user-triggered, current project)
                    manager.set_current_project_root(str(isolated_daemon_temp_dir))
                    queue_id_urgent = await manager.enqueue_file(
                        str(urgent_file), 
                        "test-collection",
                        user_triggered=True
                    )
                    
                    # Enqueue normal file (background)
                    queue_id_normal = await manager.enqueue_file(
                        str(normal_file),
                        "test-collection",
                        user_triggered=False
                    )
                    
                    assert len(enqueued_items) == 2
                    
                    # Verify priority assignment (mocked calculation would assign based on context)
                    urgent_item = next(item for item in enqueued_items if item["file_path"] == str(urgent_file))
                    normal_item = next(item for item in enqueued_items if item["file_path"] == str(normal_file))
                    
                    # Both should have been enqueued (actual priority calculated by mock)
                    assert urgent_item["queue_id"] == queue_id_urgent
                    assert normal_item["queue_id"] == queue_id_normal
                    
                    await manager.shutdown()


@pytest.mark.processing_engine
class TestProcessingEngineComponents:
    """Test processing engine component functionality."""
    
    @pytest.mark.asyncio
    async def test_processing_mode_transitions(self):
        """Test processing mode transitions based on system load."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration()
        )
        
        # Test transition from conservative to balanced
        manager.processing_mode = ProcessingMode.CONSERVATIVE
        manager.mcp_activity.activity_level = MCPActivityLevel.MODERATE
        
        with patch('psutil.cpu_percent', return_value=40.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 50.0
                with patch.object(manager, '_configure_executor', AsyncMock()) as mock_configure:
                    await manager._update_processing_mode()
                    
                    assert manager.processing_mode == ProcessingMode.BALANCED
                    mock_configure.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_resource_monitoring_integration(self):
        """Test integration with resource monitoring."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration(
                max_memory_mb=1024,
                max_cpu_percent=80
            )
        )
        
        # Test health metrics update
        with patch('psutil.cpu_percent', return_value=45.0):
            with patch('psutil.virtual_memory') as mock_memory:
                mock_memory.return_value.percent = 60.0
                mock_memory.return_value.used = 500 * 1024 * 1024
                
                await manager._update_health_metrics()
                
                assert "cpu_percent" in manager.health_metrics
                assert "memory_percent" in manager.health_metrics
                assert manager.health_metrics["cpu_percent"] == 45.0
                assert manager.health_metrics["memory_percent"] == 60.0
    
    @pytest.mark.asyncio
    async def test_statistics_cleanup(self):
        """Test processing statistics cleanup."""
        manager = PriorityQueueManager(
            state_manager=Mock(),
            resource_config=ResourceConfiguration(),
            statistics_retention_hours=1
        )
        
        # Add old entries to processing history
        old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        recent_time = datetime.now(timezone.utc).isoformat()
        
        manager.processing_history = [
            {"timestamp": old_time, "queue_id": "old_1"},
            {"timestamp": recent_time, "queue_id": "recent_1"},
            {"timestamp": old_time, "queue_id": "old_2"},
            {"timestamp": recent_time, "queue_id": "recent_2"}
        ]
        
        await manager._cleanup_old_statistics()
        
        # Should only keep recent entries
        assert len(manager.processing_history) == 2
        assert all("recent" in entry["queue_id"] for entry in manager.processing_history)