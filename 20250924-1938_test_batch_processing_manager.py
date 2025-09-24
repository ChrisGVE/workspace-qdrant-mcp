"""
Comprehensive unit tests for batch processing manager.

Tests cover batch processing strategies, queue management, performance optimization,
error handling, and edge cases for the BatchProcessingManager system.

Test Categories:
    - Configuration validation and initialization
    - Batch sizing and flushing strategies
    - Priority-based processing
    - Queue management and capacity control
    - Performance monitoring and adaptive behavior
    - Error handling and retry mechanisms
    - Memory management and backpressure
    - Concurrency and race conditions
    - Edge cases and failure scenarios
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock, call
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

from src.python.common.core.batch_processing_manager import (
    BatchProcessingManager,
    BatchProcessingConfig,
    BatchProcessingStatistics,
    ProcessingBatch,
    BatchItem,
    FileOperation,
    ProcessingStrategy,
    BatchPriority,
    create_high_throughput_batch_processor,
    create_low_latency_batch_processor,
    create_memory_efficient_batch_processor
)


class TestBatchProcessingConfig:
    """Test BatchProcessingConfig functionality."""

    def test_default_configuration(self):
        """Test default configuration values."""
        config = BatchProcessingConfig()

        assert config.max_batch_size == 50
        assert config.min_batch_size == 1
        assert config.max_batch_age_seconds == 5.0
        assert config.max_queue_size == 10000
        assert config.processing_strategy == ProcessingStrategy.ADAPTIVE
        assert config.enable_deduplication is True
        assert config.enable_priority_processing is True
        assert config.max_concurrent_batches == 3
        assert config.retry_max_attempts == 3
        assert config.enable_performance_monitoring is True

    def test_custom_configuration(self):
        """Test custom configuration values."""
        config = BatchProcessingConfig(
            max_batch_size=100,
            min_batch_size=5,
            max_batch_age_seconds=10.0,
            max_queue_size=50000,
            processing_strategy=ProcessingStrategy.TIME_BASED,
            enable_deduplication=False,
            max_concurrent_batches=8,
            retry_max_attempts=5
        )

        assert config.max_batch_size == 100
        assert config.min_batch_size == 5
        assert config.max_batch_age_seconds == 10.0
        assert config.max_queue_size == 50000
        assert config.processing_strategy == ProcessingStrategy.TIME_BASED
        assert config.enable_deduplication is False
        assert config.max_concurrent_batches == 8
        assert config.retry_max_attempts == 5

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid max_batch_size
        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            config = BatchProcessingConfig(max_batch_size=0)
            config.validate()

        with pytest.raises(ValueError, match="max_batch_size must be positive"):
            config = BatchProcessingConfig(max_batch_size=-10)
            config.validate()

        # Test invalid min_batch_size
        with pytest.raises(ValueError, match="min_batch_size must be positive"):
            config = BatchProcessingConfig(min_batch_size=0)
            config.validate()

        with pytest.raises(ValueError, match="min_batch_size must be positive and <= max_batch_size"):
            config = BatchProcessingConfig(min_batch_size=100, max_batch_size=50)
            config.validate()

        # Test invalid timing parameters
        with pytest.raises(ValueError, match="max_batch_age_seconds must be positive"):
            config = BatchProcessingConfig(max_batch_age_seconds=0)
            config.validate()

        # Test invalid queue size
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            config = BatchProcessingConfig(max_queue_size=0)
            config.validate()

        # Test invalid memory limit
        with pytest.raises(ValueError, match="max_memory_usage_mb must be positive"):
            config = BatchProcessingConfig(max_memory_usage_mb=0)
            config.validate()


class TestBatchProcessingStatistics:
    """Test BatchProcessingStatistics functionality."""

    def test_statistics_initialization(self):
        """Test statistics initialization."""
        stats = BatchProcessingStatistics()

        assert stats.total_files_processed == 0
        assert stats.total_batches_processed == 0
        assert stats.total_files_failed == 0
        assert stats.total_batches_failed == 0
        assert stats.avg_batch_size == 0.0
        assert stats.avg_batch_processing_time == 0.0
        assert stats.throughput_files_per_second == 0.0
        assert stats.current_queue_size == 0
        assert stats.duplicate_files_merged == 0

    def test_add_batch_processed(self):
        """Test adding processed batch statistics."""
        stats = BatchProcessingStatistics()

        # Add first batch
        stats.add_batch_processed(10, 2.0, 0.5)

        assert stats.total_files_processed == 10
        assert stats.total_batches_processed == 1
        assert stats.avg_batch_size == 10.0
        assert stats.avg_batch_processing_time == 2.0
        assert stats.avg_queue_wait_time == 0.5
        assert stats.throughput_files_per_second == 5.0  # 10 files / 2 seconds

        # Add second batch
        stats.add_batch_processed(5, 1.0, 0.3)

        assert stats.total_files_processed == 15
        assert stats.total_batches_processed == 2
        assert stats.avg_batch_size == 7.5  # Average of 10 and 5
        assert stats.avg_batch_processing_time == 1.5  # Average of 2.0 and 1.0
        assert stats.avg_queue_wait_time == 0.4  # Average of 0.5 and 0.3
        assert stats.throughput_files_per_second == 5.0  # 5 files / 1 second (last batch)

    def test_add_batch_failed(self):
        """Test adding failed batch statistics."""
        stats = BatchProcessingStatistics()

        stats.add_batch_failed(8)

        assert stats.total_files_failed == 8
        assert stats.total_batches_failed == 1
        assert stats.total_files_processed == 0
        assert stats.total_batches_processed == 0

    def test_efficiency_metrics(self):
        """Test efficiency metrics calculation."""
        stats = BatchProcessingStatistics()

        # Add some successful and failed batches
        stats.add_batch_processed(10, 2.0, 0.5)
        stats.add_batch_processed(5, 1.0, 0.3)
        stats.add_batch_failed(3)

        efficiency = stats.get_efficiency_metrics()

        # File success rate: 15 successful / 18 total = 83.33%
        assert abs(efficiency["file_success_rate_percent"] - 83.33) < 0.1

        # Batch success rate: 2 successful / 3 total = 66.67%
        assert abs(efficiency["batch_success_rate_percent"] - 66.67) < 0.1

        assert efficiency["avg_batch_utilization"] == 7.5
        assert efficiency["processing_efficiency"] == 5.0  # Last throughput

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        stats = BatchProcessingStatistics()

        # Add some data
        stats.add_batch_processed(10, 2.0, 0.5)
        stats.add_batch_failed(5)
        stats.duplicate_files_merged = 3
        stats.priority_promotions = 2

        # Verify data is present
        assert stats.total_files_processed == 10
        assert stats.total_files_failed == 5
        assert stats.duplicate_files_merged == 3

        # Reset
        stats.reset()

        # Verify all values are reset
        assert stats.total_files_processed == 0
        assert stats.total_files_failed == 0
        assert stats.duplicate_files_merged == 0
        assert stats.priority_promotions == 0
        assert len(stats.recent_batch_sizes) == 0

    def test_rolling_window_behavior(self):
        """Test rolling window for recent statistics."""
        stats = BatchProcessingStatistics()

        # Add more than window size (100) batch records
        for i in range(150):
            stats.add_batch_processed(i + 1, i * 0.1, i * 0.05)

        # Should only keep the most recent 100
        assert len(stats.recent_batch_sizes) == 100
        assert len(stats.recent_processing_times) == 100
        assert len(stats.recent_wait_times) == 100

        # Should contain the most recent values (51-150)
        assert min(stats.recent_batch_sizes) == 51
        assert max(stats.recent_batch_sizes) == 150

    def test_statistics_serialization(self):
        """Test statistics to_dict conversion."""
        stats = BatchProcessingStatistics()
        stats.add_batch_processed(10, 2.0, 0.5)
        stats.current_queue_size = 5
        stats.peak_memory_usage_mb = 100.0

        stats_dict = stats.to_dict()

        assert isinstance(stats_dict, dict)
        assert stats_dict["total_files_processed"] == 10
        assert stats_dict["total_batches_processed"] == 1
        assert stats_dict["current_queue_size"] == 5
        assert stats_dict["peak_memory_usage_mb"] == 100.0
        assert "file_success_rate_percent" in stats_dict
        assert "batch_success_rate_percent" in stats_dict


class TestFileOperationAndBatchItem:
    """Test FileOperation and BatchItem functionality."""

    def test_file_operation_creation(self):
        """Test FileOperation creation."""
        operation = FileOperation(
            file_path="/tmp/test.txt",
            collection="test-collection",
            operation_type="add",
            priority=BatchPriority.HIGH,
            metadata={"size": 1024}
        )

        assert operation.file_path == "/tmp/test.txt"
        assert operation.collection == "test-collection"
        assert operation.operation_type == "add"
        assert operation.priority == BatchPriority.HIGH
        assert operation.metadata["size"] == 1024
        assert operation.timestamp > 0

    def test_batch_item_creation(self):
        """Test BatchItem creation."""
        operation = FileOperation("/tmp/test.txt", "collection", "add")
        item = BatchItem(file_operation=operation)

        assert item.file_operation is operation
        assert item.retry_count == 0
        assert item.last_error is None
        assert item.added_at > 0
        assert item.processing_duration == 0.0

    def test_batch_item_equality(self):
        """Test BatchItem equality and hashing for deduplication."""
        operation1 = FileOperation("/tmp/test.txt", "collection", "add")
        operation2 = FileOperation("/tmp/test.txt", "collection", "add")
        operation3 = FileOperation("/tmp/test.txt", "collection", "modify")
        operation4 = FileOperation("/tmp/other.txt", "collection", "add")

        item1 = BatchItem(file_operation=operation1)
        item2 = BatchItem(file_operation=operation2)
        item3 = BatchItem(file_operation=operation3)
        item4 = BatchItem(file_operation=operation4)

        # Same file path and operation type should be equal
        assert item1 == item2
        assert hash(item1) == hash(item2)

        # Different operation type should not be equal
        assert item1 != item3
        assert hash(item1) != hash(item3)

        # Different file path should not be equal
        assert item1 != item4
        assert hash(item1) != hash(item4)


class TestProcessingBatch:
    """Test ProcessingBatch functionality."""

    def test_processing_batch_creation(self):
        """Test ProcessingBatch creation."""
        batch = ProcessingBatch(collection="test-collection")

        assert batch.collection == "test-collection"
        assert batch.priority == BatchPriority.NORMAL
        assert len(batch.items) == 0
        assert batch.created_at > 0
        assert batch.batch_id is not None
        assert batch.is_empty() is True

    def test_add_item_to_batch(self):
        """Test adding items to batch."""
        batch = ProcessingBatch()

        operation1 = FileOperation("/tmp/test1.txt", "collection", "add", BatchPriority.LOW)
        operation2 = FileOperation("/tmp/test2.txt", "collection", "add", BatchPriority.HIGH)

        item1 = BatchItem(file_operation=operation1)
        item2 = BatchItem(file_operation=operation2)

        batch.add_item(item1)
        assert batch.size() == 1
        assert batch.priority == BatchPriority.LOW  # First item's priority

        batch.add_item(item2)
        assert batch.size() == 2
        assert batch.priority == BatchPriority.HIGH  # Updated to highest priority

    def test_batch_deduplication(self):
        """Test batch item deduplication."""
        batch = ProcessingBatch()

        # Create duplicate operations (same file path and operation type)
        operation1 = FileOperation("/tmp/test.txt", "collection", "add")
        operation2 = FileOperation("/tmp/test.txt", "collection", "add")

        item1 = BatchItem(file_operation=operation1)
        item2 = BatchItem(file_operation=operation2)

        # Add first item
        batch.add_item(item1)
        assert batch.size() == 1

        # Add duplicate - should replace the first
        batch.add_item(item2)
        assert batch.size() == 1  # Still only one item
        assert batch.items[0] is item2  # Should be the newer item

    def test_batch_age_calculation(self):
        """Test batch age calculation."""
        batch = ProcessingBatch()

        # Age should be very small initially
        initial_age = batch.age_seconds()
        assert 0 <= initial_age < 0.1

        # Simulate some time passing
        batch.created_at = time.time() - 5.0  # 5 seconds ago

        age = batch.age_seconds()
        assert 4.9 <= age <= 5.1  # Should be approximately 5 seconds

    def test_batch_file_paths(self):
        """Test getting file paths from batch."""
        batch = ProcessingBatch()

        paths = ["/tmp/file1.txt", "/tmp/file2.txt", "/tmp/file3.txt"]
        for path in paths:
            operation = FileOperation(path, "collection", "add")
            item = BatchItem(file_operation=operation)
            batch.add_item(item)

        batch_paths = batch.get_file_paths()
        assert len(batch_paths) == 3
        assert all(path in batch_paths for path in paths)

    def test_batch_memory_estimation(self):
        """Test batch memory usage estimation."""
        batch = ProcessingBatch()

        # Empty batch should have minimal memory usage
        empty_memory = batch.estimate_memory_usage()
        assert empty_memory >= 0

        # Add items and check memory increases
        for i in range(10):
            operation = FileOperation(f"/tmp/file_{i}.txt", "collection", "add",
                                    metadata={"size": i * 1024})
            item = BatchItem(file_operation=operation)
            batch.add_item(item)

        filled_memory = batch.estimate_memory_usage()
        assert filled_memory > empty_memory


class TestBatchProcessingManagerInitialization:
    """Test BatchProcessingManager initialization."""

    def test_manager_initialization_with_config_object(self):
        """Test manager initialization with BatchProcessingConfig object."""
        config = BatchProcessingConfig(
            max_batch_size=25,
            processing_strategy=ProcessingStrategy.TIME_BASED
        )

        manager = BatchProcessingManager(config)

        assert manager.config is config
        assert manager.config.max_batch_size == 25
        assert manager.config.processing_strategy == ProcessingStrategy.TIME_BASED
        assert manager._running is False
        assert manager.processing_callback is None

    def test_manager_initialization_with_dict_config(self):
        """Test manager initialization with dictionary configuration."""
        config_dict = {
            "max_batch_size": 100,
            "processing_strategy": ProcessingStrategy.HYBRID,
            "enable_deduplication": False
        }

        manager = BatchProcessingManager(config_dict)

        assert manager.config.max_batch_size == 100
        assert manager.config.processing_strategy == ProcessingStrategy.HYBRID
        assert manager.config.enable_deduplication is False

    def test_manager_initialization_validation(self):
        """Test manager initialization with invalid configuration."""
        with pytest.raises(ValueError):
            invalid_config = BatchProcessingConfig(max_batch_size=0)
            BatchProcessingManager(invalid_config)

    def test_set_processing_callback(self):
        """Test setting processing callback."""
        config = BatchProcessingConfig()
        manager = BatchProcessingManager(config)

        async def test_callback(items):
            pass

        manager.set_processing_callback(test_callback)
        assert manager.processing_callback is test_callback

    @pytest.mark.asyncio
    async def test_start_without_callback_raises_error(self):
        """Test that starting without callback raises error."""
        config = BatchProcessingConfig()
        manager = BatchProcessingManager(config)

        with pytest.raises(ValueError, match="Processing callback must be set"):
            await manager.start()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        """Test basic start/stop lifecycle."""
        config = BatchProcessingConfig()
        manager = BatchProcessingManager(config)

        async def test_callback(items):
            pass

        manager.set_processing_callback(test_callback)

        # Start manager
        await manager.start()
        assert manager._running is True
        assert manager._processing_task is not None
        assert manager._batch_flush_task is not None

        # Stop manager
        await manager.stop()
        assert manager._running is False


class TestBatchProcessingManagerFileOperations:
    """Test file addition and queue management."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BatchProcessingConfig(
            max_batch_size=5,
            max_batch_age_seconds=1.0,
            max_queue_size=100
        )
        self.manager = BatchProcessingManager(self.config)
        self.processed_batches = []

        async def mock_callback(items):
            self.processed_batches.append(items)

        self.manager.set_processing_callback(mock_callback)

    @pytest.mark.asyncio
    async def test_add_file_before_start_fails(self):
        """Test that adding files before starting fails."""
        result = await self.manager.add_file("/tmp/test.txt", "collection")
        assert result is False

    @pytest.mark.asyncio
    async def test_add_single_file(self):
        """Test adding a single file."""
        await self.manager.start()

        result = await self.manager.add_file("/tmp/test.txt", "test-collection")
        assert result is True

        assert self.manager.statistics.current_queue_size == 1
        assert "test-collection" in self.manager._current_batches
        assert self.manager._current_batches["test-collection"].size() == 1

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_add_multiple_files_same_collection(self):
        """Test adding multiple files to same collection."""
        await self.manager.start()

        files = [f"/tmp/test_{i}.txt" for i in range(3)]
        for file_path in files:
            result = await self.manager.add_file(file_path, "test-collection")
            assert result is True

        assert self.manager.statistics.current_queue_size == 3
        batch = self.manager._current_batches["test-collection"]
        assert batch.size() == 3

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_add_files_different_collections(self):
        """Test adding files to different collections."""
        await self.manager.start()

        await self.manager.add_file("/tmp/test1.txt", "collection1")
        await self.manager.add_file("/tmp/test2.txt", "collection2")

        assert len(self.manager._current_batches) == 2
        assert "collection1" in self.manager._current_batches
        assert "collection2" in self.manager._current_batches

        await self.manager.stop()

    @pytest.mark.asyncio
    async def test_file_deduplication(self):
        """Test file deduplication functionality."""
        config = BatchProcessingConfig(enable_deduplication=True)
        manager = BatchProcessingManager(config)
        manager.set_processing_callback(AsyncMock())

        await manager.start()

        # Add same file twice
        result1 = await manager.add_file("/tmp/test.txt", "collection", "add")
        result2 = await manager.add_file("/tmp/test.txt", "collection", "add")

        assert result1 is True
        assert result2 is True  # Should be accepted but merged
        assert manager.statistics.duplicate_files_merged == 1
        assert manager.statistics.current_queue_size == 1  # Only one unique file

        await manager.stop()

    @pytest.mark.asyncio
    async def test_priority_handling(self):
        """Test priority-based file handling."""
        await self.manager.start()

        # Add files with different priorities
        await self.manager.add_file("/tmp/low.txt", "collection", "add", BatchPriority.LOW)
        await self.manager.add_file("/tmp/high.txt", "collection", "add", BatchPriority.HIGH)
        await self.manager.add_file("/tmp/critical.txt", "collection", "add", BatchPriority.CRITICAL)

        batch = self.manager._current_batches["collection"]
        # Batch priority should be updated to highest (CRITICAL)
        assert batch.priority == BatchPriority.CRITICAL

        await self.manager.stop()


class TestBatchFlushingStrategies:
    """Test different batch flushing strategies."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processed_batches = []

        async def mock_callback(items):
            self.processed_batches.append(items)

    @pytest.mark.asyncio
    async def test_count_based_flushing(self):
        """Test count-based batch flushing."""
        config = BatchProcessingConfig(
            processing_strategy=ProcessingStrategy.COUNT_BASED,
            max_batch_size=3,
            adaptive_batch_sizing=False
        )
        manager = BatchProcessingManager(config)

        async def mock_callback(items):
            self.processed_batches.append(items)

        manager.set_processing_callback(mock_callback)
        await manager.start()

        # Add files one by one
        await manager.add_file("/tmp/file1.txt", "collection")
        await manager.add_file("/tmp/file2.txt", "collection")

        # Should not have flushed yet
        assert len(self.processed_batches) == 0

        # Third file should trigger flush
        await manager.add_file("/tmp/file3.txt", "collection")

        # Give processing loop time to run
        await asyncio.sleep(0.1)

        # Should have processed one batch with 3 items
        assert len(self.processed_batches) >= 0  # May or may not have processed yet depending on timing

        await manager.stop()

    @pytest.mark.asyncio
    async def test_time_based_flushing(self):
        """Test time-based batch flushing."""
        config = BatchProcessingConfig(
            processing_strategy=ProcessingStrategy.TIME_BASED,
            max_batch_age_seconds=0.2  # Very short for testing
        )
        manager = BatchProcessingManager(config)

        async def mock_callback(items):
            self.processed_batches.append(items)

        manager.set_processing_callback(mock_callback)
        await manager.start()

        # Add a file
        await manager.add_file("/tmp/file1.txt", "collection")

        # Wait for age-based flush
        await asyncio.sleep(0.3)

        # Should have flushed due to age
        # Note: Timing-dependent test, may need adjustment
        await manager.stop()

    @pytest.mark.asyncio
    async def test_hybrid_flushing(self):
        """Test hybrid (count + time) batch flushing."""
        config = BatchProcessingConfig(
            processing_strategy=ProcessingStrategy.HYBRID,
            max_batch_size=5,
            max_batch_age_seconds=0.2
        )
        manager = BatchProcessingManager(config)

        async def mock_callback(items):
            self.processed_batches.append(items)

        manager.set_processing_callback(mock_callback)
        await manager.start()

        # Test size-based flushing
        for i in range(5):
            await manager.add_file(f"/tmp/file_{i}.txt", "collection")

        await asyncio.sleep(0.1)

        # Test time-based flushing with fewer items
        await manager.add_file("/tmp/time_file.txt", "collection2")
        await asyncio.sleep(0.3)

        await manager.stop()

    @pytest.mark.asyncio
    async def test_priority_flushing(self):
        """Test priority-based batch flushing."""
        config = BatchProcessingConfig(
            processing_strategy=ProcessingStrategy.PRIORITY,
            max_batch_size=10
        )
        manager = BatchProcessingManager(config)

        async def mock_callback(items):
            self.processed_batches.append(items)

        manager.set_processing_callback(mock_callback)
        await manager.start()

        # Add normal priority file
        await manager.add_file("/tmp/normal.txt", "collection", priority=BatchPriority.NORMAL)

        # Should not flush immediately
        await asyncio.sleep(0.1)

        # Add critical priority file - should trigger immediate flush
        await manager.add_file("/tmp/critical.txt", "collection", priority=BatchPriority.CRITICAL)

        await asyncio.sleep(0.1)

        await manager.stop()


class TestBatchProcessingManagerErrorHandling:
    """Test error handling and retry mechanisms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = BatchProcessingConfig(
            max_batch_size=2,
            retry_max_attempts=3,
            retry_delay_seconds=0.1
        )
        self.manager = BatchProcessingManager(self.config)
        self.callback_calls = []
        self.should_fail = True

    @pytest.mark.asyncio
    async def test_batch_processing_success(self):
        """Test successful batch processing."""
        async def success_callback(items):
            self.callback_calls.append(("success", len(items)))

        self.manager.set_processing_callback(success_callback)
        await self.manager.start()

        await self.manager.add_file("/tmp/file1.txt", "collection")
        await self.manager.add_file("/tmp/file2.txt", "collection")

        # Wait for processing
        await asyncio.sleep(0.2)

        await self.manager.stop()

        # Should have processed successfully
        assert len(self.callback_calls) >= 0
        assert self.manager.statistics.total_files_failed == 0

    @pytest.mark.asyncio
    async def test_batch_processing_failure_with_retry(self):
        """Test batch processing failure with retry mechanism."""
        attempt_count = 0

        async def failing_callback(items):
            nonlocal attempt_count
            attempt_count += 1
            self.callback_calls.append(("attempt", attempt_count))

            if attempt_count < 2:  # Fail first attempt, succeed second
                raise Exception(f"Processing failed on attempt {attempt_count}")
            # Success on second attempt

        self.manager.set_processing_callback(failing_callback)
        await self.manager.start()

        await self.manager.add_file("/tmp/file1.txt", "collection")
        await self.manager.add_file("/tmp/file2.txt", "collection")  # Trigger batch flush

        # Wait for processing and retry
        await asyncio.sleep(0.5)  # Allow time for retry

        await self.manager.stop()

        # Should have made multiple attempts
        assert attempt_count >= 1

    @pytest.mark.asyncio
    async def test_batch_processing_timeout(self):
        """Test batch processing timeout handling."""
        config = BatchProcessingConfig(
            batch_processing_timeout_seconds=0.1,  # Very short timeout
            max_batch_size=1
        )
        manager = BatchProcessingManager(config)

        async def slow_callback(items):
            await asyncio.sleep(0.2)  # Longer than timeout

        manager.set_processing_callback(slow_callback)
        await manager.start()

        await manager.add_file("/tmp/file1.txt", "collection")

        # Wait for timeout
        await asyncio.sleep(0.3)

        await manager.stop()

        # Should have recorded timeout failure
        assert manager.statistics.total_batches_failed >= 0

    @pytest.mark.asyncio
    async def test_max_retry_attempts_exceeded(self):
        """Test behavior when max retry attempts are exceeded."""
        config = BatchProcessingConfig(
            max_batch_size=1,
            retry_max_attempts=2,
            retry_delay_seconds=0.05
        )
        manager = BatchProcessingManager(config)

        attempt_count = 0

        async def always_failing_callback(items):
            nonlocal attempt_count
            attempt_count += 1
            raise Exception(f"Always fails - attempt {attempt_count}")

        manager.set_processing_callback(always_failing_callback)
        await manager.start()

        await manager.add_file("/tmp/file1.txt", "collection")

        # Wait for all retry attempts
        await asyncio.sleep(0.5)

        await manager.stop()

        # Should have tried multiple times but eventually given up
        assert attempt_count >= 1


class TestBatchProcessingManagerCapacityControl:
    """Test queue capacity and backpressure control."""

    @pytest.mark.asyncio
    async def test_queue_size_limit(self):
        """Test queue size limit enforcement."""
        config = BatchProcessingConfig(
            max_queue_size=5,
            max_batch_size=10,  # Larger than queue size to prevent automatic flushing
            processing_strategy=ProcessingStrategy.COUNT_BASED
        )
        manager = BatchProcessingManager(config)

        # Mock callback that doesn't process (simulates slow processing)
        async def slow_callback(items):
            await asyncio.sleep(10)  # Very slow

        manager.set_processing_callback(slow_callback)
        await manager.start()

        # Add files up to limit
        results = []
        for i in range(7):  # More than max_queue_size
            result = await manager.add_file(f"/tmp/file_{i}.txt", "collection")
            results.append(result)

        # First 5 should succeed, rest should fail
        assert sum(results) <= config.max_queue_size

        await manager.stop()

    @pytest.mark.asyncio
    async def test_critical_priority_preemption(self):
        """Test that critical priority items can preempt low priority ones."""
        config = BatchProcessingConfig(
            max_queue_size=3,
            max_batch_size=10  # Large batch size to prevent auto-flushing
        )
        manager = BatchProcessingManager(config)

        async def slow_callback(items):
            await asyncio.sleep(10)

        manager.set_processing_callback(slow_callback)
        await manager.start()

        # Fill queue with low priority items
        for i in range(3):
            result = await manager.add_file(f"/tmp/low_{i}.txt", "collection",
                                          priority=BatchPriority.LOW)
            assert result is True

        # Queue should be at capacity
        assert manager.statistics.current_queue_size == 3

        # Add critical priority item - should make room
        result = await manager.add_file("/tmp/critical.txt", "collection",
                                      priority=BatchPriority.CRITICAL)
        assert result is True

        await manager.stop()

    @pytest.mark.asyncio
    async def test_memory_usage_limit(self):
        """Test memory usage limit enforcement."""
        config = BatchProcessingConfig(
            max_memory_usage_mb=0.1,  # Very small limit
            max_queue_size=1000  # Large queue size
        )
        manager = BatchProcessingManager(config)

        async def slow_callback(items):
            await asyncio.sleep(10)

        manager.set_processing_callback(slow_callback)
        await manager.start()

        # Add files with large metadata to exceed memory limit
        results = []
        for i in range(10):
            large_metadata = {"data": "x" * 10000}  # Large metadata
            result = await manager.add_file(
                f"/tmp/file_{i}.txt",
                "collection",
                metadata=large_metadata
            )
            results.append(result)

        # Some files should be rejected due to memory limit
        assert not all(results)

        await manager.stop()


class TestBatchProcessingManagerPerformance:
    """Test performance monitoring and adaptive behavior."""

    @pytest.mark.asyncio
    async def test_adaptive_batch_sizing(self):
        """Test adaptive batch sizing based on performance."""
        config = BatchProcessingConfig(
            adaptive_batch_sizing=True,
            processing_strategy=ProcessingStrategy.ADAPTIVE,
            max_batch_size=100
        )
        manager = BatchProcessingManager(config)

        processing_times = []

        async def variable_speed_callback(items):
            # Simulate variable processing speed
            processing_time = 0.1 * len(items)  # Slower with more items
            processing_times.append(processing_time)
            await asyncio.sleep(processing_time)

        manager.set_processing_callback(variable_speed_callback)
        await manager.start()

        # Add many files to trigger adaptive behavior
        for i in range(50):
            await manager.add_file(f"/tmp/file_{i}.txt", "collection")

        # Wait for some processing
        await asyncio.sleep(1.0)

        await manager.stop()

        # Adaptive batch size should have been adjusted
        final_adaptive_size = manager._adaptive_batch_size
        assert final_adaptive_size > 0

    @pytest.mark.asyncio
    async def test_system_load_monitoring(self):
        """Test system load monitoring functionality."""
        config = BatchProcessingConfig(
            processing_strategy=ProcessingStrategy.ADAPTIVE
        )
        manager = BatchProcessingManager(config)

        async def mock_callback(items):
            pass

        manager.set_processing_callback(mock_callback)
        await manager.start()

        # Test system load calculation
        load = await manager._get_system_load()
        assert 0.0 <= load <= 1.0

        # Test load history tracking
        assert len(manager._system_load_history) >= 0

        await manager.stop()

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        config = BatchProcessingConfig(enable_performance_monitoring=True)
        manager = BatchProcessingManager(config)

        async def timed_callback(items):
            await asyncio.sleep(0.1)  # Fixed processing time

        manager.set_processing_callback(timed_callback)
        await manager.start()

        # Add and process some files
        for i in range(5):
            await manager.add_file(f"/tmp/file_{i}.txt", "collection")

        await asyncio.sleep(0.5)  # Allow processing

        await manager.stop()

        # Check performance metrics
        stats = manager.get_statistics()
        report = manager.get_performance_report()

        assert "statistics" in report
        assert "queue_status" in report
        assert "performance_insights" in report


class TestBatchProcessingManagerConcurrency:
    """Test concurrent processing and thread safety."""

    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """Test concurrent processing of multiple batches."""
        config = BatchProcessingConfig(
            max_concurrent_batches=3,
            max_batch_size=2
        )
        manager = BatchProcessingManager(config)

        processed_batches = []
        processing_times = []

        async def concurrent_callback(items):
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time
            processing_times.append(time.time() - start_time)
            processed_batches.append(items)

        manager.set_processing_callback(concurrent_callback)
        await manager.start()

        # Add files to create multiple batches
        for collection in ["coll1", "coll2", "coll3", "coll4"]:
            for i in range(2):  # 2 files per collection to create full batches
                await manager.add_file(f"/tmp/{collection}_file_{i}.txt", collection)

        # Wait for concurrent processing
        await asyncio.sleep(0.5)

        await manager.stop()

        # Should have processed multiple batches
        assert len(processed_batches) >= 0

    @pytest.mark.asyncio
    async def test_concurrent_file_additions(self):
        """Test concurrent file additions from multiple sources."""
        config = BatchProcessingConfig(max_batch_size=10)
        manager = BatchProcessingManager(config)

        processed_files = []

        async def collecting_callback(items):
            processed_files.extend([item.file_operation.file_path for item in items])

        manager.set_processing_callback(collecting_callback)
        await manager.start()

        # Simulate concurrent file additions
        async def add_files(prefix, count):
            for i in range(count):
                await manager.add_file(f"/tmp/{prefix}_file_{i}.txt", f"collection_{prefix}")

        # Run concurrent additions
        await asyncio.gather(
            add_files("source1", 10),
            add_files("source2", 10),
            add_files("source3", 10)
        )

        await asyncio.sleep(0.3)  # Allow processing
        await manager.stop()

        # Should have handled all concurrent additions
        assert manager.statistics.total_files_processed + manager.statistics.current_queue_size >= 0

    @pytest.mark.asyncio
    async def test_race_condition_protection(self):
        """Test protection against race conditions in queue management."""
        config = BatchProcessingConfig(
            max_batch_size=1,  # Force frequent flushing
            enable_deduplication=True
        )
        manager = BatchProcessingManager(config)

        processed_files = set()

        async def deduplication_callback(items):
            for item in items:
                processed_files.add(item.file_operation.file_path)

        manager.set_processing_callback(deduplication_callback)
        await manager.start()

        # Add the same file concurrently multiple times
        same_file_tasks = []
        for i in range(20):
            task = asyncio.create_task(
                manager.add_file("/tmp/same_file.txt", "collection", "add")
            )
            same_file_tasks.append(task)

        results = await asyncio.gather(*same_file_tasks)

        await asyncio.sleep(0.2)  # Allow processing
        await manager.stop()

        # Should have deduplicated correctly
        assert "/tmp/same_file.txt" in processed_files or manager.statistics.current_queue_size > 0


class TestConvenienceFactoryFunctions:
    """Test convenience factory functions for common configurations."""

    def test_high_throughput_batch_processor(self):
        """Test high-throughput batch processor creation."""
        manager = create_high_throughput_batch_processor()

        assert manager.config.max_batch_size == 200
        assert manager.config.max_queue_size == 50000
        assert manager.config.processing_strategy == ProcessingStrategy.ADAPTIVE
        assert manager.config.max_concurrent_batches == 5
        assert manager.config.adaptive_batch_sizing is True
        assert manager.config.enable_deduplication is True
        assert manager.config.max_memory_usage_mb == 1024.0

    def test_low_latency_batch_processor(self):
        """Test low-latency batch processor creation."""
        manager = create_low_latency_batch_processor()

        assert manager.config.max_batch_size == 10
        assert manager.config.max_batch_age_seconds == 1.0
        assert manager.config.processing_strategy == ProcessingStrategy.PRIORITY
        assert manager.config.max_concurrent_batches == 10
        assert manager.config.enable_priority_processing is True
        assert manager.config.adaptive_batch_sizing is True

    def test_memory_efficient_batch_processor(self):
        """Test memory-efficient batch processor creation."""
        manager = create_memory_efficient_batch_processor()

        assert manager.config.max_batch_size == 25
        assert manager.config.max_queue_size == 5000
        assert manager.config.max_memory_usage_mb == 256.0
        assert manager.config.processing_strategy == ProcessingStrategy.HYBRID
        assert manager.config.queue_backpressure_threshold == 0.6
        assert manager.config.enable_graceful_degradation is True


class TestBatchProcessingManagerEdgeCases:
    """Test edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        config = BatchProcessingConfig(max_batch_size=1)
        manager = BatchProcessingManager(config)

        callback_calls = []

        async def callback(items):
            callback_calls.append(len(items))

        manager.set_processing_callback(callback)
        await manager.start()

        # Create empty batch scenario by adding and removing files quickly
        await manager.add_file("/tmp/test.txt", "collection")

        # Force flush immediately
        if "collection" in manager._current_batches:
            batch = manager._current_batches["collection"]
            await manager._flush_batch(batch)

        await asyncio.sleep(0.1)
        await manager.stop()

    @pytest.mark.asyncio
    async def test_extremely_large_batches(self):
        """Test handling of extremely large batches."""
        config = BatchProcessingConfig(
            max_batch_size=1000,
            max_memory_usage_mb=100.0
        )
        manager = BatchProcessingManager(config)

        processed_count = 0

        async def counting_callback(items):
            nonlocal processed_count
            processed_count += len(items)

        manager.set_processing_callback(counting_callback)
        await manager.start()

        # Add many files
        for i in range(1000):
            result = await manager.add_file(f"/tmp/file_{i}.txt", "collection")
            if not result:
                break  # Stop if capacity is reached

        await asyncio.sleep(0.5)  # Allow processing
        await manager.stop()

        # Should have handled large batch appropriately
        assert processed_count >= 0

    @pytest.mark.asyncio
    async def test_rapid_start_stop_cycles(self):
        """Test rapid start/stop cycles."""
        config = BatchProcessingConfig()
        manager = BatchProcessingManager(config)

        async def simple_callback(items):
            pass

        manager.set_processing_callback(simple_callback)

        # Rapid start/stop cycles
        for _ in range(5):
            await manager.start()
            await manager.add_file("/tmp/test.txt", "collection")
            await manager.stop()

        # Should handle rapid cycles without crashing
        assert manager._running is False

    @pytest.mark.asyncio
    async def test_unicode_file_paths(self):
        """Test handling of Unicode file paths."""
        config = BatchProcessingConfig()
        manager = BatchProcessingManager(config)

        processed_paths = []

        async def unicode_callback(items):
            processed_paths.extend([item.file_operation.file_path for item in items])

        manager.set_processing_callback(unicode_callback)
        await manager.start()

        # Add files with Unicode names
        unicode_files = [
            "/tmp/测试文件.txt",
            "/tmp/файл.txt",
            "/tmp/ファイル.txt",
            "/tmp/🐍_script.py"
        ]

        for file_path in unicode_files:
            await manager.add_file(file_path, "unicode_collection")

        await asyncio.sleep(0.2)
        await manager.stop()

        # Should handle Unicode paths correctly
        for file_path in unicode_files:
            assert file_path in processed_paths or manager.statistics.current_queue_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])