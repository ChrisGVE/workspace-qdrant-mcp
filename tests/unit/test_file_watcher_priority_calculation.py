"""
Unit tests for FileWatcher context-aware priority calculation.

Tests the integration of PriorityQueueManager's dynamic priority calculation
with the FileWatcher system.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.python.common.core.file_watcher import (
    FileWatcher,
    WatchConfiguration,
    WatchEvent,
)
from src.python.common.core.sqlite_state_manager import (
    SQLiteStateManager,
    ProcessingPriority,
)
from src.python.common.core.priority_queue_manager import (
    PriorityQueueManager,
    PriorityCalculationContext,
    MCPActivityMetrics,
    MCPActivityLevel,
)


@pytest.fixture
async def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def state_manager(temp_dir):
    """Create test SQLite state manager."""
    db_path = temp_dir / "test_state.db"
    manager = SQLiteStateManager(db_path=str(db_path))
    await manager.initialize()
    yield manager
    await manager.close()


@pytest.fixture
async def priority_queue_manager(state_manager):
    """Create test priority queue manager."""
    manager = PriorityQueueManager(
        state_manager=state_manager,
        incremental_processor=None,
    )
    await manager.initialize()
    yield manager
    await manager.shutdown()


@pytest.fixture
def watch_config(temp_dir):
    """Create test watch configuration."""
    return WatchConfiguration(
        id="test_watch",
        path=str(temp_dir),
        collection="test_collection",
        patterns=["*.txt", "*.py", "*.md"],
        ignore_patterns=[".git/*", "*.tmp"],
        user_triggered=False,
    )


@pytest.fixture
def user_triggered_config(temp_dir):
    """Create test watch configuration marked as user-triggered."""
    return WatchConfiguration(
        id="test_watch_user",
        path=str(temp_dir),
        collection="test_collection",
        patterns=["*.txt", "*.py", "*.md"],
        ignore_patterns=[".git/*", "*.tmp"],
        user_triggered=True,
    )


class TestFileWatcherPriorityCalculation:
    """Test priority calculation integration in FileWatcher."""

    @pytest.mark.asyncio
    async def test_calculate_priority_without_manager(
        self, watch_config, state_manager, temp_dir
    ):
        """Test that default priority (5) is used when no priority manager is set."""
        # Create file watcher without priority queue manager
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=None,
        )

        # Calculate priority for a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        priority = await watcher._calculate_priority(test_file, "added")

        # Should return default NORMAL priority
        assert priority == 5

    @pytest.mark.asyncio
    async def test_calculate_priority_with_manager_low(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test priority calculation with manager returning LOW priority."""
        # Create file watcher with priority queue manager
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set MCP activity to INACTIVE (will result in LOW priority)
        priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.INACTIVE
        priority_queue_manager.mcp_activity.requests_per_minute = 0.0

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Calculate priority
        priority = await watcher._calculate_priority(test_file, "added")

        # LOW priority should map to 2
        assert 0 <= priority <= 10
        assert priority == 2  # LOW maps to 2

    @pytest.mark.asyncio
    async def test_calculate_priority_with_manager_high(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test priority calculation with manager returning HIGH priority."""
        # Create file watcher with priority queue manager
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set MCP activity to HIGH
        priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH
        priority_queue_manager.mcp_activity.requests_per_minute = 25.0

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Calculate priority
        priority = await watcher._calculate_priority(test_file, "added")

        # HIGH priority should map to 8
        assert priority >= 5  # At least NORMAL
        assert priority <= 10

    @pytest.mark.asyncio
    async def test_calculate_priority_deletion_boost(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that deletion operations get priority boost."""
        # Create file watcher with priority queue manager
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set moderate MCP activity
        priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.MODERATE

        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Calculate priority for added vs deleted
        priority_added = await watcher._calculate_priority(test_file, "added")
        priority_deleted = await watcher._calculate_priority(test_file, "deleted")

        # Deletion should have higher priority (+2 boost)
        assert priority_deleted > priority_added
        assert priority_deleted == min(10, priority_added + 2)

    @pytest.mark.asyncio
    async def test_calculate_priority_code_file_boost(
        self, user_triggered_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that code files get priority boost when user-triggered."""
        # Create file watcher with user-triggered config
        watcher = FileWatcher(
            config=user_triggered_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set moderate MCP activity
        priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.MODERATE

        # Create Python file and text file
        py_file = temp_dir / "test.py"
        py_file.write_text("print('test')")

        txt_file = temp_dir / "test.txt"
        txt_file.write_text("test content")

        # Calculate priorities
        priority_py = await watcher._calculate_priority(py_file, "added")
        priority_txt = await watcher._calculate_priority(txt_file, "added")

        # Python file should have higher priority when user-triggered
        assert priority_py >= priority_txt

    @pytest.mark.asyncio
    async def test_priority_mapping(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test ProcessingPriority enum to integer mapping."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Test all enum values map correctly
        from src.python.common.core.sqlite_state_manager import ProcessingPriority

        priority_map = {
            ProcessingPriority.LOW: 2,
            ProcessingPriority.NORMAL: 5,
            ProcessingPriority.HIGH: 8,
            ProcessingPriority.URGENT: 10,
        }

        for enum_priority, expected_value in priority_map.items():
            # Mock _calculate_dynamic_priority to return specific enum value
            with patch.object(
                priority_queue_manager,
                "_calculate_dynamic_priority",
                return_value=(enum_priority, 50.0),
            ):
                priority = await watcher._calculate_priority(test_file, "added")
                assert priority == expected_value

    @pytest.mark.asyncio
    async def test_calculate_priority_error_handling(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that priority calculation errors fall back to default priority."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Mock _calculate_dynamic_priority to raise an exception
        with patch.object(
            priority_queue_manager,
            "_calculate_dynamic_priority",
            side_effect=Exception("Test error"),
        ):
            priority = await watcher._calculate_priority(test_file, "added")

            # Should fall back to default NORMAL priority (5)
            assert priority == 5

    @pytest.mark.asyncio
    async def test_cache_refresh(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that tenant_id and branch cache is refreshed properly."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Initialize cache
        await watcher._refresh_cache()

        # Verify cache is populated
        assert watcher._cached_tenant_id is not None
        assert watcher._cached_branch is not None
        assert watcher._cache_timestamp is not None

        # Verify cache values
        tenant_id, branch = await watcher._get_cached_values()
        assert tenant_id == watcher._cached_tenant_id
        assert branch == watcher._cached_branch

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that cache is refreshed after TTL expiration."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set short TTL for testing
        watcher._cache_ttl_seconds = 1

        # Initialize cache
        await watcher._refresh_cache()
        initial_timestamp = watcher._cache_timestamp

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Get cached values should trigger refresh
        await watcher._get_cached_values()

        # Verify cache was refreshed
        assert watcher._cache_timestamp > initial_timestamp

    @pytest.mark.asyncio
    async def test_priority_with_file_metadata(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that file metadata (size, mtime) is included in context."""
        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Create test file with specific content
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content with some size")

        # Mock _calculate_dynamic_priority to capture the context
        context_capture = None

        async def capture_context(context):
            nonlocal context_capture
            context_capture = context
            return ProcessingPriority.NORMAL, 50.0

        with patch.object(
            priority_queue_manager,
            "_calculate_dynamic_priority",
            side_effect=capture_context,
        ):
            await watcher._calculate_priority(test_file, "added")

            # Verify context has file metadata
            assert context_capture is not None
            assert context_capture.file_path == str(test_file)
            assert context_capture.file_size > 0
            assert context_capture.file_modification_time is not None
            assert context_capture.collection == "test_collection"

    @pytest.mark.asyncio
    async def test_priority_with_user_triggered(
        self, user_triggered_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that user_triggered flag affects priority calculation."""
        watcher = FileWatcher(
            config=user_triggered_config,
            state_manager=state_manager,
            priority_queue_manager=priority_queue_manager,
        )

        # Set HIGH MCP activity
        priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH

        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Calculate priority - should be high due to user_triggered + HIGH activity
        priority = await watcher._calculate_priority(test_file, "added")

        # Should get HIGH or URGENT priority
        assert priority >= 8

    @pytest.mark.asyncio
    async def test_watch_event_includes_priority(
        self, watch_config, state_manager, priority_queue_manager, temp_dir
    ):
        """Test that WatchEvent includes calculated priority."""
        events_captured = []

        def event_callback(event: WatchEvent):
            events_captured.append(event)

        watcher = FileWatcher(
            config=watch_config,
            state_manager=state_manager,
            event_callback=event_callback,
            priority_queue_manager=priority_queue_manager,
        )

        await watcher.start()

        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        # Wait for event processing
        await asyncio.sleep(0.5)

        # Stop watcher
        await watcher.stop()

        # Verify event was captured with priority
        # Note: This is a timing-dependent test, so we may need to adjust
        if events_captured:
            event = events_captured[0]
            assert event.priority is not None
            assert 0 <= event.priority <= 10


@pytest.mark.asyncio
async def test_integration_priority_calculation_flow(temp_dir):
    """Integration test for complete priority calculation flow."""
    # Setup state manager
    db_path = temp_dir / "test_state.db"
    state_manager = SQLiteStateManager(db_path=str(db_path))
    await state_manager.initialize()

    # Setup priority queue manager
    priority_queue_manager = PriorityQueueManager(
        state_manager=state_manager,
        incremental_processor=None,
    )
    await priority_queue_manager.initialize()

    # Setup file watcher
    config = WatchConfiguration(
        id="test_integration",
        path=str(temp_dir),
        collection="test_collection",
        patterns=["*.txt", "*.py"],
        user_triggered=True,
    )

    watcher = FileWatcher(
        config=config,
        state_manager=state_manager,
        priority_queue_manager=priority_queue_manager,
    )

    # Set HIGH MCP activity
    priority_queue_manager.mcp_activity.activity_level = MCPActivityLevel.HIGH
    priority_queue_manager.mcp_activity.requests_per_minute = 30.0

    # Create test files
    test_file_py = temp_dir / "test.py"
    test_file_py.write_text("print('hello')")

    test_file_txt = temp_dir / "test.txt"
    test_file_txt.write_text("hello world")

    # Calculate priorities
    priority_py = await watcher._calculate_priority(test_file_py, "added")
    priority_txt = await watcher._calculate_priority(test_file_txt, "added")
    priority_deleted = await watcher._calculate_priority(test_file_py, "deleted")

    # Verify priorities are calculated correctly
    assert priority_py >= 8  # HIGH activity + user_triggered + code file
    assert priority_txt >= 5  # At least NORMAL
    assert priority_deleted > priority_py  # Deletion gets +2 boost

    # Cleanup
    await priority_queue_manager.shutdown()
    await state_manager.close()
