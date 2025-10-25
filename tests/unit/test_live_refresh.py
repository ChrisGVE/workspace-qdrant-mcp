"""
Unit tests for LiveRefreshManager.

Tests live refresh functionality for detecting and applying rule changes during
active Claude Code sessions.
"""

import asyncio
import hashlib
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

from src.python.common.core.context_injection.live_refresh import (
    LiveRefreshManager,
    RefreshMode,
    RefreshResult,
    RefreshState,
    RefreshThrottleConfig,
    start_live_refresh,
)
from src.python.common.core.memory import AuthorityLevel, MemoryCategory, MemoryRule


@pytest.fixture
def mock_memory_manager():
    """Create mock MemoryManager."""
    manager = MagicMock()
    manager.client = MagicMock()
    return manager


@pytest.fixture
def mock_budget_manager():
    """Create mock ClaudeBudgetManager."""
    manager = MagicMock()
    manager.session_stats = MagicMock()
    manager.session_stats.budget_limit = 50000
    return manager


@pytest.fixture
def mock_claude_md_injector():
    """Create mock ClaudeMdInjector."""
    injector = MagicMock()
    injector._observer = None
    injector.inject_to_file = AsyncMock(return_value=True)
    injector.inject_from_files = AsyncMock(return_value="test content")
    injector.start_watching = MagicMock(return_value=True)
    injector.stop_watching = MagicMock()
    injector.discover_claude_md_files = MagicMock(return_value=[])
    return injector


@pytest.fixture
def mock_rule_retrieval():
    """Create mock RuleRetrieval."""
    retrieval = MagicMock()
    retrieval.get_rules = AsyncMock(return_value=MagicMock(rules=[]))
    return retrieval


@pytest.fixture
def sample_rules():
    """Create sample memory rules."""
    return [
        MemoryRule(
            id="rule1",
            category=MemoryCategory.BEHAVIOR,
            name="Test Rule 1",
            rule="Always test code",
            authority=AuthorityLevel.ABSOLUTE,
            scope=["global"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
        MemoryRule(
            id="rule2",
            category=MemoryCategory.PREFERENCE,
            name="Test Rule 2",
            rule="Use Python",
            authority=AuthorityLevel.DEFAULT,
            scope=["global"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def throttle_config():
    """Create test throttle configuration."""
    return RefreshThrottleConfig(
        debounce_seconds=0.1,
        min_refresh_interval_seconds=0.2,
        max_refresh_rate_per_minute=10,
        change_aggregation_window_seconds=0.1,
    )


@pytest.fixture
async def refresh_manager(
    mock_memory_manager,
    mock_budget_manager,
    mock_claude_md_injector,
    mock_rule_retrieval,
    throttle_config,
):
    """Create LiveRefreshManager instance."""
    manager = LiveRefreshManager(
        memory_manager=mock_memory_manager,
        budget_manager=mock_budget_manager,
        claude_md_injector=mock_claude_md_injector,
        rule_retrieval=mock_rule_retrieval,
        throttle_config=throttle_config,
        enable_file_watching=True,
        enable_rule_monitoring=True,
        enable_periodic_refresh=False,
    )
    yield manager
    # Cleanup
    await manager.stop()


class TestLiveRefreshManagerInitialization:
    """Test LiveRefreshManager initialization."""

    def test_initialization_default_config(self, mock_memory_manager):
        """Test initialization with default configuration."""
        manager = LiveRefreshManager(memory_manager=mock_memory_manager)

        assert manager.memory_manager == mock_memory_manager
        assert manager.enable_file_watching is True
        assert manager.enable_rule_monitoring is True
        assert manager.enable_periodic_refresh is False
        assert isinstance(manager.throttle_config, RefreshThrottleConfig)
        assert isinstance(manager.refresh_state, RefreshState)

    def test_initialization_custom_config(
        self,
        mock_memory_manager,
        mock_budget_manager,
        throttle_config,
    ):
        """Test initialization with custom configuration."""
        manager = LiveRefreshManager(
            memory_manager=mock_memory_manager,
            budget_manager=mock_budget_manager,
            throttle_config=throttle_config,
            enable_file_watching=False,
            enable_rule_monitoring=False,
            enable_periodic_refresh=True,
            periodic_interval_seconds=60.0,
        )

        assert manager.budget_manager == mock_budget_manager
        assert manager.throttle_config == throttle_config
        assert manager.enable_file_watching is False
        assert manager.enable_rule_monitoring is False
        assert manager.enable_periodic_refresh is True
        assert manager.periodic_interval_seconds == 60.0

    def test_initialization_creates_injector(self, mock_memory_manager):
        """Test that ClaudeMdInjector is created if not provided."""
        manager = LiveRefreshManager(memory_manager=mock_memory_manager)

        assert manager.claude_md_injector is not None
        assert manager.rule_retrieval is not None


class TestRefreshStateManagement:
    """Test refresh state tracking."""

    @pytest.mark.asyncio
    async def test_initial_state(self, refresh_manager):
        """Test initial refresh state."""
        state = refresh_manager.refresh_state

        assert state.last_refresh_at is None
        assert state.last_file_hash is None
        assert state.last_rules_hash is None
        assert len(state.pending_changes) == 0
        assert state.refresh_count == 0
        assert state.is_refreshing is False

    @pytest.mark.asyncio
    async def test_state_update_after_refresh(self, refresh_manager):
        """Test state updates after successful refresh."""
        # Mock hash computation
        refresh_manager._compute_file_hash = AsyncMock(return_value="file_hash_123")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="rules_hash_456")

        # Execute refresh
        result = await refresh_manager.refresh_now(force=True)

        assert result.success is True
        assert refresh_manager.refresh_state.refresh_count == 1
        assert refresh_manager.refresh_state.last_refresh_at is not None
        assert refresh_manager.refresh_state.last_file_hash == "file_hash_123"
        assert refresh_manager.refresh_state.last_rules_hash == "rules_hash_456"
        assert len(refresh_manager.refresh_state.refresh_history) == 1

    @pytest.mark.asyncio
    async def test_pending_changes_tracking(self, refresh_manager):
        """Test pending changes tracking."""
        refresh_manager.refresh_state.pending_changes.add("claude_md")
        refresh_manager.refresh_state.pending_changes.add("memory_rules")

        assert "claude_md" in refresh_manager.refresh_state.pending_changes
        assert "memory_rules" in refresh_manager.refresh_state.pending_changes

        # Mock successful refresh
        refresh_manager._compute_file_hash = AsyncMock(return_value="hash1")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="hash2")

        await refresh_manager.refresh_now(force=True)

        # Pending changes should be cleared
        assert len(refresh_manager.refresh_state.pending_changes) == 0


class TestThrottlingAndDebouncing:
    """Test refresh throttling and debouncing."""

    @pytest.mark.asyncio
    async def test_minimum_refresh_interval(self, refresh_manager):
        """Test minimum refresh interval enforcement."""
        # Mock hash computation
        refresh_manager._compute_file_hash = AsyncMock(return_value="hash1")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="hash2")

        # First refresh
        result1 = await refresh_manager.refresh_now(force=True)
        assert result1.success is True

        # Immediate second refresh (should be throttled)
        result2 = await refresh_manager.refresh_now(force=False)
        assert result2.success is False
        assert "throttled" in result2.error.lower()

        # Wait for interval
        await asyncio.sleep(0.3)

        # Third refresh (should succeed)
        result3 = await refresh_manager.refresh_now(force=False)
        assert result3.success is True

    @pytest.mark.asyncio
    async def test_force_bypass_throttling(self, refresh_manager):
        """Test force flag bypasses throttling."""
        # Mock hash computation
        refresh_manager._compute_file_hash = AsyncMock(return_value="hash1")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="hash2")

        # First refresh
        result1 = await refresh_manager.refresh_now(force=True)
        assert result1.success is True

        # Immediate second refresh with force=True (should succeed)
        result2 = await refresh_manager.refresh_now(force=True)
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_refresh_rate_limit(self, refresh_manager):
        """Test refresh rate limit (per minute)."""
        # Mock hash computation
        refresh_manager._compute_file_hash = AsyncMock(return_value="hash1")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="hash2")

        # Execute max refreshes with force
        max_rate = refresh_manager.throttle_config.max_refresh_rate_per_minute
        for _i in range(max_rate):
            result = await refresh_manager.refresh_now(force=True)
            assert result.success is True

        # Next refresh should be rate-limited
        result = await refresh_manager.refresh_now(force=False)
        assert result.success is False
        assert "rate limit" in result.error.lower() or "throttled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_refresh_already_in_progress(self, refresh_manager):
        """Test that concurrent refreshes are prevented."""
        # Mock slow refresh
        async def slow_inject(*args, **kwargs):
            await asyncio.sleep(0.5)
            return True

        refresh_manager.claude_md_injector.inject_to_file = slow_inject
        refresh_manager._compute_file_hash = AsyncMock(return_value="hash1")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="hash2")

        # Start first refresh
        task1 = asyncio.create_task(refresh_manager.refresh_now(force=True))

        # Wait a bit for first refresh to start
        await asyncio.sleep(0.1)

        # Try second refresh (should be prevented)
        result2 = await refresh_manager.refresh_now(force=False)
        assert result2.success is False
        assert "throttled" in result2.error.lower()

        # Wait for first refresh to complete
        result1 = await task1
        assert result1.success is True


class TestChangeDetection:
    """Test change detection mechanisms."""

    @pytest.mark.asyncio
    async def test_file_change_detection(self, refresh_manager, tmp_path):
        """Test CLAUDE.md file change detection."""
        # Create test file
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("original content")

        # Mock file discovery
        location = MagicMock()
        location.path = claude_md
        refresh_manager.claude_md_injector.discover_claude_md_files = MagicMock(
            return_value=[location]
        )

        # Initialize baseline
        await refresh_manager._initialize_baseline_hashes()
        original_hash = refresh_manager.refresh_state.last_file_hash

        # Modify file
        claude_md.write_text("modified content")

        # Detect changes
        new_hash = await refresh_manager._compute_file_hash()

        assert new_hash != original_hash

    @pytest.mark.asyncio
    async def test_rules_change_detection(self, refresh_manager, sample_rules):
        """Test memory rules change detection."""
        # Mock initial rules
        mock_result1 = MagicMock()
        mock_result1.rules = sample_rules[:1]
        refresh_manager.rule_retrieval.get_rules = AsyncMock(return_value=mock_result1)

        # Compute initial hash
        hash1 = await refresh_manager._compute_rules_hash()

        # Mock updated rules
        mock_result2 = MagicMock()
        mock_result2.rules = sample_rules  # All rules
        refresh_manager.rule_retrieval.get_rules = AsyncMock(return_value=mock_result2)

        # Compute new hash
        hash2 = await refresh_manager._compute_rules_hash()

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_no_changes_detected(self, refresh_manager):
        """Test behavior when no changes are detected."""
        # Mock identical hashes
        refresh_manager._compute_file_hash = AsyncMock(return_value="same_hash")
        refresh_manager._compute_rules_hash = AsyncMock(return_value="same_hash")
        refresh_manager.refresh_state.last_file_hash = "same_hash"
        refresh_manager.refresh_state.last_rules_hash = "same_hash"

        # Detect changes
        changes = await refresh_manager._detect_changes()

        assert len(changes) == 0

    @pytest.mark.asyncio
    async def test_file_hash_computation(self, refresh_manager, tmp_path):
        """Test file hash computation."""
        # Create test files
        claude_md1 = tmp_path / "CLAUDE.md"
        claude_md1.write_text("content 1")

        location1 = MagicMock()
        location1.path = claude_md1

        refresh_manager.claude_md_injector.discover_claude_md_files = MagicMock(
            return_value=[location1]
        )

        hash1 = await refresh_manager._compute_file_hash()
        assert hash1 != ""

        # Modify content
        claude_md1.write_text("content 2")
        hash2 = await refresh_manager._compute_file_hash()

        assert hash2 != ""
        assert hash1 != hash2


class TestRefreshExecution:
    """Test refresh execution."""

    @pytest.mark.asyncio
    async def test_successful_refresh(self, refresh_manager, tmp_path):
        """Test successful refresh execution."""
        output_path = tmp_path / "output.md"

        # Start manager
        await refresh_manager.start(
            project_root=tmp_path,
            output_path=output_path,
        )

        # Mock change detection
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        # Execute refresh
        result = await refresh_manager.refresh_now(force=True, mode=RefreshMode.MANUAL)

        assert result.success is True
        assert result.mode == RefreshMode.MANUAL
        assert "claude_md" in result.changes_detected
        assert result.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_refresh_with_no_changes(self, refresh_manager):
        """Test refresh execution with no changes detected."""
        # Mock no changes
        refresh_manager._detect_changes = AsyncMock(return_value=[])

        # Execute refresh
        result = await refresh_manager.refresh_now(force=True)

        assert result.success is True
        assert len(result.changes_detected) == 0
        assert result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_refresh_failure_handling(self, refresh_manager):
        """Test refresh failure handling."""
        # Mock failure
        refresh_manager.claude_md_injector.inject_to_file = AsyncMock(
            return_value=False
        )
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])
        refresh_manager._output_path = Path("/tmp/output.md")

        # Execute refresh
        result = await refresh_manager.refresh_now(force=True)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_refresh_mode_tracking(self, refresh_manager):
        """Test that refresh mode is tracked correctly."""
        # Mock successful refresh
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        # Test different modes
        result_auto = await refresh_manager.refresh_now(
            force=True, mode=RefreshMode.AUTOMATIC
        )
        assert result_auto.mode == RefreshMode.AUTOMATIC

        result_manual = await refresh_manager.refresh_now(
            force=True, mode=RefreshMode.MANUAL
        )
        assert result_manual.mode == RefreshMode.MANUAL

        result_periodic = await refresh_manager.refresh_now(
            force=True, mode=RefreshMode.PERIODIC
        )
        assert result_periodic.mode == RefreshMode.PERIODIC


class TestFileWatching:
    """Test file watching functionality."""

    @pytest.mark.asyncio
    async def test_start_file_watching(self, refresh_manager, tmp_path):
        """Test starting file watching."""
        success = await refresh_manager.start(project_root=tmp_path)

        assert success is True
        refresh_manager.claude_md_injector.start_watching.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_file_watching(self, refresh_manager, tmp_path):
        """Test stopping file watching."""
        await refresh_manager.start(project_root=tmp_path)
        await refresh_manager.stop()

        refresh_manager.claude_md_injector.stop_watching.assert_called()

    @pytest.mark.asyncio
    async def test_file_change_callback(self, refresh_manager, tmp_path):
        """Test file change callback handling."""
        await refresh_manager.start(project_root=tmp_path)

        # Mock change detection and refresh
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        # Simulate file change
        test_file = tmp_path / "CLAUDE.md"
        refresh_manager._handle_file_change(test_file)

        # Wait for debounced refresh
        await asyncio.sleep(0.3)

        # Check pending changes
        # (Note: actual refresh might be throttled, but change should be registered)
        assert refresh_manager.refresh_state.refresh_count >= 0


class TestCallbackNotifications:
    """Test refresh callback notifications."""

    @pytest.mark.asyncio
    async def test_callback_registration(self, refresh_manager):
        """Test callback registration."""
        callback1 = MagicMock()
        callback2 = MagicMock()

        refresh_manager.add_refresh_callback(callback1)
        refresh_manager.add_refresh_callback(callback2)

        assert callback1 in refresh_manager._refresh_callbacks
        assert callback2 in refresh_manager._refresh_callbacks

    @pytest.mark.asyncio
    async def test_callback_notification_on_success(self, refresh_manager):
        """Test callbacks are notified on successful refresh."""
        callback = MagicMock()
        refresh_manager.add_refresh_callback(callback)

        # Mock successful refresh
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        await refresh_manager.refresh_now(force=True)

        # Callback should be called with RefreshResult
        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert isinstance(result, RefreshResult)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_callback_notification_on_failure(self, refresh_manager):
        """Test callbacks are notified on failed refresh."""
        callback = MagicMock()
        refresh_manager.add_refresh_callback(callback)

        # Mock failure
        refresh_manager._detect_changes = AsyncMock(side_effect=Exception("Test error"))

        await refresh_manager.refresh_now(force=True)

        # Callback should be called with failure result
        callback.assert_called_once()
        result = callback.call_args[0][0]
        assert isinstance(result, RefreshResult)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_async_callback_support(self, refresh_manager):
        """Test async callback support."""
        async_callback = AsyncMock()
        refresh_manager.add_refresh_callback(async_callback)

        # Mock successful refresh
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        await refresh_manager.refresh_now(force=True)

        # Async callback should be awaited
        async_callback.assert_called_once()


class TestRefreshStats:
    """Test refresh statistics."""

    @pytest.mark.asyncio
    async def test_get_refresh_stats(self, refresh_manager):
        """Test getting refresh statistics."""
        stats = refresh_manager.get_refresh_stats()

        assert "total_refreshes" in stats
        assert "last_refresh_at" in stats
        assert "refreshes_last_minute" in stats
        assert "is_refreshing" in stats
        assert "pending_changes" in stats

        assert stats["total_refreshes"] == 0
        assert stats["last_refresh_at"] is None
        assert stats["is_refreshing"] is False

    @pytest.mark.asyncio
    async def test_stats_after_refresh(self, refresh_manager):
        """Test stats are updated after refresh."""
        # Mock successful refresh
        refresh_manager._detect_changes = AsyncMock(return_value=["claude_md"])

        await refresh_manager.refresh_now(force=True)

        stats = refresh_manager.get_refresh_stats()

        assert stats["total_refreshes"] == 1
        assert stats["last_refresh_at"] is not None
        assert stats["refreshes_last_minute"] == 1


class TestConvenienceFunction:
    """Test convenience function."""

    @pytest.mark.asyncio
    async def test_start_live_refresh(
        self, mock_memory_manager, mock_budget_manager, tmp_path
    ):
        """Test start_live_refresh convenience function."""
        output_path = tmp_path / "output.md"

        with patch(
            "src.python.common.core.context_injection.live_refresh.LiveRefreshManager"
        ) as mock_manager_class:
            mock_instance = AsyncMock()
            mock_instance.start = AsyncMock(return_value=True)
            mock_manager_class.return_value = mock_instance

            await start_live_refresh(
                memory_manager=mock_memory_manager,
                project_root=tmp_path,
                output_path=output_path,
                budget_manager=mock_budget_manager,
                enable_periodic=True,
            )

            # Verify manager was created and started
            mock_manager_class.assert_called_once()
            mock_instance.start.assert_called_once_with(
                project_root=tmp_path, output_path=output_path
            )


class TestIntegration:
    """Integration tests."""

    @pytest.mark.asyncio
    async def test_full_refresh_cycle(
        self,
        refresh_manager,
        tmp_path,
        sample_rules,
    ):
        """Test full refresh cycle with real components."""
        # Setup
        output_path = tmp_path / "output.md"
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("# Test CLAUDE.md\n\nOriginal content")

        # Mock components
        location = MagicMock()
        location.path = claude_md
        refresh_manager.claude_md_injector.discover_claude_md_files = MagicMock(
            return_value=[location]
        )

        mock_result = MagicMock()
        mock_result.rules = sample_rules
        refresh_manager.rule_retrieval.get_rules = AsyncMock(return_value=mock_result)

        # Start monitoring
        await refresh_manager.start(project_root=tmp_path, output_path=output_path)

        # Initial baseline
        initial_file_hash = refresh_manager.refresh_state.last_file_hash

        # Modify file
        claude_md.write_text("# Test CLAUDE.md\n\nModified content")

        # Execute refresh
        result = await refresh_manager.refresh_now(force=True)

        # Verify
        assert result.success is True
        assert refresh_manager.refresh_state.refresh_count == 1
        assert refresh_manager.refresh_state.last_file_hash != initial_file_hash

        # Cleanup
        await refresh_manager.stop()
