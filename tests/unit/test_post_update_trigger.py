"""
Unit tests for post-update trigger system (Task 301.4).

Tests the post-update trigger functionality including:
- PostUpdateTrigger execution with debouncing
- Change batch processing
- Integration with ON_RULE_UPDATE phase
- File change monitoring integration
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.python.common.core.context_injection.session_trigger import (
    PostUpdateTrigger,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerContext,
)
from src.python.common.memory.types import MemoryRule, MemoryCategory, AuthorityLevel


@pytest.fixture
def mock_memory_manager():
    """Provide mock MemoryManager."""
    manager = AsyncMock()
    manager.get_rules = AsyncMock(return_value=[])
    manager.add_rule = AsyncMock()
    return manager


@pytest.fixture
def mock_claude_code_session():
    """Provide mock ClaudeCodeSession."""
    from src.python.common.core.context_injection.claude_code_detector import (
        ClaudeCodeSession,
    )

    return ClaudeCodeSession(
        is_active=True,
        entrypoint="cli",
        detection_method="test",
        session_id="test_session",
    )


@pytest.fixture
def trigger_context(mock_memory_manager, mock_claude_code_session, tmp_path):
    """Provide TriggerContext for testing."""
    return TriggerContext(
        session=mock_claude_code_session,
        project_root=tmp_path,
        memory_manager=mock_memory_manager,
    )


@pytest.mark.asyncio
class TestPostUpdateTrigger:
    """Test PostUpdateTrigger class."""

    async def test_initialization(self, tmp_path):
        """Test trigger initialization."""
        trigger = PostUpdateTrigger(
            debounce_seconds=2.0,
            batch_window_seconds=5.0,
            output_path=tmp_path / "output.md",
            token_budget=10000,
        )

        assert trigger.name == "post_update_refresh"
        assert trigger.phase == TriggerPhase.ON_RULE_UPDATE
        assert trigger.priority == TriggerPriority.HIGH
        assert trigger.debounce_seconds == 2.0
        assert trigger.batch_window_seconds == 5.0
        assert trigger.output_path == tmp_path / "output.md"
        assert trigger.token_budget == 10000
        assert trigger._last_trigger_time is None
        assert trigger._pending_changes == []

    async def test_execute_with_refresh(self, trigger_context, tmp_path):
        """Test execution triggers refresh."""
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            result = await trigger.execute(trigger_context)

            assert result.success
            assert result.phase == TriggerPhase.ON_RULE_UPDATE
            assert result.trigger_name == "post_update_refresh"
            assert result.execution_time_ms > 0
            assert trigger._last_trigger_time is not None

    async def test_debouncing_prevents_duplicate_triggers(
        self, trigger_context, tmp_path
    ):
        """Test that debouncing prevents rapid re-triggers."""
        trigger = PostUpdateTrigger(
            debounce_seconds=2.0,
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # First execution
            result1 = await trigger.execute(trigger_context)
            assert result1.success
            first_trigger_time = trigger._last_trigger_time

            # Immediate second execution (should be debounced)
            result2 = await trigger.execute(trigger_context)
            assert result2.success
            assert result2.metadata.get("debounced") is True
            assert trigger._last_trigger_time == first_trigger_time

    async def test_trigger_after_debounce_delay(self, trigger_context, tmp_path):
        """Test that trigger executes after debounce window expires."""
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # First execution
            result1 = await trigger.execute(trigger_context)
            assert result1.success
            first_time = trigger._last_trigger_time

            # Wait for debounce window to expire
            await asyncio.sleep(1.1)

            # Second execution (should succeed)
            result2 = await trigger.execute(trigger_context)
            assert result2.success
            assert result2.metadata.get("debounced") is not True
            assert trigger._last_trigger_time > first_time

    async def test_record_change(self, tmp_path):
        """Test recording changes for batch processing."""
        trigger = PostUpdateTrigger(
            output_path=tmp_path / "context.md",
        )

        # Record multiple changes
        trigger.record_change("Rule added: use type hints")
        trigger.record_change("Rule updated: use docstrings")
        trigger.record_change("Configuration changed: token budget")

        # Verify changes recorded
        changes = trigger.get_pending_changes()
        assert len(changes) == 3
        assert "Rule added: use type hints" in changes
        assert "Rule updated: use docstrings" in changes
        assert "Configuration changed: token budget" in changes

    async def test_pending_changes_included_in_metadata(
        self, trigger_context, tmp_path
    ):
        """Test that pending changes are included in execution metadata."""
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Record some changes
        trigger.record_change("Rule 1 added")
        trigger.record_change("Rule 2 updated")

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            result = await trigger.execute(trigger_context)

            assert result.success
            assert "changes_processed" in result.metadata
            changes_count = result.metadata["changes_processed"]
            assert changes_count == 2

    async def test_pending_changes_cleared_after_execution(
        self, trigger_context, tmp_path
    ):
        """Test that pending changes are cleared after successful execution."""
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Record changes
        trigger.record_change("Change 1")
        trigger.record_change("Change 2")
        assert len(trigger.get_pending_changes()) == 2

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            await trigger.execute(trigger_context)

            # Changes should be cleared
            assert len(trigger.get_pending_changes()) == 0

    async def test_execution_failure(self, trigger_context, tmp_path):
        """Test handling of execution failures."""
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector to fail
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=False)
            mock_injector_class.return_value = mock_injector

            result = await trigger.execute(trigger_context)

            assert not result.success
            assert result.error is not None
            assert "Failed to refresh after update" in result.error

    async def test_batch_window_accumulation(self, tmp_path):
        """Test that changes accumulate within batch window."""
        trigger = PostUpdateTrigger(
            batch_window_seconds=2.0,
            output_path=tmp_path / "context.md",
        )

        # Record changes over time
        trigger.record_change("Change 1")
        await asyncio.sleep(0.5)
        trigger.record_change("Change 2")
        await asyncio.sleep(0.5)
        trigger.record_change("Change 3")

        # All changes should be present
        changes = trigger.get_pending_changes()
        assert len(changes) == 3

    async def test_custom_priority(self, tmp_path):
        """Test trigger with custom priority."""
        trigger = PostUpdateTrigger(
            output_path=tmp_path / "context.md",
            priority=TriggerPriority.CRITICAL,
        )

        assert trigger.priority == TriggerPriority.CRITICAL


@pytest.mark.asyncio
class TestTriggerManagerPostUpdate:
    """Test TriggerManager integration with post-update triggers."""

    async def test_register_post_update_trigger(self, mock_memory_manager, tmp_path):
        """Test registering post-update trigger with TriggerManager."""
        manager = TriggerManager(mock_memory_manager)

        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        manager.register_trigger(trigger)

        # Verify trigger is registered
        on_rule_update_triggers = manager._triggers[TriggerPhase.ON_RULE_UPDATE]
        assert len(on_rule_update_triggers) == 1
        assert on_rule_update_triggers[0] == trigger

    async def test_execute_on_rule_update_phase(
        self, mock_memory_manager, tmp_path
    ):
        """Test executing ON_RULE_UPDATE phase."""
        manager = TriggerManager(mock_memory_manager)

        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )
        manager.register_trigger(trigger)

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            results = await manager.execute_phase(
                TriggerPhase.ON_RULE_UPDATE,
                project_root=tmp_path,
            )

            assert len(results) == 1
            assert results[0].success
            assert results[0].phase == TriggerPhase.ON_RULE_UPDATE

    async def test_multiple_post_update_triggers(
        self, mock_memory_manager, tmp_path
    ):
        """Test multiple post-update triggers with different priorities."""
        manager = TriggerManager(mock_memory_manager)

        # Register triggers with different priorities
        trigger_high = PostUpdateTrigger(
            output_path=tmp_path / "context1.md",
            priority=TriggerPriority.HIGH,
        )
        trigger_normal = PostUpdateTrigger(
            output_path=tmp_path / "context2.md",
            priority=TriggerPriority.NORMAL,
        )

        manager.register_trigger(trigger_high)
        manager.register_trigger(trigger_normal)

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            results = await manager.execute_phase(
                TriggerPhase.ON_RULE_UPDATE,
                project_root=tmp_path,
            )

            assert len(results) == 2
            # Both should succeed
            assert all(r.success for r in results)


@pytest.mark.asyncio
class TestPostUpdateIntegration:
    """Test post-update trigger integration scenarios."""

    async def test_rule_update_workflow(
        self, mock_memory_manager, tmp_path
    ):
        """Test typical rule update workflow."""
        # Create trigger
        trigger = PostUpdateTrigger(
            debounce_seconds=1.0,
            output_path=tmp_path / "context.md",
        )

        # Simulate rule updates
        trigger.record_change("Added: Use pytest for testing")
        trigger.record_change("Updated: Error handling guidelines")
        trigger.record_change("Removed: Obsolete logging rule")

        # Verify changes tracked
        assert len(trigger.get_pending_changes()) == 3

        # Mock execution context
        from src.python.common.core.context_injection.claude_code_detector import (
            ClaudeCodeSession,
        )
        session = ClaudeCodeSession(
            is_active=True,
            entrypoint="cli",
            detection_method="test",
            session_id="test",
        )
        context = TriggerContext(
            session=session,
            project_root=tmp_path,
            memory_manager=mock_memory_manager,
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # Execute trigger
            result = await trigger.execute(context)

            assert result.success
            assert result.metadata["changes_processed"] == 3
            assert len(trigger.get_pending_changes()) == 0

    async def test_debounce_with_multiple_rapid_changes(
        self, trigger_context, tmp_path
    ):
        """Test debouncing with multiple rapid rule changes."""
        trigger = PostUpdateTrigger(
            debounce_seconds=2.0,
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # Rapid changes
            trigger.record_change("Change 1")
            result1 = await trigger.execute(trigger_context)
            assert result1.success

            trigger.record_change("Change 2")
            result2 = await trigger.execute(trigger_context)
            assert result2.metadata.get("debounced") is True

            trigger.record_change("Change 3")
            result3 = await trigger.execute(trigger_context)
            assert result3.metadata.get("debounced") is True

            # Wait for debounce
            await asyncio.sleep(2.1)

            trigger.record_change("Change 4")
            result4 = await trigger.execute(trigger_context)
            assert result4.success
            assert result4.metadata.get("debounced") is not True
