"""
Unit tests for on-demand refresh triggers (Task 301.3).

Tests the on-demand refresh trigger functionality including:
- OnDemandRefreshTrigger execution
- Duplicate request prevention
- TriggerManager.trigger_manual_refresh()
- TriggerManager.execute_on_demand()
- refresh_claude_code_context() convenience function
"""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from src.python.common.core.context_injection.session_trigger import (
    OnDemandRefreshTrigger,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerContext,
    refresh_claude_code_context,
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
class TestOnDemandRefreshTrigger:
    """Test OnDemandRefreshTrigger class."""

    async def test_initialization(self, tmp_path):
        """Test trigger initialization."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
            output_path=tmp_path / "output.md",
            token_budget=10000,
        )

        assert trigger.name == "on_demand_refresh_full"
        assert trigger.phase == TriggerPhase.ON_DEMAND
        assert trigger.priority == TriggerPriority.HIGH
        assert trigger.refresh_type == "full"
        assert trigger.output_path == tmp_path / "output.md"
        assert trigger.token_budget == 10000
        assert trigger._refresh_count == 0
        assert trigger._last_refresh_time is None

    async def test_execute_full_refresh(self, trigger_context, tmp_path):
        """Test full refresh execution."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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
            assert result.phase == TriggerPhase.ON_DEMAND
            assert result.trigger_name == "on_demand_refresh_full"
            assert result.execution_time_ms > 0
            assert result.metadata["refresh_type"] == "full"
            assert result.metadata["refresh_count"] == 1
            assert trigger._refresh_count == 1

    async def test_duplicate_request_prevention(self, trigger_context, tmp_path):
        """Test that duplicate requests within 1 second are prevented."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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
            assert trigger._refresh_count == 1

            # Immediate second execution (should be skipped)
            result2 = await trigger.execute(trigger_context)
            assert result2.success
            assert result2.metadata.get("skipped") is True
            assert result2.metadata.get("reason") == "duplicate_request"
            assert trigger._refresh_count == 1  # Count unchanged

    async def test_refresh_after_delay(self, trigger_context, tmp_path):
        """Test that refresh works after sufficient delay."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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
            assert trigger._refresh_count == 1

            # Wait >1 second
            await asyncio.sleep(1.1)

            # Second execution (should succeed)
            result2 = await trigger.execute(trigger_context)
            assert result2.success
            assert result2.metadata.get("skipped") is not True
            assert trigger._refresh_count == 2

    async def test_get_refresh_stats(self, trigger_context, tmp_path):
        """Test refresh statistics retrieval."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="rules_only",
            output_path=tmp_path / "context.md",
        )

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # Before any execution
            stats = trigger.get_refresh_stats()
            assert stats["refresh_count"] == 0
            assert stats["last_refresh_time"] is None
            assert stats["seconds_since_last_refresh"] is None
            assert stats["refresh_type"] == "rules_only"

            # After execution
            await trigger.execute(trigger_context)
            stats = trigger.get_refresh_stats()
            assert stats["refresh_count"] == 1
            assert stats["last_refresh_time"] is not None
            assert stats["seconds_since_last_refresh"] is not None
            assert stats["seconds_since_last_refresh"] < 1.0

    async def test_execution_failure(self, trigger_context, tmp_path):
        """Test handling of execution failures."""
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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
            assert "Failed to refresh context" in result.error


@pytest.mark.asyncio
class TestTriggerManagerRefreshMethods:
    """Test TriggerManager on-demand refresh methods."""

    async def test_trigger_manual_refresh(self, mock_memory_manager, tmp_path):
        """Test TriggerManager.trigger_manual_refresh() method."""
        manager = TriggerManager(mock_memory_manager)

        # Register on-demand trigger
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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

            # Trigger manual refresh
            results = await manager.trigger_manual_refresh(project_root=tmp_path)

            assert len(results) == 1
            assert results[0].success
            assert results[0].phase == TriggerPhase.ON_DEMAND

    async def test_execute_on_demand(self, mock_memory_manager, tmp_path):
        """Test TriggerManager.execute_on_demand() method."""
        manager = TriggerManager(mock_memory_manager)

        # Register on-demand trigger
        trigger = OnDemandRefreshTrigger(
            refresh_type="full",
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

            # Execute on-demand
            results = await manager.execute_on_demand(project_root=tmp_path)

            assert len(results) == 1
            assert results[0].success
            assert results[0].phase == TriggerPhase.ON_DEMAND

    async def test_multiple_on_demand_triggers(self, mock_memory_manager, tmp_path):
        """Test execution of multiple on-demand triggers."""
        manager = TriggerManager(mock_memory_manager)

        # Register multiple on-demand triggers
        trigger1 = OnDemandRefreshTrigger(
            refresh_type="full",
            output_path=tmp_path / "context1.md",
            priority=TriggerPriority.HIGH,
        )
        trigger2 = OnDemandRefreshTrigger(
            refresh_type="rules_only",
            output_path=tmp_path / "context2.md",
            priority=TriggerPriority.NORMAL,
        )

        manager.register_trigger(trigger1)
        manager.register_trigger(trigger2)

        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            # Execute on-demand
            results = await manager.execute_on_demand(project_root=tmp_path)

            assert len(results) == 2
            assert all(r.success for r in results)
            # Higher priority trigger should execute first
            assert results[0].trigger_name == "on_demand_refresh_full"
            assert results[1].trigger_name == "on_demand_refresh_rules_only"


@pytest.mark.asyncio
class TestRefreshConvenienceFunction:
    """Test refresh_claude_code_context() convenience function."""

    async def test_refresh_claude_code_context(self, mock_memory_manager, tmp_path):
        """Test the refresh_claude_code_context() convenience function."""
        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            results = await refresh_claude_code_context(
                memory_manager=mock_memory_manager,
                project_root=tmp_path,
                output_path=tmp_path / "custom_output.md",
                token_budget=20000,
                refresh_type="full",
            )

            assert len(results) == 1
            assert results[0].success
            assert results[0].phase == TriggerPhase.ON_DEMAND
            assert results[0].metadata["refresh_type"] == "full"

    async def test_refresh_with_default_paths(self, mock_memory_manager, tmp_path):
        """Test refresh with default output path."""
        # Mock ClaudeMdInjector
        with patch(
            "src.python.common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as mock_injector_class:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            mock_injector_class.return_value = mock_injector

            results = await refresh_claude_code_context(
                memory_manager=mock_memory_manager,
                project_root=tmp_path,
            )

            assert len(results) == 1
            assert results[0].success
            # Should use default output path .claude/context.md
            output_path = results[0].metadata.get("output_path")
            assert output_path is None or ".claude/context.md" in str(output_path)
