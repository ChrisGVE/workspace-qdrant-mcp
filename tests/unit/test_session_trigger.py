"""
Unit tests for session trigger handling.

Tests trigger execution, lifecycle management, cleanup, and integration
with detector and injector modules.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from common.core.context_injection import (
    ClaudeCodeSession,
    ClaudeMdFileTrigger,
    CleanupTrigger,
    CustomCallbackTrigger,
    SessionTrigger,
    SystemPromptTrigger,
    TriggerContext,
    TriggerManager,
    TriggerPhase,
    TriggerPriority,
    TriggerResult,
    cleanup_claude_code_session,
    prepare_claude_code_session,
)
from common.core.context_injection.system_prompt_injector import (
    InjectionMode,
    SystemPromptConfig,
)
from common.core.memory import (
    AuthorityLevel,
    MemoryCategory,
    MemoryManager,
    MemoryRule,
)


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager."""
    manager = AsyncMock(spec=MemoryManager)
    return manager


@pytest.fixture
def sample_session():
    """Create a sample Claude Code session."""
    return ClaudeCodeSession(
        is_active=True,
        entrypoint="cli",
        detection_method="environment_variable_claudecode",
    )


@pytest.fixture
def sample_context(mock_memory_manager, sample_session, tmp_path):
    """Create a sample TriggerContext."""
    return TriggerContext(
        session=sample_session,
        project_root=tmp_path,
        memory_manager=mock_memory_manager,
    )


class TestTriggerResult:
    """Test TriggerResult dataclass."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = TriggerResult(
            success=True,
            phase=TriggerPhase.PRE_SESSION,
            trigger_name="test_trigger",
            execution_time_ms=10.5,
            metadata={"key": "value"},
        )

        assert result.success is True
        assert result.phase == TriggerPhase.PRE_SESSION
        assert result.trigger_name == "test_trigger"
        assert result.execution_time_ms == 10.5
        assert result.error is None
        assert result.metadata == {"key": "value"}

    def test_failure_result(self):
        """Test creating a failure result."""
        result = TriggerResult(
            success=False,
            phase=TriggerPhase.POST_SESSION,
            trigger_name="cleanup",
            execution_time_ms=5.0,
            error="File not found",
        )

        assert result.success is False
        assert result.error == "File not found"


class TestTriggerContext:
    """Test TriggerContext dataclass."""

    def test_context_creation(self, sample_session, mock_memory_manager, tmp_path):
        """Test creating TriggerContext."""
        context = TriggerContext(
            session=sample_session,
            project_root=tmp_path,
            memory_manager=mock_memory_manager,
        )

        assert context.session == sample_session
        assert context.project_root == tmp_path
        assert context.memory_manager == mock_memory_manager
        assert context.trigger_metadata == {}

    def test_shared_metadata(self, sample_context):
        """Test shared metadata between triggers."""
        sample_context.trigger_metadata["test_key"] = "test_value"
        assert sample_context.trigger_metadata["test_key"] == "test_value"


class MockTrigger(SessionTrigger):
    """Mock trigger for testing."""

    def __init__(
        self,
        name: str = "mock_trigger",
        phase: TriggerPhase = TriggerPhase.PRE_SESSION,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        should_fail: bool = False,
    ):
        super().__init__(name, phase, priority)
        self.should_fail = should_fail
        self.execute_called = False
        self.cleanup_called = False

    async def execute(self, context: TriggerContext) -> TriggerResult:
        """Execute mock trigger."""
        self.execute_called = True

        if self.should_fail:
            return TriggerResult(
                success=False,
                phase=self.phase,
                trigger_name=self.name,
                execution_time_ms=1.0,
                error="Mock failure",
            )

        return TriggerResult(
            success=True,
            phase=self.phase,
            trigger_name=self.name,
            execution_time_ms=1.0,
        )

    async def cleanup(self, context: TriggerContext) -> None:
        """Cleanup mock trigger."""
        self.cleanup_called = True


class TestClaudeMdFileTrigger:
    """Test ClaudeMdFileTrigger class."""

    @pytest.mark.asyncio
    async def test_trigger_initialization(self, tmp_path):
        """Test trigger initialization."""
        output_path = tmp_path / "output.md"
        trigger = ClaudeMdFileTrigger(output_path=output_path, token_budget=10000)

        assert trigger.name == "claude_md_file_injection"
        assert trigger.phase == TriggerPhase.PRE_SESSION
        assert trigger.priority == TriggerPriority.HIGH
        assert trigger.output_path == output_path
        assert trigger.token_budget == 10000

    @pytest.mark.asyncio
    async def test_execute_success(self, sample_context, tmp_path):
        """Test successful execution."""
        output_path = tmp_path / "output.md"
        trigger = ClaudeMdFileTrigger(output_path=output_path)

        # Mock the injector
        with patch(
            "common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            MockInjector.return_value = mock_injector

            result = await trigger.execute(sample_context)

            assert result.success is True
            assert result.phase == TriggerPhase.PRE_SESSION
            assert result.trigger_name == "claude_md_file_injection"
            assert result.execution_time_ms >= 0
            assert str(output_path) in str(result.metadata["output_path"])

    @pytest.mark.asyncio
    async def test_execute_failure(self, sample_context, tmp_path):
        """Test execution failure."""
        output_path = tmp_path / "output.md"
        trigger = ClaudeMdFileTrigger(output_path=output_path)

        # Mock the injector to fail
        with patch(
            "common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=False)
            MockInjector.return_value = mock_injector

            result = await trigger.execute(sample_context)

            assert result.success is False
            assert result.error == "Failed to inject content"

    @pytest.mark.asyncio
    async def test_cleanup(self, sample_context, tmp_path):
        """Test cleanup stops watching."""
        output_path = tmp_path / "output.md"
        trigger = ClaudeMdFileTrigger(output_path=output_path)

        # Create mock injector
        trigger._injector = Mock()
        trigger._injector.stop_watching = Mock()

        await trigger.cleanup(sample_context)

        trigger._injector.stop_watching.assert_called_once()


class TestSystemPromptTrigger:
    """Test SystemPromptTrigger class."""

    @pytest.mark.asyncio
    async def test_trigger_initialization(self):
        """Test trigger initialization."""
        config = SystemPromptConfig(token_budget=5000)
        trigger = SystemPromptTrigger(config=config)

        assert trigger.name == "system_prompt_generation"
        assert trigger.phase == TriggerPhase.PRE_SESSION
        assert trigger.config == config

    @pytest.mark.asyncio
    async def test_execute_success(self, sample_context):
        """Test successful system prompt generation."""
        trigger = SystemPromptTrigger()

        # Mock the injector
        with patch(
            "common.core.context_injection.session_trigger.SystemPromptInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.generate_system_prompt = AsyncMock(
                return_value="Generated prompt content"
            )
            MockInjector.return_value = mock_injector

            result = await trigger.execute(sample_context)

            assert result.success is True
            assert result.metadata["prompt_length"] == len("Generated prompt content")
            assert trigger.get_prompt() == "Generated prompt content"
            assert sample_context.trigger_metadata["system_prompt"] == "Generated prompt content"

    @pytest.mark.asyncio
    async def test_execute_with_file_output(self, sample_context, tmp_path):
        """Test system prompt generation with file output."""
        output_path = tmp_path / "prompt.txt"
        trigger = SystemPromptTrigger(output_path=output_path)

        # Mock the injector
        with patch(
            "common.core.context_injection.session_trigger.SystemPromptInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.generate_system_prompt = AsyncMock(
                return_value="Test prompt"
            )
            MockInjector.return_value = mock_injector

            result = await trigger.execute(sample_context)

            assert result.success is True
            assert output_path.exists()
            assert output_path.read_text() == "Test prompt"


class TestCleanupTrigger:
    """Test CleanupTrigger class."""

    @pytest.mark.asyncio
    async def test_trigger_initialization(self):
        """Test trigger initialization."""
        paths = [Path("/tmp/file1"), Path("/tmp/file2")]
        trigger = CleanupTrigger(cleanup_paths=paths)

        assert trigger.name == "session_cleanup"
        assert trigger.phase == TriggerPhase.POST_SESSION
        assert trigger.cleanup_paths == paths

    @pytest.mark.asyncio
    async def test_cleanup_files(self, sample_context, tmp_path):
        """Test cleaning up files."""
        # Create test files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("test")
        file2.write_text("test")

        trigger = CleanupTrigger(cleanup_paths=[file1, file2])
        result = await trigger.execute(sample_context)

        assert result.success is True
        assert not file1.exists()
        assert not file2.exists()
        assert len(result.metadata["removed_files"]) == 2

    @pytest.mark.asyncio
    async def test_cleanup_empty_directory(self, sample_context, tmp_path):
        """Test cleaning up empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        trigger = CleanupTrigger(cleanup_paths=[empty_dir])
        result = await trigger.execute(sample_context)

        assert result.success is True
        assert not empty_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_file(self, sample_context, tmp_path):
        """Test cleaning up nonexistent file (should succeed)."""
        nonexistent = tmp_path / "nonexistent.txt"

        trigger = CleanupTrigger(cleanup_paths=[nonexistent])
        result = await trigger.execute(sample_context)

        assert result.success is True
        assert len(result.metadata["removed_files"]) == 0

    @pytest.mark.asyncio
    async def test_add_cleanup_path(self):
        """Test adding cleanup path."""
        trigger = CleanupTrigger()
        path = Path("/tmp/test.txt")

        trigger.add_cleanup_path(path)

        assert path in trigger.cleanup_paths


class TestCustomCallbackTrigger:
    """Test CustomCallbackTrigger class."""

    @pytest.mark.asyncio
    async def test_sync_callback(self, sample_context):
        """Test synchronous callback execution."""
        callback_executed = []

        def sync_callback(context: TriggerContext):
            callback_executed.append(True)
            return "callback_result"

        trigger = CustomCallbackTrigger(
            name="test_callback",
            callback=sync_callback,
            is_async=False,
        )

        result = await trigger.execute(sample_context)

        assert result.success is True
        assert callback_executed == [True]
        assert result.metadata["callback_result"] == "callback_result"

    @pytest.mark.asyncio
    async def test_async_callback(self, sample_context):
        """Test asynchronous callback execution."""
        callback_executed = []

        async def async_callback(context: TriggerContext):
            callback_executed.append(True)
            await asyncio.sleep(0.001)  # Simulate async work
            return "async_result"

        trigger = CustomCallbackTrigger(
            name="test_async_callback",
            callback=async_callback,
            is_async=True,
        )

        result = await trigger.execute(sample_context)

        assert result.success is True
        assert callback_executed == [True]
        assert result.metadata["callback_result"] == "async_result"

    @pytest.mark.asyncio
    async def test_callback_failure(self, sample_context):
        """Test callback that raises exception."""

        def failing_callback(context: TriggerContext):
            raise ValueError("Callback failed")

        trigger = CustomCallbackTrigger(
            name="failing_callback",
            callback=failing_callback,
        )

        result = await trigger.execute(sample_context)

        assert result.success is False
        assert "Callback failed" in result.error


class TestTriggerManager:
    """Test TriggerManager class."""

    def test_initialization(self, mock_memory_manager):
        """Test manager initialization."""
        manager = TriggerManager(mock_memory_manager)

        assert manager.memory_manager == mock_memory_manager
        assert manager.detector is not None

    def test_register_trigger(self, mock_memory_manager):
        """Test registering a trigger."""
        manager = TriggerManager(mock_memory_manager)
        trigger = MockTrigger()

        manager.register_trigger(trigger)

        triggers = manager.get_triggers(TriggerPhase.PRE_SESSION)
        assert trigger in triggers

    def test_unregister_trigger(self, mock_memory_manager):
        """Test unregistering a trigger."""
        manager = TriggerManager(mock_memory_manager)
        trigger = MockTrigger()

        manager.register_trigger(trigger)
        result = manager.unregister_trigger(trigger)

        assert result is True
        triggers = manager.get_triggers(TriggerPhase.PRE_SESSION)
        assert trigger not in triggers

    def test_get_triggers_sorted_by_priority(self, mock_memory_manager):
        """Test getting triggers sorted by priority."""
        manager = TriggerManager(mock_memory_manager)

        high_trigger = MockTrigger(name="high", priority=TriggerPriority.HIGH)
        low_trigger = MockTrigger(name="low", priority=TriggerPriority.LOW)
        critical_trigger = MockTrigger(name="critical", priority=TriggerPriority.CRITICAL)

        manager.register_trigger(low_trigger)
        manager.register_trigger(high_trigger)
        manager.register_trigger(critical_trigger)

        triggers = manager.get_triggers(TriggerPhase.PRE_SESSION)

        # Should be sorted by priority (highest first)
        assert triggers[0] == critical_trigger
        assert triggers[1] == high_trigger
        assert triggers[2] == low_trigger

    def test_get_triggers_filters_disabled(self, mock_memory_manager):
        """Test that disabled triggers are filtered out."""
        manager = TriggerManager(mock_memory_manager)

        enabled_trigger = MockTrigger(name="enabled")
        disabled_trigger = MockTrigger(name="disabled")
        disabled_trigger.enabled = False

        manager.register_trigger(enabled_trigger)
        manager.register_trigger(disabled_trigger)

        triggers = manager.get_triggers(TriggerPhase.PRE_SESSION)

        assert enabled_trigger in triggers
        assert disabled_trigger not in triggers

    @pytest.mark.asyncio
    async def test_execute_phase_success(self, mock_memory_manager, tmp_path):
        """Test executing a phase successfully."""
        manager = TriggerManager(mock_memory_manager)

        trigger1 = MockTrigger(name="trigger1")
        trigger2 = MockTrigger(name="trigger2")

        manager.register_trigger(trigger1)
        manager.register_trigger(trigger2)

        results = await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
        )

        assert len(results) == 2
        assert all(r.success for r in results)
        assert trigger1.execute_called
        assert trigger2.execute_called

    @pytest.mark.asyncio
    async def test_execute_phase_with_failure(self, mock_memory_manager, tmp_path):
        """Test executing a phase with one failure."""
        manager = TriggerManager(mock_memory_manager)

        success_trigger = MockTrigger(name="success")
        fail_trigger = MockTrigger(name="fail", should_fail=True)

        manager.register_trigger(success_trigger)
        manager.register_trigger(fail_trigger)

        results = await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
        )

        assert len(results) == 2
        success_results = [r for r in results if r.success]
        fail_results = [r for r in results if not r.success]

        assert len(success_results) == 1
        assert len(fail_results) == 1

    @pytest.mark.asyncio
    async def test_execute_phase_fail_fast(self, mock_memory_manager, tmp_path):
        """Test fail_fast stops execution on first failure."""
        manager = TriggerManager(mock_memory_manager)

        # Register in order: success, fail, success
        # With fail_fast, third should not execute
        trigger1 = MockTrigger(name="trigger1", priority=TriggerPriority.HIGH)
        trigger2 = MockTrigger(name="trigger2", priority=TriggerPriority.NORMAL, should_fail=True)
        trigger3 = MockTrigger(name="trigger3", priority=TriggerPriority.LOW)

        manager.register_trigger(trigger1)
        manager.register_trigger(trigger2)
        manager.register_trigger(trigger3)

        await manager.execute_phase(
            TriggerPhase.PRE_SESSION,
            project_root=tmp_path,
            fail_fast=True,
        )

        # Should stop after trigger2 fails
        assert trigger1.execute_called
        assert trigger2.execute_called
        assert not trigger3.execute_called

    @pytest.mark.asyncio
    async def test_execute_session_lifecycle(self, mock_memory_manager, tmp_path):
        """Test executing complete session lifecycle."""
        manager = TriggerManager(mock_memory_manager)

        pre_trigger = MockTrigger(name="pre", phase=TriggerPhase.PRE_SESSION)
        post_trigger = MockTrigger(name="post", phase=TriggerPhase.POST_SESSION)

        manager.register_trigger(pre_trigger)
        manager.register_trigger(post_trigger)

        results = await manager.execute_session_lifecycle(project_root=tmp_path)

        assert TriggerPhase.PRE_SESSION in results
        assert TriggerPhase.POST_SESSION in results
        assert pre_trigger.execute_called
        assert post_trigger.execute_called

    @pytest.mark.asyncio
    async def test_cleanup_all(self, mock_memory_manager, tmp_path):
        """Test cleanup on all triggers."""
        manager = TriggerManager(mock_memory_manager)

        trigger1 = MockTrigger(name="trigger1")
        trigger2 = MockTrigger(name="trigger2")

        manager.register_trigger(trigger1)
        manager.register_trigger(trigger2)

        await manager.cleanup_all(project_root=tmp_path)

        assert trigger1.cleanup_called
        assert trigger2.cleanup_called

    def test_execution_history(self, mock_memory_manager):
        """Test execution history tracking."""
        manager = TriggerManager(mock_memory_manager)

        assert len(manager.get_execution_history()) == 0

        # Add some results directly to history
        result1 = TriggerResult(
            success=True,
            phase=TriggerPhase.PRE_SESSION,
            trigger_name="test1",
            execution_time_ms=1.0,
        )
        manager._execution_history.append(result1)

        assert len(manager.get_execution_history()) == 1

        manager.clear_history()
        assert len(manager.get_execution_history()) == 0


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_prepare_claude_code_session(self, mock_memory_manager, tmp_path):
        """Test prepare_claude_code_session convenience function."""
        output_path = tmp_path / ".claude" / "context.md"

        # Mock the injector
        with patch(
            "common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            MockInjector.return_value = mock_injector

            results = await prepare_claude_code_session(
                memory_manager=mock_memory_manager,
                project_root=tmp_path,
                output_path=output_path,
            )

            assert len(results) >= 1
            assert any(r.success for r in results)

    @pytest.mark.asyncio
    async def test_cleanup_claude_code_session(self, mock_memory_manager, tmp_path):
        """Test cleanup_claude_code_session convenience function."""
        # Create a test file to cleanup
        test_file = tmp_path / "cleanup_test.txt"
        test_file.write_text("test")

        results = await cleanup_claude_code_session(
            memory_manager=mock_memory_manager,
            cleanup_paths=[test_file],
            project_root=tmp_path,
        )

        assert len(results) >= 1
        assert any(r.success for r in results)
        assert not test_file.exists()


class TestIntegration:
    """Integration tests combining multiple triggers."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle_integration(
        self, mock_memory_manager, tmp_path
    ):
        """Test complete session lifecycle with multiple triggers."""
        output_file = tmp_path / "output.md"
        cleanup_file = tmp_path / "cleanup.txt"
        cleanup_file.write_text("test")

        # Create manager
        manager = TriggerManager(mock_memory_manager)

        # Register pre-session trigger
        with patch(
            "common.core.context_injection.session_trigger.ClaudeMdInjector"
        ) as MockInjector:
            mock_injector = AsyncMock()
            mock_injector.inject_to_file = AsyncMock(return_value=True)
            MockInjector.return_value = mock_injector

            file_trigger = ClaudeMdFileTrigger(output_path=output_file)
            manager.register_trigger(file_trigger)

            # Register post-session cleanup
            cleanup_trigger = CleanupTrigger(cleanup_paths=[cleanup_file])
            manager.register_trigger(cleanup_trigger)

            # Execute lifecycle
            results = await manager.execute_session_lifecycle(project_root=tmp_path)

            # Verify pre-session executed
            assert TriggerPhase.PRE_SESSION in results
            assert any(r.success for r in results[TriggerPhase.PRE_SESSION])

            # Verify post-session cleanup executed
            assert TriggerPhase.POST_SESSION in results
            assert any(r.success for r in results[TriggerPhase.POST_SESSION])
            assert not cleanup_file.exists()

    @pytest.mark.asyncio
    async def test_metadata_sharing_between_triggers(self, mock_memory_manager, tmp_path):
        """Test triggers sharing data via context metadata."""
        manager = TriggerManager(mock_memory_manager)

        # First trigger stores metadata
        def first_callback(context: TriggerContext):
            context.trigger_metadata["shared_data"] = "test_value"

        first_trigger = CustomCallbackTrigger(
            name="first",
            callback=first_callback,
            priority=TriggerPriority.HIGH,
        )

        # Second trigger reads metadata
        second_result = []

        def second_callback(context: TriggerContext):
            second_result.append(context.trigger_metadata.get("shared_data"))

        second_trigger = CustomCallbackTrigger(
            name="second",
            callback=second_callback,
            priority=TriggerPriority.NORMAL,
        )

        manager.register_trigger(first_trigger)
        manager.register_trigger(second_trigger)

        await manager.execute_phase(TriggerPhase.PRE_SESSION, project_root=tmp_path)

        assert second_result == ["test_value"]
