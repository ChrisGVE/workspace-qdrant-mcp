"""
Tests for Component Lifecycle Manager.

This module tests the component lifecycle orchestration capabilities including
startup sequences, dependency management, health monitoring, and graceful shutdown.
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

from common.core.component_lifecycle import (
    ComponentLifecycleManager,
    ComponentConfig,
    ComponentState,
    LifecyclePhase,
    StartupDependency,
    LifecycleEvent,
    get_lifecycle_manager,
    shutdown_lifecycle_manager,
)
from common.core.component_coordination import (
    ComponentType,
    ComponentStatus,
    ComponentHealth,
)


class TestComponentLifecycleManager:
    """Test Component Lifecycle Manager functionality."""

    @pytest.fixture
    async def lifecycle_manager(self):
        """Create a test lifecycle manager."""
        # Use temporary database
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        manager = ComponentLifecycleManager(
            db_path=db_path,
            project_name="test_project",
            project_path="/tmp/test_project"
        )

        # Mock the coordinator to avoid actual SQLite operations
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_coordinator.register_component = AsyncMock(return_value="test-component-id")
            mock_coordinator.update_component_status = AsyncMock(return_value=True)
            mock_coordinator.update_component_health = AsyncMock(return_value=True)
            mock_coordinator.enqueue_processing_item = AsyncMock(return_value="queue-item-id")
            mock_coordinator.close = AsyncMock()
            mock_get_coordinator.return_value = mock_coordinator

            await manager.initialize()

        yield manager

        await manager.shutdown_sequence()

    @pytest.fixture
    def mock_daemon_manager(self):
        """Mock daemon manager for testing."""
        with patch('common.core.component_lifecycle.DaemonManager') as mock_dm:
            yield mock_dm

    @pytest.fixture
    def mock_ensure_daemon_running(self):
        """Mock ensure_daemon_running function."""
        with patch('common.core.component_lifecycle.ensure_daemon_running') as mock_edr:
            mock_daemon = Mock()
            mock_daemon.health_check = AsyncMock(return_value=True)
            mock_daemon.config.grpc_port = 50051
            mock_daemon.status.state = "running"
            mock_daemon.shutdown = AsyncMock()
            mock_edr.return_value = mock_daemon
            yield mock_edr

    @pytest.fixture
    def mock_grpc_client(self):
        """Mock gRPC workspace client."""
        with patch('common.core.component_lifecycle.GrpcWorkspaceClient') as mock_grpc:
            mock_client = AsyncMock()
            mock_client.initialize = AsyncMock()
            mock_client.get_operation_mode = Mock(return_value="grpc")
            mock_client.get_status = AsyncMock(return_value={"qdrant_available": True})
            mock_client.is_grpc_available = Mock(return_value=True)
            mock_client.close = AsyncMock()
            mock_grpc.return_value = mock_client
            yield mock_grpc

    @pytest.fixture
    def mock_cli_app(self):
        """Mock CLI application."""
        with patch('common.core.component_lifecycle.cli_app') as mock_cli:
            yield mock_cli

    def test_initialization(self, lifecycle_manager):
        """Test lifecycle manager initialization."""
        assert lifecycle_manager.project_name == "test_project"
        assert lifecycle_manager.project_path == "/tmp/test_project"
        assert lifecycle_manager.current_phase == LifecyclePhase.INITIALIZATION
        assert lifecycle_manager.coordinator is not None
        assert len(lifecycle_manager.component_configs) == 4

    def test_default_configs(self):
        """Test default component configurations."""
        manager = ComponentLifecycleManager()

        # Check Rust daemon config
        rust_config = manager.component_configs[ComponentType.RUST_DAEMON]
        assert rust_config.startup_dependency == StartupDependency.RUST_DAEMON
        assert rust_config.startup_timeout == 45.0
        assert "grpc_server_responsive" in rust_config.readiness_checks

        # Check Python MCP server config
        mcp_config = manager.component_configs[ComponentType.PYTHON_MCP_SERVER]
        assert mcp_config.startup_dependency == StartupDependency.PYTHON_MCP_SERVER
        assert mcp_config.startup_timeout == 30.0
        assert "mcp_server_listening" in mcp_config.readiness_checks

    def test_startup_order(self):
        """Test component startup order."""
        manager = ComponentLifecycleManager()

        expected_order = [
            StartupDependency.SQLITE_MANAGER,
            StartupDependency.RUST_DAEMON,
            StartupDependency.PYTHON_MCP_SERVER,
            StartupDependency.CLI_CONTEXT_INJECTOR,
        ]

        assert manager.STARTUP_ORDER == expected_order
        assert manager.SHUTDOWN_ORDER == list(reversed(expected_order))

    @pytest.mark.asyncio
    async def test_mark_sqlite_ready(self, lifecycle_manager):
        """Test marking SQLite manager as ready."""
        await lifecycle_manager._mark_sqlite_ready()

        # Verify SQLite manager is registered in coordinator
        lifecycle_manager.coordinator.register_component.assert_called_once()
        lifecycle_manager.coordinator.update_component_status.assert_called_once()

        # Verify lifecycle event was logged
        assert len(lifecycle_manager.startup_events) > 0
        event = lifecycle_manager.startup_events[-1]
        assert event.component_id == "sqlite_manager"
        assert event.phase == LifecyclePhase.COMPONENT_STARTUP

    @pytest.mark.asyncio
    async def test_start_rust_daemon(
        self,
        lifecycle_manager,
        mock_ensure_daemon_running
    ):
        """Test starting Rust daemon component."""
        config = lifecycle_manager.component_configs[ComponentType.RUST_DAEMON]

        instance = await lifecycle_manager._start_rust_daemon(config)

        assert instance is not None
        mock_ensure_daemon_running.assert_called_once()

        # Verify daemon health check was called
        instance.health_check.assert_called()

    @pytest.mark.asyncio
    async def test_start_python_mcp_server(
        self,
        lifecycle_manager,
        mock_grpc_client
    ):
        """Test starting Python MCP server component."""
        config = lifecycle_manager.component_configs[ComponentType.PYTHON_MCP_SERVER]

        with patch('workspace_qdrant_mcp.server.Config') as mock_config:
            mock_config.return_value = Mock()

            instance = await lifecycle_manager._start_python_mcp_server(config)

            assert instance is not None
            mock_grpc_client.assert_called_once()

            # Verify gRPC client was initialized
            instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_cli_utility(self, lifecycle_manager, mock_cli_app):
        """Test starting CLI utility component."""
        config = lifecycle_manager.component_configs[ComponentType.CLI_UTILITY]

        with patch('workspace_qdrant_mcp.cli.main.app', mock_cli_app):
            instance = await lifecycle_manager._start_cli_utility(config)

            assert instance is not None
            assert instance["status"] == "ready"

    @pytest.mark.asyncio
    async def test_start_context_injector(self, lifecycle_manager):
        """Test starting context injector component."""
        config = lifecycle_manager.component_configs[ComponentType.CONTEXT_INJECTOR]

        instance = await lifecycle_manager._start_context_injector(config)

        assert instance is not None
        assert instance["status"] == "ready"
        assert instance["hooks_registered"] is True

    @pytest.mark.asyncio
    async def test_validate_component_readiness(self, lifecycle_manager):
        """Test component readiness validation."""
        # Mock a component instance
        mock_instance = Mock()
        mock_instance.health_check = AsyncMock(return_value=True)
        mock_instance.status.state = "running"
        mock_instance.get_operation_mode = Mock(return_value="grpc")
        mock_instance.get_status = AsyncMock(return_value={"qdrant_available": True})
        mock_instance.is_grpc_available = Mock(return_value=True)
        mock_instance.get = Mock(return_value="ready")

        lifecycle_manager.component_instances[ComponentType.RUST_DAEMON] = mock_instance

        # Test successful validation
        result = await lifecycle_manager._validate_component_readiness(ComponentType.RUST_DAEMON)
        assert result is True
        assert lifecycle_manager.component_states[ComponentType.RUST_DAEMON] == ComponentState.OPERATIONAL

    @pytest.mark.asyncio
    async def test_readiness_checks(self, lifecycle_manager):
        """Test specific readiness checks."""
        # Test Rust daemon checks
        mock_daemon = Mock()
        mock_daemon.health_check = AsyncMock(return_value=True)
        mock_daemon.status.state = "running"

        lifecycle_manager.component_instances[ComponentType.RUST_DAEMON] = mock_daemon

        # Test gRPC server responsive check
        result = await lifecycle_manager._perform_readiness_check(
            ComponentType.RUST_DAEMON, "grpc_server_responsive"
        )
        assert result is True

        # Test SQLite connection active check
        result = await lifecycle_manager._perform_readiness_check(
            ComponentType.RUST_DAEMON, "sqlite_connection_active"
        )
        assert result is True

        # Test process health check
        result = await lifecycle_manager._perform_readiness_check(
            ComponentType.RUST_DAEMON, "process_health_ok"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_component_registration(self, lifecycle_manager):
        """Test component registration in coordinator."""
        component_id = await lifecycle_manager._register_component(ComponentType.RUST_DAEMON)

        assert component_id == "test-component-id"
        lifecycle_manager.coordinator.register_component.assert_called_once()

        # Verify registration parameters
        call_args = lifecycle_manager.coordinator.register_component.call_args
        assert call_args[1]["component_type"] == ComponentType.RUST_DAEMON
        assert "rust_daemon-test_project" in call_args[1]["instance_id"]

    @pytest.mark.asyncio
    async def test_lifecycle_event_logging(self, lifecycle_manager):
        """Test lifecycle event logging."""
        # Clear any existing events from initialization
        lifecycle_manager.startup_events.clear()

        await lifecycle_manager._log_lifecycle_event(
            component_id="test_component",
            phase=LifecyclePhase.COMPONENT_STARTUP,
            event_type="startup",
            message="Test event",
            duration_ms=100.0,
            details={"test": "data"}
        )

        assert len(lifecycle_manager.startup_events) == 1
        event = lifecycle_manager.startup_events[0]

        assert event.component_id == "test_component"
        assert event.phase == LifecyclePhase.COMPONENT_STARTUP
        assert event.event_type == "startup"
        assert event.message == "Test event"
        assert event.duration_ms == 100.0
        assert event.details == {"test": "data"}
        assert event.success is True

    @pytest.mark.asyncio
    async def test_startup_sequence_success(
        self,
        lifecycle_manager,
        mock_ensure_daemon_running,
        mock_grpc_client,
        mock_cli_app
    ):
        """Test successful startup sequence."""
        # Mock all component starting methods
        lifecycle_manager._start_rust_daemon = AsyncMock(return_value=Mock())
        lifecycle_manager._start_python_mcp_server = AsyncMock(return_value=Mock())
        lifecycle_manager._start_cli_utility = AsyncMock(return_value=Mock())
        lifecycle_manager._start_context_injector = AsyncMock(return_value=Mock())

        # Mock readiness validation
        lifecycle_manager._validate_component_readiness = AsyncMock(return_value=True)
        lifecycle_manager._validate_all_components_ready = AsyncMock(return_value=True)

        # Mock monitoring tasks
        lifecycle_manager._start_lifecycle_monitoring = AsyncMock()

        # Mock CLI imports
        with patch('workspace_qdrant_mcp.cli.main.app') as mock_cli, \
             patch('workspace_qdrant_mcp.server.Config') as mock_config:
            mock_config.return_value = Mock()
            mock_cli.return_value = Mock()

        result = await lifecycle_manager.startup_sequence()

        assert result is True
        assert lifecycle_manager.current_phase == LifecyclePhase.OPERATIONAL
        assert lifecycle_manager.startup_start_time is not None

        # Verify components were started
        lifecycle_manager._start_rust_daemon.assert_called_once()
        lifecycle_manager._start_python_mcp_server.assert_called_once()
        lifecycle_manager._start_cli_utility.assert_called_once()
        lifecycle_manager._start_context_injector.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_sequence_failure(self, lifecycle_manager):
        """Test startup sequence failure handling."""
        # Mock component start failure
        lifecycle_manager._start_rust_daemon = AsyncMock(return_value=None)
        lifecycle_manager._handle_startup_failure = AsyncMock()
        lifecycle_manager._start_lifecycle_monitoring = AsyncMock()

        result = await lifecycle_manager.startup_sequence()

        assert result is False
        lifecycle_manager._handle_startup_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_sequence(self, lifecycle_manager):
        """Test graceful shutdown sequence."""
        # Add mock component instances
        lifecycle_manager.component_instances = {
            ComponentType.RUST_DAEMON: Mock(),
            ComponentType.PYTHON_MCP_SERVER: Mock(),
            ComponentType.CLI_UTILITY: Mock(),
            ComponentType.CONTEXT_INJECTOR: Mock(),
        }

        # Mock shutdown methods
        lifecycle_manager._shutdown_component = AsyncMock()
        lifecycle_manager._cleanup_lifecycle_resources = AsyncMock()

        result = await lifecycle_manager.shutdown_sequence()

        assert result is True
        assert lifecycle_manager.current_phase == LifecyclePhase.STOPPED
        assert lifecycle_manager.shutdown_start_time is not None

        # Verify shutdown was called for each component
        assert lifecycle_manager._shutdown_component.call_count == 4

    @pytest.mark.asyncio
    async def test_restart_component(self, lifecycle_manager):
        """Test component restart functionality."""
        # Add mock component instance
        mock_instance = Mock()
        lifecycle_manager.component_instances[ComponentType.RUST_DAEMON] = mock_instance

        # Mock methods
        lifecycle_manager._shutdown_component = AsyncMock()
        lifecycle_manager._start_component = AsyncMock(return_value=True)
        lifecycle_manager._validate_component_readiness = AsyncMock(return_value=True)

        result = await lifecycle_manager.restart_component(ComponentType.RUST_DAEMON)

        assert result is True
        lifecycle_manager._shutdown_component.assert_called_once_with(ComponentType.RUST_DAEMON)
        lifecycle_manager._start_component.assert_called_once_with(ComponentType.RUST_DAEMON)

    @pytest.mark.asyncio
    async def test_restart_nonexistent_component(self, lifecycle_manager):
        """Test restart of non-existent component."""
        result = await lifecycle_manager.restart_component(ComponentType.RUST_DAEMON)
        assert result is False

    @pytest.mark.asyncio
    async def test_get_component_status(self, lifecycle_manager):
        """Test getting comprehensive component status."""
        # Add mock component instances
        lifecycle_manager.component_instances[ComponentType.RUST_DAEMON] = Mock()
        lifecycle_manager.component_states[ComponentType.RUST_DAEMON] = ComponentState.OPERATIONAL

        # Mock coordinator status
        lifecycle_manager.coordinator.get_component_status = AsyncMock(
            return_value={"status": "healthy"}
        )

        status = await lifecycle_manager.get_component_status()

        assert "lifecycle_manager" in status
        assert "components" in status
        assert "startup_events" in status

        assert status["lifecycle_manager"]["project_name"] == "test_project"
        assert ComponentType.RUST_DAEMON.value in status["components"]

        component_status = status["components"][ComponentType.RUST_DAEMON.value]
        assert component_status["state"] == ComponentState.OPERATIONAL.value
        assert component_status["instance_active"] is True

    @pytest.mark.asyncio
    async def test_health_monitoring(self, lifecycle_manager):
        """Test component health monitoring."""
        # Mock component instance
        mock_instance = Mock()
        lifecycle_manager.component_instances[ComponentType.RUST_DAEMON] = mock_instance
        lifecycle_manager.component_states[ComponentType.RUST_DAEMON] = ComponentState.OPERATIONAL

        # Mock readiness validation to fail
        lifecycle_manager._validate_component_readiness = AsyncMock(return_value=False)

        # Start monitoring task
        task = asyncio.create_task(lifecycle_manager._component_health_monitor())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify component state was updated to degraded
        assert lifecycle_manager.component_states[ComponentType.RUST_DAEMON] == ComponentState.DEGRADED

    @pytest.mark.asyncio
    async def test_dependency_monitoring(self, lifecycle_manager):
        """Test component dependency monitoring."""
        # Set component as failed
        lifecycle_manager.component_states[ComponentType.RUST_DAEMON] = ComponentState.FAILED

        # Mock restart method
        lifecycle_manager.restart_component = AsyncMock(return_value=True)

        # Start monitoring task
        task = asyncio.create_task(lifecycle_manager._dependency_monitor())

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify restart was attempted
        lifecycle_manager.restart_component.assert_called_with(ComponentType.RUST_DAEMON)

    @pytest.mark.asyncio
    async def test_startup_retry_logic(self, lifecycle_manager):
        """Test component startup retry logic."""
        config = lifecycle_manager.component_configs[ComponentType.RUST_DAEMON]
        config.max_startup_retries = 3

        # Mock start_component to fail twice, then succeed
        call_count = 0

        async def mock_start_component(component_type):
            nonlocal call_count
            call_count += 1
            return call_count >= 3  # Succeed on third attempt

        lifecycle_manager._start_component = mock_start_component

        result = await lifecycle_manager._start_component_with_retry(ComponentType.RUST_DAEMON)

        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_startup_retry_exhaustion(self, lifecycle_manager):
        """Test startup retry exhaustion."""
        config = lifecycle_manager.component_configs[ComponentType.RUST_DAEMON]
        config.max_startup_retries = 2

        # Mock start_component to always fail
        lifecycle_manager._start_component = AsyncMock(return_value=False)

        result = await lifecycle_manager._start_component_with_retry(ComponentType.RUST_DAEMON)

        assert result is False
        assert lifecycle_manager._start_component.call_count == 2

    @pytest.mark.asyncio
    async def test_custom_component_configs(self):
        """Test custom component configurations."""
        custom_config = ComponentConfig(
            component_type=ComponentType.RUST_DAEMON,
            startup_dependency=StartupDependency.RUST_DAEMON,
            startup_timeout=60.0,
            readiness_checks=["custom_check"]
        )

        custom_configs = {ComponentType.RUST_DAEMON: custom_config}

        manager = ComponentLifecycleManager(
            component_configs=custom_configs
        )

        rust_config = manager.component_configs[ComponentType.RUST_DAEMON]
        assert rust_config.startup_timeout == 60.0
        assert rust_config.readiness_checks == ["custom_check"]

    @pytest.mark.asyncio
    async def test_event_logging_queue_integration(self, lifecycle_manager):
        """Test lifecycle event logging with coordinator queue."""
        await lifecycle_manager._log_lifecycle_event(
            component_id="test_component",
            phase=LifecyclePhase.COMPONENT_STARTUP,
            event_type="startup",
            message="Test event"
        )

        # Verify event was enqueued to coordinator
        lifecycle_manager.coordinator.enqueue_processing_item.assert_called()

        call_args = lifecycle_manager.coordinator.enqueue_processing_item.call_args
        assert call_args[1]["component_id"] == "test_component"
        assert call_args[1]["queue_type"].value == "admin_commands"  # Handle enum value
        assert "event_data" in call_args[1]["payload"]

    def test_component_config_defaults(self):
        """Test ComponentConfig default values."""
        config = ComponentConfig(
            component_type=ComponentType.RUST_DAEMON,
            startup_dependency=StartupDependency.RUST_DAEMON
        )

        assert config.startup_timeout == 30.0
        assert config.shutdown_timeout == 15.0
        assert config.health_check_interval == 5.0
        assert config.max_startup_retries == 3
        assert config.readiness_checks == []
        assert config.config_overrides == {}
        assert config.environment_variables == {}

    def test_lifecycle_event_creation(self):
        """Test LifecycleEvent creation and defaults."""
        event = LifecycleEvent(
            event_id="test-123",
            component_id="test_component",
            phase=LifecyclePhase.COMPONENT_STARTUP,
            event_type="startup",
            message="Test message"
        )

        assert event.event_id == "test-123"
        assert event.component_id == "test_component"
        assert event.phase == LifecyclePhase.COMPONENT_STARTUP
        assert event.success is True
        assert event.details == {}
        assert isinstance(event.timestamp, datetime)

    def test_project_name_detection(self):
        """Test project name detection from current directory."""
        manager = ComponentLifecycleManager()

        # Should detect from current working directory
        expected_name = Path.cwd().name
        assert manager.project_name == expected_name


class TestGlobalLifecycleManager:
    """Test global lifecycle manager functions."""

    @pytest.mark.asyncio
    async def test_get_lifecycle_manager(self):
        """Test getting global lifecycle manager instance."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_get_coordinator.return_value = mock_coordinator

            # First call creates instance
            manager1 = await get_lifecycle_manager(
                db_path="test.db",
                project_name="test_project"
            )

            assert manager1 is not None
            assert manager1.project_name == "test_project"

            # Second call returns same instance
            manager2 = await get_lifecycle_manager()
            assert manager1 is manager2

            # Cleanup
            await shutdown_lifecycle_manager()

    @pytest.mark.asyncio
    async def test_shutdown_lifecycle_manager(self):
        """Test shutting down global lifecycle manager."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_get_coordinator.return_value = mock_coordinator

            # Create manager
            manager = await get_lifecycle_manager(db_path="test.db")
            manager.shutdown_sequence = AsyncMock(return_value=True)

            # Shutdown
            await shutdown_lifecycle_manager()

            # Verify shutdown was called
            manager.shutdown_sequence.assert_called_once()

            # Verify global instance is cleared
            from common.core.component_lifecycle import _lifecycle_manager
            assert _lifecycle_manager is None


@pytest.mark.integration
class TestComponentLifecycleIntegration:
    """Integration tests for component lifecycle management."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_sequence(self):
        """Test complete lifecycle sequence from startup to shutdown."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            # Mock coordinator
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_coordinator.register_component = AsyncMock(return_value="test-component-id")
            mock_coordinator.update_component_status = AsyncMock(return_value=True)
            mock_coordinator.update_component_health = AsyncMock(return_value=True)
            mock_coordinator.enqueue_processing_item = AsyncMock(return_value="queue-item-id")
            mock_coordinator.close = AsyncMock()
            mock_get_coordinator.return_value = mock_coordinator

            # Mock all external dependencies
            with patch('common.core.component_lifecycle.ensure_daemon_running') as mock_daemon, \
                 patch('common.core.component_lifecycle.GrpcWorkspaceClient') as mock_grpc, \
                 patch('common.core.component_lifecycle.DaemonManager'):

                # Configure mocks
                mock_daemon_instance = Mock()
                mock_daemon_instance.health_check = AsyncMock(return_value=True)
                mock_daemon_instance.status.state = "running"
                mock_daemon_instance.config.grpc_port = 50051
                mock_daemon_instance.shutdown = AsyncMock()
                mock_daemon.return_value = mock_daemon_instance

                mock_grpc_instance = AsyncMock()
                mock_grpc_instance.initialize = AsyncMock()
                mock_grpc_instance.get_operation_mode = Mock(return_value="grpc")
                mock_grpc_instance.get_status = AsyncMock(return_value={"qdrant_available": True})
                mock_grpc_instance.is_grpc_available = Mock(return_value=True)
                mock_grpc_instance.close = AsyncMock()
                mock_grpc.return_value = mock_grpc_instance

                # Create manager and test full lifecycle
                manager = ComponentLifecycleManager(
                    db_path="test.db",
                    project_name="integration_test"
                )

                # Initialize
                assert await manager.initialize() is True
                assert manager.current_phase == LifecyclePhase.INITIALIZATION

                # Startup sequence
                assert await manager.startup_sequence() is True
                assert manager.current_phase == LifecyclePhase.OPERATIONAL

                # Verify components are running
                status = await manager.get_component_status()
                assert "components" in status
                assert len(manager.component_instances) > 0

                # Shutdown sequence
                assert await manager.shutdown_sequence() is True
                assert manager.current_phase == LifecyclePhase.STOPPED

                # Verify cleanup
                assert len(manager.component_instances) == 0

    @pytest.mark.asyncio
    async def test_component_failure_recovery(self):
        """Test component failure detection and recovery."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            # Mock coordinator
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_coordinator.register_component = AsyncMock(return_value="test-component-id")
            mock_coordinator.update_component_status = AsyncMock(return_value=True)
            mock_coordinator.update_component_health = AsyncMock(return_value=True)
            mock_coordinator.close = AsyncMock()
            mock_get_coordinator.return_value = mock_coordinator

            manager = ComponentLifecycleManager(
                db_path="test.db",
                project_name="failure_test"
            )

            await manager.initialize()

            # Simulate component failure
            manager.component_states[ComponentType.RUST_DAEMON] = ComponentState.FAILED

            # Mock restart functionality
            manager.restart_component = AsyncMock(return_value=True)

            # Start dependency monitor briefly
            task = asyncio.create_task(manager._dependency_monitor())
            await asyncio.sleep(0.1)
            task.cancel()

            try:
                await task
            except asyncio.CancelledError:
                pass

            # Verify restart was attempted
            manager.restart_component.assert_called_with(ComponentType.RUST_DAEMON)

    @pytest.mark.asyncio
    async def test_startup_timeout_handling(self):
        """Test startup timeout handling."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_get_coordinator.return_value = mock_coordinator

            # Create manager with short timeout
            custom_config = ComponentConfig(
                component_type=ComponentType.RUST_DAEMON,
                startup_dependency=StartupDependency.RUST_DAEMON,
                startup_timeout=0.1  # Very short timeout
            )

            manager = ComponentLifecycleManager(
                db_path="test.db",
                project_name="timeout_test",
                component_configs={ComponentType.RUST_DAEMON: custom_config}
            )

            await manager.initialize()

            # Mock slow daemon startup
            with patch('common.core.component_lifecycle.ensure_daemon_running') as mock_daemon:
                mock_daemon_instance = Mock()
                mock_daemon_instance.health_check = AsyncMock(side_effect=asyncio.sleep(1))  # Slow health check
                mock_daemon.return_value = mock_daemon_instance

                # Test that timeout is respected
                result = await manager._start_rust_daemon(custom_config)
                assert result is None  # Should timeout and return None

    @pytest.mark.asyncio
    async def test_event_logging_integration(self):
        """Test comprehensive event logging throughout lifecycle."""
        with patch('common.core.component_lifecycle.get_component_coordinator') as mock_get_coordinator:
            mock_coordinator = AsyncMock()
            mock_coordinator.initialize = AsyncMock(return_value=True)
            mock_coordinator.register_component = AsyncMock(return_value="test-component-id")
            mock_coordinator.enqueue_processing_item = AsyncMock(return_value="queue-item-id")
            mock_coordinator.close = AsyncMock()
            mock_get_coordinator.return_value = mock_coordinator

            manager = ComponentLifecycleManager(
                db_path="test.db",
                project_name="logging_test"
            )

            await manager.initialize()

            # Log various events
            await manager._log_lifecycle_event(
                component_id="test_component",
                phase=LifecyclePhase.INITIALIZATION,
                event_type="initialization",
                message="Test initialization"
            )

            await manager._log_lifecycle_event(
                component_id="test_component",
                phase=LifecyclePhase.COMPONENT_STARTUP,
                event_type="startup",
                message="Test startup",
                success=False,
                error_message="Test error"
            )

            # Verify events were logged
            assert len(manager.startup_events) == 3  # 2 test events + 1 from initialization

            # Verify events have correct structure
            for event in manager.startup_events:
                assert hasattr(event, 'event_id')
                assert hasattr(event, 'timestamp')
                assert isinstance(event.timestamp, datetime)

            # Verify coordinator integration
            assert mock_coordinator.enqueue_processing_item.call_count >= 2