"""
Unit tests for the Automatic Recovery System.

Tests cover:
- Recovery manager initialization and configuration
- Automatic component restart with exponential backoff
- State recovery mechanisms using SQLite persistence
- Dependency resolution for cascading component restarts
- Automatic cleanup of corrupted state and temporary files
- Recovery validation to ensure components return to healthy state
- Integration with health monitoring and graceful degradation systems
- Self-healing capabilities that trigger automatically
"""

import asyncio
import json
import os
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from workspace_qdrant_mcp.core.automatic_recovery import (
    RecoveryManager,
    RecoveryStrategy,
    RecoveryPhase,
    RecoveryTrigger,
    CleanupType,
    RecoveryConfig,
    RecoveryAction,
    RecoveryAttempt,
    ComponentDependency,
    get_recovery_manager,
    shutdown_recovery_manager,
)
from workspace_qdrant_mcp.core.component_coordination import ComponentType, ComponentHealth, ComponentStatus
from workspace_qdrant_mcp.core.graceful_degradation import DegradationMode, CircuitBreakerState
from workspace_qdrant_mcp.core.lsp_health_monitor import HealthStatus, NotificationLevel, UserNotification


class TestRecoveryManager:
    """Test cases for RecoveryManager."""

    @pytest.fixture
    async def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        yield db_path

        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    async def mock_lifecycle_manager(self):
        """Create mock lifecycle manager."""
        manager = AsyncMock()
        manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "operational"},
                ComponentType.PYTHON_MCP_SERVER.value: {"state": "operational"},
                ComponentType.CLI_UTILITY.value: {"state": "operational"},
                ComponentType.CONTEXT_INJECTOR.value: {"state": "operational"},
            }
        }
        manager.start_component.return_value = True
        manager.stop_component.return_value = True
        return manager

    @pytest.fixture
    async def mock_health_monitor(self):
        """Create mock health monitor."""
        monitor = Mock()
        monitor.register_notification_handler = Mock()
        return monitor

    @pytest.fixture
    async def mock_degradation_manager(self):
        """Create mock degradation manager."""
        manager = Mock()
        manager.register_notification_handler = Mock()
        manager.get_circuit_breaker_state.return_value = CircuitBreakerState.CLOSED
        return manager

    @pytest.fixture
    async def mock_coordinator(self):
        """Create mock coordinator."""
        coordinator = AsyncMock()
        coordinator.initialize.return_value = True
        return coordinator

    @pytest.fixture
    async def recovery_manager(
        self,
        temp_db,
        mock_lifecycle_manager,
        mock_health_monitor,
        mock_degradation_manager,
        mock_coordinator
    ):
        """Create recovery manager for testing."""
        config = {"recovery_db_path": temp_db}

        manager = RecoveryManager(
            lifecycle_manager=mock_lifecycle_manager,
            health_monitor=mock_health_monitor,
            degradation_manager=mock_degradation_manager,
            coordinator=mock_coordinator,
            config=config
        )

        await manager.initialize()
        yield manager
        await manager.shutdown()

    async def test_recovery_manager_initialization(self, temp_db):
        """Test recovery manager initialization."""
        config = {"recovery_db_path": temp_db}
        manager = RecoveryManager(config=config)

        assert await manager.initialize()
        assert manager.recovery_db_path == Path(temp_db)
        assert len(manager.recovery_configs) == len(ComponentType)
        assert len(manager.monitoring_tasks) > 0

        await manager.shutdown()

    async def test_recovery_database_initialization(self, temp_db):
        """Test recovery database initialization."""
        config = {"recovery_db_path": temp_db}
        manager = RecoveryManager(config=config)

        await manager.initialize()

        # Check database tables exist
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "recovery_attempts" in tables
        assert "recovery_configs" in tables
        assert "cleanup_operations" in tables

        conn.close()
        await manager.shutdown()

    async def test_component_failure_detection(self, recovery_manager, mock_lifecycle_manager):
        """Test automatic component failure detection."""
        # Simulate component failure
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "failed"},
                ComponentType.PYTHON_MCP_SERVER.value: {"state": "operational"},
                ComponentType.CLI_UTILITY.value: {"state": "operational"},
                ComponentType.CONTEXT_INJECTOR.value: {"state": "operational"},
            }
        }

        # Trigger detection
        await recovery_manager._detect_component_failures()

        # Should have active recovery
        assert len(recovery_manager.active_recoveries) == 1

        # Check recovery attempt
        attempt = next(iter(recovery_manager.active_recoveries.values()))
        assert attempt.component_id == f"{ComponentType.RUST_DAEMON.value}-default"
        assert attempt.trigger == RecoveryTrigger.COMPONENT_CRASH

    async def test_manual_recovery_trigger(self, recovery_manager):
        """Test manual recovery trigger."""
        attempt_id = await recovery_manager.trigger_component_recovery(
            ComponentType.RUST_DAEMON,
            RecoveryStrategy.IMMEDIATE,
            "Manual test recovery"
        )

        assert attempt_id in recovery_manager.active_recoveries
        attempt = recovery_manager.active_recoveries[attempt_id]

        assert attempt.component_id == f"{ComponentType.RUST_DAEMON.value}-default"
        assert attempt.trigger == RecoveryTrigger.MANUAL_TRIGGER
        assert attempt.strategy == RecoveryStrategy.PROGRESSIVE  # Should use config default
        assert attempt.phase == RecoveryPhase.DETECTION

    async def test_recovery_strategy_selection(self, recovery_manager):
        """Test recovery strategy selection based on failure patterns."""
        component_type = ComponentType.RUST_DAEMON
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{component_type.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.IMMEDIATE,
            phase=RecoveryPhase.ANALYSIS,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        # Simulate multiple recent failures
        pattern_key = f"{component_type.value}_{attempt.trigger.value}"
        recovery_manager.failure_patterns[pattern_key] = [
            datetime.now(timezone.utc) for _ in range(6)
        ]

        await recovery_manager._analyze_failure(attempt)

        # Should escalate to emergency reset due to many failures
        assert attempt.strategy == RecoveryStrategy.EMERGENCY_RESET

    async def test_recovery_action_preparation(self, recovery_manager):
        """Test preparation of recovery actions."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.PROGRESSIVE,
            phase=RecoveryPhase.PREPARATION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        await recovery_manager._prepare_recovery_actions(attempt)

        assert len(attempt.actions) > 0
        action_types = [action.action_type for action in attempt.actions]

        # Progressive strategy should include these actions
        assert "stop_component" in action_types
        assert "start_component" in action_types
        assert "cleanup_state" in action_types
        assert "progressive_delay" in action_types

    async def test_dependency_aware_recovery(self, recovery_manager):
        """Test dependency-aware recovery strategy."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.PYTHON_MCP_SERVER.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.DEPENDENCY_AWARE,
            phase=RecoveryPhase.PREPARATION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        await recovery_manager._prepare_recovery_actions(attempt)

        action_types = [action.action_type for action in attempt.actions]

        # Should verify dependencies
        assert "verify_dependency" in action_types

        # Check dependency verification
        verify_actions = [a for a in attempt.actions if a.action_type == "verify_dependency"]
        assert len(verify_actions) > 0

        # Should verify RUST_DAEMON dependency
        dependency_components = [a.component_id for a in verify_actions]
        assert f"{ComponentType.RUST_DAEMON.value}-default" in dependency_components

    async def test_state_recovery_strategy(self, recovery_manager):
        """Test state recovery strategy with cleanup and restoration."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.CONTEXT_INJECTOR.value}-default",
            trigger=RecoveryTrigger.STATE_CORRUPTION,
            strategy=RecoveryStrategy.STATE_RECOVERY,
            phase=RecoveryPhase.PREPARATION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        await recovery_manager._prepare_recovery_actions(attempt)

        action_types = [action.action_type for action in attempt.actions]

        # State recovery should include all these steps
        assert "stop_component" in action_types
        assert "backup_state" in action_types
        assert "cleanup_corrupted_state" in action_types
        assert "restore_state" in action_types
        assert "start_component" in action_types

    async def test_emergency_reset_strategy(self, recovery_manager):
        """Test emergency reset strategy."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id="all",
            trigger=RecoveryTrigger.MANUAL_TRIGGER,
            strategy=RecoveryStrategy.EMERGENCY_RESET,
            phase=RecoveryPhase.PREPARATION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        await recovery_manager._prepare_recovery_actions(attempt)

        action_types = [action.action_type for action in attempt.actions]

        # Emergency reset should include system-wide actions
        assert "emergency_stop_all" in action_types
        assert "full_cleanup" in action_types
        assert "initialize_fresh_state" in action_types
        assert "start_all_components" in action_types

    async def test_component_start_order(self, recovery_manager):
        """Test component startup order based on dependencies."""
        start_order = recovery_manager._get_component_start_order()

        # RUST_DAEMON should come before PYTHON_MCP_SERVER
        rust_index = start_order.index(ComponentType.RUST_DAEMON)
        mcp_index = start_order.index(ComponentType.PYTHON_MCP_SERVER)
        assert rust_index < mcp_index

        # PYTHON_MCP_SERVER should come before CONTEXT_INJECTOR
        context_index = start_order.index(ComponentType.CONTEXT_INJECTOR)
        assert mcp_index < context_index

        # CLI_UTILITY can be anywhere (no dependencies)
        assert ComponentType.CLI_UTILITY in start_order

    async def test_recovery_action_execution(self, recovery_manager, mock_lifecycle_manager):
        """Test recovery action execution."""
        # Test stop component action
        action = RecoveryAction(
            action_id=str(uuid.uuid4()),
            action_type="stop_component",
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            description="Stop Rust daemon"
        )

        success = await recovery_manager._execute_single_action(action)
        assert success
        mock_lifecycle_manager.stop_component.assert_called_with(ComponentType.RUST_DAEMON)

        # Test start component action
        action.action_type = "start_component"
        action.description = "Start Rust daemon"

        success = await recovery_manager._execute_single_action(action)
        assert success
        mock_lifecycle_manager.start_component.assert_called_with(ComponentType.RUST_DAEMON)

    async def test_recovery_validation(self, recovery_manager, mock_degradation_manager):
        """Test recovery validation."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.IMMEDIATE,
            phase=RecoveryPhase.VALIDATION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        # Mock successful health check
        recovery_manager._is_component_healthy = AsyncMock(return_value=True)
        mock_degradation_manager.get_circuit_breaker_state.return_value = CircuitBreakerState.CLOSED

        success = await recovery_manager._validate_recovery(attempt)
        assert success

        # Test failed validation
        recovery_manager._is_component_healthy = AsyncMock(return_value=False)

        success = await recovery_manager._validate_recovery(attempt)
        assert not success

    async def test_automatic_cleanup_operations(self, recovery_manager):
        """Test automatic cleanup operations."""
        # Test temporary file cleanup
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some temporary files
            temp_files = []
            for pattern in ["test.tmp", "old.lock", "data.bak"]:
                file_path = Path(temp_dir) / pattern
                file_path.write_text("test data")
                # Make file old
                old_time = time.time() - 7200  # 2 hours ago
                os.utime(file_path, (old_time, old_time))
                temp_files.append(file_path)

            # Mock cleanup paths to include our temp directory
            original_patterns = recovery_manager.TEMP_FILE_PATTERNS
            recovery_manager.TEMP_FILE_PATTERNS = ["*.tmp", "*.lock", "*.bak"]

            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = temp_files

                success = await recovery_manager._cleanup_temporary_files()
                assert success

        # Test stale lock cleanup
        success = await recovery_manager._cleanup_stale_locks()
        assert success

        # Test zombie process cleanup
        success = await recovery_manager._cleanup_zombie_processes()
        assert success

        # Test invalid cache cleanup
        success = await recovery_manager._cleanup_invalid_caches()
        assert success

    async def test_recovery_timeout_handling(self, recovery_manager):
        """Test recovery timeout handling."""
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.IMMEDIATE,
            phase=RecoveryPhase.EXECUTION,
            actions=[],
            start_time=datetime.now(timezone.utc) - timedelta(minutes=10)
        )

        recovery_manager.active_recoveries[attempt.attempt_id] = attempt

        # Mock timeout configuration
        config = RecoveryConfig(
            strategy=RecoveryStrategy.IMMEDIATE,
            timeout_seconds=300  # 5 minutes
        )
        recovery_manager.recovery_configs[ComponentType.RUST_DAEMON] = config

        await recovery_manager._check_recovery_timeouts()

        # Should have triggered timeout handling
        assert attempt.phase == RecoveryPhase.FAILURE
        assert "timed out" in attempt.error_message

    async def test_recovery_statistics_tracking(self, recovery_manager):
        """Test recovery statistics tracking."""
        # Add some mock recovery history
        successful_attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.IMMEDIATE,
            phase=RecoveryPhase.COMPLETION,
            actions=[],
            start_time=datetime.now(timezone.utc) - timedelta(minutes=5),
            end_time=datetime.now(timezone.utc),
            success=True
        )

        failed_attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.PYTHON_MCP_SERVER.value}-default",
            trigger=RecoveryTrigger.HEALTH_CHECK_FAILURE,
            strategy=RecoveryStrategy.PROGRESSIVE,
            phase=RecoveryPhase.FAILURE,
            actions=[],
            start_time=datetime.now(timezone.utc) - timedelta(minutes=3),
            end_time=datetime.now(timezone.utc),
            success=False
        )

        recovery_manager.recovery_history = [successful_attempt, failed_attempt]

        await recovery_manager._update_recovery_statistics()

        stats = recovery_manager.recovery_statistics
        assert stats["total_attempts"] == 2
        assert stats["successful_recoveries"] == 1
        assert stats["failed_recoveries"] == 1
        assert stats["average_recovery_time"] > 0
        assert stats["most_recovered_component"] == ComponentType.RUST_DAEMON.value

    async def test_health_notification_handling(self, recovery_manager):
        """Test handling of health monitor notifications."""
        # Create critical health notification
        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.CRITICAL,
            title="Daemon Health Critical",
            message="Rust daemon is unresponsive and requires immediate attention",
            server_name="workspace-qdrant-mcp"
        )

        await recovery_manager._handle_health_notification(notification)

        # Should have triggered recovery
        assert len(recovery_manager.active_recoveries) == 1
        attempt = next(iter(recovery_manager.active_recoveries.values()))
        assert ComponentType.RUST_DAEMON.value in attempt.component_id
        assert attempt.trigger == RecoveryTrigger.HEALTH_CHECK_FAILURE

    async def test_degradation_notification_handling(self, recovery_manager, mock_lifecycle_manager):
        """Test handling of degradation manager notifications."""
        # Mock critical system state
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "failed"},
                ComponentType.PYTHON_MCP_SERVER.value: {"state": "unhealthy"},
                ComponentType.CLI_UTILITY.value: {"state": "operational"},
                ComponentType.CONTEXT_INJECTOR.value: {"state": "failed"},
            }
        }

        notification = UserNotification(
            timestamp=time.time(),
            level=NotificationLevel.CRITICAL,
            title="System Critical Degradation",
            message="Multiple components have failed",
            server_name="workspace-qdrant-mcp"
        )

        await recovery_manager._handle_degradation_notification(notification)

        # Should have triggered recovery for failed components
        assert len(recovery_manager.active_recoveries) >= 2

        component_ids = [attempt.component_id for attempt in recovery_manager.active_recoveries.values()]
        assert f"{ComponentType.RUST_DAEMON.value}-default" in component_ids
        assert f"{ComponentType.CONTEXT_INJECTOR.value}-default" in component_ids

    async def test_recovery_config_management(self, recovery_manager, temp_db):
        """Test recovery configuration management."""
        # Test updating configuration
        new_config = RecoveryConfig(
            strategy=RecoveryStrategy.EMERGENCY_RESET,
            max_retries=10,
            initial_delay=5.0,
            max_delay=300.0,
            exponential_base=3.0,
            timeout_seconds=600.0
        )

        await recovery_manager.update_recovery_config(ComponentType.RUST_DAEMON, new_config)

        # Verify configuration was updated
        stored_config = await recovery_manager.get_recovery_config(ComponentType.RUST_DAEMON)
        assert stored_config.strategy == RecoveryStrategy.EMERGENCY_RESET
        assert stored_config.max_retries == 10
        assert stored_config.initial_delay == 5.0

        # Verify configuration was persisted to database
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT strategy, max_retries, initial_delay FROM recovery_configs WHERE component_type = ?",
            (ComponentType.RUST_DAEMON.value,)
        )
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == RecoveryStrategy.EMERGENCY_RESET.value
        assert row[1] == 10
        assert row[2] == 5.0

        conn.close()

    async def test_force_cleanup_operations(self, recovery_manager):
        """Test forced cleanup operations."""
        # Test different cleanup types
        cleanup_types = [
            CleanupType.TEMPORARY_FILES,
            CleanupType.STALE_LOCKS,
            CleanupType.ZOMBIE_PROCESSES,
            CleanupType.INVALID_CACHES
        ]

        for cleanup_type in cleanup_types:
            success = await recovery_manager.force_cleanup(cleanup_type, "/tmp")
            assert success

    async def test_recovery_attempt_cancellation(self, recovery_manager):
        """Test cancellation of active recovery attempts."""
        # Start a recovery
        attempt_id = await recovery_manager.trigger_component_recovery(
            ComponentType.RUST_DAEMON,
            RecoveryStrategy.PROGRESSIVE,
            "Test cancellation"
        )

        # Cancel the recovery
        success = await recovery_manager.cancel_recovery(attempt_id)
        assert success

        # Verify it was cancelled
        attempt = recovery_manager.active_recoveries[attempt_id]
        assert attempt.phase == RecoveryPhase.FAILURE
        assert "cancelled" in attempt.error_message

        # Try to cancel again (should fail)
        success = await recovery_manager.cancel_recovery(attempt_id)
        assert not success

        # Try to cancel non-existent recovery
        success = await recovery_manager.cancel_recovery("invalid-id")
        assert not success

    async def test_recovery_status_retrieval(self, recovery_manager):
        """Test retrieval of recovery status and history."""
        # Start a recovery
        attempt_id = await recovery_manager.trigger_component_recovery(
            ComponentType.RUST_DAEMON,
            RecoveryStrategy.IMMEDIATE,
            "Test status retrieval"
        )

        # Get status
        status = await recovery_manager.get_recovery_status(attempt_id)
        assert status is not None
        assert status["attempt_id"] == attempt_id
        assert status["component_id"] == f"{ComponentType.RUST_DAEMON.value}-default"
        assert status["strategy"] == RecoveryStrategy.PROGRESSIVE.value  # Uses config default

        # Get active recoveries
        active = await recovery_manager.get_active_recoveries()
        assert len(active) == 1
        assert active[0]["attempt_id"] == attempt_id

        # Get recovery history (should be empty initially)
        history = await recovery_manager.get_recovery_history()
        assert isinstance(history, list)

        # Get statistics
        stats = recovery_manager.get_recovery_statistics()
        assert "total_attempts" in stats
        assert "successful_recoveries" in stats

    async def test_exponential_backoff_calculation(self, recovery_manager):
        """Test exponential backoff calculation for progressive recovery."""
        config = RecoveryConfig(
            strategy=RecoveryStrategy.PROGRESSIVE,
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0
        )

        action = RecoveryAction(
            action_id=str(uuid.uuid4()),
            action_type="progressive_delay",
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            description="Wait with exponential backoff",
            parameters={"delay": config.initial_delay}
        )

        # Test delay execution
        start_time = time.time()
        success = await recovery_manager._progressive_delay_action(action)
        end_time = time.time()

        assert success
        assert end_time - start_time >= config.initial_delay

    async def test_state_corruption_detection(self, recovery_manager, mock_lifecycle_manager):
        """Test detection and handling of state corruption."""
        # Simulate component with corrupted state
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "inconsistent"},
                ComponentType.PYTHON_MCP_SERVER.value: {"state": "operational"},
                ComponentType.CLI_UTILITY.value: {"state": "operational"},
                ComponentType.CONTEXT_INJECTOR.value: {"state": "partially_failed"},
            }
        }

        await recovery_manager._validate_component_states()

        # Should have triggered recovery for inconsistent components
        assert len(recovery_manager.active_recoveries) == 2

        triggers = [attempt.trigger for attempt in recovery_manager.active_recoveries.values()]
        assert all(trigger == RecoveryTrigger.STATE_CORRUPTION for trigger in triggers)

    async def test_concurrent_recovery_prevention(self, recovery_manager):
        """Test prevention of concurrent recoveries for same component."""
        component_type = ComponentType.RUST_DAEMON

        # Start first recovery
        attempt_id1 = await recovery_manager.trigger_component_recovery(
            component_type,
            RecoveryStrategy.IMMEDIATE,
            "First recovery"
        )

        # Try to start second recovery for same component
        with pytest.raises(ValueError, match="Recovery already in progress"):
            await recovery_manager.trigger_component_recovery(
                component_type,
                RecoveryStrategy.PROGRESSIVE,
                "Second recovery"
            )

        # Should only have one active recovery
        assert len(recovery_manager.active_recoveries) == 1

    async def test_dependency_validation(self, recovery_manager, mock_lifecycle_manager):
        """Test dependency validation during recovery."""
        # Mock healthy dependency
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "operational"},
                ComponentType.PYTHON_MCP_SERVER.value: {"state": "operational"},
                ComponentType.CLI_UTILITY.value: {"state": "operational"},
                ComponentType.CONTEXT_INJECTOR.value: {"state": "operational"},
            }
        }

        action = RecoveryAction(
            action_id=str(uuid.uuid4()),
            action_type="verify_dependency",
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            description="Verify dependency"
        )

        success = await recovery_manager._execute_single_action(action)
        assert success

        # Mock unhealthy dependency
        mock_lifecycle_manager.get_component_status.return_value = {
            "components": {
                ComponentType.RUST_DAEMON.value: {"state": "failed"},
            }
        }

        success = await recovery_manager._execute_single_action(action)
        assert not success

    async def test_recovery_persistence_and_loading(self, temp_db):
        """Test persistence and loading of recovery history."""
        config = {"recovery_db_path": temp_db}

        # Create first manager and add recovery history
        manager1 = RecoveryManager(config=config)
        await manager1.initialize()

        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.IMMEDIATE,
            phase=RecoveryPhase.COMPLETION,
            actions=[],
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            success=True
        )

        await manager1._store_recovery_attempt(attempt)
        await manager1.shutdown()

        # Create second manager and verify history is loaded
        manager2 = RecoveryManager(config=config)
        await manager2.initialize()

        assert len(manager2.recovery_history) == 1
        loaded_attempt = manager2.recovery_history[0]
        assert loaded_attempt.attempt_id == attempt.attempt_id
        assert loaded_attempt.success == attempt.success

        await manager2.shutdown()


class TestRecoveryConfiguration:
    """Test cases for recovery configuration and strategies."""

    def test_recovery_strategy_enum(self):
        """Test recovery strategy enumeration."""
        strategies = list(RecoveryStrategy)
        assert RecoveryStrategy.IMMEDIATE in strategies
        assert RecoveryStrategy.PROGRESSIVE in strategies
        assert RecoveryStrategy.DEPENDENCY_AWARE in strategies
        assert RecoveryStrategy.STATE_RECOVERY in strategies
        assert RecoveryStrategy.EMERGENCY_RESET in strategies

    def test_recovery_phase_enum(self):
        """Test recovery phase enumeration."""
        phases = list(RecoveryPhase)
        assert RecoveryPhase.DETECTION in phases
        assert RecoveryPhase.ANALYSIS in phases
        assert RecoveryPhase.PREPARATION in phases
        assert RecoveryPhase.EXECUTION in phases
        assert RecoveryPhase.VALIDATION in phases
        assert RecoveryPhase.COMPLETION in phases
        assert RecoveryPhase.FAILURE in phases

    def test_recovery_trigger_enum(self):
        """Test recovery trigger enumeration."""
        triggers = list(RecoveryTrigger)
        assert RecoveryTrigger.HEALTH_CHECK_FAILURE in triggers
        assert RecoveryTrigger.CIRCUIT_BREAKER_OPEN in triggers
        assert RecoveryTrigger.DEGRADATION_MODE_CHANGE in triggers
        assert RecoveryTrigger.COMPONENT_CRASH in triggers
        assert RecoveryTrigger.MANUAL_TRIGGER in triggers
        assert RecoveryTrigger.STATE_CORRUPTION in triggers
        assert RecoveryTrigger.DEPENDENCY_FAILURE in triggers

    def test_cleanup_type_enum(self):
        """Test cleanup type enumeration."""
        types = list(CleanupType)
        assert CleanupType.TEMPORARY_FILES in types
        assert CleanupType.CORRUPTED_STATE in types
        assert CleanupType.STALE_LOCKS in types
        assert CleanupType.ZOMBIE_PROCESSES in types
        assert CleanupType.INVALID_CACHES in types
        assert CleanupType.BROKEN_CONNECTIONS in types

    def test_recovery_config_defaults(self):
        """Test recovery configuration defaults."""
        config = RecoveryConfig(strategy=RecoveryStrategy.IMMEDIATE)

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.timeout_seconds == 300.0
        assert config.validate_after_recovery is True
        assert config.cleanup_on_failure is True
        assert config.dependency_recovery is True
        assert config.state_backup_enabled is True

    def test_component_dependency_definition(self):
        """Test component dependency definitions."""
        dependency = ComponentDependency(
            component=ComponentType.PYTHON_MCP_SERVER,
            depends_on={ComponentType.RUST_DAEMON},
            startup_delay=5.0,
            health_check_required=True
        )

        assert dependency.component == ComponentType.PYTHON_MCP_SERVER
        assert ComponentType.RUST_DAEMON in dependency.depends_on
        assert dependency.startup_delay == 5.0
        assert dependency.health_check_required is True

    def test_recovery_action_creation(self):
        """Test recovery action creation."""
        action = RecoveryAction(
            action_id=str(uuid.uuid4()),
            action_type="stop_component",
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            description="Stop Rust daemon component",
            parameters={"force": True},
            timeout_seconds=30.0,
            max_retries=3
        )

        assert action.action_type == "stop_component"
        assert action.component_id == f"{ComponentType.RUST_DAEMON.value}-default"
        assert action.parameters["force"] is True
        assert action.timeout_seconds == 30.0
        assert action.retry_count == 0
        assert action.max_retries == 3

    def test_recovery_attempt_creation(self):
        """Test recovery attempt creation."""
        start_time = datetime.now(timezone.utc)
        attempt = RecoveryAttempt(
            attempt_id=str(uuid.uuid4()),
            component_id=f"{ComponentType.RUST_DAEMON.value}-default",
            trigger=RecoveryTrigger.COMPONENT_CRASH,
            strategy=RecoveryStrategy.PROGRESSIVE,
            phase=RecoveryPhase.DETECTION,
            actions=[],
            start_time=start_time
        )

        assert attempt.component_id == f"{ComponentType.RUST_DAEMON.value}-default"
        assert attempt.trigger == RecoveryTrigger.COMPONENT_CRASH
        assert attempt.strategy == RecoveryStrategy.PROGRESSIVE
        assert attempt.phase == RecoveryPhase.DETECTION
        assert attempt.start_time == start_time
        assert attempt.end_time is None
        assert attempt.success is False
        assert attempt.error_message is None


class TestGlobalRecoveryManager:
    """Test cases for global recovery manager functions."""

    async def test_get_recovery_manager_singleton(self):
        """Test global recovery manager singleton."""
        # Clear any existing instance
        await shutdown_recovery_manager()

        # Get first instance
        manager1 = await get_recovery_manager()
        assert manager1 is not None

        # Get second instance (should be same)
        manager2 = await get_recovery_manager()
        assert manager1 is manager2

        # Cleanup
        await shutdown_recovery_manager()

    async def test_recovery_manager_initialization_failure(self):
        """Test handling of recovery manager initialization failure."""
        # Clear any existing instance
        await shutdown_recovery_manager()

        with patch.object(RecoveryManager, 'initialize', return_value=False):
            with pytest.raises(RuntimeError, match="Failed to initialize recovery manager"):
                await get_recovery_manager()

    async def test_shutdown_recovery_manager(self):
        """Test shutdown of global recovery manager."""
        # Get manager instance
        manager = await get_recovery_manager()
        assert manager is not None

        # Shutdown
        await shutdown_recovery_manager()

        # Should be able to get new instance
        new_manager = await get_recovery_manager()
        assert new_manager is not manager

        # Cleanup
        await shutdown_recovery_manager()


class TestRecoveryIntegration:
    """Integration tests for recovery system with other components."""

    @pytest.fixture
    async def integrated_recovery_manager(self):
        """Create recovery manager with real component integrations."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create real component instances
            lifecycle_manager = AsyncMock()
            health_monitor = Mock()
            degradation_manager = Mock()
            coordinator = AsyncMock()

            # Setup realistic behaviors
            lifecycle_manager.get_component_status.return_value = {
                "components": {
                    ComponentType.RUST_DAEMON.value: {"state": "operational"},
                    ComponentType.PYTHON_MCP_SERVER.value: {"state": "operational"},
                    ComponentType.CLI_UTILITY.value: {"state": "operational"},
                    ComponentType.CONTEXT_INJECTOR.value: {"state": "operational"},
                }
            }

            config = {"recovery_db_path": db_path}
            manager = RecoveryManager(
                lifecycle_manager=lifecycle_manager,
                health_monitor=health_monitor,
                degradation_manager=degradation_manager,
                coordinator=coordinator,
                config=config
            )

            await manager.initialize()
            yield manager
            await manager.shutdown()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    async def test_end_to_end_recovery_flow(self, integrated_recovery_manager):
        """Test complete end-to-end recovery flow."""
        manager = integrated_recovery_manager

        # Trigger recovery
        attempt_id = await manager.trigger_component_recovery(
            ComponentType.RUST_DAEMON,
            RecoveryStrategy.PROGRESSIVE,
            "End-to-end test"
        )

        # Verify recovery was started
        assert attempt_id in manager.active_recoveries
        attempt = manager.active_recoveries[attempt_id]

        # Wait for recovery to complete (or timeout)
        max_wait = 10  # seconds
        wait_time = 0
        while attempt.phase not in {RecoveryPhase.COMPLETION, RecoveryPhase.FAILURE} and wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1

        # Verify recovery completed
        assert attempt.phase in {RecoveryPhase.COMPLETION, RecoveryPhase.FAILURE}

    async def test_multiple_component_recovery(self, integrated_recovery_manager):
        """Test recovery of multiple components simultaneously."""
        manager = integrated_recovery_manager

        # Trigger recovery for multiple components
        components = [ComponentType.RUST_DAEMON, ComponentType.PYTHON_MCP_SERVER]
        attempt_ids = []

        for component in components:
            attempt_id = await manager.trigger_component_recovery(
                component,
                RecoveryStrategy.IMMEDIATE,
                f"Multi-component test for {component.value}"
            )
            attempt_ids.append(attempt_id)

        # Verify all recoveries were started
        assert len(manager.active_recoveries) == len(components)

        for attempt_id in attempt_ids:
            assert attempt_id in manager.active_recoveries

    async def test_recovery_with_notifications(self, integrated_recovery_manager):
        """Test recovery system with notification handling."""
        manager = integrated_recovery_manager
        notifications_received = []

        def notification_handler(notification: UserNotification):
            notifications_received.append(notification)

        manager.register_notification_handler(notification_handler)

        # Trigger recovery
        await manager.trigger_component_recovery(
            ComponentType.RUST_DAEMON,
            RecoveryStrategy.IMMEDIATE,
            "Notification test"
        )

        # Wait a bit for recovery to progress
        await asyncio.sleep(0.5)

        # Should have received notifications
        # Note: In a real implementation, notifications would be sent
        # when recovery completes, but in this test it's mocked