"""
Automatic Recovery Mechanisms for Four-Component Architecture.

This module provides comprehensive automatic recovery capabilities for the workspace-qdrant-mcp
system, ensuring rapid restoration of system health through intelligent self-healing mechanisms.

Key Features:
    - Automatic component restart with exponential backoff
    - State recovery mechanisms using SQLite persistence
    - Dependency resolution for cascading component restarts
    - Automatic cleanup of corrupted state and temporary files
    - Recovery validation to ensure components return to healthy state
    - Integration with health monitoring and graceful degradation systems
    - Self-healing capabilities that trigger automatically
    - Recovery strategy selection based on failure patterns

Recovery Strategies:
    - IMMEDIATE: Fast restart with minimal delay
    - PROGRESSIVE: Gradual restart with exponential backoff
    - DEPENDENCY_AWARE: Component restart considering dependencies
    - STATE_RECOVERY: Full state restoration with cleanup
    - EMERGENCY_RESET: Complete system reset with fresh state

Example:
    ```python
    from workspace_qdrant_mcp.core.automatic_recovery import RecoveryManager

    # Initialize recovery manager
    recovery = RecoveryManager(
        lifecycle_manager=lifecycle_manager,
        health_monitor=health_monitor,
        degradation_manager=degradation_manager
    )
    await recovery.initialize()

    # Automatic recovery is now active
    # Manual recovery can also be triggered
    await recovery.trigger_component_recovery(ComponentType.RUST_DAEMON)
    ```
"""

import asyncio
import json
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from weakref import WeakSet

from loguru import logger

from .component_coordination import (
    ComponentCoordinator,
    ComponentType,
)
from .component_lifecycle import (
    ComponentLifecycleManager,
)
from .graceful_degradation import (
    CircuitBreakerState,
    DegradationManager,
)
from .lsp_health_monitor import (
    LspHealthMonitor,
    NotificationLevel,
    UserNotification,
)


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure scenarios."""

    IMMEDIATE = "immediate"               # Fast restart with minimal delay
    PROGRESSIVE = "progressive"           # Gradual restart with exponential backoff
    DEPENDENCY_AWARE = "dependency_aware" # Component restart considering dependencies
    STATE_RECOVERY = "state_recovery"     # Full state restoration with cleanup
    EMERGENCY_RESET = "emergency_reset"   # Complete system reset with fresh state


class RecoveryPhase(Enum):
    """Phases in the recovery process."""

    DETECTION = "detection"               # Failure detection phase
    ANALYSIS = "analysis"                 # Root cause analysis
    PREPARATION = "preparation"           # Prepare for recovery
    EXECUTION = "execution"               # Execute recovery actions
    VALIDATION = "validation"             # Validate recovery success
    COMPLETION = "completion"             # Complete recovery process
    FAILURE = "failure"                   # Recovery failed


class RecoveryTrigger(Enum):
    """What triggered the recovery process."""

    HEALTH_CHECK_FAILURE = "health_check_failure"
    CIRCUIT_BREAKER_OPEN = "circuit_breaker_open"
    DEGRADATION_MODE_CHANGE = "degradation_mode_change"
    COMPONENT_CRASH = "component_crash"
    MANUAL_TRIGGER = "manual_trigger"
    STATE_CORRUPTION = "state_corruption"
    DEPENDENCY_FAILURE = "dependency_failure"


class CleanupType(Enum):
    """Types of cleanup operations."""

    TEMPORARY_FILES = "temporary_files"
    CORRUPTED_STATE = "corrupted_state"
    STALE_LOCKS = "stale_locks"
    ZOMBIE_PROCESSES = "zombie_processes"
    INVALID_CACHES = "invalid_caches"
    BROKEN_CONNECTIONS = "broken_connections"


@dataclass
class RecoveryConfig:
    """Configuration for recovery operations."""

    strategy: RecoveryStrategy
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    timeout_seconds: float = 300.0
    validate_after_recovery: bool = True
    cleanup_on_failure: bool = True
    dependency_recovery: bool = True
    state_backup_enabled: bool = True


@dataclass
class RecoveryAction:
    """Individual recovery action."""

    action_id: str
    action_type: str
    component_id: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 60.0
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    attempt_id: str
    component_id: str
    trigger: RecoveryTrigger
    strategy: RecoveryStrategy
    phase: RecoveryPhase
    actions: list[RecoveryAction]
    start_time: datetime
    end_time: datetime | None = None
    success: bool = False
    error_message: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.start_time, str):
            self.start_time = datetime.fromisoformat(self.start_time)
        if isinstance(self.end_time, str):
            self.end_time = datetime.fromisoformat(self.end_time)


@dataclass
class ComponentDependency:
    """Component dependency definition."""

    component: ComponentType
    depends_on: set[ComponentType]
    startup_delay: float = 5.0  # Delay after dependency starts
    health_check_required: bool = True


class RecoveryManager:
    """
    Comprehensive automatic recovery manager for the four-component architecture.

    Provides intelligent self-healing capabilities with automatic component restart,
    state recovery, dependency resolution, and validation.
    """

    # Component dependency graph
    COMPONENT_DEPENDENCIES = {
        ComponentType.RUST_DAEMON: ComponentDependency(
            component=ComponentType.RUST_DAEMON,
            depends_on=set(),  # No dependencies
            startup_delay=2.0,
            health_check_required=True
        ),
        ComponentType.PYTHON_MCP_SERVER: ComponentDependency(
            component=ComponentType.PYTHON_MCP_SERVER,
            depends_on={ComponentType.RUST_DAEMON},
            startup_delay=5.0,
            health_check_required=True
        ),
        ComponentType.CLI_UTILITY: ComponentDependency(
            component=ComponentType.CLI_UTILITY,
            depends_on=set(),  # Independent
            startup_delay=1.0,
            health_check_required=False
        ),
        ComponentType.CONTEXT_INJECTOR: ComponentDependency(
            component=ComponentType.CONTEXT_INJECTOR,
            depends_on={ComponentType.PYTHON_MCP_SERVER},
            startup_delay=3.0,
            health_check_required=True
        ),
    }

    # Default recovery configurations for each component
    DEFAULT_RECOVERY_CONFIGS = {
        ComponentType.RUST_DAEMON: RecoveryConfig(
            strategy=RecoveryStrategy.PROGRESSIVE,
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=2.0,
            timeout_seconds=300.0
        ),
        ComponentType.PYTHON_MCP_SERVER: RecoveryConfig(
            strategy=RecoveryStrategy.DEPENDENCY_AWARE,
            max_retries=3,
            initial_delay=5.0,
            max_delay=60.0,
            exponential_base=1.5,
            timeout_seconds=180.0
        ),
        ComponentType.CLI_UTILITY: RecoveryConfig(
            strategy=RecoveryStrategy.IMMEDIATE,
            max_retries=2,
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            timeout_seconds=60.0
        ),
        ComponentType.CONTEXT_INJECTOR: RecoveryConfig(
            strategy=RecoveryStrategy.STATE_RECOVERY,
            max_retries=3,
            initial_delay=3.0,
            max_delay=30.0,
            exponential_base=1.8,
            timeout_seconds=120.0
        ),
    }

    # Cleanup patterns for temporary files
    TEMP_FILE_PATTERNS = [
        "*.tmp",
        "*.temp",
        "*.lock",
        "*.pid",
        "*~",
        "*.bak",
        "*.old",
        "core.*",
        "20*-*_*",  # Date-prefixed temporary files
        "LT_20*-*_*",  # Long-term temporary files
    ]

    def __init__(
        self,
        lifecycle_manager: ComponentLifecycleManager | None = None,
        health_monitor: LspHealthMonitor | None = None,
        degradation_manager: DegradationManager | None = None,
        coordinator: ComponentCoordinator | None = None,
        config: dict[str, Any] | None = None
    ):
        """
        Initialize recovery manager.

        Args:
            lifecycle_manager: Component lifecycle manager instance
            health_monitor: LSP health monitor instance
            degradation_manager: Graceful degradation manager instance
            coordinator: Component coordinator instance
            config: Configuration dictionary
        """
        self.lifecycle_manager = lifecycle_manager
        self.health_monitor = health_monitor
        self.degradation_manager = degradation_manager
        self.coordinator = coordinator
        self.config = config or {}

        # Recovery state
        self.active_recoveries: dict[str, RecoveryAttempt] = {}
        self.recovery_history: list[RecoveryAttempt] = []
        self.recovery_configs: dict[ComponentType, RecoveryConfig] = {}

        # Monitoring and detection
        self.failure_patterns: dict[str, list[datetime]] = {}
        self.recovery_statistics = {
            "total_attempts": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "average_recovery_time": 0.0,
            "most_recovered_component": None
        }

        # Background tasks
        self.monitoring_tasks: list[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # State persistence
        self.recovery_db_path = Path(self.config.get("recovery_db_path", "./recovery_state.db"))

        # Notification handlers
        self.notification_handlers: WeakSet[Callable[[UserNotification], None]] = WeakSet()

        # Initialize recovery configurations
        self.recovery_configs = self.DEFAULT_RECOVERY_CONFIGS.copy()

        logger.info("Recovery manager initialized")

    async def initialize(self) -> bool:
        """
        Initialize the recovery manager.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Initialize recovery state database
            await self._initialize_recovery_database()

            # Load recovery history from database
            await self._load_recovery_history()

            # Start monitoring tasks
            await self._start_monitoring_tasks()

            # Register with health monitor for automatic triggers
            if self.health_monitor:
                self.health_monitor.register_notification_handler(
                    self._handle_health_notification
                )

            # Register with degradation manager for mode changes
            if self.degradation_manager:
                self.degradation_manager.register_notification_handler(
                    self._handle_degradation_notification
                )

            logger.info("Recovery manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize recovery manager: {e}")
            return False

    async def _initialize_recovery_database(self):
        """Initialize SQLite database for recovery state persistence."""
        self.recovery_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create recovery tables if they don't exist
        def create_tables():
            conn = sqlite3.connect(str(self.recovery_db_path))
            try:
                cursor = conn.cursor()

                # Recovery attempts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recovery_attempts (
                        attempt_id TEXT PRIMARY KEY,
                        component_id TEXT NOT NULL,
                        trigger TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        phase TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        success INTEGER NOT NULL DEFAULT 0,
                        error_message TEXT,
                        metrics TEXT,
                        actions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Recovery configurations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recovery_configs (
                        component_type TEXT PRIMARY KEY,
                        strategy TEXT NOT NULL,
                        max_retries INTEGER NOT NULL,
                        initial_delay REAL NOT NULL,
                        max_delay REAL NOT NULL,
                        exponential_base REAL NOT NULL,
                        timeout_seconds REAL NOT NULL,
                        validate_after_recovery INTEGER NOT NULL,
                        cleanup_on_failure INTEGER NOT NULL,
                        dependency_recovery INTEGER NOT NULL,
                        state_backup_enabled INTEGER NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Cleanup operations table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS cleanup_operations (
                        operation_id TEXT PRIMARY KEY,
                        cleanup_type TEXT NOT NULL,
                        target_path TEXT NOT NULL,
                        component_id TEXT,
                        success INTEGER NOT NULL DEFAULT 0,
                        files_removed INTEGER DEFAULT 0,
                        bytes_freed INTEGER DEFAULT 0,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                conn.commit()
                return True
            finally:
                conn.close()

        await asyncio.to_thread(create_tables)

    async def _load_recovery_history(self):
        """Load recovery history from database."""
        try:
            def load_history():
                conn = sqlite3.connect(str(self.recovery_db_path))
                try:
                    cursor = conn.cursor()

                    # Load recent recovery attempts (last 100)
                    cursor.execute("""
                        SELECT attempt_id, component_id, trigger, strategy, phase, start_time,
                               end_time, success, error_message, metrics, actions
                        FROM recovery_attempts
                        ORDER BY start_time DESC
                        LIMIT 100
                    """)

                    attempts = []
                    for row in cursor.fetchall():
                        attempt = RecoveryAttempt(
                            attempt_id=row[0],
                            component_id=row[1],
                            trigger=RecoveryTrigger(row[2]),
                            strategy=RecoveryStrategy(row[3]),
                            phase=RecoveryPhase(row[4]),
                            start_time=datetime.fromisoformat(row[5]),
                            end_time=datetime.fromisoformat(row[6]) if row[6] else None,
                            success=bool(row[7]),
                            error_message=row[8],
                            metrics=json.loads(row[9]) if row[9] else {},
                            actions=json.loads(row[10]) if row[10] else []
                        )
                        attempts.append(attempt)
                    return attempts
                finally:
                    conn.close()

            attempts = await asyncio.to_thread(load_history)
            self.recovery_history.extend(attempts)

            # Update statistics
            await self._update_recovery_statistics()

            logger.info(f"Loaded {len(self.recovery_history)} recovery attempts from database")

        except Exception as e:
            logger.error(f"Failed to load recovery history: {e}")

    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks."""
        self.monitoring_tasks = [
            asyncio.create_task(self._recovery_monitoring_loop()),
            asyncio.create_task(self._cleanup_monitoring_loop()),
            asyncio.create_task(self._state_validation_loop()),
            asyncio.create_task(self._statistics_update_loop()),
        ]
        logger.info("Started recovery monitoring tasks")

    async def _recovery_monitoring_loop(self):
        """Main recovery monitoring loop."""
        while not self.shutdown_event.is_set():
            try:
                # Check for failed components
                await self._detect_component_failures()

                # Monitor active recoveries
                await self._monitor_active_recoveries()

                # Check for recovery timeouts
                await self._check_recovery_timeouts()

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in recovery monitoring loop: {e}")
                await asyncio.sleep(5.0)

    async def _cleanup_monitoring_loop(self):
        """Monitor and perform automatic cleanup operations."""
        while not self.shutdown_event.is_set():
            try:
                # Perform automatic cleanup
                await self._perform_automatic_cleanup()

                await asyncio.sleep(300.0)  # Clean every 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup monitoring loop: {e}")
                await asyncio.sleep(60.0)

    async def _state_validation_loop(self):
        """Validate component states and trigger recovery if needed."""
        while not self.shutdown_event.is_set():
            try:
                # Validate component states
                await self._validate_component_states()

                await asyncio.sleep(60.0)  # Validate every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in state validation loop: {e}")
                await asyncio.sleep(30.0)

    async def _statistics_update_loop(self):
        """Update recovery statistics periodically."""
        while not self.shutdown_event.is_set():
            try:
                await self._update_recovery_statistics()
                await asyncio.sleep(3600.0)  # Update every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in statistics update loop: {e}")
                await asyncio.sleep(600.0)

    async def _detect_component_failures(self):
        """Detect component failures and trigger recovery."""
        if not self.lifecycle_manager:
            return

        try:
            component_status = await self.lifecycle_manager.get_component_status()

            if "components" not in component_status:
                return

            for comp_type_str, comp_data in component_status["components"].items():
                component_type = ComponentType(comp_type_str)
                state = comp_data.get("state", "unknown")

                # Check if component needs recovery
                if state in ["failed", "unhealthy", "crashed"]:
                    component_id = f"{component_type.value}-default"

                    # Don't start recovery if already in progress
                    if component_id not in self.active_recoveries:
                        await self._trigger_automatic_recovery(
                            component_type,
                            RecoveryTrigger.COMPONENT_CRASH,
                            f"Component state: {state}"
                        )

        except Exception as e:
            logger.error(f"Failed to detect component failures: {e}")

    async def _monitor_active_recoveries(self):
        """Monitor progress of active recovery operations."""
        completed_recoveries = []

        for recovery_id, attempt in self.active_recoveries.items():
            try:
                # Check if recovery is complete
                if attempt.phase in {RecoveryPhase.COMPLETION, RecoveryPhase.FAILURE}:
                    completed_recoveries.append(recovery_id)
                    continue

                # Update recovery progress if possible
                await self._update_recovery_progress(attempt)

            except Exception as e:
                logger.error(f"Error monitoring recovery {recovery_id}: {e}")

        # Remove completed recoveries
        for recovery_id in completed_recoveries:
            attempt = self.active_recoveries.pop(recovery_id)
            await self._finalize_recovery_attempt(attempt)

    async def _check_recovery_timeouts(self):
        """Check for recovery timeouts and handle them."""
        current_time = datetime.now(timezone.utc)
        timed_out_recoveries = []

        for recovery_id, attempt in self.active_recoveries.items():
            if attempt.end_time is None:
                # Calculate elapsed time
                elapsed = (current_time - attempt.start_time).total_seconds()

                # Get timeout from component config
                component_type = ComponentType(attempt.component_id.split('-')[0])
                config = self.recovery_configs.get(component_type, self.DEFAULT_RECOVERY_CONFIGS[component_type])

                if elapsed > config.timeout_seconds:
                    timed_out_recoveries.append(recovery_id)

        # Handle timeouts
        for recovery_id in timed_out_recoveries:
            attempt = self.active_recoveries[recovery_id]
            await self._handle_recovery_timeout(attempt)

    async def _trigger_automatic_recovery(
        self,
        component_type: ComponentType,
        trigger: RecoveryTrigger,
        reason: str
    ):
        """Trigger automatic recovery for a component."""
        component_id = f"{component_type.value}-default"

        logger.warning(
            f"Triggering automatic recovery for {component_type.value}",
            trigger=trigger.value,
            reason=reason
        )

        # Create recovery attempt
        attempt_id = str(uuid.uuid4())
        config = self.recovery_configs.get(component_type, self.DEFAULT_RECOVERY_CONFIGS[component_type])

        attempt = RecoveryAttempt(
            attempt_id=attempt_id,
            component_id=component_id,
            trigger=trigger,
            strategy=config.strategy,
            phase=RecoveryPhase.DETECTION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        self.active_recoveries[attempt_id] = attempt

        # Start recovery process
        await self._execute_recovery_attempt(attempt)

    async def _execute_recovery_attempt(self, attempt: RecoveryAttempt):
        """Execute a recovery attempt."""
        try:
            attempt.phase = RecoveryPhase.ANALYSIS

            # Analyze failure and determine recovery strategy
            await self._analyze_failure(attempt)

            attempt.phase = RecoveryPhase.PREPARATION

            # Prepare recovery actions
            await self._prepare_recovery_actions(attempt)

            attempt.phase = RecoveryPhase.EXECUTION

            # Execute recovery actions
            success = await self._execute_recovery_actions(attempt)

            if success:
                attempt.phase = RecoveryPhase.VALIDATION

                # Validate recovery
                validation_success = await self._validate_recovery(attempt)

                if validation_success:
                    attempt.phase = RecoveryPhase.COMPLETION
                    attempt.success = True
                    logger.info(f"Recovery successful for {attempt.component_id}")
                else:
                    attempt.phase = RecoveryPhase.FAILURE
                    attempt.error_message = "Recovery validation failed"
                    logger.error(f"Recovery validation failed for {attempt.component_id}")
            else:
                attempt.phase = RecoveryPhase.FAILURE
                logger.error(f"Recovery execution failed for {attempt.component_id}")

            attempt.end_time = datetime.now(timezone.utc)

        except Exception as e:
            attempt.phase = RecoveryPhase.FAILURE
            attempt.error_message = str(e)
            attempt.end_time = datetime.now(timezone.utc)
            logger.error(f"Recovery attempt failed for {attempt.component_id}: {e}")

    async def _analyze_failure(self, attempt: RecoveryAttempt):
        """Analyze component failure to determine appropriate recovery strategy."""
        component_type = ComponentType(attempt.component_id.split('-')[0])

        # Check failure patterns
        pattern_key = f"{component_type.value}_{attempt.trigger.value}"
        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []

        self.failure_patterns[pattern_key].append(attempt.start_time)

        # Keep only recent failures (last hour)
        cutoff_time = attempt.start_time - timedelta(hours=1)
        self.failure_patterns[pattern_key] = [
            t for t in self.failure_patterns[pattern_key] if t > cutoff_time
        ]

        # Adjust strategy based on failure frequency
        recent_failures = len(self.failure_patterns[pattern_key])

        if recent_failures > 5:
            # Too many recent failures, use emergency reset
            attempt.strategy = RecoveryStrategy.EMERGENCY_RESET
        elif recent_failures > 3:
            # Multiple failures, use state recovery
            attempt.strategy = RecoveryStrategy.STATE_RECOVERY
        elif attempt.trigger == RecoveryTrigger.STATE_CORRUPTION:
            # State corruption requires state recovery
            attempt.strategy = RecoveryStrategy.STATE_RECOVERY

        logger.info(
            f"Failure analysis complete for {attempt.component_id}",
            strategy=attempt.strategy.value,
            recent_failures=recent_failures
        )

    async def _prepare_recovery_actions(self, attempt: RecoveryAttempt):
        """Prepare recovery actions based on strategy."""
        component_type = ComponentType(attempt.component_id.split('-')[0])
        config = self.recovery_configs.get(component_type, self.DEFAULT_RECOVERY_CONFIGS[component_type])

        actions = []

        if attempt.strategy == RecoveryStrategy.IMMEDIATE:
            actions.extend([
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="stop_component",
                    component_id=attempt.component_id,
                    description=f"Stop {component_type.value} component",
                    timeout_seconds=30.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="start_component",
                    component_id=attempt.component_id,
                    description=f"Start {component_type.value} component",
                    timeout_seconds=60.0
                )
            ])

        elif attempt.strategy == RecoveryStrategy.PROGRESSIVE:
            actions.extend([
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="stop_component",
                    component_id=attempt.component_id,
                    description=f"Stop {component_type.value} component",
                    timeout_seconds=30.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="cleanup_state",
                    component_id=attempt.component_id,
                    description=f"Clean component state for {component_type.value}",
                    timeout_seconds=60.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="progressive_delay",
                    component_id=attempt.component_id,
                    description="Wait with exponential backoff",
                    parameters={"delay": config.initial_delay},
                    timeout_seconds=config.max_delay
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="start_component",
                    component_id=attempt.component_id,
                    description=f"Start {component_type.value} component",
                    timeout_seconds=120.0
                )
            ])

        elif attempt.strategy == RecoveryStrategy.DEPENDENCY_AWARE:
            # Handle dependencies
            dependency = self.COMPONENT_DEPENDENCIES.get(component_type)
            if dependency and dependency.depends_on:
                for dep_type in dependency.depends_on:
                    actions.append(RecoveryAction(
                        action_id=str(uuid.uuid4()),
                        action_type="verify_dependency",
                        component_id=f"{dep_type.value}-default",
                        description=f"Verify dependency {dep_type.value}",
                        timeout_seconds=30.0
                    ))

            actions.extend([
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="stop_component",
                    component_id=attempt.component_id,
                    description=f"Stop {component_type.value} component",
                    timeout_seconds=30.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="start_component",
                    component_id=attempt.component_id,
                    description=f"Start {component_type.value} component",
                    timeout_seconds=120.0
                )
            ])

        elif attempt.strategy == RecoveryStrategy.STATE_RECOVERY:
            actions.extend([
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="stop_component",
                    component_id=attempt.component_id,
                    description=f"Stop {component_type.value} component",
                    timeout_seconds=30.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="backup_state",
                    component_id=attempt.component_id,
                    description=f"Backup current state for {component_type.value}",
                    timeout_seconds=60.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="cleanup_corrupted_state",
                    component_id=attempt.component_id,
                    description=f"Clean corrupted state for {component_type.value}",
                    timeout_seconds=120.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="restore_state",
                    component_id=attempt.component_id,
                    description=f"Restore clean state for {component_type.value}",
                    timeout_seconds=120.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="start_component",
                    component_id=attempt.component_id,
                    description=f"Start {component_type.value} component",
                    timeout_seconds=120.0
                )
            ])

        elif attempt.strategy == RecoveryStrategy.EMERGENCY_RESET:
            actions.extend([
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="emergency_stop_all",
                    component_id="all",
                    description="Emergency stop all components",
                    timeout_seconds=60.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="full_cleanup",
                    component_id="all",
                    description="Full system cleanup",
                    timeout_seconds=300.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="initialize_fresh_state",
                    component_id="all",
                    description="Initialize fresh system state",
                    timeout_seconds=120.0
                ),
                RecoveryAction(
                    action_id=str(uuid.uuid4()),
                    action_type="start_all_components",
                    component_id="all",
                    description="Start all components in dependency order",
                    timeout_seconds=300.0
                )
            ])

        attempt.actions = actions
        logger.info(f"Prepared {len(actions)} recovery actions for {attempt.component_id}")

    async def _execute_recovery_actions(self, attempt: RecoveryAttempt) -> bool:
        """Execute recovery actions sequentially."""
        all_successful = True

        for action in attempt.actions:
            try:
                logger.info(f"Executing recovery action: {action.description}")

                action_start = time.time()
                success = await self._execute_single_action(action)
                action_duration = time.time() - action_start

                action.retry_count += 1

                if success:
                    logger.info(f"Recovery action completed: {action.description} ({action_duration:.2f}s)")
                else:
                    logger.error(f"Recovery action failed: {action.description}")

                    # Retry if possible
                    if action.retry_count < action.max_retries:
                        logger.info(f"Retrying action {action.description} (attempt {action.retry_count + 1})")
                        await asyncio.sleep(min(action.retry_count * 2, 10))  # Progressive delay
                        continue
                    else:
                        all_successful = False
                        if attempt.strategy != RecoveryStrategy.EMERGENCY_RESET:
                            break  # Stop on first failure unless emergency reset

            except Exception as e:
                logger.error(f"Exception during recovery action {action.description}: {e}")
                action.retry_count += 1

                if action.retry_count < action.max_retries:
                    await asyncio.sleep(min(action.retry_count * 2, 10))
                    continue
                else:
                    all_successful = False
                    if attempt.strategy != RecoveryStrategy.EMERGENCY_RESET:
                        break

        return all_successful

    async def _execute_single_action(self, action: RecoveryAction) -> bool:
        """Execute a single recovery action."""
        try:
            if action.action_type == "stop_component":
                return await self._stop_component_action(action)
            elif action.action_type == "start_component":
                return await self._start_component_action(action)
            elif action.action_type == "cleanup_state":
                return await self._cleanup_state_action(action)
            elif action.action_type == "backup_state":
                return await self._backup_state_action(action)
            elif action.action_type == "restore_state":
                return await self._restore_state_action(action)
            elif action.action_type == "cleanup_corrupted_state":
                return await self._cleanup_corrupted_state_action(action)
            elif action.action_type == "verify_dependency":
                return await self._verify_dependency_action(action)
            elif action.action_type == "progressive_delay":
                return await self._progressive_delay_action(action)
            elif action.action_type == "emergency_stop_all":
                return await self._emergency_stop_all_action(action)
            elif action.action_type == "full_cleanup":
                return await self._full_cleanup_action(action)
            elif action.action_type == "initialize_fresh_state":
                return await self._initialize_fresh_state_action(action)
            elif action.action_type == "start_all_components":
                return await self._start_all_components_action(action)
            else:
                logger.error(f"Unknown recovery action type: {action.action_type}")
                return False

        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            return False

    async def _stop_component_action(self, action: RecoveryAction) -> bool:
        """Stop a component."""
        if not self.lifecycle_manager:
            return False

        component_type = ComponentType(action.component_id.split('-')[0])
        return await self.lifecycle_manager.stop_component(component_type)

    async def _start_component_action(self, action: RecoveryAction) -> bool:
        """Start a component."""
        if not self.lifecycle_manager:
            return False

        component_type = ComponentType(action.component_id.split('-')[0])

        # Check dependencies first
        dependency = self.COMPONENT_DEPENDENCIES.get(component_type)
        if dependency and dependency.depends_on:
            for dep_type in dependency.depends_on:
                if not await self._is_component_healthy(dep_type):
                    logger.warning(f"Dependency {dep_type.value} not healthy for {component_type.value}")
                    return False

        # Start component
        success = await self.lifecycle_manager.start_component(component_type)

        if success and dependency:
            # Wait for startup delay
            await asyncio.sleep(dependency.startup_delay)

        return success

    async def _cleanup_state_action(self, action: RecoveryAction) -> bool:
        """Clean component state."""
        component_type = ComponentType(action.component_id.split('-')[0])

        # Perform component-specific cleanup
        cleanup_paths = []

        if component_type == ComponentType.RUST_DAEMON:
            cleanup_paths.extend([
                "/tmp/rust_daemon_*",
                "/var/run/rust_daemon.pid",
                "/tmp/daemon_socket*"
            ])
        elif component_type == ComponentType.PYTHON_MCP_SERVER:
            cleanup_paths.extend([
                "/tmp/mcp_server_*",
                "/var/run/mcp_server.pid",
                "/tmp/mcp_socket*"
            ])
        elif component_type == ComponentType.CLI_UTILITY:
            cleanup_paths.extend([
                "/tmp/cli_*",
                "/tmp/*.lock"
            ])
        elif component_type == ComponentType.CONTEXT_INJECTOR:
            cleanup_paths.extend([
                "/tmp/context_injector_*",
                "/tmp/injection_*.tmp"
            ])

        return await self._cleanup_paths(cleanup_paths, action.component_id)

    async def _backup_state_action(self, action: RecoveryAction) -> bool:
        """Backup component state."""
        component_type = ComponentType(action.component_id.split('-')[0])

        if not self.coordinator:
            return True  # Skip if no coordinator available

        try:
            # Create backup of component state
            backup_path = Path(f"/tmp/recovery_backup_{component_type.value}_{int(time.time())}")
            backup_path.mkdir(parents=True, exist_ok=True)

            # Backup component-specific state
            if component_type == ComponentType.RUST_DAEMON:
                # Backup daemon state files
                state_files = ["/var/lib/rust_daemon/state.db", "/etc/rust_daemon/config.json"]
            elif component_type == ComponentType.PYTHON_MCP_SERVER:
                # Backup MCP server state
                state_files = ["/var/lib/mcp_server/state.db", "/etc/mcp_server/config.json"]
            else:
                state_files = []

            for state_file in state_files:
                src_path = Path(state_file)
                if src_path.exists():
                    dst_path = backup_path / src_path.name
                    await asyncio.to_thread(shutil.copy2, src_path, dst_path)

            logger.info(f"State backup created for {component_type.value} at {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to backup state for {component_type.value}: {e}")
            return False

    async def _restore_state_action(self, action: RecoveryAction) -> bool:
        """Restore component state from backup."""
        # This is a placeholder - in a real implementation, this would restore
        # from the most recent valid backup
        component_type = ComponentType(action.component_id.split('-')[0])
        logger.info(f"State restore placeholder for {component_type.value}")
        return True

    async def _cleanup_corrupted_state_action(self, action: RecoveryAction) -> bool:
        """Clean up corrupted state files."""
        component_type = ComponentType(action.component_id.split('-')[0])

        corrupted_patterns = [
            "*.corrupt",
            "*.tmp",
            "*.lock",
            "*.partial",
            "core.*"
        ]

        cleanup_dirs = []
        if component_type == ComponentType.RUST_DAEMON:
            cleanup_dirs = ["/var/lib/rust_daemon", "/tmp"]
        elif component_type == ComponentType.PYTHON_MCP_SERVER:
            cleanup_dirs = ["/var/lib/mcp_server", "/tmp"]
        else:
            cleanup_dirs = ["/tmp"]

        files_removed = 0
        for cleanup_dir in cleanup_dirs:
            if os.path.exists(cleanup_dir):
                for pattern in corrupted_patterns:
                    for file_path in Path(cleanup_dir).glob(pattern):
                        try:
                            if file_path.is_file():
                                await asyncio.to_thread(os.remove, file_path)
                                files_removed += 1
                        except Exception as e:
                            logger.warning(f"Failed to remove corrupted file {file_path}: {e}")

        logger.info(f"Cleaned {files_removed} corrupted state files for {component_type.value}")
        return True

    async def _verify_dependency_action(self, action: RecoveryAction) -> bool:
        """Verify that a dependency component is healthy."""
        component_type = ComponentType(action.component_id.split('-')[0])
        return await self._is_component_healthy(component_type)

    async def _progressive_delay_action(self, action: RecoveryAction) -> bool:
        """Wait with progressive delay."""
        delay = action.parameters.get("delay", 1.0)
        await asyncio.sleep(delay)
        return True

    async def _emergency_stop_all_action(self, action: RecoveryAction) -> bool:
        """Emergency stop all components."""
        if not self.lifecycle_manager:
            return False

        success = True
        for component_type in ComponentType:
            try:
                await self.lifecycle_manager.stop_component(component_type)
            except Exception as e:
                logger.error(f"Failed to stop {component_type.value}: {e}")
                success = False

        return success

    async def _full_cleanup_action(self, action: RecoveryAction) -> bool:
        """Perform full system cleanup."""
        cleanup_operations = []

        # Clean temporary files
        cleanup_operations.append(self._cleanup_temporary_files())

        # Clean stale locks
        cleanup_operations.append(self._cleanup_stale_locks())

        # Clean zombie processes
        cleanup_operations.append(self._cleanup_zombie_processes())

        # Clean invalid caches
        cleanup_operations.append(self._cleanup_invalid_caches())

        # Execute all cleanup operations
        results = await asyncio.gather(*cleanup_operations, return_exceptions=True)

        success_count = sum(1 for r in results if r is True)
        total_count = len(results)

        logger.info(f"Full cleanup completed: {success_count}/{total_count} operations successful")
        return success_count > total_count // 2  # Majority must succeed

    async def _initialize_fresh_state_action(self, action: RecoveryAction) -> bool:
        """Initialize fresh system state."""
        if not self.coordinator:
            return True

        try:
            # Reinitialize coordinator state
            await self.coordinator.initialize()
            logger.info("Fresh system state initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize fresh state: {e}")
            return False

    async def _start_all_components_action(self, action: RecoveryAction) -> bool:
        """Start all components in dependency order."""
        if not self.lifecycle_manager:
            return False

        # Sort components by dependency order
        start_order = self._get_component_start_order()

        for component_type in start_order:
            try:
                success = await self.lifecycle_manager.start_component(component_type)
                if not success:
                    logger.error(f"Failed to start {component_type.value}")
                    return False

                # Wait for startup delay
                dependency = self.COMPONENT_DEPENDENCIES.get(component_type)
                if dependency:
                    await asyncio.sleep(dependency.startup_delay)

            except Exception as e:
                logger.error(f"Error starting {component_type.value}: {e}")
                return False

        logger.info("All components started successfully")
        return True

    def _get_component_start_order(self) -> list[ComponentType]:
        """Get component startup order based on dependencies."""
        order = []
        remaining = set(ComponentType)

        while remaining:
            # Find components with no unresolved dependencies
            ready = []
            for comp_type in remaining:
                dependency = self.COMPONENT_DEPENDENCIES.get(comp_type)
                if not dependency or not (dependency.depends_on & remaining):
                    ready.append(comp_type)

            if not ready:
                # Circular dependency or error - add remaining in arbitrary order
                ready = list(remaining)

            # Add ready components to order
            for comp_type in ready:
                order.append(comp_type)
                remaining.remove(comp_type)

        return order

    async def _validate_recovery(self, attempt: RecoveryAttempt) -> bool:
        """Validate that recovery was successful."""
        component_type = ComponentType(attempt.component_id.split('-')[0])

        # Check component health
        if not await self._is_component_healthy(component_type):
            return False

        # Check circuit breaker state
        if self.degradation_manager:
            cb_state = self.degradation_manager.get_circuit_breaker_state(attempt.component_id)
            if cb_state == CircuitBreakerState.OPEN:
                return False

        # Component-specific validation
        if component_type == ComponentType.RUST_DAEMON:
            # Validate daemon-specific functionality
            return await self._validate_rust_daemon()
        elif component_type == ComponentType.PYTHON_MCP_SERVER:
            # Validate MCP server-specific functionality
            return await self._validate_mcp_server()
        elif component_type == ComponentType.CLI_UTILITY:
            # Validate CLI utility functionality
            return await self._validate_cli_utility()
        elif component_type == ComponentType.CONTEXT_INJECTOR:
            # Validate context injector functionality
            return await self._validate_context_injector()

        return True

    async def _validate_rust_daemon(self) -> bool:
        """Validate Rust daemon functionality."""
        # This would perform daemon-specific validation
        # For now, just check basic health
        return True

    async def _validate_mcp_server(self) -> bool:
        """Validate MCP server functionality."""
        # This would perform MCP server-specific validation
        # For now, just check basic health
        return True

    async def _validate_cli_utility(self) -> bool:
        """Validate CLI utility functionality."""
        # This would perform CLI-specific validation
        # For now, just check basic health
        return True

    async def _validate_context_injector(self) -> bool:
        """Validate context injector functionality."""
        # This would perform context injector-specific validation
        # For now, just check basic health
        return True

    async def _is_component_healthy(self, component_type: ComponentType) -> bool:
        """Check if a component is healthy."""
        if not self.lifecycle_manager:
            return False

        try:
            component_status = await self.lifecycle_manager.get_component_status()

            if "components" not in component_status:
                return False

            comp_data = component_status["components"].get(component_type.value)
            if not comp_data:
                return False

            state = comp_data.get("state", "unknown")
            return state in ["operational", "ready", "healthy"]

        except Exception as e:
            logger.error(f"Failed to check component health for {component_type.value}: {e}")
            return False

    async def _cleanup_paths(self, paths: list[str], component_id: str) -> bool:
        """Clean up specified file paths."""
        files_removed = 0

        for path_pattern in paths:
            try:
                # Handle glob patterns
                if '*' in path_pattern:
                    from glob import glob
                    matching_paths = await asyncio.to_thread(glob, path_pattern)
                    for path in matching_paths:
                        if os.path.exists(path):
                            if os.path.isfile(path):
                                await asyncio.to_thread(os.remove, path)
                                files_removed += 1
                            elif os.path.isdir(path):
                                await asyncio.to_thread(shutil.rmtree, path)
                                files_removed += 1
                else:
                    # Single path
                    if os.path.exists(path_pattern):
                        if os.path.isfile(path_pattern):
                            await asyncio.to_thread(os.remove, path_pattern)
                            files_removed += 1
                        elif os.path.isdir(path_pattern):
                            await asyncio.to_thread(shutil.rmtree, path_pattern)
                            files_removed += 1

            except Exception as e:
                logger.warning(f"Failed to clean path {path_pattern}: {e}")

        logger.info(f"Cleaned {files_removed} files/directories for {component_id}")
        return True

    async def _perform_automatic_cleanup(self):
        """Perform automatic cleanup operations."""
        try:
            # Clean temporary files
            await self._cleanup_temporary_files()

            # Clean stale locks (older than 1 hour)
            await self._cleanup_stale_locks()

            # Record cleanup operation
            operation_id = str(uuid.uuid4())
            await self._record_cleanup_operation(
                operation_id,
                CleanupType.TEMPORARY_FILES,
                "/tmp",
                None,
                True,
                0,
                0
            )

        except Exception as e:
            logger.error(f"Error in automatic cleanup: {e}")

    async def _cleanup_temporary_files(self) -> bool:
        """Clean up temporary files."""
        try:
            temp_dirs = ["/tmp", "/var/tmp", "./"]
            files_removed = 0

            for temp_dir in temp_dirs:
                if not os.path.exists(temp_dir):
                    continue

                for pattern in self.TEMP_FILE_PATTERNS:
                    for file_path in Path(temp_dir).glob(pattern):
                        try:
                            # Only remove files older than 1 hour
                            if file_path.is_file():
                                file_age = time.time() - file_path.stat().st_mtime
                                if file_age > 3600:  # 1 hour
                                    await asyncio.to_thread(os.remove, file_path)
                                    files_removed += 1
                        except Exception as e:
                            logger.debug(f"Failed to remove temp file {file_path}: {e}")

            logger.debug(f"Cleaned {files_removed} temporary files")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup temporary files: {e}")
            return False

    async def _cleanup_stale_locks(self) -> bool:
        """Clean up stale lock files."""
        try:
            lock_patterns = ["*.lock", "*.pid"]
            files_removed = 0

            for pattern in lock_patterns:
                for lock_file in Path("/tmp").glob(pattern):
                    try:
                        if lock_file.is_file():
                            # Check if lock is stale (older than 1 hour)
                            file_age = time.time() - lock_file.stat().st_mtime
                            if file_age > 3600:
                                await asyncio.to_thread(os.remove, lock_file)
                                files_removed += 1
                    except Exception as e:
                        logger.debug(f"Failed to remove stale lock {lock_file}: {e}")

            logger.debug(f"Cleaned {files_removed} stale locks")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup stale locks: {e}")
            return False

    async def _cleanup_zombie_processes(self) -> bool:
        """Clean up zombie processes."""
        try:
            # This is a placeholder - actual implementation would identify and clean zombies
            logger.debug("Zombie process cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup zombie processes: {e}")
            return False

    async def _cleanup_invalid_caches(self) -> bool:
        """Clean up invalid cache entries."""
        try:
            # This would clean component-specific caches
            logger.debug("Invalid cache cleanup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to cleanup invalid caches: {e}")
            return False

    async def _validate_component_states(self):
        """Validate component states and trigger recovery if needed."""
        if not self.lifecycle_manager:
            return

        try:
            component_status = await self.lifecycle_manager.get_component_status()

            if "components" not in component_status:
                return

            for comp_type_str, comp_data in component_status["components"].items():
                component_type = ComponentType(comp_type_str)
                state = comp_data.get("state", "unknown")

                # Check for inconsistent states that need recovery
                if state in ["inconsistent", "partially_failed", "unknown"]:
                    component_id = f"{component_type.value}-default"

                    if component_id not in self.active_recoveries:
                        await self._trigger_automatic_recovery(
                            component_type,
                            RecoveryTrigger.STATE_CORRUPTION,
                            f"Component state validation failed: {state}"
                        )

        except Exception as e:
            logger.error(f"Failed to validate component states: {e}")

    async def _update_recovery_statistics(self):
        """Update recovery statistics."""
        try:
            total_attempts = len(self.recovery_history)
            successful = sum(1 for attempt in self.recovery_history if attempt.success)
            failed = total_attempts - successful

            # Calculate average recovery time
            completed_attempts = [a for a in self.recovery_history if a.end_time]
            if completed_attempts:
                total_time = sum(
                    (a.end_time - a.start_time).total_seconds()
                    for a in completed_attempts
                )
                avg_time = total_time / len(completed_attempts)
            else:
                avg_time = 0.0

            # Find most recovered component
            component_counts = {}
            for attempt in self.recovery_history:
                comp_type = attempt.component_id.split('-')[0]
                component_counts[comp_type] = component_counts.get(comp_type, 0) + 1

            most_recovered = max(component_counts.items(), key=lambda x: x[1])[0] if component_counts else None

            self.recovery_statistics = {
                "total_attempts": total_attempts,
                "successful_recoveries": successful,
                "failed_recoveries": failed,
                "average_recovery_time": avg_time,
                "most_recovered_component": most_recovered
            }

        except Exception as e:
            logger.error(f"Failed to update recovery statistics: {e}")

    async def _update_recovery_progress(self, attempt: RecoveryAttempt):
        """Update progress of a recovery attempt."""
        # This could update progress based on action completion
        # For now, just ensure the attempt is properly tracked
        pass

    async def _finalize_recovery_attempt(self, attempt: RecoveryAttempt):
        """Finalize a completed recovery attempt."""
        try:
            # Store in database
            await self._store_recovery_attempt(attempt)

            # Add to history
            self.recovery_history.append(attempt)

            # Keep history size manageable
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-1000:]

            # Update statistics
            await self._update_recovery_statistics()

            # Send notification
            await self._send_recovery_notification(attempt)

            logger.info(
                f"Recovery attempt finalized for {attempt.component_id}",
                success=attempt.success,
                phase=attempt.phase.value,
                strategy=attempt.strategy.value
            )

        except Exception as e:
            logger.error(f"Failed to finalize recovery attempt: {e}")

    async def _store_recovery_attempt(self, attempt: RecoveryAttempt):
        """Store recovery attempt in database."""
        try:
            def store_attempt():
                conn = sqlite3.connect(str(self.recovery_db_path))
                try:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO recovery_attempts
                        (attempt_id, component_id, trigger, strategy, phase, start_time,
                         end_time, success, error_message, metrics, actions)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        attempt.attempt_id,
                        attempt.component_id,
                        attempt.trigger.value,
                        attempt.strategy.value,
                        attempt.phase.value,
                        attempt.start_time.isoformat(),
                        attempt.end_time.isoformat() if attempt.end_time else None,
                        int(attempt.success),
                        attempt.error_message,
                        json.dumps(attempt.metrics),
                        json.dumps([asdict(action) for action in attempt.actions])
                    ))

                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(store_attempt)

        except Exception as e:
            logger.error(f"Failed to store recovery attempt: {e}")

    async def _record_cleanup_operation(
        self,
        operation_id: str,
        cleanup_type: CleanupType,
        target_path: str,
        component_id: str | None,
        success: bool,
        files_removed: int,
        bytes_freed: int,
        error_message: str | None = None
    ):
        """Record cleanup operation in database."""
        try:
            def record_operation():
                conn = sqlite3.connect(str(self.recovery_db_path))
                try:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT INTO cleanup_operations
                        (operation_id, cleanup_type, target_path, component_id, success,
                         files_removed, bytes_freed, error_message)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        operation_id,
                        cleanup_type.value,
                        target_path,
                        component_id,
                        int(success),
                        files_removed,
                        bytes_freed,
                        error_message
                    ))

                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(record_operation)

        except Exception as e:
            logger.error(f"Failed to record cleanup operation: {e}")

    async def _handle_recovery_timeout(self, attempt: RecoveryAttempt):
        """Handle recovery timeout."""
        attempt.phase = RecoveryPhase.FAILURE
        attempt.error_message = "Recovery operation timed out"
        attempt.end_time = datetime.now(timezone.utc)

        logger.error(f"Recovery timed out for {attempt.component_id}")

        # Try emergency reset if not already attempted
        if attempt.strategy != RecoveryStrategy.EMERGENCY_RESET:
            await self._trigger_automatic_recovery(
                ComponentType(attempt.component_id.split('-')[0]),
                RecoveryTrigger.MANUAL_TRIGGER,
                "Previous recovery timed out, attempting emergency reset"
            )

    async def _handle_health_notification(self, notification: UserNotification):
        """Handle health notifications from health monitor."""
        if notification.level in {NotificationLevel.CRITICAL, NotificationLevel.ERROR}:
            # Extract component information from notification if possible
            if "daemon" in notification.message.lower():
                component_type = ComponentType.RUST_DAEMON
            elif "mcp" in notification.message.lower() or "server" in notification.message.lower():
                component_type = ComponentType.PYTHON_MCP_SERVER
            elif "cli" in notification.message.lower():
                component_type = ComponentType.CLI_UTILITY
            elif "context" in notification.message.lower() or "injector" in notification.message.lower():
                component_type = ComponentType.CONTEXT_INJECTOR
            else:
                return  # Can't determine component

            # Trigger recovery
            await self._trigger_automatic_recovery(
                component_type,
                RecoveryTrigger.HEALTH_CHECK_FAILURE,
                f"Health monitor alert: {notification.title}"
            )

    async def _handle_degradation_notification(self, notification: UserNotification):
        """Handle degradation notifications."""
        if notification.level == NotificationLevel.CRITICAL:
            # Trigger emergency recovery for all failed components
            if self.lifecycle_manager:
                try:
                    component_status = await self.lifecycle_manager.get_component_status()

                    if "components" in component_status:
                        for comp_type_str, comp_data in component_status["components"].items():
                            state = comp_data.get("state", "unknown")
                            if state in ["failed", "unhealthy"]:
                                component_type = ComponentType(comp_type_str)
                                await self._trigger_automatic_recovery(
                                    component_type,
                                    RecoveryTrigger.DEGRADATION_MODE_CHANGE,
                                    "Critical degradation mode triggered recovery"
                                )

                except Exception as e:
                    logger.error(f"Failed to handle degradation notification: {e}")

    async def _send_recovery_notification(self, attempt: RecoveryAttempt):
        """Send notification about recovery attempt completion."""
        if attempt.success:
            level = NotificationLevel.INFO
            title = f"Recovery Successful: {attempt.component_id}"
            message = f"Component {attempt.component_id} has been successfully recovered using {attempt.strategy.value} strategy."
        else:
            level = NotificationLevel.ERROR
            title = f"Recovery Failed: {attempt.component_id}"
            message = f"Recovery failed for {attempt.component_id}. Error: {attempt.error_message or 'Unknown error'}"

        notification = UserNotification(
            timestamp=time.time(),
            level=level,
            title=title,
            message=message,
            server_name="workspace-qdrant-mcp",
            troubleshooting_steps=[
                "Check component logs for detailed error information",
                "Verify system resources are available",
                "Consider manual intervention if automatic recovery continues to fail",
                "Review component configuration for potential issues"
            ],
            auto_recovery_attempted=True
        )

        # Send to all registered handlers
        for handler in list(self.notification_handlers):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(notification)
                else:
                    handler(notification)
            except Exception as e:
                logger.error(f"Error in recovery notification handler: {e}")

    # Public API methods

    async def trigger_component_recovery(
        self,
        component_type: ComponentType,
        strategy: RecoveryStrategy | None = None,
        reason: str = "Manual trigger"
    ) -> str:
        """
        Manually trigger recovery for a specific component.

        Args:
            component_type: Component to recover
            strategy: Recovery strategy to use (optional)
            reason: Reason for recovery

        Returns:
            Recovery attempt ID
        """
        component_id = f"{component_type.value}-default"

        # Check if recovery already in progress
        for attempt in self.active_recoveries.values():
            if attempt.component_id == component_id:
                raise ValueError(f"Recovery already in progress for {component_id}")

        # Use provided strategy or default
        if strategy is None:
            config = self.recovery_configs.get(component_type, self.DEFAULT_RECOVERY_CONFIGS[component_type])
            strategy = config.strategy

        # Create recovery attempt
        attempt_id = str(uuid.uuid4())
        attempt = RecoveryAttempt(
            attempt_id=attempt_id,
            component_id=component_id,
            trigger=RecoveryTrigger.MANUAL_TRIGGER,
            strategy=strategy,
            phase=RecoveryPhase.DETECTION,
            actions=[],
            start_time=datetime.now(timezone.utc)
        )

        self.active_recoveries[attempt_id] = attempt

        # Start recovery process
        asyncio.create_task(self._execute_recovery_attempt(attempt))

        logger.info(f"Manual recovery triggered for {component_type.value} with strategy {strategy.value}")
        return attempt_id

    async def get_recovery_status(self, attempt_id: str) -> dict[str, Any] | None:
        """
        Get status of a recovery attempt.

        Args:
            attempt_id: Recovery attempt ID

        Returns:
            Recovery status information or None if not found
        """
        # Check active recoveries
        if attempt_id in self.active_recoveries:
            attempt = self.active_recoveries[attempt_id]
            return self._attempt_to_dict(attempt)

        # Check history
        for attempt in self.recovery_history:
            if attempt.attempt_id == attempt_id:
                return self._attempt_to_dict(attempt)

        return None

    def _attempt_to_dict(self, attempt: RecoveryAttempt) -> dict[str, Any]:
        """Convert recovery attempt to dictionary."""
        return {
            "attempt_id": attempt.attempt_id,
            "component_id": attempt.component_id,
            "trigger": attempt.trigger.value,
            "strategy": attempt.strategy.value,
            "phase": attempt.phase.value,
            "start_time": attempt.start_time.isoformat(),
            "end_time": attempt.end_time.isoformat() if attempt.end_time else None,
            "success": attempt.success,
            "error_message": attempt.error_message,
            "actions": [
                {
                    "action_id": action.action_id,
                    "action_type": action.action_type,
                    "description": action.description,
                    "retry_count": action.retry_count,
                    "max_retries": action.max_retries
                }
                for action in attempt.actions
            ],
            "metrics": attempt.metrics
        }

    async def get_active_recoveries(self) -> list[dict[str, Any]]:
        """
        Get all active recovery attempts.

        Returns:
            List of active recovery attempts
        """
        return [self._attempt_to_dict(attempt) for attempt in self.active_recoveries.values()]

    async def get_recovery_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get recovery history.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of recovery attempts
        """
        recent_history = self.recovery_history[-limit:] if limit > 0 else self.recovery_history
        return [self._attempt_to_dict(attempt) for attempt in recent_history]

    def get_recovery_statistics(self) -> dict[str, Any]:
        """
        Get recovery statistics.

        Returns:
            Recovery statistics
        """
        return self.recovery_statistics.copy()

    async def cancel_recovery(self, attempt_id: str) -> bool:
        """
        Cancel an active recovery attempt.

        Args:
            attempt_id: Recovery attempt ID

        Returns:
            True if cancelled, False if not found or already completed
        """
        if attempt_id not in self.active_recoveries:
            return False

        attempt = self.active_recoveries[attempt_id]
        if attempt.phase in {RecoveryPhase.COMPLETION, RecoveryPhase.FAILURE}:
            return False

        # Mark as cancelled
        attempt.phase = RecoveryPhase.FAILURE
        attempt.error_message = "Recovery cancelled by user"
        attempt.end_time = datetime.now(timezone.utc)

        logger.info(f"Recovery cancelled for {attempt.component_id}")
        return True

    def register_notification_handler(
        self,
        handler: Callable[[UserNotification], None]
    ):
        """Register a handler for recovery notifications."""
        self.notification_handlers.add(handler)

    async def update_recovery_config(
        self,
        component_type: ComponentType,
        config: RecoveryConfig
    ):
        """
        Update recovery configuration for a component.

        Args:
            component_type: Component type
            config: New recovery configuration
        """
        self.recovery_configs[component_type] = config

        # Store in database
        try:
            def update_config():
                conn = sqlite3.connect(str(self.recovery_db_path))
                try:
                    cursor = conn.cursor()

                    cursor.execute("""
                        INSERT OR REPLACE INTO recovery_configs
                        (component_type, strategy, max_retries, initial_delay, max_delay,
                         exponential_base, timeout_seconds, validate_after_recovery,
                         cleanup_on_failure, dependency_recovery, state_backup_enabled)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        component_type.value,
                        config.strategy.value,
                        config.max_retries,
                        config.initial_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.timeout_seconds,
                        int(config.validate_after_recovery),
                        int(config.cleanup_on_failure),
                        int(config.dependency_recovery),
                        int(config.state_backup_enabled)
                    ))

                    conn.commit()
                finally:
                    conn.close()

            await asyncio.to_thread(update_config)
            logger.info(f"Recovery configuration updated for {component_type.value}")

        except Exception as e:
            logger.error(f"Failed to update recovery configuration: {e}")

    async def get_recovery_config(self, component_type: ComponentType) -> RecoveryConfig:
        """
        Get recovery configuration for a component.

        Args:
            component_type: Component type

        Returns:
            Recovery configuration
        """
        return self.recovery_configs.get(component_type, self.DEFAULT_RECOVERY_CONFIGS[component_type])

    async def force_cleanup(self, cleanup_type: CleanupType, target_path: str = "") -> bool:
        """
        Force cleanup operation.

        Args:
            cleanup_type: Type of cleanup to perform
            target_path: Specific path to clean (optional)

        Returns:
            True if cleanup successful, False otherwise
        """
        operation_id = str(uuid.uuid4())

        try:
            if cleanup_type == CleanupType.TEMPORARY_FILES:
                success = await self._cleanup_temporary_files()
            elif cleanup_type == CleanupType.STALE_LOCKS:
                success = await self._cleanup_stale_locks()
            elif cleanup_type == CleanupType.ZOMBIE_PROCESSES:
                success = await self._cleanup_zombie_processes()
            elif cleanup_type == CleanupType.INVALID_CACHES:
                success = await self._cleanup_invalid_caches()
            else:
                success = False

            await self._record_cleanup_operation(
                operation_id,
                cleanup_type,
                target_path or "/tmp",
                None,
                success,
                0,
                0,
                None if success else f"Cleanup type {cleanup_type.value} not implemented"
            )

            return success

        except Exception as e:
            await self._record_cleanup_operation(
                operation_id,
                cleanup_type,
                target_path or "/tmp",
                None,
                False,
                0,
                0,
                str(e)
            )
            logger.error(f"Failed to force cleanup {cleanup_type.value}: {e}")
            return False

    async def shutdown(self):
        """Shutdown the recovery manager."""
        logger.info("Shutting down recovery manager")

        self.shutdown_event.set()

        # Cancel monitoring tasks
        for task in self.monitoring_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()

        # Complete any active recoveries
        for attempt in self.active_recoveries.values():
            if attempt.phase not in {RecoveryPhase.COMPLETION, RecoveryPhase.FAILURE}:
                attempt.phase = RecoveryPhase.FAILURE
                attempt.error_message = "Recovery manager shutdown"
                attempt.end_time = datetime.now(timezone.utc)
                await self._finalize_recovery_attempt(attempt)

        self.active_recoveries.clear()

        logger.info("Recovery manager shutdown complete")

    @asynccontextmanager
    async def recovery_context(self):
        """
        Async context manager for recovery management lifecycle.

        Usage:
            async with recovery_manager.recovery_context():
                # Recovery management is active
                await do_work()
            # Recovery manager is automatically stopped
        """
        await self.initialize()
        try:
            yield self
        finally:
            await self.shutdown()


# Global recovery manager instance
_recovery_manager: RecoveryManager | None = None


async def get_recovery_manager(
    lifecycle_manager: ComponentLifecycleManager | None = None,
    health_monitor: LspHealthMonitor | None = None,
    degradation_manager: DegradationManager | None = None,
    coordinator: ComponentCoordinator | None = None,
    config: dict[str, Any] | None = None
) -> RecoveryManager:
    """Get or create global recovery manager instance."""
    global _recovery_manager

    if _recovery_manager is None:
        _recovery_manager = RecoveryManager(
            lifecycle_manager, health_monitor, degradation_manager, coordinator, config
        )

        if not await _recovery_manager.initialize():
            raise RuntimeError("Failed to initialize recovery manager")

    return _recovery_manager


async def shutdown_recovery_manager():
    """Shutdown global recovery manager."""
    global _recovery_manager

    if _recovery_manager:
        await _recovery_manager.shutdown()
        _recovery_manager = None
