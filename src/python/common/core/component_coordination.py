"""
Component Coordination System for Four-Component Architecture.

This module extends the SQLite State Manager with component coordination capabilities
for managing the four-component architecture: Rust daemon, Python MCP server,
CLI utility, and context injector.

Key Features:
    - Component registration and lifecycle management
    - Inter-component communication state tracking
    - Component health monitoring and failure detection
    - Processing queue coordination between components
    - Component recovery and failover mechanisms
    - Resource usage tracking and coordination
    - Component synchronization and dependency management

Architecture Components:
    1. Rust Daemon (Heavy Processing Engine)
    2. Python MCP Server (Intelligent Interface)
    3. CLI Utility (User Control & Administration)
    4. Context Injector (LLM Integration/Hook)

Example:
    ```python
    from workspace_qdrant_mcp.core.component_coordination import ComponentCoordinator

    # Initialize component coordinator
    coordinator = ComponentCoordinator(db_path="./workspace_state.db")
    await coordinator.initialize()

    # Register component
    await coordinator.register_component(
        component_type=ComponentType.RUST_DAEMON,
        instance_id="daemon-001",
        config={"grpc_port": 50051}
    )

    # Track component health
    await coordinator.update_component_health(
        component_id="daemon-001",
        health_status=ComponentHealth.HEALTHY,
        metrics={"cpu_usage": 25.0, "memory_mb": 128}
    )
    ```
"""

import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

# Import base SQLite state manager
from python.common.core.sqlite_state_manager import SQLiteStateManager, DatabaseTransaction


class ComponentType(Enum):
    """Component types in the four-component architecture."""

    RUST_DAEMON = "rust_daemon"
    PYTHON_MCP_SERVER = "python_mcp_server"
    CLI_UTILITY = "cli_utility"
    CONTEXT_INJECTOR = "context_injector"


class ComponentStatus(Enum):
    """Component operational status."""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


class ComponentHealth(Enum):
    """Component health status for monitoring."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CommunicationChannel(Enum):
    """Inter-component communication channels."""

    GRPC = "grpc"
    SQLITE_SHARED = "sqlite_shared"
    SIGNAL_HANDLING = "signal_handling"
    DIRECT_QDRANT = "direct_qdrant"
    HOOK_STREAMING = "hook_streaming"


class ProcessingQueueType(Enum):
    """Types of processing queues for component coordination."""

    FILE_INGESTION = "file_ingestion"
    SEARCH_REQUESTS = "search_requests"
    RULE_UPDATES = "rule_updates"
    HEALTH_CHECKS = "health_checks"
    ADMIN_COMMANDS = "admin_commands"
    CONTEXT_INJECTION = "context_injection"


@dataclass
class ComponentRecord:
    """Record for tracking component registration and status."""

    component_id: str
    component_type: ComponentType
    instance_id: str
    status: ComponentStatus
    health: ComponentHealth
    version: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    endpoints: Optional[Dict[str, str]] = None
    capabilities: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    resources: Optional[Dict[str, Any]] = None
    last_heartbeat: Optional[datetime] = None
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    restart_count: int = 0
    failure_count: int = 0
    recovery_attempts: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None
    updated_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at


@dataclass
class CommunicationRecord:
    """Record for tracking inter-component communication."""

    communication_id: str
    source_component: str
    target_component: str
    channel: CommunicationChannel
    message_type: str
    status: str  # pending, sent, received, failed, timeout
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    timeout_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


@dataclass
class HealthMetric:
    """Record for component health metrics."""

    metric_id: str
    component_id: str
    metric_name: str
    metric_value: float
    metric_unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    is_alert: bool = False
    alert_level: Optional[str] = None
    recorded_at: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.recorded_at is None:
            self.recorded_at = datetime.now(timezone.utc)


@dataclass
class ComponentQueueItem:
    """Item in component-specific processing queues."""

    queue_item_id: str
    component_id: str
    queue_type: ProcessingQueueType
    priority: int  # 1=highest, 10=lowest
    payload: Dict[str, Any]
    status: str  # pending, processing, completed, failed, cancelled
    assigned_worker: Optional[str] = None
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    dependencies: Optional[List[str]] = None
    timeout_seconds: Optional[int] = None
    scheduled_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.scheduled_at is None:
            self.scheduled_at = self.created_at


class ComponentCoordinator(SQLiteStateManager):
    """
    Component Coordination System extending SQLite State Manager.

    Manages component registration, health monitoring, inter-component communication,
    and processing queue coordination for the four-component architecture.
    """

    SCHEMA_VERSION = 4  # Extended from base schema version 3
    HEARTBEAT_INTERVAL = 30  # seconds
    HEALTH_CHECK_INTERVAL = 60  # seconds
    QUEUE_CLEANUP_INTERVAL = 300  # 5 minutes
    COMPONENT_TIMEOUT = 120  # 2 minutes

    def __init__(self, db_path: str = "workspace_state.db"):
        """
        Initialize Component Coordinator.

        Args:
            db_path: Path to SQLite database file
        """
        super().__init__(db_path)
        self._coordination_tasks: List[asyncio.Task] = []
        self._component_registry: Dict[str, ComponentRecord] = {}
        self._communication_channels: Dict[str, List[str]] = {}

    async def initialize(self) -> bool:
        """
        Initialize component coordinator with extended schema.

        Returns:
            True if initialization successful, False otherwise
        """
        # Initialize base state manager
        if not await super().initialize():
            return False

        # Setup component coordination schema
        await self._setup_coordination_schema()

        # Start coordination background tasks
        await self._start_coordination_tasks()

        logger.info("Component coordination system initialized")
        return True

    async def _setup_coordination_schema(self):
        """Setup extended schema for component coordination."""
        with self._lock:
            connection = self.connection
            with DatabaseTransaction(connection):
                # Component registry table
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS component_registry (
                        component_id TEXT PRIMARY KEY,
                        component_type TEXT NOT NULL,
                        instance_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        health TEXT NOT NULL,
                        version TEXT,
                        config TEXT,  -- JSON
                        endpoints TEXT,  -- JSON
                        capabilities TEXT,  -- JSON array
                        dependencies TEXT,  -- JSON array
                        resources TEXT,  -- JSON
                        last_heartbeat TEXT,  -- ISO datetime
                        started_at TEXT,  -- ISO datetime
                        stopped_at TEXT,  -- ISO datetime
                        restart_count INTEGER DEFAULT 0,
                        failure_count INTEGER DEFAULT 0,
                        recovery_attempts INTEGER DEFAULT 0,
                        metadata TEXT,  -- JSON
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                # Inter-component communication tracking
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS component_communication (
                        communication_id TEXT PRIMARY KEY,
                        source_component TEXT NOT NULL,
                        target_component TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        request_data TEXT,  -- JSON
                        response_data TEXT,  -- JSON
                        latency_ms REAL,
                        error_message TEXT,
                        retry_count INTEGER DEFAULT 0,
                        timeout_at TEXT,  -- ISO datetime
                        sent_at TEXT,  -- ISO datetime
                        received_at TEXT,  -- ISO datetime
                        metadata TEXT,  -- JSON
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (source_component) REFERENCES component_registry(component_id),
                        FOREIGN KEY (target_component) REFERENCES component_registry(component_id)
                    )
                """)

                # Component health metrics
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS component_health_metrics (
                        metric_id TEXT PRIMARY KEY,
                        component_id TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metric_unit TEXT NOT NULL,
                        threshold_warning REAL,
                        threshold_critical REAL,
                        is_alert BOOLEAN DEFAULT FALSE,
                        alert_level TEXT,
                        recorded_at TEXT NOT NULL,  -- ISO datetime
                        metadata TEXT,  -- JSON
                        FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
                    )
                """)

                # Component processing queues
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS component_processing_queue (
                        queue_item_id TEXT PRIMARY KEY,
                        component_id TEXT NOT NULL,
                        queue_type TEXT NOT NULL,
                        priority INTEGER NOT NULL,
                        payload TEXT NOT NULL,  -- JSON
                        status TEXT NOT NULL,
                        assigned_worker TEXT,
                        processing_started TEXT,  -- ISO datetime
                        processing_completed TEXT,  -- ISO datetime
                        retry_count INTEGER DEFAULT 0,
                        max_retries INTEGER DEFAULT 3,
                        error_message TEXT,
                        dependencies TEXT,  -- JSON array
                        timeout_seconds INTEGER,
                        scheduled_at TEXT,  -- ISO datetime
                        metadata TEXT,  -- JSON
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
                    )
                """)

                # Component recovery log
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS component_recovery_log (
                        recovery_id TEXT PRIMARY KEY,
                        component_id TEXT NOT NULL,
                        failure_type TEXT NOT NULL,
                        failure_details TEXT,  -- JSON
                        recovery_action TEXT NOT NULL,
                        recovery_result TEXT NOT NULL,
                        recovery_duration_ms INTEGER,
                        automatic_recovery BOOLEAN DEFAULT TRUE,
                        recovery_metadata TEXT,  -- JSON
                        failed_at TEXT NOT NULL,  -- ISO datetime
                        recovered_at TEXT,  -- ISO datetime
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (component_id) REFERENCES component_registry(component_id)
                    )
                """)

                # Create indexes for performance
                connection.execute("CREATE INDEX IF NOT EXISTS idx_component_type ON component_registry(component_type)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_component_status ON component_registry(status)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_component_health ON component_registry(health)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_communication_source ON component_communication(source_component)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_communication_target ON component_communication(target_component)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_communication_status ON component_communication(status)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_health_component ON component_health_metrics(component_id)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_health_recorded ON component_health_metrics(recorded_at)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_queue_component ON component_processing_queue(component_id)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_queue_type ON component_processing_queue(queue_type)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON component_processing_queue(status)")
                connection.execute("CREATE INDEX IF NOT EXISTS idx_queue_priority ON component_processing_queue(priority)")

                logger.info("Component coordination schema setup completed")

    async def register_component(
        self,
        component_type: ComponentType,
        instance_id: str,
        config: Optional[Dict[str, Any]] = None,
        endpoints: Optional[Dict[str, str]] = None,
        capabilities: Optional[List[str]] = None,
        dependencies: Optional[List[str]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Register a component in the coordination system.

        Args:
            component_type: Type of component
            instance_id: Unique instance identifier
            config: Component configuration
            endpoints: Component endpoints (e.g., gRPC port)
            capabilities: List of component capabilities
            dependencies: List of component dependencies
            version: Component version

        Returns:
            Component ID for the registered component
        """
        component_id = f"{component_type.value}-{instance_id}"

        record = ComponentRecord(
            component_id=component_id,
            component_type=component_type,
            instance_id=instance_id,
            status=ComponentStatus.STARTING,
            health=ComponentHealth.UNKNOWN,
            version=version,
            config=config,
            endpoints=endpoints,
            capabilities=capabilities or [],
            dependencies=dependencies or [],
            started_at=datetime.now(timezone.utc)
        )

        with self._lock:
            connection = self.connection
            with DatabaseTransaction(connection):
                connection.execute("""
                    INSERT OR REPLACE INTO component_registry (
                        component_id, component_type, instance_id, status, health,
                        version, config, endpoints, capabilities, dependencies,
                        resources, last_heartbeat, started_at, stopped_at,
                        restart_count, failure_count, recovery_attempts,
                        metadata, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.component_id,
                    record.component_type.value,
                    record.instance_id,
                    record.status.value,
                    record.health.value,
                    record.version,
                    json.dumps(record.config) if record.config else None,
                    json.dumps(record.endpoints) if record.endpoints else None,
                    json.dumps(record.capabilities) if record.capabilities else None,
                    json.dumps(record.dependencies) if record.dependencies else None,
                    json.dumps(record.resources) if record.resources else None,
                    record.last_heartbeat.isoformat() if record.last_heartbeat else None,
                    record.started_at.isoformat() if record.started_at else None,
                    record.stopped_at.isoformat() if record.stopped_at else None,
                    record.restart_count,
                    record.failure_count,
                    record.recovery_attempts,
                    json.dumps(record.metadata) if record.metadata else None,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat()
                ))

        # Cache the component record
        self._component_registry[component_id] = record

        logger.info(f"Registered component: {component_id} ({component_type.value})")
        return component_id

    async def update_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        health: Optional[ComponentHealth] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update component status and health.

        Args:
            component_id: Component identifier
            status: New component status
            health: New health status (optional)
            metadata: Additional metadata

        Returns:
            True if update successful, False otherwise
        """
        try:
            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    # Get current record
                    cursor = connection.execute(
                        "SELECT * FROM component_registry WHERE component_id = ?",
                        (component_id,)
                    )
                    row = cursor.fetchone()
                    if not row:
                        logger.error(f"Component not found: {component_id}")
                        return False

                    # Update fields
                    update_fields = {
                        "status": status.value,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }

                    if health is not None:
                        update_fields["health"] = health.value

                    if metadata is not None:
                        update_fields["metadata"] = json.dumps(metadata)

                    if status == ComponentStatus.STOPPED:
                        update_fields["stopped_at"] = datetime.now(timezone.utc).isoformat()

                    # Build dynamic update query
                    set_clause = ", ".join([f"{k} = ?" for k in update_fields.keys()])
                    values = list(update_fields.values()) + [component_id]

                    connection.execute(
                        f"UPDATE component_registry SET {set_clause} WHERE component_id = ?",
                        values
                    )

            # Update cache
            if component_id in self._component_registry:
                record = self._component_registry[component_id]
                record.status = status
                if health is not None:
                    record.health = health
                record.updated_at = datetime.now(timezone.utc)

            logger.debug(f"Updated component status: {component_id} -> {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update component status: {e}")
            return False

    async def record_heartbeat(self, component_id: str) -> bool:
        """
        Record a heartbeat for a component.

        Args:
            component_id: Component identifier

        Returns:
            True if heartbeat recorded successfully, False otherwise
        """
        try:
            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    now = datetime.now(timezone.utc)
                    connection.execute("""
                        UPDATE component_registry
                        SET last_heartbeat = ?, updated_at = ?
                        WHERE component_id = ?
                    """, (now.isoformat(), now.isoformat(), component_id))

            # Update cache
            if component_id in self._component_registry:
                self._component_registry[component_id].last_heartbeat = now

            return True

        except Exception as e:
            logger.error(f"Failed to record heartbeat for {component_id}: {e}")
            return False

    async def update_component_health(
        self,
        component_id: str,
        health_status: ComponentHealth,
        metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Update component health status and optionally record metrics.

        Args:
            component_id: Component identifier
            health_status: New health status
            metrics: Health metrics to record

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Update health status
            await self.update_component_status(component_id, None, health_status)

            # Record metrics if provided
            if metrics:
                for metric_name, metric_value in metrics.items():
                    await self._record_health_metric(
                        component_id=component_id,
                        metric_name=metric_name,
                        metric_value=metric_value,
                        metric_unit=self._get_metric_unit(metric_name)
                    )

            return True

        except Exception as e:
            logger.error(f"Failed to update component health for {component_id}: {e}")
            return False

    async def _record_health_metric(
        self,
        component_id: str,
        metric_name: str,
        metric_value: float,
        metric_unit: str,
        threshold_warning: Optional[float] = None,
        threshold_critical: Optional[float] = None
    ) -> str:
        """Record a health metric for a component."""
        metric_id = str(uuid.uuid4())

        # Check for alerts
        is_alert = False
        alert_level = None

        if threshold_critical and metric_value >= threshold_critical:
            is_alert = True
            alert_level = "critical"
        elif threshold_warning and metric_value >= threshold_warning:
            is_alert = True
            alert_level = "warning"

        metric = HealthMetric(
            metric_id=metric_id,
            component_id=component_id,
            metric_name=metric_name,
            metric_value=metric_value,
            metric_unit=metric_unit,
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            is_alert=is_alert,
            alert_level=alert_level
        )

        with self._lock:
            connection = self.connection
            with DatabaseTransaction(connection):
                connection.execute("""
                    INSERT INTO component_health_metrics (
                        metric_id, component_id, metric_name, metric_value, metric_unit,
                        threshold_warning, threshold_critical, is_alert, alert_level,
                        recorded_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.metric_id,
                    metric.component_id,
                    metric.metric_name,
                    metric.metric_value,
                    metric.metric_unit,
                    metric.threshold_warning,
                    metric.threshold_critical,
                    metric.is_alert,
                    metric.alert_level,
                    metric.recorded_at.isoformat(),
                    json.dumps(metric.metadata) if metric.metadata else None
                ))

        return metric_id

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get the appropriate unit for a metric name."""
        unit_map = {
            "cpu_usage": "percent",
            "memory_mb": "megabytes",
            "memory_usage": "percent",
            "disk_usage": "percent",
            "disk_io": "bytes_per_sec",
            "network_io": "bytes_per_sec",
            "response_time": "milliseconds",
            "error_rate": "percent",
            "queue_size": "count",
            "active_connections": "count"
        }
        return unit_map.get(metric_name, "value")

    async def get_component_status(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get component status information.

        Args:
            component_id: Specific component ID (optional, returns all if None)

        Returns:
            Component status information
        """
        try:
            with self._lock:
                connection = self.connection

                if component_id:
                    cursor = connection.execute(
                        "SELECT * FROM component_registry WHERE component_id = ?",
                        (component_id,)
                    )
                    rows = cursor.fetchall()
                else:
                    cursor = connection.execute("SELECT * FROM component_registry")
                    rows = cursor.fetchall()

                components = []
                for row in rows:
                    component_data = {
                        "component_id": row[0],
                        "component_type": row[1],
                        "instance_id": row[2],
                        "status": row[3],
                        "health": row[4],
                        "version": row[5],
                        "config": json.loads(row[6]) if row[6] else None,
                        "endpoints": json.loads(row[7]) if row[7] else None,
                        "capabilities": json.loads(row[8]) if row[8] else [],
                        "dependencies": json.loads(row[9]) if row[9] else [],
                        "resources": json.loads(row[10]) if row[10] else None,
                        "last_heartbeat": row[11],
                        "started_at": row[12],
                        "stopped_at": row[13],
                        "restart_count": row[14],
                        "failure_count": row[15],
                        "recovery_attempts": row[16],
                        "metadata": json.loads(row[17]) if row[17] else None,
                        "created_at": row[18],
                        "updated_at": row[19]
                    }
                    components.append(component_data)

                if component_id:
                    return components[0] if components else {}
                else:
                    return {"components": components, "total_count": len(components)}

        except Exception as e:
            logger.error(f"Failed to get component status: {e}")
            return {"error": str(e)}

    async def enqueue_processing_item(
        self,
        component_id: str,
        queue_type: ProcessingQueueType,
        payload: Dict[str, Any],
        priority: int = 5,
        dependencies: Optional[List[str]] = None,
        timeout_seconds: Optional[int] = None,
        scheduled_at: Optional[datetime] = None
    ) -> str:
        """
        Add an item to a component's processing queue.

        Args:
            component_id: Target component identifier
            queue_type: Type of processing queue
            payload: Processing payload data
            priority: Priority level (1=highest, 10=lowest)
            dependencies: List of dependency queue item IDs
            timeout_seconds: Processing timeout
            scheduled_at: Scheduled processing time

        Returns:
            Queue item ID
        """
        queue_item_id = str(uuid.uuid4())

        item = ComponentQueueItem(
            queue_item_id=queue_item_id,
            component_id=component_id,
            queue_type=queue_type,
            priority=priority,
            payload=payload,
            status="pending",
            dependencies=dependencies,
            timeout_seconds=timeout_seconds,
            scheduled_at=scheduled_at
        )

        with self._lock:
            connection = self.connection
            with DatabaseTransaction(connection):
                connection.execute("""
                    INSERT INTO component_processing_queue (
                        queue_item_id, component_id, queue_type, priority, payload,
                        status, assigned_worker, processing_started, processing_completed,
                        retry_count, max_retries, error_message, dependencies,
                        timeout_seconds, scheduled_at, metadata, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.queue_item_id,
                    item.component_id,
                    item.queue_type.value,
                    item.priority,
                    json.dumps(item.payload),
                    item.status,
                    item.assigned_worker,
                    item.processing_started.isoformat() if item.processing_started else None,
                    item.processing_completed.isoformat() if item.processing_completed else None,
                    item.retry_count,
                    item.max_retries,
                    item.error_message,
                    json.dumps(item.dependencies) if item.dependencies else None,
                    item.timeout_seconds,
                    item.scheduled_at.isoformat() if item.scheduled_at else None,
                    json.dumps(item.metadata) if item.metadata else None,
                    item.created_at.isoformat()
                ))

        logger.debug(f"Enqueued processing item: {queue_item_id} for {component_id}")
        return queue_item_id

    async def get_next_queue_item(
        self,
        component_id: str,
        queue_type: Optional[ProcessingQueueType] = None,
        worker_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next item from a component's processing queue.

        Args:
            component_id: Component identifier
            queue_type: Specific queue type (optional)
            worker_id: Worker identifier to assign the item

        Returns:
            Next queue item or None if queue is empty
        """
        try:
            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    # Build query based on parameters
                    where_clause = "component_id = ? AND status = 'pending'"
                    params = [component_id]

                    if queue_type:
                        where_clause += " AND queue_type = ?"
                        params.append(queue_type.value)

                    # Check scheduled time
                    now = datetime.now(timezone.utc).isoformat()
                    where_clause += " AND (scheduled_at IS NULL OR scheduled_at <= ?)"
                    params.append(now)

                    # Order by priority, then created_at
                    query = f"""
                        SELECT * FROM component_processing_queue
                        WHERE {where_clause}
                        ORDER BY priority ASC, created_at ASC
                        LIMIT 1
                    """

                    cursor = connection.execute(query, params)
                    row = cursor.fetchone()

                    if not row:
                        return None

                    # Mark as processing if worker_id provided
                    processing_started_time = None
                    if worker_id:
                        queue_item_id = row[0]
                        processing_started_time = datetime.now(timezone.utc).isoformat()

                        connection.execute("""
                            UPDATE component_processing_queue
                            SET status = 'processing', assigned_worker = ?, processing_started = ?
                            WHERE queue_item_id = ?
                        """, (worker_id, processing_started_time, queue_item_id))

                    # Convert row to dictionary (with updated values if worker_id was provided)
                    queue_item = {
                        "queue_item_id": row[0],
                        "component_id": row[1],
                        "queue_type": row[2],
                        "priority": row[3],
                        "payload": json.loads(row[4]),
                        "status": "processing" if worker_id else row[5],
                        "assigned_worker": worker_id if worker_id else row[6],
                        "processing_started": processing_started_time if worker_id else row[7],
                        "processing_completed": row[8],
                        "retry_count": row[9],
                        "max_retries": row[10],
                        "error_message": row[11],
                        "dependencies": json.loads(row[12]) if row[12] else None,
                        "timeout_seconds": row[13],
                        "scheduled_at": row[14],
                        "metadata": json.loads(row[15]) if row[15] else None,
                        "created_at": row[16]
                    }

                    return queue_item

        except Exception as e:
            logger.error(f"Failed to get next queue item for {component_id}: {e}")
            return None

    async def complete_queue_item(
        self,
        queue_item_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
        result_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Mark a queue item as completed.

        Args:
            queue_item_id: Queue item identifier
            success: Whether processing was successful
            error_message: Error message if failed
            result_metadata: Additional result metadata

        Returns:
            True if update successful, False otherwise
        """
        try:
            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    status = "completed" if success else "failed"
                    completed_at = datetime.now(timezone.utc).isoformat()

                    connection.execute("""
                        UPDATE component_processing_queue
                        SET status = ?, processing_completed = ?, error_message = ?, metadata = ?
                        WHERE queue_item_id = ?
                    """, (
                        status,
                        completed_at,
                        error_message,
                        json.dumps(result_metadata) if result_metadata else None,
                        queue_item_id
                    ))

            logger.debug(f"Completed queue item: {queue_item_id} (success: {success})")
            return True

        except Exception as e:
            logger.error(f"Failed to complete queue item {queue_item_id}: {e}")
            return False

    async def get_queue_status(
        self,
        component_id: Optional[str] = None,
        queue_type: Optional[ProcessingQueueType] = None
    ) -> Dict[str, Any]:
        """
        Get processing queue status information.

        Args:
            component_id: Specific component (optional)
            queue_type: Specific queue type (optional)

        Returns:
            Queue status information
        """
        try:
            with self._lock:
                connection = self.connection

                # Build where clause
                where_parts = []
                params = []

                if component_id:
                    where_parts.append("component_id = ?")
                    params.append(component_id)

                if queue_type:
                    where_parts.append("queue_type = ?")
                    params.append(queue_type.value)

                where_clause = " WHERE " + " AND ".join(where_parts) if where_parts else ""

                # Get status counts
                cursor = connection.execute(f"""
                    SELECT status, COUNT(*) as count
                    FROM component_processing_queue{where_clause}
                    GROUP BY status
                """, params)

                status_counts = {row[0]: row[1] for row in cursor.fetchall()}

                # Get queue type breakdown
                cursor = connection.execute(f"""
                    SELECT queue_type, status, COUNT(*) as count
                    FROM component_processing_queue{where_clause}
                    GROUP BY queue_type, status
                """, params)

                queue_breakdown = {}
                for row in cursor.fetchall():
                    queue_name = row[0]
                    status = row[1]
                    count = row[2]

                    if queue_name not in queue_breakdown:
                        queue_breakdown[queue_name] = {}
                    queue_breakdown[queue_name][status] = count

                return {
                    "status_counts": status_counts,
                    "queue_breakdown": queue_breakdown,
                    "total_items": sum(status_counts.values())
                }

        except Exception as e:
            logger.error(f"Failed to get queue status: {e}")
            return {"error": str(e)}

    async def _start_coordination_tasks(self):
        """Start background coordination tasks."""
        self._coordination_tasks = [
            asyncio.create_task(self._health_monitoring_loop()),
            asyncio.create_task(self._queue_cleanup_loop()),
            asyncio.create_task(self._component_timeout_monitor())
        ]
        logger.info("Started component coordination background tasks")

    async def _health_monitoring_loop(self):
        """Background task for health monitoring."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_component_health()
                await asyncio.sleep(self.HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(30)  # Brief pause on error

    async def _queue_cleanup_loop(self):
        """Background task for queue cleanup."""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_queue_items()
                await asyncio.sleep(self.QUEUE_CLEANUP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue cleanup loop: {e}")
                await asyncio.sleep(60)  # Brief pause on error

    async def _component_timeout_monitor(self):
        """Background task for monitoring component timeouts."""
        while not self._shutdown_event.is_set():
            try:
                await self._check_component_timeouts()
                await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in component timeout monitor: {e}")
                await asyncio.sleep(30)  # Brief pause on error

    async def _check_component_health(self):
        """Check health status of all registered components."""
        try:
            components = await self.get_component_status()

            if "components" in components:
                for component in components["components"]:
                    component_id = component["component_id"]
                    last_heartbeat = component.get("last_heartbeat")

                    if last_heartbeat:
                        last_heartbeat_dt = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                        time_since_heartbeat = datetime.now(timezone.utc) - last_heartbeat_dt

                        # Update health based on heartbeat age
                        if time_since_heartbeat.total_seconds() > self.COMPONENT_TIMEOUT:
                            await self.update_component_status(
                                component_id,
                                ComponentStatus.UNHEALTHY,
                                ComponentHealth.CRITICAL
                            )
                        elif time_since_heartbeat.total_seconds() > self.COMPONENT_TIMEOUT / 2:
                            await self.update_component_status(
                                component_id,
                                ComponentStatus.DEGRADED,
                                ComponentHealth.WARNING
                            )

        except Exception as e:
            logger.error(f"Error checking component health: {e}")

    async def _cleanup_old_queue_items(self):
        """Clean up old completed or failed queue items."""
        try:
            # Remove items older than 24 hours
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)

            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    connection.execute("""
                        DELETE FROM component_processing_queue
                        WHERE (status = 'completed' OR status = 'failed')
                        AND processing_completed < ?
                    """, (cutoff_time.isoformat(),))

                    deleted_count = connection.total_changes

            if deleted_count > 0:
                logger.debug(f"Cleaned up {deleted_count} old queue items")

        except Exception as e:
            logger.error(f"Error cleaning up old queue items: {e}")

    async def _check_component_timeouts(self):
        """Check for component timeouts and handle recovery."""
        try:
            components = await self.get_component_status()

            if "components" in components:
                for component in components["components"]:
                    component_id = component["component_id"]
                    status = component["status"]
                    last_heartbeat = component.get("last_heartbeat")

                    # Skip if component is already stopped or shutting down
                    if status in [ComponentStatus.STOPPED.value, ComponentStatus.SHUTTING_DOWN.value]:
                        continue

                    if last_heartbeat:
                        last_heartbeat_dt = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                        time_since_heartbeat = datetime.now(timezone.utc) - last_heartbeat_dt

                        # Mark as failed if timeout exceeded
                        if time_since_heartbeat.total_seconds() > self.COMPONENT_TIMEOUT:
                            await self.update_component_status(
                                component_id,
                                ComponentStatus.FAILED,
                                ComponentHealth.CRITICAL
                            )

                            # Log failure for potential recovery
                            await self._log_component_failure(
                                component_id,
                                "heartbeat_timeout",
                                {"timeout_seconds": self.COMPONENT_TIMEOUT}
                            )

        except Exception as e:
            logger.error(f"Error checking component timeouts: {e}")

    async def _log_component_failure(
        self,
        component_id: str,
        failure_type: str,
        failure_details: Dict[str, Any]
    ):
        """Log a component failure for recovery tracking."""
        try:
            recovery_id = str(uuid.uuid4())

            with self._lock:
                connection = self.connection
                with DatabaseTransaction(connection):
                    connection.execute("""
                        INSERT INTO component_recovery_log (
                            recovery_id, component_id, failure_type, failure_details,
                            recovery_action, recovery_result, recovery_duration_ms,
                            automatic_recovery, recovery_metadata, failed_at,
                            recovered_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        recovery_id,
                        component_id,
                        failure_type,
                        json.dumps(failure_details),
                        "pending",  # recovery_action
                        "pending",  # recovery_result
                        None,  # recovery_duration_ms
                        True,  # automatic_recovery
                        None,  # recovery_metadata
                        datetime.now(timezone.utc).isoformat(),  # failed_at
                        None,  # recovered_at
                        datetime.now(timezone.utc).isoformat()  # created_at
                    ))

            logger.warning(f"Logged component failure: {component_id} ({failure_type})")

        except Exception as e:
            logger.error(f"Failed to log component failure: {e}")

    async def close(self):
        """Close the component coordinator and cleanup resources."""
        # Signal shutdown
        self._shutdown_event.set()

        # Cancel coordination tasks
        for task in self._coordination_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self._coordination_tasks:
            await asyncio.gather(*self._coordination_tasks, return_exceptions=True)

        # Close base state manager
        await super().close()

        logger.info("Component coordination system closed")


# Global component coordinator instance
_component_coordinator: Optional[ComponentCoordinator] = None


async def get_component_coordinator(db_path: str = "workspace_state.db") -> ComponentCoordinator:
    """Get or create global component coordinator instance."""
    global _component_coordinator

    if _component_coordinator is None:
        _component_coordinator = ComponentCoordinator(db_path)

        if not await _component_coordinator.initialize():
            raise RuntimeError("Failed to initialize component coordinator")

    return _component_coordinator


async def shutdown_component_coordinator():
    """Shutdown global component coordinator."""
    global _component_coordinator

    if _component_coordinator:
        await _component_coordinator.close()
        _component_coordinator = None