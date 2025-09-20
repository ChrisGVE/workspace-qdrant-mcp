"""
Tests for Component Coordination System.

This module tests the component coordination capabilities built on top of
the SQLite State Manager, including component registration, health monitoring,
inter-component communication, and processing queue coordination.
"""

import asyncio
import json
import pytest
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from common.core.component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ComponentStatus,
    ComponentHealth,
    CommunicationChannel,
    ProcessingQueueType,
    ComponentRecord,
    CommunicationRecord,
    HealthMetric,
    ComponentQueueItem,
    get_component_coordinator,
    shutdown_component_coordinator
)


class TestComponentCoordinator:
    """Test Component Coordinator functionality."""

    @pytest.fixture
    async def coordinator(self):
        """Create a test component coordinator."""
        # Use temporary database
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        coordinator = ComponentCoordinator(db_path)
        await coordinator.initialize()

        yield coordinator

        await coordinator.close()
        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_coordinator_initialization(self, coordinator):
        """Test component coordinator initialization."""
        assert coordinator is not None
        assert coordinator.connection is not None

        # Check that extended tables were created
        with coordinator._lock:
            cursor = coordinator.connection.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'component_%'
            """)
            tables = [row[0] for row in cursor.fetchall()]

            expected_tables = [
                'component_registry',
                'component_communication',
                'component_health_metrics',
                'component_processing_queue',
                'component_recovery_log'
            ]

            for table in expected_tables:
                assert table in tables

    @pytest.mark.asyncio
    async def test_component_registration(self, coordinator):
        """Test component registration functionality."""
        # Register a Rust daemon component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001",
            config={"grpc_port": 50051, "max_workers": 4},
            endpoints={"grpc": "localhost:50051"},
            capabilities=["file_processing", "lsp_integration"],
            dependencies=["qdrant_server"],
            version="0.3.0"
        )

        assert component_id == "rust_daemon-daemon-001"

        # Verify component was stored
        status = await coordinator.get_component_status(component_id)
        assert status["component_id"] == component_id
        assert status["component_type"] == ComponentType.RUST_DAEMON.value
        assert status["instance_id"] == "daemon-001"
        assert status["status"] == ComponentStatus.STARTING.value
        assert status["health"] == ComponentHealth.UNKNOWN.value
        assert status["version"] == "0.3.0"
        assert status["config"]["grpc_port"] == 50051
        assert status["endpoints"]["grpc"] == "localhost:50051"
        assert "file_processing" in status["capabilities"]
        assert "qdrant_server" in status["dependencies"]

    @pytest.mark.asyncio
    async def test_component_status_updates(self, coordinator):
        """Test component status and health updates."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="mcp-001"
        )

        # Update status to healthy
        success = await coordinator.update_component_status(
            component_id=component_id,
            status=ComponentStatus.HEALTHY,
            health=ComponentHealth.HEALTHY,
            metadata={"startup_time": 2.5}
        )
        assert success is True

        # Verify status update
        status = await coordinator.get_component_status(component_id)
        assert status["status"] == ComponentStatus.HEALTHY.value
        assert status["health"] == ComponentHealth.HEALTHY.value
        assert status["metadata"]["startup_time"] == 2.5

        # Update to degraded
        success = await coordinator.update_component_status(
            component_id=component_id,
            status=ComponentStatus.DEGRADED,
            health=ComponentHealth.WARNING
        )
        assert success is True

        status = await coordinator.get_component_status(component_id)
        assert status["status"] == ComponentStatus.DEGRADED.value
        assert status["health"] == ComponentHealth.WARNING.value

    @pytest.mark.asyncio
    async def test_heartbeat_recording(self, coordinator):
        """Test heartbeat recording functionality."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        # Record heartbeat
        success = await coordinator.record_heartbeat(component_id)
        assert success is True

        # Verify heartbeat was recorded
        status = await coordinator.get_component_status(component_id)
        assert status["last_heartbeat"] is not None

        # Parse heartbeat timestamp
        heartbeat_time = datetime.fromisoformat(status["last_heartbeat"].replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        time_diff = (now - heartbeat_time).total_seconds()

        # Should be very recent (within 5 seconds)
        assert time_diff < 5

    @pytest.mark.asyncio
    async def test_health_metrics_recording(self, coordinator):
        """Test health metrics recording."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Update health with metrics
        success = await coordinator.update_component_health(
            component_id=component_id,
            health_status=ComponentHealth.HEALTHY,
            metrics={
                "cpu_usage": 25.5,
                "memory_mb": 128.0,
                "active_connections": 5.0
            }
        )
        assert success is True

        # Verify health status was updated
        status = await coordinator.get_component_status(component_id)
        assert status["health"] == ComponentHealth.HEALTHY.value

        # Check that metrics were recorded
        with coordinator._lock:
            cursor = coordinator.connection.execute("""
                SELECT metric_name, metric_value, metric_unit
                FROM component_health_metrics
                WHERE component_id = ?
                ORDER BY metric_name
            """, (component_id,))
            metrics = cursor.fetchall()

        assert len(metrics) == 3
        metric_dict = {row[0]: (row[1], row[2]) for row in metrics}

        assert metric_dict["active_connections"] == (5.0, "count")
        assert metric_dict["cpu_usage"] == (25.5, "percent")
        assert metric_dict["memory_mb"] == (128.0, "megabytes")

    @pytest.mark.asyncio
    async def test_processing_queue_operations(self, coordinator):
        """Test processing queue operations."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Enqueue processing items
        queue_item_id_1 = await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file_path": "/test/file1.txt", "collection": "test-project"},
            priority=1  # High priority
        )

        queue_item_id_2 = await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file_path": "/test/file2.txt", "collection": "test-project"},
            priority=5  # Normal priority
        )

        assert queue_item_id_1 is not None
        assert queue_item_id_2 is not None

        # Get next item (should be highest priority first)
        next_item = await coordinator.get_next_queue_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            worker_id="worker-001"
        )

        assert next_item is not None
        assert next_item["queue_item_id"] == queue_item_id_1  # Higher priority
        assert next_item["priority"] == 1
        assert next_item["payload"]["file_path"] == "/test/file1.txt"
        assert next_item["status"] == "processing"
        assert next_item["assigned_worker"] == "worker-001"

        # Complete the item
        success = await coordinator.complete_queue_item(
            queue_item_id=queue_item_id_1,
            success=True,
            result_metadata={"processing_time_ms": 1500}
        )
        assert success is True

        # Get next item (should be the second one now)
        next_item = await coordinator.get_next_queue_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION
        )

        assert next_item["queue_item_id"] == queue_item_id_2
        assert next_item["priority"] == 5

    @pytest.mark.asyncio
    async def test_queue_status_reporting(self, coordinator):
        """Test queue status reporting."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Enqueue various items
        await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file": "test1.txt"},
            priority=1
        )

        await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.SEARCH_REQUESTS,
            payload={"query": "test"},
            priority=2
        )

        # Get overall queue status
        queue_status = await coordinator.get_queue_status(component_id=component_id)

        assert "status_counts" in queue_status
        assert "queue_breakdown" in queue_status
        assert queue_status["total_items"] == 2
        assert queue_status["status_counts"]["pending"] == 2

        # Check queue type breakdown
        breakdown = queue_status["queue_breakdown"]
        assert ProcessingQueueType.FILE_INGESTION.value in breakdown
        assert ProcessingQueueType.SEARCH_REQUESTS.value in breakdown
        assert breakdown[ProcessingQueueType.FILE_INGESTION.value]["pending"] == 1
        assert breakdown[ProcessingQueueType.SEARCH_REQUESTS.value]["pending"] == 1

    @pytest.mark.asyncio
    async def test_multiple_component_status(self, coordinator):
        """Test getting status for multiple components."""
        # Register multiple components
        daemon_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001",
            version="0.3.0"
        )

        mcp_id = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="mcp-001",
            version="0.3.0"
        )

        cli_id = await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001",
            version="0.3.0"
        )

        # Update statuses
        await coordinator.update_component_status(daemon_id, ComponentStatus.HEALTHY)
        await coordinator.update_component_status(mcp_id, ComponentStatus.HEALTHY)
        await coordinator.update_component_status(cli_id, ComponentStatus.DEGRADED)

        # Get all component status
        all_status = await coordinator.get_component_status()

        assert "components" in all_status
        assert all_status["total_count"] == 3

        components = all_status["components"]
        component_ids = [c["component_id"] for c in components]

        assert daemon_id in component_ids
        assert mcp_id in component_ids
        assert cli_id in component_ids

        # Check specific statuses
        daemon_component = next(c for c in components if c["component_id"] == daemon_id)
        assert daemon_component["status"] == ComponentStatus.HEALTHY.value

        cli_component = next(c for c in components if c["component_id"] == cli_id)
        assert cli_component["status"] == ComponentStatus.DEGRADED.value

    @pytest.mark.asyncio
    async def test_component_failure_detection(self, coordinator):
        """Test component failure detection and timeout monitoring."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.CONTEXT_INJECTOR,
            instance_id="injector-001"
        )

        # Record initial heartbeat
        await coordinator.record_heartbeat(component_id)

        # Simulate old heartbeat (component timeout)
        old_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=coordinator.COMPONENT_TIMEOUT + 10)

        with coordinator._lock:
            coordinator.connection.execute("""
                UPDATE component_registry
                SET last_heartbeat = ?
                WHERE component_id = ?
            """, (old_heartbeat.isoformat(), component_id))

        # Run timeout check
        await coordinator._check_component_timeouts()

        # Verify component was marked as failed
        status = await coordinator.get_component_status(component_id)
        assert status["status"] == ComponentStatus.FAILED.value
        assert status["health"] == ComponentHealth.CRITICAL.value

        # Check that failure was logged
        with coordinator._lock:
            cursor = coordinator.connection.execute("""
                SELECT failure_type, recovery_action
                FROM component_recovery_log
                WHERE component_id = ?
            """, (component_id,))
            failure_log = cursor.fetchone()

        assert failure_log is not None
        assert failure_log[0] == "heartbeat_timeout"
        assert failure_log[1] == "pending"

    @pytest.mark.asyncio
    async def test_queue_item_dependencies(self, coordinator):
        """Test queue item dependencies and scheduling."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Enqueue item with future scheduled time
        future_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        future_item_id = await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file": "future.txt"},
            priority=1,
            scheduled_at=future_time
        )

        # Enqueue item with dependencies
        dependency_item_id = await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file": "dependent.txt"},
            priority=1,
            dependencies=[future_item_id]
        )

        # Should not get future scheduled item
        next_item = await coordinator.get_next_queue_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION
        )

        # Should get None since future item is not ready and dependent item has dependencies
        assert next_item is None

    @pytest.mark.asyncio
    async def test_queue_cleanup(self, coordinator):
        """Test queue cleanup of old completed items."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Enqueue and complete an item
        queue_item_id = await coordinator.enqueue_processing_item(
            component_id=component_id,
            queue_type=ProcessingQueueType.FILE_INGESTION,
            payload={"file": "test.txt"},
            priority=1
        )

        await coordinator.complete_queue_item(queue_item_id, success=True)

        # Simulate old completion time
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)

        with coordinator._lock:
            coordinator.connection.execute("""
                UPDATE component_processing_queue
                SET processing_completed = ?
                WHERE queue_item_id = ?
            """, (old_time.isoformat(), queue_item_id))

        # Run cleanup
        await coordinator._cleanup_old_queue_items()

        # Verify item was cleaned up
        with coordinator._lock:
            cursor = coordinator.connection.execute("""
                SELECT COUNT(*) FROM component_processing_queue
                WHERE queue_item_id = ?
            """, (queue_item_id,))
            count = cursor.fetchone()[0]

        assert count == 0

    @pytest.mark.asyncio
    async def test_health_monitoring_loop(self, coordinator):
        """Test health monitoring background loop."""
        # Register component
        component_id = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Set old heartbeat (degraded)
        old_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=coordinator.COMPONENT_TIMEOUT / 2 + 10)

        with coordinator._lock:
            coordinator.connection.execute("""
                UPDATE component_registry
                SET last_heartbeat = ?
                WHERE component_id = ?
            """, (old_heartbeat.isoformat(), component_id))

        # Run health check
        await coordinator._check_component_health()

        # Verify component was marked as degraded
        status = await coordinator.get_component_status(component_id)
        assert status["status"] == ComponentStatus.DEGRADED.value
        assert status["health"] == ComponentHealth.WARNING.value

    @pytest.mark.asyncio
    async def test_component_coordination_concurrent_access(self, coordinator):
        """Test concurrent access to component coordination."""
        # Register multiple components concurrently
        async def register_component(instance_num):
            return await coordinator.register_component(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                instance_id=f"mcp-{instance_num:03d}"
            )

        # Register 10 components concurrently
        tasks = [register_component(i) for i in range(10)]
        component_ids = await asyncio.gather(*tasks)

        assert len(component_ids) == 10
        assert len(set(component_ids)) == 10  # All unique

        # Update all components concurrently
        async def update_component(component_id):
            return await coordinator.update_component_status(
                component_id=component_id,
                status=ComponentStatus.HEALTHY,
                health=ComponentHealth.HEALTHY
            )

        update_tasks = [update_component(cid) for cid in component_ids]
        results = await asyncio.gather(*update_tasks)

        assert all(results)  # All updates successful

        # Verify all components are healthy
        all_status = await coordinator.get_component_status()
        for component in all_status["components"]:
            assert component["status"] == ComponentStatus.HEALTHY.value
            assert component["health"] == ComponentHealth.HEALTHY.value

    @pytest.mark.asyncio
    async def test_global_coordinator_instance(self):
        """Test global coordinator instance management."""
        # Test getting global instance
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        try:
            coordinator1 = await get_component_coordinator(db_path)
            coordinator2 = await get_component_coordinator(db_path)

            # Should be the same instance
            assert coordinator1 is coordinator2

            # Test shutdown
            await shutdown_component_coordinator()

            # Should get new instance after shutdown
            coordinator3 = await get_component_coordinator(db_path)
            assert coordinator3 is not coordinator1

            await coordinator3.close()

        finally:
            # Cleanup
            Path(db_path).unlink(missing_ok=True)


class TestComponentDataStructures:
    """Test component coordination data structures."""

    def test_component_record_creation(self):
        """Test ComponentRecord creation and defaults."""
        record = ComponentRecord(
            component_id="test-component",
            component_type=ComponentType.RUST_DAEMON,
            instance_id="test-001",
            status=ComponentStatus.STARTING,
            health=ComponentHealth.UNKNOWN
        )

        assert record.component_id == "test-component"
        assert record.component_type == ComponentType.RUST_DAEMON
        assert record.instance_id == "test-001"
        assert record.status == ComponentStatus.STARTING
        assert record.health == ComponentHealth.UNKNOWN
        assert record.restart_count == 0
        assert record.failure_count == 0
        assert record.recovery_attempts == 0
        assert record.created_at is not None
        assert record.updated_at is not None

    def test_communication_record_creation(self):
        """Test CommunicationRecord creation and defaults."""
        record = CommunicationRecord(
            communication_id="comm-001",
            source_component="rust_daemon-001",
            target_component="python_mcp_server-001",
            channel=CommunicationChannel.GRPC,
            message_type="search_request",
            status="pending"
        )

        assert record.communication_id == "comm-001"
        assert record.source_component == "rust_daemon-001"
        assert record.target_component == "python_mcp_server-001"
        assert record.channel == CommunicationChannel.GRPC
        assert record.message_type == "search_request"
        assert record.status == "pending"
        assert record.retry_count == 0
        assert record.created_at is not None

    def test_health_metric_creation(self):
        """Test HealthMetric creation and defaults."""
        metric = HealthMetric(
            metric_id="metric-001",
            component_id="rust_daemon-001",
            metric_name="cpu_usage",
            metric_value=45.5,
            metric_unit="percent",
            threshold_warning=80.0,
            threshold_critical=95.0
        )

        assert metric.metric_id == "metric-001"
        assert metric.component_id == "rust_daemon-001"
        assert metric.metric_name == "cpu_usage"
        assert metric.metric_value == 45.5
        assert metric.metric_unit == "percent"
        assert metric.threshold_warning == 80.0
        assert metric.threshold_critical == 95.0
        assert metric.is_alert is False
        assert metric.alert_level is None
        assert metric.recorded_at is not None

    def test_component_queue_item_creation(self):
        """Test ComponentQueueItem creation and defaults."""
        item = ComponentQueueItem(
            queue_item_id="queue-001",
            component_id="rust_daemon-001",
            queue_type=ProcessingQueueType.FILE_INGESTION,
            priority=5,
            payload={"file_path": "/test/file.txt", "collection": "test"}
        )

        assert item.queue_item_id == "queue-001"
        assert item.component_id == "rust_daemon-001"
        assert item.queue_type == ProcessingQueueType.FILE_INGESTION
        assert item.priority == 5
        assert item.payload["file_path"] == "/test/file.txt"
        assert item.status == "pending"  # Default status
        assert item.retry_count == 0
        assert item.max_retries == 3
        assert item.created_at is not None
        assert item.scheduled_at is not None


class TestComponentEnums:
    """Test component coordination enums."""

    def test_component_type_enum(self):
        """Test ComponentType enum values."""
        assert ComponentType.RUST_DAEMON.value == "rust_daemon"
        assert ComponentType.PYTHON_MCP_SERVER.value == "python_mcp_server"
        assert ComponentType.CLI_UTILITY.value == "cli_utility"
        assert ComponentType.CONTEXT_INJECTOR.value == "context_injector"

    def test_component_status_enum(self):
        """Test ComponentStatus enum values."""
        assert ComponentStatus.STARTING.value == "starting"
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.DEGRADED.value == "degraded"
        assert ComponentStatus.UNHEALTHY.value == "unhealthy"
        assert ComponentStatus.SHUTTING_DOWN.value == "shutting_down"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.FAILED.value == "failed"
        assert ComponentStatus.UNKNOWN.value == "unknown"

    def test_component_health_enum(self):
        """Test ComponentHealth enum values."""
        assert ComponentHealth.HEALTHY.value == "healthy"
        assert ComponentHealth.WARNING.value == "warning"
        assert ComponentHealth.CRITICAL.value == "critical"
        assert ComponentHealth.UNKNOWN.value == "unknown"

    def test_communication_channel_enum(self):
        """Test CommunicationChannel enum values."""
        assert CommunicationChannel.GRPC.value == "grpc"
        assert CommunicationChannel.SQLITE_SHARED.value == "sqlite_shared"
        assert CommunicationChannel.SIGNAL_HANDLING.value == "signal_handling"
        assert CommunicationChannel.DIRECT_QDRANT.value == "direct_qdrant"
        assert CommunicationChannel.HOOK_STREAMING.value == "hook_streaming"

    def test_processing_queue_type_enum(self):
        """Test ProcessingQueueType enum values."""
        assert ProcessingQueueType.FILE_INGESTION.value == "file_ingestion"
        assert ProcessingQueueType.SEARCH_REQUESTS.value == "search_requests"
        assert ProcessingQueueType.RULE_UPDATES.value == "rule_updates"
        assert ProcessingQueueType.HEALTH_CHECKS.value == "health_checks"
        assert ProcessingQueueType.ADMIN_COMMANDS.value == "admin_commands"
        assert ProcessingQueueType.CONTEXT_INJECTION.value == "context_injection"


if __name__ == "__main__":
    # Run basic tests
    import sys
    sys.exit(pytest.main([__file__, "-v"]))