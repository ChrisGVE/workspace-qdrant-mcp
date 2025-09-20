"""
Integration Tests for Component Coordination System.

This module provides integration tests that verify the complete
component coordination system works end-to-end with real scenarios.
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

from common.core.component_coordination import (
    ComponentCoordinator,
    ComponentType,
    ComponentStatus,
    ComponentHealth,
    ProcessingQueueType,
    get_component_coordinator,
    shutdown_component_coordinator
)
from common.core.component_migration import ComponentMigrator


class TestComponentIntegration:
    """Integration tests for component coordination system."""

    @pytest.mark.asyncio
    async def test_four_component_scenario(self):
        """Test complete four-component coordination scenario."""
        # Use temporary database
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        try:
            # Initialize coordinator
            coordinator = ComponentCoordinator(db_path)
            await coordinator.initialize()

            # Register all four components
            daemon_id = await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id="daemon-001",
                config={"grpc_port": 50051, "max_workers": 4},
                endpoints={"grpc": "localhost:50051"},
                capabilities=["file_processing", "lsp_integration", "embedding_generation"],
                version="0.3.0"
            )

            mcp_id = await coordinator.register_component(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                instance_id="mcp-001",
                config={"host": "localhost", "port": 8000},
                endpoints={"http": "localhost:8000"},
                capabilities=["search_interface", "memory_management", "query_optimization"],
                dependencies=["rust_daemon-daemon-001"],
                version="0.3.0"
            )

            cli_id = await coordinator.register_component(
                component_type=ComponentType.CLI_UTILITY,
                instance_id="cli-001",
                config={"log_level": "INFO"},
                capabilities=["system_admin", "collection_management", "daemon_control"],
                version="0.3.0"
            )

            injector_id = await coordinator.register_component(
                component_type=ComponentType.CONTEXT_INJECTOR,
                instance_id="injector-001",
                config={"memory_collection": "workspace-memory"},
                endpoints={"hook": "memory://workspace-memory"},
                capabilities=["rule_injection", "context_management"],
                dependencies=["python_mcp_server-mcp-001"],
                version="0.3.0"
            )

            # Update all components to healthy
            await coordinator.update_component_status(daemon_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)
            await coordinator.update_component_status(mcp_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)
            await coordinator.update_component_status(cli_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)
            await coordinator.update_component_status(injector_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)

            # Record heartbeats
            await coordinator.record_heartbeat(daemon_id)
            await coordinator.record_heartbeat(mcp_id)
            await coordinator.record_heartbeat(cli_id)
            await coordinator.record_heartbeat(injector_id)

            # Add health metrics for daemon
            await coordinator.update_component_health(
                daemon_id,
                ComponentHealth.HEALTHY,
                metrics={
                    "cpu_usage": 15.5,
                    "memory_mb": 256.0,
                    "active_connections": 3.0,
                    "queue_size": 12.0
                }
            )

            # Add processing queue items for different components
            file_item_id = await coordinator.enqueue_processing_item(
                component_id=daemon_id,
                queue_type=ProcessingQueueType.FILE_INGESTION,
                payload={"file_path": "/project/src/main.py", "collection": "project-code"},
                priority=2
            )

            search_item_id = await coordinator.enqueue_processing_item(
                component_id=mcp_id,
                queue_type=ProcessingQueueType.SEARCH_REQUESTS,
                payload={"query": "authentication patterns", "collection": "project-code"},
                priority=1
            )

            rule_item_id = await coordinator.enqueue_processing_item(
                component_id=injector_id,
                queue_type=ProcessingQueueType.RULE_UPDATES,
                payload={"rule": "Always use TypeScript strict mode", "authority": "default"},
                priority=3
            )

            # Get status of all components
            all_status = await coordinator.get_component_status()
            assert all_status["total_count"] == 4

            # Verify component types are represented
            component_types = [c["component_type"] for c in all_status["components"]]
            expected_types = [t.value for t in ComponentType]
            for expected_type in expected_types:
                assert expected_type in component_types

            # Process queue items
            # Daemon processes file ingestion
            daemon_item = await coordinator.get_next_queue_item(
                component_id=daemon_id,
                queue_type=ProcessingQueueType.FILE_INGESTION,
                worker_id="daemon-worker-001"
            )
            assert daemon_item is not None
            assert daemon_item["payload"]["file_path"] == "/project/src/main.py"

            await coordinator.complete_queue_item(
                queue_item_id=daemon_item["queue_item_id"],
                success=True,
                result_metadata={"symbols_extracted": 45, "processing_time_ms": 1200}
            )

            # MCP server processes search request
            mcp_item = await coordinator.get_next_queue_item(
                component_id=mcp_id,
                queue_type=ProcessingQueueType.SEARCH_REQUESTS,
                worker_id="mcp-worker-001"
            )
            assert mcp_item is not None
            assert mcp_item["payload"]["query"] == "authentication patterns"

            await coordinator.complete_queue_item(
                queue_item_id=mcp_item["queue_item_id"],
                success=True,
                result_metadata={"results_count": 15, "query_time_ms": 85}
            )

            # Context injector processes rule update
            injector_item = await coordinator.get_next_queue_item(
                component_id=injector_id,
                queue_type=ProcessingQueueType.RULE_UPDATES,
                worker_id="injector-worker-001"
            )
            assert injector_item is not None
            assert injector_item["payload"]["rule"] == "Always use TypeScript strict mode"

            await coordinator.complete_queue_item(
                queue_item_id=injector_item["queue_item_id"],
                success=True,
                result_metadata={"rule_stored": True, "memory_collection": "workspace-memory"}
            )

            # Check queue status
            queue_status = await coordinator.get_queue_status()
            assert queue_status["status_counts"]["completed"] == 3
            assert queue_status["total_items"] == 3

            # Verify different queue types were processed
            breakdown = queue_status["queue_breakdown"]
            assert ProcessingQueueType.FILE_INGESTION.value in breakdown
            assert ProcessingQueueType.SEARCH_REQUESTS.value in breakdown
            assert ProcessingQueueType.RULE_UPDATES.value in breakdown

            await coordinator.close()

        finally:
            # Cleanup
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_component_failure_recovery_scenario(self):
        """Test component failure detection and recovery scenario."""
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        try:
            coordinator = ComponentCoordinator(db_path)
            coordinator.COMPONENT_TIMEOUT = 5  # Reduce timeout for testing
            await coordinator.initialize()

            # Register a component
            daemon_id = await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id="daemon-001"
            )

            # Mark as healthy
            await coordinator.update_component_status(daemon_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)
            await coordinator.record_heartbeat(daemon_id)

            # Add work to queue
            await coordinator.enqueue_processing_item(
                component_id=daemon_id,
                queue_type=ProcessingQueueType.FILE_INGESTION,
                payload={"file": "important.py"},
                priority=1
            )

            # Simulate component failure by not sending heartbeats
            await asyncio.sleep(1)

            # Manually trigger timeout check (simulate background task)
            await coordinator._check_component_timeouts()

            # Component should still be healthy (timeout not reached)
            status = await coordinator.get_component_status(daemon_id)
            assert status["status"] != ComponentStatus.FAILED.value

            # Simulate longer failure period
            import datetime
            old_time = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)

            with coordinator._lock:
                coordinator.connection.execute("""
                    UPDATE component_registry
                    SET last_heartbeat = ?
                    WHERE component_id = ?
                """, (old_time.isoformat(), daemon_id))

            # Run timeout check again
            await coordinator._check_component_timeouts()

            # Now component should be marked as failed
            status = await coordinator.get_component_status(daemon_id)
            assert status["status"] == ComponentStatus.FAILED.value
            assert status["health"] == ComponentHealth.CRITICAL.value

            # Check that failure was logged
            with coordinator._lock:
                cursor = coordinator.connection.execute("""
                    SELECT failure_type, recovery_action
                    FROM component_recovery_log
                    WHERE component_id = ?
                """, (daemon_id,))
                failure_log = cursor.fetchone()

            assert failure_log is not None
            assert failure_log[0] == "heartbeat_timeout"

            # Simulate component recovery
            await coordinator.update_component_status(daemon_id, ComponentStatus.STARTING)
            await coordinator.record_heartbeat(daemon_id)
            await coordinator.update_component_status(daemon_id, ComponentStatus.HEALTHY, ComponentHealth.HEALTHY)

            # Verify component is back online
            status = await coordinator.get_component_status(daemon_id)
            assert status["status"] == ComponentStatus.HEALTHY.value
            assert status["health"] == ComponentHealth.HEALTHY.value

            await coordinator.close()

        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_migration_integration(self):
        """Test migration from base schema to component coordination schema."""
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        try:
            # First create a base SQLite state manager
            from common.core.sqlite_state_manager import SQLiteStateManager

            base_manager = SQLiteStateManager(db_path)
            await base_manager.initialize()

            # Add some data to base schema
            await base_manager.record_search_operation(
                query="test search",
                results_count=5,
                query_time_ms=100
            )

            await base_manager.close()

            # Test migration
            migrator = ComponentMigrator(db_path)

            # Check if migration is needed
            needs_migration = await migrator.needs_migration()
            assert needs_migration is True

            # Perform migration
            migration_success = await migrator.migrate()
            assert migration_success is True

            # Verify migration
            needs_migration_after = await migrator.needs_migration()
            assert needs_migration_after is False

            # Test that component coordinator works with migrated database
            coordinator = ComponentCoordinator(db_path)
            await coordinator.initialize()

            # Register a component
            component_id = await coordinator.register_component(
                component_type=ComponentType.CLI_UTILITY,
                instance_id="test-001"
            )

            assert component_id is not None

            # Verify original data is preserved
            with coordinator._lock:
                cursor = coordinator.connection.execute("SELECT COUNT(*) FROM search_history")
                search_count = cursor.fetchone()[0]

            assert search_count == 1  # Original search operation should be preserved

            await coordinator.close()

        finally:
            Path(db_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_global_coordinator_integration(self):
        """Test global coordinator instance management."""
        db_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        db_path = db_file.name
        db_file.close()

        try:
            # Get global coordinator instance
            coordinator1 = await get_component_coordinator(db_path)
            coordinator2 = await get_component_coordinator(db_path)

            # Should be the same instance
            assert coordinator1 is coordinator2

            # Register component using global instance
            component_id = await coordinator1.register_component(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                instance_id="global-test-001"
            )

            # Verify using other reference
            status = await coordinator2.get_component_status(component_id)
            assert status["component_id"] == component_id

            # Shutdown global instance
            await shutdown_component_coordinator()

            # Get new instance after shutdown
            coordinator3 = await get_component_coordinator(db_path)
            assert coordinator3 is not coordinator1

            # Verify data persisted
            status = await coordinator3.get_component_status(component_id)
            assert status["component_id"] == component_id

            await coordinator3.close()

        finally:
            Path(db_path).unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))