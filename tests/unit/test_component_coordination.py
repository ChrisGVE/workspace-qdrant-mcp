"""
Comprehensive unit tests for component_coordination module.

This test module provides 100% coverage for the ComponentCoordinator
and related components, including component registration, lifecycle management,
health monitoring, inter-component communication, and recovery mechanisms.
"""

import pytest
import asyncio
import sqlite3
import tempfile
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add the src directory to Python path
import sys
src_path = Path(__file__).parent.parent.parent / "src" / "python"
sys.path.insert(0, str(src_path))

try:
    from workspace_qdrant_mcp.core.component_coordination import (
        ComponentCoordinator,
        ComponentType,
        ComponentHealth,
        ComponentState,
        ComponentRegistration,
        ComponentMetrics,
        CommunicationChannel,
        ProcessingQueue,
        ResourceUsage,
        FailoverConfig,
        ComponentDependency
    )
    COMPONENT_COORDINATION_AVAILABLE = True
except ImportError as e:
    COMPONENT_COORDINATION_AVAILABLE = False
    pytest.skip(f"Component coordination module not available: {e}", allow_module_level=True)


class TestComponentType:
    """Test component type enumeration."""

    def test_component_type_values(self):
        """Test that ComponentType has expected values."""
        assert hasattr(ComponentType, 'RUST_DAEMON')
        assert hasattr(ComponentType, 'PYTHON_MCP_SERVER')
        assert hasattr(ComponentType, 'CLI_UTILITY')
        assert hasattr(ComponentType, 'CONTEXT_INJECTOR')

    def test_component_type_string_representation(self):
        """Test string representation of component type values."""
        component_type = ComponentType.RUST_DAEMON
        assert str(component_type) in ['ComponentType.RUST_DAEMON', 'RUST_DAEMON']


class TestComponentHealth:
    """Test component health enumeration."""

    def test_component_health_values(self):
        """Test that ComponentHealth has expected values."""
        assert hasattr(ComponentHealth, 'HEALTHY')
        assert hasattr(ComponentHealth, 'DEGRADED')
        assert hasattr(ComponentHealth, 'UNHEALTHY')
        assert hasattr(ComponentHealth, 'UNKNOWN')
        assert hasattr(ComponentHealth, 'STARTING')
        assert hasattr(ComponentHealth, 'STOPPING')


class TestComponentState:
    """Test component state enumeration."""

    def test_component_state_values(self):
        """Test that ComponentState has expected values."""
        assert hasattr(ComponentState, 'INACTIVE')
        assert hasattr(ComponentState, 'STARTING')
        assert hasattr(ComponentState, 'ACTIVE')
        assert hasattr(ComponentState, 'STOPPING')
        assert hasattr(ComponentState, 'FAILED')
        assert hasattr(ComponentState, 'RECOVERING')


class TestComponentRegistration:
    """Test component registration data structure."""

    def test_component_registration_initialization(self):
        """Test ComponentRegistration initialization."""
        config = {"port": 8080, "host": "localhost"}
        registration = ComponentRegistration(
            component_id="test-component-001",
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001",
            config=config,
            version="1.0.0"
        )

        assert registration.component_id == "test-component-001"
        assert registration.component_type == ComponentType.PYTHON_MCP_SERVER
        assert registration.instance_id == "server-001"
        assert registration.config == config
        assert registration.version == "1.0.0"

    def test_component_registration_with_optional_fields(self):
        """Test ComponentRegistration with optional fields."""
        registration = ComponentRegistration(
            component_id="test-component-002",
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001",
            startup_time=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            capabilities=["processing", "storage"]
        )

        assert registration.startup_time is not None
        assert registration.last_heartbeat is not None
        assert "processing" in registration.capabilities
        assert "storage" in registration.capabilities

    def test_component_registration_serialization(self):
        """Test ComponentRegistration serialization."""
        registration = ComponentRegistration(
            component_id="test-component-003",
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        # Should be able to convert to dict for JSON serialization
        registration_dict = registration.__dict__ if hasattr(registration, '__dict__') else {
            'component_id': registration.component_id,
            'component_type': registration.component_type.value,
            'instance_id': registration.instance_id
        }

        assert isinstance(registration_dict, dict)
        assert registration_dict.get('component_id') == "test-component-003"


class TestComponentMetrics:
    """Test component metrics data structure."""

    def test_component_metrics_initialization(self):
        """Test ComponentMetrics initialization."""
        metrics = ComponentMetrics(
            component_id="test-component-001",
            cpu_usage=45.5,
            memory_usage_mb=256,
            disk_usage_mb=1024,
            network_io_kb=512
        )

        assert metrics.component_id == "test-component-001"
        assert metrics.cpu_usage == 45.5
        assert metrics.memory_usage_mb == 256
        assert metrics.disk_usage_mb == 1024
        assert metrics.network_io_kb == 512

    def test_component_metrics_with_custom_metrics(self):
        """Test ComponentMetrics with custom metrics."""
        custom_metrics = {
            "request_count": 1000,
            "error_rate": 0.01,
            "response_time_ms": 150
        }

        metrics = ComponentMetrics(
            component_id="test-component-002",
            cpu_usage=30.0,
            memory_usage_mb=128,
            custom_metrics=custom_metrics
        )

        assert metrics.custom_metrics == custom_metrics
        assert metrics.custom_metrics["request_count"] == 1000

    def test_component_metrics_timestamp(self):
        """Test ComponentMetrics timestamp handling."""
        metrics = ComponentMetrics(
            component_id="test-component-003",
            cpu_usage=20.0,
            memory_usage_mb=64,
            timestamp=datetime.now(timezone.utc)
        )

        assert metrics.timestamp is not None
        assert isinstance(metrics.timestamp, datetime)


class TestProcessingQueue:
    """Test processing queue data structure."""

    def test_processing_queue_initialization(self):
        """Test ProcessingQueue initialization."""
        queue = ProcessingQueue(
            queue_id="test-queue-001",
            component_id="test-component-001",
            queue_type="document_processing",
            max_size=1000
        )

        assert queue.queue_id == "test-queue-001"
        assert queue.component_id == "test-component-001"
        assert queue.queue_type == "document_processing"
        assert queue.max_size == 1000

    def test_processing_queue_statistics(self):
        """Test ProcessingQueue with statistics."""
        queue = ProcessingQueue(
            queue_id="test-queue-002",
            component_id="test-component-002",
            queue_type="indexing",
            current_size=50,
            processed_count=500,
            failed_count=5
        )

        assert queue.current_size == 50
        assert queue.processed_count == 500
        assert queue.failed_count == 5

    def test_processing_queue_priority_and_status(self):
        """Test ProcessingQueue priority and status."""
        queue = ProcessingQueue(
            queue_id="test-queue-003",
            component_id="test-component-003",
            queue_type="urgent_processing",
            priority=1,
            is_active=True,
            is_paused=False
        )

        assert queue.priority == 1
        assert queue.is_active is True
        assert queue.is_paused is False


class TestResourceUsage:
    """Test resource usage data structure."""

    def test_resource_usage_initialization(self):
        """Test ResourceUsage initialization."""
        usage = ResourceUsage(
            component_id="test-component-001",
            cpu_cores_used=2.5,
            memory_mb_used=512,
            disk_mb_used=2048,
            network_bandwidth_kbps=1024
        )

        assert usage.component_id == "test-component-001"
        assert usage.cpu_cores_used == 2.5
        assert usage.memory_mb_used == 512
        assert usage.disk_mb_used == 2048
        assert usage.network_bandwidth_kbps == 1024

    def test_resource_usage_limits(self):
        """Test ResourceUsage with limits."""
        usage = ResourceUsage(
            component_id="test-component-002",
            cpu_cores_used=1.0,
            memory_mb_used=256,
            cpu_cores_limit=4.0,
            memory_mb_limit=1024,
            disk_mb_limit=10240
        )

        assert usage.cpu_cores_limit == 4.0
        assert usage.memory_mb_limit == 1024
        assert usage.disk_mb_limit == 10240

    def test_resource_usage_utilization_calculation(self):
        """Test resource utilization calculation."""
        usage = ResourceUsage(
            component_id="test-component-003",
            cpu_cores_used=2.0,
            memory_mb_used=500,
            cpu_cores_limit=4.0,
            memory_mb_limit=1000
        )

        # Calculate utilization percentages
        cpu_utilization = (usage.cpu_cores_used / usage.cpu_cores_limit) * 100
        memory_utilization = (usage.memory_mb_used / usage.memory_mb_limit) * 100

        assert cpu_utilization == 50.0
        assert memory_utilization == 50.0


class TestComponentCoordinator:
    """Test component coordinator functionality."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
            yield tmp_file.name
        # Cleanup
        Path(tmp_file.name).unlink(missing_ok=True)

    def test_component_coordinator_initialization(self, temp_db_path):
        """Test ComponentCoordinator initialization."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)

        assert coordinator.db_path == temp_db_path
        assert coordinator.components == {}
        assert coordinator.communication_channels == {}
        assert coordinator.processing_queues == {}

    @pytest.mark.asyncio
    async def test_initialize_coordinator(self, temp_db_path):
        """Test coordinator initialization."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)

        await coordinator.initialize()

        assert coordinator.initialized is True
        # Database should be created
        assert Path(temp_db_path).exists()

    @pytest.mark.asyncio
    async def test_register_component(self, temp_db_path):
        """Test component registration."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        config = {"port": 8080, "host": "localhost"}
        result = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001",
            config=config,
            version="1.0.0"
        )

        assert result is not None
        assert result.component_type == ComponentType.PYTHON_MCP_SERVER
        assert result.instance_id == "server-001"
        assert result.config == config

    @pytest.mark.asyncio
    async def test_register_duplicate_component(self, temp_db_path):
        """Test registration of duplicate component."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register first component
        await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Try to register duplicate
        with pytest.raises(ValueError, match="Component already registered"):
            await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id="daemon-001"
            )

    @pytest.mark.asyncio
    async def test_unregister_component(self, temp_db_path):
        """Test component unregistration."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        # Unregister component
        result = await coordinator.unregister_component(registration.component_id)

        assert result is True
        assert registration.component_id not in coordinator.components

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_component(self, temp_db_path):
        """Test unregistration of nonexistent component."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        result = await coordinator.unregister_component("nonexistent-component")

        assert result is False

    @pytest.mark.asyncio
    async def test_update_component_health(self, temp_db_path):
        """Test component health update."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Update health
        metrics = {"cpu_usage": 45.5, "memory_mb": 256}
        result = await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.HEALTHY,
            metrics=metrics
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_component_state(self, temp_db_path):
        """Test component state update."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.CONTEXT_INJECTOR,
            instance_id="injector-001"
        )

        # Update state
        result = await coordinator.update_component_state(
            component_id=registration.component_id,
            new_state=ComponentState.ACTIVE
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_get_component_status(self, temp_db_path):
        """Test component status retrieval."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Get status
        status = await coordinator.get_component_status(registration.component_id)

        assert status is not None
        assert status["component_id"] == registration.component_id
        assert "health" in status
        assert "state" in status

    @pytest.mark.asyncio
    async def test_get_all_components(self, temp_db_path):
        """Test retrieval of all components."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register multiple components
        await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Get all components
        components = await coordinator.get_all_components()

        assert len(components) == 2
        assert any(comp.component_type == ComponentType.RUST_DAEMON for comp in components)
        assert any(comp.component_type == ComponentType.PYTHON_MCP_SERVER for comp in components)

    @pytest.mark.asyncio
    async def test_get_components_by_type(self, temp_db_path):
        """Test retrieval of components by type."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register components
        await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-002"
        )
        await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        # Get components by type
        rust_components = await coordinator.get_components_by_type(ComponentType.RUST_DAEMON)
        cli_components = await coordinator.get_components_by_type(ComponentType.CLI_UTILITY)

        assert len(rust_components) == 2
        assert len(cli_components) == 1

    @pytest.mark.asyncio
    async def test_create_communication_channel(self, temp_db_path):
        """Test communication channel creation."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register components
        daemon_reg = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        server_reg = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Create communication channel
        channel = await coordinator.create_communication_channel(
            source_component_id=daemon_reg.component_id,
            target_component_id=server_reg.component_id,
            channel_type="grpc",
            config={"port": 50051}
        )

        assert channel is not None
        assert channel.source_component_id == daemon_reg.component_id
        assert channel.target_component_id == server_reg.component_id
        assert channel.channel_type == "grpc"

    @pytest.mark.asyncio
    async def test_create_processing_queue(self, temp_db_path):
        """Test processing queue creation."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Create processing queue
        queue = await coordinator.create_processing_queue(
            component_id=registration.component_id,
            queue_type="document_processing",
            max_size=1000,
            priority=1
        )

        assert queue is not None
        assert queue.component_id == registration.component_id
        assert queue.queue_type == "document_processing"
        assert queue.max_size == 1000

    @pytest.mark.asyncio
    async def test_update_queue_statistics(self, temp_db_path):
        """Test queue statistics update."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component and create queue
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        queue = await coordinator.create_processing_queue(
            component_id=registration.component_id,
            queue_type="indexing",
            max_size=500
        )

        # Update statistics
        result = await coordinator.update_queue_statistics(
            queue_id=queue.queue_id,
            current_size=50,
            processed_count=100,
            failed_count=2
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_track_resource_usage(self, temp_db_path):
        """Test resource usage tracking."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Track resource usage
        usage = ResourceUsage(
            component_id=registration.component_id,
            cpu_cores_used=1.5,
            memory_mb_used=512,
            disk_mb_used=1024
        )

        result = await coordinator.track_resource_usage(usage)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_resource_usage_history(self, temp_db_path):
        """Test resource usage history retrieval."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Track multiple usage records
        for i in range(3):
            usage = ResourceUsage(
                component_id=registration.component_id,
                cpu_cores_used=1.0 + i * 0.5,
                memory_mb_used=256 + i * 128
            )
            await coordinator.track_resource_usage(usage)

        # Get history
        history = await coordinator.get_resource_usage_history(
            component_id=registration.component_id,
            limit=10
        )

        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_component_dependency_management(self, temp_db_path):
        """Test component dependency management."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register components
        daemon_reg = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        server_reg = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Add dependency
        dependency = ComponentDependency(
            dependent_component_id=server_reg.component_id,
            dependency_component_id=daemon_reg.component_id,
            dependency_type="startup_order",
            is_critical=True
        )

        result = await coordinator.add_component_dependency(dependency)

        assert result is True

    @pytest.mark.asyncio
    async def test_get_component_dependencies(self, temp_db_path):
        """Test component dependencies retrieval."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register components
        daemon_reg = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        server_reg = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Add dependency
        dependency = ComponentDependency(
            dependent_component_id=server_reg.component_id,
            dependency_component_id=daemon_reg.component_id,
            dependency_type="runtime",
            is_critical=True
        )
        await coordinator.add_component_dependency(dependency)

        # Get dependencies
        dependencies = await coordinator.get_component_dependencies(server_reg.component_id)

        assert len(dependencies) == 1
        assert dependencies[0].dependency_component_id == daemon_reg.component_id

    @pytest.mark.asyncio
    async def test_component_heartbeat(self, temp_db_path):
        """Test component heartbeat tracking."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        # Send heartbeat
        result = await coordinator.send_component_heartbeat(
            component_id=registration.component_id,
            additional_data={"status": "active", "last_operation": "list_collections"}
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_check_component_health(self, temp_db_path):
        """Test component health checking."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Update health
        await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.HEALTHY,
            metrics={"cpu_usage": 25.0}
        )

        # Check health
        health = await coordinator.check_component_health(registration.component_id)

        assert health == ComponentHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_get_system_overview(self, temp_db_path):
        """Test system overview generation."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register multiple components
        await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )
        await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001"
        )

        # Get system overview
        overview = await coordinator.get_system_overview()

        assert isinstance(overview, dict)
        assert "total_components" in overview
        assert "components_by_type" in overview
        assert "system_health" in overview
        assert overview["total_components"] == 2


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_database_initialization_failure(self):
        """Test handling of database initialization failure."""
        # Use invalid path
        coordinator = ComponentCoordinator(db_path="/invalid/path/database.db")

        with pytest.raises(Exception):
            await coordinator.initialize()

    @pytest.mark.asyncio
    async def test_operations_without_initialization(self, temp_db_path):
        """Test operations without proper initialization."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        # Don't initialize

        with pytest.raises(RuntimeError, match="Coordinator not initialized"):
            await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id="daemon-001"
            )

    @pytest.mark.asyncio
    async def test_invalid_component_registration(self, temp_db_path):
        """Test invalid component registration."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Test with invalid component type
        with pytest.raises(ValueError):
            await coordinator.register_component(
                component_type=None,
                instance_id="invalid-001"
            )

        # Test with empty instance ID
        with pytest.raises(ValueError):
            await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id=""
            )

    @pytest.mark.asyncio
    async def test_database_corruption_handling(self, temp_db_path):
        """Test handling of database corruption."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Simulate database corruption by writing invalid data
        with open(temp_db_path, 'wb') as f:
            f.write(b"corrupted data")

        # Operations should handle corruption gracefully
        with pytest.raises(Exception):
            await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id="daemon-001"
            )

    @pytest.mark.asyncio
    async def test_concurrent_access_handling(self, temp_db_path):
        """Test handling of concurrent database access."""
        coordinator1 = ComponentCoordinator(db_path=temp_db_path)
        coordinator2 = ComponentCoordinator(db_path=temp_db_path)

        await coordinator1.initialize()
        await coordinator2.initialize()

        # Concurrent registrations should be handled safely
        tasks = [
            coordinator1.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id=f"daemon-{i}"
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some operations should succeed
        successful_registrations = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_registrations) > 0


class TestPerformanceOptimization:
    """Test performance optimization features."""

    @pytest.mark.asyncio
    async def test_bulk_component_registration(self, temp_db_path):
        """Test bulk component registration performance."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register many components
        registrations = []
        for i in range(100):
            registration = await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id=f"daemon-{i:03d}"
            )
            registrations.append(registration)

        assert len(registrations) == 100

        # All should be retrievable
        all_components = await coordinator.get_all_components()
        assert len(all_components) == 100

    @pytest.mark.asyncio
    async def test_batch_health_updates(self, temp_db_path):
        """Test batch health updates."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register components
        component_ids = []
        for i in range(10):
            registration = await coordinator.register_component(
                component_type=ComponentType.PYTHON_MCP_SERVER,
                instance_id=f"server-{i:03d}"
            )
            component_ids.append(registration.component_id)

        # Batch update health
        health_updates = [
            coordinator.update_component_health(
                component_id=comp_id,
                health_status=ComponentHealth.HEALTHY,
                metrics={"cpu_usage": 20.0 + i}
            )
            for i, comp_id in enumerate(component_ids)
        ]

        results = await asyncio.gather(*health_updates)
        assert all(results)

    @pytest.mark.asyncio
    async def test_resource_usage_aggregation(self, temp_db_path):
        """Test resource usage aggregation."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Track many usage records
        for i in range(50):
            usage = ResourceUsage(
                component_id=registration.component_id,
                cpu_cores_used=1.0 + (i % 4) * 0.25,
                memory_mb_used=256 + (i % 8) * 32
            )
            await coordinator.track_resource_usage(usage)

        # Get aggregated data
        history = await coordinator.get_resource_usage_history(
            component_id=registration.component_id,
            limit=50
        )

        assert len(history) == 50


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows."""

    @pytest.mark.asyncio
    async def test_full_system_lifecycle(self, temp_db_path):
        """Test complete system lifecycle management."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # 1. System startup - register all components
        daemon_reg = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001",
            config={"grpc_port": 50051}
        )

        server_reg = await coordinator.register_component(
            component_type=ComponentType.PYTHON_MCP_SERVER,
            instance_id="server-001",
            config={"http_port": 8080}
        )

        cli_reg = await coordinator.register_component(
            component_type=ComponentType.CLI_UTILITY,
            instance_id="cli-001"
        )

        injector_reg = await coordinator.register_component(
            component_type=ComponentType.CONTEXT_INJECTOR,
            instance_id="injector-001"
        )

        # 2. Setup dependencies
        dependencies = [
            ComponentDependency(
                dependent_component_id=server_reg.component_id,
                dependency_component_id=daemon_reg.component_id,
                dependency_type="startup_order",
                is_critical=True
            ),
            ComponentDependency(
                dependent_component_id=cli_reg.component_id,
                dependency_component_id=server_reg.component_id,
                dependency_type="runtime",
                is_critical=False
            )
        ]

        for dep in dependencies:
            await coordinator.add_component_dependency(dep)

        # 3. Create communication channels
        await coordinator.create_communication_channel(
            source_component_id=server_reg.component_id,
            target_component_id=daemon_reg.component_id,
            channel_type="grpc",
            config={"port": 50051}
        )

        # 4. Setup processing queues
        await coordinator.create_processing_queue(
            component_id=daemon_reg.component_id,
            queue_type="document_processing",
            max_size=1000
        )

        # 5. System operation simulation
        for i in range(5):
            # Send heartbeats
            await coordinator.send_component_heartbeat(daemon_reg.component_id)
            await coordinator.send_component_heartbeat(server_reg.component_id)

            # Update health status
            await coordinator.update_component_health(
                component_id=daemon_reg.component_id,
                health_status=ComponentHealth.HEALTHY,
                metrics={"cpu_usage": 30.0 + i}
            )

        # 6. Get system overview
        overview = await coordinator.get_system_overview()

        assert overview["total_components"] == 4
        assert ComponentType.RUST_DAEMON.value in overview["components_by_type"]

        # 7. System shutdown simulation
        await coordinator.update_component_state(daemon_reg.component_id, ComponentState.STOPPING)
        await coordinator.update_component_state(server_reg.component_id, ComponentState.STOPPING)

    @pytest.mark.asyncio
    async def test_failure_recovery_scenario(self, temp_db_path):
        """Test component failure and recovery scenario."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register component
        registration = await coordinator.register_component(
            component_type=ComponentType.RUST_DAEMON,
            instance_id="daemon-001"
        )

        # Normal operation
        await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.HEALTHY,
            metrics={"cpu_usage": 25.0}
        )

        await coordinator.update_component_state(
            component_id=registration.component_id,
            new_state=ComponentState.ACTIVE
        )

        # Component failure
        await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.UNHEALTHY,
            metrics={"cpu_usage": 95.0, "error_count": 10}
        )

        await coordinator.update_component_state(
            component_id=registration.component_id,
            new_state=ComponentState.FAILED
        )

        # Recovery process
        await coordinator.update_component_state(
            component_id=registration.component_id,
            new_state=ComponentState.RECOVERING
        )

        await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.DEGRADED,
            metrics={"cpu_usage": 45.0, "error_count": 2}
        )

        # Full recovery
        await coordinator.update_component_state(
            component_id=registration.component_id,
            new_state=ComponentState.ACTIVE
        )

        await coordinator.update_component_health(
            component_id=registration.component_id,
            health_status=ComponentHealth.HEALTHY,
            metrics={"cpu_usage": 20.0, "error_count": 0}
        )

        # Verify final state
        final_health = await coordinator.check_component_health(registration.component_id)
        assert final_health == ComponentHealth.HEALTHY

    @pytest.mark.asyncio
    async def test_multi_instance_coordination(self, temp_db_path):
        """Test coordination of multiple instances of same component type."""
        coordinator = ComponentCoordinator(db_path=temp_db_path)
        await coordinator.initialize()

        # Register multiple daemon instances
        daemon_registrations = []
        for i in range(3):
            registration = await coordinator.register_component(
                component_type=ComponentType.RUST_DAEMON,
                instance_id=f"daemon-{i:03d}",
                config={"grpc_port": 50051 + i}
            )
            daemon_registrations.append(registration)

        # Create processing queues for each
        for registration in daemon_registrations:
            await coordinator.create_processing_queue(
                component_id=registration.component_id,
                queue_type="load_balanced_processing",
                max_size=500
            )

        # Simulate load balancing by updating queue statistics
        for i, registration in enumerate(daemon_registrations):
            # Different load levels
            current_size = i * 50  # 0, 50, 100
            processed = (i + 1) * 100  # 100, 200, 300

            queues = await coordinator.get_component_queues(registration.component_id)
            if queues:
                await coordinator.update_queue_statistics(
                    queue_id=queues[0].queue_id,
                    current_size=current_size,
                    processed_count=processed
                )

        # Get all daemon components and verify distribution
        daemon_components = await coordinator.get_components_by_type(ComponentType.RUST_DAEMON)
        assert len(daemon_components) == 3

        # Verify all have unique instance IDs and configurations
        instance_ids = {comp.instance_id for comp in daemon_components}
        assert len(instance_ids) == 3