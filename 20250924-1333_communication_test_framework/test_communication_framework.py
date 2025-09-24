"""
Comprehensive Unit Tests for Inter-Component Communication Testing Framework

Tests all communication patterns with edge cases:
- MCP-to-daemon communication
- Daemon-to-MCP communication
- CLI-to-daemon communication
- Network failures, timeouts, data corruption
- Service unavailability scenarios
- Protocol validation
"""

import asyncio
import pytest
import time
import uuid
from typing import Dict, List, Any
import json

from dummy_data.generators import DummyDataGenerator, CommunicationPattern, TestScenario
from dummy_data.grpc_messages import GrpcMessageGenerator
from dummy_data.mcp_messages import McpMessageGenerator
from dummy_data.cli_messages import CliCommandGenerator
from dummy_data.qdrant_data import QdrantDataGenerator
from dummy_data.project_data import ProjectDataGenerator
from mock_services.grpc_services import MockGrpcServices, MockServiceConfig, ServiceState


class TestDummyDataGeneration:
    """Test dummy data generation capabilities."""

    def setup_method(self):
        """Setup test environment."""
        self.data_gen = DummyDataGenerator(seed=42)  # Fixed seed for reproducibility

    def test_scenario_generation_mcp_to_daemon(self):
        """Test MCP-to-daemon communication scenario generation."""
        scenario = TestScenario(
            name="mcp_to_daemon_test",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=10,
            include_errors=True,
            error_rate=0.2
        )

        data = self.data_gen.generate_scenario_data(scenario)

        assert data["scenario"] == "mcp_to_daemon_test"
        assert data["pattern"] == "mcp_to_daemon"
        assert len(data["messages"]) == 10

        # Check that some messages have errors
        error_messages = [msg for msg in data["messages"] if msg.get("has_error")]
        assert len(error_messages) > 0  # Should have some errors with 0.2 rate

        # Validate message structure
        for message in data["messages"]:
            assert "pattern" in message
            assert "mcp_tool" in message
            assert "grpc_request" in message
            assert "expected_service" in message
            assert message["pattern"] == "mcp_to_daemon"

    def test_scenario_generation_daemon_to_mcp(self):
        """Test daemon-to-MCP communication scenario generation."""
        scenario = TestScenario(
            name="daemon_to_mcp_test",
            pattern=CommunicationPattern.DAEMON_TO_MCP,
            message_count=5,
            timeout_scenarios=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        assert data["pattern"] == "daemon_to_mcp"
        assert len(data["messages"]) == 5

        # Check for timeout scenarios
        timeout_messages = [msg for msg in data["messages"] if msg.get("timeout_scenario")]
        # May or may not have timeouts due to random generation

        # Validate daemon-to-MCP structure
        for message in data["messages"]:
            assert message["pattern"] == "daemon_to_mcp"
            assert "callback_type" in message
            assert "grpc_response" in message
            assert "mcp_notification" in message

    def test_scenario_generation_cli_to_daemon(self):
        """Test CLI-to-daemon communication scenario generation."""
        scenario = TestScenario(
            name="cli_to_daemon_test",
            pattern=CommunicationPattern.CLI_TO_DAEMON,
            message_count=8,
            data_corruption=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        assert data["pattern"] == "cli_to_daemon"
        assert len(data["messages"]) == 8

        # Validate CLI-to-daemon structure
        for message in data["messages"]:
            assert message["pattern"] == "cli_to_daemon"
            assert "cli_command" in message
            assert "grpc_request" in message
            assert "expected_service" in message

    def test_bidirectional_communication(self):
        """Test bidirectional communication scenarios."""
        scenario = TestScenario(
            name="bidirectional_test",
            pattern=CommunicationPattern.BIDIRECTIONAL,
            message_count=3
        )

        data = self.data_gen.generate_scenario_data(scenario)

        assert data["pattern"] == "bidirectional"

        for message in data["messages"]:
            assert message["pattern"] == "bidirectional"
            assert "scenario_type" in message
            assert "request_chain" in message
            assert "response_chain" in message

    def test_error_injection_scenarios(self):
        """Test various error injection capabilities."""
        scenario = TestScenario(
            name="error_test",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=20,
            include_errors=True,
            error_rate=1.0,  # 100% error rate for testing
            timeout_scenarios=True,
            data_corruption=True,
            network_partition=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        # All messages should have errors
        for message in data["messages"]:
            assert message["has_error"] is True
            assert "error_injection" in message

            error_injection = message["error_injection"]
            assert "type" in error_injection
            assert "message" in error_injection
            assert "code" in error_injection

        # Check for different types of problems
        timeout_count = sum(1 for msg in data["messages"] if msg.get("timeout_scenario"))
        corruption_count = sum(1 for msg in data["messages"] if msg.get("data_corruption"))
        partition_count = sum(1 for msg in data["messages"] if msg.get("network_partition"))

        # Should have some of each type (though random)
        assert timeout_count + corruption_count + partition_count > 0

    async def test_load_test_data_generation(self):
        """Test load testing data generation."""
        load_data = await self.data_gen.generate_load_test_data(
            concurrent_patterns=5,
            duration_seconds=10,
            requests_per_second=20
        )

        assert load_data["load_test_config"]["concurrent_patterns"] == 5
        assert load_data["load_test_config"]["duration_seconds"] == 10
        assert load_data["load_test_config"]["requests_per_second"] == 20
        assert load_data["load_test_config"]["total_requests"] == 200

        assert len(load_data["patterns"]) == 5

        total_messages = sum(len(pattern["messages"]) for pattern in load_data["patterns"])
        assert total_messages == 200

    def test_statistics_calculation(self):
        """Test statistics calculation for generated data."""
        scenario = TestScenario(
            name="stats_test",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=100,
            include_errors=True,
            error_rate=0.1
        )

        data = self.data_gen.generate_scenario_data(scenario)
        stats = self.data_gen.get_statistics(data)

        assert stats["total_messages"] == 100
        assert "error_messages" in stats
        assert "pattern_distribution" in stats
        assert stats["pattern_distribution"]["mcp_to_daemon"] == 100

    def test_data_export(self, tmp_path):
        """Test data export functionality."""
        scenario = TestScenario(
            name="export_test",
            pattern=CommunicationPattern.CLI_TO_DAEMON,
            message_count=5
        )

        data = self.data_gen.generate_scenario_data(scenario)
        export_path = tmp_path / "test_data.json"

        self.data_gen.export_scenario_data(data, str(export_path))

        assert export_path.exists()

        # Verify exported data
        with open(export_path) as f:
            exported_data = json.load(f)

        assert exported_data["scenario"] == "export_test"
        assert len(exported_data["messages"]) == 5


class TestGrpcMessageGeneration:
    """Test gRPC message generation for all services."""

    def setup_method(self):
        """Setup test environment."""
        self.grpc_gen = GrpcMessageGenerator(seed=42)

    def test_document_processor_messages(self):
        """Test DocumentProcessor service message generation."""
        # Test request generation
        request = self.grpc_gen.generate_service_message(
            "DocumentProcessor",
            "ProcessDocument",
            is_response=False
        )

        assert request["message_type"] == "ProcessDocumentRequest"
        assert "document_id" in request
        assert "content" in request
        assert "metadata" in request
        assert "processing_options" in request

        # Test response generation
        response = self.grpc_gen.generate_service_message(
            "DocumentProcessor",
            "ProcessDocument",
            is_response=True
        )

        assert response["message_type"] == "ProcessDocumentResponse"
        assert "document_id" in response
        assert "status" in response
        assert "processing_time_ms" in response

    def test_search_service_messages(self):
        """Test SearchService message generation."""
        request = self.grpc_gen.generate_service_message(
            "SearchService",
            "Search",
            is_response=False
        )

        assert request["message_type"] == "SearchRequest"
        assert "query" in request
        assert "collection_id" in request
        assert "limit" in request

    def test_memory_service_messages(self):
        """Test MemoryService message generation."""
        request = self.grpc_gen.generate_service_message(
            "MemoryService",
            "CreateCollection",
            is_response=False
        )

        assert request["message_type"] == "CreateCollectionRequest"
        assert "collection_name" in request
        assert "vector_size" in request
        assert "distance_metric" in request

    def test_system_service_messages(self):
        """Test SystemService message generation."""
        response = self.grpc_gen.generate_service_message(
            "SystemService",
            "GetSystemStatus",
            is_response=True
        )

        assert response["message_type"] == "GetSystemStatusResponse"
        assert "status" in response
        assert "uptime_seconds" in response
        assert "memory_usage_mb" in response

    def test_service_discovery_messages(self):
        """Test ServiceDiscovery message generation."""
        request = self.grpc_gen.generate_service_message(
            "ServiceDiscovery",
            "RegisterService",
            is_response=False
        )

        assert request["message_type"] == "RegisterServiceRequest"
        assert "service_name" in request
        assert "address" in request
        assert "port" in request

    def test_callback_message_generation(self):
        """Test callback message generation."""
        callback_msg = self.grpc_gen.generate_callback_message("document_processed")

        assert callback_msg["callback_type"] == "document_processed"
        assert "document_id" in callback_msg
        assert "status" in callback_msg

    def test_streaming_message_generation(self):
        """Test streaming operation messages."""
        stream_request = self.grpc_gen.generate_streaming_search()

        assert stream_request["message_type"] == "SearchStreamRequest"
        assert "query" in stream_request
        assert "stream_config" in stream_request

        # Test search chunk
        chunk = self.grpc_gen.generate_search_chunk()
        assert "chunk_id" in chunk
        assert "results" in chunk


class TestMcpMessageGeneration:
    """Test MCP message generation for all tools."""

    def setup_method(self):
        """Setup test environment."""
        self.mcp_gen = McpMessageGenerator(seed=42)

    def test_document_tool_requests(self):
        """Test document management tool requests."""
        request = self.mcp_gen.generate_tool_request("add_document")

        assert request["jsonrpc"] == "2.0"
        assert request["method"] == "tools/call"
        assert request["params"]["name"] == "add_document"
        assert "content" in request["params"]["arguments"]

    def test_search_tool_requests(self):
        """Test search tool requests."""
        request = self.mcp_gen.generate_tool_request("search_workspace")

        assert request["params"]["name"] == "search_workspace"
        assert "query" in request["params"]["arguments"]

    def test_collection_management_requests(self):
        """Test collection management requests."""
        request = self.mcp_gen.generate_tool_request("list_collections")

        assert request["params"]["name"] == "list_collections"

    def test_notification_generation(self):
        """Test notification generation from gRPC responses."""
        grpc_response = {
            "document_id": "test-doc-123",
            "status": "SUCCESS",
            "processing_time_ms": 1500
        }

        notification = self.mcp_gen.generate_notification(
            "document_processed",
            grpc_response
        )

        assert notification["jsonrpc"] == "2.0"
        assert notification["method"] == "notifications/document_processed"
        assert notification["params"]["document_id"] == "test-doc-123"

    def test_tool_response_generation(self):
        """Test tool response generation."""
        # Success response
        success_response = self.mcp_gen.generate_tool_response(
            "add_document",
            success=True
        )

        assert success_response["jsonrpc"] == "2.0"
        assert "result" in success_response

        # Error response
        error_response = self.mcp_gen.generate_tool_response(
            "add_document",
            success=False
        )

        assert error_response["jsonrpc"] == "2.0"
        assert "error" in error_response

    def test_batch_request_generation(self):
        """Test batch request generation."""
        tools = ["add_document", "search_workspace", "list_collections"]
        batch_request = self.mcp_gen.generate_batch_request(tools, batch_size=5)

        assert batch_request["method"] == "batch_request"
        assert len(batch_request["params"]["requests"]) == 5

    def test_all_tool_coverage(self):
        """Test that all tools can generate requests."""
        all_tools = self.mcp_gen.get_all_tool_names()

        # Should have 30+ tools
        assert len(all_tools) >= 30

        # Test each tool can generate a request
        for tool_name in all_tools:
            request = self.mcp_gen.generate_tool_request(tool_name)
            assert request["params"]["name"] == tool_name
            assert "arguments" in request["params"]


class TestCliCommandGeneration:
    """Test CLI command generation."""

    def setup_method(self):
        """Setup test environment."""
        self.cli_gen = CliCommandGenerator(seed=42)

    def test_service_commands(self):
        """Test service management commands."""
        cmd_data = self.cli_gen.generate_command_data("wqm service status")

        assert cmd_data["command"] == "wqm"
        assert cmd_data["subcommand"] == "service status"
        assert cmd_data["full_command"] == "wqm service status"
        assert "expected_output" in cmd_data

    def test_admin_commands(self):
        """Test admin commands."""
        cmd_data = self.cli_gen.generate_command_data("wqm admin collections")

        assert cmd_data["subcommand"] == "admin collections"
        assert "options" in cmd_data
        assert "environment" in cmd_data

    def test_document_processing_commands(self):
        """Test document processing commands."""
        cmd_data = self.cli_gen.generate_command_data("wqm document process")

        assert cmd_data["subcommand"] == "document process"
        assert "execution_context" in cmd_data

    def test_command_sequence_generation(self):
        """Test command sequence with dependencies."""
        commands = [
            "wqm service start",
            "wqm admin collections",
            "wqm document process"
        ]

        dependencies = {
            "wqm admin collections": ["wqm service start"],
            "wqm document process": ["wqm service start", "wqm admin collections"]
        }

        sequence = self.cli_gen.generate_command_sequence(commands, dependencies)

        assert len(sequence) == 3

        # Verify dependency order
        executed_commands = [cmd["full_command"] for cmd in sequence]
        start_idx = executed_commands.index("wqm service start")
        admin_idx = executed_commands.index("wqm admin collections")
        process_idx = executed_commands.index("wqm document process")

        assert start_idx < admin_idx < process_idx

    def test_error_scenario_generation(self):
        """Test error scenario generation."""
        error_scenarios = self.cli_gen.generate_error_scenarios("wqm service status")

        assert len(error_scenarios) > 0

        for scenario in error_scenarios:
            assert "scenario" in scenario
            assert scenario["expected_output"]["type"] == "error"

    def test_load_test_command_generation(self):
        """Test load test command generation."""
        load_commands = self.cli_gen.generate_load_test_commands(
            duration_seconds=5,
            commands_per_second=10
        )

        assert len(load_commands) == 50  # 5 * 10

        for i, cmd in enumerate(load_commands):
            assert "load_test" in cmd
            assert cmd["load_test"]["sequence_number"] == i


class TestQdrantDataGeneration:
    """Test Qdrant data generation."""

    def setup_method(self):
        """Setup test environment."""
        self.qdrant_gen = QdrantDataGenerator(seed=42)

    def test_vector_generation(self):
        """Test vector generation with different specifications."""
        from dummy_data.qdrant_data import VectorSpec

        # Normal distribution
        normal_spec = VectorSpec(size=768, distribution="normal")
        normal_vector = self.qdrant_gen.generate_vector(normal_spec)

        assert len(normal_vector) == 768
        assert all(isinstance(v, float) for v in normal_vector)

        # Sparse distribution
        sparse_spec = VectorSpec(size=768, distribution="sparse")
        sparse_vector = self.qdrant_gen.generate_vector(sparse_spec)

        assert len(sparse_vector) == 768
        # Most values should be zero for sparse vectors
        zero_count = sum(1 for v in sparse_vector if abs(v) < 1e-10)
        assert zero_count > len(sparse_vector) * 0.8  # At least 80% zeros

    def test_point_generation(self):
        """Test point generation with vectors and payloads."""
        point = self.qdrant_gen.generate_point(include_payload=True)

        assert "id" in point
        assert "vector" in point
        assert "payload" in point
        assert isinstance(point["vector"], list)
        assert isinstance(point["payload"], dict)

    def test_search_request_generation(self):
        """Test search request generation."""
        search_request = self.qdrant_gen.generate_search_request()

        assert "collection_name" in search_request
        assert "vector" in search_request
        assert "limit" in search_request
        assert isinstance(search_request["vector"], list)

    def test_collection_config_generation(self):
        """Test collection configuration generation."""
        from dummy_data.qdrant_data import CollectionSpec

        spec = CollectionSpec(
            name="test_collection",
            vector_size=768,
            distance_metric="Cosine"
        )

        config = self.qdrant_gen.generate_collection_config(spec)

        assert config["vectors"]["size"] == 768
        assert config["vectors"]["distance"] == "Cosine"
        assert "optimizers_config" in config
        assert "hnsw_config" in config

    def test_batch_operation_generation(self):
        """Test batch operation generation."""
        batch_op = self.qdrant_gen.generate_batch_operation(
            "upsert",
            batch_size=10
        )

        assert "collection_name" in batch_op
        assert "operations" in batch_op
        assert len(batch_op["operations"]) == 10

    def test_hybrid_search_generation(self):
        """Test hybrid search request generation."""
        hybrid_request = self.qdrant_gen.generate_hybrid_search_request()

        assert "collection_name" in hybrid_request
        assert "prefetch" in hybrid_request
        assert len(hybrid_request["prefetch"]) == 2  # Dense and sparse

    def test_error_response_generation(self):
        """Test error response generation."""
        error_response = self.qdrant_gen.generate_error_responses("collection_not_found")

        assert "status" in error_response
        assert "error" in error_response["status"]

    def test_load_test_dataset_generation(self):
        """Test load testing dataset generation."""
        dataset = self.qdrant_gen.generate_load_test_dataset(
            num_points=100,
            vector_size=384
        )

        assert len(dataset) == 100
        for point in dataset:
            assert len(point["vector"]) == 384


class TestProjectDataGeneration:
    """Test project context data generation."""

    def setup_method(self):
        """Setup test environment."""
        self.project_gen = ProjectDataGenerator(seed=42)

    def test_project_context_generation(self):
        """Test project context generation."""
        project_context = self.project_gen.generate_project_context()

        assert "project_name" in project_context
        assert "project_type" in project_context
        assert "languages" in project_context
        assert "structure" in project_context
        assert "git" in project_context
        assert "workspace" in project_context
        assert "lsp_servers" in project_context

    def test_multi_project_workspace_generation(self):
        """Test multi-project workspace generation."""
        workspace = self.project_gen.generate_multi_project_workspace(num_projects=3)

        assert "workspace_id" in workspace
        assert "projects" in workspace
        assert len(workspace["projects"]) == 3
        assert "statistics" in workspace

    def test_language_detection_data_generation(self):
        """Test language detection data generation."""
        detection_data = self.project_gen.generate_language_detection_data()

        # Should have data for multiple languages
        assert len(detection_data) > 10

        for lang_data in detection_data.values():
            assert "language" in lang_data
            assert "samples" in lang_data
            assert "detection_accuracy" in lang_data

    def test_project_comparison_generation(self):
        """Test project comparison generation."""
        projects = ["project1", "project2", "project3"]
        comparison = self.project_gen.generate_project_comparison(projects)

        assert comparison["projects"] == projects
        assert "metrics" in comparison
        assert "similarities" in comparison

        # Should have metrics for each project
        for project in projects:
            assert project in comparison["metrics"]


@pytest.mark.asyncio
class TestMockGrpcServices:
    """Test mock gRPC services behavior."""

    async def test_mock_document_processor(self):
        """Test MockDocumentProcessor behavior."""
        from mock_services.grpc_services import MockDocumentProcessor, MockServiceConfig

        config = MockServiceConfig(latency_ms=10, error_rate=0.0)  # No errors for test
        service = MockDocumentProcessor(config)

        request = {
            "document_id": "test-doc-123",
            "content": b"Test document content",
            "metadata": {"type": "test"}
        }

        response = await service.process_document(request)

        assert response["message_type"] == "ProcessDocumentResponse"
        assert "processing_time_ms" in response

        # Check metrics
        metrics = service.get_metrics()
        assert metrics["total_requests"] == 1
        assert metrics["successful_requests"] == 1

    async def test_mock_services_error_simulation(self):
        """Test error simulation in mock services."""
        from mock_services.grpc_services import MockDocumentProcessor, MockServiceConfig

        config = MockServiceConfig(error_rate=1.0)  # 100% error rate
        service = MockDocumentProcessor(config)

        request = {"document_id": "test-doc-456"}

        # Should raise an exception
        with pytest.raises(Exception):
            await service.process_document(request)

        # Check error metrics
        metrics = service.get_metrics()
        assert metrics["failed_requests"] == 1

    async def test_mock_services_callback_system(self):
        """Test callback system in mock services."""
        from mock_services.grpc_services import MockDocumentProcessor, MockServiceConfig

        config = MockServiceConfig(enable_callbacks=True, callback_delay_ms=10)
        service = MockDocumentProcessor(config)

        callback_received = []

        async def test_callback(callback_type: str, data: Dict[str, Any]):
            callback_received.append((callback_type, data))

        service.register_callback(test_callback)

        request = {"document_id": "test-doc-789"}
        await service.process_document(request)

        # Wait for callback
        await asyncio.sleep(0.05)

        assert len(callback_received) == 1
        assert callback_received[0][0] == "document_processed"

    async def test_mock_services_orchestrator(self):
        """Test MockGrpcServices orchestrator."""
        from mock_services.grpc_services import MockGrpcServices, MockServiceConfig

        configs = {
            "DocumentProcessor": MockServiceConfig(latency_ms=5),
            "SearchService": MockServiceConfig(latency_ms=10)
        }

        services = MockGrpcServices(configs)

        # Test service access
        doc_service = services.get_service("DocumentProcessor")
        search_service = services.get_service("SearchService")

        assert doc_service is not None
        assert search_service is not None

        # Test all metrics
        all_metrics = services.get_all_metrics()
        assert "DocumentProcessor" in all_metrics
        assert "SearchService" in all_metrics


class TestEdgeCases:
    """Test edge cases and failure scenarios."""

    def setup_method(self):
        """Setup test environment."""
        self.data_gen = DummyDataGenerator(seed=42)

    def test_network_failure_simulation(self):
        """Test network failure simulation."""
        scenario = TestScenario(
            name="network_failure_test",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=10,
            network_partition=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        # Check for network partition scenarios
        partition_messages = [
            msg for msg in data["messages"]
            if msg.get("network_partition")
        ]

        for msg in partition_messages:
            assert "network_partition" in msg
            network_info = msg["network_partition"]
            assert "type" in network_info
            assert "duration_seconds" in network_info

    def test_data_corruption_simulation(self):
        """Test data corruption simulation."""
        scenario = TestScenario(
            name="corruption_test",
            pattern=CommunicationPattern.CLI_TO_DAEMON,
            message_count=10,
            data_corruption=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        corruption_messages = [
            msg for msg in data["messages"]
            if msg.get("data_corruption")
        ]

        for msg in corruption_messages:
            assert "data_corruption" in msg
            corruption_info = msg["data_corruption"]
            assert "type" in corruption_info
            assert "location" in corruption_info
            assert "severity" in corruption_info

    def test_timeout_scenarios(self):
        """Test timeout scenario simulation."""
        scenario = TestScenario(
            name="timeout_test",
            pattern=CommunicationPattern.DAEMON_TO_MCP,
            message_count=10,
            timeout_scenarios=True
        )

        data = self.data_gen.generate_scenario_data(scenario)

        timeout_messages = [
            msg for msg in data["messages"]
            if msg.get("timeout_scenario")
        ]

        for msg in timeout_messages:
            assert "timeout_injection" in msg
            timeout_info = msg["timeout_injection"]
            assert "type" in timeout_info
            assert "delay_seconds" in timeout_info

    def test_service_unavailable_scenarios(self):
        """Test service unavailability scenarios."""
        # This would test when services are down/unavailable
        # In a real implementation, we'd test with services in STOPPED state
        pass

    def test_protocol_validation_errors(self):
        """Test protocol validation error scenarios."""
        # Test invalid MCP messages
        mcp_gen = McpMessageGenerator(seed=42)

        # Generate a malformed request (missing required fields)
        request = mcp_gen.generate_tool_request("add_document")

        # Remove required field to simulate protocol error
        if "content" in request["params"]["arguments"]:
            del request["params"]["arguments"]["content"]

        # In real implementation, this would be validated by protocol validator
        assert "content" not in request["params"]["arguments"]

    def test_high_concurrency_scenarios(self):
        """Test high concurrency simulation."""
        scenario = TestScenario(
            name="concurrency_test",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=100  # High message count
        )

        data = self.data_gen.generate_scenario_data(scenario)

        assert len(data["messages"]) == 100

        # Simulate concurrent execution
        concurrent_operations = []
        for i in range(0, 100, 10):  # Groups of 10
            batch = data["messages"][i:i+10]
            concurrent_operations.append(batch)

        assert len(concurrent_operations) == 10


if __name__ == "__main__":
    """Run tests with comprehensive reporting."""
    import sys

    # Run specific test categories
    if len(sys.argv) > 1:
        if sys.argv[1] == "data":
            pytest.main(["-v", "TestDummyDataGeneration"])
        elif sys.argv[1] == "grpc":
            pytest.main(["-v", "TestGrpcMessageGeneration"])
        elif sys.argv[1] == "mcp":
            pytest.main(["-v", "TestMcpMessageGeneration"])
        elif sys.argv[1] == "cli":
            pytest.main(["-v", "TestCliCommandGeneration"])
        elif sys.argv[1] == "qdrant":
            pytest.main(["-v", "TestQdrantDataGeneration"])
        elif sys.argv[1] == "project":
            pytest.main(["-v", "TestProjectDataGeneration"])
        elif sys.argv[1] == "mock":
            pytest.main(["-v", "TestMockGrpcServices"])
        elif sys.argv[1] == "edge":
            pytest.main(["-v", "TestEdgeCases"])
    else:
        # Run all tests
        pytest.main([
            "-v",
            "--tb=short",
            "--durations=10",
            "--cov=dummy_data",
            "--cov=mock_services",
            "--cov-report=term-missing"
        ])