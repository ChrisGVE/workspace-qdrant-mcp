"""
Main Dummy Data Generator

Central class that orchestrates all dummy data generation for testing
inter-component communication patterns.
"""

import random
import uuid
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

from .grpc_messages import GrpcMessageGenerator
from .mcp_messages import McpMessageGenerator
from .cli_messages import CliCommandGenerator
from .qdrant_data import QdrantDataGenerator
from .project_data import ProjectDataGenerator


class CommunicationPattern(Enum):
    """Supported communication patterns for testing."""
    MCP_TO_DAEMON = "mcp_to_daemon"
    DAEMON_TO_MCP = "daemon_to_mcp"
    CLI_TO_DAEMON = "cli_to_daemon"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class TestScenario:
    """Defines a complete test scenario with dummy data requirements."""
    name: str
    pattern: CommunicationPattern
    message_count: int
    include_errors: bool = False
    error_rate: float = 0.1
    timeout_scenarios: bool = False
    data_corruption: bool = False
    network_partition: bool = False


class DummyDataGenerator:
    """
    Main dummy data generator that coordinates all sub-generators
    to create comprehensive test datasets for communication testing.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize dummy data generator.

        Args:
            seed: Random seed for reproducible test data generation
        """
        self.seed = seed or int(time.time())
        random.seed(self.seed)

        # Initialize sub-generators
        self.grpc_gen = GrpcMessageGenerator(seed=self.seed)
        self.mcp_gen = McpMessageGenerator(seed=self.seed)
        self.cli_gen = CliCommandGenerator(seed=self.seed)
        self.qdrant_gen = QdrantDataGenerator(seed=self.seed)
        self.project_gen = ProjectDataGenerator(seed=self.seed)

    def generate_scenario_data(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Generate complete dummy data for a test scenario.

        Args:
            scenario: Test scenario specification

        Returns:
            Dictionary containing all generated test data
        """
        data = {
            "scenario": scenario.name,
            "pattern": scenario.pattern.value,
            "seed": self.seed,
            "timestamp": time.time(),
            "messages": []
        }

        for i in range(scenario.message_count):
            # Determine if this message should have errors
            has_error = scenario.include_errors and random.random() < scenario.error_rate

            message_data = self._generate_message_for_pattern(
                scenario.pattern,
                message_id=i,
                has_error=has_error,
                timeout_scenario=scenario.timeout_scenarios and random.random() < 0.05,
                data_corruption=scenario.data_corruption and random.random() < 0.03,
                network_partition=scenario.network_partition and random.random() < 0.02
            )

            data["messages"].append(message_data)

        return data

    def _generate_message_for_pattern(
        self,
        pattern: CommunicationPattern,
        message_id: int,
        has_error: bool = False,
        timeout_scenario: bool = False,
        data_corruption: bool = False,
        network_partition: bool = False
    ) -> Dict[str, Any]:
        """Generate a single message for the specified communication pattern."""

        base_message = {
            "id": message_id,
            "timestamp": time.time(),
            "has_error": has_error,
            "timeout_scenario": timeout_scenario,
            "data_corruption": data_corruption,
            "network_partition": network_partition
        }

        if pattern == CommunicationPattern.MCP_TO_DAEMON:
            message = self._generate_mcp_to_daemon_message()
        elif pattern == CommunicationPattern.DAEMON_TO_MCP:
            message = self._generate_daemon_to_mcp_message()
        elif pattern == CommunicationPattern.CLI_TO_DAEMON:
            message = self._generate_cli_to_daemon_message()
        elif pattern == CommunicationPattern.BIDIRECTIONAL:
            message = self._generate_bidirectional_message()
        else:
            raise ValueError(f"Unknown communication pattern: {pattern}")

        # Apply error conditions
        if has_error:
            message = self._inject_error(message)
        if timeout_scenario:
            message = self._inject_timeout_scenario(message)
        if data_corruption:
            message = self._corrupt_data(message)
        if network_partition:
            message = self._simulate_network_partition(message)

        base_message.update(message)
        return base_message

    def _generate_mcp_to_daemon_message(self) -> Dict[str, Any]:
        """Generate MCP tool -> gRPC service message."""
        # Choose random MCP tool and corresponding gRPC service
        mcp_tool = random.choice([
            "add_document", "search_workspace", "get_document",
            "hybrid_search_advanced", "setup_folder_watch",
            "list_collections", "workspace_status"
        ])

        mcp_request = self.mcp_gen.generate_tool_request(mcp_tool)
        grpc_request = self.grpc_gen.generate_corresponding_grpc_message(mcp_tool, mcp_request)

        return {
            "pattern": "mcp_to_daemon",
            "mcp_tool": mcp_tool,
            "mcp_request": mcp_request,
            "grpc_request": grpc_request,
            "expected_service": self._get_grpc_service_for_tool(mcp_tool),
            "project_context": self.project_gen.generate_project_context()
        }

    def _generate_daemon_to_mcp_message(self) -> Dict[str, Any]:
        """Generate gRPC service -> MCP callback message."""
        callback_type = random.choice([
            "document_processed", "search_results", "error_notification",
            "progress_update", "health_status"
        ])

        grpc_response = self.grpc_gen.generate_callback_message(callback_type)
        mcp_notification = self.mcp_gen.generate_notification(callback_type, grpc_response)

        return {
            "pattern": "daemon_to_mcp",
            "callback_type": callback_type,
            "grpc_response": grpc_response,
            "mcp_notification": mcp_notification,
            "service_origin": random.choice([
                "DocumentProcessor", "SearchService", "MemoryService",
                "SystemService", "ServiceDiscovery"
            ])
        }

    def _generate_cli_to_daemon_message(self) -> Dict[str, Any]:
        """Generate CLI command -> gRPC service message."""
        cli_command = random.choice([
            "wqm service status", "wqm admin collections",
            "wqm health check", "wqm document process",
            "wqm search query", "wqm collection create"
        ])

        cli_data = self.cli_gen.generate_command_data(cli_command)
        grpc_request = self.grpc_gen.generate_cli_grpc_message(cli_command, cli_data)

        return {
            "pattern": "cli_to_daemon",
            "cli_command": cli_command,
            "cli_data": cli_data,
            "grpc_request": grpc_request,
            "expected_service": self._get_grpc_service_for_cli_command(cli_command)
        }

    def _generate_bidirectional_message(self) -> Dict[str, Any]:
        """Generate bidirectional communication scenario."""
        scenario_type = random.choice([
            "document_upload_with_progress", "streaming_search",
            "batch_processing", "health_monitoring"
        ])

        return {
            "pattern": "bidirectional",
            "scenario_type": scenario_type,
            "request_chain": self._generate_request_chain(scenario_type),
            "response_chain": self._generate_response_chain(scenario_type)
        }

    def _generate_request_chain(self, scenario_type: str) -> List[Dict[str, Any]]:
        """Generate a chain of requests for bidirectional scenarios."""
        chains = {
            "document_upload_with_progress": [
                {"type": "mcp_tool", "data": self.mcp_gen.generate_tool_request("add_document")},
                {"type": "grpc_request", "data": self.grpc_gen.generate_document_upload()},
                {"type": "progress_callback", "data": self.grpc_gen.generate_progress_update()}
            ],
            "streaming_search": [
                {"type": "mcp_tool", "data": self.mcp_gen.generate_tool_request("search_workspace")},
                {"type": "grpc_stream", "data": self.grpc_gen.generate_streaming_search()},
                {"type": "result_chunk", "data": self.grpc_gen.generate_search_chunk()}
            ],
            "batch_processing": [
                {"type": "cli_command", "data": self.cli_gen.generate_command_data("wqm document batch")},
                {"type": "grpc_batch", "data": self.grpc_gen.generate_batch_request()},
                {"type": "batch_status", "data": self.grpc_gen.generate_batch_status()}
            ],
            "health_monitoring": [
                {"type": "health_check", "data": self.grpc_gen.generate_health_check()},
                {"type": "metrics_collection", "data": self.grpc_gen.generate_metrics()},
                {"type": "alert_notification", "data": self.grpc_gen.generate_alert()}
            ]
        }

        return chains.get(scenario_type, [])

    def _generate_response_chain(self, scenario_type: str) -> List[Dict[str, Any]]:
        """Generate a chain of responses for bidirectional scenarios."""
        chains = {
            "document_upload_with_progress": [
                {"type": "upload_started", "data": self.grpc_gen.generate_upload_response()},
                {"type": "progress_updates", "data": [self.grpc_gen.generate_progress_update() for _ in range(3)]},
                {"type": "upload_completed", "data": self.grpc_gen.generate_completion_response()}
            ],
            "streaming_search": [
                {"type": "search_started", "data": self.grpc_gen.generate_search_response()},
                {"type": "result_chunks", "data": [self.grpc_gen.generate_search_chunk() for _ in range(5)]},
                {"type": "search_completed", "data": self.grpc_gen.generate_search_completion()}
            ],
            "batch_processing": [
                {"type": "batch_accepted", "data": self.grpc_gen.generate_batch_response()},
                {"type": "status_updates", "data": [self.grpc_gen.generate_batch_status() for _ in range(4)]},
                {"type": "batch_completed", "data": self.grpc_gen.generate_batch_completion()}
            ],
            "health_monitoring": [
                {"type": "health_status", "data": self.grpc_gen.generate_health_status()},
                {"type": "metrics_report", "data": self.grpc_gen.generate_metrics_report()},
                {"type": "alert_resolved", "data": self.grpc_gen.generate_alert_resolution()}
            ]
        }

        return chains.get(scenario_type, [])

    def _get_grpc_service_for_tool(self, mcp_tool: str) -> str:
        """Map MCP tool to corresponding gRPC service."""
        mappings = {
            "add_document": "DocumentProcessor",
            "get_document": "DocumentProcessor",
            "search_workspace": "SearchService",
            "hybrid_search_advanced": "SearchService",
            "list_collections": "MemoryService",
            "setup_folder_watch": "SystemService",
            "workspace_status": "SystemService"
        }
        return mappings.get(mcp_tool, "SystemService")

    def _get_grpc_service_for_cli_command(self, cli_command: str) -> str:
        """Map CLI command to corresponding gRPC service."""
        if "service" in cli_command:
            return "SystemService"
        elif "admin" in cli_command:
            return "MemoryService"
        elif "health" in cli_command:
            return "SystemService"
        elif "document" in cli_command:
            return "DocumentProcessor"
        elif "search" in cli_command:
            return "SearchService"
        else:
            return "SystemService"

    def _inject_error(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Inject various error conditions into messages."""
        error_types = [
            "invalid_request", "missing_field", "type_mismatch",
            "authentication_failed", "permission_denied", "resource_not_found",
            "service_unavailable", "timeout", "internal_error"
        ]

        error_type = random.choice(error_types)
        message["error_injection"] = {
            "type": error_type,
            "message": f"Simulated {error_type} error",
            "code": random.choice([400, 401, 403, 404, 500, 502, 503, 504])
        }

        return message

    def _inject_timeout_scenario(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Inject timeout scenarios."""
        message["timeout_injection"] = {
            "type": random.choice(["connection_timeout", "read_timeout", "write_timeout"]),
            "delay_seconds": random.uniform(10, 60),
            "should_retry": random.choice([True, False])
        }

        return message

    def _corrupt_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate data corruption scenarios."""
        corruption_types = [
            "truncated_message", "invalid_utf8", "missing_bytes",
            "extra_bytes", "wrong_encoding", "malformed_json"
        ]

        message["data_corruption"] = {
            "type": random.choice(corruption_types),
            "location": random.choice(["header", "body", "metadata"]),
            "severity": random.choice(["minor", "major", "critical"])
        }

        return message

    def _simulate_network_partition(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate network partition scenarios."""
        message["network_partition"] = {
            "type": random.choice(["complete_loss", "intermittent", "degraded"]),
            "duration_seconds": random.uniform(1, 30),
            "affects_response": random.choice([True, False]),
            "recovery_strategy": random.choice(["reconnect", "failover", "wait"])
        }

        return message

    async def generate_load_test_data(
        self,
        concurrent_patterns: int = 10,
        duration_seconds: int = 60,
        requests_per_second: int = 100
    ) -> Dict[str, Any]:
        """
        Generate data for load testing scenarios.

        Args:
            concurrent_patterns: Number of concurrent communication patterns
            duration_seconds: Duration of load test
            requests_per_second: Target requests per second

        Returns:
            Load test data specification
        """
        total_requests = duration_seconds * requests_per_second
        requests_per_pattern = total_requests // concurrent_patterns

        load_test_data = {
            "load_test_config": {
                "concurrent_patterns": concurrent_patterns,
                "duration_seconds": duration_seconds,
                "requests_per_second": requests_per_second,
                "total_requests": total_requests
            },
            "patterns": []
        }

        for pattern_id in range(concurrent_patterns):
            pattern = random.choice(list(CommunicationPattern))
            scenario = TestScenario(
                name=f"load_test_pattern_{pattern_id}",
                pattern=pattern,
                message_count=requests_per_pattern,
                include_errors=True,
                error_rate=0.05,  # Lower error rate for load testing
                timeout_scenarios=True
            )

            pattern_data = self.generate_scenario_data(scenario)
            load_test_data["patterns"].append(pattern_data)

        return load_test_data

    def export_scenario_data(self, data: Dict[str, Any], filepath: str) -> None:
        """Export generated scenario data to file."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def get_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about generated data."""
        messages = data.get("messages", [])
        total_messages = len(messages)

        stats = {
            "total_messages": total_messages,
            "error_messages": sum(1 for m in messages if m.get("has_error")),
            "timeout_scenarios": sum(1 for m in messages if m.get("timeout_scenario")),
            "data_corruption_cases": sum(1 for m in messages if m.get("data_corruption")),
            "network_partition_cases": sum(1 for m in messages if m.get("network_partition")),
            "pattern_distribution": {}
        }

        # Calculate pattern distribution
        for message in messages:
            pattern = message.get("pattern", "unknown")
            stats["pattern_distribution"][pattern] = stats["pattern_distribution"].get(pattern, 0) + 1

        return stats