"""
Integration Test Suite for Inter-Component Communication

This module provides comprehensive integration tests for all communication patterns
in the workspace-qdrant-mcp system, testing real-world scenarios with multiple
components interacting simultaneously.
"""

import asyncio
import time
import uuid
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

from dummy_data.generators import DummyDataGenerator, CommunicationPattern, TestScenario
from mock_services.grpc_services import MockGrpcServices, MockServiceConfig, ServiceState


class IntegrationTestResult(Enum):
    """Integration test result status."""
    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class CommunicationMetrics:
    """Metrics for communication testing."""
    total_messages: int = 0
    successful_messages: int = 0
    failed_messages: int = 0
    avg_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    min_response_time_ms: float = float('inf')
    timeout_count: int = 0
    error_count: int = 0
    protocol_errors: int = 0
    network_failures: int = 0


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    test_duration_seconds: int = 60
    concurrent_patterns: int = 3
    messages_per_pattern: int = 100
    error_injection_rate: float = 0.1
    timeout_simulation_rate: float = 0.05
    network_failure_rate: float = 0.02
    enable_callbacks: bool = True
    validate_protocol: bool = True
    collect_metrics: bool = True


class CommunicationTestSuite:
    """Comprehensive integration test suite for inter-component communication."""

    def __init__(self, config: Optional[IntegrationTestConfig] = None):
        """Initialize the test suite."""
        self.config = config or IntegrationTestConfig()
        self.data_generator = DummyDataGenerator()
        self.mock_services = None
        self.test_results = {}
        self.communication_metrics = CommunicationMetrics()
        self.callbacks_received = []
        self.protocol_violations = []

    async def initialize_test_environment(self):
        """Initialize the test environment with mock services."""
        print("Initializing test environment...")

        # Configure mock services with realistic behavior
        service_configs = {
            "DocumentProcessor": MockServiceConfig(
                latency_ms=150,
                error_rate=self.config.error_injection_rate * 0.8,  # Lower error rate for critical service
                timeout_rate=self.config.timeout_simulation_rate,
                enable_callbacks=self.config.enable_callbacks
            ),
            "SearchService": MockServiceConfig(
                latency_ms=80,
                error_rate=self.config.error_injection_rate,
                timeout_rate=self.config.timeout_simulation_rate,
                enable_callbacks=self.config.enable_callbacks
            ),
            "MemoryService": MockServiceConfig(
                latency_ms=100,
                error_rate=self.config.error_injection_rate * 0.5,  # Very reliable service
                timeout_rate=self.config.timeout_simulation_rate * 0.5,
                enable_callbacks=self.config.enable_callbacks
            ),
            "SystemService": MockServiceConfig(
                latency_ms=50,
                error_rate=self.config.error_injection_rate * 0.3,  # Most reliable
                timeout_rate=self.config.timeout_simulation_rate * 0.2,
                enable_callbacks=self.config.enable_callbacks
            ),
            "ServiceDiscovery": MockServiceConfig(
                latency_ms=30,
                error_rate=self.config.error_injection_rate * 0.2,  # Extremely reliable
                timeout_rate=0.0,  # No timeouts for service discovery
                enable_callbacks=True
            )
        }

        self.mock_services = MockGrpcServices(service_configs)

        # Register global callback to collect all service events
        await self._register_global_callbacks()

        print("Test environment initialized successfully")

    async def _register_global_callbacks(self):
        """Register callbacks to monitor all inter-service communication."""
        async def global_callback(service_name: str, callback_type: str, data: Dict[str, Any]):
            callback_event = {
                "timestamp": time.time(),
                "service": service_name,
                "type": callback_type,
                "data": data,
                "test_id": getattr(self, '_current_test_id', 'unknown')
            }
            self.callbacks_received.append(callback_event)

        self.mock_services.register_global_callback(global_callback)

    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete integration test suite."""
        print("Starting comprehensive inter-component communication test suite...")

        await self.initialize_test_environment()

        test_results = {
            "suite_start_time": time.time(),
            "configuration": {
                "test_duration_seconds": self.config.test_duration_seconds,
                "concurrent_patterns": self.config.concurrent_patterns,
                "messages_per_pattern": self.config.messages_per_pattern,
                "error_injection_rate": self.config.error_injection_rate
            },
            "tests": {}
        }

        # Test 1: MCP-to-Daemon Communication
        print("\n=== Test 1: MCP-to-Daemon Communication ===")
        test_results["tests"]["mcp_to_daemon"] = await self.test_mcp_to_daemon_communication()

        # Test 2: Daemon-to-MCP Communication
        print("\n=== Test 2: Daemon-to-MCP Communication ===")
        test_results["tests"]["daemon_to_mcp"] = await self.test_daemon_to_mcp_communication()

        # Test 3: CLI-to-Daemon Communication
        print("\n=== Test 3: CLI-to-Daemon Communication ===")
        test_results["tests"]["cli_to_daemon"] = await self.test_cli_to_daemon_communication()

        # Test 4: Bidirectional Communication
        print("\n=== Test 4: Bidirectional Communication ===")
        test_results["tests"]["bidirectional"] = await self.test_bidirectional_communication()

        # Test 5: Error Handling and Recovery
        print("\n=== Test 5: Error Handling and Recovery ===")
        test_results["tests"]["error_handling"] = await self.test_error_handling_scenarios()

        # Test 6: High Concurrency
        print("\n=== Test 6: High Concurrency ===")
        test_results["tests"]["high_concurrency"] = await self.test_high_concurrency_scenarios()

        # Test 7: Network Failure Scenarios
        print("\n=== Test 7: Network Failure Scenarios ===")
        test_results["tests"]["network_failures"] = await self.test_network_failure_scenarios()

        # Test 8: Protocol Validation
        print("\n=== Test 8: Protocol Validation ===")
        test_results["tests"]["protocol_validation"] = await self.test_protocol_validation()

        # Test 9: Service Degradation
        print("\n=== Test 9: Service Degradation ===")
        test_results["tests"]["service_degradation"] = await self.test_service_degradation()

        # Test 10: Load Testing
        print("\n=== Test 10: Load Testing ===")
        test_results["tests"]["load_testing"] = await self.test_load_scenarios()

        # Final metrics and summary
        test_results["suite_end_time"] = time.time()
        test_results["suite_duration_seconds"] = test_results["suite_end_time"] - test_results["suite_start_time"]
        test_results["overall_metrics"] = await self._collect_final_metrics()
        test_results["summary"] = self._generate_test_summary(test_results)

        await self._cleanup_test_environment()

        return test_results

    async def test_mcp_to_daemon_communication(self) -> Dict[str, Any]:
        """Test MCP tool calls to gRPC daemon services."""
        self._current_test_id = "mcp_to_daemon"

        scenario = TestScenario(
            name="mcp_to_daemon_integration",
            pattern=CommunicationPattern.MCP_TO_DAEMON,
            message_count=self.config.messages_per_pattern,
            include_errors=True,
            error_rate=self.config.error_injection_rate
        )

        test_data = self.data_generator.generate_scenario_data(scenario)
        results = {
            "test_name": "MCP-to-Daemon Communication",
            "start_time": time.time(),
            "messages_tested": len(test_data["messages"]),
            "results": [],
            "metrics": CommunicationMetrics(),
            "status": IntegrationTestResult.PASSED.value
        }

        for i, message in enumerate(test_data["messages"]):
            print(f"  Processing message {i+1}/{len(test_data['messages'])}: {message['mcp_tool']}")

            start_time = time.time()
            try:
                # Simulate MCP tool call to corresponding gRPC service
                service_name = message["expected_service"]
                service = self.mock_services.get_service(service_name)

                # Extract method from MCP tool
                method_name = self._mcp_tool_to_grpc_method(message["mcp_tool"])

                # Execute gRPC service call
                response = await service.simulate_request(
                    method_name,
                    message["grpc_request"],
                    simulate_errors=not message.get("has_error", False)  # Don't double-inject errors
                )

                response_time = (time.time() - start_time) * 1000

                results["results"].append({
                    "message_id": i,
                    "mcp_tool": message["mcp_tool"],
                    "service": service_name,
                    "method": method_name,
                    "response_time_ms": response_time,
                    "success": True,
                    "error": None
                })

                results["metrics"].successful_messages += 1

            except Exception as e:
                response_time = (time.time() - start_time) * 1000

                results["results"].append({
                    "message_id": i,
                    "mcp_tool": message["mcp_tool"],
                    "service": message["expected_service"],
                    "response_time_ms": response_time,
                    "success": False,
                    "error": str(e)
                })

                results["metrics"].failed_messages += 1
                if "timeout" in str(e).lower():
                    results["metrics"].timeout_count += 1
                else:
                    results["metrics"].error_count += 1

            results["metrics"].total_messages += 1

            # Update response time metrics
            if response_time > results["metrics"].max_response_time_ms:
                results["metrics"].max_response_time_ms = response_time
            if response_time < results["metrics"].min_response_time_ms:
                results["metrics"].min_response_time_ms = response_time

        # Calculate averages
        if results["metrics"].total_messages > 0:
            total_response_time = sum(r["response_time_ms"] for r in results["results"])
            results["metrics"].avg_response_time_ms = total_response_time / results["metrics"].total_messages

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Determine overall status
        success_rate = results["metrics"].successful_messages / results["metrics"].total_messages
        if success_rate >= 0.95:
            results["status"] = IntegrationTestResult.PASSED.value
        elif success_rate >= 0.80:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['metrics'].successful_messages}/{results['metrics'].total_messages} successful")
        return results

    async def test_daemon_to_mcp_communication(self) -> Dict[str, Any]:
        """Test daemon callbacks to MCP server notifications."""
        self._current_test_id = "daemon_to_mcp"

        results = {
            "test_name": "Daemon-to-MCP Communication",
            "start_time": time.time(),
            "callbacks_tested": 0,
            "callbacks_received": 0,
            "callback_types": {},
            "status": IntegrationTestResult.PASSED.value
        }

        # Clear previous callbacks
        initial_callback_count = len(self.callbacks_received)

        # Trigger various operations that should generate callbacks
        callback_generating_operations = [
            ("DocumentProcessor", "ProcessDocument", {"document_id": "test-doc-1", "content": "test content"}),
            ("DocumentProcessor", "BatchProcess", {"document_ids": ["doc1", "doc2", "doc3"]}),
            ("SearchService", "Search", {"query": "test query", "collection": "test"}),
            ("MemoryService", "CreateCollection", {"collection_name": "test_collection"}),
            ("SystemService", "StartFileWatcher", {"watch_paths": ["/tmp/test"]})
        ]

        for service_name, method_name, request_data in callback_generating_operations:
            try:
                service = self.mock_services.get_service(service_name)
                await service.simulate_request(method_name, request_data)
                results["callbacks_tested"] += 1
                print(f"  Triggered callback operation: {service_name}.{method_name}")
            except Exception as e:
                print(f"  Failed to trigger {service_name}.{method_name}: {e}")

        # Wait for callbacks to be processed
        await asyncio.sleep(0.5)

        # Analyze received callbacks
        new_callbacks = self.callbacks_received[initial_callback_count:]
        results["callbacks_received"] = len(new_callbacks)

        for callback in new_callbacks:
            callback_type = callback["type"]
            if callback_type not in results["callback_types"]:
                results["callback_types"][callback_type] = 0
            results["callback_types"][callback_type] += 1

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate results
        if results["callbacks_received"] >= results["callbacks_tested"] * 0.8:  # 80% callback success rate
            results["status"] = IntegrationTestResult.PASSED.value
        elif results["callbacks_received"] > 0:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['callbacks_received']} callbacks received from {results['callbacks_tested']} operations")
        return results

    async def test_cli_to_daemon_communication(self) -> Dict[str, Any]:
        """Test CLI commands to daemon services."""
        self._current_test_id = "cli_to_daemon"

        from dummy_data.cli_messages import CliCommandGenerator

        cli_gen = CliCommandGenerator()
        common_commands = [
            "wqm service status",
            "wqm admin collections",
            "wqm health check",
            "wqm document process",
            "wqm search query",
            "wqm collection create"
        ]

        results = {
            "test_name": "CLI-to-Daemon Communication",
            "start_time": time.time(),
            "commands_tested": 0,
            "commands_successful": 0,
            "command_results": [],
            "status": IntegrationTestResult.PASSED.value
        }

        for command in common_commands:
            try:
                # Generate CLI command data
                cmd_data = cli_gen.generate_command_data(command)

                # Map CLI command to appropriate service
                service_name = self._cli_command_to_service(command)
                method_name = self._cli_command_to_method(command)

                service = self.mock_services.get_service(service_name)

                start_time = time.time()
                response = await service.simulate_request(method_name, cmd_data)
                response_time = (time.time() - start_time) * 1000

                results["command_results"].append({
                    "command": command,
                    "service": service_name,
                    "method": method_name,
                    "response_time_ms": response_time,
                    "success": True
                })

                results["commands_successful"] += 1
                print(f"  Executed: {command} -> {service_name}.{method_name} ({response_time:.1f}ms)")

            except Exception as e:
                results["command_results"].append({
                    "command": command,
                    "service": service_name if 'service_name' in locals() else "unknown",
                    "success": False,
                    "error": str(e)
                })
                print(f"  Failed: {command} - {e}")

            results["commands_tested"] += 1

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate success rate
        success_rate = results["commands_successful"] / results["commands_tested"] if results["commands_tested"] > 0 else 0
        if success_rate >= 0.90:
            results["status"] = IntegrationTestResult.PASSED.value
        elif success_rate >= 0.70:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['commands_successful']}/{results['commands_tested']} commands successful")
        return results

    async def test_bidirectional_communication(self) -> Dict[str, Any]:
        """Test bidirectional communication scenarios."""
        self._current_test_id = "bidirectional"

        results = {
            "test_name": "Bidirectional Communication",
            "start_time": time.time(),
            "scenarios": [],
            "total_operations": 0,
            "successful_operations": 0,
            "status": IntegrationTestResult.PASSED.value
        }

        # Test document upload with progress callbacks
        scenario_1 = await self._test_document_upload_with_progress()
        results["scenarios"].append(scenario_1)

        # Test streaming search
        scenario_2 = await self._test_streaming_search()
        results["scenarios"].append(scenario_2)

        # Test batch processing with status updates
        scenario_3 = await self._test_batch_processing_with_updates()
        results["scenarios"].append(scenario_3)

        # Aggregate results
        for scenario in results["scenarios"]:
            results["total_operations"] += scenario.get("operations", 1)
            if scenario.get("success", False):
                results["successful_operations"] += scenario.get("operations", 1)

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Determine overall status
        if results["total_operations"] > 0:
            success_rate = results["successful_operations"] / results["total_operations"]
            if success_rate >= 0.90:
                results["status"] = IntegrationTestResult.PASSED.value
            elif success_rate >= 0.70:
                results["status"] = IntegrationTestResult.PARTIAL.value
            else:
                results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['successful_operations']}/{results['total_operations']} bidirectional operations successful")
        return results

    async def test_error_handling_scenarios(self) -> Dict[str, Any]:
        """Test error handling and recovery mechanisms."""
        self._current_test_id = "error_handling"

        results = {
            "test_name": "Error Handling and Recovery",
            "start_time": time.time(),
            "error_scenarios": [],
            "status": IntegrationTestResult.PASSED.value
        }

        # Test service unavailable scenario
        print("  Testing service unavailable scenario...")
        doc_service = self.mock_services.get_service("DocumentProcessor")
        original_state = doc_service.config.state
        doc_service.config.state = ServiceState.ERROR

        try:
            await doc_service.simulate_request("ProcessDocument", {"document_id": "test"})
            results["error_scenarios"].append({"scenario": "service_unavailable", "handled": False})
        except Exception as e:
            results["error_scenarios"].append({
                "scenario": "service_unavailable",
                "handled": True,
                "error_type": type(e).__name__
            })

        # Restore service state
        doc_service.config.state = original_state

        # Test timeout scenario
        print("  Testing timeout scenario...")
        search_service = self.mock_services.get_service("SearchService")
        original_timeout_rate = search_service.config.timeout_rate
        search_service.config.timeout_rate = 1.0  # 100% timeout rate

        try:
            await search_service.simulate_request("Search", {"query": "test"})
            results["error_scenarios"].append({"scenario": "timeout", "handled": False})
        except Exception as e:
            results["error_scenarios"].append({
                "scenario": "timeout",
                "handled": True,
                "error_type": type(e).__name__
            })

        # Restore timeout rate
        search_service.config.timeout_rate = original_timeout_rate

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate error handling
        handled_count = sum(1 for scenario in results["error_scenarios"] if scenario["handled"])
        if handled_count == len(results["error_scenarios"]):
            results["status"] = IntegrationTestResult.PASSED.value
        elif handled_count > 0:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {handled_count}/{len(results['error_scenarios'])} error scenarios handled correctly")
        return results

    async def test_high_concurrency_scenarios(self) -> Dict[str, Any]:
        """Test high concurrency communication patterns."""
        self._current_test_id = "high_concurrency"

        results = {
            "test_name": "High Concurrency",
            "start_time": time.time(),
            "concurrent_operations": self.config.concurrent_patterns * 10,  # More operations for concurrency test
            "completed_operations": 0,
            "failed_operations": 0,
            "average_response_time": 0.0,
            "status": IntegrationTestResult.PASSED.value
        }

        # Create concurrent tasks
        tasks = []
        for i in range(results["concurrent_operations"]):
            # Distribute across different services
            service_name = ["DocumentProcessor", "SearchService", "MemoryService"][i % 3]
            method_name = {
                "DocumentProcessor": "ProcessDocument",
                "SearchService": "Search",
                "MemoryService": "ListCollections"
            }[service_name]

            task = self._execute_concurrent_operation(service_name, method_name, {"request_id": i})
            tasks.append(task)

        print(f"  Executing {len(tasks)} concurrent operations...")
        start_time = time.time()

        # Execute all tasks concurrently
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        execution_time = end_time - start_time

        # Analyze results
        response_times = []
        for result in task_results:
            if isinstance(result, Exception):
                results["failed_operations"] += 1
            else:
                results["completed_operations"] += 1
                response_times.append(result.get("response_time", 0))

        if response_times:
            results["average_response_time"] = sum(response_times) / len(response_times)

        results["total_execution_time"] = execution_time
        results["operations_per_second"] = results["concurrent_operations"] / execution_time

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate concurrency performance
        success_rate = results["completed_operations"] / results["concurrent_operations"]
        if success_rate >= 0.95 and results["operations_per_second"] > 10:
            results["status"] = IntegrationTestResult.PASSED.value
        elif success_rate >= 0.80:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['completed_operations']}/{results['concurrent_operations']} operations ({results['operations_per_second']:.1f} ops/sec)")
        return results

    async def test_network_failure_scenarios(self) -> Dict[str, Any]:
        """Test network failure simulation and recovery."""
        self._current_test_id = "network_failures"

        results = {
            "test_name": "Network Failure Scenarios",
            "start_time": time.time(),
            "failure_scenarios": ["connection_loss", "intermittent", "degraded_performance"],
            "scenario_results": [],
            "status": IntegrationTestResult.PASSED.value
        }

        for scenario in results["failure_scenarios"]:
            print(f"  Testing {scenario} scenario...")
            scenario_result = await self._simulate_network_failure(scenario)
            results["scenario_results"].append(scenario_result)

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate network failure handling
        handled_scenarios = sum(1 for s in results["scenario_results"] if s.get("recovery_successful", False))
        if handled_scenarios == len(results["failure_scenarios"]):
            results["status"] = IntegrationTestResult.PASSED.value
        elif handled_scenarios > 0:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {handled_scenarios}/{len(results['failure_scenarios'])} network failure scenarios handled")
        return results

    async def test_protocol_validation(self) -> Dict[str, Any]:
        """Test protocol validation and message format compliance."""
        self._current_test_id = "protocol_validation"

        results = {
            "test_name": "Protocol Validation",
            "start_time": time.time(),
            "validation_tests": 0,
            "validation_failures": 0,
            "protocol_errors": [],
            "status": IntegrationTestResult.PASSED.value
        }

        # Test MCP protocol compliance
        from dummy_data.mcp_messages import McpMessageGenerator
        mcp_gen = McpMessageGenerator()

        test_tools = ["add_document", "search_workspace", "list_collections"]

        for tool in test_tools:
            request = mcp_gen.generate_tool_request(tool)
            results["validation_tests"] += 1

            # Validate MCP protocol structure
            validation_errors = self._validate_mcp_message(request)
            if validation_errors:
                results["validation_failures"] += 1
                results["protocol_errors"].extend(validation_errors)

        # Test gRPC message validation
        from dummy_data.grpc_messages import GrpcMessageGenerator
        grpc_gen = GrpcMessageGenerator()

        services = ["DocumentProcessor", "SearchService", "MemoryService"]
        methods = ["ProcessDocument", "Search", "CreateCollection"]

        for service, method in zip(services, methods):
            request = grpc_gen.generate_service_message(service, method)
            results["validation_tests"] += 1

            # Validate gRPC message structure
            validation_errors = self._validate_grpc_message(request)
            if validation_errors:
                results["validation_failures"] += 1
                results["protocol_errors"].extend(validation_errors)

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate protocol compliance
        if results["validation_failures"] == 0:
            results["status"] = IntegrationTestResult.PASSED.value
        elif results["validation_failures"] < results["validation_tests"] * 0.1:  # Less than 10% failures
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['validation_tests'] - results['validation_failures']}/{results['validation_tests']} protocol validations passed")
        return results

    async def test_service_degradation(self) -> Dict[str, Any]:
        """Test service degradation scenarios."""
        self._current_test_id = "service_degradation"

        results = {
            "test_name": "Service Degradation",
            "start_time": time.time(),
            "degradation_scenarios": [],
            "status": IntegrationTestResult.PASSED.value
        }

        # Test gradual performance degradation
        search_service = self.mock_services.get_service("SearchService")
        original_latency = search_service.config.latency_ms

        degradation_levels = [200, 500, 1000, 2000]  # Increasing latency

        for latency in degradation_levels:
            search_service.config.latency_ms = latency

            start_time = time.time()
            try:
                await search_service.simulate_request("Search", {"query": "test"})
                response_time = (time.time() - start_time) * 1000

                results["degradation_scenarios"].append({
                    "configured_latency": latency,
                    "actual_response_time": response_time,
                    "degradation_detected": response_time > latency * 0.8,  # Within 20% tolerance
                    "success": True
                })

            except Exception as e:
                results["degradation_scenarios"].append({
                    "configured_latency": latency,
                    "success": False,
                    "error": str(e)
                })

        # Restore original latency
        search_service.config.latency_ms = original_latency

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate degradation handling
        successful_scenarios = sum(1 for s in results["degradation_scenarios"] if s["success"])
        if successful_scenarios == len(results["degradation_scenarios"]):
            results["status"] = IntegrationTestResult.PASSED.value
        elif successful_scenarios > len(results["degradation_scenarios"]) * 0.7:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {successful_scenarios}/{len(results['degradation_scenarios'])} degradation scenarios handled")
        return results

    async def test_load_scenarios(self) -> Dict[str, Any]:
        """Test high-load communication scenarios."""
        self._current_test_id = "load_testing"

        load_data = await self.data_generator.generate_load_test_data(
            concurrent_patterns=self.config.concurrent_patterns,
            duration_seconds=min(30, self.config.test_duration_seconds),  # Limit load test duration
            requests_per_second=50  # Moderate load for testing
        )

        results = {
            "test_name": "Load Testing",
            "start_time": time.time(),
            "total_requests": load_data["load_test_config"]["total_requests"],
            "successful_requests": 0,
            "failed_requests": 0,
            "average_throughput": 0.0,
            "status": IntegrationTestResult.PASSED.value
        }

        print(f"  Executing load test with {results['total_requests']} requests...")

        # Execute load test
        load_start_time = time.time()
        tasks = []

        for pattern in load_data["patterns"]:
            for message in pattern["messages"]:
                task = self._execute_load_test_message(message)
                tasks.append(task)

        # Execute all load test tasks
        load_results = await asyncio.gather(*tasks, return_exceptions=True)

        load_end_time = time.time()
        load_duration = load_end_time - load_start_time

        # Analyze load test results
        for result in load_results:
            if isinstance(result, Exception):
                results["failed_requests"] += 1
            else:
                results["successful_requests"] += 1

        results["load_duration_seconds"] = load_duration
        results["average_throughput"] = results["total_requests"] / load_duration

        results["end_time"] = time.time()
        results["duration_seconds"] = results["end_time"] - results["start_time"]

        # Evaluate load test performance
        success_rate = results["successful_requests"] / results["total_requests"]
        if success_rate >= 0.95 and results["average_throughput"] > 20:  # Good throughput
            results["status"] = IntegrationTestResult.PASSED.value
        elif success_rate >= 0.85:
            results["status"] = IntegrationTestResult.PARTIAL.value
        else:
            results["status"] = IntegrationTestResult.FAILED.value

        print(f"  Completed: {results['successful_requests']}/{results['total_requests']} requests ({results['average_throughput']:.1f} req/sec)")
        return results

    # Helper methods
    def _mcp_tool_to_grpc_method(self, mcp_tool: str) -> str:
        """Map MCP tool to gRPC method name."""
        mappings = {
            "add_document": "ProcessDocument",
            "get_document": "GetDocumentStatus",
            "search_workspace": "Search",
            "hybrid_search_advanced": "HybridSearch",
            "list_collections": "ListCollections",
            "create_workspace_collection": "CreateCollection",
            "setup_folder_watch": "StartFileWatcher",
            "workspace_status": "GetSystemStatus"
        }
        return mappings.get(mcp_tool, "UnknownMethod")

    def _cli_command_to_service(self, command: str) -> str:
        """Map CLI command to service name."""
        if "service" in command:
            return "SystemService"
        elif "admin" in command or "collection" in command:
            return "MemoryService"
        elif "health" in command:
            return "SystemService"
        elif "document" in command:
            return "DocumentProcessor"
        elif "search" in command:
            return "SearchService"
        else:
            return "SystemService"

    def _cli_command_to_method(self, command: str) -> str:
        """Map CLI command to method name."""
        if "status" in command:
            return "GetSystemStatus"
        elif "collections" in command:
            return "ListCollections"
        elif "health" in command:
            return "GetSystemStatus"
        elif "process" in command:
            return "ProcessDocument"
        elif "search" in command:
            return "Search"
        else:
            return "GetSystemStatus"

    async def _test_document_upload_with_progress(self) -> Dict[str, Any]:
        """Test document upload with progress callbacks."""
        doc_service = self.mock_services.get_service("DocumentProcessor")

        callback_count_before = len(self.callbacks_received)

        # Trigger batch processing which generates progress callbacks
        await doc_service.batch_process({
            "document_ids": [f"doc_{i}" for i in range(5)],
            "processing_options": {"parallel": True}
        })

        # Wait for progress callbacks
        await asyncio.sleep(1.0)

        new_callbacks = self.callbacks_received[callback_count_before:]
        progress_callbacks = [cb for cb in new_callbacks if cb["type"] == "progress_update"]

        return {
            "scenario": "document_upload_with_progress",
            "operations": 1,
            "progress_callbacks_received": len(progress_callbacks),
            "success": len(progress_callbacks) > 0
        }

    async def _test_streaming_search(self) -> Dict[str, Any]:
        """Test streaming search operation."""
        search_service = self.mock_services.get_service("SearchService")

        try:
            stream_responses = await search_service.search_stream({
                "query": "test streaming search",
                "collection": "test_collection"
            })

            return {
                "scenario": "streaming_search",
                "operations": len(stream_responses),
                "success": len(stream_responses) > 0
            }
        except Exception as e:
            return {
                "scenario": "streaming_search",
                "operations": 0,
                "success": False,
                "error": str(e)
            }

    async def _test_batch_processing_with_updates(self) -> Dict[str, Any]:
        """Test batch processing with status updates."""
        doc_service = self.mock_services.get_service("DocumentProcessor")

        callback_count_before = len(self.callbacks_received)

        try:
            await doc_service.batch_process({
                "document_ids": [f"batch_doc_{i}" for i in range(3)]
            })

            # Wait for completion callbacks
            await asyncio.sleep(1.5)

            new_callbacks = self.callbacks_received[callback_count_before:]
            batch_callbacks = [cb for cb in new_callbacks if "batch" in cb.get("type", "")]

            return {
                "scenario": "batch_processing_with_updates",
                "operations": 3,  # 3 documents in batch
                "batch_callbacks_received": len(batch_callbacks),
                "success": len(batch_callbacks) > 0
            }
        except Exception as e:
            return {
                "scenario": "batch_processing_with_updates",
                "operations": 3,
                "success": False,
                "error": str(e)
            }

    async def _execute_concurrent_operation(self, service_name: str, method_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single concurrent operation."""
        start_time = time.time()
        try:
            service = self.mock_services.get_service(service_name)
            response = await service.simulate_request(method_name, request_data)
            response_time = (time.time() - start_time) * 1000

            return {
                "service": service_name,
                "method": method_name,
                "response_time": response_time,
                "success": True
            }
        except Exception as e:
            return {
                "service": service_name,
                "method": method_name,
                "response_time": (time.time() - start_time) * 1000,
                "success": False,
                "error": str(e)
            }

    async def _simulate_network_failure(self, failure_type: str) -> Dict[str, Any]:
        """Simulate different types of network failures."""
        result = {
            "failure_type": failure_type,
            "recovery_successful": False,
            "recovery_time_seconds": 0.0
        }

        search_service = self.mock_services.get_service("SearchService")
        original_error_rate = search_service.config.error_rate
        original_latency = search_service.config.latency_ms

        start_time = time.time()

        if failure_type == "connection_loss":
            # Simulate complete connection loss
            search_service.config.error_rate = 1.0  # 100% failure

            # Try to recover after short period
            await asyncio.sleep(0.5)
            search_service.config.error_rate = original_error_rate

            # Test recovery
            try:
                await search_service.simulate_request("Search", {"query": "recovery test"})
                result["recovery_successful"] = True
            except Exception:
                result["recovery_successful"] = False

        elif failure_type == "intermittent":
            # Simulate intermittent failures
            search_service.config.error_rate = 0.5  # 50% failure rate

            attempts = 5
            successful_attempts = 0

            for _ in range(attempts):
                try:
                    await search_service.simulate_request("Search", {"query": "intermittent test"})
                    successful_attempts += 1
                except Exception:
                    pass
                await asyncio.sleep(0.1)

            result["recovery_successful"] = successful_attempts > 0
            search_service.config.error_rate = original_error_rate

        elif failure_type == "degraded_performance":
            # Simulate degraded performance
            search_service.config.latency_ms = 5000  # Very slow

            try:
                await search_service.simulate_request("Search", {"query": "degraded test"})
                result["recovery_successful"] = True  # Still works, just slow
            except Exception:
                result["recovery_successful"] = False

            search_service.config.latency_ms = original_latency

        result["recovery_time_seconds"] = time.time() - start_time
        return result

    async def _execute_load_test_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single load test message."""
        try:
            pattern = message.get("pattern", "unknown")

            if pattern == "mcp_to_daemon":
                service_name = message["expected_service"]
                method_name = self._mcp_tool_to_grpc_method(message["mcp_tool"])
            elif pattern == "cli_to_daemon":
                service_name = self._cli_command_to_service(message["cli_command"])
                method_name = self._cli_command_to_method(message["cli_command"])
            else:
                # Default to a simple operation
                service_name = "SystemService"
                method_name = "GetSystemStatus"

            service = self.mock_services.get_service(service_name)
            await service.simulate_request(method_name, {})

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _validate_mcp_message(self, message: Dict[str, Any]) -> List[str]:
        """Validate MCP message format."""
        errors = []

        # Check required MCP fields
        if message.get("jsonrpc") != "2.0":
            errors.append("Invalid or missing jsonrpc field")

        if "method" not in message:
            errors.append("Missing method field")

        if "params" not in message:
            errors.append("Missing params field")

        if "id" not in message:
            errors.append("Missing id field")

        return errors

    def _validate_grpc_message(self, message: Dict[str, Any]) -> List[str]:
        """Validate gRPC message format."""
        errors = []

        # Check required gRPC fields
        if "message_type" not in message:
            errors.append("Missing message_type field")

        # Check for reasonable message structure
        if len(message) < 2:
            errors.append("Message too small, missing required fields")

        return errors

    async def _collect_final_metrics(self) -> Dict[str, Any]:
        """Collect final metrics from all services."""
        service_metrics = self.mock_services.get_all_metrics()

        total_requests = sum(metrics["total_requests"] for metrics in service_metrics.values())
        total_successful = sum(metrics["successful_requests"] for metrics in service_metrics.values())
        total_failed = sum(metrics["failed_requests"] for metrics in service_metrics.values())

        avg_response_times = [
            metrics["avg_response_time_ms"]
            for metrics in service_metrics.values()
            if metrics["avg_response_time_ms"] > 0
        ]

        overall_avg_response_time = sum(avg_response_times) / len(avg_response_times) if avg_response_times else 0

        return {
            "service_metrics": service_metrics,
            "overall_totals": {
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "success_rate": total_successful / max(1, total_requests),
                "overall_avg_response_time_ms": overall_avg_response_time
            },
            "callbacks": {
                "total_callbacks_received": len(self.callbacks_received),
                "callback_types": {}
            },
            "protocol_violations": len(self.protocol_violations)
        }

    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        tests = test_results["tests"]

        total_tests = len(tests)
        passed_tests = sum(1 for test in tests.values() if test.get("status") == IntegrationTestResult.PASSED.value)
        partial_tests = sum(1 for test in tests.values() if test.get("status") == IntegrationTestResult.PARTIAL.value)
        failed_tests = sum(1 for test in tests.values() if test.get("status") == IntegrationTestResult.FAILED.value)

        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "partial_tests": partial_tests,
            "failed_tests": failed_tests,
            "overall_success_rate": passed_tests / max(1, total_tests),
            "test_duration_seconds": test_results["suite_duration_seconds"],
            "overall_status": (
                IntegrationTestResult.PASSED.value if failed_tests == 0 and partial_tests <= 1
                else IntegrationTestResult.PARTIAL.value if failed_tests <= total_tests * 0.2
                else IntegrationTestResult.FAILED.value
            )
        }

    async def _cleanup_test_environment(self):
        """Clean up test environment."""
        print("\nCleaning up test environment...")

        if self.mock_services:
            await self.mock_services.shutdown_all()

        print("Test environment cleanup completed")


# Example usage and test runner
async def run_integration_tests():
    """Run the complete integration test suite."""
    config = IntegrationTestConfig(
        test_duration_seconds=120,  # 2 minutes
        concurrent_patterns=5,
        messages_per_pattern=50,
        error_injection_rate=0.1,
        timeout_simulation_rate=0.05,
        network_failure_rate=0.02
    )

    test_suite = CommunicationTestSuite(config)

    print("=" * 80)
    print("WORKSPACE-QDRANT-MCP INTER-COMPONENT COMMUNICATION TEST SUITE")
    print("=" * 80)

    results = await test_suite.run_comprehensive_test_suite()

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(f"Suite Duration: {results['suite_duration_seconds']:.1f} seconds")
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Partial: {results['summary']['partial_tests']}")
    print(f"Failed: {results['summary']['failed_tests']}")
    print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.1%}")
    print(f"Overall Status: {results['summary']['overall_status'].upper()}")

    # Export detailed results
    results_file = f"/tmp/integration_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDetailed results exported to: {results_file}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_integration_tests())
    else:
        print("Usage: python integration_test_suite.py run")
        print("This will run the complete inter-component communication test suite")