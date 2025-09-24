# Inter-Component Communication Testing Framework

A comprehensive testing framework for validating communication patterns in the workspace-qdrant-mcp system, supporting MCP-to-daemon, daemon-to-MCP, and CLI-to-daemon interactions with realistic error simulation and edge case testing.

## Overview

This framework provides dummy data simulation and mock services for testing all communication patterns in the workspace-qdrant-mcp system without requiring actual services to be running. It includes comprehensive error injection, timeout simulation, protocol validation, and performance testing capabilities.

### Supported Communication Patterns

1. **MCP-to-daemon**: Python MCP server tools → gRPC → Rust daemon services
2. **Daemon-to-MCP**: Rust daemon services → gRPC → Python MCP server callbacks
3. **CLI-to-daemon**: CLI utility (wqm) → gRPC → Rust daemon services
4. **Bidirectional**: Complex scenarios with request-response-callback chains

### Key Features

- **Comprehensive Data Generation**: Realistic dummy data for all message types and protocols
- **Mock Services**: Full mock implementations of all 5 gRPC services with realistic behavior
- **Error Injection**: Network failures, timeouts, data corruption, service unavailability
- **Protocol Validation**: MCP and gRPC message format compliance testing
- **Performance Testing**: Load testing, concurrency testing, degradation scenarios
- **Metrics Collection**: Detailed performance and reliability metrics
- **Integration Testing**: Multi-component scenario testing with callbacks and state management

## Architecture

```
20250924-1333_communication_test_framework/
├── __init__.py                     # Framework entry point
├── dummy_data/                     # Data generation modules
│   ├── generators.py               # Main data generator orchestrator
│   ├── grpc_messages.py           # gRPC message generators (5 services)
│   ├── mcp_messages.py            # MCP tool message generators (30+ tools)
│   ├── cli_messages.py            # CLI command generators (wqm commands)
│   ├── qdrant_data.py             # Qdrant operations data generators
│   └── project_data.py            # Project context and workspace data
├── mock_services/                  # Mock service implementations
│   ├── grpc_services.py           # Mock gRPC services with realistic behavior
│   ├── mcp_server.py              # Mock MCP server (planned)
│   ├── cli_interface.py           # Mock CLI interface (planned)
│   └── orchestrator.py            # Service coordination (planned)
├── test_communication_framework.py # Comprehensive unit tests
├── integration_test_suite.py      # Full integration test suite
└── README.md                      # This documentation
```

## Quick Start

### Basic Usage

```python
from dummy_data.generators import DummyDataGenerator, CommunicationPattern, TestScenario
from mock_services.grpc_services import MockGrpcServices

# Generate test data
data_gen = DummyDataGenerator(seed=42)
scenario = TestScenario(
    name="mcp_to_daemon_test",
    pattern=CommunicationPattern.MCP_TO_DAEMON,
    message_count=10,
    include_errors=True,
    error_rate=0.1
)

test_data = data_gen.generate_scenario_data(scenario)

# Setup mock services
mock_services = MockGrpcServices()
doc_service = mock_services.get_service("DocumentProcessor")

# Execute test
response = await doc_service.process_document({
    "document_id": "test-doc-123",
    "content": "Test content",
    "metadata": {"type": "test"}
})
```

### Running Integration Tests

```bash
# Run the complete integration test suite
python integration_test_suite.py run

# Run specific unit test categories
python test_communication_framework.py data    # Data generation tests
python test_communication_framework.py grpc    # gRPC message tests
python test_communication_framework.py mcp     # MCP message tests
python test_communication_framework.py mock    # Mock services tests
python test_communication_framework.py edge    # Edge case tests

# Run all unit tests with coverage
python test_communication_framework.py
```

## Components

### Dummy Data Generators

#### Main Generator (`generators.py`)
- **DummyDataGenerator**: Central orchestrator for all data generation
- **TestScenario**: Configuration for test scenarios with error injection
- **CommunicationPattern**: Enum for different communication types
- **Load testing**: Generate high-volume test datasets

#### gRPC Messages (`grpc_messages.py`)
- **All 5 Services**: DocumentProcessor, SearchService, MemoryService, SystemService, ServiceDiscovery
- **Request/Response**: Generate realistic gRPC request and response messages
- **Callbacks**: Generate callback messages for async operations
- **Streaming**: Support for streaming operations and batch processing
- **Error Responses**: Generate various error scenarios

#### MCP Messages (`mcp_messages.py`)
- **30+ Tools**: Complete coverage of all MCP server tools
- **Tool Categories**: Document management, search, collection management, multi-tenant, etc.
- **Protocol Compliance**: Generate MCP 2.0 compliant messages
- **Notifications**: Generate notifications from gRPC callbacks
- **Batch Operations**: Support for batch MCP requests

#### CLI Commands (`cli_messages.py`)
- **Complete wqm Suite**: Service, admin, health, document, search, collection, watch, config commands
- **Option Generation**: Realistic command-line options and arguments
- **Error Scenarios**: Invalid options, permission denied, service unavailable
- **Command Sequences**: Dependencies and execution ordering
- **Load Testing**: Generate high-frequency command execution

#### Qdrant Data (`qdrant_data.py`)
- **Vector Generation**: Normal, uniform, and sparse distributions
- **Point Management**: Realistic document points with metadata
- **Collection Configuration**: HNSW, WAL, quantization settings
- **Search Operations**: Hybrid search, batch operations, scroll requests
- **Error Simulation**: Collection not found, timeout, internal errors

#### Project Data (`project_data.py`)
- **500+ Languages**: Complete language configurations with LSP servers
- **Project Templates**: Web apps, microservices, ML pipelines, CLI tools, mobile apps
- **Git Context**: Repository information, branches, commits, submodules
- **Workspace Management**: Multi-project workspaces and collection mapping

### Mock Services

#### gRPC Services (`grpc_services.py`)
- **Realistic Behavior**: Configurable latency, error rates, timeout simulation
- **Service State**: Running, degraded, error, stopped states
- **Metrics Collection**: Request counts, response times, error tracking
- **Callback System**: Async operation notifications
- **Concurrent Handling**: Request limiting and overload simulation

**Services Implemented:**
- **MockDocumentProcessor**: Document processing with batch support and progress tracking
- **MockSearchService**: Search operations including hybrid search and streaming
- **MockMemoryService**: Collection management with backup and statistics
- **MockSystemService**: System status and file watching with event simulation
- **MockServiceDiscovery**: Service registration and health monitoring

### Test Suites

#### Unit Tests (`test_communication_framework.py`)
- **Data Generation**: Test all dummy data generators with edge cases
- **Message Validation**: Verify protocol compliance and message structure
- **Mock Services**: Test service behavior, error injection, and callbacks
- **Edge Cases**: Network failures, timeouts, corruption, service unavailability
- **Coverage Testing**: Comprehensive test coverage with pytest and coverage.py

#### Integration Tests (`integration_test_suite.py`)
- **10 Test Categories**: Complete communication pattern validation
- **Multi-Component**: Complex scenarios with service interactions
- **Performance Testing**: Load testing, concurrency, degradation scenarios
- **Error Recovery**: Failure simulation and recovery validation
- **Metrics Collection**: Detailed performance and reliability metrics

**Test Categories:**
1. MCP-to-Daemon Communication
2. Daemon-to-MCP Communication
3. CLI-to-Daemon Communication
4. Bidirectional Communication
5. Error Handling and Recovery
6. High Concurrency
7. Network Failure Scenarios
8. Protocol Validation
9. Service Degradation
10. Load Testing

## Configuration

### TestScenario Configuration

```python
scenario = TestScenario(
    name="comprehensive_test",
    pattern=CommunicationPattern.MCP_TO_DAEMON,
    message_count=100,
    include_errors=True,
    error_rate=0.1,              # 10% error injection
    timeout_scenarios=True,
    data_corruption=True,
    network_partition=True
)
```

### MockServiceConfig Configuration

```python
config = MockServiceConfig(
    latency_ms=100,              # Base latency
    error_rate=0.05,             # 5% error rate
    timeout_rate=0.02,           # 2% timeout rate
    max_concurrent_requests=100,
    enable_callbacks=True,
    callback_delay_ms=50
)
```

### IntegrationTestConfig Configuration

```python
config = IntegrationTestConfig(
    test_duration_seconds=120,
    concurrent_patterns=5,
    messages_per_pattern=100,
    error_injection_rate=0.1,
    timeout_simulation_rate=0.05,
    network_failure_rate=0.02,
    enable_callbacks=True,
    validate_protocol=True,
    collect_metrics=True
)
```

## Error Simulation

### Supported Error Types

- **Network Failures**: Connection loss, intermittent connectivity, degraded performance
- **Timeouts**: Connection timeouts, read timeouts, write timeouts
- **Data Corruption**: Truncated messages, invalid encoding, malformed data
- **Service Errors**: Unavailable, overloaded, internal errors
- **Protocol Errors**: Invalid requests, missing fields, type mismatches

### Error Injection

```python
# Inject various error conditions
message = data_gen._inject_error(message)           # General errors
message = data_gen._inject_timeout_scenario(message) # Timeout scenarios
message = data_gen._corrupt_data(message)           # Data corruption
message = data_gen._simulate_network_partition(message) # Network issues
```

## Metrics and Reporting

### Communication Metrics

```python
@dataclass
class CommunicationMetrics:
    total_messages: int
    successful_messages: int
    failed_messages: int
    avg_response_time_ms: float
    timeout_count: int
    error_count: int
    protocol_errors: int
    network_failures: int
```

### Service Metrics

```python
metrics = service.get_metrics()
# Returns: total_requests, successful_requests, failed_requests,
#          avg_response_time_ms, error_rates, concurrent_requests
```

### Test Results Export

```python
# Export test results to JSON
data_gen.export_scenario_data(data, "test_results.json")

# Integration test results are automatically exported
results_file = f"/tmp/integration_test_results_{timestamp}.json"
```

## Advanced Features

### Load Testing

```python
# Generate load test dataset
load_data = await data_gen.generate_load_test_data(
    concurrent_patterns=10,
    duration_seconds=60,
    requests_per_second=100
)

# Execute load test
test_suite = CommunicationTestSuite(config)
results = await test_suite.test_load_scenarios()
```

### Callback Monitoring

```python
# Register callback to monitor all service events
async def callback_monitor(service_name, callback_type, data):
    print(f"Callback: {service_name} -> {callback_type}")

mock_services.register_global_callback(callback_monitor)
```

### Protocol Validation

```python
# Validate MCP protocol compliance
errors = test_suite._validate_mcp_message(mcp_request)

# Validate gRPC message structure
errors = test_suite._validate_grpc_message(grpc_request)
```

### Service State Management

```python
# Change service states for testing
service.config.state = ServiceState.DEGRADED
service.config.state = ServiceState.ERROR
service.config.state = ServiceState.STOPPED

# Update service configuration
service.update_config(
    latency_ms=500,
    error_rate=0.2,
    timeout_rate=0.1
)
```

## Example Scenarios

### MCP-to-Daemon Document Processing

```python
# Generate MCP tool request
mcp_gen = McpMessageGenerator()
request = mcp_gen.generate_tool_request("add_document", {
    "content": "Test document content",
    "title": "Test Document"
})

# Convert to gRPC request
grpc_gen = GrpcMessageGenerator()
grpc_request = grpc_gen.generate_corresponding_grpc_message(
    "add_document", request["params"]["arguments"]
)

# Execute with mock service
doc_service = mock_services.get_service("DocumentProcessor")
response = await doc_service.process_document(grpc_request)
```

### Daemon-to-MCP Callbacks

```python
# Register callback handler
callbacks_received = []

async def callback_handler(callback_type, data):
    callbacks_received.append((callback_type, data))

doc_service.register_callback(callback_handler)

# Trigger operation that generates callback
await doc_service.batch_process({
    "document_ids": ["doc1", "doc2", "doc3"]
})

# Wait for callbacks
await asyncio.sleep(1.0)
print(f"Received {len(callbacks_received)} callbacks")
```

### CLI-to-Daemon Commands

```python
# Generate CLI command
cli_gen = CliCommandGenerator()
cmd_data = cli_gen.generate_command_data("wqm service status")

# Map to service call
service_name = "SystemService"
method_name = "GetSystemStatus"

# Execute
system_service = mock_services.get_service(service_name)
response = await system_service.simulate_request(method_name, cmd_data)
```

### Error Recovery Testing

```python
# Test service unavailable scenario
doc_service = mock_services.get_service("DocumentProcessor")
doc_service.config.state = ServiceState.ERROR

try:
    await doc_service.process_document({"document_id": "test"})
except ServiceUnavailableError:
    print("Service unavailable error handled correctly")

# Restore service and test recovery
doc_service.config.state = ServiceState.RUNNING
response = await doc_service.process_document({"document_id": "test"})
print("Service recovered successfully")
```

## Dependencies

### Python Packages
- `asyncio`: Async operation support
- `pytest`: Unit testing framework
- `uuid`: Unique identifier generation
- `json`: Data serialization
- `time`: Timing and timestamps
- `random`: Randomization for realistic data
- `numpy`: Vector operations (for Qdrant data)
- `dataclasses`: Configuration and result objects
- `typing`: Type hints and annotations
- `enum`: Enumeration support

### Development Dependencies
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities

## Usage Examples

### Simple Communication Test

```python
import asyncio
from dummy_data.generators import DummyDataGenerator, CommunicationPattern, TestScenario
from mock_services.grpc_services import MockGrpcServices

async def simple_test():
    # Setup
    data_gen = DummyDataGenerator()
    mock_services = MockGrpcServices()

    # Generate test scenario
    scenario = TestScenario(
        name="simple_test",
        pattern=CommunicationPattern.MCP_TO_DAEMON,
        message_count=5
    )

    test_data = data_gen.generate_scenario_data(scenario)

    # Execute tests
    for message in test_data["messages"]:
        service = mock_services.get_service(message["expected_service"])
        method = message["mcp_tool"]

        try:
            response = await service.simulate_request(
                method, message["grpc_request"]
            )
            print(f"✓ {method}: Success")
        except Exception as e:
            print(f"✗ {method}: {e}")

# Run test
asyncio.run(simple_test())
```

### Comprehensive Test Suite

```python
import asyncio
from integration_test_suite import CommunicationTestSuite, IntegrationTestConfig

async def comprehensive_test():
    config = IntegrationTestConfig(
        test_duration_seconds=60,
        concurrent_patterns=3,
        messages_per_pattern=25,
        error_injection_rate=0.1
    )

    test_suite = CommunicationTestSuite(config)
    results = await test_suite.run_comprehensive_test_suite()

    print(f"Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed_tests']}")
    print(f"Success Rate: {results['summary']['overall_success_rate']:.1%}")

# Run comprehensive test
asyncio.run(comprehensive_test())
```

## Contributing

This framework is designed to be extensible. To add new communication patterns or test scenarios:

1. **Add Data Generators**: Extend existing generators or create new ones
2. **Add Mock Services**: Implement additional mock services as needed
3. **Add Test Cases**: Extend unit tests and integration tests
4. **Add Error Scenarios**: Implement new error injection patterns
5. **Add Metrics**: Extend metrics collection for new scenarios

## License

This framework is part of the workspace-qdrant-mcp project and follows the same licensing terms.

## Support

For issues, questions, or contributions related to this testing framework, please refer to the main workspace-qdrant-mcp project documentation and issue tracking.

---

**Generated as part of task 244: Inter-Component Communication Testing with Dummy Data**

This framework provides comprehensive testing capabilities for validating all communication patterns in the workspace-qdrant-mcp system, ensuring robust and reliable inter-component interactions.