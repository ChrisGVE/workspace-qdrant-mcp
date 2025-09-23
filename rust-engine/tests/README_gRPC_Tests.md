# gRPC Protocol Correctness Integration Tests

This directory contains comprehensive gRPC protocol integration tests designed to validate protocol correctness, serialization, service routing, and compliance for the workspace-qdrant-daemon.

## Test Suite Overview

### Files Structure

- **`grpc_protocol_basic.rs`** - Basic protocol functionality tests
- **`grpc_protocol_comprehensive.rs`** - Comprehensive protocol validation tests
- **`grpc_protocol_integration.rs`** - Full bidirectional streaming and advanced protocol tests
- **`../src/grpc/shared_test_utils.rs`** - Shared test utilities and infrastructure

### Coverage Goals

- **Target: 90%+ coverage** of gRPC protocol implementation
- **Validation method**: `cargo tarpaulin` for coverage measurement
- **Testing approach**: TDD (Test-Driven Development) with comprehensive edge case coverage

## Test Categories

### 1. Protocol Buffer Message Serialization Tests

**Purpose**: Validate correct serialization/deserialization of all protobuf messages

**Coverage**:
- Complex nested message structures
- Unicode content and special characters
- Optional and repeated fields
- All enum value combinations
- Large message payloads
- Round-trip serialization validation

**Key Tests**:
- `test_protobuf_serialization_document_processing()`
- `test_protobuf_serialization_search_operations()`
- `test_protobuf_serialization_memory_operations()`

### 2. Service Method Routing and Dispatch Tests

**Purpose**: Ensure all gRPC service methods are correctly routed and dispatched

**Coverage**:
- DocumentProcessor service: `ProcessDocument`, `GetProcessingStatus`, `CancelProcessing`
- SearchService routing: `HybridSearch`, `SemanticSearch`, `KeywordSearch`
- SystemService routing: `HealthCheck`, `GetStatus`, `GetConfig`
- MemoryService operations: `AddDocument`, `GetDocument`, `UpdateDocument`, etc.

**Key Tests**:
- `test_document_processor_service_routing()`
- `test_search_service_routing()`
- `test_system_service_routing()`

### 3. Error Status Code Mapping Tests

**Purpose**: Validate proper gRPC error code mapping and propagation

**Coverage**:
- `INVALID_ARGUMENT` for malformed requests
- `NOT_FOUND` for non-existent resources
- `DEADLINE_EXCEEDED` for timeout scenarios
- `RESOURCE_EXHAUSTED` for rate limiting
- `UNAVAILABLE` for service unavailability

**Key Tests**:
- `test_grpc_error_status_codes()`
- `test_grpc_timeout_handling()`

### 4. Metadata and Header Propagation Tests

**Purpose**: Ensure metadata and headers are correctly propagated through the protocol stack

**Coverage**:
- Client metadata injection and propagation
- Authentication headers (Bearer tokens, API keys)
- Custom headers and trace IDs
- Binary metadata handling

**Key Tests**:
- `test_grpc_metadata_propagation()`
- `test_grpc_authentication_headers()`

### 5. Bidirectional Streaming Tests

**Purpose**: Validate streaming protocols with proper flow control

**Coverage**:
- Request/response streaming
- Backpressure handling
- Stream error recovery
- Concurrent stream management
- Flow control mechanisms

**Key Tests**:
- `test_bidirectional_streaming_document_processing()`
- `test_streaming_flow_control_and_backpressure()`
- `test_streaming_error_handling_and_recovery()`

### 6. Protocol Compliance Tests

**Purpose**: Ensure full gRPC protocol compliance and reflection support

**Coverage**:
- gRPC reflection service integration
- Protocol buffer schema validation
- Enum value serialization compliance
- Complex nested message validation

**Key Tests**:
- `test_enum_values_protocol_compliance()`
- `test_protobuf_schema_compliance()`
- `test_complex_nested_message_validation()`

### 7. Concurrent Protocol Access Tests

**Purpose**: Validate protocol correctness under concurrent load

**Coverage**:
- Multiple concurrent clients
- Cross-service concurrent requests
- Error distribution analysis
- Success rate validation (80%+ threshold)
- Resource contention handling

**Key Tests**:
- `test_concurrent_protocol_correctness()`
- `test_concurrent_protocol_access()`

## Shared Test Utilities

### TestGrpcServer

Provides isolated test server instances with ephemeral ports:

```rust
let server = TestGrpcServer::start().await?;
let clients = server.get_clients().await?;
// ... run tests
server.shutdown().await;
```

### TestDataFactory

Factory methods for creating test data:

```rust
let request = TestDataFactory::create_process_document_request(
    "/test/file.txt",
    "project_id",
    "collection_name",
    DocumentType::Text
);
```

### TestValidators

Validation helpers for protocol compliance:

```rust
assert!(TestValidators::validate_process_document_response(&response));
assert!(TestValidators::validate_recent_timestamp(&timestamp));
```

### ConcurrentTestRunner

Utilities for concurrent testing:

```rust
let results = ConcurrentTestRunner::run_concurrent_requests(50, |i| {
    // async request logic
}).await;

let stats = ConcurrentTestRunner::analyze_results(&results);
assert!(stats.meets_success_threshold(0.8));
```

## Running Tests

### Individual Test Suites

```bash
# Basic protocol tests
cargo test --test grpc_protocol_basic

# Comprehensive protocol tests
cargo test --test grpc_protocol_comprehensive

# Full integration tests
cargo test --test grpc_protocol_integration
```

### Coverage Measurement

```bash
# Run coverage script
./scripts/run_grpc_coverage.sh

# Manual tarpaulin execution
cargo tarpaulin --test grpc_protocol_comprehensive --out Html --output-dir target/tarpaulin-grpc
```

### Concurrent Load Testing

```bash
# Run specific concurrent tests
cargo test --test grpc_protocol_comprehensive test_concurrent_protocol_correctness -- --nocapture
```

## Test Configuration

### Test Server Configuration

Tests use `TestConfigBuilder` for configurable daemon setups:

```rust
let config = TestConfigBuilder::new()
    .with_port(0)  // Ephemeral port
    .with_database_path(":memory:".to_string())
    .with_qdrant_url("http://localhost:6333".to_string())
    .with_connection_limits(100, 5)
    .build();
```

### Environment Requirements

- **Rust 2021 Edition**
- **tonic 0.12+** for gRPC implementation
- **tokio 1.0+** for async runtime
- **tonic-reflection** for protocol validation
- **cargo-tarpaulin** for coverage measurement

### Optional Dependencies

- **Qdrant server** (for full integration tests)
- **testcontainers** (for isolated service testing)

## Implementation Notes

### Protocol Validation

- Uses `tonic-reflection` for schema validation
- Includes protocol buffer descriptor validation
- Tests all enum values for serialization correctness

### Error Handling

- Current implementation provides placeholder responses
- Real implementation would validate inputs and return appropriate errors
- Tests designed to accommodate both current and future implementations

### Concurrency

- Tests use ephemeral ports for parallel execution
- Isolated daemon instances prevent test interference
- Connection pooling tested under concurrent load

### Coverage Analysis

Tests are designed to achieve 90%+ coverage of:

- gRPC service implementations (`src/grpc/services/`)
- Protocol message handling (`src/grpc/`)
- Server and client infrastructure
- Error handling and status mapping

## Contributing

When adding new gRPC protocol tests:

1. Use shared test utilities for consistency
2. Follow TDD approach with failing tests first
3. Include both success and error scenarios
4. Add concurrent access tests for new endpoints
5. Validate coverage with `cargo tarpaulin`
6. Update this documentation

## Troubleshooting

### Common Issues

1. **Port conflicts**: Tests use ephemeral ports to avoid conflicts
2. **Timeout errors**: Increase timeout values for slow systems
3. **Compilation time**: Large dependency tree requires patience
4. **Memory usage**: Concurrent tests may require increased memory limits

### Debug Mode

Run tests with debug output:

```bash
cargo test --test grpc_protocol_comprehensive -- --nocapture
RUST_LOG=debug cargo test --test grpc_protocol_basic
```