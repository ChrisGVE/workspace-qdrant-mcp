# External Dependency Mocking Strategy

This document provides comprehensive guidance for using the external dependency mocking infrastructure in workspace-qdrant-mcp tests.

## Overview

The mocking infrastructure enables reliable testing without external service dependencies while maintaining realistic behavior simulation. It supports:

- **Error injection** with configurable failure scenarios
- **Realistic behavior** simulation with appropriate delays and responses
- **Comprehensive coverage** of all external dependencies
- **Easy configuration** for different testing scenarios

## Architecture

### Core Components

1. **Error Injection Framework** (`error_injection.py`)
   - Configurable failure scenarios
   - Realistic error rates and patterns
   - Cross-component error coordination

2. **Component-Specific Mocks**
   - `qdrant_mocks.py` - Enhanced Qdrant client with search/CRUD operations
   - `filesystem_mocks.py` - File system operations and watching
   - `grpc_mocks.py` - gRPC communication between components
   - `network_mocks.py` - HTTP requests and network operations
   - `lsp_mocks.py` - LSP server communication and metadata extraction
   - `embedding_mocks.py` - Embedding generation and vector operations
   - `external_service_mocks.py` - Third-party API integrations

3. **Fixture Library** (`fixtures.py`)
   - Pre-configured mock suites
   - Test-type specific configurations
   - Validation helpers

## Usage Guide

### Basic Usage

```python
import pytest
from tests.mocks import create_realistic_qdrant_mock, create_basic_embedding_service

def test_document_search():
    # Create realistic mock with occasional errors
    qdrant_mock = create_realistic_qdrant_mock()

    # Use in your test
    results = await qdrant_mock.search("test-collection", [0.1] * 384)
    assert len(results) > 0
```

### Using Fixtures

```python
def test_with_fixtures(mock_dependency_suite):
    """Test using complete mock suite."""
    qdrant = mock_dependency_suite["qdrant"]
    filesystem = mock_dependency_suite["filesystem"]

    # Test cross-component functionality
    await qdrant.upsert("collection", points)
    filesystem.write_text("/path/file.txt", "content")
```

### Error Testing

```python
def test_error_handling(mock_qdrant_client_failing):
    """Test error handling with high failure rate."""
    # This mock has 50% failure rate
    with pytest.raises(Exception):
        for _ in range(10):  # Eventually will hit an error
            await mock_qdrant_client_failing.search("collection", [0.1] * 384)
```

### Scenario-Based Testing

```python
def test_network_partition(mock_error_scenarios, mock_dependency_suite):
    """Test behavior during network partition."""
    # Apply network partition scenario
    mock_error_scenarios["apply_scenario"]("connection_issues", ["qdrant", "grpc"])

    # Test system behavior under network stress
    with pytest.raises(ConnectionError):
        await mock_dependency_suite["qdrant"].search("collection", [0.1] * 384)
```

## Mock Configuration

### Error Injection Levels

| Level | Error Rate | Use Case |
|-------|------------|----------|
| `unit_testing` | 1% | Stable unit tests |
| `integration_testing` | 5% | Integration tests |
| `stress_testing` | 20% | Stress and chaos testing |
| `production_simulation` | 0.1% | Production-like behavior |

### Performance Profiles

| Profile | Response Times | Use Case |
|---------|----------------|----------|
| `fast` | 1-50ms | Unit tests |
| `realistic` | 50-500ms | Integration tests |
| `slow` | 500ms-3s | Timeout testing |

## Component-Specific Guides

### Qdrant Mocking

```python
from tests.mocks import create_enhanced_qdrant_client

# Basic usage
client = create_enhanced_qdrant_client()

# With error injection
client = create_enhanced_qdrant_client(
    with_error_injection=True,
    error_probability=0.1,
    collections=["test-collection"]
)

# Configure specific error types
client.error_injector.configure_connection_issues(0.05)
client.error_injector.configure_service_issues(0.03)
```

### Filesystem Mocking

```python
from tests.mocks import create_filesystem_mock

# Create filesystem mock
fs = create_filesystem_mock(with_error_injection=True)

# Add files to virtual filesystem
fs.add_file("/test/file.txt", "content")
fs.add_directory("/test/dir")

# Use filesystem operations
content = fs.read_text("/test/file.txt")
fs.write_text("/test/new.txt", "new content")
```

### gRPC Mocking

```python
from tests.mocks import create_realistic_daemon_communication

# Create daemon communication mock
daemon = create_realistic_daemon_communication()

# Initialize and use
await daemon.initialize_daemon({"config": "value"})
result = await daemon.client.ingest_document(
    collection_name="test",
    document_path="/path/doc.txt",
    content="document content"
)
```

### Network Mocking

```python
from tests.mocks import create_realistic_network_client

# Create network client
client = create_realistic_network_client()

# Make HTTP requests
response = await client.get("https://api.example.com/data")
assert response.status_code == 200

# POST with data
response = await client.post(
    "https://api.example.com/items",
    json={"name": "test item"}
)
```

### LSP Mocking

```python
from tests.mocks import create_basic_lsp_server

# Create LSP server for Python
lsp = create_basic_lsp_server("python")

# Initialize and use
await lsp.initialize(["/workspace"], {})
await lsp.open_document("/file.py", "def hello(): pass", "python")
symbols = await lsp.get_symbols("/file.py")
```

### Embedding Mocking

```python
from tests.mocks import create_realistic_embedding_service

# Create embedding service
embedder = create_realistic_embedding_service()

# Initialize and generate embeddings
await embedder.initialize()
result = await embedder.generate_embeddings("test text")
assert "dense" in result
assert "sparse" in result
```

## Advanced Usage

### Custom Error Scenarios

```python
from tests.mocks import ErrorInjector, FailureScenarios

# Create custom error injector
injector = ErrorInjector()

# Configure specific failure modes
injector.configure_failure_mode(
    "custom_error",
    probability=0.1,
    error_message="Custom error condition"
)

# Apply to mock
mock = create_enhanced_qdrant_client(error_injector=injector)
```

### Cross-Component Error Coordination

```python
from tests.mocks import ErrorModeManager

# Create error manager
manager = ErrorModeManager()

# Register component injectors
manager.register_injector("qdrant", qdrant_mock.error_injector)
manager.register_injector("grpc", grpc_mock.error_injector)

# Apply coordinated scenario
manager.apply_scenario("stress_testing", ["qdrant", "grpc"])
```

### Performance Testing

```python
import time
from tests.mocks import mock_performance_profiles

def test_response_times(mock_dependency_suite, mock_performance_profiles):
    """Test that operations complete within expected timeframes."""
    profile = mock_performance_profiles["realistic"]

    start_time = time.time()
    await mock_dependency_suite["qdrant"].search("collection", [0.1] * 384)
    duration = time.time() - start_time

    expected_max = profile["delays"]["qdrant_search"] * 2  # Allow some margin
    assert duration <= expected_max
```

## Best Practices

### 1. Use Appropriate Mock Levels

- **Unit tests**: Use `mock_for_unit_tests` fixture with minimal error injection
- **Integration tests**: Use `mock_for_integration_tests` with moderate errors
- **Stress tests**: Use `mock_for_stress_tests` with high error rates

### 2. Reset State Between Tests

```python
def test_something(mock_dependency_suite):
    # Use mocks
    # ...

    # Reset state for clean slate
    for mock_obj in mock_dependency_suite.values():
        if hasattr(mock_obj, 'reset_state'):
            mock_obj.reset_state()
```

### 3. Validate Mock Behavior

```python
def test_with_validation(mock_dependency_suite, mock_validation_helpers):
    # Perform operations
    await mock_dependency_suite["qdrant"].search("test", [0.1] * 384)

    # Validate expected operations occurred
    mock_validation_helpers.assert_operation_history(
        mock_dependency_suite["qdrant"],
        ["search"]
    )
```

### 4. Use Realistic Data

```python
def test_with_realistic_data(mock_dependency_suite, mock_data_generators):
    # Generate realistic test documents
    documents = mock_data_generators.generate_documents(100)

    # Use in tests
    for doc in documents:
        await mock_dependency_suite["qdrant"].upsert("collection", [doc])
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Correct import
   from tests.mocks import create_realistic_qdrant_mock

   # Or use fixtures
   def test_something(mock_qdrant_client_enhanced):
       pass
   ```

2. **Error Injection Not Working**
   ```python
   # Ensure error injection is enabled
   mock = create_enhanced_qdrant_client(with_error_injection=True)

   # Check error injector configuration
   stats = mock.error_injector.get_statistics()
   assert stats["enabled"] == True
   ```

3. **Unrealistic Behavior**
   ```python
   # Use realistic mocks for integration tests
   mock = create_realistic_qdrant_mock()  # 5% error rate

   # Not this for integration tests
   mock = create_failing_qdrant_mock()    # 50% error rate
   ```

### Debugging Mock Behavior

```python
def debug_mock_operations(mock_obj):
    """Debug helper to inspect mock operations."""
    if hasattr(mock_obj, 'get_operation_history'):
        history = mock_obj.get_operation_history()
        print(f"Operations performed: {len(history)}")
        for op in history[-5:]:  # Last 5 operations
            print(f"  {op}")

    if hasattr(mock_obj, 'error_injector'):
        stats = mock_obj.error_injector.get_statistics()
        print(f"Error stats: {stats}")
```

## Testing the Mocks Themselves

The mocking infrastructure includes self-tests to ensure reliability:

```bash
# Run mock-specific tests
pytest tests/unit/test_mock_infrastructure.py

# Test error injection
pytest tests/unit/test_error_injection.py

# Test specific mock components
pytest tests/unit/test_qdrant_mocks.py
pytest tests/unit/test_grpc_mocks.py
```

## Contributing

When adding new mocks or enhancing existing ones:

1. Follow the established patterns in existing mock files
2. Include comprehensive error injection capabilities
3. Provide realistic behavior simulation
4. Add convenience functions for common scenarios
5. Update this documentation
6. Add tests for the mock components themselves

## Performance Considerations

- Mock operations are designed to be fast while maintaining realism
- Use `fast` performance profile for unit tests
- Use `realistic` profile for integration tests
- Avoid deep object hierarchies in mock responses for performance

## Security Considerations

- Mocks should never connect to real external services
- API keys and credentials used in mocks should be dummy values
- Ensure mocks don't inadvertently expose real service endpoints