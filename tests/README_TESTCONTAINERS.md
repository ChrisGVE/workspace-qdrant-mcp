# Testcontainers Integration for Isolated Qdrant Testing

This document describes the testcontainers infrastructure for workspace-qdrant-mcp that provides clean, isolated Qdrant instances for testing to prevent data contamination and ensure reliable test results.

## Overview

The testcontainers integration provides:

- **Isolated Qdrant Containers**: Clean instances for each test preventing data contamination
- **Container Lifecycle Management**: Automatic start/stop/cleanup procedures
- **Multiple Scopes**: Function, class, module, and session-scoped containers
- **Health Checks**: Validation that containers are ready before tests run
- **Performance Optimization**: Shared containers with reset capability for faster tests
- **Integration**: Works with existing FastMCP, pytest-mcp, and AI evaluation frameworks

## Architecture

### Core Components

1. **IsolatedQdrantContainer**: Enhanced container wrapper with lifecycle management
2. **QdrantContainerManager**: Manager for container lifecycle across test scopes
3. **Pytest Fixtures**: Integration with pytest for different testing scenarios
4. **Configuration**: Automatic test configuration generation

### Container Types

#### Isolated Containers (Recommended for Unit Tests)
- Fresh container for each test
- Complete isolation between tests
- Slower but most reliable
- Use fixture: `isolated_qdrant_container`

#### Shared Containers (Recommended for Integration Tests)
- Single container per test session
- Reset state between tests
- Faster but requires proper cleanup
- Use fixture: `shared_qdrant_container`

#### Session Containers (For Test Suites)
- Single container for entire test session
- Manual state management required
- Fastest for large test suites
- Use fixture: `session_qdrant_container`

## Usage

### Basic Test Setup

```python
import pytest
from qdrant_client.http import models

@pytest.mark.requires_docker
@pytest.mark.isolated_container
def test_my_functionality(isolated_qdrant_client):
    """Test with isolated Qdrant container."""
    # Create test collection
    isolated_qdrant_client.create_collection(
        collection_name="test_collection",
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )

    # Test your functionality
    collections = isolated_qdrant_client.get_collections()
    assert "test_collection" in [c.name for c in collections.collections]
```

### Workspace Client Testing

```python
@pytest.mark.requires_docker
async def test_workspace_functionality(test_workspace_client):
    """Test with workspace client connected to isolated container."""
    # Client is already initialized and connected
    status = await test_workspace_client.get_status()
    assert status["connected"] is True

    # Test workspace operations
    await test_workspace_client.store_document(
        content="Test document",
        metadata={"source": "test"}
    )
```

### Async Context Manager

```python
from tests.utils.testcontainers_qdrant import isolated_qdrant_instance

@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_with_context_manager():
    """Test using async context manager."""
    async with isolated_qdrant_instance() as (container, client):
        # Container is automatically started and will be cleaned up
        collections = client.get_collections()
        assert isinstance(collections, models.CollectionsResponse)
```

### Performance Tests with Shared Containers

```python
@pytest.mark.requires_docker
@pytest.mark.shared_container
@pytest.mark.performance
class TestPerformance:
    """Performance tests using shared container for speed."""

    def test_ingestion_speed(self, shared_qdrant_client):
        """Test document ingestion performance."""
        # Container is shared across all tests in this class
        # State is reset between each test
        start_time = time.time()

        # Your performance test
        for i in range(100):
            shared_qdrant_client.upsert(
                collection_name="perf_test",
                points=[{
                    "id": i,
                    "vector": [0.1] * 384,
                    "payload": {"doc_id": i}
                }]
            )

        duration = time.time() - start_time
        assert duration < 10.0  # Should complete in under 10 seconds
```

## Available Fixtures

### Container Fixtures

- `qdrant_container_manager`: Session-scoped container manager
- `session_qdrant_container`: Session-scoped Qdrant container
- `isolated_qdrant_container`: Function-scoped isolated container
- `shared_qdrant_container`: Shared container with state reset
- `containerized_qdrant_instance`: Async context manager

### Client Fixtures

- `isolated_qdrant_client`: Client connected to isolated container
- `shared_qdrant_client`: Client connected to shared container
- `test_workspace_client`: Workspace client with isolated container
- `test_config`: Configuration object for isolated container

## Test Markers

### Required Markers

- `@pytest.mark.requires_docker`: For tests needing Docker daemon
- `@pytest.mark.requires_qdrant_container`: For tests needing Qdrant containers

### Strategy Markers

- `@pytest.mark.isolated_container`: Tests using isolated containers
- `@pytest.mark.shared_container`: Tests using shared containers
- `@pytest.mark.performance`: Performance/benchmark tests

### Example Test Configuration

```python
# Isolated unit test
@pytest.mark.requires_docker
@pytest.mark.isolated_container
def test_isolated_functionality(isolated_qdrant_client):
    pass

# Shared integration test
@pytest.mark.requires_docker
@pytest.mark.shared_container
@pytest.mark.integration
class TestIntegrationSuite:
    pass

# Performance test
@pytest.mark.requires_docker
@pytest.mark.shared_container
@pytest.mark.performance
def test_performance_metrics(shared_qdrant_client):
    pass
```

## Configuration

### Container Settings

Default container configuration:
- **Image**: `qdrant/qdrant:v1.7.4`
- **HTTP Port**: 6333 (mapped to random host port)
- **gRPC Port**: 6334 (mapped to random host port)
- **Startup Timeout**: 60 seconds
- **Health Check Interval**: 1.0 seconds

### Customization

```python
from tests.utils.testcontainers_qdrant import IsolatedQdrantContainer

# Custom container configuration
container = IsolatedQdrantContainer(
    image="qdrant/qdrant:latest",
    startup_timeout=30,
    health_check_interval=0.5
)
```

## Integration with Existing Frameworks

### FastMCP Integration

The testcontainers work seamlessly with FastMCP test infrastructure:

```python
@pytest.mark.requires_docker
async def test_mcp_with_containers(fastmcp_test_client, test_workspace_client):
    """Test MCP tools with containerized Qdrant."""
    # Use both FastMCP client and containerized workspace client
    result = await fastmcp_test_client.call_tool(
        "search_workspace",
        {"query": "test", "limit": 10}
    )
    assert result is not None
```

### AI-Powered Evaluation

```python
@pytest.mark.requires_docker
@pytest.mark.ai_evaluation
async def test_ai_evaluation_with_containers(ai_test_evaluator, test_workspace_client):
    """Test AI evaluation with containerized backend."""
    result = await ai_test_evaluator.evaluate_search_quality(
        workspace_client=test_workspace_client,
        test_queries=["python function", "API documentation"]
    )
    assert result.overall_score > 0.8
```

## Performance Considerations

### Container Startup Time

- **Isolated containers**: ~10-15 seconds per test (complete isolation)
- **Shared containers**: ~10-15 seconds per test class (with resets)
- **Session containers**: ~10-15 seconds per test session

### Optimization Strategies

1. **Use shared containers for integration test suites**
2. **Group related tests into test classes to share containers**
3. **Use isolated containers only when complete isolation is required**
4. **Consider using mocked clients for pure unit tests**

### Memory Usage

Each container uses approximately:
- **RAM**: ~100-200MB per container
- **Disk**: ~50MB per container (temporary)
- **Network**: Random ports allocated by Docker

## Troubleshooting

### Common Issues

#### Docker Not Available
```
pytest.skip("Docker not available for integration tests")
```
- Ensure Docker daemon is running
- Check Docker permissions for test user

#### Container Startup Timeout
```
TimeoutError: Qdrant container failed to become healthy within 60 seconds
```
- Check Docker resource limits
- Increase `startup_timeout` parameter
- Check system resources (CPU/memory)

#### Port Conflicts
Testcontainers automatically handles port allocation, but if issues occur:
- Restart Docker daemon
- Check for port exhaustion
- Use explicit port ranges in Docker configuration

### Debugging

Enable debug logging:
```python
import logging
logging.getLogger("testcontainers").setLevel(logging.DEBUG)
logging.getLogger("tests.utils.testcontainers_qdrant").setLevel(logging.DEBUG)
```

Check container logs:
```python
def test_with_logging(isolated_qdrant_container):
    container = isolated_qdrant_container.container
    logs = container.get_logs()
    print(f"Container logs: {logs}")
```

## Running Tests

### Run All Container Tests
```bash
# Run all tests requiring containers
pytest -m "requires_docker"

# Run only isolated container tests
pytest -m "isolated_container"

# Run only shared container tests
pytest -m "shared_container"
```

### Skip Container Tests
```bash
# Skip all Docker-dependent tests
pytest -m "not requires_docker"
```

### Performance Test Suite
```bash
# Run performance tests with shared containers
pytest -m "performance and shared_container" --benchmark-only
```

## Best Practices

1. **Always use appropriate markers** (`@pytest.mark.requires_docker`)
2. **Choose the right container type** for your test scenario
3. **Clean up test data** properly (containers handle this automatically)
4. **Use shared containers** for integration test suites when possible
5. **Mock external dependencies** that don't need real containers
6. **Group related tests** to maximize container reuse
7. **Monitor resource usage** in CI/CD environments
8. **Test container health** in your test setup

## Example Test Patterns

See `tests/test_testcontainers_integration.py` for comprehensive examples of:
- Container lifecycle testing
- Data isolation validation
- Fixture usage patterns
- Performance testing
- Error handling
- Health check validation

This infrastructure ensures reliable, isolated testing while integrating seamlessly with the existing test frameworks and maintaining good performance characteristics.