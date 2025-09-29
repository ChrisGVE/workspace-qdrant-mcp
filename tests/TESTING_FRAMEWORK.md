# Test Framework Documentation

Comprehensive guide to the workspace-qdrant-mcp test framework structure, organization, and usage.

## Framework Overview

The test framework provides a structured approach to testing workspace-qdrant-mcp with:

- **4 Test Domains**: daemon, MCP server, CLI, integration
- **3 Test Categories**: nominal (happy path), edge (boundaries), stress (performance)
- **Comprehensive Fixtures**: Shared and domain-specific test utilities
- **Automatic Categorization**: Markers auto-applied based on test location
- **Isolated Testing**: Testcontainers for clean Qdrant instances
- **Performance Tracking**: Built-in benchmarking and monitoring

## Directory Structure

```
tests/
├── daemon/                    # Rust daemon tests
│   ├── conftest.py           # Daemon fixtures
│   └── README.md
│
├── mcp_server/               # MCP server tests
│   ├── nominal/              # Happy path
│   ├── edge/                 # Edge cases
│   ├── stress/               # Performance
│   ├── conftest.py
│   └── README.md
│
├── cli/                      # CLI tests
│   ├── nominal/
│   ├── edge/
│   ├── stress/
│   ├── parsers/
│   ├── conftest.py
│   └── README.md
│
├── shared/                   # Shared utilities
│   ├── testcontainers_utils.py
│   ├── assertions.py
│   ├── test_data.py
│   └── fixtures.py
│
└── conftest.py              # Root configuration
```

## Test Categories

### Nominal Tests
**Location**: `*/nominal/` directories
**Marker**: `@pytest.mark.nominal`
**Purpose**: Validate expected behavior with valid inputs

Example:
```python
@pytest.mark.nominal
@pytest.mark.mcp_server
async def test_store_document_success():
    """Test storing valid document succeeds."""
    result = await store_document("content", {"source": "test"})
    assert result["success"]
```

### Edge Tests
**Location**: `*/edge/` directories
**Marker**: `@pytest.mark.edge`
**Purpose**: Test boundaries and error handling

Example:
```python
@pytest.mark.edge
@pytest.mark.mcp_server
async def test_store_empty_document():
    """Test empty content raises error."""
    with pytest.raises(ValueError):
        await store_document("", {})
```

### Stress Tests
**Location**: `*/stress/` directories
**Markers**: `@pytest.mark.stress`, `@pytest.mark.slow`
**Purpose**: Performance and scalability validation

Example:
```python
@pytest.mark.stress
@pytest.mark.slow
async def test_concurrent_storage():
    """Test 1000 concurrent document stores."""
    tasks = [store_document(f"Doc {i}") for i in range(1000)]
    results = await asyncio.gather(*tasks)
    assert len(results) == 1000
```

## Test Domains

### Daemon Tests (`tests/daemon/`)
Tests for Rust daemon component:
- File watching and change detection
- Document processing pipeline
- gRPC communication
- Async task queue management

**Current Status**: Placeholder with mocked daemon (Rust implementation pending)

### MCP Server Tests (`tests/mcp_server/`)
Tests for FastMCP server:
- Tool registration and execution
- Document storage and retrieval
- Hybrid search functionality
- Protocol compliance
- Client interactions

### CLI Tests (`tests/cli/`)
Tests for `wqm` command-line interface:
- Command execution and output
- Document ingestion workflows
- Search operations
- Collection management
- Configuration handling

### Integration Tests (`tests/integration/`)
Cross-component integration tests:
- End-to-end workflows
- Component interactions
- System-level behavior

## Markers and Selection

### Running Tests by Domain

```bash
# Daemon tests
pytest -m daemon

# MCP server tests
pytest -m mcp_server

# CLI tests
pytest -m cli

# Integration tests
pytest -m integration
```

### Running Tests by Category

```bash
# Nominal tests
pytest -m nominal

# Edge tests
pytest -m edge

# Stress tests
pytest -m stress

# Exclude slow tests
pytest -m "not slow"
```

### Combining Markers

```bash
# MCP server nominal tests
pytest -m "mcp_server and nominal"

# CLI without slow tests
pytest -m "cli and not slow"

# All edge and stress tests
pytest -m "edge or stress"
```

## Fixtures Guide

### Testcontainer Fixtures

```python
# Isolated container (fresh per test)
async def test_with_isolation(isolated_qdrant_container):
    url = isolated_qdrant_container.get_http_url()
    # Container auto-cleaned after test

# Shared container (session-scoped, reset between tests)
async def test_with_sharing(shared_qdrant_container):
    url = shared_qdrant_container.get_http_url()
    # Container shared, state reset
```

### Test Data Fixtures

```python
def test_with_data(test_data_generator, sample_documents):
    # Generate custom data
    doc = test_data_generator.generate_document(size="large")

    # Use pre-generated samples
    docs = sample_documents  # 3 sample documents
```

### Domain Fixtures

```python
# MCP Server
async def test_mcp(mock_qdrant_client, mock_fastmcp_server):
    # Mocked Qdrant and FastMCP

# CLI
def test_cli(cli_runner, cli_config_file):
    # CLI runner and config

# Daemon
async def test_daemon(mock_daemon_client, daemon_config):
    # Mocked daemon client
```

## Custom Assertions

Use domain-specific assertions from `tests/shared/assertions.py`:

```python
from tests.shared.assertions import (
    assert_search_results_valid,
    assert_collection_valid,
    assert_mcp_response_valid,
    assert_hybrid_search_results_valid,
)

# Validate search results
results = await search("query")
assert_search_results_valid(results, min_score=0.5)

# Validate MCP response
response = await mcp_call("tool", {})
assert_mcp_response_valid(response, expected_id=1)

# Validate hybrid search
results = await hybrid_search("query")
assert_hybrid_search_results_valid(results, query="query", min_results=1)
```

## Performance Testing

### Using Performance Fixtures

```python
async def test_with_monitoring(performance_metrics, performance_thresholds):
    start = time.time()

    # Perform operation
    await store_document("content")

    duration = (time.time() - start) * 1000  # ms

    # Check against threshold
    threshold = performance_thresholds["document_storage"]
    assert duration < threshold, f"Took {duration}ms, limit {threshold}ms"
```

### Benchmarking

```python
@pytest.mark.performance
@pytest.mark.benchmark
def test_search_performance(benchmark, mock_qdrant_client):
    """Benchmark search operation."""
    result = benchmark(search_documents, query="test")
    assert result is not None
```

## Coverage Reports

### Generate Coverage

```bash
# HTML report
pytest --cov=src/python --cov-report=html
open htmlcov/index.html

# Terminal report with missing lines
pytest --cov=src/python --cov-report=term-missing

# XML for CI
pytest --cov=src/python --cov-report=xml

# Fail if below threshold
pytest --cov=src/python --cov-fail-under=80
```

### Coverage Configuration

In `pyproject.toml`:
```toml
[tool.coverage.run]
source = ["src/python"]
branch = true
omit = ["*/tests/*", "*/__init__.py"]

[tool.coverage.report]
show_missing = true
fail_under = 10  # Minimum 10% coverage
```

## Automatic Marker Application

Markers are automatically applied based on:

1. **Directory**: `tests/mcp_server/` → `@pytest.mark.mcp_server`
2. **Subdirectory**: `tests/mcp_server/nominal/` → `@pytest.mark.nominal`
3. **Test Name**: `test_stress_*` → `@pytest.mark.stress`, `@pytest.mark.slow`

This happens via hooks in `tests/conftest.py`:
```python
def pytest_collection_modifyitems(config, items):
    for item in items:
        # Auto-apply markers based on location
        ...
```

## Writing New Tests

### 1. Choose Location

Place test in appropriate directory:
- Domain: `daemon/`, `mcp_server/`, `cli/`, or `integration/`
- Category: `nominal/`, `edge/`, or `stress/` subdirectory

### 2. Use Fixtures

Leverage shared and domain fixtures:
```python
async def test_feature(
    isolated_qdrant_container,
    test_data_generator,
    mock_qdrant_client,
):
    # Test implementation
    pass
```

### 3. Apply Markers (if needed)

Explicit markers when auto-detection isn't sufficient:
```python
@pytest.mark.mcp_server
@pytest.mark.nominal
@pytest.mark.requires_qdrant
async def test_something():
    pass
```

### 4. Follow Structure

```python
async def test_description():
    """
    Test [what].

    Given: [preconditions]
    When: [action]
    Then: [expected outcome]
    """
    # Arrange
    setup_code()

    # Act
    result = await action()

    # Assert
    assert result == expected
```

## CI Integration

Example GitHub Actions:
```yaml
- name: Run tests
  run: |
    uv run pytest \
      --cov=src/python \
      --cov-report=xml \
      --junitxml=test-results/junit.xml \
      -m "not slow"

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: coverage.xml
```

## Troubleshooting

### Import Errors
```bash
# Install dev dependencies
uv sync --dev
```

### Docker Not Available
```bash
# Skip Docker tests
pytest -m "not requires_docker"
```

### Tests Too Slow
```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

### Qdrant Unavailable
```bash
# Use mocked Qdrant
pytest -m "not requires_qdrant"
```

## Best Practices

1. **One test, one concept**: Each test should verify one specific behavior
2. **Descriptive names**: `test_store_document_with_empty_content_raises_error`
3. **Use fixtures**: Don't duplicate setup code
4. **Clean tests**: Leave no test data or side effects
5. **Fast by default**: Save slow tests for stress category
6. **Document intent**: Include docstrings explaining test purpose
7. **Test errors too**: Don't just test happy paths
8. **Use assertions**: Leverage custom assertions for clarity