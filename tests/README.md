# Test Suite for workspace-qdrant-mcp

Comprehensive test suite covering unit tests, integration tests, and end-to-end tests for the workspace-qdrant-mcp project.

## Overview

This test suite provides **80%+ code coverage** and includes:

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests for component interactions and workflows
- **End-to-End Tests**: Complete workflow tests with realistic scenarios
- **Performance Tests**: Benchmarking and load testing

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and test configuration
├── unit/                    # Unit tests for individual modules
│   ├── test_config.py       # Configuration management tests
│   ├── test_client.py       # QdrantWorkspaceClient tests
│   ├── test_hybrid_search.py # Hybrid search and RRF tests
│   ├── test_project_detection.py # Git project detection tests
│   └── test_config_validator.py # Configuration validation tests
├── integration/             # Integration tests
│   └── test_server_integration.py # FastMCP server integration tests
├── e2e/                     # End-to-end tests
│   └── test_full_workflow.py # Complete workflow tests
└── fixtures/                # Test data and fixtures
```

## Running Tests

### Install Dependencies

```bash
# Install with development dependencies
pip install -e ".[dev]"
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m e2e          # End-to-end tests only
```

### Run Tests by Module

```bash
# Test specific modules
pytest tests/unit/test_config.py
pytest tests/unit/test_hybrid_search.py
pytest tests/integration/
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov-report=html

# View coverage report
open htmlcov/index.html

# Generate terminal coverage report
pytest --cov-report=term-missing
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test individual functions and classes in isolation

- `test_config.py` - Configuration loading, validation, environment variables
- `test_client.py` - QdrantWorkspaceClient initialization and operations
- `test_hybrid_search.py` - RRF fusion, hybrid search algorithms
- `test_project_detection.py` - Git repository analysis, submodule detection
- `test_config_validator.py` - Configuration validation logic

**Coverage Target**: 90%+ of core business logic

### Integration Tests (`tests/integration/`)

**Purpose**: Test component interactions and service integration

- `test_server_integration.py` - FastMCP server with mocked Qdrant
- MCP tool endpoint testing
- Configuration validation workflows
- Error handling across service boundaries

**Coverage Target**: 70%+ of integration paths

### End-to-End Tests (`tests/e2e/`)

**Purpose**: Test complete user workflows and scenarios

- `test_full_workflow.py` - Complete document lifecycle
- Project detection in real Git repositories
- Hybrid search with large result sets
- Multi-project workspace scenarios
- Performance benchmarking

**Coverage Target**: 60%+ of user-facing workflows

## Test Markers

Use pytest markers to categorize and selectively run tests:

```python
@pytest.mark.unit          # Unit test
@pytest.mark.integration   # Integration test
@pytest.mark.e2e          # End-to-end test
@pytest.mark.slow         # Slow-running test
@pytest.mark.requires_qdrant  # Requires Qdrant server
@pytest.mark.requires_git  # Requires Git repository
@pytest.mark.benchmark    # Performance benchmark
```

### Running Specific Categories

```bash
# Fast tests only (exclude slow tests)
pytest -m "not slow"

# Tests that don't require external services
pytest -m "not requires_qdrant and not requires_git"

# Performance benchmarks only
pytest -m benchmark
```

## Test Configuration

### Environment Variables

Test-specific environment variables are managed in `conftest.py`:

```python
# Automatically set for tests
WORKSPACE_QDRANT_HOST=127.0.0.1
WORKSPACE_QDRANT_PORT=8000
WORKSPACE_QDRANT_DEBUG=true
WORKSPACE_QDRANT_GITHUB_USER=testuser
```

### Mock Configuration

Extensive mocking infrastructure provides:

- Mock Qdrant client with realistic responses
- Mock embedding service with sample vectors
- Mock Git repositories with submodules
- Mock project detection scenarios
- Mock FastMCP application

## Fixtures and Test Data

### Key Fixtures (`conftest.py`)

- `mock_config` - Complete test configuration
- `mock_qdrant_client` - Qdrant client with realistic responses
- `mock_workspace_client` - Initialized workspace client
- `temp_git_repo` - Temporary Git repository for testing
- `temp_git_repo_with_submodules` - Git repo with mock submodules
- `sample_documents` - Document test data
- `sample_embeddings` - Vector embedding test data
- `environment_variables` - Test environment setup

### Creating Test Data

```python
# Use fixtures in your tests
def test_my_feature(mock_workspace_client, sample_documents):
    # Test implementation using provided fixtures
    pass

# Create custom test data
def test_custom_scenario():
    test_doc = create_test_point(
        point_id="doc1",
        content="Test content",
        metadata={"source": "test"}
    )
```

## Performance Testing

Benchmark tests measure performance characteristics:

```bash
# Run performance tests
pytest -m benchmark -v

# Run with benchmark plugin (if installed)
pytest --benchmark-only
```

Benchmarked areas:
- Hybrid search fusion algorithms
- Large document processing
- Project detection with many submodules
- Search result ranking

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest --cov --cov-fail-under=80
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall
- **Unit Tests**: 90% of core modules
- **Integration Tests**: 70% of service interactions
- **Critical Paths**: 95% coverage required

## Test Data Management

### Mock Data Strategy

- **Deterministic**: Same inputs always produce same outputs
- **Realistic**: Mock data represents real-world scenarios
- **Comprehensive**: Edge cases and error conditions covered
- **Maintainable**: Easy to update when requirements change

### External Dependencies

- **Qdrant**: Mocked for unit/integration tests, optional real instance for E2E
- **Git**: Temporary repositories created for testing
- **File System**: Temporary directories for safe testing
- **Network**: No external network calls in unit tests

## Writing New Tests

### Test Naming Convention

```python
# Unit tests
def test_function_name_expected_behavior():
    pass

def test_function_name_error_condition():
    pass

def test_function_name_edge_case():
    pass

# Parameterized tests
@pytest.mark.parametrize("input,expected", [
    ("input1", "expected1"),
    ("input2", "expected2"),
])
def test_function_name_parametrized(input, expected):
    pass
```

### Test Structure

```python
def test_my_feature():
    # Arrange - Set up test data and mocks
    config = create_test_config()
    client = MyClass(config)
    
    # Act - Execute the code under test
    result = client.my_method("test_input")
    
    # Assert - Verify the results
    assert result.status == "success"
    assert "expected" in result.message
```

### Async Test Patterns

```python
@pytest.mark.asyncio
async def test_async_feature():
    # Use AsyncMock for async dependencies
    mock_service = AsyncMock()
    mock_service.async_method.return_value = "expected_result"
    
    # Test async code
    result = await my_async_function(mock_service)
    
    assert result == "expected_result"
    mock_service.async_method.assert_called_once()
```

## Debugging Tests

### Running Single Test

```bash
# Run specific test with full output
pytest tests/unit/test_config.py::TestConfig::test_default_values -v -s

# Drop into debugger on failure
pytest --pdb

# Stop on first failure
pytest -x
```

### Logging in Tests

```python
import logging

def test_with_logging(caplog):
    with caplog.at_level(logging.INFO):
        # Code that logs
        my_function_that_logs()
    
    assert "Expected log message" in caplog.text
```

## Common Test Patterns

### Testing Exceptions

```python
def test_function_raises_exception():
    with pytest.raises(ValueError, match="Expected error message"):
        my_function_that_raises()
```

### Testing Async Functions

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await my_async_function()
    assert result is not None
```

### Testing with Temporary Files

```python
def test_with_temp_file():
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file.write(b"test data")
        temp_file.flush()
        
        result = process_file(temp_file.name)
        assert result.success
```

## Contributing

When adding new functionality:

1. **Write tests first** (TDD approach recommended)
2. **Achieve 80%+ coverage** for new code
3. **Add appropriate test markers** for categorization
4. **Update fixtures** if new test data patterns are needed
5. **Run full test suite** before submitting PR

### Pre-commit Checks

```bash
# Run before committing
pytest --cov --cov-fail-under=80  # Tests with coverage
black tests/ src/                 # Format code
ruff tests/ src/                  # Lint code
mypy src/                         # Type checking
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed with `pip install -e .`
2. **Async Test Failures**: Use `pytest-asyncio` and `@pytest.mark.asyncio`
3. **Mock Issues**: Verify mock patch paths are correct
4. **Git Test Failures**: Ensure Git is configured in test environment

### Test Environment

```bash
# Verify test environment
python -m pytest --version
python -c "import workspace_qdrant_mcp; print('Package imported successfully')"
```

### Debug Configuration

```python
# In conftest.py or test files
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use caplog fixture for test-specific logging
def test_with_debug_logs(caplog):
    with caplog.at_level(logging.DEBUG):
        # Test code here
        pass
```

For additional help, see the main project documentation or create an issue in the repository.
