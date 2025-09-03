# Integration Testing Guide

This guide provides comprehensive information about the integration testing suite for the workspace-qdrant-mcp project, covering test execution, performance monitoring, and CI/CD integration.

## Overview

The integration testing suite provides comprehensive validation of the entire system, including:

- **Document ingestion pipeline** from file watching to vector storage
- **Python-Rust gRPC communication** with various payload sizes
- **Daemon lifecycle management** and error recovery
- **Performance regression testing** with baseline establishment
- **Error recovery scenarios** under various failure conditions
- **Code coverage measurement** targeting 80% coverage

## Quick Start

### Local Testing

```bash
# Install dependencies with testcontainers
pip install -e ".[dev]"

# Run basic integration tests
python scripts/run_integration_tests.py --categories integration

# Run with coverage
python scripts/run_integration_tests.py --categories integration --coverage-threshold 80

# Run performance tests
python scripts/run_integration_tests.py --categories performance --no-coverage
```

### Docker Environment

```bash
# Start test environment
cd docker/integration-tests
docker-compose up -d qdrant

# Run integration tests in container
docker-compose --profile test-runner run --rm test-runner

# Run specific test categories
docker-compose --profile test-runner run --rm test-runner \
  python scripts/run_integration_tests.py --categories smoke --verbose
```

## Test Categories

### Integration Tests (`integration`)
- Complete document ingestion pipeline validation
- End-to-end workflow testing from file watching to vector storage
- Search functionality verification with ingested documents
- Configuration synchronization testing

**Files:**
- `test_document_ingestion_pipeline.py`
- `test_daemon_lifecycle_integration.py`

**Duration:** ~20-30 minutes
**Coverage Target:** 80%+

### Performance Tests (`performance`)
- Ingestion throughput measurement
- Search latency benchmarking
- Concurrent operation testing
- Memory usage validation

**Files:**
- `test_performance_regression.py`

**Duration:** ~15-25 minutes
**Baseline Establishment:** Required on first run

### gRPC Communication Tests (`integration`)
- Small, medium, large payload handling
- Streaming vs unary RPC patterns
- Connection pooling and timeout behavior
- Error handling across Python-Rust boundary

**Files:**
- `test_grpc_payload_communication.py`

**Duration:** ~10-15 minutes

### Error Recovery Tests (`regression`)
- Network connectivity failure scenarios
- Service unavailability recovery
- File system error handling
- Memory pressure management
- Configuration error recovery

**Files:**
- `test_error_recovery_scenarios.py`

**Duration:** ~15-20 minutes

### Smoke Tests (`smoke`)
- Basic functionality validation
- Quick health checks
- Essential feature verification

**Duration:** ~5-10 minutes

## Test Infrastructure

### Testcontainers Integration

All integration tests use testcontainers to provide isolated Qdrant instances:

```python
@pytest.fixture(scope="module")
def isolated_qdrant():
    """Start isolated Qdrant instance for testing."""
    compose_file = """
    version: '3.8'
    services:
      qdrant:
        image: qdrant/qdrant:v1.7.4
        ports:
          - "6333:6333"
          - "6334:6334"
    """
    # Container management code...
```

### Performance Baselines

Performance tests establish and compare against baselines:

```json
{
  "ingestion_throughput": {
    "throughput_docs_per_sec": 2.5,
    "avg_processing_time_ms": 400
  },
  "search_latency": {
    "avg_latency_ms": 150,
    "p95_latency_ms": 300
  }
}
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
parallel = true

[tool.coverage.report]
fail_under = 80
exclude_lines = [
  "pragma: no cover",
  "@abstract",
  "except ImportError:",
]
```

## Running Tests

### Command Line Options

```bash
python scripts/run_integration_tests.py [OPTIONS]

Options:
  --categories CATEGORIES     Test categories to run (default: integration)
  --no-coverage              Disable coverage measurement
  --no-performance           Disable performance tests
  --parallel                 Run tests in parallel
  --verbose                  Verbose output
  --coverage-threshold INT   Coverage threshold percentage (default: 80)
```

### pytest Direct Execution

```bash
# Run all integration tests
pytest tests/integration/ -m integration

# Run specific test file
pytest tests/integration/test_document_ingestion_pipeline.py -v

# Run with coverage
pytest tests/integration/ --cov=src/workspace_qdrant_mcp --cov-report=html

# Run performance tests only
pytest tests/integration/ -m performance --benchmark-only
```

### Docker Compose Profiles

```bash
# Basic Qdrant service
docker-compose up -d qdrant

# Multi-instance testing
docker-compose --profile multi-instance up -d

# Test runner service
docker-compose --profile test-runner run --rm test-runner

# Performance monitoring
docker-compose --profile monitoring up -d performance-monitor
```

## CI/CD Integration

### GitHub Actions Workflows

**Integration Tests** (`.github/workflows/integration-tests.yml`):
- Runs on push to main/develop and PRs
- Matrix strategy for different test suites
- Coverage reporting to Codecov
- PR comments with test results

**Performance Monitoring** (`.github/workflows/performance-monitoring.yml`):
- Daily scheduled runs
- Baseline comparison and regression detection
- Performance issue creation for regressions
- Historical trend tracking

### Workflow Triggers

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:     # Manual trigger
```

### Environment Setup

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - 6333:6333
      - 6334:6334
    options: >-
      --health-cmd "curl -f http://localhost:6333/health"
      --health-interval 10s
```

## Test Data Management

### Test Data Factory

The integration test suite includes a comprehensive test data factory:

```python
class TestDataFactory:
    @staticmethod
    def create_text_document(size: str = "small", topic: str = "general") -> str:
        """Create text document of specified size."""
        multipliers = {
            "tiny": 5,      # ~100 characters
            "small": 50,    # ~1KB  
            "medium": 500,  # ~10KB
            "large": 5000,  # ~100KB
            "huge": 50000   # ~1MB
        }
        # Document generation logic...
```

### Temporary Workspaces

Tests create realistic project structures:

```python
@pytest.fixture
def temp_workspace():
    """Create temporary workspace with realistic project structure."""
    # Creates: src/, docs/, tests/, config/, data/ directories
    # Adds common files: README.md, pyproject.toml, .gitignore, etc.
```

## Performance Testing

### Benchmark Categories

1. **Ingestion Throughput**
   - Documents per second processing
   - Average processing time per document
   - Memory usage during ingestion

2. **Search Latency**
   - Simple vs complex query performance
   - Result set size impact
   - P95 latency measurement

3. **Concurrent Operations**
   - Mixed workload performance
   - Resource contention handling
   - Error isolation

### Performance Thresholds

```python
PERFORMANCE_THRESHOLDS = {
    "ingestion": {
        "small_doc_max_time_ms": 1000,
        "min_throughput_docs_per_sec": 0.5
    },
    "search": {
        "simple_query_max_time_ms": 1000,
        "p95_latency_ms": 3000
    },
    "system": {
        "memory_max_mb": 500,
        "startup_max_time_ms": 15000
    }
}
```

### Regression Detection

Performance regressions are detected by comparing against baselines:

- **Threshold:** 20% performance degradation
- **Action:** Create GitHub issue for investigation
- **Trend Analysis:** Historical performance tracking

## Error Recovery Testing

### Failure Scenarios

1. **Network Connectivity**
   - Connection drops during operations
   - DNS resolution failures
   - Network timeout scenarios

2. **Service Unavailability**
   - Qdrant service restart
   - gRPC server unavailability
   - Database connection issues

3. **File System Errors**
   - Permission denied scenarios
   - Corrupted file handling
   - Disk space exhaustion

4. **Memory Pressure**
   - Large document processing
   - Memory exhaustion handling
   - Garbage collection triggers

5. **Configuration Errors**
   - Invalid parameter values
   - Missing configuration files
   - Environment variable issues

### Recovery Validation

```python
async def test_network_connectivity_failure_recovery():
    """Test recovery from network connectivity failures."""
    # Simulate network failure
    with patch.object(client, '_qdrant_client') as mock_client:
        mock_client.upsert.side_effect = [
            ConnectionError("Network unreachable"),
            ConnectionError("Connection timeout"),
            MagicMock(operation_id=12345, status="completed")  # Recovery
        ]
        # Test retry logic and recovery...
```

## Troubleshooting

### Common Issues

**Docker Not Available:**
```bash
# Check Docker installation
docker --version

# Verify Docker daemon
docker ps

# Fix permissions (Linux)
sudo usermod -aG docker $USER
```

**Test Timeouts:**
```bash
# Increase timeout for slow tests
pytest tests/integration/ --timeout=600

# Run specific slow tests
pytest tests/integration/ -m slow --timeout=900
```

**Coverage Too Low:**
```bash
# Generate detailed coverage report
pytest --cov=src --cov-report=html --cov-report=term-missing

# View uncovered lines
open htmlcov/index.html

# Run with branch coverage
pytest --cov=src --cov-branch
```

**Qdrant Connection Issues:**
```bash
# Check Qdrant health
curl http://localhost:6333/health

# View Qdrant logs
docker-compose logs qdrant

# Reset Qdrant data
docker-compose down -v && docker-compose up -d qdrant
```

### Debug Mode

Enable detailed debugging:

```bash
# Verbose pytest output
pytest tests/integration/ -v -s --tb=long

# Debug specific test
pytest tests/integration/test_document_ingestion_pipeline.py::test_end_to_end_document_ingestion -v -s

# Integration test runner with debugging
python scripts/run_integration_tests.py --categories integration --verbose
```

### Performance Debugging

```bash
# Profile test execution
python -m cProfile -o profile_output scripts/run_integration_tests.py --categories performance

# Memory profiling
python -m memory_profiler scripts/run_integration_tests.py

# Benchmark analysis
pytest --benchmark-only --benchmark-sort=mean tests/integration/
```

## Best Practices

### Writing Integration Tests

1. **Use Isolated Environments**
   ```python
   @pytest.fixture(scope="module")
   def isolated_qdrant():
       # Use testcontainers for isolation
   ```

2. **Clean Test Data**
   ```python
   @pytest.fixture
   def cleanup_tracker():
       # Register cleanup functions
       yield register_cleanup
       # Automatic cleanup
   ```

3. **Realistic Test Scenarios**
   ```python
   def test_real_world_workflow():
       # Test actual user workflows
   ```

4. **Performance Aware**
   ```python
   @pytest.mark.performance
   def test_with_benchmarks(benchmark):
       # Include performance validation
   ```

### Test Organization

- Group related tests in classes
- Use descriptive test names
- Include docstrings explaining test purpose
- Tag tests with appropriate markers
- Separate setup/teardown logic into fixtures

### Continuous Improvement

- Monitor test execution time
- Update performance baselines regularly
- Add tests for new features
- Refactor slow or flaky tests
- Review coverage reports regularly

## Integration with Development Workflow

### Pre-commit Integration

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run integration tests in pre-commit
# Add to .pre-commit-config.yaml:
- repo: local
  hooks:
    - id: integration-tests
      name: Integration Tests
      entry: python scripts/run_integration_tests.py --categories smoke
```

### IDE Integration

**VS Code Configuration** (`.vscode/settings.json`):
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/integration/"],
  "python.testing.cwd": "${workspaceFolder}"
}
```

**PyCharm Configuration:**
- Set test runner to pytest
- Configure test directories
- Add environment variables for integration testing

This comprehensive integration testing suite ensures high-quality, reliable software delivery while maintaining development velocity and providing confidence in system behavior under various conditions.