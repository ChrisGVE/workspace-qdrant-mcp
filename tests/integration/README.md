# Integration Test Framework

Comprehensive Docker-based integration testing framework for workspace-qdrant-mcp, validating the entire MCP-daemon-Qdrant stack with performance monitoring and detailed reporting.

## Overview

This integration test framework provides **79 comprehensive tests** across **8 test suites**, validating:
- Real-time file watching and ingestion
- gRPC communication under load and stress
- Connection failure and recovery scenarios
- Multi-component state consistency
- Concurrent operation conflict resolution
- MCP-daemon communication protocols
- End-to-end ingestion workflows
- Performance monitoring and regression detection

## Quick Start

### Run All Integration Tests

```bash
# With automatic Docker management
./scripts/run_integration_tests.py

# Skip Docker management (assume services running)
./scripts/run_integration_tests.py --skip-docker

# With cleanup after execution
./scripts/run_integration_tests.py --cleanup
```

### Run Specific Test Suite

```bash
# Run just one suite
./scripts/run_integration_tests.py --suite test_grpc_load_stress.py

# Run with pytest directly
uv run pytest tests/integration/test_realtime_file_watching.py -v
```

### Manual Docker Setup

```bash
# Start services
cd docker/integration-tests
docker-compose up -d

# Wait for health checks
sleep 10

# Run tests
uv run pytest tests/integration/ -v

# Stop services
docker-compose down -v
```

## Test Suites

### 1. MCP-Daemon Communication (9 tests)
**File**: `test_mcp_daemon_docker_integration.py`

Tests gRPC communication between MCP server and Rust daemon.

- gRPC call success/failure handling
- Connection timeout behavior
- Protocol compliance (message format, error codes)
- Message serialization/deserialization
- Lifecycle management (startup/shutdown coordination)
- Connection error recovery

**Key Validations**:
- gRPC calls complete within timeout (5 seconds)
- Error responses include proper error codes
- Message format follows protocol specification
- Connections clean up properly on shutdown

### 2. End-to-End Ingestion Workflow (8 tests)
**File**: `test_e2e_ingestion_workflow.py`

Tests complete file ingestion pipeline from MCP trigger to Qdrant storage.

- Text file ingestion with metadata preservation
- Code file processing (Python, JavaScript, etc.)
- Markdown file processing
- JSON document handling
- Large file processing (>1MB)
- Metadata extraction and enrichment
- Pipeline error handling

**Key Validations**:
- Files processed end-to-end (MCP → daemon → Qdrant)
- Metadata preserved throughout pipeline
- Chunks created with correct parameters
- Search returns ingested content

### 3. Real-Time File Watching (10 tests)
**File**: `test_realtime_file_watching.py`

Tests SQLite-based file watching integration.

- File creation/modification/deletion detection
- Nested directory watching
- SQLite watch configuration (CRUD)
- Enable/disable watch folders
- Multiple concurrent watches
- Event debouncing (2.0s threshold)
- Batch processing optimization

**Key Validations**:
- Daemon polls SQLite for config changes
- File events trigger ingestion
- Debouncing prevents event storms
- Multiple watches operate independently

### 4. gRPC Load and Stress Testing (13 tests)
**File**: `test_grpc_load_stress.py`

Tests gRPC performance under various load conditions.

- Concurrent request handling (10-200 clients)
- Large payload processing (1MB-50MB)
- Rapid request sequences (bursts, sustained)
- Connection pooling efficiency
- Timeout handling under load
- Graceful degradation under stress
- Recovery after stress

**Performance Benchmarks**:
- Average latency: < 1.0s (P95 < 2.0s)
- Throughput: >10 req/s (ingestion), >50 req/s (search)
- Success rate under stress: >70%
- Memory increase: <500MB under stress
- Connection reuse: 50% latency reduction

### 5. Connection Failure and Recovery (17 tests)
**File**: `test_connection_failure_recovery.py`

Tests Write Path Architecture fallback and recovery mechanisms.

- Daemon unavailable at startup
- Fallback to direct Qdrant writes
- Connection loss during operations
- Daemon restart and reconnection
- Network partition detection
- Exponential backoff (100ms → 3200ms)
- State recovery from SQLite

**Key Validations**:
- MCP server starts despite daemon unavailability
- Fallback mode logs WARNING messages
- Response includes `fallback_mode: "direct_qdrant_write"`
- Periodic availability checks exit fallback mode
- Automatic reconnection after daemon restart

### 6. State Consistency Validation (14 tests)
**File**: `test_state_consistency.py`

Tests data consistency across all components.

- SQLite ACID properties (atomicity, isolation, rollback)
- Qdrant collection consistency (point counts, metadata)
- Cross-component synchronization (SQLite ↔ daemon ↔ Qdrant)
- Partial failure recovery and cleanup
- WAL mode concurrent access
- Transaction integrity
- Crash recovery protocols

**Key Validations**:
- Atomic operations (all-or-nothing commits)
- Transaction rollback preserves consistency
- Concurrent transactions isolated properly
- WAL mode enables concurrent reads during writes
- State consistent after component failures

### 7. Concurrent Operation Conflicts (11 tests)
**File**: `test_concurrent_operations.py`

Tests conflict handling between MCP and CLI operations.

- Simultaneous MCP-CLI operations on same collections
- Concurrent same-file ingestion prevention
- Overlapping watch folder detection
- Collection-level locking
- SQLite WAL mode write serialization
- Qdrant concurrent upsert handling
- Race condition prevention (compare-and-swap)

**Conflict Resolution**:
- Same-file ingestion: First wins, second rejected
- Overlapping watches: Allow with deduplication
- Duplicate watches: Keep first, disable second
- Metadata conflicts: Version check + retry
- Collection locks: Wait and retry with timeout

### 8. Performance Monitoring (14 tests)
**File**: `test_performance_monitoring.py`

Tests performance monitoring system for regression detection.

- Metrics collection (latency, throughput, resources)
- Latency percentiles (P50, P95, P99)
- Resource monitoring (CPU %, memory MB)
- Baseline management (persistence, loading)
- Regression detection (latency, throughput, memory)
- Report generation

**Regression Thresholds**:
- Latency: 20% increase tolerance
- Throughput: 20% decrease tolerance
- Memory: 30% increase tolerance

## Test Orchestration

The `run_integration_tests.py` script provides:

- **Automated Docker Management**: Starts/stops services automatically
- **Sequential Execution**: Runs all 8 test suites in order
- **Comprehensive Reporting**: HTML + JSON reports
- **Failure Analysis**: Detailed error output and debugging info
- **CI/CD Integration**: JSON output for pipeline integration
- **Test Data Cleanup**: Automatic cleanup of temporary files

### Usage

```bash
# Full test run with reports
./scripts/run_integration_tests.py

# Specific suite only
./scripts/run_integration_tests.py --suite test_grpc_load_stress.py

# Skip Docker (assume services already running)
./scripts/run_integration_tests.py --skip-docker

# With cleanup
./scripts/run_integration_tests.py --cleanup

# Custom output directory
./scripts/run_integration_tests.py --output-dir custom_results/
```

### Generated Reports

- `comprehensive_report.html` - HTML report with all test results
- `test_results.json` - JSON report for CI/CD integration
- `{suite_name}_junit.xml` - JUnit XML for each suite
- `{suite_name}_report.html` - HTML report for each suite

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: |
          pip install uv
          uv sync --dev

      - name: Run integration tests
        run: ./scripts/run_integration_tests.py

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: test_results/

      - name: Publish test report
        if: always()
        uses: mikepenz/action-junit-report@v3
        with:
          report_paths: 'test_results/*_junit.xml'
```

## Performance Monitoring

Integration tests automatically collect performance metrics:

- **Latency**: Average, median, P95, P99, min, max
- **Throughput**: Operations per second
- **Resources**: CPU %, memory MB (average and peak)
- **Errors**: Count of errors and warnings

### Setting Baselines

```python
from tests.integration.performance_monitoring import PerformanceMonitor

monitor = PerformanceMonitor()

async with monitor.monitor_test("my_test"):
    # Run test operations
    pass

# Set baseline for future comparisons
monitor.set_baseline(monitor.current_metrics)
```

### Checking Regressions

```python
# Automatically checks against baseline
warnings = monitor.check_regression(metrics)

if warnings:
    for warning in warnings:
        print(f"REGRESSION: {warning}")
```

## Docker Services

The integration tests use Docker Compose with the following services:

- **qdrant**: Vector database (ports 6333, 6334)
- **daemon**: Rust file watching and ingestion engine
- **mcp-server**: FastAPI MCP server (port 8000)

### Service Configuration

```yaml
# docker/integration-tests/docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports: ["6333:6333", "6334:6334"]

  daemon:
    build: ../../
    environment:
      QDRANT_URL: http://qdrant:6333

  mcp-server:
    build: ../../
    ports: ["8000:8000"]
    depends_on: [daemon, qdrant]
```

## Troubleshooting

### Docker Services Won't Start

```bash
# Check Docker is running
docker ps

# Clean up existing services
cd docker/integration-tests
docker-compose down -v
docker system prune -f

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

### Tests Timeout

- Increase timeout in test fixtures
- Check Docker resource limits
- Verify network connectivity between containers

### Permission Errors

```bash
# Make scripts executable
chmod +x scripts/run_integration_tests.py

# Fix ownership if needed
sudo chown -R $USER:$USER test_results/
```

### Port Conflicts

```bash
# Check what's using ports
lsof -i :6333  # Qdrant HTTP
lsof -i :6334  # Qdrant gRPC
lsof -i :8000  # MCP server

# Kill conflicting processes
kill -9 <PID>
```

## Test Development

### Adding New Tests

1. Create test file in `tests/integration/`
2. Use existing fixtures (`docker_services`, `docker_compose_file`)
3. Follow naming convention: `test_*.py`
4. Add to orchestration script test suites list

### Example Test

```python
import pytest

@pytest.mark.asyncio
async def test_my_integration(docker_services):
    """Test my integration scenario."""
    # Use docker_services fixture for service URLs
    qdrant_url = docker_services["qdrant_url"]

    # Your test code here
    assert True
```

### Performance Test

```python
from tests.integration.performance_monitoring import PerformanceMonitor

@pytest.mark.asyncio
async def test_with_performance_monitoring():
    monitor = PerformanceMonitor()

    async with monitor.monitor_test("my_perf_test"):
        for i in range(100):
            start = time.time()
            await do_work()
            latency_ms = (time.time() - start) * 1000
            monitor.record_operation(latency_ms)

    # Automatically checks for regressions
    report = monitor.generate_report(monitor.current_metrics)
    print(report)
```

## Architecture Validation

The integration tests validate the complete architecture:

```
┌─────────────┐      gRPC      ┌──────────────┐
│ MCP Server  │ ◄─────────────► │ Rust Daemon  │
│ (FastAPI)   │                 │ (memexd)     │
└─────────────┘                 └──────────────┘
      │                               │
      │                               │
      ▼                               ▼
┌─────────────┐                 ┌──────────────┐
│   SQLite    │                 │    Qdrant    │
│ (state.db)  │                 │  (vectors)   │
└─────────────┘                 └──────────────┘
```

**Validated Flows**:
1. MCP → SQLite (watch config)
2. Daemon → SQLite (poll for changes)
3. Daemon → Qdrant (write vectors)
4. MCP → Qdrant (fallback writes)
5. CLI → Daemon → Qdrant (ingestion)

## Summary

**Total Test Coverage**: 79 tests across 8 suites
**Test Execution Time**: ~2-5 minutes (depending on load tests)
**Report Formats**: HTML, JSON, JUnit XML
**CI/CD Ready**: ✓
**Performance Monitoring**: ✓
**Docker Orchestration**: ✓

For questions or issues, see the project's main README or create an issue on GitHub.
