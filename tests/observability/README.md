# Observability Testing Framework

This directory contains comprehensive tests for observability functionality of workspace-qdrant-mcp, including log aggregation and health monitoring.

## Overview

The log aggregation testing framework validates all aspects of the logging system including:

- **Log Levels**: Testing DEBUG, INFO, WARNING, ERROR, CRITICAL levels and filtering
- **Structured Logging**: JSON serialization with required fields (timestamp, level, message, context)
- **Multi-Source Collection**: Log aggregation from MCP server, Rust daemon, and CLI components
- **Log Rotation**: File size-based rotation with configurable thresholds
- **Log Retention**: Automatic cleanup of old log files based on retention policies
- **Correlation IDs**: Distributed tracing across multiple components
- **Log Persistence**: File-based storage and retrieval
- **Buffering & Batching**: Async log buffering for performance
- **Filtering**: Content-based, level-based, and regex-based filtering

## Running Tests

### Run All Log Aggregation Tests

```bash
cd /Users/chris/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp
uv run pytest tests/observability/test_log_aggregation.py -v
```

### Run Specific Test Classes

```bash
# Test log levels
uv run pytest tests/observability/test_log_aggregation.py::TestLogLevels -v

# Test structured logging
uv run pytest tests/observability/test_log_aggregation.py::TestStructuredLogging -v

# Test log rotation
uv run pytest tests/observability/test_log_aggregation.py::TestLogRotationAndRetention -v

# Test correlation IDs
uv run pytest tests/observability/test_log_aggregation.py::TestCorrelationIDs -v

# Test log persistence
uv run pytest tests/observability/test_log_aggregation.py::TestLogPersistenceAndRetrieval -v

# Test buffering
uv run pytest tests/observability/test_log_aggregation.py::TestLogBufferingAndBatching -v

# Test filtering
uv run pytest tests/observability/test_log_aggregation.py::TestLogFiltering -v

# Test integration scenarios
uv run pytest tests/observability/test_log_aggregation.py::TestIntegrationScenarios -v
```

### Run Individual Tests

```bash
# Test specific functionality
uv run pytest tests/observability/test_log_aggregation.py::TestLogLevels::test_debug_level_logging -v
uv run pytest tests/observability/test_log_aggregation.py::TestStructuredLogging::test_json_serialization -v
```

### Run with Coverage

```bash
uv run pytest tests/observability/test_log_aggregation.py --cov=src/python/common/logging --cov=src/python/common/observability --cov-report=html
```

## Test Categories

### 1. Log Level Tests (`TestLogLevels`)

Tests that validate proper logging at all severity levels:

- `test_debug_level_logging`: Validates DEBUG level logs
- `test_info_level_logging`: Validates INFO level logs
- `test_warning_level_logging`: Validates WARNING level logs
- `test_error_level_logging`: Validates ERROR level logs with exceptions
- `test_critical_level_logging`: Validates CRITICAL level logs
- `test_log_level_filtering`: Validates level-based filtering

### 2. Structured Logging Tests (`TestStructuredLogging`)

Tests for JSON serialization and required fields:

- `test_json_serialization`: Validates JSON log format
- `test_required_fields_present`: Checks timestamp, level, message, context
- `test_custom_fields_and_metadata`: Validates custom field preservation
- `test_stack_traces_for_errors`: Validates exception stack trace capture

### 3. Multi-Source Collection Tests (`TestLogCollectionFromMultipleSources`)

Tests for aggregating logs from different components:

- `test_mcp_server_logs`: Validates MCP server log collection
- `test_rust_daemon_logs`: Validates Rust daemon log collection
- `test_cli_tool_logs`: Validates CLI tool log collection
- `test_multiple_sources_aggregation`: Validates combined aggregation

### 4. Log Rotation Tests (`TestLogRotationAndRetention`)

Tests for file rotation and retention policies:

- `test_rotation_on_size`: Validates size-based rotation (triggers at 10KB)
- `test_retention_policy`: Validates old file cleanup (keeps max 3 files)
- `test_compression_on_rotation`: Validates gzip compression of rotated files

### 5. Correlation ID Tests (`TestCorrelationIDs`)

Tests for distributed tracing:

- `test_correlation_id_propagation`: Validates ID propagation across operations
- `test_operation_monitor_with_correlation_id`: Validates OperationMonitor IDs
- `test_nested_operations_with_correlation`: Validates nested operation tracking

### 6. Persistence Tests (`TestLogPersistenceAndRetrieval`)

Tests for log storage and retrieval:

- `test_log_persistence_to_file`: Validates file persistence
- `test_log_retrieval_by_level`: Validates level-based retrieval
- `test_log_retrieval_by_timestamp`: Validates time-based retrieval
- `test_log_searching_by_content`: Validates content-based search

### 7. Buffering Tests (`TestLogBufferingAndBatching`)

Tests for performance optimization:

- `test_buffered_logging`: Validates log buffering mechanism
- `test_batch_logging_performance`: Validates batch write performance
- `test_async_logging_non_blocking`: Validates non-blocking async logging

### 8. Filtering Tests (`TestLogFiltering`)

Tests for advanced filtering:

- `test_filter_by_component`: Validates component-based filtering
- `test_filter_by_custom_criteria`: Validates custom field filtering
- `test_regex_filtering`: Validates regex pattern matching

### 9. Integration Tests (`TestIntegrationScenarios`)

End-to-end scenarios:

- `test_end_to_end_distributed_operation`: Complete distributed operation flow
- `test_concurrent_logging_from_multiple_sources`: Concurrent async logging
- `test_error_handling_with_full_context`: Error propagation with context

## Interpreting Test Results

### Successful Test Run

```
tests/observability/test_log_aggregation.py::TestLogLevels::test_debug_level_logging PASSED [ 2%]
tests/observability/test_log_aggregation.py::TestLogLevels::test_info_level_logging PASSED [ 4%]
...
================================= 45 passed in 2.34s =================================
```

### Test Failures

If a test fails, examine the output for:

1. **Assertion errors**: Check what was expected vs. what was found
2. **File not found errors**: Verify log directory creation
3. **Timeout errors**: Increase wait times for async operations
4. **Content mismatch**: Check log format and structure

### Common Issues

**Issue**: Tests fail with "log file not found"
**Solution**: Ensure temp directory fixtures are working correctly

**Issue**: Rotation tests don't create rotated files
**Solution**: Check if enough data is being written to trigger rotation

**Issue**: JSON parsing fails
**Solution**: Verify `serialize=True` is set on the logger handler

**Issue**: Correlation IDs not found
**Solution**: Check that extra fields are being logged correctly

## Log Format Examples

### Standard Format (Text)

```
2025-10-21 14:30:45.123 | INFO     | module:function:42 | {'component': 'mcp_server'} | Message here
```

### JSON Format

```json
{
  "text": "Message here",
  "record": {
    "time": {"repr": "2025-10-21T14:30:45.123Z", "timestamp": 1729519845.123},
    "level": {"name": "INFO", "no": 20},
    "name": "module",
    "function": "function",
    "line": 42,
    "extra": {
      "component": "mcp_server",
      "correlation_id": "uuid-here"
    }
  }
}
```

## Performance Benchmarks

Expected performance characteristics:

- **Buffered logging**: 1000 messages should complete in < 1 second
- **Batch logging**: 100 messages should complete in < 5 seconds
- **Rotation**: Should occur within 100ms after size threshold
- **Compression**: Rotated files should be readable with gzip

## Configuration

The tests use isolated logger instances with test-specific configuration:

```python
logger.add(
    log_file,
    rotation="1 MB",      # Rotate at 1MB
    retention="2 days",   # Keep for 2 days
    compression="gz",     # Compress rotated files
    serialize=True,       # JSON format
    enqueue=True          # Async buffering
)
```

## Maintenance

### Adding New Tests

1. Create new test method in appropriate class
2. Use `isolated_logger` or `json_logger` fixture
3. Write assertions that validate specific behavior
4. Document expected behavior in docstring

### Updating Tests

When logging infrastructure changes:

1. Update affected test cases
2. Verify all test categories still pass
3. Add new tests for new features
4. Update this README with new information

## References

- **Loguru documentation**: https://loguru.readthedocs.io/
- **Log aggregation patterns**: See `src/python/common/logging/`
- **Monitoring framework**: See `src/python/common/observability/monitoring.py`
- **Metrics system**: See `src/python/common/observability/metrics.py`

## Support

For issues or questions about log aggregation testing:

1. Check test output for detailed error messages
2. Review log files in temporary test directories
3. Examine loguru configuration in `src/python/common/logging/loguru_config.py`
4. Verify environment variables (WQM_STDIO_MODE, etc.)

---

# Health Check Endpoint Verification Tests

Comprehensive health check endpoint verification tests validating all health monitoring infrastructure.

## Overview

The health check testing framework validates:

- **Liveness checks**: Is the service running?
- **Readiness checks**: Is the service ready to accept requests?
- **Startup checks**: Has initialization completed?
- **Deep health checks**: Are all dependencies healthy?
- **Response validation**: Status codes, JSON format, response timing, error messages
- **Component health**: MCP server, Qdrant, Rust daemon, SQLite, embedding model
- **Health states**: Healthy, degraded, unhealthy, recovery scenarios
- **Health aggregation**: Cross-component health coordination

## Running Health Check Tests

### Run All Health Check Tests

```bash
cd /Users/chris/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp
uv run pytest tests/observability/test_health_checks.py -v
```

### Run Specific Test Classes

```bash
# Test liveness checks
uv run pytest tests/observability/test_health_checks.py::TestLivenessChecks -v

# Test readiness checks
uv run pytest tests/observability/test_health_checks.py::TestReadinessChecks -v

# Test startup checks
uv run pytest tests/observability/test_health_checks.py::TestStartupChecks -v

# Test deep health checks
uv run pytest tests/observability/test_health_checks.py::TestDeepHealthChecks -v

# Test response validation
uv run pytest tests/observability/test_health_checks.py::TestResponseValidation -v

# Test component health
uv run pytest tests/observability/test_health_checks.py::TestComponentHealth -v

# Test health states
uv run pytest tests/observability/test_health_checks.py::TestHealthStateScenarios -v

# Test health aggregation
uv run pytest tests/observability/test_health_checks.py::TestHealthAggregation -v

# Test coordinator integration
uv run pytest tests/observability/test_health_checks.py::TestHealthCoordinatorIntegration -v

# Test performance and edge cases
uv run pytest tests/observability/test_health_checks.py::TestPerformanceAndEdgeCases -v
```

### Run Individual Tests

```bash
# Test specific functionality
uv run pytest tests/observability/test_health_checks.py::TestLivenessChecks::test_health_status_basic_liveness -v
uv run pytest tests/observability/test_health_checks.py::TestResponseValidation::test_response_timing_fast -v
```

### Run with Coverage

```bash
uv run pytest tests/observability/test_health_checks.py --cov=src/python/common/observability --cov-report=html
```

## Test Categories

### 1. Liveness Checks (`TestLivenessChecks`) - 4 tests

Tests basic service availability:

- `test_health_checker_instantiation`: Validates HealthChecker can be created
- `test_get_health_checker_singleton`: Validates singleton pattern
- `test_health_status_basic_liveness`: Validates basic health status retrieval
- `test_liveness_check_response_structure`: Validates response structure

**What it validates**: Service is running and can respond to health requests.

### 2. Readiness Checks (`TestReadinessChecks`) - 4 tests

Tests service readiness to accept requests:

- `test_readiness_all_checks_enabled`: Validates all required checks are registered
- `test_readiness_check_execution`: Validates checks can execute
- `test_readiness_response_time_acceptable`: Validates response time < 150ms
- `test_readiness_critical_components`: Validates critical component marking

**What it validates**: Service is initialized and dependencies are ready.

### 3. Startup Checks (`TestStartupChecks`) - 3 tests

Tests initialization completeness:

- `test_startup_initialization_complete`: Validates standard checks initialized
- `test_startup_check_registration`: Validates custom check registration
- `test_startup_background_monitoring`: Validates background monitoring startup

**What it validates**: Initialization has completed successfully.

### 4. Deep Health Checks (`TestDeepHealthChecks`) - 4 tests

Tests comprehensive dependency health:

- `test_deep_health_all_components`: Validates all components checked
- `test_deep_health_system_resources`: Validates resource metrics (CPU, memory, disk)
- `test_deep_health_detailed_diagnostics`: Validates detailed diagnostic information
- `test_deep_health_dependency_tracking`: Validates dependency health tracking

**What it validates**: All system components and dependencies are healthy.

### 5. Response Validation (`TestResponseValidation`) - 6 tests

Tests response format and characteristics:

- `test_response_status_codes`: Validates status values (healthy/degraded/unhealthy)
- `test_response_json_format`: Validates JSON serializability
- `test_response_timing_fast`: Validates overall response time < 200ms
- `test_response_component_timing`: Validates component response time tracking
- `test_response_timestamp_present`: Validates timestamp presence and recency
- `test_response_error_messages`: Validates error message inclusion

**What it validates**: Responses follow expected format and performance standards.

### 6. Component Health (`TestComponentHealth`) - 5 tests

Tests individual component health checks:

- `test_component_system_resources_healthy`: Validates system resources check
- `test_component_health_status_enum`: Validates HealthStatus enum usage
- `test_component_health_with_details`: Validates detailed metrics inclusion
- `test_component_check_timeout`: Validates timeout handling (0.5s timeout)
- `test_component_consecutive_failures`: Validates failure tracking

**What it validates**: Individual components report health correctly.

### 7. Health State Scenarios (`TestHealthStateScenarios`) - 4 tests

Tests various health states:

- `test_healthy_state_all_components`: Validates healthy when all components healthy
- `test_degraded_state_non_critical_failure`: Validates degraded on non-critical failure
- `test_unhealthy_state_critical_failure`: Validates unhealthy on critical failure
- `test_recovery_scenario`: Validates recovery from unhealthy to healthy

**What it validates**: Health state transitions work correctly.

### 8. Health Aggregation (`TestHealthAggregation`) - 5 tests

Tests cross-component health aggregation:

- `test_aggregation_multiple_components`: Validates multi-component aggregation
- `test_aggregation_overall_status_logic`: Validates status aggregation logic
- `test_aggregation_message_generation`: Validates aggregated message generation
- `test_aggregation_concurrent_checks`: Validates concurrent check execution
- `test_aggregation_critical_vs_non_critical`: Validates critical/non-critical distinction

**What it validates**: Health aggregation logic is correct.

### 9. Coordinator Integration (`TestHealthCoordinatorIntegration`) - 4 tests

Tests HealthCoordinator advanced features:

- `test_coordinator_initialization`: Validates coordinator initialization
- `test_coordinator_unified_health_status`: Validates unified status API
- `test_coordinator_component_metrics`: Validates component metrics tracking
- `test_coordinator_alert_generation`: Validates alert generation

**What it validates**: Advanced health coordination features work correctly.

### 10. Performance and Edge Cases (`TestPerformanceAndEdgeCases`) - 5 tests

Tests performance and error handling:

- `test_health_check_under_load`: Validates 10 concurrent health checks
- `test_health_check_with_disabled_checker`: Validates disabled checker behavior
- `test_health_check_exception_handling`: Validates exception handling
- `test_health_check_missing_component`: Validates missing component handling
- `test_health_check_disabled_component`: Validates disabled component handling

**What it validates**: System handles edge cases and performs well under load.

## Interpreting Health Check Test Results

### Successful Test Run

```
tests/observability/test_health_checks.py::TestLivenessChecks::test_health_status_basic_liveness PASSED [  2%]
tests/observability/test_health_checks.py::TestReadinessChecks::test_readiness_all_checks_enabled PASSED [ 11%]
...
================================= 44 passed in 5.24s =================================
```

### Test Failures

Common failure scenarios and solutions:

**Issue**: `test_readiness_response_time_acceptable` fails with timing > 150ms
**Cause**: Slow system or high load during test execution
**Solution**: Normal on slower systems; timing test allows reasonable tolerance

**Issue**: `test_coordinator_initialization` fails
**Cause**: Database initialization issue
**Solution**: Check SQLite availability and permissions

**Issue**: Component health checks return "unhealthy"
**Cause**: Dependencies not available (Qdrant, embedding service)
**Solution**: Health checks are designed to handle missing dependencies gracefully

**Issue**: Timeout in concurrent tests
**Cause**: System resource constraints
**Solution**: Reduce concurrent test load or increase timeout

## Health Check Response Format

### Standard Health Response

```json
{
  "status": "healthy",
  "message": "All systems operational",
  "timestamp": 1729519845.123,
  "components": {
    "system_resources": {
      "status": "healthy",
      "message": "System resources OK",
      "details": {
        "memory": {
          "percent_used": 45.2,
          "available_gb": 8.5,
          "total_gb": 16.0
        },
        "cpu": {
          "percent_used": 15.3,
          "core_count": 8
        },
        "disk": {
          "percent_used": 62.1,
          "free_gb": 250.0,
          "total_gb": 500.0
        }
      },
      "last_check": 1729519845.120,
      "response_time": 0.103,
      "error": null
    },
    "qdrant_connectivity": {
      "status": "healthy",
      "message": "Qdrant healthy with 5 collections",
      "details": {
        "qdrant_url": "http://localhost:6333",
        "collections_count": 5,
        "project": "test_project",
        "embedding_model": "all-MiniLM-L6-v2"
      },
      "response_time": 0.015,
      "error": null
    }
  }
}
```

### Health Status Values

- `healthy`: All critical components operational
- `degraded`: Some non-critical components have issues
- `unhealthy`: Critical components failing
- `unknown`: Health check disabled or error

### HTTP Status Code Mapping (for HTTP endpoints)

- `200 OK`: healthy
- `429 Too Many Requests`: degraded (rate limiting)
- `503 Service Unavailable`: unhealthy

## Performance Benchmarks

Expected performance characteristics validated by tests:

- **Liveness check**: < 50ms response time
- **Readiness check**: < 150ms response time (allows for dependency checks)
- **Deep health check**: < 200ms response time
- **Concurrent checks**: 10 concurrent checks should complete in < 2s
- **Component timeout**: 0.5s to 5s depending on component

## Component Health Checks

Tests validate health checks for these components:

1. **System Resources**
   - CPU usage (< 90% healthy, < 95% degraded, >= 95% unhealthy)
   - Memory usage (< 85% healthy, < 95% degraded, >= 95% unhealthy)
   - Disk usage (< 90% healthy, < 95% degraded, >= 95% unhealthy)

2. **Qdrant Connectivity**
   - Connection status
   - Collection count
   - Response time

3. **Embedding Service**
   - Model availability
   - Embedding generation capability
   - Response time

4. **File Watchers**
   - Active watch count
   - Error watch count
   - Paused watch count

5. **Configuration**
   - Configuration validity
   - Configuration warnings

## Using Health Checks in Production

### CLI Usage

```bash
# Check health status
workspace-qdrant-health

# Continuous monitoring
workspace-qdrant-health --watch

# Detailed analysis
workspace-qdrant-health --analyze

# Generate report
workspace-qdrant-health --report --output health_report.json
```

### Programmatic Usage

```python
from common.observability.health import get_health_checker

# Get health checker instance
health_checker = get_health_checker()

# Get overall health status
health_status = await health_checker.get_health_status()
print(f"Status: {health_status['status']}")

# Run specific component check
component_health = await health_checker.run_check("qdrant_connectivity")
print(f"Qdrant: {component_health.status.value}")

# Get detailed diagnostics
diagnostics = await health_checker.get_detailed_diagnostics()
```

### Health Coordinator Usage

```python
from common.observability.health_coordinator import get_health_coordinator

# Get coordinator instance
coordinator = await get_health_coordinator()

# Start monitoring
await coordinator.start_monitoring()

# Get unified health status
unified_status = await coordinator.get_unified_health_status()

# Get dashboard data
dashboard_data = await coordinator.get_health_dashboard_data()

# Stop monitoring
await coordinator.stop_monitoring()
```

## CI/CD Integration

Health check tests are ideal for CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run health check tests
  run: |
    uv run pytest tests/observability/test_health_checks.py -v

# Example with coverage
- name: Run health check tests with coverage
  run: |
    uv run pytest tests/observability/test_health_checks.py \
      --cov=src/python/common/observability \
      --cov-report=xml \
      --cov-fail-under=80
```

## Maintenance

### Adding New Component Health Checks

1. Register check in HealthChecker initialization:

```python
self.register_check("new_component", self._check_new_component)
```

2. Implement check method:

```python
async def _check_new_component(self) -> Dict[str, Any]:
    try:
        # Check logic here
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "Component healthy",
            "details": {"metric": "value"}
        }
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"Component check failed: {e}",
            "details": {"error": str(e)}
        }
```

3. Add tests for new component:

```python
@pytest.mark.asyncio
async def test_new_component_health(self, health_checker):
    """Test new component health check."""
    result = await health_checker.run_check("new_component")
    assert result.status in [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY]
```

### Updating Tests

When health check infrastructure changes:

1. Update affected test cases
2. Verify all 44 tests still pass
3. Add new tests for new features
4. Update this README with new information

## References

- **Health check implementation**: `src/python/common/observability/health.py`
- **Health coordinator**: `src/python/common/observability/health_coordinator.py`
- **CLI health command**: `src/python/wqm_cli/cli/health.py`
- **Metrics system**: `src/python/common/observability/metrics.py`
