# Log Aggregation Testing Framework

This directory contains comprehensive tests for the log aggregation functionality of workspace-qdrant-mcp.

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
