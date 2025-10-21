# Test Result Aggregation System

Unified system for aggregating and analyzing test results from multiple sources in workspace-qdrant-mcp.

## Overview

The test result aggregation system provides:

- **Unified data model** for all test types (unit, integration, e2e, benchmark, performance)
- **Multiple parsers** for different test result formats (pytest, cargo, custom JSON)
- **SQLite storage** for persistent test history
- **Query API** for analysis, trend detection, and flaky test identification
- **Incremental aggregation** for CI/CD pipeline stages

## Supported Formats

### pytest
- JUnit XML (`pytest --junit-xml results.xml`)
- JSON output (via `pytest-json-report` plugin)
- pytest-benchmark results

### Cargo Test
- Text output (default `cargo test` format)
- JSON output (`cargo test -- --format json`)

### Custom Formats
- Benchmark JSON (workspace-qdrant-mcp benchmark results)
- gRPC integration test results

## Quick Start

### Basic Usage

```python
from tests.reporting import TestResultAggregator, TestSource

# Create aggregator
aggregator = TestResultAggregator()

# Aggregate pytest results
run = aggregator.aggregate_from_file('junit.xml', TestSource.PYTEST)

print(f"Total tests: {run.total_tests}")
print(f"Success rate: {run.success_rate:.1f}%")
print(f"Failed: {run.failed_tests}")
```

### Multi-Source Aggregation (CI/CD)

```python
from tests.reporting import TestResultAggregator, TestSource

aggregator = TestResultAggregator()

# Aggregate multiple test stages into one run
combined_run = aggregator.aggregate_multiple([
    {'file': 'unit_tests.xml', 'source': TestSource.PYTEST},
    {'file': 'integration_tests.xml', 'source': TestSource.PYTEST},
    {'file': 'cargo_test_output.txt', 'source': TestSource.CARGO},
    {'file': 'benchmarks.json', 'source': TestSource.BENCHMARK_JSON},
], run_id='ci-pipeline-123')

print(f"Combined: {combined_run.total_tests} tests from {len(combined_run.suites)} suites")
```

### Query API

```python
from tests.reporting import TestResultQuery

query = TestResultQuery()

# Get latest test run
latest = query.get_latest_run()
print(f"Latest run: {latest['run_id']} with {latest['success_rate']:.1f}% success")

# Find failed tests
failed = query.get_failed_tests(run_id)
for test in failed:
    print(f"  FAILED: {test['name']} - {test['error_message']}")

# Identify slow tests
slow_tests = query.get_slow_tests(run_id, threshold_ms=1000, limit=10)
for test in slow_tests:
    print(f"  SLOW: {test['name']} - {test['duration_ms']:.0f}ms")

# Get benchmark results
benchmarks = query.get_benchmark_results(run_id)
for bench in benchmarks:
    print(f"  {bench['name']}: {bench['avg_ms']:.2f}ms (min: {bench['min_ms']:.2f}, max: {bench['max_ms']:.2f})")
```

### Flaky Test Detection

```python
from tests.reporting import TestResultQuery

query = TestResultQuery()

# Identify tests with intermittent failures
flaky_tests = query.get_flaky_tests(
    days=30,  # Look back 30 days
    min_runs=5,  # At least 5 executions
    flaky_threshold=0.2  # Up to 20% failure rate
)

for test in flaky_tests:
    print(f"FLAKY: {test['name']}")
    print(f"  Failure rate: {test['failure_rate']*100:.1f}%")
    print(f"  Runs: {test['total_runs']} (passed: {test['passed']}, failed: {test['failed']})")
```

### Trend Analysis

```python
from tests.reporting import TestResultQuery

query = TestResultQuery()

# Get success rate trend over time
trend = query.get_trend_data(days=30, metric='success_rate')

for point in trend[:5]:
    print(f"{point['timestamp']}: {point['value']:.1f}%")

# Compare two test runs
comparison = query.compare_runs('run-1', 'run-2')
print(f"Success rate delta: {comparison['differences']['success_rate_delta']:.1f}%")
```

## Architecture

### Data Model Hierarchy

```
TestRun (top-level execution)
  ├─ metadata (environment, git info, CI context)
  ├─ timestamp
  └─ TestSuite[] (group by type: unit, integration, benchmark)
       ├─ test_type
       └─ TestCase[] (individual tests)
            ├─ file_path, line_number
            ├─ markers/tags
            └─ TestResult[] (execution history)
                 ├─ status (passed, failed, skipped, error)
                 ├─ duration_ms
                 ├─ error_message, error_traceback
                 └─ PerformanceMetrics (for benchmarks)
                      ├─ min/max/avg/median/p95/p99 (ms)
                      ├─ operations_per_second
                      └─ memory_mb, cpu_percent
```

### Storage Schema

SQLite database with tables:
- `test_runs`: Top-level test executions
- `test_suites`: Test suite groupings
- `test_cases`: Individual test cases
- `test_results`: Actual test results with metrics

Features:
- Foreign key constraints for data integrity
- Cascading deletes
- Indexes for common queries
- JSON columns for flexible metadata

### Parser Architecture

```
BaseParser (abstract)
  ├─ PytestParser
  │    ├─ JUnit XML parser
  │    └─ pytest JSON parser
  ├─ CargoTestParser
  │    ├─ Text output parser
  │    └─ JSON output parser
  └─ BenchmarkJsonParser
       ├─ Custom benchmark parser
       └─ gRPC integration test parser
```

## Advanced Usage

### Custom Parsers

```python
from tests.reporting import TestResultAggregator, TestSource
from tests.reporting.parsers.base import BaseParser
from tests.reporting.models import TestRun

class CustomParser(BaseParser):
    def parse(self, source):
        # Your parsing logic
        test_run = TestRun.create(source=TestSource.CUSTOM)
        # ... populate test_run
        return test_run

# Register custom parser
aggregator = TestResultAggregator()
aggregator.register_parser(TestSource.CUSTOM, CustomParser())

# Use it
run = aggregator.aggregate_from_file('custom.txt', TestSource.CUSTOM)
```

### Direct Storage Access

```python
from tests.reporting import TestResultStorage

storage = TestResultStorage(db_path='custom_results.db')

# List all runs
runs = storage.list_test_runs(limit=10)

# Get specific run
run = storage.get_test_run('run-id')

# Delete old runs
storage.delete_test_run('old-run-id')

# Get statistics
stats = storage.get_statistics()
print(f"Total runs: {stats['total_runs']}")
print(f"Total results: {stats['total_results']}")
print(f"Average success rate: {stats['average_success_rate']:.1f}%")
```

### Test History Tracking

```python
from tests.reporting import TestResultQuery

query = TestResultQuery()

# Get execution history for a specific test
history = query.get_test_history('test_hybrid_search', days=90)

for execution in history:
    print(f"{execution['timestamp']}: {execution['status']} ({execution['duration_ms']:.0f}ms)")
```

## Integration Examples

### Pytest Integration

Generate JUnit XML or JSON:

```bash
# JUnit XML
pytest --junit-xml=results.xml

# JSON (requires pytest-json-report)
pip install pytest-json-report
pytest --json-report --json-report-file=results.json
```

Then aggregate:

```python
from tests.reporting import aggregate_test_results, TestSource

run = aggregate_test_results('results.xml', TestSource.PYTEST)
```

### Cargo Integration

Generate test output:

```bash
# Text output
cargo test > cargo_results.txt 2>&1

# JSON output
cargo test -- --format json > cargo_results.json 2>&1
```

Then aggregate:

```python
from tests.reporting import aggregate_test_results, TestSource

run = aggregate_test_results('cargo_results.txt', TestSource.CARGO)
```

### CI/CD Pipeline Integration

```python
# ci_pipeline.py
from tests.reporting import TestResultAggregator, TestSource
import os

def aggregate_ci_results():
    aggregator = TestResultAggregator()

    # Get run ID from CI environment
    run_id = os.environ.get('CI_PIPELINE_ID', 'local-run')

    # Aggregate all test stages
    run = aggregator.aggregate_multiple([
        {'file': 'unit_tests.xml', 'source': TestSource.PYTEST},
        {'file': 'integration_tests.xml', 'source': TestSource.PYTEST},
        {'file': 'e2e_tests.xml', 'source': TestSource.PYTEST},
        {'file': 'rust_tests.json', 'source': TestSource.CARGO},
        {'file': 'benchmarks.json', 'source': TestSource.BENCHMARK_JSON},
    ], run_id=run_id)

    # Fail CI if success rate is too low
    if run.success_rate < 95.0:
        print(f"FAILURE: Success rate {run.success_rate:.1f}% is below 95%")
        exit(1)

    print(f"SUCCESS: {run.total_tests} tests, {run.success_rate:.1f}% success rate")

if __name__ == '__main__':
    aggregate_ci_results()
```

## Storage Location

Default database location: `tests/reporting/test_results.db`

Customize with:

```python
from tests.reporting import TestResultStorage
from pathlib import Path

storage = TestResultStorage(db_path=Path('/custom/path/results.db'))
```

## Query Performance

The storage backend uses:
- Indexed columns for common queries (timestamp, status, source)
- WAL mode for concurrent read/write access
- Foreign key constraints for data integrity
- Efficient JSON storage for flexible metadata

Typical query times (on macOS M1):
- Retrieve test run: ~0.1ms
- List 100 runs: ~1ms
- Flaky test detection (30 days): ~50ms
- Trend analysis (1000 runs): ~100ms

## API Reference

### TestResultAggregator

Main API for parsing and storing test results.

```python
aggregator = TestResultAggregator(storage=None)

# Aggregate from file
run = aggregator.aggregate_from_file(file_path, source, run_id=None)

# Aggregate from data
run = aggregator.aggregate_from_data(data, source, run_id=None)

# Aggregate multiple sources
run = aggregator.aggregate_multiple(sources, run_id=None)

# Retrieve run
run = aggregator.get_test_run(run_id)

# List runs with filtering
runs = aggregator.list_test_runs(limit=100, offset=0, source=None, start_date=None, end_date=None)

# Delete run
deleted = aggregator.delete_test_run(run_id)

# Get statistics
stats = aggregator.get_statistics()
```

### TestResultQuery

Query API for analysis and reporting.

```python
query = TestResultQuery(storage=None)

# Get latest run
run = query.get_latest_run()

# Get run summary
summary = query.get_run_summary(run_id)

# Get failed tests
failed = query.get_failed_tests(run_id)

# Get slow tests
slow = query.get_slow_tests(run_id, threshold_ms=1000, limit=10)

# Get benchmarks
benchmarks = query.get_benchmark_results(run_id)

# Compare runs
comparison = query.compare_runs(run_id_1, run_id_2)

# Get test history
history = query.get_test_history(test_name, days=30, limit=50)

# Find flaky tests
flaky = query.get_flaky_tests(days=30, min_runs=5, flaky_threshold=0.2)

# Get statistics
stats = query.get_statistics_summary()

# Get trend data
trend = query.get_trend_data(days=30, metric='success_rate')
```

## Future Enhancements

Planned features for future iterations:
- HTML/PDF report generation (Subtask 307.2)
- Coverage tracking integration (Subtask 307.3)
- Failure pattern analysis (Subtask 307.4)
- Web dashboard for visualization
- Integration with observability tests (Task 313)
- Performance regression detection
- Automated flaky test notifications

## Contributing

When adding new parsers:
1. Inherit from `BaseParser`
2. Implement `parse()` method
3. Return `TestRun` object
4. Add tests in `tests/unit/test_result_aggregation.py`

When extending data models:
1. Update `models.py` dataclasses
2. Update `storage.py` schema
3. Add migration if needed
4. Update serialization methods (`to_dict`, `from_dict`)
