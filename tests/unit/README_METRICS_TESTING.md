# Metrics Collection Validation Tests

Comprehensive test suite for validating the accuracy and reliability of the metrics collection system in workspace-qdrant-mcp.

## Overview

The metrics collection validation tests ensure that all metric types (counters, gauges, histograms) accurately track application behavior and can be reliably exported in industry-standard formats (Prometheus, JSON).

## Test Coverage

### 1. Counter Metrics (`TestCounterMetrics`)

Tests for monotonically increasing counters:

- **Basic increment**: Single and multiple increments with custom amounts
- **Increment accuracy**: Validates 100+ operations with integer and floating point values
- **Labeled counters**: Multi-dimensional label tracking (method, status, etc.)
- **Label serialization**: Consistent key generation regardless of label order
- **Thread safety**: Concurrent increments from multiple threads
- **Labeled thread safety**: Concurrent updates to different label combinations

**Example validation**:
```python
counter.increment(5.0, method="GET", status="200")
assert counter.get_value(method="GET", status="200") == 5.0
```

### 2. Gauge Metrics (`TestGaugeMetrics`)

Tests for current value tracking:

- **Basic operations**: Set, add, subtract operations
- **Float accuracy**: Validates precision with 100+ small increments
- **Labeled gauges**: Independent tracking per label combination
- **Labeled operations**: Add/subtract on specific labeled values
- **Thread safety**: Concurrent gauge modifications
- **Negative values**: Support for gauges that can go negative

**Example validation**:
```python
gauge.set(100, process="api")
gauge.add(50, process="api")
assert gauge.get_value(process="api") == 150
```

### 3. Histogram Metrics (`TestHistogramMetrics`)

Tests for distribution tracking:

- **Basic observation**: Count, sum, average calculations
- **Bucket distribution**: Correct cumulative bucket counts
- **Custom buckets**: User-defined bucket boundaries
- **Labeled histograms**: Separate distributions per label
- **Percentile calculation**: Bucket-based percentile estimation
- **Thread safety**: Concurrent observations
- **Edge values**: Zero, very small, and very large values

**Example validation**:
```python
for value in [0.1, 0.5, 1.0, 2.0]:
    histogram.observe(value)
assert histogram.get_count() == 4
assert histogram.get_average() == 0.9
```

### 4. MetricsCollector (`TestMetricsCollector`)

Tests for central metrics management:

- **Initialization**: Standard metrics created on startup
- **Metric creation**: Counter, gauge, histogram creation and retrieval
- **Convenience methods**: Simplified increment/set/record APIs
- **Timer context manager**: Automatic duration measurement
- **Operation tracking**: Start/complete operation lifecycle
- **Error tracking**: Failed operation metrics
- **System metrics**: Memory, CPU, connection tracking

**Example validation**:
```python
with collector.timer("operation"):
    # Perform operation
    time.sleep(0.1)
histogram = collector.histograms["operation"]
assert 0.09 < histogram.get_average() < 0.15
```

### 5. Prometheus Export (`TestPrometheusExport`)

Tests for Prometheus format compliance:

- **Format structure**: HELP, TYPE directives, metric lines
- **Counter format**: Proper counter type declaration and labels
- **Histogram format**: Bucket, sum, count lines with le labels
- **Gauge format**: Gauge type declaration and values

**Example validation**:
```python
output = collector.export_prometheus_format()
assert "# TYPE http_requests_total counter" in output
assert 'http_requests_total{method="GET"} 10' in output
```

### 6. JSON Export (`TestJSONExport`)

Tests for JSON format export:

- **Structure validation**: Timestamp, counters, gauges, histograms sections
- **Counter export**: Value, labeled_values, description fields
- **Histogram export**: Count, sum, average, buckets data

**Example validation**:
```python
summary = collector.get_metrics_summary()
assert "counters" in summary
assert "gauges" in summary
assert "histograms" in summary
```

### 7. Load Conditions (`TestLoadConditions`)

Tests under high-load scenarios:

- **High-frequency counters**: 10,000+ rapid increments
- **High-frequency histograms**: 10,000+ rapid observations
- **Concurrent metric types**: Multiple metric types updated simultaneously
- **Many labeled metrics**: 100+ unique label combinations
- **Export performance**: Large metric set export timing

**Example validation**:
```python
for i in range(10000):
    collector.increment_counter("high_freq")
assert collector.counters["high_freq"].get_value() == 10000
```

### 8. Labels and Dimensions (`TestMetricLabelsAndDimensions`)

Tests for label functionality:

- **Label types**: String, integer, float, boolean values
- **Special characters**: Dashes, underscores, dots in labels
- **Multi-dimensional**: Multiple label keys per metric
- **High cardinality**: 1000+ unique label combinations

**Example validation**:
```python
histogram.observe(0.1, method="GET", endpoint="/api", status="200")
assert histogram.get_count(method="GET", endpoint="/api", status="200") == 1
```

### 9. Accuracy Validation (`TestMetricsAccuracyValidation`)

Tests for numerical accuracy:

- **Large values**: Counter accuracy with values > 1,000,000
- **Small increments**: Gauge precision with 0.001 increments
- **Histogram sum**: Validates sum calculation accuracy
- **Histogram average**: Validates average calculation
- **Zero values**: Proper handling of zero increments/observations

**Example validation**:
```python
values = [0.1, 0.2, 0.3, 0.4, 0.5]
for v in values:
    histogram.observe(v)
expected_sum = sum(values)  # 1.5
assert abs(histogram.get_sum() - expected_sum) < 0.001
```

### 10. Singleton Pattern (`TestMetricsCollectorSingleton`)

Tests for singleton instance management:

- **Same instance**: Multiple calls return same collector
- **Cross-module**: Singleton works across different imports

## Running the Tests

### Run all metrics tests:
```bash
pytest tests/unit/test_metrics_collection.py -v
```

### Run specific test class:
```bash
pytest tests/unit/test_metrics_collection.py::TestCounterMetrics -v
pytest tests/unit/test_metrics_collection.py::TestHistogramMetrics -v
pytest tests/unit/test_metrics_collection.py::TestLoadConditions -v
```

### Run specific test:
```bash
pytest tests/unit/test_metrics_collection.py::TestCounterMetrics::test_counter_thread_safety -v
```

### Run with timing information:
```bash
pytest tests/unit/test_metrics_collection.py -v --durations=10
```

### Run with coverage:
```bash
pytest tests/unit/test_metrics_collection.py --cov=common.observability.metrics --cov-report=html
```

## Interpreting Results

### Successful Test Run

```
============================= test session starts ==============================
...
tests/unit/test_metrics_collection.py::TestCounterMetrics::test_counter_basic_increment PASSED [  1%]
tests/unit/test_metrics_collection.py::TestCounterMetrics::test_counter_thread_safety PASSED [  9%]
...
=================== 53 passed, 1 skipped, 1 warning in 0.49s ===================
```

**What this means**:
- ✅ All metric types working correctly
- ✅ Thread safety verified
- ✅ Export formats validated
- ✅ Performance acceptable (< 0.5s for full suite)

### Test Failures

If tests fail, check:

1. **Floating point precision**: Use approximate comparisons (`abs(a - b) < 0.001`) instead of exact equality
2. **Thread safety**: Ensure locks are properly acquired/released
3. **Label serialization**: Verify label order doesn't affect key generation
4. **Bucket boundaries**: Histogram buckets must be cumulative counts

### Performance Expectations

| Test Type | Expected Duration | Warning Threshold |
|-----------|------------------|-------------------|
| Counter tests | < 0.01s | > 0.1s |
| Gauge tests | < 0.01s | > 0.1s |
| Histogram tests | < 0.05s | > 0.2s |
| Thread safety tests | < 0.05s | > 0.5s |
| High-frequency tests | < 0.1s | > 1.0s |
| Export tests | < 0.05s | > 0.5s |
| Full suite | < 0.5s | > 2.0s |

## Known Issues

### Skipped Test: test_metrics_reset

**Issue**: Deadlock in `MetricsCollector.reset_metrics()`

**Root cause**: The method holds `self._lock` while calling `_initialize_standard_metrics()`, which calls `create_counter/create_gauge/create_histogram`, each of which tries to acquire `self._lock` again.

**Impact**: Metrics cannot be safely reset during runtime.

**Workaround**: Create new `MetricsCollector` instance instead of resetting.

**Fix required in core code**:
```python
def reset_metrics(self):
    # Option 1: Don't hold lock during initialization
    with self._lock:
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
    # Now safe to call without deadlock
    self._initialize_standard_metrics()

    # OR Option 2: Use internal methods that don't acquire locks
```

## Best Practices for Metrics Testing

### 1. Use Approximate Comparisons for Floats

```python
# ❌ Brittle
assert histogram.get_sum() == 0.3

# ✅ Robust
assert abs(histogram.get_sum() - 0.3) < 0.001
```

### 2. Test Thread Safety with Realistic Loads

```python
num_threads = 10
operations_per_thread = 1000

def worker():
    for _ in range(operations_per_thread):
        counter.increment()

threads = [Thread(target=worker) for _ in range(num_threads)]
# Start all, wait for completion
```

### 3. Verify Both Labeled and Unlabeled Metrics

```python
counter.increment(10)  # Unlabeled
counter.increment(5, label="test")  # Labeled

assert counter.get_value() == 10  # Only unlabeled
assert counter.get_value(label="test") == 5  # Only this label
```

### 4. Test Export Formats with Real Data

```python
# Generate realistic metrics
collector.increment_counter("requests_total", method="GET", status="200")
collector.record_histogram("latency", 0.123)

# Verify format compliance
prometheus = collector.export_prometheus_format()
assert "# TYPE requests_total counter" in prometheus
assert "_bucket{le=" in prometheus  # Histogram buckets
```

### 5. Validate Under Various Load Patterns

```python
# Burst traffic
for i in range(1000):
    collector.increment_counter("burst")

# Sustained load
for i in range(10000):
    if i % 100 == 0:
        time.sleep(0.001)
    collector.increment_counter("sustained")

# Concurrent access
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(operation) for _ in range(100)]
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Run metrics collection tests
  run: |
    pytest tests/unit/test_metrics_collection.py -v \
      --cov=common.observability.metrics \
      --cov-report=xml \
      --junitxml=test-results/metrics.xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: coverage.xml
    flags: metrics
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running metrics collection tests..."
pytest tests/unit/test_metrics_collection.py -q

if [ $? -ne 0 ]; then
    echo "❌ Metrics tests failed. Commit aborted."
    exit 1
fi

echo "✅ Metrics tests passed"
```

## Troubleshooting

### Tests Timeout

**Symptom**: Tests hang or timeout

**Possible causes**:
- Deadlock in metrics collection code
- Infinite loop in metric calculation
- Thread safety issue with locks

**Debug steps**:
```bash
# Run with timeout and verbose output
pytest tests/unit/test_metrics_collection.py -v --timeout=60 --timeout-method=thread

# Check for deadlocks
pytest tests/unit/test_metrics_collection.py --timeout=30 --tb=long
```

### Inconsistent Results

**Symptom**: Tests pass/fail randomly

**Possible causes**:
- Race conditions in thread safety tests
- Shared state between tests
- System resource constraints

**Debug steps**:
```bash
# Run test multiple times
for i in {1..10}; do pytest tests/unit/test_metrics_collection.py::TestCounterMetrics::test_counter_thread_safety; done

# Run with xdist for isolation
pytest tests/unit/test_metrics_collection.py -n auto
```

### Performance Degradation

**Symptom**: Tests take longer than expected

**Possible causes**:
- System under load
- Inefficient metric collection
- Too many metrics created

**Debug steps**:
```bash
# Profile test execution
pytest tests/unit/test_metrics_collection.py --durations=20 --profile

# Check system resources
top -pid $(pgrep pytest)
```

## Related Documentation

- [Metrics Collection System](../../src/python/common/observability/metrics.py)
- [Monitoring Integration Tests](../test_monitoring_integration.py)
- [Prometheus Documentation](https://prometheus.io/docs/concepts/metric_types/)
- [Production Deployment Guide](../../README.md#monitoring)

## Contributing

When adding new metric types or modifying metrics collection:

1. Add corresponding validation tests
2. Test thread safety if applicable
3. Validate export format compliance
4. Test under high load (10k+ operations)
5. Verify floating point precision
6. Document any known limitations

## Support

For issues with metrics testing:

1. Check this documentation first
2. Review existing test failures for similar patterns
3. Check metrics collection source code for recent changes
4. File issue with:
   - Test name and failure message
   - Expected vs actual behavior
   - System information (OS, Python version)
   - Steps to reproduce
