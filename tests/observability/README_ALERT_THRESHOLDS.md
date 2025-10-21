# Alert Threshold Testing Framework

Comprehensive testing suite for alert threshold functionality in workspace-qdrant-mcp's observability infrastructure.

## Overview

The alert threshold testing framework validates all aspects of the alerting system including:

- **Threshold Comparison Logic**: All operators (>, <, ==, >=, <=, !=)
- **Performance Alerts**: Latency, throughput, processing time percentiles
- **Resource Alerts**: Queue size, memory, CPU, disk usage, backpressure
- **Error Rate Alerts**: Failure rates, success rates
- **Health Degradation Alerts**: Health score thresholds, status changes
- **Queue Depth Alerts**: Backlog growth, critical depth warnings
- **Alert Lifecycle**: Activation, persistence, deactivation, cooldown
- **Alert Severity Levels**: INFO, WARNING, ERROR, CRITICAL
- **Composite Alerts**: AND/OR logic with multiple conditions
- **Windowed Alerts**: Sustained threshold violations over time
- **Rate of Change Alerts**: Rapid metric increases/decreases
- **Metrics Integration**: Cross-source correlation and validation

## Running Tests

### Run All Alert Threshold Tests

```bash
cd /Users/chris/dev/ai/claude-code-cfg/mcp/workspace-qdrant-mcp
uv run pytest tests/observability/test_alert_thresholds.py -v
```

### Run Specific Test Classes

```bash
# Test threshold comparison logic
uv run pytest tests/observability/test_alert_thresholds.py::TestThresholdComparison -v

# Test performance alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestPerformanceAlerts -v

# Test resource alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestResourceAlerts -v

# Test error rate alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestErrorRateAlerts -v

# Test health degradation
uv run pytest tests/observability/test_alert_thresholds.py::TestHealthDegradationAlerts -v

# Test queue depth alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestQueueDepthAlerts -v

# Test alert lifecycle
uv run pytest tests/observability/test_alert_thresholds.py::TestAlertLifecycle -v

# Test severity levels
uv run pytest tests/observability/test_alert_thresholds.py::TestAlertSeverityLevels -v

# Test composite alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestCompositeAlerts -v

# Test windowed alerts
uv run pytest tests/observability/test_alert_thresholds.py::TestWindowedAlerts -v

# Test rate of change
uv run pytest tests/observability/test_alert_thresholds.py::TestRateOfChangeAlerts -v

# Test metrics integration
uv run pytest tests/observability/test_alert_thresholds.py::TestMetricsIntegration -v

# Test edge cases
uv run pytest tests/observability/test_alert_thresholds.py::TestEdgeCases -v
```

### Run Individual Tests

```bash
# Test specific functionality
uv run pytest tests/observability/test_alert_thresholds.py::TestThresholdComparison::test_greater_than_operator_triggered -v
uv run pytest tests/observability/test_alert_thresholds.py::TestAlertLifecycle::test_alert_cooldown_prevents_spam -v
```

### Run with Coverage

```bash
uv run pytest tests/observability/test_alert_thresholds.py \
    --cov=src/python/common/core/queue_alerting \
    --cov=src/python/common/core/queue_health \
    --cov=src/python/common/core/queue_performance_metrics \
    --cov=src/python/common/core/queue_backpressure \
    --cov-report=html
```

## Test Categories

### 1. Threshold Comparison Logic (10 tests)

Tests validating threshold comparison operators:

- `test_greater_than_operator_triggered`: Validates > operator
- `test_greater_than_operator_not_triggered`: Validates > operator boundary
- `test_less_than_operator_triggered`: Validates < operator
- `test_less_than_operator_not_triggered`: Validates < operator boundary
- `test_equals_operator_exact_match`: Validates == with exact match
- `test_equals_operator_within_tolerance`: Validates == with float tolerance
- `test_equals_operator_outside_tolerance`: Validates == rejection
- `test_greater_equals_operator`: Validates >= operator
- `test_less_equals_operator`: Validates <= operator
- `test_boundary_value_exactly_at_threshold`: Tests all operators at boundary

**What it validates**: Core threshold evaluation logic works correctly for all comparison operators.

### 2. Performance Alerts (3 tests)

Tests for performance-related alert types:

- `test_high_latency_alert`: Alert when latency exceeds threshold
- `test_low_throughput_alert`: Alert when throughput drops
- `test_p95_latency_alert`: Alert on P95 latency threshold

**What it validates**: Performance metric alerts trigger correctly.

**Metrics used**:
- `latency_avg_ms` from `QueuePerformanceCollector.get_latency_metrics()`
- `throughput_items_per_second` from `QueuePerformanceCollector.get_throughput_metrics()`
- `processing_time_p95` from `QueuePerformanceCollector.get_processing_time_stats()`

### 3. Resource Alerts (3 tests)

Tests for resource usage alerts:

- `test_queue_size_alert`: Alert when queue size exceeds threshold
- `test_backpressure_capacity_alert`: Alert when capacity usage is high
- `test_multiple_resource_thresholds`: Alert with AND logic for multiple resources

**What it validates**: Resource utilization alerts work correctly.

**Metrics used**:
- `queue_size` from `QueueStatisticsCollector.get_current_statistics()`
- `backpressure_capacity_used` from `BackpressureDetector.get_backpressure_indicators()`

### 4. Error Rate Alerts (2 tests)

Tests for error rate monitoring:

- `test_high_error_rate_alert`: Alert when error rate exceeds threshold
- `test_low_success_rate_alert`: Alert when success rate drops

**What it validates**: Error rate thresholds trigger appropriately.

**Metrics used**:
- `error_rate` (maps to `failure_rate` from `QueueStatisticsCollector`)
- `success_rate` from `QueueStatisticsCollector.get_current_statistics()`

### 5. Health Degradation Alerts (2 tests)

Tests for system health monitoring:

- `test_low_health_score_alert`: Alert when health score drops
- `test_health_status_degraded_alert`: Alert when status becomes degraded

**What it validates**: Health degradation detection works correctly.

**Metrics used**:
- `health_score` from `QueueHealthCalculator.calculate_health()`
- `health_status_value` (computed from `HealthStatus` enum)

### 6. Queue Depth Alerts (2 tests)

Tests for queue backlog monitoring:

- `test_queue_backlog_growth_alert`: Alert when queue is growing
- `test_critical_queue_depth_alert`: Critical alert for extreme queue depth

**What it validates**: Queue backlog alerts trigger at appropriate levels.

**Metrics used**:
- `backpressure_growth_rate` from `BackpressureDetector`
- `queue_size` from `QueueStatisticsCollector`

### 7. Alert Lifecycle (5 tests)

Tests for alert state management:

- `test_alert_activation`: Alert activates when threshold crossed
- `test_alert_persistence`: Alert persists during sustained violation
- `test_alert_deactivation_via_acknowledgment`: Alert deactivates when acknowledged
- `test_alert_cooldown_prevents_spam`: Cooldown prevents alert spam
- `test_alert_cooldown_expiry`: Alert retriggering after cooldown

**What it validates**: Alert lifecycle management works correctly including cooldown logic.

### 8. Alert Severity Levels (5 tests)

Tests for severity classification:

- `test_info_severity_alert`: INFO level alerts
- `test_warning_severity_alert`: WARNING level alerts
- `test_error_severity_alert`: ERROR level alerts
- `test_critical_severity_alert`: CRITICAL level alerts
- `test_severity_prioritization_in_composite_alerts`: Highest severity used in composite alerts

**What it validates**: Alert severity levels are correctly assigned and prioritized.

### 9. Composite Alerts (5 tests)

Tests for multi-condition alerts:

- `test_and_logic_all_conditions_met`: AND logic with all conditions met
- `test_and_logic_partial_conditions`: AND logic with partial conditions
- `test_or_logic_one_condition_met`: OR logic with one condition
- `test_or_logic_no_conditions_met`: OR logic with no conditions
- `test_complex_composite_alert`: Complex alert with 3+ conditions

**What it validates**: Composite alert logic (AND/OR) works correctly.

### 10. Windowed Alerts (2 tests)

Tests for sustained violations:

- `test_sustained_violation_detection`: Detection of sustained violations
- `test_intermittent_violation_pattern`: Handling intermittent violations

**What it validates**: Alert system can detect sustained vs. transient issues.

### 11. Rate of Change Alerts (2 tests)

Tests for rapid metric changes:

- `test_rapid_queue_growth_detection`: Detection of rapid queue growth
- `test_rapid_throughput_decrease_detection`: Detection of throughput drops

**What it validates**: Alerts trigger on rate of change, not just absolute values.

### 12. Metrics Integration (3 tests)

Tests for cross-source metric collection:

- `test_all_metric_sources_available`: All metric sources integrated
- `test_metric_collection_failure_handling`: Graceful handling of failures
- `test_cross_metric_correlation_alert`: Alerts using metrics from multiple sources

**What it validates**: Integration with metrics from subtask 313.1 works correctly.

**Metric sources validated**:
- `QueueStatisticsCollector` - Queue stats (size, rates, errors)
- `QueuePerformanceCollector` - Performance metrics (latency, throughput)
- `QueueHealthCalculator` - Health assessment (score, status)
- `BackpressureDetector` - Backpressure indicators (growth rate, capacity)

### 13. Edge Cases (5 tests)

Tests for error handling and robustness:

- `test_missing_metric_handling`: Handling of missing metrics
- `test_null_metric_value_handling`: Handling of null values
- `test_disabled_threshold_ignored`: Disabled thresholds are skipped
- `test_alert_with_empty_threshold_list`: Empty threshold list doesn't crash
- `test_concurrent_alert_evaluations`: Concurrent evaluations don't interfere

**What it validates**: Alert system is robust to edge cases and errors.

## Test Fixtures

### Core Fixtures

- `temp_db`: Temporary database for isolated testing
- `alert_system`: Initialized `QueueAlertingSystem` with temp database
- `configured_alert_system`: Alert system with all metric collectors configured

### Metric Collector Mocks

- `mock_stats_collector`: Mocked `QueueStatisticsCollector` with sample data
- `mock_performance_collector`: Mocked `QueuePerformanceCollector` with latency/throughput
- `mock_health_calculator`: Mocked `QueueHealthCalculator` with health scores
- `mock_backpressure_detector`: Mocked `BackpressureDetector` with backpressure indicators

## Interpreting Test Results

### Successful Test Run

```
tests/observability/test_alert_thresholds.py::TestThresholdComparison::test_greater_than_operator_triggered PASSED [ 2%]
tests/observability/test_alert_thresholds.py::TestPerformanceAlerts::test_high_latency_alert PASSED [ 5%]
...
======================== 49 passed in 3.45s =========================
```

### Test Failures

Common failure scenarios and solutions:

**Issue**: Alert not triggered when expected
**Cause**: Metric value doesn't meet threshold or metrics not properly mocked
**Solution**: Verify mock data values and threshold comparisons

**Issue**: Cooldown test fails
**Cause**: Timing issue or concurrent test execution
**Solution**: Ensure cooldown durations are appropriate for testing environment

**Issue**: Metrics not available
**Cause**: Collector mock not properly configured
**Solution**: Check that all collector mocks return proper data structures

**Issue**: SQLite thread safety errors
**Cause**: Concurrent database access from multiple threads
**Solution**: Use separate database instances per test or serialize tests

## Alert Metrics Reference

### Available Metrics

From `QueueStatisticsCollector`:
- `queue_size`: Current queue size (integer)
- `processing_rate`: Items processed per minute (float)
- `error_rate`: Failure rate as decimal (0.02 = 2%)
- `success_rate`: Success rate as decimal (0.98 = 98%)

From `QueuePerformanceCollector`:
- `throughput_items_per_second`: Throughput rate (float)
- `throughput_items_per_minute`: Throughput per minute (float)
- `latency_avg_ms`: Average latency in ms (float)
- `latency_max_ms`: Maximum latency in ms (float)
- `processing_time_p95`: P95 processing time (float)
- `processing_time_p99`: P99 processing time (float)

From `QueueHealthCalculator`:
- `health_score`: Health score 0-100 (float)
- `health_status_value`: Numeric value for status (0=CRITICAL, 40=UNHEALTHY, 70=DEGRADED, 100=HEALTHY)

From `BackpressureDetector`:
- `backpressure_growth_rate`: Queue growth rate (float)
- `backpressure_capacity_used`: Capacity utilization 0-1 (float)
- `backpressure_severity_value`: Numeric severity (0=NONE, 25=LOW, 50=MEDIUM, 75=HIGH, 100=CRITICAL)

## Performance Benchmarks

Expected performance characteristics:

- **Threshold Evaluation**: < 1ms per threshold
- **Rule Evaluation**: < 50ms for 10 rules
- **Metrics Collection**: < 100ms for all sources
- **Alert Creation**: < 10ms per alert
- **Database Operations**: < 20ms per operation

## Adding New Tests

To add a new alert threshold test:

1. Create test method in appropriate class
2. Use `configured_alert_system` fixture for full metrics integration
3. Create `AlertRule` with desired thresholds
4. Use `cooldown_minutes=0` for testing to avoid delays
5. Call `evaluate_rules()` and assert on results
6. Document expected behavior in docstring

Example:

```python
@pytest.mark.asyncio
async def test_custom_metric_alert(self, configured_alert_system):
    """Test alert on custom metric threshold."""
    rule = AlertRule(
        rule_name="custom_test",
        description="Test custom metric",
        thresholds=[
            AlertThreshold(
                metric_name="custom_metric",
                operator=">",
                value=100.0,
                severity=AlertSeverity.WARNING,
            )
        ],
        recipients=["log"],
        cooldown_minutes=0,
    )

    await configured_alert_system.create_alert_rule(rule)
    alerts = await configured_alert_system.evaluate_rules()

    assert len(alerts) == 1
    assert alerts[0].severity == AlertSeverity.WARNING
```

## Integration with Metrics Collection

This test suite validates integration with metrics collection from **Subtask 313.1**:

- All metric collectors are mocked but use real data structures
- Tests validate that alerting system correctly fetches metrics from all sources
- Cross-source correlation is tested (alerts using metrics from multiple collectors)
- Failure handling ensures graceful degradation when collectors are unavailable

## Maintenance

### Updating Tests

When alert infrastructure changes:

1. Update affected test cases
2. Update mock data structures to match new metric formats
3. Verify all test categories still pass
4. Add new tests for new features
5. Update this README with new information

### Common Maintenance Tasks

**Adding new metric source**:
1. Create new fixture mocking the collector
2. Add to `configured_alert_system` fixture
3. Add tests validating the new metrics
4. Document metrics in "Alert Metrics Reference"

**Adding new alert type**:
1. Create new test class following existing patterns
2. Add tests for all operators and edge cases
3. Document in "Test Categories" section
4. Add to "Running Tests" examples

## References

- **Alert implementation**: `src/python/common/core/queue_alerting.py`
- **Queue statistics**: `src/python/common/core/queue_statistics.py`
- **Performance metrics**: `src/python/common/core/queue_performance_metrics.py`
- **Health calculator**: `src/python/common/core/queue_health.py`
- **Backpressure detector**: `src/python/common/core/queue_backpressure.py`
- **Observability README**: `tests/observability/README.md`

## Support

For issues or questions about alert threshold testing:

1. Check test output for detailed error messages
2. Review this README for common issues
3. Examine alert database in temp directory during test failures
4. Verify mock data matches expected metric structures
5. Check that all collectors are properly initialized

---

**Test Suite Status**: ✓ Comprehensive framework created covering all alert types and scenarios
**Integration Status**: ✓ Validates integration with metrics from subtask 313.1
**Documentation Status**: ✓ Complete with examples and troubleshooting guide
