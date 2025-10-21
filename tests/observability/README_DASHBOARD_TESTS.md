# Dashboard Functionality Validation Tests

This directory contains comprehensive validation tests for dashboard data API endpoints, completing the observability test suite (Subtask 313.5).

## Overview

The dashboard validation tests ensure that all observability data from previous subtasks (313.1-313.4) can be properly aggregated and served for dashboard visualization.

## Test Coverage

### 1. Dashboard API Endpoints (`TestDashboardAPIEndpoints`)
- Real-time metrics data retrieval
- Historical metrics queries
- Aggregation intervals (1m, 5m, 15m, 1h)
- Time range filtering (1h, 6h, 24h, 7d)

### 2. Data Accuracy Validation (`TestDashboardDataAccuracy`)
- Metrics match performance collector (313.1)
- Health status matches health checker (313.3)
- Alerts match alert system (313.4)
- Queue stats match statistics collector (313.1)

### 3. Time-Series Queries (`TestDashboardTimeSeriesQueries`)
- Time-series data retrieval for specific metrics
- Different aggregation intervals
- Custom time ranges
- Chronological data ordering

### 4. Query Parameters (`TestDashboardQueryParameters`)
- Component filtering
- Pagination for large datasets
- Logical metric grouping

### 5. Performance Testing (`TestDashboardPerformance`)
- Query response times (< 500ms)
- Time-series query performance (< 200ms)
- Concurrent query handling
- Data refresh rates

### 6. Cross-Component Integration (`TestCrossComponentIntegration`)
- Metrics and alerts correlation
- Health status reflected in metrics
- Alert severity matching metric deviation
- All data sources accessible from single endpoint
- Timestamp consistency
- Graceful handling of missing data

## Running the Tests

```bash
# Run all dashboard validation tests
uv run pytest tests/observability/test_dashboard_functionality.py -v

# Run specific test class
uv run pytest tests/observability/test_dashboard_functionality.py::TestDashboardAPIEndpoints -v

# Run with coverage report
uv run pytest tests/observability/test_dashboard_functionality.py \
    --cov=src/python/common/dashboard \
    --cov=src/python/common/observability \
    --cov-report=html

# Run performance tests only
uv run pytest tests/observability/test_dashboard_functionality.py::TestDashboardPerformance -v
```

## Integration Points

This test suite integrates with all previous observability subtasks:

- **Subtask 313.1**: Queue performance metrics, throughput, latency
  - `QueuePerformanceCollector`
  - `ThroughputMetrics`
  - `LatencyMetrics`
  - `MetricsAggregator`

- **Subtask 313.2**: Log aggregation and structured logging
  - `LogContext`
  - Structured JSON logging

- **Subtask 313.3**: Health checks and monitoring
  - `HealthChecker`
  - `HealthCoordinator`
  - `ComponentHealth`

- **Subtask 313.4**: Alert thresholds and notifications
  - `QueueAlertingSystem`
  - `AlertRule`
  - `AlertNotification`

## Test Architecture

The test suite uses a `DashboardDataAggregator` fixture that combines data from all observability components:

```python
async def get_dashboard_data(
    time_range: str = "1h",
    aggregation_interval: str = "5m",
    filter_components: List[str] = None,
) -> Dict[str, Any]:
    """Get aggregated dashboard data from all sources."""
    # Returns comprehensive dashboard data including:
    # - Metrics (throughput, latency, processing time, queue stats)
    # - Health status (overall status, component health)
    # - Alerts (active alerts with severity and details)
```

## Dashboard Data Structure

The dashboard API returns data in this format:

```python
{
    "timestamp": "2025-10-21T11:00:00+00:00",
    "time_range": "1h",
    "aggregation_interval": "5m",
    "metrics": {
        "throughput": {
            "items_per_second": 45.8,
            "items_per_minute": 2748.0,
            "total_items": 5000,
            "window_seconds": 300
        },
        "latency": {
            "min_ms": 8.2,
            "max_ms": 450.3,
            "avg_ms": 95.7,
            "total_items": 5000
        },
        "processing_time": {
            "min": 8.2,
            "max": 450.3,
            "avg": 95.7,
            "p50": 82.1,
            "p95": 215.4,
            "p99": 389.2,
            "count": 5000
        },
        "queue": {
            "size": 850,
            "processing_rate": 45.0,
            "success_rate": 0.982,
            "failure_rate": 0.018,
            "error_count": 45
        }
    },
    "health": {
        "overall_status": "healthy",
        "components": {
            "system_resources": {...},
            "qdrant_connectivity": {...},
            "embedding_service": {...}
        }
    },
    "alerts": {
        "active_count": 2,
        "alerts": [
            {
                "alert_id": "alert_001",
                "severity": "WARNING",
                "metric_name": "queue_size",
                "metric_value": 850.0,
                "message": "Queue size exceeds threshold",
                "timestamp": "2025-10-21T11:00:00+00:00"
            }
        ]
    }
}
```

## Performance Benchmarks

The dashboard validation tests enforce these performance requirements:

- **Dashboard data query**: < 500ms response time
- **Time-series query**: < 200ms response time
- **Concurrent queries**: 5 concurrent queries in < 1000ms
- **Data refresh**: Updates available within 5 seconds

## Notes

- These tests focus on API endpoints that serve dashboard data
- No web UI browser automation (tests are API-level only)
- Tests use mock data aggregation to validate integration patterns
- Production dashboard implementations should implement error handling for graceful degradation when components are unavailable

## Future Enhancements

For production dashboard implementations:

1. Add error handling with try-except blocks around collector calls
2. Return partial data with error indicators when components fail
3. Implement caching for frequently accessed dashboard data
4. Add WebSocket support for real-time updates
5. Implement dashboard data export capabilities
