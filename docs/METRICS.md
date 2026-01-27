# Metrics Reference

This document describes all metrics collected by workspace-qdrant-mcp for monitoring, alerting, and observability.

## Overview

workspace-qdrant-mcp exposes Prometheus-compatible metrics for monitoring queue health, MCP tool performance, and system resources. Metrics are collected via the `MetricsCollector` class and can be exported in Prometheus exposition format.

## Metric Categories

- [Queue Processor Metrics](#queue-processor-metrics) - Queue depth, throughput, latency
- [Dual-Write Migration Metrics](#dual-write-migration-metrics) - Migration monitoring
- [MCP Tool Metrics](#mcp-tool-metrics) - Tool call performance
- [System Metrics](#system-metrics) - Resource utilization

---

## Queue Processor Metrics

These metrics track the unified queue processor performance and health.

### Counters

#### `queue_items_enqueued_total`
**Type:** Counter
**Labels:** `item_type`, `op`
**Description:** Total number of items added to the queue.

```promql
# Enqueue rate by item type
rate(queue_items_enqueued_total[5m])

# Total files enqueued
sum(queue_items_enqueued_total{item_type="file"})
```

#### `queue_items_processed_total`
**Type:** Counter
**Labels:** `item_type`, `op`, `status`
**Description:** Total items processed. Status is `done` or `failed`.

```promql
# Processing success rate
sum(rate(queue_items_processed_total{status="done"}[5m])) /
sum(rate(queue_items_processed_total[5m]))

# Failure count by item type
sum by (item_type) (queue_items_processed_total{status="failed"})
```

#### `queue_items_retried_total`
**Type:** Counter
**Labels:** `item_type`
**Description:** Total items that were retried after failure.

```promql
# Retry rate over time
rate(queue_items_retried_total[5m])
```

### Gauges

#### `queue_depth_current`
**Type:** Gauge
**Labels:** `status`
**Description:** Current queue depth by status (`pending`, `in_progress`, `done`, `failed`).

```promql
# Pending items
queue_depth_current{status="pending"}

# Total active items
sum(queue_depth_current{status=~"pending|in_progress"})
```

#### `queue_age_oldest_pending_seconds`
**Type:** Gauge
**Labels:** None
**Description:** Age in seconds of the oldest pending item. Zero if no pending items.

```promql
# Alert if oldest item is over 5 minutes
queue_age_oldest_pending_seconds > 300
```

#### `active_projects_count`
**Type:** Gauge
**Labels:** None
**Description:** Number of projects with active queue items.

### Histograms

#### `queue_processing_duration_seconds`
**Type:** Histogram
**Labels:** `item_type`, `op`
**Buckets:** `[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]`
**Description:** Time from enqueue to completion (done/failed).

```promql
# p95 processing duration
histogram_quantile(0.95, rate(queue_processing_duration_seconds_bucket[5m]))

# Average processing time
rate(queue_processing_duration_seconds_sum[5m]) /
rate(queue_processing_duration_seconds_count[5m])
```

#### `queue_wait_duration_seconds`
**Type:** Histogram
**Labels:** `item_type`
**Buckets:** `[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]`
**Description:** Time spent in pending state before processing begins.

```promql
# p99 wait time
histogram_quantile(0.99, rate(queue_wait_duration_seconds_bucket[5m]))
```

---

## Dual-Write Migration Metrics

These metrics track the dual-write migration process during the transition from legacy queues to the unified queue.

### Counters

#### `dual_write_legacy_success_total`
**Type:** Counter
**Labels:** `item_type`, `legacy_queue`
**Description:** Successful writes to legacy queue during dual-write mode.

#### `dual_write_legacy_failure_total`
**Type:** Counter
**Labels:** `item_type`, `legacy_queue`, `error_type`
**Description:** Failed writes to legacy queue during dual-write mode.

```promql
# Dual-write failure rate
sum(rate(dual_write_legacy_failure_total[5m])) /
(sum(rate(dual_write_legacy_success_total[5m])) +
 sum(rate(dual_write_legacy_failure_total[5m])))
```

#### `dual_write_drift_detected_total`
**Type:** Counter
**Labels:** `drift_type`
**Description:** Drift detection events. Types: `missing_in_legacy`, `missing_in_unified`, `status_mismatch`.

### Gauges

#### `dual_write_queue_drift_items`
**Type:** Gauge
**Labels:** `drift_type`
**Description:** Current number of items with drift between unified and legacy queues.

```promql
# Alert on any drift
sum(dual_write_queue_drift_items) > 0
```

---

## MCP Tool Metrics

These metrics track MCP tool call performance.

### Counters

#### `wqm_tool_calls_total`
**Type:** Counter
**Labels:** `tool_name`, `status`
**Description:** Total MCP tool calls. Status: `success`, `error`, `logical_failure`.

```promql
# Tool success rate
sum(rate(wqm_tool_calls_total{status="success"}[5m])) /
sum(rate(wqm_tool_calls_total[5m]))

# Calls by tool
sum by (tool_name) (wqm_tool_calls_total)
```

#### `wqm_search_scope_total`
**Type:** Counter
**Labels:** `scope`
**Description:** Search calls by scope (`project`, `global`, `all`).

### Histograms

#### `wqm_tool_duration_seconds`
**Type:** Histogram
**Labels:** `tool_name`
**Buckets:** `[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]`
**Description:** Duration of MCP tool calls.

```promql
# p95 tool latency
histogram_quantile(0.95, rate(wqm_tool_duration_seconds_bucket[5m]))
```

#### `wqm_search_results`
**Type:** Histogram
**Labels:** `scope`
**Buckets:** `[0, 1, 5, 10, 25, 50, 100, 250, 500, 1000]`
**Description:** Number of search results returned.

#### `wqm_search_latency_seconds`
**Type:** Histogram
**Labels:** `scope`
**Buckets:** `[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]`
**Description:** Search latency by scope.

---

## System Metrics

Standard system resource metrics.

### Gauges

#### `memory_usage_bytes`
**Type:** Gauge
**Description:** Process memory usage in bytes.

#### `cpu_usage_percent`
**Type:** Gauge
**Description:** Process CPU usage percentage.

#### `active_connections`
**Type:** Gauge
**Description:** Number of active file descriptors (connections).

### Watch System

#### `watch_events_total`
**Type:** Counter
**Description:** Total file watch events processed.

#### `watches_active`
**Type:** Gauge
**Description:** Number of active file watches.

#### `watch_errors_total`
**Type:** Counter
**Description:** Total watch system errors.

---

## Prometheus Export

Metrics can be exported in Prometheus exposition format:

```python
from workspace_qdrant_mcp.observability.metrics import get_metrics_collector

collector = get_metrics_collector()
prometheus_text = collector.export_prometheus_format()
```

Example output:

```prometheus
# HELP queue_items_enqueued_total Total items enqueued by item_type and operation
# TYPE queue_items_enqueued_total counter
queue_items_enqueued_total 42
queue_items_enqueued_total{item_type="file", op="ingest"} 35
queue_items_enqueued_total{item_type="content", op="update"} 7

# HELP queue_depth_current Current queue depth by status
# TYPE queue_depth_current gauge
queue_depth_current{status="pending"} 12
queue_depth_current{status="in_progress"} 3
```

---

## Grafana Dashboard

A pre-built Grafana dashboard is available at:

```
assets/grafana/queue_dashboard.json
```

Import this dashboard into Grafana to visualize queue metrics. The dashboard includes:

- Queue overview (pending, in-progress, oldest age, active projects)
- Throughput metrics (enqueue/process rates, retries/failures)
- Latency metrics (processing duration, wait duration percentiles)
- Dual-write migration status

### Dashboard Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DS_PROMETHEUS` | Prometheus datasource | `Prometheus` |

---

## Alerting Rules

Suggested alerting rules for Prometheus Alertmanager:

```yaml
groups:
  - name: wqm-queue-alerts
    rules:
      - alert: QueueBacklogHigh
        expr: queue_depth_current{status="pending"} > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High queue backlog
          description: "{{ $value }} items pending in queue"

      - alert: QueueBacklogCritical
        expr: queue_depth_current{status="pending"} > 1000
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: Critical queue backlog
          description: "{{ $value }} items pending in queue"

      - alert: OldestPendingTooOld
        expr: queue_age_oldest_pending_seconds > 300
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Queue items aging
          description: "Oldest pending item is {{ $value }}s old"

      - alert: HighFailureRate
        expr: >
          sum(rate(queue_items_processed_total{status="failed"}[5m])) /
          sum(rate(queue_items_processed_total[5m])) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: High queue failure rate
          description: "Queue failure rate is {{ $value | humanizePercentage }}"

      - alert: QueueDriftDetected
        expr: sum(dual_write_queue_drift_items) > 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: Queue drift detected
          description: "{{ $value }} items have drift between unified and legacy queues"
```

---

## Python API

### Recording Metrics

```python
from workspace_qdrant_mcp.observability.metrics import (
    record_queue_enqueue,
    record_queue_processed,
    record_wait_duration,
    record_processing_duration,
    set_queue_depth,
    set_oldest_pending_age,
    get_queue_processor_metrics_summary,
)

# Record enqueue
record_queue_enqueue(item_type="file", op="ingest")

# Record completion
record_queue_processed(item_type="file", op="ingest", status="done")

# Record latencies
record_wait_duration(item_type="file", duration_seconds=2.5)
record_processing_duration(item_type="file", op="ingest", duration_seconds=15.3)

# Update gauges
set_queue_depth(status="pending", count=42)
set_oldest_pending_age(age_seconds=120.5)

# Get summary
summary = get_queue_processor_metrics_summary()
```

### Tool Metrics Decorator

```python
from workspace_qdrant_mcp.observability.metrics import track_tool

@track_tool("search")
async def search(query: str, ...) -> dict:
    # Automatically tracks:
    # - wqm_tool_calls_total{tool_name="search", status="success|error|logical_failure"}
    # - wqm_tool_duration_seconds{tool_name="search"}
    ...
```

---

## Best Practices

1. **Use rate() for counters**: Always use `rate()` or `increase()` when querying counters to get meaningful values.

2. **Choose appropriate time windows**: Use 5m for normal monitoring, 1h for trends, 15m for alerts.

3. **Monitor queue depth trends**: A steadily increasing `queue_depth_current{status="pending"}` indicates processing cannot keep up with ingest.

4. **Track wait duration percentiles**: High p95/p99 wait times indicate bottlenecks even if average is acceptable.

5. **Set up drift alerts during migration**: Queue drift should be zero after dual-write stabilization.

6. **Review Grafana dashboard regularly**: The pre-built dashboard provides at-a-glance health monitoring.
