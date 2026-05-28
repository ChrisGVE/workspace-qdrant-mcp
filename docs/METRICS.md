# Metrics Reference

This document describes all metrics collected by workspace-qdrant-mcp for monitoring, alerting, and observability.

## Overview

workspace-qdrant-mcp exposes Prometheus-compatible metrics for monitoring queue health, MCP tool performance, and system resources. The Rust daemon (`memexd`) collects them and serves them in Prometheus exposition format on its metrics endpoint (default port 9091).

## Metric Categories

- [Queue Processor Metrics](#queue-processor-metrics) - Queue depth, throughput, latency
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

#### `memexd_unified_queue_depth_by_tenant`
**Type:** Gauge
**Labels:** `tenant_id`, `status` (`pending`, `in_progress`, `failed` — `done` is excluded because those rows are deleted by `cleanup_completed_unified_items`)
**Description:** Per-tenant unified-queue depth. Drives the Grafana
"Indexing progress (per tenant)" panel set and gives operators a way to
spot a single project that's lagging behind the others. Refreshed every
10 s by the queue-depth exporter, alongside the global
`memexd_unified_queue_depth`.

```promql
# Total still-pending items across all tenants
sum(memexd_unified_queue_depth_by_tenant{status=~"pending|in_progress"})

# Top 10 tenants by pending depth
topk(10, sum by (tenant_id) (memexd_unified_queue_depth_by_tenant{status="pending"}))

# Alert when one tenant has more than 5 000 pending for >10 min
max by (tenant_id) (memexd_unified_queue_depth_by_tenant{status="pending"}) > 5000
```

#### `memexd_indexing_eta_seconds_by_tenant`
**Type:** Gauge
**Labels:** `tenant_id`
**Description:** Estimated seconds until each tenant's queue is fully
drained. Derived from the rate at which `tracked_files.updated_at`
advances over a 5-minute window, capped at 24 h. Set to `-1` when the
daemon can't estimate (cold-start, zero throughput with pending > 0,
or queue already drained) — Prometheus has no native null. Filter the
sentinel out in PromQL with `>= 0`.

```promql
# Only series with a real estimate
memexd_indexing_eta_seconds_by_tenant >= 0

# Tenants in "warming up" state right now
count(memexd_indexing_eta_seconds_by_tenant == -1)

# Alert when one tenant's ETA exceeds 1h for >10 minutes
max by (tenant_id) (memexd_indexing_eta_seconds_by_tenant >= 0) > 3600
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

The daemon serves the metrics on its Prometheus endpoint (default `http://127.0.0.1:9091/metrics`). Example output:

```prometheus
# HELP queue_items_enqueued_total Total items enqueued by item_type and operation
# TYPE queue_items_enqueued_total counter
queue_items_enqueued_total 42
queue_items_enqueued_total{item_type="file", op="add"} 35
queue_items_enqueued_total{item_type="text", op="update"} 7

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
```

---

## Best Practices

1. **Use rate() for counters**: Always use `rate()` or `increase()` when querying counters to get meaningful values.

2. **Choose appropriate time windows**: Use 5m for normal monitoring, 1h for trends, 15m for alerts.

3. **Monitor queue depth trends**: A steadily increasing `queue_depth_current{status="pending"}` indicates processing cannot keep up with ingest.

4. **Track wait duration percentiles**: High p95/p99 wait times indicate bottlenecks even if average is acceptable.

5. **Review Grafana dashboard regularly**: The pre-built dashboard provides at-a-glance health monitoring.
