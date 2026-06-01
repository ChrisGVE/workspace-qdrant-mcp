# Telemetry reference

## Architecture

```text
memexd daemon  ──scrape──▶  Prometheus ──▶  Grafana
                 :6337
MCP server     ──scrape──▶  Prometheus ──▶  Grafana
(http mode)      :9092

MCP server     ──push──▶  otel-collector ──▶  Prometheus ──▶  Grafana
(stdio mode)     OTLP
```

In `stdio` mode (default), the MCP server cannot serve an HTTP endpoint. It
pushes accumulated metrics to the OpenTelemetry Collector via OTLP on process
exit. The otel-collector then exposes them at `:8888` for Prometheus to scrape.

In `http` mode (`MCP_SERVER_MODE=http`), the MCP server serves `/metrics`
directly at `:9092`.

## Prometheus scrape jobs

Defined in `docker/prometheus/prometheus.yml`:

| Job | Target | Scrape path |
|---|---|---|
| `memexd` | `memexd:6337` | `/metrics` |
| `mcp` | `mcp:9092` | `/metrics` |
| `qdrant` | `qdrant:6333` | `/metrics` |
| `otel-collector` | `otel-collector:8888` | `/metrics` |

Scrape interval: 15 s. Rule evaluation interval: 15 s.

## MCP server metrics

Source: `src/typescript/mcp-server/src/telemetry/metrics.ts`

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_mcp_tool_invocations_total` | Counter | `tool`, `status` | Total tool calls; `status` is `success` or `error` |
| `wqm_mcp_tool_duration_seconds` | Histogram | `tool` | Tool execution time; buckets: 0.01, 0.05, 0.1, 0.5, 1, 5 s |
| `wqm_mcp_session_count` | Gauge | — | Active MCP sessions |
| `wqm_mcp_daemon_fallback_total` | Counter | `tool`, `reason` | Times the daemon was unreachable and a fallback was triggered |
| `wqm_mcp_cache_hits_total` | Counter | `cache` | Cache hits by cache name (defined; no cache layer at v0.1.3 — always 0) |
| `wqm_mcp_cache_misses_total` | Counter | `cache` | Cache misses by cache name (defined; no cache layer at v0.1.3 — always 0) |

## Daemon metrics

Source: `src/rust/daemon/core/src/monitoring/metrics_core.rs`

All daemon metrics use the `memexd` namespace prefix except
`wqm_memexd_queue_oldest_pending_age_seconds`.

### Queue metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_memexd_queue_depth` | Gauge | `priority`, `collection` | Current depth of the legacy queue by priority and collection |
| `wqm_memexd_queue_items_processed_total` | Counter | `priority`, `status` | Items processed; `status` is `success`, `failure`, or `skipped` |
| `wqm_memexd_queue_processing_time_seconds` | Histogram | `priority` | Processing time per item |
| `wqm_memexd_queue_oldest_pending_age_seconds` | Gauge | — | Age in seconds of the oldest pending item (0 if queue is empty) |
| `wqm_memexd_unified_queue_depth` | Gauge | `item_type`, `status` | Unified queue depth by type and status |
| `wqm_memexd_unified_queue_processing_time_seconds` | Histogram | `item_type` | Unified queue processing time |
| `wqm_memexd_unified_queue_items_total` | Counter | `item_type`, `op`, `result` | Unified queue items processed |
| `wqm_memexd_unified_queue_enqueues_total` | Counter | `source` | Enqueues by source |
| `wqm_memexd_unified_queue_dequeues_total` | Counter | `item_type` | Dequeues by item type |
| `wqm_memexd_unified_queue_stale_items` | Gauge | — | Expired leases not yet recovered |
| `wqm_memexd_unified_queue_retries_total` | Counter | `item_type` | Retry count by item type |

### Session metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_memexd_active_sessions` | Gauge | `project_id`, `priority` | Active sessions by project and priority |
| `wqm_memexd_total_sessions` | Counter | `project_id` | Lifetime session count by project |
| `wqm_memexd_session_duration_seconds` | Histogram | `project_id` | Session duration |

### System metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_memexd_uptime_seconds` | Gauge | — | Daemon process uptime |
| `wqm_memexd_ingestion_errors_total` | Counter | `error_type` | Ingestion errors by type |
| `wqm_memexd_heartbeat_latency_seconds` | Histogram | `project_id` | Heartbeat processing latency |

### Watch metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_memexd_watch_errors_total` | Counter | `watch_id` | Cumulative watch errors |
| `wqm_memexd_watch_consecutive_errors` | Gauge | `watch_id` | Current run of consecutive errors |
| `wqm_memexd_watch_health_status` | Gauge | `watch_id`, `health_status` | Health state flag (1 = in this state); states: `healthy`, `degraded`, `backoff`, `disabled` |
| `wqm_memexd_watches_in_backoff` | Gauge | — | Watches currently in exponential backoff |
| `wqm_memexd_watch_recovery_time_seconds` | Histogram | `watch_id` | Time from first error to recovery |
| `wqm_memexd_watch_events_throttled_total` | Counter | `watch_id`, `load_level` | Events dropped due to queue pressure; `load_level` is `high` or `critical` |

### Per-tenant metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_memexd_tenant_documents_total` | Gauge | `tenant_id`, `collection` | Document count per tenant and collection |
| `wqm_memexd_tenant_search_requests_total` | Counter | `tenant_id` | Search requests per tenant |
| `wqm_memexd_tenant_storage_bytes` | Gauge | `tenant_id` | Estimated storage usage per tenant |

## Histogram bucket layouts (stable API)

Histogram bucket boundaries are a **stable API**. Dashboards built on
`histogram_quantile()` and heatmaps interpolate across these boundaries, so
changing a layout silently breaks every panel that spans the changed edge.
**To change buckets, rename the metric** — never edit a layout in place. The
frozen layouts are declared as single-source-of-truth consts in
`src/rust/daemon/core/src/monitoring/metrics_factories.rs`:

| Metric | Const | Bucket boundaries (seconds) |
|---|---|---|
| `wqm_memexd_embedding_duration_seconds` | `EMBEDDING_DURATION_BUCKETS` | `0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0` |
| `wqm_memexd_processing_duration_seconds` | `PROCESSING_DURATION_BUCKETS` | `0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0` |
| `wqm_memexd_search_duration_seconds` | `SEARCH_DURATION_BUCKETS` | `0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0` |
| `wqm_memexd_search_result_count` | `SEARCH_RESULT_COUNT_BUCKETS` | `1, 2, 5, 10, 20, 50, 100, 250, 500, 1000` (result counts, not seconds) |

The embedding layout gained the `10.0` and `30.0` upper buckets so cold model
loads and large-batch embeds (which exceed 5s) remain measurable at p99
instead of collapsing into `+Inf`. The processing layout is a strict superset
of the embedding layout's upper region (both end `…10.0, 30.0`) so quantile
comparisons across the two histograms align on shared boundaries.

## Prometheus query recipes

### Tool invocation rate (per tool, per second)

```promql
sum by (tool) (rate(wqm_mcp_tool_invocations_total[5m]))
```

### Tool error rate

```promql
sum by (tool) (rate(wqm_mcp_tool_invocations_total{status="error"}[5m]))
```

### Tool P99 latency

```promql
histogram_quantile(0.99,
  sum by (tool, le) (rate(wqm_mcp_tool_duration_seconds_bucket[5m]))
)
```

### Queue depth (unified, pending items only)

```promql
sum(wqm_memexd_unified_queue_depth{status="pending"})
```

### Queue failure rate (per hour)

```promql
increase(wqm_memexd_queue_items_processed_total{status="failure"}[1h])
```

### Oldest pending item age

```promql
wqm_memexd_queue_oldest_pending_age_seconds
```

### Daemon uptime

```promql
max(wqm_memexd_uptime_seconds)
```

### Watch health — any watch in backoff?

```promql
sum(wqm_memexd_watches_in_backoff) > 0
```

### Daemon fallback rate

```promql
sum by (tool, reason) (rate(wqm_mcp_daemon_fallback_total[5m]))
```

## Alert rules

Defined in `docker/prometheus/alerts.yml`. Six rules in the
`workspace-qdrant-alerts` group, evaluated every 30 s.

| Alert | Severity | Condition | Fires after |
|---|---|---|---|
| `QueueStuck` | warning | Oldest pending item older than 12 hours | 5 m |
| `QueueFailedWarning` | warning | Any queue failures in the last hour | 5 m |
| `QueueFailedCritical` | critical | More than 10 failures in the last hour | 5 m |
| `DaemonDown` | critical | `up{job="memexd"} == 0` | 5 m |
| `QdrantUnreachable` | critical | `up{job="qdrant"} == 0` | 5 m |
| `MCPNoInvocations` | info | Session active but no tool calls for 15 m | 5 m |

### Alert details

**QueueStuck** — `max_over_time(wqm_memexd_queue_oldest_pending_age_seconds[1h]) > 43200`  
The oldest pending queue item has not been picked up in 12 hours. Likely cause:
queue processor stopped or a task type is permanently erroring.

**QueueFailedWarning** — `increase(wqm_memexd_queue_items_processed_total{status="failed"}[1h]) > 0`  
At least one item failed processing in the last hour. Inspect `docker logs memexd`
for the failure reason and `error_type` label in `wqm_memexd_ingestion_errors_total`.

**QueueFailedCritical** — same counter, threshold 10/hour  
Ten or more failures in one hour. Indicates a systemic problem rather than an
isolated error.

**DaemonDown** — `up{job="memexd"} == 0`  
Prometheus cannot reach memexd at `memexd:6337`. The container may have crashed
or the network route is broken.

**QdrantUnreachable** — `up{job="qdrant"} == 0`  
Prometheus cannot reach Qdrant. All write and search operations will fail.

**MCPNoInvocations** — `rate(wqm_mcp_tool_invocations_total[15m]) == 0 and on() wqm_mcp_session_count > 0`  
An MCP session is registered but no tools have been called for 15 minutes. This
is informational — may indicate an idle Claude Code session or a stalled client.

### Note on `status="failed"` vs `status="failure"`

The alert rules filter `status="failed"` but the daemon counter uses `status`
values `success`, `failure`, and `skipped` as defined in `metrics_core.rs`. The
rules in `alerts.yml` use `status="failed"` which will match zero rows; this is
a known discrepancy between the alert file and the actual metric labels. The
alert fires on no matches (rate == 0), meaning `QueueFailedWarning` and
`QueueFailedCritical` currently never fire. The correct filter is
`status="failure"`. This will be corrected in a future patch.

## OpenTelemetry Collector

Config: `docker/otel/otel-collector-config.yml`

The collector receives OTLP metrics (from the MCP server in stdio mode) and
exposes them as a Prometheus scrape target at `:8888`. Batch size: 10 items,
timeout: 10 s.

## Adaptive resource management in containers

The daemon has an adaptive resource manager that scales embedding concurrency
and inter-item delay based on host user activity (Normal → Active → RampingUp
→ Burst). Idle detection uses CoreGraphics `CGEventSourceSecondsSinceLastEventType`
with an IOKit `HIDIdleTime` fallback — **macOS host only**.

Inside a Docker container (always Linux, including Docker Desktop for Mac),
`seconds_since_last_input()` returns `None` and the state machine stays at
Normal / Active:

- `Burst` and `RampingUp` modes are unreachable
- `+50% concurrency when user present and queue has work` degenerates to
  `+50% whenever queue has work`
- Queue-depth throttling (`watching_queue/throttle.rs`) and CPU-pressure
  detection (load average) continue to work unchanged

In practice the daemon is slightly more conservative in Docker than on a
macOS host — it cannot exploit the "user away, ramp up" window. No crash,
no misbehaviour.

Future: a Linux-native idle signal (systemd-logind `IdleHint`, host bind
mount of `/run/user/$UID`, or a manual gRPC "go fast" command) is tracked
for a later minor release.

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
