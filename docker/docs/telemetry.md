# Telemetry reference

## Architecture

```text
memexd daemon  --scrape-->  Prometheus --> Grafana
                 :9091

MCP server     --scrape-->  Prometheus --> Grafana
(http mode)      :9092

MCP server     --push-->    otel-collector --> Prometheus --> Grafana
(stdio mode)     OTLP
```

In `stdio` mode (the default), the MCP server cannot serve an HTTP endpoint.
It pushes metrics to the OpenTelemetry Collector via OTLP when the process
exits. The collector then exposes them at `:8888` for Prometheus to scrape.

In `http` mode (`MCP_SERVER_MODE=http`), the MCP server serves `/metrics`
directly at `:9092`.

## Prometheus scrape jobs

Defined in `docker/prometheus/prometheus.yml`:

| Job | Target | Scrape path |
|---|---|---|
| `memexd` | `memexd:9091` | `/metrics` |
| `mcp` | `mcp:9092` | `/metrics` |
| `qdrant` | `qdrant:6333` | `/metrics` |
| `otel-collector` | `otel-collector:8888` | `/metrics` |

Scrape interval: 15 s. Rule evaluation interval: 15 s.

## MCP server metrics

Source: `src/typescript/mcp-server/src/telemetry/metrics.ts`

| Metric | Type | Labels | Description |
|---|---|---|---|
| `wqm_mcp_tool_invocations_total` | Counter | `tool`, `status` | Total tool calls; `status` is `success` or `error`. |
| `wqm_mcp_tool_duration_seconds` | Histogram | `tool` | Tool execution time. |
| `wqm_mcp_session_count` | Gauge | - | Active MCP sessions. |
| `wqm_mcp_daemon_fallback_total` | Counter | `tool`, `reason` | Times the daemon was unreachable and a fallback was used. |
| `wqm_mcp_cache_hits_total` | Counter | `cache` | Cache hits by cache name. The current build exposes the metric, but no cache layer is wired yet. |
| `wqm_mcp_cache_misses_total` | Counter | `cache` | Cache misses by cache name. |
| `wqm_mcp_http_requests_total` | Counter | `path`, `status_class`, `status` | HTTP request volume in transport mode. |
| `wqm_mcp_http_auth_failures_total` | Counter | `reason` | HTTP auth failures by reason. |
| `wqm_mcp_http_rate_limited_total` | Counter | - | Requests throttled by the per-IP limiter. |

## Daemon metrics

Source: `src/rust/daemon/core/src/monitoring/metrics_core.rs`

The current daemon build in this fork exports a smaller surface than the older
queue-depth dashboards did. The queue-depth and per-watch counters that older
docs mentioned are no longer the authoritative source of truth here.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `memexd_uptime_seconds` | Gauge | - | Daemon process uptime. |
| `memexd_unified_queue_stale_items` | Gauge | - | Stale lease items in the unified queue. |
| `wqm_queue_oldest_pending_age_seconds` | Gauge | - | Age in seconds of the oldest pending queue item. |

## Qdrant metrics

Source: Qdrant native `/metrics`

| Metric | Type | Labels | Description |
|---|---|---|---|
| `collections_total` | Gauge | - | Number of collections. |
| `collections_vector_total` | Gauge | - | Total vectors across all collections. |
| `collection_points` | Gauge | `id` | Approximate point count per collection. |
| `collection_vectors` | Gauge | `collection`, `vector` | Number of vectors per collection and vector name. |
| `collection_indexed_only_excluded_points` | Gauge | `id`, `vector` | Number of points excluded from `indexed_only` search per collection and vector name. |
| `collection_running_optimizations` | Gauge | `id` | Running optimisation tasks per collection. |
| `rest_responses_total` | Counter | `method`, `endpoint`, `status` | REST request volume. |
| `rest_responses_fail_total` | Counter | `method`, `endpoint`, `status` | REST failures. |
| `rest_responses_duration_seconds` | Histogram | `method`, `endpoint`, `status`, `le` | REST latency histogram. |
| `grpc_responses_total` | Counter | `endpoint` | gRPC request volume. |
| `grpc_responses_fail_total` | Counter | `endpoint` | gRPC failures. |
| `grpc_responses_duration_seconds` | Histogram | `endpoint`, `le` | gRPC latency histogram. |
| `memory_resident_bytes` | Gauge | - | Resident memory usage. |
| `process_open_fds` | Gauge | - | Open file descriptors. |
| `process_max_fds` | Gauge | - | File descriptor limit. |
| `process_threads` | Gauge | - | Active thread count. |

## Prometheus query recipes

### Tool invocation rate

```promql
sum by (tool) (rate(wqm_mcp_tool_invocations_total[5m]))
```

### Tool error rate

```promql
sum by (tool) (rate(wqm_mcp_tool_invocations_total{status="error"}[5m]))
```

### Queue staleness

```promql
sum(memexd_unified_queue_stale_items)
```

### Oldest pending item age

```promql
wqm_queue_oldest_pending_age_seconds
```

### REST request rate

```promql
sum by (endpoint) (rate(rest_responses_total[2m]))
```

### REST P99 latency

```promql
histogram_quantile(0.99, sum by (endpoint, le) (rate(rest_responses_duration_seconds_bucket[5m])))
```

### gRPC P99 latency

```promql
histogram_quantile(0.99, sum by (endpoint, le) (rate(grpc_responses_duration_seconds_bucket[5m])))
```

### Open FD saturation

```promql
process_open_fds / clamp_min(process_max_fds, 1)
```

## Alert rules

Defined in `docker/prometheus/alerts.yml`.

| Alert | Severity | Condition |
|---|---|---|
| `QueueStuck` | warning | Oldest pending queue item older than 12 hours |
| `QueueStaleWarning` | warning | One or more stale queue items are present for 5 minutes |
| `QueueStaleCritical` | critical | More than 10 stale queue items are present for 5 minutes |
| `DaemonDown` | critical | `up{job="memexd"} == 0` |
| `QdrantUnreachable` | critical | `up{job="qdrant"} == 0` |
| `MCPNoInvocations` | info | Session active but no tool invocations for 15 minutes |

`QueueStuck` and the stale-queue alerts are aligned with the current daemon
surface because they use `wqm_queue_oldest_pending_age_seconds` and
`memexd_unified_queue_stale_items`.

## OpenTelemetry Collector

Config: `docker/otel/otel-collector-config.yml`

The collector receives OTLP metrics from the MCP server in `stdio` mode and
exposes them as a Prometheus scrape target at `:8888`.

## Adaptive resource management in containers

The daemon has an adaptive resource manager that scales embedding concurrency
and inter-item delay based on host user activity (Normal -> Active -> RampingUp
-> Burst). Inside a Docker container the idle signal is usually unavailable, so
the state machine stays in the lower modes.

That behaviour is expected and does not indicate a failure.

_workspace-qdrant-mcp v0.1.3 - documentation updated 2026-05-24_
