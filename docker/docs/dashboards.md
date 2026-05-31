# Grafana dashboards

Four pre-built dashboards are provisioned automatically when Grafana starts with
the config in `docker/grafana/provisioning/`. All dashboards auto-refresh every
30 seconds and default to a 1-hour time window.

Import JSON files from `docker/grafana/dashboards/` if auto-provisioning is not
in use (Dashboards → Import in the Grafana UI).

## Dashboard list

| File | UID | Title | Focus |
|---|---|---|---|
| `system-overview.json` | `wqm-system` | WQM — System Overview | Service health, queue summary, error events |
| `memexd.json` | `wqm-memexd` | WQM — memexd Daemon | Queue depth, throughput, latency, sessions |
| `claude-mcp.json` | `wqm-claude-mcp` | WQM — MCP Server | Tool rates, durations, sessions, fallbacks |
| `qdrant.json` | `wqm-qdrant` | WQM — Qdrant | Collections, vectors, REST/gRPC latency |

Navigation links in the System Overview header jump directly to the other three
dashboards.

---

## WQM — System Overview (`system-overview.json`)

Start here. Gives a health snapshot across all services and surfaces active
problems.

[Screenshot: five stat panels in a row, then two time series below, then a table]

### Panel inventory

**Row 1 — Service health (stat panels)**

| Panel | Metric | Interpretation |
|---|---|---|
| memexd | `up{job="memexd"}` | GREEN=UP / RED=DOWN. Any red value requires immediate attention. |
| MCP Server | `up{job="mcp"}` | Same. Note: only meaningful in HTTP mode; in stdio mode the scrape job is absent. |
| Qdrant | `up{job="qdrant"}` | RED means all searches and writes are failing. |
| Total Pending Queue Items | `sum(wqm_memexd_unified_queue_depth{status="pending"})` | Counts items waiting to be processed. Amber above 100, red above 500. |
| Active MCP Sessions | `wqm_mcp_session_count` | Number of connected Claude Code sessions. |

**Row 2 — Activity trend (time series)**

| Panel | Query summary | Interpretation |
|---|---|---|
| MCP Tool Invocation Rate | `sum(rate(wqm_mcp_tool_invocations_total[2m]))` | Overall call rate across all tools. A flat line at zero when sessions are active indicates a stalled client. |
| Queue Depth Trend | `sum(wqm_memexd_queue_depth)` (legacy) + `sum(wqm_memexd_unified_queue_depth{status="pending"})` | Rising trend with no corresponding throughput increase indicates a queue processor problem. |

**Row 3 — Error events (table)**

Instant snapshot of the current error rate across four sources:

- `wqm_memexd_ingestion_errors_total` — ingestion failures by error type
- `wqm_memexd_watch_errors_total` — filesystem watch failures by watch ID
- `wqm_mcp_tool_invocations_total{status="error"}` — MCP tool errors by tool
- `wqm_mcp_daemon_fallback_total` — daemon-unreachable fallbacks by tool and reason

All rows are green at zero. Non-zero rows with a yellow or red background mean
active errors. Use the tool and error_type labels to identify the failing component.

---

## WQM — memexd Daemon (`memexd.json`)

Deep-dive on the Rust daemon internals.

[Screenshot: four stat panels in a row, then four time series below, then a full-width time series]

### Panel inventory

**Row 1 — Instant state (stat panels)**

| Panel | Metric | Thresholds | Interpretation |
|---|---|---|---|
| Daemon Uptime | `max(wqm_memexd_uptime_seconds)` | Red < 60 s, yellow < 5 min, green ≥ 5 min | A low value after a known-stable run indicates a recent restart. |
| Active Sessions | `sum(wqm_memexd_active_sessions)` | Yellow ≥ 50, red ≥ 200 | Sessions across all projects and priorities. |
| Watches in Backoff | `sum(wqm_memexd_watches_in_backoff)` | Yellow ≥ 1, red ≥ 5 | Any non-zero value means at least one filesystem watch is experiencing errors and is throttling events. Investigate with `wqm project watch status`. |
| Oldest Pending Queue Item Age | `wqm_memexd_queue_oldest_pending_age_seconds` | Yellow ≥ 60 s, red ≥ 300 s | The QueueStuck alert fires at 12 hours; this panel gives earlier warning. |

**Row 2 — Queue (time series)**

| Panel | Queries | Interpretation |
|---|---|---|
| Queue Depth by Priority | `sum by (priority) (wqm_memexd_queue_depth)` | Separate lines per priority level. A persistent non-draining queue indicates processor slowness or blockage. |
| Queue Processing Throughput | Success: `rate(wqm_memexd_queue_items_processed_total{status="success"})` / Failure: same with `status="failure"` | Failure lines in red. Any visible failure rate warrants log investigation. |

**Row 3 — Latency and errors (time series)**

| Panel | Queries | Interpretation |
|---|---|---|
| Queue Processing Latency (P50/P95/P99) | `histogram_quantile(0.50|0.95|0.99, rate(wqm_memexd_queue_processing_time_seconds_bucket[5m]))` | Normal processing should stay well under 1 s at P99. Elevated P99 with normal P50 suggests occasional slow items (large files, tree-sitter parse time). |
| Ingestion and Watch Error Rates | `rate(wqm_memexd_ingestion_errors_total[2m])` + `rate(wqm_memexd_watch_errors_total[2m])` | Both should be zero in steady state. Persistent watch errors often indicate a deleted or unmounted directory that memexd is trying to watch. |

**Row 4 — Unified queue (time series)**

| Panel | Query | Interpretation |
|---|---|---|
| Unified Queue Depth by Type and Status | `sum by (item_type, status) (wqm_memexd_unified_queue_depth)` | Breaks down pending, in_progress, done, failed items per item type. A growing `in_progress` count with no corresponding `done` increase indicates stalled processing. |

---

## WQM — MCP Server (`claude-mcp.json`)

MCP-layer observability: what tools are being called, how fast, and whether the
daemon connection is healthy.

[Screenshot: one stat, two time series on row 1; two time series on row 2; two time series on row 3; one full-width bar chart]

### Panel inventory

**Row 1 — Session and call rates**

| Panel | Metric | Interpretation |
|---|---|---|
| Active MCP Sessions | `wqm_mcp_session_count` | Gauge. Increments on session open, decrements on close. |
| Tool Invocation Rate | `sum by (tool) (rate(wqm_mcp_tool_invocations_total[2m]))` | Per-tool call rates. Identifies which tools are most active. |
| Tool Error Rate | `sum by (tool) (rate(wqm_mcp_tool_invocations_total{status="error"}[2m]))` | Red lines. Non-zero indicates tool failures; correlate with `docker logs workspace-qdrant-mcp`. |

**Row 2 — Latency**

| Panel | Queries | Interpretation |
|---|---|---|
| Tool P99 Duration | `histogram_quantile(0.99, rate(wqm_mcp_tool_duration_seconds_bucket[5m]))` | Per-tool 99th-percentile latency. Consistently above 1 s is amber; above 5 s is red. |
| Tool Duration Heatmap (p50/p95/p99) | Three quantiles across all tools, bar gauge | Horizontal bars. Useful for spotting which tools are consistently slow vs. occasionally spiking. |

**Row 3 — Connectivity and cache**

| Panel | Metric | Interpretation |
|---|---|---|
| Daemon Fallback Rate | `sum by (tool, reason) (rate(wqm_mcp_daemon_fallback_total[2m]))` | Non-zero means the MCP server could not reach memexd. Check the `reason` label (`connection_failed`, `timeout`, etc.) and inspect memexd health. |
| Cache Hit Ratio | `hits / (hits + misses)` using `wqm_mcp_cache_hits_total` and `wqm_mcp_cache_misses_total` | Always zero at v0.1.3 — no cache layer is implemented. Panel is present for future use. |

**Row 4 — Cumulative breakdown (bar chart)**

| Panel | Query | Interpretation |
|---|---|---|
| Tool Success / Error Breakdown | `sum by (tool, status) (increase(wqm_mcp_tool_invocations_total[$__range]))` | Absolute counts for the selected time range, stacked by status. Identifies error-prone tools at a glance. |

---

## WQM — Qdrant (`qdrant.json`)

Monitors the Qdrant vector database. Qdrant metrics are native to Qdrant — this
dashboard does not require any workspace-qdrant instrumentation.

[Screenshot: three stat panels on row 1; two time series on row 2; two time series on row 3; one full-width time series]

### Panel inventory

**Row 1 — Database state (stat panels)**

| Panel | Metric | Interpretation |
|---|---|---|
| Collections | `qdrant_collections_total` | Total number of Qdrant collections. workspace-qdrant uses 4: `projects`, `libraries`, `rules`, `scratchpad`. |
| Total Vectors | `qdrant_collections_vector_total` | All vectors across all collections. Amber at 1 M, red at 10 M. |
| Qdrant Up | `up{job="qdrant"}` | UP / DOWN scrape health indicator. |

**Row 2 — REST traffic**

| Panel | Metric | Interpretation |
|---|---|---|
| REST Request Rate | `sum by (endpoint) (rate(qdrant_rest_responses_total[2m]))` | Per-endpoint HTTP throughput. |
| REST P99 Latency | `histogram_quantile(0.99, rate(qdrant_rest_responses_duration_seconds_bucket[5m]))` | 99th-percentile response time. Amber above 100 ms, red above 1 s. |

**Row 3 — gRPC traffic**

| Panel | Metric | Interpretation |
|---|---|---|
| gRPC Request Rate | `sum by (endpoint) (rate(qdrant_grpc_responses_total[2m]))` | gRPC call throughput. Absent if Qdrant runs without gRPC enabled. |
| gRPC P99 Latency | `histogram_quantile(0.99, rate(qdrant_grpc_responses_duration_seconds_bucket[5m]))` | Amber above 50 ms, red above 500 ms. |

**Row 4 — Compaction**

| Panel | Metric | Interpretation |
|---|---|---|
| Pending Optimisations | `sum by (collection_name, optimizer_name) (qdrant_collections_optimizers_status)` | Collection segments under active optimisation. A sustained non-zero value is normal during heavy ingestion. A value that never returns to zero indicates a stuck optimizer. |

---

## Navigation tips

- Use the **time range picker** (top right) to zoom into incident windows. The
  dashboards use 30 s auto-refresh; disable it to keep a static view during
  investigation.
- The System Overview dashboard links (top header) jump directly to the other
  three dashboards with the same time range preserved.
- In table panels, click a column header to sort. Click a row value to copy it
  as a filter for further exploration in Explore.
- In time series panels, click a legend entry to isolate that series; Shift+click
  to add more.

## Adding custom alerts to dashboard panels

1. Open the panel in edit mode (click the panel title → Edit).
2. Go to **Alert → New alert rule**.
3. Set the condition using the existing query or a modified version.
4. Assign a contact point in **Alerting → Contact points** first.

Alternatively, add rules directly to `docker/prometheus/alerts.yml` and reload
Prometheus. This keeps alert definitions in version control alongside the rest of
the deployment config.

_workspace-qdrant-mcp v0.1.3 — documentation updated 2026-04-18_
