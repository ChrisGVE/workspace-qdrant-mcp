# Logging & aggregation reference

How the daemon's structured logs are written, shipped to Loki, and queried in
Grafana. This is the C1–C3 logging path; for metrics/traces see
[`telemetry.md`](telemetry.md).

## Architecture

```text
memexd daemon ──writes──▶ daemon.jsonl ──tails──▶ promtail ──push──▶ Loki ──▶ Grafana
              JSON lines   (host logs dir)         :9080            :3100      :3000
```

The daemon writes one JSON object per line (the B3 trace-aware formatter:
each record carries `timestamp`, `level`, `target`, `span`, and — when emitted
inside an instrumented span — `trace_id`/`span_id`). Promtail tails those files
and ships them to Loki; Grafana queries Loki and renders `trace_id` as a
clickable derived field for log→trace navigation.

## Enabling / disabling

File logging is **on by default** (PRD C1). It is controlled entirely by
environment variables read at daemon startup — no config file edit required:

| Env var | Default | Effect |
|---|---|---|
| `WQM_LOG_FILE` | `true` | `false`/`0` opts out of file logging entirely |
| `WQM_LOG_FILE_PATH` | *(canonical dir)* | Override the log file path; setting it also forces file logging on |
| `WQM_LOG_LEVEL` | `INFO` | Log level (`WQM_LOG_LEVEL` > `RUST_LOG` > default) |
| `WQM_LOG_JSON` | `true`* | JSON line format (required for promtail parsing) |
| `WQM_LOG_CONSOLE` | mode-dependent | Mirror logs to stderr/console |
| `WQM_LOG_ROTATION_SIZE_MB` | *(built-in)* | Rotate the log file at this size |
| `WQM_LOG_METRICS` / `WQM_LOG_ERROR_TRACKING` | on | Performance-metric / error-tracking log channels |

\* In daemon (service) mode the canonical `daemon.jsonl` is JSON; keep
`WQM_LOG_JSON=true` so promtail's `json` pipeline stage can parse records.

To **disable** aggregation without touching the daemon, simply do not start the
Loki/promtail services (they live only in the `observability.yml` overlay).

## Paths

| What | Path |
|---|---|
| Daemon log file (canonical) | `<state dir>/logs/daemon.jsonl` (via `get_canonical_log_dir()`; on macOS `~/Library/Logs/workspace-qdrant/`, on Linux `~/.local/share/workspace-qdrant/logs/`) |
| Host mount in the compose stack | `${WQM_STATE_DIR}/memexd/logs` |
| Promtail read-only re-mount | `/var/log/memexd/*.jsonl` |
| Promtail position file | `/tmp/positions.yaml` (container) |
| Loki storage | filesystem under `/loki` (volume `loki_data`) |

## Ports

| Service | Port | Exposure |
|---|---|---|
| Loki HTTP | `3100` | published **localhost-only** (`127.0.0.1:${LOKI_PORT:-3100}`) |
| Loki gRPC | `9096` | in-network only |
| Promtail HTTP | `9080` | in-network only (no host port) |
| Grafana | `${GRAFANA_PORT:-3000}` | published |

## Querying in Grafana (LogQL)

Loki is auto-provisioned as a Grafana datasource (`uid: loki`). Examples:

```logql
# All daemon logs (by job)
{job="memexd"}

# Only warnings and errors
{job="memexd", level=~"WARN|ERROR"}

# A specific service, filtered to a substring
{service="workspace-qdrant-mcp"} |= "queue"

# Parse JSON fields at query time (trace_id/span/target are NOT labels)
{job="memexd"} | json | trace_id="3f9a1c..."

# Error rate over time
sum(count_over_time({job="memexd", level="ERROR"}[5m]))
```

**Log → trace navigation:** the Loki datasource defines a `trace_id` derived
field (`matcherRegex: "trace_id":"(\w+)"`) that links to the `tempo`
datasource. Even when no trace store (Tempo) is deployed, `trace_id` still
renders as a clickable field on each log line for copy/paste correlation.

## Label schema & bounded-cardinality rationale

Promtail emits **only** these labels (the entire Loki index dimension set):

| Label | Source | Cardinality |
|---|---|---|
| `job` | static (`memexd`) | 1 |
| `service` | static (`workspace-qdrant-mcp`) | 1 |
| `level` | parsed from `level` | ~5 (`TRACE`..`ERROR`) |

`trace_id`, `span_id`, `span`, and `target` are **parsed** from the JSON (for
query-time use via `| json` and the Grafana derived field) but are deliberately
**not** promoted to labels. Per-`trace_id` labels would create one Loki series
per trace, exploding the index — a documented DoS/cost-control limit. Loki also
defends this server-side (`max_label_names_per_series: 12`), and retention is
bounded to **7 days** (`retention_period: 168h`) with capped ingestion
(`ingestion_rate_mb: 8`, burst `16`) to keep single-host disk usage bounded.

> Do not add `trace_id`/`span_id`/`file_path` to the promtail `labels` stage.
