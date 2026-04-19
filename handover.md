# Handover — 2026-04-19 (session 2)

## Current work

Worked through **issue-62** (CLAUDE_CONFIG_DIR for hooks) completely — 6/6 tasks done, pushed to `main`. Started **issue-64** (daemon telemetry): Task 1 (config schema) done and pushed. Stopped before Task 2 because it overlaps with the existing `monitoring::metrics_core::DaemonMetrics` registry and requires a scope call.

## Task-master

- **Active tag**: `issue-64` (1/12 tasks done)
- **Completed tags this session**:
  - `issue-62` — 6/6 — shipped in commits 98361db59..ded19dd7a, issue #62 closed
- **Partially done**:
  - `issue-64` — 1/12 — config schema in place (`TelemetryConfig.service_name`, `.prometheus`, `.otlp` + `OtlpProtocol` enum, env overrides, validation, default YAML)
- **Not started**:
  - `issue-63` — 10 tasks — branch/worktree audit
  - `docker` — 15 tasks — docker compose + MCP HTTP transport
- **In progress**: none
- **Blocked**: none, but see **Pending decisions** below

## Resume instructions

Recommended next step: **resolve the Task 2 scope call below**, then continue `issue-64` with `task-master next`. If the user wants to hold on `issue-64` decisions, switch to `issue-63` (audit, no blocking decisions) or the MCP-HTTP half of `docker` (no blocking decisions on that half).

### Decision required before `issue-64` Task 2

During Task 1 implementation, I discovered `src/rust/daemon/core/src/monitoring/metrics_core.rs` (483 lines) already defines a `DaemonMetrics` registry with queue, unified queue, watch error, session, tenant, and system metrics — backed by a Prometheus `Registry`, exposed via an axum server in `monitoring/metrics_server.rs` (188 lines).

Task 2 as written says to create a **new** `src/rust/daemon/core/src/telemetry/metrics.rs` with a parallel `TELEMETRY_METRICS` struct. Two options:

1. **Extend existing `DaemonMetrics`** (add watcher event counters, gRPC request/duration, embedding/SQLite/Qdrant latencies). Smaller diff, no registry duplication, reuses `monitoring::metrics_server::MetricsServer` after a small config-awareness patch. **Recommended.**
2. **Create parallel `telemetry::` module** as written. Clean separation but duplicates the registry, the server, and the label-management helpers.

Need user call before proceeding. My recommendation is Option 1 and I'll rewrite tasks 2/3 via `task-master update-task` to reflect that before coding.

### Still pending from session 1 (carry-over)

- **Metrics SDK choice for daemon (issue-64 task 10)**: Prometheus-native registry + OTLP traces only, **or** OpenTelemetry metrics SDK with a Prometheus exporter? Defaults I'd pick without further input: prometheus crate for pull metrics (already in deps), `opentelemetry` + `opentelemetry-otlp` + `tracing-opentelemetry` for traces only (all already in deps), defer OTLP metrics export until explicitly requested.
- **Docker image distribution**: GHCR vs local-build. PRD recommends GHCR; need explicit sign-off before Phase 2 of `docker` stream.
- **Docker token rotation UX**: static `.env` vs admin endpoint.
- **TLS strategy for HTTP MCP**: native vs reverse-proxy default.
- **MCP HTTP port default**: `6335` proposed (adjacent to Qdrant 6333/6334); needs sign-off before codifying in compose.

## Key context

### Shipped this session (all on `main`)

| Commit | Summary |
|--------|---------|
| 98361db59 | feat(cli): honor CLAUDE_CONFIG_DIR in hooks settings resolver |
| d5a732df3 | test(cli): install flow respects CLAUDE_CONFIG_DIR |
| 15acc49db | test(cli): uninstall and status flows respect CLAUDE_CONFIG_DIR |
| 0e2ef2f11 | ci: guard against hardcoded .claude paths in hooks module |
| fcfb14b33 | docs(cli): document CLAUDE_CONFIG_DIR for hooks commands |
| ded19dd7a | feat(cli): surface CLAUDE_CONFIG_DIR in hook error diagnostics |
| c9eba544c | feat(daemon): add telemetry export config schema (Prometheus + OTLP) |

### Files touched in `issue-64` Task 1

- `src/rust/common/src/yaml_defaults/infrastructure.rs` — added `YamlPrometheusConfig`, `YamlOtlpConfig`; extended `YamlTelemetryConfig` with `service_name`/`prometheus`/`otlp`
- `src/rust/common/src/yaml_defaults/tests.rs` — `test_telemetry_export_defaults_parse_from_yaml`
- `src/rust/daemon/core/src/config/observability.rs` — new `OtlpProtocol` enum, `PrometheusExportConfig`, `OtlpExportConfig`; extended `TelemetryConfig` with `apply_env_overrides()` + `validate()`; 14 unit tests (serial_test for env-mutating tests)
- `src/rust/daemon/core/src/config/mod.rs` — re-exports + `build_observability_config` wiring
- `src/rust/daemon/core/src/unified_config/env_overrides.rs` — calls `apply_env_overrides` at end of load pipeline
- `src/rust/daemon/core/src/unified_config/validation.rs` — calls `telemetry.validate()` in validation pass
- `assets/default_configuration.yaml` — documents the new schema with comments

### Env var conventions codified in Task 1

- `OTEL_SERVICE_NAME` → `telemetry.service_name`
- `OTEL_EXPORTER_OTLP_ENDPOINT` → `telemetry.otlp.endpoint` (also flips `otlp.enabled = true`)
- `OTEL_EXPORTER_OTLP_PROTOCOL` → `telemetry.otlp.protocol` (`http/protobuf` | `grpc`)
- `OTEL_EXPORTER_OTLP_HEADERS` → `telemetry.otlp.headers` (comma-separated `k=v`)
- `OTEL_TRACES_SAMPLER_ARG` → `telemetry.otlp.sample_rate` (silently dropped if out of [0,1])
- `WQM_PROMETHEUS_ENABLED` / `WQM_PROMETHEUS_PORT` / `WQM_PROMETHEUS_BIND` → pull endpoint

Precedence: env > YAML > compiled-in defaults.

### GitHub issues filed in session 1

- #62 — CLOSED (this session)
- #63 — https://github.com/ChrisGVE/workspace-qdrant-mcp/issues/63 — branch/worktree audit
- #64 — https://github.com/ChrisGVE/workspace-qdrant-mcp/issues/64 — daemon OTLP + Prometheus (in progress, 1/12)

### PRD files (.taskmaster/docs/)

- `20260419-2128_workspace-qdrant-mcp_v0.1.3_PRD_issue-62-hooks-claude-config-dir.txt` — delivered
- `20260419-2128_workspace-qdrant-mcp_v0.1.3_PRD_issue-63-branch-worktree-audit.txt`
- `20260419-2128_workspace-qdrant-mcp_v0.1.3_PRD_issue-64-daemon-otlp.txt`
- `20260419-2128_workspace-qdrant-mcp_v0.1.3_PRD_docker-http-transport.txt`

### Reference files for remaining work

- Existing metrics registry (extend, don't duplicate): `src/rust/daemon/core/src/monitoring/metrics_core.rs`
- Existing axum /metrics server (extend for config-driven startup): `src/rust/daemon/core/src/monitoring/metrics_server.rs`
- Existing OTLP trace setup (extend for config-driven sampler/protocol): `src/rust/daemon/core/src/tracing_otel.rs`
- TS OTLP reference: `src/typescript/mcp-server/src/telemetry/otlp.ts`
- MCP transport code: `src/typescript/mcp-server/src/server.ts:194-200` (stdio-only today)
- Hooks settings path fix (done): `src/rust/cli/src/commands/hooks/settings.rs`
- Existing compose pattern reference: `/Users/chris/dev/tools/main-docker/docker-compose.yml`
- Architectural invariants: `docs/specs/04-write-path.md` (ADR-003 daemon owns state), ADR-001 (canonical collections)

### Gotchas carried forward

- `task-master parse-prd` takes ~3 min per PRD.
- `numTasks` parameter is a hint, not exact.
- Task-master tag names cannot start with `#`.
- Env-mutating tests must be marked `#[serial]` (`serial_test` crate already a workspace dev-dep); `wqm-cli` had to add its own `serial_test` dev-dep in session 2.
- Clippy run on workspace-qdrant-core emits warnings from `daemon/core/src/graph/algorithms/community.rs`, `graph/sqlite_store.rs`, `patterns/detection/detector.rs`, `keyword_extraction/semantic_rerank.rs`. Pre-existing, unrelated to this work. Exit 0 though — they're deferred cleanup.
- `MetricsServer::start()` in `monitoring/metrics_server.rs` ignores the new `telemetry.prometheus.*` config today; the wiring patch is part of Task 3 scope.
