# Handover — 2026-04-18 (Session 12 setup → Session 13 start)

## What to do first in the next session

1. Verify branch: `git branch --show-current` → must be `dev`
2. Switch task-master tag: confirm via `task-master list-tag` or MCP, active tag must be `v0-1-3`
3. Read PRD: `.taskmaster/docs/20260418-0033_project_0.1.3_PRD_docker-telemetry-dashboards.md`
4. `task-master next` → should return Task 1 (Dockerfiles)
5. Begin execution per CLAUDE.md plan execution protocol — sequential agents, one at a time (project preference)

## Current state (end of session 12 setup)

- **Branch:** `dev` (created from `main` this session, zero commits on dev yet — clean tree)
- **task-master active tag:** `v0-1-3` (dot not allowed in tag names; corresponds to project version 0.1.3)
- **16 tasks generated + expanded** from PRD; complexity report saved at `.taskmaster/reports/task-complexity-report_v0-1-3.json`
- **All 3518 tests still passing** from v0.1.2; no code changes this session
- **Daemon running via launchctl** from v0.1.2 binary (schema v34)
- **GH milestones:** v0.1.3 (#4), v0.1.4 (#5)
- **GH issues for v0.1.4:** #57 (rules UUID bug), #58 (legacy rules backfill), #59 (reconciliation batching)

## What was done in session 12 (setup only)

Session scoped to PRD decisions + task-master setup. No implementation code written.

### Decisions resolved (PRD open questions)

1. **MCP stdio cold-start (~1-2s)**: accepted for 0.1.3. Daemon-mode MCP = potential v0.1.4 enhancement.
2. **Prometheus retention**: inherit from host `main-docker/prometheus.yml` (360d currently). No override in our compose.
3. **Alerting**: in scope for 0.1.3. 6 rules defined in PRD §G5b:
   - QueueStuck (warning, > 12h oldest pending)
   - QueueFailedWarning (warning, any failed in 1h)
   - QueueFailedCritical (critical, > 10 failed in 1h)
   - DaemonDown (critical, memexd scrape fails > 5m)
   - QdrantUnreachable (critical, qdrant scrape fails > 5m)
   - MCPNoInvocations (info, active session but zero tool activity 15m)
4. **Image signing (cosign)**: deferred to v0.1.4.
5. **Session 11 bugs**: filed as GH issues #57/#58/#59 on v0.1.4 milestone.

### PRD additions

- §G3 Telemetry pipeline: added "Retention: inherited from host Prometheus config (360d). No override in our compose files."
- §G5b Alerting rules: new section, 6 alerts with expressions + severities
- Replaced §Open Questions with §Resolved Decisions
- New required metric: `wqm_queue_oldest_pending_age_seconds` (gauge, daemon-side) — driver for Task 7

### GitHub actions this session

- Created milestones v0.1.3 + v0.1.4 (API — not in UI until items assigned)
- Created labels: (none new — `rust`, `typescript`, `bug`, `performance` all exist from v0.1.2)
- Filed 3 issues on v0.1.4 (see above)

### Task-master actions this session

- `add_tag v0-1-3` (dot characters rejected → used hyphens, equivalent to `v0.1.3`)
- `use_tag v0-1-3`
- `parse_prd --append` on PRD: 16 top-level tasks generated
- `analyze_project_complexity`: 1 high (task 3, complexity 8), 12 medium (5-7), 3 low (tasks 7, 9, 16)
- `expand_all`: 16/16 expanded, 0 failed, 0 skipped

## Task execution plan

### Wave 1 — no dependencies (can parallelize, but per project preference = sequential)

- **Task 1**: Dockerfiles (memexd + MCP), multi-arch, < 400MB / < 200MB — complexity 7 → agent: `docker-expert`
- **Task 5**: MCP TS prom-client instrumentation — complexity 6 → agent: `typescript-pro`
- **Task 7**: `wqm_queue_oldest_pending_age_seconds` gauge in daemon — complexity 4 → agent: `rust-engineer`
- **Task 10**: `DaemonSource` enum + detect.rs — complexity 6 → agent: `rust-engineer`

### Wave 2 — deps on wave 1

- **Task 2** (deps 1): GH Actions Docker publish workflow → `github-actions-expert`
- **Task 3** (deps 1): compose files for 3 deployment modes (complexity 8 — highest)
- **Task 4** (deps 3): prometheus.yml scrape targets
- **Task 6** (deps 5): OTLP push from MCP stdio mode
- **Task 9** (deps 7): `alerts.yml` with 6 alerts
- **Task 11** (deps 10): update `wqm service {start,stop,restart,status}` to honor DaemonSource

### Wave 3 — UI + integration

- **Task 8** (deps 4, 7): 4 Grafana dashboards JSON + provisioning config → design-heavy
- **Task 13** (deps 1, 2): GH Actions docker-test.yml (pre-publish)
- **Task 14** (deps 3, 4, 8, 9): integrate prometheus/grafana/otel-collector into docker-compose.yml
- **Task 12** (deps 3, 4, 8, 9): 6 markdown files in `docker/docs/`
- **Task 15** (deps 3, 4, 14): e2e integration test script

### Wave 4 — release

- **Task 16** (deps 2, 12, 15): version bump + tag `v0.1.3` push

## Key context for next session

### User's main-docker stack (already running)

`/Users/chris/dev/tools/main-docker/docker-compose.yml` — contains `qdrant`, `otel-collector`, `prometheus` (360d retention), `grafana`, `watchtower`. Full-stack.yml will extend this (not replace).

### Docker Hub secrets (GitHub repo)

`DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` already added. GHCR uses `GITHUB_TOKEN` (auto-available).

### Daemon Prometheus infrastructure (already implemented, just needs wiring)

- `src/rust/daemon/core/src/monitoring/metrics_core.rs` — Counter/Histogram/Gauge registry
- `src/rust/daemon/core/src/monitoring/metrics_server.rs` — HTTP server (`/metrics` endpoint)
- `src/rust/daemon/memexd/src/background.rs:30` — `start_metrics_server(port)` wiring
- `--metrics-port` CLI flag in `src/rust/daemon/memexd/src/startup.rs`

Task 4 just needs to pass the flag in docker-compose + add prometheus.yml scrape target.

### MCP server instrumentation (greenfield)

`src/typescript/mcp-server/src/` — no `prom-client`, no metrics yet. Task 5 adds:
- `src/typescript/mcp-server/src/telemetry/metrics.ts` — prom-client registry + tool wrapper
- `src/typescript/mcp-server/src/telemetry/http-server.ts` — `/metrics` endpoint (HTTP mode)
- OTLP exporter (Task 6, stdio mode)

### Cold-build environment

```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
cargo build --release --manifest-path src/rust/Cargo.toml --package memexd --package wqm-cli
```

Cold build: 30-50 min (LTO + codegen-units=1). Use incremental `cargo build --release` for iteration or temporarily disable LTO.

### Git / branch discipline

- All v0.1.3 work commits to `dev` branch
- Merge `dev` → `main` at release (Task 16) — user approval required before merge
- Atomic commits per Git Discipline (CLAUDE.md §IV)
- Push to origin every 5-10 commits

### File layout expected for v0.1.3

```
docker/
├── Dockerfile.memexd           (task 1)
├── Dockerfile.mcp              (task 1)
├── compose/
│   ├── minimal.yml             (task 3)
│   ├── full-stack.yml          (task 3)
│   ├── standalone-memexd.yml   (task 3)
│   └── standalone-mcp.yml      (task 3)
├── prometheus/
│   ├── prometheus.yml          (task 4)
│   └── alerts.yml              (task 9)
├── grafana/
│   ├── provisioning/
│   │   └── dashboards/dashboards.yml   (task 8)
│   └── dashboards/
│       ├── claude-mcp.json     (task 8)
│       ├── qdrant.json         (task 8)
│       ├── memexd.json         (task 8)
│       └── system-overview.json (task 8)
├── otel/
│   └── otel-collector-config.yml       (task 14)
└── docs/
    ├── README.md               (task 12)
    ├── full-stack.md           (task 12)
    ├── minimal.md              (task 12)
    ├── standalone.md           (task 12)
    ├── telemetry.md            (task 12)
    └── dashboards.md           (task 12)
```

Note: existing `docker/docker-compose.yml` (418 lines, Python-based) will need integration review before touching — may be legacy, leave alone and build fresh structure in `docker/compose/`.

### Testing strategy

- All 3518 existing tests must still pass after each task
- Task 5 adds new TS tests for MCP instrumentation
- Task 7 adds Rust tests for queue age gauge
- Task 10 adds unit tests for detect.rs (mock docker/pid/grpc)
- Task 11 adds unit tests for service command branching
- Task 15 is the end-to-end integration test (full-stack compose boot → MCP scratchpad → verify Prometheus targets)

## Unresolved items (watch for)

- **Existing `docker/docker-compose.yml` (Python-based, 418 lines)**: decide whether to retire or keep. Task 14 notes: new telemetry services go into `docker/compose/full-stack.yml`, not the legacy file. Verify during Task 3.
- **Docker image naming consistency**: `chrisgve/memexd` vs `chrisgve/workspace-qdrant-mcp` — confirm with user before Task 2 pushes first image.
- **MCP HTTP transport**: explicitly out-of-scope for 0.1.3 (per Non-Goals). Stdio + OTLP push is the pattern. Don't get tempted to add HTTP mode mid-task.

## Task-master quick reference

```bash
# Switch tag (MCP tool)
mcp__task-master__use_tag name=v0-1-3 projectRoot=/Users/chris/dev/projects/mcp/workspace-qdrant-mcp

# Next task
mcp__task-master__next_task projectRoot=...

# Show task
mcp__task-master__get_task id=1 projectRoot=...

# Mark in-progress → done
mcp__task-master__set_task_status id=1 status=in-progress projectRoot=...
mcp__task-master__set_task_status id=1 status=done projectRoot=...

# Add implementation notes to subtask
mcp__task-master__update_subtask id=1.2 prompt="note" projectRoot=...
```

## Git state

```
branch: dev (fresh from main, 0 commits ahead)
main:   9db5f0cdb docs: session 11 close handover (v0.1.2 done, v0.1.3 PRD drafted)
tag:    v0.1.2 on origin (unchanged)
```

No uncommitted changes. Safe to start Task 1.
