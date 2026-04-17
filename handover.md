# Handover — 2026-04-18 (Session 11 close → Session 12 start)

## What to do first in the next session

1. Read the PRD: `.taskmaster/docs/20260418-0033_project_0.1.3_PRD_docker-telemetry-dashboards.md`
2. Run `task-master add-tag v0.1.3 --copy-from-current` (or use existing tag framework — check first with `task-master list-tags`)
3. Run `task-master parse-prd .taskmaster/docs/20260418-0033_project_0.1.3_PRD_docker-telemetry-dashboards.md --append` (append to avoid overwriting smart-processing tag)
4. Expand tasks, then begin execution per CLAUDE.md plan execution protocol

## Current state (end of session 11)

- Branch `main`, pushed to origin
- v0.1.2 released (tag on origin): scratchpad tags bug fix + full TUI completion + test alignment
- Schema v34 on daemon
- All tests green: 213 common + 2197 core + 143 grpc + 541 cli + 424 TS = **3518 passing**, 0 failing, 2 TS skipped
- smart-processing task-master tag: 48 done / 6 cancelled / 0 pending / 0 blocked
- Daemon running via launchctl (`pid` may differ — just check `pgrep memexd`)

## What was done in session 11

### Tags bug fix validated end-to-end
The `c123aed79 fix(common): handle stringified JSON array in scratchpad tags` commit was made in session 10 but the binary was NOT rebuilt. Session 11 rebuilt + deployed:
- `src/rust/target/release/memexd` mtime = Apr 17 12:38
- Deployed to `~/.local/bin/memexd` via `cp`
- Restarted via launchctl
- Submitted 5 test scratchpad entries via MCP (empty title, empty tags, Unicode, 10KB long content, session marker) — all processed
- Re-ran failed session-10 test item `b24d714d` (payload with stringified tags) — processed in 1025ms with new binary. Tags fix confirmed in production.

### GitHub label fix
Dependabot PR #56 flagged: missing labels `typescript` and `rust`. Created both via `gh label create`. Backfilled PR #56 with `typescript`. Future dependabot PRs will find labels on first attempt.

### v0.1.2 release
- v0.1.1 tag already existed on origin at commit 96471381 (initial scratchpad_mirror release) — not moved
- Bumped all Cargo.toml files + MCP package.json to 0.1.2
- `cargo update --offline` refreshed Cargo.lock
- Tag `v0.1.2` created + pushed
- Commit `0c159a76d chore: bump version to 0.1.2`

### Commits this session
- `420638221 test(ts): align mockDaemonClient with current DaemonClient interface`
- `d8de6c09a chore: bump version to 0.1.1` (superseded)
- `0c159a76d chore: bump version to 0.1.2`
- `a2badee65 docs: update handover for session 11 (v0.1.2 release)`

## Bugs discovered (not yet filed as issues)

1. **`wqm rules add` gRPC rejection:**
   ```
   status: InvalidArgument
   message: "Document ID must be a valid UUID"
   ```
   CLI falls back to enqueue and it processes via queue, but gRPC direct path is broken. Daemon expects UUID but CLI passes label.

2. **Legacy rules have no scope/label payload fields:**
   The 3 rules currently in Qdrant (stored via a pre-schema path) have `scope: None`, `label: None` in the payload. The `RULE label:X scope:Y` info is only embedded in the `content` text. This causes:
   - `wqm rules list` shows empty Label column
   - `wqm rules inject` SessionStart hook exits 0 with no output (filter requires `scope` field)
   Needs a backfill or inject-side fallback that parses content.

3. **Reconciliation is expensive on daemon startup:**
   For projects with many files (thales: 93K), `ignore_sync` walks filesystem + enqueues every missing file one-by-one. Each INSERT takes 1-8s under SQLite lock contention → can block gRPC readiness for 5+ minutes. Fix: batch-insert, or defer reconciliation to background task so gRPC serves immediately.

## Session 12 focus

**Primary: v0.1.3 per the PRD above** — Docker publishing (Hub + GHCR), integration into user's main-docker stack, telemetry pipeline, 4 Grafana dashboards, wqm CLI docker-awareness.

**Secondary (file as GitHub issues early in session 12, link to v0.1.3 milestone if related or defer to v0.1.4):**
- Bug 1 (rules add UUID)
- Bug 2 (legacy rules backfill)
- Bug 3 (reconciliation batching)

## Key context for next session

### User's main-docker stack
`/Users/chris/dev/tools/main-docker/docker-compose.yml` — already contains:
- `qdrant` on 6333/6334, volumes to `~/.local/share/claude/persistent_memory/qdrant/*`
- `otel-collector` on 4317/4318/8889/13133, config at `/Users/chris/dev/tools/main-docker/otel-config.yaml`
- `prometheus` on 9090, retention 360d
- `grafana` on 3000, anonymous Admin enabled
- `watchtower` daily checks with cleanup

Session 12 will add `memexd` + `workspace-qdrant-mcp` to this same compose file (or via compose override), with telemetry scraping configured in `prometheus.yml`.

### Docker Hub secrets
User added `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` to GitHub repo secrets. Ready for workflow.

### Daemon Prometheus infrastructure (already built)
- `src/rust/daemon/core/src/monitoring/metrics_core.rs` — counter/histogram/gauge
- `src/rust/daemon/core/src/monitoring/metrics_server.rs` — HTTP endpoint
- `src/rust/daemon/memexd/src/background.rs:30` — `start_metrics_server(port)` wiring
- `--metrics-port` CLI flag exists in `startup.rs`

Just needs compose to pass the flag + prometheus scrape config to enable.

### MCP server has NO Prometheus instrumentation yet
`src/typescript/mcp-server/src/` — no `prom-client`, no metrics. Session 12 adds:
- `src/typescript/mcp-server/src/telemetry/metrics.ts` — prom-client registry + tool instrumentation wrapper
- `src/typescript/mcp-server/src/telemetry/http-server.ts` — `/metrics` endpoint (docker/HTTP mode)
- OTLP exporter fallback for stdio mode

## Build environment
```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
cargo build --release --manifest-path src/rust/Cargo.toml --package memexd --package wqm-cli
```

Note: cold release builds take 30-50 min due to LTO + codegen-units=1. Use `cargo build --release` incrementally, or temporarily disable LTO for fast iteration during dev.

## Task-master state
- Current tag: `smart-processing` (all 48 + 6 cancelled = 54 tasks resolved)
- For v0.1.3: use `task-master add-tag 0.1.3 --copy-from=smart-processing` then parse the new PRD with `--append`
