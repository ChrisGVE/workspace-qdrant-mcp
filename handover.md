# Handover — 2026-04-20 (session 6)

## Current work

Session 6 closed four open GitHub issues on `main` and pushed them to
`origin/main`. All Rust + TypeScript tests green (workspace-qdrant-core
queue_operations 63 passed, startup 23 passed, workspace-qdrant-grpc 149
passed incl. 21 project_service, TS suite 472 passed). Release binaries
built + deployed + daemon restarted via launchctl; daemon comes up
healthy within ~10s with the new background reconciliation.

1. **#58 — legacy rules payload backfill** (commit `cd5d0aa72`). Added a
   shared `wqm_common::rules_legacy::parse_rule_header` parser, a new
   `wqm admin rebuild rules-payload` one-shot that runs in the daemon
   (`rules_payload_backfill::backfill_rules_payload`), and an
   inject-time fallback in `wqm rules inject` that scrolls the whole
   collection and synthesises payloads for any point whose label/scope
   are still only in the `content` text. `rebuild_all` now runs
   `rules-payload` ahead of `rules` so the later reconciliation scan
   sees recovered labels.

2. **#70 — project register persistence + activate empty path**
   (commit `e0f8cf2f6`). Two problems compounded:
   - `(tenant, add)` items had no explicit dequeue priority, so new
     project registrations sat behind a large `(file, add)` backlog for
     already-active projects indefinitely. Added a `(tenant, add)`
     priority bucket in `queue_operations/dequeue.rs::build_dequeue_query`
     so registrations cut the line. Regression test:
     `test_tenant_add_priority_over_file_add_backlog`.
   - `wqm project activate` sends `path=""`, and the daemon rejected
     every call with `"path cannot be empty"`. Relaxed
     `handle_register_project` to accept empty path when `project_id`
     is provided (activation flow). Both-empty still rejected. Tests:
     `test_empty_path_and_project_id_returns_error`,
     `test_empty_path_with_project_id_is_allowed_for_activation`.

3. **#55 — DaemonClient "Client not connected" never recovers**
   (commit `bcdf0b6b1`). `callWithRetry` now runs `ensureConnected()`
   on every attempt, so a stale handle (initial `connect()` failed,
   channel closed, etc.) triggers a reconnect on the next RPC rather
   than throwing forever. Added `connecting` reentrancy guard to stop
   `ensureConnected → connect → healthCheck → callWithRetry →
   ensureConnected` recursion. Expanded `isRetryableError` to catch
   `"Client not connected"`, `"channel has been closed"`,
   `"Channel has been shut down"`. Tests:
   `auto-reconnect > should attempt to connect on RPC call when never
   connected`, `should re-attempt connect after close()`.

4. **#59 — reconciliation blocks gRPC readiness** (commit `e5fd74cff`).
   Split `run_reconciliation` into a fast path (SQL-only cleanup +
   watch folder validation, runs before gRPC binds) and a new
   `spawn_background_reconciliation` (tokio::spawn from `run_daemon`
   after queue processor starts) for the slow ignore-rule diff. Added
   `QueueManager::enqueue_unified_batch` — a single-transaction bulk
   insert — and rewrote `reconcile_ignore_rules` to use it with a
   `IGNORE_SYNC_BATCH_SIZE = 500` chunking. Tests:
   `test_enqueue_unified_batch_is_single_transaction` (asserts
   `PRAGMA data_version` bumps by 1 for a 10-row batch),
   `test_enqueue_unified_batch_deduplicates`.

## Task-master

- **Tag/milestone**: `issue-63` still active (complete, 10/10). No new
  task-master tasks were created for session 6 — all four items were
  standalone chore/bug commits driven directly from GitHub issues and
  tracked via the Task tool locally.
- **In progress**: none.
- **Blocked**: none (except `docker` — still pending the 4 user
  decisions, unchanged from session 5).
- **Available next tags**: `docker` only (blocked).

## Resume instructions

1. Read this file, then `git log --oneline origin/main -10` for recent
   commits.
2. Two GitHub issues remain open:
   - **#60** — Linux-native idle detection for adaptive resource
     management. Feature work, needs a decision (systemd-logind vs
     /proc heuristic vs manual gRPC command vs Wayland/X11 bind-mount).
     Cannot be validated on macOS dev env. Ask the user which of the
     four options to implement; the implementation then lives in
     `src/rust/daemon/core/src/adaptive_resources/` behind a Linux
     `cfg` gate + `resource_limits.linux_idle_source` config flag.
   - **#61** — re-enable `linux/arm64` in docker-publish workflow.
     Still blocked on the `docker` tag decisions (GHCR vs local-build,
     token rotation UX, TLS strategy, HTTP port default). Ask before
     starting.
3. If picking up `docker`: `task-master use-tag docker`. **Blocked**
   on four decisions (see Pending decisions). Ask the user before
   starting.
4. If addressing audit gaps from session 4 (§§1.6, 2.4, 3.4, 4.2, 4.3
   in `docs/specs/19-branch-worktree-audit.md`): pick the gap, file a
   GitHub issue, add a follow-up task.
5. `task-master next` in the active tag — expect "no tasks
   available" outside `docker`.

## Pending decisions

- **#60 idle-source backend**: which of the four proposed options to
  implement. systemd-logind is the most semantically correct but adds
  a DBus socket bind-mount requirement; `/proc` heuristic is lowest
  friction; manual gRPC is simplest; Wayland/X11 mount is the most
  permissive requirement.
- **Docker image distribution** (for `docker` tag, Phase 2): GHCR vs
  local-build. PRD recommends GHCR. Ask before starting `docker`
  Phase 2.
- **Docker token rotation UX** (for `docker` tag): static `.env` vs
  admin endpoint.
- **TLS strategy for HTTP MCP** (for `docker` tag): native vs
  reverse-proxy default.
- **MCP HTTP port default** (for `docker` tag): `6335` proposed;
  sign-off before codifying in compose.

## Key context

### Shipped this session (all on `main`, pushed)

| Commit | Summary |
|--------|---------|
| `cd5d0aa72` | `fix(rules): backfill payload fields for legacy RULE-headered content` (closes #58) |
| `e0f8cf2f6` | `fix(daemon): prioritize project registrations and accept activate empty path` (closes #70) |
| `bcdf0b6b1` | `fix(mcp): auto-reconnect DaemonClient when gRPC handles are stale` (closes #55) |
| `e5fd74cff` | `fix(daemon): defer ignore reconciliation and batch the enqueues` (closes #59) |

(Commit hashes above are truncated — `git log --oneline origin/main -4`
gives the exact IDs.)

### GitHub issues touched this session

- **#55** — CLOSED — auto-reconnect in DaemonClient.
- **#58** — CLOSED — `wqm admin rebuild rules-payload` + inject-time
  fallback.
- **#59** — CLOSED — background ignore reconcile + batch enqueue.
- **#70** — CLOSED — `(tenant, add)` queue priority + activate empty
  path.

### Still OPEN after this session

- **#60** — Linux-native idle detection (needs decision).
- **#61** — re-enable `linux/arm64` in docker-publish workflow
  (blocked on `docker` tag decisions).

### Architectural invariants (carried forward)

- `DaemonMetrics` is the single metrics registry — do **not** spawn a
  parallel `telemetry::metrics` module.
- Prometheus is the canonical metrics surface. OTLP carries **traces
  only** until task 10 of issue-64 is re-opened.
- Daemon-only write path: Qdrant + SQLite mutations go through the
  queue processor (ADR-003).
- 4 canonical collections: `projects` (by `tenant_id`), `libraries`,
  `rules`, `scratchpad`.
- `ProjectIdCalculator::normalize_git_url` is the one-and-only URL
  normaliser — used by daemon and CLI both.
- `BranchLifecycleDetector::scan_for_changes` order is
  delete → new → expire; required for rename correlation.
- `TENANT_GLOBAL = "global"` is the one-and-only sentinel for the
  global-scope `tenant_id` payload field. Add future writes through
  `wqm_common::constants::TENANT_GLOBAL` (Rust) or
  `./constants/tenants.ts` (TS). Test fixtures keep the literal on
  purpose so they catch drift if the constant is ever changed.
- Rules `document_id` in Qdrant is always a UUID. The human-readable
  label lives in `payload.label` and is the SQLite `rules_mirror`
  primary key.
- **New (session 6)**: legacy RULE-headered content (`RULE\nlabel:X\n
  scope:Y\n…\n---\nbody`) is parsed by
  `wqm_common::rules_legacy::parse_rule_header`. Both the daemon
  backfill (`services::rules_payload_backfill`) and the CLI inject
  fallback reuse it. Post-backfill, `content` is the body only — the
  header is stripped.
- **New (session 6)**: `(tenant, add)` gets a dedicated dequeue
  priority bucket ahead of all non-destructive ops. Scan/uplift
  tenant items do *not* get this priority (would regress #59 by
  pushing bursty scans ahead of user work).
- **New (session 6)**: `QueueManager::enqueue_unified_batch` is the
  single-transaction bulk insert. Use it for any enqueue loop where
  N ≫ 1 and items share `(item_type, op, tenant_id, collection)`. It
  still runs full payload validation and idempotency dedup per row —
  only the commit is amortised.
- **New (session 6)**: ignore reconciliation at startup runs on a
  detached `tokio::spawn`. Callers must not assume the index is
  consistent the instant `run_daemon` returns from Phase 6; the
  background task finishes asynchronously and logs
  `[startup-bg] Ignore reconciliation complete in …`.

### Gotchas carried forward

- `task-master parse-prd` takes ~3 min per PRD.
- `numTasks` is a hint, not exact.
- Task-master tag names cannot start with `#`.
- Env-mutating tests must be `#[serial]`.
- Prometheus `METRICS` is a global singleton; tests that read counters
  must use delta assertions with `>=`.
- `opentelemetry-otlp` 0.14 headers injection requires tonic-version
  alignment; `OTEL_EXPORTER_OTLP_HEADERS` env is the supported path.
- Clippy on `workspace-qdrant-core` still emits pre-existing warnings
  in `graph/algorithms/community.rs`, `graph/sqlite_store.rs`,
  `patterns/detection/detector.rs`,
  `keyword_extraction/semantic_rerank.rs`. Exit 0; deferred cleanup.
- Git index.lock occasionally lingers after background task-master
  calls — `rm -f .git/index.lock` if it blocks a commit.
- `ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib` required for
  every `cargo build` / `cargo test`.
- `wqm-cli` generates ~47 pre-existing compile warnings; none
  introduced by sessions 4, 5, or 6.
- The DaemonClient auto-reconnect retries up to `maxRetries` (default
  3) so the first user-facing error from a never-connected client is
  the underlying gRPC `UNAVAILABLE`, not `"Client not connected"`.
  Keep MCP tests that assert the latter scoped to the direct
  `grpcUnary` helper, not the high-level RPC surface.

### Reference files (session 6)

- `src/rust/common/src/rules_legacy.rs` — RULE-header parser and
  `split_scope` helper.
- `src/rust/daemon/grpc/src/services/rules_payload_backfill.rs` —
  `backfill_rules_payload` + payload/mirror upsert helpers.
- `src/rust/daemon/grpc/src/services/system_service/rebuild.rs` —
  `rebuild_rules_payload` wiring; `"rules-payload"` dispatch case;
  `rebuild_all` ordering adjusted.
- `src/rust/daemon/grpc/src/services/system_service/rpc_handlers.rs` —
  `"rules-payload"` added to `VALID_TARGETS`.
- `src/rust/cli/src/commands/rebuild.rs` — new
  `RebuildCommand::RulesPayload` CLI subcommand.
- `src/rust/cli/src/commands/rules/inject.rs` —
  `augment_with_legacy_rules` fallback.
- `src/rust/daemon/core/src/queue_operations/dequeue.rs` — new
  `(tenant, add)` priority bucket in `build_dequeue_query`.
- `src/rust/daemon/core/src/queue_operations/enqueue.rs` — new
  `enqueue_unified_batch` single-transaction bulk insert.
- `src/rust/daemon/core/src/startup/reconciliation/ignore_sync.rs` —
  `IGNORE_SYNC_BATCH_SIZE` constant + `enqueue_ignore_ops` batch
  helper.
- `src/rust/daemon/memexd/src/database.rs` — split
  `run_reconciliation` / `spawn_background_reconciliation`.
- `src/rust/daemon/memexd/src/main.rs` — launches the background
  reconciliation after Phase 6 gRPC startup.
- `src/rust/daemon/grpc/src/services/project_service/registration.rs`
  — accepts empty `path` when `project_id` supplied.
- `src/typescript/mcp-server/src/clients/daemon-client.ts` —
  `ensureConnected`, `connecting` reentrancy guard, expanded
  `isRetryableError`.
