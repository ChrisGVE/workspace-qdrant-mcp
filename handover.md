# Handover — 2026-04-20 (session 5)

## Current work

Session 5 cleared two rounds of open-issue work on `main`:

1. **#66 — `TENANT_GLOBAL` sentinel** (commit `55f215ba7`). Extracted a
   shared constant in `wqm-common::constants` and
   `src/typescript/mcp-server/src/constants/tenants.ts`. Swept
   production-code `"global"` tenant/scope literals across CLI, TUI,
   daemon, and MCP server (tests and type-union positions intentionally
   kept literal). Added a one-shot startup UPDATE in
   `startup/rules_backfill::coerce_legacy_global_values` that coerces
   any historical `_global`/`_global_` rows in `rules_mirror` and
   `scratchpad_mirror` to the canonical `global` sentinel.
2. **#57 / #65 / #68 — rules-add UUID error** (commit `6a99f2086`). CLI
   (`src/rust/cli/src/commands/rules/add.rs`) and TS MCP
   (`src/typescript/mcp-server/src/tools/rules-mutations.ts::addRule`)
   now pass a generated UUID as `document_id` instead of the label. The
   label is still carried in payload metadata (already indexed); the
   SQLite mirror is keyed by label so rebuild-from-Qdrant stays
   consistent. Updated `tests/tools/rules-crud.test.ts` assertion — the
   `label` field in the response is now the user-supplied label, not
   the daemon-assigned UUID.

All tests pass (Rust touched modules + full TS suite, 470/470). Tree
clean, pushed to `origin/main`.

## Task-master

- **Tag/milestone**: `issue-63` is still the active tag (complete,
  10/10). No new task-master tasks created for this session's work —
  both items were standalone chore/bug commits driven directly from
  GitHub issues.
- **In progress**: none.
- **Blocked**: none (except `docker` — still pending the 4 user
  decisions, see Pending Decisions).
- **Available next tags**: `docker` only (blocked).

## Resume instructions

1. Read this file, then `git log --oneline origin/main -10` for recent
   commits.
2. If picking up open GitHub issues — all remaining are user-reported
   bugs or heavier work that needs a decision or repro:
   - **#58** — legacy rules missing scope/label payload fields. Needs
     a `wqm admin rebuild rules-payload` backfill command +
     inject-side fallback. Medium scope; fix options already written
     in the issue.
   - **#59** — reconciliation on daemon startup blocks gRPC readiness.
     Timing/startup-ordering bug; requires daemon boot flow investigation.
   - **#60** — linux-native idle detection for adaptive resource
     management. Linux-specific feature work.
   - **#61** — re-enable `linux/arm64` in docker-publish workflow.
     Likely blocked on the `docker` tag decisions (see below) since the
     compose/transport story is still open.
   - **#70** — `wqm project register` claims success but the project
     does not appear in `wqm project list` afterward. Likely
     persistence/commit issue in the register path; needs end-to-end
     trace from CLI → daemon → SQLite.
   - **#55** — daemon client reports "Client not connected" despite
     daemon healthy. Connection-pool / reconnect behavior.
3. If picking up `docker`: `task-master use-tag docker`. **Blocked** on
   four decisions (see Pending Decisions). Ask the user before starting.
4. If addressing audit gaps from session 4 (§§1.6, 2.4, 3.4, 4.2, 4.3
   in `docs/specs/19-branch-worktree-audit.md`): pick the gap, file a
   GitHub issue, add a follow-up task.
5. `task-master next` in the active tag — expect "no tasks
   available" outside `docker`.

## Pending decisions

_(None blocking the session-5 commits — both shipped.)_

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
| `55f215ba7` | `refactor(common): extract TENANT_GLOBAL sentinel constant` (closes #66) |
| `6a99f2086` | `fix(rules): generate UUID for document_id on rules add` (closes #57, #65, #68) |

### GitHub issues touched this session

- **#57** — CLOSED — CLI/MCP now send a valid UUID.
- **#65** — CLOSED — duplicate of #57, fixed by the same commit.
- **#66** — CLOSED — `TENANT_GLOBAL` sentinel shipped.
- **#68** — CLOSED — duplicate of #57, fixed by the same commit.

### Still OPEN after this session

- **#55** — daemon client reports "Client not connected".
- **#58** — legacy rules missing scope/label payload fields.
- **#59** — reconciliation blocks gRPC readiness.
- **#60** — linux-native idle detection.
- **#61** — re-enable `linux/arm64` in docker-publish workflow.
- **#70** — `wqm project register` success but not listed.

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
- **New (session 5)**: `TENANT_GLOBAL = "global"` is the one-and-only
  sentinel for the global-scope `tenant_id` payload field. Add future
  writes through `wqm_common::constants::TENANT_GLOBAL` (Rust) or
  `./constants/tenants.ts` (TS). Test fixtures keep the literal on
  purpose so they catch drift if the constant is ever changed.
- **New (session 5)**: Rules `document_id` in Qdrant is always a UUID.
  The human-readable label lives in `payload.label` and is the
  SQLite `rules_mirror` primary key. Backfill from Qdrant uses
  `payload.label` → `rules_mirror.rule_id`, so live writes must keep
  the same invariant (mirror keyed by label, Qdrant keyed by UUID).

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
  introduced by sessions 4 or 5.

### Reference files (session 5)

- `src/rust/common/src/constants.rs` — `TENANT_GLOBAL` added.
- `src/rust/daemon/core/src/startup/rules_backfill.rs` —
  `coerce_legacy_global_values` helper + unit test
  (`test_coerce_legacy_global_values`).
- `src/typescript/mcp-server/src/constants/tenants.ts` — TS mirror of
  the constant (uses `as const` so it narrows to the literal type).
- `src/rust/cli/src/commands/rules/add.rs` — UUID generation for
  `document_id`.
- `src/typescript/mcp-server/src/tools/rules-mutations.ts` — UUID
  generation for `document_id`; mirror keyed by label; response
  returns the input label rather than the daemon-assigned UUID.
