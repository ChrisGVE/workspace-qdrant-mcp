# Handover — 2026-04-20 (session 4)

## Current work

Session 4 shipped the `issue-63` branch/worktree audit end-to-end. All 10
audit tasks marked done in task-master, issue #63 closed, one bug found
during the audit filed as #69 and **fixed** in the same session. Repo is
clean (handover.md is the only pending change).

## Task-master

- **Tag/milestone**: `issue-63` (complete, 10/10). Previous tag
  `issue-64` also complete (12/12).
- **In progress**: none.
- **Blocked**: none.
- **Recently completed this session**:
  - `issue-63/1` — `GitFixtures` builder in
    `src/rust/daemon/shared-test-utils/src/git_fixtures.rs` (9 smoke
    tests). 9 scenarios: plain_clone, no_remote, detached_head,
    mid_rebase, shallow_clone, multiple_clones(n), worktree,
    nested_worktree, with_submodule.
  - `issue-63/2..9` — 16 integration tests in
    `src/rust/daemon/core/tests/branch_worktree_audit.rs` driving each
    scenario through `detect_git_status`, `find_main_worktree_path`,
    `BranchLifecycleDetector`, `ProjectIdCalculator`,
    `DisambiguationPathComputer`, and `PathValidator`.
  - `issue-63/10` — report in
    `docs/specs/19-branch-worktree-audit.md` (14 ok, 5 gaps, 1 bug
    fixed).
- **Available next tags**:
  - `docker` — 15 tasks — docker compose + MCP HTTP transport (not
    started, has pending decisions — see below).
  - No other pending audit work.

## Resume instructions

1. Read this file, then `git log --oneline origin/main -20` for recent
   commits.
2. If picking up `docker`: `task-master use-tag docker`. **Blocked** on
   four decisions (see Pending Decisions). Ask the user before starting.
3. If addressing open bugs from the audit gaps (see §§1.6, 2.4, 3.4, 4.2,
   4.3 in `docs/specs/19-branch-worktree-audit.md`): pick the gap, file
   an issue, add a follow-up task.
4. If addressing #66 (TENANT_GLOBAL constant): start in
   `src/rust/common/src/constants.rs`; add
   `pub const TENANT_GLOBAL: &str = "global";` and sweep the ~15 literal
   sites listed in the issue. Mirror in TypeScript. Add a one-shot UPDATE
   in `rules_backfill.rs` to coerce `_global`/`_global_` rows.
5. If addressing #65 / #68 (rules add UUID error) or #70 (register-claims-
   success-but-not-listed): user-reported bugs, repro + fix.
6. `task-master next` in the active tag.

## Pending decisions

_(None blocking for issue-63/-64 — both fully shipped.)_

- **Docker image distribution** (for `docker` tag, Phase 2): GHCR vs
  local-build. PRD recommends GHCR. Ask before starting `docker` Phase 2.
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
| f60883bdf | test(shared-test-utils): add GitFixtures builder for branch/worktree audit |
| 29f09d222 | test(monitoring): add otlp_headers and otlp_protocol to OtelConfig literal (compile fix) |
| f95bcc37c | test(core): add branch/worktree audit integration tests + findings doc |
| bf6682b09 | fix(daemon): correlate atomic branch rename in BranchLifecycleDetector (closes #69) |
| a latest  | docs(audit): fix #69 issue URL in branch-worktree audit spec |

### GitHub issues

- **#63** — CLOSED (this session) — branch/worktree audit.
- **#69** — CLOSED (this session) — atomic-rename misclassification,
  fixed in bf6682b09.
- **#64** — CLOSED (previous session) — daemon telemetry.
- **#66** — OPEN — extract `TENANT_GLOBAL` constant (chore).
- **#65** / **#68** — OPEN — rules add UUID error.
- **#70** — OPEN — `wqm project register` claims success but project
  not in list.
- **#61** — OPEN — re-enable linux/arm64 in docker-publish workflow.
- **#60** — OPEN — linux-native idle detection.
- **#59** — OPEN — reconciliation on daemon startup blocks gRPC
  readiness.
- **#58** — OPEN — legacy rules missing scope/label payload fields.

### Audit finding summary

`docs/specs/19-branch-worktree-audit.md` has the full record. 16
integration tests pass. Remaining gaps captured in the report, each
with a follow-up sketch (full-stack orphan cleanup, queue dedup under
rapid switches, symlink canonicalisation, submodule auto-discovery
integration, shallow-clone reflog parsing). Two ADR notes on
ambiguous invariants (symlink clone identity, worktree tenant
equivalence).

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
- `BranchLifecycleDetector::scan_for_changes` order is now
  delete → new → expire; this order is required for rename correlation.

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

### New reference files (session 4)

- Fixtures: `src/rust/daemon/shared-test-utils/src/git_fixtures.rs`
  (depends on `git2` + `git` CLI for worktree/rebase/submodule ops).
- Audit tests: `src/rust/daemon/core/tests/branch_worktree_audit.rs`.
- Audit doc: `docs/specs/19-branch-worktree-audit.md`.
- Branch lifecycle fix: `src/rust/daemon/core/src/git/branch_lifecycle/detector.rs::scan_for_changes`.
