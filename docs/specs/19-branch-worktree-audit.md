# 19 — Branch / Worktree Audit

Audit scope: GitHub issue [#63](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues/63).

This document consolidates findings from the branch- and worktree-handling
audit. Per-scenario tests live in
`src/rust/daemon/core/tests/branch_worktree_audit.rs`. Deterministic git
fixtures are in `src/rust/daemon/shared-test-utils/src/git_fixtures.rs`.

Each finding has a **Status** (`ok` / `gap` / `bug`), **Evidence** (test
references + code paths), and, for bugs and gaps, a follow-up action.

---

## 1. Worktree Detection

### 1.1 Plain clone is not a worktree

- **Status**: ok
- **Evidence**: `task2_plain_clone_detects_as_main_repo`
  (`branch_worktree_audit.rs`). `detect_git_status`
  (`git/types.rs:63`) returns `is_worktree=false`, `branch="main"`,
  `commit_hash=Some(_)` on a repo whose `.git` is a directory.

### 1.2 Linked worktree is flagged

- **Status**: ok
- **Evidence**: `task2_linked_worktree_flagged_as_worktree`. For a checkout
  created via `git worktree add`, the `.git` entry is a file and
  `detect_git_status` returns `is_worktree=true` with `branch` matching the
  linked branch name.

### 1.3 Nested worktree is flagged

- **Status**: ok
- **Evidence**: `task2_nested_worktree_still_detects_as_worktree`. A
  worktree created inside another worktree checkout still sets
  `is_worktree=true`; both linked checkouts resolve to the main repo.

### 1.4 commondir resolution back to main repo

- **Status**: ok
- **Evidence**: `task2_worktree_commondir_resolves_to_main`.
  `find_main_worktree_path` (`git/worktree.rs:24`) correctly canonicalises
  the `commondir` chain and returns the main working-tree path.

### 1.5 Detached HEAD in a worktree

- **Status**: ok
- **Evidence**: `task2_detached_head_uses_short_sha_for_branch`. In a
  detached-HEAD state, `detect_git_status` emits the first eight characters
  of the commit SHA as the branch label and keeps `commit_hash`.

### 1.6 Orphan cleanup when a worktree path disappears

- **Status**: ok (pure-function test); **gap** for full-stack behavior.
- **Evidence**: `task3_pathvalidator_flags_missing_worktree_after_grace`.
  `PathValidator::validate_projects`
  (`watching/path_validator.rs:147`) returns the orphan after two passes
  even with `grace_period_minutes=0`, confirming the pending→confirmed
  state machine.
- **Gap**: end-to-end wiring from the daemon's reconciliation tick through
  `OrphanCleanupActions::sqlite_cleanup_statements` / `qdrant_tenant_filter`
  into archived `watch_folders` rows was not exercised in this audit. Covered
  by existing unit tests but not by a single daemon-level integration test.
  Follow-up: add such a test under `core/tests/daemon_integration/`.

---

## 2. Branch Lifecycle

### 2.1 Atomic branch rename

- **Status**: **bug** — filed as [#69](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues/69), fixed in the same session (see §6 summary).
- **Evidence**: `task4_branch_rename_emits_renamed_event_within_timeout`.
  After `git branch -m main trunk`, `BranchLifecycleDetector::scan_for_changes`
  (`git/branch_lifecycle/detector.rs:193`) emits
  `Created { branch: "trunk" }` + `DefaultChanged { main → trunk }`, and (in
  a later scan) `Deleted { branch: "main" }` once the rename-correlation
  timeout expires. It should emit a single
  `Renamed { old_name: "main", new_name: "trunk" }`.
- **Root cause**: in `scan_for_changes`, `detect_new_branches` runs before
  `detect_deleted_branches`, so the rename-correlation lookup operates on an
  empty pending-delete list. The new branch is classified as `Created`
  immediately, and the paired delete never gets a chance to correlate.
- **Fix sketch**: swap the call order so `detect_deleted_branches` populates
  `pending` first, then `detect_new_branches` checks it for commit-hash
  matches, then `emit_expired_deletes` drains anything left.
- **Follow-up**: fix + flip the assertion in the audit test back to expect a
  single `Renamed`.

### 2.2 Branch deletion (no rename pairing)

- **Status**: ok
- **Evidence**: `task5_branch_deletion_emits_deleted_after_rename_timeout`.
  Deleting a branch without a matching create within
  `rename_correlation_timeout_ms` yields exactly one `BranchEvent::Deleted`
  event after the timeout expires. No duplicate deletes on repeat scans.

### 2.3 Default branch change via HEAD

- **Status**: ok
- **Evidence**: `task6_default_branch_change_via_head_rename_is_detected`.
  When the current branch is renamed, the detector reads `.git/HEAD`
  (`git/branch_lifecycle/detector.rs:141`) and emits
  `DefaultChanged { main → trunk }` on the next scan.
- **Caveat**: `DefaultChanged` also fires as a side effect of a rename
  (see §2.1). Once #69 is fixed the event should remain even after the
  rename is re-classified, because the default tracking is independent of
  the create/delete correlation path.

### 2.4 Rapid branch switches

- **Status**: ok (post-switch state); **gap** (queue-level assertions).
- **Evidence**: `task7_rapid_branch_switch_lands_on_final_branch`.
  After rapid `checkout feature && checkout main && checkout feature &&
  checkout main`, `detect_git_status` reports the final branch `main`.
- **Gap**: the debouncer / queue-dedup claims in the task spec
  ("each file path appears exactly once, all queue items have branch=main")
  require a live daemon with notify-debouncer-full running. Not covered by
  pure-function tests. Follow-up: add a file-watcher integration test
  under `core/tests/` that stands up an `EnhancedFileWatcher`, triggers the
  switch sequence, and asserts unified_queue rows.

---

## 3. Multi-Clone Disambiguation

### 3.1 Shared remote hash, distinct tenant_ids

- **Status**: ok
- **Evidence**: `task8_multiple_clones_share_remote_hash_but_get_distinct_ids`.
  `ProjectIdCalculator::calculate_remote_hash` is stable across clones of
  the same remote; `DisambiguationPathComputer::recompute_all` produces a
  unique disambiguation path per clone; `ProjectIdCalculator::calculate`
  yields three distinct tenant_ids for three sibling clones under a common
  ancestor.

### 3.2 No-remote fallback

- **Status**: ok
- **Evidence**: `task8_no_remote_yields_local_prefixed_id`. Without a
  configured remote, `calculate` returns an id prefixed with `local_`
  derived from the canonical path.

### 3.3 Disambiguation determinism

- **Status**: ok
- **Evidence**: `task8_two_independent_clones_recomputed_get_stable_disambig`.
  Two calls to `recompute_all` on the same path set yield identical
  mappings — daemon restarts re-register clones without flapping
  tenant_ids.

### 3.4 Symlink / bind-mount canonicalisation

- **Status**: **gap** (not tested in this audit pass).
- **Evidence**: not covered.
- **Follow-up**: add a fixture variant that creates a symlink to an
  existing clone and verifies `ProjectIdCalculator::calculate` canonicalises
  both paths to the same id (or flags it as a distinct disambiguation
  target — the expected behavior is ambiguous; see ADR note below).

---

## 4. Cross-Cutting Cases

### 4.1 Mid-rebase state

- **Status**: ok
- **Evidence**: `task9_mid_rebase_still_reports_is_git`. With
  `.git/rebase-apply/` present, `detect_git_status` still returns
  `is_git=true`; HEAD is detached during rebase, branch is short SHA or
  `"HEAD"`, no panic.

### 4.2 Submodule

- **Status**: ok
- **Evidence**: `task9_submodule_has_its_own_git_pointer_and_distinct_tenant`.
  Submodule checkout has its own `.git` (gitlink). When calculated with the
  submodule's own remote URL, its tenant_id differs from the parent repo's.
- **Gap**: the audit test builds the submodule manually; the daemon's
  submodule auto-discovery path
  (`diff_tree::ls_tree_submodules` → `parent_watch_id` wiring in
  `daemon_state`) was not exercised end-to-end in this pass. Covered by
  `daemon_state/tests/submodule_tests.rs` unit tests, but not by a fixture-
  driven integration test.

### 4.3 Shallow clone

- **Status**: ok
- **Evidence**: `task9_shallow_clone_is_git_and_has_commit`. A
  `--depth=1` clone still exposes HEAD commit, branch `"main"`, and the
  `.git/shallow` marker file. No panics in `detect_git_status`.
- **Note**: reflog parsing (`git/reflog.rs`) was not exercised against a
  shallow clone in this pass. Listed as a gap if the daemon consumes
  reflog history on such repos.

---

## 5. ADR Notes (ambiguous invariants)

### ADR note A — should a symlinked clone share the tenant_id of the target?

Two reasonable interpretations:

1. **Canonical path dominance**: a symlink and its target are the same
   project; both paths yield the same tenant_id (requires canonicalisation
   before disambiguation).
2. **Observed path dominance**: each registration path is treated as a
   distinct clone, disambiguated accordingly.

Current code calls `canonicalize()` only in the no-remote fallback
(`project_id/calculator.rs:52`), so for remote-backed clones the observed
path is used. Docker bind-mounts will trigger disambiguation even when
they resolve to the same underlying inode. The daemon should pick one
rule and document it. Recommendation: canonicalise in both branches and
treat symlinks as aliases.

### ADR note B — worktree vs main-repo tenant equivalence

The `worktree_tests.rs` unit tests assert that worktrees share the main
repo's tenant_id. This is correct when worktrees share a remote, but the
audit shows nothing enforces this at the registration layer — the
invariant relies on the call site passing the main repo's tenant_id down
into the worktree registration. If a caller forgets, the worktree ends
up with its own disambiguated tenant_id and appears as a separate
project. Recommendation: add a registration-time check in
`register_project_with_disambiguation` that, when `is_worktree=true`, looks
up `main_worktree_watch_id.tenant_id` and forces the same value.

---

## 6. Summary

| Category | ok | gap | bug |
|----------|----|-----|-----|
| 1. Worktree Detection | 5 | 1 | 0 |
| 2. Branch Lifecycle | 3 | 1 | 1 |
| 3. Multi-Clone Disambiguation | 3 | 1 | 0 |
| 4. Cross-Cutting Cases | 3 | 2 | 0 |
| **Total** | **14** | **5** | **1** |

Bugs filed:

- **#69** — BranchLifecycleDetector misclassifies atomic rename as Created+Deleted. **Fixed** in the same session by reordering `scan_for_changes` (delete → new → expire). The audit test now asserts a single `Renamed` event.

Gaps captured as follow-up tasks (see §§1.6, 2.4, 3.4, 4.2, 4.3).

---

## 7. Follow-up audit (MCP/host-side, 2026-05)

A second audit, focused on the MCP-server and host-script layers (the
original scope was the Rust daemon), surfaced seven additional issues.
All seven are now resolved.

### 7.1 MCP TS — `isGitRepository` rejects linked worktrees

- **Status**: **bug** — fixed.
- **Evidence**: `src/typescript/mcp-server/src/utils/git-utils.ts:11` required
  `.git` to be a directory via `statSync(...).isDirectory()`. Linked worktrees
  store `.git` as a regular file (`gitdir: ...`) and were rejected, so any
  agent operating inside a worktree saw the MCP report "not a repo".
- **Fix**: switched to `existsSync` to accept both forms.

### 7.2 MCP TS — `getGitRemoteUrl` cannot follow worktrees

- **Status**: **bug** — fixed.
- **Evidence**: `git-utils.ts:45` opened `<repo>/.git/config` directly, which
  does not exist in a linked worktree (the `.git` file points to the parent
  repo's git dir). Always returned `null` for worktrees, which propagated to
  the daemon's `tenant_id` calculation and produced `local_*` IDs for what
  should have been git-tracked projects.
- **Fix**: shell out to `git -C <repo> config --get remote.origin.url` so the
  gitdir indirection resolves automatically.

### 7.3 MCP TS — `Rust CanonicalPath` rejects Windows drive paths

- **Status**: **bug** — fixed.
- **Evidence**: `src/rust/common/src/paths/normalize.rs:56` rejected any input
  not starting with `/`, including `C:/Users/...` and `C:\Users\...`. Every
  Windows-host MCP call was returning `INVALID_ARGUMENT: path must be
  absolute`. The TS mirror in `src/typescript/mcp-server/src/common/paths.ts`
  had the same defect.
- **Fix**: `normalize_path` now recognises a Windows drive prefix
  (`[A-Za-z]:/`), normalises backslashes to forward slashes, and emits the
  canonical form `C:/foo/bar`. Same change applied in TS. Cross-language
  fixtures in `tests/path-fixtures/cases.json` lock the behavior.

### 7.4 MCP TS — session start auto-registers system directories

- **Status**: **bug** — fixed.
- **Evidence**: `session-lifecycle.ts:32` used `findProjectRoot(cwd) ?? cwd`.
  When the MCP server was spawned by a GUI or service with `cwd` set to
  `C:\WINDOWS\system32` / `/` / `/usr`, the walk-up found no marker and the
  cwd itself ended up registered as a project.
- **Fix**: added a curated `SUSPICIOUS_CWD_PATTERNS` list. Detection refuses to
  walk up from a suspicious cwd and refuses to register a discovered root
  that is itself suspicious. `WQM_PROJECT_ROOT` env var provides an explicit
  override for service-style launches.

### 7.5 daemon — `register_if_new=true` skips worktree auto-register

- **Status**: **bug** — fixed.
- **Evidence**: `services/project_service/registration.rs` only ran
  `try_worktree_auto_register` on the `!register_if_new` branch and only when
  `!project_exists(project_id)`. A worktree registered via the MCP fallback
  (`register_if_new=true`) was enqueued as a brand-new tenant; a worktree
  registered when its main repo was already known reactivated the main
  watch_folder without creating a worktree-specific row.
- **Fix**: `determine_registration_action` now calls `try_worktree_auto_register`
  unconditionally before classifying the request. The function itself is
  idempotent — checks both that the main repo is registered and that the
  worktree path is not already in `watch_folders` — so calling it on the
  existing-project branch is safe.

### 7.6 MCP TS — branch/worktree never re-detected after session start

- **Status**: **bug** — fixed.
- **Evidence**: `SessionState` had no `currentBranch` field; the dispatcher
  only sent a heartbeat per tool call. Switching branches mid-session left
  the MCP returning results from the original branch (or no branch filter at
  all, depending on tool).
- **Fix**: added `currentBranch` and `lastBranchRefreshAt` to `SessionState`.
  `ensureProjectFresh()` in the dispatcher re-reads via `getCurrentBranch` and
  `isWorktree` with a 5-second TTL. `search` and `grep` tool builders now
  accept a `defaults.branch` and substitute `sessionState.currentBranch` when
  the caller omits `branch`. Agents that want cross-branch search pass
  `branch: "*"` explicitly.

### 7.7 daemon — confusing `register_session(..., "main")` parameter

- **Status**: cosmetic — fixed.
- **Evidence**: `priority_manager::register_session` and `unregister_session`
  take a second `_branch: &str` argument that has been ignored for several
  versions (underscore prefix). Call sites passed the literal `"main"`,
  which read as "this routine touches the main branch" — which it does not.
- **Fix**: renamed the unused parameter to `_session_tag` and updated the
  two production call sites (`registration.rs`, `deactivation.rs`) to pass
  `"session"`. Test fixtures keep the legacy `"main"` value; behavior is
  unchanged because the parameter is never read.
