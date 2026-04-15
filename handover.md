# Handover — 2026-04-15 (Session 8 complete)

## Current State

Branch `main`, pushed to remote. Tag `smart-processing`. Qdrant back up.

## What Was Done (Session 8)

### Test Fixes (5 Rust + 20 TS pre-existing failures)
- `CREATE_WATCH_FOLDERS_SQL` missing `is_worktree` + `main_worktree_watch_id` columns (v31 added them, base DDL never updated)
- `test_empty_file_handling` expected Ok but `process_file` returns `EmptyFile` error
- gRPC integration tests: accept `FailedPrecondition` when Qdrant unavailable
- TS: `enqueueUnified` changed sync→async (daemon gRPC); rewrote 6 test files with mock daemon client, async patterns, connectivity error matching

### Task 4: Ignore File Reconciliation (ALL 8 SUBTASKS DONE)
1. ✅ Subtask 1 (prior session) — `!pattern` + `- pattern` negation
2. ✅ Subtask 2 — `ProjectIgnoreMatcher::for_dir(dir, project_root)` parent cascade. Walks ancestors from root→dir accumulating .gitignore/.wqmignore rules. 5 new tests. Caller in `scan.rs` passes `None` (backward compat).
3. ✅ Subtask 3 — `patterns/eligibility_trie.rs`: HashMap<PathBuf, EligibilityStatus> built from WalkBuilder. `.add_custom_ignore_filename(".gitignore")` for non-git dirs. 5 tests.
4. ✅ Subtask 4 — `startup/reconciliation/ignore_sync.rs`: walks project tree, diffs vs tracked_files, enqueues file/delete (stale) and file/add (missing). 4 tests.
5. ✅ Subtask 5 — v34 migration: `ignore_file_mtimes` table. `ignore_mtime.rs` with get/set/clear helpers. 5 tests.
6. ✅ Subtask 6 — `reconcile_all_ignore_rules()` in reconciliation/mod.rs, called from `database.rs::run_reconciliation` after stale cleanup.
7. ✅ Subtask 7 — `watching_queue/ignore_watch.rs`: intercepts .gitignore/.wqmignore Create/Modify events in `process_file_event`, compares mtime, triggers reconciliation.
8. ✅ Subtask 8 — Watcher calls `reconcile_ignore_rules` directly (not via queue op) for immediate stale cleanup.

### Tasks 11-16: Scratchpad Rebuild (ALL DONE)
Code was already implemented. Verified: `scratchpad_rebuild.rs` exists, wired into `mod.rs`, dispatch handles "scratchpad" and "all", CLI has `Scratchpad` variant. Marked done.

### Tasks 17, 21-23: Build, Docs, Tests (ALL DONE)
- Release binaries built and deployed to `~/.local/bin/`
- API reference updated
- All test suites green: core 2593, gRPC 143, CLI 494, TS 424 = **3654 pass, 0 fail**

## What Remains

### Task 18: Restart daemon (Qdrant now available)
```bash
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
sleep 2
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
```
Verify: `wqm service status`, check schema_version = 34 (v33 scratchpad_mirror + v34 ignore_file_mtimes).

### Task 19: E2E scratchpad store + rebuild test
1. `wqm store` a scratchpad entry via MCP
2. Verify in scratchpad_mirror: `sqlite3 ~/.local/share/workspace-qdrant/state.db 'SELECT * FROM scratchpad_mirror'`
3. Verify in Qdrant scratchpad collection
4. Delete Qdrant point, run `wqm rebuild scratchpad`, verify restored
5. Run `wqm rebuild all`, verify scratchpad included

### Task 20: Scratchpad edge cases
Empty title, empty tags, unicode content, long content (10KB+), concurrent stores, rebuild with empty mirror, rebuild when collection missing.

### Task 24: Tag v0.1.1
After tasks 18-20 verified. Annotated tag + push.

## Build Environment Note
Rust builds require `LIBRARY_PATH` for clang_rt:
```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
```

## Key New Files
- `src/rust/daemon/core/src/patterns/gitignore.rs` — parent cascade
- `src/rust/daemon/core/src/patterns/eligibility_trie.rs` — new
- `src/rust/daemon/core/src/startup/reconciliation/ignore_sync.rs` — new
- `src/rust/daemon/core/src/ignore_mtime.rs` — new
- `src/rust/daemon/core/src/schema_version/v34.rs` — new migration
- `src/rust/daemon/core/src/watching_queue/ignore_watch.rs` — new
- `src/rust/daemon/core/src/watching_queue/file_watcher_ops.rs` — intercept added
- `src/rust/daemon/memexd/src/database.rs` — startup integration

## Test Counts
Core: 2593 pass, 16 ignored | gRPC: 143 | CLI: 494 | TS: 424+2 skip = **3654 pass, 0 fail**
