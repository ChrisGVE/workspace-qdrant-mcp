# Handover: Exclusion Cleanup Not Deleting Points

**Date:** 2026-02-06
**Status:** One bug to investigate

## Completed Work (This Session)

1. **Fixed priority system** (commit `de7907d7`)
   - Removed `calculate_priority()` from `watching_queue.rs` (was assigning static 3/5/8 by op type)
   - Changed all `enqueue_unified()` callers across daemon, CLI, gRPC, and tests to pass `0`
   - Updated schema default from `5` to `0` with documentation comment
   - Updated metric labels from operation-based to `"dynamic"`
   - Rewrote `test_unified_queue_priority_calculation` → `test_unified_queue_priority_always_zero`
   - 10 files changed, all 22 queue_operations tests pass

2. **Added should_exclude_file() to file watcher** (commit `b71e6d39`)
   - Added exclusion check early in `enqueue_file_operation()` in `watching_queue.rs`
   - Prevents `.fastembed_cache`, `.mypy_cache`, `node_modules`, etc. from being queued going forward
   - All 13 exclusion pattern tests pass

3. **Deployed and restarted daemon** — running via launchd

## Next Task: Fix cleanup_excluded_files() — Points Not Being Deleted

**Problem**: After deploying the exclusion fix, points for excluded files (e.g. `.fastembed_cache`, hidden dirs) are NOT being removed from the active project's Qdrant collection. The point count is not decreasing.

**Where the cleanup logic lives**:
- `src/rust/daemon/core/src/unified_queue_processor.rs`
- `cleanup_excluded_files()` at line ~1084
- Called at the end of `scan_project_directory()` at line ~1062

**How it's supposed to work**:
1. `scan_project_directory()` is triggered by a `(Project, Scan)` queue item
2. After scanning and queueing new files, it calls `cleanup_excluded_files()`
3. `cleanup_excluded_files()` lists all files in Qdrant for the project
4. For each file, it strips the project root to get a relative path
5. It calls `should_exclude_file(&rel_path)` — if excluded, it queues a `(File, Delete)` item
6. The delete item is then processed to remove the point from Qdrant

**What to investigate**:
1. **Is a scan actually being triggered?** Check if the active project (this repo: `workspace-qdrant-mcp`) has had a `(Project, Scan)` item queued recently. Query: `SELECT * FROM unified_queue WHERE item_type='project' AND op='scan' ORDER BY created_at DESC LIMIT 5;`
2. **Is cleanup_excluded_files() being reached?** Add logging or check existing logs. The function logs `files_cleaned` at the end of `scan_project_directory()`.
3. **Is the Qdrant file listing working?** `cleanup_excluded_files()` queries Qdrant for stored file paths. If the query returns empty or fails silently, no deletions happen.
4. **Is the relative path stripping correct?** The function uses `strip_prefix(project_root)` — if the stored paths in Qdrant don't match the project_root format (e.g. trailing slash mismatch), the strip would fail and the file would be kept.
5. **Are the queued delete items actually being processed?** Check if `(File, Delete)` items are in the queue: `SELECT * FROM unified_queue WHERE op='delete' ORDER BY created_at DESC LIMIT 10;`

**Key files**:
- `src/rust/daemon/core/src/unified_queue_processor.rs` — `cleanup_excluded_files()` ~line 1084, `scan_project_directory()` ~line 920
- `src/rust/daemon/core/src/patterns/exclusion.rs` — `should_exclude_file()`
- SQLite state: `~/.workspace-qdrant/state.db`

**Quick diagnostic commands**:
```bash
# Check queue for scan/delete operations
sqlite3 ~/.workspace-qdrant/state.db "SELECT item_type, op, status, COUNT(*) FROM unified_queue GROUP BY item_type, op, status;"

# Check if any delete items were queued
sqlite3 ~/.workspace-qdrant/state.db "SELECT * FROM unified_queue WHERE op='delete' AND created_at > datetime('now', '-1 hour') LIMIT 10;"

# Check if scans ran recently
sqlite3 ~/.workspace-qdrant/state.db "SELECT * FROM unified_queue WHERE item_type='project' AND op='scan' ORDER BY created_at DESC LIMIT 5;"
```

## Pre-existing Issues (unchanged)
- 1 pre-existing test failure: `test_watch_manager_with_configuration` (missing `library_watches` table)
- 2 missing test modules: `lsp_daemon_integration_tests`, `daemon_state_persistence_tests`
- 64+ deprecation warnings in `queue_operations.rs` (legacy API)
- 70 warnings in `workspace-qdrant-core` lib (dead code, deprecations)

## Daemon Status
- Running via launchd: `com.workspace-qdrant.memexd` (healthy)
- Binary: `/Users/chris/.local/bin/memexd`
- Build: `ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib cargo build --release --manifest-path src/rust/Cargo.toml --package memexd`
