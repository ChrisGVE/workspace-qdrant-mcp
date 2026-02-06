# Handover: Priority System Fix + File Watcher Exclusion

**Date:** 2026-02-06
**Status:** Two tasks ready for implementation

## Completed Work (This Session)

1. **Fixed daemon SIGABRT crash** (commit `38e068f5`)
   - Root cause: `document_processor.rs` byte-offset string slicing inside multi-byte UTF-8 char (`─`)
   - Added `floor_char_boundary()` helper, fixed both `chunk_by_paragraphs` and `chunk_by_characters`
   - Fixed infinite loop guard in `chunk_by_characters`
   - 3 new unit tests, all 13 document_processor tests pass

2. **Resolved 167 stuck queue items** — all processed, queue empty

3. **Pushed all commits to origin/main**

## Next Tasks

### Task 1: Fix Priority System (INCORRECT implementation)

**Spec requirement (two-level priority, computed at query time):**
- **High priority (1)**: active projects + memory collection
- **Low priority (0)**: libraries + inactive projects
- Priority must NOT be stored in the queue schema — it is dynamic based on project activity
- The CASE expression in the dequeue query determines priority at query time (JOIN with watch_folders)
- The fairness scheduler alternates between DESC (high first) and ASC (low first) every 10 items

**What's ALREADY CORRECT (dequeue query in `queue_operations.rs:1843-1849`):**
```sql
ORDER BY
    CASE
        WHEN q.collection = 'memory' THEN 1
        WHEN q.collection = 'libraries' THEN 0
        WHEN w.is_active = 1 THEN 1
        ELSE 0
    END {DESC|ASC},
    q.created_at ASC
```
This query correctly computes priority at query time using the collection and watch_folders.is_active.
There are 4 copies of this query in `dequeue_unified()` (lines 1830-1920) for different filter combinations.

**What needs to be REMOVED/FIXED:**

1. **`calculate_priority()` in `watching_queue.rs:879-885`** — incorrectly assigns static priorities (3, 5, 8) based on operation type. This function should be REMOVED. The priority field passed to `enqueue_unified()` is meaningless because the dequeue query ignores it and computes priority dynamically.

2. **`priority` column in `unified_queue` schema (`unified_queue_schema.rs:597`)** — has `DEFAULT 5` and `CHECK (priority >= 0 AND priority <= 10)`. Since priority is computed at query time, this column is dead weight. Options:
   - **Recommended**: Keep the column but always set it to 0 (no-op), document that it's unused. This avoids a schema migration.
   - Alternative: Remove the column entirely (requires schema migration).

3. **Priority references in enqueue paths** — `enqueue_unified()` accepts a `priority: i32` parameter. All callers set meaningless values (5 for most, 7 for cleanup deletions). The parameter should either be removed or ignored.

4. **Priority metric labels in `queue_operations.rs:486-491`** — maps priority values (1, 3, 5) to labels ("high", "normal", "low"). Should reflect the two-level system.

5. **Document the convention** in CLAUDE.md and/or the queue schema: "Priority is computed at dequeue time, not stored. The queue's `priority` column is unused."

**Key design insight from user**: Since the queue has `collection` and `tenant_id`, and the dequeue query JOINs with `watch_folders` to get `is_active`, priority is fully deterministic at query time. Storing it in the row would require updating it every time a project's activity changes, which is worse.

### Task 2: Add Exclusion Check to File Watcher

**Problem**: `watching_queue.rs` only uses basic glob patterns from `watch_folders.ignore_patterns` (e.g., `.git/*`, `__pycache__/*`). It does NOT call the comprehensive `should_exclude_file()` from `patterns/exclusion.rs`. This allowed `.fastembed_cache` and `.mypy_cache` files to get queued.

**Fix**: Add `should_exclude_file()` check in `enqueue_file_operation()` (watching_queue.rs, around line 940) BEFORE enqueuing. Use the relative path (strip project root first).

**Key files:**
- `src/rust/daemon/core/src/watching_queue.rs` — `enqueue_file_operation()` line ~940
- `src/rust/daemon/core/src/patterns/exclusion.rs` — `should_exclude_file()` function
- Import: `use crate::patterns::exclusion::should_exclude_file;`

**Implementation notes:**
- The file path in `enqueue_file_operation` is available as `event.path` (absolute)
- Need to strip project root to get relative path for `should_exclude_file()`
- Project root can be obtained from the watch config
- Also check the absolute path as fallback (same pattern as `scan_project_directory`)

## Pre-existing Issues
- 1 flaky test: `watching::tests::single_folder_watch_tests::test_detect_file_modification`
- 2 missing test modules: `lsp_daemon_integration_tests`, `daemon_state_persistence_tests`
- 64+ deprecation warnings in `queue_operations.rs` (legacy API)

## Daemon Status
- Running via launchd: `com.workspace-qdrant.memexd` (healthy)
- Binary: `/Users/chris/.local/bin/memexd` (statically linked ONNX Runtime)
- Queue: empty
- Build: `ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib cargo build --release --manifest-path src/rust/Cargo.toml --package memexd`
