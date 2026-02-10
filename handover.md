# Handover

## Current State

Daemon (`memexd`) is running and actively processing queue items. All fixes are deployed and pushed to remote.

PRD created and parsed into task-master. Spec updated with changes.

## Completed Work

### Queue Processing Fix (4 commits, pushed)

The daemon had 516 pending queue items that were never being processed. Root cause analysis and fixes:

1. **`fix(core): handle file_path UNIQUE constraint in enqueue_unified`** (`be6b504f`)
   - `INSERT OR IGNORE` was silently ignored when `file_path` UNIQUE constraint triggered (different from `idempotency_key`)
   - The subsequent `SELECT queue_id WHERE idempotency_key = ?` found nothing -> "no rows returned" error
   - Fix: Added fallback lookup by `file_path` when idempotency_key lookup fails

2. **`perf(core): skip excluded directories in WalkDir filesystem walks`** (`147f7a3e`)
   - WalkDir in startup_recovery walked ALL 562K files including `target/` (500K+), `node_modules/`, `.git/`
   - Added `should_exclude_directory()` function to `patterns/exclusion.rs`
   - Applied `filter_entry` to WalkDir in `startup_recovery.rs` and `unified_queue_processor.rs`

3. **`fix(daemon): start queue processor before startup recovery`** (`0062a8f1`)
   - `unified_queue_processor.start()` was called AFTER `run_startup_recovery()` which blocked for ages
   - Moved `start()` before recovery, spawned recovery as background tokio task
   - Queue processor now starts immediately, recovery runs concurrently

4. **`fix(core): prevent UTF-8 boundary panic in tree-sitter chunker`** (`4cc3e7cd`)
   - `split_chunk_with_overlap()` used byte offsets to slice `&str` without char boundary checks
   - Box-drawing characters (U+2500, 3 bytes) caused panics when byte arithmetic landed mid-character
   - Added `safe_char_boundary()` helper, applied to all byte-offset slice operations

### Test Artifact Prevention (1 commit, from previous session)

- **`fix(test): prevent Qdrant test artifacts from persisting`** (`3f25c032`)
  - Added `BenchCleanupGuard` Drop guard for benchmark collections
  - Added `cleanup_test_tenant_data()` to gRPC tests
  - Added `delete_points_by_payload_field()` to StorageClient

### PRD and Spec Updates (this session)

- **PRD created**: `.taskmaster/docs/20260210-2045_project_0.4.0_PRD_routing-sessions-priority.md`
- **13 tasks appended** to task-master (IDs 561-573), covering:
  - Asymmetric batch sizes in fairness scheduler (561)
  - Unified extension allowlist (562)
  - Always-track behavior audit (563)
  - tracked_files schema migration for collection column (564)
  - FileRoute enum and route_file() method (565, depends on 562+564)
  - Startup recovery routing (566, depends on 565)
  - File watcher routing (567, depends on 565)
  - Queue processor routing (568, depends on 566+567)
  - Daemon inactivity timeout (569)
  - Activity inheritance for RegisterProject/DeprioritizeProject (570)
  - Claude Code hook documentation (571)
  - Third-party license generation (572)
  - Release workflow license inclusion (573, depends on 572)
- **Spec updated** (`WORKSPACE_QDRANT_MCP.md`):
  - tracked_files schema: added `collection TEXT NOT NULL DEFAULT 'projects'`
  - Anti-starvation mechanism: updated to asymmetric batching (high=10, low=3)
  - Sort alternation: updated to asymmetric description
  - File Type Allowlist design: updated to two-tier allowlist with format-based routing

## Next Steps

1. **Expand tasks** 561-573 into subtasks via `task-master expand`
2. **Start Phase 1** (independent tasks): 561, 562, 563 can be worked in parallel
3. **Commit spec changes** (WORKSPACE_QDRANT_MCP.md updates)

## Known Issues

- 3 pre-existing test failures in `document_service_basic_tests.rs` (collection name validation not rejecting invalid names)
- 1 pre-existing failure in `collection_service_tests.rs` (`test_alias_canonical_name_rejection`)
- `watching.rs` WalkDir at line 1660 doesn't yet use `filter_entry` for directory exclusion (lower priority since it's the watcher's initial scan)

## Open Design Questions

- **LSP cross-boundary resolution for submodules**: Deferred. Submodules that are libraries used by the parent project create LSP symbol resolution that crosses project boundaries. Current approach: independent LSP instances per project, cross-project symbol references captured via file content indexing.

## Queue Status

Daemon actively processing. Queue draining since last session.
