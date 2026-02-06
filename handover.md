# Handover: Discussion Items and Current State

**Date:** 2026-02-06
**Status:** Spec updated, discussion items pending

## Spec Updates Completed (This Session)

1. **Resource limits** (commit `fd973006`) — Added `daemon.resource_limits` to spec:
   - `max_concurrent_embeddings` (default 2) — Limits parallel ONNX embedding ops
   - `inter_item_delay_ms` (default 50) — Breathing room between items
   - `nice_level` (default 10) — OS-level process priority
   - `max_memory_percent` (default 70) — Memory pressure pause threshold

2. **File metadata for recovery** — Added `file_mtime` and `file_hash` to projects payload schema. These are required for daemon startup recovery.

3. **Phase 4: Daemon Startup Recovery** — New processing phase for reconciling Qdrant state with filesystem on daemon restart.

4. **Phase 2 correction** — Fixed "File modified → file/ingest" to "File modified → file/update" with atomicity documentation.

## Code Fixes (Previous Commits This Session)

1. **Scan trigger for unscanned projects** (commit `33af5153`) — `register_project()` now also queues a scan when `last_scan = NULL`. **NOTE: This may need reverting per discussion item below.**

2. **last_scan update** (same commit) — `process_project_item()` now updates `last_scan` after successful scan.

## Discussion Items Requiring Decision

### 1. Scan Policy: When Should Scans Trigger?

**Chris's position:** Scans should only be for NEW projects. Existing projects rely on the file watcher notification system. Recovery is a separate daemon startup concern.

**Current implementation:** Commit `33af5153` added scan triggering for existing projects with `last_scan = NULL` on re-activation via gRPC.

**Options:**
- **A) Revert the `needs_scan` change** — Scans only for new projects. Cleanup of stale data deferred to the Phase 4 recovery mechanism (not yet implemented).
- **B) Keep for now** — The `needs_scan` scan acts as a temporary workaround until Phase 4 recovery is implemented, then remove it.
- **C) Move to startup** — Instead of triggering on gRPC `register_project()`, trigger unscanned project scans during daemon startup (aligns with spec's existing Startup section which already says: "If folder not yet scanned (last_scan IS NULL): Queue folder for initial scan").

### 2. Updates Visibility and Detection

**Observation:** Zero `file/update` operations have been logged since the daemon restarted. This appears to be because:
- The scan queued 519 `file/ingest` items, filling the queue with ingest ops
- The `file_path UNIQUE` constraint silently drops any `file/update` attempts while an ingest for the same file is already pending
- Once the queue drains, new modify events should produce `file/update` items correctly

**Not a bug per se** — the spec says "If file modified before ingestion: ingested state is final" (the ingest reads current file content at processing time). But worth verifying that updates are seen during normal steady-state operation (no active scan flooding the queue).

**Action:** After the current queue drains completely, modify a watched file and verify a `file/update` item appears in the queue.

### 3. Recovery Mechanism Design

**Question from Chris:** "Can we easily obtain from Qdrant a list of all files and folders ingested?"

**Answer:** Yes — `scroll_file_paths_by_tenant()` already scrolls all file_paths for a tenant. This is what `cleanup_excluded_files()` uses. However, scrolling 28k+ points takes ~8 minutes due to 100-item batches.

**Question from Chris:** "Do we store creation and last modification date of the files we ingest?"

**Answer:** No. Currently stored fields per point: `content`, `chunk_index`, `file_path`, `tenant_id`, `branch`, `document_type`, `file_type`, `item_type`, `lsp_enrichment_status`. No `file_mtime`, no `file_hash`, no `created_at`.

**Implication:** Until `file_mtime` and `file_hash` are added to point payloads, recovery would need to either:
- **Option A:** Re-read and re-hash every file to detect changes (expensive but correct)
- **Option B:** Use file existence only (can detect added/deleted, not modified)
- **Option C:** Trust filesystem mtime vs a stored ingestion timestamp (requires storing ingestion timestamp at minimum)

**Spec updated** to require `file_mtime` and `file_hash` in payload. Implementation task needed.

### 4. Tree-sitter and LSP Verification

Not yet verified with the new ingestion pipeline. The processor code at line 654-730 of `unified_queue_processor.rs` shows LSP enrichment is attempted during file processing. The `lsp_enrichment_status: Partial` field on Qdrant points suggests it's partially working.

**Action needed:** Verify tree-sitter chunking produces meaningful `chunk_*` metadata and LSP enrichment adds useful symbol/reference data. Test with a known source file.

### 5. Resource Limits Implementation Priority

The CPU greediness issue is immediate. Options in order of implementation effort:

1. **`nice_level`** — Trivial: one `libc::setpriority()` call at daemon startup. Reduces CPU priority at OS level.
2. **`inter_item_delay_ms`** — Simple: add `tokio::time::sleep()` between items in the processing loop.
3. **`max_concurrent_embeddings`** — Medium: wrap embedding generator with a `tokio::sync::Semaphore`.
4. **`max_memory_percent`** — Medium: periodic `sysinfo` check, pause processing when exceeded.

Recommend implementing 1 + 2 first for immediate relief, then 3 + 4 as follow-up.

## Current Queue Status

As of writing, the daemon is processing delete items (excluded files cleanup):
- ~6,900 file/delete remaining (down from 9,076)
- Point count trending down as deletes process

## Previous Fixes (Earlier This Session)

1. **Fixed priority system** (commit `de7907d7`)
2. **Added should_exclude_file() to file watcher** (commit `b71e6d39`)

## Pre-existing Issues

- 1 pre-existing test failure: `test_empty_project_id` (test expects error for empty project_id, but code auto-generates)
- 1 pre-existing test failure: `test_watch_manager_with_configuration` (missing `library_watches` table)
- 2 missing test modules: `lsp_daemon_integration_tests`, `daemon_state_persistence_tests`
- 64+ deprecation warnings in `queue_operations.rs` (legacy API)
- 70 warnings in `workspace-qdrant-core` lib (dead code, deprecations)

## Key Files

- **Spec:** `WORKSPACE_QDRANT_MCP.md` (updated this session)
- **Queue processor:** `src/rust/daemon/core/src/unified_queue_processor.rs`
- **Project service:** `src/rust/daemon/grpc/src/services/project_service.rs`
- **File watcher queue:** `src/rust/daemon/core/src/watching_queue.rs`
- **Config:** `src/rust/daemon/core/src/config.rs`
- **Daemon logs:** `/Users/chris/Library/Logs/workspace-qdrant/daemon.jsonl`
