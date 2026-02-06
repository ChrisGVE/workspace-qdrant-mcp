# Handover: Implementation Plan Active

**Date:** 2026-02-06
**Status:** Specs finalized, implementation tasks created in task-master

## Completed This Session

1. **Spec: Five-table SQLite design** (commit `e350ea6b`)
   - `tracked_files` table: authoritative file inventory with metadata
   - `qdrant_chunks` table: per-chunk Qdrant point tracking
   - Transaction specifications for file ingest/delete/update
   - Error message accumulation format
   - Removed `file_mtime`/`file_hash` from Qdrant payload (moved to SQLite)

2. **Spec: Expanded configuration structure** (same commit)
   - All 12 `DaemonConfig` sections documented with defaults
   - `resource_limits` section with 4 levels

3. **Spec: Recovery and cleanup updated** (same commit)
   - Phase 4 recovery now uses `tracked_files` (milliseconds vs 8-minute Qdrant scroll)
   - Phase 1 scan notes `tracked_files` population
   - Deployment diagram shows 5 tables

4. **CLAUDE.md updated** — Core Tables count updated to 5

## Task-Master Tasks (development tag)

| ID  | Title | Priority | Dependencies | Status |
|-----|-------|----------|-------------|--------|
| 504 | Implement resource limits configuration and enforcement | high | none | pending |
| 505 | Implement tracked_files and qdrant_chunks SQLite tables | high | 504 | pending |
| 506 | Wire tracked_files/qdrant_chunks into file processing transactions | high | 505 | pending |
| 507 | Implement daemon startup recovery using tracked_files | medium | 506 | pending |
| 508 | Benchmark Qdrant ingestion vs deletion timing | medium | 505 | pending |
| 509 | Revert needs_scan change (scan only for new projects) | medium | 507 | pending |

**Dependency chain:** 504 → 505 → 506 → 507 → 509
**Independent after 505:** 508 (benchmark)

## Key Design Decisions Made

1. **file_path is relative** to `watch_folders.path` — project moves don't require updates
2. **file_hash kept** (file_size dropped) — needed for recovery and update-skip optimization
3. **qdrant_chunks** (not tracked_chunks) — tracks individual Qdrant point UUIDs per file
4. **Transaction invariant**: SQL transaction opens BEFORE Qdrant operation; on success contains dequeue + file tracking + chunk tracking; on failure contains only retry update with accumulated error
5. **Error accumulation**: `error_message` in unified_queue uses `{datetime} error\n...` format across retries
6. **parallel_processing always true** — not configurable (no single-core systems)
7. **Scan policy**: Scans only for new projects; recovery (task 507) handles daemon restart
8. **Surgical updates** (future): Enabled by `qdrant_chunks.content_hash` — compare old vs new chunk hashes

## Queue Status

Delete queue was at ~6,365 items at session start. Benchmark (task 508) should wait until queue drains.

## Key Files

- **Spec:** `WORKSPACE_QDRANT_MCP.md` (updated this session)
- **Scratchpad with full proposal:** see session scratchpad `spec_changes_proposal.md`
- **Config:** `src/rust/daemon/core/src/config.rs`
- **Queue processor:** `src/rust/daemon/core/src/unified_queue_processor.rs`
- **Project service:** `src/rust/daemon/grpc/src/services/project_service.rs`
- **Schema:** `src/rust/daemon/core/src/schema_version.rs`

## Previous Session Context

- Commits `33af5153` through `5d52518a` from earlier in this session (scan fix, spec updates, handover)
- Pre-existing test failures: `test_empty_project_id`, `test_watch_manager_with_configuration`
- 64+ deprecation warnings in `queue_operations.rs`, 70 warnings in `workspace-qdrant-core`
