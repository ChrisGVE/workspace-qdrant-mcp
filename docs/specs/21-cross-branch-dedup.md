# Cross-Branch File Dedup (Design Proposal)

**Status**: Design proposed, not implemented.

## Problem

When a user switches branches (`git checkout main` → `git checkout fork/fixes`),
the daemon re-indexes the working tree under the new branch label. Concretely:

- `tracked_files` gets a new row per file with `branch=fork/fixes`.
- `compute_base_point` includes `branch` in its hash, so the new tracked_files
  row gets a different `base_point` than the `main` row for the same physical
  content.
- Qdrant gets a fresh set of points (one per chunk) with new point_ids.
- The expensive **embed** step (FastEmbed ONNX inference per chunk) runs again
  for every chunk.

Measured on workspace-qdrant-mcp: **13 out of 15 sample cross-branch pairs have
IDENTICAL file_hash** between `main` and `fork/fixes` (87% are wasted
re-ingestion). Only 2/15 are real edits.

## Why the architecture mandates this today

1. `base_point = SHA256(tenant_id | branch | relative_path | file_hash)[:32]`
   — branch is in the hash, so identical content on different branches gets
   different base_points.
2. Qdrant payload stores `branch` as a single scalar; search filters by
   `payload.branch == "X"` for branch-scoped queries.
3. `file_metadata.branch` (search.db) plays the same role for FTS5 search.
4. Branch deletion is implemented as `DELETE WHERE branch = ?` — depends on
   each branch owning its own rows.

So the cost of branch isolation is the cost of branch re-ingestion.

## Schema infrastructure already present

- `tracked_files.base_point` (nullable) — exists, populated at insert.
- `idx_tracked_files_base_point` — index on base_point.
- `idx_tracked_files_refcount` on `(base_point, watch_folder_id)` — hints
  that ref-counted sharing was anticipated.
- `idx_tracked_files_dedup` on `(watch_folder_id, relative_path, file_hash)`
  — added 2026-05-27 for fast pre-ingestion lookup.
- `watch_folders.is_active` is already a counter (`SET is_active = is_active + 1`),
  so the codebase has precedent for refcount-style sharing.

## Proposed implementation (two layers)

### Layer 1: Skip the embed step (cheap)

Before `parse_document` in [`strategies/processing/file/ingest.rs`](../src/rust/daemon/core/src/strategies/processing/file/ingest.rs):

```rust
let existing = sqlx::query_as::<_, (String, i64)>(
    "SELECT base_point, file_id FROM tracked_files \
     WHERE watch_folder_id = ?1 AND relative_path = ?2 AND file_hash = ?3 \
       AND branch != ?4 AND base_point IS NOT NULL \
     ORDER BY updated_at DESC LIMIT 1",
)
.bind(watch_folder_id).bind(relative_path).bind(&file_hash).bind(&item.branch)
.fetch_optional(pool).await?;
```

If found:
1. Compute the **new** base_point (current branch).
2. Call `storage_client.scroll_points(filter_by_base_point=existing.0)` → get
   vectors + content + payload.
3. Re-upsert under new base_point with payload updated (`branch =
   current_branch`, `base_point = new`).
4. Insert tracked_files row with new base_point + current branch +
   `chunk_count` copied from existing.
5. Skip parse + embed entirely.
6. Run FTS5 indexing as usual (new file_id, current branch).

Cost reduction: ~80% per duplicate file (embed dominates the per-file cost).

### Layer 2: Share base_point across branches (architectural)

Bigger lift; requires schema migration.

- Drop `branch` from `compute_base_point` formula (breaking change for
  existing base_points — migration script needed).
- Move `branch` out of Qdrant payload's single-value field; add
  `branches: Vec<String>` array. Search filter becomes
  `branch IN payload.branches`.
- `file_metadata.branches` likewise.
- On ingestion of identical content for a new branch: scroll points, append
  branch to `branches` array (`set_payload`).
- On branch deletion: remove branch from each point's `branches` array;
  delete the point only when `branches` becomes empty.
- Refcount on `tracked_files.base_point` already supports the SQLite side.

This eliminates storage duplication entirely. ~500 lines + migration.

## Recommendation

Implement **Layer 1** first. It addresses the dominant cost (embed) without
schema changes, is backward-compatible, and lets the team gather data before
committing to Layer 2.

## Acceptance signals (post-Layer-1)

- Switching `main` → `fork/fixes` on workspace-qdrant-mcp re-ingests the
  ~28 modified files at full cost, but the ~1935 unchanged files complete
  in <100ms each (vs. ~2-3s for embed today).
- `wqm_unified_queue_processing_time_seconds` for `file/add` op drops by ~80%
  on branch-switch bursts.
- `tracked_files` still grows linearly with branch count (no schema change),
  but Qdrant `points_count` grows only with unique content (since we
  scroll-and-copy from existing).
