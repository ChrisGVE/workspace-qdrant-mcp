# Cross-Branch File Dedup

**Status**: **Layer 1 (skip the embed) — SHIPPED.** Layer 2 (share one point
across branches) — designed, not implemented.

> **Update 2026-05-31.** Layer 1 below is implemented and wired into the
> ingestion path; the per-file embed is skipped on cross-branch duplicates by
> copying the existing Qdrant vectors under a new `base_point`. A second,
> related mechanism — in-place `tracked_files` re-keying on `git checkout` — also
> shipped. Both are described in [What shipped](#what-shipped). Layer 2 (drop
> `branch` from `base_point` and share a single point via a `branches[]` payload
> array) remains the open follow-up. The original design text is preserved below
> for rationale.

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

## What shipped

Two complementary mechanisms, both of which reuse the dense+sparse vectors
verbatim and skip the dominant per-file cost (FastEmbed ONNX inference):

### A. Ingestion fast-path — `branch_dedup.rs` (this spec's Layer 1)

[`strategies/processing/file/branch_dedup.rs`](../../src/rust/daemon/core/src/strategies/processing/file/branch_dedup.rs),
called from [`ingest.rs`](../../src/rust/daemon/core/src/strategies/processing/file/ingest.rs)
before parse/embed. When a `file/add` or `file/update` arrives and another branch
already has the same `(watch_folder_id, relative_path, file_hash)`:

1. One SQL probe on `idx_tracked_files_dedup` finds the existing `base_point`.
2. `scroll_with_filter_and_vectors` pulls the old points **with vectors**.
3. `rekey_point` re-keys each point to the new `base_point` + branch (reusing the
   dense/sparse vectors and payload verbatim; only `point_id`, `base_point`,
   `branch`, `absolute_path` change) and re-upserts them.
4. A new `tracked_files` row is inserted for the branch with `source="dedup_clone"`
   and the same `chunk_count`.
5. Parse + embed are **skipped**. FTS5 (`search.db`) IS re-indexed for the new
   branch — search filters by `fm.branch = ?`, so FTS rows can't be shared.

On a 0-point scroll (stale row / partial cleanup) it falls back to a full ingest.

### B. Branch-switch re-keying — `branch_switch/db.rs`

On `git checkout`, files **not** in the changed-paths set don't re-ingest at all:
[`branch_switch/db.rs`](../../src/rust/daemon/core/src/branch_switch/db.rs)
recomputes each unchanged file's `base_point` for the new branch and `UPDATE`s the
`tracked_files` rows in place (`SET branch = ?, base_point = ?`) via a temp-table
join — no parse, no embed.

Net effect: switching or branching is near-free on the indexed-data side; only
genuinely changed files pay the full embed cost.

## Original design (Layer 1 as proposed, plus the open Layer 2)

### Layer 1: Skip the embed step (cheap) — SHIPPED, see [What shipped](#what-shipped)

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

~~Implement **Layer 1** first.~~ **Done** — see [What shipped](#what-shipped).
Layer 1 addressed the dominant cost (embed) without schema changes and is
backward-compatible.

**Open decision — Layer 2.** Layer 1 still **duplicates storage**: each branch
keeps its own copy of every point (same vectors, different `point_id`), so
`points_count` grows with `branches × unique-content`, not just unique content.
Layer 2 (share one point via a `branches[]` payload array) removes that
duplication but needs the `base_point` formula change + a migration. Pursue it
only if Qdrant storage from many long-lived branches becomes a real cost — the
compute waste (the expensive part) is already gone.

## Acceptance signals (post-Layer-1) — expected (criteria, not yet re-measured)

- Switching `main` → `fork/fixes` on workspace-qdrant-mcp re-ingests the
  ~28 modified files at full cost, but the ~1935 unchanged files complete
  in <100ms each (vs. ~2-3s for embed today).
- `wqm_unified_queue_processing_time_seconds` for `file/add` op drops by ~80%
  on branch-switch bursts.
- `tracked_files` still grows linearly with branch count (no schema change),
  but Qdrant `points_count` grows only with unique content (since we
  scroll-and-copy from existing).
