# Multi-Clone Tenant — Correctness Audit & Design Note

**Date:** 2026-05-31
**Status:** Audit + recommendation. No runtime behavior changed by this document.
**Companion:** memory `project_multiclone_tenant_registration_knot`; fixes `545c4cd09`, `ee2e925b7`, `988b3e0f5`.

## Summary

"Multi-clone" = two or more working copies of the **same git remote** share one
`tenant_id` and each gets its own `watch_folders` row (`path` is `UNIQUE`,
`tenant_id` is **not**). The feature shipped (`545c4cd09`) but the **"one
watch_folder per tenant" assumption leaked into many path-resolution call-sites**
that key on `tenant_id` alone. Three were fixed in the last days; this audit finds
**2 more active bugs + 1 dead-code bug + a few low-impact reads**, and recommends a
durable structural fix so the class can't keep recurring.

## The root pattern

```sql
SELECT path FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1
```

`LIMIT 1` on a tenant with N watch_folders returns an **arbitrary** row (in
practice the lowest rowid, often the oldest/stale clone). When the resolved `path`
is then used to **anchor a relative file path** (build the absolute path), the
file is looked up under the wrong clone's root → `exists()` is false → the file is
treated as missing/dropped/skipped. This is exactly how the canonical
`workspace-qdrant-mcp` checkout failed to index after migrating onto the shared
`367157a01d98` tenant (its first-by-rowid sibling `_wq-roadmap-pr` was stale/gone
on disk).

**Key invariant that makes most aggregate queries safe:** `base_point =
SHA256(tenant_id | branch | relative_path | file_hash)` does **not** include the
clone's absolute path. So identical content across clones maps to the **same**
Qdrant point — search/graph/FTS aggregate over the tenant *correctly* and even
get free cross-clone dedup. The bug is **only** in resolving *which clone's
on-disk root* to use for path operations.

## Audit table

Legend — **Impact**: HIGH = wrong file path/skip/drop; LOW = cosmetic/aggregate; OK = correct by design.

| Call-site | Keys on | Picks 1 row? | Used for | Multi-clone impact | Status |
|---|---|---|---|---|---|
| `file/mod.rs:410` `resolve_watch_folder` | tenant+coll | LIMIT 1 | anchor abs path for ingest | wrong clone → file "missing" → never indexed | **FIXED** `988b3e0f5` (disambiguate-by-disk) |
| `tenant/project.rs:281` `handle_project_scan` last_scan | tenant+coll | read+UPDATE | mtime prune | new clone read sibling's `last_scan` → all files pruned → 0 indexed | **FIXED** `ee2e925b7` (key on path) |
| `RegisterProject` / `path_has_watch_folder` | path | — | register sibling clone | 2nd clone couldn't register | **FIXED** `545c4cd09` |
| `processing/submitter.rs:362` (spill) | tenant+coll | LIMIT 1 | anchor spilled file path | wrong clone → spill path wrong / spill aborts | **OPEN — HIGH** |
| `queue_operations/triage.rs:228` `should_drop_failed_add_update` | tenant+coll | LIMIT 1 | build abs path to test existence | wrong clone → wrongly drops (or keeps) a failed item | **OPEN — HIGH** |
| `branch_switch/queue.rs:151` `lookup_watch_folder_root` | tenant+coll | LIMIT 1 | anchor `enqueue_file_op` path | wrong clone → anchoring fails | **OPEN — dead_code** (`#[allow(dead_code)]`; fix or delete) |
| `folder/delete.rs:29` (via `lookup_watch_folder`) | tenant+coll | LIMIT 1 | folder-delete cleanup | cleans only one clone's tracked_files | **REVIEW — MED** |
| `tenant/cleanup.rs:37` (via `lookup_watch_folder`) | tenant+coll | LIMIT 1 | exclusion cleanup of tracked_files | cleans only one clone | **REVIEW — MED** |
| `daemon_state/lifecycle_ops.rs:37/58/102` `get_watch_folder_by_tenant_id` | tenant+coll, parent NULL | LIMIT 1 | activation/lifecycle (returns watch_id/path) | picks one clone for lifecycle side-effects | **REVIEW — LOW** |
| `priority_manager/manager.rs:195` `is_active LIMIT 1` | tenant+coll | LIMIT 1 | compute `previous_priority` (return value) | cosmetic; the mutation uses tenant-wide `activate_by_tenant` | LOW |
| `priority_manager/manager.rs:106` `MAX(is_active)` | tenant | aggregate | activity gate | wants max across clones | OK |
| `code_lines_schema` FTS search (`fm.tenant_id=?`) | tenant(+branch) | aggregate | search | spans all clones (intended) | OK |
| `grep_search`, `graph/sqlite_store`, `cooccurrence_schema` | tenant(+coll) | aggregate | search/graph | per-tenant by design | OK |
| `tenant/delete.rs`, `tenant/project.rs:377` | tenant | all rows | delete tenant | removes everything (intended) | OK |
| `folder/strategy.rs:202`, `tenant/library.rs:86` | tenant+coll | all (no LIMIT) | enumerate tenant paths | iterates all rows — appears correct (verify) | OK* |
| `grouping/git_org.rs`, `grouping/workspace/mod.rs` (`tenant_id != ?`) | tenant | aggregate | project grouping | correct | OK |

\* `OK*` = looks correct on read but not exercised under a live multi-clone tenant; verify.

## (B) Architectural alternatives

### Option 1 — Keep shared tenant, harden path resolution (recommended)
- **Durable fix:** carry `watch_folder_id` in the File queue payload (`FilePayload`)
  so every file item resolves its watch_folder by **`watch_id`** (unambiguous),
  not by `(tenant_id, collection) LIMIT 1`. This *eliminates the entire bug class*
  at the source — no call-site ever has to guess "the one watch folder".
- **Interim:** apply the `disambiguate_multi_path` (resolve-by-on-disk-existence)
  pattern already added in `file/mod.rs` to the 2 open HIGH sites (spill, triage),
  and fix-or-delete the dead `branch_switch/queue.rs` helper.
- Keeps cross-clone dedup (shared `base_point`) and unified search.
- **Cost:** a payload field + threading it through enqueue sites; backward-compatible
  (fall back to disk-disambiguation when the field is absent).

### Option 2 — Per-working-copy tenant
- Derive a distinct `tenant_id` per clone (e.g. `remote_hash` + `disambiguation_path`).
- **Pro:** one watch_folder per tenant → every audited bug disappears; path
  resolution is trivial.
- **Con:** `base_point` diverges per clone → **loses cross-clone dedup** (identical
  files re-embedded per clone = storage + compute duplication); search no longer
  unifies clones; needs a tenant-id migration; and it re-introduces a variant of
  the original "knot" (the whole point of `545c4cd09` was to let clones coexist).
- Net: trades a contained, fixable bug class for a permanent storage/dedup
  regression. Not recommended.

### Stale-clone hygiene (orthogonal, needed either way)
`_wq-roadmap-pr` was registered but **gone on disk**; its first-by-rowid position
is what made the resolution bug bite. The daemon should **deregister watch_folders
whose `path` no longer exists** during startup recovery (it already detects
"Watch folder path does not exist" — it should prune the row, not just warn). This
shrinks the multi-clone surface and removes stale first-rowid landmines.

## (C) Recommendation

**Keep the shared-tenant model; do not move to per-working-copy tenants** (Option 2
would forfeit cross-clone dedup and unified search for a bug class that is
fixable). Instead:

1. **Structural:** thread `watch_folder_id` into `FilePayload` and resolve files by
   `watch_id`. This is the real fix — it retires the `(tenant_id, collection) LIMIT 1`
   anti-pattern everywhere at once. (Largest change; do once.)
2. **Until then:** patch the 2 OPEN HIGH sites (`submitter.rs` spill,
   `triage.rs` drop-decision) with the `disambiguate_multi_path` resolve-by-disk
   helper; fix or delete the dead `branch_switch/queue.rs` helper.
3. **Review** the MED sites (`folder/delete.rs`, `tenant/cleanup.rs`): confirm
   whether single-clone cleanup leaves other clones' rows orphaned.
4. **Add stale-watch_folder pruning** to startup recovery (deregister rows whose
   path is gone on disk).
5. Add a regression test: a tenant with 2 watch_folders where the first-by-rowid
   path is missing on disk — assert files under the second path still index.

## Appendix — verification commands

```bash
# Tenants with >1 watch_folder (multi-clone candidates):
docker run --rm -v workspace-qdrant-mcp_memexd_db:/db:ro alpine sh -c \
  "apk add -q sqlite; sqlite3 'file:/db/memexd.db?immutable=1' \
   'SELECT tenant_id, COUNT(*) FROM watch_folders GROUP BY tenant_id HAVING COUNT(*)>1;'"

# Watch_folders whose path is gone on disk (stale-clone candidates) — check each path
# with: docker exec wqm-memexd test -d <path> && echo ok || echo STALE
```
