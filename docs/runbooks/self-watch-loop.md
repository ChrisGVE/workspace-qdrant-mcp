# Self-Watch Feedback Loop — Recovery Runbook

This runbook covers the case where the daemon enters a feedback loop because the `workspace-qdrant-mcp` repo itself is registered as an indexed project, and `state/qdrant/storage/` (Qdrant's working data dir, host-bind-mounted into the container) is NOT excluded by `global.wqmignore`. Symptoms, diagnosis queries, recovery steps, and the prevention template are documented here.

## Symptom Quick Match

| Observation | Likely cause |
|---|---|
| Grafana queue-depth panel never reaches zero, oscillating around N for hours | Self-watch loop |
| `docker logs wqm-memexd` shows `process_item ... op=Delete ... state/qdrant/storage/...` lines repeatedly | Loop confirmed |
| `Successfully processed unified item ... (type=File, op=Delete) in ~2000ms` with `point_count=0` deletes | Loop items are no-ops but still cost 2s each |
| Multiple `local_*` tenants in the queue all targeting paths inside `state/qdrant/`, `state/memexd/`, or `.fastembed_cache/` | Both Docker-Desktop mount aliases registered the repo |

## Diagnosis

Run these queries against the daemon's SQLite (works from any host with `docker` + the `memexd_db` named volume):

```
docker run --rm --user 0:0 -v workspace-qdrant-mcp_memexd_db:/data \
  keinos/sqlite3:latest sqlite3 //data/memexd.db \
  "SELECT tenant_id, op, COUNT(*) FROM unified_queue WHERE status='pending' GROUP BY tenant_id, op ORDER BY 3 DESC LIMIT 20"
```

Tenants whose pending items all target paths under `state/qdrant/` or `state/memexd/` are the loop participants. Cross-reference watch_folders to confirm:

```
docker run --rm --user 0:0 -v workspace-qdrant-mcp_memexd_db:/data \
  keinos/sqlite3:latest sqlite3 //data/memexd.db \
  "SELECT watch_id, tenant_id, path, collection, enabled FROM watch_folders WHERE enabled=1 ORDER BY path"
```

If two distinct `watch_id` rows point at the same physical project under different mount-path prefixes (e.g. `/mnt/c/...` and `/run/desktop/mnt/host/c/...`), Docker Desktop is exposing the same bind mount under two aliases. Both watch folders trigger filesystem events for the same Qdrant write — doubling queue volume.

## Recovery Procedure

The fix has three steps. Execute in order.

### Step 1 — Add the self-storage exclusions to `global.wqmignore`

The live file is at `state/memexd/global.wqmignore` (host bind-mount target for `/var/lib/memexd/global.wqmignore` inside the daemon container). Append:

```
# Self-storage (Qdrant + memexd + MCP local state)
state/
**/state/qdrant/
**/state/memexd/
**/state/mcp/
**/.fastembed_cache/
```

The full template lives at `assets/global.wqmignore.example` — copy it wholesale on a fresh install.

### Step 2 — Cancel the already-queued loop items

These items were enqueued before the exclusion took effect; the daemon will still grind through them at ~2s each unless cancelled. Stop the daemon, delete, restart:

```
docker compose stop memexd

docker run --rm --user 0:0 -v workspace-qdrant-mcp_memexd_db:/data \
  keinos/sqlite3:latest sqlite3 //data/memexd.db \
  "DELETE FROM unified_queue WHERE status='pending' AND (json_extract(payload_json, '\$.file_path') LIKE '%state/qdrant/%' OR json_extract(payload_json, '\$.file_path') LIKE '%state/memexd/%' OR json_extract(payload_json, '\$.file_path') LIKE '%state/mcp/%' OR json_extract(payload_json, '\$.file_path') LIKE '%/.fastembed_cache/%')"

docker compose up -d memexd
```

Note: `--user 0:0` is required because the named volume is owned by UID 1000 inside the container, and the sqlite container's default user cannot write to it otherwise.

### Step 3 — Verify

After the daemon comes back healthy:

```
docker run --rm --user 0:0 -v workspace-qdrant-mcp_memexd_db:/data \
  keinos/sqlite3:latest sqlite3 //data/memexd.db \
  "SELECT COUNT(*) FROM unified_queue WHERE status='pending' AND json_extract(payload_json, '\$.file_path') LIKE '%state/qdrant/%'"

docker logs wqm-memexd --since 60s 2>&1 | grep -c "state/qdrant"
```

Both should report `0`. The Grafana queue-depth panel will continue to show the legitimate backlog (ignore_sync re-adds and project files) draining at the normal rate.

## Prevention

Fresh installs and forks should start from `assets/global.wqmignore.example`, which includes the self-storage section by default. The bundled docker compose mount path is:

```
volumes:
  - ${WQM_STATE_DIR:-./state}/memexd/global.wqmignore:/var/lib/memexd/global.wqmignore
```

Copying the example into `state/memexd/global.wqmignore` is sufficient — no daemon rebuild needed.

## Why This Happens

When Qdrant rotates segments (rename to `.deleted/`, then physically remove) under `state/qdrant/storage/segments/...`, the daemon sees those filesystem events through `notify-debouncer-full` and — unless the path is excluded — enqueues `Add`/`Update`/`Delete` operations. Each `Delete` calls `delete_by_filter` on Qdrant; for files that were never indexed as content the call returns `point_count=0` but still costs a network round-trip (~2s). While the daemon processes one loop item, Qdrant may rotate more segments, producing more loop items. Steady state: queue size oscillates around N, never zero.

### Where `global.wqmignore` is enforced

All three paths that can enqueue file work apply `global.wqmignore`, so an excluded path (`state/`, `.fastembed_cache/`, …) is dropped *before* it reaches the queue:

- **File watcher** (`watching_queue/file_watcher_ops.rs`, `file_watcher_ops_helpers.rs`) — calls `patterns::global_ignore::is_globally_ignored()` in `should_filter_event`, `should_filter_debounced_event`, and the `enqueue_file_operation` chokepoint (gates `Add`/`Update`/`Delete`). **This is the gate that breaks the feedback loop at its source.**
- **Folder scan** (`strategies/processing/folder/scan.rs`) — `is_ignored_by_matcher()` checks `is_globally_ignored()` for both directories (pruned before descent) and files, on top of per-project `.gitignore`/`.wqmignore`.
- **Ignore reconciler** (`startup/reconciliation/ignore_sync.rs`) — `walk_eligible_files` walks with `WalkBuilder::add_ignore(global.wqmignore)` AND post-filters the eligible set through `global_ignore::matcher_from(global_path)` (root-anchored). `add_ignore` alone only matches depth-1 project paths reliably; the post-filter catches deep matches so the reconciler agrees with the watcher/folder-scan and marks stale (then deletes) residuals like `state/qdrant`/`generated`.

> Historical note: before the watcher/folder-scan gates were added, *only* the reconciler honoured `global.wqmignore` — and only at depth-1 (its `add_ignore` anchor leaks deeper paths). Editing the file did not reliably stop the reconciler from re-adding deep `state/qdrant`/`generated`, and the live watcher kept enqueuing `op=Update` events on every segment rotation, so the loop returned after each restart. The `is_globally_ignored()` gates (watcher + folder-scan) plus the reconciler post-filter close the hole on all three paths. The matcher is rebuilt automatically when the file's mtime changes, so edits via the admin UI take effect without a restart.

> Anchoring: the watcher/folder-scan matcher and the reconciler post-filter all anchor at `/` (root) via `global_ignore`. The reconciler's underlying `WalkBuilder::add_ignore` anchors at the ignore file's parent dir and leaks depth-2+ matches — which is exactly why the post-filter exists. Reserve root-anchored (`/foo`) patterns for project-level `.wqmignore`.

## Related

- `assets/global.wqmignore.example` — the template that includes the fix.
- `docs/runbooks/qdrant-corruption.md` — separate runbook for Qdrant collection corruption, often a co-symptom (the loop accelerates segment churn, which makes corruption more likely on dirty shutdown).
