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

The daemon's ignore reconciliation in `src/rust/daemon/core/src/startup/reconciliation/ignore_sync.rs` compares `tracked_files` against the filesystem subject to the active ignore rules. Without exclusions for `state/`, any file Qdrant writes under `state/qdrant/storage/segments/...` is considered an eligible project file. When Qdrant rotates segments (rename to `.deleted/`, then physically remove), the daemon sees these filesystem events through `notify-debouncer-full` and enqueues `Add`/`Update`/`Delete` operations.

Each `Delete` calls `delete_by_filter` on Qdrant; for files that were never indexed as content the call returns `point_count=0` but still costs a network round-trip (~2s). While the daemon processes one loop item, Qdrant may rotate more segments, producing more loop items. Steady state: queue size oscillates around N, never zero.

Adding `state/` and `.fastembed_cache/` to `global.wqmignore` breaks the loop at the source — filesystem events under those paths are filtered before they reach the queue.

## Related

- `assets/global.wqmignore.example` — the template that includes the fix.
- `docs/runbooks/qdrant-corruption.md` — separate runbook for Qdrant collection corruption, often a co-symptom (the loop accelerates segment churn, which makes corruption more likely on dirty shutdown).
