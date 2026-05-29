# Qdrant Collection Corruption — Recovery Runbook

This runbook covers the case where one or more Qdrant collections enter an unloadable state after a dirty shutdown (host reboot mid-write, OOM kill during ingestion, hard power-off). Symptoms, automatic recovery flow, manual fallback, and post-recovery verification are all documented here.

## Table of Contents

1. [Symptom Quick Match](#symptom-quick-match)
2. [Automatic Recovery (Entrypoint Wrapper)](#automatic-recovery-entrypoint-wrapper)
3. [Manual Recovery Procedure](#manual-recovery-procedure)
4. [Post-Recovery Drift Cleanup](#post-recovery-drift-cleanup)
5. [Verification](#verification)
6. [Known Panic Patterns](#known-panic-patterns)
7. [Operator Q&A](#operator-qa)
8. [Forensics: Inspecting a Quarantined Collection](#forensics-inspecting-a-quarantined-collection)

## Symptom Quick Match

| Observation | Likely cause | Jump to |
|---|---|---|
| `docker ps` shows `wqm-qdrant Restarting (101)` repeatedly | Shard load panic on startup | [Automatic Recovery](#automatic-recovery-entrypoint-wrapper) |
| `wqm-memexd` and `wqm-mcp` stuck in `Created` state | Blocked by `depends_on: qdrant healthy` | Same — fix qdrant first |
| `docker logs wqm-qdrant` shows `Failed to load local shard "./storage/collections/<name>/...` | Segment corrupted on disk | [Automatic Recovery](#automatic-recovery-entrypoint-wrapper) |
| `docker logs wqm-qdrant` shows `gridstore.rs.*LiteralOutOfBounds` | Same root cause, different layer | [Automatic Recovery](#automatic-recovery-entrypoint-wrapper) |
| Stack healthy but MCP search returns 0 results for files you know were indexed | Quarantine ran; SQLite drift | [Post-Recovery Drift Cleanup](#post-recovery-drift-cleanup) |
| `state/qdrant/storage/.corrupted_*` directories present | A previous auto-quarantine ran | [Forensics](#forensics-inspecting-a-quarantined-collection) |

## Automatic Recovery (Entrypoint Wrapper)

**Component:** [`docker/qdrant-quarantine-wrapper.sh`](../../docker/qdrant-quarantine-wrapper.sh), bound to Qdrant's entrypoint via [`docker-compose.yml`](../../docker-compose.yml).

**What it does on every Qdrant startup attempt:**

1. Runs Qdrant's original entrypoint (`/qdrant/entrypoint.sh`).
2. Tees combined stdout+stderr into a temp log so `docker logs` still works AND the wrapper can inspect after process exit.
3. If Qdrant exits with non-zero status, greps the log for `Failed to load local shard "./storage/collections/<name>/`.
4. For each matched `<name>`, moves `storage/collections/<name>` to `storage/.corrupted_<UTC_TS>_<name>` and appends a line to `storage/.quarantine_log` in the format `<utc_ts>|<collection>|shard-load-panic`.
5. Re-executes Qdrant. With the offending collection out of the way, Qdrant boots clean.
6. Caps retries at 3 (configurable via `QDRANT_QUARANTINE_MAX_RETRIES`). After max retries, propagates the failure.

**Operator observability:**

```sh
# Did the wrapper trigger?
docker logs wqm-qdrant 2>&1 | grep -E "qdrant-quarantine"

# Persistent trail of every quarantine
cat state/qdrant/storage/.quarantine_log
# Example:
# 20260527_144417Z|projects|shard-load-panic

# What survived
docker run --rm -v workspace-qdrant-mcp_qdrant_storage:/data alpine ls /data/collections
# (collections without panics are intact)
```

**Expected log lines when a quarantine succeeds:**

```
[qdrant-quarantine] wrapping /qdrant/entrypoint.sh (max_retries=3)
... (Qdrant attempts to load, panics, exits) ...
[qdrant-quarantine] qdrant exited with failure on attempt 1
[qdrant-quarantine] QUARANTINED projects -> .corrupted_20260527_144417Z_projects
[qdrant-quarantine] retrying (1/3)...
... (Qdrant boots clean) ...
INFO qdrant::actix: Qdrant HTTP listening on 6333
```

The stack is now functional. Continue to [Post-Recovery Drift Cleanup](#post-recovery-drift-cleanup).

## Manual Recovery Procedure

Use this when the entrypoint wrapper is not yet deployed, has been bypassed, or encounters a panic pattern it does not recognize.

```sh
# 1. Stop the dependent stack so memexd doesn't try to talk to half-broken Qdrant
docker stop wqm-memexd wqm-mcp wqm-qdrant

# 2. Identify the corrupted collection from the previous Qdrant logs
docker logs wqm-qdrant 2>&1 | grep -oE 'Failed to load local shard "[^"]+"' | sort -u

# 3. Move it aside (preserve for forensics — do NOT delete)
TS=$(date -u +%Y%m%d_%H%M%SZ)
mv state/qdrant/storage/collections/<name> state/qdrant/storage/.corrupted_${TS}_<name>

# 4. Start qdrant alone first; verify it comes up healthy
docker compose -f docker-compose.yml up -d qdrant
docker inspect wqm-qdrant --format '{{.State.Health.Status}}'  # want: healthy

# 5. If qdrant is healthy, bring up the rest
docker compose -f docker-compose.yml up -d
```

Then continue to [Post-Recovery Drift Cleanup](#post-recovery-drift-cleanup).

## Post-Recovery Drift Cleanup

**Why this step matters:** the daemon's SQLite (`tracked_files` and `qdrant_chunks` tables) still believes the now-quarantined collection holds the points it pushed there before the dirty shutdown. After Qdrant recreates the collection empty, `ignore_sync` sees `indexed = <stale N>, eligible = <disk count>`, computes `missing = eligible - N`, and enqueues only the gap. The N "indexed-but-actually-gone" files are silently absent from search until you clear them.

**Recommended cleanup (drops drift, forces a clean re-walk):**

```sh
# Trigger the daemon's existing reembed flow — it drops + recreates the
# canonical collections at the configured dim and re-enqueues from
# watch_folders, rules_mirror, scratchpad_mirror.
wqm admin reembed --confirm
```

This is the safest option because it goes through the daemon's `TriggerReembed` gRPC, which respects ADR-003 (daemon owns all persistent state).

**Faster alternative if you only want to clear the drift for the quarantined collection** (advanced — daemon stopped):

```sh
docker stop wqm-memexd

# Wipe tracked_files + qdrant_chunks for the affected collection only.
# Adjust the WHERE clause if your quarantine touched multiple collections.
docker run --rm -u 1000:1000 -v workspace-qdrant-mcp_memexd_db:/data python:3.12-slim python - <<'PY'
import sqlite3
conn = sqlite3.connect('/data/memexd.db', timeout=30, isolation_level=None)
conn.execute('PRAGMA busy_timeout=30000')
cur = conn.cursor()
cur.execute('BEGIN IMMEDIATE')
# Use the actual quarantined collection name(s) from .quarantine_log
qchunks = cur.execute(
    "DELETE FROM qdrant_chunks WHERE file_id IN ("
    "  SELECT tf.file_id FROM tracked_files tf "
    "  JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id "
    "  WHERE wf.collection = ?)", ('projects',)
).rowcount
tfiles = cur.execute(
    "DELETE FROM tracked_files WHERE watch_folder_id IN ("
    "  SELECT watch_id FROM watch_folders WHERE collection = ?)", ('projects',)
).rowcount
conn.commit()
print(f'cleared {qchunks} qdrant_chunks + {tfiles} tracked_files rows')
cur.execute('PRAGMA wal_checkpoint(TRUNCATE)')
PY

docker compose -f docker-compose.yml up -d memexd
```

After either path, the daemon's `ignore_sync` at next startup sees the wiped state, treats every eligible file as `missing`, and re-enqueues. Expect a fresh bulk drain.

## Verification

```sh
# 1. All four canonical collections exist and respond
for c in projects libraries rules scratchpad; do
  curl -s "http://localhost:6333/collections/$c" | python -c "import sys,json; d=json.load(sys.stdin); print(f'$c: status={d[\"status\"]} points={d[\"result\"][\"points_count\"]}')"
done

# 2. Daemon is healthy + processing
docker inspect wqm-memexd --format '{{.State.Health.Status}}'
docker logs wqm-memexd --since 2m 2>&1 | grep -E "Successfully processed unified item" | wc -l   # > 0

# 3. Grafana 'Unified Queue Depth' panel is draining (or empty)
# Open http://localhost:3000 and inspect the dashboard
```

## Known Panic Patterns

| Panic | Recognized by wrapper? | Notes |
|---|---|---|
| `Failed to load local shard "./storage/collections/<name>/..."` followed by `LiteralOutOfBounds` | Yes | Most common after dirty shutdown — segment writeahead mid-flush |
| `Failed to load local shard "./storage/collections/<name>/..."` with other root cause | Yes | The wrapper quarantines on the "Failed to load" line regardless of the underlying panic |
| `gridstore.rs.*LiteralOutOfBounds` without `Failed to load local shard` line nearby | No (logged but not quarantined — collection name is not in scope) | Manual recovery required; capture the log for an upstream Qdrant report |
| Out-of-memory kills, missing config, permission errors | No (by design) | These are not data corruption; fix the underlying cause |

When you encounter a new panic pattern, extend the regex in `quarantine_corrupted()` inside `docker/qdrant-quarantine-wrapper.sh`. Keep the change minimal: prefer matching `Failed to load local shard` lines (which always include the collection name) over greedy patterns that could mis-quarantine on benign warnings.

## Operator Q&A

**Q: The wrapper made me lose data. Can I get it back?**
A: The data still exists in `state/qdrant/storage/.corrupted_<ts>_<name>/`. The wrapper *moves*, never deletes. Upstream Qdrant tools (`qdrant-storage-tool`, snapshot APIs) may be able to extract surviving segments. For most cases, re-indexing from the source filesystem via `wqm admin reembed --confirm` is faster than salvage.

**Q: How do I know if the wrapper triggered while I was away?**
A: `cat state/qdrant/storage/.quarantine_log` shows every event with its UTC timestamp and collection name. The Grafana dashboard (when the metric panel lands — see the spawn_task referenced in the related memory) will also show a counter.

**Q: Can the wrapper auto-quarantine the wrong collection?**
A: Only collections explicitly named in a `Failed to load local shard "..."` log line are quarantined. The regex is anchored to that exact format. Other panic messages are logged and propagated without quarantine.

**Q: What if Qdrant has been quarantined three times in a row?**
A: After `QDRANT_QUARANTINE_MAX_RETRIES` (default 3), the wrapper gives up and Qdrant stays down. Investigate manually — repeated quarantines on the same collection usually mean either (a) the underlying storage volume itself is corrupted (check disk), or (b) a new panic pattern that's tripping the wrapper to quarantine immediately on retry.

**Q: Does this affect other collections (`libraries`, `rules`, `scratchpad`)?**
A: Only if their segments are also corrupted. The wrapper processes each quarantineable collection name in the panic log independently. Healthy collections are untouched.

**Q: How do I disable the wrapper temporarily?**
A: Remove or comment the `entrypoint:` line for the `qdrant` service in `docker-compose.yml`, then `docker compose up -d --force-recreate qdrant`. Qdrant reverts to its original ENTRYPOINT.

## Forensics: Inspecting a Quarantined Collection

```sh
# List quarantined collections
ls -la state/qdrant/storage/ | grep "^d.*\.corrupted_"

# Inspect a specific quarantined collection
QC="state/qdrant/storage/.corrupted_20260527_144417Z_projects"
ls -la "$QC/0/segments/"            # which segments existed
ls -la "$QC/0/wal/"                  # write-ahead log; usually small after a clean shutdown
cat "$QC/config.json" | python -m json.tool   # collection schema
```

Once you've confirmed the root cause and exported any forensic data you need, the `.corrupted_*` directory can be safely deleted to reclaim disk:

```sh
rm -rf state/qdrant/storage/.corrupted_20260527_144417Z_projects
```

Do not delete `.quarantine_log` — it's the audit trail that points at the original incident timestamps.

## Related Documentation

- [BACKUP_RESTORE.md](../BACKUP_RESTORE.md) — proactive snapshots that reduce reliance on this recovery path
- [ARCHITECTURE.md](../ARCHITECTURE.md) — full data flow showing where Qdrant fits between the daemon and search results
- Memory: `project_qdrant_corruption_autofix.md` — concise summary of this runbook for cross-session Claude access
