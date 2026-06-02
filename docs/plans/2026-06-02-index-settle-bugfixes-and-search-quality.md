# 2026-06-02 — ext4 stabilization: index-settle bug fixes + search-quality measurement

## Context

After migrating the repo to native **ext4** (`/home/alkmimm/respositorios/workspace-qdrant-mcp`,
Docker WSL-integration bind-mount, no 9P), the daemon was stable when idle but
**the index never reached a settled state** under reconcile, which blocked any
trustworthy semantic-search quality measurement (`search_eval` / the host
benchmark time out or run against a churning daemon). This session root-caused
and fixed the settle blockers, then measured search quality on a settled index.

All daemon edits were built into `workspace-qdrant-mcp-memexd:local` and deployed.
Code is in the Windows checkout (git source of truth) and mirrored to the ext4
build copy. **Not yet committed** — awaiting go-ahead.

## Bugs fixed (6) + verification

| # | Bug | Root cause | Fix (file) | Verified |
|---|-----|-----------|------------|----------|
| 1 | Daemon clean **exit-0 crash** every ~6-7 min under reconcile | SQLite connection-pool starvation: 10-conn pool + per-acquire validation ping + 10s/30s exporters running 28s full-table `unified_queue` GROUP BYs → multi-second lock/pool waits → daemon recycle | pool 10→24, min 2→4, `test_before_acquire`→false (`queue_config.rs`); queue-depth exporter 10s→60s, inventory exporter 30s→300s (`memexd/background.rs`) | 15+ min reconcile, RestartCount=0, `slow_stmts=0` |
| 2 | Container **unstoppable** (zombie wedge) | memexd is PID 1 and didn't reap child procs (LSP/grammar/git) → `PID is zombie and can not be killed` | `init: true` on the memexd compose service (`docker-compose.yml`) | PID 1 = `docker-init`, clean recreate |
| 3 | **Folder-scan re-enqueue loop** (queue stuck ~900 pending) | folder-scan idempotency key hashed the volatile `last_scan` timestamp → every pass got a new key → `INSERT OR IGNORE` never deduped (folders aren't covered by the partial `file_path` UNIQUE index either) | `idempotency_payload_json()` strips `last_scan` from the key payload only; stored payload keeps it for mtime pruning (`queue_operations/enqueue.rs`) | pending folder-scans 923→27 |
| 4 | **Project `.wqmignore` ignored** (eval-artifact leakage) | scan path called `ProjectIgnoreMatcher::for_dir(dir, None)` (subdir-only, root `.wqmignore` not inherited — issue #49 regressed); reconciliation's `add_custom_ignore_filename` didn't honor root-anchored deep paths in a git repo | scan passes `Some(watch_folder_root)` (`folder/scan.rs`); reconciliation adds explicit `add_ignore(project_root/.wqmignore)` (`reconciliation/ignore_sync.rs`) | `reports/`+`benchmark-data/` points → 0 |
| 5 | **Git-watcher re-emit loop** (~5 events/sec) | `process_events` emitted a `GitEvent` on every notify event without comparing to last; `parse_reflog_last_entry` keeps returning the same stale entry (the original `clone`, old=0000000) | dedup on `(old_sha, new_sha, branch)`; skip unchanged (`git/watcher.rs`) | git events 45/45s → 0 |
| 6 | **File-watcher phantom-event churn** (~4700 `op=Update` hash-skips/min on mtime-stable files) | the file watcher enqueued `File/Update` on every notify event including `Access`/`Modify(Metadata)` (atime/ctime), which fire spuriously across the Docker bind-mount and from the daemon's own reads | filter non-content events at ingestion — drop `Access(_)` and `Modify(Metadata(_))` (`watching_queue/file_watcher.rs`) | churn 4700→~1600/min, **pending → 0 (index settles)** |

**Bonus (verified deployed):** the `wqm project activate` empty-path / LSP-cache-poison
bug (#70) — `activate_project_side_effects` now resolves the real root from
`watch_folders` when the request path is empty (`project_service/registration.rs`).
`wqm project activate <id>` now spawns 7 LSP servers (was 0 + poisoned the
language-detection cache).

## Search-quality measurement (settled index, pending=0)

Host runner, canonical tenant `367157a01d98`, 12-query dataset, sidecar `memexd.db`.

| Mode | top1 | top3 | top10 | recall@10 | MRR | avg ms |
|------|------|------|-------|-----------|-----|--------|
| semantic | 17% | 50% | 67% | **38%** | 0.34 | 932 |
| hybrid | 25% | 58% | 67% | **38%** | 0.39 | 1198 |
| (baseline, committed report) | 17% | 50% | 83% | **71%** | 0.39 | **45** |

**Finding — the cross-encoder reranker is halving recall@10.** The recall gap is
**not** the churn (settled `pending=0` gives the same 38% as the churning runs)
and **not** incomplete indexing (the missed expected files — e.g.
`watching/telemetry.rs`, `processing/metrics.rs` — are confirmed indexed). The
tell is latency: the baseline ran at **45 ms** (bi-encoder + path-boost, **no
cross-encoder**) and scored recall **71%**; the current path runs at **932 ms**
(**with** the cross-encoder rerank) and scores recall **38%**. Inspecting
`topResults`, implementation `.rs`/`.ts` files receive **negative** cross-encoder
scores while docs/specs/config rank at the top for "where is X implemented"
queries — so rerank pushes relevant code out of the top-10. The pre-rerank,
path-boosted order (`deduped` in `search-helpers.ts`) is the one that achieved 71%.

## Recommended next steps

1. **Confirm + fix the reranker regression (highest quality leverage).** Run an
   A/B with `rerank: false` (the search option already gates it,
   `search-helpers.ts:550`) on the settled index — expectation: recall@10
   jumps back to ~71%. If confirmed, either disable cross-encoder rerank for
   code-heavy "where is X" queries, blend rerank score with the bi-encoder +
   path-boost score instead of replacing the order, or use a code-aware
   reranker. The current cross-encoder favors prose over code.
2. **Eliminate the residual file-watcher churn (~1600/min).** The event-kind
   filter cut it ~65% and got `pending` to 0, but content-kind phantom events
   (`Modify(Data/Any)`) still fire on the Docker bind-mount. Add enqueue-time
   change-detection (per-path (mtime,size) cache or a `tracked_files` lookup)
   so unchanged files are not enqueued at all.
3. **Commit the fixes** (8 files) once reviewed.

## Files changed

`queue_config.rs`, `memexd/background.rs`, `docker-compose.yml`,
`queue_operations/enqueue.rs`, `folder/scan.rs`, `reconciliation/ignore_sync.rs`,
`git/watcher.rs`, `watching_queue/file_watcher.rs`,
`project_service/registration.rs`.
