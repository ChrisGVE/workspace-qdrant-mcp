# Handover — 2026-03-11

## Current State

All tasks in the `perf-extract` tag are complete and pushed to main.

## Completed Tasks (perf-extract tag)

1. Embedding cache reuse
2. Hapax eviction
3. LRU phrase cache
4. Background persistence
5. IDF drift correction (`wqm admin rebalance-idf`)

## Bug Fix (committed to main, 2026-03-11)

Diagnosed and fixed two bugs that caused 4 queue items to loop indefinitely:

- `mark_unified_retry` / `mark_unified_permanent` / `resurrect_failed_transient` now
  reset `qdrant_status = NULL` and `search_status = NULL` when requeueing, preventing
  stale `in_progress` from surviving across retry cycles.
- The "file no longer exists" cleanup path now explicitly marks both destinations `Done`
  before returning `Ok(())`, so `check_and_finalize` can delete the item.

All 4 stuck items were cleared immediately after daemon restart.

## Feature (committed to main, 2026-03-11)

CLI path arguments now expand tilde and env vars before use (`feat(cli): expand tilde and env vars in path arguments`).
Affected commands: `library add/watch/ingest/set-incremental`, `ingest file/folder`, `backup create --output`.
Implementation: `wqm-common::env_expand::expand_path`, `cli::path_arg::parse_path` clap value_parser.
Command substitution `$(...)` remains shell-level only.

No pending work. Await new instructions.
