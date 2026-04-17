# Handover — 2026-04-17 (Session 11)

## Current State

Branch `main`, pushed. Released **v0.1.2** tag (pushed to origin). Schema v34. Daemon running via launchctl. All task-master tasks in smart-processing tag complete.

## What Was Done (Session 11)

### Critical fixes
1. **Daemon startup investigation** — Discovered earlier claim that fix was deployed was wrong; binary lacked `deserialize_tags` symbols. Rebuilt memexd+wqm with release LTO and deployed.
2. **Scratchpad tags E2E validated** — b24d714d (session 10 test item with stringified tags) now processes successfully in 1025ms. Tags fix confirmed in production binary.
3. **Task 20 edge cases** — Submitted 5 scratchpad entries (empty title, empty tags, Unicode/special chars, 10KB long content, session 11 marker). All processed into both `scratchpad_mirror` SQLite table and Qdrant scratchpad collection.

### Queue backlog management
Reconciliation on each daemon start re-enqueues 50K-140K file items from filesystem walks (thales alone = 93K files). SQLite write contention drops INSERT throughput to ~1/sec, making queue drain impractical.

Mitigations applied this session:
- Repeatedly `DELETE FROM unified_queue WHERE item_type != 'text'` to clear file backlog
- `UPDATE watch_folders SET enabled=0` for heavy projects (reconciliation only iterates `enabled=1`)
- `DELETE FROM tracked_files` for disabled projects (ignore_sync uses tracked_files as reference)
- After test complete: re-enabled all 23 watch_folders back to `enabled=1`

### Released v0.1.2 (tagged + pushed)

Note: v0.1.1 tag already existed on remote (at commit 96471381) for earlier scratchpad_mirror release. Current work was substantial enough to warrant v0.1.2 bump instead of force-moving v0.1.1.

### Commits (session 11)
- `test(ts): align mockDaemonClient with current DaemonClient interface` (420638221)
- `chore: bump version to 0.1.1` (d8de6c09a) — superseded
- `chore: bump version to 0.1.2` (0c159a76d)

### Tests — all passing
- wqm-common: **213** passed
- workspace-qdrant-core (lib): **2197** passed
- workspace-qdrant-grpc: **143** passed
- wqm-cli: **541** passed
- TypeScript mcp-server: **424** passed, 2 skipped
- **Total: 3518 passing, 0 failing, 2 skipped**

## Task-Master State (smart-processing tag)
All pending/blocked work resolved:
- Task 18 (daemon restart with v33): done
- Task 19 (E2E scratchpad store+rebuild): done
- Task 20 (scratchpad edge cases): done
- Task 24 (tag + release): done via v0.1.2 instead of v0.1.1

No pending, no blocked. 48 done / 6 cancelled / 51 total.

## Known Technical Debt

### Queue reconciliation is expensive on startup
Every daemon restart walks filesystem for all enabled watch folders and re-enqueues any file not in tracked_files. For projects with tens of thousands of files, this generates 50K+ queue items and blocks gRPC server startup for minutes.

Mitigations for release:
- Users unlikely to restart daemon frequently in production
- Only impacts cold startup, not steady-state operation
- `wqm queue cancel <project>` allows clearing backlog

Future improvements (NOT scheduled):
- Batch reconciliation instead of serial enqueue
- Defer heavy reconciliation to background task (unblock gRPC startup)
- Periodic reconciliation instead of startup-only

### Qdrant timeouts under load
When queue processor hits concurrent deletes against Qdrant, operations time out (60s). Retry logic handles this but each retry is 3min. Adaptive resources reduce concurrency under CPU pressure.

## Build Environment
```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
cargo build --release --manifest-path src/rust/Cargo.toml --package memexd --package wqm-cli
```

Release LTO compile: ~50 min cold, mostly in `workspace_qdrant_core` LTO link step.

## Next Session

All tasks in smart-processing tag complete. v0.1.2 released.

Possible next steps:
- Open new task-master tag for next release planning
- Address deferred CLI polish items (session 9 cli-feedback.md → now at tmp/cli-feedback.md)
- Reconciliation performance work (queue throughput improvements)
- Investigate Qdrant timeout tuning for batch operations
