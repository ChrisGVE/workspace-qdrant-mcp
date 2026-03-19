# Handover: Daemon-Exclusive SQLite Write Path

## Branch
`worktree-state-db` (git worktree at `.claude/worktrees/state-db`)

## PRD
`.taskmaster/docs/20260319-1720_state-db_0.1.0_PRD_daemon-exclusive-writes.txt`

## Status: WriteActor Implemented

The daemon is now the **sole writer to state.db**. Both CLI and MCP server send all mutations via gRPC and use read-only SQLite connections. All gRPC write services now serialize mutations through a WriteActor channel.

## What's Done

### Phase 1: Proto definitions (Tasks 2-4)
- 5 new gRPC write services in `workspace_daemon.proto`: QueueWriteService (7 RPCs), WatchWriteService (6), LibraryWriteService (6), TrackingWriteService (4), AdminWriteService (2)
- Rust server stubs + TypeScript client types generated

### Phase 2: Daemon handlers (Tasks 3, 5-9)
- All 5 service handlers implemented (now delegating to WriteActor)
- All registered in gRPC server factory (`factory.rs`)

### Phase 3: CLI migration (Tasks 11-16)
- `DaemonClient` extended with 5 write service clients + `ensure_daemon_available()` helper
- ALL CLI write commands migrated to gRPC (queue, watch, library, admin, stats, ingest, scratch, rules)
- `UnifiedQueueClient` deleted (queue.rs removed)
- All CLI SQLite connections switched to read-only
- Dead code removed (connect_readwrite, resolver, signal_daemon_watch_folders)

### Phase 4: MCP server migration (Tasks 17-21)
- `DaemonClient` extended with `enqueueItem`, `logSearchEvent`, `updateSearchEvent`, `upsertRuleMirror`, `deleteRuleMirror`
- `enqueueUnified` rewritten to use gRPC (returns degraded result when daemon unavailable)
- Search event logging/updating uses gRPC (fire-and-forget)
- Rules mirror upsert/delete uses gRPC (fire-and-forget)
- SQLite connection switched to `readonly: true`

### Phase 5: WriteActor (Tasks 22-25) - DONE
- `CleanQueueByCollection` RPC added to QueueWriteService (7th RPC)
- CLI `collections/reset.rs` migrated from direct SQLite to gRPC
- WriteActor module in `daemon/core/src/write_actor/`:
  - `commands.rs` — WriteCommand enum with 24 variants (all gRPC write RPCs)
  - `actor.rs` — WriteActor task + WriteActorHandle with typed async helpers
  - `exec_queue.rs` — SQL execution for 7 queue commands
  - `exec_watch.rs` — SQL execution for 6 watch commands
  - `exec_library.rs` — SQL execution for 6 library commands
  - `exec_tracking.rs` — SQL execution for 4 tracking commands
  - `exec_admin.rs` — SQL execution for 2 admin commands
- All 5 gRPC write services rewritten as thin wrappers around WriteActorHandle
- Factory spawns WriteActor from pool (or uses externally injected handle via `with_write_actor`)
- Builder method `with_write_actor()` added to GrpcServer

### Phase 6 (partial): Cleanup (Tasks 26-29)
- Dead code removed from CLI and MCP server
- `docs/specs/04-write-path.md` updated with new architecture
- Proto header comments updated

## Deferred (Future PRs)

### Internal daemon mutations via WriteActor
The WriteActor currently handles gRPC write service commands only. Internal daemon mutations (queue processor dequeue/mark status, file tracker, metrics) stay on the pool. Future PR to migrate them.

### E2E and load tests (Tasks 30-32)
Require daemon start/stop orchestration. Best done as integration tests in a separate PR.

### Remaining direct writes
- `recover-state` — intentionally direct (daemon not running during recovery)

## Key Decisions
1. No `busy_timeout` fix for MCP server — went straight to read-only
2. gRPC write services are domain-scoped (not monolithic) for extensibility
3. `recover-state` stays as direct SQLite write
4. `rename-tenant` removed direct SQLite fallback — requires daemon
5. TrackingWriteService errors logged but not propagated (instrumentation never blocks)
6. WriteActor uses `String` errors (not `tonic::Status`) to avoid coupling core crate to tonic
7. Factory auto-spawns WriteActor if no external handle is injected

## How to Resume
1. Create PR from `worktree-state-db` to `main`
2. Future PR: Migrate internal daemon mutations to WriteActor
3. Future PR: E2E tests with daemon start/stop orchestration
