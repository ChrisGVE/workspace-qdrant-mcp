# Handover — 2026-03-11

## Current State

All three `smart-processing` tasks are complete. Two commits on the `resource-management`
branch have been pushed to remote. The branch is NOT yet merged to `main` — it was kept
separate for experimentation per user instructions.

## Completed Work (smart-processing tag)

### Task 1 — mtime-based scan pruning (commit `aacfd8d6f`, main)

Eliminates redundant queue churn on daemon restart. `FolderPayload` now carries
`last_scan: Option<String>`. During `scan_directory_single_level`, files with
`mtime ≤ last_scan` are skipped. The timestamp is propagated down the directory tree in
each child's payload so child scans skip the DB query. `handle_project_scan` reads
`last_scan` from `watch_folders` before scanning and updates it after.

### Task 2 — CLAUDE.md workspace-qdrant mandatory first tool (committed, main)

Global `~/.claude/CLAUDE.md` and `README.md` CLAUDE.md snippet both updated to:
- Three-step protocol: start with workspace-qdrant → fall back only if insufficient → store findings
- Mandatory for main thread AND every sub-agent (verbatim instruction required in each agent prompt)

### Task 3 — Adaptive resource management (branch: `resource-management`)

Two gaps in the existing adaptive resource manager were closed:

**Semaphore scale-down** (`loop_core.rs`):
When the adaptive profile target decreases, `embedding_semaphore.forget_permits(excess)`
is now called. Previously scale-down was documented as a known gap (permits only drained
naturally as in-flight ops completed).

**Active Processing Mode wiring** (`manager.rs`, `queue_init.rs`):
`AdaptiveResourceManager::start()` now accepts `queue_depth: Option<Arc<AtomicUsize>>`.
The `queue_depth_counter` from `UnifiedQueueProcessor` is passed at init time.
When user is present (not idle) and state machine is at Normal but queue has pending work,
the manager emits the `Active` profile (+50% concurrency, 25ms delay) as an overlay —
the state machine level stays at Normal so ramp-up/ramp-down is unaffected.

Spec `docs/specs/04-write-path.md` updated to reflect actual heartbeat log format.

## Branch Status

- `main`: Tasks 1 and 2 merged and pushed.
- `resource-management`: Task 3 (2 commits). Awaiting user decision to merge.

## No Pending Work

All tasks complete. Await new instructions.
