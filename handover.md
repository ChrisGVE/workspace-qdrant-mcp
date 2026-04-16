# Handover — 2026-04-16 (Session 10)

## Current State

Branch `main`, pushed. 9 new commits this session (79e0af852..c123aed79). Schema v34. **Daemon not starting** — exits code 1 with no stderr; investigate on next session.

## What Was Done (Session 10)

### TUI Feature Completion (8 commits)

Built from the existing 6820-line TUI (Dashboard, Queue, Projects, Libraries, Logs) to a complete interactive terminal UI.

**New features:**
1. **Shared theme module** (`tui/theme.rs`) — centralized colors, alarm state (dark red bg), gutter symbols, composite styles
2. **Rules screen** — scrollable list from `rules_mirror` table, detail popup with full rule text
3. **Scratchpad screen** — read-only list from `scratchpad_mirror` table, scrollable detail popup
4. **Service screen** — daemon/Qdrant status with alarm indicator, queue summary, active counts
5. **Bottom status bar** — contextual keybinding hints per view, SERVICE DOWN alarm banner
6. **Search (`/`)** — reusable `SearchState` with case-insensitive substring matching across all 5 list views
7. **Toggle commands** — pause/resume watchers via gRPC from Service view (p/r keys)
8. **Destructive guard** (`tui/guard.rs`) — two-step confirmation: type exact identifier → type "yes"
9. **gRPC commands** — rebuild collection, delete project wrappers in `tui/commands.rs`

**CLI cleanup:**
- Removed `admin collections list` and `admin collections reset` (internal, not user-facing)

**TUI now has 8 views:** Dashboard (1), Queue (2), Projects (3), Libraries (4), Rules (5), Scratchpad (6), Service (7), Logs (8)

### Scratchpad Tags Bug Fix (1 commit)

Fixed `ScratchpadPayload` tags deserialization: MCP clients send tags as stringified JSON array (`"[\"a\",\"b\"]"`) instead of native array. Added custom `deserialize_tags` that handles both forms. This was causing permanent queue failures.

### Queue Backlog Cleared

Purged 137K stale project pending items from `unified_queue`. These accumulated while Qdrant was offline. Items will re-enqueue on next rebuild.

## Key Files Changed

### TUI (new/modified)
- `src/rust/cli/src/tui/theme.rs` — shared color palette and styles
- `src/rust/cli/src/tui/search.rs` — reusable search/filter state
- `src/rust/cli/src/tui/guard.rs` — destructive command guard dialog
- `src/rust/cli/src/tui/commands.rs` — gRPC command wrappers (pause/resume/rebuild/delete)
- `src/rust/cli/src/tui/app.rs` — 8 views, status bar, alarm state, key handlers
- `src/rust/cli/src/tui/views/rules.rs` + `rules_data.rs`
- `src/rust/cli/src/tui/views/scratchpad.rs` + `scratchpad_data.rs`
- `src/rust/cli/src/tui/views/service.rs`

### Bug fix
- `src/rust/common/src/payloads/content.rs` — `deserialize_tags` for stringified arrays

### CLI cleanup
- `src/rust/cli/src/commands/admin/mod.rs` — removed Collections subcommand

## Pending — Next Session

### 1. Daemon Not Starting (CRITICAL)
After deploying rebuilt binaries and restarting, daemon exits code 1 with no stderr. Was running fine before rebuild. Need to investigate:
- Run `memexd` directly with `RUST_LOG=debug` to get output
- Check if it's an ORT linking issue, config issue, or port conflict
- Previous daemon was running schema v34 fine

### 2. E2E Scratchpad Test (Task 19)
Queue was cleared, tags bug fixed, but daemon needs to be running first. Once daemon is up:
- Retry the failed scratchpad queue item (b24d714d) or create a new one
- Verify it processes successfully with the tags fix
- Confirm entry appears in both Qdrant and scratchpad_mirror

### 3. Scratchpad Edge Cases (Task 20)
After task 19 passes:
- Test empty content, very long content, Unicode content
- Test duplicate detection (idempotency key)
- Test tag variations (empty, many tags, special chars)

### 4. Tag v0.1.1 (Task 24)
After tasks 19-20 pass. Dependencies 22, 23 already done.

### 5. CLI Status Feedback (from Session 9 — deferred)
Items from cli-feedback.md that were deferred:
- Verbose-only annotations (currently shown in compact too)
- Narrow terminal overflow handling
- Header compliance with columnar template spec
- Queue health reason unclear labeling
- Processing metrics labeling clarity
- These are polish items, not blocking release

## Task-Master State (smart-processing tag)
- 47/51 done, 6 cancelled (unnecessary abstraction), 1 pending (24), 3 blocked (18→19→20, but 18 is functionally done — just task-master not updated)
- Task 18: functionally done (daemon was running v34). Mark done once daemon starts again.
- Task 19: blocked on daemon running + tags fix deployed
- Task 20: blocked on 19
- Task 24: deps 22/23 done, can proceed after 19/20

## Test Counts
CLI: 541 pass | Core: 2593 pass, 16 ignored | gRPC: 143 | Common: 223 | TS: 424+2 skip

## Build Environment
```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
```

## PRD
TUI PRD: `.taskmaster/docs/20260416-1841_project_0.1.1_PRD_tui-renderer-unification.md`
