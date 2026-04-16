# Handover — 2026-04-16 (Session 9)

## Current State

Branch `main`, not pushed. 6 new commits (634b8e1dc..b3ce55a8a). Qdrant currently down (connection refused). Daemon running, schema v34.

## What Was Done (Session 9)

### CLI Status Display Overhaul (6 commits)

Reworked `wqm status` and `wqm status -v` per cli-feedback.md design guidelines.

**Completed:**
1. **Prefixed entity names** — tenant IDs replaced with `prj:`, `lib:`, `rls:`, `scp:` prefixed human-readable names. `build_full_tenant_name_map()` in `watch/helpers.rs` covers projects + libraries. `prefixed_display_name()` adds collection prefix.
2. **Underline on Total** — underline under Queue Total line, indented to start under key text.
3. **Two-column entity layout** — Queue by Entity section renders entity pairs side-by-side per double-columnar template spec. Falls back to single column when terminal too narrow.
4. **Active Projects in compact mode** — project names shown in both compact and verbose.
5. **Annotations column** — right-aligned annotation column added to ColumnarBuilder (`kv_annotated`, `kv_underline_annotated`, `aligned_group_annotated`). Response times, health reason, avg processing, ETA rendered as annotations.
6. **ColumnarBuilder.full_width()** — separators span full terminal width in verbose mode for hybrid layout compliance.

**Key files changed:**
- `src/rust/cli/src/commands/status/overview.rs` — main status display + two-column renderer
- `src/rust/cli/src/output/columnar.rs` — annotation support, full_width, underline, literal_key
- `src/rust/cli/src/commands/watch/helpers.rs` — `build_full_tenant_name_map`, `prefixed_display_name`, `collection_prefix`
- `src/rust/cli/src/commands/watch/mod.rs` — made helpers `pub(crate)`
- `src/rust/cli/src/data/queries.rs` — `health_reason()`, `get_avg_processing_ms()`

**Tests:** CLI 497 pass (3 new helper tests added)

## Pending Feedback — Address Next Session

### 1. Verbose-only annotations
Response times, queue health reason, avg processing time, ETA are currently shown in BOTH compact and verbose. User wants these **verbose-only**. Compact should show just the core values without annotations.

### 2. Narrow terminal output mangled
When terminal is narrow, output wraps/mangles. The two-column entity section has a fallback, but the main columnar block does not handle narrow terminals. Lines with long annotation text overflow.

### 3. Header not consistent with design guidelines
Verbose mode header (the main columnar section) does not fully comply with cli-feedback.md. The separator lines span full width but content is left-aligned with wide empty space to the right. Needs review against the columnar template spec points 1-5.

### 4. Queue health reason unclear
Current reason string: `oldest pending: 5d 16h (>24h), 132 failed`
- Not clear what each piece means. Thresholds from `queries.rs`:
  - **Unhealthy**: oldest pending > 24h OR failed > 10% of active
  - **Degraded**: failed > 0 OR oldest pending > 1h
- User asks: if both age and failed count are shown, what is the comparison point for failed? The `(>24h)` threshold is shown but no threshold for failed count. Need clearer labeling, e.g. `stale: 5d 16h` and `failures: 132 (0.1%)` with explicit threshold.

### 5. Processing metrics labeling
- `avg 7.6s/item` should say "avg processing: 7.6s/item" to clarify it's processing time not wall time
- `est. 12d 3h` should indicate it's estimated time to drain the entire queue, not just pending

### 6. Consider existing crates for terminal formatting
User feels we're reinventing the wheel with custom columnar/table formatting. Research terminal output crates that handle:
- Responsive layout (adapt to terminal width)
- Columnar/table formatting with alignment
- Two-column layouts
- Candidates to evaluate: `ratatui` (already in deps for TUI), `comfy-table`, `tabled` (already in deps), `terminal_size` (already in deps), `textwrap`
- Assess whether replacing custom `ColumnarBuilder` with an existing crate is worthwhile vs continuing to extend it.

## Remaining from Session 8

### Task 18: ✅ Daemon verified
Schema v34 confirmed. `ignore_file_mtimes` and `scratchpad_mirror` tables present.

### Task 19: E2E scratchpad test — BLOCKED
Scratchpad entry stored in SQLite mirror but stuck in 139K-item queue backlog. Queue needs draining before E2E can complete. `wqm admin rebuild scratchpad` enqueues items but doesn't bypass queue. Consider clearing stale backlog first.

### Task 20: Scratchpad edge cases — BLOCKED on Task 19

### Task 24: Tag v0.1.1 — BLOCKED on Tasks 19-20

## Queue Backlog Issue
~139K items pending from when Qdrant was offline for days. Queue processes ~3 items concurrently at ~7.6s/item avg. ETA ~12 days to drain naturally. Options:
1. Clear stale project items (`DELETE FROM unified_queue WHERE status='pending' AND collection='projects'`) — they'll re-enqueue via next rebuild
2. Wait for natural drain
3. Increase parallelism in daemon queue processor

## Build Environment
```bash
export LIBRARY_PATH="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang/21/lib/darwin:${LIBRARY_PATH}"
export ORT_LIB_LOCATION=/Users/chris/.onnxruntime-static/lib
```

## Test Counts
CLI: 497 pass | Core: 2593 pass, 16 ignored | gRPC: 143 | TS: 424+2 skip
