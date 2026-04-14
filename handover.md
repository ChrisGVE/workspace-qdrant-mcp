# Handover — 2026-04-14 (Session 7 complete)

## Current State

Branch `feature/cli-ux-ui-review`, pushed to remote. **All 3 phases complete.** Tasks 36-65 done. 494 tests passing.

## What's Been Done (Sessions 1–6)

### Phase 1: Core Data Layer (complete)
Created `src/rust/cli/src/data/` with 4 submodules:
- `db.rs` — single `connect_readonly()` replacing 6 duplicates, all 33+ callers migrated
- `queries.rs` — canonical typed queries: QueueStats, HealthLevel, ProjectInfo (with date fields), DocumentCounts, ReconcileStats, LibraryInfo, get_languages (1% threshold). 14 unit tests.
- `health.rs` — `check_qdrant()` with 3s timeout
- `orphans.rs` — `detect_orphans()` via Qdrant scroll (slow, intentionally not used in project list)

### Phase 2: Presentation Layer (complete)

**Output infrastructure:**
- Gutter symbols: `●` green, `○` yellow, `◆` blue, `▲` yellow, `✗` red, `·` blue. `SYMBOL_WIDTH=1` (tables) vs `WIDTH=2` (columnar).
- ColumnarBuilder: section auto-indent (rule 2), anonymous nested groups (`nested("", b)`), `aligned_group()` for right-aligned decompositions (rule 4 exception).
- Table: `render_table` uses `SYMBOL_WIDTH` for correct gutter alignment. `ExpandEven` peaker for rule 14 even spread.
- `format_date_short()` — dd-mmm-yyyy formatter.
- `peakers` and `table` modules promoted to `pub(crate)`.
- `render.rs` — extracted gutter-aware rendering (GutterRow, render_table, print_table_summary).

**Commands redesigned (13 total):**

| Command | Template | Notable |
|---------|----------|---------|
| `wqm status` | columnar | queue decomposition, `-v` (project names + per-project queue) |
| `wqm project list` | table (Builder) | optional date cols at width 130/150/170, `-v` (ID/Languages/Chunks) |
| `wqm project status` | columnar | reconcile stats with gutter, active branch display |
| `wqm library list` | table (Builder) | ●/·/▲ gutter for watching/auto-routed/orphan |
| `wqm queue stats` | columnar | status decomposition via aligned_group |
| `wqm status health` | columnar | per-component health gutter |
| `wqm service status` | columnar | per-component health gutter |
| `wqm graph stats` | columnar | node/edge type counts via aligned_group |
| `wqm status watch` | columnar | active project list with nested builder |
| `wqm status history` | columnar | per-metric sections (latest/avg/min/max/samples) |
| `wqm status messages` | columnar | log locations, error metrics from daemon |
| `wqm admin perf` | table + columnar | ColumnHints on PerfRow, summary via ColumnarBuilder |
| `wqm admin collections list` | table (render_table) | GutterRow with Sync/None, ColumnHints |

### Phase 3: CLI structure and polish (complete — Session 6)

**Task 54: Help system overhaul**
- `-h` shows compact syntax-only help (no examples)
- `--help` shows verbose help with description and examples
- All `after_help` → `after_long_help` in main.rs (17 commands)
- Fixed stale references to removed commands (info, check, watch list, branch list)
- Updated descriptions to match current subcommands

**Task 55: Remove -V/--version from subcommands**
- Already correct — clap doesn't propagate version to subcommands

**Task 56: Standardize command naming**
- Already correct — register/delete/add/remove consistent across domains

**Task 57: Remove redundant project commands**
- `info` and `check` already removed in prior sessions

**Task 58: Merge watch list into project list**
- `wqm watch` already hidden; project list shows active/inactive status

**Task 59: Branch info in project status**
- Added `current_git_branch()` — shows active branch via `git rev-parse`

**Task 60: Unify project detection**
- Already unified — all commands use `resolver.rs` with path/ID/name/worktree support

**Task 61: Register pre-check**
- Already implemented — checks SQLite for existing registration before prompting

**Task 62: Delete worktree warning**
- Added explicit warning when running `wqm project delete` from a worktree directory

**Task 63: Write path audit**
- All user-facing writes use gRPC; only `recover_state` does direct SQLite (intentional)

**Task 64: Data module tests**
- Added 7 new tests: total_document_count, all_document_counts, active_collection_count, queue health levels, null field handling
- Total: 494 tests passing

**Task 65: TTY/pipe output tests**
- Added 4 tests: pipe width default, minimum width, script ANSI-free, strip_ansi correctness

**Hidden library commands:** `status`, `rescan`, `config` hidden from help but still functional

## Key Files

### Data layer
- `src/rust/cli/src/data/queries.rs` — canonical queries (14 tests)
- `src/rust/cli/src/data/db.rs` — unified SQLite connection
- `src/rust/cli/src/data/health.rs` — service health checks

### Output infrastructure
- `src/rust/cli/src/output/columnar.rs` — ColumnarBuilder
- `src/rust/cli/src/output/table.rs` — generic table utilities
- `src/rust/cli/src/output/render.rs` — gutter-aware rendering
- `src/rust/cli/src/output/gutter.rs` — symbol set
- `src/rust/cli/src/output/tests.rs` — 26 output tests

### Reference implementations
- `src/rust/cli/src/commands/status/overview.rs` — best columnar example
- `src/rust/cli/src/commands/project/list.rs` — best table example
- `src/rust/cli/src/commands/graph/stats.rs` — clean aligned_group example

### CLI structure
- `src/rust/cli/src/main.rs` — help template, command grouping
- `src/rust/cli/src/commands/project/resolver.rs` — unified project detection
- `src/rust/cli/src/commands/project/delete.rs` — worktree warning

## Task-Master State

- **Tag**: `cli-ux-ui-review`
- **Tasks 36-65**: All done
- **No remaining tasks in this tag**

Tests: 494 passing.
