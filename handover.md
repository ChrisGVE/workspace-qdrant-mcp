# Handover — Memory Graph Implementation

## Current State

### Completed This Session

1. **Task #21 (arch-refactor): Split WORKSPACE_QDRANT_MCP.md into modular specs** (commit `cb7e613e7`)
   - Split 6141-line monolithic spec into 15 files under `docs/specs/`
   - Main file now a 56-line table of contents with links
   - All cross-references updated (cross-file anchors, ADR paths, relative links)
   - CLAUDE.md updated to reference new structure
   - Verified: all content preserved, no broken links

2. **Created Memory Graph (Graph RAG) PRD and task-master tasks**
   - PRD: `.taskmaster/docs/20260223-1648_workspace-qdrant-mcp_0.1.0-beta1_PRD_memory-graph.md`
   - Tag: `graph-rag` (8 tasks, all pending)
   - Comprehensive plan covering Phase 1 (SQLite CTEs) with Phase 2 (LadybugDB) path documented

### Task-Master Tasks (graph-rag tag)

| ID | Title | Status | Dependencies |
|----|-------|--------|-------------|
| 1 | Graph Module Foundation and SQLite Schema | pending | none |
| 2 | Graph Relationship Extractor | pending | 1 |
| 3 | Ingestion Pipeline Integration | pending | 1, 2 |
| 4 | Read-Write Concurrency with RwLock | pending | 1 |
| 5 | gRPC GraphService Implementation | pending | 1, 4 |
| 6 | CLI Graph Commands | pending | 5 |
| 7 | Performance Validation and Benchmarking | pending | 3 |
| 8 | MCP Search Graph Context Integration | pending | 5 |

## Next Session Instructions

### Priority 1: Execute graph-rag tasks

Switch to graph-rag tag and begin executing tasks in dependency order:

1. **Task 1** (Graph Module Foundation) — start here, no dependencies
   - Create `src/rust/daemon/core/src/graph/` module tree
   - `GraphStore` trait, data types, `SqliteGraphStore`, SQLite schema
   - Dedicated `graph.db` (separate from `state.db`)
   - Recursive CTE queries for N-hop traversal
   - Unit tests

2. **Tasks 2 + 4** can run in parallel after Task 1:
   - Task 2: Graph Relationship Extractor (extract edges from SemanticChunk)
   - Task 4: Read-Write Concurrency (RwLock wrapper)

3. **Task 3** after 1 + 2: Ingestion Pipeline Integration
4. **Task 5** after 1 + 4: gRPC GraphService
5. **Tasks 6, 7, 8** after their dependencies

### Key Design Decisions (Already Made)

- **Single daemon** — graph embedded in memexd, no separate process
- **Phase 1: SQLite CTEs** — zero new dependencies, covers 1-2 hop queries
- **Separate graph.db** — avoid lock contention with state.db queue operations
- **Non-blocking graph failures** — graph errors logged, don't block ingestion
- **`GraphStore` trait** — enables future LadybugDB swap (Phase 2)

### Existing Code to Build On

- `SemanticChunk` in `src/rust/daemon/core/src/tree_sitter/types.rs` — already has `calls`, `parent_symbol`, `signature`
- `cooccurrence_graph.rs` in `keyword_extraction/` — symbol extraction and pair generation
- `lsp_candidates.rs` — import/use statement extraction
- `search_db.rs` — reference for SQLite setup patterns
- `schema_version.rs` — reference for migration patterns

## Key Files

| File | Purpose |
|------|---------|
| `WORKSPACE_QDRANT_MCP.md` | Spec index (links to `docs/specs/`) |
| `docs/specs/` | 15 modular spec files |
| `docs/specs/14-future-development.md` | Graph RAG architecture decision and spec |
| `.taskmaster/docs/20260223-1648_..._PRD_memory-graph.md` | Memory Graph PRD |
| `src/rust/daemon/core/src/tree_sitter/types.rs` | SemanticChunk (graph data source) |
| `src/rust/daemon/core/src/keyword_extraction/cooccurrence_graph.rs` | Existing graph infrastructure |
