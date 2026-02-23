# Handover — Memory Graph Planning

## Current State

### Completed This Session

1. **grep-searcher fallback for high-frequency regex patterns** (commit `0624055a0`)
   - Added `grep-searcher`, `grep-regex`, `grep-matcher` dependencies
   - Created `src/rust/daemon/core/src/grep_search.rs` — ripgrep library-based file scanning
   - Hybrid dispatch: lightweight FTS5-only probe (`LIMIT 1 OFFSET 5000`, no JOINs) decides engine
   - Method chains pattern: 27.9ms → 15.2ms (0.60x vs rg, was 1.4x)
   - 95 tests pass (87 text_search + 8 grep_search)

2. **Redundant AND elimination in FTS5 query builder** (same commit)
   - When affix-merged prefix is present in all alternation branches, skip the standalone AND term
   - Reduces FTS5 intersection overhead for queries like `pub (fn|struct|enum|trait|type) \w+`

3. **Architecture decision: single daemon with embedded graph** (commit `5d9ca2eb2`)
   - Full analysis of single vs dual daemon for memory graph
   - Decision: embed graph in memexd using `GraphStore` trait
   - Phase 1: SQLite recursive CTEs (zero deps, 1-2 hop queries)
   - Phase 2: LadybugDB (`lbug` crate, MIT, full Cypher)
   - Updated WORKSPACE_QDRANT_MCP.md with full analysis, comparison tables, concurrency model

4. **Spec updates pushed** — grep-searcher docs + architecture decision in specs

### Task-Master Tasks

- **Task #21** (arch-refactor tag): Break down WORKSPACE_QDRANT_MCP.md into modular spec files
  - Split monolithic ~6000-line spec into `docs/specs/` directory
  - Table of contents with cross-references
  - Audit for orphan spec fragments in research/, .taskmaster/docs/, tmp/

## Next Session Instructions

### Priority 1: Execute Task #21 — Spec Breakdown

Split `WORKSPACE_QDRANT_MCP.md` into modular files under `docs/specs/`. Natural section boundaries:
- Overview & Architecture
- Point Identity & Collections
- Queue & Processing
- Search DB (FTS5) & Text Search
- Embedding & Keyword Extraction
- File Watching & Git Integration
- Graph RAG (architecture decision + schema + pipeline)
- Tagging & Classification
- Cross-Project Search & Library Management
- CLI & gRPC Services
- Configuration & Deployment
- Future Development (wishlist items remaining after graph moves to active)

### Priority 2: Plan Memory Graph Implementation

Create a comprehensive implementation plan for the memory graph, covering both Phase 1 (SQLite CTEs) and Phase 2 (LadybugDB). Chris's directions:

**Architecture (decided):**
- Single daemon — graph embedded in memexd
- `GraphStore` trait with swappable backends
- Phase 1: SQLite CTEs in `state.db` or dedicated `graph.db`
- Phase 2: LadybugDB (`lbug` crate) for deep queries

**Performance requirements:**
- Must not affect current ingestion and watching speed (or minimize impact)
- Graph edge writes during ingestion should be async/non-blocking where possible
- Existing queue processor throughput must be preserved

**Concurrency model (Chris's direction):**
- Multi-threaded graph READ queries (concurrent readers via spawn_blocking thread pool)
- Self-managed read-write coordination:
  - When a write completes, if reads are pending, reads run first
  - Reads signal "needs to run" and the next write waits for all pending reads to complete
  - When no reads are pending, writes proceed immediately
  - Think of it as a fair RwLock where readers get priority after a write, but writers aren't starved indefinitely
- Validate this approach in the plan — does it work with SQLite WAL? With LadybugDB MVCC?

**Plan should include:**
1. `GraphStore` trait definition (methods for add_node, add_edge, remove_edges, query_related, etc.)
2. SQLite CTE schema (adjacency tables, indexes, recursive CTE query patterns)
3. Integration points in the ingestion pipeline (which processing strategies add graph edges)
4. gRPC service additions for graph queries
5. CLI commands (wqm graph query, wqm graph impact, wqm graph stats)
6. Phase 2 migration path to LadybugDB (feature flag, data migration, dual-backend testing)
7. Test strategy for graph operations
8. Impact analysis on current benchmark performance

**Existing building blocks to leverage:**
- `keyword_extraction/cooccurrence_graph.rs` — symbol extraction, pair generation, centrality (deferred but infrastructure exists)
- Tree-sitter semantic chunking — already extracts function/class/method/struct/trait with `parent_symbol`
- LSP integration — provides resolved references, type info, cross-file relationships
- `tracked_files` + `qdrant_chunks` tables — file-to-chunk tracking
- Graph schema already defined in specs: File, Function, Class, Method, Struct, Trait, Module, Document nodes; CALLS, IMPORTS, EXTENDS, IMPLEMENTS, USES_TYPE, CONTAINS, MENTIONS edges

**Create the plan using task-master** — parse a PRD if needed, or add tasks directly. The plan should be comprehensive enough to execute without further clarification.

## Key Files

| File | Purpose |
|------|---------|
| `WORKSPACE_QDRANT_MCP.md` | Full project spec (to be split into docs/specs/) |
| `src/rust/daemon/core/src/grep_search.rs` | grep-searcher fallback module (new) |
| `src/rust/daemon/core/src/text_search.rs` | FTS5 search with hybrid dispatch |
| `src/rust/daemon/core/src/keyword_extraction/cooccurrence_graph.rs` | Existing graph infrastructure |
| `src/rust/daemon/core/src/context.rs` | ProcessingContext (holds all shared state) |
| `src/rust/daemon/core/src/strategies/processing/file/keyword_extract.rs` | Where graph edges would be added |
| `CLAUDE.md` | Project conventions and refactoring scope |
