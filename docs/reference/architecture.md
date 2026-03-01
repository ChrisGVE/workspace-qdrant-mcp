# Architecture Overview

A concise reference for the workspace-qdrant-mcp system architecture.
For visual diagrams and complete component details, see [docs/ARCHITECTURE.md](../ARCHITECTURE.md).

## Component Diagram

```
  Claude / MCP Client
         |
         | STDIO (MCP Protocol)
         v
  +-------------------+
  |   MCP Server      |  TypeScript  — 6 tools: search, grep, list,
  |   (mcp-server)    |                          store, retrieve, rules
  +-------------------+
         |                  \
         | gRPC :50051        \ REST :6333 (reads only)
         v                    v
  +-------------------+    +------------------+
  |   Rust Daemon     |    |   Qdrant         |
  |   (memexd)        |--->|   Vector DB      |
  +-------------------+    +------------------+
         |
         | reads/writes
         v
  +-------------------+
  |   SQLite          |  ~/.workspace-qdrant/state.db
  |   State DB        |
  +-------------------+
         ^
         |
  +-------------------+
  |   File System     |  notify-rs watcher — inotify / FSEvents / kqueue
  +-------------------+

  +-------------------+
  |   CLI (wqm)       |  Rust — writes to SQLite queue, reads state
  +-------------------+
```

**Write rule (ADR-002):** Only the daemon writes to Qdrant. All writes from the MCP server and CLI go through the SQLite queue.

**Read rule:** MCP server and CLI may read Qdrant directly for search queries.

**Schema rule (ADR-003):** The daemon owns the SQLite database. It creates the schema on startup and handles all migrations. Other components read and write to tables but never create them.

---

## Data Flow

### File ingestion

```
File changes on disk
  → notify-rs debouncer (buffers rapid changes)
  → unified_queue table (SQLite, ACID-guaranteed)
  → queue processor dequeues item
  → Tree-sitter semantic chunking (8 built-in languages)
  → LSP enrichment (active projects: references, types, resolved imports)
  → keyword/tag extraction (8-stage pipeline)
  → FastEmbed dense vector (384-dim, all-MiniLM-L6-v2)
  → BM25 sparse vector (per-collection IDF, persisted vocabulary)
  → batch upsert to Qdrant projects collection
  → SQLite tracked_files updated
```

### Search request

```
LLM calls search() tool
  → MCP server detects project (tenant_id from git remote or path hash)
  → gRPC EmbedText + GenerateSparseVector to daemon
  → Qdrant parallel query:
      dense vector similarity (HNSW index)  +  BM25 keyword search
  → Reciprocal Rank Fusion merges ranked lists
  → results filtered by tenant_id, branch, file_type
  → returned to LLM
```

---

## Collections

Exactly four canonical collections exist in Qdrant (ADR-001). No others are created.

| Collection | Isolation key | Contents |
|------------|---------------|----------|
| `projects` | `tenant_id` (per project) | All indexed project files — code, docs, tests, configs |
| `libraries` | `library_name` | Reference documentation ingested via `store` or `wqm library` |
| `rules` | `scope` + `project_id` | LLM behavioral rules managed via `rules` tool |
| `scratchpad` | — | Temporary working notes stored via `store(type="scratchpad")` |

Multi-tenant isolation is achieved through Qdrant payload filtering, not separate collections. The `projects` collection contains all indexed projects, each distinguished by its `tenant_id` field. This allows cross-project search by removing the filter, while preventing accidental data leakage through hard tenant filtering on per-session queries.

---

## Code Intelligence Pipeline

Applied to every code file during ingestion.

```
File read
  ↓
Tree-sitter parse (always runs)
  — language detection from grammar
  — semantic chunking by symbol type:
      preamble | function | class | method | struct | trait
  — 8 built-in grammars: Rust, TypeScript, Python, Go, C, C++, Java, JavaScript
  — all other languages: text overlap chunking (384-char target, 58-char overlap)
  ↓
LSP enrichment (active projects only)
  — per-project language server: rust-analyzer, pyright, typescript-language-server,
    gopls, clangd
  — adds: symbol references (where used), type info, resolved imports
  ↓
Keyword / tag extraction (8-stage pipeline)
  — quasi-summary → lexical candidates → LSP candidates → semantic rerank
  → keyword selection (IDF penalty) → tag selection (MMR diversity)
  → basket assignment → structural tags
  ↓
Embedding generation
  — dense:  FastEmbed all-MiniLM-L6-v2 (384-dim)
  — sparse: BM25 with per-collection persisted IDF vocabulary
  ↓
Qdrant upsert (daemon only)
```

---

## Key Design Decisions

### ADR-001 — Canonical Collection Names

Four fixed collections with no underscore prefix: `projects`, `libraries`, `rules`, `scratchpad`. Multi-tenant isolation is achieved with payload field filtering (`tenant_id`, `library_name`), not per-tenant collections. This keeps Qdrant index count constant regardless of how many projects are indexed.

Reference: [docs/adr/ADR-001-canonical-collection-architecture.md](../adr/ADR-001-canonical-collection-architecture.md)

### ADR-002 — Daemon-Only Qdrant Writes

The Rust daemon (memexd) is the only component that writes to Qdrant. The MCP server and CLI enqueue write requests to the SQLite `unified_queue` table. The daemon dequeues, processes (embed, enrich), and writes.

This ensures: consistent metadata across all documents, single embedding model, centralized audit trail, crash-safe queue recovery.

Session management messages (`RegisterProject`, `DeprioritizeProject`) bypass the queue and go directly to the daemon via gRPC — these are lifecycle signals, not content writes.

Reference: [docs/adr/ADR-002-daemon-only-write-policy.md](../adr/ADR-002-daemon-only-write-policy.md)

### ADR-003 — Daemon Owns SQLite

The daemon creates the database file (`~/.workspace-qdrant/state.db`), all tables, and all schema migrations. The MCP server and CLI may read from and write to tables, but must not create tables or run migrations. If a table does not exist, components return degraded responses rather than attempting to create it.

Reference: [docs/adr/ADR-003-daemon-owns-sqlite.md](../adr/ADR-003-daemon-owns-sqlite.md)

---

## Session Lifecycle

When an MCP client (Claude) connects:

1. MCP server detects project from working directory, computes `tenant_id`
2. Server calls `GetProjectStatus` on daemon to check if project is registered
3. If registered: server calls `RegisterProject` (re-activation, `register_if_new=false`) — daemon sets `is_active=true`
4. If not registered: server logs a warning; search tools work but file watching is not active
5. Heartbeat runs every 30 seconds to prevent session timeout (60-second timeout)

When the client disconnects (`server.onclose`):

1. Server calls `DeprioritizeProject` — daemon sets `is_active=false`
2. Daemon shuts down any spawned LSP servers for the project (after queue drains)
3. Project processing priority reverts to normal

Queue priority is computed at dequeue time:
- Active projects (is_active=true): high priority, processed first
- Inactive projects: normal priority, processed in order
- Anti-starvation: fairness scheduler alternates high-priority batches (10 items) with low-priority batches (3 items)

---

## Further Reading

- [docs/ARCHITECTURE.md](../ARCHITECTURE.md) — visual Mermaid diagrams of all subsystems
- [docs/specs/](../specs/) — modular specification (authoritative)
- [docs/adr/](../adr/) — architecture decision records
- [docs/API.md](../API.md) — complete MCP tool reference
- [docs/CLI.md](../CLI.md) — wqm CLI command reference
