## Architecture

### Two-Process Architecture

The system consists of two primary processes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACES                                 │
├─────────────────────────────────────────────────────────────────────┤
│   Claude Desktop/Code          CLI (wqm)                            │
│         │                          │                                │
│         │ MCP Protocol             │ Direct SQLite                  │
│         ▼                          ▼                                │
├─────────────────────────────────────────────────────────────────────┤
│                      TYPESCRIPT MCP SERVER                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  MCP Application (TypeScript)                             │     │
│   │  - store: Content storage to libraries collection         │     │
│   │  - search: Hybrid semantic + keyword search               │     │
│   │  - memory: Behavioral rules management                    │     │
│   │  - retrieve: Direct document access                       │     │
│   └──────────────────────────────────────────────────────────┘     │
│         │                                                           │
│         │ gRPC (port 50051)                                         │
│         ▼                                                           │
├─────────────────────────────────────────────────────────────────────┤
│                      RUST DAEMON (memexd)                           │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  - Document processing and embedding generation           │     │
│   │  - File watching with platform-native watchers            │     │
│   │  - LSP integration for code intelligence                  │     │
│   │  - Queue processing for deferred writes                   │     │
│   │  - ONLY component that writes to Qdrant                   │     │
│   └──────────────────────────────────────────────────────────┘     │
│         │                          │                                │
│         │ Vector writes            │ State reads/writes             │
│         ▼                          ▼                                │
├─────────────────────────────────────────────────────────────────────┤
│   ┌──────────────────┐    ┌──────────────────────────────────┐     │
│   │  Qdrant Vector   │    │  SQLite State DB                  │     │
│   │  Database        │    │  - unified_queue                  │     │
│   │  - projects      │    │  - watch_folders                  │     │
│   │  - libraries     │    │  - project_state                  │     │
│   │  - memory        │    │  - ingestion_status               │     │
│   └──────────────────┘    └──────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component       | Language   | Responsibilities                                                                       | Writes To           |
| --------------- | ---------- | -------------------------------------------------------------------------------------- | ------------------- |
| **MCP Server**  | TypeScript | Query processing, project detection, gRPC client, session hooks, fallback queue        | SQLite (queue only) |
| **Rust Daemon** | Rust       | Document processing, embeddings, file watching, Qdrant writes, **SQLite schema owner** | SQLite + Qdrant     |
| **CLI (wqm)**   | Rust       | Service management, library ingestion, admin operations                                | SQLite (queue only) |
| **SQLite**      | -          | State persistence, queue management, watch configuration                               | N/A (database)      |
| **Qdrant**      | -          | Vector storage, semantic search, payload filtering                                     | N/A (database)      |

### TypeScript MCP Server

**Why TypeScript:**

- Type safety for structured data (gRPC, Qdrant payloads, MCP tool schemas)
- MCP ecosystem is TypeScript-first
- Native SDK support with `@modelcontextprotocol/sdk`

**Dependencies:**

- `@modelcontextprotocol/sdk` - MCP server framework (tool registration, transports, lifecycle callbacks)
- `@qdrant/js-client-rest` - Qdrant queries
- `better-sqlite3` - SQLite queue access
- `@grpc/grpc-js` - gRPC client for daemon communication

**Session Lifecycle:**

The MCP SDK provides lifecycle callbacks via `Server`:
- `server.onclose` - Called when session ends (for cleanup)
- `server.onerror` - Called on protocol errors
- For HTTP transport: `onsessioninitialized` callback available

**Note on Claude Code Hooks:** Claude Code's `SessionStart` and `SessionEnd` hooks are **external** to MCP servers. They are shell commands configured in `~/.claude/settings.json` or `.claude/settings.json`, not SDK callbacks. If memory injection at session start is needed, it could be implemented via:
1. An external hook script that calls our MCP `memory` tool
2. Server initialization logic that runs when the transport connects

### SQLite Database Ownership

**Reference:** [ADR-003](../adr/ADR-003-daemon-owns-sqlite.md)

**The Rust daemon (memexd) is the sole owner of the SQLite database.**

| Aspect            | Owner  | Details                                         |
| ----------------- | ------ | ----------------------------------------------- |
| Database creation | Daemon | Creates `state.db` if absent (path from config) |
| Schema creation   | Daemon | Creates all tables on startup                   |
| Schema migrations | Daemon | Handles all schema version upgrades             |
| Schema versioning | Daemon | Maintains `schema_version` table                |

**Other components (MCP Server, CLI):**

- May read from any table
- May write to specific tables (e.g., `unified_queue`, `watch_folders`)
- Must NOT create tables or modify schema
- Must handle "table not found" gracefully (daemon not yet run)

**Graceful "table not found" handling:**

When MCP Server or CLI attempts to access a table before the daemon has created it:

1. **Query fails with "no such table" error** - SQLite returns this for missing tables
2. **Component catches the error** and returns a degraded response:
   - For reads: Return empty results with `status: "degraded"` indicator
   - For writes: Return error with clear message: "Daemon has not initialized database. Start the daemon first."
3. **Do not create the table** - Only daemon creates schema
4. **Log the condition** for debugging purposes

Example degraded response:

```json
{
  "results": [],
  "status": "degraded",
  "reason": "database_not_initialized",
  "message": "Daemon has not run yet. Results may be incomplete."
}
```

**Default database path:** `~/.workspace-qdrant/state.db`

---
