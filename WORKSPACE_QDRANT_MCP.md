# workspace-qdrant-mcp Specification

**Version:** 1.7.0
**Date:** 2026-02-07
**Status:** Authoritative Specification
**Supersedes:** CONSOLIDATED_PRD_V2.md, PRDv3.txt, PRDv3-snapshot1.txt

---

## Table of Contents

1. [Overview and Vision](#overview-and-vision)
2. [Architecture](#architecture)
3. [Collection Architecture](#collection-architecture)
   - [Project ID Generation](#project-id-generation)
4. [Write Path Architecture](#write-path-architecture)
5. [Memory System](#memory-system)
6. [File Watching and Ingestion](#file-watching-and-ingestion)
7. [API Reference](#api-reference)
8. [Configuration Reference](#configuration-reference)

---

## Overview and Vision

### Purpose

workspace-qdrant-mcp is a Model Context Protocol (MCP) server providing project-scoped Qdrant vector database operations with hybrid search capabilities. It enables LLM agents to:

- Store and retrieve project-specific knowledge
- Search across code, documentation, and notes using semantic similarity
- Maintain behavioral rules and preferences through persistent memory
- Index reference documentation libraries for cross-project search

### Design Philosophy

The system optimizes for:

1. **Conversational Memory**: Natural rule updates over configuration management
2. **Project Context**: Automatic workspace awareness over explicit collection selection
3. **Semantic Discovery**: Cross-content-type search over format-specific queries
4. **Behavioral Persistence**: Consistent LLM behavior over session configuration
5. **Intelligent Processing**: LSP-enhanced code understanding over text-only search

### Core Principles

See [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md) for the complete architectural philosophy. Key principles:

- **Test Driven Development**: Unit tests written immediately after code
- **Memory-Driven Behavioral Persistence**: Rules stored in memory collection
- **Project-Scoped Semantic Context**: Automatic project detection and filtering
- **Daemon-Only Writes**: Single writer to Qdrant for consistency (see [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md))
- **Four Collections Only**: Exactly `projects`, `libraries`, `memory`, `scratchpad` (see [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md))

---

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

**Reference:** [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)

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

## Collection Architecture

**Reference:** [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md)

### Canonical Collections

The system uses exactly **4 collections**:

| Collection    | Purpose                 | Multi-Tenant Key        | Example                      |
| ------------- | ----------------------- | ----------------------- | ---------------------------- |
| `projects`    | All project content     | `project_id`            | Code, docs, tests, configs   |
| `libraries`   | Reference documentation | `library_name`          | Books, papers, API docs      |
| `memory`      | Behavioral rules        | `project_id` (nullable) | LLM preferences, constraints |
| `scratchpad`  | Temporary working storage | `project_id`          | Scratch notes, intermediate results |

**Memory collection multi-tenancy:** Rules with `scope="global"` have `project_id=null` and apply to all projects. Rules with `scope="project"` have a specific `project_id` and apply only to that project.

**No other collections are permitted.** No underscore prefixes, no per-project collections, no `{basename}-{type}` patterns.

### Multi-Tenant Isolation

Projects and libraries are isolated via payload metadata filtering:

```typescript
// Project-scoped search (automatic in MCP)
search({
    collection: "projects",
    filter: { must: [{ key: "project_id", match: { value: "a1b2c3d4e5f6" } }] },
});

// Cross-project search (global scope)
search({ collection: "projects" }); // No project_id filter

// Library search
search({
    collection: "libraries",
    filter: { must: [{ key: "library_name", match: { value: "numpy" } }] },
});

// Memory search (global rules only)
search({
    collection: "memory",
    filter: { must: [{ key: "scope", match: { value: "global" } }] },
});

// Memory search (project-specific rules)
search({
    collection: "memory",
    filter: { must: [{ key: "project_id", match: { value: "a1b2c3d4e5f6" } }] },
});
```

### Project ID Generation

Project IDs (`project_id`) are 12-character hex hashes that uniquely identify a project clone. The system handles multiple clones of the same repository, filesystem moves, and local projects gaining git remotes.

#### Core Algorithm

```rust
fn calculate_project_id(project_root: &Path, disambiguation_path: Option<&str>) -> String {
    use sha2::{Sha256, Digest};

    let git_remote = get_git_remote_url(project_root);

    if let Some(remote) = git_remote {
        // Normalize git remote URL
        let normalized = normalize_git_url(&remote);
        let input = match disambiguation_path {
            Some(path) => format!("{}|{}", normalized, path),
            None => normalized,
        };
        let hash = Sha256::digest(input.as_bytes());
        return hex::encode(&hash[..6]) // 12 hex chars = 6 bytes
    }

    // Local project: use canonical path hash
    let canonical = project_root.canonicalize().unwrap();
    let input = format!("local_{}", canonical.display());
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(&hash[..6])
}
```

#### Git Remote URL Normalization

All git remote URLs are normalized for consistency:

1. Remove `.git` suffix
2. Convert to lowercase
3. Convert SSH format to HTTPS format
4. Remove protocol prefix for hashing

```rust
fn normalize_git_url(url: &str) -> String {
    // Examples:
    //   git@github.com:user/repo.git  → github.com/user/repo
    //   https://github.com/User/Repo.git → github.com/user/repo
    //   ssh://git@gitlab.com/user/repo → gitlab.com/user/repo

    let mut url = url.to_string();

    // Remove .git suffix
    if url.ends_with(".git") {
        url.truncate(url.len() - 4);
    }

    // Convert SSH to HTTPS-like format
    if url.starts_with("git@") {
        url = url.replacen("git@", "", 1).replacen(':', "/", 1);
    } else if url.starts_with("ssh://git@") {
        url = url.replacen("ssh://git@", "", 1);
    }

    // Remove protocol
    for protocol in &["https://", "http://", "ssh://"] {
        if url.starts_with(protocol) {
            url = url[protocol.len()..].to_string();
            break;
        }
    }

    // Lowercase for consistency
    url.to_lowercase()
}
```

#### Duplicate Clone Handling

When the same repository is cloned multiple times on the filesystem, each clone gets a unique `project_id` through disambiguation.

**Scenario:**

```
/Users/chris/work/client-a/myproject     (remote: github.com/user/myproject)
/Users/chris/personal/myproject          (remote: github.com/user/myproject)
```

**Algorithm:**

1. Detect duplicate when second clone is registered (same `remote_hash`)
2. Find first differing ancestor folder
3. Compute disambiguation path from project root to differing ancestor
4. Update both project_ids with disambiguation

```
Clone 1: work/client-a/myproject → project_id = sha256("github.com/user/myproject|work/client-a/myproject")[:12]
Clone 2: personal/myproject      → project_id = sha256("github.com/user/myproject|personal/myproject")[:12]
```

**Single clone:** No disambiguation needed (uses `sha256(normalized_remote)[:12]`)

**Disambiguation applied on-demand:** Only when second clone becomes active. A project is considered "actively used" when the MCP server sends a `RegisterProject` message to the daemon, indicating the user is working in that project directory.

#### Local Projects (No Git Remote)

For projects without a git remote:

- Use **container folder name** only (not full path)
- State database stores full `project_path` for move detection

```
/Users/chris/experiments/my-test-project → project_id = sha256("my-test-project")[:12]
```

#### Branch Handling

- **Branch-agnostic project_id**: All branches share the same `project_id`
- **Branch-scoped point IDs**: Each Qdrant point ID includes the branch, ensuring no collisions across branches
- **Branch stored as metadata**: `branch` field in Qdrant payload (for filtering)
- **Default search**: Returns results from all branches
- **Filtered search**: Use `branch="main"` to scope to specific branch

**Point ID formula:**

```
point_id = SHA256(tenant_id | branch | file_path | chunk_index)
```

All four components are always known at write time. This ensures each branch gets independent points with no collision risk.

**Branch lifecycle:**

- **New branch detected**: For each file, check if identical `content_hash` exists on the source branch (via `tracked_files`/`qdrant_chunks`). If yes: create new point with branch-qualified ID and **copy the vector** from the existing point (no re-embedding). If no: embed and create new point normally.
- **Branch deleted**: Delete all documents with `branch="deleted_branch"` from Qdrant (trivial — filter by branch in payload)
- **Branch renamed**: Treat as delete + create (Git doesn't track renames)

**Rationale for hybrid approach (branch-qualified IDs + vector copy):**
- Clean mental model: each branch is self-contained, no reference counting
- Trivial deletion and search scoping (filter by branch payload)
- Eliminates embedding CPU cost for identical content (vector copy is a memcpy)
- Storage is bounded: typically 3-5 active branches, not hundreds
- Content hash lookup for vector reuse is a cheap SQLite query

#### Local Project Gains Remote

When a local project (identified by container folder) gains a git remote:

1. Daemon detects `.git/config` change during watching
2. If project at same location:
   - Compute new `project_id` from normalized remote URL
   - Execute queue-mediated cascade rename (see [Cascade Rename Mechanism](#cascade-rename-mechanism))
3. No re-ingestion required

#### Remote Rename Detection

The daemon periodically checks whether a project's git remote URL has changed by comparing the current `git remote get-url origin` output with the stored `git_remote_url` in `watch_folders`. The stored value serves as the "previous" remote.

**Detection triggers:**
- During daemon polling cycle (periodic)
- On `.git/config` file change events (if watching is active)

**On detection of change:** Execute queue-mediated cascade rename (see below).

#### Cascade Rename Mechanism

When `tenant_id` must change (local project gains remote, remote URL renamed, project moved), the system uses a queue-mediated cascade to maintain integrity:

1. **SQLite transaction** (atomic, source of truth):
   - Update `watch_folders`: `tenant_id`, `git_remote_url`, `remote_hash`
   - Update `tracked_files`: all rows for this `watch_folder_id`
   - Update `qdrant_chunks`: inherits via FK relationship
   - Enqueue a single `cascade_rename` queue item:
     ```json
     {
       "item_type": "rename",
       "op": "update",
       "tenant_id": "<new_tenant_id>",
       "payload_json": {
         "old_tenant_id": "<old_tenant_id>",
         "new_tenant_id": "<new_tenant_id>",
         "collection": "projects"
       }
     }
     ```
   - Commit transaction

2. **Queue processor** picks up `cascade_rename` item:
   ```rust
   // Qdrant set_payload with filter — updates all matching points atomically
   client.set_payload(
       "projects",
       SetPayload {
           payload: hashmap!{ "project_id" => new_tenant_id.into() },
           filter: Some(Filter::must(vec![
               FieldCondition::new_match("project_id", old_tenant_id.into()),
           ])),
           ..Default::default()
       },
   ).await?;
   ```

3. **On Qdrant failure**: Queue retries with existing backoff/retry logic. SQLite is already consistent (source of truth). Qdrant is eventually consistent.

**This mechanism applies to all tenant_id changes:** local-gains-remote, remote rename, project move. It also cascades to the `memory` collection for project-scoped rules.

#### Operation Timing

| Operation                            | When Triggered                                              |
| ------------------------------------ | ----------------------------------------------------------- |
| Local project gains remote           | Background (daemon watching `.git/config`)                  |
| New branch detected                  | Background (daemon watching)                                |
| Duplicate detection & disambiguation | On-demand (when `RegisterProject` received from MCP server) |
| Project move detection               | On-demand (when `RegisterProject` received from MCP server) |

**"Active" definition:** A project becomes active when it has been explicitly registered (via CLI `wqm project add` or MCP with `register_if_new=true`) AND the MCP server sends `RegisterProject` to re-activate it. The MCP server does NOT auto-register new projects — it only re-activates existing entries in `watch_folders`. This triggers on-demand operations like duplicate detection.

#### Watch Folders Table (Unified)

All projects, libraries, and submodules are tracked in a single unified `watch_folders` table. This replaces the previously separate `registered_projects`, `project_submodules`, and `watch_folders` tables.

```sql
CREATE TABLE watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,              -- Absolute filesystem path
    collection TEXT NOT NULL,               -- "projects" or "libraries"
    tenant_id TEXT NOT NULL,                -- project_id (projects) or library name (libraries)

    -- Hierarchy (for submodules)
    parent_watch_id TEXT,                   -- NULL for top-level, references parent for submodules
    submodule_path TEXT,                    -- Relative path within parent (NULL if not submodule)

    -- Project-specific (NULL for libraries)
    git_remote_url TEXT,                    -- Normalized remote URL
    remote_hash TEXT,                       -- sha256(remote_url)[:12] for grouping duplicates
    disambiguation_path TEXT,               -- Path suffix for clone disambiguation
    is_active INTEGER DEFAULT 0,            -- Activity flag (inherited by subprojects)
    last_activity_at TEXT,                  -- Synced across parent and all subprojects

    -- Library-specific (NULL for projects)
    library_mode TEXT,                      -- "sync" or "incremental"

    -- Pause state
    is_paused INTEGER DEFAULT 0,           -- 1 = file event processing paused
    pause_start_time TEXT,                 -- ISO 8601 timestamp when pause began (NULL if not paused)

    -- Archive state
    is_archived INTEGER DEFAULT 0,         -- 1 = archived (no watching/ingesting, still searchable)

    -- Shared
    follow_symlinks INTEGER DEFAULT 0,
    enabled INTEGER DEFAULT 1,
    cleanup_on_disable INTEGER DEFAULT 0,   -- Remove content when disabled
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_scan TEXT,                         -- NULL if never scanned

    FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id)
);

-- Index for finding duplicates (same remote, different paths)
CREATE INDEX idx_watch_remote_hash ON watch_folders(remote_hash);
-- Index for active project lookups (used in queue priority calculation)
CREATE INDEX idx_watch_active ON watch_folders(is_active);
-- Index for daemon polling
CREATE INDEX idx_watch_updated ON watch_folders(updated_at);
-- Index for enabled watches
CREATE INDEX idx_watch_enabled ON watch_folders(enabled);
-- Index for subproject hierarchy
CREATE INDEX idx_watch_parent ON watch_folders(parent_watch_id);
```

**Key columns:**

- `tenant_id`: `project_id` for projects, library name for libraries
- `parent_watch_id`: Links submodules to their parent project
- `is_active`: Activity flag - **inherited by all subprojects** (see below)
- `last_activity_at`: Timestamp - **synced across parent and all subprojects**
- `is_archived`: Archive flag. Archived projects stop watching/ingesting but remain **fully searchable** in Qdrant. No search exclusion — archived projects are fair game for code exploration. User can un-archive. Archiving preserves `parent_watch_id` links (historical fact, no detaching).
- `library_mode`: Only for libraries (`sync` = full sync, `incremental` = no deletes)

**Activity Inheritance for Subprojects:**

When a project has submodules (subprojects), they share activity state with the parent. Any activity on the parent OR any subproject updates the entire group:

```sql
-- On RegisterProject or any activity (given a watch_id):
UPDATE watch_folders
SET is_active = 1, last_activity_at = datetime('now')
WHERE watch_id = :watch_id
   OR parent_watch_id = :watch_id
   OR watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id)
   OR parent_watch_id = (SELECT parent_watch_id FROM watch_folders WHERE watch_id = :watch_id);
```

This ensures:
- Activating a parent activates all its submodules
- Activating a submodule activates the parent and all sibling submodules
- All related projects share the same priority in the queue

**Deactivation:** `is_active` set to 0 (for entire group) when:

- MCP server sends `DeprioritizeProject`
- `last_activity_at` exceeds timeout (default 12h, configurable)

#### Tracked Files Table

The `tracked_files` table is the authoritative file inventory. It replaces Qdrant scrolling for file listings, recovery, and cleanup operations.

```sql
CREATE TABLE tracked_files (
    file_id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_folder_id TEXT NOT NULL,          -- FK to watch_folders.watch_id
    file_path TEXT NOT NULL,                -- RELATIVE path within project/library root
    branch TEXT,                            -- Git branch at ingestion (NULL for libraries)
    collection TEXT NOT NULL DEFAULT 'projects', -- Destination collection (format-based routing)

    -- File metadata (from filesystem at ingestion time)
    file_type TEXT,                         -- code|doc|test|config|note|artifact
    language TEXT,                          -- python, rust, javascript, etc.
    file_mtime TEXT NOT NULL,               -- Filesystem mtime at last ingestion
    file_hash TEXT NOT NULL,                -- SHA256 of file content at last ingestion

    -- Processing metadata
    chunk_count INTEGER DEFAULT 0,          -- Number of Qdrant points for this file
    chunking_method TEXT,                   -- "tree_sitter" | "overlap" | "plain"
    lsp_status TEXT DEFAULT 'none',         -- "none" | "partial" | "complete" | "failed"
    treesitter_status TEXT DEFAULT 'none',  -- "none" | "parsed" | "fallback" | "failed"
    last_error TEXT,                        -- Last processing error (NULL if success)

    -- Timestamps
    created_at TEXT NOT NULL,               -- First ingestion time
    updated_at TEXT NOT NULL,               -- Last successful processing time

    FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
    UNIQUE(watch_folder_id, file_path, branch)
);

-- Index for recovery: walk all files for a project
CREATE INDEX idx_tracked_files_watch ON tracked_files(watch_folder_id);
-- Index for finding files by path (e.g., file watcher events)
CREATE INDEX idx_tracked_files_path ON tracked_files(file_path);
-- Index for branch operations
CREATE INDEX idx_tracked_files_branch ON tracked_files(watch_folder_id, branch);
```

**Key columns:**

- `file_path`: **Relative** to `watch_folders.path`. When a project moves, only the watch_folder root path changes; tracked_files entries remain valid.
- `collection`: Destination Qdrant collection for this file. Normally matches `watch_folders.collection`, but format-based routing can override it (e.g., a `.pdf` in a project folder routes to `libraries`).
- `file_hash`: SHA256 of file content at ingestion. Used for recovery (detect changes during daemon downtime, including git checkout/rsync which change content without changing mtime) and update optimization (skip re-embedding when content hasn't changed).
- `file_mtime`: Filesystem modification time at ingestion. Used for fast change detection during recovery.
- `lsp_status` / `treesitter_status`: Track processing pipeline state per file. Enables re-processing files that had partial or failed enrichment.
- `chunk_count`: Cached count of Qdrant points for this file (also available via `qdrant_chunks` table).

#### Qdrant Chunks Table

The `qdrant_chunks` table tracks individual Qdrant points per file. It is a child of `tracked_files` with CASCADE delete.

```sql
CREATE TABLE qdrant_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,               -- FK to tracked_files.file_id
    point_id TEXT NOT NULL,                 -- Qdrant point UUID
    chunk_index INTEGER NOT NULL,           -- Position within file (0-based)
    content_hash TEXT NOT NULL,             -- SHA256 of chunk content

    -- Chunk metadata
    chunk_type TEXT,                        -- "function" | "class" | "method" | "module" | "fragment" | "plain"
    symbol_name TEXT,                       -- For tree-sitter chunks (e.g., "validate_token")
    start_line INTEGER,                     -- Source line range
    end_line INTEGER,

    -- Timestamps
    created_at TEXT NOT NULL,

    FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE,
    UNIQUE(file_id, chunk_index)
);

-- Index for looking up chunks by Qdrant point ID
CREATE INDEX idx_qdrant_chunks_point ON qdrant_chunks(point_id);
-- Index for file's chunks
CREATE INDEX idx_qdrant_chunks_file ON qdrant_chunks(file_id);
```

**Key columns:**

- `point_id`: Qdrant point UUID. Enables precise deletion without Qdrant scrolling.
- `content_hash`: SHA256 of chunk content. Enables future surgical updates: compare old vs new chunk hashes to only upsert changed chunks and delete removed ones.
- `chunk_type` / `symbol_name`: Metadata from tree-sitter parsing. Useful for debugging and statistics.
- `start_line` / `end_line`: Source line range for this chunk.

**Benefits:**

1. **Precise deletion**: Read point_ids from SQLite, delete from Qdrant by ID — no scrolling
2. **Future surgical updates**: Compare content_hashes to only modify changed chunks
3. **Debugging**: See exactly what's in Qdrant per file without querying Qdrant
4. **Statistics**: Chunks per file/language/project — instant from SQLite

### Vector Configuration

All collections use identical vector configuration:

```yaml
Vector:
  model: FastEmbed all-MiniLM-L6-v2
  dimensions: 384
  distance: Cosine

Sparse Vector:
  name: text
  modifier: idf

HNSW:
  m: 16
  ef_construct: 100
```

### Payload Schemas

**Projects Collection:**

```json
{
  "project_id": "a1b2c3d4e5f6", // Required, indexed (is_tenant=true)
  "project_name": "my-project",
  "file_path": "src/main.rs", // Relative path
  "file_type": "code", // code|doc|test|config|note|artifact
  "language": "rust",
  "branch": "main",
  "symbols": ["MyClass", "my_function"],
  "chunk_index": 0,
  "total_chunks": 5,
  "created_at": "2026-01-28T12:00:00Z"
}
```

**Note:** File metadata (`file_mtime`, `file_hash`, chunk details) is tracked in the
SQLite `tracked_files` and `qdrant_chunks` tables, not in Qdrant payloads. This keeps
Qdrant payloads lean (content-related fields only) and enables fast recovery queries
via SQLite instead of slow Qdrant scrolling. See [Tracked Files Table](#tracked-files-table).

**Libraries Collection:**

```json
{
  "library_name": "numpy", // Required, indexed (is_tenant=true)
  "source_file": "/path/to/doc.pdf",
  "file_type": "pdf",
  "title": "NumPy Documentation",
  "topics": ["arrays", "math"],
  "page_number": 42,
  "chunk_index": 0,
  "created_at": "2026-01-28T12:00:00Z"
}
```

**Memory Collection:**

```json
{
  "label": "prefer-uv", // Human-readable identifier (unique per scope)
  "content": "Use uv instead of pip for Python packages",
  "scope": "global", // global|project
  "project_id": null, // null for global, "abc123" for project-specific
  "created_at": "2026-01-30T12:00:00Z"
}
```

---

## Write Path Architecture

**Reference:** [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md)

### Core Rules

1. **Daemon-only Qdrant writes**: The Rust daemon (memexd) is the ONLY component that writes to Qdrant. No exceptions.
2. **Queue-only content writes**: ALL content writes from MCP/CLI go through SQLite queue. No direct gRPC for content.
3. **Direct reads**: MCP and CLI read from Qdrant directly (no daemon intermediary).

### Write Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         WRITE PATH                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  MCP Server                CLI                                       │
│      │                      │                                        │
│      └──────────┬───────────┘                                        │
│                 │                                                    │
│                 ▼                                                    │
│        ┌────────────────┐                                           │
│        │  SQLite Queue  │  ← ALL content writes go here             │
│        │  (unified_     │                                           │
│        │   queue)       │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │  Rust Daemon   │  ← Polls queue, processes items           │
│        │  (memexd)      │                                           │
│        └───────┬────────┘                                           │
│                │                                                    │
│                ▼                                                    │
│        ┌────────────────┐                                           │
│        │    Qdrant      │                                           │
│        └────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Read Flow

```
MCP Server ──→ Qdrant (direct, no daemon)
CLI (wqm) ───→ Qdrant (direct, no daemon)
```

### Session Management (Direct gRPC)

Only session lifecycle messages go directly to daemon via gRPC:

| Message                     | Direction    | Purpose                     |
| --------------------------- | ------------ | --------------------------- |
| `RegisterProject(path)`     | MCP → Daemon | Project is now active       |
| `DeprioritizeProject(path)` | MCP → Daemon | Project is no longer active |

**All other operations use the queue.**

### Collection Ownership

- **Daemon owns all collections**: Creates the 4 canonical collections on startup
- **No collection creation via MCP/CLI**: Only `projects`, `libraries`, `memory`, `scratchpad` exist
- **No user-created collections**: The 4-collection model is fixed

### gRPC Methods (Reserved)

The daemon exposes gRPC methods for content ingestion (`IngestText`, etc.) but these are **reserved for administrative/diagnostic use only**. Production code paths (MCP, CLI) must NOT use them directly - all content goes through the queue.

### Unified Queue

**ALL writes go through the SQLite queue.** The queue serves as the transaction log for daemon processing. This includes daemon file watcher events - the daemon queues its own events for centralized processing.

**All database operations MUST be enclosed in transactions** to ensure integrity with concurrent read/write access from MCP server, CLI, and daemon.

#### Priority System

Priority is **calculated at query time**, not stored in the queue:

| Item Type                            | Priority | Calculation                            |
| ------------------------------------ | -------- | -------------------------------------- |
| `memory` (any scope)                 | 1 (high) | Always high priority                   |
| `file`/`folder` for active project   | 1 (high) | JOIN with `watch_folders.is_active`    |
| `file`/`folder` for inactive project | 0 (low)  | JOIN with `watch_folders.is_active`    |
| `library`                            | 0 (low)  | Background processing                  |

**Anti-starvation mechanism (asymmetric batching):** The fairness scheduler alternates between priority directions with different batch sizes:

- **High-priority batch** (default 10): `ORDER BY priority DESC, created_at ASC` (active projects first)
- **Low-priority batch** (default 3): `ORDER BY priority ASC, created_at ASC` (inactive projects get a turn)

The asymmetric sizes (10:3) ensure ~77% of processing capacity goes to active projects while still preventing starvation. Equal batch sizes would neutralize priority advantages when library files are significantly larger than source code files.

#### Project Activity Tracking

The `watch_folders` table tracks activity state:

| Field              | Purpose                              |
| ------------------ | ------------------------------------ |
| `is_active`        | Boolean, true when project is active |
| `last_activity_at` | Timestamp, updated on any activity   |

**Activation:** MCP server sends `RegisterProject` → `is_active=true`, `last_activity_at=now()`

**Reactivation:** If already active, just update `last_activity_at=now()`

**Deactivation triggers:**

1. MCP server sends `DeprioritizeProject` (explicit sign-out)
2. Timeout from `last_activity_at` (default: 12 hours, configurable)

**Keep-alive:** MCP server checks at `timeout/4` interval (default: every 3 hours) if current project was wrongly deactivated by timeout, and reactivates it.

#### Queue Schema

```sql
CREATE TABLE unified_queue (
    -- Identity
    queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
    idempotency_key TEXT NOT NULL UNIQUE,  -- SHA256 hash for deduplication

    -- Item classification
    item_type TEXT NOT NULL CHECK (item_type IN (
        'content', 'file', 'folder', 'project', 'library',
        'delete_tenant', 'delete_document', 'rename'
    )),
    op TEXT NOT NULL CHECK (op IN ('ingest', 'update', 'delete', 'scan')),
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,            -- projects|libraries|memory

    -- Processing control
    priority INTEGER NOT NULL DEFAULT 5 CHECK (priority >= 0 AND priority <= 10),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'in_progress', 'done', 'failed'
    )),

    -- Timestamps
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    -- Lease-based crash recovery
    lease_until TEXT,                    -- Expiration timestamp for current lease
    worker_id TEXT,                      -- ID of worker holding lease

    -- Payload and metadata
    payload_json TEXT NOT NULL DEFAULT '{}',
    branch TEXT DEFAULT 'main',
    metadata TEXT DEFAULT '{}',

    -- Error handling
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT
);

-- Primary dequeue index (pending items by priority)
CREATE INDEX idx_unified_queue_dequeue
    ON unified_queue(status, priority DESC, created_at ASC)
    WHERE status = 'pending';

-- Idempotency enforcement (unique constraint creates implicit index)
CREATE UNIQUE INDEX idx_unified_queue_idempotency
    ON unified_queue(idempotency_key);

-- Stale lease detection for crash recovery
CREATE INDEX idx_unified_queue_lease_expiry
    ON unified_queue(lease_until)
    WHERE status = 'in_progress';

-- Tenant-based queries
CREATE INDEX idx_unified_queue_collection_tenant
    ON unified_queue(collection, tenant_id);
```

**Robust design features:**

- **status column:** Tracks item lifecycle (pending → in_progress → done/failed)
- **lease_until/worker_id:** Enables crash recovery by detecting stale leases
- **idempotency_key:** SHA256 hash of `item_type|op|tenant_id|collection|payload_json` - prevents duplicate processing even for content items without file paths
- **priority column:** Allows different priority levels for different item types (e.g., MCP content = 8, file watch = 5)
- **updated_at:** Tracks when item status last changed
- **branch:** Preserves branch context for project items

**Idempotency key calculation:**

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

**Crash recovery:** On daemon startup, scan for `status='in_progress'` with `lease_until < now()` and reset to `pending`.

#### Folder Move Detection Strategy

**Problem:** Absolute `file_path` as unique key breaks when folders move.

**Solution:** Use `notify-debouncer-full` with `FileIdMap` + periodic validation.

**notify-debouncer-full capabilities:**

- Correlates rename events via filesystem IDs
- Memory: O(n) where n = watched files (acceptable for typical projects)
- CPU: minimal (hashmap lookups)

**Platform behavior for root folder moves:**

| Platform | Event     | Watch Follows | Paths Correct |
| -------- | --------- | ------------- | ------------- |
| macOS    | RENAME    | ❌            | ⚠️            |
| Linux    | MOVE_SELF | ⚠️            | ❌ (bug #555) |
| Windows  | None      | ⚠️            | ❌            |

**Handling strategy:**

1. **Rename detection (within same filesystem):**
   - `notify-debouncer-full` correlates MOVED_FROM + MOVED_TO via cookie
   - Daemon updates queue entries and Qdrant metadata with new paths

2. **Cross-filesystem moves:**
   - Appear as unrelated delete + create
   - Treated as deletion (unavoidable limitation)

3. **Root folder move recovery:**
   - Detect via MOVE_SELF/RENAME event or path validation failure
   - Unwatch old path, watch new path
   - Update `watch_folders` table
   - Update queue entries with new paths
   - Update Qdrant metadata via bulk `set_payload`

4. **Periodic path validation (projects only):**
   - Every hour (configurable), validate watched paths exist
   - Clock resets when folder operation notification received
   - If path doesn't exist:
     - Delete tenant from Qdrant
     - Remove all queue entries for that tenant
     - Remove entry from `watch_folders`
   - Prevents orphaned data accumulation

#### Queue Error Handling

Single daemon process - no need for complex status states.

**On processing failure:**

1. Increment `retry_count`
2. Append timestamped error to `error_message` (accumulated log)
3. Update `last_error_at` with current timestamp
4. If `retry_count >= max_retries`: set `status = 'failed'`

**Error message format:** `error_message` accumulates across retries as a newline-separated log:

```
2026-02-06T12:00:00Z Qdrant connection refused: timeout after 30s
2026-02-06T12:05:00Z Qdrant connection refused: server unavailable
2026-02-06T12:15:00Z Qdrant upsert failed: collection not found
```

Update SQL:
```sql
UPDATE unified_queue SET
  retry_count = retry_count + 1,
  error_message = COALESCE(error_message || char(10), '') || strftime('%Y-%m-%dT%H:%M:%fZ', 'now') || ' ' || :error_text,
  last_error_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
  status = CASE WHEN retry_count + 1 >= max_retries THEN 'failed' ELSE status END,
  updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE queue_id = :queue_id;
```

**Failed items:**

- Stay in queue with `status = 'failed'`
- Skipped by normal processing (query: `WHERE status = 'pending'`)
- CLI displays failed items with full error history

**CLI commands for failed items:**

```bash
wqm queue list --failed          # List all failed items with error messages
wqm queue show <queue_id>        # Show single item with full error history
```

**Reset mechanism:** TBD once failure patterns are better understood. The important requirement is visibility - CLI must show what failed and why.

**Retry backoff:** Optional exponential backoff based on `retry_count`

#### Batch Processing Flow

**Daemon state:** Maintains `sort_ascending` flag (boolean, flips every batch)

```
1. Start SQL transaction
2. Read up to 10 elements:
   - WHERE failed = 0
   - JOIN watch_folders for priority calculation
   - ORDER BY priority [DESC|ASC based on sort_ascending], created_at ASC
3. Flip sort_ascending flag for next batch
4. Group items by (op, collection) for efficient Qdrant batching
5. For each group:
   a. Build Qdrant batch request
   b. Execute Qdrant batch (atomic within batch)
   c. On success: DELETE items from queue
   d. On failure: UPDATE retry_count++, append to errors, set failed if max reached
6. Create new queue entries (folder scans from processed items)
7. Commit SQL transaction
```

**Sort alternation (asymmetric):** Daemon alternates with different batch sizes per direction:

- **High-priority batch** (default 10): `ORDER BY priority DESC` (active projects first)
- **Low-priority batch** (default 3): `ORDER BY priority ASC` (inactive projects get a turn)

This prevents starvation while maintaining effective priority differentiation.

**Qdrant atomicity:** Each batch request to Qdrant is atomic. Grouping by (op, collection) leverages this for efficiency.

**Idempotency:** All operations are idempotent - retries are safe (delete non-existent = no-op, upsert = replace).

#### File Operation Transactions

All file operations follow a consistent transaction pattern. The SQL transaction opens
**before** the Qdrant operation. On success, the same transaction records the file
tracking state and marks the queue item done. On failure, the transaction only records
the retry information with accumulated error log.

##### File Ingest (new file)

```
BEGIN TRANSACTION;
  1. Read file from filesystem
  2. Compute file_hash (SHA256), read file_mtime
  3. Check ingestion gates: allowlist → exclusion → size limit (skip if rejected → mark done, COMMIT)
  4. Chunk file (tree-sitter or fallback overlap)
  5. Generate embeddings for all chunks
  6. Upsert all points to Qdrant (batch)
  7. If Qdrant succeeds:
       INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_type,
         language, file_mtime, file_hash, chunk_count, chunking_method,
         lsp_status, treesitter_status, created_at, updated_at) VALUES (...);
       INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash,
         chunk_type, symbol_name, start_line, end_line, created_at) VALUES (...);
         -- repeated for each chunk
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  8. If Qdrant fails:
       UPDATE unified_queue SET retry_count++, error_message append, last_error_at...;
       -- If retry_count >= max_retries: also SET status = 'failed'
COMMIT;
```

##### File Delete

```
BEGIN TRANSACTION;
  1. Look up file in tracked_files by (watch_folder_id, file_path, branch)
  2. Read all point_ids from qdrant_chunks for that file_id
  3. Delete points from Qdrant by point_ids (batch)
  4. If Qdrant succeeds:
       DELETE FROM qdrant_chunks WHERE file_id = ?;  -- CASCADE handles this too
       DELETE FROM tracked_files WHERE file_id = ?;
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  5. If Qdrant fails:
       Update queue with retry info + accumulated error.
  6. If file not found in tracked_files:
       Attempt Qdrant delete by filter (file_path + tenant_id) as fallback.
       Mark queue item done.
COMMIT;
```

##### File Update (delete + reingest)

```
BEGIN TRANSACTION;
  1. Look up existing file in tracked_files
  2. Read file from filesystem, compute new file_hash and file_mtime
  3. If file_hash unchanged → mark queue item done, COMMIT (skip processing)
  4. Read old point_ids from qdrant_chunks
  5. Chunk new file content, generate embeddings
  6. Delete old points from Qdrant (batch by point_ids)
  7. Upsert new points to Qdrant (batch)
  8. If both Qdrant operations succeed:
       DELETE FROM qdrant_chunks WHERE file_id = ?;
       UPDATE tracked_files SET file_mtime = ?, file_hash = ?, chunk_count = ?,
         chunking_method = ?, lsp_status = ?, treesitter_status = ?,
         last_error = NULL, updated_at = ... WHERE file_id = ?;
       INSERT INTO qdrant_chunks (...) VALUES (...);  -- for each new chunk
       UPDATE unified_queue SET status = 'done', updated_at = ... WHERE queue_id = ?;
  9. If Qdrant fails:
       Update queue with retry info + accumulated error.
COMMIT;
```

##### File Update — Surgical (future development)

An optimization enabled by `qdrant_chunks.content_hash`. Instead of full delete + reingest:

```
1. Read old qdrant_chunks with content_hashes for the file
2. Chunk new file content, compute content_hash for each chunk
3. Compare old vs new by content_hash:
   - Unchanged (same hash, same index): skip entirely
   - Modified (same index, different hash): upsert to Qdrant, update qdrant_chunks
   - New (index > old count): upsert to Qdrant, insert qdrant_chunks
   - Removed (old index > new count): delete from Qdrant, delete qdrant_chunks
4. Update tracked_files metadata, mark queue done
```

This reduces Qdrant operations when only part of a file changes (e.g., one function
edited in a large file).

#### Item Types

##### `memory` - Behavioral Rules

**Purpose:** LLM behavioral rules that persist across sessions.

**Writers:** MCP server (`memory` tool), CLI (`wqm memory add/update/remove`)

**Target collection:** `memory`

**Priority:** 1 (high) - always processed with active project priority

**Valid operations:**

| Operation | Description                  |
| --------- | ---------------------------- |
| `ingest`  | Add new rule                 |
| `update`  | Modify existing rule content |
| `delete`  | Remove rule                  |

**Queue fields:**

- `tenant_id`: `"global"` for global scope, or `<project_id>` for project scope
- `collection`: `"memory"`

**Payload structure:**

```json
{
  "label": "prefer-uv",
  "content": "Use uv instead of pip for Python packages",
  "scope": "global"
}
```

**Payload fields:**

- `label`: Human-readable identifier, unique within scope
- `content`: The actual rule text
- `scope`: `"global"` or `"project"` (mirrors tenant_id for validation)

**For `delete` operation:**

```json
{
  "label": "prefer-uv",
  "scope": "global"
}
```

**Idempotency key:** `SHA256(memory|<op>|<tenant_id>|memory|<payload_json>)[:32]`

##### `library` - Reference Documentation

**Purpose:** Reference documentation (books, papers, API docs, websites) - NOT programming libraries (use context7 MCP for those).

**Writers:**

- MCP server (`store` tool) - single file/webpage
- CLI (`wqm library ingest`) - single file/webpage to global library
- CLI (`wqm library add`) - register library folder → writes to `watch_folders` table
- Daemon - watches registered library folders, queues individual file operations

**Target collection:** `libraries`

**Priority:** 0 (low) - background processing

**Valid operations:**

| Operation | Description               | Writer                  |
| --------- | ------------------------- | ----------------------- |
| `ingest`  | Add new document/content  | MCP, CLI, Daemon        |
| `update`  | Replace existing document | Daemon (on file change) |
| `delete`  | Remove document           | Daemon (on file delete) |

**Tenant ID structure:**

| Context            | Format                       | Example                    | Use Case                             |
| ------------------ | ---------------------------- | -------------------------- | ------------------------------------ |
| Registered library | `folder.subfolder.filename`  | `rust-book.chapter1.intro` | Fine-grained search by hierarchy     |
| Project-specific   | `<project_id>.<payload_ref>` | `a1b2c3d4e5f6.design-spec` | Non-tracked files related to project |
| Global (catch-all) | `"global"`                   | `global`                   | Content without clear categorization |

**Queue fields:**

- `tenant_id`: See structure above
- `collection`: `"libraries"`

**Payload structure (MCP `store` / CLI single file):**

```json
{
  "content": "The actual text content...",
  "source": "user_input|web|file",
  "url": "https://...",
  "file_path": "/original/path.pdf"
}
```

**Note:** Title is not stored in payload - derived from tenant_id (filename/path) or extracted from content (first heading) during processing.

**Payload structure (Daemon from watched folder):**

```json
{
  "file_path": "/path/to/library/folder/chapter1/intro.md",
  "library_name": "rust-book",
  "relative_path": "chapter1/intro.md"
}
```

**Notes:**

- MCP/CLI provide content directly in payload (for single items)
- Daemon reads file content during processing (for watched folders)
- Dot-delimited tenant_id enables hierarchical search: `tenant_id LIKE 'rust-book.chapter1.%'`

**Idempotency key:** `SHA256(library|<op>|<tenant_id>|libraries|<payload_json>)[:32]`

##### `file` - Project/Library Source Files

**Purpose:** Individual files from watched folders (projects or libraries)

**Writers:** Daemon only (from file watcher or folder scan)

**Target collection:** `projects` or `libraries` (depending on watch type)

**Priority:** Calculated from `watch_folders.is_active` (for projects), always 0 for libraries

**Valid operations:**

| Operation | Description             | Trigger                  |
| --------- | ----------------------- | ------------------------ |
| `ingest`  | Add/update file content | File created or modified |
| `delete`  | Remove file from index  | File deleted             |

**Queue fields:**

- `tenant_id`: `<project_id>` or library tenant format
- `collection`: `"projects"` or `"libraries"`

**Payload structure:**

```json
{
  "file_path": "/absolute/path/to/file.rs",
  "relative_path": "src/main.rs"
}
```

**Notes:**

- Full path is unique in queue (first debounce level)
- Daemon computes metadata during processing (branch, file_type, language, symbols via LSP/tree-sitter)
- At processing time, daemon adapts to current state:
  - File doesn't exist but in collection → remove from collection, pop queue
  - File doesn't exist and not in collection → just pop queue
  - File exists → ingest/update as normal

##### `folder` - Directory Scan

**Purpose:** Trigger recursive scanning of a folder's contents

**Writers:** Daemon only (from folder creation event or initial registration)

**Target collection:** N/A (expands into `file` and `folder` entries)

**Priority:** Same as parent project/library

**Valid operations:**

| Operation | Description                         | Trigger                                |
| --------- | ----------------------------------- | -------------------------------------- |
| `scan`    | List folder contents and queue them | Folder created or initial registration |

**Queue fields:**

- `tenant_id`: `<project_id>` or library tenant format
- `collection`: `"projects"` or `"libraries"`

**Payload structure:**

```json
{
  "folder_path": "/absolute/path/to/folder",
  "relative_path": "src/utils"
}
```

**Processing behavior:**

- List all eligible files → queue as `file` with `op=ingest`
- List all subfolders → queue as `folder` with `op=scan`
- Transaction encompasses all additions + pop of scan entry
- Full path uniqueness prevents duplicate queueing

---

### Daemon Processing Phases

#### Phase 1: Initial Registration (Folder Scan)

When a new project or library folder is registered:

```
1. Root folder queued as `folder` with `op=scan`
2. Daemon processes scan:
   - List eligible files → queue each as `file/ingest`
   - List subfolders → queue each as `folder/scan`
   - Transaction: all additions + pop scan entry
3. Daemon processes files (per File Ingest transaction):
   - Ingest with LSP/tree-sitter metadata
   - Record in tracked_files + qdrant_chunks (within same transaction)
   - Pop queue entry on success
   - If file no longer exists: just pop (no error)
   - If file modified before ingestion: ingested state is final
4. Daemon processes subfolders:
   - Repeat step 2 for each subfolder
5. Continues until queue empty (no more folders or files)
```

**Atomic unit:** One folder level (all its direct contents queued in one transaction)

#### Phase 2: Ongoing Watching (File Changes)

Once initial scan complete, daemon watches for changes:

| Event          | Action                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| File created   | Queue `file/ingest` (uniqueness prevents duplicates)                   |
| File modified  | Queue `file/update` (atomic: delete existing points + reingest)        |
| File deleted   | Queue `file/delete`                                                    |
| Folder created | Queue `folder/scan`                                                    |
| Folder deleted | Remove all files in collection with path prefix                        |

**Update atomicity:** A `file/update` is processed as a single atomic operation:
delete all existing points for the file path, then reingest current file content.
This ensures no window of inconsistency where a file has partial data.

**Queue deduplication:** The `file_path UNIQUE` constraint ensures only one operation
per file can be queued at a time. If a file already has a pending `ingest` and gets
modified, the `update` is silently dropped — the pending ingest will read the file's
current content at processing time, achieving the same result.

**Debouncing:**

1. **Queue uniqueness:** Full path can't be queued twice (first level)
2. **External debounce:** Configurable delay before queueing (second level)

**Incremental libraries:** Ignore delete events (additions and updates only)

**Processing adaptation:** At ingestion time, daemon checks current state:

- File gone but in collection → remove, pop
- File gone and not in collection → just pop
- File exists → ingest normally

#### Phase 3: Removal (Project/Library Deletion)

When a project or library is removed:

```
DELETE FROM qdrant WHERE project_id = '<id>'
-- or --
DELETE FROM qdrant WHERE tenant_id LIKE '<library_prefix>%'
```

Blunt force removal of all content. No queue entries needed - direct Qdrant operation.

#### Phase 4: Daemon Startup Automation

On daemon start (or restart), the daemon runs a 6-step startup sequence before entering
its normal event loop. This handles schema migrations, configuration changes, state
reconciliation, and crash recovery.

```
Step 1: Schema Integrity Check
   - Verify all 5 tables exist (schema_version, unified_queue, watch_folders,
     tracked_files, qdrant_chunks)
   - Run pending migrations if schema_version is behind
   - ABORT startup if schema check fails (cannot proceed without tables)

Step 2: Configuration Change Reconciliation
   - Compute config fingerprint = SHA256(sorted(allowed_extensions) + sorted(allowed_filenames)
     + sorted(exclude_directories) + sorted(exclude_patterns))
   - Compare fingerprint with stored value in schema_version metadata
   - If fingerprint differs (config changed since last run):
     a. For files in tracked_files that are NOW excluded (new exclusion rule):
        → queue file/delete for each
     b. For files in tracked_files whose extension is NO LONGER on allowlist:
        → queue file/delete for each
     c. Store new fingerprint
   - NOTE: Newly-allowed files are discovered during Step 5 filesystem walk

Step 3: Qdrant Collection Verification
   - Ensure 4 canonical collections exist: projects, libraries, memory, scratchpad
   - Create any missing collections with correct vector configuration
   - Verify named vector configuration (dense + sparse) matches expectations

Step 4: Watch Folder Path Validation
   - For each watch_folder entry WHERE enabled = 1:
     a. Validate path exists on filesystem
     b. If path invalid: set enabled = 0, deactivate, log warning
   - This catches projects that were moved or deleted while daemon was stopped

Step 5: Filesystem Recovery (tracked_files reconciliation)
   - For each watch_folder WHERE enabled = 1:
     a. Query tracked_files for all files with this watch_folder_id
     b. Walk filesystem to get current eligible files with mtime + hash
        (eligible = passes allowlist + exclusion + size gates)
     c. Compare (file_path is relative to watch_folder.path):
        - In tracked_files but not on disk → queue file/delete
        - On disk but not in tracked_files → queue file/ingest
        - In both but file_mtime or file_hash changed → queue file/update
        - In both and unchanged → skip (no action)
   - For first startup with empty tracked_files, the initial scan (Phase 1)
     handles population

Step 6: Crash Recovery
   - Reset stale in_progress queue items:
     WHERE status = 'in_progress' AND lease_expires_at < now()
     SET status = 'pending', leased_by = NULL, lease_expires_at = NULL,
         retry_count = retry_count + 1
   - Items exceeding max_retries are set to status = 'failed'
```

**Performance:** Steps 1-4 query SQLite only (milliseconds). Step 5 performs filesystem
walks but compares against SQLite (fast). Step 6 is a single SQL UPDATE.

**Scan distinction:** Initial scan (Phase 1) is for newly registered projects only.
Startup automation (Phase 4) runs on every daemon startup for all existing watched projects.
The file watcher (Phase 2) handles all changes during normal daemon operation.

### Daemon Watch Management

The daemon manages filesystem watches based on the `watch_folders` table.

#### Startup

```
1. Read all entries from watch_folders WHERE enabled = 1
2. For each folder:
   a. Validate path exists
   b. Set up recursive filesystem watch (notify crate)
   c. If folder not yet scanned (last_scan IS NULL):
      Queue folder for initial scan (folder/scan)
3. Start queue processor loop
4. Start watch_folders polling loop
```

#### Runtime: New Folder Registered

The daemon polls `watch_folders` table periodically (default: every 5 seconds) for changes.

```
1. Detect new entry or updated_at changed
2. If enabled = 1 and not already watching:
   a. Set up recursive filesystem watch
   b. Queue folder for initial scan
3. If enabled = 0 and currently watching:
   a. Remove filesystem watch
   b. Optionally trigger Phase 3 cleanup (if configured)
```

#### Runtime: Folder Unregistered

When a watch entry is deleted or disabled:

```
1. Remove filesystem watch for that path
2. If cleanup_on_disable configured:
   a. Delete all content from Qdrant for that tenant
   b. Update watch_folders.last_scan = NULL
```

#### Runtime: Pause/Resume

Global pause halts file event processing for maintenance or backups. CLI writes directly to SQLite; daemon detects changes via polling.

**Pause (`wqm watch pause`):**
```
1. CLI sets is_paused = 1 and pause_start_time on all enabled watches
2. Daemon polls DB every 5 seconds, detects change
3. In-memory AtomicBool pause flag set to true
4. FileWatcher buffers incoming events (up to 10K capacity, FIFO eviction)
5. Queue processor skips paused items
```

**Resume (`wqm watch resume`):**
```
1. CLI sets is_paused = 0 and clears pause_start_time on all paused watches
2. Daemon polls DB, detects change
3. In-memory pause flag set to false
4. FileWatcher drains buffered events into normal processing
5. Queue processor resumes normal operation
```

**gRPC alternative:** `PauseAllWatchers` / `ResumeAllWatchers` RPCs update DB and flag atomically.

**Persistence:** Pause state survives daemon restarts. On startup, daemon calls `poll_pause_state()` to restore the flag from DB.

**Diagnostic entries:** Each pause/resume writes a metadata entry to `unified_queue` for audit.

#### Watch Folders Table Reference

See [Watch Folders Table (Unified)](#watch-folders-table-unified) in the Collection Architecture section for the complete schema.

**Daemon polling query:**

```sql
SELECT * FROM watch_folders
WHERE updated_at > :last_poll_time OR enabled != :cached_enabled_state
```

**Item Types (MCP-relevant):**

| item_type | Used By             | payload_json                                  |
| --------- | ------------------- | --------------------------------------------- |
| `memory`  | MCP `memory` tool   | `{label, content, scope, project_id}`         |
| `library` | MCP `store` tool    | `{library_name, content, title, source, url}` |
| `project` | MCP `store` tool    | `{path, name}` (registers via gRPC, not queue) |
| `file`    | Daemon file watcher | `{file_path, ...}`                            |
| `folder`  | Daemon folder scan  | `{folder_path, patterns, ...}`                |

### Idempotency

All queue operations use SHA256-based idempotency keys:

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

### Queue Response

When operations are queued:

```json
{
  "success": true,
  "status": "queued",
  "message": "Operation queued for daemon processing.",
  "queue_id": "abc123"
}
```

---

## Memory System

### Purpose

The memory collection stores LLM behavioral rules that persist across sessions. Rules are injected into Claude's context at session start.

### Rule Schema

```json
{
  "label": "prefer-uv", // Human-readable identifier
  "content": "Use uv instead of pip for Python packages",
  "scope": "global", // global | project
  "project_id": null, // null for global, project_id for project-specific
  "created_at": "2026-01-30T12:00:00Z"
}
```

**Uniqueness constraint:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

### Rule Scope

| Scope     | Application           | project_id    |
| --------- | --------------------- | ------------- |
| `global`  | All projects          | `null`        |
| `project` | Specific project only | `"abc123..."` |

### Context Injection

At session start:

1. MCP server queries `memory` collection
2. Filters: all global rules + current project's rules
3. Orders: global rules first (by creation date), then project rules (by creation date)
4. Formatted and injected into system context

### Rule Management

**Via CLI:**

```bash
wqm memory list                      # List all rules (global + all projects)
wqm memory list --global             # List global rules only
wqm memory list --project <path>     # List rules for specific project
wqm memory add --label "prefer-uv" --content "Use uv instead of pip" --global
wqm memory add --label "use-pytest" --content "Use pytest for testing" --project .
wqm memory remove --label "prefer-uv" --global
```

**Via MCP:**

```typescript
memory({ action: "list" });                // List global + current project rules
memory({ action: "add", label: "...", content: "...", scope: "project" });
memory({ action: "remove", label: "...", scope: "global" });
```

### Conversational Updates

Rules can be added conversationally:

```
User: "For future reference, always use uv instead of pip"
→ Creates memory rule: {label: "prefer-uv", content: "Use uv for Python packages", scope: "global"}
```

---

## File Watching and Ingestion

### Watch Sources

| Source  | Target Collection | Trigger                                                          |
| ------- | ----------------- | ---------------------------------------------------------------- |
| **MCP** | `projects`        | `RegisterProject` → re-activates existing watch entry only       |
| **CLI** | `projects`        | `wqm project add` → explicit new project registration            |
| **CLI** | `libraries`       | `wqm library add` → explicit library registration                |

**Note:** The MCP server does NOT auto-register new projects. It only re-activates projects that were previously registered (via CLI or a prior MCP session with `register_if_new=true`). If the project is not found in `watch_folders`, MCP logs a warning and continues without registration. CLI is the primary path for registering new projects and libraries.

### Watch Table Schema

See [Watch Folders Table (Unified)](#watch-folders-table-unified) in the Collection Architecture section for the complete schema.

**Library modes (libraries only):**

- `sync`: Full synchronization - additions, updates, AND deletions
- `incremental`: Additions and updates only, no deletions

**Always recursive:** No depth limit configuration needed.

### Ingestion Filtering

Ingestion filtering is defined system-wide in the configuration file (not per-watch). The system uses a multi-layered approach with the **file type allowlist** as the primary gate.

See [File Type Allowlist](#file-type-allowlist) below for the complete specification including:
- Ingestion gate layering (allowlist → exclusions → size limits)
- Allowed extensions by category (400+ extensions across 21 categories)
- Allowed extension-less filenames (30+ exact names)
- Size-restricted extensions (with configurable stricter limits)
- Mandatory excluded directories (50+ build/cache directories)

### File Type Allowlist

The allowlist is the **primary ingestion gate** — files not on the allowlist are silently skipped (never queued, never tracked). This prevents the system from ingesting binary files, media, build artifacts, and other non-textual content.

#### Design

- **Two-tier allowlist**: Project extensions (source code, configs, text) and library extensions (superset: project + reference formats)
- **Library allowlist** = project_extensions UNION library_only_extensions (`.pdf`, `.epub`, `.djvu`, `.docx`, `.mobi`, `.odt`, `.rtf`, `.doc`, `.ppt`, `.pptx`, `.xls`, `.xlsx`, `.csv`, `.tsv`, `.parquet`)
- **Format-based routing**: Files with library-only extensions found in project folders are routed to the `libraries` collection with `source_project_id` metadata (see [Project vs. Library Boundary](#project-vs-library-boundary))
- Defined in YAML config under `watching.allowed_extensions` and `watching.allowed_filenames`
- Compile-time embedded defaults in all three components (daemon, CLI, MCP server)
- User config can override (extend or restrict) the defaults

#### Ingestion Gate Layering

Every file event passes through four gates in order:

```
1. ALLOWLIST:        Extension or exact filename must be on the list     → NO  = skip
2. EXCLUSION:        Must not match directory or pattern exclusions      → YES = skip
3. SIZE LIMIT:       Must be under max_file_size_mb                     → OVER = skip
   3a. SIZE-RESTRICTED: If extension is size-restricted, apply stricter limit → OVER = skip
4. Queue for processing
```

The allowlist supersedes the old `media_files` exclusion for `.pdf` — PDF is explicitly on the allowlist. The exclusion rules remain as defense-in-depth for directories and patterns.

#### Allowed Extensions by Category

**1. Systems Languages**

| Extension | Language |
|-----------|----------|
| `.c`, `.h` | C |
| `.cpp`, `.cxx`, `.cc`, `.c++`, `.hpp`, `.hxx`, `.hh`, `.h++`, `.ipp`, `.tpp` | C++ |
| `.rs` | Rust |
| `.go` | Go |
| `.zig` | Zig |
| `.nim`, `.nims`, `.nimble` | Nim |
| `.d`, `.di` | D |
| `.v` | V |
| `.odin` | Odin |
| `.s`, `.S`, `.asm` | Assembly |

**2. JVM Languages**

| Extension | Language |
|-----------|----------|
| `.java` | Java |
| `.kt`, `.kts` | Kotlin |
| `.scala`, `.sc`, `.sbt` | Scala |
| `.clj`, `.cljs`, `.cljc`, `.edn` | Clojure |
| `.groovy`, `.gvy`, `.gy`, `.gsh` | Groovy |

**3. .NET Languages**

| Extension | Language |
|-----------|----------|
| `.cs`, `.csx` | C# |
| `.fs`, `.fsi`, `.fsx`, `.fsscript` | F# |
| `.vb` | VB.NET |
| `.csproj`, `.fsproj`, `.vbproj`, `.sln`, `.props`, `.targets` | Project files |
| `.xaml` | XAML |
| `.razor`, `.cshtml` | Razor |
| `.nuspec` | NuGet spec |

**4. Scripting Languages**

| Extension | Language |
|-----------|----------|
| `.py`, `.pyi`, `.pyw`, `.pyx`, `.pxd` | Python |
| `.rb`, `.rbw`, `.rake`, `.gemspec` | Ruby |
| `.pl`, `.pm`, `.pod`, `.t`, `.psgi` | Perl |
| `.lua` | Lua |
| `.php`, `.phtml`, `.php3`, `.php4`, `.php5`, `.php7`, `.phps` | PHP |
| `.tcl`, `.tk` | Tcl/Tk |
| `.r`, `.R`, `.Rmd`, `.Rnw` | R |
| `.dart` | Dart |
| `.raku`, `.rakumod`, `.rakutest`, `.p6`, `.pm6` | Raku |

**5. Functional Languages**

| Extension | Language |
|-----------|----------|
| `.hs`, `.lhs` | Haskell |
| `.ml`, `.mli`, `.mll`, `.mly` | OCaml |
| `.erl`, `.hrl` | Erlang |
| `.ex`, `.exs` | Elixir |
| `.lsp`, `.lisp`, `.cl`, `.fasl` | Common Lisp |
| `.scm`, `.ss` | Scheme |
| `.rkt` | Racket |
| `.elm` | Elm |
| `.purs` | PureScript |
| `.nix` | Nix |
| `.lean`, `.olean` | Lean |
| `.agda` | Agda |
| `.idr`, `.ipkg` | Idris |
| `.sml`, `.sig`, `.fun` | Standard ML |

**6. Web Technologies**

| Extension | Language |
|-----------|----------|
| `.js`, `.mjs`, `.cjs`, `.jsx` | JavaScript |
| `.ts`, `.mts`, `.cts`, `.tsx` | TypeScript |
| `.html`, `.htm`, `.xhtml` | HTML |
| `.css` | CSS |
| `.scss`, `.sass` | SCSS/Sass |
| `.less` | Less |
| `.styl`, `.stylus` | Stylus |
| `.vue` | Vue |
| `.svelte` | Svelte |
| `.astro` | Astro |
| `.mdx` | MDX |
| `.coffee`, `.litcoffee` | CoffeeScript |
| `.wasm`, `.wat` | WebAssembly |

**7. Shell and Scripting**

| Extension | Language |
|-----------|----------|
| `.sh`, `.bash`, `.zsh`, `.fish` | Shell |
| `.ps1`, `.psm1`, `.psd1` | PowerShell |
| `.bat`, `.cmd` | Batch |
| `.mk` | Make |
| `.awk` | AWK |
| `.sed` | sed |

**8. Legacy Languages**

| Extension | Language |
|-----------|----------|
| `.cob`, `.cbl`, `.cpy` | COBOL |
| `.f`, `.f90`, `.f95`, `.f03`, `.f08`, `.for`, `.fpp` | Fortran |
| `.pas`, `.pp`, `.dpr`, `.dpk`, `.dfm`, `.lfm` | Pascal/Delphi |
| `.adb`, `.ads` | Ada |
| `.bas`, `.vbs`, `.vba`, `.cls`, `.frm` | BASIC/VBA |
| `.rpg`, `.rpgle`, `.sqlrpgle` | RPG |
| `.abap` | ABAP |

**9. Apple Ecosystem**

| Extension | Language |
|-----------|----------|
| `.swift` | Swift |
| `.m`, `.mm` | Objective-C |
| `.applescript`, `.scpt` | AppleScript |
| `.plist` | Property list |
| `.pbxproj` | Xcode project |
| `.storyboard`, `.xib` | Interface Builder |
| `.entitlements` | Entitlements |
| `.xcconfig` | Xcode config |
| `.xcscheme` | Xcode scheme |
| `.metal` | Metal shader |
| `.strings`, `.stringsdict` | Localization |
| `.xcdatamodeld` | Core Data model |

**10. Data Science and Scientific Computing**

| Extension | Language |
|-----------|----------|
| `.jl` | Julia |
| `.m` | MATLAB |
| `.wl`, `.wls`, `.nb` | Mathematica/Wolfram |
| `.mpl`, `.mw` | Maple |
| `.oct` | Octave |
| `.ipynb` | Jupyter |
| `.qmd` | Quarto |
| `.sas` | SAS |
| `.do`, `.ado` | Stata |
| `.stan` | Stan |
| `.sage`, `.spyx` | SageMath |

**11. DevOps and Configuration**

| Extension | Language |
|-----------|----------|
| `.yaml`, `.yml` | YAML |
| `.toml` | TOML |
| `.json`, `.jsonc`, `.json5` | JSON |
| `.xml` | XML |
| `.ini`, `.cfg` | INI |
| `.tf`, `.tfvars` | Terraform |
| `.hcl` | HCL |
| `.env.example`, `.env.template` | Env templates |
| `.properties` | Java properties |
| `.conf` | Config |
| `.desktop` | Desktop entry |
| `.service`, `.timer`, `.socket` | systemd |

**12. Build Systems**

| Extension | Language |
|-----------|----------|
| `.cmake` | CMake |
| `.bzl`, `.bazel` | Bazel |
| `.gradle` | Gradle |
| `.ninja` | Ninja |
| `.meson` | Meson |
| `.gn`, `.gni` | GN |
| `.spec` | RPM spec |

**13. Documents and Text**

| Extension | Language |
|-----------|----------|
| `.md`, `.markdown` | Markdown |
| `.rst` | reStructuredText |
| `.adoc`, `.asciidoc` | AsciiDoc |
| `.org` | Org-mode |
| `.tex`, `.latex`, `.sty`, `.cls`, `.bib`, `.bst` | LaTeX |
| `.pdf` | PDF |
| `.epub` | EPUB |
| `.docx` | Word |
| `.odt` | OpenDocument |
| `.rtf` | Rich Text |
| `.csv`, `.tsv`, `.tab` | Delimited data |
| `.svg` | SVG |
| `.txt`, `.text` | Plain text |
| `.man`, `.1`, `.2`, `.3`, `.4`, `.5`, `.6`, `.7`, `.8`, `.9` | Man pages |
| `.diff`, `.patch` | Patches |

**14. Templates**

| Extension | Language |
|-----------|----------|
| `.j2`, `.jinja`, `.jinja2` | Jinja2 |
| `.hbs`, `.handlebars` | Handlebars |
| `.mustache` | Mustache |
| `.ejs` | EJS |
| `.pug`, `.jade` | Pug |
| `.slim` | Slim |
| `.haml` | Haml |
| `.liquid` | Liquid |
| `.twig` | Twig |
| `.blade.php` | Blade |
| `.erb` | ERB |
| `.njk`, `.nunjucks` | Nunjucks |

**15. Protocol and Schema Definitions**

| Extension | Language |
|-----------|----------|
| `.proto` | Protocol Buffers |
| `.graphql`, `.gql` | GraphQL |
| `.thrift` | Thrift |
| `.fbs` | FlatBuffers |
| `.capnp` | Cap'n Proto |
| `.xsd`, `.xsl`, `.xslt` | XML Schema/XSLT |
| `.cue` | CUE |
| `.avsc`, `.avdl` | Avro |

**16. Database**

| Extension | Language |
|-----------|----------|
| `.sql` | SQL |
| `.plsql`, `.pls`, `.plb` | PL/SQL |
| `.tsql` | T-SQL |
| `.pgsql` | PostgreSQL |
| `.mysql` | MySQL |
| `.prisma` | Prisma |
| `.edgeql`, `.esdl` | EdgeDB |

**17. Graph Languages**

| Extension | Language |
|-----------|----------|
| `.cypher`, `.cql` | Cypher (Neo4j) |
| `.rq`, `.sparql` | SPARQL |
| `.dot`, `.gv` | Graphviz |

**18. Shaders**

| Extension | Language |
|-----------|----------|
| `.glsl`, `.vert`, `.frag`, `.geom`, `.tesc`, `.tese`, `.comp` | GLSL |
| `.hlsl`, `.fx`, `.fxh` | HLSL |
| `.metal` | Metal |
| `.wgsl` | WGSL |
| `.cg`, `.cgfx` | Cg |
| `.shader`, `.compute` | Unity |

**19. Hardware and Embedded**

| Extension | Language |
|-----------|----------|
| `.v`, `.sv`, `.svh`, `.vh` | Verilog/SystemVerilog |
| `.vhd`, `.vhdl` | VHDL |
| `.ino` | Arduino |
| `.dts`, `.dtsi` | Device Tree |
| `.ld`, `.lds` | Linker scripts |

**20. Blockchain**

| Extension | Language |
|-----------|----------|
| `.sol` | Solidity |
| `.vy` | Vyper |
| `.cairo` | Cairo |
| `.move` | Move |
| `.clar` | Clarity |

**21. Diagram and Visual**

| Extension | Language |
|-----------|----------|
| `.dot`, `.gv` | Graphviz |
| `.mmd` | Mermaid |
| `.puml`, `.plantuml` | PlantUML |
| `.d2` | D2 |

#### Allowed Extension-less Filenames

These files are recognized by exact name (case-sensitive):

| Filename | Category |
|----------|----------|
| `Makefile`, `GNUmakefile` | Build |
| `Dockerfile`, `Containerfile` | Container |
| `Jenkinsfile` | CI/CD |
| `Vagrantfile` | Virtualization |
| `Rakefile` | Ruby build |
| `Gemfile` | Ruby deps |
| `Podfile` | CocoaPods |
| `Fastfile`, `Appfile`, `Matchfile`, `Snapfile` | Fastlane |
| `Brewfile` | Homebrew |
| `Procfile` | Process |
| `Justfile`, `justfile` | Just |
| `BUILD`, `WORKSPACE` | Bazel |
| `CODEOWNERS` | GitHub |
| `LICENSE`, `LICENCE`, `COPYING` | Legal |
| `README`, `CHANGELOG`, `AUTHORS`, `CONTRIBUTORS` | Project docs |
| `CMakeLists.txt` | CMake |
| `.gitignore`, `.gitattributes`, `.gitmodules` | Git config |
| `.dockerignore` | Docker |
| `.editorconfig` | Editor config |
| `.clang-format`, `.clang-tidy` | C/C++ tools |
| `.eslintrc`, `.prettierrc`, `.stylelintrc` | JS/TS tools |
| `.rubocop.yml` | Ruby tools |
| `.flake8`, `.pylintrc`, `.mypy.ini` | Python tools |
| `.rustfmt.toml`, `.clippy.toml` | Rust tools |
| `.swiftlint.yml` | Swift tools |

#### Size-Restricted Extensions

Some extensions can contain either small config/schema files or massive datasets. These extensions are on the allowlist but subject to a **stricter size limit** (configurable, default 1 MB instead of the general `max_file_size_mb`):

| Extension | Risk | Rationale |
|-----------|------|-----------|
| `.csv`, `.tsv`, `.tab` | Dataset dumps | Can be multi-GB data exports |
| `.json`, `.jsonc`, `.json5` | Data dumps | Can be large API responses, datasets |
| `.xml`, `.xsd`, `.xsl` | Data dumps | Can be large data exports, SOAP payloads |
| `.jsonl`, `.ndjson` | Streaming data | Can be unbounded log/event streams |
| `.log` | Log files | Can grow unbounded |
| `.sql` | Database dumps | Can contain full database exports |

Config keys: `watching.size_restricted_extensions` with `watching.size_restricted_max_mb` (default: 1 MB).
Files with these extensions exceeding the restricted size limit are skipped with an INFO log.

#### Mandatory Excluded Directories

These directories are always excluded regardless of configuration. They are checked by path component (any directory segment matching is excluded):

| Directory | Reason |
|-----------|--------|
| `node_modules` | NPM packages |
| `target` | Rust/Maven build output |
| `build` | General build output |
| `dist` | Distribution output |
| `out` | Compiler output |
| `.git` | Git internals |
| `__pycache__` | Python bytecode |
| `.venv`, `venv`, `.env` | Python virtual environments |
| `.tox` | Python tox |
| `.mypy_cache` | Mypy cache |
| `.pytest_cache` | Pytest cache |
| `.ruff_cache` | Ruff cache |
| `.gradle` | Gradle cache |
| `.next` | Next.js build |
| `.nuxt` | Nuxt.js build |
| `.svelte-kit` | SvelteKit build |
| `.astro` | Astro build |
| `Pods` | CocoaPods |
| `DerivedData` | Xcode build |
| `.build` | Swift PM build |
| `.swiftpm` | Swift PM cache |
| `.fastembed_cache` | FastEmbed model cache |
| `.terraform` | Terraform state |
| `.terragrunt-cache` | Terragrunt cache |
| `coverage` | Test coverage |
| `.nyc_output` | NYC coverage |
| `.cargo` | Cargo cache |
| `.rustup` | Rustup toolchains |
| `vendor` | Vendored deps |
| `.bundle` | Ruby bundle |
| `.cache` | General cache |
| `.tmp`, `tmp` | Temporary files |
| `.DS_Store` | macOS metadata |
| `.idea`, `.vscode` | IDE settings |
| `.settings`, `.project`, `.classpath` | Eclipse |
| `bin`, `obj` | .NET build output |
| `.zig-cache` | Zig cache |
| `zig-out` | Zig output |
| `elm-stuff` | Elm packages |
| `.stack-work` | Haskell Stack |
| `_build` | Elixir/Phoenix build |
| `deps` | Elixir deps |
| `.dart_tool` | Dart tool cache |
| `.pub-cache` | Pub cache |

### Git Submodules

When a subfolder contains a `.git` directory (submodule):

1. **Detect:** Daemon detects `.git` in subfolder during scanning
2. **Separate entry:** Submodule is registered in `watch_folders` with its own `tenant_id` (project_id)
3. **Link via parent_watch_id:** Submodule entry references parent project via `parent_watch_id` and stores relative path in `submodule_path`
4. **Activity inheritance:** Submodules share `is_active` and `last_activity_at` with parent (see [Activity Inheritance](#watch-folders-table-unified))

See [Watch Folders Table (Unified)](#watch-folders-table-unified) for the schema that handles both projects and submodules.

**Submodule archive safety:**

When archiving a project that has submodules, the system must check cross-references before archiving submodule data:

1. Set `is_archived = 1` on the parent project's `watch_folders` entry
2. For each submodule entry (linked via `parent_watch_id`):
   a. Check if the same `remote_hash`/`git_remote_url` exists in any other **active** (non-archived) `watch_folders` entry
   b. If yes: the submodule data stays fully active — another project still references it
   c. If no: set `is_archived = 1` on the submodule entry (data remains searchable, watching stops)
3. **Preserve `parent_watch_id` as-is** — it is historical fact. No detaching on archive.
4. Qdrant data is **never deleted** on archive. Archived content remains fully searchable.

**Un-archiving:** Set `is_archived = 0` on the project and its submodule entries. Daemon resumes watching/ingesting.

### Daemon Polling

The daemon:

1. Polls `watch_folders` table every 5 seconds
2. Detects changes via `updated_at` timestamp
3. Updates file watchers dynamically
4. Processes file events through ingestion queue

### Ingestion Pipeline

Different operations have different pipelines:

| Event            | Pipeline                                                  |
| ---------------- | --------------------------------------------------------- |
| **New file**     | Debounce → Read → Parse/Chunk → Embed → Upsert            |
| **File changed** | Debounce → Read → Parse/Chunk → Embed → Upsert (replace)  |
| **File deleted** | Delete from Qdrant (filter by `file_path` + `project_id`) |
| **File renamed** | Delete old + Upsert new (simple approach)                 |

**Common processing steps:**

```
Read Content → Parse/Chunk → Generate Embeddings → Upsert to Qdrant
                   │
                   ├── Tree-sitter parsing (always, for code files)
                   ├── LSP enrichment (active projects only)
                   ├── Metadata extraction (file_type, language)
                   └── Content hashing (deduplication)
```

---

## Code Intelligence

### Architecture: Tree-sitter Baseline + LSP Enhancement

| Component       | When                     | What it provides                                 |
| --------------- | ------------------------ | ------------------------------------------------ |
| **Tree-sitter** | Always, during ingestion | Symbol definitions, language, syntax structure   |
| **LSP**         | Active projects only     | References (where used), types, resolved imports |

**Rationale:** Tree-sitter is fast and always available. LSP provides richer data but requires spawning language servers, so it's reserved for active projects.

### Tree-sitter (Baseline)

Runs on every code file during ingestion:

- **Symbol definitions:** Function, class, method, struct names and locations
- **Language detection:** From grammar matching
- **Syntax structure:** Imports, exports, declarations
- **Semantic chunking:** Split files into meaningful units (see below)

**Grammars:** Downloaded on demand. Common languages bundled.

### LSP (Enhancement for Active Projects)

Runs when project is active:

```
1. Project activated (RegisterProject received)
2. Daemon spawns language server(s) for detected languages
3. LSP queries enrich existing Qdrant entries:
   - Symbol references (where used)
   - Type information
   - Resolved imports
4. Server kept alive until:
   - Project deactivated, AND
   - All queued items for project processed
```

**One server per language per project.** Multi-target projects (e.g., Cargo workspace with multiple crates) are handled by single language server.

### LSP Server Lifecycle

The daemon manages LSP server instances through a state machine:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSP Server Lifecycle                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   RegisterProject          DeprioritizeProject                       │
│        │                          │                                  │
│        ▼                          ▼                                  │
│  ┌───────────┐            ┌───────────────┐                         │
│  │  Stopped  │──spawn──→  │  Initializing │                         │
│  └───────────┘            └───────┬───────┘                         │
│        ▲                          │                                  │
│        │                          ▼ initialized                      │
│        │                   ┌───────────┐                            │
│        │◄──stop────────────│  Running  │◄──────┐                    │
│        │                   └───────┬───┘       │                    │
│        │                          │            │                    │
│        │                          ▼ unhealthy  │ healthy            │
│        │                   ┌───────────┐       │                    │
│        │◄──max retries─────│  Failed   │───────┘                    │
│        │                   └───────────┘  restart                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Lifecycle states:**
- **Stopped**: No server process running
- **Initializing**: Server spawned, waiting for LSP initialize handshake
- **Running**: Server healthy, accepting queries
- **Failed**: Server crashed or unhealthy

**Deferred shutdown:**
When a project is deprioritized, the daemon checks the queue before stopping LSP servers:
1. Check `unified_queue` for pending items with `tenant_id = project_id`
2. If queue has items, defer shutdown (configurable delay, default 60s)
3. Re-check queue after delay
4. Only stop server when queue is empty

This prevents stopping LSP servers while enrichment queries are still pending.

**State persistence (Task 1.18):**
Server states are persisted to SQLite for recovery after daemon restart:
- Stored: `project_id`, `language`, `project_root`, `restart_count`, `last_started_at`
- Cleaned up: States older than 24 hours
- On initialization: Restore states and re-spawn servers for active projects

**Language server management:**

```bash
wqm lsp install python    # Installs pyright or pylsp
wqm lsp install rust      # Installs rust-analyzer
wqm lsp list              # Shows available/installed servers
wqm lsp remove <lang>     # Removes language server
wqm lsp status            # Show running LSP servers and metrics
```

**PATH configuration:** CLI manages `environment.user_path` in the configuration file.

**Update triggers:**

- CLI installation
- Every CLI invocation
- MCP server startup (which invokes CLI)

**Processing steps:**

1. **Expansion:** Retrieve `$PATH` and expand all environment variables recursively (e.g., `~` → `/Users/chris`, `$XDG_CONFIG_HOME` → `$HOME/.config` → `/Users/chris/.config`)

2. **Merge:** Append the existing `user_path` from config to the expanded `$PATH`, split by OS path separator (`:` on Unix, `;` on Windows), preserving order

3. **Deduplicate:** Remove duplicate path segments, keeping the **first occurrence** only (earlier entries take precedence)

4. **Save:** Recombine segments into a string and write to config **only if different** from the current value (avoids unnecessary disk writes)

**Note:** Only the CLI writes to the configuration file. Daemon reads `user_path` on startup and uses it to locate language server binaries.

### Semantic Code Chunking

Instead of arbitrary text chunks, code files are split into semantic units:

#### Chunk Types

| Chunk Type         | Contains                                       | Example                   |
| ------------------ | ---------------------------------------------- | ------------------------- |
| `preamble`         | Imports, module docstring, constants           | File header               |
| `function`         | Complete function with docstring               | `def validate_token(...)` |
| `class`            | Class signature, docstrings, class-level attrs | `class AuthService:`      |
| `method`           | Method body (linked to parent class)           | `def login(self, ...)`    |
| `struct`           | Struct/dataclass definition                    | `struct Config { ... }`   |
| `trait`/`protocol` | Interface definition                           | `trait Validator { ... }` |

#### Chunking Algorithm

```
For each code file:
1. Parse with Tree-sitter → AST
2. Extract preamble (imports, module-level items)
3. Walk AST, create chunk for each:
   - Function definition
   - Class/struct definition (signature only)
   - Method definition (separate chunk, linked to class)
   - Trait/protocol/interface
4. For large units (>200 lines):
   - Fall back to overlap chunking
   - Mark as is_fragment=true
```

#### Chunk Payload Schema

```json
{
  "project_id": "abc123",
  "file_path": "src/auth.rs",
  "chunk_type": "function",
  "symbol_name": "validate_token",
  "symbol_kind": "function",
  "parent_symbol": null,
  "language": "rust",
  "start_line": 42,
  "end_line": 67,
  "docstring": "Validates JWT token and returns claims.",
  "signature": "fn validate_token(token: &str) -> Result<bool>",
  "calls": ["decode_jwt", "check_expiry"],
  "is_fragment": false
}
```

**LSP enrichment adds:**

```json
{
  "references": [
    { "file": "src/api.rs", "line": 23 },
    { "file": "src/middleware.rs", "line": 56 }
  ],
  "type_info": "fn(&str) -> Result<bool>"
}
```

#### Benefits

| Benefit               | Explanation                              |
| --------------------- | ---------------------------------------- |
| Complete context      | LLM gets whole function, not fragments   |
| Better search         | Query returns complete, meaningful units |
| Symbol association    | Function name tied to its implementation |
| Relationship tracking | Method → Class, Function → Module        |

### CLI Commands

```bash
# Libraries only (projects are auto-watched via MCP)
wqm library add /path/to/docs --name numpy --mode sync
wqm library add /path/to/docs --name pandas --mode incremental
wqm library list
wqm library remove numpy

# Watch management (admin)
wqm watch list                    # List all watches
wqm watch disable <watch_id>      # Temporarily disable
wqm watch enable <watch_id>       # Re-enable
```

---

## API Reference

### MCP Tools

The server provides exactly **4 tools**: `search`, `retrieve`, `memory`, and `store`.

**Important design principles:**

- The MCP server does NOT store content to the `projects` collection. Project content is ingested by the daemon via file watching.
- The MCP can store to `memory` (rules) and `libraries` (reference documentation), and can register new projects with the daemon via `store` with `type: "project"`.
- Session management (activate/deactivate) is automated, not exposed as a tool.
- Health monitoring is server-internal and affects search response metadata.

#### search

Semantic search with optional direct retrieval mode.

```typescript
search({
    query: string,                      // Required: search query
    collection?: "projects" | "libraries" | "memory", // default: "projects"
    mode?: "hybrid" | "semantic" | "keyword" | "retrieve", // default: "hybrid"
    limit?: number,                     // default: 10
    score_threshold?: number,           // default: 0.3 (ignored in retrieve mode)
    // Collection-specific scope filters (see below)
    scope?: string,                     // Scope within collection
    branch?: string,                    // For projects: branch filter
    project_id?: string,               // For projects: specific project
    library_name: str = None,        # For libraries: specific library
    # Content type filters
    file_type: str = None,           # Filter by document type (see below)
    tag: str = None,                 # Tag filter (dot-separated hierarchy)
    # Cross-collection options
    include_libraries: bool = False, # Also search libraries collection
    include_deleted: bool = False    # Include deleted documents (libraries only)
)
```

**Modes:**

- `hybrid`: Semantic + keyword search (default)
- `semantic`: Pure vector similarity
- `keyword`: Keyword/exact matching
- `retrieve`: Direct document access by ID or metadata (no ranking)

**file_type values:**

| Value    | Description                                |
| -------- | ------------------------------------------ |
| `code`   | Source code files (.rs, .py, .ts, etc.)    |
| `doc`    | Documentation files (.md, .txt, .rst)      |
| `test`   | Test files (test_*, *_test.*, etc.)        |
| `config` | Configuration files (.yaml, .json, .toml)  |
| `note`   | User notes and scratch content             |
| `artifact`| Build outputs, generated files            |

**include_libraries:**

When `include_libraries=True`, search queries the `libraries` collection in addition to the primary collection. This enables cross-collection search for finding related documentation alongside project code. Results from both collections are fused using Reciprocal Rank Fusion.

**Collection-specific scope:**

| Collection  | Scope Options              | Notes                                           |
| ----------- | -------------------------- | ----------------------------------------------- |
| `memory`    | `all`, `global`, `project` | `project` = current project's rules             |
| `projects`  | `all`, `current`, `other`  | Combined with `branch` and `project_id` filters |
| `libraries` | `all`, `<library_name>`    | Filter by specific library                      |

**Project scope examples:**

```typescript
// Current project, current branch (default)
search({ query: "auth", collection: "projects", scope: "current" });

// Current project, all branches
search({ query: "auth", collection: "projects", scope: "current", branch: "*" });

// All projects
search({ query: "auth", collection: "projects", scope: "all" });

// Specific project
search({ query: "auth", collection: "projects", scope: "other", project_id: "abc123" });
```

**project_id handling:** The MCP server FETCHES `project_id` from the daemon's state database (not calculated locally). This prevents drift between MCP and daemon. The fetch happens on first search operation to allow time for daemon to register the watch folder.

#### retrieve

Direct document access for chunk-by-chunk retrieval.

```typescript
retrieve({
    document_id?: string,              // Specific document ID
    collection?: "projects" | "libraries" | "memory", // default: "projects"
    metadata?: Record<string, unknown>, // Metadata filters
    limit?: number,                    // default: 10
    offset?: number,                   // default: 0, for pagination
});
```

**Use case:** Retrieving large documents chunk by chunk without overwhelming context. Use `search` with `mode="retrieve"` for metadata-based retrieval, or `retrieve` for ID-based access.

#### memory

Manage memory rules (behavioral preferences).

```typescript
memory({
    action: "add" | "update" | "remove" | "list", // Required
    label?: string,                    // Rule label (unique per scope)
    content?: string,                  // Rule content (for add/update)
    scope?: "project" | "global",      // default: "project"
    project_id?: string,               // For project-scoped rules (auto-detected if omitted)
});
```

**Actions:**

- `add`: Create new rule (queued for daemon)
- `update`: Update existing rule (queued for daemon)
- `remove`: Remove rule (queued for daemon)
- `list`: List rules (implemented as a search query against the memory collection)

**Implementation note:** The `list` action is internally a semantic search with scope filtering. This allows consistent behavior with other collection queries while providing a simpler interface for rule management.

**Uniqueness:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

**LLM Generation Guidelines:**

When creating memory rules, the LLM should generate metadata fields following these constraints:

| Field | Constraints | Examples |
|-------|-------------|----------|
| `label` | Max 15 chars, format: `word-word-word` (lowercase, hyphen-separated) | `prefer-uv`, `use-pytest`, `strict-types`, `no-mock-fs` |
| `title` | Max 50 chars, human-readable summary | "Use uv instead of pip for Python packages" |
| `tags` | Max 5 tags, max 20 chars each | `["python", "tooling"]`, `["testing", "best-practice"]` |

**Conversational Flow Examples:**

```
User: "For future reference, always use uv instead of pip"
→ LLM calls: memory(action="add", label="prefer-uv", title="Use uv for Python packages",
                    content="Always use uv instead of pip for Python package management...",
                    tags=["python", "tooling"])

User: "Remember to run tests before committing"
→ LLM calls: memory(action="add", label="pre-commit-tests", title="Run tests before commits",
                    content="Always run the test suite before committing changes...",
                    tags=["git", "testing"])

User: "Actually, let me update that rule about testing"
→ LLM calls: memory(action="update", label="pre-commit-tests", title="Run tests before commits",
                    content="Run the full test suite AND linting before committing...")
```

**Label Generation Strategy:**

1. Extract key concepts from user's instruction
2. Form 2-3 word hyphenated identifier
3. Keep descriptive but concise
4. Use action verbs when appropriate: `use-`, `prefer-`, `avoid-`, `no-`

#### store

Store content to libraries collection or register a project with the daemon. The `type` parameter determines the operation mode.

**Type: `"library"` (default) — Store reference documentation:**

```typescript
store({
    type?: "library",                  // Default: "library"
    content: string,                   // Required: text content
    library_name: string,              // Required: library identifier (acts as tag)
    title?: string,                    // Document title
    source?: "user_input" | "web" | "file", // default: "user_input"
    url?: string,                      // Source URL (for web content)
    metadata?: Record<string, unknown>, // Additional metadata
});
```

**Type: `"project"` — Register a project directory:**

```typescript
store({
    type: "project",                   // Required: must be "project"
    path: string,                      // Required: absolute path to project directory
    name?: string,                     // Optional: display name (defaults to directory name)
});
```

Registers a new project with the daemon for file watching and ingestion. Uses `register_if_new: true` so the daemon will create the project in `watch_folders` if it doesn't already exist. Returns `{ success, project_id, created, is_active, message }`.

**Note:** `store` with `type: "library"` is for adding reference documentation to the `libraries` collection (like adding books to a library). It is NOT for project content (handled by daemon file watching) or memory rules (use `memory` tool). Use `type: "project"` to register a new project directory with the daemon.

**Libraries definition:** Libraries are collections of reference information (books, documentation, papers, websites) - NOT programming libraries (use context7 MCP for those).

### Session Lifecycle

Session lifecycle is **automatic**, managed by the MCP server using the MCP SDK's `server.onclose` callback and server initialization logic.

**Implementation:** The MCP server uses `@modelcontextprotocol/sdk` which provides:
- Session initialization on transport connection (stdio or HTTP)
- `server.onclose` callback for cleanup when session ends

#### On Server Start (Transport Connection)

When the MCP server connects to the transport (stdio or HTTP):

1. **Project detection and conditional activation:**
   - Server detects project from working directory and computes `project_id`
   - Server queries daemon via `GetProjectStatus` to check if project exists in `watch_folders`
   - **If registered:** Server sends `RegisterProject` (with `register_if_new=false`, the default)
     - Daemon sets `is_active = true` and updates `last_activity_at`
   - **If not registered:** Server logs a warning ("Project not registered, use `wqm project add` or the `store` tool with `type: \"project\"` to register") and continues without activation — search and memory tools still work, but file watching is not started

2. **Start heartbeat:**
   - Periodic heartbeat with daemon to prevent timeout-based deactivation
   - Only started if project was successfully activated in step 1

#### On Session End (server.onclose)

1. **Project deactivation:**
   - Server sends `DeprioritizeProject` to daemon
   - Daemon sets `is_active = false` for the project

2. **Process cleanup:**
   - Daemon shuts down any spawned processes for that project (e.g., LSP servers)

#### Memory Injection

Memory injection is handled via the `memory` MCP tool:
- Claude reads memory rules by calling the `memory` tool with `action: "list"`
- Memory is NOT automatically injected at session start (that would require external hooks)

**Optional Enhancement via Claude Code Hooks:** For automatic memory injection, users can configure a `SessionStart` hook in their `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "/path/to/inject-memory.sh"
      }]
    }]
  }
}
```
This is external to the MCP server and optional.

#### Memory and Project ID Changes

When a project is renamed or its `project_id` changes (e.g., due to disambiguation when a second clone is detected), the memory records in Qdrant must have their `project_id` field updated to maintain association.

### Health Status Integration

Health monitoring is **server-internal** and affects search responses:

**When system is healthy** (daemon + Qdrant connected):

```json
{
  "results": [...],
  "status": "healthy"
}
```

**When system is unhealthy** (daemon or Qdrant unavailable):

```json
{
  "results": [...],
  "status": "uncertain",
  "reason": "daemon_unavailable",
  "message": "Results may be incomplete. File changes since daemon went offline are not reflected."
}
```

The `uncertain` status indicates that:

- Search results are from the last known state
- Recent file changes may not be indexed
- Memory rules may be stale
- The user should be aware results might be incomplete

### Removed/Automated Features

The following are **not exposed as MCP tools**:

| Feature                         | Status          | Reason                                        |
| ------------------------------- | --------------- | --------------------------------------------- |
| `health`                        | Server-internal | Affects search response metadata (see above)  |
| `session` (activate/deactivate) | Automated       | MCP server handles automatically              |
| `list_collections`              | CLI only        | Diagnostic, use `wqm admin collections`       |
| `collection_info`               | CLI only        | Diagnostic, use `wqm admin`                   |
| `workspace_status`              | Removed         | Replaced by health status in search responses |
| `init_project`                  | Removed         | Daemon handles via watching                   |
| `cleanup`                       | Removed         | Daemon handles internally                     |
| `create_collection`             | Removed         | Daemon owns collections                       |
| `delete_collection`             | Removed         | Daemon owns collections                       |
| `mark_library_deleted`          | CLI only        | Use `wqm library` commands                    |
| `restore_deleted_library`       | CLI only        | Use `wqm library` commands                    |
| `list_deleted_libraries`        | CLI only        | Use `wqm library` commands                    |

**The legacy `manage` tool is completely removed.** Memory operations use the dedicated `memory` tool.

### gRPC Services

The daemon exposes 5 gRPC services on port 50051.

#### SystemService (7 RPCs)

System-level operations for monitoring, metrics, and lifecycle management.

| Method                | Used By  | Purpose                         | Status     |
| --------------------- | -------- | ------------------------------- | ---------- |
| `Health`              | MCP, CLI | Quick health check for alerting | Production |
| `GetStatus`           | CLI      | Comprehensive system snapshot   | Production |
| `GetMetrics`          | CLI      | Performance metrics             | Production |
| `GetQueueStats`       | CLI      | Queue statistics                | Production |
| `Shutdown`            | CLI      | Graceful daemon shutdown        | Production |
| `SendRefreshSignal`   | CLI      | Signal database state changes   | Production |
| `NotifyServerStatus`  | MCP      | Server lifecycle notifications  | Production |

#### CollectionService (7 RPCs)

Collection CRUD and alias management. Most methods are daemon-internal.

| Method                  | Status              | Notes                                 |
| ----------------------- | ------------------- | ------------------------------------- |
| `CreateCollection`      | **Daemon internal** | Daemon creates collections on startup |
| `DeleteCollection`      | **Not used**        | Fixed 4-collection model              |
| `ListCollections`       | Read-only           | Can be exposed to MCP/CLI             |
| `GetCollection`         | Read-only           | Can be exposed to MCP/CLI             |
| `CreateCollectionAlias` | **Daemon internal** | For tenant_id changes                 |
| `DeleteCollectionAlias` | **Daemon internal** | Alias cleanup                         |
| `RenameCollectionAlias` | **Daemon internal** | Atomic alias rename                   |

**MCP/CLI must NOT call collection mutation methods.** Only read-only methods are permitted.

#### DocumentService (3 RPCs)

Document ingestion and management. Reserved for admin/diagnostic use.

| Method           | Status       | Notes                     |
| ---------------- | ------------ | ------------------------- |
| `IngestText`     | **Reserved** | Admin/diagnostic use only |
| `UpdateDocument` | **Reserved** | Admin/diagnostic use only |
| `DeleteDocument` | **Reserved** | Admin/diagnostic use only |

**Production writes use SQLite queue.** These methods exist for administrative and diagnostic purposes but are not called by MCP or CLI in normal operation.

#### EmbeddingService (2 RPCs)

Embedding generation for TypeScript MCP server. Centralizes embedding model in daemon.

| Method                 | Used By | Purpose                           | Status     |
| ---------------------- | ------- | --------------------------------- | ---------- |
| `EmbedText`            | MCP     | Generate dense vector (384 dims)  | Production |
| `GenerateSparseVector` | MCP     | Generate BM25 sparse vector       | Production |

**Usage:**
- TypeScript MCP server calls `EmbedText` when performing hybrid search
- Dense vectors use FastEmbed `all-MiniLM-L6-v2` model (384 dimensions)
- Sparse vectors use BM25 algorithm with configurable `k1` (default: 1.2) and `b` (default: 0.75)

#### ProjectService (5 RPCs)

Multi-tenant project lifecycle and session management.

| Method                | Used By  | Purpose                          | Status                       |
| --------------------- | -------- | -------------------------------- | ---------------------------- |
| `RegisterProject`     | MCP, CLI | Re-activate or register project  | **Production (direct gRPC)** |
| `DeprioritizeProject` | MCP      | Deactivate project session       | **Production (direct gRPC)** |
| `GetProjectStatus`    | MCP      | Get project status               | Production                   |
| `ListProjects`        | CLI      | List all registered projects     | Production                   |
| `Heartbeat`           | MCP      | Keep session alive (60s)         | Production                   |

**Registration Policy:**
- `RegisterProject` accepts a `register_if_new` boolean field (default: `false`)
  - `register_if_new=false` (MCP default): Only re-activates existing `watch_folders` entries. Returns error if project not found.
  - `register_if_new=true` (CLI-initiated): Creates a new `watch_folders` entry if project doesn't exist, then activates it.
- MCP servers call `RegisterProject` on startup with `register_if_new=false` — they only re-activate known projects
- CLI `wqm project add` calls `RegisterProject` with `register_if_new=true` to create new entries

**Session Management:**
- `Heartbeat` must be called periodically (within 60s timeout) to maintain session
- `DeprioritizeProject` is called when MCP server stops

---

## Grammar and Runtime Management

The daemon manages tree-sitter grammars and ONNX runtime as external dependencies with automatic updates.

### Cache Locations

| Component            | Default Location                | Config Key            |
| -------------------- | ------------------------------- | --------------------- |
| Tree-sitter grammars | `~/.workspace-qdrant/grammars/` | `grammars.cache_dir`  |
| Embedding models     | `~/.workspace-qdrant/models/`   | `embedding.cache_dir` |

### Tree-sitter Grammar Management

**Configuration:**

```yaml
grammars:
  cache_dir: "~/.workspace-qdrant/grammars/"
  required:
    - rust
    - python
    - typescript
    - javascript
  auto_download: true # Download missing grammars automatically
```

**Daemon behavior:**

1. On startup, check each required grammar exists in cache
2. If grammar missing and `auto_download: true`, download from grammar repository
3. If grammar version mismatches tree-sitter runtime version, replace with compatible version
4. If `auto_download: false` and grammar missing, log warning and skip that language

### Manual Updates via CLI

```bash
wqm update                    # Check for updates and install if available
wqm update --check            # Check only, don't install
wqm update --force            # Force reinstall current version
wqm update --version 1.2.3    # Install specific version
```

**Update behavior:**

1. Query GitHub releases API for latest version
2. Compare with currently installed version
3. If newer version available (or `--force`):
   - Download appropriate binary for current platform
   - Verify checksum
   - Stop running daemon gracefully
   - Replace binary
   - Restart daemon
4. Report success/failure

**Configuration:**

```yaml
updates:
  check_on_startup: false # Auto-check for updates when daemon starts
  notify_only: true # If true, only notify; don't auto-install
  channel: "stable" # stable|beta|nightly
```

### Continuous Integration

**Automated releases triggered by upstream updates:**

| Trigger                  | Action                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| New tree-sitter release  | Rebuild daemon, bump patch version, release with message "tree-sitter version bump to X.Y.Z"  |
| New ONNX Runtime release | Rebuild daemon, bump patch version, release with message "ONNX Runtime version bump to X.Y.Z" |

**Target platforms (6 binaries per release):**

| Platform            | Target Triple               |
| ------------------- | --------------------------- |
| Linux ARM64         | `aarch64-unknown-linux-gnu` |
| Linux x86_64        | `x86_64-unknown-linux-gnu`  |
| macOS Apple Silicon | `aarch64-apple-darwin`      |
| macOS Intel         | `x86_64-apple-darwin`       |
| Windows ARM64       | `aarch64-pc-windows-msvc`   |
| Windows x86_64      | `x86_64-pc-windows-msvc`    |

**CI workflow:**

1. Monitor upstream releases (tree-sitter, ONNX Runtime) via GitHub Actions or webhook
2. On new release detected, trigger build pipeline
3. Build for all 6 targets
4. Run integration tests on each platform
5. Create GitHub release with all binaries
6. Update homebrew formula / other package managers

---

## Configuration Reference

### Single Configuration File

**All components (Daemon, MCP Server, CLI) share the same configuration file.** This is a key architectural requirement to prevent configuration drift between components.

**Configuration file search order (first found wins, identical across all components):**

1. `WQM_CONFIG_PATH` environment variable (explicit override)
2. `~/.workspace-qdrant/config.yaml` (or `.yml`)
3. `$XDG_CONFIG_HOME/workspace-qdrant/config.yaml` (or `.yml`; defaults to `~/.config`)
4. `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS only)

No project-local `.workspace-qdrant.yaml` is searched. All components use the same cascade.

**Built-in defaults:** The configuration template (`assets/default_configuration.yaml`) defines all default values. User configuration only needs to specify overrides.

**Embedded defaults:** All components embed `assets/default_configuration.yaml` at build time:
- **Rust (daemon + CLI):** The shared `wqm-common` crate uses `include_str!()` to embed the YAML and provides `DEFAULT_YAML_CONFIG` (a `LazyLock<YamlConfig>`) as the single source of truth. Both daemon and CLI derive their `Default` impls from this parsed config.
- **TypeScript MCP server:** build step generates `default-config.ts` from YAML, or inline import

This ensures defaults are always in sync with the binary version without requiring a deployed config file.

#### Directory Organization

The `~/.workspace-qdrant/` directory is for **configuration and state only**, not logs:

| Directory | Purpose | XDG Equivalent |
|-----------|---------|----------------|
| `~/.workspace-qdrant/config.yaml` | Configuration file | `$XDG_CONFIG_HOME/workspace-qdrant/` |
| `~/.workspace-qdrant/state.db` | SQLite database | `$XDG_DATA_HOME/workspace-qdrant/` |

**Logs use OS-canonical paths** (see [Logging and Observability](#logging-and-observability)):

| OS | Log Directory |
|----|---------------|
| Linux | `$XDG_STATE_HOME/workspace-qdrant/logs/` (default: `~/.local/state/workspace-qdrant/logs/`) |
| macOS | `~/Library/Logs/workspace-qdrant/` |
| Windows | `%LOCALAPPDATA%\workspace-qdrant\logs\` |

**XDG Base Directory compliance (Linux):** If `$XDG_CONFIG_HOME` is set, configuration searches `$XDG_CONFIG_HOME/workspace-qdrant/` before `~/.workspace-qdrant/`.

### Environment Variables

Environment variables override configuration file values:

| Variable            | Description                        | Default                            |
| ------------------- | ---------------------------------- | ---------------------------------- |
| `WQM_CONFIG_PATH`   | Explicit config file path          | None (uses search order)           |
| `WQM_DATABASE_PATH` | Override database location         | `~/.workspace-qdrant/state.db`     |
| `QDRANT_URL`        | Qdrant server URL                  | `http://localhost:6333`            |
| `QDRANT_API_KEY`    | Qdrant API key                     | None                               |
| `FASTEMBED_MODEL`   | Embedding model                    | `all-MiniLM-L6-v2`                 |
| `WQM_DAEMON_PORT`   | Daemon gRPC port                   | `50051`                            |
| `WQM_STDIO_MODE`    | Force stdio mode                   | `false`                            |
| `WQM_CLI_MODE`      | Force CLI mode                     | `false`                            |
| `WQM_LOG_LEVEL`     | Log level (DEBUG, INFO, WARN, ERROR) | `INFO`                           |

### Configuration Structure

The unified configuration file contains all settings. The following shows every section parsed by `DaemonConfig` with defaults:

```yaml
# Database configuration
database:
  path: ~/.workspace-qdrant/state.db

# Qdrant connection
qdrant:
  url: http://localhost:6333
  api_key: null
  timeout_ms: 30000
  max_retries: 3
  retry_delay_ms: 1000
  transport: grpc                       # grpc | http
  pool_size: 10
  tls: false
  dense_vector_size: 1536
  check_compatibility: true

# Daemon top-level settings
daemon:
  log_file: null                        # Override log file path
  log_level: info                       # DEBUG, INFO, WARN, ERROR
  max_concurrent_tasks: 4               # Max parallel processing tasks
  default_timeout_ms: 30000             # Task timeout
  enable_preemption: true               # Allow task preemption
  chunk_size: 1000                      # Default batch processing unit size
  grpc_port: 50051

  # Resource limits (prevent daemon from consuming excessive CPU/memory)
  resource_limits:
    nice_level: 10                      # OS-level priority (-20 highest to 19 lowest)
    inter_item_delay_ms: 50             # Breathing room between queue items (0-5000)
    max_concurrent_embeddings: 2        # Concurrent ONNX embedding ops (1-8)
    max_memory_percent: 70              # Pause processing above this % (20-95)

# Auto-ingestion settings
auto_ingestion:
  enabled: true
  auto_create_watches: true
  include_common_files: true
  include_source_files: true
  target_collection_suffix: scratchbook
  max_files_per_batch: 5
  batch_delay_seconds: 2.0
  max_file_size_mb: 50
  recursive_depth: 5
  debounce_seconds: 10

# Queue processor configuration
queue_processor:
  batch_size: 10                        # Items per dequeue batch
  poll_interval_ms: 500                 # Poll interval between batches
  max_retries: 5                        # Max retry attempts before marking failed
  retry_delays_seconds: [60, 300, 900, 3600]  # Backoff schedule
  target_throughput: 1000               # Target docs/min for monitoring
  enable_metrics: true                  # Enable performance metrics
  # Env overrides: WQM_QUEUE_BATCH_SIZE, WQM_QUEUE_POLL_INTERVAL_MS,
  #   WQM_QUEUE_MAX_RETRIES, WQM_QUEUE_TARGET_THROUGHPUT, WQM_QUEUE_ENABLE_METRICS

# Logging configuration
logging:
  info_includes_connection_events: true
  info_includes_transport_details: true
  info_includes_retry_attempts: true
  info_includes_fallback_behavior: true
  error_includes_stack_trace: true
  error_includes_connection_state: true

# Tool monitoring
monitoring:
  enable_monitoring: true
  check_on_startup: true
  check_interval_hours: 24
  # Env overrides: WQM_MONITOR_CHECK_INTERVAL_HOURS, WQM_MONITOR_CHECK_ON_STARTUP,
  #   WQM_MONITOR_ENABLE

# Observability (metrics and telemetry)
observability:
  collection_interval: 60              # Seconds between metric snapshots
  metrics:
    enabled: false
  telemetry:
    enabled: false
    history_retention: 120
    cpu_usage: true
    memory_usage: true
    latency: true
    queue_depth: true
    throughput: true

# Git integration
git:
  enable_branch_detection: true
  cache_ttl_seconds: 60                # Branch info cache TTL
  # Env overrides: WQM_GIT_ENABLE_BRANCH_DETECTION, WQM_GIT_CACHE_TTL_SECONDS

# Embedding generation
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim, 256-token max
  enable_sparse_vectors: true          # Enable BM25 sparse vectors for hybrid search
  chunk_size: 384                      # Chars per chunk (≈82 prose tokens, ≈110 code tokens)
  chunk_overlap: 58                    # 15% overlap for context preservation
  cache_max_entries: 1000              # Max cached embedding results
  model_cache_dir: null                # Override model download dir (~/.cache/fastembed/)
  # Env overrides: WQM_EMBEDDING_CACHE_MAX_ENTRIES, WQM_EMBEDDING_MODEL_CACHE_DIR

# LSP (Language Server Protocol) integration
lsp:
  user_path: null                      # User PATH for finding language servers
  max_servers_per_project: 3
  auto_start_on_activation: true
  deactivation_delay_secs: 60         # Delay before stopping servers on deactivation
  enable_enrichment_cache: true
  cache_ttl_secs: 300
  startup_timeout_secs: 30
  request_timeout_secs: 10
  health_check_interval_secs: 60
  max_restart_attempts: 3
  restart_backoff_multiplier: 2.0
  enable_auto_restart: true
  stability_reset_secs: 3600          # Reset restart count after this period

# Tree-sitter grammar configuration
grammars:
  cache_dir: ~/.workspace-qdrant/grammars
  required: [rust, python, javascript, typescript, go, java, c, cpp]
  auto_download: true
  tree_sitter_version: "0.24"
  download_base_url: "https://github.com/tree-sitter/tree-sitter-{language}/releases/download/v{version}/tree-sitter-{language}-{platform}.{ext}"
  verify_checksums: true
  lazy_loading: true
  check_interval_hours: 168            # Weekly grammar update check

# Daemon self-update
updates:
  auto_check: true
  channel: stable                      # stable | beta | dev
  notify_only: true                    # Announce but don't auto-install
  check_interval_hours: 24

# File watching and ingestion filtering
watching:
  # Allowlist: Only files with these extensions are eligible for ingestion.
  # See "File Type Allowlist" section for the complete categorized list.
  # User config can extend or restrict this list.
  allowed_extensions:
    - ".rs"       # Rust
    - ".py"       # Python
    - ".js"       # JavaScript
    - ".ts"       # TypeScript
    - ".go"       # Go
    - ".java"     # Java
    - ".md"       # Markdown
    # ... 400+ extensions (see default_configuration.yaml for complete list)

  # Extension-less files recognized by exact name
  allowed_filenames:
    - "Makefile"
    - "Dockerfile"
    - "Jenkinsfile"
    - "Gemfile"
    - "Rakefile"
    - "CMakeLists.txt"
    - ".gitignore"
    - ".editorconfig"
    # ... 30+ filenames (see default_configuration.yaml for complete list)

  # Directories always excluded (checked by path component)
  exclude_directories:
    - "node_modules"
    - "target"
    - "build"
    - "dist"
    - ".git"
    - "__pycache__"
    - ".venv"
    - "DerivedData"
    - ".fastembed_cache"
    # ... 40+ directories (see default_configuration.yaml for complete list)

  # Additional file pattern exclusions
  exclude_patterns:
    - "*.pyc"
    - "*.class"
    - "*.o"
    - "*.obj"
    - "*.lock"

  # Size-restricted extensions: allowed but with stricter size limit
  size_restricted_extensions:
    - ".csv"
    - ".tsv"
    - ".json"
    - ".jsonc"
    - ".json5"
    - ".xml"
    - ".jsonl"
    - ".ndjson"
    - ".log"
    - ".sql"
  size_restricted_max_mb: 1            # Max size for restricted extensions (default 1MB)

# Collections configuration
collections:
  memory_collection_name: "memory"

# User environment (written by CLI, read by daemon)
environment:
  user_path: "/usr/local/bin:/opt/homebrew/bin:..." # Set by CLI on first run
```

**Note:** The `watching` section replaces the old `patterns`/`ignore_patterns` approach. The allowlist (`allowed_extensions` + `allowed_filenames`) is the primary ingestion gate. See the [File Type Allowlist](#file-type-allowlist) section for the complete categorized list and ingestion gate layering. The full default list is embedded at build time from `assets/default_configuration.yaml`.

### Qdrant Dashboard Visualization

When using the Qdrant dashboard (web UI) to visualize collections, note that this system uses **named vectors**. The standard vector visualization will not work without specifying the vector name.

**Dashboard configuration:**
- In the dashboard's "Visualize" tab, use the `using` parameter to specify which named vector to use
- Select `dense` for semantic (embedding) visualization — this produces meaningful spatial clusters
- Do NOT select `sparse` — sparse (BM25) vectors are not visualizable in 2D/3D projections

**Available named vectors per collection:**

| Collection | Named Vectors | Visualization |
|------------|--------------|---------------|
| `projects` | `dense` (384-dim), `sparse` (BM25) | Use `dense` |
| `libraries` | `dense` (384-dim), `sparse` (BM25) | Use `dense` |
| `memory` | `dense` (384-dim) | Use `dense` |

**Distance Matrix API:** For graph visualization of semantically similar documents, use Qdrant's Distance Matrix API to compute pairwise distances between points using the `dense` vector. This can reveal clusters of related code files or documentation.

**Future consideration:** GraphRAG patterns could leverage the distance matrix to build code intelligence graphs (e.g., "files that are semantically related" or "functions that co-occur in similar contexts"). See the [Future Development](#future-development-wishlist-not-yet-scoped) section for detailed research findings.

### SQLite Database

**Path:** `~/.workspace-qdrant/state.db`

**Owner:** Rust daemon (memexd) - see [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)

**Core Tables (5):**

| Table            | Purpose                                                           | Used By          |
| ---------------- | ----------------------------------------------------------------- | ---------------- |
| `schema_version` | Schema version tracking                                           | Daemon           |
| `unified_queue`  | Write queue for daemon processing                                 | MCP, CLI, Daemon |
| `watch_folders`  | Unified table for projects, libraries, and submodules (see below) | MCP, CLI, Daemon |
| `tracked_files`  | Authoritative file inventory with metadata                        | Daemon (write), CLI (read) |
| `qdrant_chunks`  | Qdrant point tracking per file chunk (child of tracked_files)     | Daemon only      |

**Note:** The `watch_folders` table consolidates what were previously separate `registered_projects`, `project_submodules`, and `watch_folders` tables. See [Watch Folders Table (Unified)](#watch-folders-table-unified) for the complete schema.

**Note:** `tracked_files` + `qdrant_chunks` together form the authoritative file inventory, replacing the need to scroll Qdrant for file listings. See [Tracked Files Table](#tracked-files-table) and [Qdrant Chunks Table](#qdrant-chunks-table) for schemas.

#### Schema Version Table

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
```

The daemon checks this table on startup and runs migrations if needed. Other components must NOT modify this table.

**Other components (MCP, CLI) may read/write to tables but must NOT create tables or run migrations.**

---

## Deployment and Installation

This section documents the deployment architecture, platform support, installation methods, and operational procedures for workspace-qdrant-mcp.

### Deployment Architecture Overview

The system consists of three primary components that work together:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Claude Desktop / Claude Code               │
│                         (MCP Client)                            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ MCP Protocol (stdio)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Server (TypeScript)                     │
│                  src/typescript/mcp-server/                     │
│         Exposes: search, retrieve, memory, store tools          │
└──────────┬──────────────────────────────────────────┬───────────┘
           │ gRPC (localhost:50051)                   │ HTTP REST
           ▼                                          ▼
┌─────────────────────────┐              ┌────────────────────────┐
│    Daemon (memexd)      │              │   Qdrant Vector DB     │
│    Rust binary          │◄────────────►│   (localhost:6333)     │
│                         │  Qdrant API  │                        │
│  - File watching        │              │  Collections:          │
│  - Queue processing     │              │  - projects (unified)  │
│  - Code intelligence    │              │  - libraries (unified) │
│  - Embedding generation │              │  - memory              │
└──────────┬──────────────┘              └────────────────────────┘
           │
           ▼
┌─────────────────────────┐
│    SQLite Database      │
│  ~/.workspace-qdrant/   │
│      state.db           │
│                         │
│  - unified_queue        │
│  - watch_folders        │
│  - tracked_files        │
│  -   qdrant_chunks      │
│  - schema_version       │
└─────────────────────────┘
```

**Deployment Modes:**

| Mode | Qdrant Location | Use Case |
|------|-----------------|----------|
| All-in-One | Bundled in Docker | Development, testing |
| Qdrant-External | Separate Docker container | Production self-hosted |
| Qdrant-Local | Host-installed Qdrant | Development with persistence |
| Qdrant-Cloud | Qdrant Cloud service | Production managed |

### Platform Support Matrix

The system supports 6 platform/architecture combinations:

| Platform | Architecture | Target Triple | Notes |
|----------|--------------|---------------|-------|
| Linux | x86_64 | `x86_64-unknown-linux-gnu` | Primary CI/CD target |
| Linux | ARM64 | `aarch64-unknown-linux-gnu` | AWS Graviton, Raspberry Pi |
| macOS | Apple Silicon | `aarch64-apple-darwin` | M1/M2/M3 Macs |
| macOS | Intel | `x86_64-apple-darwin` | Pre-2020 Macs |
| Windows | x86_64 | `x86_64-pc-windows-msvc` | Windows 10/11 |
| Windows | ARM64 | `aarch64-pc-windows-msvc` | Surface Pro X, ARM laptops |

**Platform-Specific Notes:**

- **Linux**: Uses `inotify` for file watching. Systemd user service for daemon management.
- **macOS**: Uses `FSEvents` for file watching. Launchd plist for daemon management.
- **Windows**: Uses `ReadDirectoryChangesW` for file watching. Service support planned.

**ONNX Runtime Static Linking:**

The Rust daemon statically links ONNX Runtime for embedding generation. This ensures self-contained binaries that work without external dependencies:

| Platform | Build Approach |
|----------|----------------|
| Linux x86_64 | Static linking via `ort` crate prebuilt binaries |
| Linux ARM64 | Static linking via `ort` crate prebuilt binaries |
| macOS ARM64 | Static linking via `ort` crate prebuilt binaries |
| macOS Intel | **Special case**: ONNX Runtime compiled standalone first, then statically linked (no prebuilt binaries available from `ort` crate) |
| Windows x86_64 | Static linking via `ort` crate prebuilt binaries |
| Windows ARM64 | Static linking via `ort` crate prebuilt binaries |

**Critical Requirements:**
- Release binaries MUST be self-contained (no external ONNX Runtime dependency)
- Source builds MUST also produce self-contained binaries
- Users should NEVER need to install ONNX Runtime separately (e.g., via Homebrew)
- The `ort-load-dynamic` feature MUST NOT be used in production builds

**Allowed External Dependencies (per platform):**
- **macOS**: System libraries in `/usr/lib/` and `/System/Library/Frameworks/` only
- **Linux**: `libc`, `libm`, `libdl`, `libpthread`, `libgcc_s`, `libstdc++`, `librt` only
- **Windows**: Windows system DLLs (`KERNEL32`, `ADVAPI32`, `WS2_32`, `bcrypt`, etc.) only
- **All platforms**: Tree-sitter grammar files (external `.so`/`.dylib`/`.dll` in cache dir) and LSP binaries (external executables)

**Release Verification:**
The CI release workflow (`release.yml`) enforces self-contained binaries via:
1. Per-platform dependency verification scripts (`scripts/verify-deps-{linux,macos,windows}.*`) that **fail the build** if unexpected dynamic dependencies are found
2. Smoke tests (`--version`) that verify binaries are runnable before release

**Intel Mac Build Pipeline:**
Since the `ort` crate doesn't provide prebuilt static libraries for `x86_64-apple-darwin`:
1. CI downloads ONNX Runtime source or prebuilt static library
2. Compiles/extracts static library for Intel Mac target
3. Links statically with the daemon during cargo build
4. Produces self-contained `memexd` binary

**Tree-sitter Grammar Compatibility:**

Pre-compiled grammar libraries are included for each platform. Custom grammars can be loaded from `~/.workspace-qdrant/grammars/`.

### Installation Methods

#### Binary Releases (Recommended)

Download pre-built binaries from GitHub Releases:

**Quick Install (Recommended):**

```bash
# Linux/macOS - one-liner
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash

# Windows (PowerShell) - one-liner
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

**Manual Download:**

```bash
# Linux/macOS
VERSION="v0.4.0"  # Or use 'latest'
PLATFORM="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | sed 's/x86_64/x64/' | sed 's/aarch64/arm64/')"
curl -fsSL "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/${VERSION}/workspace-qdrant-mcp-${PLATFORM}.tar.gz" | tar xz
sudo mv wqm memexd /usr/local/bin/

# Windows (PowerShell)
$Version = "v0.4.0"
Invoke-WebRequest -Uri "https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/download/$Version/workspace-qdrant-mcp-windows-x64.zip" -OutFile wqm.zip
Expand-Archive wqm.zip -DestinationPath "$env:LOCALAPPDATA\Programs\wqm"
# Add to PATH manually
```

**Release Assets:**

Each release includes:
- `wqm` - CLI binary
- `memexd` - Daemon binary
- `grammars/` - Pre-compiled tree-sitter grammars
- `assets/` - Default configuration files

#### Source Build

For development or custom builds:

```bash
# Prerequisites
# - Rust 1.75+ (rustup recommended)
# - Node.js 18+ with npm
# - Protocol Buffers compiler (protoc)

# Clone and build
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp

# Use the install script
./install.sh                    # Interactive mode
./install.sh --release          # Release build
./install.sh --prefix=/usr/local # Custom install location
./install.sh --no-daemon        # CLI only, no daemon

# Or build manually
cd src/rust/daemon && cargo build --release
cd src/rust/cli && cargo build --release
cd src/typescript/mcp-server && npm install && npm run build
```

**Install Script Options:**

| Option | Description |
|--------|-------------|
| `--prefix PATH` | Installation directory (default: `~/.local`) |
| `--force` | Clean rebuild from scratch (cargo clean) |
| `--cli-only` | Build only CLI, skip daemon |
| `--no-service` | Skip daemon service setup instructions |
| `--no-verify` | Skip verification steps |

#### npm Global Install (MCP Server Only)

For MCP server without daemon:

```bash
npm install -g workspace-qdrant-mcp

# Verify installation
npx workspace-qdrant-mcp --version
```

> **Note:** The npm package only includes the MCP server. For full functionality, install the Rust daemon separately.

#### Package Managers

**Homebrew (macOS/Linux):**

A Homebrew formula is available in the repository:

```bash
# Install from local formula (when building from source)
brew install --build-from-source ./Formula/workspace-qdrant-mcp.rb

# Or install from tap (when tap is configured)
brew tap ChrisGVE/workspace-qdrant-mcp
brew install workspace-qdrant-mcp
```

The formula downloads pre-built binaries and configures the daemon as a Homebrew service.

> **Note:** Homebrew tap publication is planned for a future release.

**apt (Debian/Ubuntu):**
```bash
# Planned for future release
# sudo add-apt-repository ppa:workspace-qdrant-mcp/stable
# sudo apt update
# sudo apt install workspace-qdrant-mcp
```

**winget (Windows):**
```powershell
# Planned for future release
# winget install workspace-qdrant-mcp
```

### Docker Deployment Options

Four Docker deployment modes are supported:

#### Mode 1: All-in-One (Development)

Bundled Qdrant for quick setup:

```yaml
# docker-compose.dev.yml
services:
  memexd:
    image: workspace-qdrant-mcp/daemon:latest
    volumes:
      - ~/.workspace-qdrant:/data
      - ~/projects:/projects:ro
    environment:
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant:v1.7.3
    volumes:
      - qdrant_data:/qdrant/storage
```

```bash
docker-compose -f docker/docker-compose.dev.yml up -d
```

#### Mode 2: Qdrant-External (Production)

Separate Qdrant container for isolation:

```yaml
# docker-compose.prod.yml
services:
  memexd:
    image: workspace-qdrant-mcp/daemon:latest
    environment:
      - QDRANT_URL=http://qdrant:6333
      - QDRANT_API_KEY=${QDRANT_API_KEY}
```

#### Mode 3: Qdrant-Local

Use host-installed Qdrant:

```bash
# Install Qdrant on host
docker run -p 6333:6333 qdrant/qdrant:v1.7.3

# Run daemon pointing to host
docker run -e QDRANT_URL=http://host.docker.internal:6333 workspace-qdrant-mcp/daemon
```

#### Mode 4: Qdrant-Cloud

Connect to managed Qdrant Cloud:

```bash
# Set environment variables
export QDRANT_URL=https://your-cluster.qdrant.io:6333
export QDRANT_API_KEY=your-api-key

# Run daemon
docker run -e QDRANT_URL -e QDRANT_API_KEY workspace-qdrant-mcp/daemon
```

**Docker Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | (none) | Qdrant API key |
| `WQM_DATABASE_PATH` | `~/.workspace-qdrant/state.db` | SQLite database path |
| `WQM_LOG_LEVEL` | `INFO` | Log level (DEBUG, INFO, WARN, ERROR) |
| `WQM_GRPC_PORT` | `50051` | gRPC server port |

#### Docker Images

Docker images are published to GitHub Container Registry on each release:

```bash
# Pull latest version
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:latest

# Pull specific version
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0.4.0
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0.4
docker pull ghcr.io/chrisgve/workspace-qdrant-mcp:0

# Run the daemon
docker run -d \
  -p 50051:50051 \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  -v ~/.workspace-qdrant:/data \
  ghcr.io/chrisgve/workspace-qdrant-mcp:latest
```

Available platforms: `linux/amd64`, `linux/arm64`

### Initial Configuration

#### First-Run Setup

On first run, the system initializes automatically:

1. **Config file generation:**
   ```bash
   wqm config init                 # Generate default config
   wqm config show                 # Display current config
   wqm config edit                 # Open config in editor
   ```

2. **Config file locations (search order):**
   - `~/.workspace-qdrant/config.yaml`
   - `~/.config/workspace-qdrant/config.yaml`
   - `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS)
   - `%APPDATA%\workspace-qdrant\config.yaml` (Windows)

3. **Qdrant connection validation:**
   ```bash
   wqm admin health               # Check Qdrant connectivity
   wqm admin collections          # List collections
   ```

4. **SQLite database initialization:**
   The daemon automatically creates `~/.workspace-qdrant/state.db` on first run with all required tables.

#### MCP Client Configuration

**Claude Desktop (`~/.config/claude/config.json`):**

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "npx",
      "args": ["workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Claude Code (`.mcp.json` in project root):**

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "npx",
      "args": ["workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Service Management

The daemon (`memexd`) runs as a background service for continuous file watching and processing.

#### CLI Commands

```bash
wqm service install              # Install service (platform-specific)
wqm service start                # Start daemon
wqm service stop                 # Stop daemon
wqm service restart              # Restart daemon
wqm service status               # Check daemon status
wqm service logs                 # View daemon logs
wqm service logs --follow        # Follow logs in real-time
```

#### macOS (launchd)

The daemon installs as a launchd user agent:

**Plist location:** `~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist`

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.workspace-qdrant.memexd</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/memexd</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/Library/Logs/workspace-qdrant/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>~/Library/Logs/workspace-qdrant/daemon.err</string>
</dict>
</plist>
```

**Manual control:**
```bash
launchctl load ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl unload ~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist
launchctl list | grep memexd
```

#### Linux (systemd)

The daemon installs as a systemd user service:

**Service file:** `~/.config/systemd/user/memexd.service`

```ini
[Unit]
Description=Workspace Qdrant Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/memexd
Restart=on-failure
RestartSec=5
Environment=QDRANT_URL=http://localhost:6333

# Logging: stdout/stderr go to journald
StandardOutput=journal
StandardError=journal
SyslogIdentifier=memexd

# State directory for file-based logs
StateDirectory=workspace-qdrant

[Install]
WantedBy=default.target
```

**Manual control:**
```bash
systemctl --user daemon-reload
systemctl --user enable memexd
systemctl --user start memexd
systemctl --user status memexd

# View logs via journalctl
journalctl --user -u memexd -f

# Or via file (daemon writes to both)
tail -f ~/.local/state/workspace-qdrant/logs/daemon.jsonl
```

#### Windows

Windows service support is available via `wqm service` commands:

```powershell
# Install as Windows service (requires Administrator)
wqm service install

# Start/stop/status
wqm service start
wqm service stop
wqm service status

# View logs
wqm service logs --lines 50

# Uninstall service
wqm service uninstall
```

**Manual startup (without service):**
```powershell
# Run in foreground
memexd.exe --foreground

# Run as background process
Start-Process -NoNewWindow memexd.exe
```

> **Note:** Windows service management requires Administrator privileges. The service runs as LocalSystem by default.

#### Health Checks

```bash
# Quick health check
wqm admin health

# Detailed diagnostics
wqm admin health --verbose

# Check specific components
wqm admin health --component qdrant
wqm admin health --component daemon
wqm admin health --component database
```

**Health check output:**
```
Component       Status    Latency   Details
─────────────────────────────────────────────
Qdrant          healthy   12ms      v1.7.3, 4 collections
Daemon          healthy   2ms       pid=12345, uptime=2h15m
Database        healthy   1ms       23 watch folders, 156 queue items
```

### Logging and Observability

The daemon produces structured logs using the `tracing` crate with JSON output for machine parsing. Logs follow OS-canonical paths and integrate with platform service managers.

#### Canonical Log Paths

Log files follow platform-specific conventions:

| OS | Log Directory | Environment Override |
|----|---------------|---------------------|
| **Linux** | `$XDG_STATE_HOME/workspace-qdrant/logs/` (default: `~/.local/state/workspace-qdrant/logs/`) | `WQM_LOG_DIR` |
| **macOS** | `~/Library/Logs/workspace-qdrant/` | `WQM_LOG_DIR` |
| **Windows** | `%LOCALAPPDATA%\workspace-qdrant\logs\` | `WQM_LOG_DIR` |

**Log files:**

| File | Component | Format | Description |
|------|-----------|--------|-------------|
| `daemon.jsonl` | Rust daemon | JSON Lines | Primary daemon structured logs |
| `mcp-server.jsonl` | MCP Server | JSON Lines | TypeScript MCP server logs |

**Note:** Daemon and MCP Server logs are kept in **separate files** to prevent corruption if one component crashes while writing. The CLI merges them for unified viewing.

**Important:** The `~/.workspace-qdrant/` directory is reserved for **configuration only**, not logs. On Linux, if `$XDG_CONFIG_HOME` is set, configuration moves to `$XDG_CONFIG_HOME/workspace-qdrant/`.

#### Service Manager Integration

When running as a managed service, logs are captured by the platform service manager in addition to file output:

**Linux (systemd):**
```bash
# Primary access via journalctl
journalctl --user -u memexd -f              # Follow logs
journalctl --user -u memexd -n 100          # Last 100 entries
journalctl --user -u memexd --since "1 hour ago" --output=json

# File-based logs also available
cat ~/.local/state/workspace-qdrant/logs/daemon.jsonl | jq .
```

**macOS (launchctl):**
```bash
# LaunchAgent captures stdout/stderr to plist-specified paths
tail -f ~/Library/Logs/workspace-qdrant/daemon.log

# macOS unified log (limited - mainly for crashes)
log show --predicate 'process == "memexd"' --last 1h
```

**Windows:**
```powershell
# File-based logs
Get-Content "$env:LOCALAPPDATA\workspace-qdrant\logs\daemon.jsonl" -Tail 100

# Windows Event Log (for service events)
Get-EventLog -LogName Application -Source memexd -Newest 50
```

#### Log Format (JSON Lines)

Each log entry is a single JSON object:

```json
{"timestamp":"2026-02-05T10:30:45.123Z","level":"INFO","target":"memexd::processing","message":"Document processed","fields":{"document_id":"abc123","duration_ms":45.2,"collection":"projects","tenant_id":"proj_xyz"}}
```

**Standard fields:**

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | ISO 8601 | UTC timestamp |
| `level` | string | TRACE, DEBUG, INFO, WARN, ERROR |
| `target` | string | Rust module path |
| `message` | string | Human-readable message |
| `fields` | object | Structured context (varies by log type) |
| `span` | object | Active tracing span (if any) |

#### MCP Server Logging

The MCP Server uses `pino` for structured JSON logging directly to file. **No stderr output** is used to avoid potential future MCP protocol conflicts.

**Why file-only (no stderr):**

- MCP stdio transport uses stdout for protocol messages
- stderr could be used by future MCP protocol extensions
- File logging is reliable and doesn't interfere with any transport

**MCP Server log format:**

```json
{"level":30,"time":1707134445123,"pid":12345,"hostname":"workstation","name":"mcp-server","msg":"Tool called","session_id":"abc123","tool":"search","duration_ms":45}
```

**MCP Server log fields:**

| Field | Type | Description |
|-------|------|-------------|
| `level` | number | pino level (10=trace, 20=debug, 30=info, 40=warn, 50=error) |
| `time` | number | Unix timestamp (milliseconds) |
| `name` | string | Always `"mcp-server"` |
| `msg` | string | Log message |
| `session_id` | string | MCP session identifier (for correlation) |

**TypeScript implementation:**

```typescript
import pino from 'pino';
import { getLogDirectory } from './utils/paths';

const logger = pino({
  name: 'mcp-server',
  level: process.env.WQM_LOG_LEVEL || 'info',
  transport: {
    target: 'pino/file',
    options: { destination: `${getLogDirectory()}/mcp-server.jsonl` }
  }
});

// Usage
logger.info({ session_id, tool: 'search', duration_ms: 45 }, 'Tool called');
```

**Log rotation:** MCP Server logs follow the same rotation settings as daemon logs.

#### CLI Log Access

The CLI provides unified log access across all platforms, merging daemon and MCP server logs:

```bash
wqm debug logs                       # Show recent logs from all components
wqm debug logs -n 100                # Last 100 entries
wqm debug logs --follow              # Follow in real-time (both files)
wqm debug logs --errors-only         # Filter to WARN and ERROR
wqm debug logs --json                # Output raw JSON (for piping to jq)
wqm debug logs --since "1 hour ago"

# Component filtering
wqm debug logs --component daemon      # Daemon logs only
wqm debug logs --component mcp-server  # MCP Server logs only
wqm debug logs --component all         # Both (default)

# Correlation
wqm debug logs --session <session_id>  # Filter by MCP session
```

**Log merging behavior:**

- CLI reads both `daemon.jsonl` and `mcp-server.jsonl`
- Entries are merged and sorted by timestamp
- Component origin is indicated in output (unless `--json`)

**Log source priority:**

1. Canonical log files (daemon.jsonl, mcp-server.jsonl)
2. Service manager logs (journalctl on Linux) - daemon only
3. Fallback locations (for backwards compatibility)

#### Log Rotation

Log rotation is handled by the daemon:

| Setting | Default | Description |
|---------|---------|-------------|
| `max_log_size` | 50 MB | Rotate when file exceeds this size |
| `max_log_files` | 5 | Number of rotated files to keep |
| `compress_rotated` | true | Gzip rotated files |

**Rotated file naming:** `daemon.jsonl.1`, `daemon.jsonl.2.gz`, etc.

#### OpenTelemetry Integration (Optional)

For production deployments with distributed tracing infrastructure:

```bash
# Enable OTLP export
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317
export OTEL_SERVICE_NAME=memexd
export OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampling

# Traces are exported to the configured backend
# View in Jaeger, Zipkin, Grafana Tempo, etc.
```

**Note:** OpenTelemetry is for distributed tracing correlation, not log viewing. It requires external infrastructure and is optional.

#### Environment Variables

**Shared (Daemon and MCP Server):**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_LOG_DIR` | (OS-canonical) | Override log directory for both components |
| `WQM_LOG_LEVEL` | `info` | Minimum log level (trace, debug, info, warn, error) |

**Daemon-specific:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_LOG_JSON` | `true` | Enable JSON output (daemon) |
| `WQM_LOG_CONSOLE` | `false` (service) / `true` (foreground) | Console output (daemon) |
| `RUST_LOG` | - | Fine-grained module filtering (e.g., `memexd=debug,hyper=warn`) |

**MCP Server-specific:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WQM_MCP_LOG_LEVEL` | `WQM_LOG_LEVEL` | Override log level for MCP Server only |

**Note:** MCP Server does not support console output to avoid protocol interference.

---

### CI/CD and Release Process

#### Version Tagging

Semantic versioning with optional pre-release tags:

| Pattern | Example | Description |
|---------|---------|-------------|
| `vX.Y.Z` | `v0.4.1` | Stable release |
| `vX.Y.Z-rc.N` | `v0.5.0-rc.1` | Release candidate |
| `vX.Y.Z-beta.N` | `v0.5.0-beta.1` | Beta release |
| `vX.Y.Z-alpha.N` | `v0.5.0-alpha.1` | Alpha release |

#### Release Workflow

```
Tag Push (vX.Y.Z)
       │
       ▼
┌──────────────────┐
│  Build Matrix    │  6 platform builds in parallel
│  (GitHub Actions)│
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Test Suite      │  Unit tests, integration tests
│                  │  per platform
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Create Release  │  Draft release with
│                  │  changelog generation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Upload Assets   │  Binaries, checksums,
│                  │  documentation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Publish Release │  Make release public
└──────────────────┘
```

**GitHub Workflows:**

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push, PR | Build and test |
| `release.yml` | Tag push | Create release |
| `tree-sitter-version-bump.yml` | Schedule | Update tree-sitter grammars |
| `onnx-runtime-version-bump.yml` | Schedule | Update ONNX Runtime |

#### Automated Dependency Updates

- **Tree-sitter grammars:** Weekly check for grammar updates
- **ONNX Runtime:** Monthly check for new versions
- **Rust dependencies:** Dependabot PRs for security updates

### Upgrade and Migration

#### Release Packaging (Placeholder)

Define per-OS release packaging and installer expectations, and how the CLI performs updates.

- **macOS**: Specify installer format (e.g., `.pkg` or `.dmg`) and signing/notarization requirements.
- **Windows**: Specify installer format (e.g., `.msi` or `.exe`) and code signing requirements.
- **Linux**: Specify package formats (e.g., `.deb`, `.rpm`, AppImage, or tarball) and repository strategy.
- **CLI-driven updates**: `wqm update` is the primary mechanism for installing updates across all OSes and should handle download, verification, install, and service restart steps.

> Placeholder: fill in exact packaging formats, install locations, and service integration per OS.

#### Standard Upgrade

```bash
# Stop the daemon
wqm service stop

# Download new version
curl -fsSL https://github.com/[org]/workspace-qdrant-mcp/releases/latest/download/wqm-$(uname -s)-$(uname -m).tar.gz | tar xz
sudo mv wqm /usr/local/bin/
sudo mv memexd /usr/local/bin/

# Start the daemon (auto-migrates database)
wqm service start

# Verify
wqm --version
wqm admin health
```

#### Database Migration

The daemon automatically applies database migrations on startup:

1. Reads `schema_version` table to determine current version
2. Applies pending migrations in order
3. Updates `schema_version` table

**Manual migration check:**
```bash
wqm admin db-version              # Show current schema version
wqm admin db-migrate --dry-run    # Preview pending migrations
```

#### Rollback Procedure

If an upgrade causes issues:

```bash
# Stop the daemon
wqm service stop

# Restore previous binaries (keep backups!)
sudo mv /usr/local/bin/wqm.backup /usr/local/bin/wqm
sudo mv /usr/local/bin/memexd.backup /usr/local/bin/memexd

# Start the daemon
wqm service start
```

> **Warning:** Database schema rollbacks are not supported. Create SQLite backups before upgrading.

#### Self-Update Command

The `wqm update` command provides in-place binary updates:

```bash
# Check for updates
wqm update check

# Update to latest stable version
wqm update

# Update to latest version in specific channel
wqm update --channel stable       # Default: stable releases only
wqm update --channel beta         # Include beta releases
wqm update --channel rc           # Include release candidates
wqm update --channel alpha        # Include alpha releases

# Update to specific version
wqm update --version v0.5.0

# Force reinstall current version
wqm update --force

# Install specific version with force
wqm update install --version v0.4.0 --force
```

**Update process:**
1. Fetches release info from GitHub API
2. Downloads platform-specific binary
3. Verifies SHA256 checksum
4. Stops running daemon
5. Replaces binary (with backup)
6. Restarts daemon

> **Note:** Updates require write permission to the installation directory. The daemon is automatically restarted after update.

### Troubleshooting

#### Common Issues

**Issue: Daemon won't start**

```bash
# Check if already running
pgrep memexd

# Check port availability
lsof -i :50051

# Check logs
wqm service logs --lines 50

# Run in foreground for debugging
memexd --foreground --log-level debug
```

**Issue: Qdrant connection failed**

```bash
# Test Qdrant connectivity
curl http://localhost:6333/health

# Check environment
echo $QDRANT_URL

# Verify config
wqm config show | grep qdrant
```

**Issue: File changes not detected**

```bash
# Check watch folders
wqm watch list

# Verify file system events
wqm admin debug --watch-events /path/to/project

# Check inotify limits (Linux)
cat /proc/sys/fs/inotify/max_user_watches
# Increase if needed:
echo 524288 | sudo tee /proc/sys/fs/inotify/max_user_watches
```

**Issue: High memory usage**

```bash
# Check queue size
wqm queue stats

# Clear old queue items
wqm queue clean --days 7

# Reduce batch size in config
wqm config set daemon.queue_batch_size 5
```

#### Diagnostic Commands

```bash
# Full system diagnostic
wqm admin diagnose

# Export diagnostic report
wqm admin diagnose --output report.json

# Check specific subsystems
wqm admin diagnose --component file-watching
wqm admin diagnose --component embedding
wqm admin diagnose --component grpc
```

---

## Future Development (Wishlist, Not Yet Scoped)

This section documents research findings and architectural ideas that may be pursued in future development cycles. These items are exploratory and have not been scoped into the project plan.

### Graph RAG (Knowledge Graph-Enhanced Retrieval)

**What it is:** Graph RAG augments traditional vector search with knowledge graph traversal, enabling relationship-aware retrieval that understands structural connections between code entities (function calls, imports, type hierarchies, module dependencies).

**Measured benefits (external benchmarks):**

- Lettria (2024): 20-25% accuracy improvement over vector-only RAG on relational queries
- Neo4j benchmark (2024): 3.4x accuracy improvement for schema-heavy and relationship-dependent queries
- Microsoft GraphRAG (2024): Significant improvement on "global" questions requiring synthesis across documents
- Matt Ambrogi retrieval study: Smaller, focused chunks (128 tokens) achieved 3x better MRR than larger chunks (256 tokens)

**Where it adds value in this project:**

- Cross-file navigation: "What functions call this method?" or "What imports this module?"
- Impact analysis: "What would break if I change this interface?"
- Architectural understanding: "Show me the dependency chain from this entry point"
- Cross-language boundaries: Connecting TypeScript MCP server to Rust daemon via gRPC definitions
- Multi-product relationships: CLI ↔ daemon SQLite schema sharing, MCP ↔ daemon gRPC communication

**Existing building blocks already in the codebase:**

- Tree-sitter semantic chunking extracts functions, classes, methods, structs, traits, enums, and their hierarchical relationships (`parent_symbol`)
- LSP integration provides resolved references, type information, and cross-file relationships
- SQLite infrastructure is already in place (`state.db` with WAL mode, ACID guarantees)
- Hybrid search (dense + sparse + RRF) provides the vector retrieval foundation
- `tracked_files` and `qdrant_chunks` tables already track file-to-chunk relationships

**Graph database evaluation (2026-02-09, updated 2026-02-10):**

| DB | License | Embeddable | Query Language | Multi-process R/W | Verdict |
|----|---------|------------|----------------|-------------------|---------|
| Kuzu | MIT | Yes (Rust + Node.js) | Full Cypher | No concurrent R+W | ~~Best graph engine~~ **ARCHIVED Oct 2025** — see note below |
| LadybugDB | MIT | Yes (Kuzu fork) | Cypher (inherited) | TBD | Fork of Kuzu, carries legacy but under new maintenance |
| HelixDB | AGPL-3.0 | Rust-native | HelixQL (proprietary) | TBD | Graph+vector in one, YC-backed, active dev — AGPL license concern |
| SQLite adjacency | Public domain | Yes | SQL + recursive CTE | WAL: 1 writer + N readers | Viable fallback, but 3+ hop queries get expensive |
| Neo4j Community | GPLv3 + Commons Clause | No (separate JVM) | Cypher | Yes | Too heavy, licensing friction (GPLv3 + Commons Clause) |
| SurrealDB | Apache 2.0 | Yes (Rust) | SurrealQL | Yes | Multi-model, vector HNSW persistence incomplete |
| Oxigraph | Apache 2.0/MIT | Yes (Rust) | SPARQL (RDF only) | N/A | Wrong data model (RDF triples, not property graphs) |

**Kuzu archived (2026-02-10 update):** The [Kuzu GitHub repository was archived on Oct 10, 2025](https://github.com/kuzudb/kuzu). The maintainers stated they are "working on something new" with no further details. No new releases or bug fixes are expected. Existing releases remain usable but the project is effectively abandoned. All references to Kuzu in this spec are retained for historical context but should be considered superseded by the alternatives below when Graph RAG work begins.

**Alternatives to evaluate when Graph RAG is scoped:**
- **[LadybugDB](https://github.com/ladybugdb/ladybugdb)**: Fork of Kuzu under new maintenance. Inherits Kuzu's MIT license, Cypher support, and embeddable architecture. Carries Kuzu's codebase legacy (both strengths and technical debt). Maturity and long-term commitment to be assessed.
- **[HelixDB](https://github.com/helixdb/helix-db)**: Rust-native graph+vector database with built-in embeddings. Active development (YC-backed, 165+ releases). Uses proprietary HelixQL query language (not Cypher). **AGPL-3.0 license** is a concern for our MIT-licensed project — would require careful evaluation of linking/embedding implications vs. using as a separate service.
- **SQLite recursive CTEs**: Already available, no new dependency. Sufficient for shallow graph queries (1-2 hops) but expensive for deep traversals and graph algorithms.

**Qdrant confirmed** as the right choice for vector search. No multi-model database matches Qdrant's vector performance. The graph database selection is deferred until Graph RAG work is scoped — the dedicated daemon architecture (`graphd` with gRPC) remains valid regardless of which graph engine is chosen.

**Recommended approach: Dedicated graph daemon (graphd)**

A separate Rust daemon (`graphd`) owns the graph database exclusively, serving both write and query operations via gRPC. This resolves concurrency limitations while providing full query capabilities at query time.

```
MCP Server (TypeScript)
    ├── Vector search    → Qdrant (direct HTTP/gRPC)
    ├── Graph queries    → graphd (gRPC)
    └── State/metadata   → SQLite state.db (direct read-only)

memexd (Rust daemon)
    ├── File watching    → filesystem (notify)
    ├── Embedding        → ONNX Runtime (all-MiniLM-L6-v2)
    ├── Vector writes    → Qdrant (direct)
    ├── Graph edges      → graphd (gRPC)
    └── State writes     → SQLite state.db (WAL mode)

graphd (Rust daemon)
    ├── Graph storage    → Graph database (exclusive access; engine TBD — see evaluation table)
    ├── Write API        → gRPC (receives edges from memexd)
    ├── Query API        → gRPC (serves MCP server, CLI, memexd)
    └── Analytics        → Graph query language (community detection, centrality, impact analysis)

CLI (Rust)
    ├── Vector queries   → Qdrant (direct)
    ├── Graph queries    → graphd (gRPC)
    └── State queries    → SQLite state.db (direct read-only)
```

**Why a separate daemon, not embedded:**
- Eliminates graph engine concurrency limitations (single process = single owner)
- Full Cypher available at query time (multi-hop traversal, path finding, graph algorithms)
- Follows the established pattern (same launchd management, gRPC integration, binary distribution as memexd)
- Future-proof: graph engine can be swapped behind the gRPC interface without consumer changes
- Clean separation of concerns: each daemon does one thing well

**Graph daemon storage:** `~/.workspace-qdrant/graph/` (graph database directory)

**Graph daemon management:** Same launchd plist pattern as memexd. Single-threaded with internal queue for write operations. Read queries served directly. Managed via CLI (`wqm service install --graph`, `wqm service start --graph`).

**Query pattern:**

1. Vector search finds semantically relevant chunks (Qdrant)
2. Graph traversal expands results to structurally related code (graphd, full Cypher — no hop limit)
3. Results re-ranked combining semantic similarity + graph proximity
4. Advanced analytics (community detection, centrality, impact analysis) available as direct Cypher queries

**Graph schema (property graph, engine-agnostic):**

Node types: `File`, `Function`, `Class`, `Method`, `Struct`, `Trait`, `Module`, `Document`

Edge types: `CALLS`, `IMPORTS`, `EXTENDS`, `IMPLEMENTS`, `USES_TYPE`, `CONTAINS`, `MENTIONS`

Each node and edge carries a `tenant_id` property for project isolation. Cross-tenant queries are explicitly forbidden.

### Cross-Project Search

**Current model:** Projects are isolated by `tenant_id` in the `projects` collection. Search is project-scoped.

**Revised model: Tiered search with automated grouping**

Search should support three scopes, selectable per query:
1. **Project scope** (default): Search within the current project's `tenant_id`
2. **Group scope**: Search within an automatically detected project group
3. **All projects scope**: Search across the entire `projects` collection (no tenant filter)

**Automated project grouping (no manual configuration required):**
- Shared dependencies: Projects using the same libraries (parsed from Cargo.toml, package.json, requirements.txt)
- Git organization: Projects under the same GitHub org/user
- Explicit cross-references: Projects linked by gRPC proto imports, shared crate dependencies, or workspace membership
- Embedding similarity: Projects whose README/description embeddings cluster together

**Relevance ranking across projects:** When searching beyond the current project, results from the current project receive full weight. Results from group projects receive a decay factor (e.g., 0.7). Results from unrelated projects receive a further decay (e.g., 0.4). This preserves signal-to-noise while enabling cross-project discovery.

**Graph RAG across projects:** Cross-project graph edges (e.g., shared dependency usage patterns, similar function signatures) enable structural code reuse discovery that vector search alone misses. GitHub research shows ~5% of code across repositories is cross-project clones, concentrated within similar domains.

### Project vs. Library Boundary

**Observation:** Not all files in a project folder are "project code." Research papers, development notes, experiment results, and reference PDFs are background knowledge that supports the project but clutters code search results.

**Revised approach — format-based routing (deterministic, no content classification):**
- **Developer-created text files** (`.md`, `.txt`, `.rst`, source code, configs, specs, tests): Stay in `projects` collection with project `tenant_id`. This includes experiment notes, PoC write-ups, and development logs — these are project context, not parasites. Vector search naturally ranks code higher for code queries and notes higher for design rationale queries.
- **Binary/downloaded reference files** (`.pdf`, `.epub`, `.djvu`, `.docx`): Route to `libraries` collection with a `source_project_id` metadata field linking them back to their project. These are external reference material, not developer-created project artifacts.

**Rationale for keeping .md experiments in projects:** Experiment notes, PoC results, and development logs answer questions like "why did we choose this algorithm?" and "what did we try that didn't work?" — genuinely valuable project context. The occasional noise in code search is far less costly than the complexity of content-based classification to sort .md files. The hybrid search system handles relevance naturally: code-related queries rank code chunks higher; design rationale queries rank notes higher.

**Configurable edge cases:** `.docx` files could exist in either context (a spec written in Word vs. a downloaded paper). The default routes `.docx` to libraries, but users can override via configuration. The routing rule is extension-based and lives in the watching configuration.

**Cross-collection search:** Qdrant does not support native cross-collection queries. Implementation requires issuing parallel queries to both collections and merging results via RRF. The MCP `search` tool would accept an optional `include_libraries: true` parameter (default false for code queries, true for general knowledge queries).

### Library Collection Tenancy (Revised)

**Current model:** Libraries isolated by `library_name`. Each library is a separate tenant.

**Revised model: Hierarchical naming + cross-library search by default**

**Hierarchical tenant naming** derived from the filesystem structure relative to the watched folder root:
```
library_name.relative_path_segments.document_name
```
Example: Library root "main" containing `computer_science/design_patterns/Gang_of_Four.pdf`:
- `library_name`: "main"
- `library_path`: "computer_science/design_patterns"
- `document_name`: "Gang_of_Four.pdf"
- Full tenant: "main.computer_science.design_patterns.Gang_of_Four"

**Search scoping via prefix matching:**
- `library_name = "main"` → search entire main library
- `library_name = "main" AND library_path LIKE "computer_science%"` → search CS subdomain
- No filter → search all libraries (default for knowledge queries)
- `source_project_id = "<project>"` → search project-associated references

**Cross-library search is the default.** The `library_name` and `library_path` fields are metadata for management (deletion, updates) and optional scoping, not mandatory isolation. Knowledge synthesis across complementary sources (e.g., multiple physics textbooks) happens naturally when search is unfiltered.

### Automated Tagging and Grouping

**Problem:** Manual tagging creates mental overhead and inconsistent coverage. Tags should be generated automatically from available signals.

**Core principle: per-chunk aggregation.** Tags are not derived from a document summary or its first N tokens. Instead, each chunk (without overlap) is independently tagged, then tags are aggregated across all chunks by frequency. Tags appearing in the highest fraction of chunks represent the document's dominant topics. This naturally weights tags by coverage — a QM textbook mentioning "Schrödinger equation" in 40% of chunks gets a strong tag; "classical limit" in 3% gets a weak tag. A configurable `min_frequency` threshold (e.g., 10% of chunks) filters noise.

**Concept normalization:** Raw implementation-level names (library names, API names) must be mapped to concept-level tags. `regex` (Rust crate), `re` (Python stdlib), `pcre` (C library), and `oniguruma` (Ruby) all map to the concept `regular-expressions`. Without normalization, two projects using the same concept in different languages appear unrelated.

Normalization sources (in order of preference):
1. **Package registry categories**: crates.io categories (`text-processing`, `parsing`), npm keywords, PyPI classifiers. These are already concept-level and maintained by the package ecosystem.
2. **Curated concept dictionary**: A mapping from common library names to concepts, generated once by an LLM and stored as a configuration file (`assets/concept_normalization.yaml`). One-time cost, periodically refreshed.
3. **Embedding-based semantic grouping**: Embed dependency descriptions (from registry metadata). Libraries with similar descriptions cluster into shared concept tags.

**Automated tagging pipeline:**

**Tier 1 — Zero-cost heuristics (always active, no ML):**
1. **Path-derived tags**: Parse directory names and filenames into normalized topic tags. `computer_science/design_patterns/` → tags: `computer-science`, `design-patterns`. Most reliable signal because directory names tend to be conceptual.
2. **PDF metadata extraction**: Extract title, author, subject, keywords from PDF document metadata fields. Available in most academic papers and textbooks without content analysis.
3. **Dependency-derived concepts** (projects only): Parse dependency files (Cargo.toml, package.json, requirements.txt), normalize via concept dictionary/registry categories. Produces project-level topic tags.

**Tier 2 — Embedding-based (uses existing model, no LLM):**
4. **TF-IDF keyword extraction per chunk**: Extract top distinctive terms from each chunk, aggregate by frequency across all chunks. Filtered against stop-word list. Produces content-level topic tags.
5. **Embedding-based clustering**: Group documents by embedding similarity (embeddings already generated during ingestion). Cluster labels derived from distinctive terms within each cluster.
6. **Zero-shot classification**: Compare document embedding against embeddings of a predefined topic taxonomy (~100 terms). If cosine similarity exceeds threshold, apply the tag. Uses existing embedding model.

**Tier 3 — LLM-assisted (optional, configurable, independent from MCP session):**
7. **LLM-based chunk tagging**: For each chunk, ask a configured LLM for 3-5 topic tags. Aggregate across chunks by frequency. Top-N by frequency = document tags.

**Tier 3 LLM configuration:** The tagging LLM is independent from the MCP-connected LLM session. Tagging happens during daemon ingestion, which may run when no LLM session is active. The model is configurable:

```yaml
# In default_configuration.yaml
tagging:
  auto_tag: true
  min_frequency: 0.1              # Tag must appear in ≥10% of chunks
  top_n_tags: 10                  # Maximum tags per document
  tier3_enabled: false            # LLM tagging off by default
  tier3_provider: "anthropic"     # anthropic, openai, ollama, none
  tier3_model: "claude-haiku"     # Cheapest appropriate model
  tier3_api_key_env: "ANTHROPIC_API_KEY"
  tier3_ollama_url: "http://localhost:11434"  # For local models
```

Supported providers (prefer subscription/local models over per-call API charges):
- **Ollama** (local models — zero marginal cost, no network dependency, recommended for users with hardware)
- **Subscription-based platforms** (e.g., Abacus.ai, or subscription plans that include API access)
- **Anthropic** (Haiku for cost efficiency, only when API calls are acceptable)
- **OpenAI** (GPT-4o-mini or similar low-cost model, only when API calls are acceptable)
- **None** (Tiers 1-2 only — default)

**Tag inheritance:** Tags propagate to contained documents. A folder tagged `physics` automatically tags all documents within it. Document-level tags extend folder-level tags.

**Applicability to projects:** The same pipeline applies to project grouping. Dependency analysis with concept normalization provides strong domain signals. README content embedding provides semantic classification. Two projects with different dependency names but the same concepts (e.g., `regex` and `re`) are correctly grouped.

**Tag evolution — lifecycle tied to ingestion pipeline:**

Tags are dynamic and must evolve as content changes. Tag updates piggyback on the existing file change processing pipeline — no separate watcher or queue needed.

| Event | Tag action |
|-------|-----------|
| File created | Chunks tagged (Tiers 1-2) → aggregate by frequency → store document-level tags |
| File modified | Chunks re-generated → re-tagged → document tags recomputed from new frequencies → old zero-frequency tags removed, new tags added |
| File deleted | Document tags removed → cluster memberships recalculated if clustering active |
| Dependency file changed | Concept tags re-derived from updated dependencies → project-level tags updated |
| Folder renamed | Path-derived tags re-derived from new path |

**Tag storage (dual):**
- **Qdrant payload**: Each point carries chunk-level tags (enables tag-filtered vector search)
- **SQLite `tracked_files`**: Document-level tag summary with frequencies (enables browsing, management, cross-document tag analysis)

**Tag drift:** A project that starts as `data-processing` and evolves toward `machine-learning` naturally reflects this because tags are re-derived from content on every re-ingestion, never manually pinned. Frequency-based aggregation means dominant topics surface automatically as the codebase evolves.

#### Automated Affinity Grouping

**Goal:** Automatically group related projects without user intervention, producing both cluster membership and human-readable group labels.

**Embedding-based affinity pipeline (LLM-free):**

1. **Per-project aggregate embedding**: Average all chunk embeddings for a project into a single vector. Chunk embeddings already exist from ingestion — no additional embedding cost.
2. **Pairwise cosine similarity**: Compare aggregate embeddings between projects. Projects above a configurable threshold (e.g., 0.7) form an affinity group.
3. **Group labeling via taxonomy matching**: Compare the group's centroid embedding against a predefined taxonomy (see below). Top-N matching taxonomy terms become the group's label.

This pipeline is LLM-free, uses only the existing FastEmbed model, and runs as a background daemon task after ingestion.

**Taxonomy source — package registry categories:**

The taxonomy is sourced from community-curated package registry categories, not manually defined:
- **crates.io**: ~70 categories (`algorithms`, `authentication`, `command-line-interface`, `concurrency`, `cryptography`, `database`, `network-programming`, `text-processing`, `web-programming`, etc.)
- **npm**: similar keyword ecosystem
- **PyPI**: detailed classifiers (`Topic :: Scientific/Engineering :: Artificial Intelligence`, etc.)

Combined and deduplicated, these provide ~150-200 concept-level terms. They are embedded once at daemon startup using FastEmbed and cached.

**Zero-shot taxonomy matching:**

For each document or project, compare its aggregate embedding against all taxonomy embeddings via cosine similarity. Top-N matches above threshold become tags. This replaces manual tagging entirely for code projects.

**Open questions and concerns (to be validated empirically):**

1. **Embedding dimensionality mismatch**: 384-dim MiniLM embeddings of short taxonomy phrases (e.g., "cryptography") may not produce reliable cosine similarity against averaged code chunk embeddings. The semantic spaces may be too different. Empirical testing required.
2. **Tier 1 heuristic quality**: Path-derived tags are unreliable (directory names are often structural, not conceptual — `src/`, `lib/`, `utils/`). Dependency-derived concepts require a concept dictionary to map library names to concepts (e.g., `tokio` → `async-runtime`, `serde` → `serialization`). Without this mapping, raw dependency names are not useful as tags. PDF/EPUB metadata is valuable when present but not universally available.
3. **TF-IDF produces terms, not concepts**: TF-IDF (Term Frequency – Inverse Document Frequency) extracts distinctive keywords by scoring words that are frequent in a specific chunk but rare across all documents. For example, in an async runtime library, `async`, `executor`, `spawn` score high while `the`, `function`, `return` are filtered out. However, TF-IDF produces raw terms (`tokio`, `serde`, `reqwest`), not concept-level tags (`async-runtime`, `serialization`, `http-client`). A concept normalization step is still needed.
4. **Concept labeling gap**: The fundamental challenge is turning raw signals (keywords, library names, embedding clusters) into meaningful human-readable concept tags. Without either an LLM or a curated mapping, embedding clustering gives groups but cannot name them. The registry-based taxonomy approach closes this gap for code projects but may be insufficient for non-code content (documents, research papers).
5. **Fallback strategy**: If zero-shot taxonomy matching proves too noisy, the fallback is TF-IDF keywords matched against the `concept_normalization.yaml` dictionary (one-time LLM cost to generate the mapping, then static and periodically refreshed).

**Implementation plan:**

Phase 1 (current): Implement zero-shot taxonomy matching using registry categories as taxonomy source. Embed taxonomy terms at startup, compare against document/project aggregate embeddings during ingestion. Evaluate quality empirically.

Phase 2 (if Phase 1 insufficient): Add TF-IDF keyword extraction + concept dictionary mapping. Generate `concept_normalization.yaml` once using LLM, store as static config.

Phase 3 (optional): Enable Tier 3 LLM-assisted tagging for users who want higher quality and have API access configured.

### Knowledge Overlap and Complementary Sources

**Challenge:** Multiple sources covering the same topic (e.g., physics textbooks with math prerequisite chapters) overlap and complement each other. The system should synthesize across sources rather than treating each in isolation.

**Approach:**
- **Cross-library search by default** ensures overlapping content from different sources is surfaced together
- **Provenance metadata** (library_name, library_path, document_name) returned with every search result enables the consumer to assess source diversity and reliability
- **Source diversity in results**: When multiple chunks from different sources match a query, prefer diverse results (1 chunk from each of 3 books) over concentrated results (3 chunks from 1 book). This is a post-retrieval re-ranking step.
- **Contradiction handling**: Not automatically resolved. Provenance metadata enables the LLM consumer to reason about conflicting sources. Academic research confirms this remains an unsolved problem for automated systems.

### Graph Daemon (graphd) — Graph Database Service

**Note (2026-02-10):** This section was originally written for Kuzu, which has since been [archived (Oct 2025)](https://github.com/kuzudb/kuzu). The dedicated daemon architecture described below remains valid regardless of which graph engine is chosen. When Graph RAG work is scoped, evaluate LadybugDB (Kuzu fork, MIT), HelixDB (Rust-native, AGPL-3.0), or SQLite CTEs as the backing engine. See the [Graph RAG evaluation table](#graph-rag-knowledge-graph-enhanced-retrieval) for details.

**Original evaluation:** Kuzu (MIT, embeddable, Cypher, 188x faster than Neo4j for analytical queries) was the strongest candidate for graph storage and analytics with official Rust and Node.js bindings.

**Concurrency constraint and daemon solution:** The dedicated daemon pattern (`graphd`) remains the recommended architecture regardless of graph engine choice. A separate Rust daemon owns the graph database exclusively, serving both write and query operations via gRPC. This resolves concurrency limitations while providing full query capabilities at query time — same architectural pattern as memexd with Qdrant.

**graphd responsibilities:**
- Own the graph database at `~/.workspace-qdrant/graph/` (exclusive access)
- Accept graph edge writes from memexd via gRPC (during ingestion)
- Serve graph queries from MCP server, CLI, and memexd via gRPC (full Cypher)
- Run graph analytics (community detection, centrality, impact analysis) on demand or as background tasks
- Internal write queue for serialized operations; read queries served directly
- Managed via launchd (same pattern as memexd): `~/Library/LaunchAgents/com.workspace-qdrant.graphd.plist`

**gRPC service definition (conceptual):**
```protobuf
service GraphService {
    // Write operations (from memexd during ingestion)
    rpc AddEdge(AddEdgeRequest) returns (AddEdgeResponse);
    rpc AddNode(AddNodeRequest) returns (AddNodeResponse);
    rpc RemoveFileEdges(RemoveFileEdgesRequest) returns (RemoveFileEdgesResponse);

    // Query operations (from MCP server, CLI)
    rpc QueryCypher(CypherRequest) returns (CypherResponse);
    rpc GetRelated(GetRelatedRequest) returns (GetRelatedResponse);
    rpc GetCallChain(CallChainRequest) returns (CallChainResponse);
    rpc GetImpactAnalysis(ImpactRequest) returns (ImpactResponse);

    // Analytics (from CLI or daemon batch)
    rpc RunCommunityDetection(AnalyticsRequest) returns (AnalyticsResponse);
    rpc RunCentralityAnalysis(AnalyticsRequest) returns (AnalyticsResponse);

    // Health
    rpc Health(HealthRequest) returns (HealthResponse);
}
```

**Graph evolution — delete/re-ingest pattern (same as Qdrant):**

Graph data follows the same lifecycle as vector data. When a file changes, old graph edges are deleted and new ones are extracted and inserted.

| Event | Graph action |
|-------|-------------|
| File created | Tree-sitter extracts symbols (nodes) → LSP resolves references (edges) → sent to graphd |
| File modified | Delete edges WHERE `source_file = modified_file` → re-extract → re-insert. Nodes updated in place (symbol identity persists even if signature changes) |
| File deleted | Delete edges WHERE `source_file = deleted_file` → orphan node cleanup (nodes with no remaining edges pruned) |
| Project deleted | Delete all nodes/edges WHERE `tenant_id = project_id` |

**Node vs. edge ownership:**
- **Edges** are owned by the source file. When `bar.rs` changes, edges *from* symbols in `bar.rs` are deleted and re-created. Edges *to* symbols in `bar.rs` from other files remain untouched — they update when their source files are re-processed.
- **Nodes** represent symbols and are updated in place. A function may change its signature but keeps its graph identity. Orphaned nodes (no remaining edges) are pruned during cleanup.

**Pipeline integration — additions to existing ingestion flow:**
```
File change detected (file watcher)
    → Queue item created (unified_queue)
        → memexd processes:
            1. Delete old Qdrant chunks for file       (existing)
            2. Delete old graph edges for file          (NEW → gRPC to graphd)
            3. Re-chunk (tree-sitter or fixed-size)     (existing)
            4. Re-embed chunks                          (existing)
            5. Extract relationships from AST/LSP       (NEW)
            6. Upsert chunks to Qdrant                  (existing)
            7. Send edges to graphd                     (NEW → gRPC)
            8. Update tags (per-chunk, aggregate)       (NEW)
```

Steps 2, 5, 7, 8 are additions. The queue, file watching, debouncing, and processing loop remain unchanged.

**Eventual consistency:** During batch operations (e.g., `git checkout` changing many files), the graph may have stale edges briefly while files are processed through the queue. This is the same trade-off accepted with Qdrant — the queue guarantees all files are eventually processed.

**Future-proofing:** The gRPC interface is the stable contract. The graph engine behind `graphd` can be swapped (LadybugDB, HelixDB, SQLite CTEs, or any future candidate) without affecting consumers. This was the original intent even before Kuzu's archival — the daemon abstraction isolates the engine choice.

**CLI management:**
```bash
wqm service install --graph     # Install graphd launchd plist
wqm service start --graph       # Start graphd
wqm service status --graph      # Check graphd health
wqm graph query "MATCH (f:Function)-[:CALLS]->(g:Function) WHERE f.name = 'main' RETURN g"
wqm graph impact --symbol parse --file document_processor.rs
wqm graph communities           # List detected code communities
```

### Knowledge Manager Product Potential

The capabilities being built form the foundation for a general-purpose knowledge management system that extends well beyond code intelligence:

**Current capabilities (developer tool):**
- Automatic file watching with intelligent ingestion
- Hybrid semantic + keyword search with RRF fusion
- Multi-tenant knowledge organization (projects, libraries, memory)
- Semantic code intelligence (tree-sitter, LSP)
- MCP interface (any LLM client can use it)
- CLI for direct access and diagnostics

**Planned capabilities (knowledge base):**
- Graph relationship tracking (graphd — engine TBD, see evaluation table)
- Automated tagging and classification (Tiers 1-3)
- Cross-collection and cross-project search
- Content-type routing (code vs. reference material)
- Hierarchical library organization with cross-library synthesis
- Source diversity in results and provenance metadata

**Product evolution path:**
1. **Current**: Developer tool — code intelligence + reference libraries for individual developers
2. **Near-term**: Personal knowledge base — automated tagging, cross-library synthesis, background knowledge routing. The user's entire document collection (textbooks, research papers, notes, code) becomes a searchable, interconnected knowledge graph.
3. **Long-term**: Team knowledge manager — multi-user access control, shared libraries, collaborative tagging, organizational knowledge graph

**The MCP interface is the critical enabler.** It means the knowledge manager is accessible from any LLM-powered tool (Claude Desktop, Claude Code, Cursor, or any future MCP client) without building a custom UI. The LLM conversation itself becomes the interface. The underlying engine (memexd + graphd + Qdrant + SQLite) is domain-agnostic — it handles code, documents, research papers, and any text content through the same pipeline.

### Chunk Size Optimization Research

**Finding:** 384 characters with 15% overlap (58 characters) is optimal for the all-MiniLM-L6-v2 embedding model.

**Evidence:**

- Internal benchmark (2026-02-09, 500KB text): 384 chars achieved 86.16 ms/KB, 19% better throughput than the previous 512-char default
- Matt Ambrogi study: 128 tokens had 3x better MRR than 256 tokens for retrieval
- Chroma Research: 200-token chunks had 2x better precision than 400-token chunks
- all-MiniLM-L6-v2 is trained on 128-token sequences (256-token max); 384 chars ≈ 82 tokens (prose) or ≈ 110 tokens (code)
- NVIDIA research: 15% overlap is optimal for context preservation across chunk boundaries

**Status:** Defaults updated to 384 chars / 58 overlap in configuration and `ChunkingConfig`.

**Note:** These defaults apply to fixed-size text chunking only. Tree-sitter semantic chunking uses natural code boundaries (functions, classes, methods) and has its own size limits (`DEFAULT_MAX_CHUNK_SIZE = 8000` estimated tokens).

### Distance Matrix Visualization

Qdrant's Distance Matrix API can compute pairwise distances between points using the `dense` vector. This could power interactive code intelligence visualizations showing clusters of semantically related files, functions, or documentation. Could serve as a stepping stone toward full Graph RAG by revealing natural code clusters. See the [Qdrant Dashboard Visualization](#qdrant-dashboard-visualization) section for current capabilities.

---

## Related Documents

| Document                                                           | Purpose                          |
| ------------------------------------------------------------------ | -------------------------------- |
| [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md)                       | Architectural philosophy         |
| [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md) | Collection architecture decision |
| [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md)          | Write policy decision            |
| [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)                | SQLite ownership decision        |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)                     | Visual architecture diagrams     |
| [docs/LSP_INTEGRATION.md](./docs/LSP_INTEGRATION.md)               | LSP integration guide            |
| [README.md](./README.md)                                           | User documentation               |

---

**Version:** 1.7.0
**Last Updated:** 2026-02-07
**Changes:**

- v1.8.0: Branch-scoped point IDs with hybrid vector copy approach — new formula `SHA256(tenant_id | branch | file_path | chunk_index)`; added `is_archived` column to `watch_folders` with archive semantics (no watching/ingesting, fully searchable, no search exclusion); documented submodule archive safety with cross-reference checks; added cascade rename mechanism (queue-mediated `tenant_id` changes via SQLite-first + Qdrant eventual consistency); added remote rename detection (periodic git remote check vs stored URL); memory tool default scope changed from `global` to `project`; converted all Python code examples to Rust/TypeScript; added automated affinity grouping section (embedding-based pipeline, registry-sourced taxonomy, zero-shot classification, phased implementation plan)
- v1.7.0: Added comprehensive file type allowlist as primary ingestion gate (400+ extensions across 21 categories, 30+ extension-less filenames, size-restricted extensions, mandatory excluded directories); updated MCP registration policy — MCP server no longer auto-registers new projects, only re-activates existing entries (`register_if_new` field added to `RegisterProject` gRPC); expanded daemon startup automation from simple recovery to 6-step sequence (schema check, config reconciliation with fingerprinting, Qdrant collection verification, path validation, filesystem recovery, crash recovery); updated configuration reference — dropped legacy `.wq_config.yaml` name, documented embedded defaults via `include_str!()`, added complete `watching` section with allowlist/exclusion/size-restriction keys; added Qdrant dashboard visualization guide for named vectors; documented Distance Matrix API for graph visualization
- v1.6.7: Added comprehensive Deployment and Installation section documenting deployment architecture, platform support matrix (6 platforms), installation methods (binary, source, npm), Docker deployment modes, service management (macOS/Linux), CI/CD process, upgrade/migration procedures, and troubleshooting; renamed memory tool ruleId to label with LLM generation guidelines (max 15 chars, word-word-word format); added memory_limits config section; updated Docker compose files to reflect TypeScript/Rust architecture
- v1.6.6: Corrected session lifecycle documentation - clarified that Claude Code's SessionStart/SessionEnd are external hooks (shell commands configured in settings.json), not SDK callbacks; removed incorrect @anthropic-ai/claude-agent-sdk dependency (not needed); documented actual MCP SDK callbacks (server.onclose, onsessioninitialized for HTTP transport); clarified memory injection is via memory tool, not automatic session hooks
- v1.6.5: Updated all references from legacy `registered_projects` table to consolidated `watch_folders` table (priority calculation, activity tracking, batch processing); Python codebase removed in preparation for TypeScript MCP server
- v1.6.4: Updated unified_queue schema to match robust implementation - added status column for lifecycle tracking, lease_until/worker_id for crash recovery, idempotency_key for deduplication (supports content items without file paths), priority column for item type prioritization; documented idempotency key calculation and crash recovery procedure
- v1.6.3: Corrected SDK references from non-existent @anthropic/claude-code-sdk to actual packages (@modelcontextprotocol/sdk and @anthropic-ai/claude-agent-sdk); added SDK Architecture section explaining dual-SDK pattern
- v1.6.2: Consolidated database tables - merged `registered_projects`, `project_submodules`, and separate `watch_folders` tables into single unified `watch_folders` table; added activity inheritance for subprojects (parent and all submodules share `is_active` and `last_activity_at`); removed `project_aliases` table (dead code)
- v1.6.1: Clarified PATH configuration (expansion, merge, deduplication steps); documented session lifecycle with memory injection via Claude SDK; added Grammar and Runtime Management section (dynamic grammar loading, CLI update command, CI automation for 6 platforms)
- v1.6: MCP server rewrite decision - TypeScript instead of Python for native SessionStart/SessionEnd hook support; added TypeScript dependencies and rationale; updated architecture diagram
- v1.5: Major queue and daemon architecture update - simplified queue schema (removed status states, added failed flag and errors array); defined batch processing flow with sort alternation for anti-starvation; documented three daemon phases (initial scan, watching, removal); added daemon watch management lifecycle; defined semantic code chunking strategy; added Tree-sitter baseline + LSP enhancement architecture; defined LSP lifecycle and language server management; added PATH configuration for daemon
- v1.4: Clarified 4 MCP tools only (search, retrieve, memory, store); removed health/session as tools (health is server-internal affecting search responses with uncertainty status, session is automated); clarified memory collection is multi-tenant via nullable project_id; added detailed "table not found" graceful handling documentation; clarified "actively used" means RegisterProject received from MCP server; documented memory list action as search in disguise
- v1.3: Major API redesign - replaced `manage` tool with dedicated tools (`memory`, `health`, `session`); clarified MCP does NOT store to `projects` collection (daemon handles via file watching); added single configuration file requirement with cascade search; updated queue schema to include `memory` item_type; documented libraries as reference documentation (not programming libraries)
- v1.2: Updated API Reference to match actual implementation; clarified manage actions (removed create/delete collection as MCP actions); added pattern configuration documentation; updated gRPC services table format
- v1.1: Added comprehensive Project ID specification with duplicate handling, branch lifecycle, and registered_projects schema
