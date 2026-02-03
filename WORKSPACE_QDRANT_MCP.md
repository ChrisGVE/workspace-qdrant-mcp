# workspace-qdrant-mcp Specification

**Version:** 1.6.7
**Date:** 2026-02-03
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
- **Three Collections Only**: Exactly `projects`, `libraries`, `memory` (see [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md))

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

The system uses exactly **3 collections**:

| Collection  | Purpose                 | Multi-Tenant Key        | Example                      |
| ----------- | ----------------------- | ----------------------- | ---------------------------- |
| `projects`  | All project content     | `project_id`            | Code, docs, tests, configs   |
| `libraries` | Reference documentation | `library_name`          | Books, papers, API docs      |
| `memory`    | Behavioral rules        | `project_id` (nullable) | LLM preferences, constraints |

**Memory collection multi-tenancy:** Rules with `scope="global"` have `project_id=null` and apply to all projects. Rules with `scope="project"` have a specific `project_id` and apply only to that project.

**No other collections are permitted.** No underscore prefixes, no per-project collections, no `{basename}-{type}` patterns.

### Multi-Tenant Isolation

Projects and libraries are isolated via payload metadata filtering:

```python
# Project-scoped search (automatic in MCP)
search(
    collection="projects",
    filter={"must": [{"key": "project_id", "match": {"value": "a1b2c3d4e5f6"}}]}
)

# Cross-project search (global scope)
search(collection="projects")  # No project_id filter

# Library search
search(
    collection="libraries",
    filter={"must": [{"key": "library_name", "match": {"value": "numpy"}}]}
)

# Memory search (global rules only)
search(
    collection="memory",
    filter={"must": [{"key": "scope", "match": {"value": "global"}}]}
)

# Memory search (project-specific rules)
search(
    collection="memory",
    filter={"must": [{"key": "project_id", "match": {"value": "a1b2c3d4e5f6"}}]}
)
```

### Project ID Generation

Project IDs (`project_id`) are 12-character hex hashes that uniquely identify a project clone. The system handles multiple clones of the same repository, filesystem moves, and local projects gaining git remotes.

#### Core Algorithm

```python
def calculate_project_id(project_root: Path, disambiguation_path: str = None) -> str:
    """
    Calculate project_id for a project.

    Args:
        project_root: Absolute path to project root
        disambiguation_path: Optional path suffix for duplicate detection
    """
    git_remote = get_git_remote_url(project_root)

    if git_remote:
        # Normalize git remote URL
        normalized = normalize_git_url(git_remote)
        if disambiguation_path:
            return hashlib.sha256(f"{normalized}|{disambiguation_path}".encode()).hexdigest()[:12]
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    # Local project: use container folder name
    container_folder = project_root.name
    return hashlib.sha256(container_folder.encode()).hexdigest()[:12]
```

#### Git Remote URL Normalization

All git remote URLs are normalized for consistency:

1. Remove `.git` suffix
2. Convert to lowercase
3. Convert SSH format to HTTPS format
4. Remove protocol prefix for hashing

```python
def normalize_git_url(url: str) -> str:
    """
    Normalize git URL for consistent hashing.

    Examples:
        git@github.com:user/repo.git  → github.com/user/repo
        https://github.com/User/Repo.git → github.com/user/repo
        ssh://git@gitlab.com/user/repo → gitlab.com/user/repo
    """
    # Remove .git suffix
    url = url.removesuffix(".git")

    # Convert SSH to HTTPS-like format
    if url.startswith("git@"):
        url = url.replace("git@", "").replace(":", "/")
    elif url.startswith("ssh://git@"):
        url = url.replace("ssh://git@", "")

    # Remove protocol
    for protocol in ["https://", "http://", "ssh://"]:
        url = url.removeprefix(protocol)

    # Lowercase for consistency
    return url.lower()
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
- **Branch stored as metadata**: `branch` field in Qdrant payload
- **Default search**: Returns results from all branches
- **Filtered search**: Use `branch="main"` to scope to specific branch

**Branch lifecycle:**

- **New branch detected**: Auto-ingest during file watching
- **Branch deleted**: Delete all documents with `branch="deleted_branch"` from Qdrant
- **Branch renamed**: Treat as delete + create (Git doesn't track renames)

#### Local Project Gains Remote

When a local project (identified by container folder) gains a git remote:

1. Daemon detects `.git/config` change during watching
2. If project at same location:
   - Compute new `project_id` from normalized remote URL
   - Update `watch_folders` table
   - Bulk update Qdrant documents via `set_payload` API:
     ```python
     client.set_payload(
         collection_name="projects",
         payload={"project_id": new_project_id},
         filter=Filter(must=[
             FieldCondition(key="project_id", match=MatchValue(value=old_project_id))
         ])
     )
     ```
3. No re-ingestion required

#### Operation Timing

| Operation                            | When Triggered                                              |
| ------------------------------------ | ----------------------------------------------------------- |
| Local project gains remote           | Background (daemon watching `.git/config`)                  |
| New branch detected                  | Background (daemon watching)                                |
| Duplicate detection & disambiguation | On-demand (when `RegisterProject` received from MCP server) |
| Project move detection               | On-demand (when `RegisterProject` received from MCP server) |

**"Active" definition:** A project becomes active when the MCP server automatically sends `RegisterProject` to the daemon upon detecting the user's working directory. This triggers on-demand operations like duplicate detection.

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
  "file_path": "src/main.py", // Relative path
  "file_type": "code", // code|doc|test|config|note|artifact
  "language": "python",
  "branch": "main",
  "symbols": ["MyClass", "my_function"],
  "chunk_index": 0,
  "total_chunks": 5,
  "created_at": "2026-01-28T12:00:00Z"
}
```

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

- **Daemon owns all collections**: Creates the 3 canonical collections on startup
- **No collection creation via MCP/CLI**: Only `projects`, `libraries`, `memory` exist
- **No user-created collections**: The 3-collection model is fixed

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

**Anti-starvation mechanism:** Every 10 queue pops, alternate between:

- `ORDER BY priority DESC, created_at ASC` (active projects first)
- `ORDER BY priority ASC, created_at ASC` (inactive projects get a turn)

This prevents inactive projects from being starved indefinitely.

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
2. Append error message to `errors` (JSON array)
3. If `retry_count >= max_retries`: set `failed = 1`

**Failed items:**

- Stay in queue with `failed = 1`
- Skipped by normal processing (query: `WHERE failed = 0`)
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

**Sort alternation:** Every 10 items, daemon flips between:

- `ORDER BY priority DESC` (active projects first)
- `ORDER BY priority ASC` (inactive projects get a turn)

This prevents starvation of low-priority items.

**Qdrant atomicity:** Each batch request to Qdrant is atomic. Grouping by (op, collection) leverages this for efficiency.

**Idempotency:** All operations are idempotent - retries are safe (delete non-existent = no-op, upsert = replace).

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
  "file_path": "/absolute/path/to/file.py",
  "relative_path": "src/main.py"
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
3. Daemon processes files:
   - Ingest with LSP/tree-sitter metadata
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

| Event          | Action                                                      |
| -------------- | ----------------------------------------------------------- |
| File created   | Queue `file/ingest` (uniqueness prevents duplicates)        |
| File modified  | Queue `file/ingest` (uniqueness: if already queued, ignore) |
| File deleted   | Queue `file/delete`                                         |
| Folder created | Queue `folder/scan`                                         |
| Folder deleted | Remove all files in collection with path prefix             |

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

```python
memory(action="list")                # List global + current project rules
memory(action="add", label="...", content="...", scope="global|project")
memory(action="remove", label="...", scope="global|project")
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

| Source  | Target Collection | Trigger                                                      |
| ------- | ----------------- | ------------------------------------------------------------ |
| **MCP** | `projects`        | `activate_project` → daemon auto-creates watch if not exists |
| **CLI** | `libraries`       | `wqm library add` → explicit library registration            |

**Note:** Projects are watched automatically when activated via MCP. CLI is only used for library registration.

### Watch Table Schema

See [Watch Folders Table (Unified)](#watch-folders-table-unified) in the Collection Architecture section for the complete schema.

**Library modes (libraries only):**

- `sync`: Full synchronization - additions, updates, AND deletions
- `incremental`: Additions and updates only, no deletions

**Always recursive:** No depth limit configuration needed.

### Patterns Configuration

Patterns are defined in configuration file, not per-watch:

```yaml
# ~/.workspace-qdrant/config.yaml
watching:
  patterns:
    - "*.py"
    - "*.rs"
    - "*.md"
    - "*.txt"
    - "*.js"
    - "*.ts"
  ignore_patterns:
    - "*.pyc"
    - "__pycache__/*"
    - ".git/*"
    - "node_modules/*"
    - "target/*"
    - ".venv/*"
    - "*.lock"
```

### Git Submodules

When a subfolder contains a `.git` directory (submodule):

1. **Detect:** Daemon detects `.git` in subfolder during scanning
2. **Separate entry:** Submodule is registered in `watch_folders` with its own `tenant_id` (project_id)
3. **Link via parent_watch_id:** Submodule entry references parent project via `parent_watch_id` and stores relative path in `submodule_path`
4. **Activity inheritance:** Submodules share `is_active` and `last_activity_at` with parent (see [Activity Inheritance](#watch-folders-table-unified))

See [Watch Folders Table (Unified)](#watch-folders-table-unified) for the schema that handles both projects and submodules.

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
  "file_path": "src/auth.py",
  "chunk_type": "function",
  "symbol_name": "validate_token",
  "symbol_kind": "function",
  "parent_symbol": null,
  "language": "python",
  "start_line": 42,
  "end_line": 67,
  "docstring": "Validates JWT token and returns claims.",
  "signature": "def validate_token(token: str) -> bool",
  "calls": ["decode_jwt", "check_expiry"],
  "is_fragment": false
}
```

**LSP enrichment adds:**

```json
{
  "references": [
    { "file": "src/api.py", "line": 23 },
    { "file": "src/middleware.py", "line": 56 }
  ],
  "type_info": "Callable[[str], bool]"
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

- The MCP server does NOT store to the `projects` collection. Project content is ingested by the daemon via file watching.
- The MCP can only store to `memory` (rules) and `libraries` (reference documentation).
- Session management (activate/deactivate) is automated, not exposed as a tool.
- Health monitoring is server-internal and affects search response metadata.

#### search

Semantic search with optional direct retrieval mode.

```python
search(
    query: str,                      # Required: search query
    collection: str = "projects",    # projects|libraries|memory
    mode: str = "hybrid",            # hybrid|semantic|keyword|retrieve
    limit: int = 10,                 # Max results
    score_threshold: float = 0.3,    # Minimum similarity score (ignored in retrieve mode)
    # Collection-specific scope filters (see below)
    scope: str = "project",          # Scope within collection
    branch: str = None,              # For projects: branch filter
    project_id: str = None,          # For projects: specific project
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

```python
# Current project, current branch (default)
search(query="auth", collection="projects", scope="current")

# Current project, all branches
search(query="auth", collection="projects", scope="current", branch="*")

# All projects
search(query="auth", collection="projects", scope="all")

# Specific project
search(query="auth", collection="projects", scope="other", project_id="abc123")
```

**project_id handling:** The MCP server FETCHES `project_id` from the daemon's state database (not calculated locally). This prevents drift between MCP and daemon. The fetch happens on first search operation to allow time for daemon to register the watch folder.

#### retrieve

Direct document access for chunk-by-chunk retrieval.

```python
retrieve(
    document_id: str = None,         # Specific document ID
    collection: str = "projects",    # Target collection
    metadata: dict = None,           # Metadata filters
    limit: int = 10,                 # Max documents
    offset: int = 0                  # For pagination
)
```

**Use case:** Retrieving large documents chunk by chunk without overwhelming context. Use `search` with `mode="retrieve"` for metadata-based retrieval, or `retrieve` for ID-based access.

#### memory

Manage memory rules (behavioral preferences).

```python
memory(
    action: str,                     # Required: add|update|remove|list
    label: str = None,               # Rule label (unique per scope)
    content: str = None,             # Rule content (for add/update)
    scope: str = "global",           # global|project
    project_id: str = None           # For project-scoped rules
)
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

Store content to libraries collection only.

```python
store(
    content: str,                    # Required: text content
    library_name: str,               # Required: library identifier (acts as tag)
    title: str = None,               # Document title
    source: str = "user_input",      # Source type (user_input|web|file)
    url: str = None,                 # Source URL (for web content)
    metadata: dict = None            # Additional metadata
)
```

**Note:** `store` is for adding reference documentation to the `libraries` collection (like adding books to a library). It is NOT for project content (handled by daemon file watching) or memory rules (use `memory` tool).

**Libraries definition:** Libraries are collections of reference information (books, documentation, papers, websites) - NOT programming libraries (use context7 MCP for those).

### Session Lifecycle

Session lifecycle is **automatic**, managed by the MCP server using the MCP SDK's `server.onclose` callback and server initialization logic.

**Implementation:** The MCP server uses `@modelcontextprotocol/sdk` which provides:
- Session initialization on transport connection (stdio or HTTP)
- `server.onclose` callback for cleanup when session ends

#### On Server Start (Transport Connection)

When the MCP server connects to the transport (stdio or HTTP):

1. **Project detection and activation:**
   - Server detects project from working directory
   - Server sends `RegisterProject` to daemon with current `project_id`
   - Daemon sets `is_active = true` and updates `last_activity_at`

2. **Start heartbeat:**
   - Periodic heartbeat with daemon to prevent timeout-based deactivation

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
| `DeleteCollection`      | **Not used**        | Fixed 3-collection model              |
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

| Method                | Used By | Purpose                     | Status                       |
| --------------------- | ------- | --------------------------- | ---------------------------- |
| `RegisterProject`     | MCP     | Register project as active  | **Production (direct gRPC)** |
| `DeprioritizeProject` | MCP     | Decrement session count     | **Production (direct gRPC)** |
| `GetProjectStatus`    | MCP     | Get project status          | Production                   |
| `ListProjects`        | CLI     | List all registered projects| Production                   |
| `Heartbeat`           | MCP     | Keep session alive (60s)    | Production                   |

**Session Management:**
- MCP servers call `RegisterProject` on startup to prioritize their project's ingestion
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

**Configuration file search order (first found wins):**

1. `WQM_CONFIG_PATH` environment variable (explicit override)
2. `.wq_config.yaml` or `.workspace-qdrant.yaml` (project-local, daemon/MCP only)
3. `~/.workspace-qdrant/config.yaml` (or `.yml`)
4. `~/.config/workspace-qdrant/config.yaml`
5. `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS)

**Note:** The CLI does not search project-local configs since it operates system-wide.

**Built-in defaults:** The configuration template (`assets/default_configuration.yaml`) defines all default values. User configuration only needs to specify overrides.

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

The unified configuration file contains all settings:

```yaml
# Database configuration
database:
  path: ~/.workspace-qdrant/state.db

# Qdrant connection
qdrant:
  url: http://localhost:6333
  api_key: null
  timeout: 30s

# Daemon settings
daemon:
  grpc_port: 50051
  queue_poll_interval_ms: 1000
  queue_batch_size: 10

# File watching patterns (inline, not separate files)
watching:
  patterns:
    - "*.py"
    - "*.rs"
    - "*.md"
    - "*.js"
    - "*.ts"
  ignore_patterns:
    - "*.pyc"
    - "__pycache__/*"
    - ".git/*"
    - "node_modules/*"
    - "target/*"
    - ".venv/*"

# Collections configuration
collections:
  memory_collection_name: "memory"

# User environment (written by CLI, read by daemon)
environment:
  user_path: "/usr/local/bin:/opt/homebrew/bin:..." # Set by CLI on first run
```

**Note:** Pattern configuration is part of the unified config file, not separate YAML files. The `patterns/` directory files serve as reference documentation for the comprehensive default patterns.

### SQLite Database

**Path:** `~/.workspace-qdrant/state.db`

**Owner:** Rust daemon (memexd) - see [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)

**Core Tables:**

| Table            | Purpose                                                           | Used By          |
| ---------------- | ----------------------------------------------------------------- | ---------------- |
| `schema_version` | Schema version tracking                                           | Daemon           |
| `unified_queue`  | Write queue for daemon processing                                 | MCP, CLI, Daemon |
| `watch_folders`  | Unified table for projects, libraries, and submodules (see below) | MCP, CLI, Daemon |

**Note:** The `watch_folders` table consolidates what were previously separate `registered_projects`, `project_submodules`, and `watch_folders` tables. See [Watch Folders Table (Unified)](#watch-folders-table-unified) for the complete schema.

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

**ONNX Runtime Bundling:**

The Rust daemon bundles ONNX Runtime for embedding generation:
- Linux: Dynamic linking with bundled `.so` files
- macOS: Framework bundle for both architectures
- Windows: Bundled `.dll` files

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
    <string>~/.workspace-qdrant/logs/memexd.log</string>
    <key>StandardErrorPath</key>
    <string>~/.workspace-qdrant/logs/memexd.err</string>
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

[Install]
WantedBy=default.target
```

**Manual control:**
```bash
systemctl --user daemon-reload
systemctl --user enable memexd
systemctl --user start memexd
systemctl --user status memexd
journalctl --user -u memexd -f
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
Qdrant          healthy   12ms      v1.7.3, 3 collections
Daemon          healthy   2ms       pid=12345, uptime=2h15m
Database        healthy   1ms       23 watch folders, 156 queue items
```

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
| `update-grammars.yml` | Schedule | Update tree-sitter grammars |
| `update-onnx.yml` | Schedule | Update ONNX Runtime |

#### Automated Dependency Updates

- **Tree-sitter grammars:** Weekly check for grammar updates
- **ONNX Runtime:** Monthly check for new versions
- **Rust dependencies:** Dependabot PRs for security updates

### Upgrade and Migration

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

**Version:** 1.6.7
**Last Updated:** 2026-02-03
**Changes:**

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
