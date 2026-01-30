# workspace-qdrant-mcp Specification

**Version:** 1.3
**Date:** 2026-01-30
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
│                      PYTHON MCP SERVER                              │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  FastMCP Application                                      │     │
│   │  - store: Content storage with auto-categorization        │     │
│   │  - search: Hybrid semantic + keyword search               │     │
│   │  - manage: Collection and system management               │     │
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

| Component | Responsibilities | Writes To |
|-----------|------------------|-----------|
| **MCP Server** | Query processing, project detection, gRPC client, fallback queue | SQLite (queue only) |
| **Rust Daemon** | Document processing, embeddings, file watching, Qdrant writes, **SQLite schema owner** | SQLite + Qdrant |
| **CLI (wqm)** | Service management, library ingestion, admin operations | SQLite (queue only) |
| **SQLite** | State persistence, queue management, watch configuration | N/A (database) |
| **Qdrant** | Vector storage, semantic search, payload filtering | N/A (database) |

### SQLite Database Ownership

**Reference:** [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)

**The Rust daemon (memexd) is the sole owner of the SQLite database.**

| Aspect | Owner | Details |
|--------|-------|---------|
| Database creation | Daemon | Creates `state.db` if absent (path from config) |
| Schema creation | Daemon | Creates all tables on startup |
| Schema migrations | Daemon | Handles all schema version upgrades |
| Schema versioning | Daemon | Maintains `schema_version` table |

**Other components (MCP Server, CLI):**
- May read from any table
- May write to specific tables (e.g., `unified_queue`, `watch_folders`)
- Must NOT create tables or modify schema
- Must handle "table not found" gracefully (daemon not yet run)

**Default database path:** `~/.workspace-qdrant/state.db`

---

## Collection Architecture

**Reference:** [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md)

### Canonical Collections

The system uses exactly **3 collections**:

| Collection | Purpose | Multi-Tenant Key | Example |
|------------|---------|------------------|---------|
| `projects` | All project content | `project_id` | Code, docs, tests, configs |
| `libraries` | Reference documentation | `library_name` | Books, papers, API docs |
| `memory` | Behavioral rules | N/A | LLM preferences, constraints |

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

**Disambiguation applied on-demand:** Only when second clone is actively used with an agent.

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
   - Update `registered_projects` table
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

| Operation | When Triggered |
|-----------|----------------|
| Local project gains remote | Background (daemon watching) |
| New branch detected | Background (daemon watching) |
| Duplicate detection & disambiguation | On-demand (when starting work on project) |
| Project move detection | On-demand (when starting work on project) |

#### Registered Projects Table

```sql
CREATE TABLE registered_projects (
    project_id TEXT PRIMARY KEY,           -- Unique per clone (12-char hex)
    project_path TEXT NOT NULL UNIQUE,     -- Current filesystem path
    git_remote_url TEXT,                   -- Normalized remote (null if local)
    remote_hash TEXT,                      -- sha256(remote_url)[:12] for grouping duplicates
    disambiguation_path TEXT,              -- Path suffix used for disambiguation (null if none)
    container_folder TEXT NOT NULL,        -- Parent folder name
    created_at TIMESTAMP NOT NULL,
    last_seen_at TIMESTAMP,
    last_active_at TIMESTAMP               -- When project was last actively used
);

-- Index for finding duplicates (same remote, different paths)
CREATE INDEX idx_remote_hash ON registered_projects(remote_hash);
```

**Columns:**
- `project_id`: Unique identifier used in Qdrant payload
- `project_path`: Full filesystem path (for move detection)
- `git_remote_url`: Normalized URL (null for local projects)
- `remote_hash`: Groups clones of same repo for duplicate detection
- `disambiguation_path`: Non-null when disambiguation applied
- `container_folder`: Folder name containing project
- `last_active_at`: Tracks when project was last worked on (not just seen)

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
  "project_id": "a1b2c3d4e5f6",    // Required, indexed (is_tenant=true)
  "project_name": "my-project",
  "file_path": "src/main.py",       // Relative path
  "file_type": "code",              // code|doc|test|config|note|artifact
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
  "library_name": "numpy",          // Required, indexed (is_tenant=true)
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
  "label": "prefer-uv",             // Human-readable identifier (unique per scope)
  "content": "Use uv instead of pip for Python packages",
  "scope": "global",                // global|project
  "project_id": null,               // null for global, "abc123" for project-specific
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

| Message | Direction | Purpose |
|---------|-----------|---------|
| `RegisterProject(path)` | MCP → Daemon | Project is now active |
| `DeprioritizeProject(path)` | MCP → Daemon | Project is no longer active |

**All other operations use the queue.**

### Collection Ownership

- **Daemon owns all collections**: Creates the 3 canonical collections on startup
- **No collection creation via MCP/CLI**: Only `projects`, `libraries`, `memory` exist
- **No user-created collections**: The 3-collection model is fixed

### gRPC Methods (Reserved)

The daemon exposes gRPC methods for content ingestion (`IngestText`, etc.) but these are **reserved for administrative/diagnostic use only**. Production code paths (MCP, CLI) must NOT use them directly - all content goes through the queue.

### Unified Queue

**ALL writes go through the SQLite queue.** The queue serves as the transaction log for daemon processing.

```sql
CREATE TABLE unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,        -- memory|library|file|folder|project|delete_*|rename
    op TEXT NOT NULL,               -- ingest|update|delete|scan
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',  -- pending|in_progress|done|failed
    payload_json TEXT,
    created_at TEXT NOT NULL,
    leased_by TEXT,
    lease_expires_at TEXT
);
```

**Item Types (MCP-relevant):**

| item_type | Used By | payload_json |
|-----------|---------|--------------|
| `memory` | MCP `memory` tool | `{label, content, scope, project_id}` |
| `library` | MCP `store` tool | `{library_name, content, title, source, url}` |
| `file` | Daemon file watcher | `{file_path, ...}` |
| `folder` | Daemon folder scan | `{folder_path, patterns, ...}` |

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
  "label": "prefer-uv",                    // Human-readable identifier
  "content": "Use uv instead of pip for Python packages",
  "scope": "global",                       // global | project
  "project_id": null,                      // null for global, project_id for project-specific
  "created_at": "2026-01-30T12:00:00Z"
}
```

**Uniqueness constraint:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

### Rule Scope

| Scope | Application | project_id |
|-------|-------------|------------|
| `global` | All projects | `null` |
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
manage(action="list_rules")          # List global + current project rules
manage(action="add_rule", label="...", content="...", scope="global|project")
manage(action="remove_rule", label="...", scope="global|project")
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

| Source | Target Collection | Trigger |
|--------|-------------------|---------|
| **MCP** | `projects` | `activate_project` → daemon auto-creates watch if not exists |
| **CLI** | `libraries` | `wqm library add` → explicit library registration |

**Note:** Projects are watched automatically when activated via MCP. CLI is only used for library registration.

### Watch Table Schema

```sql
CREATE TABLE watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    collection TEXT NOT NULL,           -- "projects" or "libraries"
    project_id TEXT,                    -- For projects (null for libraries)
    library_name TEXT,                  -- For libraries (null for projects)
    library_mode TEXT,                  -- "sync" or "incremental" (libraries only, null for projects)
    follow_symlinks INTEGER DEFAULT 0,  -- Default: don't follow symlinks
    auto_ingest INTEGER DEFAULT 1,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

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
2. **Separate project:** Submodule treated as separate project with its own `project_id`
3. **Link:** Main project stores reference to submodule

```sql
CREATE TABLE project_submodules (
    parent_project_id TEXT NOT NULL,
    submodule_project_id TEXT NOT NULL,
    submodule_path TEXT NOT NULL,        -- Relative path within parent
    created_at TEXT NOT NULL,
    PRIMARY KEY (parent_project_id, submodule_path),
    FOREIGN KEY (parent_project_id) REFERENCES registered_projects(project_id),
    FOREIGN KEY (submodule_project_id) REFERENCES registered_projects(project_id)
);
```

### Daemon Polling

The daemon:
1. Polls `watch_folders` table every 5 seconds
2. Detects changes via `updated_at` timestamp
3. Updates file watchers dynamically
4. Processes file events through ingestion queue

### Ingestion Pipeline

Different operations have different pipelines:

| Event | Pipeline |
|-------|----------|
| **New file** | Debounce → Read → Parse/Chunk → Embed → Upsert |
| **File changed** | Debounce → Read → Parse/Chunk → Embed → Upsert (replace) |
| **File deleted** | Delete from Qdrant (filter by `file_path` + `project_id`) |
| **File renamed** | Delete old + Upsert new (simple approach) |

**Common processing steps:**
```
Read Content → Parse/Chunk → Generate Embeddings → Upsert to Qdrant
                   │
                   ├── LSP symbols (for code files)
                   ├── Metadata extraction (file_type, language)
                   └── Content hashing (deduplication)
```

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

The server provides tools for search, memory management, and health monitoring.

**Important design principle:** The MCP server does NOT store to the `projects` collection. Project content is ingested by the daemon via file watching. The MCP can only store to `memory` (rules) and `libraries` (reference documentation).

#### search

Semantic search with optional direct retrieval mode.

```python
search(
    query: str,                      # Required: search query
    collection: str,                 # Required: projects|libraries|memory
    mode: str = "hybrid",            # hybrid|semantic|keyword|retrieve
    limit: int = 10,                 # Max results
    score_threshold: float = 0.3,    # Minimum similarity score (ignored in retrieve mode)
    # Collection-specific scope filters (see below)
    scope: str = None,               # Scope within collection
    branch: str = None,              # For projects: branch filter
    project_id: str = None,          # For projects: specific project
    library_name: str = None         # For libraries: specific library
)
```

**Modes:**
- `hybrid`: Semantic + keyword search (default)
- `semantic`: Pure vector similarity
- `keyword`: Keyword/exact matching
- `retrieve`: Direct document access by ID or metadata (no ranking)

**Collection-specific scope:**

| Collection | Scope Options | Notes |
|------------|---------------|-------|
| `memory` | `all`, `global`, `project` | `project` = current project's rules |
| `projects` | `all`, `current`, `other` | Combined with `branch` and `project_id` filters |
| `libraries` | `all`, `<library_name>` | Filter by specific library |

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
- `list`: List rules (read-only, direct query)

**Uniqueness:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

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

#### health

System health check.

```python
health() -> dict
```

**Returns:**
- Daemon connection status
- Queue depth and status
- Qdrant connection status
- Active project info

#### session

Session lifecycle management.

```python
session(
    action: str                      # Required: activate|deactivate
)
```

**Actions:**
- `activate`: Register current project with daemon, start heartbeat (direct gRPC)
- `deactivate`: Stop heartbeat, deprioritize project (direct gRPC)

### Removed Actions

The following actions from the legacy `manage` tool are **removed**:

| Removed | Reason |
|---------|--------|
| `list_collections` | Diagnostic only, use CLI |
| `collection_info` | Diagnostic only, use CLI |
| `workspace_status` | Replaced by `health` tool |
| `init_project` | Daemon handles via watching |
| `cleanup` | Daemon handles internally |
| `create_collection` | Daemon owns collections |
| `delete_collection` | Daemon owns collections |
| `mark_library_deleted` | CLI only |
| `restore_deleted_library` | CLI only |
| `list_deleted_libraries` | CLI only |

### gRPC Services

The daemon exposes 3 gRPC services on port 50051.

#### SystemService

| Method | Used By | Purpose | Status |
|--------|---------|---------|--------|
| `Health` | MCP, CLI | Health check | Production |
| `GetMetrics` | CLI | System metrics | Production |
| `GetQueueStats` | CLI | Queue statistics | Production |
| `RegisterProject` | MCP | Notify project is active | **Production (direct gRPC)** |
| `DeprioritizeProject` | MCP | Notify project is inactive | **Production (direct gRPC)** |
| `Heartbeat` | MCP | Session heartbeat | Production |
| `Shutdown` | CLI | Graceful shutdown | Production |

#### CollectionService

| Method | Status | Notes |
|--------|--------|-------|
| `CreateCollection` | **Daemon internal** | Daemon creates collections on startup |
| `DeleteCollection` | **Not used** | Fixed 3-collection model |
| `ListCollections` | Read-only | Can be exposed to MCP/CLI |
| `GetCollection` | Read-only | Can be exposed to MCP/CLI |
| `UpdateCollectionAlias` | **Daemon internal** | |

**MCP/CLI must NOT call collection mutation methods.** Only read-only methods are permitted.

#### DocumentService

| Method | Status | Notes |
|--------|--------|-------|
| `IngestText` | **Reserved** | Admin/diagnostic use only |
| `UpdateDocument` | **Reserved** | Admin/diagnostic use only |
| `DeleteDocument` | **Reserved** | Admin/diagnostic use only |

**Production writes use SQLite queue.** These methods exist for administrative and diagnostic purposes but are not called by MCP or CLI in normal operation.

---

## Configuration Reference

### Single Configuration File

**All components (Daemon, MCP Server, CLI) share the same configuration file.** This is a key architectural requirement to prevent configuration drift between components.

**Configuration file search order (first found wins):**
1. `~/.workspace-qdrant/config.yaml` (or `.yml`)
2. `~/.config/workspace-qdrant/config.yaml`
3. `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS)

**Built-in defaults:** The configuration template (`assets/default_configuration.yaml`) defines all default values. User configuration only needs to specify overrides.

### Environment Variables

Environment variables override configuration file values:

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | None |
| `FASTEMBED_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `WQM_STDIO_MODE` | Force stdio mode | `false` |
| `WQM_CLI_MODE` | Force CLI mode | `false` |

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
```

**Note:** Pattern configuration is part of the unified config file, not separate YAML files. The `patterns/` directory files serve as reference documentation for the comprehensive default patterns.

### SQLite Database

**Path:** `~/.workspace-qdrant/state.db`

**Owner:** Rust daemon (memexd) - see [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md)

**Core Tables:**

| Table | Purpose | Used By |
|-------|---------|---------|
| `schema_version` | Schema version tracking | Daemon |
| `unified_queue` | Write queue for daemon processing | MCP, CLI, Daemon |
| `watch_folders` | File watching configuration | MCP, CLI, Daemon |
| `registered_projects` | Project registration and disambiguation | Daemon |
| `project_submodules` | Git submodule relationships | Daemon |
| `file_processing` | File ingestion tracking | Daemon |

**Other components (MCP, CLI) may read/write to tables but must NOT create tables or run migrations.**

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md) | Architectural philosophy |
| [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md) | Collection architecture decision |
| [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md) | Write policy decision |
| [ADR-003](./docs/adr/ADR-003-daemon-owns-sqlite.md) | SQLite ownership decision |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Visual architecture diagrams |
| [README.md](./README.md) | User documentation |

---

**Version:** 1.3
**Last Updated:** 2026-01-30
**Changes:**
- v1.3: Major API redesign - replaced `manage` tool with dedicated tools (`memory`, `health`, `session`); clarified MCP does NOT store to `projects` collection (daemon handles via file watching); added single configuration file requirement with cascade search; updated queue schema to include `memory` item_type; documented libraries as reference documentation (not programming libraries)
- v1.2: Updated API Reference to match actual implementation; clarified manage actions (removed create/delete collection as MCP actions); added pattern configuration documentation; updated gRPC services table format
- v1.1: Added comprehensive Project ID specification with duplicate handling, branch lifecycle, and registered_projects schema
