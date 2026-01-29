# workspace-qdrant-mcp Specification

**Version:** 1.0
**Date:** 2026-01-28
**Status:** Authoritative Specification
**Supersedes:** CONSOLIDATED_PRD_V2.md, PRDv3.txt, PRDv3-snapshot1.txt

---

## Table of Contents

1. [Overview and Vision](#overview-and-vision)
2. [Architecture](#architecture)
3. [Collection Architecture](#collection-architecture)
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
| **Rust Daemon** | Document processing, embeddings, file watching, Qdrant writes | SQLite + Qdrant |
| **CLI (wqm)** | Service management, library ingestion, admin operations | SQLite (queue only) |
| **SQLite** | State persistence, queue management, watch configuration | N/A (database) |
| **Qdrant** | Vector storage, semantic search, payload filtering | N/A (database) |

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

Project IDs are 12-character hex hashes derived from:
1. **Git remote URL** (normalized): Preferred for collaborative projects
2. **Path hash**: Fallback for local-only projects

```python
def calculate_tenant_id(project_root: Path) -> str:
    # Try git remote first
    git_remote = get_git_remote_url(project_root)
    if git_remote:
        normalized = normalize_git_url(git_remote)
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]

    # Fall back to path hash
    return hashlib.sha256(str(project_root).encode()).hexdigest()[:12]
```

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
  "rule_id": "prefer-uv",
  "rule_type": "preference",        // preference|behavior|constraint|pattern
  "content": "Use uv instead of pip for Python packages",
  "priority": 7,                    // 1-10
  "scope": "global",                // global|project|language
  "enabled": true,
  "created_at": "2026-01-28T12:00:00Z"
}
```

---

## Write Path Architecture

**Reference:** [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md)

### Core Rule

**The Rust daemon (memexd) is the ONLY component that writes to Qdrant. No exceptions.**

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
│        ┌────────────────┐     ┌─────────────────┐                   │
│        │  Try gRPC to   │────▶│  Daemon writes  │                   │
│        │  Daemon        │     │  to Qdrant      │                   │
│        └───────┬────────┘     └─────────────────┘                   │
│                │                                                     │
│        [Daemon unavailable]                                          │
│                │                                                     │
│                ▼                                                     │
│        ┌────────────────┐                                           │
│        │  SQLite Queue  │  ← Fallback path                          │
│        │  (unified_     │                                           │
│        │   queue)       │                                           │
│        └───────┬────────┘                                           │
│                │                                                     │
│        [Daemon starts]                                               │
│                │                                                     │
│                ▼                                                     │
│        ┌────────────────┐                                           │
│        │  Queue         │────▶  Qdrant                              │
│        │  Processor     │                                           │
│        └────────────────┘                                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified Queue

When daemon is unavailable, writes are queued in SQLite:

```sql
CREATE TABLE unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,
    item_type TEXT NOT NULL,        -- content|file|folder|project|library|delete_*|rename
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

### Idempotency

All queue operations use SHA256-based idempotency keys:

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

### Fallback Response

When content is queued instead of directly processed:

```json
{
  "success": true,
  "status": "queued",
  "message": "Content queued for processing. Daemon will ingest when available.",
  "queue_id": "abc123",
  "fallback_mode": "unified_queue"
}
```

---

## Memory System

### Purpose

The memory collection stores LLM behavioral rules that persist across sessions. Rules are injected into Claude's context at session start.

### Rule Types

| Type | Description | Example |
|------|-------------|---------|
| `preference` | User preferences | "Use TypeScript strict mode" |
| `behavior` | Agent behaviors | "Always run tests before committing" |
| `constraint` | Hard constraints | "Never delete production data" |
| `pattern` | Code patterns | "Use dependency injection" |

### Rule Scope

| Scope | Application |
|-------|-------------|
| `global` | All projects, all languages |
| `project` | Specific project only |
| `language` | Specific programming language |

### Context Injection

At session start:
1. MCP server queries `memory` collection for applicable rules
2. Rules filtered by scope (global + current project + current language)
3. Rules sorted by priority (higher first)
4. Formatted and injected into system context

### Conversational Updates

Rules can be added conversationally:

```
User: "For future reference, always use uv instead of pip"
→ Creates memory rule: {rule_type: "preference", content: "Use uv for Python packages", scope: "global"}
```

---

## File Watching and Ingestion

### Watch Configuration

Watch folders are configured via CLI and stored in SQLite:

```bash
# Add watch folder
wqm watch add /path/to/project --collection projects --patterns "*.py,*.md"

# List watches
wqm watch list

# Remove watch
wqm watch remove /path/to/project
```

### Watch Table Schema

```sql
CREATE TABLE watch_folders (
    watch_id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    collection TEXT NOT NULL,
    patterns TEXT NOT NULL,           -- JSON array
    ignore_patterns TEXT NOT NULL,    -- JSON array
    auto_ingest INTEGER DEFAULT 1,
    recursive INTEGER DEFAULT 1,
    recursive_depth INTEGER DEFAULT 10,
    debounce_seconds REAL DEFAULT 2.0,
    enabled INTEGER DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

### Daemon Polling

The daemon:
1. Polls `watch_folders` table every 5 seconds
2. Detects changes via `updated_at` timestamp
3. Updates file watchers dynamically
4. Processes file events through ingestion queue

### Ingestion Pipeline

```
File Event → Debounce → Read Content → Parse/Chunk → Generate Embeddings → Upsert to Qdrant
                                          │
                                          ├── LSP symbols (for code)
                                          ├── Metadata extraction
                                          └── Content hashing (deduplication)
```

---

## API Reference

### MCP Tools

The server provides 4 comprehensive MCP tools:

#### store

Store content with automatic categorization.

```python
store(
    content: str,                    # Required: text content
    collection: str = "projects",    # Target collection
    metadata: dict = {},             # Additional metadata
    main_tag: str = None,            # Primary tag
    full_tag: str = None             # Full tag path
)
```

#### search

Hybrid semantic + keyword search.

```python
search(
    query: str,                      # Required: search query
    collection: str = "projects",    # Collection to search
    mode: str = "hybrid",            # hybrid|semantic|exact|keyword
    scope: str = "project",          # project|collection|global|all
    file_type: str = None,           # Filter by file type
    branch: str = None,              # Filter by branch
    limit: int = 10                  # Max results
)
```

#### manage

Collection and system management.

```python
manage(
    action: str,                     # Required: action to perform
    **kwargs                         # Action-specific parameters
)

# Actions:
# - init_project: Initialize current directory as project
# - activate_project: Set project as active (high priority)
# - deactivate_project: Set project as inactive
# - create_collection: Create new collection
# - delete_collection: Delete collection
# - list_collections: List all collections
# - health: System health check
```

#### retrieve

Direct document access.

```python
retrieve(
    document_id: str = None,         # Specific document ID
    file_path: str = None,           # Filter by file path
    collection: str = "projects",    # Collection to query
    include_content: bool = True     # Include document content
)
```

### gRPC Services

The daemon exposes 3 gRPC services on port 50051:

#### SystemService
- `Health`: Health check
- `GetMetrics`: System metrics
- `GetQueueStats`: Queue statistics
- `RegisterProject`: Register project session
- `DeprioritizeProject`: End project session
- `Heartbeat`: Session heartbeat
- `Shutdown`: Graceful shutdown

#### CollectionService
- `CreateCollection`: Create new collection
- `DeleteCollection`: Delete collection
- `ListCollections`: List all collections
- `GetCollection`: Get collection info
- `UpdateCollectionAlias`: Manage aliases

#### DocumentService
- `IngestText`: Ingest text content
- `UpdateDocument`: Update existing document
- `DeleteDocument`: Delete document

---

## Configuration Reference

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key | None |
| `FASTEMBED_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `WQM_STDIO_MODE` | Force stdio mode | `false` |
| `WQM_CLI_MODE` | Force CLI mode | `false` |

### Configuration Files

| File | Purpose |
|------|---------|
| `.env` | Environment variables |
| `.wq_config.yaml` | Project-specific MCP configuration (preferred) |
| `.workspace-qdrant.yaml` | Project-specific MCP configuration (alternate) |
| `assets/default_configuration.yaml` | System defaults |

### SQLite Database

Location: `~/.workspace-qdrant/state.db` (shared between Python MCP and Rust daemon)

Tables:
- `unified_queue`: Write queue for daemon processing
- `watch_folders`: File watching configuration
- `project_state`: Active project sessions
- `ingestion_status`: File ingestion tracking
- `tenant_aliases`: Project ID aliases

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [FIRST-PRINCIPLES.md](./FIRST-PRINCIPLES.md) | Architectural philosophy |
| [ADR-001](./docs/adr/ADR-001-canonical-collection-architecture.md) | Collection architecture decision |
| [ADR-002](./docs/adr/ADR-002-daemon-only-write-policy.md) | Write policy decision |
| [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) | Visual architecture diagrams |
| [docs/architecture/multi-tenant-collection-schema.md](./docs/architecture/multi-tenant-collection-schema.md) | Detailed schema specification |
| [README.md](./README.md) | User documentation |

---

**End of workspace-qdrant-mcp Specification v1.0**
