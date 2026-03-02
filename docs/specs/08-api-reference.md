## API Reference

### MCP Tools

The server provides exactly **6 tools**: `search`, `retrieve`, `rules`, `store`, `grep`, and `list`.

**Important design principles:**

- The MCP server does NOT store content to the `projects` collection. Project content is ingested exclusively by the daemon via file watching.
- The MCP server CAN register new projects and activate/keep-active existing ones via `store` with `type: "project"`.
- The MCP server stores ad-hoc content (strings, files, URLs) to the `scratchpad` collection, which supports both global and project-scoped entries.
- The MCP server can store content to a named `library` only when the user explicitly instructs it to do so (not autonomously).
- The MCP server manages behavioral rules (strings only, global or per-project) via the `rules` tool.
- The MCP server searches code with exact/regex matching via `grep`, and browses project structure via `list`.
- Session management (activate/deactivate) is automated, not exposed as a tool.
- Health monitoring is server-internal and affects search response metadata.

#### search

Hybrid semantic and keyword search across project content, libraries, and rules.

```typescript
search({
    query: string,                      // Required: search query
    collection?: "projects" | "libraries" | "rules", // default: "projects"
    mode?: "hybrid" | "semantic" | "keyword", // default: "hybrid"
    limit?: number,                     // default: 10
    scoreThreshold?: number,           // Minimum similarity score (0-1, default: 0.3)
    // Collection-specific scope filters (see below)
    scope?: string,                     // Scope within collection
    branch?: string,                    // For projects: branch filter
    project_id?: string,               // For projects: specific project
    library_name?: string,             // For libraries: specific library
    // Content type filters
    file_type?: string,                // Filter by document type (see below)
    tag?: string,                      // Filter by concept tag (exact match on concept_tags)
    tags?: string[],                   // Filter by multiple concept tags (OR logic)
    // Cross-collection options
    include_libraries?: boolean,       // Also search libraries collection
    // Exact search options
    exact?: boolean,                   // Use exact substring search instead of semantic search (default: false)
    contextLines?: number,             // Lines of context before/after matches in exact mode (default: 0)
    // Code intelligence options
    includeGraphContext?: boolean,     // Include code relationship graph context (callers/callees) for matched symbols (default: false)
    // Structural filters
    component?: string,               // Filter by project component (e.g., "daemon", "daemon.core"). Supports prefix matching.
    pathGlob?: string,                // File path glob filter (e.g., "**/*.rs", "src/**/*.ts")
)
```

**Modes:**

- `hybrid`: Semantic + keyword search (default)
- `semantic`: Pure vector similarity
- `keyword`: Keyword/exact matching

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
| `rules`     | `all`, `global`, `project` | `project` = current project's rules             |
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
    collection?: "projects" | "libraries" | "rules", // default: "projects"
    metadata?: Record<string, unknown>, // Metadata filters
    limit?: number,                    // default: 10
    offset?: number,                   // default: 0, for pagination
});
```

**Use case:** Retrieving specific documents by ID or metadata filter for chunk-by-chunk access without overwhelming context. Use `search` for discovery by query, and `retrieve` for direct access when you know the document ID or metadata.

#### rules

Manage behavioral rules (persistent preferences).

```typescript
rules({
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
- `list`: List rules (implemented as a search query against the rules collection)

**Implementation note:** The `list` action is internally a semantic search with scope filtering. This allows consistent behavior with other collection queries while providing a simpler interface for rule management.

**Uniqueness:** `label` + `scope` must be unique. A global rule and a project rule can have the same label.

**LLM Generation Guidelines:**

When creating rules, the LLM should generate metadata fields following these constraints:

| Field | Constraints | Examples |
|-------|-------------|----------|
| `label` | Max 15 chars, format: `word-word-word` (lowercase, hyphen-separated) | `prefer-uv`, `use-pytest`, `strict-types`, `no-mock-fs` |
| `title` | Max 50 chars, human-readable summary | "Use uv instead of pip for Python packages" |
| `tags` | Max 5 tags, max 20 chars each | `["python", "tooling"]`, `["testing", "best-practice"]` |

**Conversational Flow Examples:**

```
User: "For future reference, always use uv instead of pip"
→ LLM calls: rules(action="add", label="prefer-uv", title="Use uv for Python packages",
                    content="Always use uv instead of pip for Python package management...",
                    tags=["python", "tooling"])

User: "Remember to run tests before committing"
→ LLM calls: rules(action="add", label="pre-commit-tests", title="Run tests before commits",
                    content="Always run the test suite before committing changes...",
                    tags=["git", "testing"])

User: "Actually, let me update that rule about testing"
→ LLM calls: rules(action="update", label="pre-commit-tests", title="Run tests before commits",
                    content="Run the full test suite AND linting before committing...")
```

**Label Generation Strategy:**

1. Extract key concepts from user's instruction
2. Form 2-3 word hyphenated identifier
3. Keep descriptive but concise
4. Use action verbs when appropriate: `use-`, `prefer-`, `avoid-`, `no-`

#### store

Store content or register a project. The `type` parameter determines the operation mode.

```typescript
store({
    type?: "library" | "url" | "scratchpad" | "project", // What to store (default: "library")
    // Common parameters
    content?: string,                  // Text content (required for type "library")
    title?: string,                    // Content title (for type "library")
    url?: string,                      // Source URL (for web content)
    filePath?: string,                 // Source file path
    metadata?: Record<string, unknown>, // Additional metadata
    // Library-specific parameters
    libraryName?: string,              // Library identifier (required for type "library" unless forProject is true)
    forProject?: boolean,              // When true, store to libraries collection scoped to current project. libraryName defaults to "project-refs".
    sourceType?: "user_input" | "web" | "file" | "scratchbook" | "note", // Source type (default: "user_input")
    // Scratchpad-specific parameters
    tags?: string[],                   // Tags for scratchpad entries
    // Project-specific parameters
    path?: string,                     // Absolute path to project directory (required for type "project")
    name?: string,                     // Display name (defaults to directory name, for type "project")
})
```

**Understanding `type` vs collection:** The `type` parameter describes the **nature of the input** (a document, a URL to fetch, a note, a project registration), not the destination collection. The destination collection is determined by context and other parameters.

**Type: `"library"` (default) — Store reference documentation:**

Stores content to the `libraries` collection under a named library. Requires `content` and `libraryName` (unless `forProject` is true, in which case `libraryName` defaults to `"project-refs"`). This type should only be used when the user **explicitly instructs** the LLM to store into a named library — not autonomously.

**Type: `"url"` — Fetch and ingest a web page:**

Queues a URL for daemon-side fetch and ingestion. The destination collection depends on context: stores to `libraries` if `libraryName` is provided (for adding URLs to a named library), otherwise stores to `scratchpad` for ad-hoc reference. URLs (webpages, websites) can also be part of a tracked library folder as webarchive files.

**Type: `"scratchpad"` — Persistent notes/scratch space:**

Stores content to the `scratchpad` collection. Supports both global and project-scoped entries. Global entries can be further segregated by user-defined themes or keywords. This is the **preferred destination** for ad-hoc content ingestion by the server (strings, files, URLs). Supports `tags` for categorization.

**Type: `"project"` — Register a project directory:**

Registers a new project with the daemon for file watching and ingestion. Uses `register_if_new: true` so the daemon will create the project in `watch_folders` if it doesn't already exist. The server can also activate and keep-active existing projects. Returns `{ success, project_id, created, is_active, message }`.

**Collection routing summary:**

| Type | Destination | Condition |
|------|------------|-----------|
| `library` | `libraries` | Always (explicit user instruction only) |
| `url` | `libraries` | When `libraryName` provided |
| `url` | `scratchpad` | Default (ad-hoc reference) |
| `scratchpad` | `scratchpad` | Always |
| `project` | N/A | Registers project with daemon |

**Note:** The server cannot store content to the `projects` collection — project content is ingested exclusively by the daemon via file watching. Behavioral rules use the dedicated `rules` tool. Libraries are collections of reference information (books, documentation, papers, websites) — NOT programming libraries (use context7 MCP for those).

#### grep

Search code with exact substring or regex pattern matching via FTS5 trigram index.

```typescript
grep({
    pattern: string,                    // Required: search pattern (exact substring or regex)
    regex?: boolean,                    // Treat pattern as regex (default: false)
    caseSensitive?: boolean,            // Case-sensitive matching (default: true)
    pathGlob?: string,                  // File path glob filter (e.g., "**/*.rs", "src/**/*.ts")
    scope?: "project" | "all",          // Search scope (default: "project")
    contextLines?: number,              // Lines of context before/after each match (default: 0)
    maxResults?: number,                // Maximum results to return (default: 1000)
    branch?: string,                    // Filter by branch name
    projectId?: string,                 // Specific project ID to search
})
```

**Use case:** Finding exact code patterns, function calls, imports, or string literals across the indexed codebase. Unlike `search` which uses semantic similarity, `grep` finds exact text matches. Results include file path, line number, and matched content.

#### list

List project files and folder structure. Shows only indexed files (excludes gitignored, node_modules, etc).

```typescript
list({
    path?: string,                      // Subfolder relative to project root (default: root)
    depth?: number,                     // Max directory depth (default: 3, max: 10)
    format?: "tree" | "summary" | "flat", // Output format (default: tree)
    fileType?: string,                  // Filter: "code", "text", "data", "config", "build", "web"
    language?: string,                  // Filter by programming language (e.g., "rust", "typescript")
    extension?: string,                 // Filter by file extension (e.g., "rs", "ts")
    pattern?: string,                   // Glob pattern on relative path (e.g., "**/*.test.ts")
    includeTests?: boolean,             // Include test files (default: true)
    limit?: number,                     // Max entries returned (default: 200, max: 500)
    projectId?: string,                 // Specific project ID (default: current project)
    component?: string,                 // Filter by component (e.g., "daemon", "daemon.core")
})
```

**Formats:**

- `tree`: Hierarchical directory tree with file counts per directory
- `summary`: High-level project overview with component breakdown, language statistics, and directory structure
- `flat`: Simple flat list of file paths matching filters

**Use case:** Understanding project structure before diving into code. Start with `format: "summary"` for an overview, then use `path` to drill into specific directories.

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
   - **If not registered:** Server logs a warning ("Project not registered, use `wqm project add` or the `store` tool with `type: \"project\"` to register") and continues without activation — search and rules tools still work, but file watching is not started

2. **Start heartbeat:**
   - Periodic heartbeat with daemon to prevent timeout-based deactivation
   - Only started if project was successfully activated in step 1

#### On Session End (server.onclose)

1. **Project deactivation:**
   - Server sends `DeprioritizeProject` to daemon
   - Daemon sets `is_active = false` for the project

2. **Process cleanup:**
   - Daemon shuts down any spawned processes for that project (e.g., LSP servers)

#### Rules Injection

Rule injection is handled via the `rules` MCP tool:
- Claude reads behavioral rules by calling the `rules` tool with `action: "list"`
- Rules are NOT automatically injected at session start (that would require external hooks)

**Optional Enhancement via Claude Code Hooks:** For automatic rules injection, users can configure a `SessionStart` hook in their `~/.claude/settings.json`:
```json
{
  "hooks": {
    "SessionStart": [{
      "hooks": [{
        "type": "command",
        "command": "wqm rules inject"
      }]
    }]
  }
}
```
This is external to the MCP server and optional.

#### Rules and Project ID Changes

When a project is renamed or its `project_id` changes (e.g., due to disambiguation when a second clone is detected), the rules records in Qdrant must have their `project_id` field updated to maintain association.

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

**The legacy `manage` tool is completely removed.** Rule operations use the dedicated `rules` tool.

### gRPC Services

The daemon exposes 7 gRPC services on port 50051.

#### SystemService (10 RPCs)

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
| `PauseAllWatchers`    | CLI      | Pause all file watchers         | Production |
| `ResumeAllWatchers`   | CLI      | Resume all paused watchers      | Production |
| `RebuildIndex`        | CLI      | Rebuild FTS5 search index       | Production |

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
- Sparse vectors use BM25 algorithm with IDF weighting: `IDF * (k1 * tf) / (tf + k1)` where `IDF = ln((N - df + 0.5) / (df + 0.5))`, clamped at 0. IDF vocabulary and corpus statistics are persisted in `sparse_vocabulary` and `corpus_statistics` SQLite tables (schema v15). Per-collection BM25 instances maintain independent document frequency counts.

#### TextSearchService (2 RPCs)

Exact substring and regex code search via FTS5 trigram index. Used by the `grep` MCP tool.

| Method         | Used By | Purpose                            | Status     |
| -------------- | ------- | ---------------------------------- | ---------- |
| `Search`       | MCP     | Exact/regex pattern matching       | Production |
| `CountMatches` | MCP     | Count matches without full results | Production |

**Result caching:** TextSearchService maintains a short-lived cache (TTL of a few seconds) keyed by the search parameters. This allows a CountMatches call to warm the cache so that a subsequent Search with the same parameters can return full results without re-executing the query, and vice versa. The cache is invalidated after the TTL expires.

#### ProjectService (7 RPCs)

Multi-tenant project lifecycle and session management.

| Method                | Used By  | Purpose                          | Status                       |
| --------------------- | -------- | -------------------------------- | ---------------------------- |
| `RegisterProject`     | MCP, CLI | Re-activate or register project  | **Production (enqueue-only)** |
| `DeleteProject`       | CLI      | Delete project and all data      | **Production (enqueue-only)** |
| `DeprioritizeProject` | MCP      | Deactivate project session       | **Production (direct gRPC)** |
| `GetProjectStatus`    | MCP      | Get project status               | Production                   |
| `ListProjects`        | CLI      | List all registered projects     | Production                   |
| `Heartbeat`           | MCP      | Keep session alive (60s)         | Production                   |
| `RenameTenant`        | CLI      | Rename tenant_id across collections | Production                |

**Enqueue-Only Pattern (RegisterProject, DeleteProject):**
- These handlers do NOT perform direct SQLite mutations
- Instead, they enqueue `(Tenant, Add)` or `(Tenant, Delete)` to the unified queue
- The queue processor handles all database writes and Qdrant operations
- This ensures consistency, crash recovery, and single-writer ownership

**Registration Policy:**
- `RegisterProject` accepts a `register_if_new` boolean field (default: `false`)
  - `register_if_new=false` (MCP default): Only re-activates existing `watch_folders` entries. Returns error if project not found.
  - `register_if_new=true` (CLI-initiated): Enqueues `(Tenant, Add)` if project doesn't exist. Queue processor creates watch_folder entry and triggers initial scan.
- MCP servers call `RegisterProject` on startup with `register_if_new=false` — they only re-activate known projects
- CLI `wqm project add` calls `RegisterProject` with `register_if_new=true` to create new entries
- For existing high-priority projects, synchronous activation (direct gRPC) is preserved for MCP server flow

**Deletion Policy:**
- `DeleteProject` enqueues `(Tenant, Delete)` and returns `status="queued"`
- Queue processor performs full cascade: Qdrant point deletion → SQLite cleanup (qdrant_chunks, tracked_files, watch_folders)

**Session Management:**
- `Heartbeat` must be called periodically (within 60s timeout) to maintain session
- `DeprioritizeProject` is called when MCP server stops

#### GraphService (7 RPCs)

Code relationship graph queries and algorithms. Operates on the embedded graph database (`graph.db`).

| Method               | Used By | Purpose                                    | Status     |
| -------------------- | ------- | ------------------------------------------ | ---------- |
| `QueryRelated`       | CLI     | Traverse related nodes within N hops       | Production |
| `ImpactAnalysis`     | CLI     | Find nodes affected by changing a symbol   | Production |
| `GetGraphStats`      | CLI     | Node/edge counts by type                   | Production |
| `ComputePageRank`    | CLI     | Rank nodes by structural importance        | Production |
| `DetectCommunities`  | CLI     | Label propagation community detection      | Production |
| `ComputeBetweenness` | CLI     | Betweenness centrality scores              | Production |
| `MigrateGraph`       | CLI     | Migrate data between graph backends        | Production |

**QueryRelated:**
- Traverses the graph from a starting node up to `max_hops` depth (1-5)
- Optional `edge_types` filter (CALLS, IMPORTS, CONTAINS, USES_TYPE, EXTENDS, IMPLEMENTS)
- Returns `TraversalNode` entries with symbol name, type, file path, edge type, and depth
- Uses recursive CTEs (SQLite backend) for efficient multi-hop traversal

**ImpactAnalysis:**
- Finds all nodes that would be affected by modifying a symbol
- Searches by `symbol_name` with optional `file_path` to narrow scope
- Returns nodes grouped by distance (direct callers at distance 1, indirect at 2+)

**ComputePageRank:**
- Configurable damping factor (default 0.85), max iterations (default 100), convergence tolerance (default 1e-6)
- Optional `top_k` to limit results and `edge_types` filter
- Returns ranked entries with score, symbol name, type, and file path

**DetectCommunities:**
- Label propagation algorithm for detecting code communities
- Configurable max iterations (default 50) and minimum community size (default 2)
- Returns community groups with member lists

**ComputeBetweenness:**
- Betweenness centrality identifies bottleneck/bridge nodes in the graph
- Sampled BFS for large graphs (`max_samples` parameter, 0 = all nodes)
- Optional `top_k` and `edge_types` filters

**MigrateGraph:**
- Exports nodes/edges from one backend, imports into another
- Supports `sqlite` and `ladybug` backend identifiers
- Optional `tenant_id` for per-tenant migration, `batch_size` for import batching
- Returns export/import counts and validation (nodes_match, edges_match)

---

