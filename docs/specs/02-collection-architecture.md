## Collection Architecture

**Reference:** [ADR-001](../adr/ADR-001-canonical-collection-architecture.md)

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

Project IDs (`project_id`, also referred to as `tenant_id`) are 12-character hex hashes that uniquely identify a project clone. The system handles multiple clones of the same repository, filesystem moves, and local projects gaining git remotes.

#### Tenant ID Precedence

The `tenant_id` is determined by the following precedence rules:

1. **If git remote exists** (takes precedence): `tenant_id = hash(normalized_remote_url)` — remote-based identity
2. **If no remote**: `tenant_id = hash("path_" + filesystem_path)` — path-based identity, prefixed with `path_` to distinguish from remote-based IDs

**Key consequence:** Multiple clones of the same remote produce the **same `tenant_id`**. This enables content-addressed deduplication — identical file content across clones shares Qdrant points naturally. See [Cross-Instance Deduplication](#cross-instance-deduplication).

**When a local project gains a remote:** The system performs a cascade rename from the path-based `tenant_id` to the remote-based `tenant_id`. See [Local Project Gains Remote](#local-project-gains-remote) and [Cascade Rename Mechanism](#cascade-rename-mechanism).

#### Core Algorithm

```rust
fn calculate_project_id(project_root: &Path, disambiguation_path: Option<&str>) -> String {
    use sha2::{Sha256, Digest};

    let git_remote = get_git_remote_url(project_root);

    if let Some(remote) = git_remote {
        // Remote-based identity (takes precedence)
        let normalized = normalize_git_url(&remote);
        let input = match disambiguation_path {
            Some(path) => format!("{}|{}", normalized, path),
            None => normalized,
        };
        let hash = Sha256::digest(input.as_bytes());
        return hex::encode(&hash[..6]) // 12 hex chars = 6 bytes
    }

    // Path-based identity (prefixed to distinguish from remote-based)
    let canonical = project_root.canonicalize().unwrap();
    let input = format!("path_{}", canonical.display());
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

- Use `path_` prefix + **canonical filesystem path** for identity
- State database stores full `project_path` for move detection
- The `path_` prefix ensures local project IDs never collide with remote-based IDs

```
/Users/chris/experiments/my-test-project → project_id = sha256("path_/Users/chris/experiments/my-test-project")[:12]
```

#### Branch Handling

- **Branch-agnostic project_id**: All branches share the same `project_id`
- **Branch-scoped point IDs**: Each Qdrant point ID includes the branch, ensuring no collisions across branches
- **Branch stored as metadata**: `branch` field in Qdrant payload (for filtering)
- **Default search**: Returns results from all branches
- **Filtered search**: Use `branch="main"` to scope to specific branch

**Base Point Identity Model:**

The system uses a two-level identity scheme built on the **base point** concept:

```
base_point = hash(tenant_id, branch, relative_path, file_hash)
```

- `tenant_id`: derived from git remote URL hash (git) or path hash (non-git)
- `branch`: current branch name (git) or `"default"` (non-git)
- `relative_path`: file path relative to project root (portable across moves/clones). Absolute path stored in Qdrant payload for display and file access.
- `file_hash`: SHA256 of file content, or git blob SHA for git-tracked projects (via `git hash-object <file>`)

The base point identifies a specific **version** of a specific file. It is deterministic (same inputs always produce the same base point) and content-aware (different file content = different base point).

**Qdrant point ID** (one per chunk):

```
point_id = hash(base_point, chunk_index)
```

**Search DB file_id** uses the base point as its unique identity (one per file version). See [Search DB (FTS5)](12-configuration.md#search-db-fts5) for details.

All components are always known at write time. This ensures each branch gets independent points with no collision risk, and enables content-addressed deduplication across watch folder instances (see [Cross-Instance Deduplication](#cross-instance-deduplication)).

**Branch lifecycle:**

- **Branch detection**: Via the git watcher (Layer 2 — see [Two-Layer Watching Architecture](06-file-watching.md#two-layer-watching-architecture)). A `.git/HEAD` change triggers reflog parsing to identify the operation type.
- **Branch switch protocol**: On branch switch detection:
  1. Parse reflog to get old/new commit SHAs
  2. `git diff-tree --name-status old_sha new_sha` to get the exact list of changed files
  3. For changed files (M = modified): apply reference-counting deletion logic for old version, create new version
  4. For files only on new branch (A = added): ingest as new
  5. For files only on old branch (D = removed): apply reference-counting deletion logic
  6. For **unchanged files**: update `branch` in `tracked_files` only (no re-ingestion needed)
  7. Update `watch_folders.last_commit_hash = new_sha`
- **First branch switch optimization**: When switching to a branch with no existing `tracked_files` entries, batch-copy entries from the old branch to the new branch (updating branch and file_hash where files differ). This avoids full re-ingestion.
- **New branch detected (non-switch)**: For each file, check if identical `file_hash` exists on the source branch (via `tracked_files`). If yes: create new point with branch-qualified ID and **copy the vector** from the existing point (no re-embedding). If no: embed and create new point normally.
- **Branch deleted**: Delete all documents with `branch="deleted_branch"` from Qdrant (trivial — filter by branch in payload). Apply reference-counting deletion to avoid removing points still referenced by other watch folder instances.
- **Branch renamed**: Treat as delete + create (Git doesn't track renames)

**Rationale for hybrid approach (base point identity + reference counting):**
- Content-addressed: identical content across instances = shared points, divergent content = separate points
- Precise change detection via `git diff-tree` eliminates full re-scanning
- Reference counting prevents data loss when multiple watch folders share points
- Trivial deletion and search scoping (filter by branch payload)
- Eliminates embedding CPU cost for identical content (vector copy is a memcpy)
- Storage is bounded: typically 3-5 active branches, not hundreds
- Content hash lookup for vector reuse is a cheap SQLite query

#### Local Project Gains Remote

When a local project (identified by path-based `tenant_id`) gains a git remote:

1. Daemon detects `.git/config` change during watching (Layer 1 file watcher)
2. If project at same location:
   - Compute new remote-based `tenant_id` from normalized remote URL (per [Tenant ID Precedence](#tenant-id-precedence))
   - Execute queue-mediated cascade rename from `path_`-based ID to remote-based ID (see [Cascade Rename Mechanism](#cascade-rename-mechanism))
   - This cascades through `watch_folders`, `tracked_files`, `qdrant_chunks`, Qdrant payloads, and search DB entries
3. No re-ingestion required — only the `tenant_id` changes, all content remains valid

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

    -- Git tracking
    is_git_tracked INTEGER DEFAULT 0 CHECK (is_git_tracked IN (0, 1)),
    last_commit_hash TEXT,                  -- HEAD commit SHA (NULL if not git-tracked)

    -- Project-specific (NULL for libraries)
    git_remote_url TEXT,                    -- Normalized remote URL
    remote_hash TEXT,                       -- sha256(remote_url)[:12] for grouping duplicates
    disambiguation_path TEXT,               -- Path suffix for clone disambiguation
    is_active INTEGER DEFAULT 0,            -- Activity flag (inherited by subprojects via junction table)
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
    last_scan TEXT                          -- NULL if never scanned
);

-- Index for finding duplicates (same remote, different paths)
CREATE INDEX idx_watch_remote_hash ON watch_folders(remote_hash);
-- Index for active project lookups (used in queue priority calculation)
CREATE INDEX idx_watch_active ON watch_folders(is_active);
-- Index for daemon polling
CREATE INDEX idx_watch_updated ON watch_folders(updated_at);
-- Index for enabled watches
CREATE INDEX idx_watch_enabled ON watch_folders(enabled);
```

**Note:** Submodule relationships are stored in the separate `watch_folder_submodules` junction table (see [Git Submodules](06-file-watching.md#git-submodules)), not via a FK on this table.

**Key columns:**

- `tenant_id`: `project_id` for projects, library name for libraries
- `is_git_tracked`: Whether this is a git-tracked project (enables Layer 2 git watcher)
- `last_commit_hash`: HEAD commit SHA for git-tracked projects (used for submodule pin comparison and dedup assessment)
- `is_active`: Activity flag - **inherited by all subprojects** via the junction table (see below)
- `last_activity_at`: Timestamp - **synced across parent and all subprojects**
- `is_archived`: Archive flag. Archived projects stop watching/ingesting but remain **fully searchable** in Qdrant. No search exclusion — archived projects are fair game for code exploration. User can un-archive. Archiving preserves junction table entries (historical fact, no detaching).
- `library_mode`: Only for libraries (`sync` = full sync, `incremental` = no deletes)

**Activity Inheritance for Subprojects:**

When a project has submodules (subprojects), they share activity state with the parent via the `watch_folder_submodules` junction table. Any activity on the parent OR any subproject updates the entire group:

```sql
-- On RegisterProject or any activity (given a watch_id):
UPDATE watch_folders
SET is_active = 1, last_activity_at = datetime('now')
WHERE watch_id = :watch_id
   OR watch_id IN (SELECT child_watch_id FROM watch_folder_submodules WHERE parent_watch_id = :watch_id)
   OR watch_id IN (SELECT parent_watch_id FROM watch_folder_submodules WHERE child_watch_id = :watch_id)
   OR watch_id IN (
       SELECT ws2.child_watch_id FROM watch_folder_submodules ws1
       JOIN watch_folder_submodules ws2 ON ws1.parent_watch_id = ws2.parent_watch_id
       WHERE ws1.child_watch_id = :watch_id
   );
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
    relative_path TEXT,                     -- Alias for file_path (for base_point computation clarity)
    branch TEXT,                            -- Git branch at ingestion (NULL for libraries)
    collection TEXT NOT NULL DEFAULT 'projects', -- Destination collection (format-based routing)

    -- Identity
    base_point TEXT,                        -- hash(tenant_id, branch, relative_path, file_hash)

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

    -- Library routing
    incremental INTEGER DEFAULT 0 CHECK (incremental IN (0, 1)),
                                            -- "Do not delete" flag for library-routed project files

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
-- Index for base_point lookups (reference counting, dedup)
CREATE INDEX idx_tracked_files_base_point ON tracked_files(base_point);
```

**Key columns:**

- `file_path`: **Relative** to `watch_folders.path`. When a project moves, only the watch_folder root path changes; tracked_files entries remain valid.
- `base_point`: Computed as `hash(tenant_id, branch, relative_path, file_hash)`. Identifies a specific version of a specific file. Used for reference counting across watch folder instances and as the identity key in the search DB. See [Base Point Identity Model](#base-point-identity-model).
- `collection`: Destination Qdrant collection for this file. Normally matches `watch_folders.collection`, but format-based routing can override it (e.g., a `.pdf` in a project folder routes to `libraries`).
- `file_hash`: SHA256 of file content at ingestion (or git blob SHA for git-tracked projects). Used for recovery (detect changes during daemon downtime, including git checkout/rsync which change content without changing mtime) and update optimization (skip re-embedding when content hasn't changed).
- `file_mtime`: Filesystem modification time at ingestion. Used for fast change detection during recovery.
- `lsp_status` / `treesitter_status`: Track processing pipeline state per file. Enables re-processing files that had partial or failed enrichment.
- `chunk_count`: Cached count of Qdrant points for this file (also available via `qdrant_chunks` table).
- `incremental`: "Do not delete" flag for library-routed project files. When set, file deletions do not propagate — only full project deletion cascades through this flag.

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
  "base_point": "hash...",        // hash(tenant_id, branch, relative_path, file_hash)
  "relative_path": "src/main.rs", // Path relative to project root (portable)
  "absolute_path": "/Users/dev/repo/src/main.rs", // For display and file access
  "file_hash": "sha256...",       // SHA256 of file content (or git blob SHA)
  "file_type": "code", // code|doc|test|config|note|artifact
  "language": "rust",
  "branch": "main",
  "commit_hash": "abc123def...",  // HEAD commit SHA at ingestion time
  "symbols": ["MyClass", "my_function"],
  "chunk_index": 0,
  "total_chunks": 5,
  "created_at": "2026-01-28T12:00:00Z",
  // Keyword/tag extraction fields (populated by daemon pipeline)
  "concept_tags": ["async-runtime", "error-handling"],  // Semantic concept tags
  "structural_tags": ["language:rust", "framework:tokio"],  // Structural metadata tags
  "keywords": ["tokio", "spawn", "executor"],  // Raw extracted keywords
  "keyword_baskets": {                          // Tag-to-keywords mapping
    "async-runtime": ["tokio", "spawn", "executor", "async"]
  }
}
```

**Note:** File processing metadata (`file_mtime`, chunk details, LSP/tree-sitter status) is tracked in the SQLite `tracked_files` and `qdrant_chunks` tables. However, key identity fields (`base_point`, `relative_path`, `absolute_path`, `file_hash`, `commit_hash`) are stored in Qdrant payloads to enable disaster recovery — `wqm admin recover-state` can rebuild state.db from Qdrant payloads alone. See [Tracked Files Table](#tracked-files-table) and [Disaster Recovery](12-configuration.md#disaster-recovery).

**Libraries Collection:**

The libraries collection uses a **parent-unit architecture** to provide rich context for search results. Each library document is split into two types of records:

1. **Parent records** (no vectors): Store document/unit-level metadata and full text for context expansion
2. **Child chunks** (with vectors): Store searchable chunks that reference their parent via `parent_unit_id`

**Parent Record Payload (no vectors):**

```json
{
  "library_name": "numpy",              // Required, indexed (is_tenant=true)
  "doc_id": "550e8400-e29b-...",         // UUID v5 from library_name + path
  "doc_title": "NumPy User Guide",       // Extracted via priority cascade
  "doc_authors": ["Travis Oliphant", "..."], // From embedded metadata
  "doc_source": "/docs/numpy-user-guide.pdf", // Original file path
  "doc_fingerprint": "abc123def456...",  // SHA256 of file bytes
  "doc_type": "page_based",              // "page_based" or "stream_based"
  "source_format": "pdf",                // pdf|docx|epub|html|markdown|text
  "unit_type": "document",               // "document" or "code_block"
  "unit_text": "Full document text...", // Complete text for context
  "locator": {                           // Format-specific location
    "page": 1,
    "section": "Introduction"
  },
  "created_at": "2026-02-15T12:00:00Z"
}
```

**Child Chunk Payload (with vectors):**

```json
{
  "library_name": "numpy",              // Required, indexed (is_tenant=true)
  "doc_id": "550e8400-e29b-...",         // Same as parent
  "doc_title": "NumPy User Guide",       // Denormalized for filtering
  "parent_unit_id": "parent-point-id",   // Points to parent record
  "chunk_text_raw": "Array operations allow...", // Original chunk text
  "chunk_text_indexed": "## Introduction\nArray operations allow...", // With heading context
  "char_start": 1024,                    // Character offset in unit_text
  "char_end": 1536,                      // End character offset
  "extractor": "pdf_extract",            // Extractor used
  "source_format": "pdf",                // Same as parent
  "chunk_index": 0,                      // Chunk sequence number
  "created_at": "2026-02-15T12:00:00Z",
  // Keyword/tag extraction fields (populated by daemon pipeline)
  "concept_tags": ["numerical-computing", "array-operations"],
  "structural_tags": ["domain:science", "format:tutorial"],
  "keywords": ["array", "ndarray", "broadcasting"],
  "keyword_baskets": {
    "numerical-computing": ["ndarray", "broadcasting", "vectorization"]
  }
}
```

**Provenance fields** (present on both parent and child records):
- `doc_id` - Unique document identifier (UUID v5)
- `doc_title` - Document title (via title extraction cascade)
- `doc_authors` - Document authors (from metadata)
- `doc_source` - Original file path
- `doc_fingerprint` - SHA256 hash for change detection
- `doc_type` - Document family: `page_based` or `stream_based`
- `source_format` - Actual file format (pdf, docx, epub, html, markdown, text)

**Search context expansion:**
When `expandContext: true` is set on the `search` tool, the MCP server batch-fetches parent records from Qdrant and includes them in the `parent_context` field of search results, providing full document/unit-level context.

---

