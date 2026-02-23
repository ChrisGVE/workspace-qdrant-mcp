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

**Logs use OS-canonical paths** (see [Logging and Observability](13-deployment.md#logging-and-observability)):

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

**Note:** The `watching` section replaces the old `patterns`/`ignore_patterns` approach. The allowlist (`allowed_extensions` + `allowed_filenames`) is the primary ingestion gate. See the [File Type Allowlist](06-file-watching.md#file-type-allowlist) section for the complete categorized list and ingestion gate layering. The full default list is embedded at build time from `assets/default_configuration.yaml`.

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

**Future consideration:** GraphRAG patterns could leverage the distance matrix to build code intelligence graphs (e.g., "files that are semantically related" or "functions that co-occur in similar contexts"). See the [Future Development](14-future-development.md#future-development-wishlist-not-yet-scoped) section for detailed research findings.

### SQLite Database

**Path:** `~/.workspace-qdrant/state.db`

**Owner:** Rust daemon (memexd) - see [ADR-003](../adr/ADR-003-daemon-owns-sqlite.md)

**state.db Core Tables (17):**

| Table            | Purpose                                                           | Used By          |
| ---------------- | ----------------------------------------------------------------- | ---------------- |
| `schema_version` | Schema version tracking                                           | Daemon           |
| `unified_queue`  | Write queue with per-destination state machine                    | MCP, CLI, Daemon |
| `watch_folders`  | Unified table for projects and libraries                          | MCP, CLI, Daemon |
| `watch_folder_submodules` | Many-to-many junction table for submodule relationships | Daemon, CLI      |
| `tracked_files`  | Authoritative file inventory with base_point identity             | Daemon (write), CLI (read) |
| `qdrant_chunks`  | Qdrant point tracking per file chunk (child of tracked_files)     | Daemon only      |
| `memory_mirror`  | Write-through copy of memory collection (for reverse recovery)    | Daemon           |
| `search_events`  | Search instrumentation logs (all tools)                           | MCP, CLI, External |
| `resolution_events` | Document open/expand events after search                       | External (future) |
| `sparse_vocabulary` | BM25 IDF term-to-document-count mapping (schema v15)           | Daemon only      |
| `corpus_statistics` | BM25 IDF total document counts per collection (schema v15)     | Daemon only      |
| `keywords`       | Per-document keyword records with scores (schema v16)             | Daemon (write), MCP (read) |
| `tags`           | Per-document tag records with type and diversity scoring (v16)    | Daemon (write), MCP (read) |
| `keyword_baskets`| Keyword-to-tag assignments for query expansion (v16)              | Daemon (write), MCP (read) |
| `canonical_tags` | Deduplicated cross-document tag hierarchy nodes (v16)             | Daemon only      |
| `tag_hierarchy_edges` | Parent-child relationships between canonical tags (v16)      | Daemon only      |

**search.db Tables (2):**

| Table            | Purpose                                                           | Used By          |
| ---------------- | ----------------------------------------------------------------- | ---------------- |
| `file_metadata`  | Per-file-version metadata keyed by base_point                     | Daemon (write), MCP/CLI (read) |
| `code_lines`     | FTS5 virtual table for full-text code search                      | Daemon (write), MCP/CLI (read) |

See [Search DB (FTS5)](#search-db-fts5) for schemas.

**Note:** The `watch_folders` table consolidates what were previously separate `registered_projects`, `project_submodules`, and `watch_folders` tables. Submodule relationships are now stored in the `watch_folder_submodules` junction table. See [Watch Folders Table (Unified)](02-collection-architecture.md#watch-folders-table-unified) and [Git Submodules](06-file-watching.md#git-submodules) for schemas.

**Note:** `tracked_files` + `qdrant_chunks` together form the authoritative file inventory, replacing the need to scroll Qdrant for file listings. See [Tracked Files Table](02-collection-architecture.md#tracked-files-table) and [Qdrant Chunks Table](02-collection-architecture.md#qdrant-chunks-table) for schemas.

#### Schema Version Table

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now'))
);
```

The daemon checks this table on startup and runs migrations if needed. Other components must NOT modify this table.

**Other components (MCP, CLI) may read/write to tables but must NOT create tables or run migrations.**

#### Search DB (FTS5)

The search DB (`search.db`) is a **separate SQLite database** from `state.db`, dedicated to full-text search via FTS5 virtual tables. It provides the `workspace-qdrant/grep` MCP tool with fast pattern-matching search across all indexed code.

**Path:** `~/.workspace-qdrant/search.db`

**Owner:** Rust daemon (memexd) — same ownership model as state.db

##### file_metadata Table

Stores per-file-version metadata, keyed by the base point:

```sql
CREATE TABLE IF NOT EXISTS file_metadata (
    metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
    base_point TEXT NOT NULL UNIQUE,    -- hash(tenant_id, branch, relative_path, file_hash)
    tenant_id TEXT NOT NULL,
    branch TEXT,
    relative_path TEXT NOT NULL,
    absolute_path TEXT,                 -- For display and file access
    file_hash TEXT NOT NULL,
    language TEXT,
    file_type TEXT,
    line_count INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX idx_file_metadata_tenant ON file_metadata(tenant_id);
CREATE INDEX idx_file_metadata_base_point ON file_metadata(base_point);
```

##### code_lines FTS5 Virtual Table

Full-text search over individual code lines, referencing `file_metadata`:

```sql
CREATE VIRTUAL TABLE code_lines USING fts5(
    line_text,
    metadata_id UNINDEXED,            -- FK to file_metadata.metadata_id
    line_number UNINDEXED,
    content='code_lines_content',
    content_rowid='rowid'
);
```

##### Regex Search: Hybrid FTS5 + grep-searcher Dispatch

Regex search uses a hybrid dispatch strategy. For most queries, FTS5 trigram pre-filtering narrows candidates efficiently. But for high-frequency patterns (e.g., `\.(await|unwrap|expect)\b` producing 10K+ candidates), SQLite row-fetch overhead (~3μs/row) dominates. In those cases, the search engine delegates to ripgrep's `grep-searcher` crate for SIMD-accelerated file scanning.

**Dispatch flow:**

1. Extract literal substrings from the regex for FTS5 pre-filtering
2. Run a lightweight FTS5-only probe: `SELECT rowid FROM code_lines_fts WHERE content MATCH ?1 LIMIT 1 OFFSET ?2` (no JOINs, sub-millisecond)
3. If candidates exceed threshold (5,000) → delegate to `grep-searcher` module which scans source files directly via `file_metadata` paths
4. Otherwise → stream FTS5 candidates with Rust regex verification

**Dependencies:** `grep-searcher`, `grep-regex`, `grep-matcher` (ripgrep's library crates)

**Module:** `src/rust/daemon/core/src/grep_search.rs`

The grep path uses `tokio::task::spawn_blocking` for synchronous file I/O, supports context lines via grep-searcher's built-in `before_context`/`after_context`, and applies the same glob/scope filters as the FTS5 path. The `SearchResults.search_engine` field indicates which path was used (`"fts5"` or `"grep"`).

##### FTS5 Query Optimization: Redundant AND Elimination

When affix merging prepends a prefix to all alternation branches (e.g., `pub (fn|struct|enum|trait|type) \w+` → branches `"pub fn "`, `"pub struct "`, etc.), the standalone mandatory term `"pub "` is redundant since it's a prefix of every branch. The query builder detects this and omits the redundant AND clause, reducing FTS5 intersection work.

##### Consistency with Qdrant

The search DB follows the same reference-counting deletion logic as Qdrant — the decision (keep/delete old base_point) is made **once** in the queue processor's decision phase and applied to **both** destinations:

- **Qdrant**: Delete/create chunk points
- **Search DB**: Delete/create file_metadata + code_lines entries

Within search.db, the delete-old + insert-new is **atomic** (SQLite transaction). This ensures no window where a file version is partially present.

The per-destination state machine in the unified queue (`qdrant_status`, `search_status`) tracks completion independently. Both destinations execute in **parallel** with no ordering dependency. See [Per-destination processing flow](04-write-path.md#queue-schema) for details.

##### Memory Mirror Table

Stores a copy of memory collection entries for reverse recovery (Qdrant rebuild from state.db):

```sql
CREATE TABLE IF NOT EXISTS memory_mirror (
    memory_id TEXT PRIMARY KEY,
    rule_text TEXT NOT NULL,
    scope TEXT,
    tenant_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

Write-through on every memory upsert ensures memory rules survive total Qdrant loss and can be used to rebuild the memory collection.

### Cross-Instance Deduplication

Multiple watch folders can share the same `tenant_id` when they are clones of the same repository (same git remote URL produces the same hash — see [Tenant ID Precedence](02-collection-architecture.md#tenant-id-precedence)). The system handles this naturally through content-addressed identity.

**How it works:**

- Each watch folder has its own `tracked_files` entries (independent file inventories)
- The `base_point = hash(tenant_id, branch, relative_path, file_hash)` includes the `file_hash`, so:
  - **Identical content** across clones = same `base_point` = shared Qdrant points and search DB entries
  - **Divergent content** (different edits in different clones) = different `base_point` = separate points
- No special handling is needed — the `file_hash` in the base point naturally separates divergent content

**Reference counting for deletion:**

Before deleting old points when a file changes in one clone, check whether any OTHER watch folder instance still references that file version:

```sql
-- Given: our watch_folder_id, tenant_id, branch, relative_path, old_file_hash
SELECT COUNT(*) FROM tracked_files tf
JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
WHERE wf.tenant_id = :tenant_id
  AND tf.branch = :branch
  AND tf.relative_path = :relative_path
  AND tf.file_hash = :old_file_hash
  AND tf.watch_folder_id != :our_watch_folder_id
```

If count > 0: keep old base point (and all its chunks/lines). If 0: delete.

This query runs within the decision phase (state.db transaction) before any Qdrant or search DB operations. The decision is stored in `decision_json` and applied to both destinations.

**Example scenario:**

```
Clone A: /Users/chris/work/myproject       (tenant_id = abc123)
Clone B: /Users/chris/personal/myproject   (tenant_id = abc123)

Both clones track src/main.rs with identical content:
  base_point = hash(abc123, main, src/main.rs, sha256_of_content)
  → One set of Qdrant points, one search DB entry (shared)

Clone A edits src/main.rs:
  1. Compute new file_hash → new base_point
  2. Update Clone A's tracked_files entry
  3. Reference count: Clone B still has old file_hash → keep old points
  4. Create new points for Clone A's version
  → Now two sets of points coexist (old version for Clone B, new version for Clone A)

Clone B also edits src/main.rs:
  1. Compute new file_hash → new base_point
  2. Update Clone B's tracked_files entry
  3. Reference count: no one has old file_hash anymore → delete old points
  4. Create new points for Clone B's version
```

### Disaster Recovery

The system provides two recovery directions: Qdrant-first (primary) and state.db-first (for memory rules).

#### Primary Recovery: Rebuild state.db from Qdrant

`wqm admin recover-state` scrolls all Qdrant collections and reconstructs state.db tables from payload metadata. Qdrant is always at or ahead of any state.db snapshot, making this the most reliable recovery path.

**Recovery process:**

1. Scroll all Qdrant collections (`projects`, `libraries`, `memory`, `scratchpad`)
2. Extract payload metadata from all points
3. Reconstruct tables:
   - `watch_folders`: infer from `tenant_id` + `collection` + common ancestor of absolute file paths per tenant
   - `tracked_files`: from `file_path`, `branch`, `file_hash`, `language`, `base_point`
   - `qdrant_chunks`: from point IDs and chunk metadata (`chunk_index`, `content_hash`)
4. Regenerate derived data:
   - `sparse_vocabulary` / `corpus_statistics`: re-tokenize content from Qdrant payloads
   - `keywords` / `tags`: re-run extraction pipeline
   - LSP enrichment: daemon re-enriches automatically on startup
5. Restore `memory_mirror` from memory collection points

**Qdrant payloads contain sufficient data for recovery:**

```json
{
    "content": "chunk text",
    "base_point": "hash...",
    "tenant_id": "abc123",
    "branch": "main",
    "relative_path": "src/main.rs",
    "absolute_path": "/Users/dev/repo/src/main.rs",
    "file_hash": "sha256...",
    "language": "rust",
    "chunk_index": 0,
    ...
}
```

The `absolute_path` enables inferring `watch_folders.path` (common ancestor of all absolute paths for a tenant). The `relative_path` + `branch` + `file_hash` enable reconstructing `tracked_files`.

#### Reverse Recovery: Memory Mirror

The `memory_mirror` table in state.db stores a write-through copy of all memory collection entries. If Qdrant is lost entirely, memory rules can be restored from this mirror.

**Usage:** `wqm admin recover-qdrant --memory-only` (rebuilds memory collection from `memory_mirror` table)

#### Qdrant Snapshot Management

CLI commands for managing Qdrant snapshots:

```bash
wqm admin qdrant-snapshot create [--collection NAME]   # Create snapshot
wqm admin qdrant-snapshot list                          # List available snapshots
wqm admin qdrant-snapshot delete <ID>                   # Delete snapshot
wqm admin qdrant-snapshot download <ID> [--output PATH] # Download snapshot
wqm admin qdrant-snapshot upload <PATH>                 # Upload and restore snapshot
```

Users manage their own backup schedule (e.g., via cron). Qdrant-first recovery (`wqm admin recover-state`) is more reliable than snapshot restore because Qdrant is always at or ahead of any snapshot point in time.

---

