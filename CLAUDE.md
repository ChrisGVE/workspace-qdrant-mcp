# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

workspace-qdrant-mcp (v0.1.0-beta1) is a Model Context Protocol (MCP) server providing project-scoped Qdrant vector database operations with hybrid search capabilities.

**Core Architecture:**
- **MCP Server**: TypeScript (in development)
- **Daemon**: Rust (high-performance file watching and processing)
- **CLI**: Rust (high-performance command-line interface)

**Core Features:**
- 6 MCP tools: store, search, rules, retrieve, grep, list
- Hybrid search combining dense (semantic) and sparse (keyword) vectors
- Automatic project detection with Git integration
- Behavioral rules via persistent rules collection
- Rust daemon for high-performance file watching and processing

**v0.1.0-beta1 Features:**
- Tree-sitter semantic code chunking by function/class/method (dynamic grammar auto-download)
- LSP integration for code intelligence (per-project, active projects only)
- Branch lifecycle management (create, delete, rename, default tracking)
- Project ID disambiguation for multi-clone repositories
- Enhanced folder move detection with notify-debouncer-full
- Path validation for orphaned project cleanup
- Code relationship graph (SQLite CTEs) with algorithms: PageRank, community detection, betweenness centrality
- Graph CLI (`wqm graph`) with 7 subcommands: query, impact, stats, pagerank, communities, betweenness, migrate

**Critical Design Files:**
1. **FIRST-PRINCIPLES.md** - Core architectural principles (in project root)
2. **WORKSPACE_QDRANT_MCP.md** - Specification index (modular specs in `docs/specs/`)
3. **research/languages/** - Language research (LSP, Tree-sitter, 500+ languages)
4. **assets/default_configuration.yaml** - Default system configuration
5. **.taskmaster/docs/** - Additional PRD documents for specific features

Backward compatibility is not necessary as this project is a work in progress and has not been released.

## Refactoring Scope

**Code size breakdown and refactoring: IN SCOPE**

When modifying files that exceed the code size guidance (see global CLAUDE.md), refactor gradually by extracting the section being changed into its own module. When files are addressed, update this CLAUDE.md file by removing the files addressed.

### Rust source files exceeding 500-line limit

All Critical (>2000) and High (1000-2000) files have been split across arch-refactor rounds 1-3.
22 Moderate files were also addressed. Remaining oversized source files:

**High (>700 lines):**

| File | Lines |
|------|------:|
| `daemon/core/src/strategies/processing/file/chunk_embed.rs` | 761 |
| `daemon/core/src/embedding.rs` | 748 |
| `cli/src/commands/recover_state.rs` | 703 |
| `daemon/core/src/text_search/exact_search.rs` | 700 |

**Moderate (550–700 lines):**

| File | Lines |
|------|------:|
| `daemon/core/src/keyword_extraction/canonical_tags.rs` | 696 |
| `cli/src/commands/hooks.rs` | 695 |
| `cli/src/commands/tags.rs` | 648 |
| `daemon/core/src/lexicon.rs` | 647 |
| `daemon/core/src/parent_unit.rs` | 637 |
| `daemon/core/src/unified_queue_schema.rs` | 630 |
| `daemon/core/src/title_extraction.rs` | 629 |
| `daemon/core/src/branch_switch.rs` | 624 |
| `cli/src/commands/config_cmd.rs` | 624 |
| `daemon/core/src/processing/executor.rs` | 622 |
| `common/src/queue_types.rs` | 617 |
| `cli/src/commands/graph.rs` | 607 |
| `daemon/core/src/queue_error_handler.rs` | 604 |
| `cli/src/commands/language.rs` | 602 |
| `cli/src/commands/admin.rs` | 602 |
| `common/src/payloads.rs` | 596 |
| `cli/src/commands/update.rs` | 596 |
| `daemon/core/src/grouping/affinity.rs` | 589 |
| `daemon/core/src/cooccurrence_schema.rs` | 586 |
| `daemon/core/src/allowed_extensions.rs` | 585 |
| `daemon/core/src/service_discovery/manager.rs` | 575 |
| `daemon/core/src/processing/request_queue.rs` | 573 |
| `daemon/core/src/unified_config.rs` | 567 |
| `daemon/core/src/patterns/detection.rs` | 566 |
| `daemon/core/src/keyword_extraction/hierarchy_builder.rs` | 566 |
| `cli/src/commands/restore.rs` | 565 |
| `daemon/grpc/src/services/graph_service.rs` | 563 |
| `daemon/core/src/metadata_enrichment.rs` | 561 |
| `daemon/core/src/keyword_extraction/lsp_candidates.rs` | 560 |
| `cli/src/commands/ingest.rs` | 557 |
| `daemon/core/src/idle_history.rs` | 553 |
| `daemon/core/src/graph/ladybug_store.rs` | 547 |
| `daemon/core/src/grep_search.rs` | 546 |
| `daemon/grpc/src/services/collection_service.rs` | 539 |
| `cli/src/output.rs` | 536 |
| `daemon/core/src/strategies/processing/folder.rs` | 534 |
| `cli/src/commands/backup.rs` | 534 |
| `daemon/core/src/image_extraction.rs` | 531 |
| `daemon/core/src/text_search/regex_parser.rs` | 530 |
| `daemon/core/src/service_discovery/network.rs` | 528 |
| `daemon/core/src/file_classification.rs` | 519 |
| `daemon/core/src/storage/collections.rs` | 518 |
| `daemon/core/src/watching/move_detector.rs` | 515 |
| `common/src/project_id.rs` | 513 |
| `daemon/core/src/image_ingestion.rs` | 508 |
| `daemon/core/src/grouping/workspace.rs` | 505 |
| `cli/src/commands/collections.rs` | 504 |
| `daemon/core/src/tree_sitter/grammar_cache.rs` | 503 |

### TypeScript source files exceeding 300-line limit

All TypeScript source files are now under the 300-line limit (arch-refactor rounds 1-3 complete).

## Development Commands

### Build and Test Commands

```bash
# Rust daemon (workspace at src/rust/daemon)
cd src/rust/daemon
cargo build                     # Build daemon with all services
cargo test                      # Run Rust tests
cargo build --release           # Release build
cargo bench                     # Run benchmarks

# Binary location after build:
# src/rust/daemon/target/release/memexd

# Rust CLI (workspace at src/rust/cli)
cd src/rust/cli
cargo build --release
cargo test
# Binary at: src/rust/cli/target/release/wqm

# TypeScript MCP Server (workspace at src/typescript/mcp-server)
cd src/typescript/mcp-server
npm install
npm run build
npm test

# Rust tests for specific features
cargo test --package workspace-qdrant-core -- git_integration       # Git integration tests
cargo test --package workspace-qdrant-core -- project_disambiguation # Project ID tests
cargo test --package workspace-qdrant-core -- watching              # File watching tests
cargo test --package workspace-qdrant-core -- tree_sitter           # Semantic chunking tests
cargo test --package workspace-qdrant-core -- schema_version        # Schema version tests
cargo test --package workspace-qdrant-core -- watch_folder          # Watch folder tests
cargo test --package workspace-qdrant-core -- keyword               # Keyword extraction tests
cargo test --package workspace-qdrant-core -- tag                   # Tag extraction tests
cargo test --package workspace-qdrant-core -- graph                 # Graph subsystem tests

# Graph benchmarks (criterion)
cargo bench --package workspace-qdrant-core --bench graph_bench     # Run graph benchmarks
```

### Server Operations

```bash
# CLI operations (Rust CLI)
wqm service install             # Install daemon service
wqm service start               # Start service
wqm service status              # Check service status
wqm admin collections           # List collections
wqm admin health                # Health diagnostics
wqm queue list                  # List queue items
wqm queue stats                 # Queue statistics
wqm watch pause                 # Pause all file watchers (buffers events)
wqm watch resume                # Resume all paused watchers
wqm library ingest <file> --library <tag>  # Ingest single document
wqm stats overview              # Search instrumentation analytics
wqm stats log-search --tool=rg --query="pattern"  # Log search event
wqm tags list                   # List tags with document counts
wqm tags show <tag>             # Show tag details and keyword basket
wqm tags rebuild-hierarchy      # Trigger hierarchy rebuild
wqm tags stats                  # Tag extraction statistics
wqm graph stats --tenant <t>    # Graph node/edge counts
wqm graph query --node-id <id> --tenant <t> --hops 2  # Traverse related nodes
wqm graph impact --symbol <name> --tenant <t>          # Impact analysis
wqm graph pagerank --tenant <t> --top-k 20             # PageRank centrality
wqm graph communities --tenant <t>                     # Community detection
wqm graph betweenness --tenant <t> --top-k 20          # Betweenness centrality

# Build Rust CLI from source
cd src/rust/cli && cargo build --release
# Binary at: src/rust/cli/target/release/wqm
```

## Code Architecture

### Project Structure

```
src/
├── typescript/
│   └── mcp-server/              # TypeScript MCP server (in development)
│       ├── src/
│       │   └── index.ts         # Server entry point
│       └── package.json
└── rust/
    ├── cli/                     # Rust CLI (PRIMARY - high-performance)
    │   └── src/
    │       ├── main.rs          # Entry point
    │       ├── config.rs        # CLI configuration
    │       ├── queue.rs         # Unified queue client
    │       └── commands/        # Command implementations
    └── daemon/
        ├── core/                # Rust daemon core
        │   └── src/
        │       ├── daemon_state.rs    # SQLite state management
        │       ├── schema_version.rs  # Schema migrations
        │       ├── watch_folders_schema.rs  # Watch folders table
        │       ├── unified_queue_schema.rs  # Queue table
        │       ├── processing.rs      # Document processing
        │       ├── document_processor.rs  # Library document extraction & chunking
        │       ├── title_extraction.rs    # Document title extraction
        │       ├── watching.rs        # File watching
        │       ├── lexicon.rs           # Dynamic lexicon manager (BM25 persistence)
        │       ├── metadata_uplift.rs   # Idle-time metadata re-enrichment
        │       ├── fairness_scheduler.rs  # Anti-starvation queue scheduler
        │       ├── keywords_schema.rs     # Keyword/tag SQLite schema (v16)
        │       ├── keyword_extraction/    # Extraction pipeline
        │       │   ├── pipeline.rs        # 8-stage extraction pipeline
        │       │   └── collection_config.rs  # Per-collection tuning
        │       └── graph/               # Code relationship graph
        │           ├── mod.rs           # GraphStore trait, node/edge types
        │           ├── sqlite_store.rs  # SQLite CTE implementation
        │           ├── algorithms.rs    # PageRank, communities, betweenness
        │           ├── extractor.rs     # Tree-sitter → graph edge extraction
        │           ├── migrator.rs      # Backend migration utility
        │           ├── factory.rs       # Backend instantiation
        │           ├── schema.rs        # Graph DB schema management
        │           └── shared.rs        # Arc<dyn GraphStore> wrapper
        ├── grpc/                # gRPC service
        └── memexd/              # Daemon binary
```

### Key Design Patterns

**MCP Server (TypeScript):**
- 6 tools:
- `store` - Content storage to libraries collection
- `search` - Hybrid semantic + keyword search
- `rules` - Behavioral rules management
- `retrieve` - Direct document access
- `grep` - Exact substring/regex code search via FTS5
- `list` - Project file/folder structure listing (tree, summary, flat)

**MCP SDK Session Lifecycle (Verified Task 2):**
- Using `@modelcontextprotocol/sdk` ^1.0.0
- **Session Start**: Triggered when `server.connect(transport)` completes
  - `initializeSession()` generates session ID, detects project, registers with daemon
- **Session End**: Triggered via `server.onclose` callback
  - `cleanup()` stops heartbeat, deprioritizes project, closes connections
- **Available Hooks**:
  - `server.onerror` - fires on server errors
  - `server.onclose` - fires when connection/transport closes
  - HTTP transports additionally have `onsessioninitialized` and `onsessionclosed`
- **STDIO Transport**: No explicit session hooks; uses `server.onclose` from Server class
- **Implementation**: `src/typescript/mcp-server/src/server.ts`

**Hybrid Search:**
- Dense vectors: FastEmbed semantic search
- Sparse vectors: BM25 with IDF weighting (per-collection vocabulary, persisted in SQLite)
- Reciprocal Rank Fusion (RRF) for result combination

**Intelligence Layer:**
- **Dynamic Lexicon** (`lexicon.rs`): Per-collection BM25 vocabulary manager with SQLite persistence. Tracks document frequency for IDF computation. Auto-persists after 50 documents.
- **Keyword/Tag Extraction** (`keyword_extraction/pipeline.rs`): 8-stage pipeline — quasi-summary → lexical candidates → LSP candidates → semantic rerank → keyword selection (IDF penalty) → tag selection (MMR diversity) → basket assignment → structural tags.
- **IDF-Weighted Sparse Vectors**: File chunk sparse vectors use the LexiconManager's persisted BM25 (true IDF) instead of the EmbeddingGenerator's ephemeral BM25. Falls back to ephemeral BM25 when lexicon has no corpus stats yet.
- **Metadata Uplifting** (`metadata_uplift.rs`): Background process that runs when the queue is idle. Scans Qdrant for points with failed/partial LSP enrichment or missing concept tags. Re-enriches using the dynamic lexicon without re-embedding. Tracks `uplift_generation` per point to avoid infinite re-processing.

**Project Detection:**
- Automatic Git repo analysis with submodule support
- All project content in unified `projects` collection (multi-tenant by `tenant_id`)
- Libraries in unified `libraries` collection (multi-tenant by `library_name`)

**Library Document Ingestion:**
- Two families: page-based (PDF, DOCX, PPTX, ODT, ODS, RTF) and stream-based (EPUB, HTML, MD, TXT)
- Parent-unit architecture: parent records (no vectors) + child chunks (with vectors)
- Token-based chunking (105 target, 12 overlap) via HuggingFace tokenizer
- Provenance metadata: doc_id, doc_title, doc_fingerprint, source_format
- Search instrumentation: search_events, resolution_events, search_behavior view

**Watch Management Architecture:**
- SQLite-based configuration via `watch_folders` table
- Rust daemon polls SQLite for watch configuration changes
- Single high-performance Rust file watcher for all projects
- CLI writes watch configs directly to SQLite (no gRPC)
- Crash-resistant with WAL mode and ACID guarantees

**Daemon Project Lifecycle (Updated 2026-02-16):**
- **Enqueue-only gRPC pattern**: RegisterProject and DeleteProject do NOT perform direct SQLite mutations. They enqueue `(Tenant, Add)` or `(Tenant, Delete)` to the unified queue. The queue processor handles all database writes and Qdrant operations.
- **Daemon is SOLE owner of watch_folders**: Creates, activates, and deactivates projects via queue processing
- **MCP Server role**: Sends activation/deactivation gRPC messages with project info
- **No session counting needed**: Simple boolean `is_active` state, not integer counter
- **Registration flow (new project)**:
  1. gRPC handler enqueues `(Tenant, Add)` with project payload
  2. Queue processor creates collection, inserts watch_folder, enqueues `(Tenant, Scan)`
  3. Scan uses progressive single-level directory enumeration
- **Registration flow (existing project, high priority)**: Synchronous activation preserved for MCP server flow
- **Deletion flow**:
  1. gRPC handler enqueues `(Tenant, Delete)`
  2. Queue processor deletes Qdrant points, cascades SQLite cleanup
- **Daemon responsibilities on deactivation**:
  - Set `is_active = 0` in watch_folders
  - Keep notify registration active (file changes still tracked)
  - Continue processing any queued items
- **CLI/MCP Server can only add**: Library folders for ingestion, rules entries
- **Daemon polling**: Periodically checks watch_folders table and unified_queue
- **Startup recovery**: Progressive enqueue-first approach — enqueues `(Tenant, Scan)` items for each watch folder instead of WalkDir full-tree scan. Checks tracked_files for stale entries.

**Queue Priority (Computed at Dequeue):**
- Op-based priority: delete(10) > reset(8) > scan(5) > update(3) > add(1) — always DESC
- Collection/activity: rules=1(high), active projects=1(high), libraries=0(low), inactive=0(low) — configurable DESC/ASC
- Anti-starvation: Fairness scheduler alternates between high-priority DESC batches (10 items) and low-priority ASC batches (3 items)

**Write Path Architecture (First Principle 10):**
- **DAEMON OWNS PERSISTENT STATE**: Daemon owns both Qdrant (vectors) and SQLite (state/schema)
- **DAEMON-ONLY QDRANT WRITES**: All Qdrant write operations MUST route through the daemon
- **DAEMON OWNS SQLITE SCHEMA**: Daemon creates database and all tables (per ADR-003)
- **Canonical Collections (4 only)** (per ADR-001):
  * `projects`: Multi-tenant, all project content isolated by `tenant_id`
  * `libraries`: Multi-tenant, all libraries isolated by `library_name`
  * `rules`: Behavioral rules collection
  * `scratchpad`: Temporary working storage
- **Write Priority**: Daemon (gRPC) → Unified Queue (SQLite) → No direct Qdrant writes
- **Idempotency**: SHA256-based deduplication prevents duplicate processing

## SQLite State Management

### Database Ownership (ADR-003)
- **Daemon owns the database**: The Rust daemon (memexd) is the sole owner of SQLite
- **Database path**: `~/.workspace-qdrant/state.db`
- **Schema creation**: Daemon creates all tables on startup
- **Schema migrations**: Daemon handles all version upgrades
- **MCP Server/CLI**: May read/write to tables, but must NOT create tables or run migrations
- **Graceful degradation**: If daemon hasn't run, other components handle missing tables gracefully

### Database Transaction Principle
**All database operations MUST be conducted within a transaction** to preserve database integrity.

### Core Tables (14)
1. **schema_version** - Migration tracking
2. **unified_queue** - Write queue with lease-based processing
3. **watch_folders** - Project and library watch configurations
4. **tracked_files** - Authoritative file inventory with metadata (daemon writes, CLI reads)
5. **qdrant_chunks** - Qdrant point tracking per file chunk (child of tracked_files, daemon only)
6. **search_events** - Search instrumentation (tool, query, latency, session)
7. **resolution_events** - Document open/expand events after search
8. **sparse_vocabulary** - BM25 IDF term-to-document-count mapping (schema v15)
9. **corpus_statistics** - BM25 IDF total document counts per collection (schema v15)
10. **keywords** - Per-document keyword records with scores (schema v16)
11. **tags** - Per-document concept/structural tag records (schema v16)
12. **keyword_baskets** - Tag-to-keywords mapping for query expansion (schema v16)
13. **canonical_tags** - Deduplicated cross-document tag hierarchy nodes (schema v16)
14. **tag_hierarchy_edges** - Parent-child tag relationships (schema v16)

### Unified Queue Schema

```sql
CREATE TABLE unified_queue (
    queue_id TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE NOT NULL,  -- SHA256 hash for deduplication
    item_type TEXT NOT NULL,               -- text, file, url, website, doc, folder, tenant, collection
    op TEXT NOT NULL,                      -- add, update, delete, scan, rename, uplift, reset
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',         -- pending, in_progress, done, failed
    branch TEXT,
    payload_json TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_error TEXT,
    leased_by TEXT,
    lease_expires_at TEXT
);
```

### Idempotency Key Generation

All implementations (TypeScript, Rust daemon, Rust CLI) use the same algorithm:

```
idempotency_key = SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
```

### Queue Monitoring

```bash
wqm queue list                # List queue items
wqm queue stats               # Queue statistics
wqm queue show <id>           # Show item details
wqm queue clean --days 7      # Clean old items
```

## Rust Daemon Development

1. **Unified workspace**: `src/rust/daemon/` (contains core, grpc, memexd)
2. **Build from workspace root**: `cd src/rust/daemon && cargo build --release`
3. **Binary location**: `src/rust/daemon/target/release/memexd`
4. **Run tests**: `cd src/rust/daemon && cargo test`
5. **gRPC services**: SystemService, CollectionService, DocumentService
6. **Protocol definition**: `src/rust/daemon/proto/workspace_daemon.proto`

### Intel Mac (x86_64-apple-darwin) Build Requirements

**Option 1: Pre-built static library (Recommended)**

Download pre-built static ONNX Runtime from supertone-inc (same as release CI):
```bash
mkdir -p ~/.onnxruntime-static
curl -L "https://github.com/supertone-inc/onnxruntime-build/releases/download/v1.23.2/onnxruntime-osx-universal2-static_lib-1.23.2.tgz" \
  -o ~/.onnxruntime-static/ort.tgz
tar xzf ~/.onnxruntime-static/ort.tgz -C ~/.onnxruntime-static
rm ~/.onnxruntime-static/ort.tgz
ORT_LIB_LOCATION=~/.onnxruntime-static/lib cargo build --release
```

**Option 2: Homebrew ONNX Runtime (dynamic linking)**

For quick development builds (binary will depend on homebrew's dylib):
```bash
brew install onnxruntime
ORT_LIB_LOCATION=/usr/local/Cellar/onnxruntime/1.23.2_2 ORT_PREFER_DYNAMIC_LINK=1 cargo build --release
```

**Option 3: GitHub Actions release workflow**

Trigger the release workflow for fully tested cross-platform builds:
```bash
gh workflow run release.yml
```

See https://ort.pyke.io/setup/linking for detailed instructions.

## gRPC Services

**Services** (defined in `src/rust/daemon/proto/workspace_daemon.proto`):
1. **SystemService** - Health checks, metrics, lifecycle management
2. **CollectionService** - Collection CRUD, alias management
3. **DocumentService** - Text ingestion, updates, deletion
4. **EmbeddingService** - Dense/sparse vector generation for MCP server
5. **ProjectService** - Project lifecycle, session management
6. **GraphService** - Code relationship graph queries and algorithms (7 RPCs)

### Testing gRPC Connectivity

```bash
# Verify daemon is listening on gRPC port
lsof -i :50051

# Test with grpcurl (if installed)
grpcurl -plaintext localhost:50051 list
```

## Environment Configuration

**Required for Qdrant:**
- `QDRANT_URL` - Server URL (default: http://localhost:6333)
- `QDRANT_API_KEY` - API key (required for Qdrant Cloud)

**Optional:**
- `FASTEMBED_MODEL` - Embedding model (default: all-MiniLM-L6-v2)
- `WQM_DATABASE_PATH` - Override database path
- `WQM_LOG_LEVEL` - Log level (DEBUG, INFO, WARN, ERROR)

## Task Master AI Instructions

**Import Task Master's development workflow commands and guidelines.**
@./.taskmaster/CLAUDE.md

## Task Master Workflow Preferences

- **Task Expansion**: Always expand tasks with `research=false` to avoid unnecessary API calls and delays
- **Task Execution**: Execute plans continuously, only stopping for disambiguation or non-obvious design decisions
- **Design Decisions**: Make obvious design decisions that align with project plan and requirements autonomously
- **Agent Execution Mode**: Execute agents SEQUENTIALLY (one at a time), not in parallel, to conserve API usage

## Code Intelligence: Serena MCP

Use Serena's symbolic tools proactively for code navigation and refactoring:
- **`get_symbols_overview`**: Before modifying a file, get the symbol map instead of reading the whole file
- **`find_symbol`**: Locate specific structs, methods, functions by name path (e.g., `FairnessSchedulerConfig`, `AllowedExtensions/is_allowed`)
- **`find_referencing_symbols`**: Find all callers/references before changing a function signature or struct field
- **`replace_symbol_body`**: For precise method-level replacements (safer than regex for Rust code)
- **`insert_after_symbol`** / **`insert_before_symbol`**: For adding new methods or structs near related code
- Prefer Serena's symbolic tools over Grep when the query is about code structure (who calls X, what implements Y) rather than text content

At end of session: leave a summary of Serena usage findings (what worked well, what didn't, comparison with Grep/Read).

## Critical Development Rules

**⚠️ NO MIGRATION EFFORT**: This project requires NO migration effort of any kind. The project is a work in progress and has not been released - there are no users to migrate.

- **DO NOT CHANGE THE MCP CONFIGURATION**: Never modify MCP server configuration without explicit permission
- **NO .mcp.json FILE**: The MCP server is already installed system-wide. Do not create `.mcp.json` in this project. If the server needs reconnecting, the user will handle it manually.
- **Server Stability**: Always verify server starts without crashes after changes
- **Git Discipline**: Follow strict temporary file naming conventions (YYYYMMDD-HHMM_name format)
- **Atomic Commits**: Make focused, single-purpose commits with clear messages
- When even a small error is detected, or a compilation warning is showing up, immediately address their root cause and do not silence them
- **Fix all failing tests**: When you discover a test that fails, fix it immediately - even if the failure is not caused by your current work. Do not skip or defer pre-existing test failures.
- Do not create backup folders to maintain "old" code. Make modifications directly.

## Future Enhancements / Parking Lot

### Memory Rule Duplication Detection
Use embeddings to detect similar rules on insert. When a new rule has cosine similarity ≥ 0.7 with existing rules, surface candidates to user for review.

### Claude Code Integration Hook
HTTP POST endpoint on MCP server for file change notifications, enabling real-time ingestion during active coding sessions.

### Image Handling in Documents (Tier 2 — OCR)
Integrate OCR (e.g., tesseract via `leptess` or `tesseract-rs`) to extract text from images embedded in documents. Target: equations, labeled diagrams, screenshots of code, tables rendered as images. Extracted text is concatenated into the document's text stream at the image's position. High-value for technical/academic content where equations are frequently image-rendered.

### Image Embedding (Tier 3 — Multimodal)
CLIP-style model (e.g., `clip-ViT-B-32` via fastembed or separate ONNX model) to embed images into their own vector space. Requires: new `images` collection in Qdrant with different dimensionality, cross-modal search (text query → CLIP text encoder → image vector search), storage of image thumbnails or references to source documents. Architecturally significant — different embedding model, collection, and search path.
