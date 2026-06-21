# Changelog

All notable changes to the workspace-qdrant-mcp project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **LadybugDB default code-graph backend (A0.7)** — `graph.db` is now backed by LadybugDB (a maintained Kuzu fork with embedded Cypher) by default, replacing the SQLite recursive-CTE adjacency store as the shipped default. The SQLite backend remains available as a fallback via `graph.backend: sqlite`.
- **Backend-agnostic `GraphStore` trait + conformance suite** — graph operations are routed through a single `GraphStore` trait implemented by both the SQLite and LadybugDB backends, with a shared conformance test suite (`core/tests/graph_backend_conformance.rs`) that runs the identical assertions against each backend.
- **New gRPC GraphService RPCs** — `NarrativeQuery` (narrative-layer graph query; SQLite-only, returns UNIMPLEMENTED on the LadybugDB backend) and `QueryCrossBoundary` (cross-boundary graph traversal across layer/module boundaries).
- **`qdrant_point_id` graph link columns (graph schema v5)** — graph nodes carry a `qdrant_point_id` linking each node to its corresponding Qdrant vector point, enabling graph↔vector cross-references.
- **CI graph backend lanes** — a SQLite-only conformance lane runs on every PR, and a scheduled lane exercises the full LadybugDB conformance suite.
- **Path abstraction system (spec 16)** — type-system-enforced root/relative path discipline for host/Docker deployment parity. `CanonicalPath` for root paths (watch folders, project roots), `RelativePath` for content paths (files under a root). All paths stored relative to their watch folder root; absolute paths reconstructed via FK join at query time.
- **RelativePath newtype** (`wqm-common::paths`) — validates relative paths (rejects absolute, `..` segments, embedded NUL), serde-transparent for wire compatibility.
- **Schema v37 migration** — drops denormalized `file_path` columns from `tracked_files`, truncates ingest tables, implements crash-safe 4-phase migration protocol with `relative_path_migration_in_progress` marker table.
- **Post-startup migration hook** — after file watchers start, detects in-progress migration, truncates Qdrant ingest collections (with retry), waits for queue drain, and finalizes migration.
- **gRPC path validation macros** — `extract_canonical_path!`, `extract_relative_path!`, `extract_relative_paths!` for handler-entry validation. All path-accepting handlers now validate at entry with `InvalidArgument` on bad input.
- **CLI migration banner** — `wqm status health` shows progress banner during relative-path migration with percentage complete.
- **Zero-byte file handling** — empty files are recorded in tracked_files for inventory (chunk_count=0, Skipped status) without hitting the embedding pipeline.
- **Migration crash-recovery tests** — 9 integration tests simulating crashes at each phase boundary of the v37 migration.
- **Unified `DaemonConfig::validate()`** — chains every subconfig validator (queue_processor, monitoring, git, observability, embedding, lsp, grammars, updates, resource_limits, startup, daemon_endpoint, ingestion_limits, auto_ingestion). Errors are prefixed with the subsystem name for quick diagnosis. Called in `main.rs` immediately after `load_config`; invalid config causes non-zero exit before the daemon starts. (F-047)
- **`IngestionLimits` and `AutoIngestion` validators** — bounds checks for per-extension size limits and auto-ingestion settings, now wired into the unified validation chain. (F-047)
- **Cross-process single-instance lock for memexd (spec 16 §10.1, T9)** — memexd now binds a TCP listener to `127.0.0.1:7799` at startup, before opening SQLite. A second memexd (host or docker) attempting to start refuses with an actionable error mentioning the port and the `--control-port` override. Process death releases the bind immediately; no stale-lock cleanup needed. A diagnostic JSON identity stamp (`mode`, `pid`, `started_at`, `port`) is written to `~/.local/share/workspace-qdrant/memexd.lock` (host) or `/var/lib/wqm/memexd.lock` (docker) — best-effort, the socket is authoritative. Port resolution precedence: `--control-port` CLI flag > `WQM_CONTROL_PORT` env > `DaemonConfig.control_port` > built-in 7799. New `control_port: Option<u16>` field on `DaemonConfig` (defaults to `None` ⇒ 7799).
- **Schema migration v35 atomicity** — v35 table rebuild now runs inside `BEGIN IMMEDIATE … COMMIT / ROLLBACK`, with a `ForeignKeysGuard` RAII wrapper that restores `PRAGMA foreign_keys` on error. Prevents partial DDL state on crash mid-migration. (F-046)

### Changed

- **Default graph backend is now `ladybug`** — the `graph.backend` default in `assets/default_configuration.yaml` is now `ladybug` (was `sqlite`). Existing deployments keep their configured backend; set `graph.backend: sqlite` to retain the recursive-CTE store.
- **BREAKING: Schema v37** — first startup after upgrade triggers a wipe-and-rebuild of all ingested data (tracked_files, qdrant_chunks, unified_queue). Re-ingestion runs automatically; large repos may take 30+ minutes. Monitor via `wqm status health` banner.
- **BREAKING: File paths stored as relative** — `tracked_files` no longer has a `file_path` column; paths are stored as `relative_path` anchored to the watch folder root. Queries that previously used absolute file paths must join through `watch_folders.path`.
- **Proto field annotations** — all path fields in `workspace_daemon.proto` annotated with `[canonical]`, `[relative]`, or `[non-path]` comments.
- **Search cache-miss result cap** — exact-search cache misses previously ran with `max_results: usize::MAX`, streaming every matching FTS row into memory. Cache misses are now capped at a bounded limit and context lines are attached without re-executing the full search. (F-028, F-029)
- **TypeScript CI** — `better-sqlite3` is rebuilt against the active Node.js ABI, and unit and integration tests run as separate steps to surface failures more precisely.
- **Dependabot coverage expanded** — now covers the full Cargo workspace (daemon, CLI, common, grpc, memexd) in addition to the existing npm scope.
- **Docker base images pinned by SHA256** — `node:20-slim` base images in all Dockerfiles are now pinned to a specific digest rather than a floating tag. (supply chain hardening)
- **fastembed patch annotated; checksum enforced** — the fastembed Cargo patch entry carries a comment explaining the override, and the download-install scripts now fail-closed when a checksum is absent or mismatched.

### Fixed

- **Search filter dropped on >500 base-points** — the fallback code path that disabled base-point filtering and broadened to all-tenant results when a project had more than 500 active base-points now falls back to the primary base-point instead of dropping the filter entirely. (F-012)
- **FTS position-aware sequence assignment** — `diff_apply` previously appended all middle-of-file insertions after `MAX(seq)`, causing incorrect FTS line ordering. Sequences are now assigned at the correct target index. (F-018)
- **Project path canonicalization on registration** — project roots are now canonicalized and validated to exist before enqueueing; relative paths, symlinks, and nonexistent paths are rejected at registration time rather than producing silent failures. (F-019)
- **Queue re-lease on memory pressure** — under memory pressure, the queue processor previously re-leased only the current item and left the rest of the batch stuck `in_progress` until lease expiry. All remaining batch items are now re-leased together. (F-044)
- **FastEmbed init panic replaced with gRPC error** — the `.expect()` call in the gRPC embedding service is replaced with a controlled error response; a missing or undownloadable model now returns a gRPC status rather than crashing the service. (F-048)
- **`cleanupSession` double-invocation** — `cleanupSession` now carries an idempotence flag so that the `onclose` and `stop()` paths cannot both fire `recordSessionEnd()`, preventing metrics from being decremented twice. (F-049)
- **`max_retries` column removed from queue schema** — stale `max_retries` column removed from `CREATE_UNIFIED_QUEUE_SQL` and all migration copies, eliminating schema/code drift.
- **`--allow-default` env-var override gap** — when `--allow-default` triggered a fallback to built-in defaults, `apply_env_overrides` was not called on the fallback config. `OTEL_*` and `WQM_PROMETHEUS_*` env vars are now applied on both the normal and fallback paths.

### Security

- **`download-install.ps1` fails closed on missing checksum** — the PowerShell installer now aborts instead of continuing when a checksum file is absent or does not match. (supply chain hardening, T20-closeout)
- **ONNX Runtime version-bump CI gate** — the scheduled ONNX Runtime version-bump workflow now runs the full build-and-test matrix before committing the version change, preventing a broken version from landing. (T20-closeout)
- **Security advisory exemption table** — `security.yml` now contains a documented exemption table for known low-risk advisories, making audit decisions explicit and reviewable. (T20-closeout)
- **Provenance / SBOM preservation markers** — `docker-publish.yml` attestation steps carry `F-P2` markers to preserve SLSA provenance and SBOM generation through future edits.

## [0.1.3] - 2026-04-18

### Added
- **Docker images** — published to Docker Hub and GHCR for `memexd` (Rust daemon) and `workspace-qdrant-mcp` (TypeScript MCP server), multi-arch `linux/amd64` + `linux/arm64`.
- **Three deployment bundles** under `docker/compose/`:
  - `minimal.yml` — memexd + MCP against an external Qdrant
  - `full-stack.yml` — attaches to the user's existing main-docker stack
  - `standalone-memexd.yml` / `standalone-mcp.yml` — single-service deployments
  - `observability.yml` — self-contained Prometheus + Grafana + OpenTelemetry collector overlay
- **Telemetry pipeline**:
  - MCP server instrumented with `prom-client` — exports `wqm_mcp_tool_invocations_total`, `wqm_mcp_tool_duration_seconds`, `wqm_mcp_session_count`, `wqm_mcp_daemon_fallback_total`, `wqm_mcp_cache_{hits,misses}_total`.
  - HTTP `/metrics` endpoint on port 9092 when `MCP_SERVER_MODE=http`.
  - OTLP push export on session end in stdio mode (1s fire-and-forget).
  - New daemon gauge `wqm_queue_oldest_pending_age_seconds` driving the QueueStuck alert.
- **Prometheus alerting rules** (`docker/prometheus/alerts.yml`) — 6 rules covering queue health (QueueStuck, QueueFailedWarning, QueueFailedCritical), daemon/qdrant availability (DaemonDown, QdrantUnreachable), and MCP session idleness (MCPNoInvocations).
- **Four Grafana dashboards** auto-provisioned under `Workspace`: claude-mcp, qdrant, memexd, system-overview.
- **Deployment documentation** under `docker/docs/` — README + minimal / full-stack / standalone / telemetry / dashboards guides.
- **`wqm service` DaemonSource awareness** — `start`, `stop`, `restart`, `status` now detect whether the daemon runs locally, in Docker, both, remotely, or not at all, and branch behaviour accordingly.
- **CI workflows** — `.github/workflows/docker-publish.yml` (tag-triggered multi-arch publish to Docker Hub + GHCR), `docker-test.yml` (pre-publish smoke test), `docker-integration.yml` (full-stack e2e).

### Changed
- Dockerfiles use Rust 1.90 base.

### Deprecated
- **Legacy Queue Tables** - `ingestion_queue` and `content_ingestion_queue` tables are deprecated
  - Use `unified_queue` table instead
  - Will be removed in v0.5.0
  - See [MIGRATION.md](docs/MIGRATION.md#queue-migration-guide-legacy--unified-queue-v040)
- **SQLiteQueueClient** - This class is deprecated
  - Use `SQLiteStateManager.enqueue_unified()` instead
  - Runtime `DeprecationWarning` emitted on instantiation
  - Will be removed in v0.5.0
- **Dual-write mode** - `queue_processor.enable_dual_write` defaults to `false`
  - Only enable for migration compatibility
  - Will be removed in v0.5.0

## [0.4.1] - 2026-03-25

### Fixed

#### Critical: Processing Pipeline Memory Leak
- **Dynamic grammar memory leak** — Using cached `.dylib` grammars for languages with
  static support (Python, Rust, TypeScript, etc.) caused ~270MB/s memory growth in the
  tree-sitter parser due to ABI mismatches. The daemon now prefers statically compiled
  grammars and only falls back to dynamic grammars for languages without built-in support.
  Before: RSS 85MB to 2.4GB in 10 seconds, 0 items completed.
  After: RSS stable at ~620MB, 100+ items processed per minute.
- **macOS memory pressure detection** — Replaced `sysinfo` `used_memory()` (which
  reports compressor pages as used on macOS, showing ~84% usage on idle 64GB systems)
  with `kern.memorystatus_level` sysctl for accurate available-memory reporting.
- **Process RSS safety valve** — Added per-process RSS check (2GB default) using
  `mach task_info` on macOS / `/proc/self/statm` on Linux. Fires at loop top and
  between batch items to prevent runaway memory growth.
- **Qdrant readiness wait** — Added exponential-backoff gRPC readiness wait at startup
  (up to ~90s) to prevent circuit breaker from tripping when memexd starts before
  Qdrant on system boot.
- **sysinfo allocation leak** — Replaced per-poll `System::new()` allocation with
  `thread_local!` reuse in the sysinfo fallback path.

### Added

#### CLI UX Overhaul
- **Interactive TUI dashboard** — New `wqm tui` command with a 2x4 interactive grid
  showing service status, queue stats, projects, libraries, graph, config, languages,
  and recent activity. Each cell supports popup detail views.
- **TUI browsers** — Project browser, library browser, and queue browser with
  filtering, scrolling, and detail popups.
- **Output design system** — Standardized CLI output with borderless tables, consistent
  key-value formatting, path shortening (`~/` for home), 8-char hash/UUID truncation,
  section headers, and summary count footers.
- **Contextual empty states** — Commands that return no results now display helpful
  messages with suggested next actions.
- **Queue list improvements** — Added project name, subject, and error columns; `--all`
  flag to override default pagination; rich detail view for `queue show`.

### Changed
- **CLI renamed to Companion** — `wqm-cli` description updated, help styling improved,
  `init completions` and `init man` commands flattened under `init` namespace.
- **Command hierarchy reorganized** — `watch` moved under `project`, `collections` under
  `admin`, `backup`/`restore` under `admin`, `man`/`hooks` under `init`.
- **Consistent output formatting** — All commands converted from bordered tables to the
  new clean borderless style with dots for path indicators and dynamic separator widths.

## [0.4.0] - 2025-01-19

### Added

#### Unified Multi-Tenant Collection Architecture
- **Single `_projects` Collection** - ALL project content now stored in one unified collection with tenant isolation via `tenant_id` payload field (12-char hex derived from normalized git remote URL or path hash). See [ARCHITECTURE.md](docs/ARCHITECTURE.md#collection-structure) for details.
- **Single `_libraries` Collection** - ALL library documentation now stored in one unified collection with tenant isolation via `library_name` payload field
- **Payload Indexing** - O(1) filtering performance via indexed payload fields on `tenant_id` and `library_name`
- **Hard Tenant Isolation** - Queries automatically filter by tenant_id, preventing cross-tenant data leakage
- **Cross-Project Search** - Search across all projects with `scope="all"` parameter
- **Library Inclusion** - Include library documentation in searches with `include_libraries=True` parameter

#### Session Lifecycle Management (ProjectService gRPC)
- **RegisterProject RPC** - Register project for high-priority processing when MCP server starts. See [GRPC_API.md](docs/GRPC_API.md#registerproject) for usage.
- **DeprioritizeProject RPC** - Decrement session count when MCP server stops gracefully
- **Heartbeat RPC** - Keep session alive with 30-second intervals (60-second timeout)
- **GetProjectStatus RPC** - Query current project status, priority, and session count
- **ListProjects RPC** - List all registered projects with priority and active-only filtering
- **Priority-Based Processing** - Active sessions get HIGH priority (queue position 1), inactive get NORMAL (position 5)
- **Automatic Project Registration** - MCP server automatically registers project on startup, daemon begins watching immediately
- **Orphaned Session Detection** - Sessions missing heartbeat for 60 seconds marked orphaned, cleaned up periodically
- **Session Revival** - New MCP connections can revive orphaned projects without re-registration. See [ARCHITECTURE.md](docs/ARCHITECTURE.md#session-lifecycle-management) for state machine diagram.

#### Search Enhancements
- **`scope` Parameter** - Control search scope: `"project"` (current project only, default), `"global"` (global collections), `"all"` (all projects). See [API.md](API.md#search) for examples.
- **`include_libraries` Parameter** - Include `_libraries` collection in search results
- **`branch` Parameter** - Filter results by git branch (supports `"*"` for all branches)
- **`collections_searched` Response Field** - Lists collections that were searched
- **Metadata-Based Multi-Project Filtering** - Use `tenant_id` in filter conditions for advanced queries

#### Documentation
- **[docs/GRPC_API.md](docs/GRPC_API.md)** - Comprehensive gRPC API reference documenting all 4 services (20 RPCs): SystemService, CollectionService, DocumentService, ProjectService
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Updated with session lifecycle management section including state machine and sequence diagrams
- **[API.md](API.md)** - Updated MCP tools reference with multi-tenant search examples
- **[CLI.md](CLI.md)** - Updated CLI reference with unified collection commands
- **[MIGRATION.md](MIGRATION.md)** - Migration guide from v0.3.x to v0.4.0 unified model

### Changed

#### Breaking Changes

**Collection Architecture (MAJOR)**
- **Unified `_projects` Collection** replaces per-project `_{project_id}` collections
  - All project documents now stored in single collection with `tenant_id` payload field
  - Previous: One collection per project (e.g., `_abc123def456`, `_789xyz012abc`)
  - New: Single `_projects` collection with tenant isolation via payload filtering
- **Unified `_libraries` Collection** replaces per-library `_{library_name}` collections
  - All library documents now stored in single collection with `library_name` payload field
  - Previous: One collection per library (e.g., `_fastapi`, `_react`)
  - New: Single `_libraries` collection with library isolation via payload filtering
- **Only 4 Collection Types** - System now uses exactly 4 collection types:
  1. `_projects` - Unified collection for ALL project content
  2. `_libraries` - Unified collection for ALL library documentation
  3. `{user}-{type}` - User collections (e.g., `work-notes`, `myapp-scratchbook`)
  4. `_memory`, `_agent_memory` - System and agent memory rules

**Search API Changes**
- **Default scope is `"project"`** - Searches current project only by default (previously searched all content)
- **`include_libraries` required** - Libraries not included unless explicitly requested
- **Response format updated** - Now includes `tenant_id` and `collections_searched` fields

**Configuration**
- Collection naming constants updated:
  - `UNIFIED_PROJECTS_COLLECTION = "_projects"`
  - `UNIFIED_LIBRARIES_COLLECTION = "_libraries"`

### Migration

**Automatic Migration Available**
```bash
# Preview migration (dry run)
wqm admin migrate-to-unified --dry-run

# Execute migration
wqm admin migrate-to-unified

# Verify migration
wqm admin collections --verbose
```

**Migration Process:**
1. Creates `_projects` and `_libraries` collections if not exist
2. Copies documents from old `_{project_id}` collections to `_projects`
3. Adds `tenant_id` metadata to each document
4. Copies documents from old `_{library_name}` collections to `_libraries`
5. Adds `library_name` metadata to each document
6. Optionally deletes old collections after verification

**Search Migration:**
```python
# Old behavior (v0.3.x)
search(query="authentication")  # Searched all content

# New behavior (v0.4.0)
search(query="authentication")                              # Current project only
search(query="authentication", scope="all")                 # All projects
search(query="authentication", include_libraries=True)      # Current project + libraries
search(query="authentication", scope="all", include_libraries=True)  # Everything
```

### Benefits of Unified Model

- **Scalability** - Supports thousands of projects without collection sprawl
- **Efficiency** - Single HNSW index per collection type (better memory utilization)
- **Cross-Project Discovery** - Semantic search across all projects with one query
- **Simpler Operations** - Fewer collections to manage and monitor
- **Better Isolation** - Hard tenant filtering prevents data leakage

---

## [0.3.0] - 2025-10-21

### Added

#### Documentation
- **[API.md](API.md)** - Comprehensive MCP tools API reference with detailed parameter descriptions, usage examples, and Claude Desktop integration instructions
- **[CLI.md](CLI.md)** - Complete CLI reference for all `wqm` commands including service management, memory operations, admin tools, configuration, watch folders, document ingestion, search, library management, LSP integration, and observability
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive troubleshooting guide covering installation issues, Qdrant connection problems, MCP server debugging, daemon operations, performance tuning, configuration, debugging commands, log locations, and common error messages
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Visual architecture documentation with mermaid diagrams showing system overview, component architecture, hybrid search flow, collection structure, SQLite state management, write path architecture, and data flow patterns

#### Core Features
- **Rust Daemon (memexd)** - High-performance processing engine for document ingestion and file watching
- **Hybrid Search** - Reciprocal Rank Fusion (RRF) combining semantic vector search with BM25-style keyword matching
- **Multi-tenant Collections** - Project-scoped collection architecture with metadata-based filtering
  - PROJECT collections: `_{project_id}` (auto-created by daemon)
  - LIBRARY collections: `_{library_name}` (user-managed)
  - USER collections: `{basename}-{type}` (optional custom organization)
- **SQLite State Management** - Unified state database for watch folder configuration, daemon coordination, and project metadata
- **Four-tier Context Hierarchy** - Global, project, file_type, and branch-level filtering

#### LLM Context Injection System
- **Trigger Framework** - Comprehensive hook/trigger system for Claude Code integration (Task 301)
  - `OnDemandRefreshTrigger` - Manual context refresh with duplicate prevention (Task 301.3)
  - `PostUpdateTrigger` - Automatic refresh after rule/config updates with debouncing (Task 301.4)
  - `ToolAwareTrigger` - LLM tool detection with automatic formatter selection (Task 301.5)
- **Real-time Token Tracking** - Token usage monitoring for context budget management (Task 302.2)
- **Session Detection** - Copilot and other LLM tool session detection (Task 298.1)

#### Testing Framework
- **LLM Injection Integration Tests** (Task 337)
  - Behavioral harness for mocking LLM interactions
  - Multi-tool integration testing (MCP server, CLI, direct API)
  - Project-specific rule activation tests
  - Cross-session persistence testing
  - Performance and stress testing (16 comprehensive tests)
- **Property-based Testing** - Proptest integration for filesystem events, file processing, error handling, and data serialization
- **End-to-End Testing** - Complete workflow testing including file ingestion, search, administration, and project switching
- **Performance Monitoring** - PerformanceMonitor and MemoryProfiler utilities with statistical analysis

#### LSP Integration
- **Ecosystem-aware LSP Detection** - Automatic language server discovery with 500+ language support
- **Symbol Extraction** - O(1) symbol lookup system for code intelligence
- **LSP Configuration Management** - Comprehensive configuration system with health monitoring
- **Integration Testing Framework** - LSP initialization handshake and capability negotiation tests (Task 278.1)

#### Advanced Features
- **Web Crawling System** - Integrated web crawler with rate limiting, robots.txt compliance, link discovery, and recursive crawling
- **Intelligent Auto-ingestion** - File watcher with debouncing, filtering, and batch processing
- **Performance Monitoring** - Self-contained monitoring system with metrics collection, alerting, statistical analysis, and predictive models
- **Security Monitoring** - Comprehensive security system with alerting, compliance validation, and audit logging
- **Service Discovery** - Multi-instance daemon coordination with automatic service discovery
- **Circuit Breaker Pattern** - Error recovery with exponential backoff and graceful degradation
- **Queue Management** - Python SQLite queue client for MCP server with priority ordering and retry logic

#### MCP Server Enhancements
- **Four Streamlined Tools** - Simplified from complex tool matrix to four comprehensive tools:
  - `store` - Content storage with automatic embedding and metadata
  - `search` - Hybrid semantic + keyword search with branch/file type filtering
  - `manage` - Collection and system management
  - `retrieve` - Direct document access with metadata filtering
- **Stdio Mode Compliance** - Complete console output suppression for MCP protocol compliance
- **Tool Availability Monitoring** - Real-time monitoring of tool availability and health

#### CLI Tools
- **Memory Collection Management** - System for managing `_memory` and `_agent_memory` collections
- **Watch Folder Commands** - Add, list, remove, status, pause, resume, configure, and sync watch folders
- **Health Diagnostics** - `workspace-qdrant-health` command for system diagnostics
- **Service Management** - Install, uninstall, start, stop, restart, status, and logs for daemon service
- **Observability Commands** - Health, metrics, diagnostics, and monitoring tools

### Changed

#### Breaking Changes

**Configuration**
- Configuration format updated to match PRDv3 specification
  - All timeout and size values now require explicit units (e.g., `"100MB"`, `"30s"`, `"100ms"`)
  - Removed hardcoded file paths in favor of XDG standard locations
  - `auto_ingestion.project_collection` → `auto_ingestion.auto_create_project_collections` (boolean)
  - Collection naming standardized to `_{project_id}` pattern for project collections

**Storage**
- Watch configuration storage migrated from JSON files to SQLite database
  - Location: `~/.local/share/workspace-qdrant/daemon_state.db`
  - Maintains backward compatibility through automatic migration
  - Unified data storage with Rust daemon
  - Proper database indexes for performance

**Architecture**
- Collection architecture redesigned for multi-tenant support
  - Single collection per project with metadata filtering
  - Branch-aware querying with Git integration
  - File type differentiation (code, test, docs, config)
- Daemon instance architecture redesigned for project isolation
  - Per-project daemon support with service discovery
  - Dynamic port management and assignment
  - Resource coordination for multi-instance daemons

**Dependencies**
- Migrated from `structlog` to `loguru` for logging
- Migrated from manual configuration to unified config system
- Added Rust components (Cargo workspace with daemon, grpc, and python-bindings)

### Deprecated

- `auto_ingestion.project_collection` configuration setting (use `auto_create_project_collections` instead)
- Legacy backward compatibility methods removed in favor of new unified APIs
- Collection defaults system eliminated system-wide

### Removed

- **Backward Compatibility Layer** - Removed legacy compatibility methods
- **Structlog Dependency** - Replaced with loguru for unified logging
- **Hardcoded Patterns** - Replaced with PatternManager integration
- **Collection Defaults** - Eliminated to prevent collection sprawl
- **Manual JSON Configuration** - Replaced with SQLite state management
- **Duplicate Client Patterns** - Consolidated with shared utilities

### Fixed

#### Critical Fixes
- **MCP Protocol Compliance** - Redirected all logs to stderr in stdio mode to prevent protocol contamination
- **Service Management** - Modernized macOS service management with proper user domain and shutdown handling
- **Configuration Loading** - Resolved daemon configuration loading with correct database paths
- **Import Path Standardization** - Migrated all imports from relative to absolute paths
- **SSL Warnings** - Comprehensive SSL warning suppression to clean up output

#### Performance Improvements
- **High-throughput Document Processing** - Optimized ingestion pipeline with batch processing
- **Intelligent Batching** - Smart batch processing manager for bulk file changes
- **Connection Pooling** - Qdrant client with connection pooling and circuit breaker
- **Graceful Degradation** - Comprehensive degradation strategies for resilience

#### Testing Improvements
- **Test Isolation** - Implemented isolated Qdrant testing infrastructure with testcontainers
- **Continuous Testing** - Python testing loop with coverage measurement
- **Property-based Testing** - Comprehensive proptest integration for edge case validation
- **Mock Infrastructure** - Enhanced mocks and error injection framework

#### Bug Fixes (254 total)
- Resolved circular import issues across codebase
- Fixed configuration field mappings for auto-ingestion and file-watcher
- Corrected import paths throughout CLI modules and common package
- Fixed daemon processor access and queue item status updates
- Resolved compilation errors in Rust modules
- Fixed test timeout issues and achieved working coverage measurement
- Corrected memory collection patterns and access control
- Fixed file watcher path bugs in config transition
- Resolved service status detection for launchd
- Fixed indentation errors and syntax issues across codebase

### Security

- **Input Validation** - Enhanced validation for all user-provided data
- **Security Monitoring** - Comprehensive monitoring and alerting system
- **Audit Logging** - Security event tracking and compliance validation
- **Privacy Controls** - Analytics system with configurable privacy controls

---

## [0.1.0] - 2024-08-28

### Features

#### MCP Server Core
- **Project-scoped Qdrant integration** with automatic collection management
- **FastEmbed integration** for high-performance embeddings (384-dim)
- **Multi-modal document ingestion** supporting text, code, markdown, and JSON
- **Intelligent chunking** with configurable size and overlap
- **Vector similarity search** with configurable top-k results
- **Exact text matching** for precise symbol and keyword searches
- **Collection lifecycle management** with automatic cleanup

#### Search Capabilities
- **Semantic search**: Natural language queries with 94.2% precision, 78.3% recall
- **Symbol search**: Code symbol lookup with 100% precision/recall (1,930 queries)
- **Exact search**: Keyword matching with 100% precision/recall (10,000 queries)
- **Hybrid search modes** combining semantic and exact matching
- **Metadata filtering** by file paths and document types

#### CLI Tools & Administration
- **workspace-qdrant-mcp**: Main MCP server with FastMCP integration
- **workspace-qdrant-validate**: Configuration validation and health checks
- **workspace-qdrant-admin**: Collection management and administrative tasks
  - Safe collection deletion with confirmation prompts
  - Collection statistics and health monitoring
  - Bulk operations for collection management

#### Developer Experience
- **Comprehensive test suite** with 80%+ code coverage
- **Performance benchmarking** with evidence-based quality thresholds
- **Configuration management** with environment variable support
- **Detailed logging** with configurable verbosity levels
- **Error handling** with graceful degradation

### Performance & Quality

#### Evidence-Based Thresholds (21,930 total queries)
- **Symbol Search**: ≥90% precision/recall (measured: 100%, n=1,930)
- **Exact Search**: ≥90% precision/recall (measured: 100%, n=10,000)
- **Semantic Search**: ≥84% precision, ≥70% recall (measured: 94.2%/78.3%, n=10,000)

#### Test Coverage
- Unit tests for all core components
- Integration tests with real Qdrant instances
- End-to-end MCP protocol testing
- Performance regression testing
- Security vulnerability scanning

### Technical Architecture

#### Dependencies
- **FastMCP** ≥0.3.0 for MCP server implementation
- **Qdrant Client** ≥1.7.0 for vector database operations
- **FastEmbed** ≥0.2.0 for embedding generation
- **GitPython** ≥3.1.0 for repository integration
- **Pydantic** ≥2.0.0 for configuration management
- **Typer** ≥0.9.0 for CLI interfaces

#### Configuration
- Environment-based configuration with .env support
- Configurable embedding models and dimensions
- Adjustable chunk sizes and overlap settings
- Customizable search result limits
- Optional authentication for Qdrant instances

### Security
- Input validation for all user-provided data
- Secure credential management through environment variables
- Protection against path traversal attacks
- Sanitized logging to prevent information disclosure
- Dependency vulnerability scanning in CI/CD

### DevOps & CI/CD
- **Multi-Python support**: Python 3.8-3.12 compatibility
- **Comprehensive CI pipeline** with GitHub Actions
- **Automated testing** across Python versions
- **Security scanning** with Bandit and Safety
- **Code quality enforcement** with Ruff, Black, and MyPy
- **Performance monitoring** with automated benchmarks
- **Release automation** with semantic versioning

### Documentation
- Comprehensive README with setup instructions
- API documentation with usage examples
- Configuration guide with all available options
- Performance benchmarking methodology
- Contributing guidelines and development setup
- Security policy and vulnerability reporting

### Installation & Usage

```bash
pip install workspace-qdrant-mcp
```

#### Console Scripts
- `workspace-qdrant-mcp` - Start the MCP server
- `workspace-qdrant-validate` - Validate configuration
- `workspace-qdrant-admin` - Administrative operations

### Performance Highlights
- **High-throughput ingestion** with optimized chunking
- **Fast similarity search** with vector indexing
- **Memory-efficient operations** with streaming processing
- **Concurrent query handling** with async/await patterns
- **Caching support** for frequently accessed embeddings

---

[Unreleased]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.1.2...v0.3.0
[0.1.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/tag/v0.1.0
