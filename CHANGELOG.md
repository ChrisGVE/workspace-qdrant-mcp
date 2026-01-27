# Changelog

All notable changes to the workspace-qdrant-mcp project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3 cutover automation script (`scripts/phase3_cutover.sh`) for unified queue migration
- Queue drift detection (`wqm admin drift-report`) to compare unified and legacy queues
- Dual-write metrics for migration monitoring

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

[Unreleased]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.1.2...v0.3.0
[0.1.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/tag/v0.1.0
