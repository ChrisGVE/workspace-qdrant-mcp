# Changelog

All notable changes to the workspace-qdrant-mcp project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.1.2...HEAD
[0.3.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/compare/v0.1.2...v0.3.0
[0.1.0]: https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/tag/v0.1.0
