# Workspace-Qdrant-MCP Consolidated Product Requirements Document v2.0

**Document Version**: 2.0 (Consolidated)
**Date**: 2025-09-15
**Status**: Foundational
**Scope**: Complete system architecture based on chronological PRD analysis

## Executive Summary

The workspace-qdrant-mcp is a **memory-driven semantic workspace platform** that has evolved from a simple Python port into a sophisticated multi-component system. This consolidated PRD establishes the **true North** architecture based on chronological analysis of 12 PRD documents spanning August-September 2025, applying precedence rules where later requirements override earlier ones.

## 1. Foundational Architecture

### 1.1 Core System Vision

**Memory-Driven AI Workspace Management**: Transform basic file indexing into intelligent workspace assistance where user preferences and behavioral rules persistently control LLM behavior across all Claude Code sessions.

### 1.2 Two-Process Architecture (FINAL)

```
┌─────────────────────┐    gRPC     ┌──────────────────────┐
│   Python MCP Server│◄──────────►│   Rust Daemon       │
│   - FastMCP tools   │             │   - File processing  │
│   - Search interface│             │   - LSP integration  │
│   - Memory mgmt     │             │   - File watching    │
│   - Claude Code API │             │   - Heavy lifting    │
└─────────────────────┘             └──────────────────────┘
```

**Communication Protocol**: gRPC with graceful fallback to direct Qdrant access
**Lifecycle Management**: Coordinated shutdown with work-in-progress cancellation support

### 1.3 Component Responsibilities

**Python MCP Server**:
- FastMCP-based tool interface for Claude Code
- Memory management and behavioral rule storage
- Search orchestration and result presentation
- Configuration management and validation

**Rust Daemon (memexd)**:
- High-performance file processing and ingestion
- LSP server integration and code enrichment
- File system watching and change detection
- Priority-based job scheduling and resource management

**WQM CLI**:
- Administrative operations and system management
- Service lifecycle control (start/stop/status)
- State database inspection and maintenance
- Debugging and health monitoring

## 2. Collection Architecture (FINAL)

### 2.1 Multi-Tenant Collection Strategy

**Architecture Principle**: Single collections shared across projects with metadata-based isolation rather than separate collections per project.

```
Multi-Tenant Collections:
├── _{project_collection_name}    (read-only, all project artifacts)
├── {root_name}-docs              (read-write, project documentation)
├── {root_name}-notes             (read-write, project notes)
├── {root_name}-scratchbook       (read-write, project workspace)
└── __memory                      (system, behavioral rules)
```

### 2.2 Collection Naming Conventions

**User Configuration**:
```yaml
workspace:
  root_name: "workspace"          # User-defined root for multi-tenant collections
  collection_types: ["docs", "notes", "scratchbook"]
  project_collection_name: "codebase"  # Name for read-only project artifacts
  memory_collection_name: "memory"     # System memory (__ prefix added internally)
```

**Resulting Collections**:
- `workspace-docs` (multi-tenant, metadata filtered by project)
- `workspace-notes` (multi-tenant, metadata filtered by project)
- `workspace-scratchbook` (multi-tenant, metadata filtered by project)
- `_codebase` (read-only, multi-tenant project artifacts)
- `__memory` (system collection, CLI-writable from MCP)

### 2.3 Collection Prefixes and Access Control

- **No prefix**: Read-write workspace collections (`workspace-docs`)
- **Single underscore `_`**: Read-only collections (`_codebase`)
- **Double underscore `__`**: System collections (`__memory`)

**Access Rules**:
- MCP Server: Read-write to workspace collections, read-only to `_` collections, special access to `__memory`
- Daemon: Full access for ingestion and processing
- CLI: Administrative access to all collections

## 3. Memory System (FINAL)

### 3.1 Memory-Driven LLM Behavior

**Core Concept**: User preferences and behavioral rules stored in persistent collections automatically injected into Claude Code sessions to control AI behavior consistently.

### 3.2 Memory Architecture

**System Memory** (`__memory`):
- User preferences: "Use uv for Python", "Call me Chris"
- Global behavioral rules: "Always make atomic commits", "Read 2000 lines before editing"
- Cross-project agent library and deployment decisions

**Project Memory** (`{project}-memory` or metadata in multi-tenant):
- Project-specific context and active agent tracking
- Local behavioral overrides and preferences
- Project history and decision logs

### 3.3 Memory Rule Structure

```python
@dataclass
class MemoryRule:
    category: MemoryCategory  # PREFERENCE, BEHAVIOR, AGENT
    authority: AuthorityLevel # ABSOLUTE, DEFAULT
    content: str             # The rule content
    context: str             # When/where it applies
    conflict_resolution: str # How to handle conflicts
```

**Authority Levels**:
- **ABSOLUTE**: Non-negotiable rules that override AI defaults
- **DEFAULT**: Overridable preferences that AI can adapt

### 3.4 Claude Code Integration

**Session Initialization**: Memory rules automatically injected into Claude Code context at session start
**Conflict Resolution**: User choice required when conflicting rules detected
**Conditional Rules**: Support for context-dependent behavioral modifications

## 4. Tool Architecture (FINAL)

### 4.1 Four Consolidated Tools

**Requirement**: Consolidate current 30+ individual tools into 4 unified tools with strict scope control.

**Tool Specification**:

1. **`qdrant_store`**: Content ingestion and document management
   - Document upload with automatic classification
   - Metadata enrichment and LSP integration
   - Batch processing and progress tracking

2. **`qdrant_find`**: Search and retrieval with scope control
   - Scope options: "collection", "project", "workspace", "all", "memory"
   - Hybrid search (dense + sparse vectors)
   - Advanced filtering and ranking

3. **`qdrant_manage`**: System administration and collection management
   - Collection operations: create/delete/rename/status
   - System health monitoring
   - Configuration validation

4. **`qdrant_read`**: Direct document retrieval without search
   - Get specific documents by ID
   - List documents with filtering
   - Metadata inspection

### 4.2 Search Scope Definitions

- **"collection"**: Specific collection (requires collection parameter)
- **"project"**: Current project collections including project memory
- **"workspace"**: Project + global collections (excludes readonly/system)
- **"all"**: Project + global + library collections (excludes system)
- **"memory"**: Both system and project memory collections

## 5. LSP Integration (FINAL)

### 5.1 Intelligent Code Processing

**Automatic LSP Detection**: Scan for available language servers (rust-analyzer, ruff, typescript-language-server, etc.)
**Dynamic Configuration**: File extension patterns derived from available LSPs
**Code Enrichment**: Symbols, relationships, type information, documentation extracted and stored

### 5.2 Language Support

**Minimum Requirement**: 10+ programming languages
**Current Implementation**: 20+ languages supported (exceeds requirement)
**Supported Languages**: Python, Rust, TypeScript, JavaScript, Java, Go, C/C++, C#, Swift, Kotlin, Scala, Ruby, PHP, and more

### 5.3 LSP Storage Strategy

**"Interface + Minimal Context"** approach:
- Store function signatures and definitions
- Exclude implementation details to reduce noise
- Include type information and documentation
- Maintain symbol relationships and dependencies

### 5.4 LSP Health Management

**No Auto-Installation**: Users maintain control over LSP installation
**Health Monitoring**: Track LSP availability and functionality
**Graceful Degradation**: System works when LSP servers unavailable
**Recovery**: Re-process files when LSPs become available

## 6. Process Priority Management (FINAL)

### 6.1 Three-Tier Priority System

**Tier 1: Command Queue Priorities**
- MCP Server active: High priority for MCP commands
- WQM commands: Queued but lower priority when MCP active
- State database tracks pending operations

**Tier 2: Background Execution Priorities**

*When MCP Server Active*:
- MCP command execution: Highest priority
- File watching: High priority
- Background ingestion: Lower priority

*When MCP Server Inactive*:
- All operations: Equal priority
- Overall process: Lower system priority to preserve GUI responsiveness
- WQM requests: Immediate handling (except watch list additions)

**Tier 3: Machine Idle Negotiation**
- Idle detection: Configurable timeout (default 30 minutes)
- Resource negotiation with other idle-aware processes
- Never prevent machine sleep or create system interference

### 6.2 Dynamic Resource Management

**Adaptive CPU Usage**: Replace fixed percentages with multicore-aware dynamic allocation
**Memory Management**: Intelligent limits based on available system resources
**I/O Throttling**: Adaptive based on system load and MCP activity

### 6.3 Processing Operations

1. **Change Detection**: Monitor file system for modifications and new files
2. **LSP-Enhanced Ingestion**: Process project files with available language servers
3. **Metadata Enrichment**: Extract and store code intelligence information

## 7. Configuration System (FINAL)

### 7.1 OS-Standard Directory Compliance

**XDG Base Directory Specification**:
- Configuration: `$XDG_CONFIG_HOME/workspace-qdrant/` or `~/.config/workspace-qdrant/`
- State: `$XDG_STATE_HOME/workspace-qdrant/` or `~/.local/state/workspace-qdrant/`
- Cache: `$XDG_CACHE_HOME/workspace-qdrant/` or `~/.cache/workspace-qdrant/`

**Platform-Specific Fallbacks**:
- macOS: `~/Library/Application Support/`, `~/Library/Logs/`, `~/Library/Caches/`
- Windows: `%APPDATA%/workspace-qdrant/`, `%LOCALAPPDATA%/workspace-qdrant/`

### 7.2 Hardcoded Pattern System

**Research-Backed Patterns**: Comprehensive inclusion/exclusion patterns embedded in system
**Custom Extensions**: Users can extend but not replace hardcoded patterns
**Pattern Coverage**: 95%+ of development scenarios without user configuration

### 7.3 Configuration Simplification Principle

**Adult User Respect**: Provide intelligent defaults, let users override when needed
**Minimal Required Configuration**: Most installations work with zero configuration
**Progressive Disclosure**: Advanced options available but not required

## 8. Logging and Protocol Compliance (FINAL)

### 8.1 Single Logging System

**Loguru-Based Architecture**: Replace fragmented logging with unified loguru system
**MCP stdio Compliance**: Perfect console silence in stdio mode
**Development Experience**: Rich console logging in verbose/development modes

### 8.2 Protocol Compliance Requirements

**Zero Console Output**: MCP stdio mode must produce no console interference
**Third-Party Suppression**: All library outputs suppressed in stdio mode
**Graceful TTY Detection**: Automatic stdio vs TTY mode detection
**Development Support**: Elegant banners and formatting in development mode

## 9. Pattern Research and Coverage (FINAL)

### 9.1 Comprehensive Language Support

**Missing from Initial Research** (User Feedback):
- OCaml, R, SQL, Perl, Visual Basic, Objective-C, PL/SQL
- Lua, Haskell, Clojure, Elixir, Elm, Erlang
- Fortran, COBOL, Forth, Pascal, Modula-2, Delphi, Zig
- Shell scripting variants (bash, zsh, fish, etc.)

### 9.2 Infrastructure and Configuration

**Required Additions**:
- Terraform configurations (`*.tf`, `*.tfvars`)
- Kubernetes manifests (`*.k8s.yaml`, kustomization files)
- Complete shell scripting ecosystem
- Additional build systems and package managers

### 9.3 Document Type Considerations

**Conservative Approach**:
- Include parseable document types (PDF, Markdown, text)
- Leave presentations and spreadsheets as user custom includes
- Focus on content that provides development context value

### 9.4 Exclusion Strategy

**Pattern Priority**: Exclusion-first for performance (user feedback: "intuitively exclusion first")
**Dot File Policy**: Generally exclude hidden files/folders with strategic exceptions
**IDE/Build Exclusions**: Comprehensive coverage for all missing language ecosystems

## 10. State Management (FINAL)

### 10.1 SQLite State Database

**Purpose**: Track daemon operations, LSP status, processing queue, and recovery state
**Location**: OS-standard state directory (`state.db`)
**Schema**: Persistent storage for job queue, LSP health, ingestion status

### 10.2 LSP Tracking Requirements

**Missing LSP Detection**: Track which LSPs are expected but unavailable
**Re-ingestion Triggers**: Monitor for newly available LSPs to refresh collections
**Health Reporting**: WQM integration for LSP status and recommendations

### 10.3 Recovery and Resilience

**Crash Recovery**: Resume processing from last known state
**Queue Persistence**: Maintain operation queue across daemon restarts
**Conflict Resolution**: Handle concurrent access and operation conflicts

## 11. Architectural Decisions (RESOLVED)

### 11.1 Collection Architecture - FINAL

**Decision**: Multi-tenant collections with project-based tenancy
- **Project collections**: Multi-tenant with metadata filtering (`workspace-docs`, `workspace-notes`, `workspace-scratchbook`)
- **Global collections**: Single tenant (no project dependency)
- **Memory collections**: Single tenant (system-wide behavioral rules)
- **Readonly collections**: Multi-tenant (library collections via wqm watch)
- **Search capability**: Both single-tenant and cross-tenant searches supported

**Multi-Tenant Implementation Details**:
- **Tenant Isolation**: Configurable (preserve isolation vs global deduplication)
- **Result Deduplication**: Content hash → file path → document ID fallback
- **Score Aggregation**: Max score, average score, or sum score methods
- **API Consistency**: TenantAwareResult converts to standard API format
- **Performance**: Score normalization and collection-level result limiting

### 11.2 Collection Naming Strategy - FINAL

**Decision**: Type-based configuration
- Users configure `collection_types: ["docs", "notes", "scratchbook"]`
- System handles internal naming (`workspace-docs`, `workspace-notes`, etc.)
- Configuration emphasizes intentionality and purpose rather than naming mechanics

### 11.3 Memory System Architecture - FINAL

**Decision**: Single memory system with clear purpose distinction
- **Memory**: Single system-wide source of truth for behavioral rules (like CLAUDE.md context injection)
- **Project information**: Rich contextual data including LSP metadata, symbols, definitions, usage patterns
- **Purpose**: One-stop semantic workspace eliminating grep searches and providing rich symbol/code context
- **Extension**: Same system captures web content/sites to reduce complex web searches

**Status**: All architectural contradictions resolved. System ready for implementation roadmap.

## 12. Configuration Architecture (FINAL)

### 12.1 Unified Configuration System
**Single Source of Truth**: One YAML configuration file serves all components (daemon, MCP server, WQM CLI)

**Configuration Hierarchy** (highest to lowest priority):
1. Command line arguments (component-specific)
2. Environment variables (`WORKSPACE_QDRANT_*`)
3. YAML configuration file (unified across components)
4. Hardcoded defaults

**File Discovery Order**:
1. `--config` parameter (explicit file path)
2. `workspace_qdrant_config.yaml` (project-specific, current directory)
3. `~/.config/workspace-qdrant/workspace_qdrant_config.yaml` (user XDG config)
4. System defaults (lowest priority)

**Extensions**: Both `.yaml` and `.yml` supported

### 12.2 Core Configuration Schema
```yaml
# Qdrant database connection
qdrant:
  url: "http://localhost:6333"    # Qdrant server URL
  api_key: null                   # Optional API key for cloud
  timeout: 30                     # Connection timeout in seconds
  prefer_grpc: true              # Use gRPC for better performance

# Text embedding configuration
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"  # Dense embedding model
  enable_sparse_vectors: true                       # Enable BM25 for hybrid search
  chunk_size: 800                                  # Text chunk size
  chunk_overlap: 120                               # Chunk overlap
  batch_size: 50                                   # Processing batch size

# Workspace and collection management
workspace:
  collection_types: ["docs", "notes", "scratchbook"]  # Multi-tenant collection types
  global_collections: []                               # Cross-project collections
  github_user: null                                   # For project detection
  custom_include_patterns: []                         # Extend default patterns
  custom_exclude_patterns: []                         # Additional exclusions

# gRPC daemon integration
grpc:
  enabled: false                  # Enable Rust daemon integration
  host: "127.0.0.1"
  port: 50051
  fallback_to_direct: true       # Fallback to direct Qdrant access
  connection_timeout: 10.0
  max_retries: 3
```

### 12.3 OS-Standard Directory Compliance
**XDG Base Directory Support**:
- `XDG_CONFIG_HOME` for configuration files
- `XDG_STATE_HOME` for runtime state (planned)
- `XDG_CACHE_HOME` for cache data (planned)
- Platform-specific fallbacks (Windows, macOS)

### 12.4 Pattern System Architecture
**Hardcoded Research-Backed Patterns**: 250+ inclusion patterns covering:
- **Source Code**: 60+ languages (C/C++, Python, JavaScript, TypeScript, Rust, Go, Java, etc.)
- **Web Development**: HTML, CSS, templates, frameworks
- **Infrastructure**: Docker, Kubernetes, Terraform, CI/CD
- **Documentation**: Markdown, text files, wikis
- **Configuration**: YAML, JSON, TOML, INI files

**Custom Extensions**: Users can extend (not replace) hardcoded patterns
- `custom_include_patterns`: Additional inclusion patterns
- `custom_exclude_patterns`: Additional exclusion patterns
- Performance-first: Exclusion patterns processed before inclusion

**Missing Language Coverage** (identified gaps):
- OCaml, R, SQL, Perl, Visual Basic, Objective-C
- Lua, Haskell, Clojure, Elixir, Elm, Erlang
- Fortran, COBOL, Forth, Pascal, Modula-2, Delphi, Zig
- All shell scripting flavors, PL/SQL, TypeScript, JavaScript variations

### 12.5 Configuration Implementation Requirements (Critical Feedback)

**Sparse Embedding Model**: Must specify which model is used for sparse BM25 embeddings
**Auto-Creation Logic**: Collections auto-created by system, no user flag needed
**gRPC Default**: Must default to `true` (performance optimization)
**Priority System Architecture**: Three-tier dynamic priority system:
1. **Command Queue Priorities**: MCP server commands take precedence over WQM commands
2. **Background Execution Priorities**:
   - MCP server up: Commands + watching take precedence, other tasks lower priority
   - MCP server down: Equal priority background, lower system priority for GUI responsiveness
   - Manual WQM requests: Immediate handling (except folder watch additions)
3. **Machine Idle Negotiation**: After configurable idle time (default 30min), negotiate with other idle processes

**Collection Naming Corrections**:
- `memory_collection_name`: User provides clean name, `__` prefix added internally
- `project_collection_name`: Replaces `code_collection_name`, read-only collection with `_` prefix
- Multi-tenant root name: User provides root, system creates `{root}-{type}` collections

**State Management**:
- Default database: `state.db` in XDG_STATE_HOME/workspace-qdrant or system canonical location
- Tracks missing LSP servers and projects needing metadata refresh
- Enables background LSP integration when servers become available

**Pattern Priority**: Exclusion-first processing for performance optimization

## 13. Implementation Roadmap

### Phase 1: Tool Consolidation (Critical Gap)
- **Current State**: 30+ individual tools (30% alignment)
- **Target**: 4 consolidated tools (qdrant_store, qdrant_find, qdrant_manage, qdrant_read)
- **Priority**: High - Major architectural misalignment

### Phase 2: Pattern Research Completion
- **Current State**: Incomplete language coverage
- **Target**: Comprehensive coverage including user-identified missing languages
- **Priority**: High - Foundation for accurate file detection

### Phase 3: Configuration Architecture Finalization
- **Current State**: Over-engineered configuration system
- **Target**: Minimal user configuration with intelligent defaults
- **Priority**: Medium - Usability improvement

### Phase 4: Memory System Integration
- **Current State**: Core system implemented, Claude Code integration unknown
- **Target**: Complete memory-driven behavior with session injection
- **Priority**: Medium - Core differentiating feature

### Phase 5: Protocol and Performance Optimization
- **Current State**: Good MCP compliance, unknown edge cases
- **Target**: Perfect stdio silence and optimal performance
- **Priority**: Low - Polish and reliability

## 13. Success Criteria

### 13.1 User Experience Success
- **Zero Configuration**: New installations work immediately
- **Memory Persistence**: User preferences and behavioral rules automatically persist
- **Intelligent Assistance**: AI behavior consistently matches user expectations

### 13.2 Technical Success
- **Protocol Compliance**: Perfect MCP stdio compliance
- **Performance**: Sub-200ms search responses for workspace queries
- **Reliability**: Graceful degradation when components unavailable
- **Resource Efficiency**: Dynamic resource management without system interference

### 13.3 Architectural Success
- **Tool Consolidation**: 4 unified tools replace 30+ individual tools
- **Multi-Component Harmony**: Seamless Python MCP + Rust daemon integration
- **Platform Universality**: Cross-platform service management (macOS, Linux, Windows)

## 14. References and Historical Architecture

### 14.1 Legacy Architecture Documentation (Extracted from specs/)

**Original TypeScript Base**: workspace-qdrant-mcp was initially conceived as a Python port of [marlian/claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) using FastMCP framework.

**Evolution to Hybrid System**: The architecture has evolved significantly from simple port to sophisticated multi-component system with Rust daemon integration.

**Core Embedding Configuration (Historical)**:
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Sparse Vectors**: BM25 implementation for hybrid search
- **Performance**: ONNX Runtime for fast inference

**Admin CLI Architecture (Preserved)**:
- **Safety Features**: Project scoping, protected collection identification, confirmation prompts
- **Commands**: list-collections, delete-collection, collection-info
- **Protection Pattern**: Automatically protects memexd daemon collections (ending with `-code`)
- **Automation Support**: Dry-run mode, force flags, batch operations

**Performance Targets (Historical)**:
- Collection detection: < 1 second for typical projects
- Embedding generation: > 100 docs/second on CPU
- Search latency: < 200ms for workspace queries
- Memory usage: < 150MB RSS when active

### 14.2 Documentation Integration Summary

**Comprehensive Documentation Available** (extracted from docs/):
- API specifications and tool interfaces
- Enterprise deployment patterns (RBAC, authentication, multitenancy)
- Container orchestration (Docker, Kubernetes)
- Cross-platform compilation guides
- Observability and monitoring patterns
- Release automation and trusted publishing
- Configuration management and pattern exclusions
- SSL optimization and warning resolution guides

**Production Monitoring Stack** (extracted from monitoring/):
- **Architecture**: Multi-layer monitoring with Prometheus, Grafana, Alertmanager
- **Components**: Python MCP Server (8000), Rust Engine (8002), Qdrant (6333)
- **Metrics Collection**: Prometheus (9090), Health Checks (8080), Node Exporter (9100)
- **Deployment**: Docker Compose with runbooks and alert configurations
- **Log Aggregation**: Centralized logging with multiple output targets

### 14.3 Technical References
- FastMCP framework documentation
- Qdrant client API specifications
- FastEmbed model documentation
- Rust-Python interop patterns
- gRPC protocol specifications

## 15. Conclusion

This consolidated PRD establishes the **true North** for workspace-qdrant-mcp as a memory-driven semantic workspace platform. The system succeeds when developers experience seamless, memory-aware AI assistance with zero configuration overhead while maintaining complete protocol compliance and cross-platform reliability.

**Architecture Status**: All three critical contradictions resolved (Section 11) - ready for implementation roadmap execution.