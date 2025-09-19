# Workspace-Qdrant-MCP Consolidated Product Requirements Document v2.0

**Document Version**: 2.0 (Consolidated)
**Target System Version**: 0.3.0dev0
**Date**: 2025-09-15
**Status**: Foundational
**Scope**: Complete system architecture based on chronological PRD analysis

## First Principles

Before defining the "true North" architecture, we establish the fundamental first principles that guide all design decisions:

1. **No rewrite, no reinventing the wheel**: When a library/crate/package exist, is maintained, and relatively popular (some might be very specialized with a small user population) we always give them preference over writing code from scratch.
2. **Memory-First Design**: User preferences and behavioral rules are the primary drivers of system behavior, not secondary features
3. **Zero Configuration Principle**: The system works immediately upon installation without user configuration for 95% of development scenarios
4. **Intelligent Degradation**: Every component gracefully handles the absence of dependencies while maintaining core functionality
5. **Performance Through Exclusion**: Exclusion patterns are processed first for optimal performance, with strategic inclusions as exceptions
6. **Adult User Respect**: Provide intelligent defaults while allowing complete user control when needed
7. **Multi-Tenant Efficiency**: Share collections across projects with metadata isolation rather than collection proliferation
8. **Component Autonomy**: Each component (MCP server, daemon, CLI) operates independently while cooperating seamlessly
9. **Resource Awareness**: Dynamic resource allocation based on system state and user activity, never interfering with user experience

## Executive Summary

The workspace-qdrant-mcp is a **memory-driven semantic workspace platform** that has evolved from a simple Python port into a sophisticated multi-component system. This consolidated PRD establishes the **true North** architecture based on chronological analysis of 12 PRD documents spanning August-September 2025, applying precedence rules where later requirements override earlier ones.

## 1. Foundational Architecture

### 1.1 Core System Vision

**Memory-Driven AI Workspace Management**: Transform basic file indexing into intelligent workspace assistance where user preferences and behavioral rules persistently control LLM behavior across all Claude Code sessions.

### 1.2 Two-Process Architecture (FINAL)

```
┌─────────────────────┐    gRPC     ┌──────────────────────┐
│   Python MCP Server │◄──────────► │   Rust Daemon        │
│   - FastMCP tools   │             │   - File processing  │
│   - Search interface│             │   - LSP integration  │
│   - Memory mgmt     │             │   - File watching    │
│   - Claude Code API │             │   - Heavy lifting    │
│   - Qdrant connect  │             │   - Qdrant connect   │
└─────────────────────┘             └──────────────────────┘
           │                                   │
           └──────────── Qdrant ───────────────┘
```

[Question: I have seen that there are other protocols, such as JSON-RPC. Now, it is not clear to me whether JSON_RPC is a really different protocol or if it uses/or is based on gRPC. Since LLM are excellent at using JSON, I was wondering if this could have been an option, or if gRPC is anyway the "right" answer in our use case.]

**Communication Protocol**: gRPC is the primary and default protocol used between all the components. When the protocol does not work for a query, we fallback to direct access. The next access uses again the gRPC protocol, only after X attempts (defined by configuration, default 5) and try again using gRPC after Y minutes (defined by configuration default 10) [Do you think these default values are sensible?]
**Lifecycle Management**: Coordinated shutdown with work-in-progress cancellation support
**Database Access**: Both components can directly connect to Qdrant for optimal performance

### 1.3 Component Responsibilities

**Python MCP Server**:

- FastMCP-based tool interface for Claude Code (and other MCP compliant clients)
- Memory management and behavioral rule storage
- Search orchestration and result presentation
- Configuration management and validation

**Rust Daemon (memexd)**:

- High-performance file processing and ingestion
- LSP server integration and code enrichment
- File system watching and change detection
- Priority-based job scheduling and resource management

**WQM CLI**:

- Administrative operations and system management (including web interface)
- Service lifecycle control (start/stop/status)
- State database inspection and maintenance
- Debugging and health monitoring
- Collection management and document ingestion
- Folder watching registration and management

## 2. Collection Architecture (FINAL)

### 2.1 Multi- and Single-Tenant Collection Strategy

**Architecture Principle**: Efficiently share collections across projects with metadata-based isolation rather than separate collections per project, while maintaining single-tenant collections for specialized use cases.

```
Multi-Tenant Collections:
├── _{project_collection_name}    (read-only, all project artifacts, auto-ingested)
├── {root_name}-docs              (read-write, project documentation)
├── {root_name}-notes             (read-write, project notes)
├── {root_name}-scratchbook       (read-write, project workspace)
├── __memory                      (system, behavioral rules)
└── Library collections           (read-only, managed via WQM)

Single-Tenant Collections:
├── Global collections            (read-write, cross-project scope)
└── Library collections           (read-only, managed via WQM)
```

**Prefix Conventions and Collision Handling**:

- **Single underscore `_`**: Read-only collections (`_project` → exposed as `project`)
- **Double underscore `__`**: System collections (not accessible via MCP server)
- **No prefix**: Regular read-write collections
- **Collision Prevention**: Cannot have both `_collection` and `collection` - first created wins
- **Server and WQM Restrictions**: Cannot create collections that collide with configuration-defined schemas (i.e. any name that would match any of the configured patterns)

### 2.2 Collection Naming Conventions

**User Configuration**:

```yaml
workspace:
  root_name: "workspace" # User-defined root for multi-tenant collections, default: "workspace"
  collection_types: ["docs", "notes", "scratchbook"] # User-defined, no default
  project_collection_name: "codebase" # User-defined name for read-only project artifacts, default: "project" (_ prefix added internally)
  memory_collection_name: "memory" # User-defined name for System memory, default "llm_rules" (__ prefix added internally)
```

**Resulting Collections**:

- `workspace-docs` (multi-tenant, metadata filtered by project)
- `workspace-notes` (multi-tenant, metadata filtered by project)
- `workspace-scratchbook` (multi-tenant, metadata filtered by project)
- `_codebase` (read-only, multi-tenant project artifacts, metadata filtered by project)
- `__memory` (system collection, CLI-writable only)

### 2.3 Access Control Matrix

**MCP Server Access**:

- **Workspace collections** (no prefix): Full read-write access
- **Read-only collections** (`_` prefix): Read-only access, exposed without prefix
- **System collections** (`__` prefix): No access, auto-injected into context
- **Project Scope**: Access to current project and subprojects (GitHub/GitLab submodules from same user)
- **Repository Detection**: Local repositories considered as belonging to user
- **Tenant Management**: Sub-projects use existing tenant, no new tenant creation
- **Platform Support**: GitHub (current), GitLab (future release)

**Rust Daemon Access**:

- **All collections**: Full access for ingestion and processing
- **No restrictions**: Filtering handled upstream by caller
- **Purpose**: High-performance background processing and LSP integration

**WQM CLI Access**:

- **Configuration-defined collections**: Cannot delete or rename (protection)
- **Project management**: Can remove projects with safeguards if no longer present on storage
- **Collection creation**: Can create both single-tenant and multi-tenant collections
- **Content management**: Can add documents, folders, webpages, websites to any collection
- **Folder watching**: Can register/remove watched folders with optional data cleanup
- **System collections**: Full CRUD access (content stored as strings)
- **Multi-tenant folder mapping**: Folder becomes tenant, subfolders become separate tenants (\*)
- **Single-tenant library**: single monolithic library coming from a single folder and subfolders (\*)
- **Multi-tenant library**: Library containing multiple folders (each folder is a tenant), distinct from the previous folder mapping variant as the folder mapping is automated, i.e., all folder found becomes a tenant. In this case the tenants are added manually. (\*)

  (\*) each of these libraries can be either "one-shot" folders which are not watched, or "watched folders"

## 3. Memory System (FINAL)

### 3.1 Memory-Driven LLM Behavior

**Core Concept**: User preferences and behavioral rules stored in persistent collections automatically injected into Claude Code sessions to control AI behavior consistently. Rules are either defined as `{name}, {string}` or `{name}, {markdown}`, markdown being ingested as a file with the WQM tool.

**Integration Method**: Claude Code hook-based injection requiring MCP server endpoint to provide memory content to the hook script. The `name` are identifiers for the LLM, i.e., the user can save a rules, which after injection can be used as "use rule 'Git discipline'".

### 3.2 Memory Architecture

**System Memory** (`__memory`):

- **User Preferences**: "Use uv for Python", "Call me Chris"
- **Global Behavioral Rules**: "Always make atomic commits", "Read 2000 lines before editing"
- **Cross-Project Decisions**: Agent library choices, deployment patterns, coding standards

**Project Memory** (via collection types):

- **No Auto-Ingestion**: Memories are explicitly managed, not automatically captured
- **User-Controlled Orchestration**: Directives for retrieving information from specific collection types
- **Custom Applications**: Project context, agent tracking, decision logs, PRD history
- **Rule-Driven**: Behavior controlled via system prompts or rules stored in system memory

### 3.3 Memory Rule Structure

```python
@dataclass
class MemoryRule:
    label: str               # User-friendly rule label for easy retrieval
    tag: str                 # Classification tag for organization
    pos: int                 # LLM injection position (user-controlled via CLI)
    category: MemoryCategory # PREFERENCE, BEHAVIOR, AGENT
    content: str             # The actual rule content
    context: str             # When/where the rule applies
```

**Rule Authority**: All rules are **non-negotiable** - the AI must strictly follow them at all times without exception.

### 3.4 Claude Code Integration

**Session Initialization**: Memory rules automatically injected at Claude Code session start via hook mechanism
**Hook Endpoint**: MCP server provides dedicated endpoint for memory content retrieval
**Conflict Resolution**: System requires user choice when conflicting rules are detected

## 4. Tool Architecture (FINAL)

### 4.1 Four Consolidated Tools

**Tool Specification**:

1. **`qdrant_store`**: Content ingestion and document management
   - **Content Types**: Strings, documents (by path), folders (with/without watching)
   - **Web Content**: Individual webpages, entire websites (top-level page + linked pages)
   - **Exclusions**: Does NOT include automatic project folder storage (handled by daemon at startup)
   - **Target Collections**: Single-tenant, multi-tenant, and library collections

2. **`qdrant_find`**: Search and retrieval with scope control
   - **Hybrid Search**: Dense + sparse vectors with reciprocal rank fusion
   - **Advanced Features**: Filtering, ranking, metadata-based result refinement
   - **Scope Options**:
     - `"collection"`: Specific collection (requires collection + project parameters)
     - `"project"`: Current project collection (read-only code content)
     - `"workspace"`: All collection_types (docs, notes, scratchbook)
     - `"extended_workspace"`: Project + workspace combined
     - `"global"`: All global collections and server-created collections
     - `"knowledge"`: All read-only collections except project + global
     - `"all"`: Everything except system collections

3. **`qdrant_manage`**: System administration and collection management
   - **Collection Operations**: Create/delete/rename (global collections only, not pre-defined)
   - **Collection Listing**: Returns non-system collections with attributes:
     - Name (without `_` prefix for read-only collections)
     - Type (single-tenant vs multi-tenant)
     - Access level (read-only vs read-write)
     - Status ("clean" vs "dirty")
     - Progress (100% if clean, percentage if dirty, "n/d" if unknown)
   - **Project Status**: Dedicated endpoint for current project collection status
   - **Health Monitoring**: Both `memexd` daemon and Qdrant server health
   - **System Status**: Ingestion progress, synchronization state, folder monitoring

4. **`qdrant_read`**: Direct document retrieval without search
   - **Direct Access**: Get specific documents by ID
   - **Filtered Listing**: List documents with metadata-based filtering
   - **Metadata Inspection**: Full document metadata and relationship analysis
   - **Symbol-Based Retrieval**: [QUESTION: Can symbols be used for direct read, or is search required?]

### 4.2 Search Scope Definitions

None of the searches include any system collections

- **"collection"**: Specific collection (requires collection parameter, name and for multi-tenant the project)
- **"project"**: Current project collection (read/only) returns information about the code
- **"workspace"**: Workspace (all collection_types)
- **"extended_workspace"**: project + workspace
- **"global"**: all global collections, and server created collections
- **"knowledge"**: all read-only collections except the project collection + global
- **"all"**: project + extended_workspace + global + knowledge

## 5. Language Processing Architecture (FINAL)

### 5.1 Mutually Exclusive LSP vs Tree-sitter Strategy

**Architecture Decision**: LSP servers take absolute priority over Tree-sitter when available. Tree-sitter serves as fallback for languages without LSP support, ensuring comprehensive language coverage while leveraging existing solutions.

**Processing Hierarchy**:
1. **LSP Server Available**: Full semantic analysis with symbols, types, relationships
2. **LSP Unavailable + Tree-sitter Grammar**: Syntax-aware parsing with AST extraction
3. **Neither Available**: Text-only processing with pattern-based heuristics

**Performance Optimization**: Single LSP selection per language for optimal performance, no alternative LSP support

### 5.2 Comprehensive Language Support (Research-Backed)

**Research Foundation**: Completed systematic analysis covering 150+ languages from Wikipedia's programming language taxonomy with verified LSP server availability and Tree-sitter grammar coverage.

**Tier 1: Premier LSP Support** (Production-Ready):
- **Systems**: C/C++ (clangd), Rust (rust-analyzer), C# (omnisharp-roslyn)
- **Web/Modern**: TypeScript/JavaScript (typescript-language-server), Go (gopls), Dart (dartls)
- **Enterprise**: Java (eclipse.jdt.ls), Python (ruff-lsp), Swift (sourcekit-lsp)
- **Functional**: Haskell (haskell-language-server), Elixir (elixir-ls), F# (fsautocomplete)
- **Scripting**: Bash (bash-language-server), PowerShell (powershell-es)

**Tier 2: Good LSP Support** (Viable):
- **Legacy/Specialized**: COBOL, Fortran (fortls), Perl (perl-languageserver)
- **Graphics/Game**: GLSL (glsl-language-server), GDScript (Godot built-in)
- **Mathematical**: R (languageserver), MATLAB (matlab-language-server)
- **Database**: SQL (sqls), PL/SQL

**Tree-sitter Fallback Coverage**: For languages with Tree-sitter grammars but no viable LSP servers, providing syntax-aware parsing for 50+ additional languages including:
- **Legacy**: Assembly variants, AWK, COBOL dialects, Fortran variants
- **Domain-Specific**: Verilog, VHDL, Gnuplot, TeX/LaTeX
- **Configuration**: Nginx, Apache, INI variants
- **Esoteric**: Educational and research languages

### 5.3 Language Detection and Environment Handling

**Detection Hierarchy**:
1. **Project Signatures**: Language-specific configuration files (.cargo/config.toml, package.json, pyproject.toml)
2. **File Extensions**: Comprehensive mapping based on research (see Appendix A.1)
3. **Content Analysis**: Shebang patterns, magic numbers, syntax signatures
4. **LSP/Tree-sitter Availability**: Preference for languages with better parsing support

**Environment Requirements**:
- **Python**: Virtual environment detection (.venv/, venv/, poetry.lock)
- **Rust**: Cargo workspace awareness, target detection
- **Node.js**: package.json, node_modules, workspace detection
- **Java**: Maven/Gradle project structure, classpath management
- **Multi-language Projects**: Polyglot project detection with primary language inference

### 5.4 Tree-sitter Integration Strategy

**Grammar Storage**: `$XDG_DATA_HOME/workspace-qdrant/tree-sitter-grammars/` with automatic grammar download and compilation

**Tree-sitter Capabilities**:
- **Syntax Tree Extraction**: AST nodes for navigation and symbol detection
- **Error Recovery**: Graceful handling of incomplete or syntactically incorrect code
- **Incremental Parsing**: Efficient updates for file changes
- **Query Language**: Tree-sitter queries for symbol extraction and pattern matching

**Fallback Processing Pipeline**:
```
File Detection → Tree-sitter Grammar Available? → Parse AST → Extract Symbols → Store Metadata
               ↘ No Grammar Available → Text Processing → Pattern Heuristics → Basic Metadata
```

### 5.5 LSP Selection Criteria and Configuration

**Opinionated LSP Selection** (aligned with "Leverage Existing Solutions" principle):
- **Speed**: Fast startup and response times
- **Multi-platform**: Consistent behavior across macOS, Linux, Windows
- **Active Development**: Regular updates and community support
- **Standard Compliance**: Full LSP protocol compliance
- **Resource Efficiency**: Reasonable memory and CPU usage

**Environment-Specific Configuration**:
- **Python**: Detect virtual environments and Python version
- **Rust**: Cargo.toml workspace awareness, target triple detection
- **TypeScript**: tsconfig.json detection, workspace configuration
- **Java**: Build tool detection (Maven/Gradle), classpath resolution

### 5.6 Code Intelligence Storage Strategy

**"Interface + Minimal Context"** approach:
- **Symbol Definitions**: Function signatures, type definitions, interface contracts
- **Relationship Mapping**: Import/export dependencies, call graphs, inheritance hierarchies
- **Metadata Enrichment**: Documentation strings, parameter types, return types
- **Cross-Reference Links**: Declaration sites, usage locations, related symbols
- **Implementation Exclusion**: Avoid storing function bodies (available in project documents)

**LSP vs Tree-sitter Data Distinction**:
- **LSP Data**: Semantic information, type analysis, cross-references, diagnostics
- **Tree-sitter Data**: Syntactic structure, AST nodes, pattern-based symbol extraction
- **Unified Storage**: Both sources contribute to the same symbol collection with metadata indicating source

### 5.7 Language Processing Health Management

**No Auto-Installation Philosophy**: Users maintain control over LSP server and Tree-sitter grammar installation

**Health Monitoring**:
- **LSP Server Status**: Availability, version, performance metrics
- **Tree-sitter Grammar Status**: Installed grammars, compilation status
- **Processing Coverage**: Files processed with LSP vs Tree-sitter vs text-only
- **Error Tracking**: Failed language server launches, parsing errors

**Graceful Degradation Pipeline**:
```
LSP Available → Full Semantic Processing
     ↓ Failure
Tree-sitter Grammar → Syntax-Aware Processing
     ↓ Failure
Text Processing → Pattern-Based Extraction
```

**State Database Tracking**:
- Files processed without LSP metadata requiring re-processing
- Languages without LSP servers for user awareness
- Tree-sitter grammars available but not installed
- Processing quality metrics per file/language

**WQM Health Reporting**:
- Missing LSP servers with installation recommendations
- Available Tree-sitter grammars for unsupported languages
- Processing coverage statistics by language
- Performance metrics and bottleneck identification

## 6. Process Priority Management (FINAL)

### 6.1 Three-Tier Priority System

**Tier 1: Command Queue Priorities**

- MCP Server active: High priority for MCP commands
- WQM commands: Queued but lower priority when MCP active, high priority when MCP inactive
- State database tracks pending operations
- State database tracks list of watched folder
  [I would suggest that we list all that the state database keep track of such that we don't have any surprise]

**Tier 2: Background Execution Priorities**

_When MCP Server Active_:

- MCP command execution: Highest priority
- Current Project Folder watching: High priority
- Non-project and other Project Folder watching and ingestion: Low priority when project collection is dirty or when executing an MCP server command, High when project collections is clean

_When MCP Server Inactive_:

- All operations: Equal priority
- Overall process: Non-greedy prioritization, lower system priority to preserve computer responsiveness for the user
- WQM requests: Immediate handling, via the queue for new folder watching or for folder ingestion without watching

**Tier 3: Machine Idle Negotiation**

- Idle detection: Configurable timeout (default 30 minutes)
- Resource negotiation with other idle-aware processes
- When all queued ingestions are completed, never prevent machine sleep, or create system interference

### 6.2 Dynamic Resource Management

**Adaptive CPU Usage**: Multicore-aware dynamic allocation replacing fixed percentages
**Memory Management**: Intelligent limits based on real-time system resource availability
**I/O Throttling**: Adaptive throttling based on system idle state, load, and MCP activity

**System Dependencies and Rust Crates**:

- **Idle Detection**: `sysinfo` crate for cross-platform system monitoring
- **Resource Monitoring**: `sysinfo` + `psutil` equivalent functionality
- **CPU Usage**: Native OS APIs via `sysinfo` for accurate multicore awareness
- **Cross-Platform**: Single crate approach for universal compatibility

### 6.3 Processing Operations

1. **Change Detection**: File system monitoring for modifications and new files
2. **LSP-Enhanced Ingestion**: Process project files with available language servers
3. **Metadata Enrichment**: Extract and store code intelligence information

**File System Monitoring**:

- **Recursive Watching**: `notify` crate for cross-platform file system events
- **Performance**: Efficient recursive monitoring without polling
- **Event Filtering**: Smart filtering to reduce noise from temporary files

## 7. Configuration System (FINAL)

### 7.1 OS-Standard Directory Compliance

**XDG Base Directory Specification (Linux)**:

- **Configuration**: `$XDG_CONFIG_HOME/workspace-qdrant/` or `~/.config/workspace-qdrant/`
- **State**: `$XDG_STATE_HOME/workspace-qdrant/` or `~/.local/state/workspace-qdrant/`
- **Cache**: `$XDG_CACHE_HOME/workspace-qdrant/` or `~/.cache/workspace-qdrant/`
- **Logs**: `$XDG_STATE_HOME/workspace-qdrant/logs/` or `~/.local/state/workspace-qdrant/logs/`

**Platform-Specific Paths**:

**macOS**:

- **Configuration**: `~/Library/Application Support/workspace-qdrant/`
- **State**: `~/Library/Application Support/workspace-qdrant/state/`
- **Cache**: `~/Library/Caches/workspace-qdrant/`
- **Logs**: `~/Library/Logs/workspace-qdrant/`

**Windows**:

- **Configuration**: `%APPDATA%/workspace-qdrant/`
- **State**: `%LOCALAPPDATA%/workspace-qdrant/state/`
- **Cache**: `%LOCALAPPDATA%/workspace-qdrant/cache/`
- **Logs**: `%LOCALAPPDATA%/workspace-qdrant/logs/`

**Linux** (XDG fallbacks):

- **Configuration**: `~/.config/workspace-qdrant/`
- **State**: `~/.local/state/workspace-qdrant/`
- **Cache**: `~/.cache/workspace-qdrant/`
- **Logs**: `~/.local/state/workspace-qdrant/logs/`

### 7.2 Hardcoded Pattern System

**Research-Backed Patterns**: Comprehensive inclusion/exclusion patterns embedded in system.
**Custom Extensions**: Users can extend but not replace hardcoded patterns.
**Pattern Coverage**: 95%+ of development scenarios without user configuration.
**Full documentation**: All inclusion/exclusion patterns documented for the user, all project detection patterns documented for the user.

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

**Zero Console Output**: MCP stdio mode must produce no console interference (standard FastMCP output when launched from the terminal)
**Third-Party Suppression**: All library outputs suppressed in stdio mode
**Graceful TTY Detection**: Automatic stdio vs TTY mode detection
**Development Support**: Elegant banners and formatting in development mode

## 9. Pattern Research and Coverage (FINAL)

### 9.1 Comprehensive Language Support

**Research Scope**: Comprehensive coverage based on systematic analysis
**Primary Source**: [Wikipedia List of Programming Languages](https://en.wikipedia.org/wiki/List_of_programming_languages) and related taxonomies

**Initial Missing Languages** (User Feedback):

- **Functional**: OCaml, Haskell, Clojure, Elixir, Elm, Erlang
- **Legacy/Specialized**: COBOL, Fortran, Pascal, Modula-2, Forth, Delphi
- **Modern**: Zig, Lua
- **Database**: SQL, PL/SQL, R
- **Microsoft Stack**: Visual Basic, Objective-C
- **Shell Variants**: bash, zsh, fish, PowerShell, nushell

**Research Requirements**:

- **Systematic Coverage**: Analysis of Wikipedia's complete language taxonomy
- **File Extension Mapping**: Comprehensive extension-to-language mapping
- **Priority-Based**: Common languages first, specialized languages included for completeness
- **Maintenance**: Regular updates as new languages emerge

### 9.2 Infrastructure and Configuration

**Required Additions**:

- **Infrastructure as Code**: Terraform (`*.tf`, `*.tfvars`), CloudFormation, ARM templates
- **Container Orchestration**: Kubernetes manifests (`*.k8s.yaml`), Kustomization files, Helm charts
- **Shell Ecosystem**: Complete coverage (sh, csh, tcsh, scsh, ksh, zsh, ash, cmd, PowerShell, fish, nushell)
- **Build Systems**: All modern package managers and build tools across ecosystems

### 9.3 Document Type Considerations

**Conservative Approach**:

- **Parseable Documents**: PDF, Markdown, text files, e-books (epub, mobi)
- **Presentations/Spreadsheets**: User custom includes (track parse failures in state)
- **Focus**: Content that provides development context value

**OCR and Parse Failure Tracking**:

- **Unparseable PDFs**: Track OCR-required PDFs in state database
- **Query Interface**: User can query which files failed parsing
- **State Management**: Maintain registry of parse attempts and failures

**Future Enhancement - Image Extraction**:

- **Document Images**: Extract embedded images from documents (not scanned pages)
- **Image Collection**: Dedicated collection with source metadata linking
- **Bidirectional Links**: Source documents link to extracted images
- **Use Cases**: Diagrams, charts, technical illustrations in documentation

### 9.4 Exclusion Strategy

**Pattern Priority**: Exclusion-first for performance (user feedback: "intuitively exclusion first, such that we can make exception in the inclusion section")
**Dot File Policy**: Generally exclude hidden files/folders with strategic exceptions
**IDE/Build Exclusions**: Comprehensive coverage for all missing language ecosystems

## 10. State Management (FINAL)

### 10.1 SQLite State Database

**Purpose**: Comprehensive state tracking for daemon operations, LSP status, processing queues, error handling, folder watching registry, and recovery state management.

**Location**: OS-standard state directory (`state.db`)

**Schema Design Philosophy**: Generic schema approach to prevent schema explosion
**Core Tables**:

- **`operations`**: Generic job queue with JSON metadata for operation-specific data
- **`resources`**: Generic resource tracking (files, folders, LSPs) with type and status
- **`events`**: Generic event log with timestamps and JSON payloads
- **`configuration`**: Key-value pairs for runtime configuration state

[This should be reviewed in light of the known use-cases, and then generalized without adding unnecessary complexity, the list can always be increased later]

[An option could be to set up the tables as we would do in JSON with concepts like key-value pairs, lists, and combinations. Maybe this could lead to another database system, as long as it support transactions]

**Transaction based operations**: All database updates are made within a transaction environment.

**Benefits**:

- **Flexibility**: Single schema handles multiple operation types
- **Maintainability**: Reduced complexity compared to specialized tables
- **Extensibility**: Easy addition of new operation types without schema changes

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
- **Readonly collections**: Multi-tenant (library collections via wqm watch, option for single-tenant library as well, use case: very large libraries)
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
2. YAML configuration file (unified across components)
3. Hardcoded defaults

**Environment Variable**: API keys only, using `${ENV_VAR}` syntax in YAML configuration

**File Discovery Order**:

1. `--config` parameter (explicit file path)
2. `workspace_qdrant_config.yaml` (project-specific, current directory)
3. `~/.config/workspace-qdrant/workspace_qdrant_config.yaml` (user XDG config)
4. System defaults (lowest priority)

**Extensions**: Both `.yaml` and `.yml` supported

### 12.2 Complete Configuration Schema

**Full Configuration Template** (with defaults and documentation):

```yaml
# Qdrant Vector Database Connection
qdrant:
  url: "http://localhost:6333" # Qdrant server URL
  api_key: ${QDRANT_API_KEY} # Optional API key for Qdrant Cloud
  timeout: 30 # Connection timeout in seconds
  prefer_grpc: true # Use gRPC protocol for optimal performance

# Text Embedding Configuration
embedding:
  # Dense Vector Embeddings
  dense_model: "sentence-transformers/all-MiniLM-L6-v2" # Primary embedding model (384-dim)

  # Sparse Vector Embeddings (BM25-style)
  enable_sparse_vectors: true # Enable hybrid search capabilities
  sparse_model: "bm25" # Sparse embedding implementation

  # Image Embeddings (Future Enhancement)
  image_model: null # Image embedding model (planned)
  enable_image_embeddings: false # Enable image content processing

  # Text Processing Parameters
  chunk_size: 800 # Text chunk size for processing
  chunk_overlap: 120 # Overlap between chunks
  batch_size: 50 # Processing batch size
  max_tokens: 8192 # Maximum tokens per document

# Workspace and Collection Management
workspace:
  # Multi-tenant Collection Configuration
  root_name: "workspace" # Root name for multi-tenant collections
  collection_types: ["docs", "notes", "scratchbook"] # Collection types to create

  # Project Detection
  project_collection_name: "project" # Name for read-only project collection (_project)
  memory_collection_name: "llm_rules" # Name for system memory (__llm_rules)

  # Repository Integration
  github_user: null # GitHub username for project detection
  gitlab_user: null # GitLab username (future release)

  # Global Collections
  global_collections: [] # Cross-project single-tenant collections

  # Pattern Customization
  custom_include_patterns: [] # Additional inclusion patterns
  custom_exclude_patterns: [] # Additional exclusion patterns

# Inter-Process Communication
grpc:
  enabled: true # Enable Rust daemon integration (default: true)
  host: "127.0.0.1" # gRPC server host
  port: 50051 # gRPC server port

  # Fallback Configuration
  fallback_to_direct: true # Fallback to direct Qdrant on gRPC failure
  max_retry_attempts: 5 # Retry attempts before fallback
  retry_interval_minutes: 10 # Minutes to wait before retrying gRPC

  # Connection Parameters
  connection_timeout: 10.0 # Connection timeout in seconds
  request_timeout: 30.0 # Individual request timeout
  max_retries: 3 # Maximum connection retries

# Language Processing Integration
language_processing:
  # LSP Server Configuration
  lsp:
    enabled: true # Enable LSP integration
    health_check_interval: 300 # LSP health check interval (seconds)
    timeout: 10.0 # LSP request timeout

    # Opinionated LSP Selection (single choice per language)
    servers:
      python: "ruff-lsp" # Fast Python LSP
      rust: "rust-analyzer" # Standard Rust LSP
      typescript: "typescript-language-server" # TypeScript/JavaScript LSP
      javascript: "typescript-language-server" # JavaScript via TypeScript LSP
      go: "gopls" # Official Go LSP
      java: "eclipse.jdt.ls" # Eclipse Java LSP
      c: "clangd" # LLVM C/C++ LSP
      cpp: "clangd" # LLVM C/C++ LSP
      csharp: "omnisharp-roslyn" # C# LSP
      swift: "sourcekit-lsp" # Apple Swift LSP
      bash: "bash-language-server" # Bash scripting LSP
      powershell: "powershell-es" # PowerShell LSP
      haskell: "haskell-language-server" # Haskell LSP
      elixir: "elixir-ls" # Elixir LSP
      erlang: "erlang_ls" # Erlang LSP
      gleam: "gleam" # Built-in Gleam LSP
      gdscript: "godot" # Godot built-in LSP
      glsl: "glsl-language-server" # OpenGL Shading Language
      r: "languageserver" # R statistical language
      sql: "sqls" # SQL LSP
      # Additional LSPs from research findings

  # Tree-sitter Fallback Configuration
  tree_sitter:
    enabled: true # Enable Tree-sitter fallback parsing
    grammar_path: "${XDG_DATA_HOME}/workspace-qdrant/tree-sitter-grammars" # Grammar storage location
    auto_install_grammars: false # Manual grammar management

    # Grammar priorities for languages with multiple parsers
    preferred_grammars:
      assembly: "tree-sitter-asm" # Assembly language grammar
      verilog: "tree-sitter-verilog" # Hardware description language
      vhdl: "tree-sitter-vhdl" # Hardware description language
      latex: "tree-sitter-latex" # LaTeX document preparation
      gnuplot: "tree-sitter-gnuplot" # Plotting language
      nginx: "tree-sitter-nginx" # Nginx configuration

  # Language Detection Configuration
  detection:
    # Project signature patterns (highest priority)
    project_signatures:
      python: ["pyproject.toml", "requirements.txt", "setup.py", "Pipfile"]
      rust: ["Cargo.toml", "Cargo.lock"]
      javascript: ["package.json", "package-lock.json", "yarn.lock"]
      typescript: ["tsconfig.json", "package.json"]
      java: ["pom.xml", "build.gradle", "gradle.properties"]
      go: ["go.mod", "go.sum"]
      csharp: ["*.csproj", "*.sln", "project.json"]

    # Environment detection patterns
    environments:
      python: [".venv/", "venv/", "poetry.lock", "conda-meta/"]
      node: ["node_modules/", ".nvmrc", ".node-version"]
      rust: ["target/", "Cargo.lock"]
      java: [".m2/", "target/", "build/"]

# Logging and Development
logging:
  level: "INFO" # Log level: DEBUG, INFO, WARN, ERROR
  console_output: false # Console output in stdio mode
  file_logging: true # Enable file logging
  max_file_size: "10MB" # Maximum log file size
  backup_count: 5 # Number of backup log files

# Performance and Resource Management
performance:
  # CPU and Memory
  max_cpu_cores: null # Max CPU cores (null = auto-detect)
  memory_limit_mb: null # Memory limit (null = auto-detect)

  # Processing Priorities
  idle_timeout_minutes: 30 # Machine idle detection timeout
  background_priority: "low" # Background process priority

  # File System Monitoring
  watch_debounce_ms: 500 # File change debounce interval
  max_watch_depth: 10 # Maximum directory depth for watching

# Development and Debug Options
debug:
  enabled: false # Enable debug mode
  profile_performance: false # Enable performance profiling
  verbose_lsp: false # Verbose LSP communication logging
  verbose_tree_sitter: false # Verbose Tree-sitter parsing logs
  trace_requests: false # Trace all MCP requests
```

### 12.3 OS-Standard Directory Compliance

**XDG Base Directory Support**:

- `XDG_CONFIG_HOME` for configuration files
- `XDG_STATE_HOME` for runtime state and processing queues
- `XDG_CACHE_HOME` for embedding cache and temporary processing
- `XDG_DATA_HOME` for Tree-sitter grammar storage and language models
- Platform-specific fallbacks (Windows, macOS, Linux)

**Tree-sitter Grammar Storage**:
- **Linux**: `$XDG_DATA_HOME/workspace-qdrant/tree-sitter-grammars/` or `~/.local/share/workspace-qdrant/tree-sitter-grammars/`
- **macOS**: `~/Library/Application Support/workspace-qdrant/tree-sitter-grammars/`
- **Windows**: `%LOCALAPPDATA%/workspace-qdrant/tree-sitter-grammars/`

### 12.4 Pattern System Architecture

**Hardcoded Research-Backed Patterns**: 250+ inclusion patterns covering:

- **Source Code**: 60+ languages (C/C++, Python, JavaScript, TypeScript, Rust, Go, Java, etc.)
- **Web Development**: HTML, CSS, templates, frameworks
- **Infrastructure**: Docker, Kubernetes, Terraform, CI/CD
- **Documentation**: Markdown, text files, wikis
- **Configuration**: YAML, JSON, TOML, INI files

[We need to make selection of LSP <-> Language, criteria: fast, multi-platform. The list of supported LSPs will have to be included into the configuration.]
[Question: to which extend would linters be useful?]

**Custom Extensions**: Users can extend (not replace) hardcoded patterns

- `custom_include_patterns`: Additional inclusion patterns
- `custom_exclude_patterns`: Additional exclusion patterns
- Performance-first: Exclusion patterns processed before inclusion

**Research-Backed Language Coverage** (based on comprehensive A-G language analysis):

**Languages with Premier LSP Support**: Go, Gleam, GDScript, GLSL, Groovy (limited), GNU Octave (via mlang)
**Languages with Tree-sitter Fallback**: Assembly variants, AWK, configuration languages, legacy languages
**Multi-language Project Support**: Polyglot detection with primary language inference
**Environment Detection**: Python virtual environments, Rust cargo workspaces, Node.js projects, Java build tools

**Systematic Research Coverage**: Completed analysis of 150+ languages with verified LSP availability and Tree-sitter grammar support for comprehensive language ecosystem coverage.

### 12.5 Configuration Implementation Requirements (Critical Feedback)

**Language Processing Integration**: LSP vs Tree-sitter mutually exclusive strategy with graceful degradation
**Environment Detection**: Automatic detection of Python virtual environments, Rust cargo workspaces, Node.js projects
**Project Signature Priority**: Language detection through build configuration files takes precedence over file extensions
**XDG Data Home**: Tree-sitter grammar storage follows OS-standard directory conventions

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
- Tracks files processed with Tree-sitter vs LSP vs text-only for coverage analysis
- Enables background LSP integration when servers become available

**Pattern Priority**: Exclusion-first processing for performance optimization

### 12.6 Rust Dependencies for Language Processing

**Core Language Processing Crates** (based on research findings):
```toml
# Tree-sitter integration
tree-sitter = "0.20"
tree-sitter-highlight = "0.20"
tree-sitter-tags = "0.20"

# LSP client implementation
lsp-types = "0.94"
lsp-server = "0.7"
tower-lsp = "0.20"

# High-priority language grammars (Premier LSP support languages)
tree-sitter-c = "0.20"
tree-sitter-cpp = "0.20"
tree-sitter-rust = "0.20"
tree-sitter-python = "0.20"
tree-sitter-javascript = "0.20"
tree-sitter-typescript = "0.20"
tree-sitter-go = "0.20"
tree-sitter-java = "0.20"
tree-sitter-bash = "0.20"

# Fallback grammars (languages without viable LSP)
tree-sitter-awk = "0.20"
tree-sitter-make = "0.20"
tree-sitter-cmake = "0.20"
tree-sitter-dockerfile = "0.20"
tree-sitter-nginx = "0.20"
tree-sitter-toml = "0.20"
tree-sitter-yaml = "0.20"
tree-sitter-json = "0.20"
tree-sitter-html = "0.20"
tree-sitter-css = "0.20"

# Performance optimization
rayon = "1.7"        # Parallel processing for language analysis
dashmap = "5.4"      # Concurrent HashMap for language state caching
bytes = "1.4"        # Efficient buffer management for LSP communication
```

**Language Processing Pipeline Crates**:
```toml
# AST processing and symbol extraction
syn = "2.0"          # Rust-specific semantic analysis
tokio = "1.0"        # Async runtime for LSP communication
serde = "1.0"        # Serialization for LSP messages and Tree-sitter output

# Error handling for graceful degradation
anyhow = "1.0"       # Error context for LSP failures
thiserror = "1.0"    # Custom error types for processing pipeline
```

## 13. Implementation Roadmap

### Phase 1: Tool Consolidation (Critical Gap)

- **Current State**: 30+ individual tools (30% alignment)
- **Target**: 4 consolidated tools (qdrant_store, qdrant_find, qdrant_manage, qdrant_read)
- **Priority**: High - Major architectural misalignment

### Phase 2: Language Processing Architecture Implementation

- **Current State**: Basic LSP integration research completed (A-G languages analyzed)
- **Target**: LSP vs Tree-sitter mutually exclusive strategy with comprehensive language support
- **Components**:
  - LSP server integration for 40+ premier languages
  - Tree-sitter fallback for 50+ additional languages
  - Environment detection (Python venvs, Rust workspaces, etc.)
  - XDG-compliant grammar storage
- **Priority**: High - Foundation for semantic code understanding

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

---

## Appendix A: Complete Programming Language Support

### A.1 All Programming Languages with LSP and Linter Support

**Modern Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Python** | .py, .pyw, .pyi | ruff-lsp | pylsp, pyright | ruff, black, isort | Modern |
| **Rust** | .rs | rust-analyzer | - | clippy (built-in) | Modern |
| **JavaScript** | .js, .mjs, .cjs | typescript-language-server | eslint-lsp | eslint, prettier | Modern |
| **TypeScript** | .ts, .tsx, .mts, .cts | typescript-language-server | - | eslint, prettier | Modern |
| **Go** | .go | gopls | - | golint, gofmt (built-in) | Modern |
| **Swift** | .swift | sourcekit-lsp | - | swiftlint | Modern |
| **Kotlin** | .kt, .kts | kotlin-language-server | - | ktlint, detekt | Modern |
| **Dart** | .dart | dartls | - | dart analyze (built-in) | Modern |
| **Zig** | .zig | zls | - | zig fmt (built-in) | Emerging |
| **Crystal** | .cr | crystalline | - | ameba | Modern |
| **Nim** | .nim, .nims | nimlsp | - | - | Modern |
| **V** | .v | vls | - | v fmt (built-in) | Emerging |
| **Carbon** | .carbon | - | - | - | Experimental |

**Enterprise/JVM Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Java** | .java | eclipse.jdt.ls | - | checkstyle, spotbugs | Enterprise |
| **Scala** | .scala, .sc | metals | - | scalafmt, scalafix | Enterprise |
| **Clojure** | .clj, .cljs, .cljc | clojure-lsp | - | clj-kondo | Enterprise |
| **Groovy** | .groovy, .gvy | groovy-language-server | - | codenarc | Enterprise |

**Functional Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Haskell** | .hs, .lhs | haskell-language-server | - | hlint | Functional |
| **OCaml** | .ml, .mli | ocaml-lsp | - | ocp-indent | Functional |
| **F#** | .fs, .fsi, .fsx | fsautocomplete | - | fantomas | Functional |
| **Erlang** | .erl, .hrl | erlang_ls | - | elvis | Functional |
| **Elixir** | .ex, .exs | elixir-ls | - | credo | Functional |
| **Elm** | .elm | elm-language-server | - | elm-format | Functional |
| **PureScript** | .purs | purescript-language-server | - | - | Functional |
| **Idris** | .idr | idris2-lsp | - | - | Functional |
| **Agda** | .agda | agda-language-server | - | - | Functional |
| **Lean** | .lean | lean-language-server | - | - | Functional |

**Systems Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **C** | .c, .h | clangd | ccls | clang-tidy, cppcheck | Systems |
| **C++** | .cc, .cpp, .cxx, .hpp | clangd | ccls | clang-tidy, cppcheck | Systems |
| **C#** | .cs, .csx | omnisharp-roslyn | - | .editorconfig | Systems |
| **Objective-C** | .m, .mm | clangd | - | clang-tidy | Systems |
| **Assembly** | .s, .asm | asm-lsp | - | - | Systems |
| **D** | .d | serve-d | - | dscanner | Systems |

**Scripting Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Ruby** | .rb, .rbw | solargraph | - | rubocop | Scripting |
| **PHP** | .php, .phtml | phpactor | intelephense | phpstan, psalm | Scripting |
| **Perl** | .pl, .pm | perl-languageserver | - | perlcritic | Scripting |
| **Lua** | .lua | lua-language-server | - | luacheck | Scripting |
| **Shell** | .sh, .bash, .zsh | bash-language-server | - | shellcheck | Scripting |
| **PowerShell** | .ps1, .psm1, .psd1 | powershell-es | - | PSScriptAnalyzer | Scripting |
| **Tcl** | .tcl | - | - | nagelfar | Scripting |
| **Awk** | .awk | - | - | - | Scripting |
| **Sed** | .sed | - | - | - | Scripting |

**Legacy/Historical Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **COBOL** | .cob, .cbl | - | - | - | Legacy |
| **Fortran** | .f, .f90, .f95 | fortls | - | - | Legacy |
| **Pascal** | .pas, .pp | pasls | - | - | Legacy |
| **Modula-2** | .mod, .def | - | - | - | Legacy |
| **Ada** | .adb, .ads | ada_language_server | - | - | Legacy |
| **ALGOL** | .alg | - | - | - | Historical |
| **Forth** | .fth, .forth | - | - | - | Legacy |
| **Lisp** | .lisp, .lsp | - | - | - | Historical |
| **Scheme** | .scm, .ss | - | - | - | Historical |
| **Smalltalk** | .st | - | - | - | Historical |
| **Prolog** | .pl, .pro | - | - | - | Logic |
| **APL** | .apl | - | - | - | Mathematical |
| **J** | .ijs | - | - | - | Mathematical |
| **K** | .k | - | - | - | Mathematical |

**Specialized Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **R** | .r, .R | languageserver | - | lintr | Statistical |
| **MATLAB** | .m | matlab-language-server | - | mlint | Mathematical |
| **Mathematica** | .m, .nb | - | - | - | Mathematical |
| **Julia** | .jl | julia-language-server | - | - | Scientific |
| **SAS** | .sas | - | - | - | Statistical |
| **SPSS** | .sps | - | - | - | Statistical |
| **Stata** | .do, .ado | - | - | - | Statistical |
| **SQL** | .sql | sqls | sqlls | sqlfluff | Database |
| **PL/SQL** | .pls, .plsql | - | - | - | Database |
| **T-SQL** | .sql | mssql-language-server | - | - | Database |

**Esoteric/Educational Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Brainfuck** | .bf | - | - | - | Esoteric |
| **Whitespace** | .ws | - | - | - | Esoteric |
| **Malbolge** | .mal | - | - | - | Esoteric |
| **Befunge** | .bf | - | - | - | Esoteric |
| **INTERCAL** | .i | - | - | - | Esoteric |
| **Ook!** | .ook | - | - | - | Esoteric |
| **Piet** | .piet | - | - | - | Esoteric |
| **Chef** | .chef | - | - | - | Esoteric |

**Domain-Specific Languages**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **VHDL** | .vhd, .vhdl | vhdl_ls | - | - | Hardware |
| **Verilog** | .v, .vh | verible-verilog-ls | - | verilator | Hardware |
| **SystemVerilog** | .sv, .svh | verible-verilog-ls | - | - | Hardware |
| **Chisel** | .scala (chisel) | metals | - | - | Hardware |
| **PostScript** | .ps | - | - | - | Graphics |
| **TeX** | .tex | texlab | - | chktex | Typesetting |
| **LaTeX** | .tex, .latex | texlab | - | chktex | Typesetting |
| **Gnuplot** | .gp, .gnuplot | - | - | - | Plotting |

**Visual Programming**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Scratch** | .sb3 | - | - | - | Educational |
| **LabVIEW** | .vi | - | - | - | Visual |
| **Simulink** | .slx | - | - | - | Visual |
| **Blockly** | .xml | - | - | - | Educational |

**Microsoft Ecosystem**:
| Language | File Extensions | Primary LSP | Alternative LSPs | Linters | Era/Category |
|----------|----------------|-------------|------------------|---------|--------------|
| **Visual Basic** | .vb | - | - | - | Microsoft |
| **VBA** | .vba | - | - | - | Microsoft |
| **VB.NET** | .vb | omnisharp-roslyn | - | - | Microsoft |
| **F#** | .fs, .fsi, .fsx | fsautocomplete | - | fantomas | Microsoft |
| **PowerShell** | .ps1, .psm1, .psd1 | powershell-es | - | PSScriptAnalyzer | Microsoft |

## Appendix B: File Pattern Specifications

### B.1 Inclusion Patterns by Category

**Source Code Patterns** (250+ patterns):

```
# Programming Languages
*.py, *.pyw, *.pyi          # Python
*.rs                        # Rust
*.js, *.mjs, *.cjs         # JavaScript
*.ts, *.tsx, *.mts, *.cts  # TypeScript
*.java                      # Java
*.c, *.h, *.cc, *.cpp, *.cxx, *.hpp  # C/C++
*.cs, *.csx                # C#
*.go                        # Go
*.php, *.phtml             # PHP
*.rb, *.rbw                # Ruby
*.swift                     # Swift
*.kt, *.kts                # Kotlin
*.scala, *.sc              # Scala
# [Continue for all 60+ languages]
```

**Configuration Patterns**:

```
*.yml, *.yaml              # YAML configuration
*.json, *.jsonc            # JSON configuration
*.toml                     # TOML configuration
*.ini, *.cfg               # INI-style config
*.conf, *.config           # Generic config
*.env, *.env.*             # Environment files
```

**Infrastructure Patterns**:

```
Dockerfile, *.dockerfile   # Docker
*.tf, *.tfvars            # Terraform
*.k8s.yaml, *.k8s.yml     # Kubernetes
docker-compose*.yml        # Docker Compose
*.helm.yaml               # Helm charts
```

### B.2 Exclusion Patterns by Category

**Build Artifacts**:

```
build/, dist/, target/     # Build directories
*.o, *.obj, *.exe         # Compiled objects
*.so, *.dll, *.dylib      # Shared libraries
node_modules/             # Node.js dependencies
__pycache__/              # Python cache
.gradle/, .maven/         # Java build
```

**Version Control**:

```
.git/, .svn/, .hg/        # VCS directories
*.orig, *.rej             # Merge artifacts
.gitignore, .svnignore    # VCS ignore files
```

**IDE and Editor Files**:

```
.vscode/, .idea/          # IDE settings
*.swp, *.swo, *~          # Editor temp files
.DS_Store                 # macOS system files
Thumbs.db                 # Windows thumbnails
```

## Appendix C: State Database Schema

### C.1 Generic Schema Tables

**Operations Table**:

```sql
CREATE TABLE operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,              -- 'ingestion', 'lsp_check', 'cleanup'
    status TEXT NOT NULL,            -- 'pending', 'running', 'completed', 'failed'
    priority INTEGER DEFAULT 5,      -- 1 (highest) to 10 (lowest)
    metadata JSON,                   -- Operation-specific data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Resources Table**:

```sql
CREATE TABLE resources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,              -- 'file', 'folder', 'lsp', 'collection'
    path TEXT,                       -- File/folder path or resource identifier
    status TEXT NOT NULL,            -- 'available', 'missing', 'error', 'processing'
    metadata JSON,                   -- Resource-specific data
    last_checked TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Events Table**:

```sql
CREATE TABLE events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,        -- 'file_change', 'lsp_error', 'ingestion_complete'
    source TEXT,                     -- Component that generated event
    payload JSON,                    -- Event-specific data
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### C.2 State Database Use Cases

**LSP Health Tracking**:

- Track which LSPs are expected but unavailable
- Monitor LSP performance and error rates
- Schedule re-processing when LSPs become available

**File Processing State**:

- Track ingestion progress for large directories
- Maintain checksums for change detection
- Store parsing failure information for user queries

**Collection Management**:

- Track collection status (clean/dirty)
- Monitor tenant mappings and project associations
- Maintain collection metadata and statistics

---

## Appendix D: Research References

### D.1 Language Research Foundation

**Systematic Language Analysis** - Comprehensive research spanning A-Z programming languages providing foundation for LSP integration and Tree-sitter fallback strategies:

**Core Language Research Files** (research/languages/):
- **A-Z Language Research**: `A_LANGUAGES_LSP_RESEARCH.md` through `Z_LANGUAGES_LSP_RESEARCH.md` - Systematic analysis of 200+ programming languages covering LSP server availability, installation methods, feature completeness, and development activity
- **Master Summary**: `MASTER_LSP_RESEARCH_SUMMARY.md` - Synthesized findings with tier classifications and strategic implementation recommendations
- **Progress Tracker**: `LSP_RESEARCH_PROGRESS.md` - Research methodology and completion status tracking
- **Completion Reports**: `20250118-1452_G_LANGUAGES_RESEARCH_*.md` - Final research validation and summary documentation

**Tree-sitter Ecosystem Research**:
- **`20250919-1034_TREE_SITTER_ECOSYSTEM_RESEARCH.md`** - Comprehensive analysis of Tree-sitter grammar ecosystem including installation methods, grammar availability, and fallback strategies for languages without LSP support
- **`20250919-0607_LSP_TREESITTER_SIMULTANEOUS_USAGE_RESEARCH.md`** - Research on mutually exclusive LSP vs Tree-sitter strategy, performance implications, and graceful degradation approaches

**Language Detection and Processing**:
- **`20250919-1103_COMPREHENSIVE_LANGUAGE_DETECTION_RESEARCH.md`** - Project signature patterns, file extension mapping, environment detection (Python virtual environments, Rust cargo workspaces, Node.js projects), and polyglot project handling

**Research Summaries**:
- **`LSP_LINTER_RESEARCH_SUMMARY.md`** - Comprehensive linter ecosystem analysis complementing LSP research
- **Core language files in project root**: `A_LANGUAGES_LSP_RESEARCH.md` through `L_LANGUAGES_LSP_RESEARCH.md` - Additional research files providing language-specific implementation details

### D.2 Architecture Research

**First Principles Framework**: `FIRST-PRINCIPLES.md` - Foundational architectural principles establishing memory-driven behavioral persistence, project-scoped semantic context, heterogeneous content unification, and four-component separation as core design philosophy

**Implementation Guidance**: Research findings directly inform:
- **Section 5**: Language Processing Architecture (LSP vs Tree-sitter mutually exclusive strategy)
- **Section 12**: Configuration Architecture (language-specific LSP server selection and Tree-sitter grammar storage)
- **Appendix A**: Complete Programming Language Support (research-backed language coverage)

### D.3 Research Methodology

**Systematic Approach**: Alphabet-based analysis ensuring comprehensive coverage of programming language ecosystem without gaps or omissions

**Evaluation Criteria**:
- Speed and responsiveness of LSP servers
- Cross-platform compatibility (macOS, Linux, Windows)
- Feature completeness (semantic analysis, refactoring, debugging)
- Active development and community support
- LSP protocol compliance and standards adherence

**Quality Tiers**:
- **Tier 1**: Exceptional LSP support (Zig, Rust, TypeScript, Go, Python, C#, Java)
- **Tier 2**: Good LSP support (PHP, Ruby, Swift, Kotlin, Scala, Dart, Lua, Bash, SQL)
- **Tier 3**: Basic LSP support (Haskell, Clojure, Erlang/Elixir, F#, OCaml)
- **Tier 4**: Minimal/No LSP support (legacy and domain-specific languages)

---

## Appendix E: Implementation Task Mapping

### E.1 Current Task-Master Implementation Status

**Active Development Tasks**: 51 pending tasks focusing on core architecture implementation with 75.7% overall completion rate (181 of 239 tasks completed)

**Priority Distribution**:
- **High Priority**: 4 tasks (LSP integration, error handling, testing, deployment validation)
- **Medium Priority**: 37 tasks (configuration, multi-tenancy, documentation, performance)
- **Low Priority**: 10 tasks (monitoring, user experience, cleanup)

### E.2 Task Alignment with PRD Sections

**Section 2: Collection Architecture → Tasks 97, 226, 230, 235, 236**
- **Task 97**: Default collection configuration for folder ingestion (High Priority)
- **Task 226**: Collection type-based configuration implementation
- **Task 230**: Collection naming collision prevention system
- **Task 235**: Auto-ingestion with new pattern system enhancement
- **Task 236**: Memory collection management with conversational updates

**Section 5: Language Processing → Tasks 130-140, 142**
- **Task 132**: Multi-Language LSP Server Support Matrix (Medium Priority)
- **Task 135**: Error Handling and Graceful Degradation System (High Priority)
- **Task 136**: LSP Integration Testing Suite
- **Task 140**: Production Deployment and Scaling Validation (High Priority)
- **Task 142**: Fix GitHub Issue #14 - Advanced search type conversion errors

**Section 12: Configuration System → Tasks 228-229, 237, 180, 184**
- **Task 228**: OS-Standard Directory System implementation
- **Task 229**: Configuration file discovery and hierarchy
- **Task 237**: Configuration validation and health checks
- **Task 180**: Complete TOML configuration schema
- **Task 184**: Configuration migration and backward compatibility

**Section 13: Implementation Roadmap → Tasks 109, 111, 151-170**
- **Phase 1 Tool Consolidation**: Tasks 174-185 (Memory system integration)
- **Phase 2 Language Processing**: Tasks 130-140 (LSP architecture implementation)
- **Phase 3 Testing Infrastructure**: Tasks 151-170 (Comprehensive testing campaign)

### E.3 Testing and Validation Framework → Tasks 151-170

**Large-Scale Performance Testing**:
- **Task 155**: QMK firmware dataset benchmarking (22,066+ files)
- **Task 160**: Performance monitoring and resource management
- **Task 169**: Production readiness assessment with quantified evidence
- **Task 170**: Comprehensive testing campaign execution

**Multi-Component Integration**:
- **Task 161**: Multi-component synchronization validation (High Priority)
- **Task 162**: Realistic development workflow simulation (High Priority)
- **Task 163**: Edge case and error handling validation

### E.4 Service Management and Deployment → Tasks 189-193

**Daemon Lifecycle Management**:
- **Task 190**: Health check endpoints and monitoring
- **Task 191**: Signal handling and graceful shutdown
- **Task 192**: Service configuration validation
- **Task 193**: Comprehensive integration tests

### E.5 Advanced Features → Tasks 103-104, 179, 183

**Multi-Instance Support**:
- **Task 103**: Isolated daemon processes for different projects
- **Task 179**: Hook-based context persistence system
- **Task 183**: Context generation and re-injection logic

**Multimodal Processing**:
- **Task 104**: Text-image bidirectional linking with separate embeddings

### E.6 Implementation Dependencies and Critical Path

**Foundation Dependencies** (Must Complete First):
1. **Collection Architecture** (Tasks 226, 230) → Enables multi-tenancy
2. **Configuration System** (Tasks 228, 229) → Enables OS-standard compliance
3. **LSP Integration Core** (Tasks 132, 135) → Enables language processing

**Critical Path for Production Readiness**:
1. **Error Handling** (Task 135) → **Testing Suite** (Task 136) → **Production Validation** (Task 140)
2. **Performance Testing** (Task 155) → **Resource Management** (Task 160) → **Readiness Assessment** (Task 169)

### E.7 Research-to-Implementation Mapping

**Research Foundation → Active Tasks**:
- **A-Z Language Research** → **Task 132** (Multi-Language LSP Support Matrix)
- **Tree-sitter Ecosystem Research** → **Task 135** (Graceful Degradation System)
- **Language Detection Research** → **Task 132** (Environment detection implementation)
- **First Principles Architecture** → **Tasks 174-185** (Memory system integration)

**Evidence-Based Implementation**: All language processing tasks (130-140) directly implement findings from systematic A-Z language research, ensuring comprehensive coverage of 200+ programming languages with verified LSP availability and Tree-sitter fallback strategies.

**Quality Assurance Alignment**: Testing tasks (151-170) validate research predictions about performance characteristics, error handling, and cross-platform compatibility derived from LSP ecosystem analysis.

### E.8 PRD Completeness Validation

**Complete Alignment**: Analysis confirms no task-master tasks exist without corresponding PRD sections - all 51 pending tasks map directly to defined architecture components

**Implementation Gaps Identified**:
- **Memory System Integration** (Tasks 174-185): Well-covered in Section 3 and First Principles
- **Tool Consolidation** (Current 30+ tools → 4 consolidated): Addressed in Section 4
- **Multi-Component Synchronization**: Covered in Section 1 (Two-Process Architecture)

**Missing PRD Coverage**: None - comprehensive task coverage validates PRD completeness

### E.9 Implementation Status Summary

**Immediate Priority** (High Priority Tasks):
1. **Task 97**: Default collection configuration (Collection Architecture implementation)
2. **Task 135**: Error handling and graceful degradation (Language Processing core)
3. **Task 140**: Production deployment validation (Quality assurance)
4. **Tasks 161-162**: Multi-component integration testing (System validation)

**Research Foundation Complete**: All language processing implementation tasks have complete research backing through systematic A-Z language analysis and Tree-sitter ecosystem research

**Production Readiness Path**: Clear dependency chain from foundation tasks through testing validation to production deployment, with quantified success criteria and evidence-based quality gates
