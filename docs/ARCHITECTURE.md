# workspace-qdrant-mcp Architecture

Visual architecture documentation for the workspace-qdrant-mcp system, showing component interactions, data flow, and system design.

## Table of Contents

- [System Overview](#system-overview)
- [Component Architecture](#component-architecture)
- [MCP Server Components](#mcp-server-components)
- [Rust Daemon Architecture](#rust-daemon-architecture)
- [Hybrid Search Flow](#hybrid-search-flow)
- [Collection Structure](#collection-structure)
- [SQLite State Management](#sqlite-state-management)
- [Write Path Architecture](#write-path-architecture)
- [Data Flow Patterns](#data-flow-patterns)

## System Overview

workspace-qdrant-mcp implements a four-component architecture for high-performance semantic search and document management integrated with Claude Desktop and Claude Code.

```mermaid
graph TB
    subgraph "User Interfaces"
        Claude[Claude Desktop/Code]
        CLI[CLI - wqm]
    end

    subgraph "Python Components"
        MCP[MCP Server<br/>FastMCP<br/>4 Tools]
        CI[Context Injector<br/>Rule Management<br/>Hook System]
    end

    subgraph "Rust Component"
        Daemon[Rust Daemon<br/>memexd<br/>Processing Engine]
    end

    subgraph "Data Stores"
        SQLite[(SQLite DB<br/>State & Config)]
        Qdrant[(Qdrant<br/>Vector DB)]
    end

    Claude -->|MCP Protocol<br/>stdio| MCP
    CLI -->|Commands<br/>Direct Access| SQLite
    CLI -.->|Signals| Daemon

    MCP <-->|gRPC<br/><50ms| Daemon
    CI -->|Hooks<br/>Events| MCP
    CI -->|Read Rules| Qdrant

    Daemon <-->|Read/Write<br/>State| SQLite
    Daemon <-->|Vector Ops<br/><100ms| Qdrant

    style MCP fill:#e1f5ff
    style Daemon fill:#ffe1e1
    style CI fill:#e1ffe1
    style CLI fill:#fff5e1
```

**Key Components:**
- **MCP Server**: Intelligent interface layer for Claude integration
- **Rust Daemon**: Heavy processing engine for document ingestion and search
- **CLI Utility**: Administrative control and system management
- **Context Injector**: LLM rule injection and behavioral context
- **SQLite**: Shared state and configuration database
- **Qdrant**: Vector database for semantic search

## Component Architecture

Detailed view of the four-component architecture showing responsibilities and communication patterns.

```mermaid
graph TB
    subgraph "Component 1: Python MCP Server"
        MCP_Tools[4 MCP Tools<br/>store, search,<br/>manage, retrieve]
        MCP_gRPC[gRPC Client]
        MCP_Session[Session Manager]
        MCP_Query[Query Optimizer]

        MCP_Tools --> MCP_Query
        MCP_Query --> MCP_gRPC
        MCP_Tools --> MCP_Session
    end

    subgraph "Component 2: Rust Daemon"
        D_gRPC[gRPC Server]
        D_Watch[File Watcher]
        D_LSP[LSP Integration]
        D_Process[Document Processor]
        D_Embed[Embedding Generator]
        D_Queue[Processing Queue]

        D_gRPC --> D_Queue
        D_Watch --> D_Queue
        D_Queue --> D_Process
        D_Process --> D_LSP
        D_Process --> D_Embed
    end

    subgraph "Component 3: CLI Utility"
        CLI_Commands[CLI Commands<br/>wqm]
        CLI_Admin[Admin Operations]
        CLI_Config[Config Manager]
        CLI_Monitor[Health Monitor]

        CLI_Commands --> CLI_Admin
        CLI_Commands --> CLI_Config
        CLI_Commands --> CLI_Monitor
    end

    subgraph "Component 4: Context Injector"
        CI_Hooks[Hook System]
        CI_Rules[Rule Fetcher]
        CI_Format[Rule Formatter]
        CI_Inject[Context Injector]

        CI_Hooks --> CI_Rules
        CI_Rules --> CI_Format
        CI_Format --> CI_Inject
    end

    MCP_gRPC <-->|gRPC Protocol| D_gRPC
    CLI_Admin -.->|Signals| D_Watch
    CI_Inject -->|Inject Rules| MCP_Session

    D_Embed -->|Vectors| Qdrant[(Qdrant)]
    D_Process -->|State| SQLite[(SQLite)]
    CLI_Config -->|Direct Access| SQLite
    CI_Rules -->|Read| Qdrant

    style MCP_Tools fill:#e1f5ff
    style D_gRPC fill:#ffe1e1
    style CLI_Commands fill:#fff5e1
    style CI_Hooks fill:#e1ffe1
```

**Communication Patterns:**
- **gRPC**: MCP Server ↔ Rust Daemon (operational communication)
- **Signals**: CLI ↔ Daemon (lifecycle management)
- **SQLite**: Shared state with component-specific access
- **Hooks**: Event-driven Context Injector activation

## MCP Server Components

Detailed architecture of the FastMCP server implementation showing tool routing and processing.

```mermaid
graph TB
    subgraph "MCP Protocol Layer"
        STDIO[stdio Transport<br/>Claude Desktop]
        HTTP[HTTP Transport<br/>Testing/Debug]
    end

    subgraph "FastMCP Application"
        Router[Tool Router<br/>@app.tool]

        subgraph "4 MCP Tools"
            Store[store<br/>Content Storage]
            Search[search<br/>Hybrid Search]
            Manage[manage<br/>Collection Mgmt]
            Retrieve[retrieve<br/>Direct Access]
        end
    end

    subgraph "Core Services"
        ProjDet[Project Detection<br/>calculate_tenant_id]
        Branch[Git Branch<br/>get_current_branch]
        Embed[Embedding Model<br/>FastEmbed]
        Client[Daemon Client<br/>gRPC]
    end

    subgraph "Data Processing"
        Metadata[Metadata Builder<br/>file_type, branch]
        Filters[Filter Builder<br/>branch, file_type]
        Response[Response Formatter<br/>JSON]
    end

    STDIO --> Router
    HTTP --> Router

    Router --> Store
    Router --> Search
    Router --> Manage
    Router --> Retrieve

    Store --> ProjDet
    Store --> Metadata
    Store --> Client

    Search --> Branch
    Search --> Filters
    Search --> Client

    Manage --> Client
    Retrieve --> Filters
    Retrieve --> Client

    Client -->|gRPC| Daemon[Rust Daemon]

    Store --> Response
    Search --> Response
    Manage --> Response
    Retrieve --> Response

    Response -->|MCP Protocol| STDIO
    Response -->|HTTP| HTTP

    style Router fill:#e1f5ff
    style Store fill:#d4edff
    style Search fill:#d4edff
    style Manage fill:#d4edff
    style Retrieve fill:#d4edff
```

**Tool Capabilities:**
- **store**: Automatic collection routing, metadata enrichment, daemon-based writes
- **search**: Branch filtering, file type filtering, hybrid search modes
- **manage**: Collection lifecycle, health monitoring, system status
- **retrieve**: Direct ID lookup, metadata filtering, branch-aware queries

## Rust Daemon Architecture

High-performance document processing engine with concurrent operation handling.

```mermaid
graph TB
    subgraph "Daemon Entry Points"
        gRPC_Server[gRPC Server<br/>Port 50051]
        FileWatcher[File Watcher<br/>notify-rs]
        SignalHandler[Signal Handler<br/>SIGHUP/TERM]
    end

    subgraph "Processing Core"
        Queue[Processing Queue<br/>Bounded Channel]
        Workers[Worker Pool<br/>Tokio Tasks]

        subgraph "Document Processing"
            Parser[File Parser<br/>Multi-format]
            LSP[LSP Client<br/>Code Analysis]
            Chunker[Text Chunker<br/>800/120]
            Embedder[Embedding Gen<br/>FastEmbed]
        end
    end

    subgraph "State Management"
        SQLite_Mgr[SQLite Manager<br/>WAL Mode]
        WatchConfig[Watch Config<br/>watch_folders]
        StateCache[State Cache<br/>In-memory]
    end

    subgraph "Vector Operations"
        QdrantClient[Qdrant Client<br/>HTTP/gRPC]
        CollMgr[Collection Manager<br/>Create/Delete]
        VectorWriter[Vector Writer<br/>Batch Upsert]
    end

    gRPC_Server --> Queue
    FileWatcher --> Queue
    SignalHandler --> SQLite_Mgr

    Queue --> Workers

    Workers --> Parser
    Parser --> LSP
    Parser --> Chunker
    Chunker --> Embedder

    Workers --> SQLite_Mgr
    SQLite_Mgr <--> WatchConfig
    SQLite_Mgr <--> StateCache

    Embedder --> VectorWriter
    VectorWriter --> QdrantClient
    gRPC_Server --> CollMgr
    CollMgr --> QdrantClient

    FileWatcher <-.->|Poll Config| WatchConfig

    style Queue fill:#ffe1e1
    style Workers fill:#ffcccc
    style SQLite_Mgr fill:#fff5e1
    style QdrantClient fill:#e1ffe1
```

**Performance Characteristics:**
- **Throughput**: 1000+ documents/minute sustained
- **Memory**: <500MB sustained operation
- **Latency**: <50ms gRPC response time
- **Concurrency**: Multi-threaded worker pool with bounded queues

## Hybrid Search Flow

End-to-end search flow showing semantic and keyword search combination using RRF.

```mermaid
sequenceDiagram
    participant Claude as Claude Code
    participant MCP as MCP Server
    participant Daemon as Rust Daemon
    participant Qdrant as Qdrant DB

    Note over Claude,Qdrant: Hybrid Search Flow (semantic + keyword)

    Claude->>MCP: search(query="auth logic",<br/>mode="hybrid", file_type="code")

    activate MCP
    Note over MCP: 1. Project Detection
    MCP->>MCP: get_project_collection()<br/>→ _a1b2c3d4e5f6

    Note over MCP: 2. Filter Building
    MCP->>MCP: build_metadata_filters()<br/>branch="main"<br/>file_type="code"

    Note over MCP: 3. Embedding Generation
    MCP->>MCP: generate_embeddings()<br/>→ [0.123, 0.456, ...]

    MCP->>Daemon: gRPC: SearchRequest<br/>collection="_a1b2c3d4e5f6"<br/>query_vector=[...]<br/>filters={branch, file_type}
    deactivate MCP

    activate Daemon
    Note over Daemon: 4. Parallel Search

    par Semantic Search
        Daemon->>Qdrant: vector_search()<br/>query_vector<br/>filters<br/>limit=10
        Qdrant-->>Daemon: semantic_results[]<br/>scored by similarity
    and Keyword Search
        Daemon->>Qdrant: scroll()<br/>keyword="auth logic"<br/>filters<br/>limit=20
        Qdrant-->>Daemon: keyword_results[]<br/>scored by frequency
    end

    Note over Daemon: 5. RRF Fusion
    Daemon->>Daemon: reciprocal_rank_fusion()<br/>combine results<br/>deduplicate

    Daemon-->>MCP: SearchResponse<br/>results[]<br/>search_time_ms
    deactivate Daemon

    activate MCP
    Note over MCP: 6. Response Formatting
    MCP->>MCP: format_results()<br/>add metadata<br/>filter info

    MCP-->>Claude: SearchResults<br/>3 documents<br/>45.23ms
    deactivate MCP

    Note over Claude,Qdrant: Result: High precision (94.2% semantic, 100% symbol)
```

**Search Modes:**
- **hybrid**: Combines semantic and keyword with RRF (default)
- **semantic**: Pure vector similarity search
- **exact**: Keyword and symbol exact matching
- **keyword**: Simple keyword matching

**Performance Targets:**
- Total latency: <150ms end-to-end
- MCP processing: <50ms
- gRPC communication: <50ms
- Qdrant search: <50ms

## Collection Structure

Multi-tenant collection architecture with metadata-based differentiation.

```mermaid
graph TB
    subgraph "Collection Types"
        subgraph "PROJECT Collections"
            P1[_a1b2c3d4e5f6<br/>Project Alpha]
            P2[_b2c3d4e5f6a1<br/>Project Beta]
            P3[_c3d4e5f6a1b2<br/>Project Gamma]
        end

        subgraph "USER Collections"
            U1[myapp-notes<br/>User Notes]
            U2[research-refs<br/>References]
        end

        subgraph "LIBRARY Collections"
            L1[_fastapi<br/>FastAPI Docs]
            L2[_react<br/>React Docs]
        end

        subgraph "MEMORY Collections"
            M1[_memory<br/>System Rules]
            M2[_agent_memory<br/>Agent Context]
        end
    end

    subgraph "Metadata Structure"
        direction LR
        Meta[Document Metadata]

        Meta --> FT[file_type:<br/>code, test, docs,<br/>config, data,<br/>build, other]
        Meta --> BR[branch:<br/>main, develop,<br/>feature/*, etc.]
        Meta --> PID[project_id:<br/>12-char hex hash]
        Meta --> FP[file_path:<br/>src/auth.py]
        Meta --> SYM[symbols:<br/>functions, classes]
    end

    P1 -.->|contains| Meta
    U1 -.->|enriched with| PID

    style P1 fill:#e1f5ff
    style P2 fill:#e1f5ff
    style P3 fill:#e1f5ff
    style U1 fill:#fff5e1
    style U2 fill:#fff5e1
    style L1 fill:#e1ffe1
    style L2 fill:#e1ffe1
    style M1 fill:#ffe1e1
    style M2 fill:#ffe1e1
```

**Naming Conventions:**
- **PROJECT**: `_{project_id}` - Single collection per project (12-char hex hash)
- **USER**: `{basename}-{type}` - User-created collections
- **LIBRARY**: `_{library_name}` - External library documentation
- **MEMORY**: `_memory`, `_agent_memory` - System and agent rules

**Differentiation Strategy:**
- Single collection per project reduces Qdrant collection count
- Metadata-based filtering for branch, file_type, symbols
- Automatic project_id enrichment by daemon
- Branch-scoped queries by default (configurable with `branch="*"`)

## SQLite State Management

SQLite-based watch folder configuration with daemon polling for changes.

```mermaid
graph TB
    subgraph "CLI Operations"
        CLI_Add[wqm watch add<br/>Create Config]
        CLI_List[wqm watch list<br/>View Watches]
        CLI_Remove[wqm watch remove<br/>Delete Config]
        CLI_Update[wqm watch configure<br/>Modify Settings]
    end

    subgraph "SQLite Database"
        WF_Table[(watch_folders Table)]

        subgraph "Table Schema"
            Cols[watch_id PK<br/>path TEXT<br/>collection TEXT<br/>patterns JSON<br/>ignore_patterns JSON<br/>auto_ingest BOOL<br/>recursive BOOL<br/>debounce_seconds REAL<br/>enabled BOOL<br/>created_at TIMESTAMP<br/>updated_at TIMESTAMP]
        end

        WF_Table --> Cols
    end

    subgraph "Daemon Polling"
        Poll[Config Poller<br/>Every 5s]
        Detect[Change Detection<br/>updated_at]
        Reload[Watcher Reload<br/>Apply Changes]
    end

    subgraph "File Watching"
        Watcher[notify-rs Watcher<br/>Platform-optimized]
        Events[File Events<br/>Create/Modify/Delete]
        Queue[Ingestion Queue<br/>Daemon Processing]
    end

    CLI_Add -->|INSERT| WF_Table
    CLI_List -->|SELECT| WF_Table
    CLI_Remove -->|DELETE| WF_Table
    CLI_Update -->|UPDATE| WF_Table

    Poll -->|Query updated_at| WF_Table
    Poll --> Detect
    Detect -->|Changes found| Reload
    Reload --> Watcher

    Watcher --> Events
    Events --> Queue

    Queue -.->|Async Processing| Daemon[Document Processing]

    style WF_Table fill:#fff5e1
    style Poll fill:#ffe1e1
    style Watcher fill:#e1ffe1
```

**Key Features:**
- **Crash-Resistant**: WAL mode with ACID guarantees
- **No gRPC Required**: CLI writes directly to SQLite
- **Daemon Autonomy**: Daemon polls independently
- **Configuration Flexibility**: Per-watch settings (patterns, debounce, recursion)

**Watch Configuration Fields:**
- `patterns`: File patterns to include (e.g., `["*.py", "*.md"]`)
- `ignore_patterns`: File patterns to exclude (e.g., `["*.pyc", "__pycache__/*"]`)
- `debounce_seconds`: Wait time before processing (prevents rapid re-processing)
- `recursive`: Watch subdirectories (configurable depth)
- `enabled`: Active/inactive toggle without deletion

## Write Path Architecture

Daemon-only write architecture ensuring all Qdrant writes flow through the high-performance Rust daemon.

```mermaid
graph TB
    subgraph "Write Sources"
        MCP_Store[MCP: store tool<br/>Content storage]
        MCP_Manage[MCP: manage tool<br/>Collection ops]
        CLI_Add[CLI: wqm add<br/>Document addition]
        Watcher[File Watcher<br/>Auto-ingest]
    end

    subgraph "Write Path Decision"
        Check{Daemon<br/>Available?}
    end

    subgraph "PRIMARY: Daemon Write Path"
        DC[DaemonClient<br/>Python gRPC Client]

        subgraph "Daemon Operations"
            IngestText[ingest_text<br/>Document ingestion]
            CreateColl[create_collection_v2<br/>Collection creation]
            DeleteColl[delete_collection_v2<br/>Collection deletion]
        end

        DC --> IngestText
        DC --> CreateColl
        DC --> DeleteColl
    end

    subgraph "FALLBACK: Direct Write Path"
        FallbackNote[⚠️ Backwards Compatibility<br/>Logged as WARNING]
        DirectClient[Direct Qdrant Client<br/>HTTP API]
    end

    subgraph "EXCEPTION: Memory Collections"
        MemoryNote[✅ Architectural Decision<br/>Meta-level data]
        MemoryDirect[Direct Write Allowed<br/>_memory, _agent_memory]
    end

    subgraph "Qdrant Vector DB"
        Collections[(Collections<br/>_{project_id}<br/>{user}-{type}<br/>_{library})]
    end

    MCP_Store --> Check
    MCP_Manage --> Check
    CLI_Add --> Check
    Watcher --> Check

    Check -->|YES| DC
    Check -->|NO| FallbackNote

    FallbackNote --> DirectClient

    IngestText --> Collections
    CreateColl --> Collections
    DeleteColl --> Collections
    DirectClient -.->|fallback_mode flag| Collections

    MemoryNote --> MemoryDirect
    MemoryDirect --> Collections

    style DC fill:#e1ffe1
    style FallbackNote fill:#ffe1e1
    style MemoryNote fill:#e1f5ff
    style Collections fill:#fff5e1
```

**Write Priority Levels:**
1. **PRIMARY**: `DaemonClient` → Daemon → Qdrant (all writes)
2. **FALLBACK**: Direct `qdrant_client` writes (daemon unavailable, logged with warnings)
3. **EXCEPTION**: `_memory` collections (direct writes allowed, meta-level data)

**Validation (Task 375.6):**
- ✅ 18 comprehensive tests
- ✅ 47 write operations audited
- ✅ Zero violations found
- ✅ Complete compliance documentation

**Benefits:**
- **Consistent Metadata**: All writes enriched with project_id, branch, file_type
- **Performance**: Rust daemon handles heavy operations
- **Reliability**: Single code path reduces bugs
- **Monitoring**: Centralized write operations tracking

## Data Flow Patterns

Common data flow patterns showing typical user operations.

### Document Ingestion Flow

```mermaid
sequenceDiagram
    participant User as User/File System
    participant Watcher as File Watcher
    participant Queue as Processing Queue
    participant Worker as Worker Thread
    participant LSP as LSP Server
    participant Embed as Embedding Gen
    participant SQLite as SQLite DB
    participant Qdrant as Qdrant DB

    User->>Watcher: File Change Event<br/>src/auth.py modified

    activate Watcher
    Watcher->>Watcher: Debounce check<br/>2 seconds
    Watcher->>Queue: Enqueue file<br/>path, collection
    deactivate Watcher

    activate Queue
    Queue->>Worker: Dequeue for processing
    deactivate Queue

    activate Worker
    Worker->>Worker: Read file content
    Worker->>LSP: Request symbols<br/>functions, classes
    LSP-->>Worker: Symbol metadata

    Worker->>Worker: Chunk text<br/>800 chars, 120 overlap

    loop For each chunk
        Worker->>Embed: Generate embeddings<br/>chunk text
        Embed-->>Worker: Vector [384D]
    end

    Worker->>SQLite: Update state<br/>file_hash, processed_at

    Worker->>Qdrant: Batch upsert<br/>vectors + metadata<br/>project_id, branch, file_type
    Qdrant-->>Worker: Success

    Worker->>SQLite: Mark complete<br/>success status
    deactivate Worker
```

### Collection Management Flow

```mermaid
sequenceDiagram
    participant User as User/Claude
    participant MCP as MCP Server
    participant Daemon as Rust Daemon
    participant SQLite as SQLite DB
    participant Qdrant as Qdrant DB

    User->>MCP: manage(action="init_project")

    activate MCP
    MCP->>MCP: calculate_tenant_id()<br/>→ a1b2c3d4e5f6
    MCP->>MCP: build_project_collection_name()<br/>→ _a1b2c3d4e5f6

    MCP->>Daemon: gRPC: CreateCollection<br/>name="_a1b2c3d4e5f6"<br/>vector_size=384<br/>distance="Cosine"
    deactivate MCP

    activate Daemon
    Daemon->>Qdrant: create_collection()<br/>VectorParams(384, Cosine)
    Qdrant-->>Daemon: CollectionInfo

    Daemon->>SQLite: Record collection<br/>created_at, config
    SQLite-->>Daemon: Success

    Daemon-->>MCP: CreateCollectionResponse<br/>success=true
    deactivate Daemon

    activate MCP
    MCP->>MCP: Format response<br/>add metadata

    MCP-->>User: {"success": true,<br/>"collection": "_a1b2c3d4e5f6",<br/>"message": "Created"}
    deactivate MCP
```

### Context Injection Flow

```mermaid
sequenceDiagram
    participant Claude as Claude Code
    participant Hook as Hook System
    participant CI as Context Injector
    participant Qdrant as Qdrant DB
    participant MCP as MCP Server

    Claude->>Hook: Session Init Event<br/>project detected

    activate Hook
    Hook->>CI: Trigger context injection<br/>project_id
    deactivate Hook

    activate CI
    CI->>CI: Detect active tool<br/>LLMToolDetector

    CI->>Qdrant: Query _memory<br/>filter by project_id<br/>rule priority
    Qdrant-->>CI: Rule documents<br/>sorted by priority

    CI->>CI: Format rules<br/>tool-specific format<br/>claude/codex/gemini

    CI->>CI: Check token budget<br/>ensure within limits

    CI->>MCP: Inject formatted rules<br/>via hook interface
    MCP-->>Claude: Updated context<br/>with project rules
    deactivate CI

    Note over Claude: Session ready with<br/>project-specific context
```

---

## Additional Resources

For detailed component specifications and implementation details:

- **[Component Boundaries](./architecture/component-boundaries.md)** - Formal component responsibilities
- **[Interface Specifications](./architecture/interface-specifications.md)** - API and interface details
- **[Data Flow and Isolation](./architecture/data-flow-and-isolation.md)** - System workflows
- **[Implementation Roadmap](./architecture/implementation-roadmap.md)** - Migration path
- **[Write Path Enforcement](./architecture/write-path-enforcement.md)** - Write path validation
- **[LLM Context Injection](./architecture/llm-context-injection.md)** - Context system details

**Reference Implementation:**
- **Server**: `src/python/workspace_qdrant_mcp/server.py` - MCP tools implementation
- **Daemon**: `src/rust/daemon/` - Unified Rust daemon (current), `src/rust/daemon/core/` (archived)
- **CLI**: `src/python/wqm_cli/` - Command-line interface
- **State**: `src/python/common/core/sqlite_state_manager.py` - SQLite management

**Version**: 1.0
**Last Updated**: 2025-01-21
**PRD Alignment**: v3.0 Four-Component Architecture
