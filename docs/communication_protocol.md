# Workspace-Qdrant-MCP Communication Protocol Specification

**Version:** 1.0
**Date:** 2025-01-02
**Status:** Design Document

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Principles](#architecture-principles)
3. [Protocol Stack](#protocol-stack)
4. [Service Definitions](#service-definitions)
5. [Message Types](#message-types)
6. [Communication Patterns](#communication-patterns)
7. [Error Handling](#error-handling)
8. [Configuration](#configuration)
9. [Future Considerations](#future-considerations)

---

## Overview

This document specifies the complete communication protocol between the components of the workspace-qdrant-mcp system. The system consists of three primary components:

1. **MCP Server (Python)** - FastMCP server providing Model Context Protocol interface to Claude Desktop
2. **Rust Daemon** - High-performance background service handling file watching, queue processing, and Qdrant writes
3. **CLI (Python)** - Command-line interface for system management and operations

### Design Goals

- **Lean and focused** - Minimal RPCs covering only essential operations
- **Performance-oriented** - Binary protobuf encoding for efficiency
- **Clear responsibilities** - Single writer (daemon), multiple readers pattern
- **Type safety** - Strongly-typed protobuf messages throughout
- **Extensibility** - Designed for future enhancements without breaking changes

---

## Architecture Principles

### 1. Single Writer Pattern

**The Rust daemon is the EXCLUSIVE writer to Qdrant.**

```
┌─────────────┐                    ┌──────────────┐
│ MCP Server  │                    │              │
│  (Python)   │────── Reads ──────→│   Qdrant     │
│             │                    │   Database   │
└─────────────┘                    │              │
                                   │              │
┌─────────────┐                    │              │
│     CLI     │────── Reads ──────→│              │
│  (Python)   │                    │              │
└─────────────┘                    │              │
                                   │              │
┌─────────────┐                    │              │
│   Daemon    │────── Writes ─────→│              │
│   (Rust)    │────── Reads ───────→│              │
└─────────────┘                    └──────────────┘
```

**Write Operations (Daemon only):**
- Document upserts (embeddings, metadata)
- Document deletions
- Collection creation/deletion
- Collection alias management

**Read Operations (All components):**
- Vector search (hybrid, semantic, sparse)
- Document retrieval
- Collection listing
- Metadata queries

### 2. Queue-Based Asynchronous Processing

File-based operations are asynchronous via SQLite queue:

```
MCP/CLI → SQLite (enqueue_file) → Daemon polls queue → Processes → Qdrant
```

**Not:**
```
MCP/CLI → gRPC AddDocument → Daemon → Qdrant  ❌
```

### 3. Direct String Ingestion

For non-file content (user-provided text, chat snippets, etc.), use synchronous gRPC:

```
MCP/CLI → gRPC IngestText → Daemon processes immediately → Qdrant
```

### 4. Event-Driven Memory Refresh

Python components signal database changes to daemon via lightweight gRPC:

```
MCP → SQLite write → SendRefreshSignal(INGEST_QUEUE) → Daemon batches refresh
```

Daemon batches signals and refreshes internal state periodically (threshold: 10 items or 5 seconds).

### 5. Connection Strategies

**Python to Qdrant:**
- Small lazy connection pool (max 2 connections)
- Ephemeral for occasional queries
- Close connections after 60s idle

**Daemon to Qdrant:**
- Persistent connection pool (max 10 connections)
- Long-lived for continuous processing
- Configured via `default_configuration.yaml`

**Python to Daemon (gRPC):**
- Connection pool managed by gRPC client library
- Automatic reconnection on failures
- Health checks via `HealthCheck()` RPC

---

## Protocol Stack

### Transport Layer

**Protocol:** gRPC with Protocol Buffers v3

**Advantages:**
- Binary encoding (3-5x smaller payloads vs JSON)
- 2-3x faster parsing than JSON
- Type safety with schema validation
- Automatic code generation
- Backward compatibility support
- Streaming support for future use

**Port Allocation:**
- Daemon gRPC: **50051** (configurable in `default_configuration.yaml`)
- MCP HTTP endpoint: **8765** (configurable, avoiding common collisions)
- Qdrant HTTP: **6333** (standard)
- Qdrant gRPC: **6334** (standard)

### Schema Definition

**Location:** `src/rust/daemon/core/proto/workspace_daemon.proto` (DEPRECATED - see `src/rust/daemon/proto/`)

**Package:** `workspace_daemon`

**Common imports:**
```protobuf
import "google/protobuf/timestamp.proto";
import "google/protobuf/empty.proto";
```

---

## Service Definitions

The daemon exposes **3 gRPC services** with **15 total RPCs**.

### 1. SystemService (7 RPCs)

**Purpose:** System health monitoring, status reporting, refresh signaling, and lifecycle management.

```protobuf
service SystemService {
    // Quick health check for monitoring/alerting
    rpc HealthCheck(google.protobuf.Empty) returns (HealthCheckResponse);

    // Comprehensive system state snapshot
    rpc GetStatus(google.protobuf.Empty) returns (SystemStatusResponse);

    // Current performance metrics (no historical data)
    rpc GetMetrics(google.protobuf.Empty) returns (MetricsResponse);

    // Signal database state changes for event-driven refresh
    rpc SendRefreshSignal(RefreshSignalRequest) returns (google.protobuf.Empty);

    // MCP/CLI server lifecycle notifications
    rpc NotifyServerStatus(ServerStatusNotification) returns (google.protobuf.Empty);

    // Pause all file watchers (master switch)
    rpc PauseAllWatchers(google.protobuf.Empty) returns (google.protobuf.Empty);

    // Resume all file watchers (master switch)
    rpc ResumeAllWatchers(google.protobuf.Empty) returns (google.protobuf.Empty);
}
```

#### HealthCheck

**Purpose:** Minimal overhead health probe for liveness checks.

**Request:** Empty

**Response:**
```protobuf
message HealthCheckResponse {
    ServiceStatus status = 1;                     // Overall health
    repeated ComponentHealth components = 2;      // Per-component status
    google.protobuf.Timestamp timestamp = 3;      // Check timestamp
}

enum ServiceStatus {
    SERVICE_STATUS_UNSPECIFIED = 0;
    SERVICE_STATUS_HEALTHY = 1;
    SERVICE_STATUS_DEGRADED = 2;
    SERVICE_STATUS_UNHEALTHY = 3;
    SERVICE_STATUS_UNAVAILABLE = 4;
}

message ComponentHealth {
    string component_name = 1;                    // e.g., "queue_processor", "file_watcher"
    ServiceStatus status = 2;
    string message = 3;                           // Human-readable status
    google.protobuf.Timestamp last_check = 4;
}
```

**Use cases:**
- MCP startup health verification
- CLI `wqm service status` command
- Monitoring system health checks
- Load balancer health probes

#### GetStatus

**Purpose:** Detailed system state for dashboards and diagnostics.

**Request:** Empty

**Response:**
```protobuf
message SystemStatusResponse {
    ServiceStatus status = 1;
    SystemMetrics metrics = 2;                    // CPU, memory, disk
    repeated string active_projects = 3;          // Currently watched projects
    int32 total_documents = 4;                    // Documents in Qdrant
    int32 total_collections = 5;                  // Collections in Qdrant
    google.protobuf.Timestamp uptime_since = 6;   // Daemon start time
}

message SystemMetrics {
    double cpu_usage_percent = 1;
    int64 memory_usage_bytes = 2;
    int64 memory_total_bytes = 3;
    int64 disk_usage_bytes = 4;
    int64 disk_total_bytes = 5;
    int32 active_connections = 6;                 // Active gRPC connections
    int32 pending_operations = 7;                 // Queue depth
}
```

**Use cases:**
- CLI `wqm admin status` command
- System dashboard displays
- Capacity planning
- Performance troubleshooting

#### GetMetrics

**Purpose:** Current performance metrics snapshot (no time-series).

**Request:** Empty (removed `since` parameter and filtering)

**Response:**
```protobuf
message MetricsResponse {
    repeated Metric metrics = 1;
    google.protobuf.Timestamp collected_at = 2;
}

message Metric {
    string name = 1;                              // e.g., "queue_throughput"
    string type = 2;                              // "counter", "gauge", "histogram"
    map<string, string> labels = 3;               // Metric dimensions
    double value = 4;
    google.protobuf.Timestamp timestamp = 5;
}
```

**Design decision:** No historical metrics. Daemon reports current snapshot only. Time-series tracking is the responsibility of external monitoring systems (Prometheus, Grafana, etc.).

**Use cases:**
- Real-time performance monitoring
- Bottleneck identification
- Export to monitoring systems

#### SendRefreshSignal

**Purpose:** Event-driven memory refresh when database state changes (Task 367).

**Request:**
```protobuf
message RefreshSignalRequest {
    QueueType queue_type = 1;
    repeated string lsp_languages = 2;            // For TOOLS_AVAILABLE signal
    repeated string grammar_languages = 3;        // For TOOLS_AVAILABLE signal
}

enum QueueType {
    QUEUE_TYPE_UNSPECIFIED = 0;
    INGEST_QUEUE = 1;           // New items in ingestion_queue table
    WATCHED_PROJECTS = 2;        // New project in projects table
    WATCHED_FOLDERS = 3;         // New watch in watch_folders table
    TOOLS_AVAILABLE = 4;         // LSP/tree-sitter became available
}
```

**Response:** Empty (fire-and-forget)

**Behavior:**
1. Python writes to SQLite database
2. Python sends lightweight refresh signal
3. Daemon batches signals (threshold: 10 signals OR 5 seconds)
4. Daemon refreshes internal state from database
5. Daemon processes batched changes

**Configuration:** Batching thresholds configurable in `default_configuration.yaml`:
```yaml
refresh_batch_config:
  count_threshold: 10
  time_threshold_secs: 5
  enable_batching: true
```

**Use cases:**
- MCP enqueues file → signals `INGEST_QUEUE`
- CLI adds watch folder → signals `WATCHED_FOLDERS`
- LSP server detected → signals `TOOLS_AVAILABLE`
- MCP detects new project → signals `WATCHED_PROJECTS`

#### NotifyServerStatus

**Purpose:** MCP/CLI server lifecycle notifications for coordination and tracking.

**Request:**
```protobuf
message ServerStatusNotification {
    ServerState state = 1;                        // UP or DOWN
    optional string project_name = 2;             // Git repo name or folder name
    optional string project_root = 3;             // Absolute path to project root
}

enum ServerState {
    SERVER_STATE_UNSPECIFIED = 0;
    SERVER_STATE_UP = 1;                          // Server started
    SERVER_STATE_DOWN = 2;                        // Server shutting down
}
```

**Response:** Empty

**Behavior:**
1. On `SERVER_STATE_UP` with `project_root`: Daemon starts watching project (if not already watched)
2. On `SERVER_STATE_DOWN`: Daemon logs disconnect, tracks session end
3. Daemon maintains registry of active MCP sessions per project

**Use cases:**
- MCP server startup: notify daemon of active project
- MCP server shutdown: clean disconnect notification
- Daemon tracks which projects have active editing sessions
- Future: auto-pause watchers when MCP active in project

**Example usage:**
```python
# MCP server startup
daemon_client.notify_server_status(
    state=ServerState.SERVER_STATE_UP,
    project_name="workspace-qdrant-mcp",
    project_root="/Users/chris/dev/projects/workspace-qdrant-mcp"
)

# MCP server shutdown
daemon_client.notify_server_status(
    state=ServerState.SERVER_STATE_DOWN
)
```

#### PauseAllWatchers

**Purpose:** Master switch to pause all file watching across all projects.

**Request:** Empty

**Response:** Empty

**Behavior:**
1. Daemon pauses file system event monitoring for all watched projects
2. Queue processing continues (existing queue items still processed)
3. New file changes are not detected until watchers resumed

**Use cases:**
- During active editing sessions via MCP HTTP endpoint
- Maintenance operations that modify many files
- Testing/debugging scenarios
- Reducing system resource usage temporarily

#### ResumeAllWatchers

**Purpose:** Master switch to resume all file watching.

**Request:** Empty

**Response:** Empty

**Behavior:**
1. Daemon resumes file system event monitoring for all watched projects
2. Performs catch-up scan to detect changes made while paused
3. Enqueues any new/modified files found during catch-up

**Use cases:**
- Resume after editing session ends
- Resume after maintenance operations complete
- Re-enable normal operation after testing

---

### 2. CollectionService (5 RPCs)

**Purpose:** Qdrant collection lifecycle and alias management.

**Design note:** ListCollections removed - MCP queries Qdrant directly.

```protobuf
service CollectionService {
    // Create collection with proper configuration
    rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);

    // Delete collection and all its data
    rpc DeleteCollection(DeleteCollectionRequest) returns (google.protobuf.Empty);

    // Create collection alias (for tenant_id changes)
    rpc CreateCollectionAlias(CreateAliasRequest) returns (google.protobuf.Empty);

    // Delete collection alias
    rpc DeleteCollectionAlias(DeleteAliasRequest) returns (google.protobuf.Empty);

    // Atomically rename alias (safer than delete + create)
    rpc RenameCollectionAlias(RenameAliasRequest) returns (google.protobuf.Empty);
}
```

#### CreateCollection

**Purpose:** Create Qdrant collection with proper vector configuration.

**Request:**
```protobuf
message CreateCollectionRequest {
    string collection_name = 1;
    string project_id = 2;                        // Optional project association
    CollectionConfig config = 3;                  // Vector configuration
}

message CollectionConfig {
    int32 vector_size = 1;                        // Must match embedding model (e.g., 384)
    string distance_metric = 2;                   // "Cosine", "Euclidean", "Dot"
    bool enable_indexing = 3;                     // HNSW index activation
    map<string, string> metadata_schema = 4;      // Expected metadata fields
}
```

**Response:**
```protobuf
message CreateCollectionResponse {
    bool success = 1;
    string error_message = 2;
    string collection_id = 3;                     // Qdrant's internal ID
}
```

**Validation:**
- Collection name must not already exist (Qdrant enforces)
- vector_size must match configured embedding model
- distance_metric must be valid Qdrant enum

**Use cases:**
- MCP creates collection on first document for new project
- CLI `wqm admin collections create` command
- Automatic collection creation during project detection

#### DeleteCollection

**Purpose:** Remove collection and all vectors.

**Request:**
```protobuf
message DeleteCollectionRequest {
    string collection_name = 1;
    string project_id = 2;                        // For validation
    bool force = 3;                               // Skip confirmation checks
}
```

**Response:** Empty

**Validation:**
- Collection must exist
- If `force=false`, fail if collection has documents
- Check for aliases pointing to this collection

**Use cases:**
- CLI `wqm admin collections delete` command
- Cleanup during project removal
- Testing/development reset operations

#### CreateCollectionAlias

**Purpose:** Create alias pointing to collection (for tenant_id transitions).

**Request:**
```protobuf
message CreateAliasRequest {
    string alias_name = 1;                        // The new name
    string collection_name = 2;                   // Points to this collection
}
```

**Response:** Empty

**Qdrant behavior:**
- Aliases and collections share unified namespace
- Alias name must be unique (cannot conflict with collection names or other aliases)
- One collection can have multiple aliases
- Queries work with both collection name and alias

**Collision handling:**
- Qdrant returns error if alias name already exists
- Daemon enforces strict uniqueness (no collision avoidance, only disallowance)

**Use cases:**
- Project transition local → remote (tenant_id change)
- Collection renaming without data copy
- Maintaining backward compatibility during migrations

#### DeleteCollectionAlias

**Purpose:** Remove alias (collection remains).

**Request:**
```protobuf
message DeleteAliasRequest {
    string alias_name = 1;
}
```

**Response:** Empty

**Use cases:**
- Cleanup old aliases after transition complete
- Remove deprecated collection names

#### RenameCollectionAlias

**Purpose:** Atomically rename alias (safer than separate delete + create).

**Request:**
```protobuf
message RenameAliasRequest {
    string old_alias_name = 1;
    string new_alias_name = 2;
    string collection_name = 3;                   // The collection it points to
}
```

**Response:** Empty

**Implementation:** Uses Qdrant's atomic alias update operation documented as safer than separate delete/create.

**Use cases:**
- Updating alias during tenant_id changes
- Correcting alias naming mistakes

---

### 3. DocumentService (3 RPCs)

**Purpose:** Direct text ingestion (not file-based).

**Design rationale:** For content not originating from files - user input, chat snippets, scraped web content, manual notes.

```protobuf
service DocumentService {
    // Ingest text content directly (synchronous)
    rpc IngestText(IngestTextRequest) returns (IngestTextResponse);

    // Update previously ingested text
    rpc UpdateText(UpdateTextRequest) returns (UpdateTextResponse);

    // Delete ingested text document
    rpc DeleteText(DeleteTextRequest) returns (google.protobuf.Empty);
}
```

#### IngestText

**Purpose:** Process and index user-provided text immediately.

**Request:**
```protobuf
message IngestTextRequest {
    string content = 1;                           // The text to ingest
    string collection_basename = 2;               // e.g., "memory", "scratchbook"
    string tenant_id = 3;                         // Multi-tenant identifier
    optional string document_id = 4;              // Custom ID (generated if omitted)
    map<string, string> metadata = 5;             // Additional metadata
    bool chunk_text = 6;                          // Whether to chunk (default: true)
}
```

**Response:**
```protobuf
message IngestTextResponse {
    string document_id = 1;                       // For future updates/deletes
    bool success = 2;
    int32 chunks_created = 3;                     // Number of chunks generated
    string error_message = 4;
}
```

**Processing flow:**
1. Daemon receives request
2. Chunks text if `chunk_text=true` (using configured chunk_size/overlap)
3. Generates embeddings (dense + sparse if enabled)
4. Constructs collection name from `collection_basename` and `tenant_id`
5. Creates collection if doesn't exist (auto-create)
6. Upserts vectors to Qdrant
7. Returns document_id for tracking

**Synchronous operation:** User waits for completion (typically <1s for small texts).

**Use cases:**
- MCP stores conversation snippet
- User pastes code via CLI for quick search
- Scraping web content into knowledge base
- Manual note-taking

#### UpdateText

**Purpose:** Update content of previously ingested text.

**Request:**
```protobuf
message UpdateTextRequest {
    string document_id = 1;                       // From IngestTextResponse
    string content = 2;                           // New content
    optional string collection_name = 3;          // If moving to different collection
    map<string, string> metadata = 4;             // Updated metadata
}
```

**Response:**
```protobuf
message UpdateTextResponse {
    bool success = 1;
    string error_message = 2;
    google.protobuf.Timestamp updated_at = 3;
}
```

**Processing flow:**
1. Delete old vectors for document_id
2. Re-chunk and re-embed new content
3. Upsert new vectors
4. Update metadata timestamp

**Use cases:**
- Editing saved notes
- Updating conversation context
- Correcting ingested content

#### DeleteText

**Purpose:** Remove ingested text document.

**Request:**
```protobuf
message DeleteTextRequest {
    string document_id = 1;
    string collection_name = 2;                   // For validation
}
```

**Response:** Empty

**Use cases:**
- Removing outdated notes
- Cleaning up test data
- User-requested deletion

---

## Message Types

### Common Enums

#### DocumentType
```protobuf
enum DocumentType {
    DOCUMENT_TYPE_UNSPECIFIED = 0;
    DOCUMENT_TYPE_CODE = 1;
    DOCUMENT_TYPE_PDF = 2;
    DOCUMENT_TYPE_EPUB = 3;
    DOCUMENT_TYPE_MOBI = 4;
    DOCUMENT_TYPE_HTML = 5;
    DOCUMENT_TYPE_TEXT = 6;
    DOCUMENT_TYPE_MARKDOWN = 7;
    DOCUMENT_TYPE_JSON = 8;
    DOCUMENT_TYPE_XML = 9;
}
```

#### ProcessingStatus
```protobuf
enum ProcessingStatus {
    PROCESSING_STATUS_UNSPECIFIED = 0;
    PROCESSING_STATUS_PENDING = 1;
    PROCESSING_STATUS_IN_PROGRESS = 2;
    PROCESSING_STATUS_COMPLETED = 3;
    PROCESSING_STATUS_FAILED = 4;
    PROCESSING_STATUS_CANCELLED = 5;
}
```

### Timestamp Handling

All timestamps use `google.protobuf.Timestamp` (RFC 3339 format):
- Stored as seconds + nanoseconds since Unix epoch
- Timezone-aware (UTC)
- Automatic conversion in generated code

---

## Communication Patterns

### 1. Request-Response (Synchronous)

**Pattern:** Client sends request, waits for response.

**Examples:**
- `HealthCheck()` - Quick status check
- `CreateCollection()` - Immediate collection creation
- `IngestText()` - Synchronous text processing

**Timeout configuration:**
```yaml
grpc:
  connection_timeout: 10s
  # Per-operation timeouts in client code
```

**Error handling:**
- Network errors: Automatic retry with exponential backoff
- Application errors: Returned in response message
- Timeout: Client receives `DEADLINE_EXCEEDED` error

### 2. Fire-and-Forget (Asynchronous Signal)

**Pattern:** Client sends signal, doesn't wait for processing.

**Examples:**
- `SendRefreshSignal()` - Daemon batches and processes later

**Implementation:**
```python
# Client side
daemon_client.send_refresh_signal(QueueType.INGEST_QUEUE)
# Returns immediately, daemon processes in background
```

**Reliability:**
- No delivery guarantee (acceptable for hints/signals)
- Daemon batches for efficiency
- If signal lost, next signal triggers catch-up

### 3. Database-Mediated (Queue Pattern)

**Pattern:** Asynchronous via SQLite queue.

**Flow:**
```
1. MCP writes to SQLite: state_manager.enqueue_file(path, operation, priority)
2. MCP signals daemon: daemon_client.send_refresh_signal(INGEST_QUEUE)
3. Daemon polls queue periodically
4. Daemon processes items from queue
5. Daemon writes to Qdrant
6. Daemon marks item complete in queue
```

**Advantages:**
- Decoupled components
- Survives daemon restarts (persistent queue)
- Automatic retry on failures
- Priority-based processing

**Use cases:**
- File ingestion (all file-based operations)
- Bulk operations
- Long-running processing

---

## Error Handling

### gRPC Status Codes

Standard gRPC status codes used:

| Code | Usage |
|------|-------|
| `OK` | Successful operation |
| `CANCELLED` | Client cancelled request |
| `INVALID_ARGUMENT` | Invalid request parameters |
| `DEADLINE_EXCEEDED` | Operation timeout |
| `NOT_FOUND` | Resource doesn't exist |
| `ALREADY_EXISTS` | Collection/alias name collision |
| `PERMISSION_DENIED` | Authorization failure |
| `RESOURCE_EXHAUSTED` | Rate limit or quota exceeded |
| `FAILED_PRECONDITION` | Operation not allowed in current state |
| `INTERNAL` | Daemon internal error |
| `UNAVAILABLE` | Service temporarily unavailable |

### Error Response Pattern

**For operations returning custom response:**
```protobuf
message CreateCollectionResponse {
    bool success = 1;
    string error_message = 2;  // Human-readable error if success=false
    string collection_id = 3;   // Only set if success=true
}
```

**For operations returning Empty:**
```protobuf
rpc DeleteCollection(...) returns (google.protobuf.Empty);
// Errors returned via gRPC status code + details
```

### Retry Strategy

**Client-side retry logic:**

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=1, max=30),
    retry=retry_if_exception_type(grpc.RpcError),
    retry_error_callback=lambda retry_state: log_retry(retry_state)
)
async def call_daemon_rpc(...):
    ...
```

**Retryable errors:**
- `UNAVAILABLE` - Service temporarily down
- `DEADLINE_EXCEEDED` - Timeout, may succeed on retry
- `RESOURCE_EXHAUSTED` - Rate limit, retry after backoff

**Non-retryable errors:**
- `INVALID_ARGUMENT` - Client bug, won't succeed on retry
- `ALREADY_EXISTS` - Collision, retry won't help
- `NOT_FOUND` - Resource missing, retry won't help

### Circuit Breaker

**Configuration:**
```yaml
qdrant:
  circuit_breaker:
    enabled: true
    failure_threshold: 5       # Open after 5 failures
    success_threshold: 3        # Close after 3 successes
    timeout: 60s               # Stay open for 60s
    half_open_timeout: 30s     # Test probe timeout
```

**States:**
1. **Closed:** Normal operation
2. **Open:** Fast-fail without attempting request (after threshold failures)
3. **Half-Open:** Testing recovery with limited requests

---

## Configuration

### Single Source of Truth

**File:** `assets/default_configuration.yaml`

**All components read same configuration file.**

### gRPC Configuration Section

```yaml
grpc:
  enabled: true
  host: "127.0.0.1"              # Localhost only for security
  port: 50051                    # Standard gRPC port
  fallback_to_direct: true       # Use direct Qdrant if gRPC fails
  connection_timeout: 10s
  max_retries: 3
  retry_backoff_multiplier: 1.5
  health_check_interval: 30s
  max_message_length: 100MB      # Large for batch operations
  keepalive_time: 30s
```

### MCP HTTP Endpoint Configuration

```yaml
server:
  host: "127.0.0.1"
  port: 8765                     # MCP HTTP endpoint (avoiding collisions)
  debug: false
```

### Rust Client Configuration

```yaml
rust:
  client:
    connection_timeout: 30s
    request_timeout: 60s
    max_retries: 3
    retry_delay: 1s
    max_retry_delay: 30s
```

---

## Future Considerations

### Multi-Tenancy Support

**Status:** Deferred to later design phase.

**Current approach:**
- `tenant_id` as separate field in messages
- Daemon constructs collection names internally
- Format TBD during multi-tenancy design

**Open questions:**
- Collection naming scheme
- Tenant isolation guarantees
- Cross-tenant operations
- Tenant lifecycle management

### Streaming Operations

**Potential future additions:**

```protobuf
// Server-side streaming for large result sets
rpc StreamSearchResults(SearchRequest) returns (stream SearchResult);

// Client-side streaming for bulk ingestion
rpc StreamIngestText(stream IngestTextRequest) returns (IngestSummaryResponse);

// Bidirectional streaming for real-time sync
rpc SyncDocuments(stream DocumentUpdate) returns (stream SyncStatus);
```

**Use cases:**
- Large search result pagination
- Bulk text ingestion
- Real-time collaboration features

### Additional Services

**Not currently needed, but may be added:**

**WatchService:**
```protobuf
service WatchService {
    rpc AddWatch(AddWatchRequest) returns (google.protobuf.Empty);
    rpc RemoveWatch(RemoveWatchRequest) returns (google.protobuf.Empty);
    rpc ListWatches() returns (ListWatchesResponse);
}
```

**Why not included:** Watch configuration via direct SQLite writes is simpler.

**SearchService:**
```protobuf
service SearchService {
    rpc HybridSearch(HybridSearchRequest) returns (HybridSearchResponse);
    rpc SemanticSearch(SemanticSearchRequest) returns (SemanticSearchResponse);
    rpc KeywordSearch(KeywordSearchRequest) returns (KeywordSearchResponse);
}
```

**Why not included:** MCP queries Qdrant directly for reads (no extra hop needed).

### Authentication & Authorization

**Current:** No authentication (local-only deployment).

**Future considerations:**
- mTLS for gRPC connections
- API key authentication
- Per-collection ACLs
- Audit logging

---

## Appendix: Removed Services

### Services Eliminated During Design

**DocumentProcessor Service** - Redundant
- Processing happens via queue (file-based) or IngestText (string-based)
- No need for separate batch processing RPCs

**SearchService** - Redundant
- MCP queries Qdrant directly
- No value in daemon proxy for read operations

**MemoryService** - Split and renamed
- Document operations → DocumentService (direct text only)
- Collection operations → CollectionService
- Reads removed (MCP queries Qdrant directly)

**ServiceDiscovery** - Over-engineered
- Not building distributed system (single daemon instance)
- YAGNI (You Ain't Gonna Need It)

**ProjectService** - Removed entirely
- `ListProjects` = query SQLite `projects` table or Qdrant collections
- No daemon involvement needed for read operations

### RPCs Eliminated

From original 20+ RPCs to final 12 RPCs:

**Removed from consideration:**
- `GetConfig` / `UpdateConfig` - Just read config file, restart daemon
- `ProcessDocument` / `ProcessDocuments` - Use queue instead
- `GetProcessingStatus` / `CancelProcessing` - Queue status via SQLite
- All search operations - MCP queries Qdrant directly
- `AddDocument` / `UpdateDocument` / `RemoveDocument` - Use queue or IngestText
- `GetDocument` / `ListDocuments` - MCP queries Qdrant directly
- `ListCollections` - MCP queries Qdrant directly
- `DetectProject` - MCP does this via file scanning
- `ListProjects` - MCP queries SQLite or Qdrant
- All service discovery operations - Not needed

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-02 | Design Session | Initial specification |
| 1.1 | 2025-01-02 | Design Session | Added SystemService RPCs: NotifyServerStatus, PauseAllWatchers, ResumeAllWatchers (15 total RPCs) |
| 1.2 | 2025-01-02 | Design Session | Added MCP HTTP Endpoint Protocol specification with 9 hook endpoints (2 critical, 7 placeholders) |

---

## MCP HTTP Endpoint Protocol

**Purpose:** Allow Claude Code hooks and external integrations to interact with MCP server.

**Transport:** HTTP/1.1 with JSON payloads

**Port:** 8765 (configurable in `default_configuration.yaml`)

**Base URL:** `http://127.0.0.1:8765/api/v1`

### Configuration

```yaml
# In assets/default_configuration.yaml
http_endpoint:
  enabled: true
  host: "127.0.0.1"        # Localhost-only for security
  port: 8765               # Avoiding common port collisions
  timeout_seconds: 30
```

---

### Hook Event Endpoints

The MCP HTTP endpoint provides endpoints corresponding to Claude Code hook events. All endpoints accept JSON requests and return JSON responses.

#### 1. PreToolUse Hook

```http
POST /api/v1/hooks/pre-tool-use

Request:
{
  "tool_name": "Edit",
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Validate tool usage against project policies
- Pre-process tool parameters
- Log tool usage

---

#### 2. PostToolUse Hook

```http
POST /api/v1/hooks/post-tool-use

Request:
{
  "tool_name": "Edit",
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Enqueue modified files for ingestion
- Track file changes per session
- Update project statistics

---

#### 3. UserPromptSubmit Hook

```http
POST /api/v1/hooks/user-prompt-submit

Request:
{
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Inject project context into prompts
- Update memory before prompt processing
- Log user interactions

---

#### 4. Notification Hook

```http
POST /api/v1/hooks/notification

Request:
{
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Forward notifications to external systems
- Log notification events
- Trigger custom actions

---

#### 5. Stop Hook

```http
POST /api/v1/hooks/stop

Request:
{
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Flush pending operations
- Update session statistics
- Trigger post-response actions

---

#### 6. SubagentStop Hook

```http
POST /api/v1/hooks/subagent-stop

Request:
{
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Track subagent completion
- Aggregate subagent results
- Update agent statistics

---

#### 7. PreCompact Hook

```http
POST /api/v1/hooks/pre-compact

Request:
{
  "session_id": "string",
  "project_dir": "string"
}

Response 200 OK:
{
  "success": true,
  "message": "Hook received"
}
```

**Status:** Placeholder - not implemented. Reserved for future use.

**Potential use cases:**
- Save session state before compaction
- Archive conversation history
- Update memory collections

---

#### 8. SessionStart Hook (CRITICAL - Implemented)

```http
POST /api/v1/hooks/session-start

Request:
{
  "session_id": "string",
  "project_dir": "string",
  "source": "startup" | "clear" | "compact"
}

Response 200 OK:
{
  "success": true,
  "message": "Session started, memory collection ingestion triggered"
}
```

**Status:** CRITICAL - Implementation required.

**Behavior:**

When `source` is one of `["startup", "clear", "compact"]`:
1. Trigger memory collection ingestion for the project
2. Begin tracking session for file changes
3. Return immediately (ingestion is asynchronous)

**Notes:**
- Server starts asynchronously; user may prompt before ingestion completes
- Ingestion latency is acceptable (background operation)
- Memory collection design to be fleshed out in implementation phase

**Use cases:**
- New session: ingest latest project state into memory
- After clear: refresh memory from current project state
- After compact: re-establish memory context

---

#### 9. SessionEnd Hook (CRITICAL - Implemented)

```http
POST /api/v1/hooks/session-end

Request:
{
  "session_id": "string",
  "reason": "clear" | "logout" | "other" | "prompt_input_exit"
}

Response 200 OK:
{
  "success": true,
  "message": "Session ended, daemon notified"
}
```

**Status:** CRITICAL - Implementation required.

**Behavior:**

When `reason` is one of `["other", "prompt_input_exit"]`:
1. Send `NotifyServerStatus(state=SERVER_STATE_DOWN)` to daemon via gRPC
2. Clean up session tracking
3. Return success

When `reason` is `"clear"` or `"logout"`:
- Session transitions, not full shutdown
- Do NOT notify daemon of shutdown
- Clean up session-specific state only

**Use cases:**
- Session exit: notify daemon server is shutting down
- Clear/logout: clean up session state without full shutdown

---

### Utility Endpoints

#### Health Check

```http
GET /api/v1/health

Response 200 OK:
{
  "status": "healthy",
  "daemon_connected": true,
  "qdrant_connected": true,
  "version": "0.2.1"
}

Response 503 Service Unavailable:
{
  "status": "unhealthy",
  "daemon_connected": false,
  "qdrant_connected": true,
  "version": "0.2.1"
}
```

**Status:** Standard - always implemented.

**Use cases:**
- Monitoring and health checks
- Hook script validation
- Service availability verification

---

### Error Handling

All endpoints use standard HTTP status codes:

| Status | Meaning |
|--------|---------|
| 200 OK | Request processed successfully |
| 400 Bad Request | Invalid JSON or missing required fields |
| 500 Internal Server Error | Server-side error during processing |
| 503 Service Unavailable | Server not ready or dependencies unavailable |

**Error response format:**
```json
{
  "success": false,
  "error": "Invalid JSON payload",
  "details": "Expected field 'session_id'"
}
```

---

### Security Considerations

**Localhost-only binding:**
- Server binds to 127.0.0.1 only
- No external network access
- No authentication required (local trust model)

**Future considerations:**
- API key authentication for remote access
- HTTPS/TLS for encrypted transport
- Rate limiting per client

---

### Implementation Priority

**Phase 1 (Critical):**
1. SessionStart hook endpoint
2. SessionEnd hook endpoint
3. Health check endpoint

**Phase 2 (Future):**
- PostToolUse for file change tracking
- UserPromptSubmit for context injection
- Other hooks as use cases emerge

---

**End of Communication Protocol Specification**
