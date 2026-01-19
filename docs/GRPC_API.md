# gRPC API Reference

Complete reference documentation for the workspace-qdrant-mcp gRPC API. The daemon exposes 4 services with 20 RPCs for communication with MCP server and CLI.

## Table of Contents

- [Overview](#overview)
- [Connection Details](#connection-details)
- [Services](#services)
  - [SystemService](#systemservice)
  - [CollectionService](#collectionservice)
  - [DocumentService](#documentservice)
  - [ProjectService](#projectservice)
- [Common Types](#common-types)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Overview

The unified daemon (`memexd`) provides gRPC services for:

- **SystemService** (7 RPCs): Health monitoring, metrics, lifecycle management
- **CollectionService** (5 RPCs): Collection CRUD, alias management
- **DocumentService** (3 RPCs): Direct text ingestion, updates, deletion
- **ProjectService** (5 RPCs): Multi-tenant project registration and session management

**Design Principles:**
- **Single writer pattern**: Only daemon writes to Qdrant
- **Queue-based async processing**: File operations via SQLite queue
- **Direct sync ingestion**: Text content via gRPC `IngestText`
- **Session-based priority**: Active MCP sessions get HIGH priority

**Protocol Definition:** `src/rust/daemon/proto/workspace_daemon.proto`

## Connection Details

```
Host: 127.0.0.1 (configurable via WQM_DAEMON_HOST)
Port: 50051 (configurable via WQM_DAEMON_PORT)
Protocol: gRPC (HTTP/2)
TLS: Optional (recommended for production)
```

**Python Client Example:**

```python
import grpc
from workspace_daemon_pb2_grpc import SystemServiceStub

channel = grpc.insecure_channel("127.0.0.1:50051")
client = SystemServiceStub(channel)
```

## Services

### SystemService

Health monitoring, status reporting, and lifecycle management.

#### HealthCheck

Quick health check for monitoring and alerting.

```protobuf
rpc HealthCheck(google.protobuf.Empty) returns (HealthCheckResponse);
```

**Response:**
- `status`: Overall service status (HEALTHY, DEGRADED, UNHEALTHY, UNAVAILABLE)
- `components[]`: Per-component health (queue_processor, file_watcher, etc.)
- `timestamp`: Check timestamp

**Example:**

```python
response = client.HealthCheck(Empty())
print(f"Status: {response.status}")
for comp in response.components:
    print(f"  {comp.component_name}: {comp.status}")
```

#### GetStatus

Comprehensive system state snapshot.

```protobuf
rpc GetStatus(google.protobuf.Empty) returns (SystemStatusResponse);
```

**Response:**
- `status`: Overall status
- `metrics`: CPU, memory, disk usage
- `active_projects[]`: Currently watched projects
- `total_documents`: Documents in Qdrant
- `total_collections`: Collections in Qdrant
- `uptime_since`: Daemon start time

#### GetMetrics

Current performance metrics (no historical data).

```protobuf
rpc GetMetrics(google.protobuf.Empty) returns (MetricsResponse);
```

**Response:**
- `metrics[]`: Array of metric objects with name, type, labels, value
- `collected_at`: Collection timestamp

#### SendRefreshSignal

Signal database state changes for event-driven refresh.

```protobuf
rpc SendRefreshSignal(RefreshSignalRequest) returns (google.protobuf.Empty);
```

**Request:**
- `queue_type`: INGEST_QUEUE, WATCHED_PROJECTS, WATCHED_FOLDERS, TOOLS_AVAILABLE
- `lsp_languages[]`: For TOOLS_AVAILABLE signal
- `grammar_languages[]`: For TOOLS_AVAILABLE signal

#### NotifyServerStatus

MCP/CLI server lifecycle notifications.

```protobuf
rpc NotifyServerStatus(ServerStatusNotification) returns (google.protobuf.Empty);
```

**Request:**
- `state`: SERVER_STATE_UP or SERVER_STATE_DOWN
- `project_name`: Git repo name or folder name (optional)
- `project_root`: Absolute path to project root (optional)

#### PauseAllWatchers / ResumeAllWatchers

Master switch for file watchers.

```protobuf
rpc PauseAllWatchers(google.protobuf.Empty) returns (google.protobuf.Empty);
rpc ResumeAllWatchers(google.protobuf.Empty) returns (google.protobuf.Empty);
```

---

### CollectionService

Qdrant collection lifecycle and alias management.

#### CreateCollection

Create collection with proper configuration.

```protobuf
rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
```

**Request:**
- `collection_name`: Name (e.g., `_projects`, `myapp-notes`)
- `project_id`: Optional project association
- `config`: Vector configuration (size, distance, indexing)

**Response:**
- `success`: Boolean
- `error_message`: Error details if failed
- `collection_id`: Qdrant's internal ID

**Example:**

```python
from workspace_daemon_pb2 import CreateCollectionRequest, CollectionConfig

request = CreateCollectionRequest(
    collection_name="_projects",
    config=CollectionConfig(
        vector_size=384,
        distance_metric="Cosine",
        enable_indexing=True
    )
)
response = collection_client.CreateCollection(request)
```

#### DeleteCollection

Delete collection and all its data.

```protobuf
rpc DeleteCollection(DeleteCollectionRequest) returns (google.protobuf.Empty);
```

**Request:**
- `collection_name`: Collection to delete
- `project_id`: For validation
- `force`: Skip confirmation checks

#### CreateCollectionAlias / DeleteCollectionAlias / RenameCollectionAlias

Manage collection aliases (useful for tenant_id changes).

```protobuf
rpc CreateCollectionAlias(CreateAliasRequest) returns (google.protobuf.Empty);
rpc DeleteCollectionAlias(DeleteAliasRequest) returns (google.protobuf.Empty);
rpc RenameCollectionAlias(RenameAliasRequest) returns (google.protobuf.Empty);
```

---

### DocumentService

Direct text ingestion for non-file content (user input, chat snippets, scraped web content).

#### IngestText

Ingest text content directly (synchronous).

```protobuf
rpc IngestText(IngestTextRequest) returns (IngestTextResponse);
```

**Request:**
- `content`: The text to ingest (required)
- `collection_basename`: e.g., "notes", "scratchbook"
- `tenant_id`: Multi-tenant identifier (12-char hex)
- `document_id`: Custom ID (generated if omitted)
- `metadata`: Additional key-value pairs
- `chunk_text`: Whether to chunk (default: true)

**Response:**
- `document_id`: For future updates/deletes
- `success`: Boolean
- `chunks_created`: Number of chunks generated
- `error_message`: Error details if failed

**Example:**

```python
from workspace_daemon_pb2 import IngestTextRequest

request = IngestTextRequest(
    content="Important note about authentication flow",
    collection_basename="myapp-notes",
    tenant_id="github_com_user_repo",  # Current project
    metadata={"source": "user_input", "topic": "auth"}
)
response = doc_client.IngestText(request)
print(f"Stored as {response.document_id}, {response.chunks_created} chunks")
```

#### UpdateText

Update previously ingested text.

```protobuf
rpc UpdateText(UpdateTextRequest) returns (UpdateTextResponse);
```

**Request:**
- `document_id`: From IngestTextResponse
- `content`: New content
- `collection_name`: If moving to different collection
- `metadata`: Updated metadata

#### DeleteText

Delete ingested text document.

```protobuf
rpc DeleteText(DeleteTextRequest) returns (google.protobuf.Empty);
```

---

### ProjectService

Multi-tenant project lifecycle and session management. This service enables priority-based ingestion where active MCP sessions get faster processing.

#### RegisterProject

Register a project for high-priority processing. Called when MCP server starts.

```protobuf
rpc RegisterProject(RegisterProjectRequest) returns (RegisterProjectResponse);
```

**Request:**
- `path`: Absolute path to project root (required)
- `project_id`: 12-char hex tenant identifier (required)
- `name`: Human-readable project name (optional)
- `git_remote`: Git remote URL for normalization (optional)

**Response:**
- `created`: True if new project, false if existing
- `project_id`: Confirmed project ID
- `priority`: Current priority ("high", "normal", "low")
- `active_sessions`: Number of active MCP sessions

**Example:**

```python
from workspace_daemon_pb2 import RegisterProjectRequest

request = RegisterProjectRequest(
    path="/Users/dev/myproject",
    project_id="github_com_user_repo",
    name="My Project",
    git_remote="https://github.com/user/repo.git"
)
response = project_client.RegisterProject(request)
print(f"Priority: {response.priority}, Sessions: {response.active_sessions}")
```

**Priority Behavior:**
- When `active_sessions > 0`: Priority is HIGH (queue position 1)
- When `active_sessions == 0`: Priority is NORMAL (queue position 5)
- File changes for HIGH priority projects are processed immediately

#### DeprioritizeProject

Decrement session count when MCP server stops.

```protobuf
rpc DeprioritizeProject(DeprioritizeProjectRequest) returns (DeprioritizeProjectResponse);
```

**Request:**
- `project_id`: 12-char hex identifier

**Response:**
- `success`: Boolean
- `remaining_sessions`: Sessions after decrement
- `new_priority`: Priority after demotion

**Example:**

```python
# On MCP server shutdown
request = DeprioritizeProjectRequest(project_id="github_com_user_repo")
response = project_client.DeprioritizeProject(request)
print(f"Remaining sessions: {response.remaining_sessions}")
```

#### GetProjectStatus

Get current status of a specific project.

```protobuf
rpc GetProjectStatus(GetProjectStatusRequest) returns (GetProjectStatusResponse);
```

**Response:**
- `found`: Whether project exists
- `project_id`, `project_name`, `project_root`
- `priority`: "high", "normal", "low"
- `active_sessions`: Current session count
- `last_active`: Last heartbeat/activity timestamp
- `registered_at`: Registration timestamp
- `git_remote`: Git remote URL (if available)

#### ListProjects

List all registered projects with optional filtering.

```protobuf
rpc ListProjects(ListProjectsRequest) returns (ListProjectsResponse);
```

**Request:**
- `priority_filter`: Filter by "high", "normal", "low", or "all"
- `active_only`: Only return projects with active_sessions > 0

**Response:**
- `projects[]`: Array of ProjectInfo
- `total_count`: Total matching projects

**Example:**

```python
# List all active projects
request = ListProjectsRequest(active_only=True)
response = project_client.ListProjects(request)
for proj in response.projects:
    print(f"{proj.project_name}: {proj.priority} ({proj.active_sessions} sessions)")
```

#### Heartbeat

Keep session alive. Must be called periodically (every 30 seconds recommended).

```protobuf
rpc Heartbeat(HeartbeatRequest) returns (HeartbeatResponse);
```

**Request:**
- `project_id`: 12-char hex identifier

**Response:**
- `acknowledged`: True if heartbeat was accepted
- `next_heartbeat_by`: Deadline for next heartbeat (60s timeout)

**Timeout Behavior:**
- Sessions missed for 60 seconds are marked as orphaned
- Orphaned sessions are cleaned up periodically
- New MCP connections can revive orphaned projects

**Example:**

```python
import asyncio

async def heartbeat_loop(project_id: str):
    while True:
        request = HeartbeatRequest(project_id=project_id)
        response = project_client.Heartbeat(request)
        if not response.acknowledged:
            # Session expired, re-register
            await register_project(project_id)
        await asyncio.sleep(30)  # Send every 30 seconds
```

---

## Common Types

### ServiceStatus

```protobuf
enum ServiceStatus {
    SERVICE_STATUS_UNSPECIFIED = 0;
    SERVICE_STATUS_HEALTHY = 1;
    SERVICE_STATUS_DEGRADED = 2;
    SERVICE_STATUS_UNHEALTHY = 3;
    SERVICE_STATUS_UNAVAILABLE = 4;
}
```

### ProjectPriority

```protobuf
enum ProjectPriority {
    PROJECT_PRIORITY_UNSPECIFIED = 0;
    PROJECT_PRIORITY_HIGH = 1;    // Active agent sessions
    PROJECT_PRIORITY_NORMAL = 2;  // Registered without active sessions
    PROJECT_PRIORITY_LOW = 3;     // Background/inactive
}
```

### QueueType

```protobuf
enum QueueType {
    QUEUE_TYPE_UNSPECIFIED = 0;
    INGEST_QUEUE = 1;        // New items in ingestion_queue table
    WATCHED_PROJECTS = 2;    // New project registered
    WATCHED_FOLDERS = 3;     // New watch folder configured
    TOOLS_AVAILABLE = 4;     // LSP/tree-sitter became available
}
```

### Payload Schemas

For multi-tenant collection payload structures, see the proto file:

- `ProjectPayload`: Documents in `_projects` collection (tenant_id, file_path, branch, symbols, etc.)
- `LibraryPayload`: Documents in `_libraries` collection (library_name, source_file, topics, etc.)
- `MemoryPayload`: Documents in `_memory` collections (rule_id, rule_type, priority, scope, etc.)

---

## Error Handling

gRPC errors use standard status codes:

| Code | Description | Common Causes |
|------|-------------|---------------|
| `INVALID_ARGUMENT` | Bad request parameters | Missing required fields, invalid tenant_id format |
| `NOT_FOUND` | Resource doesn't exist | Unknown collection, invalid document_id |
| `ALREADY_EXISTS` | Duplicate resource | Collection already exists |
| `PERMISSION_DENIED` | Access denied | Auth failure, read-only collection |
| `UNAVAILABLE` | Service unavailable | Daemon not running, Qdrant down |
| `INTERNAL` | Internal error | Unexpected daemon failure |

**Python Error Handling:**

```python
import grpc

try:
    response = client.RegisterProject(request)
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        print(f"Invalid request: {e.details()}")
    elif e.code() == grpc.StatusCode.UNAVAILABLE:
        print("Daemon not available, using fallback")
    else:
        raise
```

---

## Usage Examples

### Complete Session Lifecycle

```python
import grpc
from workspace_daemon_pb2_grpc import ProjectServiceStub
from workspace_daemon_pb2 import (
    RegisterProjectRequest,
    HeartbeatRequest,
    DeprioritizeProjectRequest
)

channel = grpc.insecure_channel("127.0.0.1:50051")
project_client = ProjectServiceStub(channel)

# 1. Register project on MCP server start
register_req = RegisterProjectRequest(
    path="/Users/dev/myproject",
    project_id="github_com_user_repo",
    name="My Project"
)
register_resp = project_client.RegisterProject(register_req)
print(f"Registered with priority: {register_resp.priority}")

# 2. Start heartbeat loop (in background task)
import threading
import time

def heartbeat_thread():
    while session_active:
        try:
            hb_req = HeartbeatRequest(project_id="github_com_user_repo")
            project_client.Heartbeat(hb_req)
        except grpc.RpcError:
            pass  # Handle reconnection
        time.sleep(30)

session_active = True
threading.Thread(target=heartbeat_thread, daemon=True).start()

# 3. On MCP server shutdown
session_active = False
deprio_req = DeprioritizeProjectRequest(project_id="github_com_user_repo")
project_client.DeprioritizeProject(deprio_req)
```

### Ingesting User Notes

```python
from workspace_daemon_pb2_grpc import DocumentServiceStub
from workspace_daemon_pb2 import IngestTextRequest

doc_client = DocumentServiceStub(channel)

# Store a note in user collection
request = IngestTextRequest(
    content="Meeting notes: Discussed API rate limiting strategy",
    collection_basename="work-notes",
    tenant_id="github_com_user_repo",  # Auto-enriched with project context
    metadata={
        "source": "user_input",
        "topic": "meeting",
        "date": "2025-01-19"
    }
)
response = doc_client.IngestText(request)
print(f"Stored: {response.document_id}")
```

---

**Version**: 1.0
**Last Updated**: 2025-01-19
**Protocol Version**: workspace_daemon.proto v2.0
