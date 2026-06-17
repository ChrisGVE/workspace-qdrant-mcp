# gRPC API Reference

Complete reference documentation for the workspace-qdrant-mcp gRPC API. The daemon exposes 13 services with RPCs for communication with MCP server and CLI. This document covers the services most relevant to integration; see the proto file for the full catalog.

## Table of Contents

- [Overview](#overview)
- [Connection Details](#connection-details)
- [Services](#services)
  - [SystemService](#systemservice)
  - [CollectionService](#collectionservice)
  - [DocumentService](#documentservice)
  - [ProjectService](#projectservice)
  - [LibraryWriteService](#librarywriteservice)
- [Common Types](#common-types)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

## Overview

The unified daemon (`memexd`) provides gRPC services for:

- **SystemService** (7 RPCs): Health monitoring, metrics, lifecycle management
- **CollectionService** (5 RPCs): Collection CRUD, alias management
- **DocumentService** (3 RPCs): Direct text ingestion, updates, deletion
- **ProjectService** (6 RPCs): Multi-tenant project registration, session management, and recovery
- **LibraryWriteService** (7 RPCs): Library management mutations, including path recovery

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

**Registration reconciliation (path-based, issues #138/#139):**

`RegisterProject` reconciles by **path** before deciding a project is new,
so an already-registered project is recognized even when its identity inputs
changed since the last registration:

- **Moved path** (#138): if a registered project re-registers at a new path
  while its identity (git remote) is unchanged, the stored `path` is updated
  in place — no duplicate `watch_folders` row, no orphaned data. The response
  reports `created=false`.
- **Tenancy-type flip** (#139): a project's `tenant_id` is `local_<hash(path)>`
  when it has no git remote and `<hash(remote)>` when it does. If a registered
  project gains or loses a remote between registrations, its recomputed id
  changes. Registration detects this (same path, different recomputed id),
  renames the tenant across SQLite (`watch_folders`, `unified_queue`,
  `tracked_files`), and enqueues a Qdrant cascade-rename for the `projects`
  and `rules` collections — instead of registering a second, duplicate project.

  Coverage boundary: the cascade does **not** yet migrate other tenant-keyed
  SQLite tables (`symbol_cooccurrence`, the graph store, `keywords`, `tags`,
  `processing_timings`, priority/affinity tables) or other tenant-keyed Qdrant
  collections (`scratchpad`, `images`). Full coverage is tracked by #140.

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

#### RecoverProject

Reconcile a drifted project registration (#140): re-point a moved project and/or
flip its tenancy (local ↔ remote), rewriting stored file paths and migrating all
tenant_id-keyed data across SQLite and Qdrant.

```protobuf
rpc RecoverProject(RecoverProjectRequest) returns (RecoverProjectResponse);
```

**Request:**
- `project_id`: stored tenant_id of the project to recover
- `new_path` (optional): new absolute path the project moved to; unset keeps the stored path
- `rescan_remote`: recompute tenancy from the current git remote (local ↔ remote)
- `dry_run`: report the planned old→new id/path and row/point counts without writing

**Response:**
- `success`, `dry_run`, `changed` (false = already consistent, idempotent no-op)
- `old_tenant_id` / `new_tenant_id`, `old_path` / `new_path`
- `sqlite_rows_updated`, `qdrant_points_updated`
- `message`: human-readable summary

**Cascade coverage.** A tenancy flip re-keys every tenant_id-keyed table in
state.db (`watch_folders`, `unified_queue`, `keywords`, `tags`, `keyword_baskets`,
`canonical_tags`, `tag_hierarchy_edges`, `rules_mirror`, `symbol_cooccurrence`,
`project_groups`, `project_embeddings`, `processing_timings`), plus `file_metadata`
in search.db and `graph_nodes`/`graph_edges` in graph.db, and enqueues a Qdrant
cascade-rename for the `projects`, `rules`, `scratchpad`, and `images` collections.
A path move additionally rewrites the absolute path columns/payloads old→new prefix
(`watch_folders.path`, `unified_queue.file_path`, `file_metadata.file_path`, and the
Qdrant `file_path`/`absolute_path` payload keys). `tracked_files` and the graph
store hold paths relative to the watch root and need no path rewrite.

MCP agents reach the same operation through the `store` tool with `type:"recover"`
(`projectId`, `path` = new path, `rescanRemote`, `dryRun`). The library equivalent
is `LibraryWriteService.RecoverLibrary` (`wqm library recover`), a path-only
re-point since a library's tenant_id is its tag.

---

### LibraryWriteService

Library management mutations. Daemon-exclusive writes to `watch_folders`,
`tracked_files`, `project_components`, and `unified_queue`. Exposes 7 RPCs:
`AddLibrary`, `RemoveLibrary`, `WatchLibrary`, `UnwatchLibrary`,
`ConfigureLibrary`, `SetIncremental`, and `RecoverLibrary`.

#### RecoverLibrary

Re-point a library to a new source path, rewriting stored absolute file paths in
state.db, search.db, and Qdrant (#140). A library's `tenant_id` is its tag, so
re-pointing never changes tenancy — only paths are rewritten. The CLI surface is
`wqm library recover`; see the [CLI reference](./reference/cli.md#wqm-library).

```protobuf
rpc RecoverLibrary(RecoverLibraryRequest) returns (RecoverLibraryResponse);
```

**Request fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tag` | string | Library tag (= tenant_id). Required; must identify a registered library. |
| `new_path` | string (optional) | New absolute source path the library has moved to. Unset → no-op detect (returns `changed=false` if already consistent). |
| `dry_run` | bool | Report planned path change and row/point counts without writing. Default: false. |

**Response fields:**

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | False when the library tag is not found or an internal error occurs. |
| `dry_run` | bool | Echoes the request flag. |
| `changed` | bool | False when the stored path already matches `new_path` (idempotent no-op). |
| `old_path` | string | The previously stored source path. |
| `new_path` | string | The new source path (= `old_path` when `changed=false`). |
| `sqlite_rows_updated` | int32 | Total rows updated (or that would be updated) in state.db + search.db. |
| `qdrant_points_updated` | int32 | Total Qdrant points whose `file_path`/`absolute_path` payload was (or would be) rewritten. |
| `message` | string | Human-readable summary. |

**Semantics and idempotency:**

- When `new_path == old_path`, the RPC returns immediately with `changed=false` and makes no writes. Re-running after a successful apply is therefore safe.
- The path rewrite is a prefix substitution: every stored absolute path that begins with `old_path` is rewritten to begin with `new_path`. Relative paths (`tracked_files.relative_path`, the graph store, Qdrant `relative_path` payload) are watch-root-relative and are not touched.
- `dry_run=true` counts the affected rows/points without writing. The dry-run count for Qdrant uses a synchronous no-write pass through the storage client, so the dry-run and apply counts should agree.
- The satellite search.db is opened on demand from the state.db path. If it does not yet exist, it contributes zero rows and is not created.
- `success=false` is returned (not a gRPC error status) only for invalid arguments (`tag` empty) or a missing library; internal database errors surface as gRPC `Status::internal`.

**Example:**

```python
from workspace_daemon_pb2 import RecoverLibraryRequest
from workspace_daemon_pb2_grpc import LibraryWriteServiceStub

stub = LibraryWriteServiceStub(channel)

# Dry run: check what would change
req = RecoverLibraryRequest(
    tag="rust-docs",
    new_path="/new/location/rust-docs",
    dry_run=True,
)
resp = stub.RecoverLibrary(req)
print(f"Would update {resp.sqlite_rows_updated} SQLite rows and "
      f"{resp.qdrant_points_updated} Qdrant points")

# Apply the re-point
req = RecoverLibraryRequest(
    tag="rust-docs",
    new_path="/new/location/rust-docs",
    dry_run=False,
)
resp = stub.RecoverLibrary(req)
assert resp.success and resp.changed
print(resp.message)
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
- `RulesPayload`: Documents in `rules` collection (rule_id, rule_type, priority, scope, etc.)

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
