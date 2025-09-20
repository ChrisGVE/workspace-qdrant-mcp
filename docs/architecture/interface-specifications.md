# Interface Specifications

**Document Version**: 1.0
**Date**: 2025-09-21
**PRD Alignment**: v3.0 Four-Component Architecture
**Task**: 252.1 - Component Boundaries and Interface Definitions

## Overview

This document defines the formal interface specifications for all communication between components in the workspace-qdrant-mcp four-component architecture. Each interface is designed for clean separation of concerns, high performance, and independent component development.

## Interface Categories

1. **gRPC Interface**: MCP Server ↔ Rust Daemon
2. **CLI Interface**: User ↔ CLI Utility
3. **Hook Interface**: System Events ↔ Context Injector
4. **SQLite Interface**: Shared Database Access
5. **Configuration Interface**: System Configuration Management

---

## 1. gRPC Interface (MCP Server ↔ Rust Daemon)

### Service Definition

```protobuf
syntax = "proto3";

package workspace_qdrant.v1;

service WorkspaceQdrantService {
  // Document Operations
  rpc SearchDocuments(SearchRequest) returns (SearchResponse);
  rpc IngestDocument(IngestRequest) returns (IngestResponse);
  rpc DeleteDocument(DeleteRequest) returns (DeleteResponse);
  rpc GetDocument(GetDocumentRequest) returns (GetDocumentResponse);

  // Collection Management
  rpc CreateCollection(CreateCollectionRequest) returns (CreateCollectionResponse);
  rpc DeleteCollection(DeleteCollectionRequest) returns (DeleteCollectionResponse);
  rpc GetCollectionStatus(StatusRequest) returns (StatusResponse);
  rpc ListCollections(ListCollectionsRequest) returns (ListCollectionsResponse);

  // Memory and Rules
  rpc UpdateMemoryRules(RulesRequest) returns (RulesResponse);
  rpc GetMemoryRules(GetRulesRequest) returns (GetRulesResponse);
  rpc InjectContextRules(ContextRequest) returns (ContextResponse);

  // System Operations
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetMetrics(MetricsRequest) returns (MetricsResponse);
  rpc GetLSPStatus(LSPStatusRequest) returns (LSPStatusResponse);
}
```

### Message Definitions

#### Search Operations
```protobuf
message SearchRequest {
  string query = 1;
  string collection_name = 2;
  SearchType search_type = 3;
  int32 limit = 4;
  float threshold = 5;
  repeated string filters = 6;
  bool include_metadata = 7;
}

enum SearchType {
  SEMANTIC = 0;
  KEYWORD = 1;
  HYBRID = 2;
}

message SearchResponse {
  repeated SearchResult results = 1;
  SearchMetadata metadata = 2;
  string error = 3;
}

message SearchResult {
  string document_id = 1;
  string content = 2;
  float score = 3;
  map<string, string> metadata = 4;
  string file_path = 5;
}
```

#### Document Operations
```protobuf
message IngestRequest {
  string file_path = 1;
  string collection_name = 2;
  string content = 3;
  map<string, string> metadata = 4;
  bool force_reprocess = 5;
  LSPMetadata lsp_metadata = 6;
}

message IngestResponse {
  string document_id = 1;
  bool success = 2;
  string error = 3;
  IngestMetadata metadata = 4;
}

message LSPMetadata {
  string language = 1;
  repeated Symbol symbols = 2;
  repeated Diagnostic diagnostics = 3;
  map<string, string> properties = 4;
}
```

#### Memory Rules
```protobuf
message RulesRequest {
  string rule_id = 1;
  string content = 2;
  AuthorityLevel authority = 3;
  string project_context = 4;
  RuleOperation operation = 5;
}

enum AuthorityLevel {
  DEFAULT = 0;
  ABSOLUTE = 1;
}

enum RuleOperation {
  CREATE = 0;
  UPDATE = 1;
  DELETE = 2;
}

message RulesResponse {
  bool success = 1;
  string rule_id = 2;
  string error = 3;
  ConflictResolution conflicts = 4;
}
```

### Performance Specifications

| Operation | Target Latency | Timeout | Retry Policy |
|-----------|---------------|---------|--------------|
| SearchDocuments | <50ms | 5s | 3 retries, exponential backoff |
| IngestDocument | <100ms | 30s | 2 retries, linear backoff |
| GetHealth | <10ms | 1s | No retries |
| UpdateMemoryRules | <50ms | 10s | 2 retries, exponential backoff |

### Error Handling

```protobuf
enum ErrorCode {
  OK = 0;
  INVALID_REQUEST = 1;
  COLLECTION_NOT_FOUND = 2;
  DOCUMENT_NOT_FOUND = 3;
  PROCESSING_ERROR = 4;
  TIMEOUT = 5;
  INTERNAL_ERROR = 6;
  AUTHENTICATION_ERROR = 7;
}

message ErrorDetail {
  ErrorCode code = 1;
  string message = 2;
  string details = 3;
  string request_id = 4;
}
```

---

## 2. CLI Interface (User ↔ CLI Utility)

### Command Structure

```bash
wqm <domain> <action> [options] [arguments]
```

### Domain Commands

#### Daemon Management
```bash
# Lifecycle operations
wqm daemon start [--config=path] [--background]
wqm daemon stop [--force] [--timeout=30s]
wqm daemon restart [--graceful] [--timeout=60s]
wqm daemon status [--verbose] [--json]

# Configuration operations
wqm daemon config get [key]
wqm daemon config set <key> <value>
wqm daemon config validate [--file=path]
wqm daemon config reset [--confirm]
```

#### Collection Management
```bash
# Collection operations
wqm collections list [--format=table|json] [--filter=pattern]
wqm collections create <name> [--vector-size=384] [--distance=cosine]
wqm collections delete <name> [--force] [--backup]
wqm collections stats <name> [--detailed]

# Collection maintenance
wqm collections optimize <name> [--vacuum] [--reindex]
wqm collections backup <name> [--output=path]
wqm collections restore <name> <backup-path> [--force]
```

#### Health and Diagnostics
```bash
# System health
wqm health check [--component=daemon|qdrant|mcp] [--verbose]
wqm health monitor [--interval=30s] [--duration=5m]
wqm health benchmark [--collection=name] [--operations=1000]

# Performance monitoring
wqm metrics get [--format=json] [--interval=1s] [--count=10]
wqm metrics export [--output=path] [--format=csv|json]
wqm logs tail [--component=all|daemon|cli] [--lines=100]
```

#### Project Management
```bash
# Project operations
wqm project init [--name=project] [--config=template]
wqm project status [--verbose] [--collections]
wqm project clean [--temp-files] [--old-logs] [--confirm]

# Memory management
wqm memory rules list [--project=name] [--authority=all|absolute|default]
wqm memory rules add <content> [--authority=absolute] [--project=name]
wqm memory rules delete <rule-id> [--confirm]
```

### CLI Response Formats

#### Status Output (JSON)
```json
{
  "daemon": {
    "status": "running",
    "pid": 1234,
    "uptime": "2h 15m 30s",
    "memory_usage": "245MB",
    "cpu_usage": "2.3%"
  },
  "qdrant": {
    "status": "connected",
    "url": "http://localhost:6333",
    "collections": 5,
    "total_points": 15420
  },
  "collections": [
    {
      "name": "project-main",
      "points": 1250,
      "vectors": 1250,
      "disk_usage": "15.2MB"
    }
  ]
}
```

#### Error Response Format
```json
{
  "error": {
    "code": "DAEMON_NOT_RUNNING",
    "message": "Daemon process is not running",
    "details": "No process found with expected PID",
    "suggestions": ["Run 'wqm daemon start' to start the daemon"]
  }
}
```

### Exit Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 0 | Success | Operation completed successfully |
| 1 | General error | Unknown or unspecified error |
| 2 | Invalid arguments | Command syntax or argument error |
| 3 | Configuration error | Configuration file or setting error |
| 4 | Connection error | Cannot connect to daemon or Qdrant |
| 5 | Permission error | Insufficient permissions for operation |
| 6 | Resource error | Resource unavailable (disk, memory, etc.) |

---

## 3. Hook Interface (System Events ↔ Context Injector)

### Event Types

```python
@dataclass
class HookEvent:
    event_type: str
    timestamp: datetime
    source_component: str
    payload: Dict[str, Any]
    session_id: Optional[str] = None
    project_context: Optional[str] = None

# Event types
SESSION_INIT = "session.init"
PROJECT_CHANGE = "project.change"
RULE_UPDATE = "rule.update"
CONFIG_CHANGE = "config.change"
USER_PROMPT = "user.prompt"
```

### Hook Registration
```python
class HookRegistry:
    def register_hook(self, event_type: str, callback: Callable[[HookEvent], None]) -> str:
        """Register a hook callback for specific event type"""

    def unregister_hook(self, hook_id: str) -> bool:
        """Unregister a hook by ID"""

    def trigger_event(self, event: HookEvent) -> List[str]:
        """Trigger all hooks for an event type"""
```

### Context Injection Interface
```python
class ContextInjector:
    def inject_rules(self, session_id: str, project_context: str) -> ContextData:
        """Inject behavioral rules into LLM context"""

    def update_context(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update context for active session"""

    def get_session_context(self, session_id: str) -> Optional[ContextData]:
        """Retrieve current context for session"""

@dataclass
class ContextData:
    rules: List[Rule]
    project_context: str
    session_metadata: Dict[str, Any]
    injection_timestamp: datetime
    authority_conflicts: List[ConflictResolution]
```

---

## 4. SQLite Interface (Shared Database Access)

### Database Schema

#### Core Tables
```sql
-- Component state and coordination
CREATE TABLE component_state (
    component_name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    pid INTEGER,
    last_heartbeat TIMESTAMP,
    configuration BLOB,
    metadata JSON
);

-- Collection metadata
CREATE TABLE collections (
    name TEXT PRIMARY KEY,
    vector_size INTEGER NOT NULL,
    distance_metric TEXT NOT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    point_count INTEGER DEFAULT 0,
    metadata JSON
);

-- Document tracking
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    collection_name TEXT NOT NULL,
    hash TEXT NOT NULL,
    processed_at TIMESTAMP,
    lsp_metadata JSON,
    metadata JSON,
    FOREIGN KEY (collection_name) REFERENCES collections(name)
);

-- Memory rules
CREATE TABLE memory_rules (
    rule_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    authority_level TEXT NOT NULL,
    project_context TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    metadata JSON
);
```

### Access Patterns

#### Rust Daemon Access
```rust
pub struct DaemonDatabase {
    connection: Connection,
}

impl DaemonDatabase {
    pub fn update_component_status(&self, status: ComponentStatus) -> Result<()>;
    pub fn log_document_processing(&self, doc: ProcessedDocument) -> Result<()>;
    pub fn get_collection_metadata(&self, name: &str) -> Result<CollectionMeta>;
    pub fn acquire_lock(&self, resource: &str) -> Result<LockGuard>;
}
```

#### CLI Utility Access
```python
class AdminDatabase:
    def get_daemon_status(self) -> ComponentStatus:
        """Get current daemon status and metadata"""

    def update_configuration(self, component: str, config: Dict) -> bool:
        """Update component configuration"""

    def get_collection_stats(self) -> List[CollectionStats]:
        """Get statistics for all collections"""

    def cleanup_stale_locks(self) -> int:
        """Remove expired locks and clean up state"""
```

### Transaction Patterns
```sql
-- Safe configuration update
BEGIN TRANSACTION;
UPDATE component_state
SET configuration = ?, updated_at = CURRENT_TIMESTAMP
WHERE component_name = ?;
INSERT INTO configuration_log (component, change, timestamp)
VALUES (?, ?, CURRENT_TIMESTAMP);
COMMIT;

-- Document processing coordination
BEGIN TRANSACTION;
INSERT OR REPLACE INTO documents (id, file_path, collection_name, hash, processed_at)
VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP);
UPDATE collections SET point_count = point_count + 1, updated_at = CURRENT_TIMESTAMP
WHERE name = ?;
COMMIT;
```

---

## 5. Configuration Interface

### Configuration Hierarchy
```yaml
# System-wide configuration
system:
  components:
    rust_daemon:
      enabled: true
      config_path: "daemon.yaml"
    python_mcp_server:
      enabled: true
      config_path: "mcp_server.yaml"
    cli_utility:
      enabled: true
      config_path: "cli.yaml"
    context_injector:
      enabled: true
      config_path: "injector.yaml"

# Component-specific configurations loaded from separate files
```

### Configuration Validation
```python
class ConfigValidator:
    def validate_component_config(self, component: str, config: Dict) -> ValidationResult:
        """Validate component-specific configuration"""

    def validate_system_config(self, config: Dict) -> ValidationResult:
        """Validate system-wide configuration"""

    def check_config_compatibility(self) -> CompatibilityReport:
        """Check compatibility between component configurations"""

@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
```

### Configuration Updates
```python
class ConfigManager:
    def update_config(self, component: str, updates: Dict, validate: bool = True) -> bool:
        """Update configuration with validation"""

    def reload_config(self, component: str) -> bool:
        """Reload configuration for component"""

    def backup_config(self, output_path: str) -> bool:
        """Create configuration backup"""

    def restore_config(self, backup_path: str, confirm: bool = False) -> bool:
        """Restore configuration from backup"""
```

## Interface Testing and Validation

### gRPC Interface Testing
```python
async def test_grpc_interface():
    """Test gRPC communication between MCP server and daemon"""
    client = WorkspaceQdrantServiceClient("localhost:50051")

    # Test search operation
    request = SearchRequest(query="test", collection_name="test-collection")
    response = await client.SearchDocuments(request)
    assert response.results is not None

    # Test performance requirements
    start_time = time.time()
    response = await client.SearchDocuments(request)
    latency = time.time() - start_time
    assert latency < 0.05  # <50ms requirement
```

### CLI Interface Testing
```bash
#!/bin/bash
# Test CLI interface functionality

# Test daemon management
wqm daemon status --json > status.json
assert_json_field "daemon.status" "running" status.json

# Test performance requirements
start_time=$(date +%s%N)
wqm collections list > /dev/null
end_time=$(date +%s%N)
latency=$(((end_time - start_time) / 1000000))  # Convert to ms
assert_less_than $latency 500  # <500ms requirement
```

### Hook Interface Testing
```python
def test_hook_interface():
    """Test hook system responsiveness and reliability"""
    injector = ContextInjector()

    # Test hook registration
    hook_id = injector.register_hook("session.init", mock_callback)
    assert hook_id is not None

    # Test event triggering
    event = HookEvent("session.init", datetime.now(), "test", {})
    results = injector.trigger_event(event)
    assert len(results) > 0

    # Test performance requirements
    start_time = time.time()
    injector.inject_rules("test-session", "test-project")
    latency = time.time() - start_time
    assert latency < 0.05  # <50ms requirement
```

This interface specification serves as the authoritative reference for all inter-component communication in the workspace-qdrant-mcp system and ensures clean, performant, and reliable component interactions.