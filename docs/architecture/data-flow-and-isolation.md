# Data Flow and Isolation Boundaries

**Document Version**: 1.0
**Date**: 2025-09-21
**PRD Alignment**: v3.0 Four-Component Architecture
**Task**: 252.1 - Define Data Flow and Component Isolation

## Overview

This document defines the data flow patterns and isolation boundaries for the workspace-qdrant-mcp four-component architecture. It establishes how information moves through the system and ensures clean separation between components while maintaining high performance and reliability.

## System Data Flow Architecture

### Primary Data Flows

#### 1. Search and Query Flow
```
┌─────────────┐    MCP Protocol    ┌─────────────┐    gRPC    ┌─────────────┐    Qdrant API    ┌─────────────┐
│ Claude Code │ ─────────────────→ │ MCP Server  │ ─────────→ │ Rust Daemon │ ────────────────→ │   Qdrant    │
│             │                    │  (Python)   │            │             │                   │  Database   │
│             │ ←───────────────── │             │ ←───────── │             │ ←──────────────── │             │
└─────────────┘    Search Results  └─────────────┘  Results   └─────────────┘    Vector Data    └─────────────┘
```

**Flow Details**:
1. Claude Code sends search query via MCP protocol
2. MCP Server validates and optimizes query
3. MCP Server forwards request to Rust Daemon via gRPC
4. Rust Daemon performs vector search in Qdrant
5. Results flow back through same path with ranking and metadata

**Performance**: End-to-end latency <150ms (MCP: <50ms, gRPC: <50ms, Qdrant: <50ms)

#### 2. Document Ingestion Flow
```
┌─────────────┐    File Events    ┌─────────────┐    LSP Processing    ┌─────────────┐    Vector Storage    ┌─────────────┐
│ File System │ ─────────────────→ │ Rust Daemon │ ───────────────────→ │ LSP Servers │                     │   Qdrant    │
│   Changes   │                    │  (Watcher)  │                      │             │                     │  Database   │
└─────────────┘                    │             │ ←─────────────────── │             │                     │             │
                                   │             │    Metadata          └─────────────┘                     │             │
┌─────────────┐    Embeddings     │             │                                                           │             │
│  FastEmbed  │ ←─────────────────│             │ ─────────────────────────────────────────────────────────→ │             │
│   Models    │                    │             │    Document + Vectors + Metadata                          │             │
└─────────────┘                    └─────────────┘                                                           └─────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │   SQLite    │
                                   │ State DB    │
                                   └─────────────┘
```

**Flow Details**:
1. File system events trigger Rust Daemon watcher
2. Daemon processes file through appropriate LSP server
3. LSP metadata extracted and enriched
4. Document content processed through FastEmbed for embeddings
5. Combined data (content + vectors + metadata) stored in Qdrant
6. Processing state and metadata recorded in SQLite

**Performance**: 1000+ documents/minute sustained throughput

#### 3. Administrative Control Flow
```
┌─────────────┐    CLI Commands    ┌─────────────┐    Direct SQLite    ┌─────────────┐    Process Signals    ┌─────────────┐
│    User     │ ─────────────────→ │ CLI Utility │ ───────────────────→ │   SQLite    │                      │ Rust Daemon │
│             │                    │             │                      │ State DB    │                      │             │
│             │ ←───────────────── │             │ ←─────────────────── │             │ ←──────────────────── │             │
└─────────────┘    Status/Results  └─────────────┘    State/Config     └─────────────┘    Status Updates    └─────────────┘
```

**Flow Details**:
1. User issues administrative commands via CLI
2. CLI Utility accesses SQLite database directly for state/config
3. CLI sends process signals to Rust Daemon for lifecycle management
4. Daemon updates state and responds with status
5. CLI provides formatted response to user

**Performance**: <500ms for all CLI operations, <100ms for status queries

#### 4. Context Injection Flow
```
┌─────────────┐    Session Init    ┌─────────────┐    Rule Retrieval    ┌─────────────┐    Memory Access    ┌─────────────┐
│ Claude Code │ ─────────────────→ │   Context   │ ───────────────────→ │   Qdrant    │                    │ Memory      │
│  Session    │                    │  Injector   │                      │  Database   │ ─────────────────→ │ Collection  │
│             │ ←───────────────── │             │ ←─────────────────── │             │   Rules Query      │             │
└─────────────┘   Injected Rules   └─────────────┘    Rules Data       └─────────────┘                    └─────────────┘
                                          │
                                          ▼ Hook Triggers
                                   ┌─────────────┐
                                   │ System      │
                                   │ Events      │
                                   └─────────────┘
```

**Flow Details**:
1. Session initialization or system events trigger Context Injector
2. Injector queries memory collection for applicable rules
3. Rules processed for conflicts and authority levels
4. Formatted context data injected into Claude Code session
5. Hook system monitors for updates and re-injection triggers

**Performance**: <50ms for rule retrieval and formatting, <20ms for injection

### Cross-Component Data Synchronization

#### State Consistency Model
```
┌─────────────┐    State Updates    ┌─────────────┐    Read Operations    ┌─────────────┐
│ Rust Daemon │ ───────────────────→ │   SQLite    │ ───────────────────→ │ CLI Utility │
│             │    (Primary Writer) │ State DB    │   (Administrative)   │             │
└─────────────┘                     │             │                      └─────────────┘
                                    │             │
                                    │             │    Read Operations    ┌─────────────┐
                                    │             │ ───────────────────→ │   Context   │
                                    └─────────────┘     (Rules Only)     │  Injector   │
                                                                         └─────────────┘
```

**Consistency Rules**:
- **Single Writer**: Only Rust Daemon writes operational state
- **Administrative Override**: CLI can update configuration and force state changes
- **Read-Only Access**: Context Injector only reads memory collection data
- **Atomic Updates**: All state changes use SQLite transactions
- **Event Notification**: State changes trigger appropriate component notifications

## Component Isolation Boundaries

### Rust Daemon Isolation

**Process Boundaries**:
- Runs as independent system process/service
- Own memory space and resource limits
- No direct Python dependencies or imports
- Platform-specific process management (systemd, launchd, Windows Service)

**Communication Boundaries**:
- **Inbound**: gRPC server, process signals, SQLite database access
- **Outbound**: Qdrant HTTP/gRPC client, LSP protocol clients, file system access
- **Forbidden**: Direct Python module imports, MCP protocol, CLI command execution

**Resource Boundaries**:
- Memory limit: 500MB sustained operation
- CPU priority: Can be elevated during MCP server sessions
- File access: Read/write to configured workspace directories only
- Network access: Qdrant server and LSP server connections only

**Security Boundaries**:
- Runs with minimal required permissions
- No user credential access beyond Qdrant authentication
- Sandboxed file system access to workspace directories
- No network access beyond configured endpoints

### Python MCP Server Isolation

**Process Boundaries**:
- Runs as MCP server process (stdio or HTTP)
- Stateless operation with external state management
- No long-term data persistence beyond session state
- Managed lifecycle by Claude Code or MCP client

**Communication Boundaries**:
- **Inbound**: MCP protocol (stdio/HTTP), configuration file access
- **Outbound**: gRPC client to Rust Daemon, basic logging
- **Forbidden**: Direct file system access, SQLite database access, Qdrant direct access

**Resource Boundaries**:
- Memory: Minimal footprint with session-only state
- CPU: Low utilization with fast query processing
- File access: Configuration files and logs only
- Network access: gRPC to localhost Rust Daemon only

**Security Boundaries**:
- No direct database access
- No file system traversal beyond configuration
- Authentication delegated to Rust Daemon
- Session isolation between concurrent Claude Code instances

### CLI Utility Isolation

**Process Boundaries**:
- Runs as user command-line application
- Independent operation without daemon dependency for core functions
- Short-lived processes for command execution
- No persistent state beyond configuration

**Communication Boundaries**:
- **Inbound**: Command-line arguments, configuration files
- **Outbound**: SQLite database access, process signals, standard output
- **Forbidden**: gRPC communication, MCP protocol, direct Qdrant access

**Resource Boundaries**:
- Memory: Minimal footprint for command execution
- CPU: Burst usage for administrative operations
- File access: Configuration files, log files, SQLite database
- Network access: None (all operations through local resources)

**Security Boundaries**:
- User permission inheritance for file operations
- Administrative database access with transaction safety
- Process management permissions for daemon control
- No network communication or external service access

### Context Injector Isolation

**Process Boundaries**:
- Lightweight component with minimal dependencies
- Event-driven activation only (no persistent process)
- Can be embedded in MCP Server or run independently
- Stateless operation with external configuration

**Communication Boundaries**:
- **Inbound**: Hook events, session initialization triggers
- **Outbound**: Memory collection queries, LLM context injection
- **Forbidden**: File system access, daemon communication, administrative operations

**Resource Boundaries**:
- Memory: Minimal footprint with rule caching
- CPU: Low utilization with event-driven operation
- File access: None (configuration through environment/parameters)
- Network access: Read-only Qdrant access for memory collection only

**Security Boundaries**:
- Read-only access to memory collection
- No administrative privileges or system access
- Session-specific context isolation
- No persistent data storage or state management

## Shared Resource Access Patterns

### SQLite Database Coordination

#### Lock Management
```sql
-- Advisory locking for coordination
CREATE TABLE resource_locks (
    resource_name TEXT PRIMARY KEY,
    holder_component TEXT NOT NULL,
    holder_pid INTEGER NOT NULL,
    acquired_at TIMESTAMP NOT NULL,
    expires_at TIMESTAMP,
    lock_type TEXT DEFAULT 'exclusive'
);

-- Automatic cleanup of stale locks
CREATE TRIGGER cleanup_stale_locks
AFTER INSERT ON component_heartbeat
BEGIN
    DELETE FROM resource_locks
    WHERE expires_at < CURRENT_TIMESTAMP
    OR (holder_pid NOT IN (SELECT pid FROM component_state WHERE status = 'running'));
END;
```

#### Transaction Patterns
```sql
-- Component state update pattern
BEGIN IMMEDIATE TRANSACTION;
INSERT OR REPLACE INTO component_heartbeat (component, pid, timestamp)
VALUES (?, ?, CURRENT_TIMESTAMP);
UPDATE component_state SET status = ?, last_heartbeat = CURRENT_TIMESTAMP
WHERE component_name = ?;
COMMIT;

-- Configuration update pattern (CLI only)
BEGIN EXCLUSIVE TRANSACTION;
INSERT INTO configuration_backup (component, config, timestamp)
SELECT component_name, configuration, CURRENT_TIMESTAMP
FROM component_state WHERE component_name = ?;
UPDATE component_state SET configuration = ?, updated_at = CURRENT_TIMESTAMP
WHERE component_name = ?;
COMMIT;
```

### Qdrant Collection Isolation

#### Collection Naming Convention
```
Format: {project}_{component}_{type}
Examples:
- myproject_daemon_documents    (Main document storage)
- myproject_memory_rules        (Behavioral rules)
- myproject_scratch_notes       (Temporary scratchbook)
- global_system_config         (System-wide settings)
```

#### Access Control Matrix
| Component | Document Collections | Memory Collections | System Collections | Admin Operations |
|-----------|--------------------|--------------------|-------------------|------------------|
| Rust Daemon | Read/Write | Read/Write | Read | Limited |
| MCP Server | Read via gRPC | Read via gRPC | Read via gRPC | None |
| CLI Utility | Admin only | Admin only | Read/Write | Full |
| Context Injector | None | Read only | None | None |

### Configuration File Management

#### File Ownership
```
/config/
├── system.yaml           # System-wide (CLI manages)
├── daemon.yaml          # Rust Daemon (CLI validates)
├── mcp_server.yaml      # MCP Server (CLI validates)
├── cli.yaml             # CLI Utility (CLI owns)
└── injector.yaml        # Context Injector (CLI validates)
```

#### Update Coordination
1. **CLI Utility**: Validates all configuration changes
2. **Component Reload**: Signals sent for configuration reload
3. **Version Control**: Configuration changes logged with timestamps
4. **Rollback Support**: Previous configurations backed up automatically

### Log File Coordination

#### Log File Structure
```
/logs/
├── system/
│   ├── daemon.log          # Rust Daemon operational logs
│   ├── mcp_server.log      # MCP Server session logs
│   ├── cli.log             # CLI Utility operation logs
│   └── injector.log        # Context Injector event logs
├── performance/
│   ├── search_metrics.log  # Search performance tracking
│   ├── ingestion_metrics.log # Document processing metrics
│   └── system_metrics.log  # System resource utilization
└── audit/
    ├── configuration.log   # Configuration change audit
    ├── access.log         # Component access audit
    └── errors.log         # System error aggregation
```

#### Log Aggregation
- **CLI Utility**: Provides log aggregation and analysis tools
- **Centralized Errors**: All components report errors to centralized log
- **Performance Metrics**: Automated collection and analysis
- **Audit Trail**: Administrative operations and configuration changes tracked

## Error Handling and Recovery

### Component Failure Isolation

#### Failure Scenarios and Responses

| Failed Component | Impact | Recovery Strategy | Isolation Mechanism |
|-----------------|--------|-------------------|-------------------|
| Rust Daemon | Search/ingestion unavailable | Auto-restart, queue preservation | MCP Server falls back to cached data |
| MCP Server | Claude Code disconnection | Session restart, state recovery | Daemon continues processing queue |
| CLI Utility | Admin operations unavailable | No system impact | Other components unaffected |
| Context Injector | No rule injection | Graceful degradation | Session continues without rules |

#### Recovery Procedures
```python
class ComponentRecovery:
    def detect_failure(self, component: str) -> FailureType:
        """Detect and classify component failures"""

    def isolate_failure(self, component: str) -> bool:
        """Isolate failed component to prevent cascade"""

    def recover_component(self, component: str, strategy: RecoveryStrategy) -> bool:
        """Execute recovery strategy for failed component"""

    def verify_recovery(self, component: str) -> bool:
        """Verify successful component recovery"""
```

### Data Integrity Protection

#### Transaction Boundaries
- **SQLite**: All state changes use explicit transactions with rollback
- **Qdrant**: Batch operations with error handling and partial rollback
- **Configuration**: Backup before changes with automatic rollback on failure

#### Consistency Checks
```sql
-- Data consistency verification
CREATE VIEW consistency_check AS
SELECT
    d.collection_name,
    COUNT(d.id) as sqlite_count,
    -- Compare with Qdrant point count
    c.point_count as qdrant_count,
    CASE
        WHEN COUNT(d.id) = c.point_count THEN 'consistent'
        ELSE 'inconsistent'
    END as status
FROM documents d
JOIN collections c ON d.collection_name = c.name
GROUP BY d.collection_name;
```

This document establishes the complete data flow architecture and isolation boundaries for the workspace-qdrant-mcp four-component system, ensuring clean separation while maintaining high performance and reliability.