# Component Boundaries and Responsibilities

**Document Version**: 1.0
**Date**: 2025-09-21
**PRD Alignment**: v3.0 Four-Component Architecture
**Task**: 252.1 - Define Component Boundaries and Responsibilities

## Overview

This document establishes the formal component boundaries and responsibilities for the workspace-qdrant-mcp four-component architecture as specified in PRD v3.0. It defines clear separation of concerns, interfaces, and interaction patterns between all system components.

## Four-Component Architecture

### Component 1: Rust Daemon (Heavy Processing Engine)

**Primary Role**: High-Performance Processing Powerhouse

**Core Responsibilities**:
- **File Ingestion**: Process 1000+ documents per minute with high-throughput parsing
- **LSP Integration**: Manage LSP server lifecycle, communication, and metadata extraction
- **Document Conversion**: Handle text extraction from various file formats (PDF, DOCX, etc.)
- **Embedding Generation**: Generate semantic embeddings using FastEmbed models
- **File System Watching**: Monitor file changes with platform-optimized watchers
- **State Management**: Maintain SQLite database for operational state and metadata
- **Vector Storage**: Interface with Qdrant for vector storage and retrieval

**Performance Requirements**:
- Memory usage: <500MB sustained operation
- Processing throughput: 1000+ documents/minute
- Startup time: <2 seconds for daemon initialization
- gRPC response latency: <50ms for standard operations

**Isolation Boundaries**:
- Operates as independent system daemon
- No direct Python dependencies or imports
- Communicates exclusively through gRPC and SQLite
- Manages own configuration and logging

**Location**: `/src/rust/daemon/`

---

### Component 2: Python MCP Server (Intelligent Interface)

**Primary Role**: Intelligent Interface Layer

**Core Responsibilities**:
- **MCP Protocol**: Implement complete MCP protocol for Claude Code integration
- **Search Interface**: Provide hybrid search (semantic + keyword) with query optimization
- **Memory Management**: Handle conversational memory and rule injection
- **Session Management**: Maintain Claude Code session state and context
- **Query Intelligence**: Optimize search queries and result ranking
- **gRPC Client**: Communicate with Rust daemon for heavy operations
- **Context Orchestration**: Coordinate context injection and rule application

**Performance Requirements**:
- Query response time: <100ms for standard searches
- Session initialization: <2 seconds including rule injection
- Memory efficiency: Stateless operation with minimal session data
- Concurrent sessions: Support multiple simultaneous Claude Code sessions

**Isolation Boundaries**:
- Operates as MCP server process (stdio/HTTP)
- Stateless operation with external state management
- No direct file system access beyond configuration
- Communicates with daemon exclusively through gRPC

**Location**: `/src/python/workspace_qdrant_mcp/`

---

### Component 3: CLI Utility (User Control & Administration)

**Primary Role**: User Control and Administration

**Core Responsibilities**:
- **System Administration**: Complete system control and configuration management
- **Daemon Lifecycle**: Start, stop, restart, status monitoring of Rust daemon
- **Collection Management**: Create, delete, maintain Qdrant collections
- **Configuration Management**: User settings, system configuration, validation
- **Performance Monitoring**: System diagnostics, health checks, metrics
- **Library Management**: Dependency management and updates
- **Independent Operation**: Function without requiring MCP server or daemon

**Performance Requirements**:
- Immediate response: <500ms for all CLI operations
- Low overhead: Minimal system resource consumption
- Reliable operation: Graceful handling of daemon unavailability
- Clear feedback: Comprehensive status and error reporting

**Isolation Boundaries**:
- Independent CLI application
- Direct SQLite access for administration
- No dependency on MCP server or daemon for core operations
- Own configuration and logging system

**Location**: `/src/rust/cli/` (Rust CLI - `wqm` binary)

---

### Component 4: Context Injector (LLM Integration/Hook)

**Primary Role**: LLM Context Injection and Rule Management

**Core Responsibilities**:
- **Rule Injection**: Fetch and inject behavioral rules into LLM context
- **Memory Integration**: Access dedicated memory collection for rules
- **Hook System**: Respond to system events and triggers for context updates
- **Session Context**: Manage session-specific context and rule application
- **Conflict Resolution**: Handle conflicting rules and authority levels
- **Streaming Interface**: Provide simple text streaming to LLM context
- **Event-Driven Operation**: Trigger context updates based on system state

**Performance Requirements**:
- Context injection latency: <50ms for rule retrieval and formatting
- Memory efficiency: Minimal memory footprint with rule caching
- Event response time: <100ms for hook-triggered operations
- Session isolation: Clean separation between different LLM sessions

**Isolation Boundaries**:
- Lightweight component with minimal dependencies
- Read-only access to memory collection
- Event-driven activation only
- No direct user interface or administrative functions

**Location**: `/src/python/context_injector/` (TO BE CREATED)

## Component Communication Matrix

| Source Component | Target Component | Communication Method | Purpose | Performance |
|-----------------|------------------|---------------------|---------|-------------|
| MCP Server | Rust Daemon | gRPC Client → Server | Search, operations, state queries | <50ms |
| CLI Utility | Rust Daemon | Signal handling, SQLite | Administration, lifecycle management | <100ms |
| CLI Utility | SQLite Database | Direct access | Configuration, state management | <10ms |
| Context Injector | Memory Collection | Direct Qdrant access | Rule retrieval, context building | <50ms |
| Context Injector | LLM Context | Hook/streaming API | Rule injection, context updates | <20ms |
| Rust Daemon | SQLite Database | Direct access | State persistence, metadata storage | <5ms |
| Rust Daemon | Qdrant | HTTP/gRPC client | Vector storage, search operations | <100ms |

## Data Flow Patterns

### Primary Search Flow
```
Claude Code → MCP Server → gRPC → Rust Daemon → Qdrant → Vector Results → gRPC → MCP Server → Claude Code
```

### File Processing Flow
```
File System Events → Rust Daemon → LSP Processing → Embedding Generation → Qdrant Storage → SQLite State Update
```

### Administrative Flow
```
CLI Commands → SQLite Direct Access → Daemon Signal Handling → Operation Execution → Status Response
```

### Context Injection Flow
```
Session Init/Hook → Context Injector → Memory Collection → Rule Retrieval → LLM Context → Rule Application
```

## Shared Resource Access Patterns

### SQLite Database
- **Primary Access**: Rust Daemon (read/write for operational state)
- **Administrative Access**: CLI Utility (read/write for configuration)
- **Coordination**: File-based locking, transaction boundaries
- **Schema**: Shared schema with component-specific tables

### Qdrant Vector Database
- **Primary Access**: Rust Daemon (all vector operations)
- **Read-Only Access**: Context Injector (memory collection only)
- **Coordination**: Collection-based isolation, no shared collections
- **Authentication**: Shared credentials, component-specific collections

### Configuration Files
- **Ownership**: Each component owns its configuration
- **Shared Config**: System-wide settings in separate files
- **Validation**: CLI utility validates all configurations
- **Updates**: Coordinated through CLI utility only

### Log Files
- **Isolation**: Component-specific log files
- **Centralization**: Optional centralized logging through CLI utility
- **Rotation**: Independent log rotation per component
- **Aggregation**: CLI utility provides log aggregation tools

## Interface Specifications

### gRPC Interface (MCP Server ↔ Rust Daemon)

**Service Definition**: `WorkspaceQdrantService`

**Key Methods**:
- `SearchDocuments(SearchRequest) → SearchResponse`
- `IngestDocument(IngestRequest) → IngestResponse`
- `GetCollectionStatus(StatusRequest) → StatusResponse`
- `UpdateMemoryRules(RulesRequest) → RulesResponse`

### CLI Interface (User ↔ CLI Utility)

**Command Structure**: `wqm <domain> <action> [options]`

**Key Domains**:
- `daemon`: Lifecycle management (`start`, `stop`, `status`, `restart`)
- `collections`: Collection management (`list`, `create`, `delete`, `stats`)
- `config`: Configuration management (`get`, `set`, `validate`, `reset`)
- `health`: System diagnostics (`check`, `monitor`, `benchmark`)

### Hook Interface (System ↔ Context Injector)

**Trigger Events**:
- Session initialization
- Project context changes
- Rule updates in memory collection
- Configuration changes

**Response Interface**:
- Context data streaming
- Rule formatting and injection
- Error handling and fallback

## Migration from Current State

### Current Architecture Analysis
- **Existing Components**: Rust daemon (partial), Python MCP server (functional), CLI utility (basic)
- **Missing Components**: Context injector (complete gap)
- **Alignment**: ~35% alignment with PRD v3.0 specification

### Migration Steps
1. **Context Injector Creation**: Implement missing component
2. **Interface Standardization**: Formalize gRPC and CLI interfaces
3. **Boundary Enforcement**: Remove cross-component dependencies
4. **Resource Isolation**: Implement shared resource access patterns
5. **Documentation Update**: Align with formal specifications

### Success Criteria
- [ ] All four components operational with defined boundaries
- [ ] Clean interface specifications implemented
- [ ] No cross-component tight coupling
- [ ] Shared resources accessed through defined patterns
- [ ] Performance requirements met for each component
- [ ] Independent development and testing possible

## Validation and Testing

### Component Isolation Testing
- Each component must function independently
- Interface contracts must be verifiable
- Resource access patterns must be testable
- Performance requirements must be measurable

### Integration Testing
- End-to-end workflows must function correctly
- Component communication must be reliable
- Shared resource access must be safe
- Error handling must be robust

This document establishes the foundation for the four-component architecture implementation and serves as the authoritative reference for component boundaries and responsibilities.