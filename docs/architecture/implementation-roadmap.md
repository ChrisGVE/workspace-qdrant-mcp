# Implementation Roadmap

**Document Version**: 1.0
**Date**: 2025-09-21
**PRD Alignment**: v3.0 Four-Component Architecture
**Task**: 252.1 - Implementation Path for Component Boundaries

## Overview

This document provides a concrete implementation roadmap for migrating from the current hybrid architecture to the target four-component architecture defined in PRD v3.0. It includes current state analysis, gap identification, and step-by-step implementation phases.

## Current State Analysis

### Architecture Assessment (v0.2.x)

**Current Components**:
1. ✅ **Rust Daemon**: Partially implemented (`/src/rust/daemon/`)
   - Core processing: 70% complete
   - LSP integration: 60% complete
   - gRPC server: 80% complete
   - File watching: 85% complete

2. ✅ **Python MCP Server**: Functional (`/src/python/workspace_qdrant_mcp/`)
   - MCP protocol: 90% complete
   - Search interface: 85% complete
   - Memory management: 70% complete
   - gRPC client: 75% complete

3. ✅ **CLI Utility**: Complete implementation (`/src/rust/cli/`)
   - Basic commands: 100% complete
   - Daemon management: 100% complete (wqm service commands)
   - Configuration: 100% complete
   - Administration: 100% complete (wqm admin commands)

4. ❌ **Context Injector**: Missing (0% complete)
   - Component does not exist
   - No hook system implementation
   - No context injection capability
   - No memory rule integration

**Overall Alignment**: ~35% alignment with PRD v3.0 specification

### Current Architectural Issues

#### Boundary Violations
- **MCP Server → File System**: Direct file access instead of gRPC delegation
- **CLI → Qdrant**: Some direct database access instead of daemon coordination
- **Shared Code**: Common modules with tight coupling between components

#### Missing Interfaces
- **Context Injection**: No system for rule injection into LLM context
- **Hook System**: No event-driven architecture for context updates
- **Administrative gRPC**: CLI uses mixed access patterns instead of consistent interface

#### Performance Gaps
- **Search Latency**: Currently ~200ms, target <150ms end-to-end
- **Ingestion Throughput**: Currently ~400 docs/min, target 1000+ docs/min
- **CLI Response**: Currently ~800ms for complex operations, target <500ms

## Gap Analysis and Prioritization

### Critical Gaps (Must Fix)

1. **Context Injector Component** (Priority: Critical)
   - No component exists
   - Required for memory rule injection
   - Blocks session initialization improvements
   - Essential for conversational memory updates

2. **Interface Standardization** (Priority: Critical)
   - Inconsistent communication patterns
   - Missing gRPC service definitions
   - CLI access pattern inconsistencies
   - No formal API contracts

3. **Component Isolation** (Priority: High)
   - Cross-component file access
   - Shared module dependencies
   - Inconsistent resource access patterns
   - No clean separation of concerns

### Important Gaps (Should Fix)

4. **Performance Optimization** (Priority: High)
   - Search query optimization
   - Ingestion pipeline efficiency
   - CLI response time improvement
   - Memory usage optimization

5. **Error Handling** (Priority: Medium)
   - Component failure isolation
   - Recovery procedures
   - Graceful degradation
   - Data consistency protection

6. **Testing Framework** (Priority: Medium)
   - Component isolation testing
   - Interface contract testing
   - Performance regression testing
   - Integration testing automation

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

#### 1.1 Create Context Injector Component
```bash
# Create component structure
mkdir -p /src/python/context_injector/{core,hooks,tests}

# Implement core interfaces
src/python/context_injector/
├── __init__.py           # Component entry point
├── core/
│   ├── injector.py      # Main injection logic
│   ├── rules.py         # Rule processing and conflict resolution
│   ├── session.py       # Session context management
│   └── memory.py        # Memory collection interface
├── hooks/
│   ├── registry.py      # Hook registration and management
│   ├── events.py        # Event type definitions
│   └── triggers.py      # Event triggering mechanisms
└── tests/
    ├── test_injector.py # Component unit tests
    ├── test_hooks.py    # Hook system tests
    └── test_integration.py # Integration tests
```

**Deliverables**:
- [ ] Context Injector component structure created
- [ ] Basic rule injection functionality implemented
- [ ] Hook system for event-driven updates
- [ ] Memory collection integration
- [ ] Unit tests with >80% coverage

#### 1.2 Standardize gRPC Interface
```protobuf
# Update existing gRPC service definition
service WorkspaceQdrantService {
  // Document operations (existing, enhanced)
  rpc SearchDocuments(SearchRequest) returns (SearchResponse);
  rpc IngestDocument(IngestRequest) returns (IngestResponse);

  // Memory operations (new)
  rpc UpdateMemoryRules(RulesRequest) returns (RulesResponse);
  rpc GetMemoryRules(GetRulesRequest) returns (GetRulesResponse);

  // Context operations (new)
  rpc InjectContextRules(ContextRequest) returns (ContextResponse);
  rpc UpdateSessionContext(SessionRequest) returns (SessionResponse);

  // System operations (enhanced)
  rpc GetHealth(HealthRequest) returns (HealthResponse);
  rpc GetComponentStatus(StatusRequest) returns (StatusResponse);
}
```

**Deliverables**:
- [ ] Complete gRPC service definition updated
- [ ] Rust daemon gRPC server implementation
- [ ] Python MCP server gRPC client updates
- [ ] Interface contract tests implemented

#### 1.3 CLI Interface Standardization
```bash
# Standardize CLI command structure
wqm daemon {start|stop|restart|status|config}
wqm collections {list|create|delete|stats|optimize}
wqm health {check|monitor|benchmark}
wqm memory {rules|list|add|delete}
wqm project {init|status|clean}
```

**Deliverables**:
- [ ] Complete CLI command restructure
- [ ] Consistent output formats (JSON/table)
- [ ] Proper exit codes and error handling
- [ ] Command completion and help text

### Phase 2: Isolation (Weeks 3-4)

#### 2.1 Remove Boundary Violations

**MCP Server File Access Removal**:
```python
# Before: Direct file access
with open(file_path, 'r') as f:
    content = f.read()

# After: gRPC delegation
async def get_document_content(self, file_path: str) -> str:
    request = GetDocumentRequest(file_path=file_path)
    response = await self.grpc_client.GetDocument(request)
    return response.content
```

**CLI Qdrant Access Standardization**:
```python
# Before: Direct Qdrant access
from qdrant_client import QdrantClient
client = QdrantClient("localhost", 6333)

# After: SQLite state queries
def get_collection_stats(self, name: str) -> CollectionStats:
    return self.db.query_collection_metadata(name)
```

**Deliverables**:
- [ ] All direct file access removed from MCP Server
- [ ] CLI uses SQLite/gRPC for all operations
- [ ] Shared modules refactored into component-specific code
- [ ] Component dependency analysis passes

#### 2.2 Resource Access Pattern Implementation

**SQLite Access Control**:
```python
class ComponentDatabase:
    """Base class for component-specific database access"""
    def __init__(self, component_name: str, access_level: AccessLevel):
        self.component = component_name
        self.access = access_level

class DaemonDatabase(ComponentDatabase):
    """Full read/write access for operational state"""
    def __init__(self):
        super().__init__("rust_daemon", AccessLevel.READ_WRITE)

class CLIDatabase(ComponentDatabase):
    """Administrative access for configuration"""
    def __init__(self):
        super().__init__("cli_utility", AccessLevel.ADMIN)
```

**Deliverables**:
- [ ] Component-specific database access classes
- [ ] Resource access validation and enforcement
- [ ] Transaction boundary implementation
- [ ] Lock management for coordination

### Phase 3: Performance Optimization (Weeks 5-6)

#### 3.1 Search Pipeline Optimization

**Target Metrics**:
- End-to-end search latency: <150ms (current: ~200ms)
- gRPC communication: <50ms (current: ~75ms)
- Qdrant query execution: <50ms (current: ~80ms)
- Result processing: <50ms (current: ~45ms)

**Implementation**:
```rust
// Optimized search pipeline in Rust daemon
impl SearchOptimizer {
    async fn execute_hybrid_search(&self, request: SearchRequest) -> Result<SearchResponse> {
        // Parallel execution of semantic and keyword search
        let (semantic_task, keyword_task) = tokio::join!(
            self.execute_semantic_search(&request),
            self.execute_keyword_search(&request)
        );

        // Fast result fusion with pre-computed weights
        let results = self.fuse_results(semantic_task?, keyword_task?).await?;
        Ok(SearchResponse { results, metadata: self.build_metadata() })
    }
}
```

**Deliverables**:
- [ ] Parallel search execution implementation
- [ ] Result caching for frequent queries
- [ ] Connection pooling for Qdrant
- [ ] Performance benchmarking and validation

#### 3.2 Ingestion Pipeline Optimization

**Target Metrics**:
- Document processing: 1000+ docs/minute (current: ~400/minute)
- Memory usage: <500MB (current: ~650MB)
- LSP processing efficiency: 2x improvement
- Batch processing capabilities

**Implementation**:
```rust
// High-throughput ingestion pipeline
impl IngestionOptimizer {
    async fn process_document_batch(&self, documents: Vec<Document>) -> Result<BatchResult> {
        // Parallel LSP processing with resource limits
        let lsp_tasks = self.create_lsp_workers(documents, self.config.max_workers).await?;

        // Streaming embedding generation
        let embedding_stream = self.generate_embeddings_stream(lsp_tasks).await?;

        // Batch Qdrant upload with error recovery
        let results = self.batch_upload_to_qdrant(embedding_stream).await?;
        Ok(results)
    }
}
```

**Deliverables**:
- [ ] Batch processing implementation
- [ ] Parallel LSP processing
- [ ] Memory-efficient streaming pipelines
- [ ] Throughput monitoring and optimization

### Phase 4: Integration and Testing (Weeks 7-8)

#### 4.1 Component Integration Testing

**Test Scenarios**:
```python
class ComponentIntegrationTests:
    async def test_full_search_pipeline(self):
        """Test end-to-end search flow through all components"""

    async def test_document_ingestion_pipeline(self):
        """Test complete document processing workflow"""

    async def test_context_injection_flow(self):
        """Test memory rule injection and session management"""

    async def test_administrative_operations(self):
        """Test CLI administrative operations coordination"""

    async def test_component_failure_recovery(self):
        """Test graceful degradation and recovery procedures"""
```

**Performance Validation**:
```python
class PerformanceValidationTests:
    def test_search_latency_requirements(self):
        """Validate <150ms end-to-end search latency"""

    def test_ingestion_throughput_requirements(self):
        """Validate 1000+ documents/minute processing"""

    def test_cli_response_time_requirements(self):
        """Validate <500ms CLI operation response times"""

    def test_memory_usage_requirements(self):
        """Validate <500MB sustained memory usage"""
```

**Deliverables**:
- [ ] Complete integration test suite
- [ ] Performance validation framework
- [ ] Automated testing pipeline
- [ ] Component isolation verification

#### 4.2 Documentation and Migration Guide

**Component Documentation**:
- [ ] API reference documentation for all interfaces
- [ ] Component architecture diagrams
- [ ] Deployment and configuration guides
- [ ] Troubleshooting and maintenance procedures

**Migration Guide**:
- [ ] Step-by-step migration procedures
- [ ] Configuration migration scripts
- [ ] Data migration and validation tools
- [ ] Rollback procedures and safety checks

## Success Criteria and Validation

### Functional Requirements
- [ ] All four components operational with defined boundaries
- [ ] Complete interface specifications implemented
- [ ] No cross-component boundary violations
- [ ] Independent component development and testing capability

### Performance Requirements
- [ ] Search latency: <150ms end-to-end
- [ ] Ingestion throughput: 1000+ documents/minute
- [ ] CLI response time: <500ms for all operations
- [ ] Memory usage: <500MB sustained operation

### Quality Requirements
- [ ] Component test coverage: >80% for all components
- [ ] Integration test coverage: >90% for critical workflows
- [ ] Performance regression testing: Automated and validated
- [ ] Documentation completeness: 100% API coverage

### Operational Requirements
- [ ] Independent component deployment capability
- [ ] Graceful failure handling and recovery
- [ ] Administrative tools for monitoring and maintenance
- [ ] Configuration management and validation

## Risk Mitigation

### Technical Risks
- **Component Communication**: gRPC interface complexity → Incremental implementation and testing
- **Performance Impact**: Overhead from proper isolation → Parallel optimization during refactoring
- **Data Consistency**: Multi-component coordination → Strong transaction boundaries and testing

### Implementation Risks
- **Timeline Pressure**: Complex refactoring scope → Phased approach with working system at each phase
- **Resource Constraints**: Development capacity → Focus on critical gaps first, defer nice-to-have features
- **Testing Complexity**: Multi-component testing → Automated testing infrastructure investment

### Operational Risks
- **Migration Complexity**: User disruption → Backward compatibility preservation and migration tools
- **Performance Regression**: Temporary degradation → Performance monitoring and rollback procedures
- **Documentation Gaps**: Support burden → Documentation-first approach with user validation

This implementation roadmap provides a concrete path from the current hybrid architecture to the target four-component architecture while maintaining system functionality and meeting performance requirements throughout the transition.