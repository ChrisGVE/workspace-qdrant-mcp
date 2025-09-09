# Service Discovery System Architecture Design

## Current Architecture Analysis

### Rust Daemon (rust-engine/core)
- Location: `rust-engine/core/src/`
- IPC Module: `ipc.rs` - handles inter-process communication with Python MCP
- Config Module: `config.rs` - daemon configuration
- Unified Config: `unified_config.rs` - integration with Python config system
- Main entry: `bin/` directory

### Python MCP Server
- Location: `src/workspace_qdrant_mcp/`
- Main server: `server.py` - FastMCP server implementation
- Config system: `config/` directory - unified configuration management
- Core modules: `core/` directory

### Current Communication Gaps
1. No automatic service discovery - components must be manually configured
2. Hardcoded endpoints in configuration files
3. No automatic health checking between services
4. No graceful handling of component restarts

## Service Discovery Architecture Design

### Core Components

#### 1. Service Registry (File-based)
**Location**: `~/.workspace-qdrant/services.json`

**Schema**:
```json
{
  "version": "1.0.0",
  "services": {
    "rust-daemon": {
      "host": "127.0.0.1",
      "port": 8080,
      "pid": 12345,
      "startup_time": "2025-01-01T12:00:00Z",
      "auth_token": "secure-token",
      "health_endpoint": "/health",
      "grpc_port": 50051,
      "status": "healthy|unhealthy|starting|stopping"
    },
    "python-mcp": {
      "host": "127.0.0.1", 
      "port": 8000,
      "pid": 12346,
      "startup_time": "2025-01-01T12:01:00Z",
      "auth_token": "secure-token",
      "health_endpoint": "/health",
      "transport": "stdio|http",
      "status": "healthy|unhealthy|starting|stopping"
    }
  },
  "last_updated": "2025-01-01T12:01:30Z"
}
```

#### 2. Discovery Strategies (Priority Order)
1. **File-based Registry Check** - Primary discovery method
2. **Network Discovery** - UDP multicast/broadcast fallback
3. **Configuration Fallback** - Use unified config system
4. **Default Endpoints** - Standard ports as last resort

#### 3. Network Discovery Protocol
**UDP Multicast**: 239.255.42.42:9999
**Message Format**:
```json
{
  "type": "discovery_request|discovery_response|health_ping",
  "service_name": "rust-daemon|python-mcp",
  "timestamp": "2025-01-01T12:00:00Z",
  "payload": {
    // Service-specific information
  }
}
```

### Implementation Plan

#### Phase 1: Rust Service Discovery Module
1. Create `service_discovery.rs` in `rust-engine/core/src/`
2. Implement `ServiceRegistry` struct for file-based registry
3. Add `NetworkDiscovery` for UDP multicast
4. Create `DiscoveryManager` orchestrator
5. Integrate with `unified_config.rs`

#### Phase 2: Python Service Discovery Integration
1. Create `service_discovery.py` in `src/workspace_qdrant_mcp/core/`
2. Implement registry file management
3. Add network discovery client
4. Integrate with existing MCP server startup

#### Phase 3: Health Checking & Lifecycle Management
1. Implement health check endpoints in both services
2. Add stale entry cleanup (process PID validation)
3. Implement graceful shutdown with registry cleanup

#### Phase 4: Security & Authentication
1. Generate shared authentication tokens
2. Process ownership validation
3. File permission management

#### Phase 5: Cross-platform Compatibility
1. Handle Windows/macOS/Linux differences
2. Registry file location per platform
3. Network interface detection

### File Structure

```
rust-engine/core/src/
├── service_discovery/
│   ├── mod.rs           # Main module
│   ├── registry.rs      # File-based service registry
│   ├── network.rs       # UDP multicast discovery
│   ├── health.rs        # Health checking
│   └── manager.rs       # Discovery manager orchestrator
├── service_discovery.rs  # Re-export module

src/workspace_qdrant_mcp/core/
├── service_discovery/
│   ├── __init__.py      # Main module
│   ├── registry.py      # File-based service registry
│   ├── network.py       # UDP multicast discovery
│   ├── health.py        # Health checking
│   └── manager.py       # Discovery manager orchestrator
```

### Integration Points

#### Rust Daemon Integration
- Modify `main.rs` to initialize discovery on startup
- Update `ipc.rs` to use discovered endpoints
- Integrate with `unified_config.rs` for fallback configuration

#### Python MCP Server Integration  
- Modify `server.py` to register service on startup
- Update client connection logic to use discovery
- Integrate with existing config system

### Testing Strategy

#### Unit Tests
- Registry file operations (atomic writes, locks)
- Network discovery message parsing
- Health check implementations
- Cross-platform compatibility

#### Integration Tests
- Daemon-MCP discovery scenarios
- Startup order independence
- Failure recovery and re-discovery
- Stale entry cleanup validation

#### End-to-End Tests
- Full communication establishment
- Health monitoring and recovery
- Configuration fallback scenarios
- Multi-instance collision handling

### Error Handling & Logging

#### Discovery Events to Log
- Service registration/deregistration
- Discovery attempts and results
- Health check status changes
- Network discovery timeouts
- Registry file access issues

#### Graceful Degradation
- Fall back to configuration-based connection
- Continue operating with last known endpoints
- Retry discovery with exponential backoff
- Alert on persistent discovery failures

This design provides a robust, cross-platform service discovery system that enhances the communication reliability between the Rust daemon and Python MCP server.