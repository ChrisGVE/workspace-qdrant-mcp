# System Architecture Issues & Requirements

## Critical Issues Identified

### 1. Binary Installation & Distribution
- **Current**: Binary is built in project directory (`./rust-engine/target/release/memexd`)
- **Required**: System-wide installation in `/usr/local/bin` or `/opt/bin`
- **Impact**: Users cannot access daemon from anywhere, not following standard Unix practices

### 2. Configuration Management Mismatch
- **Current**: Two separate config systems:
  - MCP server: YAML config (`~/.config/workspace-qdrant-mcp/config.yaml`)
  - Daemon: TOML config (currently project-local)
- **Problem**: No unified configuration, duplication of settings
- **Required**: Single source of truth for shared settings

### 3. Missing Configuration Integration
- **Current**: Daemon has no knowledge of MCP server configuration
- **Missing Settings**:
  - Embeddings model configuration
  - Qdrant URL/port/API key
  - gRPC vs HTTP preference  
  - Download/cache settings for embeddings
- **Impact**: Components cannot communicate properly

### 4. Service Discovery & Communication
- **Current**: Hardcoded assumptions about endpoints
- **Problem**: Daemon and MCP server don't know each other's addresses
- **Required**: 
  - Daemon should have configurable listen address/port
  - MCP server should know daemon endpoint
  - Proper service discovery mechanism

### 5. Multi-Instance Support
- **Current**: Single daemon assumption
- **Required**: Support for:
  - Multiple MCP server instances (different projects)
  - Each with potentially different configurations
  - Concurrent operation without conflicts
  - Project-specific vs global settings

## Proposed Architecture

### Configuration Hierarchy
1. **Global Config**: `~/.config/workspace-qdrant-mcp/config.yaml`
   - Default embeddings model
   - Default Qdrant settings
   - Global daemon settings
   
2. **Project Config**: `<project>/.workspace-qdrant-mcp/config.yaml`
   - Project-specific overrides
   - Local daemon port (if different)
   - Project-specific embeddings/collections

### Communication Flow
```
Claude Code MCP Client
    ↓
MCP Server (workspace-qdrant-mcp)
    ↓ gRPC/IPC
Daemon (memexd) 
    ↓ HTTP/gRPC
Qdrant Database
```

### Service Management
- Daemon should be installable as system service
- MCP server should auto-start daemon if not running
- Proper cleanup and resource management
- Health checks and auto-recovery

## Immediate Actions Required

1. **Install daemon system-wide**
2. **Create unified configuration system**
3. **Implement proper service discovery**
4. **Add embedding auto-download**
5. **Test multi-instance scenarios**

## Testing Strategy

1. Single instance basic functionality
2. Multiple concurrent MCP servers
3. Different project configurations
4. Daemon restart/recovery scenarios
5. Network connectivity issues

---

*This document tracks the fundamental architectural issues that must be resolved for a production-ready system.*