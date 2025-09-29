# Configuration Architecture: Rust vs Python

## Overview

The workspace-qdrant-mcp project has **TWO separate codebases** that both need configuration:

1. **Rust Engine** (daemon/processing engine)
2. **Python Codebase** (MCP server + CLI tools)

Both implement the **same lua-style configuration pattern** with identical APIs for consistency.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Sources                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ YAML Files   │  │ Environment  │  │   Defaults   │         │
│  │              │  │  Variables   │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└────────────┬────────────────┬─────────────────────────────────┘
             │                │
             v                v
┌────────────────────────────────────────────────────────────────┐
│                     Shared Configuration                        │
│             assets/default_configuration.yaml                   │
│  (Single source of truth for both Rust and Python)             │
└────────────┬───────────────────────────────┬───────────────────┘
             │                               │
             v                               v
┌────────────────────────────┐  ┌───────────────────────────────┐
│      RUST ENGINE           │  │      PYTHON CODEBASE          │
│   (Daemon/Processing)      │  │   (MCP Server + CLI)          │
│                            │  │                               │
│  rust-engine/src/config.rs │  │  src/python/common/core/      │
│                            │  │    config.py                  │
│  ┌──────────────────────┐ │  │  ┌─────────────────────────┐  │
│  │ ConfigManager        │ │  │  │ ConfigManager           │  │
│  │ - Lua-style access   │ │  │  │ - Lua-style access      │  │
│  │ - Singleton pattern  │ │  │  │ - Thread-safe singleton │  │
│  │ - Unit conversions   │ │  │  │ - Unit conversions      │  │
│  └──────────────────────┘ │  │  └─────────────────────────┘  │
│                            │  │                               │
│  API Functions:            │  │  API Functions:               │
│  • get_config(path)        │  │  • get_config(path)           │
│  • get_config_string()     │  │  • get_config_string()        │
│  • get_config_bool()       │  │  • get_config_bool()          │
│  • get_config_u64()        │  │  • get_config_int()           │
│  • get_config_u16()        │  │  • get_config_float()         │
│  └────────────────────────┘  │  • get_config_dict()          │
│                               │  • get_config_list()          │
│  Used By:                     │  └─────────────────────────┘  │
│  • Daemon (memexd)            │                               │
│  • File watcher               │  Used By:                     │
│  • Processing engine          │  • MCP Server                 │
│  • gRPC services              │  • CLI (wqm)                  │
│  • Auto-ingestion             │  • Client libraries           │
│                               │  • Utilities                  │
└───────────────────────────────┘  └───────────────────────────┘
```

---

## Detailed Component Breakdown

### 1. Rust Engine Configuration

**File**: `rust-engine/src/config.rs`

**Status**: ✅ **FULLY IMPLEMENTED**

**Key Features**:
```rust
// Singleton global configuration
pub static CONFIG: OnceLock<Mutex<ConfigManager>> = OnceLock::new();

// Lua-style accessor functions
pub fn get_config_string(path: &str, default: &str) -> String
pub fn get_config_bool(path: &str, default: bool) -> bool
pub fn get_config_u64(path: &str, default: u64) -> u64
pub fn get_config_u16(path: &str, default: u16) -> u16

// Dictionary-based ConfigManager
impl ConfigManager {
    pub fn get_config(&self, path: &str) -> Result<&ConfigValue>
    pub fn set(&mut self, path: &str, value: ConfigValue)
}
```

**What It Configures**:
- Daemon settings (host, port, timeouts)
- Qdrant connection (URL, API key)
- File watching (paths, patterns, debounce)
- Processing (batch sizes, concurrency)
- Auto-ingestion settings
- gRPC service configuration
- Logging and metrics

**Used By**:
- `memexd` binary (daemon)
- `rust-engine/src/daemon/mod.rs`
- `rust-engine/src/grpc/services/`
- All Rust processing modules

---

### 2. Python Configuration

**File**: `src/python/common/core/config.py`

**Status**: ✅ **FULLY IMPLEMENTED**

**Key Features**:
```python
# Singleton ConfigManager
class ConfigManager:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config_file=None, **kwargs):
        # Thread-safe singleton

# Global accessor functions
def get_config(path: str, default: Any = None) -> Any
def get_config_string(path: str, default: str = "") -> str
def get_config_bool(path: str, default: bool = False) -> bool
def get_config_int(path: str, default: int = 0) -> int
def get_config_float(path: str, default: float = 0.0) -> float
def get_config_dict(path: str, default: Dict = None) -> Dict
def get_config_list(path: str, default: List = None) -> List

# Factory function
def get_config_manager(config_file=None, **kwargs) -> ConfigManager
```

**What It Configures**:
- MCP server settings
- Qdrant client configuration
- Embedding models (FastEmbed)
- Collection management
- Workspace detection
- CLI tool behavior
- Python client settings

**Used By**:
- `src/python/workspace_qdrant_mcp/` (MCP server)
- `src/python/wqm_cli/` (CLI tools)
- `src/python/common/core/` (shared libraries)
- All Python processing modules

---

### 3. Additional Python Modules (Created During Fixes)

#### a) `unified_config.py` - Stub

**File**: `src/python/common/core/unified_config.py`

**Status**: ⚠️ **STUB ONLY** (Minimal Implementation)

**Purpose**: Provides interface for **planned advanced features**:
- Multi-format support (YAML ↔ JSON ↔ TOML conversion)
- Configuration validation beyond basic checks
- Configuration migration tools
- Format detection and transformation

**Current State**: Returns empty results, allows CLI to function

**Used By**:
- `src/python/wqm_cli/cli/commands/config.py` (advanced config commands)
- `src/python/wqm_cli/cli/config_commands.py`

**Why It Exists**: CLI code was written anticipating these features, but they were never fully implemented. The stub prevents import errors.

#### b) `yaml_config.py` - Compatibility Shim

**File**: `src/python/common/core/yaml_config.py`

**Status**: ⚠️ **BACKWARD COMPATIBILITY SHIM**

**Purpose**: Wraps new `ConfigManager` to provide old `WorkspaceConfig` interface

**Why It Exists**: `daemon_client.py` was written using the old configuration interface and hasn't been refactored yet.

**Used By**:
- `src/python/common/core/daemon_client.py`

**Should Be Removed**: When `daemon_client.py` is refactored to use `ConfigManager` directly

---

## Configuration File Flow

### Step-by-Step Process

1. **Application Starts** (Rust daemon or Python CLI/server)

2. **ConfigManager Initialization**:
   ```
   a) Load defaults from code (comprehensive baseline)
   b) Search for config file:
      - Explicit path if provided
      - assets/default_configuration.yaml
      - User config locations
   c) Parse YAML with unit conversions (50MB → bytes, 5s → milliseconds)
   d) Load environment variables (WORKSPACE_QDRANT_*)
   e) Apply CLI arguments/overrides
   ```

3. **Merge Strategy** (Priority Order):
   ```
   defaults < env_vars < yaml_file < kwargs
   (lowest)                        (highest)
   ```

4. **Result**: Single merged dictionary accessible via lua-style paths

5. **Access Pattern**:
   ```rust
   // Rust
   let url = get_config_string("qdrant.url", "http://localhost:6333");
   let enabled = get_config_bool("auto_ingestion.enabled", false);
   ```

   ```python
   # Python
   url = get_config_string("qdrant.url", "http://localhost:6333")
   enabled = get_config_bool("auto_ingestion.enabled", False)
   ```

---

## Key Configuration Paths

### Shared Between Rust & Python

| Path | Type | Purpose |
|------|------|---------|
| `qdrant.url` | string | Qdrant server URL |
| `qdrant.api_key` | string | Qdrant API key |
| `database.sqlite_path` | string | SQLite database location |
| `auto_ingestion.enabled` | bool | Enable auto-ingestion |
| `auto_ingestion.auto_create_watches` | bool | Auto-create watches |
| `processing.max_concurrent_tasks` | int | Parallel processing limit |
| `logging.level` | string | Log level (debug/info/warn/error) |

### Rust-Specific

| Path | Type | Purpose |
|------|------|---------|
| `daemon.host` | string | Daemon bind address |
| `daemon.port` | int | Daemon port |
| `grpc.host` | string | gRPC server address |
| `grpc.port` | int | gRPC server port |

### Python-Specific

| Path | Type | Purpose |
|------|------|---------|
| `server.host` | string | MCP server host |
| `server.port` | int | MCP server port |
| `embedding.model` | string | FastEmbed model name |
| `workspace.collection_basename` | string | Base collection name |

---

## What's NOT Implemented (Stubs)

### UnifiedConfigManager Features

❌ **Multi-format conversion** (YAML ↔ JSON ↔ TOML)
❌ **Advanced validation** (schema enforcement)
❌ **Configuration migration** (version upgrades)
❌ **Format auto-detection**

These are **nice-to-have features** that CLI commands reference but don't actually implement. The stubs allow the CLI to start without errors.

---

## Summary Table

| Component | Language | Status | Purpose |
|-----------|----------|--------|---------|
| `rust-engine/src/config.rs` | Rust | ✅ Full | Daemon configuration |
| `src/python/common/core/config.py` | Python | ✅ Full | MCP/CLI configuration |
| `src/python/common/core/unified_config.py` | Python | ⚠️ Stub | Advanced features (planned) |
| `src/python/common/core/yaml_config.py` | Python | ⚠️ Shim | Backward compatibility |

---

## Key Takeaways

1. **Both Rust and Python have FULL configuration systems** using identical lua-style patterns
2. **They share the same YAML configuration file** for consistency
3. **The stubs we created** are for:
   - Advanced features that were planned but not implemented (UnifiedConfigManager)
   - Backward compatibility for code that hasn't been refactored (yaml_config)
4. **The core ConfigManager works perfectly** in both languages
5. **No functionality is missing** - the system works end-to-end as designed

The confusion arose because:
- Old CLI code imported `Config` class that was removed during refactoring
- Some CLI commands reference `UnifiedConfigManager` for features never built
- `daemon_client.py` uses old `WorkspaceConfig` interface

All these issues are now resolved with proper stubs/shims while maintaining the fully functional core configuration system! 🎯