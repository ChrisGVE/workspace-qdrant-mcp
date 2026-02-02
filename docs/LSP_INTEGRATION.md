# LSP Integration Guide

**Version:** 1.0
**Date:** 2026-02-02
**Status:** Active

This document provides comprehensive documentation for the Language Server Protocol (LSP) integration in workspace-qdrant-mcp.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Server Lifecycle](#server-lifecycle)
4. [Supported Languages](#supported-languages)
5. [Configuration Reference](#configuration-reference)
6. [Enrichment Data Structure](#enrichment-data-structure)
7. [CLI Commands](#cli-commands)
8. [Metrics and Monitoring](#metrics-and-monitoring)
9. [Troubleshooting](#troubleshooting)
10. [Performance Considerations](#performance-considerations)

---

## Overview

The LSP integration provides code intelligence features for active projects:

- **Symbol references**: Where a function/class/variable is used
- **Type information**: Type signatures and documentation
- **Import resolution**: Resolved import targets

### Design Philosophy

```
Tree-sitter (always)  →  Semantic chunking, symbol definitions
         +
   LSP (active only)  →  References, types, resolved imports
         =
   Rich code context  →  Better search results
```

**Tree-sitter baseline**: Always runs during ingestion. Provides symbol definitions, language detection, and semantic chunking.

**LSP enhancement**: Runs only for active projects. Adds references, type information, and resolved imports.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Code Intelligence Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   File Change                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌────────────────┐                                                  │
│  │  Tree-sitter   │  ← Always runs                                   │
│  │  (baseline)    │     • Symbol extraction                          │
│  └───────┬────────┘     • Semantic chunking                          │
│          │              • Language detection                          │
│          ▼                                                           │
│  ┌────────────────┐                                                  │
│  │  LSP Manager   │  ← Active projects only                          │
│  │  (enhancement) │     • Reference queries                          │
│  └───────┬────────┘     • Type hover                                 │
│          │              • Import resolution                          │
│          ▼                                                           │
│  ┌────────────────┐                                                  │
│  │    Qdrant      │  ← Enriched vectors                              │
│  │   (storage)    │                                                  │
│  └────────────────┘                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| `LspServerDetector` | Find available language servers on system |
| `LanguageServerManager` | Per-project server lifecycle management |
| `StateManager` | SQLite persistence for recovery |
| `ServerInstance` | Individual server process wrapper |

### Modules

```
daemon/core/src/lsp/
├── mod.rs              # Module exports
├── config.rs           # LspConfig, language configs
├── detection.rs        # Server discovery, LspServerDetector
├── instance.rs         # ServerInstance, JSON-RPC client
├── lifecycle.rs        # Health monitoring, restart policies
├── project_manager.rs  # LanguageServerManager, enrichment
├── state.rs            # SQLite persistence, StateManager
└── tests.rs            # Integration tests
```

---

## Server Lifecycle

### State Machine

```
         RegisterProject                DeprioritizeProject
              │                                │
              ▼                                ▼
        ┌─────────┐                    ┌─────────────┐
        │ Stopped │───spawn──────────→│ Initializing│
        └─────────┘                    └──────┬──────┘
              ▲                               │
              │                               ▼ LSP initialized
              │                        ┌─────────┐
              │◄────stop───────────────│ Running │◄──────┐
              │                        └────┬────┘       │
              │                             │            │
              │                             ▼ unhealthy  │ healthy
              │                        ┌─────────┐       │
              │◄────max retries────────│ Failed  │───────┘
              │                        └─────────┘  restart
              │
        ┌─────────────┐
        │  Unavailable │  (marked after max restarts)
        └─────────────┘
```

### Lifecycle Events

| Event | Trigger | Action |
|-------|---------|--------|
| Start server | RegisterProject + language detected | Spawn LSP process, initialize |
| Health check | Periodic (30s default) | Verify process alive, test response |
| Auto-restart | Health check fails | Kill zombie, respawn if under max_restarts |
| Stop server | DeprioritizeProject + queue empty | Graceful shutdown |
| Mark unavailable | max_restarts exceeded | Stop attempts, log warning |
| Stability reset | 1 hour healthy | Reset restart_count to 0 |

### Deferred Shutdown

When a project is deprioritized, the daemon doesn't immediately stop LSP servers:

```python
# Pseudocode for deferred shutdown
def on_deprioritize_project(project_id):
    queue_depth = get_queue_depth(project_id)

    if queue_depth > 0:
        # Items still pending, defer shutdown
        schedule_deferred_shutdown(
            project_id,
            delay=deactivation_delay_secs,  # default 60s
        )
    else:
        # Queue empty, stop immediately
        stop_all_servers(project_id)

def deferred_shutdown_check(project_id):
    # Re-check if project was reactivated
    if is_project_active(project_id):
        cancel_deferred_shutdown(project_id)
        return

    queue_depth = get_queue_depth(project_id)
    if queue_depth == 0:
        stop_all_servers(project_id)
    else:
        # Still items, reschedule
        schedule_deferred_shutdown(project_id, delay=10)
```

### State Persistence

Server states are persisted to SQLite for daemon restart recovery:

**Table: `lsp_project_server_states`**

```sql
CREATE TABLE lsp_project_server_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    language TEXT NOT NULL,
    project_root TEXT NOT NULL,
    restart_count INTEGER NOT NULL DEFAULT 0,
    last_started_at TEXT NOT NULL,
    executable_path TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, language)
);
```

**Recovery flow:**

1. On daemon startup, query `lsp_project_server_states`
2. Clean up stale states (older than 24 hours)
3. For each restored state where project is still active:
   - Verify `project_root` path exists
   - Re-spawn server with stored configuration
   - Restore `restart_count` for proper restart limit tracking

---

## Supported Languages

### Language Detection

Languages are detected from file extensions:

| Language | Extensions | Default Server |
|----------|------------|----------------|
| Python | `.py`, `.pyw`, `.pyi` | `pyright`, `pylsp` |
| Rust | `.rs` | `rust-analyzer` |
| TypeScript | `.ts`, `.tsx` | `typescript-language-server` |
| JavaScript | `.js`, `.jsx`, `.mjs` | `typescript-language-server` |
| Go | `.go` | `gopls` |
| C | `.c`, `.h` | `clangd` |
| C++ | `.cpp`, `.hpp`, `.cc`, `.cxx` | `clangd` |
| Java | `.java` | `jdtls` |
| Kotlin | `.kt`, `.kts` | `kotlin-language-server` |
| Swift | `.swift` | `sourcekit-lsp` |
| Zig | `.zig` | `zls` |
| Ruby | `.rb` | `solargraph` |
| PHP | `.php` | `phpactor` |

### Server Requirements

Each language server must support these LSP methods:

| Method | Used For |
|--------|----------|
| `textDocument/references` | Find symbol usages |
| `textDocument/hover` | Type information |
| `textDocument/definition` | Import resolution |

### Installing Language Servers

```bash
# Via CLI
wqm lsp install python    # Installs pyright
wqm lsp install rust      # rust-analyzer via rustup
wqm lsp install typescript

# Manual installation
pip install pyright       # or pylsp
cargo install rust-analyzer
npm install -g typescript-language-server
```

---

## Configuration Reference

### LSP Settings in `config.yaml`

```yaml
# ~/.workspace-qdrant/config.yaml
lsp:
  # Enable/disable LSP integration
  enabled: true

  # User PATH for finding language servers
  user_path: "/usr/local/bin:/opt/homebrew/bin"

  # Maximum LSP servers per project
  max_servers_per_project: 3

  # Auto-start servers when project activates
  auto_start_on_activation: true

  # Delay before stopping servers after deactivation (seconds)
  deactivation_delay_secs: 60

  # Enable enrichment result caching
  enable_enrichment_cache: true

  # Cache TTL in seconds
  cache_ttl_secs: 300

  # Timeouts
  startup_timeout_secs: 30
  request_timeout_secs: 10

  # Health monitoring
  health_check_interval_secs: 30
  enable_auto_restart: true
  max_restart_attempts: 3
  restart_backoff_multiplier: 2.0

  # Reset restart count after this many seconds healthy
  stability_reset_secs: 3600
```

### Per-Language Configuration

```yaml
lsp:
  language_configs:
    python:
      servers: ["pyright", "pylsp"]
      priority: 1  # pyright preferred

    rust:
      servers: ["rust-analyzer"]
      initialization_options:
        cargo:
          allFeatures: true

    typescript:
      servers: ["typescript-language-server"]
      workspace_config:
        typescript:
          preferences:
            importModuleSpecifier: "relative"
```

### Server-Specific Configuration

```yaml
lsp:
  server_configs:
    rust-analyzer:
      command: "rust-analyzer"
      arguments: []
      root_patterns: ["Cargo.toml", "Cargo.lock"]
      initialization_options:
        checkOnSave:
          command: "clippy"

    pyright:
      command: "pyright-langserver"
      arguments: ["--stdio"]
      root_patterns: ["pyproject.toml", "setup.py", "requirements.txt"]
```

---

## Enrichment Data Structure

### Enrichment Payload in Qdrant

When LSP enrichment succeeds, the document payload includes:

```json
{
  "project_id": "a1b2c3d4e5f6",
  "file_path": "src/auth.py",
  "chunk_type": "function",
  "symbol_name": "validate_token",
  "language": "python",
  "start_line": 42,
  "end_line": 67,

  "lsp_enrichment": {
    "enrichment_status": "success",

    "references": [
      {
        "file": "src/api.py",
        "line": 23,
        "column": 15,
        "end_line": 23,
        "end_column": 29
      },
      {
        "file": "src/middleware.py",
        "line": 56,
        "column": 8,
        "end_line": 56,
        "end_column": 22
      }
    ],

    "type_info": {
      "type_signature": "Callable[[str], bool]",
      "documentation": "Validates JWT token and returns True if valid.",
      "kind": "function",
      "container": "auth"
    },

    "resolved_imports": [
      {
        "import_name": "jwt.decode",
        "target_file": "/usr/lib/python3.11/site-packages/jwt/__init__.py",
        "target_symbol": "decode",
        "is_stdlib": false,
        "resolved": true
      }
    ],

    "definition": null,
    "error_message": null
  }
}
```

### Enrichment Status Values

| Status | Meaning |
|--------|---------|
| `success` | All LSP queries returned data |
| `partial` | Some queries succeeded, some failed |
| `failed` | All LSP queries failed |
| `skipped` | Project not active, no LSP available |

### Graceful Degradation

When LSP is unavailable, enrichment is skipped:

```json
{
  "lsp_enrichment": {
    "enrichment_status": "skipped",
    "references": [],
    "type_info": null,
    "resolved_imports": [],
    "definition": null,
    "error_message": "Project not active"
  }
}
```

Documents are still indexed with Tree-sitter baseline data.

---

## CLI Commands

### Server Management

```bash
# List available and installed servers
wqm lsp list

# Install a language server
wqm lsp install <language>
wqm lsp install python
wqm lsp install rust
wqm lsp install typescript

# Remove a language server
wqm lsp remove <language>

# Show status of running servers
wqm lsp status

# Show detailed server information
wqm lsp info <language>
```

### Diagnostics

```bash
# Show LSP metrics
wqm admin metrics --lsp

# Check server health
wqm lsp health

# View LSP logs
wqm logs --filter lsp

# Test LSP for a specific file
wqm lsp test <file_path>
```

### Example Output

```bash
$ wqm lsp status
LSP Server Status
═══════════════════════════════════════════════════════════════

Project: my-project (a1b2c3d4)
  Language    Server           Status      Uptime    Restarts
  ─────────────────────────────────────────────────────────────
  rust        rust-analyzer    running     2h 15m    0
  python      pyright          running     1h 45m    1

Project: api-server (e5f6g7h8)
  Language    Server           Status      Uptime    Restarts
  ─────────────────────────────────────────────────────────────
  typescript  tsserver         running     45m       0

Total: 3 servers, 2 projects
```

---

## Metrics and Monitoring

### Available Metrics

The `LanguageServerManager` tracks these metrics:

| Metric | Description |
|--------|-------------|
| `total_enrichment_queries` | Total enrichment attempts |
| `successful_enrichments` | Full enrichment (all queries worked) |
| `partial_enrichments` | Some queries failed |
| `failed_enrichments` | All queries failed |
| `skipped_enrichments` | Project inactive |
| `cache_hits` | Enrichment returned from cache |
| `cache_misses` | Enrichment not in cache |
| `total_references_queries` | LSP references calls |
| `total_type_info_queries` | LSP hover calls |
| `total_import_queries` | LSP definition calls |
| `total_server_starts` | Server spawn count |
| `total_server_restarts` | Auto-restart count |
| `total_server_stops` | Server stop count |

### Computed Metrics

```rust
// Cache hit rate (0-100%)
cache_hit_rate = (cache_hits / (cache_hits + cache_misses)) * 100

// Enrichment success rate (0-100%)
enrichment_success_rate = (successful_enrichments / total_enrichment_queries) * 100
```

### Accessing Metrics

```bash
# Via CLI
wqm admin metrics --lsp

# Via gRPC
GetMetrics(filter: "lsp")
```

### Example Metrics Output

```json
{
  "lsp_metrics": {
    "total_enrichment_queries": 1500,
    "successful_enrichments": 1200,
    "partial_enrichments": 200,
    "failed_enrichments": 50,
    "skipped_enrichments": 50,
    "cache_hits": 800,
    "cache_misses": 700,
    "cache_hit_rate": 53.3,
    "enrichment_success_rate": 80.0,
    "total_server_starts": 5,
    "total_server_restarts": 2,
    "total_server_stops": 3
  }
}
```

---

## Troubleshooting

### Common Issues

#### Server Won't Start

**Symptoms:**
- `wqm lsp status` shows server stuck in "initializing"
- Enrichment always returns "skipped"

**Diagnostics:**
```bash
# Check if server binary exists
which rust-analyzer
which pyright-langserver

# Check server logs
wqm logs --filter "lsp.*start"

# Test server manually
rust-analyzer --version
```

**Solutions:**
1. Verify server is installed: `wqm lsp install <language>`
2. Check PATH: `wqm lsp info <language>` shows search paths
3. Check permissions on server binary
4. Verify project has correct root files (e.g., `Cargo.toml` for Rust)

#### Server Keeps Restarting

**Symptoms:**
- High `total_server_restarts` metric
- Server marked as "unavailable" after max restarts

**Diagnostics:**
```bash
# Check health metrics
wqm lsp health

# View restart history
wqm logs --filter "lsp.*restart"
```

**Solutions:**
1. Check server logs for crash reason
2. Verify project files are valid (e.g., valid `Cargo.toml`)
3. Increase memory limits if server is OOMing
4. Check for conflicting processes on same files

#### Enrichment Not Working

**Symptoms:**
- Documents have `enrichment_status: "skipped"`
- No references or type info in search results

**Diagnostics:**
```bash
# Check if project is active
wqm admin projects --active

# Check if LSP is enabled
wqm config get lsp.enabled

# Test enrichment directly
wqm lsp test src/main.rs
```

**Solutions:**
1. Verify project is active (RegisterProject was called)
2. Check LSP is enabled in config
3. Verify language server supports the file type
4. Check server is healthy: `wqm lsp status`

#### Slow Enrichment

**Symptoms:**
- Ingestion takes much longer than expected
- High latency on enrichment queries

**Diagnostics:**
```bash
# Check request timeout
wqm config get lsp.request_timeout_secs

# Check cache hit rate
wqm admin metrics --lsp | grep cache_hit_rate
```

**Solutions:**
1. Enable enrichment caching: `wqm config set lsp.enable_enrichment_cache true`
2. Increase cache TTL for stable codebases
3. Ensure project is on SSD (LSP I/O intensive)
4. Reduce concurrent enrichment queries

### Log Locations

| Log | Location |
|-----|----------|
| Daemon logs | `~/.workspace-qdrant/logs/daemon.log` |
| LSP-specific | Filter with `--filter lsp` |
| Server stdout | `~/.workspace-qdrant/logs/lsp/<server>.log` |

### Debug Mode

Enable verbose LSP logging:

```yaml
# config.yaml
logging:
  level: debug
  lsp:
    level: trace  # Very verbose
```

---

## Performance Considerations

### Resource Usage

| Resource | Per Server | Notes |
|----------|-----------|-------|
| Memory | 100-500 MB | rust-analyzer tends to use more |
| CPU | Low (idle) | Spikes during queries |
| Disk I/O | Medium | Indexing project on start |

### Optimization Tips

1. **Limit servers per project**
   ```yaml
   lsp:
     max_servers_per_project: 2
   ```

2. **Use caching**
   ```yaml
   lsp:
     enable_enrichment_cache: true
     cache_ttl_secs: 600  # 10 minutes
   ```

3. **Tune timeouts**
   ```yaml
   lsp:
     startup_timeout_secs: 60  # Large projects need more time
     request_timeout_secs: 5   # Reduce if servers are fast
   ```

4. **Disable for large projects**
   If a project is too large for LSP to handle efficiently, rely on Tree-sitter baseline only:
   ```yaml
   lsp:
     excluded_projects:
       - "/path/to/huge/project"
   ```

### Scaling Guidelines

| Project Size | Files | Recommendation |
|--------------|-------|----------------|
| Small | < 1,000 | All features enabled |
| Medium | 1,000 - 10,000 | Enable caching, tune timeouts |
| Large | > 10,000 | Consider disabling LSP for inactive areas |

### Memory Management

The daemon automatically:
- Stops servers for inactive projects
- Cleans up stale state (24h threshold)
- Caches enrichment results to avoid repeated queries

To manually clean up:
```bash
wqm lsp stop --inactive     # Stop all servers for inactive projects
wqm admin cleanup --lsp     # Clear stale LSP state
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [WORKSPACE_QDRANT_MCP.md](../WORKSPACE_QDRANT_MCP.md) | Main specification |
| [ARCHITECTURE.md](./ARCHITECTURE.md) | System architecture |
| [CLI.md](./CLI.md) | CLI command reference |
| [METRICS.md](./METRICS.md) | Metrics documentation |

---

**Version:** 1.0
**Last Updated:** 2026-02-02
