# Rust CLI Migration Guide

This guide covers migration from the Python `wqm` CLI to the new high-performance Rust CLI implementation.

## Table of Contents

- [Overview](#overview)
- [Performance Benefits](#performance-benefits)
- [Installation](#installation)
- [Dual Entry Points](#dual-entry-points)
- [Command Compatibility Matrix](#command-compatibility-matrix)
- [Phase 1 Commands Reference](#phase-1-commands-reference)
- [Breaking Changes](#breaking-changes)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Overview

The Rust CLI (`wqm`) provides a dramatically faster command-line experience compared to the Python implementation. The primary goal is **sub-100ms startup time** versus the 1-2 second startup of the Python CLI due to heavy import chains.

### Key Improvements

| Aspect | Python CLI | Rust CLI |
|--------|------------|----------|
| **Startup Time** | 1-2 seconds | <15ms |
| **Binary Size** | ~200MB (with venv) | <5MB |
| **Dependencies** | Python 3.10+, pip packages | None (static binary) |
| **Runtime** | Python interpreter | Native execution |

### Architecture

The Rust CLI communicates with the daemon via gRPC using the same `workspace_daemon.proto` protocol. It does not perform any embedding operations - all document processing is handled by the daemon.

```
┌────────────┐      gRPC       ┌────────────┐
│  Rust CLI  │◄───────────────►│  memexd    │
│  (wqm)     │  port 50051     │  (daemon)  │
└────────────┘                 └────────────┘
       │                              │
       │ ~15ms startup                │ Embedding
       │ Sub-command routing          │ Processing
       │ Output formatting            │ File watching
       │                              │ Qdrant writes
       └──────────────────────────────┘
```

## Performance Benefits

### Benchmark Results

Measured on macOS x86_64 with hyperfine:

| Command | Rust CLI | Python CLI | Speedup |
|---------|----------|------------|---------|
| `--version` | 11ms | ~1500ms | ~136x |
| `--help` | 11ms | ~1600ms | ~145x |
| `service --help` | 12ms | ~1700ms | ~141x |
| `admin --help` | 12ms | ~1700ms | ~141x |

### Why So Fast?

1. **No interpreter startup**: Native binary execution
2. **Minimal runtime**: Single-threaded tokio for async gRPC
3. **Lazy loading**: Commands are compiled but not initialized until needed
4. **Static linking**: LTO optimization for smallest binary size
5. **No embedding libraries**: All heavy processing in daemon

## Installation

### Option 1: Build from Source (Development)

```bash
# Build Rust CLI
cd src/rust/cli
cargo build --release

# Binary location: src/rust/cli/target/release/wqm

# Test it
./target/release/wqm --version
```

### Option 2: Install to System

```bash
# After building, install to system path
# macOS/Linux
sudo cp src/rust/cli/target/release/wqm /usr/local/bin/

# Or user-local (no sudo)
cp src/rust/cli/target/release/wqm ~/.local/bin/

# Verify installation
wqm --version
```

### Option 3: Via Python Package (Automatic)

When you install the Python package, the `wqm` command automatically delegates to the Rust binary if found:

```bash
# Install/update Python package
uv sync

# wqm now uses Rust binary (if found)
# Falls back to Python CLI (wqm-py) if Rust binary not found
wqm --version
```

The Python wrapper searches for the Rust binary in:
1. Development build: `src/rust/cli/target/release/wqm`
2. Debug build: `src/rust/cli/target/debug/wqm`
3. System paths: `/usr/local/bin`, `/opt/homebrew/bin`, `~/.local/bin`
4. Cargo install: `~/.cargo/bin`
5. System PATH

## Dual Entry Points

During the transition period, both CLIs are available:

| Entry Point | Implementation | Use Case |
|-------------|----------------|----------|
| `wqm` | Rust (via Python wrapper) | Primary, fast CLI |
| `wqm-py` | Python | Fallback, debugging |

### When to Use Each

**Use `wqm` (Rust):**
- Daily development workflow
- CI/CD pipelines (faster execution)
- Interactive use (instant response)
- System monitoring and health checks

**Use `wqm-py` (Python):**
- If Rust binary not installed
- Debugging Python-specific issues
- Testing new features not yet in Rust
- Legacy scripts depending on Python behavior

### Forcing Python CLI

If you need to explicitly use the Python CLI:

```bash
# Use Python CLI directly
wqm-py service status

# Or set environment variable
WQM_FORCE_PYTHON=1 wqm service status
```

## Command Compatibility Matrix

### Phase 1 Commands (Available Now)

| Command | Rust CLI | Python CLI | Status |
|---------|----------|------------|--------|
| `wqm service install` | Yes | Yes | Full compatibility |
| `wqm service start` | Yes | Yes | Full compatibility |
| `wqm service stop` | Yes | Yes | Full compatibility |
| `wqm service restart` | Yes | Yes | Full compatibility |
| `wqm service status` | Yes | Yes | Full compatibility |
| `wqm service logs` | Yes | Yes | Full compatibility |
| `wqm admin status` | Yes | Yes | Full compatibility |
| `wqm admin collections` | Yes | Yes | Full compatibility |
| `wqm admin health` | Yes | Yes | Full compatibility |
| `wqm admin projects` | Yes | Yes | Full compatibility |
| `wqm admin queue` | Yes | Yes | Full compatibility |
| `wqm status` | Yes | Yes | Consolidated (see below) |
| `wqm status --queue` | Yes | Yes | Replaces old `wqm queue` |
| `wqm status --watch` | Yes | Yes | Watch folder status |
| `wqm status --performance` | Yes | Yes | Resource metrics |
| `wqm status --live` | Yes | Yes | Real-time dashboard |
| `wqm status messages` | Yes | Yes | Replaces old `wqm messages` |
| `wqm status errors` | Yes | Yes | Replaces old `wqm errors` |
| `wqm library list` | Yes | Yes | Tag-based listing |
| `wqm library add` | Yes | Yes | Add with tag |
| `wqm library watch` | Yes | Yes | Watch with tag |
| `wqm library unwatch` | Yes | Yes | Unwatch by tag |
| `wqm library rescan` | Yes | Yes | Rescan by tag |
| `wqm library info` | Yes | Yes | Info by tag |
| `wqm library status` | Yes | Yes | Watch status |

### Phase 2 Commands (Coming Soon)

| Command | Rust CLI | Python CLI | Notes |
|---------|----------|------------|-------|
| `wqm search` | Planned | Yes | Multi-scope search |
| `wqm ingest` | Planned | Yes | Document ingestion |
| `wqm backup` | Planned | Yes | Qdrant snapshots |
| `wqm memory` | Planned | Yes | LLM rules |
| `wqm language` | Planned | Yes | LSP + grammar (merged) |
| `wqm project` | Planned | Yes | watch + branch (merged) |

### Removed/Deprecated Commands

| Old Command | Migration Path |
|-------------|---------------|
| `wqm queue` | Use `wqm status --queue` |
| `wqm messages` | Use `wqm status messages` |
| `wqm errors` | Use `wqm status errors` |
| `wqm observability` | Use `wqm status` with flags |
| `wqm tools` | Removed |
| `wqm collections` | Use `wqm admin collections` |
| `wqm migrate` | Use separate migration tool |
| `wqm branch` | Will be `wqm project branch` |
| `wqm grammar` | Will be `wqm language` |

## Phase 1 Commands Reference

### Service Commands

```bash
# Daemon lifecycle management
wqm service install      # Install as user service (launchd/systemd)
wqm service uninstall    # Remove service
wqm service start        # Start daemon
wqm service stop         # Stop daemon
wqm service restart      # Restart daemon
wqm service status       # Check if running
wqm service logs         # View logs (--lines, --follow)
```

### Admin Commands

```bash
# System administration
wqm admin status         # Comprehensive system status
wqm admin collections    # List all collections
wqm admin health         # Health check with component details
wqm admin projects       # List registered projects
wqm admin queue          # Queue depth and statistics
```

### Status Commands (Consolidated)

```bash
# Monitoring and observability (replaces old commands)
wqm status               # Default consolidated view
wqm status history       # Historical metrics
wqm status queue         # Ingestion queue details
wqm status watch         # File watcher status
wqm status performance   # CPU/memory/disk metrics
wqm status live          # Real-time updating dashboard
wqm status messages      # System messages (list/clear)
wqm status errors        # Recent errors
wqm status health        # System health check
```

### Library Commands (Tag-Based)

```bash
# Library management with tags (not collection names)
wqm library list                    # List all libraries
wqm library add <tag> <path>        # Add library (metadata only)
wqm library watch <tag> <path>      # Watch library folder
wqm library unwatch <tag>           # Stop watching
wqm library rescan <tag>            # Re-process all files
wqm library info [tag]              # Library information
wqm library status                  # Watch status for all libraries
```

### Global Options

```bash
wqm --format <table|json|plain>    # Output format (default: table)
wqm -v, --verbose                  # Verbose output
wqm --daemon-addr <url>            # Daemon address override
wqm -h, --help                     # Help
wqm -V, --version                  # Version
```

## Breaking Changes

### Library Command Changes

The `library` command now uses **tags** instead of direct collection names:

```bash
# Old (Python CLI)
wqm library add /path/to/docs --collection _fastapi-docs

# New (Rust CLI)
wqm library add fastapi /path/to/docs
# Creates collection: _fastapi
```

This provides:
- Cleaner user interface
- Automatic collection naming with `_` prefix
- Better organization by semantic tag

### Status Command Consolidation

Multiple commands merged into `wqm status`:

```bash
# Old commands (deprecated)
wqm queue status
wqm messages list
wqm errors show
wqm observability metrics

# New unified commands
wqm status --queue
wqm status messages
wqm status errors
wqm status --performance
```

### Output Format Changes

The Rust CLI uses consistent output formatting:

```bash
# Table output (default)
wqm admin status

# JSON output (for scripting)
wqm admin status --format json

# Plain text (minimal)
wqm admin status --format plain
```

## Troubleshooting

### Rust Binary Not Found

```
Error: wqm Rust binary not found.
```

**Solution:** Build the Rust CLI or install it:

```bash
# Build from source
cd src/rust/cli && cargo build --release

# Or install to system
sudo cp target/release/wqm /usr/local/bin/
```

### Daemon Connection Failed

```
Error: Failed to connect to memexd daemon
```

**Solution:** Ensure daemon is running:

```bash
wqm service status
wqm service start
```

### gRPC Timeout

```
Error: gRPC request timed out
```

**Solution:** Check daemon health and increase timeout:

```bash
# Check daemon
wqm service logs

# Restart if needed
wqm service restart
```

### Wrong CLI Version

To verify which CLI is running:

```bash
# Check version (shows "wqm 0.3.0" for Rust)
wqm --version

# Check binary path
which wqm

# Force Python CLI for comparison
wqm-py --version
```

## FAQ

### Q: Can I use both CLIs simultaneously?

Yes. The entry points are independent:
- `wqm` - Rust CLI (via Python wrapper)
- `wqm-py` - Python CLI

### Q: Will the Python CLI be deprecated?

The Python CLI will remain available as `wqm-py` for backward compatibility. However, `wqm` (the default command) will use the Rust implementation when available.

### Q: How do I contribute to the Rust CLI?

The Rust CLI source is at `src/rust/cli/`. See the development guide:

```bash
cd src/rust/cli
cargo build           # Development build
cargo test            # Run tests
cargo clippy          # Linting
cargo fmt             # Formatting
```

### Q: What about shell completions?

Shell completions will be added in Phase 3 via `wqm init`:

```bash
# Coming soon
wqm init bash     # Generate bash completions
wqm init zsh      # Generate zsh completions
wqm init fish     # Generate fish completions
```

### Q: How do I report issues?

File issues at: https://github.com/ChrisGVE/workspace-qdrant-mcp/issues

Include:
- `wqm --version` output
- Platform (macOS/Linux/Windows)
- Full error message
- Steps to reproduce

---

**For full CLI reference, see [CLI.md](CLI.md)**

**For MCP server documentation, see [README.md](../README.md)**
