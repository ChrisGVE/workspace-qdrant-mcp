# CLI Migration Guide: Python to Rust

This guide helps you migrate from the deprecated Python CLI (`wqm-py`) to the
new Rust CLI (`wqm`).

## Why Migrate?

The Rust CLI offers significant advantages:

- **Performance**: <100ms startup time vs ~500ms for Python
- **Native Integration**: Direct gRPC communication with the memexd daemon
- **Cross-Platform**: Single binary for macOS, Linux, and Windows
- **Memory Efficiency**: Lower memory footprint
- **No Python Runtime**: Works without Python installed

## Installation

### From Binary Release

```bash
# macOS (Apple Silicon)
curl -L https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/latest/download/wqm-darwin-arm64 -o /usr/local/bin/wqm
chmod +x /usr/local/bin/wqm

# macOS (Intel)
curl -L https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/latest/download/wqm-darwin-x64 -o /usr/local/bin/wqm
chmod +x /usr/local/bin/wqm

# Linux (x64)
curl -L https://github.com/ChrisGVE/workspace-qdrant-mcp/releases/latest/download/wqm-linux-x64 -o /usr/local/bin/wqm
chmod +x /usr/local/bin/wqm
```

### From Source

```bash
cd src/rust/cli
cargo build --release
# Binary at: target/release/wqm
```

## Command Mapping

### Service Management (unchanged)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm service install` | `wqm service install` | Identical |
| `wqm service uninstall` | `wqm service uninstall` | Identical |
| `wqm service start` | `wqm service start` | Identical |
| `wqm service stop` | `wqm service stop` | Identical |
| `wqm service restart` | `wqm service restart` | Identical |
| `wqm service status` | `wqm service status` | Identical |
| `wqm service logs` | `wqm service logs` | Identical |

### Admin Commands (mostly unchanged)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm admin status` | `wqm admin status` | Identical |
| `wqm admin collections` | `wqm admin collections` | Identical |
| `wqm admin health` | `wqm admin health` | Identical |
| `wqm admin projects` | `wqm admin projects` | Identical |
| `wqm admin queue` | `wqm status --queue` | Moved to status |
| `wqm admin metrics` | `wqm status --metrics` | Moved to status |
| `wqm admin config` | `wqm admin config` | Identical |

### Library Management (unchanged)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm library list` | `wqm library list` | Identical |
| `wqm library create` | `wqm library create` | Identical |
| `wqm library remove` | `wqm library remove` | Identical |
| `wqm library status` | `wqm library status` | Identical |
| `wqm library info` | `wqm library info` | Identical |
| `wqm library add` | `wqm library watch` | Renamed for clarity |
| `wqm library watches` | `wqm library watches` | Identical |
| `wqm library unwatch` | `wqm library unwatch` | Identical |

### Status Monitoring

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm status` | `wqm status` | Identical |
| `wqm status --live` | `wqm status --live` | Identical |
| `wqm queue status` | `wqm status --queue` | Integrated into status |

### Search (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm search project` | `wqm search project` | Requires `--features phase2` |
| `wqm search collection` | `wqm search collection` | Requires `--features phase2` |
| `wqm search global` | `wqm search global` | Requires `--features phase2` |
| `wqm search memory` | `wqm search memory` | Requires `--features phase2` |

### Ingest (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm ingest file` | `wqm ingest file` | Requires `--features phase2` |
| `wqm ingest folder` | `wqm ingest folder` | Requires `--features phase2` |
| `wqm ingest web` | `wqm ingest web` | Requires `--features phase2` |
| `wqm ingest yaml` | `wqm ingest yaml` | Requires `--features phase2` |

### Memory Management (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm memory list` | `wqm memory list` | Requires `--features phase2` |
| `wqm memory add` | `wqm memory add` | Requires `--features phase2` |
| `wqm memory edit` | `wqm memory edit` | Requires `--features phase2` |
| `wqm memory remove` | `wqm memory remove` | Requires `--features phase2` |

### Project Management (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm project list` | `wqm project list` | Requires `--features phase2` |
| `wqm project info` | `wqm project info` | Requires `--features phase2` |
| `wqm branch list` | `wqm project branches` | Merged into project |
| `wqm branch delete` | `wqm project branch-delete` | Merged into project |
| `wqm branch rename` | `wqm project branch-rename` | Merged into project |

### Watch Management

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm watch add` | `wqm library watch` | Merged into library |
| `wqm watch list` | `wqm library watches` | Merged into library |
| `wqm watch remove` | `wqm library unwatch` | Merged into library |
| `wqm watch status` | `wqm status --watch` | Integrated into status |

### Language/Grammar (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm grammar list` | `wqm language grammar-list` | Merged into language |
| `wqm grammar install` | `wqm language grammar-install` | Merged into language |
| `wqm lsp status` | `wqm language lsp-status` | Merged into language |
| `wqm lsp config` | `wqm language lsp-config` | Merged into language |

### Backup (Phase 2 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm backup create` | `wqm backup create` | Requires `--features phase2` |
| `wqm backup list` | `wqm backup list` | Requires `--features phase2` |
| `wqm backup restore` | `wqm backup restore` | Requires `--features phase2` |
| `wqm backup validate` | `wqm backup validate` | Requires `--features phase2` |

### Shell Completion (Phase 3 Feature Flag)

| Python CLI | Rust CLI | Notes |
|------------|----------|-------|
| `wqm init` | `wqm init` | Requires `--features phase3` |

## Commands Without Direct Equivalent

Some Python CLI commands don't have direct Rust equivalents because they were:
- Rarely used and not prioritized
- Functionality moved to other tools
- Deprecated for architectural reasons

| Python CLI | Status | Alternative |
|------------|--------|-------------|
| `wqm collections migrate-type` | Deprecated | Use `wqm admin migrate` |
| `wqm web start` | Removed | Security concerns |
| `wqm errors monitor` | In progress | `wqm status --live` |
| `wqm messages *` | In progress | `wqm admin messages` |
| `wqm observability` | In progress | `wqm admin health --deep` |
| `wqm wizard` | Phase 3 | Coming soon |

## Global Options

Global options work the same in both CLIs:

```bash
# Output format
wqm --format json admin status

# Verbose output
wqm -v service status
wqm --verbose service status

# Custom daemon address
wqm --daemon-addr http://custom:50051 admin status
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WQM_DAEMON_ADDR` | Daemon gRPC address | `http://127.0.0.1:50051` |
| `WQM_NO_DEPRECATION_WARNING` | Suppress Python CLI warning | `0` |

## Building with Feature Flags

To enable Phase 2 or Phase 3 commands:

```bash
# Phase 2 commands (search, ingest, memory, project, backup, language)
cargo build --release --features phase2

# Phase 3 commands (init, help, wizard)
cargo build --release --features phase3

# All features
cargo build --release --features "phase2,phase3"
```

## Suppressing Deprecation Warning

If you need to continue using the Python CLI temporarily:

```bash
export WQM_NO_DEPRECATION_WARNING=1
```

Or per-command:

```bash
WQM_NO_DEPRECATION_WARNING=1 wqm-py service status
```

## Getting Help

Both CLIs provide help:

```bash
# List all commands
wqm --help

# Command-specific help
wqm service --help
wqm admin --help
```

## Timeline

- **v0.4.0**: Python CLI marked as deprecated (current)
- **v0.5.0**: Python CLI removed from default installation
- **v1.0.0**: Python CLI removed entirely

## Feedback

If you encounter issues migrating, please:
1. Open an issue: https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
2. Tag with `cli-migration` label
3. Include the command you're trying to run
