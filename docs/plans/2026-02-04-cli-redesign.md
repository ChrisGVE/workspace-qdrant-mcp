# CLI Redesign Specification

**Date:** 2026-02-04
**Status:** Complete
**Version:** 1.0.0

---

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Command Structure](#command-structure)
4. [Command Reference](#command-reference)
5. [Removed Features](#removed-features)
6. [Help System](#help-system)
7. [Implementation Notes](#implementation-notes)
8. [Installation System](#installation-system)

---

## Overview

This document specifies the redesigned CLI (`wqm`) for workspace-qdrant-mcp. The redesign moves away from the phase-based feature flag system toward a holistic, user-intent-driven command structure.

### Goals

1. **Remove artificial barriers** - No more phase2/phase3 feature flags; all commands always available
2. **Organize by user intent** - Commands grouped by what users want to accomplish
3. **Consolidate related functionality** - LSP and grammar under `language`; diagnostics under `debug`
4. **Clear separation of concerns** - CLI handles runtime operations; installation script handles first-time setup

### Primary Use Cases

1. **Developer daily workflow** - Status checks, queue monitoring, debugging issues during active coding
2. **System administration** - Service lifecycle, library management, configuration

---

## Design Principles

### 1. No Feature Flags

All commands are always available. The phase1/phase2/phase3 feature flag system is removed entirely.

### 2. OS-Agnostic Service Management

The `service` command provides a unified interface across macOS (launchctl), Linux (systemctl), and Windows (Windows Services). Users don't need to know the underlying mechanism.

### 3. Helpful Error Messages

When users provide invalid input or miss required arguments, the CLI displays the full help for that command instead of a terse error. Configuration:

```rust
#[command(arg_required_else_help = true)]
```

### 4. Separation from Installation

The CLI (`wqm`) does NOT handle:
- First-time installation
- Service registration with the OS
- Initial configuration file creation
- Location selection (config, cache, state)

These are handled by the installation script (see Section 8).

---

## Command Structure

```
wqm (13 top-level commands)
├── service      Daemon lifecycle control
├── status       System monitoring dashboard
├── library      Library content management
├── project      Project lifecycle
├── queue        Queue inspection
├── language     LSP + Tree-sitter management
├── memory       Behavioral rules management
├── search       Semantic search
├── update       System updates
├── debug        Troubleshooting and diagnostics
├── backup       Create Qdrant snapshots
├── restore      Restore from snapshots
└── init         Shell completion setup
```

---

## Command Reference

### 1. service

**Purpose:** OS-agnostic daemon (memexd) lifecycle control.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `start` | Start the daemon |
| `stop` | Stop the daemon |
| `restart` | Stop and start the daemon |
| `status` | Show daemon status (running/stopped, port) |

**Examples:**
```bash
wqm service start
wqm service status
wqm service restart
```

**Notes:**
- Wraps launchctl (macOS), systemctl (Linux), Windows Services (Windows)
- Does NOT include `install` or `uninstall` - that's the installation script's job

---

### 2. status

**Purpose:** System monitoring dashboard for at-a-glance health checks.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| *(default)* | Overview of all system components |
| `queue` | Queue status summary |
| `watch` | File watcher status |
| `performance` | Performance metrics |
| `health` | Component health check |
| `live` | Real-time updating dashboard |

**Examples:**
```bash
wqm status              # Overview
wqm status health       # Component health
wqm status live         # Real-time dashboard
```

---

### 3. library

**Purpose:** Manage library content (documentation, reference materials indexed for search).

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List all registered libraries |
| `add` | Register a new library folder |
| `ingest` | Ingest a single file or webpage into a library |
| `watch` | Start watching a library folder for changes |
| `unwatch` | Stop watching a library folder |
| `remove` | Remove a library and its indexed content |
| `config` | Configure library settings (mode, patterns, tags, rename) |

**Library Config Options:**
- `mode` - sync (deletes removed files) vs incremental (append-only)
- `patterns` - file patterns to include/exclude
- `tags` - metadata tags for the library
- `name` - rename the library

**Examples:**
```bash
wqm library list
wqm library add docs /path/to/docs --mode sync
wqm library ingest my-lib /path/to/file.pdf
wqm library watch my-lib
wqm library config my-lib --mode incremental
wqm library remove my-lib
```

---

### 4. project

**Purpose:** Manage tracked projects (projects are auto-registered by the MCP server, but can be listed, inspected, or removed).

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List all tracked projects |
| `info` | Show detailed project information |
| `remove` | Remove a project from tracking (deletes indexed content) |

**Notes:**
- Projects are NOT registered via CLI - the MCP server handles registration automatically when a session starts
- CLI only provides listing, inspection, and removal

**Examples:**
```bash
wqm project list
wqm project info my-project-abc123
wqm project remove my-project-abc123
```

---

### 5. queue

**Purpose:** Inspect the unified processing queue for monitoring and debugging.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List queue items with filters |
| `show` | Show detailed information for a specific queue item |
| `stats` | Show queue statistics (counts by status, type, etc.) |

**Notes:**
- No `clean` subcommand - queue cleanup is automatic

**Examples:**
```bash
wqm queue list --status pending
wqm queue show abc123
wqm queue stats
```

---

### 6. language

**Purpose:** Manage language support (Tree-sitter grammars for parsing, LSP servers for code intelligence).

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List available/installed languages |
| `ts-install` | Install Tree-sitter grammar for a language |
| `ts-remove` | Remove Tree-sitter grammar for a language |
| `lsp-install` | Install LSP server for a language |
| `lsp-remove` | Remove LSP server for a language |
| `status` | Show status of language support |

**Notes:**
- Consolidates the previous `lsp` and `grammar` commands
- Clear distinction between Tree-sitter (ts-*) and LSP (lsp-*) operations

**Examples:**
```bash
wqm language list
wqm language ts-install rust
wqm language lsp-install python
wqm language status rust
```

---

### 7. memory

**Purpose:** Manage behavioral rules (persistent instructions that guide LLM behavior).

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `list` | List all memory rules |
| `add` | Add a new memory rule |
| `remove` | Remove a memory rule |
| `update` | Update an existing memory rule |
| `search` | Search memory rules semantically |
| `scope` | Change rule scope (project to global or vice versa) |

**Examples:**
```bash
wqm memory list
wqm memory add "Always use TypeScript strict mode" --scope project
wqm memory search "coding style"
wqm memory scope rule-123 --global
wqm memory remove rule-123
```

---

### 8. search

**Purpose:** Semantic search across indexed content.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `project` | Search within current project |
| `library` | Search within libraries |
| `memory` | Search memory rules |
| `global` | Search across all content |

**Examples:**
```bash
wqm search project "authentication flow"
wqm search library "API reference" --library docs
wqm search memory "coding standards"
wqm search global "error handling"
```

---

### 9. update

**Purpose:** Update the workspace-qdrant-mcp system (daemon, CLI, MCP server).

**Behavior:**
- Checks for available updates
- If update available, downloads and installs
- If Docker mode was selected during installation, triggers container rebuild
- If no update available, reports current version

**Options:**

| Option | Description |
|--------|-------------|
| `--check` | Check only, don't install |
| `--force` | Force reinstall current version |
| `--component` | Update specific component (daemon/cli/mcp/all) |

**Examples:**
```bash
wqm update                    # Check and install if available
wqm update --check            # Check only
wqm update --component cli    # Update CLI only
```

---

### 10. debug

**Purpose:** Troubleshooting tools for diagnosing issues.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `logs` | Show daemon logs |
| `errors` | Show recent errors |
| `queue-errors` | Show failed queue items with error details |
| `language` | Diagnose language support issues |

**Examples:**
```bash
wqm debug logs --lines 100
wqm debug logs --follow
wqm debug errors
wqm debug queue-errors
wqm debug language rust
```

---

### 11. backup

**Purpose:** Create Qdrant collection snapshots for backup.

**Behavior:**
- Creates snapshots of Qdrant collections
- Wraps Qdrant's native snapshot API for convenience

**Options:**

| Option | Description |
|--------|-------------|
| `--collection` | Specific collection to backup (default: all) |
| `--output` | Output directory for snapshot files |

**Examples:**
```bash
wqm backup
wqm backup --collection projects
wqm backup --output /path/to/backups
```

---

### 12. restore

**Purpose:** Restore Qdrant collections from snapshots.

**Behavior:**
- Restores from snapshot files created by `wqm backup`
- Wraps Qdrant's native snapshot restore API

**Options:**

| Option | Description |
|--------|-------------|
| `--collection` | Target collection to restore |
| `--snapshot` | Path to snapshot file |

**Examples:**
```bash
wqm restore --snapshot /path/to/snapshot.snapshot
wqm restore --collection projects --snapshot /path/to/projects.snapshot
```

---

### 13. init

**Purpose:** Set up shell completion for the CLI.

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `bash` | Generate bash completion script |
| `zsh` | Generate zsh completion script |
| `fish` | Generate fish completion script |

**Examples:**
```bash
wqm init bash >> ~/.bashrc
wqm init zsh >> ~/.zshrc
wqm init fish > ~/.config/fish/completions/wqm.fish
```

---

## Removed Features

### Commands Removed

| Command | Reason | Alternative |
|---------|--------|-------------|
| `wizard` | Absorbed into installation script | Use install script |
| `admin` | No clear purpose; functionality distributed | Use `status`, `backup`, `restore` |
| `lsp` (standalone) | Merged into `language` | Use `wqm language lsp-*` |
| `grammar` (standalone) | Merged into `language` | Use `wqm language ts-*` |

### Feature Flags Removed

| Flag | Status |
|------|--------|
| `phase2` | Removed - all commands always available |
| `phase3` | Removed - all commands always available |

### Subcommands Removed

| Command | Subcommand | Reason |
|---------|------------|--------|
| `service` | `install` | Moved to installation script |
| `service` | `uninstall` | Documented manual steps |
| `service` | `logs` | Moved to `debug logs` |
| `queue` | `clean` | Queue cleanup is automatic |

---

## Help System

### Built-in Help via Clap

The CLI uses clap's built-in help system with enhanced configuration:

1. **Root help on no arguments:**
   ```
   $ wqm
   <displays full help>
   ```

2. **Command help on missing required arguments:**
   ```
   $ wqm memory add
   <displays full memory add help>
   ```

3. **Help flag at all levels:**
   ```
   $ wqm --help
   $ wqm service --help
   $ wqm memory add --help
   ```

### Configuration

Apply `arg_required_else_help = true` to:
- Root CLI struct
- All command argument structs that have required parameters

```rust
#[derive(Parser)]
#[command(name = "wqm")]
#[command(arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Args)]
#[command(arg_required_else_help = true)]
pub struct MemoryAddArgs {
    /// The rule content
    content: String,
    // ...
}
```

---

## Implementation Notes

### Migration from Current CLI

1. **Remove feature flags** from `Cargo.toml` and all `#[cfg(feature = "...")]` annotations
2. **Merge `lsp` and `grammar`** into new `language` command
3. **Move diagnostics** from `language diagnose` to `debug language`
4. **Move logs** from `service logs` to `debug logs`
5. **Remove `admin`** command entirely
6. **Remove `wizard`** command entirely
7. **Remove `service install/uninstall`** subcommands
8. **Remove `queue clean`** subcommand
9. **Add new commands**: `backup`, `restore`
10. **Add `memory scope`** subcommand
11. **Add `library config`** subcommand
12. **Configure `arg_required_else_help`** on all commands

### Files to Modify

| File | Changes |
|------|---------|
| `src/rust/cli/Cargo.toml` | Remove phase2/phase3 features |
| `src/rust/cli/src/main.rs` | Restructure Commands enum |
| `src/rust/cli/src/commands/mod.rs` | Update module exports |
| `src/rust/cli/src/commands/service.rs` | Remove install/uninstall/logs |
| `src/rust/cli/src/commands/language.rs` | Rewrite with ts-*/lsp-* structure |
| `src/rust/cli/src/commands/debug.rs` | New file with logs, errors, queue-errors, language |
| `src/rust/cli/src/commands/backup.rs` | Rewrite as top-level command |
| `src/rust/cli/src/commands/restore.rs` | New file |
| `src/rust/cli/src/commands/memory.rs` | Add scope subcommand |
| `src/rust/cli/src/commands/library.rs` | Add config subcommand |

### Files to Delete

| File | Reason |
|------|--------|
| `src/rust/cli/src/commands/wizard.rs` | Removed |
| `src/rust/cli/src/commands/admin.rs` | Removed |
| `src/rust/cli/src/commands/lsp.rs` | Merged into language |
| `src/rust/cli/src/commands/grammar.rs` | Merged into language |
| `src/rust/cli/src/commands/help.rs` | Using clap built-in |

---

## Installation System

### Overview

The installation system handles first-time setup and is **completely separate** from the CLI. After initial installation, all operations go through `wqm`.

### Separation of Concerns

| Responsibility | Install Script | CLI (`wqm`) |
|---------------|----------------|-------------|
| First-time installation | ✅ | |
| Choose deployment mode (Docker/Native) | ✅ | |
| Configure locations (config, cache, state) | ✅ | |
| Register service with OS | ✅ | |
| Copy default config, customize paths | ✅ | |
| Download/place binaries | ✅ | |
| Start/stop/restart daemon | | ✅ |
| Check status | | ✅ |
| Updates (binaries or Docker rebuild) | | ✅ |
| All other runtime operations | | ✅ |
| Uninstall | Documented manual steps | |

### Deployment Configurations

Three deployment configurations are supported:

| Config | In Docker | Local | Use Case |
|--------|-----------|-------|----------|
| **Full Docker** | Qdrant + memexd + MCP server | wqm only | Users wanting complete containerization |
| **Partial Docker** | memexd only | Qdrant + MCP server + wqm | Containerized daemon with local tooling |
| **Native** | Nothing | Everything | Traditional local installation |

### Port Assignments

| Port | Service | Notes |
|------|---------|-------|
| 50051 | gRPC (memexd) | CLI and MCP server communicate with daemon |
| 50052 | MCP HTTP | Only used in Full Docker mode (Claude Code → MCP) |
| 6333 | Qdrant | Standard Qdrant port |

**Port exposure by configuration:**

| Config | 50051 (gRPC) | 50052 (MCP HTTP) | 6333 (Qdrant) |
|--------|--------------|------------------|---------------|
| Full Docker | Exposed | Exposed | Optional (for debugging) |
| Partial Docker | Exposed | N/A (stdio) | N/A (external) |
| Native | N/A | N/A | N/A |

### MCP Server Transport Modes

| Configuration | Transport | How Claude Code Connects |
|---------------|-----------|-------------------------|
| Native | stdio | Spawns MCP server as subprocess |
| Partial Docker | stdio | Spawns MCP server as subprocess |
| Full Docker | HTTP | Connects to `http://localhost:50052` |

### Binary Placement Logic

The install script determines binary location based on platform and permissions:

**macOS:**
```
1. /usr/local/bin (if writable without sudo)
2. ~/.local/bin (fallback)
```

**Linux:**
```
1. /usr/local/bin (if writable without sudo)
2. /opt/workspace-qdrant/bin (if writable)
3. ~/.local/bin (fallback)
```

**Windows:**
```
1. C:\Program Files\workspace-qdrant\ (if admin)
2. %LOCALAPPDATA%\Programs\workspace-qdrant\ (fallback)
```

User can always override during installation.

### XDG Compliance

When XDG environment variables are set, use them as defaults:

| XDG Variable | Purpose | Fallback |
|--------------|---------|----------|
| `XDG_CONFIG_HOME` | Config file | `~/.workspace-qdrant/` |
| `XDG_DATA_HOME` | State database | `~/.workspace-qdrant/` |
| `XDG_CACHE_HOME` | Cache files | `~/.workspace-qdrant/` |

User is prompted for each location with XDG/fallback as default (press Enter to accept).

### Service Registration

Services run as **user services** (not system services):

| OS | Service Manager | Service File Location |
|----|-----------------|----------------------|
| macOS | launchctl | `~/Library/LaunchAgents/com.workspace-qdrant.memexd.plist` |
| Linux | systemd | `~/.config/systemd/user/memexd.service` |
| Windows | Windows Services | User service via `sc.exe` |

**Rationale for user services:**
- No admin/sudo required for install
- Daemon needs access to user's projects and PATH
- File watching makes more sense in user context
- Runs when user is logged in (appropriate for dev tool)

### Installation Flow

```
┌─────────────────────────────────────────────────────────────┐
│           workspace-qdrant-mcp install script               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Deployment mode?                                         │
│    [Native] [Partial Docker] [Full Docker]                  │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ NATIVE          │  │ PARTIAL DOCKER  │  │ FULL DOCKER     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Binary location? (Native only)                           │
│    [detected default] or custom path                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Config file location?                                    │
│    [XDG_CONFIG_HOME or ~/.workspace-qdrant] or custom       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. State database location?                                 │
│    [XDG_DATA_HOME or ~/.workspace-qdrant] or custom         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Cache location?                                          │
│    [XDG_CACHE_HOME or ~/.workspace-qdrant] or custom        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Qdrant URL? (Native and Partial Docker only)             │
│    [http://localhost:6333] or custom                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ DOCKER ONLY: Additional prompts                             │
├─────────────────────────────────────────────────────────────┤
│ 7. Project paths to mount? (can add multiple)               │
│    Example: /home/user/projects                             │
├─────────────────────────────────────────────────────────────┤
│ 8. Library paths to mount? (can add multiple)               │
│    Example: /home/user/docs                                 │
├─────────────────────────────────────────────────────────────┤
│ 9. Tool paths for LSP access?                               │
│    [current $PATH] or reduce scope                          │
├─────────────────────────────────────────────────────────────┤
│ 10. Custom ports? (Full Docker only)                        │
│     gRPC: [50051], MCP HTTP: [50052], Qdrant: [6333]       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Installation actions:                                       │
│ - Download/extract binaries (Native) or pull images (Docker)│
│ - Create config file with chosen locations                  │
│ - Register user service with OS                             │
│ - Update PATH if needed                                     │
│ - Start daemon                                              │
│ - Verify installation                                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Success! Next steps:                                        │
│ - wqm status (verify daemon running)                        │
│ - wqm init <shell> (set up completions)                     │
│ - Configure Claude Code to use MCP server                   │
└─────────────────────────────────────────────────────────────┘
```

### Docker-Specific Configuration

**Full Docker mode creates:**
- `docker-compose.yml` with Qdrant, memexd, and MCP server
- Volume mounts for config, state, cache, projects, libraries
- PATH mounts for LSP tool access
- Port mappings for gRPC (50051), MCP HTTP (50052), optionally Qdrant (6333)

**Partial Docker mode creates:**
- `docker-compose.yml` with memexd only
- Volume mounts for config, state, cache, projects, libraries
- PATH mounts for LSP tool access
- Port mapping for gRPC (50051) only

**LSP access in Docker:**
- Host tool paths are mounted into container
- Container's PATH is configured to include mounted tool paths
- Default: mount paths from user's current `$PATH`
- User can reduce scope to specific directories

### Configuration File

The install script creates the config file at the chosen location with:

```yaml
# workspace-qdrant-mcp configuration
# Generated by install script on YYYY-MM-DD

deployment:
  mode: native  # native | partial-docker | full-docker

paths:
  config: /path/to/config
  state: /path/to/state
  cache: /path/to/cache

qdrant:
  url: http://localhost:6333

daemon:
  grpc_port: 50051

# Docker-specific (only present in Docker modes)
docker:
  mcp_http_port: 50052  # Full Docker only
  project_mounts:
    - /home/user/projects
    - /home/user/work
  library_mounts:
    - /home/user/docs
  tool_paths:
    - /usr/local/bin
    - /home/user/.cargo/bin
```

### Uninstall Process

Uninstallation is **documented manual steps**, not a script:

**Native:**
1. Stop service: `wqm service stop`
2. Remove service registration (platform-specific commands documented)
3. Remove binaries from installation directory
4. Remove config/state/cache directories
5. Remove PATH entries if added

**Docker:**
1. Stop containers: `docker-compose down`
2. Remove Docker images
3. Remove config/state/cache directories
4. Remove docker-compose.yml

### Post-Install Updates

After initial installation, `wqm update` handles all updates:

**Native mode:**
- Downloads new binaries from GitHub releases
- Replaces existing binaries
- Restarts daemon if running

**Docker modes:**
- Pulls new images
- Rebuilds containers with `docker-compose up --build -d`
- Preserves mounted volumes (config, state, data)

### Install Script Implementation

The install script will be:
- **Bash script** for macOS/Linux
- **PowerShell script** for Windows
- Available via curl one-liner: `curl -fsSL https://get.workspace-qdrant.dev | bash`

Alternative: Download and run manually for users who don't trust piped scripts.

---

## Appendix A: Command Quick Reference

```
wqm service start|stop|restart|status
wqm status [queue|watch|performance|health|live]
wqm library list|add|ingest|watch|unwatch|remove|config
wqm project list|info|remove
wqm queue list|show|stats
wqm language list|ts-install|ts-remove|lsp-install|lsp-remove|status
wqm memory list|add|remove|update|search|scope
wqm search project|library|memory|global
wqm update [--check] [--force] [--component]
wqm debug logs|errors|queue-errors|language
wqm backup [--collection] [--output]
wqm restore --snapshot [--collection]
wqm init bash|zsh|fish
```
