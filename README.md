# workspace-qdrant-mcp

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/workspace-qdrant-mcp)](https://github.com/ChrisGVE/workspace-qdrant-mcp/releases)
[![Glama](https://glama.ai/mcp/servers/ChrisGVE/workspace-qdrant-mcp/badges/score.svg)](https://glama.ai/mcp/servers/ChrisGVE/workspace-qdrant-mcp)
[![Homebrew](https://img.shields.io/badge/Homebrew-tap-orange.svg)](https://github.com/ChrisGVE/homebrew-tap)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech)

Project-scoped vector database for AI assistants, providing hybrid semantic + keyword search with automatic project detection.

> ### 🚧 v0.2.0 rebuild in progress
>
> **workspace-qdrant-mcp is being rebuilt from the ground up in preparation for v0.2.0** —
> a unified storage model, better search quality, more reliable file watching, and a cleaner
> architecture, with a no-re-index migration for existing users. Once the design is locked
> (targeted early July) we'll open the work to outside contributors. See the
> **[Roadmap](ROADMAP.md)** for the top-line plan.

## Features

- **Hybrid Search** - Combines semantic similarity with keyword matching using Reciprocal Rank Fusion
- **Project Detection** - Automatic Git repository awareness and project-scoped collections
- **7 MCP Tools** - search, retrieve, rules, store, grep, list, embedding
- **Code Intelligence** - Tree-sitter semantic chunking + LSP integration for active projects
- **Code Graph** - Relationship graph with algorithms (PageRank, community detection, betweenness centrality)
- **High-Performance CLI** - Rust-based `wqm` command-line tool
- **Background Daemon** - `memexd` for continuous file monitoring and processing

## Quick Start

### Prerequisites

- **Qdrant** - `docker run -d -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant`
- **C compiler** - Required for compiling Tree-sitter grammars on first use. Tree-sitter grammars are distributed as C source and compiled locally.
  - **macOS**: `xcode-select --install` (Xcode Command Line Tools)
  - **Linux**: `apt install build-essential` (Debian/Ubuntu) or `dnf groupinstall "Development Tools"` (Fedora)
  - **Windows**: Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with C++ workload
- **Clang/LLVM** - Required only to **build `memexd` from source**, for the LadybugDB C++ core (the default graph backend). Pre-built binaries (Homebrew, release artifacts) do not need it.
  - **macOS**: Xcode Command Line Tools include Clang (`xcode-select --install`)
  - **Linux**: `apt install clang libclang-dev` (Debian/Ubuntu) or `dnf install clang` (Fedora)
  - **Alternative**: build without the C++ toolchain using the SQLite-only backend — `cargo build --no-default-features --features sqlite`

### Install

**Option 1: Homebrew (Recommended — macOS & Linux)**

```bash
brew install ChrisGVE/tap/workspace-qdrant
brew services start workspace-qdrant
```

**Option 2: Pre-built Binaries**

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

Installs `wqm`, `memexd`, and `workspace-qdrant-mcp` to `~/.local/bin` (Linux/macOS) or `%LOCALAPPDATA%\wqm\bin` (Windows).

**Option 3: Build from Source**

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
./install.sh
```

See [Installation Reference](docs/reference/installation.md) for detailed instructions and platform-specific notes. For Windows, see the [Windows Installation Guide](docs/reference/windows-installation.md).

### Configure MCP

**Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add workspace-qdrant-mcp -- workspace-qdrant-mcp
```

### Verify

```bash
wqm --version
wqm status health
```

### CLAUDE.md Integration

Add the following to your project's `CLAUDE.md` (or your global `~/.claude/CLAUDE.md`) so Claude Code uses workspace-qdrant proactively:

````markdown
## workspace-qdrant

The `workspace-qdrant` MCP server provides codebase-aware search, a library knowledge base, a scratchpad for accumulated insights, and persistent behavioral rules. The tool schemas are self-describing; these instructions cover *when* and *how* to use them.

### Primary Search and Knowledge Base

**Use `workspace-qdrant` first whenever context is uncertain** — first session on a project, returning after a significant gap, or exploring an unfamiliar subsystem. It is faster and more accurate than walking files manually, and it retrieves findings from prior sessions that would otherwise be lost.

**Three-step protocol:**
1. **Search** with `workspace-qdrant` (`search`, `grep`, `list`, or `retrieve`)
2. **Fall back** to `Grep`, `Glob`, `WebSearch` only when workspace-qdrant is insufficient or unavailable
3. **Store** any new findings, analysis, or design rationale via `store` so they are retrievable in future sessions

When a fresh handover or strong prior context already covers what you need, skip the exploratory search — but always store new findings at the end.

**Collections and their purpose:**
- `projects` — indexed codebase; use `scope="project"` (current project) or `scope="all"` (across all projects)
- `libraries` — external reference docs, API specs, third-party documentation; add via `store` with `collection="libraries"` and search with `includeLibraries=true`
- `scratchpad` — analysis, design rationale, research transcripts, architectural insights; complements session handovers by building a growing, semantically searchable knowledge layer across sessions
- `rules` — persistent behavioral rules; load at session start via `rules` → `action="list"`

**Practical notes:**
- Use `grep` for exact strings or regex; `list` with `format="summary"` to explore project structure
- Store external docs or specs into `libraries` so they are searchable alongside code
- Use the scratchpad to record *why* decisions were made, not just *what* was done — future sessions can retrieve the reasoning

### Sub-Agents

Sub-agents start with only the prompt you give them — they have no session history or handover context. They must always use `workspace-qdrant` first for any code exploration, without exception. Include this verbatim in every agent prompt:

> "You have no prior context about this codebase. Use `workspace-qdrant` as your mandatory first tool for ALL code searches — symbols, functions, architecture, patterns, prior findings. Use `search`, `grep`, `list`, or `retrieve` before touching any file with Read/Grep/Glob. Store any new findings, analysis, or design rationale via `store` (scratchpad for insights, libraries for reference docs) so they persist for future sessions."

### Project Registration

At session start, check whether the current project is registered with workspace-qdrant. If it is not, ask the user whether they want to register it (do not register silently). Once registered, the daemon handles file watching and ingestion automatically — no further action is needed.

### Behavioral Rules

The `rules` tool manages persistent rules that are injected into context across sessions. Rules are **user-initiated only** — add rules when the user explicitly instructs you to, never autonomously. Use `action="list"` at session start to load active rules.

### Issue Reporting

workspace-qdrant is under active development. If you encounter errors, unexpected behavior, or limitations with any workspace-qdrant tool, report them as GitHub issues at https://github.com/ChrisGVE/workspace-qdrant-mcp/issues using the `gh` CLI.
````

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search` | Hybrid semantic + keyword search across indexed content |
| `retrieve` | Direct document lookup by ID or metadata filter |
| `rules` | Manage persistent behavioral rules |
| `store` | Store content, register projects, save notes |
| `grep` | Exact substring or regex search using FTS5 |
| `list` | List project files and folder structure |

See [MCP Tools Reference](docs/reference/mcp-tools.md) for parameters and examples.

## Collections

| Collection | Purpose | Isolation |
|------------|---------|-----------|
| `projects` | Project code and documentation | Multi-tenant by `tenant_id` |
| `libraries` | Reference documentation (books, papers, docs) | Multi-tenant by `library_name` |
| `rules` | Behavioral rules and preferences | Multi-tenant by `project_id` |
| `scratchpad` | Temporary working storage | Per-session |

## CLI Reference

```bash
# Service management
wqm service start              # Start background daemon
wqm service status             # Check daemon status
wqm status health              # System health check

# Search and content
wqm search "query"             # Search collections
wqm ingest file path.py        # Ingest a file
wqm rules list                 # List behavioral rules

# Project and library
wqm project list               # List registered projects
wqm project watch pause        # Pause file watchers
wqm library list               # List libraries
wqm tags list                  # List tags with counts

# Administration
wqm admin collections list     # List collections
wqm admin rebuild all          # Rebuild all indexes
wqm admin backup create        # Backup snapshots
wqm admin stats overview       # Search analytics

# Code graph
wqm graph stats --tenant <t>   # Node/edge counts
wqm graph query --node-id <id> --tenant <t> --hops 2   # Related nodes
wqm graph impact --symbol <name> --tenant <t>           # Impact analysis
wqm graph pagerank --tenant <t> --top-k 20              # PageRank centrality

# Setup
wqm init completions zsh       # Shell completions
wqm init man install           # Install man pages
wqm init hooks install         # Install Claude Code hooks (respects CLAUDE_CONFIG_DIR)

# Queue and monitoring
wqm queue stats                # Queue statistics
```

See [CLI Reference](docs/reference/cli.md) for complete documentation.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | - | API key (required for Qdrant Cloud) |
| `FASTEMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |

### Claude Code Integration

`wqm init hooks` reads and writes Claude Code's `settings.json`. The
location is resolved from:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CONFIG_DIR` | `~/.claude` | Claude Code config directory used by `wqm init hooks install/uninstall/status`. Set this for Claude Code Enterprise or any non-default install. |

Example — Claude Code Enterprise:

```bash
export CLAUDE_CONFIG_DIR=~/.config/claude/claude-ent
wqm init hooks install
```

## Observability

The daemon exposes metrics and traces. Both are disabled by default.

### Prometheus (`/metrics`, pull)

Enable via config or env var, then scrape:

```yaml
# in the daemon config
observability:
  telemetry:
    prometheus:
      enabled: true
      port: 9464
      bind: 0.0.0.0
```

or:

```bash
WQM_PROMETHEUS_ENABLED=true WQM_PROMETHEUS_PORT=9464 memexd --foreground
curl http://localhost:9464/metrics | head
```

The `--metrics-port <N>` CLI flag is a shortcut that forces
`enabled=true` and overrides the port. See
`docs/observability/prometheus-scrape-example.yaml` for a
`scrape_configs` snippet and
`docs/observability/memexd-telemetry-dashboard.json` for a Grafana 10
dashboard.

### OTLP traces (push)

`#[tracing::instrument]` spans on the queue processor, watcher, gRPC,
embedding, and Qdrant paths are exported over OTLP/gRPC when:

```yaml
observability:
  telemetry:
    service_name: memexd
    otlp:
      enabled: true
      endpoint: http://collector.example:4317
      protocol: grpc   # http/protobuf is also recognized (logs a warning)
      sample_rate: 0.1
```

Standard OpenTelemetry env vars are honored: `OTEL_SERVICE_NAME`,
`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_PROTOCOL`,
`OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_TRACES_SAMPLER_ARG`.

OTLP metrics export is **not** currently implemented — Prometheus is
the canonical metrics surface.

## Architecture

```
                    +-----------------+
                    |  Claude/Client  |
                    +--------+--------+
                             |
                    +--------v--------+
                    |   MCP Server    |  (TypeScript)
                    +--------+--------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+           +--------v--------+
     |   Rust Daemon   |           |     Qdrant      |
     |    (memexd)     |           | Vector Database |
     +--------+--------+           +-----------------+
              |
     +--------v--------+
     |  File Watcher   |
     |  Code Graph     |
     |  Embeddings     |
     +-----------------+
```

The Rust daemon handles file watching, embedding generation, code graph extraction, and queue processing. All writes route through the daemon for consistency.

## Documentation

**User guides:**
- [Quick Start](docs/quick-start.md) — get running in 5 minutes
- [User Manual](docs/user-manual.md) — full usage guide
- [LLM Integration](docs/reference/mcp-best-practices.md) — best practices for Claude

**Reference:**
- [Installation](docs/reference/installation.md) | [Windows](docs/reference/windows-installation.md)
- [CLI Reference](docs/reference/cli.md) — all `wqm` commands
- [MCP Tools](docs/reference/mcp-tools.md) — tool parameters and examples
- [Configuration](docs/reference/configuration.md) — all options and defaults
- [Architecture](docs/reference/architecture.md) — component overview

See the [Documentation Index](docs/INDEX.md) for specifications, ADRs, and developer resources.

## Development

```bash
# Rust daemon, CLI, and MCP server (from src/rust/)
# Builds memexd (daemon), wqm (CLI), and workspace-qdrant-mcp (MCP server)
cargo build --release
cargo test

# Graph benchmarks
cargo bench --package workspace-qdrant-core --bench graph_bench

# Binaries output to:
# - target/release/wqm
# - target/release/memexd
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

*Inspired by [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp)*
