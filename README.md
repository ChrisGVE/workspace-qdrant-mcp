# workspace-qdrant-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ChrisGVE/workspace-qdrant-mcp)](https://github.com/ChrisGVE/workspace-qdrant-mcp/releases)
[![Glama](https://glama.ai/mcp/servers/ChrisGVE/workspace-qdrant-mcp/badges/score.svg)](https://glama.ai/mcp/servers/ChrisGVE/workspace-qdrant-mcp)
[![Homebrew](https://img.shields.io/badge/Homebrew-tap-orange.svg)](https://github.com/ChrisGVE/homebrew-tap)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0%2B-blue.svg)](https://www.typescriptlang.org/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech)

Project-scoped vector database for AI assistants, providing hybrid semantic + keyword search with automatic project detection.

## Features

- **Hybrid Search** - Combines semantic similarity with keyword matching using Reciprocal Rank Fusion
- **Project Detection** - Automatic Git repository awareness and project-scoped collections
- **6 MCP Tools** - search, retrieve, rules, store, grep, list
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

Installs `wqm` and `memexd` to `~/.local/bin` (Linux/macOS) or `%LOCALAPPDATA%\wqm\bin` (Windows).

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
      "command": "node",
      "args": ["/path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add workspace-qdrant-mcp -- node /path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js
```

### Verify

```bash
wqm --version
wqm admin health
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
wqm admin health               # System health check

# Search and content
wqm search "query"             # Search collections
wqm ingest file path.py        # Ingest a file
wqm rules list                 # List behavioral rules

# Project and library
wqm project list               # List registered projects
wqm library list               # List libraries
wqm tags list                  # List tags with counts

# Code graph
wqm graph stats --tenant <t>   # Node/edge counts
wqm graph query --node-id <id> --tenant <t> --hops 2   # Related nodes
wqm graph impact --symbol <name> --tenant <t>           # Impact analysis
wqm graph pagerank --tenant <t> --top-k 20              # PageRank centrality

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
# TypeScript MCP server
cd src/typescript/mcp-server && npm install && npm run build && npm test

# Rust daemon and CLI (from src/rust/)
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

MIT License - see [LICENSE](LICENSE) for details.

---

*Inspired by [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp)*
