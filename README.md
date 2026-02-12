# workspace-qdrant-mcp

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech)

Project-scoped vector database for AI assistants, providing hybrid semantic + keyword search with automatic project detection.

## Features

- **Hybrid Search** - Combines semantic similarity with keyword matching using Reciprocal Rank Fusion
- **Project Detection** - Automatic Git repository awareness and project-scoped collections
- **4 MCP Tools** - store, search, manage, retrieve for complete vector operations
- **High-Performance CLI** - Rust-based `wqm` command-line tool
- **Background Daemon** - `memexd` for continuous file monitoring and processing

## Quick Start

### Prerequisites

- **Qdrant server** - `docker run -p 6333:6333 qdrant/qdrant`

For MCP server only:
- **Python 3.10+** with [uv](https://github.com/astral-sh/uv)

For CLI + daemon (from source):
- **Rust 1.75+** - [rustup.rs](https://rustup.rs)
- **ONNX Runtime** (for daemon) - `brew install onnxruntime` or [download](https://github.com/microsoft/onnxruntime/releases)

### Install

**Option 1: Pre-built Binaries (Recommended for CLI + Daemon)**

```bash
# Linux/macOS - one-liner
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash

# Windows (PowerShell)
irm https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.ps1 | iex
```

This downloads pre-built `wqm` and `memexd` binaries to `~/.local/bin` (or `%LOCALAPPDATA%\wqm\bin` on Windows).

Options (run script locally):
- `--prefix /path` - Custom installation directory
- `--version v0.4.0` - Specific version (default: latest)
- `--cli-only` - Skip daemon

**Option 2: MCP Server Only (Python)**

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
uv tool install .
```

This installs the `workspace-qdrant-mcp` command globally. No Rust toolchain required.

**Option 3: Build from Source**

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
./install.sh
```

The installer builds Rust binaries (`wqm`, `memexd`) to `~/.local/bin` and installs Python dependencies.

Options:
- `--prefix /path` - Custom installation directory
- `--force` - Clean rebuild from scratch
- `--cli-only` - Skip daemon build

For Windows: `.\install.ps1`

### Configure MCP

**Claude Desktop** (`claude_desktop_config.json`):

If installed via `uv tool install`:
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

If using from source directory:
```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "uv",
      "args": ["--directory", "/path/to/workspace-qdrant-mcp", "run", "workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

**Claude Code**:

```bash
# If installed via uv tool install
claude mcp add workspace-qdrant-mcp -- workspace-qdrant-mcp

# If using from source directory
claude mcp add workspace-qdrant-mcp -- uv --directory /path/to/workspace-qdrant-mcp run workspace-qdrant-mcp
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

Use the `workspace-qdrant` MCP server proactively as your project knowledge base:

- **`search`**: Search project code, library docs, or memory rules with hybrid semantic + keyword matching. Use `scope="project"` for current project, `scope="all"` to include libraries.
- **`store`**: Register projects for file watching (`type="project"`, `path="/absolute/path"`) or store library reference docs (`type="library"`, `libraryName="...", content="..."`).
- **`memory`**: Manage persistent behavioral rules that survive across sessions. Use `action="add"` with a short `label` (e.g., `"prefer-uv"`) and `content` describing the rule. Use `action="list"` to review existing rules.
- **`retrieve`**: Fetch specific documents by ID or metadata filter from `projects`, `libraries`, or `memory` collections.

When starting a session, search for relevant context. When discovering user preferences or project conventions, store them as memory rules.
````

## MCP Tools

### store

Store content in the vector database with automatic embedding generation.

```
store(content="Your text here", collection="myapp-notes")
```

### search

Hybrid semantic + keyword search across collections.

```
search(query="authentication", scope="project")
search(query="JWT tokens", scope="all", include_libraries=true)
```

Parameters:
- `query` - Search text (required)
- `mode` - `"hybrid"` (default), `"semantic"`, or `"exact"`
- `scope` - `"project"` (default), `"global"`, or `"all"`
- `include_libraries` - Include library docs (default: false)
- `limit` - Max results (default: 10)

### manage

Collection and system management.

```
manage(action="list_collections")
manage(action="get_status")
manage(action="create_collection", collection_name="myproject-notes")
```

### retrieve

Direct document access by ID or metadata.

```
retrieve(document_id="abc123", collection="myapp-notes")
```

## Collection Types

| Type | Pattern | Example | Purpose |
|------|---------|---------|---------|
| PROJECT | `_{project_id}` | `_a1b2c3d4e5f6` | Auto-created for file watching |
| USER | `{name}-{type}` | `myapp-notes` | User notes and scratchbooks |
| LIBRARY | `_{library_name}` | `_numpy` | External documentation |
| MEMORY | `_memory` | `_memory` | Agent behavioral rules |

See [Collection Naming Guide](docs/COLLECTION_NAMING.md) for details.

## CLI Reference

```bash
wqm service start          # Start background daemon
wqm service status         # Check daemon status
wqm admin collections      # List all collections
wqm admin health           # System health check
wqm queue stats            # Queue statistics
wqm search "query"         # Search collections
wqm ingest file path.py    # Ingest a file
```

See [CLI Reference](docs/CLI.md) for complete documentation.

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
                    |   MCP Server    |  (Python - FastMCP)
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
     |   File Watcher  |
     +-----------------+
```

The Rust daemon handles file watching, embedding generation, and queue processing. All writes route through the daemon for consistency.

## Documentation

- [API Reference](docs/API.md) - MCP tools documentation
- [CLI Reference](docs/CLI.md) - Command-line interface
- [Architecture](docs/ARCHITECTURE.md) - System design
- [Collection Naming](docs/COLLECTION_NAMING.md) - Naming conventions
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues

## Development

```bash
# Python tests
uv run pytest

# Rust build (from src/rust/)
cargo build --release

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
