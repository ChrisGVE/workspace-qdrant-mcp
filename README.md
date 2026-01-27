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

- **Python 3.10+** with [uv](https://github.com/astral-sh/uv)
- **Rust toolchain** - [rustup.rs](https://rustup.rs)
- **Qdrant server** - `docker run -p 6333:6333 qdrant/qdrant`
- **ONNX Runtime** (for daemon) - `brew install onnxruntime` or [download](https://github.com/microsoft/onnxruntime/releases)

### Install

```bash
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp
./install.sh
```

The installer builds binaries to `~/.local/bin` by default. Use `--prefix /path` for custom location.

For Windows: `.\install.ps1`

### Configure MCP

**Claude Desktop** (`claude_desktop_config.json`):

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
claude mcp add workspace-qdrant-mcp -- uv --directory /path/to/workspace-qdrant-mcp run workspace-qdrant-mcp
```

### Verify

```bash
wqm --version
wqm admin health
```

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
