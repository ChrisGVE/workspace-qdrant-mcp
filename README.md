[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/chrisgve-workspace-qdrant-mcp-badge.png)](https://mseep.ai/app/chrisgve-workspace-qdrant-mcp)

# workspace-qdrant-mcp

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech)
[![FastMCP](https://img.shields.io/badge/FastMCP-0.3%2B-orange.svg)](https://github.com/jlowin/fastmcp)

**Project-scoped Qdrant MCP server with hybrid search and configurable collections**

*Inspired by [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) with enhanced project detection, Python implementation, and flexible collection management.*

workspace-qdrant-mcp provides intelligent vector database operations through the Model Context Protocol (MCP), featuring automatic project detection, hybrid search capabilities, and configurable collection management for seamless integration with Claude Desktop and Claude Code.

## Prerequisites

**Qdrant server must be running** - workspace-qdrant-mcp connects to Qdrant for vector operations.

- **Local**: Default `http://localhost:6333`
- **Cloud**: Requires `QDRANT_API_KEY` environment variable

For local installation, see the [Qdrant repository](https://github.com/qdrant/qdrant). For documentation examples, we assume the default local setup.

## Installation

```bash
# Install globally with uv (recommended)
uv tool install workspace-qdrant-mcp

# Or with pip
pip install workspace-qdrant-mcp
```

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

## MCP Integration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "COLLECTIONS": "project",
        "GLOBAL_COLLECTIONS": "docs,references"
      }
    }
  }
}
```

### Claude Code

```bash
claude mcp add workspace-qdrant-mcp
```

Configure environment variables through Claude Code's settings or your shell environment.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | _(none)_ | Required for Qdrant cloud, optional for local |
| `COLLECTIONS` | `project` | Collection suffixes (comma-separated) |
| `GLOBAL_COLLECTIONS` | _(none)_ | Global collection names (comma-separated) |
| `GITHUB_USER` | _(none)_ | Filter projects by GitHub username |
| `FASTEMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |

### Collection Naming

Collections are automatically created based on your project and configuration:

**Project Collections:**
- `COLLECTIONS="project"` → creates `{project-name}-project`
- `COLLECTIONS="scratchbook,docs"` → creates `{project-name}-scratchbook`, `{project-name}-docs`

**Global Collections:**
- `GLOBAL_COLLECTIONS="docs,references"` → creates `docs`, `references`

**Example:** For project "my-app" with `COLLECTIONS="scratchbook,docs"`:
- `my-app-scratchbook` (project-specific notes)
- `my-app-docs` (project-specific documentation)
- `docs` (global documentation)
- `references` (global references)

## Usage

Interact with your collections through natural language commands in Claude:

**Store Information:**
- "Store this note in my project scratchbook: [your content]"
- "Add this document to my docs collection: [document content]"

**Search & Retrieve:**
- "Search my project for information about authentication"
- "Find all references to the API endpoint in my scratchbook"
- "What documentation do I have about deployment?"

**Hybrid Search:**
- Combines semantic search (meaning-based) with keyword search (exact matches)
- Automatically optimizes results using reciprocal rank fusion (RRF)
- Searches across project and global collections

## CLI Tool

Use `wqutil` for collection management and diagnostics:

```bash
# List collections
wqutil list-collections

# Collection information  
wqutil collection-info my-project-scratchbook

# Validate configuration
workspace-qdrant-validate

# Check workspace status
wqutil workspace-status
```

*Note: Legacy `workspace-qdrant-admin` command is also available.*

## Documentation

- **[API Reference](API.md)** - Complete MCP tools documentation
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and guidelines
- **[Benchmarking](benchmarking/README.md)** - Performance testing and metrics

## Troubleshooting

**Connection Issues:**
```bash
# Verify Qdrant is running
curl http://localhost:6333/collections

# Test configuration
workspace-qdrant-validate
```

**Collection Issues:**
```bash
# List current collections
wqutil list-collections

# Check project detection
wqutil workspace-status
```

For detailed troubleshooting, see [API.md](API.md#troubleshooting).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Related Projects:**
- [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) - Original TypeScript implementation
- [Qdrant](https://qdrant.tech) - Vector database
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework