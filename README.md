# workspace-qdrant-mcp

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech)
[![FastMCP](https://img.shields.io/badge/FastMCP-0.3%2B-orange.svg)](https://github.com/jlowin/fastmcp)

**Project-scoped Qdrant MCP server with hybrid search and universal scratchbook**

*Inspired by [claude-qdrant-mcp](https://github.com/example/claude-qdrant-mcp) with enhanced project detection and scratchbook functionality.*

workspace-qdrant-mcp provides intelligent vector database operations through the Model Context Protocol (MCP), featuring automatic project detection, hybrid search capabilities, and cross-project scratchbook management.

**Key Features** • **Project-Aware Collections** • **Hybrid Search (97.1% precision)** • **Universal Scratchbook** • **MCP Integration**

## Prerequisites

**Qdrant server is required** - workspace-qdrant-mcp connects to Qdrant for vector operations.

```bash
# Start Qdrant with Docker (recommended)
docker run -p 6333:6333 qdrant/qdrant
```

Visit the [Qdrant repository](https://github.com/qdrant/qdrant) for installation alternatives.

## Installation

### Quick Install (Recommended)

```bash
# Install globally with uv
uv tool install workspace-qdrant-mcp

# Or with pip
pip install workspace-qdrant-mcp
```

### Development Installation

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development setup instructions.

## MCP Integration

### Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "GITHUB_USER": "your-username"
      }
    }
  }
}
```

### Claude Code

workspace-qdrant-mcp works seamlessly with Claude Code's MCP integration.

```json
{
  "mcp": {
    "servers": {
      "workspace-qdrant": {
        "command": "workspace-qdrant-mcp",
        "args": ["--port", "8000"]
      }
    }
  }
}
```

### Parameter Configuration

**Required environment variables:**
- `QDRANT_URL`: Qdrant server URL (default: `http://localhost:6333`)

**Optional environment variables:**
- `GITHUB_USER`: Filter project detection to your repositories
- `COLLECTIONS`: Comma-separated list (default: `"scratchbook"`)

**Example configurations:**
```bash
# Single collection per project
COLLECTIONS="scratchbook" workspace-qdrant-mcp

# Multiple collections per project  
COLLECTIONS="scratchbook,docs,notes" workspace-qdrant-mcp
# Creates: project-scratchbook, project-docs, project-notes
```

## Basic Usage

### Start MCP Server

```bash
# Default configuration (port 8000)
workspace-qdrant-mcp

# Custom configuration
workspace-qdrant-mcp --host 0.0.0.0 --port 8001 --debug
```

### Verify Setup

```bash
# Validate configuration
workspace-qdrant-validate

# Check server health
curl http://localhost:8000/health
```

## Configuration Options

### Environment Variables

Create `.env` file in your project:

```bash
# Qdrant Configuration (required)
QDRANT_URL=http://localhost:6333

# Project Detection (optional)
GITHUB_USER=your-username

# Collection Configuration (optional)
COLLECTIONS=scratchbook,docs

# Server Configuration (optional)
WORKSPACE_QDRANT_HOST=127.0.0.1
WORKSPACE_QDRANT_PORT=8000
WORKSPACE_QDRANT_DEBUG=false
```

### Collection Configuration

workspace-qdrant-mcp creates collections based on your project:

**Default behavior:**
- Single collection: `{project-name}-scratchbook`

**Multiple collections:**
- Configure: `COLLECTIONS="scratchbook,docs,notes"`
- Creates: `{project-name}-scratchbook`, `{project-name}-docs`, `{project-name}-notes`

**Project detection:**
- Auto-detects Git repository name
- Filters by `GITHUB_USER` if specified
- Handles submodules and nested projects

## CLI Tool (wqutil)

The `wqutil` (workspace-qdrant utility) command provides collection management:

```bash
# List workspace collections
wqutil list-collections

# Show collection details
wqutil collection-info my-project-scratchbook

# Safe collection deletion (with confirmation)
wqutil delete-collection old-project-docs

# Show help
wqutil --help
```

**Legacy compatibility:** `workspace-qdrant-admin` command still available.

## Quick Examples

### Search Your Workspace

```python
# Via Python MCP client
results = await mcp_call("search_workspace", {
    "query": "vector database implementation",
    "mode": "hybrid",  # Best results (97.1% precision)
    "limit": 10
})
```

### Manage Scratchbook Notes

```python
# Add a note
await mcp_call("update_scratchbook", {
    "content": "Remember to optimize the search algorithm",
    "metadata": {"category": "todo", "priority": "high"}
})

# Search notes across all projects
results = await mcp_call("search_scratchbook", {
    "query": "search algorithm optimization"
})
```

### Add Documentation

```python
# Add document to project collection
await mcp_call("add_document", {
    "content": "API documentation content...",
    "collection": "my-project-docs",
    "metadata": {"type": "api-docs", "version": "1.0"}
})
```

## Documentation

- **[API Reference](API.md)** - Comprehensive API documentation
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and guidelines
- **[Benchmarking](benchmarking/README.md)** - Performance metrics and testing

## Troubleshooting

### Common Issues

**Cannot connect to Qdrant:**
```bash
# Check Qdrant is running
curl http://localhost:6333/collections

# Start Qdrant if needed
docker run -p 6333:6333 qdrant/qdrant
```

**Project not detected:**
```bash
# Check Git configuration
git remote -v

# Set GitHub user for better detection
GITHUB_USER=your-username workspace-qdrant-mcp
```

**Collections not created:**
```bash
# Validate setup
workspace-qdrant-validate --verbose

# Check server logs
workspace-qdrant-mcp --debug
```

### Debug Mode

```bash
# Enable comprehensive logging
workspace-qdrant-mcp --debug

# Validate configuration with details
workspace-qdrant-validate --verbose --fix
```

### Get Help

- **[GitHub Issues](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions)** - Questions and community
- **[Documentation](https://github.com/ChrisGVE/workspace-qdrant-mcp/wiki)** - Detailed guides

## License

MIT License - see [LICENSE](LICENSE) for details.

**Related Projects:**
- [FastMCP](https://github.com/jlowin/fastmcp) - Modern MCP server framework
- [Qdrant](https://github.com/qdrant/qdrant) - High-performance vector database
- [claude-qdrant-mcp](https://github.com/example/claude-qdrant-mcp) - Original inspiration

---

**Built with ❤️ by the workspace-qdrant-mcp community**