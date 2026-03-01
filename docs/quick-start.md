# Quick Start

Get workspace-qdrant-mcp running in 5 minutes.

## 1. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

## 2. Install

```bash
# macOS / Linux
curl -fsSL https://raw.githubusercontent.com/ChrisGVE/workspace-qdrant-mcp/main/scripts/download-install.sh | bash
```

This installs `wqm` (CLI) and `memexd` (daemon) to `~/.local/bin`.

## 3. Start the daemon

```bash
wqm service install   # One-time: install as system service
wqm service start     # Start background daemon
wqm admin health      # Verify: should show "healthy"
```

## 4. Register a project

```bash
wqm project add /path/to/your/project
```

The daemon automatically watches the directory, detects files, and indexes them. Check progress:

```bash
wqm queue stats       # Watch queue drain to 0
wqm project list      # See registered projects
```

## 5. Search

```bash
wqm search "authentication middleware"           # Semantic search
wqm search "handleRequest" --exact               # Exact match
wqm search "error handling" --collection libraries  # Search docs
```

## 6. Connect to Claude

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["<install-path>/src/typescript/mcp-server/dist/index.js"],
      "env": { "QDRANT_URL": "http://localhost:6333" }
    }
  }
}
```

**Claude Code:**

```bash
claude mcp add workspace-qdrant -- node <install-path>/src/typescript/mcp-server/dist/index.js
```

## Next steps

- [User Manual](user-manual.md) — detailed usage guide
- [CLI Reference](reference/cli.md) — all `wqm` commands
- [MCP Tools Reference](reference/mcp-tools.md) — tool parameters and examples
