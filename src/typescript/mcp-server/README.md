# workspace-qdrant-mcp

MCP server providing project-scoped Qdrant vector database operations with hybrid search capabilities.

## Features

- **4 MCP Tools**: `search`, `retrieve`, `memory`, `store`
- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) vectors using Reciprocal Rank Fusion
- **Project-Scoped**: Automatic project detection with Git integration
- **Behavioral Rules**: Persistent memory collection for AI guidance
- **Graceful Degradation**: Works with or without the Rust daemon (memexd)

## Requirements

- **Node.js**: 18.0.0 or later
- **Qdrant**: Vector database instance (local or cloud)
- **Optional**: Rust daemon (memexd) for enhanced performance

## Installation

### From npm (when published)

```bash
npm install -g workspace-qdrant-mcp
```

### From Source

```bash
# Clone the repository
git clone https://github.com/ChrisGVE/workspace-qdrant-mcp.git
cd workspace-qdrant-mcp/src/typescript/mcp-server

# Install dependencies and build
npm install
npm run build

# Option 1: Link globally (for development)
npm link

# Option 2: Use the install script
./scripts/install.sh --link   # For development
./scripts/install.sh --global # For global installation
```

## Configuration

### Claude Desktop

Add the following to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

#### Option 1: Using absolute path to node

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "node",
      "args": ["/absolute/path/to/workspace-qdrant-mcp/src/typescript/mcp-server/dist/index.js"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

#### Option 2: Using npm-linked global command

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

#### Option 3: Using npx (without installation)

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "npx",
      "args": ["workspace-qdrant-mcp"],
      "env": {
        "QDRANT_URL": "http://localhost:6333"
      }
    }
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Qdrant API key (required for Qdrant Cloud) | - |
| `DAEMON_GRPC_PORT` | Rust daemon gRPC port | `50051` |
| `WQM_CLI_MODE` | Run in CLI mode (disables stdio) | `false` |
| `WQM_HTTP_MODE` | Run in HTTP mode | `false` |
| `WQM_CONFIG_PATH` | Custom configuration file path | - |

### Configuration File

The server looks for configuration in these locations (in order):

1. `$WQM_CONFIG_PATH` (if set)
2. `./.workspace-qdrant/config.yaml` (project-local)
3. `~/.workspace-qdrant/config.yaml` (user-global)
4. Built-in defaults

Example `config.yaml`:

```yaml
qdrant:
  url: http://localhost:6333
  # apiKey: your-api-key  # For Qdrant Cloud

daemon:
  grpcPort: 50051

database:
  path: ~/.workspace-qdrant/state.db

collections:
  memoryCollectionName: memory
```

## MCP Tools

### search

Search for documents using hybrid semantic and keyword search.

```json
{
  "query": "authentication implementation",
  "collection": "projects",
  "mode": "hybrid",
  "scope": "project",
  "limit": 10
}
```

**Parameters:**
- `query` (required): Search query text
- `collection`: `projects` | `libraries` | `memory` (default: `projects`)
- `mode`: `hybrid` | `semantic` | `keyword` (default: `hybrid`)
- `scope`: `project` | `global` | `all` (default: `project`)
- `limit`: Maximum results (default: 10)
- `projectId`: Specific project ID
- `libraryName`: Library name (for libraries collection)
- `branch`: Filter by Git branch
- `fileType`: Filter by file type

### retrieve

Retrieve documents by ID or metadata filter.

```json
{
  "documentId": "abc123",
  "collection": "projects"
}
```

**Parameters:**
- `documentId`: Document ID to retrieve
- `collection`: `projects` | `libraries` | `memory` (default: `projects`)
- `filter`: Metadata key-value pairs for filtering
- `limit`: Maximum results (default: 10)
- `offset`: Pagination offset (default: 0)

### memory

Manage behavioral rules for AI guidance.

```json
{
  "action": "add",
  "content": "Always use TypeScript strict mode",
  "scope": "global",
  "title": "Strict Mode Rule",
  "tags": ["typescript", "best-practices"]
}
```

**Parameters:**
- `action` (required): `add` | `update` | `remove` | `list`
- `content`: Rule content (required for add/update)
- `ruleId`: Rule ID (required for update/remove)
- `scope`: `global` | `project` (default: `global`)
- `projectId`: Project ID for project-scoped rules
- `title`: Rule title
- `tags`: Array of categorization tags
- `priority`: Rule priority (higher = more important)

### store

Store content to collections.

```json
{
  "content": "Document content to store",
  "collection": "projects",
  "title": "My Document",
  "sourceType": "user_input"
}
```

**Parameters:**
- `content` (required): Content to store
- `collection`: `projects` | `libraries` (default: `projects`)
- `title`: Content title
- `url`: Source URL (for web content)
- `filePath`: Source file path
- `sourceType`: `user_input` | `web` | `file` | `scratchbook` | `note`
- `libraryName`: Library name (required for libraries collection)
- `branch`: Git branch
- `fileType`: File type/extension
- `metadata`: Additional metadata object

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP Protocol (stdio)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              workspace-qdrant-mcp (TypeScript)               │
│  ┌─────────┐ ┌──────────┐ ┌────────┐ ┌───────┐             │
│  │ search  │ │ retrieve │ │ memory │ │ store │             │
│  └────┬────┘ └────┬─────┘ └───┬────┘ └───┬───┘             │
│       │           │           │          │                  │
│  ┌────┴───────────┴───────────┴──────────┴────┐            │
│  │              Tool Handlers                  │            │
│  └────────────────────┬───────────────────────┘            │
│                       │                                     │
│  ┌────────────────────┴───────────────────────┐            │
│  │         Session & Health Management         │            │
│  └────────────────────┬───────────────────────┘            │
└───────────────────────┼─────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌───────────────┐ ┌───────────┐ ┌───────────────┐
│ Rust Daemon   │ │  Qdrant   │ │    SQLite     │
│   (memexd)    │ │  Server   │ │   State DB    │
│   [gRPC]      │ │  [REST]   │ │   [local]     │
└───────────────┘ └───────────┘ └───────────────┘
```

### Collections (per ADR-001)

- **projects**: Multi-tenant by `tenant_id` - all project content
- **libraries**: Multi-tenant by `library_name` - external documentation
- **memory**: Multi-tenant by `project_id` (nullable for global) - behavioral rules

### Write Path (per ADR-002)

All writes route through the Rust daemon when available:
1. Try daemon gRPC `ingestText` RPC
2. On failure, queue to `unified_queue` SQLite table
3. Daemon processes queue on startup/reconnection

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Run in development mode
npm run dev

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Type check
npm run typecheck

# Lint
npm run lint

# Format
npm run format
```

## Health Status

The server monitors system health and reports status in search results:

- **healthy**: Both daemon and Qdrant available
- **uncertain**: One or both services unavailable (results may be incomplete)

When uncertain, search responses include health metadata:

```json
{
  "results": [...],
  "health": {
    "status": "uncertain",
    "reason": "daemon_unavailable",
    "message": "Daemon is unavailable. Search results may use cached data."
  }
}
```

## Troubleshooting

### Server won't start

1. Check Node.js version: `node --version` (must be 18+)
2. Verify Qdrant is running: `curl http://localhost:6333/health`
3. Check configuration path and syntax

### Search returns no results

1. Verify content was indexed (check daemon logs)
2. Check collection name is correct
3. Try broader search scope (`scope: "all"`)

### Daemon connection fails

The server works without the daemon in degraded mode:
- Read operations work directly with Qdrant
- Write operations queue to SQLite for later processing

## License

MIT
