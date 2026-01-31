# workspace-qdrant-mcp (TypeScript)

MCP server providing project-scoped Qdrant vector database operations with hybrid search capabilities.

## Overview

This is the TypeScript implementation of the MCP server, rewritten from Python to leverage the Claude Code SDK's native `SessionStart` and `SessionEnd` hooks for proper session lifecycle management.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      TYPESCRIPT MCP SERVER                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │  MCP Application (TypeScript)                             │     │
│   │  - store: Content storage to libraries collection         │     │
│   │  - search: Hybrid semantic + keyword search               │     │
│   │  - memory: Behavioral rules management                    │     │
│   │  - retrieve: Direct document access                       │     │
│   └──────────────────────────────────────────────────────────┘     │
│         │                                                           │
│         │ gRPC (port 50051)                                         │
│         ▼                                                           │
│   ┌──────────────────┐                                             │
│   │  Rust Daemon     │  (memexd - handles all Qdrant writes)       │
│   └──────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Node.js 18+
- Running Qdrant instance (default: http://localhost:6333)
- Running Rust daemon (memexd) for write operations

## Installation

```bash
npm install
```

## Development

```bash
# Build
npm run build

# Development with watch mode
npm run dev

# Type checking
npm run typecheck

# Linting
npm run lint
npm run lint:fix

# Formatting
npm run format
npm run format:check

# Testing
npm run test
npm run test:watch
npm run test:coverage
```

## Configuration

Configuration is loaded from (in order):
1. `~/.workspace-qdrant/config.yaml`
2. `~/.config/workspace-qdrant/config.yaml`
3. `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS)

Environment variables override config file values:
- `QDRANT_URL` - Qdrant server URL
- `QDRANT_API_KEY` - Qdrant API key
- `WQM_DATABASE_PATH` - SQLite database path
- `WQM_DAEMON_PORT` - Daemon gRPC port

## MCP Tools

Exactly 4 tools are provided:

1. **search** - Semantic search with hybrid mode (semantic + keyword)
2. **retrieve** - Direct document access by ID or metadata
3. **memory** - Manage behavioral rules (add/update/remove/list)
4. **store** - Store content to libraries collection

## Project Structure

```
src/
├── clients/          # gRPC and Qdrant clients
├── tools/            # MCP tool implementations
├── types/            # TypeScript type definitions
├── utils/            # Utility functions
├── config.ts         # Configuration loading
└── index.ts          # Entry point
tests/
└── *.test.ts         # Test files
```

## Dependencies

- `@modelcontextprotocol/sdk` - MCP protocol with session hooks
- `@qdrant/js-client-rest` - Qdrant queries
- `better-sqlite3` - SQLite queue access
- `@grpc/grpc-js` - gRPC client for daemon communication

## License

MIT
