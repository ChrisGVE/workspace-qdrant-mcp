# MCP Tools API Reference

Complete documentation for workspace-qdrant-mcp's Model Context Protocol (MCP) tools.

## Table of Contents

- [Overview](#overview)
- [MCP Tools](#mcp-tools)
  - [store](#store)
  - [search](#search)
  - [manage](#manage)
  - [retrieve](#retrieve)
- [Integration](#integration)
  - [Claude Desktop (stdio mode)](#claude-desktop-stdio-mode)
  - [HTTP Mode](#http-mode)
- [Troubleshooting](#troubleshooting)

## Overview

workspace-qdrant-mcp provides 4 comprehensive MCP tools that enable natural language interaction with your Qdrant vector database through Claude Desktop and Claude Code. These tools leverage the Model Context Protocol for seamless AI-assisted vector operations.

**Key Capabilities:**
- Store any content with automatic embedding generation
- Search using hybrid semantic + keyword matching
- Manage collections and monitor system health
- Retrieve documents by ID or metadata with filtering

**Architecture Features:**
- **Unified Multi-Tenant Model**: Only 4 collections total (`_projects`, `_libraries`, user, memory)
- **Tenant Isolation**: `tenant_id` for projects, `library_name` for libraries (payload-indexed)
- **Cross-Project Search**: Search all projects with `scope="all"`
- **Library Inclusion**: Include documentation with `include_libraries=true`
- Branch-aware querying with Git integration
- File type differentiation via metadata
- Reciprocal Rank Fusion (RRF) for optimal search results

## MCP Tools

### store

Store any type of content in the vector database with automatic embedding generation and metadata enrichment.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `content` | string | **Yes** | - | The text content to store |
| `title` | string | No | Auto-generated | Title for the document |
| `metadata` | object | No | `{}` | Additional metadata to attach |
| `collection` | string | No | Auto-detected | Override automatic collection selection |
| `source` | string | No | `"user_input"` | Source type (user_input, scratchbook, file, web, etc.) |
| `document_type` | string | No | `"text"` | Type of document (text, code, note, etc.) |
| `file_path` | string | No | `null` | Path to source file (influences collection choice) |
| `url` | string | No | `null` | Source URL (influences collection choice) |
| `project_name` | string | No | Auto-detected | Override automatic project detection |

#### Usage Examples

**Store a note in your project scratchbook:**
```json
{
  "content": "Discussed API rate limiting in team meeting - need to implement exponential backoff",
  "title": "Rate Limiting Discussion",
  "source": "scratchbook",
  "metadata": {
    "tags": ["meeting", "api", "todo"]
  }
}
```

**Store code documentation:**
```json
{
  "content": "The authenticate() function validates JWT tokens and returns user context...",
  "title": "Authentication Documentation",
  "source": "docs",
  "document_type": "documentation",
  "metadata": {
    "module": "auth",
    "file_type": "docs"
  }
}
```

**Store web content:**
```json
{
  "content": "Article content about microservices architecture...",
  "title": "Microservices Best Practices",
  "url": "https://example.com/microservices-guide",
  "source": "web",
  "metadata": {
    "domain": "example.com",
    "category": "architecture"
  }
}
```

#### Response Format

```json
{
  "success": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "collection": "_a1b2c3d4e5f6",
  "title": "Rate Limiting Discussion",
  "content_length": 89,
  "chunks_created": 1,
  "metadata": {
    "title": "Rate Limiting Discussion",
    "source": "scratchbook",
    "document_type": "text",
    "created_at": "2025-01-21T10:30:00.000Z",
    "project": "my-project",
    "tags": ["meeting", "api", "todo"],
    "content_preview": "Discussed API rate limiting in team meeting - need to implement exponential backoff"
  }
}
```

**Fallback Mode Response** (when daemon unavailable):
```json
{
  "success": true,
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "collection": "_a1b2c3d4e5f6",
  "title": "Rate Limiting Discussion",
  "content_length": 89,
  "metadata": { "..." },
  "fallback_mode": "direct_qdrant_write"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Failed to store document: Connection refused"
}
```

---

### search

Search across collections with hybrid semantic + keyword matching, branch filtering, and file type filtering.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | **Yes** | - | Search query text |
| `scope` | string | No | `"project"` | Search scope: `"project"` (current project), `"global"`, `"all"` (all projects) |
| `include_libraries` | boolean | No | `false` | Include library documentation (`_libraries` collection) in results |
| `collection` | string | No | Auto-detected | Specific collection to search (overrides scope) |
| `mode` | string | No | `"hybrid"` | Search mode: "hybrid", "semantic", "exact", or "keyword" |
| `limit` | integer | No | `10` | Maximum number of results to return |
| `score_threshold` | float | No | `0.3` | Minimum similarity score (0.0-1.0) |
| `filters` | object | No | `{}` | Additional metadata filters |
| `branch` | string | No | Current branch | Git branch to search (null=current, "*"=all branches) |
| `file_type` | string | No | `null` | File type filter ("code", "test", "docs", "config", "data", "build", "other") |
| `project_name` | string | No | Auto-detected | **DEPRECATED** - use `scope` instead |

#### Search Modes

**hybrid** (default) - Combines semantic similarity with keyword matching using Reciprocal Rank Fusion:
- Best for: General-purpose search, finding both conceptual matches and exact terms
- Performance: 94.2% semantic recall, 100% symbol precision

**semantic** - Pure vector similarity search:
- Best for: Conceptual searches, finding semantically similar content
- Example: "error handling" matches "exception management", "fault tolerance"

**exact** - Keyword and symbol exact matching:
- Best for: Finding specific function names, variable names, exact phrases
- Example: "def authenticate" finds exact function definitions

**keyword** - Simple keyword matching:
- Best for: Quick text searches without semantic understanding

#### Usage Examples

**Basic hybrid search (current project, current branch):**
```json
{
  "query": "authentication logic",
  "mode": "hybrid"
}
```

**Search all projects:**
```json
{
  "query": "authentication patterns",
  "scope": "all"
}
```

**Search with library documentation:**
```json
{
  "query": "JWT token validation",
  "include_libraries": true
}
```

**Search all projects with libraries:**
```json
{
  "query": "FastAPI dependency injection",
  "scope": "all",
  "include_libraries": true
}
```

**Search for exact function definitions in code files:**
```json
{
  "query": "def authenticate",
  "mode": "exact",
  "file_type": "code"
}
```

**Search across all branches for documentation:**
```json
{
  "query": "deployment process",
  "mode": "semantic",
  "branch": "*",
  "file_type": "docs"
}
```

**Search specific branch with custom filters:**
```json
{
  "query": "payment processing",
  "branch": "feature/payments",
  "file_type": "code",
  "filters": {
    "module": "payments",
    "priority": "high"
  },
  "limit": 20,
  "score_threshold": 0.5
}
```

#### Response Format

```json
{
  "success": true,
  "query": "authentication logic",
  "mode": "hybrid",
  "scope": "project",
  "collections_searched": ["_projects"],
  "tenant_id": "github_com_user_repo",
  "include_libraries": false,
  "total_results": 3,
  "results": [
    {
      "id": "doc-id-1",
      "score": 0.89,
      "collection": "_projects",
      "content": "The authentication logic validates JWT tokens...",
      "title": "Auth Module Documentation",
      "metadata": {
        "tenant_id": "github_com_user_repo",
        "file_type": "code",
        "branch": "main",
        "file_path": "src/auth.py",
        "created_at": "2025-01-21T10:00:00.000Z"
      }
    },
    {
      "id": "doc-id-2",
      "score": 0.76,
      "collection": "_projects",
      "content": "User authentication flow diagram...",
      "title": "Authentication Flow",
      "metadata": {
        "tenant_id": "github_com_user_repo",
        "file_type": "docs",
        "branch": "main"
      }
    }
  ],
  "search_time_ms": 45.23,
  "filters_applied": {
    "tenant_id": "github_com_user_repo",
    "branch": "main",
    "file_type": null,
    "custom": {}
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Search failed: Collection not found: invalid-collection",
  "results": []
}
```

---

### manage

Manage collections, system status, and configuration with comprehensive administrative operations.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | string | **Yes** | - | Management action to perform |
| `collection` | string | No | `null` | Target collection name (for collection-specific actions) |
| `name` | string | No | `null` | Name for new collections or operations |
| `project_name` | string | No | Auto-detected | Project context for workspace operations |
| `config` | object | No | `{}` | Additional configuration for operations |

#### Available Actions

| Action | Description | Required Parameters |
|--------|-------------|-------------------|
| `list_collections` | List all collections with statistics (unified model: `_projects`, `_libraries`, user, memory) | None |
| `create_collection` | Create a new collection | `name` |
| `delete_collection` | Delete a collection | `name` or `collection` |
| `collection_info` | Get detailed info about a collection (shows tenant counts for unified collections) | `name` or `collection` |
| `workspace_status` | Get system status, health check, and current `tenant_id` | None |
| `init_project` | Register project in unified `_projects` collection (creates collection if needed) | None |
| `cleanup` | Remove empty collections and optimize | None |
| `migrate_to_unified` | Migrate old per-project collections to unified model | None |

#### Usage Examples

**List all collections:**
```json
{
  "action": "list_collections"
}
```

**Create a new collection:**
```json
{
  "action": "create_collection",
  "name": "my-project-experiments",
  "config": {
    "vector_size": 384,
    "distance": "Cosine"
  }
}
```

**Delete a collection:**
```json
{
  "action": "delete_collection",
  "name": "old-collection"
}
```

**Get collection details:**
```json
{
  "action": "collection_info",
  "name": "_a1b2c3d4e5f6"
}
```

**Check workspace status:**
```json
{
  "action": "workspace_status"
}
```

**Initialize project collection:**
```json
{
  "action": "init_project",
  "project_name": "my-project"
}
```

**Cleanup empty collections:**
```json
{
  "action": "cleanup"
}
```

#### Response Formats

**list_collections response:**
```json
{
  "success": true,
  "action": "list_collections",
  "collections": [
    {
      "name": "_a1b2c3d4e5f6",
      "points_count": 1523,
      "segments_count": 3,
      "status": "green",
      "vector_size": 384,
      "distance": "Cosine"
    },
    {
      "name": "docs",
      "points_count": 89,
      "segments_count": 1,
      "status": "green",
      "vector_size": 384,
      "distance": "Cosine"
    }
  ],
  "total_collections": 2
}
```

**create_collection response:**
```json
{
  "success": true,
  "action": "create_collection",
  "collection_name": "my-project-experiments",
  "message": "Collection 'my-project-experiments' created successfully via daemon"
}
```

**workspace_status response:**
```json
{
  "success": true,
  "action": "workspace_status",
  "timestamp": "2025-01-21T10:30:00.000Z",
  "current_project": "my-project",
  "project_collection": "_a1b2c3d4e5f6",
  "branch": "main",
  "qdrant_status": "connected",
  "cluster_info": {
    "peer_id": "peer-1",
    "raft_info": {}
  },
  "project_collections": ["_a1b2c3d4e5f6"],
  "total_collections": 5,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**collection_info response:**
```json
{
  "success": true,
  "action": "collection_info",
  "collection_name": "_a1b2c3d4e5f6",
  "info": {
    "points_count": 1523,
    "segments_count": 3,
    "status": "green",
    "vector_size": 384,
    "distance": "Cosine",
    "indexed": 1523,
    "optimizer_status": "ok"
  }
}
```

**Error response:**
```json
{
  "success": false,
  "error": "Unknown action: invalid_action",
  "available_actions": [
    "list_collections",
    "create_collection",
    "delete_collection",
    "collection_info",
    "workspace_status",
    "init_project",
    "cleanup"
  ]
}
```

---

### retrieve

Retrieve documents directly by ID or metadata without search ranking, with branch and file type filtering.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `document_id` | string | No* | `null` | Direct document ID to retrieve |
| `collection` | string | No | Auto-detected | Specific collection to retrieve from |
| `metadata` | object | No* | `null` | Metadata filters for document selection |
| `limit` | integer | No | `10` | Maximum number of documents to retrieve |
| `project_name` | string | No | Auto-detected | Limit retrieval to project collections |
| `branch` | string | No | Current branch | Git branch to filter by (null=current, "*"=all branches) |
| `file_type` | string | No | `null` | File type filter ("code", "test", "docs", etc.) |

*Note: Either `document_id` or `metadata` must be provided

#### Retrieval Methods

**Direct ID Lookup** - Retrieve a specific document by its UUID:
- Fastest retrieval method
- Exact match only
- Still respects branch and file_type filters

**Metadata-based Retrieval** - Filter documents by metadata fields:
- Flexible filtering on any metadata field
- Combine multiple filters
- Supports branch and file_type filtering

#### Usage Examples

**Retrieve document by ID (current branch):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Retrieve document by ID from specific branch:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "branch": "develop"
}
```

**Retrieve all test files from develop branch:**
```json
{
  "metadata": {
    "file_type": "test"
  },
  "branch": "develop",
  "limit": 50
}
```

**Retrieve documents by custom metadata:**
```json
{
  "metadata": {
    "module": "auth",
    "priority": "high"
  },
  "file_type": "code",
  "branch": "*"
}
```

**Retrieve all code files across all branches:**
```json
{
  "metadata": {},
  "file_type": "code",
  "branch": "*",
  "limit": 100
}
```

#### Response Format

**Successful retrieval:**
```json
{
  "success": true,
  "total_results": 2,
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "collection": "_a1b2c3d4e5f6",
      "content": "def authenticate(token): ...",
      "title": "Authentication Module",
      "metadata": {
        "file_type": "code",
        "branch": "main",
        "file_path": "src/auth.py",
        "module": "auth",
        "created_at": "2025-01-21T10:00:00.000Z"
      }
    },
    {
      "id": "another-doc-id",
      "collection": "_a1b2c3d4e5f6",
      "content": "class UserAuth: ...",
      "title": "User Auth Class",
      "metadata": {
        "file_type": "code",
        "branch": "main",
        "file_path": "src/user_auth.py",
        "module": "auth"
      }
    }
  ],
  "query_type": "metadata_filter",
  "filters_applied": {
    "branch": "main",
    "file_type": "code",
    "metadata": {
      "module": "auth"
    }
  }
}
```

**Document not found on requested branch:**
```json
{
  "success": true,
  "total_results": 0,
  "results": [],
  "query_type": "id_lookup",
  "message": "Document found but not on branch 'develop'"
}
```

**Error response:**
```json
{
  "success": false,
  "error": "Either document_id or metadata filters must be provided",
  "results": []
}
```

---

## Integration

### Claude Desktop (stdio mode)

Claude Desktop communicates with MCP servers using the **stdio (standard input/output) transport protocol**. This is the recommended mode for MCP integration.

#### Configuration

Add workspace-qdrant-mcp to your Claude Desktop configuration file:

**Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Example configuration:**

```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "",
        "FASTEMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}
```

**For Qdrant Cloud:**

```json
{
  "mcpServers": {
    "workspace-qdrant-mcp": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "https://your-cluster.qdrant.io",
        "QDRANT_API_KEY": "your-api-key-here",
        "FASTEMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
      }
    }
  }
}
```

#### stdio Mode Behavior

In stdio mode, workspace-qdrant-mcp:
- Automatically detects stdio mode and silences all console output
- Redirects stdout/stderr to devnull to maintain protocol compliance
- Disables all logging to prevent protocol contamination
- Communicates exclusively through JSON-RPC messages

**Important:** No output is visible when running in stdio mode. This is correct behavior to maintain MCP protocol compliance.

#### Verification

After configuring Claude Desktop:

1. Restart Claude Desktop completely
2. Open a new conversation
3. Type: "List all my Qdrant collections"
4. Claude should respond with collection information

If the server is not connecting, see [Troubleshooting](#troubleshooting).

---

### HTTP Mode

For testing, debugging, or custom integrations, you can run workspace-qdrant-mcp in HTTP mode.

#### When to Use HTTP Mode

**Use HTTP mode for:**
- Testing tools with curl or Postman
- Debugging tool behavior with visible responses
- Custom integrations outside Claude Desktop
- Development and troubleshooting

**Use stdio mode for:**
- Production Claude Desktop integration
- Claude Code integration
- Standard MCP client usage

#### Starting HTTP Server

```bash
# Start on default port (8000)
workspace-qdrant-mcp --transport http

# Custom host and port
workspace-qdrant-mcp --transport http --host 127.0.0.1 --port 3000
```

#### Testing HTTP Endpoints

**List collections:**
```bash
curl -X POST http://localhost:8000/manage \
  -H "Content-Type: application/json" \
  -d '{
    "action": "list_collections"
  }'
```

**Search documents:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication",
    "mode": "hybrid",
    "limit": 5
  }'
```

**Store a document:**
```bash
curl -X POST http://localhost:8000/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Test document content",
    "title": "Test Document",
    "source": "api_test"
  }'
```

#### Other Transport Modes

workspace-qdrant-mcp also supports:
- **sse** - Server-Sent Events for streaming
- **streamable-http** - HTTP with streaming support

```bash
# Server-Sent Events
workspace-qdrant-mcp --transport sse

# Streamable HTTP
workspace-qdrant-mcp --transport streamable-http
```

---

## Troubleshooting

### MCP Connection Issues

**Problem: Claude Desktop not showing MCP tools**

1. Check configuration file location and syntax:
   ```bash
   # Validate JSON syntax
   cat ~/Library/Application\ Support/Claude/claude_desktop_config.json | jq .
   ```

2. Verify workspace-qdrant-mcp is installed:
   ```bash
   which workspace-qdrant-mcp
   # Should output: /Users/yourname/.local/bin/workspace-qdrant-mcp
   ```

3. Test the server manually:
   ```bash
   # Test in HTTP mode (should show FastMCP server starting)
   workspace-qdrant-mcp --transport http
   ```

4. Restart Claude Desktop completely (quit and relaunch)

5. Check Claude Desktop logs for errors:
   - macOS: `~/Library/Logs/Claude/`
   - Windows: `%APPDATA%\Claude\Logs\`
   - Linux: `~/.config/Claude/logs/`

**Problem: Tools available but returning errors**

1. Verify Qdrant is running:
   ```bash
   curl http://localhost:6333/collections
   # Should return JSON with collections list
   ```

2. Check Qdrant connection:
   ```bash
   workspace-qdrant-health
   # Run health diagnostics
   ```

3. Verify environment variables:
   ```bash
   echo $QDRANT_URL
   echo $QDRANT_API_KEY  # Should be set for Qdrant Cloud
   ```

4. Test with HTTP mode to see detailed errors:
   ```bash
   workspace-qdrant-mcp --transport http
   # Then test with curl to see actual error messages
   ```

### Qdrant Connection Errors

**Problem: "Connection refused" errors**

```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Start Qdrant if not running
docker run -p 6333:6333 qdrant/qdrant

# Or with docker-compose
docker-compose up -d qdrant
```

**Problem: "Unauthorized" or authentication errors**

- Verify `QDRANT_API_KEY` is set correctly for Qdrant Cloud
- For local Qdrant, no API key should be needed
- Check Qdrant Cloud cluster URL is correct

### Collection Issues

**Problem: Collections not found or empty**

```bash
# List all collections
curl http://localhost:6333/collections

# Check workspace status
workspace-qdrant-mcp --transport http
# Then: curl -X POST http://localhost:8000/manage -d '{"action":"workspace_status"}'

# Initialize project collection
curl -X POST http://localhost:8000/manage \
  -H "Content-Type: application/json" \
  -d '{"action":"init_project"}'
```

**Problem: Wrong collection being searched**

- Check `get_project_collection()` is detecting the right project
- Verify Git repository detection with `git remote get-url origin`
- Use `workspace-status` action to see current project detection
- Override with `collection` parameter if needed

### Performance Issues

**Problem: Slow search responses**

1. Check Qdrant server resources:
   ```bash
   # Monitor Qdrant container
   docker stats qdrant
   ```

2. Run benchmarks:
   ```bash
   workspace-qdrant-test --benchmark
   ```

3. Optimize collections:
   ```bash
   # Cleanup empty collections
   curl -X POST http://localhost:8000/manage \
     -H "Content-Type: application/json" \
     -d '{"action":"cleanup"}'
   ```

4. Consider lighter embedding model if memory constrained:
   ```json
   "env": {
     "FASTEMBED_MODEL": "sentence-transformers/all-MiniLM-L6-v2"
   }
   ```

### Embedding Issues

**Problem: Embedding generation failures**

1. Check model is downloaded:
   ```bash
   # FastEmbed models cached in ~/.cache/fastembed
   ls ~/.cache/fastembed/
   ```

2. Try fallback model:
   ```bash
   export FASTEMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   workspace-qdrant-mcp --transport http
   ```

3. Check available disk space (models can be 100MB+)

### Daemon Issues

**Problem: Daemon not processing documents**

```bash
# Check daemon status
wqm service status

# View daemon logs
wqm service logs

# Restart daemon
wqm service restart

# Check daemon health
workspace-qdrant-health --daemon
```

### Getting Help

If issues persist:

1. Run comprehensive diagnostics:
   ```bash
   workspace-qdrant-test --report diagnostic_report.json
   workspace-qdrant-health --report health_report.json
   ```

2. Check documentation:
   - [README.md](README.md) - Installation and setup
   - [CLI.md](CLI.md) - Command-line reference
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup

3. Open an issue on GitHub with:
   - Diagnostic reports
   - Error messages
   - Configuration (redact API keys)
   - Steps to reproduce

---

**Additional Resources:**
- [MCP Protocol Documentation](https://github.com/anthropics/mcp)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [FastMCP Framework](https://github.com/jlowin/fastmcp)
