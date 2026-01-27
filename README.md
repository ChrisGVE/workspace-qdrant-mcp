# workspace-qdrant-mcp

**Project-scoped Qdrant MCP server with hybrid search and configurable collections**

> **v0.4.0** introduces the unified multi-tenant architecture with a high-performance Rust daemon (`memexd`) and CLI (`wqm`). The MCP server, CLI, and daemon are all fully functional.

[![PyPI version](https://badge.fury.io/py/workspace-qdrant-mcp.svg)](https://pypi.org/project/workspace-qdrant-mcp/) [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/) [![Downloads](https://pepy.tech/badge/workspace-qdrant-mcp)](https://pepy.tech/project/workspace-qdrant-mcp) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Semantic Release](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/semantic-release.yml/badge.svg)](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/semantic-release.yml) [![Release Verification](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/release-verification.yml/badge.svg)](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/release-verification.yml) [![Quality Assurance](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/quality.yml/badge.svg)](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/quality.yml) [![Security Scan](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/security.yml/badge.svg)](https://github.com/ChrisGVE/workspace-qdrant-mcp/actions/workflows/security.yml) [![Codecov](https://codecov.io/gh/ChrisGVE/workspace-qdrant-mcp/branch/main/graph/badge.svg)](https://codecov.io/gh/ChrisGVE/workspace-qdrant-mcp) [![Qdrant](https://img.shields.io/badge/Qdrant-1.7%2B-red.svg)](https://qdrant.tech) [![FastMCP](https://img.shields.io/badge/FastMCP-0.3%2B-orange.svg)](https://github.com/jlowin/fastmcp) [![GitHub Discussions](https://img.shields.io/github/discussions/ChrisGVE/workspace-qdrant-mcp?style=social&logo=github&label=Discussions)](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions) [![GitHub stars](https://img.shields.io/github/stars/ChrisGVE/workspace-qdrant-mcp.svg?style=social&label=Stars)](https://github.com/ChrisGVE/workspace-qdrant-mcp/stargazers) [![MseeP.ai Security Assessment](https://mseep.net/pr/chrisgve-workspace-qdrant-mcp-badge.png)](https://mseep.ai/app/chrisgve-workspace-qdrant-mcp)

</div>

---

_Inspired by [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) with enhanced project detection, Python implementation, and flexible collection management._

workspace-qdrant-mcp provides intelligent vector database operations through the Model Context Protocol (MCP), featuring automatic project detection, hybrid search capabilities, and configurable collection management for seamless integration with Claude Desktop and Claude Code.

## ✨ Key Features

- 🏗️ **Auto Project Detection** - Smart workspace-scoped collections with Git repository awareness
- 🔍 **Hybrid Search** - Combines semantic and keyword search with reciprocal rank fusion
- 📝 **Scratchbook Collections** - Personal development journals for each project
- 🎯 **Subproject Support** - Git submodules with user-filtered collection creation
- 🌐 **Cross-Platform** - Native support for macOS, Linux, and Windows
- 🛡️ **Enterprise Ready** - Comprehensive security scanning and quality assurance
- 🚀 **High Performance Rust CLI** - Fast `wqm` command-line interface for all operations
- 📁 **Background Daemon** - `memexd` for continuous document monitoring and processing
- ⚙️ **Interactive Setup Wizard** - Guided configuration with health checks

## 🔧 MCP Tools

workspace-qdrant-mcp provides 4 comprehensive MCP tools for vector database operations:

### 1. **store** - Content Storage
Store any type of content in the vector database with automatic embedding generation and metadata enrichment.
- Supports text, code, documentation, notes, and more
- Automatic project detection and collection routing
- Metadata enrichment (file_type, branch, tenant_id)

**Parameters:**
- `content` (required): The text content to store
- `collection`: Target collection name (auto-detected if omitted)
- `file_type`: Content classification (code, doc, note, etc.)
- `metadata`: Additional key-value metadata

### 2. **search** - Hybrid Search
Search across collections with powerful hybrid semantic + keyword matching.
- **hybrid mode**: Combines semantic similarity with keyword matching (default)
- **semantic mode**: Pure vector similarity search for conceptual matches
- **exact mode**: Keyword and symbol exact matching
- Automatic result optimization using Reciprocal Rank Fusion (RRF)

**Parameters:**
- `query` (required): The search query
- `mode`: Search mode - `"hybrid"` (default), `"semantic"`, or `"exact"`
- `scope`: Search scope - `"project"` (default), `"global"`, or `"all"`
- `include_libraries`: Include library documentation - `true`/`false` (default: false)
- `branch`: Filter by git branch (default: current branch, `"*"` for all)
- `file_type`: Filter by content type (code, doc, test, config, note, etc.)
- `limit`: Maximum results to return (default: 10)

**Example:**
```python
# Search current project only
search(query="authentication", scope="project")

# Search all projects with library documentation
search(query="JWT tokens", scope="all", include_libraries=True)

# Search specific branch
search(query="bugfix", branch="feature/auth")
```

### 3. **manage** - Collection Management
Manage collections, system status, and configuration.
- List all collections with statistics
- Create and delete collections
- Get workspace status and health information
- Initialize project collections
- Cleanup empty collections and optimize storage

**Operations:**
- `list_collections`: Show all collections with document counts
- `create_collection`: Create new collection with proper configuration
- `delete_collection`: Remove collection and all its data
- `get_status`: System health and metrics

### 4. **retrieve** - Direct Document Access
Retrieve documents by ID or metadata without search ranking.
- Direct document ID lookup
- Metadata-based filtering
- Branch and file type filtering
- Efficient bulk retrieval

**Parameters:**
- `document_id`: Retrieve specific document by ID
- `collection`: Target collection name
- `branch`: Filter by branch
- `file_type`: Filter by content type
- `limit`: Maximum documents to return

All tools seamlessly integrate with Claude Desktop and Claude Code for natural language interaction.

## Table of Contents

- [✨ Key Features](#-key-features)
- [🔧 MCP Tools](#-mcp-tools)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [MCP Integration](#mcp-integration)
- [Configuration](#configuration)
- [Usage](#usage)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### MCP Server

The MCP server provides all 4 tools (`store`, `search`, `manage`, `retrieve`) for use with Claude Desktop and Claude Code:

```bash
# Use uvx to run the MCP server directly (no installation needed)
uvx workspace-qdrant-mcp

# Or install and run
uv tool install workspace-qdrant-mcp
workspace-qdrant-mcp
```

### Full Installation with CLI + Daemon

For production deployments with background processing:

```bash
# 1. Install the Python package (provides MCP server + wqm CLI)
uv tool install workspace-qdrant-mcp

# 2. Install and start the daemon service
wqm service install
wqm service start

# 3. Verify installation
wqm service status
```

The daemon provides:
- Continuous document monitoring and processing
- Background embedding generation with file watching
- Automatic startup on system boot with crash recovery
- High-performance gRPC communication with MCP server

## Prerequisites

**Qdrant server must be running** - workspace-qdrant-mcp connects to Qdrant for vector operations.

- **Local**: Default `http://localhost:6333`
- **Cloud**: Requires `QDRANT_API_KEY` environment variable

For local installation, see the [Qdrant repository](https://github.com/qdrant/qdrant). For documentation examples, we assume the default local setup.

## Installation

### Prerequisites

- **Python 3.10+** - For the MCP server
- **Qdrant server** - Running locally or in cloud (see [Prerequisites](#prerequisites))

### Install MCP Server

```bash
# Install globally with uv (recommended)
uv tool install workspace-qdrant-mcp

# Or with pip
pip install workspace-qdrant-mcp
```

This installs the `workspace-qdrant-mcp` command which runs the MCP server.

### Rust CLI and Daemon

The `wqm` CLI and `memexd` daemon are included with the Python package:

```bash
# CLI is available after installing the package
wqm --help

# For development builds from source (requires Rust toolchain)
cd src/rust/daemon
cargo build --release
# Binary at: target/release/memexd
```

For development setup, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Development Build

For building from source or contributing to the project:

```bash
# Python development dependencies
uv sync --dev

# Build unified Rust daemon (workspace at src/rust/daemon)
cd src/rust/daemon
cargo build --release

# The daemon binary will be at:
# target/release/memexd

# Run Python tests (requires running Qdrant server)
cd ../../..
uv run pytest

# Run Rust tests
cd src/rust/daemon
cargo test
```

The unified daemon workspace at `src/rust/daemon` contains:
- `core/` - Daemon core logic and processing
- `grpc/` - gRPC service implementations (SystemService, CollectionService, DocumentService)
- `python-bindings/` - Python bindings for Rust components
- `proto/` - Protocol buffer definitions

## Daemon Service

The unified daemon (`memexd`) provides continuous document processing and monitoring capabilities for production deployments.

### Daemon Features

The unified daemon (`memexd`) provides:
- 📁 **Real-time file monitoring** with SQLite-driven watch configuration
- 🤖 **Background embedding generation** for optimal performance
- 🔄 **gRPC services**: SystemService, CollectionService, DocumentService
- 🔌 **Single writer pattern**: All Qdrant writes route through daemon for consistency
- 🚀 **Automatic startup** on system boot with crash recovery
- 📊 **Health monitoring** with comprehensive metrics and status reporting

### gRPC Services

The daemon exposes three gRPC services:

**1. SystemService** - Health monitoring and lifecycle management
**2. CollectionService** - Qdrant collection lifecycle management
**3. DocumentService** - Direct text ingestion (non-file content)

**📖 Architecture Documentation:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design specifications.

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

| Variable             | Default                                  | Description                                   |
| -------------------- | ---------------------------------------- | --------------------------------------------- |
| `QDRANT_URL`         | `http://localhost:6333`                  | Qdrant server URL                             |
| `QDRANT_API_KEY`     | _(none)_                                 | Required for Qdrant cloud, optional for local |
| `COLLECTIONS`        | `project`                                | Collection suffixes (comma-separated)         |
| `GLOBAL_COLLECTIONS` | _(none)_                                 | Global collection names (comma-separated)     |
| `GITHUB_USER`        | _(none)_                                 | Filter projects by GitHub username            |
| `FASTEMBED_MODEL`    | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (see options below)           |

### YAML Configuration

For easier project-specific configuration management, you can use YAML configuration files:

```bash
# Start with YAML configuration
workspace-qdrant-mcp --config=project-config.yaml
```

**Configuration Precedence:**
1. Command line arguments (highest)
2. YAML configuration file
3. Environment variables  
4. Default values (lowest)

**Example YAML configuration:**

```yaml
# project-config.yaml
host: "127.0.0.1"
port: 8000
debug: false

qdrant:
  url: "http://localhost:6333"
  api_key: null
  timeout: 30
  prefer_grpc: false

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  enable_sparse_vectors: true
  chunk_size: 800
  chunk_overlap: 120
  batch_size: 50

workspace:
  collection_types: ["project"]
  global_collections: ["docs", "references", "standards"]
  github_user: null
  auto_create_collections: false
  memory_collection_name: "__memory"
  code_collection_name: "__code"
```

**Benefits:**
- **Project-specific**: Each project can have its own config
- **Version control**: YAML configs can be committed to your repo
- **Team sharing**: Easy to share configurations
- **Validation**: Built-in validation with helpful error messages
- **Documentation**: Self-documenting with structure and comments

See [YAML_CONFIG.md](YAML_CONFIG.md) for complete documentation and examples.

### Embedding Model Options

Choose the embedding model that best fits your system resources and quality requirements:

**Lightweight (384D) - Good for limited resources:**

- `sentence-transformers/all-MiniLM-L6-v2` (default) - Fast, low memory

**Balanced (768D) - Better quality, moderate resources:**

- `BAAI/bge-base-en-v1.5` - Excellent for most use cases
- `jinaai/jina-embeddings-v2-base-en` - Good multilingual support
- `thenlper/gte-base` - Google's T5-based model

**High Quality (1024D) - Best results, high resource usage:**

- `BAAI/bge-large-en-v1.5` - Top performance for English
- `mixedbread-ai/mxbai-embed-large-v1` - Latest state-of-the-art

**Configuration example:**

```bash
# Use a more powerful model
export FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"

# Or in Claude Desktop config
"env": {
  "FASTEMBED_MODEL": "BAAI/bge-base-en-v1.5"
}
```

### Multi-Tenant Collection Architecture

workspace-qdrant-mcp uses a **unified multi-tenant architecture** with only 4 collections, providing efficient storage and fast cross-project search:

**Unified Projects Collection** (`_projects`):

- **Single collection** for ALL projects with tenant-based filtering
- Uses `tenant_id` (derived from git remote URL or path hash) for isolation
- Supports cross-project search with proper scoping
- Multi-branch support via `branch` metadata field
- Automatic project detection with Git repository awareness

**Unified Libraries Collection** (`_libraries`):

- **Single collection** for ALL external documentation and references
- Uses `library_name` for tenant isolation (e.g., `react`, `numpy`, `fastapi`)
- Semantic search across documentation with optional library filtering
- Folder structure preserved in metadata for topic navigation

**User Collections** (`{basename}-{type}`):

- User-defined collections for notes, ideas, and personal content
- Examples: `work-notes`, `personal-snippets`, `project-ideas`
- Auto-enriched with current `project_id` when accessed from project directory
- Searchable across all projects with optional project filtering

**Memory Collections** (`_memory`, `_agent_memory`):

- Global collections for user preferences and LLM behavioral rules
- Conversational memory and cross-project learnings
- Fixed names for consistent access

**Benefits of Multi-Tenant Architecture:**
- 🚀 **Scalable**: Handles hundreds of projects without collection explosion
- 🔍 **Efficient search**: Single index for fast cross-project queries
- 🔒 **Isolated**: Hard tenant filtering prevents data leakage
- 📊 **Optimized**: Qdrant payload indexing on tenant_id for fast filtering

**📖 For complete documentation:** See [Collection Naming Guide](docs/COLLECTION_NAMING.md) and [Multi-Tenancy Architecture](docs/multitenancy_architecture.md) for detailed specifications.

### Development Notes Collections

You can create user collections for capturing development thoughts and notes across projects. These are your **personal development journals** accessible from any project.

**What goes in user note collections:**

- 📝 **Meeting notes** and action items
- 💡 **Ideas** and implementation thoughts
- ✅ **TODOs** and reminders
- 🔧 **Code snippets** and implementation patterns
- 🏗️ **Architecture decisions** and rationale
- 🐛 **Bug reports** and troubleshooting notes
- 📊 **Research findings** and links
- 🎯 **Project goals** and milestones

**Example usage:**

```bash
# Store note in user collection (auto-enriched with current project context)
wqm add --collection myapp-notes "Discussed API rate limiting - implement exponential backoff"

# Search across all your notes
wqm search "rate limiting" --collection myapp-notes
```

**Collection naming:** User collections follow the pattern `{basename}-{type}` (e.g., `work-notes`, `personal-snippets`) and are automatically enriched with the current project context when accessed.

### Subproject Support (Git Submodules)

For repositories with **Git submodules**, the daemon automatically detects and tracks each submodule as a separate tenant within the unified `_projects` collection:

**Requirements:**

- Must set `WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER=yourusername`
- Only submodules **owned by you** are tracked (prevents vendor/third-party sprawl)
- Without `github_user` configured, only main project is tracked (conservative approach)

**How it works:**

Each submodule gets its own `tenant_id` (normalized git remote URL) within the unified `_projects` collection. The daemon identifies ownership by comparing git remote usernames.

**Example with subprojects:**

```bash
# Repository: my-monorepo with submodules
# - frontend/ (github.com/myuser/frontend)
# - backend/ (github.com/myuser/backend)
# - vendor-lib/ (github.com/vendor/lib) ← ignored (different owner)

# All content stored in unified _projects collection with tenant_id:
# - tenant_id: "github_com_myuser_monorepo" (main project)
# - tenant_id: "github_com_myuser_frontend" (frontend submodule)
# - tenant_id: "github_com_myuser_backend" (backend submodule)
# - vendor-lib content NOT ingested (different owner)
```

### Cross-Tenant Search

**Important: MCP search intelligently scopes queries based on context.**

When you use Claude with commands like:

- "Search my project for authentication code"
- "Find all references to the payment API"
- "What documentation do I have about deployment?"

The search **automatically handles:**

- ✅ **Project scope**: Searches current project's `tenant_id` in `_projects` collection
- ✅ **Library inclusion**: Optionally includes `_libraries` collection for documentation context
- ✅ **Cross-project**: Can search across all projects when explicitly requested
- ✅ **User collections**: Includes user notes collections when relevant

**Search parameters:**
- `scope`: `"project"` (default) or `"all"` for cross-project search
- `include_libraries`: `true` to include library documentation
- `branch`: Filter by specific branch (default: current branch)

This gives you **intelligent search with proper scoping** - no accidental data leakage between projects.

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

## CLI Tools

The `wqm` CLI provides comprehensive command-line tools for managing your semantic workspace.

### CLI Features

The CLI provides:

- **Service Management**: `wqm service start|stop|restart|status`
- **Memory Management**: `wqm memory list|add|edit|remove`
- **Document Ingestion**: `wqm ingest file|folder|web`
- **Search**: `wqm search project|collection|global`
- **Library Management**: `wqm library list|add|watch`
- **System Admin**: `wqm admin status|collections|health`
- **Queue Inspector**: `wqm queue list|show|stats|clean` - Debug unified queue state

**📖 CLI Documentation:** See [docs/CLI.md](docs/CLI.md) for the complete command reference.

## Documentation

**Core Documentation:**
- **[API Reference](docs/API.md)** - Complete MCP tools documentation
- **[CLI Reference](docs/CLI.md)** - Command-line interface reference
- **[Collection Naming Guide](docs/COLLECTION_NAMING.md)** - Collection types and naming conventions
- **[Search Examples](docs/EXAMPLES.md)** - Multi-tenant search scenarios
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and testing

**Advanced Documentation:**
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and daemon design
- **[gRPC API Reference](docs/GRPC_API.md)** - gRPC protocol documentation
- **[Migration Guide](docs/MIGRATION.md)** - v0.3.x to v0.4.0 upgrade instructions
- **[Metrics Reference](docs/METRICS.md)** - Queue and system metrics for monitoring
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Troubleshooting and debugging
- **[OS Compatibility](docs/OS_COMPATIBILITY.md)** - Platform support and requirements
- **[Release Notes](docs/RELEASE_NOTES.md)** - v0.4.0 features and changelog

## Troubleshooting

**Verify Qdrant Connection:**

```bash
# Check if Qdrant is running
curl http://localhost:6333/collections

# Test MCP server startup
workspace-qdrant-mcp --help
```

**Common Issues:**

- **"Connection refused"**: Ensure Qdrant server is running on `http://localhost:6333`
- **"API key required"**: Set `QDRANT_API_KEY` environment variable for Qdrant Cloud
- **"Module not found"**: Reinstall with `uv tool install workspace-qdrant-mcp`

For detailed troubleshooting, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## 🚀 Release Process

This project uses **fully automated semantic versioning** and PyPI publishing. Every commit to the main branch is analyzed for release necessity using conventional commits.

### Automated Release Pipeline

- **Semantic Analysis**: Commits analyzed for version impact (major/minor/patch)
- **Cross-Platform Builds**: Automatic wheel building for Linux, macOS, Windows
- **Comprehensive Testing**: TestPyPI validation before production release
- **Security Scanning**: Dependency and vulnerability analysis
- **Release Verification**: Multi-platform installation testing
- **Emergency Rollback**: Automated rollback capabilities for critical issues

### Commit Message Format

```bash
# Feature releases (minor version bump: 1.0.0 → 1.1.0)
git commit -m "feat: add new hybrid search algorithm"

# Bug fixes (patch version bump: 1.0.0 → 1.0.1)
git commit -m "fix: resolve memory leak in document processing"

# Breaking changes (major version bump: 1.0.0 → 2.0.0)
git commit -m "feat!: redesign MCP tool interface

BREAKING CHANGE: Tool parameters have changed."

# No release (documentation, tests, chores)
git commit -m "docs: update API examples"
git commit -m "test: add integration tests"
git commit -m "chore: update dependencies"
```

**📚 Documentation**: See [CI/CD Processes](docs/ci-cd-processes.md) for complete release documentation and emergency procedures.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up the development environment
- Running tests and benchmarks
- Code style and quality requirements
- Submitting pull requests

## 💬 Community

Join our community discussions for support, ideas, and collaboration:

- **[GitHub Discussions](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions)** - Community Q&A, feature ideas, and showcases
- **[GitHub Issues](https://github.com/ChrisGVE/workspace-qdrant-mcp/issues)** - Bug reports and feature requests

For support, check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) first, then open a discussion or issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Related Projects:**

- [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) - Original TypeScript implementation
- [Qdrant](https://qdrant.tech) - Vector database
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
<!-- CI trigger: Fri Aug 29 12:45:26 CEST 2025 -->
