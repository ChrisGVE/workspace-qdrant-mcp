# THIS PROJECT IS IN ACTIVE DEVELOPMENT AND IS NOT YET READY FOR PRODUCTION, BUT SOON!

# workspace-qdrant-mcp

**Project-scoped Qdrant MCP server with hybrid search and configurable collections**

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
- ⚙️ **Interactive Setup** - Guided configuration wizard with health checks
- 🚀 **High Performance** - Rust-powered components with evidence-based benchmarks
- 🌐 **Cross-Platform** - Native support for macOS (Intel/ARM), Linux (x86_64/ARM64), Windows (x86_64/ARM64)
- 🛡️ **Enterprise Ready** - Comprehensive security scanning and quality assurance

## 🔧 MCP Tools

workspace-qdrant-mcp provides 4 comprehensive MCP tools for vector database operations:

### 1. **store** - Content Storage
Store any type of content in the vector database with automatic embedding generation and metadata enrichment.
- Supports text, code, documentation, notes, and more
- Automatic project detection and collection routing
- Metadata enrichment (file_type, branch, project_id)
- **Daemon-first write architecture**: All writes route through the Rust daemon for consistency
- Background processing via Rust daemon for optimal performance

### 2. **search** - Hybrid Search
Search across collections with powerful hybrid semantic + keyword matching.
- **hybrid mode**: Combines semantic similarity with keyword matching (default)
- **semantic mode**: Pure vector similarity search for conceptual matches
- **exact mode**: Keyword and symbol exact matching
- Automatic result optimization using Reciprocal Rank Fusion (RRF)
- Branch filtering (search current branch or all branches)
- File type filtering (code, docs, tests, etc.)

### 3. **manage** - Collection Management
Manage collections, system status, and configuration via the daemon's gRPC interface.
- List all collections with statistics
- Create and delete collections through daemon
- Get workspace status and health information
- Initialize project collections
- Cleanup empty collections and optimize storage

### 4. **retrieve** - Direct Document Access
Retrieve documents by ID or metadata without search ranking.
- Direct document ID lookup
- Metadata-based filtering
- Branch and file type filtering
- Efficient bulk retrieval

All tools seamlessly integrate with Claude Desktop and Claude Code for natural language interaction.

## Table of Contents

- [✨ Key Features](#-key-features)
- [🔧 MCP Tools](#-mcp-tools)
- [Quick Start](#quick-start)
  - [Daemon Service Installation](#daemon-service-installation)
  - [Interactive Setup](#interactive-setup)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Daemon Service Setup](#daemon-service-setup)
- [MCP Integration](#mcp-integration)
- [Configuration](#configuration)
- [Usage](#usage)
- [CLI Tools](#cli-tools)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

### Daemon Service Installation

For production deployments with continuous document processing:

1. **Install the package**: `uv tool install workspace-qdrant-mcp`
2. **Install daemon service**: `wqm service install`
3. **Start the service**: `wqm service start`
4. **Verify installation**: `wqm service status`

The daemon service provides:
- ✅ Continuous document monitoring and processing
- ✅ Background embedding generation with file watching
- ✅ Automatic startup on system boot with crash recovery
- ✅ Robust error recovery and structured logging
- ✅ IPC communication for Python integration

**📖 Complete Installation Guide**: See [CLI Reference](CLI.md#service-management) for service setup

### Interactive Setup

For quick testing and development:

1. **Install the package**: `uv tool install workspace-qdrant-mcp`
2. **Run the setup wizard**: `workspace-qdrant-setup`
3. **Start using with Claude**: The wizard configures everything automatically

**Or use YAML configuration for project-specific setups:**
```bash
workspace-qdrant-mcp --config=my-project.yaml
```

### Docker Deployment (Coming Soon)

**Note:** Docker images will be available starting with v0.3.1. The complete Docker and Kubernetes infrastructure exists in the `docker/` directory. For v0.3.0, please use pip/uv installation methods above.

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

**After installation, run the setup wizard:**

```bash
workspace-qdrant-setup
```

This interactive wizard will guide you through configuration, test your setup, and get you ready to use the MCP server with Claude in minutes.

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

## Daemon Service Setup

The `memexd` daemon provides continuous document processing and monitoring capabilities for production deployments:

### Quick Service Installation

```bash
# Install daemon service (auto-detects platform)
wqm service install

# Start the service
wqm service start

# Verify installation
wqm service status
```

### Service Management

```bash
# Service control
wqm service start|stop|restart|status

# View logs
wqm service logs

# Health monitoring
workspace-qdrant-health --daemon
```

### Daemon Architecture

The unified daemon (`memexd`) provides:
- 📁 **Real-time file monitoring** with SQLite-driven watch configuration
- 🤖 **Background embedding generation** for optimal performance
- 🔄 **gRPC services**: SystemService, CollectionService, DocumentService (15 RPCs total)
- 🔌 **Single writer pattern**: All Qdrant writes route through daemon for consistency
- 🚀 **Automatic startup** on system boot with crash recovery
- 📊 **Health monitoring** with comprehensive metrics and status reporting

### gRPC Services Architecture

The daemon exposes three gRPC services for communication with MCP server and CLI:

**1. SystemService (7 RPCs)** - Health monitoring and lifecycle management:
- `HealthCheck` - Quick health status for monitoring/alerting
- `GetStatus` - Comprehensive system state snapshot
- `GetMetrics` - Current performance metrics
- `SendRefreshSignal` - Event-driven state change notifications
- `NotifyServerStatus` - MCP/CLI server lifecycle events
- `PauseAllWatchers` / `ResumeAllWatchers` - Master file watcher control

**2. CollectionService (5 RPCs)** - Qdrant collection lifecycle:
- `CreateCollection` - Create collection with proper configuration
- `DeleteCollection` - Remove collection and all data
- `CreateCollectionAlias` - Create alias for tenant_id changes
- `DeleteCollectionAlias` - Remove collection alias
- `RenameCollectionAlias` - Atomically rename alias

**3. DocumentService (3 RPCs)** - Direct text ingestion (non-file content):
- `IngestText` - Synchronous text ingestion with chunking
- `UpdateText` - Update previously ingested text
- `DeleteText` - Delete ingested document

**Design Principles:**
- **Single writer pattern**: Only daemon writes to Qdrant
- **Queue-based async processing**: File operations via SQLite queue
- **Direct sync ingestion**: Text content via gRPC IngestText
- **Event-driven refresh**: Lightweight signals for state changes

**📖 For detailed daemon installation:** See [CLI Reference](CLI.md#service-management) - Covers systemd (Linux), launchd (macOS), and Windows Service with security configurations.

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

### Collection Naming

workspace-qdrant-mcp uses **four collection types** with strict naming conventions and **required non-empty basenames** for proper daemon routing and metadata enrichment.

**Project Collections** (Auto-created for file watching):

- Pattern: `_{project_id}` where `project_id` is 12-character hex hash
- Example: `_a1b2c3d4e5f6` (single collection per project)
- Basename: `"code"` (required)
- All content types in one collection, differentiated by metadata

**User Collections** (Shared across projects):

- Pattern: `{basename}-{type}` (e.g., `myapp-notes`, `work-scratchbook`)
- Basename: `"notes"` (default, customizable)
- Auto-enriched with current project_id when accessed from project directory
- Searchable across all projects with optional project filtering

**Library Collections** (External documentation):

- Pattern: `_{library_name}` (e.g., `_numpy`, `_react`, `_fastapi`)
- Basename: `"lib"` (required)
- Global shared documentation and references
- Distinguished from PROJECT collections by length (>13 chars)

**Memory Collections** (Meta-level data):

- Fixed names: `_memory`, `_agent_memory`
- Basename: `"memory"` (required)
- Agent conversation history and user preferences

**📖 For complete documentation:** See [Collection Naming Guide](docs/COLLECTION_NAMING.md) for detailed naming conventions, validation rules, and usage examples.

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

For repositories with **Git submodules**, the daemon automatically detects and tracks each submodule as a separate project:

**Requirements:**

- Must set `WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER=yourusername`
- Only submodules **owned by you** are tracked (prevents vendor/third-party sprawl)
- Without `github_user` configured, only main project collections are created (conservative approach)

**How it works:**

Each submodule gets its own project_id (normalized git remote URL) and separate project-scoped collections. The daemon identifies ownership by comparing git remote usernames.

**Example with subprojects:**

```bash
# Repository: my-monorepo with submodules
# - frontend/ (github.com/myuser/frontend)
# - backend/ (github.com/myuser/backend)
# - vendor-lib/ (github.com/vendor/lib) ← ignored (different owner)

# Collections created (assuming COLLECTIONS="code,docs"):
monorepo-hash_code         # Main project code
monorepo-hash_docs         # Main project docs
frontend-hash_code         # Frontend code (separate project_id)
frontend-hash_docs         # Frontend docs
backend-hash_code          # Backend code (separate project_id)
backend-hash_docs          # Backend docs
# No collections for vendor-lib (different owner)
```

### Cross-Collection Search

**Important: All MCP search commands search across ALL your collections simultaneously.**

When you use Claude with commands like:

- "Search my project for authentication code"
- "Find all references to the payment API"
- "What documentation do I have about deployment?"

The search **automatically includes:**

- ✅ Main project collections (`project-name-*`)
- ✅ All subproject collections (`subproject-*`)
- ✅ Global collections (`docs`, `references`, etc.)
- ✅ All scratchbook collections (your notes and ideas)

This gives you **unified search across your entire workspace** - you don't need to specify which collection to search.

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

### Interactive Setup Wizard

Get up and running in minutes with the guided setup wizard:

```bash
# Interactive setup with guided prompts
workspace-qdrant-setup

# Advanced mode with all configuration options
workspace-qdrant-setup --advanced

# Non-interactive mode for automation
workspace-qdrant-setup --non-interactive
```

The setup wizard:

- Tests Qdrant connectivity and validates configuration
- Helps choose optimal embedding models
- Configures Claude Desktop integration automatically
- Creates sample documents for immediate testing
- Provides final system verification

### Diagnostics and Testing

Comprehensive troubleshooting and health monitoring:

```bash
# Full system diagnostics
workspace-qdrant-test

# Test specific components
workspace-qdrant-test --component qdrant
workspace-qdrant-test --component embedding

# Include performance benchmarks
workspace-qdrant-test --benchmark

# Generate detailed report
workspace-qdrant-test --report diagnostic_report.json
```

### Health Monitoring

Real-time system health and performance monitoring:

```bash
# One-time health check
workspace-qdrant-health

# Continuous monitoring with live dashboard
workspace-qdrant-health --watch

# Detailed analysis with optimization recommendations
workspace-qdrant-health --analyze

# Generate health report
workspace-qdrant-health --report health_report.json
```

### Collection Management

Use `wqutil` for collection management and administration:

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

### Document Ingestion

Batch process documents for immediate searchability:

```bash
# Ingest documents from a directory
workspace-qdrant-ingest /path/to/docs --collection my-project

# Process specific formats only
workspace-qdrant-ingest /path/to/docs -c my-project -f pdf,md

# Preview what would be processed (dry run)
workspace-qdrant-ingest /path/to/docs -c my-project --dry-run
```

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** - System architecture, unified daemon design, and gRPC protocol
- **[Collection Naming Guide](docs/COLLECTION_NAMING.md)** - Collection types, naming conventions, and basename requirements
- **[CLI Reference](CLI.md)** - Complete command-line reference for all `wqm` commands
- **[API Reference](API.md)** - Complete MCP tools documentation
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Comprehensive troubleshooting and debugging
- **[Migration Guide](MIGRATION.md)** - v0.2.x to v0.3.0 upgrade instructions
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup, building from source, testing
- **[CI/CD Processes](docs/ci-cd-processes.md)** - Automated releases and deployment
- **[Benchmarking](tests/benchmarks/README.md)** - Performance testing and metrics
- **[Protocol Definition](src/rust/daemon/proto/workspace_daemon.proto)** - Complete gRPC protocol specification

## Troubleshooting

**Quick Diagnostics:**

```bash
# Run comprehensive system diagnostics
workspace-qdrant-test

# Get real-time health status
workspace-qdrant-health

# Run setup wizard to reconfigure
workspace-qdrant-setup
```

**Connection Issues:**

```bash
# Test Qdrant connectivity specifically
workspace-qdrant-test --component qdrant

# Verify Qdrant is running
curl http://localhost:6333/collections

# Validate complete configuration
workspace-qdrant-validate
```

**Performance Issues:**

```bash
# Run performance benchmarks
workspace-qdrant-test --benchmark

# Monitor system resources
workspace-qdrant-health --watch

# Get optimization recommendations
workspace-qdrant-health --analyze
```

**Collection Issues:**

```bash
# List current collections
wqutil list-collections

# Check project detection
wqutil workspace-status
```

For detailed troubleshooting, see [API.md](API.md#troubleshooting).

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
