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
- 🛡️ **Enterprise Ready** - Comprehensive security scanning and quality assurance

## Table of Contents

- [✨ Key Features](#-key-features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [MCP Integration](#mcp-integration)
- [Configuration](#configuration)
- [Usage](#usage)
- [CLI Tools](#cli-tools)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

Get up and running in just a few minutes:

1. **Install the package**: `uv tool install workspace-qdrant-mcp`
2. **Run the setup wizard**: `workspace-qdrant-setup`
3. **Start using with Claude**: The wizard configures everything automatically

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

Collections are automatically created based on your project and configuration:

**Always Created:**

- `{project-name}-scratchbook` → Auto-created for notes, ideas, todos, and code snippets

**Project Collections:**

- `COLLECTIONS="project"` → creates `{project-name}-project`
- `COLLECTIONS="docs,tests"` → creates `{project-name}-docs`, `{project-name}-tests`

**Global Collections (User Choice):**

- `GLOBAL_COLLECTIONS="docs,references"` → creates `docs`, `references` (shared across projects)

**Example:** For project "my-app" with `COLLECTIONS="docs,tests"`:

- `my-app-scratchbook` (automatically created for notes)
- `my-app-docs` (project documentation)
- `my-app-tests` (test-related documents)
- `docs` (global documentation)
- `references` (global references)

### Scratchbook Collections

Every project automatically gets a `{project-name}-scratchbook` collection for capturing development thoughts and notes. This is your **personal development journal** for the project.

**What goes in scratchbook collections:**

- 📝 **Meeting notes** and action items
- 💡 **Ideas** and implementation thoughts
- ✅ **TODOs** and reminders
- 🔧 **Code snippets** and implementation patterns
- 🏗️ **Architecture decisions** and rationale
- 🐛 **Bug reports** and troubleshooting notes
- 📊 **Research findings** and links
- 🎯 **Project goals** and milestones

**Example scratchbook entries:**

```
"Discussed API rate limiting in team meeting - need to implement exponential backoff"
"Found solution for memory leak in worker threads - use weak references"
"TODO: Update deployment docs after container changes"
"Code snippet: async context manager pattern for database connections"
```

### Subproject Support (Git Submodules)

For repositories with **Git submodules**, additional collections are created automatically:

**Requirements:**

- Must set `WORKSPACE_QDRANT_WORKSPACE__GITHUB_USER=yourusername`
- Only submodules **owned by you** get collections (prevents vendor/third-party sprawl)
- Without `github_user` configured, only main project collections are created (conservative approach)

**Example with subprojects:**

```bash
# Repository: my-monorepo with submodules
# - frontend/ (github.com/myuser/frontend)
# - backend/ (github.com/myuser/backend)
# - vendor-lib/ (github.com/vendor/lib) ← ignored

# Collections created:
my-monorepo-scratchbook    # Main project notes
my-monorepo-project        # Main project docs
frontend-scratchbook       # Frontend notes
frontend-project          # Frontend docs
backend-scratchbook       # Backend notes
backend-project           # Backend docs
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

- **[API Reference](API.md)** - Complete MCP tools documentation
- **[Contributing Guide](CONTRIBUTING.md)** - Development setup and guidelines
- **[Release Process](docs/RELEASE_PROCESS.md)** - Automated releases and emergency procedures
- **[Trusted Publishing Setup](docs/TRUSTED_PUBLISHING_SETUP.md)** - PyPI security configuration
- **[Benchmarking](benchmarking/README.md)** - Performance testing and metrics

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

**📚 Documentation**: See [docs/RELEASE_PROCESS.md](docs/RELEASE_PROCESS.md) for complete release documentation and emergency procedures.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Setting up the development environment
- Running tests and benchmarks
- Code style and quality requirements
- Submitting pull requests

## 💬 Community

Join our community discussions for support, ideas, and collaboration:

- **[GitHub Discussions](https://github.com/ChrisGVE/workspace-qdrant-mcp/discussions)** - Community Q&A, feature ideas, and showcases
- **[Community Guidelines](docs/COMMUNITY_GUIDELINES.md)** - How we work together
- **[Discussion Guide](docs/DISCUSSIONS_GUIDE.md)** - Getting the most from community discussions

For bug reports and specific feature requests, please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Related Projects:**

- [claude-qdrant-mcp](https://github.com/marlian/claude-qdrant-mcp) - Original TypeScript implementation
- [Qdrant](https://qdrant.tech) - Vector database
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP server framework
<!-- CI trigger: Fri Aug 29 12:45:26 CEST 2025 -->
