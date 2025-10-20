# workspace-qdrant-mcp CLI Reference

Complete command-line reference for workspace-qdrant-mcp (`wqm`) CLI tool.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Global Options](#global-options)
- [Command Categories](#command-categories)
  - [Service Management](#service-management)
  - [Memory Management](#memory-management)
  - [Admin Operations](#admin-operations)
  - [Configuration](#configuration)
  - [Watch Folders](#watch-folders)
  - [Document Ingestion](#document-ingestion)
  - [Search](#search)
  - [Library Management](#library-management)
  - [LSP Integration](#lsp-integration)
  - [Observability](#observability)
  - [Status & Monitoring](#status--monitoring)
- [Common Workflows](#common-workflows)
- [Environment Variables](#environment-variables)

## Overview

The `wqm` (Workspace Qdrant MCP) CLI provides comprehensive command-line tools for managing your semantic workspace, including:

- Daemon service management
- Memory rules and LLM behavior configuration
- Document ingestion and processing
- Collection management
- Real-time folder watching
- System health monitoring
- LSP server integration

## Installation

```bash
# Install via uv (recommended)
uv tool install workspace-qdrant-mcp

# Or via pip
pip install workspace-qdrant-mcp

# Verify installation
wqm --version
```

## Global Options

Available for all wqm commands:

```bash
-v, --version      # Show version information
--verbose          # Show verbose version information
-c, --config TEXT  # Custom configuration file
--debug            # Enable debug mode with verbose logging
--help             # Show help message
```

**Examples:**

```bash
# Check version
wqm --version

# Use custom config
wqm --config my-project.yaml service status

# Enable debug output
wqm --debug admin health
```

## Command Categories

### Service Management

Manage the memexd daemon service for continuous document processing.

#### `wqm service`

```bash
wqm service [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `install` - Install daemon as user service (systemd/launchd/Windows Service)
- `uninstall` - Uninstall daemon service
- `start` - Start daemon service
- `stop` - Stop daemon service
- `restart` - Restart daemon service
- `status` - Show daemon status
- `logs` - View daemon logs

**Examples:**

```bash
# Install and start daemon
wqm service install
wqm service start

# Check daemon status
wqm service status

# View recent logs
wqm service logs

# Restart after config changes
wqm service restart
```

### Memory Management

Manage memory rules and LLM behavior configuration.

#### `wqm memory`

```bash
wqm memory [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `list` - Show all memory rules
- `add` - Add new memory rule (preference or behavior)
- `edit` - Edit specific memory rule
- `remove` - Remove memory rule
- `tokens` - Show token usage statistics
- `trim` - Interactive token optimization
- `conflicts` - Detect and resolve memory conflicts
- `parse` - Parse conversational memory update
- `web` - Start web interface for memory curation

**Examples:**

```bash
# List all memory rules
wqm memory list

# Add a new coding preference
wqm memory add "Always use type hints in Python code"

# Check token usage
wqm memory tokens

# Interactive token optimization
wqm memory trim

# Detect conflicts in memory rules
wqm memory conflicts

# Start web interface for memory management
wqm memory web
```

### Admin Operations

System administration and configuration.

#### `wqm admin`

```bash
wqm admin [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `status` - Show comprehensive system status
- `config` - Configuration management
- `start-engine` - Start Rust processing engine
- `stop-engine` - Stop processing engine
- `restart-engine` - Restart engine with new configuration
- `collections` - List and manage collections
- `health` - Comprehensive health check

**Examples:**

```bash
# System status
wqm admin status

# Health check
wqm admin health

# List all collections
wqm admin collections

# Restart processing engine
wqm admin restart-engine
```

### Configuration

Configuration file management and validation.

#### `wqm config`

```bash
wqm config [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `show` - Show current configuration
- `get` - Get specific configuration value
- `set` - Set configuration value
- `edit` - Edit configuration in default editor
- `validate` - Validate configuration file
- `info` - Display configuration information
- `convert` - Convert configuration between formats
- `init-unified` - Initialize default configuration files
- `watch` - Watch configuration for changes
- `env-vars` - Show environment variable overrides
- `ingestion` - Ingestion configuration management
- `ingestion-show` - Show ingestion configuration
- `ingestion-edit` - Edit ingestion configuration
- `ingestion-validate` - Validate ingestion configuration
- `ingestion-reset` - Reset ingestion to defaults
- `ingestion-info` - Show ingestion system information

**Examples:**

```bash
# Show current configuration
wqm config show

# Get specific value
wqm config get qdrant.url

# Set configuration value
wqm config set embedding.model "BAAI/bge-base-en-v1.5"

# Edit config in editor
wqm config edit

# Validate configuration
wqm config validate

# Show environment variable overrides
wqm config env-vars

# Ingestion configuration
wqm config ingestion-show
wqm config ingestion-edit
wqm config ingestion-validate
```

### Watch Folders

Configure automatic folder watching and ingestion.

#### `wqm watch`

```bash
wqm watch [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `add` - Add folder to watch for automatic ingestion
- `list` - Show all active watches
- `remove` - Stop watching folder
- `status` - Watch activity and statistics
- `pause` - Pause all or specific watches
- `resume` - Resume paused watches
- `configure` - Configure existing watch
- `sync` - Manually sync watched folders

**Examples:**

```bash
# Add folder to watch
wqm watch add /path/to/docs --collection my-project-docs

# List all watches
wqm watch list

# Watch status
wqm watch status

# Pause specific watch
wqm watch pause /path/to/docs

# Resume all watches
wqm watch resume

# Manual sync
wqm watch sync

# Remove watch
wqm watch remove /path/to/docs
```

### Document Ingestion

Manual document processing and ingestion.

#### `wqm ingest`

```bash
wqm ingest [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `file` - Ingest single file
- `folder` - Ingest all files in folder
- `yaml` - Process completed YAML metadata file
- `generate-yaml` - Generate YAML metadata for library docs
- `web` - Crawl and ingest web pages
- `status` - Show ingestion status and statistics
- `validate` - Validate files for ingestion compatibility
- `smart` - Smart ingestion with auto-detection

**Examples:**

```bash
# Ingest single file
wqm ingest file document.pdf --collection my-project

# Ingest folder
wqm ingest folder /path/to/docs --collection my-project-docs

# Smart ingestion (auto-detection)
wqm ingest smart /path/to/mixed-docs

# Web crawling
wqm ingest web https://docs.example.com --collection external-docs

# Ingestion status
wqm ingest status

# Validate files before ingestion
wqm ingest validate /path/to/docs

# Generate YAML metadata
wqm ingest generate-yaml /path/to/library --output library-metadata.yaml
```

### Search

Command-line search interface with multiple modes.

#### `wqm search`

```bash
wqm search [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `project` - Search current project collections
- `collection` - Search specific collection
- `global` - Search global collections (library and system)
- `all` - Search all collections
- `memory` - Search memory rules and knowledge graph
- `research` - Advanced research mode with analysis

**Examples:**

```bash
# Search current project
wqm search project "authentication implementation"

# Search specific collection
wqm search collection my-project-docs "API documentation"

# Search all collections
wqm search all "error handling"

# Search memory rules
wqm search memory "coding preferences"

# Research mode with analysis
wqm search research "microservices architecture best practices"
```

### Library Management

Manage library collections for shared documentation.

#### `wqm library`

```bash
wqm library [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `list` - Show all library collections
- `create` - Create new library collection
- `remove` - Remove library collection
- `status` - Show library statistics and health
- `info` - Show detailed library information
- `rename` - Rename library collection
- `copy` - Copy library collection

**Examples:**

```bash
# List all libraries
wqm library list

# Create library
wqm library create python-stdlib

# Library information
wqm library info python-stdlib

# Rename library
wqm library rename old-name new-name

# Copy library
wqm library copy source-lib target-lib

# Remove library
wqm library remove deprecated-lib
```

### LSP Integration

LSP server management and monitoring.

#### `wqm lsp`

```bash
wqm lsp [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `status` - Show LSP server health and capabilities
- `install` - Guided LSP server installation
- `restart` - Restart specific LSP server
- `config` - LSP configuration management
- `diagnose` - Run troubleshooting and diagnostics
- `setup` - Interactive setup wizard
- `list` - List available and installed LSP servers
- `performance` - Monitor LSP server performance

**Examples:**

```bash
# LSP status
wqm lsp status

# Install LSP server
wqm lsp install pyright

# Setup wizard
wqm lsp setup

# Performance monitoring
wqm lsp performance

# Diagnose issues
wqm lsp diagnose

# Restart server
wqm lsp restart pyright
```

### Observability

System monitoring and health checks.

#### `wqm observability`

```bash
wqm observability [OPTIONS] COMMAND [ARGS]
```

**Commands:**

- `health` - Check system health status
- `metrics` - Display current system metrics
- `diagnostics` - Generate comprehensive diagnostics
- `monitor` - Run continuous monitoring mode

**Examples:**

```bash
# Health check
wqm observability health

# Current metrics
wqm observability metrics

# Comprehensive diagnostics
wqm observability diagnostics

# Continuous monitoring
wqm observability monitor
```

### Status & Monitoring

Processing status and user feedback system.

#### `wqm status`

```bash
wqm status [OPTIONS]
```

**Options:**

- `--history` - Show processing history
- `--queue` - Show detailed queue statistics
- `--watch` - Show watch folder status
- `--performance` - Show performance metrics
- `--live` - Enable live monitoring mode
- `--stream` - Enable real-time gRPC streaming (requires daemon)
- `-i, --interval INTEGER` - Live update interval in seconds (default: 5)
- `--grpc-host TEXT` - gRPC daemon host (default: 127.0.0.1)
- `--grpc-port INTEGER` - gRPC daemon port (default: 50051)
- `--export TEXT` - Export format: json, csv
- `-o, --output PATH` - Output file path
- `-c, --collection TEXT` - Filter by collection
- `--status TEXT` - Filter by status: success, failed, skipped, pending
- `-d, --days INTEGER` - Number of days for history (default: 7)
- `-l, --limit INTEGER` - Maximum records to show (default: 100)
- `-v, --verbose` - Show verbose output
- `-q, --quiet` - Minimal output

**Examples:**

```bash
# Basic status
wqm status

# Processing history
wqm status --history

# Queue statistics
wqm status --queue

# Watch folder status
wqm status --watch

# Performance metrics
wqm status --performance

# Live monitoring
wqm status --live

# Real-time streaming (requires daemon)
wqm status --stream

# Filter by collection
wqm status --collection my-project-docs

# Failed processing only
wqm status --status failed --days 7

# Export to JSON
wqm status --export json --output status-report.json

# Verbose output
wqm status --history --verbose
```

## Common Workflows

### Initial Setup

```bash
# Install daemon service
wqm service install
wqm service start

# Verify system health
wqm observability health
wqm admin status

# Configure watches
wqm watch add ./docs --collection my-project-docs
wqm watch add ./src --collection my-project-code
```

### Daily Development

```bash
# Check processing status
wqm status --queue

# Search for information
wqm search project "authentication flow"

# Add memory rule
wqm memory add "Use pytest fixtures for test setup"

# Manual file ingestion
wqm ingest file new-document.md --collection my-project
```

### Maintenance

```bash
# Health check
wqm observability diagnostics

# Daemon status
wqm service status

# Collection management
wqm admin collections

# Configuration validation
wqm config validate

# Performance monitoring
wqm status --performance --live
```

### Troubleshooting

```bash
# Comprehensive diagnostics
wqm observability diagnostics

# View daemon logs
wqm service logs

# LSP diagnostics
wqm lsp diagnose

# Check failed processing
wqm status --status failed --verbose

# Validate configuration
wqm config validate
```

## Environment Variables

Key environment variables that affect `wqm` behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_API_KEY` | _(none)_ | Qdrant API key (required for cloud) |
| `FASTEMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `WQM_CONFIG_PATH` | `~/.config/workspace-qdrant-mcp/config.yaml` | Configuration file path |
| `WQM_DAEMON_HOST` | `127.0.0.1` | Daemon gRPC host |
| `WQM_DAEMON_PORT` | `50051` | Daemon gRPC port |
| `WQM_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `WQM_DATA_DIR` | `~/.local/share/workspace-qdrant-mcp` | Data directory |

**Override configuration with environment variables:**

```bash
# Use different Qdrant server
export QDRANT_URL="https://my-qdrant-cloud.io"
export QDRANT_API_KEY="your-api-key"

# Use different embedding model
export FASTEMBED_MODEL="BAAI/bge-base-en-v1.5"

# Enable debug logging
export WQM_LOG_LEVEL="DEBUG"

# Run command with overrides
wqm admin status
```

---

**For MCP tool documentation, see [README.md](README.md#-mcp-tools)**

**For complete system documentation, see [README.md](README.md)**
