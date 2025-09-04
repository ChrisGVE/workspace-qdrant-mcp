# Installation Guide

## UV Tool Installation (Recommended)

workspace-qdrant-mcp is designed for seamless installation using UV:

```bash
# Install from local source (development)
uv tool install --editable .

# Install from repository (once published)
uv tool install workspace-qdrant-mcp
```

## Available Commands

After installation, three CLI tools will be available:

### 1. Main MCP Server
```bash
workspace-qdrant-mcp --help
workspace-qdrant-mcp --host 127.0.0.1 --port 8000
workspace-qdrant-mcp --config-file ./custom.toml
```

### 2. Configuration Validator
```bash
workspace-qdrant-validate --help
workspace-qdrant-validate --verbose
workspace-qdrant-validate --config /path/to/config.toml
workspace-qdrant-validate --guide  # Show setup instructions
```

### 3. Administrative CLI
```bash
workspace-qdrant-admin --help
workspace-qdrant-admin list-collections --stats
workspace-qdrant-admin delete-collection my-collection
workspace-qdrant-admin search "query text" --limit 10
workspace-qdrant-admin reset-project --dry-run
workspace-qdrant-admin health
```

## Prerequisites

1. **Python 3.9+**: Required for compatibility
2. **Qdrant Database**: Either local or remote instance
3. **Environment Variables** (optional):
   - `QDRANT_URL`: Qdrant endpoint (default: http://localhost:6333)
   - `OPENAI_API_KEY`: For OpenAI embeddings (optional)
   - `CONFIG_FILE`: Custom configuration file path

## Quick Start

1. **Install the package:**
   ```bash
   uv tool install workspace-qdrant-mcp
   ```

2. **Start Qdrant** (if running locally):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

3. **Validate configuration:**
   ```bash
   workspace-qdrant-validate --guide
   ```

4. **Start the MCP server:**
   ```bash
   workspace-qdrant-mcp
   ```

## Verification

After installation, verify all components work:

```bash
# Check main server help
workspace-qdrant-mcp --help

# Check validator
workspace-qdrant-validate --help

# Check admin tools
workspace-qdrant-admin --help

# Test configuration validation
workspace-qdrant-validate --verbose
```

## Development Installation

For development work:

```bash
# Clone and install in editable mode
git clone <repository-url>
cd workspace-qdrant-mcp
uv tool install --editable .

# Install development dependencies
uv sync --extra dev
```

## Configuration

The package supports multiple configuration methods:
1. Environment variables
2. Configuration files (TOML/JSON)
3. Command-line arguments

Use the validation tool to check your setup:
```bash
workspace-qdrant-validate --verbose --guide
```

## Troubleshooting

- **Command not found**: Ensure UV's tool bin directory is in your PATH
- **Permission errors**: Use `--user` flag if needed
- **Configuration issues**: Run `workspace-qdrant-validate --guide`
- **Connection problems**: Check Qdrant URL and network connectivity

For detailed help on any command, use the `--help` flag.