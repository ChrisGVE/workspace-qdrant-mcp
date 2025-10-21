# Migration Guide: v0.2.x → v0.3.0

This guide provides step-by-step instructions for upgrading from workspace-qdrant-mcp v0.2.x to v0.3.0.

## Overview

Version 0.3.0 introduces significant architectural improvements and breaking changes. This migration is recommended for all users to benefit from:

- **Rust daemon (memexd)** for high-performance processing
- **Hybrid search** with Reciprocal Rank Fusion (RRF)
- **Multi-tenant collections** with project isolation
- **SQLite state management** replacing JSON files
- **Comprehensive testing framework**
- **LLM context injection system**

## Prerequisites

Before upgrading, ensure you have:

1. **Python 3.10+** (v0.3.0 requires Python ≥3.10)
2. **Rust toolchain** (for building the daemon)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
3. **Backup of existing data**
   ```bash
   # Backup Qdrant data
   qdrant-cli backup create --output ~/qdrant-backup-$(date +%Y%m%d).tar.gz

   # Backup configuration
   cp ~/.config/workspace-qdrant/config.yaml ~/config-backup-$(date +%Y%m%d).yaml
   ```

## Step 1: Review Breaking Changes

### Configuration Format Changes

**Old format (v0.2.x):**
```yaml
auto_ingestion:
  max_file_size_mb: 100
  timeout_seconds: 30
  project_collection: "projects_content"
```

**New format (v0.3.0):**
```yaml
auto_ingestion:
  max_file_size: "100MB"  # Explicit units required
  timeout: "30s"          # Explicit units required
  auto_create_project_collections: true  # Boolean flag
```

**Units supported:**
- **Size:** `B`, `KB`, `MB`, `GB`, `TB` (or `K`, `M`, `G`, `T`)
- **Time:** `ms`, `s`, `m`, `h`

### Collection Architecture Changes

**Old architecture:**
- Custom collection names via `project_collection` setting
- Manual collection management

**New architecture:**
- **PROJECT collections:** `_{project_id}` (auto-created by daemon)
- **LIBRARY collections:** `_{library_name}` (user-managed)
- **USER collections:** `{basename}-{type}` (optional)

### Storage Migration: JSON → SQLite

**Old location:**
```
~/.config/workspace-qdrant/watch_configs/*.json
```

**New location:**
```
~/.local/share/workspace-qdrant/daemon_state.db
```

Migration happens automatically on first run.

## Step 2: Update Configuration

### 2.1 Update config.yaml

1. **Locate your configuration:**
   ```bash
   # Default location
   ~/.config/workspace-qdrant/config.yaml
   ```

2. **Update timeout and size values:**
   ```bash
   # Use sed to add units (BACKUP FIRST!)
   sed -i.bak 's/max_file_size_mb: \([0-9]*\)/max_file_size: "\1MB"/' config.yaml
   sed -i.bak 's/timeout_seconds: \([0-9]*\)/timeout: "\1s"/' config.yaml
   ```

3. **Replace deprecated settings:**
   ```yaml
   # REMOVE:
   # project_collection: "projects_content"

   # ADD:
   auto_create_project_collections: true
   ```

4. **Validate configuration:**
   ```bash
   wqm config validate
   ```

### 2.2 Environment Variables

No changes required for environment variables (`QDRANT_URL`, `QDRANT_API_KEY`).

## Step 3: Install v0.3.0

### 3.1 Upgrade Package

```bash
# Using pip
pip install --upgrade workspace-qdrant-mcp

# Using uv (recommended)
uv sync --upgrade
```

### 3.2 Build Rust Daemon

```bash
# Build release version
cd src/rust/daemon/core
cargo build --release

# Verify installation
which memexd
# Should show: /usr/local/bin/memexd
```

### 3.3 Install Daemon Service

```bash
# Install as system service
wqm service install

# Start daemon
wqm service start

# Verify status
wqm service status
```

## Step 4: Migrate Data

### 4.1 Watch Folder Configuration

The migration from JSON to SQLite happens automatically:

```bash
# Check migration status
wqm watch list

# Verify all watch folders migrated
# Compare with old JSON files:
ls ~/.config/workspace-qdrant/watch_configs/
```

If any watch folders are missing, re-add them:

```bash
wqm watch add /path/to/project \
  --collection project-code \
  --patterns "*.py" "*.js" \
  --auto-ingest
```

### 4.2 Collection Migration

**If using default collection names:** No action required.

**If using custom collection names:**

1. **List existing collections:**
   ```bash
   wqm admin collections
   ```

2. **For each custom collection:**
   ```bash
   # Option 1: Keep as USER collection (requires basename)
   # Rename: my-project → myapp-code
   # (Manual process via Qdrant API or wqm library commands)

   # Option 2: Migrate to PROJECT collection
   # Let daemon auto-create _{project_id} collection
   # Re-ingest files from watch folders
   ```

3. **Verify migration:**
   ```bash
   wqm admin collections --verbose
   ```

## Step 5: Update MCP Configuration

### 5.1 Claude Desktop Config

Update `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "workspace-qdrant": {
      "command": "workspace-qdrant-mcp",
      "env": {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

No changes required from v0.2.x configuration.

### 5.2 Verify MCP Server

```bash
# Test server startup
workspace-qdrant-mcp --transport http --port 8000

# Should start without errors
# Check: http://localhost:8000/health
```

## Step 6: Test Migration

### 6.1 Verify Daemon

```bash
# Check daemon status
wqm service status

# View daemon logs
wqm service logs

# Check health
workspace-qdrant-health
```

### 6.2 Test Search

```bash
# Test hybrid search
wqm search project "your search query"

# Test with filters
wqm search project "query" --branch main --file-type code
```

### 6.3 Test MCP Tools

In Claude Desktop:

1. Try the `search` tool:
   ```
   Search for "authentication" in the current project
   ```

2. Try the `store` tool:
   ```
   Store this note: "Migration completed successfully"
   ```

3. Try the `manage` tool:
   ```
   Show me the current collections
   ```

## Step 7: Clean Up (Optional)

### 7.1 Remove Old JSON Files

```bash
# After verifying SQLite migration worked
rm -rf ~/.config/workspace-qdrant/watch_configs/
```

### 7.2 Remove Old Logs

```bash
# Old log location (if exists)
rm -rf ~/.local/share/workspace-qdrant/logs/*.old
```

## Troubleshooting

### Issue: Daemon won't start

**Symptom:** `wqm service status` shows "not running"

**Solution:**
```bash
# Check logs for errors
wqm service logs

# Common issues:
# 1. Port conflict (default: 50051)
wqm config set daemon.grpc_port 50052

# 2. Missing database directory
mkdir -p ~/.local/share/workspace-qdrant

# 3. Restart daemon
wqm service restart
```

### Issue: Watch folders not working

**Symptom:** Files not being ingested automatically

**Solution:**
```bash
# 1. Verify watch folder configuration
wqm watch list --verbose

# 2. Check daemon is watching
wqm watch status

# 3. Manually sync
wqm watch sync /path/to/project

# 4. Check daemon logs
wqm service logs | grep -i watch
```

### Issue: MCP server connection fails

**Symptom:** Claude Desktop shows "MCP server error"

**Solution:**
```bash
# 1. Test server manually
workspace-qdrant-mcp

# 2. Check stdio mode output
# All output should go to stderr, not stdout

# 3. Verify Qdrant connection
curl http://localhost:6333/health

# 4. Check Claude Desktop logs
# macOS: ~/Library/Logs/Claude/mcp*.log
# Windows: %APPDATA%/Claude/logs/mcp*.log
```

### Issue: Configuration validation fails

**Symptom:** `wqm config validate` shows errors

**Solution:**
```bash
# 1. Check specific error message
wqm config validate --verbose

# 2. Common fixes:
# - Missing units: Add "MB", "s", etc.
# - Deprecated settings: Remove project_collection
# - Invalid paths: Use absolute paths

# 3. Use default config as reference
cp assets/default_configuration.yaml ~/.config/workspace-qdrant/config.yaml
```

### Issue: Search returns no results

**Symptom:** `wqm search` or MCP search tool returns empty

**Solution:**
```bash
# 1. Check collections exist
wqm admin collections

# 2. Verify collection has documents
wqm admin collections --collection _{your-project-id} --stats

# 3. Check indexing status
wqm admin status

# 4. Re-index if needed
wqm ingest folder /path/to/project
```

## Performance Tuning

### Daemon Performance

```yaml
# config.yaml
daemon:
  max_concurrent_tasks: 10  # Increase for faster processing
  batch_size: 50            # Larger batches = faster ingestion
```

### Search Performance

```yaml
# config.yaml
search:
  default_limit: 10         # Fewer results = faster response
  enable_hybrid: true       # Hybrid search for better relevance
```

### Memory Usage

```yaml
# config.yaml
embeddings:
  batch_size: 32            # Reduce if memory constrained
  cache_size: 1000          # Reduce cache for lower memory
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check comprehensive troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. **Review architecture docs:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Search GitHub issues:** https://github.com/ChrisGVE/workspace-qdrant-mcp/issues
4. **Create new issue:** Include:
   - Version: `wqm --version`
   - Platform: `uname -a`
   - Logs: `wqm service logs`
   - Config (sanitized): `wqm config show`

## Rollback Instructions

If you need to rollback to v0.2.x:

```bash
# 1. Stop daemon
wqm service stop

# 2. Uninstall service
wqm service uninstall

# 3. Downgrade package
pip install workspace-qdrant-mcp==0.2.1

# 4. Restore configuration backup
cp ~/config-backup-YYYYMMDD.yaml ~/.config/workspace-qdrant/config.yaml

# 5. Restore Qdrant backup
qdrant-cli backup restore ~/qdrant-backup-YYYYMMDD.tar.gz
```

**Note:** SQLite data will remain, but will not be used by v0.2.x (which uses JSON files).

## Next Steps

After successful migration:

1. **Explore new features:** Check [CHANGELOG.md](CHANGELOG.md) for full feature list
2. **Configure LLM injection:** See context injection documentation
3. **Set up watch folders:** Automate document ingestion
4. **Optimize performance:** Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md#performance-troubleshooting)
5. **Update workflows:** Leverage new MCP tools and CLI commands

## Version Support

- **v0.3.x:** Current, fully supported
- **v0.2.x:** Maintenance only (critical bugs only)
- **v0.1.x:** End of life (upgrade recommended)
