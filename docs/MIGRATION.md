# Migration Guide: v0.2.x → v0.3.0

> **Note**: This guide includes information about CLI commands and daemon setup that are planned for v0.4.0 and not yet available. For current functionality, focus on the MCP server configuration sections.

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
- One collection per project (collection sprawl)

**New architecture (Unified Multi-Tenant Model):**
- **PROJECTS collection:** `_projects` - Single unified collection for ALL projects
  - Tenant isolation via `tenant_id` payload field (12-char hex from git URL or path hash)
  - Cross-project search available via `scope="all"` parameter
  - Payload indexing on `tenant_id` for O(1) filtering
- **LIBRARIES collection:** `_libraries` - Single unified collection for ALL reference docs
  - Tenant isolation via `library_name` payload field
  - Include in search via `include_libraries=True` parameter
- **USER collections:** `{basename}-{type}` - User-created notes (e.g., `myapp-notes`)
- **MEMORY collections:** `_memory`, `_agent_memory` - System and agent rules

**Key Benefits:**
- Only 4 collection types total (scalable to thousands of projects)
- Single HNSW index per collection type (efficient)
- Cross-project semantic search (powerful)
- Hard tenant filtering prevents data leakage (secure)

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

**Unified Collections:** v0.4.0 introduces the unified multi-tenant collection model. All project content now goes into a single `_projects` collection with tenant isolation via `tenant_id`.

**Migration Steps:**

1. **List existing collections:**
   ```bash
   wqm admin collections
   ```

2. **Identify old per-project collections:**
   ```bash
   # Old format: _{project_id} (12-char hex per project)
   # These will be migrated to unified _projects collection
   ```

3. **Run automatic migration:**
   ```bash
   # Migration script consolidates old collections into unified model
   wqm admin migrate-to-unified

   # This will:
   # - Create _projects collection if not exists
   # - Copy documents from _{project_id} collections to _projects
   # - Add tenant_id metadata to each document
   # - Optionally delete old collections after verification
   ```

4. **For custom USER collections:**
   ```bash
   # User collections ({basename}-{type}) remain unchanged
   # They are auto-enriched with project_id when accessed from project directory
   ```

5. **Verify migration:**
   ```bash
   wqm admin collections --verbose

   # Should show:
   # _projects (unified, contains all projects with tenant_id)
   # _libraries (unified, contains all libraries with library_name)
   # {user}-{type} collections (unchanged)
   ```

6. **Test search with new parameters:**
   ```bash
   # Search current project only (default)
   wqm search "authentication" --scope project

   # Search all projects
   wqm search "authentication" --scope all

   # Include library documentation
   wqm search "fastapi routing" --include-libraries
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
# Test hybrid search (default: current project scope)
wqm search "your search query"

# Test with scope options
wqm search "query" --scope project   # Current project only (default)
wqm search "query" --scope all       # All projects
wqm search "query" --include-libraries  # Include library docs

# Test with filters
wqm search "query" --branch main --file-type code

# Test cross-project search
wqm search "authentication patterns" --scope all --include-libraries
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
# 1. Check unified collections exist
wqm admin collections
# Should show: _projects, _libraries

# 2. Verify _projects collection has documents
wqm admin collections --collection _projects --stats

# 3. Check tenant_id for current project
wqm admin status
# Note the tenant_id (e.g., github_com_user_repo)

# 4. Verify documents exist for this tenant
wqm search "test" --scope all  # Search all projects
wqm search "test" --scope project  # Search current project only

# 5. If no documents, re-ingest
wqm ingest folder /path/to/project

# 6. Check if searching wrong scope
# Default is "project" - try "all" to search everything
wqm search "authentication" --scope all --include-libraries
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

- **v0.4.x:** Current, fully supported (v0.3.x compatibility maintained)
- **v0.3.x:** Fully supported
- **v0.2.x:** Maintenance only (critical bugs only)
- **v0.1.x:** End of life (upgrade recommended)

---

# Queue Migration Guide: Legacy → Unified Queue (v0.4.0)

> **Timeline:**
> - **v0.4.0**: Legacy queues deprecated, unified_queue is primary
> - **v0.5.0**: Legacy queue tables will be removed

## Overview

Version 0.4.0 introduces the **unified_queue** as the single queue system, replacing the legacy `ingestion_queue` and `content_ingestion_queue` tables. This migration:

- **Consolidates** two queue tables into one with polymorphic item types
- **Improves** idempotency key generation (cross-language compatible)
- **Adds** lease-based locking for concurrent processing
- **Enables** better queue health monitoring and metrics

## Deprecated Components

The following components are deprecated in v0.4.0 and will be removed in v0.5.0:

### Database Tables
- `ingestion_queue` → Use `unified_queue` with `item_type='file'` or `item_type='folder'`
- `content_ingestion_queue` → Use `unified_queue` with `item_type='content'`

### Python Classes
- `SQLiteQueueClient` (queue_client.py) → Use `SQLiteStateManager.enqueue_unified()`
- `_dual_write_to_legacy_queue()` → Dual-write mode disabled by default

### Configuration
- `queue_processor.enable_dual_write` → Set to `false` (default)

## Migration Steps

### Step 1: Check Current State

```bash
# Check if dual-write is enabled
grep enable_dual_write ~/.config/workspace-qdrant/config.yaml

# Check queue depths
wqm admin drift-report
```

### Step 2: Drain Legacy Queues

Ensure all items in legacy queues are processed:

```bash
# Check legacy queue status
sqlite3 ~/Library/Application\ Support/workspace-qdrant-mcp/state.db \
  "SELECT COUNT(*) as pending FROM ingestion_queue WHERE status='pending';"

sqlite3 ~/Library/Application\ Support/workspace-qdrant-mcp/state.db \
  "SELECT COUNT(*) as pending FROM content_ingestion_queue WHERE status='pending';"
```

Wait for counts to reach zero or manually process remaining items.

### Step 3: Disable Dual-Write

Update your configuration:

```yaml
# config.yaml
queue_processor:
  enable_dual_write: false  # Default in v0.4.0
```

### Step 4: Update Code (If Using Python API)

**Before (deprecated):**
```python
from workspace_qdrant_mcp.core.queue_client import SQLiteQueueClient

client = SQLiteQueueClient()
await client.initialize()
await client.enqueue_file("/path/to/file.py", "my-collection")
```

**After (unified queue):**
```python
from workspace_qdrant_mcp.core.sqlite_state_manager import SQLiteStateManager

state_manager = SQLiteStateManager()
await state_manager.initialize()
await state_manager.enqueue_unified(
    item_type="file",
    op="ingest",
    tenant_id="my-project",
    collection="my-collection",
    payload={"file_path": "/path/to/file.py"},
    priority=5,
    branch="main"
)
```

### Step 5: Run Phase 3 Cutover (Optional)

For automated migration with safety checks:

```bash
# Dry run first
./scripts/phase3_cutover.sh --dry-run

# Execute cutover
./scripts/phase3_cutover.sh --force
```

### Step 6: Verify Migration

```bash
# Check for drift
wqm admin drift-report --verbose

# Monitor queue metrics
wqm admin metrics
```

## Unified Queue Schema

The new `unified_queue` table structure:

```sql
CREATE TABLE unified_queue (
    queue_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,           -- 'file', 'folder', 'content', 'collection', 'search'
    op TEXT NOT NULL,                  -- 'ingest', 'update', 'delete', 'search', 'create', 'drop'
    tenant_id TEXT NOT NULL,
    collection TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'pending',
    idempotency_key TEXT NOT NULL UNIQUE,
    payload_json TEXT,
    branch TEXT,
    metadata TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    lease_until TEXT,
    worker_id TEXT,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    error_message TEXT,
    last_error_at TEXT
);
```

## Idempotency Key Format

The unified queue uses a standardized idempotency key format:

```
{item_type}:{op}:{tenant_id}:{collection}:{content_hash}
```

Example: `file:ingest:my-project:code:abc123def456`

This format is consistent across Python and Rust implementations.

## Rollback

If issues occur after migration:

```bash
# Re-enable dual-write
./scripts/phase3_cutover.sh --rollback
```

Or manually:

```yaml
# config.yaml
queue_processor:
  enable_dual_write: true
```

Then restart the daemon:
```bash
wqm service restart
```

## Cleanup (v0.5.0)

Once stable, legacy tables can be removed:

```bash
# Automated cleanup
./scripts/phase3_cutover.sh --drop-tables

# Or manual SQL
sqlite3 ~/Library/Application\ Support/workspace-qdrant-mcp/state.db \
  "DROP TABLE IF EXISTS ingestion_queue; DROP TABLE IF EXISTS content_ingestion_queue;"
```

## Troubleshooting

### Deprecation Warnings

If you see warnings like:
```
DeprecationWarning: SQLiteQueueClient is deprecated...
```

Update your code to use `SQLiteStateManager.enqueue_unified()`.

### Queue Drift

If `wqm admin drift-report` shows discrepancies:

1. Enable dual-write temporarily
2. Re-process drifted items
3. Disable dual-write once synchronized

### Missing Items

Check the unified queue for processing status:

```sql
SELECT status, COUNT(*)
FROM unified_queue
GROUP BY status;
```
