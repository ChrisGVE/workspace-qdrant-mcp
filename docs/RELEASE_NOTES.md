# Release Notes: workspace-qdrant-mcp v0.4.0

## What's New

Version 0.4.0 introduces a major architectural evolution: **Unified Multi-Tenant Collections**. This release transforms how projects and libraries are organized, providing better scalability, simpler management, and more powerful cross-project search capabilities.

## Highlights

### Unified Multi-Tenant Architecture

**What it means for you:** Simpler collection management with better scalability and cross-project search.

**Before (v0.3.x):**
- One collection per project: `_abc123`, `_def456`, `_ghi789`
- One collection per library: `_fastapi`, `_react`, `_numpy`
- 100 projects = 100+ collections to manage
- Cross-project search required querying multiple collections

**After (v0.4.0):**
- Single `_projects` collection for ALL project content
- Single `_libraries` collection for ALL library documentation
- Tenant isolation via `tenant_id` payload field
- Cross-project search with a single query

**Benefits:**
- **Scalable:** Handle thousands of projects without collection sprawl
- **Efficient:** Single HNSW index per collection type (better memory utilization)
- **Discoverable:** Semantic search across all projects with one query
- **Simple:** Fewer collections to manage and monitor
- **Secure:** Hard tenant filtering prevents data leakage

### Automatic Session Management

**What it means for you:** Projects are automatically prioritized based on active development.

New ProjectService gRPC API handles project lifecycle:
- **Automatic Registration:** Projects registered when MCP server starts
- **Priority Queuing:** Active sessions get HIGH priority processing
- **Heartbeat Monitoring:** 30-second heartbeats keep sessions alive
- **Orphan Detection:** Stale sessions detected and cleaned up automatically

```bash
# Check project status
wqm admin projects

# View active sessions
wqm admin status
```

### Enhanced Search Scoping

**What it means for you:** More control over what you search.

New search parameters for precise scoping:

```python
# Search current project only (default)
search(query="authentication", scope="project")

# Search all projects
search(query="rate limiting", scope="all")

# Include library documentation
search(query="FastAPI routing", include_libraries=True)

# Combined: search everything
search(query="JWT tokens", scope="all", include_libraries=True)
```

**CLI equivalent:**
```bash
wqm search "authentication"                              # Current project
wqm search "authentication" --scope all                  # All projects
wqm search "authentication" --include-libraries          # With libraries
wqm search "authentication" --scope all --include-libraries  # Everything
```

### Branch-Aware Search

**What it means for you:** Find code across branches or filter to specific ones.

```python
# Search specific branch
search(query="feature code", branch="feature/auth")

# Search all branches (find deleted code)
search(query="deprecated function", branch="*")

# Default: current branch only
search(query="new feature")
```

## What Changed

### Breaking Changes

#### Collection Architecture (Migration Required)

| Aspect | v0.3.x | v0.4.0 |
|--------|--------|--------|
| Project collections | `_{project_id}` per project | Single `_projects` collection |
| Library collections | `_{library_name}` per library | Single `_libraries` collection |
| Isolation method | Separate collections | `tenant_id` payload filtering |
| Cross-project search | Multiple queries | Single query with `scope="all"` |

#### Search API Changes

| Parameter | v0.3.x | v0.4.0 |
|-----------|--------|--------|
| Default scope | All content | Current project only |
| `scope` parameter | Not available | `"project"`, `"global"`, `"all"` |
| `include_libraries` | Always included | Opt-in with `include_libraries=True` |
| Response fields | Basic | Includes `tenant_id`, `collections_searched` |

**Migration example:**
```python
# v0.3.x behavior (search everything)
search(query="authentication")

# v0.4.0 equivalent (explicit scope)
search(query="authentication", scope="all", include_libraries=True)
```

### Automatic Migration

Run the migration command to transition existing data:

```bash
# Preview what will change
wqm admin migrate-to-unified --dry-run

# Execute migration
wqm admin migrate-to-unified

# Verify migration completed
wqm admin collections --verbose
```

**Migration process:**
1. Creates `_projects` and `_libraries` collections
2. Copies documents with added `tenant_id` metadata
3. Preserves all existing content and embeddings
4. Optionally removes old collections after verification

## Who Should Upgrade

### Recommended for:

- **Users with many projects** - Dramatic reduction in collection count
- **Teams needing cross-project search** - Find patterns across your codebase
- **Anyone wanting simpler management** - Fewer collections to monitor
- **Users who forgot to include libraries** - Explicit `include_libraries` prevents confusion

### Consider carefully if:

- **Heavy reliance on collection-per-project semantics** - Test migration thoroughly
- **Custom collection naming conventions** - Review migration plan first
- **Production systems with tight SLAs** - Plan migration during maintenance window

## Upgrade Guide

### Quick Upgrade (10 minutes)

```bash
# 1. Backup current data
wqm backup create --output ~/backup-v03.tar.gz

# 2. Upgrade package
pip install --upgrade workspace-qdrant-mcp

# 3. Preview migration
wqm admin migrate-to-unified --dry-run

# 4. Execute migration
wqm admin migrate-to-unified

# 5. Restart services
wqm service restart

# 6. Verify
wqm admin collections --verbose
wqm admin status
```

### Update Search Queries

If you were relying on the old default search behavior:

```python
# OLD: Searched everything by default
results = search(query="my query")

# NEW: Add explicit scope for same behavior
results = search(query="my query", scope="all", include_libraries=True)
```

## Performance Comparison

| Metric | v0.3.x (100 projects) | v0.4.0 | Improvement |
|--------|----------------------|--------|-------------|
| **Collections** | 100+ | 4 | 96% reduction |
| **Memory (indexes)** | 2.5 GB | 400 MB | 84% reduction |
| **Cross-project search** | 100 queries | 1 query | 99% reduction |
| **Collection management** | Manual | Automatic | Eliminated |

*Based on typical usage with 100 projects, 10 libraries*

## New Features Summary

### ProjectService gRPC API (5 RPCs)

| RPC | Purpose |
|-----|---------|
| `RegisterProject` | Register project for high-priority processing |
| `DeprioritizeProject` | Decrement session count on graceful shutdown |
| `Heartbeat` | Keep session alive (30s interval, 60s timeout) |
| `GetProjectStatus` | Query project status, priority, sessions |
| `ListProjects` | List all projects with filtering |

### Search Enhancements

| Feature | Description |
|---------|-------------|
| `scope` parameter | Control search scope (`project`, `global`, `all`) |
| `include_libraries` | Opt-in library documentation in results |
| `branch` parameter | Filter by git branch (supports `*` wildcard) |
| `collections_searched` | Response field showing which collections were queried |

### Session Lifecycle

| State | Description |
|-------|-------------|
| `ACTIVE` | MCP server connected, heartbeat received |
| `IDLE` | Registered but no active sessions |
| `ORPHANED` | Missed heartbeat for 60+ seconds |

## Documentation

### New Documentation

- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - Comprehensive multi-tenant search examples
- **[docs/multitenancy_architecture.md](docs/multitenancy_architecture.md)** - Architecture deep-dive
- **[docs/GRPC_API.md](docs/GRPC_API.md)** - Complete gRPC API reference (20 RPCs)

### Updated Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Complete v0.4.0 change history
- **[README.md](README.md)** - Updated architecture overview
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Session lifecycle diagrams

## What's Next

Planned for v0.5.0:
- **Docker images** - Container deployment support
- **Kubernetes Helm chart** - Enterprise deployment
- **Enhanced analytics** - Usage insights dashboard
- **Plugin system** - Custom document parsers

## Support

Need help with the upgrade?

- **Migration Guide:** [MIGRATION.md](MIGRATION.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Search Examples:** [docs/EXAMPLES.md](docs/EXAMPLES.md)
- **GitHub Issues:** https://github.com/ChrisGVE/workspace-qdrant-mcp/issues

## Thank You

Thank you for using workspace-qdrant-mcp! The v0.4.0 multi-tenant architecture makes managing multiple projects dramatically simpler while enabling powerful cross-project discovery.

**Happy coding!**

---

*Released: 2025-01-19*
*Download: `pip install workspace-qdrant-mcp==0.4.0`*
