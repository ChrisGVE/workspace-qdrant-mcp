# ADR-001: Multi-Tenant Collection Architecture

**Status:** Accepted
**Date:** 2026-01-22
**Decision Makers:** Project maintainers
**Supersedes:** All prior collection naming conventions in PRD v1, v2, CONSOLIDATED_PRD_V2.md

## Context

The workspace-qdrant-mcp project has accumulated multiple conflicting collection naming schemes across PRD versions:

- **PRD v1**: Per-project collections (`{project}-scratchbook`, `{project}-docs`) with global collections
- **PRD v2**: Reserved names with underscore-prefix for libraries (`_library`)
- **CONSOLIDATED_PRD_V2**: Double-underscore for system collections (`__memory`), configurable names
- **PRDv3**: Configurable memory collection, context injector component

This fragmentation has led to:
- Data split across inconsistent collection names
- Memory rules not reliably retrieved
- Conflicting validation logic between components
- Maintenance burden across multiple code paths

## Decision

### Canonical Collection Set

The system shall use exactly **three collections** with multi-tenant isolation:

| Collection   | Purpose                                    | Tenant Isolation                |
|--------------|--------------------------------------------|---------------------------------|
| `memory`     | Global behavioral rules and instructions   | `project_id` tag (optional)     |
| `projects`   | Project code, docs, notes, scratchbook     | `project_id` + `branch` tags    |
| `libraries`  | Reference documentation and external docs  | `library_name` tag              |

### Deprecated Patterns

The following naming patterns are **deprecated** and shall be removed:

- `_{project_id}` - per-project collections with underscore prefix
- `_projects`, `_libraries`, `_memory` - unified collections with underscore prefix
- `__memory`, `__system` - double-underscore system collections
- `{project}-scratchbook`, `{project}-docs`, `{project}-code` - per-project suffixed collections
- `memory_{tenant_id}` - tenant-suffixed memory collections

### Tenant Isolation via Metadata

Multi-tenancy is achieved through **payload metadata filtering**, not separate collections:

```json
{
  "project_id": "workspace-qdrant-mcp",
  "branch": "main",
  "file_type": "python",
  "path": "src/python/server.py",
  "tags": ["workspace-qdrant-mcp.main.src.python"]
}
```

Required payload indexes for efficient filtering:
- `project_id` (keyword, `is_tenant: true`)
- `library_name` (keyword, `is_tenant: true`)
- `branch` (keyword)
- `tags` (keyword, for hierarchical filtering)

### Tag Schema

Tags use **dot-separated hierarchy**:

**Projects:**
```
<project_id>.<branch>[.<path_segments>...]

Examples:
- workspace-qdrant-mcp.main
- workspace-qdrant-mcp.main.src.python
- workspace-qdrant-mcp.feature-auth.tests
```

**Libraries:**
```
<library_name>[.<subfolder_segments>...]

Examples:
- python-docs
- python-docs.asyncio
- rust-book.ownership.borrowing
```

### Memory Scope

The `memory` collection is **global** for the initial release:
- All behavioral rules stored in single collection
- Project-specific rules distinguished by optional `project_id` tag
- Future enhancement: formal project-scoped memory with tag filtering

### Write Path Policy

**Daemon-only writes** are enforced:

1. All write operations route through the Rust daemon (memexd)
2. MCP server and CLI **never write directly to Qdrant**
3. When daemon is unavailable, writes queue to **SQLite state database**
4. Daemon processes queue on recovery
5. Transient daemon unresponsiveness does NOT trigger direct Qdrant writes

### Project Activation Policy

**Explicit activation required:**

1. MCP does NOT auto-register projects on startup
2. Projects require explicit `init` or `setup` command
3. MCP only notifies daemon about project activation/deactivation
4. Daemon owns all file watching and ingestion
5. Active projects receive priority 1 in daemon queue

### Library Deletion Policy

**Additive semantics for libraries:**

1. Library deletions do NOT remove vectors from Qdrant
2. Deletion events tracked in SQLite to prevent re-ingestion loops
3. Libraries can be explicitly purged via CLI command
4. Rationale: Reference docs rarely change; additive model simplifies sync

### Daemon Availability Policy

When daemon is unresponsive (2 failed gRPC attempts, 10s deadline each):

1. **Project queries**: Disabled (return error)
2. **Memory queries**: Disabled (skip injection)
3. **Library queries**: Always allowed (read-only, no daemon dependency)
4. **Write requests**: Queue to SQLite (do not drop)

Availability checks are **on-demand per interaction**, not periodic polling.

## Consequences

### Positive

- Single source of truth for collection architecture
- Simplified maintenance with 3 collections vs N per-project
- Efficient multi-tenant queries via indexed metadata
- Clear separation of concerns (daemon writes, MCP queries)
- Resilient fallback via SQLite queue

### Negative

- Migration required from existing per-project collections
- All components must align to new naming (breaking change)
- Tag cardinality may grow large in monorepos (mitigated by indexes)

### Risks

- **Tag explosion**: Large monorepos may create many tag combinations
  - Mitigation: Use facets with limits, SQLite tag index as source of truth
- **Branch drift**: Renamed branches leave orphaned tags
  - Mitigation: Periodic cleanup job, explicit branch management commands

## Implementation Notes

### Migration Path

1. Create new canonical collections (`memory`, `projects`, `libraries`)
2. Migrate data from legacy collections with transformed metadata
3. Update all collection references in Python MCP server
4. Update Rust daemon gRPC services
5. Remove legacy collection creation/validation code
6. Archive deprecated proto definitions

### Affected Code Locations

- `src/python/workspace_qdrant_mcp/server.py` - Collection routing
- `src/python/common/core/collection_naming.py` - Validation logic
- `src/rust/daemon/grpc/src/services/document_service.rs` - Memory routing
- `src/rust/daemon/proto/workspace_daemon.proto` - Collection constants

## References

- [Code Audit 2026-01-21](/.taskmaster/docs/code_audit.md) - Canonical Direction section
- [FIRST-PRINCIPLES.md](/FIRST-PRINCIPLES.md) - Principle 10: Daemon-only writes
- [Qdrant Payload Indexing](https://qdrant.tech/documentation/concepts/indexing/)
- [Qdrant Facets API](https://api.qdrant.tech/api-reference/points/facet)
