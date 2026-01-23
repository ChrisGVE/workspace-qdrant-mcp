# ADR-001: Canonical Collection Architecture

**Status:** Accepted
**Date:** 2026-01-23
**Deciders:** Architecture Team
**Supersedes:** Multi-tenant Collection Schema v1.0, COLLECTION_NAMING.md (partial)

## Context

The workspace-qdrant-mcp system evolved through several collection naming iterations:

1. **v0.1**: Per-project collections (`_{project_id}` for each project)
2. **v0.2**: Unified collections with underscore prefix (`_projects`, `_libraries`, `_memory`)
3. **v0.3**: Current state - mixed patterns causing confusion

This inconsistency creates:
- Code maintenance burden (multiple naming patterns to support)
- Developer confusion about which pattern to use
- Validation complexity in the Rust daemon
- Test brittleness from inconsistent assertions

## Decision

Adopt **canonical collection names without underscore prefix**:

| Collection | Canonical Name | Purpose |
|------------|---------------|---------|
| Projects | `projects` | All project code, docs, tests, configs |
| Libraries | `libraries` | External library/framework documentation |
| Memory | `memory` | LLM behavioral rules and preferences |

### Key Architectural Decisions

#### 1. No Underscore Prefix

**Decision:** Collection names use NO underscore prefix.

**Rationale:**
- Cleaner, more intuitive naming
- Aligns with Qdrant community conventions
- Reduces confusion between collection types
- Simplifies validation logic

**Migration:**
- `_projects` → `projects`
- `_libraries` → `libraries`
- `_memory` → `memory`

#### 2. Multi-Tenant Isolation via Metadata

**Decision:** All tenant isolation uses payload metadata filtering, NOT separate collections.

**Implementation:**
```python
# Projects: isolated by project_id
search(collection="projects", filter={"project_id": "a1b2c3d4e5f6"})

# Libraries: isolated by library_name
search(collection="libraries", filter={"library_name": "numpy"})
```

**Benefits:**
- Qdrant optimizes `is_tenant=True` indexed fields
- Cross-project search is trivial (remove filter)
- Fewer collections = better resource utilization

#### 3. Daemon-Only Writes (First Principle 10)

**Decision:** ALL Qdrant write operations MUST route through the Rust daemon.

**Exception:** Memory collection allows direct writes (meta-level data).

**Write Path:**
```
MCP Server → SQLite Queue → Daemon → Qdrant
         ↓
    (fallback with warning)
         ↓
    Direct Qdrant Write
```

**Fallback Behavior:**
- When daemon unavailable, content queued in SQLite
- Daemon processes queue when available
- Clear logging indicates fallback mode

#### 4. Explicit Project Activation

**Decision:** Projects must be explicitly activated; NO auto-registration.

**Rationale:**
- Prevents accidental registration of transient directories
- Clear user intent for project tracking
- Enables proper daemon session management
- Supports priority-based ingestion

**Activation Flow:**
```python
# MCP
manage(action="activate_project")                    # Current directory
manage(action="activate_project", project_path="/path")  # Specific path

# Deactivation
manage(action="deactivate_project")
```

#### 5. Dot-Separated Tag Hierarchy

**Decision:** Use dot-separated tags for hierarchical organization.

**Format:** `main_tag.sub_tag`

**Examples:**
- Projects: `workspace-qdrant-mcp.feature-auth` (project_id.branch)
- Libraries: `numpy.1.24.0` (library_name.version)

**Benefits:**
- Hierarchical filtering (prefix matching)
- Clear organization within collections
- Enables branch-scoped and version-scoped queries

#### 6. Additive Library Deletion Policy

**Decision:** Libraries are NEVER physically deleted from Qdrant.

**Implementation:**
- Mark as deleted: `deleted=true`, `deleted_at=timestamp`
- Filter out deleted by default in search
- Allow searching deleted with explicit flag
- Re-ingestion clears deletion markers

**Rationale:**
- Preserves historical context
- Enables undo/restore operations
- Audit trail for compliance

## Implementation Constants

### Python (server.py)

```python
# Canonical collection names (ADR-001)
CANONICAL_COLLECTIONS = {
    "projects": "projects",    # All project content
    "libraries": "libraries",  # Library documentation
    "memory": "memory",        # LLM rules and preferences
}

def get_canonical_collection(collection_type: str) -> str:
    """Get canonical collection name for a type."""
    if collection_type not in CANONICAL_COLLECTIONS:
        raise ValueError(f"Unknown collection type: {collection_type}")
    return CANONICAL_COLLECTIONS[collection_type]

def validate_collection_name(name: str) -> bool:
    """Validate collection name is canonical (not deprecated)."""
    deprecated_patterns = ["_projects", "_libraries", "_memory", "_agent_memory"]
    if name in deprecated_patterns or name.startswith("__"):
        return False
    return True
```

### Rust (daemon)

```rust
/// Canonical collection names (ADR-001)
pub const CANONICAL_PROJECTS: &str = "projects";
pub const CANONICAL_LIBRARIES: &str = "libraries";
pub const CANONICAL_MEMORY: &str = "memory";

/// Validate collection name is not deprecated
pub fn validate_collection_name(name: &str) -> Result<(), ValidationError> {
    let deprecated = ["_projects", "_libraries", "_memory", "_agent_memory"];
    if deprecated.contains(&name) || name.starts_with("__") {
        return Err(ValidationError::DeprecatedCollectionName(name.to_string()));
    }
    Ok(())
}
```

## Deprecated Patterns

The following patterns are **DEPRECATED** and should be rejected with helpful error messages:

| Deprecated Pattern | Canonical Replacement | Migration Action |
|-------------------|----------------------|------------------|
| `_projects` | `projects` | Rename collection |
| `_libraries` | `libraries` | Rename collection |
| `_memory` | `memory` | Rename collection |
| `_agent_memory` | `memory` | Merge into `memory` |
| `_{project_id}` | `projects` + metadata | Migrate data with `project_id` field |
| `{basename}-{type}` | `projects` + metadata | Migrate data with `source` field |

## Migration Path

### Phase 1: Add Validation (Non-Breaking)
1. Add `validate_collection_name()` with warnings for deprecated patterns
2. Log deprecation warnings but allow operations to proceed
3. Update documentation to reference canonical names

### Phase 2: Update Code Paths
1. Rename `UNIFIED_COLLECTIONS` to `CANONICAL_COLLECTIONS`
2. Remove underscore prefixes from collection names
3. Update all tests to use canonical names
4. Update daemon validation to enforce canonical names

### Phase 3: Data Migration
1. Provide `wqm admin migrate-to-canonical` command
2. Copy data from deprecated to canonical collections
3. Update metadata fields as needed
4. Verify data integrity

### Phase 4: Remove Deprecated Support
1. Convert warnings to errors
2. Remove fallback code paths
3. Delete deprecated collections after verification

## Consequences

### Positive
- Cleaner, more consistent codebase
- Reduced confusion for developers and users
- Better alignment with Qdrant conventions
- Simplified validation and routing logic
- Improved testability

### Negative
- Breaking change requires migration
- Temporary support for both patterns during transition
- Documentation updates needed across project

### Risks
- Data loss if migration not handled carefully
- User confusion during transition period
- Potential for missed code paths using old names

## Related Documents

- `docs/architecture/multi-tenant-collection-schema.md` - Detailed schema specification
- `docs/COLLECTION_NAMING.md` - Collection naming guide (to be updated)
- `FIRST-PRINCIPLES.md` - Principle 10 (Daemon-Only Writes)
- `CLAUDE.md` - Pending architectural decisions section

## Changelog

| Date | Change |
|------|--------|
| 2026-01-23 | Initial ADR created |
