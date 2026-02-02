# ADR-003: Daemon Owns SQLite Database

**Status:** Accepted
**Date:** 2026-01-30
**Deciders:** Architecture Team
**Related:** ADR-002 (Daemon-Only Write Policy)

## Context

The workspace-qdrant-mcp system uses SQLite for state persistence, queue management, and watch configuration. The Rust daemon is the sole owner of all database operations, including schema creation and migrations.

Key tables managed by the daemon:
- `unified_queue` - Consolidated write queue for all operations
- `watch_folders` - File watch configuration
- `projects` - Project metadata and tracking
- `schema_version` - Database version tracking

## Decision

**The Rust daemon (memexd) is the sole owner of the SQLite database.**

### Ownership Responsibilities

| Responsibility | Owner | Details |
|----------------|-------|---------|
| Database file creation | Daemon | Creates `state.db` if absent |
| Table creation | Daemon | Creates ALL tables on startup |
| Schema migrations | Daemon | Handles all version upgrades |
| Schema versioning | Daemon | Maintains `schema_version` table |
| Index creation | Daemon | Creates all indexes |

### Other Components

**MCP Server (TypeScript):**
- Reads from tables created by daemon
- Writes to: `unified_queue`, `watch_folders`
- Must NOT create tables or run migrations
- Must handle "table not found" errors gracefully (daemon not yet started)

**CLI (wqm):**
- Reads from tables created by daemon
- Writes to: `unified_queue`, `watch_folders`
- Must NOT create tables or run migrations
- Must use consistent database path: `~/.workspace-qdrant/state.db`

### Database Path

**Canonical path:** `~/.workspace-qdrant/state.db`

The daemon reads this from configuration:
```yaml
database:
  path: ~/.workspace-qdrant/state.db
```

All components MUST use this same path. No platform-specific variations.

### Schema Version Table

```sql
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    applied_by TEXT DEFAULT 'daemon'
);
```

### Startup Behavior

**Daemon startup:**
1. Read database path from configuration
2. Create database file if absent
3. Check `schema_version` table
4. Run migrations if needed
5. Create missing tables
6. Start normal operation

**MCP Server startup:**
1. Attempt to connect to database
2. If database missing → log warning, queue operations will fail gracefully
3. If tables missing → log warning, specific operations will fail gracefully
4. Normal operation with graceful degradation

**CLI startup:**
1. Attempt to connect to database
2. If database missing → error with message "Run daemon first: wqm service start"
3. Normal operation

## Rationale

### Why Daemon Owns Database?

1. **Consistency with ADR-002**: Daemon already owns Qdrant writes; SQLite ownership follows same pattern
2. **Single source of truth**: One component responsible for schema = no conflicts
3. **Rust performance**: Schema operations in Rust are faster and more reliable
4. **Simplified MCP server**: TypeScript server has no migration logic
5. **Clear responsibility**: "Daemon owns all persistent state" is easy to understand

### Why Not MCP Server?

1. MCP server would need to coordinate with Rust tables
2. MCP server may not run before daemon (daemon is background service)
3. Rust daemon is the long-running component; MCP server is request-driven

### Why Not Shared Ownership?

1. Requires coordination protocol between TypeScript and Rust
2. Risk of race conditions during migrations
3. More complex testing and debugging
4. Violates "single responsibility" principle

## Implementation

### Rust Daemon Schema Management

```rust
// daemon/core/src/schema.rs

pub const SCHEMA_VERSION: i32 = 1;

pub async fn initialize_database(pool: &SqlitePool) -> Result<()> {
    // Create schema_version table
    // Check current version
    // Run migrations if needed
    // Create all tables
}

pub async fn create_tables(pool: &SqlitePool) -> Result<()> {
    // Create all tables with proper schema
    // unified_queue, watch_folders, projects, etc.
}
```

### CLI Path Resolution

```rust
// All CLI commands must use:
fn get_database_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(format!("{}/.workspace-qdrant/state.db", home))
}
```

### MCP Server (TypeScript)

The TypeScript MCP server:
- Connects to database at standard path
- Reads and writes to tables but never creates them
- Handles missing tables gracefully with appropriate error responses

## Consequences

### Positive

- Single source of truth for database schema
- Simplified MCP server code (no migrations)
- Consistent with daemon-only-writes philosophy
- Easier debugging (one place to check schema)
- Rust handles schema operations efficiently

### Negative

- Daemon must run before MCP server can fully operate
- MCP server cannot self-initialize database

### Risks

- Users must start daemon before using MCP server
- Error messages must clearly indicate "start daemon first"

## Related Documents

- [ADR-002](./ADR-002-daemon-only-write-policy.md) - Daemon-only writes to Qdrant
- [WORKSPACE_QDRANT_MCP.md](../../WORKSPACE_QDRANT_MCP.md) - Main specification
- [FIRST-PRINCIPLES.md](../../FIRST-PRINCIPLES.md) - Architectural principles

## Changelog

| Date | Change |
|------|--------|
| 2026-01-30 | Initial ADR created |
