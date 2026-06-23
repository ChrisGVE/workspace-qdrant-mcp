//! Per-branch SQLite schema (DDL) — the write crate's exclusive home for table
//! definitions.
//!
//! Location: `wqm-storage-write/src/schema.rs`. Logical context: the canonical,
//! write-only home of the per-branch store's `CREATE TABLE`/index DDL. The read
//! crate (`wqm-storage`) cannot reach these functions — Guard 2 makes any such
//! call a compile error. Concrete DDL is added by later features (F3+); this
//! module is the home those features extend.
//!
//! Neighbors: [`crate::migrations`] (applies these statements in order).

/// The ordered DDL statements that materialize a fresh per-branch store.
///
/// Empty until the schema-defining features land; [`crate::migrations`] applies
/// whatever this returns. Keeping the home explicit now fixes the single source
/// of truth for the schema and lets Guard 2 anchor the read/write boundary.
pub fn ddl_statements() -> &'static [&'static str] {
    &[]
}

/// Build the full schema by joining [`ddl_statements`]. Used by callers that
/// want one script rather than the statement slice.
pub fn create_schema() -> String {
    ddl_statements().join(";\n")
}
