//! Per-branch SQLite migration runner — write-crate only.
//!
//! Location: `wqm-storage-write/src/migrations.rs`. Logical context: the
//! exclusive home of forward-only schema migrations for a per-branch store. The
//! read crate cannot call into here (Guard 2). Concrete migrations are added by
//! later features (F3+); this module is the home and the apply order they extend.
//!
//! Neighbors: [`crate::schema`] (the DDL these migrations apply).

/// One forward migration: a stable version number and the SQL that advances the
/// store to it. Later features push entries onto [`MIGRATIONS`].
#[derive(Debug, Clone, Copy)]
pub struct Migration {
    /// Monotonic schema version this migration produces.
    pub version: u32,
    /// SQL applied (idempotently) to reach `version`.
    pub sql: &'static str,
}

/// The ordered migration set. Empty until the schema-defining features land.
pub const MIGRATIONS: &[Migration] = &[];

/// Return the migrations not yet applied given the store's `current_version`.
pub fn pending(current_version: u32) -> Vec<Migration> {
    MIGRATIONS
        .iter()
        .copied()
        .filter(|m| m.version > current_version)
        .collect()
}

/// Run all forward migrations. With no migrations defined yet this is a no-op
/// that reports zero applied; later features apply [`pending`] against a pool.
pub fn run_migrations() -> usize {
    MIGRATIONS.len()
}
