//! DDL for the `xrefs` table (arch §5.2).
//!
//! File: `wqm-storage-write/src/schema/branch.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: Per-project `store.db` DDL. Named `branch.rs` per PRD AC-F3.5 / arch §9
//!   file-layout spec. Contains the `xrefs` table: per-branch cross-reference rows
//!   (call-site resolution). Symbol DEFINITIONS live in `blobs.symbol_name`; call-site
//!   RESOLUTION is per-branch because which `bar()` a call resolves to differs per branch.
//!
//! Neighbors: [`super::blobs`] (symbol definitions), [`super::files`] (branches/files
//!   tables referenced here via FKs), [`super::mod`] (aggregate apply order).

/// DDL for the `xrefs` table — per-branch cross-reference rows.
pub const CREATE_XREFS: &str = r#"CREATE TABLE xrefs (
    xref_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    concrete_id        INTEGER NOT NULL REFERENCES concrete(concrete_id) ON DELETE CASCADE,
    blob_id            INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
    symbol_name        TEXT NOT NULL,
    xref_type          TEXT NOT NULL,
    target_symbol      TEXT,
    target_branch_id   TEXT REFERENCES branches(branch_id),
    target_concrete_id INTEGER REFERENCES concrete(concrete_id) ON DELETE SET NULL,
    created_at         TEXT NOT NULL
)"#;

/// Index on `xrefs(concrete_id)` for caller-side lookup.
pub const IDX_XREFS_CONCRETE: &str = "CREATE INDEX idx_xrefs_concrete ON xrefs(concrete_id)";

/// Index on `xrefs(symbol_name, xref_type)` for symbol-definition lookups.
pub const IDX_XREFS_SYMBOL: &str = "CREATE INDEX idx_xrefs_symbol ON xrefs(symbol_name, xref_type)";

/// Index on `xrefs(target_branch_id, target_symbol)` for resolution queries.
pub const IDX_XREFS_TARGET: &str =
    "CREATE INDEX idx_xrefs_target ON xrefs(target_branch_id, target_symbol)";

/// All DDL statements for this module, in application order.
pub const STATEMENTS: &[&str] = &[
    CREATE_XREFS,
    IDX_XREFS_CONCRETE,
    IDX_XREFS_SYMBOL,
    IDX_XREFS_TARGET,
];
