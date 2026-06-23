//! Per-project `store.db` schema — write-crate exclusive home for all DDL.
//!
//! File: `wqm-storage-write/src/schema/mod.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: workspace-qdrant-mcp branch-storage model (arch §5.2, §9 Crate 2). Owns
//!   the 9-table DDL for the per-project `store.db`, the two FTS5 sync triggers, the
//!   cross-tenant BEFORE INSERT trigger, and all indexes exactly as arch §5.2 specifies.
//!   The read crate (`wqm-storage`) cannot call into any execution function here —
//!   Guard 2 makes any such call a compile error (trybuild, AC-F1.5).
//!
//!   Apply order (dependencies must precede dependents):
//!     1. `files`       -> `branches`, `files` tables + indexes
//!     2. `blobs`       -> `blobs`, `blob_refs`, `concrete` tables + indexes
//!     3. `branch`      -> `xrefs` table + indexes  (branch-scoped cross-refs)
//!     4. `fts`         -> `fts_content` virtual table, `blobs_ai`/`blobs_ad` triggers,
//!                         `fts_branch_membership` table + index
//!     5. `store_meta`  -> `store_meta` table, `blobs_bi` cross-tenant trigger
//!
//! Neighbors: [`crate::migrations`] (applies these statements),
//!   [`crate::connection`] (connection factory that runs the open protocol).

pub mod blobs;
pub mod branch;
pub mod files;
pub mod fts;
pub mod store_meta;
pub mod xrefs;

/// All DDL statements that materialize a fresh per-project `store.db`, in the
/// order they must be applied (FK dependencies precede their referrers).
///
/// Each statement is a single SQL DDL string without a trailing semicolon —
/// suitable for execution via `sqlx::query(stmt).execute(pool).await`.
pub fn ddl_statements() -> &'static [&'static str] {
    // Apply order mirrors the module doc: files -> blobs -> branch/xrefs -> fts -> store_meta.
    // Using a runtime-built Vec keeps the source readable without a concat_slices macro.
    // The return is &'static because we leak once via Box::leak on first call.
    use std::sync::OnceLock;
    static ALL: OnceLock<Vec<&'static str>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut v: Vec<&'static str> = Vec::new();
        v.extend_from_slice(files::STATEMENTS);
        v.extend_from_slice(blobs::STATEMENTS);
        v.extend_from_slice(branch::STATEMENTS);
        v.extend_from_slice(fts::STATEMENTS);
        v.extend_from_slice(store_meta::STATEMENTS);
        v
    })
}

/// Build the full schema as one semicolon-joined SQL script. Used by callers
/// that want a single string (e.g. `sqlite3` shell, snapshot tests).
pub fn create_schema() -> String {
    ddl_statements().join(";\n")
}
