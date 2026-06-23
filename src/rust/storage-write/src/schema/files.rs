//! DDL for the `files` and `branches` tables (arch §5.2).
//!
//! File: `wqm-storage-write/src/schema/files.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: Per-project `store.db` DDL — write-crate exclusive home. Contains the
//!   `files` table (one row per (branch_id, relative_path) pair) and the `branches`
//!   table (per-project mirror of state.db project_locations). Both tables cascade-
//!   delete via the `branch_id` foreign-key chain.
//!
//! Neighbors: [`super::blobs`] (blob content), [`super::branch`] referenced here
//!   via FK, [`super::mod`] (aggregate apply order).

/// DDL for the `branches` table — must be created BEFORE `files` because
/// `files(branch_id)` carries a REFERENCES branches(branch_id) constraint.
pub const CREATE_BRANCHES: &str = r#"CREATE TABLE branches (
    branch_id     TEXT PRIMARY KEY,
    branch_name   TEXT NOT NULL,
    location      TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    sync_state    TEXT NOT NULL DEFAULT 'pending'
                      CHECK (sync_state IN ('pending','indexing','current','error')),
    sync_metadata TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
)"#;

/// DDL for the `files` table — one row per (branch_id, path) pair.
pub const CREATE_FILES: &str = r#"CREATE TABLE files (
    file_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id        TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
    relative_path    TEXT NOT NULL,
    file_type        TEXT,
    language         TEXT,
    extension        TEXT,
    is_test          INTEGER NOT NULL DEFAULT 0,
    collection       TEXT NOT NULL DEFAULT 'projects',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE (branch_id, relative_path)
)"#;

/// Index on `files(branch_id)` for branch-wide scans.
pub const IDX_FILES_BRANCH: &str = "CREATE INDEX idx_files_branch ON files(branch_id)";

/// Covering index on `files(branch_id, relative_path)` for path lookups.
pub const IDX_FILES_PATH: &str = "CREATE INDEX idx_files_path ON files(branch_id, relative_path)";

/// All DDL statements for this module, in application order.
pub const STATEMENTS: &[&str] = &[
    CREATE_BRANCHES,
    CREATE_FILES,
    IDX_FILES_BRANCH,
    IDX_FILES_PATH,
];
