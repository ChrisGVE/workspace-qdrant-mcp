//! DDL for `fts_content` (FTS5 virtual table), `fts_branch_membership`, and the
//! FTS5 sync triggers `blobs_ai` / `blobs_ad` (arch §5.2, AC-F3.2).
//!
//! File: `wqm-storage-write/src/schema/fts.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: Per-project `store.db` DDL — the full-text search layer. `fts_content`
//!   is an FTS5 external-content virtual table over `blobs.raw_text`, keyed by
//!   `blob_id` via `content_rowid`. `fts_branch_membership` is an indexed scalar
//!   junction enabling branch-scoped FTS5 queries without `json_each`.
//!
//!   The two triggers keep `fts_content` in sync with `blobs`:
//!     - `blobs_ai` (AFTER INSERT): supplies `(rowid, raw_text)` with `rowid = new.blob_id`.
//!     - `blobs_ad` (AFTER DELETE): issues the FTS5 `'delete'` command form.
//!   Without these triggers a GC'd blob leaves a ghost FTS rowid; a new blob is
//!   invisible to FTS (arch §5.2 sync contract, AC-F3.2).
//!
//! Neighbors: [`super::blobs`] (triggers fire on `blobs`), [`super::mod`] (apply order
//!   requires `blobs` and `branches` exist before this module's statements execute).

/// DDL for the FTS5 external-content virtual table over `blobs.raw_text`.
///
/// `content_rowid="blob_id"` maps FTS rowid to `blobs.blob_id`, enabling the
/// join `fts_content JOIN fts_branch_membership USING (blob_id)` without a
/// redundant `blob_id` column in the FTS table.
pub const CREATE_FTS_CONTENT: &str = r#"CREATE VIRTUAL TABLE fts_content USING fts5 (
    raw_text,
    content="blobs",
    content_rowid="blob_id"
)"#;

/// AFTER INSERT trigger on `blobs` — inserts into `fts_content` with `rowid = new.blob_id`.
/// AC-F3.2: the rowid MUST be supplied explicitly; a missing rowid yields a NULL join key
/// and zero FTS results for all queries.
pub const TRIGGER_BLOBS_AI: &str = r#"CREATE TRIGGER blobs_ai AFTER INSERT ON blobs BEGIN
    INSERT INTO fts_content(rowid, raw_text) VALUES (new.blob_id, new.raw_text);
END"#;

/// AFTER DELETE trigger on `blobs` — uses the FTS5 `'delete'` command form to
/// remove the row from the external-content index. AC-F3.2: without this trigger
/// a GC'd blob leaves a ghost FTS rowid.
pub const TRIGGER_BLOBS_AD: &str = r#"CREATE TRIGGER blobs_ad AFTER DELETE ON blobs BEGIN
    INSERT INTO fts_content(fts_content, rowid, raw_text) VALUES ('delete', old.blob_id, old.raw_text);
END"#;

/// DDL for the branch-membership junction table for branch-scoped FTS5 queries.
/// PRIMARY KEY (blob_id, branch_id) enforces uniqueness; the supplementary index
/// on (branch_id, blob_id) supports the branch-filter probe direction.
pub const CREATE_FTS_BRANCH_MEMBERSHIP: &str = r#"CREATE TABLE fts_branch_membership (
    blob_id    INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE CASCADE,
    branch_id  TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
    PRIMARY KEY (blob_id, branch_id)
)"#;

/// Index on `fts_branch_membership(branch_id, blob_id)` for branch-scoped FTS lookups.
pub const IDX_FTS_BRANCH: &str =
    "CREATE INDEX idx_fts_branch ON fts_branch_membership(branch_id, blob_id)";

/// All DDL statements for this module, in application order.
/// Triggers must follow the virtual table and the `blobs` table (created in `blobs.rs`).
pub const STATEMENTS: &[&str] = &[
    CREATE_FTS_CONTENT,
    TRIGGER_BLOBS_AI,
    TRIGGER_BLOBS_AD,
    CREATE_FTS_BRANCH_MEMBERSHIP,
    IDX_FTS_BRANCH,
];
