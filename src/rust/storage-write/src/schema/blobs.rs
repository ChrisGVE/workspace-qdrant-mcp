//! DDL for the `blobs`, `blob_refs`, and `concrete` tables (arch §5.2).
//!
//! File: `wqm-storage-write/src/schema/blobs.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: Per-project `store.db` DDL. Contains the content-addressed blob store
//!   (`blobs`), the referrer junction table (`blob_refs`), and the per-branch
//!   ingest-status table (`concrete`). The `blobs` table is the dedup unit: one row
//!   per unique chunk of content. `blob_refs` is the GC referrer ledger; `concrete`
//!   carries per-branch enrichment metadata.
//!
//! KEY DESIGN NOTES (AC-F3.1):
//!   - `chunk_index` lives in `blob_refs`, NOT in `blobs` (positional membership).
//!   - `idx_blob_refs_blob` is DELIBERATELY ABSENT: `idx_blob_refs_covering` covers
//!     all blob_id lookups as a prefix; a separate single-column index would add 33%
//!     WAL amplification on the highest-write-rate table.
//!   - `blobs.content_key` encodes the FOUR-SLOT form:
//!       content_key(tenant_id, "code", chunk_content_hash, "")
//!     NOT the stale three-slot form (PRD AC-F3.1 / MF-1b).
//!   - `blobs.dense_vec` and `blobs.sparse_vec` are BLOB columns (binary, not TEXT).
//!
//! Neighbors: [`super::files`] (files/branches), [`super::fts`] (FTS5 sync triggers
//!   on this table), [`super::store_meta`] (tenant guard trigger on this table).

/// DDL for the `blobs` table — one row per unique deduped chunk.
///
/// The `content_key` column stores the four-slot key:
///   `content_key(tenant_id, "code", chunk_content_hash, "")`
/// This is NOT the stale three-slot form; see PRD AC-F3.1 / MF-1b.
pub const CREATE_BLOBS: &str = r#"CREATE TABLE blobs (
    blob_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    content_key          TEXT NOT NULL UNIQUE,
    chunk_content_hash   TEXT NOT NULL,
    point_id             TEXT NOT NULL UNIQUE,
    tenant_id            TEXT NOT NULL,
    raw_text             TEXT NOT NULL,
    dense_vec            BLOB NOT NULL,
    sparse_vec           BLOB NOT NULL,
    chunk_type           TEXT,
    symbol_name          TEXT,
    start_line           INTEGER,
    end_line             INTEGER,
    created_at           TEXT NOT NULL
)"#;

/// Supplementary index on `blobs(chunk_content_hash)`.
pub const IDX_BLOBS_CHUNK_CONTENT_HASH: &str =
    "CREATE INDEX idx_blobs_chunk_content_hash ON blobs(chunk_content_hash)";

/// Supplementary index on `blobs(point_id)`.
pub const IDX_BLOBS_POINT_ID: &str = "CREATE INDEX idx_blobs_point_id ON blobs(point_id)";

/// Supplementary index on `blobs(tenant_id)` — recovery cursor filter.
pub const IDX_BLOBS_TENANT: &str = "CREATE INDEX idx_blobs_tenant ON blobs(tenant_id)";

/// DDL for the `blob_refs` junction table — the GC referrer ledger.
///
/// `idx_blob_refs_blob` is DELIBERATELY ABSENT (see module doc).
pub const CREATE_BLOB_REFS: &str = r#"CREATE TABLE blob_refs (
    ref_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id    TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
    file_id      INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,
    blob_id      INTEGER NOT NULL REFERENCES blobs(blob_id) ON DELETE RESTRICT,
    UNIQUE (branch_id, file_id, chunk_index)
)"#;

/// Index on `blob_refs(branch_id)`.
pub const IDX_BLOB_REFS_BRANCH: &str = "CREATE INDEX idx_blob_refs_branch ON blob_refs(branch_id)";

/// Index on `blob_refs(file_id)`.
pub const IDX_BLOB_REFS_FILE: &str = "CREATE INDEX idx_blob_refs_file ON blob_refs(file_id)";

/// Covering index for search-enrichment JOINs and the GC GROUP BY scan.
/// Covers single-column `blob_id` lookups via index prefix — making a separate
/// `idx_blob_refs_blob` fully redundant (arch §5.2 deliberate absence note).
pub const IDX_BLOB_REFS_COVERING: &str =
    "CREATE INDEX idx_blob_refs_covering ON blob_refs(blob_id, branch_id, file_id)";

/// DDL for the `concrete` table — per-branch ingest-status + enrichment metadata.
pub const CREATE_CONCRETE: &str = r#"CREATE TABLE concrete (
    concrete_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    branch_id         TEXT NOT NULL REFERENCES branches(branch_id) ON DELETE CASCADE,
    file_id           INTEGER NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    file_mtime        TEXT NOT NULL,
    file_hash         TEXT NOT NULL,
    lsp_status        TEXT NOT NULL DEFAULT 'none'
                          CHECK (lsp_status IN ('none','done','failed','skipped')),
    treesitter_status TEXT NOT NULL DEFAULT 'none'
                          CHECK (treesitter_status IN ('none','done','failed','skipped')),
    component         TEXT,
    routing_reason    TEXT,
    last_error        TEXT,
    needs_reconcile   INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    UNIQUE (branch_id, file_id)
)"#;

/// Index on `concrete(branch_id)`.
pub const IDX_CONCRETE_BRANCH: &str = "CREATE INDEX idx_concrete_branch ON concrete(branch_id)";

/// Index on `concrete(file_id)`.
pub const IDX_CONCRETE_FILE: &str = "CREATE INDEX idx_concrete_file ON concrete(file_id)";

/// Partial index for the reconcile sweep — only rows that actually need work.
pub const IDX_CONCRETE_RECONCILE: &str =
    "CREATE INDEX idx_concrete_reconcile ON concrete(branch_id, needs_reconcile) WHERE needs_reconcile = 1";

/// All DDL statements for this module, in application order.
pub const STATEMENTS: &[&str] = &[
    CREATE_BLOBS,
    IDX_BLOBS_CHUNK_CONTENT_HASH,
    IDX_BLOBS_POINT_ID,
    IDX_BLOBS_TENANT,
    CREATE_BLOB_REFS,
    IDX_BLOB_REFS_BRANCH,
    IDX_BLOB_REFS_FILE,
    IDX_BLOB_REFS_COVERING,
    CREATE_CONCRETE,
    IDX_CONCRETE_BRANCH,
    IDX_CONCRETE_FILE,
    IDX_CONCRETE_RECONCILE,
];
