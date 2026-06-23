//! Execution-free table and column name constants for the per-project `store.db`.
//!
//! File: `wqm-storage/src/schema/columns.rs`
//! Location: `src/rust/storage/src/schema/` (read-crate schema submodule)
//! Context: workspace-qdrant-mcp branch-storage model (arch §5.2, §9 Crate 1).
//!   The READ crate carries ONLY these name constants — NO DDL, NO execution functions.
//!   All DDL and connection-open logic lives exclusively in `wqm-storage-write`
//!   (Guard 2 enforces this at compile time).
//!
//!   Constants are grouped by table. All values match arch §5.2 verbatim so that
//!   query builders in the read-path use the same names as the write-path DDL without
//!   a second source of truth (FP-2 / DR GP-9).
//!
//! Neighbors: `wqm-storage-write::schema` (canonical DDL that defines these names),
//!   [`crate::qdrant`] (read client that queries against these tables).

// ---------------------------------------------------------------------------
// Table names
// ---------------------------------------------------------------------------

pub const TABLE_FILES: &str = "files";
pub const TABLE_BRANCHES: &str = "branches";
pub const TABLE_BLOBS: &str = "blobs";
pub const TABLE_BLOB_REFS: &str = "blob_refs";
pub const TABLE_CONCRETE: &str = "concrete";
pub const TABLE_XREFS: &str = "xrefs";
pub const TABLE_FTS_CONTENT: &str = "fts_content";
pub const TABLE_FTS_BRANCH_MEMBERSHIP: &str = "fts_branch_membership";
pub const TABLE_STORE_META: &str = "store_meta";

// ---------------------------------------------------------------------------
// `branches` columns
// ---------------------------------------------------------------------------

pub const BRANCHES_BRANCH_ID: &str = "branch_id";
pub const BRANCHES_BRANCH_NAME: &str = "branch_name";
pub const BRANCHES_LOCATION: &str = "location";
pub const BRANCHES_ACTIVE: &str = "active";
pub const BRANCHES_SYNC_STATE: &str = "sync_state";
pub const BRANCHES_SYNC_METADATA: &str = "sync_metadata";
pub const BRANCHES_CREATED_AT: &str = "created_at";
pub const BRANCHES_UPDATED_AT: &str = "updated_at";

// ---------------------------------------------------------------------------
// `files` columns
// ---------------------------------------------------------------------------

pub const FILES_FILE_ID: &str = "file_id";
pub const FILES_BRANCH_ID: &str = "branch_id";
pub const FILES_RELATIVE_PATH: &str = "relative_path";
pub const FILES_FILE_TYPE: &str = "file_type";
pub const FILES_LANGUAGE: &str = "language";
pub const FILES_EXTENSION: &str = "extension";
pub const FILES_IS_TEST: &str = "is_test";
pub const FILES_COLLECTION: &str = "collection";
pub const FILES_CREATED_AT: &str = "created_at";
pub const FILES_UPDATED_AT: &str = "updated_at";

// ---------------------------------------------------------------------------
// `blobs` columns
// ---------------------------------------------------------------------------

pub const BLOBS_BLOB_ID: &str = "blob_id";
/// Four-slot content_key: content_key(tenant_id, "code", chunk_content_hash, "")
pub const BLOBS_CONTENT_KEY: &str = "content_key";
pub const BLOBS_CHUNK_CONTENT_HASH: &str = "chunk_content_hash";
pub const BLOBS_POINT_ID: &str = "point_id";
pub const BLOBS_TENANT_ID: &str = "tenant_id";
pub const BLOBS_RAW_TEXT: &str = "raw_text";
/// BLOB column: f32[] little-endian, length=768
pub const BLOBS_DENSE_VEC: &str = "dense_vec";
/// BLOB column: (u32 index, f32 value)[] pairs, serialized
pub const BLOBS_SPARSE_VEC: &str = "sparse_vec";
pub const BLOBS_CHUNK_TYPE: &str = "chunk_type";
pub const BLOBS_SYMBOL_NAME: &str = "symbol_name";
pub const BLOBS_START_LINE: &str = "start_line";
pub const BLOBS_END_LINE: &str = "end_line";
pub const BLOBS_CREATED_AT: &str = "created_at";

// ---------------------------------------------------------------------------
// `blob_refs` columns
// ---------------------------------------------------------------------------

pub const BLOB_REFS_REF_ID: &str = "ref_id";
pub const BLOB_REFS_BRANCH_ID: &str = "branch_id";
pub const BLOB_REFS_FILE_ID: &str = "file_id";
/// chunk_index is in blob_refs (positional membership), NOT in blobs (arch §5.2).
pub const BLOB_REFS_CHUNK_INDEX: &str = "chunk_index";
pub const BLOB_REFS_BLOB_ID: &str = "blob_id";

// ---------------------------------------------------------------------------
// `concrete` columns
// ---------------------------------------------------------------------------

pub const CONCRETE_CONCRETE_ID: &str = "concrete_id";
pub const CONCRETE_BRANCH_ID: &str = "branch_id";
pub const CONCRETE_FILE_ID: &str = "file_id";
pub const CONCRETE_FILE_MTIME: &str = "file_mtime";
pub const CONCRETE_FILE_HASH: &str = "file_hash";
pub const CONCRETE_LSP_STATUS: &str = "lsp_status";
pub const CONCRETE_TREESITTER_STATUS: &str = "treesitter_status";
pub const CONCRETE_COMPONENT: &str = "component";
pub const CONCRETE_ROUTING_REASON: &str = "routing_reason";
pub const CONCRETE_LAST_ERROR: &str = "last_error";
pub const CONCRETE_NEEDS_RECONCILE: &str = "needs_reconcile";
pub const CONCRETE_CREATED_AT: &str = "created_at";
pub const CONCRETE_UPDATED_AT: &str = "updated_at";

// ---------------------------------------------------------------------------
// `xrefs` columns
// ---------------------------------------------------------------------------

pub const XREFS_XREF_ID: &str = "xref_id";
pub const XREFS_CONCRETE_ID: &str = "concrete_id";
pub const XREFS_BLOB_ID: &str = "blob_id";
pub const XREFS_SYMBOL_NAME: &str = "symbol_name";
pub const XREFS_XREF_TYPE: &str = "xref_type";
pub const XREFS_TARGET_SYMBOL: &str = "target_symbol";
pub const XREFS_TARGET_BRANCH_ID: &str = "target_branch_id";
pub const XREFS_TARGET_CONCRETE_ID: &str = "target_concrete_id";
pub const XREFS_CREATED_AT: &str = "created_at";

// ---------------------------------------------------------------------------
// `fts_branch_membership` columns
// ---------------------------------------------------------------------------

pub const FTS_BRANCH_MEMBERSHIP_BLOB_ID: &str = "blob_id";
pub const FTS_BRANCH_MEMBERSHIP_BRANCH_ID: &str = "branch_id";

// ---------------------------------------------------------------------------
// `store_meta` columns
// ---------------------------------------------------------------------------

pub const STORE_META_TENANT_ID: &str = "tenant_id";
