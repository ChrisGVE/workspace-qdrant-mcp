//! wqm-storage-write — write-side storage engine for workspace-qdrant-mcp.
//!
//! Location: `src/rust/storage-write/` (crate `wqm-storage-write`). Logical
//! context: the WRITE half of the branch-storage two-crate split (arch §9). It
//! owns every mutation — SQLite DDL ([`schema`]) and [`migrations`], Qdrant
//! upserts/deletes/payload writes ([`qdrant::QdrantWriteClient`]), the per-branch
//! write-handle [`registry`], and (in later features) git2-backed operations.
//!
//! The boundary is one-directional: this crate depends on `wqm-storage` (read),
//! never the reverse. CI Guards 1-3 enforce it — Guard 1 keeps this crate out of
//! `mcp-server`'s feature closure, Guard 2 makes a read-crate call to [`schema`]/
//! [`migrations`] a compile error, Guard 3 asserts no mutating Qdrant symbol is
//! reachable in the `mcp-server`/`wqm-cli` release binaries.
//!
//! Neighbors: `wqm-storage` (read sibling), `wqm-common` (canonical types — F0),
//! `wqm-client`. Outward errors are `wqm_common::StorageError` (DR GP-9).

pub mod blob;
pub mod branch;
pub mod connection;
pub mod migrations;
pub mod qdrant;
pub mod registry;
pub mod schema;
pub mod single_writer;

pub use blob::{
    blob_refcount, delete_file_from_branch, delete_orphan_blob_row, ingest_file,
    ContentKeyLockManager, Embedder, IngestParams,
};
pub use branch::{branch_delete, delete_decision, probe_branch, DeleteAction, GitBranchProbe};
pub use connection::{open_store, open_store_write};
pub use qdrant::{MembershipPutBatch, PendingMembershipPut, QdrantWriteClient};
pub use single_writer::{DaemonLock, DaemonLockConfig};
