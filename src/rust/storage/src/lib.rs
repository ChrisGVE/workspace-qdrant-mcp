//! wqm-storage — read-side storage facade for workspace-qdrant-mcp.
//!
//! Location: `src/rust/storage/` (crate `wqm-storage`). Logical context: the
//! READ half of the branch-storage two-crate split (arch §9). Every read path
//! over the per-branch SQLite databases and the Qdrant collections lands here;
//! all mutation (DDL, migrations, Qdrant upserts/deletes, git2) lives in the
//! sibling `wqm-storage-write` crate. The split is a hard architectural boundary
//! policed by three CI guards:
//!   * Guard 1 — `wqm-storage-write` is unreachable from `mcp-server`'s feature
//!     closure (`cargo tree`).
//!   * Guard 2 — a call to a `schema::*`/`migrations::*` execution function from
//!     this crate fails to compile (trybuild, `tests/guard2_read_cannot_write.rs`).
//!   * Guard 3 — no Qdrant-mutating symbol is reachable in the `mcp-server` /
//!     `wqm-cli` release binaries (`nm` scan, `scripts/ci/storage-guards.sh`).
//!
//! Neighbors: `wqm-common` (canonical `StorageError`/`SearchResult`/`FileChange`/
//! RRF — F0), `wqm-client`, and `wqm-storage-write` (the write sibling). Outward
//! errors are `wqm_common::StorageError` (DR GP-9 — one error type, never a
//! parallel definition).

pub mod qdrant;
pub mod types;

pub use qdrant::QdrantReadClient;
