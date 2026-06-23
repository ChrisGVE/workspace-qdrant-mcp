//! Re-export shim: the `xrefs` table DDL lives in [`super::branch`].
//!
//! File: `wqm-storage-write/src/schema/xrefs.rs`
//! Location: `src/rust/storage-write/src/schema/` (write-crate schema submodule)
//! Context: PRD AC-F3.5 / arch §9 name the six schema files as
//!   `{files, blobs, branch, xrefs, fts, store_meta}.rs`. The `xrefs` table DDL is
//!   consolidated in `branch.rs` (same apply-order group as the branch-scoped tables).
//!   This shim re-exports it so the six-file surface matches the spec exactly.
//!
//! Neighbors: [`super::branch`] (canonical DDL home for xrefs).

pub use super::branch::{
    CREATE_XREFS, IDX_XREFS_CONCRETE, IDX_XREFS_SYMBOL, IDX_XREFS_TARGET, STATEMENTS,
};
