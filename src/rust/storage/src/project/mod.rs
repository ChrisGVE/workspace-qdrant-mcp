//! Project resolution — `ProjectRegistry` and related types.
//!
//! File: `wqm-storage/src/project/mod.rs`
//! Location: `src/rust/storage/src/project/` (read crate)
//! Context: arch §6.2 / §8 nexus. The single CWD->tenant resolver (FP-2).
//!   All read-path components that need a `ProjectBinding` route through here;
//!   no component path-walks on its own.
//!
//! Neighbors: `resolver.rs` (implementation), `crate::types::binding`.

pub mod resolver;

pub use resolver::{BranchEntry, ProjectRegistry, SearchScope};
