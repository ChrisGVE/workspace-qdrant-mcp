//! Read-crate schema surface: execution-free name constants only.
//!
//! File: `wqm-storage/src/schema/mod.rs`
//! Location: `src/rust/storage/src/schema/` (read-crate schema submodule)
//! Context: workspace-qdrant-mcp branch-storage model (arch §5.2, §9 Crate 1).
//!   The read crate carries NO DDL and NO execution functions. Only the table and
//!   column name constants in [`columns`] live here, so read-path query builders
//!   use the same literal names as the write-path DDL without a second definition
//!   (FP-2). Guard 2 (trybuild) ensures no call to `wqm_storage_write::schema::*`
//!   execution functions can compile in this crate.
//!
//! Neighbors: `wqm-storage-write::schema` (canonical DDL that defines these names).

pub mod columns;
