//! Storage facade modules for the read crate.
//!
//! File: `wqm-storage/src/facade/mod.rs`
//! Location: `src/rust/storage/src/facade/` (read crate)
//! Context: arch §6.2. The `read` submodule holds `ReadStoreFacade` and its
//!   supporting search/list implementations. No write facade lives here.
//!
//! Neighbors: `read/mod.rs` (ReadStoreFacade), `crate::project` (registry).

pub mod read;

pub use read::ReadStoreFacade;
