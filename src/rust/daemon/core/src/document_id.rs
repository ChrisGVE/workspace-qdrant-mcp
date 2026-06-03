//! Stable document and point ID generation.
//!
//! The canonical implementation now lives in `wqm_common::document_id` (WI-b1);
//! this module re-exports it so the existing `crate::document_id::*` and
//! `crate::generate_*` paths keep resolving unchanged.

pub use wqm_common::document_id::*;
