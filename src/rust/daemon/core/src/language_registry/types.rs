//! Core data types for the dynamic language registry.
//!
//! The canonical definitions now live in `wqm_common::language_registry::types`
//! (WI-b1). This module re-exports them so the existing
//! `crate::language_registry::types::*` paths keep resolving unchanged.

pub use wqm_common::language_registry::types::*;
