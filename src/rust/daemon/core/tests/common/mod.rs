//! Shared test helpers for integration tests
//!
//! This module provides reusable utilities shared across multiple integration
//! test files. Integration tests in Rust's `tests/` directory are each their
//! own crate, so this `common/mod.rs` pattern is the canonical way to share
//! helpers between them.

pub mod graph_helpers;
pub mod stress;
