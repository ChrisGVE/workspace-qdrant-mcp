//! Clap value parser for path arguments.
//!
//! Applies tilde and environment-variable expansion to every user-supplied
//! path before clap hands it to a command handler, enabling fully scriptable
//! invocations such as:
//!
//!   wqm library watch mytag ~/docs
//!   wqm library watch mytag "$MY_DOCS_DIR"
//!   wqm ingest file "${PROJECT_ROOT}/README.md"
//!
//! Command substitution `$(...)` is a shell feature and must be expanded by
//! the calling shell (use double quotes: `wqm ingest file "$(pwd)/file.md"`).

use std::path::PathBuf;

/// Clap value_parser: expand tilde + env vars then return a PathBuf.
///
/// Suitable for any `#[arg(value_parser = parse_path)]` annotation.
pub fn parse_path(s: &str) -> Result<PathBuf, String> {
    Ok(wqm_common::env_expand::expand_path(s))
}
