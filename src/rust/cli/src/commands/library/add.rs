//! Library add subcommand

use std::path::PathBuf;

use anyhow::{Context, Result};

use super::helpers::LibraryMode;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::AddLibraryRequest;
use crate::output;
use crate::output::style::home_to_tilde;

/// Add a library (unwatched - metadata only)
pub async fn execute(tag: &str, path: &PathBuf, mode: LibraryMode) -> Result<()> {
    output::section(format!("Add Library: {}", tag));

    // Validate path exists (client-side check for immediate feedback)
    if !path.exists() {
        output::error(format!("Path does not exist: {}", path.display()));
        return Ok(());
    }

    let abs_path = path
        .canonicalize()
        .context("Could not resolve absolute path")?;
    let abs_path_str = abs_path.to_string_lossy().to_string();

    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .add_library(AddLibraryRequest {
            tag: tag.to_string(),
            path: abs_path_str.clone(),
            mode: mode.to_string(),
        })
        .await?
        .into_inner();

    if !response.success {
        output::error(response.message);
        return Ok(());
    }

    output::success(format!("Library '{}' added (not watching yet)", tag));
    output::kv("  Tag", tag);
    output::kv("  Path", home_to_tilde(&abs_path_str));
    output::kv("  Mode", mode.to_string());
    output::separator();
    output::info("To start watching: wqm library watch <tag> <path>");

    Ok(())
}
