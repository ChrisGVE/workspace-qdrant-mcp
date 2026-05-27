//! Library set-incremental subcommand

use std::path::PathBuf;

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::SetIncrementalRequest;
use crate::output;

/// Set or clear the incremental (do-not-delete) flag on tracked files.
///
/// File paths are sent as relative content paths (e.g. `src/main.rs`).
/// The gRPC handler validates them via `extract_relative_paths!` and the
/// write-actor matches directly against `tracked_files.relative_path`.
pub async fn execute(files: &[PathBuf], clear: bool, tag: &str) -> Result<()> {
    let file_paths: Vec<String> = files
        .iter()
        .map(|f| f.to_string_lossy().to_string())
        .collect();

    let watch_folder_id = format!("lib-{}", tag);
    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .set_incremental(SetIncrementalRequest {
            file_paths,
            clear,
            watch_folder_id,
        })
        .await?
        .into_inner();

    let action = if clear { "cleared" } else { "set" };

    if response.updated > 0 || response.not_found > 0 {
        output::info(format!(
            "{} incremental flag on {} file(s), {} not found (total: {})",
            action,
            response.updated,
            response.not_found,
            files.len()
        ));
    }

    Ok(())
}
