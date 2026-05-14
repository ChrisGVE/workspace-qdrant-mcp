//! Library set-incremental subcommand

use std::path::PathBuf;

use anyhow::Result;

use super::helpers::canonical_from_cli_path;
use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::SetIncrementalRequest;
use crate::output;

/// Set or clear the incremental (do-not-delete) flag on tracked files.
pub async fn execute(files: &[PathBuf], clear: bool) -> Result<()> {
    let file_paths: Vec<String> = files
        .iter()
        .map(|f| match canonical_from_cli_path(f) {
            Ok(cp) => cp.into_string(),
            Err(_) => f.to_string_lossy().to_string(),
        })
        .collect();

    let mut client = ensure_daemon_available().await?;

    let response = client
        .library_write()
        .set_incremental(SetIncrementalRequest { file_paths, clear })
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
