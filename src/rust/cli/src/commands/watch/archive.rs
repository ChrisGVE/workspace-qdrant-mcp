//! Watch archive and unarchive subcommands

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{ArchiveWatchRequest, WatchIdRequest};
use crate::output;

pub async fn archive(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .archive_watch(ArchiveWatchRequest {
            watch_id: watch_id.to_string(),
            cascade_submodules: true,
        })
        .await?
        .into_inner();

    if response.affected_count == 0 {
        output::warning(format!(
            "Watch '{}' not found or already archived",
            watch_id
        ));
        return Ok(());
    }

    output::success(format!("Archived watch '{}'", watch_id));

    if response.submodules_archived > 0 || response.submodules_skipped > 0 {
        output::info(format!(
            "{} submodule(s) archived, {} shared submodule(s) kept active",
            response.submodules_archived, response.submodules_skipped
        ));
    }

    output::info("Watching and ingesting stopped; data remains fully searchable");

    Ok(())
}

pub async fn unarchive(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .unarchive_watch(WatchIdRequest {
            watch_id: watch_id.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Unarchived watch '{}'", watch_id));
        output::info("Watching and ingesting will resume on next poll cycle");
    } else {
        output::warning(format!("Watch '{}' not found or not archived", watch_id));
    }

    Ok(())
}
