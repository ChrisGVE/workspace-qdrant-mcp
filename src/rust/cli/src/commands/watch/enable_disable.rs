//! Watch enable and disable subcommands

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::WatchIdRequest;
use crate::output;

pub async fn enable(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .enable_watch(WatchIdRequest {
            watch_id: watch_id.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Watch '{}' enabled", watch_id));
        output::info("Daemon will pick up this watch on next poll cycle");
    } else {
        output::warning(format!("Watch '{}' not found or already enabled", watch_id));
    }

    Ok(())
}

pub async fn disable(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .disable_watch(WatchIdRequest {
            watch_id: watch_id.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Watch '{}' disabled", watch_id));
        output::info("Daemon will stop watching this folder on next poll cycle");
    } else {
        output::warning(format!(
            "Watch '{}' not found or already disabled",
            watch_id
        ));
    }

    Ok(())
}
