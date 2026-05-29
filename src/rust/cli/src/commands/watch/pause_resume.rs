//! Watch pause and resume subcommands

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::WatchIdRequest;
use crate::output;

pub async fn pause() -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client.watch_write().pause_watchers(()).await?.into_inner();

    if response.affected_count > 0 {
        output::success(format!(
            "Paused {} watch folder(s)",
            response.affected_count
        ));
        output::info("File events will be buffered until watchers are resumed");
    } else {
        output::info(
            "No active watchers to pause \
             (all already paused or none enabled)",
        );
    }

    Ok(())
}

pub async fn resume() -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client.watch_write().resume_watchers(()).await?.into_inner();

    if response.affected_count > 0 {
        output::success(format!(
            "Resumed {} watch folder(s)",
            response.affected_count
        ));
        output::info("Buffered file events will be processed");
    } else {
        output::info("No paused watchers to resume");
    }

    Ok(())
}

pub async fn pause_one(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .pause_watch(WatchIdRequest {
            watch_id: watch_id.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Paused watch folder: {watch_id}"));
    } else {
        output::info(format!(
            "No change for {watch_id} (already paused, disabled, or not found)"
        ));
    }

    Ok(())
}

pub async fn resume_one(watch_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .watch_write()
        .resume_watch(WatchIdRequest {
            watch_id: watch_id.to_string(),
        })
        .await?
        .into_inner();

    if response.affected_count > 0 {
        output::success(format!("Resumed watch folder: {watch_id}"));
    } else {
        output::info(format!(
            "No change for {watch_id} (not currently paused, disabled, or not found)"
        ));
    }

    Ok(())
}
