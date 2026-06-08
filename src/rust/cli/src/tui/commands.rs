//! TUI command executor for gRPC actions.
//!
//! Provides synchronous wrappers around async gRPC calls for use
//! within the TUI event loop. Commands are fire-and-forget with
//! result feedback via status messages.

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::WatchIdRequest;

/// Pause all active watch folders via gRPC.
pub fn pause_watchers() -> String {
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client.watch_write().pause_watchers(()).await?.into_inner();
        Ok::<_, anyhow::Error>(response.affected_count)
    }) {
        Ok(count) if count > 0 => format!("Paused {} watch folder(s)", count),
        Ok(_) => "No active watchers to pause".to_string(),
        Err(e) => format!("Pause failed: {}", e),
    }
}

/// Resume all paused watch folders via gRPC.
pub fn resume_watchers() -> String {
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client.watch_write().resume_watchers(()).await?.into_inner();
        Ok::<_, anyhow::Error>(response.affected_count)
    }) {
        Ok(count) if count > 0 => format!("Resumed {} watch folder(s)", count),
        Ok(_) => "No paused watchers to resume".to_string(),
        Err(e) => format!("Resume failed: {}", e),
    }
}

/// Enable or disable tracking for a single watch folder by ID via gRPC.
///
/// Daemon-only write: the CLI never touches `watch_folders.enabled` directly;
/// it routes through `EnableWatch`/`DisableWatch` so the daemon owns the state.
pub fn set_watch_enabled(watch_id: &str, enable: bool) -> String {
    let watch_id = watch_id.to_string();
    let result = tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let req = WatchIdRequest {
            watch_id: watch_id.clone(),
        };
        let response = if enable {
            client.watch_write().enable_watch(req).await?
        } else {
            client.watch_write().disable_watch(req).await?
        };
        Ok::<_, anyhow::Error>(response.into_inner().affected_count)
    });
    let verb = if enable { "enabled" } else { "disabled" };
    match result {
        Ok(count) if count > 0 => format!("Tracking {} for {}", verb, watch_id),
        Ok(_) => format!("No change: already {}", verb),
        Err(e) => format!("Toggle failed: {}", e),
    }
}
