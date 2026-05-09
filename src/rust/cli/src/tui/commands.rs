//! TUI command executor for gRPC actions.
//!
//! Provides synchronous wrappers around async gRPC calls for use
//! within the TUI event loop. Commands are fire-and-forget with
//! result feedback via status messages.

use crate::grpc::ensure_daemon_available;

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
