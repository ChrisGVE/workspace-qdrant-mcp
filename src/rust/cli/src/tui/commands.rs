//! TUI command executor for gRPC actions.
//!
//! Provides synchronous wrappers around async gRPC calls for use
//! within the TUI event loop. Commands are fire-and-forget with
//! result feedback via status messages.

use crate::grpc::ensure_daemon_available;

/// Result of a TUI command execution.
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub success: bool,
    pub message: String,
}

/// Pause all active watch folders via gRPC.
pub fn pause_watchers() -> CommandResult {
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client.watch_write().pause_watchers(()).await?.into_inner();
        Ok::<_, anyhow::Error>(response.affected_count)
    }) {
        Ok(count) if count > 0 => CommandResult {
            success: true,
            message: format!("Paused {} watch folder(s)", count),
        },
        Ok(_) => CommandResult {
            success: true,
            message: "No active watchers to pause".to_string(),
        },
        Err(e) => CommandResult {
            success: false,
            message: format!("Pause failed: {}", e),
        },
    }
}

/// Resume all paused watch folders via gRPC.
pub fn resume_watchers() -> CommandResult {
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client.watch_write().resume_watchers(()).await?.into_inner();
        Ok::<_, anyhow::Error>(response.affected_count)
    }) {
        Ok(count) if count > 0 => CommandResult {
            success: true,
            message: format!("Resumed {} watch folder(s)", count),
        },
        Ok(_) => CommandResult {
            success: true,
            message: "No paused watchers to resume".to_string(),
        },
        Err(e) => CommandResult {
            success: false,
            message: format!("Resume failed: {}", e),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_result_creation() {
        let r = CommandResult {
            success: true,
            message: "ok".to_string(),
        };
        assert!(r.success);
        assert_eq!(r.message, "ok");
    }
}
