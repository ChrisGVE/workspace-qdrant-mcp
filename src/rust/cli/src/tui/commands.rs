//! TUI command executor for gRPC actions.
//!
//! Provides synchronous wrappers around async gRPC calls for use
//! within the TUI event loop. Commands are fire-and-forget with
//! result feedback via status messages.

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::RebuildIndexRequest;

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

/// Rebuild all indexes for a specific collection via gRPC.
pub fn rebuild_collection(collection: &str) -> CommandResult {
    let coll = collection.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let request = RebuildIndexRequest {
            target: "all".to_string(),
            collection: Some(coll.clone()),
            tenant_id: None,
            force: Some(false),
        };
        client.system().rebuild_index(request).await?;
        Ok::<_, anyhow::Error>(())
    }) {
        Ok(()) => CommandResult {
            success: true,
            message: format!("Rebuild enqueued for {collection}"),
        },
        Err(e) => CommandResult {
            success: false,
            message: format!("Rebuild failed: {}", e),
        },
    }
}

/// Delete a project by its project_id (tenant_id) via gRPC.
pub fn delete_project(project_id: &str, delete_qdrant_data: bool) -> CommandResult {
    let pid = project_id.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        use crate::grpc::proto::DeleteProjectRequest;
        let request = DeleteProjectRequest {
            project_id: pid,
            delete_qdrant_data,
        };
        client.project().delete_project(request).await?;
        Ok::<_, anyhow::Error>(())
    }) {
        Ok(()) => CommandResult {
            success: true,
            message: "Project deleted".to_string(),
        },
        Err(e) => CommandResult {
            success: false,
            message: format!("Delete failed: {}", e),
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
