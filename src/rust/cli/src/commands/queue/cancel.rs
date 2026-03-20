//! Bulk-cancel pending queue items for a project.

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::CancelItemsRequest;
use crate::output;

/// Execute `wqm queue cancel`.
pub async fn execute(project: &str, statuses: &[&str], dry_run: bool, yes: bool) -> Result<()> {
    if !yes && !dry_run {
        output::warning(
            "Will cancel items for the specified project. Use -y to confirm or --dry-run to preview.",
        );
        return Ok(());
    }

    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .cancel_items(CancelItemsRequest {
            tenant_id: project.to_string(),
            statuses: statuses.iter().map(|s| s.to_string()).collect(),
            dry_run,
        })
        .await?
        .into_inner();

    let tid_short = &response.tenant_id[..response.tenant_id.len().min(12)];

    if response.count == 0 {
        output::success(format!(
            "No items to cancel for project {} ({})",
            response.project_path, tid_short,
        ));
        return Ok(());
    }

    if response.is_dry_run {
        output::info(format!(
            "[dry-run] Would cancel {} items for project {} ({})",
            response.count, response.project_path, tid_short,
        ));
    } else {
        output::success(format!(
            "Cancelled {} items for project {} ({})",
            response.count, response.project_path, tid_short,
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    // Tests for cancel logic now live in the daemon's QueueWriteService handler.
    // CLI tests would require a running daemon (integration tests).
}
