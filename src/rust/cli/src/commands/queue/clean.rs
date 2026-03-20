//! Queue clean subcommand

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::CleanQueueRequest;
use crate::output;

pub async fn execute(days: i64, status_filter: Option<String>, yes: bool) -> Result<()> {
    // Validate status filter
    let statuses = match status_filter.as_deref() {
        Some("done") => vec!["done".to_string()],
        Some("failed") => vec!["failed".to_string()],
        Some(other) => anyhow::bail!("Invalid status '{}'. Use 'done' or 'failed'.", other),
        None => vec!["done".to_string(), "failed".to_string()],
    };

    if !yes {
        output::warning(format!(
            "Will remove {} items older than {} days. Use -y to skip this confirmation.",
            statuses.join("/"),
            days
        ));
        return Ok(());
    }

    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .clean_queue(CleanQueueRequest {
            older_than_days: days as i32,
            statuses,
        })
        .await?
        .into_inner();

    if response.deleted_count == 0 {
        output::success(format!("No old items to clean (older than {} days)", days));
    } else {
        output::success(format!(
            "Removed {} old queue items",
            response.deleted_count
        ));
    }

    Ok(())
}
