//! Queue remove subcommand

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::RemoveItemRequest;
use crate::output;

use super::formatters::format_status;

pub async fn execute(queue_id: &str) -> Result<()> {
    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .remove_item(RemoveItemRequest {
            queue_id: queue_id.to_string(),
        })
        .await?
        .into_inner();

    if !response.found {
        output::error(format!("Queue item not found: {}", queue_id));
        return Ok(());
    }

    let short_id = &response.resolved_id[..8.min(response.resolved_id.len())];

    if response.status == "in_progress" {
        output::warning(format!(
            "Item {} is currently in_progress -- removing may cause the processor to error",
            short_id
        ));
    }

    output::success(format!(
        "Removed queue item {} ({} {} in {}), was {}",
        short_id,
        response.item_type,
        response.op,
        response.collection,
        format_status(&response.status),
    ));

    Ok(())
}
