//! Queue retry subcommand

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::output;

pub async fn execute(queue_id: Option<String>, all: bool, all_transient: bool) -> Result<()> {
    if !all && !all_transient && queue_id.is_none() {
        anyhow::bail!("Specify a queue_id, --all, or --all-transient to retry failed items");
    }

    let mut client = ensure_daemon_available().await?;

    if all {
        retry_all(&mut client).await
    } else if all_transient {
        // TODO: add RetryAllTransient gRPC method to QueueWriteService
        anyhow::bail!("--all-transient is not yet supported via the daemon write path")
    } else {
        retry_one(&mut client, &queue_id.unwrap()).await
    }
}

async fn retry_all(client: &mut crate::grpc::DaemonClient) -> Result<()> {
    let response = client.queue_write().retry_all(()).await?.into_inner();

    if response.reset_count == 0 {
        output::success("No failed items to retry");
    } else {
        output::success(format!(
            "Reset {} failed items to pending",
            response.reset_count
        ));
    }

    Ok(())
}

async fn retry_one(client: &mut crate::grpc::DaemonClient, id: &str) -> Result<()> {
    use crate::grpc::proto::RetryItemRequest;

    let response = client
        .queue_write()
        .retry_item(RetryItemRequest {
            queue_id: id.to_string(),
        })
        .await?
        .into_inner();

    if !response.found {
        output::error(format!("Queue item not found: {}", id));
        return Ok(());
    }

    if !response.reset {
        output::warning(format!(
            "Item {} has status '{}', not 'failed'. \
             Use --status filter with list to find failed items.",
            response.resolved_id, response.previous_status
        ));
        return Ok(());
    }

    output::success(format!(
        "Reset item {} to pending (was retry {})",
        response.resolved_id, response.previous_retry_count
    ));

    Ok(())
}
