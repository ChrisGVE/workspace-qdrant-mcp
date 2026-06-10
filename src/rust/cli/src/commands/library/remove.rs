//! Library remove subcommand

use anyhow::Result;
use wqm_common::constants::COLLECTION_LIBRARIES;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{
    EnqueueItemRequest, QueueType, RefreshSignalRequest, RemoveLibraryRequest,
};
use crate::output;

async fn queue_vector_deletion(client: &mut crate::grpc::client::DaemonClient, tag: &str) {
    output::info("Queueing vector deletion...");
    let payload_json = serde_json::json!({ "tenant_id_to_delete": tag }).to_string();

    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "tenant".to_string(),
            op: "delete".to_string(),
            tenant_id: tag.to_string(),
            collection: COLLECTION_LIBRARIES.to_string(),
            payload_json,
            branch: String::new(),
            metadata_json: None,
        })
        .await
    {
        Ok(resp) => {
            let inner = resp.into_inner();
            if inner.is_new {
                output::success(format!(
                    "Vector deletion queued (queue_id: {})",
                    inner.queue_id
                ));
            } else {
                output::info("Vector deletion already queued (duplicate)");
            }
        }
        Err(e) => {
            output::warning(format!("Could not queue vector deletion: {}", e));
        }
    }
}

/// Remove a library (deletes watch config AND queues vector deletion)
pub async fn execute(tag: &str, skip_confirm: bool) -> Result<()> {
    output::section(format!("Remove Library: {}", tag));

    // Confirm deletion unless --yes flag (typed confirmation, #123)
    if !skip_confirm {
        output::warning(format!(
            "This will delete ALL vectors for library '{}' from Qdrant.",
            tag
        ));
        output::warning("This action cannot be undone.");
        if !output::typed_confirm(tag) {
            output::info("Cancelled.");
            return Ok(());
        }
    }

    let mut client = ensure_daemon_available().await?;

    // Remove watch config atomically via daemon
    output::separator();
    output::info("Removing watch configuration...");

    let response = client
        .library_write()
        .remove_library(RemoveLibraryRequest {
            tag: tag.to_string(),
        })
        .await?
        .into_inner();

    if response.queue_items_cancelled > 0 {
        output::info(format!(
            "Cancelled {} pending queue items",
            response.queue_items_cancelled
        ));
    }
    output::success(format!("Removed watch config for '{}'", tag));

    queue_vector_deletion(&mut client, tag).await;

    // Signal daemon to refresh watch configuration
    if let Ok(resp) = client
        .system()
        .send_refresh_signal(RefreshSignalRequest {
            queue_type: QueueType::WatchedFolders as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        })
        .await
    {
        let _ = resp;
        output::success("Daemon notified - will process deletion shortly");
    }

    output::separator();
    output::success(format!("Library '{}' removed", tag));

    Ok(())
}
