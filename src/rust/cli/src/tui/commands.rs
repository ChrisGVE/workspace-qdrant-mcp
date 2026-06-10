//! TUI command executor for gRPC actions.
//!
//! Provides synchronous wrappers around async gRPC calls for use
//! within the TUI event loop. Commands are fire-and-forget with
//! result feedback via status messages.
//!
//! All functions use `tokio::runtime::Handle::current().block_on` to drive
//! the async gRPC call from the synchronous TUI event loop. Each function
//! returns a human-readable `String` describing the outcome, which the caller
//! stores in its `message` field for display in the summary bar.

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{
    CancelItemsRequest, DeleteDocumentRequest, EnqueueItemRequest, RefreshSignalRequest,
    RemoveItemRequest, RetryItemRequest, WatchIdRequest,
};
use wqm_common::constants::{COLLECTION_LIBRARIES, COLLECTION_PROJECTS, COLLECTION_RULES};

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

// ─── Queue actions ───────────────────────────────────────────────────────────

/// Retry a single failed queue item by its full queue_id.
///
/// Mirrors `commands/queue/retry.rs` → `retry_item(RetryItemRequest)`.
pub fn queue_retry(queue_id: &str) -> String {
    let queue_id = queue_id.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client
            .queue_write()
            .retry_item(RetryItemRequest {
                queue_id: queue_id.clone(),
            })
            .await?
            .into_inner();
        Ok::<_, anyhow::Error>(response)
    }) {
        Ok(r) if !r.found => format!("Item not found: {}", queue_id),
        Ok(r) if !r.reset => format!(
            "Item {} has status '{}', not 'failed'",
            r.resolved_id, r.previous_status
        ),
        Ok(r) => format!(
            "Reset {} to pending (was retry {})",
            r.resolved_id, r.previous_retry_count
        ),
        Err(e) => format!("Retry failed: {}", e),
    }
}

/// Cancel queue items for the tenant that owns the given queue_id.
///
/// Mirrors `commands/queue/cancel.rs` → `cancel_items(CancelItemsRequest)`.
/// Cancels only `pending` items for the queue item's tenant.
pub fn queue_cancel(tenant_id: &str) -> String {
    let tenant_id = tenant_id.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client
            .queue_write()
            .cancel_items(CancelItemsRequest {
                tenant_id: tenant_id.clone(),
                statuses: vec!["pending".to_string()],
                dry_run: false,
            })
            .await?
            .into_inner();
        Ok::<_, anyhow::Error>(response)
    }) {
        Ok(r) if r.count == 0 => format!("No pending items for {}", r.project_path),
        Ok(r) => format!(
            "Cancelled {} pending item(s) for {}",
            r.count, r.project_path
        ),
        Err(e) => format!("Cancel failed: {}", e),
    }
}

/// Remove (drop) a single queue item by its full queue_id.
///
/// Mirrors `commands/queue/drop.rs` → `remove_item(RemoveItemRequest)`.
pub fn queue_remove(queue_id: &str) -> String {
    let queue_id = queue_id.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let response = client
            .queue_write()
            .remove_item(RemoveItemRequest {
                queue_id: queue_id.clone(),
            })
            .await?
            .into_inner();
        Ok::<_, anyhow::Error>(response.found)
    }) {
        Ok(true) => format!("Removed item {}", queue_id),
        Ok(false) => format!("Item not found: {}", queue_id),
        Err(e) => format!("Remove failed: {}", e),
    }
}

// ─── Rule action ─────────────────────────────────────────────────────────────

/// Delete a rule by its rule_id (which is used as document_id in the collection).
///
/// Mirrors `commands/rules/remove.rs` → `delete_document(DeleteDocumentRequest)`.
pub fn rule_delete(rule_id: &str) -> String {
    let rule_id = rule_id.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        client
            .document()
            .delete_document(DeleteDocumentRequest {
                document_id: rule_id.clone(),
                collection_name: COLLECTION_RULES.to_string(),
            })
            .await?;
        Ok::<_, anyhow::Error>(())
    }) {
        Ok(()) => format!("Rule {} deleted", rule_id),
        Err(e) => format!("Delete failed: {}", e),
    }
}

// ─── Project / Library nudge ─────────────────────────────────────────────────

/// Nudge a project to re-scan: enqueue a folder scan and signal the daemon.
///
/// Mirrors `commands/library/rescan.rs` → `enqueue_item` + `send_refresh_signal`.
/// The `tenant_id` is the project's tenant_id; `collection` is "projects".
pub fn project_nudge(tenant_id: &str) -> String {
    nudge_watch_folder(tenant_id, COLLECTION_PROJECTS)
}

/// Nudge a library to re-scan: enqueue a folder scan and signal the daemon.
///
/// Mirrors `commands/library/rescan.rs` → `enqueue_item` + `send_refresh_signal`.
/// The `tenant_id` is the library tag; `collection` is "libraries".
pub fn library_nudge(tenant_id: &str) -> String {
    nudge_watch_folder(tenant_id, COLLECTION_LIBRARIES)
}

/// Shared implementation: enqueue a folder scan for a watch folder and signal
/// the daemon to start processing.
fn nudge_watch_folder(tenant_id: &str, collection: &str) -> String {
    use crate::grpc::proto::QueueType;
    let tenant_id = tenant_id.to_string();
    let collection = collection.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        let payload_json = serde_json::json!({ "recursive": true }).to_string();
        let enqueue_resp = client
            .queue_write()
            .enqueue_item(EnqueueItemRequest {
                item_type: "folder".to_string(),
                op: "scan".to_string(),
                tenant_id: tenant_id.clone(),
                collection: collection.clone(),
                payload_json,
                branch: String::new(),
                metadata_json: None,
            })
            .await?
            .into_inner();
        // Signal daemon regardless of whether the item was newly queued.
        let _ = client
            .system()
            .send_refresh_signal(RefreshSignalRequest {
                queue_type: QueueType::IngestQueue as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            })
            .await;
        Ok::<_, anyhow::Error>(enqueue_resp.is_new)
    }) {
        Ok(true) => "Rescan queued — daemon notified".to_string(),
        Ok(false) => "Rescan already queued".to_string(),
        Err(e) => format!("Nudge failed: {}", e),
    }
}

// ─── Library book removal ────────────────────────────────────────────────────

/// Remove a single library document (book) by its absolute path.
///
/// Available only for incremental libraries (sync libraries are managed by
/// the file watcher; their books cannot be removed here).
///
/// Mirrors `commands/rules/remove.rs` → `delete_document(DeleteDocumentRequest)`.
/// The document_id is the book's absolute file path; the collection is "libraries".
pub fn library_book_remove(abs_path: &str) -> String {
    let abs_path = abs_path.to_string();
    match tokio::runtime::Handle::current().block_on(async {
        let mut client = ensure_daemon_available().await?;
        client
            .document()
            .delete_document(DeleteDocumentRequest {
                document_id: abs_path.clone(),
                collection_name: COLLECTION_LIBRARIES.to_string(),
            })
            .await?;
        Ok::<_, anyhow::Error>(())
    }) {
        Ok(()) => format!("Removed book {}", abs_path),
        Err(e) => format!("Remove failed: {}", e),
    }
}
