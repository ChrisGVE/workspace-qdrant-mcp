//! Add scratchpad entry handler

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, QueueType, RefreshSignalRequest};
use crate::output;

use super::client::resolve_tenant_id;

pub(super) async fn add_entry(
    content: String,
    title: Option<String>,
    tags: Option<String>,
    project: Option<String>,
) -> Result<()> {
    let tenant_id = resolve_tenant_id(project.as_deref())?;

    let tag_vec: Vec<String> = tags
        .map(|t| {
            t.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    let payload_json = serde_json::json!({
        "content": content,
        "title": title,
        "tags": tag_vec,
        "source_type": "scratchpad",
    })
    .to_string();

    let mut client = ensure_daemon_available().await?;

    let response = client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "text".to_string(),
            op: "add".to_string(),
            tenant_id: tenant_id.to_string(),
            collection: wqm_common::constants::COLLECTION_SCRATCHPAD.to_string(),
            payload_json,
            branch: "main".to_string(),
            metadata_json: None,
        })
        .await?
        .into_inner();

    // Signal daemon to process queue
    let request = RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let _ = client.system().send_refresh_signal(request).await;

    output::section("Scratchpad Entry Queued");
    output::kv("Queue ID", &response.queue_id);
    output::kv("Tenant", &tenant_id);
    if let Some(t) = &title {
        output::kv("Title", t);
    }
    if !tag_vec.is_empty() {
        output::kv("Tags", tag_vec.join(", "));
    }
    let preview = if content.len() > 80 {
        format!("{}...", &content[..77])
    } else {
        content
    };
    output::kv("Content", &preview);
    if !response.is_new {
        output::warning("Duplicate entry (already queued)");
    }

    Ok(())
}
