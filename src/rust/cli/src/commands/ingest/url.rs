//! URL ingest subcommand handler

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, QueueType, RefreshSignalRequest};
use crate::output;

use super::detect::{detect_branch, detect_tenant_id};

async fn enqueue_url_and_notify(
    client: &mut crate::grpc::client::DaemonClient,
    tenant_id: &str,
    collection: &str,
    payload_json: &str,
    branch: &str,
) {
    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "url".to_string(),
            op: "add".to_string(),
            tenant_id: tenant_id.to_string(),
            collection: collection.to_string(),
            payload_json: payload_json.to_string(),
            branch: branch.to_string(),
            metadata_json: None,
        })
        .await
    {
        Ok(resp) => {
            let inner = resp.into_inner();
            if inner.is_new {
                output::success("URL queued for ingestion");
                output::kv("Queue ID", &inner.queue_id);
                output::kv("Status", "pending");
            } else {
                output::warning("URL already queued (duplicate)");
                output::kv("Idempotency Key", &inner.idempotency_key);
            }
            let request = RefreshSignalRequest {
                queue_type: QueueType::IngestQueue as i32,
                lsp_languages: vec![],
                grammar_languages: vec![],
            };
            let _ = client.system().send_refresh_signal(request).await;
        }
        Err(e) => {
            output::error(format!("Failed to enqueue URL: {}", e));
        }
    }
}

pub async fn ingest_url(
    url: &str,
    collection: Option<String>,
    library: Option<String>,
    title: Option<String>,
) -> Result<()> {
    output::section("Ingest URL");

    // Validate URL format
    if !url.starts_with("http://") && !url.starts_with("https://") {
        output::error("URL must start with http:// or https://");
        return Ok(());
    }

    // Determine collection and tenant_id based on library flag
    let (target_collection, tenant_id) = if let Some(ref lib_name) = library {
        (
            wqm_common::constants::COLLECTION_LIBRARIES.to_string(),
            lib_name.clone(),
        )
    } else if let Some(ref coll) = collection {
        (coll.clone(), detect_tenant_id())
    } else {
        (
            wqm_common::constants::COLLECTION_PROJECTS.to_string(),
            detect_tenant_id(),
        )
    };

    output::kv("URL", url);
    output::kv("Collection", &target_collection);
    output::kv("Tenant", &tenant_id);
    if let Some(ref t) = title {
        output::kv("Title", t);
    }
    if let Some(ref lib) = library {
        output::kv("Library", lib);
    }
    output::separator();

    let branch = detect_branch();

    // Build URL payload
    let payload_json = serde_json::json!({
        "url": url,
        "crawl": false,
        "max_depth": 0,
        "max_pages": 1,
        "content_type": null,
        "library_name": library,
        "title": title,
    })
    .to_string();

    let mut client = ensure_daemon_available().await?;

    enqueue_url_and_notify(
        &mut client,
        &tenant_id,
        &target_collection,
        &payload_json,
        &branch,
    )
    .await;

    Ok(())
}
