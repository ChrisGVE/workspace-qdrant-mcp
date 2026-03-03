//! URL ingest subcommand handler

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::{QueueType, RefreshSignalRequest};
use crate::output;
use crate::queue::{UnifiedQueueClient, UrlPayload as QueueUrlPayload};

use super::detect::{detect_branch, detect_tenant_id};

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
        (wqm_common::constants::COLLECTION_LIBRARIES.to_string(), lib_name.clone())
    } else if let Some(ref coll) = collection {
        (coll.clone(), detect_tenant_id())
    } else {
        (wqm_common::constants::COLLECTION_PROJECTS.to_string(), detect_tenant_id())
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
    let url_payload = QueueUrlPayload {
        url: url.to_string(),
        crawl: false,
        max_depth: 0,
        max_pages: 1,
        content_type: None,
        library_name: library,
        title,
    };

    // Try unified queue (works even without daemon running)
    match UnifiedQueueClient::connect() {
        Ok(queue_client) => {
            match queue_client.enqueue_url(
                &tenant_id,
                &target_collection,
                &url_payload,
                &branch,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::warning("URL already queued (duplicate)");
                        output::kv("Idempotency Key", &result.idempotency_key);
                    } else {
                        output::success("URL queued for ingestion");
                        output::kv("Queue ID", &result.queue_id);
                        output::kv("Status", "pending");
                    }

                    // Signal daemon to process queue if running
                    if let Ok(mut client) = DaemonClient::connect_default().await {
                        let request = RefreshSignalRequest {
                            queue_type: QueueType::IngestQueue as i32,
                            lsp_languages: vec![],
                            grammar_languages: vec![],
                        };
                        let _ = client.system().send_refresh_signal(request).await;
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to enqueue URL: {}", e));
                }
            }
        }
        Err(e) => {
            output::error(format!("Failed to connect to queue database: {}", e));
            output::info("Ensure the daemon has run at least once to create the database.");
        }
    }

    Ok(())
}
