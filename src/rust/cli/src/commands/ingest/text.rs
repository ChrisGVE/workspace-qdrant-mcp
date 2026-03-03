//! Text ingest subcommand handler

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::IngestTextRequest;
use crate::output;
use crate::queue::{ContentPayload as QueueContentPayload, UnifiedQueueClient};

use super::detect::{detect_branch, detect_tenant_id};

pub async fn ingest_text(content: &str, collection: &str, title: Option<String>) -> Result<()> {
    output::section("Ingest Text");

    let preview = if content.len() > 50 {
        format!("{}...", &content[..50])
    } else {
        content.to_string()
    };

    output::kv("Content", &preview);
    output::kv("Collection", collection);
    if let Some(t) = &title {
        output::kv("Title", t);
    }
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            output::info("Ingesting text via daemon...");

            let request = IngestTextRequest {
                content: content.to_string(),
                collection_basename: collection.to_string(),
                tenant_id: String::new(), // Auto-detected by daemon
                document_id: title.clone(),
                metadata: std::collections::HashMap::new(),
                chunk_text: true,
            };

            match client.document().ingest_text(request).await {
                Ok(response) => {
                    let result = response.into_inner();
                    if result.success {
                        output::success("Text ingested successfully");
                        output::kv("Document ID", &result.document_id);
                        output::kv("Chunks Created", &result.chunks_created.to_string());
                    } else {
                        output::error(format!("Ingestion failed: {}", result.error_message));
                    }
                }
                Err(e) => {
                    output::error(format!("Failed to ingest text: {}", e));
                    // Try unified queue fallback
                    try_queue_fallback_text(content, collection, &title).await?;
                }
            }
        }
        Err(_) => {
            // Daemon not running - use unified queue fallback (Task 37.10)
            output::warning("Daemon not running, using unified queue fallback");
            try_queue_fallback_text(content, collection, &title).await?;
        }
    }

    Ok(())
}

/// Fallback to unified queue when daemon is unavailable (Task 37.10)
async fn try_queue_fallback_text(
    content: &str,
    collection: &str,
    _title: &Option<String>,
) -> Result<()> {
    output::info("Enqueueing to unified_queue for later processing...");

    match UnifiedQueueClient::connect() {
        Ok(queue_client) => {
            // Create content payload
            let payload = QueueContentPayload {
                content: content.to_string(),
                source_type: "cli".to_string(),
                main_tag: None,
                full_tag: None,
            };

            // Get tenant_id from current directory (auto-detect project)
            let tenant_id = detect_tenant_id();
            let branch = detect_branch();

            match queue_client.enqueue_content(
                &tenant_id,
                collection,
                &payload,
                &branch,
            ) {
                Ok(result) => {
                    if result.was_duplicate {
                        output::warning("Content already queued (duplicate)");
                        output::kv("Idempotency Key", &result.idempotency_key);
                    } else {
                        output::success("Content queued for processing");
                        output::kv("Queue ID", &result.queue_id);
                        output::kv("Status", "pending");
                        output::kv("Fallback Mode", "unified_queue");
                    }
                    output::separator();
                    output::info("The content will be processed when the daemon starts.");
                    output::info("Check status with: wqm status queue");
                }
                Err(e) => {
                    output::error(format!("Failed to enqueue content: {}", e));
                }
            }
        }
        Err(e) => {
            output::error(format!("Failed to connect to queue database: {}", e));
            output::info("Ensure the workspace-qdrant directory exists.");
        }
    }

    Ok(())
}
