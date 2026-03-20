//! Add rule subcommand
//!
//! Adds a new behavioral rule via daemon gRPC. Tries IngestText first,
//! falls back to EnqueueItem if IngestText fails.

use anyhow::Result;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, IngestTextRequest, QueueType, RefreshSignalRequest};
use crate::output;

/// Add a rule via daemon gRPC.
pub async fn add_rule(
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &Option<String>,
    priority: u32,
) -> Result<()> {
    output::section("Add Rule");

    let scope_str = scope.as_deref().unwrap_or("global");

    output::kv("Label", label);
    output::kv("Content", content);
    output::kv("Type", rule_type);
    output::kv("Scope", scope_str);
    output::kv("Priority", priority.to_string());
    output::separator();

    let mut client = ensure_daemon_available().await?;

    // Try direct IngestText first
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("label".to_string(), label.to_string());
    metadata.insert("rule_type".to_string(), rule_type.to_string());
    metadata.insert("scope".to_string(), scope_str.to_string());
    metadata.insert("priority".to_string(), priority.to_string());
    metadata.insert("enabled".to_string(), "true".to_string());

    let request = IngestTextRequest {
        content: content.to_string(),
        collection_basename: "rules".to_string(),
        tenant_id: String::new(),
        document_id: Some(label.to_string()),
        metadata,
        chunk_text: false,
    };

    match client.document().ingest_text(request).await {
        Ok(response) => {
            let result = response.into_inner();
            if result.success {
                output::success("Rule added");
                output::kv("Label", label);
                output::kv("Rule ID", &result.document_id);
            } else {
                output::error(format!("Failed to add rule: {}", result.error_message));
                // Fall back to enqueue
                enqueue_rule_via_grpc(&mut client, label, content, rule_type, scope_str, priority)
                    .await?;
            }
        }
        Err(e) => {
            output::error(format!("Failed to add rule via daemon: {}", e));
            // Fall back to enqueue
            enqueue_rule_via_grpc(&mut client, label, content, rule_type, scope_str, priority)
                .await?;
        }
    }

    Ok(())
}

/// Enqueue a rule via gRPC QueueWriteService.
async fn enqueue_rule_via_grpc(
    client: &mut crate::grpc::DaemonClient,
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &str,
    priority: u32,
) -> Result<()> {
    output::info("Enqueueing rule for later processing...");

    let full_content = format!(
        "RULE\nlabel:{}\ntype:{}\nscope:{}\npriority:{}\n---\n{}",
        label, rule_type, scope, priority, content
    );

    let payload_json = serde_json::json!({
        "content": full_content,
        "source_type": "cli_rules",
        "main_tag": format!("rules_{}", rule_type),
        "full_tag": format!("rules_{}_{}", rule_type, scope),
    })
    .to_string();

    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "text".to_string(),
            op: "add".to_string(),
            tenant_id: "_global".to_string(),
            collection: "rules".to_string(),
            payload_json,
            branch: "main".to_string(),
            metadata_json: None,
        })
        .await
    {
        Ok(resp) => {
            let inner = resp.into_inner();
            if inner.is_new {
                output::success("Rule queued for processing");
                output::kv("Label", label);
                output::kv("Queue ID", &inner.queue_id);
                output::kv("Status", "pending");
            } else {
                output::warning("Rule already queued (duplicate)");
                output::kv("Idempotency Key", &inner.idempotency_key);
            }
            output::separator();
            output::info("The rule will be added when the daemon processes the queue.");
            output::info("Check status with: wqm status queue");
        }
        Err(e) => {
            output::error(format!("Failed to enqueue rule: {}", e));
        }
    }

    // Signal daemon to process
    let request = RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let _ = client.system().send_refresh_signal(request).await;

    Ok(())
}
