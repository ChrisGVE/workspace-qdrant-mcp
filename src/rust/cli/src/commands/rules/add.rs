//! Add rule subcommand
//!
//! Adds a new behavioral rule via daemon gRPC. Tries IngestText first,
//! falls back to EnqueueItem if IngestText fails.

use anyhow::Result;
use wqm_common::constants::TENANT_GLOBAL;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::{EnqueueItemRequest, IngestTextRequest, QueueType, RefreshSignalRequest};
use crate::output;

/// Derive tenant_id from scope string.
///
/// Rules use [`TENANT_GLOBAL`] as tenant for global scope, or the project
/// ID extracted from `"project:<id>"` for project-scoped rules.
fn tenant_id_from_scope(scope: &str) -> String {
    if let Some(project_id) = scope.strip_prefix("project:") {
        project_id.to_string()
    } else {
        TENANT_GLOBAL.to_string()
    }
}

/// Add a rule via daemon gRPC.
pub async fn add_rule(
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &Option<String>,
) -> Result<()> {
    output::section("Add Rule");

    let scope_str = scope.as_deref().unwrap_or(TENANT_GLOBAL);
    let tenant_id = tenant_id_from_scope(scope_str);

    output::kv("Label", label);
    output::kv("Content", content);
    output::kv("Type", rule_type);
    output::kv("Scope", scope_str);
    output::separator();

    let mut client = ensure_daemon_available().await?;

    // Try direct IngestText first
    let mut metadata = std::collections::HashMap::new();
    metadata.insert("label".to_string(), label.to_string());
    metadata.insert("rule_type".to_string(), rule_type.to_string());
    metadata.insert("scope".to_string(), scope_str.to_string());
    metadata.insert("priority".to_string(), "5".to_string());
    metadata.insert("enabled".to_string(), "true".to_string());

    let request = IngestTextRequest {
        content: content.to_string(),
        collection_basename: "rules".to_string(),
        tenant_id: tenant_id.clone(),
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
                enqueue_rule_via_grpc(
                    &mut client,
                    label,
                    content,
                    rule_type,
                    scope_str,
                    &tenant_id,
                )
                .await?;
            }
        }
        Err(e) => {
            output::error(format!("Failed to add rule via daemon: {}", e));
            // Fall back to enqueue
            enqueue_rule_via_grpc(
                &mut client,
                label,
                content,
                rule_type,
                scope_str,
                &tenant_id,
            )
            .await?;
        }
    }

    Ok(())
}

/// Enqueue a rule via gRPC QueueWriteService.
///
/// Payload uses structured fields so the queue processor can map them
/// directly to Qdrant payload fields (label, scope, priority, etc.).
async fn enqueue_rule_via_grpc(
    client: &mut crate::grpc::DaemonClient,
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &str,
    tenant_id: &str,
) -> Result<()> {
    output::info("Enqueueing rule for later processing...");

    let payload_json = serde_json::json!({
        "content": content,
        "source_type": "rule",
        "label": label,
        "scope": scope,
        "rule_type": rule_type,
        "priority": 5,
    })
    .to_string();

    match client
        .queue_write()
        .enqueue_item(EnqueueItemRequest {
            item_type: "text".to_string(),
            op: "add".to_string(),
            tenant_id: tenant_id.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id_from_global_scope() {
        assert_eq!(tenant_id_from_scope("global"), "global");
    }

    #[test]
    fn test_tenant_id_from_project_scope() {
        assert_eq!(tenant_id_from_scope("project:4ed81466dec7"), "4ed81466dec7");
    }

    #[test]
    fn test_tenant_id_from_empty_scope() {
        assert_eq!(tenant_id_from_scope(""), "global");
    }
}
