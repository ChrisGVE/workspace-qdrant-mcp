//! Add rule subcommand
//!
//! Adds a new behavioral rule via daemon gRPC or falls back to
//! the unified queue when the daemon is unavailable.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::IngestTextRequest;
use crate::output;
use crate::queue::{ContentPayload as QueueContentPayload, UnifiedQueueClient};

/// Add a rule via daemon gRPC, falling back to unified queue.
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

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            add_via_daemon(&mut client, label, content, rule_type, scope_str, priority).await
        }
        Err(_) => {
            // Daemon not running - use unified queue fallback (Task 37.12)
            output::warning("Daemon not running, using unified queue fallback");
            try_queue_fallback_rules(label, content, rule_type, scope_str, priority)
        }
    }
}

/// Send the rule to the daemon via gRPC IngestText.
async fn add_via_daemon(
    client: &mut DaemonClient,
    label: &str,
    content: &str,
    rule_type: &str,
    scope_str: &str,
    priority: u32,
) -> Result<()> {
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
        document_id: Some(label.to_string()), // Use label as document ID
        metadata,
        chunk_text: false, // Don't chunk rules
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
                // Try unified queue fallback
                try_queue_fallback_rules(label, content, rule_type, scope_str, priority)?;
            }
        }
        Err(e) => {
            output::error(format!("Failed to add rule via daemon: {}", e));
            // Try unified queue fallback
            try_queue_fallback_rules(label, content, rule_type, scope_str, priority)?;
        }
    }

    Ok(())
}

/// Fallback to unified queue when daemon is unavailable (Task 37.12)
fn try_queue_fallback_rules(
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &str,
    priority: u32,
) -> Result<()> {
    output::info("Enqueueing rule to unified_queue for later processing...");

    match UnifiedQueueClient::connect() {
        Ok(queue_client) => enqueue_rule(&queue_client, label, content, rule_type, scope, priority),
        Err(e) => {
            output::error(format!("Failed to connect to queue database: {}", e));
            output::info("Ensure the workspace-qdrant directory exists.");
            Ok(())
        }
    }
}

/// Enqueue a single rule into the unified queue.
fn enqueue_rule(
    queue_client: &UnifiedQueueClient,
    label: &str,
    content: &str,
    rule_type: &str,
    scope: &str,
    priority: u32,
) -> Result<()> {
    // Create content payload with rule metadata in the content
    let full_content = format!(
        "RULE\nlabel:{}\ntype:{}\nscope:{}\npriority:{}\n---\n{}",
        label, rule_type, scope, priority, content
    );

    let payload = QueueContentPayload {
        content: full_content,
        source_type: "cli_rules".to_string(),
        main_tag: Some(format!("rules_{}", rule_type)),
        full_tag: Some(format!("rules_{}_{}", rule_type, scope)),
    };

    // Rules use "rules" collection
    match queue_client.enqueue_content(
        "_global", // Rules are global
        "rules", &payload, "main", // Rules are branch-agnostic
    ) {
        Ok(result) => {
            if result.was_duplicate {
                output::warning("Rule already queued (duplicate)");
                output::kv("Idempotency Key", &result.idempotency_key);
            } else {
                output::success("Rule queued for processing");
                output::kv("Label", label);
                output::kv("Queue ID", &result.queue_id);
                output::kv("Status", "pending");
                output::kv("Fallback Mode", "unified_queue");
            }
            output::separator();
            output::info("The rule will be added when the daemon starts.");
            output::info("Check status with: wqm status queue");
        }
        Err(e) => {
            output::error(format!("Failed to enqueue rule: {}", e));
        }
    }

    Ok(())
}
