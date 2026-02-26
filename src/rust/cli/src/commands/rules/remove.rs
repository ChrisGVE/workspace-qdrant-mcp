//! Remove rule subcommand
//!
//! Deletes a rule from the Qdrant `rules` collection via daemon gRPC.

use anyhow::Result;

use crate::grpc::client::DaemonClient;
use crate::grpc::proto::DeleteDocumentRequest;
use crate::output;

/// Remove a rule by label via daemon gRPC.
pub async fn remove_rule(label: &str, scope: &Option<String>) -> Result<()> {
    output::section("Remove Rule");

    let scope_str = scope.as_deref().unwrap_or("all");

    output::kv("Label", label);
    output::kv("Scope", scope_str);
    output::separator();

    match DaemonClient::connect_default().await {
        Ok(mut client) => {
            // Use label as document_id since we store rules with label as ID
            let request = DeleteDocumentRequest {
                document_id: label.to_string(),
                collection_name: "rules".to_string(),
            };

            match client.document().delete_document(request).await {
                Ok(_) => {
                    output::success("Rule removed");
                    output::kv("Label", label);
                }
                Err(e) => {
                    output::error(format!("Failed to remove rule: {}", e));
                }
            }
        }
        Err(_) => {
            output::error("Daemon not running. Start with: wqm service start");
        }
    }

    Ok(())
}
