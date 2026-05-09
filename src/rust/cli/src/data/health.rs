//! Health check helpers for external services.
//!
//! Every command that reports Qdrant-sourced data must verify
//! connectivity before displaying metrics.

use crate::commands::qdrant_helpers::{build_qdrant_http_client, qdrant_base_url};

/// Result of a Qdrant health check.
#[derive(Debug, Clone)]
pub struct QdrantHealth {
    pub reachable: bool,
    pub collection_count: usize,
    pub error: Option<String>,
}

/// Check if Qdrant is reachable and return collection count.
///
/// Uses a short timeout (3s) to avoid blocking the CLI when Qdrant is down.
/// Returns a health struct rather than erroring — callers decide how to
/// display the result (warning line, omit fields, etc.).
pub async fn check_qdrant() -> QdrantHealth {
    let client = match build_qdrant_http_client() {
        Ok(c) => c,
        Err(e) => {
            return QdrantHealth {
                reachable: false,
                collection_count: 0,
                error: Some(format!("Failed to build HTTP client: {}", e)),
            }
        }
    };

    let url = format!("{}/collections", qdrant_base_url());

    // Use a shorter timeout for health checks
    let short_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
        .unwrap_or(client);

    match short_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            let count = resp
                .json::<serde_json::Value>()
                .await
                .ok()
                .and_then(|v| v["result"]["collections"].as_array().map(|a| a.len()))
                .unwrap_or(0);

            QdrantHealth {
                reachable: true,
                collection_count: count,
                error: None,
            }
        }
        Ok(resp) => QdrantHealth {
            reachable: false,
            collection_count: 0,
            error: Some(format!("Qdrant returned status {}", resp.status())),
        },
        Err(e) => QdrantHealth {
            reachable: false,
            collection_count: 0,
            error: Some(if e.is_timeout() {
                "Qdrant connection timed out".to_string()
            } else if e.is_connect() {
                "Qdrant not reachable (connection refused)".to_string()
            } else {
                format!("Qdrant error: {}", e)
            }),
        },
    }
}

