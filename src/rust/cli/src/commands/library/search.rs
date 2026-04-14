//! Search library content
//!
//! Semantic search over the libraries collection via EmbeddingService + Qdrant HTTP.

use anyhow::{Context, Result};
use tabled::Tabled;

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::EmbedTextRequest;
use crate::output::{self, ColumnHints};

fn qdrant_url() -> String {
    std::env::var("QDRANT_URL")
        .unwrap_or_else(|_| wqm_common::constants::DEFAULT_QDRANT_URL.to_string())
}

/// Library search result row
#[derive(Tabled)]
struct LibrarySearchRow {
    #[tabled(rename = "Library")]
    library: String,
    #[tabled(rename = "Title")]
    title: String,
    #[tabled(rename = "Content")]
    content: String,
}

impl ColumnHints for LibrarySearchRow {
    fn content_columns() -> &'static [usize] {
        &[2] // Content column
    }
}

/// Build a reqwest client with optional Qdrant API key.
fn build_qdrant_client() -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Ok(key) = std::env::var("QDRANT_API_KEY") {
        headers.insert(
            "api-key",
            reqwest::header::HeaderValue::from_str(&key).context("Invalid QDRANT_API_KEY")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")
}

/// Semantic search over libraries collection.
pub async fn search_library(query: &str, library: Option<String>, limit: usize) -> Result<()> {
    let mut daemon = ensure_daemon_available().await?;

    let embed_response = daemon
        .embedding()
        .embed_text(EmbedTextRequest {
            text: query.to_string(),
            model: None,
        })
        .await
        .context("Failed to generate query embedding")?
        .into_inner();

    if !embed_response.success {
        anyhow::bail!(
            "Embedding generation failed: {}",
            embed_response.error_message
        );
    }

    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_LIBRARIES;
    let url = format!("{}/collections/{}/points/search", qdrant_url(), collection);

    let mut body = serde_json::json!({
        "vector": embed_response.embedding,
        "limit": limit,
        "with_payload": true,
    });

    if let Some(ref lib_tag) = library {
        body["filter"] = serde_json::json!({
            "must": [{ "key": "library_name", "match": { "value": lib_tag } }]
        });
    }

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to search Qdrant")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            output::info("Libraries collection does not exist yet.");
            return Ok(());
        }
        anyhow::bail!("Qdrant search failed ({}): {}", status, text);
    }

    let json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse search response")?;

    let points = json["result"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or(&[]);

    if points.is_empty() {
        output::info("No matching library content found.");
        return Ok(());
    }

    output::section("Library Search Results");
    output::kv("Query", query);
    if let Some(lib) = &library {
        output::kv("Library", lib);
    }
    output::separator();

    let rows: Vec<LibrarySearchRow> = points
        .iter()
        .filter_map(|p| p.get("payload"))
        .map(|payload| {
            let content = payload
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let preview = if content.len() > 120 {
                format!("{}...", &content[..117])
            } else {
                content.to_string()
            };
            LibrarySearchRow {
                library: payload
                    .get("library_name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-")
                    .to_string(),
                title: payload
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("-")
                    .to_string(),
                content: preview,
            }
        })
        .collect();

    let count = rows.len();
    output::print_table_auto(&rows);
    output::summary(output::summary_line(count, count, "results"));

    Ok(())
}
