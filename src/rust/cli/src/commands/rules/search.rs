//! Search rules subcommand
//!
//! Semantic search over rules collection via EmbeddingService + Qdrant HTTP.

use anyhow::{Context, Result};

use crate::grpc::ensure_daemon_available;
use crate::grpc::proto::EmbedTextRequest;
use crate::output;

use super::helpers::{
    build_qdrant_client, build_scope_filter, format_title_with_project, load_project_names,
    payload_str, qdrant_url, RuleRow, ScrollResponse,
};
use wqm_common::schema::qdrant::rules as rules_schema;

/// Search rules using semantic similarity via daemon embedding + Qdrant.
pub async fn search_rules(query: &str, scope: Option<String>, limit: usize) -> Result<()> {
    // Generate embedding for the query via daemon
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

    // Search Qdrant rules collection with the embedding vector
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_RULES;
    let url = format!("{}/collections/{}/points/search", qdrant_url(), collection);

    let mut body = serde_json::json!({
        "vector": embed_response.embedding,
        "limit": limit,
        "with_payload": true,
    });

    if let Some(ref scope_str) = scope {
        body["filter"] = build_scope_filter(scope_str);
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
            output::info("Rules collection does not exist yet. No rules stored.");
            return Ok(());
        }
        anyhow::bail!("Qdrant search failed ({}): {}", status, text);
    }

    // Parse Qdrant search response (similar to scroll but with score)
    let json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse Qdrant search response")?;

    let points = json["result"]
        .as_array()
        .map(|a| a.as_slice())
        .unwrap_or(&[]);

    if points.is_empty() {
        output::info("No matching rules found.");
        return Ok(());
    }

    output::section("Search Results");
    output::kv("Query", query);
    if let Some(s) = &scope {
        output::kv("Scope", s);
    }
    output::separator();

    let project_names = load_project_names();
    let rows: Vec<RuleRow> = points
        .iter()
        .filter_map(|p| p.get("payload"))
        .map(|payload| RuleRow {
            label: payload_str(payload, rules_schema::LABEL.name),
            title: format_title_with_project(payload, &project_names, false),
            scope: payload_str(payload, rules_schema::SCOPE.name),
            created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                payload,
                rules_schema::CREATED_AT.name,
            )),
        })
        .collect();

    let count = rows.len();
    output::print_table_auto(&rows);
    output::summary(output::summary_line(count, count, "matching rules"));

    Ok(())
}
