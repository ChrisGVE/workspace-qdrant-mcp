//! Scratchpad search handler — semantic vector search

use anyhow::Context;
use anyhow::Result;

use crate::output;

use super::client::{build_qdrant_client, qdrant_url, resolve_tenant_id};
use super::types::{payload_str, payload_tags, ScratchRow};

pub(super) async fn search_entries(
    query: &str,
    project: Option<String>,
    limit: usize,
) -> Result<()> {
    let project_names = crate::commands::tenant::load_project_names();
    let embedding = generate_embedding(query).await?;
    let points = search_qdrant(&embedding, project.as_deref(), limit).await?;

    if points.is_empty() {
        output::info("No matching scratchpad entries found.");
        return Ok(());
    }

    print_search_results(query, project.as_deref(), &points, &project_names);
    Ok(())
}

/// Generate an embedding vector for the query text via the daemon.
async fn generate_embedding(query: &str) -> Result<Vec<f32>> {
    use crate::grpc::proto::EmbedTextRequest;

    let mut daemon = crate::grpc::ensure_daemon_available().await?;

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

    Ok(embed_response.embedding)
}

/// Search the scratchpad collection in Qdrant with the given embedding.
async fn search_qdrant(
    embedding: &[f32],
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<serde_json::Value>> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_SCRATCHPAD;
    let url = format!("{}/collections/{}/points/search", qdrant_url(), collection);

    let mut body = serde_json::json!({
        "vector": embedding,
        "limit": limit,
        "with_payload": true,
    });

    if let Some(proj) = project {
        let tenant_id = resolve_tenant_id(Some(proj))?;
        body["filter"] = serde_json::json!({
            "must": [{ "key": "tenant_id", "match": { "value": tenant_id } }]
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
            return Ok(Vec::new());
        }
        anyhow::bail!("Qdrant search failed ({}): {}", status, text);
    }

    let json: serde_json::Value = response
        .json()
        .await
        .context("Failed to parse search response")?;

    Ok(json["result"].as_array().cloned().unwrap_or_default())
}

/// Display search results as a table.
fn print_search_results(
    query: &str,
    project: Option<&str>,
    points: &[serde_json::Value],
    project_names: &std::collections::HashMap<String, String>,
) {
    output::section("Scratchpad Search Results");
    output::kv("Query", query);
    if let Some(p) = project {
        output::kv("Project", p);
    }
    output::separator();

    let rows: Vec<ScratchRow> = points
        .iter()
        .filter_map(|p| p.get("payload"))
        .map(|payload| ScratchRow {
            title: payload_str(payload, "title"),
            tenant_id: crate::commands::tenant::resolve_tenant_name(
                &payload_str(payload, "tenant_id"),
                project_names,
            ),
            tags: payload_tags(payload).join(", "),
            created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                payload,
                "created_at",
            )),
        })
        .collect();

    let count = rows.len();
    output::print_table_auto(&rows);
    output::summary(output::summary_line(count, count, "results"));
}
