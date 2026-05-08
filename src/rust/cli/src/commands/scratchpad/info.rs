//! Scratchpad info handler — display detailed entry by title substring

use anyhow::Context;
use anyhow::Result;

use crate::output;

use super::client::{build_qdrant_client, qdrant_url};
use super::types::{payload_str, payload_tags, ScratchJson, ScrollResponse};

pub(super) async fn scratchpad_info(identifier: &str, json: bool) -> Result<()> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_SCRATCHPAD;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    let body = serde_json::json!({
        "limit": 1,
        "with_payload": true,
        "filter": {
            "must": [{
                "key": "title",
                "match": { "text": identifier }
            }]
        }
    });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("Failed to connect to Qdrant")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            output::info("Scratchpad collection does not exist yet.");
            return Ok(());
        }
        anyhow::bail!("Qdrant request failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant response")?;

    if scroll.result.points.is_empty() {
        output::warning(format!(
            "No scratchpad entry found matching '{}'",
            identifier
        ));
        return Ok(());
    }

    let payload = match &scroll.result.points[0].payload {
        Some(p) => p,
        None => {
            output::error("Entry has no payload data");
            return Ok(());
        }
    };

    if json {
        let entry = ScratchJson {
            title: payload_str(payload, "title"),
            content: payload_str(payload, "content"),
            tenant_id: payload_str(payload, "tenant_id"),
            tags: payload_tags(payload),
            source_type: payload_str(payload, "source_type"),
            created_at: payload_str(payload, "created_at"),
        };
        output::print_json(&entry);
        return Ok(());
    }

    print_info_table(payload)
}

fn print_info_table(payload: &serde_json::Value) -> Result<()> {
    let project_names = crate::commands::tenant::load_project_names();
    let title = payload_str(payload, "title");

    output::section(format!(
        "Scratchpad: {}",
        if title.is_empty() {
            "(untitled)"
        } else {
            &title
        }
    ));
    output::kv("Title", if title.is_empty() { "-" } else { &title });
    output::kv(
        "Tenant",
        crate::commands::tenant::resolve_tenant_name(
            &payload_str(payload, "tenant_id"),
            &project_names,
        ),
    );

    let tags = payload_tags(payload);
    if !tags.is_empty() {
        output::kv("Tags", tags.join(", "));
    }

    output::kv(
        "Created",
        wqm_common::timestamp_fmt::format_local(&payload_str(payload, "created_at")),
    );
    output::separator();
    output::section("Content");
    println!("{}", payload_str(payload, "content"));

    Ok(())
}
