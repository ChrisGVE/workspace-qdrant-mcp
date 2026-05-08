//! Scratchpad list handler — scroll and render entries

use anyhow::Context;
use anyhow::Result;

use crate::output;

use super::client::{build_qdrant_client, qdrant_url, resolve_tenant_id};
use super::types::{
    payload_str, payload_tags, QdrantPoint, ScratchJson, ScratchRow, ScratchRowVerbose,
    ScrollResponse,
};

pub(super) async fn list_entries(
    project: Option<String>,
    limit: usize,
    verbose: bool,
    format: &str,
    script: bool,
    no_headers: bool,
) -> Result<()> {
    let client = build_qdrant_client()?;
    let points = fetch_scroll_points(&client, project.as_deref(), limit).await?;

    if points.is_empty() {
        output::info("No scratchpad entries found.");
        return Ok(());
    }

    if format == "json" {
        print_json_entries(&points);
        return Ok(());
    }

    if script {
        print_script_entries(&points, verbose, no_headers);
        return Ok(());
    }

    print_table_entries(&points, project.as_deref(), verbose);
    Ok(())
}

async fn fetch_scroll_points(
    client: &reqwest::Client,
    project: Option<&str>,
    limit: usize,
) -> Result<Vec<QdrantPoint>> {
    let collection = wqm_common::constants::COLLECTION_SCRATCHPAD;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    let mut body = serde_json::json!({
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
        .context("Failed to connect to Qdrant")?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        if status.as_u16() == 404 {
            output::info("Scratchpad collection does not exist yet. No entries stored.");
            return Ok(Vec::new());
        }
        anyhow::bail!("Qdrant scroll failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant scroll response")?;

    Ok(scroll.result.points)
}

fn print_json_entries(points: &[QdrantPoint]) {
    let project_names = crate::commands::tenant::load_project_names();
    let entries: Vec<ScratchJson> = points
        .iter()
        .filter_map(|p| p.payload.as_ref())
        .map(|payload| ScratchJson {
            title: payload_str(payload, "title"),
            content: payload_str(payload, "content"),
            tenant_id: crate::commands::tenant::resolve_tenant_name(
                &payload_str(payload, "tenant_id"),
                &project_names,
            ),
            tags: payload_tags(payload),
            source_type: payload_str(payload, "source_type"),
            created_at: payload_str(payload, "created_at"),
        })
        .collect();
    output::print_json(&entries);
}

fn print_script_entries(points: &[QdrantPoint], verbose: bool, no_headers: bool) {
    let project_names = crate::commands::tenant::load_project_names();
    if verbose {
        let rows: Vec<ScratchRowVerbose> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRowVerbose {
                title: payload_str(payload, "title"),
                tenant_id: crate::commands::tenant::resolve_tenant_name(
                    &payload_str(payload, "tenant_id"),
                    &project_names,
                ),
                tags: payload_tags(payload).join(","),
                content: payload_str(payload, "content"),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    "created_at",
                )),
            })
            .collect();
        output::print_script(&rows, !no_headers);
    } else {
        let rows: Vec<ScratchRow> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRow {
                title: payload_str(payload, "title"),
                tenant_id: crate::commands::tenant::resolve_tenant_name(
                    &payload_str(payload, "tenant_id"),
                    &project_names,
                ),
                tags: payload_tags(payload).join(","),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    "created_at",
                )),
            })
            .collect();
        output::print_script(&rows, !no_headers);
    }
}

fn print_table_entries(points: &[QdrantPoint], project: Option<&str>, verbose: bool) {
    let project_names = crate::commands::tenant::load_project_names();
    output::section("Scratchpad Entries");
    if let Some(p) = project {
        output::kv("Filter", p);
    }
    output::separator();

    let count = points.len();

    if verbose {
        let rows: Vec<ScratchRowVerbose> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRowVerbose {
                title: payload_str(payload, "title"),
                tenant_id: crate::commands::tenant::resolve_tenant_name(
                    &payload_str(payload, "tenant_id"),
                    &project_names,
                ),
                tags: payload_tags(payload).join(", "),
                content: payload_str(payload, "content"),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    "created_at",
                )),
            })
            .collect();
        output::print_table_auto(&rows);
    } else {
        let rows: Vec<ScratchRow> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| ScratchRow {
                title: payload_str(payload, "title"),
                tenant_id: crate::commands::tenant::resolve_tenant_name(
                    &payload_str(payload, "tenant_id"),
                    &project_names,
                ),
                tags: payload_tags(payload).join(", "),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    "created_at",
                )),
            })
            .collect();
        output::print_table_auto(&rows);
    }

    output::summary(output::summary_line(count, count, "scratchpad entries"));
}
