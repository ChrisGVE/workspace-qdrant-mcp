//! List rules subcommand
//!
//! Fetches and displays rules from the Qdrant `rules` collection,
//! with optional scope filtering, verbose output, and JSON format.

use anyhow::{Context, Result};

use wqm_common::schema::qdrant::rules as rules_schema;

use crate::output;

use super::helpers::{
    build_qdrant_client, build_scope_filter, format_title_with_project, load_project_names,
    normalize_commas, payload_str, payload_u32, qdrant_url, RuleJson, RuleRow, RuleRowVerbose,
    ScrollResponse,
};

/// List rules from Qdrant with optional scope and type filters.
pub async fn list_rules(
    scope: Option<String>,
    _rule_type: Option<String>,
    verbose: bool,
    format: &str,
) -> Result<()> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_RULES;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    // Build scroll request with optional scope filter
    let mut body = serde_json::json!({
        "limit": 100,
        "with_payload": true,
    });

    if let Some(ref scope_str) = scope {
        let filter = build_scope_filter(scope_str);
        body["filter"] = filter;
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
            output::info("Rules collection does not exist yet. No rules stored.");
            return Ok(());
        }
        anyhow::bail!("Qdrant scroll failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant scroll response")?;

    let points = &scroll.result.points;

    if points.is_empty() {
        output::info("No rules found.");
        return Ok(());
    }

    // JSON output
    if format == "json" {
        return print_json_output(points);
    }

    // Table output
    print_table_output(points, &scope, verbose);

    Ok(())
}

/// Render rules as JSON array to stdout.
fn print_json_output(
    points: &[super::helpers::QdrantPoint],
) -> Result<()> {
    let rules: Vec<RuleJson> = points
        .iter()
        .filter_map(|p| p.payload.as_ref())
        .map(|payload| RuleJson {
            label: payload_str(payload, rules_schema::LABEL.name),
            title: payload_str(payload, rules_schema::TITLE.name),
            content: payload_str(payload, rules_schema::CONTENT.name),
            scope: payload_str(payload, rules_schema::SCOPE.name),
            project_id: payload
                .get(rules_schema::PROJECT_ID.name)
                .and_then(|v| v.as_str())
                .map(String::from),
            source_type: payload_str(payload, rules_schema::SOURCE_TYPE.name),
            priority: payload_u32(payload, rules_schema::PRIORITY.name),
            tags: payload
                .get(rules_schema::TAGS.name)
                .and_then(|v| v.as_str())
                .map(|s| s.split(',').map(String::from).collect())
                .unwrap_or_default(),
            created_at: payload_str(payload, rules_schema::CREATED_AT.name),
            updated_at: payload_str(payload, rules_schema::UPDATED_AT.name),
        })
        .collect();
    output::print_json(&rules);
    Ok(())
}

/// Render rules as a formatted table to stdout.
fn print_table_output(
    points: &[super::helpers::QdrantPoint],
    scope: &Option<String>,
    verbose: bool,
) {
    output::section("Rules");
    output::kv("Total", &points.len().to_string());
    if let Some(s) = scope {
        output::kv("Filter", s);
    }
    output::separator();

    let project_names = load_project_names();

    if verbose {
        let rows: Vec<RuleRowVerbose> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| RuleRowVerbose {
                label: payload_str(payload, rules_schema::LABEL.name),
                title: format_title_with_project(payload, &project_names, true),
                scope: payload_str(payload, rules_schema::SCOPE.name),
                priority: payload_u32(payload, rules_schema::PRIORITY.name)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                tags: normalize_commas(&payload_str(payload, rules_schema::TAGS.name)),
                content: payload_str(payload, rules_schema::CONTENT.name),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    rules_schema::CREATED_AT.name,
                )),
            })
            .collect();
        output::print_table_auto(&rows);
    } else {
        let rows: Vec<RuleRow> = points
            .iter()
            .filter_map(|p| p.payload.as_ref())
            .map(|payload| RuleRow {
                label: payload_str(payload, rules_schema::LABEL.name),
                title: format_title_with_project(payload, &project_names, false),
                scope: payload_str(payload, rules_schema::SCOPE.name),
                priority: payload_u32(payload, rules_schema::PRIORITY.name)
                    .map(|p| p.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                created_at: wqm_common::timestamp_fmt::format_local(&payload_str(
                    payload,
                    rules_schema::CREATED_AT.name,
                )),
            })
            .collect();
        output::print_table_auto(&rows);
    }
}
