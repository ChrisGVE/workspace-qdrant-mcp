//! Show detailed information about a specific rule by label.

use std::collections::HashMap;

use anyhow::{Context, Result};

use wqm_common::schema::qdrant::rules as rules_schema;

use crate::output;

use super::helpers::{
    build_qdrant_client, load_project_names, normalize_commas, payload_str, payload_u32,
    qdrant_url, RuleJson, ScrollResponse,
};

/// Show detailed info for a single rule.
pub async fn rule_info(label: &str, json: bool) -> Result<()> {
    let client = build_qdrant_client()?;
    let collection = wqm_common::constants::COLLECTION_RULES;
    let url = format!("{}/collections/{}/points/scroll", qdrant_url(), collection);

    let body = serde_json::json!({
        "limit": 1,
        "with_payload": true,
        "filter": {
            "must": [{
                "key": rules_schema::LABEL.name,
                "match": { "value": label }
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
            output::info("Rules collection does not exist yet.");
            return Ok(());
        }
        anyhow::bail!("Qdrant request failed ({}): {}", status, text);
    }

    let scroll: ScrollResponse = response
        .json()
        .await
        .context("Failed to parse Qdrant response")?;

    let points = &scroll.result.points;
    if points.is_empty() {
        output::warning(format!("No rule found with label '{}'", label));
        return Ok(());
    }

    let payload = match &points[0].payload {
        Some(p) => p,
        None => {
            output::error("Rule has no payload data");
            return Ok(());
        }
    };

    if json {
        print_rule_json(payload);
    } else {
        print_rule_human(label, payload);
    }

    Ok(())
}

/// Print rule as JSON.
fn print_rule_json(payload: &serde_json::Value) {
    let rule = RuleJson {
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
    };
    output::print_json(&rule);
}

/// Print rule in human-readable columnar format.
fn print_rule_human(label: &str, payload: &serde_json::Value) {
    let project_names = load_project_names();

    output::section(format!("Rule: {}", label));
    output::kv("Label", payload_str(payload, rules_schema::LABEL.name));
    output::kv("Title", payload_str(payload, rules_schema::TITLE.name));
    output::kv("Scope", payload_str(payload, rules_schema::SCOPE.name));

    print_project_field(payload, &project_names);

    output::kv("Type", payload_str(payload, rules_schema::SOURCE_TYPE.name));

    let tags = payload_str(payload, rules_schema::TAGS.name);
    if !tags.is_empty() {
        output::kv("Tags", normalize_commas(&tags));
    }

    output::kv(
        "Created",
        wqm_common::timestamp_fmt::format_local(&payload_str(
            payload,
            rules_schema::CREATED_AT.name,
        )),
    );
    output::kv(
        "Updated",
        wqm_common::timestamp_fmt::format_local(&payload_str(
            payload,
            rules_schema::UPDATED_AT.name,
        )),
    );

    output::separator();
    output::section("Content");
    println!("{}", payload_str(payload, rules_schema::CONTENT.name));
}

/// Print the project field with name resolution.
fn print_project_field(payload: &serde_json::Value, project_names: &HashMap<String, String>) {
    if let Some(pid) = payload
        .get(rules_schema::PROJECT_ID.name)
        .and_then(|v| v.as_str())
    {
        let name = project_names.get(pid).map(|n| n.as_str()).unwrap_or(pid);
        output::kv("Project", name);
    }
}
