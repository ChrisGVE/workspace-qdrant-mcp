//! `wqm graph topics` — show topic coverage for a concept, grouped by depth.
//!
//! Uses the NarrativeQuery gRPC RPC with `concept_name` query target to
//! retrieve narrative nodes linked to a concept and groups them by depth level.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::grpc::client::workspace_daemon::{
    narrative_query_request::QueryTarget, NarrativeQueryRequest,
};
use crate::output::canvas;
use crate::output::{self};

/// JSON output structure for the topics command.
#[derive(Serialize)]
struct TopicsJson {
    concept: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tenant_id: Option<String>,
    total: usize,
    depth_groups: Vec<DepthGroupJson>,
}

#[derive(Serialize)]
struct DepthGroupJson {
    depth: String,
    count: usize,
    nodes: Vec<TopicNodeJson>,
}

#[derive(Serialize)]
struct TopicNodeJson {
    symbol_type: String,
    file_path: String,
    symbol_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    edge_type: Option<String>,
}

/// A parsed narrative node with an extracted depth level for grouping.
struct ParsedNode {
    depth: String,
    symbol_type: String,
    file_path: String,
    symbol_name: String,
}

/// Extract the depth level string from a NarrativeNode's metadata_json.
///
/// Expected format: `{"depth":"rigorous"}`. Falls back to "unknown" if
/// parsing fails or metadata is absent.
fn extract_depth(metadata_json: &Option<String>) -> String {
    metadata_json
        .as_deref()
        .and_then(|json| {
            let trimmed = json.trim();
            let start = trimmed.find("\"depth\"")?;
            let rest = &trimmed[start + 7..];
            let colon = rest.find(':')?;
            let after_colon = rest[colon + 1..].trim().trim_start_matches('"');
            let end = after_colon.find('"')?;
            Some(after_colon[..end].to_string())
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Canonical depth ordering for display.
fn depth_sort_key(depth: &str) -> u8 {
    match depth {
        "reference" => 0,
        "rigorous" => 1,
        "intermediate" => 2,
        "introductory" => 3,
        "qualitative" => 4,
        _ => 5,
    }
}

/// Execute the `wqm graph topics` subcommand.
pub async fn topics(concept: &str, tenant_id: Option<&str>, json: bool) -> Result<()> {
    let auto_tenant;
    let tid = match tenant_id {
        Some(t) => t,
        None => {
            auto_tenant = crate::commands::ingest::detect::detect_tenant_id();
            auto_tenant.as_str()
        }
    };

    let mut client = crate::grpc::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .narrative_query(NarrativeQueryRequest {
            tenant_id: tid.to_string(),
            query_target: Some(QueryTarget::ConceptName(concept.to_string())),
            edge_types: vec![],
            max_depth: 2,
            max_results: 200,
        })
        .await
        .context("NarrativeQuery RPC failed")?
        .into_inner();

    if resp.nodes.is_empty() {
        if json {
            output::print_json(&TopicsJson {
                concept: concept.to_string(),
                tenant_id: Some(tid.to_string()),
                total: 0,
                depth_groups: Vec::new(),
            });
        } else {
            output::info(format!(
                "No coverage found for concept '{}'. \
                 Check wqm graph concepts for available concepts.",
                concept,
            ));
        }
        return Ok(());
    }

    // Parse nodes and group by depth
    let parsed: Vec<ParsedNode> = resp
        .nodes
        .iter()
        .map(|n| ParsedNode {
            depth: extract_depth(&n.metadata_json),
            symbol_type: n.symbol_type.clone(),
            file_path: n.file_path.clone(),
            symbol_name: n.symbol_name.clone(),
        })
        .collect();

    let mut by_depth: BTreeMap<String, Vec<&ParsedNode>> = BTreeMap::new();
    for node in &parsed {
        by_depth.entry(node.depth.clone()).or_default().push(node);
    }

    // Sort groups by canonical depth ordering
    let mut groups: Vec<(String, Vec<&ParsedNode>)> = by_depth.into_iter().collect();
    groups.sort_by_key(|(depth, _)| depth_sort_key(depth));

    if json {
        print_topics_json(concept, Some(tid), &groups);
    } else {
        print_topics_table(concept, &groups);
    }

    Ok(())
}

fn print_topics_json(
    concept: &str,
    tenant_id: Option<&str>,
    groups: &[(String, Vec<&ParsedNode>)],
) {
    let total: usize = groups.iter().map(|(_, nodes)| nodes.len()).sum();
    let depth_groups: Vec<DepthGroupJson> = groups
        .iter()
        .map(|(depth, nodes)| DepthGroupJson {
            depth: depth.clone(),
            count: nodes.len(),
            nodes: nodes
                .iter()
                .map(|n| TopicNodeJson {
                    symbol_type: n.symbol_type.clone(),
                    file_path: n.file_path.clone(),
                    symbol_name: n.symbol_name.clone(),
                    edge_type: None,
                })
                .collect(),
        })
        .collect();

    output::print_json(&TopicsJson {
        concept: concept.to_string(),
        tenant_id: tenant_id.map(String::from),
        total,
        depth_groups,
    });
}

fn print_topics_table(concept: &str, groups: &[(String, Vec<&ParsedNode>)]) {
    canvas::print_title(&format!("Topic Coverage: {}", concept));
    canvas::print_blank();

    for (depth, nodes) in groups {
        let label = format!("{} ({}):", capitalize_first(depth), nodes.len(),);
        output::info(label);

        for node in nodes {
            let loc = if node.file_path.is_empty() {
                "(no file)".to_string()
            } else {
                node.file_path.clone()
            };
            println!("  [{}] {} \"{}\"", node.symbol_type, loc, node.symbol_name,);
        }
        println!();
    }
}

/// Capitalize the first letter of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            format!("{upper}{}", chars.as_str())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_depth_from_valid_json() {
        let meta = Some(r#"{"depth":"rigorous"}"#.to_string());
        assert_eq!(extract_depth(&meta), "rigorous");
    }

    #[test]
    fn extract_depth_from_none() {
        assert_eq!(extract_depth(&None), "unknown");
    }

    #[test]
    fn extract_depth_from_malformed_json() {
        let meta = Some("not json".to_string());
        assert_eq!(extract_depth(&meta), "unknown");
    }

    #[test]
    fn depth_sort_order() {
        assert!(depth_sort_key("reference") < depth_sort_key("rigorous"));
        assert!(depth_sort_key("rigorous") < depth_sort_key("intermediate"));
        assert!(depth_sort_key("intermediate") < depth_sort_key("introductory"));
        assert!(depth_sort_key("introductory") < depth_sort_key("qualitative"));
        assert!(depth_sort_key("qualitative") < depth_sort_key("anything-else"));
    }

    #[test]
    fn capitalize_first_basic() {
        assert_eq!(capitalize_first("rigorous"), "Rigorous");
        assert_eq!(capitalize_first(""), "");
        assert_eq!(capitalize_first("a"), "A");
    }

    #[test]
    fn topics_json_serializes_empty() {
        let json_out = TopicsJson {
            concept: "test-concept".to_string(),
            tenant_id: None,
            total: 0,
            depth_groups: Vec::new(),
        };
        let serialized = serde_json::to_string(&json_out).unwrap();
        assert!(serialized.contains("test-concept"));
        assert!(!serialized.contains("tenant_id"));
    }
}
