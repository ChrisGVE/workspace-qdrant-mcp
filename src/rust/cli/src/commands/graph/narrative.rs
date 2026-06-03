//! Graph narrative subcommand — narrative nodes linked to code symbols or concepts

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::grpc::client::workspace_daemon::{
    narrative_query_request::QueryTarget, NarrativeQueryRequest,
};
use crate::output::canvas;

/// Query target: either a symbol or a concept.
pub enum Target {
    Symbol(String),
    Concept(String),
}

/// Serialisable representation of a narrative node (for `--json`).
#[derive(Serialize)]
struct NarrativeNodeJson {
    node_id: String,
    symbol_name: String,
    symbol_type: String,
    file_path: String,
    edge_type: String,
    depth: i32,
    path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<serde_json::Value>,
}

/// Serialisable response wrapper.
#[derive(Serialize)]
struct NarrativeResponseJson {
    query_target: String,
    tenant_id: String,
    total_found: i32,
    nodes: Vec<NarrativeNodeJson>,
}

pub async fn narrative_query(
    target: Target,
    tenant_id: &str,
    depth: i32,
    limit: i32,
    edge_types: Vec<String>,
    json: bool,
) -> Result<()> {
    let (label, name, query_target) = match &target {
        Target::Symbol(s) => ("Symbol", s.as_str(), QueryTarget::SymbolName(s.clone())),
        Target::Concept(c) => ("Concept", c.as_str(), QueryTarget::ConceptName(c.clone())),
    };

    let mut client = crate::grpc::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .narrative_query(NarrativeQueryRequest {
            tenant_id: tenant_id.to_string(),
            query_target: Some(query_target),
            edge_types,
            max_depth: depth,
            max_results: limit,
        })
        .await
        .context("NarrativeQuery RPC failed")?
        .into_inner();

    // --- JSON output path -------------------------------------------------
    if json {
        let nodes: Vec<NarrativeNodeJson> = resp
            .nodes
            .iter()
            .map(|n| NarrativeNodeJson {
                node_id: n.node_id.clone(),
                symbol_name: n.symbol_name.clone(),
                symbol_type: n.symbol_type.clone(),
                file_path: n.file_path.clone(),
                edge_type: n.edge_type.clone(),
                depth: n.depth,
                path: n.path.clone(),
                metadata: n
                    .metadata_json
                    .as_deref()
                    .and_then(|s| serde_json::from_str(s).ok()),
            })
            .collect();

        let out = NarrativeResponseJson {
            query_target: format!("{label}: {name}"),
            tenant_id: tenant_id.to_string(),
            total_found: resp.total_found,
            nodes,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    // --- Human-readable output path ----------------------------------------
    canvas::print_title("Narrative Graph");
    canvas::print_blank();
    println!("  {label}: {name}");
    println!("  Tenant: {tenant_id}");

    if resp.nodes.is_empty() {
        canvas::print_blank();
        println!(
            "No narrative nodes found for {l} \"{n}\". \
             Narrative graph populates after re-ingestion with narrative extraction enabled.",
            l = label.to_lowercase(),
            n = name,
        );
        return Ok(());
    }

    // Group by edge_type, preserving insertion order via BTreeMap.
    let mut by_edge: BTreeMap<&str, Vec<_>> = BTreeMap::new();
    for node in &resp.nodes {
        by_edge.entry(&node.edge_type).or_default().push(node);
    }

    for (edge_type, nodes) in &by_edge {
        canvas::print_blank();
        println!("{edge_type} ({count}):", count = nodes.len());
        for n in nodes {
            let location = if n.file_path.is_empty() {
                String::new()
            } else {
                format!(" {}", n.file_path)
            };
            let name_part = if n.symbol_name.is_empty() {
                String::new()
            } else {
                format!(" \"{}\"", n.symbol_name)
            };
            println!(
                "  [{symbol_type}]{location}{name_part}",
                symbol_type = n.symbol_type,
            );
        }
    }

    canvas::print_blank();
    println!("Total: {} narrative nodes", resp.total_found);

    Ok(())
}
