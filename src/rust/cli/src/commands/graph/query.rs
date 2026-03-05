//! Graph query subcommand — related nodes within N hops

use anyhow::{Context, Result};

use crate::grpc::client::workspace_daemon::QueryRelatedRequest;
use crate::grpc::client::DaemonClient;
use crate::output;

pub async fn query_related(
    node_id: &str,
    tenant_id: &str,
    max_hops: u32,
    edge_types: Vec<String>,
) -> Result<()> {
    output::section("Graph Query");
    output::kv("Node ID", node_id);
    output::kv("Tenant", tenant_id);
    output::kv("Max Hops", max_hops.to_string());
    if !edge_types.is_empty() {
        output::kv("Edge Types", edge_types.join(", "));
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .query_related(QueryRelatedRequest {
            tenant_id: tenant_id.to_string(),
            node_id: node_id.to_string(),
            max_hops,
            edge_types,
        })
        .await
        .context("QueryRelated RPC failed")?
        .into_inner();

    if resp.nodes.is_empty() {
        println!("No related nodes found.");
        return Ok(());
    }

    // Group by depth
    let mut by_depth: std::collections::BTreeMap<u32, Vec<_>> = std::collections::BTreeMap::new();
    for node in &resp.nodes {
        by_depth.entry(node.depth).or_default().push(node);
    }

    for (depth, nodes) in &by_depth {
        println!("\nDepth {} ({} nodes):", depth, nodes.len());
        for n in nodes {
            let loc = if n.file_path.is_empty() {
                "(stub)".to_string()
            } else {
                n.file_path.clone()
            };
            println!(
                "  {} {} ({}) [{}]",
                n.edge_type, n.symbol_name, n.symbol_type, loc
            );
        }
    }

    println!(
        "\nTotal: {} related nodes ({}ms)",
        resp.total, resp.query_time_ms
    );

    Ok(())
}
