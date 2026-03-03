//! Graph stats subcommand — node/edge counts

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::GraphStatsRequest;
use crate::output;

pub async fn graph_stats(tenant_id: Option<String>) -> Result<()> {
    output::section("Graph Statistics");
    if let Some(ref t) = tenant_id {
        output::kv("Tenant", t);
    } else {
        output::kv("Scope", "all tenants");
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .get_graph_stats(GraphStatsRequest { tenant_id })
        .await
        .context("GetGraphStats RPC failed")?
        .into_inner();

    println!("Total nodes: {}", resp.total_nodes);
    println!("Total edges: {}", resp.total_edges);

    if !resp.nodes_by_type.is_empty() {
        println!("\nNodes by type:");
        let mut sorted: Vec<_> = resp.nodes_by_type.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (t, count) in sorted {
            println!("  {:15} {:>8}", t, count);
        }
    }

    if !resp.edges_by_type.is_empty() {
        println!("\nEdges by type:");
        let mut sorted: Vec<_> = resp.edges_by_type.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (t, count) in sorted {
            println!("  {:15} {:>8}", t, count);
        }
    }

    Ok(())
}
