//! Graph pagerank subcommand — PageRank scores for graph nodes

use anyhow::{Context, Result};

use crate::grpc::client::workspace_daemon::PageRankRequest;
use crate::grpc::client::DaemonClient;
use crate::output;

pub async fn pagerank(
    tenant_id: &str,
    damping: Option<f64>,
    max_iterations: Option<u32>,
    tolerance: Option<f64>,
    top_k: Option<u32>,
    edge_types: Vec<String>,
) -> Result<()> {
    output::section("PageRank");
    output::kv("Tenant", tenant_id);
    if let Some(d) = damping {
        output::kv("Damping", &format!("{:.2}", d));
    }
    if let Some(k) = top_k {
        output::kv("Top K", &k.to_string());
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .compute_page_rank(PageRankRequest {
            tenant_id: tenant_id.to_string(),
            damping,
            max_iterations,
            tolerance,
            edge_types,
            top_k,
        })
        .await
        .context("ComputePageRank RPC failed")?
        .into_inner();

    if resp.entries.is_empty() {
        println!("No nodes found.");
        return Ok(());
    }

    println!("{:<10} {:<30} {:<12} {}", "SCORE", "SYMBOL", "TYPE", "FILE");
    for e in &resp.entries {
        let loc = if e.file_path.is_empty() {
            "(stub)".to_string()
        } else {
            e.file_path.clone()
        };
        println!(
            "{:<10.6} {:<30} {:<12} {}",
            e.score, e.symbol_name, e.symbol_type, loc
        );
    }

    println!(
        "\nShowing {}/{} nodes ({}ms)",
        resp.entries.len(),
        resp.total,
        resp.query_time_ms
    );

    Ok(())
}
