//! Graph betweenness subcommand — betweenness centrality scores

use anyhow::{Context, Result};

use crate::grpc::client::workspace_daemon::BetweennessRequest;
use crate::grpc::client::DaemonClient;
use crate::output;

pub async fn betweenness(
    tenant_id: &str,
    top_k: Option<u32>,
    max_samples: Option<u32>,
    edge_types: Vec<String>,
) -> Result<()> {
    output::section("Betweenness Centrality");
    output::kv("Tenant", tenant_id);
    if let Some(k) = top_k {
        output::kv("Top K", &k.to_string());
    }
    if let Some(s) = max_samples {
        output::kv("Max Samples", &s.to_string());
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .compute_betweenness(BetweennessRequest {
            tenant_id: tenant_id.to_string(),
            edge_types,
            max_samples,
            top_k,
        })
        .await
        .context("ComputeBetweenness RPC failed")?
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
