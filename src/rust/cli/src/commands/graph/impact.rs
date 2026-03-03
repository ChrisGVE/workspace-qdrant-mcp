//! Graph impact analysis subcommand — nodes affected by changing a symbol

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::ImpactAnalysisRequest;
use crate::output;

pub async fn impact_analysis(
    symbol_name: &str,
    tenant_id: &str,
    file_path: Option<String>,
) -> Result<()> {
    output::section("Impact Analysis");
    output::kv("Symbol", symbol_name);
    output::kv("Tenant", tenant_id);
    if let Some(ref fp) = file_path {
        output::kv("File", fp);
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .impact_analysis(ImpactAnalysisRequest {
            tenant_id: tenant_id.to_string(),
            symbol_name: symbol_name.to_string(),
            file_path,
        })
        .await
        .context("ImpactAnalysis RPC failed")?
        .into_inner();

    if resp.impacted_nodes.is_empty() {
        println!("No impacted nodes found.");
        return Ok(());
    }

    // Group by impact type
    let mut direct = Vec::new();
    let mut indirect = Vec::new();

    for node in &resp.impacted_nodes {
        if node.distance <= 1 {
            direct.push(node);
        } else {
            indirect.push(node);
        }
    }

    if !direct.is_empty() {
        println!("\nDirect callers ({}):", direct.len());
        for n in &direct {
            println!("  {} ({})", n.symbol_name, n.file_path);
        }
    }

    if !indirect.is_empty() {
        println!("\nIndirect callers ({}):", indirect.len());
        for n in &indirect {
            println!(
                "  {} ({}) [distance: {}]",
                n.symbol_name, n.file_path, n.distance
            );
        }
    }

    println!(
        "\nTotal impacted: {} nodes ({}ms)",
        resp.total_impacted, resp.query_time_ms
    );

    Ok(())
}
