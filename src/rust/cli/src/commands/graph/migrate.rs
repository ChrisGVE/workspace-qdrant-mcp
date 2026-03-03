//! Graph migrate subcommand — migrate graph data between backends

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::GraphMigrateRequest;
use crate::output;

pub async fn migrate(
    from: &str,
    to: &str,
    tenant_id: Option<String>,
    batch_size: Option<u32>,
) -> Result<()> {
    output::section("Graph Migration");
    output::kv("From", from);
    output::kv("To", to);
    if let Some(ref t) = tenant_id {
        output::kv("Tenant", t);
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .migrate_graph(GraphMigrateRequest {
            from_backend: from.to_string(),
            to_backend: to.to_string(),
            tenant_id,
            batch_size,
        })
        .await
        .context("MigrateGraph RPC failed")?
        .into_inner();

    println!("Success:        {}", resp.success);
    println!("Nodes exported: {}", resp.nodes_exported);
    println!("Nodes imported: {}", resp.nodes_imported);
    println!("Edges exported: {}", resp.edges_exported);
    println!("Edges imported: {}", resp.edges_imported);
    println!("Nodes match:    {}", resp.nodes_match);
    println!("Edges match:    {}", resp.edges_match);

    if !resp.warnings.is_empty() {
        println!("\nWarnings:");
        for w in &resp.warnings {
            println!("  {}", w);
        }
    }

    Ok(())
}
