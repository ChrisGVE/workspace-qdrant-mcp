//! Graph communities subcommand — community detection via label propagation

use anyhow::{Context, Result};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::CommunityRequest;
use crate::output;

pub async fn communities(
    tenant_id: &str,
    max_iterations: Option<u32>,
    min_size: Option<u32>,
    edge_types: Vec<String>,
) -> Result<()> {
    output::section("Community Detection");
    output::kv("Tenant", tenant_id);
    if let Some(ms) = min_size {
        output::kv("Min Size", &ms.to_string());
    }
    output::separator();

    let mut client = DaemonClient::connect_default()
        .await
        .context("Cannot connect to daemon")?;

    let resp = client
        .graph()
        .detect_communities(CommunityRequest {
            tenant_id: tenant_id.to_string(),
            max_iterations,
            min_community_size: min_size,
            edge_types,
        })
        .await
        .context("DetectCommunities RPC failed")?
        .into_inner();

    if resp.communities.is_empty() {
        println!("No communities detected.");
        return Ok(());
    }

    for c in &resp.communities {
        println!(
            "\nCommunity {} ({} members):",
            c.community_id,
            c.members.len()
        );
        for m in &c.members {
            let loc = if m.file_path.is_empty() {
                "(stub)".to_string()
            } else {
                m.file_path.clone()
            };
            println!("  {} ({}) [{}]", m.symbol_name, m.symbol_type, loc);
        }
    }

    println!(
        "\nTotal: {} communities ({}ms)",
        resp.total_communities, resp.query_time_ms
    );

    Ok(())
}
