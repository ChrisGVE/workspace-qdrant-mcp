//! Graph command - code relationship queries
//!
//! Subcommands: query, impact, stats

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::{
    GraphStatsRequest, ImpactAnalysisRequest, QueryRelatedRequest,
};
use crate::output;

/// Graph command arguments
#[derive(Args)]
pub struct GraphArgs {
    #[command(subcommand)]
    command: GraphCommand,
}

/// Graph subcommands
#[derive(Subcommand)]
enum GraphCommand {
    /// Query nodes related to a symbol within N hops
    Query {
        /// Node ID to query from
        #[arg(long)]
        node_id: String,

        /// Project tenant_id
        #[arg(long)]
        tenant: String,

        /// Maximum traversal depth (1-5)
        #[arg(long, default_value = "2")]
        hops: u32,

        /// Edge type filter (comma-separated: CALLS,IMPORTS,CONTAINS,USES_TYPE,EXTENDS,IMPLEMENTS)
        #[arg(long, value_delimiter = ',')]
        edge_types: Vec<String>,
    },

    /// Impact analysis: find nodes affected by changing a symbol
    Impact {
        /// Symbol name to analyze
        #[arg(long)]
        symbol: String,

        /// Project tenant_id
        #[arg(long)]
        tenant: String,

        /// Narrow to specific file path
        #[arg(long)]
        file: Option<String>,
    },

    /// Graph statistics (node/edge counts)
    Stats {
        /// Project tenant_id (omit for all tenants)
        #[arg(long)]
        tenant: Option<String>,
    },
}

pub async fn execute(args: GraphArgs) -> Result<()> {
    match args.command {
        GraphCommand::Query {
            node_id,
            tenant,
            hops,
            edge_types,
        } => query_related(&node_id, &tenant, hops, edge_types).await,
        GraphCommand::Impact {
            symbol,
            tenant,
            file,
        } => impact_analysis(&symbol, &tenant, file).await,
        GraphCommand::Stats { tenant } => graph_stats(tenant).await,
    }
}

async fn query_related(
    node_id: &str,
    tenant_id: &str,
    max_hops: u32,
    edge_types: Vec<String>,
) -> Result<()> {
    output::section("Graph Query");
    output::kv("Node ID", node_id);
    output::kv("Tenant", tenant_id);
    output::kv("Max Hops", &max_hops.to_string());
    if !edge_types.is_empty() {
        output::kv("Edge Types", &edge_types.join(", "));
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

async fn impact_analysis(
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

async fn graph_stats(tenant_id: Option<String>) -> Result<()> {
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
