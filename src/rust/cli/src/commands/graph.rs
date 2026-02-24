//! Graph command - code relationship queries and algorithms
//!
//! Subcommands: query, impact, stats, pagerank, communities, betweenness, migrate

use anyhow::{Context, Result};
use clap::{Args, Subcommand};

use crate::grpc::client::DaemonClient;
use crate::grpc::client::workspace_daemon::{
    BetweennessRequest, CommunityRequest, GraphMigrateRequest, GraphStatsRequest,
    ImpactAnalysisRequest, PageRankRequest, QueryRelatedRequest,
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

    /// Compute PageRank scores for graph nodes
    Pagerank {
        /// Project tenant_id
        #[arg(long)]
        tenant: String,

        /// Damping factor (default: 0.85)
        #[arg(long)]
        damping: Option<f64>,

        /// Maximum iterations (default: 100)
        #[arg(long)]
        max_iterations: Option<u32>,

        /// Convergence tolerance (default: 1e-6)
        #[arg(long)]
        tolerance: Option<f64>,

        /// Return only top K results
        #[arg(long)]
        top_k: Option<u32>,

        /// Edge type filter (comma-separated)
        #[arg(long, value_delimiter = ',')]
        edge_types: Vec<String>,
    },

    /// Detect code communities via label propagation
    Communities {
        /// Project tenant_id
        #[arg(long)]
        tenant: String,

        /// Max label propagation iterations (default: 50)
        #[arg(long)]
        max_iterations: Option<u32>,

        /// Minimum community size to include (default: 2)
        #[arg(long)]
        min_size: Option<u32>,

        /// Edge type filter (comma-separated)
        #[arg(long, value_delimiter = ',')]
        edge_types: Vec<String>,
    },

    /// Compute betweenness centrality scores
    Betweenness {
        /// Project tenant_id
        #[arg(long)]
        tenant: String,

        /// Return only top K results
        #[arg(long)]
        top_k: Option<u32>,

        /// Sample N source nodes for large graphs (0 = all)
        #[arg(long)]
        max_samples: Option<u32>,

        /// Edge type filter (comma-separated)
        #[arg(long, value_delimiter = ',')]
        edge_types: Vec<String>,
    },

    /// Migrate graph data between backends
    Migrate {
        /// Source backend (sqlite or ladybug)
        #[arg(long, default_value = "sqlite")]
        from: String,

        /// Target backend (sqlite or ladybug)
        #[arg(long, default_value = "ladybug")]
        to: String,

        /// Migrate specific tenant (omit for all)
        #[arg(long)]
        tenant: Option<String>,

        /// Import batch size (default: 500)
        #[arg(long)]
        batch_size: Option<u32>,
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
        GraphCommand::Pagerank {
            tenant,
            damping,
            max_iterations,
            tolerance,
            top_k,
            edge_types,
        } => pagerank(&tenant, damping, max_iterations, tolerance, top_k, edge_types).await,
        GraphCommand::Communities {
            tenant,
            max_iterations,
            min_size,
            edge_types,
        } => communities(&tenant, max_iterations, min_size, edge_types).await,
        GraphCommand::Betweenness {
            tenant,
            top_k,
            max_samples,
            edge_types,
        } => betweenness(&tenant, top_k, max_samples, edge_types).await,
        GraphCommand::Migrate {
            from,
            to,
            tenant,
            batch_size,
        } => migrate(&from, &to, tenant, batch_size).await,
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

async fn pagerank(
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

    println!(
        "{:<10} {:<30} {:<12} {}",
        "SCORE", "SYMBOL", "TYPE", "FILE"
    );
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

async fn communities(
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

async fn betweenness(
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

    println!(
        "{:<10} {:<30} {:<12} {}",
        "SCORE", "SYMBOL", "TYPE", "FILE"
    );
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

async fn migrate(
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
