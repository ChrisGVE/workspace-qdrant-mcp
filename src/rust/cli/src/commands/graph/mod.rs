//! Graph command - code relationship queries and algorithms
//!
//! Subcommands: query, impact, stats, pagerank, communities, betweenness,
//! migrate, narrative, concepts, topics

use anyhow::Result;
use clap::{Args, Subcommand};

mod betweenness;
mod communities;
mod concepts;
mod impact;
mod migrate;
mod narrative;
mod pagerank;
mod query;
mod stats;
mod topics;

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

        /// Project name or tenant id (partial input resolved)
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

        /// Project name or tenant id (partial input resolved)
        #[arg(long)]
        tenant: String,

        /// Narrow to specific file path
        #[arg(long)]
        file: Option<String>,
    },

    /// Graph statistics (node/edge counts)
    Stats {
        /// Project name or tenant id (partial input resolved; omit for all tenants)
        #[arg(long)]
        tenant: Option<String>,
    },

    /// Compute PageRank scores for graph nodes
    Pagerank {
        /// Project name or tenant id (partial input resolved)
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
        /// Project name or tenant id (partial input resolved)
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
        /// Project name or tenant id (partial input resolved)
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

        /// Migrate specific project — name or tenant id, partial input resolved (omit for all)
        #[arg(long)]
        tenant: Option<String>,

        /// Import batch size (default: 500)
        #[arg(long)]
        batch_size: Option<u32>,
    },

    /// Query narrative nodes linked to a code symbol or concept
    #[command(
        long_about = "Traverse the narrative graph to find documentation, comments, and \
            explanatory content linked to a code symbol or concept. Results are grouped \
            by edge type (DESCRIBES, EXPLAINS, COVERS_TOPIC, etc.).",
        after_long_help = "Examples:\n  \
            wqm graph narrative --symbol validate_token --tenant proj-abc123\n  \
            wqm graph narrative --concept authentication --tenant proj-abc123 --depth 3\n  \
            wqm graph narrative --symbol parse --tenant t1 --edge-type DESCRIBES,EXPLAINS\n  \
            wqm graph narrative --symbol main --tenant t1 --json"
    )]
    Narrative {
        /// Code symbol name to find narrative for
        #[arg(long, group = "target")]
        symbol: Option<String>,

        /// Concept node name to traverse from
        #[arg(long, group = "target")]
        concept: Option<String>,

        /// Project name or tenant id (partial input resolved)
        #[arg(long)]
        tenant: String,

        /// Maximum traversal depth (1-5, default: 2)
        #[arg(long, default_value = "2")]
        depth: i32,

        /// Maximum number of results (1-200, default: 50)
        #[arg(long, default_value = "50")]
        limit: i32,

        /// Edge type filter (comma-separated, e.g. DESCRIBES,EXPLAINS)
        #[arg(long, value_delimiter = ',')]
        edge_type: Vec<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List concept nodes with IMPLEMENTS_CONCEPT and COVERS_TOPIC counts
    #[command(
        long_about = "List concept nodes from the graph, showing how many code symbols \
            implement each concept (IMPLEMENTS_CONCEPT edges) and how many narrative \
            sources cover it (COVERS_TOPIC edges). Reads directly from SQLite.",
        after_long_help = "Examples:\n  \
            wqm graph concepts --tenant proj-abc123\n  \
            wqm graph concepts --tenant proj-abc123 --concept async\n  \
            wqm graph concepts --tenant proj-abc123 --depth rigorous --top 10\n  \
            wqm graph concepts --tenant proj-abc123 --json"
    )]
    Concepts {
        /// Project name or tenant id (partial input resolved)
        #[arg(long)]
        tenant: String,

        /// Filter by concept name (substring match)
        #[arg(long)]
        concept: Option<String>,

        /// Filter by depth level (qualitative, introductory, intermediate, rigorous, reference)
        #[arg(long)]
        depth: Option<String>,

        /// Maximum number of results
        #[arg(long, default_value = "20")]
        top: u32,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Show topic coverage for a concept, grouped by depth level
    #[command(
        long_about = "Show which narrative sources (documents, sections) cover a given \
            concept, grouped by depth level (qualitative through reference). Uses the \
            NarrativeQuery gRPC RPC with concept_name target.",
        after_long_help = "Examples:\n  \
            wqm graph topics --concept async-runtime\n  \
            wqm graph topics --concept error-handling --tenant proj-abc123\n  \
            wqm graph topics --concept authentication --json"
    )]
    Topics {
        /// Concept name to query coverage for
        #[arg(long)]
        concept: String,

        /// Project name or tenant id (partial input resolved; auto-detected from CWD if omitted)
        #[arg(long)]
        tenant: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}

pub async fn execute(args: GraphArgs) -> Result<()> {
    // Tenant args accept a project name (or partial name/id) — resolve to the
    // canonical tenant id before dispatch; ambiguity is a listed error.
    use crate::data::tenants::resolve_tenant;
    let resolve_opt =
        |t: Option<String>| -> Result<Option<String>> { t.map(|t| resolve_tenant(&t)).transpose() };

    match args.command {
        GraphCommand::Query {
            node_id,
            tenant,
            hops,
            edge_types,
        } => query::query_related(&node_id, &resolve_tenant(&tenant)?, hops, edge_types).await,
        GraphCommand::Impact {
            symbol,
            tenant,
            file,
        } => impact::impact_analysis(&symbol, &resolve_tenant(&tenant)?, file).await,
        GraphCommand::Stats { tenant } => stats::graph_stats(resolve_opt(tenant)?).await,
        GraphCommand::Pagerank {
            tenant,
            damping,
            max_iterations,
            tolerance,
            top_k,
            edge_types,
        } => {
            pagerank::pagerank(
                &resolve_tenant(&tenant)?,
                damping,
                max_iterations,
                tolerance,
                top_k,
                edge_types,
            )
            .await
        }
        GraphCommand::Communities {
            tenant,
            max_iterations,
            min_size,
            edge_types,
        } => {
            communities::communities(
                &resolve_tenant(&tenant)?,
                max_iterations,
                min_size,
                edge_types,
            )
            .await
        }
        GraphCommand::Betweenness {
            tenant,
            top_k,
            max_samples,
            edge_types,
        } => {
            betweenness::betweenness(&resolve_tenant(&tenant)?, top_k, max_samples, edge_types)
                .await
        }
        GraphCommand::Migrate {
            from,
            to,
            tenant,
            batch_size,
        } => migrate::migrate(&from, &to, resolve_opt(tenant)?, batch_size).await,
        GraphCommand::Narrative {
            symbol,
            concept,
            tenant,
            depth,
            limit,
            edge_type,
            json,
        } => {
            let target = match (symbol, concept) {
                (Some(s), _) => narrative::Target::Symbol(s),
                (_, Some(c)) => narrative::Target::Concept(c),
                (None, None) => {
                    anyhow::bail!(
                        "Either --symbol or --concept is required. \
                         Run `wqm graph narrative --help` for usage."
                    );
                }
            };
            narrative::narrative_query(
                target,
                &resolve_tenant(&tenant)?,
                depth,
                limit,
                edge_type,
                json,
            )
            .await
        }
        GraphCommand::Concepts {
            tenant,
            concept,
            depth,
            top,
            json,
        } => {
            concepts::concepts(
                &resolve_tenant(&tenant)?,
                concept.as_deref(),
                depth.as_deref(),
                top,
                json,
            )
            .await
        }
        GraphCommand::Topics {
            concept,
            tenant,
            json,
        } => topics::topics(&concept, resolve_opt(tenant)?.as_deref(), json).await,
    }
}
