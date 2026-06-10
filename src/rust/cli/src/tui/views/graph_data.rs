//! Data types and background fetcher for the TUI Graph view.
//!
//! Located at: `src/rust/cli/src/tui/views/graph_data.rs`
//!
//! # Architecture context
//! The TUI render/event loop is synchronous. Graph data requires async gRPC
//! calls to the daemon's GraphService. This module owns the background thread
//! (tokio single-threaded runtime) that makes those calls, and exposes a
//! shared `Arc<Mutex<GraphSnapshot>>` that the sync view reads on each draw.
//!
//! Mirrors the pattern established in `service_data.rs` (background prober)
//! and uses the same gRPC types as `commands/graph/` subcommands.

use std::sync::{Arc, Mutex};
use std::thread;

use crate::data::db::connect_readonly;
use crate::data::tenants::load_tenants;
use crate::grpc::client::workspace_daemon::{
    BetweennessRequest, CommunityRequest, GraphStatsRequest, ImpactAnalysisRequest, PageRankRequest,
};

// ─── Display interval ────────────────────────────────────────────────────────

/// How long to wait between automatic Stats refreshes.
pub const REFRESH_INTERVAL_SECS: u64 = 15;

// ─── Data models ─────────────────────────────────────────────────────────────

/// High-level graph statistics for the Stats mode panel.
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub total_nodes: u64,
    pub total_edges: u64,
    /// Node type → count, sorted descending by count.
    pub nodes_by_type: Vec<(String, u64)>,
    /// Edge type → count, sorted descending by count.
    pub edges_by_type: Vec<(String, u64)>,
    /// Wall-clock milliseconds the RPC took (informational).
    pub query_time_ms: u64,
}

/// A single node entry for the PageRank or Betweenness list modes.
#[derive(Debug, Clone)]
pub struct RankedSymbol {
    pub symbol: String,
    pub symbol_type: String,
    pub score: f64,
    pub file_path: String,
}

/// A detected community (cluster of related symbols).
#[derive(Debug, Clone)]
pub struct Community {
    pub id: u32,
    pub member_count: usize,
    /// Individual member symbols (name, type, file path).
    pub members: Vec<(String, String, String)>,
}

/// The result of an impact-analysis query on a specific symbol.
#[derive(Debug, Clone, Default)]
pub struct ImpactResult {
    /// Directly affected symbols (distance == 1).
    pub direct: Vec<(String, String)>,
    /// Transitively affected symbols (distance > 1, with distance).
    pub transitive: Vec<(String, String, u32)>,
    /// Unique affected file paths.
    pub affected_files: Vec<String>,
    /// Total impacted node count as reported by the daemon.
    pub total_impacted: u64,
    pub query_time_ms: u64,
}

/// Registered tenant (project) entry for the tenant cycler.
#[derive(Debug, Clone)]
pub struct TenantRef {
    pub tenant_id: String,
    /// Display name (path basename), used in the top-bar.
    pub name: String,
}

// ─── Snapshot ────────────────────────────────────────────────────────────────

/// The full data snapshot written by the background thread and read by the view.
///
/// Tenant cycling state lives in `GraphView` (the view's own `tenants` Vec),
/// not here, because the view needs to cycle tenants synchronously without
/// waiting for a background fetch cycle.
#[derive(Debug, Clone, Default)]
pub struct GraphSnapshot {
    /// Graph statistics for the selected tenant (Stats mode).
    pub stats: Option<GraphStats>,
    /// PageRank results for the selected tenant.
    pub pagerank: Vec<RankedSymbol>,
    /// Communities for the selected tenant.
    pub communities: Vec<Community>,
    /// Betweenness centrality results for the selected tenant.
    pub betweenness: Vec<RankedSymbol>,
    /// Last impact query result (None until `i` is used).
    pub impact: Option<ImpactResult>,
    /// Human-readable error from the last RPC attempt (cleared on next success).
    pub last_error: Option<String>,
    /// Whether a fetch is currently in progress.
    pub loading: bool,
}

// ─── Fetch request ───────────────────────────────────────────────────────────

/// A request sent from the view to the background thread.
#[derive(Debug, Clone)]
pub enum FetchRequest {
    /// Reload Stats, PageRank, Communities, and Betweenness for this tenant.
    Reload { tenant_id: String },
    /// Run impact analysis for a symbol.
    Impact { tenant_id: String, symbol: String },
}

// ─── Background fetcher ──────────────────────────────────────────────────────

/// Spawn the background graph-data fetcher. Returns:
/// - A shared snapshot (view reads this every draw).
/// - A channel sender (view sends `FetchRequest`s to trigger fetches).
pub fn spawn_graph_fetcher() -> (
    Arc<Mutex<GraphSnapshot>>,
    std::sync::mpsc::Sender<FetchRequest>,
) {
    let shared = Arc::new(Mutex::new(GraphSnapshot::default()));
    let shared_clone = Arc::clone(&shared);
    let (tx, rx) = std::sync::mpsc::channel::<FetchRequest>();

    thread::spawn(move || {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(_) => return,
        };

        rt.block_on(async move {
            // Process requests until the sender (view) is dropped.
            while let Ok(req) = rx.recv() {
                // Signal loading to the view.
                if let Ok(mut g) = shared_clone.lock() {
                    g.loading = true;
                    g.last_error = None;
                }

                match req {
                    FetchRequest::Reload { tenant_id } => {
                        let result = fetch_graph_data(&tenant_id).await;
                        if let Ok(mut g) = shared_clone.lock() {
                            match result {
                                Ok((stats, pagerank, communities, betweenness)) => {
                                    g.stats = Some(stats);
                                    g.pagerank = pagerank;
                                    g.communities = communities;
                                    g.betweenness = betweenness;
                                    g.last_error = None;
                                }
                                Err(e) => {
                                    g.last_error = Some(e.to_string());
                                }
                            }
                            g.loading = false;
                        }
                    }
                    FetchRequest::Impact { tenant_id, symbol } => {
                        let result = fetch_impact(&tenant_id, &symbol).await;
                        if let Ok(mut g) = shared_clone.lock() {
                            match result {
                                Ok(impact) => {
                                    g.impact = Some(impact);
                                    g.last_error = None;
                                }
                                Err(e) => {
                                    g.last_error = Some(e.to_string());
                                }
                            }
                            g.loading = false;
                        }
                    }
                }
            }
        });
    });

    (shared, tx)
}

// ─── Async gRPC helpers ──────────────────────────────────────────────────────

/// Fetch all non-impact graph data for one tenant in a single daemon round-trip
/// sequence (Stats, then PageRank, then Communities, then Betweenness).
///
/// Returns `Err` if the daemon connection fails; partial failures per-RPC are
/// reported as empty results rather than propagating so the view degrades
/// gracefully.
async fn fetch_graph_data(
    tenant_id: &str,
) -> anyhow::Result<(
    GraphStats,
    Vec<RankedSymbol>,
    Vec<Community>,
    Vec<RankedSymbol>,
)> {
    let mut client = crate::grpc::connect_default().await?;

    // Stats ─────────────────────────────────────────────────────────────────
    let stats = match client
        .graph()
        .get_graph_stats(GraphStatsRequest {
            tenant_id: Some(tenant_id.to_string()),
            branch: None,
        })
        .await
    {
        Ok(r) => {
            let r = r.into_inner();
            let mut nodes: Vec<(String, u64)> = r
                .nodes_by_type
                .into_iter()
                .map(|(k, v)| (k, v as u64))
                .collect();
            nodes.sort_by_key(|a| std::cmp::Reverse(a.1));
            let mut edges: Vec<(String, u64)> = r
                .edges_by_type
                .into_iter()
                .map(|(k, v)| (k, v as u64))
                .collect();
            edges.sort_by_key(|a| std::cmp::Reverse(a.1));
            GraphStats {
                total_nodes: r.total_nodes as u64,
                total_edges: r.total_edges as u64,
                nodes_by_type: nodes,
                edges_by_type: edges,
                query_time_ms: 0,
            }
        }
        Err(_) => GraphStats::default(),
    };

    // PageRank ──────────────────────────────────────────────────────────────
    let pagerank = match client
        .graph()
        .compute_page_rank(PageRankRequest {
            tenant_id: tenant_id.to_string(),
            damping: None,
            max_iterations: None,
            tolerance: None,
            edge_types: vec![],
            top_k: Some(50),
        })
        .await
    {
        Ok(r) => r
            .into_inner()
            .entries
            .into_iter()
            .map(|e| RankedSymbol {
                symbol: e.symbol_name,
                symbol_type: e.symbol_type,
                score: e.score,
                file_path: e.file_path,
            })
            .collect(),
        Err(_) => vec![],
    };

    // Communities ───────────────────────────────────────────────────────────
    let communities = match client
        .graph()
        .detect_communities(CommunityRequest {
            tenant_id: tenant_id.to_string(),
            max_iterations: None,
            min_community_size: Some(2),
            edge_types: vec![],
        })
        .await
    {
        Ok(r) => r
            .into_inner()
            .communities
            .into_iter()
            .map(|c| {
                let members: Vec<(String, String, String)> = c
                    .members
                    .into_iter()
                    .map(|m| (m.symbol_name, m.symbol_type, m.file_path))
                    .collect();
                Community {
                    id: c.community_id,
                    member_count: members.len(),
                    members,
                }
            })
            .collect(),
        Err(_) => vec![],
    };

    // Betweenness ───────────────────────────────────────────────────────────
    let betweenness = match client
        .graph()
        .compute_betweenness(BetweennessRequest {
            tenant_id: tenant_id.to_string(),
            edge_types: vec![],
            max_samples: None,
            top_k: Some(50),
        })
        .await
    {
        Ok(r) => r
            .into_inner()
            .entries
            .into_iter()
            .map(|e| RankedSymbol {
                symbol: e.symbol_name,
                symbol_type: e.symbol_type,
                score: e.score,
                file_path: e.file_path,
            })
            .collect(),
        Err(_) => vec![],
    };

    Ok((stats, pagerank, communities, betweenness))
}

/// Run an impact analysis RPC and return a structured `ImpactResult`.
async fn fetch_impact(tenant_id: &str, symbol: &str) -> anyhow::Result<ImpactResult> {
    let mut client = crate::grpc::connect_default().await?;

    let resp = client
        .graph()
        .impact_analysis(ImpactAnalysisRequest {
            tenant_id: tenant_id.to_string(),
            symbol_name: symbol.to_string(),
            file_path: None,
            branch: None,
        })
        .await?
        .into_inner();

    let mut direct = Vec::new();
    let mut transitive = Vec::new();
    let mut file_set = std::collections::HashSet::new();

    for node in &resp.impacted_nodes {
        file_set.insert(node.file_path.clone());
        if node.distance <= 1 {
            direct.push((node.symbol_name.clone(), node.file_path.clone()));
        } else {
            transitive.push((
                node.symbol_name.clone(),
                node.file_path.clone(),
                node.distance,
            ));
        }
    }

    let mut affected_files: Vec<String> = file_set.into_iter().collect();
    affected_files.sort();

    Ok(ImpactResult {
        direct,
        transitive,
        affected_files,
        total_impacted: resp.total_impacted as u64,
        query_time_ms: resp.query_time_ms as u64,
    })
}

// ─── SQLite tenant list ───────────────────────────────────────────────────────

/// Load registered tenants from the local SQLite state database.
///
/// Returns an empty list (rather than an error) when the database is
/// unavailable — the view gracefully shows "no tenants" in that case.
pub fn load_tenant_list() -> Vec<TenantRef> {
    let Ok(conn) = connect_readonly() else {
        return vec![];
    };
    let Ok(entries) = load_tenants(&conn) else {
        return vec![];
    };
    // Deduplicate by tenant_id (load_tenants returns one row per watch folder,
    // including sub-folders, so we take the first occurrence of each id).
    let mut seen = std::collections::HashSet::new();
    entries
        .into_iter()
        .filter(|e| seen.insert(e.tenant_id.clone()))
        .map(|e| TenantRef {
            name: e.name,
            tenant_id: e.tenant_id,
        })
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_stats_default_is_zero() {
        let s = GraphStats::default();
        assert_eq!(s.total_nodes, 0);
        assert_eq!(s.total_edges, 0);
        assert!(s.nodes_by_type.is_empty());
        assert!(s.edges_by_type.is_empty());
    }

    #[test]
    fn ranked_symbol_fields() {
        let r = RankedSymbol {
            symbol: "parse".into(),
            symbol_type: "fn".into(),
            score: 0.42,
            file_path: "src/lib.rs".into(),
        };
        assert_eq!(r.symbol, "parse");
        assert!((r.score - 0.42).abs() < 1e-9);
    }

    #[test]
    fn community_member_count_matches_vec() {
        let c = Community {
            id: 1,
            member_count: 3,
            members: vec![
                ("a".into(), "fn".into(), "f.rs".into()),
                ("b".into(), "fn".into(), "f.rs".into()),
                ("c".into(), "struct".into(), "g.rs".into()),
            ],
        };
        assert_eq!(c.member_count, c.members.len());
    }

    #[test]
    fn impact_result_default_is_empty() {
        let r = ImpactResult::default();
        assert!(r.direct.is_empty());
        assert!(r.transitive.is_empty());
        assert!(r.affected_files.is_empty());
        assert_eq!(r.total_impacted, 0);
    }

    #[test]
    fn tenant_ref_fields() {
        let t = TenantRef {
            tenant_id: "abc123".into(),
            name: "my-project".into(),
        };
        assert_eq!(t.name, "my-project");
        assert_eq!(t.tenant_id, "abc123");
    }

    #[test]
    fn graph_snapshot_default_is_empty() {
        let s = GraphSnapshot::default();
        assert!(s.stats.is_none());
        assert!(s.pagerank.is_empty());
        assert!(s.communities.is_empty());
        assert!(s.betweenness.is_empty());
        assert!(s.impact.is_none());
        assert!(s.last_error.is_none());
        assert!(!s.loading);
    }

    #[test]
    fn load_tenant_list_does_not_panic_without_db() {
        // Without a real database this returns an empty list rather than panicking.
        // (The WQM_DATA_DIR env var can be set to a non-existent path to ensure
        // no accidental connection to the user's live database during tests.)
        let _tenants = load_tenant_list();
    }

    #[test]
    fn fetch_request_reload_variant() {
        let req = FetchRequest::Reload {
            tenant_id: "t1".into(),
        };
        matches!(req, FetchRequest::Reload { .. });
    }

    #[test]
    fn fetch_request_impact_variant() {
        let req = FetchRequest::Impact {
            tenant_id: "t1".into(),
            symbol: "foo".into(),
        };
        matches!(req, FetchRequest::Impact { .. });
    }
}
