//! Graph analytics handler implementations (PageRank, communities, betweenness).
//!
//! Each handler calls `SharedGraphStore::export_adjacency` exactly once,
//! releases the read lock (LOCK-SCOPE contract), checks the materialized size,
//! and then runs the algorithm on the owned [`AdjacencyExport`] — no
//! `SqlitePool` reference is held during algorithm execution.
//!
//! Algorithms return topology-only results (empty display fields); each handler
//! then enriches them with `symbol_name` / `symbol_type` / `file_path` via the
//! backend-agnostic `SharedGraphStore::fetch_node_metadata`, keeping the
//! algorithm layer free of any database coupling.

use std::collections::HashMap;

use tonic::{Response, Status};
use tracing::{error, warn};
use workspace_qdrant_core::graph::algorithms::{
    compute_betweenness_centrality, compute_pagerank, detect_communities, CommunityConfig,
    PageRankConfig,
};
use workspace_qdrant_core::graph::{AdjacencyExport, EdgeType, NodeMetadata};

use crate::proto::{
    BetweennessNodeProto, BetweennessRequest, BetweennessResponse, CommunityMemberProto,
    CommunityProto, CommunityRequest, CommunityResponse, PageRankNodeProto, PageRankRequest,
    PageRankResponse,
};

use super::helpers::parse_edge_type_filter;
use super::service_impl::GraphServiceImpl;
use super::validation::{
    validate_betweenness_params, validate_community_params, validate_pagerank_params,
    DEFAULT_MAX_NODES,
};

impl GraphServiceImpl {
    pub(super) async fn handle_page_rank(
        &self,
        req: PageRankRequest,
    ) -> Result<Response<PageRankResponse>, Status> {
        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let (damping, max_iterations, tolerance) =
            validate_pagerank_params(req.damping, req.max_iterations, req.tolerance)?;

        let config = PageRankConfig {
            damping,
            max_iterations,
            tolerance,
        };

        let edge_types = parse_edge_types(&req.edge_types)?;

        tracing::debug!(
            "GraphService.ComputePageRank: tenant={} damping={} max_iter={} tolerance={}",
            req.tenant_id,
            config.damping,
            config.max_iterations,
            config.tolerance
        );

        let start = std::time::Instant::now();

        // Acquire export (lock released on return from export_adjacency).
        let adj = export_adjacency(self, &req.tenant_id, edge_types.as_deref()).await?;

        check_export_size(&adj, DEFAULT_MAX_NODES, "PageRank")?;

        let mut entries = compute_pagerank(&adj, &config);

        workspace_qdrant_core::graph::metrics::record_graph_algorithm_run(
            "pagerank",
            &req.tenant_id,
            start.elapsed().as_secs_f64(),
        );

        let total = entries.len() as u32;

        if let Some(k) = req.top_k {
            if k > 0 && (k as usize) < entries.len() {
                entries.truncate(k as usize);
            }
        }

        let query_time_ms = start.elapsed().as_millis() as i64;

        // Enrich topology-only results with display metadata (backend-agnostic).
        let meta = fetch_metadata(self, &req.tenant_id).await?;
        let proto_entries: Vec<PageRankNodeProto> = entries
            .into_iter()
            .map(|e| {
                let (symbol_name, symbol_type, file_path) = split_meta(&meta, &e.node_id);
                PageRankNodeProto {
                    node_id: e.node_id,
                    symbol_name,
                    symbol_type,
                    file_path,
                    score: e.score,
                }
            })
            .collect();

        Ok(Response::new(PageRankResponse {
            entries: proto_entries,
            total,
            query_time_ms,
        }))
    }

    pub(super) async fn handle_detect_communities(
        &self,
        req: CommunityRequest,
    ) -> Result<Response<CommunityResponse>, Status> {
        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let (max_iterations, min_community_size) =
            validate_community_params(req.max_iterations, req.min_community_size)?;

        let config = CommunityConfig {
            max_iterations,
            min_community_size,
        };

        let edge_types = parse_edge_types(&req.edge_types)?;

        tracing::debug!(
            "GraphService.DetectCommunities: tenant={} max_iter={} min_size={}",
            req.tenant_id,
            config.max_iterations,
            config.min_community_size
        );

        let start = std::time::Instant::now();

        // Acquire export (lock released on return from export_adjacency).
        let adj = export_adjacency(self, &req.tenant_id, edge_types.as_deref()).await?;

        check_export_size(&adj, DEFAULT_MAX_NODES, "community detection")?;

        let communities = detect_communities(&adj, &config);

        workspace_qdrant_core::graph::metrics::record_graph_algorithm_run(
            "community",
            &req.tenant_id,
            start.elapsed().as_secs_f64(),
        );

        let total_communities = communities.len() as u32;
        let query_time_ms = start.elapsed().as_millis() as i64;

        // Enrich topology-only members with display metadata (backend-agnostic).
        let meta = fetch_metadata(self, &req.tenant_id).await?;
        let proto_communities: Vec<CommunityProto> = communities
            .into_iter()
            .map(|c| CommunityProto {
                community_id: c.community_id,
                members: c
                    .members
                    .into_iter()
                    .map(|m| {
                        let (symbol_name, symbol_type, file_path) = split_meta(&meta, &m.node_id);
                        CommunityMemberProto {
                            node_id: m.node_id,
                            symbol_name,
                            symbol_type,
                            file_path,
                        }
                    })
                    .collect(),
            })
            .collect();

        Ok(Response::new(CommunityResponse {
            communities: proto_communities,
            total_communities,
            query_time_ms,
        }))
    }

    pub(super) async fn handle_compute_betweenness(
        &self,
        req: BetweennessRequest,
    ) -> Result<Response<BetweennessResponse>, Status> {
        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let edge_types = parse_edge_types(&req.edge_types)?;
        let max_samples = validate_betweenness_params(req.max_samples)?;

        tracing::debug!(
            "GraphService.ComputeBetweenness: tenant={} max_samples={:?}",
            req.tenant_id,
            max_samples
        );

        let start = std::time::Instant::now();

        // Acquire export (lock released on return from export_adjacency).
        let adj = export_adjacency(self, &req.tenant_id, edge_types.as_deref()).await?;

        check_export_size(&adj, DEFAULT_MAX_NODES, "betweenness")?;

        let mut entries = compute_betweenness_centrality(&adj, max_samples);

        workspace_qdrant_core::graph::metrics::record_graph_algorithm_run(
            "betweenness",
            &req.tenant_id,
            start.elapsed().as_secs_f64(),
        );

        let total = entries.len() as u32;

        if let Some(k) = req.top_k {
            if k > 0 && (k as usize) < entries.len() {
                entries.truncate(k as usize);
            }
        }

        let query_time_ms = start.elapsed().as_millis() as i64;

        // Enrich topology-only results with display metadata (backend-agnostic).
        let meta = fetch_metadata(self, &req.tenant_id).await?;
        let proto_entries: Vec<BetweennessNodeProto> = entries
            .into_iter()
            .map(|e| {
                let (symbol_name, symbol_type, file_path) = split_meta(&meta, &e.node_id);
                BetweennessNodeProto {
                    node_id: e.node_id,
                    symbol_name,
                    symbol_type,
                    file_path,
                    score: e.score,
                }
            })
            .collect();

        Ok(Response::new(BetweennessResponse {
            entries: proto_entries,
            total,
            query_time_ms,
        }))
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Parse proto edge-type strings into typed `EdgeType` values, forwarding
/// validation errors as `Status::invalid_argument`.
///
/// Returns `None` when the list is empty (= all edge types).
fn parse_edge_types(types: &[String]) -> Result<Option<Vec<EdgeType>>, Status> {
    let strs = parse_edge_type_filter(types)?;
    Ok(strs.map(|v| {
        v.iter()
            .filter_map(|s| EdgeType::from_str(s))
            .collect::<Vec<_>>()
    }))
}

/// Call `SharedGraphStore::export_adjacency` and convert `GraphDbError` to
/// `Status`.  The read lock is acquired and released inside this call.
async fn export_adjacency(
    svc: &GraphServiceImpl,
    tenant_id: &str,
    edge_types: Option<&[EdgeType]>,
) -> Result<AdjacencyExport, Status> {
    svc.graph_store
        .export_adjacency(tenant_id, edge_types)
        .await
        .map_err(|e| {
            error!("export_adjacency failed: {}", e);
            Status::internal(format!("export_adjacency failed: {}", e))
        })
}

/// Fetch per-node display metadata for a tenant, keyed by `node_id`.
///
/// Acquires and releases its own read lock (after the algorithm has run); used
/// to enrich topology-only analytics results before building proto responses.
async fn fetch_metadata(
    svc: &GraphServiceImpl,
    tenant_id: &str,
) -> Result<HashMap<String, NodeMetadata>, Status> {
    svc.graph_store
        .fetch_node_metadata(tenant_id)
        .await
        .map_err(|e| {
            error!("fetch_node_metadata failed: {}", e);
            Status::internal(format!("fetch_node_metadata failed: {}", e))
        })
}

/// Look up display fields for a node, returning empty strings when the node has
/// no metadata (e.g. removed between export and enrichment).
fn split_meta(meta: &HashMap<String, NodeMetadata>, node_id: &str) -> (String, String, String) {
    meta.get(node_id)
        .map(|m| {
            (
                m.symbol_name.clone(),
                m.symbol_type.clone(),
                m.file_path.clone(),
            )
        })
        .unwrap_or_default()
}

/// Check that the materialized export does not exceed the node limit.
///
/// Emits a warning when the count exceeds 80 % of the limit.
fn check_export_size(adj: &AdjacencyExport, limit: u32, algorithm: &str) -> Result<(), Status> {
    let node_count = adj.node_ids.len();
    let limit_usize = limit as usize;
    if node_count > limit_usize {
        warn!(
            node_count,
            limit, "Graph exceeds materialization limit for {}", algorithm
        );
        return Err(Status::failed_precondition(format!(
            "Graph has {} nodes, exceeding materialization limit of {}. \
             Use edge_types filter to reduce scope.",
            node_count, limit
        )));
    }
    if node_count as f64 > limit_usize as f64 * 0.8 {
        warn!(
            node_count,
            limit, "Graph size approaching materialization limit"
        );
    }
    Ok(())
}
