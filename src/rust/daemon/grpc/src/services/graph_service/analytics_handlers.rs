//! Graph analytics handler implementations (PageRank, communities, betweenness).
//!
//! Extracted from handlers.rs to keep file sizes within project limits.

use tonic::{Response, Status};
use tracing::{debug, error, warn};
use workspace_qdrant_core::graph::algorithms::{
    compute_betweenness_centrality, compute_pagerank, detect_communities, CommunityConfig,
    PageRankConfig,
};

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

        let edge_filter = parse_edge_type_filter(&req.edge_types)?;
        let edge_refs: Option<Vec<&str>> = edge_filter
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        debug!(
            "GraphService.ComputePageRank: tenant={} damping={} max_iter={} tolerance={}",
            req.tenant_id, config.damping, config.max_iterations, config.tolerance
        );

        let start = std::time::Instant::now();

        let pool = {
            let guard = self.graph_store.read().await.map_err(|e| {
                error!("Failed to acquire graph read lock: {}", e);
                Status::unavailable(format!("Graph store busy: {}", e))
            })?;
            guard.pool().clone()
        };

        check_graph_size(&pool, &req.tenant_id, "PageRank").await?;

        match compute_pagerank(&pool, &req.tenant_id, &config, edge_refs.as_deref()).await {
            Ok(mut entries) => {
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

                let proto_entries: Vec<PageRankNodeProto> = entries
                    .into_iter()
                    .map(|e| PageRankNodeProto {
                        node_id: e.node_id,
                        symbol_name: e.symbol_name,
                        symbol_type: e.symbol_type,
                        file_path: e.file_path,
                        score: e.score,
                    })
                    .collect();

                Ok(Response::new(PageRankResponse {
                    entries: proto_entries,
                    total,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("GraphService.ComputePageRank failed: {}", e);
                Err(Status::internal(format!("PageRank failed: {}", e)))
            }
        }
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

        let edge_filter = parse_edge_type_filter(&req.edge_types)?;
        let edge_refs: Option<Vec<&str>> = edge_filter
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        debug!(
            "GraphService.DetectCommunities: tenant={} max_iter={} min_size={}",
            req.tenant_id, config.max_iterations, config.min_community_size
        );

        let start = std::time::Instant::now();

        let pool = {
            let guard = self.graph_store.read().await.map_err(|e| {
                error!("Failed to acquire graph read lock: {}", e);
                Status::unavailable(format!("Graph store busy: {}", e))
            })?;
            guard.pool().clone()
        };

        check_graph_size(&pool, &req.tenant_id, "community detection").await?;

        match detect_communities(&pool, &req.tenant_id, &config, edge_refs.as_deref()).await {
            Ok(communities) => {
                workspace_qdrant_core::graph::metrics::record_graph_algorithm_run(
                    "community",
                    &req.tenant_id,
                    start.elapsed().as_secs_f64(),
                );
                let total_communities = communities.len() as u32;
                let query_time_ms = start.elapsed().as_millis() as i64;

                let proto_communities: Vec<CommunityProto> = communities
                    .into_iter()
                    .map(|c| CommunityProto {
                        community_id: c.community_id,
                        members: c
                            .members
                            .into_iter()
                            .map(|m| CommunityMemberProto {
                                node_id: m.node_id,
                                symbol_name: m.symbol_name,
                                symbol_type: m.symbol_type,
                                file_path: m.file_path,
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
            Err(e) => {
                error!("GraphService.DetectCommunities failed: {}", e);
                Err(Status::internal(format!(
                    "Community detection failed: {}",
                    e
                )))
            }
        }
    }

    pub(super) async fn handle_compute_betweenness(
        &self,
        req: BetweennessRequest,
    ) -> Result<Response<BetweennessResponse>, Status> {
        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let edge_filter = parse_edge_type_filter(&req.edge_types)?;
        let edge_refs: Option<Vec<&str>> = edge_filter
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        let max_samples = validate_betweenness_params(req.max_samples)?;

        debug!(
            "GraphService.ComputeBetweenness: tenant={} max_samples={:?}",
            req.tenant_id, max_samples
        );

        let start = std::time::Instant::now();

        let pool = {
            let guard = self.graph_store.read().await.map_err(|e| {
                error!("Failed to acquire graph read lock: {}", e);
                Status::unavailable(format!("Graph store busy: {}", e))
            })?;
            guard.pool().clone()
        };

        check_graph_size(&pool, &req.tenant_id, "betweenness").await?;

        match compute_betweenness_centrality(
            &pool,
            &req.tenant_id,
            edge_refs.as_deref(),
            max_samples,
        )
        .await
        {
            Ok(mut entries) => {
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

                let proto_entries: Vec<BetweennessNodeProto> = entries
                    .into_iter()
                    .map(|e| BetweennessNodeProto {
                        node_id: e.node_id,
                        symbol_name: e.symbol_name,
                        symbol_type: e.symbol_type,
                        file_path: e.file_path,
                        score: e.score,
                    })
                    .collect();

                Ok(Response::new(BetweennessResponse {
                    entries: proto_entries,
                    total,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("GraphService.ComputeBetweenness failed: {}", e);
                Err(Status::internal(format!(
                    "Betweenness centrality failed: {}",
                    e
                )))
            }
        }
    }
}

/// Check graph size and return an error if it exceeds the materialization limit.
async fn check_graph_size(
    pool: &sqlx::SqlitePool,
    tenant_id: &str,
    algorithm: &str,
) -> Result<(), Status> {
    let node_count = count_tenant_nodes(pool, tenant_id).await?;
    if node_count > DEFAULT_MAX_NODES as u64 {
        warn!(
            tenant_id = %tenant_id,
            node_count,
            limit = DEFAULT_MAX_NODES,
            "Graph exceeds materialization limit for {}", algorithm
        );
        return Err(Status::failed_precondition(format!(
            "Graph has {} nodes, exceeding materialization limit of {}. \
             Use edge_types filter to reduce scope.",
            node_count, DEFAULT_MAX_NODES
        )));
    }
    if node_count as f64 > DEFAULT_MAX_NODES as f64 * 0.8 {
        warn!(
            tenant_id = %tenant_id,
            node_count,
            limit = DEFAULT_MAX_NODES,
            "Graph size approaching materialization limit"
        );
    }
    Ok(())
}

/// Count total nodes for a tenant to enforce materialization limits.
pub(super) async fn count_tenant_nodes(
    pool: &sqlx::SqlitePool,
    tenant_id: &str,
) -> Result<u64, Status> {
    let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM graph_nodes WHERE tenant_id = ?1")
        .bind(tenant_id)
        .fetch_one(pool)
        .await
        .map_err(|e| {
            error!("Failed to count graph nodes: {}", e);
            Status::internal(format!("Failed to count graph nodes: {}", e))
        })?;
    Ok(row.0 as u64)
}
