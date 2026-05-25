//! gRPC handler implementations for GraphService.

use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};
use workspace_qdrant_core::graph::algorithms::{
    compute_betweenness_centrality, compute_pagerank, detect_communities, CommunityConfig,
    PageRankConfig,
};
use workspace_qdrant_core::graph::EdgeType;

use crate::proto::{
    graph_service_server::GraphService, BetweennessNodeProto, BetweennessRequest,
    BetweennessResponse, CommunityMemberProto, CommunityProto, CommunityRequest, CommunityResponse,
    FindPathRequest, FindPathResponse, GraphMigrateRequest, GraphMigrateResponse,
    GraphStatsRequest, GraphStatsResponse, ImpactAnalysisRequest, ImpactAnalysisResponse,
    ImpactNodeProto, PageRankNodeProto, PageRankRequest, PageRankResponse, QueryRelatedRequest,
    QueryRelatedResponse, TraversalNodeProto,
};
use crate::validation::extract_relative_path;

use super::helpers::parse_edge_type_filter;
use super::service_impl::GraphServiceImpl;
use super::validation::{
    validate_betweenness_params, validate_community_params, validate_pagerank_params,
    DEFAULT_MAX_NODES,
};

/// Extract branch filter from an optional proto string field.
///
/// Returns `None` (cross-branch) when the field is absent or empty.
fn branch_filter(branch: &Option<String>) -> Option<&str> {
    branch.as_deref().filter(|b| !b.is_empty())
}

#[tonic::async_trait]
impl GraphService for GraphServiceImpl {
    #[tracing::instrument(skip_all, fields(method = "GraphService.query_related"))]
    async fn query_related(
        &self,
        request: Request<QueryRelatedRequest>,
    ) -> Result<Response<QueryRelatedResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }
        if req.node_id.is_empty() {
            return Err(Status::invalid_argument("node_id is required"));
        }

        let max_hops = req.max_hops.clamp(0, 5);

        // Parse edge type filters
        let edge_types: Option<Vec<EdgeType>> = if req.edge_types.is_empty() {
            None
        } else {
            let mut types = Vec::with_capacity(req.edge_types.len());
            for t in &req.edge_types {
                match EdgeType::from_str(t) {
                    Some(et) => types.push(et),
                    None => {
                        return Err(Status::invalid_argument(format!(
                            "unknown edge type: {}",
                            t
                        )));
                    }
                }
            }
            Some(types)
        };

        let branch = branch_filter(&req.branch);

        debug!(
            "GraphService.QueryRelated: tenant={} node={} hops={} edge_types={:?} branch={:?}",
            req.tenant_id, req.node_id, max_hops, edge_types, branch
        );

        let start = std::time::Instant::now();

        let result = self
            .graph_store
            .query_related(
                &req.tenant_id,
                &req.node_id,
                max_hops,
                edge_types.as_deref(),
                branch,
            )
            .await;

        match result {
            Ok(nodes) => {
                let query_time_ms = start.elapsed().as_millis() as i64;
                let total = nodes.len() as u32;

                let proto_nodes: Vec<TraversalNodeProto> = nodes
                    .into_iter()
                    .map(|n| TraversalNodeProto {
                        node_id: n.node_id,
                        symbol_name: n.symbol_name,
                        symbol_type: n.symbol_type,
                        file_path: n.file_path,
                        edge_type: n.edge_type,
                        depth: n.depth,
                        path: n.path,
                    })
                    .collect();

                Ok(Response::new(QueryRelatedResponse {
                    nodes: proto_nodes,
                    total,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("GraphService.QueryRelated failed: {}", e);
                Err(Status::internal(format!("Graph query failed: {}", e)))
            }
        }
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.impact_analysis"))]
    async fn impact_analysis(
        &self,
        request: Request<ImpactAnalysisRequest>,
    ) -> Result<Response<ImpactAnalysisResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }
        if req.symbol_name.is_empty() {
            return Err(Status::invalid_argument("symbol_name is required"));
        }

        // Validate optional file_path as RelativePath when present and non-empty.
        let validated_file_path = match req.file_path.as_deref().filter(|p| !p.is_empty()) {
            Some(raw) => {
                let rp = extract_relative_path!(raw.to_string(), "file_path")?;
                Some(rp.into_string())
            }
            None => None,
        };

        let branch = branch_filter(&req.branch);

        debug!(
            "GraphService.ImpactAnalysis: tenant={} symbol={} file={:?} branch={:?}",
            req.tenant_id, req.symbol_name, validated_file_path, branch
        );

        let start = std::time::Instant::now();

        let result = self
            .graph_store
            .impact_analysis(
                &req.tenant_id,
                &req.symbol_name,
                validated_file_path.as_deref(),
                branch,
            )
            .await;

        match result {
            Ok(report) => {
                let query_time_ms = start.elapsed().as_millis() as i64;

                let impacted_nodes: Vec<ImpactNodeProto> = report
                    .impacted_nodes
                    .into_iter()
                    .map(|n| ImpactNodeProto {
                        node_id: n.node_id,
                        symbol_name: n.symbol_name,
                        file_path: n.file_path,
                        impact_type: n.impact_type,
                        distance: n.distance,
                    })
                    .collect();

                let total_impacted = report.total_impacted;

                Ok(Response::new(ImpactAnalysisResponse {
                    impacted_nodes,
                    total_impacted,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("GraphService.ImpactAnalysis failed: {}", e);
                Err(Status::internal(format!("Impact analysis failed: {}", e)))
            }
        }
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.get_graph_stats"))]
    async fn get_graph_stats(
        &self,
        request: Request<GraphStatsRequest>,
    ) -> Result<Response<GraphStatsResponse>, Status> {
        let req = request.into_inner();

        let tenant_filter = req.tenant_id.as_deref().filter(|s| !s.is_empty());
        let branch = branch_filter(&req.branch);

        debug!(
            "GraphService.GetGraphStats: tenant={:?} branch={:?}",
            tenant_filter, branch
        );

        let start = std::time::Instant::now();

        match self.graph_store.stats(tenant_filter, branch).await {
            Ok(stats) => {
                let query_time_ms = start.elapsed().as_millis();
                debug!(
                    "GraphService.GetGraphStats: {} nodes, {} edges in {}ms",
                    stats.total_nodes, stats.total_edges, query_time_ms
                );

                Ok(Response::new(GraphStatsResponse {
                    total_nodes: stats.total_nodes,
                    total_edges: stats.total_edges,
                    nodes_by_type: stats.nodes_by_type,
                    edges_by_type: stats.edges_by_type,
                }))
            }
            Err(e) => {
                error!("GraphService.GetGraphStats failed: {}", e);
                Err(Status::internal(format!("Graph stats query failed: {}", e)))
            }
        }
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.compute_page_rank"))]
    async fn compute_page_rank(
        &self,
        request: Request<PageRankRequest>,
    ) -> Result<Response<PageRankResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        // Validate parameters before algorithm execution
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

        let guard = self.graph_store.read().await.map_err(|e| {
            error!("Failed to acquire graph read lock: {}", e);
            Status::unavailable(format!("Graph store busy: {}", e))
        })?;
        let pool = guard.pool();

        // Check graph size before full materialization
        let node_count = count_tenant_nodes(pool, &req.tenant_id).await?;
        if node_count > DEFAULT_MAX_NODES as u64 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph exceeds materialization limit for PageRank"
            );
            return Err(Status::failed_precondition(format!(
                "Graph has {} nodes, exceeding materialization limit of {}. \
                 Use edge_types filter to reduce scope.",
                node_count, DEFAULT_MAX_NODES
            )));
        }
        if node_count as f64 > DEFAULT_MAX_NODES as f64 * 0.8 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph size approaching materialization limit"
            );
        }

        match compute_pagerank(pool, &req.tenant_id, &config, edge_refs.as_deref()).await {
            Ok(mut entries) => {
                let total = entries.len() as u32;

                // Apply top_k if requested
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

    #[tracing::instrument(skip_all, fields(method = "GraphService.detect_communities"))]
    async fn detect_communities(
        &self,
        request: Request<CommunityRequest>,
    ) -> Result<Response<CommunityResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        // Validate parameters before algorithm execution
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

        let guard = self.graph_store.read().await.map_err(|e| {
            error!("Failed to acquire graph read lock: {}", e);
            Status::unavailable(format!("Graph store busy: {}", e))
        })?;
        let pool = guard.pool();

        // Check graph size before full materialization
        let node_count = count_tenant_nodes(pool, &req.tenant_id).await?;
        if node_count > DEFAULT_MAX_NODES as u64 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph exceeds materialization limit for community detection"
            );
            return Err(Status::failed_precondition(format!(
                "Graph has {} nodes, exceeding materialization limit of {}. \
                 Use edge_types filter to reduce scope.",
                node_count, DEFAULT_MAX_NODES
            )));
        }
        if node_count as f64 > DEFAULT_MAX_NODES as f64 * 0.8 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph size approaching materialization limit"
            );
        }

        match detect_communities(pool, &req.tenant_id, &config, edge_refs.as_deref()).await {
            Ok(communities) => {
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

    #[tracing::instrument(skip_all, fields(method = "GraphService.compute_betweenness"))]
    async fn compute_betweenness(
        &self,
        request: Request<BetweennessRequest>,
    ) -> Result<Response<BetweennessResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let edge_filter = parse_edge_type_filter(&req.edge_types)?;
        let edge_refs: Option<Vec<&str>> = edge_filter
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        // Validate max_samples parameter
        let max_samples = validate_betweenness_params(req.max_samples)?;

        debug!(
            "GraphService.ComputeBetweenness: tenant={} max_samples={:?}",
            req.tenant_id, max_samples
        );

        let start = std::time::Instant::now();

        let guard = self.graph_store.read().await.map_err(|e| {
            error!("Failed to acquire graph read lock: {}", e);
            Status::unavailable(format!("Graph store busy: {}", e))
        })?;
        let pool = guard.pool();

        // Check graph size before full materialization
        let node_count = count_tenant_nodes(pool, &req.tenant_id).await?;
        if node_count > DEFAULT_MAX_NODES as u64 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph exceeds materialization limit for betweenness"
            );
            return Err(Status::failed_precondition(format!(
                "Graph has {} nodes, exceeding materialization limit of {}. \
                 Use edge_types filter or max_samples to reduce scope.",
                node_count, DEFAULT_MAX_NODES
            )));
        }
        if node_count as f64 > DEFAULT_MAX_NODES as f64 * 0.8 {
            warn!(
                tenant_id = %req.tenant_id,
                node_count,
                limit = DEFAULT_MAX_NODES,
                "Graph size approaching materialization limit"
            );
        }

        match compute_betweenness_centrality(
            pool,
            &req.tenant_id,
            edge_refs.as_deref(),
            max_samples,
        )
        .await
        {
            Ok(mut entries) => {
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

    #[tracing::instrument(skip_all, fields(method = "GraphService.migrate_graph"))]
    async fn migrate_graph(
        &self,
        request: Request<GraphMigrateRequest>,
    ) -> Result<Response<GraphMigrateResponse>, Status> {
        let req = request.into_inner();

        // Currently only sqlite->sqlite is supported (ladybug requires feature flag)
        if req.from_backend != "sqlite" {
            return Err(Status::unimplemented(format!(
                "Export from '{}' is not yet supported. Only 'sqlite' is available.",
                req.from_backend
            )));
        }

        if req.to_backend != "sqlite" && req.to_backend != "ladybug" {
            return Err(Status::invalid_argument(format!(
                "Unknown target backend: '{}'. Use 'sqlite' or 'ladybug'.",
                req.to_backend
            )));
        }

        if req.to_backend == "ladybug" {
            return Err(Status::unimplemented(
                "LadybugDB migration via gRPC is not yet implemented. \
                 Use the CLI: wqm graph migrate --from sqlite --to ladybug",
            ));
        }

        let tenant_id = req.tenant_id.as_deref().filter(|s| !s.is_empty());
        let batch_size = req.batch_size.unwrap_or(500) as usize;

        info!(
            "GraphService.MigrateGraph: {} -> {} (tenant={:?}, batch={})",
            req.from_backend, req.to_backend, tenant_id, batch_size
        );

        let guard = self.graph_store.read().await.map_err(|e| {
            error!("Failed to acquire graph read lock: {}", e);
            Status::unavailable(format!("Graph store busy: {}", e))
        })?;
        let pool = guard.pool();

        // Export from SQLite
        let snapshot = workspace_qdrant_core::graph::migrator::export_sqlite(pool, tenant_id)
            .await
            .map_err(|e| {
                error!("Migration export failed: {}", e);
                Status::internal(format!("Export failed: {}", e))
            })?;

        let report =
            workspace_qdrant_core::graph::migrator::import_to_store(&snapshot, &*guard, batch_size)
                .await
                .map_err(|e| {
                    error!("Migration import failed: {}", e);
                    Status::internal(format!("Import failed: {}", e))
                })?;

        Ok(Response::new(GraphMigrateResponse {
            success: report.nodes_match && report.edges_match,
            nodes_exported: report.nodes_exported,
            edges_exported: report.edges_exported,
            nodes_imported: report.nodes_imported,
            edges_imported: report.edges_imported,
            nodes_match: report.nodes_match,
            edges_match: report.edges_match,
            warnings: report.warnings,
        }))
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.find_path"))]
    async fn find_path(
        &self,
        request: Request<FindPathRequest>,
    ) -> Result<Response<FindPathResponse>, Status> {
        let start = std::time::Instant::now();
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }
        if req.source_node_id.is_empty() {
            return Err(Status::invalid_argument("source_node_id is required"));
        }
        if req.target_node_id.is_empty() {
            return Err(Status::invalid_argument("target_node_id is required"));
        }

        let max_depth = req.max_depth.clamp(1, 10);
        let edge_type_strs = parse_edge_type_filter(&req.edge_types)?;
        let edge_types: Option<Vec<EdgeType>> = edge_type_strs
            .as_ref()
            .map(|strs| strs.iter().filter_map(|s| EdgeType::from_str(s)).collect());

        let branch = branch_filter(&req.branch);

        let result = self
            .graph_store
            .find_path(
                &req.tenant_id,
                &req.source_node_id,
                &req.target_node_id,
                max_depth,
                edge_types.as_deref(),
                branch,
            )
            .await
            .map_err(|e| {
                error!("FindPath failed: {}", e);
                Status::internal(format!("FindPath error: {}", e))
            })?;

        let elapsed = start.elapsed().as_millis() as i64;

        match result {
            Some(path) => {
                let path_length = if path.len() > 1 {
                    path.len() as u32 - 1
                } else {
                    0
                };
                let path_nodes: Vec<TraversalNodeProto> = path
                    .into_iter()
                    .map(|n| TraversalNodeProto {
                        node_id: n.node_id,
                        symbol_name: n.symbol_name,
                        symbol_type: n.symbol_type,
                        file_path: n.file_path,
                        edge_type: String::new(),
                        depth: n.depth,
                        path: String::new(),
                    })
                    .collect();

                debug!(path_length, elapsed_ms = elapsed, "FindPath: found path");

                Ok(Response::new(FindPathResponse {
                    path_found: true,
                    path_nodes,
                    path_length,
                    query_time_ms: elapsed,
                }))
            }
            None => {
                debug!(elapsed_ms = elapsed, "FindPath: no path found");
                Ok(Response::new(FindPathResponse {
                    path_found: false,
                    path_nodes: Vec::new(),
                    path_length: 0,
                    query_time_ms: elapsed,
                }))
            }
        }
    }
}

/// Count total nodes for a tenant to enforce materialization limits.
///
/// Uses a lightweight COUNT(*) query instead of fetching all rows.
async fn count_tenant_nodes(pool: &sqlx::SqlitePool, tenant_id: &str) -> Result<u64, Status> {
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
