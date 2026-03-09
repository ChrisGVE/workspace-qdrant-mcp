//! gRPC handler implementations for GraphService.

use tonic::{Request, Response, Status};
use tracing::{debug, error, info};
use workspace_qdrant_core::graph::algorithms::{
    compute_betweenness_centrality, compute_pagerank, detect_communities, CommunityConfig,
    PageRankConfig,
};
use workspace_qdrant_core::graph::EdgeType;

use crate::proto::{
    graph_service_server::GraphService, BetweennessNodeProto, BetweennessRequest,
    BetweennessResponse, CommunityMemberProto, CommunityProto, CommunityRequest, CommunityResponse,
    GraphMigrateRequest, GraphMigrateResponse, GraphStatsRequest, GraphStatsResponse,
    ImpactAnalysisRequest, ImpactAnalysisResponse, ImpactNodeProto, PageRankNodeProto,
    PageRankRequest, PageRankResponse, QueryRelatedRequest, QueryRelatedResponse,
    TraversalNodeProto,
};

use super::helpers::parse_edge_type_filter;
use super::service_impl::GraphServiceImpl;

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

        debug!(
            "GraphService.QueryRelated: tenant={} node={} hops={} edge_types={:?}",
            req.tenant_id, req.node_id, max_hops, edge_types
        );

        let start = std::time::Instant::now();

        let result = self
            .graph_store
            .query_related(
                &req.tenant_id,
                &req.node_id,
                max_hops,
                edge_types.as_deref(),
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

        debug!(
            "GraphService.ImpactAnalysis: tenant={} symbol={} file={:?}",
            req.tenant_id, req.symbol_name, req.file_path
        );

        let start = std::time::Instant::now();

        let result = self
            .graph_store
            .impact_analysis(&req.tenant_id, &req.symbol_name, req.file_path.as_deref())
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

        debug!("GraphService.GetGraphStats: tenant={:?}", tenant_filter);

        let start = std::time::Instant::now();

        match self.graph_store.stats(tenant_filter).await {
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

        let config = PageRankConfig {
            damping: req.damping.unwrap_or(0.85),
            max_iterations: req.max_iterations.unwrap_or(100) as usize,
            tolerance: req.tolerance.unwrap_or(1e-6),
        };

        let edge_filter = parse_edge_type_filter(&req.edge_types)?;
        let edge_refs: Option<Vec<&str>> = edge_filter
            .as_ref()
            .map(|v| v.iter().map(|s| s.as_str()).collect());

        debug!(
            "GraphService.ComputePageRank: tenant={} damping={} max_iter={}",
            req.tenant_id, config.damping, config.max_iterations
        );

        let start = std::time::Instant::now();

        let guard = self.graph_store.read().await;
        let pool = guard.pool();

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

        let config = CommunityConfig {
            max_iterations: req.max_iterations.unwrap_or(50) as usize,
            min_community_size: req.min_community_size.unwrap_or(2) as usize,
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

        let guard = self.graph_store.read().await;
        let pool = guard.pool();

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

        let max_samples = req.max_samples.filter(|&v| v > 0).map(|v| v as usize);

        debug!(
            "GraphService.ComputeBetweenness: tenant={} max_samples={:?}",
            req.tenant_id, max_samples
        );

        let start = std::time::Instant::now();

        let guard = self.graph_store.read().await;
        let pool = guard.pool();

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

        // Currently only sqlite→sqlite is supported (ladybug requires feature flag)
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
            "GraphService.MigrateGraph: {} → {} (tenant={:?}, batch={})",
            req.from_backend, req.to_backend, tenant_id, batch_size
        );

        let guard = self.graph_store.read().await;
        let pool = guard.pool();

        // Export from SQLite
        let snapshot = workspace_qdrant_core::graph::migrator::export_sqlite(pool, tenant_id)
            .await
            .map_err(|e| {
                error!("Migration export failed: {}", e);
                Status::internal(format!("Export failed: {}", e))
            })?;

        // For now, import back to the same SQLite store (real ladybug migration
        // requires runtime construction of the ladybug store which needs the
        // graph config from daemon state — future enhancement)
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
}
