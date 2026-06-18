//! gRPC handler implementations for GraphService.
//!
//! Analytics handlers (PageRank, communities, betweenness) are in
//! `analytics_handlers.rs`.

use tonic::{Request, Response, Status};
use tracing::{debug, error, info};
use workspace_qdrant_core::graph::EdgeType;

use crate::proto::{
    graph_service_server::GraphService, narrative_query_request::QueryTarget, BetweennessRequest,
    BetweennessResponse, CommunityRequest, CommunityResponse, FindPathRequest, FindPathResponse,
    GraphMigrateRequest, GraphMigrateResponse, GraphStatsRequest, GraphStatsResponse,
    ImpactAnalysisRequest, ImpactAnalysisResponse, ImpactNodeProto, NarrativeQueryRequest,
    NarrativeQueryResponse, PageRankRequest, PageRankResponse, QueryCrossBoundaryRequest,
    QueryCrossBoundaryResponse, QueryRelatedRequest, QueryRelatedResponse, TraversalNodeProto,
};
use crate::validation::extract_relative_path;

use super::helpers::parse_edge_type_filter;
use super::narrative_query::execute_narrative_query;
use super::service_impl::GraphServiceImpl;

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
                        tenant_id: n.tenant_id,
                        edge_confidence: n.edge_confidence,
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
        self.handle_page_rank(request.into_inner()).await
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.detect_communities"))]
    async fn detect_communities(
        &self,
        request: Request<CommunityRequest>,
    ) -> Result<Response<CommunityResponse>, Status> {
        self.handle_detect_communities(request.into_inner()).await
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.compute_betweenness"))]
    async fn compute_betweenness(
        &self,
        request: Request<BetweennessRequest>,
    ) -> Result<Response<BetweennessResponse>, Status> {
        self.handle_compute_betweenness(request.into_inner()).await
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

        let store_clone = {
            let sqlite = self.sqlite_store.as_ref().ok_or_else(|| {
                Status::unimplemented("MigrateGraph is only supported on the SQLite backend")
            })?;
            let guard = sqlite.read().await.map_err(|e| {
                error!("Failed to acquire graph read lock: {}", e);
                Status::unavailable(format!("Graph store busy: {}", e))
            })?;
            (*guard).clone()
        };

        // Export from SQLite
        let snapshot =
            workspace_qdrant_core::graph::migrator::export_sqlite(store_clone.pool(), tenant_id)
                .await
                .map_err(|e| {
                    error!("Migration export failed: {}", e);
                    Status::internal(format!("Export failed: {}", e))
                })?;

        let report = workspace_qdrant_core::graph::migrator::import_to_store(
            &snapshot,
            &store_clone,
            batch_size,
        )
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
                        tenant_id: n.tenant_id,
                        edge_confidence: n.edge_confidence,
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

    #[tracing::instrument(skip_all, fields(method = "GraphService.narrative_query"))]
    async fn narrative_query(
        &self,
        request: Request<NarrativeQueryRequest>,
    ) -> Result<Response<NarrativeQueryResponse>, Status> {
        let req = request.into_inner();

        if req.tenant_id.is_empty() {
            return Err(Status::invalid_argument("tenant_id is required"));
        }

        let query_target = req
            .query_target
            .ok_or_else(|| Status::invalid_argument("query_target is required"))?;

        let (query_name, is_concept) = match &query_target {
            QueryTarget::SymbolName(s) if s.is_empty() => {
                return Err(Status::invalid_argument("symbol_name must not be empty"));
            }
            QueryTarget::ConceptName(s) if s.is_empty() => {
                return Err(Status::invalid_argument("concept_name must not be empty"));
            }
            QueryTarget::SymbolName(s) => (s.clone(), false),
            QueryTarget::ConceptName(s) => (s.clone(), true),
        };

        // Clamp max_depth: 0 or unset -> 2, negative -> InvalidArgument
        if req.max_depth < 0 {
            return Err(Status::invalid_argument("max_depth must not be negative"));
        }
        let max_depth = if req.max_depth == 0 {
            2
        } else {
            req.max_depth.clamp(1, 5)
        } as u32;

        // Clamp max_results: 0 or unset -> 50, negative -> InvalidArgument
        if req.max_results < 0 {
            return Err(Status::invalid_argument("max_results must not be negative"));
        }
        let max_results = if req.max_results == 0 {
            50
        } else {
            req.max_results.clamp(1, 200)
        } as u32;

        // Validate edge type filters
        let edge_filter = parse_edge_type_filter(&req.edge_types)?;

        debug!(
            "GraphService.NarrativeQuery: tenant={} target={} concept={} \
             depth={} limit={} edge_types={:?}",
            req.tenant_id, query_name, is_concept, max_depth, max_results, edge_filter
        );

        let start = std::time::Instant::now();

        let pool = {
            let sqlite = self.sqlite_store.as_ref().ok_or_else(|| {
                Status::unimplemented("NarrativeQuery is not supported on non-SQLite backends")
            })?;
            let guard = sqlite.read().await.map_err(|e| {
                error!("Failed to acquire graph read lock: {}", e);
                Status::unavailable(format!("Graph store busy: {}", e))
            })?;
            guard.pool().clone()
        };

        let nodes = execute_narrative_query(
            &pool,
            &req.tenant_id,
            &query_name,
            is_concept,
            edge_filter.as_deref(),
            max_depth,
            max_results,
        )
        .await?;

        let elapsed = start.elapsed().as_millis();
        let total_found = nodes.len() as i32;

        debug!(
            "GraphService.NarrativeQuery: found {} nodes in {}ms",
            total_found, elapsed
        );

        Ok(Response::new(NarrativeQueryResponse { nodes, total_found }))
    }

    #[tracing::instrument(skip_all, fields(method = "GraphService.query_cross_boundary"))]
    async fn query_cross_boundary(
        &self,
        request: Request<QueryCrossBoundaryRequest>,
    ) -> Result<Response<QueryCrossBoundaryResponse>, Status> {
        let req = request.into_inner();

        if req.source_tenant.is_empty() {
            return Err(Status::invalid_argument("source_tenant is required"));
        }
        if req.source_node_id.is_empty() {
            return Err(Status::invalid_argument("source_node_id is required"));
        }

        // Hard cap 3, default 2. Reject 0 and >=4 explicitly.
        if req.max_hops == 0 || req.max_hops > 3 {
            return Err(Status::invalid_argument("max_hops must be between 1 and 3"));
        }
        let max_hops = req.max_hops;

        // `two_sided_confidence` is a precision gate over the fused candidate
        // set (require both the code-side IMPLEMENTS_CONCEPT and the doc-side
        // COVERS_TOPIC to clear the confidence floor). It is applied at the MCP
        // fusion layer, which sees the vector candidates alongside the graph
        // expansion; the daemon traversal returns the full bidirectional set and
        // carries per-edge `edge_confidence` so the fusion layer can enforce it.
        let _two_sided_confidence = req.two_sided_confidence;

        // Edge types must parse; an empty list yields no traversal.
        let mut edge_types = Vec::with_capacity(req.edge_types.len());
        for t in &req.edge_types {
            match EdgeType::from_str(t) {
                Some(et) => edge_types.push(et),
                None => {
                    return Err(Status::invalid_argument(format!(
                        "unknown edge type: {}",
                        t
                    )));
                }
            }
        }

        debug!(
            "GraphService.QueryCrossBoundary: tenant={} node={} hops={} \
             edge_types={:?} library_tenants={:?} two_sided={}",
            req.source_tenant,
            req.source_node_id,
            max_hops,
            edge_types,
            req.library_tenants,
            req.two_sided_confidence
        );

        let start = std::time::Instant::now();

        let result = self
            .graph_store
            .query_cross_boundary(
                &req.source_tenant,
                &req.source_node_id,
                &edge_types,
                max_hops,
                &req.library_tenants,
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
                        tenant_id: n.tenant_id,
                        edge_confidence: n.edge_confidence,
                    })
                    .collect();

                Ok(Response::new(QueryCrossBoundaryResponse {
                    nodes: proto_nodes,
                    total,
                    query_time_ms,
                }))
            }
            Err(e) => {
                error!("GraphService.QueryCrossBoundary failed: {}", e);
                Err(Status::internal(format!(
                    "Cross-boundary query failed: {}",
                    e
                )))
            }
        }
    }
}
