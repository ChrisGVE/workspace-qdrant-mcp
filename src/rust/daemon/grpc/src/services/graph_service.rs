//! GraphService gRPC implementation
//!
//! Provides code relationship graph queries: traversal, impact analysis,
//! and statistics. All queries use a shared read lock on the graph store.

use tonic::{Request, Response, Status};
use tracing::{debug, error};
use workspace_qdrant_core::graph::{EdgeType, SharedGraphStore, SqliteGraphStore};

use crate::proto::{
    graph_service_server::GraphService, GraphStatsRequest, GraphStatsResponse,
    ImpactAnalysisRequest, ImpactAnalysisResponse, ImpactNodeProto, QueryRelatedRequest,
    QueryRelatedResponse, TraversalNodeProto,
};

/// GraphService implementation backed by SharedGraphStore.
pub struct GraphServiceImpl {
    graph_store: SharedGraphStore<SqliteGraphStore>,
}

impl GraphServiceImpl {
    /// Create a new GraphService with a shared graph store handle.
    pub fn new(graph_store: SharedGraphStore<SqliteGraphStore>) -> Self {
        Self { graph_store }
    }
}

#[tonic::async_trait]
impl GraphService for GraphServiceImpl {
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
            .impact_analysis(
                &req.tenant_id,
                &req.symbol_name,
                req.file_path.as_deref(),
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
                Err(Status::internal(format!(
                    "Impact analysis failed: {}",
                    e
                )))
            }
        }
    }

    async fn get_graph_stats(
        &self,
        request: Request<GraphStatsRequest>,
    ) -> Result<Response<GraphStatsResponse>, Status> {
        let req = request.into_inner();

        let tenant_filter = req.tenant_id.as_deref().filter(|s| !s.is_empty());

        debug!(
            "GraphService.GetGraphStats: tenant={:?}",
            tenant_filter
        );

        let start = std::time::Instant::now();

        match self.graph_store.stats(tenant_filter).await {
            Ok(stats) => {
                let query_time_ms = start.elapsed().as_millis();
                debug!("GraphService.GetGraphStats: {} nodes, {} edges in {}ms",
                    stats.total_nodes, stats.total_edges, query_time_ms);

                Ok(Response::new(GraphStatsResponse {
                    total_nodes: stats.total_nodes,
                    total_edges: stats.total_edges,
                    nodes_by_type: stats.nodes_by_type,
                    edges_by_type: stats.edges_by_type,
                }))
            }
            Err(e) => {
                error!("GraphService.GetGraphStats failed: {}", e);
                Err(Status::internal(format!(
                    "Graph stats query failed: {}",
                    e
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_type_parsing() {
        assert!(EdgeType::from_str("CALLS").is_some());
        assert!(EdgeType::from_str("IMPORTS").is_some());
        assert!(EdgeType::from_str("USES_TYPE").is_some());
        assert!(EdgeType::from_str("CONTAINS").is_some());
        assert!(EdgeType::from_str("EXTENDS").is_some());
        assert!(EdgeType::from_str("IMPLEMENTS").is_some());
        assert!(EdgeType::from_str("INVALID").is_none());
    }
}
