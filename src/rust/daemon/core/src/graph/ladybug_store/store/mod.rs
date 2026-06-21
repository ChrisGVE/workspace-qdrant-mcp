//! `ladybug_store/store/mod.rs` — module router and single `GraphStore` impl.
//!
//! Declares the five focused submodules that together implement `GraphStore`
//! for LadybugDB and hosts the single `impl GraphStore for LadybugGraphStore`
//! delegation block. All trait methods delegate to the `pub(super)` inherent
//! methods defined in `mutate` (write path) and `query` (read path).
//!
//! Submodule responsibilities:
//! - `helpers`   — constants, data structures, value extractors, `lbug_call`
//!                panic guard, `tenant_param_list`, `distinct_frontier_tails`.
//! - `init`      — `LadybugGraphStore` struct, constructor, schema DDL, FFI
//!                choke-point methods (`connect`, `exec`, `run_prepared`),
//!                low-level DML primitives (`upsert_node_with_conn`,
//!                `insert_edge_with_conn`), and public Cypher APIs.
//! - `traversal` — private BFS helpers (`cross_boundary_neighbours`,
//!                `find_path_self`, `reconstruct_path`, `fetch_path_node`,
//!                `find_path_neighbours`) used by `path` via `self.*` calls.
//! - `mutate`    — write-path inherent methods (upsert, insert, delete, prune,
//!                stub resolution).
//! - `path`      — BFS-driver methods (`do_find_path`, `do_query_cross_boundary`)
//!                separated from `query` to keep both files under 500 lines.
//! - `query`     — simpler read-path methods (query_related, impact_analysis,
//!                stats, graph_tenants, code_symbols, metadata, export).

pub(super) mod helpers;
pub(super) mod init;
pub(super) mod mutate;
pub(super) mod path;
pub(super) mod query;
pub(super) mod traversal;

pub use init::LadybugGraphStore;

// ---- Single GraphStore trait implementation ----------------------------------
//
// Rust permits only one `impl Trait for Type` per crate. The write-path methods
// live in `mutate` and read-path methods in `query`, each as `pub(super)`
// inherent methods on `LadybugGraphStore` with a `do_` prefix. This block
// is the sole trait impl: it is a thin delegation layer with no logic of its
// own.

use async_trait::async_trait;

use crate::graph::{
    schema::GraphDbResult, AdjacencyExport, EdgeType, GraphEdge, GraphNode, GraphStats, GraphStore,
    ImpactReport, NodeMetadata, SymbolRow, TraversalNode,
};

#[async_trait]
impl GraphStore for LadybugGraphStore {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        self.do_upsert_node(node).await
    }

    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        self.do_upsert_nodes(nodes).await
    }

    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        self.do_insert_edge(edge).await
    }

    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        self.do_insert_edges(edges).await
    }

    async fn delete_edges_by_file(&self, tenant_id: &str, file_path: &str) -> GraphDbResult<u64> {
        self.do_delete_edges_by_file(tenant_id, file_path).await
    }

    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        self.do_delete_tenant(tenant_id).await
    }

    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        self.do_query_related(tenant_id, node_id, max_hops, edge_types, branch)
            .await
    }

    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        self.do_impact_analysis(tenant_id, symbol_name, file_path, branch)
            .await
    }

    async fn find_path(
        &self,
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        self.do_find_path(
            tenant_id, source_id, target_id, max_depth, edge_types, branch,
        )
        .await
    }

    async fn stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        self.do_stats(tenant_id, branch).await
    }

    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        self.do_prune_orphans(tenant_id).await
    }

    async fn graph_tenants(&self) -> GraphDbResult<Vec<String>> {
        self.do_graph_tenants().await
    }

    async fn resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        self.do_resolve_stub_edges(tenant_id).await
    }

    async fn resolve_all_stub_edges(&self) -> GraphDbResult<u64> {
        self.do_resolve_all_stub_edges().await
    }

    async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        self.do_query_code_symbols(tenant_id).await
    }

    async fn fetch_node_metadata(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, NodeMetadata>> {
        self.do_fetch_node_metadata(tenant_id).await
    }

    async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        self.do_delete_narrative_nodes_by_file(tenant_id, file_path)
            .await
    }

    async fn query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
        library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        self.do_query_cross_boundary(
            source_tenant,
            source_node_id,
            edge_types,
            max_hops,
            library_tenants,
        )
        .await
    }

    async fn export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        self.do_export_adjacency(tenant_id, edge_types).await
    }

    async fn export_nodes_for_tenant(&self, tenant_id: &str) -> GraphDbResult<Vec<GraphNode>> {
        self.do_export_nodes_for_tenant(tenant_id).await
    }

    async fn export_edges_for_tenant(&self, tenant_id: &str) -> GraphDbResult<Vec<GraphEdge>> {
        self.do_export_edges_for_tenant(tenant_id).await
    }
}
