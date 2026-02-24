//! Shared graph store with read-write coordination.
//!
//! Wraps a `GraphStore` in `Arc<RwLock<...>>` so that batch writes
//! (delete-then-insert during re-ingestion) appear atomic to concurrent
//! readers. SQLite WAL handles DB-level concurrency; this RwLock
//! coordinates the Rust-level access pattern.

use std::sync::Arc;

use tokio::sync::RwLock;

use super::{
    EdgeType, GraphDbResult, GraphEdge, GraphNode, GraphStats, GraphStore, ImpactReport,
    TraversalNode,
};

/// Thread-safe, cloneable handle to a `GraphStore` with read-write coordination.
///
/// - **Readers** (gRPC query handlers): acquire a shared read lock.
/// - **Writers** (queue processor): acquire an exclusive write lock for the
///   full delete-then-insert cycle, so readers never see a half-updated file.
///
/// Cloning is cheap (Arc bump).
#[derive(Clone)]
pub struct SharedGraphStore<S: GraphStore> {
    inner: Arc<RwLock<S>>,
}

impl<S: GraphStore> SharedGraphStore<S> {
    /// Wrap a store in a shared handle.
    pub fn new(store: S) -> Self {
        Self {
            inner: Arc::new(RwLock::new(store)),
        }
    }

    /// Access the inner store under a read lock for advanced operations.
    pub async fn read(&self) -> tokio::sync::RwLockReadGuard<'_, S> {
        self.inner.read().await
    }

    // ── Read operations (shared lock) ────────────────────────────────

    /// Query nodes related to a given node within N hops.
    pub async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        let guard = self.inner.read().await;
        guard
            .query_related(tenant_id, node_id, max_hops, edge_types)
            .await
    }

    /// Impact analysis for a symbol change.
    pub async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        let guard = self.inner.read().await;
        guard
            .impact_analysis(tenant_id, symbol_name, file_path)
            .await
    }

    /// Graph statistics.
    pub async fn stats(&self, tenant_id: Option<&str>) -> GraphDbResult<GraphStats> {
        let guard = self.inner.read().await;
        guard.stats(tenant_id).await
    }

    // ── Write operations (exclusive lock) ────────────────────────────

    /// Upsert a batch of nodes (exclusive lock).
    pub async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        let guard = self.inner.write().await;
        guard.upsert_nodes(nodes).await
    }

    /// Insert a batch of edges (exclusive lock).
    pub async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        let guard = self.inner.write().await;
        guard.insert_edges(edges).await
    }

    /// Atomic re-ingestion: delete old edges for a file, then insert new
    /// nodes and edges. Holds the write lock for the entire operation so
    /// readers never see a partially-updated file.
    pub async fn reingest_file(
        &self,
        tenant_id: &str,
        file_path: &str,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> GraphDbResult<()> {
        let guard = self.inner.write().await;
        guard.delete_edges_by_file(tenant_id, file_path).await?;
        guard.upsert_nodes(nodes).await?;
        guard.insert_edges(edges).await?;
        Ok(())
    }

    /// Delete all data for a tenant (exclusive lock).
    pub async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let guard = self.inner.write().await;
        guard.delete_tenant(tenant_id).await
    }

    /// Prune orphaned nodes (exclusive lock).
    pub async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let guard = self.inner.write().await;
        guard.prune_orphans(tenant_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SqliteGraphStore;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

    async fn test_shared_store() -> SharedGraphStore<SqliteGraphStore> {
        let opts = SqliteConnectOptions::new()
            .filename(":memory:")
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .unwrap();

        // Run schema
        sqlx::query(
            "CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                signature TEXT,
                language TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("CREATE INDEX idx_nodes_tenant ON graph_nodes(tenant_id)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_nodes_file ON graph_nodes(tenant_id, file_path)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_nodes_symbol ON graph_nodes(tenant_id, symbol_name)")
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(
            "CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                source_file TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                metadata_json TEXT,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
            )",
        )
        .execute(&pool)
        .await
        .unwrap();
        sqlx::query("CREATE INDEX idx_edges_tenant ON graph_edges(tenant_id)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_edges_source ON graph_edges(source_node_id)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_edges_target ON graph_edges(target_node_id)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_edges_source_file ON graph_edges(tenant_id, source_file)")
            .execute(&pool)
            .await
            .unwrap();
        sqlx::query("CREATE INDEX idx_edges_type ON graph_edges(edge_type)")
            .execute(&pool)
            .await
            .unwrap();

        SharedGraphStore::new(SqliteGraphStore::new(pool))
    }

    const T: &str = "test-tenant";

    #[tokio::test]
    async fn test_shared_write_then_read() {
        let store = test_shared_store().await;

        let nodes = vec![
            GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function),
            GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function),
        ];
        store.upsert_nodes(&nodes).await.unwrap();

        let edge = GraphEdge::new(
            T,
            &nodes[0].node_id,
            &nodes[1].node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store.insert_edges(&[edge]).await.unwrap();

        let stats = store.stats(Some(T)).await.unwrap();
        assert_eq!(stats.total_nodes, 2);
        assert_eq!(stats.total_edges, 1);
    }

    #[tokio::test]
    async fn test_reingest_file_atomic() {
        let store = test_shared_store().await;

        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        let b = GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function);
        store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let old_edge = GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store.insert_edges(&[old_edge]).await.unwrap();

        // Re-ingest a.rs with a new edge target
        let c = GraphNode::new(T, "c.rs", "c", super::super::NodeType::Function);
        let new_edge = GraphEdge::new(
            T,
            &a.node_id,
            &c.node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store
            .reingest_file(T, "a.rs", &[a.clone(), c], &[new_edge])
            .await
            .unwrap();

        let stats = store.stats(Some(T)).await.unwrap();
        // Old a->b edge deleted, new a->c edge inserted
        assert_eq!(stats.total_edges, 1);
    }

    #[tokio::test]
    async fn test_concurrent_readers() {
        let store = test_shared_store().await;

        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        let b = GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function);
        store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let edge = GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store.insert_edges(&[edge]).await.unwrap();

        // Spawn 10 concurrent readers
        let mut handles = Vec::new();
        for _ in 0..10 {
            let s = store.clone();
            let node_id = a.node_id.clone();
            handles.push(tokio::spawn(async move {
                s.query_related(T, &node_id, 1, None).await.unwrap()
            }));
        }

        for handle in handles {
            let results = handle.await.unwrap();
            assert_eq!(results.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_clone_is_cheap() {
        let store = test_shared_store().await;
        let clone1 = store.clone();
        let clone2 = store.clone();

        // All clones share the same underlying data
        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        store.upsert_nodes(&[a]).await.unwrap();

        let stats1 = clone1.stats(Some(T)).await.unwrap();
        let stats2 = clone2.stats(Some(T)).await.unwrap();
        assert_eq!(stats1.total_nodes, 1);
        assert_eq!(stats2.total_nodes, 1);
    }
}
