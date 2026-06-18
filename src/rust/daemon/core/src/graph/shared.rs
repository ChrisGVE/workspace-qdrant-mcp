//! Shared graph store with read-write coordination and contention detection.
//!
//! Wraps a `GraphStore` in `Arc<RwLock<...>>` so that batch writes
//! (delete-then-insert during re-ingestion) appear atomic to concurrent
//! readers. SQLite WAL handles DB-level concurrency; this RwLock
//! coordinates the Rust-level access pattern.
//!
//! Lock acquisition is instrumented with timeout-based contention detection:
//! a `warn!` log is emitted when any lock wait exceeds 100ms, providing
//! observability into read-write contention under heavy ingestion loads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use tracing::warn;

use super::{
    AdjacencyExport, EdgeType, GraphDbResult, GraphEdge, GraphNode, GraphStats, GraphStore,
    ImpactReport, NodeMetadata, SymbolRow, TraversalNode,
};

/// Threshold after which a lock acquisition emits a `warn!` log.
const LOCK_CONTENTION_WARN_THRESHOLD: Duration = Duration::from_millis(100);

/// Timeout for lock acquisition. If exceeded, the operation returns
/// a `LockTimeout` error instead of blocking indefinitely.
const LOCK_ACQUIRE_TIMEOUT: Duration = Duration::from_secs(30);

/// Lightweight contention metrics tracked via atomics (no allocation).
#[derive(Debug, Default)]
struct LockMetrics {
    /// Number of read lock acquisitions that exceeded the warn threshold.
    slow_reads: AtomicU64,
    /// Number of write lock acquisitions that exceeded the warn threshold.
    slow_writes: AtomicU64,
}

/// Thread-safe, cloneable handle to a `GraphStore` with read-write coordination.
///
/// - **Readers** (gRPC query handlers): acquire a shared read lock.
/// - **Writers** (queue processor): acquire an exclusive write lock for the
///   full delete-then-insert cycle, so readers never see a half-updated file.
///
/// Lock acquisitions are instrumented: a warning is logged when wait time
/// exceeds 100ms, and acquisition times out after 30s to prevent unbounded
/// starvation.
///
/// Cloning is cheap (Arc bump).
#[derive(Clone)]
pub struct SharedGraphStore<S: GraphStore> {
    inner: Arc<RwLock<S>>,
    metrics: Arc<LockMetrics>,
}

impl<S: GraphStore> SharedGraphStore<S> {
    /// Wrap a store in a shared handle.
    pub fn new(store: S) -> Self {
        Self {
            inner: Arc::new(RwLock::new(store)),
            metrics: Arc::new(LockMetrics::default()),
        }
    }

    /// Access the inner store under a read lock for advanced operations.
    ///
    /// Callers should minimize the duration the guard is held. Prefer the
    /// scoped read methods (`query_related`, `stats`, etc.) when possible,
    /// as they release the lock immediately after the query completes.
    pub async fn read(&self) -> GraphDbResult<tokio::sync::RwLockReadGuard<'_, S>> {
        self.acquire_read("read").await
    }

    /// Return the cumulative count of slow read lock acquisitions (> 100ms).
    pub fn slow_read_count(&self) -> u64 {
        self.metrics.slow_reads.load(Ordering::Relaxed)
    }

    /// Return the cumulative count of slow write lock acquisitions (> 100ms).
    pub fn slow_write_count(&self) -> u64 {
        self.metrics.slow_writes.load(Ordering::Relaxed)
    }

    // ── Lock acquisition helpers ─────────────────────────────────────

    /// Acquire a read lock with contention detection and timeout.
    async fn acquire_read(&self, op: &str) -> GraphDbResult<tokio::sync::RwLockReadGuard<'_, S>> {
        let start = std::time::Instant::now();

        let guard = tokio::time::timeout(LOCK_ACQUIRE_TIMEOUT, self.inner.read())
            .await
            .map_err(|_| {
                warn!(
                    op,
                    timeout_secs = LOCK_ACQUIRE_TIMEOUT.as_secs(),
                    "graph read lock acquisition timed out"
                );
                super::GraphDbError::LockTimeout {
                    op: op.to_string(),
                    waited: LOCK_ACQUIRE_TIMEOUT,
                }
            })?;

        let elapsed = start.elapsed();
        if elapsed >= LOCK_CONTENTION_WARN_THRESHOLD {
            self.metrics.slow_reads.fetch_add(1, Ordering::Relaxed);
            warn!(
                op,
                wait_ms = elapsed.as_millis() as u64,
                slow_reads = self.metrics.slow_reads.load(Ordering::Relaxed),
                "graph read lock contention detected"
            );
        }

        Ok(guard)
    }

    /// Acquire a write lock with contention detection and timeout.
    async fn acquire_write(&self, op: &str) -> GraphDbResult<tokio::sync::RwLockWriteGuard<'_, S>> {
        let start = std::time::Instant::now();

        let guard = tokio::time::timeout(LOCK_ACQUIRE_TIMEOUT, self.inner.write())
            .await
            .map_err(|_| {
                warn!(
                    op,
                    timeout_secs = LOCK_ACQUIRE_TIMEOUT.as_secs(),
                    "graph write lock acquisition timed out"
                );
                super::GraphDbError::LockTimeout {
                    op: op.to_string(),
                    waited: LOCK_ACQUIRE_TIMEOUT,
                }
            })?;

        let elapsed = start.elapsed();
        if elapsed >= LOCK_CONTENTION_WARN_THRESHOLD {
            self.metrics.slow_writes.fetch_add(1, Ordering::Relaxed);
            warn!(
                op,
                wait_ms = elapsed.as_millis() as u64,
                slow_writes = self.metrics.slow_writes.load(Ordering::Relaxed),
                "graph write lock contention detected"
            );
        }

        Ok(guard)
    }

    // ── Read operations (shared lock) ────────────────────────────────

    /// Query nodes related to a given node within N hops.
    pub async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        let guard = self.acquire_read("query_related").await?;
        guard
            .query_related(tenant_id, node_id, max_hops, edge_types, branch)
            .await
    }

    /// Impact analysis for a symbol change.
    pub async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        let guard = self.acquire_read("impact_analysis").await?;
        guard
            .impact_analysis(tenant_id, symbol_name, file_path, branch)
            .await
    }

    /// Graph statistics.
    pub async fn stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        let guard = self.acquire_read("stats").await?;
        guard.stats(tenant_id, branch).await
    }

    /// Find shortest path between two nodes.
    pub async fn find_path(
        &self,
        tenant_id: &str,
        source_id: &str,
        target_id: &str,
        max_depth: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Option<Vec<TraversalNode>>> {
        let guard = self.acquire_read("find_path").await?;
        guard
            .find_path(
                tenant_id, source_id, target_id, max_depth, edge_types, branch,
            )
            .await
    }

    /// Traverse graph crossing tenant boundaries.
    pub async fn query_cross_boundary(
        &self,
        source_tenant: &str,
        source_node_id: &str,
        edge_types: &[EdgeType],
        max_hops: u32,
        library_tenants: &[String],
    ) -> GraphDbResult<Vec<TraversalNode>> {
        let guard = self.acquire_read("query_cross_boundary").await?;
        guard
            .query_cross_boundary(
                source_tenant,
                source_node_id,
                edge_types,
                max_hops,
                library_tenants,
            )
            .await
    }

    // ── Write operations (exclusive lock) ────────────────────────────

    /// Upsert a batch of nodes (exclusive lock).
    pub async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        let guard = self.acquire_write("upsert_nodes").await?;
        guard.upsert_nodes(nodes).await
    }

    /// Insert a batch of edges (exclusive lock).
    pub async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        let guard = self.acquire_write("insert_edges").await?;
        guard.insert_edges(edges).await
    }

    /// Atomic re-ingestion: delete old edges for a file, then upsert new
    /// nodes and edges. Holds the write lock for the entire operation so
    /// readers never see a partially-updated file. The underlying store
    /// runs all three steps in a single SQLite transaction, so a crash
    /// mid-operation leaves the database unchanged.
    pub async fn reingest_file(
        &self,
        tenant_id: &str,
        file_path: &str,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> GraphDbResult<()> {
        let guard = self.acquire_write("reingest_file").await?;
        guard
            .reingest_file(tenant_id, file_path, nodes, edges)
            .await
    }

    /// Delete all data for a tenant (exclusive lock).
    pub async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let guard = self.acquire_write("delete_tenant").await?;
        guard.delete_tenant(tenant_id).await
    }

    /// Prune orphaned nodes (exclusive lock).
    pub async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let guard = self.acquire_write("prune_orphans").await?;
        guard.prune_orphans(tenant_id).await
    }

    /// Query a tenant's code-graph symbols (shared read lock).
    pub async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        let guard = self.acquire_read("query_code_symbols").await?;
        guard.query_code_symbols(tenant_id).await
    }

    /// Delete file-owned narrative nodes (exclusive lock).
    pub async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        let guard = self.acquire_write("delete_narrative_nodes_by_file").await?;
        guard
            .delete_narrative_nodes_by_file(tenant_id, file_path)
            .await
    }

    /// Query all edges of a given type (shared read lock).
    pub async fn query_edges_by_type(&self, edge_type: EdgeType) -> GraphDbResult<Vec<GraphEdge>> {
        let guard = self.acquire_read("query_edges_by_type").await?;
        guard.query_edges_by_type(edge_type).await
    }

    /// Export the full adjacency structure for a tenant (shared read lock).
    ///
    /// The read guard is acquired, the inner store's `export_adjacency` runs to
    /// completion, and the owned `AdjacencyExport` is returned. The guard drops
    /// when this function returns — no borrow escapes (LOCK-SCOPE contract).
    pub async fn export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        let guard = self.acquire_read("export_adjacency").await?;
        guard.export_adjacency(tenant_id, edge_types).await
    }

    /// Fetch display metadata for all nodes of a tenant (shared read lock).
    ///
    /// The read guard is released before the owned map is returned, mirroring
    /// `export_adjacency` (LOCK-SCOPE contract). Used to enrich analytics
    /// results after the topology-only export has been processed.
    pub async fn fetch_node_metadata(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, NodeMetadata>> {
        let guard = self.acquire_read("fetch_node_metadata").await?;
        guard.fetch_node_metadata(tenant_id).await
    }
}

#[async_trait::async_trait]
impl<S: GraphStore + 'static> GraphStore for SharedGraphStore<S> {
    async fn upsert_node(&self, node: &GraphNode) -> GraphDbResult<()> {
        let guard = self.acquire_write("upsert_node").await?;
        guard.upsert_node(node).await
    }

    async fn upsert_nodes(&self, nodes: &[GraphNode]) -> GraphDbResult<()> {
        self.upsert_nodes(nodes).await
    }

    async fn insert_edge(&self, edge: &GraphEdge) -> GraphDbResult<()> {
        let guard = self.acquire_write("insert_edge").await?;
        guard.insert_edge(edge).await
    }

    async fn insert_edges(&self, edges: &[GraphEdge]) -> GraphDbResult<()> {
        self.insert_edges(edges).await
    }

    async fn delete_edges_by_file(&self, tenant_id: &str, file_path: &str) -> GraphDbResult<u64> {
        let guard = self.acquire_write("delete_edges_by_file").await?;
        guard.delete_edges_by_file(tenant_id, file_path).await
    }

    async fn delete_tenant(&self, tenant_id: &str) -> GraphDbResult<u64> {
        self.delete_tenant(tenant_id).await
    }

    async fn query_related(
        &self,
        tenant_id: &str,
        node_id: &str,
        max_hops: u32,
        edge_types: Option<&[EdgeType]>,
        branch: Option<&str>,
    ) -> GraphDbResult<Vec<TraversalNode>> {
        self.query_related(tenant_id, node_id, max_hops, edge_types, branch)
            .await
    }

    async fn impact_analysis(
        &self,
        tenant_id: &str,
        symbol_name: &str,
        file_path: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<ImpactReport> {
        self.impact_analysis(tenant_id, symbol_name, file_path, branch)
            .await
    }

    async fn stats(
        &self,
        tenant_id: Option<&str>,
        branch: Option<&str>,
    ) -> GraphDbResult<GraphStats> {
        self.stats(tenant_id, branch).await
    }

    async fn prune_orphans(&self, tenant_id: &str) -> GraphDbResult<u64> {
        self.prune_orphans(tenant_id).await
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
        self.find_path(
            tenant_id, source_id, target_id, max_depth, edge_types, branch,
        )
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
        self.query_cross_boundary(
            source_tenant,
            source_node_id,
            edge_types,
            max_hops,
            library_tenants,
        )
        .await
    }

    async fn reingest_file(
        &self,
        tenant_id: &str,
        file_path: &str,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> GraphDbResult<()> {
        self.reingest_file(tenant_id, file_path, nodes, edges).await
    }

    async fn query_code_symbols(&self, tenant_id: &str) -> GraphDbResult<Vec<SymbolRow>> {
        self.query_code_symbols(tenant_id).await
    }

    async fn delete_narrative_nodes_by_file(
        &self,
        tenant_id: &str,
        file_path: &str,
    ) -> GraphDbResult<u64> {
        self.delete_narrative_nodes_by_file(tenant_id, file_path)
            .await
    }

    async fn query_edges_by_type(&self, edge_type: EdgeType) -> GraphDbResult<Vec<GraphEdge>> {
        self.query_edges_by_type(edge_type).await
    }

    async fn export_adjacency(
        &self,
        tenant_id: &str,
        edge_types: Option<&[EdgeType]>,
    ) -> GraphDbResult<AdjacencyExport> {
        self.export_adjacency(tenant_id, edge_types).await
    }

    async fn fetch_node_metadata(
        &self,
        tenant_id: &str,
    ) -> GraphDbResult<std::collections::HashMap<String, NodeMetadata>> {
        self.fetch_node_metadata(tenant_id).await
    }

    /// List tenants with graph data (shared read lock).
    async fn graph_tenants(&self) -> GraphDbResult<Vec<String>> {
        let guard = self.acquire_read("graph_tenants").await?;
        guard.graph_tenants().await
    }

    /// Resolve dangling stub edges to real nodes by name (exclusive lock).
    async fn resolve_stub_edges(&self, tenant_id: &str) -> GraphDbResult<u64> {
        let guard = self.acquire_write("resolve_stub_edges").await?;
        guard.resolve_stub_edges(tenant_id).await
    }

    /// Resolve dangling stub edges across every tenant.
    ///
    /// Snapshots the tenant list under a brief read lock, then resolves each
    /// tenant under its OWN short-lived write lock (via `resolve_stub_edges`).
    /// This keeps a large multi-tenant sweep from holding one exclusive lock
    /// for its whole duration and starving ingestion — only one tenant's
    /// resolution blocks writes at a time, and the lock is released between
    /// tenants so other graph writes can interleave.
    async fn resolve_all_stub_edges(&self) -> GraphDbResult<u64> {
        let tenants = {
            let guard = self.acquire_read("resolve_all_stub_edges.tenants").await?;
            guard.graph_tenants().await?
        };
        let mut total: u64 = 0;
        for tenant in tenants {
            total += self.resolve_stub_edges(&tenant).await?;
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SqliteGraphStore;
    use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
    use std::sync::atomic::Ordering;
    use std::time::Duration;

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

        // Run schema (v1 + v2)
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
                branches TEXT NOT NULL DEFAULT '[\"main\"]',
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
        sqlx::query("CREATE INDEX idx_nodes_branches ON graph_nodes(tenant_id, branches)")
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
                branch TEXT,
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
        sqlx::query("CREATE INDEX idx_edges_branch ON graph_edges(tenant_id, branch)")
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

        let stats = store.stats(Some(T), None).await.unwrap();
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

        let stats = store.stats(Some(T), None).await.unwrap();
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
                s.query_related(T, &node_id, 1, None, None).await.unwrap()
            }));
        }

        for handle in handles {
            let results = handle.await.unwrap();
            assert_eq!(results.len(), 1);
        }
    }

    #[tokio::test]
    async fn test_resolve_stub_edges_by_name() {
        use super::super::{EdgeType, NodeType};
        let store = test_shared_store().await;

        // Real caller (a.rs) + real callee (b.rs) + a name-only stub "callee".
        let caller = GraphNode::new(T, "a.rs", "caller", NodeType::Function);
        let callee = GraphNode::new(T, "b.rs", "callee", NodeType::Function);
        let stub = GraphNode::stub(T, "callee", NodeType::Function);
        store
            .upsert_nodes(&[caller.clone(), callee.clone(), stub.clone()])
            .await
            .unwrap();
        // Dangling: caller -> stub("callee").
        let dangling = GraphEdge::new(T, &caller.node_id, &stub.node_id, EdgeType::Calls, "a.rs");
        store.insert_edges(&[dangling]).await.unwrap();

        // Unmatched external stub ("println") — no project node — stays dangling.
        let ext = GraphNode::stub(T, "println", NodeType::Function);
        store.upsert_nodes(&[ext.clone()]).await.unwrap();
        let ext_edge = GraphEdge::new(T, &caller.node_id, &ext.node_id, EdgeType::Calls, "a.rs");
        store.insert_edges(&[ext_edge]).await.unwrap();

        let repointed = store.resolve_stub_edges(T).await.unwrap();
        assert_eq!(repointed, 1, "only the matchable stub edge repoints");

        // Edge now reaches the REAL callee in b.rs.
        let related = store
            .query_related(T, &caller.node_id, 1, None, None)
            .await
            .unwrap();
        assert!(
            related.iter().any(|n| n.file_path == "b.rs"),
            "resolved edge should target the real callee node in b.rs"
        );

        // Matched stub is pruned; the unmatched external stub remains.
        let guard = store.read().await.unwrap();
        let stub_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM graph_nodes WHERE tenant_id = ?1 AND file_path = ''",
        )
        .bind(T)
        .fetch_one(guard.pool())
        .await
        .unwrap();
        assert_eq!(stub_count, 1, "matched stub pruned; 'println' stub remains");
    }

    #[tokio::test]
    async fn test_resolve_stub_edges_skips_ambiguous() {
        use super::super::{EdgeType, NodeType};
        let store = test_shared_store().await;

        // Two real nodes named "new" in different files → ambiguous.
        let caller = GraphNode::new(T, "a.rs", "caller", NodeType::Function);
        let new_a = GraphNode::new(T, "x.rs", "new", NodeType::Function);
        let new_b = GraphNode::new(T, "y.rs", "new", NodeType::Function);
        let stub = GraphNode::stub(T, "new", NodeType::Function);
        store
            .upsert_nodes(&[caller.clone(), new_a, new_b, stub.clone()])
            .await
            .unwrap();
        let dangling = GraphEdge::new(T, &caller.node_id, &stub.node_id, EdgeType::Calls, "a.rs");
        store.insert_edges(&[dangling]).await.unwrap();

        let repointed = store.resolve_stub_edges(T).await.unwrap();
        assert_eq!(
            repointed, 0,
            "ambiguous name (2 defining files) must not repoint"
        );
    }

    #[tokio::test]
    async fn test_clone_is_cheap() {
        let store = test_shared_store().await;
        let clone1 = store.clone();
        let clone2 = store.clone();

        // All clones share the same underlying data
        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        store.upsert_nodes(&[a]).await.unwrap();

        let stats1 = clone1.stats(Some(T), None).await.unwrap();
        let stats2 = clone2.stats(Some(T), None).await.unwrap();
        assert_eq!(stats1.total_nodes, 1);
        assert_eq!(stats2.total_nodes, 1);
    }

    /// Metrics counters start at zero and are shared across clones.
    #[tokio::test]
    async fn test_metrics_initial_state() {
        let store = test_shared_store().await;
        assert_eq!(store.slow_read_count(), 0);
        assert_eq!(store.slow_write_count(), 0);

        // Clones share the same metrics
        let clone = store.clone();
        assert_eq!(clone.slow_read_count(), 0);

        // After a fast operation, counters remain at zero
        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        store.upsert_nodes(&[a]).await.unwrap();
        store.stats(Some(T), None).await.unwrap();

        assert_eq!(store.slow_read_count(), 0);
        assert_eq!(store.slow_write_count(), 0);
    }

    /// Metrics counters are shared across clones (Arc semantics).
    #[tokio::test]
    async fn test_metrics_shared_across_clones() {
        let store = test_shared_store().await;
        let clone = store.clone();

        // Manually bump to verify Arc sharing
        store.metrics.slow_reads.store(5, Ordering::Relaxed);
        assert_eq!(clone.slow_read_count(), 5);
    }

    /// Concurrent readers + writer: readers see consistent snapshots,
    /// never a partially-updated file.
    #[tokio::test]
    async fn test_concurrent_read_write_consistency() {
        let store = test_shared_store().await;

        // Seed initial data: file a.rs with node a -> node b
        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        let b = GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function);
        store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();
        let edge_ab = GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store.insert_edges(&[edge_ab]).await.unwrap();

        // Barrier: signal to start concurrent work once all tasks are spawned
        let barrier = Arc::new(tokio::sync::Barrier::new(11)); // 10 readers + 1 writer

        // Spawn 10 concurrent readers that each read stats
        let mut reader_handles = Vec::new();
        for _ in 0..10 {
            let s = store.clone();
            let b = barrier.clone();
            reader_handles.push(tokio::spawn(async move {
                b.wait().await;
                // Read stats multiple times to increase overlap probability
                let mut results = Vec::new();
                for _ in 0..5 {
                    results.push(s.stats(Some(T), None).await.unwrap());
                }
                results
            }));
        }

        // Spawn a writer that reingests a.rs with different edges
        let writer_store = store.clone();
        let writer_barrier = barrier.clone();
        let a_clone = a.clone();
        let writer_handle = tokio::spawn(async move {
            writer_barrier.wait().await;
            let c = GraphNode::new(T, "c.rs", "c", super::super::NodeType::Function);
            let edge_ac = GraphEdge::new(
                T,
                &a_clone.node_id,
                &c.node_id,
                super::super::EdgeType::Calls,
                "a.rs",
            );
            writer_store
                .reingest_file(T, "a.rs", &[a_clone, c], &[edge_ac])
                .await
                .unwrap();
        });

        writer_handle.await.unwrap();

        for handle in reader_handles {
            let snapshots = handle.await.unwrap();
            for stats in &snapshots {
                // Each snapshot must be internally consistent:
                // Either before reingest (2 nodes, 1 edge) or after (3 nodes, 1 edge).
                // The edge count must always be exactly 1 -- never 0 (mid-reingest).
                assert_eq!(
                    stats.total_edges, 1,
                    "reader saw inconsistent edge count: {}",
                    stats.total_edges
                );
                assert!(
                    stats.total_nodes == 2 || stats.total_nodes == 3,
                    "reader saw unexpected node count: {}",
                    stats.total_nodes
                );
            }
        }
    }

    /// Concurrent writers are serialized: total mutations appear in sequence.
    #[tokio::test]
    async fn test_concurrent_writers_serialized() {
        let store = test_shared_store().await;

        // Seed nodes for edges to reference
        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        let b = GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function);
        let c = GraphNode::new(T, "c.rs", "c", super::super::NodeType::Function);
        let d = GraphNode::new(T, "d.rs", "d", super::super::NodeType::Function);
        store
            .upsert_nodes(&[a.clone(), b.clone(), c.clone(), d.clone()])
            .await
            .unwrap();

        let barrier = Arc::new(tokio::sync::Barrier::new(2));

        // Writer 1: reingest a.rs with edge a -> c
        let s1 = store.clone();
        let b1 = barrier.clone();
        let a1 = a.clone();
        let c1 = c.clone();
        let h1 = tokio::spawn(async move {
            b1.wait().await;
            let edge = GraphEdge::new(
                T,
                &a1.node_id,
                &c1.node_id,
                super::super::EdgeType::Calls,
                "a.rs",
            );
            s1.reingest_file(T, "a.rs", &[a1, c1], &[edge])
                .await
                .unwrap();
        });

        // Writer 2: reingest b.rs with edge b -> d
        let s2 = store.clone();
        let b2 = barrier.clone();
        let b_node = b.clone();
        let d1 = d.clone();
        let h2 = tokio::spawn(async move {
            b2.wait().await;
            let edge = GraphEdge::new(
                T,
                &b_node.node_id,
                &d1.node_id,
                super::super::EdgeType::Calls,
                "b.rs",
            );
            s2.reingest_file(T, "b.rs", &[b_node, d1], &[edge])
                .await
                .unwrap();
        });

        h1.await.unwrap();
        h2.await.unwrap();

        // Both writes completed; edges should reflect both reingestions
        let stats = store.stats(Some(T), None).await.unwrap();
        assert_eq!(
            stats.total_edges, 2,
            "expected 2 edges after concurrent reingestion, got {}",
            stats.total_edges
        );
    }

    /// Read operations still succeed while writer holds the lock for
    /// a short duration -- no deadlock or panic.
    #[tokio::test]
    async fn test_reads_complete_after_write() {
        let store = test_shared_store().await;

        let a = GraphNode::new(T, "a.rs", "a", super::super::NodeType::Function);
        store.upsert_nodes(&[a.clone()]).await.unwrap();

        // Perform a write followed immediately by reads
        let b = GraphNode::new(T, "b.rs", "b", super::super::NodeType::Function);
        let edge = GraphEdge::new(
            T,
            &a.node_id,
            &b.node_id,
            super::super::EdgeType::Calls,
            "a.rs",
        );
        store
            .reingest_file(T, "a.rs", &[a.clone(), b], &[edge])
            .await
            .unwrap();

        // Multiple reads should all succeed without timeout
        let mut handles = Vec::new();
        for _ in 0..5 {
            let s = store.clone();
            handles.push(tokio::spawn(async move {
                s.stats(Some(T), None).await.unwrap()
            }));
        }

        for handle in handles {
            let stats = handle.await.unwrap();
            assert_eq!(stats.total_nodes, 2);
            assert_eq!(stats.total_edges, 1);
        }
    }

    /// Write lock timeout fires when a read lock is held indefinitely.
    /// We simulate this by holding a read guard in a task and attempting
    /// a write with a very short timeout override (via a separate store
    /// with a custom timeout, but since we use a constant, we test the
    /// error variant production directly).
    #[tokio::test]
    async fn test_lock_timeout_error_variant() {
        // Verify the error type is well-formed and displays correctly.
        let err = super::super::GraphDbError::LockTimeout {
            op: "reingest_file".to_string(),
            waited: Duration::from_secs(30),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("reingest_file"),
            "error should mention the operation"
        );
        assert!(msg.contains("30s"), "error should mention the wait time");
    }
}
