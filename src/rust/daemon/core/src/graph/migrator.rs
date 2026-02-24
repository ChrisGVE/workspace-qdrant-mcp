/// Graph migration utility for moving data between SQLite and LadybugDB backends.
///
/// Exports nodes and edges from one backend, imports to another in batches,
/// then validates counts match. Designed for `wqm graph migrate` CLI command.

use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use tracing::{debug, info, warn};

use super::schema::GraphDbResult;
use super::{EdgeType, GraphEdge, GraphNode, GraphStore, NodeType};

/// Default batch size for import operations.
const DEFAULT_BATCH_SIZE: usize = 500;

/// Report produced after a migration completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationReport {
    /// Number of nodes exported from the source.
    pub nodes_exported: u64,
    /// Number of edges exported from the source.
    pub edges_exported: u64,
    /// Number of nodes imported to the target.
    pub nodes_imported: u64,
    /// Number of edges imported to the target.
    pub edges_imported: u64,
    /// Whether node counts match.
    pub nodes_match: bool,
    /// Whether edge counts match.
    pub edges_match: bool,
    /// Tenant IDs that were migrated (None = all).
    pub tenants: Vec<String>,
    /// Any warnings or issues encountered.
    pub warnings: Vec<String>,
}

/// Snapshot of graph data for migration.
#[derive(Debug)]
pub struct GraphSnapshot {
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

// ─── Export from SQLite ─────────────────────────────────────────────────

/// Export all nodes from SQLite, optionally filtered by tenant.
pub async fn export_nodes_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphNode>> {
    let rows = match tenant_id {
        Some(tid) => {
            sqlx::query(
                "SELECT node_id, tenant_id, symbol_name, symbol_type,
                        file_path, start_line, end_line, signature, language
                 FROM graph_nodes WHERE tenant_id = ?1
                 ORDER BY node_id",
            )
            .bind(tid)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT node_id, tenant_id, symbol_name, symbol_type,
                        file_path, start_line, end_line, signature, language
                 FROM graph_nodes ORDER BY node_id",
            )
            .fetch_all(pool)
            .await?
        }
    };

    let nodes = rows
        .iter()
        .filter_map(|row| {
            let stype_str: String = row.get("symbol_type");
            let symbol_type = NodeType::from_str(&stype_str)?;
            Some(GraphNode {
                node_id: row.get("node_id"),
                tenant_id: row.get("tenant_id"),
                symbol_name: row.get("symbol_name"),
                symbol_type,
                file_path: row.get("file_path"),
                start_line: row
                    .get::<Option<i64>, _>("start_line")
                    .map(|v| v as u32),
                end_line: row.get::<Option<i64>, _>("end_line").map(|v| v as u32),
                signature: row.get("signature"),
                language: row.get("language"),
            })
        })
        .collect::<Vec<_>>();

    info!("Exported {} nodes from SQLite", nodes.len());
    Ok(nodes)
}

/// Export all edges from SQLite, optionally filtered by tenant.
pub async fn export_edges_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphEdge>> {
    let rows = match tenant_id {
        Some(tid) => {
            sqlx::query(
                "SELECT edge_id, tenant_id, source_node_id, target_node_id,
                        edge_type, source_file, weight, metadata_json
                 FROM graph_edges WHERE tenant_id = ?1
                 ORDER BY edge_id",
            )
            .bind(tid)
            .fetch_all(pool)
            .await?
        }
        None => {
            sqlx::query(
                "SELECT edge_id, tenant_id, source_node_id, target_node_id,
                        edge_type, source_file, weight, metadata_json
                 FROM graph_edges ORDER BY edge_id",
            )
            .fetch_all(pool)
            .await?
        }
    };

    let mut edges = Vec::with_capacity(rows.len());
    let mut skipped = 0u64;

    for row in &rows {
        let etype_str: String = row.get("edge_type");
        match EdgeType::from_str(&etype_str) {
            Some(edge_type) => {
                edges.push(GraphEdge {
                    edge_id: row.get("edge_id"),
                    tenant_id: row.get("tenant_id"),
                    source_node_id: row.get("source_node_id"),
                    target_node_id: row.get("target_node_id"),
                    edge_type,
                    source_file: row.get("source_file"),
                    weight: row.get("weight"),
                    metadata_json: row.get("metadata_json"),
                });
            }
            None => {
                skipped += 1;
                warn!("Skipping edge with unknown type: {}", etype_str);
            }
        }
    }

    if skipped > 0 {
        warn!("Skipped {} edges with unrecognized types", skipped);
    }
    info!("Exported {} edges from SQLite", edges.len());
    Ok(edges)
}

/// Export a full snapshot from SQLite.
pub async fn export_sqlite(
    pool: &SqlitePool,
    tenant_id: Option<&str>,
) -> GraphDbResult<GraphSnapshot> {
    let nodes = export_nodes_sqlite(pool, tenant_id).await?;
    let edges = export_edges_sqlite(pool, tenant_id).await?;
    Ok(GraphSnapshot { nodes, edges })
}

// ─── Export from LadybugDB ──────────────────────────────────────────────

/// Export all nodes from LadybugDB, optionally filtered by tenant.
#[cfg(feature = "ladybug")]
pub fn export_nodes_ladybug(
    store: &super::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphNode>> {
    let filter = match tenant_id {
        Some(tid) => format!(
            " WHERE n.tenant_id = '{}'",
            tid.replace('\'', "\\'")
        ),
        None => String::new(),
    };

    let cypher = format!(
        "MATCH (n:GraphNode){} \
         RETURN n.node_id, n.tenant_id, n.symbol_name, n.symbol_type, \
                n.file_path, n.start_line, n.end_line, n.signature, n.language",
        filter
    );

    let rows = store.execute_cypher(&cypher)?;
    let mut nodes = Vec::with_capacity(rows.len());

    for row in &rows {
        if row.len() < 5 {
            continue;
        }
        let symbol_type = match NodeType::from_str(&row[3]) {
            Some(t) => t,
            None => continue,
        };
        nodes.push(GraphNode {
            node_id: row[0].clone(),
            tenant_id: row[1].clone(),
            symbol_name: row[2].clone(),
            symbol_type,
            file_path: row[4].clone(),
            start_line: row.get(5).and_then(|s| s.parse().ok()),
            end_line: row.get(6).and_then(|s| s.parse().ok()),
            signature: row.get(7).map(|s| s.clone()).filter(|s| !s.is_empty()),
            language: row.get(8).map(|s| s.clone()).filter(|s| !s.is_empty()),
        });
    }

    info!("Exported {} nodes from LadybugDB", nodes.len());
    Ok(nodes)
}

/// Export all edges from LadybugDB, optionally filtered by tenant.
#[cfg(feature = "ladybug")]
pub fn export_edges_ladybug(
    store: &super::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<Vec<GraphEdge>> {
    let mut edges = Vec::new();

    for edge_type_str in &[
        "CALLS",
        "CONTAINS",
        "IMPORTS",
        "USES_TYPE",
        "EXTENDS",
        "IMPLEMENTS",
    ] {
        let filter = match tenant_id {
            Some(tid) => format!(
                " WHERE r.tenant_id = '{}'",
                tid.replace('\'', "\\'")
            ),
            None => String::new(),
        };

        let cypher = format!(
            "MATCH (a:GraphNode)-[r:{}]->(b:GraphNode){} \
             RETURN r.edge_id, r.tenant_id, a.node_id, b.node_id, \
                    r.source_file, r.weight",
            edge_type_str, filter
        );

        let edge_type = match EdgeType::from_str(edge_type_str) {
            Some(t) => t,
            None => continue,
        };

        let rows = store.execute_cypher(&cypher)?;
        for row in &rows {
            if row.len() < 6 {
                continue;
            }
            edges.push(GraphEdge {
                edge_id: row[0].clone(),
                tenant_id: row[1].clone(),
                source_node_id: row[2].clone(),
                target_node_id: row[3].clone(),
                edge_type,
                source_file: row[4].clone(),
                weight: row[5].parse().unwrap_or(1.0),
                metadata_json: None,
            });
        }
    }

    info!("Exported {} edges from LadybugDB", edges.len());
    Ok(edges)
}

/// Export a full snapshot from LadybugDB.
#[cfg(feature = "ladybug")]
pub fn export_ladybug(
    store: &super::LadybugGraphStore,
    tenant_id: Option<&str>,
) -> GraphDbResult<GraphSnapshot> {
    let nodes = export_nodes_ladybug(store, tenant_id)?;
    let edges = export_edges_ladybug(store, tenant_id)?;
    Ok(GraphSnapshot { nodes, edges })
}

// ─── Import to any GraphStore ───────────────────────────────────────────

/// Import a graph snapshot into a target store in batches.
pub async fn import_to_store<S: GraphStore>(
    snapshot: &GraphSnapshot,
    target: &S,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    let batch_size = if batch_size == 0 {
        DEFAULT_BATCH_SIZE
    } else {
        batch_size
    };

    let mut warnings = Vec::new();

    // Import nodes in batches
    let mut nodes_imported = 0u64;
    for chunk in snapshot.nodes.chunks(batch_size) {
        match target.upsert_nodes(chunk).await {
            Ok(()) => {
                nodes_imported += chunk.len() as u64;
                debug!(
                    "Imported node batch: {}/{}",
                    nodes_imported,
                    snapshot.nodes.len()
                );
            }
            Err(e) => {
                let msg = format!(
                    "Failed to import node batch at offset {}: {}",
                    nodes_imported, e
                );
                warn!("{}", msg);
                warnings.push(msg);
            }
        }
    }

    // Import edges in batches
    let mut edges_imported = 0u64;
    for chunk in snapshot.edges.chunks(batch_size) {
        match target.insert_edges(chunk).await {
            Ok(()) => {
                edges_imported += chunk.len() as u64;
                debug!(
                    "Imported edge batch: {}/{}",
                    edges_imported,
                    snapshot.edges.len()
                );
            }
            Err(e) => {
                let msg = format!(
                    "Failed to import edge batch at offset {}: {}",
                    edges_imported, e
                );
                warn!("{}", msg);
                warnings.push(msg);
            }
        }
    }

    // Collect tenant IDs
    let mut tenant_set: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for node in &snapshot.nodes {
        tenant_set.insert(&node.tenant_id);
    }

    let report = MigrationReport {
        nodes_exported: snapshot.nodes.len() as u64,
        edges_exported: snapshot.edges.len() as u64,
        nodes_imported,
        edges_imported,
        nodes_match: nodes_imported == snapshot.nodes.len() as u64,
        edges_match: edges_imported == snapshot.edges.len() as u64,
        tenants: tenant_set.into_iter().map(|s| s.to_string()).collect(),
        warnings,
    };

    info!(
        "Migration complete: {} nodes, {} edges, match={}",
        report.nodes_imported,
        report.edges_imported,
        report.nodes_match && report.edges_match
    );

    Ok(report)
}

// ─── Full migration pipelines ───────────────────────────────────────────

/// Migrate from SQLite to any GraphStore target.
pub async fn migrate_from_sqlite<S: GraphStore>(
    source_pool: &SqlitePool,
    target: &S,
    tenant_id: Option<&str>,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    info!(
        "Starting SQLite export (tenant: {:?})",
        tenant_id
    );
    let snapshot = export_sqlite(source_pool, tenant_id).await?;
    info!(
        "Exported {} nodes, {} edges from SQLite",
        snapshot.nodes.len(),
        snapshot.edges.len()
    );
    import_to_store(&snapshot, target, batch_size).await
}

/// Migrate from LadybugDB to any GraphStore target.
#[cfg(feature = "ladybug")]
pub async fn migrate_from_ladybug<S: GraphStore>(
    source: &super::LadybugGraphStore,
    target: &S,
    tenant_id: Option<&str>,
    batch_size: usize,
) -> GraphDbResult<MigrationReport> {
    info!(
        "Starting LadybugDB export (tenant: {:?})",
        tenant_id
    );
    let snapshot = export_ladybug(source, tenant_id)?;
    info!(
        "Exported {} nodes, {} edges from LadybugDB",
        snapshot.nodes.len(),
        snapshot.edges.len()
    );
    import_to_store(&snapshot, target, batch_size).await
}

/// Validate a migration by comparing stats between source and target.
pub async fn validate_migration<S: GraphStore>(
    source_pool: &SqlitePool,
    target: &S,
    tenant_id: Option<&str>,
) -> GraphDbResult<bool> {
    // Count source
    let (source_nodes, source_edges) = match tenant_id {
        Some(tid) => {
            let n: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM graph_nodes WHERE tenant_id = ?1",
            )
            .bind(tid)
            .fetch_one(source_pool)
            .await?;
            let e: (i64,) = sqlx::query_as(
                "SELECT COUNT(*) FROM graph_edges WHERE tenant_id = ?1",
            )
            .bind(tid)
            .fetch_one(source_pool)
            .await?;
            (n.0 as u64, e.0 as u64)
        }
        None => {
            let n: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM graph_nodes")
                    .fetch_one(source_pool)
                    .await?;
            let e: (i64,) =
                sqlx::query_as("SELECT COUNT(*) FROM graph_edges")
                    .fetch_one(source_pool)
                    .await?;
            (n.0 as u64, e.0 as u64)
        }
    };

    // Count target
    let target_stats = target.stats(tenant_id).await?;

    let nodes_ok = source_nodes == target_stats.total_nodes;
    let edges_ok = source_edges == target_stats.total_edges;

    if nodes_ok && edges_ok {
        info!(
            "Validation passed: {} nodes, {} edges",
            source_nodes, source_edges
        );
    } else {
        warn!(
            "Validation FAILED: source({} nodes, {} edges) vs target({} nodes, {} edges)",
            source_nodes, source_edges, target_stats.total_nodes, target_stats.total_edges
        );
    }

    Ok(nodes_ok && edges_ok)
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SqliteGraphStore;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn setup_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();

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
                created_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT ''
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
                created_at TEXT NOT NULL DEFAULT '',
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

        pool
    }

    async fn seed_graph(pool: &SqlitePool) {
        let store = SqliteGraphStore::new(pool.clone());
        let a = GraphNode::new("t1", "a.rs", "alpha", NodeType::Function);
        let b = GraphNode::new("t1", "b.rs", "beta", NodeType::Function);
        let c = GraphNode::new("t1", "c.rs", "gamma", NodeType::Struct);
        store.upsert_nodes(&[a.clone(), b.clone(), c.clone()]).await.unwrap();

        let e1 = GraphEdge::new("t1", &a.node_id, &b.node_id, EdgeType::Calls, "a.rs");
        let e2 = GraphEdge::new("t1", &b.node_id, &c.node_id, EdgeType::UsesType, "b.rs");
        store.insert_edges(&[e1, e2]).await.unwrap();
    }

    #[tokio::test]
    async fn test_export_nodes_sqlite_all() {
        let pool = setup_pool().await;
        seed_graph(&pool).await;

        let nodes = export_nodes_sqlite(&pool, None).await.unwrap();
        assert_eq!(nodes.len(), 3);
    }

    #[tokio::test]
    async fn test_export_nodes_sqlite_filtered() {
        let pool = setup_pool().await;
        seed_graph(&pool).await;

        // Add a node for a different tenant
        let store = SqliteGraphStore::new(pool.clone());
        let d = GraphNode::new("t2", "d.rs", "delta", NodeType::Function);
        store.upsert_nodes(&[d]).await.unwrap();

        let nodes = export_nodes_sqlite(&pool, Some("t1")).await.unwrap();
        assert_eq!(nodes.len(), 3);

        let nodes_all = export_nodes_sqlite(&pool, None).await.unwrap();
        assert_eq!(nodes_all.len(), 4);
    }

    #[tokio::test]
    async fn test_export_edges_sqlite() {
        let pool = setup_pool().await;
        seed_graph(&pool).await;

        let edges = export_edges_sqlite(&pool, None).await.unwrap();
        assert_eq!(edges.len(), 2);
    }

    #[tokio::test]
    async fn test_export_snapshot() {
        let pool = setup_pool().await;
        seed_graph(&pool).await;

        let snapshot = export_sqlite(&pool, None).await.unwrap();
        assert_eq!(snapshot.nodes.len(), 3);
        assert_eq!(snapshot.edges.len(), 2);
    }

    #[tokio::test]
    async fn test_import_to_store() {
        let source_pool = setup_pool().await;
        seed_graph(&source_pool).await;

        let snapshot = export_sqlite(&source_pool, None).await.unwrap();

        // Import to a fresh store
        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        let report = import_to_store(&snapshot, &target, 2).await.unwrap();

        assert_eq!(report.nodes_exported, 3);
        assert_eq!(report.edges_exported, 2);
        assert_eq!(report.nodes_imported, 3);
        assert_eq!(report.edges_imported, 2);
        assert!(report.nodes_match);
        assert!(report.edges_match);
        assert!(report.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_migrate_sqlite_to_sqlite() {
        let source_pool = setup_pool().await;
        seed_graph(&source_pool).await;

        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        let report =
            migrate_from_sqlite(&source_pool, &target, None, 100).await.unwrap();

        assert!(report.nodes_match);
        assert!(report.edges_match);
        assert_eq!(report.tenants.len(), 1);
        assert!(report.tenants.contains(&"t1".to_string()));
    }

    #[tokio::test]
    async fn test_migrate_filtered_tenant() {
        let source_pool = setup_pool().await;
        seed_graph(&source_pool).await;

        // Add extra tenant
        let store = SqliteGraphStore::new(source_pool.clone());
        let d = GraphNode::new("t2", "d.rs", "delta", NodeType::Function);
        store.upsert_nodes(&[d]).await.unwrap();

        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        let report =
            migrate_from_sqlite(&source_pool, &target, Some("t1"), 100)
                .await
                .unwrap();

        assert_eq!(report.nodes_imported, 3); // only t1 nodes
        assert!(report.nodes_match);
    }

    #[tokio::test]
    async fn test_validate_migration() {
        let source_pool = setup_pool().await;
        seed_graph(&source_pool).await;

        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        // Migrate
        migrate_from_sqlite(&source_pool, &target, None, 100)
            .await
            .unwrap();

        // Validate
        let valid =
            validate_migration(&source_pool, &target, None).await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_validate_mismatch() {
        let source_pool = setup_pool().await;
        seed_graph(&source_pool).await;

        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        // Don't migrate — target is empty
        let valid =
            validate_migration(&source_pool, &target, None).await.unwrap();
        assert!(!valid);
    }

    #[tokio::test]
    async fn test_export_empty_graph() {
        let pool = setup_pool().await;

        let snapshot = export_sqlite(&pool, None).await.unwrap();
        assert!(snapshot.nodes.is_empty());
        assert!(snapshot.edges.is_empty());
    }

    #[tokio::test]
    async fn test_import_empty_snapshot() {
        let pool = setup_pool().await;
        let target = SqliteGraphStore::new(pool.clone());

        let snapshot = GraphSnapshot {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        let report = import_to_store(&snapshot, &target, 100).await.unwrap();
        assert_eq!(report.nodes_imported, 0);
        assert_eq!(report.edges_imported, 0);
        assert!(report.nodes_match);
        assert!(report.edges_match);
    }

    #[tokio::test]
    async fn test_import_batching() {
        let source_pool = setup_pool().await;

        // Create 10 nodes
        let store = SqliteGraphStore::new(source_pool.clone());
        let mut nodes = Vec::new();
        for i in 0..10 {
            let n = GraphNode::new(
                "t1",
                &format!("{}.rs", i),
                &format!("fn_{}", i),
                NodeType::Function,
            );
            nodes.push(n);
        }
        store.upsert_nodes(&nodes).await.unwrap();

        let snapshot = export_sqlite(&source_pool, None).await.unwrap();
        assert_eq!(snapshot.nodes.len(), 10);

        // Import with batch size 3 (should do 4 batches: 3+3+3+1)
        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        let report = import_to_store(&snapshot, &target, 3).await.unwrap();
        assert_eq!(report.nodes_imported, 10);
        assert!(report.nodes_match);
    }

    #[tokio::test]
    async fn test_node_type_preservation() {
        let pool = setup_pool().await;
        let store = SqliteGraphStore::new(pool.clone());

        let nodes = vec![
            GraphNode::new("t1", "a.rs", "MyStruct", NodeType::Struct),
            GraphNode::new("t1", "a.rs", "MyTrait", NodeType::Trait),
            GraphNode::new("t1", "a.rs", "MyEnum", NodeType::Enum),
        ];
        store.upsert_nodes(&nodes).await.unwrap();

        let exported = export_nodes_sqlite(&pool, None).await.unwrap();
        assert_eq!(exported.len(), 3);

        let types: Vec<NodeType> = exported.iter().map(|n| n.symbol_type).collect();
        assert!(types.contains(&NodeType::Struct));
        assert!(types.contains(&NodeType::Trait));
        assert!(types.contains(&NodeType::Enum));
    }

    #[tokio::test]
    async fn test_edge_type_preservation() {
        let pool = setup_pool().await;
        let store = SqliteGraphStore::new(pool.clone());

        let a = GraphNode::new("t1", "a.rs", "a", NodeType::Function);
        let b = GraphNode::new("t1", "b.rs", "b", NodeType::Function);
        store.upsert_nodes(&[a.clone(), b.clone()]).await.unwrap();

        let edges = vec![
            GraphEdge::new("t1", &a.node_id, &b.node_id, EdgeType::Calls, "a.rs"),
            GraphEdge::new("t1", &a.node_id, &b.node_id, EdgeType::Imports, "a.rs"),
        ];
        store.insert_edges(&edges).await.unwrap();

        let exported = export_edges_sqlite(&pool, None).await.unwrap();
        assert_eq!(exported.len(), 2);

        let types: Vec<EdgeType> = exported.iter().map(|e| e.edge_type).collect();
        assert!(types.contains(&EdgeType::Calls));
        assert!(types.contains(&EdgeType::Imports));
    }

    #[tokio::test]
    async fn test_migration_report_tenants() {
        let pool = setup_pool().await;
        let store = SqliteGraphStore::new(pool.clone());

        // Two tenants
        let a = GraphNode::new("t1", "a.rs", "a", NodeType::Function);
        let b = GraphNode::new("t2", "b.rs", "b", NodeType::Function);
        store.upsert_nodes(&[a, b]).await.unwrap();

        let snapshot = export_sqlite(&pool, None).await.unwrap();
        let target_pool = setup_pool().await;
        let target = SqliteGraphStore::new(target_pool.clone());

        let report = import_to_store(&snapshot, &target, 100).await.unwrap();
        assert_eq!(report.tenants.len(), 2);
    }

    #[tokio::test]
    async fn test_default_batch_size() {
        let pool = setup_pool().await;
        let target = SqliteGraphStore::new(pool.clone());

        let snapshot = GraphSnapshot {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        // batch_size=0 should use default
        let report = import_to_store(&snapshot, &target, 0).await.unwrap();
        assert!(report.nodes_match);
    }
}
