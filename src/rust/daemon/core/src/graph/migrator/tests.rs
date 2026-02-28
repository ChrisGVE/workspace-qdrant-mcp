use super::*;
use crate::graph::{EdgeType, GraphEdge, GraphNode, NodeType, SqliteGraphStore, GraphStore};
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;

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
