//! Unit tests for graph module: CRUD, recursive CTE queries, impact analysis.
//!
//! Split into focused submodules:
//! - `id_tests`: Node/edge ID determinism, constructors, enum round-trips
//! - `store_tests`: CRUD operations (upsert, insert, delete)
//! - `query_tests`: Traversal, impact analysis, stats, orphan pruning

mod id_tests;
mod query_tests;
mod store_tests;

use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};

use super::*;

const TENANT: &str = "test-tenant";

/// Create an in-memory graph database for testing.
async fn test_store() -> SqliteGraphStore {
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

    // Run schema migration manually (GraphDbManager::migrate_v1 logic)
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

    SqliteGraphStore::new(pool)
}
