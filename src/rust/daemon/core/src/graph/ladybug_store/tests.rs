/// Tests for the LadybugDB graph store.

use std::path::PathBuf;

use crate::graph::{EdgeType, GraphEdge, GraphNode, GraphStore};

use super::config::LadybugConfig;
use super::store::LadybugGraphStore;

#[test]
fn test_escape_cypher() {
    use super::store::escape_cypher;
    assert_eq!(escape_cypher("hello"), "hello");
    assert_eq!(escape_cypher("it's"), "it\\'s");
    assert_eq!(escape_cypher("a'b'c"), "a\\'b\\'c");
}

#[test]
fn test_default_config() {
    let config = LadybugConfig::default();
    assert!(config.db_path.to_string_lossy().contains("graph"));
    assert_eq!(config.max_num_threads, 4);
}

#[test]
fn test_custom_config() {
    let config = LadybugConfig {
        db_path: PathBuf::from("/tmp/test-graph"),
        buffer_pool_size: 512 * 1024 * 1024,
        max_num_threads: 8,
    };
    assert_eq!(config.db_path, PathBuf::from("/tmp/test-graph"));
    assert_eq!(config.buffer_pool_size, 512 * 1024 * 1024);
}

#[tokio::test]
async fn test_ladybug_store_create() {
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join("graph_test"),
        buffer_pool_size: 0,
        max_num_threads: 2,
    };
    let store = LadybugGraphStore::new(config);
    assert!(store.is_ok(), "Should create store: {:?}", store.err());
}

#[tokio::test]
async fn test_ladybug_upsert_and_stats() {
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join("graph_upsert"),
        buffer_pool_size: 0,
        max_num_threads: 2,
    };
    let store = LadybugGraphStore::new(config).unwrap();

    let node = GraphNode::new(
        "test-tenant", "src/main.rs", "main",
        crate::graph::NodeType::Function,
    );
    let result = store.upsert_node(&node).await;
    assert!(result.is_ok(), "upsert failed: {:?}", result.err());

    let stats = store.stats(Some("test-tenant")).await.unwrap();
    assert_eq!(stats.total_nodes, 1);
}

#[tokio::test]
async fn test_ladybug_insert_edge() {
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join("graph_edge"),
        buffer_pool_size: 0,
        max_num_threads: 2,
    };
    let store = LadybugGraphStore::new(config).unwrap();

    let node_a = GraphNode::new(
        "t1", "a.rs", "foo", crate::graph::NodeType::Function,
    );
    let node_b = GraphNode::new(
        "t1", "b.rs", "bar", crate::graph::NodeType::Function,
    );
    store.upsert_nodes(&[node_a.clone(), node_b.clone()]).await.unwrap();

    let edge = GraphEdge::new(
        "t1", &node_a.node_id, &node_b.node_id,
        EdgeType::Calls, "a.rs",
    );
    let result = store.insert_edge(&edge).await;
    assert!(result.is_ok(), "insert_edge failed: {:?}", result.err());
}

#[tokio::test]
async fn test_ladybug_execute_cypher() {
    let tmp = tempfile::tempdir().unwrap();
    let config = LadybugConfig {
        db_path: tmp.path().join("graph_cypher"),
        buffer_pool_size: 0,
        max_num_threads: 2,
    };
    let store = LadybugGraphStore::new(config).unwrap();
    let rows = store.execute_cypher("RETURN 1 + 2 AS result").unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], "3");
}
