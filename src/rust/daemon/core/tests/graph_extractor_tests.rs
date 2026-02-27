//! Integration tests for the graph extractor: TextChunk metadata extraction
//! and backend configuration validation.

mod common;

use std::collections::HashMap;

use common::graph_helpers::{create_factory_store, TENANT};
use tempfile::tempdir;
use workspace_qdrant_core::graph::{self, extractor, EdgeType, NodeType};
use workspace_qdrant_core::TextChunk;

// ────────────────────────────────────────────────────────────────────────────
// TextChunk-based extraction
// ────────────────────────────────────────────────────────────────────────────

/// Test the TextChunk-based extraction path (used in the processing pipeline).
#[tokio::test]
async fn test_extract_from_text_chunks() {
    let dir = tempdir().unwrap();
    let store = create_factory_store(dir.path()).await;

    let mut meta_fn = HashMap::new();
    meta_fn.insert("chunk_type".to_string(), "function".to_string());
    meta_fn.insert("symbol_name".to_string(), "calculate".to_string());
    meta_fn.insert("language".to_string(), "rust".to_string());
    meta_fn.insert("start_line".to_string(), "5".to_string());
    meta_fn.insert("end_line".to_string(), "15".to_string());
    meta_fn.insert(
        "signature".to_string(),
        "fn calculate(input: Vec<Data>) -> Result<Report, Error>".to_string(),
    );
    meta_fn.insert("calls".to_string(), "validate,transform".to_string());

    let text_chunk = TextChunk {
        content: "fn calculate(input: Vec<Data>) -> Result<Report, Error> { validate(); transform(); }".to_string(),
        chunk_index: 0,
        start_char: 0,
        end_char: 82,
        metadata: meta_fn,
    };

    let result = extractor::extract_edges_from_text_chunks(&[text_chunk], TENANT, "src/calc.rs");

    // Should produce function node
    let fn_nodes: Vec<_> = result
        .nodes
        .iter()
        .filter(|n| n.symbol_type == NodeType::Function && n.symbol_name == "calculate")
        .collect();
    assert_eq!(
        fn_nodes.len(),
        1,
        "should create a function node for 'calculate'"
    );

    // Should have CALLS edges
    let call_edges: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::Calls)
        .collect();
    assert_eq!(
        call_edges.len(),
        2,
        "should create 2 CALLS edges for validate,transform"
    );

    // Should have USES_TYPE edges from signature
    let type_edges: Vec<_> = result
        .edges
        .iter()
        .filter(|e| e.edge_type == EdgeType::UsesType)
        .collect();
    assert!(
        type_edges.len() >= 2,
        "should create USES_TYPE edges for Vec, Data, Result, Report, Error: got {}",
        type_edges.len()
    );

    // Ingest into store and verify
    store.upsert_nodes(&result.nodes).await.unwrap();
    store.insert_edges(&result.edges).await.unwrap();

    let stats = store.stats(Some(TENANT)).await.unwrap();
    assert!(stats.total_nodes > 0);
    assert!(stats.total_edges > 0);
}

// ────────────────────────────────────────────────────────────────────────────
// Schema and configuration
// ────────────────────────────────────────────────────────────────────────────

/// Graph backend validation.
#[test]
fn test_backend_validation() {
    use workspace_qdrant_core::graph::{GraphBackend, GraphConfig};

    let config = GraphConfig::default();
    assert_eq!(config.backend, GraphBackend::Sqlite);

    assert!(graph::factory::validate_backend(&GraphBackend::Sqlite).is_ok());

    let ladybug_result = graph::factory::validate_backend(&GraphBackend::Ladybug);
    #[cfg(feature = "ladybug")]
    assert!(ladybug_result.is_ok());
    #[cfg(not(feature = "ladybug"))]
    assert!(ladybug_result.is_err());
}
