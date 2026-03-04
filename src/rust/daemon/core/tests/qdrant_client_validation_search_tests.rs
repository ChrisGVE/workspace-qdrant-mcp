//! Qdrant client search and error handling validation tests
//!
//! Tests for search operations with various algorithms and error handling
//! for failure scenarios against a live Qdrant instance.

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use workspace_qdrant_core::storage::{
    DocumentPoint, HybridSearchMode, SearchParams, StorageClient, StorageConfig, TransportMode,
};

/// Test Qdrant connection configuration
fn create_test_storage_config() -> StorageConfig {
    StorageConfig {
        url: std::env::var("TEST_QDRANT_URL")
            .unwrap_or_else(|_| "http://localhost:6333".to_string()),
        api_key: std::env::var("TEST_QDRANT_API_KEY").ok(),
        timeout_ms: 10000,
        max_retries: 3,
        retry_delay_ms: 500,
        transport: TransportMode::Http, // Use HTTP for testing
        pool_size: 5,
        tls: false,
        dense_vector_size: 384, // Use smaller vectors for testing
        sparse_vector_size: None,
        check_compatibility: false, // Disable for clean output
        ..Default::default()
    }
}

/// Generate test document point
fn create_test_document(id: &str, content: &str) -> DocumentPoint {
    let mut payload = HashMap::new();
    payload.insert(
        "content".to_string(),
        serde_json::Value::String(content.to_string()),
    );
    payload.insert(
        "chunk_index".to_string(),
        serde_json::Value::Number(0.into()),
    );
    payload.insert(
        "file_path".to_string(),
        serde_json::Value::String(format!("/test/{}.txt", id)),
    );
    payload.insert(
        "timestamp".to_string(),
        serde_json::Value::String(chrono::Utc::now().to_rfc3339()),
    );

    // Generate a simple test vector (normalized)
    let vector: Vec<f32> = (0..384)
        .map(|i| ((i as f32 * 0.01 + content.len() as f32 * 0.001) % 1.0).sin())
        .collect();

    DocumentPoint {
        id: id.to_string(),
        dense_vector: vector,
        sparse_vector: None,
        payload,
    }
}

/// Create a test collection with diverse search documents, returning the collection name.
async fn setup_search_collection(client: &StorageClient) -> String {
    let collection_name = format!(
        "test_search_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    client
        .create_collection(&collection_name, Some(384), None)
        .await
        .expect("Should create test collection");

    let docs = vec![
        create_test_document("rust_doc", "Rust programming language is memory safe and fast"),
        create_test_document("python_doc", "Python is a high-level programming language for data science"),
        create_test_document("ml_doc", "Machine learning and artificial intelligence applications"),
        create_test_document("web_doc", "Web development with modern JavaScript frameworks"),
        create_test_document("db_doc", "Database systems and vector storage technologies"),
    ];

    for doc in docs {
        client.insert_point(&collection_name, doc).await.expect("Should insert doc");
    }

    sleep(Duration::from_secs(1)).await;
    collection_name
}

/// Generate a 384-dim test search vector.
fn test_search_vector(scale: f32) -> Vec<f32> {
    (0..384).map(|i| ((i as f32 * scale) % 1.0).sin()).collect()
}

/// Build dense search params with given limit and threshold.
fn dense_search_params(vector: Vec<f32>, limit: usize, threshold: Option<f32>) -> SearchParams {
    SearchParams {
        dense_vector: Some(vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit,
        score_threshold: threshold,
        filter: None,
    }
}

/// Test search operations with various algorithms
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_search_operations() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    if !client.test_connection().await.unwrap_or(false) {
        tracing::warn!("Qdrant not available, skipping search operations test");
        return;
    }

    let collection_name = setup_search_collection(&client).await;
    let search_vector = test_search_vector(0.02);

    // Test basic dense search
    let results = client
        .search(&collection_name, dense_search_params(search_vector.clone(), 3, Some(0.0)))
        .await
        .expect("Dense search should succeed");

    assert!(!results.is_empty(), "Dense search should return results");
    assert!(results.len() <= 3, "Should respect limit parameter");
    for result in &results {
        assert!(!result.id.is_empty(), "Result should have valid ID");
        assert!(result.score >= 0.0, "Score should be non-negative");
        assert!(result.payload.contains_key("content"), "Payload should contain content");
    }

    // Test search with high score threshold
    let thresholded = client
        .search(&collection_name, dense_search_params(search_vector.clone(), 10, Some(0.8)))
        .await
        .expect("Threshold search should succeed");
    for result in &thresholded {
        assert!(result.score >= 0.8, "All results should meet score threshold");
    }

    // Test hybrid search mode
    let hybrid_search = SearchParams {
        dense_vector: Some(search_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Hybrid { dense_weight: 0.7, sparse_weight: 0.3 },
        limit: 5,
        score_threshold: None,
        filter: None,
    };
    let hybrid_res = client.search(&collection_name, hybrid_search).await.expect("Hybrid search should succeed");
    assert!(!hybrid_res.is_empty(), "Hybrid search should return results");

    let _ = client.delete_collection(&collection_name).await;
}

/// Test error handling for various failure scenarios
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_error_handling() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping error handling test");
        return;
    }

    // Test operations on non-existent collection
    let non_existent = "non_existent_collection_12345";

    // Test collection existence check
    let exists = client
        .collection_exists(non_existent)
        .await
        .expect("Should check non-existent collection");
    assert!(!exists, "Non-existent collection should return false");

    // Test point insertion on non-existent collection
    let test_doc = create_test_document("error_test", "Error test document");
    let insert_result = client.insert_point(non_existent, test_doc).await;
    assert!(
        insert_result.is_err(),
        "Insert on non-existent collection should fail"
    );

    // Test search on non-existent collection
    let search_params = SearchParams {
        dense_vector: Some(vec![0.1; 384]),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 10,
        score_threshold: None,
        filter: None,
    };
    let search_result = client.search(non_existent, search_params).await;
    assert!(
        search_result.is_err(),
        "Search on non-existent collection should fail"
    );

    // Test invalid search parameters
    let invalid_search = SearchParams {
        dense_vector: None, // No vector provided
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 10,
        score_threshold: None,
        filter: None,
    };

    let collection_name = format!(
        "test_errors_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    // Create collection for invalid operation tests
    let _ = client
        .create_collection(&collection_name, Some(384), None)
        .await;

    let invalid_search_result = client.search(&collection_name, invalid_search).await;
    assert!(
        invalid_search_result.is_err(),
        "Search without vector should fail"
    );

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
}
