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

/// Test search operations with various algorithms
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_search_operations() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping search operations test");
        return;
    }

    let collection_name = format!(
        "test_search_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    // Create collection
    client
        .create_collection(&collection_name, Some(384), None)
        .await
        .expect("Should create test collection");

    // Insert diverse test documents for search testing
    let search_docs = vec![
        create_test_document("rust_doc", "Rust programming language is memory safe and fast"),
        create_test_document(
            "python_doc",
            "Python is a high-level programming language for data science",
        ),
        create_test_document(
            "ml_doc",
            "Machine learning and artificial intelligence applications",
        ),
        create_test_document(
            "web_doc",
            "Web development with modern JavaScript frameworks",
        ),
        create_test_document(
            "db_doc",
            "Database systems and vector storage technologies",
        ),
    ];

    for doc in search_docs {
        client
            .insert_point(&collection_name, doc)
            .await
            .expect("Should insert search test document");
    }

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Test dense vector search with various parameters
    let search_vector: Vec<f32> = (0..384)
        .map(|i| ((i as f32 * 0.02) % 1.0).sin())
        .collect();

    // Test basic dense search
    let dense_search = SearchParams {
        dense_vector: Some(search_vector.clone()),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 3,
        score_threshold: Some(0.0),
        filter: None,
    };

    let dense_results = client.search(&collection_name, dense_search).await;
    assert!(dense_results.is_ok(), "Dense search should succeed");

    let results = dense_results.unwrap();
    assert!(!results.is_empty(), "Dense search should return results");
    assert!(results.len() <= 3, "Should respect limit parameter");

    // Verify result structure and scores
    for result in &results {
        assert!(!result.id.is_empty(), "Result should have valid ID");
        assert!(result.score >= 0.0, "Score should be non-negative");
        assert!(!result.payload.is_empty(), "Result should have payload");
        assert!(
            result.payload.contains_key("content"),
            "Payload should contain content"
        );
    }

    // Test search with high score threshold
    let threshold_search = SearchParams {
        dense_vector: Some(search_vector.clone()),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 10,
        score_threshold: Some(0.8), // High threshold
        filter: None,
    };

    let threshold_results = client.search(&collection_name, threshold_search).await;
    assert!(
        threshold_results.is_ok(),
        "Threshold search should succeed"
    );

    let thresholded = threshold_results.unwrap();
    for result in &thresholded {
        assert!(
            result.score >= 0.8,
            "All results should meet score threshold"
        );
    }

    // Test hybrid search mode (even though sparse is not fully implemented)
    let hybrid_search = SearchParams {
        dense_vector: Some(search_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Hybrid {
            dense_weight: 0.7,
            sparse_weight: 0.3,
        },
        limit: 5,
        score_threshold: None,
        filter: None,
    };

    let hybrid_results = client.search(&collection_name, hybrid_search).await;
    assert!(hybrid_results.is_ok(), "Hybrid search should succeed");

    let hybrid_res = hybrid_results.unwrap();
    assert!(
        !hybrid_res.is_empty(),
        "Hybrid search should return results"
    );

    // Cleanup
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
