//! Qdrant client CRUD validation tests
//!
//! Tests for vector insertion, retrieval, collection management, and batch
//! operations against a live Qdrant instance.

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

/// Test vector insertion and retrieval operations
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_vector_insertion_retrieval() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Test connection first
    match client.test_connection().await {
        Ok(true) => tracing::info!("Connected to test Qdrant instance"),
        Ok(false) => {
            tracing::warn!("Connection test returned false, skipping test");
            return;
        }
        Err(e) => {
            tracing::warn!("Failed to connect to Qdrant ({}), skipping test", e);
            return;
        }
    }

    let collection_name = format!(
        "test_vectors_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    // Create collection
    let create_result = client
        .create_collection(&collection_name, Some(384), None)
        .await;
    assert!(
        create_result.is_ok(),
        "Collection creation should succeed: {:?}",
        create_result
    );

    // Test single point insertion
    let test_doc = create_test_document("vector_test_1", "Test document for vector insertion");
    let insert_result = client
        .insert_point(&collection_name, test_doc.clone())
        .await;
    assert!(
        insert_result.is_ok(),
        "Vector insertion should succeed: {:?}",
        insert_result
    );

    // Wait for indexing
    sleep(Duration::from_millis(500)).await;

    // Test vector retrieval through similarity search
    let search_params = SearchParams {
        dense_vector: Some(test_doc.dense_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 1,
        score_threshold: Some(0.5),
        filter: None,
    };

    let search_result = client.search(&collection_name, search_params).await;
    assert!(
        search_result.is_ok(),
        "Vector search should succeed: {:?}",
        search_result
    );

    let results = search_result.unwrap();
    assert!(!results.is_empty(), "Should find the inserted vector");
    assert_eq!(
        results[0].id, "vector_test_1",
        "Should retrieve the correct document"
    );

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
}

/// Test collection management operations
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_collection_management() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping collection management test");
        return;
    }

    let collection_name = format!("test_mgmt_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Test collection creation with different configurations
    let create_result = client
        .create_collection(&collection_name, Some(384), None)
        .await;
    assert!(create_result.is_ok(), "Collection creation should succeed");

    // Test collection existence check
    let exists = client
        .collection_exists(&collection_name)
        .await
        .expect("Should check collection existence");
    assert!(exists, "Collection should exist after creation");

    // Test large vector collection
    let large_collection = format!("{}_large", collection_name);
    let large_result = client
        .create_collection(&large_collection, Some(1536), None)
        .await;
    assert!(
        large_result.is_ok(),
        "Large vector collection creation should succeed"
    );

    // Test collection deletion
    let delete_result = client.delete_collection(&large_collection).await;
    assert!(delete_result.is_ok(), "Collection deletion should succeed");

    // Verify collection is deleted
    let exists_after_delete = client
        .collection_exists(&large_collection)
        .await
        .expect("Should check collection existence");
    assert!(
        !exists_after_delete,
        "Collection should not exist after deletion"
    );

    // Cleanup main collection
    let _ = client.delete_collection(&collection_name).await;
}

/// Test batch operations for efficiency
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_batch_operations() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping batch operations test");
        return;
    }

    let collection_name = format!(
        "test_batch_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    // Create collection
    client
        .create_collection(&collection_name, Some(384), None)
        .await
        .expect("Should create test collection");

    // Create test documents for batch insertion
    let batch_docs: Vec<DocumentPoint> = (0..25)
        .map(|i| {
            create_test_document(
                &format!("batch_doc_{}", i),
                &format!("Batch document number {} with test content", i),
            )
        })
        .collect();

    // Test batch insertion
    let batch_result = client
        .insert_points_batch(
            &collection_name,
            batch_docs,
            Some(10), // Small batch size for testing
        )
        .await;

    assert!(
        batch_result.is_ok(),
        "Batch insertion should succeed: {:?}",
        batch_result
    );

    let batch_stats = batch_result.unwrap();
    assert_eq!(batch_stats.total_points, 25, "Should process all points");
    assert!(
        batch_stats.successful >= 20,
        "Most points should be successfully inserted"
    );
    assert!(
        batch_stats.throughput > 0.0,
        "Should have positive throughput"
    );

    tracing::info!("Batch insertion stats: {:?}", batch_stats);

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Test batch search operations
    let search_vector: Vec<f32> = (0..384).map(|i| ((i as f32 * 0.01) % 1.0).sin()).collect();

    let search_params = SearchParams {
        dense_vector: Some(search_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 10,
        score_threshold: Some(0.1),
        filter: None,
    };

    let search_result = client.search(&collection_name, search_params).await;
    assert!(search_result.is_ok(), "Batch search should succeed");

    let results = search_result.unwrap();
    assert!(
        !results.is_empty(),
        "Should find results in batch-inserted data"
    );
    assert!(results.len() <= 10, "Should respect search limit");

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
}
