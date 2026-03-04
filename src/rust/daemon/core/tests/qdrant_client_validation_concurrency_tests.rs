//! Qdrant client concurrency and multi-tenant validation tests
//!
//! Tests for connection pooling under concurrent load, multi-tenant
//! architecture patterns, and comprehensive end-to-end workflows
//! against a live Qdrant instance.

use std::collections::HashMap;
use std::sync::Arc;
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

/// Test connection pooling and performance under concurrent load
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_connection_pooling() {
    let config = create_test_storage_config();
    let client = Arc::new(StorageClient::with_config(config));

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping connection pooling test");
        return;
    }

    let collection_name = format!(
        "test_pool_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    // Create collection
    client
        .create_collection(&collection_name, Some(384), None)
        .await
        .expect("Should create test collection");

    // Test concurrent operations to validate connection pooling
    let collection_name_arc = Arc::new(collection_name.clone());
    let mut handles = Vec::new();

    // Spawn multiple concurrent insert operations
    for i in 0..8 {
        let client_clone = Arc::clone(&client);
        let collection_clone = Arc::clone(&collection_name_arc);

        let handle = tokio::spawn(async move {
            let doc = create_test_document(
                &format!("concurrent_{}", i),
                &format!("Concurrent document {}", i),
            );

            client_clone.insert_point(&collection_clone, doc).await
        });

        handles.push(handle);
    }

    // Wait for all operations to complete
    let mut success_count = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(_)) => success_count += 1,
            Ok(Err(e)) => tracing::warn!("Concurrent operation failed: {}", e),
            Err(e) => tracing::error!("Task panicked: {}", e),
        }
    }

    assert!(
        success_count >= 6,
        "Most concurrent operations should succeed (got {})",
        success_count
    );

    // Test connection statistics
    let stats = client
        .get_stats()
        .await
        .expect("Should get connection stats");

    assert!(
        stats.contains_key("total_requests"),
        "Should track total requests"
    );
    assert!(
        stats["total_requests"] >= success_count as u64,
        "Should track request counts"
    );

    tracing::info!("Connection pool stats: {:?}", stats);

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
}

/// Generate a 384-dim test search vector.
fn test_search_vector(scale: f32) -> Vec<f32> {
    (0..384).map(|i| ((i as f32 * scale) % 1.0).sin()).collect()
}

/// Create tenant collections with isolated data, returning collection names.
async fn setup_tenant_collections(client: &StorageClient, count: usize) -> Vec<String> {
    let base_name = format!("tenant_{}", Uuid::new_v4().to_string().replace('-', "_"));
    let collections: Vec<String> = (0..count)
        .map(|i| format!("{}_project_{}", base_name, (b'a' + i as u8) as char))
        .collect();

    for collection in &collections {
        client
            .create_collection(collection, Some(384), None)
            .await
            .expect("Should create tenant collection");
    }

    for (i, collection) in collections.iter().enumerate() {
        let docs: Vec<DocumentPoint> = (0..3)
            .map(|j| {
                create_test_document(
                    &format!("tenant_{}_{}", i, j),
                    &format!("Tenant {} document {} - isolated data", i, j),
                )
            })
            .collect();

        client
            .insert_points_batch(collection, docs, Some(5))
            .await
            .expect("Should insert tenant data");
    }

    sleep(Duration::from_secs(1)).await;
    collections
}

/// Test multi-tenant architecture patterns
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_multi_tenant_architecture() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    if !client.test_connection().await.unwrap_or(false) {
        tracing::warn!("Qdrant not available, skipping multi-tenant test");
        return;
    }

    let tenant_collections = setup_tenant_collections(&client, 3).await;
    let search_vector = test_search_vector(0.01);

    // Verify data isolation between tenants
    for (tenant_id, collection) in tenant_collections.iter().enumerate() {
        let search_params = SearchParams {
            dense_vector: Some(search_vector.clone()),
            sparse_vector: None,
            search_mode: HybridSearchMode::Dense,
            limit: 10,
            score_threshold: None,
            filter: None,
        };

        let results = client
            .search(collection, search_params)
            .await
            .expect("Should search tenant collection");

        assert!(!results.is_empty(), "Each tenant should have data");
        for result in results {
            let content = result.payload.get("content").and_then(|v| v.as_str()).unwrap_or("");
            assert!(
                content.contains(&format!("Tenant {}", tenant_id)),
                "Result should belong to correct tenant: {}", content
            );
        }
    }

    // Cleanup and verify deletion
    for collection in &tenant_collections {
        client.delete_collection(collection).await.expect("Should delete tenant collection");
        assert!(
            !client.collection_exists(collection).await.expect("Should check existence"),
            "Collection should be deleted"
        );
    }
}

/// Create a collection with batch-inserted documents, returning the collection name.
async fn setup_workflow_collection(client: &StorageClient) -> String {
    let collection_name = format!(
        "test_workflow_{}",
        Uuid::new_v4().to_string().replace('-', "_")
    );

    client.create_collection(&collection_name, Some(384), None).await.expect("Should create collection");
    assert!(client.collection_exists(&collection_name).await.expect("Should check existence"));

    let documents = vec![
        create_test_document("doc1", "Rust systems programming language"),
        create_test_document("doc2", "Python data science and machine learning"),
        create_test_document("doc3", "JavaScript web development framework"),
        create_test_document("doc4", "Database vector storage technology"),
        create_test_document("doc5", "Artificial intelligence research paper"),
    ];

    client
        .insert_points_batch(&collection_name, documents, Some(3))
        .await
        .expect("Batch insertion should succeed");

    sleep(Duration::from_secs(1)).await;
    collection_name
}

/// Integration test for comprehensive Qdrant workflow
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_comprehensive_workflow() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    if !client.test_connection().await.unwrap_or(false) {
        tracing::warn!("Qdrant not available, skipping comprehensive workflow test");
        return;
    }

    let collection_name = setup_workflow_collection(&client).await;

    // Search and retrieval
    let search_params = SearchParams {
        dense_vector: Some(test_search_vector(0.015)),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 3,
        score_threshold: Some(0.1),
        filter: None,
    };

    let results = client.search(&collection_name, search_params).await.expect("Search should succeed");
    assert!(!results.is_empty(), "Should find relevant documents");
    assert!(results.len() <= 3, "Should respect search limit");

    for result in &results {
        assert!(!result.id.is_empty(), "Should have document ID");
        assert!(result.score >= 0.1, "Should meet score threshold");
        assert!(result.payload.contains_key("content"), "Should have content");
    }

    let stats = client.get_stats().await.expect("Should get client stats");
    assert!(stats["total_requests"] > 0, "Should have processed requests");
    tracing::info!("Workflow completed. Stats: {:?}", stats);

    client.delete_collection(&collection_name).await.expect("Should delete collection");
    assert!(!client.collection_exists(&collection_name).await.expect("Should check existence"));
}
