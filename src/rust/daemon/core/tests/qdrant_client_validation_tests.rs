//! Qdrant client operations validation tests
//!
//! This module provides comprehensive validation of Qdrant client operations
//! These tests validate the Rust daemon's integration with Qdrant for all
//! vector database operations required by subtask 243.5

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

use workspace_qdrant_core::storage::{
    StorageClient, StorageConfig, DocumentPoint, SearchParams,
    HybridSearchMode, TransportMode,
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
    payload.insert("content".to_string(), serde_json::Value::String(content.to_string()));
    payload.insert("chunk_index".to_string(), serde_json::Value::Number(0.into()));
    payload.insert("file_path".to_string(), serde_json::Value::String(format!("/test/{}.txt", id)));
    payload.insert("timestamp".to_string(), serde_json::Value::String(
        chrono::Utc::now().to_rfc3339()
    ));

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
        },
        Err(e) => {
            tracing::warn!("Failed to connect to Qdrant ({}), skipping test", e);
            return;
        }
    }

    let collection_name = format!("test_vectors_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collection
    let create_result = client.create_collection(&collection_name, Some(384), None).await;
    assert!(create_result.is_ok(), "Collection creation should succeed: {:?}", create_result);

    // Test single point insertion
    let test_doc = create_test_document("vector_test_1", "Test document for vector insertion");
    let insert_result = client.insert_point(&collection_name, test_doc.clone()).await;
    assert!(insert_result.is_ok(), "Vector insertion should succeed: {:?}", insert_result);

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
    assert!(search_result.is_ok(), "Vector search should succeed: {:?}", search_result);

    let results = search_result.unwrap();
    assert!(!results.is_empty(), "Should find the inserted vector");
    assert_eq!(results[0].id, "vector_test_1", "Should retrieve the correct document");

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
    let create_result = client.create_collection(&collection_name, Some(384), None).await;
    assert!(create_result.is_ok(), "Collection creation should succeed");

    // Test collection existence check
    let exists = client.collection_exists(&collection_name).await
        .expect("Should check collection existence");
    assert!(exists, "Collection should exist after creation");

    // Test large vector collection
    let large_collection = format!("{}_large", collection_name);
    let large_result = client.create_collection(&large_collection, Some(1536), None).await;
    assert!(large_result.is_ok(), "Large vector collection creation should succeed");

    // Test collection deletion
    let delete_result = client.delete_collection(&large_collection).await;
    assert!(delete_result.is_ok(), "Collection deletion should succeed");

    // Verify collection is deleted
    let exists_after_delete = client.collection_exists(&large_collection).await
        .expect("Should check collection existence");
    assert!(!exists_after_delete, "Collection should not exist after deletion");

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

    let collection_name = format!("test_batch_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collection
    client.create_collection(&collection_name, Some(384), None).await
        .expect("Should create test collection");

    // Create test documents for batch insertion
    let batch_docs: Vec<DocumentPoint> = (0..25)
        .map(|i| create_test_document(
            &format!("batch_doc_{}", i),
            &format!("Batch document number {} with test content", i)
        ))
        .collect();

    // Test batch insertion
    let batch_result = client.insert_points_batch(
        &collection_name,
        batch_docs,
        Some(10) // Small batch size for testing
    ).await;

    assert!(batch_result.is_ok(), "Batch insertion should succeed: {:?}", batch_result);

    let batch_stats = batch_result.unwrap();
    assert_eq!(batch_stats.total_points, 25, "Should process all points");
    assert!(batch_stats.successful >= 20, "Most points should be successfully inserted");
    assert!(batch_stats.throughput > 0.0, "Should have positive throughput");

    tracing::info!("Batch insertion stats: {:?}", batch_stats);

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Test batch search operations
    let search_vector: Vec<f32> = (0..384)
        .map(|i| ((i as f32 * 0.01) % 1.0).sin())
        .collect();

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
    assert!(!results.is_empty(), "Should find results in batch-inserted data");
    assert!(results.len() <= 10, "Should respect search limit");

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
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

    let collection_name = format!("test_search_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collection
    client.create_collection(&collection_name, Some(384), None).await
        .expect("Should create test collection");

    // Insert diverse test documents for search testing
    let search_docs = vec![
        create_test_document("rust_doc", "Rust programming language is memory safe and fast"),
        create_test_document("python_doc", "Python is a high-level programming language for data science"),
        create_test_document("ml_doc", "Machine learning and artificial intelligence applications"),
        create_test_document("web_doc", "Web development with modern JavaScript frameworks"),
        create_test_document("db_doc", "Database systems and vector storage technologies"),
    ];

    for doc in search_docs {
        client.insert_point(&collection_name, doc).await
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
        assert!(result.payload.contains_key("content"), "Payload should contain content");
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
    assert!(threshold_results.is_ok(), "Threshold search should succeed");

    let thresholded = threshold_results.unwrap();
    for result in &thresholded {
        assert!(result.score >= 0.8, "All results should meet score threshold");
    }

    // Test hybrid search mode (even though sparse is not fully implemented)
    let hybrid_search = SearchParams {
        dense_vector: Some(search_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Hybrid {
            dense_weight: 0.7,
            sparse_weight: 0.3
        },
        limit: 5,
        score_threshold: None,
        filter: None,
    };

    let hybrid_results = client.search(&collection_name, hybrid_search).await;
    assert!(hybrid_results.is_ok(), "Hybrid search should succeed");

    let hybrid_res = hybrid_results.unwrap();
    assert!(!hybrid_res.is_empty(), "Hybrid search should return results");

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
    let exists = client.collection_exists(non_existent).await
        .expect("Should check non-existent collection");
    assert!(!exists, "Non-existent collection should return false");

    // Test point insertion on non-existent collection
    let test_doc = create_test_document("error_test", "Error test document");
    let insert_result = client.insert_point(non_existent, test_doc).await;
    assert!(insert_result.is_err(), "Insert on non-existent collection should fail");

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
    assert!(search_result.is_err(), "Search on non-existent collection should fail");

    // Test invalid search parameters
    let invalid_search = SearchParams {
        dense_vector: None, // No vector provided
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 10,
        score_threshold: None,
        filter: None,
    };

    let collection_name = format!("test_errors_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collection for invalid operation tests
    let _ = client.create_collection(&collection_name, Some(384), None).await;

    let invalid_search_result = client.search(&collection_name, invalid_search).await;
    assert!(invalid_search_result.is_err(), "Search without vector should fail");

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
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

    let collection_name = format!("test_pool_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collection
    client.create_collection(&collection_name, Some(384), None).await
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
                &format!("Concurrent document {}", i)
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

    assert!(success_count >= 6, "Most concurrent operations should succeed (got {})", success_count);

    // Test connection statistics
    let stats = client.get_stats().await
        .expect("Should get connection stats");

    assert!(stats.contains_key("total_requests"), "Should track total requests");
    assert!(stats["total_requests"] >= success_count as u64, "Should track request counts");

    tracing::info!("Connection pool stats: {:?}", stats);

    // Cleanup
    let _ = client.delete_collection(&collection_name).await;
}

/// Test multi-tenant architecture patterns
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_multi_tenant_architecture() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping multi-tenant test");
        return;
    }

    let base_name = format!("tenant_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // Create collections for different tenants (simulating project isolation)
    let tenant_collections = vec![
        format!("{}_project_a", base_name),
        format!("{}_project_b", base_name),
        format!("{}_project_c", base_name),
    ];

    // Create all tenant collections
    for collection in &tenant_collections {
        client.create_collection(collection, Some(384), None).await
            .expect("Should create tenant collection");
    }

    // Insert different data in each tenant collection
    for (i, collection) in tenant_collections.iter().enumerate() {
        let tenant_docs: Vec<DocumentPoint> = (0..3)
            .map(|j| create_test_document(
                &format!("tenant_{}_{}", i, j),
                &format!("Tenant {} document {} - isolated data", i, j)
            ))
            .collect();

        let batch_result = client.insert_points_batch(collection, tenant_docs, Some(5)).await;
        assert!(batch_result.is_ok(), "Should insert tenant data: {:?}", batch_result);
    }

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // Verify data isolation between tenants
    let search_vector: Vec<f32> = (0..384)
        .map(|i| ((i as f32 * 0.01) % 1.0).sin())
        .collect();

    for (tenant_id, collection) in tenant_collections.iter().enumerate() {
        let search_params = SearchParams {
            dense_vector: Some(search_vector.clone()),
            sparse_vector: None,
            search_mode: HybridSearchMode::Dense,
            limit: 10,
            score_threshold: None,
            filter: None,
        };

        let results = client.search(collection, search_params).await
            .expect("Should search tenant collection");

        assert!(!results.is_empty(), "Each tenant should have data");

        // Verify all results belong to this tenant
        for result in results {
            let content = result.payload.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            assert!(
                content.contains(&format!("Tenant {}", tenant_id)),
                "Result should belong to correct tenant: {}", content
            );
        }
    }

    // Test cross-tenant isolation by ensuring collections are separate
    for collection in &tenant_collections {
        let exists = client.collection_exists(collection).await
            .expect("Should check collection existence");
        assert!(exists, "Tenant collection should exist");
    }

    // Cleanup all tenant collections
    for collection in &tenant_collections {
        client.delete_collection(collection).await
            .expect("Should delete tenant collection");

        let exists = client.collection_exists(collection).await
            .expect("Should check collection existence");
        assert!(!exists, "Collection should be deleted");
    }
}

/// Integration test for comprehensive Qdrant workflow
#[tokio::test]
#[serial_test::serial]
#[tracing_test::traced_test]
async fn test_comprehensive_workflow() {
    let config = create_test_storage_config();
    let client = StorageClient::with_config(config);

    // Skip if Qdrant is not available
    if client.test_connection().await.unwrap_or(false) == false {
        tracing::warn!("Qdrant not available, skipping comprehensive workflow test");
        return;
    }

    let collection_name = format!("test_workflow_{}", Uuid::new_v4().to_string().replace('-', "_"));

    // 1. Collection lifecycle
    client.create_collection(&collection_name, Some(384), None).await
        .expect("Should create collection");

    let exists = client.collection_exists(&collection_name).await
        .expect("Should check existence");
    assert!(exists, "Collection should exist");

    // 2. Document ingestion workflow
    let documents = vec![
        create_test_document("doc1", "Rust systems programming language"),
        create_test_document("doc2", "Python data science and machine learning"),
        create_test_document("doc3", "JavaScript web development framework"),
        create_test_document("doc4", "Database vector storage technology"),
        create_test_document("doc5", "Artificial intelligence research paper"),
    ];

    // Insert documents in batch
    let batch_result = client.insert_points_batch(&collection_name, documents, Some(3)).await;
    assert!(batch_result.is_ok(), "Batch insertion should succeed");

    // Wait for indexing
    sleep(Duration::from_secs(1)).await;

    // 3. Search and retrieval workflow
    let query_vector: Vec<f32> = (0..384)
        .map(|i| ((i as f32 * 0.015) % 1.0).sin())
        .collect();

    let search_params = SearchParams {
        dense_vector: Some(query_vector),
        sparse_vector: None,
        search_mode: HybridSearchMode::Dense,
        limit: 3,
        score_threshold: Some(0.1),
        filter: None,
    };

    let search_results = client.search(&collection_name, search_params).await;
    assert!(search_results.is_ok(), "Search should succeed");

    let results = search_results.unwrap();
    assert!(!results.is_empty(), "Should find relevant documents");
    assert!(results.len() <= 3, "Should respect search limit");

    // 4. Verify search quality
    for result in &results {
        assert!(!result.id.is_empty(), "Should have document ID");
        assert!(result.score >= 0.1, "Should meet score threshold");
        assert!(result.payload.contains_key("content"), "Should have content");
    }

    // 5. Performance validation
    let stats = client.get_stats().await
        .expect("Should get client stats");

    assert!(stats["total_requests"] > 0, "Should have processed requests");
    tracing::info!("Workflow completed. Stats: {:?}", stats);

    // 6. Cleanup
    client.delete_collection(&collection_name).await
        .expect("Should delete collection");

    let exists_after = client.collection_exists(&collection_name).await
        .expect("Should check existence");
    assert!(!exists_after, "Collection should be deleted");
}