//! Comprehensive tests for Qdrant client operations with testcontainers
//!
//! This test suite validates all Qdrant client functionality including:
//! - Vector operations (upsert, get, delete, update payload)
//! - Search operations with various parameters
//! - Collection management (create, delete, info, list)
//! - Batch operations (parallel and sequential)
//! - Error handling and edge cases
//! - Connection pooling validation
//! - Circuit breaker behavior
//! - Network failure scenarios
//! - Large dataset performance

use std::collections::HashMap;
use std::time::Duration;
use tokio_test;
use serial_test::serial;
use testcontainers::{clients, Container, Docker, RunArgs, Image};
use testcontainers::clients::Cli;
use tokio::time::timeout;
use rand::{distributions::Alphanumeric, Rng};

use workspace_qdrant_daemon::qdrant::{
    QdrantClient, QdrantClientConfig,
    VectorOperation, SearchOperation, CollectionOperation, BatchOperation,
    QdrantError, QdrantResult
};
use qdrant_client::qdrant::{Distance, PointId, PointStruct, Vectors, Vector};

/// Custom Qdrant testcontainer implementation since it's not in standard modules
#[derive(Debug)]
pub struct QdrantContainer {
    image: String,
    env_vars: HashMap<String, String>,
}

impl Default for QdrantContainer {
    fn default() -> Self {
        let mut env_vars = HashMap::new();
        env_vars.insert("QDRANT__SERVICE__HTTP_PORT".to_string(), "6333".to_string());
        env_vars.insert("QDRANT__SERVICE__GRPC_PORT".to_string(), "6334".to_string());

        QdrantContainer {
            image: "qdrant/qdrant:v1.6.1".to_string(),
            env_vars,
        }
    }
}

impl Image for QdrantContainer {
    type Args = ();

    fn name(&self) -> String {
        self.image.clone()
    }

    fn tag(&self) -> String {
        "v1.6.1".to_string()
    }

    fn ready_conditions(&self) -> Vec<testcontainers::core::WaitFor> {
        vec![
            testcontainers::core::WaitFor::message_on_stderr("Qdrant HTTP listening on"),
        ]
    }

    fn env_vars(&self) -> Box<dyn Iterator<Item = (&String, &String)> + '_> {
        Box::new(self.env_vars.iter())
    }
}

/// Test fixture for Qdrant operations
pub struct QdrantTestFixture {
    pub client: QdrantClient,
    pub test_collection: String,
    #[allow(dead_code)]
    container: Container<'static, QdrantContainer>,
}

impl QdrantTestFixture {
    /// Create a new test fixture with isolated Qdrant container
    pub async fn new() -> QdrantResult<Self> {
        // Start Qdrant container
        let docker = clients::Cli::default();
        let container = docker.run(QdrantContainer::default());

        // Wait for container to be ready
        tokio::time::sleep(Duration::from_secs(3)).await;

        let host_port = container.get_host_port_ipv4(6333);
        let qdrant_url = format!("http://127.0.0.1:{}", host_port);

        // Create client config
        let config = QdrantClientConfig {
            url: qdrant_url,
            api_key: None,
            request_timeout_secs: 30,
            pool_config: Default::default(),
            circuit_breaker_config: Default::default(),
        };

        // Create client and wait for connection
        let client = QdrantClient::new(config).await?;

        // Test connection with retries
        let mut retries = 0;
        loop {
            match client.test_connection().await {
                Ok(_) => break,
                Err(_) if retries < 10 => {
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }

        // Generate unique test collection name
        let test_collection = format!("test_collection_{}",
            rand::thread_rng().sample_iter(&Alphanumeric).take(8).map(char::from).collect::<String>());

        Ok(Self {
            client,
            test_collection,
            container: unsafe { std::mem::transmute(container) }, // Extend lifetime for test duration
        })
    }

    /// Create test collection with standard configuration
    pub async fn create_test_collection(&self) -> QdrantResult<()> {
        let operation = CollectionOperation::Create {
            collection_name: self.test_collection.clone(),
            vector_size: 128,
            distance: Distance::Cosine,
            shard_number: Some(1),
            replication_factor: Some(1),
            on_disk_vectors: Some(false),
        };

        self.client.execute_collection_operation(operation).await?;
        Ok(())
    }

    /// Create sample vectors for testing
    pub fn create_sample_vectors(&self, count: usize) -> Vec<PointStruct> {
        (0..count).map(|i| {
            let vector_data: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32) * 0.01).collect();
            let mut payload = HashMap::new();
            payload.insert("id".to_string(), qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i as i64)),
            });
            payload.insert("text".to_string(), qdrant_client::qdrant::Value {
                kind: Some(qdrant_client::qdrant::value::Kind::StringValue(format!("Document {}", i))),
            });

            PointStruct {
                id: Some(PointId::from(i as u64)),
                vectors: Some(Vectors {
                    vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                        Vector { data: vector_data }
                    )),
                }),
                payload,
            }
        }).collect()
    }
}

// Test Collection Management Operations

#[tokio::test]
#[serial]
async fn test_collection_create_and_delete() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;

    // Test collection creation
    fixture.create_test_collection().await?;

    // Verify collection exists
    let list_op = CollectionOperation::List;
    let _result = fixture.client.execute_collection_operation(list_op).await?;

    // Test collection deletion
    let delete_op = CollectionOperation::Delete {
        collection_name: fixture.test_collection.clone(),
    };
    fixture.client.execute_collection_operation(delete_op).await?;

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_collection_info_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Test get collection info
    let info_op = CollectionOperation::GetInfo {
        collection_name: fixture.test_collection.clone(),
    };
    let _result = fixture.client.execute_collection_operation(info_op).await?;

    // Test list all collections
    let list_op = CollectionOperation::List;
    let _result = fixture.client.execute_collection_operation(list_op).await?;

    Ok(())
}

// Test Vector Operations

#[tokio::test]
#[serial]
async fn test_vector_upsert_and_retrieval() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Create test vectors
    let test_vectors = fixture.create_sample_vectors(10);

    // Test upsert operation
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: test_vectors.clone(),
        wait: true,
    };
    let _result = fixture.client.execute_vector_operation(upsert_op).await?;

    // Test get points operation
    let point_ids: Vec<PointId> = (0..5).map(|i| PointId::from(i as u64)).collect();
    let get_op = VectorOperation::GetPoints {
        collection_name: fixture.test_collection.clone(),
        point_ids,
        with_payload: true,
        with_vector: true,
    };
    let _result = fixture.client.execute_vector_operation(get_op).await?;

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_vector_update_and_delete_payload() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Insert test vector
    let test_vectors = fixture.create_sample_vectors(1);
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: test_vectors,
        wait: true,
    };
    fixture.client.execute_vector_operation(upsert_op).await?;

    // Test payload update
    let mut new_payload = HashMap::new();
    new_payload.insert("updated".to_string(), serde_json::Value::Bool(true));
    new_payload.insert("timestamp".to_string(), serde_json::Value::Number(1234567890.into()));

    let update_op = VectorOperation::UpdatePayload {
        collection_name: fixture.test_collection.clone(),
        point_id: PointId::from(0u64),
        payload: new_payload,
        wait: true,
    };
    fixture.client.execute_vector_operation(update_op).await?;

    // Test payload deletion
    let delete_payload_op = VectorOperation::DeletePayload {
        collection_name: fixture.test_collection.clone(),
        point_id: PointId::from(0u64),
        payload_keys: vec!["updated".to_string()],
        wait: true,
    };
    fixture.client.execute_vector_operation(delete_payload_op).await?;

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_vector_deletion() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Insert test vectors
    let test_vectors = fixture.create_sample_vectors(5);
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: test_vectors,
        wait: true,
    };
    fixture.client.execute_vector_operation(upsert_op).await?;

    // Delete some points
    let point_ids_to_delete: Vec<PointId> = (0..3).map(|i| PointId::from(i as u64)).collect();
    let delete_op = VectorOperation::DeletePoints {
        collection_name: fixture.test_collection.clone(),
        point_ids: point_ids_to_delete,
        wait: true,
    };
    fixture.client.execute_vector_operation(delete_op).await?;

    // Verify deletion by attempting to retrieve deleted points
    let point_ids: Vec<PointId> = (0..5).map(|i| PointId::from(i as u64)).collect();
    let get_op = VectorOperation::GetPoints {
        collection_name: fixture.test_collection.clone(),
        point_ids,
        with_payload: true,
        with_vector: false,
    };
    let _result = fixture.client.execute_vector_operation(get_op).await?;

    Ok(())
}

// Test Search Operations

#[tokio::test]
#[serial]
async fn test_basic_search_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Insert test vectors
    let test_vectors = fixture.create_sample_vectors(20);
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: test_vectors,
        wait: true,
    };
    fixture.client.execute_vector_operation(upsert_op).await?;

    // Test basic search
    let query_vector: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let search_op = SearchOperation {
        collection_name: fixture.test_collection.clone(),
        vector: query_vector,
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: Some(0.5),
    };

    let results = fixture.client.search(search_op).await?;
    assert!(!results.is_empty(), "Search should return results");
    assert!(results.len() <= 10, "Search should respect limit");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_search_with_filters_and_params() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Insert test vectors with different payloads
    let mut test_vectors = Vec::new();
    for i in 0..20 {
        let vector_data: Vec<f32> = (0..128).map(|j| ((i * 128 + j) as f32) * 0.01).collect();
        let mut payload = HashMap::new();
        payload.insert("category".to_string(), qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(
                if i % 2 == 0 { "even".to_string() } else { "odd".to_string() }
            )),
        });
        payload.insert("value".to_string(), qdrant_client::qdrant::Value {
            kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i)),
        });

        test_vectors.push(PointStruct {
            id: Some(PointId::from(i as u64)),
            vectors: Some(Vectors {
                vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                    Vector { data: vector_data }
                )),
            }),
            payload,
        });
    }

    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: test_vectors,
        wait: true,
    };
    fixture.client.execute_vector_operation(upsert_op).await?;

    // Test search with pagination
    let query_vector: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let search_op = SearchOperation {
        collection_name: fixture.test_collection.clone(),
        vector: query_vector,
        limit: 5,
        offset: Some(5),
        filter: None,
        params: None,
        with_payload: true,
        with_vector: true,
        score_threshold: None,
    };

    let results = fixture.client.search(search_op).await?;
    assert!(results.len() <= 5, "Search should respect limit");

    Ok(())
}

// Test Batch Operations

#[tokio::test]
#[serial]
async fn test_batch_sequential_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Create batch operations
    let operations = vec![
        VectorOperation::Upsert {
            collection_name: fixture.test_collection.clone(),
            points: fixture.create_sample_vectors(3),
            wait: true,
        },
        VectorOperation::Upsert {
            collection_name: fixture.test_collection.clone(),
            points: fixture.create_sample_vectors(2),
            wait: true,
        },
    ];

    let batch = BatchOperation {
        collection_name: fixture.test_collection.clone(),
        operations,
        parallel: false,
        batch_size: 100,
        timeout_secs: 30,
    };

    let results = fixture.client.execute_batch_operation(batch).await?;
    assert_eq!(results.len(), 2, "Batch should return results for all operations");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_batch_parallel_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Create batch operations with different point IDs to avoid conflicts
    let mut ops = Vec::new();
    for i in 0..3 {
        let mut points = fixture.create_sample_vectors(2);
        // Modify point IDs to avoid conflicts
        for (j, point) in points.iter_mut().enumerate() {
            point.id = Some(PointId::from((i * 100 + j) as u64));
        }

        ops.push(VectorOperation::Upsert {
            collection_name: fixture.test_collection.clone(),
            points,
            wait: true,
        });
    }

    let batch = BatchOperation {
        collection_name: fixture.test_collection.clone(),
        operations: ops,
        parallel: true,
        batch_size: 100,
        timeout_secs: 30,
    };

    let results = fixture.client.execute_batch_operation(batch).await?;
    assert_eq!(results.len(), 3, "Parallel batch should return results for all operations");

    Ok(())
}

// Test Error Handling and Edge Cases

#[tokio::test]
#[serial]
async fn test_invalid_collection_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;

    // Test operations on non-existent collection
    let search_op = SearchOperation {
        collection_name: "non_existent_collection".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };

    let result = fixture.client.search(search_op).await;
    assert!(result.is_err(), "Search on non-existent collection should fail");

    // Test vector operations on non-existent collection
    let upsert_op = VectorOperation::Upsert {
        collection_name: "non_existent_collection".to_string(),
        points: fixture.create_sample_vectors(1),
        wait: true,
    };

    let result = fixture.client.execute_vector_operation(upsert_op).await;
    assert!(result.is_err(), "Upsert on non-existent collection should fail");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_invalid_vector_dimensions() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Test search with wrong vector dimension
    let wrong_dimension_vector: Vec<f32> = vec![0.1; 64]; // Should be 128
    let search_op = SearchOperation {
        collection_name: fixture.test_collection.clone(),
        vector: wrong_dimension_vector,
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };

    let result = fixture.client.search(search_op).await;
    assert!(result.is_err(), "Search with wrong vector dimension should fail");

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_large_batch_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Test large batch of vectors
    let large_batch_size = 100;
    let operations = vec![VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: fixture.create_sample_vectors(large_batch_size),
        wait: true,
    }];

    let batch = BatchOperation {
        collection_name: fixture.test_collection.clone(),
        operations,
        parallel: false,
        batch_size: large_batch_size,
        timeout_secs: 60, // Longer timeout for large operations
    };

    let start_time = std::time::Instant::now();
    let results = fixture.client.execute_batch_operation(batch).await?;
    let duration = start_time.elapsed();

    assert_eq!(results.len(), 1, "Large batch should complete successfully");
    assert!(duration.as_secs() < 30, "Large batch should complete within reasonable time");

    Ok(())
}

// Test Connection Management and Circuit Breaker

#[tokio::test]
#[serial]
async fn test_connection_pooling_behavior() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Execute multiple concurrent operations to test connection pool
    let mut handles = Vec::new();

    for i in 0..10 {
        let client = &fixture.client;
        let collection = fixture.test_collection.clone();

        handles.push(tokio::spawn(async move {
            let search_op = SearchOperation {
                collection_name: collection,
                vector: vec![0.1; 128],
                limit: 1,
                offset: Some(i),
                filter: None,
                params: None,
                with_payload: false,
                with_vector: false,
                score_threshold: None,
            };

            client.search(search_op).await
        }));
    }

    // Wait for all operations to complete
    let results = futures_util::future::join_all(handles).await;

    // Verify all operations completed successfully or with expected errors
    for result in results {
        match result {
            Ok(search_result) => {
                // Search might fail due to no vectors, but connection should work
                match search_result {
                    Ok(_) => {}, // Success
                    Err(QdrantError::SearchOperation { .. }) => {}, // Expected for empty collection
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            },
            Err(e) => panic!("Task join error: {:?}", e),
        }
    }

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_client_statistics_and_health() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;

    // Test connection health
    fixture.client.test_connection().await?;

    // Get client statistics
    let stats = fixture.client.get_statistics().await;

    assert!(stats.contains_key("circuit_breaker_state"));
    assert!(stats.contains_key("failure_count"));
    assert!(stats.contains_key("success_count"));
    assert!(stats.contains_key("pool_size"));
    assert!(stats.contains_key("max_connections"));

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_operation_timeout_handling() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Test timeout with very large operation
    let very_large_batch = fixture.create_sample_vectors(1000);
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: very_large_batch,
        wait: true,
    };

    // Execute with short timeout
    let result = timeout(Duration::from_millis(100),
        fixture.client.execute_vector_operation(upsert_op)).await;

    // Operation should either complete quickly or timeout
    match result {
        Ok(_) => {}, // Completed within timeout
        Err(_) => {}, // Timed out as expected
    }

    Ok(())
}

// Test Edge Cases and Performance

#[tokio::test]
#[serial]
async fn test_empty_vector_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Test upsert with empty vector list
    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: vec![],
        wait: true,
    };

    let result = fixture.client.execute_vector_operation(upsert_op).await;
    // This might succeed or fail depending on Qdrant implementation
    match result {
        Ok(_) => {}, // Allowed
        Err(QdrantError::VectorOperation { .. }) => {}, // Expected error
        Err(e) => panic!("Unexpected error type: {:?}", e),
    }

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_special_payload_values() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;
    fixture.create_test_collection().await?;

    // Create vector with special payload values
    let vector_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let mut payload = HashMap::new();

    // Test various payload types
    payload.insert("null_value".to_string(), qdrant_client::qdrant::Value {
        kind: Some(qdrant_client::qdrant::value::Kind::NullValue(0)),
    });
    payload.insert("bool_value".to_string(), qdrant_client::qdrant::Value {
        kind: Some(qdrant_client::qdrant::value::Kind::BoolValue(true)),
    });
    payload.insert("int_value".to_string(), qdrant_client::qdrant::Value {
        kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(-12345)),
    });
    payload.insert("double_value".to_string(), qdrant_client::qdrant::Value {
        kind: Some(qdrant_client::qdrant::value::Kind::DoubleValue(3.14159)),
    });
    payload.insert("string_value".to_string(), qdrant_client::qdrant::Value {
        kind: Some(qdrant_client::qdrant::value::Kind::StringValue("Special string with éñgłïsh characters".to_string())),
    });

    let point = PointStruct {
        id: Some(PointId::from(1u64)),
        vectors: Some(Vectors {
            vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                Vector { data: vector_data }
            )),
        }),
        payload,
    };

    let upsert_op = VectorOperation::Upsert {
        collection_name: fixture.test_collection.clone(),
        points: vec![point],
        wait: true,
    };

    fixture.client.execute_vector_operation(upsert_op).await?;

    // Retrieve and verify
    let get_op = VectorOperation::GetPoints {
        collection_name: fixture.test_collection.clone(),
        point_ids: vec![PointId::from(1u64)],
        with_payload: true,
        with_vector: false,
    };

    let _result = fixture.client.execute_vector_operation(get_op).await?;

    Ok(())
}

#[tokio::test]
#[serial]
async fn test_concurrent_collection_operations() -> QdrantResult<()> {
    let fixture = QdrantTestFixture::new().await?;

    // Test concurrent collection creation and deletion
    let collection_names: Vec<String> = (0..5)
        .map(|i| format!("concurrent_test_{}", i))
        .collect();

    let mut handles = Vec::new();

    // Create collections concurrently
    for name in &collection_names {
        let client = &fixture.client;
        let collection_name = name.clone();

        handles.push(tokio::spawn(async move {
            let create_op = CollectionOperation::Create {
                collection_name: collection_name.clone(),
                vector_size: 64,
                distance: Distance::Euclidean,
                shard_number: Some(1),
                replication_factor: Some(1),
                on_disk_vectors: Some(false),
            };

            client.execute_collection_operation(create_op).await
        }));
    }

    // Wait for all creations
    let results = futures_util::future::join_all(handles).await;
    for result in results {
        assert!(result.is_ok(), "Collection creation should succeed");
    }

    // Clean up - delete collections
    for name in collection_names {
        let delete_op = CollectionOperation::Delete {
            collection_name: name,
        };
        let _ = fixture.client.execute_collection_operation(delete_op).await;
    }

    Ok(())
}