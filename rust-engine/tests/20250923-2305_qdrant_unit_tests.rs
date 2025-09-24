//! Unit tests for Qdrant client components without external dependencies
//!
//! This test suite validates Qdrant client logic without requiring actual Qdrant server:
//! - Configuration validation
//! - Error handling and classification
//! - Circuit breaker state management
//! - Operation validation
//! - Data structure conversions
//! - Timeout and retry logic

use std::collections::HashMap;
use std::time::Duration;
use tokio_test;
use workspace_qdrant_daemon::qdrant::{
    QdrantClientConfig, QdrantError, QdrantResult,
    VectorOperation, SearchOperation, CollectionOperation, BatchOperation
};
use qdrant_client::qdrant::{Distance, PointId, PointStruct, Vectors, Vector};

// Test Configuration and Validation

#[test]
fn test_qdrant_config_validation_success() {
    let config = QdrantClientConfig {
        url: "http://localhost:6333".to_string(),
        api_key: Some("test-key".to_string()),
        request_timeout_secs: 30,
        pool_config: Default::default(),
        circuit_breaker_config: Default::default(),
    };

    assert!(config.validate().is_ok());
}

#[test]
fn test_qdrant_config_validation_invalid_url() {
    let config = QdrantClientConfig {
        url: "invalid-url".to_string(),
        api_key: None,
        request_timeout_secs: 30,
        pool_config: Default::default(),
        circuit_breaker_config: Default::default(),
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_qdrant_config_validation_invalid_timeout() {
    let config = QdrantClientConfig {
        url: "http://localhost:6333".to_string(),
        api_key: None,
        request_timeout_secs: 0, // Invalid timeout
        pool_config: Default::default(),
        circuit_breaker_config: Default::default(),
    };

    assert!(config.validate().is_err());
}

#[test]
fn test_qdrant_config_test_helper() {
    let config = QdrantClientConfig::test_config();
    assert!(config.validate().is_ok());
    assert_eq!(config.url, "http://localhost:6333");
}

// Test Error Types and Classification

#[test]
fn test_error_retryability_classification() {
    // Retryable errors
    assert!(QdrantError::Connection { message: "timeout".to_string() }.is_retryable());
    assert!(QdrantError::ResourceExhausted {
        resource: "pool".to_string(),
        message: "full".to_string()
    }.is_retryable());
    assert!(QdrantError::VectorOperation {
        operation: "upsert".to_string(),
        message: "temporary failure".to_string()
    }.is_retryable());

    // Non-retryable errors
    assert!(!QdrantError::Authentication { message: "invalid key".to_string() }.is_retryable());
    assert!(!QdrantError::Validation { message: "bad input".to_string() }.is_retryable());
    assert!(!QdrantError::Serialization { message: "parse error".to_string() }.is_retryable());

    // Circuit breaker errors should not be retried
    assert!(!QdrantError::CircuitBreakerOpen {
        operation: "search".to_string()
    }.is_retryable());
}

#[test]
fn test_error_display_formatting() {
    let error = QdrantError::SearchOperation {
        message: "Query failed".to_string()
    };
    let formatted = format!("{}", error);
    assert!(formatted.contains("Query failed"));

    let error = QdrantError::CollectionOperation {
        operation: "create".to_string(),
        message: "Already exists".to_string()
    };
    let formatted = format!("{}", error);
    assert!(formatted.contains("create"));
    assert!(formatted.contains("Already exists"));
}

// Test Operation Validation

#[test]
fn test_search_operation_validation() {
    // Valid search operation
    let valid_search = SearchOperation {
        collection_name: "test_collection".to_string(),
        vector: vec![0.1, 0.2, 0.3],
        limit: 10,
        offset: Some(5),
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: Some(0.8),
    };
    assert!(valid_search.validate().is_ok());

    // Invalid search operation - empty collection name
    let invalid_search = SearchOperation {
        collection_name: "".to_string(),
        vector: vec![0.1, 0.2, 0.3],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };
    assert!(invalid_search.validate().is_err());

    // Invalid search operation - empty vector
    let invalid_search = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };
    assert!(invalid_search.validate().is_err());

    // Invalid search operation - limit too high
    let invalid_search = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1, 0.2],
        limit: 10001, // Exceeds reasonable limit
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };
    assert!(invalid_search.validate().is_err());
}

#[test]
fn test_batch_operation_validation() {
    // Valid batch operation
    let valid_batch = BatchOperation {
        collection_name: "test".to_string(),
        operations: vec![
            VectorOperation::Upsert {
                collection_name: "test".to_string(),
                points: create_test_point_structs(1),
                wait: true,
            }
        ],
        parallel: true,
        batch_size: 100,
        timeout_secs: 30,
    };
    assert!(valid_batch.validate().is_ok());

    // Invalid batch - empty operations
    let invalid_batch = BatchOperation {
        collection_name: "test".to_string(),
        operations: vec![],
        parallel: false,
        batch_size: 100,
        timeout_secs: 30,
    };
    assert!(invalid_batch.validate().is_err());

    // Invalid batch - too many operations
    let mut large_ops = Vec::new();
    for i in 0..1001 {
        large_ops.push(VectorOperation::Upsert {
            collection_name: "test".to_string(),
            points: create_test_point_structs(1),
            wait: true,
        });
    }
    let invalid_batch = BatchOperation {
        collection_name: "test".to_string(),
        operations: large_ops,
        parallel: true,
        batch_size: 100,
        timeout_secs: 30,
    };
    assert!(invalid_batch.validate().is_err());

    // Invalid batch - zero timeout
    let invalid_batch = BatchOperation {
        collection_name: "test".to_string(),
        operations: vec![VectorOperation::Upsert {
            collection_name: "test".to_string(),
            points: create_test_point_structs(1),
            wait: true,
        }],
        parallel: false,
        batch_size: 100,
        timeout_secs: 0,
    };
    assert!(invalid_batch.validate().is_err());
}

// Test Vector Operation Types

#[test]
fn test_vector_operation_variants() {
    // Test Upsert operation
    let upsert_op = VectorOperation::Upsert {
        collection_name: "test".to_string(),
        points: create_test_point_structs(3),
        wait: true,
    };
    match upsert_op {
        VectorOperation::Upsert { collection_name, points, wait } => {
            assert_eq!(collection_name, "test");
            assert_eq!(points.len(), 3);
            assert!(wait);
        }
        _ => panic!("Expected Upsert variant"),
    }

    // Test GetPoints operation
    let get_op = VectorOperation::GetPoints {
        collection_name: "test".to_string(),
        point_ids: vec![PointId::from(1u64), PointId::from(2u64)],
        with_payload: true,
        with_vector: false,
    };
    match get_op {
        VectorOperation::GetPoints { collection_name, point_ids, with_payload, with_vector } => {
            assert_eq!(collection_name, "test");
            assert_eq!(point_ids.len(), 2);
            assert!(with_payload);
            assert!(!with_vector);
        }
        _ => panic!("Expected GetPoints variant"),
    }

    // Test DeletePoints operation
    let delete_op = VectorOperation::DeletePoints {
        collection_name: "test".to_string(),
        point_ids: vec![PointId::from(1u64)],
        wait: false,
    };
    match delete_op {
        VectorOperation::DeletePoints { collection_name, point_ids, wait } => {
            assert_eq!(collection_name, "test");
            assert_eq!(point_ids.len(), 1);
            assert!(!wait);
        }
        _ => panic!("Expected DeletePoints variant"),
    }

    // Test UpdatePayload operation
    let mut payload = HashMap::new();
    payload.insert("key".to_string(), serde_json::Value::String("value".to_string()));

    let update_op = VectorOperation::UpdatePayload {
        collection_name: "test".to_string(),
        point_id: PointId::from(1u64),
        payload,
        wait: true,
    };
    match update_op {
        VectorOperation::UpdatePayload { collection_name, point_id, payload, wait } => {
            assert_eq!(collection_name, "test");
            assert!(payload.contains_key("key"));
            assert!(wait);
        }
        _ => panic!("Expected UpdatePayload variant"),
    }

    // Test DeletePayload operation
    let delete_payload_op = VectorOperation::DeletePayload {
        collection_name: "test".to_string(),
        point_id: PointId::from(1u64),
        payload_keys: vec!["old_key".to_string(), "obsolete".to_string()],
        wait: true,
    };
    match delete_payload_op {
        VectorOperation::DeletePayload { collection_name, point_id, payload_keys, wait } => {
            assert_eq!(collection_name, "test");
            assert_eq!(payload_keys.len(), 2);
            assert!(wait);
        }
        _ => panic!("Expected DeletePayload variant"),
    }
}

// Test Collection Operation Types

#[test]
fn test_collection_operation_variants() {
    // Test Create operation
    let create_op = CollectionOperation::Create {
        collection_name: "new_collection".to_string(),
        vector_size: 256,
        distance: Distance::Dot,
        shard_number: Some(2),
        replication_factor: Some(1),
        on_disk_vectors: Some(true),
    };
    match create_op {
        CollectionOperation::Create {
            collection_name,
            vector_size,
            distance,
            shard_number,
            replication_factor,
            on_disk_vectors
        } => {
            assert_eq!(collection_name, "new_collection");
            assert_eq!(vector_size, 256);
            assert_eq!(distance, Distance::Dot);
            assert_eq!(shard_number, Some(2));
            assert_eq!(replication_factor, Some(1));
            assert_eq!(on_disk_vectors, Some(true));
        }
        _ => panic!("Expected Create variant"),
    }

    // Test Delete operation
    let delete_op = CollectionOperation::Delete {
        collection_name: "old_collection".to_string(),
    };
    match delete_op {
        CollectionOperation::Delete { collection_name } => {
            assert_eq!(collection_name, "old_collection");
        }
        _ => panic!("Expected Delete variant"),
    }

    // Test GetInfo operation
    let info_op = CollectionOperation::GetInfo {
        collection_name: "info_collection".to_string(),
    };
    match info_op {
        CollectionOperation::GetInfo { collection_name } => {
            assert_eq!(collection_name, "info_collection");
        }
        _ => panic!("Expected GetInfo variant"),
    }

    // Test List operation
    let list_op = CollectionOperation::List;
    match list_op {
        CollectionOperation::List => {
            // Success - no fields to check
        }
        _ => panic!("Expected List variant"),
    }

    // Test Update operation
    let update_op = CollectionOperation::Update {
        collection_name: "update_collection".to_string(),
        optimizers_config: Some(serde_json::json!({"indexing_threshold": 20000})),
        params: Some(serde_json::json!({"replication_factor": 2})),
    };
    match update_op {
        CollectionOperation::Update { collection_name, optimizers_config, params } => {
            assert_eq!(collection_name, "update_collection");
            assert!(optimizers_config.is_some());
            assert!(params.is_some());
        }
        _ => panic!("Expected Update variant"),
    }

    // Test CreateAlias operation
    let alias_op = CollectionOperation::CreateAlias {
        collection_name: "real_collection".to_string(),
        alias_name: "alias_collection".to_string(),
    };
    match alias_op {
        CollectionOperation::CreateAlias { collection_name, alias_name } => {
            assert_eq!(collection_name, "real_collection");
            assert_eq!(alias_name, "alias_collection");
        }
        _ => panic!("Expected CreateAlias variant"),
    }

    // Test DeleteAlias operation
    let delete_alias_op = CollectionOperation::DeleteAlias {
        alias_name: "old_alias".to_string(),
    };
    match delete_alias_op {
        CollectionOperation::DeleteAlias { alias_name } => {
            assert_eq!(alias_name, "old_alias");
        }
        _ => panic!("Expected DeleteAlias variant"),
    }
}

// Test Data Structure Helpers and Conversions

#[test]
fn test_point_struct_creation() {
    let points = create_test_point_structs(3);
    assert_eq!(points.len(), 3);

    for (i, point) in points.iter().enumerate() {
        assert!(point.id.is_some());
        assert!(point.vectors.is_some());
        assert!(!point.payload.is_empty());

        // Verify vector data structure
        if let Some(vectors) = &point.vectors {
            if let Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(vector)) = &vectors.vectors_options {
                assert_eq!(vector.data.len(), 128); // Default vector size in test helper
            }
        }

        // Verify payload structure
        assert!(point.payload.contains_key("id"));
        assert!(point.payload.contains_key("text"));
    }
}

#[test]
fn test_search_result_structure() {
    // This tests the SearchResult structure used in search operations
    use workspace_qdrant_daemon::qdrant::operations::SearchResult;

    let result = SearchResult {
        id: "test_point_1".to_string(),
        score: 0.95,
        payload: Some({
            let mut payload = HashMap::new();
            payload.insert("category".to_string(), serde_json::Value::String("test".to_string()));
            payload.insert("value".to_string(), serde_json::Value::Number(42.into()));
            payload
        }),
        vector: Some(vec![0.1, 0.2, 0.3, 0.4]),
    };

    assert_eq!(result.id, "test_point_1");
    assert_eq!(result.score, 0.95);
    assert!(result.payload.is_some());
    assert!(result.vector.is_some());

    if let Some(payload) = result.payload {
        assert_eq!(payload.len(), 2);
        assert!(payload.contains_key("category"));
        assert!(payload.contains_key("value"));
    }

    if let Some(vector) = result.vector {
        assert_eq!(vector.len(), 4);
        assert_eq!(vector[0], 0.1);
    }
}

// Test Configuration Defaults and Builders

#[test]
fn test_pool_config_defaults() {
    use workspace_qdrant_daemon::qdrant::config::PoolConfig;

    let default_config = PoolConfig::default();
    assert!(default_config.max_connections > 0);
    assert!(default_config.min_idle_connections >= 0);
    assert!(default_config.connection_timeout_secs > 0);
    assert!(default_config.idle_timeout_secs > 0);
}

#[test]
fn test_circuit_breaker_config_defaults() {
    use workspace_qdrant_daemon::qdrant::config::CircuitBreakerConfig;

    let default_config = CircuitBreakerConfig::default();
    assert!(default_config.failure_threshold > 0);
    assert!(default_config.success_threshold > 0);
    assert!(default_config.timeout_secs > 0);
    assert!(default_config.half_open_timeout_secs > 0);
}

#[test]
fn test_client_config_builder() {
    let config = QdrantClientConfig::builder()
        .url("http://custom-host:8080".to_string())
        .api_key(Some("custom-key".to_string()))
        .request_timeout_secs(45)
        .build();

    assert_eq!(config.url, "http://custom-host:8080");
    assert_eq!(config.api_key, Some("custom-key".to_string()));
    assert_eq!(config.request_timeout_secs, 45);
}

// Test Edge Cases and Boundary Conditions

#[test]
fn test_empty_collection_name_handling() {
    let search_op = SearchOperation {
        collection_name: "".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };

    assert!(search_op.validate().is_err());
}

#[test]
fn test_large_vector_dimension_handling() {
    let large_vector = vec![0.1; 10000]; // Very large vector
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: large_vector,
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };

    // Should validate successfully but might fail at runtime
    assert!(search_op.validate().is_ok());
}

#[test]
fn test_negative_limit_and_offset_handling() {
    // Test with limit of 0 (edge case)
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1; 128],
        limit: 0, // Edge case
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };
    assert!(search_op.validate().is_err());

    // Test with very large offset
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: Some(1000000), // Very large offset
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: None,
    };
    // Should validate but might be inefficient
    assert!(search_op.validate().is_ok());
}

#[test]
fn test_score_threshold_edge_cases() {
    // Test with score threshold of 0.0
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: Some(0.0),
    };
    assert!(search_op.validate().is_ok());

    // Test with score threshold of 1.0
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: Some(1.0),
    };
    assert!(search_op.validate().is_ok());

    // Test with score threshold greater than 1.0
    let search_op = SearchOperation {
        collection_name: "test".to_string(),
        vector: vec![0.1; 128],
        limit: 10,
        offset: None,
        filter: None,
        params: None,
        with_payload: true,
        with_vector: false,
        score_threshold: Some(1.5), // May be valid depending on distance metric
    };
    assert!(search_op.validate().is_ok()); // Should be handled by Qdrant
}

// Helper Functions

fn create_test_point_structs(count: usize) -> Vec<PointStruct> {
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

// Additional property-based tests using proptest

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_vector_operation_properties(
            collection_name in "[a-zA-Z0-9_]{1,50}",
            vector_size in 1usize..=2048,
            point_count in 1usize..=100
        ) {
            let vector_data: Vec<f32> = (0..vector_size).map(|i| (i as f32) * 0.01).collect();
            let mut points = Vec::new();

            for i in 0..point_count {
                let mut payload = HashMap::new();
                payload.insert("id".to_string(), qdrant_client::qdrant::Value {
                    kind: Some(qdrant_client::qdrant::value::Kind::IntegerValue(i as i64)),
                });

                points.push(PointStruct {
                    id: Some(PointId::from(i as u64)),
                    vectors: Some(Vectors {
                        vectors_options: Some(qdrant_client::qdrant::vectors::VectorsOptions::Vector(
                            Vector { data: vector_data.clone() }
                        )),
                    }),
                    payload,
                });
            }

            let upsert_op = VectorOperation::Upsert {
                collection_name: collection_name.clone(),
                points: points.clone(),
                wait: true,
            };

            // The operation should be structurally valid
            match upsert_op {
                VectorOperation::Upsert { collection_name: ref name, ref points, wait } => {
                    prop_assert_eq!(name, &collection_name);
                    prop_assert_eq!(points.len(), point_count);
                    prop_assert!(wait);
                }
                _ => prop_assert!(false, "Expected Upsert variant"),
            }
        }

        #[test]
        fn test_search_operation_properties(
            collection_name in "[a-zA-Z0-9_]{1,50}",
            vector_size in 1usize..=2048,
            limit in 1u64..=1000,
            score_threshold in 0.0f32..=1.0f32
        ) {
            let vector: Vec<f32> = (0..vector_size).map(|i| (i as f32) * 0.001).collect();

            let search_op = SearchOperation {
                collection_name: collection_name.clone(),
                vector,
                limit,
                offset: None,
                filter: None,
                params: None,
                with_payload: true,
                with_vector: false,
                score_threshold: Some(score_threshold),
            };

            // Validation should succeed for reasonable inputs
            let validation_result = search_op.validate();
            if collection_name.len() > 0 && vector_size > 0 && limit > 0 {
                prop_assert!(validation_result.is_ok());
            }
        }
    }
}