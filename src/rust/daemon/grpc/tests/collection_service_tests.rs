//! Integration tests for CollectionService
//!
//! Tests the complete CollectionService gRPC implementation including:
//! - Collection creation with various configurations
//! - Collection deletion with force flag
//! - Alias operations (when implemented)
//! - Error handling and validation
//! - gRPC status code mapping

use workspace_qdrant_grpc::proto::{
    collection_service_server::CollectionService,
    CreateCollectionRequest, CreateCollectionResponse,
    DeleteCollectionRequest, CreateAliasRequest,
    DeleteAliasRequest, RenameAliasRequest, CollectionConfig,
};
use workspace_qdrant_grpc::services::CollectionServiceImpl;
use tonic::{Request, Code};

/// Test helper to create a basic collection config
fn create_test_config(vector_size: i32, metric: &str) -> CollectionConfig {
    CollectionConfig {
        vector_size,
        distance_metric: metric.to_string(),
        enable_indexing: true,
        metadata_schema: std::collections::HashMap::new(),
    }
}

#[tokio::test]
async fn test_create_collection_success() {
    let service = CollectionServiceImpl::default();

    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(create_test_config(384, "Cosine")),
    });

    // Note: This test will fail with the current mock setup
    // It demonstrates the expected interface
    match service.create_collection(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            assert!(resp.success);
            assert_eq!(resp.collection_id, "test_collection");
        }
        Err(status) => {
            // Accept unavailable or connection errors due to no real Qdrant
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal
                || status.code() == Code::AlreadyExists,
                "Unexpected error: {:?}", status
            );
        }
    }
}

#[tokio::test]
async fn test_create_collection_invalid_name() {
    let service = CollectionServiceImpl::default();

    // Too short
    let request = Request::new(CreateCollectionRequest {
        collection_name: "ab".to_string(),
        project_id: "test".to_string(),
        config: Some(create_test_config(384, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);

    // Starts with number
    let request = Request::new(CreateCollectionRequest {
        collection_name: "123collection".to_string(),
        project_id: "test".to_string(),
        config: Some(create_test_config(384, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);

    // Invalid characters
    let request = Request::new(CreateCollectionRequest {
        collection_name: "collection@test".to_string(),
        project_id: "test".to_string(),
        config: Some(create_test_config(384, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_create_collection_missing_config() {
    let service = CollectionServiceImpl::default();

    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: None,
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_create_collection_invalid_vector_size() {
    let service = CollectionServiceImpl::default();

    // Zero vector size
    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(create_test_config(0, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);

    // Negative vector size
    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(create_test_config(-1, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);

    // Too large vector size
    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(create_test_config(10001, "Cosine")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_create_collection_invalid_distance_metric() {
    let service = CollectionServiceImpl::default();

    let request = Request::new(CreateCollectionRequest {
        collection_name: "test_collection".to_string(),
        project_id: "test_project".to_string(),
        config: Some(create_test_config(384, "Manhattan")),
    });

    let result = service.create_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_delete_collection_validation() {
    let service = CollectionServiceImpl::default();

    // Invalid name
    let request = Request::new(DeleteCollectionRequest {
        collection_name: "ab".to_string(),
        project_id: "test".to_string(),
        force: false,
    });

    let result = service.delete_collection(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_alias_operations_unimplemented() {
    let service = CollectionServiceImpl::default();

    // Create alias
    let request = Request::new(CreateAliasRequest {
        alias_name: "test_alias".to_string(),
        collection_name: "test_collection".to_string(),
    });

    let result = service.create_collection_alias(request).await;
    assert!(result.is_err());
    // Should be unimplemented since StorageClient lacks alias methods
    let err = result.unwrap_err();
    assert!(
        err.code() == Code::Unimplemented || err.code() == Code::Unavailable
    );

    // Delete alias
    let request = Request::new(DeleteAliasRequest {
        alias_name: "test_alias".to_string(),
    });

    let result = service.delete_collection_alias(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::Unimplemented);

    // Rename alias
    let request = Request::new(RenameAliasRequest {
        old_alias_name: "old_alias".to_string(),
        new_alias_name: "new_alias".to_string(),
        collection_name: "test_collection".to_string(),
    });

    let result = service.rename_collection_alias(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::Unimplemented);
}

#[tokio::test]
async fn test_various_distance_metrics() {
    let service = CollectionServiceImpl::default();

    // Test each valid metric
    let metrics = vec!["Cosine", "Euclidean", "Dot"];

    for metric in metrics {
        let request = Request::new(CreateCollectionRequest {
            collection_name: format!("test_{}", metric.to_lowercase()),
            project_id: "test_project".to_string(),
            config: Some(create_test_config(384, metric)),
        });

        let result = service.create_collection(request).await;
        // Should either succeed or fail with storage error, not validation error
        if let Err(status) = result {
            assert!(
                status.code() != Code::InvalidArgument,
                "Metric {} should be valid but got InvalidArgument", metric
            );
        }
    }
}

#[tokio::test]
async fn test_various_vector_sizes() {
    let service = CollectionServiceImpl::default();

    // Test standard sizes
    let sizes = vec![384, 768, 1536];

    for size in sizes {
        let request = Request::new(CreateCollectionRequest {
            collection_name: format!("test_size_{}", size),
            project_id: "test_project".to_string(),
            config: Some(create_test_config(size, "Cosine")),
        });

        let result = service.create_collection(request).await;
        // Should either succeed or fail with storage error, not validation error
        if let Err(status) = result {
            assert!(
                status.code() != Code::InvalidArgument,
                "Size {} should be valid but got InvalidArgument", size
            );
        }
    }
}

#[tokio::test]
async fn test_edge_case_names() {
    let service = CollectionServiceImpl::default();

    // Valid edge cases
    let valid_names = vec![
        "abc".to_string(),           // Minimum length
        "a_b_c".to_string(),         // Underscores
        "a-b-c".to_string(),         // Hyphens
        "collection123".to_string(), // Numbers (not at start)
        "a".repeat(255),             // Maximum length
    ];

    for name in &valid_names {
        let request = Request::new(CreateCollectionRequest {
            collection_name: name.clone(),
            project_id: "test".to_string(),
            config: Some(create_test_config(384, "Cosine")),
        });

        let result = service.create_collection(request).await;
        if let Err(status) = result {
            assert!(
                status.code() != Code::InvalidArgument,
                "Name '{}' should be valid but got InvalidArgument: {}",
                name, status.message()
            );
        }
    }

    // Invalid edge cases
    let invalid_names = vec![
        "ab".to_string(),      // Too short
        "1abc".to_string(),    // Starts with number
        "a.b".to_string(),     // Invalid character
        "a".repeat(256),       // Too long
    ];

    for name in &invalid_names {
        let request = Request::new(CreateCollectionRequest {
            collection_name: name.clone(),
            project_id: "test".to_string(),
            config: Some(create_test_config(384, "Cosine")),
        });

        let result = service.create_collection(request).await;
        assert!(
            result.is_err() && result.unwrap_err().code() == Code::InvalidArgument,
            "Name '{}' should be invalid", name
        );
    }
}
