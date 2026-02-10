//! Basic ingestion tests for DocumentService
//!
//! Tests basic request/response patterns for text ingestion operations:
//! - IngestText RPC
//! - Message serialization/deserialization
//! - Response format validation
//! - Input validation

use workspace_qdrant_grpc::proto::{
    document_service_server::DocumentService,
    IngestTextRequest, IngestTextResponse,
    UpdateDocumentRequest,
    DeleteDocumentRequest,
};
use workspace_qdrant_grpc::services::DocumentServiceImpl;
use workspace_qdrant_core::StorageClient;
use tonic::{Request, Code};
use prost::Message;
use std::collections::HashMap;

/// Test helper to create DocumentService instance
fn create_service() -> DocumentServiceImpl {
    DocumentServiceImpl::default()
}

/// Clean up any test data written to a real Qdrant instance.
/// The gRPC routing sends tenant_id="test-tenant" (non-hex) to the `libraries`
/// collection with payload field `library_name = "test-tenant"`.
/// This is best-effort: silently ignores errors when Qdrant is unavailable.
async fn cleanup_test_tenant_data() {
    let storage = StorageClient::new();
    let _ = storage.delete_points_by_payload_field("libraries", "library_name", "test-tenant").await;
}

// =============================================================================
// Basic IngestText Tests
// =============================================================================

#[tokio::test]
async fn test_ingest_text_basic_request() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "This is a test document for ingestion".to_string(),
        collection_basename: "test-notes".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    // Note: This test will fail with unavailable/connection errors due to no real Qdrant
    // It demonstrates the expected interface and validates request structure
    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            assert!(resp.success);
            assert!(!resp.document_id.is_empty());
        }
        Err(status) => {
            // Accept unavailable or connection errors due to no real Qdrant
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal
                || status.code() == Code::InvalidArgument,
                "Unexpected error: {:?}", status
            );
        }
    }
    cleanup_test_tenant_data().await;
}

#[tokio::test]
async fn test_ingest_text_with_custom_id() {
    let service = create_service();

    let custom_id = "custom-doc-123".to_string();
    let request = Request::new(IngestTextRequest {
        content: "Test content".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: Some(custom_id.clone()),
        metadata: HashMap::new(),
        chunk_text: true,
    });

    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            // Should use the provided custom ID
            assert_eq!(resp.document_id, custom_id);
        }
        Err(status) => {
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal
                || status.code() == Code::InvalidArgument,
                "Unexpected error: {:?}", status
            );
        }
    }
    cleanup_test_tenant_data().await;
}

#[tokio::test]
async fn test_ingest_text_with_metadata() {
    let service = create_service();

    let mut metadata = HashMap::new();
    metadata.insert("author".to_string(), "test-user".to_string());
    metadata.insert("category".to_string(), "notes".to_string());
    metadata.insert("priority".to_string(), "high".to_string());

    let request = Request::new(IngestTextRequest {
        content: "Test content with metadata".to_string(),
        collection_basename: "test-notes".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: metadata.clone(),
        chunk_text: true,
    });

    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            assert!(resp.success);
        }
        Err(status) => {
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal
                || status.code() == Code::InvalidArgument,
                "Unexpected error: {:?}", status
            );
        }
    }
    cleanup_test_tenant_data().await;
}

#[tokio::test]
async fn test_ingest_text_without_chunking() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Short text that should not be chunked".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: false,
    });

    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            assert!(resp.success);
            // Should create exactly 1 chunk when chunking is disabled
            assert_eq!(resp.chunks_created, 1);
        }
        Err(status) => {
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal
                || status.code() == Code::InvalidArgument,
                "Unexpected error: {:?}", status
            );
        }
    }
    cleanup_test_tenant_data().await;
}

// =============================================================================
// Input Validation Tests
// =============================================================================

#[tokio::test]
async fn test_ingest_text_empty_content() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    let result = service.ingest_text(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_ingest_text_invalid_collection_name_too_short() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Test content".to_string(),
        collection_basename: "ab".to_string(), // Too short
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    let result = service.ingest_text(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_ingest_text_invalid_collection_name_starts_with_number() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Test content".to_string(),
        collection_basename: "123collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    let result = service.ingest_text(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_ingest_text_invalid_collection_name_special_chars() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Test content".to_string(),
        collection_basename: "collection@test".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    let result = service.ingest_text(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

#[tokio::test]
async fn test_ingest_text_empty_tenant_id() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Test content".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    let result = service.ingest_text(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), Code::InvalidArgument);
}

// =============================================================================
// Message Serialization Tests
// =============================================================================

#[tokio::test]
async fn test_ingest_text_request_serialization() {
    let mut metadata = HashMap::new();
    metadata.insert("key1".to_string(), "value1".to_string());
    metadata.insert("key2".to_string(), "value2".to_string());

    let request = IngestTextRequest {
        content: "Test content for serialization".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: Some("doc-123".to_string()),
        metadata: metadata.clone(),
        chunk_text: true,
    };

    // Test serialization
    let bytes = request.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = IngestTextRequest::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.content, "Test content for serialization");
    assert_eq!(decoded.collection_basename, "test-collection");
    assert_eq!(decoded.tenant_id, "test-tenant");
    assert_eq!(decoded.document_id, Some("doc-123".to_string()));
    assert_eq!(decoded.metadata.len(), 2);
    assert_eq!(decoded.chunk_text, true);
}

#[tokio::test]
async fn test_ingest_text_response_serialization() {
    let response = IngestTextResponse {
        document_id: "generated-uuid-123".to_string(),
        success: true,
        chunks_created: 5,
        error_message: String::new(),
    };

    // Test serialization
    let bytes = response.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = IngestTextResponse::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.document_id, "generated-uuid-123");
    assert_eq!(decoded.success, true);
    assert_eq!(decoded.chunks_created, 5);
    assert_eq!(decoded.error_message, "");
}

#[tokio::test]
async fn test_update_text_request_serialization() {
    let mut metadata = HashMap::new();
    metadata.insert("updated".to_string(), "true".to_string());

    let request = UpdateDocumentRequest {
        document_id: "doc-123".to_string(),
        content: "Updated content".to_string(),
        collection_name: Some("new-collection".to_string()),
        metadata,
    };

    // Test serialization
    let bytes = request.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = UpdateDocumentRequest::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.document_id, "doc-123");
    assert_eq!(decoded.content, "Updated content");
    assert_eq!(decoded.collection_name, Some("new-collection".to_string()));
}

#[tokio::test]
async fn test_delete_text_request_serialization() {
    let request = DeleteDocumentRequest {
        document_id: "doc-to-delete".to_string(),
        collection_name: "test-collection".to_string(),
    };

    // Test serialization
    let bytes = request.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = DeleteDocumentRequest::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.document_id, "doc-to-delete");
    assert_eq!(decoded.collection_name, "test-collection");
}

// =============================================================================
// Response Format Validation Tests
// =============================================================================

#[tokio::test]
async fn test_ingest_response_structure() {
    let service = create_service();

    let request = Request::new(IngestTextRequest {
        content: "Test content for response validation".to_string(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            // Validate response structure
            assert!(!resp.document_id.is_empty(), "Document ID should not be empty");
            assert!(resp.chunks_created >= 0, "Chunks created should be non-negative");
            if !resp.success {
                assert!(!resp.error_message.is_empty(), "Error message should be present when success=false");
            }
        }
        Err(_) => {
            // Expected when Qdrant is not available
        }
    }
    cleanup_test_tenant_data().await;
}

#[tokio::test]
async fn test_large_content_ingestion() {
    let service = create_service();

    // Create large content (5KB)
    let large_content = "Lorem ipsum dolor sit amet. ".repeat(200);

    let request = Request::new(IngestTextRequest {
        content: large_content.clone(),
        collection_basename: "test-collection".to_string(),
        tenant_id: "test-tenant".to_string(),
        document_id: None,
        metadata: HashMap::new(),
        chunk_text: true,
    });

    match service.ingest_text(request).await {
        Ok(response) => {
            let resp = response.into_inner();
            // Should create multiple chunks for large content
            assert!(resp.chunks_created > 1, "Large content should be split into multiple chunks");
        }
        Err(status) => {
            assert!(
                status.code() == Code::Unavailable
                || status.code() == Code::Internal,
                "Unexpected error: {:?}", status
            );
        }
    }
    cleanup_test_tenant_data().await;
}

#[tokio::test]
async fn test_various_content_types() {
    let service = create_service();

    let test_cases = vec![
        ("Plain text", "test-collection"),
        ("Text with\nnewlines\nand\ntabs\t", "test-collection"),
        ("Unicode content: 你好世界 🌍", "test-collection"),
        ("Code snippet: fn main() { println!(\"Hello\"); }", "test-collection"),
        ("JSON content: {\"key\": \"value\"}", "test-collection"),
    ];

    for (content, collection) in test_cases {
        let request = Request::new(IngestTextRequest {
            content: content.to_string(),
            collection_basename: collection.to_string(),
            tenant_id: "test-tenant".to_string(),
            document_id: None,
            metadata: HashMap::new(),
            chunk_text: true,
        });

        // Should not panic or reject valid UTF-8 content
        let _ = service.ingest_text(request).await;
    }
    cleanup_test_tenant_data().await;
}
