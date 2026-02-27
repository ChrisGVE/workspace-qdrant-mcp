//! Queue validation tests (Task 46) and library document payload validation.

use super::*;

#[tokio::test]
async fn test_validation_empty_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_tenant.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Empty tenant_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
}

#[tokio::test]
async fn test_validation_whitespace_tenant_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_ws_tenant.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Whitespace-only tenant_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "   ",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyTenantId));
}

#[tokio::test]
async fn test_validation_empty_collection() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_collection.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Empty collection should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::EmptyCollection));
}

#[tokio::test]
async fn test_validation_invalid_json_payload() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_json.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Invalid JSON should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            "not valid json",
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), QueueError::InvalidPayloadJson(_)));
}

#[tokio::test]
async fn test_validation_file_missing_file_path() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_file_path.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File item without file_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"other_field":"value"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "file_path"
    ));
}

#[tokio::test]
async fn test_validation_content_missing_content() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_content.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Content item without content field should fail
    let result = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"source_type":"mcp"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "text" && field == "content"
    ));
}

#[tokio::test]
async fn test_validation_delete_document_missing_document_id() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_delete_doc.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // DeleteDocument without document_id should fail
    let result = manager
        .enqueue_unified(
            ItemType::Doc,
            UnifiedOp::Delete,
            "test-tenant",
            "test-collection",
            r#"{"point_ids":["abc"]}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "doc" && field == "document_id"
    ));
}

#[tokio::test]
async fn test_validation_file_rename_missing_old_path() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_rename.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File rename without old_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Rename,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/new.rs"}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "old_path"
    ));
}

#[tokio::test]
async fn test_validation_valid_items_pass() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_valid.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // Valid file item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/file.rs"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid content item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::Text,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"content":"test content","source_type":"mcp"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid file rename item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Rename,
            "test-tenant",
            "test-collection",
            r#"{"file_path":"/test/new.rs","old_path":"/test/old.rs"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());

    // Valid doc delete item should succeed
    let result = manager
        .enqueue_unified(
            ItemType::Doc,
            UnifiedOp::Delete,
            "test-tenant",
            "test-collection",
            r#"{"document_id":"doc-123"}"#,
            None,
            None,
        )
        .await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_validation_empty_string_in_required_field() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_validation_empty_field.db");

    let config = QueueConnectionConfig::with_database_path(&db_path);
    let pool = config.create_pool().await.unwrap();

    let manager = QueueManager::new(pool);
    manager.init_unified_queue().await.unwrap();

    // File with empty file_path should fail
    let result = manager
        .enqueue_unified(
            ItemType::File,
            UnifiedOp::Add,
            "test-tenant",
            "test-collection",
            r#"{"file_path":""}"#,
            None,
            None,
        )
        .await;

    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        QueueError::MissingPayloadField { item_type, field }
        if item_type == "file" && field == "file_path"
    ));
}

// ===== Library document payload validation =====

#[test]
fn test_validate_library_document_payload_valid() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        "document_type": "page_based",
        "source_format": "pdf",
        "doc_id": "550e8400-e29b-41d4-a716-446655440000",
    });
    assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
}

#[test]
fn test_validate_library_document_payload_stream_based() {
    let payload = serde_json::json!({
        "document_path": "/books/novel.epub",
        "library_name": "ebooks",
        "document_type": "stream_based",
        "source_format": "epub",
        "doc_id": "uuid-here",
    });
    assert!(QueueManager::validate_library_document_payload(&payload).is_ok());
}

#[test]
fn test_validate_library_document_payload_missing_field() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        // missing document_type, source_format, doc_id
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err());
}

#[test]
fn test_validate_library_document_payload_invalid_document_type() {
    let payload = serde_json::json!({
        "document_path": "/docs/report.pdf",
        "library_name": "internal-docs",
        "document_type": "unknown_type",
        "source_format": "pdf",
        "doc_id": "uuid-here",
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err());
}

#[test]
fn test_validate_library_document_payload_empty_fields() {
    let payload = serde_json::json!({
        "document_path": "",
        "library_name": "docs",
        "document_type": "page_based",
        "source_format": "pdf",
        "doc_id": "uuid",
    });
    let result = QueueManager::validate_library_document_payload(&payload);
    assert!(result.is_err()); // document_path is empty
}
