//! Tests for the unified queue schema types, SQL constants, and key generation.

use super::*;

#[test]
fn test_item_type_display() {
    assert_eq!(ItemType::Text.to_string(), "text");
    assert_eq!(ItemType::Tenant.to_string(), "tenant");
}

#[test]
fn test_item_type_from_str() {
    assert_eq!(ItemType::parse_str("text"), Some(ItemType::Text));
    assert_eq!(ItemType::parse_str("tenant"), Some(ItemType::Tenant));
    // Legacy values still work
    assert_eq!(ItemType::parse_str("content"), Some(ItemType::Text));
    assert_eq!(ItemType::parse_str("delete_tenant"), Some(ItemType::Tenant));
    assert_eq!(ItemType::parse_str("invalid"), None);
}

#[test]
fn test_operation_validity() {
    // Text: add, update, delete, uplift
    assert!(QueueOperation::Add.is_valid_for(ItemType::Text));
    assert!(QueueOperation::Update.is_valid_for(ItemType::Text));
    assert!(QueueOperation::Delete.is_valid_for(ItemType::Text));
    assert!(!QueueOperation::Scan.is_valid_for(ItemType::Text));

    // Folder: delete, scan, rename
    assert!(!QueueOperation::Update.is_valid_for(ItemType::Folder));
    assert!(QueueOperation::Scan.is_valid_for(ItemType::Folder));

    // Tenant: add, update, delete, scan, rename, uplift
    assert!(QueueOperation::Add.is_valid_for(ItemType::Tenant));
    assert!(QueueOperation::Delete.is_valid_for(ItemType::Tenant));

    // File: add, update, delete, rename, uplift
    assert!(QueueOperation::Rename.is_valid_for(ItemType::File));
    assert!(!QueueOperation::Rename.is_valid_for(ItemType::Text));
}

#[test]
fn test_idempotency_key_generation() {
    let key1 = generate_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
    let key2 = generate_idempotency_key(ItemType::File, "my-collection", "/path/to/file.txt");
    assert_eq!(key1, key2); // Same inputs = same key

    let key3 = generate_idempotency_key(ItemType::File, "my-collection", "/path/to/other.txt");
    assert_ne!(key1, key3); // Different inputs = different key
}

#[test]
fn test_queue_status_display() {
    assert_eq!(QueueStatus::Pending.to_string(), "pending");
    assert_eq!(QueueStatus::InProgress.to_string(), "in_progress");
}

#[test]
fn test_unified_idempotency_key_generation() {
    let key1 = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/file.rs"}"#,
    )
    .unwrap();

    assert_eq!(key1.len(), 32);

    let key2 = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/file.rs"}"#,
    )
    .unwrap();
    assert_eq!(key1, key2);

    let key3 = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/other.rs"}"#,
    )
    .unwrap();
    assert_ne!(key1, key3);

    let key4 = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Update,
        "proj_abc123",
        "my-project-code",
        r#"{"file_path":"/path/to/file.rs"}"#,
    )
    .unwrap();
    assert_ne!(key1, key4);
}

#[test]
fn test_unified_idempotency_key_validation() {
    let result = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "",
        "my-collection",
        "{}",
    );
    assert_eq!(result, Err(IdempotencyKeyError::EmptyTenantId));

    let result = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "",
        "{}",
    );
    assert_eq!(result, Err(IdempotencyKeyError::EmptyCollection));

    let result = generate_unified_idempotency_key(
        ItemType::Collection,
        QueueOperation::Add,
        "proj_abc123",
        "my-collection",
        "{}",
    );
    assert!(matches!(
        result,
        Err(IdempotencyKeyError::InvalidOperationForType { .. })
    ));
}

#[test]
fn test_unified_idempotency_key_cross_language_compatibility() {
    let key = generate_unified_idempotency_key(
        ItemType::File,
        QueueOperation::Add,
        "proj_abc123",
        "my-project-code",
        "{}",
    )
    .unwrap();

    assert!(key.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(key.len(), 32);
}

#[test]
fn test_queue_decision_serde_roundtrip() {
    let decision = QueueDecision {
        delete_old: true,
        old_base_point: Some("abc123".to_string()),
        new_base_point: "def456".to_string(),
        old_file_hash: Some("oldhash".to_string()),
        new_file_hash: "newhash".to_string(),
    };

    let json = serde_json::to_string(&decision).unwrap();
    let parsed: QueueDecision = serde_json::from_str(&json).unwrap();

    assert!(parsed.delete_old);
    assert_eq!(parsed.old_base_point, Some("abc123".to_string()));
    assert_eq!(parsed.new_base_point, "def456");
    assert_eq!(parsed.old_file_hash, Some("oldhash".to_string()));
    assert_eq!(parsed.new_file_hash, "newhash");
}

#[test]
fn test_queue_decision_no_old() {
    let decision = QueueDecision {
        delete_old: false,
        old_base_point: None,
        new_base_point: "def456".to_string(),
        old_file_hash: None,
        new_file_hash: "newhash".to_string(),
    };

    let json = serde_json::to_string(&decision).unwrap();
    let parsed: QueueDecision = serde_json::from_str(&json).unwrap();

    assert!(!parsed.delete_old);
    assert!(parsed.old_base_point.is_none());
}

#[test]
fn test_check_completion_both_done() {
    let item = UnifiedQueueItem {
        queue_id: "q1".into(),
        idempotency_key: "k1".into(),
        item_type: ItemType::File,
        op: QueueOperation::Add,
        tenant_id: "t1".into(),
        collection: "projects".into(),
        status: QueueStatus::InProgress,
        branch: "main".into(),
        payload_json: "{}".into(),
        metadata: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: Some(DestinationStatus::Done),
        search_status: Some(DestinationStatus::Done),
        decision_json: None,
    };
    assert_eq!(item.check_completion(), QueueStatus::Done);
}

#[test]
fn test_check_completion_partial_done() {
    let item = UnifiedQueueItem {
        queue_id: "q1".into(),
        idempotency_key: "k1".into(),
        item_type: ItemType::File,
        op: QueueOperation::Add,
        tenant_id: "t1".into(),
        collection: "projects".into(),
        status: QueueStatus::InProgress,
        branch: "main".into(),
        payload_json: "{}".into(),
        metadata: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: Some(DestinationStatus::Done),
        search_status: Some(DestinationStatus::Pending),
        decision_json: None,
    };
    assert_eq!(item.check_completion(), QueueStatus::InProgress);
}

#[test]
fn test_check_completion_one_failed() {
    let item = UnifiedQueueItem {
        queue_id: "q1".into(),
        idempotency_key: "k1".into(),
        item_type: ItemType::File,
        op: QueueOperation::Add,
        tenant_id: "t1".into(),
        collection: "projects".into(),
        status: QueueStatus::InProgress,
        branch: "main".into(),
        payload_json: "{}".into(),
        metadata: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: Some(DestinationStatus::Done),
        search_status: Some(DestinationStatus::Failed),
        decision_json: None,
    };
    assert_eq!(item.check_completion(), QueueStatus::Failed);
}

#[test]
fn test_check_completion_none_defaults_pending() {
    let item = UnifiedQueueItem {
        queue_id: "q1".into(),
        idempotency_key: "k1".into(),
        item_type: ItemType::File,
        op: QueueOperation::Add,
        tenant_id: "t1".into(),
        collection: "projects".into(),
        status: QueueStatus::Pending,
        branch: "main".into(),
        payload_json: "{}".into(),
        metadata: None,
        created_at: "2025-01-01T00:00:00Z".into(),
        updated_at: "2025-01-01T00:00:00Z".into(),
        lease_until: None,
        worker_id: None,
        retry_count: 0,
        error_message: None,
        last_error_at: None,
        file_path: None,
        qdrant_status: None,
        search_status: None,
        decision_json: None,
    };
    // Both None → treated as pending → InProgress
    assert_eq!(item.check_completion(), QueueStatus::InProgress);
}

#[test]
fn test_destination_status_display() {
    assert_eq!(DestinationStatus::Pending.to_string(), "pending");
    assert_eq!(DestinationStatus::InProgress.to_string(), "in_progress");
    assert_eq!(DestinationStatus::Done.to_string(), "done");
    assert_eq!(DestinationStatus::Failed.to_string(), "failed");
}

#[test]
fn test_destination_status_from_str() {
    assert_eq!(
        DestinationStatus::parse_str("pending"),
        Some(DestinationStatus::Pending)
    );
    assert_eq!(
        DestinationStatus::parse_str("done"),
        Some(DestinationStatus::Done)
    );
    assert_eq!(DestinationStatus::parse_str("invalid"), None);
}
