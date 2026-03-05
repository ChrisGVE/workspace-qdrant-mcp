//! Spill-to-SQLite queue tests for pipeline overflow handling

use super::super::*;
use std::sync::Arc;

#[tokio::test]
async fn test_spill_to_sqlite_process_document() {
    use crate::queue_operations::QueueManager;
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let mut pipeline = Pipeline::new(2);
    pipeline.set_spill_queue(queue_manager.clone());
    let submitter = pipeline.task_submitter();

    let context = TaskContext {
        task_id: Uuid::new_v4(),
        priority: TaskPriority::ProjectWatching,
        created_at: chrono::Utc::now(),
        timeout_ms: None,
        source: TaskSource::ProjectWatcher {
            project_path: "/tmp/test_project".to_string(),
        },
        metadata: HashMap::new(),
        checkpoint_id: None,
        supports_checkpointing: true,
        tenant_id: None,
    };
    let payload = TaskPayload::ProcessDocument {
        file_path: std::path::PathBuf::from("/tmp/test_project/src/main.rs"),
        collection: "projects".to_string(),
        branch: "main".to_string(),
    };

    submitter
        .spill_to_sqlite(&queue_manager, &context, &payload)
        .await
        .expect("Spill should succeed");

    let row: (String, String, String, String) =
        sqlx::query_as("SELECT item_type, op, collection, status FROM unified_queue LIMIT 1")
            .fetch_one(&pool)
            .await
            .expect("Should have one spilled item");

    assert_eq!(row.0, "file");
    assert_eq!(row.1, "add");
    assert_eq!(row.2, "projects");
    assert_eq!(row.3, "pending");

    let payload_json: (String,) = sqlx::query_as("SELECT payload_json FROM unified_queue LIMIT 1")
        .fetch_one(&pool)
        .await
        .expect("Should have payload");

    let payload_val: serde_json::Value = serde_json::from_str(&payload_json.0).unwrap();
    assert_eq!(
        payload_val["file_path"].as_str().unwrap(),
        "/tmp/test_project/src/main.rs"
    );

    let metadata_json: (String,) = sqlx::query_as("SELECT metadata FROM unified_queue LIMIT 1")
        .fetch_one(&pool)
        .await
        .expect("Should have metadata");

    let meta_val: serde_json::Value = serde_json::from_str(&metadata_json.0).unwrap();
    assert_eq!(meta_val["spilled_from"].as_str().unwrap(), "pipeline");
    assert_eq!(
        meta_val["original_priority"].as_str().unwrap(),
        "ProjectWatching"
    );
}

#[tokio::test]
async fn test_spill_non_process_document_fails() {
    use crate::queue_operations::QueueManager;
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let mut pipeline = Pipeline::new(2);
    pipeline.set_spill_queue(queue_manager.clone());
    let submitter = pipeline.task_submitter();

    let context = TaskContext {
        task_id: Uuid::new_v4(),
        priority: TaskPriority::BackgroundWatching,
        created_at: chrono::Utc::now(),
        timeout_ms: None,
        source: TaskSource::Generic {
            operation: "test".to_string(),
        },
        metadata: HashMap::new(),
        checkpoint_id: None,
        supports_checkpointing: false,
        tenant_id: None,
    };

    let payload = TaskPayload::Generic {
        operation: "test".to_string(),
        parameters: HashMap::new(),
    };
    let result = submitter
        .spill_to_sqlite(&queue_manager, &context, &payload)
        .await;
    assert!(result.is_err(), "Generic tasks should not be spillable");

    let payload = TaskPayload::ExecuteQuery {
        query: "test".to_string(),
        collection: "test".to_string(),
        limit: 10,
    };
    let result = submitter
        .spill_to_sqlite(&queue_manager, &context, &payload)
        .await;
    assert!(
        result.is_err(),
        "ExecuteQuery tasks should not be spillable"
    );
}

#[tokio::test]
async fn test_spill_with_background_watcher_source() {
    use crate::queue_operations::QueueManager;
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    let queue_manager = Arc::new(QueueManager::new(pool.clone()));

    let mut pipeline = Pipeline::new(2);
    pipeline.set_spill_queue(queue_manager.clone());
    let submitter = pipeline.task_submitter();

    let context = TaskContext {
        task_id: Uuid::new_v4(),
        priority: TaskPriority::BackgroundWatching,
        created_at: chrono::Utc::now(),
        timeout_ms: None,
        source: TaskSource::BackgroundWatcher {
            folder_path: "/home/user/docs".to_string(),
        },
        metadata: HashMap::new(),
        checkpoint_id: None,
        supports_checkpointing: true,
        tenant_id: None,
    };
    let payload = TaskPayload::ProcessDocument {
        file_path: std::path::PathBuf::from("/home/user/docs/readme.md"),
        collection: "libraries".to_string(),
        branch: "main".to_string(),
    };

    submitter
        .spill_to_sqlite(&queue_manager, &context, &payload)
        .await
        .expect("Spill with BackgroundWatcher source should succeed");

    let row: (String,) = sqlx::query_as("SELECT collection FROM unified_queue LIMIT 1")
        .fetch_one(&pool)
        .await
        .expect("Should have spilled item");

    assert_eq!(row.0, "libraries");
}

#[tokio::test]
async fn test_pipeline_stats_include_spill_count() {
    let pipeline = Pipeline::new(2);
    let stats = pipeline.stats().await;
    assert_eq!(stats.queue_spills, 0);
}

#[tokio::test]
async fn test_queue_metrics_include_spill_count() {
    let mut pipeline = Pipeline::new(2);
    pipeline.start().await.expect("Failed to start pipeline");

    let metrics = pipeline.get_priority_system_metrics().await;
    assert_eq!(metrics.queue.queue_spill_count, 0);
}

#[tokio::test]
async fn test_spill_queue_configuration() {
    use crate::queue_operations::QueueManager;
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;

    let pool = sqlx::SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    let queue_manager = Arc::new(QueueManager::new(pool));

    let pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();
    assert!(submitter.spill_queue.is_none());

    let mut pipeline = Pipeline::new(2);
    pipeline.set_spill_queue(queue_manager);
    let submitter = pipeline.task_submitter();
    assert!(submitter.spill_queue.is_some());
}
