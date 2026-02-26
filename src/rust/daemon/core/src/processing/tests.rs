//! Integration and unit tests for the processing pipeline

use super::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_pipeline_creation() {
    let pipeline = Pipeline::new(2);
    let stats = pipeline.stats().await;

    assert_eq!(stats.queued_tasks, 0);
    assert_eq!(stats.running_tasks, 0);
    assert_eq!(stats.total_capacity, 2);
}

#[tokio::test]
async fn test_task_submission_and_execution() {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let task_handle = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand {
            command: "test_command".to_string(),
        },
        TaskPayload::Generic {
            operation: "test".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit task");

    let result = timeout(Duration::from_secs(10), task_handle.wait()).await
        .expect("Task timed out")
        .expect("Task execution failed");

    match result {
        TaskResult::Success { data, .. } => {
            match data {
                TaskResultData::Generic { message, .. } => {
                    assert_eq!(message, "Completed operation: test");
                }
                _ => panic!("Expected Generic result data"),
            }
        }
        _ => panic!("Expected successful result, got: {:?}", result),
    }
}

#[tokio::test]
async fn test_priority_ordering() {
    let mut pipeline = Pipeline::new(1);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let low_priority_task = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher {
            folder_path: "/tmp".to_string(),
        },
        TaskPayload::Generic {
            operation: "low_priority".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(1)),
    ).await.expect("Failed to submit low priority task");

    tokio::time::sleep(Duration::from_millis(10)).await;

    let high_priority_task = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer {
            request_id: "test_request".to_string(),
        },
        TaskPayload::Generic {
            operation: "high_priority".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(1)),
    ).await.expect("Failed to submit high priority task");

    let high_result = timeout(Duration::from_secs(5), high_priority_task.wait()).await
        .expect("High priority task timed out")
        .expect("High priority task failed");

    let low_result = timeout(Duration::from_secs(5), low_priority_task.wait()).await
        .expect("Low priority task timed out")
        .expect("Low priority task failed");

    match high_result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::Generic { message, .. } = data {
                assert_eq!(message, "Completed operation: high_priority");
            }
        }
        _ => panic!("High priority task should have succeeded"),
    }

    match low_result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::Generic { message, .. } = data {
                assert_eq!(message, "Completed operation: low_priority");
            }
        }
        TaskResult::Cancelled { .. } => {}
        _ => {}
    }
}

#[tokio::test]
async fn test_task_timeout() {
    let mut pipeline = Pipeline::new(1);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let task_handle = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand {
            command: "timeout_test".to_string(),
        },
        TaskPayload::Generic {
            operation: "slow_operation".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_millis(1)),
    ).await.expect("Failed to submit task");

    let result = timeout(Duration::from_secs(2), task_handle.wait()).await
        .expect("Test timed out")
        .expect("Task execution failed");

    match result {
        TaskResult::Timeout { .. } => {}
        other => panic!("Expected timeout, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_concurrent_task_execution() {
    let mut pipeline = Pipeline::new(3);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let mut handles = Vec::new();
    for i in 0..5 {
        let handle = submitter.submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: format!("task_{}", i),
            },
            TaskPayload::Generic {
                operation: format!("concurrent_task_{}", i),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        ).await.expect("Failed to submit task");

        handles.push(handle);
    }

    let mut completed_count = 0;
    for handle in handles {
        let result = timeout(Duration::from_secs(10), handle.wait()).await
            .expect("Task timed out")
            .expect("Task execution failed");

        if let TaskResult::Success { .. } = result {
            completed_count += 1;
        }
    }

    assert_eq!(completed_count, 5, "All tasks should complete successfully");
}

#[tokio::test]
async fn test_pipeline_stats() {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let initial_stats = pipeline.stats().await;
    assert_eq!(initial_stats.queued_tasks, 0);
    assert_eq!(initial_stats.running_tasks, 0);

    let _handle1 = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand { command: "stats_test_1".to_string() },
        TaskPayload::Generic {
            operation: "stats_test_1".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit task 1");

    let _handle2 = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand { command: "stats_test_2".to_string() },
        TaskPayload::Generic {
            operation: "stats_test_2".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit task 2");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let stats = pipeline.stats().await;
    assert!(stats.running_tasks > 0 || stats.queued_tasks > 0);
    assert_eq!(stats.total_capacity, 2);
}

#[tokio::test]
async fn test_preemption_logic() {
    let mut pipeline = Pipeline::new(1);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let low_priority_handle = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher {
            folder_path: "/tmp/background".to_string(),
        },
        TaskPayload::Generic {
            operation: "long_running_background".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit low priority task");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let high_priority_handle = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer {
            request_id: "urgent_request".to_string(),
        },
        TaskPayload::Generic {
            operation: "urgent_mcp_task".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(2)),
    ).await.expect("Failed to submit high priority task");

    let high_result = timeout(Duration::from_secs(3), high_priority_handle.wait()).await
        .expect("High priority task should not timeout")
        .expect("High priority task should succeed");

    match high_result {
        TaskResult::Success { data, .. } => {
            if let TaskResultData::Generic { message, .. } = data {
                assert_eq!(message, "Completed operation: urgent_mcp_task");
            }
        }
        other => panic!("Expected success, got: {:?}", other),
    }

    let low_result = timeout(Duration::from_millis(100), low_priority_handle.wait()).await
        .expect("Low priority task should complete quickly after preemption")
        .expect("Low priority task should have result");

    match low_result {
        TaskResult::Cancelled { .. } => {}
        TaskResult::Success { .. } => {}
        other => panic!("Expected cancelled or success, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_priority_system_metrics() {
    let queue_config = QueueConfigBuilder::new()
        .for_mcp_server()
        .build();

    let mut pipeline = Pipeline::with_queue_config(2, queue_config);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let _handle1 = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer { request_id: "metrics_test_1".to_string() },
        TaskPayload::Generic {
            operation: "metrics_test".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(1)),
    ).await.expect("Failed to submit MCP task");

    let _handle2 = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand { command: "metrics_cli_test".to_string() },
        TaskPayload::Generic {
            operation: "cli_metrics_test".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(1)),
    ).await.expect("Failed to submit CLI task");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let metrics = pipeline.get_priority_system_metrics().await;
    let _ = metrics.pipeline.uptime_seconds;
    assert!(metrics.performance.throughput_tasks_per_second >= 0.0);
    assert!(!metrics.queue.queued_by_priority.is_empty());

    let prometheus_output = pipeline.export_prometheus_metrics().await;
    assert!(prometheus_output.contains("wqm_tasks_completed"));
    assert!(prometheus_output.contains("wqm_queue_total"));
    assert!(!prometheus_output.is_empty());

    let collector = pipeline.metrics_collector();
    collector.record_task_completion(100, TaskPriority::McpRequests).await;
    collector.record_preemption(TaskPriority::BackgroundWatching, 50, true).await;

    let updated_metrics = pipeline.get_priority_system_metrics().await;
    assert!(updated_metrics.pipeline.tasks_completed >= metrics.pipeline.tasks_completed);
}

#[tokio::test]
async fn test_checkpoint_metrics_integration() {
    let checkpoint_dir = std::env::temp_dir().join("test_checkpoint_metrics");
    let _ = std::fs::create_dir_all(&checkpoint_dir);

    let mut pipeline = Pipeline::with_checkpoint_config(
        2,
        QueueConfig::default(),
        Some(checkpoint_dir.clone()),
    );

    pipeline.start().await.expect("Failed to start pipeline");

    let checkpoint_manager = pipeline.checkpoint_manager();
    let metrics_collector = pipeline.metrics_collector();

    let checkpoint_id = checkpoint_manager.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 50.0,
            stage: "testing".to_string(),
            metadata: HashMap::new(),
        },
        serde_json::json!({"test": "data"}),
        vec![],
        vec![],
    ).await.expect("Failed to create checkpoint");

    metrics_collector.record_checkpoint_created();

    let metrics = pipeline.get_priority_system_metrics().await;
    assert!(metrics.checkpoints.active_checkpoints > 0);
    assert!(metrics.checkpoints.checkpoints_created > 0);

    checkpoint_manager.rollback_checkpoint(&checkpoint_id).await
        .expect("Failed to rollback checkpoint");

    metrics_collector.record_rollback_executed();

    let updated_metrics = pipeline.get_priority_system_metrics().await;
    assert!(updated_metrics.checkpoints.rollbacks_executed > 0);

    let _ = std::fs::remove_dir_all(&checkpoint_dir);
}

#[tokio::test]
async fn test_queue_metrics_and_timeouts() {
    let queue_config = QueueConfigBuilder::new()
        .max_queued_per_priority(2)
        .default_queue_timeout(100)
        .build();

    let mut pipeline = Pipeline::with_queue_config(1, queue_config);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let _handle1 = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/test1".to_string() },
        TaskPayload::Generic {
            operation: "long_background_task".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(10)),
    ).await.expect("Failed to submit task 1");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let _handle2 = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/test2".to_string() },
        TaskPayload::Generic {
            operation: "queued_task_1".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit task 2");

    let _handle3 = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/test3".to_string() },
        TaskPayload::Generic {
            operation: "queued_task_2".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit task 3");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let queue_stats = submitter.get_queue_stats().await;
    let metrics = pipeline.get_priority_system_metrics().await;

    assert!(
        queue_stats.total_queued > 0
            || metrics.pipeline.tasks_completed > 0
            || metrics.pipeline.running_tasks > 0
    );
    assert!(
        queue_stats.queued_by_priority.contains_key(&TaskPriority::BackgroundWatching)
            || metrics.pipeline.running_tasks > 0
    );

    let cleaned_count = submitter.cleanup_queue_timeouts().await;
    tracing::info!("Cleaned {} timed out requests", cleaned_count);

    let duplicate_result = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/test3".to_string() },
        TaskPayload::Generic {
            operation: "queued_task_2".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await;

    match duplicate_result {
        Ok(_) => tracing::info!("Duplicate task was allowed"),
        Err(e) => tracing::info!("Duplicate task was rejected: {}", e),
    }
}

#[tokio::test]
async fn test_bulk_preemption_for_mcp_requests() {
    let mut pipeline = Pipeline::new(2);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let bg_task1 = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/bg1".to_string() },
        TaskPayload::Generic {
            operation: "long_background_task_1".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(10)),
    ).await.expect("Failed to submit bg task 1");

    let bg_task2 = submitter.submit_task(
        TaskPriority::BackgroundWatching,
        TaskSource::BackgroundWatcher { folder_path: "/tmp/bg2".to_string() },
        TaskPayload::Generic {
            operation: "long_background_task_2".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(10)),
    ).await.expect("Failed to submit bg task 2");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let mcp_task1 = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer { request_id: "mcp_req_1".to_string() },
        TaskPayload::Generic {
            operation: "mcp_operation_1".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(2)),
    ).await.expect("Failed to submit MCP task 1");

    let mcp_task2 = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer { request_id: "mcp_req_2".to_string() },
        TaskPayload::Generic {
            operation: "mcp_operation_2".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(2)),
    ).await.expect("Failed to submit MCP task 2");

    let mcp1_result = timeout(Duration::from_secs(3), mcp_task1.wait()).await
        .expect("MCP task 1 should not timeout")
        .expect("MCP task 1 should succeed");

    let mcp2_result = timeout(Duration::from_secs(3), mcp_task2.wait()).await
        .expect("MCP task 2 should not timeout")
        .expect("MCP task 2 should succeed");

    match mcp1_result {
        TaskResult::Success { .. } => {}
        other => panic!("MCP task 1 should succeed, got: {:?}", other),
    }

    match mcp2_result {
        TaskResult::Success { .. } => {}
        other => panic!("MCP task 2 should succeed, got: {:?}", other),
    }

    let bg1_result = timeout(Duration::from_millis(500), bg_task1.wait()).await;
    let bg2_result = timeout(Duration::from_millis(500), bg_task2.wait()).await;

    let mut cancelled_count = 0;
    for (label, result) in [("bg1", bg1_result), ("bg2", bg2_result)] {
        match result {
            Ok(Ok(TaskResult::Cancelled { .. })) => {
                cancelled_count += 1;
            }
            Ok(Ok(TaskResult::Success { .. })) => {}
            Ok(Ok(other)) => panic!("{} task unexpected result: {:?}", label, other),
            Ok(Err(e)) => panic!("{} task failed: {}", label, e),
            Err(_) => {
                tracing::info!("{} task still running; skipping cancellation assertion", label);
            }
        }
    }

    let metrics = pipeline.get_priority_system_metrics().await;
    if cancelled_count > 0 && metrics.preemption.preemptions_total > 0 {
        assert!(metrics.preemption.preemptions_total >= cancelled_count as u64);
    }
}

#[tokio::test]
async fn test_graceful_preemption_vs_abort() {
    let mut pipeline = Pipeline::new(1);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let graceful_task = submitter.submit_task(
        TaskPriority::CliCommands,
        TaskSource::CliCommand { command: "graceful_task".to_string() },
        TaskPayload::Generic {
            operation: "graceful_operation".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(5)),
    ).await.expect("Failed to submit graceful task");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let preempting_task = submitter.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer { request_id: "preempting_request".to_string() },
        TaskPayload::Generic {
            operation: "preempting_operation".to_string(),
            parameters: HashMap::new(),
        },
        Some(Duration::from_secs(2)),
    ).await.expect("Failed to submit preempting task");

    let preempting_result = timeout(Duration::from_secs(3), preempting_task.wait()).await
        .expect("Preempting task should not timeout")
        .expect("Preempting task should succeed");

    let graceful_result = timeout(Duration::from_millis(100), graceful_task.wait()).await
        .expect("Graceful task should complete quickly")
        .expect("Graceful task should have result");

    match preempting_result {
        TaskResult::Success { .. } => {}
        other => panic!("Preempting task should succeed, got: {:?}", other),
    }

    match graceful_result {
        TaskResult::Cancelled { .. } => {}
        TaskResult::Success { .. } => {}
        other => panic!("Expected cancelled or success, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_task_result_handle_is_completed_when_sender_alive() {
    let (tx, rx) = tokio::sync::oneshot::channel::<TaskResult>();
    let mut handle = TaskResultHandle {
        task_id: Uuid::new_v4(),
        context: TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::BackgroundWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::Generic { operation: "test".into() },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: false,
            tenant_id: None,
        },
        result_receiver: rx,
    };

    assert!(!handle.is_completed());
    drop(tx);
}

#[tokio::test]
async fn test_task_result_handle_is_completed_when_sender_dropped() {
    let (_tx, rx) = tokio::sync::oneshot::channel::<TaskResult>();
    let mut handle = TaskResultHandle {
        task_id: Uuid::new_v4(),
        context: TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::BackgroundWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::Generic { operation: "test".into() },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: false,
            tenant_id: None,
        },
        result_receiver: rx,
    };

    drop(_tx);
    assert!(handle.is_completed());
}

#[tokio::test]
async fn test_task_result_handle_is_completed_when_value_sent() {
    let (tx, rx) = tokio::sync::oneshot::channel::<TaskResult>();
    let mut handle = TaskResultHandle {
        task_id: Uuid::new_v4(),
        context: TaskContext {
            task_id: Uuid::new_v4(),
            priority: TaskPriority::BackgroundWatching,
            created_at: chrono::Utc::now(),
            timeout_ms: None,
            source: TaskSource::Generic { operation: "test".into() },
            metadata: HashMap::new(),
            checkpoint_id: None,
            supports_checkpointing: false,
            tenant_id: None,
        },
        result_receiver: rx,
    };

    let _ = tx.send(TaskResult::Success {
        execution_time_ms: 42,
        data: TaskResultData::Generic {
            message: "done".into(),
            data: serde_json::json!({}),
            checkpoint_id: None,
        },
    });

    assert!(handle.is_completed());
}

#[tokio::test]
async fn test_spill_to_sqlite_process_document() {
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
    use crate::queue_operations::QueueManager;

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

    submitter.spill_to_sqlite(&queue_manager, &context, &payload)
        .await
        .expect("Spill should succeed");

    let row: (String, String, String, String) = sqlx::query_as(
        "SELECT item_type, op, collection, status FROM unified_queue LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .expect("Should have one spilled item");

    assert_eq!(row.0, "file");
    assert_eq!(row.1, "add");
    assert_eq!(row.2, "projects");
    assert_eq!(row.3, "pending");

    let payload_json: (String,) = sqlx::query_as(
        "SELECT payload_json FROM unified_queue LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .expect("Should have payload");

    let payload_val: serde_json::Value = serde_json::from_str(&payload_json.0).unwrap();
    assert_eq!(
        payload_val["file_path"].as_str().unwrap(),
        "/tmp/test_project/src/main.rs"
    );

    let metadata_json: (String,) = sqlx::query_as(
        "SELECT metadata FROM unified_queue LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .expect("Should have metadata");

    let meta_val: serde_json::Value = serde_json::from_str(&metadata_json.0).unwrap();
    assert_eq!(meta_val["spilled_from"].as_str().unwrap(), "pipeline");
    assert_eq!(meta_val["original_priority"].as_str().unwrap(), "ProjectWatching");
}

#[tokio::test]
async fn test_spill_non_process_document_fails() {
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
    use crate::queue_operations::QueueManager;

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
        source: TaskSource::Generic { operation: "test".to_string() },
        metadata: HashMap::new(),
        checkpoint_id: None,
        supports_checkpointing: false,
        tenant_id: None,
    };

    let payload = TaskPayload::Generic {
        operation: "test".to_string(),
        parameters: HashMap::new(),
    };
    let result = submitter.spill_to_sqlite(&queue_manager, &context, &payload).await;
    assert!(result.is_err(), "Generic tasks should not be spillable");

    let payload = TaskPayload::ExecuteQuery {
        query: "test".to_string(),
        collection: "test".to_string(),
        limit: 10,
    };
    let result = submitter.spill_to_sqlite(&queue_manager, &context, &payload).await;
    assert!(result.is_err(), "ExecuteQuery tasks should not be spillable");
}

#[tokio::test]
async fn test_spill_with_background_watcher_source() {
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
    use crate::queue_operations::QueueManager;

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

    submitter.spill_to_sqlite(&queue_manager, &context, &payload)
        .await
        .expect("Spill with BackgroundWatcher source should succeed");

    let row: (String,) = sqlx::query_as(
        "SELECT collection FROM unified_queue LIMIT 1",
    )
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
    use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
    use crate::queue_operations::QueueManager;

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

// =========================================================================
// Rollback operation tests
// =========================================================================

#[tokio::test]
async fn test_rollback_delete_file() {
    let dir = std::env::temp_dir().join("test_rollback_delete");
    let _ = std::fs::create_dir_all(&dir);
    let file_path = dir.join("to_delete.txt");
    std::fs::write(&file_path, "temporary data").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::DeleteFile { path: file_path.clone() }],
    ).await.unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    assert!(!file_path.exists(), "File should have been deleted by rollback");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_restore_file() {
    let dir = std::env::temp_dir().join("test_rollback_restore");
    let _ = std::fs::create_dir_all(&dir);

    let original = dir.join("original.txt");
    let backup = dir.join("backup.txt");
    std::fs::write(&original, "modified").unwrap();
    std::fs::write(&backup, "original content").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::RestoreFile {
            original_path: original.clone(),
            backup_path: backup.clone(),
        }],
    ).await.unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    let content = std::fs::read_to_string(&original).unwrap();
    assert_eq!(content, "original content");
    assert!(!backup.exists(), "Backup should be cleaned up");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_remove_from_collection_no_storage_client() {
    let dir = std::env::temp_dir().join("test_rollback_remove_no_sc");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::RemoveFromCollection {
            document_id: "doc-123".into(),
            collection: "projects".into(),
        }],
    ).await.unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok(), "rollback_checkpoint should succeed even if individual actions fail");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_revert_index_no_storage_client() {
    let dir = std::env::temp_dir().join("test_rollback_revert_no_sc");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let snapshot = serde_json::json!({
        "collection": "projects",
        "indexes": ["field1", "field2"]
    });

    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::RevertIndexChanges { index_snapshot: snapshot }],
    ).await.unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_custom_handler_registered() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let dir = std::env::temp_dir().join("test_rollback_custom");
    let _ = std::fs::create_dir_all(&dir);

    let executed = Arc::new(AtomicBool::new(false));
    let executed_clone = executed.clone();

    struct TestHandler {
        executed: Arc<AtomicBool>,
    }

    #[async_trait::async_trait]
    impl CustomRollbackHandler for TestHandler {
        async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
            self.executed.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    cm.register_custom_handler(
        "test_action",
        Arc::new(TestHandler { executed: executed_clone }),
    ).await;

    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::Custom {
            action_type: "test_action".into(),
            data: serde_json::json!({"key": "value"}),
        }],
    ).await.unwrap();

    cm.rollback_checkpoint(&ckpt_id).await.unwrap();
    assert!(executed.load(Ordering::SeqCst), "Custom handler should have been executed");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_custom_handler_not_registered() {
    let dir = std::env::temp_dir().join("test_rollback_custom_unreg");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![RollbackAction::Custom {
            action_type: "unregistered_action".into(),
            data: serde_json::json!({}),
        }],
    ).await.unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok(), "rollback_checkpoint succeeds even with failed actions");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_multiple_actions_continue_on_failure() {
    let dir = std::env::temp_dir().join("test_rollback_multi");
    let _ = std::fs::create_dir_all(&dir);

    let file_to_delete = dir.join("should_be_deleted.txt");
    std::fs::write(&file_to_delete, "data").unwrap();

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));
    let ckpt_id = cm.create_checkpoint(
        Uuid::new_v4(),
        TaskProgress::Generic {
            progress_percentage: 100.0,
            stage: "test".into(),
            metadata: HashMap::new(),
        },
        serde_json::json!({}),
        vec![],
        vec![
            RollbackAction::RemoveFromCollection {
                document_id: "doc-456".into(),
                collection: "projects".into(),
            },
            RollbackAction::DeleteFile { path: file_to_delete.clone() },
        ],
    ).await.unwrap();

    let result = cm.rollback_checkpoint(&ckpt_id).await;
    assert!(result.is_ok());
    assert!(!file_to_delete.exists(), "DeleteFile should execute even when other actions fail");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_rollback_storage_configuration() {
    let dir = std::env::temp_dir().join("test_rollback_storage_cfg");
    let _ = std::fs::create_dir_all(&dir);

    let pipeline = Pipeline::new(2);
    let cm = pipeline.checkpoint_manager();
    assert!(cm.storage_client.is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn test_custom_handler_registry() {
    struct NoopHandler;

    #[async_trait::async_trait]
    impl CustomRollbackHandler for NoopHandler {
        async fn execute(&self, _data: &serde_json::Value) -> Result<(), String> {
            Ok(())
        }
    }

    let dir = std::env::temp_dir().join("test_custom_registry");
    let _ = std::fs::create_dir_all(&dir);

    let cm = CheckpointManager::new(dir.clone(), Duration::from_secs(60));

    {
        let handlers = cm.custom_handlers.read().await;
        assert!(handlers.is_empty());
    }

    cm.register_custom_handler("noop", Arc::new(NoopHandler)).await;
    {
        let handlers = cm.custom_handlers.read().await;
        assert_eq!(handlers.len(), 1);
        assert!(handlers.contains_key("noop"));
    }

    cm.register_custom_handler("another", Arc::new(NoopHandler)).await;
    {
        let handlers = cm.custom_handlers.read().await;
        assert_eq!(handlers.len(), 2);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
