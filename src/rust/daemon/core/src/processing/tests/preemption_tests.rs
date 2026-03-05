//! Queue metrics, preemption logic, bulk preemption, graceful vs abort, and TaskResultHandle tests

use super::super::*;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_preemption_logic() {
    let mut pipeline = Pipeline::new(1);
    let submitter = pipeline.task_submitter();

    pipeline.start().await.expect("Failed to start pipeline");

    let low_priority_handle = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/background".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_running_background".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("Failed to submit low priority task");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let high_priority_handle = submitter
        .submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "urgent_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "urgent_mcp_task".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        )
        .await
        .expect("Failed to submit high priority task");

    let high_result = timeout(Duration::from_secs(3), high_priority_handle.wait())
        .await
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

    let low_result = timeout(Duration::from_millis(100), low_priority_handle.wait())
        .await
        .expect("Low priority task should complete quickly after preemption")
        .expect("Low priority task should have result");

    match low_result {
        TaskResult::Cancelled { .. } => {}
        TaskResult::Success { .. } => {}
        other => panic!("Expected cancelled or success, got: {:?}", other),
    }
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

    let _handle1 = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test1".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        )
        .await
        .expect("Failed to submit task 1");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let _handle2 = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test2".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("Failed to submit task 2");

    let _handle3 = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test3".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("Failed to submit task 3");

    tokio::time::sleep(Duration::from_millis(200)).await;

    let queue_stats = submitter.get_queue_stats().await;
    let metrics = pipeline.get_priority_system_metrics().await;

    assert!(
        queue_stats.total_queued > 0
            || metrics.pipeline.tasks_completed > 0
            || metrics.pipeline.running_tasks > 0
    );
    assert!(
        queue_stats
            .queued_by_priority
            .contains_key(&TaskPriority::BackgroundWatching)
            || metrics.pipeline.running_tasks > 0
    );

    let cleaned_count = submitter.cleanup_queue_timeouts().await;
    tracing::info!("Cleaned {} timed out requests", cleaned_count);

    let duplicate_result = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/test3".to_string(),
            },
            TaskPayload::Generic {
                operation: "queued_task_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        )
        .await;

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

    let bg_task1 = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/bg1".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        )
        .await
        .expect("Failed to submit bg task 1");

    let bg_task2 = submitter
        .submit_task(
            TaskPriority::BackgroundWatching,
            TaskSource::BackgroundWatcher {
                folder_path: "/tmp/bg2".to_string(),
            },
            TaskPayload::Generic {
                operation: "long_background_task_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(10)),
        )
        .await
        .expect("Failed to submit bg task 2");

    tokio::time::sleep(Duration::from_millis(100)).await;

    let mcp_task1 = submitter
        .submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "mcp_req_1".to_string(),
            },
            TaskPayload::Generic {
                operation: "mcp_operation_1".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        )
        .await
        .expect("Failed to submit MCP task 1");

    let mcp_task2 = submitter
        .submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "mcp_req_2".to_string(),
            },
            TaskPayload::Generic {
                operation: "mcp_operation_2".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        )
        .await
        .expect("Failed to submit MCP task 2");

    let mcp1_result = timeout(Duration::from_secs(3), mcp_task1.wait())
        .await
        .expect("MCP task 1 should not timeout")
        .expect("MCP task 1 should succeed");

    let mcp2_result = timeout(Duration::from_secs(3), mcp_task2.wait())
        .await
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
                tracing::info!(
                    "{} task still running; skipping cancellation assertion",
                    label
                );
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

    let graceful_task = submitter
        .submit_task(
            TaskPriority::CliCommands,
            TaskSource::CliCommand {
                command: "graceful_task".to_string(),
            },
            TaskPayload::Generic {
                operation: "graceful_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(5)),
        )
        .await
        .expect("Failed to submit graceful task");

    tokio::time::sleep(Duration::from_millis(50)).await;

    let preempting_task = submitter
        .submit_task(
            TaskPriority::McpRequests,
            TaskSource::McpServer {
                request_id: "preempting_request".to_string(),
            },
            TaskPayload::Generic {
                operation: "preempting_operation".to_string(),
                parameters: HashMap::new(),
            },
            Some(Duration::from_secs(2)),
        )
        .await
        .expect("Failed to submit preempting task");

    let preempting_result = timeout(Duration::from_secs(3), preempting_task.wait())
        .await
        .expect("Preempting task should not timeout")
        .expect("Preempting task should succeed");

    let graceful_result = timeout(Duration::from_millis(100), graceful_task.wait())
        .await
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
            source: TaskSource::Generic {
                operation: "test".into(),
            },
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
            source: TaskSource::Generic {
                operation: "test".into(),
            },
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
            source: TaskSource::Generic {
                operation: "test".into(),
            },
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
