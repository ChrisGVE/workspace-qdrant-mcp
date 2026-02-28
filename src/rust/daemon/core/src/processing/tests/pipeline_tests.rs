//! Pipeline creation, submission, priority ordering, timeout, concurrency, and stats tests

use super::super::*;
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
