use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::RwLock;
use uuid::Uuid;

use crate::processing::{TaskPayload, TaskPriority, TaskResultHandle, TaskSource};

#[tokio::test]
async fn test_ipc_server_client_communication() {
    let (server, client) = IpcServer::new(2);

    server.start().await.expect("Failed to start IPC server");

    let request_id = client.health_check().await.expect("Failed to send health check");

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client.recv_response()
    ).await.expect("Health check timed out").expect("Failed to receive response");

    match response {
        IpcResponse::HealthCheckOk { request_id: resp_id, status } => {
            assert_eq!(resp_id, request_id);
            assert_eq!(status, "OK");
        }
        other => panic!("Expected HealthCheckOk, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_ipc_task_submission() {
    let (server, client) = IpcServer::new(2);

    server.start().await.expect("Failed to start IPC server");

    let request_id = client.submit_task(
        TaskPriority::McpRequests,
        TaskSource::McpServer {
            request_id: "test_request".to_string(),
        },
        TaskPayload::Generic {
            operation: "test_operation".to_string(),
            parameters: HashMap::new(),
        },
        Some(5000),
    ).await.expect("Failed to submit task");

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client.recv_response()
    ).await.expect("Task submission timed out").expect("Failed to receive response");

    match response {
        IpcResponse::TaskSubmitted { task_id: _, request_id: resp_id } => {
            assert_eq!(resp_id, request_id);
        }
        other => panic!("Expected TaskSubmitted, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_ipc_get_stats() {
    let (server, client) = IpcServer::new(3);

    server.start().await.expect("Failed to start IPC server");

    let request_id = client.get_stats().await.expect("Failed to get stats");

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client.recv_response()
    ).await.expect("Get stats timed out").expect("Failed to receive response");

    match response {
        IpcResponse::Stats { stats, request_id: resp_id } => {
            assert_eq!(resp_id, request_id);
            assert_eq!(stats.total_capacity, 3);
        }
        other => panic!("Expected Stats, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_ipc_configuration() {
    let (server, client) = IpcServer::new(2);

    server.start().await.expect("Failed to start IPC server");

    let settings = EngineSettings {
        max_concurrent_tasks: Some(5),
        default_timeout_ms: Some(10000),
        enable_preemption: Some(false),
        log_level: Some("debug".to_string()),
    };

    let request_id = client.configure(settings).await.expect("Failed to configure");

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client.recv_response()
    ).await.expect("Configuration timed out").expect("Failed to receive response");

    match response {
        IpcResponse::ConfigurationApplied { request_id: resp_id } => {
            assert_eq!(resp_id, request_id);
        }
        other => panic!("Expected ConfigurationApplied, got: {:?}", other),
    }
}

/// Helper to create a test TaskResultHandle
fn make_test_handle(
    id: Uuid,
    rx: tokio::sync::oneshot::Receiver<crate::processing::TaskResult>,
) -> crate::processing::TaskResultHandle {
    use crate::processing::TaskContext;
    use chrono::Utc;

    let ctx = TaskContext {
        task_id: id,
        priority: TaskPriority::BackgroundWatching,
        created_at: Utc::now(),
        timeout_ms: None,
        source: TaskSource::Generic { operation: "test".into() },
        metadata: HashMap::new(),
        checkpoint_id: None,
        supports_checkpointing: false,
        tenant_id: None,
    };

    crate::processing::TaskResultHandle::new_for_test(id, ctx, rx)
}

#[tokio::test]
async fn test_cleanup_completed_tasks_removes_finished() {
    use crate::processing::TaskResult;
    use tokio::sync::oneshot;

    let active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>> =
        Arc::new(RwLock::new(HashMap::new()));

    // Create a "completed" task (sender dropped)
    let id1 = Uuid::new_v4();
    let (_tx1, rx1) = oneshot::channel::<TaskResult>();
    drop(_tx1); // simulate completion
    let handle1 = make_test_handle(id1, rx1);

    // Create an "active" task (sender still alive)
    let id2 = Uuid::new_v4();
    let (tx2, rx2) = oneshot::channel::<TaskResult>();
    let handle2 = make_test_handle(id2, rx2);

    {
        let mut lock = active_tasks.write().await;
        lock.insert(id1, handle1);
        lock.insert(id2, handle2);
    }

    assert_eq!(active_tasks.read().await.len(), 2);

    IpcServer::cleanup_completed_tasks(&active_tasks).await;

    let remaining = active_tasks.read().await;
    assert_eq!(remaining.len(), 1);
    assert!(remaining.contains_key(&id2));

    drop(tx2);
}

#[tokio::test]
async fn test_cleanup_completed_tasks_noop_when_all_active() {
    use crate::processing::TaskResult;
    use tokio::sync::oneshot;

    let active_tasks: Arc<RwLock<HashMap<Uuid, TaskResultHandle>>> =
        Arc::new(RwLock::new(HashMap::new()));

    let id1 = Uuid::new_v4();
    let (tx1, rx1) = oneshot::channel::<TaskResult>();
    let handle1 = make_test_handle(id1, rx1);

    {
        let mut lock = active_tasks.write().await;
        lock.insert(id1, handle1);
    }

    IpcServer::cleanup_completed_tasks(&active_tasks).await;

    assert_eq!(active_tasks.read().await.len(), 1);

    drop(tx1);
}

#[tokio::test]
async fn test_ipc_shutdown() {
    let (server, client) = IpcServer::new(2);

    server.start().await.expect("Failed to start IPC server");

    let request_id = client.shutdown(true, Some(1000)).await.expect("Failed to shutdown");

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        client.recv_response()
    ).await.expect("Shutdown timed out").expect("Failed to receive response");

    match response {
        IpcResponse::ShutdownAck { request_id: resp_id } => {
            assert_eq!(resp_id, request_id);
        }
        other => panic!("Expected ShutdownAck, got: {:?}", other),
    }

    tokio::time::timeout(
        Duration::from_secs(2),
        server.wait_for_shutdown()
    ).await.expect("Server did not shut down in time");
}
