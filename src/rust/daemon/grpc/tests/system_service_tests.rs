//! Integration tests for SystemService
//!
//! Tests the complete SystemService gRPC implementation including:
//! - HealthCheck RPC with component health reporting
//! - GetStatus RPC for system state snapshots
//! - GetMetrics RPC for performance metrics
//! - Message serialization/deserialization
//! - Response format validation
//! - Status code verification

use workspace_qdrant_grpc::proto::{
    system_service_server::SystemService,
    HealthResponse, SystemStatusResponse, MetricsResponse,
    RefreshSignalRequest, ServerStatusNotification,
    ServiceStatus, QueueType, ServerState,
};
use workspace_qdrant_grpc::services::SystemServiceImpl;
use tonic::Request;
use prost::Message;

/// Test helper to create SystemService instance
fn create_service() -> SystemServiceImpl {
    SystemServiceImpl::default()
}

#[tokio::test]
async fn test_health_check_success() {
    let service = create_service();
    let request = Request::new(());

    let response = service.health(request).await;
    assert!(response.is_ok());

    let health_response = response.unwrap().into_inner();
    assert_eq!(health_response.status, ServiceStatus::Healthy as i32);
    assert!(!health_response.components.is_empty());
    assert!(health_response.timestamp.is_some());
}

#[tokio::test]
async fn test_health_check_components() {
    let service = create_service();
    let request = Request::new(());

    let response = service.health(request).await.unwrap();
    let health_response = response.into_inner();

    // Should have at least one component
    assert!(!health_response.components.is_empty());

    // Check first component structure
    let component = &health_response.components[0];
    assert_eq!(component.component_name, "grpc_server");
    assert_eq!(component.status, ServiceStatus::Healthy as i32);
    assert!(!component.message.is_empty());
    assert!(component.last_check.is_some());
}

#[tokio::test]
async fn test_health_check_serialization() {
    let service = create_service();
    let request = Request::new(());

    let response = service.health(request).await.unwrap();
    let health_response = response.into_inner();

    // Test protobuf serialization
    let bytes = health_response.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = HealthResponse::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.status, ServiceStatus::Healthy as i32);
    assert_eq!(decoded.components.len(), health_response.components.len());
}

#[tokio::test]
async fn test_get_status_success() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_status(request).await;
    assert!(response.is_ok());

    let status_response = response.unwrap().into_inner();
    assert_eq!(status_response.status, ServiceStatus::Healthy as i32);
    assert!(status_response.metrics.is_some());
    assert!(status_response.uptime_since.is_some());
}

#[tokio::test]
async fn test_get_status_metrics_structure() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_status(request).await.unwrap();
    let status_response = response.into_inner();

    let metrics = status_response.metrics.unwrap();
    assert_eq!(metrics.cpu_usage_percent, 0.0);
    assert_eq!(metrics.memory_usage_bytes, 0);
    assert_eq!(metrics.memory_total_bytes, 0);
    assert_eq!(metrics.disk_usage_bytes, 0);
    assert_eq!(metrics.disk_total_bytes, 0);
    assert_eq!(metrics.active_connections, 1);
    assert_eq!(metrics.pending_operations, 0);
}

#[tokio::test]
async fn test_get_status_project_collections() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_status(request).await.unwrap();
    let status_response = response.into_inner();

    // Empty in stub implementation
    assert_eq!(status_response.active_projects.len(), 0);
    assert_eq!(status_response.total_documents, 0);
    assert_eq!(status_response.total_collections, 0);
}

#[tokio::test]
async fn test_get_status_serialization() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_status(request).await.unwrap();
    let status_response = response.into_inner();

    // Test protobuf serialization
    let bytes = status_response.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = SystemStatusResponse::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.status, ServiceStatus::Healthy as i32);
    assert!(decoded.metrics.is_some());
    assert!(decoded.uptime_since.is_some());
}

#[tokio::test]
async fn test_get_metrics_success() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_metrics(request).await;
    assert!(response.is_ok());

    let metrics_response = response.unwrap().into_inner();
    assert!(metrics_response.collected_at.is_some());
    assert!(!metrics_response.metrics.is_empty());
}

#[tokio::test]
async fn test_get_metrics_structure() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_metrics(request).await.unwrap();
    let metrics_response = response.into_inner();

    // Check first metric structure
    assert!(!metrics_response.metrics.is_empty());
    let metric = &metrics_response.metrics[0];
    assert_eq!(metric.name, "requests_total");
    assert_eq!(metric.r#type, "counter");
    assert!(metric.timestamp.is_some());
}

#[tokio::test]
async fn test_get_metrics_serialization() {
    let service = create_service();
    let request = Request::new(());

    let response = service.get_metrics(request).await.unwrap();
    let metrics_response = response.into_inner();

    // Test protobuf serialization
    let bytes = metrics_response.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = MetricsResponse::decode(&bytes[..]).unwrap();
    assert!(decoded.collected_at.is_some());
    assert_eq!(decoded.metrics.len(), metrics_response.metrics.len());
}

#[tokio::test]
async fn test_send_refresh_signal_ingest_queue() {
    let service = create_service();
    let request = Request::new(RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    });

    let response = service.send_refresh_signal(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_watched_projects() {
    let service = create_service();
    let request = Request::new(RefreshSignalRequest {
        queue_type: QueueType::WatchedProjects as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    });

    let response = service.send_refresh_signal(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_tools_available() {
    let service = create_service();
    let request = Request::new(RefreshSignalRequest {
        queue_type: QueueType::ToolsAvailable as i32,
        lsp_languages: vec!["rust".to_string(), "python".to_string()],
        grammar_languages: vec!["tree-sitter-rust".to_string()],
    });

    let response = service.send_refresh_signal(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_serialization() {
    let signal = RefreshSignalRequest {
        queue_type: QueueType::WatchedFolders as i32,
        lsp_languages: vec!["typescript".to_string()],
        grammar_languages: vec![],
    };

    // Test serialization
    let bytes = signal.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = RefreshSignalRequest::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.queue_type, QueueType::WatchedFolders as i32);
    assert_eq!(decoded.lsp_languages.len(), 1);
    assert_eq!(decoded.lsp_languages[0], "typescript");
}

#[tokio::test]
async fn test_notify_server_status_up() {
    let service = create_service();
    let request = Request::new(ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("test-project".to_string()),
        project_root: Some("/path/to/project".to_string()),
    });

    let response = service.notify_server_status(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_notify_server_status_down() {
    let service = create_service();
    let request = Request::new(ServerStatusNotification {
        state: ServerState::Down as i32,
        project_name: Some("test-project".to_string()),
        project_root: None,
    });

    let response = service.notify_server_status(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_notify_server_status_serialization() {
    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("workspace-qdrant-mcp".to_string()),
        project_root: Some("/Users/test/projects/workspace-qdrant-mcp".to_string()),
    };

    // Test serialization
    let bytes = notification.encode_to_vec();
    assert!(!bytes.is_empty());

    // Test deserialization
    let decoded = ServerStatusNotification::decode(&bytes[..]).unwrap();
    assert_eq!(decoded.state, ServerState::Up as i32);
    assert_eq!(decoded.project_name, Some("workspace-qdrant-mcp".to_string()));
}

#[tokio::test]
async fn test_pause_all_watchers() {
    let service = create_service();
    let request = Request::new(());

    let response = service.pause_all_watchers(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_resume_all_watchers() {
    let service = create_service();
    let request = Request::new(());

    let response = service.resume_all_watchers(request).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_multiple_health_checks() {
    let service = create_service();

    // Multiple calls should all succeed
    for _ in 0..5 {
        let response = service.health(Request::new(())).await;
        assert!(response.is_ok());

        let health_response = response.unwrap().into_inner();
        assert_eq!(health_response.status, ServiceStatus::Healthy as i32);
    }
}

#[tokio::test]
async fn test_concurrent_status_requests() {
    use tokio::task::JoinSet;

    let service = std::sync::Arc::new(create_service());
    let mut join_set = JoinSet::new();

    // Spawn 10 concurrent status requests
    for _ in 0..10 {
        let service_clone = service.clone();
        join_set.spawn(async move {
            service_clone.get_status(Request::new(())).await
        });
    }

    // All should succeed
    while let Some(result) = join_set.join_next().await {
        let response = result.unwrap();
        assert!(response.is_ok());
    }
}

#[tokio::test]
async fn test_enum_serialization() {
    // Test ServiceStatus enum values
    assert_eq!(ServiceStatus::Unspecified as i32, 0);
    assert_eq!(ServiceStatus::Healthy as i32, 1);
    assert_eq!(ServiceStatus::Degraded as i32, 2);
    assert_eq!(ServiceStatus::Unhealthy as i32, 3);
    assert_eq!(ServiceStatus::Unavailable as i32, 4);

    // Test QueueType enum values
    assert_eq!(QueueType::Unspecified as i32, 0);
    assert_eq!(QueueType::IngestQueue as i32, 1);
    assert_eq!(QueueType::WatchedProjects as i32, 2);
    assert_eq!(QueueType::WatchedFolders as i32, 3);
    assert_eq!(QueueType::ToolsAvailable as i32, 4);

    // Test ServerState enum values
    assert_eq!(ServerState::Unspecified as i32, 0);
    assert_eq!(ServerState::Up as i32, 1);
    assert_eq!(ServerState::Down as i32, 2);
}

#[tokio::test]
async fn test_timestamp_fields() {
    let service = create_service();

    // HealthCheck timestamp
    let health = service.health(Request::new(())).await.unwrap().into_inner();
    assert!(health.timestamp.is_some());
    let ts = health.timestamp.unwrap();
    assert!(ts.seconds > 0);

    // GetStatus uptime_since
    let status = service.get_status(Request::new(())).await.unwrap().into_inner();
    assert!(status.uptime_since.is_some());

    // GetMetrics collected_at
    let metrics = service.get_metrics(Request::new(())).await.unwrap().into_inner();
    assert!(metrics.collected_at.is_some());
}

#[tokio::test]
async fn test_response_consistency() {
    let service = create_service();

    // Get status twice and check uptime_since is the same (service start time)
    let status1 = service.get_status(Request::new(())).await.unwrap().into_inner();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let status2 = service.get_status(Request::new(())).await.unwrap().into_inner();

    // Uptime start time should be identical across calls
    assert_eq!(status1.uptime_since, status2.uptime_since);
}

#[tokio::test]
async fn test_empty_message_serialization() {
    // Test that empty () messages serialize correctly
    let empty = ();
    let bytes = prost::Message::encode_to_vec(&());
    assert_eq!(bytes.len(), 0); // Empty message has zero bytes
}
