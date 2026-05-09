//! Tests for SystemService RPC methods: shutdown, server status notifications,
//! refresh signals, and watcher pause/resume.

use tonic::Request;

use crate::proto::{
    system_service_server::SystemService, QueueType, RefreshSignalRequest, ServerState,
    ServerStatusNotification,
};

use super::super::service_impl::SystemServiceImpl;

#[tokio::test]
async fn test_shutdown() {
    let service = SystemServiceImpl::new();
    let response = service.shutdown(Request::new(())).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_status_store_initialized_empty() {
    let service = SystemServiceImpl::new();
    let store = service.status_store.read().await;
    assert!(store.is_empty());
}

#[tokio::test]
async fn test_notify_server_status_stores_entry() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("test-project".to_string()),
        project_root: Some("/tmp/test-project".to_string()),
    };

    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    // Verify the entry was stored
    let store = service.status_store.read().await;
    assert_eq!(store.len(), 1);
    let entry = store.get("test-project").unwrap();
    assert_eq!(entry.state, ServerState::Up);
    assert_eq!(entry.project_name.as_deref(), Some("test-project"));
    assert_eq!(entry.project_root.as_deref(), Some("/tmp/test-project"));
}

#[tokio::test]
async fn test_notify_server_status_transitions() {
    let service = SystemServiceImpl::new();

    // First: UP
    let up_notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("my-app".to_string()),
        project_root: Some("/home/user/my-app".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(up_notification))
        .await;
    assert!(response.is_ok());

    // Then: DOWN
    let down_notification = ServerStatusNotification {
        state: ServerState::Down as i32,
        project_name: Some("my-app".to_string()),
        project_root: Some("/home/user/my-app".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(down_notification))
        .await;
    assert!(response.is_ok());

    // Verify the final state is DOWN
    let store = service.status_store.read().await;
    let entry = store.get("my-app").unwrap();
    assert_eq!(entry.state, ServerState::Down);
}

#[tokio::test]
async fn test_notify_server_status_uses_project_root_as_fallback_key() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: None,
        project_root: Some("/tmp/fallback".to_string()),
    };
    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    let store = service.status_store.read().await;
    assert!(store.contains_key("/tmp/fallback"));
}

#[tokio::test]
async fn test_notify_server_status_unknown_fallback() {
    let service = SystemServiceImpl::new();

    let notification = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: None,
        project_root: None,
    };
    let response = service
        .notify_server_status(Request::new(notification))
        .await;
    assert!(response.is_ok());

    let store = service.status_store.read().await;
    assert!(store.contains_key("unknown"));
}

#[tokio::test]
async fn test_send_refresh_signal_without_db_pool() {
    // Without a database pool, refresh signal should return Ok but do nothing
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::IngestQueue as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_tools_available() {
    // ToolsAvailable is informational and should always succeed
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::ToolsAvailable as i32,
        lsp_languages: vec!["rust".to_string(), "python".to_string()],
        grammar_languages: vec!["javascript".to_string()],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_send_refresh_signal_unspecified() {
    let service = SystemServiceImpl::new();

    let request = RefreshSignalRequest {
        queue_type: QueueType::Unspecified as i32,
        lsp_languages: vec![],
        grammar_languages: vec![],
    };
    let response = service.send_refresh_signal(Request::new(request)).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_multiple_components_tracked_independently() {
    let service = SystemServiceImpl::new();

    // Register two different components
    let notification1 = ServerStatusNotification {
        state: ServerState::Up as i32,
        project_name: Some("project-a".to_string()),
        project_root: Some("/tmp/a".to_string()),
    };
    let notification2 = ServerStatusNotification {
        state: ServerState::Down as i32,
        project_name: Some("project-b".to_string()),
        project_root: Some("/tmp/b".to_string()),
    };

    service
        .notify_server_status(Request::new(notification1))
        .await
        .unwrap();
    service
        .notify_server_status(Request::new(notification2))
        .await
        .unwrap();

    let store = service.status_store.read().await;
    assert_eq!(store.len(), 2);
    assert_eq!(store.get("project-a").unwrap().state, ServerState::Up);
    assert_eq!(store.get("project-b").unwrap().state, ServerState::Down);
}

#[tokio::test]
async fn test_pause_all_watchers_without_db_pool() {
    let service = SystemServiceImpl::new();
    let response = service.pause_all_watchers(Request::new(())).await;
    assert!(response.is_ok());
}

#[tokio::test]
async fn test_resume_all_watchers_without_db_pool() {
    let service = SystemServiceImpl::new();
    let response = service.resume_all_watchers(Request::new(())).await;
    assert!(response.is_ok());
}
