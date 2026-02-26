//! Tests for ProjectService gRPC implementation

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sqlx::SqlitePool;
use tempfile::tempdir;
use tokio::sync::RwLock;
use tonic::Request;

use workspace_qdrant_core::{DaemonStateManager, PriorityManager, ProjectLanguageDetector};

use crate::proto::project_service_server::ProjectService;
use crate::proto::{
    DeprioritizeProjectRequest, GetProjectStatusRequest, HeartbeatRequest, ListProjectsRequest,
    RegisterProjectRequest,
};

use super::ProjectServiceImpl;

/// Helper to create test database with schema
async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_project_service.db");

    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
    let pool = SqlitePool::connect(&db_url).await.unwrap();

    sqlx::query(workspace_qdrant_core::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query(workspace_qdrant_core::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .unwrap();

    (pool, temp_dir)
}

/// Helper to create a watch_folder entry for a project (simulates daemon creating the project)
async fn create_test_watch_folder(pool: &SqlitePool, project_id: &str, path: &str) {
    let now = chrono::Utc::now().to_rfc3339();
    let watch_id = format!("test-{project_id}");
    sqlx::query(
        r#"
        INSERT INTO watch_folders (
            watch_id, path, collection, tenant_id, is_active,
            follow_symlinks, enabled, cleanup_on_disable, created_at, updated_at
        ) VALUES (?1, ?2, 'projects', ?3, 0, 0, 1, 0, ?4, ?4)
    "#,
    )
    .bind(&watch_id)
    .bind(path)
    .bind(project_id)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}

/// Alias for setup_test_db (includes all required tables)
async fn setup_test_db_with_queue() -> (SqlitePool, tempfile::TempDir) {
    setup_test_db().await
}

/// Helper to construct a ProjectServiceImpl with custom fields for testing
fn build_test_service(pool: SqlitePool, deactivation_delay_secs: u64) -> ProjectServiceImpl {
    ProjectServiceImpl {
        priority_manager: PriorityManager::new(pool.clone()),
        state_manager: DaemonStateManager::with_pool(pool.clone()),
        db_pool: pool,
        lsp_manager: None,
        language_detector: Arc::new(ProjectLanguageDetector::new()),
        deactivation_delay_secs,
        pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        watch_refresh_signal: None,
        storage: None,
    }
}

#[tokio::test]
async fn test_register_new_project_with_register_if_new() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: true,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.created);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.priority, "high");
    assert!(response.is_active);
    assert!(response.newly_registered);
}

#[tokio::test]
async fn test_register_new_project_without_register_if_new() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.priority, "none");
    assert!(!response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_register_existing_project() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("Test Project".to_string()),
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert!(response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_deprioritize_project() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.deprioritize_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.success);
    assert!(!response.is_active);
    assert_eq!(response.new_priority, "normal");
}

#[tokio::test]
async fn test_get_project_status() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;
    sqlx::query("UPDATE watch_folders SET git_remote_url = ?1 WHERE tenant_id = ?2")
        .bind("https://github.com/user/repo.git")
        .bind("abcd12345678")
        .execute(&pool)
        .await
        .unwrap();

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: Some("My Project".to_string()),
        git_remote: Some("https://github.com/user/repo.git".to_string()),
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(GetProjectStatusRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.get_project_status(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.found);
    assert_eq!(response.project_id, "abcd12345678");
    assert_eq!(response.project_name, "project");
    assert_eq!(response.project_root, "/test/project");
    assert_eq!(response.priority, "high");
    assert!(response.is_active);
    assert_eq!(
        response.git_remote,
        Some("https://github.com/user/repo.git".to_string())
    );
}

#[tokio::test]
async fn test_get_nonexistent_project_status() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(GetProjectStatusRequest {
        project_id: "nonexistent12".to_string(),
    });

    let response = service.get_project_status(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.found);
}

#[tokio::test]
async fn test_list_projects() {
    let (pool, _temp_dir) = setup_test_db().await;

    let project_ids = ["aaa000000001", "bbb000000002", "ccc000000003"];
    for (i, project_id) in project_ids.iter().enumerate() {
        create_test_watch_folder(&pool, project_id, &format!("/test/project{i}")).await;
    }

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(ListProjectsRequest {
        priority_filter: None,
        active_only: false,
    });

    let response = service.list_projects(request).await.unwrap();
    let response = response.into_inner();

    assert_eq!(response.total_count, 3);
    assert_eq!(response.projects.len(), 3);
}

#[tokio::test]
async fn test_list_projects_active_only() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });
    service.deprioritize_project(request).await.unwrap();

    let request = Request::new(ListProjectsRequest {
        priority_filter: None,
        active_only: true,
    });

    let response = service.list_projects(request).await.unwrap();
    let response = response.into_inner();

    assert_eq!(response.total_count, 0);
}

#[tokio::test]
async fn test_heartbeat() {
    let (pool, _temp_dir) = setup_test_db().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(HeartbeatRequest {
        project_id: "abcd12345678".to_string(),
    });

    let response = service.heartbeat(request).await.unwrap();
    let response = response.into_inner();

    assert!(response.acknowledged);
    assert!(response.next_heartbeat_by.is_some());
}

#[tokio::test]
async fn test_invalid_project_id_format() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "short".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_empty_project_id_generates_local_id() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let response = service.register_project(request).await.unwrap();
    let response = response.into_inner();

    assert!(!response.created);
    assert!(response.project_id.starts_with("local_"));
    assert_eq!(response.priority, "none");
    assert!(!response.is_active);
    assert!(!response.newly_registered);
}

#[tokio::test]
async fn test_empty_path_returns_error() {
    let (pool, _temp_dir) = setup_test_db().await;
    let service = ProjectServiceImpl::new(pool);

    let request = Request::new(RegisterProjectRequest {
        path: "".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });

    let result = service.register_project(request).await;
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
}

#[tokio::test]
async fn test_queue_depth_returns_zero_for_empty_queue() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    let depth = super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab")
        .await
        .unwrap();
    assert_eq!(depth, 0);
}

#[tokio::test]
async fn test_queue_depth_counts_pending_items() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;
    let now = chrono::Utc::now().to_rfc3339();

    sqlx::query(
        r#"
        INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at)
        VALUES ('q1', 'key1', 'file', 'add', 'test123456ab', 'test-code', 'pending', ?1, ?1),
               ('q2', 'key2', 'file', 'add', 'test123456ab', 'test-code', 'pending', ?1, ?1),
               ('q3', 'key3', 'file', 'add', 'other1234567', 'test-code', 'pending', ?1, ?1),
               ('q4', 'key4', 'file', 'add', 'test123456ab', 'test-code', 'done', ?1, ?1)
    "#,
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    let depth = super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab")
        .await
        .unwrap();
    assert_eq!(depth, 2);
}

#[tokio::test]
async fn test_deferred_shutdown_scheduled_when_delay_set() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = build_test_service(pool, 30);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });
    service.deprioritize_project(request).await.unwrap();

    let pending = service.get_pending_shutdowns().await;
    assert!(pending.contains_key("abcd12345678"));
}

#[tokio::test]
async fn test_reactivation_cancels_deferred_shutdown() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

    let service = build_test_service(pool, 60);

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    let request = Request::new(DeprioritizeProjectRequest {
        project_id: "abcd12345678".to_string(),
    });
    service.deprioritize_project(request).await.unwrap();

    assert!(service
        .get_pending_shutdowns()
        .await
        .contains_key("abcd12345678"));

    let request = Request::new(RegisterProjectRequest {
        path: "/test/project".to_string(),
        project_id: "abcd12345678".to_string(),
        name: None,
        git_remote: None,
        register_if_new: false,
        priority: Some("high".to_string()),
    });
    service.register_project(request).await.unwrap();

    assert!(!service
        .get_pending_shutdowns()
        .await
        .contains_key("abcd12345678"));
}

#[tokio::test]
async fn test_queue_depth_handles_missing_table() {
    let (pool, _temp_dir) = setup_test_db().await;

    let result = super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab").await;
    match result {
        Ok(depth) => assert_eq!(depth, 0),
        Err(status) => {
            assert!(
                status.message().contains("not initialized") || status.code() == tonic::Code::Ok
            );
        }
    }
}

#[tokio::test]
async fn test_execute_deferred_shutdown_checks_queue() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;
    let now = chrono::Utc::now().to_rfc3339();

    let service = build_test_service(pool.clone(), 0);

    sqlx::query(
        r#"
        INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at)
        VALUES ('q1', 'key1', 'file', 'add', 'abcd12345678', 'test-code', 'pending', ?1, ?1)
    "#,
    )
    .bind(&now)
    .execute(&pool)
    .await
    .unwrap();

    super::lsp_lifecycle::schedule_deferred_shutdown(
        &service.pending_shutdowns,
        service.deactivation_delay_secs,
        "abcd12345678",
        true,
    )
    .await;

    let result = service
        .execute_deferred_shutdown("abcd12345678")
        .await
        .unwrap();
    assert!(!result);

    assert!(service
        .get_pending_shutdowns()
        .await
        .contains_key("abcd12345678"));
}

#[tokio::test]
async fn test_background_monitor_can_be_started() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    let service = build_test_service(pool, 0);

    service.start_deferred_shutdown_monitor();

    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn test_execute_deferred_shutdown_succeeds_when_queue_empty() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    let service = build_test_service(pool, 0);

    {
        let mut shutdowns = service.pending_shutdowns.write().await;
        shutdowns.insert(
            "abcd12345678".to_string(),
            (Instant::now() - Duration::from_secs(1), true),
        );
    }

    let result = service
        .execute_deferred_shutdown("abcd12345678")
        .await
        .unwrap();
    assert!(result);

    assert!(!service
        .get_pending_shutdowns()
        .await
        .contains_key("abcd12345678"));
}
