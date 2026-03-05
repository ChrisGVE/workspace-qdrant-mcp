//! Deferred shutdown, queue depth, and background monitor tests

use std::time::{Duration, Instant};

use tonic::Request;

use crate::proto::project_service_server::ProjectService;
use crate::proto::{DeprioritizeProjectRequest, RegisterProjectRequest};

use super::{
    build_test_service, create_test_watch_folder, setup_test_db, setup_test_db_with_queue,
};

#[tokio::test]
async fn test_queue_depth_returns_zero_for_empty_queue() {
    let (pool, _temp_dir) = setup_test_db_with_queue().await;

    let depth = super::super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab")
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

    let depth = super::super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab")
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

    let result = super::super::lsp_lifecycle::get_project_queue_depth(&pool, "test123456ab").await;
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

    super::super::lsp_lifecycle::schedule_deferred_shutdown(
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
