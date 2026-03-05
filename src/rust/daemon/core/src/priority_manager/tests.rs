use super::*;
use crate::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL;
use crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL;
use chrono::{Duration as ChronoDuration, Utc};
use sqlx::SqlitePool;
use std::time::Duration;
use tempfile::tempdir;

/// Helper to create test database with spec-compliant schema
async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_priority.db");
    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

    let pool = SqlitePool::connect(&db_url).await.unwrap();

    // Initialize spec-compliant schema
    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .unwrap();

    sqlx::query(CREATE_WATCH_FOLDERS_SQL)
        .execute(&pool)
        .await
        .unwrap();

    (pool, temp_dir)
}

/// Helper to create a test watch folder (project)
async fn create_test_project(pool: &SqlitePool, tenant_id: &str, path: &str) {
    let watch_id = format!("watch_{}", tenant_id);
    sqlx::query(
        r#"
        INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
        VALUES (?1, ?2, 'projects', ?3, 0, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        "#,
    )
    .bind(&watch_id)
    .bind(path)
    .bind(tenant_id)
    .execute(pool)
    .await
    .unwrap();
}

#[tokio::test]
async fn test_empty_parameters_error() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool);

    // Empty tenant_id
    let result = priority_manager.register_session("", "main").await;
    assert!(matches!(result, Err(PriorityError::EmptyParameter)));
}

// =========================================================================
// Session Tracking Tests (using watch_folders)
// =========================================================================

#[tokio::test]
async fn test_register_session_activates_project() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project
    create_test_project(&pool, "abcd12345678", "/test/project").await;

    // Register session
    let count = priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();
    assert_eq!(count, 1); // Returns 1 to indicate active

    // Verify session info
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(info.is_active);
    assert_eq!(info.priority, "high");
}

#[tokio::test]
async fn test_unregister_session_deactivates_project() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project and register session
    create_test_project(&pool, "abcd12345678", "/test/project").await;
    priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();

    // Unregister session
    let count = priority_manager
        .unregister_session("abcd12345678", "main")
        .await
        .unwrap();
    assert_eq!(count, 0); // Returns 0 to indicate inactive

    // Verify demoted to normal priority
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(!info.is_active);
    assert_eq!(info.priority, "normal");
}

#[tokio::test]
async fn test_heartbeat_updates_timestamp() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project and register session
    create_test_project(&pool, "abcd12345678", "/test/project").await;
    priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();

    // Get initial timestamp
    let info_before = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();

    // Wait briefly and send heartbeat
    tokio::time::sleep(Duration::from_millis(10)).await;
    let updated = priority_manager.heartbeat("abcd12345678").await.unwrap();
    assert!(updated);

    // Verify timestamp updated
    let info_after = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(info_after.last_activity_at >= info_before.last_activity_at);
}

#[tokio::test]
async fn test_heartbeat_ignored_without_active_session() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project without active sessions
    create_test_project(&pool, "abcd12345678", "/test/project").await;

    // Heartbeat should be ignored
    let updated = priority_manager.heartbeat("abcd12345678").await.unwrap();
    assert!(!updated);
}

#[tokio::test]
async fn test_cleanup_orphaned_sessions() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project
    create_test_project(&pool, "abcd12345678", "/test/project").await;

    // Register session
    priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();

    // Manually set last_activity_at to old timestamp to simulate orphaned session
    let old_time = Utc::now() - ChronoDuration::minutes(5);
    sqlx::query("UPDATE watch_folders SET last_activity_at = ?1 WHERE tenant_id = ?2")
        .bind(old_time.to_rfc3339())
        .bind("abcd12345678")
        .execute(&pool)
        .await
        .unwrap();

    // Cleanup with 60 second timeout - should detect orphaned session
    let cleanup = priority_manager
        .cleanup_orphaned_sessions(60)
        .await
        .unwrap();

    assert_eq!(cleanup.projects_affected, 1);
    assert_eq!(cleanup.sessions_cleaned, 1);
    assert!(cleanup
        .demoted_projects
        .contains(&"abcd12345678".to_string()));

    // Verify session cleaned up
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(!info.is_active);
    assert_eq!(info.priority, "normal");
}

#[tokio::test]
async fn test_no_orphaned_sessions_with_recent_heartbeat() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project and register session (sets last_activity_at to now)
    create_test_project(&pool, "abcd12345678", "/test/project").await;
    priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();

    // Cleanup with 60 second timeout - should NOT detect orphaned session
    let cleanup = priority_manager
        .cleanup_orphaned_sessions(60)
        .await
        .unwrap();

    assert_eq!(cleanup.projects_affected, 0);
    assert_eq!(cleanup.sessions_cleaned, 0);

    // Verify session still active
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(info.is_active);
    assert_eq!(info.priority, "high");
}

#[tokio::test]
async fn test_get_high_priority_projects() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create multiple test projects
    create_test_project(&pool, "project1aaaa", "/test/project1").await;
    create_test_project(&pool, "project2bbbb", "/test/project2").await;
    create_test_project(&pool, "project3cccc", "/test/project3").await;

    // Register sessions for some projects
    priority_manager
        .register_session("project1aaaa", "main")
        .await
        .unwrap();
    priority_manager
        .register_session("project2bbbb", "main")
        .await
        .unwrap();

    // Get high priority projects
    let high_priority = priority_manager.get_high_priority_projects().await.unwrap();

    assert_eq!(high_priority.len(), 2);
    let tenant_ids: Vec<_> = high_priority.iter().map(|p| &p.tenant_id).collect();
    assert!(tenant_ids.contains(&&"project1aaaa".to_string()));
    assert!(tenant_ids.contains(&&"project2bbbb".to_string()));
    assert!(!tenant_ids.contains(&&"project3cccc".to_string()));
}

#[tokio::test]
async fn test_register_nonexistent_project_fails() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool);

    // Try to register session for non-existent project
    let result = priority_manager
        .register_session("nonexistent12", "main")
        .await;

    assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
}

#[tokio::test]
async fn test_unregister_nonexistent_project_fails() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool);

    // Try to unregister session for non-existent project
    let result = priority_manager
        .unregister_session("nonexistent12", "main")
        .await;

    assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
}

#[tokio::test]
async fn test_priority_constants() {
    assert_eq!(priority::HIGH, 1);
    assert_eq!(priority::NORMAL, 3);
    assert_eq!(priority::LOW, 5);
}

#[tokio::test]
async fn test_set_priority_normal_to_high() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project (starts inactive/normal)
    create_test_project(&pool, "abcd12345678", "/test/project").await;

    // Set priority to high
    let (previous, queue_updated) = priority_manager
        .set_priority("abcd12345678", "high")
        .await
        .unwrap();

    assert_eq!(previous, "normal");
    // Queue items are not updated — ordering is computed at dequeue time
    assert_eq!(queue_updated, 0);

    // Verify project is now active
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(info.is_active);
    assert_eq!(info.priority, "high");
}

#[tokio::test]
async fn test_set_priority_high_to_normal() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool.clone());

    // Create test project and activate it
    create_test_project(&pool, "abcd12345678", "/test/project").await;
    priority_manager
        .register_session("abcd12345678", "main")
        .await
        .unwrap();

    // Set priority to normal
    let (previous, queue_updated) = priority_manager
        .set_priority("abcd12345678", "normal")
        .await
        .unwrap();

    assert_eq!(previous, "high");
    // Queue items are not updated — ordering is computed at dequeue time
    assert_eq!(queue_updated, 0);

    // Verify project is now inactive
    let info = priority_manager
        .get_session_info("abcd12345678")
        .await
        .unwrap()
        .unwrap();
    assert!(!info.is_active);
    assert_eq!(info.priority, "normal");
}

#[tokio::test]
async fn test_set_priority_nonexistent_project() {
    let (pool, _temp_dir) = setup_test_db().await;
    let priority_manager = PriorityManager::new(pool);

    let result = priority_manager.set_priority("nonexistent12", "high").await;
    assert!(matches!(result, Err(PriorityError::ProjectNotFound(_))));
}
