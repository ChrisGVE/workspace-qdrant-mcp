//! Tests for ProjectService gRPC implementation
//!
//! Split into focused modules:
//! - `registration_tests`: project registration, validation, and ID handling
//! - `query_tests`: project status queries, listing, and heartbeat
//! - `lifecycle_tests`: deferred shutdown, queue depth, and background monitor
//! - `path_validation_tests`: negative tests for path validation at handler entry

mod lifecycle_tests;
mod path_validation_tests;
mod query_tests;
mod recover_tests;
mod registration_tests;

use std::collections::HashMap;
use std::sync::Arc;

use sqlx::SqlitePool;
use tokio::sync::RwLock;

use workspace_qdrant_core::{DaemonStateManager, PriorityManager, ProjectLanguageDetector};

use super::ProjectServiceImpl;

/// Helper to create test database with schema
async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
    let temp_dir = tempfile::tempdir().unwrap();
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

/// Helper to create a watch_folder entry with an explicit git remote URL.
///
/// Used by the reconciliation tests (#138/#139): a project with a remote is
/// keyed by `<hash(remote)>`, so the stored `git_remote_url` must be present
/// to exercise the remote -> local tenancy flip and the moved-path case.
async fn create_test_watch_folder_with_remote(
    pool: &SqlitePool,
    project_id: &str,
    path: &str,
    git_remote_url: &str,
) {
    let now = chrono::Utc::now().to_rfc3339();
    let watch_id = format!("test-{project_id}");
    sqlx::query(
        r#"
        INSERT INTO watch_folders (
            watch_id, path, collection, tenant_id, git_remote_url, is_git_tracked,
            is_active, follow_symlinks, enabled, cleanup_on_disable, created_at, updated_at
        ) VALUES (?1, ?2, 'projects', ?3, ?4, 1, 0, 0, 1, 0, ?5, ?5)
    "#,
    )
    .bind(&watch_id)
    .bind(path)
    .bind(project_id)
    .bind(git_remote_url)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}

/// Fetch the stored (tenant_id, path) for the single project watch_folder row
/// matching `path` OR `tenant_id`. Returns all matching rows so tests can
/// assert there is no duplicate after a reconcile.
async fn fetch_project_rows(pool: &SqlitePool) -> Vec<(String, String)> {
    sqlx::query_as::<_, (String, String)>(
        "SELECT tenant_id, path FROM watch_folders WHERE collection = 'projects' ORDER BY path",
    )
    .fetch_all(pool)
    .await
    .unwrap()
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
