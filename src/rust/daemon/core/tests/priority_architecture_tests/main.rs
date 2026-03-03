//! Integration tests for computed priority dequeue and enqueue-first architecture.
//!
//! Tests:
//! - Computed priority: active projects > memory > inactive > libraries
//! - Op-based priority: delete > reset > scan > update > add
//! - Fairness scheduler alternation (DESC/ASC)
//! - Progressive scanning queue growth
//! - Submodule detection at folder boundaries

use workspace_qdrant_core::{
    QueueManager,
    unified_queue_schema::{
        ItemType, QueueOperation,
        CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL,
    },
    fairness_scheduler::{FairnessScheduler, FairnessSchedulerConfig},
};
use sqlx::SqlitePool;
use tempfile::TempDir;

mod priority_tests;
mod scanning_tests;

/// Helper to create in-memory SQLite database with required tables
pub async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
        .execute(&pool)
        .await
        .expect("Failed to create unified_queue table");

    for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
        sqlx::query(index_sql).execute(&pool).await
            .expect("Failed to create unified_queue index");
    }

    sqlx::query(
        r#"
        CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            is_active INTEGER DEFAULT 0,
            git_remote_url TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            last_scan TEXT,
            last_activity_at TEXT,
            enabled INTEGER DEFAULT 1,
            cleanup_on_disable INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(path, collection)
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    sqlx::query(
        r#"
        CREATE TABLE tracked_files (
            file_id TEXT PRIMARY KEY,
            watch_folder_id TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            file_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_hash TEXT,
            file_size INTEGER,
            processing_status TEXT DEFAULT 'pending',
            last_processed_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        "#
    )
    .execute(&pool)
    .await
    .expect("Failed to create tracked_files table");

    pool
}

/// Insert a watch folder with given activity status
pub async fn insert_watch_folder(pool: &SqlitePool, watch_id: &str, path: &str, tenant_id: &str, collection: &str, is_active: bool) {
    let now = wqm_common::timestamps::now_utc();
    sqlx::query(
        r#"INSERT INTO watch_folders (watch_id, path, collection, tenant_id, is_active, created_at, updated_at)
           VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)"#
    )
    .bind(watch_id)
    .bind(path)
    .bind(collection)
    .bind(tenant_id)
    .bind(is_active as i32)
    .bind(&now)
    .execute(pool)
    .await
    .unwrap();
}
