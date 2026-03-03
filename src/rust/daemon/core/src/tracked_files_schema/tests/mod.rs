mod batch_tests;
mod crud_tests;
mod transaction_tests;
mod unit_tests;

use super::*;
use sqlx::sqlite::SqlitePoolOptions;
use sqlx::SqlitePool;
use std::time::Duration;

async fn create_test_pool() -> SqlitePool {
    SqlitePoolOptions::new()
        .max_connections(1)
        .acquire_timeout(Duration::from_secs(5))
        .connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool")
}

async fn setup_tables(pool: &SqlitePool) {
    // Enable foreign keys
    sqlx::query("PRAGMA foreign_keys = ON")
        .execute(pool)
        .await
        .unwrap();

    // Create watch_folders (needed for FK)
    sqlx::query(crate::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
        .execute(pool)
        .await
        .unwrap();

    // Create tracked_files
    sqlx::query(CREATE_TRACKED_FILES_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_TRACKED_FILES_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }

    // Create qdrant_chunks
    sqlx::query(CREATE_QDRANT_CHUNKS_SQL)
        .execute(pool)
        .await
        .unwrap();
    for idx in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
        sqlx::query(idx).execute(pool).await.unwrap();
    }

    // Insert a test watch_folder
    sqlx::query(
        "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/home/user/project', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')",
    )
    .execute(pool)
    .await
    .unwrap();
}
