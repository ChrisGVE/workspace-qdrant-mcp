//! Shared test helpers for WriteActor tests.

use sqlx::SqlitePool;

use crate::write_actor::actor::{WriteActor, WriteActorHandle};

/// Create the `unified_queue` table.
async fn setup_queue_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE unified_queue (
            queue_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            branch TEXT DEFAULT 'main',
            payload_json TEXT NOT NULL DEFAULT '{}',
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            last_error_at TEXT,
            lease_until TEXT,
            worker_id TEXT,
            qdrant_status TEXT,
            search_status TEXT,
            decision_json TEXT,
            file_path TEXT
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `watch_folders` table.
async fn setup_watch_folders_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            collection TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            is_active INTEGER NOT NULL DEFAULT 0,
            is_paused INTEGER NOT NULL DEFAULT 0,
            is_archived INTEGER DEFAULT 0,
            pause_start_time TEXT,
            library_mode TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            cleanup_on_disable INTEGER DEFAULT 0,
            parent_watch_id TEXT,
            remote_hash TEXT,
            git_remote_url TEXT,
            last_activity_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `tracked_files` table.
///
/// Mirrors the live schema's column set for the fields write_actor touches:
/// no `tenant_id`, `file_path`, or `branch` columns exist on the real table —
/// tenancy is reached through `watch_folders` and paths are stored
/// watch-root-relative (post-v37).
async fn setup_tracked_files_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            relative_path TEXT,
            incremental INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `project_components` table.
async fn setup_project_components_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE project_components (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            name TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `search_events` table.
async fn setup_search_events_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE search_events (
            id TEXT PRIMARY KEY,
            ts TEXT,
            session_id TEXT,
            project_id TEXT,
            actor TEXT,
            tool TEXT,
            op TEXT,
            query_text TEXT,
            filters TEXT,
            top_k INTEGER,
            result_count INTEGER,
            latency_ms INTEGER,
            top_result_refs TEXT,
            outcome TEXT,
            parent_event_id TEXT,
            created_at TEXT
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `rules_mirror` table.
async fn setup_rules_mirror_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE rules_mirror (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            scope TEXT,
            tenant_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `corpus_statistics` table.
async fn setup_corpus_statistics_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE corpus_statistics (
            collection TEXT PRIMARY KEY,
            last_corrected_n INTEGER DEFAULT 0
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create the `watch_folder_submodules` table.
async fn setup_watch_folder_submodules_table(pool: &SqlitePool) {
    sqlx::query(
        "CREATE TABLE watch_folder_submodules (
            parent_watch_id TEXT NOT NULL,
            child_watch_id TEXT NOT NULL,
            PRIMARY KEY (parent_watch_id, child_watch_id)
        )",
    )
    .execute(pool)
    .await
    .unwrap();
}

/// Create an in-memory SQLite pool with the minimal schema needed by the
/// WriteActor exec methods, then spawn the actor and return both.
pub async fn setup_test_db() -> (SqlitePool, WriteActorHandle) {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory SQLite pool");

    setup_queue_table(&pool).await;
    setup_watch_folders_table(&pool).await;
    setup_tracked_files_table(&pool).await;
    setup_project_components_table(&pool).await;
    setup_search_events_table(&pool).await;
    setup_rules_mirror_table(&pool).await;
    setup_corpus_statistics_table(&pool).await;
    setup_watch_folder_submodules_table(&pool).await;

    let handle = WriteActor::spawn(pool.clone());
    (pool, handle)
}
