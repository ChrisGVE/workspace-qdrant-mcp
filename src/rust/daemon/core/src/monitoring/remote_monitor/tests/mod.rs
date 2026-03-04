use super::*;
use std::path::Path;

use git2::Repository;
use sqlx::SqlitePool;

mod remote_url;
mod git_state;

/// Helper to create in-memory SQLite database with watch_folders schema
pub(super) async fn create_test_database() -> SqlitePool {
    let pool = SqlitePool::connect("sqlite::memory:")
        .await
        .expect("Failed to create in-memory database");

    sqlx::query(
        r#"
        CREATE TABLE watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            is_active INTEGER DEFAULT 0 CHECK (is_active IN (0, 1)),
            is_archived INTEGER DEFAULT 0 CHECK (is_archived IN (0, 1)),
            is_git_tracked INTEGER DEFAULT 0 CHECK (is_git_tracked IN (0, 1)),
            git_remote_url TEXT,
            remote_hash TEXT,
            disambiguation_path TEXT,
            patterns TEXT NOT NULL DEFAULT '[]',
            ignore_patterns TEXT NOT NULL DEFAULT '[]',
            auto_ingest BOOLEAN NOT NULL DEFAULT 1,
            recursive BOOLEAN NOT NULL DEFAULT 1,
            recursive_depth INTEGER NOT NULL DEFAULT 10,
            debounce_seconds REAL NOT NULL DEFAULT 2.0,
            enabled BOOLEAN NOT NULL DEFAULT 1,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TEXT,
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folders table");

    // Task 14: Junction table for submodule relationships
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS watch_folder_submodules (
            parent_watch_id TEXT NOT NULL,
            child_watch_id TEXT NOT NULL,
            submodule_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (parent_watch_id, child_watch_id),
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE,
            FOREIGN KEY (child_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create watch_folder_submodules table");

    sqlx::query(
        r#"
        CREATE TABLE unified_queue (
            queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
            lease_until TEXT,
            worker_id TEXT,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL DEFAULT '{}',
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            last_error_at TEXT,
            branch TEXT DEFAULT 'main',
            metadata TEXT DEFAULT '{}',
            file_path TEXT UNIQUE
        )
        "#,
    )
    .execute(&pool)
    .await
    .expect("Failed to create unified_queue table");

    pool
}

/// Create a real git repo with a remote set
pub(super) fn create_git_repo_with_remote(dir: &Path, remote_url: &str) {
    let repo = Repository::init(dir).expect("Failed to init git repo");
    repo.remote("origin", remote_url)
        .expect("Failed to set remote");
}
