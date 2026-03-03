//! Database schema creation for state recovery.
//!
//! Creates the SQLite tables and indexes needed to reconstruct state.db
//! from Qdrant collection data. The daemon handles any remaining migrations
//! on its next startup.

use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::params;

/// Create core tables: watch_folders, tracked_files, qdrant_chunks.
pub(super) fn create_core_tables(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS watch_folders (
            watch_id TEXT PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            collection TEXT NOT NULL CHECK (collection IN ('projects', 'libraries')),
            tenant_id TEXT NOT NULL,
            parent_watch_id TEXT,
            submodule_path TEXT,
            git_remote_url TEXT,
            remote_hash TEXT,
            disambiguation_path TEXT,
            is_active INTEGER DEFAULT 0,
            last_activity_at TEXT,
            is_paused INTEGER DEFAULT 0,
            pause_start_time TEXT,
            is_archived INTEGER DEFAULT 0,
            last_commit_hash TEXT,
            is_git_tracked INTEGER DEFAULT 0,
            library_mode TEXT,
            follow_symlinks INTEGER DEFAULT 0,
            enabled INTEGER DEFAULT 1,
            cleanup_on_disable INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_scan TEXT,
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )",
    )
    .context("Failed to create watch_folders")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS tracked_files (
            file_id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_folder_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            branch TEXT,
            file_type TEXT,
            language TEXT,
            file_mtime TEXT NOT NULL,
            file_hash TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0,
            chunking_method TEXT,
            lsp_status TEXT DEFAULT 'none',
            treesitter_status TEXT DEFAULT 'none',
            last_error TEXT,
            needs_reconcile INTEGER DEFAULT 0,
            reconcile_reason TEXT,
            extension TEXT,
            is_test INTEGER DEFAULT 0,
            collection TEXT NOT NULL DEFAULT 'projects',
            base_point TEXT,
            relative_path TEXT,
            incremental INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (watch_folder_id) REFERENCES watch_folders(watch_id),
            UNIQUE(watch_folder_id, file_path, branch)
        )",
    )
    .context("Failed to create tracked_files")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS qdrant_chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL,
            point_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            chunk_type TEXT,
            symbol_name TEXT,
            start_line INTEGER,
            end_line INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (file_id) REFERENCES tracked_files(file_id) ON DELETE CASCADE,
            UNIQUE(file_id, chunk_index)
        )",
    )
    .context("Failed to create qdrant_chunks")
}

/// Create auxiliary tables: rules_mirror, unified_queue, submodules, operational_state.
pub(super) fn create_auxiliary_tables(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS rules_mirror (
            rule_id TEXT PRIMARY KEY,
            rule_text TEXT NOT NULL,
            scope TEXT,
            tenant_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .context("Failed to create rules_mirror")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS unified_queue (
            queue_id TEXT PRIMARY KEY,
            idempotency_key TEXT UNIQUE NOT NULL,
            item_type TEXT NOT NULL,
            op TEXT NOT NULL,
            tenant_id TEXT NOT NULL,
            collection TEXT NOT NULL,
            priority INTEGER DEFAULT 5,
            status TEXT DEFAULT 'pending',
            branch TEXT,
            payload_json TEXT,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0,
            last_error TEXT,
            leased_by TEXT,
            lease_expires_at TEXT
        )",
    )
    .context("Failed to create unified_queue")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS watch_folder_submodules (
            parent_watch_id TEXT NOT NULL,
            child_watch_id TEXT NOT NULL,
            submodule_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (parent_watch_id, child_watch_id),
            FOREIGN KEY (parent_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE,
            FOREIGN KEY (child_watch_id) REFERENCES watch_folders(watch_id) ON DELETE CASCADE
        )",
    )
    .context("Failed to create watch_folder_submodules")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS operational_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
    )
    .context("Failed to create operational_state")
}

/// Create indexes for recovery tables.
pub(super) fn create_recovery_indexes(conn: &rusqlite::Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE INDEX IF NOT EXISTS idx_watch_collection_tenant ON watch_folders(collection, tenant_id);
         CREATE INDEX IF NOT EXISTS idx_watch_path ON watch_folders(path);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_watch ON tracked_files(watch_folder_id);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_path ON tracked_files(file_path);
         CREATE INDEX IF NOT EXISTS idx_tracked_files_base_point ON tracked_files(base_point);
         CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_point ON qdrant_chunks(point_id);
         CREATE INDEX IF NOT EXISTS idx_qdrant_chunks_file ON qdrant_chunks(file_id);",
    )
    .context("Failed to create indexes")
}

/// Create a fresh SQLite database with all required tables.
///
/// Creates only the tables needed for recovery. The daemon will handle any
/// remaining migrations on its next startup.
pub(super) fn create_fresh_database(db_path: &Path) -> Result<rusqlite::Connection> {
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent).context("Failed to create database directory")?;
    }

    let conn = rusqlite::Connection::open(db_path).context("Failed to create state database")?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA synchronous=NORMAL;
         PRAGMA foreign_keys=ON;",
    )
    .context("Failed to set SQLite pragmas")?;

    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL
        )",
    )
    .context("Failed to create schema_version")?;

    let now = wqm_common::timestamps::now_utc();
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?1, ?2)",
        params![21, now],
    )
    .context("Failed to insert schema version")?;

    create_core_tables(&conn)?;
    create_auxiliary_tables(&conn)?;

    conn.execute(
        "INSERT INTO operational_state (key, value, updated_at) VALUES ('recovery_done', 'true', ?1)",
        params![now],
    )
    .context("Failed to set recovery flag")?;

    create_recovery_indexes(&conn)?;

    Ok(conn)
}
