//! Schema Version Table and Migration System
//!
//! This module implements the schema version tracking and migration system.
//! Per WORKSPACE_QDRANT_MCP.md spec v1.6.3, the daemon is the sole owner of
//! the SQLite database schema. It creates all tables on startup and handles
//! all schema version upgrades.
//!
//! Other components (MCP Server, CLI) must NOT create tables or run migrations.

use sqlx::{SqlitePool, Row, sqlite::SqliteRow};
use thiserror::Error;
use tracing::{debug, info};

/// Current schema version - increment when adding new migrations
pub const CURRENT_SCHEMA_VERSION: i32 = 21;

/// Errors that can occur during schema operations
#[derive(Error, Debug)]
pub enum SchemaError {
    #[error("SQLite error: {0}")]
    SqlxError(#[from] sqlx::Error),

    #[error("Migration error: {0}")]
    MigrationError(String),

    #[error("Schema version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: i32, found: i32 },

    #[error("Downgrade not supported: database version {db_version} > code version {code_version}")]
    DowngradeNotSupported { db_version: i32, code_version: i32 },
}

/// SQL to create the schema_version table
pub const CREATE_SCHEMA_VERSION_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
"#;

/// Schema version entry
#[derive(Debug, Clone)]
pub struct SchemaVersionEntry {
    pub version: i32,
    pub applied_at: String,
}

/// Schema manager handles version tracking and migrations
pub struct SchemaManager {
    pool: SqlitePool,
}

impl SchemaManager {
    /// Create a new schema manager
    pub fn new(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Initialize the schema version table
    pub async fn initialize(&self) -> Result<(), SchemaError> {
        debug!("Initializing schema version table");
        sqlx::query(CREATE_SCHEMA_VERSION_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Get the current schema version from the database
    /// Returns None if no version is set (fresh database)
    pub async fn get_current_version(&self) -> Result<Option<i32>, SchemaError> {
        // Check if table exists first
        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')"
        )
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            return Ok(None);
        }

        let result: Option<i32> = sqlx::query_scalar("SELECT MAX(version) FROM schema_version")
            .fetch_optional(&self.pool)
            .await?
            .flatten();

        Ok(result)
    }

    /// Record a migration as applied
    pub async fn record_migration(&self, version: i32) -> Result<(), SchemaError> {
        sqlx::query("INSERT INTO schema_version (version) VALUES (?1)")
            .bind(version)
            .execute(&self.pool)
            .await?;
        info!("Recorded migration to schema version {}", version);
        Ok(())
    }

    /// Get all applied migrations
    pub async fn get_migration_history(&self) -> Result<Vec<SchemaVersionEntry>, SchemaError> {
        let rows: Vec<SqliteRow> = sqlx::query(
            "SELECT version, applied_at FROM schema_version ORDER BY version ASC"
        )
        .fetch_all(&self.pool)
        .await?;

        let mut result = Vec::new();
        for row in rows {
            result.push(SchemaVersionEntry {
                version: row.get("version"),
                applied_at: row.get("applied_at"),
            });
        }

        Ok(result)
    }

    /// Check if migrations are needed and run them
    pub async fn run_migrations(&self) -> Result<(), SchemaError> {
        // Ensure schema_version table exists
        self.initialize().await?;

        let current = self.get_current_version().await?.unwrap_or(0);
        info!("Current schema version: {}, target: {}", current, CURRENT_SCHEMA_VERSION);

        if current > CURRENT_SCHEMA_VERSION {
            return Err(SchemaError::DowngradeNotSupported {
                db_version: current,
                code_version: CURRENT_SCHEMA_VERSION,
            });
        }

        if current == CURRENT_SCHEMA_VERSION {
            debug!("Schema is up to date");
            return Ok(());
        }

        // Run migrations in order
        for version in (current + 1)..=CURRENT_SCHEMA_VERSION {
            info!("Running migration to version {}", version);
            self.run_migration(version).await?;
            self.record_migration(version).await?;
        }

        info!("Schema migrations complete. Now at version {}", CURRENT_SCHEMA_VERSION);
        Ok(())
    }

    /// Run a specific migration
    async fn run_migration(&self, version: i32) -> Result<(), SchemaError> {
        match version {
            1 => self.migrate_v1().await,
            2 => self.migrate_v2().await,
            3 => self.migrate_v3().await,
            4 => self.migrate_v4().await,
            5 => self.migrate_v5().await,
            6 => self.migrate_v6().await,
            7 => self.migrate_v7().await,
            8 => self.migrate_v8().await,
            9 => self.migrate_v9().await,
            10 => self.migrate_v10().await,
            11 => self.migrate_v11().await,
            12 => self.migrate_v12().await,
            13 => self.migrate_v13().await,
            14 => self.migrate_v14().await,
            15 => self.migrate_v15().await,
            16 => self.migrate_v16().await,
            17 => self.migrate_v17().await,
            18 => self.migrate_v18().await,
            19 => self.migrate_v19().await,
            20 => self.migrate_v20().await,
            21 => self.migrate_v21().await,
            _ => Err(SchemaError::MigrationError(format!(
                "Unknown migration version: {}", version
            ))),
        }
    }

    /// Migration v1: Create initial spec-compliant tables
    async fn migrate_v1(&self) -> Result<(), SchemaError> {
        info!("Migration v1: Creating spec-compliant tables");

        // Import the schema SQL constants
        use super::watch_folders_schema::{CREATE_WATCH_FOLDERS_SQL, CREATE_WATCH_FOLDERS_INDEXES_SQL};
        use super::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL};

        // Create watch_folders table
        debug!("Creating watch_folders table");
        sqlx::query(CREATE_WATCH_FOLDERS_SQL)
            .execute(&self.pool)
            .await?;

        // Create watch_folders indexes
        for index_sql in CREATE_WATCH_FOLDERS_INDEXES_SQL {
            debug!("Creating watch_folders index");
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        // Create unified_queue table
        debug!("Creating unified_queue table");
        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&self.pool)
            .await?;

        // Create unified_queue indexes
        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            debug!("Creating unified_queue index");
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v1 complete");
        Ok(())
    }

    /// Migration v2: Create tracked_files and qdrant_chunks tables
    async fn migrate_v2(&self) -> Result<(), SchemaError> {
        info!("Migration v2: Creating tracked_files and qdrant_chunks tables");

        use super::tracked_files_schema::{
            CREATE_TRACKED_FILES_SQL, CREATE_TRACKED_FILES_INDEXES_SQL,
            CREATE_QDRANT_CHUNKS_SQL, CREATE_QDRANT_CHUNKS_INDEXES_SQL,
        };

        // Enable foreign keys (required for CASCADE)
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&self.pool)
            .await?;

        // Create tracked_files table
        debug!("Creating tracked_files table");
        sqlx::query(CREATE_TRACKED_FILES_SQL)
            .execute(&self.pool)
            .await?;

        // Create tracked_files indexes
        for index_sql in CREATE_TRACKED_FILES_INDEXES_SQL {
            debug!("Creating tracked_files index");
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        // Create qdrant_chunks table
        debug!("Creating qdrant_chunks table");
        sqlx::query(CREATE_QDRANT_CHUNKS_SQL)
            .execute(&self.pool)
            .await?;

        // Create qdrant_chunks indexes
        for index_sql in CREATE_QDRANT_CHUNKS_INDEXES_SQL {
            debug!("Creating qdrant_chunks index");
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v2 complete");
        Ok(())
    }

    /// Migration v3: Add needs_reconcile columns to tracked_files for transactional integrity
    async fn migrate_v3(&self) -> Result<(), SchemaError> {
        info!("Migration v3: Adding needs_reconcile columns to tracked_files");

        use super::tracked_files_schema::{MIGRATE_V3_SQL, CREATE_RECONCILE_INDEX_SQL};

        // Check if columns already exist (handles CREATE TABLE that already includes them)
        let has_reconcile: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_reconcile {
            for alter_sql in MIGRATE_V3_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            debug!("needs_reconcile columns already exist, skipping ALTER TABLE");
        }

        // Index creation is idempotent (IF NOT EXISTS)
        debug!("Creating reconcile index");
        sqlx::query(CREATE_RECONCILE_INDEX_SQL)
            .execute(&self.pool)
            .await?;

        info!("Migration v3 complete");
        Ok(())
    }

    /// Migration v4: Add is_paused and pause_start_time columns to watch_folders
    async fn migrate_v4(&self) -> Result<(), SchemaError> {
        info!("Migration v4: Adding pause columns to watch_folders");

        use super::watch_folders_schema::MIGRATE_V4_PAUSE_SQL;

        // Check if columns already exist (handles CREATE TABLE that already includes them)
        let has_is_paused: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_is_paused {
            for alter_sql in MIGRATE_V4_PAUSE_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            debug!("Pause columns already exist, skipping ALTER TABLE");
        }

        info!("Migration v4 complete");
        Ok(())
    }

    /// Migration v5: Create metrics_history table for time-series storage
    async fn migrate_v5(&self) -> Result<(), SchemaError> {
        info!("Migration v5: Creating metrics_history table");

        use super::metrics_history_schema::{CREATE_METRICS_HISTORY_SQL, CREATE_METRICS_HISTORY_INDEXES_SQL};

        // Create metrics_history table
        debug!("Creating metrics_history table");
        sqlx::query(CREATE_METRICS_HISTORY_SQL)
            .execute(&self.pool)
            .await?;

        // Create indexes
        for index_sql in CREATE_METRICS_HISTORY_INDEXES_SQL {
            debug!("Creating metrics_history index");
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v5 complete");
        Ok(())
    }

    /// Migration v6: Add collection column to tracked_files for format-based routing
    async fn migrate_v6(&self) -> Result<(), SchemaError> {
        info!("Migration v6: Adding collection column to tracked_files");

        use super::tracked_files_schema::MIGRATE_V6_SQL;

        // Check if column already exists (handles fresh installs that include it in CREATE TABLE)
        let has_collection: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_collection {
            debug!("Running ALTER TABLE: {}", MIGRATE_V6_SQL);
            sqlx::query(MIGRATE_V6_SQL)
                .execute(&self.pool)
                .await?;
        } else {
            debug!("Collection column already exists, skipping ALTER TABLE");
        }

        info!("Migration v6 complete");
        Ok(())
    }

    /// Migration v7: Add is_archived column to watch_folders
    async fn migrate_v7(&self) -> Result<(), SchemaError> {
        info!("Migration v7: Adding is_archived column to watch_folders");

        use super::watch_folders_schema::MIGRATE_V7_ARCHIVE_SQL;

        // Check if column already exists (handles fresh installs that include it in CREATE TABLE)
        let has_archived: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_archived'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_archived {
            debug!("Running ALTER TABLE: {}", MIGRATE_V7_ARCHIVE_SQL);
            sqlx::query(MIGRATE_V7_ARCHIVE_SQL)
                .execute(&self.pool)
                .await?;
        } else {
            debug!("is_archived column already exists, skipping ALTER TABLE");
        }

        info!("Migration v7 complete");
        Ok(())
    }

    /// Migration v8: Add extension and is_test columns to tracked_files
    async fn migrate_v8(&self) -> Result<(), SchemaError> {
        info!("Migration v8: Adding extension and is_test columns to tracked_files");

        use super::tracked_files_schema::MIGRATE_V8_ADD_COLUMNS_SQL;
        use crate::file_classification::{get_extension_for_storage, is_test_file};
        use std::path::Path;

        // Check if columns already exist (handles fresh installs that include them in CREATE TABLE)
        let has_extension: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'extension'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_extension {
            for alter_sql in MIGRATE_V8_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            debug!("extension column already exists, skipping ALTER TABLE");
        }

        // Backfill extension and is_test for existing rows using Rust classification logic
        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT rowid, file_path FROM tracked_files WHERE extension IS NULL"
        )
        .fetch_all(&self.pool)
        .await?;

        if !rows.is_empty() {
            info!("Backfilling extension and is_test for {} existing rows", rows.len());
            for (rowid, file_path) in &rows {
                let path = Path::new(file_path);
                let extension = get_extension_for_storage(path);
                let is_test = is_test_file(path);
                sqlx::query(
                    "UPDATE tracked_files SET extension = ?1, is_test = ?2 WHERE rowid = ?3"
                )
                .bind(extension.as_deref())
                .bind(is_test as i32)
                .bind(rowid)
                .execute(&self.pool)
                .await?;
            }
            info!("Backfill complete");
        }

        info!("Migration v8 complete");
        Ok(())
    }

    /// Migration v9: Add 'url' to unified_queue item_type CHECK constraint
    ///
    /// SQLite does not support ALTER TABLE to modify CHECK constraints,
    /// so we recreate the table preserving all existing data.
    async fn migrate_v9(&self) -> Result<(), SchemaError> {
        info!("Migration v9: Adding 'url' item_type to unified_queue CHECK constraint");

        use super::unified_queue_schema::{CREATE_UNIFIED_QUEUE_SQL, CREATE_UNIFIED_QUEUE_INDEXES_SQL};

        // Recreate using canonical schema (which now includes 'url')
        sqlx::query("ALTER TABLE unified_queue RENAME TO unified_queue_old")
            .execute(&self.pool).await?;

        sqlx::query(CREATE_UNIFIED_QUEUE_SQL)
            .execute(&self.pool).await?;

        // Copy all existing data
        sqlx::query(
            "INSERT INTO unified_queue (
                queue_id, item_type, op, tenant_id, collection, priority,
                status, created_at, updated_at, lease_until, worker_id,
                idempotency_key, payload_json, retry_count, max_retries,
                error_message, last_error_at, branch, metadata, file_path
            )
            SELECT
                queue_id, item_type, op, tenant_id, collection, priority,
                status, created_at, updated_at, lease_until, worker_id,
                idempotency_key, payload_json, retry_count, max_retries,
                error_message, last_error_at, branch, metadata, file_path
            FROM unified_queue_old"
        )
        .execute(&self.pool).await?;

        sqlx::query("DROP TABLE unified_queue_old")
            .execute(&self.pool).await?;

        // Recreate indexes
        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool).await?;
        }

        info!("Migration v9 complete");
        Ok(())
    }

    /// Migration v10: Fix stale file_type='test' rows in tracked_files
    ///
    /// Before v8, some code paths wrote file_type='test' instead of using
    /// classify_file_type() + is_test. Reclassify these rows using the
    /// canonical file_classification logic.
    async fn migrate_v10(&self) -> Result<(), SchemaError> {
        info!("Migration v10: Fixing stale file_type='test' rows in tracked_files");

        use crate::file_classification::{classify_file_type, is_test_file};
        use std::path::Path;

        let rows: Vec<(i64, String)> = sqlx::query_as(
            "SELECT rowid, file_path FROM tracked_files WHERE file_type = 'test'"
        )
        .fetch_all(&self.pool)
        .await?;

        if rows.is_empty() {
            info!("No stale file_type='test' rows found");
        } else {
            info!("Reclassifying {} rows with file_type='test'", rows.len());
            for (rowid, file_path) in &rows {
                let path = Path::new(file_path);
                let file_type = classify_file_type(path);
                let is_test = is_test_file(path);
                sqlx::query(
                    "UPDATE tracked_files SET file_type = ?1, is_test = ?2 WHERE rowid = ?3"
                )
                .bind(file_type.as_str())
                .bind(is_test as i32)
                .bind(rowid)
                .execute(&self.pool)
                .await?;
            }
            info!("Reclassification complete");
        }

        info!("Migration v10 complete");
        Ok(())
    }

    /// Migration v11: Queue taxonomy overhaul
    ///
    /// Migrates item_type and op values in unified_queue to the new taxonomy:
    /// - ItemType: content→text, project/library/delete_tenant→tenant, delete_document→doc
    /// - QueueOperation: ingest→add
    /// - Rename items: item_type='rename' becomes item_type='file', op='rename'
    ///
    /// SQLite CHECK constraints cannot be altered, so we recreate the table.
    async fn migrate_v11(&self) -> Result<(), SchemaError> {
        info!("Migration v11: Queue taxonomy overhaul (item_type + op values)");

        // On a fresh database, v1 already creates unified_queue with the current
        // (v11) schema. Check if migration is needed by inspecting the table DDL.
        let table_sql: Option<String> = sqlx::query_scalar(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='unified_queue'"
        )
        .fetch_optional(&self.pool)
        .await?;

        let needs_migration = match &table_sql {
            Some(sql) => sql.contains("'content'") || sql.contains("'ingest'") || sql.contains("'project'"),
            None => false, // Table doesn't exist; v1 will create it
        };

        if !needs_migration {
            info!("Migration v11: Table already has correct schema, skipping");
            return Ok(());
        }

        // SQLite doesn't support ALTER TABLE to change CHECK constraints.
        // We must: create new table → copy data with transforms → drop old → rename.
        // Clean up any leftover temp table from a previous failed attempt.
        sqlx::query("DROP TABLE IF EXISTS unified_queue_v11")
            .execute(&self.pool).await?;

        sqlx::query(r#"
            CREATE TABLE unified_queue_v11 (
                queue_id TEXT PRIMARY KEY NOT NULL DEFAULT (lower(hex(randomblob(16)))),
                item_type TEXT NOT NULL CHECK (item_type IN (
                    'text', 'file', 'url', 'website', 'doc', 'folder', 'tenant', 'collection'
                )),
                op TEXT NOT NULL CHECK (op IN ('add', 'update', 'delete', 'scan', 'rename', 'uplift', 'reset')),
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0 CHECK (priority >= 0 AND priority <= 10),
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN (
                    'pending', 'in_progress', 'done', 'failed'
                )),
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                lease_until TEXT,
                worker_id TEXT,
                idempotency_key TEXT NOT NULL UNIQUE,
                payload_json TEXT NOT NULL DEFAULT '{}',
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                error_message TEXT,
                last_error_at TEXT,
                branch TEXT DEFAULT 'main',
                metadata TEXT DEFAULT '{}',
                file_path TEXT UNIQUE
            )
        "#).execute(&self.pool).await?;

        // Copy data with value transformations
        sqlx::query(r#"
            INSERT INTO unified_queue_v11 (
                queue_id, item_type, op, tenant_id, collection, priority, status,
                created_at, updated_at, lease_until, worker_id, idempotency_key,
                payload_json, retry_count, max_retries, error_message, last_error_at,
                branch, metadata, file_path
            )
            SELECT
                queue_id,
                CASE item_type
                    WHEN 'content' THEN 'text'
                    WHEN 'project' THEN 'tenant'
                    WHEN 'library' THEN 'tenant'
                    WHEN 'delete_tenant' THEN 'tenant'
                    WHEN 'delete_document' THEN 'doc'
                    WHEN 'rename' THEN 'file'
                    ELSE item_type
                END,
                CASE
                    WHEN item_type = 'rename' THEN 'rename'
                    WHEN op = 'ingest' THEN 'add'
                    ELSE op
                END,
                tenant_id, collection, priority, status,
                created_at, updated_at, lease_until, worker_id, idempotency_key,
                payload_json, retry_count, max_retries, error_message, last_error_at,
                branch, metadata, file_path
            FROM unified_queue
        "#).execute(&self.pool).await?;

        // Drop old table and rename
        sqlx::query("DROP TABLE unified_queue").execute(&self.pool).await?;
        sqlx::query("ALTER TABLE unified_queue_v11 RENAME TO unified_queue").execute(&self.pool).await?;

        // Recreate indexes (they were dropped with the old table)
        use super::unified_queue_schema::CREATE_UNIFIED_QUEUE_INDEXES_SQL;
        for index_sql in CREATE_UNIFIED_QUEUE_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }

        info!("Migration v11 complete");
        Ok(())
    }

    /// Migration v12: Create search_events table for pipeline instrumentation
    async fn migrate_v12(&self) -> Result<(), SchemaError> {
        info!("Migration v12: Creating search_events table");

        use super::search_events_schema::{CREATE_SEARCH_EVENTS_SQL, CREATE_SEARCH_EVENTS_INDEXES_SQL};

        sqlx::query(CREATE_SEARCH_EVENTS_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_SEARCH_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v12 complete");
        Ok(())
    }

    async fn migrate_v13(&self) -> Result<(), SchemaError> {
        info!("Migration v13: Creating resolution_events table");

        use super::resolution_events_schema::{CREATE_RESOLUTION_EVENTS_SQL, CREATE_RESOLUTION_EVENTS_INDEXES_SQL};

        sqlx::query(CREATE_RESOLUTION_EVENTS_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_RESOLUTION_EVENTS_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v13 complete");
        Ok(())
    }

    /// Migration v14: Create search_behavior view for bypass/success/fallback classification
    async fn migrate_v14(&self) -> Result<(), SchemaError> {
        info!("Migration v14: Creating search_behavior view");

        sqlx::query(
            r#"
            CREATE VIEW IF NOT EXISTS search_behavior AS
            WITH windowed_events AS (
                SELECT
                    session_id,
                    tool,
                    op,
                    ts,
                    LAG(tool) OVER (PARTITION BY session_id ORDER BY ts) AS prev_tool,
                    LAG(ts) OVER (PARTITION BY session_id ORDER BY ts) AS prev_ts,
                    LEAD(op) OVER (PARTITION BY session_id ORDER BY ts) AS next_op,
                    (julianday(ts) - julianday(LAG(ts) OVER (PARTITION BY session_id ORDER BY ts))) AS time_since_prev
                FROM search_events
                WHERE session_id IS NOT NULL
            )
            SELECT
                session_id,
                tool,
                op,
                ts,
                prev_tool,
                next_op,
                CASE
                    WHEN tool IN ('rg', 'grep') AND prev_tool IS NULL THEN 'bypass'
                    WHEN tool = 'mcp_qdrant' AND (next_op = 'open' OR next_op = 'expand') THEN 'success'
                    WHEN tool = 'mcp_qdrant' AND time_since_prev < 0.00139
                         AND prev_tool IN ('rg', 'grep', 'mcp_qdrant') THEN 'fallback'
                    ELSE 'unknown'
                END AS behavior
            FROM windowed_events
            "#
        )
        .execute(&self.pool)
        .await?;

        info!("Migration v14 complete");
        Ok(())
    }

    /// Migration v15: Create sparse_vocabulary and corpus_statistics tables for BM25 IDF
    async fn migrate_v15(&self) -> Result<(), SchemaError> {
        info!("Migration v15: Creating BM25 IDF tables (sparse_vocabulary, corpus_statistics)");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS sparse_vocabulary (
                term_id INTEGER NOT NULL,
                term TEXT NOT NULL,
                collection TEXT NOT NULL,
                document_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (term_id, collection),
                UNIQUE (term, collection)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sparse_vocabulary_collection ON sparse_vocabulary (collection)",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS corpus_statistics (
                collection TEXT PRIMARY KEY NOT NULL,
                total_documents INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        info!("Migration v15 complete");
        Ok(())
    }

    /// Migration v16: Create keyword/tag extraction tables
    async fn migrate_v16(&self) -> Result<(), SchemaError> {
        info!("Migration v16: Creating keyword/tag extraction tables");

        use super::keywords_schema::{
            CREATE_KEYWORDS_SQL, CREATE_KEYWORDS_INDEXES_SQL,
            CREATE_TAGS_SQL, CREATE_TAGS_INDEXES_SQL,
            CREATE_KEYWORD_BASKETS_SQL, CREATE_KEYWORD_BASKETS_INDEXES_SQL,
            CREATE_CANONICAL_TAGS_SQL, CREATE_CANONICAL_TAGS_INDEXES_SQL,
            CREATE_TAG_HIERARCHY_EDGES_SQL, CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL,
        };

        // Create tables in dependency order (baskets depend on tags, hierarchy depends on canonical)
        sqlx::query(CREATE_KEYWORDS_SQL).execute(&self.pool).await?;
        sqlx::query(CREATE_TAGS_SQL).execute(&self.pool).await?;
        sqlx::query(CREATE_KEYWORD_BASKETS_SQL).execute(&self.pool).await?;
        sqlx::query(CREATE_CANONICAL_TAGS_SQL).execute(&self.pool).await?;
        sqlx::query(CREATE_TAG_HIERARCHY_EDGES_SQL).execute(&self.pool).await?;

        // Create all indexes
        for index_sql in CREATE_KEYWORDS_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }
        for index_sql in CREATE_TAGS_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }
        for index_sql in CREATE_KEYWORD_BASKETS_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }
        for index_sql in CREATE_CANONICAL_TAGS_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }
        for index_sql in CREATE_TAG_HIERARCHY_EDGES_INDEXES_SQL {
            sqlx::query(index_sql).execute(&self.pool).await?;
        }

        info!("Migration v16 complete");
        Ok(())
    }

    /// Migration v17: Create operational_state table for cross-component state tracking
    async fn migrate_v17(&self) -> Result<(), SchemaError> {
        info!("Migration v17: Creating operational_state table");

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS operational_state (
                key TEXT NOT NULL,
                component TEXT NOT NULL CHECK(component IN ('daemon', 'server', 'cli')),
                value TEXT NOT NULL,
                project_id TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                PRIMARY KEY (key, component, project_id)
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_operational_state_project ON operational_state (project_id) WHERE project_id IS NOT NULL",
        )
        .execute(&self.pool)
        .await?;

        info!("Migration v17 complete");
        Ok(())
    }

    /// Migration v18: Create indexed_content cache table for diff-based skip detection
    async fn migrate_v18(&self) -> Result<(), SchemaError> {
        info!("Migration v18: Creating indexed_content cache table");

        use super::indexed_content_schema::{CREATE_INDEXED_CONTENT_SQL, CREATE_INDEXED_CONTENT_INDEXES_SQL};

        sqlx::query(CREATE_INDEXED_CONTENT_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_INDEXED_CONTENT_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        info!("Migration v18 complete");
        Ok(())
    }

    /// Migration v19: Add base_point, relative_path, incremental columns to tracked_files
    ///
    /// These columns support content-addressed identity for state integrity:
    /// - base_point: SHA256(tenant_id|branch|relative_path|file_hash)[:32]
    /// - relative_path: normalized path within project root (mirrors file_path)
    /// - incremental: whether this file supports chunk-level updates
    async fn migrate_v19(&self) -> Result<(), SchemaError> {
        info!("Migration v19: Adding base_point, relative_path, incremental to tracked_files");

        use super::tracked_files_schema::{
            MIGRATE_V19_ADD_COLUMNS_SQL,
            CREATE_BASE_POINT_INDEX_SQL,
            CREATE_REFCOUNT_INDEX_SQL,
        };

        // Check if columns already exist (handles fresh installs)
        let has_base_point: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'base_point'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_base_point {
            for alter_sql in MIGRATE_V19_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            debug!("base_point column already exists, skipping ALTER TABLE");
        }

        // Backfill relative_path from file_path (already stored as relative)
        let backfilled: u64 = sqlx::query(
            "UPDATE tracked_files SET relative_path = file_path WHERE relative_path IS NULL"
        )
        .execute(&self.pool)
        .await?
        .rows_affected();

        if backfilled > 0 {
            info!("Backfilled relative_path for {} rows", backfilled);
        }

        // Backfill base_point using Rust hash computation
        // Requires joining with watch_folders to get tenant_id
        let rows: Vec<(i64, String, String, String, String)> = sqlx::query_as(
            "SELECT tf.file_id, wf.tenant_id, COALESCE(tf.branch, 'default'), tf.file_path, tf.file_hash
             FROM tracked_files tf
             JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
             WHERE tf.base_point IS NULL"
        )
        .fetch_all(&self.pool)
        .await?;

        if !rows.is_empty() {
            info!("Backfilling base_point for {} existing rows", rows.len());
            for (file_id, tenant_id, branch, file_path, file_hash) in &rows {
                let base_point = wqm_common::hashing::compute_base_point(
                    tenant_id, branch, file_path, file_hash,
                );
                sqlx::query(
                    "UPDATE tracked_files SET base_point = ?1 WHERE file_id = ?2"
                )
                .bind(&base_point)
                .bind(file_id)
                .execute(&self.pool)
                .await?;
            }
            info!("base_point backfill complete");
        }

        // Create indexes
        sqlx::query(CREATE_BASE_POINT_INDEX_SQL)
            .execute(&self.pool)
            .await?;
        sqlx::query(CREATE_REFCOUNT_INDEX_SQL)
            .execute(&self.pool)
            .await?;

        info!("Migration v19 complete");
        Ok(())
    }

    /// Migration v20: Add qdrant_status, search_status, decision_json to unified_queue
    ///
    /// Per-destination status tracking for dual-write queue processing.
    /// Items are only "done" when both Qdrant and search DB writes complete.
    async fn migrate_v20(&self) -> Result<(), SchemaError> {
        info!("Migration v20: Adding per-destination status columns to unified_queue");

        use super::unified_queue_schema::{
            MIGRATE_V20_ADD_COLUMNS_SQL,
            CREATE_QDRANT_STATUS_INDEX_SQL,
            CREATE_SEARCH_STATUS_INDEX_SQL,
        };

        // Check if columns already exist (handles fresh installs)
        let has_qdrant_status: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('unified_queue') WHERE name = 'qdrant_status'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_qdrant_status {
            for alter_sql in MIGRATE_V20_ADD_COLUMNS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }
        } else {
            debug!("qdrant_status column already exists, skipping ALTER TABLE");
        }

        // Backfill: set qdrant_status and search_status based on existing status
        // - done items → both destinations done
        // - failed items → leave as pending (will be re-evaluated)
        // - pending/in_progress → leave as pending
        let backfilled: u64 = sqlx::query(
            "UPDATE unified_queue SET qdrant_status = 'done', search_status = 'done'
             WHERE status = 'done' AND qdrant_status = 'pending'"
        )
        .execute(&self.pool)
        .await?
        .rows_affected();

        if backfilled > 0 {
            info!("Backfilled qdrant_status/search_status for {} done items", backfilled);
        }

        // Create indexes
        sqlx::query(CREATE_QDRANT_STATUS_INDEX_SQL)
            .execute(&self.pool)
            .await?;
        sqlx::query(CREATE_SEARCH_STATUS_INDEX_SQL)
            .execute(&self.pool)
            .await?;

        info!("Migration v20 complete");
        Ok(())
    }

    async fn migrate_v21(&self) -> Result<(), SchemaError> {
        info!("Migration v21: Adding git-tracking columns, memory_mirror, and submodule junction table");

        use super::watch_folders_schema::{
            MIGRATE_V21_WATCH_FOLDERS_SQL,
            CREATE_MEMORY_MIRROR_SQL,
            CREATE_WATCH_FOLDER_SUBMODULES_SQL,
            CREATE_WATCH_FOLDER_SUBMODULES_INDEXES_SQL,
            MIGRATE_V21_SUBMODULE_DATA_SQL,
        };

        // 1. Add git-tracking columns to watch_folders (idempotent check)
        let has_last_commit_hash: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'last_commit_hash'"
        )
        .fetch_one(&self.pool)
        .await?;

        if !has_last_commit_hash {
            for alter_sql in MIGRATE_V21_WATCH_FOLDERS_SQL {
                debug!("Running ALTER TABLE: {}", alter_sql);
                sqlx::query(alter_sql)
                    .execute(&self.pool)
                    .await?;
            }

            // Backfill: detect git-tracked projects (those with non-NULL git_remote_url)
            let backfilled: u64 = sqlx::query(
                "UPDATE watch_folders SET is_git_tracked = 1 WHERE git_remote_url IS NOT NULL"
            )
            .execute(&self.pool)
            .await?
            .rows_affected();

            if backfilled > 0 {
                info!("Backfilled is_git_tracked for {} watch folders with git remote URLs", backfilled);
            }
        } else {
            debug!("last_commit_hash column already exists, skipping ALTER TABLE");
        }

        // 2. Create memory_mirror table
        sqlx::query(CREATE_MEMORY_MIRROR_SQL)
            .execute(&self.pool)
            .await?;

        // 3. Create watch_folder_submodules junction table
        sqlx::query(CREATE_WATCH_FOLDER_SUBMODULES_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_WATCH_FOLDER_SUBMODULES_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        // 4. Migrate existing submodule relationships to junction table
        let migrated: u64 = sqlx::query(MIGRATE_V21_SUBMODULE_DATA_SQL)
            .execute(&self.pool)
            .await?
            .rows_affected();

        if migrated > 0 {
            info!("Migrated {} submodule relationships to junction table", migrated);
        }

        info!("Migration v21 complete");
        Ok(())
    }
}

/// Check if schema is initialized (for graceful degradation by MCP/CLI)
pub async fn is_schema_initialized(pool: &SqlitePool) -> bool {
    let result: Result<bool, _> = sqlx::query_scalar(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )
    .fetch_optional(pool)
    .await
    .map(|opt: Option<i32>| opt.is_some());

    result.unwrap_or(false)
}

/// Get schema version (for health checks)
pub async fn get_schema_version(pool: &SqlitePool) -> Option<i32> {
    let manager = SchemaManager::new(pool.clone());
    manager.get_current_version().await.ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;
    use std::time::Duration;

    /// Create an in-memory SQLite pool for testing
    async fn create_test_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .acquire_timeout(Duration::from_secs(5))
            .connect("sqlite::memory:")
            .await
            .expect("Failed to create in-memory SQLite pool")
    }

    #[test]
    fn test_sql_constant_is_valid() {
        assert!(CREATE_SCHEMA_VERSION_SQL.contains("CREATE TABLE"));
        assert!(CREATE_SCHEMA_VERSION_SQL.contains("schema_version"));
        assert!(CREATE_SCHEMA_VERSION_SQL.contains("version INTEGER PRIMARY KEY"));
    }

    #[test]
    fn test_current_version_is_positive() {
        assert!(CURRENT_SCHEMA_VERSION > 0);
    }

    #[tokio::test]
    async fn test_initialize_creates_table() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        manager.initialize().await.expect("Failed to initialize");

        // Verify table exists
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert!(exists, "schema_version table should exist after initialization");
    }

    #[tokio::test]
    async fn test_get_version_empty_db() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, None, "Version should be None for fresh database");
    }

    #[tokio::test]
    async fn test_record_and_get_version() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        manager.initialize().await.expect("Failed to initialize");
        manager.record_migration(1).await.expect("Failed to record migration");

        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, Some(1), "Version should be 1 after recording");
    }

    #[tokio::test]
    async fn test_migration_history() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        manager.initialize().await.expect("Failed to initialize");
        manager.record_migration(1).await.expect("Failed to record v1");
        manager.record_migration(2).await.expect("Failed to record v2");

        let history = manager.get_migration_history().await.expect("Failed to get history");
        assert_eq!(history.len(), 2, "Should have 2 migration entries");
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[tokio::test]
    async fn test_is_schema_initialized() {
        let pool = create_test_pool().await;

        // Before initialization
        assert!(!is_schema_initialized(&pool).await, "Should not be initialized yet");

        // After initialization
        let manager = SchemaManager::new(pool.clone());
        manager.initialize().await.expect("Failed to initialize");

        assert!(is_schema_initialized(&pool).await, "Should be initialized");
    }

    #[tokio::test]
    async fn test_get_schema_version_helper() {
        let pool = create_test_pool().await;

        // Before any migrations
        let version = get_schema_version(&pool).await;
        assert_eq!(version, None);

        // After recording a migration
        let manager = SchemaManager::new(pool.clone());
        manager.initialize().await.expect("Failed to initialize");
        manager.record_migration(3).await.expect("Failed to record");

        let version = get_schema_version(&pool).await;
        assert_eq!(version, Some(3));
    }

    #[tokio::test]
    async fn test_run_migrations_from_scratch() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Run migrations on fresh database
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify version is at CURRENT_SCHEMA_VERSION
        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        // Verify v1 tables were created
        let watch_folders_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_folders')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(watch_folders_exists, "watch_folders table should exist after migration");

        let unified_queue_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='unified_queue')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(unified_queue_exists, "unified_queue table should exist after migration");

        // Verify v2 tables were created
        let tracked_files_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(tracked_files_exists, "tracked_files table should exist after migration v2");

        let qdrant_chunks_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='qdrant_chunks')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(qdrant_chunks_exists, "qdrant_chunks table should exist after migration v2");
    }

    #[tokio::test]
    async fn test_run_migrations_idempotent() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        // Run migrations twice - should not fail
        manager.run_migrations().await.expect("First migration failed");
        manager.run_migrations().await.expect("Second migration should be idempotent");
    }

    #[tokio::test]
    async fn test_incremental_migration_v1_to_current() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Simulate starting at v1 by running v1 migration manually
        manager.initialize().await.expect("Failed to initialize");
        manager.run_migration(1).await.expect("Failed to run v1");
        manager.record_migration(1).await.expect("Failed to record v1");

        // Verify only v1 tables exist
        let tracked_exists_before: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(!tracked_exists_before, "tracked_files should NOT exist before v2 migration");

        // Run remaining migrations (v2 + v3)
        manager.run_migrations().await.expect("Failed to run migrations from v1");

        // Verify v2 tables now exist
        let tracked_exists_after: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(tracked_exists_after, "tracked_files should exist after v2 migration");

        // Verify v3 columns exist (needs_reconcile)
        let has_reconcile: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_reconcile, "needs_reconcile column should exist after v3 migration");

        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
    }

    #[tokio::test]
    async fn test_qdrant_chunks_cascade_delete() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Run all migrations
        manager.run_migrations().await.expect("Failed to run migrations");

        // Enable foreign keys for this connection
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool)
            .await
            .unwrap();

        // Insert a watch_folder first (required FK)
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a tracked_file
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, chunk_count, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', '2025-01-01T00:00:00Z', 'abc123', 2, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Get the file_id
        let file_id: i64 = sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'src/main.rs'")
            .fetch_one(&pool)
            .await
            .unwrap();

        // Insert two qdrant_chunks
        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
             VALUES (?1, 'point-1', 0, 'hash1', '2025-01-01T00:00:00Z')"
        )
        .bind(file_id)
        .execute(&pool)
        .await
        .unwrap();

        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
             VALUES (?1, 'point-2', 1, 'hash2', '2025-01-01T00:00:00Z')"
        )
        .bind(file_id)
        .execute(&pool)
        .await
        .unwrap();

        // Verify 2 chunks exist
        let chunk_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(chunk_count, 2);

        // Delete the tracked_file - should CASCADE to qdrant_chunks
        sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id)
            .execute(&pool)
            .await
            .unwrap();

        // Verify chunks are also deleted
        let chunk_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(chunk_count_after, 0, "qdrant_chunks should be deleted via CASCADE");
    }

    #[tokio::test]
    async fn test_tracked_files_unique_constraint() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Insert a watch_folder
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert a tracked_file
        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        // Duplicate insert should fail (same watch_folder_id, file_path, branch)
        let result = sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-02T00:00:00Z', 'hash2', '2025-01-02T00:00:00Z', '2025-01-02T00:00:00Z')"
        )
        .execute(&pool)
        .await;

        assert!(result.is_err(), "Duplicate (watch_folder_id, file_path, branch) should violate UNIQUE constraint");
    }

    #[tokio::test]
    async fn test_downgrade_not_supported() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        // Initialize and record a future version
        manager.initialize().await.expect("Failed to initialize");
        manager.record_migration(CURRENT_SCHEMA_VERSION + 10).await.expect("Failed to record future version");

        // Attempt to run migrations should fail with DowngradeNotSupported
        let result = manager.run_migrations().await;
        assert!(result.is_err());

        match result.unwrap_err() {
            SchemaError::DowngradeNotSupported { db_version, code_version } => {
                assert_eq!(db_version, CURRENT_SCHEMA_VERSION + 10);
                assert_eq!(code_version, CURRENT_SCHEMA_VERSION);
            }
            other => panic!("Expected DowngradeNotSupported error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_incremental_migration_v2_to_v3() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Simulate v2 state: run v1 and v2 only
        manager.initialize().await.expect("Failed to initialize");
        manager.run_migration(1).await.expect("Failed to run v1");
        manager.record_migration(1).await.expect("Failed to record v1");
        manager.run_migration(2).await.expect("Failed to run v2");
        manager.record_migration(2).await.expect("Failed to record v2");

        // Verify needs_reconcile column does NOT exist yet (v2 CREATE TABLE doesn't have it
        // when created via migration, since the SQL constant was updated but migration v2
        // runs the CREATE TABLE which now includes the column)
        // Actually, since we updated CREATE_TRACKED_FILES_SQL, v2 migration creates the table
        // with the column already. v3 migration handles this gracefully.

        // Run remaining migrations (v3, v4)
        manager.run_migrations().await.expect("Failed to run migrations from v2");

        // Verify all migrations completed
        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        // Verify the reconcile index exists
        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_reconcile'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_index, "Reconcile index should exist after v3 migration");
    }

    #[tokio::test]
    async fn test_incremental_migration_v3_to_v4() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Simulate v3 state: run v1, v2, v3 only
        manager.initialize().await.expect("Failed to initialize");
        manager.run_migration(1).await.expect("Failed to run v1");
        manager.record_migration(1).await.expect("Failed to record v1");
        manager.run_migration(2).await.expect("Failed to run v2");
        manager.record_migration(2).await.expect("Failed to record v2");
        manager.run_migration(3).await.expect("Failed to run v3");
        manager.record_migration(3).await.expect("Failed to record v3");

        // Run remaining migrations (v4)
        manager.run_migrations().await.expect("Failed to run migrations from v3 to v4");

        // Verify all migrations completed (v4 + v5)
        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        // Verify is_paused column exists
        let has_is_paused: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_is_paused, "is_paused column should exist after v4 migration");

        // Verify pause_start_time column exists
        let has_pause_start_time: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'pause_start_time'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_pause_start_time, "pause_start_time column should exist after v4 migration");
    }

    #[tokio::test]
    async fn test_incremental_migration_v4_to_current() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        // Simulate v4 state: run v1 through v4
        manager.initialize().await.expect("Failed to initialize");
        for v in 1..=4 {
            manager.run_migration(v).await.unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
            manager.record_migration(v).await.unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
        }

        // Verify metrics_history does NOT exist yet
        let exists_before: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(!exists_before, "metrics_history should NOT exist before v5 migration");

        // Run remaining migrations (v5, v6)
        manager.run_migrations().await.expect("Failed to run migrations from v4 to current");

        // Verify all migrations completed
        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION as i32));

        // Verify metrics_history table exists (v5)
        let exists_after: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists_after, "metrics_history table should exist after v5 migration");

        // Verify indexes exist (v5)
        let idx_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_metrics_%'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(idx_count, 3, "Should have 3 metrics_history indexes");

        // Verify collection column exists in tracked_files (v6)
        let has_collection: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_collection, "collection column should exist after v6 migration");
    }

    #[tokio::test]
    async fn test_search_behavior_view_classification() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.initialize().await.unwrap();
        manager.run_migrations().await.unwrap();

        // Insert synthetic search events for different behavior patterns
        // Session A: bypass (rg first with no prior event)
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e1', 'sess-a', 'claude', 'rg', 'search', '2025-01-01T00:00:00.000Z')"
        ).execute(&pool).await.unwrap();

        // Session B: success (mcp_qdrant followed by open)
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e2', 'sess-b', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:01:00.000Z')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e3', 'sess-b', 'claude', 'mcp_qdrant', 'open', '2025-01-01T00:01:30.000Z')"
        ).execute(&pool).await.unwrap();

        // Session C: fallback (rg then mcp_qdrant within 2 minutes — user falls back to qdrant)
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e4', 'sess-c', 'claude', 'rg', 'search', '2025-01-01T00:02:00.000Z')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e5', 'sess-c', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:02:30.000Z')"
        ).execute(&pool).await.unwrap();

        // Query the view
        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT session_id, tool, behavior FROM search_behavior ORDER BY ts"
        )
        .fetch_all(&pool)
        .await
        .unwrap();

        assert!(!rows.is_empty(), "search_behavior view should return rows");

        // Check bypass: first rg event in sess-a (no prior event)
        let bypass = rows.iter().find(|(s, _, b)| s == "sess-a" && b == "bypass");
        assert!(bypass.is_some(), "Should detect bypass pattern (rg as first event)");

        // Check success: mcp_qdrant in sess-b where next_op = 'open'
        let success = rows.iter().find(|(s, _, b)| s == "sess-b" && b == "success");
        assert!(success.is_some(), "Should detect success pattern (mcp_qdrant followed by open)");

        // Check fallback: mcp_qdrant in sess-c following rg within 2 min
        let fallback = rows.iter().find(|(s, t, b)| s == "sess-c" && t == "mcp_qdrant" && b == "fallback");
        assert!(fallback.is_some(), "Should detect fallback pattern (mcp_qdrant after rg within 2 min)");
    }

    #[tokio::test]
    async fn test_migration_v16_keywords_tables() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify all 5 tables exist
        for table in &["keywords", "tags", "keyword_baskets", "canonical_tags", "tag_hierarchy_edges"] {
            let exists: bool = sqlx::query_scalar(
                &format!("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{}')", table)
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            assert!(exists, "{} table should exist after v16 migration", table);
        }

        // Verify key indexes exist
        let idx_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND (
                name LIKE 'idx_keywords_%' OR
                name LIKE 'idx_tags_%' OR
                name LIKE 'idx_keyword_baskets_%' OR
                name LIKE 'idx_canonical_tags_%' OR
                name LIKE 'idx_hierarchy_edges_%'
            )"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(idx_count, 14, "Should have 14 keyword/tag indexes (3+3+2+3+3)");
    }

    #[tokio::test]
    async fn test_migration_v16_cascade_deletes() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Enable foreign keys
        sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

        // Insert a tag, then a basket referencing it
        sqlx::query(
            "INSERT INTO tags (doc_id, tag, collection, tenant_id) VALUES ('doc1', 'vector-search', 'projects', 't1')"
        ).execute(&pool).await.unwrap();

        let tag_id: i64 = sqlx::query_scalar("SELECT tag_id FROM tags WHERE doc_id = 'doc1'")
            .fetch_one(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO keyword_baskets (tag_id, keywords_json, tenant_id) VALUES (?1, '[\"qdrant\",\"embedding\"]', 't1')"
        ).bind(tag_id).execute(&pool).await.unwrap();

        // Verify basket exists
        let basket_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(basket_count, 1);

        // Delete the tag — basket should cascade delete
        sqlx::query("DELETE FROM tags WHERE tag_id = ?1").bind(tag_id).execute(&pool).await.unwrap();

        let basket_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(basket_count_after, 0, "keyword_baskets should cascade delete when tag is deleted");
    }

    #[tokio::test]
    async fn test_migration_v17_operational_state() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify table exists
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='operational_state')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "operational_state table should exist after v17 migration");

        // Verify index exists
        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_operational_state_project'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_index, "project index should exist on operational_state");

        // Test INSERT round-trip
        sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'test_val', '2026-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await
        .unwrap();

        let val: String = sqlx::query_scalar(
            "SELECT value FROM operational_state WHERE key = 'test_key' AND component = 'daemon'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert_eq!(val, "test_val");

        // Test CHECK constraint: invalid component should fail
        let result = sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('k2', 'invalid', 'v', '2026-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await;
        assert!(result.is_err(), "Invalid component should violate CHECK constraint");

        // Test PK uniqueness: same (key, component, project_id) should fail on plain INSERT
        let result = sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'dup', '2026-01-01T00:00:00Z')"
        )
        .execute(&pool)
        .await;
        assert!(result.is_err(), "Duplicate (key, component, project_id) should violate PRIMARY KEY");
    }

    #[tokio::test]
    async fn test_migration_v18_indexed_content() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify table exists
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_content')"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(exists, "indexed_content table should exist after v18 migration");

        // Verify index exists
        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_indexed_content_hash'"
        )
        .fetch_one(&pool)
        .await
        .unwrap();
        assert!(has_index, "hash index should exist on indexed_content");

        // Verify FK constraint structure: file_id references tracked_files
        // Insert a watch_folder and tracked_file first
        sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'test.rs', '2025-01-01T00:00:00Z', 'h1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let file_id: i64 = sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'test.rs'")
            .fetch_one(&pool).await.unwrap();

        // Insert indexed_content
        sqlx::query(
            "INSERT INTO indexed_content (file_id, content, hash, updated_at) VALUES (?1, X'48454C4C4F', 'testhash', '2025-01-01T00:00:00Z')"
        ).bind(file_id).execute(&pool).await.unwrap();

        // Verify round-trip
        let hash: String = sqlx::query_scalar("SELECT hash FROM indexed_content WHERE file_id = ?1")
            .bind(file_id).fetch_one(&pool).await.unwrap();
        assert_eq!(hash, "testhash");

        // Verify CASCADE: delete tracked_file should remove indexed_content
        sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id).execute(&pool).await.unwrap();

        let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM indexed_content")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(count, 0, "indexed_content should cascade delete with tracked_files");
    }

    #[tokio::test]
    async fn test_migration_v16_multi_tenant_isolation() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Insert keywords for two different tenants
        sqlx::query(
            "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc1', 'qdrant', 0.9, 'projects', 't1')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc2', 'redis', 0.8, 'projects', 't2')"
        ).execute(&pool).await.unwrap();

        // Query by tenant
        let t1_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM keywords WHERE tenant_id = 't1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(t1_count, 1);

        let t2_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM keywords WHERE tenant_id = 't2'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(t2_count, 1);
    }

    #[tokio::test]
    async fn test_migration_v19_base_point_columns() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify new columns exist
        let has_base_point: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'base_point'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_base_point, "base_point column should exist");

        let has_relative_path: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'relative_path'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_relative_path, "relative_path column should exist");

        let has_incremental: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'incremental'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_incremental, "incremental column should exist");

        // Verify indexes exist
        let has_bp_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_base_point'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_bp_index, "base_point index should exist");

        let has_refcount_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_refcount'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_refcount_index, "refcount index should exist");
    }

    #[tokio::test]
    async fn test_migration_v19_backfill() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Insert a watch_folder and tracked_file to test backfill
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/project', 'projects', 'tenant_abc', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'deadbeef', 'projects', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        // Verify the tracked_file has base_point and relative_path as NULL (migration already ran on fresh DB)
        // For fresh installs, columns exist from CREATE TABLE but are NULL
        let bp: Option<String> = sqlx::query_scalar(
            "SELECT base_point FROM tracked_files WHERE file_path = 'src/main.rs'"
        ).fetch_one(&pool).await.unwrap();
        // base_point is NULL because the insert above didn't provide it
        assert!(bp.is_none(), "base_point should be NULL for insert without explicit value");

        // Verify incremental defaults to 0
        let incr: i32 = sqlx::query_scalar(
            "SELECT incremental FROM tracked_files WHERE file_path = 'src/main.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(incr, 0, "incremental should default to 0");
    }

    #[sqlx::test]
    async fn test_migration_v20_destination_columns() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify new columns exist in unified_queue
        let columns: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM pragma_table_info('unified_queue') ORDER BY name"
        ).fetch_all(&pool).await.unwrap();

        assert!(columns.contains(&"qdrant_status".to_string()), "qdrant_status column missing");
        assert!(columns.contains(&"search_status".to_string()), "search_status column missing");
        assert!(columns.contains(&"decision_json".to_string()), "decision_json column missing");
    }

    #[sqlx::test]
    async fn test_migration_v20_defaults_and_constraints() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Insert an item without specifying qdrant_status/search_status — should default to 'pending'
        sqlx::query(
            "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, priority, branch, payload_json, created_at, updated_at)
             VALUES ('q1', 'k1', 'file', 'add', 't1', 'projects', 'pending', 5, 'main', '{}', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let qdrant_status: String = sqlx::query_scalar(
            "SELECT qdrant_status FROM unified_queue WHERE queue_id = 'q1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(qdrant_status, "pending", "qdrant_status should default to 'pending'");

        let search_status: String = sqlx::query_scalar(
            "SELECT search_status FROM unified_queue WHERE queue_id = 'q1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(search_status, "pending", "search_status should default to 'pending'");

        let decision: Option<String> = sqlx::query_scalar(
            "SELECT decision_json FROM unified_queue WHERE queue_id = 'q1'"
        ).fetch_one(&pool).await.unwrap();
        assert!(decision.is_none(), "decision_json should default to NULL");

        // Verify CHECK constraint rejects invalid values
        let invalid = sqlx::query(
            "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, priority, branch, payload_json, qdrant_status, created_at, updated_at)
             VALUES ('q2', 'k2', 'file', 'add', 't1', 'projects', 'pending', 5, 'main', '{}', 'bogus', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await;
        assert!(invalid.is_err(), "CHECK constraint should reject invalid qdrant_status");
    }

    #[sqlx::test]
    async fn test_migration_v21_git_tracking_columns() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify new columns exist in watch_folders
        let columns: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM pragma_table_info('watch_folders') ORDER BY name"
        ).fetch_all(&pool).await.unwrap();

        assert!(columns.contains(&"last_commit_hash".to_string()), "last_commit_hash column missing");
        assert!(columns.contains(&"is_git_tracked".to_string()), "is_git_tracked column missing");

        // Verify defaults
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w-test', '/tmp/test-project', 'projects', 'tenant_test', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let is_git_tracked: i32 = sqlx::query_scalar(
            "SELECT is_git_tracked FROM watch_folders WHERE watch_id = 'w-test'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(is_git_tracked, 0, "is_git_tracked should default to 0");

        let commit_hash: Option<String> = sqlx::query_scalar(
            "SELECT last_commit_hash FROM watch_folders WHERE watch_id = 'w-test'"
        ).fetch_one(&pool).await.unwrap();
        assert!(commit_hash.is_none(), "last_commit_hash should default to NULL");
    }

    #[sqlx::test]
    async fn test_migration_v21_memory_mirror_table() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify memory_mirror table exists
        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='memory_mirror'"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "memory_mirror table should exist");

        // Test CRUD operations
        sqlx::query(
            "INSERT INTO memory_mirror (memory_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES ('m1', 'Always use snake_case', 'global', NULL, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let rule: String = sqlx::query_scalar(
            "SELECT rule_text FROM memory_mirror WHERE memory_id = 'm1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(rule, "Always use snake_case");
    }

    #[sqlx::test]
    async fn test_migration_v21_submodule_junction_table() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        // Verify watch_folder_submodules table exists
        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='watch_folder_submodules'"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "watch_folder_submodules table should exist");

        // Create parent and child watch folders
        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w-parent', '/tmp/parent', 'projects', 'tenant1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path, created_at, updated_at)
             VALUES ('w-child', '/tmp/parent/lib', 'projects', 'tenant2', 'w-parent', 'lib', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        // Insert into junction table
        sqlx::query(
            "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, submodule_path, created_at)
             VALUES ('w-parent', 'w-child', 'lib', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folder_submodules WHERE parent_watch_id = 'w-parent'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(count, 1);

        // Test CASCADE delete — deleting parent should remove junction rows
        sqlx::query("DELETE FROM watch_folders WHERE watch_id = 'w-parent'")
            .execute(&pool).await.unwrap();

        let count_after: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folder_submodules"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(count_after, 0, "CASCADE delete should remove junction rows");
    }
}
