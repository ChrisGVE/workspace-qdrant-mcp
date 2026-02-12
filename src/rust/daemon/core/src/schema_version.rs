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
pub const CURRENT_SCHEMA_VERSION: i32 = 7;

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
}
