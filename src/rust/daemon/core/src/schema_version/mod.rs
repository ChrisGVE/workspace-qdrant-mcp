//! Schema Version Table and Migration System
//!
//! This module implements the schema version tracking and migration system.
//! Per WORKSPACE_QDRANT_MCP.md spec v1.6.3, the daemon is the sole owner of
//! the SQLite database schema. It creates all tables on startup and handles
//! all schema version upgrades.
//!
//! Other components (MCP Server, CLI) must NOT create tables or run migrations.

pub mod migration;

mod v01;
mod v02;
mod v03;
mod v04;
mod v05;
mod v06;
mod v07;
mod v08;
mod v09;
mod v10;
mod v11;
mod v12;
mod v13;
mod v14;
mod v15;
mod v16;
mod v17;
mod v18;
mod v19;
mod v20;
mod v21;
mod v22;
mod v23;
mod v24;
mod v25;
mod v26;
mod v27;
mod v28;

use sqlx::{SqlitePool, Row, sqlite::SqliteRow};
use thiserror::Error;
use tracing::{debug, info};

pub use self::migration::{Migration, MigrationRegistry};

/// Current schema version - increment when adding new migrations
pub const CURRENT_SCHEMA_VERSION: i32 = 28;

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

    /// Get the current schema version from the database.
    /// Returns None if no version is set (fresh database).
    pub async fn get_current_version(&self) -> Result<Option<i32>, SchemaError> {
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

        let registry = Self::build_registry();

        for version in (current + 1)..=CURRENT_SCHEMA_VERSION {
            info!("Running migration to version {}", version);
            self.run_migration_from_registry(&registry, version).await?;
            self.record_migration(version).await?;
        }

        info!("Schema migrations complete. Now at version {}", CURRENT_SCHEMA_VERSION);
        Ok(())
    }

    /// Run a specific migration by version number (used by tests).
    pub async fn run_migration(&self, version: i32) -> Result<(), SchemaError> {
        let registry = Self::build_registry();
        self.run_migration_from_registry(&registry, version).await
    }

    /// Build the migration registry with all 28 migrations.
    fn build_registry() -> MigrationRegistry {
        let mut registry = MigrationRegistry::new();
        registry.register(Box::new(v01::V01Migration));
        registry.register(Box::new(v02::V02Migration));
        registry.register(Box::new(v03::V03Migration));
        registry.register(Box::new(v04::V04Migration));
        registry.register(Box::new(v05::V05Migration));
        registry.register(Box::new(v06::V06Migration));
        registry.register(Box::new(v07::V07Migration));
        registry.register(Box::new(v08::V08Migration));
        registry.register(Box::new(v09::V09Migration));
        registry.register(Box::new(v10::V10Migration));
        registry.register(Box::new(v11::V11Migration));
        registry.register(Box::new(v12::V12Migration));
        registry.register(Box::new(v13::V13Migration));
        registry.register(Box::new(v14::V14Migration));
        registry.register(Box::new(v15::V15Migration));
        registry.register(Box::new(v16::V16Migration));
        registry.register(Box::new(v17::V17Migration));
        registry.register(Box::new(v18::V18Migration));
        registry.register(Box::new(v19::V19Migration));
        registry.register(Box::new(v20::V20Migration));
        registry.register(Box::new(v21::V21Migration));
        registry.register(Box::new(v22::V22Migration));
        registry.register(Box::new(v23::V23Migration));
        registry.register(Box::new(v24::V24Migration));
        registry.register(Box::new(v25::V25Migration));
        registry.register(Box::new(v26::V26Migration));
        registry.register(Box::new(v27::V27Migration));
        registry.register(Box::new(v28::V28Migration));
        registry
    }

    /// Run a migration from the registry.
    async fn run_migration_from_registry(
        &self,
        registry: &MigrationRegistry,
        version: i32,
    ) -> Result<(), SchemaError> {
        match registry.get(version) {
            Some(migration) => migration.up(&self.pool).await,
            None => Err(SchemaError::MigrationError(format!(
                "Unknown migration version: {}", version
            ))),
        }
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

        assert!(!is_schema_initialized(&pool).await, "Should not be initialized yet");

        let manager = SchemaManager::new(pool.clone());
        manager.initialize().await.expect("Failed to initialize");

        assert!(is_schema_initialized(&pool).await, "Should be initialized");
    }

    #[tokio::test]
    async fn test_get_schema_version_helper() {
        let pool = create_test_pool().await;

        let version = get_schema_version(&pool).await;
        assert_eq!(version, None);

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

        manager.run_migrations().await.expect("Failed to run migrations");

        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        let watch_folders_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='watch_folders')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(watch_folders_exists, "watch_folders table should exist after migration");

        let unified_queue_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='unified_queue')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(unified_queue_exists, "unified_queue table should exist after migration");

        let tracked_files_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(tracked_files_exists, "tracked_files table should exist after migration v2");

        let qdrant_chunks_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='qdrant_chunks')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(qdrant_chunks_exists, "qdrant_chunks table should exist after migration v2");
    }

    #[tokio::test]
    async fn test_run_migrations_idempotent() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        manager.run_migrations().await.expect("First migration failed");
        manager.run_migrations().await.expect("Second migration should be idempotent");
    }

    #[tokio::test]
    async fn test_incremental_migration_v1_to_current() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        manager.initialize().await.expect("Failed to initialize");
        manager.run_migration(1).await.expect("Failed to run v1");
        manager.record_migration(1).await.expect("Failed to record v1");

        let tracked_exists_before: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(!tracked_exists_before, "tracked_files should NOT exist before v2 migration");

        manager.run_migrations().await.expect("Failed to run migrations from v1");

        let tracked_exists_after: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='tracked_files')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(tracked_exists_after, "tracked_files should exist after v2 migration");

        let has_reconcile: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'needs_reconcile'"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(has_reconcile, "needs_reconcile column should exist after v3 migration");

        let version = manager.get_current_version().await.expect("Failed to get version");
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));
    }

    #[tokio::test]
    async fn test_qdrant_chunks_cascade_delete() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, file_mtime, file_hash, chunk_count, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', '2025-01-01T00:00:00Z', 'abc123', 2, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let file_id: i64 = sqlx::query_scalar("SELECT file_id FROM tracked_files WHERE file_path = 'src/main.rs'")
            .fetch_one(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
             VALUES (?1, 'point-1', 0, 'hash1', '2025-01-01T00:00:00Z')"
        ).bind(file_id).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO qdrant_chunks (file_id, point_id, chunk_index, content_hash, created_at)
             VALUES (?1, 'point-2', 1, 'hash2', '2025-01-01T00:00:00Z')"
        ).bind(file_id).execute(&pool).await.unwrap();

        let chunk_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(chunk_count, 2);

        sqlx::query("DELETE FROM tracked_files WHERE file_id = ?1")
            .bind(file_id).execute(&pool).await.unwrap();

        let chunk_count_after: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM qdrant_chunks")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(chunk_count_after, 0, "qdrant_chunks should be deleted via CASCADE");
    }

    #[tokio::test]
    async fn test_tracked_files_unique_constraint() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/test', 'projects', 't1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'hash1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let result = sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-02T00:00:00Z', 'hash2', '2025-01-02T00:00:00Z', '2025-01-02T00:00:00Z')"
        ).execute(&pool).await;

        assert!(result.is_err(), "Duplicate (watch_folder_id, file_path, branch) should violate UNIQUE constraint");
    }

    #[tokio::test]
    async fn test_downgrade_not_supported() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool);

        manager.initialize().await.expect("Failed to initialize");
        manager.record_migration(CURRENT_SCHEMA_VERSION + 10).await.expect("Failed to record future version");

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

        manager.initialize().await.expect("Failed to initialize");
        manager.run_migration(1).await.expect("Failed to run v1");
        manager.record_migration(1).await.expect("Failed to record v1");
        manager.run_migration(2).await.expect("Failed to run v2");
        manager.record_migration(2).await.expect("Failed to record v2");

        manager.run_migrations().await.expect("Failed to run migrations from v2");

        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_tracked_files_reconcile'"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(has_index, "Reconcile index should exist after v3 migration");
    }

    #[tokio::test]
    async fn test_incremental_migration_v3_to_v4() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        manager.initialize().await.expect("Failed to initialize");
        for v in 1..=3 {
            manager.run_migration(v).await.unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
            manager.record_migration(v).await.unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
        }

        manager.run_migrations().await.expect("Failed to run migrations from v3 to v4");

        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION));

        let has_is_paused: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'is_paused'"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(has_is_paused, "is_paused column should exist after v4 migration");

        let has_pause_start_time: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('watch_folders') WHERE name = 'pause_start_time'"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(has_pause_start_time, "pause_start_time column should exist after v4 migration");
    }

    #[tokio::test]
    async fn test_incremental_migration_v4_to_current() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());

        manager.initialize().await.expect("Failed to initialize");
        for v in 1..=4 {
            manager.run_migration(v).await.unwrap_or_else(|e| panic!("Failed to run v{}: {}", v, e));
            manager.record_migration(v).await.unwrap_or_else(|e| panic!("Failed to record v{}: {}", v, e));
        }

        let exists_before: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(!exists_before, "metrics_history should NOT exist before v5 migration");

        manager.run_migrations().await.expect("Failed to run migrations from v4 to current");

        let version = manager.get_current_version().await.unwrap();
        assert_eq!(version, Some(CURRENT_SCHEMA_VERSION as i32));

        let exists_after: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='metrics_history')"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(exists_after, "metrics_history table should exist after v5 migration");

        let idx_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name LIKE 'idx_metrics_%'"
        )
        .fetch_one(&pool).await.unwrap();
        assert_eq!(idx_count, 3, "Should have 3 metrics_history indexes");

        let has_collection: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM pragma_table_info('tracked_files') WHERE name = 'collection'"
        )
        .fetch_one(&pool).await.unwrap();
        assert!(has_collection, "collection column should exist after v6 migration");
    }

    #[tokio::test]
    async fn test_search_behavior_view_classification() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.initialize().await.unwrap();
        manager.run_migrations().await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e1', 'sess-a', 'claude', 'rg', 'search', '2025-01-01T00:00:00.000Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e2', 'sess-b', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:01:00.000Z')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e3', 'sess-b', 'claude', 'mcp_qdrant', 'open', '2025-01-01T00:01:30.000Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e4', 'sess-c', 'claude', 'rg', 'search', '2025-01-01T00:02:00.000Z')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO search_events (id, session_id, actor, tool, op, ts) VALUES ('e5', 'sess-c', 'claude', 'mcp_qdrant', 'search', '2025-01-01T00:02:30.000Z')"
        ).execute(&pool).await.unwrap();

        let rows: Vec<(String, String, String)> = sqlx::query_as(
            "SELECT session_id, tool, behavior FROM search_behavior ORDER BY ts"
        )
        .fetch_all(&pool).await.unwrap();

        assert!(!rows.is_empty(), "search_behavior view should return rows");

        let bypass = rows.iter().find(|(s, _, b)| s == "sess-a" && b == "bypass");
        assert!(bypass.is_some(), "Should detect bypass pattern (rg as first event)");

        let success = rows.iter().find(|(s, _, b)| s == "sess-b" && b == "success");
        assert!(success.is_some(), "Should detect success pattern (mcp_qdrant followed by open)");

        let fallback = rows.iter().find(|(s, t, b)| s == "sess-c" && t == "mcp_qdrant" && b == "fallback");
        assert!(fallback.is_some(), "Should detect fallback pattern (mcp_qdrant after rg within 2 min)");
    }

    #[tokio::test]
    async fn test_migration_v16_keywords_tables() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        for table in &["keywords", "tags", "keyword_baskets", "canonical_tags", "tag_hierarchy_edges"] {
            let exists: bool = sqlx::query_scalar(
                &format!("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='{}')", table)
            )
            .fetch_one(&pool).await.unwrap();
            assert!(exists, "{} table should exist after v16 migration", table);
        }

        let idx_count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND (
                name LIKE 'idx_keywords_%' OR
                name LIKE 'idx_tags_%' OR
                name LIKE 'idx_keyword_baskets_%' OR
                name LIKE 'idx_canonical_tags_%' OR
                name LIKE 'idx_hierarchy_edges_%'
            )"
        )
        .fetch_one(&pool).await.unwrap();
        assert_eq!(idx_count, 14, "Should have 14 keyword/tag indexes (3+3+2+3+3)");
    }

    #[tokio::test]
    async fn test_migration_v16_cascade_deletes() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        sqlx::query("PRAGMA foreign_keys = ON").execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tags (doc_id, tag, collection, tenant_id) VALUES ('doc1', 'vector-search', 'projects', 't1')"
        ).execute(&pool).await.unwrap();

        let tag_id: i64 = sqlx::query_scalar("SELECT tag_id FROM tags WHERE doc_id = 'doc1'")
            .fetch_one(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO keyword_baskets (tag_id, keywords_json, tenant_id) VALUES (?1, '[\"qdrant\",\"embedding\"]', 't1')"
        ).bind(tag_id).execute(&pool).await.unwrap();

        let basket_count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM keyword_baskets")
            .fetch_one(&pool).await.unwrap();
        assert_eq!(basket_count, 1);

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

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='operational_state')"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "operational_state table should exist after v17 migration");

        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_operational_state_project'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_index, "project index should exist on operational_state");

        sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'test_val', '2026-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let val: String = sqlx::query_scalar(
            "SELECT value FROM operational_state WHERE key = 'test_key' AND component = 'daemon'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(val, "test_val");

        let result = sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('k2', 'invalid', 'v', '2026-01-01T00:00:00Z')"
        ).execute(&pool).await;
        assert!(result.is_err(), "Invalid component should violate CHECK constraint");

        let result = sqlx::query(
            "INSERT INTO operational_state (key, component, value, updated_at) VALUES ('test_key', 'daemon', 'dup', '2026-01-01T00:00:00Z')"
        ).execute(&pool).await;
        assert!(result.is_err(), "Duplicate (key, component, project_id) should violate PRIMARY KEY");
    }

    #[tokio::test]
    async fn test_migration_v18_indexed_content() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='indexed_content')"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "indexed_content table should exist after v18 migration");

        let has_index: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='index' AND name='idx_indexed_content_hash'"
        ).fetch_one(&pool).await.unwrap();
        assert!(has_index, "hash index should exist on indexed_content");

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

        sqlx::query(
            "INSERT INTO indexed_content (file_id, content, hash, updated_at) VALUES (?1, X'48454C4C4F', 'testhash', '2025-01-01T00:00:00Z')"
        ).bind(file_id).execute(&pool).await.unwrap();

        let hash: String = sqlx::query_scalar("SELECT hash FROM indexed_content WHERE file_id = ?1")
            .bind(file_id).fetch_one(&pool).await.unwrap();
        assert_eq!(hash, "testhash");

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

        sqlx::query(
            "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc1', 'qdrant', 0.9, 'projects', 't1')"
        ).execute(&pool).await.unwrap();
        sqlx::query(
            "INSERT INTO keywords (doc_id, keyword, score, collection, tenant_id) VALUES ('doc2', 'redis', 0.8, 'projects', 't2')"
        ).execute(&pool).await.unwrap();

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

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w1', '/tmp/project', 'projects', 'tenant_abc', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO tracked_files (watch_folder_id, file_path, branch, file_mtime, file_hash, collection, created_at, updated_at)
             VALUES ('w1', 'src/main.rs', 'main', '2025-01-01T00:00:00Z', 'deadbeef', 'projects', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let bp: Option<String> = sqlx::query_scalar(
            "SELECT base_point FROM tracked_files WHERE file_path = 'src/main.rs'"
        ).fetch_one(&pool).await.unwrap();
        assert!(bp.is_none(), "base_point should be NULL for insert without explicit value");

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

        sqlx::query(
            "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, created_at, updated_at)
             VALUES ('q1', 'k1', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
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

        let invalid = sqlx::query(
            "INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, branch, payload_json, qdrant_status, created_at, updated_at)
             VALUES ('q2', 'k2', 'file', 'add', 't1', 'projects', 'pending', 'main', '{}', 'bogus', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await;
        assert!(invalid.is_err(), "CHECK constraint should reject invalid qdrant_status");
    }

    #[sqlx::test]
    async fn test_migration_v21_git_tracking_columns() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        let columns: Vec<String> = sqlx::query_scalar(
            "SELECT name FROM pragma_table_info('watch_folders') ORDER BY name"
        ).fetch_all(&pool).await.unwrap();

        assert!(columns.contains(&"last_commit_hash".to_string()), "last_commit_hash column missing");
        assert!(columns.contains(&"is_git_tracked".to_string()), "is_git_tracked column missing");

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
    async fn test_migration_v21_rules_mirror_table() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='rules_mirror'"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "rules_mirror table should exist");

        sqlx::query(
            "INSERT INTO rules_mirror (rule_id, rule_text, scope, tenant_id, created_at, updated_at)
             VALUES ('m1', 'Always use snake_case', 'global', NULL, '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let rule: String = sqlx::query_scalar(
            "SELECT rule_text FROM rules_mirror WHERE rule_id = 'm1'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(rule, "Always use snake_case");
    }

    #[sqlx::test]
    async fn test_migration_v21_submodule_junction_table() {
        let pool = create_test_pool().await;
        let manager = SchemaManager::new(pool.clone());
        manager.run_migrations().await.expect("Failed to run migrations");

        let exists: bool = sqlx::query_scalar(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='watch_folder_submodules'"
        ).fetch_one(&pool).await.unwrap();
        assert!(exists, "watch_folder_submodules table should exist");

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, created_at, updated_at)
             VALUES ('w-parent', '/tmp/parent', 'projects', 'tenant1', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO watch_folders (watch_id, path, collection, tenant_id, parent_watch_id, submodule_path, created_at, updated_at)
             VALUES ('w-child', '/tmp/parent/lib', 'projects', 'tenant2', 'w-parent', 'lib', '2025-01-01T00:00:00Z', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        sqlx::query(
            "INSERT INTO watch_folder_submodules (parent_watch_id, child_watch_id, submodule_path, created_at)
             VALUES ('w-parent', 'w-child', 'lib', '2025-01-01T00:00:00Z')"
        ).execute(&pool).await.unwrap();

        let count: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folder_submodules WHERE parent_watch_id = 'w-parent'"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(count, 1);

        sqlx::query("DELETE FROM watch_folders WHERE watch_id = 'w-parent'")
            .execute(&pool).await.unwrap();

        let count_after: i32 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM watch_folder_submodules"
        ).fetch_one(&pool).await.unwrap();
        assert_eq!(count_after, 0, "CASCADE delete should remove junction rows");
    }

    #[test]
    fn test_build_registry_has_all_migrations() {
        let registry = SchemaManager::build_registry();
        for v in 1..=CURRENT_SCHEMA_VERSION {
            assert!(
                registry.get(v).is_some(),
                "Migration v{} should be registered",
                v
            );
            assert_eq!(registry.get(v).unwrap().version(), v);
        }
    }
}
