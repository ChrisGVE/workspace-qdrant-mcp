//! Search Database Manager
//!
//! Manages a separate SQLite database (`search.db`) for FTS5 code search index.
//! This is separated from `state.db` to eliminate lock contention between state
//! operations and FTS5 batch writes (which can take 2+ seconds).
//!
//! Schema versioning is independent from `state.db` — search.db starts at version 1.
//! WAL mode is enabled for concurrent read access during writes.

use std::path::{Path, PathBuf};
use sqlx::{SqlitePool, sqlite::{SqliteConnectOptions, SqlitePoolOptions}};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Current schema version for search.db
pub const SEARCH_SCHEMA_VERSION: i32 = 1;

/// Default search database filename
pub const SEARCH_DB_FILENAME: &str = "search.db";

/// Errors from search database operations
#[derive(Error, Debug)]
pub enum SearchDbError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Schema migration error: {0}")]
    Migration(String),

    #[error("Downgrade not supported: database version {db_version} > code version {code_version}")]
    DowngradeNotSupported { db_version: i32, code_version: i32 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for search database operations
pub type SearchDbResult<T> = Result<T, SearchDbError>;

/// Search database manager for FTS5 code search index.
///
/// Manages a separate SQLite database alongside `state.db` with:
/// - Independent schema versioning (starts at v1)
/// - WAL mode for concurrent reads during FTS5 writes
/// - Foreign keys enabled
pub struct SearchDbManager {
    pool: SqlitePool,
    path: PathBuf,
}

impl SearchDbManager {
    /// Create a new search database manager.
    ///
    /// Opens (or creates) the database at the given path, enables WAL mode
    /// and foreign keys, then runs any pending schema migrations.
    pub async fn new<P: AsRef<Path>>(database_path: P) -> SearchDbResult<Self> {
        let path = database_path.as_ref().to_path_buf();
        info!("Initializing search database: {}", path.display());

        let connect_options = SqliteConnectOptions::new()
            .filename(&path)
            .create_if_missing(true)
            .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
            .foreign_keys(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        // Verify WAL mode is active
        let journal_mode: String =
            sqlx::query_scalar("PRAGMA journal_mode")
                .fetch_one(&pool)
                .await?;
        if journal_mode.to_lowercase() != "wal" {
            warn!(
                "Expected WAL journal mode, got '{}'. Performance may be degraded.",
                journal_mode
            );
        } else {
            debug!("WAL mode confirmed for search.db");
        }

        let manager = Self { pool, path };
        manager.run_migrations().await?;

        Ok(manager)
    }

    /// Create a search database manager from an existing pool.
    ///
    /// Use when you already have a connection pool (e.g., in tests).
    /// Caller is responsible for running migrations.
    pub fn with_pool(pool: SqlitePool, path: PathBuf) -> Self {
        Self { pool, path }
    }

    /// Get a reference to the connection pool.
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Get the database file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Close the database connection pool.
    pub async fn close(&self) {
        info!("Closing search database: {}", self.path.display());
        self.pool.close().await;
    }

    // ========================================================================
    // Schema management
    // ========================================================================

    /// Get the current schema version. Returns None for a fresh database.
    pub async fn get_schema_version(&self) -> SearchDbResult<Option<i32>> {
        let table_exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_schema_version')",
        )
        .fetch_one(&self.pool)
        .await?;

        if !table_exists {
            return Ok(None);
        }

        let version: Option<i32> =
            sqlx::query_scalar("SELECT MAX(version) FROM search_schema_version")
                .fetch_optional(&self.pool)
                .await?
                .flatten();

        Ok(version)
    }

    /// Run all pending migrations up to SEARCH_SCHEMA_VERSION.
    async fn run_migrations(&self) -> SearchDbResult<()> {
        // Create the schema version table if it doesn't exist
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS search_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            )
            "#,
        )
        .execute(&self.pool)
        .await?;

        let current = self.get_schema_version().await?.unwrap_or(0);
        info!(
            "Search DB schema version: {}, target: {}",
            current, SEARCH_SCHEMA_VERSION
        );

        if current > SEARCH_SCHEMA_VERSION {
            return Err(SearchDbError::DowngradeNotSupported {
                db_version: current,
                code_version: SEARCH_SCHEMA_VERSION,
            });
        }

        if current == SEARCH_SCHEMA_VERSION {
            debug!("Search DB schema is up to date");
            return Ok(());
        }

        for version in (current + 1)..=SEARCH_SCHEMA_VERSION {
            info!("Running search DB migration to version {}", version);
            self.run_migration(version).await?;
            sqlx::query("INSERT INTO search_schema_version (version) VALUES (?1)")
                .bind(version)
                .execute(&self.pool)
                .await?;
        }

        info!(
            "Search DB migrations complete. Now at version {}",
            SEARCH_SCHEMA_VERSION
        );
        Ok(())
    }

    /// Dispatch a single migration by version number.
    async fn run_migration(&self, version: i32) -> SearchDbResult<()> {
        match version {
            1 => self.migrate_v1().await,
            _ => Err(SearchDbError::Migration(format!(
                "Unknown search DB migration version: {}",
                version
            ))),
        }
    }

    /// Migration v1: placeholder — tables will be created in Tasks 46-47.
    ///
    /// This migration establishes the search.db database with WAL mode and
    /// schema versioning. Actual FTS5 tables are added in subsequent tasks.
    async fn migrate_v1(&self) -> SearchDbResult<()> {
        info!("Search DB migration v1: database initialized (tables added in Tasks 46-47)");
        Ok(())
    }
}

/// Derive the search.db path from the state.db path.
///
/// Given `~/.workspace-qdrant/state.db`, returns `~/.workspace-qdrant/search.db`.
pub fn search_db_path_from_state(state_db_path: &Path) -> PathBuf {
    let parent = state_db_path.parent().unwrap_or(Path::new("."));
    if parent.as_os_str().is_empty() {
        PathBuf::from(SEARCH_DB_FILENAME)
    } else {
        parent.join(SEARCH_DB_FILENAME)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_create_search_db() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        assert!(db_path.exists(), "search.db should be created");
        assert_eq!(manager.path(), db_path);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_wal_mode_enabled() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let mode: String =
            sqlx::query_scalar("PRAGMA journal_mode")
                .fetch_one(manager.pool())
                .await
                .unwrap();

        assert_eq!(mode.to_lowercase(), "wal");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_foreign_keys_enabled() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let fk: i32 =
            sqlx::query_scalar("PRAGMA foreign_keys")
                .fetch_one(manager.pool())
                .await
                .unwrap();

        assert_eq!(fk, 1, "foreign_keys should be enabled");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_schema_version_after_init() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let version = manager.get_schema_version().await.unwrap();
        assert_eq!(version, Some(SEARCH_SCHEMA_VERSION));

        manager.close().await;
    }

    #[tokio::test]
    async fn test_schema_version_table_exists() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_schema_version')",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "search_schema_version table should exist");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_idempotent_initialization() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        // First init
        let manager1 = SearchDbManager::new(&db_path).await.unwrap();
        let v1 = manager1.get_schema_version().await.unwrap();
        manager1.close().await;

        // Second init — should not fail or change version
        let manager2 = SearchDbManager::new(&db_path).await.unwrap();
        let v2 = manager2.get_schema_version().await.unwrap();
        manager2.close().await;

        assert_eq!(v1, v2, "Version should be unchanged after re-init");
    }

    #[tokio::test]
    async fn test_search_db_path_from_state() {
        let state_path = PathBuf::from("/home/user/.workspace-qdrant/state.db");
        let search_path = search_db_path_from_state(&state_path);
        assert_eq!(
            search_path,
            PathBuf::from("/home/user/.workspace-qdrant/search.db")
        );
    }

    #[tokio::test]
    async fn test_search_db_path_from_state_relative() {
        let state_path = PathBuf::from("state.db");
        let search_path = search_db_path_from_state(&state_path);
        assert_eq!(search_path, PathBuf::from("search.db"));
    }

    #[tokio::test]
    async fn test_concurrent_reads_during_write() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");

        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Create a simple test table
        sqlx::query("CREATE TABLE IF NOT EXISTS test_data (id INTEGER PRIMARY KEY, value TEXT)")
            .execute(manager.pool())
            .await
            .unwrap();

        // Insert some data
        sqlx::query("INSERT INTO test_data (id, value) VALUES (1, 'hello')")
            .execute(manager.pool())
            .await
            .unwrap();

        // Start a write transaction
        let pool = manager.pool().clone();
        let write_handle = tokio::spawn(async move {
            let mut tx = pool.begin().await.unwrap();
            sqlx::query("INSERT INTO test_data (id, value) VALUES (2, 'world')")
                .execute(&mut *tx)
                .await
                .unwrap();
            // Hold the transaction open briefly
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            tx.commit().await.unwrap();
        });

        // Concurrent read should succeed (WAL mode)
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let count: i32 =
            sqlx::query_scalar("SELECT COUNT(*) FROM test_data")
                .fetch_one(manager.pool())
                .await
                .unwrap();

        // Should see at least the first row (WAL snapshot isolation)
        assert!(count >= 1, "Should read at least 1 row concurrently");

        write_handle.await.unwrap();
        manager.close().await;
    }
}
