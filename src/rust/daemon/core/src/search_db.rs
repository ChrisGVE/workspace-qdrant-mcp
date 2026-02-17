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
pub const SEARCH_SCHEMA_VERSION: i32 = 3;

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
            2 => self.migrate_v2().await,
            3 => self.migrate_v3().await,
            _ => Err(SearchDbError::Migration(format!(
                "Unknown search DB migration version: {}",
                version
            ))),
        }
    }

    /// Migration v1: database initialization.
    ///
    /// Establishes search.db with WAL mode and schema versioning.
    async fn migrate_v1(&self) -> SearchDbResult<()> {
        info!("Search DB migration v1: database initialized");
        Ok(())
    }

    /// Migration v2: Create code_lines table for line-level code index.
    ///
    /// Uses gap-based seq ordering (REAL) for efficient line insertions.
    /// Line numbers derived via ROW_NUMBER() at query time.
    async fn migrate_v2(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::{CREATE_CODE_LINES_SQL, CREATE_CODE_LINES_INDEXES_SQL};

        info!("Search DB migration v2: creating code_lines table");

        sqlx::query(CREATE_CODE_LINES_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_CODE_LINES_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

        Ok(())
    }

    /// Migration v3: Create FTS5 trigram virtual table for substring search.
    ///
    /// External content mode links to `code_lines` via `line_id`.
    /// Trigram tokenizer enables fast substring matching.
    async fn migrate_v3(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::CREATE_CODE_LINES_FTS_SQL;

        info!("Search DB migration v3: creating code_lines_fts virtual table (FTS5 trigram)");

        sqlx::query(CREATE_CODE_LINES_FTS_SQL)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    // ========================================================================
    // FTS5 operations
    // ========================================================================

    /// Rebuild the FTS5 index from the external content table.
    ///
    /// Must be called after batch inserts/updates/deletes to `code_lines`
    /// to synchronize the FTS index with the content table.
    pub async fn rebuild_fts(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_REBUILD_SQL;
        debug!("Rebuilding FTS5 index");
        sqlx::query(FTS5_REBUILD_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Optimize the FTS5 index by merging internal b-tree segments.
    ///
    /// Call after large batch operations or periodically during idle time
    /// for improved query performance.
    pub async fn optimize_fts(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_OPTIMIZE_SQL;
        debug!("Optimizing FTS5 index");
        sqlx::query(FTS5_OPTIMIZE_SQL)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Rebuild and optionally optimize the FTS5 index.
    ///
    /// If `lines_affected` exceeds `FTS5_OPTIMIZE_THRESHOLD`, runs optimize
    /// after rebuild for better query performance.
    pub async fn rebuild_and_maybe_optimize_fts(&self, lines_affected: usize) -> SearchDbResult<()> {
        use crate::code_lines_schema::FTS5_OPTIMIZE_THRESHOLD;
        self.rebuild_fts().await?;
        if lines_affected >= FTS5_OPTIMIZE_THRESHOLD {
            info!("Lines affected ({}) >= threshold ({}), optimizing FTS5 index",
                lines_affected, FTS5_OPTIMIZE_THRESHOLD);
            self.optimize_fts().await?;
        }
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

    // ── code_lines table tests (Task 46) ──

    #[tokio::test]
    async fn test_code_lines_table_exists() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_lines')",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "code_lines table should exist after migration v2");
        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_index_exists() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name='idx_code_lines_file')",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "idx_code_lines_file index should exist");
        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_insert_and_query() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert lines for a hypothetical file_id=1
        let lines = vec!["fn main() {", "    println!(\"hello\");", "}"];
        for (i, line) in lines.iter().enumerate() {
            let seq = crate::code_lines_schema::initial_seq(i);
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(1_i64)
                .bind(seq)
                .bind(*line)
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Query with line numbers
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<i64, _>("line_number"), 1);
        assert_eq!(rows[0].get::<String, _>("content"), "fn main() {");
        assert_eq!(rows[1].get::<i64, _>("line_number"), 2);
        assert_eq!(rows[2].get::<i64, _>("line_number"), 3);
        assert_eq!(rows[2].get::<String, _>("content"), "}");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_gap_insertion() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert 3 lines with standard gaps
        for i in 0..3 {
            let seq = crate::code_lines_schema::initial_seq(i);
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(seq)
                .bind(format!("line {}", i + 1))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Insert a new line between line 1 (seq=1000) and line 2 (seq=2000)
        let mid = crate::code_lines_schema::midpoint_seq(1000.0, 2000.0);
        assert_eq!(mid, 1500.0);
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, 'inserted line')")
            .bind(mid)
            .execute(manager.pool())
            .await
            .unwrap();

        // Query and verify ordering
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].get::<String, _>("content"), "line 1");
        assert_eq!(rows[1].get::<String, _>("content"), "inserted line");
        assert_eq!(rows[1].get::<i64, _>("line_number"), 2);
        assert_eq!(rows[2].get::<String, _>("content"), "line 2");
        assert_eq!(rows[2].get::<i64, _>("line_number"), 3);
        assert_eq!(rows[3].get::<String, _>("content"), "line 3");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_unique_constraint() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'first')")
            .execute(manager.pool())
            .await
            .unwrap();

        // Inserting same (file_id, seq) should fail
        let result = sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'duplicate')")
            .execute(manager.pool())
            .await;

        assert!(result.is_err(), "Should reject duplicate (file_id, seq)");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_delete_by_file() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert lines for two files
        for i in 0..3 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(1_i64)
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("file1 line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }
        for i in 0..2 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                .bind(2_i64)
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("file2 line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Delete all lines for file 1
        sqlx::query("DELETE FROM code_lines WHERE file_id = 1")
            .execute(manager.pool())
            .await
            .unwrap();

        // File 1 should have no lines
        let count1: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count1, 0);

        // File 2 should still have its lines
        let count2: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 2")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count2, 2);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_unicode_content() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let unicode_lines = vec![
            "// 日本語コメント",
            "let π = 3.14159;",
            "println!(\"🦀 Rust!\");",
            "// Box-drawing: ┌─┐│└─┘",
        ];

        for (i, line) in unicode_lines.iter().enumerate() {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(*line)
                .execute(manager.pool())
                .await
                .unwrap();
        }

        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].get::<String, _>("content"), "// 日本語コメント");
        assert_eq!(rows[1].get::<String, _>("content"), "let π = 3.14159;");
        assert_eq!(rows[2].get::<String, _>("content"), "println!(\"🦀 Rust!\");");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_empty_content() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Empty lines are valid (blank lines in source code)
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, '')")
            .execute(manager.pool())
            .await
            .unwrap();

        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<String, _>("content"), "");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_code_lines_1000_lines() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Batch insert 1000 lines
        let mut tx = manager.pool().begin().await.unwrap();
        for i in 0..1000 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("line {}", i + 1))
                .execute(&mut *tx)
                .await
                .unwrap();
        }
        tx.commit().await.unwrap();

        // Verify count
        let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM code_lines WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count, 1000);

        // Verify line numbers 1-1000
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1000);
        assert_eq!(rows[0].get::<i64, _>("line_number"), 1);
        assert_eq!(rows[0].get::<String, _>("content"), "line 1");
        assert_eq!(rows[999].get::<i64, _>("line_number"), 1000);
        assert_eq!(rows[999].get::<String, _>("content"), "line 1000");

        manager.close().await;
    }

    // ── FTS5 trigram tests (Task 47) ──

    #[tokio::test]
    async fn test_fts5_table_exists() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='code_lines_fts')",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "code_lines_fts virtual table should exist after migration v3");
        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_match_basic() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert test code lines
        let lines = vec![
            "fn main() {",
            "    println!(\"hello world\");",
            "    let x = 42;",
            "}",
        ];
        for (i, line) in lines.iter().enumerate() {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(*line)
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Rebuild FTS index
        manager.rebuild_fts().await.unwrap();

        // Search for "println" — trigram matching
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("println")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1, "Should find exactly 1 line with 'println'");
        assert_eq!(rows[0].get::<String, _>("content"), "    println!(\"hello world\");");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_match_multiple_results() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let lines = vec![
            "fn foo() -> i32 {",
            "    return 1;",
            "}",
            "fn bar() -> i32 {",
            "    return 2;",
            "}",
        ];
        for (i, line) in lines.iter().enumerate() {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(*line)
                .execute(manager.pool())
                .await
                .unwrap();
        }

        manager.rebuild_fts().await.unwrap();

        // Search for "return" — should match 2 lines
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("return")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 2, "Should find 2 lines with 'return'");
        assert!(rows[0].get::<String, _>("content").contains("return 1"));
        assert!(rows[1].get::<String, _>("content").contains("return 2"));

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_search_by_file() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // File 1
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn hello() {}')")
            .execute(manager.pool()).await.unwrap();
        // File 2
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn hello_world() {}')")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search "hello" scoped to file 2
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_FILE_SQL)
            .bind("hello")
            .bind(2_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<i64, _>("file_id"), 2);
        assert!(rows[0].get::<String, _>("content").contains("hello_world"));

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_no_results() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn main() {}')")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search for something that doesn't exist
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("nonexistent_xyz")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 0, "Should find no results for nonexistent term");
        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_rebuild_after_insert() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert and rebuild
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'first line')")
            .execute(manager.pool()).await.unwrap();
        manager.rebuild_fts().await.unwrap();

        // Verify first line is findable
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("first")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);

        // Insert more without rebuild — FTS won't see it yet
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.0, 'second line')")
            .execute(manager.pool()).await.unwrap();

        let rows_before = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("second")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows_before.len(), 0, "New content not visible before rebuild");

        // Rebuild and verify
        manager.rebuild_fts().await.unwrap();
        let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("second")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows_after.len(), 1, "New content visible after rebuild");
        assert_eq!(rows_after[0].get::<String, _>("content"), "second line");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_rebuild_after_delete() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'deletable line')")
            .execute(manager.pool()).await.unwrap();
        manager.rebuild_fts().await.unwrap();

        // Verify it's findable
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("deletable")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);

        // Delete and rebuild
        sqlx::query("DELETE FROM code_lines WHERE file_id = 1")
            .execute(manager.pool()).await.unwrap();
        manager.rebuild_fts().await.unwrap();

        let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("deletable")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows_after.len(), 0, "Deleted content should not appear after rebuild");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_optimize() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert data and rebuild
        for i in 0..100 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("line {} content", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }
        manager.rebuild_fts().await.unwrap();

        // Optimize should not fail
        manager.optimize_fts().await.unwrap();

        // Verify search still works after optimize
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("content")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 100, "All 100 lines should match 'content'");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_rebuild_and_maybe_optimize() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert 1500 lines (above threshold)
        let mut tx = manager.pool().begin().await.unwrap();
        for i in 0..1500 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("code line {}", i))
                .execute(&mut *tx)
                .await
                .unwrap();
        }
        tx.commit().await.unwrap();

        // Should rebuild + optimize (1500 > 1000 threshold)
        manager.rebuild_and_maybe_optimize_fts(1500).await.unwrap();

        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("code line")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1500);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_rebuild_below_threshold_no_optimize() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert 50 lines (below threshold)
        for i in 0..50 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("small batch {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Should rebuild only (50 < 1000 threshold)
        manager.rebuild_and_maybe_optimize_fts(50).await.unwrap();

        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("small batch")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 50);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_unicode_search() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let lines = vec![
            "// 日本語コメント",
            "let emoji = \"🦀\";",
            "fn process_data() {}",
        ];
        for (i, line) in lines.iter().enumerate() {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(*line)
                .execute(manager.pool())
                .await
                .unwrap();
        }
        manager.rebuild_fts().await.unwrap();

        // Search for ASCII substring in mixed content
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("process_data")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<String, _>("content"), "fn process_data() {}");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_external_content_linkage() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert a line and get its line_id
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'external content test')")
            .execute(manager.pool()).await.unwrap();

        let line_id: i64 = sqlx::query_scalar(
            "SELECT line_id FROM code_lines WHERE file_id = 1 AND seq = 1000.0"
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        manager.rebuild_fts().await.unwrap();

        // FTS5 match should return the correct line_id
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("external content")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<i64, _>("line_id"), line_id,
            "FTS5 should return the correct line_id from external content table");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_schema_version_is_3() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let version = manager.get_schema_version().await.unwrap();
        assert_eq!(version, Some(3), "Schema version should be 3 after FTS5 migration");

        manager.close().await;
    }
}
