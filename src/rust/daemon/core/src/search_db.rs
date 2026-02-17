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
pub const SEARCH_SCHEMA_VERSION: i32 = 4;

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
            4 => self.migrate_v4().await,
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

    /// Migration v4: Create file_metadata table for project/branch/path scoping.
    ///
    /// Denormalizes tenant_id, branch, and file_path into search.db so
    /// FTS5 queries can be scoped without cross-database JOINs.
    async fn migrate_v4(&self) -> SearchDbResult<()> {
        use crate::code_lines_schema::{CREATE_FILE_METADATA_SQL, CREATE_FILE_METADATA_INDEXES_SQL};

        info!("Search DB migration v4: creating file_metadata table for project/branch/path scoping");

        sqlx::query(CREATE_FILE_METADATA_SQL)
            .execute(&self.pool)
            .await?;

        for index_sql in CREATE_FILE_METADATA_INDEXES_SQL {
            sqlx::query(index_sql)
                .execute(&self.pool)
                .await?;
        }

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

    // ========================================================================
    // Code line insertion with gap management (Task 50)
    // ========================================================================

    /// Insert a single code line with the given seq value.
    ///
    /// Returns the new `line_id`.
    async fn insert_code_line_raw(
        &self,
        file_id: i64,
        seq: f64,
        content: &str,
    ) -> SearchDbResult<i64> {
        let result = sqlx::query(
            "INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)",
        )
        .bind(file_id)
        .bind(seq)
        .bind(content)
        .execute(&self.pool)
        .await?;

        Ok(result.last_insert_rowid())
    }

    /// Insert a code line between two adjacent seq values using midpoint insertion.
    ///
    /// Computes `new_seq = (before_seq + after_seq) / 2.0` and inserts there.
    /// If the resulting gap is below `MIN_SEQ_GAP`, triggers file-local rebalancing.
    ///
    /// Returns the inserted line's `line_id` and `seq`, plus whether rebalancing occurred.
    pub async fn insert_line_between(
        &self,
        file_id: i64,
        before_seq: f64,
        after_seq: f64,
        content: &str,
    ) -> SearchDbResult<(InsertedLine, bool)> {
        use crate::code_lines_schema::{midpoint_seq, MIN_SEQ_GAP};

        let new_seq = midpoint_seq(before_seq, after_seq);
        let gap = (after_seq - before_seq) / 2.0;

        let line_id = self.insert_code_line_raw(file_id, new_seq, content).await?;

        // Check if rebalancing is needed
        let rebalanced = if gap < MIN_SEQ_GAP {
            debug!(
                "Gap {:.6} < MIN_SEQ_GAP {:.6} for file_id={}, triggering rebalance",
                gap, MIN_SEQ_GAP, file_id
            );
            self.rebalance_file_seqs(file_id).await?;
            true
        } else {
            false
        };

        // If rebalanced, the seq may have changed — look up the new value
        let final_seq = if rebalanced {
            sqlx::query_scalar("SELECT seq FROM code_lines WHERE line_id = ?1")
                .bind(line_id)
                .fetch_one(&self.pool)
                .await?
        } else {
            new_seq
        };

        Ok((InsertedLine { line_id, seq: final_seq }, rebalanced))
    }

    /// Insert a code line at the start of a file (before all existing lines).
    ///
    /// If the file has existing lines, the new line gets `seq = first_seq / 2.0`.
    /// If the file is empty, the new line gets `seq = INITIAL_SEQ_GAP`.
    /// Triggers rebalance if the new seq is below `MIN_SEQ_GAP`.
    pub async fn insert_line_at_start(
        &self,
        file_id: i64,
        content: &str,
    ) -> SearchDbResult<(InsertedLine, bool)> {
        use crate::code_lines_schema::{INITIAL_SEQ_GAP, MIN_SEQ_GAP};

        let first_seq: Option<f64> = sqlx::query_scalar(
            "SELECT MIN(seq) FROM code_lines WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let new_seq = match first_seq {
            Some(first) => first / 2.0,
            None => INITIAL_SEQ_GAP,
        };

        let line_id = self.insert_code_line_raw(file_id, new_seq, content).await?;

        let rebalanced = if first_seq.is_some() && new_seq < MIN_SEQ_GAP {
            debug!(
                "Start-of-file seq {:.6} < MIN_SEQ_GAP for file_id={}, triggering rebalance",
                new_seq, file_id
            );
            self.rebalance_file_seqs(file_id).await?;
            true
        } else {
            false
        };

        let final_seq = if rebalanced {
            sqlx::query_scalar("SELECT seq FROM code_lines WHERE line_id = ?1")
                .bind(line_id)
                .fetch_one(&self.pool)
                .await?
        } else {
            new_seq
        };

        Ok((InsertedLine { line_id, seq: final_seq }, rebalanced))
    }

    /// Insert a code line at the end of a file (after all existing lines).
    ///
    /// New line gets `seq = last_seq + INITIAL_SEQ_GAP`.
    /// If the file is empty, gets `seq = INITIAL_SEQ_GAP`.
    /// Appending never triggers rebalance since the gap is always `INITIAL_SEQ_GAP`.
    pub async fn insert_line_at_end(
        &self,
        file_id: i64,
        content: &str,
    ) -> SearchDbResult<InsertedLine> {
        use crate::code_lines_schema::INITIAL_SEQ_GAP;

        let last_seq: Option<f64> = sqlx::query_scalar(
            "SELECT MAX(seq) FROM code_lines WHERE file_id = ?1",
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let new_seq = match last_seq {
            Some(last) => last + INITIAL_SEQ_GAP,
            None => INITIAL_SEQ_GAP,
        };

        let line_id = self.insert_code_line_raw(file_id, new_seq, content).await?;

        Ok(InsertedLine { line_id, seq: new_seq })
    }

    /// Rebalance all seq values for a file.
    ///
    /// Reads all lines ordered by current seq, then reassigns seq values
    /// starting at `INITIAL_SEQ_GAP` with `INITIAL_SEQ_GAP` increments
    /// (1000.0, 2000.0, 3000.0, ...).
    ///
    /// Uses a two-phase update to avoid UNIQUE constraint violations:
    /// 1. Set all seqs to negative (`-line_id`) — guaranteed unique
    /// 2. Assign final seq values
    ///
    /// This is file-local: only affects lines with the given `file_id`.
    /// The FTS5 index does NOT need rebuilding after rebalance because
    /// only `seq` changes — `line_id` and `content` remain the same.
    ///
    /// Returns the number of lines rebalanced and the new gap.
    pub async fn rebalance_file_seqs(&self, file_id: i64) -> SearchDbResult<RebalanceResult> {
        use crate::code_lines_schema::INITIAL_SEQ_GAP;

        // Read all line_ids in current seq order
        let line_ids: Vec<i64> = sqlx::query_scalar(
            "SELECT line_id FROM code_lines WHERE file_id = ?1 ORDER BY seq",
        )
        .bind(file_id)
        .fetch_all(&self.pool)
        .await?;

        if line_ids.is_empty() {
            return Ok(RebalanceResult {
                lines_rebalanced: 0,
                new_gap: INITIAL_SEQ_GAP,
            });
        }

        let mut tx = self.pool.begin().await?;

        // Phase 1: Set all seqs to negative line_id (unique, avoids constraint violations)
        for line_id in &line_ids {
            sqlx::query("UPDATE code_lines SET seq = ?1 WHERE line_id = ?2")
                .bind(-(*line_id as f64))
                .bind(*line_id)
                .execute(&mut *tx)
                .await?;
        }

        // Phase 2: Assign final seq values
        for (i, line_id) in line_ids.iter().enumerate() {
            let new_seq = (i as f64 + 1.0) * INITIAL_SEQ_GAP;
            sqlx::query("UPDATE code_lines SET seq = ?1 WHERE line_id = ?2")
                .bind(new_seq)
                .bind(*line_id)
                .execute(&mut *tx)
                .await?;
        }

        tx.commit().await?;

        let count = line_ids.len();
        info!("Rebalanced {} lines for file_id={}", count, file_id);

        Ok(RebalanceResult {
            lines_rebalanced: count,
            new_gap: INITIAL_SEQ_GAP,
        })
    }

    /// Get the seq values of lines adjacent to a given seq in a file.
    ///
    /// Returns `(before_seq, after_seq)` where either may be `None`
    /// if the given seq is at the start or end of the file.
    pub async fn get_adjacent_seqs(
        &self,
        file_id: i64,
        target_seq: f64,
    ) -> SearchDbResult<(Option<f64>, Option<f64>)> {
        let before: Option<f64> = sqlx::query_scalar(
            "SELECT MAX(seq) FROM code_lines WHERE file_id = ?1 AND seq < ?2",
        )
        .bind(file_id)
        .bind(target_seq)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        let after: Option<f64> = sqlx::query_scalar(
            "SELECT MIN(seq) FROM code_lines WHERE file_id = ?1 AND seq > ?2",
        )
        .bind(file_id)
        .bind(target_seq)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        Ok((before, after))
    }

    /// Check if a gap between two seq values requires rebalancing.
    pub fn needs_rebalance(gap: f64) -> bool {
        use crate::code_lines_schema::MIN_SEQ_GAP;
        gap < MIN_SEQ_GAP
    }

    /// Get the minimum gap between adjacent seq values for a file.
    ///
    /// Returns `None` if the file has fewer than 2 lines.
    pub async fn min_seq_gap(&self, file_id: i64) -> SearchDbResult<Option<f64>> {
        let gap: Option<f64> = sqlx::query_scalar(
            r#"
            SELECT MIN(next_seq - seq) FROM (
                SELECT seq, LEAD(seq) OVER (ORDER BY seq) AS next_seq
                FROM code_lines
                WHERE file_id = ?1
            ) WHERE next_seq IS NOT NULL
            "#,
        )
        .bind(file_id)
        .fetch_optional(&self.pool)
        .await?
        .flatten();

        Ok(gap)
    }
}

/// Result of a code line insertion.
#[derive(Debug, Clone)]
pub struct InsertedLine {
    pub line_id: i64,
    pub seq: f64,
}

/// Result of a rebalance operation.
#[derive(Debug, Clone)]
pub struct RebalanceResult {
    pub lines_rebalanced: usize,
    pub new_gap: f64,
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
    async fn test_schema_version_is_current() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let version = manager.get_schema_version().await.unwrap();
        assert_eq!(version, Some(SEARCH_SCHEMA_VERSION), "Schema version should be current");

        manager.close().await;
    }

    // ── file_metadata tests (Task 6: project/branch/path scoping) ──

    #[tokio::test]
    async fn test_file_metadata_table_exists() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='file_metadata')",
        )
        .fetch_one(manager.pool())
        .await
        .unwrap();

        assert!(exists, "file_metadata table should exist after migration v4");
        manager.close().await;
    }

    #[tokio::test]
    async fn test_file_metadata_indexes_exist() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        for idx_name in &["idx_file_metadata_tenant", "idx_file_metadata_tenant_branch"] {
            let exists: bool = sqlx::query_scalar(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='index' AND name=?1)",
            )
            .bind(*idx_name)
            .fetch_one(manager.pool())
            .await
            .unwrap();

            assert!(exists, "Index {} should exist", idx_name);
        }

        manager.close().await;
    }

    #[tokio::test]
    async fn test_file_metadata_upsert() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64)
            .bind("project-abc")
            .bind("main")
            .bind("/src/lib.rs")
            .execute(manager.pool())
            .await
            .unwrap();

        let row = sqlx::query("SELECT tenant_id, branch, file_path FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(row.get::<String, _>("tenant_id"), "project-abc");
        assert_eq!(row.get::<String, _>("branch"), "main");
        assert_eq!(row.get::<String, _>("file_path"), "/src/lib.rs");

        // Upsert (update branch)
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64)
            .bind("project-abc")
            .bind("feature/new")
            .bind("/src/lib.rs")
            .execute(manager.pool())
            .await
            .unwrap();

        let row2 = sqlx::query("SELECT branch FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(row2.get::<String, _>("branch"), "feature/new");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_file_metadata_null_branch() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64)
            .bind("project-xyz")
            .bind(None::<String>)
            .bind("/readme.md")
            .execute(manager.pool())
            .await
            .unwrap();

        let row = sqlx::query("SELECT branch FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert!(row.get::<Option<String>, _>("branch").is_none());

        manager.close().await;
    }

    #[tokio::test]
    async fn test_file_metadata_delete() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64)
            .bind("proj")
            .bind("main")
            .bind("/a.rs")
            .execute(manager.pool())
            .await
            .unwrap();

        sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_SQL)
            .bind(1_i64)
            .execute(manager.pool())
            .await
            .unwrap();

        let count: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE file_id = 1")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count, 0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_file_metadata_delete_by_tenant() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        for i in 1..=3 {
            sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
                .bind(i as i64)
                .bind("proj-a")
                .bind("main")
                .bind(format!("/file{}.rs", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }
        // Different tenant
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(10_i64)
            .bind("proj-b")
            .bind("main")
            .bind("/other.rs")
            .execute(manager.pool())
            .await
            .unwrap();

        sqlx::query(crate::code_lines_schema::DELETE_FILE_METADATA_BY_TENANT_SQL)
            .bind("proj-a")
            .execute(manager.pool())
            .await
            .unwrap();

        let count_a: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-a'")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count_a, 0, "All proj-a rows should be deleted");

        let count_b: i32 = sqlx::query_scalar("SELECT COUNT(*) FROM file_metadata WHERE tenant_id = 'proj-b'")
            .fetch_one(manager.pool())
            .await
            .unwrap();
        assert_eq!(count_b, 1, "proj-b should be untouched");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_search_by_project() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert code_lines + file_metadata for two projects
        // Project A - file 1
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn alpha() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64).bind("proj-a").bind("main").bind("/src/alpha.rs")
            .execute(manager.pool()).await.unwrap();

        // Project B - file 2
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn alpha_beta() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(2_i64).bind("proj-b").bind("main").bind("/src/beta.rs")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search "alpha" scoped to proj-a
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_SQL)
            .bind("alpha")
            .bind("proj-a")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1, "Should find only proj-a's match");
        assert_eq!(rows[0].get::<i64, _>("file_id"), 1);
        assert_eq!(rows[0].get::<String, _>("tenant_id"), "proj-a");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_search_by_project_branch() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Same project, different branches
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn feature_code() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64).bind("proj-a").bind("main").bind("/src/main.rs")
            .execute(manager.pool()).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn feature_code_v2() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(2_i64).bind("proj-a").bind("feature/v2").bind("/src/main.rs")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search "feature_code" on branch "feature/v2"
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_BRANCH_SQL)
            .bind("feature_code")
            .bind("proj-a")
            .bind("feature/v2")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<String, _>("branch"), "feature/v2");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_search_by_path_prefix() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn handler() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64).bind("proj").bind("main").bind("/src/api/handler.rs")
            .execute(manager.pool()).await.unwrap();

        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn handler_test() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(2_i64).bind("proj").bind("main").bind("/tests/api_test.rs")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search "handler" under /src/
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PATH_PREFIX_SQL)
            .bind("handler")
            .bind("/src/%")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1, "Should find only the /src/ file");
        assert_eq!(rows[0].get::<String, _>("file_path"), "/src/api/handler.rs");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_search_by_project_path() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // proj-a, /src/
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'fn widget() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(1_i64).bind("proj-a").bind("main").bind("/src/widget.rs")
            .execute(manager.pool()).await.unwrap();

        // proj-b, /src/
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, 1000.0, 'fn widget_v2() {}')")
            .execute(manager.pool()).await.unwrap();
        sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
            .bind(2_i64).bind("proj-b").bind("main").bind("/src/widget.rs")
            .execute(manager.pool()).await.unwrap();

        manager.rebuild_fts().await.unwrap();

        // Search "widget" scoped to proj-a + /src/
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_PATH_SQL)
            .bind("widget")
            .bind("proj-a")
            .bind("/src/%")
            .fetch_all(manager.pool())
            .await
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get::<String, _>("tenant_id"), "proj-a");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_fts5_scoped_search_performance_1000_files() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert 1000 files across 10 projects, each with 10 code lines
        let mut tx = manager.pool().begin().await.unwrap();
        for file_idx in 0..1000_i64 {
            let tenant = format!("proj-{}", file_idx % 10);
            let file_path = format!("/src/module{}/file{}.rs", file_idx / 100, file_idx);

            sqlx::query(crate::code_lines_schema::UPSERT_FILE_METADATA_SQL)
                .bind(file_idx)
                .bind(&tenant)
                .bind("main")
                .bind(&file_path)
                .execute(&mut *tx)
                .await
                .unwrap();

            for line_idx in 0..10 {
                let seq = crate::code_lines_schema::initial_seq(line_idx);
                let content = format!("fn process_item_{}() {{ /* file {} line {} */ }}", file_idx, file_idx, line_idx);
                sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (?1, ?2, ?3)")
                    .bind(file_idx)
                    .bind(seq)
                    .bind(&content)
                    .execute(&mut *tx)
                    .await
                    .unwrap();
            }
        }
        tx.commit().await.unwrap();
        manager.rebuild_fts().await.unwrap();

        // Scoped search: "process_item" within proj-0 (100 files)
        let start = std::time::Instant::now();
        let rows = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_BY_PROJECT_SQL)
            .bind("process_item")
            .bind("proj-0")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        let elapsed = start.elapsed();

        // Should return ~1000 lines (100 files * 10 lines each)
        assert_eq!(rows.len(), 1000, "Should find 1000 lines for proj-0");
        assert!(elapsed.as_millis() < 5000, "Scoped search should complete in <5s, took {}ms", elapsed.as_millis());

        // Verify all results are from proj-0
        for row in &rows {
            assert_eq!(row.get::<String, _>("tenant_id"), "proj-0");
        }

        manager.close().await;
    }

    // ── seq gap management and rebalancing tests (Task 50) ──

    #[tokio::test]
    async fn test_insert_line_at_end_empty_file() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let result = manager.insert_line_at_end(1, "first line").await.unwrap();
        assert_eq!(result.seq, 1000.0);
        assert!(result.line_id > 0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_line_at_end_existing_lines() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        manager.insert_line_at_end(1, "line 1").await.unwrap();
        let r2 = manager.insert_line_at_end(1, "line 2").await.unwrap();
        assert_eq!(r2.seq, 2000.0);

        let r3 = manager.insert_line_at_end(1, "line 3").await.unwrap();
        assert_eq!(r3.seq, 3000.0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_line_at_start_empty_file() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let (result, rebalanced) = manager.insert_line_at_start(1, "first line").await.unwrap();
        assert_eq!(result.seq, 1000.0);
        assert!(!rebalanced);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_line_at_start_existing_lines() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert initial line at seq=1000.0
        manager.insert_line_at_end(1, "original first").await.unwrap();

        // Insert at start: should get seq=500.0
        let (result, rebalanced) = manager.insert_line_at_start(1, "new first").await.unwrap();
        assert_eq!(result.seq, 500.0);
        assert!(!rebalanced);

        // Verify ordering
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].get::<String, _>("content"), "new first");
        assert_eq!(rows[1].get::<String, _>("content"), "original first");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_line_between_basic() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert two lines with standard gap
        manager.insert_line_at_end(1, "line 1").await.unwrap();
        manager.insert_line_at_end(1, "line 2").await.unwrap();

        // Insert between them: midpoint of 1000.0 and 2000.0 = 1500.0
        let (result, rebalanced) = manager
            .insert_line_between(1, 1000.0, 2000.0, "inserted")
            .await
            .unwrap();
        assert_eq!(result.seq, 1500.0);
        assert!(!rebalanced);

        // Verify ordering
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<String, _>("content"), "line 1");
        assert_eq!(rows[1].get::<String, _>("content"), "inserted");
        assert_eq!(rows[2].get::<String, _>("content"), "line 2");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_needs_rebalance() {
        assert!(SearchDbManager::needs_rebalance(0.0005));
        assert!(SearchDbManager::needs_rebalance(0.0001));
        assert!(!SearchDbManager::needs_rebalance(0.001));
        assert!(!SearchDbManager::needs_rebalance(1.0));
        assert!(!SearchDbManager::needs_rebalance(1000.0));
    }

    #[tokio::test]
    async fn test_rebalance_basic() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert lines with cramped seq values
        for i in 0..5 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(1.0 + i as f64 * 0.0001)
                .bind(format!("line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Rebalance
        let result = manager.rebalance_file_seqs(1).await.unwrap();
        assert_eq!(result.lines_rebalanced, 5);
        assert_eq!(result.new_gap, 1000.0);

        // Verify new seq values
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
        assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
        assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);
        assert_eq!(rows[3].get::<f64, _>("seq"), 4000.0);
        assert_eq!(rows[4].get::<f64, _>("seq"), 5000.0);

        // Verify content order preserved
        assert_eq!(rows[0].get::<String, _>("content"), "line 0");
        assert_eq!(rows[4].get::<String, _>("content"), "line 4");

        manager.close().await;
    }

    #[tokio::test]
    async fn test_rebalance_empty_file() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        let result = manager.rebalance_file_seqs(999).await.unwrap();
        assert_eq!(result.lines_rebalanced, 0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_rebalance_file_local() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // File 1: cramped
        for i in 0..3 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(0.5 + i as f64 * 0.0001)
                .bind(format!("f1 line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // File 2: normal gaps
        for i in 0..3 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (2, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("f2 line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Rebalance file 1 only
        manager.rebalance_file_seqs(1).await.unwrap();

        // File 2 should be untouched
        let rows2 = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 2 ORDER BY seq")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows2[0].get::<f64, _>("seq"), 1000.0);
        assert_eq!(rows2[1].get::<f64, _>("seq"), 2000.0);
        assert_eq!(rows2[2].get::<f64, _>("seq"), 3000.0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_between_triggers_rebalance() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert two lines very close together (gap = 0.0002)
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(1.0)
            .bind("line a")
            .execute(manager.pool())
            .await
            .unwrap();
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
            .bind(1.0002)
            .bind("line b")
            .execute(manager.pool())
            .await
            .unwrap();

        // Insert between: gap = 0.0001, below MIN_SEQ_GAP => rebalance
        let (result, rebalanced) = manager
            .insert_line_between(1, 1.0, 1.0002, "line mid")
            .await
            .unwrap();
        assert!(rebalanced, "Should have triggered rebalance");

        // After rebalance, seqs should be 1000.0, 2000.0, 3000.0
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
        assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
        assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);

        // Content order preserved: a, mid, b
        assert_eq!(rows[0].get::<String, _>("content"), "line a");
        assert_eq!(rows[1].get::<String, _>("content"), "line mid");
        assert_eq!(rows[2].get::<String, _>("content"), "line b");

        // The returned seq should match the post-rebalance value
        assert_eq!(result.seq, 2000.0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_1000_midpoint_insertions_between_adjacent() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Start with two lines
        manager.insert_line_at_end(1, "first").await.unwrap();
        manager.insert_line_at_end(1, "last").await.unwrap();

        // Insert 1000 lines between the first and second line
        // Each insertion goes between the first line and the previously inserted line
        let mut before_seq = 1000.0_f64;
        let mut after_seq = 2000.0_f64;
        let mut rebalance_count = 0;

        for i in 0..1000 {
            let (result, rebalanced) = manager
                .insert_line_between(1, before_seq, after_seq, &format!("mid {}", i))
                .await
                .unwrap();

            if rebalanced {
                rebalance_count += 1;
                // After rebalance, look up the actual seq values for the next iteration
                // Find the seq of "first" (always at line_number 1)
                let rows = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq")
                    .fetch_all(manager.pool())
                    .await
                    .unwrap();
                before_seq = rows[0].get::<f64, _>("seq");
                // The just-inserted line is now at position 1 (after "first"), so the next
                // insertion should go between "first" and the newly inserted line
                after_seq = rows[1].get::<f64, _>("seq");
            } else {
                // next insertion between "first" (before_seq unchanged) and just-inserted
                after_seq = result.seq;
            }
        }

        // Verify all lines are present and properly ordered
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1002, "Should have 1002 lines (2 original + 1000 inserted)");

        // First should still be "first", last should still be "last"
        assert_eq!(rows[0].get::<String, _>("content"), "first");
        assert_eq!(rows[1001].get::<String, _>("content"), "last");

        // All seq values should be strictly increasing
        let seqs: Vec<f64> = rows.iter().map(|r| r.get::<f64, _>("seq")).collect();
        for i in 1..seqs.len() {
            assert!(
                seqs[i] > seqs[i - 1],
                "seq[{}]={} should be > seq[{}]={}",
                i, seqs[i], i - 1, seqs[i - 1]
            );
        }

        // At least one rebalance should have occurred (20 midpoint halves exhaust 1000.0 gap)
        assert!(
            rebalance_count > 0,
            "Should have triggered at least one rebalance during 1000 insertions"
        );

        manager.close().await;
    }

    #[tokio::test]
    async fn test_get_adjacent_seqs() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert 3 lines
        for i in 0..3 {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(crate::code_lines_schema::initial_seq(i))
                .bind(format!("line {}", i))
                .execute(manager.pool())
                .await
                .unwrap();
        }

        // Middle line (seq=2000.0)
        let (before, after) = manager.get_adjacent_seqs(1, 2000.0).await.unwrap();
        assert_eq!(before, Some(1000.0));
        assert_eq!(after, Some(3000.0));

        // First line (seq=1000.0)
        let (before, after) = manager.get_adjacent_seqs(1, 1000.0).await.unwrap();
        assert_eq!(before, None);
        assert_eq!(after, Some(2000.0));

        // Last line (seq=3000.0)
        let (before, after) = manager.get_adjacent_seqs(1, 3000.0).await.unwrap();
        assert_eq!(before, Some(2000.0));
        assert_eq!(after, None);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_min_seq_gap() {
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // No lines => None
        let gap = manager.min_seq_gap(1).await.unwrap();
        assert_eq!(gap, None);

        // One line => None (need at least 2)
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1000.0, 'a')")
            .execute(manager.pool())
            .await
            .unwrap();
        let gap = manager.min_seq_gap(1).await.unwrap();
        assert_eq!(gap, None);

        // Two lines with gap 500.0
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1500.0, 'b')")
            .execute(manager.pool())
            .await
            .unwrap();
        let gap = manager.min_seq_gap(1).await.unwrap();
        assert_eq!(gap, Some(500.0));

        // Add line with smaller gap
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 1501.0, 'c')")
            .execute(manager.pool())
            .await
            .unwrap();
        let gap = manager.min_seq_gap(1).await.unwrap();
        assert_eq!(gap, Some(1.0));

        manager.close().await;
    }

    #[tokio::test]
    async fn test_rebalance_preserves_fts5() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert cramped lines and build FTS
        let contents = vec!["fn hello_world()", "fn goodbye_world()", "fn third_function()"];
        for (i, content) in contents.iter().enumerate() {
            sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, ?1, ?2)")
                .bind(1.0 + i as f64 * 0.0001)
                .bind(*content)
                .execute(manager.pool())
                .await
                .unwrap();
        }
        manager.rebuild_fts().await.unwrap();

        // Verify FTS works before rebalance
        let rows_before = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("hello_world")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows_before.len(), 1);

        // Rebalance
        manager.rebalance_file_seqs(1).await.unwrap();

        // FTS should still work (line_id and content unchanged, only seq changed)
        // Note: since FTS5 uses external content mode with content_rowid=line_id,
        // and line_id didn't change, the FTS index remains valid
        let rows_after = sqlx::query(crate::code_lines_schema::FTS5_SEARCH_SQL)
            .bind("hello_world")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows_after.len(), 1);
        assert_eq!(rows_after[0].get::<String, _>("content"), "fn hello_world()");

        // Verify seq values are now spread out
        let rows = sqlx::query("SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
        assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
        assert_eq!(rows[2].get::<f64, _>("seq"), 3000.0);

        manager.close().await;
    }

    #[tokio::test]
    async fn test_insert_10000_random_positions() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Insert initial 100 lines
        for i in 0..100 {
            manager
                .insert_line_at_end(1, &format!("initial {}", i))
                .await
                .unwrap();
        }

        // Insert 200 more lines at various positions (between existing lines)
        // Use a deterministic pattern to avoid true randomness in tests
        for i in 0..200 {
            // Get current lines to find insertion points
            let seqs: Vec<f64> = sqlx::query_scalar(
                "SELECT seq FROM code_lines WHERE file_id = 1 ORDER BY seq",
            )
            .fetch_all(manager.pool())
            .await
            .unwrap();

            // Pick two adjacent lines using a deterministic index
            let idx = i % (seqs.len() - 1);
            manager
                .insert_line_between(1, seqs[idx], seqs[idx + 1], &format!("inserted {}", i))
                .await
                .unwrap();
        }

        // Verify all 300 lines are present
        let rows = sqlx::query(crate::code_lines_schema::LINE_NUMBER_QUERY)
            .bind(1_i64)
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows.len(), 300);

        // All seq values should be strictly increasing
        let seqs: Vec<f64> = rows.iter().map(|r| r.get::<f64, _>("seq")).collect();
        for i in 1..seqs.len() {
            assert!(
                seqs[i] > seqs[i - 1],
                "seq[{}]={} should be > seq[{}]={} (line_number {})",
                i, seqs[i], i - 1, seqs[i - 1], i + 1
            );
        }

        manager.close().await;
    }

    #[tokio::test]
    async fn test_rebalance_with_unique_constraint_conflicts() {
        use sqlx::Row;
        let tmp = TempDir::new().unwrap();
        let db_path = tmp.path().join("search.db");
        let manager = SearchDbManager::new(&db_path).await.unwrap();

        // Create a scenario where naive rebalancing would hit UNIQUE conflicts:
        // Line at seq=2000.0 and another at seq=2000.001
        // Rebalancing to 1000.0, 2000.0 would conflict if done naively
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.0, 'a')")
            .execute(manager.pool())
            .await
            .unwrap();
        sqlx::query("INSERT INTO code_lines (file_id, seq, content) VALUES (1, 2000.001, 'b')")
            .execute(manager.pool())
            .await
            .unwrap();

        // Rebalance should succeed despite the overlap scenario
        let result = manager.rebalance_file_seqs(1).await.unwrap();
        assert_eq!(result.lines_rebalanced, 2);

        // Verify final state
        let rows = sqlx::query("SELECT seq, content FROM code_lines WHERE file_id = 1 ORDER BY seq")
            .fetch_all(manager.pool())
            .await
            .unwrap();
        assert_eq!(rows[0].get::<f64, _>("seq"), 1000.0);
        assert_eq!(rows[0].get::<String, _>("content"), "a");
        assert_eq!(rows[1].get::<f64, _>("seq"), 2000.0);
        assert_eq!(rows[1].get::<String, _>("content"), "b");

        manager.close().await;
    }
}
