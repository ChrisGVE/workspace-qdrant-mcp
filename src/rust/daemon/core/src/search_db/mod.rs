//! Search Database Manager
//!
//! Manages a separate SQLite database (`search.db`) for FTS5 code search index.
//! This is separated from `state.db` to eliminate lock contention between state
//! operations and FTS5 batch writes (which can take 2+ seconds).
//!
//! Schema versioning is independent from `state.db` -- search.db starts at version 1.
//! WAL mode is enabled for concurrent read access during writes.

mod code_lines;
mod fts;
mod migrations;
pub mod types;

#[cfg(test)]
mod tests_schema;
#[cfg(test)]
mod tests_code_lines;
#[cfg(test)]
mod tests_fts;
#[cfg(test)]
mod tests_metadata;
#[cfg(test)]
mod tests_rebalance;
#[cfg(test)]
mod tests_rebalance_stress;

pub use types::{
    InsertedLine, RebalanceResult, SearchDbError, SearchDbResult,
    search_db_path_from_state, SEARCH_DB_FILENAME, SEARCH_SCHEMA_VERSION,
};

use std::path::{Path, PathBuf};
use sqlx::{SqlitePool, sqlite::{SqliteConnectOptions, SqlitePoolOptions}};
use tracing::{debug, info, warn};

use types::SearchDbResult as Result;

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
    pub async fn new<P: AsRef<Path>>(database_path: P) -> Result<Self> {
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
    pub async fn get_schema_version(&self) -> Result<Option<i32>> {
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
    async fn run_migrations(&self) -> Result<()> {
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
            migrations::run_migration(&self.pool, version).await?;
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
}
