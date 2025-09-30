// Queue Database Configuration Module
//
// Provides SQLite connection configuration for the daemon with WAL mode,
// connection pooling, and settings compatible with Python components.

use sqlx::{
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions, SqliteSynchronous},
    SqlitePool,
};
use std::path::Path;
use std::str::FromStr;
use std::time::Duration;
use tracing::{debug, info};

/// Configuration for SQLite connection pool
#[derive(Debug, Clone)]
pub struct QueueConnectionConfig {
    /// Database file path
    pub database_path: String,

    /// Maximum number of connections in pool
    pub max_connections: u32,

    /// Minimum number of idle connections
    pub min_connections: u32,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Busy timeout (time to wait when database is locked)
    pub busy_timeout: Duration,

    /// WAL checkpoint interval (in number of pages)
    pub wal_autocheckpoint: i32,

    /// Cache size (in pages, negative for KB)
    pub cache_size: i32,

    /// Memory-mapped I/O size (in bytes)
    pub mmap_size: i64,

    /// Synchronous mode
    pub synchronous: SqliteSynchronous,

    /// Whether to create database if it doesn't exist
    pub create_if_missing: bool,
}

impl Default for QueueConnectionConfig {
    fn default() -> Self {
        Self {
            database_path: "workspace_state.db".to_string(),
            max_connections: 10,
            min_connections: 2,
            connection_timeout: Duration::from_secs(30),
            busy_timeout: Duration::from_secs(30),
            wal_autocheckpoint: 1000,
            cache_size: 10000,  // ~40MB
            mmap_size: 268435456,  // 256MB
            synchronous: SqliteSynchronous::Normal,
            create_if_missing: true,
        }
    }
}

impl QueueConnectionConfig {
    /// Create configuration with custom database path
    pub fn with_database_path<P: AsRef<Path>>(path: P) -> Self {
        let mut config = Self::default();
        config.database_path = path.as_ref().to_string_lossy().to_string();
        config
    }

    /// Build SQLite connection options from configuration
    pub fn build_connection_options(&self) -> SqliteConnectOptions {
        SqliteConnectOptions::from_str(&format!("sqlite:{}", self.database_path))
            .expect("Invalid database path")
            // WAL mode for concurrent access
            .journal_mode(SqliteJournalMode::Wal)
            // Synchronous mode (NORMAL is safe with WAL)
            .synchronous(self.synchronous)
            // Enable foreign keys
            .foreign_keys(true)
            // Create database if missing
            .create_if_missing(self.create_if_missing)
            // Busy timeout
            .busy_timeout(self.busy_timeout)
            // Pragma settings applied after connection
            .pragma("cache_size", self.cache_size.to_string())
            .pragma("mmap_size", self.mmap_size.to_string())
            .pragma("temp_store", "memory")
            .pragma("wal_autocheckpoint", self.wal_autocheckpoint.to_string())
            // Optimize for performance
            .optimize_on_close(true, None)
    }

    /// Create connection pool with this configuration
    pub async fn create_pool(&self) -> Result<SqlitePool, sqlx::Error> {
        info!("Creating SQLite connection pool: {}", self.database_path);

        let pool = SqlitePoolOptions::new()
            .max_connections(self.max_connections)
            .min_connections(self.min_connections)
            .acquire_timeout(self.connection_timeout)
            .idle_timeout(Some(Duration::from_secs(600)))  // 10 minutes
            .max_lifetime(Some(Duration::from_secs(3600)))  // 1 hour
            .test_before_acquire(true)  // Verify connection health
            .connect_with(self.build_connection_options())
            .await?;

        debug!("Connection pool created with {} max connections", self.max_connections);

        // Verify WAL mode is enabled
        let row: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await?;

        if row.0.to_uppercase() != "WAL" {
            tracing::warn!("Expected WAL mode but got: {}", row.0);
        } else {
            debug!("WAL mode confirmed");
        }

        Ok(pool)
    }
}

/// Perform WAL checkpoint on a connection pool
pub async fn checkpoint_wal(
    pool: &SqlitePool,
    mode: CheckpointMode,
) -> Result<CheckpointResult, sqlx::Error> {
    let mode_str = match mode {
        CheckpointMode::Passive => "PASSIVE",
        CheckpointMode::Full => "FULL",
        CheckpointMode::Restart => "RESTART",
        CheckpointMode::Truncate => "TRUNCATE",
    };

    let query = format!("PRAGMA wal_checkpoint({})", mode_str);
    let row: (i32, i32, i32) = sqlx::query_as(&query).fetch_one(pool).await?;

    Ok(CheckpointResult {
        busy: row.0,
        log_size: row.1,
        checkpointed: row.2,
    })
}

/// WAL checkpoint modes
#[derive(Debug, Clone, Copy)]
pub enum CheckpointMode {
    /// Passive checkpoint (non-blocking)
    Passive,
    /// Full checkpoint (waits for readers)
    Full,
    /// Restart checkpoint (starts new WAL)
    Restart,
    /// Truncate checkpoint (removes WAL file)
    Truncate,
}

/// Result of WAL checkpoint operation
#[derive(Debug, Clone)]
pub struct CheckpointResult {
    /// Number of frames that couldn't be checkpointed due to concurrent readers
    pub busy: i32,
    /// Total size of WAL log in frames
    pub log_size: i32,
    /// Number of frames checkpointed
    pub checkpointed: i32,
}

impl CheckpointResult {
    /// Check if checkpoint was fully successful
    pub fn is_complete(&self) -> bool {
        self.busy == 0
    }

    /// Check if WAL has significant size
    pub fn needs_checkpoint(&self) -> bool {
        self.log_size > 1000  // More than 1000 frames
    }
}

/// Periodic WAL checkpoint task
pub async fn run_checkpoint_loop(
    pool: SqlitePool,
    interval: Duration,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) {
    info!("Starting periodic WAL checkpoint loop (interval: {:?})", interval);

    loop {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {
                debug!("Performing periodic WAL checkpoint");

                match checkpoint_wal(&pool, CheckpointMode::Passive).await {
                    Ok(result) => {
                        debug!(
                            "Checkpoint complete: busy={}, log_size={}, checkpointed={}",
                            result.busy, result.log_size, result.checkpointed
                        );

                        if !result.is_complete() {
                            debug!("Checkpoint incomplete due to concurrent readers");
                        }

                        if result.needs_checkpoint() {
                            debug!("WAL log size is large, consider more frequent checkpoints");
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to perform WAL checkpoint: {}", e);
                    }
                }
            }
            _ = shutdown.changed() => {
                if *shutdown.borrow() {
                    info!("Checkpoint loop shutting down");
                    break;
                }
            }
        }
    }

    // Perform final checkpoint on shutdown
    info!("Performing final WAL checkpoint");
    match checkpoint_wal(&pool, CheckpointMode::Full).await {
        Ok(result) => {
            info!(
                "Final checkpoint complete: checkpointed {} frames",
                result.checkpointed
            );
        }
        Err(e) => {
            tracing::error!("Failed to perform final checkpoint: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_queue.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Verify WAL mode
        let row: (String,) = sqlx::query_as("PRAGMA journal_mode")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(row.0.to_uppercase(), "WAL");

        // Verify foreign keys
        let row: (i32,) = sqlx::query_as("PRAGMA foreign_keys")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(row.0, 1);
    }

    #[tokio::test]
    async fn test_wal_checkpoint() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_checkpoint.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Create a test table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Insert some data
        for i in 0..100 {
            sqlx::query("INSERT INTO test (value) VALUES (?)")
                .bind(format!("value_{}", i))
                .execute(&pool)
                .await
                .unwrap();
        }

        // Perform checkpoint
        let result = checkpoint_wal(&pool, CheckpointMode::Passive)
            .await
            .unwrap();

        assert!(result.checkpointed > 0, "Should have checkpointed some frames");
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_concurrent.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Create test table
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)",
        )
        .execute(&pool)
        .await
        .unwrap();

        // Spawn multiple concurrent writers
        let mut handles = vec![];

        for i in 0..10 {
            let pool_clone = pool.clone();
            let handle = tokio::spawn(async move {
                for j in 0..10 {
                    sqlx::query("INSERT INTO test (value) VALUES (?)")
                        .bind(format!("thread_{}_value_{}", i, j))
                        .execute(&pool_clone)
                        .await
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            handle.await.unwrap();
        }

        // Verify all inserts succeeded
        let row: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM test")
            .fetch_one(&pool)
            .await
            .unwrap();

        assert_eq!(row.0, 100, "All concurrent inserts should succeed");
    }
}
