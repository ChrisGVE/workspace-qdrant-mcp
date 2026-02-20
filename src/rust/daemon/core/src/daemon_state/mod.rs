//! Daemon State Management Module
//!
//! This module handles SQLite-based persistence for the daemon's watch folders
//! configuration. Per ADR-003 and the 3-table SQLite compliance requirement,
//! only three tables are allowed: schema_version, unified_queue, and watch_folders.
//!
//! The daemon is the sole owner of the SQLite database schema. This module
//! initializes the spec-required tables via SchemaManager.

mod archive;
mod disambiguation;
mod lifecycle_ops;
mod operational_state;
mod pause;
mod record;
mod stats;
mod watch_crud;

#[cfg(test)]
mod tests;

// Re-export all public items to preserve backward compatibility
pub use self::operational_state::{get_operational_state, poll_pause_state, set_operational_state};
pub use self::record::WatchFolderRecord;

use std::path::Path;
use sqlx::{SqlitePool, sqlite::{SqlitePoolOptions, SqliteConnectOptions}};
use tracing::info;

use crate::schema_version::{SchemaManager, SchemaError};

/// Daemon state management errors
#[derive(thiserror::Error, Debug)]
pub enum DaemonStateError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Schema migration error: {0}")]
    Schema(#[from] SchemaError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("State error: {0}")]
    State(String),
}

/// Result type for daemon state operations
pub type DaemonStateResult<T> = Result<T, DaemonStateError>;

impl From<crate::lifecycle::WatchFolderLifecycleError> for DaemonStateError {
    fn from(err: crate::lifecycle::WatchFolderLifecycleError) -> Self {
        match err {
            crate::lifecycle::WatchFolderLifecycleError::Database(e) => Self::Database(e),
            crate::lifecycle::WatchFolderLifecycleError::NotFound(msg) => Self::State(msg),
        }
    }
}

/// Daemon state manager for SQLite persistence
pub struct DaemonStateManager {
    pool: SqlitePool,
}

impl DaemonStateManager {
    /// Create a new daemon state manager
    pub async fn new<P: AsRef<Path>>(database_path: P) -> DaemonStateResult<Self> {
        info!("Initializing daemon state manager with database: {}",
            database_path.as_ref().display());

        let connect_options = SqliteConnectOptions::new()
            .filename(database_path.as_ref())
            .create_if_missing(true);

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_with(connect_options)
            .await?;

        Ok(Self { pool })
    }

    /// Create a daemon state manager from an existing pool
    ///
    /// Use this when you already have a database pool (e.g., in gRPC services)
    /// and don't want to create a new connection.
    pub fn with_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    /// Initialize the database schema
    ///
    /// Per ADR-003, the daemon owns the SQLite database and is responsible for:
    /// Running schema migrations for spec-required tables (schema_version,
    /// unified_queue, watch_folders) - these are the ONLY allowed tables per
    /// the 3-table SQLite compliance requirement.
    pub async fn initialize(&self) -> DaemonStateResult<()> {
        info!("Initializing daemon state database schema");

        // Run schema migrations for spec-required tables
        // This creates schema_version, unified_queue, and watch_folders tables
        let schema_manager = SchemaManager::new(self.pool.clone());
        schema_manager.run_migrations().await?;

        let version = schema_manager.get_current_version().await?.unwrap_or(0);
        info!("Schema at version {}", version);

        info!("Daemon state database schema initialized successfully");
        Ok(())
    }

    /// Close the database connection
    pub async fn close(&self) -> DaemonStateResult<()> {
        info!("Closing daemon state manager");
        self.pool.close().await;
        Ok(())
    }

    /// Get a reference to the underlying SQLite pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

impl Clone for DaemonStateManager {
    fn clone(&self) -> Self {
        Self {
            pool: self.pool.clone(),
        }
    }
}
