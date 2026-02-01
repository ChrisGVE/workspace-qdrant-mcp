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
pub const CURRENT_SCHEMA_VERSION: i32 = 1;

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
    applied_at TEXT DEFAULT (datetime('now'))
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
}
