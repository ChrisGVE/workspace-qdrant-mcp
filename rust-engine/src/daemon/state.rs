//! Daemon state management using SQLite

use crate::config::DatabaseConfig;
use crate::error::{DaemonError, DaemonResult};
use sqlx::{SqlitePool, Row};
use std::collections::HashMap;
use tracing::{info, debug};

/// Daemon state manager
#[derive(Debug)]
pub struct DaemonState {
    pool: SqlitePool,
}

impl DaemonState {
    /// Create a new state manager
    pub async fn new(config: &DatabaseConfig) -> DaemonResult<Self> {
        info!("Initializing database at: {}", config.sqlite_path);

        let pool = SqlitePool::connect(&config.sqlite_path).await?;

        // Run migrations
        Self::run_migrations(&pool).await?;

        Ok(Self { pool })
    }

    /// Run database migrations
    async fn run_migrations(pool: &SqlitePool) -> DaemonResult<()> {
        debug!("Running database migrations");

        // Create projects table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL UNIQUE,
                git_repository TEXT,
                git_branch TEXT,
                metadata TEXT, -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        "#)
        .execute(pool)
        .await?;

        // Create collections table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS collections (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                project_id TEXT NOT NULL,
                config TEXT, -- JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id),
                UNIQUE (name, project_id)
            )
        "#)
        .execute(pool)
        .await?;

        // Create processing_operations table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS processing_operations (
                id TEXT PRIMARY KEY,
                project_id TEXT,
                status TEXT NOT NULL,
                total_documents INTEGER DEFAULT 0,
                processed_documents INTEGER DEFAULT 0,
                failed_documents INTEGER DEFAULT 0,
                error_messages TEXT, -- JSON array
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        "#)
        .execute(pool)
        .await?;

        info!("Database migrations completed");
        Ok(())
    }

    /// Get the connection pool
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    /// Health check
    pub async fn health_check(&self) -> DaemonResult<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db_config() -> DatabaseConfig {
        DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        }
    }

    #[tokio::test]
    async fn test_daemon_state_new() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test that the state was created successfully
        assert!(state.pool().is_closed() == false);

        // Test debug formatting
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DaemonState"));
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Health check should pass
        state.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_pool_access() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Pool should be accessible
        let pool = state.pool();
        assert!(!pool.is_closed());
    }

    #[tokio::test]
    async fn test_migrations_create_tables() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Verify tables were created by querying them
        let projects_result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
            .fetch_optional(state.pool())
            .await
            .unwrap();
        assert!(projects_result.is_some());

        let collections_result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name='collections'")
            .fetch_optional(state.pool())
            .await
            .unwrap();
        assert!(collections_result.is_some());

        let processing_ops_result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name='processing_operations'")
            .fetch_optional(state.pool())
            .await
            .unwrap();
        assert!(processing_ops_result.is_some());
    }

    #[tokio::test]
    async fn test_invalid_database_path() {
        let config = DatabaseConfig {
            sqlite_path: "/invalid/path/db.sqlite".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        };

        let result = DaemonState::new(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_migrations_directly() {
        let config = create_test_db_config();
        let pool = SqlitePool::connect(&config.sqlite_path).await.unwrap();

        // Test migrations run successfully
        let result = DaemonState::run_migrations(&pool).await;
        assert!(result.is_ok());

        // Test running migrations again (should be idempotent)
        let result2 = DaemonState::run_migrations(&pool).await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_database_config_validation() {
        // Test with in-memory database
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 10,
            enable_wal: true,
        };

        let state = DaemonState::new(&config).await.unwrap();
        state.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_multiple_state_instances() {
        let config1 = create_test_db_config();
        let config2 = create_test_db_config();

        let state1 = DaemonState::new(&config1).await.unwrap();
        let state2 = DaemonState::new(&config2).await.unwrap();

        // Both should be functional
        state1.health_check().await.unwrap();
        state2.health_check().await.unwrap();
    }
}