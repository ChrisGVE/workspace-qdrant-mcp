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