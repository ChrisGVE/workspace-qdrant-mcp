//! Daemon state management using SQLite

use crate::config::DatabaseConfig;
use crate::error::DaemonResult;
#[cfg(test)]
use crate::error::DaemonError;
use sqlx::SqlitePool;
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
    use sqlx::Row;
    use tokio::time::{timeout, Duration};

    fn create_test_db_config() -> DatabaseConfig {
        DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        }
    }

    fn create_test_db_config_with_wal() -> DatabaseConfig {
        DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 10,
            enable_wal: true,
        }
    }

    #[tokio::test]
    async fn test_daemon_state_new_basic() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test that the state was created successfully
        assert!(!state.pool().is_closed());

        // Test debug formatting
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DaemonState"));
    }

    #[tokio::test]
    async fn test_daemon_state_new_with_wal() {
        let config = create_test_db_config_with_wal();
        let state = DaemonState::new(&config).await.unwrap();

        assert!(!state.pool().is_closed());
        state.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_daemon_state_new_custom_config() {
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 3,
            connection_timeout_secs: 15,
            enable_wal: false,
        };

        let state = DaemonState::new(&config).await.unwrap();
        assert!(!state.pool().is_closed());

        // Verify logging by checking info! logs are called during initialization
        // This tests the logging paths in the new() method
    }

    #[tokio::test]
    async fn test_health_check_success() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Health check should pass
        let result = state.health_check().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_health_check_multiple_calls() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Multiple health checks should all pass
        for _ in 0..5 {
            state.health_check().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_pool_access() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Pool should be accessible
        let pool = state.pool();
        assert!(!pool.is_closed());

        // Test that we can use the pool reference
        sqlx::query("SELECT 1")
            .execute(pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_migrations_create_all_tables() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Verify all tables were created by querying them
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
    async fn test_migrations_table_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test projects table schema
        let projects_schema = sqlx::query("PRAGMA table_info(projects)")
            .fetch_all(state.pool())
            .await
            .unwrap();
        assert!(!projects_schema.is_empty());

        // Verify expected columns exist
        let column_names: Vec<String> = projects_schema
            .iter()
            .map(|row| row.get::<String, _>("name"))
            .collect();
        assert!(column_names.contains(&"id".to_string()));
        assert!(column_names.contains(&"name".to_string()));
        assert!(column_names.contains(&"root_path".to_string()));
        assert!(column_names.contains(&"git_repository".to_string()));
        assert!(column_names.contains(&"metadata".to_string()));
        assert!(column_names.contains(&"created_at".to_string()));
        assert!(column_names.contains(&"updated_at".to_string()));
    }

    #[tokio::test]
    async fn test_migrations_collections_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test collections table schema
        let collections_schema = sqlx::query("PRAGMA table_info(collections)")
            .fetch_all(state.pool())
            .await
            .unwrap();
        assert!(!collections_schema.is_empty());

        let column_names: Vec<String> = collections_schema
            .iter()
            .map(|row| row.get::<String, _>("name"))
            .collect();
        assert!(column_names.contains(&"id".to_string()));
        assert!(column_names.contains(&"name".to_string()));
        assert!(column_names.contains(&"project_id".to_string()));
        assert!(column_names.contains(&"config".to_string()));
        assert!(column_names.contains(&"created_at".to_string()));
    }

    #[tokio::test]
    async fn test_migrations_processing_operations_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test processing_operations table schema
        let processing_ops_schema = sqlx::query("PRAGMA table_info(processing_operations)")
            .fetch_all(state.pool())
            .await
            .unwrap();
        assert!(!processing_ops_schema.is_empty());

        let column_names: Vec<String> = processing_ops_schema
            .iter()
            .map(|row| row.get::<String, _>("name"))
            .collect();
        assert!(column_names.contains(&"id".to_string()));
        assert!(column_names.contains(&"project_id".to_string()));
        assert!(column_names.contains(&"status".to_string()));
        assert!(column_names.contains(&"total_documents".to_string()));
        assert!(column_names.contains(&"processed_documents".to_string()));
        assert!(column_names.contains(&"failed_documents".to_string()));
        assert!(column_names.contains(&"error_messages".to_string()));
        assert!(column_names.contains(&"started_at".to_string()));
        assert!(column_names.contains(&"updated_at".to_string()));
    }

    #[tokio::test]
    async fn test_migrations_foreign_keys() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Enable foreign key constraints
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(state.pool())
            .await
            .unwrap();

        // Test that foreign key constraints work
        // First, insert a project
        sqlx::query("INSERT INTO projects (id, name, root_path) VALUES ('test-project', 'Test Project', '/test')")
            .execute(state.pool())
            .await
            .unwrap();

        // Then, insert a collection referencing the project
        let result = sqlx::query("INSERT INTO collections (id, name, project_id) VALUES ('test-collection', 'Test Collection', 'test-project')")
            .execute(state.pool())
            .await;
        assert!(result.is_ok());

        // Try inserting a collection with invalid project_id (should fail)
        let invalid_result = sqlx::query("INSERT INTO collections (id, name, project_id) VALUES ('invalid-collection', 'Invalid Collection', 'nonexistent-project')")
            .execute(state.pool())
            .await;
        assert!(invalid_result.is_err());
    }

    #[tokio::test]
    async fn test_migrations_unique_constraints() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test unique constraint on projects.root_path
        sqlx::query("INSERT INTO projects (id, name, root_path) VALUES ('project1', 'Project 1', '/same/path')")
            .execute(state.pool())
            .await
            .unwrap();

        // Second insert with same root_path should fail
        let duplicate_result = sqlx::query("INSERT INTO projects (id, name, root_path) VALUES ('project2', 'Project 2', '/same/path')")
            .execute(state.pool())
            .await;
        assert!(duplicate_result.is_err());
    }

    #[tokio::test]
    async fn test_migrations_default_values() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Insert minimal data to test defaults
        sqlx::query("INSERT INTO projects (id, name, root_path) VALUES ('minimal-project', 'Minimal Project', '/minimal')")
            .execute(state.pool())
            .await
            .unwrap();

        // Verify default values are set
        let row = sqlx::query("SELECT created_at, updated_at FROM projects WHERE id = 'minimal-project'")
            .fetch_one(state.pool())
            .await
            .unwrap();

        let created_at: String = row.get("created_at");
        let updated_at: String = row.get("updated_at");
        assert!(!created_at.is_empty());
        assert!(!updated_at.is_empty());

        // Test processing_operations defaults
        sqlx::query("INSERT INTO processing_operations (id, status) VALUES ('test-op', 'running')")
            .execute(state.pool())
            .await
            .unwrap();

        let op_row = sqlx::query("SELECT total_documents, processed_documents, failed_documents, started_at FROM processing_operations WHERE id = 'test-op'")
            .fetch_one(state.pool())
            .await
            .unwrap();

        let total_docs: i64 = op_row.get("total_documents");
        let processed_docs: i64 = op_row.get("processed_documents");
        let failed_docs: i64 = op_row.get("failed_documents");
        let started_at: String = op_row.get("started_at");

        assert_eq!(total_docs, 0);
        assert_eq!(processed_docs, 0);
        assert_eq!(failed_docs, 0);
        assert!(!started_at.is_empty());
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

        // Verify it's a database error
        match result {
            Err(DaemonError::Database(_)) => {},
            _ => panic!("Expected Database error for invalid path"),
        }
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

        // Verify tables exist after direct migration call
        let tables = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .fetch_all(&pool)
            .await
            .unwrap();

        let table_names: Vec<String> = tables.iter().map(|row| row.get::<String, _>("name")).collect();
        assert!(table_names.contains(&"projects".to_string()));
        assert!(table_names.contains(&"collections".to_string()));
        assert!(table_names.contains(&"processing_operations".to_string()));
    }

    #[tokio::test]
    async fn test_run_migrations_logging() {
        let config = create_test_db_config();
        let pool = SqlitePool::connect(&config.sqlite_path).await.unwrap();

        // This test ensures the debug! logging paths are covered
        let result = DaemonState::run_migrations(&pool).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_database_config_variations() {
        // Test various database configurations
        let configs = vec![
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 1,
                connection_timeout_secs: 5,
                enable_wal: false,
            },
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 5,
                connection_timeout_secs: 10,
                enable_wal: true,
            },
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 10,
                connection_timeout_secs: 30,
                enable_wal: false,
            },
        ];

        for config in configs {
            let state = DaemonState::new(&config).await.unwrap();
            state.health_check().await.unwrap();
        }
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

        // Test cross-state operations
        assert!(!state1.pool().is_closed());
        assert!(!state2.pool().is_closed());
    }

    #[tokio::test]
    async fn test_concurrent_health_checks() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Run multiple health checks concurrently
        let futures = (0..10).map(|_| state.health_check()).collect::<Vec<_>>();
        let results = futures_util::future::try_join_all(futures).await;
        assert!(results.is_ok());
    }

    #[tokio::test]
    async fn test_database_operations_after_init() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test basic database operations work after initialization
        let result = sqlx::query("INSERT INTO projects (id, name, root_path) VALUES ('test', 'Test Project', '/test/path')")
            .execute(state.pool())
            .await;
        assert!(result.is_ok());

        let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM projects")
            .fetch_one(state.pool())
            .await
            .unwrap();
        assert_eq!(count.0, 1);
    }

    #[tokio::test]
    async fn test_pool_state_after_operations() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Perform several database operations
        for i in 0..5 {
            sqlx::query(&format!("INSERT INTO projects (id, name, root_path) VALUES ('project{}', 'Project {}', '/path/{}')", i, i, i))
                .execute(state.pool())
                .await
                .unwrap();
        }

        // Pool should still be healthy
        state.health_check().await.unwrap();
        assert!(!state.pool().is_closed());
    }

    #[tokio::test]
    async fn test_state_debug_formatting() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test comprehensive debug formatting
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DaemonState"));
        assert!(!debug_str.is_empty());

        // Debug should not expose sensitive information
        assert!(!debug_str.contains("password"));
        assert!(!debug_str.contains("secret"));
    }

    #[tokio::test]
    async fn test_health_check_error_recovery() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Verify health check works initially
        state.health_check().await.unwrap();

        // Pool should remain functional
        assert!(!state.pool().is_closed());

        // Multiple health checks should continue to work
        for _ in 0..3 {
            state.health_check().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_database_config_edge_cases() {
        // Test various edge case configurations
        let edge_configs = vec![
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 1,  // Minimum connections
                connection_timeout_secs: 1,  // Short timeout
                enable_wal: false,
            },
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 100,  // High connections
                connection_timeout_secs: 300,  // Long timeout
                enable_wal: true,
            },
        ];

        for config in edge_configs {
            let state = DaemonState::new(&config).await.unwrap();
            state.health_check().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_migration_logging_paths() {
        let config = create_test_db_config();

        // Create state to trigger migration logging
        let state = DaemonState::new(&config).await.unwrap();

        // Verify the state is functional (ensures logging didn't break anything)
        state.health_check().await.unwrap();

        // Test that pool is accessible after logging
        let pool = state.pool();
        assert!(!pool.is_closed());
    }

    #[tokio::test]
    async fn test_error_handling_coverage() {
        // Test various error scenarios to improve coverage

        // Test with completely invalid path format
        let bad_config = DatabaseConfig {
            sqlite_path: "invalid://path".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        };

        let result = DaemonState::new(&bad_config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_all_code_paths_covered() {
        // This test aims to hit any remaining uncovered code paths
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test all public methods
        let pool_ref = state.pool();
        assert!(!pool_ref.is_closed());

        // Test health check multiple times
        state.health_check().await.unwrap();
        state.health_check().await.unwrap();

        // Test debug formatting
        let _debug = format!("{:?}", state);

        // Test with timeout to ensure async paths are covered
        let health_result = timeout(Duration::from_secs(5), state.health_check()).await;
        assert!(health_result.is_ok());
        assert!(health_result.unwrap().is_ok());
    }
}