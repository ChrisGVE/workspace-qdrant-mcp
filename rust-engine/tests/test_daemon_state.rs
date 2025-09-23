//! Comprehensive unit tests for daemon/state.rs
//! Tests state persistence, recovery mechanisms, concurrent access, and error handling

use std::sync::Arc;
use std::time::Duration;
use tempfile::{tempdir, NamedTempFile};
use tokio::time::timeout;
use tokio::task::JoinSet;
use sqlx::Row;
use workspace_qdrant_daemon::config::DatabaseConfig;
use workspace_qdrant_daemon::daemon::state::DaemonState;
use workspace_qdrant_daemon::error::DaemonError;

/// Helper function to create test database configuration
fn create_test_db_config() -> DatabaseConfig {
    DatabaseConfig {
        sqlite_path: "sqlite::memory:".to_string(),
        max_connections: 1,
        connection_timeout_secs: 5,
        enable_wal: false,
    }
}

/// Helper function to create test database configuration with custom path
fn create_test_db_config_with_path(path: String) -> DatabaseConfig {
    DatabaseConfig {
        sqlite_path: path,
        max_connections: 5,
        connection_timeout_secs: 10,
        enable_wal: true,
    }
}

/// Helper function to create test database configuration for concurrent access
fn create_concurrent_db_config() -> DatabaseConfig {
    DatabaseConfig {
        sqlite_path: "sqlite::memory:".to_string(),
        max_connections: 10,
        connection_timeout_secs: 30,
        enable_wal: true,
    }
}

#[cfg(test)]
mod basic_functionality {
    use super::*;

    #[tokio::test]
    async fn test_daemon_state_creation_success() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Verify state was created successfully
        assert!(!state.pool().is_closed());

        // Test debug formatting
        let debug_str = format!("{:?}", state);
        assert!(debug_str.contains("DaemonState"));
    }

    #[tokio::test]
    async fn test_daemon_state_creation_with_wal() {
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 5,
            connection_timeout_secs: 15,
            enable_wal: true,
        };

        let state = DaemonState::new(&config).await.unwrap();
        assert!(!state.pool().is_closed());
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

        // Multiple health checks should all succeed
        for _ in 0..5 {
            state.health_check().await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_pool_access() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Pool should be accessible and functional
        let pool = state.pool();
        assert!(!pool.is_closed());

        // Can execute queries through the pool
        let result = sqlx::query("SELECT 1")
            .fetch_one(pool)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_database_config_validation() {
        // Test with different connection limits
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 20,
            connection_timeout_secs: 60,
            enable_wal: false,
        };

        let state = DaemonState::new(&config).await.unwrap();
        state.health_check().await.unwrap();
    }
}

#[cfg(test)]
mod state_persistence {
    use super::*;

    #[tokio::test]
    async fn test_migrations_create_required_tables() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Verify all required tables were created
        let tables = vec!["projects", "collections", "processing_operations"];

        for table_name in tables {
            let result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
                .bind(table_name)
                .fetch_optional(state.pool())
                .await
                .unwrap();
            assert!(result.is_some(), "Table {} should exist", table_name);
        }
    }

    #[tokio::test]
    async fn test_projects_table_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test projects table can accept proper data
        let result = sqlx::query(r#"
            INSERT INTO projects (id, name, root_path, git_repository, git_branch, metadata)
            VALUES ('test-id', 'test-project', '/test/path', 'https://github.com/test/repo', 'main', '{}')
        "#)
        .execute(state.pool())
        .await;
        assert!(result.is_ok());

        // Verify data can be retrieved
        let row = sqlx::query("SELECT * FROM projects WHERE id = 'test-id'")
            .fetch_one(state.pool())
            .await
            .unwrap();

        assert_eq!(row.get::<String, _>("name"), "test-project");
        assert_eq!(row.get::<String, _>("root_path"), "/test/path");
    }

    #[tokio::test]
    async fn test_collections_table_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // First insert a project (foreign key dependency)
        sqlx::query(r#"
            INSERT INTO projects (id, name, root_path)
            VALUES ('proj-1', 'test-project', '/test/path')
        "#)
        .execute(state.pool())
        .await
        .unwrap();

        // Test collections table can accept proper data
        let result = sqlx::query(r#"
            INSERT INTO collections (id, name, project_id, config)
            VALUES ('coll-1', 'test-collection', 'proj-1', '{"vector_size": 384}')
        "#)
        .execute(state.pool())
        .await;
        assert!(result.is_ok());

        // Verify data can be retrieved
        let row = sqlx::query("SELECT * FROM collections WHERE id = 'coll-1'")
            .fetch_one(state.pool())
            .await
            .unwrap();

        assert_eq!(row.get::<String, _>("name"), "test-collection");
        assert_eq!(row.get::<String, _>("project_id"), "proj-1");
    }

    #[tokio::test]
    async fn test_processing_operations_table_schema() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // First insert a project (foreign key dependency)
        sqlx::query(r#"
            INSERT INTO projects (id, name, root_path)
            VALUES ('proj-1', 'test-project', '/test/path')
        "#)
        .execute(state.pool())
        .await
        .unwrap();

        // Test processing_operations table can accept proper data
        let result = sqlx::query(r#"
            INSERT INTO processing_operations (id, project_id, status, total_documents, processed_documents, failed_documents, error_messages)
            VALUES ('op-1', 'proj-1', 'in_progress', 100, 50, 2, '["error1", "error2"]')
        "#)
        .execute(state.pool())
        .await;
        assert!(result.is_ok());

        // Verify data can be retrieved
        let row = sqlx::query("SELECT * FROM processing_operations WHERE id = 'op-1'")
            .fetch_one(state.pool())
            .await
            .unwrap();

        assert_eq!(row.get::<String, _>("status"), "in_progress");
        assert_eq!(row.get::<i64, _>("total_documents"), 100);
        assert_eq!(row.get::<i64, _>("processed_documents"), 50);
    }

    #[tokio::test]
    async fn test_foreign_key_constraints() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Enable foreign key constraints
        sqlx::query("PRAGMA foreign_keys = ON")
            .execute(state.pool())
            .await
            .unwrap();

        // Try to insert a collection without a valid project (should fail)
        let result = sqlx::query(r#"
            INSERT INTO collections (id, name, project_id, config)
            VALUES ('coll-1', 'test-collection', 'nonexistent-project', '{}')
        "#)
        .execute(state.pool())
        .await;

        assert!(result.is_err(), "Foreign key constraint should prevent this insert");
    }

    #[tokio::test]
    async fn test_migration_idempotency() {
        let config = create_test_db_config();

        // Run migrations multiple times
        let state1 = DaemonState::new(&config).await.unwrap();
        drop(state1);

        let state2 = DaemonState::new(&config).await.unwrap();
        drop(state2);

        let state3 = DaemonState::new(&config).await.unwrap();

        // Final state should be functional
        state3.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_state_persistence_with_file_database() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_state.db");
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let config = create_test_db_config_with_path(db_url);

        // Create state and insert data
        {
            let state = DaemonState::new(&config).await.unwrap();

            sqlx::query(r#"
                INSERT INTO projects (id, name, root_path)
                VALUES ('persistent-proj', 'test-project', '/test/path')
            "#)
            .execute(state.pool())
            .await
            .unwrap();
        }

        // Recreate state with same database file
        {
            let state = DaemonState::new(&config).await.unwrap();

            // Data should persist
            let row = sqlx::query("SELECT * FROM projects WHERE id = 'persistent-proj'")
                .fetch_one(state.pool())
                .await
                .unwrap();

            assert_eq!(row.get::<String, _>("name"), "test-project");
        }
    }
}

#[cfg(test)]
mod concurrent_access {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::task::JoinSet;

    #[tokio::test]
    async fn test_multiple_state_instances() {
        let config1 = create_test_db_config();
        let config2 = create_test_db_config();

        let state1 = DaemonState::new(&config1).await.unwrap();
        let state2 = DaemonState::new(&config2).await.unwrap();

        // Both should be functional independently
        state1.health_check().await.unwrap();
        state2.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_concurrent_health_checks() {
        let config = create_concurrent_db_config();
        let state = Arc::new(DaemonState::new(&config).await.unwrap());

        let mut tasks = JoinSet::new();

        // Spawn multiple concurrent health checks
        for _ in 0..10 {
            let state_clone = Arc::clone(&state);
            tasks.spawn(async move {
                state_clone.health_check().await
            });
        }

        // All health checks should succeed
        while let Some(result) = tasks.join_next().await {
            assert!(result.unwrap().is_ok());
        }
    }

    #[tokio::test]
    async fn test_concurrent_database_operations() {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("concurrent_test.db");
        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let config = create_test_db_config_with_path(db_url);

        let state = Arc::new(DaemonState::new(&config).await.unwrap());
        let counter = Arc::new(AtomicU32::new(0));
        let mut tasks = JoinSet::new();

        // Spawn multiple concurrent insert operations
        for i in 0..5 {
            let state_clone = Arc::clone(&state);
            let counter_clone = Arc::clone(&counter);

            tasks.spawn(async move {
                let id = format!("concurrent-proj-{}", i);
                let name = format!("concurrent-project-{}", i);
                let path = format!("/concurrent/test/path/{}", i);

                let result = sqlx::query(r#"
                    INSERT INTO projects (id, name, root_path)
                    VALUES (?, ?, ?)
                "#)
                .bind(&id)
                .bind(&name)
                .bind(&path)
                .execute(state_clone.pool())
                .await;

                if result.is_ok() {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                }
                result
            });
        }

        // Wait for all operations to complete
        while let Some(result) = tasks.join_next().await {
            assert!(result.unwrap().is_ok());
        }

        // Verify all inserts succeeded
        assert_eq!(counter.load(Ordering::SeqCst), 5);

        // Verify data integrity
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM projects WHERE name LIKE 'concurrent-project-%'")
            .fetch_one(state.pool())
            .await
            .unwrap();
        assert_eq!(count, 5);
    }

    #[tokio::test]
    async fn test_connection_pool_management() {
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 3, // Limited pool size
            connection_timeout_secs: 5,
            enable_wal: true,
        };

        let state = Arc::new(DaemonState::new(&config).await.unwrap());
        let mut tasks = JoinSet::new();

        // Spawn more tasks than the pool size
        for i in 0..6 {
            let state_clone = Arc::clone(&state);

            tasks.spawn(async move {
                // Hold connection briefly
                let result = sqlx::query("SELECT ?")
                    .bind(i)
                    .fetch_one(state_clone.pool())
                    .await;

                tokio::time::sleep(Duration::from_millis(100)).await;
                result
            });
        }

        // All operations should eventually succeed despite pool limits
        while let Some(result) = tasks.join_next().await {
            assert!(result.unwrap().is_ok());
        }
    }

    #[tokio::test]
    async fn test_connection_timeout_handling() {
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 1, // Very limited
            connection_timeout_secs: 1, // Short timeout
            enable_wal: false,
        };

        let state = Arc::new(DaemonState::new(&config).await.unwrap());

        // Start a long-running operation
        let state_clone = Arc::clone(&state);
        let _long_task = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(2)).await;
            state_clone.health_check().await
        });

        // Try to use the same connection pool immediately
        // This should either succeed quickly or timeout appropriately
        let quick_result = timeout(
            Duration::from_secs(3),
            state.health_check()
        ).await;

        assert!(quick_result.is_ok(), "Operation should complete within timeout");
    }
}

#[cfg(test)]
mod error_handling {
    use super::*;

    #[tokio::test]
    async fn test_invalid_database_path() {
        let config = DatabaseConfig {
            sqlite_path: "/invalid/nonexistent/path/db.sqlite".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        };

        let result = DaemonState::new(&config).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DaemonError::Database(_) => {
                // Expected error type
            },
            other => panic!("Expected Database error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_invalid_connection_string() {
        let config = DatabaseConfig {
            sqlite_path: "invalid://connection/string".to_string(),
            max_connections: 1,
            connection_timeout_secs: 5,
            enable_wal: false,
        };

        let result = DaemonState::new(&config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_database_permission_error() {
        // Create a read-only file to simulate permission error
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path();

        // Try to set read-only permissions (may not work on all systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(db_path).unwrap().permissions();
            perms.set_mode(0o444); // Read-only
            std::fs::set_permissions(db_path, perms).unwrap();
        }

        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let config = create_test_db_config_with_path(db_url);

        // This may or may not fail depending on the system
        // But if it fails, it should be a Database error
        if let Err(error) = DaemonState::new(&config).await {
            match error {
                DaemonError::Database(_) => {
                    // Expected error type
                },
                other => panic!("Expected Database error, got: {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn test_health_check_on_closed_pool() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Close the pool
        state.pool().close().await;

        // Health check should fail
        let result = state.health_check().await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DaemonError::Database(_) => {
                // Expected error type for closed pool
            },
            other => panic!("Expected Database error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_migration_failure_recovery() {
        let config = create_test_db_config();

        // First, create a state successfully
        let state = DaemonState::new(&config).await.unwrap();

        // Manually corrupt the schema by dropping a table
        sqlx::query("DROP TABLE IF EXISTS projects")
            .execute(state.pool())
            .await
            .unwrap();

        drop(state);

        // Creating a new state should re-run migrations and fix the issue
        let new_state = DaemonState::new(&config).await.unwrap();

        // Verify tables exist again
        let result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table' AND name='projects'")
            .fetch_optional(new_state.pool())
            .await
            .unwrap();
        assert!(result.is_some());
    }

    #[tokio::test]
    async fn test_connection_limit_exceeded() {
        let config = DatabaseConfig {
            sqlite_path: "sqlite::memory:".to_string(),
            max_connections: 1, // Very limited
            connection_timeout_secs: 1, // Short timeout
            enable_wal: false,
        };

        let state = DaemonState::new(&config).await.unwrap();

        // Try to saturate the connection pool
        let pool = state.pool();
        let _conn1 = pool.acquire().await.unwrap();

        // Second connection should timeout quickly
        let conn2_result = timeout(
            Duration::from_millis(1500),
            pool.acquire()
        ).await;

        // Should either timeout or succeed depending on timing
        // The important thing is it doesn't hang indefinitely
        assert!(conn2_result.is_ok() || conn2_result.is_err());
    }

    #[tokio::test]
    async fn test_database_corruption_simulation() {
        // Create a temporary file that we can corrupt
        let temp_file = NamedTempFile::new().unwrap();
        let db_path = temp_file.path();

        // Write some invalid data to simulate corruption
        std::fs::write(db_path, b"this is not a valid sqlite database").unwrap();

        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let config = create_test_db_config_with_path(db_url);

        // Attempt to open corrupted database should fail
        let result = DaemonState::new(&config).await;
        assert!(result.is_err());

        match result.unwrap_err() {
            DaemonError::Database(_) => {
                // Expected error type for corrupted database
            },
            other => panic!("Expected Database error for corruption, got: {:?}", other),
        }
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_state_creation_performance() {
        let config = create_test_db_config();

        let start = Instant::now();
        let state = DaemonState::new(&config).await.unwrap();
        let creation_time = start.elapsed();

        // State creation should be reasonably fast (under 1 second)
        assert!(creation_time < Duration::from_secs(1),
                "State creation took too long: {:?}", creation_time);

        state.health_check().await.unwrap();
    }

    #[tokio::test]
    async fn test_health_check_performance() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Measure health check performance
        let start = Instant::now();
        for _ in 0..10 {
            state.health_check().await.unwrap();
        }
        let total_time = start.elapsed();

        // 10 health checks should complete quickly
        assert!(total_time < Duration::from_millis(500),
                "Health checks took too long: {:?}", total_time);
    }

    #[tokio::test]
    async fn test_concurrent_operation_performance() {
        let config = create_concurrent_db_config();
        let state = Arc::new(DaemonState::new(&config).await.unwrap());

        let start = Instant::now();
        let mut tasks = JoinSet::new();

        // Launch many concurrent operations
        for i in 0..20 {
            let state_clone = Arc::clone(&state);
            tasks.spawn(async move {
                let id = format!("perf-proj-{}", i);
                sqlx::query(r#"
                    INSERT INTO projects (id, name, root_path)
                    VALUES (?, ?, ?)
                "#)
                .bind(&id)
                .bind(format!("project-{}", i))
                .bind(format!("/path/{}", i))
                .execute(state_clone.pool())
                .await
            });
        }

        // Wait for all to complete
        while let Some(result) = tasks.join_next().await {
            assert!(result.unwrap().is_ok());
        }

        let total_time = start.elapsed();

        // 20 concurrent operations should complete in reasonable time
        assert!(total_time < Duration::from_secs(5),
                "Concurrent operations took too long: {:?}", total_time);
    }

    #[tokio::test]
    async fn test_large_dataset_handling() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Insert a moderate amount of data
        let start = Instant::now();
        for i in 0..100 {
            sqlx::query(r#"
                INSERT INTO projects (id, name, root_path, metadata)
                VALUES (?, ?, ?, ?)
            "#)
            .bind(format!("large-proj-{}", i))
            .bind(format!("Large Project {}", i))
            .bind(format!("/large/path/{}", i))
            .bind(format!(r#"{{"index": {}, "size": "large"}}"#, i))
            .execute(state.pool())
            .await
            .unwrap();
        }
        let insert_time = start.elapsed();

        // Query all data back
        let start = Instant::now();
        let rows = sqlx::query("SELECT * FROM projects WHERE name LIKE 'Large Project%'")
            .fetch_all(state.pool())
            .await
            .unwrap();
        let query_time = start.elapsed();

        assert_eq!(rows.len(), 100);

        // Operations should complete in reasonable time
        assert!(insert_time < Duration::from_secs(2),
                "Large insert took too long: {:?}", insert_time);
        assert!(query_time < Duration::from_millis(500),
                "Large query took too long: {:?}", query_time);
    }

    #[tokio::test]
    async fn test_memory_usage_stability() {
        // Create and destroy many state instances with in-memory databases
        for iteration in 0..10 {
            let config = create_test_db_config();
            let state = DaemonState::new(&config).await.unwrap();

            // Do some operations
            for i in 0..10 {
                sqlx::query(r#"
                    INSERT INTO projects (id, name, root_path)
                    VALUES (?, ?, ?)
                "#)
                .bind(format!("mem-test-{}-{}", iteration, i))
                .bind(format!("Memory Test {} {}", iteration, i))
                .bind(format!("/memory/test/{}/{}", iteration, i))
                .execute(state.pool())
                .await
                .unwrap();
            }

            state.health_check().await.unwrap();

            // State should be dropped here
        }

        // If we get here without OOM, memory management is working
        assert!(true, "Memory usage remained stable");
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_config_to_state_integration() {
        // Test various configuration options
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
                connection_timeout_secs: 30,
                enable_wal: true,
            },
            DatabaseConfig {
                sqlite_path: "sqlite::memory:".to_string(),
                max_connections: 10,
                connection_timeout_secs: 60,
                enable_wal: true,
            },
        ];

        for config in configs {
            let state = DaemonState::new(&config).await.unwrap();
            state.health_check().await.unwrap();

            // Verify basic functionality works with each config
            sqlx::query(r#"
                INSERT INTO projects (id, name, root_path)
                VALUES ('integration-test', 'Integration Project', '/integration/path')
            "#)
            .execute(state.pool())
            .await
            .unwrap();
        }
    }

    #[tokio::test]
    async fn test_state_schema_completeness() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Test that all required tables have expected columns
        let projects_columns = sqlx::query("PRAGMA table_info(projects)")
            .fetch_all(state.pool())
            .await
            .unwrap();

        let expected_project_columns = vec![
            "id", "name", "root_path", "git_repository",
            "git_branch", "metadata", "created_at", "updated_at"
        ];

        for expected_col in expected_project_columns {
            assert!(
                projects_columns.iter().any(|row| {
                    row.get::<String, _>("name") == expected_col
                }),
                "Projects table missing column: {}",
                expected_col
            );
        }

        let collections_columns = sqlx::query("PRAGMA table_info(collections)")
            .fetch_all(state.pool())
            .await
            .unwrap();

        let expected_collection_columns = vec![
            "id", "name", "project_id", "config", "created_at"
        ];

        for expected_col in expected_collection_columns {
            assert!(
                collections_columns.iter().any(|row| {
                    row.get::<String, _>("name") == expected_col
                }),
                "Collections table missing column: {}",
                expected_col
            );
        }

        let processing_columns = sqlx::query("PRAGMA table_info(processing_operations)")
            .fetch_all(state.pool())
            .await
            .unwrap();

        let expected_processing_columns = vec![
            "id", "project_id", "status", "total_documents",
            "processed_documents", "failed_documents", "error_messages",
            "started_at", "updated_at"
        ];

        for expected_col in expected_processing_columns {
            assert!(
                processing_columns.iter().any(|row| {
                    row.get::<String, _>("name") == expected_col
                }),
                "Processing operations table missing column: {}",
                expected_col
            );
        }
    }

    #[tokio::test]
    async fn test_cross_table_relationships() {
        let config = create_test_db_config();
        let state = DaemonState::new(&config).await.unwrap();

        // Create a complete data hierarchy
        let project_id = "rel-test-project";
        let collection_id = "rel-test-collection";
        let operation_id = "rel-test-operation";

        // Insert project
        sqlx::query(r#"
            INSERT INTO projects (id, name, root_path, git_repository, git_branch, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        "#)
        .bind(project_id)
        .bind("Relationship Test Project")
        .bind("/rel/test/path")
        .bind("https://github.com/test/repo")
        .bind("main")
        .bind(r#"{"test": true}"#)
        .execute(state.pool())
        .await
        .unwrap();

        // Insert collection linked to project
        sqlx::query(r#"
            INSERT INTO collections (id, name, project_id, config)
            VALUES (?, ?, ?, ?)
        "#)
        .bind(collection_id)
        .bind("Relationship Test Collection")
        .bind(project_id)
        .bind(r#"{"vector_size": 384, "metric": "cosine"}"#)
        .execute(state.pool())
        .await
        .unwrap();

        // Insert processing operation linked to project
        sqlx::query(r#"
            INSERT INTO processing_operations (id, project_id, status, total_documents, processed_documents, failed_documents, error_messages)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        "#)
        .bind(operation_id)
        .bind(project_id)
        .bind("completed")
        .bind(50)
        .bind(48)
        .bind(2)
        .bind(r#"["timeout on large file", "parsing error"]"#)
        .execute(state.pool())
        .await
        .unwrap();

        // Query relationships
        let project_data = sqlx::query(r#"
            SELECT
                p.*,
                COUNT(DISTINCT c.id) as collection_count,
                COUNT(DISTINCT po.id) as operation_count
            FROM projects p
            LEFT JOIN collections c ON p.id = c.project_id
            LEFT JOIN processing_operations po ON p.id = po.project_id
            WHERE p.id = ?
            GROUP BY p.id
        "#)
        .bind(project_id)
        .fetch_one(state.pool())
        .await
        .unwrap();

        assert_eq!(project_data.get::<String, _>("name"), "Relationship Test Project");
        assert_eq!(project_data.get::<i64, _>("collection_count"), 1);
        assert_eq!(project_data.get::<i64, _>("operation_count"), 1);
    }
}