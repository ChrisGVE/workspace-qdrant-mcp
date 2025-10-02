//! Priority Manager Module
//!
//! Manages dynamic priority adjustments for queue items based on server lifecycle events.
//! When an MCP server starts for a project, related files are bumped to high priority (1).
//! When a server stops, related files are demoted to normal priority (3).

use sqlx::SqlitePool;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Priority management errors
#[derive(Error, Debug)]
pub enum PriorityError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Invalid priority value: {0}")]
    InvalidPriority(i32),

    #[error("Empty tenant_id or branch")]
    EmptyParameter,
}

/// Result type for priority operations
pub type PriorityResult<T> = Result<T, PriorityError>;

/// Priority transition information
#[derive(Debug, Clone)]
pub struct PriorityTransition {
    /// Priority before transition
    pub from_priority: u8,

    /// Priority after transition
    pub to_priority: u8,

    /// Number of items affected in ingestion_queue
    pub ingestion_queue_affected: usize,

    /// Number of items affected in missing_metadata_queue
    pub missing_metadata_queue_affected: usize,

    /// Total items affected across all queues
    pub total_affected: usize,
}

impl PriorityTransition {
    /// Create a new transition record
    pub fn new(from_priority: u8, to_priority: u8) -> Self {
        Self {
            from_priority,
            to_priority,
            ingestion_queue_affected: 0,
            missing_metadata_queue_affected: 0,
            total_affected: 0,
        }
    }

    /// Add counts from queue updates
    pub fn add_counts(&mut self, ingestion: usize, missing_metadata: usize) {
        self.ingestion_queue_affected = ingestion;
        self.missing_metadata_queue_affected = missing_metadata;
        self.total_affected = ingestion + missing_metadata;
    }
}

/// Priority Manager for server lifecycle-driven priority adjustments
#[derive(Clone)]
pub struct PriorityManager {
    db_pool: SqlitePool,
}

impl PriorityManager {
    /// Create a new PriorityManager with existing database pool
    pub fn new(db_pool: SqlitePool) -> Self {
        Self { db_pool }
    }

    /// Handle server start event - bump priority from 3 to 1 (urgent)
    ///
    /// When an MCP server starts for a project, all pending items for that project
    /// should be processed urgently to provide fresh context.
    ///
    /// # Arguments
    /// * `tenant_id` - Project/tenant identifier
    /// * `branch` - Git branch name
    ///
    /// # Returns
    /// PriorityTransition with count of affected items
    pub async fn on_server_start(
        &self,
        tenant_id: &str,
        branch: &str,
    ) -> PriorityResult<PriorityTransition> {
        // Validate inputs
        if tenant_id.is_empty() || branch.is_empty() {
            warn!(
                "Empty tenant_id or branch in on_server_start: tenant_id='{}', branch='{}'",
                tenant_id, branch
            );
            return Err(PriorityError::EmptyParameter);
        }

        const FROM_PRIORITY: u8 = 3; // Normal priority
        const TO_PRIORITY: u8 = 1; // Urgent priority

        info!(
            "Server START: Bumping priority {} → {} for tenant_id='{}', branch='{}'",
            FROM_PRIORITY, TO_PRIORITY, tenant_id, branch
        );

        self.bulk_update_priority(tenant_id, branch, FROM_PRIORITY, TO_PRIORITY)
            .await
    }

    /// Handle server stop event - demote priority from 1 to 3 (normal)
    ///
    /// When an MCP server stops, pending items can be deprioritized as they're
    /// no longer urgently needed for active development.
    ///
    /// # Arguments
    /// * `tenant_id` - Project/tenant identifier
    /// * `branch` - Git branch name
    ///
    /// # Returns
    /// PriorityTransition with count of affected items
    pub async fn on_server_stop(
        &self,
        tenant_id: &str,
        branch: &str,
    ) -> PriorityResult<PriorityTransition> {
        // Validate inputs
        if tenant_id.is_empty() || branch.is_empty() {
            warn!(
                "Empty tenant_id or branch in on_server_stop: tenant_id='{}', branch='{}'",
                tenant_id, branch
            );
            return Err(PriorityError::EmptyParameter);
        }

        const FROM_PRIORITY: u8 = 1; // Urgent priority
        const TO_PRIORITY: u8 = 3; // Normal priority

        info!(
            "Server STOP: Demoting priority {} → {} for tenant_id='{}', branch='{}'",
            FROM_PRIORITY, TO_PRIORITY, tenant_id, branch
        );

        self.bulk_update_priority(tenant_id, branch, FROM_PRIORITY, TO_PRIORITY)
            .await
    }

    /// Bulk update priorities for all matching queue items
    ///
    /// Updates both ingestion_queue and missing_metadata_queue in a single transaction.
    /// Only items with the exact from_priority are affected, preventing unintended changes.
    ///
    /// # Arguments
    /// * `tenant_id` - Filter by tenant/project
    /// * `branch` - Filter by git branch
    /// * `from_priority` - Only update items with this priority
    /// * `to_priority` - Set priority to this value
    ///
    /// # Returns
    /// PriorityTransition with counts of affected items
    async fn bulk_update_priority(
        &self,
        tenant_id: &str,
        branch: &str,
        from_priority: u8,
        to_priority: u8,
    ) -> PriorityResult<PriorityTransition> {
        // Validate priority range
        if from_priority > 10 {
            return Err(PriorityError::InvalidPriority(from_priority as i32));
        }
        if to_priority > 10 {
            return Err(PriorityError::InvalidPriority(to_priority as i32));
        }

        // Start transaction for atomic updates
        let mut tx = self.db_pool.begin().await?;

        // Update ingestion_queue
        let ingestion_query = r#"
            UPDATE ingestion_queue
            SET priority = ?1
            WHERE tenant_id = ?2
              AND branch = ?3
              AND priority = ?4
        "#;

        let ingestion_result = sqlx::query(ingestion_query)
            .bind(to_priority as i32)
            .bind(tenant_id)
            .bind(branch)
            .bind(from_priority as i32)
            .execute(&mut *tx)
            .await?;

        let ingestion_affected = ingestion_result.rows_affected() as usize;

        // Update missing_metadata_queue
        let missing_query = r#"
            UPDATE missing_metadata_queue
            SET priority = ?1
            WHERE tenant_id = ?2
              AND branch = ?3
              AND priority = ?4
        "#;

        let missing_result = sqlx::query(missing_query)
            .bind(to_priority as i32)
            .bind(tenant_id)
            .bind(branch)
            .bind(from_priority as i32)
            .execute(&mut *tx)
            .await?;

        let missing_affected = missing_result.rows_affected() as usize;

        // Commit transaction
        tx.commit().await?;

        // Create transition record
        let mut transition = PriorityTransition::new(from_priority, to_priority);
        transition.add_counts(ingestion_affected, missing_affected);

        // Log results
        if transition.total_affected == 0 {
            debug!(
                "No items found with priority {} for tenant_id='{}', branch='{}'",
                from_priority, tenant_id, branch
            );
        } else {
            info!(
                "Priority transition complete: {} items updated (ingestion: {}, missing_metadata: {})",
                transition.total_affected,
                transition.ingestion_queue_affected,
                transition.missing_metadata_queue_affected
            );

            // Warn if unusually large batch
            if transition.total_affected > 1000 {
                warn!(
                    "Large priority update: {} items affected for tenant_id='{}', branch='{}'",
                    transition.total_affected, tenant_id, branch
                );
            }
        }

        Ok(transition)
    }

    /// Get count of items with specific priority for a tenant/branch
    ///
    /// Utility method for testing and monitoring.
    pub async fn count_items_with_priority(
        &self,
        tenant_id: &str,
        branch: &str,
        priority: u8,
    ) -> PriorityResult<usize> {
        if priority > 10 {
            return Err(PriorityError::InvalidPriority(priority as i32));
        }

        let query = r#"
            SELECT
                (SELECT COUNT(*) FROM ingestion_queue
                 WHERE tenant_id = ?1 AND branch = ?2 AND priority = ?3) +
                (SELECT COUNT(*) FROM missing_metadata_queue
                 WHERE tenant_id = ?1 AND branch = ?2 AND priority = ?3) as total
        "#;

        let count: i64 = sqlx::query_scalar(query)
            .bind(tenant_id)
            .bind(branch)
            .bind(priority as i32)
            .fetch_one(&self.db_pool)
            .await?;

        Ok(count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::queue_config::QueueConnectionConfig;
    use crate::queue_operations::{QueueManager, QueueOperation};
    use tempfile::tempdir;

    /// Helper to create test database with schema
    async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_priority.db");

        let config = QueueConnectionConfig::with_database_path(&db_path);
        let pool = config.create_pool().await.unwrap();

        // Initialize schema
        sqlx::query(include_str!("../../../../python/common/core/queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(include_str!("../../../../python/common/core/missing_metadata_queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    #[tokio::test]
    async fn test_server_start_bumps_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items with normal priority (3)
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        queue_manager
            .enqueue_file(
                "/test/file2.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Trigger server start - should bump priority to 1
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 3);
        assert_eq!(transition.to_priority, 1);
        assert_eq!(transition.ingestion_queue_affected, 2);
        assert_eq!(transition.total_affected, 2);

        // Verify items now have priority 1
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 1)
            .await
            .unwrap();
        assert_eq!(count, 2);

        // Verify no items remain with priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 3)
            .await
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test]
    async fn test_server_stop_demotes_priority() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items with urgent priority (1)
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                1,
                None,
            )
            .await
            .unwrap();

        // Trigger server stop - should demote priority to 3
        let transition = priority_manager
            .on_server_stop("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.from_priority, 1);
        assert_eq!(transition.to_priority, 3);
        assert_eq!(transition.ingestion_queue_affected, 1);
        assert_eq!(transition.total_affected, 1);

        // Verify item now has priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 3)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_branch_isolation() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue items on different branches
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        queue_manager
            .enqueue_file(
                "/test/file2.rs",
                "test-collection",
                "test-tenant",
                "feature-branch",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Trigger server start for main branch only
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.total_affected, 1);

        // Verify main branch item has priority 1
        let count = priority_manager
            .count_items_with_priority("test-tenant", "main", 1)
            .await
            .unwrap();
        assert_eq!(count, 1);

        // Verify feature branch item still has priority 3
        let count = priority_manager
            .count_items_with_priority("test-tenant", "feature-branch", 3)
            .await
            .unwrap();
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn test_empty_parameters_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // Empty tenant_id
        let result = priority_manager.on_server_start("", "main").await;
        assert!(matches!(result, Err(PriorityError::EmptyParameter)));

        // Empty branch
        let result = priority_manager.on_server_start("test-tenant", "").await;
        assert!(matches!(result, Err(PriorityError::EmptyParameter)));
    }

    #[tokio::test]
    async fn test_no_matching_items() {
        let (pool, _temp_dir) = setup_test_db().await;
        let priority_manager = PriorityManager::new(pool);

        // No items in queue - should succeed with 0 affected
        let transition = priority_manager
            .on_server_start("nonexistent-tenant", "main")
            .await
            .unwrap();

        assert_eq!(transition.total_affected, 0);
    }

    #[tokio::test]
    async fn test_transaction_rollback_on_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool.clone());

        // Enqueue an item
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Close the pool to force an error
        pool.close().await;

        // Attempt priority update - should fail
        let result = priority_manager
            .on_server_start("test-tenant", "main")
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_missing_metadata_queue_update() {
        let (pool, _temp_dir) = setup_test_db().await;
        let queue_manager = QueueManager::new(pool.clone());
        let priority_manager = PriorityManager::new(pool);

        // Enqueue item in ingestion_queue
        queue_manager
            .enqueue_file(
                "/test/file1.rs",
                "test-collection",
                "test-tenant",
                "main",
                QueueOperation::Ingest,
                3,
                None,
            )
            .await
            .unwrap();

        // Manually add item to missing_metadata_queue with priority 3
        let insert_query = r#"
            INSERT INTO missing_metadata_queue (
                queue_id, file_absolute_path, collection_name, tenant_id, branch,
                operation, priority, missing_tools, queued_timestamp
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        "#;

        sqlx::query(insert_query)
            .bind("test-queue-id")
            .bind("/test/file2.rs")
            .bind("test-collection")
            .bind("test-tenant")
            .bind("main")
            .bind("ingest")
            .bind(3)
            .bind(r#"[{"LspServer": {"language": "rust"}}]"#)
            .bind("2024-01-01T00:00:00Z")
            .execute(&priority_manager.db_pool)
            .await
            .unwrap();

        // Trigger server start
        let transition = priority_manager
            .on_server_start("test-tenant", "main")
            .await
            .unwrap();

        // Should affect both queues
        assert_eq!(transition.ingestion_queue_affected, 1);
        assert_eq!(transition.missing_metadata_queue_affected, 1);
        assert_eq!(transition.total_affected, 2);
    }
}
