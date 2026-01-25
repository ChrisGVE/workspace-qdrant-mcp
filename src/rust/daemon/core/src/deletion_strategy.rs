//! Deletion Strategy Module
//!
//! Implements type-specific deletion handling for collection types with different
//! deletion modes: Dynamic (immediate) vs Cumulative (mark + batch cleanup).
//!
//! Collection Type Mapping:
//! - SYSTEM (__ prefix): Cumulative deletion
//! - LIBRARY (_ prefix): Cumulative deletion
//! - PROJECT ({project_id}-{suffix}): Dynamic deletion
//! - GLOBAL (fixed names): Dynamic deletion

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::queue_operations::QueueItem;
use crate::storage::StorageClient;

/// Deletion mode enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeletionMode {
    /// Immediate deletion from Qdrant (PROJECT, GLOBAL collections)
    Dynamic,
    /// Mark as deleted, batch cleanup later (SYSTEM, LIBRARY collections)
    Cumulative,
}

/// Collection type for deletion handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeletionCollectionType {
    /// System collections (__ prefix)
    System,
    /// Library collections (_ prefix)
    Library,
    /// Project collections ({project_id}-{suffix})
    Project,
    /// Global collections (fixed names)
    Global,
}

impl DeletionCollectionType {
    /// Detect collection type from collection name
    pub fn from_name(collection_name: &str) -> Self {
        if collection_name.starts_with("__") {
            DeletionCollectionType::System
        } else if collection_name.starts_with('_') {
            DeletionCollectionType::Library
        } else if Self::is_global_collection(collection_name) {
            DeletionCollectionType::Global
        } else {
            // Default to Project for any other pattern
            DeletionCollectionType::Project
        }
    }

    /// Check if collection name matches known global collections
    fn is_global_collection(name: &str) -> bool {
        matches!(
            name,
            "algorithms"
                | "codebase"
                | "context"
                | "documents"
                | "knowledge"
                | "memory"
                | "projects"
                | "workspace"
        )
    }

    /// Get deletion mode for this collection type
    pub fn deletion_mode(&self) -> DeletionMode {
        match self {
            DeletionCollectionType::System | DeletionCollectionType::Library => {
                DeletionMode::Cumulative
            }
            DeletionCollectionType::Project | DeletionCollectionType::Global => {
                DeletionMode::Dynamic
            }
        }
    }
}

/// Deletion strategy errors
#[derive(Error, Debug)]
pub enum DeletionError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Deletion failed: {0}")]
    DeletionFailed(String),

    #[error("Batch cleanup failed: {0}")]
    BatchCleanupFailed(String),
}

/// Result type for deletion operations
pub type DeletionResult<T> = Result<T, DeletionError>;

/// Deletion strategy trait
#[async_trait]
pub trait DeletionStrategy: Send + Sync {
    /// Execute the deletion operation
    async fn execute(
        &self,
        item: &QueueItem,
        storage_client: &Arc<StorageClient>,
        pool: &SqlitePool,
    ) -> DeletionResult<()>;

    /// Get the deletion mode
    fn mode(&self) -> DeletionMode;
}

/// Dynamic deletion strategy - immediate deletion from Qdrant
pub struct DynamicDeletionStrategy;

impl DynamicDeletionStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DynamicDeletionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeletionStrategy for DynamicDeletionStrategy {
    async fn execute(
        &self,
        item: &QueueItem,
        storage_client: &Arc<StorageClient>,
        _pool: &SqlitePool,
    ) -> DeletionResult<()> {
        info!(
            "Dynamic deletion: {} from collection {}",
            item.file_absolute_path, item.collection_name
        );

        // Check if collection exists
        let collection_exists = storage_client
            .collection_exists(&item.collection_name)
            .await
            .map_err(|e| DeletionError::Storage(e.to_string()))?;

        if !collection_exists {
            warn!(
                "Collection {} does not exist, skipping delete",
                item.collection_name
            );
            return Ok(());
        }

        // Delete all points matching the file_path
        storage_client
            .delete_points_by_filter(&item.collection_name, &item.file_absolute_path)
            .await
            .map_err(|e| DeletionError::Storage(e.to_string()))?;

        info!(
            "Successfully executed dynamic deletion for {} from {}",
            item.file_absolute_path, item.collection_name
        );

        Ok(())
    }

    fn mode(&self) -> DeletionMode {
        DeletionMode::Dynamic
    }
}

/// Cumulative deletion strategy - mark as deleted, batch cleanup later
pub struct CumulativeDeletionStrategy;

impl CumulativeDeletionStrategy {
    pub fn new() -> Self {
        Self
    }

    /// Table name for tracking cumulative deletions
    const CUMULATIVE_DELETIONS_TABLE: &'static str = "cumulative_deletions_queue";

    /// Initialize the cumulative deletions table
    pub async fn init_table(pool: &SqlitePool) -> DeletionResult<()> {
        let create_table_query = r#"
            CREATE TABLE IF NOT EXISTS cumulative_deletions_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_absolute_path TEXT NOT NULL,
                collection_name TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                branch TEXT NOT NULL,
                marked_for_deletion_at TEXT NOT NULL,
                deletion_scheduled_at TEXT,
                deleted_at TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                retry_count INTEGER NOT NULL DEFAULT 0,
                error_message TEXT,
                UNIQUE(file_absolute_path, collection_name, tenant_id, branch)
            )
        "#;

        sqlx::query(create_table_query)
            .execute(pool)
            .await?;

        // Create index for efficient queries
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_cumulative_deletions_status
             ON cumulative_deletions_queue(status)"
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_cumulative_deletions_scheduled
             ON cumulative_deletions_queue(deletion_scheduled_at)"
        )
        .execute(pool)
        .await?;

        debug!("Cumulative deletions table initialized");
        Ok(())
    }
}

impl Default for CumulativeDeletionStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeletionStrategy for CumulativeDeletionStrategy {
    async fn execute(
        &self,
        item: &QueueItem,
        _storage_client: &Arc<StorageClient>,
        pool: &SqlitePool,
    ) -> DeletionResult<()> {
        info!(
            "Cumulative deletion: marking {} from collection {} for batch cleanup",
            item.file_absolute_path, item.collection_name
        );

        let now = Utc::now().to_rfc3339();

        // Insert or update cumulative deletion record
        let insert_query = format!(
            r#"
            INSERT INTO {}
                (file_absolute_path, collection_name, tenant_id, branch, marked_for_deletion_at, status)
            VALUES (?, ?, ?, ?, ?, 'pending')
            ON CONFLICT(file_absolute_path, collection_name, tenant_id, branch)
            DO UPDATE SET
                marked_for_deletion_at = ?,
                status = 'pending',
                retry_count = 0,
                error_message = NULL
            "#,
            Self::CUMULATIVE_DELETIONS_TABLE
        );

        sqlx::query(&insert_query)
            .bind(&item.file_absolute_path)
            .bind(&item.collection_name)
            .bind(&item.tenant_id)
            .bind(&item.branch)
            .bind(&now)
            .bind(&now)
            .execute(pool)
            .await?;

        info!(
            "Successfully marked {} from {} for cumulative deletion",
            item.file_absolute_path, item.collection_name
        );

        Ok(())
    }

    fn mode(&self) -> DeletionMode {
        DeletionMode::Cumulative
    }
}

/// Deletion strategy factory
pub struct DeletionStrategyFactory;

impl DeletionStrategyFactory {
    /// Create appropriate deletion strategy for the given collection
    pub fn create_strategy(
        collection_name: &str,
    ) -> Box<dyn DeletionStrategy> {
        let collection_type = DeletionCollectionType::from_name(collection_name);
        let deletion_mode = collection_type.deletion_mode();

        debug!(
            "Creating deletion strategy for collection '{}': type={:?}, mode={:?}",
            collection_name, collection_type, deletion_mode
        );

        match deletion_mode {
            DeletionMode::Dynamic => Box::new(DynamicDeletionStrategy::new()),
            DeletionMode::Cumulative => Box::new(CumulativeDeletionStrategy::new()),
        }
    }

    /// Get deletion mode for a collection without creating strategy
    pub fn get_deletion_mode(collection_name: &str) -> DeletionMode {
        let collection_type = DeletionCollectionType::from_name(collection_name);
        collection_type.deletion_mode()
    }
}

/// Batch cleanup manager for cumulative deletions
pub struct BatchCleanupManager {
    pool: SqlitePool,
    storage_client: Arc<StorageClient>,
}

impl BatchCleanupManager {
    pub fn new(pool: SqlitePool, storage_client: Arc<StorageClient>) -> Self {
        Self {
            pool,
            storage_client,
        }
    }

    /// Get pending cumulative deletions ready for cleanup
    pub async fn get_pending_deletions(&self, limit: i32) -> DeletionResult<Vec<CumulativeDeletionItem>> {
        let query = format!(
            r#"
            SELECT
                id, file_absolute_path, collection_name, tenant_id, branch,
                marked_for_deletion_at, deletion_scheduled_at, status, retry_count
            FROM {}
            WHERE status = 'pending'
            ORDER BY marked_for_deletion_at ASC
            LIMIT ?
            "#,
            CumulativeDeletionStrategy::CUMULATIVE_DELETIONS_TABLE
        );

        let rows = sqlx::query(&query)
            .bind(limit)
            .fetch_all(&self.pool)
            .await?;

        let mut items = Vec::new();
        for row in rows {
            let id: i64 = row.get("id");
            let file_path: String = row.get("file_absolute_path");
            let collection: String = row.get("collection_name");
            let tenant_id: String = row.get("tenant_id");
            let branch: String = row.get("branch");
            let marked_at: String = row.get("marked_for_deletion_at");
            let scheduled_at: Option<String> = row.get("deletion_scheduled_at");
            let status: String = row.get("status");
            let retry_count: i32 = row.get("retry_count");

            items.push(CumulativeDeletionItem {
                id,
                file_absolute_path: file_path,
                collection_name: collection,
                tenant_id,
                branch,
                marked_for_deletion_at: DateTime::parse_from_rfc3339(&marked_at)
                    .ok()
                    .map(|dt| dt.with_timezone(&Utc)),
                deletion_scheduled_at: scheduled_at
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
                status,
                retry_count,
            });
        }

        Ok(items)
    }

    /// Execute batch cleanup for cumulative deletions
    pub async fn execute_batch_cleanup(&self, batch_size: i32) -> DeletionResult<CleanupStats> {
        info!("Starting batch cleanup for cumulative deletions (batch_size={})", batch_size);

        let pending_items = self.get_pending_deletions(batch_size).await?;

        if pending_items.is_empty() {
            debug!("No pending cumulative deletions to clean up");
            return Ok(CleanupStats::default());
        }

        info!("Found {} items for cumulative deletion cleanup", pending_items.len());

        let mut stats = CleanupStats::default();
        stats.total_items = pending_items.len() as u64;

        for item in pending_items {
            match self.cleanup_item(&item).await {
                Ok(()) => {
                    stats.items_deleted += 1;
                    info!("Successfully cleaned up: {} from {}", item.file_absolute_path, item.collection_name);
                }
                Err(e) => {
                    stats.items_failed += 1;
                    warn!("Failed to clean up {} from {}: {}", item.file_absolute_path, item.collection_name, e);

                    // Update error in database
                    self.mark_cleanup_failed(&item, &e.to_string()).await?;
                }
            }
        }

        info!(
            "Batch cleanup completed: deleted={}, failed={}, total={}",
            stats.items_deleted, stats.items_failed, stats.total_items
        );

        Ok(stats)
    }

    /// Clean up a single item
    async fn cleanup_item(&self, item: &CumulativeDeletionItem) -> DeletionResult<()> {
        // Check if collection exists
        let collection_exists = self.storage_client
            .collection_exists(&item.collection_name)
            .await
            .map_err(|e| DeletionError::Storage(e.to_string()))?;

        if !collection_exists {
            warn!("Collection {} does not exist, marking as deleted", item.collection_name);
            self.mark_cleanup_complete(item).await?;
            return Ok(());
        }

        // Delete all points matching the file_path
        self.storage_client
            .delete_points_by_filter(&item.collection_name, &item.file_absolute_path)
            .await
            .map_err(|e| DeletionError::Storage(e.to_string()))?;

        self.mark_cleanup_complete(item).await?;
        Ok(())
    }

    /// Mark cleanup as complete
    async fn mark_cleanup_complete(&self, item: &CumulativeDeletionItem) -> DeletionResult<()> {
        let now = Utc::now().to_rfc3339();
        let query = format!(
            r#"
            UPDATE {}
            SET status = 'deleted', deleted_at = ?
            WHERE id = ?
            "#,
            CumulativeDeletionStrategy::CUMULATIVE_DELETIONS_TABLE
        );

        sqlx::query(&query)
            .bind(&now)
            .bind(item.id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    /// Mark cleanup as failed
    async fn mark_cleanup_failed(&self, item: &CumulativeDeletionItem, error: &str) -> DeletionResult<()> {
        let query = format!(
            r#"
            UPDATE {}
            SET status = 'failed', retry_count = retry_count + 1, error_message = ?
            WHERE id = ?
            "#,
            CumulativeDeletionStrategy::CUMULATIVE_DELETIONS_TABLE
        );

        sqlx::query(&query)
            .bind(error)
            .bind(item.id)
            .execute(&self.pool)
            .await?;

        Ok(())
    }
}

/// Cumulative deletion item
#[derive(Debug, Clone)]
pub struct CumulativeDeletionItem {
    pub id: i64,
    pub file_absolute_path: String,
    pub collection_name: String,
    pub tenant_id: String,
    pub branch: String,
    pub marked_for_deletion_at: Option<DateTime<Utc>>,
    pub deletion_scheduled_at: Option<DateTime<Utc>>,
    pub status: String,
    pub retry_count: i32,
}

/// Cleanup statistics
#[derive(Debug, Clone, Default)]
pub struct CleanupStats {
    pub total_items: u64,
    pub items_deleted: u64,
    pub items_failed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_type_detection() {
        assert_eq!(
            DeletionCollectionType::from_name("__system_docs"),
            DeletionCollectionType::System
        );
        assert_eq!(
            DeletionCollectionType::from_name("_python_stdlib"),
            DeletionCollectionType::Library
        );
        assert_eq!(
            DeletionCollectionType::from_name("myproject123-docs"),
            DeletionCollectionType::Project
        );
        assert_eq!(
            DeletionCollectionType::from_name("workspace"),
            DeletionCollectionType::Global
        );
        assert_eq!(
            DeletionCollectionType::from_name("knowledge"),
            DeletionCollectionType::Global
        );
    }

    #[test]
    fn test_deletion_mode_mapping() {
        assert_eq!(
            DeletionCollectionType::System.deletion_mode(),
            DeletionMode::Cumulative
        );
        assert_eq!(
            DeletionCollectionType::Library.deletion_mode(),
            DeletionMode::Cumulative
        );
        assert_eq!(
            DeletionCollectionType::Project.deletion_mode(),
            DeletionMode::Dynamic
        );
        assert_eq!(
            DeletionCollectionType::Global.deletion_mode(),
            DeletionMode::Dynamic
        );
    }

    #[test]
    fn test_factory_deletion_mode() {
        assert_eq!(
            DeletionStrategyFactory::get_deletion_mode("__system"),
            DeletionMode::Cumulative
        );
        assert_eq!(
            DeletionStrategyFactory::get_deletion_mode("_library"),
            DeletionMode::Cumulative
        );
        assert_eq!(
            DeletionStrategyFactory::get_deletion_mode("project-docs"),
            DeletionMode::Dynamic
        );
        assert_eq!(
            DeletionStrategyFactory::get_deletion_mode("workspace"),
            DeletionMode::Dynamic
        );
    }
}
