//! Checkpoint management for task state persistence and rollback
//!
//! Provides checkpointing capabilities for long-running tasks, enabling
//! graceful preemption with state preservation and rollback on failure.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::storage::StorageClient;
use super::PriorityError;

/// Checkpoint data for task resumption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCheckpoint {
    pub checkpoint_id: String,
    pub task_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub task_progress: TaskProgress,
    pub state_data: serde_json::Value,
    pub files_modified: Vec<PathBuf>,
    pub rollback_actions: Vec<RollbackAction>,
}

/// Different types of progress tracking for different task types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskProgress {
    DocumentProcessing {
        chunks_processed: usize,
        total_chunks: usize,
        current_chunk_offset: usize,
    },
    FileWatching {
        files_processed: usize,
        current_directory: PathBuf,
        processed_files: Vec<PathBuf>,
    },
    QueryExecution {
        query_stage: String,
        results_collected: usize,
    },
    Generic {
        progress_percentage: f32,
        stage: String,
        metadata: HashMap<String, serde_json::Value>,
    },
}

/// Actions needed to rollback changes if task is cancelled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RollbackAction {
    DeleteFile { path: PathBuf },
    RestoreFile { original_path: PathBuf, backup_path: PathBuf },
    RemoveFromCollection { document_id: String, collection: String },
    RevertIndexChanges { index_snapshot: serde_json::Value },
    Custom { action_type: String, data: serde_json::Value },
}

/// Handler for custom rollback actions
///
/// Implement this trait to register domain-specific rollback logic
/// via `CheckpointManager::register_custom_handler`.
#[async_trait::async_trait]
pub trait CustomRollbackHandler: Send + Sync {
    /// Execute the rollback action with the provided data payload
    async fn execute(&self, data: &serde_json::Value) -> Result<(), String>;
}

/// Checkpoint manager for handling task state persistence
pub struct CheckpointManager {
    /// Storage for active checkpoints
    pub(crate) checkpoints: Arc<RwLock<HashMap<String, TaskCheckpoint>>>,
    /// Cleanup interval for old checkpoints
    checkpoint_retention: Duration,
    /// Directory for checkpoint file storage
    pub(crate) checkpoint_dir: PathBuf,
    /// Storage client for Qdrant rollback operations (RemoveFromCollection)
    pub(crate) storage_client: Option<Arc<StorageClient>>,
    /// Registry of custom rollback handlers
    pub(crate) custom_handlers: Arc<RwLock<HashMap<String, Arc<dyn CustomRollbackHandler>>>>,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new(checkpoint_dir: PathBuf, retention: Duration) -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            checkpoint_retention: retention,
            checkpoint_dir,
            storage_client: None,
            custom_handlers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set the storage client for Qdrant rollback operations
    pub fn set_storage_client(&mut self, client: Arc<StorageClient>) {
        self.storage_client = Some(client);
    }

    /// Register a custom rollback handler for a given action type
    pub async fn register_custom_handler(
        &self,
        action_type: impl Into<String>,
        handler: Arc<dyn CustomRollbackHandler>,
    ) {
        let mut handlers = self.custom_handlers.write().await;
        handlers.insert(action_type.into(), handler);
    }

    /// Create a checkpoint for a task
    pub async fn create_checkpoint(
        &self,
        task_id: Uuid,
        progress: TaskProgress,
        state_data: serde_json::Value,
        files_modified: Vec<PathBuf>,
        rollback_actions: Vec<RollbackAction>,
    ) -> Result<String, PriorityError> {
        let checkpoint_id = format!("ckpt_{}_{}", task_id, chrono::Utc::now().timestamp());

        let checkpoint = TaskCheckpoint {
            checkpoint_id: checkpoint_id.clone(),
            task_id,
            created_at: chrono::Utc::now(),
            task_progress: progress,
            state_data,
            files_modified,
            rollback_actions,
        };

        // Store in memory
        {
            let mut checkpoints_lock = self.checkpoints.write().await;
            checkpoints_lock.insert(checkpoint_id.clone(), checkpoint.clone());
        }

        // Persist to disk
        let checkpoint_file = self.checkpoint_dir.join(format!("{checkpoint_id}.json"));
        let checkpoint_json = serde_json::to_string(&checkpoint)
            .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;

        tokio::fs::write(&checkpoint_file, checkpoint_json).await
            .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;

        tracing::debug!("Created checkpoint {} for task {}", checkpoint_id, task_id);
        Ok(checkpoint_id)
    }

    /// Retrieve a checkpoint
    pub async fn get_checkpoint(&self, checkpoint_id: &str) -> Option<TaskCheckpoint> {
        let checkpoints_lock = self.checkpoints.read().await;
        checkpoints_lock.get(checkpoint_id).cloned()
    }

    /// Delete a checkpoint (task completed successfully)
    pub async fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<(), PriorityError> {
        // Remove from memory
        {
            let mut checkpoints_lock = self.checkpoints.write().await;
            checkpoints_lock.remove(checkpoint_id);
        }

        // Remove from disk
        let checkpoint_file = self.checkpoint_dir.join(format!("{checkpoint_id}.json"));
        if checkpoint_file.exists() {
            tokio::fs::remove_file(checkpoint_file).await
                .map_err(|e| PriorityError::Checkpoint(e.to_string()))?;
        }

        Ok(())
    }

    /// Rollback changes using checkpoint data
    pub async fn rollback_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<(), PriorityError> {
        let checkpoint = self.get_checkpoint(checkpoint_id).await
            .ok_or_else(|| PriorityError::Checkpoint(
                format!("Checkpoint {checkpoint_id} not found"),
            ))?;

        tracing::info!(
            "Rolling back checkpoint {} for task {}",
            checkpoint_id, checkpoint.task_id
        );

        // Execute rollback actions in reverse order
        for action in checkpoint.rollback_actions.iter().rev() {
            match self.execute_rollback_action(action).await {
                Ok(()) => {
                    tracing::debug!("Successfully executed rollback action: {:?}", action);
                }
                Err(e) => {
                    tracing::error!("Failed to execute rollback action {:?}: {}", action, e);
                    // Continue with other rollback actions even if one fails
                }
            }
        }

        // Clean up the checkpoint after rollback
        self.delete_checkpoint(checkpoint_id).await?;

        Ok(())
    }

    /// Execute a single rollback action
    async fn execute_rollback_action(
        &self,
        action: &RollbackAction,
    ) -> Result<(), PriorityError> {
        match action {
            RollbackAction::DeleteFile { path } => {
                if path.exists() {
                    tokio::fs::remove_file(path).await
                        .map_err(|e| PriorityError::RollbackFailed(e.to_string()))?;
                }
            }
            RollbackAction::RestoreFile { original_path, backup_path } => {
                if backup_path.exists() {
                    tokio::fs::copy(backup_path, original_path).await
                        .map_err(|e| PriorityError::RollbackFailed(e.to_string()))?;
                    let _ = tokio::fs::remove_file(backup_path).await;
                }
            }
            RollbackAction::RemoveFromCollection { document_id, collection } => {
                self.rollback_remove_from_collection(document_id, collection).await?;
            }
            RollbackAction::RevertIndexChanges { index_snapshot } => {
                self.rollback_revert_index(index_snapshot).await;
            }
            RollbackAction::Custom { action_type, data } => {
                self.rollback_custom(action_type, data).await?;
            }
        }

        Ok(())
    }

    /// Handle RemoveFromCollection rollback action
    async fn rollback_remove_from_collection(
        &self,
        document_id: &str,
        collection: &str,
    ) -> Result<(), PriorityError> {
        if let Some(ref storage) = self.storage_client {
            tracing::info!(
                "Rollback: removing document '{}' from collection '{}'",
                document_id, collection
            );
            match storage.delete_points_by_document_id(collection, document_id).await {
                Ok(count) => {
                    tracing::info!(
                        "Rollback: deleted {} points for document '{}' from '{}'",
                        count, document_id, collection
                    );
                }
                Err(e) => {
                    return Err(PriorityError::RollbackFailed(
                        format!(
                            "Failed to remove document '{}' from '{}': {}",
                            document_id, collection, e
                        ),
                    ));
                }
            }
        } else {
            return Err(PriorityError::RollbackFailed(
                format!(
                    "No storage client configured; cannot remove document '{}' from '{}'",
                    document_id, collection
                ),
            ));
        }
        Ok(())
    }

    /// Handle RevertIndexChanges rollback action (logs warning, no atomic revert)
    async fn rollback_revert_index(&self, index_snapshot: &serde_json::Value) {
        if let Some(ref storage) = self.storage_client {
            if let Some(collection) = index_snapshot.get("collection").and_then(|v| v.as_str()) {
                match storage.get_collection_info(collection).await {
                    Ok(info) => {
                        tracing::warn!(
                            "Rollback: index revert requested for collection '{}' \
                             (status={}, points={}). Snapshot: {}",
                            collection, info.status, info.points_count,
                            serde_json::to_string(index_snapshot).unwrap_or_default()
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Rollback: index revert requested but collection query failed: {}. \
                             Snapshot: {}",
                            e,
                            serde_json::to_string(index_snapshot).unwrap_or_default()
                        );
                    }
                }
            } else {
                tracing::warn!(
                    "Rollback: index revert requested but snapshot missing 'collection' field. \
                     Snapshot: {}",
                    serde_json::to_string(index_snapshot).unwrap_or_default()
                );
            }
        } else {
            tracing::warn!(
                "Rollback: index revert requested but no storage client configured. \
                 Snapshot: {}",
                serde_json::to_string(index_snapshot).unwrap_or_default()
            );
        }
    }

    /// Handle Custom rollback action via registered handlers
    async fn rollback_custom(
        &self,
        action_type: &str,
        data: &serde_json::Value,
    ) -> Result<(), PriorityError> {
        let handlers = self.custom_handlers.read().await;
        if let Some(handler) = handlers.get(action_type) {
            tracing::info!("Rollback: executing custom handler '{}'", action_type);
            handler.execute(data).await
                .map_err(|e| PriorityError::RollbackFailed(
                    format!("Custom rollback '{}' failed: {}", action_type, e)
                ))?;
            tracing::info!("Rollback: custom handler '{}' completed", action_type);
        } else {
            return Err(PriorityError::RollbackFailed(
                format!("No handler registered for custom rollback type '{}'", action_type),
            ));
        }
        Ok(())
    }

    /// Clean up old checkpoints
    pub async fn cleanup_old_checkpoints(&self) -> usize {
        let mut cleaned_count = 0;
        let cutoff_time = chrono::Utc::now()
            - chrono::Duration::from_std(self.checkpoint_retention)
                .unwrap_or(chrono::Duration::hours(24));

        let checkpoint_ids_to_remove: Vec<String> = {
            let checkpoints_lock = self.checkpoints.read().await;
            checkpoints_lock
                .iter()
                .filter(|(_, checkpoint)| checkpoint.created_at < cutoff_time)
                .map(|(id, _)| id.clone())
                .collect()
        };

        for checkpoint_id in checkpoint_ids_to_remove {
            if let Err(e) = self.delete_checkpoint(&checkpoint_id).await {
                tracing::error!("Failed to cleanup checkpoint {}: {}", checkpoint_id, e);
            } else {
                cleaned_count += 1;
            }
        }

        tracing::info!("Cleaned up {} old checkpoints", cleaned_count);
        cleaned_count
    }
}
