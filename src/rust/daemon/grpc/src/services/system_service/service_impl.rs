//! SystemServiceImpl struct definition, constructor, builder methods, and helper methods

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use tokio::sync::{Notify, RwLock};
use tracing::{debug, error, info, warn};
use workspace_qdrant_core::adaptive_resources::AdaptiveResourceState;
use workspace_qdrant_core::lifecycle::WatchFolderLifecycle;
use workspace_qdrant_core::QueueProcessorHealth;
use wqm_common::timestamps;

use crate::proto::{ComponentHealth, Metric, ServiceStatus};

use crate::proto::ServerState;

use super::types::{ServerStatusEntry, ServerStatusStore};

/// SystemService implementation
///
/// Provides health monitoring, status reporting, and lifecycle management.
/// Can be connected to actual queue processor health state for real metrics.
pub struct SystemServiceImpl {
    pub(super) start_time: SystemTime,
    /// Optional queue processor health state
    pub(super) queue_health: Option<Arc<QueueProcessorHealth>>,
    /// Optional database pool for refresh signal operations
    pub(super) db_pool: Option<sqlx::SqlitePool>,
    /// Server status store for tracking component status
    pub(super) status_store: ServerStatusStore,
    /// Shared pause flag for propagation to file watchers
    /// When the gRPC endpoint pauses/resumes, this flag is toggled atomically
    /// so that any FileWatcher sharing this flag reacts immediately.
    pub(super) pause_flag: Arc<AtomicBool>,
    /// Signal to trigger immediate WatchManager refresh
    pub(super) watch_refresh_signal: Option<Arc<Notify>>,
    /// Adaptive resource state for idle/burst mode reporting
    pub(super) adaptive_state: Option<Arc<AdaptiveResourceState>>,
    /// Hierarchy builder for tag hierarchy rebuild via RebuildIndex RPC
    pub(super) hierarchy_builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    /// Search database manager for FTS5 rebuild
    pub(super) search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
    /// Lexicon manager for vocabulary rebuild
    pub(super) lexicon_manager: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    /// Storage client for Qdrant operations (rules rebuild)
    pub(super) storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
}

impl std::fmt::Debug for SystemServiceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemServiceImpl")
            .field("start_time", &self.start_time)
            .field("queue_health", &self.queue_health.is_some())
            .field("db_pool", &self.db_pool.is_some())
            .field("hierarchy_builder", &self.hierarchy_builder.is_some())
            .field("search_db", &self.search_db.is_some())
            .field("lexicon_manager", &self.lexicon_manager.is_some())
            .field("storage_client", &self.storage_client.is_some())
            .finish()
    }
}

impl SystemServiceImpl {
    /// Create a new SystemService
    pub fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            queue_health: None,
            db_pool: None,
            status_store: Arc::new(RwLock::new(HashMap::new())),
            pause_flag: Arc::new(AtomicBool::new(false)),
            watch_refresh_signal: None,
            adaptive_state: None,
            hierarchy_builder: None,
            search_db: None,
            lexicon_manager: None,
            storage_client: None,
        }
    }

    /// Set queue processor health state for monitoring
    pub fn with_queue_health(mut self, queue_health: Arc<QueueProcessorHealth>) -> Self {
        self.queue_health = Some(queue_health);
        self
    }

    /// Set the database pool for refresh signal operations
    pub fn with_database_pool(mut self, pool: sqlx::SqlitePool) -> Self {
        self.db_pool = Some(pool);
        self
    }

    /// Set a shared pause flag for propagation to file watchers.
    /// The returned `Arc<AtomicBool>` should be passed to the FileWatcher so both
    /// the gRPC endpoint and the watcher share the same atomic flag.
    pub fn with_pause_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.pause_flag = flag;
        self
    }

    /// Set the watch refresh signal for triggering WatchManager refresh
    pub fn with_watch_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set the adaptive resource state for idle/burst mode reporting
    pub fn with_adaptive_state(mut self, state: Arc<AdaptiveResourceState>) -> Self {
        self.adaptive_state = Some(state);
        self
    }

    /// Set the hierarchy builder for tag hierarchy rebuild
    pub fn with_hierarchy_builder(
        mut self,
        builder: Arc<workspace_qdrant_core::HierarchyBuilder>,
    ) -> Self {
        self.hierarchy_builder = Some(builder);
        self
    }

    /// Set the search database manager for FTS5 rebuild
    pub fn with_search_db(
        mut self,
        search_db: Arc<workspace_qdrant_core::SearchDbManager>,
    ) -> Self {
        self.search_db = Some(search_db);
        self
    }

    /// Set the lexicon manager for vocabulary rebuild
    pub fn with_lexicon_manager(
        mut self,
        lexicon: Arc<workspace_qdrant_core::LexiconManager>,
    ) -> Self {
        self.lexicon_manager = Some(lexicon);
        self
    }

    /// Set the storage client for Qdrant operations (rules rebuild)
    pub fn with_storage_client(
        mut self,
        client: Arc<workspace_qdrant_core::StorageClient>,
    ) -> Self {
        self.storage_client = Some(client);
        self
    }

    /// Get a clone of the pause flag for sharing with file watchers
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_flag)
    }

    /// Get queue processor health component
    pub(super) fn get_queue_processor_health(&self) -> ComponentHealth {
        if let Some(health) = &self.queue_health {
            let is_running = health.is_running.load(Ordering::SeqCst);
            let secs_since_poll = health.seconds_since_last_poll();
            let error_count = health.error_count.load(Ordering::SeqCst);

            // Determine status based on health indicators
            let (status, message) = if !is_running {
                (ServiceStatus::Unhealthy, "Queue processor is not running")
            } else if secs_since_poll > 60 {
                (
                    ServiceStatus::Degraded,
                    "Queue processor may be stalled (>60s since last poll)",
                )
            } else if error_count > 100 {
                (ServiceStatus::Degraded, "High error count detected")
            } else {
                (ServiceStatus::Healthy, "Running normally")
            };

            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: status as i32,
                message: message.to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        } else {
            // No health state connected - report unknown
            ComponentHealth {
                component_name: "queue_processor".to_string(),
                status: ServiceStatus::Unspecified as i32,
                message: "Health monitoring not connected".to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            }
        }
    }

    /// Get queue processor metrics
    pub(super) fn get_queue_metrics(&self) -> Vec<Metric> {
        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        if let Some(health) = &self.queue_health {
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.queue_depth.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_processed.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_failed.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_errors".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.error_count.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processing_avg_ms".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.avg_processing_time_ms.load(Ordering::SeqCst) as f64,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processor_running".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: if health.is_running.load(Ordering::SeqCst) {
                        1.0
                    } else {
                        0.0
                    },
                    timestamp: now,
                },
            ]
        } else {
            // No health state connected - return placeholder metrics
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now,
                },
            ]
        }
    }

    /// Enqueue scan operations for all enabled watch folders.
    ///
    /// Used by the send_refresh_signal RPC to trigger rescans.
    /// Returns Ok(()) on success or a tonic Status error.
    pub(super) async fn enqueue_folder_scans(
        &self,
        pool: &sqlx::SqlitePool,
    ) -> Result<(), tonic::Status> {
        let folders = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT watch_id, path, collection, tenant_id FROM watch_folders WHERE enabled = 1",
        )
        .fetch_all(pool)
        .await
        .map_err(|e| {
            error!("Failed to query watch_folders: {}", e);
            tonic::Status::internal(format!("Database error: {}", e))
        })?;

        let mut scans_queued = 0u32;
        let now = timestamps::now_utc();

        for (watch_id, path, collection, tenant_id) in &folders {
            let payload = serde_json::json!({
                "folder_path": path,
                "recursive": true,
                "recursive_depth": 10,
                "patterns": [],
                "ignore_patterns": []
            });
            let payload_json = payload.to_string();

            // Compute idempotency key: SHA256(item_type|op|tenant_id|collection|payload_json)[:32]
            use sha2::{Digest, Sha256};
            let key_input = format!("folder|scan|{}|{}|{}", tenant_id, collection, payload_json);
            let hash = format!("{:x}", Sha256::digest(key_input.as_bytes()));
            let idempotency_key = &hash[..32];

            let queue_id = uuid::Uuid::new_v4().to_string();
            let result = sqlx::query(
                "INSERT OR IGNORE INTO unified_queue \
                 (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                  status, payload_json, created_at, updated_at) \
                 VALUES (?1, ?2, 'folder', 'scan', ?3, ?4, 'pending', ?5, ?6, ?7)",
            )
            .bind(&queue_id)
            .bind(idempotency_key)
            .bind(tenant_id)
            .bind(collection)
            .bind(&payload_json)
            .bind(&now)
            .bind(&now)
            .execute(pool)
            .await;

            match result {
                Ok(r) if r.rows_affected() > 0 => {
                    scans_queued += 1;
                    debug!(
                        "Queued refresh scan for watch_folder {} (path={})",
                        watch_id, path
                    );
                }
                Ok(_) => {
                    debug!(
                        "Refresh scan already queued for watch_folder {} (deduplicated)",
                        watch_id
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to queue refresh scan for watch_folder {}: {}",
                        watch_id, e
                    );
                }
            }
        }

        info!(
            "Refresh signal processed: {} watch folders found, {} scans queued",
            folders.len(),
            scans_queued
        );

        Ok(())
    }

    /// Handle a server status notification: store the entry, log transitions,
    /// and update watch folder activation via lifecycle manager.
    pub(super) async fn handle_server_notification(
        &self,
        state: ServerState,
        project_name: Option<String>,
        project_root: Option<String>,
    ) {
        // Build a component key from project info
        let component_key = project_name
            .clone()
            .or_else(|| project_root.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Log with appropriate level based on state
        match state {
            ServerState::Up => {
                info!(
                    "Server UP: component={}, project_name={:?}, project_root={:?}",
                    component_key, project_name, project_root
                );
            }
            ServerState::Down => {
                warn!(
                    "Server DOWN: component={}, project_name={:?}, project_root={:?}",
                    component_key, project_name, project_root
                );
            }
            _ => {
                debug!(
                    "Server status unspecified: component={}, state={:?}",
                    component_key, state
                );
            }
        }

        // Store the status entry
        let entry = ServerStatusEntry {
            state,
            project_name: project_name.clone(),
            project_root: project_root.clone(),
            updated_at: SystemTime::now(),
        };

        let previous_state = {
            let mut store = self.status_store.write().await;
            let prev = store.get(&component_key).map(|e| e.state);
            store.insert(component_key.clone(), entry);
            prev
        };

        // Log state transitions
        if let Some(prev) = previous_state {
            if prev != state {
                info!(
                    "Server state transition: component={}, {:?} -> {:?}",
                    component_key, prev, state
                );
            }
        }

        // Delegate is_active mutation to WatchFolderLifecycle
        if let (Some(pool), Some(ref root)) = (&self.db_pool, &project_root) {
            let is_active = matches!(state, ServerState::Up);
            let lifecycle = WatchFolderLifecycle::new(pool.clone());

            match lifecycle.set_active_by_path(root, is_active).await {
                Ok(rows) if rows > 0 => {
                    info!(
                        "Updated watch_folder activation: path={}, is_active={}",
                        root, is_active
                    );
                }
                Ok(_) => {
                    debug!(
                        "No watch_folder found for path={} (may not be registered yet)",
                        root
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to update watch_folder activation for {}: {}",
                        root, e
                    );
                }
            }
        }
    }
}

impl Default for SystemServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}
