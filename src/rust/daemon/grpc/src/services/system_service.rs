//! SystemService gRPC implementation
//!
//! Handles system health monitoring, status reporting, refresh signaling,
//! and lifecycle management operations.
//! Provides 9 RPCs: Health, GetStatus, GetMetrics, GetQueueStats, Shutdown,
//! SendRefreshSignal, NotifyServerStatus, PauseAllWatchers, ResumeAllWatchers

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::{Notify, RwLock};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};
use wqm_common::timestamps;
use workspace_qdrant_core::QueueProcessorHealth;
use workspace_qdrant_core::adaptive_resources::AdaptiveResourceState;
use workspace_qdrant_core::lifecycle::WatchFolderLifecycle;

use crate::proto::{
    system_service_server::SystemService,
    HealthResponse, SystemStatusResponse, MetricsResponse, QueueStatsResponse,
    RefreshSignalRequest, ServerStatusNotification,
    RebuildIndexRequest, RebuildIndexResponse,
    ComponentHealth, SystemMetrics, Metric,
    ServiceStatus, QueueType, ServerState,
};

/// Tracks the status of a server component (MCP server, CLI, etc.)
#[derive(Debug, Clone)]
pub struct ServerStatusEntry {
    /// Current state (UP or DOWN)
    pub state: ServerState,
    /// Project name (if project-scoped)
    pub project_name: Option<String>,
    /// Project root path
    pub project_root: Option<String>,
    /// Timestamp of last status update
    pub updated_at: SystemTime,
}

/// Thread-safe store for server status entries, keyed by component identifier
type ServerStatusStore = Arc<RwLock<HashMap<String, ServerStatusEntry>>>;

/// SystemService implementation
///
/// Provides health monitoring, status reporting, and lifecycle management.
/// Can be connected to actual queue processor health state for real metrics.
pub struct SystemServiceImpl {
    start_time: SystemTime,
    /// Optional queue processor health state
    queue_health: Option<Arc<QueueProcessorHealth>>,
    /// Optional database pool for refresh signal operations
    db_pool: Option<sqlx::SqlitePool>,
    /// Server status store for tracking component status
    status_store: ServerStatusStore,
    /// Shared pause flag for propagation to file watchers
    /// When the gRPC endpoint pauses/resumes, this flag is toggled atomically
    /// so that any FileWatcher sharing this flag reacts immediately.
    pause_flag: Arc<AtomicBool>,
    /// Signal to trigger immediate WatchManager refresh
    watch_refresh_signal: Option<Arc<Notify>>,
    /// Adaptive resource state for idle/burst mode reporting
    adaptive_state: Option<Arc<AdaptiveResourceState>>,
    /// Hierarchy builder for tag hierarchy rebuild via RebuildIndex RPC
    hierarchy_builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    /// Search database manager for FTS5 rebuild
    search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
    /// Lexicon manager for vocabulary rebuild
    lexicon_manager: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    /// Storage client for Qdrant operations (rules rebuild)
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
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
    pub fn with_hierarchy_builder(mut self, builder: Arc<workspace_qdrant_core::HierarchyBuilder>) -> Self {
        self.hierarchy_builder = Some(builder);
        self
    }

    /// Set the search database manager for FTS5 rebuild
    pub fn with_search_db(mut self, search_db: Arc<workspace_qdrant_core::SearchDbManager>) -> Self {
        self.search_db = Some(search_db);
        self
    }

    /// Set the lexicon manager for vocabulary rebuild
    pub fn with_lexicon_manager(mut self, lexicon: Arc<workspace_qdrant_core::LexiconManager>) -> Self {
        self.lexicon_manager = Some(lexicon);
        self
    }

    /// Set the storage client for Qdrant operations (rules rebuild)
    pub fn with_storage_client(mut self, client: Arc<workspace_qdrant_core::StorageClient>) -> Self {
        self.storage_client = Some(client);
        self
    }

    /// Get a clone of the pause flag for sharing with file watchers
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_flag)
    }

    /// Get queue processor health component
    fn get_queue_processor_health(&self) -> ComponentHealth {
        if let Some(health) = &self.queue_health {
            let is_running = health.is_running.load(Ordering::SeqCst);
            let secs_since_poll = health.seconds_since_last_poll();
            let error_count = health.error_count.load(Ordering::SeqCst);

            // Determine status based on health indicators
            let (status, message) = if !is_running {
                (ServiceStatus::Unhealthy, "Queue processor is not running")
            } else if secs_since_poll > 60 {
                (ServiceStatus::Degraded, "Queue processor may be stalled (>60s since last poll)")
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
    fn get_queue_metrics(&self) -> Vec<Metric> {
        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        if let Some(health) = &self.queue_health {
            vec![
                Metric {
                    name: "queue_pending".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.queue_depth.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_processed.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_failed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.items_failed.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_errors".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.error_count.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processing_avg_ms".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: health.avg_processing_time_ms.load(Ordering::SeqCst) as f64,
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processor_running".to_string(),
                    r#type: "gauge".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: if health.is_running.load(Ordering::SeqCst) { 1.0 } else { 0.0 },
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
                    timestamp: now.clone(),
                },
                Metric {
                    name: "queue_processed".to_string(),
                    r#type: "counter".to_string(),
                    labels: std::collections::HashMap::new(),
                    value: 0.0,
                    timestamp: now.clone(),
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
}

impl Default for SystemServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

#[tonic::async_trait]
impl SystemService for SystemServiceImpl {
    /// Quick health check for monitoring/alerting (spec: Health)
    async fn health(
        &self,
        _request: Request<()>,
    ) -> Result<Response<HealthResponse>, Status> {
        debug!("Health check requested");

        // Build component health list
        let mut components = vec![
            ComponentHealth {
                component_name: "grpc_server".to_string(),
                status: ServiceStatus::Healthy as i32,
                message: "Running".to_string(),
                last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
            },
        ];

        // Add queue processor health
        let queue_health = self.get_queue_processor_health();
        let queue_status = queue_health.status;
        components.push(queue_health);

        // Determine overall status (worst of all components)
        let overall_status = if components.iter().any(|c| c.status == ServiceStatus::Unhealthy as i32) {
            ServiceStatus::Unhealthy
        } else if components.iter().any(|c| c.status == ServiceStatus::Degraded as i32) {
            ServiceStatus::Degraded
        } else if components.iter().any(|c| c.status == ServiceStatus::Unspecified as i32) {
            // If queue health is unknown but gRPC is healthy, still report healthy
            if queue_status == ServiceStatus::Unspecified as i32 {
                ServiceStatus::Healthy
            } else {
                ServiceStatus::Unspecified
            }
        } else {
            ServiceStatus::Healthy
        };

        let response = HealthResponse {
            status: overall_status as i32,
            components,
            timestamp: Some(prost_types::Timestamp::from(SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    /// Queue statistics for monitoring (spec: GetQueueStats)
    async fn get_queue_stats(
        &self,
        _request: Request<()>,
    ) -> Result<Response<QueueStatsResponse>, Status> {
        debug!("Queue stats requested");

        let (pending, in_progress, completed, failed) = if let Some(health) = &self.queue_health {
            (
                health.queue_depth.load(Ordering::SeqCst) as i32,
                0, // Would need additional tracking
                health.items_processed.load(Ordering::SeqCst) as i32,
                health.items_failed.load(Ordering::SeqCst) as i32,
            )
        } else {
            (0, 0, 0, 0)
        };

        let response = QueueStatsResponse {
            pending_count: pending,
            in_progress_count: in_progress,
            completed_count: completed,
            failed_count: failed,
            by_item_type: std::collections::HashMap::new(),
            by_collection: std::collections::HashMap::new(),
            stale_items_count: 0,
            collected_at: Some(prost_types::Timestamp::from(SystemTime::now())),
        };

        Ok(Response::new(response))
    }

    /// Graceful daemon shutdown (spec: Shutdown)
    async fn shutdown(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        warn!("Shutdown requested via gRPC");

        // In a real implementation, this would trigger graceful shutdown
        // For now, we just acknowledge the request
        // The actual shutdown would be handled by the main daemon process

        Ok(Response::new(()))
    }

    /// Comprehensive system state snapshot
    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        info!("System status requested");

        // Get queue depth from health state if available
        let pending_operations = self.queue_health
            .as_ref()
            .map(|h| h.queue_depth.load(Ordering::SeqCst) as i32)
            .unwrap_or(0);

        // Read adaptive resource state for idle/burst reporting
        let (resource_mode, idle_seconds, current_max_embeddings, current_inter_item_delay_ms) =
            if let Some(ref state) = self.adaptive_state {
                let mode = state.mode();
                (
                    Some(mode.as_str().to_string()),
                    Some(state.idle_seconds()),
                    Some(state.max_concurrent_embeddings() as i32),
                    Some(state.inter_item_delay_ms() as i64),
                )
            } else {
                (None, None, None, None)
            };

        let response = SystemStatusResponse {
            status: ServiceStatus::Healthy as i32,
            metrics: Some(SystemMetrics {
                cpu_usage_percent: 0.0,
                memory_usage_bytes: 0,
                memory_total_bytes: 0,
                disk_usage_bytes: 0,
                disk_total_bytes: 0,
                active_connections: 1,
                pending_operations,
            }),
            active_projects: vec![],
            total_documents: 0,
            total_collections: 0,
            uptime_since: Some(prost_types::Timestamp::from(self.start_time)),
            resource_mode,
            idle_seconds,
            current_max_embeddings,
            current_inter_item_delay_ms,
        };

        Ok(Response::new(response))
    }

    /// Current performance metrics (no historical data)
    async fn get_metrics(
        &self,
        _request: Request<()>,
    ) -> Result<Response<MetricsResponse>, Status> {
        debug!("Metrics requested");

        let now = Some(prost_types::Timestamp::from(SystemTime::now()));

        // Start with general metrics
        let mut metrics = vec![
            Metric {
                name: "requests_total".to_string(),
                r#type: "counter".to_string(),
                labels: std::collections::HashMap::new(),
                value: 0.0,
                timestamp: now.clone(),
            },
            Metric {
                name: "uptime_seconds".to_string(),
                r#type: "gauge".to_string(),
                labels: std::collections::HashMap::new(),
                value: self.start_time
                    .elapsed()
                    .map(|d| d.as_secs() as f64)
                    .unwrap_or(0.0),
                timestamp: now.clone(),
            },
        ];

        // Add queue processor metrics
        metrics.extend(self.get_queue_metrics());

        let response = MetricsResponse {
            metrics,
            collected_at: now,
        };

        Ok(Response::new(response))
    }

    /// Signal database state changes for event-driven refresh
    ///
    /// Handles different queue types:
    /// - INGEST_QUEUE: Triggers scan of all active watch folders
    /// - WATCHED_PROJECTS: Re-validates watch folder configurations
    /// - WATCHED_FOLDERS: Same as WATCHED_PROJECTS
    /// - TOOLS_AVAILABLE: Logs tool availability change (LSP/grammar)
    async fn send_refresh_signal(
        &self,
        request: Request<RefreshSignalRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let queue_type = QueueType::try_from(req.queue_type)
            .unwrap_or(QueueType::Unspecified);

        info!(
            "Refresh signal received: queue_type={:?}, lsp_languages={:?}, grammar_languages={:?}",
            queue_type, req.lsp_languages, req.grammar_languages
        );

        let pool = match &self.db_pool {
            Some(p) => p,
            None => {
                warn!("Refresh signal received but no database pool configured");
                return Ok(Response::new(()));
            }
        };

        match queue_type {
            QueueType::IngestQueue | QueueType::WatchedProjects | QueueType::WatchedFolders => {
                // Query all enabled watch folders and enqueue scan operations
                let folders = sqlx::query_as::<_, (String, String, String, String)>(
                    "SELECT watch_id, path, collection, tenant_id FROM watch_folders WHERE enabled = 1"
                )
                .fetch_all(pool)
                .await
                .map_err(|e| {
                    error!("Failed to query watch_folders: {}", e);
                    Status::internal(format!("Database error: {}", e))
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
                    use sha2::{Sha256, Digest};
                    let key_input = format!("folder|scan|{}|{}|{}", tenant_id, collection, payload_json);
                    let hash = format!("{:x}", Sha256::digest(key_input.as_bytes()));
                    let idempotency_key = &hash[..32];

                    let queue_id = uuid::Uuid::new_v4().to_string();
                    let result = sqlx::query(
                        "INSERT OR IGNORE INTO unified_queue \
                         (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
                          status, payload_json, created_at, updated_at) \
                         VALUES (?1, ?2, 'folder', 'scan', ?3, ?4, 'pending', ?5, ?6, ?7)"
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
                            debug!("Queued refresh scan for watch_folder {} (path={})", watch_id, path);
                        }
                        Ok(_) => {
                            debug!("Refresh scan already queued for watch_folder {} (deduplicated)", watch_id);
                        }
                        Err(e) => {
                            warn!("Failed to queue refresh scan for watch_folder {}: {}", watch_id, e);
                        }
                    }
                }

                info!(
                    "Refresh signal processed: {} watch folders found, {} scans queued",
                    folders.len(), scans_queued
                );

                // Signal WatchManager for immediate config refresh
                if let Some(ref signal) = self.watch_refresh_signal {
                    signal.notify_one();
                    info!("Notified WatchManager to refresh watch configurations");
                }
            }
            QueueType::ToolsAvailable => {
                info!(
                    "Tools available signal: lsp_languages={:?}, grammar_languages={:?}",
                    req.lsp_languages, req.grammar_languages
                );
                // Tool availability is informational - the queue processor will pick up
                // new tool capabilities on its next processing cycle
            }
            QueueType::Unspecified => {
                warn!("Unspecified queue type in refresh signal");
            }
        }

        Ok(Response::new(()))
    }

    /// MCP/CLI server lifecycle notifications
    ///
    /// Stores server status updates and logs state transitions.
    /// When a project goes DOWN, its watch folder is deactivated.
    /// When a project comes UP, its watch folder is activated.
    async fn notify_server_status(
        &self,
        request: Request<ServerStatusNotification>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let state = ServerState::try_from(req.state)
            .unwrap_or(ServerState::Unspecified);

        // Build a component key from project info
        let component_key = req.project_name.clone()
            .or_else(|| req.project_root.clone())
            .unwrap_or_else(|| "unknown".to_string());

        // Log with appropriate level based on state
        match state {
            ServerState::Up => {
                info!(
                    "Server UP: component={}, project_name={:?}, project_root={:?}",
                    component_key, req.project_name, req.project_root
                );
            }
            ServerState::Down => {
                warn!(
                    "Server DOWN: component={}, project_name={:?}, project_root={:?}",
                    component_key, req.project_name, req.project_root
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
            project_name: req.project_name.clone(),
            project_root: req.project_root.clone(),
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
        if let (Some(pool), Some(project_root)) = (&self.db_pool, &req.project_root) {
            let is_active = matches!(state, ServerState::Up);
            let lifecycle = WatchFolderLifecycle::new(pool.clone());

            match lifecycle.set_active_by_path(project_root, is_active).await {
                Ok(rows) if rows > 0 => {
                    info!(
                        "Updated watch_folder activation: path={}, is_active={}",
                        project_root, is_active
                    );
                }
                Ok(_) => {
                    debug!(
                        "No watch_folder found for path={} (may not be registered yet)",
                        project_root
                    );
                }
                Err(e) => {
                    warn!(
                        "Failed to update watch_folder activation for {}: {}",
                        project_root, e
                    );
                }
            }
        }

        Ok(Response::new(()))
    }

    /// Pause all file watchers (master switch)
    ///
    /// Sets is_paused=1 in watch_folders for all enabled watches and toggles
    /// the shared pause flag so connected FileWatcher instances react immediately.
    async fn pause_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Pause all watchers requested");

        // Toggle the in-memory pause flag immediately for connected watchers
        self.pause_flag.store(true, Ordering::SeqCst);

        let pool = match &self.db_pool {
            Some(p) => p,
            None => {
                warn!("Pause requested but no database pool configured (in-memory flag set)");
                return Ok(Response::new(()));
            }
        };

        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 1, \
             pause_start_time = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE enabled = 1 AND is_paused = 0"
        )
        .execute(pool)
        .await
        .map_err(|e| {
            error!("Failed to pause watchers: {}", e);
            Status::internal(format!("Database error: {}", e))
        })?;

        let affected = result.rows_affected();
        info!("Paused {} watch folder(s)", affected);

        // Insert diagnostic entry into unified_queue for audit trail
        let now = timestamps::now_utc();
        let queue_id = format!("pause-{}", &now);
        let metadata = serde_json::json!({
            "action": "pause",
            "affected_watchers": affected,
        }).to_string();
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, metadata, created_at, updated_at) \
             VALUES (?1, ?2, 'metadata', 'pause', '_system', '_system', 'done', ?3, ?4, ?5)"
        )
        .bind(&queue_id)
        .bind(&queue_id)
        .bind(&metadata)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;

        Ok(Response::new(()))
    }

    /// Resume all file watchers (master switch)
    ///
    /// Sets is_paused=0 in watch_folders for all enabled watches and clears
    /// the shared pause flag so connected FileWatcher instances resume processing.
    async fn resume_all_watchers(
        &self,
        _request: Request<()>,
    ) -> Result<Response<()>, Status> {
        info!("Resume all watchers requested");

        // Clear the in-memory pause flag immediately for connected watchers
        self.pause_flag.store(false, Ordering::SeqCst);

        let pool = match &self.db_pool {
            Some(p) => p,
            None => {
                warn!("Resume requested but no database pool configured (in-memory flag cleared)");
                return Ok(Response::new(()));
            }
        };

        let result = sqlx::query(
            "UPDATE watch_folders SET is_paused = 0, \
             pause_start_time = NULL, \
             updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
             WHERE enabled = 1 AND is_paused = 1"
        )
        .execute(pool)
        .await
        .map_err(|e| {
            error!("Failed to resume watchers: {}", e);
            Status::internal(format!("Database error: {}", e))
        })?;

        let affected = result.rows_affected();
        info!("Resumed {} watch folder(s)", affected);

        // Insert diagnostic entry into unified_queue for audit trail
        let now = timestamps::now_utc();
        let queue_id = format!("resume-{}", &now);
        let metadata = serde_json::json!({
            "action": "resume",
            "affected_watchers": affected,
        }).to_string();
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, metadata, created_at, updated_at) \
             VALUES (?1, ?2, 'metadata', 'resume', '_system', '_system', 'done', ?3, ?4, ?5)"
        )
        .bind(&queue_id)
        .bind(&queue_id)
        .bind(&metadata)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;

        Ok(Response::new(()))
    }

    async fn rebuild_index(
        &self,
        request: Request<RebuildIndexRequest>,
    ) -> Result<Response<RebuildIndexResponse>, Status> {
        let req = request.into_inner();
        let target = req.target.to_lowercase();

        info!(target = %target, tenant = ?req.tenant_id, "RebuildIndex requested");

        let base_target = target.as_str();

        const VALID_TARGETS: &[&str] = &[
            "tags", "search", "vocabulary", "keywords", "rules",
            "projects", "libraries", "all",
        ];
        if !VALID_TARGETS.contains(&base_target) {
            return Err(Status::invalid_argument(format!(
                "Unknown rebuild target '{}'. Valid targets: {}",
                target,
                VALID_TARGETS.join(", ")
            )));
        }

        // Clone shared resources for the background task
        let hierarchy_builder = self.hierarchy_builder.clone();
        let search_db = self.search_db.clone();
        let lexicon_manager = self.lexicon_manager.clone();
        let storage_client = self.storage_client.clone();
        let db_pool = self.db_pool.clone();
        let tenant_id = req.tenant_id.clone();
        let collection = req.collection.clone().unwrap_or_else(|| "projects".into());
        let target_owned = base_target.to_string();

        // Spawn rebuild as a background task to avoid gRPC timeout
        tokio::spawn(async move {
            match target_owned.as_str() {
                "tags" => rebuild_tags(hierarchy_builder, tenant_id.as_deref()).await,
                "search" => rebuild_search(search_db).await,
                "vocabulary" => rebuild_vocabulary(lexicon_manager, db_pool.as_ref(), &collection).await,
                "keywords" => rebuild_keywords(db_pool.as_ref(), tenant_id.as_deref(), &collection).await,
                "rules" => rebuild_rules(storage_client, db_pool.as_ref()).await,
                "projects" => rebuild_watch_folders(db_pool.as_ref(), "projects", tenant_id.as_deref()).await,
                "libraries" => rebuild_watch_folders(db_pool.as_ref(), "libraries", tenant_id.as_deref()).await,
                "all" => {
                    info!("Starting full rebuild (all targets)");
                    rebuild_vocabulary(lexicon_manager, db_pool.as_ref(), &collection).await;
                    rebuild_search(search_db).await;
                    rebuild_tags(hierarchy_builder, tenant_id.as_deref()).await;
                    rebuild_keywords(db_pool.as_ref(), tenant_id.as_deref(), &collection).await;
                    rebuild_rules(storage_client, db_pool.as_ref()).await;
                    rebuild_watch_folders(db_pool.as_ref(), "projects", tenant_id.as_deref()).await;
                    rebuild_watch_folders(db_pool.as_ref(), "libraries", tenant_id.as_deref()).await;
                    info!("Full rebuild complete (all targets)");
                }
                _ => {} // Validated above
            }
        });

        // Return immediately — rebuild runs in background
        let mut details = std::collections::HashMap::new();
        if let Some(tid) = &req.tenant_id {
            details.insert("tenant_id".into(), tid.clone());
        }
        details.insert("target".into(), req.target);

        Ok(Response::new(RebuildIndexResponse {
            success: true,
            message: "Rebuild started in background".into(),
            duration_ms: 0,
            details,
        }))
    }
}

// ============================================================================
// Rebuild target helper functions
// ============================================================================

/// Rebuild canonical tag hierarchy.
async fn rebuild_tags(
    builder: Option<Arc<workspace_qdrant_core::HierarchyBuilder>>,
    tenant_id: Option<&str>,
) {
    let Some(builder) = builder else {
        error!(target = "tags", "Hierarchy builder not configured");
        return;
    };
    let start = std::time::Instant::now();
    if let Some(tid) = tenant_id {
        match builder.rebuild_tenant(tid).await {
            Ok(Some(r)) => {
                let total = r.level1_count + r.level2_count + r.level3_count;
                info!(target = "tags", tenant = tid, canonical_tags = total,
                    edges = r.edges_created, duration_ms = start.elapsed().as_millis() as u64,
                    "Tag hierarchy rebuild complete");
            }
            Ok(None) => info!(target = "tags", tenant = tid, "Skipped (too few tags)"),
            Err(e) => error!(target = "tags", tenant = tid, error = %e, "Tag hierarchy rebuild failed"),
        }
    } else {
        match builder.rebuild_all().await {
            Ok(r) => info!(target = "tags", tenants = r.tenants_processed,
                canonical_tags = r.total_canonical_tags, edges = r.total_edges,
                duration_ms = start.elapsed().as_millis() as u64,
                "Tag hierarchy rebuild complete (all tenants)"),
            Err(e) => error!(target = "tags", error = %e, "Tag hierarchy rebuild failed (all tenants)"),
        }
    }
}

/// Rebuild FTS5 search index.
async fn rebuild_search(
    search_db: Option<Arc<workspace_qdrant_core::SearchDbManager>>,
) {
    let Some(sdb) = search_db else {
        error!(target = "search", "SearchDbManager not configured");
        return;
    };
    let start = std::time::Instant::now();
    match sdb.rebuild_fts().await {
        Ok(()) => {
            if let Err(e) = sdb.optimize_fts().await {
                warn!(target = "search", error = %e, "FTS5 rebuilt but optimize failed");
            } else {
                info!(target = "search", duration_ms = start.elapsed().as_millis() as u64,
                    "FTS5 search index rebuilt and optimized");
            }
        }
        Err(e) => error!(target = "search", error = %e, "FTS5 rebuild failed"),
    }
}

/// Rebuild BM25 sparse vocabulary.
async fn rebuild_vocabulary(
    lexicon: Option<Arc<workspace_qdrant_core::LexiconManager>>,
    db_pool: Option<&sqlx::SqlitePool>,
    collection: &str,
) {
    let Some(lexicon) = lexicon else {
        error!(target = "vocabulary", "LexiconManager not configured");
        return;
    };
    let Some(pool) = db_pool else {
        error!(target = "vocabulary", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    // Step 1: Cleanup junk terms
    let junk_removed = match lexicon.cleanup_junk_terms().await {
        Ok(n) => n,
        Err(e) => {
            error!(target = "vocabulary", error = %e, "Junk cleanup failed");
            return;
        }
    };

    // Step 2: Delete vocabulary for the collection and reset corpus stats
    let vocab_deleted = match sqlx::query(
        "DELETE FROM sparse_vocabulary WHERE collection = ?1",
    )
    .bind(collection)
    .execute(pool)
    .await
    {
        Ok(r) => r.rows_affected(),
        Err(e) => {
            error!(target = "vocabulary", error = %e, "Vocabulary delete failed");
            return;
        }
    };

    if let Err(e) = sqlx::query(
        "DELETE FROM corpus_statistics WHERE collection = ?1",
    )
    .bind(collection)
    .execute(pool)
    .await
    {
        error!(target = "vocabulary", error = %e, "Corpus stats delete failed");
        return;
    }

    // Step 3: Clear in-memory BM25 state
    lexicon.clear_all().await;

    info!(target = "vocabulary", vocab_deleted, junk_removed, collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Vocabulary cleared. Will rebuild incrementally on next processing.");
}

/// Re-extract keywords/tags by enqueuing uplift operations.
async fn rebuild_keywords(
    db_pool: Option<&sqlx::SqlitePool>,
    tenant_id: Option<&str>,
    collection: &str,
) {
    let Some(pool) = db_pool else {
        error!(target = "keywords", "Database pool not configured");
        return;
    };

    let start = std::time::Instant::now();

    // Fetch all files for the scope
    let files: Vec<(i64, String, String)> = if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.file_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.tenant_id = ?1 AND wf.collection = ?2",
        )
        .bind(tid)
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                return;
            }
        }
    } else {
        match sqlx::query_as::<_, (i64, String, String)>(
            "SELECT tf.file_id, tf.file_path, wf.tenant_id \
             FROM tracked_files tf JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id \
             WHERE wf.collection = ?1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!(target = "keywords", error = %e, "Failed to fetch files");
                return;
            }
        }
    };

    if files.is_empty() {
        info!(target = "keywords", collection, "No files found");
        return;
    }

    // Delete existing keyword/tag data for the scope
    if tenant_id.is_some() {
        // Tenant-scoped: delete by file IDs
        for (file_id, _, _) in &files {
            let _ = sqlx::query("DELETE FROM keywords WHERE file_id = ?1")
                .bind(file_id).execute(pool).await;
            let _ = sqlx::query("DELETE FROM tags WHERE file_id = ?1")
                .bind(file_id).execute(pool).await;
        }
    } else {
        // Collection-wide: bulk delete
        let _ = sqlx::query("DELETE FROM keywords WHERE collection = ?1")
            .bind(collection).execute(pool).await;
        let _ = sqlx::query("DELETE FROM tags WHERE collection = ?1")
            .bind(collection).execute(pool).await;
        let _ = sqlx::query("DELETE FROM keyword_baskets WHERE collection = ?1")
            .bind(collection).execute(pool).await;
    }

    // Enqueue uplift operations for all files
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (_file_id, file_path, tid) in &files {
        let payload = serde_json::json!({ "file_path": file_path, "rebuild": true });
        let payload_str = payload.to_string();
        let idem_key = wqm_common::hashing::compute_content_hash(
            &format!("file|uplift|{}|{}|{}", tid, collection, payload_str),
        );

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'file', 'uplift', ?3, ?4, 3, 'pending', ?5, ?6, ?7)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(&idem_key[..32])
        .bind(tid)
        .bind(collection)
        .bind(&payload_str)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;
        enqueued += 1;
    }

    info!(target = "keywords", enqueued, collection,
        duration_ms = start.elapsed().as_millis() as u64,
        "Cleared keyword/tag data and enqueued uplift operations");
}

/// Self-diagnosing rules reconciliation.
///
/// Compares Qdrant `rules` collection against SQLite `rules_mirror` and fixes
/// discrepancies in both directions without user-specified direction. Also
/// detects and removes duplicate labels and duplicate content in Qdrant.
///
/// Reconciliation steps:
/// 1. Scroll all Qdrant rules, index by label
/// 2. Read all SQLite rules_mirror rows, index by rule_id (= label)
/// 3. Detect duplicate labels in Qdrant → keep newest, delete older points
/// 4. Detect duplicate content in Qdrant → keep one, delete duplicates
/// 5. Rules in Qdrant but not SQLite → insert into rules_mirror
/// 6. Rules in SQLite but not Qdrant → enqueue re-ingestion via unified_queue
/// 7. Rules in both but content differs → Qdrant is authoritative, update SQLite
async fn rebuild_rules(
    storage_client: Option<Arc<workspace_qdrant_core::StorageClient>>,
    db_pool: Option<&sqlx::SqlitePool>,
) {
    let start = std::time::Instant::now();
    let Some(pool) = db_pool else {
        error!("[rebuild:rules] Database pool not configured");
        return;
    };
    let Some(storage) = storage_client else {
        error!("[rebuild:rules] Storage client not configured");
        return;
    };

    use qdrant_client::qdrant::{Filter, value::Kind};

    // --- Step 1: Scroll all Qdrant rules ---
    // Verify collection exists before scrolling
    match storage.collection_exists("rules").await {
        Ok(false) => {
            info!("[rebuild:rules] Rules collection does not exist — nothing to reconcile");
            return;
        }
        Err(e) => {
            error!("[rebuild:rules] Failed to check rules collection: {}", e);
            return;
        }
        Ok(true) => {}
    }

    // Rules collections are small — single scroll avoids pagination offset issues
    let all_points = match storage.scroll_with_filter("rules", Filter::default(), 10000, None).await {
        Ok(points) => points,
        Err(e) => {
            error!("[rebuild:rules] Failed to scroll rules from Qdrant: {}", e);
            return;
        }
    };

    let extract_str = |point: &qdrant_client::qdrant::RetrievedPoint, key: &str| -> Option<String> {
        point.payload.get(key).and_then(|v| {
            v.kind.as_ref().and_then(|k| match k {
                Kind::StringValue(s) => Some(s.clone()),
                _ => None,
            })
        })
    };

    let extract_point_id_str = |point: &qdrant_client::qdrant::RetrievedPoint| -> Option<String> {
        point.id.as_ref().and_then(|pid| {
            pid.point_id_options.as_ref().map(|opts| match opts {
                qdrant_client::qdrant::point_id::PointIdOptions::Uuid(u) => u.clone(),
                qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => n.to_string(),
            })
        })
    };

    // Build lookup: label → Vec<(point_id, content, scope, tenant, updated_at)>
    let mut qdrant_by_label: std::collections::HashMap<String, Vec<(String, String, Option<String>, Option<String>, String)>> =
        std::collections::HashMap::new();
    let mut qdrant_unlabeled = Vec::new();

    for point in &all_points {
        let point_id = match extract_point_id_str(point) {
            Some(id) => id,
            None => continue,
        };
        let content = extract_str(point, "content").unwrap_or_default();
        let scope = extract_str(point, "scope");
        let tenant = extract_str(point, "tenant_id");
        let updated_at = extract_str(point, "updated_at").unwrap_or_default();
        let label = extract_str(point, "label");

        match label {
            Some(l) if !l.is_empty() => {
                qdrant_by_label.entry(l).or_default()
                    .push((point_id, content, scope, tenant, updated_at));
            }
            _ => {
                qdrant_unlabeled.push(point_id.clone());
                warn!("[rebuild:rules] Qdrant rule point {} has no label — skipping", point_id);
            }
        }
    }

    info!("[rebuild:rules] Found {} Qdrant rules ({} unique labels, {} unlabeled)",
        all_points.len(), qdrant_by_label.len(), qdrant_unlabeled.len());

    // --- Step 2: Read all SQLite rules_mirror ---
    let db_rules: Vec<(String, String, Option<String>, Option<String>)> =
        match sqlx::query_as::<_, (String, String, Option<String>, Option<String>)>(
            "SELECT rule_id, rule_text, scope, tenant_id FROM rules_mirror",
        )
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!("[rebuild:rules] Failed to read rules_mirror: {}", e);
                return;
            }
        };

    let db_by_label: std::collections::HashMap<String, (String, Option<String>, Option<String>)> =
        db_rules.into_iter()
            .map(|(label, text, scope, tenant)| (label, (text, scope, tenant)))
            .collect();

    info!("[rebuild:rules] Found {} rules in SQLite rules_mirror", db_by_label.len());

    // --- Step 3: Deduplicate labels in Qdrant ---
    let mut duplicate_ids_to_delete = Vec::new();
    let mut dedup_label_count = 0u64;

    for (label, entries) in &qdrant_by_label {
        if entries.len() > 1 {
            dedup_label_count += 1;
            // Keep the entry with the latest updated_at, delete the rest
            let mut sorted = entries.clone();
            sorted.sort_by(|a, b| b.4.cmp(&a.4)); // DESC by updated_at
            let kept = &sorted[0];
            info!("[rebuild:rules] Label '{}' has {} duplicates — keeping point {}", label, entries.len(), kept.0);
            for stale in &sorted[1..] {
                duplicate_ids_to_delete.push(stale.0.clone());
            }
        }
    }

    // --- Step 4: Deduplicate content in Qdrant ---
    // Build content → Vec<(label, point_id)> to find same content under different labels
    let mut content_map: std::collections::HashMap<String, Vec<(String, String)>> =
        std::collections::HashMap::new();
    let mut dedup_content_count = 0u64;

    for (label, entries) in &qdrant_by_label {
        // Use only the first (winning) entry for each label after Step 3 dedup
        if let Some(entry) = entries.first() {
            // Skip if this point is already marked for deletion from Step 3
            if !duplicate_ids_to_delete.contains(&entry.0) {
                content_map.entry(entry.1.clone()).or_default()
                    .push((label.clone(), entry.0.clone()));
            }
        }
    }

    for (content_preview, entries) in &content_map {
        if entries.len() > 1 {
            dedup_content_count += 1;
            let preview = if content_preview.len() > 60 {
                format!("{}...", &content_preview[..60])
            } else {
                content_preview.clone()
            };
            info!("[rebuild:rules] Duplicate content across {} labels: {:?} — keeping '{}'",
                entries.len(),
                entries.iter().map(|(l, _)| l.as_str()).collect::<Vec<_>>(),
                entries[0].0);
            for dup in &entries[1..] {
                duplicate_ids_to_delete.push(dup.1.clone());
                // Also remove the stale label from rules_mirror
                let _ = sqlx::query("DELETE FROM rules_mirror WHERE rule_id = ?1")
                    .bind(&dup.0)
                    .execute(pool)
                    .await;
            }
            let _ = preview; // suppress unused warning
        }
    }

    // Execute Qdrant deletions
    let deleted_count = duplicate_ids_to_delete.len() as u64;
    if !duplicate_ids_to_delete.is_empty() {
        match storage.delete_points_by_ids("rules", &duplicate_ids_to_delete).await {
            Ok(_) => info!("[rebuild:rules] Deleted {} duplicate Qdrant points", deleted_count),
            Err(e) => error!("[rebuild:rules] Failed to delete duplicate points: {}", e),
        }
    }

    // --- Build the deduplicated Qdrant state (label → single winning entry) ---
    let mut qdrant_deduped: std::collections::HashMap<String, (String, Option<String>, Option<String>)> =
        std::collections::HashMap::new();

    for (label, entries) in &qdrant_by_label {
        // After dedup, pick the first entry not in the deletion list
        if let Some(winner) = entries.iter().find(|e| !duplicate_ids_to_delete.contains(&e.0)) {
            qdrant_deduped.insert(label.clone(), (winner.1.clone(), winner.2.clone(), winner.3.clone()));
        }
    }

    // --- Step 5 & 6 & 7: Bidirectional reconciliation ---
    let now = wqm_common::timestamps::now_utc();
    let mut mirror_inserted = 0u64;
    let mut mirror_updated = 0u64;
    let mut enqueued_to_qdrant = 0u64;

    // 5. Rules in Qdrant but not SQLite → insert into rules_mirror
    // 7. Rules in both but content differs → Qdrant authoritative, update SQLite
    for (label, (q_content, q_scope, q_tenant)) in &qdrant_deduped {
        match db_by_label.get(label) {
            None => {
                // Missing from SQLite — insert
                let _ = sqlx::query(
                    "INSERT OR IGNORE INTO rules_mirror \
                     (rule_id, rule_text, scope, tenant_id, created_at, updated_at) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                )
                .bind(label)
                .bind(q_content)
                .bind(q_scope)
                .bind(q_tenant)
                .bind(&now)
                .bind(&now)
                .execute(pool)
                .await;
                mirror_inserted += 1;
            }
            Some((db_content, _, _)) if db_content != q_content => {
                // Content mismatch — Qdrant is authoritative
                let _ = sqlx::query(
                    "UPDATE rules_mirror SET rule_text = ?1, scope = ?2, tenant_id = ?3, updated_at = ?4 \
                     WHERE rule_id = ?5",
                )
                .bind(q_content)
                .bind(q_scope)
                .bind(q_tenant)
                .bind(&now)
                .bind(label)
                .execute(pool)
                .await;
                mirror_updated += 1;
            }
            _ => {} // In sync — nothing to do
        }
    }

    // 6. Rules in SQLite but not Qdrant → enqueue re-ingestion
    for (label, (db_content, db_scope, db_tenant)) in &db_by_label {
        if !qdrant_deduped.contains_key(label) {
            let tid = db_tenant.as_deref().unwrap_or("global");
            let payload = serde_json::json!({
                "content": db_content,
                "scope": db_scope,
                "label": label,
            });
            let payload_str = payload.to_string();
            let idem_key = wqm_common::hashing::compute_content_hash(
                &format!("text|add|{}|rules|{}", tid, payload_str),
            );

            let _ = sqlx::query(
                "INSERT OR IGNORE INTO unified_queue \
                 (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
                 VALUES (?1, ?2, 'text', 'add', ?3, 'rules', 8, 'pending', ?4, ?5, ?6)",
            )
            .bind(uuid::Uuid::new_v4().to_string())
            .bind(&idem_key[..32])
            .bind(tid)
            .bind(&payload_str)
            .bind(&now)
            .bind(&now)
            .execute(pool)
            .await;
            enqueued_to_qdrant += 1;
        }
    }

    // --- Summary ---
    info!("[rebuild:rules] Reconciliation complete in {}ms: \
        qdrant_total={}, db_total={}, \
        label_dups_removed={}, content_dups_removed={}, qdrant_points_deleted={}, \
        mirror_inserted={}, mirror_updated={}, enqueued_to_qdrant={}",
        start.elapsed().as_millis(),
        all_points.len(), db_by_label.len(),
        dedup_label_count, dedup_content_count, deleted_count,
        mirror_inserted, mirror_updated, enqueued_to_qdrant);
}

/// Rescan watch folders by enqueuing scan operations.
async fn rebuild_watch_folders(
    db_pool: Option<&sqlx::SqlitePool>,
    collection: &str,
    tenant_id: Option<&str>,
) {
    let Some(pool) = db_pool else {
        error!("[rebuild:{}] Database pool not configured", collection);
        return;
    };

    // Fetch watch folders for the given collection (and optional tenant)
    let folders: Vec<(String, String, String)> = if let Some(tid) = tenant_id {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND tenant_id = ?2 AND enabled = 1",
        )
        .bind(collection)
        .bind(tid)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!("[rebuild:{}] Failed to fetch watch folders: {}", collection, e);
                return;
            }
        }
    } else {
        match sqlx::query_as::<_, (String, String, String)>(
            "SELECT watch_id, tenant_id, path FROM watch_folders \
             WHERE collection = ?1 AND enabled = 1",
        )
        .bind(collection)
        .fetch_all(pool)
        .await
        {
            Ok(rows) => rows,
            Err(e) => {
                error!("[rebuild:{}] Failed to fetch watch folders: {}", collection, e);
                return;
            }
        }
    };

    if folders.is_empty() {
        info!("[rebuild:{}] No enabled watch folders found", collection);
        return;
    }

    // Enqueue scan operations for each watch folder
    let now = wqm_common::timestamps::now_utc();
    let mut enqueued = 0u64;
    for (watch_id, tid, path) in &folders {
        let payload = serde_json::json!({
            "path": path,
            "watch_id": watch_id,
            "rebuild": true,
        });
        let payload_str = payload.to_string();
        let idem_key = wqm_common::hashing::compute_content_hash(
            &format!("tenant|scan|{}|{}|{}", tid, collection, payload_str),
        );

        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, priority, status, payload_json, created_at, updated_at) \
             VALUES (?1, ?2, 'tenant', 'scan', ?3, ?4, 5, 'pending', ?5, ?6, ?7)",
        )
        .bind(uuid::Uuid::new_v4().to_string())
        .bind(&idem_key[..32])
        .bind(tid)
        .bind(collection)
        .bind(&payload_str)
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await;
        enqueued += 1;
    }

    info!("[rebuild:{}] Enqueued {} scan operations for watch folders", collection, enqueued);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_creation() {
        let service = SystemServiceImpl::new();
        assert!(service.start_time <= SystemTime::now());
    }

    #[tokio::test]
    async fn test_default_trait() {
        let _service = SystemServiceImpl::default();
        // Should not panic
    }

    #[tokio::test]
    async fn test_service_with_queue_health() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.set_queue_depth(42);

        let service = SystemServiceImpl::new().with_queue_health(health.clone());
        assert!(service.queue_health.is_some());

        // Test health check includes queue processor
        let response = service.health(Request::new(())).await.unwrap();
        let health_response = response.into_inner();
        assert!(health_response.components.len() >= 2);
        assert!(health_response.components.iter().any(|c| c.component_name == "queue_processor"));
    }

    #[tokio::test]
    async fn test_queue_processor_health_metrics() {
        let health = QueueProcessorHealth::new();

        // Test initial state
        assert!(!health.is_running.load(Ordering::SeqCst));
        assert_eq!(health.error_count.load(Ordering::SeqCst), 0);

        // Test running state
        health.set_running(true);
        assert!(health.is_running.load(Ordering::SeqCst));

        // Test error recording
        health.record_error();
        health.record_error();
        assert_eq!(health.error_count.load(Ordering::SeqCst), 2);

        // Test success recording
        health.record_success(100);
        health.record_success(200);
        assert_eq!(health.items_processed.load(Ordering::SeqCst), 2);
        // Average should be approximately 150
        let avg = health.avg_processing_time_ms.load(Ordering::SeqCst);
        assert!(avg > 0);

        // Test failure recording
        health.record_failure();
        assert_eq!(health.items_failed.load(Ordering::SeqCst), 1);

        // Test queue depth
        health.set_queue_depth(100);
        assert_eq!(health.queue_depth.load(Ordering::SeqCst), 100);
    }

    #[tokio::test]
    async fn test_queue_processor_health_poll_time() {
        let health = QueueProcessorHealth::new();

        // Before any poll, should return MAX
        assert_eq!(health.seconds_since_last_poll(), u64::MAX);

        // After poll, should return small number
        health.record_poll();
        let secs = health.seconds_since_last_poll();
        assert!(secs < 2); // Should be nearly instant
    }

    #[tokio::test]
    async fn test_metrics_include_queue_metrics() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.set_queue_depth(10);
        health.record_success(50);

        let service = SystemServiceImpl::new().with_queue_health(health);
        let response = service.get_metrics(Request::new(())).await.unwrap();
        let metrics = response.into_inner().metrics;

        // Check that queue metrics are present
        let metric_names: Vec<&str> = metrics.iter().map(|m| m.name.as_str()).collect();
        assert!(metric_names.contains(&"queue_pending"));
        assert!(metric_names.contains(&"queue_processed"));
        assert!(metric_names.contains(&"queue_failed"));
        assert!(metric_names.contains(&"queue_processor_running"));

        // Verify values
        let pending = metrics.iter().find(|m| m.name == "queue_pending").unwrap();
        assert_eq!(pending.value, 10.0);

        let running = metrics.iter().find(|m| m.name == "queue_processor_running").unwrap();
        assert_eq!(running.value, 1.0);
    }

    #[tokio::test]
    async fn test_status_includes_queue_depth() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_queue_depth(25);

        let service = SystemServiceImpl::new().with_queue_health(health);
        let response = service.get_status(Request::new(())).await.unwrap();
        let status = response.into_inner();

        assert_eq!(status.metrics.unwrap().pending_operations, 25);
    }

    #[tokio::test]
    async fn test_health_status_degraded_on_high_errors() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(true);
        health.record_poll();

        // Record many errors
        for _ in 0..101 {
            health.record_error();
        }

        let service = SystemServiceImpl::new().with_queue_health(health);
        let response = service.health(Request::new(())).await.unwrap();
        let health_response = response.into_inner();

        let queue_comp = health_response.components.iter()
            .find(|c| c.component_name == "queue_processor")
            .unwrap();
        assert_eq!(queue_comp.status, ServiceStatus::Degraded as i32);
    }

    #[tokio::test]
    async fn test_health_status_unhealthy_when_not_running() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_running(false);

        let service = SystemServiceImpl::new().with_queue_health(health);
        let response = service.health(Request::new(())).await.unwrap();
        let health_response = response.into_inner();

        let queue_comp = health_response.components.iter()
            .find(|c| c.component_name == "queue_processor")
            .unwrap();
        assert_eq!(queue_comp.status, ServiceStatus::Unhealthy as i32);
    }

    #[tokio::test]
    async fn test_get_queue_stats() {
        let health = Arc::new(QueueProcessorHealth::new());
        health.set_queue_depth(15);
        health.record_success(100);
        health.record_success(200);
        health.record_failure();

        let service = SystemServiceImpl::new().with_queue_health(health);
        let response = service.get_queue_stats(Request::new(())).await.unwrap();
        let stats = response.into_inner();

        assert_eq!(stats.pending_count, 15);
        assert_eq!(stats.completed_count, 2);
        assert_eq!(stats.failed_count, 1);
    }

    #[tokio::test]
    async fn test_shutdown() {
        let service = SystemServiceImpl::new();
        let response = service.shutdown(Request::new(())).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_status_store_initialized_empty() {
        let service = SystemServiceImpl::new();
        let store = service.status_store.read().await;
        assert!(store.is_empty());
    }

    #[tokio::test]
    async fn test_notify_server_status_stores_entry() {
        let service = SystemServiceImpl::new();

        let notification = ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: Some("test-project".to_string()),
            project_root: Some("/tmp/test-project".to_string()),
        };

        let response = service.notify_server_status(Request::new(notification)).await;
        assert!(response.is_ok());

        // Verify the entry was stored
        let store = service.status_store.read().await;
        assert_eq!(store.len(), 1);
        let entry = store.get("test-project").unwrap();
        assert_eq!(entry.state, ServerState::Up);
        assert_eq!(entry.project_name.as_deref(), Some("test-project"));
        assert_eq!(entry.project_root.as_deref(), Some("/tmp/test-project"));
    }

    #[tokio::test]
    async fn test_notify_server_status_transitions() {
        let service = SystemServiceImpl::new();

        // First: UP
        let up_notification = ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: Some("my-app".to_string()),
            project_root: Some("/home/user/my-app".to_string()),
        };
        let response = service.notify_server_status(Request::new(up_notification)).await;
        assert!(response.is_ok());

        // Then: DOWN
        let down_notification = ServerStatusNotification {
            state: ServerState::Down as i32,
            project_name: Some("my-app".to_string()),
            project_root: Some("/home/user/my-app".to_string()),
        };
        let response = service.notify_server_status(Request::new(down_notification)).await;
        assert!(response.is_ok());

        // Verify the final state is DOWN
        let store = service.status_store.read().await;
        let entry = store.get("my-app").unwrap();
        assert_eq!(entry.state, ServerState::Down);
    }

    #[tokio::test]
    async fn test_notify_server_status_uses_project_root_as_fallback_key() {
        let service = SystemServiceImpl::new();

        let notification = ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: None,
            project_root: Some("/tmp/fallback".to_string()),
        };
        let response = service.notify_server_status(Request::new(notification)).await;
        assert!(response.is_ok());

        let store = service.status_store.read().await;
        assert!(store.contains_key("/tmp/fallback"));
    }

    #[tokio::test]
    async fn test_notify_server_status_unknown_fallback() {
        let service = SystemServiceImpl::new();

        let notification = ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: None,
            project_root: None,
        };
        let response = service.notify_server_status(Request::new(notification)).await;
        assert!(response.is_ok());

        let store = service.status_store.read().await;
        assert!(store.contains_key("unknown"));
    }

    #[tokio::test]
    async fn test_send_refresh_signal_without_db_pool() {
        // Without a database pool, refresh signal should return Ok but do nothing
        let service = SystemServiceImpl::new();

        let request = RefreshSignalRequest {
            queue_type: QueueType::IngestQueue as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        let response = service.send_refresh_signal(Request::new(request)).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_send_refresh_signal_tools_available() {
        // ToolsAvailable is informational and should always succeed
        let service = SystemServiceImpl::new();

        let request = RefreshSignalRequest {
            queue_type: QueueType::ToolsAvailable as i32,
            lsp_languages: vec!["rust".to_string(), "python".to_string()],
            grammar_languages: vec!["javascript".to_string()],
        };
        let response = service.send_refresh_signal(Request::new(request)).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_send_refresh_signal_unspecified() {
        let service = SystemServiceImpl::new();

        let request = RefreshSignalRequest {
            queue_type: QueueType::Unspecified as i32,
            lsp_languages: vec![],
            grammar_languages: vec![],
        };
        let response = service.send_refresh_signal(Request::new(request)).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_multiple_components_tracked_independently() {
        let service = SystemServiceImpl::new();

        // Register two different components
        let notification1 = ServerStatusNotification {
            state: ServerState::Up as i32,
            project_name: Some("project-a".to_string()),
            project_root: Some("/tmp/a".to_string()),
        };
        let notification2 = ServerStatusNotification {
            state: ServerState::Down as i32,
            project_name: Some("project-b".to_string()),
            project_root: Some("/tmp/b".to_string()),
        };

        service.notify_server_status(Request::new(notification1)).await.unwrap();
        service.notify_server_status(Request::new(notification2)).await.unwrap();

        let store = service.status_store.read().await;
        assert_eq!(store.len(), 2);
        assert_eq!(store.get("project-a").unwrap().state, ServerState::Up);
        assert_eq!(store.get("project-b").unwrap().state, ServerState::Down);
    }

    #[tokio::test]
    async fn test_pause_all_watchers_without_db_pool() {
        let service = SystemServiceImpl::new();
        let response = service.pause_all_watchers(Request::new(())).await;
        assert!(response.is_ok());
    }

    #[tokio::test]
    async fn test_resume_all_watchers_without_db_pool() {
        let service = SystemServiceImpl::new();
        let response = service.resume_all_watchers(Request::new(())).await;
        assert!(response.is_ok());
    }
}
