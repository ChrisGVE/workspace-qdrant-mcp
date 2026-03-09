//! gRPC trait implementation for SystemService
//!
//! Contains all RPC handler methods: Health, GetStatus, GetMetrics, GetQueueStats,
//! Shutdown, SendRefreshSignal, NotifyServerStatus, PauseAllWatchers,
//! ResumeAllWatchers, RebuildIndex.

use std::sync::atomic::Ordering;
use std::time::SystemTime;

use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};
use wqm_common::timestamps;

use crate::proto::{
    system_service_server::SystemService, ComponentHealth, HealthResponse, Metric, MetricsResponse,
    QueueStatsResponse, QueueType, RebuildIndexRequest, RebuildIndexResponse, RefreshSignalRequest,
    ServerState, ServerStatusNotification, ServiceStatus, SystemMetrics, SystemStatusResponse,
};

use super::rebuild;
use super::service_impl::SystemServiceImpl;

#[tonic::async_trait]
impl SystemService for SystemServiceImpl {
    /// Quick health check for monitoring/alerting (spec: Health)
    #[tracing::instrument(skip_all, fields(method = "SystemService.health"))]
    async fn health(&self, _request: Request<()>) -> Result<Response<HealthResponse>, Status> {
        debug!("Health check requested");

        // Build component health list
        let mut components = vec![ComponentHealth {
            component_name: "grpc_server".to_string(),
            status: ServiceStatus::Healthy as i32,
            message: "Running".to_string(),
            last_check: Some(prost_types::Timestamp::from(SystemTime::now())),
        }];

        // Add queue processor health
        let queue_health = self.get_queue_processor_health();
        let queue_status = queue_health.status;
        components.push(queue_health);

        // Determine overall status (worst of all components)
        let overall_status = if components
            .iter()
            .any(|c| c.status == ServiceStatus::Unhealthy as i32)
        {
            ServiceStatus::Unhealthy
        } else if components
            .iter()
            .any(|c| c.status == ServiceStatus::Degraded as i32)
        {
            ServiceStatus::Degraded
        } else if components
            .iter()
            .any(|c| c.status == ServiceStatus::Unspecified as i32)
        {
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
    #[tracing::instrument(skip_all, fields(method = "SystemService.get_queue_stats"))]
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
    #[tracing::instrument(skip_all, fields(method = "SystemService.shutdown"))]
    async fn shutdown(&self, _request: Request<()>) -> Result<Response<()>, Status> {
        warn!("Shutdown requested via gRPC");
        Ok(Response::new(()))
    }

    /// Comprehensive system state snapshot
    #[tracing::instrument(skip_all, fields(method = "SystemService.get_status"))]
    async fn get_status(
        &self,
        _request: Request<()>,
    ) -> Result<Response<SystemStatusResponse>, Status> {
        info!("System status requested");

        // Get queue depth from health state if available
        let pending_operations = self
            .queue_health
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
    #[tracing::instrument(skip_all, fields(method = "SystemService.get_metrics"))]
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
                timestamp: now,
            },
            Metric {
                name: "uptime_seconds".to_string(),
                r#type: "gauge".to_string(),
                labels: std::collections::HashMap::new(),
                value: self
                    .start_time
                    .elapsed()
                    .map(|d| d.as_secs() as f64)
                    .unwrap_or(0.0),
                timestamp: now,
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
    #[tracing::instrument(skip_all, fields(method = "SystemService.send_refresh_signal"))]
    async fn send_refresh_signal(
        &self,
        request: Request<RefreshSignalRequest>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let queue_type = QueueType::try_from(req.queue_type).unwrap_or(QueueType::Unspecified);

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
                self.enqueue_folder_scans(pool).await?;

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
    #[tracing::instrument(skip_all, fields(method = "SystemService.notify_server_status"))]
    async fn notify_server_status(
        &self,
        request: Request<ServerStatusNotification>,
    ) -> Result<Response<()>, Status> {
        let req = request.into_inner();
        let state = ServerState::try_from(req.state).unwrap_or(ServerState::Unspecified);

        self.handle_server_notification(state, req.project_name, req.project_root)
            .await;

        Ok(Response::new(()))
    }

    /// Pause all file watchers (master switch)
    ///
    /// Sets is_paused=1 in watch_folders for all enabled watches and toggles
    /// the shared pause flag so connected FileWatcher instances react immediately.
    #[tracing::instrument(skip_all, fields(method = "SystemService.pause_all_watchers"))]
    async fn pause_all_watchers(&self, _request: Request<()>) -> Result<Response<()>, Status> {
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
             WHERE enabled = 1 AND is_paused = 0",
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
        })
        .to_string();
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, metadata, created_at, updated_at) \
             VALUES (?1, ?2, 'metadata', 'pause', '_system', '_system', 'done', ?3, ?4, ?5)",
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
    #[tracing::instrument(skip_all, fields(method = "SystemService.resume_all_watchers"))]
    async fn resume_all_watchers(&self, _request: Request<()>) -> Result<Response<()>, Status> {
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
             WHERE enabled = 1 AND is_paused = 1",
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
        })
        .to_string();
        let _ = sqlx::query(
            "INSERT OR IGNORE INTO unified_queue \
             (queue_id, idempotency_key, item_type, op, tenant_id, collection, \
              status, metadata, created_at, updated_at) \
             VALUES (?1, ?2, 'metadata', 'resume', '_system', '_system', 'done', ?3, ?4, ?5)",
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

    #[tracing::instrument(skip_all, fields(method = "SystemService.rebuild_index"))]
    async fn rebuild_index(
        &self,
        request: Request<RebuildIndexRequest>,
    ) -> Result<Response<RebuildIndexResponse>, Status> {
        let req = request.into_inner();
        let target = req.target.to_lowercase();

        info!(target = %target, tenant = ?req.tenant_id, "RebuildIndex requested");

        let base_target = target.as_str();

        const VALID_TARGETS: &[&str] = &[
            "tags",
            "search",
            "vocabulary",
            "keywords",
            "rules",
            "projects",
            "libraries",
            "components",
            "all",
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
        let force = req.force.unwrap_or(false);
        let target_owned = base_target.to_string();

        // Spawn rebuild as a background task to avoid gRPC timeout
        tokio::spawn(async move {
            rebuild::dispatch(
                &target_owned,
                hierarchy_builder,
                search_db,
                lexicon_manager,
                storage_client,
                db_pool.as_ref(),
                tenant_id.as_deref(),
                &collection,
                force,
            )
            .await;
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
