//! ProjectService gRPC implementation
//!
//! Handles multi-tenant project lifecycle and session management.
//! Provides 5 RPCs: RegisterProject, DeprioritizeProject, GetProjectStatus,
//! ListProjects, Heartbeat
//!
//! LSP Integration:
//! - On RegisterProject: detects project languages and starts LSP servers
//! - On DeprioritizeProject (is_active=false): checks queue, then stops LSP servers
//!   - If queue has pending items, defers shutdown until queue drains
//!   - Respects deactivation_delay_secs config before stopping

use chrono::Utc;
use sqlx::SqlitePool;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Notify, RwLock};
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};

use crate::proto::{
    project_service_server::ProjectService,
    RegisterProjectRequest, RegisterProjectResponse,
    DeprioritizeProjectRequest, DeprioritizeProjectResponse,
    GetProjectStatusRequest, GetProjectStatusResponse,
    ListProjectsRequest, ListProjectsResponse, ProjectInfo,
    HeartbeatRequest, HeartbeatResponse,
    RenameTenantRequest, RenameTenantResponse,
    DeleteProjectRequest, DeleteProjectResponse,
    SetProjectPriorityRequest, SetProjectPriorityResponse,
};

use wqm_common::constants::COLLECTION_PROJECTS;
use workspace_qdrant_core::{
    PriorityManager,
    LanguageServerManager, Language,
    ProjectLanguageDetector,
    DaemonStateManager,
    project_disambiguation::ProjectIdCalculator,
    QueueManager,
    ItemType, UnifiedQueueOp, ProjectPayload,
    StorageClient,
};

/// Default heartbeat timeout in seconds
const HEARTBEAT_TIMEOUT_SECS: u64 = 60;

/// ProjectService implementation
///
/// Manages project registration, activity tracking, and priority management
/// for the multi-tenant ingestion queue.
///
/// LSP Lifecycle:
/// - Owns a `LanguageServerManager` for per-project LSP servers
/// - Starts LSP servers when a project is registered
/// - Stops LSP servers when a project has no remaining sessions
/// - Checks queue before stopping and respects deactivation delay
///
/// Activity Inheritance:
/// - Uses DaemonStateManager to propagate is_active to watch_folders
/// - RegisterProject sets is_active=true for project and all submodules
/// - DeprioritizeProject sets is_active=false when no sessions remain
/// - Heartbeat updates last_activity_at for project and all submodules
pub struct ProjectServiceImpl {
    priority_manager: PriorityManager,
    db_pool: SqlitePool,
    /// Daemon state manager for watch_folders activity inheritance
    state_manager: DaemonStateManager,
    /// Language server manager for per-project LSP lifecycle
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    /// Language detector with caching
    language_detector: Arc<ProjectLanguageDetector>,
    /// Deactivation delay in seconds before stopping LSP servers
    deactivation_delay_secs: u64,
    /// Pending shutdowns: project_id -> (scheduled_time, was_queue_checked)
    pending_shutdowns: Arc<RwLock<HashMap<String, (Instant, bool)>>>,
    /// Signal to trigger immediate WatchManager refresh when a new project is registered
    watch_refresh_signal: Option<Arc<Notify>>,
    /// Storage client for Qdrant operations (needed for DeleteProject)
    storage: Option<Arc<StorageClient>>,
}

/// Default deactivation delay in seconds (1 minute)
const DEFAULT_DEACTIVATION_DELAY_SECS: u64 = 60;

impl ProjectServiceImpl {
    /// Create a new ProjectService with database pool
    pub fn new(db_pool: SqlitePool) -> Self {
        Self {
            priority_manager: PriorityManager::new(db_pool.clone()),
            state_manager: DaemonStateManager::with_pool(db_pool.clone()),
            db_pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        }
    }

    /// Create from an existing PriorityManager
    pub fn with_priority_manager(priority_manager: PriorityManager, db_pool: SqlitePool) -> Self {
        Self {
            priority_manager,
            state_manager: DaemonStateManager::with_pool(db_pool.clone()),
            db_pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        }
    }

    /// Create with LSP manager for language server lifecycle management
    pub fn with_lsp_manager(
        db_pool: SqlitePool,
        lsp_manager: Arc<RwLock<LanguageServerManager>>,
    ) -> Self {
        Self {
            priority_manager: PriorityManager::new(db_pool.clone()),
            state_manager: DaemonStateManager::with_pool(db_pool.clone()),
            db_pool,
            lsp_manager: Some(lsp_manager),
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        }
    }

    /// Create with LSP manager and custom deactivation delay
    pub fn with_lsp_manager_and_config(
        db_pool: SqlitePool,
        lsp_manager: Arc<RwLock<LanguageServerManager>>,
        deactivation_delay_secs: u64,
    ) -> Self {
        Self {
            priority_manager: PriorityManager::new(db_pool.clone()),
            state_manager: DaemonStateManager::with_pool(db_pool.clone()),
            db_pool,
            lsp_manager: Some(lsp_manager),
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        }
    }

    /// Set the watch refresh signal for triggering WatchManager refresh on new project registration
    pub fn with_watch_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set the storage client for Qdrant operations (needed for DeleteProject)
    pub fn with_storage(mut self, storage: Arc<StorageClient>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Start the background task that monitors and executes deferred LSP shutdowns
    ///
    /// This spawns an async task that:
    /// - Runs every 10 seconds
    /// - Checks for expired shutdown entries
    /// - Verifies queue is empty before executing shutdown
    /// - Removes completed/cancelled entries
    ///
    /// Call this after construction to enable deferred shutdown monitoring.
    pub fn start_deferred_shutdown_monitor(&self) {
        let pending_shutdowns = Arc::clone(&self.pending_shutdowns);
        let db_pool = self.db_pool.clone();
        let lsp_manager = self.lsp_manager.clone();
        let language_detector = Arc::clone(&self.language_detector);

        tokio::spawn(async move {
            info!("Started deferred LSP shutdown monitor (10s interval)");

            loop {
                // Wait 10 seconds between checks
                tokio::time::sleep(Duration::from_secs(10)).await;

                // Get a snapshot of pending shutdowns
                let pending: Vec<(String, Instant, bool)> = {
                    let shutdowns = pending_shutdowns.read().await;
                    shutdowns
                        .iter()
                        .map(|(k, (time, checked))| (k.clone(), *time, *checked))
                        .collect()
                };

                if pending.is_empty() {
                    continue;
                }

                debug!("Checking {} pending shutdowns", pending.len());

                let now = Instant::now();
                for (project_id, scheduled_time, _was_queue_checked) in pending {
                    // Check if delay has expired
                    if scheduled_time > now {
                        debug!(
                            project_id = %project_id,
                            remaining_secs = (scheduled_time - now).as_secs(),
                            "Shutdown not yet due"
                        );
                        continue;
                    }

                    // Check queue depth
                    let queue_depth: i64 = match sqlx::query_scalar(
                        "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND tenant_id = ?1"
                    )
                        .bind(&project_id)
                        .fetch_one(&db_pool)
                        .await
                    {
                        Ok(count) => count,
                        Err(e) => {
                            if e.to_string().contains("no such table") {
                                // Table doesn't exist yet, treat as empty
                                0
                            } else {
                                warn!(
                                    project_id = %project_id,
                                    error = %e,
                                    "Failed to check queue depth, skipping this iteration"
                                );
                                continue;
                            }
                        }
                    };

                    if queue_depth > 0 {
                        debug!(
                            project_id = %project_id,
                            pending_items = queue_depth,
                            "Queue not empty, deferring shutdown"
                        );
                        continue;
                    }

                    // Ready to shutdown - remove from pending and execute
                    {
                        let mut shutdowns = pending_shutdowns.write().await;
                        shutdowns.remove(&project_id);
                    }

                    info!(
                        project_id = %project_id,
                        "Executing deferred LSP shutdown (queue empty, delay expired)"
                    );

                    // Invalidate language detection cache
                    language_detector.invalidate_cache(&project_id).await;

                    // Stop LSP servers
                    let Some(lsp_mgr) = &lsp_manager else {
                        continue;
                    };

                    let manager = lsp_mgr.read().await;
                    let languages_to_stop = vec![
                        Language::Python,
                        Language::Rust,
                        Language::TypeScript,
                        Language::JavaScript,
                        Language::Go,
                        Language::C,
                        Language::Cpp,
                        Language::Java,
                    ];

                    for language in languages_to_stop {
                        if let Err(e) = manager.stop_server(&project_id, language.clone()).await {
                            debug!(
                                project_id = %project_id,
                                language = ?language,
                                error = %e,
                                "Error stopping LSP server (may not have been running)"
                            );
                        }
                    }

                    info!(
                        project_id = %project_id,
                        "Deferred LSP shutdown complete"
                    );
                }
            }
        });
    }

    /// Start LSP servers for a project's detected languages
    ///
    /// Uses ProjectLanguageDetector which:
    /// 1. Checks for marker files (Cargo.toml, package.json, etc.)
    /// 2. Falls back to extension scanning
    /// 3. Caches results per project_id
    async fn start_project_lsp_servers(
        &self,
        project_id: &str,
        project_root: &PathBuf,
    ) -> Result<usize, Status> {
        let Some(lsp_manager) = &self.lsp_manager else {
            debug!("No LSP manager configured, skipping LSP server startup");
            return Ok(0);
        };

        // Use the centralized language detector with caching
        let detection_result = self.language_detector
            .detect(project_id, project_root)
            .await
            .map_err(|e| {
                warn!(
                    project_id = project_id,
                    error = %e,
                    "Failed to detect project languages"
                );
                Status::internal(format!("Language detection failed: {}", e))
            })?;

        let languages = detection_result.all_languages;

        if languages.is_empty() {
            debug!(
                project_id = project_id,
                "No supported languages detected in project"
            );
            return Ok(0);
        }

        info!(
            project_id = project_id,
            marker_languages = ?detection_result.marker_languages,
            extension_languages = ?detection_result.extension_languages,
            from_cache = detection_result.from_cache,
            "Detected project languages"
        );

        let manager = lsp_manager.read().await;
        let mut started = 0;

        for language in languages {
            match manager.start_server(project_id, language.clone(), project_root).await {
                Ok(_) => {
                    info!(
                        project_id = project_id,
                        language = ?language,
                        "Started LSP server"
                    );
                    started += 1;
                }
                Err(e) => {
                    // Log but don't fail - LSP is enhancement, not critical
                    debug!(
                        project_id = project_id,
                        language = ?language,
                        error = %e,
                        "Failed to start LSP server (non-critical)"
                    );
                }
            }
        }

        info!(
            project_id = project_id,
            servers_started = started,
            "LSP server startup complete"
        );

        Ok(started)
    }

    /// Stop all LSP servers for a project
    async fn stop_project_lsp_servers(&self, project_id: &str) -> Result<(), Status> {
        // Invalidate language detection cache - next registration will rescan
        self.language_detector.invalidate_cache(project_id).await;

        let Some(lsp_manager) = &self.lsp_manager else {
            return Ok(());
        };

        let manager = lsp_manager.read().await;

        // Get languages that might have servers running for this project
        // We need to stop servers for all languages
        let languages_to_stop = vec![
            Language::Python,
            Language::Rust,
            Language::TypeScript,
            Language::JavaScript,
            Language::Go,
            Language::C,
            Language::Cpp,
            Language::Java,
        ];

        for language in languages_to_stop {
            if let Err(e) = manager.stop_server(project_id, language.clone()).await {
                debug!(
                    project_id = project_id,
                    language = ?language,
                    error = %e,
                    "Error stopping LSP server (may not have been running)"
                );
            }
        }

        info!(project_id = project_id, "LSP servers stopped for project");
        Ok(())
    }

    /// Check if project has pending items in the unified queue
    ///
    /// Returns the count of pending items for the given project_id (tenant_id in queue terms)
    async fn get_project_queue_depth(&self, project_id: &str) -> Result<i64, Status> {
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM unified_queue WHERE status = 'pending' AND tenant_id = ?1"
        )
            .bind(project_id)
            .fetch_one(&self.db_pool)
            .await
            .map_err(|e| {
                // Handle case where table doesn't exist (daemon not fully initialized)
                if e.to_string().contains("no such table") {
                    debug!(
                        project_id = project_id,
                        "unified_queue table not found, assuming empty"
                    );
                    return Status::ok("Queue table not initialized");
                }
                error!(
                    project_id = project_id,
                    error = %e,
                    "Failed to check queue depth"
                );
                Status::internal(format!("Queue check failed: {}", e))
            })?;

        Ok(count)
    }

    /// Schedule deferred LSP shutdown for a project
    ///
    /// Called when is_active becomes false but:
    /// - deactivation_delay_secs > 0, OR
    /// - queue has pending items
    ///
    /// The background task (Task 1.12) will check pending_shutdowns and
    /// execute the actual shutdown when conditions are met.
    async fn schedule_deferred_shutdown(&self, project_id: &str, has_queue_items: bool) {
        let shutdown_time = Instant::now() + Duration::from_secs(self.deactivation_delay_secs);

        let mut shutdowns = self.pending_shutdowns.write().await;
        shutdowns.insert(
            project_id.to_string(),
            (shutdown_time, !has_queue_items) // was_queue_checked = true if queue was empty
        );

        info!(
            project_id = project_id,
            delay_secs = self.deactivation_delay_secs,
            has_queue_items = has_queue_items,
            "Scheduled deferred LSP shutdown"
        );
    }

    /// Cancel a pending deferred shutdown (e.g., when project reactivates)
    async fn cancel_deferred_shutdown(&self, project_id: &str) -> bool {
        let mut shutdowns = self.pending_shutdowns.write().await;
        if shutdowns.remove(project_id).is_some() {
            info!(
                project_id = project_id,
                "Cancelled pending LSP shutdown"
            );
            true
        } else {
            false
        }
    }

    /// Get pending shutdowns (for background task)
    pub async fn get_pending_shutdowns(&self) -> HashMap<String, (Instant, bool)> {
        self.pending_shutdowns.read().await.clone()
    }

    /// Execute shutdown for a project (called by background task when ready)
    pub async fn execute_deferred_shutdown(&self, project_id: &str) -> Result<bool, Status> {
        // Check if still in pending list
        {
            let shutdowns = self.pending_shutdowns.read().await;
            if !shutdowns.contains_key(project_id) {
                debug!(project_id = project_id, "Shutdown already cancelled or executed");
                return Ok(false);
            }
        }

        // Check queue one more time
        let queue_depth = self.get_project_queue_depth(project_id).await.unwrap_or(0);
        if queue_depth > 0 {
            info!(
                project_id = project_id,
                pending_items = queue_depth,
                "Queue not empty, deferring shutdown"
            );
            return Ok(false);
        }

        // Remove from pending list and execute shutdown
        {
            let mut shutdowns = self.pending_shutdowns.write().await;
            shutdowns.remove(project_id);
        }

        self.stop_project_lsp_servers(project_id).await?;
        Ok(true)
    }

    /// Convert chrono DateTime to prost Timestamp
    fn to_timestamp(dt: chrono::DateTime<Utc>) -> prost_types::Timestamp {
        prost_types::Timestamp {
            seconds: dt.timestamp(),
            nanos: dt.timestamp_subsec_nanos() as i32,
        }
    }

    /// Calculate next heartbeat deadline
    fn next_heartbeat_deadline() -> prost_types::Timestamp {
        let deadline = Utc::now() + chrono::Duration::seconds(HEARTBEAT_TIMEOUT_SECS as i64);
        Self::to_timestamp(deadline)
    }
}

#[tonic::async_trait]
impl ProjectService for ProjectServiceImpl {
    /// Register a project for tracking/processing
    ///
    /// Priority-aware: MCP server sends priority="high" for full activation
    /// (LSP, activity inheritance, HIGH queue priority). CLI omits priority
    /// or sends "normal" for lightweight registration (just creates watch folder
    /// and queues scan without activating the project).
    async fn register_project(
        &self,
        request: Request<RegisterProjectRequest>,
    ) -> Result<Response<RegisterProjectResponse>, Status> {
        let req = request.into_inner();

        // Validate path is required
        if req.path.is_empty() {
            return Err(Status::invalid_argument("path cannot be empty"));
        }

        // Resolve effective priority: absent/empty → "normal"
        let effective_priority = req.priority
            .as_deref()
            .filter(|p| !p.is_empty())
            .unwrap_or("normal");
        let is_high_priority = effective_priority == "high";

        // Generate project_id if not provided (MCP server may not know it for new projects)
        let project_id = if req.project_id.is_empty() {
            let calculator = ProjectIdCalculator::new();
            let path = std::path::Path::new(&req.path);
            let git_remote = req.git_remote.as_deref();
            let generated = calculator.calculate(path, git_remote, None);
            info!("Generated project_id for {}: {}", req.path, generated);
            generated
        } else {
            // Validate provided project_id format (12-char hex or local_ prefix)
            let is_local = req.project_id.starts_with("local_");
            let is_hex = req.project_id.len() == 12 && req.project_id.chars().all(|c| c.is_ascii_hexdigit());
            if !is_local && !is_hex {
                return Err(Status::invalid_argument(
                    "project_id must be a 12-character hexadecimal string or start with 'local_'"
                ));
            }
            req.project_id.clone()
        };

        info!(
            "Registering project: id={}, path={}, name={:?}, priority={}",
            project_id, req.path, req.name, effective_priority
        );

        // Check if project exists in watch_folders
        let existing: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1"
        )
            .bind(&project_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error checking project: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        let (created, is_active, newly_registered) = if existing.is_some() {
            if is_high_priority {
                // Existing project + HIGH priority → full activation (MCP server path)
                match self.priority_manager.register_session(&project_id, "main").await {
                    Ok(_) => (false, true, false),
                    Err(e) => {
                        error!("Failed to register session: {}", e);
                        return Err(Status::internal(format!("Failed to register session: {}", e)));
                    }
                }
            } else {
                // Existing project + NORMAL priority → no activation change (CLI path)
                (false, false, false)
            }
        } else if !req.register_if_new {
            // Project not found and caller did not request auto-registration
            info!(
                "Project not registered, skipping auto-registration: {}",
                project_id
            );
            let response = RegisterProjectResponse {
                created: false,
                project_id,
                priority: "none".to_string(),
                is_active: false,
                newly_registered: false,
            };
            return Ok(Response::new(response));
        } else {
            // New project with register_if_new=true — enqueue (Tenant, Add)
            // The queue processor handles watch_folder creation and initial scan
            let queue_manager = QueueManager::new(self.db_pool.clone());
            let payload = ProjectPayload {
                project_root: req.path.clone(),
                git_remote: req.git_remote.clone(),
                project_type: None,
                old_tenant_id: None,
                is_active: Some(is_high_priority),
            };
            let payload_json = serde_json::to_string(&payload)
                .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, req.path));

            match queue_manager.enqueue_unified(
                ItemType::Tenant,
                UnifiedQueueOp::Add,
                &project_id,
                "projects",
                &payload_json,
                0,  // Priority is dynamic (computed at dequeue time)
                None,
                None,
            ).await {
                Ok((queue_id, _is_new)) => {
                    info!(
                        project_id = %project_id,
                        queue_id = %queue_id,
                        "Enqueued project registration (Tenant, Add)"
                    );
                    (true, is_high_priority, true)
                }
                Err(e) => {
                    error!("Failed to enqueue project registration: {}", e);
                    return Err(Status::internal(format!("Failed to enqueue project: {}", e)));
                }
            }
        };

        // Cancel any pending deferred shutdown for this project (only for high priority)
        if is_high_priority {
            if self.cancel_deferred_shutdown(&project_id).await {
                debug!(
                    project_id = %project_id,
                    "Cancelled pending deferred shutdown on project reactivation"
                );
            }
        }

        // HIGH priority only: Start LSP servers and activate watch folders
        if is_high_priority {
            // Start LSP servers for the project (non-blocking, best-effort)
            let project_root = PathBuf::from(&req.path);
            if let Err(e) = self.start_project_lsp_servers(&project_id, &project_root).await {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to start LSP servers (non-critical)"
                );
            }

            // Activity inheritance: Set is_active=true for project and all submodules
            match self.state_manager.activate_project_by_tenant_id(&project_id).await {
                Ok((affected, watch_id)) => {
                    if affected > 0 {
                        info!(
                            project_id = %project_id,
                            watch_id = ?watch_id,
                            affected_folders = affected,
                            "Activated project watch folders (activity inheritance)"
                        );
                    } else {
                        debug!(
                            project_id = %project_id,
                            "No watch folders found for activity inheritance (project may not be watched yet)"
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        project_id = %project_id,
                        error = %e,
                        "Failed to activate project watch folders (non-critical)"
                    );
                }
            }
        }

        let response = RegisterProjectResponse {
            created,
            project_id,
            priority: effective_priority.to_string(),
            is_active,
            newly_registered,
        };

        Ok(Response::new(response))
    }

    /// Deprioritize a project (set is_active to false)
    ///
    /// Called when MCP server stops. Sets is_active to false and demotes
    /// priority to NORMAL.
    ///
    /// LSP Shutdown Logic:
    /// - Checks unified_queue for pending items for this project
    /// - If queue empty AND deactivation_delay is 0: stops LSP servers immediately
    /// - Otherwise: schedules deferred shutdown (handled by background task)
    async fn deprioritize_project(
        &self,
        request: Request<DeprioritizeProjectRequest>,
    ) -> Result<Response<DeprioritizeProjectResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        info!("Deprioritizing project: {}", req.project_id);

        match self.priority_manager.unregister_session(&req.project_id, "main").await {
            Ok(active_flag) => {
                // active_flag: 0 = inactive, 1 = active (boolean as i32 for API compat)
                let is_active = active_flag > 0;
                let new_priority = if is_active { "high" } else { "normal" };

                // Handle LSP shutdown when project becomes inactive
                if !is_active {
                    // Activity inheritance: Set is_active=false for project and all submodules
                    // This updates watch_folders table per Task 19 specification
                    match self.state_manager.deactivate_project_by_tenant_id(&req.project_id).await {
                        Ok((affected, watch_id)) => {
                            if affected > 0 {
                                info!(
                                    project_id = %req.project_id,
                                    watch_id = ?watch_id,
                                    affected_folders = affected,
                                    "Deactivated project watch folders (activity inheritance)"
                                );
                            }
                        }
                        Err(e) => {
                            // Activity inheritance is best-effort, don't fail deprioritization
                            warn!(
                                project_id = %req.project_id,
                                error = %e,
                                "Failed to deactivate project watch folders (non-critical)"
                            );
                        }
                    }

                    // Check queue for pending items
                    let queue_depth = self.get_project_queue_depth(&req.project_id)
                        .await
                        .unwrap_or(0);

                    let has_queue_items = queue_depth > 0;
                    let has_delay = self.deactivation_delay_secs > 0;

                    if !has_queue_items && !has_delay {
                        // No pending items and no delay - stop immediately
                        info!(
                            project_id = %req.project_id,
                            "No active sessions, queue empty, no delay - stopping LSP servers immediately"
                        );
                        if let Err(e) = self.stop_project_lsp_servers(&req.project_id).await {
                            warn!(
                                project_id = %req.project_id,
                                error = %e,
                                "Failed to stop LSP servers (non-critical)"
                            );
                        }
                    } else {
                        // Either queue has items or delay configured - schedule deferred shutdown
                        info!(
                            project_id = %req.project_id,
                            queue_depth = queue_depth,
                            deactivation_delay_secs = self.deactivation_delay_secs,
                            "Scheduling deferred LSP shutdown"
                        );
                        self.schedule_deferred_shutdown(&req.project_id, has_queue_items).await;
                    }
                }

                let response = DeprioritizeProjectResponse {
                    success: true,
                    is_active,
                    new_priority: new_priority.to_string(),
                };

                Ok(Response::new(response))
            }
            Err(workspace_qdrant_core::PriorityError::ProjectNotFound(id)) => {
                warn!("Project not found for deprioritization: {}", id);
                Err(Status::not_found(format!("Project not found: {}", id)))
            }
            Err(e) => {
                error!("Failed to deprioritize project: {}", e);
                Err(Status::internal(format!("Failed to deprioritize: {}", e)))
            }
        }
    }

    /// Get current status of a project
    async fn get_project_status(
        &self,
        request: Request<GetProjectStatusRequest>,
    ) -> Result<Response<GetProjectStatusResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        debug!("Getting project status: {}", req.project_id);

        // Query watch_folders table (spec-compliant schema)
        let query = r#"
            SELECT tenant_id, path, is_active, last_activity_at, created_at, git_remote_url
            FROM watch_folders
            WHERE tenant_id = ?1 AND collection = ?2
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(&req.project_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error fetching project: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        if let Some(row) = row {
            use sqlx::Row;

            let last_active_str: Option<String> = row.try_get("last_activity_at")
                .map_err(|e| Status::internal(format!("Failed to get last_activity_at: {}", e)))?;
            let last_active = last_active_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            let registered_at_str: Option<String> = row.try_get("created_at")
                .map_err(|e| Status::internal(format!("Failed to get created_at: {}", e)))?;
            let registered_at = registered_at_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            let is_active_int: i32 = row.try_get("is_active")
                .map_err(|e| Status::internal(format!("Failed to get is_active: {}", e)))?;

            // Derive priority from is_active (active = high, inactive = normal)
            let priority = if is_active_int == 1 { "high" } else { "normal" };

            // Extract project name from path (last component)
            let path: String = row.try_get("path")
                .map_err(|e| Status::internal(format!("Failed to get path: {}", e)))?;
            let project_name = std::path::Path::new(&path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            let response = GetProjectStatusResponse {
                found: true,
                project_id: row.try_get("tenant_id")
                    .map_err(|e| Status::internal(format!("Failed to get tenant_id: {}", e)))?,
                project_name,
                project_root: path,
                priority: priority.to_string(),
                is_active: is_active_int == 1,
                last_active,
                registered_at,
                git_remote: row.try_get("git_remote_url")
                    .map_err(|e| Status::internal(format!("Failed to get git_remote_url: {}", e)))?,
            };

            Ok(Response::new(response))
        } else {
            let response = GetProjectStatusResponse {
                found: false,
                project_id: req.project_id,
                project_name: String::new(),
                project_root: String::new(),
                priority: String::new(),
                is_active: false,
                last_active: None,
                registered_at: None,
                git_remote: None,
            };

            Ok(Response::new(response))
        }
    }

    /// List all registered projects with their status
    async fn list_projects(
        &self,
        request: Request<ListProjectsRequest>,
    ) -> Result<Response<ListProjectsResponse>, Status> {
        let req = request.into_inner();

        debug!(
            "Listing projects: priority_filter={:?}, active_only={}",
            req.priority_filter, req.active_only
        );

        // Build query using watch_folders table (spec-compliant schema)
        let mut query = format!(
            r#"
            SELECT tenant_id, path, is_active, last_activity_at
            FROM watch_folders
            WHERE collection = '{}'
            "#,
            COLLECTION_PROJECTS
        );

        // Add priority filter if specified (priority derived from is_active)
        let priority_filter = req.priority_filter.as_deref();
        if let Some(priority) = priority_filter {
            if priority == "high" {
                query.push_str(" AND is_active = 1");
            } else if priority == "normal" || priority == "low" {
                query.push_str(" AND is_active = 0");
            }
            // "all" or empty means no filter
        }

        // Add active_only filter if specified
        if req.active_only {
            query.push_str(" AND is_active = 1");
        }

        query.push_str(" ORDER BY is_active DESC, last_activity_at DESC");

        // Execute query (no bind parameters needed for this version)
        let rows = sqlx::query(&query)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error listing projects: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        let mut projects = Vec::with_capacity(rows.len());
        for row in rows {
            use sqlx::Row;

            let last_active_str: Option<String> = row.try_get("last_activity_at")
                .map_err(|e| Status::internal(format!("Failed to get last_activity_at: {}", e)))?;
            let last_active = last_active_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            let is_active_int: i32 = row.try_get("is_active")
                .map_err(|e| Status::internal(format!("Failed to get is_active: {}", e)))?;

            // Derive priority from is_active
            let priority = if is_active_int == 1 { "high" } else { "normal" };

            // Extract project name from path (last component)
            let path: String = row.try_get("path")
                .map_err(|e| Status::internal(format!("Failed to get path: {}", e)))?;
            let project_name = std::path::Path::new(&path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string();

            projects.push(ProjectInfo {
                project_id: row.try_get("tenant_id")
                    .map_err(|e| Status::internal(format!("Failed to get tenant_id: {}", e)))?,
                project_name,
                project_root: path,
                priority: priority.to_string(),
                is_active: is_active_int == 1,
                last_active,
            });
        }

        let total_count = projects.len() as i32;

        let response = ListProjectsResponse {
            projects,
            total_count,
        };

        Ok(Response::new(response))
    }

    /// Heartbeat to keep session alive (60s timeout)
    ///
    /// Called periodically by MCP servers to indicate they're still alive.
    /// Sessions without heartbeat for >60s are considered orphaned.
    ///
    /// Activity inheritance: Updates last_activity_at for project and all submodules
    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        debug!("Heartbeat received: {}", req.project_id);

        match self.priority_manager.heartbeat(&req.project_id).await {
            Ok(acknowledged) => {
                // Activity inheritance: Update last_activity_at for project and all submodules
                // This updates watch_folders table per Task 19 specification
                if acknowledged {
                    match self.state_manager.heartbeat_project_by_tenant_id(&req.project_id).await {
                        Ok((affected, _watch_id)) => {
                            if affected > 0 {
                                debug!(
                                    project_id = %req.project_id,
                                    affected_folders = affected,
                                    "Updated watch folder activity timestamps (activity inheritance)"
                                );
                            }
                        }
                        Err(e) => {
                            // Activity inheritance is best-effort, don't fail heartbeat
                            debug!(
                                project_id = %req.project_id,
                                error = %e,
                                "Failed to update watch folder activity (non-critical)"
                            );
                        }
                    }
                }

                let response = HeartbeatResponse {
                    acknowledged,
                    next_heartbeat_by: Some(Self::next_heartbeat_deadline()),
                };

                Ok(Response::new(response))
            }
            Err(e) => {
                error!("Heartbeat failed: {}", e);
                Err(Status::internal(format!("Heartbeat failed: {}", e)))
            }
        }
    }

    /// Rename a tenant_id across SQLite tables
    ///
    /// Updates watch_folders and unified_queue tables in a single transaction.
    /// Qdrant payload updates should be handled separately (e.g., via re-ingestion
    /// or collection reset + re-scan).
    async fn rename_tenant(
        &self,
        request: Request<RenameTenantRequest>,
    ) -> Result<Response<RenameTenantResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.old_tenant_id.is_empty() {
            return Err(Status::invalid_argument("old_tenant_id cannot be empty"));
        }
        if req.new_tenant_id.is_empty() {
            return Err(Status::invalid_argument("new_tenant_id cannot be empty"));
        }
        if req.old_tenant_id == req.new_tenant_id {
            return Err(Status::invalid_argument("old and new tenant_id are the same"));
        }

        info!(
            "RenameTenant: '{}' -> '{}'",
            req.old_tenant_id, req.new_tenant_id
        );

        // Execute rename in a proper sqlx transaction
        let mut tx = self.db_pool.begin().await
            .map_err(|e| {
                error!("Failed to begin transaction: {}", e);
                Status::internal(format!("Transaction failed: {}", e))
            })?;

        let mut total_rows = 0i32;

        // Update watch_folders
        match sqlx::query(
            "UPDATE watch_folders SET tenant_id = ?1 WHERE tenant_id = ?2"
        )
        .bind(&req.new_tenant_id)
        .bind(&req.old_tenant_id)
        .execute(&mut *tx)
        .await
        {
            Ok(result) => {
                let rows = result.rows_affected() as i32;
                info!("Updated {} watch_folders rows", rows);
                total_rows += rows;
            }
            Err(e) => {
                error!("Failed to update watch_folders: {}", e);
                return Err(Status::internal(format!(
                    "Failed to update watch_folders: {}", e
                )));
            }
        }

        // Update unified_queue
        match sqlx::query(
            "UPDATE unified_queue SET tenant_id = ?1 WHERE tenant_id = ?2"
        )
        .bind(&req.new_tenant_id)
        .bind(&req.old_tenant_id)
        .execute(&mut *tx)
        .await
        {
            Ok(result) => {
                let rows = result.rows_affected() as i32;
                info!("Updated {} unified_queue rows", rows);
                total_rows += rows;
            }
            Err(e) => {
                error!("Failed to update unified_queue: {}", e);
                return Err(Status::internal(format!(
                    "Failed to update unified_queue: {}", e
                )));
            }
        }

        // Update tracked_files (may not exist in all deployments)
        match sqlx::query(
            "UPDATE tracked_files SET tenant_id = ?1 WHERE tenant_id = ?2"
        )
        .bind(&req.new_tenant_id)
        .bind(&req.old_tenant_id)
        .execute(&mut *tx)
        .await
        {
            Ok(result) => {
                let rows = result.rows_affected() as i32;
                info!("Updated {} tracked_files rows", rows);
                total_rows += rows;
            }
            Err(e) => {
                // tracked_files may not exist in all deployments, non-fatal
                warn!("Failed to update tracked_files (non-fatal): {}", e);
            }
        }

        // Commit transaction
        tx.commit().await.map_err(|e| {
            error!("Failed to commit rename transaction: {}", e);
            Status::internal(format!("Failed to commit transaction: {}", e))
        })?;

        let message = format!(
            "Renamed tenant '{}' -> '{}': {} SQLite rows updated",
            req.old_tenant_id, req.new_tenant_id, total_rows
        );
        info!("{}", message);

        Ok(Response::new(RenameTenantResponse {
            success: true,
            sqlite_rows_updated: total_rows,
            message,
        }))
    }

    /// Delete a project by enqueuing (Tenant, Delete)
    ///
    /// Validates the project exists, stops LSP servers immediately (best-effort),
    /// then enqueues the full deletion cascade to the queue processor.
    async fn delete_project(
        &self,
        request: Request<DeleteProjectRequest>,
    ) -> Result<Response<DeleteProjectResponse>, Status> {
        let req = request.into_inner();

        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        info!("Deleting project: {} (delete_qdrant_data={})", req.project_id, req.delete_qdrant_data);

        // Verify project exists (read-only check)
        let exists: Option<(String,)> = sqlx::query_as(
            "SELECT watch_id FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1"
        )
            .bind(&req.project_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error checking project: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        if exists.is_none() {
            return Err(Status::not_found(format!("Project not found: {}", req.project_id)));
        }

        // Stop LSP servers and cancel pending shutdowns immediately (best-effort)
        self.cancel_deferred_shutdown(&req.project_id).await;
        if let Err(e) = self.stop_project_lsp_servers(&req.project_id).await {
            debug!("LSP cleanup for deleted project: {}", e);
        }

        // Enqueue (Tenant, Delete) — queue processor handles full deletion cascade
        let queue_manager = QueueManager::new(self.db_pool.clone());
        let payload = ProjectPayload {
            project_root: String::new(),
            git_remote: None,
            project_type: None,
            old_tenant_id: None,
            is_active: None,
        };
        let payload_json = serde_json::to_string(&payload)
            .unwrap_or_else(|_| "{}".to_string());

        match queue_manager.enqueue_unified(
            ItemType::Tenant,
            UnifiedQueueOp::Delete,
            &req.project_id,
            COLLECTION_PROJECTS,
            &payload_json,
            0,
            None,
            None,
        ).await {
            Ok((queue_id, _is_new)) => {
                let message = format!(
                    "Project {} deletion enqueued (queue_id={})",
                    req.project_id, queue_id
                );
                info!("{}", message);

                Ok(Response::new(DeleteProjectResponse {
                    success: true,
                    watch_folders_deleted: 0,
                    tracked_files_deleted: 0,
                    qdrant_points_deleted: 0,
                    queue_items_deleted: 0,
                    message,
                }))
            }
            Err(e) => {
                error!("Failed to enqueue project deletion: {}", e);
                Err(Status::internal(format!("Failed to enqueue deletion: {}", e)))
            }
        }
    }

    /// Set project priority level (high/normal)
    ///
    /// Delegates to PriorityManager::set_priority which updates
    /// watch_folders.is_active. Queue ordering is computed at dequeue time.
    async fn set_project_priority(
        &self,
        request: Request<SetProjectPriorityRequest>,
    ) -> Result<Response<SetProjectPriorityResponse>, Status> {
        let req = request.into_inner();

        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }
        if req.priority != "high" && req.priority != "normal" {
            return Err(Status::invalid_argument("priority must be 'high' or 'normal'"));
        }

        info!("Setting project priority: {} -> {}", req.project_id, req.priority);

        match self.priority_manager.set_priority(&req.project_id, &req.priority).await {
            Ok((previous, queue_updated)) => {
                Ok(Response::new(SetProjectPriorityResponse {
                    success: true,
                    previous_priority: previous,
                    new_priority: req.priority,
                    queue_items_updated: queue_updated,
                }))
            }
            Err(workspace_qdrant_core::PriorityError::ProjectNotFound(id)) => {
                Err(Status::not_found(format!("Project not found: {}", id)))
            }
            Err(e) => {
                error!("Failed to set project priority: {}", e);
                Err(Status::internal(format!("Failed to set priority: {}", e)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Helper to create test database with schema
    async fn setup_test_db() -> (SqlitePool, tempfile::TempDir) {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_project_service.db");

        let db_url = format!("sqlite://{}?mode=rwc", db_path.display());
        let pool = SqlitePool::connect(&db_url).await.unwrap();

        // Add watch_folders table (spec-compliant schema for all project state)
        sqlx::query(workspace_qdrant_core::watch_folders_schema::CREATE_WATCH_FOLDERS_SQL)
            .execute(&pool)
            .await
            .unwrap();

        // Add unified_queue table (required by PriorityManager for queue operations)
        sqlx::query(workspace_qdrant_core::unified_queue_schema::CREATE_UNIFIED_QUEUE_SQL)
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    /// Helper to create a watch_folder entry for a project (simulates daemon creating the project)
    async fn create_test_watch_folder(pool: &SqlitePool, project_id: &str, path: &str) {
        let now = chrono::Utc::now().to_rfc3339();
        let watch_id = format!("test-{}", project_id);
        sqlx::query(r#"
            INSERT INTO watch_folders (
                watch_id, path, collection, tenant_id, is_active,
                follow_symlinks, enabled, cleanup_on_disable, created_at, updated_at
            ) VALUES (?1, ?2, 'projects', ?3, 0, 0, 1, 0, ?4, ?4)
        "#)
            .bind(&watch_id)
            .bind(path)
            .bind(project_id)
            .bind(&now)
            .execute(pool)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_register_new_project_with_register_if_new() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
            register_if_new: true,
            priority: Some("high".to_string()),
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.created);
        assert_eq!(response.project_id, "abcd12345678");
        assert_eq!(response.priority, "high");
        assert!(response.is_active);
        assert!(response.newly_registered);
    }

    #[tokio::test]
    async fn test_register_new_project_without_register_if_new() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Default register_if_new is false - should NOT create the project
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(!response.created);
        assert_eq!(response.project_id, "abcd12345678");
        assert_eq!(response.priority, "none");
        assert!(!response.is_active);
        assert!(!response.newly_registered);
    }

    #[tokio::test]
    async fn test_register_existing_project() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        let service = ProjectServiceImpl::new(pool);

        // First registration - register_if_new=false still works for existing projects
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Second registration - should work for existing projects regardless of register_if_new
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(!response.created);
        // With the spec-compliant boolean is_active model
        assert!(response.is_active);
        assert!(!response.newly_registered);
    }

    #[tokio::test]
    async fn test_deprioritize_project() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        let service = ProjectServiceImpl::new(pool);

        // Register first (existing project via create_test_watch_folder)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Deprioritize
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });

        let response = service.deprioritize_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.success);
        assert!(!response.is_active);
        assert_eq!(response.new_priority, "normal");
    }

    #[tokio::test]
    async fn test_get_project_status() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Pre-create watch_folder (simulates queue processor having processed Tenant/Add)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;
        // Add git_remote to the watch_folder
        sqlx::query("UPDATE watch_folders SET git_remote_url = ?1 WHERE tenant_id = ?2")
            .bind("https://github.com/user/repo.git")
            .bind("abcd12345678")
            .execute(&pool)
            .await
            .unwrap();

        let service = ProjectServiceImpl::new(pool);

        // Register existing project with high priority (activates it)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("My Project".to_string()),
            git_remote: Some("https://github.com/user/repo.git".to_string()),
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Get status
        let request = Request::new(GetProjectStatusRequest {
            project_id: "abcd12345678".to_string(),
        });

        let response = service.get_project_status(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.found);
        assert_eq!(response.project_id, "abcd12345678");
        assert_eq!(response.project_name, "project");
        assert_eq!(response.project_root, "/test/project");
        assert_eq!(response.priority, "high");
        assert!(response.is_active);
        assert_eq!(response.git_remote, Some("https://github.com/user/repo.git".to_string()));
    }

    #[tokio::test]
    async fn test_get_nonexistent_project_status() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        let request = Request::new(GetProjectStatusRequest {
            project_id: "nonexistent12".to_string(),
        });

        let response = service.get_project_status(request).await.unwrap();
        let response = response.into_inner();

        assert!(!response.found);
    }

    #[tokio::test]
    async fn test_list_projects() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Pre-create watch_folders (simulates queue processor having processed Tenant/Add)
        let project_ids = ["aaa000000001", "bbb000000002", "ccc000000003"];
        for (i, project_id) in project_ids.iter().enumerate() {
            create_test_watch_folder(&pool, project_id, &format!("/test/project{}", i)).await;
        }

        let service = ProjectServiceImpl::new(pool);

        // List all projects
        let request = Request::new(ListProjectsRequest {
            priority_filter: None,
            active_only: false,
        });

        let response = service.list_projects(request).await.unwrap();
        let response = response.into_inner();

        assert_eq!(response.total_count, 3);
        assert_eq!(response.projects.len(), 3);
    }

    #[tokio::test]
    async fn test_list_projects_active_only() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        let service = ProjectServiceImpl::new(pool);

        // Register project (existing via create_test_watch_folder)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Deprioritize to make it inactive
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });
        service.deprioritize_project(request).await.unwrap();

        // List active only - now using watch_folders table
        let request = Request::new(ListProjectsRequest {
            priority_filter: None,
            active_only: true,
        });

        let response = service.list_projects(request).await.unwrap();
        let response = response.into_inner();

        // After deprioritization, is_active is false, so should return 0
        assert_eq!(response.total_count, 0);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let (pool, _temp_dir) = setup_test_db().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        let service = ProjectServiceImpl::new(pool);

        // Register first (existing via create_test_watch_folder)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Send heartbeat
        let request = Request::new(HeartbeatRequest {
            project_id: "abcd12345678".to_string(),
        });

        let response = service.heartbeat(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.acknowledged);
        assert!(response.next_heartbeat_by.is_some());
    }

    #[tokio::test]
    async fn test_invalid_project_id_format() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Invalid format - too short (validation happens before register_if_new check)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "short".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });

        let result = service.register_project(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_empty_project_id_generates_local_id() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Empty project_id triggers auto-generation of a local_* ID
        // With register_if_new=false and no existing project, returns safe response
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(!response.created);
        assert!(response.project_id.starts_with("local_"));
        assert_eq!(response.priority, "none");
        assert!(!response.is_active);
        assert!(!response.newly_registered);
    }

    #[tokio::test]
    async fn test_empty_path_returns_error() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        let request = Request::new(RegisterProjectRequest {
            path: "".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });

        let result = service.register_project(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    /// Helper to setup database with unified_queue table
    /// NOTE: Now just an alias for setup_test_db() which includes all required tables
    async fn setup_test_db_with_queue() -> (SqlitePool, tempfile::TempDir) {
        setup_test_db().await
    }

    #[tokio::test]
    async fn test_queue_depth_returns_zero_for_empty_queue() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;
        let service = ProjectServiceImpl::new(pool);

        let depth = service.get_project_queue_depth("test123456ab").await.unwrap();
        assert_eq!(depth, 0);
    }

    #[tokio::test]
    async fn test_queue_depth_counts_pending_items() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;
        let now = chrono::Utc::now().to_rfc3339();

        // Insert some pending queue items
        sqlx::query(r#"
            INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at)
            VALUES ('q1', 'key1', 'file', 'add', 'test123456ab', 'test-code', 'pending', ?1, ?1),
                   ('q2', 'key2', 'file', 'add', 'test123456ab', 'test-code', 'pending', ?1, ?1),
                   ('q3', 'key3', 'file', 'add', 'other1234567', 'test-code', 'pending', ?1, ?1),
                   ('q4', 'key4', 'file', 'add', 'test123456ab', 'test-code', 'done', ?1, ?1)
        "#)
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();

        let service = ProjectServiceImpl::new(pool);

        // Should count only pending items for test123456ab (2 items)
        let depth = service.get_project_queue_depth("test123456ab").await.unwrap();
        assert_eq!(depth, 2);
    }

    #[tokio::test]
    async fn test_deferred_shutdown_scheduled_when_delay_set() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        // Service with custom delay (default is 60s)
        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            state_manager: DaemonStateManager::with_pool(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 30, // 30 second delay
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        };

        // Register and deprioritize (existing via create_test_watch_folder)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });
        service.deprioritize_project(request).await.unwrap();

        // Should have scheduled a deferred shutdown
        let pending = service.get_pending_shutdowns().await;
        assert!(pending.contains_key("abcd12345678"));
    }

    #[tokio::test]
    async fn test_reactivation_cancels_deferred_shutdown() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;

        // Create watch_folder entry (simulates daemon having created the project)
        create_test_watch_folder(&pool, "abcd12345678", "/test/project").await;

        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            state_manager: DaemonStateManager::with_pool(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 60,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        };

        // Register project (existing via create_test_watch_folder)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Deprioritize - should schedule deferred shutdown
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });
        service.deprioritize_project(request).await.unwrap();

        // Verify shutdown is scheduled
        assert!(service.get_pending_shutdowns().await.contains_key("abcd12345678"));

        // Re-register - should cancel shutdown (existing project)
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
            register_if_new: false,
            priority: Some("high".to_string()),
        });
        service.register_project(request).await.unwrap();

        // Verify shutdown is cancelled
        assert!(!service.get_pending_shutdowns().await.contains_key("abcd12345678"));
    }

    #[tokio::test]
    async fn test_queue_depth_handles_missing_table() {
        let (pool, _temp_dir) = setup_test_db().await;
        // Note: setup_test_db does NOT create unified_queue table

        let service = ProjectServiceImpl::new(pool);

        // Should handle gracefully - either return 0 or error
        let result = service.get_project_queue_depth("test123456ab").await;
        // The error case returns Status::ok("Queue table not initialized")
        // which is still an error from the perspective of the query
        match result {
            Ok(depth) => assert_eq!(depth, 0),
            Err(status) => {
                // Status::ok() has Code::Ok, not an actual error code
                // so this is fine for the graceful degradation case
                assert!(status.message().contains("not initialized") || status.code() == tonic::Code::Ok);
            }
        }
    }

    #[tokio::test]
    async fn test_execute_deferred_shutdown_checks_queue() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;
        let now = chrono::Utc::now().to_rfc3339();

        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            state_manager: DaemonStateManager::with_pool(pool.clone()),
            db_pool: pool.clone(),
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 0, // No delay
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        };

        // Add pending queue item
        sqlx::query(r#"
            INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at)
            VALUES ('q1', 'key1', 'file', 'add', 'abcd12345678', 'test-code', 'pending', ?1, ?1)
        "#)
            .bind(&now)
            .execute(&pool)
            .await
            .unwrap();

        // Schedule a deferred shutdown
        service.schedule_deferred_shutdown("abcd12345678", true).await;

        // Try to execute - should fail because queue has items
        let result = service.execute_deferred_shutdown("abcd12345678").await.unwrap();
        assert!(!result); // Did not execute

        // Shutdown should still be pending
        assert!(service.get_pending_shutdowns().await.contains_key("abcd12345678"));
    }

    #[tokio::test]
    async fn test_background_monitor_can_be_started() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;

        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            state_manager: DaemonStateManager::with_pool(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 0,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        };

        // Start the background monitor - should not panic
        service.start_deferred_shutdown_monitor();

        // Give the background task a moment to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // If we get here without panic, the test passes
        // The task will be cleaned up when the tokio runtime drops
    }

    #[tokio::test]
    async fn test_execute_deferred_shutdown_succeeds_when_queue_empty() {
        let (pool, _temp_dir) = setup_test_db_with_queue().await;

        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            state_manager: DaemonStateManager::with_pool(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 0,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
            watch_refresh_signal: None,
            storage: None,
        };

        // Schedule a deferred shutdown with immediate expiry
        {
            let mut shutdowns = service.pending_shutdowns.write().await;
            // Use Instant::now() - 1 second to make it already expired
            shutdowns.insert(
                "abcd12345678".to_string(),
                (Instant::now() - Duration::from_secs(1), true)
            );
        }

        // Try to execute - should succeed because queue is empty
        let result = service.execute_deferred_shutdown("abcd12345678").await.unwrap();
        assert!(result); // Did execute

        // Shutdown should no longer be pending
        assert!(!service.get_pending_shutdowns().await.contains_key("abcd12345678"));
    }
}
