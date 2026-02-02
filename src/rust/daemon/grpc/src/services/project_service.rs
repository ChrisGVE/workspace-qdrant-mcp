//! ProjectService gRPC implementation
//!
//! Handles multi-tenant project lifecycle and session management.
//! Provides 5 RPCs: RegisterProject, DeprioritizeProject, GetProjectStatus,
//! ListProjects, Heartbeat
//!
//! LSP Integration:
//! - On RegisterProject: detects project languages and starts LSP servers
//! - On DeprioritizeProject (remaining_sessions=0): checks queue, then stops LSP servers
//!   - If queue has pending items, defers shutdown until queue drains
//!   - Respects deactivation_delay_secs config before stopping

use chrono::Utc;
use sqlx::{SqlitePool, Row};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, info, warn, error};

use crate::proto::{
    project_service_server::ProjectService,
    RegisterProjectRequest, RegisterProjectResponse,
    DeprioritizeProjectRequest, DeprioritizeProjectResponse,
    GetProjectStatusRequest, GetProjectStatusResponse,
    ListProjectsRequest, ListProjectsResponse, ProjectInfo,
    HeartbeatRequest, HeartbeatResponse,
};

use workspace_qdrant_core::{
    PriorityManager,
    LanguageServerManager, Language,
    ProjectLanguageDetector,
};

/// Default heartbeat timeout in seconds
const HEARTBEAT_TIMEOUT_SECS: u64 = 60;

/// ProjectService implementation
///
/// Manages project registration, session tracking, and priority management
/// for the multi-tenant ingestion queue.
///
/// LSP Lifecycle:
/// - Owns a `LanguageServerManager` for per-project LSP servers
/// - Starts LSP servers when a project is registered
/// - Stops LSP servers when a project has no remaining sessions
/// - Checks queue before stopping and respects deactivation delay
pub struct ProjectServiceImpl {
    priority_manager: PriorityManager,
    db_pool: SqlitePool,
    /// Language server manager for per-project LSP lifecycle
    lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    /// Language detector with caching
    language_detector: Arc<ProjectLanguageDetector>,
    /// Deactivation delay in seconds before stopping LSP servers
    deactivation_delay_secs: u64,
    /// Pending shutdowns: project_id -> (scheduled_time, was_queue_checked)
    pending_shutdowns: Arc<RwLock<HashMap<String, (Instant, bool)>>>,
}

/// Default deactivation delay in seconds (1 minute)
const DEFAULT_DEACTIVATION_DELAY_SECS: u64 = 60;

impl ProjectServiceImpl {
    /// Create a new ProjectService with database pool
    pub fn new(db_pool: SqlitePool) -> Self {
        Self {
            priority_manager: PriorityManager::new(db_pool.clone()),
            db_pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create from an existing PriorityManager
    pub fn with_priority_manager(priority_manager: PriorityManager, db_pool: SqlitePool) -> Self {
        Self {
            priority_manager,
            db_pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create with LSP manager for language server lifecycle management
    pub fn with_lsp_manager(
        db_pool: SqlitePool,
        lsp_manager: Arc<RwLock<LanguageServerManager>>,
    ) -> Self {
        Self {
            priority_manager: PriorityManager::new(db_pool.clone()),
            db_pool,
            lsp_manager: Some(lsp_manager),
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: DEFAULT_DEACTIVATION_DELAY_SECS,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
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
            db_pool,
            lsp_manager: Some(lsp_manager),
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        }
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
    /// Called when remaining_sessions reaches 0 but:
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
    /// Register a project for high-priority processing
    ///
    /// Called when MCP server starts for a project. Creates the project if new,
    /// increments session count, and bumps priority to HIGH.
    async fn register_project(
        &self,
        request: Request<RegisterProjectRequest>,
    ) -> Result<Response<RegisterProjectResponse>, Status> {
        let req = request.into_inner();

        // Validate required fields
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }
        if req.path.is_empty() {
            return Err(Status::invalid_argument("path cannot be empty"));
        }

        // Validate project_id format (12-char hex)
        if req.project_id.len() != 12 || !req.project_id.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(Status::invalid_argument(
                "project_id must be a 12-character hexadecimal string"
            ));
        }

        info!(
            "Registering project: id={}, path={}, name={:?}",
            req.project_id, req.path, req.name
        );

        // Check if project exists
        let existing: Option<(i32,)> = sqlx::query_as(
            "SELECT active_sessions FROM projects WHERE project_id = ?1"
        )
            .bind(&req.project_id)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error checking project: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        let (created, active_sessions) = if existing.is_some() {
            // Existing project - register session
            match self.priority_manager.register_session(&req.project_id, "main").await {
                Ok(sessions) => (false, sessions),
                Err(e) => {
                    error!("Failed to register session: {}", e);
                    return Err(Status::internal(format!("Failed to register session: {}", e)));
                }
            }
        } else {
            // New project - insert first
            let now = Utc::now().to_rfc3339();
            let result = sqlx::query(
                r#"
                INSERT INTO projects (
                    project_id, project_name, project_root, priority,
                    active_sessions, git_remote, registered_at, last_active,
                    created_at, updated_at
                ) VALUES (?1, ?2, ?3, 'high', 1, ?4, ?5, ?5, ?5, ?5)
                "#,
            )
            .bind(&req.project_id)
            .bind(&req.name)
            .bind(&req.path)
            .bind(&req.git_remote)
            .bind(&now)
            .execute(&self.db_pool)
            .await;

            match result {
                Ok(_) => {
                    info!("Created new project: {}", req.project_id);
                    (true, 1)
                }
                Err(e) => {
                    error!("Failed to create project: {}", e);
                    return Err(Status::internal(format!("Failed to create project: {}", e)));
                }
            }
        };

        // Cancel any pending deferred shutdown for this project
        if self.cancel_deferred_shutdown(&req.project_id).await {
            debug!(
                project_id = %req.project_id,
                "Cancelled pending deferred shutdown on project reactivation"
            );
        }

        // Start LSP servers for the project (non-blocking, best-effort)
        let project_root = PathBuf::from(&req.path);
        if let Err(e) = self.start_project_lsp_servers(&req.project_id, &project_root).await {
            warn!(
                project_id = %req.project_id,
                error = %e,
                "Failed to start LSP servers (non-critical)"
            );
        }

        let response = RegisterProjectResponse {
            created,
            project_id: req.project_id,
            priority: "high".to_string(),
            active_sessions,
        };

        Ok(Response::new(response))
    }

    /// Deprioritize a project (decrement session count)
    ///
    /// Called when MCP server stops. Decrements session count and demotes
    /// priority to NORMAL when no active sessions remain.
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
            Ok(remaining_sessions) => {
                let new_priority = if remaining_sessions > 0 { "high" } else { "normal" };

                // Handle LSP shutdown when no active sessions remain
                if remaining_sessions == 0 {
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
                    remaining_sessions,
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

        let query = r#"
            SELECT project_id, project_name, project_root, priority,
                   active_sessions, last_active, registered_at, git_remote
            FROM projects
            WHERE project_id = ?1
        "#;

        let row = sqlx::query(query)
            .bind(&req.project_id)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error fetching project: {}", e);
                Status::internal(format!("Database error: {}", e))
            })?;

        if let Some(row) = row {
            use sqlx::Row;

            let last_active_str: Option<String> = row.try_get("last_active")
                .map_err(|e| Status::internal(format!("Failed to get last_active: {}", e)))?;
            let last_active = last_active_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            let registered_at_str: Option<String> = row.try_get("registered_at")
                .map_err(|e| Status::internal(format!("Failed to get registered_at: {}", e)))?;
            let registered_at = registered_at_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            let response = GetProjectStatusResponse {
                found: true,
                project_id: row.try_get("project_id")
                    .map_err(|e| Status::internal(format!("Failed to get project_id: {}", e)))?,
                project_name: row.try_get::<Option<String>, _>("project_name")
                    .map_err(|e| Status::internal(format!("Failed to get project_name: {}", e)))?
                    .unwrap_or_default(),
                project_root: row.try_get("project_root")
                    .map_err(|e| Status::internal(format!("Failed to get project_root: {}", e)))?,
                priority: row.try_get("priority")
                    .map_err(|e| Status::internal(format!("Failed to get priority: {}", e)))?,
                active_sessions: row.try_get("active_sessions")
                    .map_err(|e| Status::internal(format!("Failed to get active_sessions: {}", e)))?,
                last_active,
                registered_at,
                git_remote: row.try_get("git_remote")
                    .map_err(|e| Status::internal(format!("Failed to get git_remote: {}", e)))?,
            };

            Ok(Response::new(response))
        } else {
            let response = GetProjectStatusResponse {
                found: false,
                project_id: req.project_id,
                project_name: String::new(),
                project_root: String::new(),
                priority: String::new(),
                active_sessions: 0,
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

        // Build query based on filters
        let mut query = String::from(
            r#"
            SELECT project_id, project_name, project_root, priority,
                   active_sessions, last_active
            FROM projects
            WHERE 1=1
            "#
        );

        // Add priority filter if specified
        let priority_filter = req.priority_filter.as_deref();
        if let Some(priority) = priority_filter {
            if priority != "all" && !priority.is_empty() {
                query.push_str(" AND priority = ?");
            }
        }

        // Add active_only filter if specified
        if req.active_only {
            query.push_str(" AND active_sessions > 0");
        }

        query.push_str(" ORDER BY priority ASC, last_active DESC");

        // Execute query
        let rows = if let Some(priority) = priority_filter {
            if priority != "all" && !priority.is_empty() {
                sqlx::query(&query)
                    .bind(priority)
                    .fetch_all(&self.db_pool)
                    .await
            } else {
                sqlx::query(&query)
                    .fetch_all(&self.db_pool)
                    .await
            }
        } else {
            sqlx::query(&query)
                .fetch_all(&self.db_pool)
                .await
        }.map_err(|e| {
            error!("Database error listing projects: {}", e);
            Status::internal(format!("Database error: {}", e))
        })?;

        let mut projects = Vec::with_capacity(rows.len());
        for row in rows {
            use sqlx::Row;

            let last_active_str: Option<String> = row.try_get("last_active")
                .map_err(|e| Status::internal(format!("Failed to get last_active: {}", e)))?;
            let last_active = last_active_str
                .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
                .map(|dt| Self::to_timestamp(dt.with_timezone(&Utc)));

            projects.push(ProjectInfo {
                project_id: row.try_get("project_id")
                    .map_err(|e| Status::internal(format!("Failed to get project_id: {}", e)))?,
                project_name: row.try_get::<Option<String>, _>("project_name")
                    .map_err(|e| Status::internal(format!("Failed to get project_name: {}", e)))?
                    .unwrap_or_default(),
                project_root: row.try_get("project_root")
                    .map_err(|e| Status::internal(format!("Failed to get project_root: {}", e)))?,
                priority: row.try_get("priority")
                    .map_err(|e| Status::internal(format!("Failed to get priority: {}", e)))?,
                active_sessions: row.try_get("active_sessions")
                    .map_err(|e| Status::internal(format!("Failed to get active_sessions: {}", e)))?,
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

        // Initialize queue schema (legacy tables for test compatibility)
        sqlx::query(include_str!("../../../core/src/schema/legacy/queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        sqlx::query(include_str!("../../../core/src/schema/legacy/missing_metadata_queue_schema.sql"))
            .execute(&pool)
            .await
            .unwrap();

        // Add projects table schema
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                project_name TEXT,
                project_root TEXT NOT NULL UNIQUE,
                priority TEXT DEFAULT 'normal' CHECK (priority IN ('high', 'normal', 'low')),
                active_sessions INTEGER DEFAULT 0,
                git_remote TEXT,
                registered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        "#)
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
    }

    #[tokio::test]
    async fn test_register_new_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.created);
        assert_eq!(response.project_id, "abcd12345678");
        assert_eq!(response.priority, "high");
        assert_eq!(response.active_sessions, 1);
    }

    #[tokio::test]
    async fn test_register_existing_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // First registration
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
        });
        service.register_project(request).await.unwrap();

        // Second registration - should increment sessions
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("Test Project".to_string()),
            git_remote: None,
        });

        let response = service.register_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(!response.created);
        assert_eq!(response.active_sessions, 2);
    }

    #[tokio::test]
    async fn test_deprioritize_project() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Register first
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
        });
        service.register_project(request).await.unwrap();

        // Deprioritize
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });

        let response = service.deprioritize_project(request).await.unwrap();
        let response = response.into_inner();

        assert!(response.success);
        assert_eq!(response.remaining_sessions, 0);
        assert_eq!(response.new_priority, "normal");
    }

    #[tokio::test]
    async fn test_get_project_status() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Register first
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: Some("My Project".to_string()),
            git_remote: Some("https://github.com/user/repo.git".to_string()),
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
        assert_eq!(response.project_name, "My Project");
        assert_eq!(response.project_root, "/test/project");
        assert_eq!(response.priority, "high");
        assert_eq!(response.active_sessions, 1);
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
        let service = ProjectServiceImpl::new(pool);

        // Register multiple projects with valid 12-char hex IDs
        let project_ids = ["aaa000000001", "bbb000000002", "ccc000000003"];
        for (i, project_id) in project_ids.iter().enumerate() {
            let request = Request::new(RegisterProjectRequest {
                path: format!("/test/project{}", i),
                project_id: project_id.to_string(),
                name: Some(format!("Project {}", i)),
                git_remote: None,
            });
            service.register_project(request).await.unwrap();
        }

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
        let service = ProjectServiceImpl::new(pool);

        // Register project
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
        });
        service.register_project(request).await.unwrap();

        // Deprioritize to make it inactive
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });
        service.deprioritize_project(request).await.unwrap();

        // List active only - should be empty
        let request = Request::new(ListProjectsRequest {
            priority_filter: None,
            active_only: true,
        });

        let response = service.list_projects(request).await.unwrap();
        let response = response.into_inner();

        assert_eq!(response.total_count, 0);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        // Register first
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
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

        // Invalid format - too short
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "short".to_string(),
            name: None,
            git_remote: None,
        });

        let result = service.register_project(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    #[tokio::test]
    async fn test_empty_project_id() {
        let (pool, _temp_dir) = setup_test_db().await;
        let service = ProjectServiceImpl::new(pool);

        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "".to_string(),
            name: None,
            git_remote: None,
        });

        let result = service.register_project(request).await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::InvalidArgument);
    }

    /// Helper to setup database with unified_queue table
    async fn setup_test_db_with_queue() -> (SqlitePool, tempfile::TempDir) {
        let (pool, temp_dir) = setup_test_db().await;

        // Add unified_queue table for queue checking tests
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS unified_queue (
                queue_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                item_type TEXT NOT NULL,
                op TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                collection TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                branch TEXT,
                payload_json TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 3,
                last_error TEXT,
                leased_by TEXT,
                lease_expires_at TEXT
            )
        "#)
            .execute(&pool)
            .await
            .unwrap();

        (pool, temp_dir)
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
            VALUES ('q1', 'key1', 'file', 'ingest', 'test123456ab', 'test-code', 'pending', ?1, ?1),
                   ('q2', 'key2', 'file', 'ingest', 'test123456ab', 'test-code', 'pending', ?1, ?1),
                   ('q3', 'key3', 'file', 'ingest', 'other1234567', 'test-code', 'pending', ?1, ?1),
                   ('q4', 'key4', 'file', 'ingest', 'test123456ab', 'test-code', 'done', ?1, ?1)
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

        // Service with custom delay (default is 60s)
        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 30, // 30 second delay
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register and deprioritize
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
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

        let service = ProjectServiceImpl {
            priority_manager: PriorityManager::new(pool.clone()),
            db_pool: pool,
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 60,
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        };

        // Register project
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
        });
        service.register_project(request).await.unwrap();

        // Deprioritize - should schedule deferred shutdown
        let request = Request::new(DeprioritizeProjectRequest {
            project_id: "abcd12345678".to_string(),
        });
        service.deprioritize_project(request).await.unwrap();

        // Verify shutdown is scheduled
        assert!(service.get_pending_shutdowns().await.contains_key("abcd12345678"));

        // Re-register - should cancel shutdown
        let request = Request::new(RegisterProjectRequest {
            path: "/test/project".to_string(),
            project_id: "abcd12345678".to_string(),
            name: None,
            git_remote: None,
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
            db_pool: pool.clone(),
            lsp_manager: None,
            language_detector: Arc::new(ProjectLanguageDetector::new()),
            deactivation_delay_secs: 0, // No delay
            pending_shutdowns: Arc::new(RwLock::new(HashMap::new())),
        };

        // Add pending queue item
        sqlx::query(r#"
            INSERT INTO unified_queue (queue_id, idempotency_key, item_type, op, tenant_id, collection, status, created_at, updated_at)
            VALUES ('q1', 'key1', 'file', 'ingest', 'abcd12345678', 'test-code', 'pending', ?1, ?1)
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
}
