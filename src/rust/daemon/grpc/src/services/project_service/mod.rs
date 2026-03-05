//! ProjectService gRPC implementation
//!
//! Handles multi-tenant project lifecycle and session management.
//! Provides 7 RPCs: RegisterProject, DeprioritizeProject, GetProjectStatus,
//! ListProjects, Heartbeat, RenameTenant, DeleteProject
//!
//! LSP Integration:
//! - On RegisterProject: detects project languages and starts LSP servers
//! - On DeprioritizeProject (is_active=false): checks queue, then stops LSP servers
//!   - If queue has pending items, defers shutdown until queue drains
//!   - Respects deactivation_delay_secs config before stopping

mod deactivation;
pub(crate) mod lsp_lifecycle;
mod mutations;
mod queries;
mod registration;

#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use sqlx::SqlitePool;
use tokio::sync::{Notify, RwLock};
use tonic::{Request, Response, Status};
use tracing::{debug, info};

use crate::proto::{
    project_service_server::ProjectService, DeleteProjectRequest, DeleteProjectResponse,
    DeprioritizeProjectRequest, DeprioritizeProjectResponse, GetProjectStatusRequest,
    GetProjectStatusResponse, HeartbeatRequest, HeartbeatResponse, ListProjectsRequest,
    ListProjectsResponse, RegisterProjectRequest, RegisterProjectResponse, RenameTenantRequest,
    RenameTenantResponse,
};

use workspace_qdrant_core::{
    DaemonStateManager, LanguageServerManager, PriorityManager, ProjectLanguageDetector,
    StorageClient,
};

/// Default deactivation delay in seconds (1 minute)
const DEFAULT_DEACTIVATION_DELAY_SECS: u64 = 60;

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
/// - Uses `DaemonStateManager` to propagate `is_active` to `watch_folders`
/// - RegisterProject sets `is_active=true` for project and all submodules
/// - DeprioritizeProject sets `is_active=false` when no sessions remain
/// - Heartbeat updates `last_activity_at` for project and all submodules
pub struct ProjectServiceImpl {
    pub(crate) priority_manager: PriorityManager,
    pub(crate) db_pool: SqlitePool,
    pub(crate) state_manager: DaemonStateManager,
    pub(crate) lsp_manager: Option<Arc<RwLock<LanguageServerManager>>>,
    pub(crate) language_detector: Arc<ProjectLanguageDetector>,
    pub(crate) deactivation_delay_secs: u64,
    pub(crate) pending_shutdowns: Arc<RwLock<HashMap<String, (Instant, bool)>>>,
    pub(crate) watch_refresh_signal: Option<Arc<Notify>>,
    pub(crate) storage: Option<Arc<StorageClient>>,
}

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

    /// Set the watch refresh signal for triggering WatchManager refresh
    pub fn with_watch_refresh_signal(mut self, signal: Arc<Notify>) -> Self {
        self.watch_refresh_signal = Some(signal);
        self
    }

    /// Set the storage client for Qdrant operations (needed for DeleteProject)
    pub fn with_storage(mut self, storage: Arc<StorageClient>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Start the background task that monitors deferred LSP shutdowns
    ///
    /// Spawns an async task that runs every 10 seconds, checking for expired
    /// shutdown entries, verifying the queue is empty, and executing shutdowns.
    pub fn start_deferred_shutdown_monitor(&self) {
        lsp_lifecycle::start_deferred_shutdown_monitor(
            Arc::clone(&self.pending_shutdowns),
            self.db_pool.clone(),
            self.lsp_manager.clone(),
            Arc::clone(&self.language_detector),
        );
    }

    /// Get pending shutdowns (for testing and background task)
    pub async fn get_pending_shutdowns(&self) -> HashMap<String, (Instant, bool)> {
        self.pending_shutdowns.read().await.clone()
    }

    /// Execute shutdown for a project (called by background task when ready)
    pub async fn execute_deferred_shutdown(&self, project_id: &str) -> Result<bool, Status> {
        {
            let shutdowns = self.pending_shutdowns.read().await;
            if !shutdowns.contains_key(project_id) {
                debug!(
                    project_id = project_id,
                    "Shutdown already cancelled or executed"
                );
                return Ok(false);
            }
        }

        let queue_depth = lsp_lifecycle::get_project_queue_depth(&self.db_pool, project_id)
            .await
            .unwrap_or(0);
        if queue_depth > 0 {
            info!(
                project_id = project_id,
                pending_items = queue_depth,
                "Queue not empty, deferring shutdown"
            );
            return Ok(false);
        }

        {
            let mut shutdowns = self.pending_shutdowns.write().await;
            shutdowns.remove(project_id);
        }

        lsp_lifecycle::stop_project_lsp_servers(
            &self.lsp_manager,
            &self.language_detector,
            project_id,
        )
        .await?;
        Ok(true)
    }
}

#[tonic::async_trait]
impl ProjectService for ProjectServiceImpl {
    async fn register_project(
        &self,
        request: Request<RegisterProjectRequest>,
    ) -> Result<Response<RegisterProjectResponse>, Status> {
        let req = request.into_inner();
        self.handle_register_project(req).await.map(Response::new)
    }

    async fn deprioritize_project(
        &self,
        request: Request<DeprioritizeProjectRequest>,
    ) -> Result<Response<DeprioritizeProjectResponse>, Status> {
        let req = request.into_inner();
        self.handle_deprioritize_project(req)
            .await
            .map(Response::new)
    }

    async fn get_project_status(
        &self,
        request: Request<GetProjectStatusRequest>,
    ) -> Result<Response<GetProjectStatusResponse>, Status> {
        let req = request.into_inner();
        self.handle_get_project_status(req).await.map(Response::new)
    }

    async fn list_projects(
        &self,
        request: Request<ListProjectsRequest>,
    ) -> Result<Response<ListProjectsResponse>, Status> {
        let req = request.into_inner();
        self.handle_list_projects(req).await.map(Response::new)
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let req = request.into_inner();
        self.handle_heartbeat(req).await.map(Response::new)
    }

    async fn rename_tenant(
        &self,
        request: Request<RenameTenantRequest>,
    ) -> Result<Response<RenameTenantResponse>, Status> {
        let req = request.into_inner();
        self.handle_rename_tenant(req).await.map(Response::new)
    }

    async fn delete_project(
        &self,
        request: Request<DeleteProjectRequest>,
    ) -> Result<Response<DeleteProjectResponse>, Status> {
        let req = request.into_inner();
        self.handle_delete_project(req).await.map(Response::new)
    }
}
