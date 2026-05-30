//! Session lifecycle: initialization, project registration, and cleanup.
//!
//! Mirrors the exported functions in
//! `src/typescript/mcp-server/src/session-lifecycle.ts`:
//!
//! | Rust function           | TS equivalent           |
//! |-------------------------|-------------------------|
//! | `initialize_session`    | `initializeSession`     |
//! | `register_project`      | `registerProject`       |
//! | `cleanup_session`       | `cleanup`               |
//!
//! # Injectability
//! All daemon I/O is abstracted behind the [`DaemonOps`] trait so unit tests
//! can inject a mock (defined in `lifecycle_tests.rs`) without a live gRPC
//! connection.  Project detection is injectable via a closure (`detect_fn`).
//!
//! # TS divergences
//! - TS `cleanup` calls `healthMonitor.stop()` — the Rust server has no
//!   health-monitor subsystem yet; that call is omitted.
//! - TS `initializeSession` imports `setSessionId` from logger — Rust uses
//!   the tracing subscriber which picks up span fields automatically.
//! - The heartbeat is owned by an `AbortHandle` in Rust rather than a
//!   JS `setInterval` handle.

use std::path::PathBuf;

use tokio::task::AbortHandle;
use tracing::{debug, error, info};

use crate::server_types::SessionState;
use crate::sqlite::manager::StateManager;

use super::project_detect::{detect_branch, detect_project, ProjectInfo};

// ─────────────────────────────────────────────────────────────────────────────
// DaemonOps trait
// ─────────────────────────────────────────────────────────────────────────────

/// Abstraction over the daemon gRPC operations needed by the session lifecycle.
///
/// Production code uses `DaemonClient` (via the blanket impl below).
/// Tests inject a mock (defined in `lifecycle_tests.rs`).
pub trait DaemonOps {
    /// Health-check — used to verify the daemon is reachable.
    /// Mirrors TS `daemonClient.connect()`.
    fn health(&mut self) -> impl std::future::Future<Output = Result<(), String>> + Send;

    /// Register / activate a project.
    /// Returns `(project_id, is_worktree, watch_path, is_active, created)`.
    fn register_project(
        &mut self,
        path: &str,
        project_id: &str,
        name: &str,
        git_remote: Option<&str>,
    ) -> impl std::future::Future<Output = Result<RegisterResponse, String>> + Send;

    /// Send one heartbeat for the given project.
    /// Returns `Ok(acknowledged)`.
    fn heartbeat(
        &mut self,
        project_id: &str,
    ) -> impl std::future::Future<Output = Result<bool, String>> + Send;

    /// Deprioritize the project on session end.
    fn deprioritize_project(
        &mut self,
        project_id: &str,
        watch_path: Option<&str>,
    ) -> impl std::future::Future<Output = Result<(), String>> + Send;
}

/// Response from `DaemonOps::register_project`.
#[derive(Debug, Clone)]
pub struct RegisterResponse {
    pub project_id: String,
    pub is_worktree: bool,
    pub watch_path: Option<String>,
    pub is_active: bool,
    pub created: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// Blanket impl: DaemonClient → DaemonOps
// ─────────────────────────────────────────────────────────────────────────────

use crate::grpc::DaemonClient;
use crate::proto::{DeprioritizeProjectRequest, HeartbeatRequest, RegisterProjectRequest};

impl DaemonOps for DaemonClient {
    async fn health(&mut self) -> Result<(), String> {
        DaemonClient::health(self)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    async fn register_project(
        &mut self,
        path: &str,
        project_id: &str,
        name: &str,
        git_remote: Option<&str>,
    ) -> Result<RegisterResponse, String> {
        let req = RegisterProjectRequest {
            path: path.to_string(),
            project_id: project_id.to_string(),
            name: Some(name.to_string()),
            git_remote: git_remote.map(str::to_string),
            register_if_new: false,
            priority: Some("high".to_string()),
        };
        let resp = DaemonClient::register_project(self, req)
            .await
            .map_err(|e| e.to_string())?;
        Ok(RegisterResponse {
            project_id: resp.project_id,
            is_worktree: resp.is_worktree,
            watch_path: resp.watch_path,
            is_active: resp.is_active,
            created: resp.created,
        })
    }

    async fn heartbeat(&mut self, project_id: &str) -> Result<bool, String> {
        let req = HeartbeatRequest {
            project_id: project_id.to_string(),
        };
        let resp = DaemonClient::heartbeat(self, req)
            .await
            .map_err(|e| e.to_string())?;
        Ok(resp.acknowledged)
    }

    async fn deprioritize_project(
        &mut self,
        project_id: &str,
        watch_path: Option<&str>,
    ) -> Result<(), String> {
        let req = DeprioritizeProjectRequest {
            project_id: project_id.to_string(),
            watch_path: watch_path.map(str::to_string),
        };
        DaemonClient::deprioritize_project(self, req)
            .await
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_session
// ─────────────────────────────────────────────────────────────────────────────

/// Initialize a session: assign UUID, detect project, connect daemon,
/// register project, and start heartbeat.
///
/// `detect_fn` receives `(cwd, state_manager)` and returns
/// `Option<ProjectInfo>`.  Injecting it lets tests supply a pre-built
/// `ProjectInfo` without a real git repository.
///
/// `start_hb_fn` is called after a successful daemon connection to start
/// the heartbeat loop.  In production this wraps
/// `heartbeat::start_heartbeat`; in tests it can be a no-op.
///
/// Mirrors `initializeSession` in `session-lifecycle.ts` (line 50).
pub async fn initialize_session<D, DetectFn, HbFn>(
    state: &mut SessionState,
    daemon: &mut D,
    cwd: &std::path::Path,
    state_manager: &StateManager,
    detect_fn: DetectFn,
    start_hb_fn: HbFn,
) where
    D: DaemonOps,
    DetectFn: FnOnce(&std::path::Path, &StateManager) -> Option<ProjectInfo>,
    HbFn: FnOnce(),
{
    // Assign session UUID.
    // (state.session_id is already set by SessionState::new(); re-assign to
    // match TS behaviour where randomUUID() is called inside initializeSession)
    state.session_id = uuid::Uuid::new_v4();

    debug!(session_id = %state.session_id, "Session start");

    // Detect project.
    apply_project_detection(state, cwd, state_manager, detect_fn);

    // Connect daemon.
    match daemon.health().await {
        Ok(()) => {
            state.daemon_connected = true;
            debug!("Daemon connected");

            // Register project if one was detected.
            if state.project_path.is_some() {
                register_project(state, daemon).await;
            }

            start_hb_fn();
        }
        Err(e) => {
            state.daemon_connected = false;
            error!(error = %e, "Daemon connection failed");
        }
    }
}

/// Apply project detection results to `state`.
fn apply_project_detection<DetectFn>(
    state: &mut SessionState,
    cwd: &std::path::Path,
    state_manager: &StateManager,
    detect_fn: DetectFn,
) where
    DetectFn: FnOnce(&std::path::Path, &StateManager) -> Option<ProjectInfo>,
{
    match detect_fn(cwd, state_manager) {
        Some(info) => {
            state.project_path = Some(info.project_path.clone());
            if let Some(pid) = info.project_id {
                state.project_id = Some(pid);
            }
            state.current_branch = Some(info.branch.clone());
            debug!(
                project_path = %info.project_path.display(),
                branch = %info.branch,
                "Project detected"
            );
        }
        None => {
            state.project_path = Some(cwd.to_path_buf());
            // Parity: TS detectProjectForSession always sets currentBranch via
            // detectCurrentBranch(projectRoot) — never null — even with no
            // registered project (session-lifecycle.ts:45-47). Mirror that so
            // the no-project path yields "default" rather than None.
            state.current_branch = Some(detect_branch(cwd));
            debug!("No project detected; using cwd as project path");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// register_project
// ─────────────────────────────────────────────────────────────────────────────

/// Register the detected project with the daemon and apply the response.
///
/// Only re-activates EXISTING projects (`register_if_new: false`).
/// If the project is not active/not created, logs and returns without error.
///
/// Mirrors `registerProject` + `applyRegistrationResponse` in
/// `session-lifecycle.ts` (lines 103–141).
pub async fn register_project<D: DaemonOps>(state: &mut SessionState, daemon: &mut D) {
    let project_path = match state.project_path.clone() {
        Some(p) => p,
        None => return,
    };

    let git_remote = get_git_remote_for_state(&project_path);
    let project_id = state.project_id.clone().unwrap_or_default();
    let name = project_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    match daemon
        .register_project(
            project_path.to_str().unwrap_or(""),
            &project_id,
            &name,
            git_remote.as_deref(),
        )
        .await
    {
        Ok(resp) => {
            if !resp.is_active && !resp.created {
                info!(
                    project_path = %project_path.display(),
                    project_id = %resp.project_id,
                    "Project not registered with daemon, skipping activation"
                );
                return;
            }
            apply_registration_response(state, &resp);
            debug!(
                project_id = %resp.project_id,
                is_worktree = resp.is_worktree,
                "Project registered"
            );
        }
        Err(e) => {
            error!(
                project_path = %project_path.display(),
                error = %e,
                "Failed to register project"
            );
        }
    }
}

/// Apply daemon registration response to `state`.
///
/// Mirrors `applyRegistrationResponse` in `session-lifecycle.ts` (line 84).
fn apply_registration_response(state: &mut SessionState, resp: &RegisterResponse) {
    // Assign project_id if not already set.
    if !resp.project_id.is_empty() && state.project_id.is_none() {
        state.project_id = Some(resp.project_id.clone());
        debug!(project_id = %resp.project_id, "Project ID assigned by daemon");
    }
    // Handle worktree.
    if resp.is_worktree {
        state.is_worktree = true;
        state.watch_path = resp
            .watch_path
            .as_deref()
            .map(PathBuf::from)
            .or_else(|| state.project_path.clone());
        info!(
            project_path = ?state.project_path,
            watch_path = ?state.watch_path,
            "Registered as worktree"
        );
    }
}

/// Read git remote URL directly from `.git/config`.
fn get_git_remote_for_state(project_path: &std::path::Path) -> Option<String> {
    super::project_detect::get_git_remote_url(project_path)
}

// ─────────────────────────────────────────────────────────────────────────────
// cleanup_session
// ─────────────────────────────────────────────────────────────────────────────

/// Clean up session resources.
///
/// Idempotent — subsequent calls after the first are no-ops (guarded by
/// `state.cleaned`).
///
/// Steps:
/// 1. Stop heartbeat (abort the tokio task).
/// 2. Deprioritize project with the daemon (if connected and registered).
/// 3. Close daemon client.
/// 4. Mark session as cleaned.
///
/// Mirrors `cleanup` in `session-lifecycle.ts` (line 266).
/// Note: the TS `healthMonitor.stop()` call is omitted (no health monitor
/// subsystem yet in the Rust server).
pub async fn cleanup_session<D: DaemonOps>(
    state: &mut SessionState,
    daemon: &mut D,
    hb_handle: Option<AbortHandle>,
) {
    if state.cleaned {
        return;
    }
    state.cleaned = true;

    // Stop heartbeat.
    if let Some(handle) = hb_handle {
        handle.abort();
        debug!("Heartbeat stopped");
    }

    // Deprioritize project.
    if let (Some(project_id), true) = (state.project_id.clone(), state.daemon_connected) {
        let watch_path = if state.is_worktree {
            state
                .watch_path
                .as_deref()
                .and_then(|p| p.to_str())
                .map(str::to_string)
        } else {
            None
        };

        match daemon
            .deprioritize_project(&project_id, watch_path.as_deref())
            .await
        {
            Ok(()) => {
                debug!(project_id = %project_id, "Project deprioritized");
            }
            Err(e) => {
                error!(project_id = %project_id, error = %e, "Failed to deprioritize project");
            }
        }
    }

    // Close daemon (set_connected = false on DaemonClient; no-op for mocks).
    // The trait method `close` is not part of DaemonOps to keep the trait
    // minimal; callers that own a DaemonClient call `.close()` directly.

    debug!(session_id = %state.session_id, "Session end");
}

// ─────────────────────────────────────────────────────────────────────────────
// Default detect_fn adapter
// ─────────────────────────────────────────────────────────────────────────────

/// Production detect function wrapping [`detect_project`].
///
/// Passed as `detect_fn` to [`initialize_session`] in normal operation.
pub fn default_detect_fn(cwd: &std::path::Path, sm: &StateManager) -> Option<ProjectInfo> {
    detect_project(cwd, sm)
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
