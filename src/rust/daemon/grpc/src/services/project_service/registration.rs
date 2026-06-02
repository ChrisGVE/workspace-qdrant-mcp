//! Project registration logic
//!
//! Handles the RegisterProject gRPC flow: validation, project ID generation,
//! queue enqueue for new projects, session registration for existing projects,
//! LSP server startup, and activity inheritance.
//!
//! # Path normalization (F-019, spec §16)
//!
//! Project paths are normalized to their syntactic-canonical UTF-8 form
//! before persistence or enqueue (spec §3.1 rules; see
//! `wqm_common::paths::CanonicalPath`). Symlinks are **not** resolved
//! (rule 7), and the stored path is the user-supplied path after
//! tilde-expansion, `.`-collapse, and duplicate-slash removal.
//! Nonexistent paths and non-directory paths are still rejected with
//! `INVALID_ARGUMENT` via a syntactic check followed by an
//! `is_dir`/`exists` filesystem probe — those probes are not
//! canonicalize() calls and stay outside the Category A ban (§3.2.2).
//!
//! Worktree auto-registration logic is in the sibling `worktree` module.

use std::path::{Path, PathBuf};

use tonic::Status;
use tracing::{debug, error, info, warn};

use crate::proto::{RegisterProjectRequest, RegisterProjectResponse};

use workspace_qdrant_core::{
    project_disambiguation::ProjectIdCalculator, ItemType, ProjectPayload, QueueManager,
    UnifiedQueueOp,
};
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::paths::CanonicalPath;
use wqm_common::project_id::detect_git_remote;

use super::worktree::WorktreeResult;
use super::ProjectServiceImpl;

/// Result of determining what action to take for a registration request
enum RegistrationAction {
    /// Project exists, session registered (high priority)
    ExistingActivated,
    /// Project exists, no activation change (normal priority)
    ExistingNoop,
    /// Project not found, caller did not request auto-registration
    NotFoundSkipped,
    /// New project enqueued for creation
    NewlyEnqueued { is_high_priority: bool },
    /// Worktree auto-registered (main project exists, worktree path is new)
    WorktreeAutoRegistered { result: WorktreeResult },
}

/// Resolve the git repository root from a given path by walking upward.
/// Returns the original path if no `.git` directory is found.
fn resolve_git_root(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    loop {
        if current.join(".git").exists() {
            return current;
        }
        match current.parent() {
            Some(parent) if parent != current => {
                current = parent.to_path_buf();
            }
            _ => break,
        }
    }
    // No .git found, return original path
    path.to_path_buf()
}

/// Normalize a project path to its syntactic-canonical form and validate
/// it points to an existing directory.
///
/// Replaces the previous `canonicalize_project_path` (which called
/// `std::fs::canonicalize`). Per spec §3.1, normalization is pure
/// syntax: tilde expansion, `.` strip, `..` reject, duplicate-`/`
/// collapse, UTF-8 verification. Symlinks are not followed.
///
/// Existence and `is_dir` are still verified via plain `Path` probes —
/// these are not canonicalize() calls and are necessary to surface the
/// existing `INVALID_ARGUMENT` errors to gRPC clients.
///
/// Returns `INVALID_ARGUMENT` if:
/// - the input path cannot be normalized (non-absolute, contains `..`,
///   non-UTF-8, …),
/// - the path does not exist on the filesystem,
/// - the path is not a directory (e.g. a regular file).
fn normalize_project_path(path: &Path) -> Result<CanonicalPath, Status> {
    let input_str = path.to_str().ok_or_else(|| {
        Status::invalid_argument(format!(
            "project path is not valid UTF-8: {}",
            path.display()
        ))
    })?;
    let canonical = CanonicalPath::from_user_input(input_str).map_err(|e| {
        Status::invalid_argument(format!(
            "project path could not be normalized: {}: {e}",
            path.display()
        ))
    })?;

    let probe = Path::new(canonical.as_str());
    if !probe.exists() {
        return Err(Status::invalid_argument(format!(
            "project path does not exist or is inaccessible: {}",
            canonical.as_str()
        )));
    }
    if !probe.is_dir() {
        return Err(Status::invalid_argument(format!(
            "project path is not a directory: {}",
            canonical.as_str()
        )));
    }
    Ok(canonical)
}

impl ProjectServiceImpl {
    /// Execute the register_project business logic
    pub(crate) async fn handle_register_project(
        &self,
        req: RegisterProjectRequest,
    ) -> Result<RegisterProjectResponse, Status> {
        if req.path.is_empty() && req.project_id.is_empty() {
            return Err(Status::invalid_argument(
                "either path or project_id must be supplied",
            ));
        }

        let (effective_path, effective_path_str) = if req.path.is_empty() {
            (PathBuf::new(), String::new())
        } else {
            let git_root = resolve_git_root(Path::new(&req.path));
            let canonical = normalize_project_path(&git_root)?;
            let s = canonical.as_str().to_string();
            let p = PathBuf::from(&s);
            if s != req.path {
                info!("Normalized project path: {} -> {}", req.path, s);
            }
            (p, s)
        };

        let effective_git_remote = if req.git_remote.as_ref().map_or(true, |r| r.is_empty()) {
            if effective_path.as_os_str().is_empty() {
                None
            } else {
                detect_git_remote(&effective_path)
            }
        } else {
            req.git_remote.clone()
        };

        let effective_priority = req
            .priority
            .as_deref()
            .filter(|p| !p.is_empty())
            .unwrap_or("normal");
        let is_high_priority = effective_priority == "high";

        let project_id =
            self.resolve_project_id(&req, &effective_path, effective_git_remote.as_deref())?;

        info!(
            "Registering project: id={}, path={}, name={:?}, priority={}",
            project_id, effective_path_str, req.name, effective_priority
        );

        let action = self
            .determine_registration_action(&project_id, &req, is_high_priority, &effective_path)
            .await?;

        let watch_meta = self
            .lookup_watch_metadata(&project_id, &effective_path_str)
            .await;

        self.apply_registration_action(
            action,
            project_id,
            &effective_path_str,
            effective_priority,
            watch_meta,
        )
        .await
    }

    async fn apply_registration_action(
        &self,
        action: RegistrationAction,
        project_id: String,
        effective_path_str: &str,
        effective_priority: &str,
        watch_meta: super::worktree::WatchMetadata,
    ) -> Result<RegisterProjectResponse, Status> {
        match action {
            RegistrationAction::NotFoundSkipped => {
                info!(
                    "Project not registered, skipping auto-registration: {}",
                    project_id
                );
                Ok(RegisterProjectResponse {
                    created: false,
                    project_id,
                    priority: "none".to_string(),
                    is_active: false,
                    newly_registered: false,
                    is_worktree: false,
                    watch_path: None,
                })
            }
            RegistrationAction::ExistingActivated => {
                self.activate_project_side_effects(&project_id, effective_path_str)
                    .await;
                self.signal_watch_refresh(&project_id);
                Ok(RegisterProjectResponse {
                    created: false,
                    project_id,
                    priority: effective_priority.to_string(),
                    is_active: true,
                    newly_registered: false,
                    is_worktree: watch_meta.is_worktree,
                    watch_path: watch_meta.watch_path,
                })
            }
            RegistrationAction::ExistingNoop => Ok(RegisterProjectResponse {
                created: false,
                project_id,
                priority: effective_priority.to_string(),
                is_active: false,
                newly_registered: false,
                is_worktree: watch_meta.is_worktree,
                watch_path: watch_meta.watch_path,
            }),
            RegistrationAction::NewlyEnqueued {
                is_high_priority: hp,
            } => {
                if hp {
                    self.activate_new_project(&project_id, effective_path_str)
                        .await;
                }
                self.signal_watch_refresh(&project_id);
                Ok(RegisterProjectResponse {
                    created: true,
                    project_id,
                    priority: effective_priority.to_string(),
                    is_active: hp,
                    newly_registered: true,
                    is_worktree: false,
                    watch_path: None,
                })
            }
            RegistrationAction::WorktreeAutoRegistered { result } => {
                self.handle_worktree_registered(result, project_id, effective_priority)
                    .await
            }
        }
    }

    async fn handle_worktree_registered(
        &self,
        result: WorktreeResult,
        project_id: String,
        effective_priority: &str,
    ) -> Result<RegisterProjectResponse, Status> {
        let WorktreeResult::Registered {
            canonical_path,
            is_high_priority: hp,
        } = result
        else {
            unreachable!("WorktreeAutoRegistered always contains Registered variant")
        };
        if hp {
            self.activate_new_project(&project_id, &canonical_path)
                .await;
        }
        self.signal_watch_refresh(&project_id);
        Ok(RegisterProjectResponse {
            created: true,
            project_id,
            priority: if hp {
                "high".to_string()
            } else {
                effective_priority.to_string()
            },
            is_active: hp,
            newly_registered: true,
            is_worktree: true,
            watch_path: Some(canonical_path),
        })
    }

    /// Signal WatchManager to refresh watch configuration
    fn signal_watch_refresh(&self, project_id: &str) {
        if let Some(ref signal) = self.watch_refresh_signal {
            signal.notify_one();
            debug!(
                project_id = %project_id,
                "Signaled WatchManager refresh for project registration/activation"
            );
        }
    }

    /// Resolve or validate the project_id from the request.
    ///
    /// Uses the resolved git root path and detected remote for ID calculation
    /// so that subfolder registrations produce the same tenant ID as root
    /// registrations.
    fn resolve_project_id(
        &self,
        req: &RegisterProjectRequest,
        effective_path: &Path,
        effective_git_remote: Option<&str>,
    ) -> Result<String, Status> {
        if req.project_id.is_empty() {
            let calculator = ProjectIdCalculator::new();
            let generated = calculator.calculate(effective_path, effective_git_remote, None);
            info!(
                "Generated project_id for {}: {}",
                effective_path.display(),
                generated
            );
            Ok(generated)
        } else {
            let is_local = req.project_id.starts_with("local_");
            let is_hex =
                req.project_id.len() == 12 && req.project_id.chars().all(|c| c.is_ascii_hexdigit());
            if !is_local && !is_hex {
                return Err(Status::invalid_argument(
                    "project_id must be a 12-character hexadecimal string or start with 'local_'",
                ));
            }
            Ok(req.project_id.clone())
        }
    }

    /// Check if a project exists in watch_folders
    async fn project_exists(&self, project_id: &str) -> Result<bool, Status> {
        let existing: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM watch_folders WHERE tenant_id = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(project_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error checking project: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        Ok(existing.is_some())
    }

    /// Check whether a specific path already has a watch_folder row.
    ///
    /// Distinct from [`project_exists`], which keys on `tenant_id`: two
    /// independent working copies of the same git remote share a tenant_id but
    /// occupy different paths. This lets the caller tell "the tenant is known"
    /// from "this exact working copy is registered", which is what enables
    /// multi-clone registration. An empty path is treated as not-registered.
    async fn path_has_watch_folder(&self, path: &str) -> Result<bool, Status> {
        if path.is_empty() {
            return Ok(false);
        }
        let existing: Option<(i32,)> = sqlx::query_as(
            "SELECT 1 FROM watch_folders WHERE path = ?1 AND collection = ?2 LIMIT 1",
        )
        .bind(path)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error checking watch folder path: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        Ok(existing.is_some())
    }

    /// Determine registration action based on project state and request parameters
    async fn determine_registration_action(
        &self,
        project_id: &str,
        req: &RegisterProjectRequest,
        is_high_priority: bool,
        effective_path: &Path,
    ) -> Result<RegistrationAction, Status> {
        let project_already_exists = self.project_exists(project_id).await?;

        // Worktree detection runs first regardless of project existence: a
        // request whose path is a linked worktree must register the worktree
        // path as its own watch_folder sharing the main repo's tenant_id.
        // Skipping this when `project_exists(project_id) == true` would mean
        // the worktree path is never tracked separately, even though its files
        // live at a different location on disk.
        //
        // `try_worktree_auto_register` itself checks whether the main repo is
        // already registered (it requires a parent watch_folder), and whether
        // the worktree path is already in watch_folders — so calling it on the
        // "project exists" branch is safe and idempotent.
        let wt_result = self
            .try_worktree_auto_register(effective_path, is_high_priority)
            .await?;
        if let WorktreeResult::Registered { .. } = &wt_result {
            return Ok(RegistrationAction::WorktreeAutoRegistered { result: wt_result });
        }

        if project_already_exists {
            // Multi-clone: the tenant_id is already registered (e.g. by a
            // sibling working copy of the same git remote), but THIS path has
            // no watch_folder row yet. Register the new clone as an additional
            // watch root under the same tenant_id instead of treating it as a
            // no-op re-registration — otherwise the second working copy never
            // gets scanned/indexed and its branches stay invisible.
            //
            // Routing through `enqueue_new_project` reuses the queue-level
            // (Tenant, Add) handler, which keys watch_folders on PATH (INSERT
            // OR IGNORE), scans `payload.project_root`, and rejects true
            // subdirectories via its own guard — so a sibling clone gets its
            // own row while a re-registration of an already-tracked path (the
            // common session-start case) still short-circuits below.
            let path_str = effective_path.to_str().unwrap_or("");
            if req.register_if_new
                && !path_str.is_empty()
                && !self.path_has_watch_folder(path_str).await?
            {
                info!(
                    "Registering additional clone of existing tenant {} at new path {}",
                    project_id, path_str
                );
                return self
                    .enqueue_new_project(project_id, req, is_high_priority)
                    .await;
            }

            if is_high_priority {
                // Second arg is an ignored legacy `_branch` parameter (see
                // priority_manager::register_session); pass a stable label so
                // it can't be mistaken for a real git ref.
                match self
                    .priority_manager
                    .register_session(project_id, "session")
                    .await
                {
                    Ok(_) => Ok(RegistrationAction::ExistingActivated),
                    Err(e) => {
                        error!("Failed to register session: {e}");
                        Err(Status::internal(format!("Failed to register session: {e}")))
                    }
                }
            } else {
                Ok(RegistrationAction::ExistingNoop)
            }
        } else if req.register_if_new {
            self.enqueue_new_project(project_id, req, is_high_priority)
                .await
        } else {
            Ok(RegistrationAction::NotFoundSkipped)
        }
    }

    /// Enqueue a new project registration via (Tenant, Add)
    async fn enqueue_new_project(
        &self,
        project_id: &str,
        req: &RegisterProjectRequest,
        is_high_priority: bool,
    ) -> Result<RegistrationAction, Status> {
        // Use the resolved and syntactically-normalized git root path for project_root.
        let effective_root_canonical =
            normalize_project_path(&resolve_git_root(Path::new(&req.path))).inspect_err(|_| {
                error!(path = %req.path, "enqueue_new_project: path normalization failed");
            })?;
        let effective_root_str = effective_root_canonical.as_str().to_string();
        let effective_root = PathBuf::from(&effective_root_str);

        let effective_git_remote = if req.git_remote.as_ref().map_or(true, |r| r.is_empty()) {
            detect_git_remote(&effective_root)
        } else {
            req.git_remote.clone()
        };

        let queue_manager = QueueManager::new(self.db_pool.clone());
        let payload = ProjectPayload {
            project_root: effective_root_str.clone(),
            git_remote: effective_git_remote,
            project_type: None,
            old_tenant_id: None,
            is_active: Some(is_high_priority),
        };
        let payload_json = serde_json::to_string(&payload)
            .unwrap_or_else(|_| format!(r#"{{"project_root":"{}"}}"#, effective_root_str));

        match queue_manager
            .enqueue_unified(
                ItemType::Tenant,
                UnifiedQueueOp::Add,
                project_id,
                "projects",
                &payload_json,
                None,
                None,
            )
            .await
        {
            Ok((queue_id, _is_new)) => {
                info!(
                    project_id = %project_id,
                    queue_id = %queue_id,
                    "Enqueued project registration (Tenant, Add)"
                );
                Ok(RegistrationAction::NewlyEnqueued { is_high_priority })
            }
            Err(e) => {
                error!("Failed to enqueue project registration: {e}");
                Err(Status::internal(format!("Failed to enqueue project: {e}")))
            }
        }
    }

    /// Perform activation side effects: cancel deferred shutdown and start LSP.
    ///
    /// Does NOT increment `is_active` — the caller is responsible for ensuring
    /// the session counter was already incremented (via `register_session` for
    /// existing projects, or `activate_project_by_tenant_id` for newly enqueued).
    async fn activate_project_side_effects(&self, project_id: &str, path: &str) {
        if super::lsp_lifecycle::cancel_deferred_shutdown(&self.pending_shutdowns, project_id).await
        {
            debug!(
                project_id = %project_id,
                "Cancelled pending deferred shutdown on project reactivation"
            );
        }

        // Issue #70: `wqm project activate` sends an empty path (the CLI only
        // knows the project_id, not the on-disk root). LSP server startup needs
        // the real project root to detect languages — with an empty root,
        // language detection finds nothing and zero servers spawn, so enrichment
        // never runs. Resolve the canonical root from watch_folders when absent.
        let resolved_path = if path.is_empty() {
            match self.find_watch_folder_by_tenant(project_id).await {
                Ok(Some(record)) => record.path,
                Ok(None) => {
                    warn!(
                        project_id = %project_id,
                        "Cannot start LSP servers: no watch folder found for project"
                    );
                    return;
                }
                Err(e) => {
                    warn!(
                        project_id = %project_id,
                        error = %e,
                        "Cannot start LSP servers: failed to resolve project root"
                    );
                    return;
                }
            }
        } else {
            path.to_string()
        };

        let project_root = PathBuf::from(&resolved_path);
        if let Err(e) = super::lsp_lifecycle::start_project_lsp_servers(
            &self.lsp_manager,
            &self.language_detector,
            project_id,
            &project_root,
        )
        .await
        {
            warn!(
                project_id = %project_id,
                error = %e,
                "Failed to start LSP servers (non-critical)"
            );
        }
    }

    /// Increment `is_active` for a newly registered project and trigger side effects.
    ///
    /// Used only for `NewlyEnqueued` and `WorktreeAutoRegistered` paths where
    /// `register_session` was NOT called (so the counter has not been incremented yet).
    async fn activate_new_project(&self, project_id: &str, path: &str) {
        match self
            .state_manager
            .activate_project_by_tenant_id(project_id)
            .await
        {
            Ok((affected, watch_id)) => {
                if affected > 0 {
                    info!(
                        project_id = %project_id,
                        watch_id = ?watch_id,
                        affected_folders = affected,
                        "Activated new project watch folders"
                    );
                }
            }
            Err(e) => {
                warn!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to activate new project watch folders (non-critical)"
                );
            }
        }

        self.activate_project_side_effects(project_id, path).await;
    }
}
