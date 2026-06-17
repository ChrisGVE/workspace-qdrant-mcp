//! Path-based registration reconciliation (issues #138 / #139)
//!
//! Located at `src/rust/daemon/grpc/src/services/project_service/reconcile.rs`.
//! Sibling of `registration.rs` (the RegisterProject flow) and `worktree.rs`
//! (worktree auto-registration). This module is the path-based reconciliation
//! step the registration flow consults *before* deciding a path is brand new.
//!
//! # Why this exists
//!
//! A project's `tenant_id` is derived from its tenancy *type*:
//! `local_<hash(path)>` when there is no git remote, or `<hash(remote)>` when
//! there is one (see `wqm_common::project_id`). The original registration flow
//! matched existing projects **only** by the freshly-recomputed `tenant_id`,
//! so two real-world events were mishandled:
//!
//! - **#138 (path move):** the directory moves but its identity (git remote)
//!   is unchanged, so the recomputed id still matches the stored row. The flow
//!   treated this as an idempotent no-op and never updated the stored `path`,
//!   leaving `wqm project list` / search / grep pointing at the old location.
//! - **#139 (tenancy flip):** the directory stays put but gains or loses a git
//!   remote, so the recomputed id no longer matches the stored row. The flow
//!   treated this as a brand-new project, creating a duplicate `watch_folders`
//!   row and orphaning all data under the old id.
//!
//! Both share one root cause: reconcile-by-id only, never by path. This module
//! adds the missing path-based reconciliation: it looks up the existing
//! registration by path (and, for the moved-path case, by the recomputed id),
//! and either updates the stored path in place or renames the tenant across
//! every tenant-keyed plane.
//!
//! # Cascade coverage boundary (tracked by #140)
//!
//! The tenancy-flip rename reuses the same coverage as the background
//! `remote_monitor` and the admin `rename_tenant` RPC: SQLite rows are renamed
//! in `watch_folders`, `unified_queue`, and `tracked_files`, and a Qdrant
//! cascade-rename is enqueued for the `projects` and `rules` collections.
//! Other tenant-keyed SQLite tables (`symbol_cooccurrence`, the graph store,
//! `keywords`, `tags`, `processing_timings`, priority/affinity tables) and
//! other tenant-keyed Qdrant collections (`scratchpad`, `images`, …) are NOT
//! yet covered. Expanding that coverage is deferred to the follow-up reconcile
//! command in #140; see `enqueue_tenant_cascade` below for the precise list.

use std::path::Path;

use tonic::Status;
use tracing::{error, info, warn};

use workspace_qdrant_core::QueueManager;
use wqm_common::constants::COLLECTION_PROJECTS;

use super::registration::RegistrationAction;
use super::ProjectServiceImpl;

/// An existing project registration as stored in `watch_folders`.
///
/// Only the fields the reconciliation logic needs are selected: the row's
/// primary key, the tenant it is currently keyed under, and the path it was
/// registered at.
pub(super) struct ExistingRegistration {
    pub watch_id: String,
    pub tenant_id: String,
    pub path: String,
}

/// What the registration flow should do for a path that matched an existing
/// registration whose identity no longer lines up with the request.
pub(super) enum ReconcileOutcome {
    /// No existing registration touched this path or recomputed id — the
    /// caller should fall through to its normal new-project handling.
    NotApplicable,
    /// The same path is already registered under the same tenant — a genuine
    /// no-op for reconciliation. The caller proceeds with its existing path.
    AlreadyConsistent,
    /// An existing registration for this identity was found at a *different*
    /// path and the stored path was updated to the new one (#138).
    PathUpdated,
    /// The path's tenancy type flipped (local ↔ remote); the tenant was
    /// renamed across SQLite and a Qdrant cascade was enqueued (#139). The
    /// old → new id pair is logged where the rename happens
    /// (`rename_tenant_for_reconcile`), so the variant itself carries no data.
    TenancyReconciled,
}

impl ProjectServiceImpl {
    /// Reconcile an incoming registration against any existing row for the
    /// same physical project, matching by path first and by recomputed id
    /// second.
    ///
    /// Returns the [`ReconcileOutcome`] describing what was done. The caller
    /// (the registration flow) uses it to decide whether to treat the project
    /// as existing (path update / tenancy flip) or to keep going with
    /// new-project handling ([`ReconcileOutcome::NotApplicable`]).
    ///
    /// `new_tenant_id` is the freshly recomputed id for `effective_path`, and
    /// `effective_git_remote` is the remote that produced it (`None` when the
    /// path is now a local project) — both are reused to refresh the row's git
    /// metadata after a tenancy flip so the background monitor does not
    /// re-detect the same transition.
    pub(super) async fn reconcile_registration(
        &self,
        new_tenant_id: &str,
        effective_git_remote: Option<&str>,
        effective_path: &Path,
        effective_path_str: &str,
    ) -> Result<ReconcileOutcome, Status> {
        // A git worktree shares its main project's tenant_id (worktrees are
        // registered by the worktree auto-registration flow, not as standalone
        // projects). Reconciliation never applies to a worktree path:
        //
        // - by-path: a re-registered worktree recomputes its *own* id, which
        //   differs from the main-based id it was stored under, and would
        //   trigger a spurious tenancy rename.
        // - by-id: a *new* worktree path shares the main project's id and would
        //   look like the main project having moved here, clobbering its path.
        //
        // Skip both and let the worktree flow own the path.
        if self.is_git_worktree(effective_path) {
            return Ok(ReconcileOutcome::NotApplicable);
        }

        // 1. Look up by path: this catches the tenancy-flip case (#139),
        //    where the directory has not moved but its id changed.
        if let Some(existing) = self.find_registration_by_path(effective_path_str).await? {
            if existing.tenant_id == new_tenant_id {
                return Ok(ReconcileOutcome::AlreadyConsistent);
            }
            self.rename_tenant_for_reconcile(
                &existing.tenant_id,
                new_tenant_id,
                effective_git_remote,
                effective_path_str,
            )
            .await?;
            return Ok(ReconcileOutcome::TenancyReconciled);
        }

        // 2. No row at this path. Look up by recomputed id: this catches the
        //    moved-path case (#138), where identity (git remote) is unchanged
        //    so the id still matches, but the stored path is stale.
        if let Some(existing) = self.find_registration_by_tenant(new_tenant_id).await? {
            if existing.path == effective_path_str {
                return Ok(ReconcileOutcome::AlreadyConsistent);
            }
            self.update_registration_path(&existing.watch_id, effective_path_str)
                .await?;
            info!(
                tenant_id = %new_tenant_id,
                old_path = %existing.path,
                new_path = %effective_path_str,
                "Reconciled moved project path"
            );
            return Ok(ReconcileOutcome::PathUpdated);
        }

        Ok(ReconcileOutcome::NotApplicable)
    }

    /// Whether `path` is a git worktree (a linked working tree, not the main
    /// working tree). Worktrees are handled by the worktree auto-registration
    /// flow, not by path-move reconciliation.
    fn is_git_worktree(&self, path: &Path) -> bool {
        workspace_qdrant_core::git::detect_git_status(path).is_worktree
    }

    /// Run path-based reconciliation and translate its outcome into a
    /// [`RegistrationAction`] when the project already existed (#138/#139).
    ///
    /// Returns `Some(action)` when an existing project was reconciled (moved
    /// path or flipped tenancy) — the registration flow short-circuits to that
    /// action. Returns `None` when reconciliation did not apply (no prior row,
    /// or the state was already consistent), letting the flow fall through to
    /// its normal id-based new/existing handling.
    ///
    /// A reconciled project already exists, so it is treated exactly like the
    /// `project_exists` path: a session is registered (activated) for
    /// high-priority requests, otherwise it is a no-op activation-wise.
    pub(super) async fn reconcile_then_classify(
        &self,
        project_id: &str,
        is_high_priority: bool,
        effective_path: &Path,
        effective_git_remote: Option<&str>,
    ) -> Result<Option<RegistrationAction>, Status> {
        // Only paths participate in reconciliation; the activation-only flow
        // (empty path, project_id supplied) has nothing to reconcile.
        let Some(effective_path_str) = effective_path.to_str() else {
            return Ok(None);
        };
        if effective_path_str.is_empty() {
            return Ok(None);
        }

        let outcome = self
            .reconcile_registration(
                project_id,
                effective_git_remote,
                effective_path,
                effective_path_str,
            )
            .await?;

        match outcome {
            ReconcileOutcome::NotApplicable | ReconcileOutcome::AlreadyConsistent => Ok(None),
            ReconcileOutcome::PathUpdated | ReconcileOutcome::TenancyReconciled => {
                if is_high_priority {
                    match self
                        .priority_manager
                        .register_session(project_id, "main")
                        .await
                    {
                        Ok(_) => Ok(Some(RegistrationAction::ExistingActivated)),
                        Err(e) => {
                            error!("Failed to register session after reconcile: {e}");
                            Err(Status::internal(format!("Failed to register session: {e}")))
                        }
                    }
                } else {
                    Ok(Some(RegistrationAction::ExistingNoop))
                }
            }
        }
    }

    /// Find the project registration stored at exactly `path`, if any.
    async fn find_registration_by_path(
        &self,
        path: &str,
    ) -> Result<Option<ExistingRegistration>, Status> {
        let row: Option<(String, String, String)> = sqlx::query_as(
            r#"SELECT watch_id, tenant_id, path FROM watch_folders
               WHERE collection = ?1 AND path = ?2
               LIMIT 1"#,
        )
        .bind(COLLECTION_PROJECTS)
        .bind(path)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error looking up registration by path: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        Ok(row.map(|(watch_id, tenant_id, path)| ExistingRegistration {
            watch_id,
            tenant_id,
            path,
        }))
    }

    /// Find the (non-worktree) project registration keyed under `tenant_id`.
    ///
    /// Excludes worktree rows: a worktree shares its main project's tenant_id,
    /// so the parent row (`main_worktree_watch_id IS NULL`) is the one whose
    /// path represents the project root that may have moved.
    async fn find_registration_by_tenant(
        &self,
        tenant_id: &str,
    ) -> Result<Option<ExistingRegistration>, Status> {
        let row: Option<(String, String, String)> = sqlx::query_as(
            r#"SELECT watch_id, tenant_id, path FROM watch_folders
               WHERE collection = ?1
                 AND tenant_id = ?2
                 AND main_worktree_watch_id IS NULL
               ORDER BY CASE WHEN is_worktree = 0 THEN 0 ELSE 1 END
               LIMIT 1"#,
        )
        .bind(COLLECTION_PROJECTS)
        .bind(tenant_id)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error looking up registration by tenant: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        Ok(row.map(|(watch_id, tenant_id, path)| ExistingRegistration {
            watch_id,
            tenant_id,
            path,
        }))
    }

    /// Update the stored path of an existing watch_folder row (#138).
    async fn update_registration_path(&self, watch_id: &str, new_path: &str) -> Result<(), Status> {
        let now = wqm_common::timestamps::now_utc();
        sqlx::query(
            r#"UPDATE watch_folders
               SET path = ?1, updated_at = ?2
               WHERE watch_id = ?3"#,
        )
        .bind(new_path)
        .bind(&now)
        .bind(watch_id)
        .execute(&self.db_pool)
        .await
        .map_err(|e| {
            error!("Database error updating registration path: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;
        Ok(())
    }

    /// Rename a tenant across SQLite and enqueue the Qdrant cascade (#139).
    ///
    /// Mirrors the background `remote_monitor` reconciliation: SQLite rows are
    /// renamed first (so the registry is immediately consistent), then the
    /// Qdrant cascade is enqueued. `git_remote_url`/`remote_hash`/
    /// `is_git_tracked` on the parent row are refreshed to match the path's
    /// current git state so a subsequent monitor pass does not re-detect a
    /// transition.
    async fn rename_tenant_for_reconcile(
        &self,
        old_tenant_id: &str,
        new_tenant_id: &str,
        effective_git_remote: Option<&str>,
        effective_path_str: &str,
    ) -> Result<(), Status> {
        let remote_hash = effective_git_remote.map(|url| {
            workspace_qdrant_core::project_disambiguation::ProjectIdCalculator::new()
                .calculate_remote_hash(url)
        });
        let is_git_tracked = std::path::Path::new(effective_path_str)
            .join(".git")
            .exists();

        let mut tx = self.db_pool.begin().await.map_err(|e| {
            error!("Failed to begin reconcile transaction: {e}");
            Status::internal(format!("Transaction failed: {e}"))
        })?;

        // Rename tenant_id across every SQLite table the rename covers today.
        Self::rename_table(&mut tx, "watch_folders", old_tenant_id, new_tenant_id).await?;
        Self::rename_table(&mut tx, "unified_queue", old_tenant_id, new_tenant_id).await?;
        match Self::rename_table(&mut tx, "tracked_files", old_tenant_id, new_tenant_id).await {
            Ok(_) => {}
            Err(_) => warn!("Failed to rename tracked_files during reconcile (non-fatal)"),
        }

        // Refresh the parent row's git metadata so the recomputed tenancy type
        // is reflected and the background monitor does not re-fire.
        let now = wqm_common::timestamps::now_utc();
        sqlx::query(
            r#"UPDATE watch_folders
               SET git_remote_url = ?1,
                   remote_hash = ?2,
                   is_git_tracked = ?3,
                   updated_at = ?4
               WHERE tenant_id = ?5
                 AND collection = ?6
                 AND main_worktree_watch_id IS NULL"#,
        )
        .bind(effective_git_remote)
        .bind(remote_hash.as_deref())
        .bind(if is_git_tracked { 1i32 } else { 0i32 })
        .bind(&now)
        .bind(new_tenant_id)
        .bind(COLLECTION_PROJECTS)
        .execute(&mut *tx)
        .await
        .map_err(|e| {
            error!("Failed to refresh git metadata during reconcile: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        tx.commit().await.map_err(|e| {
            error!("Failed to commit reconcile transaction: {e}");
            Status::internal(format!("Failed to commit transaction: {e}"))
        })?;

        info!(
            old_tenant_id = %old_tenant_id,
            new_tenant_id = %new_tenant_id,
            path = %effective_path_str,
            "Reconciled tenancy-type flip on registration (SQLite renamed)"
        );

        self.enqueue_tenant_cascade(old_tenant_id, new_tenant_id)
            .await;
        Ok(())
    }

    /// Enqueue the Qdrant cascade-rename for the tenant-keyed collections.
    ///
    /// Coverage boundary (see module docs / #140): only `projects` and `rules`
    /// are cascaded, matching the background monitor. `scratchpad`, `images`,
    /// and any other tenant-keyed collection are left for #140. The cascade is
    /// fire-and-forget with `warn!` on failure: SQLite is already consistent,
    /// so a transient enqueue failure must not fail the registration ack.
    async fn enqueue_tenant_cascade(&self, old_tenant_id: &str, new_tenant_id: &str) {
        let queue_manager = QueueManager::new(self.db_pool.clone());
        match queue_manager
            .enqueue_cascade_rename(
                old_tenant_id,
                new_tenant_id,
                &["projects", "rules"],
                "Registration-triggered tenancy-type flip",
            )
            .await
        {
            Ok(queue_ids) => info!(
                "Enqueued {} cascade rename(s) on registration: {} -> {}",
                queue_ids.len(),
                old_tenant_id,
                new_tenant_id
            ),
            Err(e) => warn!(
                "Failed to enqueue cascade rename on registration {} -> {}: {} \
                 (SQLite already consistent; Qdrant payloads may drift)",
                old_tenant_id, new_tenant_id, e
            ),
        }
    }
}
