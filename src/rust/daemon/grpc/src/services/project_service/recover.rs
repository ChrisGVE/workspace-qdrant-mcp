//! Manual project recover / re-point handler (issue #140).
//!
//! Located at `src/rust/daemon/grpc/src/services/project_service/recover.rs`,
//! sibling of `reconcile.rs` (the *automatic* registration-time reconciliation,
//! #138/#139). Where `reconcile.rs` fires implicitly during RegisterProject and
//! covers only a narrow table subset, this module is the *explicit, manual*
//! recover RPC (`wqm project recover`, MCP admin) and drives the **complete**
//! tenant-keyed cascade defined in `workspace_qdrant_core::recover`.
//!
//! # What a recover does
//!
//! Given a registered project it reconciles two independent kinds of drift in
//! one operation (either, both, or neither may apply):
//!
//! - **Path move** (`--new-path`): the stored watch root is swapped old->new,
//!   and every stored *absolute* file path is prefix-rewritten old->new across
//!   both planes — state.db (`unified_queue.file_path`), search.db
//!   (`file_metadata.file_path`), and Qdrant (`file_path` / `absolute_path`
//!   payload). Relative paths (`tracked_files.relative_path`, the graph store,
//!   the Qdrant `relative_path` payload) are watch-root-relative and untouched.
//! - **Tenancy flip** (`--rescan-remote`): the project's `tenant_id` is
//!   recomputed from its current git remote and every tenant_id-keyed row /
//!   point is re-keyed old->new across state.db, search.db, graph.db, and all
//!   tenant-keyed Qdrant collections.
//!
//! `--dry-run` reports the old->new id/path and the row/point counts that would
//! change, writing nothing. With no flags, recover auto-detects the project's
//! current path + remote and reconciles any drift. Re-running on an already
//! consistent registration is a no-op (idempotent).
//!
//! # Two-plane execution and atomicity
//!
//! The state.db work — path repoint and tenancy rename — runs under **one**
//! transaction (`recover_state_db`), so SQLite is never left half-migrated: a
//! failure rolls both back together. Order inside the transaction matters: the
//! repoint is keyed by the OLD tenant_id, so it must run before the rename
//! changes that key. The satellite search.db / graph.db live in their own
//! connection pools and are reconciled after the state.db commit.
//!
//! The SQLite↔Qdrant boundary is inherently **non-atomic**: there is no
//! cross-store transaction. After the state.db transaction commits, the Qdrant
//! path rewrite and the enqueued tenant cascade run separately. If the Qdrant
//! step fails, SQLite is already consistent and Qdrant payloads may drift; this
//! is logged (`warn!`) rather than rolled back. Re-running recover on an
//! already-repointed path is a safe no-op (the boundary-anchored prefix match
//! finds nothing to change), so the operator's remedy for a partial failure is
//! simply to re-run — a known limitation flagged for a follow-up resume marker.

use std::path::{Path, PathBuf};
use std::time::Duration;

use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePoolOptions};
use sqlx::SqlitePool;
use tonic::Status;
use tracing::{info, warn};

use workspace_qdrant_core::recover::{self, QDRANT_TENANT_COLLECTIONS};
use workspace_qdrant_core::search_db::search_db_path_from_state;
use workspace_qdrant_core::QueueManager;
use wqm_common::constants::COLLECTION_PROJECTS;
use wqm_common::paths::CanonicalPath;
use wqm_common::project_id::{detect_git_remote, ProjectIdCalculator};

use crate::proto::{RecoverProjectRequest, RecoverProjectResponse};

use super::ProjectServiceImpl;

/// The current registration of the project being recovered.
struct CurrentRegistration {
    tenant_id: String,
    path: String,
}

/// The resolved plan: what the new identity and location should be.
struct RecoverPlan {
    old_tenant_id: String,
    new_tenant_id: String,
    old_path: String,
    new_path: String,
}

impl RecoverPlan {
    fn tenancy_flips(&self) -> bool {
        self.old_tenant_id != self.new_tenant_id
    }

    fn path_moves(&self) -> bool {
        self.old_path != self.new_path
    }

    fn is_noop(&self) -> bool {
        !self.tenancy_flips() && !self.path_moves()
    }
}

impl ProjectServiceImpl {
    /// Handle the RecoverProject RPC (#140).
    pub(crate) async fn handle_recover_project(
        &self,
        req: RecoverProjectRequest,
    ) -> Result<RecoverProjectResponse, Status> {
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        let current = self.load_current_registration(&req.project_id).await?;
        let plan = self.build_plan(&current, &req)?;

        info!(
            old_tenant = %plan.old_tenant_id,
            new_tenant = %plan.new_tenant_id,
            old_path = %plan.old_path,
            new_path = %plan.new_path,
            dry_run = req.dry_run,
            "RecoverProject"
        );

        if plan.is_noop() {
            return Ok(RecoverProjectResponse {
                success: true,
                dry_run: req.dry_run,
                changed: false,
                old_tenant_id: plan.old_tenant_id.clone(),
                new_tenant_id: plan.new_tenant_id,
                old_path: plan.old_path.clone(),
                new_path: plan.new_path,
                sqlite_rows_updated: 0,
                qdrant_points_updated: 0,
                message: format!(
                    "Project '{}' already consistent — nothing to recover",
                    plan.old_tenant_id
                ),
            });
        }

        if req.dry_run {
            self.recover_dry_run(plan).await
        } else {
            self.recover_apply(plan).await
        }
    }

    /// Look up the project's stored tenant_id + path.
    ///
    /// Primary lookup is by tenant_id (the request's `project_id`). After a
    /// tenancy flip the caller may only know the path under which it was *last*
    /// registered, so a secondary by-path lookup (mirroring
    /// `reconcile::find_registration_by_path`) catches a registration whose id
    /// no longer matches the request.
    async fn load_current_registration(
        &self,
        project_id: &str,
    ) -> Result<CurrentRegistration, Status> {
        let by_id: Option<(String, String)> = sqlx::query_as(
            "SELECT tenant_id, path FROM watch_folders \
             WHERE tenant_id = ?1 AND collection = ?2 AND main_worktree_watch_id IS NULL \
             LIMIT 1",
        )
        .bind(project_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| Status::internal(format!("Database error: {e}")))?;

        if let Some((tenant_id, path)) = by_id {
            return Ok(CurrentRegistration { tenant_id, path });
        }

        // Secondary: the caller passed a path that was a previous registration
        // location, but the stored id has since flipped. Match by path.
        let by_path: Option<(String, String)> = sqlx::query_as(
            "SELECT tenant_id, path FROM watch_folders \
             WHERE path = ?1 AND collection = ?2 AND main_worktree_watch_id IS NULL \
             LIMIT 1",
        )
        .bind(project_id)
        .bind(COLLECTION_PROJECTS)
        .fetch_optional(&self.db_pool)
        .await
        .map_err(|e| Status::internal(format!("Database error: {e}")))?;

        match by_path {
            Some((tenant_id, path)) => Ok(CurrentRegistration { tenant_id, path }),
            None => Err(Status::not_found(format!(
                "Project not found: {project_id}"
            ))),
        }
    }

    /// Resolve the new path and new tenant_id from the request.
    ///
    /// `new_path` (when set) is validated as a [`CanonicalPath`] before it
    /// becomes the new watch root — rejecting relative or `..`-bearing input so
    /// it can never traverse outside an absolute root or produce a spurious
    /// flip. `rescan_remote` recomputes the tenant_id from the (new or current)
    /// path's git remote. With neither flag set, both default to the stored
    /// values and the plan is a no-op unless the on-disk remote already drifted.
    fn build_plan(
        &self,
        current: &CurrentRegistration,
        req: &RecoverProjectRequest,
    ) -> Result<RecoverPlan, Status> {
        let new_path = match req.new_path.as_deref().filter(|p| !p.is_empty()) {
            Some(p) => CanonicalPath::from_user_input(p)
                .map_err(|e| Status::invalid_argument(format!("invalid new_path: {e}")))?
                .into_string(),
            None => current.path.clone(),
        };

        // Recompute tenant_id only when asked (--rescan-remote) or when the path
        // moved (a moved local project's id is derived from its path, so it
        // changes too). When neither applies, keep the stored id.
        let new_tenant_id = if req.rescan_remote || new_path != current.path {
            let path = PathBuf::from(&new_path);
            let git_remote = detect_git_remote(&path);
            ProjectIdCalculator::new().calculate(&path, git_remote.as_deref(), None)
        } else {
            current.tenant_id.clone()
        };

        Ok(RecoverPlan {
            old_tenant_id: current.tenant_id.clone(),
            new_tenant_id,
            old_path: current.path.clone(),
            new_path,
        })
    }

    /// Report planned changes without writing.
    async fn recover_dry_run(&self, plan: RecoverPlan) -> Result<RecoverProjectResponse, Status> {
        let mut sqlite_rows = 0i64;
        let mut qdrant_points = 0i64;

        if plan.tenancy_flips() {
            let counts = recover::count_tenant_rows(&self.db_pool, &plan.old_tenant_id)
                .await
                .map_err(map_schema_err)?;
            sqlite_rows += counts.total_rows;
        }
        if plan.path_moves() {
            sqlite_rows +=
                recover::count_repoint_rows(&self.db_pool, &plan.old_tenant_id, &plan.old_path)
                    .await
                    .map_err(map_schema_err)?;

            qdrant_points += self.qdrant_repath_all(&plan, true).await?;
        }

        Ok(RecoverProjectResponse {
            success: true,
            dry_run: true,
            changed: sqlite_rows > 0 || qdrant_points > 0,
            message: format!(
                "DRY RUN: would update {sqlite_rows} SQLite row(s) and {qdrant_points} Qdrant \
                 point(s); tenant {} -> {}, path {} -> {}",
                plan.old_tenant_id, plan.new_tenant_id, plan.old_path, plan.new_path
            ),
            old_tenant_id: plan.old_tenant_id,
            new_tenant_id: plan.new_tenant_id,
            old_path: plan.old_path,
            new_path: plan.new_path,
            sqlite_rows_updated: clamp_i32(sqlite_rows),
            qdrant_points_updated: clamp_i32(qdrant_points),
        })
    }

    /// Apply the recover across both planes.
    async fn recover_apply(&self, plan: RecoverPlan) -> Result<RecoverProjectResponse, Status> {
        // --- state.db: path repoint + tenancy rename under ONE transaction. ---
        let mut sqlite_rows = self.recover_state_db(&plan).await?;

        // --- satellite DBs (search.db, graph.db), opened on demand. ---
        sqlite_rows += self.recover_satellite_dbs(&plan).await?;

        // --- Qdrant: synchronous path rewrite + enqueued tenant cascade. ---
        let mut qdrant_points = 0i64;
        if plan.path_moves() {
            qdrant_points += self.qdrant_repath_all(&plan, false).await?;
        }
        if plan.tenancy_flips() {
            self.enqueue_recover_cascade(&plan.old_tenant_id, &plan.new_tenant_id)
                .await;
        }

        Ok(RecoverProjectResponse {
            success: true,
            dry_run: false,
            changed: true,
            message: format!(
                "Recovered project: tenant {} -> {}, path {} -> {}; {sqlite_rows} SQLite row(s) \
                 and {qdrant_points} Qdrant point(s) updated{}",
                plan.old_tenant_id,
                plan.new_tenant_id,
                plan.old_path,
                plan.new_path,
                if plan.tenancy_flips() {
                    " (Qdrant tenant cascade enqueued)"
                } else {
                    ""
                }
            ),
            old_tenant_id: plan.old_tenant_id,
            new_tenant_id: plan.new_tenant_id,
            old_path: plan.old_path,
            new_path: plan.new_path,
            sqlite_rows_updated: clamp_i32(sqlite_rows),
            qdrant_points_updated: clamp_i32(qdrant_points),
        })
    }

    /// Reconcile the main state.db under one transaction: path repoint first
    /// (keyed by the OLD tenant), then tenancy rename. Either failing rolls both
    /// back, so state.db is never left half-migrated. Returns total rows.
    async fn recover_state_db(&self, plan: &RecoverPlan) -> Result<i64, Status> {
        let mut tx =
            self.db_pool.begin().await.map_err(|e| {
                Status::internal(format!("Failed to begin recover transaction: {e}"))
            })?;

        let mut rows = 0i64;
        if plan.path_moves() {
            rows += recover::repoint_path_in_tx(
                &mut tx,
                &plan.old_tenant_id,
                &plan.old_path,
                &plan.new_path,
            )
            .await
            .map_err(map_schema_err)?;
        }
        if plan.tenancy_flips() {
            let counts =
                recover::rename_tenant_in_tx(&mut tx, &plan.old_tenant_id, &plan.new_tenant_id)
                    .await
                    .map_err(map_schema_err)?;
            rows += counts.total_rows;
        }

        tx.commit()
            .await
            .map_err(|e| Status::internal(format!("Failed to commit recover transaction: {e}")))?;
        Ok(rows)
    }

    /// Rewrite the absolute path payload across every path-keyed Qdrant
    /// collection, summing the points changed. Points are still keyed under the
    /// OLD tenant until the enqueued cascade runs, so the repath uses the old id.
    async fn qdrant_repath_all(&self, plan: &RecoverPlan, dry_run: bool) -> Result<i64, Status> {
        let Some(storage) = self.storage.as_ref() else {
            return Ok(0);
        };
        let mut total = 0i64;
        for &collection in QDRANT_TENANT_COLLECTIONS {
            let n = storage
                .rewrite_path_payload_prefix_by_tenant(
                    collection,
                    &plan.old_tenant_id,
                    &plan.old_path,
                    &plan.new_path,
                    dry_run,
                )
                .await
                .map_err(|e| Status::internal(format!("Qdrant repath failed: {e}")))?;
            total += n as i64;
        }
        Ok(total)
    }

    /// Reconcile search.db + graph.db, opened read-write from the state.db path.
    ///
    /// Returns the satellite rows updated. A satellite DB that cannot be opened
    /// (not yet provisioned) contributes zero and is logged, never failing the
    /// recover — the unified processor recreates it on next ingest.
    async fn recover_satellite_dbs(&self, plan: &RecoverPlan) -> Result<i64, Status> {
        let Some(state_path) = self.state_db_path().await else {
            warn!("RecoverProject: could not resolve state.db path; skipping satellite DBs");
            return Ok(0);
        };

        let mut rows = 0i64;

        // search.db
        let search_path = search_db_path_from_state(Path::new(&state_path));
        if search_path.exists() {
            match open_rw_pool(&search_path).await {
                Ok(pool) => {
                    if plan.path_moves() {
                        rows += recover::rewrite_paths_search_db(
                            &pool,
                            &plan.old_tenant_id,
                            &plan.old_path,
                            &plan.new_path,
                        )
                        .await
                        .map_err(map_schema_err)?;
                    }
                    if plan.tenancy_flips() {
                        rows += recover::rename_tenant_search_db(
                            &pool,
                            &plan.old_tenant_id,
                            &plan.new_tenant_id,
                        )
                        .await
                        .map_err(map_schema_err)?;
                    }
                    pool.close().await;
                }
                Err(e) => warn!("RecoverProject: search.db open failed (skipped): {e}"),
            }
        }

        // graph.db — only the tenancy rename matters (paths are relative).
        if plan.tenancy_flips() {
            let graph_path = Path::new(&state_path)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join(workspace_qdrant_core::graph::GRAPH_DB_FILENAME);
            if graph_path.exists() {
                match open_rw_pool(&graph_path).await {
                    Ok(pool) => {
                        rows += recover::rename_tenant_graph_db(
                            &pool,
                            &plan.old_tenant_id,
                            &plan.new_tenant_id,
                        )
                        .await
                        .map_err(map_schema_err)?;
                        pool.close().await;
                    }
                    Err(e) => warn!("RecoverProject: graph.db open failed (skipped): {e}"),
                }
            }
        }

        Ok(rows)
    }

    /// Enqueue the Qdrant tenant_id cascade for every tenant-keyed collection.
    async fn enqueue_recover_cascade(&self, old_tenant_id: &str, new_tenant_id: &str) {
        let queue_manager = QueueManager::new(self.db_pool.clone());
        match queue_manager
            .enqueue_cascade_rename(
                old_tenant_id,
                new_tenant_id,
                QDRANT_TENANT_COLLECTIONS,
                "Manual recover (#140) tenancy flip",
            )
            .await
        {
            Ok(ids) => info!(
                "Recover enqueued {} Qdrant cascade rename(s): {} -> {}",
                ids.len(),
                old_tenant_id,
                new_tenant_id
            ),
            Err(e) => warn!(
                "Recover failed to enqueue Qdrant cascade {} -> {}: {} \
                 (SQLite already consistent; Qdrant payloads may drift)",
                old_tenant_id, new_tenant_id, e
            ),
        }
    }

    /// The state.db file path, via `pragma_database_list`.
    async fn state_db_path(&self) -> Option<String> {
        sqlx::query_scalar("SELECT file FROM pragma_database_list WHERE name = 'main'")
            .fetch_optional(&self.db_pool)
            .await
            .ok()
            .flatten()
            .filter(|p: &String| !p.is_empty())
    }
}

/// Open a read-write SQLite pool on an existing satellite database file.
///
/// WAL + a 5s `busy_timeout` so a concurrent writer (the unified processor)
/// yields a retried `SQLITE_BUSY` instead of an immediate failure. Built from
/// [`SqliteConnectOptions`] rather than a `sqlite://` URL string so a path with
/// special characters never needs URL-escaping.
async fn open_rw_pool(path: &Path) -> Result<SqlitePool, sqlx::Error> {
    let options = SqliteConnectOptions::new()
        .filename(path)
        .journal_mode(SqliteJournalMode::Wal)
        .busy_timeout(Duration::from_secs(5));
    SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
}

/// Clamp an `i64` row/point count into the proto `int32` field, saturating
/// rather than silently wrapping on the (practically unreachable) overflow.
fn clamp_i32(n: i64) -> i32 {
    i32::try_from(n).unwrap_or(i32::MAX)
}

/// Map a core `SchemaError` to a gRPC `Status`.
fn map_schema_err(e: workspace_qdrant_core::SchemaError) -> Status {
    Status::internal(format!("Recover cascade failed: {e}"))
}
