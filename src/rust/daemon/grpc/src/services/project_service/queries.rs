//! Project query operations
//!
//! Handles read-only project operations: status lookup, listing, and heartbeat.

use chrono::Utc;
use sqlx::SqlitePool;
use tonic::Status;
use tracing::{debug, error, warn};
use workspace_qdrant_core::indexing_progress::{
    estimate_eta_seconds, global_rate_files_per_sec, rate_files_per_sec,
};
use workspace_qdrant_core::queue_operations::QueueManager;

use crate::proto::{
    FailedQueueItem, GetProjectStatusRequest, GetProjectStatusResponse, HeartbeatRequest,
    HeartbeatResponse, ListFailedItemsRequest, ListFailedItemsResponse, ListProjectsRequest,
    ListProjectsResponse, ListWatchesRequest, ListWatchesResponse, ProjectInfo, WatchInfo,
};

use wqm_common::constants::COLLECTION_PROJECTS;

use super::ProjectServiceImpl;

/// Default heartbeat timeout in seconds
const HEARTBEAT_TIMEOUT_SECS: u64 = 60;

/// Per-tenant indexing-progress counts attached to `GetProjectStatusResponse`.
///
/// `pending`, `in_progress`, `failed` come from `unified_queue` (in-flight),
/// while `done` is the durable count from `tracked_files`. The queue rows
/// for completed work are deleted after retention, so we cannot derive
/// `done` from the queue alone.
///
/// `eta_seconds` is `Some` only when we have at least
/// [`MIN_RATE_WINDOW_SECONDS`] of recent indexing activity AND a positive
/// rate — callers must render "warming up" / "unknown" when it's `None`.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct IndexingProgress {
    pub pending: i64,
    pub in_progress: i64,
    pub failed: i64,
    pub done: i64,
    pub eta_seconds: Option<i64>,
}

impl IndexingProgress {
    fn total(&self) -> i64 {
        self.pending + self.in_progress + self.failed + self.done
    }

    fn percent_complete(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            100.0
        } else {
            (self.done as f64) / (total as f64) * 100.0
        }
    }
}

/// Fetch per-tenant indexing progress. Returns zeros on any DB error so the
/// caller can still return the rest of the project status; the gRPC contract
/// has the indexing block as best-effort enrichment.
pub(crate) async fn fetch_indexing_progress(
    pool: &SqlitePool,
    tenant_id: &str,
) -> IndexingProgress {
    let manager = QueueManager::new(pool.clone());
    let (pending, in_progress, failed) =
        match manager.get_in_flight_counts_by_tenant(tenant_id).await {
            Ok(counts) => counts,
            Err(e) => {
                warn!(
                    tenant_id = %tenant_id,
                    error = %e,
                    "Failed to fetch in-flight queue counts for indexing progress"
                );
                (0, 0, 0)
            }
        };

    let done: i64 = match sqlx::query_scalar::<_, i64>(
        r#"
        SELECT COUNT(*)
        FROM tracked_files tf
        JOIN watch_folders wf ON tf.watch_folder_id = wf.watch_id
        WHERE wf.tenant_id = ?1
        "#,
    )
    .bind(tenant_id)
    .fetch_one(pool)
    .await
    {
        Ok(count) => count,
        Err(e) => {
            warn!(
                tenant_id = %tenant_id,
                error = %e,
                "Failed to count tracked_files for indexing progress"
            );
            0
        }
    };

    // Prefer this tenant's own recent rate; fall back to the daemon-wide rate
    // when the tenant is idle (waiting its turn in the unified queue, which
    // processes ~one tenant at a time). Without the fallback an idle tenant's
    // window goes cold and the ETA is a perpetual "warming up".
    let rate = match rate_files_per_sec(pool, tenant_id).await {
        Some(r) => Some(r),
        None => global_rate_files_per_sec(pool).await,
    };
    let eta_seconds = estimate_eta_seconds(pending, in_progress, rate);

    IndexingProgress {
        pending,
        in_progress,
        failed,
        done,
        eta_seconds,
    }
}

impl ProjectServiceImpl {
    /// Execute the get_project_status business logic
    pub(crate) async fn handle_get_project_status(
        &self,
        req: GetProjectStatusRequest,
    ) -> Result<GetProjectStatusResponse, Status> {
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        debug!("Getting project status: {}", req.project_id);

        let query = r#"
            SELECT wf.tenant_id, wf.path, wf.is_active, wf.last_activity_at,
                   wf.created_at, wf.git_remote_url, wf.is_worktree,
                   wf.main_worktree_watch_id,
                   mw.path AS main_worktree_path
            FROM watch_folders wf
            LEFT JOIN watch_folders mw ON wf.main_worktree_watch_id = mw.watch_id
            WHERE wf.tenant_id = ?1 AND wf.collection = ?2
            LIMIT 1
        "#;

        let row = sqlx::query(query)
            .bind(&req.project_id)
            .bind(COLLECTION_PROJECTS)
            .fetch_optional(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error fetching project: {e}");
                Status::internal(format!("Database error: {e}"))
            })?;

        if let Some(row) = row {
            let mut response = Self::build_project_status_response(row)?;
            let progress = fetch_indexing_progress(&self.db_pool, &req.project_id).await;
            response.pending_count = progress.pending;
            response.in_progress_count = progress.in_progress;
            response.failed_count = progress.failed;
            response.done_count = progress.done;
            response.total_count = progress.total();
            response.percent_complete = progress.percent_complete();
            response.eta_seconds = progress.eta_seconds;
            Ok(response)
        } else {
            Ok(GetProjectStatusResponse {
                found: false,
                project_id: req.project_id,
                project_name: String::new(),
                project_root: String::new(),
                priority: String::new(),
                is_active: false,
                last_active: None,
                registered_at: None,
                git_remote: None,
                is_worktree: false,
                main_worktree_path: None,
                pending_count: 0,
                in_progress_count: 0,
                failed_count: 0,
                done_count: 0,
                total_count: 0,
                percent_complete: 0.0,
                eta_seconds: None,
            })
        }
    }

    /// Build a status response from a database row. Indexing-progress fields
    /// are left at zero here; the caller fills them via `fetch_indexing_progress`.
    fn build_project_status_response(
        row: sqlx::sqlite::SqliteRow,
    ) -> Result<GetProjectStatusResponse, Status> {
        use sqlx::Row;

        let last_active_str: Option<String> = row
            .try_get("last_activity_at")
            .map_err(|e| Status::internal(format!("Failed to get last_activity_at: {e}")))?;
        let last_active = last_active_str
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| to_timestamp(dt.with_timezone(&Utc)));

        let registered_at_str: Option<String> = row
            .try_get("created_at")
            .map_err(|e| Status::internal(format!("Failed to get created_at: {e}")))?;
        let registered_at = registered_at_str
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| to_timestamp(dt.with_timezone(&Utc)));

        let is_active_int: i32 = row
            .try_get("is_active")
            .map_err(|e| Status::internal(format!("Failed to get is_active: {e}")))?;

        let priority = if is_active_int == 1 { "high" } else { "normal" };

        let path: String = row
            .try_get("path")
            .map_err(|e| Status::internal(format!("Failed to get path: {e}")))?;
        let project_name = std::path::Path::new(&path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let is_worktree_int: i32 = row.try_get("is_worktree").unwrap_or(0);
        let main_worktree_path: Option<String> = row.try_get("main_worktree_path").unwrap_or(None);

        Ok(GetProjectStatusResponse {
            found: true,
            project_id: row
                .try_get("tenant_id")
                .map_err(|e| Status::internal(format!("Failed to get tenant_id: {e}")))?,
            project_name,
            project_root: path,
            priority: priority.to_string(),
            is_active: is_active_int == 1,
            last_active,
            registered_at,
            git_remote: row
                .try_get("git_remote_url")
                .map_err(|e| Status::internal(format!("Failed to get git_remote_url: {e}")))?,
            is_worktree: is_worktree_int != 0,
            main_worktree_path,
            pending_count: 0,
            in_progress_count: 0,
            failed_count: 0,
            done_count: 0,
            total_count: 0,
            percent_complete: 0.0,
            eta_seconds: None,
        })
    }

    /// Execute the list_failed_items business logic — read-only listing of
    /// `unified_queue` rows in the 'failed' state, optionally scoped to one
    /// tenant. Backs the admin UI's failed-items drill-down; retry is a
    /// separate QueueWriteService mutation.
    pub(crate) async fn handle_list_failed_items(
        &self,
        req: ListFailedItemsRequest,
    ) -> Result<ListFailedItemsResponse, Status> {
        use sqlx::Row;

        let tenant_filter = req
            .tenant_id
            .as_deref()
            .map(str::trim)
            .filter(|t| !t.is_empty());
        let limit: i64 = if req.limit <= 0 {
            100
        } else {
            (req.limit as i64).min(500)
        };

        // Total count for the same filter (so the UI can show "showing N of M").
        let mut count_sql =
            String::from("SELECT COUNT(*) FROM unified_queue WHERE status = 'failed'");
        if tenant_filter.is_some() {
            count_sql.push_str(" AND tenant_id = ?1");
        }
        let mut count_q = sqlx::query_scalar::<_, i64>(&count_sql);
        if let Some(t) = tenant_filter {
            count_q = count_q.bind(t.to_string());
        }
        let total_failed = count_q.fetch_one(&self.db_pool).await.map_err(|e| {
            error!("Database error counting failed items: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        // NULL last_error_at sorts last under DESC in SQLite, so recently-failed
        // rows surface first and rows that never recorded a timestamp trail.
        let mut sql = String::from(
            r#"
            SELECT queue_id, tenant_id, branch, collection, item_type, op,
                   COALESCE(file_path, '')     AS file_path,
                   COALESCE(error_message, '') AS error_message,
                   retry_count,
                   COALESCE(last_error_at, '') AS last_error_at,
                   updated_at
            FROM unified_queue
            WHERE status = 'failed'
            "#,
        );
        if tenant_filter.is_some() {
            sql.push_str(" AND tenant_id = ?1");
        }
        sql.push_str(" ORDER BY last_error_at DESC, updated_at DESC LIMIT ?");
        sql.push_str(if tenant_filter.is_some() { "2" } else { "1" });

        let mut query = sqlx::query(&sql);
        if let Some(t) = tenant_filter {
            query = query.bind(t.to_string());
        }
        query = query.bind(limit);

        let rows = query.fetch_all(&self.db_pool).await.map_err(|e| {
            error!("Database error listing failed items: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        let mut items = Vec::with_capacity(rows.len());
        for row in rows {
            items.push(FailedQueueItem {
                queue_id: row.try_get("queue_id").unwrap_or_default(),
                tenant_id: row.try_get("tenant_id").unwrap_or_default(),
                branch: row.try_get("branch").unwrap_or_default(),
                collection: row.try_get("collection").unwrap_or_default(),
                item_type: row.try_get("item_type").unwrap_or_default(),
                op: row.try_get("op").unwrap_or_default(),
                file_path: row.try_get("file_path").unwrap_or_default(),
                error_message: row.try_get("error_message").unwrap_or_default(),
                retry_count: row.try_get::<i64, _>("retry_count").unwrap_or(0) as i32,
                last_error_at: row.try_get("last_error_at").unwrap_or_default(),
                updated_at: row.try_get("updated_at").unwrap_or_default(),
            });
        }

        Ok(ListFailedItemsResponse {
            items,
            total_failed: total_failed as i32,
        })
    }

    /// Execute the list_projects business logic
    pub(crate) async fn handle_list_projects(
        &self,
        req: ListProjectsRequest,
    ) -> Result<ListProjectsResponse, Status> {
        debug!(
            "Listing projects: priority_filter={:?}, active_only={}",
            req.priority_filter, req.active_only
        );

        let mut query = format!(
            r#"
            SELECT tenant_id, path, is_active, last_activity_at, is_worktree
            FROM watch_folders
            WHERE collection = '{COLLECTION_PROJECTS}'
            "#
        );

        let priority_filter = req.priority_filter.as_deref();
        if let Some(priority) = priority_filter {
            if priority == "high" {
                query.push_str(" AND is_active = 1");
            } else if priority == "normal" || priority == "low" {
                query.push_str(" AND is_active = 0");
            }
        }

        if req.active_only {
            query.push_str(" AND is_active = 1");
        }

        query.push_str(" ORDER BY is_active DESC, last_activity_at DESC");

        let rows = sqlx::query(&query)
            .fetch_all(&self.db_pool)
            .await
            .map_err(|e| {
                error!("Database error listing projects: {e}");
                Status::internal(format!("Database error: {e}"))
            })?;

        let mut projects = Vec::with_capacity(rows.len());
        for row in rows {
            projects.push(Self::build_project_info(row)?);
        }

        let total_count = projects.len() as i32;

        Ok(ListProjectsResponse {
            projects,
            total_count,
        })
    }

    /// Build a ProjectInfo from a database row
    fn build_project_info(row: sqlx::sqlite::SqliteRow) -> Result<ProjectInfo, Status> {
        use sqlx::Row;

        let last_active_str: Option<String> = row
            .try_get("last_activity_at")
            .map_err(|e| Status::internal(format!("Failed to get last_activity_at: {e}")))?;
        let last_active = last_active_str
            .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| to_timestamp(dt.with_timezone(&Utc)));

        let is_active_int: i32 = row
            .try_get("is_active")
            .map_err(|e| Status::internal(format!("Failed to get is_active: {e}")))?;

        let priority = if is_active_int == 1 { "high" } else { "normal" };

        let path: String = row
            .try_get("path")
            .map_err(|e| Status::internal(format!("Failed to get path: {e}")))?;
        let project_name = std::path::Path::new(&path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        let is_worktree_int: i32 = row.try_get("is_worktree").unwrap_or(0);

        Ok(ProjectInfo {
            project_id: row
                .try_get("tenant_id")
                .map_err(|e| Status::internal(format!("Failed to get tenant_id: {e}")))?,
            project_name,
            project_root: path,
            priority: priority.to_string(),
            is_active: is_active_int == 1,
            last_active,
            is_worktree: is_worktree_int != 0,
        })
    }

    /// Execute the list_watches business logic — read-only listing of
    /// `watch_folders`, the gRPC equivalent of `wqm watch list`.
    pub(crate) async fn handle_list_watches(
        &self,
        req: ListWatchesRequest,
    ) -> Result<ListWatchesResponse, Status> {
        debug!(
            "Listing watches: collection={:?}, enabled_only={}",
            req.collection, req.enabled_only
        );

        let mut sql = String::from(
            r#"
            SELECT watch_id, path, collection, tenant_id, enabled, is_active,
                   is_paused, is_archived, last_scan, last_activity_at,
                   git_remote_url, library_mode
            FROM watch_folders
            WHERE 1 = 1
            "#,
        );

        // User-supplied filter — bound as a parameter, never string-interpolated.
        let collection_filter = req
            .collection
            .as_deref()
            .map(str::trim)
            .filter(|c| !c.is_empty());
        if collection_filter.is_some() {
            sql.push_str(" AND collection = ?");
        }
        if req.enabled_only {
            sql.push_str(" AND enabled = 1");
        }
        sql.push_str(" ORDER BY collection, path");

        let mut query = sqlx::query(&sql);
        if let Some(collection) = collection_filter {
            query = query.bind(collection.to_string());
        }

        let rows = query.fetch_all(&self.db_pool).await.map_err(|e| {
            error!("Database error listing watches: {e}");
            Status::internal(format!("Database error: {e}"))
        })?;

        let mut watches = Vec::with_capacity(rows.len());
        for row in rows {
            watches.push(Self::build_watch_info(row)?);
        }

        let total_count = watches.len() as i32;
        Ok(ListWatchesResponse {
            watches,
            total_count,
        })
    }

    /// Build a `WatchInfo` from a `watch_folders` row.
    fn build_watch_info(row: sqlx::sqlite::SqliteRow) -> Result<WatchInfo, Status> {
        use sqlx::Row;

        let watch_id: String = row
            .try_get("watch_id")
            .map_err(|e| Status::internal(format!("Failed to get watch_id: {e}")))?;
        let path: String = row
            .try_get("path")
            .map_err(|e| Status::internal(format!("Failed to get path: {e}")))?;
        let collection: String = row
            .try_get("collection")
            .map_err(|e| Status::internal(format!("Failed to get collection: {e}")))?;

        let tenant_id: Option<String> = row.try_get("tenant_id").unwrap_or(None);
        let enabled: i32 = row.try_get("enabled").unwrap_or(0);
        let is_active: i32 = row.try_get("is_active").unwrap_or(0);
        let is_paused: i32 = row.try_get("is_paused").unwrap_or(0);
        let is_archived: i32 = row.try_get("is_archived").unwrap_or(0);
        let last_scan: Option<String> = row.try_get("last_scan").unwrap_or(None);
        let last_activity_at: Option<String> = row.try_get("last_activity_at").unwrap_or(None);
        let git_remote_url: Option<String> = row.try_get("git_remote_url").unwrap_or(None);
        let library_mode: Option<String> = row.try_get("library_mode").unwrap_or(None);

        Ok(WatchInfo {
            watch_id,
            path,
            collection,
            tenant_id: tenant_id.unwrap_or_default(),
            enabled: enabled != 0,
            is_active: is_active != 0,
            is_paused: is_paused != 0,
            is_archived: is_archived != 0,
            last_scan: last_scan.unwrap_or_default(),
            last_activity_at: last_activity_at.unwrap_or_default(),
            git_remote_url,
            library_mode,
        })
    }

    /// Execute the heartbeat business logic
    pub(crate) async fn handle_heartbeat(
        &self,
        req: HeartbeatRequest,
    ) -> Result<HeartbeatResponse, Status> {
        if req.project_id.is_empty() {
            return Err(Status::invalid_argument("project_id cannot be empty"));
        }

        debug!("Heartbeat received: {}", req.project_id);

        match self.priority_manager.heartbeat(&req.project_id).await {
            Ok(acknowledged) => {
                if acknowledged {
                    self.propagate_heartbeat_activity(&req.project_id).await;
                }

                Ok(HeartbeatResponse {
                    acknowledged,
                    next_heartbeat_by: Some(next_heartbeat_deadline()),
                })
            }
            Err(e) => {
                error!("Heartbeat failed: {e}");
                Err(Status::internal(format!("Heartbeat failed: {e}")))
            }
        }
    }

    /// Propagate heartbeat activity to watch_folders (activity inheritance)
    async fn propagate_heartbeat_activity(&self, project_id: &str) {
        match self
            .state_manager
            .heartbeat_project_by_tenant_id(project_id)
            .await
        {
            Ok((affected, _watch_id)) => {
                if affected > 0 {
                    debug!(
                        project_id = %project_id,
                        affected_folders = affected,
                        "Updated watch folder activity timestamps (activity inheritance)"
                    );
                }
            }
            Err(e) => {
                debug!(
                    project_id = %project_id,
                    error = %e,
                    "Failed to update watch folder activity (non-critical)"
                );
            }
        }
    }
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
    to_timestamp(deadline)
}
