//! Project query operations
//!
//! Handles read-only project operations: status lookup, listing, and heartbeat.

use chrono::Utc;
use tonic::Status;
use tracing::{debug, error};

use crate::proto::{
    GetProjectStatusRequest, GetProjectStatusResponse, HeartbeatRequest, HeartbeatResponse,
    ListProjectsRequest, ListProjectsResponse, ProjectInfo,
};

use wqm_common::constants::COLLECTION_PROJECTS;

use super::ProjectServiceImpl;

/// Default heartbeat timeout in seconds
const HEARTBEAT_TIMEOUT_SECS: u64 = 60;

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
                error!("Database error fetching project: {e}");
                Status::internal(format!("Database error: {e}"))
            })?;

        if let Some(row) = row {
            Self::build_project_status_response(row)
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
            })
        }
    }

    /// Build a status response from a database row
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

        let priority = if is_active_int == 1 {
            "high"
        } else {
            "normal"
        };

        let path: String = row
            .try_get("path")
            .map_err(|e| Status::internal(format!("Failed to get path: {e}")))?;
        let project_name = std::path::Path::new(&path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

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
            SELECT tenant_id, path, is_active, last_activity_at
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

        let priority = if is_active_int == 1 {
            "high"
        } else {
            "normal"
        };

        let path: String = row
            .try_get("path")
            .map_err(|e| Status::internal(format!("Failed to get path: {e}")))?;
        let project_name = std::path::Path::new(&path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();

        Ok(ProjectInfo {
            project_id: row
                .try_get("tenant_id")
                .map_err(|e| Status::internal(format!("Failed to get tenant_id: {e}")))?,
            project_name,
            project_root: path,
            priority: priority.to_string(),
            is_active: is_active_int == 1,
            last_active,
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
