//! ProjectService RPC wrappers for [`DaemonClient`].
//!
//! Mirrors the project-related methods in TS `DaemonClientSystem`
//! (`system-methods.ts` lines 95-141):
//!
//! | Rust method              | Proto RPC                           | TS equivalent           |
//! |--------------------------|-------------------------------------|-------------------------|
//! | `register_project`       | `ProjectService::RegisterProject`   | `registerProject()`     |
//! | `deprioritize_project`   | `ProjectService::DeprioritizeProject` | `deprioritizeProject()` |
//! | `heartbeat`              | `ProjectService::Heartbeat`         | `heartbeat()`           |
//! | `resolve_search_scope`   | `ProjectService::ResolveSearchScope`| `resolveSearchScope()`  |
//!
//! All four use [`DaemonClient::call`] with default (5 s) timeout.
//! Field names are kept as snake_case matching proto definitions.

use tonic::Status;

use crate::proto::{
    DeprioritizeProjectRequest, DeprioritizeProjectResponse, HeartbeatRequest, HeartbeatResponse,
    RegisterProjectRequest, RegisterProjectResponse, ResolveSearchScopeRequest,
    ResolveSearchScopeResponse,
};

use super::client::DaemonClient;

impl DaemonClient {
    /// Register a project for tracking — mirrors TS `registerProject()`.
    ///
    /// The request fields mirror [`RegisterProjectRequest`] exactly:
    /// - `path`: canonical absolute path to project root
    /// - `project_id`: 12-char hex identifier
    /// - `name`: optional human-readable name
    /// - `git_remote`: optional git remote URL
    /// - `register_if_new`: if false, only activates existing projects
    /// - `priority`: `"high"` or `"normal"`
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn register_project(
        &mut self,
        request: RegisterProjectRequest,
    ) -> Result<RegisterProjectResponse, Status> {
        let client = self.project.clone();
        self.call("registerProject", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.register_project(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Decrement a project's session count — mirrors TS `deprioritizeProject()`.
    ///
    /// Called when the MCP server shuts down to signal that the project is
    /// no longer actively in use.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn deprioritize_project(
        &mut self,
        request: DeprioritizeProjectRequest,
    ) -> Result<DeprioritizeProjectResponse, Status> {
        let client = self.project.clone();
        self.call("deprioritizeProject", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.deprioritize_project(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Send a session heartbeat — mirrors TS `heartbeat()`.
    ///
    /// Called periodically (every ~30 s) by active MCP sessions.
    /// The daemon uses a 60 s timeout to detect orphaned sessions.
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn heartbeat(
        &mut self,
        request: HeartbeatRequest,
    ) -> Result<HeartbeatResponse, Status> {
        let client = self.project.clone();
        self.call("heartbeat", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.heartbeat(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }

    /// Resolve a search scope to concrete tenant IDs — mirrors TS `resolveSearchScope()`.
    ///
    /// The request `scope` field is `"project"`, `"group"`, or `"all"`.
    /// The response includes:
    /// - `tenant_ids`: list of tenant IDs to include in search filters
    /// - `filter_by_tenant`: `false` when scope is `"all"` (no tenant filter applied)
    /// - `decay_map`: per-tenant relevance decay multipliers
    ///
    /// # Errors
    /// Propagates any [`Status`] error from the daemon (or timeout).
    pub async fn resolve_search_scope(
        &mut self,
        request: ResolveSearchScopeRequest,
    ) -> Result<ResolveSearchScopeResponse, Status> {
        let client = self.project.clone();
        self.call("resolveSearchScope", None, || {
            let mut c = client.clone();
            let req = request.clone();
            async move {
                c.resolve_search_scope(tonic::Request::new(req))
                    .await
                    .map(|r| r.into_inner())
            }
        })
        .await
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::TenantDecay;

    // Helpers: construct proto request/response types directly.

    fn make_register_request(path: &str, project_id: &str) -> RegisterProjectRequest {
        RegisterProjectRequest {
            path: path.to_string(),
            project_id: project_id.to_string(),
            name: None,
            git_remote: None,
            register_if_new: true,
            priority: None,
        }
    }

    fn make_register_response(created: bool, project_id: &str) -> RegisterProjectResponse {
        RegisterProjectResponse {
            created,
            project_id: project_id.to_string(),
            priority: "high".to_string(),
            is_active: true,
            newly_registered: created,
            is_worktree: false,
            watch_path: None,
        }
    }

    fn make_deprioritize_request(project_id: &str) -> DeprioritizeProjectRequest {
        DeprioritizeProjectRequest {
            project_id: project_id.to_string(),
            watch_path: None,
        }
    }

    fn make_deprioritize_response() -> DeprioritizeProjectResponse {
        DeprioritizeProjectResponse {
            success: true,
            is_active: false,
            new_priority: "normal".to_string(),
        }
    }

    fn make_heartbeat_request(project_id: &str) -> HeartbeatRequest {
        HeartbeatRequest {
            project_id: project_id.to_string(),
        }
    }

    fn make_heartbeat_response(acknowledged: bool) -> HeartbeatResponse {
        HeartbeatResponse {
            acknowledged,
            next_heartbeat_by: None,
        }
    }

    fn make_scope_request(tenant_id: &str, scope: &str) -> ResolveSearchScopeRequest {
        ResolveSearchScopeRequest {
            tenant_id: tenant_id.to_string(),
            scope: scope.to_string(),
        }
    }

    fn make_scope_response_project(tenant_id: &str) -> ResolveSearchScopeResponse {
        ResolveSearchScopeResponse {
            tenant_ids: vec![tenant_id.to_string()],
            filter_by_tenant: true,
            decay_map: vec![TenantDecay {
                tenant_id: tenant_id.to_string(),
                multiplier: 1.0,
            }],
        }
    }

    fn make_scope_response_all() -> ResolveSearchScopeResponse {
        ResolveSearchScopeResponse {
            tenant_ids: vec![],
            filter_by_tenant: false,
            decay_map: vec![],
        }
    }

    // ── RegisterProjectRequest field mapping ──────────────────────────────────

    #[test]
    fn register_request_path_field() {
        let req = make_register_request("/home/user/myproject", "abc123def456");
        assert_eq!(req.path, "/home/user/myproject");
    }

    #[test]
    fn register_request_project_id_field() {
        let req = make_register_request("/home/user/myproject", "abc123def456");
        assert_eq!(req.project_id, "abc123def456");
    }

    #[test]
    fn register_request_register_if_new_true() {
        let req = make_register_request("/home/user/myproject", "abc123def456");
        assert!(req.register_if_new);
    }

    #[test]
    fn register_request_optional_fields_none() {
        let req = make_register_request("/home/user/myproject", "abc123def456");
        assert!(req.name.is_none());
        assert!(req.git_remote.is_none());
        assert!(req.priority.is_none());
    }

    #[test]
    fn register_request_with_priority() {
        let mut req = make_register_request("/home/user/myproject", "abc123def456");
        req.priority = Some("high".to_string());
        assert_eq!(req.priority.as_deref(), Some("high"));
    }

    // ── RegisterProjectResponse field mapping ─────────────────────────────────

    #[test]
    fn register_response_created_true() {
        let resp = make_register_response(true, "abc123def456");
        assert!(resp.created);
        assert!(resp.newly_registered);
    }

    #[test]
    fn register_response_created_false_existing() {
        let resp = make_register_response(false, "abc123def456");
        assert!(!resp.created);
        assert!(!resp.newly_registered);
    }

    #[test]
    fn register_response_project_id_echoed() {
        let resp = make_register_response(true, "abc123def456");
        assert_eq!(resp.project_id, "abc123def456");
    }

    #[test]
    fn register_response_is_active_flag() {
        let resp = make_register_response(true, "abc123def456");
        assert!(resp.is_active);
    }

    #[test]
    fn register_response_is_worktree_false() {
        let resp = make_register_response(true, "abc123def456");
        assert!(!resp.is_worktree);
    }

    #[test]
    fn register_response_watch_path_none() {
        let resp = make_register_response(true, "abc123def456");
        assert!(resp.watch_path.is_none());
    }

    // ── DeprioritizeProject request/response ──────────────────────────────────

    #[test]
    fn deprioritize_request_project_id() {
        let req = make_deprioritize_request("abc123def456");
        assert_eq!(req.project_id, "abc123def456");
        assert!(req.watch_path.is_none());
    }

    #[test]
    fn deprioritize_request_with_watch_path() {
        let mut req = make_deprioritize_request("abc123def456");
        req.watch_path = Some("/home/user/myproject".to_string());
        assert_eq!(req.watch_path.as_deref(), Some("/home/user/myproject"));
    }

    #[test]
    fn deprioritize_response_success() {
        let resp = make_deprioritize_response();
        assert!(resp.success);
        assert!(!resp.is_active);
        assert_eq!(resp.new_priority, "normal");
    }

    // ── Heartbeat request/response ────────────────────────────────────────────

    #[test]
    fn heartbeat_request_project_id() {
        let req = make_heartbeat_request("abc123def456");
        assert_eq!(req.project_id, "abc123def456");
    }

    #[test]
    fn heartbeat_response_acknowledged() {
        let resp = make_heartbeat_response(true);
        assert!(resp.acknowledged);
        assert!(resp.next_heartbeat_by.is_none());
    }

    #[test]
    fn heartbeat_response_not_acknowledged() {
        let resp = make_heartbeat_response(false);
        assert!(!resp.acknowledged);
    }

    // ── ResolveSearchScope request/response ───────────────────────────────────

    #[test]
    fn scope_request_project_scope() {
        let req = make_scope_request("abc123def456", "project");
        assert_eq!(req.tenant_id, "abc123def456");
        assert_eq!(req.scope, "project");
    }

    #[test]
    fn scope_request_group_scope() {
        let req = make_scope_request("abc123def456", "group");
        assert_eq!(req.scope, "group");
    }

    #[test]
    fn scope_request_all_scope() {
        let req = make_scope_request("abc123def456", "all");
        assert_eq!(req.scope, "all");
    }

    #[test]
    fn scope_response_project_has_single_tenant() {
        let resp = make_scope_response_project("abc123def456");
        assert_eq!(resp.tenant_ids, vec!["abc123def456"]);
        assert!(resp.filter_by_tenant);
    }

    #[test]
    fn scope_response_all_has_no_filter() {
        let resp = make_scope_response_all();
        assert!(resp.tenant_ids.is_empty());
        assert!(!resp.filter_by_tenant);
        assert!(resp.decay_map.is_empty());
    }

    #[test]
    fn scope_response_decay_map_multiplier() {
        let resp = make_scope_response_project("abc123def456");
        assert_eq!(resp.decay_map.len(), 1);
        assert_eq!(resp.decay_map[0].tenant_id, "abc123def456");
        // Current project gets multiplier 1.0
        assert!((resp.decay_map[0].multiplier - 1.0_f32).abs() < f32::EPSILON);
    }

    // ── DaemonClient construction ─────────────────────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_project_calls() {
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }

    // ── call() timeout dispatch for project methods ───────────────────────────

    #[tokio::test]
    async fn register_project_method_uses_5s_budget() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call(
                "registerProject",
                Some(Duration::from_millis(1)),
                || async {
                    tokio::time::sleep(Duration::from_millis(50)).await;
                    Ok(())
                },
            )
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }

    #[tokio::test]
    async fn heartbeat_method_uses_5s_budget() {
        use std::time::Duration;
        let mut client = DaemonClient::new("http://127.0.0.1:50051").unwrap();
        let result: Result<(), tonic::Status> = client
            .call("heartbeat", Some(Duration::from_millis(1)), || async {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            })
            .await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().code(), tonic::Code::DeadlineExceeded);
    }
}
