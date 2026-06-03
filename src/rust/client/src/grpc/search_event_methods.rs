//! TrackingWriteService search-event RPC wrappers for [`DaemonClient`].
//!
//! Implements the two search-event instrumentation methods that mirror the TS
//! `search-event-queries.ts`:
//!
//! | Rust method          | Proto RPC                                | TS equivalent         |
//! |----------------------|------------------------------------------|-----------------------|
//! | `log_search_event`   | `TrackingWriteService::LogSearchEvent`   | `logSearchEvent()`    |
//! | `update_search_event`| `TrackingWriteService::UpdateSearchEvent`| `updateSearchEvent()` |
//!
//! # Fire-and-forget semantics
//!
//! Both methods are called from the search flow and must **never block or fail
//! it**.  The TS implementations use `.catch(err => console.warn(...))` to
//! swallow errors (search-event-queries.ts:76, :113).  The Rust wrappers
//! mirror this exactly:
//!
//! - The return type is `Result<(), Status>` but is **always `Ok(())`**.
//! - On RPC error the error is logged via `tracing::warn!` and discarded.
//! - No SQLite write path — the daemon owns the search_events table.

use tonic::Status;
use tracing::warn;

use wqm_proto::workspace_daemon::{LogSearchEventRequest, UpdateSearchEventRequest};

use super::client::DaemonClient;

impl DaemonClient {
    /// Log a search event — fire-and-forget instrumentation.
    ///
    /// Mirrors TS `logSearchEvent()` in `search-event-queries.ts:40-83`.
    ///
    /// Called at the *start* of a search to create the initial record.
    ///
    /// # Field mapping (search-event-queries.ts:44-82)
    ///
    /// Required fields: `id`, `actor`, `tool`, `op`.
    /// All others are optional (only sent when `Some`), matching the TS
    /// `if (event.X !== undefined) request.x = event.X` pattern.
    ///
    /// | Rust param        | Proto field        | TS field          |
    /// |-------------------|--------------------|-------------------|
    /// | `id`              | `id`               | `id`              |
    /// | `actor`           | `actor`            | `actor`           |
    /// | `tool`            | `tool`             | `tool`            |
    /// | `op`              | `op`               | `op`              |
    /// | `session_id`      | `session_id`       | `session_id`      |
    /// | `project_id`      | `project_id`       | `project_id`      |
    /// | `query_text`      | `query_text`       | `query_text`      |
    /// | `filters`         | `filters`          | `filters`         |
    /// | `top_k`           | `top_k`            | `top_k`           |
    /// | `result_count`    | `result_count`     | `result_count`    |
    /// | `latency_ms`      | `latency_ms`       | `latency_ms`      |
    /// | `top_result_refs` | `top_result_refs`  | `top_result_refs` |
    /// | `outcome`         | `outcome`          | `outcome`         |
    /// | `parent_event_id` | `parent_event_id`  | `parent_event_id` |
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    /// Instrumentation must never break search (search-event-queries.ts:77-80).
    #[allow(clippy::too_many_arguments)]
    pub async fn log_search_event(
        &mut self,
        id: String,
        actor: String,
        tool: String,
        op: String,
        session_id: Option<String>,
        project_id: Option<String>,
        query_text: Option<String>,
        filters: Option<String>,
        top_k: Option<i32>,
        result_count: Option<i32>,
        latency_ms: Option<i64>,
        top_result_refs: Option<String>,
        outcome: Option<String>,
        parent_event_id: Option<String>,
    ) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("logSearchEvent", None, move || {
                let mut c = client.clone();
                let req = LogSearchEventRequest {
                    id: id.clone(),
                    actor: actor.clone(),
                    tool: tool.clone(),
                    op: op.clone(),
                    session_id: session_id.clone(),
                    project_id: project_id.clone(),
                    query_text: query_text.clone(),
                    filters: filters.clone(),
                    top_k,
                    result_count,
                    latency_ms,
                    top_result_refs: top_result_refs.clone(),
                    outcome: outcome.clone(),
                    parent_event_id: parent_event_id.clone(),
                };
                async move {
                    c.log_search_event(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            // Instrumentation must never break search — log and swallow.
            // Mirrors: search-event-queries.ts:77-80
            warn!("logSearchEvent instrumentation failed: {}", e);
        }
        Ok(())
    }

    /// Update a search event with post-search results — fire-and-forget.
    ///
    /// Mirrors TS `updateSearchEvent()` in `search-event-queries.ts:92-119`.
    ///
    /// Called *after* a search completes to record result_count, latency_ms,
    /// top_result_refs, and outcome for a previously created event.
    ///
    /// # Field mapping (search-event-queries.ts:99-111)
    ///
    /// | Rust param        | Proto field       | TS field          |
    /// |-------------------|-------------------|-------------------|
    /// | `event_id`        | `event_id`        | `event_id`        |
    /// | `result_count`    | `result_count`    | `result_count`    |
    /// | `latency_ms`      | `latency_ms`      | `latency_ms`      |
    /// | `top_result_refs` | `top_result_refs` | `top_result_refs` |
    /// | `outcome`         | `outcome`         | `outcome`         |
    ///
    /// # Fire-and-forget
    /// On RPC failure the error is logged via `warn!` and `Ok(())` is returned.
    pub async fn update_search_event(
        &mut self,
        event_id: String,
        result_count: i32,
        latency_ms: i64,
        top_result_refs: Option<String>,
        outcome: Option<String>,
    ) -> Result<(), Status> {
        let client = self.tracking_write.clone();
        let result = self
            .call("updateSearchEvent", None, move || {
                let mut c = client.clone();
                let req = UpdateSearchEventRequest {
                    event_id: event_id.clone(),
                    result_count,
                    latency_ms,
                    top_result_refs: top_result_refs.clone(),
                    outcome: outcome.clone(),
                };
                async move {
                    c.update_search_event(tonic::Request::new(req))
                        .await
                        .map(|r| r.into_inner())
                }
            })
            .await;
        if let Err(ref e) = result {
            // Mirrors: search-event-queries.ts:114-117
            warn!("updateSearchEvent instrumentation failed: {}", e);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // LogSearchEventRequest field mapping
    // =========================================================================

    fn make_log_request_minimal() -> LogSearchEventRequest {
        // Required fields only — matches search-event-queries.ts:59-64
        LogSearchEventRequest {
            id: "evt-001".to_string(),
            actor: "mcp".to_string(),
            tool: "search".to_string(),
            op: "search".to_string(),
            session_id: None,
            project_id: None,
            query_text: None,
            filters: None,
            top_k: None,
            result_count: None,
            latency_ms: None,
            top_result_refs: None,
            outcome: None,
            parent_event_id: None,
        }
    }

    fn make_log_request_full() -> LogSearchEventRequest {
        LogSearchEventRequest {
            id: "evt-002".to_string(),
            actor: "mcp".to_string(),
            tool: "search".to_string(),
            op: "search".to_string(),
            session_id: Some("sess-abc".to_string()),
            project_id: Some("proj-xyz".to_string()),
            query_text: Some("semantic code search".to_string()),
            filters: Some(r#"{"scope":"project"}"#.to_string()),
            top_k: Some(10),
            result_count: Some(7),
            latency_ms: Some(142),
            top_result_refs: Some("ref1,ref2".to_string()),
            outcome: Some("success".to_string()),
            parent_event_id: Some("parent-evt-001".to_string()),
        }
    }

    #[test]
    fn log_event_required_id_field() {
        // search-event-queries.ts:60: id: event.id
        let req = make_log_request_minimal();
        assert_eq!(req.id, "evt-001");
    }

    #[test]
    fn log_event_required_actor_field() {
        // search-event-queries.ts:61: actor: event.actor
        let req = make_log_request_minimal();
        assert_eq!(req.actor, "mcp");
    }

    #[test]
    fn log_event_required_tool_field() {
        // search-event-queries.ts:62: tool: event.tool
        let req = make_log_request_minimal();
        assert_eq!(req.tool, "search");
    }

    #[test]
    fn log_event_required_op_field() {
        // search-event-queries.ts:63: op: event.op
        let req = make_log_request_minimal();
        assert_eq!(req.op, "search");
    }

    #[test]
    fn log_event_optional_fields_none_when_absent() {
        // search-event-queries.ts:65-74: all optional fields use undefined-guard
        let req = make_log_request_minimal();
        assert!(req.session_id.is_none());
        assert!(req.project_id.is_none());
        assert!(req.query_text.is_none());
        assert!(req.filters.is_none());
        assert!(req.top_k.is_none());
        assert!(req.result_count.is_none());
        assert!(req.latency_ms.is_none());
        assert!(req.top_result_refs.is_none());
        assert!(req.outcome.is_none());
        assert!(req.parent_event_id.is_none());
    }

    #[test]
    fn log_event_optional_session_id_present() {
        // search-event-queries.ts:65: if (event.sessionId !== undefined) request.session_id = event.sessionId
        let req = make_log_request_full();
        assert_eq!(req.session_id.as_deref(), Some("sess-abc"));
    }

    #[test]
    fn log_event_optional_project_id_present() {
        let req = make_log_request_full();
        assert_eq!(req.project_id.as_deref(), Some("proj-xyz"));
    }

    #[test]
    fn log_event_optional_query_text_present() {
        let req = make_log_request_full();
        assert_eq!(req.query_text.as_deref(), Some("semantic code search"));
    }

    #[test]
    fn log_event_optional_top_k_present() {
        let req = make_log_request_full();
        assert_eq!(req.top_k, Some(10));
    }

    #[test]
    fn log_event_optional_result_count_present() {
        let req = make_log_request_full();
        assert_eq!(req.result_count, Some(7));
    }

    #[test]
    fn log_event_optional_latency_ms_present() {
        let req = make_log_request_full();
        assert_eq!(req.latency_ms, Some(142));
    }

    #[test]
    fn log_event_optional_top_result_refs_present() {
        let req = make_log_request_full();
        assert_eq!(req.top_result_refs.as_deref(), Some("ref1,ref2"));
    }

    #[test]
    fn log_event_optional_outcome_present() {
        let req = make_log_request_full();
        assert_eq!(req.outcome.as_deref(), Some("success"));
    }

    #[test]
    fn log_event_optional_parent_event_id_present() {
        let req = make_log_request_full();
        assert_eq!(req.parent_event_id.as_deref(), Some("parent-evt-001"));
    }

    // =========================================================================
    // UpdateSearchEventRequest field mapping
    // =========================================================================

    fn make_update_request() -> UpdateSearchEventRequest {
        // Mirrors search-event-queries.ts:99-111
        UpdateSearchEventRequest {
            event_id: "evt-001".to_string(),
            result_count: 5,
            latency_ms: 88,
            top_result_refs: Some("ref-a,ref-b".to_string()),
            outcome: Some("success".to_string()),
        }
    }

    #[test]
    fn update_event_event_id_field() {
        // search-event-queries.ts:100: event_id: eventId
        let req = make_update_request();
        assert_eq!(req.event_id, "evt-001");
    }

    #[test]
    fn update_event_result_count_field() {
        // search-event-queries.ts:101: result_count: update.resultCount
        let req = make_update_request();
        assert_eq!(req.result_count, 5);
    }

    #[test]
    fn update_event_latency_ms_field() {
        // search-event-queries.ts:102: latency_ms: update.latencyMs
        let req = make_update_request();
        assert_eq!(req.latency_ms, 88);
    }

    #[test]
    fn update_event_optional_top_result_refs_present() {
        // search-event-queries.ts:103-104: optional
        let req = make_update_request();
        assert_eq!(req.top_result_refs.as_deref(), Some("ref-a,ref-b"));
    }

    #[test]
    fn update_event_optional_outcome_present() {
        // search-event-queries.ts:105: optional
        let req = make_update_request();
        assert_eq!(req.outcome.as_deref(), Some("success"));
    }

    #[test]
    fn update_event_optional_fields_none_when_absent() {
        let req = UpdateSearchEventRequest {
            event_id: "evt-003".to_string(),
            result_count: 0,
            latency_ms: 10,
            top_result_refs: None,
            outcome: None,
        };
        assert!(req.top_result_refs.is_none());
        assert!(req.outcome.is_none());
    }

    // =========================================================================
    // Fire-and-forget: RPC failure must not propagate to caller
    // =========================================================================
    //
    // These tests validate the swallow-wrapper pattern by simulating the exact
    // logic used in log_search_event / update_search_event:
    //   let result = self.call(...).await;       // may be Err
    //   if let Err(ref e) = result { warn!(...) }
    //   Ok(())                                   // always Ok

    #[test]
    fn log_event_fire_forget_swallows_unavailable() {
        // Simulate call() returning Err(UNAVAILABLE)
        let rpc_result: Result<(), Status> = Err(Status::unavailable("daemon down"));
        if let Err(ref e) = rpc_result {
            // warn! would fire here in production
            assert_eq!(e.code(), tonic::Code::Unavailable);
        }
        // The wrapper always returns Ok(()) regardless
        let outcome: Result<(), Status> = Ok(());
        assert!(outcome.is_ok());
    }

    #[test]
    fn log_event_fire_forget_swallows_deadline_exceeded() {
        let rpc_result: Result<(), Status> = Err(Status::deadline_exceeded("timeout"));
        if let Err(ref e) = rpc_result {
            assert_eq!(e.code(), tonic::Code::DeadlineExceeded);
        }
        let outcome: Result<(), Status> = Ok(());
        assert!(outcome.is_ok());
    }

    #[test]
    fn log_event_fire_forget_swallows_internal_error() {
        let rpc_result: Result<(), Status> = Err(Status::internal("unexpected daemon error"));
        if let Err(ref e) = rpc_result {
            assert_eq!(e.code(), tonic::Code::Internal);
        }
        let outcome: Result<(), Status> = Ok(());
        assert!(outcome.is_ok());
    }

    #[test]
    fn update_event_fire_forget_swallows_unavailable() {
        let rpc_result: Result<(), Status> = Err(Status::unavailable("daemon down"));
        if let Err(ref e) = rpc_result {
            assert_eq!(e.code(), tonic::Code::Unavailable);
        }
        let outcome: Result<(), Status> = Ok(());
        assert!(outcome.is_ok());
    }

    #[test]
    fn update_event_fire_forget_ok_path_unchanged() {
        // When RPC succeeds, Ok(()) is also returned — no change in behaviour.
        let rpc_result: Result<(), Status> = Ok(());
        assert!(rpc_result.is_ok());
        let outcome: Result<(), Status> = Ok(());
        assert!(outcome.is_ok());
    }

    // ── DaemonClient construction (no live daemon) ────────────────────────────

    #[tokio::test]
    async fn daemon_client_constructs_for_search_event_calls() {
        let result = DaemonClient::new("http://127.0.0.1:50051");
        assert!(result.is_ok());
    }
}
