//! `search` MCP tool handler.
//!
//! Entry point: [`search_tool`] — takes parsed arguments, dependency handles,
//! and session state; returns a [`CallToolResult`] via [`ok_text`].
//!
//! ## Module layout
//! - `types`         — `SearchResult`, `SearchResponse`, `SearchMode`, `SearchScope`
//! - `options`       — `SearchInput`, `SearchOptions`, defaults
//! - `expansion`     — sparse tag-basket expansion
//! - `exact`         — FTS5 exact search via daemon TextSearchService
//! - `graph_context` — 1-hop graph context enrichment via daemon GraphService
//! - `flow`          — hybrid/semantic/keyword pipeline, fallback search
//!
//! ## Wiring
//! The main `search_tool` function bridges the concrete `DaemonClient` /
//! `QdrantReadClient` / `StateManager` / `SessionState` types to the generic
//! trait bounds used by the pipeline sub-modules so all internal logic stays
//! hermetically testable.

pub mod exact;
pub mod expansion;
pub mod flow;
pub mod flow_collect;
pub mod flow_fallback;
pub mod graph_context;
pub mod options;
pub mod types;

use std::collections::HashMap;

use rmcp::model::CallToolResult;
use serde_json::Value;

use crate::grpc::client::DaemonClient;
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::SharedStateManager;
use crate::tools::envelope::{error_text, ok_text};

pub use self::options::{SearchInput, SearchOptions, DEFAULT_SCORE_THRESHOLD};
pub use self::types::{SearchMode, SearchResponse, SearchResult, SearchScope};

// ---------------------------------------------------------------------------
// DaemonClient adapter impls
// ---------------------------------------------------------------------------

impl flow::EmbedDaemon for DaemonClient {
    fn embed_text(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<Vec<f32>, tonic::Status>> + Send {
        let text = text.to_string();
        async move {
            let resp = DaemonClient::embed_text(self, &text).await?;
            Ok(resp.embedding)
        }
    }

    fn generate_sparse_vector(
        &mut self,
        text: &str,
    ) -> impl std::future::Future<Output = Result<HashMap<u32, f32>, tonic::Status>> + Send {
        let text = text.to_string();
        async move {
            let resp = DaemonClient::generate_sparse_vector(self, &text).await?;
            Ok(resp.indices_values)
        }
    }
}

impl exact::ExactSearchDaemon for DaemonClient {
    fn text_search(
        &mut self,
        request: crate::proto::TextSearchRequest,
    ) -> impl std::future::Future<Output = Result<crate::proto::TextSearchResponse, tonic::Status>> + Send
    {
        DaemonClient::text_search(self, request)
    }
}

impl graph_context::GraphQueryDaemon for DaemonClient {
    fn query_related(
        &mut self,
        request: crate::proto::QueryRelatedRequest,
    ) -> impl std::future::Future<Output = Result<crate::proto::QueryRelatedResponse, tonic::Status>>
           + Send {
        DaemonClient::query_related(self, request)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Execute the `search` MCP tool.
///
/// Dispatches to `search_exact` or `run_search_pipeline` based on `opts.exact`.
/// Fire-and-forget `log_search_event` / `update_search_event` are wired via
/// `daemon` (task 16 methods) — errors swallowed, no effect on output.
///
/// # Send contract
/// `state: &SharedStateManager` is `Send + Sync`.  All SQLite access
/// (expansion keyword pre-computation) happens synchronously before the first
/// `.await`, so no `Connection` reference is held across any await point.
pub async fn search_tool(
    args: &serde_json::Map<String, Value>,
    daemon: &mut DaemonClient,
    qdrant: &QdrantReadClient,
    state: &SharedStateManager,
    session: &SessionState,
) -> CallToolResult {
    let input = match SearchOptions::parse_args(args) {
        Ok(i) => i,
        Err(e) => return error_text(&e),
    };

    let current_branch = session.current_branch.as_deref();
    let mut opts = SearchOptions::from_input(input, current_branch);
    let project_id = resolve_project_id(&opts, session, state);
    // Thread the resolved project_id (which includes the session fallback) back
    // into opts so that exact search (which reads opts.project_id directly) also
    // benefits from the session fallback.  Mirrors TS search-exact.ts which
    // calls `this.resolveProjectId()` to obtain the fallback.
    opts.project_id = project_id.clone();

    // Pre-compute expansion keywords synchronously BEFORE any `.await`.
    // This keeps the future `Send` by ensuring no `&Connection` is held
    // across any await point (see SharedStateManager docs).
    let expansion_keywords: Vec<String> = if !opts.exact {
        let collections = crate::qdrant::filters::determine_collections(
            opts.collection.as_deref(),
            opts.scope.as_str(),
            opts.include_libraries,
        );
        let guard = state.lock();
        expansion::collect_expansion_keywords(
            guard.connection(),
            &opts.query,
            &collections,
            project_id.as_deref(),
        )
        // guard dropped here — no Connection reference crosses an await
    } else {
        Vec::new()
    };

    // Fire-and-forget pre-search event (errors swallowed).
    let event_id = uuid::Uuid::new_v4().to_string();
    fire_log_pre(daemon, &event_id, session, &opts).await;

    let start_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);

    let response = if opts.exact {
        exact::search_exact(daemon, &opts).await
    } else {
        flow::run_search_pipeline(
            daemon,
            qdrant,
            expansion_keywords,
            &opts,
            project_id.as_deref(),
            true, // enable_tag_expansion
        )
        .await
    };

    // Fire-and-forget post-search event (errors swallowed).
    let latency_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
        - start_ms;
    fire_log_post(daemon, &event_id, &response, latency_ms).await;

    ok_text(&response)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve the effective project_id: explicit override > session project_id >
/// cwd-based project detection.
///
/// Mirrors TS `resolveProjectContext` (`search-helpers.ts`): when no explicit
/// `projectId` is supplied it falls back to detecting the current project from
/// `process.cwd()` via the project detector. Without this fallback every search
/// runs with no tenant filter, which (a) loses project isolation and (b) forces
/// the tag-expansion query into an unindexed full scan of the `tags` table
/// (GitHub #83 — the dominant search-latency cost).
///
/// The registry lookup takes a short synchronous SQLite lock that is dropped
/// before this function returns — no guard is held across any `.await`.
fn resolve_project_id(
    opts: &SearchOptions,
    session: &SessionState,
    state: &SharedStateManager,
) -> Option<String> {
    if let Some(p) = opts.project_id.clone() {
        return Some(p);
    }
    if let Some(p) = session.project_id.clone() {
        return Some(p);
    }
    detect_project_id_from_cwd(state)
}

/// Detect the current project's tenant id from the process working directory.
///
/// Mirrors `projectDetector.getProjectInfo(process.cwd())` in TS. Returns the
/// registered tenant id (`watch_folders` longest-prefix match) or `None` when
/// the cwd is not inside a known project.
fn detect_project_id_from_cwd(state: &SharedStateManager) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    let guard = state.lock();
    let info = crate::session::detect_project(&cwd, &guard)?;
    // guard dropped at end of scope — never held across an await.
    info.project_id
}

/// Fire-and-forget: log pre-search event via daemon.
async fn fire_log_pre(
    daemon: &mut DaemonClient,
    event_id: &str,
    session: &SessionState,
    opts: &SearchOptions,
) {
    let _ = daemon
        .log_search_event(
            event_id.to_string(),
            "claude".to_string(),
            "mcp_qdrant".to_string(),
            "search".to_string(),
            None,
            session.project_id.clone(),
            Some(opts.query.clone()),
            None,
            Some(opts.limit as i32),
            None,
            None,
            None,
            None,
            None,
        )
        .await;
}

/// Fire-and-forget: update search event with post-search results.
async fn fire_log_post(
    daemon: &mut DaemonClient,
    event_id: &str,
    response: &SearchResponse,
    latency_ms: i64,
) {
    let top_refs: Vec<serde_json::Value> = response
        .results
        .iter()
        .take(5)
        .map(|r| {
            serde_json::json!({
                "id": r.id,
                "score": (r.score * 1000.0).round() / 1000.0,
                "collection": r.collection,
            })
        })
        .collect();
    let top_refs_str = serde_json::to_string(&top_refs).ok();

    let _ = daemon
        .update_search_event(
            event_id.to_string(),
            response.total as i32,
            latency_ms,
            top_refs_str,
            None,
        )
        .await;
}

// ---------------------------------------------------------------------------
// Tests (sibling file)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../search_tests.rs"]
mod tests;
