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

// The SQLite-free search pipeline now lives in the shared `wqm-client` crate
// (`wqm_client::search`, WI-d4 #82). Re-export the moved modules so existing
// `crate::tools::search::{flow,exact,…}::…` paths — and the hermetic stub tests
// that exercise them — keep resolving against the now-foreign types. The
// `DaemonClient` adapter impls for `EmbedDaemon` / `ExactSearchDaemon` /
// `GraphQueryDaemon` moved with the traits (orphan rule).
pub use wqm_client::search::{
    exact, flow, flow_collect, flow_fallback, graph_context, graph_fusion, options,
};

// SQLite-bound adapters (base-point resolution, tag-basket keyword collection)
// stay local — they read the state DB and pre-resolve owned values for the
// pipeline.
pub mod expansion;
pub mod scope;
pub mod types;

use std::collections::HashMap;

use rmcp::model::CallToolResult;
use serde_json::Value;

use crate::grpc::client::DaemonClient;
use crate::proto::ResolveSearchScopeRequest;
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::SharedStateManager;
use crate::tools::envelope::{error_text, ok_text};

pub use self::options::{SearchInput, SearchOptions, DEFAULT_SCORE_THRESHOLD};
pub use self::types::{SearchMode, SearchResponse, SearchResult, SearchScope};

// ---------------------------------------------------------------------------
// Fallback metrics hook
// ---------------------------------------------------------------------------

/// Routes the shared pipeline's `FallbackMetrics` calls to the MCP server's
/// Prometheus counter. The pipeline depends only on the trait; the concrete
/// registry stays here.
struct PrometheusFallback;

impl wqm_client::search::FallbackMetrics for PrometheusFallback {
    fn record_daemon_fallback(&self, tool: &str, reason: &str) {
        crate::observability::metrics::record_daemon_fallback(tool, reason);
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

    // Pre-compute expansion keywords + base points synchronously BEFORE any
    // `.await`. Both read SQLite under a short lock that is dropped before the
    // first await — no `&Connection` ever crosses an await (SharedStateManager).
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let (expansion_keywords, base_points, bp_degraded, bp_active) = {
        let guard = state.lock();
        let conn = guard.connection();
        let keywords = if !opts.exact {
            let collections = crate::qdrant::filters::determine_collections(
                opts.collection.as_deref(),
                opts.scope.as_str(),
                opts.include_libraries,
            );
            expansion::collect_expansion_keywords(
                conn,
                &opts.query,
                &collections,
                project_id.as_deref(),
            )
        } else {
            Vec::new()
        };
        let (bp, degraded, count) =
            scope::resolve_base_points(conn, project_id.as_deref(), opts.scope, &cwd);
        (keywords, bp, degraded, count)
        // guard dropped here — no Connection reference crosses an await
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
        // Resolve scope (group/all) tenant filter + relevance decay via daemon.
        let (group_tenant_ids, decay_map) =
            resolve_scope_filter(daemon, project_id.as_deref(), opts.scope).await;
        let scope_ctx = scope::ScopeContext {
            group_tenant_ids,
            base_points,
            decay_map,
            base_points_degraded: bp_degraded,
            base_points_active_count: bp_active,
        };

        // scope=group with no resolved membership → refusal (TS search.ts).
        let group_empty = scope_ctx
            .group_tenant_ids
            .as_ref()
            .map_or(true, |v| v.is_empty());
        if opts.scope == SearchScope::Group && group_empty {
            group_refusal_response(&opts)
        } else {
            let mut resp = flow::run_search_pipeline(
                daemon,
                qdrant,
                expansion_keywords,
                &opts,
                project_id.as_deref(),
                true, // enable_tag_expansion
                &scope_ctx,
                &PrometheusFallback,
            )
            .await;
            // F-014: base-point isolation degraded → uncertain + reason.
            if scope_ctx.base_points_degraded {
                apply_base_points_degraded(&mut resp, scope_ctx.base_points_active_count);
            }
            resp
        }
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
    // Normalize blank IDs to unresolved (TS treats `''` as falsy `!projectId`),
    // so empty strings never trigger scope/decay/base-point resolution or build a
    // degenerate single-tenant filter downstream.
    let non_blank = |s: String| (!s.trim().is_empty()).then_some(s);
    if let Some(p) = opts.project_id.clone().and_then(non_blank) {
        return Some(p);
    }
    if let Some(p) = session.project_id.clone().and_then(non_blank) {
        return Some(p);
    }
    detect_project_id_from_cwd(state)
}

/// Detect the current project's tenant id from the process working directory.
///
/// Mirrors TS `projectDetector.getCurrentProject(process.cwd())`, which passes
/// cwd **directly** to the `watch_folders` longest-prefix lookup
/// (`getProjectInfo`: "Pass cwd directly — the database query uses
/// longest-prefix matching to resolve subdirectories to their registered
/// project root"). Returns the registered tenant id or `None` when the cwd is
/// not inside a known project.
///
/// IMPORTANT: this must NOT first normalize cwd to a filesystem project-root
/// (via `find_project_root`). A registered project path need not contain a
/// project marker (registration accepts raw paths), so a marker-based walk can
/// skip past a deeper registered project and resolve the wrong (ancestor)
/// tenant. The longest-prefix SQL already resolves subdirectories correctly —
/// see the `resolve_cwd_project_id_*` tests.
fn detect_project_id_from_cwd(state: &SharedStateManager) -> Option<String> {
    let cwd = std::env::current_dir().ok()?;
    resolve_cwd_project_id_locked(&cwd, state)
}

/// Resolve a tenant id for `cwd` by longest-prefix `watch_folders` match.
///
/// The SQLite lock is acquired only for the lookup and dropped at scope end —
/// never held across an `.await`.
fn resolve_cwd_project_id_locked(
    cwd: &std::path::Path,
    state: &SharedStateManager,
) -> Option<String> {
    let guard = state.lock();
    crate::session::lookup_project_id(&guard, cwd)
}

/// Resolve the group/all scope tenant filter + relevance decay via the daemon.
///
/// Mirrors TS `resolveScopeFilter`: no daemon call for `scope=project` or when
/// the project is unresolved; otherwise call `resolveSearchScope` and build the
/// `(group_tenant_ids, decay_map)` pair. Any daemon error degrades to
/// `(None, None)` (best-effort, matching the TS `catch`).
async fn resolve_scope_filter(
    daemon: &mut DaemonClient,
    project_id: Option<&str>,
    scope: SearchScope,
) -> (Option<Vec<String>>, Option<HashMap<String, f64>>) {
    let project_id = match (scope, project_id) {
        (SearchScope::Project, _) | (_, None) => return (None, None),
        (_, Some(p)) => p,
    };
    let req = ResolveSearchScopeRequest {
        tenant_id: project_id.to_string(),
        scope: scope.as_str().to_string(),
    };
    match daemon.resolve_search_scope(req).await {
        Ok(resp) => scope::scope_filter_from_response(&resp),
        Err(_) => (None, None),
    }
}

/// Build the `status='error'` refusal response for empty group membership
/// (TS `search.ts`).
fn group_refusal_response(opts: &SearchOptions) -> SearchResponse {
    let collections = crate::qdrant::filters::determine_collections(
        opts.collection.as_deref(),
        opts.scope.as_str(),
        opts.include_libraries,
    );
    SearchResponse {
        results: Vec::new(),
        total: 0,
        query: opts.query.clone(),
        mode: opts.mode,
        scope: opts.scope,
        collections_searched: collections,
        status: Some("error".to_string()),
        status_reason: Some(scope::GROUP_EMPTY_REFUSAL.to_string()),
        // TS group-refusal object omits `branch` (search.ts:246-256).
        branch: None,
        diversity_score: None,
    }
}

/// Mark a response `uncertain` and append the F-014 base-point-degraded reason.
fn apply_base_points_degraded(resp: &mut SearchResponse, active_count: Option<usize>) {
    resp.status = Some("uncertain".to_string());
    let reason = scope::format_base_points_degraded_reason(active_count);
    resp.status_reason = Some(match resp.status_reason.take() {
        Some(existing) if !existing.is_empty() => format!("{existing}; {reason}"),
        _ => reason,
    });
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
