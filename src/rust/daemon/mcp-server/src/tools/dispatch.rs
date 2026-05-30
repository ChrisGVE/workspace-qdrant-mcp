//! Central tool dispatcher — mirrors `tool-dispatcher.ts`.
//!
//! [`dispatch_tool`] is the single entry point for all `tools/call` requests.
//! It fires an implicit heartbeat (fire-and-forget, matching TS `sendHeartbeat`
//! before routing), validates the tool name against [`KNOWN_TOOLS`], routes to
//! the appropriate handler, and wraps errors in the envelope contract defined
//! by [`crate::tools::envelope`].
//!
//! # TS parity notes
//! - `KNOWN_TOOLS` = `['search','retrieve','rules','store','grep','list','embedding']`
//!   (tool-dispatcher.ts:29).
//! - Unknown name → `{ content:[{type:text, text:"Unknown tool: <name>"}], isError:true }`
//!   (tool-dispatcher.ts:97).
//! - All errors (handler throws) → `error_text` with `isError:true`
//!   (tool-dispatcher.ts:106-110) — RISK-19/20.
//! - `store` subtypes: "project" / "url" / "scratchpad" / default-library
//!   (tool-dispatcher.ts:37-43, dispatchStore).
//! - Heartbeat is fire-and-forget — failures must NOT propagate.
//! - Metrics instrumentation mirrors `withToolMetrics` in `telemetry/metrics.ts:129-141`.
//!   `status="success"` when `is_error != Some(true)`; `status="error"` otherwise.
//!   Duration is measured with `std::time::Instant` (monotonic, not wall-clock).

use std::time::Instant;

use serde_json::{Map, Value};
use tracing::debug;

use rmcp::model::CallToolResult;

use crate::grpc::client::DaemonClient;
use crate::observability::health_monitor::{augment_search_results, SharedHealthState};
use crate::observability::metrics::record_tool_call;
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::SharedStateManager;
use crate::tools::embedding::embedding_tool;
use crate::tools::envelope::{error_text, unknown_tool};
use crate::tools::grep::{grep_tool, GrepInput};
use crate::tools::list::{list_tool, ListInput};
use crate::tools::retrieve::{retrieve_tool, RetrieveInput};
use crate::tools::rules::{rules_tool, RulesInput};
use crate::tools::search::search_tool;
use crate::tools::store::{store_tool, StoreInput};

// ─────────────────────────────────────────────────────────────────────────────
// KNOWN_TOOLS — mirrors KNOWN_TOOLS in tool-dispatcher.ts:29
// ─────────────────────────────────────────────────────────────────────────────

/// The 7 tool names this dispatcher recognises.
///
/// Any `call_tool` request with a name not in this slice gets the
/// `unknown_tool` envelope (isError:true), matching TS line 95-97.
pub const KNOWN_TOOLS: &[&str] = &[
    "search",
    "retrieve",
    "rules",
    "store",
    "grep",
    "list",
    "embedding",
];

// ─────────────────────────────────────────────────────────────────────────────
// DispatchContext — bundles deps for all 7 handlers
// ─────────────────────────────────────────────────────────────────────────────

/// Borrowed references to the runtime dependencies needed by tool handlers.
///
/// All fields are mutable where handlers require `&mut` (gRPC calls mutate
/// the underlying channel state).  `qdrant` is `Send + Sync` and held by
/// shared reference.  `state` is a [`SharedStateManager`] (`Send + Sync`)
/// rather than a bare `&StateManager` so that the dispatch future satisfies
/// the `Send` bound required by `rmcp::ServerHandler`.
///
/// `health_state` is the [`SharedHealthState`] from the background health
/// monitor.  The search arm uses it to call [`augment_search_results`],
/// mirroring `healthMonitor.augmentSearchResults(...)` in
/// `tool-dispatcher.ts:66`.
pub struct DispatchContext<'a> {
    pub daemon: &'a mut DaemonClient,
    pub qdrant: &'a QdrantReadClient,
    pub state: &'a SharedStateManager,
    pub session: &'a mut SessionState,
    pub health_state: &'a SharedHealthState,
    /// Optional duplication threshold for the rules tool.
    ///
    /// Mirrors `config.rules?.duplicationThreshold` from TS server-factory.ts:52.
    /// When `None`, the rules tool uses `DEFAULT_DUPLICATION_THRESHOLD`.
    pub rules_dup_threshold: Option<f64>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fire-and-forget heartbeat helper
// ─────────────────────────────────────────────────────────────────────────────

/// Send one heartbeat tick.
///
/// Mirrors `sendHeartbeat(sessionState, daemonClient)` called at the top of
/// `dispatchToolCall` in tool-dispatcher.ts:93.
///
/// On heartbeat failure TS sets `sessionState.daemonConnected = false` and
/// records `logDaemonStatus(false, { reason: 'heartbeat_failed' })`
/// (session-lifecycle.ts:254-258).  We mirror that by flipping
/// `session.daemon_connected = false` on any RPC error.
pub(crate) async fn fire_heartbeat(daemon: &mut DaemonClient, session: &mut SessionState) {
    // TS sendHeartbeat (session-lifecycle.ts:239) returns early unless BOTH a
    // project_id is set AND the daemon is currently connected — mirror that
    // guard so we don't fire a doomed RPC while disconnected.
    if !session.daemon_connected {
        return;
    }
    if let Some(project_id) = session.project_id.as_deref() {
        use crate::proto::HeartbeatRequest;
        let req = HeartbeatRequest {
            project_id: project_id.to_string(),
        };
        match DaemonClient::heartbeat(daemon, req).await {
            Ok(_) => {
                debug!("heartbeat sent (fire-and-forget)");
            }
            Err(_) => {
                // Mirror TS session-lifecycle.ts:254-258: mark daemon disconnected.
                session.daemon_connected = false;
                debug!("heartbeat failed — daemon_connected set to false");
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main dispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatch a `tools/call` request to the correct handler.
///
/// Steps (mirrors `dispatchToolCall` in tool-dispatcher.ts):
/// 1. Fire heartbeat — fire-and-forget, no latency impact.
/// 2. Reject unknown names → `unknown_tool` envelope.
/// 3. Route to the 7 handlers, wrapped in Prometheus instrumentation that
///    mirrors `withToolMetrics(toolName, fn)` in `telemetry/metrics.ts:129-141`:
///    - records `wqm_mcp_tool_invocations_total{tool, status="success"|"error"}`
///    - observes `wqm_mcp_tool_duration_seconds{tool}`
///    - `status="error"` when `result.is_error == Some(true)` (in-band errors);
///      exceptions/panics never reach here because handlers return `CallToolResult`.
///
/// `args` is the raw JSON `arguments` object from the MCP request (may be
/// empty; never null after our normalisation in `ToolsHandler::call_tool`).
pub async fn dispatch_tool(
    name: &str,
    args: &Map<String, Value>,
    ctx: &mut DispatchContext<'_>,
) -> CallToolResult {
    // 1. Heartbeat — fire and forget (tool-dispatcher.ts:93).
    fire_heartbeat(ctx.daemon, ctx.session).await;
    // Note: fire_heartbeat may have flipped ctx.session.daemon_connected=false.

    // 2. Unknown-tool check (tool-dispatcher.ts:95-97).
    if !KNOWN_TOOLS.contains(&name) {
        return unknown_tool(name);
    }

    // 3. Route with metrics instrumentation (mirrors withToolMetrics).
    let t = Instant::now();
    let result = route_tool(name, args, ctx).await;
    let elapsed = t.elapsed().as_secs_f64();
    let status = if result.is_error == Some(true) {
        "error"
    } else {
        "success"
    };
    record_tool_call(name, status, elapsed);
    result
}

/// Augment the text payload of a search `CallToolResult` with health metadata.
///
/// Mirrors `healthMonitor.augmentSearchResults({ success: true, ...searchResult })`
/// at tool-dispatcher.ts:66.
///
/// - Parses the pretty-JSON text from the first content block.
/// - Calls [`augment_search_results`] which inserts `"health": {...}` only when
///   the system is uncertain.
/// - Re-serialises back to pretty JSON and updates the content block in place.
/// - If the health state RwLock is poisoned, or if JSON round-trip fails, the
///   original result is returned unchanged (defensive — should never occur).
fn apply_health_augmentation(
    mut result: CallToolResult,
    health_state: &SharedHealthState,
) -> CallToolResult {
    let state_guard = match health_state.read() {
        Ok(g) => g,
        Err(_) => return result,
    };
    // Only augment when there is content and it is text.
    if let Some(item) = result.content.first_mut() {
        if let Some(text_content) = item.raw.as_text() {
            let original = text_content.text.as_str();
            if let Ok(json_val) = serde_json::from_str::<Value>(original) {
                let augmented = augment_search_results(&state_guard, json_val);
                if let Ok(pretty) = serde_json::to_string_pretty(&augmented) {
                    // Replace in-place via ok_text round-trip.
                    use rmcp::model::Content;
                    *item = Content::text(pretty);
                }
            }
        }
    }
    result
}

/// Route a validated tool name to its handler.
///
/// Mirrors `routeTool` in tool-dispatcher.ts:53-83.
/// Returns `CallToolResult` — errors from handlers are returned in-band via
/// `error_text` (tool-dispatcher.ts:106-109) rather than propagated.
async fn route_tool(
    name: &str,
    args: &Map<String, Value>,
    ctx: &mut DispatchContext<'_>,
) -> CallToolResult {
    match name {
        "search" => {
            // search_tool handles its own error wrapping.
            let raw = search_tool(args, ctx.daemon, ctx.qdrant, ctx.state, ctx.session).await;
            // Augment with health metadata when system is uncertain
            // (mirrors tool-dispatcher.ts:66: healthMonitor.augmentSearchResults({success:true,...})).
            apply_health_augmentation(raw, ctx.health_state)
        }
        "retrieve" => {
            let input = RetrieveInput::from_args(args);
            let session_project_id = ctx.session.project_id.as_deref();
            retrieve_tool(input, ctx.qdrant, session_project_id).await
        }
        "rules" => {
            let input = match RulesInput::from_args(args) {
                Ok(i) => i,
                Err(e) => return error_text(&e),
            };
            let session_project_id = ctx.session.project_id.as_deref();
            rules_tool(
                input,
                ctx.daemon,
                ctx.state,
                ctx.qdrant,
                session_project_id,
                ctx.rules_dup_threshold,
            )
            .await
        }
        "store" => {
            let session_project_id = ctx.session.project_id.as_deref();
            let input = StoreInput::from_args(args, session_project_id);
            let daemon_connected = ctx.session.daemon_connected;
            store_tool(input, ctx.daemon, session_project_id, daemon_connected).await
        }
        "grep" => {
            let input = GrepInput::from_args(args);
            let session_project_id = ctx.session.project_id.as_deref();
            grep_tool(input, ctx.daemon, session_project_id).await
        }
        "list" => {
            let input = ListInput::from_args(args);
            list_tool(input, ctx.state, ctx.session)
        }
        "embedding" => embedding_tool(ctx.daemon).await,
        // This branch is unreachable after the KNOWN_TOOLS guard, but Rust
        // requires exhaustiveness.
        _ => error_text(&format!("Internal: unhandled tool '{name}'")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
