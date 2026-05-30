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

use serde_json::{Map, Value};
use tracing::debug;

use rmcp::model::CallToolResult;

use crate::grpc::client::DaemonClient;
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
pub struct DispatchContext<'a> {
    pub daemon: &'a mut DaemonClient,
    pub qdrant: &'a QdrantReadClient,
    pub state: &'a SharedStateManager,
    pub session: &'a mut SessionState,
}

// ─────────────────────────────────────────────────────────────────────────────
// Fire-and-forget heartbeat helper
// ─────────────────────────────────────────────────────────────────────────────

/// Send one heartbeat tick, swallowing any error.
///
/// Mirrors `sendHeartbeat(sessionState, daemonClient)` called at the top of
/// `dispatchToolCall` in tool-dispatcher.ts:93.
async fn fire_heartbeat(daemon: &mut DaemonClient, session: &SessionState) {
    if let Some(project_id) = session.project_id.as_deref() {
        use crate::proto::HeartbeatRequest;
        let req = HeartbeatRequest {
            project_id: project_id.to_string(),
        };
        // Fire-and-forget — ignore result just like TS sendHeartbeat.
        let _ = DaemonClient::heartbeat(daemon, req).await;
        debug!("heartbeat sent (fire-and-forget)");
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
/// 3. Route to the 7 handlers.
/// 4. Wrap handler panics / errors in `error_text`.
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

    // 2. Unknown-tool check (tool-dispatcher.ts:95-97).
    if !KNOWN_TOOLS.contains(&name) {
        return unknown_tool(name);
    }

    // 3. Route.
    route_tool(name, args, ctx).await
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
            search_tool(args, ctx.daemon, ctx.qdrant, ctx.state, ctx.session).await
        }
        "retrieve" => {
            let input = RetrieveInput::from_args(args);
            retrieve_tool(input, ctx.qdrant).await
        }
        "rules" => {
            let input = match RulesInput::from_args(args) {
                Ok(i) => i,
                Err(e) => return error_text(&e),
            };
            let session_project_id = ctx.session.project_id.as_deref();
            rules_tool(input, ctx.daemon, ctx.state, ctx.qdrant, session_project_id).await
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
