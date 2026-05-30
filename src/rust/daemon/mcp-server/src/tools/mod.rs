//! MCP tool handlers, dispatcher, and the rmcp `ServerHandler` implementation.
//!
//! [`ToolsHandler`] implements rmcp `ServerHandler`:
//! - `get_info`: returns `ServerInfo` with name, version, instructions, and
//!   tools-only capabilities (per CC-4).
//! - `list_tools`: delegates to [`definitions::list_tools`].
//! - `call_tool`: routes through [`dispatch::dispatch_tool`] which fires a
//!   heartbeat, checks against [`dispatch::KNOWN_TOOLS`], and calls the
//!   appropriate handler.
//!
//! # Send + Sync contract
//!
//! `ServerHandler` requires `Send + Sync + 'static`.  The tricky dependency
//! is `rusqlite::Connection`, which is `Send` but NOT `Sync` (it uses
//! `RefCell` internally for statement caching).
//!
//! ## Solution: `SharedStateManager`
//!
//! `StateManager` (which owns the connection) is wrapped in
//! [`crate::sqlite::SharedStateManager`], a newtype that adds a
//! `std::sync::Mutex<StateManager>`.  Because `StateManager: Send`, the
//! `Mutex<StateManager>` is `Send + Sync`, and therefore
//! `SharedStateManager: Send + Sync` and `&SharedStateManager: Send`.
//!
//! Handlers that need SQLite access receive `&SharedStateManager`, lock the
//! std-Mutex synchronously, perform their reads, and **drop the guard before
//! any `.await` point**.  This is the safety contract described in
//! `SharedStateManager`'s documentation.
//!
//! ## Remaining mutable state
//!
//! `DaemonClient` and `SessionState` are mutable (gRPC channel state / session
//! bookkeeping) and are therefore guarded by `tokio::sync::Mutex` so that
//! `call_tool`'s `&self` receiver can obtain exclusive access.

pub mod definitions;
pub mod dispatch;
pub mod embedding;
pub mod envelope;
pub mod grep;
pub mod list;
pub mod retrieve;
pub mod rules;
pub mod search;
pub mod store;
#[cfg(test)]
mod tests;

use std::sync::Arc;

use tokio::sync::Mutex;

use rmcp::{
    handler::server::ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, ErrorData, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerInfo,
    },
    service::RequestContext,
    RoleServer,
};

use crate::grpc::client::DaemonClient;
use crate::instructions::INSTRUCTIONS;
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::{server_version_string, SessionState, SERVER_NAME};
use crate::sqlite::{SharedStateManager, StateManager};
use crate::tools::dispatch::{dispatch_tool, DispatchContext};

pub use definitions::list_tools;

// ---------------------------------------------------------------------------
// ToolsHandler
// ---------------------------------------------------------------------------

/// rmcp `ServerHandler` that owns runtime dependencies and dispatches tool calls.
///
/// ## Dependency ownership pattern
///
/// | Dependency       | Type                        | Reason                             |
/// |------------------|-----------------------------|-------------------------------------|
/// | `daemon`         | `Arc<Mutex<DaemonClient>>`  | mutably accessed per call           |
/// | `qdrant`         | `Arc<QdrantReadClient>`     | `Send + Sync`, read-only            |
/// | `state`          | `Arc<SharedStateManager>`   | `Send + Sync` std-Mutex wrapper     |
/// | `session`        | `Arc<Mutex<SessionState>>`  | mutably accessed per call           |
///
/// `SharedStateManager` wraps `StateManager` (which contains a non-`Sync`
/// `rusqlite::Connection`) in a `std::sync::Mutex`, making it `Send + Sync`.
/// Handlers receive `&SharedStateManager`, lock synchronously, do SQLite
/// reads, then drop the guard before any `.await`.
pub struct ToolsHandler {
    daemon: Arc<Mutex<DaemonClient>>,
    qdrant: Arc<QdrantReadClient>,
    state: Arc<SharedStateManager>,
    session: Arc<Mutex<SessionState>>,
}

impl ToolsHandler {
    /// Create a new handler with all runtime dependencies.
    pub fn new(
        daemon: DaemonClient,
        qdrant: QdrantReadClient,
        state: StateManager,
        session: SessionState,
    ) -> Self {
        Self {
            daemon: Arc::new(Mutex::new(daemon)),
            qdrant: Arc::new(qdrant),
            state: Arc::new(SharedStateManager::new(state)),
            session: Arc::new(Mutex::new(session)),
        }
    }

    /// Access the session state (used by transport layer for cleanup).
    pub fn session(&self) -> Arc<Mutex<SessionState>> {
        Arc::clone(&self.session)
    }

    /// Access the daemon client (used by transport layer for cleanup).
    pub fn daemon(&self) -> Arc<Mutex<DaemonClient>> {
        Arc::clone(&self.daemon)
    }

    /// Access the state manager (used by transport layer when needed).
    pub fn state(&self) -> Arc<SharedStateManager> {
        Arc::clone(&self.state)
    }
}

impl ServerHandler for ToolsHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, server_version_string()))
            .with_instructions(INSTRUCTIONS)
    }

    fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, ErrorData>> + Send + '_ {
        std::future::ready(Ok(ListToolsResult::with_all_items(list_tools())))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, ErrorData>> + Send + '_ {
        // Clone Arcs cheaply — no locking here.
        let daemon = Arc::clone(&self.daemon);
        let qdrant = Arc::clone(&self.qdrant);
        let state = Arc::clone(&self.state);
        let session = Arc::clone(&self.session);

        async move {
            // `request.name` is `Cow<'static, str>` — `.as_ref()` gives `&str`.
            let name = request.name.as_ref().to_string();
            // `request.arguments` is already `Option<Map<String,Value>>`.
            let args = request.arguments.clone().unwrap_or_default();

            // Lock mutable deps (tokio Mutex — guards are Send, held across await).
            let mut daemon_guard = daemon.lock().await;
            let mut session_guard = session.lock().await;

            // `state` is Arc<SharedStateManager> — Send + Sync, no locking needed here.
            // Handlers lock the std::sync::Mutex internally when they need SQLite,
            // and MUST drop the guard before any `.await`.
            let mut ctx = DispatchContext {
                daemon: &mut daemon_guard,
                qdrant: &qdrant,
                state: &state,
                session: &mut session_guard,
            };

            Ok(dispatch_tool(&name, &args, &mut ctx).await)
        }
    }
}
