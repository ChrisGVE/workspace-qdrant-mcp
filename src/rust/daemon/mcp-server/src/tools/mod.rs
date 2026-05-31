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

#[cfg(test)]
#[path = "config_wiring_tests.rs"]
mod config_wiring_tests;
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
use tokio::task::AbortHandle;

use rmcp::{
    handler::server::ServerHandler,
    model::{
        CallToolRequestParams, CallToolResult, ErrorData, Implementation, InitializeRequestParams,
        InitializeResult, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerInfo,
    },
    service::RequestContext,
    RoleServer,
};

use crate::grpc::client::DaemonClient;
use crate::instructions::INSTRUCTIONS;
use crate::observability::health_monitor::SharedHealthState;
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
/// | `health_state`   | `SharedHealthState`         | background health monitor state     |
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
    health_state: SharedHealthState,
    /// Optional duplication threshold override for the rules tool.
    /// Sourced from `WQM_RULES_DEDUP_THRESHOLD` env var via `ServerConfig`.
    rules_dup_threshold: Option<f64>,
    /// Heartbeat task handle, set when the `initialize` lifecycle starts the
    /// heartbeat loop; read+aborted by the transport at cleanup. `std::sync::Mutex`
    /// because it is only ever locked synchronously (no `.await` held).
    hb_handle: Arc<std::sync::Mutex<Option<AbortHandle>>>,
}

impl ToolsHandler {
    /// Create a new handler with all runtime dependencies.
    ///
    /// `health_state` is the [`SharedHealthState`] produced by a running
    /// [`HealthMonitorBuilder`](crate::observability::health_monitor::HealthMonitorBuilder).
    /// Pass a default optimistic state (`Arc::new(RwLock::new(...))`) when no
    /// background monitor is running (e.g., in tests).
    pub fn new(
        daemon: DaemonClient,
        qdrant: QdrantReadClient,
        state: StateManager,
        session: SessionState,
        health_state: SharedHealthState,
    ) -> Self {
        Self::new_with_config(daemon, qdrant, state, session, health_state, None)
    }

    /// Create a handler with an optional rules duplication threshold override.
    pub fn new_with_config(
        daemon: DaemonClient,
        qdrant: QdrantReadClient,
        state: StateManager,
        session: SessionState,
        health_state: SharedHealthState,
        rules_dup_threshold: Option<f64>,
    ) -> Self {
        Self {
            daemon: Arc::new(Mutex::new(daemon)),
            qdrant: Arc::new(qdrant),
            state: Arc::new(SharedStateManager::new(state)),
            session: Arc::new(Mutex::new(session)),
            health_state,
            rules_dup_threshold,
            hb_handle: Arc::new(std::sync::Mutex::new(None)),
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

    /// Create a handler from already-Arc-wrapped shared dependencies.
    ///
    /// Used by the HTTP transport factory closure where a single
    /// `Arc<Mutex<DaemonClient>>` and `Arc<SharedStateManager>` must be
    /// shared across all per-session `ToolsHandler` instances (those types are
    /// not `Clone`).
    pub fn from_arcs(
        daemon: Arc<Mutex<DaemonClient>>,
        qdrant: Arc<QdrantReadClient>,
        state: Arc<SharedStateManager>,
        session: Arc<Mutex<SessionState>>,
        health_state: SharedHealthState,
    ) -> Self {
        Self::from_arcs_with_config(daemon, qdrant, state, session, health_state, None)
    }

    /// Create a handler from already-Arc-wrapped deps, with config overrides.
    pub fn from_arcs_with_config(
        daemon: Arc<Mutex<DaemonClient>>,
        qdrant: Arc<QdrantReadClient>,
        state: Arc<SharedStateManager>,
        session: Arc<Mutex<SessionState>>,
        health_state: SharedHealthState,
        rules_dup_threshold: Option<f64>,
    ) -> Self {
        Self {
            daemon,
            qdrant,
            state,
            session,
            health_state,
            rules_dup_threshold,
            hb_handle: Arc::new(std::sync::Mutex::new(None)),
        }
    }

    /// Access the heartbeat task handle (used by the transport layer for
    /// cleanup). The slot is populated when `initialize` starts the heartbeat.
    pub fn hb_handle(&self) -> Arc<std::sync::Mutex<Option<AbortHandle>>> {
        Arc::clone(&self.hb_handle)
    }

    /// Replace the heartbeat-handle slot with a shared one.
    ///
    /// The HTTP transport builds a fresh `ToolsHandler` per connection but they
    /// all share one `SessionState`; the heartbeat is started once (idempotency
    /// guard). Sharing one `hb_handle` slot across the per-connection handlers
    /// lets the server abort that single heartbeat at shutdown.
    pub fn with_shared_hb_handle(
        mut self,
        hb_handle: Arc<std::sync::Mutex<Option<AbortHandle>>>,
    ) -> Self {
        self.hb_handle = hb_handle;
        self
    }
}

impl ServerHandler for ToolsHandler {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(SERVER_NAME, server_version_string()))
            .with_instructions(INSTRUCTIONS)
    }

    /// Run the session lifecycle on the MCP `initialize` request.
    ///
    /// Without this override rmcp's default `initialize` only stashes peer info,
    /// so `initialize_session` never runs — leaving `session.project_id`/branch
    /// unset and the project never registered with the daemon (GitHub #84).
    ///
    /// Project detection runs under a short synchronous SQLite lock that is
    /// dropped before the daemon `.await`; the resolved `Option<ProjectInfo>` is
    /// then handed to `initialize_session`. The heartbeat is started inside the
    /// lifecycle closure and its `AbortHandle` stored for cleanup. The lifecycle
    /// is idempotent (guarded by `SessionState::initialized`), so the shared
    /// HTTP `SessionState` is initialized at most once.
    fn initialize(
        &self,
        request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<InitializeResult, ErrorData>> + Send + '_ {
        let state = Arc::clone(&self.state);
        let session = Arc::clone(&self.session);
        let daemon = Arc::clone(&self.daemon);
        let hb_slot = Arc::clone(&self.hb_handle);
        async move {
            // Preserve rmcp's default behaviour: stash the client's peer info.
            if context.peer.peer_info().is_none() {
                context.peer.set_peer_info(request);
            }

            let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

            // Detect the project under a short SQLite lock, dropped before the
            // `.await`-bearing `initialize_session` (never held across an await).
            let detected = {
                let guard = state.lock();
                crate::session::default_detect_fn(&cwd, &guard)
            };

            // Lock order MUST match `call_tool` (daemon then session) to avoid
            // an ABBA deadlock if an `initialize` and a tool call overlap.
            let mut daemon_guard = daemon.lock().await;
            let mut session_guard = session.lock().await;

            let hb_session = Arc::clone(&session);
            let hb_daemon = Arc::clone(&daemon);
            crate::session::initialize_session(
                &mut session_guard,
                &mut *daemon_guard,
                &cwd,
                detected,
                move || {
                    let handle = crate::session::start_heartbeat(hb_session, move |pid| {
                        let d = Arc::clone(&hb_daemon);
                        async move {
                            use crate::session::DaemonOps;
                            DaemonOps::heartbeat(&mut *d.lock().await, &pid).await
                        }
                    });
                    *hb_slot.lock().unwrap() = Some(handle);
                },
            )
            .await;

            // Drop in reverse acquisition order (session, then daemon).
            drop(session_guard);
            drop(daemon_guard);
            Ok(self.get_info())
        }
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
        let health_state = Arc::clone(&self.health_state);

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
                health_state: &health_state,
                rules_dup_threshold: self.rules_dup_threshold,
            };

            Ok(dispatch_tool(&name, &args, &mut ctx).await)
        }
    }
}
