//! Shared gRPC client for the memexd daemon.
//!
//! Exposes a single [`DaemonClient`] wrapper (dial / retry / timeout) plus typed
//! per-service RPC wrappers, shared by the wqm CLI and the MCP server (WI-d1/d2,
//! #82). Depends only on `wqm-proto` + `wqm-common` — never on
//! `workspace-qdrant-core`.
//!
//! Sub-modules:
//! - [`client`] — [`DaemonClient`] struct, channel construction, `call` dispatch.
//! - [`connection`] — daemon-address resolution (`WQM_DAEMON_ADDR` > profile > default).
//! - [`retry`] — `call_with_retry` with exponential backoff.
//! - [`timeouts`] — per-method timeout resolution (5 s / 10 s).
//! - `*_methods` — typed RPC wrappers per service (`impl DaemonClient`).

pub mod client;
pub mod connection;
pub mod document_methods;
pub mod embedding_methods;
pub mod graph_methods;
pub mod language_methods;
pub mod project_methods;
pub mod retry;
pub mod search_event_methods;
pub mod search_methods;
pub mod system_methods;
pub mod timeouts;
pub mod write_methods;

pub use client::{ClientError, DaemonClient};
pub use connection::{default_daemon_address, resolve_daemon_address};
