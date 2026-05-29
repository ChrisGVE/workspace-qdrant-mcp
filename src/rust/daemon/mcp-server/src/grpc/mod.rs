//! gRPC client for the memexd daemon.
//!
//! Exposes a thin [`DaemonClient`] wrapper that adds retry/backoff and
//! client-side timeout semantics matching the TypeScript MCP server.
//!
//! Sub-modules:
//! - [`client`] — [`DaemonClient`] struct and channel construction.
//! - [`retry`]   — `call_with_retry` with exponential backoff.
//! - [`timeouts`] — per-method timeout resolution (5 s / 10 s).
//! - [`system_methods`] — `health`, `get_status`, `get_embedding_provider_status`.
//! - [`project_methods`] — `register_project`, `deprioritize_project`, `heartbeat`, `resolve_search_scope`.
//! - [`embedding_methods`] — `embed_text`, `generate_sparse_vector`.
//! - [`search_methods`] — `text_search`, `count_matches`.
//! - [`graph_methods`] — `query_related`.

pub mod client;
pub mod embedding_methods;
pub mod graph_methods;
pub mod project_methods;
pub mod retry;
pub mod search_methods;
pub mod system_methods;
pub mod timeouts;

pub use client::{ClientError, DaemonClient};
