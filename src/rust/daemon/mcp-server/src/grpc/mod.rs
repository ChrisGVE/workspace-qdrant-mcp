//! gRPC client for the memexd daemon.
//!
//! Exposes a thin [`DaemonClient`] wrapper that adds retry/backoff and
//! client-side timeout semantics matching the TypeScript MCP server.
//!
//! Sub-modules:
//! - [`client`] — [`DaemonClient`] struct and channel construction.
//! - [`retry`]   — `call_with_retry` with exponential backoff.
//! - [`timeouts`] — per-method timeout resolution (5 s / 10 s).

pub mod client;
pub mod retry;
pub mod timeouts;

pub use client::{ClientError, DaemonClient};
