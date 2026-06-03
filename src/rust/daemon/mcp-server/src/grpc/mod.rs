//! gRPC client for the memexd daemon.
//!
//! The implementation now lives in the shared [`wqm_client`] crate (WI-d1/d2,
//! #82) — this module is a thin re-export so existing `crate::grpc::…` paths
//! keep resolving. The typed per-service RPC wrappers, retry/backoff, and
//! client-side timeout semantics are all provided by `wqm-client`.
//!
//! Local trait impls that bridge `DaemonClient` to MCP-specific traits
//! (e.g. `EmbedDaemon`, `StoreDaemon`, `RulesDaemon`, `DaemonOps`) live in the
//! `tools`/`session` modules and remain valid — they implement *local* traits
//! for the (now foreign) `DaemonClient` type.

/// Re-export of the shared client module so `crate::grpc::client::DaemonClient`
/// continues to resolve.
pub use wqm_client::grpc::client;

pub use wqm_client::{ClientError, DaemonClient};
