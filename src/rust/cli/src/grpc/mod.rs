//! gRPC client module.
//!
//! The [`DaemonClient`] implementation now lives in the shared `wqm-client`
//! crate (WI-d2, #82). This module re-exports it plus the proto message types,
//! and provides the CLI-side async connection helpers ([`connect_default`],
//! [`ensure_daemon_available`]) that add an eager liveness probe over the
//! lazily-connecting shared client.

#![allow(unused_imports)]

mod connect;

pub use wqm_client::workspace_daemon;
/// Backwards-compatible alias used throughout the CLI (`crate::grpc::proto`).
pub use wqm_client::workspace_daemon as proto;
pub use wqm_client::{ClientError, DaemonClient};

pub use connect::{connect_default, ensure_daemon_available};

/// Compatibility submodule so existing `crate::grpc::client::{DaemonClient,
/// workspace_daemon}` paths keep resolving after the client moved to wqm-client.
pub mod client {
    pub use wqm_client::workspace_daemon;
    pub use wqm_client::DaemonClient;
}
