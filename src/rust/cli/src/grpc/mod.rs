//! gRPC client module
//!
//! Provides connection to memexd daemon via workspace_daemon.proto.

pub mod client;

pub use client::DaemonClient;
pub use client::workspace_daemon as proto;
