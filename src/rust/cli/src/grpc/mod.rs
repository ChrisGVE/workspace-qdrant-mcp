//! gRPC client module
//!
//! Provides connection to memexd daemon via workspace_daemon.proto.

pub mod client;

pub use client::DaemonClient;
