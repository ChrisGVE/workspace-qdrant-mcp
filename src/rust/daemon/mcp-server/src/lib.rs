//! workspace-qdrant MCP server — Rust implementation.
//!
//! This crate provides a drop-in replacement for the TypeScript MCP server.
//! It exposes the same 7 MCP tools (search, retrieve, rules, store, grep, list,
//! embedding) over stdio and streamable-HTTP transports.

pub mod canonicalize;
pub mod config;
pub mod grpc;
pub mod instructions;
pub mod observability;
pub mod qdrant;
pub mod server;
pub mod server_types;
pub mod session;
pub mod sqlite;
pub mod tools;
pub mod transport;

/// Generated gRPC client stubs, now sourced from the shared `wqm-proto` crate
/// (WI-c1, #82) instead of compiled in this crate's build.rs. The `proto` alias
/// keeps every existing `proto::` call site stable.
pub use wqm_proto::workspace_daemon as proto;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        // Smoke test: verifies the test harness is wired up correctly.
        // More substantive tests live in each sub-module.
        assert!(true);
    }
}
