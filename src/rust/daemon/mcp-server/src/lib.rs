//! workspace-qdrant MCP server — Rust implementation.
//!
//! This crate provides a drop-in replacement for the TypeScript MCP server.
//! It exposes the same 6 MCP tools (store, search, rules, retrieve, grep, list)
//! over stdio and streamable-HTTP transports.

pub mod config;
pub mod instructions;
pub mod server;
pub mod server_types;

/// Generated gRPC client stubs from workspace_daemon.proto.
///
/// Mirrors the pattern in `wqm-cli/src/grpc/client.rs`.  Build-time codegen
/// is performed by build.rs using `tonic_build::configure().build_server(false)`.
pub mod proto {
    tonic::include_proto!("workspace_daemon");
}

#[cfg(test)]
mod tests {
    #[test]
    fn crate_builds() {
        // Smoke test: verifies the test harness is wired up correctly.
        // More substantive tests live in each sub-module.
        assert!(true);
    }
}
