//! Shared gRPC client stubs generated from `workspace_daemon.proto`.
//!
//! Both `wqm-cli` and the MCP server depend on this crate instead of each
//! compiling the proto themselves (WI-c1, #82). The generated types live under
//! [`workspace_daemon`], mirroring the proto's `package workspace_daemon`.

/// Generated client stubs and message types from `workspace_daemon.proto`.
pub mod workspace_daemon {
    tonic::include_proto!("workspace_daemon");
}
