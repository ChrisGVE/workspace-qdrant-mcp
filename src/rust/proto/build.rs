//! Build script for the shared `wqm-proto` crate.
//!
//! Compiles `workspace_daemon.proto` into gRPC **client** stubs (no server
//! code) — the single proto-compilation site for the CLI and MCP-server
//! clients. The daemon's `workspace-qdrant-grpc` crate compiles its own
//! server+client stubs separately.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(false)
        .build_client(true)
        .compile_protos(
            &["../daemon/proto/workspace_daemon.proto"],
            &["../daemon/proto"],
        )?;

    println!("cargo:rerun-if-changed=../daemon/proto/workspace_daemon.proto");

    Ok(())
}
