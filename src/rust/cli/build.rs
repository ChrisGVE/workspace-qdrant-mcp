//! Build script for proto compilation
//!
//! Compiles workspace_daemon.proto for gRPC client usage only.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile protos from daemon workspace
    // CLI only needs client stubs, not server code
    tonic_build::configure()
        .build_server(false) // Client only
        .compile_protos(
            &["../daemon/proto/workspace_daemon.proto"],
            &["../daemon/proto"],
        )?;

    // Note: Legacy ingestion.proto was removed. All gRPC operations now use
    // workspace_daemon.proto with services: SystemService, CollectionService,
    // DocumentService, EmbeddingService, ProjectService.

    Ok(())
}
