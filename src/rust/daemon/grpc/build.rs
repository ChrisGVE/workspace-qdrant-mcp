// Build script for gRPC proto generation

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate Rust code from workspace_daemon.proto
    // This proto file defines 4 services: SystemService, CollectionService, DocumentService, ProjectService
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &["../proto/workspace_daemon.proto"],
            &["../proto"]
        )?;

    // Rerun if proto files change
    println!("cargo:rerun-if-changed=../proto/workspace_daemon.proto");

    Ok(())
}
