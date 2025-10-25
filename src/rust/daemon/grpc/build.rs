// Build script for gRPC proto generation

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate Rust code from the legacy workspace_daemon.proto
    // This proto file defines 3 services: SystemService, CollectionService, DocumentService
    // TODO: Migrate to ../proto/ingestion.proto when daemon unification is complete
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(
            &["../../../../rust-engine-legacy/proto/workspace_daemon.proto"],
            &["../../../../rust-engine-legacy/proto"]
        )?;

    // Rerun if proto files change
    println!("cargo:rerun-if-changed=../../../../rust-engine-legacy/proto/workspace_daemon.proto");

    Ok(())
}
