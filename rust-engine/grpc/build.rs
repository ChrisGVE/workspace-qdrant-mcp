// Build script for gRPC proto generation

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate Rust code from protobuf definitions
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile(&["../proto/ingestion.proto"], &["../proto"])?;

    // Rerun if proto files change
    println!("cargo:rerun-if-changed=../proto/ingestion.proto");

    Ok(())
}
