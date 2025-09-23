use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .file_descriptor_set_path(out_dir.join("workspace_daemon_descriptor.bin"))
        .compile_protos(
            &["proto/workspace_daemon.proto"],
            &["proto"],
        )?;

    // Also compile ingestion proto if it exists
    if std::path::Path::new("../src/rust/daemon/proto/ingestion.proto").exists() {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .file_descriptor_set_path(out_dir.join("ingestion_descriptor.bin"))
            .compile_protos(
                &["../src/rust/daemon/proto/ingestion.proto"],
                &["../src/rust/daemon/proto"],
            )?;
        println!("cargo:rerun-if-changed=../src/rust/daemon/proto/ingestion.proto");
    }

    // Rerun if proto files change
    println!("cargo:rerun-if-changed=proto/workspace_daemon.proto");

    Ok(())
}