//! Build script for proto compilation and build metadata
//!
//! Compiles workspace_daemon.proto for gRPC client usage only.
//! Captures git commit count as a 4-digit hex build number.

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

    // Capture git commit count as build number
    let build_number = std::process::Command::new("git")
        .args(["rev-list", "--count", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);

    println!("cargo:rustc-env=BUILD_NUMBER={:04X}", build_number);

    // Rerun if git HEAD changes (new commits)
    println!("cargo:rerun-if-changed=../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../.git/refs/heads/");

    Ok(())
}
