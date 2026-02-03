use std::io::Result;

fn main() -> Result<()> {
    // Note: Legacy ingestion.proto was removed. All gRPC operations now use
    // workspace_daemon.proto with services: SystemService, CollectionService,
    // DocumentService, EmbeddingService, ProjectService.
    //
    // Proto compilation is handled by the grpc crate's build.rs, not here.

    // Set build metadata
    println!("cargo:rustc-env=BUILD_TIMESTAMP={}", chrono::Utc::now().timestamp());
    println!("cargo:rustc-env=GIT_HASH={}", get_git_hash().unwrap_or_else(|| "unknown".to_string()));

    Ok(())
}

fn get_git_hash() -> Option<String> {
    use std::process::Command;
    
    let output = Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}