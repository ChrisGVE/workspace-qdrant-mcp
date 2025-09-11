use std::io::Result;

fn main() -> Result<()> {
    // Generate gRPC code from proto files
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir("src/grpc/generated")
        .compile(
            &["../proto/ingestion.proto"],
            &["../proto/"],
        )?;

    // Tell Cargo to rerun this build script if the proto file changes
    println!("cargo:rerun-if-changed=../proto/ingestion.proto");
    
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