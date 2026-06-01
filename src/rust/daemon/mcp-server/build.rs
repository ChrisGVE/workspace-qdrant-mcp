//! Build script for proto compilation and build metadata
//!
//! Compiles workspace_daemon.proto for gRPC client usage only (build_server=false).
//! Captures git commit count as a 4-digit hex build number.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Proto is a sibling directory: src/rust/daemon/proto/
    // This crate lives at: src/rust/daemon/mcp-server/
    // So the relative path from here is: ../proto/
    tonic_build::configure()
        .build_server(false) // Client only — no generated server stubs
        .compile_protos(&["../proto/workspace_daemon.proto"], &["../proto"])?;

    // Capture git commit count as build number (4-digit hex)
    let build_number = std::process::Command::new("git")
        .args(["rev-list", "--count", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u32>().ok())
        .unwrap_or(0);

    println!("cargo:rustc-env=BUILD_NUMBER={:04X}", build_number);

    // Re-run if git HEAD or branch refs change
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/refs/heads/");

    Ok(())
}
