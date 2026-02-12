//! Build script for build metadata
//!
//! Captures git commit count as a 4-digit hex build number.

fn main() {
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
    println!("cargo:rerun-if-changed=../../../.git/HEAD");
    println!("cargo:rerun-if-changed=../../../.git/refs/heads/");
}
