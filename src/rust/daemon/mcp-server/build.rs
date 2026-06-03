//! Build script for build metadata.
//!
//! Proto compilation moved to the shared `wqm-proto` crate (WI-c1, #82); this
//! script only captures the git commit count as a 4-digit hex build number.

fn main() -> Result<(), Box<dyn std::error::Error>> {
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
