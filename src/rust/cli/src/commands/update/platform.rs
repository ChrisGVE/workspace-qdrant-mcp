//! Platform detection for update command
//!
//! Provides target triple, binary filenames, and install location resolution.

use anyhow::Result;
use std::path::PathBuf;

/// Platform target triple
pub fn get_target_triple() -> &'static str {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    {
        "aarch64-unknown-linux-gnu"
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        "x86_64-pc-windows-msvc"
    }
    #[cfg(all(target_os = "windows", target_arch = "aarch64"))]
    {
        "aarch64-pc-windows-msvc"
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "linux", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "aarch64"),
    )))]
    {
        "unknown"
    }
}

/// Get the binary filename for the current platform
pub fn get_binary_filename() -> String {
    let target = get_target_triple();
    #[cfg(target_os = "windows")]
    {
        format!("memexd-{}.exe", target)
    }
    #[cfg(not(target_os = "windows"))]
    {
        format!("memexd-{}", target)
    }
}

/// Get checksum filename for the current platform
pub fn get_checksum_filename() -> String {
    format!("{}.sha256", get_binary_filename())
}

/// Find the installation location of the daemon binary
pub fn find_install_location() -> Result<PathBuf> {
    // First, try to find existing installation
    if let Ok(path) = which::which("memexd") {
        return Ok(path);
    }

    // Default locations by platform
    #[cfg(target_os = "macos")]
    let default = PathBuf::from("/usr/local/bin/memexd");

    #[cfg(target_os = "linux")]
    let default = dirs::home_dir()
        .map(|h| h.join(".local/bin/memexd"))
        .unwrap_or_else(|| PathBuf::from("/usr/local/bin/memexd"));

    #[cfg(target_os = "windows")]
    let default = dirs::data_local_dir()
        .map(|d| d.join("workspace-qdrant").join("memexd.exe"))
        .unwrap_or_else(|| PathBuf::from("C:\\Program Files\\workspace-qdrant\\memexd.exe"));

    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    let default = PathBuf::from("./memexd");

    // Ensure parent directory exists
    if let Some(parent) = default.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    Ok(default)
}
