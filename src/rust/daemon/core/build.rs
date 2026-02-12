/// Build script: extract the resolved tree-sitter crate version from Cargo.lock
/// and expose it as TREE_SITTER_VERSION (e.g., "0.24.7") at compile time.
///
/// This eliminates the hardcoded version string in version_checker.rs.

use std::fs;
use std::path::Path;

fn main() {
    // Cargo.lock lives at the workspace root (src/rust/Cargo.lock)
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // src/rust/daemon
        .and_then(|p| p.parent()) // src/rust
        .map(|p| p.join("Cargo.lock"));

    if let Some(lock_path) = lock_path {
        if let Ok(contents) = fs::read_to_string(&lock_path) {
            if let Some(version) = extract_tree_sitter_version(&contents) {
                println!("cargo:rustc-env=TREE_SITTER_VERSION={}", version);
            } else {
                // Fallback: if parsing fails, don't break the build
                println!("cargo:warning=Could not extract tree-sitter version from Cargo.lock");
                println!("cargo:rustc-env=TREE_SITTER_VERSION=unknown");
            }
            // Re-run if Cargo.lock changes (dependency update)
            println!("cargo:rerun-if-changed={}", lock_path.display());
        } else {
            println!("cargo:warning=Could not read Cargo.lock at {}", lock_path.display());
            println!("cargo:rustc-env=TREE_SITTER_VERSION=unknown");
        }
    } else {
        println!("cargo:rustc-env=TREE_SITTER_VERSION=unknown");
    }
}

/// Parse Cargo.lock TOML to find the tree-sitter package version.
///
/// Cargo.lock format:
/// ```toml
/// [[package]]
/// name = "tree-sitter"
/// version = "0.24.7"
/// ```
fn extract_tree_sitter_version(lock_contents: &str) -> Option<String> {
    let mut in_tree_sitter_block = false;

    for line in lock_contents.lines() {
        let trimmed = line.trim();

        if trimmed == "[[package]]" {
            in_tree_sitter_block = false;
        }

        if trimmed == r#"name = "tree-sitter""# {
            in_tree_sitter_block = true;
            continue;
        }

        if in_tree_sitter_block && trimmed.starts_with("version = ") {
            // Extract version from: version = "0.24.7"
            let version = trimmed
                .strip_prefix("version = ")?
                .trim_matches('"')
                .to_string();
            return Some(version);
        }
    }

    None
}
