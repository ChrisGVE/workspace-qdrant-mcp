/// Build script: extract the resolved tree-sitter crate version from Cargo.lock
/// and expose it as TREE_SITTER_VERSION_MAJOR_MINOR (e.g., "0.24") at compile time.
///
/// This ensures YAML config defaults always match the actual dependency version
/// without manual synchronization.
use std::fs;
use std::path::Path;

fn main() {
    // Cargo.lock lives at the workspace root (src/rust/Cargo.lock)
    let lock_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // src/rust
        .map(|p| p.join("Cargo.lock"));

    if let Some(lock_path) = lock_path {
        if let Ok(contents) = fs::read_to_string(&lock_path) {
            if let Some(version) = extract_tree_sitter_version(&contents) {
                let major_minor = extract_major_minor(&version);
                println!(
                    "cargo:rustc-env=TREE_SITTER_VERSION_MAJOR_MINOR={}",
                    major_minor
                );
            } else {
                println!("cargo:warning=Could not extract tree-sitter version from Cargo.lock");
                println!("cargo:rustc-env=TREE_SITTER_VERSION_MAJOR_MINOR=unknown");
            }
            println!("cargo:rerun-if-changed={}", lock_path.display());
        } else {
            println!(
                "cargo:warning=Could not read Cargo.lock at {}",
                lock_path.display()
            );
            println!("cargo:rustc-env=TREE_SITTER_VERSION_MAJOR_MINOR=unknown");
        }
    } else {
        println!("cargo:rustc-env=TREE_SITTER_VERSION_MAJOR_MINOR=unknown");
    }
}

/// Extract major.minor from a semver string: "0.24.7" → "0.24"
fn extract_major_minor(version: &str) -> String {
    let parts: Vec<&str> = version.splitn(3, '.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        version.to_string()
    }
}

/// Parse Cargo.lock TOML to find the tree-sitter package version.
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
            let version = trimmed
                .strip_prefix("version = ")?
                .trim_matches('"')
                .to_string();
            return Some(version);
        }
    }

    None
}
