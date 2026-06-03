//! Thin, dependency-free path-exclusion helpers for CLI filesystem walks
//! (WI-b1).
//!
//! These are deliberately NOT the daemon's full [`ExclusionEngine`] (which
//! stays in `workspace-qdrant-core` and drives ingestion). They are a small
//! hardcoded critical-pattern set sufficient for CLI commands that walk a
//! project tree (e.g. `wqm language warm`/`projects`) and need to skip the
//! obvious build/dependency/VCS directories without pulling the daemon engine
//! into the CLI.
//!
//! For the well-known sample of directories and files these helpers return the
//! same verdict as the daemon engine (see the parity test in
//! `workspace-qdrant-core`); they intentionally do not replicate the engine's
//! full configurable pattern set.
//!
//! [`ExclusionEngine`]: (workspace-qdrant-core)

/// Well-known build / dependency / cache directory names that are always
/// skipped during a CLI walk. Hidden directories (leading `.`) are handled
/// separately by [`should_exclude_directory`].
const CRITICAL_DIRS: &[&str] = &[
    "target",
    "node_modules",
    "__pycache__",
    "dist",
    "build",
    "vendor",
];

/// Critical exact filenames that are always excluded regardless of directory.
const CRITICAL_FILES: &[&str] = &[".DS_Store", "Thumbs.db"];

/// Critical file extensions (including the dot) that are always excluded.
const CRITICAL_EXTENSIONS: &[&str] = &[".pyc", ".pyo"];

/// Whether a directory (given its final component name) should be skipped
/// entirely during a filesystem walk.
///
/// Rules (matching the daemon engine for the common cases):
/// - `.github` is explicitly whitelisted (never skipped).
/// - Any other hidden directory (leading `.`) is skipped.
/// - The well-known build/dependency directories ([`CRITICAL_DIRS`]) are
///   skipped.
pub fn should_exclude_directory(dir_name: &str) -> bool {
    if dir_name == ".github" {
        return false;
    }
    if dir_name.starts_with('.') {
        return true;
    }
    CRITICAL_DIRS.contains(&dir_name)
}

/// Whether a file path should be excluded during a CLI walk.
///
/// A path is excluded when any of the following holds:
/// - a path component is a hidden entry (leading `.`, e.g. `.git/config`);
/// - a path component is one of the critical build/dependency directories;
/// - the final component is a critical filename, or has a critical extension.
pub fn should_exclude_file(file_path: &str) -> bool {
    let normalized = file_path.replace('\\', "/");
    let components: Vec<&str> = normalized.split('/').filter(|c| !c.is_empty()).collect();

    for (idx, component) in components.iter().enumerate() {
        let is_last = idx == components.len() - 1;

        // Hidden component (file or directory) → excluded. `.` / `..` ignored.
        if component.starts_with('.') && *component != "." && *component != ".." {
            return true;
        }

        // Directory component in the critical set → excluded.
        if !is_last && CRITICAL_DIRS.contains(component) {
            return true;
        }

        if is_last {
            if CRITICAL_FILES.contains(component) {
                return true;
            }
            if CRITICAL_EXTENSIONS
                .iter()
                .any(|ext| component.ends_with(ext))
            {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directories_well_known_excluded() {
        for d in [
            "target",
            "node_modules",
            "__pycache__",
            "dist",
            "build",
            "vendor",
        ] {
            assert!(should_exclude_directory(d), "{d} should be excluded");
        }
    }

    #[test]
    fn hidden_directories_excluded_except_github() {
        assert!(should_exclude_directory(".git"));
        assert!(should_exclude_directory(".venv"));
        assert!(should_exclude_directory(".mypy_cache"));
        assert!(!should_exclude_directory(".github"));
    }

    #[test]
    fn normal_directories_kept() {
        for d in ["src", "lib", "tests", "docs", "app"] {
            assert!(!should_exclude_directory(d), "{d} should be kept");
        }
    }

    #[test]
    fn files_in_excluded_locations() {
        assert!(should_exclude_file(".git/config"));
        assert!(should_exclude_file("target/debug/app"));
        assert!(should_exclude_file("project/node_modules/pkg/index.js"));
        assert!(should_exclude_file(".DS_Store"));
        assert!(should_exclude_file("path/to/.DS_Store"));
        assert!(should_exclude_file("module/foo.pyc"));
    }

    #[test]
    fn ordinary_files_kept() {
        assert!(!should_exclude_file("src/main.rs"));
        assert!(!should_exclude_file("lib/util.py"));
        assert!(!should_exclude_file("docs/guide.md"));
    }
}
