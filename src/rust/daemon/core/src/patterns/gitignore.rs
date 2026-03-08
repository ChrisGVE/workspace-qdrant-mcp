//! Gate 0 per-project ignore file matcher (.gitignore + .wqmignore).
//!
//! Reads `.gitignore` and/or `.wqmignore` files from a scan directory and
//! produces a matcher that can be queried for individual paths.  Union
//! semantics apply: a path excluded by *either* file is excluded.
//!
//! Both files use standard gitignore syntax (powered by the `ignore` crate).
//! `.wqmignore` lets users add wqm-specific exclusions without touching the
//! project's `.gitignore`.

use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

/// Per-directory matcher for `.gitignore` and `.wqmignore` (Gate 0).
///
/// Build once per `scan_directory_single_level` call via [`ProjectIgnoreMatcher::for_dir`],
/// then test every entry with [`ProjectIgnoreMatcher::is_ignored`] before
/// applying any other exclusion rules.
pub struct ProjectIgnoreMatcher {
    gitignore: Gitignore,
}

impl ProjectIgnoreMatcher {
    /// Build a matcher from the `.gitignore` and `.wqmignore` files present in `dir`.
    ///
    /// Returns `None` when neither file exists in `dir` so callers can skip
    /// the gate entirely with zero overhead.
    pub fn for_dir(dir: &Path) -> Option<Self> {
        let gitignore_path = dir.join(".gitignore");
        let wqmignore_path = dir.join(".wqmignore");

        if !gitignore_path.exists() && !wqmignore_path.exists() {
            return None;
        }

        // Root for the builder is `dir`; patterns are interpreted relative to
        // the directory that contains the ignore file (same as git behaviour).
        let mut builder = GitignoreBuilder::new(dir);
        if gitignore_path.exists() {
            let _ = builder.add(&gitignore_path);
        }
        if wqmignore_path.exists() {
            let _ = builder.add(&wqmignore_path);
        }

        let gitignore = builder.build().ok()?;
        Some(Self { gitignore })
    }

    /// Returns `true` if `path` should be excluded by any pattern in `.gitignore`
    /// or `.wqmignore`.
    ///
    /// `is_dir` must be `true` when `path` refers to a directory entry.
    pub fn is_ignored(&self, path: &Path, is_dir: bool) -> bool {
        self.gitignore.matched(path, is_dir).is_ignore()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn tmp() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    // ── no ignore files ────────────────────────────────────────────────────────

    #[test]
    fn no_ignore_files_returns_none() {
        let dir = tmp();
        assert!(ProjectIgnoreMatcher::for_dir(dir.path()).is_none());
    }

    // ── .gitignore only ────────────────────────────────────────────────────────

    #[test]
    fn gitignore_excludes_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(m.is_ignored(&dir.path().join("datasets"), true));
    }

    #[test]
    fn gitignore_does_not_exclude_non_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    #[test]
    fn gitignore_excludes_file_by_extension() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(m.is_ignored(&dir.path().join("debug.log"), false));
        assert!(!m.is_ignored(&dir.path().join("main.rs"), false));
    }

    // ── .wqmignore only ───────────────────────────────────────────────────────

    #[test]
    fn wqmignore_only_excludes_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".wqmignore"), "large_data/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(m.is_ignored(&dir.path().join("large_data"), true));
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    // ── union semantics ────────────────────────────────────────────────────────

    #[test]
    fn union_semantics_gitignore_pattern_excludes() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "git_only/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "wqm_only/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(m.is_ignored(&dir.path().join("git_only"), true));
        assert!(m.is_ignored(&dir.path().join("wqm_only"), true));
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    #[test]
    fn union_semantics_both_files_can_match_independently() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "build/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        // .gitignore hit
        assert!(m.is_ignored(&dir.path().join("build"), true));
        // .wqmignore hit
        assert!(m.is_ignored(&dir.path().join("datasets"), true));
        // neither
        assert!(!m.is_ignored(&dir.path().join("docs"), true));
    }

    // ── no false positives on non-ignored files ───────────────────────────────

    #[test]
    fn non_ignored_file_is_not_excluded() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "*.bin\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(!m.is_ignored(&dir.path().join("README.md"), false));
        assert!(!m.is_ignored(&dir.path().join("main.rs"), false));
    }
}
