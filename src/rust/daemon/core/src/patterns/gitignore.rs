//! Gate 0 per-project ignore file matcher (.gitignore + .wqmignore).
//!
//! Reads `.gitignore` and/or `.wqmignore` files from a scan directory and
//! produces a matcher that can be queried for individual paths.
//!
//! Both files use standard gitignore syntax (powered by the `ignore` crate).
//! `.wqmignore` lets users add wqm-specific exclusions without touching the
//! project's `.gitignore`.
//!
//! ## .wqmignore negation syntax
//!
//! Lines starting with `- ` (dash space) in `.wqmignore` **re-include** paths
//! that `.gitignore` excludes. This allows indexing git-ignored files (e.g.,
//! generated docs, build artifacts) without modifying `.gitignore`.
//!
//! Resolution order:
//! 1. If `.wqmignore` re-includes the path → **not ignored** (overrides gitignore)
//! 2. If `.gitignore` or `.wqmignore` exclusions match → **ignored**
//! 3. Otherwise → **not ignored**

use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};

/// Per-directory matcher for `.gitignore` and `.wqmignore` (Gate 0).
///
/// Build once per `scan_directory_single_level` call via [`ProjectIgnoreMatcher::for_dir`],
/// then test every entry with [`ProjectIgnoreMatcher::is_ignored`] before
/// applying any other exclusion rules.
pub struct ProjectIgnoreMatcher {
    /// Combined .gitignore + .wqmignore exclusion patterns
    exclusions: Gitignore,
    /// .wqmignore re-inclusion patterns (lines starting with `- `)
    reinclusions: Option<Gitignore>,
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
        let mut exclusion_builder = GitignoreBuilder::new(dir);
        if gitignore_path.exists() {
            let _ = exclusion_builder.add(&gitignore_path);
        }

        // Parse .wqmignore: split into exclusions and re-inclusions
        let reinclusions = if wqmignore_path.exists() {
            parse_wqmignore(dir, &wqmignore_path, &mut exclusion_builder)
        } else {
            None
        };

        let exclusions = exclusion_builder.build().ok()?;
        Some(Self {
            exclusions,
            reinclusions,
        })
    }

    /// Returns `true` if `path` should be excluded.
    ///
    /// Resolution: re-inclusion wins over exclusion (allows overriding gitignore).
    /// `is_dir` must be `true` when `path` refers to a directory entry.
    pub fn is_ignored(&self, path: &Path, is_dir: bool) -> bool {
        // Re-inclusions override: if path matches a `- pattern`, it's NOT ignored
        if let Some(ref reinc) = self.reinclusions {
            if reinc.matched(path, is_dir).is_ignore() {
                return false;
            }
        }

        self.exclusions.matched(path, is_dir).is_ignore()
    }
}

/// Parse `.wqmignore`, splitting lines into exclusions (added to `builder`)
/// and re-inclusions (returned as a separate matcher).
///
/// Re-inclusion lines start with `- ` (dash space). The pattern after the
/// prefix is treated as a standard gitignore pattern.
fn parse_wqmignore(
    dir: &Path,
    wqmignore_path: &Path,
    exclusion_builder: &mut GitignoreBuilder,
) -> Option<Gitignore> {
    let content = std::fs::read_to_string(wqmignore_path).ok()?;
    let mut reinc_builder = GitignoreBuilder::new(dir);
    let mut has_reinclusions = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if let Some(pattern) = trimmed.strip_prefix("- ") {
            let pattern = pattern.trim();
            if !pattern.is_empty() {
                let _ = reinc_builder.add_line(Some(wqmignore_path.to_path_buf()), pattern);
                has_reinclusions = true;
            }
        } else {
            let _ = exclusion_builder.add_line(Some(wqmignore_path.to_path_buf()), trimmed);
        }
    }

    if has_reinclusions {
        reinc_builder.build().ok()
    } else {
        None
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

    // ── .wqmignore negation syntax ─────────────────────────────────────────

    #[test]
    fn wqmignore_reinclusion_overrides_gitignore() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        // dist/ is in .gitignore but re-included by .wqmignore
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
    }

    #[test]
    fn wqmignore_reinclusion_does_not_affect_other_exclusions() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\nnode_modules/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        // dist/ re-included
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
        // node_modules/ still excluded by .gitignore
        assert!(m.is_ignored(&dir.path().join("node_modules"), true));
    }

    #[test]
    fn wqmignore_mixed_exclusion_and_reinclusion() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "build/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "tmp/\n- build/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        // build/ is in .gitignore but re-included by .wqmignore
        assert!(!m.is_ignored(&dir.path().join("build"), true));
        // tmp/ is excluded by .wqmignore
        assert!(m.is_ignored(&dir.path().join("tmp"), true));
        // src/ is not excluded
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    #[test]
    fn wqmignore_reinclusion_with_glob_pattern() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "*.generated.js\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- *.generated.js\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(!m.is_ignored(&dir.path().join("api.generated.js"), false));
    }

    #[test]
    fn wqmignore_comments_and_blank_lines_ignored() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::write(
            dir.path().join(".wqmignore"),
            "# Re-include dist for indexing\n\n- dist/\n\n# Extra exclusion\ntmp/\n",
        )
        .unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
        assert!(m.is_ignored(&dir.path().join("tmp"), true));
    }

    #[test]
    fn wqmignore_no_reinclusions_has_no_reinclusion_matcher() {
        let dir = tmp();
        fs::write(dir.path().join(".wqmignore"), "data/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(m.reinclusions.is_none());
        assert!(m.is_ignored(&dir.path().join("data"), true));
    }

    #[test]
    fn wqmignore_only_reinclusions_no_exclusions() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "vendor/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- vendor/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path()).unwrap();
        assert!(!m.is_ignored(&dir.path().join("vendor"), true));
    }
}
