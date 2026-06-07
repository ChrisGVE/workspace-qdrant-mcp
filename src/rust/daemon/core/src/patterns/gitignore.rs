//! Gate 0 per-project ignore file matcher (.gitignore + .wqmignore).
//!
//! Reads `.gitignore` and/or `.wqmignore` files from a scan directory and
//! produces a matcher that can be queried for individual paths.
//!
//! Both files use standard gitignore syntax (powered by the `ignore` crate).
//! `.wqmignore` lets users add wqm-specific exclusions without touching the
//! project's `.gitignore`.
//!
//! ## .wqmignore negation (re-inclusion) syntax
//!
//! `.wqmignore` supports two equivalent syntaxes for re-including paths that
//! `.gitignore` excludes:
//!
//! - **Canonical**: `!pattern` — standard gitignore negation syntax
//! - **Legacy alias**: `- pattern` (dash space) — accepted for backward compatibility
//!
//! Both syntaxes are functionally identical: they cause the daemon to index a
//! path even when `.gitignore` excludes it. Use `!pattern` in new `.wqmignore`
//! files; `- pattern` continues to work for existing files.
//!
//! Note: re-inclusions are applied with a separate high-priority matcher, which
//! means they can override directory-level exclusions — something standard
//! gitignore `!` cannot do on its own.
//!
//! ## Priority rules
//!
//! `.wqmignore` always takes precedence over `.gitignore`:
//! - Both ignore → ignored
//! - Both re-include → not ignored
//! - `.gitignore` ignores, `.wqmignore` re-includes → **not ignored**
//! - `.gitignore` re-includes, `.wqmignore` ignores → **ignored**
//!
//! Resolution order:
//! 1. If `.wqmignore` re-includes the path → **not ignored** (overrides gitignore)
//! 2. If `.gitignore` or `.wqmignore` exclusions match → **ignored**
//! 3. Otherwise → **not ignored**

use std::path::Path;

use ignore::gitignore::{Gitignore, GitignoreBuilder};
use tracing::warn;

/// Per-directory matcher for `.gitignore` and `.wqmignore` (Gate 0).
///
/// Build once per `scan_directory_single_level` call via [`ProjectIgnoreMatcher::for_dir`],
/// then test every entry with [`ProjectIgnoreMatcher::is_ignored`] before
/// applying any other exclusion rules.
pub struct ProjectIgnoreMatcher {
    /// Root the matcher was built against (project root, or the scanned dir
    /// in legacy dir-only mode). Paths under this root use ancestor-aware
    /// matching so descendants of ignored directories are also ignored.
    root: std::path::PathBuf,
    /// Combined .gitignore + .wqmignore exclusion patterns
    exclusions: Gitignore,
    /// .wqmignore re-inclusion patterns (lines starting with `- `)
    reinclusions: Option<Gitignore>,
}

impl ProjectIgnoreMatcher {
    /// Build a matcher from `.gitignore` and `.wqmignore` files.
    ///
    /// When `project_root` is `Some`, walks from `project_root` down to `dir`,
    /// accumulating ignore rules from each ancestor directory. This ensures a
    /// subdirectory scan respects patterns defined in parent directories (fixes
    /// issue #49).
    ///
    /// When `project_root` is `None`, only checks `dir` itself (legacy behaviour).
    ///
    /// Returns `None` when no ignore files exist in the search path.
    pub fn for_dir(dir: &Path, project_root: Option<&Path>) -> Option<Self> {
        let root = project_root.unwrap_or(dir);

        // Collect ancestor dirs from root down to dir (inclusive).
        let ancestors = collect_ancestor_chain(root, dir);

        let mut exclusion_builder = GitignoreBuilder::new(root);
        let mut reinc_builder = GitignoreBuilder::new(root);
        let mut found_any = false;
        let mut has_reinclusions = false;

        for ancestor in &ancestors {
            let gitignore_path = ancestor.join(".gitignore");
            let wqmignore_path = ancestor.join(".wqmignore");

            if gitignore_path.exists() {
                if let Some(e) = exclusion_builder.add(&gitignore_path) {
                    warn!("Error reading {}: {}", gitignore_path.display(), e);
                }
                found_any = true;
            }

            if wqmignore_path.exists() {
                if let Some(reinc) = parse_wqmignore_into(
                    ancestor,
                    &wqmignore_path,
                    &mut exclusion_builder,
                    &mut reinc_builder,
                ) {
                    has_reinclusions = has_reinclusions || reinc;
                }
                found_any = true;
            }
        }

        if !found_any {
            return None;
        }

        let exclusions = exclusion_builder.build().ok()?;
        let reinclusions = if has_reinclusions {
            reinc_builder.build().ok()
        } else {
            None
        };

        Some(Self {
            root: root.to_path_buf(),
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
            if self.matched(reinc, path, is_dir) {
                return false;
            }
        }

        self.matched(&self.exclusions, path, is_dir)
    }

    /// Match `path` against `matcher`, treating descendants of matched
    /// directories as matched too (#105).
    ///
    /// `Gitignore::matched` tests only the path itself: pattern `foo/` matches
    /// `foo` but NOT `foo/bar`. Real gitignore handles this by pruning the
    /// walk at `foo`, but queued Folder/Scan items target arbitrary depths —
    /// a scan of a directory INSIDE an ignored tree must still see its
    /// ancestors as ignored, or already-enqueued scans keep re-spawning
    /// children (session-env scan storm, #103).
    ///
    /// `matched_path_or_any_parents` panics for paths outside the matcher
    /// root, so fall back to direct matching in that case (legacy dir-only
    /// mode where `dir` is not under `project_root`).
    fn matched(&self, matcher: &Gitignore, path: &Path, is_dir: bool) -> bool {
        if path.starts_with(&self.root) {
            matcher
                .matched_path_or_any_parents(path, is_dir)
                .is_ignore()
        } else {
            matcher.matched(path, is_dir).is_ignore()
        }
    }
}

/// Collect directory chain from `root` down to `dir` (inclusive).
///
/// If `dir` is not a descendant of `root`, returns just `[dir]`.
fn collect_ancestor_chain(root: &Path, dir: &Path) -> Vec<std::path::PathBuf> {
    // Spec §16 §3.1 rule 7: no fs canonicalize. Use the paths as-is —
    // both come from a single project walk and share their representation.
    let root_canon = root.to_path_buf();
    let dir_canon = dir.to_path_buf();

    if let Ok(suffix) = dir_canon.strip_prefix(&root_canon) {
        let mut chain = vec![root_canon.clone()];
        let mut current = root_canon;
        for component in suffix.components() {
            current = current.join(component);
            chain.push(current.clone());
        }
        chain
    } else {
        // dir is not under root — fall back to dir only
        vec![dir.to_path_buf()]
    }
}

/// Parse `.wqmignore`, adding exclusions to `exclusion_builder` and
/// re-inclusions to `reinc_builder`. Returns `Some(true)` if re-inclusions
/// were found, `Some(false)` if only exclusions, or `None` on read error.
fn parse_wqmignore_into(
    _dir: &Path,
    wqmignore_path: &Path,
    exclusion_builder: &mut GitignoreBuilder,
    reinc_builder: &mut GitignoreBuilder,
) -> Option<bool> {
    let content = std::fs::read_to_string(wqmignore_path).ok()?;
    let mut has_reinclusions = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Canonical `!pattern` syntax or legacy `- pattern` alias — both
        // indicate a re-inclusion that overrides .gitignore exclusions.
        let reinclusion_pattern = if let Some(p) = trimmed.strip_prefix("- ") {
            Some(p.trim())
        } else if let Some(p) = trimmed.strip_prefix('!') {
            Some(p.trim())
        } else {
            None
        };

        if let Some(pattern) = reinclusion_pattern {
            if !pattern.is_empty() {
                if let Err(e) = reinc_builder.add_line(Some(wqmignore_path.to_path_buf()), pattern)
                {
                    warn!(
                        "Malformed re-inclusion pattern '{}' in {}: {}",
                        pattern,
                        wqmignore_path.display(),
                        e
                    );
                }
                has_reinclusions = true;
            }
        } else if let Err(e) =
            exclusion_builder.add_line(Some(wqmignore_path.to_path_buf()), trimmed)
        {
            warn!(
                "Malformed exclusion pattern '{}' in {}: {}",
                trimmed,
                wqmignore_path.display(),
                e
            );
        }
    }

    Some(has_reinclusions)
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
        assert!(ProjectIgnoreMatcher::for_dir(dir.path(), None).is_none());
    }

    // ── .gitignore only ────────────────────────────────────────────────────────

    #[test]
    fn gitignore_excludes_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(m.is_ignored(&dir.path().join("datasets"), true));
    }

    #[test]
    fn gitignore_does_not_exclude_non_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    #[test]
    fn gitignore_excludes_file_by_extension() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "*.log\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(m.is_ignored(&dir.path().join("debug.log"), false));
        assert!(!m.is_ignored(&dir.path().join("main.rs"), false));
    }

    // ── .wqmignore only ───────────────────────────────────────────────────────

    #[test]
    fn wqmignore_only_excludes_matching_directory() {
        let dir = tmp();
        fs::write(dir.path().join(".wqmignore"), "large_data/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(m.is_ignored(&dir.path().join("large_data"), true));
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    // ── union semantics ────────────────────────────────────────────────────────

    #[test]
    fn union_semantics_gitignore_pattern_excludes() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "git_only/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "wqm_only/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(m.is_ignored(&dir.path().join("git_only"), true));
        assert!(m.is_ignored(&dir.path().join("wqm_only"), true));
        assert!(!m.is_ignored(&dir.path().join("src"), true));
    }

    #[test]
    fn union_semantics_both_files_can_match_independently() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "build/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "datasets/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
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
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("README.md"), false));
        assert!(!m.is_ignored(&dir.path().join("main.rs"), false));
    }

    // ── .wqmignore negation syntax ─────────────────────────────────────────

    #[test]
    fn wqmignore_reinclusion_overrides_gitignore() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        // dist/ is in .gitignore but re-included by .wqmignore
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
    }

    #[test]
    fn wqmignore_reinclusion_does_not_affect_other_exclusions() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\nnode_modules/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
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
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
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
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
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
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
        assert!(m.is_ignored(&dir.path().join("tmp"), true));
    }

    // ── canonical !pattern syntax ──────────────────────────────────────────

    #[test]
    fn wqmignore_exclamation_reinclusion_overrides_gitignore() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "!dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
    }

    #[test]
    fn wqmignore_exclamation_with_glob_pattern() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "*.generated.js\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "!*.generated.js\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("api.generated.js"), false));
    }

    #[test]
    fn wqmignore_mixed_legacy_and_canonical_syntax() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "build/\nvendor/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- build/\n!vendor/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        // Both legacy `- build/` and canonical `!vendor/` re-include
        assert!(!m.is_ignored(&dir.path().join("build"), true));
        assert!(!m.is_ignored(&dir.path().join("vendor"), true));
    }

    #[test]
    fn wqmignore_exclamation_does_not_affect_other_exclusions() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "dist/\nnode_modules/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "!dist/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        // dist/ re-included via canonical syntax
        assert!(!m.is_ignored(&dir.path().join("dist"), true));
        // node_modules/ still excluded by .gitignore
        assert!(m.is_ignored(&dir.path().join("node_modules"), true));
    }

    #[test]
    fn wqmignore_no_reinclusions_has_no_reinclusion_matcher() {
        let dir = tmp();
        fs::write(dir.path().join(".wqmignore"), "data/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(m.reinclusions.is_none());
        assert!(m.is_ignored(&dir.path().join("data"), true));
    }

    #[test]
    fn wqmignore_only_reinclusions_no_exclusions() {
        let dir = tmp();
        fs::write(dir.path().join(".gitignore"), "vendor/\n").unwrap();
        fs::write(dir.path().join(".wqmignore"), "- vendor/\n").unwrap();
        let m = ProjectIgnoreMatcher::for_dir(dir.path(), None).unwrap();
        assert!(!m.is_ignored(&dir.path().join("vendor"), true));
    }

    // ── parent cascade (project_root) ─────────────────────────────────────

    #[test]
    fn parent_cascade_gitignore_inherited_by_subdir() {
        let root = tmp();
        // project root has .gitignore excluding dist/
        fs::write(root.path().join(".gitignore"), "dist/\n").unwrap();
        // Create subdir/deep/ with no ignore files
        let deep = root.path().join("subdir").join("deep");
        fs::create_dir_all(&deep).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(&deep, Some(root.path())).unwrap();
        // dist/ pattern from root .gitignore should apply in deep subdir
        assert!(m.is_ignored(&deep.join("dist"), true));
        // unmatched paths still pass
        assert!(!m.is_ignored(&deep.join("src"), true));
    }

    #[test]
    fn parent_cascade_wqmignore_inherited_by_subdir() {
        let root = tmp();
        fs::write(root.path().join(".wqmignore"), "tmp/\n").unwrap();
        let subdir = root.path().join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(&subdir, Some(root.path())).unwrap();
        assert!(m.is_ignored(&subdir.join("tmp"), true));
    }

    #[test]
    fn parent_cascade_reinclusion_overrides_ancestor_gitignore() {
        let root = tmp();
        // Root .gitignore excludes build/
        fs::write(root.path().join(".gitignore"), "build/\n").unwrap();
        // Root .wqmignore re-includes build/
        fs::write(root.path().join(".wqmignore"), "!build/\n").unwrap();
        let subdir = root.path().join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(&subdir, Some(root.path())).unwrap();
        // build/ excluded by .gitignore but re-included by .wqmignore
        assert!(!m.is_ignored(&subdir.join("build"), true));
    }

    #[test]
    fn parent_cascade_mid_level_gitignore_adds_patterns() {
        let root = tmp();
        fs::write(root.path().join(".gitignore"), "*.log\n").unwrap();
        let mid = root.path().join("mid");
        fs::create_dir_all(&mid).unwrap();
        fs::write(mid.join(".gitignore"), "*.tmp\n").unwrap();
        let deep = mid.join("deep");
        fs::create_dir_all(&deep).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(&deep, Some(root.path())).unwrap();
        // Root pattern
        assert!(m.is_ignored(&deep.join("debug.log"), false));
        // Mid-level pattern
        assert!(m.is_ignored(&deep.join("scratch.tmp"), false));
        // Neither
        assert!(!m.is_ignored(&deep.join("main.rs"), false));
    }

    #[test]
    fn parent_cascade_none_root_falls_back_to_dir_only() {
        let root = tmp();
        fs::write(root.path().join(".gitignore"), "dist/\n").unwrap();
        let subdir = root.path().join("subdir");
        fs::create_dir_all(&subdir).unwrap();

        // Without project_root, subdir has no ignore files → None
        assert!(ProjectIgnoreMatcher::for_dir(&subdir, None).is_none());
    }

    // ── descendant matching (#105) ─────────────────────────────────────────

    #[test]
    fn descendants_of_ignored_directory_are_ignored() {
        let root = tmp();
        fs::write(root.path().join(".wqmignore"), "session-env/\n").unwrap();
        let deep = root.path().join("claude-max").join("session-env").join("a");
        fs::create_dir_all(&deep).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(root.path(), Some(root.path())).unwrap();
        // The ignored directory itself
        assert!(m.is_ignored(&root.path().join("claude-max").join("session-env"), true));
        // A directory INSIDE the ignored tree: `Gitignore::matched` alone
        // does not match this — ancestor-aware matching must.
        assert!(m.is_ignored(&deep, true));
        // A file inside the ignored tree
        assert!(m.is_ignored(&deep.join("state.json"), false));
    }

    #[test]
    fn reinclusion_of_directory_rescues_descendants() {
        let root = tmp();
        fs::write(root.path().join(".gitignore"), "vendor/\n").unwrap();
        fs::write(root.path().join(".wqmignore"), "!vendor/\n").unwrap();

        let m = ProjectIgnoreMatcher::for_dir(root.path(), Some(root.path())).unwrap();
        // Descendant of a re-included directory is not ignored
        assert!(!m.is_ignored(&root.path().join("vendor").join("lib").join("a.rs"), false));
    }

    #[test]
    fn descendant_matching_outside_root_falls_back_to_direct_match() {
        let root = tmp();
        fs::write(root.path().join(".gitignore"), "dist/\n").unwrap();
        let elsewhere = tmp();
        let foreign = elsewhere.path().join("dist");
        fs::create_dir_all(&foreign).unwrap();

        let m = ProjectIgnoreMatcher::for_dir(root.path(), Some(root.path())).unwrap();
        // Path outside the matcher root must not panic (the ancestor-aware
        // API panics on foreign paths; is_ignored must route around it).
        // The match outcome for foreign paths is unspecified — only the
        // no-panic guarantee matters here.
        let _ = m.is_ignored(&foreign, true);
        let _ = m.is_ignored(&foreign.join("bundle.js"), false);
    }
}
