//! Unified ignore decision for the directory-walk paths.
//!
//! "Should this path be indexed?" was historically computed in two places that
//! drifted apart and caused a series of bugs:
//!   - the folder-scan walk (`strategies/processing/folder/scan.rs`) combined a
//!     per-project [`ProjectIgnoreMatcher`] with a separate global check;
//!   - the ignore reconciler (`startup/reconciliation/ignore_sync.rs`) walked
//!     with `WalkBuilder` and post-filtered through the global matcher.
//!
//! [`IgnoreGate`] is the single type both paths now use: it bundles the
//! per-project `.gitignore`/`.wqmignore` cascade with the daemon-wide
//! `global.wqmignore`, applying them with one consistent semantics
//! (root-anchored `matched_path_or_any_parents`, re-inclusion honoured). The
//! file watcher keeps the lean [`super::global_ignore::is_globally_ignored`]
//! call instead — it filters individual fs events and does not walk a project
//! tree, so it has no `.gitignore` cascade to apply.
//!
//! Resolution order (matches [`ProjectIgnoreMatcher`] + global precedence):
//!   1. project `.wqmignore` re-inclusion wins (overrides project `.gitignore`);
//!   2. project `.gitignore`/`.wqmignore` exclusion → ignored;
//!   3. `global.wqmignore` exclusion → ignored;
//!   4. otherwise → kept.

use std::path::Path;

use ignore::gitignore::Gitignore;

use super::gitignore::ProjectIgnoreMatcher;
use super::global_ignore;

/// Combined per-project + daemon-wide ignore matcher for walk-based callers.
pub struct IgnoreGate {
    project: Option<ProjectIgnoreMatcher>,
    global: Option<Gitignore>,
}

impl IgnoreGate {
    /// Build a gate for a directory being walked.
    ///
    /// `project_root` cascades the project `.gitignore`/`.wqmignore` from the
    /// root down to `dir` (see [`ProjectIgnoreMatcher::for_dir`]). `global_path`
    /// is the resolved `global.wqmignore` location — pass
    /// [`global_ignore::resolve_global_ignore_path`] for the live daemon, or an
    /// explicit path in tests. Either layer may be absent.
    pub fn for_dir(dir: &Path, project_root: Option<&Path>, global_path: Option<&Path>) -> Self {
        Self {
            project: ProjectIgnoreMatcher::for_dir(dir, project_root),
            global: global_path.and_then(global_ignore::matcher_from),
        }
    }

    /// Returns `true` when `path` is excluded by the project ignore files OR by
    /// `global.wqmignore`. `is_dir` must be `true` for directory entries so a
    /// directory pattern prunes the subtree before the walk descends.
    pub fn is_ignored(&self, path: &Path, is_dir: bool) -> bool {
        if let Some(ref project) = self.project {
            if project.is_ignored(path, is_dir) {
                return true;
            }
        }
        if let Some(ref global) = self.global {
            if global
                .matched_path_or_any_parents(path, is_dir)
                .is_ignore()
            {
                return true;
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::TempDir;

    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        fs::write(dir.join(name), body).unwrap();
    }

    /// Build a temp project root + a temp global.wqmignore with `global_body`.
    fn setup(global_body: &str) -> (TempDir, TempDir, std::path::PathBuf) {
        let proj = tempfile::tempdir().unwrap();
        let gdir = tempfile::tempdir().unwrap();
        let gpath = gdir.path().join("global.wqmignore");
        fs::write(&gpath, global_body).unwrap();
        (proj, gdir, gpath)
    }

    #[test]
    fn global_only_excludes_deep_match() {
        let (proj, _g, gpath) = setup("**/generated/\n**/state/qdrant/\n");
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), Some(&gpath));
        let gen = proj.path().join("a/b/generated/x.dart");
        let keep = proj.path().join("a/b/src/main.rs");
        assert!(gate.is_ignored(&gen, false), "global generated must be ignored");
        assert!(!gate.is_ignored(&keep, false), "source must be kept");
    }

    #[test]
    fn project_gitignore_excludes() {
        let (proj, _g, gpath) = setup("**/generated/\n");
        write(proj.path(), ".gitignore", "secrets/\n");
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), Some(&gpath));
        assert!(gate.is_ignored(&proj.path().join("secrets"), true));
        assert!(!gate.is_ignored(&proj.path().join("lib"), true));
    }

    #[test]
    fn project_and_global_both_apply() {
        let (proj, _g, gpath) = setup("**/state/qdrant/\n");
        write(proj.path(), ".wqmignore", "datasets/\n");
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), Some(&gpath));
        // project .wqmignore
        assert!(gate.is_ignored(&proj.path().join("datasets"), true));
        // global
        assert!(gate.is_ignored(&proj.path().join("x/state/qdrant/seg.json"), false));
        // neither
        assert!(!gate.is_ignored(&proj.path().join("src/main.rs"), false));
    }

    #[test]
    fn missing_global_falls_back_to_project_only() {
        let proj = tempfile::tempdir().unwrap();
        write(proj.path(), ".gitignore", "build/\n");
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), None);
        assert!(gate.is_ignored(&proj.path().join("build"), true));
        assert!(!gate.is_ignored(&proj.path().join("src"), true));
    }

    #[test]
    fn no_matchers_keeps_everything() {
        let proj = tempfile::tempdir().unwrap();
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), None);
        assert!(!gate.is_ignored(&proj.path().join("anything.rs"), false));
    }
}
