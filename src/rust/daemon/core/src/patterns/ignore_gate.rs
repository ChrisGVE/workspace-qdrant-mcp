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
use std::sync::Arc;

use ignore::gitignore::Gitignore;

use super::gitignore::ProjectIgnoreMatcher;
use super::global_ignore;

/// Combined per-project + daemon-wide ignore matcher for walk-based callers.
pub struct IgnoreGate {
    project: Option<ProjectIgnoreMatcher>,
    global: Option<Arc<Gitignore>>,
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
        Self::with_shared_global(dir, project_root, Self::build_shared_global(global_path))
    }

    /// Compile the `global.wqmignore` matcher once for reuse across many gates.
    ///
    /// [`IgnoreGate::for_dir`] recompiles the global file on every call — fine
    /// for a single-directory scan item, wasteful for callers that build one
    /// gate per parent directory (the git-index enumeration touches hundreds).
    /// Build the shared matcher up front and hand clones of the `Arc` to
    /// [`IgnoreGate::with_shared_global`].
    pub fn build_shared_global(global_path: Option<&Path>) -> Option<Arc<Gitignore>> {
        global_path
            .and_then(global_ignore::matcher_from)
            .map(Arc::new)
    }

    /// Build a gate around an already-compiled global matcher.
    ///
    /// Same semantics as [`IgnoreGate::for_dir`]; only the global compilation
    /// is hoisted out (see [`IgnoreGate::build_shared_global`]).
    pub fn with_shared_global(
        dir: &Path,
        project_root: Option<&Path>,
        global: Option<Arc<Gitignore>>,
    ) -> Self {
        Self {
            project: ProjectIgnoreMatcher::for_dir(dir, project_root),
            global,
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
            if global.matched_path_or_any_parents(path, is_dir).is_ignore() {
                return true;
            }
        }
        false
    }

    /// Parent-aware check for a file reached *directly* rather than through a
    /// pruning walk (git-index enumeration, dequeue-time re-check).
    ///
    /// A directory walk tests `generated/`-style rules against the directory
    /// entry and never descends, so nested files are never reached. A caller
    /// that jumps straight to a nested file must replay that logic: test the
    /// file itself, then each ancestor directory up to (but not including)
    /// `root`. The global layer already consults parents via
    /// `matched_path_or_any_parents`; the replay is what extends the same
    /// guarantee to the project layer's plain `matched` semantics.
    pub fn is_ignored_with_ancestors(&self, root: &Path, abs_file: &Path) -> bool {
        if self.is_ignored(abs_file, false) {
            return true;
        }
        let mut current = abs_file.parent();
        while let Some(dir) = current {
            if dir == root || !dir.starts_with(root) {
                break;
            }
            if self.is_ignored(dir, true) {
                return true;
            }
            current = dir.parent();
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
        assert!(
            gate.is_ignored(&gen, false),
            "global generated must be ignored"
        );
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

    // ── pattern-form coverage (the generated-proto regression) ───────────────
    // global.wqmignore carried `**/proto/src/generated/` and
    // `**/*OuterClass.java` for hours while the git-index scan kept enqueueing
    // matching files: that path never consulted the global layer at all. These
    // tests pin every pattern form the operators reached for, so a matcher
    // regression in ANY caller of IgnoreGate fails here first.

    /// The four global pattern forms that must all exclude a nested file.
    const GENERATED_FORMS: &[&str] = &[
        "**/proto/src/generated/\n",   // dir-only, trailing slash
        "**/proto/src/generated/**\n", // dir contents
        "*OuterClass.java\n",          // bare glob (no slash → any depth)
        "**/*OuterClass.java\n",       // **/-anchored glob
    ];

    #[test]
    fn all_global_pattern_forms_exclude_nested_generated_file() {
        for body in GENERATED_FORMS {
            let (proj, _g, gpath) = setup(body);
            let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), Some(&gpath));
            let f = proj
                .path()
                .join("doc-backend/proto/src/generated/doc/PolicyOuterClass.java");
            assert!(
                gate.is_ignored(&f, false),
                "pattern {body:?} must exclude nested generated file via is_ignored"
            );
            assert!(
                gate.is_ignored_with_ancestors(proj.path(), &f),
                "pattern {body:?} must exclude nested generated file via ancestors"
            );
            let keep = proj.path().join("doc-backend/src/main/java/App.java");
            assert!(
                !gate.is_ignored_with_ancestors(proj.path(), &keep),
                "pattern {body:?} must keep hand-authored source"
            );
        }
    }

    #[test]
    fn project_dir_rule_needs_ancestor_replay_for_direct_file_access() {
        // A pruning walk tests `generated/` against the directory and never
        // descends. A git-index enumeration reaches the nested file directly:
        // plain is_ignored misses the dir-only project rule (this assertion
        // documents the gap); is_ignored_with_ancestors closes it.
        let proj = tempfile::tempdir().unwrap();
        write(proj.path(), ".wqmignore", "generated/\n");
        let gate = IgnoreGate::for_dir(proj.path(), Some(proj.path()), None);
        let f = proj.path().join("pkg/generated/x.dart");
        assert!(
            !gate.is_ignored(&f, false),
            "dir-only project rule does not match the file itself"
        );
        assert!(
            gate.is_ignored_with_ancestors(proj.path(), &f),
            "ancestor replay must catch the generated/ directory rule"
        );
    }

    #[test]
    fn ancestor_replay_does_not_escape_above_root() {
        // The project sits INSIDE a directory whose name matches a project
        // ignore rule. The replay walks up to, but never onto or past, the
        // watch root — a `generated/` rule must not exclude the whole project
        // just because the root's own ancestors contain a `generated` path
        // component. (The global layer intentionally differs: it matches
        // absolute paths and their parents at any depth.)
        let outer = tempfile::tempdir().unwrap();
        let root = outer.path().join("generated/proj");
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join(".wqmignore"), "generated/\n").unwrap();
        let gate = IgnoreGate::for_dir(&root, Some(&root), None);
        let f = root.join("src/lib.rs");
        assert!(
            !gate.is_ignored_with_ancestors(&root, &f),
            "rule must not match ancestors at or above the watch root"
        );
        // ...while the same rule still excludes a generated/ dir INSIDE it.
        let inner = root.join("pkg/generated/x.dart");
        assert!(gate.is_ignored_with_ancestors(&root, &inner));
    }

    #[test]
    fn shared_global_constructor_matches_for_dir_semantics() {
        let body = "**/proto/src/generated/\n*OuterClass.java\n";
        let (proj, _g, gpath) = setup(body);
        let shared = IgnoreGate::build_shared_global(Some(&gpath));
        assert!(shared.is_some(), "global matcher must compile");
        let via_shared =
            IgnoreGate::with_shared_global(proj.path(), Some(proj.path()), shared.clone());
        let via_path = IgnoreGate::for_dir(proj.path(), Some(proj.path()), Some(&gpath));

        let probes = [
            (proj.path().join("a/proto/src/generated/X.java"), true),
            (proj.path().join("a/b/FooOuterClass.java"), true),
            (proj.path().join("a/b/Foo.java"), false),
        ];
        for (p, want) in probes {
            assert_eq!(
                via_shared.is_ignored_with_ancestors(proj.path(), &p),
                want,
                "shared-global gate disagrees for {p:?}"
            );
            assert_eq!(
                via_path.is_ignored_with_ancestors(proj.path(), &p),
                want,
                "for_dir gate disagrees for {p:?}"
            );
        }
    }
}
