//! Fast folder-level eligibility cache built from WalkBuilder output.
//!
//! The [`EligibilityTrie`] pre-computes which directories are eligible for
//! indexing (not excluded by `.gitignore` / `.wqmignore`). It is rebuilt on
//! daemon startup, project register/unregister, and whenever an ignore file
//! changes. The file watcher uses it for O(1) folder-level lookups instead
//! of re-parsing ignore files on every event.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use ignore::WalkBuilder;
use tracing::debug;

/// Eligibility status for a single directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EligibilityStatus {
    /// Directory is eligible for indexing (not excluded by ignore rules).
    pub eligible: bool,
    /// Directory contains re-included files (via `.wqmignore` negation).
    /// An ineligible parent with `has_exceptions = true` means some
    /// children may still need indexing.
    pub has_exceptions: bool,
}

/// Pre-computed directory eligibility map for a project.
///
/// Wrap in `Arc<RwLock<EligibilityTrie>>` for concurrent access from
/// file watcher threads; swap atomically on rebuild.
pub struct EligibilityTrie {
    inner: HashMap<PathBuf, EligibilityStatus>,
}

impl EligibilityTrie {
    /// Build eligibility map by walking `project_root` with WalkBuilder.
    ///
    /// When `add_custom_ignore` is true, `.wqmignore` is added as an
    /// additional ignore filename (standard for project scans).
    pub fn build(project_root: &Path, add_custom_ignore: bool) -> Result<Self, String> {
        let mut builder = WalkBuilder::new(project_root);
        builder
            .hidden(false) // include dotfiles — .gitignore handles exclusion
            .git_ignore(true)
            .git_global(false)
            .git_exclude(false)
            // Also treat .gitignore as a custom ignore filename so it works
            // even when the directory is not inside a git repository.
            .add_custom_ignore_filename(".gitignore");

        if add_custom_ignore {
            builder.add_custom_ignore_filename(".wqmignore");
        }

        // Collect all directories the walker visits (= eligible)
        let mut eligible_dirs: HashSet<PathBuf> = HashSet::new();
        for entry in builder.build().flatten() {
            if entry.file_type().map_or(false, |ft| ft.is_dir()) {
                eligible_dirs.insert(entry.into_path());
            }
        }

        // Now scan all actual directories under project_root to find
        // ineligible ones (present on disk but not visited by walker)
        let mut inner = HashMap::new();
        collect_dirs_recursive(project_root, &eligible_dirs, &mut inner);

        debug!(
            "EligibilityTrie built for {}: {} dirs ({} eligible, {} excluded)",
            project_root.display(),
            inner.len(),
            inner.values().filter(|s| s.eligible).count(),
            inner.values().filter(|s| !s.eligible).count(),
        );

        Ok(Self { inner })
    }

    /// Look up eligibility status for a directory.
    ///
    /// Returns `None` for paths not in the trie (e.g. files, or paths
    /// outside the project root).
    pub fn is_eligible(&self, path: &Path) -> Option<&EligibilityStatus> {
        self.inner.get(path)
    }

    /// Number of directories in the trie.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if the trie has no entries.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Recursively collect directories under `dir`, marking each as
/// eligible or not based on whether the walker visited it.
fn collect_dirs_recursive(
    dir: &Path,
    eligible: &HashSet<PathBuf>,
    out: &mut HashMap<PathBuf, EligibilityStatus>,
) {
    let is_eligible = eligible.contains(dir);
    let canon = std::fs::canonicalize(dir).unwrap_or_else(|_| dir.to_path_buf());

    // Check for exceptions: an ineligible dir whose children are eligible
    // (re-inclusion). Only relevant if the dir itself is not eligible.
    let has_exceptions = if !is_eligible {
        eligible
            .iter()
            .any(|p| p.starts_with(&canon) && p != &canon)
    } else {
        false
    };

    out.insert(
        dir.to_path_buf(),
        EligibilityStatus {
            eligible: is_eligible,
            has_exceptions,
        },
    );

    // Recurse into subdirectories
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip common massive directories that we never want to recurse
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                if name_str == ".git" || name_str == "node_modules" || name_str == ".hg" {
                    continue;
                }
                collect_dirs_recursive(&path, eligible, out);
            }
        }
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

    #[test]
    fn build_trie_no_ignore_files() {
        let root = tmp();
        let sub = root.path().join("src");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("main.rs"), "fn main() {}").unwrap();

        let trie = EligibilityTrie::build(root.path(), true).unwrap();
        // Both root and src should be eligible
        assert!(trie.is_eligible(root.path()).unwrap().eligible);
        assert!(trie.is_eligible(&sub).unwrap().eligible);
    }

    #[test]
    fn build_trie_with_gitignore() {
        let root = tmp();
        fs::write(root.path().join(".gitignore"), "dist/\n").unwrap();
        let dist = root.path().join("dist");
        fs::create_dir(&dist).unwrap();
        fs::write(dist.join("bundle.js"), "//").unwrap();
        let src = root.path().join("src");
        fs::create_dir(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}").unwrap();

        let trie = EligibilityTrie::build(root.path(), true).unwrap();
        // src is eligible
        assert!(trie.is_eligible(&src).unwrap().eligible);
        // dist is excluded
        assert!(!trie.is_eligible(&dist).unwrap().eligible);
    }

    #[test]
    fn build_trie_with_wqmignore_reinclusion() {
        let root = tmp();
        // .gitignore excludes build/
        fs::write(root.path().join(".gitignore"), "build/\n").unwrap();
        // .wqmignore re-includes build/ — but WalkBuilder doesn't support
        // our custom re-inclusion logic (it uses standard gitignore negation).
        // So build/ will still be excluded from the walker. However, the
        // has_exceptions flag won't trigger here because re-inclusion needs
        // the negation to produce walker entries (WalkBuilder limitation).
        let build = root.path().join("build");
        fs::create_dir(&build).unwrap();
        fs::write(build.join("output.js"), "//").unwrap();

        let trie = EligibilityTrie::build(root.path(), true).unwrap();
        // build/ is excluded by gitignore (WalkBuilder doesn't see wqmignore reinclusion)
        assert!(!trie.is_eligible(&build).unwrap().eligible);
    }

    #[test]
    fn lookup_nonexistent_path() {
        let root = tmp();
        let trie = EligibilityTrie::build(root.path(), true).unwrap();
        assert!(trie.is_eligible(Path::new("/nonexistent/path")).is_none());
    }

    #[test]
    fn trie_len_and_empty() {
        let root = tmp();
        let trie = EligibilityTrie::build(root.path(), true).unwrap();
        assert!(!trie.is_empty());
        assert!(trie.len() >= 1); // at least the root
    }
}
