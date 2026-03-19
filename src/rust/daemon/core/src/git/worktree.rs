use std::path::{Path, PathBuf};
use tracing::warn;

/// Find the main working tree path for a git worktree.
///
/// Given a worktree's git directory (e.g., `/main/.git/worktrees/feature`),
/// reads the `commondir` file to find the main `.git` directory, then
/// returns its parent as the main working tree root.
///
/// Returns `None` if:
/// - The `commondir` file doesn't exist (not a worktree)
/// - The resolved path doesn't exist on disk
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use workspace_qdrant_core::git::find_main_worktree_path;
///
/// // Given a worktree git dir like /repos/main/.git/worktrees/feature
/// let main_root = find_main_worktree_path(Path::new("/repos/main/.git/worktrees/feature"));
/// // Returns Some("/repos/main")
/// ```
pub fn find_main_worktree_path(worktree_git_dir: &Path) -> Option<PathBuf> {
    let commondir_file = worktree_git_dir.join("commondir");
    let content = match std::fs::read_to_string(&commondir_file) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!(
                "Failed to read commondir at {}: {}",
                commondir_file.display(),
                e
            );
            return None;
        }
    };
    let common_path = content.trim();

    // Resolve absolute or relative path
    let resolved = if Path::new(common_path).is_absolute() {
        PathBuf::from(common_path)
    } else {
        worktree_git_dir.join(common_path)
    };

    // Canonicalize to resolve ".." components and verify existence
    let canonical = match resolved.canonicalize() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            warn!(
                "Failed to canonicalize worktree common dir {}: {}",
                resolved.display(),
                e
            );
            return None;
        }
    };

    // The common dir points to the main .git directory;
    // its parent is the main working tree root
    canonical.parent().map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// T-01: Verify correct path resolution from a simulated worktree structure
    /// with a relative commondir path.
    #[test]
    fn test_find_main_worktree_path_relative_commondir() {
        let temp = TempDir::new().unwrap();
        let main_repo = temp.path().join("main-repo");

        // Create main repo .git directory
        let main_git = main_repo.join(".git");
        fs::create_dir_all(&main_git).unwrap();

        // Create worktree git dir: main-repo/.git/worktrees/feature
        let worktree_git_dir = main_git.join("worktrees").join("feature");
        fs::create_dir_all(&worktree_git_dir).unwrap();

        // commondir typically contains a relative path like "../../"
        // which resolves from worktrees/feature back to .git
        fs::write(worktree_git_dir.join("commondir"), "../..\n").unwrap();

        let result = find_main_worktree_path(&worktree_git_dir);
        assert!(result.is_some(), "should resolve main worktree path");

        let expected = main_repo.canonicalize().unwrap();
        assert_eq!(result.unwrap(), expected);
    }

    /// Test with an absolute commondir path.
    #[test]
    fn test_find_main_worktree_path_absolute_commondir() {
        let temp = TempDir::new().unwrap();
        let main_repo = temp.path().join("main-repo");

        // Create main repo .git directory
        let main_git = main_repo.join(".git");
        fs::create_dir_all(&main_git).unwrap();

        // Create worktree git dir somewhere else
        let worktree_git_dir = temp.path().join("worktree-gitdir");
        fs::create_dir_all(&worktree_git_dir).unwrap();

        // Write absolute path to commondir
        let abs_git_path = main_git.canonicalize().unwrap();
        fs::write(
            worktree_git_dir.join("commondir"),
            abs_git_path.to_str().unwrap(),
        )
        .unwrap();

        let result = find_main_worktree_path(&worktree_git_dir);
        assert!(result.is_some(), "should resolve from absolute commondir");

        let expected = main_repo.canonicalize().unwrap();
        assert_eq!(result.unwrap(), expected);
    }

    /// T-02: Non-worktree directory (no commondir file) returns None.
    #[test]
    fn test_find_main_worktree_path_no_commondir() {
        let temp = TempDir::new().unwrap();
        let result = find_main_worktree_path(temp.path());
        assert!(
            result.is_none(),
            "should return None when commondir is missing"
        );
    }

    /// Missing commondir file returns None.
    #[test]
    fn test_find_main_worktree_path_missing_commondir_file() {
        let temp = TempDir::new().unwrap();
        let fake_git_dir = temp.path().join("fake-git");
        fs::create_dir_all(&fake_git_dir).unwrap();

        let result = find_main_worktree_path(&fake_git_dir);
        assert!(result.is_none());
    }

    /// commondir points to a non-existent path returns None.
    #[test]
    fn test_find_main_worktree_path_nonexistent_resolved_path() {
        let temp = TempDir::new().unwrap();
        let worktree_dir = temp.path().join("wt");
        fs::create_dir_all(&worktree_dir).unwrap();

        fs::write(worktree_dir.join("commondir"), "/nonexistent/path/.git\n").unwrap();

        let result = find_main_worktree_path(&worktree_dir);
        assert!(result.is_none(), "should return None for non-existent path");
    }

    /// commondir with extra whitespace is handled correctly.
    #[test]
    fn test_find_main_worktree_path_whitespace_in_commondir() {
        let temp = TempDir::new().unwrap();
        let main_repo = temp.path().join("main-repo");

        let main_git = main_repo.join(".git");
        fs::create_dir_all(&main_git).unwrap();

        let worktree_git_dir = main_git.join("worktrees").join("feature");
        fs::create_dir_all(&worktree_git_dir).unwrap();

        // Extra whitespace and newlines
        fs::write(worktree_git_dir.join("commondir"), "  ../..\n  ").unwrap();

        let result = find_main_worktree_path(&worktree_git_dir);
        assert!(result.is_some(), "should handle whitespace in commondir");

        let expected = main_repo.canonicalize().unwrap();
        assert_eq!(result.unwrap(), expected);
    }
}
