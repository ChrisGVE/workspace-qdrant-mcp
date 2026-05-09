use std::path::Path;

use super::watcher_types::GitWatcherError;

/// Get the blob hash (git object SHA) for a file at a given revision.
///
/// Uses git2 to look up the tree entry for the path.
pub fn get_blob_hash(
    repo_root: &Path,
    relative_path: &str,
    revision: &str,
) -> Result<String, GitWatcherError> {
    let repo = git2::Repository::open(repo_root)
        .map_err(|e| GitWatcherError::NotGitRepo(format!("{}: {}", repo_root.display(), e)))?;

    let rev = repo.revparse_single(revision).map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Revision not found: {}", e),
        ))
    })?;

    let commit = rev.peel_to_commit().map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Not a commit: {}", e),
        ))
    })?;

    let tree = commit.tree().map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Tree error: {}", e),
        ))
    })?;

    let entry = tree.get_path(Path::new(relative_path)).map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Path not in tree: {}", e),
        ))
    })?;

    Ok(entry.id().to_string())
}

/// List submodule entries from a commit tree (160000 mode).
///
/// Returns Vec of (submodule_path, pinned_sha).
pub fn ls_tree_submodules(
    repo_root: &Path,
    revision: &str,
) -> Result<Vec<(String, String)>, GitWatcherError> {
    let repo = git2::Repository::open(repo_root)
        .map_err(|e| GitWatcherError::NotGitRepo(format!("{}: {}", repo_root.display(), e)))?;

    let rev = repo.revparse_single(revision).map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Revision not found: {}", e),
        ))
    })?;

    let commit = rev.peel_to_commit().map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Not a commit: {}", e),
        ))
    })?;

    let tree = commit.tree().map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Tree error: {}", e),
        ))
    })?;

    let mut submodules = Vec::new();

    tree.walk(git2::TreeWalkMode::PreOrder, |root, entry| {
        // git submodule entries have filemode 0o160000 (S_IFGITLINK)
        if entry.filemode() == 0o160000 {
            let path = if root.is_empty() {
                entry.name().unwrap_or("").to_string()
            } else {
                format!("{}{}", root, entry.name().unwrap_or(""))
            };
            submodules.push((path, entry.id().to_string()));
        }
        git2::TreeWalkResult::Ok
    })
    .map_err(|e| {
        GitWatcherError::Io(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Tree walk error: {}", e),
        ))
    })?;

    Ok(submodules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_blob_hash() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        std::fs::write(temp_dir.path().join("test.txt"), "test content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "commit", &tree, &[])
            .unwrap();

        let hash = get_blob_hash(temp_dir.path(), "test.txt", "HEAD").unwrap();
        assert!(!hash.is_empty());
        assert_eq!(hash.len(), 40);
    }

    #[test]
    fn test_get_blob_hash_missing_file() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo = git2::Repository::init(temp_dir.path()).unwrap();

        let mut config = repo.config().unwrap();
        config.set_str("user.name", "Test").unwrap();
        config.set_str("user.email", "test@example.com").unwrap();

        std::fs::write(temp_dir.path().join("exists.txt"), "content").unwrap();
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("exists.txt")).unwrap();
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = repo.signature().unwrap();
        repo.commit(Some("HEAD"), &sig, &sig, "commit", &tree, &[])
            .unwrap();

        let result = get_blob_hash(temp_dir.path(), "nonexistent.txt", "HEAD");
        assert!(result.is_err());
    }
}
