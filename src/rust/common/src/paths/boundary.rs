//! Symlink-aware boundary validation for file I/O.
//!
//! The path abstraction (spec 16 §3.1 rule 7) deliberately avoids
//! `std::fs::canonicalize` at storage time so that symlink names are
//! preserved. However, at **I/O time** a symlink inside a watched root
//! can resolve to a target outside the project boundary.
//!
//! This module provides [`is_within_boundary`] which resolves both
//! `path` and `root` to their real filesystem locations and checks
//! containment. Call it just before performing any read/write
//! operation on a reconstructed absolute path.

use std::io;
use std::path::Path;

/// Returns `true` when the real (symlink-resolved) location of `path`
/// is a descendant of the real location of `root`.
///
/// Both arguments are passed through [`std::fs::canonicalize`] which
/// resolves all symbolic links, `.` and `..` components, and returns
/// the absolute real path. If either path does not exist on disk,
/// canonicalization fails and this function returns `false`.
///
/// # When to call
///
/// Call this at the I/O boundary — after reconstructing an absolute
/// path from `watch_folders.path + relative_path` and before opening,
/// hashing, or reading the file.
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use wqm_common::paths::is_within_boundary;
///
/// let root = Path::new("/Users/username/project");
/// let file = Path::new("/Users/username/project/src/main.rs");
/// assert!(is_within_boundary(file, root));
///
/// let outside = Path::new("/etc/passwd");
/// assert!(!is_within_boundary(outside, root));
/// ```
pub fn is_within_boundary(path: &Path, root: &Path) -> bool {
    match (real_path(path), real_path(root)) {
        (Ok(real_file), Ok(real_root)) => real_file.starts_with(&real_root),
        _ => false,
    }
}

/// Thin wrapper around [`std::fs::canonicalize`] for testability.
// CATEGORY-B: process-local symlink resolution for boundary validation
fn real_path(p: &Path) -> io::Result<std::path::PathBuf> {
    std::fs::canonicalize(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn regular_file_inside_root() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let sub = root.join("src");
        fs::create_dir_all(&sub).unwrap();
        let file = sub.join("main.rs");
        fs::write(&file, "fn main() {}").unwrap();

        assert!(is_within_boundary(&file, root));
    }

    #[test]
    fn file_outside_root_rejected() {
        let root = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let file = outside.path().join("secret.txt");
        fs::write(&file, "secret").unwrap();

        assert!(!is_within_boundary(&file, root.path()));
    }

    #[test]
    fn symlink_inside_root_targeting_inside() {
        let tmp = tempfile::TempDir::new().unwrap();
        let root = tmp.path();
        let real_file = root.join("real.txt");
        fs::write(&real_file, "content").unwrap();

        let link = root.join("link.txt");
        std::os::unix::fs::symlink(&real_file, &link).unwrap();

        assert!(is_within_boundary(&link, root));
    }

    #[test]
    fn symlink_inside_root_targeting_outside() {
        let root = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let external_file = outside.path().join("escape.txt");
        fs::write(&external_file, "escaped!").unwrap();

        let link = root.path().join("sneaky_link.txt");
        std::os::unix::fs::symlink(&external_file, &link).unwrap();

        assert!(
            !is_within_boundary(&link, root.path()),
            "symlink resolving outside root must be rejected"
        );
    }

    #[test]
    fn symlinked_directory_targeting_outside() {
        let root = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let external_dir = outside.path().join("external_src");
        fs::create_dir_all(&external_dir).unwrap();
        let external_file = external_dir.join("lib.rs");
        fs::write(&external_file, "pub mod ext;").unwrap();

        let link_dir = root.path().join("src");
        std::os::unix::fs::symlink(&external_dir, &link_dir).unwrap();

        let file_through_link = link_dir.join("lib.rs");
        assert!(
            !is_within_boundary(&file_through_link, root.path()),
            "file reached via symlinked directory outside root must be rejected"
        );
    }

    #[test]
    fn nonexistent_path_returns_false() {
        let root = tempfile::TempDir::new().unwrap();
        let ghost = root.path().join("does_not_exist.rs");
        assert!(!is_within_boundary(&ghost, root.path()));
    }

    #[test]
    fn nonexistent_root_returns_false() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let bad_root = Path::new("/nonexistent/root");
        assert!(!is_within_boundary(file.path(), bad_root));
    }

    #[test]
    fn broken_symlink_returns_false() {
        let root = tempfile::TempDir::new().unwrap();
        let link = root.path().join("broken");
        std::os::unix::fs::symlink("/nonexistent/target", &link).unwrap();
        assert!(!is_within_boundary(&link, root.path()));
    }

    #[test]
    fn root_itself_is_within_boundary() {
        let tmp = tempfile::TempDir::new().unwrap();
        assert!(is_within_boundary(tmp.path(), tmp.path()));
    }

    #[test]
    fn chained_symlinks_resolved() {
        let root = tempfile::TempDir::new().unwrap();
        let outside = tempfile::TempDir::new().unwrap();
        let real_file = outside.path().join("deep.txt");
        fs::write(&real_file, "deep").unwrap();

        // link1 -> link2 -> real_file (outside root)
        let link2 = root.path().join("link2");
        std::os::unix::fs::symlink(&real_file, &link2).unwrap();
        let link1 = root.path().join("link1");
        std::os::unix::fs::symlink(&link2, &link1).unwrap();

        assert!(
            !is_within_boundary(&link1, root.path()),
            "chained symlinks eventually resolving outside root must be rejected"
        );
    }
}
