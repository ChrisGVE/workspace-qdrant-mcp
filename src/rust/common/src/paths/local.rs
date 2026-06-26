//! [`LocalPath`] — process-local view of a canonical path.
//!
//! See `docs/specs/16-path-abstraction.md` §4.1.

use std::path::{Path, PathBuf};

use super::canonical::CanonicalPath;
use super::mount_map::MountMap;
use super::PathError;

/// Path as seen by the current process's filesystem.
///
/// Differs between host and container deployments. Used only at the
/// filesystem I/O boundary (open, read, stat, walk). **Never serialized**
/// — that is `CanonicalPath`'s job. Compile-time guarantee: this type
/// derives neither `serde::Serialize` nor `serde::Deserialize`.
///
/// Construct via [`LocalPath::from_canonical`] using the active
/// [`MountMap`]. Convert back via [`LocalPath::to_canonical`].
#[derive(Debug, Clone)]
pub struct LocalPath(PathBuf);

impl LocalPath {
    /// Translate a [`CanonicalPath`] to a process-local [`LocalPath`] using
    /// the active mount map.
    ///
    /// With an identity map (host == container), the canonical string is
    /// passed through verbatim. Otherwise, the longest matching host prefix
    /// is replaced with its container prefix per spec §5.2.
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NoMountCoverage`] when the mount map has at
    /// least one entry but none of them covers the canonical path. Identity
    /// maps never produce this error.
    ///
    /// # Examples
    ///
    /// ```
    /// use wqm_common::paths::{CanonicalPath, LocalPath, MountMap};
    ///
    /// // Identity map: pass-through.
    /// let cp = CanonicalPath::from_user_input("/Users/username/dev").unwrap();
    /// let local = LocalPath::from_canonical(&cp, &MountMap::identity()).unwrap();
    /// assert_eq!(local.as_std_path().to_str().unwrap(), "/Users/username/dev");
    ///
    /// // Non-mirror mount swaps the host prefix for the container prefix.
    /// let m = MountMap::new(vec![
    ///     ("/Volumes/External/books".to_string(), "/mnt/books".to_string()),
    /// ]).unwrap();
    /// let cp = CanonicalPath::from_user_input("/Volumes/External/books/x.pdf").unwrap();
    /// let local = LocalPath::from_canonical(&cp, &m).unwrap();
    /// assert_eq!(local.as_std_path().to_str().unwrap(), "/mnt/books/x.pdf");
    /// ```
    pub fn from_canonical(c: &CanonicalPath, mounts: &MountMap) -> Result<Self, PathError> {
        if mounts.is_identity() {
            return Ok(LocalPath(PathBuf::from(c.as_str())));
        }

        let entry =
            mounts
                .find_mount_for_canonical(c)
                .ok_or_else(|| PathError::NoMountCoverage {
                    canonical: c.as_str().to_string(),
                })?;

        let translated = swap_prefix(c.as_str(), entry.host.as_str(), entry.container.as_str());
        Ok(LocalPath(PathBuf::from(translated)))
    }

    /// Reverse: build a [`CanonicalPath`] from a [`LocalPath`].
    ///
    /// Identity-map case re-runs full canonicalization (the local path
    /// might still contain `.` segments etc. from a non-canonical source).
    /// Non-identity case strips the matched container prefix and prepends
    /// the corresponding host prefix, then validates via
    /// [`CanonicalPath::from_validated`].
    ///
    /// # Errors
    ///
    /// Returns [`PathError::NonUtf8`] when the underlying `PathBuf` does
    /// not round-trip through UTF-8, [`PathError::NoMountCoverage`] when
    /// no mount entry covers the local path, or any error from
    /// [`CanonicalPath::from_user_input`]/[`CanonicalPath::from_validated`].
    pub fn to_canonical(&self, mounts: &MountMap) -> Result<CanonicalPath, PathError> {
        let local_str = self.0.to_str().ok_or(PathError::NonUtf8)?;

        if mounts.is_identity() {
            return CanonicalPath::from_user_input(local_str);
        }

        let entry = mounts.find_mount_for_container(local_str).ok_or_else(|| {
            PathError::NoMountCoverage {
                canonical: local_str.to_string(),
            }
        })?;

        let translated = swap_prefix(local_str, entry.container.as_str(), entry.host.as_str());
        CanonicalPath::from_validated(translated)
    }

    /// Borrow the underlying [`Path`] for filesystem I/O.
    ///
    /// This is the **only** sanctioned way to get a [`std::path::Path`]
    /// from a path-typed value in this crate (spec §4.3). All `tokio::fs`,
    /// `std::fs`, `File::open` etc. call sites must accept a
    /// `LocalPath::as_std_path()` rather than a raw string or
    /// [`CanonicalPath`].
    pub fn as_std_path(&self) -> &Path {
        &self.0
    }

    /// Test-only constructor. Wraps an arbitrary [`PathBuf`] without any
    /// validation, so callers can build inputs that exercise the
    /// non-UTF-8 and no-mount-coverage branches of
    /// [`Self::to_canonical`]. Not part of the public API.
    #[cfg(test)]
    pub(crate) fn from_pathbuf_for_test(p: PathBuf) -> Self {
        LocalPath(p)
    }
}

/// Replace `from_prefix` with `to_prefix` at the start of `path`, joining
/// with a single `/`. Caller must have already verified `from_prefix` is
/// a component-aware prefix of `path`.
fn swap_prefix(path: &str, from_prefix: &str, to_prefix: &str) -> String {
    let suffix = &path[from_prefix.len()..];
    let suffix_trimmed = suffix.trim_start_matches('/');
    let to_trimmed = to_prefix.trim_end_matches('/');

    if suffix_trimmed.is_empty() {
        // Whole path equaled the from_prefix. Preserve the to_prefix as-is
        // (do not strip its trailing slash for the root-prefix case).
        to_prefix.to_string()
    } else if to_trimmed.is_empty() {
        // to_prefix was `/` (root). Result is `/<suffix>`.
        format!("/{suffix_trimmed}")
    } else {
        format!("{to_trimmed}/{suffix_trimmed}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_prefix_simple() {
        assert_eq!(
            swap_prefix("/Users/username/dev/x", "/Users/username/dev", "/mnt/dev"),
            "/mnt/dev/x"
        );
    }

    #[test]
    fn swap_prefix_exact_match() {
        assert_eq!(
            swap_prefix("/Users/username/dev", "/Users/username/dev", "/mnt/dev"),
            "/mnt/dev"
        );
    }

    #[test]
    fn swap_prefix_mirror() {
        assert_eq!(
            swap_prefix("/Users/username/dev/x", "/Users/username/dev", "/Users/username/dev",),
            "/Users/username/dev/x"
        );
    }
}
