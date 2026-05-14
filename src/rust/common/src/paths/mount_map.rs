//! [`MountMap`] — host ↔ container directory pairings.
//!
//! See `docs/specs/16-path-abstraction.md` §5.

use std::collections::HashSet;

use super::canonical::CanonicalPath;
use super::PathError;

/// One host ↔ container directory pairing.
///
/// Both ends are stored as [`CanonicalPath`] (host-absolute, normalized,
/// UTF-8). Construct via [`MountMap::new`]; never construct directly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    /// Host-side directory.
    pub host: CanonicalPath,
    /// Container-side directory.
    pub container: CanonicalPath,
}

/// List of mount pairings with longest-prefix-wins resolution.
///
/// The map is **immutable for process lifetime** (spec §5.3). Loaded once
/// at startup; config edits require process restart.
///
/// An identity map ([`MountMap::identity`]) has zero entries. Per spec §5.5
/// it is still the same plumbing as a non-trivial map — translation just
/// passes the canonical path through unchanged.
#[derive(Debug, Clone, Default)]
pub struct MountMap {
    entries: Vec<MountEntry>,
}

impl MountMap {
    /// Construct an identity mount map (host == container).
    ///
    /// Used for host-native deployments where no translation is required.
    /// Per spec §5.5, [`super::LocalPath::from_canonical`] with an identity
    /// map returns the canonical path verbatim as a [`super::LocalPath`].
    pub fn identity() -> Self {
        MountMap {
            entries: Vec::new(),
        }
    }

    /// Construct a [`MountMap`] from raw `(host, container)` string pairs.
    ///
    /// Each entry is fed through [`CanonicalPath::from_user_input`] — which
    /// applies `~` expansion, `.`/`..` handling, slash collapse, and UTF-8
    /// validation. Duplicate host or duplicate container prefixes are
    /// rejected (spec §5.3); overlapping non-duplicate prefixes are allowed
    /// (resolution is longest-prefix-wins, §5.2).
    ///
    /// # Errors
    ///
    /// * [`PathError::MountMapError`] — duplicate host or container prefix.
    /// * Any [`PathError`] variant produced by canonicalizing an entry.
    ///
    /// # Examples
    ///
    /// ```
    /// use wqm_common::paths::MountMap;
    ///
    /// // Multiple non-overlapping mounts.
    /// let m = MountMap::new(vec![
    ///     ("/Users/chris/dev".to_string(), "/Users/chris/dev".to_string()),
    ///     ("/Volumes/External/books".to_string(), "/mnt/books".to_string()),
    /// ]).unwrap();
    /// assert_eq!(m.len(), 2);
    /// assert!(!m.is_identity());
    ///
    /// // Duplicate host prefix is rejected.
    /// assert!(MountMap::new(vec![
    ///     ("/Users/chris".to_string(), "/mnt/a".to_string()),
    ///     ("/Users/chris".to_string(), "/mnt/b".to_string()),
    /// ]).is_err());
    /// ```
    pub fn new(raw_entries: Vec<(String, String)>) -> Result<Self, PathError> {
        let mut entries: Vec<MountEntry> = Vec::with_capacity(raw_entries.len());
        let mut seen_hosts: HashSet<String> = HashSet::with_capacity(raw_entries.len());
        let mut seen_containers: HashSet<String> = HashSet::with_capacity(raw_entries.len());

        for (host_raw, container_raw) in raw_entries {
            let host = CanonicalPath::from_user_input(&host_raw)?;
            let container = CanonicalPath::from_user_input(&container_raw)?;

            if !seen_hosts.insert(host.as_str().to_string()) {
                return Err(PathError::MountMapError(format!(
                    "duplicate host mount prefix: {host}"
                )));
            }
            if !seen_containers.insert(container.as_str().to_string()) {
                return Err(PathError::MountMapError(format!(
                    "duplicate container mount prefix: {container}"
                )));
            }

            entries.push(MountEntry { host, container });
        }

        Ok(MountMap { entries })
    }

    /// Find the mount entry whose `host` is the longest prefix of the
    /// canonical path, with component-aware matching.
    ///
    /// `/Users/chris/dev` matches `/Users/chris/dev/foo` but **not**
    /// `/Users/chris/development`. The match boundary is a `/` separator,
    /// not a raw substring.
    ///
    /// Returns `None` when no entry covers the canonical path.
    pub(super) fn find_mount_for_canonical(
        &self,
        canonical: &CanonicalPath,
    ) -> Option<&MountEntry> {
        let path_str = canonical.as_str();
        self.entries
            .iter()
            .filter(|entry| component_aware_prefix(path_str, entry.host.as_str()))
            .max_by_key(|entry| entry.host.as_str().len())
    }

    /// Reverse of [`Self::find_mount_for_canonical`]: longest container
    /// prefix covering the given process-local path string.
    pub(super) fn find_mount_for_container(&self, local_str: &str) -> Option<&MountEntry> {
        self.entries
            .iter()
            .filter(|entry| component_aware_prefix(local_str, entry.container.as_str()))
            .max_by_key(|entry| entry.container.as_str().len())
    }

    /// Whether the map is the identity map (no entries).
    pub fn is_identity(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of declared mount entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map has zero declared entries (synonym for
    /// [`Self::is_identity`]).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Whether `prefix` is a component-aware prefix of `path`.
///
/// `/a/b` is a component-aware prefix of `/a/b` and `/a/b/c` but not
/// `/a/bc`. Trailing slashes on `prefix` are tolerated for the purpose of
/// the boundary check.
fn component_aware_prefix(path: &str, prefix: &str) -> bool {
    if path == prefix {
        return true;
    }
    if !path.starts_with(prefix) {
        return false;
    }
    // Either the prefix already ends in `/` (root case `/`), or the
    // character immediately after the prefix in `path` must be `/`.
    prefix.ends_with('/') || path[prefix.len()..].starts_with('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_aware_prefix_exact_match() {
        assert!(component_aware_prefix("/a/b", "/a/b"));
    }

    #[test]
    fn component_aware_prefix_proper_prefix() {
        assert!(component_aware_prefix("/a/b/c", "/a/b"));
    }

    #[test]
    fn component_aware_prefix_rejects_substring_split() {
        // /a/bc must NOT match prefix /a/b
        assert!(!component_aware_prefix("/a/bc", "/a/b"));
    }

    #[test]
    fn component_aware_prefix_root_prefix() {
        // The root `/` is a prefix of every absolute path.
        assert!(component_aware_prefix("/a/b", "/"));
    }

    #[test]
    fn component_aware_prefix_unrelated() {
        assert!(!component_aware_prefix("/a", "/b"));
    }
}
