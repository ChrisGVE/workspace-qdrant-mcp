//! [`MountMap`] — host ↔ container directory pairings.
//!
//! See `docs/specs/16-path-abstraction.md` §5.

use std::collections::HashSet;

use sha2::{Digest, Sha256};

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
    ///     ("/Users/username/dev".to_string(), "/Users/username/dev".to_string()),
    ///     ("/Volumes/External/books".to_string(), "/mnt/books".to_string()),
    /// ]).unwrap();
    /// assert_eq!(m.len(), 2);
    /// assert!(!m.is_identity());
    ///
    /// // Duplicate host prefix is rejected.
    /// assert!(MountMap::new(vec![
    ///     ("/Users/username".to_string(), "/mnt/a".to_string()),
    ///     ("/Users/username".to_string(), "/mnt/b".to_string()),
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

    /// Construct a [`MountMap`] from declared YAML mount entries.
    ///
    /// Convenience adapter for the config loader: each entry's `host` and
    /// `container` strings are forwarded to [`MountMap::new`], which is the
    /// single source of canonicalisation (tilde expansion, absolute-path
    /// validation, `..` rejection) and validation (duplicate host / duplicate
    /// container).
    ///
    /// An empty slice yields an [identity map](MountMap::identity), matching
    /// the spec §5.5 default for host-native deployments.
    ///
    /// # Errors
    ///
    /// See [`MountMap::new`] for the full failure surface.
    ///
    /// # Examples
    ///
    /// ```
    /// use wqm_common::paths::MountMap;
    /// use wqm_common::yaml_defaults::YamlMountEntry;
    ///
    /// let m = MountMap::from_yaml_entries(&[YamlMountEntry {
    ///     host: "/Users/username/dev".to_string(),
    ///     container: "/Users/username/dev".to_string(),
    /// }])
    /// .unwrap();
    /// assert_eq!(m.len(), 1);
    /// ```
    pub fn from_yaml_entries(
        entries: &[crate::yaml_defaults::YamlMountEntry],
    ) -> Result<Self, PathError> {
        let raw: Vec<(String, String)> = entries
            .iter()
            .map(|e| (e.host.clone(), e.container.clone()))
            .collect();
        Self::new(raw)
    }

    /// Find the mount entry whose `host` is the longest prefix of the
    /// canonical path, with component-aware matching.
    ///
    /// `/Users/username/dev` matches `/Users/username/dev/foo` but **not**
    /// `/Users/username/development`. The match boundary is a `/` separator,
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

    /// Iterator over the declared mount entries in declaration order.
    ///
    /// Used by `wqm docker generate-compose` to emit one `volumes:` line per
    /// entry. The ordering matches the YAML source, which keeps the generated
    /// override stable across runs.
    pub fn iter(&self) -> std::slice::Iter<'_, MountEntry> {
        self.entries.iter()
    }
}

/// Compute the SHA-256 hash of a list of [`crate::yaml_defaults::YamlMountEntry`].
///
/// The hash is a stable function of the entries as declared in `config.yaml`
/// (order-sensitive). It is embedded as `# wqm-config-hash: <hex>` in the
/// generated `docker-compose.override.yaml` so that the daemon entrypoint
/// (spec §9.1) and `wqm docker generate-compose --check` can detect drift
/// between override file and live config.
///
/// Hashing strategy: serialise the entries to canonical YAML via
/// `serde_yaml_ng`, then SHA-256 the resulting bytes. We hash the YAML rather
/// than canonical-form strings so that:
///
/// 1. The hash domain is exactly the slice of `config.yaml` the user edits.
/// 2. `~` expansion changes (which depend on `$HOME`) are absorbed before
///    canonicalisation, keeping the hash portable across hosts with the
///    same canonical mount entries.
///
/// Note: `serde_yaml_ng::to_string` writes a stable representation for a
/// given `Vec<YamlMountEntry>` ordering, so the hash is reproducible across
/// process invocations.
///
/// # Examples
///
/// ```
/// use wqm_common::paths::mount_section_hash;
/// use wqm_common::yaml_defaults::YamlMountEntry;
///
/// let h1 = mount_section_hash(&[]);
/// let h2 = mount_section_hash(&[YamlMountEntry {
///     host: "/a".into(),
///     container: "/a".into(),
/// }]);
/// assert_ne!(h1, h2);
/// // Stable across invocations.
/// assert_eq!(h1, mount_section_hash(&[]));
/// ```
pub fn mount_section_hash(entries: &[crate::yaml_defaults::YamlMountEntry]) -> String {
    // `serde_yaml_ng::to_string` of a Vec yields a deterministic
    // representation for a given ordering. The empty-vec case still produces
    // a stable string (`"[]\n"` in current serde_yaml_ng), which is fine.
    let serialised =
        serde_yaml_ng::to_string(entries).expect("YamlMountEntry serialization is infallible");
    let mut hasher = Sha256::new();
    hasher.update(serialised.as_bytes());
    format!("{:x}", hasher.finalize())
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
    use crate::yaml_defaults::YamlMountEntry;

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

    #[test]
    fn mount_section_hash_empty_is_stable() {
        let h1 = mount_section_hash(&[]);
        let h2 = mount_section_hash(&[]);
        assert_eq!(h1, h2);
        // SHA-256 produces a 64-char lower-hex string.
        assert_eq!(h1.len(), 64);
        assert!(h1
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    }

    #[test]
    fn mount_section_hash_changes_with_entries() {
        let h_empty = mount_section_hash(&[]);
        let h_one = mount_section_hash(&[YamlMountEntry {
            host: "/a".into(),
            container: "/a".into(),
        }]);
        assert_ne!(h_empty, h_one);
    }

    #[test]
    fn mount_section_hash_order_sensitive() {
        let a = YamlMountEntry {
            host: "/a".into(),
            container: "/a".into(),
        };
        let b = YamlMountEntry {
            host: "/b".into(),
            container: "/b".into(),
        };
        let h_ab = mount_section_hash(&[a.clone(), b.clone()]);
        let h_ba = mount_section_hash(&[b, a]);
        // Ordering matters — reflects the user's literal YAML.
        assert_ne!(h_ab, h_ba);
    }

    #[test]
    fn mount_section_hash_distinguishes_host_vs_container() {
        let a = YamlMountEntry {
            host: "/x".into(),
            container: "/y".into(),
        };
        let b = YamlMountEntry {
            host: "/y".into(),
            container: "/x".into(),
        };
        assert_ne!(mount_section_hash(&[a]), mount_section_hash(&[b]));
    }
}
