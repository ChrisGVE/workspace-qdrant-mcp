//! Validation and mount-map handling for [`DaemonConfig`].
//!
//! [`DaemonConfig::validate`] is the SINGLE validation entry point (WI-a2):
//! both the loader and `memexd::main` call it. It checks top-level scalar
//! fields (via the shared `wqm_common::config` validators) and every
//! sub-configuration section, returning the first error encountered.

use wqm_common::config::validate_url;
use wqm_common::paths::MountMap;

use super::types::DaemonConfig;
use crate::storage::StorageConfig;

impl DaemonConfig {
    /// Create a daemon-mode configuration optimized for MCP stdio protocol
    /// compliance. This configuration disables compatibility checking to
    /// prevent console output.
    pub fn daemon_mode() -> Self {
        let mut config = Self::default();
        config.qdrant = StorageConfig::daemon_mode(); // Use silent StorageConfig
        config
    }

    /// Validate the full configuration, returning the first error encountered.
    ///
    /// Top-level scalar fields are checked first (consolidating what the loader
    /// previously checked separately), followed by every sub-section.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_top_level()?;

        self.queue_processor
            .validate()
            .map_err(|e| format!("queue_processor: {e}"))?;
        self.monitoring
            .validate()
            .map_err(|e| format!("monitoring: {e}"))?;
        self.git.validate().map_err(|e| format!("git: {e}"))?;
        self.observability
            .validate()
            .map_err(|e| format!("observability: {e}"))?;
        self.embedding
            .validate()
            .map_err(|e| format!("embedding: {e}"))?;
        self.lsp.validate().map_err(|e| format!("lsp: {e}"))?;
        self.grammars
            .validate()
            .map_err(|e| format!("grammars: {e}"))?;
        self.updates
            .validate()
            .map_err(|e| format!("updates: {e}"))?;
        // resource_limits uses 0 as sentinel for auto-detect; resolve
        // hardware-specific defaults on a temporary clone before validating.
        let mut resolved_limits = self.resource_limits.clone();
        resolved_limits.resolve_auto_values();
        resolved_limits
            .validate()
            .map_err(|e| format!("resource_limits: {e}"))?;
        self.startup
            .validate()
            .map_err(|e| format!("startup: {e}"))?;
        self.daemon_endpoint
            .validate()
            .map_err(|e| format!("daemon_endpoint: {e}"))?;
        self.ingestion_limits
            .validate()
            .map_err(|e| format!("ingestion_limits: {e}"))?;
        self.concept
            .validate()
            .map_err(|e| format!("concept: {e}"))?;
        self.graph_rag
            .validate()
            .map_err(|e| format!("graph_rag: {e}"))?;
        self.narrative
            .validate()
            .map_err(|e| format!("narrative: {e}"))?;
        self.auto_ingestion
            .validate()
            .map_err(|e| format!("auto_ingestion: {e}"))?;
        self.validate_mounts().map_err(|e| format!("mounts: {e}"))?;
        Ok(())
    }

    /// Validate the top-level scalar fields.
    ///
    /// Mirrors the checks formerly in `unified_config::validation::validate_config`,
    /// now expressed through the shared validators where they map cleanly.
    fn validate_top_level(&self) -> Result<(), String> {
        if let Some(max_concurrent) = self.max_concurrent_tasks {
            if max_concurrent == 0 {
                return Err("max_concurrent_tasks must be greater than 0".to_string());
            }
            if max_concurrent > 100 {
                return Err("max_concurrent_tasks should not exceed 100".to_string());
            }
        }

        if let Some(timeout) = self.default_timeout_ms {
            if timeout == 0 {
                return Err("default_timeout_ms must be greater than 0".to_string());
            }
            if timeout > 300_000 {
                return Err("default_timeout_ms should not exceed 5 minutes".to_string());
            }
        }

        if self.chunk_size == 0 {
            return Err("chunk_size must be greater than 0".to_string());
        }
        if self.chunk_size > 10_000 {
            return Err("chunk_size should not exceed 10,000".to_string());
        }

        let valid_log_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_log_levels.contains(&self.log_level.as_str()) {
            return Err(format!(
                "log_level must be one of: {}",
                valid_log_levels.join(", ")
            ));
        }

        // Shared validator: non-empty + http(s) scheme.
        validate_url(&self.qdrant.url).map_err(|e| format!("qdrant.url: {e}"))?;

        Ok(())
    }

    /// Construct the [`MountMap`] declared by this configuration.
    ///
    /// Tilde expansion, absolute-path validation, `..`-rejection, and
    /// duplicate host/container detection are all delegated to
    /// [`MountMap::from_yaml_entries`]. An empty `mounts` list yields the
    /// identity map (spec 16 §5.5).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`wqm_common::paths::PathError`] formatted
    /// as a string when any entry fails canonicalisation or validation.
    pub fn build_mount_map(&self) -> Result<MountMap, String> {
        MountMap::from_yaml_entries(&self.mounts).map_err(|e| e.to_string())
    }

    /// Validate the declared mount-map entries without keeping the result.
    ///
    /// Called by [`DaemonConfig::validate`]. The actual [`MountMap`]
    /// instance is constructed once at process startup via
    /// [`DaemonConfig::build_mount_map`] and is immutable for the process
    /// lifetime (spec 16 §5.3).
    pub fn validate_mounts(&self) -> Result<(), String> {
        // Empty list is the identity map — explicitly always valid.
        if self.mounts.is_empty() {
            return Ok(());
        }
        // Discarding the returned MountMap is intentional: this call site
        // only checks well-formedness. Live construction happens elsewhere.
        let _ = self.build_mount_map()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::DaemonConfig;
    use wqm_common::yaml_defaults::YamlMountEntry;

// ── Mount-map (T3) tests ────────────────────────────────────────────────

fn mk_mount(host: &str, container: &str) -> YamlMountEntry {
    YamlMountEntry {
        host: host.to_string(),
        container: container.to_string(),
    }
}

#[test]
fn mounts_empty_yields_identity_map() {
    // T3.10: default DaemonConfig has no mount entries → identity map.
    let cfg = DaemonConfig::default();
    assert!(cfg.mounts.is_empty(), "default mounts list must be empty");
    let map = cfg.build_mount_map().expect("identity map always builds");
    assert!(
        map.is_identity(),
        "empty mounts list must yield identity map"
    );
    assert_eq!(map.len(), 0);
    assert!(cfg.validate_mounts().is_ok());
}

#[test]
fn mounts_valid_entries_with_tilde_expansion_load() {
    // T3.11: a leading `~` is expanded once on load.
    let home = dirs::home_dir().expect("HOME must be set");
    let home_str = home.to_str().expect("home path must be UTF-8");

    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![
        mk_mount("/Users/chris/dev", "/Users/chris/dev"),
        mk_mount("~/reference", "/mnt/reference"),
    ];

    let map = cfg.build_mount_map().expect("valid mounts must load");
    assert_eq!(map.len(), 2);
    assert!(!map.is_identity());

    let expanded = format!("{home_str}/reference");
    let canon = wqm_common::paths::CanonicalPath::from_user_input(&expanded)
        .expect("expanded path must canonicalise");
    assert!(canon.as_str().starts_with(home_str));
    assert!(cfg.validate_mounts().is_ok());
}

#[test]
fn mounts_overlapping_entries_allowed() {
    // T3.12: overlap (one entry's host is a prefix of another) is allowed.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![
        mk_mount("/Users/chris", "/mnt/user"),
        mk_mount("/Users/chris/dev", "/mnt/dev"),
    ];
    let map = cfg.build_mount_map().expect("overlap must be allowed");
    assert_eq!(map.len(), 2);
    assert!(cfg.validate_mounts().is_ok());
}

#[test]
fn mounts_duplicate_host_prefix_rejected() {
    // T3.13: two entries with identical canonical host → reject.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![
        mk_mount("/Users/chris", "/mnt/a"),
        mk_mount("/Users/chris", "/mnt/b"),
    ];
    let err = cfg
        .build_mount_map()
        .expect_err("duplicate host must error");
    assert!(
        err.contains("duplicate host"),
        "expected duplicate-host message in '{err}'"
    );
    assert!(cfg.validate_mounts().is_err());
}

#[test]
fn mounts_duplicate_container_prefix_rejected() {
    // T3.14: duplicate container canonical form → reject.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![
        mk_mount("/Users/chris/a", "/mnt/shared"),
        mk_mount("/Users/chris/b", "/mnt/shared"),
    ];
    let err = cfg
        .build_mount_map()
        .expect_err("duplicate container must error");
    assert!(
        err.contains("duplicate container"),
        "expected duplicate-container message in '{err}'"
    );
    assert!(cfg.validate_mounts().is_err());
}

#[test]
fn mounts_relative_host_path_rejected() {
    // T3.15: a relative host path is rejected.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![mk_mount("relative/host", "/mnt/x")];
    assert!(cfg.build_mount_map().is_err());
    assert!(cfg.validate_mounts().is_err());
}

#[test]
fn mounts_relative_container_path_rejected() {
    // T3.16: a relative container path is rejected.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![mk_mount("/Users/chris/dev", "relative/container")];
    assert!(cfg.build_mount_map().is_err());
    assert!(cfg.validate_mounts().is_err());
}

#[test]
fn mounts_parent_dir_segment_rejected() {
    // Spec §3.1 rule 4: `..` in either host or container is rejected.
    let mut cfg_host = DaemonConfig::default();
    cfg_host.mounts = vec![mk_mount("/Users/chris/../other", "/mnt/x")];
    assert!(cfg_host.build_mount_map().is_err());

    let mut cfg_container = DaemonConfig::default();
    cfg_container.mounts = vec![mk_mount("/Users/chris/dev", "/mnt/../other")];
    assert!(cfg_container.build_mount_map().is_err());
}

#[test]
fn mounts_validate_chain_surfaces_mounts_prefix() {
    // T3.9 / T3.17: DaemonConfig::validate() routes mount errors with the
    // `mounts:` prefix so log readers can identify the failing section.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![mk_mount("relative/path", "/mnt/x")];
    let err = cfg
        .validate()
        .expect_err("invalid mount must fail validate");
    assert!(
        err.starts_with("mounts:"),
        "expected 'mounts:' prefix in '{err}'"
    );
}

#[test]
fn mounts_yaml_round_trip_preserves_entries() {
    // Acceptance (T3.19): full config-load round-trip survives the new
    // section without loss.
    let mut original = DaemonConfig::default();
    original.mounts = vec![
        mk_mount("/Users/chris/dev", "/Users/chris/dev"),
        mk_mount("/Volumes/External/books", "/mnt/external-books"),
    ];
    let yaml = serde_yaml_ng::to_string(&original).expect("serialise");
    let restored: DaemonConfig = serde_yaml_ng::from_str(&yaml).expect("deserialise");
    assert_eq!(restored.mounts, original.mounts);
    assert_eq!(restored.mounts.len(), 2);
    assert!(restored.validate().is_ok());
}

#[test]
fn mounts_acceptance_edge_cases() {
    // T3.20: edge cases bundled into a single acceptance test.

    // 1. Missing `mounts:` section.
    let full = serde_yaml_ng::to_string(&DaemonConfig::default()).expect("serialise default");
    let no_section: String = full
        .lines()
        .filter(|l| !l.starts_with("mounts:"))
        .map(|l| format!("{l}\n"))
        .collect();
    let cfg: DaemonConfig = serde_yaml_ng::from_str(&no_section)
        .expect("DaemonConfig must deserialise without mounts section");
    assert!(cfg.mounts.is_empty());
    assert!(cfg.build_mount_map().unwrap().is_identity());

    // 2. Explicit empty list embedded in an otherwise complete config.
    let mut empty = DaemonConfig::default();
    empty.mounts.clear();
    let yaml_empty = serde_yaml_ng::to_string(&empty).expect("serialise");
    let cfg: DaemonConfig = serde_yaml_ng::from_str(&yaml_empty).expect("empty mounts list parses");
    assert!(cfg.mounts.is_empty());

    // 3. Mirror mount — identical host and container.
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![mk_mount("/Users/chris/dev", "/Users/chris/dev")];
    let map = cfg.build_mount_map().expect("mirror is valid");
    assert_eq!(map.len(), 1);
    assert!(!map.is_identity(), "mirror is not the identity map");

    // 4. Three-entry mount set survives end-to-end validate().
    let mut cfg = DaemonConfig::default();
    cfg.mounts = vec![
        mk_mount("/Users/chris/dev", "/Users/chris/dev"),
        mk_mount("/Volumes/External/books", "/mnt/external-books"),
        mk_mount("/Users/chris/reference", "/mnt/reference"),
    ];
    assert!(cfg.validate().is_ok());
}

}
