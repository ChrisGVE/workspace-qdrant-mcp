//! Daemon configuration loading (WI-a2).
//!
//! Replaces the deleted `unified_config::UnifiedConfigManager`. Stateless free
//! functions discover, parse, env-override, path-expand, and validate the
//! daemon config. Discovery uses the shared `wqm_common::paths` search-path
//! cascade; path expansion uses the shared `wqm_common::env_expand`; validation
//! is the single [`DaemonConfig::validate`] method.
//!
//! User YAML is always parsed into [`YamlConfig`] first (every section carries
//! `#[serde(default)]`), then converted via `From<&YamlConfig>` so partial user
//! configs layer correctly on top of compiled-in defaults.

use std::fs;
use std::path::{Path, PathBuf};

use tracing::{debug, info};
use wqm_common::env_expand::expand_env_vars;
use wqm_common::yaml_defaults::YamlConfig;

use crate::config::env::apply_env_overrides;
use crate::config::error::ConfigError;
use crate::config::DaemonConfig;

/// Config search paths in priority order (first existing file wins).
///
/// Delegates to the shared `wqm_common::paths::get_config_search_paths()`.
pub fn config_search_paths() -> Vec<PathBuf> {
    wqm_common::paths::get_config_search_paths()
}

/// The first existing config file from the canonical search paths.
fn preferred_config_source() -> Option<PathBuf> {
    config_search_paths().into_iter().find(|p| p.exists())
}

/// Load daemon configuration, applying env overrides, path expansion, and
/// validation.
///
/// - `Some(path)` that is missing → [`ConfigError::FileNotFound`].
/// - `Some(path)` that exists, or an auto-discovered file → parse → env
///   overrides → path expansion → [`DaemonConfig::validate`].
/// - `None` with no discoverable file → built-in [`DaemonConfig::default`]
///   (no env overrides, no validation — callers apply those as needed).
pub fn load_config(config_file: Option<&Path>) -> Result<DaemonConfig, ConfigError> {
    let source = match config_file {
        Some(file) => {
            if !file.exists() {
                return Err(ConfigError::FileNotFound(file.to_path_buf()));
            }
            file.to_path_buf()
        }
        None => match preferred_config_source() {
            Some(path) => {
                info!("Loading configuration from: {}", path.display());
                path
            }
            None => {
                info!("No configuration file found, using defaults");
                return Ok(DaemonConfig::default());
            }
        },
    };

    let raw = load_config_file(&source)?;
    let with_env = apply_env_overrides(raw);
    let expanded = expand_config_paths(with_env);
    expanded.validate().map_err(ConfigError::Validation)?;

    info!("Configuration loaded and validated successfully");
    Ok(expanded)
}

/// Read and deserialise a YAML config file into [`DaemonConfig`].
fn load_config_file(path: &Path) -> Result<DaemonConfig, ConfigError> {
    let content = fs::read_to_string(path)?;
    let yaml: YamlConfig =
        serde_yaml_ng::from_str(&content).map_err(|e| ConfigError::Parse(e.to_string()))?;
    Ok(DaemonConfig::from(&yaml))
}

/// Expand environment variables in path-like configuration values.
pub(crate) fn expand_config_paths(mut config: DaemonConfig) -> DaemonConfig {
    if let Some(ref path) = config.log_file {
        config.log_file = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }
    if let Some(ref path) = config.project_path {
        config.project_path = Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }
    if let Some(ref path) = config.embedding.model_cache_dir {
        config.embedding.model_cache_dir =
            Some(PathBuf::from(expand_env_vars(&path.to_string_lossy())));
    }
    debug!("Expanded environment variables in path configuration values");
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use wqm_common::yaml_defaults::{self, YamlMountEntry};

    // ── No-project-local search paths ───────────────────────────────────────

    #[test]
    fn search_paths_exclude_project_local() {
        for p in config_search_paths() {
            let s = p.to_string_lossy();
            assert!(
                !s.ends_with(".workspace-qdrant.yaml") && !s.ends_with(".workspace-qdrant.yml"),
                "must not search project-local config: {p:?}"
            );
        }
    }

    // ── load_config behaviour ───────────────────────────────────────────────

    #[test]
    fn missing_explicit_file_is_file_not_found() {
        let missing = Path::new("/tmp/wqm-nonexistent-config-xyz.yaml");
        let err = load_config(Some(missing)).expect_err("missing file must error");
        assert!(matches!(err, ConfigError::FileNotFound(_)));
    }

    #[test]
    fn valid_yaml_file_loads_overriding_defaults() {
        // The on-disk format is the YamlConfig shape (`performance.chunk_size`),
        // NOT a serialized DaemonConfig. The loader parses YamlConfig then
        // converts via `From<&YamlConfig>`.
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        std::fs::write(
            &cfg_path,
            "performance:\n  chunk_size: 777\nqdrant:\n  url: \"http://h:6333\"\n",
        )
        .unwrap();

        let loaded = load_config(Some(&cfg_path)).expect("valid file loads");
        assert_eq!(loaded.chunk_size, 777);
        assert_eq!(loaded.qdrant.url, "http://h:6333");
    }

    #[test]
    fn malformed_file_is_parse_error() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        std::fs::write(&cfg_path, "not: valid: yaml: structure: }}").unwrap();
        let err = load_config(Some(&cfg_path)).expect_err("malformed must error");
        assert!(matches!(err, ConfigError::Parse(_)));
    }

    // ── path expansion ──────────────────────────────────────────────────────

    #[test]
    fn expand_config_paths_expands_env_vars() {
        std::env::set_var("WQM_TEST_DIR_LOADER", "/expanded/dir");
        let mut config = DaemonConfig::default();
        config.log_file = Some(PathBuf::from("${WQM_TEST_DIR_LOADER}/daemon.log"));
        config.project_path = Some(PathBuf::from("$WQM_TEST_DIR_LOADER/project"));
        config.embedding.model_cache_dir = Some(PathBuf::from("${WQM_TEST_DIR_LOADER}/models"));

        let expanded = expand_config_paths(config);
        std::env::remove_var("WQM_TEST_DIR_LOADER");

        assert_eq!(
            expanded.log_file.unwrap(),
            PathBuf::from("/expanded/dir/daemon.log")
        );
        assert_eq!(
            expanded.project_path.unwrap(),
            PathBuf::from("/expanded/dir/project")
        );
        assert_eq!(
            expanded.embedding.model_cache_dir.unwrap(),
            PathBuf::from("/expanded/dir/models")
        );
    }

    // ── mount-map end-to-end through load_config (T3.17/T3.19) ───────────────

    #[test]
    fn load_config_rejects_invalid_mount_relative_host() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![YamlMountEntry {
            host: "relative/host".to_string(),
            container: "/mnt/x".to_string(),
        }];
        std::fs::write(&cfg_path, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();

        let err = load_config(Some(&cfg_path)).expect_err("invalid mount must reject");
        match err {
            ConfigError::Validation(msg) => assert!(
                msg.contains("mounts:"),
                "expected 'mounts:' in validation error '{msg}'"
            ),
            other => panic!("unexpected error variant: {other:?}"),
        }
    }

    #[test]
    fn load_config_accepts_empty_mounts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        let mut cfg = DaemonConfig::default();
        cfg.mounts.clear();
        std::fs::write(&cfg_path, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();

        let loaded = load_config(Some(&cfg_path)).expect("empty mounts must load");
        assert!(loaded.mounts.is_empty());
        assert!(loaded.build_mount_map().unwrap().is_identity());
    }

    #[test]
    fn load_config_round_trip_with_valid_mounts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cfg_path = tmp.path().join("config.yaml");
        let mut cfg = DaemonConfig::default();
        cfg.mounts = vec![
            YamlMountEntry {
                host: "/Users/chris/dev".to_string(),
                container: "/Users/chris/dev".to_string(),
            },
            YamlMountEntry {
                host: "/Volumes/External/books".to_string(),
                container: "/mnt/external-books".to_string(),
            },
        ];
        std::fs::write(&cfg_path, serde_yaml_ng::to_string(&cfg).unwrap()).unwrap();

        let loaded = load_config(Some(&cfg_path)).expect("valid mounts must load");
        assert_eq!(loaded.mounts.len(), 2);
        let map = loaded.build_mount_map().expect("build_mount_map");
        assert_eq!(map.len(), 2);
        assert!(!map.is_identity());
    }

    // ── AC-a2.4: PURE-DEFAULTS snapshot + AC-a2.2: PARITY ───────────────────

    #[test]
    fn empty_yaml_produces_valid_daemon_config() {
        let yaml: YamlConfig = serde_yaml_ng::from_str("{}").expect("empty YAML parses");
        let config = DaemonConfig::from(&yaml);
        assert!(config.validate().is_ok(), "empty-YAML config must validate");
    }

    #[test]
    fn default_yaml_round_trips_to_daemon_config() {
        let yaml: YamlConfig =
            serde_yaml_ng::from_str(yaml_defaults::DEFAULT_YAML).expect("DEFAULT_YAML parses");
        let from_yaml = DaemonConfig::from(&yaml);
        let from_default = DaemonConfig::default();
        assert_eq!(
            from_yaml.queue_processor.batch_size,
            from_default.queue_processor.batch_size
        );
        assert_eq!(
            from_yaml.resource_limits.nice_level,
            from_default.resource_limits.nice_level
        );
        assert_eq!(
            from_yaml.embedding.cache_max_entries,
            from_default.embedding.cache_max_entries
        );
        assert!(from_yaml.validate().is_ok());
    }
}
