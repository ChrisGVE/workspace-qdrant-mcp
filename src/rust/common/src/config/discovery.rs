//! Config file search-path resolution.
//!
//! Generalizes the per-component search-path logic (MCP `config_search_paths`,
//! daemon auto-discovery, CLI search) into one parametrized resolver. The
//! resolution order is:
//!   1. explicit-path env var (if configured and set)
//!   2. `<config_dir>/<filename>` for each candidate filename
//!
//! where `config_dir` resolves to:
//!   `<config_dir_var>` env > `XDG_CONFIG_HOME/<app_subdir>` >
//!   `~/.config/<app_subdir>`.

use std::path::PathBuf;

/// Environment getter: maps a var name to its value (injected for testability).
pub type EnvGetter<'a> = dyn Fn(&str) -> Option<String> + 'a;

/// Declarative description of where a component's config file lives.
#[derive(Debug, Clone)]
pub struct ConfigDiscovery {
    /// Env var carrying an explicit file path (e.g. `WQM_CONFIG_PATH`).
    /// When set, it is the highest-priority candidate. `None` disables it.
    pub explicit_path_var: Option<String>,
    /// Env var overriding the config directory (e.g. `WQM_CONFIG_DIR`).
    /// `None` disables it; XDG/home fallback still applies.
    pub config_dir_var: Option<String>,
    /// Application subdirectory under the config base (e.g. `workspace-qdrant`).
    pub app_subdir: String,
    /// Candidate filenames tried in order within the config dir
    /// (e.g. `["config.yaml", "config.yml"]`).
    pub filenames: Vec<String>,
}

impl ConfigDiscovery {
    /// Build the ordered list of candidate config paths (pure; no I/O).
    pub fn search_paths(&self, env: &EnvGetter) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Some(var) = &self.explicit_path_var {
            if let Some(explicit) = env(var) {
                paths.push(PathBuf::from(explicit));
            }
        }

        let dir = self.config_directory(env);
        for name in &self.filenames {
            paths.push(dir.join(name));
        }
        paths
    }

    /// Return the first candidate path that exists on disk, or `None`.
    pub fn find_existing(&self, env: &EnvGetter) -> Option<PathBuf> {
        self.search_paths(env).into_iter().find(|p| p.exists())
    }

    /// Resolve the config directory: `<config_dir_var>` > `XDG_CONFIG_HOME` >
    /// `~/.config`, with `app_subdir` appended to the XDG/home base.
    fn config_directory(&self, env: &EnvGetter) -> PathBuf {
        if let Some(var) = &self.config_dir_var {
            if let Some(dir) = env(var) {
                return PathBuf::from(dir);
            }
        }
        let base = if let Some(xdg) = env("XDG_CONFIG_HOME") {
            PathBuf::from(xdg)
        } else {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join(".config")
        };
        base.join(&self.app_subdir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    fn yaml_discovery() -> ConfigDiscovery {
        ConfigDiscovery {
            explicit_path_var: Some("WQM_CONFIG_PATH".into()),
            config_dir_var: Some("WQM_CONFIG_DIR".into()),
            app_subdir: "workspace-qdrant".into(),
            filenames: vec!["config.yaml".into(), "config.yml".into()],
        }
    }

    #[test]
    fn no_env_yields_default_filenames() {
        let paths = yaml_discovery().search_paths(&env_from(&[]));
        assert_eq!(paths.len(), 2);
        assert!(paths[0].to_string_lossy().ends_with("config.yaml"));
        assert!(paths[1].to_string_lossy().ends_with("config.yml"));
        assert!(paths[0].to_string_lossy().contains("workspace-qdrant"));
    }

    #[test]
    fn explicit_path_takes_first_slot() {
        let paths =
            yaml_discovery().search_paths(&env_from(&[("WQM_CONFIG_PATH", "/custom/my.yaml")]));
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/custom/my.yaml"));
    }

    #[test]
    fn config_dir_var_overrides_base() {
        let paths = yaml_discovery().search_paths(&env_from(&[("WQM_CONFIG_DIR", "/my/dir")]));
        assert_eq!(paths[0], PathBuf::from("/my/dir/config.yaml"));
        assert_eq!(paths[1], PathBuf::from("/my/dir/config.yml"));
    }

    #[test]
    fn xdg_config_home_used_when_no_dir_var() {
        let paths = yaml_discovery().search_paths(&env_from(&[("XDG_CONFIG_HOME", "/xdg")]));
        assert_eq!(paths[0], PathBuf::from("/xdg/workspace-qdrant/config.yaml"));
    }

    #[test]
    fn explicit_and_dir_both_set_keep_order() {
        let paths = yaml_discovery().search_paths(&env_from(&[
            ("WQM_CONFIG_PATH", "/explicit/cfg.yaml"),
            ("WQM_CONFIG_DIR", "/override/dir"),
        ]));
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/explicit/cfg.yaml"));
        assert_eq!(paths[1], PathBuf::from("/override/dir/config.yaml"));
        assert_eq!(paths[2], PathBuf::from("/override/dir/config.yml"));
    }

    #[test]
    fn disabled_explicit_var_is_skipped() {
        let disc = ConfigDiscovery {
            explicit_path_var: None,
            ..yaml_discovery()
        };
        let paths = disc.search_paths(&env_from(&[("WQM_CONFIG_PATH", "/ignored.yaml")]));
        // Explicit var disabled → only the two dir candidates.
        assert_eq!(paths.len(), 2);
        assert!(paths[0].to_string_lossy().ends_with("config.yaml"));
    }

    #[test]
    fn toml_filenames_supported() {
        let disc = ConfigDiscovery {
            explicit_path_var: Some("WQM_CONFIG_PATH".into()),
            config_dir_var: Some("WQM_CONFIG_DIR".into()),
            app_subdir: "workspace-qdrant".into(),
            filenames: vec!["config.toml".into()],
        };
        let paths = disc.search_paths(&env_from(&[("WQM_CONFIG_DIR", "/d")]));
        assert_eq!(paths, vec![PathBuf::from("/d/config.toml")]);
    }

    #[test]
    fn find_existing_returns_first_present(/* uses a temp dir */) {
        let dir = tempfile::tempdir().expect("tmpdir");
        let yml = dir.path().join("config.yml");
        std::fs::write(&yml, "x: 1").expect("write");
        let dir_str = dir.path().to_string_lossy().into_owned();
        let disc = yaml_discovery();
        let getter = move |k: &str| {
            if k == "WQM_CONFIG_DIR" {
                Some(dir_str.clone())
            } else {
                None
            }
        };
        // config.yaml absent, config.yml present → returns the .yml.
        assert_eq!(disc.find_existing(&getter), Some(yml));
    }
}
