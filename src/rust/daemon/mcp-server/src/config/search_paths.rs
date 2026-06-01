//! Config file search paths.
//!
//! Mirrors `getConfigSearchPaths()` + `findConfigFile()` from
//! `src/typescript/mcp-server/src/config.ts`.
//!
//! Search order (first existing file wins):
//!   1. `WQM_CONFIG_PATH` environment variable (if set)
//!   2. `<config_dir>/config.yaml`   (where config_dir = WQM_CONFIG_DIR |
//!       XDG_CONFIG_HOME/workspace-qdrant | ~/.config/workspace-qdrant)
//!   3. `<config_dir>/config.yml`

use std::path::PathBuf;

/// Returns the ordered list of candidate config file paths.
///
/// The caller is responsible for checking which one exists (see
/// [`find_config_file`]).  This function is pure and has no I/O side effects.
pub fn config_search_paths(env_getter: &dyn Fn(&str) -> Option<String>) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = Vec::new();

    if let Some(explicit) = env_getter("WQM_CONFIG_PATH") {
        paths.push(PathBuf::from(explicit));
    }

    let config_dir = config_directory_with_getter(env_getter);
    paths.push(config_dir.join("config.yaml"));
    paths.push(config_dir.join("config.yml"));

    paths
}

/// Returns the config directory, using the injected env getter for
/// testability without mutating process-level env.
fn config_directory_with_getter(env_getter: &dyn Fn(&str) -> Option<String>) -> PathBuf {
    if let Some(dir) = env_getter("WQM_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    let base = if let Some(xdg) = env_getter("XDG_CONFIG_HOME") {
        PathBuf::from(xdg)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join(".config")
    };
    base.join("workspace-qdrant")
}

/// Scans the ordered search paths and returns the first path that exists on
/// disk.  Returns `None` when no config file is found.
pub fn find_config_file() -> Option<PathBuf> {
    let env_getter = |key: &str| std::env::var(key).ok();
    find_config_file_with_getter(&env_getter)
}

/// Testable variant: accepts an injected env getter and a path-existence
/// checker so tests do not need real filesystem files.
pub fn find_config_file_with_getter(
    env_getter: &dyn Fn(&str) -> Option<String>,
) -> Option<PathBuf> {
    for path in config_search_paths(env_getter) {
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_env<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    #[test]
    fn no_env_yields_two_default_paths() {
        let getter = make_env(&[]);
        let paths = config_search_paths(&getter);
        // Without WQM_CONFIG_PATH we should have exactly 2 candidates
        assert_eq!(paths.len(), 2);
        let s0 = paths[0].to_string_lossy();
        let s1 = paths[1].to_string_lossy();
        assert!(s0.ends_with("config.yaml"), "first={s0}");
        assert!(s1.ends_with("config.yml"), "second={s1}");
    }

    #[test]
    fn wqm_config_path_is_first() {
        let getter = make_env(&[("WQM_CONFIG_PATH", "/custom/my.yaml")]);
        let paths = config_search_paths(&getter);
        assert_eq!(paths.len(), 3, "explicit + 2 defaults");
        assert_eq!(paths[0], PathBuf::from("/custom/my.yaml"));
    }

    #[test]
    fn wqm_config_dir_overrides_default_dir() {
        let getter = make_env(&[("WQM_CONFIG_DIR", "/my/config/dir")]);
        let paths = config_search_paths(&getter);
        assert_eq!(paths[0], PathBuf::from("/my/config/dir/config.yaml"));
        assert_eq!(paths[1], PathBuf::from("/my/config/dir/config.yml"));
    }

    #[test]
    fn xdg_config_home_used_when_no_wqm_config_dir() {
        let getter = make_env(&[("XDG_CONFIG_HOME", "/xdg/config")]);
        let paths = config_search_paths(&getter);
        assert_eq!(
            paths[0],
            PathBuf::from("/xdg/config/workspace-qdrant/config.yaml")
        );
    }

    #[test]
    fn default_config_dir_contains_workspace_qdrant() {
        let getter = make_env(&[]);
        let paths = config_search_paths(&getter);
        // The default path must include "workspace-qdrant" directory component
        let s = paths[0].to_string_lossy();
        assert!(s.contains("workspace-qdrant"), "path={s}");
    }

    #[test]
    fn both_wqm_config_path_and_dir_set_path_takes_first_slot() {
        let getter = make_env(&[
            ("WQM_CONFIG_PATH", "/explicit/cfg.yaml"),
            ("WQM_CONFIG_DIR", "/override/dir"),
        ]);
        let paths = config_search_paths(&getter);
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0], PathBuf::from("/explicit/cfg.yaml"));
        assert_eq!(paths[1], PathBuf::from("/override/dir/config.yaml"));
        assert_eq!(paths[2], PathBuf::from("/override/dir/config.yml"));
    }
}
