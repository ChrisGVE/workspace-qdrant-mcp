//! Config-file discovery: which config file the daemon/CLI loads.
//!
//! Distinct from [`super::resolve`] (which resolves *directories*): these
//! functions resolve the ordered list of candidate config-file paths and the
//! first one that exists on disk.

use std::env;
use std::path::PathBuf;

use super::resolve::get_config_dir;

/// Config search paths (priority order). First existing file wins.
pub fn get_config_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(explicit_path) = env::var("WQM_CONFIG_PATH") {
        paths.push(PathBuf::from(explicit_path));
    }

    if let Ok(config_dir) = get_config_dir() {
        paths.push(config_dir.join("config.yaml"));
        paths.push(config_dir.join("config.yml"));
    }

    paths
}

/// Find the first existing config file from the canonical search paths.
pub fn find_config_file() -> Option<PathBuf> {
    get_config_search_paths()
        .into_iter()
        .find(|path| path.exists())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::resolve::ENV_MUTEX;

    #[test]
    fn test_get_config_search_paths_no_env() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_config = env::var("WQM_CONFIG_PATH").ok();
        let prev_dir = env::var("WQM_CONFIG_DIR").ok();
        env::remove_var("WQM_CONFIG_PATH");
        env::remove_var("WQM_CONFIG_DIR");

        let paths = get_config_search_paths();
        assert!(!paths.is_empty());

        let first = &paths[0];
        assert!(
            first.to_string_lossy().contains("workspace-qdrant"),
            "First path should be under workspace-qdrant: {:?}",
            first
        );

        if let Some(val) = prev_config {
            env::set_var("WQM_CONFIG_PATH", val);
        }
        if let Some(val) = prev_dir {
            env::set_var("WQM_CONFIG_DIR", val);
        }
    }

    #[test]
    fn test_get_config_search_paths_with_env() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_CONFIG_PATH").ok();
        env::set_var("WQM_CONFIG_PATH", "/custom/config.yaml");

        let paths = get_config_search_paths();
        assert_eq!(paths[0], PathBuf::from("/custom/config.yaml"));

        match prev {
            Some(val) => env::set_var("WQM_CONFIG_PATH", val),
            None => env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    #[test]
    fn test_find_config_file_nonexistent() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_CONFIG_PATH").ok();
        env::set_var("WQM_CONFIG_PATH", "/nonexistent/path/config.yaml");

        let _ = find_config_file();

        match prev {
            Some(val) => env::set_var("WQM_CONFIG_PATH", val),
            None => env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    #[test]
    fn test_find_config_file_with_tempfile() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.yaml");
        std::fs::write(&config_path, "# test config").unwrap();

        let prev = env::var("WQM_CONFIG_PATH").ok();
        env::set_var("WQM_CONFIG_PATH", config_path.to_str().unwrap());

        let found = find_config_file();
        assert_eq!(found.unwrap(), config_path);

        match prev {
            Some(val) => env::set_var("WQM_CONFIG_PATH", val),
            None => env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    /// Config paths must never contain the legacy `~/.workspace-qdrant` segment
    /// (pre-XDG layout). All returned paths use XDG base directories.
    #[test]
    fn test_no_legacy_workspace_qdrant_segment_in_config_paths() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_path = env::var("WQM_CONFIG_PATH").ok();
        let prev_dir = env::var("WQM_CONFIG_DIR").ok();
        let prev_xdg = env::var("XDG_CONFIG_HOME").ok();
        env::remove_var("WQM_CONFIG_PATH");
        env::remove_var("WQM_CONFIG_DIR");
        env::remove_var("XDG_CONFIG_HOME");

        for p in get_config_search_paths() {
            let s = p.to_string_lossy();
            // A legacy path would look like /home/user/.workspace-qdrant/...
            // The new paths look like /home/user/.config/workspace-qdrant/...
            assert!(
                !s.contains("/.workspace-qdrant/"),
                "Config search path must not use legacy segment: {s}"
            );
        }

        if let Some(v) = prev_path {
            env::set_var("WQM_CONFIG_PATH", v);
        }
        if let Some(v) = prev_dir {
            env::set_var("WQM_CONFIG_DIR", v);
        }
        if let Some(v) = prev_xdg {
            env::set_var("XDG_CONFIG_HOME", v);
        }
    }
}
