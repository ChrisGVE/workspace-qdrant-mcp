//! XDG-compliant path resolution and canonical-path abstraction for
//! workspace-qdrant.
//!
//! This module has two distinct concerns:
//!
//! 1. **Process-local directory layout** — XDG-compliant resolution of config,
//!    data, cache, and log directories. The free functions [`get_config_dir`],
//!    [`get_data_dir`], etc. are the single source of truth for filesystem
//!    layout. Every component (daemon, CLI, MCP server) MUST use these
//!    functions instead of hardcoding directory names.
//!
//! 2. **Canonical path abstraction** — newtypes [`CanonicalPath`],
//!    [`LocalPath`], and [`MountMap`] enforce a type-system discipline that
//!    keeps stored/transmitted paths in a single deployment-independent form
//!    and translates to process-local paths only at the filesystem I/O
//!    boundary. See `docs/specs/16-path-abstraction.md` for the full design.
//!
//! XDG layout (macOS, no env overrides):
//!   Config: ~/.config/workspace-qdrant/           (config.yaml, cli-config.toml)
//!   Data:   ~/.local/share/workspace-qdrant/      (state.db, search.db, graph.db)
//!   Cache:  ~/.cache/workspace-qdrant/            (grammars/, models/)
//!   Logs:   ~/Library/Logs/workspace-qdrant/
//!
//! Environment overrides (highest priority):
//!   WQM_CONFIG_PATH  — explicit config file path
//!   WQM_CONFIG_DIR   — config directory
//!   WQM_DATA_DIR     — data directory
//!   WQM_CACHE_DIR    — cache directory
//!   WQM_DATABASE_PATH — explicit database file path
//!   WQM_LOG_DIR      — log directory
//!
//! XDG variables ($XDG_CONFIG_HOME, $XDG_DATA_HOME, $XDG_CACHE_HOME) are
//! respected on all platforms.

mod canonical;
mod error;
mod local;
mod mount_map;
mod normalize;
mod relative;

#[cfg(test)]
mod tests;

pub use canonical::CanonicalPath;
pub use error::PathError;
pub use local::LocalPath;
pub use mount_map::{mount_section_hash, MountEntry, MountMap};
pub use relative::{RelativePath, RelativePathError};

use std::env;
use std::path::PathBuf;

const DIR_NAME: &str = "workspace-qdrant";

#[derive(Debug, thiserror::Error)]
pub enum ConfigPathError {
    #[error("could not determine home directory")]
    NoHomeDirectory,

    #[error("database not found at {path}; run daemon first: wqm service start")]
    DatabaseNotFound { path: PathBuf },
}

// ---------------------------------------------------------------------------
// Core directory functions
// ---------------------------------------------------------------------------

/// Config directory: settings files that the user edits.
///
/// Precedence: `WQM_CONFIG_DIR` > `XDG_CONFIG_HOME` > `~/.config`
pub fn get_config_dir() -> Result<PathBuf, ConfigPathError> {
    if let Ok(dir) = env::var("WQM_CONFIG_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let base = env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".config"))
                .ok_or(ConfigPathError::NoHomeDirectory)
        })?;
    Ok(base.join(DIR_NAME))
}

/// Data directory: databases and runtime state the daemon owns.
///
/// Precedence: `WQM_DATA_DIR` > `XDG_DATA_HOME` > `~/.local/share`
pub fn get_data_dir() -> Result<PathBuf, ConfigPathError> {
    if let Ok(dir) = env::var("WQM_DATA_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let base = env::var("XDG_DATA_HOME").map(PathBuf::from).or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join(".local").join("share"))
            .ok_or(ConfigPathError::NoHomeDirectory)
    })?;
    Ok(base.join(DIR_NAME))
}

/// Cache directory: re-downloadable artifacts (grammars, models).
///
/// Precedence: `WQM_CACHE_DIR` > `XDG_CACHE_HOME` > `~/.cache`
pub fn get_cache_dir() -> Result<PathBuf, ConfigPathError> {
    if let Ok(dir) = env::var("WQM_CACHE_DIR") {
        return Ok(PathBuf::from(dir));
    }
    let base = env::var("XDG_CACHE_HOME").map(PathBuf::from).or_else(|_| {
        dirs::home_dir()
            .map(|h| h.join(".cache"))
            .ok_or(ConfigPathError::NoHomeDirectory)
    })?;
    Ok(base.join(DIR_NAME))
}

// ---------------------------------------------------------------------------
// Config file resolution
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Database paths
// ---------------------------------------------------------------------------

/// Canonical state database path.
///
/// Precedence: `WQM_DATABASE_PATH` > `<data_dir>/state.db`
pub fn get_database_path() -> Result<PathBuf, ConfigPathError> {
    if let Ok(path) = env::var("WQM_DATABASE_PATH") {
        return Ok(PathBuf::from(path));
    }
    get_data_dir().map(|d| d.join("state.db"))
}

/// Database path, verified to exist.
pub fn get_database_path_checked() -> Result<PathBuf, ConfigPathError> {
    let path = get_database_path()?;
    if !path.exists() {
        return Err(ConfigPathError::DatabaseNotFound { path });
    }
    Ok(path)
}

// ---------------------------------------------------------------------------
// Log directory
// ---------------------------------------------------------------------------

/// Canonical log directory.
///
/// Precedence: `WQM_LOG_DIR` > platform-specific default.
///   - macOS: `~/Library/Logs/workspace-qdrant/`
///   - Linux: `$XDG_STATE_HOME/workspace-qdrant/logs/`
pub fn get_canonical_log_dir() -> PathBuf {
    if let Ok(custom_dir) = env::var("WQM_LOG_DIR") {
        return PathBuf::from(custom_dir);
    }

    #[cfg(target_os = "linux")]
    {
        env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(env::temp_dir)
                    .join(".local")
                    .join("state")
            })
            .join(DIR_NAME)
            .join("logs")
    }

    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_else(env::temp_dir)
            .join("Library")
            .join("Logs")
            .join(DIR_NAME)
    }

    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(env::temp_dir)
            .join(DIR_NAME)
            .join("logs")
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        dirs::home_dir()
            .unwrap_or_else(env::temp_dir)
            .join(".local")
            .join("share")
            .join(DIR_NAME)
            .join("logs")
    }
}

// ---------------------------------------------------------------------------
// Convenience helpers
// ---------------------------------------------------------------------------

/// Grammar cache directory (inside cache dir).
pub fn get_grammar_cache_dir() -> Result<PathBuf, ConfigPathError> {
    get_cache_dir().map(|d| d.join("grammars"))
}

/// Model cache directory (inside cache dir).
pub fn get_model_cache_dir() -> Result<PathBuf, ConfigPathError> {
    get_cache_dir().map(|d| d.join("models"))
}

#[cfg(test)]
mod xdg_tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

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

    #[test]
    fn test_get_config_dir() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_CONFIG_DIR").ok();
        env::remove_var("WQM_CONFIG_DIR");

        let dir = get_config_dir();
        assert!(dir.is_ok());
        let path = dir.unwrap();
        assert!(path.to_string_lossy().contains("workspace-qdrant"));

        if let Some(val) = prev {
            env::set_var("WQM_CONFIG_DIR", val);
        }
    }

    #[test]
    fn test_get_data_dir() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_DATA_DIR").ok();
        env::remove_var("WQM_DATA_DIR");

        let dir = get_data_dir();
        assert!(dir.is_ok());
        let path = dir.unwrap();
        assert!(path.to_string_lossy().contains("workspace-qdrant"));

        if let Some(val) = prev {
            env::set_var("WQM_DATA_DIR", val);
        }
    }

    #[test]
    fn test_get_cache_dir() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_CACHE_DIR").ok();
        env::remove_var("WQM_CACHE_DIR");

        let dir = get_cache_dir();
        assert!(dir.is_ok());
        let path = dir.unwrap();
        assert!(path.to_string_lossy().contains("workspace-qdrant"));

        if let Some(val) = prev {
            env::set_var("WQM_CACHE_DIR", val);
        }
    }

    #[test]
    fn test_get_database_path() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_DATABASE_PATH").ok();
        env::remove_var("WQM_DATABASE_PATH");

        let result = get_database_path();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains("workspace-qdrant"));
        assert!(path.to_string_lossy().ends_with("state.db"));

        if let Some(val) = prev {
            env::set_var("WQM_DATABASE_PATH", val);
        }
    }

    #[test]
    fn test_get_database_path_env_override() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_DATABASE_PATH").ok();
        env::set_var("WQM_DATABASE_PATH", "/custom/path/state.db");

        let result = get_database_path();
        assert_eq!(result.unwrap(), PathBuf::from("/custom/path/state.db"));

        match prev {
            Some(val) => env::set_var("WQM_DATABASE_PATH", val),
            None => env::remove_var("WQM_DATABASE_PATH"),
        }
    }

    #[test]
    fn test_get_database_path_checked_missing() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_DATABASE_PATH").ok();
        env::set_var(
            "WQM_DATABASE_PATH",
            "/nonexistent/path/that/does/not/exist/state.db",
        );

        let result = get_database_path_checked();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("run daemon first"));

        match prev {
            Some(val) => env::set_var("WQM_DATABASE_PATH", val),
            None => env::remove_var("WQM_DATABASE_PATH"),
        }
    }

    #[test]
    fn test_config_path_error_display() {
        let err = ConfigPathError::NoHomeDirectory;
        assert_eq!(err.to_string(), "could not determine home directory");

        let err = ConfigPathError::DatabaseNotFound {
            path: PathBuf::from("/test/state.db"),
        };
        assert!(err.to_string().contains("run daemon first"));
    }

    #[test]
    fn test_get_canonical_log_dir_env_override() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_LOG_DIR").ok();
        env::set_var("WQM_LOG_DIR", "/custom/log/dir");

        let dir = get_canonical_log_dir();
        assert_eq!(dir, PathBuf::from("/custom/log/dir"));

        match prev {
            Some(val) => env::set_var("WQM_LOG_DIR", val),
            None => env::remove_var("WQM_LOG_DIR"),
        }
    }

    #[test]
    fn test_get_canonical_log_dir_default() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_LOG_DIR").ok();
        env::remove_var("WQM_LOG_DIR");

        let dir = get_canonical_log_dir();
        let dir_str = dir.to_string_lossy();

        assert!(
            dir_str.contains("workspace-qdrant"),
            "Log dir should contain workspace-qdrant: {:?}",
            dir
        );

        #[cfg(target_os = "macos")]
        assert!(
            dir_str.contains("Library/Logs"),
            "macOS log dir should be under Library/Logs: {:?}",
            dir
        );

        match prev {
            Some(val) => env::set_var("WQM_LOG_DIR", val),
            None => env::remove_var("WQM_LOG_DIR"),
        }
    }

    #[test]
    fn test_env_overrides() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_config = env::var("WQM_CONFIG_DIR").ok();
        let prev_data = env::var("WQM_DATA_DIR").ok();
        let prev_cache = env::var("WQM_CACHE_DIR").ok();

        env::set_var("WQM_CONFIG_DIR", "/override/config");
        env::set_var("WQM_DATA_DIR", "/override/data");
        env::set_var("WQM_CACHE_DIR", "/override/cache");

        assert_eq!(get_config_dir().unwrap(), PathBuf::from("/override/config"));
        assert_eq!(get_data_dir().unwrap(), PathBuf::from("/override/data"));
        assert_eq!(get_cache_dir().unwrap(), PathBuf::from("/override/cache"));

        match prev_config {
            Some(val) => env::set_var("WQM_CONFIG_DIR", val),
            None => env::remove_var("WQM_CONFIG_DIR"),
        }
        match prev_data {
            Some(val) => env::set_var("WQM_DATA_DIR", val),
            None => env::remove_var("WQM_DATA_DIR"),
        }
        match prev_cache {
            Some(val) => env::set_var("WQM_CACHE_DIR", val),
            None => env::remove_var("WQM_CACHE_DIR"),
        }
    }

    #[test]
    fn test_grammar_and_model_cache_dirs() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev = env::var("WQM_CACHE_DIR").ok();
        env::set_var("WQM_CACHE_DIR", "/test/cache");

        assert_eq!(
            get_grammar_cache_dir().unwrap(),
            PathBuf::from("/test/cache/grammars")
        );
        assert_eq!(
            get_model_cache_dir().unwrap(),
            PathBuf::from("/test/cache/models")
        );

        match prev {
            Some(val) => env::set_var("WQM_CACHE_DIR", val),
            None => env::remove_var("WQM_CACHE_DIR"),
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

    /// Data dir must not contain the legacy `~/.workspace-qdrant` root segment.
    #[test]
    fn test_no_legacy_workspace_qdrant_segment_in_data_dir() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_data = env::var("WQM_DATA_DIR").ok();
        let prev_xdg = env::var("XDG_DATA_HOME").ok();
        env::remove_var("WQM_DATA_DIR");
        env::remove_var("XDG_DATA_HOME");

        let path = get_data_dir().unwrap();
        let s = path.to_string_lossy();
        assert!(
            !s.contains("/.workspace-qdrant/") && !s.ends_with("/.workspace-qdrant"),
            "Data dir must not use legacy segment: {s}"
        );
        // Must use .local/share layout
        assert!(
            s.contains(".local/share") || s.contains("XDG"),
            "Data dir should be under .local/share (or XDG override): {s}"
        );

        if let Some(v) = prev_data {
            env::set_var("WQM_DATA_DIR", v);
        }
        if let Some(v) = prev_xdg {
            env::set_var("XDG_DATA_HOME", v);
        }
    }

    /// Database path must not be the legacy `~/.workspace-qdrant/state.db`.
    #[test]
    fn test_no_legacy_workspace_qdrant_segment_in_database_path() {
        let _lock = ENV_MUTEX.lock().unwrap();

        let prev_db = env::var("WQM_DATABASE_PATH").ok();
        let prev_data = env::var("WQM_DATA_DIR").ok();
        let prev_xdg = env::var("XDG_DATA_HOME").ok();
        env::remove_var("WQM_DATABASE_PATH");
        env::remove_var("WQM_DATA_DIR");
        env::remove_var("XDG_DATA_HOME");

        let path = get_database_path().unwrap();
        let s = path.to_string_lossy();
        assert!(
            !s.contains("/.workspace-qdrant/state.db") || s.contains(".local/share"),
            "Database path must not be the legacy ~/.workspace-qdrant/state.db: {s}"
        );

        if let Some(v) = prev_db {
            env::set_var("WQM_DATABASE_PATH", v);
        }
        if let Some(v) = prev_data {
            env::set_var("WQM_DATA_DIR", v);
        }
        if let Some(v) = prev_xdg {
            env::set_var("XDG_DATA_HOME", v);
        }
    }
}
