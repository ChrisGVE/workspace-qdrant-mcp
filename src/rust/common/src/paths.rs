//! Canonical path resolution for configuration, database, and config directory
//!
//! This module provides the ONE canonical config search cascade used by all
//! components (daemon, CLI, MCP server). No project-local config is searched.
//!
//! Config precedence:
//! 1. `WQM_CONFIG_PATH` environment variable (explicit override)
//! 2. `~/.workspace-qdrant/config.yaml` (user home)
//! 3. `$XDG_CONFIG_HOME/workspace-qdrant/config.yaml` (XDG; defaults to `~/.config`)
//! 4. `~/Library/Application Support/workspace-qdrant/config.yaml` (macOS only)

use std::env;
use std::path::PathBuf;

/// Error type for path resolution failures
#[derive(Debug, thiserror::Error)]
pub enum ConfigPathError {
    #[error("could not determine home directory")]
    NoHomeDirectory,

    #[error("database not found at {path}; run daemon first: wqm service start")]
    DatabaseNotFound { path: PathBuf },
}

/// Get the canonical config search paths (in priority order).
///
/// Returns all potential config file paths that should be checked.
/// The first existing file wins.
///
/// No project-local `.workspace-qdrant.yaml` is searched.
pub fn get_config_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // 1. Explicit path via environment variable (highest priority)
    if let Ok(explicit_path) = std::env::var("WQM_CONFIG_PATH") {
        paths.push(PathBuf::from(explicit_path));
    }

    // 2. User home config: ~/.workspace-qdrant/config.yaml
    if let Some(home) = dirs::home_dir() {
        paths.push(home.join(".workspace-qdrant").join("config.yaml"));
        paths.push(home.join(".workspace-qdrant").join("config.yml"));
    }

    // 3. XDG config: $XDG_CONFIG_HOME/workspace-qdrant/config.yaml
    //    On macOS, dirs::config_dir() returns ~/Library/Application Support,
    //    so we manually check for ~/.config as the XDG equivalent.
    #[cfg(target_os = "macos")]
    if let Some(home) = dirs::home_dir() {
        let xdg_dir = std::env::var("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(".config"));
        paths.push(xdg_dir.join("workspace-qdrant").join("config.yaml"));
        paths.push(xdg_dir.join("workspace-qdrant").join("config.yml"));
    }

    #[cfg(not(target_os = "macos"))]
    if let Some(config_dir) = dirs::config_dir() {
        paths.push(config_dir.join("workspace-qdrant").join("config.yaml"));
        paths.push(config_dir.join("workspace-qdrant").join("config.yml"));
    }

    // 4. macOS Application Support
    #[cfg(target_os = "macos")]
    if let Some(home) = dirs::home_dir() {
        paths.push(
            home.join("Library")
                .join("Application Support")
                .join("workspace-qdrant")
                .join("config.yaml"),
        );
    }

    paths
}

/// Find the first existing config file from the canonical search paths.
pub fn find_config_file() -> Option<PathBuf> {
    get_config_search_paths()
        .into_iter()
        .find(|path| path.exists())
}

/// Get the config directory path (`~/.workspace-qdrant/`).
pub fn get_config_dir() -> Result<PathBuf, ConfigPathError> {
    dirs::home_dir()
        .map(|home| home.join(".workspace-qdrant"))
        .ok_or(ConfigPathError::NoHomeDirectory)
}

/// Get the canonical database path (`~/.workspace-qdrant/state.db`).
///
/// Checks `WQM_DATABASE_PATH` environment variable first, then falls back
/// to the canonical path.
pub fn get_database_path() -> Result<PathBuf, ConfigPathError> {
    if let Ok(path) = std::env::var("WQM_DATABASE_PATH") {
        return Ok(PathBuf::from(path));
    }

    dirs::home_dir()
        .map(|home| home.join(".workspace-qdrant").join("state.db"))
        .ok_or(ConfigPathError::NoHomeDirectory)
}

/// Get the database path, checking if it exists.
///
/// Returns an error with a helpful message if the database doesn't exist,
/// indicating the user should start the daemon first.
pub fn get_database_path_checked() -> Result<PathBuf, ConfigPathError> {
    let path = get_database_path()?;

    if !path.exists() {
        return Err(ConfigPathError::DatabaseNotFound { path });
    }

    Ok(path)
}

/// Returns the canonical OS-specific log directory for workspace-qdrant logs.
///
/// Precedence:
/// 1. `WQM_LOG_DIR` environment variable (explicit override)
/// 2. Platform-specific default:
///    - Linux: `$XDG_STATE_HOME/workspace-qdrant/logs/` (default: `~/.local/state/workspace-qdrant/logs/`)
///    - macOS: `~/Library/Logs/workspace-qdrant/`
///    - Windows: `%LOCALAPPDATA%\workspace-qdrant\logs\`
///
/// Falls back to a temp directory if home cannot be determined.
pub fn get_canonical_log_dir() -> PathBuf {
    // WQM_LOG_DIR takes highest precedence
    if let Ok(custom_dir) = env::var("WQM_LOG_DIR") {
        return PathBuf::from(custom_dir);
    }

    #[cfg(target_os = "linux")]
    {
        env::var("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                dirs::home_dir()
                    .unwrap_or_else(|| env::temp_dir())
                    .join(".local")
                    .join("state")
            })
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(target_os = "macos")]
    {
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("Library")
            .join("Logs")
            .join("workspace-qdrant")
    }

    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join("workspace-qdrant")
            .join("logs")
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        dirs::home_dir()
            .unwrap_or_else(|| env::temp_dir())
            .join(".workspace-qdrant")
            .join("logs")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_config_search_paths_no_env() {
        // Remove env override to test default behavior
        let prev = std::env::var("WQM_CONFIG_PATH").ok();
        std::env::remove_var("WQM_CONFIG_PATH");

        let paths = get_config_search_paths();

        // Should have at least the home config path
        assert!(!paths.is_empty());

        // First path should be under ~/.workspace-qdrant/
        let first = &paths[0];
        assert!(
            first.to_string_lossy().contains(".workspace-qdrant"),
            "First path should be under .workspace-qdrant: {:?}",
            first
        );

        // No paths should reference project-local config
        for p in &paths {
            let s = p.to_string_lossy();
            assert!(
                !s.ends_with(".workspace-qdrant.yaml")
                    && !s.ends_with(".workspace-qdrant.yml"),
                "Should not search project-local config: {:?}",
                p
            );
        }

        // Restore
        if let Some(val) = prev {
            std::env::set_var("WQM_CONFIG_PATH", val);
        }
    }

    #[test]
    fn test_get_config_search_paths_with_env() {
        let prev = std::env::var("WQM_CONFIG_PATH").ok();
        std::env::set_var("WQM_CONFIG_PATH", "/custom/config.yaml");

        let paths = get_config_search_paths();
        assert_eq!(paths[0], PathBuf::from("/custom/config.yaml"));

        // Restore
        match prev {
            Some(val) => std::env::set_var("WQM_CONFIG_PATH", val),
            None => std::env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    #[test]
    fn test_find_config_file_nonexistent() {
        let prev = std::env::var("WQM_CONFIG_PATH").ok();
        std::env::set_var("WQM_CONFIG_PATH", "/nonexistent/path/config.yaml");

        // Should not find a nonexistent file at the top priority
        // (but may find real user config at lower priority)
        let _ = find_config_file();

        match prev {
            Some(val) => std::env::set_var("WQM_CONFIG_PATH", val),
            None => std::env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    #[test]
    fn test_find_config_file_with_tempfile() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config_path = temp_dir.path().join("config.yaml");
        std::fs::write(&config_path, "# test config").unwrap();

        let prev = std::env::var("WQM_CONFIG_PATH").ok();
        std::env::set_var("WQM_CONFIG_PATH", config_path.to_str().unwrap());

        let found = find_config_file();
        assert_eq!(found.unwrap(), config_path);

        match prev {
            Some(val) => std::env::set_var("WQM_CONFIG_PATH", val),
            None => std::env::remove_var("WQM_CONFIG_PATH"),
        }
    }

    #[test]
    fn test_get_config_dir() {
        let dir = get_config_dir();
        assert!(dir.is_ok());
        let path = dir.unwrap();
        assert!(path.to_string_lossy().ends_with(".workspace-qdrant"));
    }

    #[test]
    fn test_get_database_path() {
        let prev = std::env::var("WQM_DATABASE_PATH").ok();
        std::env::remove_var("WQM_DATABASE_PATH");

        let result = get_database_path();
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.to_string_lossy().contains(".workspace-qdrant"));
        assert!(path.to_string_lossy().ends_with("state.db"));

        if let Some(val) = prev {
            std::env::set_var("WQM_DATABASE_PATH", val);
        }
    }

    #[test]
    fn test_get_database_path_env_override() {
        let prev = std::env::var("WQM_DATABASE_PATH").ok();
        std::env::set_var("WQM_DATABASE_PATH", "/custom/path/state.db");

        let result = get_database_path();
        assert_eq!(result.unwrap(), PathBuf::from("/custom/path/state.db"));

        match prev {
            Some(val) => std::env::set_var("WQM_DATABASE_PATH", val),
            None => std::env::remove_var("WQM_DATABASE_PATH"),
        }
    }

    #[test]
    fn test_get_database_path_checked_missing() {
        let prev = std::env::var("WQM_DATABASE_PATH").ok();
        std::env::set_var(
            "WQM_DATABASE_PATH",
            "/nonexistent/path/that/does/not/exist/state.db",
        );

        let result = get_database_path_checked();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("run daemon first"));

        match prev {
            Some(val) => std::env::set_var("WQM_DATABASE_PATH", val),
            None => std::env::remove_var("WQM_DATABASE_PATH"),
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
        let prev = std::env::var("WQM_LOG_DIR").ok();
        std::env::set_var("WQM_LOG_DIR", "/custom/log/dir");

        let dir = get_canonical_log_dir();
        assert_eq!(dir, PathBuf::from("/custom/log/dir"));

        match prev {
            Some(val) => std::env::set_var("WQM_LOG_DIR", val),
            None => std::env::remove_var("WQM_LOG_DIR"),
        }
    }

    #[test]
    fn test_get_canonical_log_dir_default() {
        let prev = std::env::var("WQM_LOG_DIR").ok();
        std::env::remove_var("WQM_LOG_DIR");

        let dir = get_canonical_log_dir();
        let dir_str = dir.to_string_lossy();

        // Should contain workspace-qdrant
        assert!(
            dir_str.contains("workspace-qdrant"),
            "Log dir should contain workspace-qdrant: {:?}",
            dir
        );

        // macOS specific check
        #[cfg(target_os = "macos")]
        assert!(
            dir_str.contains("Library/Logs"),
            "macOS log dir should be under Library/Logs: {:?}",
            dir
        );

        match prev {
            Some(val) => std::env::set_var("WQM_LOG_DIR", val),
            None => std::env::remove_var("WQM_LOG_DIR"),
        }
    }
}
