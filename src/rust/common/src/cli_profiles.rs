//! CLI connection profiles.
//!
//! Profiles describe where `wqm` should talk to: daemon gRPC address, Qdrant
//! URL, optional API-key env var, and the SQLite state path. Stored in a
//! dedicated TOML file (`cli-config.toml`) separate from the daemon's YAML
//! config, so profile switching does not move the daemon's state.
//!
//! Canonical path cascade (highest priority first):
//!
//! 1. `WQM_CLI_CONFIG` — explicit override.
//! 2. `$XDG_CONFIG_HOME/wqm/cli-config.toml` (default on Linux/macOS is
//!    `~/.config/wqm/cli-config.toml`).
//! 3. `~/.workspace-qdrant/cli-config.toml` — legacy fallback.
//!
//! The first existing file wins; if none exists, write-on-demand targets #2.
//!
//! Two profiles ship out of the box:
//!
//! - `native`: talks to a locally installed daemon (`localhost:50051`) and
//!   Qdrant (`http://localhost:6333`), reads state from
//!   `~/.workspace-qdrant/state.db`.
//! - `docker-local`: same endpoints (containers publish the same ports), but
//!   SQLite defaults to the reference-compose bind mount
//!   `~/.workspace-qdrant/docker/state.db`.
//!
//! Users may add `custom` profiles by hand — the file format is intentionally
//! small and editable.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::constants::{DEFAULT_GRPC_PORT, DEFAULT_QDRANT_URL};

const CLI_CONFIG_FILENAME: &str = "cli-config.toml";

/// Built-in profile name: local native install.
pub const PROFILE_NATIVE: &str = "native";

/// Built-in profile name: reference compose stack on the local machine.
pub const PROFILE_DOCKER_LOCAL: &str = "docker-local";

/// Errors emitted while loading or saving a CLI config file.
#[derive(Debug, thiserror::Error)]
pub enum CliProfileError {
    #[error("could not determine home directory")]
    NoHomeDirectory,

    #[error("cli-config.toml not found at {path}")]
    NotFound { path: PathBuf },

    #[error("profile {name:?} not found; known profiles: {known:?}")]
    UnknownProfile { name: String, known: Vec<String> },

    #[error("cli-config.toml at {path} is invalid: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },

    #[error("cli-config.toml at {path} could not be serialized: {source}")]
    Serialize {
        path: PathBuf,
        #[source]
        source: toml::ser::Error,
    },

    #[error("I/O error for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error("profile {name:?} is invalid: {reason}")]
    Invalid { name: String, reason: String },
}

/// A single connection profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// Stable identifier used by `wqm config use <name>`.
    pub name: String,

    /// Human-friendly explanation shown in `wqm config list`.
    #[serde(default)]
    pub description: String,

    /// Daemon gRPC endpoint (e.g. `http://127.0.0.1:50051`).
    pub daemon_address: String,

    /// Qdrant HTTP base URL (e.g. `http://localhost:6333`).
    pub qdrant_url: String,

    /// Environment variable containing the Qdrant API key, if any.
    ///
    /// Left empty for public-dev deployments. The CLI resolves this lazily so
    /// the secret never lands in `cli-config.toml`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub qdrant_api_key_env: String,

    /// SQLite state database path, as surfaced to the CLI. Empty = use the
    /// daemon default (`WQM_DATABASE_PATH` / `~/.workspace-qdrant/state.db`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub database_path: String,
}

impl Profile {
    /// Validate that required fields are present and shaped correctly.
    pub fn validate(&self) -> Result<(), CliProfileError> {
        let invalid = |reason: &str| CliProfileError::Invalid {
            name: self.name.clone(),
            reason: reason.to_string(),
        };

        if self.name.trim().is_empty() {
            return Err(invalid("profile name must not be empty"));
        }

        if self.name.chars().any(char::is_whitespace) {
            return Err(invalid("profile name must not contain whitespace"));
        }

        if !self.daemon_address.starts_with("http://")
            && !self.daemon_address.starts_with("https://")
        {
            return Err(invalid(
                "daemon_address must start with http:// or https://",
            ));
        }

        if !self.qdrant_url.starts_with("http://") && !self.qdrant_url.starts_with("https://") {
            return Err(invalid("qdrant_url must start with http:// or https://"));
        }

        Ok(())
    }

    /// Built-in native-install profile.
    pub fn native() -> Self {
        Self {
            name: PROFILE_NATIVE.to_string(),
            description: "Local native install (daemon + Qdrant on host)".to_string(),
            daemon_address: format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT),
            qdrant_url: DEFAULT_QDRANT_URL.to_string(),
            qdrant_api_key_env: String::new(),
            database_path: String::new(),
        }
    }

    /// Built-in docker-compose profile (endpoints published on host).
    pub fn docker_local() -> Self {
        Self {
            name: PROFILE_DOCKER_LOCAL.to_string(),
            description: "Reference compose stack on localhost".to_string(),
            daemon_address: format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT),
            qdrant_url: DEFAULT_QDRANT_URL.to_string(),
            qdrant_api_key_env: String::new(),
            database_path: String::new(),
        }
    }
}

/// Serialized CLI config file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CliConfigFile {
    /// Name of the currently active profile.
    pub active: String,

    /// All known profiles. Order preserved for stable display.
    #[serde(default)]
    pub profiles: Vec<Profile>,
}

impl Default for CliConfigFile {
    fn default() -> Self {
        Self {
            active: PROFILE_NATIVE.to_string(),
            profiles: vec![Profile::native(), Profile::docker_local()],
        }
    }
}

impl CliConfigFile {
    /// Look up a profile by name.
    pub fn find(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Return the currently-active profile, falling back to `native` if the
    /// active field points at an unknown profile.
    pub fn active_profile(&self) -> Profile {
        self.find(&self.active)
            .cloned()
            .unwrap_or_else(Profile::native)
    }

    /// Switch active profile. Errors if the profile is unknown.
    pub fn set_active(&mut self, name: &str) -> Result<(), CliProfileError> {
        if self.find(name).is_some() {
            self.active = name.to_string();
            Ok(())
        } else {
            Err(CliProfileError::UnknownProfile {
                name: name.to_string(),
                known: self.profiles.iter().map(|p| p.name.clone()).collect(),
            })
        }
    }

    /// Run validation on every stored profile.
    pub fn validate(&self) -> Result<(), CliProfileError> {
        for profile in &self.profiles {
            profile.validate()?;
        }
        if self.find(&self.active).is_none() {
            return Err(CliProfileError::UnknownProfile {
                name: self.active.clone(),
                known: self.profiles.iter().map(|p| p.name.clone()).collect(),
            });
        }
        Ok(())
    }
}

/// Canonical path search order for `cli-config.toml`.
pub fn get_cli_config_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    if let Ok(explicit) = std::env::var("WQM_CLI_CONFIG") {
        if !explicit.is_empty() {
            paths.push(PathBuf::from(explicit));
        }
    }

    if let Some(home) = dirs::home_dir() {
        let xdg = std::env::var("XDG_CONFIG_HOME")
            .ok()
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"));
        paths.push(xdg.join("wqm").join(CLI_CONFIG_FILENAME));

        paths.push(home.join(".workspace-qdrant").join(CLI_CONFIG_FILENAME));
    }

    paths
}

/// Return the first existing cli-config.toml path, if any.
pub fn find_cli_config_file() -> Option<PathBuf> {
    get_cli_config_search_paths()
        .into_iter()
        .find(|p| p.exists())
}

/// Default path at which new cli-config.toml files are created.
pub fn default_cli_config_path() -> Result<PathBuf, CliProfileError> {
    if let Ok(explicit) = std::env::var("WQM_CLI_CONFIG") {
        if !explicit.is_empty() {
            return Ok(PathBuf::from(explicit));
        }
    }
    let home = dirs::home_dir().ok_or(CliProfileError::NoHomeDirectory)?;
    let xdg = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".config"));
    Ok(xdg.join("wqm").join(CLI_CONFIG_FILENAME))
}

/// Load cli-config.toml from the active search path. Returns `None` if no
/// file exists; caller decides whether to bootstrap a default.
pub fn load_cli_config() -> Result<Option<(CliConfigFile, PathBuf)>, CliProfileError> {
    let Some(path) = find_cli_config_file() else {
        return Ok(None);
    };
    let file = load_cli_config_from(&path)?;
    Ok(Some((file, path)))
}

/// Load cli-config.toml from a specific path.
pub fn load_cli_config_from(path: &Path) -> Result<CliConfigFile, CliProfileError> {
    let text = fs::read_to_string(path).map_err(|e| CliProfileError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let parsed: CliConfigFile = toml::from_str(&text).map_err(|e| CliProfileError::Parse {
        path: path.to_path_buf(),
        source: e,
    })?;
    parsed.validate()?;
    Ok(parsed)
}

/// Persist `config` to `path`, creating parent directories as needed.
pub fn save_cli_config(path: &Path, config: &CliConfigFile) -> Result<(), CliProfileError> {
    config.validate()?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| CliProfileError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }
    }
    let text = toml::to_string_pretty(config).map_err(|e| CliProfileError::Serialize {
        path: path.to_path_buf(),
        source: e,
    })?;
    fs::write(path, text).map_err(|e| CliProfileError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Load the active cli-config, creating the default file if none exists.
/// Returns the file and the path it lives at.
pub fn ensure_cli_config() -> Result<(CliConfigFile, PathBuf), CliProfileError> {
    if let Some((file, path)) = load_cli_config()? {
        return Ok((file, path));
    }

    let path = default_cli_config_path()?;
    let default = CliConfigFile::default();
    save_cli_config(&path, &default)?;
    Ok((default, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn scrub_env() {
        std::env::remove_var("WQM_CLI_CONFIG");
        std::env::remove_var("XDG_CONFIG_HOME");
    }

    #[test]
    fn default_config_has_both_builtins_and_active_native() {
        let cfg = CliConfigFile::default();
        assert_eq!(cfg.active, PROFILE_NATIVE);
        assert!(cfg.find(PROFILE_NATIVE).is_some());
        assert!(cfg.find(PROFILE_DOCKER_LOCAL).is_some());
        cfg.validate().unwrap();
    }

    #[test]
    fn active_profile_falls_back_to_native() {
        let mut cfg = CliConfigFile::default();
        cfg.active = "ghost".into();
        assert_eq!(cfg.active_profile().name, PROFILE_NATIVE);
    }

    #[test]
    fn set_active_rejects_unknown_profile() {
        let mut cfg = CliConfigFile::default();
        let err = cfg.set_active("ghost").unwrap_err();
        assert!(matches!(err, CliProfileError::UnknownProfile { .. }));
    }

    #[test]
    fn set_active_accepts_known_profile() {
        let mut cfg = CliConfigFile::default();
        cfg.set_active(PROFILE_DOCKER_LOCAL).unwrap();
        assert_eq!(cfg.active, PROFILE_DOCKER_LOCAL);
    }

    #[test]
    fn profile_validate_rejects_missing_scheme() {
        let profile = Profile {
            name: "custom".into(),
            description: String::new(),
            daemon_address: "127.0.0.1:50051".into(),
            qdrant_url: "http://localhost:6333".into(),
            qdrant_api_key_env: String::new(),
            database_path: String::new(),
        };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn profile_validate_rejects_whitespace_name() {
        let profile = Profile {
            name: "bad name".into(),
            description: String::new(),
            daemon_address: "http://127.0.0.1:50051".into(),
            qdrant_url: "http://localhost:6333".into(),
            qdrant_api_key_env: String::new(),
            database_path: String::new(),
        };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn profile_validate_rejects_bad_qdrant_scheme() {
        let profile = Profile {
            name: "custom".into(),
            description: String::new(),
            daemon_address: "http://127.0.0.1:50051".into(),
            qdrant_url: "localhost:6333".into(),
            qdrant_api_key_env: String::new(),
            database_path: String::new(),
        };
        assert!(profile.validate().is_err());
    }

    #[test]
    fn save_then_load_roundtrip_preserves_fields() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("nested").join(CLI_CONFIG_FILENAME);
        let mut cfg = CliConfigFile::default();
        cfg.set_active(PROFILE_DOCKER_LOCAL).unwrap();

        save_cli_config(&path, &cfg).unwrap();
        assert!(path.exists(), "file should be written");
        let loaded = load_cli_config_from(&path).unwrap();
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn load_rejects_malformed_toml() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join(CLI_CONFIG_FILENAME);
        std::fs::write(&path, "active = \n[profiles]]").unwrap();
        let err = load_cli_config_from(&path).unwrap_err();
        assert!(matches!(err, CliProfileError::Parse { .. }));
    }

    #[test]
    fn load_rejects_unknown_active_profile() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join(CLI_CONFIG_FILENAME);
        let toml_str = r#"
active = "ghost"

[[profiles]]
name = "native"
daemon_address = "http://127.0.0.1:50051"
qdrant_url = "http://127.0.0.1:6333"
"#;
        std::fs::write(&path, toml_str).unwrap();
        let err = load_cli_config_from(&path).unwrap_err();
        assert!(matches!(err, CliProfileError::UnknownProfile { .. }));
    }

    #[test]
    fn search_paths_honor_wqm_cli_config_env() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let prev_cli = std::env::var("WQM_CLI_CONFIG").ok();
        let prev_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        scrub_env();
        std::env::set_var("WQM_CLI_CONFIG", "/tmp/override.toml");

        let paths = get_cli_config_search_paths();
        assert_eq!(paths.first(), Some(&PathBuf::from("/tmp/override.toml")));

        match prev_cli {
            Some(v) => std::env::set_var("WQM_CLI_CONFIG", v),
            None => std::env::remove_var("WQM_CLI_CONFIG"),
        }
        match prev_xdg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    fn default_cli_config_path_uses_xdg_when_set() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let prev_cli = std::env::var("WQM_CLI_CONFIG").ok();
        let prev_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        scrub_env();
        std::env::set_var("XDG_CONFIG_HOME", "/tmp/xdg-home");

        let path = default_cli_config_path().unwrap();
        assert_eq!(
            path,
            PathBuf::from("/tmp/xdg-home")
                .join("wqm")
                .join(CLI_CONFIG_FILENAME)
        );

        match prev_cli {
            Some(v) => std::env::set_var("WQM_CLI_CONFIG", v),
            None => std::env::remove_var("WQM_CLI_CONFIG"),
        }
        match prev_xdg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }

    #[test]
    fn ensure_cli_config_creates_default_when_missing() {
        let _lock = ENV_MUTEX.lock().unwrap();
        let prev_cli = std::env::var("WQM_CLI_CONFIG").ok();
        let prev_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        scrub_env();

        let tmp = tempfile::TempDir::new().unwrap();
        // Point XDG at the tempdir as well so any pre-existing
        // ~/.config/wqm/cli-config.toml on the developer machine does
        // not satisfy find_cli_config_file() before WQM_CLI_CONFIG is
        // checked.
        std::env::set_var("XDG_CONFIG_HOME", tmp.path());
        let path = tmp.path().join("wqm").join(CLI_CONFIG_FILENAME);
        std::env::set_var("WQM_CLI_CONFIG", &path);

        let (cfg, used) = ensure_cli_config().unwrap();
        assert_eq!(used, path);
        assert_eq!(cfg.active, PROFILE_NATIVE);
        assert!(path.exists());

        match prev_cli {
            Some(v) => std::env::set_var("WQM_CLI_CONFIG", v),
            None => std::env::remove_var("WQM_CLI_CONFIG"),
        }
        match prev_xdg {
            Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
            None => std::env::remove_var("XDG_CONFIG_HOME"),
        }
    }
}
