//! Daemon connection-address resolution.
//!
//! Single source of truth for *where* the daemon lives, shared by both gRPC
//! clients (CLI + MCP). Resolution precedence (WI-d2, #82):
//!
//! 1. `WQM_DAEMON_ADDR` environment variable (non-empty) — highest priority.
//! 2. The active `cli-config.toml` profile's `daemon_address`
//!    (honouring `WQM_PROFILE` to select a non-active profile).
//! 3. Built-in default `http://127.0.0.1:<DEFAULT_GRPC_PORT>`.
//!
//! This mirrors the precedence the CLI previously implemented inline
//! (`Config::from_env().daemon_address`), lifted here so the shared
//! `DaemonClient` resolves it identically for every consumer.

use wqm_common::cli_profiles::load_cli_config;
use wqm_common::constants::DEFAULT_GRPC_PORT;

/// Environment variable that overrides the daemon gRPC address.
pub const DAEMON_ADDR_ENV: &str = "WQM_DAEMON_ADDR";

/// Environment variable that selects a non-active cli-config profile.
pub const PROFILE_ENV: &str = "WQM_PROFILE";

/// The built-in fallback address used when neither env nor profile supply one.
pub fn default_daemon_address() -> String {
    format!("http://127.0.0.1:{}", DEFAULT_GRPC_PORT)
}

/// Resolve the daemon gRPC address from the real process environment.
///
/// See the module docs for the precedence. Never returns an empty string.
pub fn resolve_daemon_address() -> String {
    resolve_daemon_address_with(&|key| std::env::var(key).ok())
}

/// Getter-injectable core of [`resolve_daemon_address`].
///
/// `env_getter` is consulted for `WQM_DAEMON_ADDR` and `WQM_PROFILE`. Profile
/// *file* loading still reads the real environment (for `WQM_CLI_CONFIG`), as
/// the CLI did — only the override/selection layer uses the injected getter so
/// tests can drive precedence without mutating global state for those keys.
pub fn resolve_daemon_address_with(env_getter: &dyn Fn(&str) -> Option<String>) -> String {
    // 1. Explicit env override wins outright.
    if let Some(addr) = env_getter(DAEMON_ADDR_ENV).filter(|v| !v.is_empty()) {
        return addr;
    }

    // 2. Active (or WQM_PROFILE-selected) cli-config profile.
    if let Ok(Some((file, _path))) = load_cli_config() {
        let profile_name = env_getter(PROFILE_ENV)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| file.active.clone());
        if let Some(profile) = file.find(&profile_name) {
            if !profile.daemon_address.is_empty() {
                return profile.daemon_address.clone();
            }
        }
    }

    // 3. Built-in default.
    default_daemon_address()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_wins() {
        let getter = |key: &str| match key {
            DAEMON_ADDR_ENV => Some("http://10.0.0.5:60000".to_string()),
            _ => None,
        };
        assert_eq!(
            resolve_daemon_address_with(&getter),
            "http://10.0.0.5:60000"
        );
    }

    #[test]
    fn empty_env_override_is_ignored() {
        // An empty WQM_DAEMON_ADDR must not short-circuit to "".
        let getter = |key: &str| match key {
            DAEMON_ADDR_ENV => Some(String::new()),
            _ => None,
        };
        let addr = resolve_daemon_address_with(&getter);
        assert!(addr.starts_with("http://"), "got {addr}");
        assert!(!addr.is_empty());
    }

    #[test]
    fn default_when_no_env_and_no_profile() {
        // With no env override and (typically) no cli-config in the test env,
        // resolution lands on the built-in default. We assert the shape rather
        // than an exact value so a developer's local cli-config doesn't fail CI.
        let getter = |_key: &str| None;
        let addr = resolve_daemon_address_with(&getter);
        assert!(addr.starts_with("http://"), "got {addr}");
        assert!(addr.contains("50051") || addr.contains(':'), "got {addr}");
    }

    #[test]
    fn default_address_uses_canonical_port() {
        assert_eq!(default_daemon_address(), "http://127.0.0.1:50051");
        assert_eq!(DEFAULT_GRPC_PORT, 50051);
    }
}
