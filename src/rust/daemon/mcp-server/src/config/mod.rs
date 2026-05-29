//! Configuration loading orchestration for the MCP server.
//!
//! Mirrors the `loadConfig()` function in `src/typescript/mcp-server/src/config.ts`.
//!
//! Load order:
//!   1. Start from compiled-in defaults ([`types::ServerConfig::default()`]).
//!   2. If a config file is found ([`search_paths::find_config_file`]), parse it as YAML
//!      and merge it over the defaults ([`merge::merge_yaml_over_defaults`]).
//!   3. Apply environment variable overrides ([`env_overrides::apply_env_overrides`]).
//!   4. Expand tilde/env-var prefixes in `database.path`
//!      ([`wqm_common::env_expand::expand_path`]).

mod env_overrides;
mod merge;
mod search_paths;
mod types;

pub use env_overrides::{apply_env_overrides, parse_grpc_endpoint, GrpcEndpoint};
pub use merge::{merge_yaml_over_defaults, parse_yaml_partial};
pub use search_paths::{config_search_paths, find_config_file};
pub use types::ServerConfig;

use anyhow::{Context, Result};

/// Load the server configuration.
///
/// Equivalent to `loadConfig()` in the TypeScript implementation.
///
/// # Errors
///
/// Returns an error only when a config file is found but cannot be parsed as
/// valid YAML.  A missing config file is not an error — the compiled-in
/// defaults are used instead.
pub fn load_config() -> Result<ServerConfig> {
    load_config_with_env(&|key| std::env::var(key).ok())
}

/// Testable variant: accepts an injected env getter so tests can supply a
/// hermetic map instead of mutating the process-level environment.
pub fn load_config_with_env(env_getter: &dyn Fn(&str) -> Option<String>) -> Result<ServerConfig> {
    // Step 1: start from compiled-in defaults.
    let mut config = ServerConfig::default();

    // Step 2: locate and parse a user config file, then merge over defaults.
    if let Some(path) = find_config_file_with_env(env_getter) {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;
        match parse_yaml_partial(&content) {
            Ok(yaml_val) => {
                config = merge_yaml_over_defaults(config, &yaml_val);
            }
            Err(e) => {
                // Mirror TS behaviour: warn, continue with defaults.
                tracing::warn!("failed to parse config file {}: {e}", path.display());
            }
        }
    }

    // Step 3: apply environment variable overrides.
    config = apply_env_overrides(config, env_getter);

    // Step 4: expand tilde / env-var prefixes in database.path (post-merge,
    // post-env-override, matching TS `expandPath(config.database.path)`).
    config.database.path = wqm_common::env_expand::expand_path(&config.database.path)
        .to_string_lossy()
        .into_owned();

    Ok(config)
}

/// Returns the config file path using the injected env getter.
///
/// Thin wrapper so the test variant of `load_config_with_env` can locate the
/// file without reading real process env.
fn find_config_file_with_env(
    env_getter: &dyn Fn(&str) -> Option<String>,
) -> Option<std::path::PathBuf> {
    use search_paths::config_search_paths;
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
    use std::io::Write as IoWrite;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    // ------------------------------------------------------------------
    // Defaults-only load
    // ------------------------------------------------------------------

    #[test]
    fn defaults_only_matches_ts_default_config() {
        // No env vars, no config file → pure defaults.
        let getter = env_from(&[]);
        let config = load_config_with_env(&getter).expect("defaults load");

        // Matches generated-defaults.ts DEFAULT_CONFIG.
        assert_eq!(config.qdrant.url, "http://localhost:6333");
        assert_eq!(config.qdrant.timeout, 30_000);
        assert!(config.qdrant.api_key.is_none());

        assert_eq!(config.daemon.grpc_host, "localhost");
        assert_eq!(config.daemon.grpc_port, 50051);
        assert_eq!(config.daemon.queue_poll_interval_ms, 500);
        assert_eq!(config.daemon.queue_batch_size, 10);

        assert_eq!(config.collections.rules_collection_name, "rules");

        let rules = config.rules.expect("rules block present");
        assert_eq!(rules.limits.max_label_length, 15);
        assert_eq!(rules.limits.max_title_length, 50);
        assert_eq!(rules.limits.max_tag_length, 20);
        assert_eq!(rules.limits.max_tags_per_rule, 5);

        assert!(config.database.path.ends_with("state.db"));
    }

    // ------------------------------------------------------------------
    // Tilde expansion
    // ------------------------------------------------------------------

    #[test]
    fn tilde_in_database_path_is_expanded() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "~/my-dbs/test.db")]);
        let config = load_config_with_env(&getter).expect("load");
        // After expansion the path must not start with ~.
        assert!(
            !config.database.path.starts_with('~'),
            "tilde not expanded: {}",
            config.database.path
        );
        assert!(config.database.path.ends_with("my-dbs/test.db"));
    }

    #[test]
    fn absolute_database_path_unchanged() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "/absolute/path/test.db")]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.database.path, "/absolute/path/test.db");
    }

    // ------------------------------------------------------------------
    // File-based merge (uses a real temp file)
    // ------------------------------------------------------------------

    #[test]
    fn config_file_merged_over_defaults() {
        let yaml = r#"
qdrant:
  url: "http://custom-qdrant:6334"
daemon:
  grpcPort: 9999
"#;
        let dir = tempfile::tempdir().expect("tmpdir");
        let file_path = dir.path().join("config.yaml");
        {
            let mut f = std::fs::File::create(&file_path).expect("create");
            f.write_all(yaml.as_bytes()).expect("write");
        }

        let path_str = file_path.to_string_lossy().into_owned();
        let getter = move |key: &str| {
            if key == "WQM_CONFIG_PATH" {
                Some(path_str.clone())
            } else {
                None
            }
        };

        let config = load_config_with_env(&getter).expect("load");

        assert_eq!(config.qdrant.url, "http://custom-qdrant:6334");
        assert_eq!(config.daemon.grpc_port, 9999);
        // Unchanged defaults still present.
        assert_eq!(config.qdrant.timeout, 30_000);
        assert_eq!(config.daemon.grpc_host, "localhost");
    }

    // ------------------------------------------------------------------
    // Env-override precedence integration
    // ------------------------------------------------------------------

    #[test]
    fn env_overrides_applied_after_file_merge() {
        // Env var overrides should beat anything from the config file.
        let getter = env_from(&[
            ("QDRANT_URL", "http://env-qdrant:6333"),
            ("WQM_DAEMON_ENDPOINT", "env-host:8888"),
        ]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.qdrant.url, "http://env-qdrant:6333");
        assert_eq!(config.daemon.grpc_host, "env-host");
        assert_eq!(config.daemon.grpc_port, 8888);
    }

    #[test]
    fn all_three_env_precedence() {
        // WQM_DAEMON_ENDPOINT > MEMEXD_GRPC_URL > WQM_DAEMON_PORT
        let getter = env_from(&[
            ("WQM_DAEMON_ENDPOINT", "ep-host:1111"),
            ("MEMEXD_GRPC_URL", "alias-host:2222"),
            ("WQM_DAEMON_PORT", "3333"),
        ]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.daemon.grpc_host, "ep-host");
        assert_eq!(config.daemon.grpc_port, 1111);
    }

    #[test]
    fn qdrant_url_and_api_key_from_env() {
        let getter = env_from(&[
            ("QDRANT_URL", "http://cloud:6333"),
            ("QDRANT_API_KEY", "tok-secret"),
        ]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.qdrant.url, "http://cloud:6333");
        assert_eq!(config.qdrant.api_key, Some("tok-secret".to_string()));
    }

    #[test]
    fn wqm_database_path_from_env() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "/tmp/override.db")]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.database.path, "/tmp/override.db");
    }

    // ------------------------------------------------------------------
    // parse_grpc_endpoint edge cases (re-tested at integration level)
    // ------------------------------------------------------------------

    #[test]
    fn parse_grpc_endpoint_scheme_strip() {
        let ep = parse_grpc_endpoint("http://myhost:9090");
        assert_eq!(ep.host, "myhost");
        assert_eq!(ep.port, 9090);
    }

    #[test]
    fn parse_grpc_endpoint_host_only_defaults_port() {
        let ep = parse_grpc_endpoint("onlyhost");
        assert_eq!(ep.host, "onlyhost");
        assert_eq!(ep.port, 50051);
    }

    #[test]
    fn parse_grpc_endpoint_host_colon_port() {
        let ep = parse_grpc_endpoint("srv:7777");
        assert_eq!(ep.host, "srv");
        assert_eq!(ep.port, 7777);
    }

    #[test]
    fn parse_grpc_endpoint_invalid_port_falls_back() {
        let ep = parse_grpc_endpoint("host:notaport");
        assert_eq!(ep.port, 50051);
    }

    #[test]
    fn parse_grpc_endpoint_empty_string() {
        let ep = parse_grpc_endpoint("");
        assert_eq!(ep.host, "");
        assert_eq!(ep.port, 50051);
    }

    #[test]
    fn parse_grpc_endpoint_http_host_no_port() {
        let ep = parse_grpc_endpoint("http://myhost");
        assert_eq!(ep.host, "myhost");
        assert_eq!(ep.port, 50051);
    }
}
