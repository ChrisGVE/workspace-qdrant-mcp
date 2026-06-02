//! Configuration loading orchestration for the MCP server.
//!
//! Mirrors the `loadConfig()` function in `src/typescript/mcp-server/src/config.ts`.
//!
//! As of WI-a (shared-crate consolidation) the discovery, format parsing,
//! merge-over-defaults, the declarative env-override loop, and TS-faithful path
//! expansion all come from [`wqm_common::config`]. This module keeps only the
//! MCP-specific pieces: the [`ServerConfig`] typed view (in [`types`]) and the
//! component's env-var spec list (in [`apply_env_overrides`]).
//!
//! Load order:
//!   1. Compiled-in defaults ([`ServerConfig::default()`]).
//!   2. First existing config file (YAML) merged over defaults.
//!   3. Environment variable overrides.
//!   4. TS-faithful tilde expansion of `database.path`.

mod types;

pub use types::ServerConfig;

// Re-export the shared helpers the rest of the crate (and the TS↔Rust parity
// corpus in `tests/parity/`) reference, so call sites stay unchanged after the
// machinery moved to wqm-common.
pub use wqm_common::config::{
    expand_path_ts, expand_path_ts_with_home, parse_grpc_endpoint, parse_int_prefix, GrpcEndpoint,
};

use anyhow::{anyhow, Context, Result};
use wqm_common::config::{
    apply_env_overrides as apply_overrides, merge_over_defaults, ConfigDiscovery, ConfigFormat,
    EnvOverride,
};

/// MCP-server config discovery: `WQM_CONFIG_PATH` explicit override, then
/// `<config_dir>/config.{yaml,yml}` (config_dir from `WQM_CONFIG_DIR` >
/// `XDG_CONFIG_HOME/workspace-qdrant` > `~/.config/workspace-qdrant`).
fn discovery() -> ConfigDiscovery {
    ConfigDiscovery {
        explicit_path_var: Some("WQM_CONFIG_PATH".to_string()),
        config_dir_var: Some("WQM_CONFIG_DIR".to_string()),
        app_subdir: "workspace-qdrant".to_string(),
        filenames: vec!["config.yaml".to_string(), "config.yml".to_string()],
    }
}

/// Load the server configuration. Equivalent to `loadConfig()` in TypeScript.
///
/// # Errors
/// Returns an error only when a config file is found but cannot be read; a YAML
/// parse error is logged (warning) and falls back to defaults (TS behaviour).
pub fn load_config() -> Result<ServerConfig> {
    load_config_with_env(&|key| std::env::var(key).ok())
}

/// Testable variant: accepts an injected env getter so tests can supply a
/// hermetic map instead of mutating the process-level environment.
pub fn load_config_with_env(env_getter: &dyn Fn(&str) -> Option<String>) -> Result<ServerConfig> {
    // Step 1: compiled-in defaults.
    let mut config = ServerConfig::default();

    // Step 2: locate + parse a config file, then merge over defaults.
    if let Some(path) = discovery().find_existing(env_getter) {
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read config file: {}", path.display()))?;
        match ConfigFormat::Yaml.parse_to_value(&content) {
            Ok(value) => {
                config = merge_over_defaults(config, &value)
                    .map_err(|e| anyhow!("failed to merge config file {}: {e}", path.display()))?;
            }
            Err(e) => {
                // Mirror TS behaviour: warn, continue with defaults.
                tracing::warn!("failed to parse config file {}: {e}", path.display());
            }
        }
    }

    // Step 3: environment variable overrides.
    config = apply_env_overrides(config, env_getter);

    // Step 4: TS-faithful tilde expansion of database.path (bare leading '~'
    // only; no $VAR / ~user lookup), matching `expandPath` in config.ts.
    config.database.path = expand_path_ts(&config.database.path);

    Ok(config)
}

/// Apply environment-variable overrides to a `ServerConfig`.
///
/// The simple single-target overrides go through the shared declarative engine;
/// the daemon-endpoint precedence (`WQM_DAEMON_ENDPOINT` > `MEMEXD_GRPC_URL` >
/// `WQM_DAEMON_PORT`) keeps its TS-faithful conditional here because the
/// port-only fallback must apply only when no endpoint env var is set.
fn apply_env_overrides(
    mut config: ServerConfig,
    env_getter: &dyn Fn(&str) -> Option<String>,
) -> ServerConfig {
    let specs: Vec<EnvOverride<ServerConfig>> = vec![
        EnvOverride::single("QDRANT_URL", |c: &mut ServerConfig, v| c.qdrant.url = v),
        EnvOverride::single("QDRANT_API_KEY", |c: &mut ServerConfig, v| {
            c.qdrant.api_key = Some(v)
        }),
        EnvOverride::single("WQM_DATABASE_PATH", |c: &mut ServerConfig, v| {
            c.database.path = v
        }),
        EnvOverride::any(
            ["WQM_DAEMON_ENDPOINT", "MEMEXD_GRPC_URL"],
            |c: &mut ServerConfig, v| {
                let ep = parse_grpc_endpoint(&v);
                c.daemon.grpc_host = ep.host;
                c.daemon.grpc_port = ep.port;
            },
        ),
    ];
    apply_overrides(&mut config, env_getter, &specs);

    // Port-only legacy override: applied ONLY when no endpoint env var is set
    // (config.ts:139). parse_int_prefix mirrors JS parseInt(_, 10); TS applies
    // no upper-bound check on this path. Values outside u16 (negative / >65535)
    // cannot be represented by `grpc_port: u16` and are ignored (documented
    // type-level divergence); 0 IS honoured to match TS.
    if env_getter("WQM_DAEMON_ENDPOINT").is_none() && env_getter("MEMEXD_GRPC_URL").is_none() {
        if let Some(port_str) = env_getter("WQM_DAEMON_PORT") {
            if let Some(n) = parse_int_prefix(&port_str) {
                if (0..=65535).contains(&n) {
                    config.daemon.grpc_port = n as u16;
                }
            }
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write as IoWrite;

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    // ── PURE-DEFAULTS snapshot (AC-a2.4) ───────────────────────────────────

    #[test]
    fn defaults_only_matches_ts_default_config() {
        let config = load_config_with_env(&env_from(&[])).expect("defaults load");

        assert_eq!(config.qdrant.url, "http://localhost:6333");
        assert_eq!(config.qdrant.timeout, 30_000);
        assert!(config.qdrant.api_key.is_none());

        assert_eq!(config.daemon.grpc_host, "localhost");
        assert_eq!(config.daemon.grpc_port, 50051);
        assert_eq!(config.daemon.queue_poll_interval_ms, 500);
        assert_eq!(config.daemon.queue_batch_size, 10);

        assert_eq!(config.collections.rules_collection_name, "rules");

        let rules = config.rules.clone().expect("rules block present");
        assert_eq!(rules.limits.max_label_length, 15);
        assert_eq!(rules.limits.max_title_length, 50);
        assert_eq!(rules.limits.max_tag_length, 20);
        assert_eq!(rules.limits.max_tags_per_rule, 5);

        assert!(config.database.path.ends_with("state.db"));
    }

    #[test]
    fn pure_defaults_snapshot_equals_serverconfig_default() {
        // AC-a2.4: a no-file, no-env load is byte-identical to the compiled-in
        // default view (post tilde-expansion of the db path, which the default
        // path — absolute — does not change).
        let mut expected = ServerConfig::default();
        expected.database.path = expand_path_ts(&expected.database.path);
        let loaded = load_config_with_env(&env_from(&[])).expect("load");
        assert_eq!(loaded, expected);
    }

    // ── Tilde / path expansion parity ──────────────────────────────────────

    #[test]
    fn tilde_in_database_path_is_expanded() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "~/my-dbs/test.db")]);
        let config = load_config_with_env(&getter).expect("load");
        assert!(!config.database.path.starts_with('~'));
        assert!(config.database.path.ends_with("my-dbs/test.db"));
    }

    #[test]
    fn absolute_database_path_unchanged() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "/absolute/path/test.db")]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.database.path, "/absolute/path/test.db");
    }

    #[test]
    fn tilde_slash_foo_expanded_to_home_foo() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "~/foo/db.sqlite")]);
        let config = load_config_with_env(&getter).expect("load");
        let home = dirs::home_dir()
            .expect("home")
            .to_string_lossy()
            .into_owned();
        assert_eq!(config.database.path, format!("{home}/foo/db.sqlite"));
    }

    #[test]
    fn dollar_var_in_path_not_expanded() {
        // TS expandPath does NOT expand $VAR — returned verbatim.
        let getter = env_from(&[("WQM_DATABASE_PATH", "/data/$HOME/x")]);
        let config = load_config_with_env(&getter).expect("load");
        assert_eq!(config.database.path, "/data/$HOME/x");
    }

    // ── File-merge golden parity (AC-a2.2) ─────────────────────────────────

    const FULL_OVERRIDE_YAML: &str = r#"
database:
  path: "/custom/db.sqlite"
qdrant:
  url: "http://myqdrant:6333"
  apiKey: "secret-key"
  timeout: 5000
daemon:
  grpcHost: "myhost"
  grpcPort: 9999
  queuePollIntervalMs: 1000
  queueBatchSize: 20
watching:
  patterns:
    - "*.go"
  ignorePatterns:
    - ".custom/*"
collections:
  rulesCollectionName: "custom-rules"
environment:
  userPath: "/usr/local/bin"
rules:
  limits:
    maxLabelLength: 30
    maxTitleLength: 100
    maxTagLength: 40
    maxTagsPerRule: 10
  duplicationThreshold: 0.85
"#;

    fn load_with_file(yaml: &str, extra_env: &[(&str, &str)]) -> ServerConfig {
        let dir = tempfile::tempdir().expect("tmpdir");
        let file_path = dir.path().join("config.yaml");
        {
            let mut f = std::fs::File::create(&file_path).expect("create");
            f.write_all(yaml.as_bytes()).expect("write");
        }
        let path_str = file_path.to_string_lossy().into_owned();
        let extra: HashMap<String, String> = extra_env
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let getter = move |key: &str| {
            if key == "WQM_CONFIG_PATH" {
                Some(path_str.clone())
            } else {
                extra.get(key).cloned()
            }
        };
        load_config_with_env(&getter).expect("load")
    }

    #[test]
    fn full_override_yaml_sets_all_fields() {
        let config = load_with_file(FULL_OVERRIDE_YAML, &[]);
        assert_eq!(config.database.path, "/custom/db.sqlite");
        assert_eq!(config.qdrant.url, "http://myqdrant:6333");
        assert_eq!(config.qdrant.api_key, Some("secret-key".to_string()));
        assert_eq!(config.qdrant.timeout, 5000);
        assert_eq!(config.daemon.grpc_host, "myhost");
        assert_eq!(config.daemon.grpc_port, 9999);
        assert_eq!(config.daemon.queue_poll_interval_ms, 1000);
        assert_eq!(config.daemon.queue_batch_size, 20);
        assert_eq!(config.watching.patterns, vec!["*.go"]);
        assert_eq!(config.watching.ignore_patterns, vec![".custom/*"]);
        assert_eq!(config.collections.rules_collection_name, "custom-rules");
        assert_eq!(
            config.environment.user_path,
            Some("/usr/local/bin".to_string())
        );
        let rules = config.rules.unwrap();
        assert_eq!(rules.limits.max_label_length, 30);
        assert_eq!(rules.limits.max_title_length, 100);
        assert_eq!(rules.limits.max_tag_length, 40);
        assert_eq!(rules.limits.max_tags_per_rule, 10);
        assert_eq!(rules.duplication_threshold, Some(0.85));
    }

    #[test]
    fn minimal_yaml_overrides_only_qdrant_url() {
        let yaml = "qdrant:\n  url: \"http://custom-qdrant:6333\"\n";
        let config = load_with_file(yaml, &[]);
        let base = ServerConfig::default();
        assert_eq!(config.qdrant.url, "http://custom-qdrant:6333");
        assert_eq!(config.qdrant.timeout, base.qdrant.timeout);
        assert_eq!(config.daemon.grpc_port, base.daemon.grpc_port);
    }

    #[test]
    fn array_replace_not_append() {
        let yaml = "watching:\n  patterns:\n    - \"*.go\"\n";
        let base_ignore = ServerConfig::default().watching.ignore_patterns.len();
        let config = load_with_file(yaml, &[]);
        assert_eq!(config.watching.patterns, vec!["*.go"]);
        // ignorePatterns absent in override → keeps default count.
        assert_eq!(config.watching.ignore_patterns.len(), base_ignore);
    }

    #[test]
    fn rules_limits_partial_override() {
        let yaml = "rules:\n  limits:\n    maxTitleLength: 80\n";
        let config = load_with_file(yaml, &[]);
        let base = ServerConfig::default();
        let rules = config.rules.unwrap();
        assert_eq!(rules.limits.max_title_length, 80);
        // Unchanged defaults survive the nested merge.
        assert_eq!(
            rules.limits.max_label_length,
            base.rules.unwrap().limits.max_label_length
        );
    }

    #[test]
    fn invalid_yaml_falls_back_to_defaults() {
        // A found-but-unparseable file warns and keeps defaults (TS behaviour).
        let config = load_with_file(": invalid: yaml: {{{", &[]);
        assert_eq!(config.qdrant.url, "http://localhost:6333");
    }

    // ── Env-override precedence integration (env beats file) ────────────────

    #[test]
    fn env_overrides_beat_file_merge() {
        let config = load_with_file(
            FULL_OVERRIDE_YAML,
            &[
                ("QDRANT_URL", "http://env-qdrant:6333"),
                ("WQM_DAEMON_ENDPOINT", "env-host:8888"),
            ],
        );
        // Env wins over the file's qdrant.url + daemon endpoint.
        assert_eq!(config.qdrant.url, "http://env-qdrant:6333");
        assert_eq!(config.daemon.grpc_host, "env-host");
        assert_eq!(config.daemon.grpc_port, 8888);
        // File value survives where no env override exists.
        assert_eq!(config.qdrant.timeout, 5000);
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
}

#[cfg(test)]
#[path = "env_overrides_tests.rs"]
mod env_overrides_tests;
