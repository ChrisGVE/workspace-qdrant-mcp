//! Configuration types for the MCP server.
//!
//! Mirrors the TypeScript interface in `src/typescript/mcp-server/src/types/config.ts`
//! field-for-field so that YAML config files are interchangeable between the two
//! implementations.  YAML keys use camelCase to match the TS schema.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// DatabaseConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DatabaseConfig {
    pub path: String,
}

// ---------------------------------------------------------------------------
// QdrantConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QdrantConfig {
    pub url: String,
    // Never serialized (WI-g1): secrets must not be written back to a config
    // file. `skip_serializing_if` still emitted the key when present; use
    // `skip_serializing` so it is always omitted. Deserialization still loads an
    // operator-provided value.
    #[serde(skip_serializing)]
    pub api_key: Option<String>,
    pub timeout: u64,
}

// ---------------------------------------------------------------------------
// DaemonConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DaemonConfig {
    pub grpc_host: String,
    pub grpc_port: u16,
    pub queue_poll_interval_ms: u64,
    pub queue_batch_size: u32,
}

// ---------------------------------------------------------------------------
// WatchingConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct WatchingConfig {
    pub patterns: Vec<String>,
    pub ignore_patterns: Vec<String>,
}

// ---------------------------------------------------------------------------
// CollectionsConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CollectionsConfig {
    pub rules_collection_name: String,
}

// ---------------------------------------------------------------------------
// RuleLimitsConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RuleLimitsConfig {
    pub max_label_length: u32,
    pub max_title_length: u32,
    pub max_tag_length: u32,
    pub max_tags_per_rule: u32,
}

// ---------------------------------------------------------------------------
// RuleConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RuleConfig {
    pub limits: RuleLimitsConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplication_threshold: Option<f64>,
}

// ---------------------------------------------------------------------------
// EnvironmentConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct EnvironmentConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_path: Option<String>,
}

// ---------------------------------------------------------------------------
// ServerConfig  (top-level)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ServerConfig {
    pub database: DatabaseConfig,
    pub qdrant: QdrantConfig,
    pub daemon: DaemonConfig,
    pub watching: WatchingConfig,
    pub collections: CollectionsConfig,
    pub environment: EnvironmentConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<RuleConfig>,
}

// ---------------------------------------------------------------------------
// Default values (mirrors generated-defaults.ts / DEFAULT_CONFIG)
// ---------------------------------------------------------------------------

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: default_database_path(),
        }
    }
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self {
            url: wqm_common::constants::DEFAULT_QDRANT_URL.to_string(),
            api_key: None,
            timeout: 30_000,
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            grpc_host: "localhost".to_string(),
            grpc_port: wqm_common::constants::DEFAULT_GRPC_PORT,
            queue_poll_interval_ms: 500,
            queue_batch_size: 10,
        }
    }
}

impl Default for WatchingConfig {
    fn default() -> Self {
        Self {
            patterns: vec![
                "*.py".to_string(),
                "*.rs".to_string(),
                "*.md".to_string(),
                "*.js".to_string(),
                "*.ts".to_string(),
            ],
            ignore_patterns: vec![
                "*.pyc".to_string(),
                "*.class".to_string(),
                "*.o".to_string(),
                "*.obj".to_string(),
                "*.lock".to_string(),
                "*.min.js".to_string(),
                "*.min.css".to_string(),
                "*.map".to_string(),
                "*.bundle.js".to_string(),
                "*.chunk.js".to_string(),
                "node_modules/*".to_string(),
                "target/*".to_string(),
                "build/*".to_string(),
                "dist/*".to_string(),
                "out/*".to_string(),
                ".git/*".to_string(),
                "__pycache__/*".to_string(),
                ".venv/*".to_string(),
                "venv/*".to_string(),
                ".env/*".to_string(),
                ".tox/*".to_string(),
                ".mypy_cache/*".to_string(),
                ".pytest_cache/*".to_string(),
                ".ruff_cache/*".to_string(),
                ".gradle/*".to_string(),
                ".next/*".to_string(),
                ".nuxt/*".to_string(),
                ".svelte-kit/*".to_string(),
                ".astro/*".to_string(),
                "Pods/*".to_string(),
                "DerivedData/*".to_string(),
                ".build/*".to_string(),
                ".swiftpm/*".to_string(),
                ".fastembed_cache/*".to_string(),
                ".terraform/*".to_string(),
                ".terragrunt-cache/*".to_string(),
                "coverage/*".to_string(),
                ".nyc_output/*".to_string(),
                ".cargo/*".to_string(),
                ".rustup/*".to_string(),
                "vendor/*".to_string(),
                ".bundle/*".to_string(),
                ".cache/*".to_string(),
                ".tmp/*".to_string(),
                "tmp/*".to_string(),
                ".DS_Store/*".to_string(),
                ".idea/*".to_string(),
                ".vscode/*".to_string(),
                ".settings/*".to_string(),
                ".project/*".to_string(),
                ".classpath/*".to_string(),
                "bin/*".to_string(),
                "obj/*".to_string(),
                ".zig-cache/*".to_string(),
                "zig-out/*".to_string(),
                "elm-stuff/*".to_string(),
                ".stack-work/*".to_string(),
                "_build/*".to_string(),
                "deps/*".to_string(),
                ".dart_tool/*".to_string(),
                ".pub-cache/*".to_string(),
            ],
        }
    }
}

impl Default for CollectionsConfig {
    fn default() -> Self {
        Self {
            rules_collection_name: "rules".to_string(),
        }
    }
}

impl Default for RuleLimitsConfig {
    fn default() -> Self {
        Self {
            max_label_length: 15,
            max_title_length: 50,
            max_tag_length: 20,
            max_tags_per_rule: 5,
        }
    }
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            limits: RuleLimitsConfig::default(),
            duplication_threshold: None,
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            database: DatabaseConfig::default(),
            qdrant: QdrantConfig::default(),
            daemon: DaemonConfig::default(),
            watching: WatchingConfig::default(),
            collections: CollectionsConfig::default(),
            environment: EnvironmentConfig::default(),
            rules: Some(RuleConfig::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Path helpers (mirrors TypeScript paths.ts)
// ---------------------------------------------------------------------------

/// Returns the canonical data directory.
///
/// Precedence: `WQM_DATA_DIR` > `XDG_DATA_HOME`/workspace-qdrant >
/// `~/.local/share/workspace-qdrant`.
pub fn data_directory() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("WQM_DATA_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let base = if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        std::path::PathBuf::from(xdg)
    } else {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
            .join(".local")
            .join("share")
    };
    base.join("workspace-qdrant")
}

/// Returns the default database path (`<data_dir>/state.db`).
///
/// Does NOT apply `WQM_DATABASE_PATH`; that override is handled in
/// `env_overrides::apply_env_overrides`.
pub fn default_database_path() -> String {
    data_directory()
        .join("state.db")
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_qdrant_url() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.qdrant.url, "http://localhost:6333");
    }

    #[test]
    fn default_daemon_port() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.daemon.grpc_port, 50051);
        assert_eq!(cfg.daemon.grpc_host, "localhost");
    }

    #[test]
    fn default_daemon_poll_and_batch() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.daemon.queue_poll_interval_ms, 500);
        assert_eq!(cfg.daemon.queue_batch_size, 10);
    }

    #[test]
    fn default_rules_limits() {
        let cfg = ServerConfig::default();
        let limits = &cfg.rules.as_ref().unwrap().limits;
        assert_eq!(limits.max_label_length, 15);
        assert_eq!(limits.max_title_length, 50);
        assert_eq!(limits.max_tag_length, 20);
        assert_eq!(limits.max_tags_per_rule, 5);
    }

    #[test]
    fn default_collections() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.collections.rules_collection_name, "rules");
    }

    #[test]
    fn default_qdrant_timeout() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.qdrant.timeout, 30_000);
    }

    #[test]
    fn default_database_path_ends_with_state_db() {
        let cfg = ServerConfig::default();
        assert!(cfg.database.path.ends_with("state.db"));
    }

    #[test]
    fn default_watching_patterns_present() {
        let cfg = ServerConfig::default();
        assert!(cfg.watching.patterns.contains(&"*.rs".to_string()));
        assert!(cfg
            .watching
            .ignore_patterns
            .contains(&"target/*".to_string()));
    }

    /// Full parity test: assert `watching.patterns` and `watching.ignore_patterns`
    /// are byte-identical (in order) to the arrays in
    /// `src/typescript/mcp-server/src/types/generated-defaults.ts`.
    ///
    /// This locks the lists against future drift between the TS and Rust
    /// implementations.
    #[test]
    fn watching_patterns_full_parity_with_ts_defaults() {
        let cfg = ServerConfig::default();

        // TS generated-defaults.ts line ~29: patterns: ['*.py','*.rs','*.md','*.js','*.ts']
        let expected_patterns: Vec<&str> = vec!["*.py", "*.rs", "*.md", "*.js", "*.ts"];
        assert_eq!(
            cfg.watching.patterns,
            expected_patterns
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "watching.patterns must exactly match TS generated-defaults.ts"
        );

        // TS generated-defaults.ts lines ~30-92: ignorePatterns (61 entries, in order)
        let expected_ignore: Vec<&str> = vec![
            "*.pyc",
            "*.class",
            "*.o",
            "*.obj",
            "*.lock",
            "*.min.js",
            "*.min.css",
            "*.map",
            "*.bundle.js",
            "*.chunk.js",
            "node_modules/*",
            "target/*",
            "build/*",
            "dist/*",
            "out/*",
            ".git/*",
            "__pycache__/*",
            ".venv/*",
            "venv/*",
            ".env/*",
            ".tox/*",
            ".mypy_cache/*",
            ".pytest_cache/*",
            ".ruff_cache/*",
            ".gradle/*",
            ".next/*",
            ".nuxt/*",
            ".svelte-kit/*",
            ".astro/*",
            "Pods/*",
            "DerivedData/*",
            ".build/*",
            ".swiftpm/*",
            ".fastembed_cache/*",
            ".terraform/*",
            ".terragrunt-cache/*",
            "coverage/*",
            ".nyc_output/*",
            ".cargo/*",
            ".rustup/*",
            "vendor/*",
            ".bundle/*",
            ".cache/*",
            ".tmp/*",
            "tmp/*",
            ".DS_Store/*",
            ".idea/*",
            ".vscode/*",
            ".settings/*",
            ".project/*",
            ".classpath/*",
            "bin/*",
            "obj/*",
            ".zig-cache/*",
            "zig-out/*",
            "elm-stuff/*",
            ".stack-work/*",
            "_build/*",
            "deps/*",
            ".dart_tool/*",
            ".pub-cache/*",
        ];
        assert_eq!(
            cfg.watching.ignore_patterns,
            expected_ignore
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
            "watching.ignore_patterns must exactly match TS generated-defaults.ts (61 entries, in order)"
        );
        assert_eq!(
            cfg.watching.ignore_patterns.len(),
            61,
            "expecting exactly 61 ignore_patterns entries"
        );
    }

    #[test]
    fn server_config_roundtrip_json() {
        let cfg = ServerConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let back: ServerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg, back);
    }

    #[test]
    fn qdrant_api_key_never_serialized() {
        // WI-g1 / AC-g1.1: a configured api_key must not appear in serialized
        // output (no cleartext written back to a config file).
        let cfg = QdrantConfig {
            url: "http://localhost:6333".to_string(),
            api_key: Some("sk-secret-value".to_string()),
            timeout: 30,
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert!(
            !json.contains("sk-secret-value"),
            "api_key leaked into JSON: {json}"
        );
        // AC-g1.2: a file that DOES carry the key still loads into memory.
        let loaded: QdrantConfig =
            serde_json::from_str(r#"{"url":"u","apiKey":"sk-x","timeout":1}"#).expect("load");
        assert_eq!(loaded.api_key.as_deref(), Some("sk-x"));
    }
}
