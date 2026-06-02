//! Shared configuration machinery (WI-a1).
//!
//! Three components (daemon, CLI, MCP server) historically each carried their
//! own copy of config discovery, format parsing, merge-over-defaults, env-var
//! overrides, validators, and path expansion. This module hosts the **shared,
//! component-agnostic** primitives; each component keeps only its typed config
//! VIEW, its env-var spec list, and any component-specific validators that call
//! these primitives.
//!
//! Submodules:
//! - [`format`] — [`ConfigFormat`] detection + parse-to-`Value` (YAML / TOML).
//! - [`discovery`] — search-path resolution, parametrized by env-var names,
//!   app subdir, and candidate filenames.
//! - [`merge`] — recursive merge of an override `Value` over typed defaults.
//! - [`env_override`] — declarative env-override engine + gRPC-endpoint parsing.
//! - [`validators`] — URL / port / timeout / path validators.
//! - [`path_expand`] — TS-faithful tilde-only expansion (the full env-expand
//!   mode lives in [`crate::env_expand`]).

pub mod discovery;
pub mod env_override;
pub mod format;
pub mod merge;
pub mod path_expand;
pub mod validators;

pub use discovery::{ConfigDiscovery, EnvGetter};
pub use env_override::{
    apply_env_overrides, parse_grpc_endpoint, parse_int_prefix, EnvOverride, GrpcEndpoint,
};
pub use format::ConfigFormat;
pub use merge::{merge_over_defaults, merge_value};
pub use path_expand::{expand_path_ts, expand_path_ts_with_home};
pub use validators::{validate_path, validate_port, validate_timeout, validate_url};

/// Errors produced by the shared config machinery.
///
/// Deliberately free of `anyhow` / daemon error types so `wqm-common` stays a
/// leaf crate; components map these into their own error enums.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// A config file was found but could not be parsed.
    #[error("config parse error: {0}")]
    Parse(String),
    /// Merge of an override document over typed defaults failed (serde round-trip).
    #[error("config merge error: {0}")]
    Merge(String),
    /// A required config file was not found.
    #[error("config file not found: {0}")]
    NotFound(String),
    /// A resolved value failed validation.
    #[error("config validation error: {0}")]
    Validation(String),
}

#[cfg(test)]
mod tests {
    //! AC-a1.3: the shared loader must never surface a secret VALUE. A typed view
    //! with a manual redacting `Debug` (the pattern WI-g2 applies to the real
    //! secret-bearing structs) keeps its secret out of any log line that
    //! debug-prints the loaded config — even after a secret arrives via the
    //! merge primitive (the realistic file/env load path).

    use super::merge::merge_over_defaults;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone, Serialize, Deserialize)]
    #[serde(rename_all = "camelCase")]
    struct Secretish {
        url: String,
        // Never serialized back out (WI-g1 pattern); still loadable.
        #[serde(skip_serializing)]
        api_key: Option<String>,
    }

    impl std::fmt::Debug for Secretish {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Secretish")
                .field("url", &self.url)
                .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
                .finish()
        }
    }

    #[test]
    fn loaded_secret_is_redacted_in_debug() {
        let base = Secretish {
            url: "http://default:6333".into(),
            api_key: None,
        };
        // A secret arrives via the file-merge path.
        let loaded =
            merge_over_defaults(base, &json!({"apiKey": "sk-loader-secret"})).expect("merge");
        assert_eq!(loaded.api_key.as_deref(), Some("sk-loader-secret"));

        for rendered in [format!("{loaded:?}"), format!("{loaded:#?}")] {
            assert!(
                !rendered.contains("sk-loader-secret"),
                "secret leaked into Debug (a loader log line would carry it): {rendered}"
            );
        }
    }
}
