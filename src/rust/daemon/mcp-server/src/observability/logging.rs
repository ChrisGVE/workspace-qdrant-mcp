//! Logging initialisation for the MCP server.
//!
//! The cardinal rule: in **stdio** mode, stdout carries *only* JSON-RPC frames,
//! so every log line MUST go to stderr.  In **http** mode stdout is free, so
//! logs may go there.  Tool handlers must use the `tracing` macros (`debug!`,
//! `info!`, `warn!`, `error!`) exclusively — never `println!`/`eprintln!`.
//!
//! Log level resolution mirrors the TypeScript server (`utils/logger.ts:70`):
//! `WQM_MCP_LOG_LEVEL` takes precedence over `WQM_LOG_LEVEL`, defaulting to
//! `info` when neither is set.  Unlike the TS server (which logs to a rotating
//! JSONL file), the Rust server logs to a standard stream because the daemon
//! owns persistent log files; the parity-critical guarantee is stdout purity,
//! not the log sink.

use crate::server_types::ServerMode;
use std::sync::Once;
use tracing_subscriber::EnvFilter;

/// Guards one-time global subscriber installation so repeated calls (e.g. in
/// tests) do not panic.
static INIT: Once = Once::new();

/// Environment variable consulted first for the log level (MCP-specific).
const ENV_MCP_LOG_LEVEL: &str = "WQM_MCP_LOG_LEVEL";
/// Environment variable consulted as a fallback for the log level (shared).
const ENV_LOG_LEVEL: &str = "WQM_LOG_LEVEL";
/// Default log level when no environment override is present.
const DEFAULT_LOG_LEVEL: &str = "info";

/// Resolve the effective log-level directive from an injectable environment
/// getter, matching the TypeScript precedence:
/// `WQM_MCP_LOG_LEVEL` > `WQM_LOG_LEVEL` > `info`.
///
/// An empty-string value is treated as unset, matching the JS `??` behaviour
/// only loosely — JS `??` would accept `''`, but an empty `RUST_LOG`-style
/// directive is useless, so we fall through to the next source.
fn resolve_log_level<F>(get_env: F) -> String
where
    F: Fn(&str) -> Option<String>,
{
    for key in [ENV_MCP_LOG_LEVEL, ENV_LOG_LEVEL] {
        if let Some(val) = get_env(key) {
            if !val.trim().is_empty() {
                return val;
            }
        }
    }
    DEFAULT_LOG_LEVEL.to_string()
}

/// Whether logs should be written to stderr (vs stdout) for the given mode.
///
/// `Stdio` → stderr (stdout reserved for JSON-RPC); `Http` → stdout is allowed.
fn logs_to_stderr(mode: ServerMode) -> bool {
    matches!(mode, ServerMode::Stdio)
}

/// Build an [`EnvFilter`] from a level directive, falling back to the default
/// level if the directive is unparseable rather than panicking.
fn build_filter(level: &str) -> EnvFilter {
    EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new(DEFAULT_LOG_LEVEL))
}

/// Initialise the global tracing subscriber for the given operating mode.
///
/// Idempotent: only the first call installs a subscriber; later calls are
/// no-ops.  Must be called once at `main` startup before any logging occurs.
pub fn init_logging(mode: ServerMode) {
    init_logging_with_env(mode, |k| std::env::var(k).ok());
}

/// Testable core of [`init_logging`] with an injectable environment getter.
pub fn init_logging_with_env<F>(mode: ServerMode, get_env: F)
where
    F: Fn(&str) -> Option<String>,
{
    INIT.call_once(|| {
        let level = resolve_log_level(get_env);
        let builder = tracing_subscriber::fmt()
            .with_env_filter(build_filter(&level))
            .with_target(false);

        // Two distinct writer types require two build arms.
        if logs_to_stderr(mode) {
            builder.with_writer(std::io::stderr).init();
        } else {
            builder.with_writer(std::io::stdout).init();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_defaults_to_info_when_unset() {
        let level = resolve_log_level(|_| None);
        assert_eq!(level, "info");
    }

    #[test]
    fn mcp_log_level_takes_precedence() {
        let level = resolve_log_level(|k| match k {
            "WQM_MCP_LOG_LEVEL" => Some("debug".to_string()),
            "WQM_LOG_LEVEL" => Some("warn".to_string()),
            _ => None,
        });
        assert_eq!(level, "debug");
    }

    #[test]
    fn falls_back_to_shared_log_level() {
        let level = resolve_log_level(|k| match k {
            "WQM_LOG_LEVEL" => Some("error".to_string()),
            _ => None,
        });
        assert_eq!(level, "error");
    }

    #[test]
    fn empty_value_is_treated_as_unset() {
        let level = resolve_log_level(|k| match k {
            "WQM_MCP_LOG_LEVEL" => Some("   ".to_string()),
            "WQM_LOG_LEVEL" => Some("trace".to_string()),
            _ => None,
        });
        assert_eq!(level, "trace");
    }

    #[test]
    fn empty_both_defaults_to_info() {
        let level = resolve_log_level(|_| Some(String::new()));
        assert_eq!(level, "info");
    }

    #[test]
    fn stdio_mode_routes_to_stderr() {
        assert!(logs_to_stderr(ServerMode::Stdio));
    }

    #[test]
    fn http_mode_allows_stdout() {
        assert!(!logs_to_stderr(ServerMode::Http));
    }

    #[test]
    fn build_filter_accepts_valid_directive() {
        // Smoke test: a standard directive builds without panicking.
        let _ = build_filter("debug");
        let _ = build_filter("mcp_server=trace,info");
    }

    #[test]
    fn build_filter_falls_back_on_garbage() {
        // An invalid directive must not panic; it falls back to the default.
        let _ = build_filter("this is !!! not a filter @@@");
    }

    #[test]
    fn init_logging_is_idempotent() {
        // Calling twice must not panic (Once guards global install).
        init_logging_with_env(ServerMode::Stdio, |_| None);
        init_logging_with_env(ServerMode::Http, |_| None);
    }
}
