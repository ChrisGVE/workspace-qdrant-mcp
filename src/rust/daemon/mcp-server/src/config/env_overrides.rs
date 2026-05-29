//! Environment variable overrides for `ServerConfig`.
//!
//! Mirrors `applyEnvironmentOverrides()` and `parseGrpcEndpoint()` from
//! `src/typescript/mcp-server/src/config.ts`.
//!
//! Precedence for daemon endpoint (highest → lowest):
//!   1. `WQM_DAEMON_ENDPOINT`
//!   2. `MEMEXD_GRPC_URL`   (alias for 1)
//!   3. `WQM_DAEMON_PORT`   (port-only, applied only when neither of the
//!      above is set)
//!
//! Other overrides:
//!   - `QDRANT_URL`        → `qdrant.url`
//!   - `QDRANT_API_KEY`    → `qdrant.api_key`
//!   - `WQM_DATABASE_PATH` → `database.path`

use wqm_common::constants::DEFAULT_GRPC_PORT;

use crate::config::types::ServerConfig;

// ---------------------------------------------------------------------------
// parseGrpcEndpoint
// ---------------------------------------------------------------------------

/// Parsed result of a gRPC endpoint string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrpcEndpoint {
    pub host: String,
    pub port: u16,
}

/// Parse a gRPC endpoint string into host and port components.
///
/// Accepted formats (mirrors TypeScript `parseGrpcEndpoint`):
/// - `"http://host:port"` — scheme stripped, port parsed
/// - `"host:port"`        — bare host:port
/// - `"host"`             — host only, port defaults to [`DEFAULT_GRPC_PORT`]
///
/// Invalid or out-of-range ports fall back to [`DEFAULT_GRPC_PORT`].
/// An empty string returns `host = ""` with the default port.
pub fn parse_grpc_endpoint(endpoint: &str) -> GrpcEndpoint {
    // Strip http:// or https:// scheme if present.
    let without_scheme = strip_scheme(endpoint);

    let colon_idx = without_scheme.rfind(':');
    match colon_idx {
        None => GrpcEndpoint {
            host: without_scheme.to_owned(),
            port: DEFAULT_GRPC_PORT,
        },
        Some(idx) => {
            let host = &without_scheme[..idx];
            let port_str = &without_scheme[idx + 1..];
            let port = parse_port(port_str);
            GrpcEndpoint {
                host: host.to_owned(),
                port,
            }
        }
    }
}

/// Strip a leading `http://` or `https://` scheme from a string slice.
fn strip_scheme(s: &str) -> &str {
    if let Some(rest) = s.strip_prefix("https://") {
        rest
    } else if let Some(rest) = s.strip_prefix("http://") {
        rest
    } else {
        s
    }
}

/// Parse a port string; returns [`DEFAULT_GRPC_PORT`] for any non-positive,
/// out-of-range, or non-numeric value (mirrors TS `parseInt` + bounds check).
fn parse_port(s: &str) -> u16 {
    match s.parse::<i64>() {
        Ok(n) if n > 0 && n <= 65535 => n as u16,
        _ => DEFAULT_GRPC_PORT,
    }
}

// ---------------------------------------------------------------------------
// apply_env_overrides
// ---------------------------------------------------------------------------

/// Apply environment-variable overrides to a `ServerConfig`.
///
/// Uses the injected `env_getter` so callers in tests can supply a hermetic
/// map instead of mutating process-level environment.
pub fn apply_env_overrides(
    mut config: ServerConfig,
    env_getter: &dyn Fn(&str) -> Option<String>,
) -> ServerConfig {
    // Qdrant overrides
    if let Some(url) = env_getter("QDRANT_URL") {
        config.qdrant.url = url;
    }
    if let Some(key) = env_getter("QDRANT_API_KEY") {
        config.qdrant.api_key = Some(key);
    }

    // Database path override
    if let Some(db_path) = env_getter("WQM_DATABASE_PATH") {
        config.database.path = db_path;
    }

    // Daemon endpoint: WQM_DAEMON_ENDPOINT preferred, MEMEXD_GRPC_URL as alias.
    let endpoint_env = env_getter("WQM_DAEMON_ENDPOINT").or_else(|| env_getter("MEMEXD_GRPC_URL"));

    if let Some(ep) = endpoint_env {
        let parsed = parse_grpc_endpoint(&ep);
        config.daemon.grpc_host = parsed.host;
        config.daemon.grpc_port = parsed.port;
    } else if let Some(port_str) = env_getter("WQM_DAEMON_PORT") {
        // Port-only override (legacy); only applied when no endpoint env is set.
        if let Ok(p) = port_str.parse::<u16>() {
            config.daemon.grpc_port = p;
        }
    }

    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::ServerConfig;
    use std::collections::HashMap;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn env_from<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        let map: HashMap<&str, &str> = pairs.iter().copied().collect();
        move |key: &str| map.get(key).map(|v| v.to_string())
    }

    // ------------------------------------------------------------------
    // parse_grpc_endpoint tests
    // ------------------------------------------------------------------

    #[test]
    fn parse_host_only_defaults_port() {
        let ep = parse_grpc_endpoint("localhost");
        assert_eq!(ep.host, "localhost");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_host_colon_port() {
        let ep = parse_grpc_endpoint("myhost:9090");
        assert_eq!(ep.host, "myhost");
        assert_eq!(ep.port, 9090);
    }

    #[test]
    fn parse_http_scheme_stripped() {
        let ep = parse_grpc_endpoint("http://daemon:50051");
        assert_eq!(ep.host, "daemon");
        assert_eq!(ep.port, 50051);
    }

    #[test]
    fn parse_https_scheme_stripped() {
        let ep = parse_grpc_endpoint("https://secure-host:443");
        assert_eq!(ep.host, "secure-host");
        assert_eq!(ep.port, 443);
    }

    #[test]
    fn parse_invalid_port_falls_back_to_default() {
        let ep = parse_grpc_endpoint("host:notaport");
        assert_eq!(ep.host, "host");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_zero_port_falls_back_to_default() {
        let ep = parse_grpc_endpoint("host:0");
        assert_eq!(ep.host, "host");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_negative_port_falls_back_to_default() {
        let ep = parse_grpc_endpoint("host:-1");
        assert_eq!(ep.host, "host");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_port_65535_is_valid() {
        let ep = parse_grpc_endpoint("host:65535");
        assert_eq!(ep.port, 65535);
    }

    #[test]
    fn parse_port_65536_falls_back_to_default() {
        let ep = parse_grpc_endpoint("host:65536");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_empty_string() {
        let ep = parse_grpc_endpoint("");
        assert_eq!(ep.host, "");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    #[test]
    fn parse_http_host_no_port() {
        let ep = parse_grpc_endpoint("http://myhost");
        assert_eq!(ep.host, "myhost");
        assert_eq!(ep.port, DEFAULT_GRPC_PORT);
    }

    // ------------------------------------------------------------------
    // apply_env_overrides — individual overrides
    // ------------------------------------------------------------------

    #[test]
    fn qdrant_url_override() {
        let getter = env_from(&[("QDRANT_URL", "http://remote:6333")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.qdrant.url, "http://remote:6333");
    }

    #[test]
    fn qdrant_api_key_override() {
        let getter = env_from(&[("QDRANT_API_KEY", "tok-abc123")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.qdrant.api_key, Some("tok-abc123".to_string()));
    }

    #[test]
    fn database_path_override() {
        let getter = env_from(&[("WQM_DATABASE_PATH", "/tmp/test.db")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.database.path, "/tmp/test.db");
    }

    #[test]
    fn wqm_daemon_endpoint_sets_host_and_port() {
        let getter = env_from(&[("WQM_DAEMON_ENDPOINT", "http://daemon-host:7777")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "daemon-host");
        assert_eq!(result.daemon.grpc_port, 7777);
    }

    #[test]
    fn memexd_grpc_url_sets_host_and_port() {
        let getter = env_from(&[("MEMEXD_GRPC_URL", "localhost:8888")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "localhost");
        assert_eq!(result.daemon.grpc_port, 8888);
    }

    #[test]
    fn wqm_daemon_port_sets_only_port() {
        let getter = env_from(&[("WQM_DAEMON_PORT", "9999")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, 9999);
        // Host should remain default
        assert_eq!(result.daemon.grpc_host, "localhost");
    }

    // ------------------------------------------------------------------
    // AC-C3: Precedence tests
    // ------------------------------------------------------------------

    #[test]
    fn wqm_daemon_endpoint_beats_memexd_grpc_url() {
        // Both set — WQM_DAEMON_ENDPOINT must win.
        let getter = env_from(&[
            ("WQM_DAEMON_ENDPOINT", "preferred-host:1111"),
            ("MEMEXD_GRPC_URL", "ignored-host:2222"),
        ]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "preferred-host");
        assert_eq!(result.daemon.grpc_port, 1111);
    }

    #[test]
    fn wqm_daemon_endpoint_beats_wqm_daemon_port() {
        // Endpoint env takes precedence over port-only legacy var.
        let getter = env_from(&[
            ("WQM_DAEMON_ENDPOINT", "ep-host:5555"),
            ("WQM_DAEMON_PORT", "9999"),
        ]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "ep-host");
        assert_eq!(result.daemon.grpc_port, 5555);
    }

    #[test]
    fn memexd_grpc_url_beats_wqm_daemon_port() {
        // MEMEXD_GRPC_URL beats WQM_DAEMON_PORT.
        let getter = env_from(&[
            ("MEMEXD_GRPC_URL", "alias-host:6666"),
            ("WQM_DAEMON_PORT", "9999"),
        ]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "alias-host");
        assert_eq!(result.daemon.grpc_port, 6666);
    }

    #[test]
    fn wqm_daemon_port_only_applied_when_no_endpoint_env() {
        // No endpoint env → port-only override applies.
        let getter = env_from(&[("WQM_DAEMON_PORT", "4444")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, 4444);
    }

    #[test]
    fn no_env_vars_leaves_defaults_unchanged() {
        let getter = env_from(&[]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        let default = ServerConfig::default();
        assert_eq!(result.qdrant.url, default.qdrant.url);
        assert_eq!(result.daemon.grpc_host, default.daemon.grpc_host);
        assert_eq!(result.daemon.grpc_port, default.daemon.grpc_port);
    }

    // ------------------------------------------------------------------
    // WQM_DAEMON_PORT: JS parseInt(_, 10) leading-integer semantics
    //
    // TS config.ts lines 139-144:
    //   if (!endpointEnv && process.env['WQM_DAEMON_PORT']) {
    //     const port = parseInt(process.env['WQM_DAEMON_PORT'], 10);
    //     if (!isNaN(port)) {
    //       result.daemon = { ...result.daemon, grpcPort: port };
    //     }
    //   }
    //
    // parseInt("8080abc", 10) → 8080  (trailing non-digits ignored)
    // parseInt("  8080",  10) → 8080  (leading whitespace stripped)
    // No upper-bound check in the legacy port path (unlike parseGrpcEndpoint
    // which guards port > 65535). TS would assign 0 / -5 / 70000 to grpcPort
    // as-is; Rust u16 cannot represent them — documented divergence.
    // ------------------------------------------------------------------

    #[test]
    fn wqm_daemon_port_trailing_non_digits_accepted() {
        // parseInt("8080abc", 10) → 8080; Rust must match.
        let getter = env_from(&[("WQM_DAEMON_PORT", "8080abc")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, 8080);
    }

    #[test]
    fn wqm_daemon_port_leading_whitespace_accepted() {
        // parseInt("  8080", 10) → 8080; Rust must match.
        let getter = env_from(&[("WQM_DAEMON_PORT", "  8080")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, 8080);
    }

    #[test]
    fn wqm_daemon_port_valid_plain_number_still_works() {
        // Regression guard: a plain valid port must still be accepted.
        let getter = env_from(&[("WQM_DAEMON_PORT", "9999")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, 9999);
    }

    #[test]
    fn wqm_daemon_port_zero_leaves_default() {
        // parseInt("0", 10) → 0 (not NaN). TS would assign grpcPort = 0 (non-
        // functional). Rust u16 can represent 0, but it is not a valid TCP port.
        // PARITY DIVERGENCE (documented): TS assigns out-of-range WQM_DAEMON_PORT
        // raw to a non-functional port; Rust ignores values <= 0.
        let default_port = ServerConfig::default().daemon.grpc_port;
        let getter = env_from(&[("WQM_DAEMON_PORT", "0")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, default_port);
    }

    #[test]
    fn wqm_daemon_port_negative_leaves_default() {
        // parseInt("-5", 10) → -5. TS assigns the non-functional value; Rust
        // u16 cannot represent negatives.
        // PARITY DIVERGENCE (documented): see wqm_daemon_port_zero_leaves_default.
        let default_port = ServerConfig::default().daemon.grpc_port;
        let getter = env_from(&[("WQM_DAEMON_PORT", "-5")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, default_port);
    }

    #[test]
    fn wqm_daemon_port_out_of_u16_range_leaves_default() {
        // parseInt("70000", 10) → 70000. TS assigns the non-functional value;
        // Rust u16 cannot represent values > 65535.
        // PARITY DIVERGENCE (documented): see wqm_daemon_port_zero_leaves_default.
        let default_port = ServerConfig::default().daemon.grpc_port;
        let getter = env_from(&[("WQM_DAEMON_PORT", "70000")]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_port, default_port);
    }

    #[test]
    fn wqm_daemon_port_precedence_endpoint_still_beats_trailing_garbage() {
        // Endpoint vars must still win over WQM_DAEMON_PORT even with leading-int input.
        let getter = env_from(&[
            ("WQM_DAEMON_ENDPOINT", "ep-host:5555"),
            ("WQM_DAEMON_PORT", "8080abc"),
        ]);
        let result = apply_env_overrides(ServerConfig::default(), &getter);
        assert_eq!(result.daemon.grpc_host, "ep-host");
        assert_eq!(result.daemon.grpc_port, 5555);
    }
}
