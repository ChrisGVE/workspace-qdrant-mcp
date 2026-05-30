//! Tests for `env_overrides` — `parse_grpc_endpoint` and `apply_env_overrides`.
//!
//! Included from `env_overrides.rs` via
//! `#[cfg(test)] #[path = "env_overrides_tests.rs"] mod tests;`.

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
fn wqm_daemon_port_zero_is_honored() {
    // parseInt("0", 10) → 0. TS assigns grpcPort = 0 (a non-functional port).
    // u16 CAN represent 0, so Rust honors it too — matching TS rather than
    // silently keeping the default (fixes the round-4 LOW parity finding).
    let getter = env_from(&[("WQM_DAEMON_PORT", "0")]);
    let result = apply_env_overrides(ServerConfig::default(), &getter);
    assert_eq!(result.daemon.grpc_port, 0);
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

// ------------------------------------------------------------------
// Rules duplication threshold: NOT an env override (TS parity)
// ------------------------------------------------------------------

#[test]
fn rules_dedup_threshold_is_not_an_env_override() {
    // Parity: TS has no env var for the rules duplication threshold — it comes
    // from the loaded config only. Setting WQM_RULES_DEDUP_THRESHOLD must have
    // NO effect on the merged config (the field stays at its config-file value,
    // here the default None).
    let getter = env_from(&[("WQM_RULES_DEDUP_THRESHOLD", "0.85")]);
    let result = apply_env_overrides(ServerConfig::default(), &getter);
    let threshold = result.rules.as_ref().and_then(|r| r.duplication_threshold);
    assert!(
        threshold.is_none(),
        "WQM_RULES_DEDUP_THRESHOLD must NOT override the threshold; got {threshold:?}"
    );
}
