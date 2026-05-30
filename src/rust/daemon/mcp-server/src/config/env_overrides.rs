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
// parse_int_prefix — mirrors JS parseInt(s, 10)
// ---------------------------------------------------------------------------

/// Parse the leading integer in a string, mirroring `parseInt(s, 10)` semantics.
///
/// Algorithm:
///   1. Skip leading ASCII whitespace (space, tab, newline, etc.).
///   2. Consume an optional leading `+` or `-` sign.
///   3. Parse consecutive ASCII decimal digits.
///   4. Ignore any trailing non-digit characters.
///   5. Return `None` if no digit run was found (mirrors `NaN`).
///
/// This matches `parseInt("8080abc", 10)` → `8080` and
/// `parseInt("  8080", 10)` → `8080`, which the TypeScript legacy port
/// path relies on.  Unlike `parseGrpcEndpoint`, the TypeScript port-only
/// path (`config.ts:140`) applies no upper-bound check after `parseInt`.
fn parse_int_prefix(s: &str) -> Option<i64> {
    let s = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if s.is_empty() {
        return None;
    }
    let (s, negative) = if let Some(rest) = s.strip_prefix('-') {
        (rest, true)
    } else if let Some(rest) = s.strip_prefix('+') {
        (rest, false)
    } else {
        (s, false)
    };
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    let n: i64 = digits.parse().ok()?;
    Some(if negative { -n } else { n })
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
        // Uses parse_int_prefix to match JS parseInt(_, 10) leading-integer
        // semantics: "8080abc" → 8080, "  8080" → 8080, no-digit-run → None.
        // TS applies no range check on this path. We honor any u16-representable
        // value INCLUDING 0 (TS would also set port 0 → a non-functional port),
        // so a malformed `WQM_DAEMON_PORT=0` yields the same effective endpoint
        // as TS rather than silently keeping the default.
        // PARITY DIVERGENCE (documented, type-level only): values outside the
        // u16 range (negative or > 65535) cannot be represented by `grpc_port:
        // u16`; TS would store them raw to a non-functional port. Such pathological
        // values are ignored here.
        if let Some(n) = parse_int_prefix(&port_str) {
            if (0..=65535).contains(&n) {
                config.daemon.grpc_port = n as u16;
            }
        }
    }

    // Note: the rules duplication threshold is intentionally NOT an env
    // override. TS resolves `config.rules?.duplicationThreshold` from the loaded
    // config only (config.ts applyEnvironmentOverrides has no such env var), so a
    // Rust-only `WQM_RULES_DEDUP_THRESHOLD` would diverge from the TS server. The
    // value flows from the config file via `ServerConfig.rules.duplication_threshold`.

    config
}

#[cfg(test)]
#[path = "env_overrides_tests.rs"]
mod tests;
