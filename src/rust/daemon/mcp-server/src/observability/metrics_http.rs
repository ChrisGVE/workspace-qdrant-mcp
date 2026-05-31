//! Metrics HTTP server — exposes Prometheus `/metrics` endpoint.
//!
//! Mirrors `src/typescript/mcp-server/src/telemetry/http-server.ts`.
//!
//! ## Auth policy (mirrors http-server.ts:116-165)
//!
//! | Host          | `MCP_METRICS_TOKEN` | Behaviour                                  |
//! |---------------|---------------------|--------------------------------------------|
//! | loopback      | any                 | No auth — loopback-only is safe to scrape  |
//! | non-loopback  | set                 | Bearer token required (constant-time cmp)  |
//! | non-loopback  | unset               | Fail-closed — all requests → 401           |
//!
//! When non-loopback + no token, a startup warning is logged matching the TS
//! `process.stderr.write` in http-server.ts:117-120.
//!
//! ## Port validation
//!
//! Invalid `MCP_METRICS_PORT` → `serve_metrics` returns `Err` before binding,
//! matching the TS `throw new Error(...)` at http-server.ts:106-109.
//!
//! ## Endpoint
//!
//! Only `GET /metrics` is accepted; all other paths → 404 (`Not Found`).

use axum::{
    body::Body,
    http::{HeaderMap, Request, Response, StatusCode},
    Router,
};
use secrecy::{ExposeSecret, SecretString};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::metrics::render_metrics;
use crate::transport::auth::{constant_time_equals, extract_bearer, token_digest};

// ─────────────────────────────────────────────────────────────────────────────
// Constants (mirrors http-server.ts:39-40)
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_METRICS_PORT: u16 = 9092;
const DEFAULT_METRICS_HOST: &str = "127.0.0.1";

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Metrics server configuration resolved from environment variables.
#[derive(Clone)]
pub struct MetricsConfig {
    pub host: String,
    pub port: u16,
    /// The optional bearer token. `None` means no token is configured.
    pub token: Option<SecretString>,
}

impl MetricsConfig {
    /// Load from environment; fall back to defaults for unset vars.
    pub fn from_env() -> Result<Self, String> {
        let host =
            std::env::var("MCP_METRICS_HOST").unwrap_or_else(|_| DEFAULT_METRICS_HOST.to_string());

        let port = match std::env::var("MCP_METRICS_PORT") {
            Err(_) => DEFAULT_METRICS_PORT,
            Ok(raw) => {
                let n: u64 = raw
                    .trim()
                    .parse()
                    .map_err(|_| format!("[wqm-metrics] Invalid MCP_METRICS_PORT value \"{raw}\". Must be an integer in [1, 65535]."))?;
                if n < 1 || n > 65535 {
                    return Err(format!("[wqm-metrics] Invalid MCP_METRICS_PORT value \"{raw}\". Must be an integer in [1, 65535]."));
                }
                n as u16
            }
        };

        let token = std::env::var("MCP_METRICS_TOKEN")
            .ok()
            .map(|s| SecretString::new(s.into_boxed_str()));

        Ok(Self { host, port, token })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// is_loopback_address (mirrors http-server.ts:49-52)
// ─────────────────────────────────────────────────────────────────────────────

/// Returns true when `host` is a loopback address.
///
/// Covers IPv4 `127.0.0.1`, IPv6 `::1`, and `localhost` (case-insensitive).
/// Mirrors `isLoopbackAddress` in `http-server.ts:49-52`.
pub fn is_loopback_address(host: &str) -> bool {
    let lower = host.to_lowercase();
    lower == "127.0.0.1" || lower == "::1" || lower == "localhost"
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared handler state
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HandlerState {
    loopback: bool,
    token: Option<SecretString>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Request handler
// ─────────────────────────────────────────────────────────────────────────────

/// Handle a single metrics request.
///
/// Mirrors `handleRequest` in `http-server.ts:139-176`.
fn handle_metrics_request(
    method: &axum::http::Method,
    path: &str,
    headers: &HeaderMap,
    state: &HandlerState,
) -> Response<Body> {
    // Only GET /metrics is served (http-server.ts:145-149).
    if method != axum::http::Method::GET || path != "/metrics" {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "text/plain")
            .body(Body::from("Not Found"))
            .expect("valid response");
    }

    // Auth enforcement for non-loopback binds (http-server.ts:152-165).
    if !state.loopback {
        let auth_header = headers.get("authorization").and_then(|v| v.to_str().ok());
        let provided = extract_bearer(auth_header);

        match &state.token {
            None => {
                // No token configured — fail-closed (http-server.ts:154-158).
                return Response::builder()
                    .status(StatusCode::UNAUTHORIZED)
                    .header("Content-Type", "text/plain")
                    .header("WWW-Authenticate", "Bearer")
                    .body(Body::from("Unauthorized: MCP_METRICS_TOKEN not configured"))
                    .expect("valid response");
            }
            Some(secret) => {
                let valid = match &provided {
                    None => false,
                    Some(t) => {
                        constant_time_equals(t.as_bytes(), secret.expose_secret().as_bytes())
                    }
                };
                if !valid {
                    return Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .header("Content-Type", "text/plain")
                        .header("WWW-Authenticate", "Bearer")
                        .body(Body::from("Unauthorized"))
                        .expect("valid response");
                }
            }
        }
    }

    // Render and return metrics (http-server.ts:167-175).
    let body = render_metrics();
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        .body(Body::from(body))
        .expect("valid response")
}

// ─────────────────────────────────────────────────────────────────────────────
// Axum route handler
// ─────────────────────────────────────────────────────────────────────────────

async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<HandlerState>,
    req: Request<Body>,
) -> Response<Body> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let headers = req.headers().clone();
    handle_metrics_request(&method, &path, &headers, &state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Start the metrics HTTP server and return immediately (non-blocking).
///
/// The server runs in a background tokio task and shuts down when
/// `shutdown_token` is cancelled.
///
/// Mirrors `startMetricsServer` in `http-server.ts:97-137`.
///
/// # Errors
///
/// Returns `Err` if `MetricsConfig::from_env()` fails (invalid port) or the
/// bind address cannot be parsed.  Bind errors (EADDRINUSE, EACCES) are logged
/// as warnings but do not prevent the MCP server from starting (matching the TS
/// `server.on('error', ...)` non-fatal handler).
pub async fn serve_metrics(shutdown_token: CancellationToken) -> Result<(), String> {
    let cfg = MetricsConfig::from_env()?;
    serve_metrics_with_config(cfg, shutdown_token).await
}

/// Internal version that accepts a pre-built config — used by tests.
pub(crate) async fn serve_metrics_with_config(
    cfg: MetricsConfig,
    shutdown_token: CancellationToken,
) -> Result<(), String> {
    let loopback = is_loopback_address(&cfg.host);

    // Log the startup warning when non-loopback + no token is configured
    // (mirrors http-server.ts:116-120).
    if !loopback && cfg.token.is_none() {
        warn!(
            host = %cfg.host,
            "[wqm-metrics] metrics server bound to non-loopback host without MCP_METRICS_TOKEN \
             set. All /metrics requests will be rejected (401). Set MCP_METRICS_TOKEN to allow scraping."
        );
    }

    // Log token digest when a token is configured (like main HTTP transport).
    if let Some(ref t) = cfg.token {
        info!(
            token_digest = %token_digest(t.expose_secret()),
            "[wqm-metrics] bearer auth enabled for metrics endpoint"
        );
    }

    let bind_addr: std::net::SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .map_err(|e| format!("[wqm-metrics] Invalid bind address: {e}"))?;

    let state = HandlerState {
        loopback,
        token: cfg.token,
    };

    let router = Router::new().fallback(metrics_handler).with_state(state);

    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .map_err(|e| format!("[wqm-metrics] Failed to bind {bind_addr}: {e}"))?;

    info!(
        host = %cfg.host,
        port = cfg.port,
        "[wqm-metrics] metrics HTTP server listening on {}:{}/metrics",
        cfg.host, cfg.port
    );

    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                shutdown_token.cancelled().await;
            })
            .await
            .unwrap_or_else(|e| {
                warn!(error = %e, "[wqm-metrics] metrics server error");
            });
    });

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Method;

    // ── is_loopback_address ────────────────────────────────────────────────────

    #[test]
    fn loopback_ipv4() {
        assert!(is_loopback_address("127.0.0.1"));
    }

    #[test]
    fn loopback_ipv6() {
        assert!(is_loopback_address("::1"));
    }

    #[test]
    fn loopback_localhost_lower() {
        assert!(is_loopback_address("localhost"));
    }

    #[test]
    fn loopback_localhost_upper() {
        // case-insensitive
        assert!(is_loopback_address("LOCALHOST"));
    }

    #[test]
    fn non_loopback() {
        assert!(!is_loopback_address("0.0.0.0"));
        assert!(!is_loopback_address("192.168.1.1"));
    }

    // ── handle_metrics_request ─────────────────────────────────────────────────

    fn make_state(loopback: bool, token: Option<&str>) -> HandlerState {
        HandlerState {
            loopback,
            token: token.map(|s| SecretString::new(s.to_string().into_boxed_str())),
        }
    }

    #[test]
    fn get_metrics_loopback_no_auth_200() {
        let state = make_state(true, None);
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn get_metrics_loopback_with_token_200() {
        // Even with a token configured, loopback never checks it.
        let state = make_state(true, Some("supersecrettoken1234567890abcdef"));
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn wrong_path_404() {
        let state = make_state(true, None);
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/other", &headers, &state);
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn post_metrics_404() {
        let state = make_state(true, None);
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::POST, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[test]
    fn non_loopback_no_token_configured_401() {
        // Fail-closed: non-loopback + no token → 401 regardless of request.
        let state = make_state(false, None);
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn non_loopback_token_configured_no_bearer_401() {
        let state = make_state(false, Some("supersecrettoken1234567890abcdef"));
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn non_loopback_token_configured_wrong_bearer_401() {
        let state = make_state(false, Some("supersecrettoken1234567890abcdef"));
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            "Bearer wrongtoken".parse().expect("valid header"),
        );
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn non_loopback_token_configured_correct_bearer_200() {
        let token = "supersecrettoken1234567890abcdef";
        let state = make_state(false, Some(token));
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            format!("Bearer {token}").parse().expect("valid header"),
        );
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[test]
    fn get_metrics_200_body_contains_prometheus_exposition() {
        let state = make_state(true, None);
        let headers = HeaderMap::new();
        let resp = handle_metrics_request(&Method::GET, "/metrics", &headers, &state);
        assert_eq!(resp.status(), StatusCode::OK);
        let ct = resp
            .headers()
            .get("Content-Type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(
            ct.starts_with("text/plain"),
            "Content-Type should be text/plain; got: {ct}"
        );
    }
}
