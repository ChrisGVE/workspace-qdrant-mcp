//! Streamable-HTTP MCP transport (`serve_http`).
//!
//! Assembles the axum router that wires together:
//! - `GET /healthz` — unauthenticated liveness probe (200 `ok`).
//! - `POST|GET|DELETE <mcp_path>` — rmcp `StreamableHttpService`.
//!
//! Every non-`/healthz` request flows through a single middleware function
//! that applies checks in **exactly the same order as `mcp-http-server.ts`**:
//!
//! 1. CORS / OPTIONS preflight
//! 2. Rate-limit (per client IP)
//! 3. Bearer auth
//! 4. Route dispatch (`<mcp_path>` or 404)
//!
//! `/healthz` bypasses all four checks (it comes first in the handler).
//!
//! # TLS
//!
//! When `MCP_HTTP_TLS_CERT` and `MCP_HTTP_TLS_KEY` are set, the server
//! terminates TLS via `axum_server::bind_rustls` (axum-server 0.8 with the
//! `tls-rustls-no-provider` feature + ring crypto provider already in the
//! dependency graph via tonic).  When neither is set, the server binds a
//! plain TCP socket.  Key material is never logged.
//!
//! # Environment variables
//!
//! | Variable              | Default       | Description                         |
//! |-----------------------|---------------|-------------------------------------|
//! | `MCP_HTTP_HOST`       | `127.0.0.1`   | Bind address                        |
//! | `MCP_HTTP_PORT`       | `6335`        | Bind port                           |
//! | `MCP_HTTP_PATH`       | `/mcp`        | MCP endpoint path                   |
//! | `MCP_HTTP_TOKEN`      | —             | Bearer token (required)             |
//! | `MCP_HTTP_RATE_LIMIT` | `100`         | Requests/min per IP                 |
//! | `MCP_HTTP_CORS_ORIGINS` | *(disabled)* | Comma-separated allowed origins    |
//! | `MCP_HTTP_TLS_CERT`   | —             | PEM cert path (optional TLS)        |
//! | `MCP_HTTP_TLS_KEY`    | —             | PEM key path (optional TLS)         |
//! | `MCP_HTTP_TLS_CA`     | —             | PEM CA-bundle path (optional)       |

use std::sync::Arc;

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Request, Response, StatusCode},
    response::IntoResponse,
    routing::get,
    Router,
};
use axum_server::tls_rustls::RustlsConfig;
use tokio_util::sync::CancellationToken;
use tracing::info;

use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use tokio::sync::Mutex;

use crate::grpc::client::DaemonClient;
use crate::observability::health_monitor::{HealthState, SharedHealthState};
use crate::observability::metrics::{
    record_http_auth_failure, record_http_rate_limited, record_http_request,
};
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::{SharedStateManager, StateManager};
use crate::tools::ToolsHandler;
use crate::transport::auth::{check_bearer, require_auth, AuthConfig, BearerOutcome};
use crate::transport::cors::{add_cors_response_headers, apply_cors, CorsConfig, CorsOutcome};
use crate::transport::rate_limit::{RateLimitConfig, SlidingWindowLimiter};
use crate::transport::tls::{build_rustls_server_config, tls_config_from_env};

use super::health::healthz_response;

// ─────────────────────────────────────────────────────────────────────────────
// Defaults
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 6335;
const DEFAULT_PATH: &str = "/mcp";

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// HTTP transport configuration resolved from environment variables.
#[derive(Debug, Clone)]
pub struct HttpConfig {
    pub host: String,
    pub port: u16,
    pub mcp_path: String,
}

impl HttpConfig {
    /// Load from environment variables; fall back to defaults for unset vars.
    pub fn from_env() -> Self {
        let host = std::env::var("MCP_HTTP_HOST").unwrap_or_else(|_| DEFAULT_HOST.to_string());
        let port = std::env::var("MCP_HTTP_PORT")
            .ok()
            .and_then(|v| v.trim().parse::<u16>().ok())
            .unwrap_or(DEFAULT_PORT);
        let mcp_path = std::env::var("MCP_HTTP_PATH").unwrap_or_else(|_| DEFAULT_PATH.to_string());
        Self {
            host,
            port,
            mcp_path,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared middleware state
// ─────────────────────────────────────────────────────────────────────────────

/// All per-request state shared across the middleware closure.
#[derive(Clone)]
struct MiddlewareState {
    auth: AuthConfig,
    cors: CorsConfig,
    limiter: Arc<SlidingWindowLimiter>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Start the streamable-HTTP MCP server.
///
/// This function:
/// 1. Validates that a bearer token is configured (fatal on missing/short token).
/// 2. Builds the axum router with `/healthz` (unauth) and `<mcp_path>` (auth).
/// 3. Binds the TCP listener on `host:port` — with TLS when env vars are set.
/// 4. Drives the serve loop until `shutdown_token` is cancelled.
///
/// # Errors
///
/// Returns `Err` when the token is missing/short, TLS config is invalid, the
/// bind fails, or the serve loop encounters a fatal error.
pub async fn serve_http(
    daemon: DaemonClient,
    qdrant: QdrantReadClient,
    state: StateManager,
    session: SessionState,
    shutdown_token: CancellationToken,
) -> anyhow::Result<()> {
    // ── 1. Load all configs ──────────────────────────────────────────────────
    let http_cfg = HttpConfig::from_env();
    let auth_cfg = AuthConfig::new(std::env::var("MCP_HTTP_TOKEN").ok());
    let cors_cfg = CorsConfig::from_env();
    let rate_cfg = RateLimitConfig::from_env()
        .map_err(|e| anyhow::anyhow!("Invalid MCP_HTTP_RATE_LIMIT: {e}"))?;

    // ── 2. Require token ────────────────────────────────────────────────────
    require_auth(&auth_cfg).map_err(|e| anyhow::anyhow!("{e}"))?;

    // ── 3. Resolve TLS — fail-loud if env is set but config is invalid ───────
    let tls_cfg = tls_config_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;

    // ── 4. Build rmcp StreamableHttpService ─────────────────────────────────
    let daemon_arc: Arc<Mutex<DaemonClient>> = Arc::new(Mutex::new(daemon));
    let qdrant_arc: Arc<QdrantReadClient> = Arc::new(qdrant);
    let state_arc: Arc<SharedStateManager> = Arc::new(SharedStateManager::new(state));
    let session_arc: Arc<Mutex<SessionState>> = Arc::new(Mutex::new(session));
    // Default optimistic health state; a background health monitor can be wired
    // in later by passing an Arc<RwLock<HealthState>> from a StartedHealthMonitor.
    let health_arc: SharedHealthState = Arc::new(std::sync::RwLock::new(HealthState::initial()));

    let ct = shutdown_token.clone();
    let rmcp_cfg = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(ct.child_token())
        .with_allowed_hosts(vec![
            http_cfg.host.clone(),
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ]);

    let mcp_service: StreamableHttpService<ToolsHandler, LocalSessionManager> =
        StreamableHttpService::new(
            {
                let daemon_arc = Arc::clone(&daemon_arc);
                let qdrant_arc = Arc::clone(&qdrant_arc);
                let state_arc = Arc::clone(&state_arc);
                let session_arc = Arc::clone(&session_arc);
                move || {
                    Ok(ToolsHandler::from_arcs(
                        Arc::clone(&daemon_arc),
                        Arc::clone(&qdrant_arc),
                        Arc::clone(&state_arc),
                        Arc::clone(&session_arc),
                        Arc::clone(&health_arc),
                    ))
                }
            },
            Arc::new(LocalSessionManager::default()),
            rmcp_cfg,
        );

    // ── 5. Build middleware state ────────────────────────────────────────────
    let mw = MiddlewareState {
        auth: auth_cfg,
        cors: cors_cfg,
        limiter: Arc::new(SlidingWindowLimiter::with_config(rate_cfg)),
    };

    // ── 6. Assemble axum router ──────────────────────────────────────────────
    let mcp_path = http_cfg.mcp_path.clone();

    let router = Router::new()
        .route("/healthz", get(healthz_route))
        .fallback(move |req: Request<Body>| {
            let mw = mw.clone();
            let svc = mcp_service.clone();
            let path = mcp_path.clone();
            async move { handle_request(req, mw, svc, path).await }
        });

    // ── 7. Bind + serve ──────────────────────────────────────────────────────
    let bind_addr: std::net::SocketAddr = format!("{}:{}", http_cfg.host, http_cfg.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid bind address: {e}"))?;

    let make_svc = router.into_make_service_with_connect_info::<std::net::SocketAddr>();

    match tls_cfg {
        Some(ref cfg) => {
            // TLS env configured — terminate TLS via axum-server.
            let rustls_server_cfg = build_rustls_server_config(cfg)
                .map_err(|e| anyhow::anyhow!("TLS configuration error: {e}"))?;
            let rustls_cfg = RustlsConfig::from_config(rustls_server_cfg);

            info!(
                host = %http_cfg.host,
                port = http_cfg.port,
                path = %http_cfg.mcp_path,
                tls = true,
                "MCP HTTP transport listening (TLS)"
            );

            axum_server::bind_rustls(bind_addr, rustls_cfg)
                .handle(build_axum_server_handle(shutdown_token))
                .serve(make_svc)
                .await
                .map_err(|e| anyhow::anyhow!("MCP HTTPS serve error: {e}"))?;
        }
        None => {
            // No TLS env — plain TCP.
            info!(
                host = %http_cfg.host,
                port = http_cfg.port,
                path = %http_cfg.mcp_path,
                tls = false,
                "MCP HTTP transport listening (plaintext)"
            );

            axum_server::bind(bind_addr)
                .handle(build_axum_server_handle(shutdown_token))
                .serve(make_svc)
                .await
                .map_err(|e| anyhow::anyhow!("MCP HTTP serve error: {e}"))?;
        }
    }

    info!("MCP HTTP transport stopped");
    Ok(())
}

/// Build an `axum_server::Handle` that triggers graceful shutdown when the
/// cancellation token fires.
fn build_axum_server_handle(
    shutdown_token: CancellationToken,
) -> axum_server::Handle<std::net::SocketAddr> {
    let handle = axum_server::Handle::new();
    let h = handle.clone();
    tokio::spawn(async move {
        shutdown_token.cancelled().await;
        info!("MCP HTTP transport: shutdown signal received");
        h.graceful_shutdown(None);
    });
    handle
}

// ─────────────────────────────────────────────────────────────────────────────
// Route handlers
// ─────────────────────────────────────────────────────────────────────────────

/// Axum handler for `GET /healthz` — no auth, always 200.
async fn healthz_route() -> impl IntoResponse {
    healthz_response()
}

/// Middleware + dispatch for all non-`/healthz` paths.
///
/// Check order mirrors `mcp-http-server.ts`:
///   CORS/preflight → rate-limit → bearer auth → route (/mcp or 404)
async fn handle_request(
    req: Request<Body>,
    mw: MiddlewareState,
    mut mcp_svc: StreamableHttpService<ToolsHandler, LocalSessionManager>,
    mcp_path: String,
) -> Response<Body> {
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let path = parts.uri.path().to_string();
    let request_headers = parts.headers.clone();

    // ── 1. CORS / preflight ─────────────────────────────────────────────────
    let cors_outcome = apply_cors(&method, &parts.headers, &mw.cors);
    let origin_matched = match cors_outcome {
        CorsOutcome::Preflight(resp) => {
            record_http_request(Some(&path), resp.status().as_u16(), &mcp_path);
            return adapt_response(resp);
        }
        CorsOutcome::Continue { origin_matched } => origin_matched,
    };

    // ── 2. Rate-limit ────────────────────────────────────────────────────────
    if let Some(resp) = check_rate_limit(&parts, &mw.limiter, &path, &mcp_path) {
        return resp;
    }

    // ── 3. Bearer auth ───────────────────────────────────────────────────────
    if let Some(resp) = check_auth(&parts, &mw.auth, &path, &mcp_path) {
        return resp;
    }

    // ── 4. Route ─────────────────────────────────────────────────────────────
    // TS mcp-http-server.ts:192: `if (urlPath !== mcpPath)` — exact match only.
    // rmcp uses Mcp-Session-Id headers (not URL sub-paths) for session routing,
    // so the wildcard suffix is not required for rmcp session handling.
    let path_no_query = path.split('?').next().unwrap_or(&path);
    if path_no_query != mcp_path {
        record_http_request(Some(&path), 404, &mcp_path);
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .header("Content-Type", "text/plain")
            .body(Body::from("Not Found"))
            .expect("valid response");
    }

    // ── 5. Forward to rmcp ───────────────────────────────────────────────────
    forward_to_rmcp(
        Request::from_parts(parts, body),
        &mut mcp_svc,
        &path,
        &mcp_path,
        origin_matched,
        &request_headers,
    )
    .await
}

// ─────────────────────────────────────────────────────────────────────────────
// Middleware helpers (extracted to keep handle_request <80 lines)
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `Some(response)` when the rate limit is exceeded, `None` to continue.
fn check_rate_limit(
    parts: &axum::http::request::Parts,
    limiter: &SlidingWindowLimiter,
    path: &str,
    mcp_path: &str,
) -> Option<Response<Body>> {
    let client_ip = extract_client_ip(parts);
    if limiter.allow(&client_ip) {
        return None;
    }
    record_http_rate_limited();
    record_http_request(Some(path), 429, mcp_path);
    Some(
        Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("Content-Type", "text/plain")
            .header("Retry-After", "60")
            .body(Body::from("Too Many Requests"))
            .expect("valid response"),
    )
}

/// Returns `Some(response)` when auth fails, `None` to continue.
fn check_auth(
    parts: &axum::http::request::Parts,
    auth: &AuthConfig,
    path: &str,
    mcp_path: &str,
) -> Option<Response<Body>> {
    let auth_header = parts
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    match check_bearer(auth_header, auth) {
        BearerOutcome::Authorized => None,
        BearerOutcome::MissingHeader => {
            record_http_auth_failure("missing_header");
            record_http_request(Some(path), 401, mcp_path);
            Some(unauthorized_response(
                "Missing or malformed Authorization header",
            ))
        }
        BearerOutcome::InvalidToken => {
            record_http_auth_failure("invalid_token");
            record_http_request(Some(path), 401, mcp_path);
            Some(unauthorized_response("Invalid token"))
        }
        BearerOutcome::NotConfigured => {
            record_http_auth_failure("not_configured");
            record_http_request(Some(path), 401, mcp_path);
            Some(unauthorized_response(
                "Server is not configured for authentication",
            ))
        }
    }
}

/// Forward the request to rmcp and convert the BoxBody response to axum Body.
async fn forward_to_rmcp(
    http_req: Request<Body>,
    mcp_svc: &mut StreamableHttpService<ToolsHandler, LocalSessionManager>,
    path: &str,
    mcp_path: &str,
    origin_matched: bool,
    request_headers: &axum::http::HeaderMap,
) -> Response<Body> {
    use tower_service::Service;
    let rmcp_resp = mcp_svc.call(http_req).await;
    let status = rmcp_resp
        .as_ref()
        .map(|r| r.status().as_u16())
        .unwrap_or(500);
    record_http_request(Some(path), status, mcp_path);

    match rmcp_resp {
        Ok(r) => {
            use http_body_util::BodyExt;
            let (resp_parts, resp_body) = r.into_parts();
            let axum_body = Body::new(resp_body.map_err(|e| std::io::Error::other(e.to_string())));
            let mut resp = Response::from_parts(resp_parts, axum_body);
            if origin_matched {
                add_cors_response_headers(resp.headers_mut(), request_headers);
            }
            resp
        }
        Err(_infallible) => {
            // StreamableHttpService::Error = Infallible; unreachable.
            Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Body::from("Internal Server Error"))
                .expect("valid response")
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Adapt an axum `Response<axum::body::Body>` — pass-through kept for clarity.
fn adapt_response(resp: Response<axum::body::Body>) -> Response<Body> {
    resp
}

/// Extract the client IP from the socket peer address injected by axum's
/// `ConnectInfo` extension.
///
/// Uses **only** the socket address — X-Forwarded-For is NOT read here because
/// it is client-supplied and trivially spoofable for rate-limit bypass.
/// Mirrors `mcp-http-server.ts` which uses only the socket address
/// (`auth-middleware.ts:194`).
///
/// Strips the IPv4-mapped `::ffff:` prefix so dual-stack and pure-v4 sockets
/// share the same rate-limit bucket (mirrors `.replace(/^::ffff:/, '')` in TS).
fn extract_client_ip(parts: &axum::http::request::Parts) -> String {
    if let Some(addr) = parts.extensions.get::<ConnectInfo<std::net::SocketAddr>>() {
        let ip = addr.0.ip().to_string();
        // Strip IPv4-mapped prefix so ::ffff:1.2.3.4 and 1.2.3.4 share a bucket.
        return ip.strip_prefix("::ffff:").unwrap_or(&ip).to_string();
    }
    "unknown".to_string()
}

/// Build a 401 Unauthorized response.
fn unauthorized_response(msg: &'static str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("Content-Type", "text/plain")
        .header("WWW-Authenticate", r#"Bearer realm="workspace-qdrant-mcp""#)
        .body(Body::from(msg))
        .expect("valid response")
}

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
