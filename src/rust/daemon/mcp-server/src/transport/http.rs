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
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::info;

use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};

use tokio::sync::Mutex;

use crate::grpc::client::DaemonClient;
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
/// 3. Binds the TCP listener on `host:port`.
/// 4. Drives the serve loop until `shutdown_token` is cancelled.
///
/// # Errors
///
/// Returns `Err` when the token is missing/short, the bind fails, or the
/// serve loop encounters a fatal error.
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

    // ── 3. Build rmcp StreamableHttpService ─────────────────────────────────
    //
    // `DaemonClient` and `StateManager` are not Clone, so we wrap them in
    // Arc<Mutex<_>> / Arc<SharedStateManager> once and share the same Arc
    // across every per-session ToolsHandler created by the factory closure.
    let daemon_arc: Arc<Mutex<DaemonClient>> = Arc::new(Mutex::new(daemon));
    let qdrant_arc: Arc<QdrantReadClient> = Arc::new(qdrant);
    let state_arc: Arc<SharedStateManager> = Arc::new(SharedStateManager::new(state));
    let session_arc: Arc<Mutex<SessionState>> = Arc::new(Mutex::new(session));

    let ct = shutdown_token.clone();
    let rmcp_cfg = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(ct.child_token())
        // Allow the configured host so rmcp's DNS-rebinding guard passes.
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
                    // Each session gets a ToolsHandler that shares the same
                    // underlying Arcs — no per-session copies of non-Clone types.
                    Ok(ToolsHandler::from_arcs(
                        Arc::clone(&daemon_arc),
                        Arc::clone(&qdrant_arc),
                        Arc::clone(&state_arc),
                        Arc::clone(&session_arc),
                    ))
                }
            },
            Arc::new(LocalSessionManager::default()),
            rmcp_cfg,
        );

    // ── 4. Build middleware state ────────────────────────────────────────────
    let mw = MiddlewareState {
        auth: auth_cfg,
        cors: cors_cfg,
        limiter: Arc::new(SlidingWindowLimiter::with_config(rate_cfg)),
    };

    // ── 5. Assemble axum router ──────────────────────────────────────────────
    //
    // Route structure:
    //   GET  /healthz  → unauthenticated liveness probe
    //   *    <other>   → middleware_handler (CORS → rate-limit → auth → MCP or 404)
    //
    // The MCP service is a tower::Service; axum's `nest_service` mounts it.
    // We use a fallback for all non-healthz paths so the middleware runs first
    // and either forwards to the mcp_service or returns 404.
    let mcp_path = http_cfg.mcp_path.clone();

    let router = Router::new()
        .route("/healthz", get(healthz_route))
        .fallback(move |req: Request<Body>| {
            let mw = mw.clone();
            let svc = mcp_service.clone();
            let path = mcp_path.clone();
            async move { handle_request(req, mw, svc, path).await }
        });

    // ── 6. Bind listener ────────────────────────────────────────────────────
    let bind_addr = format!("{}:{}", http_cfg.host, http_cfg.port);
    let listener = TcpListener::bind(&bind_addr)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to bind MCP HTTP transport on {bind_addr}: {e}"))?;

    info!(
        host = %http_cfg.host,
        port = http_cfg.port,
        path = %http_cfg.mcp_path,
        "MCP HTTP transport listening"
    );

    // ── 7. Serve until shutdown ──────────────────────────────────────────────
    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        shutdown_token.cancelled().await;
        info!("MCP HTTP transport: shutdown signal received");
    })
    .await
    .map_err(|e| anyhow::anyhow!("MCP HTTP serve error: {e}"))?;

    info!("MCP HTTP transport stopped");
    Ok(())
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
    match cors_outcome {
        CorsOutcome::Preflight(resp) => {
            record_http_request(Some(&path), resp.status().as_u16(), &mcp_path);
            return adapt_response(resp);
        }
        CorsOutcome::Continue { origin_matched } => {
            // ── 2. Rate-limit ────────────────────────────────────────────────
            let client_ip = extract_client_ip(&parts);
            if !mw.limiter.allow(&client_ip) {
                record_http_rate_limited();
                record_http_request(Some(&path), 429, &mcp_path);
                let resp = Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .header("Content-Type", "text/plain")
                    .header("Retry-After", "60")
                    .body(Body::from("Too Many Requests"))
                    .expect("valid response");
                return resp;
            }

            // ── 3. Bearer auth ───────────────────────────────────────────────
            let auth_header = parts
                .headers
                .get("authorization")
                .and_then(|v| v.to_str().ok());
            let auth_outcome = check_bearer(auth_header, &mw.auth);
            match auth_outcome {
                BearerOutcome::Authorized => {}
                BearerOutcome::MissingHeader => {
                    record_http_auth_failure("missing_header");
                    record_http_request(Some(&path), 401, &mcp_path);
                    return unauthorized_response("Missing or malformed Authorization header");
                }
                BearerOutcome::InvalidToken => {
                    record_http_auth_failure("invalid_token");
                    record_http_request(Some(&path), 401, &mcp_path);
                    return unauthorized_response("Invalid token");
                }
            }

            // ── 4. Route ─────────────────────────────────────────────────────
            let path_no_query = path.split('?').next().unwrap_or(&path);
            if path_no_query != mcp_path && !path_no_query.starts_with(&format!("{mcp_path}/")) {
                record_http_request(Some(&path), 404, &mcp_path);
                return Response::builder()
                    .status(StatusCode::NOT_FOUND)
                    .header("Content-Type", "text/plain")
                    .body(Body::from("Not Found"))
                    .expect("valid response");
            }

            // ── 5. Forward to rmcp ───────────────────────────────────────────
            let http_req = Request::from_parts(parts, body);
            // rmcp's StreamableHttpService returns BoxBody — convert to Body.
            use tower_service::Service;
            let rmcp_resp = mcp_svc.call(http_req).await;
            let status = rmcp_resp
                .as_ref()
                .map(|r| r.status().as_u16())
                .unwrap_or(500);
            record_http_request(Some(&path), status, &mcp_path);

            let resp = match rmcp_resp {
                Ok(r) => {
                    // Convert BoxBody to axum Body.
                    use http_body_util::BodyExt;
                    let (resp_parts, resp_body) = r.into_parts();
                    let axum_body = Body::new(resp_body.map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
                    }));
                    let mut resp = Response::from_parts(resp_parts, axum_body);
                    // Apply CORS response headers when origin matched.
                    if origin_matched {
                        add_cors_response_headers(resp.headers_mut(), &request_headers);
                    }
                    resp
                }
                Err(_infallible) => {
                    // StreamableHttpService::Error = Infallible; this arm is unreachable.
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from("Internal Server Error"))
                        .expect("valid response")
                }
            };
            resp
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Adapt an axum `Response<axum::body::Body>` from a `Response<axum::body::Body>`.
///
/// Both types are the same here; this is a no-op pass-through kept as a named
/// function for clarity.
fn adapt_response(resp: Response<axum::body::Body>) -> Response<Body> {
    resp
}

/// Extract a best-effort client IP from request parts.
///
/// Checks `X-Forwarded-For` first (proxy deployments), then falls back to the
/// socket peer address injected by axum's `ConnectInfo` extension.
fn extract_client_ip(parts: &axum::http::request::Parts) -> String {
    // X-Forwarded-For (first entry = original client IP)
    if let Some(xff) = parts
        .headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
    {
        if let Some(first) = xff.split(',').next() {
            let ip = first.trim().to_string();
            if !ip.is_empty() {
                return ip;
            }
        }
    }
    // axum ConnectInfo (set by into_make_service_with_connect_info)
    if let Some(addr) = parts.extensions.get::<ConnectInfo<std::net::SocketAddr>>() {
        return addr.0.ip().to_string();
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
