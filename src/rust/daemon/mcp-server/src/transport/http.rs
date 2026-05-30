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
use crate::observability::metrics::record_http_request;
use crate::qdrant::client::QdrantReadClient;
use crate::server_types::SessionState;
use crate::sqlite::{SharedStateManager, StateManager};
use crate::tools::ToolsHandler;
use crate::transport::auth::{require_auth, AuthConfig};
use crate::transport::cors::{apply_cors, CorsConfig, CorsOutcome};
use crate::transport::rate_limit::{RateLimitConfig, SlidingWindowLimiter};
use crate::transport::tls::{build_rustls_server_config, tls_config_from_env};

use super::health::healthz_response;

mod http_middleware;
use http_middleware::{
    adapt_response, check_auth, check_rate_limit, extract_client_ip, forward_to_rmcp,
    unauthorized_response,
};

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

    // ── 3. Resolve TLS ───────────────────────────────────────────────────────
    let tls_cfg = tls_config_from_env().map_err(|e| anyhow::anyhow!("{e}"))?;

    // ── 4-5. Build rmcp service + middleware ─────────────────────────────────
    let (mcp_service, mw) = build_mcp_service(
        daemon,
        qdrant,
        state,
        session,
        &http_cfg,
        auth_cfg,
        cors_cfg,
        rate_cfg,
        &shutdown_token,
    );

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
    bind_and_serve(router, &http_cfg, tls_cfg, shutdown_token).await?;
    info!("MCP HTTP transport stopped");
    Ok(())
}

/// Build the rmcp `StreamableHttpService` and `MiddlewareState`.
///
/// Extracted from `serve_http` for size compliance.
fn build_mcp_service(
    daemon: DaemonClient,
    qdrant: QdrantReadClient,
    state: StateManager,
    session: SessionState,
    http_cfg: &HttpConfig,
    auth_cfg: AuthConfig,
    cors_cfg: CorsConfig,
    rate_cfg: RateLimitConfig,
    shutdown_token: &CancellationToken,
) -> (
    StreamableHttpService<ToolsHandler, LocalSessionManager>,
    MiddlewareState,
) {
    let daemon_arc: Arc<Mutex<DaemonClient>> = Arc::new(Mutex::new(daemon));
    let qdrant_arc: Arc<QdrantReadClient> = Arc::new(qdrant);
    let state_arc: Arc<SharedStateManager> = Arc::new(SharedStateManager::new(state));
    let session_arc: Arc<Mutex<SessionState>> = Arc::new(Mutex::new(session));
    let health_arc: SharedHealthState = Arc::new(std::sync::RwLock::new(HealthState::initial()));

    let rmcp_cfg = StreamableHttpServerConfig::default()
        .with_stateful_mode(true)
        .with_cancellation_token(shutdown_token.child_token())
        .with_allowed_hosts(vec![
            http_cfg.host.clone(),
            "localhost".to_string(),
            "127.0.0.1".to_string(),
        ]);

    let mcp_service = StreamableHttpService::new(
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

    let mw = MiddlewareState {
        auth: auth_cfg,
        cors: cors_cfg,
        limiter: Arc::new(SlidingWindowLimiter::with_config(rate_cfg)),
    };

    (mcp_service, mw)
}

/// Bind the TCP listener and drive the serve loop.
///
/// Extracted from `serve_http` for size compliance.
async fn bind_and_serve(
    router: Router,
    http_cfg: &HttpConfig,
    tls_cfg: Option<crate::transport::tls::TlsConfig>,
    shutdown_token: CancellationToken,
) -> anyhow::Result<()> {
    let bind_addr: std::net::SocketAddr = format!("{}:{}", http_cfg.host, http_cfg.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid bind address: {e}"))?;

    let make_svc = router.into_make_service_with_connect_info::<std::net::SocketAddr>();

    match tls_cfg {
        Some(ref cfg) => {
            let rustls_server_cfg = build_rustls_server_config(cfg)
                .map_err(|e| anyhow::anyhow!("TLS configuration error: {e}"))?;
            let rustls_cfg = RustlsConfig::from_config(rustls_server_cfg);
            info!(
                host = %http_cfg.host, port = http_cfg.port,
                path = %http_cfg.mcp_path, tls = true,
                "MCP HTTP transport listening (TLS)"
            );
            axum_server::bind_rustls(bind_addr, rustls_cfg)
                .handle(build_axum_server_handle(shutdown_token))
                .serve(make_svc)
                .await
                .map_err(|e| anyhow::anyhow!("MCP HTTPS serve error: {e}"))?;
        }
        None => {
            info!(
                host = %http_cfg.host, port = http_cfg.port,
                path = %http_cfg.mcp_path, tls = false,
                "MCP HTTP transport listening (plaintext)"
            );
            axum_server::bind(bind_addr)
                .handle(build_axum_server_handle(shutdown_token))
                .serve(make_svc)
                .await
                .map_err(|e| anyhow::anyhow!("MCP HTTP serve error: {e}"))?;
        }
    }
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

#[cfg(test)]
#[path = "http_tests.rs"]
mod tests;
