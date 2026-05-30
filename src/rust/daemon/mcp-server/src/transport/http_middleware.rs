//! Per-request middleware helpers for the streamable-HTTP transport.
//!
//! Extracted from `http.rs` for size compliance.  All items are
//! `pub(super)` — they form the internal implementation of `http.rs`.

use axum::{
    body::Body,
    extract::ConnectInfo,
    http::{Response, StatusCode},
};

use crate::observability::metrics::{
    record_http_auth_failure, record_http_rate_limited, record_http_request,
};
use crate::transport::auth::{check_bearer, AuthConfig, BearerOutcome};
use crate::transport::cors::{add_cors_response_headers, CorsConfig};
use crate::transport::rate_limit::SlidingWindowLimiter;

use crate::tools::ToolsHandler;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpService,
};

// ─────────────────────────────────────────────────────────────────────────────
// Middleware helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `Some(response)` when the rate limit is exceeded, `None` to continue.
pub(super) fn check_rate_limit(
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
pub(super) fn check_auth(
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
pub(super) async fn forward_to_rmcp(
    http_req: axum::http::Request<Body>,
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

/// Adapt an axum `Response<axum::body::Body>` — pass-through kept for clarity.
pub(super) fn adapt_response(resp: Response<axum::body::Body>) -> Response<Body> {
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
pub(super) fn extract_client_ip(parts: &axum::http::request::Parts) -> String {
    if let Some(addr) = parts.extensions.get::<ConnectInfo<std::net::SocketAddr>>() {
        let ip = addr.0.ip().to_string();
        // Strip IPv4-mapped prefix so ::ffff:1.2.3.4 and 1.2.3.4 share a bucket.
        return ip.strip_prefix("::ffff:").unwrap_or(&ip).to_string();
    }
    "unknown".to_string()
}

/// Build a 401 Unauthorized response.
pub(super) fn unauthorized_response(msg: &'static str) -> Response<Body> {
    Response::builder()
        .status(StatusCode::UNAUTHORIZED)
        .header("Content-Type", "text/plain")
        .header("WWW-Authenticate", r#"Bearer realm="workspace-qdrant-mcp""#)
        .body(Body::from(msg))
        .expect("valid response")
}
