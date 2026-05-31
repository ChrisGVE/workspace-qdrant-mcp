//! Tests for the HTTP transport middleware layer.
//!
//! These tests exercise the CORS→rate-limit→auth→route check order using a
//! minimal in-process axum router without wiring up the real rmcp service.

use std::sync::Arc;

use axum::{
    body::{to_bytes, Body},
    http::{Request, Response, StatusCode},
    routing::get,
    Router,
};
use tower::ServiceExt;

use crate::transport::auth::{check_bearer, AuthConfig, BearerOutcome};
use crate::transport::cors::{apply_cors, CorsConfig, CorsOutcome};
use crate::transport::rate_limit::{RateLimitConfig, SlidingWindowLimiter};

use super::{
    adapt_response, extract_client_ip, healthz_route, unauthorized_response, MiddlewareState,
};

const TEST_TOKEN: &str = "test-token-1234567890abcdef"; // 28 chars

/// Build a minimal test router without the real rmcp service.
///
/// A stub `middleware_only_handler` applies the full CORS→rate-limit→auth→route
/// pipeline and returns 200 for MCP path hits instead of calling rmcp.
fn make_test_router(token: Option<&str>, rate_limit: u32) -> Router {
    let auth = AuthConfig::new(token.map(str::to_string));
    let cors = CorsConfig::default();
    let limiter = Arc::new(SlidingWindowLimiter::new(
        RateLimitConfig {
            max_per_window: rate_limit,
        },
        std::time::Duration::from_secs(60),
    ));
    let mcp_path = "/mcp".to_string();

    let mw = MiddlewareState {
        auth,
        cors,
        limiter,
    };

    Router::new()
        .route("/healthz", get(healthz_route))
        .fallback(move |req: Request<Body>| {
            let mw = mw.clone();
            let path = mcp_path.clone();
            async move { middleware_only_handler(req, mw, path).await }
        })
}

/// Simplified handler: applies CORS→rate-limit→auth→route, returns 200 on MCP hit.
async fn middleware_only_handler(
    req: Request<Body>,
    mw: MiddlewareState,
    mcp_path: String,
) -> Response<Body> {
    let (parts, _body) = req.into_parts();
    let method = parts.method.clone();
    let path = parts.uri.path().to_string();

    // 1. CORS
    let cors_outcome = apply_cors(&method, &parts.headers, &mw.cors);
    if let CorsOutcome::Preflight(resp) = cors_outcome {
        return adapt_response(resp);
    }

    // 2. Rate-limit
    let client_ip = extract_client_ip(&parts);
    if !mw.limiter.allow(&client_ip) {
        return Response::builder()
            .status(StatusCode::TOO_MANY_REQUESTS)
            .header("Retry-After", "60")
            .body(Body::from("Too Many Requests"))
            .unwrap();
    }

    // 3. Auth
    let auth_header = parts
        .headers
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    match check_bearer(auth_header, &mw.auth) {
        BearerOutcome::Authorized => {}
        BearerOutcome::MissingHeader => {
            return unauthorized_response("Missing or malformed Authorization header");
        }
        BearerOutcome::InvalidToken => {
            return unauthorized_response("Invalid token");
        }
        BearerOutcome::NotConfigured => {
            return unauthorized_response("Server is not configured for authentication");
        }
    }

    // 4. Route
    let path_no_query = path.split('?').next().unwrap_or(&path);
    if path_no_query != mcp_path {
        return Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap();
    }

    Response::builder()
        .status(StatusCode::OK)
        .body(Body::from("mcp-reached"))
        .unwrap()
}

// ── /healthz bypasses auth ───────────────────────────────────────────────────

#[tokio::test]
async fn healthz_no_auth_returns_200() {
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("GET")
        .uri("/healthz")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = to_bytes(resp.into_body(), 64).await.unwrap();
    assert_eq!(&body[..], b"ok");
}

// ── Rate-limit fires before auth ─────────────────────────────────────────────

#[tokio::test]
async fn rate_limit_before_auth() {
    // Limit = 1: first request passes rate-limit (no auth → 401),
    // second request should hit rate-limit (429) even without any auth header.
    let app = make_test_router(Some(TEST_TOKEN), 1);

    // First request: passes rate-limit but fails auth (no header) → 401
    let req1 = Request::builder()
        .method("POST")
        .uri("/mcp")
        .body(Body::empty())
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(
        resp1.status(),
        StatusCode::UNAUTHORIZED,
        "first request should hit auth"
    );

    // Second request: rate-limited → 429 (before auth check)
    let req2 = Request::builder()
        .method("POST")
        .uri("/mcp")
        .body(Body::empty())
        .unwrap();
    let resp2 = app.oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request should be rate-limited"
    );
}

// ── 404 for unknown path ─────────────────────────────────────────────────────

#[tokio::test]
async fn unknown_path_returns_404() {
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("GET")
        .uri("/unknown")
        .header("Authorization", format!("Bearer {TEST_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ── Missing token → 401 ──────────────────────────────────────────────────────

#[tokio::test]
async fn missing_auth_header_returns_401() {
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ── Wrong token → 401 ───────────────────────────────────────────────────────

#[tokio::test]
async fn wrong_token_returns_401() {
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("Authorization", "Bearer wrongtoken")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

// ── MUST-FIX 2: Rate-limit uses socket IP, not X-Forwarded-For ───────────────

/// Two requests with different spoofed XFF headers but the same socket address
/// (both have no ConnectInfo extension → "unknown") share one rate-limit bucket.
/// The limiter must enforce the limit using the socket key, not the XFF value.
#[tokio::test]
async fn rate_limit_ignores_xff_uses_socket_addr() {
    // Limit = 1 so the second request from the same effective IP is denied.
    let auth = AuthConfig::new(Some(TEST_TOKEN.to_string()));
    let cors = CorsConfig::default();
    let limiter = Arc::new(SlidingWindowLimiter::new(
        RateLimitConfig { max_per_window: 1 },
        std::time::Duration::from_secs(60),
    ));
    let mcp_path = "/mcp".to_string();
    let mw = MiddlewareState {
        auth,
        cors,
        limiter,
    };

    let app = Router::new()
        .route("/healthz", get(healthz_route))
        .fallback(move |req: Request<Body>| {
            let mw = mw.clone();
            let path = mcp_path.clone();
            async move { middleware_only_handler(req, mw, path).await }
        });

    // First request: unique spoofed XFF — passes rate-limit (→ 401 no auth).
    let req1 = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("X-Forwarded-For", "10.0.0.1")
        .body(Body::empty())
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(
        resp1.status(),
        StatusCode::UNAUTHORIZED,
        "first request: should pass rate-limit (hit auth)"
    );

    // Second request: different spoofed XFF — must STILL be rate-limited
    // because the socket addr ("unknown") is the same.
    let req2 = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("X-Forwarded-For", "10.0.0.2") // different IP in XFF
        .body(Body::empty())
        .unwrap();
    let resp2 = app.clone().oneshot(req2).await.unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request: different XFF but same socket → must be rate-limited"
    );
}

/// extract_client_ip with no ConnectInfo extension returns "unknown".
/// Two calls with no extension share the same "unknown" key.
#[test]
fn extract_client_ip_no_connect_info_returns_unknown() {
    let req = Request::builder().uri("/mcp").body(()).unwrap();
    let (parts, _) = req.into_parts();
    assert_eq!(extract_client_ip(&parts), "unknown");
}

/// extract_client_ip strips the IPv4-mapped ::ffff: prefix so dual-stack
/// and v4 sockets share the same rate-limit bucket.
#[test]
fn extract_client_ip_strips_ipv4_mapped_prefix() {
    // Simulate ConnectInfo carrying ::ffff:1.2.3.4 (dual-stack listener).
    use axum::extract::ConnectInfo;
    use std::net::{IpAddr, Ipv6Addr, SocketAddr};

    // ::ffff:1.2.3.4 as a SocketAddrV6
    let ipv4_mapped: IpAddr = IpAddr::V6(Ipv6Addr::new(0, 0, 0, 0, 0, 0xffff, 0x0102, 0x0304));
    let socket_addr = SocketAddr::new(ipv4_mapped, 12345);

    let mut req = Request::builder().uri("/mcp").body(()).unwrap();
    req.extensions_mut()
        .insert(ConnectInfo::<SocketAddr>(socket_addr));
    let (parts, _) = req.into_parts();

    let ip = extract_client_ip(&parts);
    // Must be the plain v4 address, not the ::ffff: form.
    assert_eq!(ip, "1.2.3.4", "::ffff: prefix must be stripped; got: {ip}");
}

/// A pure IPv4 socket address must not be mangled.
#[test]
fn extract_client_ip_pure_v4_unchanged() {
    use axum::extract::ConnectInfo;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    let socket_addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(1, 2, 3, 4)), 5678);
    let mut req = Request::builder().uri("/mcp").body(()).unwrap();
    req.extensions_mut()
        .insert(ConnectInfo::<SocketAddr>(socket_addr));
    let (parts, _) = req.into_parts();

    let ip = extract_client_ip(&parts);
    assert_eq!(ip, "1.2.3.4");
}

// ── SHOULD-FIX 3: not_configured vs invalid_token ────────────────────────────

/// When the server has no token configured and the client sends a Bearer
/// header, the response body must be 'Server is not configured for
/// authentication' (mirrors auth-middleware.ts:163-166).
#[tokio::test]
async fn bearer_with_no_server_token_returns_not_configured_body() {
    // token = None → server is unconfigured.
    let app = make_test_router(None, 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("Authorization", "Bearer some-token-abcdefgh")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = to_bytes(resp.into_body(), 256).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert_eq!(
        body_str, "Server is not configured for authentication",
        "body must distinguish not_configured from invalid_token; got: {body_str}"
    );
}

/// When the server HAS a token configured and the client sends the wrong one,
/// the response body must be 'Invalid token' (not the not_configured message).
#[tokio::test]
async fn bearer_wrong_token_returns_invalid_token_body() {
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("Authorization", "Bearer wrong-token-value-here")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    let body = to_bytes(resp.into_body(), 256).await.unwrap();
    let body_str = std::str::from_utf8(&body).unwrap();
    assert_eq!(
        body_str, "Invalid token",
        "wrong token must return 'Invalid token', got: {body_str}"
    );
}

// ── Exact-path routing (TS parity for finding #9) ─────────────────────────

#[tokio::test]
async fn mcp_subpath_returns_404() {
    // TS mcp-http-server.ts:192: `if (urlPath !== mcpPath)` — exact match only.
    // A request to /mcp/anything must be 404'd, not forwarded to rmcp.
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp/session-id-abc")
        .header("Authorization", format!("Bearer {TEST_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "sub-path /mcp/session-id-abc must return 404 (exact match only)"
    );
}

#[tokio::test]
async fn mcp_exact_path_passes_auth_returns_200() {
    // Sanity: the exact /mcp path must still be accepted.
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("POST")
        .uri("/mcp")
        .header("Authorization", format!("Bearer {TEST_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "/mcp exact path must return 200"
    );
}

#[tokio::test]
async fn mcp_subpath_with_query_returns_404() {
    // /mcp/foo?bar must also be 404 — query-stripping happens before path check.
    let app = make_test_router(Some(TEST_TOKEN), 100);
    let req = Request::builder()
        .method("GET")
        .uri("/mcp/foo?session=1")
        .header("Authorization", format!("Bearer {TEST_TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "/mcp/foo?session=1 must return 404"
    );
}
