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
