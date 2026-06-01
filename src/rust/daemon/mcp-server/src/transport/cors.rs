//! CORS preflight and response-header handling for the HTTP transport.
//!
//! Mirrors the CORS logic in `src/typescript/mcp-server/src/auth-middleware.ts`
//! (`applyCors`, lines 121-147).
//!
//! # Behaviour
//!
//! - Env `MCP_HTTP_CORS_ORIGINS` (comma-separated list of allowed origins).
//! - Empty / unset → CORS disabled; no CORS headers are ever emitted.
//! - Matching `Origin` → echo it back with `Vary: Origin`,
//!   `Access-Control-Allow-Credentials: true`.
//! - `OPTIONS` preflight → 204 with Allow-Methods / Allow-Headers / Max-Age,
//!   regardless of whether the origin matched.
//! - Non-matching origin → no CORS headers (browser blocks access).

use axum::{
    body::Body,
    http::{
        header::{
            ACCESS_CONTROL_ALLOW_CREDENTIALS, ACCESS_CONTROL_ALLOW_HEADERS,
            ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN, ORIGIN, VARY,
        },
        HeaderMap, HeaderValue, Method, Response, StatusCode,
    },
};

/// CORS configuration parsed from environment variables.
#[derive(Debug, Clone, Default)]
pub struct CorsConfig {
    /// Allowed origins. Empty → CORS disabled.
    pub origins: Vec<String>,
}

impl CorsConfig {
    /// Parse from the `MCP_HTTP_CORS_ORIGINS` environment variable.
    ///
    /// Trims whitespace and drops empty entries, matching:
    /// ```text
    /// corsRaw.split(',').map(s => s.trim()).filter(s => s.length > 0)
    /// ```
    /// in `auth-middleware.ts:84-88`.
    pub fn from_env() -> Self {
        let raw = std::env::var("MCP_HTTP_CORS_ORIGINS").unwrap_or_default();
        Self::from_str(&raw)
    }

    /// Parse from a raw comma-separated string (extracted for testability).
    pub fn from_str(raw: &str) -> Self {
        let origins = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        Self { origins }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-request CORS logic
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of [`apply_cors`].
pub enum CorsOutcome {
    /// Caller should continue processing the request.
    Continue {
        /// Whether a matching CORS origin was found (affects response headers).
        origin_matched: bool,
    },
    /// OPTIONS preflight — the response is fully built; caller must return it.
    Preflight(Response<Body>),
}

/// Apply CORS headers and handle OPTIONS preflight.
///
/// Mirrors `applyCors()` in `auth-middleware.ts:121-147`.
///
/// Call this **before** rate-limit and auth checks (matching TS check order).
pub fn apply_cors(method: &Method, headers: &HeaderMap, config: &CorsConfig) -> CorsOutcome {
    let origin = headers
        .get(ORIGIN)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    let origin_matched = !config.origins.is_empty() && config.origins.contains(&origin);

    if method == Method::OPTIONS {
        // OPTIONS preflight → 204, always end processing.
        let mut resp = Response::builder()
            .status(StatusCode::NO_CONTENT)
            .body(Body::empty())
            .expect("valid response");

        if origin_matched {
            insert_cors_headers(resp.headers_mut(), &origin);
            resp.headers_mut().insert(
                ACCESS_CONTROL_ALLOW_METHODS,
                HeaderValue::from_static("GET, POST, DELETE, OPTIONS"),
            );
            resp.headers_mut().insert(
                ACCESS_CONTROL_ALLOW_HEADERS,
                HeaderValue::from_static("Authorization, Content-Type, Accept, Mcp-Session-Id"),
            );
            resp.headers_mut().insert(
                axum::http::header::HeaderName::from_static("access-control-max-age"),
                HeaderValue::from_static("600"),
            );
        }

        return CorsOutcome::Preflight(resp);
    }

    CorsOutcome::Continue { origin_matched }
}

/// Add the origin-echo CORS response headers to an existing [`HeaderMap`].
///
/// Called by the HTTP layer after the MCP handler responds, when
/// `origin_matched` is `true`.
pub fn add_cors_response_headers(headers: &mut HeaderMap, request_headers: &HeaderMap) {
    let origin = request_headers
        .get(ORIGIN)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    if !origin.is_empty() {
        insert_cors_headers(headers, &origin);
    }
}

fn insert_cors_headers(headers: &mut HeaderMap, origin: &str) {
    if let Ok(v) = HeaderValue::from_str(origin) {
        headers.insert(ACCESS_CONTROL_ALLOW_ORIGIN, v);
    }
    headers.insert(VARY, HeaderValue::from_static("Origin"));
    headers.insert(
        ACCESS_CONTROL_ALLOW_CREDENTIALS,
        HeaderValue::from_static("true"),
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "cors_tests.rs"]
mod tests;
