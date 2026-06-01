//! `/healthz` health-check handler for the HTTP transport.
//!
//! `GET /healthz` is intentionally **unauthenticated** — it must be reachable
//! by load-balancer probes without a bearer token. All other routes pass
//! through auth first.
//!
//! Response: `200 OK`, `Content-Type: text/plain`, body `ok`.

use axum::{
    body::Body,
    http::{Response, StatusCode},
    response::IntoResponse,
};

/// Build a 200 OK response with body `ok`.
///
/// Called directly from the axum route handler without auth.
pub fn healthz_response() -> impl IntoResponse {
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/plain")
        .body(Body::from("ok"))
        .expect("valid health response")
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn healthz_returns_200_ok_body() {
        let resp = healthz_response().into_response();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 1024).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn healthz_content_type_text_plain() {
        let resp = healthz_response().into_response();
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.contains("text/plain"), "content-type was: {ct}");
    }
}
