//! Tests for CORS handling (transport/cors.rs).

use axum::http::{HeaderMap, HeaderValue, Method};

use super::*;

fn config(origins: &[&str]) -> CorsConfig {
    CorsConfig {
        origins: origins.iter().map(|s| s.to_string()).collect(),
    }
}

fn headers_with_origin(origin: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(ORIGIN, HeaderValue::from_str(origin).unwrap());
    h
}

// ─────────────────────────────────────────────────────────────────────────────
// CorsConfig parsing
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn config_empty_string_gives_empty_origins() {
    let c = CorsConfig::from_str("");
    assert!(c.origins.is_empty());
}

#[test]
fn config_single_origin() {
    let c = CorsConfig::from_str("https://app.example.com");
    assert_eq!(c.origins, vec!["https://app.example.com"]);
}

#[test]
fn config_multiple_origins_with_spaces() {
    let c = CorsConfig::from_str(" https://a.com , https://b.com ");
    assert_eq!(c.origins, vec!["https://a.com", "https://b.com"]);
}

#[test]
fn config_skips_empty_segments() {
    let c = CorsConfig::from_str("https://a.com,,https://b.com");
    assert_eq!(c.origins, vec!["https://a.com", "https://b.com"]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Non-preflight: matching origin → Continue(origin_matched=true)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn matching_origin_returns_continue_matched() {
    let cfg = config(&["https://app.example.com"]);
    let h = headers_with_origin("https://app.example.com");
    match apply_cors(&Method::POST, &h, &cfg) {
        CorsOutcome::Continue { origin_matched } => assert!(origin_matched),
        CorsOutcome::Preflight(_) => panic!("unexpected preflight"),
    }
}

#[test]
fn non_matching_origin_returns_continue_not_matched() {
    let cfg = config(&["https://allowed.com"]);
    let h = headers_with_origin("https://other.com");
    match apply_cors(&Method::POST, &h, &cfg) {
        CorsOutcome::Continue { origin_matched } => assert!(!origin_matched),
        CorsOutcome::Preflight(_) => panic!("unexpected preflight"),
    }
}

#[test]
fn no_origin_header_returns_continue_not_matched() {
    let cfg = config(&["https://app.example.com"]);
    let h = HeaderMap::new();
    match apply_cors(&Method::GET, &h, &cfg) {
        CorsOutcome::Continue { origin_matched } => assert!(!origin_matched),
        CorsOutcome::Preflight(_) => panic!("unexpected preflight"),
    }
}

#[test]
fn cors_disabled_returns_continue_not_matched() {
    let cfg = config(&[]); // no allowed origins
    let h = headers_with_origin("https://any.com");
    match apply_cors(&Method::POST, &h, &cfg) {
        CorsOutcome::Continue { origin_matched } => assert!(!origin_matched),
        CorsOutcome::Preflight(_) => panic!("unexpected preflight"),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OPTIONS preflight → 204
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn options_preflight_returns_204() {
    let cfg = config(&["https://app.example.com"]);
    let h = headers_with_origin("https://app.example.com");
    match apply_cors(&Method::OPTIONS, &h, &cfg) {
        CorsOutcome::Preflight(resp) => {
            assert_eq!(resp.status(), axum::http::StatusCode::NO_CONTENT);
        }
        CorsOutcome::Continue { .. } => panic!("expected preflight"),
    }
}

#[test]
fn options_preflight_matching_origin_includes_cors_headers() {
    let cfg = config(&["https://app.example.com"]);
    let h = headers_with_origin("https://app.example.com");
    match apply_cors(&Method::OPTIONS, &h, &cfg) {
        CorsOutcome::Preflight(resp) => {
            let headers = resp.headers();
            assert!(headers.contains_key(ACCESS_CONTROL_ALLOW_ORIGIN));
            assert!(headers.contains_key(ACCESS_CONTROL_ALLOW_METHODS));
            assert!(headers.contains_key(ACCESS_CONTROL_ALLOW_HEADERS));
        }
        CorsOutcome::Continue { .. } => panic!("expected preflight"),
    }
}

#[test]
fn options_preflight_non_matching_origin_no_cors_headers() {
    let cfg = config(&["https://allowed.com"]);
    let h = headers_with_origin("https://evil.com");
    match apply_cors(&Method::OPTIONS, &h, &cfg) {
        CorsOutcome::Preflight(resp) => {
            assert!(!resp.headers().contains_key(ACCESS_CONTROL_ALLOW_ORIGIN));
        }
        CorsOutcome::Continue { .. } => panic!("expected preflight"),
    }
}

#[test]
fn options_no_cors_config_returns_preflight_204_no_cors_headers() {
    let cfg = config(&[]); // CORS disabled
    let h = headers_with_origin("https://any.com");
    match apply_cors(&Method::OPTIONS, &h, &cfg) {
        CorsOutcome::Preflight(resp) => {
            assert_eq!(resp.status(), axum::http::StatusCode::NO_CONTENT);
            assert!(!resp.headers().contains_key(ACCESS_CONTROL_ALLOW_ORIGIN));
        }
        CorsOutcome::Continue { .. } => panic!("expected preflight"),
    }
}
