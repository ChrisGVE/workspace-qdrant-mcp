//! Tests for bearer-token authentication (transport/auth.rs).

use super::*;

// ─────────────────────────────────────────────────────────────────────────────
// extract_bearer
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn extract_bearer_valid() {
    assert_eq!(
        extract_bearer(Some("Bearer mysecrettoken")),
        Some("mysecrettoken".to_string())
    );
}

#[test]
fn extract_bearer_extra_spaces() {
    assert_eq!(
        extract_bearer(Some("Bearer   spaced")),
        Some("spaced".to_string())
    );
}

#[test]
fn extract_bearer_missing() {
    assert_eq!(extract_bearer(None), None);
}

#[test]
fn extract_bearer_no_space_after_bearer() {
    // "Bearer" with no space must return None.
    assert_eq!(extract_bearer(Some("Bearertoken")), None);
}

#[test]
fn extract_bearer_empty_token() {
    assert_eq!(extract_bearer(Some("Bearer   ")), None);
}

#[test]
fn extract_bearer_basic_scheme() {
    // Non-Bearer scheme returns None.
    assert_eq!(extract_bearer(Some("Basic dXNlcjpwYXNz")), None);
}

// ─────────────────────────────────────────────────────────────────────────────
// constant_time_equals
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn constant_time_eq_identical() {
    assert!(constant_time_equals(b"hello", b"hello"));
}

#[test]
fn constant_time_eq_different_content() {
    assert!(!constant_time_equals(b"hello", b"world"));
}

#[test]
fn constant_time_eq_different_length() {
    // Different lengths must return false and not panic.
    assert!(!constant_time_equals(b"short", b"muchlonger"));
}

#[test]
fn constant_time_eq_empty() {
    assert!(constant_time_equals(b"", b""));
}

// ─────────────────────────────────────────────────────────────────────────────
// token_digest
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn token_digest_is_8_hex_chars() {
    let d = token_digest("my-super-secret-token-32chars!!!!");
    assert_eq!(d.len(), 8);
    assert!(d.chars().all(|c| c.is_ascii_hexdigit()));
}

#[test]
fn token_digest_is_deterministic() {
    let t = "deterministic-token-value";
    assert_eq!(token_digest(t), token_digest(t));
}

#[test]
fn token_digest_different_tokens_differ() {
    assert_ne!(
        token_digest("token_a_32chars_padding_______!!"),
        token_digest("token_b_32chars_padding_______!!")
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// require_auth
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn require_auth_absent_token_returns_exact_message() {
    let cfg = AuthConfig::new(None);
    let err = require_auth(&cfg).unwrap_err();
    assert_eq!(
        err,
        "MCP_HTTP_TOKEN is required when MCP_SERVER_MODE=http. \
         Generate one with: openssl rand -hex 32"
    );
}

#[test]
fn require_auth_too_short_returns_exact_message() {
    let cfg = AuthConfig::new(Some("short".to_string())); // 5 chars
    let err = require_auth(&cfg).unwrap_err();
    assert_eq!(
        err,
        "MCP_HTTP_TOKEN must be at least 16 characters (got 5)."
    );
}

#[test]
fn require_auth_exactly_16_chars_ok() {
    let cfg = AuthConfig::new(Some("1234567890123456".to_string()));
    assert!(require_auth(&cfg).is_ok());
}

#[test]
fn require_auth_more_than_16_chars_ok() {
    let cfg = AuthConfig::new(Some("a".repeat(32)));
    assert!(require_auth(&cfg).is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// check_bearer
// ─────────────────────────────────────────────────────────────────────────────

fn make_config(token: &str) -> AuthConfig {
    AuthConfig::new(Some(token.to_string()))
}

#[test]
fn check_bearer_valid_token() {
    let cfg = make_config("my-valid-token-1234");
    let outcome = check_bearer(Some("Bearer my-valid-token-1234"), &cfg);
    assert_eq!(outcome, BearerOutcome::Authorized);
}

#[test]
fn check_bearer_wrong_token() {
    let cfg = make_config("correct-token-1234567");
    let outcome = check_bearer(Some("Bearer wrong-token-abc"), &cfg);
    assert_eq!(outcome, BearerOutcome::InvalidToken);
}

#[test]
fn check_bearer_missing_header() {
    let cfg = make_config("some-token-12345678");
    let outcome = check_bearer(None, &cfg);
    assert_eq!(outcome, BearerOutcome::MissingHeader);
}

#[test]
fn check_bearer_malformed_header() {
    let cfg = make_config("some-token-12345678");
    let outcome = check_bearer(Some("NotBearer token"), &cfg);
    assert_eq!(outcome, BearerOutcome::MissingHeader);
}

#[test]
fn check_bearer_no_configured_token() {
    // AuthConfig with None token always returns InvalidToken if header is present.
    let cfg = AuthConfig::new(None);
    let outcome = check_bearer(Some("Bearer some-token-12345678"), &cfg);
    assert_eq!(outcome, BearerOutcome::InvalidToken);
}

#[test]
fn check_bearer_constant_time_path_exercised() {
    // Present a token of a different length than the configured one.
    // Must return InvalidToken without panicking (dummy compare).
    let cfg = make_config("short-token-1234"); // 16 chars
    let outcome = check_bearer(Some("Bearer this-is-a-much-longer-presented-token"), &cfg);
    assert_eq!(outcome, BearerOutcome::InvalidToken);
}
