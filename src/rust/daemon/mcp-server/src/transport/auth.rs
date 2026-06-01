//! Bearer-token authentication for the HTTP transport.
//!
//! Mirrors `src/typescript/mcp-server/src/auth-middleware.ts` (auth logic only;
//! rate-limit and CORS are in their own modules).
//!
//! # Guarantees
//!
//! - Token comparison is constant-time (length-insensitive, matching
//!   `timingSafeEqual` semantics in auth-middleware.ts:227).
//! - The raw token is wrapped in [`secrecy::SecretString`] and never logged.
//! - Only an 8-hex-char SHA-256 digest is logged for audit/rotation (matching
//!   auth-middleware.ts:235).
//! - Startup refuses with a specific error message when the token is absent or
//!   too short (matching auth-middleware.ts:100-107).

use secrecy::{ExposeSecret, SecretString};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

/// Hard minimum token length. Rejects trivially-guessable tokens at startup.
///
/// Mirrors `MIN_TOKEN_LENGTH` in `auth-middleware.ts:39`.
pub const MIN_TOKEN_LENGTH: usize = 16;

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

/// Authentication configuration loaded from environment variables.
///
/// Mirrors `AuthConfig` in `auth-middleware.ts:41`.
#[derive(Clone)]
pub struct AuthConfig {
    /// The bearer secret wrapped in `SecretString` so it is never `Debug`-printed.
    /// `None` means auth is disabled (HTTP mode always requires a token).
    pub token: Option<SecretString>,
}

impl AuthConfig {
    /// Build from a raw token string (typically from `std::env::var`).
    pub fn new(raw: Option<String>) -> Self {
        Self {
            token: raw.map(|s| SecretString::new(s.into_boxed_str())),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Startup validation
// ─────────────────────────────────────────────────────────────────────────────

/// Enforce that HTTP mode has a usable bearer token.
///
/// Must be called on the startup path before binding the listener.
/// Logs a redacted digest so operators can confirm rotation without the
/// secret leaving the process.
///
/// # Errors
///
/// - Token absent / empty → exact TS message (auth-middleware.ts:101-103):
///   `"MCP_HTTP_TOKEN is required when MCP_SERVER_MODE=http. Generate one with: openssl rand -hex 32"`
/// - Token too short → exact TS message (auth-middleware.ts:106-108):
///   `"MCP_HTTP_TOKEN must be at least 16 characters (got <n>)."`
pub fn require_auth(config: &AuthConfig) -> Result<(), String> {
    match &config.token {
        None => Err("MCP_HTTP_TOKEN is required when MCP_SERVER_MODE=http. \
             Generate one with: openssl rand -hex 32"
            .to_string()),
        Some(t) => {
            let len = t.expose_secret().len();
            if len < MIN_TOKEN_LENGTH {
                return Err(format!(
                    "MCP_HTTP_TOKEN must be at least {MIN_TOKEN_LENGTH} characters (got {len})."
                ));
            }
            tracing::info!(
                token_digest = %token_digest(t.expose_secret()),
                "HTTP auth enabled"
            );
            Ok(())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-request bearer check
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single bearer-token check.
#[derive(Debug, PartialEq, Eq)]
pub enum BearerOutcome {
    /// Token accepted.
    Authorized,
    /// `Authorization` header missing or malformed.
    MissingHeader,
    /// Header present but token does not match.
    InvalidToken,
    /// Header present but server has no token configured.
    ///
    /// Mirrors the `not_configured` branch in `auth-middleware.ts:163-166`.
    NotConfigured,
}

/// Check the `Authorization: Bearer <token>` header against the configured secret.
///
/// Returns [`BearerOutcome`]. The caller decides how to respond (the axum
/// middleware layer writes the HTTP response).
pub fn check_bearer(header: Option<&str>, config: &AuthConfig) -> BearerOutcome {
    let presented = match extract_bearer(header) {
        Some(t) => t,
        None => return BearerOutcome::MissingHeader,
    };
    match &config.token {
        None => BearerOutcome::NotConfigured,
        Some(secret) => {
            if constant_time_equals(presented.as_bytes(), secret.expose_secret().as_bytes()) {
                BearerOutcome::Authorized
            } else {
                BearerOutcome::InvalidToken
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse `Authorization: Bearer <token>`. Returns the token string or `None`.
///
/// Mirrors `extractBearer()` in `auth-middleware.ts:211`.
pub fn extract_bearer(header: Option<&str>) -> Option<String> {
    let h = header?;
    let rest = h.trim().strip_prefix("Bearer")?;
    // Must have at least one whitespace character after "Bearer".
    if rest.is_empty() || !rest.starts_with(|c: char| c.is_ascii_whitespace()) {
        return None;
    }
    let token = rest.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

/// Length-insensitive constant-time string compare.
///
/// When the two byte slices have different lengths we still perform a dummy
/// comparison to keep the observable work independent of the length
/// relationship, then return `false`.  This matches the TypeScript
/// `constantTimeEquals` in `auth-middleware.ts:220-231`.
pub fn constant_time_equals(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        // Dummy compare to avoid short-circuiting on length.
        let dummy = vec![0u8; a.len()];
        let _ = a.ct_eq(&dummy);
        return false;
    }
    a.ct_eq(b).into()
}

/// First 8 hex chars of SHA-256(token) — safe to log for audit/rotation.
///
/// Mirrors `tokenDigest()` in `auth-middleware.ts:234`.
pub fn token_digest(token: &str) -> String {
    let hash = Sha256::digest(token.as_bytes());
    let hex = hex::encode(hash);
    hex[..8].to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "auth_tests.rs"]
mod tests;
