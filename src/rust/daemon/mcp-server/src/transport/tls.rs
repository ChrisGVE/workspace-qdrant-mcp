//! TLS configuration for the HTTP transport.
//!
//! Reads `MCP_HTTP_TLS_CERT`, `MCP_HTTP_TLS_KEY`, and (optionally)
//! `MCP_HTTP_TLS_CA` from the environment to build a rustls server config.
//!
//! Cert + key must always be provided together — supplying only one is a
//! startup error rather than a silently-broken config.
//!
//! # Environment variables
//!
//! | Variable           | Required | Description                       |
//! |--------------------|----------|-----------------------------------|
//! | `MCP_HTTP_TLS_CERT`| Yes*     | Path to PEM certificate file      |
//! | `MCP_HTTP_TLS_KEY` | Yes*     | Path to PEM private-key file      |
//! | `MCP_HTTP_TLS_CA`  | No       | Path to PEM CA-bundle (mTLS)      |
//!
//! *Both must be provided or neither. One without the other is an error.

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// TLS configuration resolved from environment variables.
///
/// `None` means TLS is disabled (no cert/key env vars are set).
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to the PEM certificate file (env `MCP_HTTP_TLS_CERT`).
    pub cert_path: String,
    /// Path to the PEM private-key file (env `MCP_HTTP_TLS_KEY`).
    pub key_path: String,
    /// Optional path to a CA-bundle for mTLS (env `MCP_HTTP_TLS_CA`).
    pub ca_path: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Startup validation
// ─────────────────────────────────────────────────────────────────────────────

/// Resolve TLS configuration from environment variables.
///
/// Returns `Ok(None)` when TLS is disabled (neither `MCP_HTTP_TLS_CERT` nor
/// `MCP_HTTP_TLS_KEY` is set).
///
/// Returns `Ok(Some(TlsConfig))` when both are set.
///
/// Returns `Err` when:
/// - Only one of `MCP_HTTP_TLS_CERT` / `MCP_HTTP_TLS_KEY` is set.
/// - Either path is set but empty after trimming.
pub fn tls_config_from_env() -> Result<Option<TlsConfig>, String> {
    let cert = non_empty_env("MCP_HTTP_TLS_CERT");
    let key = non_empty_env("MCP_HTTP_TLS_KEY");

    match (cert, key) {
        (None, None) => Ok(None),
        (Some(cert_path), Some(key_path)) => {
            let ca_path = non_empty_env("MCP_HTTP_TLS_CA");
            Ok(Some(TlsConfig {
                cert_path,
                key_path,
                ca_path,
            }))
        }
        (Some(_), None) => Err("MCP_HTTP_TLS_CERT is set but MCP_HTTP_TLS_KEY is missing. \
             Both must be provided together."
            .to_string()),
        (None, Some(_)) => Err("MCP_HTTP_TLS_KEY is set but MCP_HTTP_TLS_CERT is missing. \
             Both must be provided together."
            .to_string()),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return `Some(value)` when `var` is set and non-empty; `None` otherwise.
fn non_empty_env(var: &str) -> Option<String> {
    std::env::var(var).ok().and_then(|v| {
        let trimmed = v.trim().to_string();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    fn clear_tls_env() {
        unsafe {
            std::env::remove_var("MCP_HTTP_TLS_CERT");
            std::env::remove_var("MCP_HTTP_TLS_KEY");
            std::env::remove_var("MCP_HTTP_TLS_CA");
        }
    }

    #[test]
    #[serial]
    fn both_absent_returns_none() {
        clear_tls_env();
        assert!(tls_config_from_env().unwrap().is_none());
    }

    #[test]
    #[serial]
    fn cert_without_key_errors() {
        clear_tls_env();
        unsafe { std::env::set_var("MCP_HTTP_TLS_CERT", "/path/to/cert.pem") };
        let err = tls_config_from_env().unwrap_err();
        assert!(err.contains("MCP_HTTP_TLS_KEY is missing"), "got: {err}");
        clear_tls_env();
    }

    #[test]
    #[serial]
    fn key_without_cert_errors() {
        clear_tls_env();
        unsafe { std::env::set_var("MCP_HTTP_TLS_KEY", "/path/to/key.pem") };
        let err = tls_config_from_env().unwrap_err();
        assert!(err.contains("MCP_HTTP_TLS_CERT is missing"), "got: {err}");
        clear_tls_env();
    }

    #[test]
    #[serial]
    fn both_set_returns_config() {
        clear_tls_env();
        unsafe {
            std::env::set_var("MCP_HTTP_TLS_CERT", "/path/to/cert.pem");
            std::env::set_var("MCP_HTTP_TLS_KEY", "/path/to/key.pem");
        }
        let config = tls_config_from_env().unwrap().unwrap();
        assert_eq!(config.cert_path, "/path/to/cert.pem");
        assert_eq!(config.key_path, "/path/to/key.pem");
        assert!(config.ca_path.is_none());
        clear_tls_env();
    }

    #[test]
    #[serial]
    fn ca_path_optional() {
        clear_tls_env();
        unsafe {
            std::env::set_var("MCP_HTTP_TLS_CERT", "/path/to/cert.pem");
            std::env::set_var("MCP_HTTP_TLS_KEY", "/path/to/key.pem");
            std::env::set_var("MCP_HTTP_TLS_CA", "/path/to/ca.pem");
        }
        let config = tls_config_from_env().unwrap().unwrap();
        assert_eq!(config.ca_path.as_deref(), Some("/path/to/ca.pem"));
        clear_tls_env();
    }

    #[test]
    #[serial]
    fn empty_cert_treated_as_absent() {
        clear_tls_env();
        unsafe {
            std::env::set_var("MCP_HTTP_TLS_CERT", "  ");
            std::env::set_var("MCP_HTTP_TLS_KEY", "/path/to/key.pem");
        }
        // cert is empty → treated as absent → key-without-cert error
        let err = tls_config_from_env().unwrap_err();
        assert!(err.contains("MCP_HTTP_TLS_CERT is missing"), "got: {err}");
        clear_tls_env();
    }
}
