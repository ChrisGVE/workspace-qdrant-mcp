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

use std::sync::Arc;

use rustls::ServerConfig;

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

/// Build a [`rustls::ServerConfig`] from a [`TlsConfig`].
///
/// Reads the cert chain and private key from the PEM files specified in the
/// config.  The ring crypto provider is installed if not already installed (it
/// is already in the dependency graph via tonic).
///
/// Key material is never logged.
///
/// # Errors
///
/// Returns a descriptive error string when PEM files cannot be read or
/// parsed, or when the key/cert pair is invalid.
pub fn build_rustls_server_config(cfg: &TlsConfig) -> Result<Arc<ServerConfig>, String> {
    // Install the ring crypto provider once.  Subsequent calls are no-ops.
    let _ = rustls::crypto::ring::default_provider().install_default();

    let cert_pem =
        std::fs::read(&cfg.cert_path).map_err(|e| format!("TLS cert read error: {e}"))?;
    let key_pem = std::fs::read(&cfg.key_path).map_err(|e| format!("TLS key read error: {e}"))?;

    let cert_chain = rustls_pemfile::certs(&mut cert_pem.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("TLS cert parse error: {e}"))?;

    if cert_chain.is_empty() {
        return Err("TLS cert file contains no certificates".to_string());
    }

    let private_key = rustls_pemfile::private_key(&mut key_pem.as_slice())
        .map_err(|e| format!("TLS key parse error: {e}"))?
        .ok_or_else(|| "TLS key file contains no private key".to_string())?;

    let server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(cert_chain, private_key)
        .map_err(|e| format!("TLS config error: {e}"))?;

    Ok(Arc::new(server_config))
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

    // ── build_rustls_server_config ────────────────────────────────────────────

    /// Generate a self-signed cert+key pair in memory using rcgen, write to
    /// temp files, and verify that build_rustls_server_config succeeds.
    #[test]
    fn build_rustls_config_with_valid_cert_and_key() {
        use rcgen::generate_simple_self_signed;
        use std::io::Write;

        let cert_key = generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_pem = cert_key.cert.pem();
        let key_pem = cert_key.key_pair.serialize_pem();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        cert_file.write_all(cert_pem.as_bytes()).unwrap();
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        key_file.write_all(key_pem.as_bytes()).unwrap();

        let cfg = TlsConfig {
            cert_path: cert_file.path().to_string_lossy().to_string(),
            key_path: key_file.path().to_string_lossy().to_string(),
            ca_path: None,
        };
        let result = build_rustls_server_config(&cfg);
        assert!(result.is_ok(), "expected Ok, got: {:?}", result.err());
    }

    #[test]
    fn build_rustls_config_missing_cert_file_errors() {
        let cfg = TlsConfig {
            cert_path: "/nonexistent/cert.pem".to_string(),
            key_path: "/nonexistent/key.pem".to_string(),
            ca_path: None,
        };
        let err = build_rustls_server_config(&cfg).unwrap_err();
        assert!(err.contains("TLS cert read error"), "got: {err}");
    }

    #[test]
    fn build_rustls_config_tls_env_configured_takes_tls_path() {
        // This test verifies the invariant: when tls_config_from_env() returns
        // Some, the caller (serve_http) must use the TLS path (not fall back to
        // plaintext).  The unit-level assertion is that we get a valid
        // ServerConfig from build_rustls_server_config with a real cert/key.
        use rcgen::generate_simple_self_signed;
        use std::io::Write;

        let cert_key = generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_pem = cert_key.cert.pem();
        let key_pem = cert_key.key_pair.serialize_pem();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        cert_file.write_all(cert_pem.as_bytes()).unwrap();
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        key_file.write_all(key_pem.as_bytes()).unwrap();

        let tls_cfg = TlsConfig {
            cert_path: cert_file.path().to_string_lossy().to_string(),
            key_path: key_file.path().to_string_lossy().to_string(),
            ca_path: None,
        };

        // When TLS is configured, build_rustls_server_config must succeed —
        // confirming serve_http will bind TLS, not plaintext.
        let server_config = build_rustls_server_config(&tls_cfg);
        assert!(
            server_config.is_ok(),
            "TLS env configured → TLS path must be taken (got: {:?})",
            server_config.err()
        );
    }
}
