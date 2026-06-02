//! Shared config validators.
//!
//! Small, component-agnostic predicates that components call from their own
//! `validate()` methods. Each returns `Ok(())` or a human-readable error
//! string (components map it into their error type / [`super::ConfigError`]).

/// Validate a server URL: non-empty and carrying an `http://` or `https://`
/// scheme. Path/query are not inspected.
pub fn validate_url(url: &str) -> Result<(), String> {
    if url.trim().is_empty() {
        return Err("url must not be empty".to_string());
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!("url must start with http:// or https://: {url}"));
    }
    Ok(())
}

/// Validate a TCP port: must be non-zero (port 0 is not a usable listen port).
pub fn validate_port(port: u16) -> Result<(), String> {
    if port == 0 {
        return Err("port must be non-zero".to_string());
    }
    Ok(())
}

/// Validate a timeout in milliseconds: must be greater than zero.
pub fn validate_timeout(timeout_ms: u64) -> Result<(), String> {
    if timeout_ms == 0 {
        return Err("timeout must be greater than zero".to_string());
    }
    Ok(())
}

/// Validate a filesystem path string: must be non-empty after trimming.
pub fn validate_path(path: &str) -> Result<(), String> {
    if path.trim().is_empty() {
        return Err("path must not be empty".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_accepts_http_and_https() {
        assert!(validate_url("http://localhost:6333").is_ok());
        assert!(validate_url("https://cloud.qdrant.io").is_ok());
    }

    #[test]
    fn url_rejects_empty_and_schemeless() {
        assert!(validate_url("").is_err());
        assert!(validate_url("   ").is_err());
        assert!(validate_url("localhost:6333").is_err());
        assert!(validate_url("ftp://x").is_err());
    }

    #[test]
    fn port_rejects_zero() {
        assert!(validate_port(0).is_err());
        assert!(validate_port(50051).is_ok());
    }

    #[test]
    fn timeout_rejects_zero() {
        assert!(validate_timeout(0).is_err());
        assert!(validate_timeout(1).is_ok());
    }

    #[test]
    fn path_rejects_empty() {
        assert!(validate_path("").is_err());
        assert!(validate_path("  ").is_err());
        assert!(validate_path("/tmp/state.db").is_ok());
    }
}
