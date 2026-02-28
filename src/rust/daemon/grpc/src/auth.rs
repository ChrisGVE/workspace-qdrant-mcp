//! Authentication and TLS configuration for the gRPC server.
//!
//! Contains the `AuthInterceptor` for validating API keys and origins,
//! along with the `AuthConfig` and `TlsConfig` configuration types.

use tonic::{Request, Status};

/// TLS configuration for secure connections
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_cert_path: Option<String>,
    pub require_client_cert: bool,
}

/// Authentication configuration
#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub jwt_secret: Option<String>,
    pub allowed_origins: Vec<String>,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            jwt_secret: None,
            allowed_origins: vec!["*".to_string()],
        }
    }
}

/// Authentication interceptor for validating gRPC requests.
///
/// Checks API key in the `authorization` header and validates
/// the `origin` header against the allowed origins list.
#[derive(Clone)]
pub struct AuthInterceptor {
    config: Option<AuthConfig>,
}

impl AuthInterceptor {
    pub fn new(config: Option<AuthConfig>) -> Self {
        Self { config }
    }

    pub fn check(&self, req: &Request<()>) -> Result<(), Status> {
        let Some(auth_config) = &self.config else {
            return Ok(()); // No auth configured
        };

        if !auth_config.enabled {
            return Ok(());
        }

        self.check_api_key(req, auth_config)?;
        self.check_origin(req, auth_config)?;

        Ok(())
    }

    fn check_api_key(&self, req: &Request<()>, auth_config: &AuthConfig) -> Result<(), Status> {
        let Some(expected_key) = &auth_config.api_key else {
            return Ok(());
        };

        let auth_header = req
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| Status::unauthenticated("Missing authorization header"))?;

        if !auth_header.starts_with("Bearer ") {
            return Err(Status::unauthenticated("Invalid authorization format"));
        }

        let token = &auth_header[7..]; // Remove "Bearer " prefix
        if token != expected_key {
            return Err(Status::unauthenticated("Invalid API key"));
        }

        Ok(())
    }

    fn check_origin(&self, req: &Request<()>, auth_config: &AuthConfig) -> Result<(), Status> {
        if auth_config.allowed_origins.contains(&"*".to_string()) {
            return Ok(());
        }

        let origin = req
            .metadata()
            .get("origin")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if !auth_config.allowed_origins.contains(&origin.to_string()) {
            return Err(Status::permission_denied("Origin not allowed"));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_interceptor_no_config() {
        let interceptor = AuthInterceptor::new(None);
        let req = Request::new(());
        assert!(interceptor.check(&req).is_ok());
    }

    #[test]
    fn test_auth_interceptor_disabled() {
        let config = AuthConfig {
            enabled: false,
            ..Default::default()
        };
        let interceptor = AuthInterceptor::new(Some(config));
        let req = Request::new(());
        assert!(interceptor.check(&req).is_ok());
    }

    #[test]
    fn test_auth_config_default() {
        let config = AuthConfig::default();
        assert!(!config.enabled);
        assert!(config.api_key.is_none());
        assert!(config.jwt_secret.is_none());
        assert_eq!(config.allowed_origins, vec!["*".to_string()]);
    }
}
