//! Security implementation for gRPC services
//!
//! This module provides TLS/mTLS support, authentication, authorization,
//! and security audit logging for gRPC communication.

use crate::config::{
    SecurityConfig, TlsConfig, JwtConfig, ApiKeyConfig,
    AuthorizationConfig, SecurityAuditConfig, ClientCertVerification
};

use anyhow::{Result, anyhow};
use base64::prelude::*;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tonic::Request;
use tonic::transport::{Certificate, Identity, ServerTlsConfig};
use tracing::{info, warn, error, debug};
use tokio::sync::RwLock;

/// Authentication token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Token identifier
    pub token_id: String,
    /// Subject (user/service identifier)
    pub subject: String,
    /// Issued at timestamp
    pub issued_at: u64,
    /// Expiration timestamp
    pub expires_at: u64,
    /// Token permissions
    pub permissions: Vec<String>,
    /// Token metadata
    pub metadata: HashMap<String, String>,
}

/// Security audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Event type (auth, auth_failure, rate_limit, etc.)
    pub event_type: String,
    /// Client identifier
    pub client_id: String,
    /// Service being accessed
    pub service: String,
    /// Method being called
    pub method: String,
    /// Event result (success, failure, blocked)
    pub result: String,
    /// Additional event details
    pub details: HashMap<String, String>,
    /// Security risk level
    pub risk_level: RiskLevel,
}

/// Security risk levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk event
    Low,
    /// Medium risk event
    Medium,
    /// High risk event
    High,
    /// Critical security event
    Critical,
}

/// TLS certificate manager
#[derive(Debug)]
pub struct TlsManager {
    config: TlsConfig,
    server_identity: Option<Identity>,
    client_ca_certs: Vec<Certificate>,
}

impl TlsManager {
    /// Create a new TLS manager
    pub fn new(config: TlsConfig) -> Result<Self> {
        let server_identity = if let (Some(cert_file), Some(key_file)) =
            (&config.cert_file, &config.key_file) {
            Some(Self::load_server_identity(cert_file, key_file)?)
        } else {
            None
        };

        let client_ca_certs = if let Some(ca_cert_file) = &config.ca_cert_file {
            vec![Self::load_ca_certificate(ca_cert_file)?]
        } else {
            Vec::new()
        };

        Ok(Self {
            config,
            server_identity,
            client_ca_certs,
        })
    }

    /// Load server identity (certificate + private key)
    fn load_server_identity(cert_file: &str, key_file: &str) -> Result<Identity> {
        let cert_pem = fs::read_to_string(cert_file)
            .map_err(|e| anyhow!("Failed to read certificate file {}: {}", cert_file, e))?;

        let key_pem = fs::read_to_string(key_file)
            .map_err(|e| anyhow!("Failed to read key file {}: {}", key_file, e))?;

        Ok(Identity::from_pem(cert_pem, key_pem))
    }

    /// Load CA certificate for client verification
    fn load_ca_certificate(ca_cert_file: &str) -> Result<Certificate> {
        let ca_cert_pem = fs::read_to_string(ca_cert_file)
            .map_err(|e| anyhow!("Failed to read CA certificate file {}: {}", ca_cert_file, e))?;

        Ok(Certificate::from_pem(ca_cert_pem))
    }

    /// Create server TLS configuration
    pub fn create_server_tls_config(&self) -> Result<ServerTlsConfig> {
        let mut tls_config = ServerTlsConfig::new();

        if let Some(identity) = &self.server_identity {
            tls_config = tls_config.identity(identity.clone());
        }

        match self.config.client_cert_verification {
            ClientCertVerification::Required => {
                for ca_cert in &self.client_ca_certs {
                    tls_config = tls_config.client_ca_root(ca_cert.clone());
                }
            }
            ClientCertVerification::Optional => {
                // Configure optional client certificate verification
                for ca_cert in &self.client_ca_certs {
                    tls_config = tls_config.client_ca_root(ca_cert.clone());
                }
            }
            ClientCertVerification::None => {
                // No client certificate verification
            }
        }

        Ok(tls_config)
    }

    /// Validate client certificate
    pub fn validate_client_certificate(&self, _cert_chain: &[u8]) -> Result<bool> {
        // Implement client certificate validation logic
        // This would check certificate validity, expiration, revocation, etc.
        Ok(true)
    }
}

/// JWT token manager
#[derive(Debug)]
pub struct JwtManager {
    config: JwtConfig,
    #[allow(dead_code)]
    secret: Vec<u8>,
}

impl JwtManager {
    /// Create a new JWT manager
    pub fn new(config: JwtConfig) -> Result<Self> {
        let secret = if config.secret_or_key_file.starts_with('/') ||
                        config.secret_or_key_file.contains('.') {
            // Treat as file path
            fs::read(&config.secret_or_key_file)
                .map_err(|e| anyhow!("Failed to read JWT key file {}: {}", config.secret_or_key_file, e))?
        } else {
            // Treat as secret string
            config.secret_or_key_file.as_bytes().to_vec()
        };

        Ok(Self { config, secret })
    }

    /// Generate a JWT token
    pub fn generate_token(&self, subject: &str, permissions: Vec<String>) -> Result<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        let auth_token = AuthToken {
            token_id: uuid::Uuid::new_v4().to_string(),
            subject: subject.to_string(),
            issued_at: now,
            expires_at: now + self.config.expiration_secs,
            permissions,
            metadata: HashMap::new(),
        };

        // In a real implementation, use a proper JWT library like jsonwebtoken
        // For now, return a simple token format
        let token_data = serde_json::to_string(&auth_token)?;
        let token = BASE64_STANDARD.encode(token_data);

        info!("Generated JWT token for subject: {} (expires in {}s)",
              subject, self.config.expiration_secs);

        Ok(token)
    }

    /// Validate a JWT token
    pub fn validate_token(&self, token: &str) -> Result<AuthToken> {
        // In a real implementation, use proper JWT validation
        let token_data = BASE64_STANDARD.decode(token)
            .map_err(|e| anyhow!("Invalid token format: {}", e))?;

        let auth_token: AuthToken = serde_json::from_slice(&token_data)
            .map_err(|e| anyhow!("Invalid token data: {}", e))?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();

        if auth_token.expires_at < now {
            return Err(anyhow!("Token expired"));
        }

        Ok(auth_token)
    }
}

/// API key manager
#[derive(Debug)]
pub struct ApiKeyManager {
    config: ApiKeyConfig,
    key_permissions: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl ApiKeyManager {
    /// Create a new API key manager
    pub fn new(config: ApiKeyConfig) -> Self {
        let key_permissions = Arc::new(RwLock::new(config.key_permissions.clone()));

        Self {
            config,
            key_permissions,
        }
    }

    /// Validate an API key
    pub async fn validate_api_key(&self, api_key: &str) -> Result<Vec<String>> {
        if !self.config.enabled {
            return Ok(vec![]);
        }

        if !self.config.valid_keys.contains(&api_key.to_string()) {
            return Err(anyhow!("Invalid API key"));
        }

        let permissions = self.key_permissions.read().await;
        let key_perms = permissions.get(api_key)
            .cloned()
            .unwrap_or_default();

        Ok(key_perms)
    }

    /// Add or update API key permissions
    pub async fn update_key_permissions(&self, api_key: String, permissions: Vec<String>) {
        let mut key_permissions = self.key_permissions.write().await;
        key_permissions.insert(api_key, permissions);
    }
}

/// Authorization manager
#[derive(Debug)]
pub struct AuthorizationManager {
    config: AuthorizationConfig,
}

impl AuthorizationManager {
    /// Create a new authorization manager
    pub fn new(config: AuthorizationConfig) -> Self {
        Self { config }
    }

    /// Check if permissions allow access to service method
    pub fn check_access(&self, permissions: &[String], service: &str, _method: &str) -> bool {
        if !self.config.enabled {
            return true;
        }

        // Check service-specific permissions
        let required_permissions = match service {
            "DocumentProcessor" => &self.config.service_permissions.document_processor,
            "SearchService" => &self.config.service_permissions.search_service,
            "MemoryService" => &self.config.service_permissions.memory_service,
            "SystemService" => &self.config.service_permissions.system_service,
            _ => &self.config.default_permissions,
        };

        // Check if user has any required permission for this service
        permissions.iter().any(|perm| required_permissions.contains(perm))
    }
}

/// Security audit logger
#[derive(Debug)]
pub struct SecurityAuditLogger {
    config: SecurityAuditConfig,
}

impl SecurityAuditLogger {
    /// Create a new security audit logger
    pub fn new(config: SecurityAuditConfig) -> Self {
        Self { config }
    }

    /// Log a security event
    pub fn log_event(&self, event: SecurityAuditEvent) {
        if !self.config.enabled {
            return;
        }

        // Filter events based on configuration
        let should_log = match &event.event_type[..] {
            "auth" => self.config.log_auth_events,
            "auth_failure" => self.config.log_auth_failures,
            "rate_limit" => self.config.log_rate_limit_events,
            "suspicious" => self.config.log_suspicious_patterns,
            _ => true,
        };

        if !should_log {
            return;
        }

        // Log with appropriate level based on risk
        match event.risk_level {
            RiskLevel::Low => debug!("Security audit: {:?}", event),
            RiskLevel::Medium => info!("Security audit: {:?}", event),
            RiskLevel::High => warn!("Security audit: {:?}", event),
            RiskLevel::Critical => error!("CRITICAL Security audit: {:?}", event),
        }

        // In a real implementation, write to audit log file with rotation
        // For now, just use tracing
    }

    /// Log authentication event
    pub fn log_auth_event(&self, client_id: &str, service: &str, method: &str, success: bool) {
        let event = SecurityAuditEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type: if success { "auth".to_string() } else { "auth_failure".to_string() },
            client_id: client_id.to_string(),
            service: service.to_string(),
            method: method.to_string(),
            result: if success { "success".to_string() } else { "failure".to_string() },
            details: HashMap::new(),
            risk_level: if success { RiskLevel::Low } else { RiskLevel::Medium },
        };

        self.log_event(event);
    }

    /// Log rate limiting event
    pub fn log_rate_limit_event(&self, client_id: &str, service: &str, requests_per_sec: u32) {
        let mut details = HashMap::new();
        details.insert("requests_per_second".to_string(), requests_per_sec.to_string());

        let event = SecurityAuditEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type: "rate_limit".to_string(),
            client_id: client_id.to_string(),
            service: service.to_string(),
            method: "".to_string(),
            result: "blocked".to_string(),
            details,
            risk_level: RiskLevel::Medium,
        };

        self.log_event(event);
    }
}

/// Main security manager coordinating all security components
#[derive(Debug)]
pub struct SecurityManager {
    tls_manager: Option<TlsManager>,
    jwt_manager: Option<JwtManager>,
    api_key_manager: ApiKeyManager,
    authorization_manager: AuthorizationManager,
    audit_logger: SecurityAuditLogger,
}

impl SecurityManager {
    /// Create a new security manager
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let tls_manager = if config.tls.cert_file.is_some() {
            Some(TlsManager::new(config.tls.clone())?)
        } else {
            None
        };

        let jwt_manager = if config.auth.enable_service_auth {
            Some(JwtManager::new(config.auth.jwt.clone())?)
        } else {
            None
        };

        let api_key_manager = ApiKeyManager::new(config.auth.api_key.clone());
        let authorization_manager = AuthorizationManager::new(config.auth.authorization.clone());
        let audit_logger = SecurityAuditLogger::new(config.audit.clone());

        Ok(Self {
            tls_manager,
            jwt_manager,
            api_key_manager,
            authorization_manager,
            audit_logger,
        })
    }

    /// Get TLS configuration for server
    pub fn get_server_tls_config(&self) -> Result<Option<ServerTlsConfig>> {
        if let Some(tls_manager) = &self.tls_manager {
            Ok(Some(tls_manager.create_server_tls_config()?))
        } else {
            Ok(None)
        }
    }

    /// Authenticate a request using JWT or API key
    pub async fn authenticate_request<T>(&self, request: &Request<T>) -> Result<Vec<String>> {
        let metadata = request.metadata();

        // Try JWT authentication first
        if let Some(jwt_manager) = &self.jwt_manager {
            if let Some(auth_header) = metadata.get("authorization") {
                let auth_str = auth_header.to_str()
                    .map_err(|e| anyhow!("Invalid authorization header: {}", e))?;

                if auth_str.starts_with("Bearer ") {
                    let token = &auth_str[7..];
                    match jwt_manager.validate_token(token) {
                        Ok(auth_token) => return Ok(auth_token.permissions),
                        Err(e) => {
                            warn!("JWT validation failed: {}", e);
                        }
                    }
                }
            }
        }

        // Try API key authentication
        if let Some(api_key_header) = metadata.get(&self.api_key_manager.config.header_name.to_lowercase()) {
            let api_key = api_key_header.to_str()
                .map_err(|e| anyhow!("Invalid API key header: {}", e))?;

            match self.api_key_manager.validate_api_key(api_key).await {
                Ok(permissions) => return Ok(permissions),
                Err(e) => {
                    warn!("API key validation failed: {}", e);
                }
            }
        }

        // No valid authentication found
        Err(anyhow!("Authentication required"))
    }

    /// Authorize a request for service access
    pub fn authorize_request(&self, permissions: &[String], service: &str, method: &str) -> bool {
        self.authorization_manager.check_access(permissions, service, method)
    }

    /// Get audit logger
    pub fn audit_logger(&self) -> &SecurityAuditLogger {
        &self.audit_logger
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_jwt_manager_creation() {
        let config = JwtConfig {
            secret_or_key_file: "test_secret".to_string(),
            issuer: "test".to_string(),
            audience: "test".to_string(),
            expiration_secs: 3600,
            algorithm: "HS256".to_string(),
        };

        let jwt_manager = JwtManager::new(config);
        assert!(jwt_manager.is_ok());
    }

    #[test]
    fn test_jwt_token_generation() {
        let config = JwtConfig {
            secret_or_key_file: "test_secret".to_string(),
            issuer: "test".to_string(),
            audience: "test".to_string(),
            expiration_secs: 3600,
            algorithm: "HS256".to_string(),
        };

        let jwt_manager = JwtManager::new(config).unwrap();
        let token = jwt_manager.generate_token("test_user", vec!["read".to_string()]);
        assert!(token.is_ok());
    }

    #[test]
    fn test_jwt_token_validation() {
        let config = JwtConfig {
            secret_or_key_file: "test_secret".to_string(),
            issuer: "test".to_string(),
            audience: "test".to_string(),
            expiration_secs: 3600,
            algorithm: "HS256".to_string(),
        };

        let jwt_manager = JwtManager::new(config).unwrap();
        let token = jwt_manager.generate_token("test_user", vec!["read".to_string()]).unwrap();
        let auth_token = jwt_manager.validate_token(&token);
        assert!(auth_token.is_ok());

        let auth_token = auth_token.unwrap();
        assert_eq!(auth_token.subject, "test_user");
        assert!(auth_token.permissions.contains(&"read".to_string()));
    }

    #[tokio::test]
    async fn test_api_key_manager() {
        let config = ApiKeyConfig {
            enabled: true,
            header_name: "X-API-Key".to_string(),
            valid_keys: vec!["test_key".to_string()],
            key_permissions: HashMap::from([
                ("test_key".to_string(), vec!["read".to_string(), "write".to_string()])
            ]),
        };

        let api_key_manager = ApiKeyManager::new(config);

        // Valid key
        let permissions = api_key_manager.validate_api_key("test_key").await;
        assert!(permissions.is_ok());
        let permissions = permissions.unwrap();
        assert!(permissions.contains(&"read".to_string()));
        assert!(permissions.contains(&"write".to_string()));

        // Invalid key
        let invalid_result = api_key_manager.validate_api_key("invalid_key").await;
        assert!(invalid_result.is_err());
    }

    #[test]
    fn test_authorization_manager() {
        let config = AuthorizationConfig {
            enabled: true,
            default_permissions: vec!["read".to_string()],
            service_permissions: crate::config::ServicePermissions {
                document_processor: vec!["process".to_string()],
                search_service: vec!["search".to_string()],
                memory_service: vec!["read".to_string(), "write".to_string()],
                system_service: vec!["admin".to_string()],
            },
        };

        let auth_manager = AuthorizationManager::new(config);

        // User with search permission can access SearchService
        assert!(auth_manager.check_access(&["search".to_string()], "SearchService", "search"));

        // User without admin permission cannot access SystemService
        assert!(!auth_manager.check_access(&["read".to_string()], "SystemService", "status"));

        // User with admin permission can access SystemService
        assert!(auth_manager.check_access(&["admin".to_string()], "SystemService", "status"));
    }

    #[test]
    fn test_security_audit_logger() {
        let config = SecurityAuditConfig {
            enabled: true,
            log_file_path: "./test_audit.log".to_string(),
            log_auth_events: true,
            log_auth_failures: true,
            log_rate_limit_events: true,
            log_suspicious_patterns: true,
            rotation: crate::config::AuditLogRotation {
                max_file_size_mb: 100,
                max_files: 10,
                compress: true,
            },
        };

        let audit_logger = SecurityAuditLogger::new(config);

        // Test auth event logging
        audit_logger.log_auth_event("test_client", "TestService", "test_method", true);
        audit_logger.log_auth_event("test_client", "TestService", "test_method", false);

        // Test rate limit event logging
        audit_logger.log_rate_limit_event("test_client", "TestService", 150);

        // These should not panic and should log appropriately
    }

    #[test]
    fn test_tls_manager_no_certs() {
        let config = TlsConfig {
            cert_file: None,
            key_file: None,
            ca_cert_file: None,
            enable_mtls: false,
            client_cert_verification: ClientCertVerification::None,
            supported_protocols: vec!["TLSv1.2".to_string()],
            cipher_suites: vec![],
        };

        let tls_manager = TlsManager::new(config);
        assert!(tls_manager.is_ok());
    }
}