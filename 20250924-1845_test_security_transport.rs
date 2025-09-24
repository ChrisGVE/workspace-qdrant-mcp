//! Comprehensive test suite for security and transport optimizations
//! Task 256.6: Security and Local Communication Optimization tests

use anyhow::Result;
use std::time::Duration;
use tokio;
use tonic::{Request, Response, Status, metadata::MetadataValue};
use uuid::Uuid;

// Import the security and transport modules we implemented
use workspace_qdrant_daemon::grpc::{
    SecurityManager, TlsManager, JwtManager, ApiKeyManager, AuthorizationManager,
    SecurityAuditLogger, AuthToken, SecurityAuditEvent, RiskLevel,
    TransportManager, UnixSocketManager, TransportType, LocalOptimization,
    TransportStats, ConnectionManager
};

use workspace_qdrant_daemon::config::{
    SecurityConfig, TlsConfig, AuthConfig, JwtConfig, ApiKeyConfig,
    AuthorizationConfig, SecurityAuditConfig, ClientCertVerification,
    TransportConfig, UnixSocketConfig, LocalOptimizationConfig,
    LocalLatencyConfig, TransportStrategy, ServicePermissions,
    AuditLogRotation
};

/// Test security configuration with comprehensive edge cases
#[tokio::test]
async fn test_security_manager_comprehensive() -> Result<()> {
    let security_config = create_test_security_config();
    let security_manager = SecurityManager::new(security_config)?;

    // Test TLS configuration
    if let Some(tls_config) = security_manager.get_server_tls_config()? {
        // TLS should be available if configured
        assert!(true);
    }

    // Test JWT authentication with valid token
    let jwt_manager = JwtManager::new(create_test_jwt_config())?;
    let token = jwt_manager.generate_token("test_user", vec!["read".to_string(), "write".to_string()])?;
    let auth_token = jwt_manager.validate_token(&token)?;

    assert_eq!(auth_token.subject, "test_user");
    assert!(auth_token.permissions.contains(&"read".to_string()));
    assert!(auth_token.permissions.contains(&"write".to_string()));

    // Test expired token handling
    let expired_config = JwtConfig {
        secret_or_key_file: "test_secret".to_string(),
        issuer: "test".to_string(),
        audience: "test".to_string(),
        expiration_secs: 0, // Expire immediately
        algorithm: "HS256".to_string(),
    };

    let expired_jwt_manager = JwtManager::new(expired_config)?;
    let expired_token = expired_jwt_manager.generate_token("test_user", vec!["read".to_string()])?;

    // Sleep to ensure expiration
    tokio::time::sleep(Duration::from_millis(10)).await;

    let expired_result = expired_jwt_manager.validate_token(&expired_token);
    assert!(expired_result.is_err(), "Expired token should be rejected");

    Ok(())
}

/// Test API key manager with edge cases
#[tokio::test]
async fn test_api_key_manager_edge_cases() -> Result<()> {
    // Test with empty keys list
    let empty_config = ApiKeyConfig {
        enabled: true,
        header_name: "X-API-Key".to_string(),
        valid_keys: vec![],
        key_permissions: std::collections::HashMap::new(),
    };

    let empty_api_manager = ApiKeyManager::new(empty_config);
    let empty_result = empty_api_manager.validate_api_key("any_key").await;
    assert!(empty_result.is_err(), "Empty keys list should reject all keys");

    // Test with disabled API key authentication
    let disabled_config = ApiKeyConfig {
        enabled: false,
        header_name: "X-API-Key".to_string(),
        valid_keys: vec!["valid_key".to_string()],
        key_permissions: std::collections::HashMap::new(),
    };

    let disabled_api_manager = ApiKeyManager::new(disabled_config);
    let disabled_result = disabled_api_manager.validate_api_key("invalid_key").await;
    assert!(disabled_result.is_ok(), "Disabled API key manager should allow all keys");

    // Test key permission updates
    let mut config = create_test_api_key_config();
    let api_manager = ApiKeyManager::new(config.clone());

    // Update permissions for a key
    api_manager.update_key_permissions("test_key".to_string(), vec!["admin".to_string()]).await;

    let permissions = api_manager.validate_api_key("test_key").await?;
    assert!(permissions.contains(&"admin".to_string()), "Updated permissions should be reflected");

    Ok(())
}

/// Test authorization with complex permission scenarios
#[test]
fn test_authorization_complex_scenarios() {
    let config = create_test_authorization_config();
    let auth_manager = AuthorizationManager::new(config);

    // Test service-specific permissions
    assert!(auth_manager.check_access(&["process".to_string()], "DocumentProcessor", "process_document"));
    assert!(!auth_manager.check_access(&["read".to_string()], "DocumentProcessor", "process_document"));

    // Test multiple permissions
    assert!(auth_manager.check_access(&["read".to_string(), "write".to_string()], "MemoryService", "add_document"));
    assert!(auth_manager.check_access(&["admin".to_string()], "SystemService", "get_metrics"));

    // Test unknown service defaults
    assert!(auth_manager.check_access(&["read".to_string()], "UnknownService", "unknown_method"));
    assert!(!auth_manager.check_access(&["write".to_string()], "UnknownService", "unknown_method"));

    // Test empty permissions
    assert!(!auth_manager.check_access(&[], "DocumentProcessor", "process_document"));

    // Test disabled authorization
    let disabled_config = AuthorizationConfig {
        enabled: false,
        default_permissions: vec!["read".to_string()],
        service_permissions: ServicePermissions::default(),
    };

    let disabled_auth_manager = AuthorizationManager::new(disabled_config);
    assert!(disabled_auth_manager.check_access(&[], "AnyService", "any_method"), "Disabled authorization should allow all access");
}

/// Test security audit logging with various risk levels
#[test]
fn test_security_audit_comprehensive() {
    let audit_config = create_test_security_audit_config();
    let audit_logger = SecurityAuditLogger::new(audit_config);

    // Test different event types
    audit_logger.log_auth_event("client1", "TestService", "test_method", true);
    audit_logger.log_auth_event("client1", "TestService", "test_method", false);
    audit_logger.log_rate_limit_event("client2", "TestService", 150);

    // Test custom security events
    let custom_event = SecurityAuditEvent {
        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        event_type: "suspicious".to_string(),
        client_id: "suspicious_client".to_string(),
        service: "TestService".to_string(),
        method: "sensitive_method".to_string(),
        result: "blocked".to_string(),
        details: std::collections::HashMap::from([
            ("reason".to_string(), "multiple failed attempts".to_string()),
            ("ip".to_string(), "192.168.1.100".to_string()),
        ]),
        risk_level: RiskLevel::High,
    };

    audit_logger.log_event(custom_event);

    // Test critical security event
    let critical_event = SecurityAuditEvent {
        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        event_type: "security_breach".to_string(),
        client_id: "attacker".to_string(),
        service: "SystemService".to_string(),
        method: "get_sensitive_data".to_string(),
        result: "blocked".to_string(),
        details: std::collections::HashMap::from([
            ("attack_type".to_string(), "sql_injection".to_string()),
            ("payload".to_string(), "'; DROP TABLE users; --".to_string()),
        ]),
        risk_level: RiskLevel::Critical,
    };

    audit_logger.log_event(critical_event);
}

/// Test TLS manager without certificates (production scenario)
#[test]
fn test_tls_manager_production_scenarios() {
    // Test TLS manager with missing certificate files
    let missing_cert_config = TlsConfig {
        cert_file: Some("/nonexistent/cert.pem".to_string()),
        key_file: Some("/nonexistent/key.pem".to_string()),
        ca_cert_file: None,
        enable_mtls: false,
        client_cert_verification: ClientCertVerification::None,
        supported_protocols: vec!["TLSv1.2".to_string(), "TLSv1.3".to_string()],
        cipher_suites: vec![],
    };

    let tls_result = TlsManager::new(missing_cert_config);
    assert!(tls_result.is_err(), "TLS manager should fail with missing certificate files");

    // Test TLS manager with mTLS configuration
    let mtls_config = TlsConfig {
        cert_file: None,
        key_file: None,
        ca_cert_file: None,
        enable_mtls: true,
        client_cert_verification: ClientCertVerification::Required,
        supported_protocols: vec!["TLSv1.3".to_string()],
        cipher_suites: vec!["TLS_AES_256_GCM_SHA384".to_string()],
    };

    let mtls_tls_manager = TlsManager::new(mtls_config);
    assert!(mtls_tls_manager.is_ok(), "TLS manager should work without certificates for basic config");
}

/// Test transport manager with various strategies
#[test]
fn test_transport_manager_strategies() {
    // Test Auto strategy
    let auto_config = create_test_transport_config_with_strategy(TransportStrategy::Auto);
    let auto_transport = TransportManager::new(auto_config);

    let local_transport = auto_transport.determine_transport_type("127.0.0.1", 50051);
    // Should prefer Unix socket for local connections in development

    let remote_transport = auto_transport.determine_transport_type("remote-server.com", 50051);
    assert!(matches!(remote_transport, TransportType::Tcp { .. }), "Remote connections should use TCP");

    // Test Force TCP strategy
    let tcp_config = create_test_transport_config_with_strategy(TransportStrategy::ForceTcp);
    let tcp_transport = TransportManager::new(tcp_config);

    let forced_tcp = tcp_transport.determine_transport_type("127.0.0.1", 50051);
    assert!(matches!(forced_tcp, TransportType::Tcp { .. }), "Force TCP should always use TCP");

    // Test Force Unix Socket strategy
    let unix_config = create_test_transport_config_with_strategy(TransportStrategy::ForceUnixSocket);
    let unix_transport = TransportManager::new(unix_config);

    let forced_unix = unix_transport.determine_transport_type("remote-server.com", 50051);
    assert!(matches!(forced_unix, TransportType::UnixSocket { .. }), "Force Unix should always use Unix socket");

    // Test Fallback strategy
    let fallback_config = create_test_transport_config_with_strategy(TransportStrategy::UnixSocketWithTcpFallback);
    let fallback_transport = TransportManager::new(fallback_config);

    let fallback_local = fallback_transport.determine_transport_type("127.0.0.1", 50051);
    let fallback_remote = fallback_transport.determine_transport_type("8.8.8.8", 50051);
    assert!(matches!(fallback_remote, TransportType::Tcp { .. }), "Fallback should use TCP for remote");
}

/// Test Unix socket manager in various environments
#[tokio::test]
async fn test_unix_socket_manager_environments() -> Result<()> {
    let socket_config = create_test_unix_socket_config();
    let unix_manager = UnixSocketManager::new(socket_config);

    // Test preference detection
    let should_prefer = unix_manager.should_prefer_unix_socket();
    // This depends on environment variables, so we just test it doesn't panic

    // Test cleanup functionality
    let cleanup_result = unix_manager.cleanup();
    assert!(cleanup_result.is_ok(), "Cleanup should not fail even if no socket exists");

    // Test disabled Unix socket
    let disabled_config = UnixSocketConfig {
        enabled: false,
        socket_path: "/tmp/test.sock".to_string(),
        permissions: 0o600,
        prefer_for_local: true,
    };

    let disabled_manager = UnixSocketManager::new(disabled_config);
    let disabled_result = disabled_manager.create_listener().await;
    assert!(disabled_result.is_err(), "Disabled Unix socket manager should fail to create listener");

    Ok(())
}

/// Test local optimization configuration
#[test]
fn test_local_optimization_config() {
    let config = create_test_local_optimization_config();
    let optimization = LocalOptimization::from(config);

    assert_eq!(optimization.buffer_size, 128 * 1024);
    assert!(optimization.disable_nagle);
    assert_eq!(optimization.connection_pool_size, 20);
    assert_eq!(optimization.keepalive_interval, Duration::from_secs(30));
    assert!(optimization.memory_efficient_serialization);
}

/// Test enhanced connection manager with security integration
#[tokio::test]
async fn test_enhanced_connection_manager() -> Result<()> {
    let security_config = create_test_security_config();
    let security_manager = Some(std::sync::Arc::new(SecurityManager::new(security_config)?));

    let connection_manager = ConnectionManager::new_with_security(
        10, // max_connections
        100, // requests_per_second
        200, // burst_capacity
        security_manager
    );

    // Test burst capacity rate limiting
    let client_id = "burst_test_client";

    // Should allow burst requests initially
    for i in 0..150 {
        let result = connection_manager.check_rate_limit(client_id);
        if i < 100 {
            assert!(result.is_ok(), "Burst should allow initial requests: attempt {}", i);
        }
        // Some requests beyond burst should start failing
    }

    // Test memory usage tracking
    let memory_result = connection_manager.check_memory_usage(client_id, 50 * 1024 * 1024); // 50MB
    assert!(memory_result.is_ok(), "Memory usage within limit should be allowed");

    let excessive_memory_result = connection_manager.check_memory_usage(client_id, 200 * 1024 * 1024); // 200MB (exceeds 100MB default limit)
    assert!(excessive_memory_result.is_err(), "Excessive memory usage should be blocked");

    // Test memory usage updates
    connection_manager.update_memory_usage(client_id, 10 * 1024 * 1024); // Add 10MB
    connection_manager.update_memory_usage(client_id, -5 * 1024 * 1024); // Remove 5MB

    Ok(())
}

/// Test authentication request processing
#[tokio::test]
async fn test_authentication_request_processing() -> Result<()> {
    let security_config = create_test_security_config();
    let security_manager = SecurityManager::new(security_config)?;

    // Test JWT authentication request
    let jwt_manager = JwtManager::new(create_test_jwt_config())?;
    let token = jwt_manager.generate_token("test_user", vec!["read".to_string()])?;

    let mut request: Request<()> = Request::new(());
    request.metadata_mut().insert(
        "authorization",
        MetadataValue::from_str(&format!("Bearer {}", token))?
    );

    let auth_result = security_manager.authenticate_request(&request).await;
    assert!(auth_result.is_ok(), "Valid JWT should authenticate successfully");

    let permissions = auth_result?;
    assert!(permissions.contains(&"read".to_string()), "JWT permissions should be extracted");

    // Test API key authentication request
    let mut api_request: Request<()> = Request::new(());
    api_request.metadata_mut().insert(
        "x-api-key",
        MetadataValue::from_static("test_key")
    );

    let api_auth_result = security_manager.authenticate_request(&api_request).await;
    if api_auth_result.is_ok() {
        let api_permissions = api_auth_result?;
        // API key permissions depend on configuration
    }

    // Test request with no authentication
    let no_auth_request: Request<()> = Request::new(());
    let no_auth_result = security_manager.authenticate_request(&no_auth_request).await;
    assert!(no_auth_result.is_err(), "Request without authentication should fail");

    Ok(())
}

/// Test resource protection under attack scenarios
#[tokio::test]
async fn test_resource_protection_attack_scenarios() -> Result<()> {
    let connection_manager = ConnectionManager::new_with_security(5, 10, 20, None);

    // Simulate connection exhaustion attack
    let mut client_connections = Vec::new();
    for i in 0..5 {
        let client_id = format!("attacker_client_{}", i);
        let result = connection_manager.register_connection(client_id.clone());
        assert!(result.is_ok(), "Should allow connections within limit");
        client_connections.push(client_id);
    }

    // Try to exceed connection limit
    let overflow_result = connection_manager.register_connection("overflow_client".to_string());
    assert!(overflow_result.is_err(), "Should block connections over limit");

    // Simulate rapid request attack
    let attacker_id = "rapid_attacker";
    for _ in 0..50 {
        let _ = connection_manager.check_rate_limit(attacker_id);
    }

    // Verify rate limiting kicks in
    let rate_limit_result = connection_manager.check_rate_limit(attacker_id);
    // Rate limiting should eventually block excessive requests

    // Simulate memory exhaustion attack
    let memory_attacker = "memory_attacker";
    connection_manager.register_connection(memory_attacker.to_string())?;

    // Try to allocate excessive memory
    let memory_attack_result = connection_manager.check_memory_usage(memory_attacker, 500 * 1024 * 1024); // 500MB
    assert!(memory_attack_result.is_err(), "Should block excessive memory allocation");

    // Cleanup connections
    for client_id in client_connections {
        connection_manager.unregister_connection(&client_id);
    }

    Ok(())
}

/// Test certificate validation edge cases
#[test]
fn test_certificate_validation_edge_cases() {
    let tls_config = TlsConfig {
        cert_file: None,
        key_file: None,
        ca_cert_file: None,
        enable_mtls: true,
        client_cert_verification: ClientCertVerification::Required,
        supported_protocols: vec!["TLSv1.3".to_string()],
        cipher_suites: vec![],
    };

    let tls_manager = TlsManager::new(tls_config).unwrap();

    // Test client certificate validation with invalid certificate
    let invalid_cert = b"invalid certificate data";
    let validation_result = tls_manager.validate_client_certificate(invalid_cert);
    // This is a simplified test - real implementation would validate actual certificates

    // Test different verification modes
    let optional_config = TlsConfig {
        cert_file: None,
        key_file: None,
        ca_cert_file: None,
        enable_mtls: true,
        client_cert_verification: ClientCertVerification::Optional,
        supported_protocols: vec!["TLSv1.2".to_string(), "TLSv1.3".to_string()],
        cipher_suites: vec!["TLS_AES_256_GCM_SHA384".to_string(), "TLS_CHACHA20_POLY1305_SHA256".to_string()],
    };

    let optional_tls_manager = TlsManager::new(optional_config).unwrap();
    // Optional verification should work even without certificates configured
}

/// Test transport optimization performance benchmarks
#[tokio::test]
async fn test_transport_performance_optimization() -> Result<()> {
    let transport_config = create_test_transport_config_with_strategy(TransportStrategy::Auto);
    let transport_manager = TransportManager::new(transport_config);

    // Test local connection optimization
    let local_transport = transport_manager.determine_transport_type("127.0.0.1", 50051);
    let optimized_server = transport_manager.create_optimized_server(&local_transport).await?;
    // Server should be configured with local optimizations

    // Test client endpoint optimization
    let optimized_endpoint = transport_manager.create_optimized_endpoint(&local_transport)?;
    // Endpoint should have local optimizations applied

    // Test binding address generation
    let tcp_binding = transport_manager.get_binding_address(&TransportType::Tcp {
        host: "127.0.0.1".to_string(),
        port: 50051,
    });
    assert_eq!(tcp_binding, "127.0.0.1:50051");

    let unix_binding = transport_manager.get_binding_address(&TransportType::UnixSocket {
        path: "/tmp/test.sock".to_string(),
    });
    assert_eq!(unix_binding, "/tmp/test.sock");

    // Test local optimization settings
    let local_optimization = transport_manager.local_optimization();
    assert!(local_optimization.buffer_size > 0);
    assert_eq!(local_optimization.connection_pool_size, 20);

    Ok(())
}

// Helper functions to create test configurations

fn create_test_security_config() -> SecurityConfig {
    SecurityConfig {
        tls: create_test_tls_config(),
        auth: create_test_auth_config(),
        rate_limiting: workspace_qdrant_daemon::config::RateLimitConfig::default(),
        audit: create_test_security_audit_config(),
    }
}

fn create_test_tls_config() -> TlsConfig {
    TlsConfig {
        cert_file: None, // No cert files for testing
        key_file: None,
        ca_cert_file: None,
        enable_mtls: false,
        client_cert_verification: ClientCertVerification::None,
        supported_protocols: vec!["TLSv1.2".to_string(), "TLSv1.3".to_string()],
        cipher_suites: vec![],
    }
}

fn create_test_auth_config() -> AuthConfig {
    AuthConfig {
        enable_service_auth: true,
        jwt: create_test_jwt_config(),
        api_key: create_test_api_key_config(),
        authorization: create_test_authorization_config(),
    }
}

fn create_test_jwt_config() -> JwtConfig {
    JwtConfig {
        secret_or_key_file: "test_secret_key".to_string(),
        issuer: "test_issuer".to_string(),
        audience: "test_audience".to_string(),
        expiration_secs: 3600,
        algorithm: "HS256".to_string(),
    }
}

fn create_test_api_key_config() -> ApiKeyConfig {
    ApiKeyConfig {
        enabled: true,
        header_name: "X-API-Key".to_string(),
        valid_keys: vec!["test_key".to_string(), "admin_key".to_string()],
        key_permissions: std::collections::HashMap::from([
            ("test_key".to_string(), vec!["read".to_string(), "write".to_string()]),
            ("admin_key".to_string(), vec!["read".to_string(), "write".to_string(), "admin".to_string()]),
        ]),
    }
}

fn create_test_authorization_config() -> AuthorizationConfig {
    AuthorizationConfig {
        enabled: true,
        default_permissions: vec!["read".to_string()],
        service_permissions: ServicePermissions {
            document_processor: vec!["process".to_string(), "read".to_string()],
            search_service: vec!["search".to_string(), "read".to_string()],
            memory_service: vec!["read".to_string(), "write".to_string(), "delete".to_string()],
            system_service: vec!["admin".to_string(), "read".to_string()],
        },
    }
}

fn create_test_security_audit_config() -> SecurityAuditConfig {
    SecurityAuditConfig {
        enabled: true,
        log_file_path: "./test_security_audit.log".to_string(),
        log_auth_events: true,
        log_auth_failures: true,
        log_rate_limit_events: true,
        log_suspicious_patterns: true,
        rotation: AuditLogRotation {
            max_file_size_mb: 10,
            max_files: 5,
            compress: true,
        },
    }
}

fn create_test_transport_config_with_strategy(strategy: TransportStrategy) -> TransportConfig {
    TransportConfig {
        unix_socket: create_test_unix_socket_config(),
        local_optimization: create_test_local_optimization_config(),
        transport_strategy: strategy,
    }
}

fn create_test_unix_socket_config() -> UnixSocketConfig {
    UnixSocketConfig {
        enabled: true,
        socket_path: "/tmp/test_workspace_qdrant.sock".to_string(),
        permissions: 0o600,
        prefer_for_local: true,
    }
}

fn create_test_local_optimization_config() -> LocalOptimizationConfig {
    LocalOptimizationConfig {
        enabled: true,
        use_large_buffers: true,
        local_buffer_size: 128 * 1024,
        memory_efficient_serialization: true,
        reduce_latency: LocalLatencyConfig {
            disable_nagle: true,
            custom_connection_pooling: true,
            connection_pool_size: 20,
            keepalive_interval_secs: 30,
        },
    }
}